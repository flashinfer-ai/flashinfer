"""Ulysses all-to-all benchmark harness (M3 performance contract).

Measures the Ulysses attention communication pattern — one *sample* is
``3 x scatter_heads + 1 x gather_heads`` (q/k/v in, output back) — across
implementations, with the methodology fixes over the original wan example
bench:

- workspace sized to the actual largest operand (``B*S_local*H*D`` elements),
  not W-times over-allocated
- per-sample timing (one CUDA event pair per iteration), not a single-window
  mean; every sample is reduced to the max across ranks (the collective is
  only done when the slowest rank is done), not rank 0's local time
- >= 5 repeats x 30 iters with the implementation order alternating between
  repeats to decorrelate thermal/clock drift from the impl identity
- p50 / p95 / std / mean over the rank-max samples, machine-readable JSON and
  CSV output

Implementations:

- ``raw``:           init_ulysses_a2a / ulysses_a2a (works on the c83e4204
                     baseline checkout and on current builds)
- ``communicator``:  UlyssesCommunicator public API (current builds only;
                     auto-skipped where unavailable)
- ``nccl``:          dist.all_to_all_single + permute glue (self-contained,
                     no dependency on flashinfer beyond torch)

Also reports a secondary end-to-end proxy ("e2e_attn"): the same pattern with
a scaled_dot_product_attention between the scatters and the gather.

Usage (run on the current checkout, then on a baseline worktree, then gate):

    python benchmarks/bench_ulysses_a2a.py run --world-size 8 \
        --impls raw,communicator,nccl --label new --out /tmp/ulysses_new_w8

    PYTHONPATH=/path/to/c83e4204-worktree \
    python benchmarks/bench_ulysses_a2a.py run --world-size 8 \
        --impls raw,nccl --label baseline --out /tmp/ulysses_base_w8

    python benchmarks/bench_ulysses_a2a.py compare \
        /tmp/ulysses_base_w8.json /tmp/ulysses_new_w8.json --threshold-pct 3
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import socket
import statistics
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

PRIMARY_UNIT = "a2a"  # regression gate applies to this unit only


def default_workload(world_size: int) -> dict:
    """Wan2.1-14B geometry (B=1, S_global=32760, 40 heads x 128, bf16) where
    the head count divides the world size; W=6 uses a divisible standalone
    workload of the same token count instead (40 % 6 != 0)."""
    heads = 40 if 40 % world_size == 0 else 48
    return dict(B=1, S_global=32760, H=heads, D=128, dtype="bfloat16")


# ---- implementations ----------------------------------------------------------


class NcclImpl:
    """all_to_all_single + permute glue; self-contained baseline."""

    name = "nccl"

    def __init__(self, group, ws, shapes, device, dtype):
        self.group = group
        self.ws = ws

    def scatter(self, x):
        B, S_local, H, D = x.shape
        W = self.ws
        H_local = H // W
        xt = x.reshape(B, S_local, W, H_local, D).permute(2, 0, 1, 3, 4).contiguous()
        recv = torch.empty_like(xt)
        dist.all_to_all_single(recv, xt, group=self.group)
        return recv.permute(1, 0, 2, 3, 4).reshape(B, W * S_local, H_local, D)

    def gather(self, y):
        B, S_global, H_local, D = y.shape
        W = self.ws
        S_local = S_global // W
        yt = y.reshape(B, W, S_local, H_local, D).permute(1, 0, 2, 3, 4).contiguous()
        recv = torch.empty_like(yt)
        dist.all_to_all_single(recv, yt, group=self.group)
        return (
            recv.permute(1, 2, 0, 3, 4).reshape(B, S_local, W * H_local, D).contiguous()
        )

    def close(self):
        pass


class RawImpl:
    """init_ulysses_a2a / ulysses_a2a; runs on the c83e4204 baseline too.

    Workspace: exactly the largest operand (B*S_local*H*D elements) — the
    original bench passed B*S_global*H*D, over-allocating by W times.
    """

    name = "raw"

    def __init__(self, group, ws, shapes, device, dtype):
        import flashinfer.comm as comm

        self.comm = comm
        self.group = group
        self.ws = ws
        self.B, self.S_local, self.H, self.D = shapes
        max_elems = self.B * self.S_local * self.H * self.D
        self.out_ptrs = comm.create_shared_buffer(
            max_elems * torch.tensor([], dtype=dtype).element_size(), group=group
        )
        self.sig_ptrs = comm.create_shared_buffer(comm.vllm_meta_size(), group=group)
        self.fa = comm.init_ulysses_a2a(
            self.out_ptrs, self.sig_ptrs, dist.get_rank(group), ws, True
        )
        dist.barrier(group=group)

    def scatter(self, x):
        B, S_local, H, D = x.shape
        out = torch.empty(
            B, S_local * self.ws, H // self.ws, D, dtype=x.dtype, device=x.device
        )
        self.comm.ulysses_a2a(self.fa, x, out, B, S_local, H, D, 0)
        return out

    def gather(self, y):
        B, S_global, H_local, D = y.shape
        S_local = S_global // self.ws
        H = H_local * self.ws
        out = torch.empty(B, S_local, H, D, dtype=y.dtype, device=y.device)
        self.comm.ulysses_a2a(self.fa, y, out, B, S_local, H, D, 1)
        return out

    def close(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        self.comm.dispose_ulysses_a2a(self.fa)
        self.comm.free_shared_buffer(self.out_ptrs, group=self.group)
        self.comm.free_shared_buffer(self.sig_ptrs, group=self.group)


class CommunicatorImpl:
    """UlyssesCommunicator public API (current builds only)."""

    name = "communicator"

    def __init__(self, group, ws, shapes, device, dtype):
        from flashinfer.comm import UlyssesCommunicator

        B, S_local, H, D = shapes
        self.comm = UlyssesCommunicator(
            group,
            max_elems=B * S_local * H * D,
            dtype=dtype,
            backend="nvlink",
            device=device,
        )

    def scatter(self, x):
        return self.comm.scatter_heads(x)

    def gather(self, y):
        return self.comm.gather_heads(y)

    def close(self):
        torch.cuda.synchronize()
        self.comm.close()


IMPLS = {c.name: c for c in (RawImpl, CommunicatorImpl, NcclImpl)}


def impl_available(name: str) -> bool:
    if name == "nccl":
        return True
    import flashinfer.comm as comm

    if name == "communicator":
        return hasattr(comm, "UlyssesCommunicator")
    return hasattr(comm, "init_ulysses_a2a")


# ---- references (independent of all impls) ------------------------------------


def _ref_scatter(x, ws, rank, group):
    H_local = x.shape[2] // ws
    gathered = [torch.empty_like(x) for _ in range(ws)]
    dist.all_gather(gathered, x.contiguous(), group=group)
    slabs = [g[:, :, rank * H_local : (rank + 1) * H_local, :] for g in gathered]
    return torch.cat(slabs, dim=1).contiguous()


def _ref_gather(y, ws, rank, group):
    S_local = y.shape[1] // ws
    gathered = [torch.empty_like(y) for _ in range(ws)]
    dist.all_gather(gathered, y.contiguous(), group=group)
    blocks = [g[:, rank * S_local : (rank + 1) * S_local, :, :] for g in gathered]
    return torch.cat(blocks, dim=2).contiguous()


# ---- measurement ---------------------------------------------------------------


def _sdpa(q, k, v):
    # [B, S, H_local, D] -> attention over the full sequence with local heads
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return o.transpose(1, 2).contiguous()


def _measure_unit(impl, unit, q, k, v, iters, group):
    """One repeat: `iters` samples of the unit, one event pair per sample."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    def run_once():
        a = impl.scatter(q)
        b = impl.scatter(k)
        c = impl.scatter(v)
        if unit == "e2e_attn":
            o = _sdpa(a, b, c)
        else:
            o = a
        return impl.gather(o)

    torch.cuda.synchronize()
    dist.barrier(group=group)
    for i in range(iters):
        starts[i].record()
        run_once()
        stops[i].record()
    torch.cuda.synchronize()
    return [starts[i].elapsed_time(stops[i]) for i in range(iters)]


def _stats(samples):
    s = sorted(samples)
    n = len(s)
    return dict(
        p50=s[n // 2],
        p95=s[min(n - 1, int(round(0.95 * (n - 1))))],
        mean=statistics.fmean(s),
        std=statistics.stdev(s) if n > 1 else 0.0,
        n=n,
    )


def _worker(rank, ws, port, args, q_out):
    result = None
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=ws,
        )
        group = dist.group.WORLD
        wl = default_workload(ws)
        if args.heads:
            wl["H"] = args.heads
        B, S_global, H, D = wl["B"], wl["S_global"], wl["H"], wl["D"]
        assert S_global % ws == 0 and H % ws == 0, (S_global, H, ws)
        S_local = S_global // ws
        dtype = getattr(torch, wl["dtype"])
        device = torch.device(f"cuda:{rank}")

        torch.manual_seed(1234 + rank)
        mk = lambda: torch.randn(  # noqa: E731
            B, S_local, H, D, dtype=dtype, device=device
        )
        q, k, v = mk(), mk(), mk()

        impl_names = [n for n in args.impls.split(",") if impl_available(n)]
        impls = {
            n: IMPLS[n](group, ws, (B, S_local, H, D), device, dtype)
            for n in impl_names
        }

        # correctness preflight: every impl against the independent reference
        for name, impl in impls.items():
            out = impl.scatter(q)
            ref = _ref_scatter(q, ws, rank, group)
            torch.cuda.synchronize()
            assert torch.equal(out, ref), f"{name} scatter mismatch on rank {rank}"
            back = impl.gather(out)
            ref2 = _ref_gather(out, ws, rank, group)
            torch.cuda.synchronize()
            assert torch.equal(back, ref2), f"{name} gather mismatch on rank {rank}"

        units = ["a2a", "e2e_attn"]
        samples = {n: {u: [] for u in units} for n in impl_names}
        for rep in range(args.repeats):
            # alternate the measurement order between repeats so slow drift
            # (clocks, thermals) does not systematically favor one impl
            order = impl_names if rep % 2 == 0 else impl_names[::-1]
            for name in order:
                for unit in units:
                    _measure_unit(  # warmup, same unit as timed
                        impls[name], unit, q, k, v, args.warmup, group
                    )
                    samples[name][unit].extend(
                        _measure_unit(impls[name], unit, q, k, v, args.iters, group)
                    )

        # per-sample max across ranks: a collective is only done when the
        # slowest rank is done
        reduced = {}
        for name in impl_names:
            reduced[name] = {}
            for unit in units:
                t = torch.tensor(samples[name][unit], device=device)
                gathered = [torch.empty_like(t) for _ in range(ws)]
                dist.all_gather(gathered, t, group=group)
                rank_max = torch.stack(gathered).max(dim=0).values.tolist()
                reduced[name][unit] = rank_max

        for impl in impls.values():
            impl.close()
        result = ("ok", reduced if rank == 0 else None)
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        import traceback

        result = (
            "err",
            f"rank {rank}: {type(e).__name__}: {e}\n{traceback.format_exc()[:2000]}",
        )
    q_out.put((rank, result))


def cmd_run(args):
    ws = args.world_size
    if torch.cuda.device_count() < ws:
        raise SystemExit(f"needs {ws} GPUs, have {torch.cuda.device_count()}")
    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    procs = [mp.Process(target=_worker, args=(r, ws, port, args, q)) for r in range(ws)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(ws):
        rank, res = q.get(timeout=1800)
        results[rank] = res
    for p in procs:
        p.join(timeout=120)
        if p.is_alive():
            p.terminate()
    errs = {r: res[1] for r, res in results.items() if res[0] != "ok"}
    if errs:
        raise SystemExit(f"workers failed: {errs}")

    reduced = results[0][1]
    try:
        # the commit of the flashinfer actually under test (PYTHONPATH may
        # point at a baseline worktree), not of this script's checkout
        import flashinfer

        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(flashinfer.__file__).resolve().parent,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    wl = default_workload(ws)
    if args.heads:
        wl["H"] = args.heads
    payload = {
        "meta": {
            "label": args.label,
            "commit": commit,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "world_size": ws,
            "workload": wl,
            "unit": "3x scatter_heads + 1x gather_heads (+sdpa for e2e_attn), ms",
            "repeats": args.repeats,
            "iters": args.iters,
            "warmup": args.warmup,
            "sample_reduction": "max across ranks per iteration",
            "order": "impl order alternates per repeat",
        },
        "results": {
            name: {
                unit: _stats(vals) | {"samples": vals} for unit, vals in units.items()
            }
            for name, units in reduced.items()
        },
    }
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "world_size",
                "impl",
                "unit",
                "p50_ms",
                "p95_ms",
                "mean_ms",
                "std_ms",
                "n",
            ]
        )
        for name, units in payload["results"].items():
            for unit, st in units.items():
                w.writerow(
                    [
                        args.label,
                        ws,
                        name,
                        unit,
                        st["p50"],
                        st["p95"],
                        st["mean"],
                        st["std"],
                        st["n"],
                    ]
                )
    print(f"[bench_ulysses_a2a] wrote {out.with_suffix('.json')} and .csv")
    for name, units in payload["results"].items():
        for unit, st in units.items():
            print(
                f"  ws={ws} {name:12s} {unit:8s} p50={st['p50']:8.3f} ms "
                f"p95={st['p95']:8.3f} std={st['std']:6.3f} n={st['n']}"
            )


def cmd_compare(args):
    base = json.loads(Path(args.baseline).read_text())
    new = json.loads(Path(args.new).read_text())
    ws_b, ws_n = base["meta"]["world_size"], new["meta"]["world_size"]
    if ws_b != ws_n:
        raise SystemExit(f"world sizes differ: {ws_b} vs {ws_n}")
    print(
        f"[compare] ws={ws_b} baseline={base['meta']['label']}@{base['meta']['commit']} "
        f"new={new['meta']['label']}@{new['meta']['commit']} "
        f"gate: {PRIMARY_UNIT} p50 regression <= {args.threshold_pct}%"
    )
    failed = False
    for pair_name, b_impl, n_impl in [
        ("raw->raw (kernel+raw path)", "raw", "raw"),
        ("raw->communicator (user-visible API)", "raw", "communicator"),
        ("nccl->nccl (fallback path)", "nccl", "nccl"),
    ]:
        if b_impl not in base["results"] or n_impl not in new["results"]:
            continue
        for unit in base["results"][b_impl]:
            if unit not in new["results"][n_impl]:
                continue
            b = base["results"][b_impl][unit]["p50"]
            n = new["results"][n_impl][unit]["p50"]
            delta = (n - b) / b * 100.0
            gate = unit == PRIMARY_UNIT
            verdict = ""
            if gate and delta > args.threshold_pct:
                verdict = "  << REGRESSION GATE EXCEEDED"
                failed = True
            print(
                f"  {pair_name:42s} {unit:8s} {b:8.3f} -> {n:8.3f} ms "
                f"({delta:+6.2f}%){' [gated]' if gate else ''}{verdict}"
            )
    if failed:
        raise SystemExit(1)
    print("[compare] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--world-size", type=int, required=True)
    r.add_argument("--impls", default="raw,communicator,nccl")
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--iters", type=int, default=30)
    r.add_argument("--warmup", type=int, default=5)
    r.add_argument("--heads", type=int, default=0, help="override head count")
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True, help="output path prefix (no suffix)")
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("baseline")
    c.add_argument("new")
    c.add_argument("--threshold-pct", type=float, default=3.0)
    c.set_defaults(func=cmd_compare)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
