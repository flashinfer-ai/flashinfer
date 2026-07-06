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
- methodology minimums are *enforced*, not defaults: >= 5 repeats x >= 30
  iters, known+available implementations, valid world size — the run refuses
  to start otherwise
- the measurement order rotates across repeats so every implementation
  occupies every position (Latin-square style), decorrelating position and
  thermal/clock drift from the impl identity; the actual per-repeat orders
  are recorded in the artifact
- p50 (conventional median) / p95 / std / mean over the rank-max samples,
  machine-readable JSON and CSV with full provenance (package commit + dirty
  state + import path, harness schema version + script hash)
- ``compare`` is fail-closed: it refuses to gate unless both artifacts come
  from the same harness/schema, identical workload/methodology metadata,
  *different* labels and commits, and every required gate pair is present

Implementations:

- ``raw``:              init_ulysses_a2a / ulysses_a2a (works on the c83e4204
                        baseline checkout and on current builds)
- ``communicator``:     UlyssesCommunicator public API, forced NVLink
                        (current builds only)
- ``communicator_nccl``: UlyssesCommunicator public API, forced NCCL —
                        measures the public fallback path / API overhead
- ``nccl_ref``:         self-contained dist.all_to_all_single + permute glue
                        algorithm control (no flashinfer dependency)

Also reports a secondary end-to-end proxy ("e2e_attn"): the same pattern with
a scaled_dot_product_attention between the scatters and the gather.

Usage (run on the current checkout, then on a baseline worktree, then gate):

    python benchmarks/bench_ulysses_a2a.py run --world-size 8 \
        --impls raw,communicator,communicator_nccl,nccl_ref \
        --label new --out /tmp/ulysses_new_w8

    PYTHONPATH=/path/to/c83e4204-worktree \
    python benchmarks/bench_ulysses_a2a.py run --world-size 8 \
        --impls raw,nccl_ref --label baseline --out /tmp/ulysses_base_w8

    python benchmarks/bench_ulysses_a2a.py compare \
        /tmp/ulysses_base_w8.json /tmp/ulysses_new_w8.json --threshold-pct 3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import socket
import statistics
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

SCHEMA_VERSION = "ulysses-a2a-m3-v2"
PRIMARY_UNIT = "a2a"  # regression gate applies to this unit only
UNITS = ["a2a", "e2e_attn"]
MIN_REPEATS = 5
MIN_ITERS = 30
SUPPORTED_WS = (2, 4, 6, 8)

# gate pairs the contract requires; compare fails if any is absent
REQUIRED_GATES = [
    ("raw->raw (kernel+raw path)", "raw", "raw"),
    ("raw->communicator (user-visible API)", "raw", "communicator"),
    ("nccl_ref control", "nccl_ref", "nccl_ref"),
]
# informational pairs, reported when present but never gated / required
OPTIONAL_PAIRS = [
    (
        "nccl_ref->communicator_nccl (public fallback API)",
        "nccl_ref",
        "communicator_nccl",
    ),
]
# metadata fields that must be identical between compared artifacts
COMPAT_META_FIELDS = [
    "schema",
    "harness_sha",
    "world_size",
    "workload",
    "unit",
    "sample_reduction",
    "order_policy",
    "repeats",
    "iters",
    "warmup",
    "torch",
    "device",
]


def harness_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def default_workload(world_size: int) -> dict:
    """Wan2.1-14B geometry (B=1, S_global=32760, 40 heads x 128, bf16) where
    the head count divides the world size; W=6 uses a divisible standalone
    workload of the same token count instead (40 % 6 != 0)."""
    heads = 40 if 40 % world_size == 0 else 48
    return dict(B=1, S_global=32760, H=heads, D=128, dtype="bfloat16")


# ---- implementations ----------------------------------------------------------


class NcclRefImpl:
    """all_to_all_single + permute glue; self-contained algorithm control
    (NOT the public UlyssesCommunicator NCCL backend — see
    ``communicator_nccl`` for that)."""

    name = "nccl_ref"

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
        # The signal zeroing inside init is an async cudaMemset; the baseline
        # (c83e4204) wrapper returns before it completes and a process-group
        # barrier is NOT a CUDA fence. Fence unconditionally (redundant but
        # harmless on current wrappers, required for the baseline).
        torch.cuda.synchronize()
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


class _CommunicatorImplBase:
    backend = "nvlink"

    def __init__(self, group, ws, shapes, device, dtype):
        from flashinfer.comm import UlyssesCommunicator

        B, S_local, H, D = shapes
        self.comm = UlyssesCommunicator(
            group,
            max_elems=B * S_local * H * D,
            dtype=dtype,
            backend=self.backend,
            device=device,
        )

    def scatter(self, x):
        return self.comm.scatter_heads(x)

    def gather(self, y):
        return self.comm.gather_heads(y)

    def close(self):
        torch.cuda.synchronize()
        self.comm.close()


class CommunicatorImpl(_CommunicatorImplBase):
    """UlyssesCommunicator public API, forced NVLink (current builds only)."""

    name = "communicator"
    backend = "nvlink"


class CommunicatorNcclImpl(_CommunicatorImplBase):
    """UlyssesCommunicator public API, forced NCCL: the public fallback path
    (its delta vs nccl_ref is the public API overhead)."""

    name = "communicator_nccl"
    backend = "nccl"


IMPLS = {
    c.name: c for c in (RawImpl, CommunicatorImpl, CommunicatorNcclImpl, NcclRefImpl)
}


def impl_available(name: str) -> bool:
    if name == "nccl_ref":
        return True
    import flashinfer.comm as comm

    if name in ("communicator", "communicator_nccl"):
        return hasattr(comm, "UlyssesCommunicator")
    return hasattr(comm, "init_ulysses_a2a")


# ---- methodology validation (pure; unit-tested on CPU) -------------------------


def validate_run_args(
    world_size: int,
    impls: list[str],
    repeats: int,
    iters: int,
    warmup: int,
    device_count: int,
    available=impl_available,
) -> None:
    """Enforce the contract's methodology minimums before any worker starts."""
    if world_size not in SUPPORTED_WS:
        raise ValueError(f"world_size must be one of {SUPPORTED_WS}, got {world_size}")
    if device_count < world_size:
        raise ValueError(f"needs {world_size} GPUs, have {device_count}")
    if repeats < MIN_REPEATS:
        raise ValueError(f"repeats must be >= {MIN_REPEATS}, got {repeats}")
    if iters < MIN_ITERS:
        raise ValueError(f"iters must be >= {MIN_ITERS}, got {iters}")
    if warmup < 1:
        raise ValueError(f"warmup must be >= 1, got {warmup}")
    if not impls:
        raise ValueError("at least one implementation is required")
    unknown = [n for n in impls if n not in IMPLS]
    if unknown:
        raise ValueError(f"unknown implementation(s) {unknown}; known: {sorted(IMPLS)}")
    unavailable = [n for n in impls if not available(n)]
    if unavailable:
        raise ValueError(
            f"implementation(s) {unavailable} unavailable in this flashinfer build"
        )
    wl = default_workload(world_size)
    if wl["S_global"] % world_size or wl["H"] % world_size:
        raise ValueError(f"workload {wl} not divisible by world_size {world_size}")


def rotation_order(impls: list[str], rep: int) -> list[str]:
    """Rotate the measurement order each repeat so every impl occupies every
    position across the run (with repeats >= len(impls))."""
    off = rep % len(impls)
    return impls[off:] + impls[:off]


def validate_compare(base: dict, new: dict) -> None:
    """Fail-closed compatibility check; raises ValueError on any mismatch."""
    import math

    for payload, role in ((base, "baseline"), (new, "new")):
        if "meta" not in payload or "results" not in payload:
            raise ValueError(f"{role} artifact is missing meta/results")
        meta = payload["meta"]
        if meta.get("package_dirty") or meta.get("commit") in (None, "", "unknown"):
            raise ValueError(
                f"{role} artifact has dirty/unknown package provenance "
                f"(commit={meta.get('commit')!r}, dirty={meta.get('package_dirty')!r}); "
                "gate only committed, clean builds"
            )
        expected_n = meta.get("repeats", 0) * meta.get("iters", 0)
        for impl, units in payload["results"].items():
            for unit, st in units.items():
                p50 = st.get("p50")
                if not (
                    isinstance(p50, (int, float)) and math.isfinite(p50) and p50 > 0
                ):
                    raise ValueError(f"{role} {impl}/{unit} has invalid p50={p50!r}")
                if st.get("n") != expected_n:
                    raise ValueError(
                        f"{role} {impl}/{unit} has n={st.get('n')} but "
                        f"repeats*iters={expected_n}; truncated or padded data"
                    )
    bm, nm = base["meta"], new["meta"]
    for field in COMPAT_META_FIELDS:
        if bm.get(field) != nm.get(field):
            raise ValueError(
                f"meta field {field!r} differs: baseline={bm.get(field)!r} "
                f"new={nm.get(field)!r}; artifacts are not comparable"
            )
    if bm.get("label") == nm.get("label"):
        raise ValueError(
            f"baseline and new labels are identical ({bm.get('label')!r}); "
            "refusing to gate an artifact against itself"
        )
    if bm.get("commit") == nm.get("commit"):
        raise ValueError(
            f"baseline and new package commits are identical ({bm.get('commit')!r}); "
            "the gate must compare different code"
        )
    missing = []
    for pair_name, b_impl, n_impl in REQUIRED_GATES:
        if b_impl not in base["results"]:
            missing.append(f"{pair_name}: {b_impl} missing from baseline")
        elif PRIMARY_UNIT not in base["results"][b_impl]:
            missing.append(f"{pair_name}: baseline {b_impl} lacks {PRIMARY_UNIT}")
        if n_impl not in new["results"]:
            missing.append(f"{pair_name}: {n_impl} missing from new")
        elif PRIMARY_UNIT not in new["results"][n_impl]:
            missing.append(f"{pair_name}: new {n_impl} lacks {PRIMARY_UNIT}")
    if missing:
        raise ValueError(
            "required gate pair(s) incomplete (the gate is fail-closed): "
            + "; ".join(missing)
        )


def compare_payloads(base: dict, new: dict, threshold_pct: float):
    """Returns (report_lines, gate_failed). Raises on incompatible inputs."""
    validate_compare(base, new)
    lines = []
    failed = False
    for pair_name, b_impl, n_impl in REQUIRED_GATES + OPTIONAL_PAIRS:
        if b_impl not in base["results"] or n_impl not in new["results"]:
            continue  # only possible for OPTIONAL_PAIRS after validation
        for unit in base["results"][b_impl]:
            if unit not in new["results"][n_impl]:
                continue
            b = base["results"][b_impl][unit]["p50"]
            n = new["results"][n_impl][unit]["p50"]
            delta = (n - b) / b * 100.0
            gated = unit == PRIMARY_UNIT and (pair_name, b_impl, n_impl) in [
                tuple(g) for g in REQUIRED_GATES
            ]
            verdict = ""
            if gated and delta > threshold_pct:
                verdict = "  << REGRESSION GATE EXCEEDED"
                failed = True
            lines.append(
                f"  {pair_name:48s} {unit:8s} {b:8.3f} -> {n:8.3f} ms "
                f"({delta:+6.2f}%){' [gated]' if gated else ''}{verdict}"
            )
    return lines, failed


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
        o = _sdpa(a, b, c) if unit == "e2e_attn" else a
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
        p50=statistics.median(s),
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
        B, S_global, H, D = wl["B"], wl["S_global"], wl["H"], wl["D"]
        S_local = S_global // ws
        dtype = getattr(torch, wl["dtype"])
        device = torch.device(f"cuda:{rank}")

        torch.manual_seed(1234 + rank)
        mk = lambda: torch.randn(  # noqa: E731
            B, S_local, H, D, dtype=dtype, device=device
        )
        q, k, v = mk(), mk(), mk()

        impl_names = args.impls.split(",")
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

        samples = {n: {u: [] for u in UNITS} for n in impl_names}
        orders = []
        for rep in range(args.repeats):
            order = rotation_order(impl_names, rep)
            orders.append(order)
            for name in order:
                for unit in UNITS:
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
            for unit in UNITS:
                t = torch.tensor(samples[name][unit], device=device)
                gathered = [torch.empty_like(t) for _ in range(ws)]
                dist.all_gather(gathered, t, group=group)
                rank_max = torch.stack(gathered).max(dim=0).values.tolist()
                reduced[name][unit] = rank_max

        for impl in impls.values():
            impl.close()
        result = ("ok", (reduced, orders) if rank == 0 else None)
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        import traceback

        result = (
            "err",
            f"rank {rank}: {type(e).__name__}: {e}\n{traceback.format_exc()[:2000]}",
        )
    q_out.put((rank, result))


def _package_provenance() -> dict:
    import flashinfer

    pkg_dir = Path(flashinfer.__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=pkg_dir, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=pkg_dir, text=True
            ).strip()
        )
    except Exception:  # noqa: BLE001
        commit, dirty = "unknown", True
    return {"commit": commit, "dirty": dirty, "import_path": str(pkg_dir)}


def cmd_run(args):
    ws = args.world_size
    impl_names = [n for n in args.impls.split(",") if n]
    validate_run_args(
        ws, impl_names, args.repeats, args.iters, args.warmup, torch.cuda.device_count()
    )
    args.impls = ",".join(impl_names)
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

    reduced, orders = results[0][1]
    prov = _package_provenance()
    payload = {
        "meta": {
            "schema": SCHEMA_VERSION,
            "harness_sha": harness_sha(),
            "label": args.label,
            "commit": prov["commit"],
            "package_dirty": prov["dirty"],
            "package_import_path": prov["import_path"],
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "world_size": ws,
            "workload": default_workload(ws),
            "unit": "3x scatter_heads + 1x gather_heads (+sdpa for e2e_attn), ms",
            "repeats": args.repeats,
            "iters": args.iters,
            "warmup": args.warmup,
            "sample_reduction": "max across ranks per iteration",
            "order_policy": "rotation per repeat (every impl visits every position)",
            "requested_impls": args.impls.split(","),
            "run_impls": list(reduced.keys()),
            "repeat_orders": orders,
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
                f"  ws={ws} {name:18s} {unit:8s} p50={st['p50']:8.3f} ms "
                f"p95={st['p95']:8.3f} std={st['std']:6.3f} n={st['n']}"
            )


def cmd_compare(args):
    base = json.loads(Path(args.baseline).read_text())
    new = json.loads(Path(args.new).read_text())
    lines, failed = compare_payloads(base, new, args.threshold_pct)
    bm, nm = base["meta"], new["meta"]
    print(
        f"[compare] ws={bm['world_size']} baseline={bm['label']}@{bm['commit'][:8]} "
        f"new={nm['label']}@{nm['commit'][:8]} "
        f"gate: {PRIMARY_UNIT} p50 regression <= {args.threshold_pct}%"
    )
    for line in lines:
        print(line)
    if failed:
        raise SystemExit(1)
    print("[compare] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--world-size", type=int, required=True)
    r.add_argument("--impls", default="raw,communicator,communicator_nccl,nccl_ref")
    r.add_argument("--repeats", type=int, default=MIN_REPEATS)
    r.add_argument("--iters", type=int, default=MIN_ITERS)
    r.add_argument("--warmup", type=int, default=5)
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
