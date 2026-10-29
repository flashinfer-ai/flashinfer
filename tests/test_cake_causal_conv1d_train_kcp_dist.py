"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Multi-GPU rows of ``cake_causal_conv1d_kcp`` under ``torch.utils.checkpoint`` with and without a recompute handle
(FlashInfer issues #7815 / #7818): ``W`` ranks over NCCL, each holding the fragments of one partition of a packed
batch (contiguous, zigzag, per-document zigzag, ragged tables with a zero-token rank, one-token fragments and a
fragment that reads its history from three predecessor ranks).  Per row and rank:

- ``y`` is bitwise the single-rank zero-history convolution of the whole pack on the rank's tokens (anchor);
- ``y`` / ``dx`` / ``dweight`` of the checkpoint with a handle (``use_reentrant`` False and True), of the plain
  checkpoint and of the direct pass are bitwise equal;
- the ``torch.distributed`` collectives issued (and the waits on their work handles) are counted per phase by
  test-local shims: the original forward and the backward issue one halo exchange each (positive control, > 0),
  the plain checkpoint's recompute issues one more, the recompute with a handle issues none;
- ``retain_graph`` + a second backward replays the entry again (no collective) and accumulates exactly twice;
- nested checkpoints: a handle created outside the outer region makes both recomputes communication-free; a handle
  created inside the outer region is empty in the outer recompute, which then exchanges once (still bitwise);
- the handle retains exactly the history pool: ``nbytes() == n_hist_rows * C * 2``, the bytes still allocated after
  the checkpointed forward are that pool (0 for the plain checkpoint).

Two launch forms (as ``tests/experimental/test_cake_kda_train_kcp_dist.py``): ``torch.multiprocessing`` (``pytest``:
skipped when fewer than ``W`` CUDA devices are visible) and one process per Slurm task (``SLURM_NTASKS == W``; the
rendezvous from ``MASTER_ADDR`` / ``MASTER_PORT``; rows of another ``W`` are skipped under a multi-task step).
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WIDTH = 4

# (document lengths, W, layout, C)
ROWS = [
    ((100, 640, 77, 1000, 129), 2, "contiguous", 512),
    ((3000,), 2, "zigzag", 256),
    ((100, 640, 77, 1000, 129), 4, "contiguous", 512),
    ((3000,), 4, "zigzag_per_document", 256),
    ((2048, 5, 3000, 1), 4, "zigzag_per_document", 384),
    ((1200 + 900 + 900,), 4, "ragged:1200,0,900,900", 256),
    ((5000, 1003), 4, "ragged:1,2,4997,1003", 512),
    ((3000,), 8, "zigzag", 256),
    ((2048, 5, 3000, 1), 8, "zigzag_per_document", 384),
    ((6, 1, 2, 3, 4000), 8, "ragged:1,1,1,3,0,2,4000,4", 256),
]
TENSORS = ("y", "dx", "dw")


def _row_id(row) -> str:
    lengths, W, layout, C = row
    return "s" + "_".join(map(str, lengths)) + f"-w{W}-{layout.split(':')[0]}-c{C}"


def _partition(plan, lengths, W: int, layout: str):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    if not layout.startswith("ragged:"):
        return plan.kcp_partition(cu, W, layout=layout), cu
    t_r = [int(x) for x in layout.split(":")[1].split(",")]
    assert len(t_r) == W and sum(t_r) == cu[-1]
    # contiguous ragged split of the global buffer, cut at document boundaries, in local order
    nested = [[] for _ in range(W)]
    pos = 0
    for r, n in enumerate(t_r):
        end = pos + n
        d = 0
        while d < len(cu) - 1 and cu[d + 1] <= pos:
            d += 1
        p = pos
        while p < end and d < len(cu) - 1:
            e = min(end, cu[d + 1])
            if e > p:
                nested[r].append((d, p - cu[d], e - p))
            p = e
            d += 1
        pos = end
    return plan.kcp_partition(cu, W, fragments=nested), cu


def _local(part, rank: int, cu: list[int], t: torch.Tensor) -> torch.Tensor:
    parts = [
        t[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length]
        for f in part.fragments[rank]
    ]
    return torch.cat(parts, 0).contiguous() if parts else t[:0].clone()


class _Counter:
    """Counts the ``torch.distributed`` collectives issued and the waits on their work handles, by phase (``orig``,
    ``recompute`` -- set by the checkpoint's ``context_fn``, re-entered per recompute -- and ``bwd``).  The shims
    replace the module attributes the conv package calls while the counter is entered; no tensor is read."""

    NAMES = (
        "all_to_all_single",
        "batch_isend_irecv",
        "all_gather_object",
        "all_gather",
        "all_gather_into_tensor",
        "all_reduce",
        "broadcast",
        "reduce_scatter_tensor",
        "isend",
        "irecv",
        "send",
        "recv",
    )

    def __init__(self) -> None:
        self.phase = "orig"
        self.calls: dict[str, int] = {}
        self.waits: dict[str, int] = {}
        self._saved: dict[str, object] = {}
        self._stack: list[str] = []

    def _bump(self, table: dict[str, int]) -> None:
        table[self.phase] = table.get(self.phase, 0) + 1

    def _work(self, w):
        counter = self

        class _Work:
            def __init__(self, inner) -> None:
                self._inner = inner

            def wait(self, *a, **kw):
                counter._bump(counter.waits)
                return self._inner.wait(*a, **kw)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        return _Work(w) if hasattr(w, "wait") else w

    def __enter__(self) -> "_Counter":
        for name in self.NAMES:
            fn = getattr(dist, name, None)
            if fn is None:
                continue
            self._saved[name] = fn

            def shim(*a, _fn=fn, **kw):
                self._bump(self.calls)
                r = _fn(*a, **kw)
                if isinstance(r, list):
                    return [self._work(x) for x in r]
                return self._work(r)

            setattr(dist, name, shim)
        return self

    def __exit__(self, *exc) -> None:
        for name, fn in self._saved.items():
            setattr(dist, name, fn)
        self._saved.clear()

    def recompute_cm(self):
        counter = self

        class _Phase:
            def __enter__(self):
                counter._stack.append(counter.phase)
                counter.phase = "recompute"
                return self

            def __exit__(self, *exc):
                counter.phase = counter._stack.pop()
                return False

        return _Phase()

    def get(self, table: str, phase: str) -> int:
        return getattr(self, table).get(phase, 0)


def _requested(device) -> int:
    return int(torch.cuda.memory_stats(device).get("requested_bytes.all.current", 0))


def _run_rank(
    rank: int, W: int, row, port: int, result_dir: str | None, local_id: int
) -> dict:
    from torch.utils.checkpoint import checkpoint

    from flashinfer import cake_causal_conv1d_train as conv
    from flashinfer.cake_causal_conv1d_train_backend import kcp as conv_kcp
    from flashinfer.experimental.cake_kda_train import cake_kcp_plan as plan

    lengths, _w, layout, C = row
    device = torch.device("cuda", local_id % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
    group = dist.group.WORLD
    part, cu = _partition(plan, lengths, W, layout)
    ctx = plan.build_kcp_context(part, group, rank, device, conv_kernel_size=WIDTH)
    T = cu[-1]
    g = torch.Generator(device="cpu").manual_seed(7815)
    x_all = torch.randn(T, C, generator=g).to(torch.bfloat16)
    w_all = (torch.randn(C, WIDTH, generator=g) * 0.5).contiguous()
    dy_all = torch.randn(T, C, generator=g).to(torch.bfloat16)
    x = _local(part, rank, cu, x_all).to(device)
    dy = _local(part, rank, cu, dy_all).to(device)
    w = w_all.to(device)
    n_hist = int(ctx.halo.n_hist_rows)

    def leaves():
        return x.detach().clone().requires_grad_(
            True
        ), w.detach().clone().requires_grad_(True)

    def fn(x_, w_, h_):
        return conv.cake_causal_conv1d_kcp(x_, w_, ctx, recompute_handle=h_)

    def run(mode: str, handle=None, *, twice: bool = False, reentrant: bool = False):
        """mode: direct | ckpt | nested_out (handle outside both regions) | nested_in (handle made in the outer)."""
        xl, wl = leaves()
        with _Counter() as cc:
            if mode == "direct":
                y = fn(xl, wl, handle)
            elif mode == "ckpt":
                kw = (
                    {"use_reentrant": True}
                    if reentrant
                    else {
                        "use_reentrant": False,
                        "context_fn": lambda: (
                            contextlib.nullcontext(),
                            cc.recompute_cm(),
                        ),
                    }
                )
                y = checkpoint(fn, xl, wl, handle, **kw)
            else:

                def outer(x_, w_, h_):
                    # the product with ones saves a tensor of the outer region (so the outer recompute runs) and
                    # changes no bit of y or of its gradient
                    hh = conv_kcp.KcpRecomputeHandle() if mode == "nested_in" else h_
                    yi = checkpoint(fn, x_, w_, hh, use_reentrant=False)
                    return yi * torch.ones_like(yi)

                y = checkpoint(
                    outer,
                    xl,
                    wl,
                    handle,
                    use_reentrant=False,
                    context_fn=lambda: (contextlib.nullcontext(), cc.recompute_cm()),
                )
            cc.phase = "bwd"
            y.backward(dy, retain_graph=twice)
            if twice:
                y.backward(dy)
        out = {"y": y.detach(), "dx": xl.grad, "dw": wl.grad}
        counts = {
            t: {p: cc.get(t, p) for p in ("orig", "recompute", "bwd")}
            for t in ("calls", "waits")
        }
        return out, counts

    def bits(t: torch.Tensor) -> torch.Tensor:
        # bit view: -0.0 vs 0.0 and NaN payloads count as differences
        return t.contiguous().view(
            torch.int16 if t.element_size() == 2 else torch.int32
        )

    def same(a: torch.Tensor, b: torch.Tensor) -> bool:
        return (
            a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))
        )

    def eq(a, b) -> bool:
        return all(same(a[n], b[n]) for n in TENSORS)

    flags: dict[str, bool] = {}
    info: dict[str, object] = {"n_hist_rows": n_hist, "T_r": int(x.shape[0])}
    direct, c_direct = run("direct")
    plain, c_plain = run("ckpt")
    hd = conv_kcp.KcpRecomputeHandle()
    withh, c_h = run("ckpt", hd)
    plain_re, c_plain_re = run("ckpt", reentrant=True)
    hd_re = conv_kcp.KcpRecomputeHandle()
    withh_re, c_h_re = run("ckpt", hd_re, reentrant=True)
    info["counts"] = {
        "direct": c_direct,
        "plain": c_plain,
        "handle": c_h,
        "plain_reentrant": c_plain_re,
        "handle_reentrant": c_h_re,
    }
    # anchor: y is the single-rank zero-history convolution of the whole pack on this rank's tokens
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
    y_single = conv.cake_causal_conv1d(x_all.to(device), w, cu_dev)
    flags["y_eq_single_rank"] = same(direct["y"], _local(part, rank, cu, y_single))
    flags["plain_eq_direct"] = eq(plain, direct)
    flags["handle_eq_plain"] = eq(withh, plain)
    flags["handle_eq_direct"] = eq(withh, direct)
    flags["reentrant_plain_eq_direct"] = eq(plain_re, direct)
    flags["reentrant_handle_eq_direct"] = eq(withh_re, direct)
    for t in ("calls", "waits"):
        # positive control: one halo exchange (and its wait) in the forward and one in the backward on every rank
        flags[f"positive_control_{t}"] = c_direct[t] == {
            "orig": 1,
            "recompute": 0,
            "bwd": 1,
        } and c_plain[t] == {"orig": 1, "recompute": 1, "bwd": 1}
        flags[f"handle_counts_{t}"] = c_h[t] == {"orig": 1, "recompute": 0, "bwd": 1}
        # reentrant: the recompute runs inside the backward (no context_fn): plain = exchange + recompute exchange
        flags[f"reentrant_counts_{t}"] = c_plain_re[t] == {
            "orig": 1,
            "recompute": 0,
            "bwd": 2,
        } and c_h_re[t] == {"orig": 1, "recompute": 0, "bwd": 1}
    flags["handle_bytes"] = (
        hd.nbytes() == n_hist * C * 2 == hd_re.nbytes()
        and hd.entries["conv"]["hist"].shape == (n_hist, C)
    )
    # retain_graph: the entry is replayed again, no collective; exactly twice the gradients
    hd2 = conv_kcp.KcpRecomputeHandle()
    twice, c_twice = run("ckpt", hd2, twice=True)
    info["counts"]["handle_twice"] = c_twice
    flags["twice_accumulates"] = (
        same(twice["dx"], direct["dx"] + direct["dx"])
        and same(twice["dw"], direct["dw"] + direct["dw"])
        and c_twice["calls"] == {"orig": 1, "recompute": 0, "bwd": 2}
    )
    # nested checkpoints
    nest_out, c_nest_out = run("nested_out", conv_kcp.KcpRecomputeHandle())
    nest_in, c_nest_in = run("nested_in", conv_kcp.KcpRecomputeHandle())
    info["counts"]["nested_out"], info["counts"]["nested_in"] = c_nest_out, c_nest_in
    flags["nested_out_bitwise"] = eq(nest_out, direct)
    flags["nested_in_bitwise"] = eq(nest_in, direct)
    # handle made outside both regions: the outer recompute (recompute phase) and the inner recompute (counted in
    # the backward phase, outside the outer's context) replay the entry; only the backward exchange remains
    flags["nested_out_counts"] = c_nest_out["calls"] == {
        "orig": 1,
        "recompute": 0,
        "bwd": 1,
    }
    # handle made inside the outer region: the outer recompute's inner forward is an original forward (exchanges)
    flags["nested_in_counts"] = c_nest_in["calls"] == {
        "orig": 1,
        "recompute": 1,
        "bwd": 1,
    }
    # retained bytes after a checkpointed forward (after the warm-up above): the history pool with a handle, 0 without
    retained = {}
    for name, h in (("plain", None), ("handle", conv_kcp.KcpRecomputeHandle())):
        xl, wl = leaves()
        torch.cuda.synchronize(device)
        gc.collect()
        before = _requested(device)
        y = checkpoint(fn, xl, wl, h, use_reentrant=False)
        torch.cuda.synchronize(device)
        retained[name] = _requested(device) - before - y.numel() * y.element_size()
        y.backward(dy)
        del y, xl, wl
    info["retained_bytes"] = retained
    flags["retained"] = retained["plain"] == 0 and retained["handle"] == n_hist * C * 2
    flag_list = [None] * W
    info_list = [None] * W
    dist.all_gather_object(flag_list, flags)
    dist.all_gather_object(info_list, info)
    report = {
        "world": W,
        "row": _row_id(row),
        "torch": torch.__version__,
        "flags": {k: all(f[k] for f in flag_list) for k in flags},
        "failing_ranks": {
            k: [r for r, f in enumerate(flag_list) if not f[k]] for k in flags
        },
        "info_by_rank": info_list,
    }
    report["pass"] = all(report["flags"].values())
    dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, _row_id(row) + ".json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_entry(rank: int, W: int, row, port: int, result_dir: str) -> None:
    try:
        report = _run_rank(rank, W, row, port, result_dir, rank)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    from flashinfer.cake_causal_conv1d_train import (
        is_cake_causal_conv1d_train_supported,
    )

    return is_cake_causal_conv1d_train_supported(torch.device("cuda"))


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_recompute_handle_rows(row) -> None:
    """Bitwise and per-phase collective counts of the checkpoint with / without a recompute handle (module docstring)."""
    W = row[1]
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _supported():
            pytest.skip("the conv package does not support this device")
        report = _run_rank(
            int(os.environ["SLURM_PROCID"]),
            W,
            row,
            int(os.environ.get("MASTER_PORT", "29500")),
            os.environ.get("KCP_TEST_RESULT_DIR"),
            int(os.environ.get("SLURM_LOCALID", "0")),
        )
        assert report["pass"], report
        return
    if not _supported():
        pytest.skip("cake_causal_conv1d_train needs a supported CUDA device")
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    with tempfile.TemporaryDirectory(prefix="cake_conv_kcp_dist_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_entry,
            args=(W, row, _free_port(), result_dir),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report
