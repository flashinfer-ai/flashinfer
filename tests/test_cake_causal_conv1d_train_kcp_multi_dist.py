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

Multi-GPU rows of ``cake_causal_conv1d_kcp_multi`` (FlashInfer issue #7815): the q / k / v convolutions of a layer
with one halo exchange per direction, against one ``cake_causal_conv1d_kcp`` call per tensor.  ``W`` ranks over NCCL,
each holding the fragments of one partition of a packed batch (contiguous, zigzag, per-document zigzag, ragged tables
with a zero-token rank, one-token fragments and a fragment that reads its history from three predecessor ranks).  Per
row and rank:

- ``y`` / ``dx`` / ``dweight`` of every tensor are bitwise those of the single calls: direct, plain checkpoint,
  checkpoint with one handle (multi) vs one handle per tensor (single), ``use_reentrant`` False and True, nested
  checkpoints, ``retain_graph`` (exactly twice); ``y`` is also bitwise the single-rank zero-history convolution;
- the ``torch.distributed`` collectives and the waits on their work handles, per phase (original forward /
  recompute / backward): multi 1/0/1 direct, 1/1/1 plain checkpoint, 1/0/1 with a handle, against 3/0/3, 3/3/3 and
  3/0/3 for three single calls;
- the multi handle retains exactly the n history pools (``nbytes`` and the allocator's bytes after the forward);
- ``graph="replay"`` of the multi forward / backward is bitwise the eager single calls on every call.

Launch forms as ``test_cake_causal_conv1d_train_kcp_dist.py``: ``torch.multiprocessing`` (``pytest``: skipped when
fewer than ``W`` CUDA devices are visible) and one process per Slurm task (``SLURM_NTASKS == W``; the rendezvous from
``MASTER_ADDR`` / ``MASTER_PORT``; rows of another ``W`` are skipped under a multi-task step).
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_cake_causal_conv1d_train_kcp_dist import (
    _free_port,
    _local,
    _partition,
    _requested,
    _supported,
)
from tests.test_helpers.kcp_collective_shims import CollectiveCounter, RecomputePhase

WIDTH = 4

# (document lengths, W, layout, channel counts of the tensors)
ROWS = [
    ((100, 640, 77, 1000, 129), 2, "contiguous", (512, 512, 512)),
    ((3000,), 2, "zigzag", (256, 128, 256)),
    ((100, 640, 77, 1000, 129), 4, "contiguous", (512, 512, 512)),
    ((3000,), 4, "zigzag_per_document", (256, 256, 256)),
    ((2048, 5, 3000, 1), 4, "zigzag_per_document", (384, 384, 128)),
    ((1200 + 900 + 900,), 4, "ragged:1200,0,900,900", (256, 256, 256)),
    ((5000, 1003), 4, "ragged:1,2,4997,1003", (512, 256, 512)),
    ((3000,), 8, "zigzag", (256, 256, 256)),
    ((2048, 5, 3000, 1), 8, "zigzag_per_document", (384, 384, 384)),
    ((6, 1, 2, 3, 4000), 8, "ragged:1,1,1,3,0,2,4000,4", (256, 128, 256)),
]
GRAPH_ROWS = [ROWS[0], ROWS[5], ROWS[8]]
TENSORS = ("y", "dx", "dw")


def _row_id(row) -> str:
    lengths, W, layout, Cs = row
    return (
        "s"
        + "_".join(map(str, lengths))
        + f"-w{W}-{layout.split(':')[0]}-c"
        + "_".join(map(str, Cs))
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    # bit view: -0.0 vs 0.0 and NaN payloads count as differences
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _setup(rank: int, W: int, row, port: int, local_id: int):
    from flashinfer.experimental.cake_kda_train import cake_kcp_plan as plan

    lengths, _w, layout, Cs = row
    device = torch.device("cuda", local_id % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
    part, cu = _partition(plan, lengths, W, layout)
    ctx = plan.build_kcp_context(
        part, dist.group.WORLD, rank, device, conv_kernel_size=WIDTH
    )
    g = torch.Generator(device="cpu").manual_seed(7815)
    xs_all, ws, dys = [], [], []
    for C in Cs:
        x_all = torch.randn(cu[-1], C, generator=g).to(torch.bfloat16)
        w_all = (torch.randn(C, WIDTH, generator=g) * 0.5).contiguous()
        dy_all = torch.randn(cu[-1], C, generator=g).to(torch.bfloat16)
        xs_all.append(x_all)
        ws.append(w_all.to(device))
        dys.append(_local(part, rank, cu, dy_all).to(device))
    xs = [_local(part, rank, cu, x).to(device) for x in xs_all]
    return device, part, cu, ctx, xs_all, xs, ws, dys


def _report(W: int, row, flags: dict, info: dict, result_dir, rank: int, tag: str):
    flag_list = [None] * W
    info_list = [None] * W
    dist.all_gather_object(flag_list, flags)
    dist.all_gather_object(info_list, info)
    report = {
        "world": W,
        "row": _row_id(row),
        "test": tag,
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
        with open(os.path.join(result_dir, f"{tag}-{_row_id(row)}.json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _run_rank(rank: int, W: int, row, port: int, result_dir, local_id: int) -> dict:
    from torch.utils.checkpoint import checkpoint

    from flashinfer import cake_causal_conv1d_train as conv
    from flashinfer.cake_causal_conv1d_train_backend import kcp as conv_kcp

    Cs = row[3]
    n = len(Cs)
    device, part, cu, ctx, xs_all, xs, ws, dys = _setup(rank, W, row, port, local_id)
    n_hist = int(ctx.halo.n_hist_rows)

    def leaves():
        return [x.detach().clone().requires_grad_(True) for x in xs], [
            w.detach().clone().requires_grad_(True) for w in ws
        ]

    def layer(multi: bool, c):
        def fn(*args):
            a_x, a_w, a_h = args[:n], args[n : 2 * n], args[2 * n :]
            if multi:
                return conv.cake_causal_conv1d_kcp_multi(
                    a_x, a_w, c, recompute_handle=a_h[0]
                )
            return tuple(
                conv.cake_causal_conv1d_kcp(x, w, c, recompute_handle=h)
                for x, w, h in zip(a_x, a_w, a_h, strict=True)
            )

        return fn

    def run(
        multi: bool, mode: str, handles=None, *, twice=False, reentrant=False, c=ctx
    ):
        """mode: direct | ckpt | nested_out (handle outside both regions) | nested_in (handle made in the outer)."""
        fn = layer(multi, c)
        if handles is None:
            handles = [None] if multi else [None] * n
        xl, wl = leaves()
        with CollectiveCounter() as cc:
            ctx_fn = lambda: (contextlib.nullcontext(), RecomputePhase(cc))  # noqa: E731
            if mode == "direct":
                ys = fn(*xl, *wl, *handles)
            elif mode == "ckpt":
                kw = (
                    {"use_reentrant": True}
                    if reentrant
                    else {"use_reentrant": False, "context_fn": ctx_fn}
                )
                ys = checkpoint(fn, *xl, *wl, *handles, **kw)
            else:

                def outer(*args):
                    hh = (
                        [conv_kcp.KcpRecomputeHandle()]
                        if mode == "nested_in"
                        else list(args[2 * n :])
                    )
                    yi = checkpoint(fn, *args[: 2 * n], *hh, use_reentrant=False)
                    return tuple(y * torch.ones_like(y) for y in yi)

                ys = checkpoint(
                    outer, *xl, *wl, *handles, use_reentrant=False, context_fn=ctx_fn
                )
            cc.phase = "bwd"
            torch.autograd.backward(list(ys), list(dys), retain_graph=twice)
            if twice:
                torch.autograd.backward(list(ys), list(dys))
        out = [
            {"y": y.detach(), "dx": x.grad, "dw": w.grad}
            for y, x, w in zip(ys, xl, wl, strict=True)
        ]
        rec = cc.record()
        counts = {
            t: {p: rec[t].get(p, 0) for p in ("orig", "recompute", "bwd")}
            for t in ("calls", "waits")
        }
        return out, counts

    def eq(a, b) -> bool:
        return all(_same(a[i][t], b[i][t]) for i in range(n) for t in TENSORS)

    def table(o, r, b):
        return {"orig": o, "recompute": r, "bwd": b}

    flags: dict[str, bool] = {}
    info: dict[str, object] = {"n_hist_rows": n_hist, "T_r": int(xs[0].shape[0])}
    s_direct, cs_direct = run(False, "direct")
    s_plain, cs_plain = run(False, "ckpt")
    s_hds = [conv_kcp.KcpRecomputeHandle() for _ in range(n)]
    s_handle, cs_handle = run(False, "ckpt", s_hds)
    m_direct, cm_direct = run(True, "direct")
    m_plain, cm_plain = run(True, "ckpt")
    m_hd = conv_kcp.KcpRecomputeHandle()
    m_handle, cm_handle = run(True, "ckpt", [m_hd])
    m_plain_re, cm_plain_re = run(True, "ckpt", reentrant=True)
    m_handle_re, cm_handle_re = run(
        True, "ckpt", [conv_kcp.KcpRecomputeHandle()], reentrant=True
    )
    m_twice, cm_twice = run(True, "ckpt", [conv_kcp.KcpRecomputeHandle()], twice=True)
    m_nest_out, cm_nest_out = run(True, "nested_out", [conv_kcp.KcpRecomputeHandle()])
    m_nest_in, cm_nest_in = run(True, "nested_in", [conv_kcp.KcpRecomputeHandle()])
    info["counts"] = {
        "single_direct": cs_direct,
        "single_plain": cs_plain,
        "single_handle": cs_handle,
        "multi_direct": cm_direct,
        "multi_plain": cm_plain,
        "multi_handle": cm_handle,
        "multi_plain_reentrant": cm_plain_re,
        "multi_handle_reentrant": cm_handle_re,
        "multi_handle_twice": cm_twice,
        "multi_nested_out": cm_nest_out,
        "multi_nested_in": cm_nest_in,
    }
    # anchor: y is the single-rank zero-history convolution of the whole pack on this rank's tokens
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
    flags["y_eq_single_rank"] = all(
        _same(
            m_direct[i]["y"],
            _local(
                part,
                rank,
                cu,
                conv.cake_causal_conv1d(xs_all[i].to(device), ws[i], cu_dev),
            ),
        )
        for i in range(n)
    )
    flags["multi_direct_eq_single_direct"] = eq(m_direct, s_direct)
    flags["multi_plain_eq_single_plain"] = eq(m_plain, s_plain)
    flags["multi_handle_eq_single_handle"] = eq(m_handle, s_handle)
    flags["single_plain_eq_direct"] = eq(s_plain, s_direct)
    flags["single_handle_eq_direct"] = eq(s_handle, s_direct)
    flags["multi_reentrant_plain_eq_direct"] = eq(m_plain_re, s_direct)
    flags["multi_reentrant_handle_eq_direct"] = eq(m_handle_re, s_direct)
    flags["multi_nested_out_bitwise"] = eq(m_nest_out, s_direct)
    flags["multi_nested_in_bitwise"] = eq(m_nest_in, s_direct)
    flags["multi_twice_accumulates"] = all(
        _same(m_twice[i][t], s_direct[i][t] + s_direct[i][t])
        for i in range(n)
        for t in ("dx", "dw")
    )
    for t in ("calls", "waits"):
        # positive control: n exchanges per direction for the single calls
        flags[f"single_counts_{t}"] = (
            cs_direct[t] == table(n, 0, n)
            and cs_plain[t] == table(n, n, n)
            and cs_handle[t] == table(n, 0, n)
        )
        flags[f"multi_counts_{t}"] = (
            cm_direct[t] == table(1, 0, 1)
            and cm_plain[t] == table(1, 1, 1)
            and cm_handle[t] == table(1, 0, 1)
        )
        # reentrant: the recompute runs inside the backward (no context_fn), counted as "bwd"
        flags[f"multi_reentrant_counts_{t}"] = cm_plain_re[t] == table(
            1, 0, 2
        ) and cm_handle_re[t] == table(1, 0, 1)
    flags["multi_twice_counts"] = cm_twice["calls"] == table(1, 0, 2)
    flags["multi_nested_out_counts"] = cm_nest_out["calls"] == table(1, 0, 1)
    flags["multi_nested_in_counts"] = cm_nest_in["calls"] == table(1, 1, 1)
    pool_bytes = sum(n_hist * C * 2 for C in Cs)
    entry = m_hd.entries["conv"]
    flags["handle_bytes"] = (
        m_hd.nbytes() == pool_bytes == sum(h.nbytes() for h in s_hds)
        and all(h.shape == (n_hist, C) for h, C in zip(entry["hists"], Cs, strict=True))
        and all(
            _same(entry["hists"][i], s_hds[i].entries["conv"]["hist"]) for i in range(n)
        )
    )
    # retained bytes after a checkpointed forward: the n history pools with a handle, 0 without
    retained = {}
    for name, h in (("plain", None), ("handle", conv_kcp.KcpRecomputeHandle())):
        xl, wl = leaves()
        torch.cuda.synchronize(device)
        gc.collect()
        before = _requested(device)
        ys = checkpoint(layer(True, ctx), *xl, *wl, h, use_reentrant=False)
        torch.cuda.synchronize(device)
        retained[name] = (
            _requested(device) - before - sum(y.numel() * y.element_size() for y in ys)
        )
        torch.autograd.backward(list(ys), list(dys))
        del ys, xl, wl
    info["retained_bytes"] = retained
    flags["retained"] = retained["plain"] == 0 and retained["handle"] == pool_bytes
    # the GDN package's context (duck-typed, the other call path): the same bits and counts
    from flashinfer.experimental.cake_gdn_train import cake_kcp_plan as gplan

    gpart, _ = _partition(gplan, row[0], W, row[2])
    gctx = gplan.build_kcp_context(gpart, dist.group.WORLD, rank, device)
    g_single, cg_single = run(False, "direct", c=gctx)
    g_multi, cg_multi = run(True, "direct", c=gctx)
    g_handle, cg_handle = run(True, "ckpt", [conv_kcp.KcpRecomputeHandle()], c=gctx)
    info["counts"]["gdn_ctx"] = {
        "single_direct": cg_single,
        "multi_direct": cg_multi,
        "multi_handle": cg_handle,
    }
    flags["gdn_ctx_bitwise"] = eq(g_multi, g_single) and eq(g_handle, g_single)
    flags["gdn_ctx_matches_kda_ctx"] = [len(f) for f in gpart.fragments] == [
        len(f) for f in part.fragments
    ] and eq(g_single, s_direct)
    flags["gdn_ctx_counts"] = (
        cg_single["calls"] == table(n, 0, n)
        and cg_multi["calls"] == table(1, 0, 1)
        and cg_handle["calls"] == table(1, 0, 1)
    )
    return _report(W, row, flags, info, result_dir, rank, "multi")


def _run_rank_graph(
    rank: int, W: int, row, port: int, result_dir, local_id: int
) -> dict:
    """``graph="replay"`` of the multi forward / backward: eager, capture, replay -- bitwise the eager single calls on
    every call; a rank without tokens is never captured (it joins each collective eagerly)."""
    from flashinfer.cake_causal_conv1d_train_backend import kcp as conv_kcp

    device, part, cu, ctx, xs_all, xs, ws, dys = _setup(rank, W, row, port, local_id)
    ref = []
    for x, w, dy in zip(xs, ws, dys, strict=True):
        y, hist = conv_kcp.cake_causal_conv1d_kcp_forward(x, w, ctx)
        dx, dw = conv_kcp.cake_causal_conv1d_kcp_backward(x, w, ctx, dy, hist)
        ref.append((y, hist, dx, dw))
    events, ok = [], True
    try:
        for _ in range(4):
            ys, hists = conv_kcp.cake_causal_conv1d_kcp_multi_forward(
                xs, ws, ctx, graph="replay"
            )
            events.append(conv_kcp.LAST_CALL["fwd_multi"]["graph_event"])
            dxs, dws = conv_kcp.cake_causal_conv1d_kcp_multi_backward(
                xs, ws, ctx, dys, hists, graph="replay"
            )
            events.append(conv_kcp.LAST_CALL["bwd_multi"]["graph_event"])
            torch.cuda.synchronize(device)
            for i, (y, hist, dx, dw) in enumerate(ref):
                ok = ok and _same(ys[i], y) and _same(hists[i], hist)
                ok = ok and _same(dxs[i], dx) and _same(dws[i], dw)
    finally:
        torch.cuda.synchronize(device)
        conv_kcp.graph_cache_clear()
    # the backward's key holds the forward's pools: it captures one call after the forward, except on a rank that
    # receives no history row (its pools are empty tensors without storage, so every forward returns the same
    # key) -- that backward captures with the forward
    if int(xs[0].shape[0]) == 0:
        expected = ["eager"] * 8
    elif int(ctx.halo.n_hist_rows) == 0:
        expected = ["eager", "eager", "capture", "capture"] + ["replay"] * 4
    else:
        expected = ["eager", "eager", "capture", "eager", "replay", "capture"]
        expected += ["replay", "replay"]
    flags = {"graph_bitwise": ok, "graph_events": events == expected}
    info = {
        "events": events,
        "T_r": int(xs[0].shape[0]),
        "n_hist_rows": int(ctx.halo.n_hist_rows),
    }
    return _report(W, row, flags, info, result_dir, rank, "graph")


def _spawn_entry(
    rank: int, W: int, row, port: int, result_dir: str, graph: bool
) -> None:
    try:
        fn = _run_rank_graph if graph else _run_rank
        report = fn(rank, W, row, port, result_dir, rank)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _launch(row, graph: bool) -> None:
    W = row[1]
    tag = "graph" if graph else "multi"
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _supported():
            pytest.skip("the conv package does not support this device")
        fn = _run_rank_graph if graph else _run_rank
        report = fn(
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
    with tempfile.TemporaryDirectory(prefix="cake_conv_kcp_multi_dist_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_entry,
            args=(W, row, _free_port(), result_dir, graph),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, f"{tag}-{_row_id(row)}.json")) as fh:
            report = json.load(fh)
    assert report["pass"], report


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_multi_rows(row) -> None:
    """Bitwise against the single calls and per-phase collective counts (module docstring)."""
    _launch(row, graph=False)


@pytest.mark.parametrize("row", GRAPH_ROWS, ids=_row_id)
def test_multi_graph_replay_rows(row) -> None:
    """``graph="replay"`` of the multi forward / backward over NCCL (module docstring)."""
    _launch(row, graph=True)
