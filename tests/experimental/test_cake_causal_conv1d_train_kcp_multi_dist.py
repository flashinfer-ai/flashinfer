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

Multi-GPU rows of the non-deterministic ``cake_causal_conv1d_multi_kcp(..., deterministic=False)`` (the q / k / v
convolutions of a layer with one halo exchange per direction) on the deterministic multi suite's partitions, W = 2 /
4 / 8 over NCCL.  Per row and rank, for every tensor:

- ``y`` and ``dx`` are bitwise the non-deterministic single calls' and the deterministic multi call's; ``y`` is also
  bitwise the single-rank zero-history convolution; each ``dweight`` partial is within the package budget of the
  deterministic one -- direct, plain checkpoint, checkpoint with one handle, ``use_reentrant`` False and True,
  nested checkpoints and ``retain_graph`` (``dx`` exactly twice, ``dweight`` within twice the budget);
- an independent reference per tensor: the rank's ``y`` and ``dx`` (its tokens gathered from the global FP64
  oracle of the whole pack under the KCP SiLU' policy) within one BF16 ulp, and the all-reduced ``dweight``
  partial sum within the package budget of the global FP64 ``dweight``;
- the collective counts per phase of the deterministic suite: multi 1/0/1 direct, 1/1/1 plain checkpoint, 1/0/1
  with a handle; single calls n/0/n;
- ``graph="replay"`` of the multi forward / backward: ``y`` / ``hist`` / ``dx`` bitwise the eager single calls on
  every call, ``dweight`` within budget, the deterministic suite's event sequence.

Launch forms as the deterministic suite.  Skipped until the exporter renders the non-deterministic package.
"""

from __future__ import annotations

import contextlib
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_cake_causal_conv1d_det_train_kcp import WIDTH, Y_ATOL, Y_RTOL, _reference
from tests.test_cake_causal_conv1d_det_train_kcp_dist import _free_port, _local
from tests.test_cake_causal_conv1d_det_train_kcp_multi_dist import (
    GRAPH_ROWS,
    ROWS,
    _report,
    _row_id,
    _same,
    _setup,
)
from tests.test_helpers.kcp_collective_shims import CollectiveCounter, RecomputePhase
from tests.test_helpers.cake_causal_conv1d_early_render import apply_to_package

# Test-only: the early render's backward rules predate the required ``slots`` (the package applies no default).
apply_to_package()

DW_ATOL, DW_RTOL = 5e-2, 1e-4
NONDET = {"deterministic": False}


def _ratio(a: torch.Tensor, b: torch.Tensor, scale: float = 1.0) -> float:
    if a.numel() == 0:
        return 0.0
    a, b = a.double(), b.double()
    if not torch.isfinite(a).all():
        return float("inf")
    return float(((a - b).abs() / (scale * DW_ATOL + scale * DW_RTOL * b.abs())).max())


def _ulp_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    """The worst ``|a - b| / (Y_ATOL + Y_RTOL |b|)`` (``<= 1``: within one BF16 ulp)."""

    if a.numel() == 0:
        return 0.0
    a, b = a.double(), b.double()
    if not torch.isfinite(a).all():
        return float("inf")
    return float(((a - b).abs() / (Y_ATOL + Y_RTOL * b.abs())).max())


def _global_inputs(row, T: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``(x_all, w_all, dy_all)`` per tensor of the whole pack, drawn in ``_setup``'s order from its seed (the
    caller checks them against ``_setup``'s tensors)."""

    g = torch.Generator(device="cpu").manual_seed(7815)
    out = []
    for C in row[3]:
        x_all = torch.randn(T, C, generator=g).to(torch.bfloat16)
        w_all = (torch.randn(C, WIDTH, generator=g) * 0.5).contiguous()
        dy_all = torch.randn(T, C, generator=g).to(torch.bfloat16)
        out.append((x_all, w_all, dy_all))
    return out


def _eq(a, b, n: int, scale: float = 1.0) -> bool:
    return all(
        _same(a[i]["y"], b[i]["y"])
        and _same(a[i]["dx"], b[i]["dx"])
        and _ratio(a[i]["dw"], b[i]["dw"], scale) <= 1.0
        for i in range(n)
    )


def _run_rank(rank: int, W: int, row, port: int, result_dir, local_id: int) -> dict:
    from torch.utils.checkpoint import checkpoint

    from flashinfer import cake_causal_conv1d_det_train as conv
    from flashinfer import cake_causal_conv1d_det_train_kcp as det_api
    from flashinfer import cake_causal_conv1d_train_kcp as kcp_api
    from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import (
        kcp as conv_kcp,
    )

    Cs = row[3]
    n = len(Cs)
    device, part, cu, ctx, xs_all, xs, ws, dys = _setup(rank, W, row, port, local_id)

    def leaves():
        return (
            [x.detach().clone().requires_grad_(True) for x in xs],
            [w.detach().clone().requires_grad_(True) for w in ws],
        )

    def layer(multi: bool, det: bool = False):
        def fn(*args):
            a_x, a_w, a_h = args[:n], args[n : 2 * n], args[2 * n :]
            if det:
                return det_api.cake_causal_conv1d_det_multi_kcp(a_x, a_w, ctx)
            if multi:
                return kcp_api.cake_causal_conv1d_multi_kcp(
                    a_x, a_w, ctx, recompute_handle=a_h[0], **NONDET
                )
            return tuple(
                kcp_api.cake_causal_conv1d_kcp(x, w, ctx, recompute_handle=h, **NONDET)
                for x, w, h in zip(a_x, a_w, a_h, strict=True)
            )

        return fn

    def run(
        multi: bool, mode: str, handles=None, *, twice=False, reentrant=False, det=False
    ):
        fn = layer(multi, det)
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
                        [conv_kcp.RecomputeHandleKcp()]
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

    def table(o, r, b):
        return {"orig": o, "recompute": r, "bwd": b}

    flags: dict[str, bool] = {}
    info: dict[str, object] = {
        "n_hist_rows": int(ctx.halo.n_hist_rows),
        "T_r": int(xs[0].shape[0]),
    }
    det, _ = run(True, "direct", det=True)
    s_direct, cs_direct = run(False, "direct")
    m_direct, cm_direct = run(True, "direct")
    m_plain, cm_plain = run(True, "ckpt")
    m_handle, cm_handle = run(True, "ckpt", [conv_kcp.RecomputeHandleKcp()])
    m_plain_re, _ = run(True, "ckpt", reentrant=True)
    m_handle_re, cm_handle_re = run(
        True, "ckpt", [conv_kcp.RecomputeHandleKcp()], reentrant=True
    )
    m_twice, cm_twice = run(True, "ckpt", [conv_kcp.RecomputeHandleKcp()], twice=True)
    m_nest_out, cm_nest_out = run(True, "nested_out", [conv_kcp.RecomputeHandleKcp()])
    m_nest_in, cm_nest_in = run(True, "nested_in", [conv_kcp.RecomputeHandleKcp()])
    info["counts"] = {
        "single_direct": cs_direct, "multi_direct": cm_direct, "multi_plain": cm_plain, "multi_handle": cm_handle,
        "multi_handle_reentrant": cm_handle_re, "multi_twice": cm_twice, "nested_out": cm_nest_out,
        "nested_in": cm_nest_in,
    }  # fmt: skip
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
    flags["y_eq_single_rank"] = all(
        _same(
            m_direct[i]["y"],
            _local(
                part,
                rank,
                cu,
                conv.cake_causal_conv1d_det(xs_all[i].to(device), ws[i], cu_dev),
            ),
        )
        for i in range(n)
    )
    flags["multi_eq_deterministic"] = _eq(m_direct, det, n)
    flags["multi_eq_single_calls"] = _eq(m_direct, s_direct, n)
    for label, got in (
        ("plain", m_plain), ("handle", m_handle), ("plain_reentrant", m_plain_re),
        ("handle_reentrant", m_handle_re), ("nested_out", m_nest_out), ("nested_in", m_nest_in),
    ):  # fmt: skip
        flags[f"{label}_eq_direct"] = _eq(got, m_direct, n)
    flags["twice_accumulates"] = all(
        _same(m_twice[i]["dx"], m_direct[i]["dx"] + m_direct[i]["dx"])
        and _ratio(m_twice[i]["dw"], m_direct[i]["dw"] + m_direct[i]["dw"], 2.0) <= 1.0
        for i in range(n)
    )
    info["worst_dw_ratio_vs_det"] = max(
        _ratio(m_direct[i]["dw"], det[i]["dw"]) for i in range(n)
    )
    # The independent reference per tensor: the global FP64 oracle of the whole pack, the rank's tokens gathered
    # from it; the rank partials' sum (the framework's reduction) against the global FP64 dweight.
    inputs = _global_inputs(row, cu[-1])
    flags["reference_inputs_match_setup"] = all(
        _same(x_all, xs_all[i])
        and _same(w_all.to(device), ws[i])
        and _same(_local(part, rank, cu, dy_all).to(device), dys[i])
        for i, (x_all, w_all, dy_all) in enumerate(inputs)
    )
    worst_y = worst_dx = worst_dw = 0.0
    for i, (x_all, _w_all, dy_all) in enumerate(inputs):
        y_ref, dx_ref, dw_ref = _reference(
            x_all.to(device), ws[i], cu_dev, dy_all.to(device)
        )
        worst_y = max(worst_y, _ulp_ratio(m_direct[i]["y"], _local(part, rank, cu, y_ref)))
        worst_dx = max(
            worst_dx, _ulp_ratio(m_direct[i]["dx"], _local(part, rank, cu, dx_ref))
        )
        total = m_direct[i]["dw"].clone()
        dist.all_reduce(total)
        worst_dw = max(worst_dw, _ratio(total, dw_ref))
    info["vs_fp64"] = {
        "worst_y_ratio_to_one_bf16_ulp": worst_y,
        "worst_dx_ratio_to_one_bf16_ulp": worst_dx,
        "worst_dw_sum_ratio_to_budget": worst_dw,
    }
    flags["y_dx_within_one_bf16_ulp_of_fp64"] = worst_y <= 1.0 and worst_dx <= 1.0
    flags["dw_sum_within_budget_of_fp64"] = worst_dw <= 1.0
    flags["counts"] = (
        cs_direct["calls"] == table(n, 0, n)
        and cm_direct["calls"] == table(1, 0, 1)
        and cm_plain["calls"] == table(1, 1, 1)
        and cm_handle["calls"] == table(1, 0, 1)
        and cm_handle_re["calls"] == table(1, 0, 1)
        and cm_twice["calls"] == table(1, 0, 2)
        and cm_nest_out["calls"] == table(1, 0, 1)
        and cm_nest_in["calls"] == table(1, 1, 1)
    )
    return _report(W, row, flags, info, result_dir, rank, "nondet-multi")


def _run_rank_graph(
    rank: int, W: int, row, port: int, result_dir, local_id: int
) -> dict:
    from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import (
        kcp as conv_kcp,
    )

    device, part, cu, ctx, xs_all, xs, ws, dys = _setup(rank, W, row, port, local_id)
    ref = []
    for x, w, dy in zip(xs, ws, dys, strict=True):
        y, hist = conv_kcp.cake_causal_conv1d_forward_kcp(x, w, ctx, **NONDET)
        dx, dw = conv_kcp.cake_causal_conv1d_backward_kcp(x, w, ctx, dy, hist, **NONDET)
        ref.append((y, hist, dx, dw))
    events, ok, worst = [], True, 0.0
    try:
        for _ in range(4):
            ys, hists = conv_kcp.cake_causal_conv1d_multi_forward_kcp(
                xs, ws, ctx, graph="replay", **NONDET
            )
            events.append(conv_kcp.LAST_CALL["fwd_multi"]["graph_event"])
            dxs, dws = conv_kcp.cake_causal_conv1d_multi_backward_kcp(
                xs, ws, ctx, dys, hists, graph="replay", **NONDET
            )
            events.append(conv_kcp.LAST_CALL["bwd_multi"]["graph_event"])
            torch.cuda.synchronize(device)
            for i, (y, hist, dx, dw) in enumerate(ref):
                ok = (
                    ok
                    and _same(ys[i], y)
                    and _same(hists[i], hist)
                    and _same(dxs[i], dx)
                )
                worst = max(worst, _ratio(dws[i], dw))
    finally:
        torch.cuda.synchronize(device)
        conv_kcp.graph_cache_clear()
    if int(xs[0].shape[0]) == 0:
        expected = ["eager"] * 8
    elif int(ctx.halo.n_hist_rows) == 0:
        expected = ["eager", "eager", "capture", "capture"] + ["replay"] * 4
    else:
        expected = [
            "eager",
            "eager",
            "capture",
            "eager",
            "replay",
            "capture",
            "replay",
            "replay",
        ]
    flags = {
        "graph_y_hist_dx_bitwise": ok,
        "graph_dw_within_budget": worst <= 1.0,
        "graph_events": events == expected,
    }
    info = {"events": events, "worst_dw_ratio": worst, "T_r": int(xs[0].shape[0])}
    return _report(W, row, flags, info, result_dir, rank, "nondet-graph")


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


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import (
        is_cake_causal_conv1d_train_supported,
    )

    return is_cake_causal_conv1d_train_supported(torch.device("cuda"))


def _launch(row, graph: bool) -> None:
    import json

    W = row[1]
    tag = "nondet-graph" if graph else "nondet-multi"
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _supported():
            pytest.skip(
                "the non-deterministic conv package is not rendered or does not support this device"
            )
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
        pytest.skip(
            "the non-deterministic conv package is not rendered or does not support this device"
        )
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    with tempfile.TemporaryDirectory(prefix="cake_conv_kcp_nondet_multi_dist_") as tmp:
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
def test_nondeterministic_multi_rows(row) -> None:
    _launch(row, graph=False)


@pytest.mark.parametrize("row", GRAPH_ROWS, ids=_row_id)
def test_nondeterministic_multi_graph_replay_rows(row) -> None:
    _launch(row, graph=True)
