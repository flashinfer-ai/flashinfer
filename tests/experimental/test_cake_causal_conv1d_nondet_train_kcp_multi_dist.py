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

Multi-GPU rows of the non-deterministic ``cake_causal_conv1d_nondet_kcp_multi`` (the q / k / v convolutions of a
layer with one halo exchange per direction) on the deterministic multi suite's partitions, W = 2 / 4 / 8 over NCCL,
plus W = 16 rows (``-k w16``) on zigzag and per-document zigzag partitions (eager, and graph replay on the
per-document one).  Per row and rank, for every tensor:

- ``y`` and ``dx`` are bitwise the non-deterministic single calls' and the deterministic multi call's; ``y`` is also
  bitwise the single-rank zero-history convolution; each ``dweight`` partial is within the package budget of the
  deterministic one -- direct, plain checkpoint, checkpoint with one handle, ``use_reentrant`` False and True,
  nested checkpoints and ``retain_graph`` (``dx`` exactly twice, ``dweight`` within twice the budget);
- an independent reference per tensor: the rank's ``y`` and ``dx`` (its tokens gathered from the global FP64
  oracle of the whole pack under the KCP SiLU' policy) within one BF16 ulp, and the all-reduced ``dweight``
  partial sum within the package budget of the global FP64 ``dweight``;
- the collective counts per phase of the deterministic suite: multi 1/0/1 direct, 1/1/1 plain checkpoint, 1/0/1
  with a handle; single calls n/0/n;
- the two-launch forward (knobs ``fwd_main`` x ``fwd_overlap``, the same on every rank): ``LAST_CALL["fwd_multi"]``
  names the form the rank ran and every tensor's ``y`` / ``hist`` are bitwise the one-launch single call's
  (``fwd_main`` 0, ``fwd_overlap`` 0: the previous forward) and the deterministic multi forward's;
- ``graph="replay"`` of the multi forward / backward: ``y`` / ``hist`` / ``dx`` bitwise the eager single calls on
  every call, ``dweight`` within budget, the deterministic suite's event sequence; and of the single backward of the
  first tensor (its head program and exchange issue on the side stream inside the capture): ``dx`` bitwise the eager
  call, ``dweight`` within budget, events eager, capture, replay, replay;
- a main-program launch that raises after the fork (single and multi backward) still joins the side stream: the
  error is the injected one and the current stream waits for work queued on the side stream before the call;
- the overlapped backward's tail rows (single and multi backward, eager): every main-program launch passes the
  context's ``tail_row_off`` as its ``skip_row_off`` when the rank receives tail rows (else the all ``-1`` table),
  every tail-row launch runs off the caller's stream (the side stream, or a stream forked from it), and ``dx`` is
  bitwise the autograd calls'.

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

from tests.test_cake_causal_conv1d_det_train_kcp import (
    WIDTH,
    Y_ATOL,
    Y_RTOL,
    _reference,
)
from tests.test_cake_causal_conv1d_det_train_kcp_dist import _free_port, _local
from tests.test_cake_causal_conv1d_det_train_kcp_multi_dist import (
    GRAPH_ROWS,
    ROWS,
    _report,
    _row_id,
    _same,
    _setup,
)
from tests.experimental.test_helpers.kcp_collective_shims import (
    CollectiveCounter,
    RecomputePhase,
)

DW_ATOL, DW_RTOL = 5e-2, 1e-4
# W = 16 (16-way context parallelism) on the deterministic rows' packs, for the layouts any world size partitions;
# the per-document zigzag row also runs under graph replay.
ROWS_W16 = [
    ((3000,), 16, "zigzag", (256, 256, 256)),
    ((2048, 5, 3000, 1), 16, "zigzag_per_document", (384, 384, 384)),
]
GRAPH_ROWS_W16 = [ROWS_W16[1]]


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


def _global_inputs(
    row, T: int
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
    from flashinfer import cake_causal_conv1d_nondet_train_kcp as kcp_api
    from flashinfer.experimental.cake.cake_causal_conv1d_nondet_train import (
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
                return det_api.cake_causal_conv1d_det_kcp_multi(a_x, a_w, ctx)
            if multi:
                return kcp_api.cake_causal_conv1d_nondet_kcp_multi(
                    a_x, a_w, ctx, recompute_handle=a_h[0]
                )
            return tuple(
                kcp_api.cake_causal_conv1d_nondet_kcp(x, w, ctx, recompute_handle=h)
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
                        [conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()]
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
    m_handle, cm_handle = run(
        True, "ckpt", [conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()]
    )
    m_plain_re, _ = run(True, "ckpt", reentrant=True)
    m_handle_re, cm_handle_re = run(
        True,
        "ckpt",
        [conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()],
        reentrant=True,
    )
    m_twice, cm_twice = run(
        True, "ckpt", [conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()], twice=True
    )
    m_nest_out, cm_nest_out = run(
        True, "nested_out", [conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()]
    )
    m_nest_in, cm_nest_in = run(
        True, "nested_in", [conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()]
    )
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
        worst_y = max(
            worst_y, _ulp_ratio(m_direct[i]["y"], _local(part, rank, cu, y_ref))
        )
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
    # The first nondet call of the context (the single calls) also runs the rank-agreement check of #8946: one
    # all_gather_into_tensor per phase; every later call issues only the halo exchanges.
    # The overlapped backward's tail rows: the main programs leave them to the tail-row programs on the side stream.
    y0, hist0 = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(xs[0], ws[0], ctx)
    dx0 = []
    flags["skip_path_single"] = _skip_path(
        conv_kcp,
        ctx,
        device,
        lambda: dx0.append(
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
                xs[0], ws[0], ctx, dys[0], hist0
            )[0]
        ),
    ) and _same(dx0[0], s_direct[0]["dx"])
    _, hists = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(xs, ws, ctx)
    dxs = []
    flags["skip_path_multi"] = _skip_path(
        conv_kcp,
        ctx,
        device,
        lambda: dxs.append(
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(
                xs, ws, ctx, dys, hists
            )[0]
        ),
    ) and all(_same(dxs[0][i], m_direct[i]["dx"]) for i in range(n))
    flags["counts"] = (
        cs_direct["calls"] == table(n + 1, 0, n + 1)
        and cm_direct["calls"] == table(1, 0, 1)
        and cm_plain["calls"] == table(1, 1, 1)
        and cm_handle["calls"] == table(1, 0, 1)
        and cm_handle_re["calls"] == table(1, 0, 1)
        and cm_twice["calls"] == table(1, 0, 2)
        and cm_nest_out["calls"] == table(1, 0, 1)
        and cm_nest_in["calls"] == table(1, 1, 1)
    )
    # The two-launch forward: every (fwd_main, fwd_overlap) form, the same on every rank, records its form in
    # LAST_CALL["fwd_multi"] (the first tensor's main program before the exchange's wait on a rank with tokens; the
    # history-row program when the rank receives history rows and the form has two launches) and gives every
    # tensor's y / hist bitwise the one-launch single call's and the deterministic multi forward's.
    from tests.cake.test_helpers.cake_package import cake_package

    det_kcp = cake_package("cake_causal_conv1d_det_train", "kcp")
    n_hist = int(ctx.halo.n_hist_rows)
    det_ys, det_hists = det_kcp.cake_causal_conv1d_det_train_kcp_multi_forward(
        xs, ws, ctx
    )
    conv_kcp._set_test_knobs({"fwd_main": 0, "fwd_overlap": 0})
    try:
        refs = [
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(x, w, ctx)
            for x, w in zip(xs, ws, strict=True)
        ]
        torch.cuda.synchronize(device)
    finally:
        conv_kcp._set_test_knobs(None)
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
    fwd_bitwise = all(
        _same(y, det_ys[i]) and _same(hist, det_hists[i])
        for i, (y, hist) in enumerate(refs)
    )
    fwd_recorded = True
    for fwd_main in (0, 1):
        for fwd_overlap in (0, 1):
            conv_kcp._set_test_knobs({"fwd_main": fwd_main, "fwd_overlap": fwd_overlap})
            try:
                for _ in range(2):
                    ys, hists = (
                        conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(
                            xs, ws, ctx
                        )
                    )
                    torch.cuda.synchronize(device)
                    want = (
                        {
                            "main": "sr" if fwd_main else "f",
                            "form": "pair" if fwd_overlap else "wait",
                            "repair": bool(fwd_main or fwd_overlap) and n_hist > 0,
                        }
                        if xs[0].shape[0]
                        else {"main": None, "form": "wait", "repair": False}
                    )
                    fwd_recorded = (
                        fwd_recorded and conv_kcp.LAST_CALL["fwd_multi"] == want
                    )
                    fwd_bitwise = fwd_bitwise and all(
                        _same(ys[i], y) and _same(hists[i], hist)
                        for i, (y, hist) in enumerate(refs)
                    )
            finally:
                conv_kcp._set_test_knobs(None)
                conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
    flags["fwd_forms_bitwise"] = fwd_bitwise
    flags["fwd_forms_recorded"] = fwd_recorded
    return _report(W, row, flags, info, result_dir, rank, "nondet-multi")


def _skip_path(conv_kcp, ctx, device, call) -> bool:
    """``call`` (a backward) with its main-program and tail-row launches recorded: True when every main-program launch
    passed the context's ``tail_row_off`` as ``skip_row_off`` if the rank receives tail rows (else the all ``-1``
    table) and the tail-row launches, one per tensor when it receives them and none otherwise, ran off the caller's
    stream; or, for a call that takes no side stream (one rank) or a rank without tokens, when it launched neither."""
    tokens = int(ctx.T_r) > 0
    if conv_kcp._side_stream(ctx, device, True) is None or not tokens:
        real = conv_kcp._launch_dyn
        hits: list[int] = []

        def counted(d, **kwargs):
            hits.append(1)
            return real(d, **kwargs)

        conv_kcp._launch_dyn = counted
        try:
            call()
            torch.cuda.synchronize(device)
        finally:
            conv_kcp._launch_dyn = real
        return tokens or not hits
    caller = torch.cuda.current_stream(device).cuda_stream
    tail = ctx.halo_tables["tail_row_off"]
    receives = int(ctx.halo.n_tail_rows) > 0
    skips: list[bool] = []
    streams: list[int] = []
    real_dyn, real_halo = conv_kcp._launch_dyn, conv_kcp._launch_halo

    def dyn(d, **kwargs):
        skip = kwargs["skip"]
        skips.append(skip is tail if receives else bool((skip < 0).all()))
        return real_dyn(d, **kwargs)

    def halo(d, **kwargs):
        if d["halo"] is not None:
            streams.append(torch.cuda.current_stream(device).cuda_stream)
        return real_halo(d, **kwargs)

    conv_kcp._launch_dyn, conv_kcp._launch_halo = dyn, halo
    try:
        call()
        torch.cuda.synchronize(device)
    finally:
        conv_kcp._launch_dyn, conv_kcp._launch_halo = real_dyn, real_halo
    if not skips or not all(skips):
        return False
    if not receives:
        return not streams
    return len(streams) == len(skips) and caller not in streams


def _join_after_failed_launch(conv_kcp, ctx, device, call) -> bool:
    """``call`` with the main-program launch raising, while the side stream sleeps from before the call: True when
    the call raised the injected error and an event recorded on the current stream right after it is still pending
    (the current stream waits for the side stream), or when the call launched no main program (no error)."""
    if conv_kcp._side_stream(ctx, device, True) is None:
        return True
    side = conv_kcp._device_stream(device, 0)
    torch.cuda.synchronize(device)
    hits = []

    def boom(*args, **kwargs):
        hits.append(1)
        raise RuntimeError("injected main-launch failure")

    real = conv_kcp._launch_dyn
    conv_kcp._launch_dyn = boom
    err = None
    try:
        with torch.cuda.stream(side):
            torch.cuda._sleep(1 << 30)
        call()
    except RuntimeError as exc:
        err = exc
    finally:
        conv_kcp._launch_dyn = real
    done = torch.cuda.Event()
    done.record(torch.cuda.current_stream(device))
    pending = not done.query()
    torch.cuda.synchronize(device)
    if not hits:
        return err is None
    return err is not None and "injected main-launch failure" in str(err) and pending


def _run_rank_graph(
    rank: int, W: int, row, port: int, result_dir, local_id: int
) -> dict:
    from flashinfer.experimental.cake.cake_causal_conv1d_nondet_train import (
        kcp as conv_kcp,
    )

    device, part, cu, ctx, xs_all, xs, ws, dys = _setup(rank, W, row, port, local_id)
    ref = []
    for x, w, dy in zip(xs, ws, dys, strict=True):
        y, hist = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(x, w, ctx)
        dx, dw = conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
            x, w, ctx, dy, hist
        )
        ref.append((y, hist, dx, dw))
    events, ok, worst = [], True, 0.0
    single_events, single_ok = [], True
    try:
        for _ in range(4):
            ys, hists = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(
                xs, ws, ctx, graph="replay"
            )
            events.append(conv_kcp.LAST_CALL["fwd_multi"]["graph_event"])
            dxs, dws = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(
                xs, ws, ctx, dys, hists, graph="replay"
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
        _, hist0, dx0, dw0 = ref[0]
        for _ in range(4):
            dx, dw = conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
                xs[0], ws[0], ctx, dys[0], hist0, graph="replay"
            )
            single_events.append(conv_kcp.LAST_CALL["bwd"]["graph_event"])
            torch.cuda.synchronize(device)
            single_ok = single_ok and _same(dx, dx0)
            worst = max(worst, _ratio(dw, dw0))
        join_single = _join_after_failed_launch(
            conv_kcp,
            ctx,
            device,
            lambda: conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
                xs[0], ws[0], ctx, dys[0], hist0
            ),
        )
        join_multi = _join_after_failed_launch(
            conv_kcp,
            ctx,
            device,
            lambda: conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(
                xs, ws, ctx, dys, hists
            ),
        )
    finally:
        torch.cuda.synchronize(device)
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()
    if int(xs[0].shape[0]) == 0:
        single_expected = ["eager"] * 4
    else:
        single_expected = ["eager", "capture", "replay", "replay"]
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
        "graph_single_dx_bitwise": single_ok,
        "graph_single_events": single_events == single_expected,
        "launch_error_joins_side_single": join_single,
        "launch_error_joins_side_multi": join_multi,
    }
    info = {
        "events": events,
        "single_events": single_events,
        "worst_dw_ratio": worst,
        "T_r": int(xs[0].shape[0]),
    }
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
    from flashinfer.experimental.cake.cake_causal_conv1d_nondet_train import (
        is_cake_causal_conv1d_nondet_train_supported,
    )

    return is_cake_causal_conv1d_nondet_train_supported(torch.device("cuda"))


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


@pytest.mark.parametrize("row", ROWS + ROWS_W16, ids=_row_id)
def test_nondeterministic_multi_rows(row) -> None:
    _launch(row, graph=False)


@pytest.mark.parametrize("row", GRAPH_ROWS + GRAPH_ROWS_W16, ids=_row_id)
def test_nondeterministic_multi_graph_replay_rows(row) -> None:
    _launch(row, graph=True)
