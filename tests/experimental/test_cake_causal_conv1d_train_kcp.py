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

Single-GPU tests of the non-deterministic KCP convolution (``cake_causal_conv1d_kcp(..., deterministic=False)``): one
process, world size 1, no process group (the halo exchange of one rank is the identity), on the deterministic KCP
suite's rows, contexts and FP64 oracle.

(a) ``y`` and ``hist`` are bitwise the deterministic KCP forward's (the same programs); ``dx`` is bitwise the
    deterministic KCP backward's (the same per-thread arithmetic and fold); the FP32 ``dweight`` partial is within
    the package budget (``atol 5e-2 + rtol 1e-4``) of the FP64 oracle and of the deterministic partial.
(b) ``REPEATS`` launches of the same inputs: ``dx`` bitwise every time, ``dweight`` within budget every time (the worst
    case is recorded); the package's accumulator workspace reads back all zero after every launch.
(c) The autograd entry equals the direct calls (``y`` / ``dx`` bitwise), activation checkpointing with and without a
    recompute handle (2 x forward + backward; one exchange per direction, none in a recompute with a handle),
    gradient accumulation, graph replay (bitwise ``y`` / ``hist`` / ``dx``), the legacy channel-tile plan, the side
    kernel of FlashInfer issues #7038 / #7548 (the backward is an ordinary launch: no stall beside a resident kernel)
    and torch's deterministic mode (the backward alerts).

Every test skips until the exporter has rendered the non-deterministic package.
"""

from __future__ import annotations

import statistics

import pytest
import torch

from flashinfer.cake_causal_conv1d_det_train_backend import kcp as det_kcp
from flashinfer.experimental.cake import cake_causal_conv1d_train_backend as pkg
from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import (
    kcp as conv_kcp,
)
from tests.test_cake_causal_conv1d_det_train_kcp import (
    DW_ATOL,
    DW_RTOL,
    ROWS,
    WIDTH,
    Y_ATOL,
    Y_RTOL,
    _assert_within,
    _bits,
    _context,
    _ExchangeCounter,
    _problem,
    _reference,
)

REPEATS = 20
NONDET = {"deterministic": False}

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not pkg.is_cake_causal_conv1d_train_supported(torch.device("cuda")),
    reason=(
        "the non-deterministic cake_causal_conv1d_train package needs a supported CUDA device (SM100 / SM103 / "
        "SM107) and its rendered build.json / csrc (TODO-EXPORT until the exporter renders it)"
    ),
)


def _fwd(x, w, ctx, **kw):
    return conv_kcp.cake_causal_conv1d_forward_kcp(x, w, ctx, **kw, **NONDET)


def _bwd(x, w, ctx, dy, hist, **kw):
    return conv_kcp.cake_causal_conv1d_backward_kcp(x, w, ctx, dy, hist, **kw, **NONDET)


def _stream_workspace() -> torch.Tensor | None:
    device = torch.device("cuda", torch.cuda.current_device())
    key = (device.index, int(torch.cuda.current_stream(device).cuda_stream))
    return pkg._BACKWARD_WORKSPACES.get(key)


def _dw_within(actual, expected, label, *, scale=1.0):
    return _assert_within(
        actual, expected, atol=scale * DW_ATOL, rtol=scale * DW_RTOL, label=label
    )


def _same(a, b, *, scale=1.0) -> bool:
    """``y`` / ``dx`` bitwise, ``dweight`` within ``scale`` x the budget."""

    if not all(torch.equal(_bits(a[n]), _bits(b[n])) for n in ("y", "dx")):
        return False
    _dw_within(a["dw"], b["dw"], "dweight", scale=scale)
    return True


@pytest.mark.parametrize("name,lens,cuts,C", ROWS, ids=[r[0] for r in ROWS])
def test_forward_is_the_deterministic_forward(name, lens, cuts, C):
    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    y, hist = _fwd(p.x_local, p.w, p.ctx)
    y_det, hist_det = det_kcp.cake_causal_conv1d_det_forward_kcp(p.x_local, p.w, p.ctx)
    assert torch.equal(_bits(y), _bits(y_det)) and torch.equal(
        _bits(hist), _bits(hist_det)
    )
    y_ref, _, _ = _reference(p.x, p.w, p.cu, p.dy)
    _assert_within(y, y_ref[p.l2g], atol=Y_ATOL, rtol=Y_RTOL, label="y vs FP64 oracle")


@pytest.mark.parametrize("name,lens,cuts,C", ROWS, ids=[r[0] for r in ROWS])
def test_backward_repeats_dx_bitwise_dweight_within_budget(
    name, lens, cuts, C, record_property
):
    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    _, hist = _fwd(p.x_local, p.w, p.ctx)
    _, dx_ref, dw_ref = _reference(p.x, p.w, p.cu, p.dy)
    _, hist_det = det_kcp.cake_causal_conv1d_det_forward_kcp(p.x_local, p.w, p.ctx)
    dx_det, dw_det = det_kcp.cake_causal_conv1d_det_backward_kcp(
        p.x_local, p.w, p.ctx, p.dy_local, hist_det
    )
    worst_ref = worst_det = 0.0
    patterns = set()
    for repeat in range(REPEATS):
        dx, dw = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
        torch.cuda.synchronize()
        assert torch.equal(_bits(dx), _bits(dx_det)), (
            f"dx differs from the deterministic KCP backward ({repeat})"
        )
        stats = _dw_within(dw, dw_ref, f"dweight vs FP64 oracle ({repeat})")
        worst_ref = max(worst_ref, stats["max_ratio_to_bound"])
        stats = _dw_within(dw, dw_det, f"dweight vs deterministic ({repeat})")
        worst_det = max(worst_det, stats["max_ratio_to_bound"])
        patterns.add(bytes(dw.contiguous().view(torch.uint8).cpu().numpy()))
        held = _stream_workspace()
        if held is not None:
            assert int(torch.count_nonzero(held)) == 0, (
                f"accumulators not zero after repeat {repeat}"
            )
    _assert_within(
        dx_det, dx_ref[p.l2g], atol=Y_ATOL, rtol=Y_RTOL, label="dx vs FP64 oracle"
    )
    print(f"{name}: {REPEATS} launches, worst dweight |err|/budget vs FP64 {worst_ref:.3f}, vs det {worst_det:.3f}, "
          f"{len(patterns)} distinct dweight bit patterns")  # fmt: skip
    record_property("repeats", REPEATS)
    record_property("worst_dweight_ratio_vs_fp64", worst_ref)
    record_property("worst_dweight_ratio_vs_det", worst_det)
    record_property("distinct_dweight_patterns", len(patterns))


@pytest.mark.parametrize("name,lens,cuts,C", ROWS[:3], ids=[r[0] for r in ROWS[:3]])
def test_autograd_function_matches_the_direct_calls(name, lens, cuts, C):
    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    runs = []
    for _ in range(2):
        x = p.x_local.clone().requires_grad_(True)
        w = p.w.clone().requires_grad_(True)
        y = conv_kcp.cake_causal_conv1d_kcp(x, w, p.ctx, **NONDET)
        y.backward(p.dy_local)
        runs.append({"y": y.detach(), "dx": x.grad, "dw": w.grad})
    y_direct, hist = _fwd(p.x_local, p.w, p.ctx)
    dx_direct, dw_direct = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
    assert _same(runs[0], runs[1])
    assert _same(runs[0], {"y": y_direct, "dx": dx_direct, "dw": dw_direct})


def test_empty_rank_returns_empty_outputs():
    device = torch.device("cuda")
    C = 512
    ctx, _, _, _ = _context((0,), (0,), device)
    x = torch.empty(0, C, dtype=torch.bfloat16, device=device)
    w = torch.randn(C, WIDTH, device=device)
    y, hist = _fwd(x, w, ctx)
    dx, dw = _bwd(x, w, ctx, torch.empty_like(x), hist)
    assert y.shape == (0, C) and dx.shape == (0, C)
    assert torch.equal(dw, torch.zeros_like(dw))


def test_rejects_mismatched_context_and_keeps_the_kcp_contract():
    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    with pytest.raises(ValueError):
        _fwd(p.x_local[:-1].contiguous(), p.w, p.ctx)
    with pytest.raises(ValueError, match="weight must be torch.float32"):
        _fwd(p.x_local, p.w.to(torch.bfloat16), p.ctx)
    with pytest.raises(TypeError):
        conv_kcp.cake_causal_conv1d_forward_kcp(
            p.x_local, p.w, p.ctx, bias=torch.zeros(p.C, device=device)
        )


def test_kcp_plans_with_the_legacy_tile_rule(monkeypatch):
    calls = []
    original = conv_kcp.plan

    def recording_plan(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(conv_kcp, "plan", recording_plan)
    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 1280, device)
    y, hist = _fwd(p.x_local, p.w, p.ctx)
    dx, dw = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
    assert len(calls) == 2 and all(call.get("legacy") is True for call in calls), calls
    monkeypatch.undo()
    _, hist_det = det_kcp.cake_causal_conv1d_det_forward_kcp(p.x_local, p.w, p.ctx)
    dx_det, _ = det_kcp.cake_causal_conv1d_det_backward_kcp(
        p.x_local, p.w, p.ctx, p.dy_local, hist_det
    )
    assert torch.equal(dx, dx_det) and torch.isfinite(dw).all()


@pytest.mark.parametrize("form", ["knob", "keyword"])
def test_graph_replay_is_bitwise_for_y_and_dx(monkeypatch, form) -> None:
    """Graph replay of the KCP calls: the deterministic suite's event sequence; ``y`` / ``hist`` / ``dx`` bitwise the
    eager results on every call, ``dweight`` within budget (the captured launch zeroes nothing itself: the
    accumulators it replays on are left zero by every launch)."""

    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    x, w, ctx = p.x_local, p.w, p.ctx
    dy = torch.randn(x.shape, device=device).to(torch.bfloat16)
    y0, hist0 = _fwd(x, w, ctx)
    dx0, dw0 = _bwd(x, w, ctx, dy, hist0)
    kw = {}
    if form == "knob":
        monkeypatch.setenv(conv_kcp.TEST_KNOBS_ENV, "1")
        conv_kcp._set_test_knobs({"graph": "replay"})
    else:
        kw = {"graph": "replay"}
    try:
        events = []
        for _ in range(4):
            y, hist = _fwd(x, w, ctx, **kw)
            events.append(conv_kcp.LAST_CALL["fwd"]["graph_event"])
            assert torch.equal(y, y0) and torch.equal(hist, hist0)
            dx, dw = _bwd(x, w, ctx, dy, hist, **kw)
            events.append(conv_kcp.LAST_CALL["bwd"]["graph_event"])
            torch.cuda.synchronize()
            assert torch.equal(dx, dx0)
            _dw_within(dw, dw0, "replayed dweight")
        assert events == [
            "eager",
            "eager",
            "capture",
            "eager",
            "replay",
            "capture",
            "replay",
            "replay",
        ], events
        assert conv_kcp.GRAPH_STATS["fallback"] == 0
    finally:
        conv_kcp._set_test_knobs(None)
        conv_kcp._GRAPHS.clear()


# ----------------------------------------------------------------------------------------------------------------
# Checkpointing and the recompute handle (#7815 / #7818).


def _ckpt_run(p, *, handle=None, reentrant=False, ckpt=True, counter=None, twice=False):
    import contextlib

    from torch.utils.checkpoint import checkpoint

    from flashinfer import cake_causal_conv1d_train_kcp as facade

    x = p.x_local.clone().requires_grad_(True)
    w = p.w.clone().requires_grad_(True)

    def fn(x_, w_, h_):
        return facade.cake_causal_conv1d_kcp(
            x_, w_, p.ctx, recompute_handle=h_, deterministic=False
        )

    if not ckpt:
        y = fn(x, w, handle)
    elif reentrant:
        y = checkpoint(fn, x, w, handle, use_reentrant=True)
    else:
        cm = counter.recompute_cm if counter is not None else contextlib.nullcontext
        y = checkpoint(
            fn,
            x,
            w,
            handle,
            use_reentrant=False,
            context_fn=lambda: (contextlib.nullcontext(), cm()),
        )
    if counter is not None:
        counter.phase = "bwd"
    y.backward(p.dy_local, retain_graph=twice)
    if twice:
        y.backward(p.dy_local)
    return {"y": y.detach(), "dx": x.grad, "dw": w.grad}


@pytest.mark.parametrize("name,lens,cuts,C", ROWS, ids=[r[0] for r in ROWS])
@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_checkpoint_and_recompute_handle(
    monkeypatch, name, lens, cuts, C, reentrant
) -> None:
    """Checkpointed (2 x forward + backward) with and without a recompute handle: ``y`` / ``dx`` bitwise the direct
    pass's and the deterministic route's, ``dweight`` within budget; the exchange counts of the deterministic suite
    (one per direction, none in a recompute with a handle)."""

    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    direct = _ckpt_run(p, ckpt=False)
    plain_cc = _ExchangeCounter(monkeypatch, conv_kcp)
    plain = _ckpt_run(p, reentrant=reentrant, counter=plain_cc)
    monkeypatch.undo()
    handle = conv_kcp.RecomputeHandleKcp()
    cc = _ExchangeCounter(monkeypatch, conv_kcp)
    conv_kcp.LAST_CALL.pop("recompute_handle", None)
    got = _ckpt_run(p, handle=handle, reentrant=reentrant, counter=cc)
    monkeypatch.undo()
    assert conv_kcp.LAST_CALL["recompute_handle"] == {"fwd": "replay"}
    assert _same(plain, direct) and _same(got, plain)
    _, hist_det = det_kcp.cake_causal_conv1d_det_forward_kcp(p.x_local, p.w, p.ctx)
    dx_det, dw_det = det_kcp.cake_causal_conv1d_det_backward_kcp(
        p.x_local, p.w, p.ctx, p.dy_local, hist_det
    )
    assert torch.equal(_bits(got["dx"]), _bits(dx_det))
    _dw_within(got["dw"], dw_det, "checkpointed dweight vs deterministic")
    if reentrant:
        assert plain_cc.table() == {"orig": 1, "recompute": 0, "bwd": 2}
        assert cc.table() == {"orig": 1, "recompute": 0, "bwd": 1}
    else:
        assert plain_cc.table() == {"orig": 1, "recompute": 1, "bwd": 1}
        assert cc.table() == {"orig": 1, "recompute": 0, "bwd": 1}
    assert handle.nbytes() == int(p.ctx.halo.n_hist_rows) * C * 2
    handle.clear()


def test_gradient_accumulation_and_retain_graph(monkeypatch) -> None:
    """``retain_graph`` + a second backward: ``dx`` exactly twice (one exact BF16 doubling of the same bits),
    ``dweight`` twice within twice the budget; the recompute exchanges nothing."""

    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    direct = _ckpt_run(p, ckpt=False)
    cc = _ExchangeCounter(monkeypatch, conv_kcp)
    twice = _ckpt_run(p, handle=conv_kcp.RecomputeHandleKcp(), counter=cc, twice=True)
    assert cc.table() == {"orig": 1, "recompute": 0, "bwd": 2}
    assert torch.equal(_bits(twice["dx"]), _bits(direct["dx"] + direct["dx"]))
    _dw_within(
        twice["dw"], direct["dw"] + direct["dw"], "accumulated dweight", scale=2.0
    )


def test_recompute_handle_misuse_raises() -> None:
    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    handle = conv_kcp.RecomputeHandleKcp()
    conv_kcp.cake_causal_conv1d_kcp(
        p.x_local, p.w, p.ctx, recompute_handle=handle, **NONDET
    )
    assert conv_kcp.LAST_CALL["recompute_handle"] == {"fwd": "capture"}
    with pytest.raises(RuntimeError, match="per call and micro-step"):
        conv_kcp.cake_causal_conv1d_kcp(
            p.x_local, p.w, p.ctx, recompute_handle=handle, **NONDET
        )
    with pytest.raises(ValueError, match="graph='off'"):
        conv_kcp.cake_causal_conv1d_kcp(
            p.x_local,
            p.w,
            p.ctx,
            graph="replay",
            recompute_handle=conv_kcp.RecomputeHandleKcp(),
            **NONDET,
        )
    with pytest.raises(TypeError, match="entries"):
        conv_kcp.cake_causal_conv1d_kcp(
            p.x_local, p.w, p.ctx, recompute_handle=object(), **NONDET
        )


# ----------------------------------------------------------------------------------------------------------------
# Streams, torch's deterministic mode.


def test_backward_keeps_its_speed_beside_a_resident_side_kernel(
    record_property,
) -> None:
    """FlashInfer issues #7038 / #7548: the main KCP backward is an ordinary launch, so one 1-thread kernel resident
    on another stream does not stall it (``busy <= 2 x alone``, medians of 7 after 3 warm-ups); ``dx`` is bitwise
    unchanged beside it."""

    device = torch.device("cuda")
    p = _problem((150000, 111393), (1, 2, 149997, 111393), 512, device)
    _, hist = _fwd(p.x_local, p.w, p.ctx)
    side = torch.cuda.Stream(device=device)
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    out = {}

    def timed(with_side_kernel: bool) -> float:
        torch.cuda.synchronize(device)
        if with_side_kernel:
            with torch.cuda.stream(side):
                torch.cuda._sleep(10_000_000)
        torch.cuda._sleep(1_000_000)
        start.record()
        out["dx"], out["dw"] = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    for _ in range(3):
        timed(False)
    alone = statistics.median(timed(False) for _ in range(7))
    dx_alone = out["dx"].clone()
    busy = statistics.median(timed(True) for _ in range(7))
    print(
        f"KCP backward alone {alone:.3f} ms, beside a resident kernel {busy:.3f} ms ({busy / alone:.2f}x)"
    )
    record_property("kcp_backward_alone_ms", f"{alone:.4f}")
    record_property("kcp_backward_beside_side_kernel_ms", f"{busy:.4f}")
    assert torch.equal(out["dx"], dx_alone)
    assert busy <= 2.0 * alone, (
        f"the KCP backward stalled beside a resident kernel: {busy:.3f} vs {alone:.3f} ms"
    )


def test_torch_deterministic_mode_alerts_on_the_backward() -> None:
    device = torch.device("cuda")
    p = _problem((65,), (64, 1), 256, device)
    y0, hist = _fwd(p.x_local, p.w, p.ctx)
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        y, _ = _fwd(p.x_local, p.w, p.ctx)
        assert torch.equal(y, y0)
        with pytest.raises(RuntimeError, match="cake_causal_conv1d_backward_kcp"):
            _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
        x = p.x_local.clone().requires_grad_(True)
        y = conv_kcp.cake_causal_conv1d_kcp(x, p.w, p.ctx, **NONDET)
        with pytest.raises(RuntimeError, match="cake_causal_conv1d_backward_kcp"):
            y.backward(p.dy_local)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(UserWarning, match="cake_causal_conv1d_backward_kcp"):
            _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
