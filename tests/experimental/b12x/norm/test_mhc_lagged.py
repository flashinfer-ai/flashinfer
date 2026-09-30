"""Prepared lagged mHC rounding, coefficient propagation and graph replay."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from b12x.norm import mhc
from b12x.testing.mhc import make_inputs as _make_inputs, pre_reference as _mhc_pre_reference, post_reference as _mhc_post_reference
from ..conftest import require_b12x as require_sm120
from ._mhc import prepare, bind, prepare_collapse, mhc_session

def _lagged_reference(residual, fn, scale, bias, incoming, weight):
    flat = residual.flatten(1).float()
    projection = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + 1e-20
    )
    predicted = torch.sigmoid(projection[:, :4] * scale[0] + bias[:4]) + 1e-6
    _, post, comb = _mhc_pre_reference(
        residual, fn, scale, bias, rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20
    )
    collapsed = (incoming.unsqueeze(-1) * residual.float()).sum(1).bfloat16()
    y = collapsed.float()
    if weight is not None:
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-20) * weight.float()
    return post, comb, y.bfloat16(), predicted


@pytest.mark.parametrize("hidden_size", [4096, 5120])
def test_mhc_lagged_multisublayer_propagation(hidden_size, mhc_session):
    device = require_sm120()
    tokens = 3
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens, hidden_size=hidden_size, seed=92150, device=device
    )
    weight = torch.linspace(0.5, 1.5, hidden_size, device=device).bfloat16()
    incoming = torch.zeros((tokens, 4), device=device)
    incoming[:, 0] = 1
    prev_post = prev_comb = None
    for layer in range(3):
        predicted = torch.full_like(incoming, float("nan"))
        options = dict(pre_mix=incoming, pre_out=predicted, rms_eps=1e-20,
                       hc_eps=1e-6, sinkhorn_iters=20, norm_weight=weight, norm_eps=1e-20)
        args = (residual, fn, scale, bias) if layer == 0 else (x, residual, prev_post, prev_comb, fn, scale, bias)
        plan = prepare(mhc_session, "pre" if layer == 0 else "post_pre", args, options, output_mode="functional")
        if layer == 0:
            actual = mhc.run_pre(
                residual, fn, scale, bias, pre_mix=incoming, pre_out=predicted, plan=plan,
                rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
                norm_weight=weight, norm_eps=1e-20,
            )
            current = residual
        else:
            current = _mhc_post_reference(x, residual, prev_post, prev_comb)
            actual = mhc.run_post_pre(
                x, residual, prev_post, prev_comb, fn, scale, bias,
                pre_mix=incoming, pre_out=predicted, plan=plan, rms_eps=1e-20,
                hc_eps=1e-6, sinkhorn_iters=20, norm_weight=weight, norm_eps=1e-20,
            )
        torch.testing.assert_close(actual[0], current, rtol=0, atol=0.008)
        expected = _lagged_reference(actual[0], fn, scale, bias, incoming, weight)
        for got, want in zip((*actual[1:], predicted), expected, strict=True):
            torch.testing.assert_close(got, want, rtol=2e-5, atol=0.008 if got.dtype == torch.bfloat16 else 4e-5)
        residual, prev_post, prev_comb, x = actual
        incoming = predicted
        # Distinct sublayer projections prevent accidentally carrying a stale mix.
        fn = -fn
        bias = bias.roll(4)


def test_mhc_lagged_rounded_variance_and_ownership(mhc_session):
    device = require_sm120()
    hidden_size = 5120
    residual, _, fn, scale, bias = _make_inputs(
        tokens=1, hidden_size=hidden_size, seed=92151, device=device
    )
    residual[:, 0] = 1
    residual[:, 1] = 1
    residual[:, 1, 4096:] = 1.0078125
    incoming = torch.tensor([[1.0, -0.999, 0.0, 0.0]], device=device)
    predicted = torch.empty_like(incoming)
    weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
    kwargs = dict(rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20)
    options = dict(kwargs, pre_mix=incoming, pre_out=predicted, norm_weight=weight, norm_eps=1e-20)
    plan = prepare(mhc_session, "pre", (residual, fn, scale, bias), options, output_mode="functional")
    provided = prepare(mhc_session, "pre", (residual, fn, scale, bias), options)
    actual = mhc.run_pre(
        residual, fn, scale, bias, pre_mix=incoming, pre_out=predicted,
        norm_weight=weight, norm_eps=1e-20, plan=plan, **kwargs,
    )
    expected = _lagged_reference(residual, fn, scale, bias, incoming, weight)
    torch.testing.assert_close(actual[-1], expected[2], rtol=0, atol=0.001)
    with pytest.raises(ValueError, match="supplied together"):
        mhc.run_pre(residual, fn, scale, bias, pre_mix=incoming, plan=plan, **kwargs)
    with pytest.raises(ValueError, match="must not alias"):
        mhc.run_pre(residual, fn, scale, bias, pre_mix=incoming, pre_out=incoming, norm_weight=weight, norm_eps=1e-20, plan=plan, **kwargs)
    with pytest.raises(ValueError, match="must not alias"):
        mhc.run_pre(
            residual, fn, scale, bias, pre_mix=incoming, pre_out=predicted,
            post_out=predicted, norm_weight=weight, norm_eps=1e-20, plan=provided, **kwargs,
        )
    binding = bind(provided, pre_out=predicted)
    with pytest.raises(ValueError, match="binding owns scratch and output buffers"):
        mhc.run_pre(
            residual, fn, scale, bias, pre_mix=incoming, pre_out=predicted,
            binding=binding, norm_weight=weight, norm_eps=1e-20, **kwargs,
        )


@pytest.mark.parametrize(
    ("phase", "capacity", "fuse_norm"),
    [("pre", 17, False), ("pre", 17, True), ("pre", 389, True),
     ("post_pre", 17, True), ("post_pre", 389, True)],
)
def test_mhc_lagged_frozen_multilive_graph(phase, capacity, fuse_norm, mhc_session):

    device = require_sm120()
    hidden_size = 5120
    residual, x, fn, scale, bias = _make_inputs(
        tokens=capacity, hidden_size=hidden_size, seed=92152, device=device
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual, fn, scale, bias, rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20
    )
    prev_post, prev_comb = prev_post.contiguous(), prev_comb.contiguous()
    weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device) if fuse_norm else None
    incoming = torch.zeros((capacity, 4), device=device)
    incoming[:, 0] = 1
    predicted = torch.empty_like(incoming)
    options = dict(pre_mix=incoming, norm_weight=weight, norm_eps=1e-20,
                   rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20)
    args = (residual, fn, scale, bias) if phase == "pre" else (x, residual, prev_post, prev_comb, fn, scale, bias)
    plan = prepare(mhc_session, phase, args, options)
    binding = bind(plan, pre_out=predicted)

    def run(live):
        if phase == "pre":
            return mhc.run_pre(
                residual[:live], fn, scale, bias, binding=binding,
                pre_mix=incoming[:live], norm_weight=weight, norm_eps=1e-20,
                rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
            )
        return mhc.run_post_pre(
            x[:live], residual[:live], prev_post[:live], prev_comb[:live],
            fn, scale, bias, binding=binding, pre_mix=incoming[:live],
            norm_weight=weight, norm_eps=1e-20, rms_eps=1e-20,
            hc_eps=1e-6, sinkhorn_iters=20,
        )

    run(capacity)
    torch.cuda.synchronize(device)
    mhc_session.freeze()
    for live in (3, capacity - 1, capacity):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run(live)
        pointers = tuple(t.data_ptr() for t in (*actual, predicted))
        # Replay must read the incoming mix anew, not the warmup one-hot value.
        incoming.copy_(incoming.roll(1, dims=1))
        for output in (*actual, predicted):
            output.fill_(float("nan"))
        allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
        assert tuple(t.data_ptr() for t in (*actual, predicted)) == pointers
        current = (
            residual[:live] if phase == "pre" else
            _mhc_post_reference(x[:live], residual[:live], prev_post[:live], prev_comb[:live])
        )
        torch.testing.assert_close(actual[0], current, rtol=0, atol=0.008)
        expected = _lagged_reference(actual[0], fn, scale, bias, incoming[:live], weight)
        for got, want in zip((*actual[1:], predicted[:live]), expected, strict=True):
            torch.testing.assert_close(got, want, rtol=2e-5, atol=0.008 if got.dtype == torch.bfloat16 else 4e-5)
        assert bool(torch.isnan(predicted[live:]).all())
        graph.reset()


def test_lagged_prefill_fp32_projection_precision(mhc_session):
    """Tensor-core projection retains FP32 mixing coefficients at long K."""
    device = require_sm120()
    hidden, rows = 5120, 513
    residual, _, fn, scale, bias = _make_inputs(
        tokens=rows, hidden_size=hidden, seed=415120, device=device
    )
    incoming = torch.full((rows, 4), 0.25, device=device)
    weight = torch.ones(hidden, dtype=torch.bfloat16, device=device)
    options = dict(pre_mix=incoming, norm_weight=weight, norm_eps=1e-20,
                   rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20)
    plan = prepare(mhc_session, "pre", (residual, fn, scale, bias), options, capacity=4096)
    assert plan.prepared.selection.config.backend == "tf32_tma"
    predicted = torch.empty_like(incoming)
    binding = bind(plan, tokens=rows, pre_out=predicted)
    _, post, _, _ = mhc.run_pre(
        residual, fn, scale, bias, binding=binding,
        pre_mix=incoming, norm_weight=weight, norm_eps=1e-20,
        rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
    )
    flat = residual.flatten(1).double()
    mixes = (flat @ fn.double().T) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + 1e-20
    )
    expected = 2 * torch.sigmoid(mixes[:, 4:8] * scale.double()[1] + bias.double()[4:8])
    torch.testing.assert_close(post.double(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("capacity", [128, 4096])
def test_lagged_prefill_scalar_parity_graph(mhc_session, capacity):
    """Projection changes preserve scalar BF16 rounding and fixed-capacity replay.

    The scalar finalizer defines BF16 normalization rounding. Independent Torch
    norm reductions can straddle a BF16 midpoint; strict scalar parity prevents
    a projection optimization from changing that rounding contract.
    """

    device = require_sm120()
    hidden = 5120
    residual, x, fn, scale, bias = _make_inputs(
        tokens=capacity, hidden_size=hidden, seed=92152, device=device
    )
    incoming = torch.zeros((capacity, 4), device=device)
    incoming[:, 0] = 1
    weight = torch.ones(hidden, dtype=torch.bfloat16, device=device)
    options = dict(pre_mix=incoming, norm_weight=weight, norm_eps=1e-20,
                   rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20)
    plans = [prepare(mhc_session, "pre", (residual, fn, scale, bias), options,
                     backend=backend) for backend in ("tf32_tma", "native")]
    bindings = [bind(plan, pre_out=torch.empty_like(incoming)) for plan in plans]

    def run(binding, live):
        return mhc.run_pre(
            residual[:live], fn, scale, bias, binding=binding,
            pre_mix=incoming[:live], norm_weight=weight, norm_eps=1e-20,
            rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
        )

    for binding in bindings:
        run(binding, capacity)
    torch.cuda.synchronize(device)
    mhc_session.freeze()
    for live in (3, capacity - 1, capacity):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run(bindings[0], live)
        incoming.copy_(incoming.roll(1, dims=1))
        for output in (*actual, bindings[0].pre_out):
            output.fill_(float("nan"))
        graph.replay()
        expected = run(bindings[1], live)
        torch.cuda.synchronize(device)
        for got, want in zip((*actual, bindings[0].pre_out[:live]),
                             (*expected, bindings[1].pre_out[:live]), strict=True):
            if got.dtype == torch.bfloat16:
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            else:
                torch.testing.assert_close(got, want, rtol=2e-6, atol=2e-6)
        assert bool(torch.isnan(bindings[0].pre_out[live:]).all())
        graph.reset()


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
def test_standalone_collapse_fp32_accumulation_and_frozen_replay(hidden, mhc_session):

    device = require_sm120()
    capacity = 9
    state = torch.empty(capacity, 4, hidden, dtype=torch.bfloat16, device=device)
    # BF16 intermediate accumulation would lose the small positive streams.
    streams = torch.tensor([256.0, 1.0, -256.0, 0.5], device=device)
    state.copy_(streams[None, :, None].expand_as(state))
    mix = torch.tensor([1.0, 0.75, 1.0, 0.25], device=device).repeat(capacity, 1)
    weighted = torch.empty(capacity, hidden, dtype=torch.bfloat16, device=device)
    mean = torch.empty_like(weighted)
    weighted_plan = prepare_collapse(mhc_session, state, mix, weighted)
    mean_plan = prepare_collapse(mhc_session, state, None, mean)

    def launch(rows):
        return (
            mhc.run_collapse(state[:rows], mix[:rows], out=weighted[:rows], plan=weighted_plan),
            mhc.run_collapse(state[:rows], None, out=mean[:rows], plan=mean_plan),
        )

    launch(capacity)
    torch.cuda.synchronize(device)
    mhc_session.freeze()
    for rows in (0, 1, 7, capacity):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = launch(rows)
        pointers = tuple(t.data_ptr() for t in outputs)
        state[:, 1].add_(0.5)
        mix[:, 3].add_(0.125)
        weighted.fill_(float("nan"))
        mean.fill_(float("nan"))
        allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations
        assert tuple(t.data_ptr() for t in outputs) == pointers
        expected = (state[:rows].float() * mix[:rows, :, None]).sum(1).bfloat16()
        expected_mean = state[:rows].float().mean(1).bfloat16()
        torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(outputs[1], expected_mean, rtol=0, atol=0)
        assert bool(torch.isnan(weighted[rows:]).all())
        assert bool(torch.isnan(mean[rows:]).all())
        graph.reset()


@pytest.mark.parametrize("tile_k,stages,tokens,geometry", [
    (32, 3, 1, {}), (32, 3, 17, {}), (64, 2, 1, {}), (64, 2, 17, {}),
    (64, 2, 1, {"projection_tile_n": 64, "projection_num_m_warps": 2,
                "projection_tile_m": 32}),
    (64, 2, 1, {"projection_tile_n": 32, "projection_num_n_warps": 4}),
    (256, 4, 1, {"projection_k_splits": 40}),
])
def test_lagged_post_pre_tf32_small_projection_tile(
    tile_k, stages, tokens, geometry, mhc_session, monkeypatch,
):
    if geometry:
        monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "1")
    device = require_sm120()
    hidden = 5120
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens, hidden_size=hidden, seed=92154, device=device,
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual, fn, scale, bias, rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
    )
    prev_post, prev_comb = prev_post.contiguous(), prev_comb.contiguous()
    incoming = torch.zeros((tokens, 4), device=device)
    incoming[:, 0] = 1
    weight = torch.ones(hidden, dtype=torch.bfloat16, device=device)
    options = dict(pre_mix=incoming, norm_weight=weight, norm_eps=1e-20,
                   rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20)
    config = mhc.MhcConfig(
        backend="tf32_tma", projection_tile_m=16, projection_tile_n=8,
        projection_tile_k=tile_k, projection_num_stages=stages,
        projection_num_m_warps=1, projection_num_n_warps=1,
        projection_k_splits=2, lagged_prepare=False,
    )
    from dataclasses import replace
    config = replace(config, **geometry)
    from b12x.preparation import FrozenMapping
    plan = mhc.plan(
        mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden),
        invocation=FrozenMapping(dict(operation="post_pre", lagged_mix=True,
            has_norm_weight=True, norm_eps=1e-20, rms_eps=1e-20,
            hc_eps=1e-6, sinkhorn_iters=20)), override=config,
    )
    if geometry:
        from b12x.norm.mhc._tuning import TUNING
        choice = config.to_dict()
        del choice["projection_tile_m"]
        TUNING.parameter_space(plan.query, None).validate(choice)
    args = (x, residual, prev_post, prev_comb, fn, scale, bias)
    prepare(mhc_session, "post_pre", args, options, plan=plan)
    predicted = torch.empty_like(incoming)
    binding = bind(plan, pre_out=predicted)
    mhc_session.freeze()
    graph = torch.cuda.CUDAGraph()
    try:
        with mhc_session.capture(), torch.cuda.graph(graph):
            actual = mhc.run_post_pre(*args, **options, binding=binding)
        pointers = tuple(t.data_ptr() for t in (*actual, predicted))
        incoming.copy_(incoming.roll(1, dims=1))
        x.neg_()
        for output in (*actual, predicted):
            output.fill_(float("nan"))
        allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
        assert tuple(t.data_ptr() for t in (*actual, predicted)) == pointers
        current = _mhc_post_reference(x, residual, prev_post, prev_comb)
        torch.testing.assert_close(actual[0], current, rtol=0, atol=0.008)
        expected = _lagged_reference(actual[0], fn, scale, bias, incoming, weight)
        for got, want in zip((*actual[1:], predicted), expected, strict=True):
            assert bool(torch.isfinite(got).all()) and bool(got.count_nonzero())
            torch.testing.assert_close(got, want, rtol=2e-5,
                atol=0.008 if got.dtype == torch.bfloat16 else 4e-5)
    finally:
        graph.reset()
