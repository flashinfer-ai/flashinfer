from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pytest
import torch

from benchmarks.experimental.b12x.common import require_sm120
from b12x.norm import mhc

from .test_mhc_lagged import _lagged_reference
from b12x.testing.mhc import (
    make_inputs as _make_inputs,
    pre_reference as _mhc_pre_reference,
    post_reference as _mhc_post_reference,
)
from ._mhc import declaration, prepare, bind, mhc_session


T = TypeVar("T")


# Keep the eager Torch source of truth out of TF32 mode.  In particular, this
# makes a mismatch in the producer's BF16 collapse visible rather than a matmul
# precision choice in the reference.
def _source_reference(call: Callable[[], T]) -> T:
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return call()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _poison(binding: object) -> None:
    for scratch in (binding.partials, binding.y):
        if scratch.is_floating_point():
            scratch.fill_(float("nan"))


def _assert_lagged_outputs(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    pre_out: torch.Tensor,
    current: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    incoming: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    post, comb, y, predicted = _source_reference(
        lambda: _lagged_reference(actual[0], fn, scale, bias, incoming, weight)
    )
    torch.testing.assert_close(actual[0], current, rtol=0.0, atol=8e-3)
    torch.testing.assert_close(actual[1], post, rtol=2e-5, atol=4e-5)
    torch.testing.assert_close(actual[2], comb, rtol=2e-5, atol=4e-5)
    # The oracle rounds the four-way collapse to BF16 *before* its RMS sum.
    torch.testing.assert_close(actual[3], y, rtol=0.0, atol=8e-3)
    torch.testing.assert_close(pre_out, predicted, rtol=2e-5, atol=4e-5)


def _nonuniform_mix(rows: int, device: torch.device) -> torch.Tensor:
    row = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)
    # Dyadic coefficients isolate the new BF16 variance boundary from the
    # existing fused-versus-unfused FP32 contraction rounding difference.
    base = torch.tensor([0.75, -0.5, 0.25, 0.125], device=device)
    slope = torch.tensor([1 / 64, -1 / 32, 1 / 128, -1 / 64], device=device)
    return base + row * slope


def test_mhc_lagged_parallel_rejects_y_output_alias(mhc_session) -> None:
    """The producer writes y before consumers finish reading the residual."""
    device = require_sm120()
    hidden = 5120
    residual, _, fn, scale, bias = _make_inputs(
        tokens=1, hidden_size=hidden, seed=923_100, device=device
    )
    incoming = _nonuniform_mix(1, device)
    pre_out = torch.empty_like(incoming)
    y_alias = residual[:, 0, :]
    assert y_alias.is_contiguous()
    options = dict(pre_mix=incoming, rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20)
    plan = prepare(mhc_session, "pre", (residual, fn, scale, bias), options)
    binding = bind(plan, pre_out=pre_out, y=y_alias)
    with pytest.raises(ValueError, match="must not alias"):
        mhc.run_pre(
            residual,
            fn,
            scale,
            bias,
            binding=binding,
            pre_mix=incoming,
            rms_eps=1e-20,
            hc_eps=1e-6,
            sinkhorn_iters=20,
        )


def test_mhc_lagged_pre_unbound_frozen_capacity_mode(
    mhc_session,
) -> None:
    """Unbound prepared pre keeps its producer/finalizer mode across live rows."""
    device = require_sm120()
    hidden, capacity = 5120, 128
    residual, _, fn, scale, bias = _make_inputs(
        tokens=capacity, hidden_size=hidden, seed=923_102, device=device
    )
    residual[:, 0].fill_(1.0)
    residual[:, 1].fill_(1.0)
    residual[:, 1, 4096:].fill_(1.0078125)
    weight = torch.linspace(0.5, 1.5, hidden, device=device).bfloat16()
    incoming = _nonuniform_mix(capacity, device)
    pre_out = torch.empty_like(incoming)
    residual_out = torch.empty_like(residual)
    y_out = torch.empty((capacity, hidden), dtype=torch.bfloat16, device=device)
    post_out = torch.empty((capacity, 4), dtype=torch.float32, device=device)
    comb_out = torch.empty((capacity, 4, 4), dtype=torch.float32, device=device)

    options = dict(
        pre_mix=incoming,
        norm_weight=weight,
        norm_eps=1e-20,
        rms_eps=1e-20,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    plan = prepare(mhc_session, "pre", (residual, fn, scale, bias), options)

    def run(live: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return mhc.run_pre(
            residual[:live],
            fn,
            scale,
            bias,
            plan=plan,
            residual_out=residual_out[:live],
            y_out=y_out[:live],
            post_out=post_out[:live],
            comb_out=comb_out[:live],
            pre_mix=incoming[:live],
            pre_out=pre_out[:live],
            norm_weight=weight,
            norm_eps=1e-20,
            rms_eps=1e-20,
            hc_eps=1e-6,
            sinkhorn_iters=20,
        )

    actual = run(3)
    _assert_lagged_outputs(
        actual, pre_out[:3], residual[:3], fn, scale, bias, incoming[:3], weight
    )
    torch.cuda.synchronize(device)
    mhc_session.freeze()

    actual = run(capacity)
    _assert_lagged_outputs(
        actual,
        pre_out,
        residual,
        fn,
        scale,
        bias,
        incoming,
        weight,
    )


@pytest.mark.parametrize("phase", ["pre", "post_pre"])
@pytest.mark.parametrize("partials_per_cta", [4, 9, 13, 25])
def test_mhc_lagged_parallel_decode_frozen_live_graph(
    phase: str, partials_per_cta: int, mhc_session, monkeypatch
) -> None:
    """Lagged decode consumes freshly produced BF16 collapse statistics on replay."""
    monkeypatch.delenv("B12X_MHC_PARTIALS_PER_CTA", raising=False)
    device = require_sm120()
    hidden, capacity = 5120, 16
    live_counts = (1, 2, 4, 6, 8, 16)
    residual, x, fn, scale, bias = _make_inputs(
        tokens=capacity, hidden_size=hidden, seed=923_101, device=device
    )
    # This is the variance-boundary pattern from the existing lagged regression:
    # a pre-rounding RMS reduction and an RMS reduction of rounded y diverge.
    residual[:, 0].fill_(1.0)
    residual[:, 1].fill_(1.0)
    residual[:, 1, 4096:].fill_(1.0078125)
    weight = torch.linspace(0.5, 1.5, hidden, device=device).bfloat16()
    incoming = _nonuniform_mix(capacity, device)
    pre_out = torch.full_like(incoming, float("nan"))
    prev_y, prev_post, prev_comb = _source_reference(
        lambda: _mhc_pre_reference(
            residual, fn, scale, bias, rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20
        )
    )
    del prev_y
    prev_post, prev_comb = prev_post.contiguous(), prev_comb.contiguous()

    options = dict(
        pre_mix=incoming,
        norm_weight=weight,
        norm_eps=1e-20,
        rms_eps=1e-20,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    args = (
        (residual, fn, scale, bias)
        if phase == "pre"
        else (x, residual, prev_post, prev_comb, fn, scale, bias)
    )
    plan = prepare(
        mhc_session,
        phase,
        args,
        options,
        lagged_prepare=True,
        partials_per_cta=partials_per_cta,
    )
    binding = bind(plan, pre_out=pre_out)

    def run(live: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if phase == "pre":
            return mhc.run_pre(
                residual[:live],
                fn,
                scale,
                bias,
                binding=binding,
                pre_mix=incoming[:live],
                norm_weight=weight,
                norm_eps=1e-20,
                rms_eps=1e-20,
                hc_eps=1e-6,
                sinkhorn_iters=20,
            )
        return mhc.run_post_pre(
            x[:live],
            residual[:live],
            prev_post[:live],
            prev_comb[:live],
            fn,
            scale,
            bias,
            binding=binding,
            pre_mix=incoming[:live],
            norm_weight=weight,
            norm_eps=1e-20,
            rms_eps=1e-20,
            hc_eps=1e-6,
            sinkhorn_iters=20,
        )

    # One prepared capacity accepts every live decode count.
    for live in live_counts:
        _poison(binding)
        run(live)
    torch.cuda.synchronize(device)
    mhc_session.freeze()

    for live in live_counts:
        _poison(binding)
        binding.pre_out.fill_(float("nan"))
        post_before = prev_post.clone()
        comb_before = prev_comb.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run(live)
        torch.cuda.synchronize(device)
        pointers = tuple(t.data_ptr() for t in (*actual, binding.pre_out))

        # Replay changes every producer input, including a non-uniform mix; a
        # stale rounded y or partial RMS statistic therefore cannot pass.
        incoming[:live].copy_(_nonuniform_mix(live, device).roll(1, dims=1))
        residual[:live, 2].add_(
            torch.arange(live, device=device).view(-1, 1).bfloat16() / 32
        )
        if phase == "post_pre":
            x[:live].add_(torch.arange(live, device=device).view(-1, 1).bfloat16() / 16)
        replay_incoming = incoming[:live].clone()
        replay_residual = residual[:live].clone()
        replay_x = x[:live].clone()
        allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
        assert tuple(t.data_ptr() for t in (*actual, binding.pre_out)) == pointers
        torch.testing.assert_close(incoming[:live], replay_incoming, rtol=0, atol=0)
        torch.testing.assert_close(residual[:live], replay_residual, rtol=0, atol=0)
        if phase == "post_pre":
            torch.testing.assert_close(x[:live], replay_x, rtol=0, atol=0)
        torch.testing.assert_close(prev_post, post_before, rtol=0, atol=0)
        torch.testing.assert_close(prev_comb, comb_before, rtol=0, atol=0)
        current = (
            residual[:live]
            if phase == "pre"
            else _source_reference(
                lambda: _mhc_post_reference(
                    x[:live], residual[:live], prev_post[:live], prev_comb[:live]
                )
            )
        )
        _assert_lagged_outputs(
            actual,
            binding.pre_out[:live],
            current,
            fn,
            scale,
            bias,
            incoming[:live],
            weight,
        )

        graph.reset()


def test_mhc_lagged_post_pre_static_split_reuses_frozen_capacity(
    mhc_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planned split producer and finalizer agree below the planned capacity."""
    from b12x.norm.mhc._preparation import _lower_native

    device = require_sm120()
    hidden, capacity = 4096, 8
    residual, x, fn, scale, bias = _make_inputs(
        tokens=capacity, hidden_size=hidden, seed=923_102, device=device
    )
    incoming = _nonuniform_mix(capacity, device)
    weight = torch.ones(hidden, dtype=torch.bfloat16, device=device)
    pre_out = torch.full_like(incoming, float("nan"))
    _, prev_post, prev_comb = _source_reference(
        lambda: _mhc_pre_reference(
            residual, fn, scale, bias, rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20
        )
    )
    prev_post, prev_comb = prev_post.contiguous(), prev_comb.contiguous()

    options = dict(
        pre_mix=incoming,
        norm_weight=weight,
        norm_eps=1e-20,
        rms_eps=1e-20,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    args = (x, residual, prev_post, prev_comb, fn, scale, bias)
    with monkeypatch.context() as controls:
        controls.setenv("B12X_MHC_DECODE_SPLITS", "4")
        controls.setenv("B12X_MHC_DECODE_TILE_N", "6")
        plan = declaration("post_pre", args, options, lagged_prepare=False)
    plan = prepare(mhc_session, "post_pre", args, options, plan=plan)
    native = _lower_native(
        plan.query, plan.prepared.selection.config, plan.prepared.device
    )
    assert native.source_splits == 4 and native.decode_tile_n == 6
    binding = bind(plan, pre_out=pre_out)

    def run(live: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return mhc.run_post_pre(
            x[:live],
            residual[:live],
            prev_post[:live],
            prev_comb[:live],
            fn,
            scale,
            bias,
            binding=binding,
            pre_mix=incoming[:live],
            norm_weight=weight,
            norm_eps=1e-20,
            rms_eps=1e-20,
            hc_eps=1e-6,
            sinkhorn_iters=20,
        )

    # Warm only the capacity specialization. The live-one call must reuse it.
    _poison(binding)
    run(capacity)
    torch.cuda.synchronize(device)
    mhc_session.freeze()

    _poison(binding)
    binding.pre_out.fill_(float("nan"))
    actual = run(1)
    torch.cuda.synchronize(device)
    current = _source_reference(
        lambda: _mhc_post_reference(x[:1], residual[:1], prev_post[:1], prev_comb[:1])
    )
    _assert_lagged_outputs(
        actual,
        binding.pre_out[:1],
        current,
        fn,
        scale,
        bias,
        incoming[:1],
        weight,
    )
