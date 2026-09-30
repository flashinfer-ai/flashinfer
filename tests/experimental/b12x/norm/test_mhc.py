from __future__ import annotations

import pytest
import torch

from b12x.norm import mhc
from b12x.norm.mhc import _impl
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall, require_prepared
from b12x.testing.mhc import make_inputs, post_reference, pre_reference

from ..conftest import require_b12x


def _declaration(device: torch.device, *, tokens: int, hidden_size: int, norm: bool = False):
    return mhc.plan(
        mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden_size, split_k=hidden_size // 64),
        invocation=FrozenMapping({
            "operation": "post_pre",
            "output_mode": "provided",
            "has_norm_weight": norm,
            "rms_eps": 1e-6,
            "hc_eps": 1e-6,
            "sinkhorn_iters": 20,
            "norm_eps": 1e-6 if norm else 0.0,
        }),
    )


def _outputs(tokens: int, hidden: int, device: torch.device):
    return (
        torch.empty((tokens, 4, hidden), dtype=torch.bfloat16, device=device),
        torch.empty((tokens, 4), dtype=torch.float32, device=device),
        torch.empty((tokens, 4, 4), dtype=torch.float32, device=device),
        torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device),
    )


@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_mhc_prepared_post_pre_matches_native_rounding(tokens: int) -> None:
    device, hidden = require_b12x(), 4096
    residual, x, fn, scale, bias = make_inputs(tokens=tokens, hidden_size=hidden, seed=91_450 + tokens, device=device)
    _, previous_post, previous_comb = pre_reference(residual, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20)
    declaration = _declaration(device, tokens=tokens, hidden_size=hidden)

    # The preparation callback receives the materialized state, binds the real
    # caller storage, and primes the same native branch used by the public call.
    def prepare(state):
        scratch = torch.empty(state.scratch_specs()[0].shape, dtype=torch.uint8, device=device)
        out, post, comb, y = _outputs(tokens, hidden, device)
        binding = state.bind(scratch=scratch, tokens=tokens, out=out, post=post, comb=comb, y=y)
        return PreparedCall(
            run=lambda: _impl._b12x_mhc_post_pre_impl(
                x, residual, previous_post.contiguous(), previous_comb.contiguous(), fn, scale, bias,
                rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20, _state=state, binding=binding,
            ),
            owners=(scratch, binding),
        )

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(name="post-pre", prepare_call=prepare),))
        plan = declaration
        scratch = torch.empty(require_prepared(plan, "norm.mhc").scratch_specs()[0].shape, dtype=torch.uint8, device=device)
        out, post, comb, y = _outputs(tokens, hidden, device)
        binding = mhc.bind(plan, scratch=scratch, tokens=tokens, out=out, post=post, comb=comb, y=y)
        actual = mhc.run_post_pre(
            x, residual, previous_post.contiguous(), previous_comb.contiguous(), fn, scale, bias,
            rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20, binding=binding,
        )
        expected_residual = post_reference(x, residual, previous_post, previous_comb)
        expected_y, expected_post, expected_comb = pre_reference(expected_residual, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20)
        assert tuple(value.data_ptr() for value in actual) == (out.data_ptr(), post.data_ptr(), comb.data_ptr(), y.data_ptr())
        torch.testing.assert_close(actual[0], expected_residual, rtol=0.0, atol=2e-2)
        torch.testing.assert_close(actual[3], expected_y, rtol=0.0, atol=8e-3)
        scalar_atol = 2e-5 if tokens >= 8 else 1e-5
        torch.testing.assert_close(actual[1], expected_post, rtol=2e-6, atol=scalar_atol)
        torch.testing.assert_close(actual[2], expected_comb, rtol=2e-6, atol=scalar_atol)


def test_mhc_prepared_post_pre_capture_replays_changed_inputs() -> None:
    device, tokens, hidden = require_b12x(), 2, 4096
    residual, x, fn, scale, bias = make_inputs(tokens=tokens, hidden_size=hidden, seed=91_460, device=device)
    _, previous_post, previous_comb = pre_reference(residual, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20)
    declaration = _declaration(device, tokens=tokens, hidden_size=hidden)

    def prepare(state):
        scratch = torch.empty(state.scratch_specs()[0].shape, dtype=torch.uint8, device=device)
        out, post, comb, y = _outputs(tokens, hidden, device)
        binding = state.bind(scratch=scratch, tokens=tokens, out=out, post=post, comb=comb, y=y)
        return PreparedCall(run=lambda: _impl._b12x_mhc_post_pre_impl(x, residual, previous_post, previous_comb, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20, _state=state, binding=binding), owners=(scratch, binding))

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(name="capture", prepare_call=prepare),))
        plan = declaration
        scratch = torch.empty(require_prepared(plan, "norm.mhc").scratch_specs()[0].shape, dtype=torch.uint8, device=device)
        binding = mhc.bind(plan, scratch=scratch, tokens=tokens, out=torch.empty((tokens, 4, hidden), dtype=torch.bfloat16, device=device), post=torch.empty((tokens, 4), dtype=torch.float32, device=device), comb=torch.empty((tokens, 4, 4), dtype=torch.float32, device=device), y=torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device))
        def run():
            return mhc.run_post_pre(x, residual, previous_post, previous_comb, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20, binding=binding)
        run()
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture():
                with torch.cuda.graph(graph):
                    outputs = run()
            pointers = tuple(value.data_ptr() for value in outputs)
            x.mul_(-0.5).add_(0.01171875)
            residual.mul_(0.625).sub_(0.01953125)
            expected_residual = post_reference(x, residual, previous_post, previous_comb)
            expected_y, expected_post, expected_comb = pre_reference(expected_residual, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20)
            for output in outputs:
                output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            assert tuple(value.data_ptr() for value in outputs) == pointers
            torch.testing.assert_close(outputs[0], expected_residual, rtol=0.0, atol=2e-2)
            torch.testing.assert_close(outputs[3], expected_y, rtol=0.0, atol=4e-3)
            torch.testing.assert_close(outputs[1], expected_post, rtol=2e-6, atol=1e-5)
            torch.testing.assert_close(outputs[2], expected_comb, rtol=2e-6, atol=1e-5)
        finally:
            graph.reset()


@pytest.mark.parametrize("tokens", [1, 3])
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
def test_mhc_prepared_post_pre_norm_uses_unrounded_variance(tokens, norm_dtype):
    from ._mhc import prepare, bind
    device, hidden = require_b12x(), 4096
    residual, x, fn, scale, bias = make_inputs(
        tokens=tokens, hidden_size=hidden, seed=91_470 + tokens, device=device)
    gen = torch.Generator(device="cpu").manual_seed(91_471 + tokens)
    norm = torch.randn(hidden, generator=gen).to(device=device, dtype=norm_dtype)
    _, previous_post, previous_comb = pre_reference(
        residual, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20)
    args = (x, residual, previous_post.contiguous(), previous_comb.contiguous(), fn, scale, bias)
    options = dict(rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20, norm_weight=norm, norm_eps=1e-6)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        plan = prepare(session, "post_pre", args, options)
        binding = bind(plan)
        session.freeze()
        current, post, comb, y = mhc.run_post_pre(*args, **options, binding=binding)
        residual_ref = post_reference(x, residual, previous_post, previous_comb)
        y_raw, post_ref, comb_ref = pre_reference(
            residual_ref, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6,
            sinkhorn_iters=20, y_dtype=torch.float32)
        rms_scale = torch.rsqrt(y_raw.square().mean(-1, keepdim=True) + 1e-6)
        expected_y = (y_raw.bfloat16().float() * rms_scale * norm.float()).bfloat16()
        torch.testing.assert_close(current, residual_ref, rtol=0, atol=2e-2)
        torch.testing.assert_close(y, expected_y, rtol=0, atol=6e-3)
        torch.testing.assert_close(post, post_ref, rtol=2e-6, atol=1e-5)
        torch.testing.assert_close(comb, comb_ref, rtol=2e-6, atol=4e-5)
