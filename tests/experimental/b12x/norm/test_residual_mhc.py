from __future__ import annotations

import pytest
import torch

from b12x.norm import mhc
from b12x.norm.mhc import _impl
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall, require_prepared
from b12x.testing.mhc import make_inputs, post_reference, pre_reference

from ..conftest import require_b12x


def _invocation(*, norm: bool, rms_eps: float = 1e-6) -> FrozenMapping:
    return FrozenMapping({
        "operation": "post_pre", "output_mode": "provided", "has_norm_weight": norm,
        "rms_eps": rms_eps, "hc_eps": 1e-6, "sinkhorn_iters": 20,
        "norm_eps": rms_eps if norm else 0.0,
    })


def _reference(residual, x, fn, scale, bias, norm, *, rms_eps):
    previous_y, previous_post, previous_comb = pre_reference(residual, fn, scale, bias, rms_eps=rms_eps, hc_eps=1e-6, sinkhorn_iters=20)
    carry = post_reference(x, residual, previous_post, previous_comb)
    raw, post, comb = pre_reference(carry, fn, scale, bias, rms_eps=rms_eps, hc_eps=1e-6, sinkhorn_iters=20, y_dtype=torch.float32)
    if norm is None:
        y = raw.bfloat16()
    else:
        y = (raw.bfloat16().float() * torch.rsqrt(raw.square().mean(dim=-1, keepdim=True) + rms_eps) * norm.float()).bfloat16()
    return previous_post, previous_comb, carry, post, comb, y


def _bound_call(state, *, x, residual, previous_post, previous_comb, fn, scale, bias, norm, rms_eps):
    tokens, hidden = x.shape
    scratch = torch.empty(state.scratch_specs()[0].shape, dtype=torch.uint8, device=x.device)
    binding = state.bind(
        scratch=scratch, tokens=tokens,
        out=torch.empty((tokens, 4, hidden), dtype=torch.bfloat16, device=x.device),
        post=torch.empty((tokens, 4), dtype=torch.float32, device=x.device),
        comb=torch.empty((tokens, 4, 4), dtype=torch.float32, device=x.device),
        y=torch.empty((tokens, hidden), dtype=torch.bfloat16, device=x.device),
    )
    return PreparedCall(run=lambda: _impl._b12x_mhc_post_pre_impl(x, residual, previous_post, previous_comb, fn, scale, bias, rms_eps=rms_eps, hc_eps=1e-6, sinkhorn_iters=20, norm_weight=norm, norm_eps=rms_eps if norm is not None else 0.0, _state=state, binding=binding), owners=(scratch, binding))


@pytest.mark.parametrize(("tokens", "rms_eps"), [(1, 1e-20), (3, 1e-6)], ids=["vision-epsilon", "glm-epsilon"])
def test_mhc_prepared_norm_matches_native_oracle(tokens: int, rms_eps: float) -> None:
    device, hidden = require_b12x(), 4096
    residual, x, fn, scale, bias = make_inputs(tokens=tokens, hidden_size=hidden, seed=53_001 + tokens, device=device)
    norm = torch.linspace(0.5, 1.5, hidden, device=device).bfloat16()
    previous_post, previous_comb, expected_carry, expected_post, expected_comb, expected_y = _reference(residual, x, fn, scale, bias, norm, rms_eps=rms_eps)
    declaration = mhc.plan(mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden, split_k=64), invocation=_invocation(norm=True, rms_eps=rms_eps))

    def prepare(state):
        return _bound_call(state, x=x, residual=residual, previous_post=previous_post, previous_comb=previous_comb, fn=fn, scale=scale, bias=bias, norm=norm, rms_eps=rms_eps)

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(name="norm", prepare_call=prepare),))
        plan = declaration
        state = require_prepared(plan, "norm.mhc")
        scratch = torch.empty(state.scratch_specs()[0].shape, dtype=torch.uint8, device=device)
        binding = mhc.bind(plan, scratch=scratch, tokens=tokens, out=torch.empty_like(residual), post=torch.empty_like(previous_post), comb=torch.empty_like(previous_comb), y=torch.empty_like(x))
        actual = mhc.run_post_pre(x, residual, previous_post, previous_comb, fn, scale, bias, rms_eps=rms_eps, hc_eps=1e-6, sinkhorn_iters=20, norm_weight=norm, norm_eps=rms_eps, binding=binding)
        torch.testing.assert_close(actual[0], expected_carry, rtol=0.0, atol=2e-2)
        torch.testing.assert_close(actual[3], expected_y, rtol=0.0, atol=6e-3)
        torch.testing.assert_close(actual[1], expected_post, rtol=2e-6, atol=1e-5)
        torch.testing.assert_close(actual[2], expected_comb, rtol=2e-6, atol=4e-5)


def test_mhc_prepared_projection_splits_share_projection_and_keep_finalizers_distinct() -> None:
    device, hidden = require_b12x(), 4096
    requests = []
    for rows in (1, 128):
        residual, x, fn, scale, bias = make_inputs(tokens=rows, hidden_size=hidden, seed=17, device=device)
        _, previous_post, previous_comb = pre_reference(residual, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20)
        norm = torch.ones(hidden, device=device, dtype=torch.bfloat16)
        for splits in (1, 16):
            config = mhc.MhcConfig(backend="tf32_tma", projection_tile_m=16, projection_tile_n=8, projection_tile_k=256, projection_num_stages=1, projection_num_m_warps=1, projection_num_n_warps=1, projection_k_splits=splits)
            declaration = mhc.plan(mhc.Caps(device=device, max_tokens=rows, hidden_size=hidden, split_k=64), invocation=_invocation(norm=True), override=config)
            name = f"m{rows}-s{splits}"
            requests.append(declaration.request(name=name, prepare_call=lambda state, values=(x, residual, previous_post, previous_comb, fn, scale, bias, norm): _bound_call(state, x=values[0], residual=values[1], previous_post=values[2], previous_comb=values[3], fn=values[4], scale=values[5], bias=values[6], norm=values[7], rms_eps=1e-6)))
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        result = session.prepare(requests)
        projection, finalizers = {}, set()
        for name, plan in result.plans.items():
            splits = name.rsplit("-s", 1)[1]
            for program in require_prepared(plan, "norm.mhc").launchers["partial"].__b12x_programs__:
                if "mhc_prefill_tf32_project_tma_" in program.name:
                    projection.setdefault(splits, set()).add(program)
                if "mhc_finalize_gram_" in program.name:
                    finalizers.add(program)
        # The TF32 projection specializes on its configured K-split count; each
        # count shares one projection across both planned capacities.
        assert sorted(projection) == ["1", "16"]
        assert all(len(programs) == 1 for programs in projection.values())
        assert projection["1"].isdisjoint(projection["16"])
        assert len(finalizers) == 2


def test_mhc_prepared_high_slot_capture_replays_changed_inputs() -> None:
    device, tokens, hidden = require_b12x(), 1, 7168
    residual, x, fn, scale, bias = make_inputs(tokens=tokens, hidden_size=hidden, seed=91_480, device=device)
    previous_post, previous_comb, _, _, _, _ = _reference(residual, x, fn, scale, bias, None, rms_eps=1e-6)
    declaration = mhc.plan(mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden, split_k=112), invocation=_invocation(norm=False))

    def prepare(state):
        return _bound_call(state, x=x, residual=residual, previous_post=previous_post, previous_comb=previous_comb, fn=fn, scale=scale, bias=bias, norm=None, rms_eps=1e-6)

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(name="high-slot", prepare_call=prepare),))
        plan = declaration
        state = require_prepared(plan, "norm.mhc")
        scratch = torch.empty(state.scratch_specs()[0].shape, dtype=torch.uint8, device=device)
        binding = mhc.bind(plan, scratch=scratch, tokens=tokens, out=torch.empty_like(residual), post=torch.empty_like(previous_post), comb=torch.empty_like(previous_comb), y=torch.empty_like(x))
        def run():
            return mhc.run_post_pre(x, residual, previous_post, previous_comb, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20, binding=binding)
        run()
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture():
                with torch.cuda.graph(graph):
                    outputs = run()
            pointers = tuple(output.data_ptr() for output in outputs)
            x.mul_(-0.5).add_(0.01171875)
            residual.mul_(0.625).sub_(0.01953125)
            expected_carry = post_reference(x, residual, previous_post, previous_comb)
            expected_y, expected_post, expected_comb = pre_reference(expected_carry, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20)
            for output in outputs:
                output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            assert tuple(output.data_ptr() for output in outputs) == pointers
            torch.testing.assert_close(outputs[0], expected_carry, rtol=0.0, atol=2e-2)
            torch.testing.assert_close(outputs[3], expected_y, rtol=0.0, atol=8e-3)
            torch.testing.assert_close(outputs[1], expected_post, rtol=2e-6, atol=1e-5)
            torch.testing.assert_close(outputs[2], expected_comb, rtol=2e-6, atol=1e-5)
        finally:
            graph.reset()


@pytest.mark.parametrize("tokens", [3, 129])
def test_mhc_prefill_capacity_reuses_launchers_and_scratch(tokens):
    from b12x._lib.runtime_control import kernel_resolution_guard

    device, hidden, capacity = require_b12x(), 4096, 4096
    residual, x, fn, scale, bias = make_inputs(tokens=tokens, hidden_size=hidden, seed=193 + tokens, device=device)
    norm = torch.ones(hidden, device=device, dtype=torch.bfloat16)
    previous_post, previous_comb, _, _, _, _ = _reference(residual, x, fn, scale, bias, norm, rms_eps=1e-6)
    declaration = mhc.plan(
        mhc.Caps(device=device, max_tokens=capacity, hidden_size=hidden, split_k=64),
        invocation=_invocation(norm=True),
    )
    def prepare(state):
        return _bound_call(state, x=x, residual=residual, previous_post=previous_post, previous_comb=previous_comb, fn=fn, scale=scale, bias=bias, norm=norm, rms_eps=1e-6)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(name="prefill-capacity", prepare_call=prepare),))
        exact = mhc.plan(
            mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden, split_k=64),
            invocation=_invocation(norm=True), override=declaration.selection.config,
        )
        session.prepare((exact.request(name="exact-configured-execution", prepare_call=prepare),))
        reference = prepare(require_prepared(exact, "norm.mhc"))
        expected = tuple(value.clone() for value in reference.invoke())
        scratch = torch.empty(declaration.scratch_specs()[0].shape, dtype=torch.uint8, device=device)
        binding = mhc.bind(declaration, scratch=scratch, tokens=tokens, out=torch.empty_like(residual), post=torch.empty_like(previous_post), comb=torch.empty_like(previous_comb), y=torch.empty_like(x))
        pointers = (scratch.data_ptr(), binding.partials.data_ptr(), binding.out.data_ptr(), binding.y.data_ptr())
        def run():
            return mhc.run_post_pre(x, residual, previous_post, previous_comb, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20, norm_weight=norm, norm_eps=1e-6, binding=binding)
        session.freeze()
        with kernel_resolution_guard("mHC prefill capacity"):
            actual = run()
            for got, want in zip(actual, expected, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            try:
                with session.capture(), torch.cuda.graph(graph):
                    captured = run()
                x.mul_(-0.5)
                graph.replay()
                torch.cuda.synchronize()
                replayed = tuple(value.clone() for value in captured)
                eager = reference.invoke()
                for got, want in zip(replayed, eager, strict=True):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
                assert (scratch.data_ptr(), binding.partials.data_ptr(), binding.out.data_ptr(), binding.y.data_ptr()) == pointers
            finally:
                graph.reset()
