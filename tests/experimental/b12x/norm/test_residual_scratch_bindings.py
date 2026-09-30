from __future__ import annotations

import pytest
import torch

from b12x.norm import mhc
from b12x.norm.mhc import _impl
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall, require_prepared
from b12x.testing.mhc import make_inputs, pre_reference

from ..conftest import require_b12x


def _prepared_post_pre(device: torch.device, *, tokens: int = 1, hidden: int = 4096):
    residual, x, fn, scale, bias = make_inputs(tokens=tokens, hidden_size=hidden, seed=190_001, device=device)
    _, post, comb = pre_reference(residual, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20)
    declaration = mhc.plan(
        mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden, split_k=hidden // 64),
        invocation=FrozenMapping({
            "operation": "post_pre", "output_mode": "provided", "has_norm_weight": False,
            "rms_eps": 1e-6, "hc_eps": 1e-6, "sinkhorn_iters": 20, "norm_eps": 0.0,
        }),
    )

    def prepare(state):
        scratch = torch.empty(state.scratch_specs()[0].shape, dtype=torch.uint8, device=device)
        binding = state.bind(
            scratch=scratch, tokens=tokens,
            out=torch.empty((tokens, 4, hidden), dtype=torch.bfloat16, device=device),
            post=torch.empty((tokens, 4), dtype=torch.float32, device=device),
            comb=torch.empty((tokens, 4, 4), dtype=torch.float32, device=device),
            y=torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device),
        )
        return PreparedCall(run=lambda: _impl._b12x_mhc_post_pre_impl(x, residual, post, comb, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20, _state=state, binding=binding), owners=(scratch, binding))

    return declaration, prepare, (x, residual, post, comb, fn, scale, bias)


def test_mhc_prepared_state_binds_caller_scratch_and_outputs() -> None:
    device = require_b12x()
    declaration, prepare, _ = _prepared_post_pre(device)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(name="scratch", prepare_call=prepare),))
        plan = declaration
        state = require_prepared(plan, "norm.mhc")
        spec = state.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=torch.uint8, device=device)
        out = torch.empty((1, 4, 4096), dtype=torch.bfloat16, device=device)
        post = torch.empty((1, 4), dtype=torch.float32, device=device)
        comb = torch.empty((1, 4, 4), dtype=torch.float32, device=device)
        y = torch.empty((1, 4096), dtype=torch.bfloat16, device=device)
        binding = mhc.bind(plan, scratch=scratch, tokens=1, out=out, post=post, comb=comb, y=y)
        assert binding.plan is plan
        assert binding.partials.shape == (1, 64, mhc.PARTIALS)
        assert binding.partials.untyped_storage().data_ptr() == scratch.untyped_storage().data_ptr()
        assert binding.out is out and binding.post_buffer is post
        assert binding.comb_buffer is comb and binding.y is y


def test_mhc_binding_rejects_unprepared_declarations_and_wrong_exact_m() -> None:
    device = require_b12x()
    declaration, prepare, _ = _prepared_post_pre(device, tokens=1)
    with pytest.raises(TypeError):
        mhc.bind(declaration, scratch=torch.empty((1,), dtype=torch.uint8, device=device))
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(name="exact-m", prepare_call=prepare),))
        plan = declaration
        spec = require_prepared(plan, "norm.mhc").scratch_specs()[0]
        with pytest.raises(ValueError, match="exact planned nonempty M"):
            mhc.bind(plan, scratch=torch.empty(spec.shape, dtype=torch.uint8, device=device), tokens=2)
