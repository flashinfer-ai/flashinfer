"""Prepared plan numerical contracts for native HyperConnection helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall
from b12x.norm import hyperconnection as hc
from b12x.norm.hyperconnection import _impl

from ..conftest import require_b12x


def _engram_reference(
    state: torch.Tensor,
    projected_kv: torch.Tensor,
    norm_weights: torch.Tensor,
    streams: int,
    eps: float,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    hidden = state.shape[1] // streams
    residual = state.view(-1, streams, hidden).float()
    key = projected_kv[:, : streams * hidden].reshape(-1, streams, hidden).float()
    value = projected_kv[:, streams * hidden :].float()
    rstd = torch.rsqrt(residual.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (residual * norm_weights.view(streams, hidden) * key).sum(-1)
    dot = dot * rstd * hidden**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask[:, None], 0)
    return (residual + gate[..., None] * value[:, None]).flatten(1).bfloat16()


def _prepare_engram(
    device: torch.device,
    *,
    capacity: int,
    hidden: int,
    state: torch.Tensor,
    projected: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor | None,
    output: torch.Tensor,
):
    declaration = hc.plan(
        hc.Caps(device=device, max_tokens=capacity, hidden_size=hidden),
        invocation=FrozenMapping(
            {"operation": "engram_mix", "eps": 1e-20, "token_mask": mask is not None}
        ),
    )
    request = declaration.request(
        name="engram",
        prepare_call=lambda prepared: PreparedCall(
            run=lambda: _impl.run_engram_mix_impl(
                state,
                projected,
                weights,
                eps=1e-20,
                plan=prepared,
                out=output,
                token_mask=mask,
            )
        ),
    )
    return declaration, request


def test_prepared_engram_uses_signed_zero_and_odd_offset_mask() -> None:
    device = require_b12x()
    streams, hidden, tokens = 4, 5120, 3
    state = torch.full((tokens, streams * hidden), -0.5, device=device, dtype=torch.bfloat16)
    projected = torch.zeros(
        (tokens, (streams + 1) * hidden), device=device, dtype=torch.bfloat16
    )
    projected[:, streams * hidden :] = 1
    weights = torch.ones(streams * hidden, device=device, dtype=torch.float32)
    mask_owner = torch.tensor([False, True, False, True], device=device)
    mask = mask_owner[1:]
    output = torch.empty_like(state)
    declaration, request = _prepare_engram(
        device, capacity=tokens, hidden=hidden, state=state, projected=projected,
        weights=weights, mask=mask, output=output,
    )

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((request,))
        actual = hc.run_engram_mix(
            state, projected, weights, eps=1e-20, plan=declaration,
            out=output, token_mask=mask,
        )
        expected = _engram_reference(state, projected, weights, streams, 1e-20, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        assert torch.all(actual[0] > 0)
        torch.testing.assert_close(actual[1], state[1], rtol=0, atol=0)


@pytest.mark.parametrize("with_mask", (False, True))
def test_prepared_engram_replays_changed_inputs(with_mask) -> None:
    device = require_b12x()
    streams, hidden, capacity = 4, 5120, 7
    generator = torch.Generator(device=device).manual_seed(4109)
    state = torch.randn((capacity, streams * hidden), generator=generator, device=device, dtype=torch.bfloat16)
    projected = torch.randn((capacity, (streams + 1) * hidden), generator=generator, device=device, dtype=torch.bfloat16)
    weights = torch.randn(streams * hidden, generator=generator, device=device)
    mask = torch.ones(capacity, dtype=torch.bool, device=device) if with_mask else None
    output = torch.empty_like(state)
    declaration, request = _prepare_engram(
        device, capacity=capacity, hidden=hidden, state=state, projected=projected,
        weights=weights, mask=mask, output=output,
    )

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((request,))
        plan = declaration

        def launch(rows: int) -> torch.Tensor:
            return hc.run_engram_mix(
                state[:rows], projected[:rows], weights, eps=1e-20,
                plan=plan, out=output[:rows], token_mask=mask[:rows] if mask is not None else None,
            )

        for rows in (0, 1, capacity):
            output.fill_(123)
            actual = launch(rows)
            expected = _engram_reference(
                state[:rows], projected[:rows], weights, streams, 1e-20, mask[:rows] if mask is not None else None
            )
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(
                output[rows:], torch.full_like(output[rows:], 123), rtol=0, atol=0
            )

        with session.capture():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = launch(capacity)
        pointer = captured.data_ptr()
        state.mul_(-0.5)
        projected.mul_(0.75)
        weights.neg_()
        if mask is not None:
            mask[::2] = False
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        graph.reset()
        assert captured.data_ptr() == pointer
        expected = _engram_reference(state, projected, weights, streams, 1e-20, mask)
        torch.testing.assert_close(captured, expected, rtol=1e-2, atol=1e-2)
        if mask is not None:
            torch.testing.assert_close(captured[~mask], state[~mask], rtol=0, atol=0)


def test_prepared_swiglu_preserves_asymmetric_clamp_and_vision_rounding() -> None:
    device = require_b12x()
    gate = torch.tensor(
        [-20.0, -2.0, -0.5, 0.5, 1.5, 20.0], device=device, dtype=torch.bfloat16
    )
    up = torch.tensor(
        [20.0, -20.0, 1.75, 1.75, -1.75, 20.0], device=device, dtype=torch.bfloat16
    )
    gate_up = torch.cat((gate, up)).reshape(1, -1)
    clamped = torch.empty(1, gate.numel(), device=device, dtype=torch.bfloat16)
    declaration = hc.plan(
        hc.Caps(device=device, max_tokens=1, hidden_size=gate.numel()),
        invocation=FrozenMapping(
            {
                "operation": "swiglu",
                "limit": 2.0,
                "round_silu": False,
                "left_dtype": "bfloat16",
                "right_dtype": "bfloat16",
                "output_dtype": "bfloat16",
            }
        ),
    )
    request = declaration.request(
        name="swiglu-clamped",
        prepare_call=lambda prepared: PreparedCall(
            run=lambda: _impl.run_swiglu_impl(
                gate_up, limit=2.0, out=clamped, round_silu=False, plan=prepared
            )
        ),
    )

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((request,))
        hc.run_swiglu(gate_up, limit=2.0, out=clamped, plan=declaration)
        expected = (
            F.silu(gate.float().clamp(max=2.0)) * up.float().clamp(-2.0, 2.0)
        ).bfloat16().reshape_as(clamped)
        torch.testing.assert_close(clamped, expected, rtol=0, atol=0)
        assert clamped[0, 0].abs() < 1e-6

    gates = torch.linspace(-4, 4, 1024, device=device).bfloat16().reshape(1, -1)
    ups = torch.full_like(gates, 1.75)
    merged = torch.cat((gates, ups), dim=1)
    shared = torch.empty_like(gates)
    vision = torch.empty_like(gates)
    standard_declaration = hc.plan(
        hc.Caps(device=device, max_tokens=1, hidden_size=gates.shape[1]),
        invocation=FrozenMapping(
            {
                "operation": "swiglu",
                "limit": "+inf",
                "round_silu": False,
                "left_dtype": "bfloat16",
                "right_dtype": "bfloat16",
                "output_dtype": "bfloat16",
            }
        ),
    )
    vision_declaration = hc.plan(
        hc.Caps(device=device, max_tokens=1, hidden_size=gates.shape[1]),
        invocation=FrozenMapping(
            {
                "operation": "swiglu",
                "limit": "+inf",
                "round_silu": True,
                "left_dtype": "bfloat16",
                "right_dtype": "bfloat16",
                "output_dtype": "bfloat16",
            }
        ),
    )
    standard_request = standard_declaration.request(
        name="swiglu-standard",
        prepare_call=lambda prepared: PreparedCall(
            run=lambda: _impl.run_swiglu_impl(
                merged, limit=float("inf"), out=shared, round_silu=False, plan=prepared
            )
        ),
    )
    vision_request = vision_declaration.request(
        name="swiglu-vision",
        prepare_call=lambda prepared: PreparedCall(
            run=lambda: _impl.run_swiglu_impl(
                merged, limit=float("inf"), out=vision, round_silu=True, plan=prepared
            )
        ),
    )
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((standard_request, vision_request))
        hc.run_swiglu(
            merged,
            limit=float("inf"),
            out=shared,
            plan=standard_declaration,
        )
        hc.run_swiglu(
            merged,
            limit=float("inf"),
            out=vision,
            plan=vision_declaration,
            round_silu=True,
        )
        expected_shared = (F.silu(gates.float()) * ups.float()).bfloat16()
        expected_vision = F.silu(gates) * ups
        torch.testing.assert_close(shared, expected_shared, rtol=0, atol=0)
        torch.testing.assert_close(vision, expected_vision, rtol=0, atol=0)
        assert bool((expected_shared != expected_vision).any())


def test_prepared_add_retains_fp32_before_cancellation() -> None:
    device = require_b12x()
    left = torch.tensor([[256.5, -256.5, 0.00390625]], device=device)
    right = torch.tensor([[-256.0, 256.0, 1.0]], device=device, dtype=torch.bfloat16)

    for output_dtype in (torch.bfloat16, torch.float32):
        out = torch.empty_like(left, dtype=output_dtype)
        declaration = hc.plan(
            hc.Caps(device=device, max_tokens=1, hidden_size=left.shape[1]),
            invocation=FrozenMapping(
                {
                    "operation": "add",
                    "left_dtype": "float32",
                    "right_dtype": "bfloat16",
                    "output_dtype": str(output_dtype).removeprefix("torch."),
                }
            ),
        )
        request = declaration.request(
            name=f"add-{output_dtype}",
            prepare_call=lambda prepared, out=out: PreparedCall(
                run=lambda: _impl.run_add_impl(left, right, out=out, plan=prepared)
            ),
        )
        with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
            session.prepare((request,))
            actual = hc.run_add(left, right, out=out, plan=declaration)
            assert actual.data_ptr() == out.data_ptr()
            torch.testing.assert_close(
                out, (left + right.float()).to(output_dtype), rtol=0, atol=0
            )
            assert out[0, 0] == 0.5 and out[0, 1] == -0.5


@pytest.mark.parametrize("hidden", (128, 512, 1280))
@pytest.mark.parametrize("weight_dtype", (torch.bfloat16, torch.float32))
def test_ordinary_rmsnorm_uses_affine_weight_without_offset(hidden, weight_dtype):
    device, capacity = require_b12x(), 7
    state = torch.randn((capacity, hidden), device=device, dtype=torch.bfloat16)
    weight = torch.linspace(-0.5, 1.5, hidden, device=device).to(weight_dtype)
    normalized = torch.empty_like(state)
    plan = hc.plan(
        hc.Caps(device=device, max_tokens=capacity, hidden_size=hidden, streams=1, lowrank=1),
        invocation=FrozenMapping({"operation": "grouped_rmsnorm", "eps": 1e-20,
                                  "zero_centered": False,
                                  "weight_dtype": str(weight_dtype).removeprefix("torch.")}),
    )
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="rmsnorm", prepare_call=lambda prepared: PreparedCall(
            run=lambda: _impl.run_grouped_rmsnorm_impl(
                state, weight, eps=1e-20, plan=prepared, out=normalized, zero_centered=False,
            ),
        )),))
        bottleneck = torch.empty((capacity, 1), device=device, dtype=torch.bfloat16)
        block_input = torch.empty_like(state)
        session.freeze()
        for rows in (0, 1, capacity):
            binding = hc.bind(plan, normalized=normalized, bottleneck=bottleneck,
                              block_input=block_input, tokens=rows)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = hc.run_grouped_rmsnorm(state[:rows], weight, eps=1e-20,
                                                binding=binding, zero_centered=False)
            state.mul_(0.75)
            weight.neg_()
            normalized.fill_(float("nan"))
            graph.replay()
            values = state[:rows].float()
            expected = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-20)
                        * weight.float()).bfloat16()
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
            assert torch.isnan(normalized[rows:]).all()
            graph.reset()
