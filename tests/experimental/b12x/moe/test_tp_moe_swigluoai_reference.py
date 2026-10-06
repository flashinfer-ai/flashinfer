from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from b12x._lib.intrinsics import pack_grouped_fp4_values, swizzle_block_scale
from b12x.moe.fused_moe._impl import (
    B12XFP4ExpertWeights,
    TPMoEFP4Binding,
    _PreparedWeightRepresentation,
    b12x_moe_fp4,
    plan_b12x_fp4_moe_weights,
)
from b12x.moe._shared.kernels.activations import (
    SWIGLUOAI_DEFAULT_ALPHA,
    SWIGLUOAI_DEFAULT_BETA,
    SWIGLUOAI_DEFAULT_LIMIT,
    SWIGLUOAI_UNINTERLEAVE,
)
from b12x.moe._shared.kernels.reference import _apply_gated_activation, moe_reference_w4a16_f32


def _pack_dense_fp4(dense: torch.Tensor) -> torch.Tensor:
    return pack_grouped_fp4_values(dense.float()).permute(2, 0, 1).contiguous()


def _blockscale_ones(groups: int, rows: int, cols: int) -> torch.Tensor:
    scales = torch.ones(groups, rows, cols // 16, dtype=torch.float32)
    return swizzle_block_scale(scales.to(torch.float8_e4m3fn))


def test_swigluoai_uninterleave_matches_minimax_default_torch_formula() -> None:
    gate = torch.tensor([8.0, -8.0, 0.5, 0.5], dtype=torch.float32)
    up = torch.tensor([0.25, 0.25, 8.0, -8.0], dtype=torch.float32)

    actual = _apply_gated_activation(
        gate,
        up,
        activation=SWIGLUOAI_UNINTERLEAVE,
        swiglu_limit=SWIGLUOAI_DEFAULT_LIMIT,
        swiglu_alpha=SWIGLUOAI_DEFAULT_ALPHA,
        swiglu_beta=SWIGLUOAI_DEFAULT_BETA,
    )

    gate_clamped = torch.clamp(gate, max=SWIGLUOAI_DEFAULT_LIMIT)
    up_clamped = torch.clamp(
        up,
        min=-SWIGLUOAI_DEFAULT_LIMIT,
        max=SWIGLUOAI_DEFAULT_LIMIT,
    )
    expected = (
        gate_clamped
        * torch.sigmoid(SWIGLUOAI_DEFAULT_ALPHA * gate_clamped)
        * (up_clamped + SWIGLUOAI_DEFAULT_BETA)
    )
    torch.testing.assert_close(actual, expected)

    wrong_lower_clamp = (
        torch.clamp(gate, min=-SWIGLUOAI_DEFAULT_LIMIT, max=SWIGLUOAI_DEFAULT_LIMIT)
        * torch.sigmoid(
            SWIGLUOAI_DEFAULT_ALPHA
            * torch.clamp(
                gate,
                min=-SWIGLUOAI_DEFAULT_LIMIT,
                max=SWIGLUOAI_DEFAULT_LIMIT,
            )
        )
        * (up_clamped + SWIGLUOAI_DEFAULT_BETA)
    )
    assert not torch.allclose(actual[1], wrong_lower_clamp[1])


def test_swigluoai_layout_sentinel_detects_gate_up_swap() -> None:
    gate = torch.tensor([2.0, -3.0], dtype=torch.float32)
    up = torch.tensor([-0.5, 1.5], dtype=torch.float32)
    correct = _apply_gated_activation(
        gate,
        up,
        activation=SWIGLUOAI_UNINTERLEAVE,
        swiglu_limit=SWIGLUOAI_DEFAULT_LIMIT,
        swiglu_alpha=SWIGLUOAI_DEFAULT_ALPHA,
        swiglu_beta=SWIGLUOAI_DEFAULT_BETA,
    )
    swapped = _apply_gated_activation(
        up,
        gate,
        activation=SWIGLUOAI_UNINTERLEAVE,
        swiglu_limit=SWIGLUOAI_DEFAULT_LIMIT,
        swiglu_alpha=SWIGLUOAI_DEFAULT_ALPHA,
        swiglu_beta=SWIGLUOAI_DEFAULT_BETA,
    )

    assert torch.max(torch.abs(correct - swapped)).item() > 1.0


def test_standard_silu_activation_is_unchanged() -> None:
    gate = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float32)
    up = torch.tensor([3.0, -1.0, 0.25, -2.0], dtype=torch.float32)

    actual = _apply_gated_activation(
        gate,
        up,
        activation="silu",
        swiglu_limit=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
    )
    expected = gate * torch.sigmoid(gate) * up
    torch.testing.assert_close(actual, expected)


def test_swigluoai_w4a16_f32_reference_matches_small_topk_torch_moe() -> None:
    experts, hidden, intermediate, topk = 3, 16, 16, 2
    x = torch.tensor(
        [
            [0.25, -0.5, 1.0, 0.5] * 4,
            [-0.25, 0.5, -1.0, 0.25] * 4,
        ],
        dtype=torch.bfloat16,
    )
    topk_ids = torch.tensor([[0, 2], [1, 0]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.75, 0.25], [0.6, 0.4]], dtype=torch.float32)

    values = torch.tensor([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5], dtype=torch.float32)
    w1_dense = torch.empty(experts, 2 * intermediate, hidden, dtype=torch.float32)
    w2_dense = torch.empty(experts, hidden, intermediate, dtype=torch.float32)
    for eid in range(experts):
        for row in range(2 * intermediate):
            for col in range(hidden):
                w1_dense[eid, row, col] = values[(eid + row + col) % values.numel()]
        for row in range(hidden):
            for col in range(intermediate):
                w2_dense[eid, row, col] = values[(2 * eid + row - col) % values.numel()]

    actual = moe_reference_w4a16_f32(
        x,
        _pack_dense_fp4(w1_dense),
        _blockscale_ones(experts, 2 * intermediate, hidden),
        torch.ones(experts, dtype=torch.float32),
        _pack_dense_fp4(w2_dense),
        _blockscale_ones(experts, hidden, intermediate),
        torch.ones(experts, dtype=torch.float32),
        topk_ids,
        topk_weights,
        experts,
        hidden,
        intermediate,
        activation=SWIGLUOAI_UNINTERLEAVE,
    )

    expected = torch.zeros(x.shape[0], hidden, dtype=torch.float32)
    for token_idx in range(x.shape[0]):
        x_row = x[token_idx].float()
        for route_idx in range(topk):
            eid = int(topk_ids[token_idx, route_idx].item())
            gate = w1_dense[eid, :intermediate] @ x_row
            up = w1_dense[eid, intermediate:] @ x_row
            act = _apply_gated_activation(
                gate,
                up,
                activation=SWIGLUOAI_UNINTERLEAVE,
                swiglu_limit=SWIGLUOAI_DEFAULT_LIMIT,
                swiglu_alpha=SWIGLUOAI_DEFAULT_ALPHA,
                swiglu_beta=SWIGLUOAI_DEFAULT_BETA,
            )
            down = w2_dense[eid] @ act
            expected[token_idx] += topk_weights[token_idx, route_idx].item() * down

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_fp4_binding_owns_swigluoai_params() -> None:
    hidden, intermediate, experts, topk = 16, 16, 1, 1
    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="modelopt_nvfp4",
        activation=SWIGLUOAI_UNINTERLEAVE,
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
    )
    layout = weight_plan.required_weight_layout("w4a16")
    assert layout is not None
    w1_fp4 = torch.zeros(
        experts, 2 * intermediate, hidden // 2, dtype=torch.uint8
    )
    w1_blockscale = torch.zeros(experts, 1, dtype=torch.uint8)
    w1_alphas = torch.ones(experts, dtype=torch.float32)
    w2_fp4 = torch.zeros(
        experts, hidden, intermediate // 2, dtype=torch.uint8
    )
    w2_blockscale = torch.zeros(experts, 1, dtype=torch.uint8)
    w2_alphas = torch.ones(experts, dtype=torch.float32)
    payload = SimpleNamespace(
        w13=w1_fp4,
        w13_scale=w1_blockscale,
        w13_global_scale=w1_alphas,
        w2=w2_fp4,
        w2_scale=w2_blockscale,
        w2_global_scale=w2_alphas,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
    )
    expert_weights = B12XFP4ExpertWeights(
        plan=weight_plan,
        a1_gscale=torch.ones(experts, dtype=torch.float32),
        w1_fp4=w1_fp4,
        w1_blockscale=w1_blockscale,
        w1_alphas=w1_alphas,
        a2_gscale=torch.ones(experts, dtype=torch.float32),
        w2_fp4=w2_fp4,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_alphas,
        representation=_PreparedWeightRepresentation(
            quant_mode="w4a16",
            layout=layout,
            value=payload,
        ),
    )
    binding = TPMoEFP4Binding(
        a=torch.zeros(1, hidden, dtype=torch.bfloat16),
        experts=expert_weights,
        topk_weights=torch.ones(1, topk, dtype=torch.float32),
        topk_ids=torch.zeros(1, topk, dtype=torch.int32),
        implementation="test",
        state_E=experts,
        weight_E=experts,
        max_rows=1,
        k=hidden,
        n=intermediate,
        num_topk=topk,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        swiglu_limit=SWIGLUOAI_DEFAULT_LIMIT,
        swiglu_alpha=SWIGLUOAI_DEFAULT_ALPHA,
        swiglu_beta=SWIGLUOAI_DEFAULT_BETA,
    )

    with pytest.raises(TypeError):
        b12x_moe_fp4(binding=binding, swiglu_alpha=2.0)
