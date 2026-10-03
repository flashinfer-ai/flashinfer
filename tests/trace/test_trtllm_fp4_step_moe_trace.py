# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Standalone NVFP4 Step references retain physical clipping and token scales."""

import pytest
import torch


@pytest.mark.parametrize("limit", [None, 0.25, 16.0])
@pytest.mark.parametrize("per_token", [False, True])
def test_nvfp4_step_serialized_reference(limit, per_token):
    from flashinfer import ActivationType
    from flashinfer.fused_moe import (
        trtllm_fp4_block_scale_moe,
        trtllm_fp4_block_scale_routed_moe,
    )

    tokens, hidden, intermediate = 4, 128, 128
    x = torch.zeros(tokens, hidden // 2, dtype=torch.uint8)
    # E2M1 low nibble is up; high nibble is gate.
    x[:, 0] = torch.tensor([0x7F, 0x6C, 0x44, 0xA7], dtype=torch.uint8)
    w1 = torch.zeros(1, 2 * intermediate, hidden // 2, dtype=torch.uint8)
    w1[0, 0, 0] = 0x02  # FC1 up selects input channel 0 with weight 1.
    w1[0, intermediate, 0] = 0x20  # FC1 gate selects channel 1.
    w2 = torch.zeros(1, hidden, intermediate // 2, dtype=torch.uint8)
    w2[0, 0, 0] = 0x02
    token_scales = torch.tensor([0.5, 2.0, 1.0, 4.0]) if per_token else None
    physical_limit = 7.0 if limit is None else limit
    kwargs = dict(
        routing_logits=torch.zeros(tokens, 1),
        routing_bias=None,
        hidden_states=x,
        hidden_states_scale=torch.ones(tokens, hidden // 16, dtype=torch.float8_e4m3fn),
        gemm1_weights=w1,
        gemm1_weights_scale=torch.ones(
            1, 2 * intermediate, hidden // 16, dtype=torch.float8_e4m3fn
        ),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None if limit is None else torch.tensor([limit / 2.0]),
        gemm2_weights=w2,
        gemm2_weights_scale=torch.ones(
            1, hidden, intermediate // 16, dtype=torch.float8_e4m3fn
        ),
        gemm2_bias=None,
        output1_scale_scalar=torch.tensor([2.0]),
        output1_scale_gate_scalar=torch.tensor([2.0]),
        output2_scale_scalar=torch.ones(1),
        per_token_scale=token_scales,
        num_experts=1,
        top_k=1,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate,
        local_expert_offset=0,
        local_num_experts=1,
        routed_scaling_factor=None,
        routing_method_type=0,
        activation_type=ActivationType.SwigluStep.value,
    )
    up = torch.tensor([-6.0, -2.0, 2.0, 6.0]) * 2.0
    gate = torch.tensor([6.0, 4.0, 2.0, -1.0]) * 2.0
    if token_scales is not None:
        up *= token_scales
        gate *= token_scales
    expected = torch.zeros(tokens, hidden, dtype=torch.bfloat16)
    expected[:, 0] = (
        up.clamp(-physical_limit, physical_limit)
        * torch.nn.functional.silu(gate).clamp(max=physical_limit)
    ).to(expected.dtype)
    for api, name in (
        (
            trtllm_fp4_block_scale_moe,
            "_trtllm_fp4_block_scale_moe_default_routing_reference",
        ),
        (
            trtllm_fp4_block_scale_routed_moe,
            "_trtllm_fp4_block_scale_routed_moe_reference",
        ),
    ):
        case = dict(kwargs)
        if api is trtllm_fp4_block_scale_routed_moe:
            case.pop("routing_logits")
            case["topk_ids"] = torch.full((tokens, 1), 0x3F80, dtype=torch.int32)
        definition = api.fi_trace(**case)
        assert definition["axes"]["activation_type"]["value"] == 7
        namespace = {}
        exec(definition["reference"], namespace)  # noqa: S102
        actual = namespace[name](**case)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
