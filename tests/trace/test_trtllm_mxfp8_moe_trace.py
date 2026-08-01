"""Trace tests for TRT-LLM Gen MXFP8 block-scale MoE."""

import torch


def _mxfp8_trace_kwargs():
    seq_len = 4
    num_experts = 2
    num_local_experts = 2
    hidden_size = 128
    intermediate_size = 128
    top_k = 1

    return dict(
        routing_logits=torch.zeros(seq_len, num_experts, dtype=torch.float32),
        routing_bias=torch.zeros(num_experts, dtype=torch.bfloat16),
        hidden_states=torch.zeros(seq_len, hidden_size, dtype=torch.float8_e4m3fn),
        hidden_states_scale=torch.ones(
            hidden_size // 128, seq_len, dtype=torch.float32
        ),
        gemm1_weights=torch.zeros(
            num_local_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.float8_e4m3fn,
        ),
        gemm1_weights_scale=torch.ones(
            num_local_experts,
            (2 * intermediate_size) // 128,
            hidden_size // 128,
            dtype=torch.float32,
        ),
        gemm2_weights=torch.zeros(
            num_local_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.float8_e4m3fn,
        ),
        gemm2_weights_scale=torch.ones(
            num_local_experts,
            hidden_size // 128,
            intermediate_size // 128,
            dtype=torch.float32,
        ),
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_local_experts,
        routed_scaling_factor=1.0,
        routing_method_type=0,
        gemm1_alpha=torch.ones(num_local_experts, dtype=torch.float32),
        gemm1_beta=torch.zeros(num_local_experts, dtype=torch.float32),
        gemm1_clamp_limit=torch.full((num_local_experts,), 2.0, dtype=torch.float32),
    )


def _mxfp8_routed_trace_kwargs():
    kwargs = _mxfp8_trace_kwargs()
    kwargs.pop("routing_logits")
    kwargs["topk_ids"] = torch.zeros(4, 1, dtype=torch.int32)
    return kwargs


def _make_identity_mxfp8_inputs():
    seq_len = 4
    hidden_size = 128
    intermediate_size = 128

    hidden_states_bf16 = torch.zeros(seq_len, hidden_size, dtype=torch.bfloat16)
    hidden_states_bf16[:, 0] = torch.tensor([-3.0, -1.0, 3.0, -4.0])
    hidden_states_bf16[:, 1] = torch.tensor([-3.0, -0.5, 4.0, 6.0])
    hidden_states = hidden_states_bf16.to(torch.float8_e4m3fn)
    hidden_states_scale = torch.ones(1, seq_len, dtype=torch.float32)

    gemm1_weights_bf16 = torch.zeros(
        1, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16
    )
    gemm1_weights_bf16[0, 0, 0] = 1.0
    gemm1_weights_bf16[0, intermediate_size, 1] = 1.0
    gemm1_weights = gemm1_weights_bf16.to(torch.float8_e4m3fn)
    gemm1_weights_scale = torch.ones(1, 2, 1, dtype=torch.float32)

    gemm2_weights_bf16 = torch.zeros(
        1, hidden_size, intermediate_size, dtype=torch.bfloat16
    )
    gemm2_weights_bf16[0, 0, 0] = 1.0
    gemm2_weights = gemm2_weights_bf16.to(torch.float8_e4m3fn)
    gemm2_weights_scale = torch.ones(1, 1, 1, dtype=torch.float32)

    return dict(
        routing_logits=torch.zeros(seq_len, 1, dtype=torch.float32),
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        num_experts=1,
        top_k=1,
        local_expert_offset=0,
        routed_scaling_factor=1.0,
        hidden_states_bf16=hidden_states_bf16,
    )


def test_mxfp8_moe_trace_schema_includes_swiglu_oa_params():
    from flashinfer.fused_moe import (
        trtllm_fp8_block_scale_moe,
        trtllm_fp8_block_scale_routed_moe,
    )

    trace_defs = [
        trtllm_fp8_block_scale_moe.fi_trace(**_mxfp8_trace_kwargs()),
        trtllm_fp8_block_scale_routed_moe.fi_trace(**_mxfp8_routed_trace_kwargs()),
    ]

    for defn in trace_defs:
        assert defn["axes"]["num_local_experts"]["value"] == 2
        for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"):
            assert defn["inputs"][name]["shape"] == ["num_local_experts"]
            assert defn["inputs"][name]["dtype"] == "float32"
            assert defn["inputs"][name]["optional"] is True
            assert defn["inputs"][name]["description"]


def test_mxfp8_moe_trace_reference_applies_swiglu_oa_params():
    from flashinfer.trace.templates.moe import (
        trtllm_fp8_block_scale_moe_default_routing_trace,
        trtllm_fp8_block_scale_routed_moe_trace,
    )

    inputs = _make_identity_mxfp8_inputs()
    hidden_states_bf16 = inputs.pop("hidden_states_bf16")
    default_inputs = dict(inputs)
    default_inputs.pop("num_experts")

    default_out = trtllm_fp8_block_scale_moe_default_routing_trace.reference(
        **default_inputs
    ).to(torch.float32)

    large_limit = torch.full((1,), 1.0e9, dtype=torch.float32)
    noop_out = trtllm_fp8_block_scale_moe_default_routing_trace.reference(
        **default_inputs,
        gemm1_alpha=torch.ones((1,), dtype=torch.float32),
        gemm1_beta=torch.zeros((1,), dtype=torch.float32),
        gemm1_clamp_limit=large_limit,
    ).to(torch.float32)

    clamp_limit = torch.full((1,), 2.0, dtype=torch.float32)
    clamp_only_out = trtllm_fp8_block_scale_moe_default_routing_trace.reference(
        **default_inputs,
        gemm1_clamp_limit=clamp_limit,
    ).to(torch.float32)
    oa_out = trtllm_fp8_block_scale_moe_default_routing_trace.reference(
        **default_inputs,
        gemm1_alpha=torch.full((1,), 1.702, dtype=torch.float32),
        gemm1_beta=torch.ones((1,), dtype=torch.float32),
        gemm1_clamp_limit=clamp_limit,
    ).to(torch.float32)

    routed_inputs = dict(inputs)
    routed_inputs.pop("routing_logits")
    routed_inputs.pop("routing_bias")
    routed_oa_out = trtllm_fp8_block_scale_routed_moe_trace.reference(
        topk_ids=torch.zeros(4, 1, dtype=torch.int32),
        intermediate_size=128,
        local_num_experts=1,
        routing_method_type=0,
        **routed_inputs,
        gemm1_alpha=torch.full((1,), 1.702, dtype=torch.float32),
        gemm1_beta=torch.ones((1,), dtype=torch.float32),
        gemm1_clamp_limit=clamp_limit,
    ).to(torch.float32)

    x1 = hidden_states_bf16[:, :1].to(torch.float32).clamp(min=-2.0, max=2.0)
    x2 = hidden_states_bf16[:, 1:2].to(torch.float32).clamp(max=2.0)
    expected_clamp_only = x2 * torch.sigmoid(x2) * x1
    expected_oa = x2 * torch.sigmoid(1.702 * x2) * (x1 + 1.0)

    torch.testing.assert_close(default_out, noop_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        clamp_only_out[:, :1],
        expected_clamp_only.to(torch.bfloat16).to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        oa_out[:, :1],
        expected_oa.to(torch.bfloat16).to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(routed_oa_out, oa_out, atol=1e-2, rtol=1e-2)
    assert not torch.allclose(default_out, clamp_only_out, atol=1e-2, rtol=1e-2)
    assert not torch.allclose(default_out, oa_out, atol=1e-2, rtol=1e-2)
