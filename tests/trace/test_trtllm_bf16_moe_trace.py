"""Trace tests for TRT-LLM BF16 MoE."""

import torch


def _bf16_trace_kwargs():
    seq_len = 4
    num_experts = 2
    num_local_experts = 2
    hidden_size = 2
    intermediate_size = 1
    top_k = 1

    return dict(
        routing_logits=torch.zeros(seq_len, num_experts, dtype=torch.float32),
        routing_bias=None,
        hidden_states=torch.zeros(seq_len, hidden_size, dtype=torch.bfloat16),
        gemm1_weights=torch.zeros(
            num_local_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
        ),
        gemm2_weights=torch.zeros(
            num_local_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
        ),
        top_k=top_k,
        n_group=None,
        topk_group=None,
        local_expert_offset=0,
        routed_scaling_factor=None,
        routing_method_type=0,
        gemm1_alpha=torch.ones(num_local_experts, dtype=torch.float32),
        gemm1_beta=torch.zeros(num_local_experts, dtype=torch.float32),
        gemm1_clamp_limit=torch.full((num_local_experts,), 2.0, dtype=torch.float32),
    )


def _bf16_routed_trace_kwargs():
    kwargs = _bf16_trace_kwargs()
    kwargs.pop("routing_logits")
    kwargs.pop("routing_bias")
    kwargs["topk_ids"] = torch.zeros(4, 1, dtype=torch.int32)
    kwargs["num_experts"] = 2
    kwargs["intermediate_size"] = 1
    return kwargs


def test_bf16_moe_trace_schema_includes_swiglu_oa_params():
    from flashinfer.fused_moe import trtllm_bf16_moe, trtllm_bf16_routed_moe

    trace_defs = [
        trtllm_bf16_moe.fi_trace(**_bf16_trace_kwargs()),
        trtllm_bf16_routed_moe.fi_trace(**_bf16_routed_trace_kwargs()),
    ]

    for defn in trace_defs:
        assert defn["axes"]["num_local_experts"]["value"] == 2
        for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"):
            assert defn["inputs"][name]["shape"] == ["num_local_experts"]
            assert defn["inputs"][name]["dtype"] == "float32"
            assert defn["inputs"][name]["optional"] is True
            assert defn["inputs"][name]["description"]


def test_bf16_moe_trace_reference_applies_swiglu_oa_params():
    from flashinfer.trace.templates.moe import (
        trtllm_bf16_moe_trace,
        trtllm_bf16_routed_moe_trace,
    )

    routing_logits = torch.zeros(4, 1, dtype=torch.float32)
    hidden_states = torch.tensor(
        [[-3.0, -3.0], [-1.0, -0.5], [3.0, 4.0], [-4.0, 6.0]],
        dtype=torch.bfloat16,
    )
    gemm1_weights = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.tensor(
        [[[1.0], [0.0]]],
        dtype=torch.bfloat16,
    )

    common_kwargs = dict(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        num_experts=1,
        top_k=1,
        local_expert_offset=0,
        routed_scaling_factor=None,
    )

    default_out = trtllm_bf16_moe_trace.reference(**common_kwargs).to(torch.float32)

    large_limit = torch.full((1,), 1.0e9, dtype=torch.float32)
    noop_out = trtllm_bf16_moe_trace.reference(
        **common_kwargs,
        gemm1_alpha=torch.ones((1,), dtype=torch.float32),
        gemm1_beta=torch.zeros((1,), dtype=torch.float32),
        gemm1_clamp_limit=large_limit,
    ).to(torch.float32)

    clamp_limit = torch.full((1,), 2.0, dtype=torch.float32)
    clamp_only_out = trtllm_bf16_moe_trace.reference(
        **common_kwargs,
        gemm1_clamp_limit=clamp_limit,
    ).to(torch.float32)
    oa_out = trtllm_bf16_moe_trace.reference(
        **common_kwargs,
        gemm1_alpha=torch.full((1,), 1.702, dtype=torch.float32),
        gemm1_beta=torch.ones((1,), dtype=torch.float32),
        gemm1_clamp_limit=clamp_limit,
    ).to(torch.float32)
    routed_oa_out = trtllm_bf16_routed_moe_trace.reference(
        topk_ids=torch.zeros(4, 1, dtype=torch.int32),
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        num_experts=1,
        top_k=1,
        local_expert_offset=0,
        routed_scaling_factor=None,
        gemm1_alpha=torch.full((1,), 1.702, dtype=torch.float32),
        gemm1_beta=torch.ones((1,), dtype=torch.float32),
        gemm1_clamp_limit=clamp_limit,
    ).to(torch.float32)

    x1 = hidden_states[:, :1].to(torch.float32).clamp(min=-2.0, max=2.0)
    x2 = hidden_states[:, 1:].to(torch.float32).clamp(max=2.0)
    expected_clamp_only = x2 * torch.sigmoid(x2) * x1
    expected_oa = x2 * torch.sigmoid(1.702 * x2) * (x1 + 1.0)

    torch.testing.assert_close(default_out, noop_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        clamp_only_out[:, :1], expected_clamp_only.to(torch.bfloat16).to(torch.float32)
    )
    torch.testing.assert_close(
        oa_out[:, :1], expected_oa.to(torch.bfloat16).to(torch.float32)
    )
    torch.testing.assert_close(routed_oa_out, oa_out)
    assert not torch.allclose(default_out, clamp_only_out, atol=1e-2, rtol=1e-2)
    assert not torch.allclose(default_out, oa_out, atol=1e-2, rtol=1e-2)
