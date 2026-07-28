"""Trace tests for routed TRT-LLM FP8 per-tensor-scale MoE."""

import torch


def _trace_kwargs():
    seq_len = 2
    num_experts = 8
    num_local_experts = 2
    hidden_size = 1
    intermediate_size = 1
    top_k = 1
    local_expert_offset = 4

    topk_ids = torch.tensor([[4], [5]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.25], [0.75]], dtype=torch.bfloat16)
    packed_topk_ids = (topk_ids << 16) | topk_weights.view(torch.int16).to(torch.int32)
    gemm1_weights = torch.tensor(
        [[[1.0], [1.0]], [[2.0], [1.0]]], dtype=torch.float8_e4m3fn
    )

    return dict(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=torch.ones(seq_len, hidden_size, dtype=torch.float8_e4m3fn),
        gemm1_weights=gemm1_weights,
        output1_scales_scalar=torch.ones(num_local_experts, dtype=torch.float32),
        output1_scales_gate_scalar=torch.ones(num_local_experts, dtype=torch.float32),
        gemm2_weights=torch.ones(
            num_local_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.float8_e4m3fn,
        ),
        output2_scales_scalar=torch.ones(num_local_experts, dtype=torch.float32),
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=num_local_experts,
        routed_scaling_factor=None,
        use_routing_scales_on_input=False,
        routing_method_type=1,
    )


def test_fp8_per_tensor_routed_moe_trace_schema():
    from flashinfer.fused_moe import trtllm_fp8_per_tensor_scale_routed_moe

    defn = trtllm_fp8_per_tensor_scale_routed_moe.fi_trace(**_trace_kwargs())

    assert defn["axes"]["num_experts"]["type"] == "var"
    assert defn["axes"]["top_k"]["value"] == 1
    assert defn["axes"]["num_local_experts"]["value"] == 2
    assert defn["axes"]["hidden_size"]["value"] == 1
    assert defn["axes"]["intermediate_size"]["type"] == "var"
    assert defn["inputs"]["topk_ids"]["dtype"] == "int32"
    assert defn["inputs"]["topk_ids"]["shape"] == ["seq_len", "top_k"]
    assert defn["outputs"]["output"]["dtype"] == "bfloat16"
    assert "reference" in defn


def test_fp8_per_tensor_routed_moe_trace_reference_unpacks_routing():
    from flashinfer.fused_moe import trtllm_fp8_per_tensor_scale_routed_moe

    kwargs = _trace_kwargs()
    defn = trtllm_fp8_per_tensor_scale_routed_moe.fi_trace(**kwargs)
    namespace = {}
    exec(defn["reference"], namespace)  # noqa: S102
    output = namespace["_trtllm_fp8_per_tensor_scale_routed_moe_reference"](**kwargs)

    sigmoid_1 = torch.sigmoid(torch.tensor(1.0, dtype=torch.float32))
    sigmoid_2 = torch.sigmoid(torch.tensor(2.0, dtype=torch.float32))
    expected = torch.tensor(
        [[0.25 * sigmoid_1], [1.5 * sigmoid_2]], dtype=torch.bfloat16
    )
    torch.testing.assert_close(output, expected)
