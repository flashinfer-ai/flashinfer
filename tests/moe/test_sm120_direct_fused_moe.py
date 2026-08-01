"""Correctness and CUDA Graph tests for SM120 direct BF16 fused MoE."""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.fused_moe import (
    sm120_direct_fused_moe,
    sm120_direct_fused_moe_workspace,
)
from flashinfer.trace.templates.moe import sm120_direct_fused_moe_trace


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="sm120_direct_fused_moe requires SM120",
)


def _make_case(num_tokens: int, intermediate_size: int, seed: int = 17):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden_size = 2048
    num_local_experts = 4
    num_global_experts = 16
    topk = 8
    hidden_states = (
        torch.randn(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.1
    ).contiguous()
    gemm1_weights = (
        torch.randn(
            num_local_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    ).contiguous()
    gemm2_weights = (
        torch.randn(
            num_local_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    ).contiguous()
    # Two local and six remote routes per token, matching EP=4 decode traffic.
    routing_rows = []
    for token in range(num_tokens):
        local = torch.tensor(
            [token % num_local_experts, (token + 1) % num_local_experts],
            dtype=torch.int32,
            device="cuda",
        )
        remote = torch.randperm(
            num_global_experts - num_local_experts,
            device="cuda",
            generator=generator,
        )[:6].to(torch.int32)
        routing_rows.append(torch.cat((local, remote + num_local_experts)))
    topk_ids = torch.stack(routing_rows).contiguous()
    topk_weights = torch.softmax(
        torch.randn(
            num_tokens,
            topk,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        ),
        dim=-1,
    ).contiguous()
    expert_map = torch.full((num_global_experts,), -1, dtype=torch.int32, device="cuda")
    expert_map[:num_local_experts] = torch.arange(
        num_local_experts, dtype=torch.int32, device="cuda"
    )
    return (
        hidden_states,
        topk_ids,
        topk_weights,
        gemm1_weights,
        gemm2_weights,
        expert_map,
    )


def _reference(
    hidden_states,
    topk_ids,
    topk_weights,
    gemm1_weights,
    gemm2_weights,
    expert_map,
):
    num_tokens, hidden_size = hidden_states.shape
    intermediate_size = gemm2_weights.shape[2]
    result = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device="cuda")
    hidden_fp32 = hidden_states.float()
    gemm1_fp32 = gemm1_weights.float()
    gemm2_fp32 = gemm2_weights.float()
    for token in range(num_tokens):
        for slot in range(topk_ids.shape[1]):
            global_expert = int(topk_ids[token, slot])
            local_expert = (
                global_expert
                if expert_map.numel() == 0
                else int(expert_map[global_expert])
            )
            if local_expert < 0:
                continue
            projection = torch.mv(gemm1_fp32[local_expert], hidden_fp32[token])
            up = projection[:intermediate_size]
            gate = projection[intermediate_size:]
            contribution = torch.mv(gemm2_fp32[local_expert], F.silu(gate) * up)
            result[token] += contribution * topk_weights[token, slot]
    return result.to(torch.bfloat16)


@pytest.mark.parametrize("num_tokens", range(1, 9))
@pytest.mark.parametrize("intermediate_size", [512, 768])
def test_sm120_direct_fused_moe_ep_accuracy(num_tokens, intermediate_size):
    args = _make_case(num_tokens, intermediate_size, seed=1701 + num_tokens)
    actual = sm120_direct_fused_moe(*args)
    expected = _reference(*args)
    torch.cuda.synchronize()

    actual_fp32 = actual.float().flatten()
    expected_fp32 = expected.float().flatten()
    cosine = F.cosine_similarity(actual_fp32, expected_fp32, dim=0)
    relative_l2 = torch.linalg.vector_norm(
        actual_fp32 - expected_fp32
    ) / torch.linalg.vector_norm(expected_fp32)
    assert float(cosine) > 0.999
    assert float(relative_l2) < 0.01


def test_sm120_direct_fused_moe_without_expert_map():
    args = list(_make_case(3, 512))
    args[1] = (args[1] % 4).contiguous()
    actual = sm120_direct_fused_moe(*args[:5])
    empty_map = torch.empty(0, dtype=torch.int32, device="cuda")
    expected = _reference(*args[:5], empty_map)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


def test_sm120_direct_fused_moe_cuda_graph():
    args = _make_case(8, 512, seed=47)
    hidden_states, topk_ids, _, _, _, _ = args
    workspace = sm120_direct_fused_moe_workspace(
        8, topk_ids.shape[1], 512, device="cuda"
    )
    output = torch.empty_like(hidden_states)
    for _ in range(3):
        sm120_direct_fused_moe(
            *args, output=output, workspace=workspace, skip_check=True
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = sm120_direct_fused_moe(
            *args, output=output, workspace=workspace, skip_check=True
        )
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference(*args)
    assert captured_output.data_ptr() == output.data_ptr()
    actual_fp32 = captured_output.float().flatten()
    expected_fp32 = expected.float().flatten()
    assert float(F.cosine_similarity(actual_fp32, expected_fp32, dim=0)) > 0.999


def test_sm120_direct_fused_moe_trace_schema():
    args = _make_case(2, 512)
    definition = sm120_direct_fused_moe.fi_trace(
        hidden_states=args[0],
        topk_ids=args[1],
        topk_weights=args[2],
        gemm1_weights=args[3],
        gemm2_weights=args[4],
        expert_map=args[5],
    )
    assert definition["op_type"] == "moe"
    assert definition["axes"]["num_tokens"] == {
        "type": "var",
        "description": "Decode batch size in [1, 8].",
    }
    assert definition["axes"]["hidden_size"]["value"] == 2048
    assert definition["outputs"]["output"]["shape"] == [
        "num_tokens",
        "hidden_size",
    ]


def test_sm120_direct_fused_moe_trace_reference():
    args = _make_case(2, 512, seed=71)
    expected = _reference(*args)
    trace_expected = sm120_direct_fused_moe_trace.reference(*args)
    torch.testing.assert_close(trace_expected, expected, rtol=0, atol=0)
