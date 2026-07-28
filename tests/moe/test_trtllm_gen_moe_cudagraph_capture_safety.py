"""CUDA-graph capture-safety regression test for trtllm-gen MoE routing scratch.

Guards flashinfer#3427 / #3168: the routing scratch (expanded_idx_to_permuted_idx
and siblings) must stay valid across CUDA-graph capture + replay. This test
captures N *separate* single-MoE graphs that share ONE graph mempool -- the
sglang piecewise-CUDA-graph structure, where each piece's routing scratch is
freed before the next piece captures, so the shared pool recycles that offset
across pieces. It runs at a token count that selects the cooperative routing
kernel (> 8192; the existing suite only covers <= 64 tokens, i.e. the
single-cluster kernel), then replays every piece and checks each replayed output
against its eager reference.

Note: the full illegal-memory-access reproduction of #3427/#3168 requires
sglang's torch.compile piecewise-cudagraph backend; a raw torch.cuda.graph
capture recycles the routing scratch but does not itself trigger the fault. This
test therefore guards capture+replay correctness of the cooperative-routing path
(new coverage) rather than reproducing the sglang-specific crash. It passes with
the persistent-arena fix and exercises exactly the code path the fix changes.
"""

import pytest
import torch

from flashinfer import RoutingMethodType, shuffle_matrix_a
from flashinfer.fused_moe import WeightLayout, convert_to_block_layout, trtllm_bf16_moe
from flashinfer.utils import device_support_pdl, get_compute_capability


def _build_shuffled_bf16_weights(num_experts, intermediate_size, hidden_size, device):
    gemm1 = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    gemm2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16
    )
    g1, g2 = [], []
    for i in range(num_experts):
        g1.append(
            convert_to_block_layout(
                shuffle_matrix_a(gemm1[i].view(torch.uint8), 64), 128
            )
        )
        g2.append(
            convert_to_block_layout(
                shuffle_matrix_a(gemm2[i].view(torch.uint8), 64), 128
            )
        )
    return torch.stack(g1).view(torch.bfloat16), torch.stack(g2).view(torch.bfloat16)


@pytest.mark.parametrize("num_tokens", [12288])  # > 8192 -> cooperative routing kernel
@pytest.mark.parametrize("pieces", [4])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("top_k", [8])
def test_trtllm_bf16_moe_cudagraph_capture_safety(
    num_tokens, pieces, num_experts, top_k
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("trtllm-gen MoE is only supported on SM100/SM103 GPUs.")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    hidden_size, intermediate_size = 1024, 1024
    enable_pdl = device_support_pdl(device)

    gemm1_weights, gemm2_weights = _build_shuffled_bf16_weights(
        num_experts, intermediate_size, hidden_size, device
    )

    def moe(routing_logits, hidden_states):
        return trtllm_bf16_moe(
            routing_logits=routing_logits,
            routing_bias=None,
            hidden_states=hidden_states,
            gemm1_weights=gemm1_weights,
            gemm2_weights=gemm2_weights,
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.Renormalize.value,
            use_shuffled_weight=True,
            weight_layout=WeightLayout.BlockMajorK,
            do_finalize=True,
            enable_pdl=enable_pdl,
        )

    # Distinct, persistent inputs per piece (persistent so each captured graph has
    # valid inputs at replay).
    inputs = []
    for _ in range(pieces):
        routing_logits = torch.rand(
            num_tokens, num_experts, device=device, dtype=torch.bfloat16
        )
        hidden_states = (torch.randn(num_tokens, hidden_size, device=device) * 0.1).to(
            torch.bfloat16
        )
        inputs.append((routing_logits, hidden_states))

    # Eager references (also warms the autotuner cache before capture).
    references = [moe(rl, h).to(torch.float32) for rl, h in inputs]
    torch.cuda.synchronize()

    # Capture each piece as a SEPARATE graph sharing ONE mempool. Each piece's
    # routing scratch is freed when the op returns, so the shared pool recycles
    # that offset across pieces -- the aliasing condition the fix guards against.
    pool = torch.cuda.graph_pool_handle()
    graphs = []
    for routing_logits, hidden_states in inputs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            out = moe(routing_logits, hidden_states)
        graphs.append((graph, out))
    torch.cuda.synchronize()

    for graph, _ in graphs:
        graph.replay()
    torch.cuda.synchronize()

    # Every replayed captured output must match its eager reference.
    for i, (_, out) in enumerate(graphs):
        torch.testing.assert_close(
            out.to(torch.float32), references[i], rtol=1e-2, atol=1e-2
        )
