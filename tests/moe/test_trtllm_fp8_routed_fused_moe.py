"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

from flashinfer import (
    RoutingMethodType,
)
from flashinfer.fused_moe import (
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_block_scale_routed_moe,
)
from flashinfer.utils import device_support_pdl

from .test_trtllm_gen_fused_moe import (
    routing_reference_renormalize,
    routing_reference_renormalize_naive,
    routing_reference_topk,
)

from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("num_tokens", [1, 8, 1024])
@pytest.mark.parametrize("hidden_size", [1024, 2048, 3072, 4096])
@pytest.mark.parametrize("intermediate_size", [1024, 2048, 3072, 4096])
@pytest.mark.parametrize("num_experts", [128, 256])
@pytest.mark.parametrize("top_k", [4, 8])
@pytest.mark.parametrize(
    "routing_method_type",
    [
        RoutingMethodType.Renormalize,
        RoutingMethodType.RenormalizeNaive,
        RoutingMethodType.TopK,
    ],
)
def test_trtllm_fp8_routed_fused_moe(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_experts: int,
    routing_method_type: RoutingMethodType,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)
    routing_logits = torch.rand(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )

    # Create FP8 hidden states and scales
    hidden_states = torch.randn(num_tokens, hidden_size, device=device).to(
        torch.float8_e4m3fn
    )
    # Block scale: [hidden_size//128, num_tokens]
    hidden_states_scale = torch.rand(
        hidden_size // 128, num_tokens, device=device, dtype=torch.float32
    )

    # Create FP8 weights and scales
    gemm1_weights = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device
    ).to(torch.float8_e4m3fn)
    gemm1_weights_scale = torch.rand(
        num_experts,
        intermediate_size * 2 // 128,
        hidden_size // 128,
        device=device,
        dtype=torch.float32,
    )

    gemm2_weights = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device
    ).to(torch.float8_e4m3fn)
    gemm2_weights_scale = torch.rand(
        num_experts,
        hidden_size // 128,
        intermediate_size // 128,
        device=device,
        dtype=torch.float32,
    )

    # Run the non-routed version as reference
    reference_output = trtllm_fp8_block_scale_moe(
        routing_logits,
        None,  # routing_bias
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        intermediate_size,
        0,  # local_expert_offset
        num_experts,
        None,  # routed_scaling_factor
        routing_method_type.value,
        False,  # use_shuffled_weight
        0,  # weight_layout
        enable_pdl,
    ).to(torch.float)

    # Compute routing for routed version
    if routing_method_type == RoutingMethodType.Renormalize:
        permute_info, expert_weights = routing_reference_renormalize(
            routing_logits, top_k, num_experts, 8
        )
    elif routing_method_type == RoutingMethodType.RenormalizeNaive:
        permute_info, expert_weights = routing_reference_renormalize_naive(
            routing_logits, top_k, num_experts, 8
        )
    elif routing_method_type == RoutingMethodType.TopK:
        permute_info, expert_weights = routing_reference_topk(
            routing_logits, top_k, num_experts, 8
        )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    expert_weights = expert_weights.view(num_tokens, num_experts)[
        torch.arange(num_tokens).unsqueeze(1), topk_ids
    ].to(torch.bfloat16)

    # Pack topk_ids and expert_weights into a single tensor
    packed_tensor = (topk_ids.to(torch.int32) << 16) | expert_weights.to(
        torch.bfloat16
    ).view(torch.int16)

    # Run the routed version
    output = trtllm_fp8_block_scale_routed_moe(
        packed_tensor,
        None,  # routing_bias
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        intermediate_size,
        0,  # local_expert_offset
        num_experts,
        None,  # routed_scaling_factor
        routing_method_type.value,
        False,  # use_shuffled_weight
        0,  # weight_layout
        enable_pdl,
    ).to(torch.float)

    # Compare outputs
    mask = torch.isclose(output, reference_output, rtol=1e-3, atol=1e-3)

    # mismatch percentage
    mismatch_pct = (~mask).float().mean().item() * 100
    assert mismatch_pct < 6, f"Mismatch percentage is {mismatch_pct:.2f}"
