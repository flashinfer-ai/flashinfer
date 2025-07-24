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

import math

import pytest
import torch
from torch.nn import functional as F

from flashinfer import (
    RoutingMethodType,
    e2m1_and_ufp8sf_scale_to_float,
    fp4_quantize,
    reorder_rows_for_gated_act_gemm,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe


def e2m1_and_ufp8_scale_to_float_tensor_v2(
        e2m1_tensor: torch.Tensor,
        ufp8_scale_tensor: torch.Tensor,
        global_scale_tensor: torch.Tensor,
        sf_vec_size,
        ufp8_type: int = 1,
        is_sf_swizzled_layout: bool = True,
):
    float_tensor = e2m1_and_ufp8sf_scale_to_float(
        e2m1_tensor.cpu(),
        ufp8_scale_tensor.cpu().reshape(-1),
        global_scale_tensor.cpu(),
        sf_vec_size,
        ufp8_type,
        is_sf_swizzled_layout,
    )
    return float_tensor




def quant_fp4(a, use_ue8m0=False, is_sf_swizzled_layout=True):
    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    sf_vec_size = 16

    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    return a_fp4, a_sf, a_global_sf


def quant_fp4_with_global_sf(
        a, a_global_sf, use_ue8m0=False, is_sf_swizzled_layout=True
):
    """
    Quantize FP4 with pre-calculated global scale factor.
    Used specifically for hidden states in CUDA graph capture to avoid runtime computation.
    """
    sf_vec_size = 16

    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    return a_fp4, a_sf, a_global_sf


def quant_fp4_batches(a, num_experts, use_ue8m0=False, is_sf_swizzled_layout=True):
    quant_a = []
    sfs = []
    global_sfs = []
    for i in range(num_experts):
        a_fp4, a_sf, a_global_sf = quant_fp4(a[i], use_ue8m0, is_sf_swizzled_layout)
        quant_a.append(a_fp4)
        sfs.append(a_sf)
        global_sfs.append(a_global_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)
    result_global_sfs = torch.stack(global_sfs)

    return result_quant_a, result_sfs, result_global_sfs


def quant_dequant_fp4(a, use_ue8m0=False, is_sf_swizzled_layout=True):
    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    sf_vec_size = 16

    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    a_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(
        a_fp4.cpu(), a_sf.cpu(), 1 / a_global_sf, sf_vec_size
    )

    return a_pt.cuda(), a_global_sf



def compute_moe_actual_with_routing(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        expert_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        top_k,
        padding,
        n_groups,
        top_k_groups,
        routed_scaling,
        routing_method_type,
        tile_tokens_dim,
        args_dequant,
        args,
):
    """
    Compute the actual MoE output using the optimized kernel with full routing support.

    Returns:
        output_dequant_actual: Actual output tensor from the kernel
    """

    def prepare_static_weights():
        """
        Handle all static weight-related preprocessing.
        This should be done once at model load time in production.

        Returns:
            Dict containing all preprocessed weight tensors and scale factors
        """
        use_ue8m0 = False
        epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

        # Quantize weights with linear layout for kernels
        _, gemm1_scales_linear_fp4_bytes, _ = quant_fp4_batches(
            gemm1_weights, num_experts, use_ue8m0, False
        )
        _, gemm2_scales_linear_fp4_bytes, _ = quant_fp4_batches(
            gemm2_weights, num_experts, use_ue8m0, False
        )

        # Convert quantized weights to proper formats
        gemm1_weights_fp4 = args.gemm1_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 2
        )  # packed fp4
        gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 16
        )  # fp8 scaling factors

        gemm2_weights_fp4 = args.gemm2_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // 2
        )  # packed fp4
        gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, hidden_size, intermediate_size // 16
        )  # fp8 scaling factors

        # Reorder rows of W1 and scales for fused gated activation
        gemm1_weights_fp4_interleaved = []
        gemm1_scales_fp4_interleaved = []
        for i in range(num_experts):
            gemm1_weights_fp4_interleaved.append(
                reorder_rows_for_gated_act_gemm(gemm1_weights_fp4[i].clone())
            )
            gemm1_scales_fp4_interleaved.append(
                reorder_rows_for_gated_act_gemm(gemm1_scales_linear_fp4[i].clone())
            )

        # Stack weights and scales for all experts
        gemm1_weights_fp4_interleaved = torch.stack(
            gemm1_weights_fp4_interleaved
        ).reshape(num_experts, 2 * intermediate_size, hidden_size // 2)
        gemm1_scales_fp4_interleaved = torch.stack(
            gemm1_scales_fp4_interleaved
        ).reshape(num_experts, 2 * intermediate_size, hidden_size // 16)

        # Shuffle weights and scaling factors for transposed mma output
        gemm1_weights_fp4_shuffled = []
        gemm1_scales_fp4_shuffled = []
        gemm2_weights_fp4_shuffled = []
        gemm2_scales_fp4_shuffled = []
        for i in range(num_experts):
            gemm1_weights_fp4_shuffled.append(
                shuffle_matrix_a(
                    gemm1_weights_fp4_interleaved[i].view(torch.uint8), epilogue_tile_m
                )
            )
            gemm1_scales_fp4_shuffled.append(
                shuffle_matrix_sf_a(
                    gemm1_scales_fp4_interleaved[i].view(torch.uint8), epilogue_tile_m
                )
            )

            gemm2_weights_fp4_shuffled.append(
                shuffle_matrix_a(
                    gemm2_weights_fp4[i].view(torch.uint8), epilogue_tile_m
                )
            )
            gemm2_scales_fp4_shuffled.append(
                shuffle_matrix_sf_a(
                    gemm2_scales_linear_fp4[i].view(torch.uint8), epilogue_tile_m
                )
            )

        # Stack weights for all experts
        gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
        gemm1_scales_fp4_shuffled = (
            torch.stack(gemm1_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, 2 * intermediate_size, hidden_size // 16)
        )

        gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
        gemm2_scales_fp4_shuffled = (
            torch.stack(gemm2_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, hidden_size, intermediate_size // 16)
        )

        # Calculate scaling factors that depend on weights
        scale_c_fc1 = (
                args_dequant.c_global_sf
                * (1.0 / args.gemm1_scales_global)
                * (1.0 / args.hidden_states_scale_global)
        )
        scale_gate_fc1 = (1.0 / args.gemm1_scales_global) * (
                1.0 / args.hidden_states_scale_global
        )
        scale_c_fc2 = (1.0 / args_dequant.c_global_sf) * (
                1.0 / args.gemm2_scales_global
        )

        return {
            "gemm1_weights_fp4_shuffled": gemm1_weights_fp4_shuffled,
            "gemm1_scales_fp4_shuffled": gemm1_scales_fp4_shuffled,
            "gemm2_weights_fp4_shuffled": gemm2_weights_fp4_shuffled,
            "gemm2_scales_fp4_shuffled": gemm2_scales_fp4_shuffled,
            "scale_c_fc1": scale_c_fc1,
            "scale_gate_fc1": scale_gate_fc1,
            "scale_c_fc2": scale_c_fc2,
        }

    # Process static weights (would be cached in production)
    static_data = prepare_static_weights()

    # Calculate global scale factor for hidden states offline (precalculated parameter)
    hidden_states_global_sf = (448 * 6) / hidden_states.float().abs().nan_to_num().max()

    use_ue8m0 = False
    do_finalize = True

    # Quantize hidden states with precalculated global scale
    hidden_states_fp4_bytes, hidden_states_scale_linear_fp4_bytes, _ = (
        quant_fp4_with_global_sf(
            hidden_states, hidden_states_global_sf, use_ue8m0, False
        )
    )
    hidden_states_fp4 = hidden_states_fp4_bytes.reshape(num_tokens, hidden_size // 2)
    hidden_states_scale_linear_fp4 = hidden_states_scale_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(-1)

    output = trtllm_fp4_block_scale_moe(
        expert_logits,
        routing_bias,
        hidden_states_fp4,
        hidden_states_scale_linear_fp4,
        static_data["gemm1_weights_fp4_shuffled"],
        static_data["gemm1_scales_fp4_shuffled"],
        static_data["gemm2_weights_fp4_shuffled"],
        static_data["gemm2_scales_fp4_shuffled"],
        static_data["scale_c_fc1"],
        static_data["scale_gate_fc1"],
        static_data["scale_c_fc2"],
        num_experts,
        top_k,
        n_groups,
        top_k_groups,
        intermediate_size,
        0,
        num_experts,
        routed_scaling,
        tile_tokens_dim,
        routing_method_type,
        do_finalize=True,
    )

    output_dequant_actual = output[0].to(torch.float)

    return output_dequant_actual


@pytest.mark.parametrize("num_tokens", [1, 1024, 4096])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [1024, 768, 384, 192])
@pytest.mark.parametrize(
    "routing_info",
    [
        pytest.param(
            {
                "num_experts": 256,
                "top_k": 8,
                "padding": 8,
                "n_groups": 8,
                "top_k_groups": 4,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
            },
            id="RoutingDSv3",
        ),
        pytest.param(
            {
                "num_experts": 72,
                "top_k": 6,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
            },
            id="RoutingDSlite",
        ),
        pytest.param(
            {
                "num_experts": 128,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Renormalize,
            },
            id="RoutingRenormalize",
        ),
        pytest.param(
            {
                "num_experts": 128,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.RenormalizeNaive,
            },
            id="RoutingRenormalizeNaive",
        ),
    ],
)
def test_moe_nvfp4(
        num_tokens,
        hidden_size,
        intermediate_size,
        routing_info,
):
    seed = 42
    torch.random.manual_seed(seed)

    # Extract routing configuration
    top_k = routing_info["top_k"]
    padding = routing_info["padding"]
    n_groups = routing_info["n_groups"]
    top_k_groups = routing_info["top_k_groups"]
    routed_scaling = routing_info["routed_scaling"]
    num_experts = routing_info["num_experts"]
    routing_method_type = routing_info["routing_method_type"]
    tile_tokens_dim = 8

    # Validation checks
    assert top_k <= num_experts
    assert top_k <= 8
    if (top_k_groups is not None) and (n_groups is not None):
        assert top_k_groups <= 4
        assert num_experts > n_groups
        assert num_experts % n_groups == 0
        assert num_experts % 4 == 0
        assert top_k < (top_k_groups * num_experts / n_groups)

    # Create expert logits based on routing method
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
            torch.float
        )
    elif (
            routing_method_type == RoutingMethodType.RenormalizeNaive
            or routing_method_type == RoutingMethodType.Renormalize
    ):
        expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
            torch.bfloat16
        )

    # Handle routing bias
    if routing_info["has_routing_bias"]:
        routing_bias = torch.randn(num_experts, device="cuda", dtype=torch.bfloat16)
    else:
        routing_bias = None

    hidden_states = 2 * torch.randn(
        (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Compute actual output using optimized kernel
    output_dequant_actual = compute_moe_actual_with_routing(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        expert_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        top_k,
        padding,
        n_groups,
        top_k_groups,
        routed_scaling,
        routing_method_type,
        tile_tokens_dim,
        args_dequant,
        args,
    )
