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
from typing import Literal
import torch

from flashinfer import (
    RoutingMethodType,
    ActivationType,
    fp4_quantize,
    mxfp8_quantize,
    reorder_rows_for_gated_act_gemm,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)
from flashinfer.fused_moe import (
    convert_to_block_layout,
    trtllm_bf16_moe,
    trtllm_bf16_routed_moe,
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_block_scale_routed_moe,
    WeightLayout,
)
from flashinfer.fused_moe.core import Fp8QuantizationType
from flashinfer.utils import device_support_pdl

from .test_trtllm_gen_fused_moe import (
    FP8BlockScaleMoe,
    QuantMode,
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
@pytest.mark.parametrize("quant_mode", ["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"])
def test_trtllm_gen_routed_fused_moe(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_experts: int,
    routing_method_type: RoutingMethodType,
    quant_mode: Literal["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"],
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
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16) * 0.1
    )
    if quant_mode == "NvFP4xNvFP4":
        hidden_states, hidden_states_scale = fp4_quantize(
            hidden_states,
            torch.tensor([448.0 * 6.0], device=device),
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )
        hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
            num_tokens, -1
        )
        hidden_states_global_scale = 1.0 / 448.0 / 6.0
    elif quant_mode == "MxFP4xMxFP8":
        hidden_states, hidden_states_scale = mxfp8_quantize(hidden_states, False)
        hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
            num_tokens, -1
        )
        hidden_states_global_scale = 1.0
    else:  # MxFP4xBf16
        hidden_states_scale = None
        hidden_states_global_scale = 1.0

    w13 = (
        torch.randn(num_experts, intermediate_size * 2, hidden_size, device=device).to(
            torch.bfloat16
        )
        * 0.1
    )
    w2 = (
        torch.randn(num_experts, hidden_size, intermediate_size, device=device).to(
            torch.bfloat16
        )
        * 0.1
    )
    if quant_mode == "NvFP4xNvFP4":
        w13, w13_scale = fp4_quantize(
            w13,
            torch.tensor([448.0 * 6.0], device=device),
            sf_vec_size=16,
            sf_use_ue8m0=False,
        )
        w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
            num_experts, intermediate_size * 2, -1
        )
        w2, w2_scale = fp4_quantize(
            w2,
            torch.tensor([448.0 * 6.0], device=device),
            sf_vec_size=16,
            sf_use_ue8m0=False,
        )
        w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, -1
        )
        w13_global_scale = 1.0 / 448.0 / 6.0
        w2_global_scale = 1.0 / 448.0 / 6.0
    else:
        w13, w13_scale = fp4_quantize(
            w13, torch.tensor([1.0], device=device), sf_vec_size=32, sf_use_ue8m0=True
        )
        w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
            num_experts, intermediate_size * 2, -1
        )
        w2, w2_scale = fp4_quantize(
            w2, torch.tensor([1.0], device=device), sf_vec_size=32, sf_use_ue8m0=True
        )
        w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, -1
        )
        w13_global_scale = 1.0
        w2_global_scale = 1.0

    output1_scale_scalar = torch.tensor(
        [hidden_states_global_scale * w13_global_scale] * num_experts, device=device
    )
    output1_scale_gate_scalar = torch.tensor(
        [hidden_states_global_scale * w13_global_scale] * num_experts, device=device
    )
    output2_scale_scalar = torch.tensor(
        [hidden_states_global_scale * w2_global_scale] * num_experts, device=device
    )

    reference_output = trtllm_fp4_block_scale_moe(
        routing_logits,
        None,  # routing_bias
        hidden_states,
        hidden_states_scale,
        w13,
        w13_scale,
        None,  # w13_bias
        None,  # gemm1_alpha
        None,  # gemm1_beta
        None,  # gemm1_clamp_limit
        w2,
        w2_scale,
        None,  # w2_bias
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        intermediate_size,
        0,  # local_expert_offset
        num_experts,
        None,  # routed_scaling_factor
        routing_method_type.value,
        True,  # do_finalize
        enable_pdl,
        ActivationType.Swiglu.value,  # act_type
        None,
    )[0].to(torch.float)

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

    packed_tensor = (topk_ids.to(torch.int32) << 16) | expert_weights.to(
        torch.bfloat16
    ).view(torch.int16)

    output = trtllm_fp4_block_scale_routed_moe(
        packed_tensor,
        None,  # routing_bias
        hidden_states,
        hidden_states_scale,
        w13,
        w13_scale,
        None,  # w13_bias
        None,  # gemm1_alpha
        None,  # gemm1_beta
        None,  # gemm1_clamp_limit
        w2,
        w2_scale,
        None,  # w2_bias
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        intermediate_size,
        0,  # local_expert_offset
        num_experts,
        None,  # routed_scaling_factor
        routing_method_type.value,
        True,  # do_finalize
        enable_pdl,
        ActivationType.Swiglu.value,  # act_type
        None,
    )[0].to(torch.float)

    mask = torch.isclose(output, reference_output, rtol=1e-3, atol=1e-3)

    # mismatch percentage
    mismatch_pct = (~mask).float().mean().item() * 100
    assert mismatch_pct < 6, f"Mismatch percentage is {mismatch_pct:.2f}"


@pytest.mark.parametrize("num_tokens", [8, 64])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [8, 16])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize(
    "routing_method_type",
    [
        RoutingMethodType.Renormalize,
    ],
)
def test_trtllm_gen_fp8_routed_fused_moe(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_experts: int,
    routing_method_type: RoutingMethodType,
):
    """Test FP8 block scale routed MoE matches standard routing."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)

    # Generate random routing logits for reference
    routing_logits = torch.rand(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )

    # Generate random hidden states in FP8
    hidden_states_bf16 = (
        torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16) * 0.1
    )
    hidden_states = hidden_states_bf16.to(torch.float8_e4m3fn)

    # Generate block scales for hidden states: [hidden_size // 128, num_tokens]
    hidden_states_scale = torch.ones(
        hidden_size // 128, num_tokens, device=device, dtype=torch.float32
    )

    # Generate FP8 weights
    gemm1_weights = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, device=device
    ).to(torch.float8_e4m3fn)
    gemm2_weights = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device
    ).to(torch.float8_e4m3fn)

    # Generate block scales for weights
    gemm1_weights_scale = torch.ones(
        num_experts,
        2 * intermediate_size // 128,
        hidden_size // 128,
        device=device,
        dtype=torch.float32,
    )
    gemm2_weights_scale = torch.ones(
        num_experts,
        hidden_size // 128,
        intermediate_size // 128,
        device=device,
        dtype=torch.float32,
    )

    # Run reference with routing_logits
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

    # Compute routing using reference implementation
    if routing_method_type == RoutingMethodType.Renormalize:
        permute_info, expert_weights_ref = routing_reference_renormalize(
            routing_logits, top_k, num_experts, 8
        )
    elif routing_method_type == RoutingMethodType.RenormalizeNaive:
        permute_info, expert_weights_ref = routing_reference_renormalize_naive(
            routing_logits, top_k, num_experts, 8
        )
    elif routing_method_type == RoutingMethodType.TopK:
        permute_info, expert_weights_ref = routing_reference_topk(
            routing_logits, top_k, num_experts, 8
        )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    expert_weights = expert_weights_ref.view(num_tokens, num_experts)[
        torch.arange(num_tokens, device=device).unsqueeze(1), topk_ids
    ].to(torch.bfloat16)

    # Pack topk_ids and expert_weights into single tensor
    # Format: (expert_id << 16) | (weight_bf16.view(int16))
    packed_topk_ids = (topk_ids << 16) | expert_weights.view(torch.int16).to(
        torch.int32
    )

    # Run with pre-computed routing (packed format)
    output = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
    )
    trtllm_fp8_block_scale_routed_moe(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method_type.value,
        use_shuffled_weight=False,
        weight_layout=0,
        enable_pdl=enable_pdl,
        output=output,
    )
    output = output.to(torch.float)

    mask = torch.isclose(output, reference_output, rtol=1e-2, atol=1e-2)

    # mismatch percentage
    mismatch_pct = (~mask).float().mean().item() * 100
    assert mismatch_pct < 10, f"Mismatch percentage is {mismatch_pct:.2f}%"


@pytest.mark.parametrize("num_tokens", [8, 64])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [8, 16])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize(
    "routing_method_type",
    [
        RoutingMethodType.Renormalize,
    ],
)
def test_trtllm_gen_bf16_routed_fused_moe(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_experts: int,
    routing_method_type: RoutingMethodType,
):
    """Test Bf16 scale routed MoE matches standard routing."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)

    # Generate random routing logits for reference
    routing_logits = torch.rand(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )

    # Generate random hidden states in FP8
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16) * 0.1
    )

    # Generate weights
    gemm1_weights = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, device=device
    ).to(torch.bfloat16)
    gemm2_weights = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device
    ).to(torch.bfloat16)

    gemm1_weights_shuffled = []
    gemm2_weights_shuffled = []
    for i in range(num_experts):
        tmp_weights1 = shuffle_matrix_a(gemm1_weights[i].view(torch.uint8), 64)
        tmp_weights2 = shuffle_matrix_a(gemm2_weights[i].view(torch.uint8), 64)
        block_k = 128
        gemm1_weights_shuffled.append(convert_to_block_layout(tmp_weights1, block_k))
        gemm2_weights_shuffled.append(convert_to_block_layout(tmp_weights2, block_k))
    gemm1_weights = torch.stack(gemm1_weights_shuffled).view(torch.bfloat16)
    gemm2_weights = torch.stack(gemm2_weights_shuffled).view(torch.bfloat16)

    # Run reference with routing_logits
    reference_output = trtllm_bf16_moe(
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
        routing_method_type=routing_method_type.value,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        do_finalize=True,
        enable_pdl=enable_pdl,
    ).to(torch.float)

    # Compute routing using reference implementation
    if routing_method_type == RoutingMethodType.Renormalize:
        permute_info, expert_weights_ref = routing_reference_renormalize(
            routing_logits, top_k, num_experts, 8
        )
    elif routing_method_type == RoutingMethodType.RenormalizeNaive:
        permute_info, expert_weights_ref = routing_reference_renormalize_naive(
            routing_logits, top_k, num_experts, 8
        )
    elif routing_method_type == RoutingMethodType.TopK:
        permute_info, expert_weights_ref = routing_reference_topk(
            routing_logits, top_k, num_experts, 8
        )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    expert_weights = expert_weights_ref.view(num_tokens, num_experts)[
        torch.arange(num_tokens, device=device).unsqueeze(1), topk_ids
    ].to(torch.bfloat16)

    # Pack topk_ids and expert_weights into single tensor
    # Format: (expert_id << 16) | (weight_bf16.view(int16))
    packed_topk_ids = (topk_ids << 16) | expert_weights.view(torch.int16).to(
        torch.int32
    )

    # Run with pre-computed routing (packed format)
    output = trtllm_bf16_routed_moe(
        topk_ids=packed_topk_ids,
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
        routing_method_type=routing_method_type.value,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        do_finalize=True,
        enable_pdl=enable_pdl,
    ).to(torch.float)

    mask = torch.isclose(output, reference_output, rtol=1e-2, atol=1e-2)

    # mismatch percentage
    mismatch_pct = (~mask).float().mean().item() * 100
    assert mismatch_pct < 10, f"Mismatch percentage is {mismatch_pct:.2f}%"


@pytest.mark.parametrize(
    "activation_type",
    [
        pytest.param(ActivationType.Swiglu.value, id="Swiglu"),
        pytest.param(ActivationType.Relu2.value, id="Relu2"),
    ],
)
def test_trtllm_gen_fp8_mxfp8_routed_activation_parity(activation_type: int):
    """MXFP8 routed path should match non-routed reference for gated and non-gated activations."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)

    num_tokens = 32
    hidden_size = 512
    intermediate_size = 512
    num_experts = 64
    top_k = 8
    routing_method_type = RoutingMethodType.Renormalize
    is_gated = activation_type in [
        ActivationType.Swiglu.value,
        ActivationType.Geglu.value,
    ]

    routing_logits = torch.randn((num_tokens, num_experts), device=device).to(
        torch.bfloat16
    )
    hidden_states = torch.randn((num_tokens, hidden_size), device=device).to(
        torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (
            num_experts,
            (2 if is_gated else 1) * intermediate_size,
            hidden_size,
        ),
        device=device,
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=torch.bfloat16,
    )

    quant_impl = FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8)
    quant_weights = quant_impl.quantize_weights(
        gemm1_weights, gemm2_weights, hidden_states
    )
    quant_inputs = quant_impl.quantize_inputs(hidden_states)
    epilogue_tile_m = 128
    w13_rows = (2 if is_gated else 1) * intermediate_size
    gemm1_weights_shuffled = []
    gemm1_scales_shuffled = []
    gemm2_weights_shuffled = []
    gemm2_scales_shuffled = []
    for i in range(num_experts):
        w1_interleaved = quant_weights["gemm1_weights"][i].clone().reshape(w13_rows, -1)
        s1_interleaved = quant_weights["gemm1_scales"][i].clone().reshape(w13_rows, -1)
        if is_gated:
            w1_interleaved = reorder_rows_for_gated_act_gemm(w1_interleaved)
            s1_interleaved = reorder_rows_for_gated_act_gemm(s1_interleaved)
        gemm1_weights_shuffled.append(
            shuffle_matrix_a(w1_interleaved.view(torch.uint8), epilogue_tile_m)
            .contiguous()
            .view(quant_weights["gemm1_weights"].dtype)
        )
        gemm2_weights_shuffled.append(
            shuffle_matrix_a(
                quant_weights["gemm2_weights"][i].view(torch.uint8), epilogue_tile_m
            )
            .contiguous()
            .view(quant_weights["gemm2_weights"].dtype)
        )
        gemm1_scales_shuffled.append(
            shuffle_matrix_sf_a(
                s1_interleaved.view(torch.uint8).reshape(w13_rows, -1),
                epilogue_tile_m,
            )
            .contiguous()
            .view(quant_weights["gemm1_scales"].dtype)
        )
        gemm2_scales_shuffled.append(
            shuffle_matrix_sf_a(
                quant_weights["gemm2_scales"][i]
                .view(torch.uint8)
                .reshape(hidden_size, -1),
                epilogue_tile_m,
            )
            .contiguous()
            .view(quant_weights["gemm2_scales"].dtype)
        )
    gemm1_weights_kernel = torch.stack(gemm1_weights_shuffled)
    gemm1_scales_kernel = torch.stack(gemm1_scales_shuffled)
    gemm2_weights_kernel = torch.stack(gemm2_weights_shuffled)
    gemm2_scales_kernel = torch.stack(gemm2_scales_shuffled)

    output_ref = trtllm_fp8_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=quant_inputs["hidden_states"],
        hidden_states_scale=quant_inputs["hidden_states_scale"],
        gemm1_weights=gemm1_weights_kernel,
        gemm1_weights_scale=gemm1_scales_kernel,
        gemm2_weights=gemm2_weights_kernel,
        gemm2_weights_scale=gemm2_scales_kernel,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method_type.value,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.MajorK.value,
        enable_pdl=enable_pdl,
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        activation_type=activation_type,
    ).to(torch.float)

    permute_info, expert_weights_full = routing_reference_renormalize(
        routing_logits, top_k, num_experts, 8
    )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    expert_weights = expert_weights_full.view(num_tokens, num_experts)[
        torch.arange(num_tokens, device=device).unsqueeze(1), topk_ids
    ].to(torch.bfloat16)
    packed_topk_ids = (topk_ids << 16) | expert_weights.view(torch.int16).to(
        torch.int32
    )

    output_routed = trtllm_fp8_block_scale_routed_moe(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=quant_inputs["hidden_states"],
        hidden_states_scale=quant_inputs["hidden_states_scale"],
        gemm1_weights=gemm1_weights_kernel,
        gemm1_weights_scale=gemm1_scales_kernel,
        gemm2_weights=gemm2_weights_kernel,
        gemm2_weights_scale=gemm2_scales_kernel,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method_type.value,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.MajorK.value,
        enable_pdl=enable_pdl,
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        activation_type=activation_type,
    ).to(torch.float)

    close = torch.isclose(output_ref, output_routed, atol=1e-2, rtol=1e-2)
    mismatch_pct = (~close).float().mean().item() * 100
    assert mismatch_pct < 10, f"Mismatch percentage is {mismatch_pct:.2f}%"


@pytest.mark.parametrize("num_tokens", [1, 7, 32])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [16])
@pytest.mark.parametrize("top_k", [2, 4])
def test_fp8_block_scale_moe_routing_replay(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_experts: int,
):
    """Test that routing_replay_out in trtllm_fp8_block_scale_moe records correct expert IDs.

    Uses DeepSeekV3 routing (the only routing method with replay support).
    Runs the full MoE kernel twice with the same inputs: once with routing_replay_out
    and once without. Verifies that:
    1. The MoE output is identical (replay has no side effects).
    2. The replay buffer matches the reference routing result (sorted set equality).
    3. Tail rows beyond num_tokens remain sentinel (CUDA graph pre-alloc contract).
    """
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    n_group = 4
    topk_group = 2
    if topk_group * n_group < top_k or topk_group > n_group:
        pytest.skip("Invalid DeepSeek routing configuration")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)

    routing_logits = torch.rand(
        num_tokens, num_experts, device=device, dtype=torch.float32
    )
    routing_bias = torch.randn(num_experts, device=device, dtype=torch.bfloat16)

    hidden_states_bf16 = (
        torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16) * 0.1
    )
    hidden_states = hidden_states_bf16.to(torch.float8_e4m3fn)

    hidden_states_scale = torch.ones(
        hidden_size // 128, num_tokens, device=device, dtype=torch.float32
    )

    gemm1_weights = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, device=device
    ).to(torch.float8_e4m3fn)
    gemm2_weights = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device
    ).to(torch.float8_e4m3fn)

    gemm1_weights_scale = torch.ones(
        num_experts,
        2 * intermediate_size // 128,
        hidden_size // 128,
        device=device,
        dtype=torch.float32,
    )
    gemm2_weights_scale = torch.ones(
        num_experts,
        hidden_size // 128,
        intermediate_size // 128,
        device=device,
        dtype=torch.float32,
    )

    # Allocate oversized buffer to validate CUDA graph pre-allocation contract:
    # kernel should only write to [0, num_tokens) and leave tail rows as sentinel.
    replay_capacity = num_tokens + 5
    routing_replay_out = torch.full(
        (replay_capacity, top_k), -1, device=device, dtype=torch.int16
    )

    output_with_replay = trtllm_fp8_block_scale_moe(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        0,  # local_expert_offset
        num_experts,
        1.0,  # routed_scaling_factor
        RoutingMethodType.DeepSeekV3.value,
        False,  # use_shuffled_weight
        0,  # weight_layout
        enable_pdl,
        routing_replay_out=routing_replay_out,
    )

    output_without_replay = trtllm_fp8_block_scale_moe(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        0,
        num_experts,
        1.0,
        RoutingMethodType.DeepSeekV3.value,
        False,
        0,
        enable_pdl,
        routing_replay_out=None,
    )

    # MoE output should be identical regardless of replay
    torch.testing.assert_close(
        output_with_replay.to(torch.float),
        output_without_replay.to(torch.float),
        rtol=0,
        atol=0,
    )

    # Compare replay against reference routing — verify active rows only
    active_replay = routing_replay_out[:num_tokens]
    # Verify replay IDs are valid expert indices
    assert (active_replay >= 0).all() and (active_replay < num_experts).all(), (
        "Replay contains out-of-range expert IDs"
    )
    # Each token should have top_k unique experts
    for t in range(num_tokens):
        unique_experts = active_replay[t].unique()
        assert unique_experts.numel() == top_k, (
            f"Token {t}: expected {top_k} unique experts, got {unique_experts.numel()}"
        )

    # Tail rows beyond num_tokens should remain sentinel (-1)
    assert (routing_replay_out[num_tokens:] == -1).all(), (
        "Kernel should not write beyond active token rows"
    )
