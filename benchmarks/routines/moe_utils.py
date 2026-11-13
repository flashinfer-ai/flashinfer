# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MoE Benchmark Utilities

Shared helper functions for MoE benchmarks including:
- FP4/FP8 quantization and dequantization
- Performance metrics calculation
- Routing utilities
- Triton kernels for expert ID packing
- Common argument parsing
- Weight layout processing
"""

import argparse
from typing import Optional, Tuple

import torch


from flashinfer import fp4_quantize, shuffle_matrix_a
from flashinfer.fused_moe import WeightLayout, convert_to_block_layout

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0

SF_VEC_SIZE = {
    "nvfp4": 16,
    "mxfp4": 32,
    "mxfp8": 32,
}


def generate_moe_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    is_gated: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random weights for MoE experts.

    Args:
        num_experts: Number of experts to generate weights for
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate FFN dimension
        device: Device to create tensors on
        dtype: Data type for the weights (default: bfloat16)
        is_gated: Whether GEMM1 has separate gate and linear projections

    Returns:
        gemm1_weights: [num_experts, (2 if is_gated else 1) * intermediate_size, hidden_size]
        gemm2_weights: [num_experts, hidden_size, intermediate_size]
    """
    gemm1_rows = (2 if is_gated else 1) * intermediate_size
    gemm1_weights = torch.randn(
        (num_experts, gemm1_rows, hidden_size),
        device=device,
        dtype=dtype,
    )
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=dtype,
    )
    return gemm1_weights, gemm2_weights


def calculate_fp4_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate global scale factor for FP4 quantization.

    Args:
        tensor: Input tensor to compute scale for

    Returns:
        Global scale factor as a scalar tensor
    """
    tensor_amax = tensor.abs().max().to(torch.float32)
    if tensor_amax == 0.0:
        global_scale = torch.tensor(0.0, dtype=torch.float32, device=tensor.device)
    else:
        global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / tensor_amax
    return global_scale


def quantize_fp4(
    tensor: torch.Tensor,
    global_scale: Optional[torch.Tensor] = None,
    use_ue8m0: bool = False,
    is_sf_swizzled_layout: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to FP4 format.

    Args:
        tensor: Input tensor to quantize
        global_scale: Optional pre-computed global scale. If None, computed from tensor.
        use_ue8m0: Whether to use UE8M0 format
        is_sf_swizzled_layout: Whether to use swizzled layout for scale factors

    Returns:
        Tuple of (quantized_data, block_scale_factors, global_scale_factor)
        - quantized_data: uint8 tensor with packed FP4 values
        - block_scale_factors: float8_e4m3fn tensor
        - global_scale_factor: float32 scalar tensor
    """
    sf_vec_size = SF_VEC_SIZE["mxfp4" if use_ue8m0 else "nvfp4"]

    if global_scale is None:
        global_scale = calculate_fp4_global_scale(tensor)

    quantized, block_scales = fp4_quantize(
        tensor, global_scale, sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    return quantized, block_scales, global_scale


def quantize_fp4_batched(
    tensor: torch.Tensor,
    num_experts: int,
    use_ue8m0: bool = False,
    is_sf_swizzled_layout: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize batched tensor to FP4 format, computing per-expert global scales.

    Args:
        tensor: Input tensor of shape [num_experts, ...]
        num_experts: Number of experts in the batch
        use_ue8m0: Whether to use UE8M0 format
        is_sf_swizzled_layout: Whether to use swizzled layout for scale factors

    Returns:
        Tuple of (quantized_data, block_scale_factors, global_scale_factors)
    """
    quant_list = []
    sf_list = []
    global_sf_list = []

    for i in range(num_experts):
        global_sf = calculate_fp4_global_scale(tensor[i])
        quantized, block_sf, _ = quantize_fp4(
            tensor[i], global_sf, use_ue8m0, is_sf_swizzled_layout
        )
        quant_list.append(quantized)
        sf_list.append(block_sf)
        global_sf_list.append(global_sf)

    return (
        torch.stack(quant_list),
        torch.stack(sf_list),
        torch.stack(global_sf_list),
    )


# Adapted from tests/moe/test_trtllm_cutlass_fused_moe.py
def dequantize_nvfp4(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    block_size: int = 16,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantize FP4 tensor back to high precision.

    Args:
        tensor_fp4: FP4 quantized tensor (uint8, packed)
        tensor_sf: Block scale factors
        global_scale: Global scale factor
        block_size: Number of elements per scale block
        dtype: Output dtype

    Returns:
        Dequantized tensor in specified dtype
    """

    def break_fp4_bytes(a, out_dtype):
        assert a.dtype == torch.uint8
        m, n = a.shape
        # Vectorized nibble processing
        a_flat = a.flatten()
        high = (a_flat & 0xF0) >> 4  # Upper nibbles
        low = a_flat & 0x0F  # Lower nibbles
        # Combine nibbles for batch processing
        combined = torch.stack((low, high), dim=1).flatten()
        # Vectorized sign and magnitude extraction
        signs = (combined & 0x08).to(torch.bool)  # Sign bits
        abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices
        # Device-aware lookup and sign application
        kE2M1ToFloat = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
        )
        kE2M1 = kE2M1ToFloat.to(device=a.device)
        values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
        # Reshape to final form
        return values.reshape(m, n * 2).to(dtype=out_dtype)

    # Two fp4 values are packed into one uint8
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # Scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)


def quantize_fp8(
    tensor: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to FP8 (per-tensor scale).

    Args:
        tensor: Input tensor to quantize
        scale: Optional pre-computed scale. If None, computed from tensor.

    Returns:
        Tuple of (quantized_tensor, scale_factor)
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    if scale is None:
        amax = tensor.abs().max().float().clamp(min=1e-6)
        scale = amax / fp8_max
    inv_scale = 1.0 / scale if scale != 0.0 else 0.0
    quantized = (
        (tensor.float() * inv_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    )
    return quantized, scale.view(1) if scale.dim() == 0 else scale


def quantize_fp8_block_scale(
    tensor: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to FP8 with block scales.

    For trtllm_fp8_block_scale_moe, hidden_states_scale shape is [hidden_size // 128, num_tokens].

    Args:
        tensor: Input tensor [num_tokens, hidden_size]
        block_size: Number of elements per scale block (default 128)

    Returns:
        Tuple of (quantized_tensor, block_scales)
        - quantized_tensor: float8_e4m3fn tensor [num_tokens, hidden_size]
        - block_scales: float32 tensor [hidden_size // block_size, num_tokens]
    """
    num_tokens, hidden_size = tensor.shape
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    # Compute per-block amax and scales
    # Reshape to [num_tokens, num_blocks, block_size]
    num_blocks = hidden_size // block_size
    reshaped = tensor.float().reshape(num_tokens, num_blocks, block_size)

    # Compute amax per block: [num_tokens, num_blocks]
    block_amax = reshaped.abs().amax(dim=-1).clamp(min=1e-6)

    # Compute scales: [num_tokens, num_blocks]
    block_scales = block_amax / fp8_max

    # Quantize each block
    inv_scales = 1.0 / block_scales  # [num_tokens, num_blocks]
    # Expand for broadcasting: [num_tokens, num_blocks, 1]
    inv_scales_expanded = inv_scales.unsqueeze(-1)
    quantized_reshaped = (reshaped * inv_scales_expanded).clamp(-fp8_max, fp8_max)
    quantized = quantized_reshaped.reshape(num_tokens, hidden_size).to(
        torch.float8_e4m3fn
    )

    # Transpose block_scales to [num_blocks, num_tokens] as expected by kernel
    block_scales_transposed = block_scales.transpose(0, 1).contiguous()

    return quantized, block_scales_transposed


def dequantize_fp8(
    tensor_fp8: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize FP8 tensor back to high precision.

    Args:
        tensor_fp8: FP8 quantized tensor (float8_e4m3fn)
        scale: Per-tensor scale factor
        dtype: Output dtype

    Returns:
        Dequantized tensor in specified dtype
    """
    return (tensor_fp8.float() * scale.float()).to(dtype)


def calculate_moe_tflops(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    time_ms: float,
    is_gated: bool = True,
) -> float:
    """
    Calculate TFLOPS for MOE operation.

    MOE computation involves:
    1. First GEMM: [num_tokens, hidden_size] x [num_experts, hidden_size, w1_cols]
       where w1_cols = 2*intermediate_size (gated) or intermediate_size (non-gated)
    2. Activation function (SwiGLU gate or ReLU2)
    3. Second GEMM: [num_tokens, intermediate_size] x [num_experts, intermediate_size, hidden_size]

    For each token, we only compute for top_k experts.

    Args:
        num_tokens: Number of input tokens
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate FFN dimension
        num_experts: Total number of experts
        top_k: Number of experts per token
        time_ms: Execution time in milliseconds
        is_gated: Whether activation is gated (SwiGLU/GeGLU) or non-gated (ReLU2)

    Returns:
        TFLOPS value
    """
    _ = num_experts  # kept for backward compatibility

    # FLOPS per token per expert
    w1_cols = (2 if is_gated else 1) * intermediate_size
    flops_per_token_per_expert = (
        2 * hidden_size * w1_cols  # First GEMM
        + 2 * intermediate_size * hidden_size  # Second GEMM
    )

    total_flops = num_tokens * top_k * flops_per_token_per_expert
    tflops = total_flops / (time_ms * 1e-3) / 1e12  # Convert to TFLOPS
    return tflops


def calculate_moe_kernel_bandwidth(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    time_ms: float,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    input_format: Optional[str] = None,
    weight_format: Optional[str] = None,
    routing_logits_dtype: Optional[torch.dtype] = torch.float32,
    active_experts: Optional[int] = None,
    verbose: int = 0,
    is_gated: bool = True,
) -> float:
    """
    Calculate memory bandwidth for MOE kernel operation in TB/sec.

    Args:
        num_tokens: Number of input tokens
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate FFN dimension
        num_experts: Total number of experts
        top_k: Number of experts per token
        time_ms: Execution time in milliseconds
        input_dtype: Data type of input
        weight_dtype: Data type of weights
        input_format: Override for input representation; None uses dtype.itemsize
        weight_format: Override for weight representation; None uses dtype.itemsize
        routing_logits_dtype: Dtype for routing logits memory accounting (default float32)
        active_experts: Number of active experts (if known)
        verbose: Verbosity level

    Returns:
        Bandwidth in TB/sec
    """

    # Get effective byte sizes
    def get_effective_bytes(
        dtype: torch.dtype, fmt: Optional[str], is_weight: bool = False
    ) -> float:
        if fmt == "nvfp4":
            return 0.5 + 1 / SF_VEC_SIZE["nvfp4"]
        elif fmt == "mxfp4":
            return 0.5 + 1 / SF_VEC_SIZE["mxfp4"]
        elif fmt == "mxfp8":
            return 1.0 + 1 / SF_VEC_SIZE["mxfp8"]
        elif fmt == "fp8":
            # 1 e4m3
            return 1.0
        elif fmt == "fp8_block_scale":
            granularity = 128 * 128 if is_weight else 128
            # 1 e4m3 + 1 float32 scale factor per block
            return 1.0 + (4 / granularity)
        return dtype.itemsize

    input_bytes_per_element = get_effective_bytes(input_dtype, input_format)
    weight_bytes_per_element = get_effective_bytes(
        weight_dtype, weight_format, is_weight=True
    )

    # Input memory: hidden states + routing logits
    routing_logits_bytes = (
        0 if routing_logits_dtype is None else routing_logits_dtype.itemsize
    )
    input_bytes = (
        # Count hidden states once; kernels typically reuse inputs for multiple experts
        num_tokens * hidden_size * input_bytes_per_element
        + num_tokens * num_experts * routing_logits_bytes
    )

    # Weight memory
    w1_cols = (2 if is_gated else 1) * intermediate_size
    weight_bytes_per_expert = (
        w1_cols * hidden_size * weight_bytes_per_element  # gemm1
        + hidden_size * intermediate_size * weight_bytes_per_element  # gemm2
    )
    if active_experts is not None:
        num_active_experts = active_experts
    else:
        num_active_experts = min(num_experts, top_k * num_tokens)
    if verbose >= 2:
        print(f"[VVERBOSE] num_active_experts = {num_active_experts}")

    weight_bytes = num_active_experts * weight_bytes_per_expert

    # Output memory (typically full precision)
    output_bytes = num_tokens * hidden_size * input_dtype.itemsize

    total_bytes = input_bytes + weight_bytes + output_bytes
    tb_per_sec = total_bytes / (time_ms * 1e-3) / 1e12  # Convert to TB/sec
    return tb_per_sec


def compute_routing(
    router_logits: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute routing weights and selected experts using standard top-k routing.

    Args:
        router_logits: [num_tokens, num_experts] routing scores
        top_k: Number of experts to select per token

    Returns:
        Tuple of (routing_weights, selected_experts)
        - routing_weights: [num_tokens, top_k] normalized routing weights
        - selected_experts: [num_tokens, top_k] selected expert indices
    """
    routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    return routing_weights, selected_experts


def add_common_moe_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common MoE CLI arguments to a parser.

    Adds num_tokens, hidden_size, num_experts, top_k and input_dtype.

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--num_tokens",
        type=int,
        required=True,
        help="Number of tokens per rank (local batch size).",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        required=True,
        help="Hidden dimension size.",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        required=True,
        help="Total number of experts.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=True,
        help="Number of experts to route each token to.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Data type of input hidden states.",
    )


def process_fp8_weight_layout(
    tensor: torch.Tensor,
    use_shuffled_weight: bool,
    weight_layout: int,
    epilogue_tile_m: int = 64,
) -> torch.Tensor:
    """
    Process FP8 weight tensor with optional shuffling and layout conversion.

    This encapsulates the common pattern of:
    1. Converting to uint8 view
    2. Applying shuffle_matrix_a
    3. Optionally converting to BlockMajorK layout

    Args:
        tensor: FP8 weight tensor (float8_e4m3fn)
        use_shuffled_weight: Whether to apply weight shuffling
        weight_layout: Weight layout (0=MajorK, 2=BlockMajorK)
        epilogue_tile_m: Tile size for shuffle operation (default 64)

    Returns:
        Processed tensor (as float8_e4m3fn view)
    """
    if use_shuffled_weight:
        # Shuffle the weight matrix
        tensor = shuffle_matrix_a(tensor.view(torch.uint8), epilogue_tile_m)

    # Apply block layout conversion if needed
    if weight_layout == WeightLayout.BlockMajorK:
        block_k = 128
        tensor = convert_to_block_layout(tensor, block_k)

    return tensor.view(torch.float8_e4m3fn)


def create_moe_output_scale_scalars(
    num_experts: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create output scale scalar tensors for MoE kernels.

    These are used by FP4 and FP8 per-tensor scale MoE kernels.

    Args:
        num_experts: Number of experts
        device: Device to create tensors on

    Returns:
        Tuple of (output1_scale_scalar, output1_scale_gate_scalar, output2_scale_scalar)
        All tensors are float32 with shape [num_experts], initialized to 1.0
    """
    output1_scale_scalar = torch.ones(num_experts, device=device, dtype=torch.float32)
    output1_scale_gate_scalar = torch.ones(
        num_experts, device=device, dtype=torch.float32
    )
    output2_scale_scalar = torch.ones(num_experts, device=device, dtype=torch.float32)
    return output1_scale_scalar, output1_scale_gate_scalar, output2_scale_scalar
