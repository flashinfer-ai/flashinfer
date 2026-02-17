# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file wraps TensorRT-LLM's CuteDSL grouped GEMM with SwiGLU fusion:
# tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm_swiglu_fusion.py
#
# Original copyright:
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Contiguous Grouped GEMM kernel with SwiGLU fusion for MoE workloads on Blackwell GPUs.

This module provides a FlashInfer-style API wrapper around the TensorRT-LLM CuteDSL
grouped GEMM kernel with fused SwiGLU activation designed for MoE GEMM1 layers:
- Input A: (permuted_m, k) - permuted tokens from all batches
- Input B: (num_experts, 2*intermediate_size, k) - expert gate and up weights interleaved
- Output C: (permuted_m, intermediate_size) - SwiGLU activated outputs

Key features:
- NVFP4 x NVFP4 grouped GEMM with FP8 scale factors
- Fused SwiGLU activation in epilogue: output = up * silu(gate)
- Optional FP4 quantization of output with scale factor generation
- Persistent tile scheduling with per-expert group mapping
- Warp specialization for overlapped memory and compute
- Support for SM100 (Blackwell) architecture
"""

from typing import Optional, Tuple

import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import functools
import torch
from cutlass.cute.runtime import from_dlpack

from flashinfer.utils import get_compute_capability
from flashinfer.api_logging import flashinfer_api
from flashinfer.cute_dsl.utils import (
    get_cutlass_dtype,
    cutlass_to_torch_dtype,
    get_num_sm,
    get_max_active_clusters,
)

# Import the TRT-LLM kernel implementation
from .blackwell.blockscaled_contiguous_grouped_gemm_swiglu_fusion import (
    Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel,
)

# Re-export the kernel class


@functools.cache
def _get_compiled_swiglu_kernel(
    ab_dtype_name: str,
    sf_dtype_name: str,
    c_dtype_name: str,
    sf_vec_size: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    vectorized_f32: bool,
):
    """Get or compile the grouped GEMM with SwiGLU kernel.

    This function is cached to avoid recompilation for the same parameters.
    Shape parameters (permuted_m, n, k, num_experts) are not included in the
    cache key since the kernel is shape-agnostic.
    """
    ab_dtype = get_cutlass_dtype(ab_dtype_name)
    sf_dtype = get_cutlass_dtype(sf_dtype_name)
    c_dtype = get_cutlass_dtype(c_dtype_name)

    # Create kernel instance
    gemm = Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vectorized_f32=vectorized_f32,
    )

    return gemm, ab_dtype, sf_dtype, c_dtype


@flashinfer_api
def blockscaled_contiguous_grouped_gemm_swiglu_fusion_nvfp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    tile_idx_to_group_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
    *,
    ab_dtype: str = "float4_e2m1fn",
    sf_dtype: str = "float8_e4m3fn",
    c_dtype: str = "bfloat16",
    sf_vec_size: int = 16,
    mma_tiler_mn: Tuple[int, int] = (256, 128),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    vectorized_f32: bool = True,
    sm_count: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Blockscaled Contiguous Grouped GEMM with SwiGLU Fusion for MoE workloads.

    Performs grouped matrix multiplication with fused SwiGLU activation:
    C[tile] = up * silu(gate), where [gate, up] = alpha[group] * (A[tile] @ B[group])

    This kernel is designed for Mixture of Experts (MoE) GEMM1 layers where:
    - Tokens are permuted and contiguously arranged by expert assignment
    - Each expert has gate and up projection weights interleaved
    - SwiGLU activation is fused into the GEMM epilogue
    - Optional FP4 quantization of output

    Args:
        a: Input tensor A (permuted tokens), shape (permuted_m, k) for FP4 stored as (permuted_m, k//2) uint8
        b: Weight tensor B (expert gate+up weights), shape (num_experts, 2*intermediate_size, k)
           for FP4 stored as (num_experts, 2*intermediate_size, k//2) uint8
           The N dimension contains interleaved gate and up projection weights.
        a_scale: Scale factors for A in MMA-compatible layout
        b_scale: Scale factors for B in MMA-compatible layout
        alpha: Per-expert scaling factors, shape (num_experts,), float32
        tile_idx_to_group_idx: Mapping from tile index to expert index, shape (num_tiles,), int32
        num_non_exiting_tiles: Number of valid tiles, shape (1,), int32
        out: Optional output tensor, shape (permuted_m, intermediate_size). Created if None.
             For FP4 output, shape is (permuted_m, intermediate_size//2) uint8.
        out_scale: Optional output scale factor tensor for FP4 quantized output.
                   Shape depends on MMA layout. Only used when c_dtype is "float4_e2m1fn".
        global_scale: Global scale factor for FP4 quantization, shape (1,), float32.
                      Required when c_dtype is "float4_e2m1fn".
        ab_dtype: Data type for A and B matrices. Default: "float4_e2m1fn"
        sf_dtype: Data type for scale factors. Default: "float8_e4m3fn"
        c_dtype: Data type for output matrix. Default: "bfloat16"
                 Set to "float4_e2m1fn" for fused FP4 quantization.
        sf_vec_size: Scale factor vector size. Default: 16 (for NVFP4)
        mma_tiler_mn: MMA tile shape (M, N). Default: (256, 128)
        cluster_shape_mn: Cluster shape (ClusterM, ClusterN). Default: (2, 1)
        vectorized_f32: Use vectorized f32x2 operations. Default: True
        sm_count: Number of SMs to use. Default: max available.

    Returns:
        Tuple of:
        - out: Output tensor C, shape (permuted_m, intermediate_size) with dtype c_dtype
               For FP4 output: (permuted_m, intermediate_size//2) uint8
        - out_scale: Output scale factors if c_dtype is FP4, else None

    Notes:
        - The B tensor N dimension is 2*intermediate_size (gate + up interleaved)
        - Output N dimension is intermediate_size (after SwiGLU)
        - Use create_tile_mapping() to create tile_idx_to_group_idx and num_non_exiting_tiles
        - Requires SM100 (Blackwell) GPU architecture
        - SwiGLU fusion significantly reduces memory bandwidth vs separate activation

    Example:
        >>> # Setup for MoE GEMM1 with 8 experts
        >>> num_experts, hidden_dim, intermediate_dim = 8, 4096, 14336
        >>>
        >>> # Create tile mapping from routing decisions
        >>> group_m = torch.tensor([256, 128, 384, 256, 128, 256, 256, 384], device="cuda")
        >>> valid_m, aligned_m, tile_map, num_tiles = create_tile_mapping(group_m, mma_tiler_m=256)
        >>>
        >>> # Run grouped GEMM with SwiGLU fusion
        >>> out, _ = blockscaled_contiguous_grouped_gemm_swiglu_fusion_nvfp4(
        ...     a=permuted_input_fp4,           # (valid_m, hidden_dim//2)
        ...     b=expert_gate_up_weights_fp4,   # (num_experts, 2*intermediate_dim, hidden_dim//2)
        ...     a_scale=input_scale,
        ...     b_scale=weight_scale,
        ...     alpha=expert_alpha,             # (num_experts,)
        ...     tile_idx_to_group_idx=tile_map,
        ...     num_non_exiting_tiles=num_tiles,
        ... )  # out shape: (valid_m, intermediate_dim)
    """
    # Validate inputs
    assert a.device.type == "cuda", "Input tensors must be on CUDA device"
    assert b.device.type == "cuda", "Input tensors must be on CUDA device"

    # Get dimensions
    permuted_m = a.shape[0]
    num_experts = b.shape[0]
    n = b.shape[1]  # This is 2*intermediate_size
    k = a.shape[1]
    if ab_dtype == "float4_e2m1fn":
        k = k * 2  # FP4 is packed 2 elements per byte

    intermediate_size = n // 2  # Output dimension after SwiGLU

    # Check compute capability
    major, minor = get_compute_capability(a.device)
    if major != 10:
        raise ValueError(
            f"Blockscaled contiguous grouped GEMM with SwiGLU requires SM100 family (Blackwell: SM100, SM103, SM110). "
            f"Got SM{major}{minor}."
        )

    # Validate configuration
    ab_dtype_cutlass = get_cutlass_dtype(ab_dtype)
    sf_dtype_cutlass = get_cutlass_dtype(sf_dtype)
    c_dtype_cutlass = get_cutlass_dtype(c_dtype)

    if not Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel.can_implement(
        ab_dtype_cutlass,
        sf_dtype_cutlass,
        sf_vec_size,
        c_dtype_cutlass,
        mma_tiler_mn,
        cluster_shape_mn,
        permuted_m,
        n,
        k,
        num_experts,
        a_major="k",
        b_major="k",
        c_major="n",
    ):
        raise ValueError(
            f"Unsupported configuration: ab_dtype={ab_dtype}, sf_dtype={sf_dtype}, "
            f"sf_vec_size={sf_vec_size}, c_dtype={c_dtype}, mma_tiler_mn={mma_tiler_mn}, "
            f"cluster_shape_mn={cluster_shape_mn}, shape=({permuted_m}, {n}, {k}, {num_experts})"
        )

    # Check if we're doing FP4 quantization
    generate_sfc = c_dtype == "float4_e2m1fn"
    if generate_sfc:
        if global_scale is None:
            raise ValueError("global_scale is required when c_dtype is 'float4_e2m1fn'")

    # Create output tensor if not provided
    if out is None:
        if generate_sfc:
            # FP4 output: 2 values per byte
            out = torch.empty(
                (permuted_m, intermediate_size // 2),
                dtype=torch.uint8,
                device=a.device,
            )
        else:
            out = torch.empty(
                (permuted_m, intermediate_size),
                dtype=cutlass_to_torch_dtype(c_dtype_cutlass),
                device=a.device,
            )

    # Create output scale tensor if needed and not provided
    if generate_sfc and out_scale is None:
        # Scale factor layout for output
        scale_intermediate_size = intermediate_size // sf_vec_size
        # MMA-compatible scale factor shape
        out_scale = torch.empty(
            (32, 4, permuted_m // 128, 4, scale_intermediate_size // 4, 1),
            dtype=torch.uint8,  # FP8 E4M3
            device=a.device,
        )

    # Get SM count
    if sm_count is None:
        sm_count = get_num_sm(a.device)

    # Get or compile the kernel
    gemm, _, _, _ = _get_compiled_swiglu_kernel(
        ab_dtype_name=ab_dtype,
        sf_dtype_name=sf_dtype,
        c_dtype_name=c_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vectorized_f32=vectorized_f32,
    )

    # Compute max active clusters (cached to avoid expensive HardwareInfo queries)
    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # Create CuTe tensors from PyTorch tensors
    # A: (permuted_m, k, 1) - single batch of permuted tokens
    # B: (n, k, num_experts) - expert weights (note: n = 2*intermediate_size)
    # C: (permuted_m, intermediate_size, 1) - output
    a_tensor = from_dlpack(a.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    b_tensor = from_dlpack(b.permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    c_tensor = from_dlpack(out.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )

    # Set the correct element types for FP4 tensors
    # from_dlpack infers dtype from PyTorch (uint8) but the kernel needs Float4E2M1FN
    a_tensor.element_type = ab_dtype_cutlass
    b_tensor.element_type = ab_dtype_cutlass
    c_tensor.element_type = c_dtype_cutlass

    # Scale factor tensors
    # For A scale: 2D or 3D linear layout with dim 1 contiguous
    sfa_tensor = from_dlpack(a_scale, assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    # For B scale: 6D strided MMA layout with physical dim 3 contiguous
    sfb_tensor = from_dlpack(b_scale, assumed_align=16).mark_layout_dynamic(
        leading_dim=3
    )

    # Set the correct element types for scale factor tensors
    sfa_tensor.element_type = sf_dtype_cutlass
    sfb_tensor.element_type = sf_dtype_cutlass

    # Optional output scale factor tensor
    if generate_sfc:
        # Output scale is contiguous 6D with dim 4 having stride 1
        sfc_tensor = from_dlpack(out_scale, assumed_align=16).mark_layout_dynamic(
            leading_dim=4
        )
        sfc_tensor.element_type = sf_dtype_cutlass
        norm_const_tensor = from_dlpack(
            global_scale, assumed_align=16
        ).mark_layout_dynamic()
    else:
        sfc_tensor = None
        norm_const_tensor = None

    # Mapping tensors
    tile_idx_tensor = from_dlpack(tile_idx_to_group_idx).mark_layout_dynamic()
    num_tiles_tensor = from_dlpack(num_non_exiting_tiles).mark_layout_dynamic()
    alpha_tensor = from_dlpack(alpha).mark_layout_dynamic()

    # Get current CUDA stream
    current_stream = cutlass_torch.current_stream()

    # Compile and run the kernel
    compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        norm_const_tensor,
        tile_idx_tensor,
        num_tiles_tensor,
        alpha_tensor,
        max_active_clusters,
        current_stream,
    )

    # Execute
    compiled_gemm(
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        norm_const_tensor,
        tile_idx_tensor,
        num_tiles_tensor,
        alpha_tensor,
        current_stream,
    )

    return out, out_scale if generate_sfc else None
