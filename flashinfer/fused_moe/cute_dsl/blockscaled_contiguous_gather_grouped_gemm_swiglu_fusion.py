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

# This file wraps TensorRT-LLM's CuteDSL grouped GEMM with gather and SwiGLU fusion:
# tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py
#
# Original copyright:
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Contiguous Grouped GEMM kernel with Gather and SwiGLU Fusion for MoE workloads on Blackwell GPUs.

This module provides a FlashInfer-style API wrapper around the TensorRT-LLM CuteDSL
grouped GEMM kernel with fused gather and SwiGLU activation designed for MoE GEMM1 layers:
- Input A: (seq_len, k) - original unpermuted tokens (no need for moe_permute!)
- Input B: (num_experts, 2*intermediate_size, k) - expert gate and up weights interleaved
- Output C: (permuted_m, intermediate_size) - SwiGLU activated outputs in permuted order

Key features:
- NVFP4 x NVFP4 grouped GEMM with FP8 scale factors
- Fused gather operation using LDGSTS instructions with token_id_mapping
- Eliminates the need for a separate moe_permute kernel
- Fused SwiGLU activation in epilogue: output = up * silu(gate)
- Optional FP4 quantization of output with scale factor generation
- Persistent tile scheduling with per-expert group mapping
- Warp specialization for overlapped memory and compute
- Support for SM100 (Blackwell) architecture

Comparison with Non-Gather SwiGLU Fusion:
- Non-Gather: Requires separate moe_permute kernel, then uses TMA for contiguous A load
- Gather: Uses LDGSTS to gather A directly using token_id_mapping, no moe_permute needed
"""

from typing import Any, Dict, List, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch

from flashinfer.utils import get_compute_capability
from flashinfer.api_logging import flashinfer_api
from flashinfer.cute_dsl.utils import (
    get_cutlass_dtype,
    cutlass_to_torch_dtype,
    get_num_sm,
    get_max_active_clusters,
    make_ptr,
)

# Import the TRT-LLM kernel implementation
from .blackwell.blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion import (
    BlockScaledContiguousGatherGroupedGemmKernel,
)

# Re-export the kernel class


def create_gather_gemm_tensors(
    seq_len: int,
    topk: int,
    group_m_list: List[int],
    mma_tiler_m: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, List[int]]:
    """Create tensors required for gather grouped GEMM.

    This function creates the mapping tensors needed for the fused gather operation
    in GEMM1 with SwiGLU activation.

    Args:
        seq_len: Number of input tokens (original sequence length before routing)
        topk: Number of experts per token
        group_m_list: List of actual (unaligned) M values per expert
        mma_tiler_m: MMA tile M dimension for alignment (128 or 256)

    Returns:
        Tuple of:
        - token_id_mapping: Maps permuted row to token_idx * topk + k_idx, shape (permuted_m,), int32
          Used by LDGSTS to gather from the original unpermuted A tensor.
          Invalid rows are marked with -1.
        - tile_idx_to_expert_idx: Tile to expert mapping, shape (num_tiles,), int32
        - tile_idx_to_mn_limit: M limit for each tile, shape (num_tiles,), int32
        - num_non_exiting_tiles: Number of valid tiles, shape (1,), int32
        - valid_m: Total valid M dimension (sum of aligned group sizes)
        - aligned_group_m_list: List of aligned M values per expert

    Example:
        >>> seq_len, topk, num_experts = 4096, 8, 8
        >>> group_m_list = [512, 480, 256, 320, 640, 512, 384, 704]  # Tokens per expert
        >>>
        >>> token_id_map, tile_map, mn_limit, num_tiles, valid_m, aligned_m = create_gather_gemm_tensors(
        ...     seq_len=seq_len,
        ...     topk=topk,
        ...     group_m_list=group_m_list,
        ...     mma_tiler_m=256,
        ... )
    """
    valid_m = 0
    aligned_group_m_list = []
    tile_idx_to_expert_idx = []
    tile_idx_to_mn_limit = []

    for i, group_m in enumerate(group_m_list):
        aligned_group_m = ((group_m + mma_tiler_m - 1) // mma_tiler_m) * mma_tiler_m
        aligned_group_m_list.append(aligned_group_m)

        # Calculate number of tiles for this group
        num_tiles_in_group = aligned_group_m // mma_tiler_m
        tile_idx_to_expert_idx.extend([i] * num_tiles_in_group)

        # M limit for boundary checking
        for tile_idx_in_group in range(num_tiles_in_group):
            tile_idx_to_mn_limit.append(
                valid_m + min(tile_idx_in_group * mma_tiler_m + mma_tiler_m, group_m)
            )
        valid_m += aligned_group_m

    num_non_exiting_tiles = len(tile_idx_to_expert_idx)

    # Create token_id_mapping for gather operation
    # Maps permuted row index to expanded_idx = token_idx * topk + k_idx
    token_id_mapping = torch.empty((valid_m,), dtype=torch.int32, device="cuda").fill_(
        -1
    )

    start_idx = 0
    for group_idx, m_per_group in enumerate(group_m_list):
        if m_per_group > 0:
            # Sequential/Blocked assignment for better memory access patterns
            # Experts are grouped into sets of size topk
            expert_set_idx = group_idx // topk
            k_in_set = group_idx % topk

            # Start token index for this expert set
            start_token = expert_set_idx * m_per_group

            # Generate sequential token indices for this expert
            token_indices = torch.arange(
                start_token, start_token + m_per_group, dtype=torch.int32, device="cuda"
            )
            token_indices = token_indices % seq_len

            # expanded_idx = token_idx * topk + k
            expanded_idx = token_indices * topk + k_in_set

            token_id_mapping[start_idx : (start_idx + m_per_group)] = expanded_idx

        # Move to next aligned group
        aligned_group_m = aligned_group_m_list[group_idx]
        start_idx += aligned_group_m

    # Convert to tensors
    tile_idx_to_expert_idx = torch.tensor(
        tile_idx_to_expert_idx, device="cuda", dtype=torch.int32
    )
    tile_idx_to_mn_limit = torch.tensor(
        tile_idx_to_mn_limit, device="cuda", dtype=torch.int32
    )
    num_non_exiting_tiles_tensor = torch.tensor(
        [num_non_exiting_tiles], device="cuda", dtype=torch.int32
    )

    return (
        token_id_mapping,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        num_non_exiting_tiles_tensor,
        valid_m,
        aligned_group_m_list,
    )


# Kernel cache for compiled kernels (class-level to persist across calls)
_gather_kernel_cache: Dict[Tuple, Any] = {}


def _get_compiled_gather_kernel(
    # Problem dimensions (runtime parameters - NOT in cache key)
    orig_m: int,
    permuted_m: int,
    n: int,  # This is 2*intermediate_size
    k: int,
    num_experts: int,
    # Tensor pointers (runtime parameters - NOT in cache key)
    a_ptr,
    b_ptr,
    a_sf_ptr,
    b_sf_ptr,
    c_ptr,
    c_sf_ptr,
    alpha_ptr,
    tile_idx_ptr,
    mn_limit_ptr,
    token_id_ptr,
    num_tiles_ptr,
    norm_const_ptr,
    max_active_clusters: int,
    stream,
    # Dtype parameters (compile-time - IN cache key)
    # cute.compile specializes on pointer types, so dtype must be in cache key
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    # Tactic parameters (compile-time - IN cache key)
    sf_vec_size: int,
    tile_size: int,
    topk: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    vectorized_f32: bool,
    raster_along_m: bool,
):
    """Get or compile the gather grouped GEMM with SwiGLU kernel.

    This function caches compiled kernels by tactic and dtype parameters.
    Problem dimensions (m, n, k, num_experts) are runtime parameters.

    The cache key includes dtype parameters because cute.compile specializes
    on the types of pointer arguments. Using the same compiled kernel with
    different dtypes would cause incorrect results or crashes.

    This matches TRT-LLM's approach where the same compiled kernel can be
    reused for different problem sizes, significantly reducing JIT compilation
    overhead during autotuning.
    """
    global _gather_kernel_cache

    # Cache key includes dtype and tactic parameters, NOT problem dimensions
    cache_key = (
        ab_dtype,
        sf_dtype,
        c_dtype,
        sf_vec_size,
        tile_size,
        topk,
        mma_tiler_mn,
        cluster_shape_mn,
        vectorized_f32,
        raster_along_m,
    )

    if cache_key not in _gather_kernel_cache:
        # Create kernel instance
        gemm = BlockScaledContiguousGatherGroupedGemmKernel(
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=vectorized_f32,
            topk=topk,
            raster_along_m=raster_along_m,
        )

        # Compile with runtime parameters - they can vary across calls
        # Order must match wrapper signature:
        # (a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, c_sf_ptr, alpha_ptr,
        #  tile_idx_to_group_idx_ptr, tile_idx_to_mn_limit_ptr, token_id_mapping_ptr,
        #  num_non_exiting_tiles_ptr, global_sf_ptr, orig_m, m, n, k, l,
        #  tile_size, scaling_vector_size, max_active_clusters, stream)
        compiled_gemm = cute.compile(
            gemm.wrapper,
            a_ptr,
            b_ptr,
            a_sf_ptr,
            b_sf_ptr,
            c_ptr,
            c_sf_ptr,
            alpha_ptr,
            tile_idx_ptr,
            mn_limit_ptr,
            token_id_ptr,
            num_tiles_ptr,
            norm_const_ptr,
            orig_m,
            permuted_m,
            n,
            k,
            num_experts,
            tile_size=tile_size,
            scaling_vector_size=sf_vec_size,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )

        _gather_kernel_cache[cache_key] = compiled_gemm

    return _gather_kernel_cache[cache_key]


@flashinfer_api
def blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    token_id_mapping: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
    *,
    topk: int = 8,
    ab_dtype: str = "float4_e2m1fn",
    sf_dtype: str = "float8_e4m3fn",
    c_dtype: str = "bfloat16",
    sf_vec_size: int = 16,
    mma_tiler_mn: Tuple[int, int] = (256, 128),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    vectorized_f32: bool = True,
    raster_along_m: bool = False,
    sm_count: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Blockscaled Contiguous Gather Grouped GEMM with SwiGLU Fusion for MoE workloads.

    Performs grouped matrix multiplication with fused gather and SwiGLU activation:
    C[row] = up * silu(gate), where [gate, up] = alpha[expert] * (A[token_id] @ B[expert])

    This kernel is designed for Mixture of Experts (MoE) GEMM1 layers where:
    - Input tokens are NOT pre-permuted (no need for moe_permute kernel!)
    - The kernel gathers input tokens using token_id_mapping during LDGSTS load
    - Each expert has gate and up projection weights interleaved
    - SwiGLU activation is fused into the GEMM epilogue
    - Optional FP4 quantization of output

    Args:
        a: Input tensor A (original unpermuted tokens), shape (seq_len, k) for FP4
           stored as (seq_len, k//2) uint8. This is the ORIGINAL unpermuted tensor!
        b: Weight tensor B (expert gate+up weights), shape (num_experts, 2*intermediate_size, k)
           for FP4 stored as (num_experts, 2*intermediate_size, k//2) uint8
           The N dimension contains interleaved gate and up projection weights.
        a_scale: Scale factors for A in MMA-compatible layout
        b_scale: Scale factors for B in MMA-compatible layout
        alpha: Per-expert scaling factors, shape (num_experts,), float32
        tile_idx_to_expert_idx: Mapping from tile index to expert index, shape (num_tiles,), int32
        tile_idx_to_mn_limit: M limit for each tile for boundary checking, shape (num_tiles,), int32
        token_id_mapping: Mapping from permuted row to token_id, shape (permuted_m,), int32
            token_id = token_idx * topk + k_idx. Invalid rows have -1.
            Used by LDGSTS to gather from A tensor.
        num_non_exiting_tiles: Number of valid tiles, shape (1,), int32
        out: Optional output tensor, shape (permuted_m, intermediate_size). Created if None.
             For FP4 output, shape is (permuted_m, intermediate_size//2) uint8.
        out_scale: Optional output scale factor tensor for FP4 quantized output.
        global_scale: Global scale factor for FP4 quantization, shape (1,), float32.
        topk: Number of experts per token. Default: 8
        ab_dtype: Data type for A and B matrices. Default: "float4_e2m1fn"
        sf_dtype: Data type for scale factors. Default: "float8_e4m3fn"
        c_dtype: Data type for output matrix. Default: "bfloat16"
        sf_vec_size: Scale factor vector size. Default: 16 (for NVFP4)
        mma_tiler_mn: MMA tile shape (M, N). Default: (256, 128)
        cluster_shape_mn: Cluster shape (ClusterM, ClusterN). Default: (2, 1)
        vectorized_f32: Use vectorized f32x2 operations. Default: True
        raster_along_m: If True, raster tiles along M dimension. Default: False
        sm_count: Number of SMs to use. Default: max available.

    Returns:
        Tuple of:
        - out: Output tensor C, shape (permuted_m, intermediate_size) with dtype c_dtype
               For FP4 output: (permuted_m, intermediate_size//2) uint8
        - out_scale: Output scale factors if c_dtype is FP4, else None

    Notes:
        - Unlike the Non-Gather SwiGLU kernel, this kernel does NOT require moe_permute!
        - The A tensor is the original unpermuted input
        - The output is in permuted order (can be fed directly to GEMM2)
        - Use create_gather_gemm_tensors() to create required mapping tensors
        - Requires SM100 (Blackwell) GPU architecture

    Example:
        >>> # Setup for MoE GEMM1 with 8 experts, no moe_permute needed!
        >>> num_experts, hidden_dim, intermediate_dim = 8, 4096, 14336
        >>> seq_len, topk = 4096, 8
        >>>
        >>> # Create gather mapping tensors
        >>> group_m = torch.tensor([512, 480, 256, 320, 640, 512, 384, 704], device="cuda")
        >>> token_map, tile_map, mn_limit, num_tiles, valid_m, aligned_m = create_gather_gemm_tensors(
        ...     seq_len=seq_len, topk=topk, group_m_list=group_m.tolist(), mma_tiler_m=256
        ... )
        >>>
        >>> # Run gathered GEMM with SwiGLU fusion - NO moe_permute needed!
        >>> out, _ = blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4(
        ...     a=original_input_fp4,            # (seq_len, hidden_dim//2) - UNPERMUTED!
        ...     b=expert_gate_up_weights_fp4,    # (num_experts, 2*intermediate_dim, hidden_dim//2)
        ...     a_scale=input_scale,
        ...     b_scale=weight_scale,
        ...     alpha=expert_alpha,              # (num_experts,)
        ...     tile_idx_to_expert_idx=tile_map,
        ...     tile_idx_to_mn_limit=mn_limit,
        ...     token_id_mapping=token_map,
        ...     num_non_exiting_tiles=num_tiles,
        ...     topk=topk,
        ... )  # out shape: (valid_m, intermediate_dim)
    """
    # Validate inputs
    assert a.device.type == "cuda", "Input tensors must be on CUDA device"
    assert b.device.type == "cuda", "Input tensors must be on CUDA device"

    # Get dimensions
    seq_len = a.shape[0]
    num_experts = b.shape[0]
    n = b.shape[1]  # This is 2*intermediate_size
    k = a.shape[1]
    if ab_dtype == "float4_e2m1fn":
        k = k * 2  # FP4 is packed 2 elements per byte

    intermediate_size = n // 2  # Output dimension after SwiGLU
    permuted_m = token_id_mapping.shape[0]

    # Check compute capability
    major, minor = get_compute_capability(a.device)
    if major != 10:
        raise ValueError(
            f"Blockscaled contiguous gather grouped GEMM with SwiGLU requires SM100 family (Blackwell: SM100, SM103, SM110). "
            f"Got SM{major}{minor}."
        )

    # Validate configuration
    ab_dtype_cutlass = get_cutlass_dtype(ab_dtype)
    sf_dtype_cutlass = get_cutlass_dtype(sf_dtype)
    c_dtype_cutlass = get_cutlass_dtype(c_dtype)

    if not BlockScaledContiguousGatherGroupedGemmKernel.can_implement(
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

    # Compute max active clusters (cached to avoid expensive HardwareInfo queries)
    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # Get tile_size from mma_tiler_mn
    tile_size = mma_tiler_mn[0]

    # Create raw pointers (TRT-LLM style) - allows same compiled kernel for different sizes
    a_ptr = make_ptr(
        ab_dtype_cutlass, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    b_ptr = make_ptr(
        ab_dtype_cutlass, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    a_sf_ptr = make_ptr(
        sf_dtype_cutlass, a_scale.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b_sf_ptr = make_ptr(
        sf_dtype_cutlass, b_scale.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    c_ptr = make_ptr(
        c_dtype_cutlass, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    if generate_sfc:
        c_sf_ptr = make_ptr(
            sf_dtype_cutlass,
            out_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        norm_const_ptr = make_ptr(
            cutlass.Float32, global_scale.data_ptr(), cute.AddressSpace.gmem
        )
    else:
        c_sf_ptr = None
        norm_const_ptr = None

    alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem)
    tile_idx_ptr = make_ptr(
        cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), cute.AddressSpace.gmem
    )
    mn_limit_ptr = make_ptr(
        cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem
    )
    token_id_ptr = make_ptr(
        cutlass.Int32, token_id_mapping.data_ptr(), cute.AddressSpace.gmem
    )
    num_tiles_ptr = make_ptr(
        cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem
    )

    # Get CUDA stream
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    # Get or compile the kernel (cached by dtype and tactic parameters)
    compiled_gemm = _get_compiled_gather_kernel(
        # Runtime parameters (problem dimensions)
        orig_m=seq_len,
        permuted_m=permuted_m,
        n=n,
        k=k,
        num_experts=num_experts,
        # Tensor pointers (order must match wrapper signature)
        a_ptr=a_ptr,
        b_ptr=b_ptr,
        a_sf_ptr=a_sf_ptr,
        b_sf_ptr=b_sf_ptr,
        c_ptr=c_ptr,
        c_sf_ptr=c_sf_ptr,
        alpha_ptr=alpha_ptr,
        tile_idx_ptr=tile_idx_ptr,
        mn_limit_ptr=mn_limit_ptr,
        token_id_ptr=token_id_ptr,
        num_tiles_ptr=num_tiles_ptr,
        norm_const_ptr=norm_const_ptr,
        max_active_clusters=max_active_clusters,
        stream=stream,
        # Dtype parameters (compile-time, in cache key)
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        # Tactic parameters (compile-time, cached)
        sf_vec_size=sf_vec_size,
        tile_size=tile_size,
        topk=topk,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vectorized_f32=vectorized_f32,
        raster_along_m=raster_along_m,
    )

    # Execute kernel with runtime parameters
    # Order must match wrapper signature:
    # (a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, c_sf_ptr, alpha_ptr,
    #  tile_idx_ptr, mn_limit_ptr, token_id_ptr, num_tiles_ptr, global_sf_ptr,
    #  orig_m, m, n, k, l, stream)
    compiled_gemm(
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        c_ptr,
        c_sf_ptr,
        alpha_ptr,
        tile_idx_ptr,
        mn_limit_ptr,
        token_id_ptr,
        num_tiles_ptr,
        norm_const_ptr,
        seq_len,  # orig_m
        permuted_m,
        n,
        k,
        num_experts,
        stream=stream,
    )

    return out, out_scale if generate_sfc else None
