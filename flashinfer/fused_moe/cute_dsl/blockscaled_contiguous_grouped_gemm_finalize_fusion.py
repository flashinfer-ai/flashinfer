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

# This file wraps TensorRT-LLM's CuteDSL grouped GEMM with finalize fusion:
# tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm_finalize_fusion.py
#
# Original copyright:
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Contiguous Grouped GEMM kernel with Finalize Fusion for MoE workloads on Blackwell GPUs.

This module provides a FlashInfer-style API wrapper around the TensorRT-LLM CuteDSL
grouped GEMM kernel with fused finalize operation designed for MoE GEMM2 layers:
- Input A: (permuted_m, k) - permuted activations from GEMM1
- Input B: (num_experts, n, k) - expert down projection weights
- Output C: (seq_len, n) - finalized output with atomic scatter reduction

Key features:
- NVFP4 x NVFP4 grouped GEMM with FP8 scale factors
- Fused finalize operation in epilogue:
  a) Map permuted rows to (token_idx, topk_idx) using permuted_idx_to_expanded_idx
  b) Apply router scale: scaled_output = gemm_output * token_final_scales[token_idx, topk_idx]
  c) Scatter-reduce to output: out[token_idx] += scaled_output (atomic add)
- Eliminates separate moe_unpermute kernel
- Persistent tile scheduling with per-expert group mapping
- Warp specialization for overlapped memory and compute
- Support for SM100 (Blackwell) architecture
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
from .blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
)

# Re-export the kernel class


def create_finalize_fusion_tensors(
    seq_len: int,
    topk: int,
    permuted_m: int,
    group_m_list: List[int],
    mma_tiler_mn: Tuple[int, int],
    final_scale_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create tensors required for finalize fusion.

    This function creates the mapping tensor and final scale tensor needed
    for the fused finalize operation in GEMM2.

    Args:
        seq_len: Number of output tokens (original sequence length)
        topk: Number of experts per token
        permuted_m: Total permuted M dimension (sum of aligned group sizes)
        group_m_list: List of actual (unaligned) M values per expert
        mma_tiler_mn: MMA tile shape (M, N) for alignment
        final_scale_dtype: Data type for token final scales. Default: torch.float32

    Returns:
        Tuple of:
        - permuted_idx_to_expanded_idx: Mapping tensor, shape (permuted_m,), int32
          Maps permuted row index to expanded_idx = token_idx * topk + k_idx
          Invalid rows are marked with -1.
        - token_final_scales: Router scale tensor, shape (seq_len, topk), final_scale_dtype
          Normalized routing weights for each (token, topk) pair.

    Example:
        >>> seq_len, topk, num_experts = 4096, 8, 8
        >>> group_m_list = [512, 480, 256, 320, 640, 512, 384, 704]  # Tokens per expert
        >>> permuted_m = sum(align_to(m, 256) for m in group_m_list)  # Aligned total
        >>>
        >>> permuted_idx_to_expanded_idx, token_final_scales = create_finalize_fusion_tensors(
        ...     seq_len=seq_len,
        ...     topk=topk,
        ...     permuted_m=permuted_m,
        ...     group_m_list=group_m_list,
        ...     mma_tiler_mn=(256, 128),
        ... )
    """
    m_aligned = mma_tiler_mn[0]

    # Initialize mapping tensor with -1 (invalid)
    permuted_idx_to_expanded_idx = torch.empty(
        (permuted_m,), dtype=torch.int32, device="cuda"
    ).fill_(-1)

    # Create normalized token final scales
    token_final_scales = torch.rand(
        seq_len, topk, dtype=final_scale_dtype, device="cuda"
    )
    token_final_scales = token_final_scales / token_final_scales.sum(
        dim=1, keepdim=True
    )

    start_idx = 0
    for group_idx, m_per_group in enumerate(group_m_list):
        if m_per_group > 0:
            # Sequential/Blocked assignment for better atomic add memory access
            # Experts are grouped into sets of size topk.
            # Expert Set S (experts S*topk ... S*topk+topk-1) serves a contiguous block of tokens.
            # This ensures that within an expert, we process tokens T, T+1, T+2... sequentially.

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

            permuted_idx_to_expanded_idx[start_idx : (start_idx + m_per_group)] = (
                expanded_idx
            )

        # Move to next aligned group
        m_aligned_per_group = ((m_per_group + m_aligned - 1) // m_aligned) * m_aligned
        start_idx += m_aligned_per_group

    return permuted_idx_to_expanded_idx, token_final_scales


# Kernel cache for compiled kernels (class-level to persist across calls)
_finalize_kernel_cache: Dict[Tuple, Any] = {}


def _get_compiled_finalize_kernel(
    # Problem dimensions (runtime parameters - NOT in cache key)
    seq_len: int,
    permuted_m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    # Tensor pointers (runtime parameters - NOT in cache key)
    a_ptr,
    b_ptr,
    a_sf_ptr,
    b_sf_ptr,
    c_ptr,
    alpha_ptr,
    tile_idx_ptr,
    mn_limit_ptr,
    permuted_idx_ptr,
    num_tiles_ptr,
    token_scales_ptr,
    max_active_clusters: int,
    stream,
    # Tactic parameters (compile-time - IN cache key)
    sf_vec_size: int,
    tile_size: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    raster_along_m: bool,
):
    """Get or compile the grouped GEMM with finalize fusion kernel.

    This function caches compiled kernels by tactic parameters only.
    Problem dimensions (m, n, k, num_experts) are runtime parameters.

    This matches TRT-LLM's approach where the same compiled kernel can be
    reused for different problem sizes, significantly reducing JIT compilation
    overhead during autotuning.
    """
    global _finalize_kernel_cache

    # Cache key only includes tactic parameters, NOT problem dimensions
    cache_key = (sf_vec_size, tile_size, mma_tiler_mn, cluster_shape_mn, raster_along_m)

    if cache_key not in _finalize_kernel_cache:
        # Create kernel instance
        gemm = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_blkred=True,
            raster_along_m=raster_along_m,
        )

        # Compile with runtime parameters - they can vary across calls
        # Order must match wrapper signature:
        # (a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, alpha_ptr,
        #  tile_idx_to_group_idx_ptr, tile_idx_to_mn_limit_ptr,
        #  permuted_idx_to_expanded_idx_ptr, num_non_exiting_tiles_ptr,
        #  token_final_scales_ptr, m, n, k, l, num_tokens, top_k,
        #  tile_size, scaling_vector_size, max_active_clusters, stream)
        compiled_gemm = cute.compile(
            gemm.wrapper,
            a_ptr,
            b_ptr,
            a_sf_ptr,
            b_sf_ptr,
            c_ptr,
            alpha_ptr,
            tile_idx_ptr,
            mn_limit_ptr,
            permuted_idx_ptr,
            num_tiles_ptr,
            token_scales_ptr,
            permuted_m,
            n,
            k,
            num_experts,
            seq_len,
            topk,
            tile_size=tile_size,
            scaling_vector_size=sf_vec_size,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )

        _finalize_kernel_cache[cache_key] = compiled_gemm

    return _finalize_kernel_cache[cache_key]


@flashinfer_api
def blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    token_final_scales: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    ab_dtype: str = "float4_e2m1fn",
    sf_dtype: str = "float8_e4m3fn",
    out_dtype: str = "bfloat16",
    sf_vec_size: int = 16,
    mma_tiler_mn: Tuple[int, int] = (256, 128),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    raster_along_m: bool = False,
    sm_count: Optional[int] = None,
) -> torch.Tensor:
    """Blockscaled Contiguous Grouped GEMM with Finalize Fusion for MoE workloads.

    Performs grouped matrix multiplication with fused finalize (scatter-reduce):
    out[token_idx] += alpha[group] * (A[row] @ B[group]) * router_scale[token_idx, topk_idx]

    This kernel is designed for Mixture of Experts (MoE) GEMM2 layers where:
    - Tokens are permuted and contiguously arranged by expert assignment
    - Each expert has a down projection weight matrix
    - The finalize operation (unpermute + scale + reduce) is fused into the epilogue
    - Uses atomic adds for scatter-reduction to handle tokens routed to multiple experts

    Args:
        a: Input tensor A (permuted activations), shape (permuted_m, k) for FP4 stored as (permuted_m, k//2) uint8
        b: Weight tensor B (expert down weights), shape (num_experts, n, k)
           for FP4 stored as (num_experts, n, k//2) uint8
        a_scale: Scale factors for A in MMA-compatible layout
        b_scale: Scale factors for B in MMA-compatible layout
        alpha: Per-expert scaling factors, shape (num_experts,), float32
        tile_idx_to_expert_idx: Mapping from tile index to expert index, shape (num_tiles,), int32
        num_non_exiting_tiles: Number of valid tiles, shape (1,), int32
        tile_idx_to_mn_limit: M limit for each tile, shape (num_tiles,), int32
        permuted_idx_to_expanded_idx: Mapping from permuted row to expanded index, shape (permuted_m,), int32
            expanded_idx = token_idx * topk + topk_idx. Invalid rows have -1.
        token_final_scales: Router scaling factors, shape (seq_len, topk), float32/bf16/fp16
        out: Optional output tensor, shape (seq_len, n). Created if None.
             This tensor is used for atomic accumulation, so it should be zero-initialized.
        ab_dtype: Data type for A and B matrices. Default: "float4_e2m1fn"
        sf_dtype: Data type for scale factors. Default: "float8_e4m3fn"
        out_dtype: Data type for output matrix. Default: "bfloat16"
        sf_vec_size: Scale factor vector size. Default: 16 (for NVFP4)
        mma_tiler_mn: MMA tile shape (M, N). Default: (256, 128)
        cluster_shape_mn: Cluster shape (ClusterM, ClusterN). Default: (2, 1)
        raster_along_m: If True, raster tiles along M dimension. Default: False
        sm_count: Number of SMs to use. Default: max available.

    Returns:
        out: Output tensor, shape (seq_len, n) with dtype out_dtype.
             Contains the finalized MoE output after scatter-reduce.

    Notes:
        - The output tensor is modified in-place using atomic adds for scatter-reduction.
        - Call create_finalize_fusion_tensors() to create permuted_idx_to_expanded_idx and token_final_scales.
        - Requires SM100 (Blackwell) GPU architecture
        - The finalize fusion eliminates the need for a separate moe_unpermute kernel

    Example:
        >>> # Setup for MoE GEMM2 with 8 experts
        >>> num_experts, intermediate_dim, hidden_dim = 8, 14336, 4096
        >>> seq_len, topk = 4096, 8
        >>>
        >>> # Create tile mapping from routing decisions
        >>> group_m = torch.tensor([512, 480, 256, 320, 640, 512, 384, 704], device="cuda")
        >>> valid_m, aligned_m, tile_map, num_tiles, mn_limit = create_tile_mapping_finalize(
        ...     group_m, mma_tiler_m=256
        ... )
        >>>
        >>> # Create finalize fusion tensors
        >>> permuted_idx, final_scales = create_finalize_fusion_tensors(
        ...     seq_len=seq_len, topk=topk, permuted_m=sum(aligned_m),
        ...     group_m_list=group_m.tolist(), mma_tiler_mn=(256, 128)
        ... )
        >>>
        >>> # Run grouped GEMM with finalize fusion
        >>> out = blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4(
        ...     a=gemm1_output_fp4,              # (valid_m, intermediate_dim//2)
        ...     b=expert_down_weights_fp4,       # (num_experts, hidden_dim, intermediate_dim//2)
        ...     a_scale=gemm1_output_scale,
        ...     b_scale=down_weight_scale,
        ...     alpha=expert_alpha,              # (num_experts,)
        ...     tile_idx_to_expert_idx=tile_map,
        ...     num_non_exiting_tiles=num_tiles,
        ...     tile_idx_to_mn_limit=mn_limit,
        ...     permuted_idx_to_expanded_idx=permuted_idx,
        ...     token_final_scales=final_scales,
        ... )  # out shape: (seq_len, hidden_dim)
    """
    # Validate inputs
    assert a.device.type == "cuda", "Input tensors must be on CUDA device"
    assert b.device.type == "cuda", "Input tensors must be on CUDA device"

    # Get dimensions
    permuted_m = a.shape[0]
    num_experts = b.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    if ab_dtype == "float4_e2m1fn":
        k = k * 2  # FP4 is packed 2 elements per byte

    seq_len = token_final_scales.shape[0]
    topk = token_final_scales.shape[1]

    # Check compute capability
    major, minor = get_compute_capability(a.device)
    if major != 10:
        raise ValueError(
            f"Blockscaled contiguous grouped GEMM with finalize fusion requires SM100 family (Blackwell: SM100, SM103, SM110). "
            f"Got SM{major}{minor}."
        )

    # Validate configuration
    ab_dtype_cutlass = get_cutlass_dtype(ab_dtype)
    sf_dtype_cutlass = get_cutlass_dtype(sf_dtype)
    out_dtype_cutlass = get_cutlass_dtype(out_dtype)

    if not Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.can_implement(
        ab_dtype_cutlass,
        sf_dtype_cutlass,
        sf_vec_size,
        out_dtype_cutlass,
        mma_tiler_mn,
        cluster_shape_mn,
        permuted_m,
        n,
        k,
        num_experts,
        a_major="k",
        b_major="k",
        out_major="n",
    ):
        raise ValueError(
            f"Unsupported configuration: ab_dtype={ab_dtype}, sf_dtype={sf_dtype}, "
            f"sf_vec_size={sf_vec_size}, out_dtype={out_dtype}, mma_tiler_mn={mma_tiler_mn}, "
            f"cluster_shape_mn={cluster_shape_mn}, shape=({permuted_m}, {n}, {k}, {num_experts})"
        )

    # Create output tensor if not provided (zero-initialized for atomic adds)
    if out is None:
        out = torch.zeros(
            (seq_len, n),
            dtype=cutlass_to_torch_dtype(out_dtype_cutlass),
            device=a.device,
        )
    else:
        # Ensure output is zero for proper accumulation
        out.zero_()

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
        out_dtype_cutlass, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem)
    tile_idx_ptr = make_ptr(
        cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), cute.AddressSpace.gmem
    )
    mn_limit_ptr = make_ptr(
        cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem
    )
    num_tiles_ptr = make_ptr(
        cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem
    )
    permuted_idx_ptr = make_ptr(
        cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), cute.AddressSpace.gmem
    )

    # Token final scales - determine dtype and create pointer
    if token_final_scales.dtype == torch.float32:
        token_scales_dtype = cutlass.Float32
    elif token_final_scales.dtype == torch.bfloat16:
        token_scales_dtype = cutlass.BFloat16
    else:
        token_scales_dtype = cutlass.Float16
    token_scales_ptr = make_ptr(
        token_scales_dtype,
        token_final_scales.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    # Get CUDA stream
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    # Get or compile the kernel (cached by tactic parameters only)
    compiled_gemm = _get_compiled_finalize_kernel(
        # Runtime parameters (problem dimensions)
        seq_len=seq_len,
        permuted_m=permuted_m,
        n=n,
        k=k,
        num_experts=num_experts,
        topk=topk,
        # Tensor pointers (order must match wrapper signature)
        a_ptr=a_ptr,
        b_ptr=b_ptr,
        a_sf_ptr=a_sf_ptr,
        b_sf_ptr=b_sf_ptr,
        c_ptr=c_ptr,
        alpha_ptr=alpha_ptr,
        tile_idx_ptr=tile_idx_ptr,
        mn_limit_ptr=mn_limit_ptr,
        permuted_idx_ptr=permuted_idx_ptr,
        num_tiles_ptr=num_tiles_ptr,
        token_scales_ptr=token_scales_ptr,
        max_active_clusters=max_active_clusters,
        stream=stream,
        # Tactic parameters (compile-time, cached)
        sf_vec_size=sf_vec_size,
        tile_size=tile_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        raster_along_m=raster_along_m,
    )

    # Execute kernel with runtime parameters
    # Order must match wrapper signature:
    # (a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, alpha_ptr, tile_idx_ptr,
    #  mn_limit_ptr, permuted_idx_ptr, num_tiles_ptr, token_scales_ptr,
    #  m, n, k, l, num_tokens, top_k, stream)
    compiled_gemm(
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        c_ptr,
        alpha_ptr,
        tile_idx_ptr,
        mn_limit_ptr,
        permuted_idx_ptr,
        num_tiles_ptr,
        token_scales_ptr,
        permuted_m,
        n,
        k,
        num_experts,
        seq_len,
        topk,
        stream=stream,
    )

    return out
