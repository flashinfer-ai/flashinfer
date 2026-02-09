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

# This file wraps TensorRT-LLM's CuteDSL grouped GEMM implementation:
# tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm.py
#
# Original copyright:
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Contiguous Grouped GEMM kernel for MoE (Mixture of Experts) workloads on Blackwell GPUs.

This module provides a FlashInfer-style API wrapper around the TensorRT-LLM CuteDSL
grouped GEMM kernel designed for MoE layers:
- Input A: (permuted_m, k) - permuted tokens from all batches
- Input B: (num_experts, n, k) - expert weights
- Output C: (permuted_m, n) - intermediate outputs

Key features:
- NVFP4 x NVFP4 grouped GEMM with FP8 scale factors
- Persistent tile scheduling with per-expert group mapping
- Warp specialization for overlapped memory and compute
- Support for SM100 (Blackwell) architecture
"""

from typing import Optional, Tuple, Type

import cutlass
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
from .blackwell.blockscaled_contiguous_grouped_gemm import (
    Sm100BlockScaledContiguousGroupedGemmKernel,
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L,
)

# Re-export the kernel class


def create_tile_mapping(
    group_m_list: torch.Tensor,
    mma_tiler_m: int,
    permuted_m: Optional[int] = None,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create tile-to-group mapping for contiguous grouped GEMM.

    This function creates the necessary mapping tensors for the grouped GEMM kernel:
    - tile_idx_to_group_idx: Maps each tile to its expert/group index
    - num_non_exiting_tiles: Number of valid tiles (for early exit)
    - aligned_group_m: Aligned M values for each group

    Args:
        group_m_list: 1D tensor of M values for each group (expert)
        mma_tiler_m: CTA tile M size (e.g., 128 or 256)
        permuted_m: Optional padded M dimension for CUDA graph support

    Returns:
        Tuple of:
        - valid_m: Total valid M (sum of aligned group M values)
        - aligned_group_m: Aligned M values per group
        - tile_idx_to_group_idx: Tensor mapping tile index to group index
        - num_non_exiting_tiles: Scalar tensor with number of valid tiles

    Example:
        >>> group_m = torch.tensor([256, 128, 384], device="cuda")
        >>> valid_m, aligned_m, tile_map, num_tiles = create_tile_mapping(group_m, mma_tiler_m=128)
    """
    device = group_m_list.device
    group_m_list_cpu = group_m_list.cpu().tolist()

    valid_m = 0
    aligned_group_m_list = []
    tile_idx_to_group_idx_list = []

    for i, group_m in enumerate(group_m_list_cpu):
        # Align each group's M to the MMA tiler M size
        aligned_group_m = ((group_m + mma_tiler_m - 1) // mma_tiler_m) * mma_tiler_m
        valid_m += aligned_group_m
        aligned_group_m_list.append(aligned_group_m)

        # Calculate number of tiles for this group
        num_tiles_in_group = aligned_group_m // mma_tiler_m
        # Add group index for each tile in this group
        tile_idx_to_group_idx_list.extend([i] * num_tiles_in_group)

    num_non_exiting_tiles = len(tile_idx_to_group_idx_list)

    # Apply padding if requested (for CUDA graph support)
    if permuted_m is not None:
        if permuted_m < valid_m:
            raise ValueError(
                f"permuted_m ({permuted_m}) must be >= valid_m ({valid_m}). "
                f"Cannot pad to a smaller size."
            )
        if permuted_m > valid_m:
            num_padding_tiles = (permuted_m - valid_m) // mma_tiler_m
            # Pad with invalid index (these tiles won't be accessed)
            tile_idx_to_group_idx_list.extend([int(-2e9)] * num_padding_tiles)

    tile_idx_to_group_idx = torch.tensor(
        tile_idx_to_group_idx_list, device=device, dtype=torch.int32
    )
    num_non_exiting_tiles_tensor = torch.tensor(
        [num_non_exiting_tiles], device=device, dtype=torch.int32
    )
    aligned_group_m = torch.tensor(
        aligned_group_m_list, device=device, dtype=torch.int32
    )

    return valid_m, aligned_group_m, tile_idx_to_group_idx, num_non_exiting_tiles_tensor


def create_scale_factor_tensor(
    l: int,
    mn: int,
    k: int,
    sf_vec_size: int,
    dtype: Type[cutlass.Numeric],
) -> Tuple[torch.Tensor, cute.Tensor, torch.Tensor]:
    """Create scale factor tensors in the MMA-compatible layout.

    This function creates scale factor tensors with the proper layout for
    the blockscaled GEMM kernel. The layout follows the MMA specification:
    (32, 4, rest_m, 4, rest_k, l) order.

    Args:
        l: Batch/expert dimension
        mn: M or N dimension
        k: K dimension
        sf_vec_size: Scale factor vector size (16 for NVF4, 32 for MXF4)
        dtype: Scale factor data type (e.g., cutlass.Float8E4M3FN)

    Returns:
        Tuple of:
        - ref_f32_torch_tensor: Reference tensor in (mn, k, l) layout for validation
        - cute_tensor: CuTe tensor in MMA layout
        - cute_torch_tensor: PyTorch tensor backing the CuTe tensor

    Example:
        >>> ref, cute_sf, torch_sf = create_scale_factor_tensor(
        ...     l=8, mn=1024, k=4096, sf_vec_size=16, dtype=cutlass.Float8E4M3FN
        ... )
    """

    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(k, sf_vec_size)
    ref_shape = (l, mn, sf_k)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )

    ref_permute_order = (1, 2, 0)
    mma_permute_order = (3, 4, 1, 5, 2, 0)

    # Create f32 ref torch tensor (cpu)
    ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        ref_shape,
        torch.float32,
        permute_order=ref_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(
            min_val=1,
            max_val=3,
        ),
    )

    # Create f32 cute torch tensor (cpu)
    cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(
            min_val=0,
            max_val=1,
        ),
    )

    # convert ref f32 tensor to cute f32 tensor
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(ref_f32_torch_tensor_cpu),
        from_dlpack(cute_f32_torch_tensor_cpu),
    )

    cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()

    # reshape makes memory contiguous
    ref_f32_torch_tensor_cpu = (
        ref_f32_torch_tensor_cpu.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, mn, sf_k, sf_vec_size)
        .reshape(l, mn, sf_k * sf_vec_size)
        .permute(*ref_permute_order)
    )
    # prune to mkl for reference check.
    ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]

    # Create dtype cute torch tensor (cpu)
    cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
        cute_f32_torch_tensor_cpu,
        dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )

    # Convert f32 cute tensor to dtype cute tensor
    cute_tensor = cutlass_torch.convert_cute_tensor(
        cute_f32_torch_tensor,
        cute_tensor,
        dtype,
        is_dynamic_layout=True,
    )
    return ref_f32_torch_tensor_cpu, cute_tensor, cute_torch_tensor


@functools.lru_cache(maxsize=None)
def _get_compiled_kernel(
    permuted_m: int,
    n: int,
    k: int,
    num_experts: int,
    ab_dtype_name: str,
    sf_dtype_name: str,
    c_dtype_name: str,
    sf_vec_size: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
):
    """Get or compile the grouped GEMM kernel.

    This function is cached to avoid recompilation for the same parameters.
    """
    ab_dtype = get_cutlass_dtype(ab_dtype_name)
    sf_dtype = get_cutlass_dtype(sf_dtype_name)
    c_dtype = get_cutlass_dtype(c_dtype_name)

    # Create kernel instance
    gemm = Sm100BlockScaledContiguousGroupedGemmKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )

    return gemm, ab_dtype, sf_dtype, c_dtype


@flashinfer_api
def blockscaled_contiguous_grouped_gemm_nvfp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    tile_idx_to_group_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    ab_dtype: str = "float4_e2m1fn",
    sf_dtype: str = "float8_e4m3fn",
    c_dtype: str = "bfloat16",
    sf_vec_size: int = 16,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    sm_count: Optional[int] = None,
) -> torch.Tensor:
    """Blockscaled Contiguous Grouped GEMM for MoE workloads with NVFP4 quantization.

    Performs grouped matrix multiplication: C[tile] = alpha[group] * (A[tile] @ B[group])

    This kernel is designed for Mixture of Experts (MoE) layers where:
    - Tokens are permuted and contiguously arranged by expert assignment
    - Each expert has its own weight matrix
    - Per-expert alpha scaling is applied to the output

    Args:
        a: Input tensor A (permuted tokens), shape (permuted_m, k) for FP4 stored as (permuted_m, k//2) uint8
        b: Weight tensor B (expert weights), shape (num_experts, n, k) for FP4 stored as (num_experts, n, k//2) uint8
        a_scale: Scale factors for A in MMA-compatible layout
        b_scale: Scale factors for B in MMA-compatible layout
        alpha: Per-expert scaling factors, shape (num_experts,), float32
        tile_idx_to_group_idx: Mapping from tile index to expert index, shape (num_tiles,), int32
        num_non_exiting_tiles: Number of valid tiles, shape (1,), int32
        out: Optional output tensor, shape (permuted_m, n). Created if None.
        ab_dtype: Data type for A and B matrices. Default: "float4_e2m1fn"
        sf_dtype: Data type for scale factors. Default: "float8_e4m3fn"
        c_dtype: Data type for output matrix. Default: "bfloat16"
        sf_vec_size: Scale factor vector size. Default: 16 (for NVFP4)
        mma_tiler_mn: MMA tile shape (M, N). Default: (128, 128)
        cluster_shape_mn: Cluster shape (ClusterM, ClusterN). Default: (1, 1)
        sm_count: Number of SMs to use. Default: max available.

    Returns:
        Output tensor C, shape (permuted_m, n) with dtype c_dtype

    Notes:
        - Use create_tile_mapping() to create tile_idx_to_group_idx and num_non_exiting_tiles
        - Use create_scale_factor_tensor() to create properly formatted scale factors
        - Requires SM100 (Blackwell) GPU architecture
        - For CUDA graph support, pre-allocate output and use fixed-size tile mapping

    Example:
        >>> # Setup for MoE with 8 experts
        >>> num_experts, hidden_dim, intermediate_dim = 8, 4096, 14336
        >>>
        >>> # Create tile mapping from routing decisions
        >>> group_m = torch.tensor([256, 128, 384, 256, 128, 256, 256, 384], device="cuda")
        >>> valid_m, aligned_m, tile_map, num_tiles = create_tile_mapping(group_m, mma_tiler_m=128)
        >>>
        >>> # Run grouped GEMM
        >>> out = blockscaled_contiguous_grouped_gemm_nvfp4(
        ...     a=permuted_input_fp4,           # (valid_m, hidden_dim//2)
        ...     b=expert_weights_fp4,           # (num_experts, intermediate_dim, hidden_dim//2)
        ...     a_scale=input_scale,
        ...     b_scale=weight_scale,
        ...     alpha=expert_alpha,             # (num_experts,)
        ...     tile_idx_to_group_idx=tile_map,
        ...     num_non_exiting_tiles=num_tiles,
        ... )
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

    # Check compute capability
    major, minor = get_compute_capability(a.device)
    if major != 10:
        raise ValueError(
            f"Blockscaled contiguous grouped GEMM requires SM100 family (Blackwell: SM100, SM103, SM110). "
            f"Got SM{major}{minor}."
        )

    # Validate configuration
    ab_dtype_cutlass = get_cutlass_dtype(ab_dtype)
    sf_dtype_cutlass = get_cutlass_dtype(sf_dtype)
    c_dtype_cutlass = get_cutlass_dtype(c_dtype)

    if not Sm100BlockScaledContiguousGroupedGemmKernel.can_implement(
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

    # Create output tensor if not provided
    if out is None:
        out = torch.empty(
            (permuted_m, n),
            dtype=cutlass_to_torch_dtype(c_dtype_cutlass),
            device=a.device,
        )

    # Get SM count
    if sm_count is None:
        sm_count = get_num_sm(a.device)

    # Get or compile the kernel
    gemm, _, _, _ = _get_compiled_kernel(
        permuted_m=permuted_m,
        n=n,
        k=k,
        num_experts=num_experts,
        ab_dtype_name=ab_dtype,
        sf_dtype_name=sf_dtype,
        c_dtype_name=c_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )

    # Compute max active clusters (cached to avoid expensive HardwareInfo queries)
    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # Create CuTe tensors from PyTorch tensors
    # A: (permuted_m, k, 1) - single batch of permuted tokens
    # B: (n, k, num_experts) - expert weights
    # C: (permuted_m, n, 1) - output
    a_tensor = from_dlpack(a.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    b_tensor = from_dlpack(b.permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    c_tensor = from_dlpack(out.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )

    # Scale factor tensors
    sfa_tensor = from_dlpack(a_scale, assumed_align=16).mark_layout_dynamic()
    sfb_tensor = from_dlpack(b_scale, assumed_align=16).mark_layout_dynamic()

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
        tile_idx_tensor,
        num_tiles_tensor,
        alpha_tensor,
        current_stream,
    )

    return out
