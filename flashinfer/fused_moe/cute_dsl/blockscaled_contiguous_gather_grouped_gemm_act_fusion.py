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
Contiguous grouped GEMM kernel with gather and FC1 activation fusion for MoE
workloads on Blackwell GPUs.

This module provides a FlashInfer-style API wrapper around the TensorRT-LLM CuteDSL
grouped GEMM kernel with fused gather and activation designed for MoE GEMM1 layers:
- Input A: (seq_len, k) - original unpermuted tokens (no need for moe_permute!)
- Input B: expert projection weights, interleaved for gated activations
- Output C: activated outputs in permuted order

Key features:
- NVFP4 x NVFP4 and MXFP8 x MXFP4 grouped GEMM paths
- Fused gather operation using LDGSTS instructions with token_id_mapping
- Eliminates the need for a separate moe_permute kernel
- Fused FC1 activation in the epilogue
- Optional FP4 quantization of output with scale factor generation
- Persistent tile scheduling with per-expert group mapping
- Warp specialization for overlapped memory and compute
- Support for SM100 (Blackwell) and SM107 (Rubin) architectures

Comparison with non-gather activation fusion:
- Non-Gather: Requires separate moe_permute kernel, then uses TMA for contiguous A load
- Gather: Uses cp.async to gather A directly using token_id_mapping, no moe_permute needed
"""

import functools
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch

from flashinfer.tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)
from flashinfer.utils import get_compute_capability
from flashinfer.cute_dsl.utils import (
    get_cutlass_dtype,
    cutlass_to_torch_dtype,
    get_num_sm,
    get_max_active_clusters,
    is_rubin_cute_dsl_available,
    make_ptr,
)
from .moe_utils import (
    normalize_cute_dsl_moe_activation_type,
    validate_cute_dsl_moe_situ_config,
)

# Import the Blackwell (SM100) kernel implementation
from .blackwell.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
    BlockScaledContiguousGatherGroupedGemmKernel,
)


@functools.cache
def _sm107_swiglu_kernel_cls():
    """Import the SM107 kernel lazily.

    It requires CuTe DSL >= 4.8 (``cutlass.utils.rubin_helpers``); importing at
    module scope would break FlashInfer on older DSL releases.
    """
    if not is_rubin_cute_dsl_available():
        raise NotImplementedError(
            "The SM107 (Rubin) CuTe DSL gather/activation-fusion grouped GEMM "
            "requires CuTe DSL >= 4.8, which provides "
            "cutlass.utils.rubin_helpers; the installed CuTe DSL does not "
            "have it."
        )
    from .rubin.blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion import (
        Sm107BlockScaledContiguousGatherGroupedGemmSwigluFusionKernel,
    )

    return Sm107BlockScaledContiguousGatherGroupedGemmSwigluFusionKernel


def create_gather_gemm_tensors(
    seq_len: int,
    topk: int,
    group_m_list: List[int],
    mma_tiler_m: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, List[int]]:
    """Create tensors required for gather grouped GEMM.

    This function creates the mapping tensors needed for the fused gather operation
    in GEMM1 with fused activation.

    Args:
        seq_len: Number of input tokens (original sequence length before routing)
        topk: Number of experts per token
        group_m_list: List of actual (unaligned) M values per expert
        mma_tiler_m: MMA tile M dimension for alignment (128 or 256)

    Returns:
        Tuple of:
        - token_id_mapping: Maps permuted row to token_idx * topk + k_idx, shape (permuted_m,), int32
          Used by cp.async to gather from the original unpermuted A tensor.
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
    a_per_token_scale_ptr,
    max_active_clusters: int,
    stream,
    # Dtype parameters (compile-time - IN cache key)
    # cute.compile specializes on pointer types, so dtype must be in cache key
    a_dtype: str,
    b_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    quantize_output: bool,
    # Tactic parameters (compile-time - IN cache key)
    sf_vec_size: int,
    tile_size: int,
    topk: int,
    cluster_shape_mn: Tuple[int, int],
    vectorized_f32: bool,
    raster_along_m: bool,
    # Blackwell-specific
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    # Rubin-specific
    mma_tiler: Optional[Tuple[int, int, int]] = None,
    mma_inst_shape: Optional[Tuple[int, int, int]] = None,
    # PDL control
    enable_pdl: bool = True,
    activation_type: Union[int, ActivationType] = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    gated: bool = True,
    use_a_per_token_scale: bool = False,
):
    """Get or compile the gather grouped GEMM with FC1 activation fusion.

    This function caches compiled kernels by tactic and dtype parameters.
    Problem dimensions (m, n, k, num_experts) are runtime parameters.

    Supports both Blackwell (SM100, via mma_tiler_mn) and Rubin (SM107,
    via mma_tiler + mma_inst_shape) architectures.
    """
    global _gather_kernel_cache
    normalized_activation_type, expected_gated = normalize_cute_dsl_moe_activation_type(
        activation_type
    )
    if gated != expected_gated:
        raise ValueError(
            f"gated={gated} is inconsistent with activation_type "
            f"{normalized_activation_type!r}"
        )
    validate_cute_dsl_moe_situ_config(
        normalized_activation_type, situ_beta, situ_linear_beta
    )

    is_rubin = mma_tiler is not None and mma_inst_shape is not None

    cache_key = (
        "sm107" if is_rubin else "sm100",
        a_dtype,
        b_dtype,
        sf_dtype,
        c_dtype,
        quantize_output,
        sf_vec_size,
        tile_size,
        topk,
        mma_tiler if is_rubin else mma_tiler_mn,
        mma_inst_shape if is_rubin else None,
        cluster_shape_mn,
        vectorized_f32,
        raster_along_m,
        enable_pdl,
        normalized_activation_type.value,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        situ_beta,
        situ_linear_beta,
        gated,
        use_a_per_token_scale,
    )

    if cache_key not in _gather_kernel_cache:
        if is_rubin:
            # The Rubin (SM107) kernel currently only implements the gated
            # (SwiGLU) activation path with the default SwiGLU constants.
            if normalized_activation_type != ActivationType.Swiglu:
                raise NotImplementedError(
                    f"activation_type {normalized_activation_type!r} is not supported by "
                    "the Rubin (SM107) gather grouped GEMM kernel yet "
                    "(SwiGLU only)."
                )
            if (swiglu_alpha, swiglu_beta, swiglu_limit) != (
                DEFAULT_SWIGLU_ALPHA,
                DEFAULT_SWIGLU_BETA,
                DEFAULT_SWIGLU_LIMIT,
            ):
                raise NotImplementedError(
                    "Custom swiglu_alpha/swiglu_beta/swiglu_limit are not "
                    "supported by the Rubin (SM107) gather grouped GEMM "
                    "kernel yet."
                )
            if use_a_per_token_scale:
                raise NotImplementedError(
                    "use_a_per_token_scale (per-token activation scale) is "
                    "not supported by the Rubin (SM107) gather grouped GEMM "
                    "kernel yet: its wrapper has no a_per_token_scale_ptr "
                    "parameter."
                )
            gemm = _sm107_swiglu_kernel_cls()(
                sf_vec_size=sf_vec_size,
                mma_inst_shape=mma_inst_shape,
                mma_tiler=mma_tiler,
                cluster_shape_mn=cluster_shape_mn,
                vectorized_f32=vectorized_f32,
                topk=topk,
                raster_along_m=raster_along_m,
                enable_pdl=enable_pdl,
            )
        else:
            # Create kernel instance
            gemm = BlockScaledContiguousGatherGroupedGemmKernel(
                sf_vec_size=sf_vec_size,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                vectorized_f32=vectorized_f32,
                topk=topk,
                raster_along_m=raster_along_m,
                enable_pdl=enable_pdl,
                activation_type=normalized_activation_type.value,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                gated=gated,
                use_a_per_token_scale=use_a_per_token_scale,
            )
        wrapper_fn = gemm.wrapper

        # Compile with runtime parameters - they can vary across calls.
        # Order must match the wrapper signature, and the two wrappers have
        # DIFFERENT arities: the Blackwell wrapper takes a_per_token_scale_ptr
        # (13 pointers), the Rubin SM107 wrapper does not (12 pointers).
        # Passing 13 pointers to the SM107 wrapper shifts every argument one
        # slot and dies with "multiple values for argument 'tile_size'".
        # (a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, c_sf_ptr, alpha_ptr,
        #  tile_idx_to_group_idx_ptr, tile_idx_to_mn_limit_ptr, token_id_mapping_ptr,
        #  num_non_exiting_tiles_ptr, global_sf_ptr, [a_per_token_scale_ptr],
        #  orig_m, m, n, k, l, tile_size, scaling_vector_size,
        #  max_active_clusters, stream)
        compiled_gemm = cute.compile(
            wrapper_fn,
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
            *([] if is_rubin else [a_per_token_scale_ptr]),
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


def blockscaled_contiguous_gather_grouped_gemm_act_fusion(
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
    a_per_token_scale: Optional[torch.Tensor] = None,
    topk: int = 8,
    a_dtype: str = "float4_e2m1fn",
    b_dtype: str = "float4_e2m1fn",
    sf_dtype: str = "float8_e4m3fn",
    c_dtype: str = "bfloat16",
    sf_vec_size: int = 16,
    quantize_output: bool = False,
    mma_tiler_mn: Tuple[int, int] = (256, 128),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    vectorized_f32: bool = True,
    raster_along_m: bool = False,
    sm_count: Optional[int] = None,
    # Rubin-specific parameters (optional; when set, use SM107 kernel)
    mma_tiler: Optional[Tuple[int, int, int]] = None,
    mma_inst_shape: Optional[Tuple[int, int, int]] = None,
    enable_pdl: bool = True,
    activation_type: Union[int, ActivationType] = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    gated: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Blockscaled contiguous gather grouped GEMM with fused FC1 activation.

    Performs grouped matrix multiplication with fused gather and activation.

    This kernel is designed for Mixture of Experts (MoE) GEMM1 layers where:
    - Input tokens are NOT pre-permuted (no need for moe_permute kernel!)
    - The kernel gathers input tokens using token_id_mapping during cp.async load
    - Gated activations use interleaved gate and up projection weights
    - The configured activation is fused into the GEMM epilogue
    - Optional block-scaled quantization of output

    Args:
        a: Input tensor A (original unpermuted tokens), shape (seq_len, k) for FP4
           stored as (seq_len, k//2) uint8. This is the ORIGINAL unpermuted tensor!
        b: Weight tensor B. Gated activations use shape
           (num_experts, 2*intermediate_size, k), stored for FP4 as
           (num_experts, 2*intermediate_size, k//2) uint8, with interleaved
           gate and up projection weights.
        a_scale: Scale factors for A in MMA-compatible layout
        b_scale: Scale factors for B in MMA-compatible layout
        alpha: Per-expert scaling factors, shape (num_experts,), float32
        tile_idx_to_expert_idx: Mapping from tile index to expert index, shape (num_tiles,), int32
        tile_idx_to_mn_limit: M limit for each tile for boundary checking, shape (num_tiles,), int32
        token_id_mapping: Mapping from permuted row to token_id, shape (permuted_m,), int32
            token_id = token_idx * topk + k_idx. Invalid rows have -1.
            Used by cp.async to gather from A tensor.
        num_non_exiting_tiles: Number of valid tiles, shape (1,), int32
        out: Optional output tensor, shape (permuted_m, intermediate_size). Created if None.
             For FP4 output, shape is (permuted_m, intermediate_size//2) uint8.
        out_scale: Optional output scale factor tensor for block-scaled
            quantized output.
        global_scale: Global scale factor for FP4 output quantization, shape
            (1,), float32.
        a_per_token_scale: Optional per-token row scale for operand A,
            shape (seq_len,), float32. Indexed by the original token ID and
            applied before the fused activation.
        topk: Number of experts per token. Default: 8
        a_dtype: Data type for the A matrix.
        b_dtype: Data type for the B matrix.
        sf_dtype: Data type for scale factors. Default: "float8_e4m3fn"
        c_dtype: Data type for output matrix. Default: "bfloat16"
        sf_vec_size: Scale factor vector size. Use 16 for W4A4 or 32 for W4A8.
        quantize_output: If True, quantize the epilogue output to a
            block-scaled format and generate out_scale. This is separate from
            c_dtype because float8_e4m3fn may also be used as a plain output
            dtype without MXFP8 scale generation.
        mma_tiler_mn: MMA tile shape (M, N). Default: (256, 128)
        cluster_shape_mn: Cluster shape (ClusterM, ClusterN). Default: (2, 1)
        vectorized_f32: Use vectorized f32x2 operations. Default: True
        raster_along_m: If True, raster tiles along M dimension. Default: False
        sm_count: Number of SMs to use. Default: max available.
        enable_pdl: Enable Programmatic Dependent Launch. Default: True.
        activation_type: Activation type for the epilogue. Use
            ActivationType.Swiglu for gated SwiGLU/OAI/SiTU,
            ActivationType.GegluTanh for tanh-approximate GeGLU, and
            ActivationType.Relu2 for non-gated mode. Setting situ_beta selects
            SiTU; swiglu_oai is represented as Swiglu with non-default
            swiglu_alpha/beta/limit.
        swiglu_alpha: SwiGLU sigmoid multiplier.
        swiglu_beta: SwiGLU up-projection bias.
        swiglu_limit: SwiGLU clamp limit.
        situ_beta: When set with ActivationType.Swiglu, use the SiTU gate
            ``beta * tanh(gate / beta) * sigmoid(gate)``.
        situ_linear_beta: Optional SiTU tanh clamp for the up branch.
        gated: Whether to run the gated SwiGLU path. If False, run non-gated
            ReLU2.

    Returns:
        Tuple of:
        - out: Output tensor C, shape (permuted_m, intermediate_size) with dtype c_dtype
               For FP4 output: (permuted_m, intermediate_size//2) uint8
        - out_scale: Output scale factors if quantize_output is True, else None

    Notes:
        - Unlike the non-gather kernel, this kernel does NOT require moe_permute!
        - The A tensor is the original unpermuted input
        - The output is in permuted order (can be fed directly to GEMM2)
        - Use create_gather_gemm_tensors() to create required mapping tensors
        - Supports SM100/SM103, plus W4A4 on SM107 with Rubin tactic parameters.

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
        >>> out, _ = blockscaled_contiguous_gather_grouped_gemm_act_fusion(
        ...     a=original_input_fp4,            # (seq_len, hidden_dim//2) - UNPERMUTED!
        ...     b=expert_gate_up_weights_fp4,    # (num_experts, 2*intermediate_dim, hidden_dim//2)
        ...     a_scale=input_scale,
        ...     b_scale=weight_scale,
        ...     alpha=expert_alpha,              # (num_experts,)
        ...     tile_idx_to_expert_idx=tile_map,
        ...     tile_idx_to_mn_limit=mn_limit,
        ...     token_id_mapping=token_map,
        ...     num_non_exiting_tiles=num_tiles,
        ...     global_scale=fc2_input_scale,
        ...     a_dtype="float4_e2m1fn",
        ...     b_dtype="float4_e2m1fn",
        ...     c_dtype="float4_e2m1fn",
        ...     quantize_output=True,
        ...     topk=topk,
        ... )  # out shape: (valid_m, intermediate_dim)
    """
    # Validate inputs
    assert a.device.type == "cuda", "Input tensors must be on CUDA device"
    assert b.device.type == "cuda", "Input tensors must be on CUDA device"
    normalized_activation_type, expected_gated = normalize_cute_dsl_moe_activation_type(
        activation_type
    )
    if gated != expected_gated:
        raise ValueError(
            f"gated={gated} is inconsistent with activation_type "
            f"{normalized_activation_type!r}"
        )
    validate_cute_dsl_moe_situ_config(
        normalized_activation_type, situ_beta, situ_linear_beta
    )

    # Get dimensions
    seq_len = a.shape[0]
    num_experts = b.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    if a_dtype == "float4_e2m1fn":
        k = k * 2  # FP4 is packed 2 elements per byte
    b_k = b.shape[2] * (2 if b_dtype == "float4_e2m1fn" else 1)
    if b_k != k:
        raise ValueError(f"A and B logical K dimensions must match, got {k} and {b_k}")

    intermediate_size = n // (2 if gated else 1)
    permuted_m = token_id_mapping.shape[0]

    use_a_per_token_scale = a_per_token_scale is not None
    if use_a_per_token_scale:
        if a_per_token_scale.device.type != "cuda":
            raise ValueError("a_per_token_scale must be on CUDA device")
        if a_per_token_scale.dtype != torch.float32:
            raise ValueError("a_per_token_scale must have dtype torch.float32")
        if not a_per_token_scale.is_contiguous():
            raise ValueError("a_per_token_scale must be contiguous")
        if a_per_token_scale.shape != (seq_len,):
            raise ValueError(
                f"a_per_token_scale must have shape ({seq_len},), "
                f"got {tuple(a_per_token_scale.shape)}"
            )

    if n % 128 != 0:
        raise ValueError(f"GEMM1 output dim n={n} must be a multiple of 128.")

    # Output quantization is explicit because Float8E4M3FN can also be used as
    # an ordinary (non-block-scaled) output by other internal configurations.
    generate_sfc = quantize_output
    if generate_sfc:
        if c_dtype not in {"float4_e2m1fn", "float8_e4m3fn"}:
            raise ValueError(
                f"Output scale generation is unsupported for c_dtype={c_dtype}"
            )
        if c_dtype == "float4_e2m1fn" and global_scale is None:
            raise ValueError("global_scale is required when c_dtype is 'float4_e2m1fn'")
        # The output scale-factor tensor is laid out in whole 128-row MMA atoms.
        if permuted_m % 128 != 0:
            raise ValueError(
                f"permuted_m={permuted_m} must be padded to a multiple of 128 "
                "when generating output scale factors"
            )
    elif out_scale is not None or global_scale is not None:
        raise ValueError(
            "out_scale and global_scale are only supported when quantize_output is True"
        )

    # Check compute capability
    major, minor = get_compute_capability(a.device)
    is_rubin = mma_tiler is not None and mma_inst_shape is not None
    if major != 10:
        raise ValueError(
            f"Blockscaled contiguous gather grouped GEMM requires SM10x family. "
            f"Got SM{major}{minor}."
        )
    # is_rubin is inferred from the tactic parameters, so it can disagree with
    # the device.  The autotuner is always self-consistent (it picks tactics by
    # capability), but the public wrappers take these parameters directly --
    # catch a mismatch here rather than deep inside kernel compilation.
    if is_rubin and minor != 7:
        raise ValueError(
            f"mma_tiler/mma_inst_shape select the Rubin (SM107) kernel, but "
            f"the device is SM{major}{minor}."
        )
    if not is_rubin and minor == 7:
        raise ValueError(
            "SM107 requires the Rubin tactic parameters mma_tiler and mma_inst_shape."
        )

    # Validate configuration
    a_dtype_cutlass = get_cutlass_dtype(a_dtype)
    b_dtype_cutlass = get_cutlass_dtype(b_dtype)
    sf_dtype_cutlass = get_cutlass_dtype(sf_dtype)
    c_dtype_cutlass = get_cutlass_dtype(c_dtype)

    if is_rubin:
        can_impl = _sm107_swiglu_kernel_cls().can_implement(
            a_dtype=a_dtype_cutlass,
            b_dtype=b_dtype_cutlass,
            sf_dtype=sf_dtype_cutlass,
            sf_vec_size=sf_vec_size,
            c_dtype=c_dtype_cutlass,
            mma_inst_shape=mma_inst_shape,
            mma_tiler=mma_tiler,
            cluster_shape_mn=cluster_shape_mn,
            m=permuted_m,
            n=n,
            k=k,
            l=num_experts,
            a_major="k",
            b_major="k",
            c_major="n",
        )
    else:
        can_impl = BlockScaledContiguousGatherGroupedGemmKernel.can_implement(
            a_dtype_cutlass,
            b_dtype_cutlass,
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
        )
    if not can_impl:
        raise ValueError(
            f"Unsupported configuration: a_dtype={a_dtype}, b_dtype={b_dtype}, "
            f"sf_dtype={sf_dtype}, "
            f"sf_vec_size={sf_vec_size}, c_dtype={c_dtype}, mma_tiler_mn={mma_tiler_mn}, "
            f"mma_tiler={mma_tiler}, mma_inst_shape={mma_inst_shape}, "
            f"cluster_shape_mn={cluster_shape_mn}, shape=({permuted_m}, {n}, {k}, {num_experts})"
        )

    # Create output tensor if not provided
    if out is None:
        if generate_sfc and c_dtype == "float4_e2m1fn":
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
        scale_atom_n = sf_vec_size * 4
        scale_rest_n = (intermediate_size + scale_atom_n - 1) // scale_atom_n
        # MMA-compatible scale factor shape
        out_scale = torch.empty(
            (32, 4, permuted_m // 128, 4, scale_rest_n, 1),
            dtype=torch.uint8,
            device=a.device,
        )

    # Get SM count
    if sm_count is None:
        sm_count = get_num_sm(a.device)

    # Compute max active clusters (cached to avoid expensive HardwareInfo queries)
    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    tile_size = mma_tiler[0] if is_rubin else mma_tiler_mn[0]

    # Create raw pointers (TRT-LLM style) - allows same compiled kernel for different sizes
    a_ptr = make_ptr(
        a_dtype_cutlass, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    b_ptr = make_ptr(
        b_dtype_cutlass, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
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
        # CuTeDSL supports FP32 -> E8M0 conversion, but the MXFP8 epilogue
        # deliberately emits the exact UE8M0 codes used by mxfp8_quantize and
        # scalar-stores them as bytes. This also avoids the unsupported
        # vector<2xE8M0> conversion/store lowering; GEMM2 reinterprets the same
        # bytes as Float8E8M0FNU scale factors.
        c_sf_storage_dtype = (
            cutlass.Uint8
            if c_dtype == "float8_e4m3fn" and sf_dtype == "float8_e8m0fnu"
            else sf_dtype_cutlass
        )
        c_sf_ptr = make_ptr(
            c_sf_storage_dtype,
            out_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        norm_const_ptr = (
            make_ptr(
                cutlass.Float32,
                global_scale.data_ptr(),
                cute.AddressSpace.gmem,
            )
            if global_scale is not None
            else None
        )
    else:
        c_sf_ptr = None
        norm_const_ptr = None

    alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem)
    if use_a_per_token_scale:
        a_per_token_scale_ptr = make_ptr(
            cutlass.Float32,
            a_per_token_scale.data_ptr(),
            cute.AddressSpace.gmem,
        )
    else:
        a_per_token_scale_ptr = None
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

    compiled_gemm = _get_compiled_gather_kernel(
        orig_m=seq_len,
        permuted_m=permuted_m,
        n=n,
        k=k,
        num_experts=num_experts,
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
        a_per_token_scale_ptr=a_per_token_scale_ptr,
        max_active_clusters=max_active_clusters,
        stream=stream,
        # Dtype parameters (compile-time, in cache key)
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        quantize_output=quantize_output,
        # Tactic parameters (compile-time, cached)
        sf_vec_size=sf_vec_size,
        tile_size=tile_size,
        topk=topk,
        cluster_shape_mn=cluster_shape_mn,
        vectorized_f32=vectorized_f32,
        raster_along_m=raster_along_m,
        mma_tiler_mn=mma_tiler_mn if not is_rubin else None,
        mma_tiler=mma_tiler if is_rubin else None,
        mma_inst_shape=mma_inst_shape if is_rubin else None,
        enable_pdl=enable_pdl,
        activation_type=normalized_activation_type.value,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        gated=gated,
        use_a_per_token_scale=use_a_per_token_scale,
    )

    # Execute kernel with runtime parameters.
    # Order must match the wrapper signature; the Rubin SM107 wrapper has no
    # a_per_token_scale_ptr parameter (see the arity note at the compile site),
    # so on Rubin the extra pointer must be omitted here too or every argument
    # shifts one slot ("multiple values for argument 'stream'").
    # (a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, c_sf_ptr, alpha_ptr,
    #  tile_idx_ptr, mn_limit_ptr, token_id_ptr, num_tiles_ptr, global_sf_ptr,
    #  [a_per_token_scale_ptr], orig_m, m, n, k, l, stream)
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
        *([] if is_rubin else [a_per_token_scale_ptr]),
        seq_len,  # orig_m
        permuted_m,
        n,
        k,
        num_experts,
        stream=stream,
    )

    return out, out_scale if generate_sfc else None


def blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4(
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
    a_per_token_scale: Optional[torch.Tensor] = None,
    topk: int = 8,
    ab_dtype: str = "float4_e2m1fn",
    sf_dtype: str = "float8_e4m3fn",
    c_dtype: str = "bfloat16",
    sf_vec_size: int = 16,
    mma_tiler_mn: Tuple[int, int] = (256, 128),
    # Rubin (SM107) tactics carry a 3D tiler plus an MMA instruction shape;
    # left as None on Blackwell, which uses mma_tiler_mn above.
    mma_tiler: Optional[Tuple[int, int, int]] = None,
    mma_inst_shape: Optional[Tuple[int, int, int]] = None,
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    vectorized_f32: bool = True,
    raster_along_m: bool = False,
    sm_count: Optional[int] = None,
    enable_pdl: bool = True,
    activation_type: Union[int, ActivationType] = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    gated: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the existing NVFP4 gather GEMM1 path.

    ``ab_dtype`` is retained for API compatibility and is applied to both MMA
    operands.  FP4 output scale generation keeps the existing global-scale
    behavior.
    """
    warnings.warn(
        "blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4 is "
        "deprecated; use blockscaled_contiguous_gather_grouped_gemm_act_fusion "
        "instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return blockscaled_contiguous_gather_grouped_gemm_act_fusion(
        a,
        b,
        a_scale,
        b_scale,
        alpha,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        out,
        out_scale,
        global_scale,
        a_per_token_scale=a_per_token_scale,
        topk=topk,
        a_dtype=ab_dtype,
        b_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        quantize_output=c_dtype == "float4_e2m1fn",
        mma_tiler_mn=mma_tiler_mn,
        mma_tiler=mma_tiler,
        mma_inst_shape=mma_inst_shape,
        cluster_shape_mn=cluster_shape_mn,
        vectorized_f32=vectorized_f32,
        raster_along_m=raster_along_m,
        sm_count=sm_count,
        enable_pdl=enable_pdl,
        activation_type=activation_type,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        gated=gated,
    )


def blockscaled_contiguous_gather_grouped_gemm_act_fusion_mxfp8_mxfp4(
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
    *,
    topk: int = 8,
    mma_tiler_mn: Tuple[int, int] = (256, 128),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    vectorized_f32: bool = True,
    raster_along_m: bool = False,
    sm_count: Optional[int] = None,
    enable_pdl: bool = True,
    activation_type: Union[int, ActivationType] = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    gated: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run MXFP8 activation x MXFP4 weight GEMM1 and quantize FC1 to MXFP8.

    A is an unpermuted ``torch.float8_e4m3fn`` tensor with linear per-token
    E8M0 scale bytes.  B is packed E2M1 in ``torch.uint8`` with an
    MMA-swizzled E8M0 scale tensor.  The returned activation is E4M3 and its
    scale tensor uses the canonical 128x4 MMA layout with vector size 32.
    """
    warnings.warn(
        "blockscaled_contiguous_gather_grouped_gemm_act_fusion_mxfp8_mxfp4 "
        "is deprecated; use "
        "blockscaled_contiguous_gather_grouped_gemm_act_fusion instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError(f"Expected A rank 2 and B rank 3, got {a.ndim} and {b.ndim}")
    if a.dtype != torch.float8_e4m3fn:
        raise TypeError(f"MXFP8 A must have dtype torch.float8_e4m3fn, got {a.dtype}")
    if b.dtype != torch.uint8:
        raise TypeError(f"Packed MXFP4 B must have dtype torch.uint8, got {b.dtype}")
    if a_scale.dtype != torch.uint8 or b_scale.dtype != torch.uint8:
        raise TypeError("MXFP8/MXFP4 E8M0 scale tensors must have dtype torch.uint8")
    if alpha.dtype != torch.float32:
        raise TypeError(f"alpha must have dtype torch.float32, got {alpha.dtype}")
    int32_tensors = {
        "tile_idx_to_expert_idx": tile_idx_to_expert_idx,
        "tile_idx_to_mn_limit": tile_idx_to_mn_limit,
        "token_id_mapping": token_id_mapping,
        "num_non_exiting_tiles": num_non_exiting_tiles,
    }
    for name, tensor in int32_tensors.items():
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    tensors = {
        "a": a,
        "b": b,
        "a_scale": a_scale,
        "alpha": alpha,
        **int32_tensors,
    }
    for name, tensor in tensors.items():
        if tensor.device != a.device:
            raise ValueError(f"{name} must be on {a.device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    # convert_sf_to_mma_layout intentionally returns a non-contiguous logical
    # 6D view over canonical contiguous scale-factor storage.  Only data_ptr()
    # is consumed below, so requiring PyTorch logical contiguity would reject
    # the standard weight-scale representation used by the existing MoE path.
    if b_scale.device != a.device:
        raise ValueError(f"b_scale must be on {a.device}, got {b_scale.device}")
    if out is not None and out.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"MXFP8 GEMM1 output must have dtype torch.float8_e4m3fn, got {out.dtype}"
        )
    if out_scale is not None and out_scale.dtype != torch.uint8:
        raise TypeError("MXFP8 output scale tensor must have dtype torch.uint8")
    for name, tensor in (("out", out), ("out_scale", out_scale)):
        if tensor is not None:
            if tensor.device != a.device:
                raise ValueError(f"{name} must be on {a.device}, got {tensor.device}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

    result, result_scale = blockscaled_contiguous_gather_grouped_gemm_act_fusion(
        a,
        b,
        a_scale,
        b_scale,
        alpha,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        out,
        out_scale,
        None,
        topk=topk,
        a_dtype="float8_e4m3fn",
        b_dtype="float4_e2m1fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="float8_e4m3fn",
        sf_vec_size=32,
        quantize_output=True,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vectorized_f32=vectorized_f32,
        raster_along_m=raster_along_m,
        sm_count=sm_count,
        enable_pdl=enable_pdl,
        activation_type=activation_type,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        gated=gated,
    )
    assert result_scale is not None
    return result, result_scale
