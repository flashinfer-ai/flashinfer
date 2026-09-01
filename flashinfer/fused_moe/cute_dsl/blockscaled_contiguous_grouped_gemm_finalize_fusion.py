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
Contiguous Grouped GEMM kernel with Finalize Fusion for MoE workloads.

This module provides a FlashInfer-style API wrapper around the CuteDSL
grouped GEMM kernel with fused finalize operation designed for MoE GEMM2 layers:
- Input A: (permuted_m, k) - permuted activations from GEMM1
- Input B: (num_experts, n, k) - expert down projection weights
- Output C: finalized token rows or unfinalized expanded token/top-k rows

Key features:
- NVFP4 x NVFP4 or MXFP8 x MXFP4 grouped GEMM with FP8 scale factors
- Optional fused finalize operation in epilogue:
  a) Map permuted rows to (token_idx, topk_idx) using permuted_idx_to_expanded_idx
  b) Apply router scale: scaled_output = gemm_output * token_final_scales[token_idx, topk_idx]
  c) Scatter-reduce to output: out[token_idx] += scaled_output (atomic add)
- Deterministic mode writes unique expanded rows for a fixed-order moe_unpermute
- Persistent tile scheduling with per-expert group mapping
- Warp specialization for overlapped memory and compute
- Support for SM100 (Blackwell) and SM107 (Rubin) architectures
"""

import functools
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch

from flashinfer.utils import get_compute_capability
from flashinfer.cute_dsl.utils import (
    get_cutlass_dtype,
    cutlass_to_torch_dtype,
    get_num_sm,
    get_max_active_clusters,
    is_rubin_cute_dsl_available,
    make_ptr,
)

# Import the Blackwell (SM100) kernel implementation
from .blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
)


@functools.cache
def _sm107_finalize_kernel_cls():
    """Import the SM107 kernel lazily.

    It requires CuTe DSL >= 4.8 (``cutlass.utils.rubin_helpers``); importing at
    module scope would break FlashInfer on older DSL releases.
    """
    if not is_rubin_cute_dsl_available():
        raise NotImplementedError(
            "The SM107 (Rubin) CuTe DSL finalize-fusion grouped GEMM requires "
            "CuTe DSL >= 4.8, which provides cutlass.utils.rubin_helpers; the "
            "installed CuTe DSL does not have it."
        )
    from .rubin.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
        Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
    )

    return Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel


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
    a_per_token_scale_ptr,
    max_active_clusters: int,
    stream,
    # Tactic parameters (compile-time - IN cache key)
    sf_vec_size: int,
    tile_size: int,
    cluster_shape_mn: Tuple[int, int],
    raster_along_m: bool,
    a_dtype: type,
    b_dtype: type,
    sf_dtype: type,
    out_dtype: type,
    final_scale_dtype: type,
    # Blackwell-specific
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    # Rubin-specific
    mma_tiler: Optional[Tuple[int, int, int]] = None,
    mma_inst_shape: Optional[Tuple[int, int, int]] = None,
    # PDL control
    enable_pdl: bool = True,
    use_a_per_token_scale: bool = False,
    use_fused_finalize: bool = True,
):
    """Get or compile the grouped GEMM with finalize fusion kernel.

    Supports both Blackwell (SM100, via mma_tiler_mn) and Rubin (SM107,
    via mma_tiler + mma_inst_shape) architectures.

    This function caches compiled kernels by tactic and dtype parameters.
    Problem dimensions (m, n, k, num_experts) are runtime parameters.

    This matches TRT-LLM's approach where the same compiled kernel can be
    reused for different problem sizes, significantly reducing JIT compilation
    overhead during autotuning.
    """
    global _finalize_kernel_cache

    is_rubin = mma_tiler is not None and mma_inst_shape is not None

    # Cache key includes tactic and pointer dtype parameters, NOT problem dimensions.
    cache_key = (
        "sm107" if is_rubin else "sm100",
        sf_vec_size,
        tile_size,
        topk,
        mma_tiler if is_rubin else mma_tiler_mn,
        mma_inst_shape if is_rubin else None,
        cluster_shape_mn,
        raster_along_m,
        a_dtype,
        b_dtype,
        sf_dtype,
        out_dtype,
        final_scale_dtype,
        enable_pdl,
        use_a_per_token_scale,
        use_fused_finalize,
    )

    if cache_key not in _finalize_kernel_cache:
        if is_rubin:
            if use_a_per_token_scale:
                raise NotImplementedError(
                    "use_a_per_token_scale (per-token activation scale) is "
                    "not supported by the Rubin (SM107) finalize grouped "
                    "GEMM kernel yet: its wrapper has no "
                    "a_per_token_scale_ptr parameter."
                )
            if not use_fused_finalize:
                # The Rubin kernel takes no use_fused_finalize parameter: it
                # always does the atomic scatter-add into token rows.  The
                # caller allocates `out` with torch.empty (not torch.zeros)
                # and shape (seq_len * topk, n) in the unfused case, so
                # silently ignoring this would accumulate into uninitialized
                # memory at the wrong shape.
                raise NotImplementedError(
                    "use_fused_finalize=False is not supported by the Rubin "
                    "(SM107) finalize grouped GEMM kernel: it always performs "
                    "the fused atomic scatter-add."
                )
            if final_scale_dtype is not cutlass.Float32:
                # The Rubin kernel hardcodes self.final_scale_dtype =
                # cutlass.Float32; its can_implement never sees this dtype.
                raise NotImplementedError(
                    "The Rubin (SM107) finalize grouped GEMM kernel supports "
                    "only Float32 router scales, got "
                    f"{final_scale_dtype}."
                )
            gemm_rubin = _sm107_finalize_kernel_cls()(
                sf_vec_size=sf_vec_size,
                mma_inst_shape=mma_inst_shape,
                mma_tiler=mma_tiler,
                cluster_shape_mn=cluster_shape_mn,
                raster_along_m=raster_along_m,
                topK=topk,
                enable_pdl=enable_pdl,
            )
            wrapper_fn = gemm_rubin.wrapper  # type: ignore[attr-defined]
        else:
            gemm_bw = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
                sf_vec_size=sf_vec_size,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                raster_along_m=raster_along_m,
                enable_pdl=enable_pdl,
                use_a_per_token_scale=use_a_per_token_scale,
                use_fused_finalize=use_fused_finalize,
            )
            wrapper_fn = gemm_bw.wrapper

        # Compile with runtime parameters - they can vary across calls.
        # Order must match the wrapper signature, and the two wrappers have
        # DIFFERENT arities: the Blackwell wrapper takes a_per_token_scale_ptr
        # (12 pointers), the Rubin SM107 wrapper does not (11 pointers).
        # Passing the extra pointer to the SM107 wrapper shifts every argument
        # one slot ("multiple values for argument 'tile_size'").
        # (a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, alpha_ptr,
        #  tile_idx_to_group_idx_ptr, tile_idx_to_mn_limit_ptr,
        #  permuted_idx_to_expanded_idx_ptr, num_non_exiting_tiles_ptr,
        #  token_final_scales_ptr, [a_per_token_scale_ptr],
        #  m, n, k, l, num_tokens, top_k,
        #  tile_size, scaling_vector_size, max_active_clusters, stream)
        compiled_gemm = cute.compile(
            wrapper_fn,
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
            *([] if is_rubin else [a_per_token_scale_ptr]),
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


def blockscaled_contiguous_grouped_gemm_finalize_fusion(
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
    a_per_token_scale: Optional[torch.Tensor] = None,
    a_dtype: str,
    b_dtype: str,
    sf_dtype: str = "float8_e4m3fn",
    out_dtype: str = "bfloat16",
    sf_vec_size: int = 16,
    mma_tiler_mn: Tuple[int, int] = (256, 128),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    raster_along_m: bool = False,
    sm_count: Optional[int] = None,
    # Rubin-specific parameters (optional; when set, use SM107 kernel)
    mma_tiler: Optional[Tuple[int, int, int]] = None,
    mma_inst_shape: Optional[Tuple[int, int, int]] = None,
    enable_pdl: bool = True,
    use_fused_finalize: bool = True,
) -> torch.Tensor:
    """Blockscaled contiguous grouped GEMM for MoE GEMM2 workloads.

    Fused mode applies routing weights and atomically reduces into token rows.
    Deterministic mode writes expanded rows before routing-weight reduction.

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
        out: Optional output tensor. Shape is ``(seq_len, n)`` in fused mode
             and ``(seq_len * topk, n)`` in deterministic mode. In fused mode,
             a provided buffer must already be zero-initialized.
        a_per_token_scale: Optional per-row operand-A scale, shape (permuted_m,).
             Used when GEMM1 output is quantized by a standalone per-token
             W4A4 quantizer instead of the fused GEMM1 epilogue.
        a_dtype: Data type for the A matrix.
        b_dtype: Data type for the B matrix.
        sf_dtype: Data type for scale factors. Default: "float8_e4m3fn"
        out_dtype: Data type for output matrix. Default: "bfloat16"
        sf_vec_size: Scale factor vector size. Use 16 for W4A4 or 32 for W4A8.
        mma_tiler_mn: MMA tile shape (M, N). Default: (256, 128)
        cluster_shape_mn: Cluster shape (ClusterM, ClusterN). Default: (2, 1)
        raster_along_m: If True, raster tiles along M dimension. Default: False
        sm_count: Number of SMs to use. Default: max available.
        use_fused_finalize: Use atomic fused finalize; otherwise write expanded
             rows for deterministic reduction. Default: True.

    Returns:
        out: Output tensor with dtype out_dtype. The shape is ``(seq_len, n)``
             in fused mode and ``(seq_len * topk, n)`` otherwise.

    Notes:
        - A caller-provided fused output must be zero-initialized.
        - Call create_finalize_fusion_tensors() to create permuted_idx_to_expanded_idx and token_final_scales.
        - Supports SM100/SM103, plus W4A4 on SM107 with Rubin tactic parameters.
        - Deterministic mode requires a separate ``moe_unpermute`` call.

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
        >>> out = blockscaled_contiguous_grouped_gemm_finalize_fusion(
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
        ...     a_dtype="float4_e2m1fn",
        ...     b_dtype="float4_e2m1fn",
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
    if a_dtype == "float4_e2m1fn":
        k = k * 2  # FP4 is packed 2 elements per byte
    b_k = b.shape[2]
    if b_dtype == "float4_e2m1fn":
        b_k = b_k * 2
    if b_k != k:
        raise ValueError(
            f"A and B logical K dimensions must match, got A K={k} and B K={b_k}"
        )

    seq_len = token_final_scales.shape[0]
    topk = token_final_scales.shape[1]

    use_a_per_token_scale = a_per_token_scale is not None
    if use_a_per_token_scale:
        if a_per_token_scale.device.type != "cuda":
            raise ValueError("a_per_token_scale must be on CUDA device")
        if a_per_token_scale.dtype != torch.float32:
            raise ValueError("a_per_token_scale must have dtype torch.float32")
        if not a_per_token_scale.is_contiguous():
            raise ValueError("a_per_token_scale must be contiguous")
        if a_per_token_scale.shape != (permuted_m,):
            raise ValueError(
                f"a_per_token_scale must have shape ({permuted_m},), "
                f"got {tuple(a_per_token_scale.shape)}"
            )

    # Check compute capability
    major, minor = get_compute_capability(a.device)
    is_rubin = mma_tiler is not None and mma_inst_shape is not None
    if major != 10:
        raise ValueError(
            f"Blockscaled contiguous grouped GEMM with finalize fusion requires SM10x family. "
            f"Got SM{major}{minor}."
        )
    # See the matching check in the gather/act-fusion entry point: is_rubin is
    # inferred from the tactic parameters and can disagree with the device.
    if is_rubin and minor != 7:
        raise ValueError(
            f"mma_tiler/mma_inst_shape select the Rubin (SM107) finalize "
            f"kernel, but the device is SM{major}{minor}."
        )
    if not is_rubin and minor == 7:
        raise ValueError(
            "SM107 requires the Rubin tactic parameters mma_tiler and mma_inst_shape."
        )

    # Validate configuration
    a_dtype_cutlass = get_cutlass_dtype(a_dtype)
    b_dtype_cutlass = get_cutlass_dtype(b_dtype)
    sf_dtype_cutlass = get_cutlass_dtype(sf_dtype)
    out_dtype_cutlass = get_cutlass_dtype(out_dtype)
    # Token final scales - determine dtype
    if token_final_scales.dtype == torch.float32:
        token_scales_dtype = cutlass.Float32
    elif token_final_scales.dtype == torch.bfloat16:
        token_scales_dtype = cutlass.BFloat16
    else:
        token_scales_dtype = cutlass.Float16

    if is_rubin:
        can_impl = _sm107_finalize_kernel_cls().can_implement(
            a_dtype=a_dtype_cutlass,
            b_dtype=b_dtype_cutlass,
            sf_dtype=sf_dtype_cutlass,
            sf_vec_size=sf_vec_size,
            c_dtype=out_dtype_cutlass,
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
        can_impl = (
            Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.can_implement(
                a_dtype_cutlass,
                b_dtype_cutlass,
                sf_dtype_cutlass,
                sf_vec_size,
                out_dtype_cutlass,
                token_scales_dtype,
                mma_tiler_mn,
                cluster_shape_mn,
                permuted_m,
                n,
                k,
                num_experts,
                a_major="k",
                b_major="k",
                out_major="n",
            )
        )
    if not can_impl:
        raise ValueError(
            f"Unsupported configuration: a_dtype={a_dtype}, b_dtype={b_dtype}, "
            f"sf_dtype={sf_dtype}, "
            f"sf_vec_size={sf_vec_size}, out_dtype={out_dtype}, "
            f"final_scale_dtype={token_final_scales.dtype}, "
            f"mma_tiler_mn={mma_tiler_mn}, mma_tiler={mma_tiler}, mma_inst_shape={mma_inst_shape}, "
            f"cluster_shape_mn={cluster_shape_mn}, shape=({permuted_m}, {n}, {k}, {num_experts})"
        )

    output_rows = seq_len if use_fused_finalize else seq_len * topk

    # Atomic fused finalize requires zero-initialized output.
    if out is None:
        allocator = torch.zeros if use_fused_finalize else torch.empty
        out = allocator(
            (output_rows, n),
            dtype=cutlass_to_torch_dtype(out_dtype_cutlass),
            device=a.device,
        )
    else:
        expected_out_dtype = cutlass_to_torch_dtype(out_dtype_cutlass)
        if out.shape != (output_rows, n):
            raise ValueError(
                f"out must have shape ({output_rows}, {n}), got {tuple(out.shape)}"
            )
        if out.dtype != expected_out_dtype:
            raise TypeError(
                f"out must have dtype {expected_out_dtype}, got {out.dtype}"
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

    # Token final scales - create pointer
    token_scales_ptr = make_ptr(
        token_scales_dtype,
        token_final_scales.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    if use_a_per_token_scale:
        a_per_token_scale_ptr = make_ptr(
            cutlass.Float32,
            a_per_token_scale.data_ptr(),
            cute.AddressSpace.gmem,
        )
    else:
        a_per_token_scale_ptr = None

    # Get CUDA stream
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled_gemm = _get_compiled_finalize_kernel(
        seq_len=seq_len,
        permuted_m=permuted_m,
        n=n,
        k=k,
        num_experts=num_experts,
        topk=topk,
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
        a_per_token_scale_ptr=a_per_token_scale_ptr,
        max_active_clusters=max_active_clusters,
        stream=stream,
        sf_vec_size=sf_vec_size,
        tile_size=tile_size,
        cluster_shape_mn=cluster_shape_mn,
        raster_along_m=raster_along_m,
        a_dtype=a_dtype_cutlass,
        b_dtype=b_dtype_cutlass,
        sf_dtype=sf_dtype_cutlass,
        out_dtype=out_dtype_cutlass,
        final_scale_dtype=token_scales_dtype,
        mma_tiler_mn=mma_tiler_mn if not is_rubin else None,
        mma_tiler=mma_tiler if is_rubin else None,
        mma_inst_shape=mma_inst_shape if is_rubin else None,
        enable_pdl=enable_pdl,
        use_fused_finalize=use_fused_finalize,
        use_a_per_token_scale=use_a_per_token_scale,
    )

    # Execute kernel with runtime parameters.
    # Order must match the wrapper signature; the Rubin SM107 wrapper has no
    # a_per_token_scale_ptr parameter (see the arity note at the compile site),
    # so on Rubin the extra pointer must be omitted here too.
    # (a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, alpha_ptr, tile_idx_ptr,
    #  mn_limit_ptr, permuted_idx_ptr, num_tiles_ptr, token_scales_ptr,
    #  [a_per_token_scale_ptr], m, n, k, l, num_tokens, top_k, stream)
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
        *([] if is_rubin else [a_per_token_scale_ptr]),
        permuted_m,
        n,
        k,
        num_experts,
        seq_len,
        topk,
        stream=stream,
    )

    return out


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
    a_per_token_scale: Optional[torch.Tensor] = None,
    ab_dtype: str = "float4_e2m1fn",
    sf_dtype: str = "float8_e4m3fn",
    out_dtype: str = "bfloat16",
    sf_vec_size: int = 16,
    mma_tiler_mn: Tuple[int, int] = (256, 128),
    # Rubin (SM107) tactics carry a 3D tiler plus an MMA instruction shape;
    # left as None on Blackwell, which uses mma_tiler_mn above.
    mma_tiler: Optional[Tuple[int, int, int]] = None,
    mma_inst_shape: Optional[Tuple[int, int, int]] = None,
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    raster_along_m: bool = False,
    sm_count: Optional[int] = None,
    enable_pdl: bool = True,
    use_fused_finalize: bool = True,
) -> torch.Tensor:
    """Run the existing homogeneous NVFP4 GEMM2 finalize kernel."""
    warnings.warn(
        "blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4 is "
        "deprecated; use blockscaled_contiguous_grouped_gemm_finalize_fusion "
        "instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return blockscaled_contiguous_grouped_gemm_finalize_fusion(
        a,
        b,
        a_scale,
        b_scale,
        alpha,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        token_final_scales,
        out,
        a_per_token_scale=a_per_token_scale,
        a_dtype=ab_dtype,
        b_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        out_dtype=out_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        mma_tiler=mma_tiler,
        mma_inst_shape=mma_inst_shape,
        cluster_shape_mn=cluster_shape_mn,
        raster_along_m=raster_along_m,
        sm_count=sm_count,
        enable_pdl=enable_pdl,
        use_fused_finalize=use_fused_finalize,
    )


def blockscaled_contiguous_grouped_gemm_finalize_fusion_mxfp8_mxfp4(
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
    mma_tiler_mn: Tuple[int, int] = (256, 128),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    raster_along_m: bool = False,
    sm_count: Optional[int] = None,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """Run GEMM2 finalize with MXFP8 activations and packed MXFP4 weights.

    ``a`` contains E4M3 values, while ``b`` contains two packed E2M1 values
    per byte. Both scale tensors use the MMA-compatible E8M0 block-32 layout.
    The finalized scatter-reduced output is BF16. The problem N dimension must
    be divisible by 128 and by ``mma_tiler_mn[1]`` because the current finalize
    epilogue does not predicate a partial N tile.
    """
    warnings.warn(
        "blockscaled_contiguous_grouped_gemm_finalize_fusion_mxfp8_mxfp4 is "
        "deprecated; use blockscaled_contiguous_grouped_gemm_finalize_fusion "
        "instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError(f"Expected A rank 2 and B rank 3, got {a.ndim} and {b.ndim}")
    if a.dtype is not torch.float8_e4m3fn:
        raise TypeError(f"MXFP8 A must have dtype torch.float8_e4m3fn, got {a.dtype}")
    if b.dtype is not torch.uint8:
        raise TypeError(f"Packed MXFP4 B must have dtype torch.uint8, got {b.dtype}")
    if a_scale.dtype is not torch.uint8 or b_scale.dtype is not torch.uint8:
        raise TypeError("MXFP8/MXFP4 E8M0 scale tensors must have dtype torch.uint8")
    if alpha.dtype is not torch.float32:
        raise TypeError(f"alpha must have dtype torch.float32, got {alpha.dtype}")
    if token_final_scales.dtype is not torch.float32:
        raise TypeError("MXFP8/MXFP4 token_final_scales must have dtype torch.float32")
    int32_tensors = {
        "tile_idx_to_expert_idx": tile_idx_to_expert_idx,
        "num_non_exiting_tiles": num_non_exiting_tiles,
        "tile_idx_to_mn_limit": tile_idx_to_mn_limit,
        "permuted_idx_to_expanded_idx": permuted_idx_to_expanded_idx,
    }
    for name, tensor in int32_tensors.items():
        if tensor.dtype is not torch.int32:
            raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    contiguous_tensors = {
        "a": a,
        "b": b,
        "a_scale": a_scale,
        "alpha": alpha,
        "token_final_scales": token_final_scales,
        **int32_tensors,
    }
    for name, tensor in contiguous_tensors.items():
        if tensor.device != a.device:
            raise ValueError(f"{name} must be on {a.device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    # b_scale is the standard non-contiguous logical view returned by
    # convert_sf_to_mma_layout; its underlying physical storage is canonical.
    if b_scale.device != a.device:
        raise ValueError(f"b_scale must be on {a.device}, got {b_scale.device}")
    if out is not None and out.dtype is not torch.bfloat16:
        raise TypeError(
            f"MXFP8/MXFP4 GEMM2 output must have dtype torch.bfloat16, got {out.dtype}"
        )
    if out is not None:
        if out.device != a.device:
            raise ValueError(f"out must be on {a.device}, got {out.device}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    return blockscaled_contiguous_grouped_gemm_finalize_fusion(
        a,
        b,
        a_scale,
        b_scale,
        alpha,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        token_final_scales,
        out,
        a_dtype="float8_e4m3fn",
        b_dtype="float4_e2m1fn",
        sf_dtype="float8_e8m0fnu",
        out_dtype="bfloat16",
        sf_vec_size=32,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        raster_along_m=raster_along_m,
        sm_count=sm_count,
        enable_pdl=enable_pdl,
    )
