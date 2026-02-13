"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

CuTe-DSL RoPE Wrapper Functions
===============================

This module provides high-level Python functions that wrap CuTe-DSL kernels.
These match the original API from flashinfer/cute_dsl/rope.py.
"""

import math
from typing import Optional, Tuple

import torch

from .compile import (
    _get_compiled_kernel,
    _get_compiled_kernel_seq_heads,
    _get_compiled_kernel_with_indptr,
    _get_compiled_cos_sin_cache_kernel,
    _get_compiled_cos_sin_cache_kernel_seq_heads,
)

# Cache for GPU-specific threshold computation
_GPU_OCCUPANCY_CACHE = {}


def _get_cos_sin_cache_seq_heads_threshold(head_dim: int) -> int:
    """
    Compute occupancy-based threshold for sequential-heads kernel selection.

    This matches CUDA's BatchQKApplyRotaryPosIdsCosSinCache approach:
    - Query GPU SM count
    - Estimate max concurrent blocks based on kernel properties
    - Use sequential-heads when we have enough token blocks to saturate the GPU

    The threshold represents the minimum number of token blocks needed before
    sequential-heads becomes beneficial (due to cos/sin cache reuse across heads).

    Args:
        head_dim: Head dimension (affects threads per block)

    Returns:
        Threshold in number of tokens. Use sequential-heads when nnz >= threshold.
    """
    device = torch.cuda.current_device()
    cache_key = (device, head_dim)

    if cache_key not in _GPU_OCCUPANCY_CACHE:
        # Get GPU properties
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

        # Calculate kernel configuration (matching kernels.py)
        elems_per_thread = 8
        bdx = head_dim // elems_per_thread  # threads for head dimension
        num_threads = max(128, bdx)
        bdy = num_threads // bdx  # tokens per block

        # Estimate blocks per SM
        # CUDA uses cudaOccupancyMaxActiveBlocksPerMultiprocessor
        # For typical RoPE kernel: ~128 threads, 0 shared memory
        # Conservative estimate: 8-16 blocks per SM depending on register pressure
        # Use 8 as a safe conservative estimate (matches observed behavior)
        estimated_blocks_per_sm = 8

        # Total concurrent blocks across GPU
        num_ctas = num_sms * estimated_blocks_per_sm

        # Threshold in tokens = num_ctas * tokens_per_block
        threshold_tokens = num_ctas * bdy

        _GPU_OCCUPANCY_CACHE[cache_key] = threshold_tokens

    return _GPU_OCCUPANCY_CACHE[cache_key]


def apply_rope_cute_dsl(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 1.0,
    old_context_len: int = 8192,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE (Rotary Positional Embeddings) using CuTe-DSL backend.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape: (nnz, num_q_heads, head_dim)
    k : torch.Tensor
        Key tensor, shape: (nnz, num_k_heads, head_dim)
    pos_ids : torch.Tensor
        Position indices, shape: (nnz,)
    rotary_dim : Optional[int]
        Dimension to apply RoPE. If None, uses full head_dim.
    interleave : bool
        If True, use interleaved (GPT-J) style. If False, use non-interleaved (NeoX) style.
    rope_scale : float
        Scaling factor for RoPE frequencies.
    rope_theta : float
        Base theta value for RoPE.
    low_freq_factor : float
        Llama 3.1 low frequency factor. Default 1.0 (no scaling).
    high_freq_factor : float
        Llama 3.1 high frequency factor. Default 1.0 (no scaling).
    old_context_len : int
        Llama 3.1 original context length. Default 8192.
    q_rope : Optional[torch.Tensor]
        Pre-allocated output for rotated queries. If None, allocated internally.
    k_rope : Optional[torch.Tensor]
        Pre-allocated output for rotated keys. If None, allocated internally.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Rotated query and key tensors.

    Notes
    -----
    Llama 3.1 frequency scaling:
        When low_freq_factor != 1.0 or high_freq_factor != 1.0, applies smooth
        interpolation between scaled and unscaled frequencies. The formula is:
            smooth = clamp(freq * smooth_a + smooth_b, 0, 1)
            freq = (1 - smooth) * (freq / rope_scale) + smooth * freq
    """
    # Validate inputs
    assert q.ndim == 3, f"q must be 3D, got {q.ndim}D"
    assert k.ndim == 3, f"k must be 3D, got {k.ndim}D"
    assert q.size(0) == k.size(0), "q and k must have same nnz"
    assert q.size(2) == k.size(2), "q and k must have same head_dim"
    assert q.is_cuda, "q must be on CUDA"
    assert k.is_cuda, "k must be on CUDA"
    assert pos_ids.is_cuda, "pos_ids must be on CUDA"

    nnz = q.size(0)
    num_qo_heads = q.size(1)
    num_kv_heads = k.size(1)
    head_dim = q.size(2)

    if rotary_dim is None:
        rotary_dim = head_dim

    assert rotary_dim <= head_dim, (
        f"rotary_dim must be <= head_dim, got {rotary_dim} > {head_dim}"
    )
    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    assert head_dim % 2 == 0, "head_dim must be even"

    # Determine dtype
    dtype = q.dtype
    assert dtype in [torch.float16, torch.bfloat16], f"Unsupported dtype: {dtype}"
    dtype_str = "float16" if dtype == torch.float16 else "bfloat16"

    # Check for aliasing (same tensor as input and output)
    # TVM FFI creates a memcpy when it detects aliasing, causing 2x slowdown on B300.
    # We avoid this by using separate tensors internally, then copying back.
    q_aliased = q_rope is not None and q_rope is q
    k_aliased = k_rope is not None and k_rope is k

    # Allocate output tensors if not provided
    if q_rope is None:
        q_rope = torch.empty_like(q)
        q_aliased = False
    if k_rope is None:
        k_rope = torch.empty_like(k)
        k_aliased = False

    # If aliased, use temporary tensors to avoid TVM FFI memcpy overhead
    if q_aliased or k_aliased:
        q_rope_temp = torch.empty_like(q) if q_aliased else q_rope
        k_rope_temp = torch.empty_like(k) if k_aliased else k_rope
    else:
        q_rope_temp = q_rope
        k_rope_temp = k_rope

    # Ensure contiguous tensors
    q = q.contiguous()
    k = k.contiguous()
    q_rope_temp = q_rope_temp.contiguous()
    k_rope_temp = k_rope_temp.contiguous()

    # Determine idtype for pos_ids (int32 or int64, matching CUDA's type dispatch)
    if pos_ids.dtype == torch.int32:
        idtype_str = "int32"
    elif pos_ids.dtype == torch.int64:
        idtype_str = "int64"
    else:
        raise ValueError(f"pos_ids must be int32 or int64, got {pos_ids.dtype}")
    pos_ids = pos_ids.contiguous()

    # Compute reciprocal scale and theta
    rope_rcp_scale = 1.0 / rope_scale
    rope_rcp_theta = 1.0 / rope_theta

    # Compute Llama 3.1 smooth_a and smooth_b
    # The CUDA kernel uses:
    #   smooth_a = old_context_len / (2 * pi * (high_freq_factor - low_freq_factor))
    #   smooth_b = -1.0 / ((high_freq_factor / low_freq_factor) - 1.0)
    #            = -low_freq_factor / (high_freq_factor - low_freq_factor)
    #
    # When high_freq_factor == low_freq_factor, this causes division by zero.
    # The CUDA kernel gets inf/NaN which clamps to smooth=0, applying full scaling.
    if high_freq_factor != low_freq_factor:
        smooth_a = old_context_len / (
            2.0 * math.pi * (high_freq_factor - low_freq_factor)
        )
        smooth_b = -1.0 / ((high_freq_factor / low_freq_factor) - 1.0)
    else:
        # When factors are equal, apply full scaling (smooth=0)
        smooth_a = 0.0
        smooth_b = 0.0  # This gives smooth=0, applying full scaling

    # Adaptive kernel selection (like CUDA C++)
    # Use sequential-heads kernel for large workloads to reduce block count
    # and reuse sin/cos computation across heads
    bdy = 128 // (head_dim // 8)  # tokens per block
    num_token_blocks = (nnz + bdy - 1) // bdy

    # Threshold for switching from parallel-heads to sequential-heads kernel.
    # Sequential-heads kernel loops over heads, reducing block count but requiring
    # more work per block. It needs sufficient token parallelism to be efficient.
    # At 1024 blocks with H100 (132 SMs), we have ~8 blocks/SM which provides
    # enough parallelism for the sequential-heads kernel to be efficient.
    # Lower thresholds (256, 512) showed slowdowns at boundary conditions.
    USE_SEQ_HEADS_THRESHOLD = 1024

    if num_token_blocks >= USE_SEQ_HEADS_THRESHOLD:
        # Large workload: use sequential-heads kernel
        # This uses fewer blocks (no head dimension) and reuses sin/cos
        kernel = _get_compiled_kernel_seq_heads(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            interleave=interleave,
            dtype_str=dtype_str,
            idtype_str=idtype_str,
        )
    else:
        # Small workload: use parallel-heads kernel for better GPU utilization
        kernel = _get_compiled_kernel(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            interleave=interleave,
            dtype_str=dtype_str,
            idtype_str=idtype_str,
        )

    kernel(
        q,
        k,
        q_rope_temp,
        k_rope_temp,
        pos_ids,
        nnz,
        num_qo_heads,
        num_kv_heads,
        rope_rcp_scale,
        rope_rcp_theta,
        smooth_a,
        smooth_b,
    )

    # Copy back if we used temporary tensors to avoid aliasing
    if q_aliased:
        q.copy_(q_rope_temp)
        # Return the original tensor (q) since it now contains the result
        q_rope = q
    if k_aliased:
        k.copy_(k_rope_temp)
        # Return the original tensor (k) since it now contains the result
        k_rope = k

    return q_rope, k_rope


def _compute_pos_ids_from_indptr_offsets(
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    nnz: int,
) -> torch.Tensor:
    """
    Compute position IDs from indptr and offsets (fully vectorized on GPU).

    For token j in sequence i (where indptr[i] <= j < indptr[i+1]):
        pos_ids[j] = j - indptr[i] + offsets[i]

    Parameters
    ----------
    indptr : torch.Tensor
        Indptr tensor of shape (batch_size + 1,). indptr[i] is the start index
        of sequence i in the ragged tensor.
    offsets : torch.Tensor
        Offset tensor of shape (batch_size,). offsets[i] is the position offset
        for sequence i.
    nnz : int
        Total number of tokens (= indptr[-1]).

    Returns
    -------
    torch.Tensor
        Position IDs tensor of shape (nnz,).
    """
    device = indptr.device

    # Use searchsorted to find batch indices - fully GPU-based, no synchronization
    token_indices = torch.arange(nnz, dtype=torch.int32, device=device)

    # Find which sequence each token belongs to
    # searchsorted(indptr[1:], token_indices, right=True) gives us the batch index
    seq_indices = torch.searchsorted(indptr[1:], token_indices, right=True)

    # Compute: pos_ids[j] = j - indptr[seq_idx] + offsets[seq_idx]
    pos_ids = token_indices - indptr[seq_indices] + offsets[seq_indices]

    return pos_ids.to(torch.int32)


def apply_rope_with_indptr_cute_dsl(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 1.0,
    old_context_len: int = 8192,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE using CuTe-DSL backend with indptr/offsets (ragged tensor format).

    This API is compatible with flashinfer.apply_rope.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: (nnz, num_q_heads, head_dim)
    k : torch.Tensor
        Key ragged tensor, shape: (nnz, num_k_heads, head_dim)
    indptr : torch.Tensor
        Indptr tensor, shape: (batch_size + 1,). Defines sequence boundaries.
    offsets : torch.Tensor
        Position offsets for each sequence, shape: (batch_size,).
    rotary_dim : Optional[int]
        Dimension to apply RoPE. If None, uses full head_dim.
    interleave : bool
        If True, use interleaved (GPT-J) style. If False, use non-interleaved (NeoX) style.
    rope_scale : float
        Scaling factor for RoPE frequencies.
    rope_theta : float
        Base theta value for RoPE.
    low_freq_factor : float
        Llama 3.1 low frequency factor. Default 1.0 (no scaling).
    high_freq_factor : float
        Llama 3.1 high frequency factor. Default 1.0 (no scaling).
    old_context_len : int
        Llama 3.1 original context length. Default 8192.
    q_rope : Optional[torch.Tensor]
        Pre-allocated output for rotated queries. If None, allocated internally.
    k_rope : Optional[torch.Tensor]
        Pre-allocated output for rotated keys. If None, allocated internally.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Rotated query and key tensors.
    """
    # Validate inputs
    assert q.ndim == 3, f"q must be 3D, got {q.ndim}D"
    assert k.ndim == 3, f"k must be 3D, got {k.ndim}D"
    assert q.size(0) == k.size(0), "q and k must have same nnz"
    assert q.size(2) == k.size(2), "q and k must have same head_dim"
    assert q.is_cuda, "q must be on CUDA"
    assert k.is_cuda, "k must be on CUDA"

    num_qo_heads = q.size(1)
    num_kv_heads = k.size(1)
    head_dim = q.size(2)
    batch_size = indptr.size(0) - 1

    if rotary_dim is None:
        rotary_dim = head_dim

    # Determine dtype
    dtype = q.dtype
    assert dtype in [torch.float16, torch.bfloat16], f"Unsupported dtype: {dtype}"
    dtype_str = "float16" if dtype == torch.float16 else "bfloat16"

    # Check for aliasing (same tensor as input and output)
    # TVM FFI creates a memcpy when it detects aliasing, causing 2x slowdown on B300.
    # We avoid this by using separate tensors internally, then copying back.
    q_aliased = q_rope is not None and q_rope is q
    k_aliased = k_rope is not None and k_rope is k

    # Allocate output tensors if not provided
    if q_rope is None:
        q_rope = torch.empty_like(q)
        q_aliased = False
    if k_rope is None:
        k_rope = torch.empty_like(k)
        k_aliased = False

    # If aliased, use temporary tensors to avoid TVM FFI memcpy overhead
    if q_aliased or k_aliased:
        q_rope_temp = torch.empty_like(q) if q_aliased else q_rope
        k_rope_temp = torch.empty_like(k) if k_aliased else k_rope
    else:
        q_rope_temp = q_rope
        k_rope_temp = k_rope

    # Ensure contiguous tensors
    q = q.contiguous()
    k = k.contiguous()
    q_rope_temp = q_rope_temp.contiguous()
    k_rope_temp = k_rope_temp.contiguous()

    # Determine idtype for indptr/offsets (int32 or int64, matching CUDA's type dispatch)
    # Note: indptr and offsets must have the same dtype
    if indptr.dtype != offsets.dtype:
        raise ValueError(
            f"indptr and offsets must have same dtype, got {indptr.dtype} and {offsets.dtype}"
        )
    if indptr.dtype == torch.int32:
        idtype_str = "int32"
    elif indptr.dtype == torch.int64:
        idtype_str = "int64"
    else:
        raise ValueError(f"indptr/offsets must be int32 or int64, got {indptr.dtype}")
    indptr = indptr.contiguous()
    offsets = offsets.contiguous()

    # Compute reciprocal scale and theta
    rope_rcp_scale = 1.0 / rope_scale
    rope_rcp_theta = 1.0 / rope_theta

    # Compute Llama 3.1 smooth_a and smooth_b
    if high_freq_factor != low_freq_factor:
        smooth_a = old_context_len / (
            2.0 * math.pi * (high_freq_factor - low_freq_factor)
        )
        smooth_b = -1.0 / ((high_freq_factor / low_freq_factor) - 1.0)
    else:
        smooth_a = 0.0
        smooth_b = 0.0

    # Use the indptr-based kernel directly (no pos_ids computation needed)
    kernel = _get_compiled_kernel_with_indptr(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        interleave=interleave,
        dtype_str=dtype_str,
        idtype_str=idtype_str,
    )
    kernel(
        q,
        k,
        q_rope_temp,
        k_rope_temp,
        indptr,
        offsets,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        rope_rcp_scale,
        rope_rcp_theta,
        smooth_a,
        smooth_b,
    )

    # Copy back if we used temporary tensors to avoid aliasing
    if q_aliased:
        q.copy_(q_rope_temp)
        # Return the original tensor (q) since it now contains the result
        q_rope = q
    if k_aliased:
        k.copy_(k_rope_temp)
        # Return the original tensor (k) since it now contains the result
        k_rope = k

    return q_rope, k_rope


def apply_llama31_rope_with_indptr_cute_dsl(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8.0,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    old_context_len: int = 8192,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Llama 3.1 style RoPE using CuTe-DSL backend with indptr/offsets.

    This API is compatible with flashinfer.apply_llama31_rope.

    Parameters are the same as apply_rope_with_indptr_cute_dsl, but with
    Llama 3.1 default parameters.
    """
    return apply_rope_with_indptr_cute_dsl(
        q=q,
        k=k,
        indptr=indptr,
        offsets=offsets,
        rotary_dim=rotary_dim,
        interleave=interleave,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        old_context_len=old_context_len,
        q_rope=q_rope,
        k_rope=k_rope,
    )


def apply_rope_with_cos_sin_cache_cute_dsl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    interleave: bool = False,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE using CuTe-DSL backend with precomputed cos/sin cache.

    This is compatible with vLLM/SGLang style APIs.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape: (nnz, num_q_heads, head_dim)
    k : torch.Tensor
        Key tensor, shape: (nnz, num_k_heads, head_dim)
    cos_sin_cache : torch.Tensor
        Precomputed cos/sin cache, shape: (max_seq_len, rotary_dim).
        First half of rotary_dim contains cos, second half contains sin.
        Must be float32.
    pos_ids : torch.Tensor
        Position indices, shape: (nnz,)
    interleave : bool
        If True, use interleaved (GPT-J) style. If False, use non-interleaved (NeoX) style.
    q_rope : Optional[torch.Tensor]
        Pre-allocated output for rotated queries. If None, allocated internally.
    k_rope : Optional[torch.Tensor]
        Pre-allocated output for rotated keys. If None, allocated internally.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Rotated query and key tensors.
    """
    assert q.ndim == 3, f"q must be 3D, got {q.ndim}D"
    assert k.ndim == 3, f"k must be 3D, got {k.ndim}D"
    assert q.size(0) == k.size(0), "q and k must have same nnz"
    assert q.size(2) == k.size(2), "q and k must have same head_dim"
    assert q.is_cuda, "q must be on CUDA"
    assert k.is_cuda, "k must be on CUDA"
    assert pos_ids.is_cuda, "pos_ids must be on CUDA"
    assert cos_sin_cache.is_cuda, "cos_sin_cache must be on CUDA"
    assert cos_sin_cache.dtype == torch.float32, "cos_sin_cache must be float32"

    nnz = q.size(0)
    num_qo_heads = q.size(1)
    num_kv_heads = k.size(1)
    head_dim = q.size(2)
    rotary_dim = cos_sin_cache.size(1)

    assert rotary_dim <= head_dim, (
        f"rotary_dim must be <= head_dim, got {rotary_dim} > {head_dim}"
    )
    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    assert head_dim % 2 == 0, "head_dim must be even"

    # Determine dtype
    dtype_str = "float16" if q.dtype == torch.float16 else "bfloat16"
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"Only fp16/bf16 supported, got {q.dtype}"
    )
    assert k.dtype == q.dtype, "q and k must have same dtype"

    # Determine idtype for pos_ids (int32 or int64, matching CUDA's type dispatch)
    if pos_ids.dtype == torch.int32:
        idtype_str = "int32"
    elif pos_ids.dtype == torch.int64:
        idtype_str = "int64"
    else:
        raise ValueError(f"pos_ids must be int32 or int64, got {pos_ids.dtype}")

    # Check for aliasing (same tensor as input and output)
    # TVM FFI creates a memcpy when it detects aliasing, causing 2x slowdown on B300.
    # We avoid this by using separate tensors internally, then copying back.
    q_aliased = q_rope is not None and q_rope is q
    k_aliased = k_rope is not None and k_rope is k

    # Allocate output if not provided
    if q_rope is None:
        q_rope = torch.empty_like(q)
        q_aliased = False
    if k_rope is None:
        k_rope = torch.empty_like(k)
        k_aliased = False

    # If aliased, use temporary tensors to avoid TVM FFI memcpy overhead
    if q_aliased or k_aliased:
        q_rope_temp = torch.empty_like(q) if q_aliased else q_rope
        k_rope_temp = torch.empty_like(k) if k_aliased else k_rope
    else:
        q_rope_temp = q_rope
        k_rope_temp = k_rope

    # Ensure contiguous (no dtype conversion needed - kernel handles both int32/int64)
    q = q.contiguous()
    k = k.contiguous()
    cos_sin_cache = cos_sin_cache.contiguous()
    pos_ids = pos_ids.contiguous()
    q_rope_temp = q_rope_temp.contiguous()
    k_rope_temp = k_rope_temp.contiguous()

    # Adaptive kernel selection (matching CUDA's occupancy-based approach):
    # - Head-parallel: Better for small workloads - maximizes GPU parallelism
    # - Sequential-heads: Better for large workloads - reduces memory traffic via cache reuse
    #
    # CUDA uses cudaOccupancyMaxActiveBlocksPerMultiprocessor to determine threshold.
    # We estimate based on GPU SM count and typical kernel occupancy.
    threshold = _get_cos_sin_cache_seq_heads_threshold(head_dim)

    if nnz >= threshold:
        # Large workload: use sequential-heads (memory-bound, cache reuse helps)
        kernel_obj = _get_compiled_cos_sin_cache_kernel_seq_heads(
            head_dim, rotary_dim, interleave, dtype_str, idtype_str
        )
    else:
        # Small workload: use head-parallel (compute-bound, parallelism matters)
        kernel_obj = _get_compiled_cos_sin_cache_kernel(
            head_dim, rotary_dim, interleave, dtype_str, idtype_str
        )

    # Launch kernel (pass torch tensors directly)
    kernel_obj(
        q,
        k,
        q_rope_temp,
        k_rope_temp,
        cos_sin_cache,
        pos_ids,
        nnz,
        num_qo_heads,
        num_kv_heads,
    )

    # Copy back if we used temporary tensors to avoid aliasing
    if q_aliased:
        q.copy_(q_rope_temp)
        # Return the original tensor (q) since it now contains the result
        q_rope = q
    if k_aliased:
        k.copy_(k_rope_temp)
        # Return the original tensor (k) since it now contains the result
        k_rope = k

    return q_rope, k_rope


__all__ = [
    # Original API names (match rope_restored.py)
    "apply_rope_cute_dsl",
    "apply_rope_with_indptr_cute_dsl",
    "apply_llama31_rope_with_indptr_cute_dsl",
    "apply_rope_with_cos_sin_cache_cute_dsl",
    "_compute_pos_ids_from_indptr_offsets",
]
