"""
Copyright (c) 2024-2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

RoPE Public API Functions
=========================

This module contains the public API functions for RoPE operations:

Standard RoPE:
- apply_rope: Apply RoPE using indptr/offsets
- apply_rope_inplace: Apply RoPE inplace using indptr/offsets
- apply_rope_pos_ids: Apply RoPE using position IDs
- apply_rope_pos_ids_inplace: Apply RoPE inplace using position IDs

Llama 3.1 style RoPE:
- apply_llama31_rope: Apply Llama 3.1 RoPE using indptr/offsets
- apply_llama31_rope_inplace: Apply Llama 3.1 RoPE inplace using indptr/offsets
- apply_llama31_rope_pos_ids: Apply Llama 3.1 RoPE using position IDs
- apply_llama31_rope_pos_ids_inplace: Apply Llama 3.1 RoPE inplace using position IDs

RoPE with cos/sin cache:
- apply_rope_with_cos_sin_cache: Apply RoPE with precomputed cos/sin cache
- apply_rope_with_cos_sin_cache_inplace: Apply RoPE with cos/sin cache inplace

RoPE + Quantize:
- rope_quantize_fp8: Apply RoPE and quantize to FP8
- mla_rope_quantize_fp8: Alias for rope_quantize_fp8
- rope_quantize_fp8_append_paged_kv_cache: RoPE + quantize + append to paged KV cache
"""

from typing import Literal, Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from .utils import _is_cute_dsl_available
from .custom_ops import (
    _apply_rope,
    _apply_llama31_rope,
    _apply_rope_pos_ids,
    _apply_llama31_rope_pos_ids,
    _apply_rope_pos_ids_cos_sin_cache,
    _rope_quantize,
    _rope_quantize_fp8_append_paged_kv_cache,
)


# ============================================================================
# Standard RoPE with indptr/offsets
# ============================================================================


@flashinfer_api
def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
    backend: Literal["cuda", "cute-dsl"] = "cuda",
) -> None:
    r"""Apply rotary embedding to a batch of queries/keys (stored as RaggedTensor) inplace.
    cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)`, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    indptr : torch.Tensor
        Indptr tensor, shape: ``(batch_size + 1)``.
    offsets : torch.Tensor
        The relative position offsets of each query in the batch, shape: ``(batch_size)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``1``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``1e4``.
    backend : Literal["cuda", "cute-dsl"]
        Backend to use for the RoPE computation, default: ``"cuda"``.

        * ``"cuda"``: Use the CUDA C++ backend (default).
        * ``"cute-dsl"``: Use the CuTe-DSL backend (requires CuTe-DSL to be available).

    See Also
    --------
    apply_rope
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)

    if backend == "cute-dsl":
        if not _is_cute_dsl_available():
            raise RuntimeError(
                "CuTe-DSL backend is not available. Please install CuTe-DSL."
            )
        from .kernels import apply_rope_with_indptr_cute_dsl

        # Write directly to input tensors (no copy needed)
        apply_rope_with_indptr_cute_dsl(
            q,
            k,
            indptr,
            offsets,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            q_rope=q,  # Write directly to input
            k_rope=k,  # Write directly to input
        )
        return

    # Default: CUDA C++ backend
    _apply_rope(
        q, k, q, k, indptr, offsets, rotary_dim, interleave, rope_scale, rope_theta
    )


@flashinfer_api
def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
    backend: Literal["cuda", "cute-dsl"] = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply rotary embedding to a batch of queries/keys (stored as RaggedTensor).
    cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)`, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    indptr : torch.Tensor
        Indptr tensor, shape: ``(batch_size + 1)``.
    offsets : torch.Tensor
        The relative position offsets of each query in the batch, shape: ``(batch_size)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``1``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``1e4``.
    backend : Literal["cuda", "cute-dsl"]
        Backend to use for the RoPE computation, default: ``"cuda"``.

        * ``"cuda"``: Use the CUDA C++ backend (default).
        * ``"cute-dsl"``: Use the CuTe-DSL backend (requires CuTe-DSL to be available).

    Returns
    -------
    q_rope : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k_rope : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads, head_dim)``.

    See Also
    --------
    apply_rope_inplace
    """
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    if rotary_dim is None:
        rotary_dim = q.size(-1)

    if backend == "cute-dsl":
        # Note: CuTe-DSL backend has a ~500K token limit per call due to Int32 offset
        # arithmetic (nnz * num_heads * head_dim must fit in Int32). This covers all
        # realistic inference workloads (e.g., batch=256 decode, batch=8 prefill with 32K seq).
        if not _is_cute_dsl_available():
            raise RuntimeError(
                "CuTe-DSL backend is not available. Please install CuTe-DSL."
            )
        from .kernels import apply_rope_with_indptr_cute_dsl

        return apply_rope_with_indptr_cute_dsl(
            q,
            k,
            indptr,
            offsets,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )

    # Default: CUDA C++ backend
    _apply_rope(
        q,
        k,
        q_rope,
        k_rope,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
    )
    return q_rope, k_rope


# ============================================================================
# Standard RoPE with position IDs
# ============================================================================


@flashinfer_api
def apply_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
    backend: Literal["cuda", "cute-dsl"] = "cuda",
) -> None:
    r"""Apply rotary embedding to a batch of queries/keys using position IDs inplace.
    cos/sin values are computed on the fly inside the kernel.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k : torch.Tensor
        Key tensor, shape: ``(nnz, num_k_heads, head_dim)``.
    pos_ids : torch.Tensor
        Position indices, shape: ``(nnz)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``1``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``1e4``.
    backend : Literal["cuda", "cute-dsl"]
        Backend to use for the RoPE computation, default: ``"cuda"``.

        * ``"cuda"``: Use the CUDA C++ backend (default).
        * ``"cute-dsl"``: Use the CuTe-DSL backend (requires CuTe-DSL to be available).

    See Also
    --------
    apply_rope_pos_ids
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)

    if backend == "cute-dsl":
        if not _is_cute_dsl_available():
            raise RuntimeError(
                "CuTe-DSL backend is not available. Please install CuTe-DSL."
            )
        from .kernels import apply_rope_cute_dsl

        # Write directly to input tensors (no copy needed)
        apply_rope_cute_dsl(
            q,
            k,
            pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            q_rope=q,  # Write directly to input
            k_rope=k,  # Write directly to input
        )
        return

    # Default: CUDA C++ backend
    _apply_rope_pos_ids(
        q, k, q, k, pos_ids, rotary_dim, interleave, rope_scale, rope_theta
    )


@flashinfer_api
def apply_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
    backend: Literal["cuda", "cute-dsl"] = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply rotary embedding to a batch of queries/keys using position IDs.
    cos/sin values are computed on the fly inside the kernel.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k : torch.Tensor
        Key tensor, shape: ``(nnz, num_k_heads, head_dim)``.
    pos_ids : torch.Tensor
        Position indices, shape: ``(nnz,)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``1``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``1e4``.
    backend : Literal["cuda", "cute-dsl"]
        Backend to use for the RoPE computation, default: ``"cuda"``.

        * ``"cuda"``: Use the CUDA C++ backend (default).
        * ``"cute-dsl"``: Use the CuTe-DSL backend (requires CuTe-DSL to be available).

    Returns
    -------
    q_rope : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k_rope : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads, head_dim)``.

    See Also
    --------
    apply_rope_inplace
    """
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    if rotary_dim is None:
        rotary_dim = q.size(-1)

    if backend == "cute-dsl":
        # Note: CuTe-DSL backend has a ~500K token limit per call due to Int32 offset
        # arithmetic (nnz * num_heads * head_dim must fit in Int32). This covers all
        # realistic inference workloads (e.g., batch=256 decode, batch=8 prefill with 32K seq).
        if not _is_cute_dsl_available():
            raise RuntimeError(
                "CuTe-DSL backend is not available. Please install CuTe-DSL."
            )
        from .kernels import apply_rope_cute_dsl

        return apply_rope_cute_dsl(
            q,
            k,
            pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )

    # Default: CUDA C++ backend
    _apply_rope_pos_ids(
        q, k, q_rope, k_rope, pos_ids, rotary_dim, interleave, rope_scale, rope_theta
    )
    return q_rope, k_rope


# ============================================================================
# Llama 3.1 style RoPE with indptr/offsets
# ============================================================================


@flashinfer_api
def apply_llama31_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
    backend: Literal["cuda", "cute-dsl"] = "cuda",
) -> None:
    r"""Apply Llama 3.1 style rotary embedding to a batch of queries/keys (stored as
    RaggedTensor) inplace. cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    indptr : torch.Tensor
        Indptr tensor, shape: ``(batch_size + 1)``.
    offsets : torch.Tensor
        The relative position offsets of each query in the batch, shape: ``(batch_size)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``8``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``5e5``.
    low_freq_factor : float
        The low frequency factor used in Llama 3.1 RoPE, default: ``1``.
    high_freq_factor : float
        The high frequency factor used in Llama 3.1 RoPE, default: ``4``.
    old_context_len : int
        The old context length used in Llama 3.1 RoPE, default: ``8192``.
    backend : Literal["cuda", "cute-dsl"]
        Backend to use for the RoPE computation, default: ``"cuda"``.

        * ``"cuda"``: Use the CUDA C++ backend (default).
        * ``"cute-dsl"``: Use the CuTe-DSL backend (requires CuTe-DSL to be available).

    See Also
    --------
    apply_llama31_rope
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)

    if backend == "cute-dsl":
        if not _is_cute_dsl_available():
            raise RuntimeError(
                "CuTe-DSL backend is not available. Please install CuTe-DSL."
            )
        from .kernels import apply_llama31_rope_with_indptr_cute_dsl

        # Write directly to input tensors (no copy needed)
        apply_llama31_rope_with_indptr_cute_dsl(
            q,
            k,
            indptr,
            offsets,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            old_context_len=old_context_len,
            q_rope=q,  # Write directly to input
            k_rope=k,  # Write directly to input
        )
        return

    # Default: CUDA C++ backend
    _apply_llama31_rope(
        q,
        k,
        q,
        k,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
    )


@flashinfer_api
def apply_llama31_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
    backend: Literal["cuda", "cute-dsl"] = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply Llama 3.1 style rotary embedding to a batch of queries/keys (stored as
    RaggedTensor). cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    indptr : torch.Tensor
        Indptr tensor, shape: ``(batch_size + 1)``.
    offsets : torch.Tensor
        The relative position offsets of each query in the batch, shape: ``(batch_size)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``8``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``5e5``.
    low_freq_factor : float
        The low frequency factor used in Llama 3.1 RoPE, default: ``1``.
    high_freq_factor : float
        The high frequency factor used in Llama 3.1 RoPE, default: ``4``.
    old_context_len : int
        The old context length used in Llama 3.1 RoPE, default: ``8192``.
    backend : Literal["cuda", "cute-dsl"]
        Backend to use for the RoPE computation, default: ``"cuda"``.

        * ``"cuda"``: Use the CUDA C++ backend (default).
        * ``"cute-dsl"``: Use the CuTe-DSL backend (requires CuTe-DSL to be available).

    Returns
    -------
    q_rope : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k_rope : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads, head_dim)``.

    See Also
    --------
    apply_llama31_rope_inplace
    """
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    if rotary_dim is None:
        rotary_dim = q.size(-1)

    if backend == "cute-dsl":
        # Note: CuTe-DSL backend has a ~500K token limit per call due to Int32 offset
        # arithmetic (nnz * num_heads * head_dim must fit in Int32). This covers all
        # realistic inference workloads (e.g., batch=256 decode, batch=8 prefill with 32K seq).
        if not _is_cute_dsl_available():
            raise RuntimeError(
                "CuTe-DSL backend is not available. Please install CuTe-DSL."
            )
        from .kernels import apply_llama31_rope_with_indptr_cute_dsl

        return apply_llama31_rope_with_indptr_cute_dsl(
            q,
            k,
            indptr,
            offsets,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            old_context_len=old_context_len,
        )

    # Default: CUDA C++ backend
    _apply_llama31_rope(
        q,
        k,
        q_rope,
        k_rope,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
    )
    return q_rope, k_rope


# ============================================================================
# Llama 3.1 style RoPE with position IDs
# ============================================================================


@flashinfer_api
def apply_llama31_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
    backend: Literal["cuda", "cute-dsl"] = "cuda",
) -> None:
    r"""Apply Llama 3.1 style rotary embedding to queries/keys using position IDs inplace.
    cos/sin values are computed on the fly inside the kernel.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k : torch.Tensor
        Key tensor, shape: ``(nnz, num_k_heads, head_dim)``.
    pos_ids : torch.Tensor
        Position indices, shape: ``(nnz)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``8``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``5e5``.
    low_freq_factor : float
        The low frequency factor used in Llama 3.1 RoPE, default: ``1``.
    high_freq_factor : float
        The high frequency factor used in Llama 3.1 RoPE, default: ``4``.
    old_context_len : int
        The old context length used in Llama 3.1 RoPE, default: ``8192``.
    backend : Literal["cuda", "cute-dsl"]
        Backend to use for the RoPE computation, default: ``"cuda"``.

        * ``"cuda"``: Use the CUDA C++ backend (default).
        * ``"cute-dsl"``: Use the CuTe-DSL backend (requires CuTe-DSL to be available).

    See Also
    --------
    apply_llama31_rope_pos_ids
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)

    if backend == "cute-dsl":
        if not _is_cute_dsl_available():
            raise RuntimeError(
                "CuTe-DSL backend is not available. Please install CuTe-DSL."
            )
        from .kernels import apply_rope_cute_dsl

        # Write directly to input tensors (no copy needed)
        apply_rope_cute_dsl(
            q,
            k,
            pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            old_context_len=old_context_len,
            q_rope=q,  # Write directly to input
            k_rope=k,  # Write directly to input
        )
        return

    # Default: CUDA C++ backend
    _apply_llama31_rope_pos_ids(
        q,
        k,
        q,
        k,
        pos_ids,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
    )


@flashinfer_api
def apply_llama31_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
    backend: Literal["cuda", "cute-dsl"] = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply Llama 3.1 style rotary embedding to queries/keys using position IDs.
    cos/sin values are computed on the fly inside the kernel.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k : torch.Tensor
        Key tensor, shape: ``(nnz, num_k_heads, head_dim)``.
    pos_ids : torch.Tensor
        Position indices, shape: ``(nnz)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``
    rope_scale : float
        The scaling factor used in the rope embedding, default: ``8``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``5e5``.
    low_freq_factor : float
        The low frequency factor used in Llama 3.1 RoPE, default: ``1``.
    high_freq_factor : float
        The high frequency factor used in Llama 3.1 RoPE, default: ``4``.
    old_context_len : int
        The old context length used in Llama 3.1 RoPE, default: ``8192``.
    backend : Literal["cuda", "cute-dsl"]
        Backend to use for the RoPE computation, default: ``"cuda"``.

        * ``"cuda"``: Use the CUDA C++ backend (default).
        * ``"cute-dsl"``: Use the CuTe-DSL backend (requires CuTe-DSL to be available).

    Returns
    -------
    q_rope : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k_rope : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads, head_dim)``.

    See Also
    --------
    apply_llama31_rope_pos_ids_inplace
    """
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    if rotary_dim is None:
        rotary_dim = q.size(-1)

    if backend == "cute-dsl":
        # Note: CuTe-DSL backend has a ~500K token limit per call due to Int32 offset
        # arithmetic (nnz * num_heads * head_dim must fit in Int32). This covers all
        # realistic inference workloads (e.g., batch=256 decode, batch=8 prefill with 32K seq).
        if not _is_cute_dsl_available():
            raise RuntimeError(
                "CuTe-DSL backend is not available. Please install CuTe-DSL."
            )
        from .kernels import apply_rope_cute_dsl

        return apply_rope_cute_dsl(
            q,
            k,
            pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            old_context_len=old_context_len,
        )

    # Default: CUDA C++ backend
    _apply_llama31_rope_pos_ids(
        q,
        k,
        q_rope,
        k_rope,
        pos_ids,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
    )
    return q_rope, k_rope


# ============================================================================
# RoPE with cos/sin cache
# ============================================================================


@flashinfer_api
def apply_rope_with_cos_sin_cache(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    backend: Literal["cuda", "cute-dsl"] = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Apply rotary embedding to keys and queries with precomputed cos/sin values.
    This is designed to be compatible with the SGL/vLLM implementation.

    Parameters
    ----------
    positions : torch.Tensor
        Position indices, shape: ``(nnz)``.
    query : torch.Tensor
        Query tensor, shape: ``(nnz, num_q_heads * head_size)``.
    key : torch.Tensor
        Key tensor, shape: ``(nnz, num_k_heads * head_size)``.
    cos_sin_cache : torch.Tensor
        Cosine and Sine cache tensor, shape: ``(max_seq_len, rotary_dim)``.
        Cosine is the first half and Sine is the second half on rotary_dim.
    is_neox : bool
        Whether to use Neox style RoPE, default: ``True``.

        * If ``True``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rorate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

        * If ``False``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

    backend : Literal["cuda", "cute-dsl"]
        Backend to use for the RoPE computation, default: ``"cuda"``.

        * ``"cuda"``: Use the CUDA C++ backend (default).
        * ``"cute-dsl"``: Use the CuTe-DSL backend (requires CuTe-DSL to be available).

    Returns
    -------
    query_out : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads * head_size)``.
    key_out : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads * head_size)``.

    Note
    ----
    The rotary dimension is determined by the cosine cache and sine cache.
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    if backend == "cute-dsl":
        if not _is_cute_dsl_available():
            raise RuntimeError(
                "CuTe-DSL backend is not available. Please install CuTe-DSL."
            )
        from .kernels import apply_rope_with_cos_sin_cache_cute_dsl

        # CuTe-DSL expects 3D tensors: (nnz, num_heads, head_dim)
        # Input query is (nnz, num_qo_heads * head_dim), reshape to (nnz, num_qo_heads, head_dim)
        query_out, key_out = apply_rope_with_cos_sin_cache_cute_dsl(
            q=query.view(query.shape[0], -1, head_size),
            k=key.view(key.shape[0], -1, head_size),
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions.view(-1),  # Flatten positions to 1D
            interleave=(not is_neox),
        )
        return query_out.view(query.shape), key_out.view(key.shape)

    # Default: CUDA C++ backend
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)

    _apply_rope_pos_ids_cos_sin_cache(
        q=query.view(query.shape[0], -1, head_size),
        k=key.view(key.shape[0], -1, head_size),
        q_rope=query_out.view(query_out.shape[0], -1, head_size),
        k_rope=key_out.view(key_out.shape[0], -1, head_size),
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions,
        interleave=(not is_neox),
    )

    return query_out, key_out


@flashinfer_api
def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    backend: Literal["cuda", "cute-dsl"] = "cuda",
) -> None:
    r"""
    Apply rotary embedding to keys and queries with precomputed cos/sin values.
    This is designed to be compatible with the SGL/vLLM implementation.
    The result is inplace applied to the input tensors.

    Parameters
    ----------
    positions : torch.Tensor
        Position indices, shape: ``(nnz)``.
    query : torch.Tensor
        Query tensor, shape: ``(nnz, num_q_heads * head_size)``.
    key : torch.Tensor
        Key tensor, shape: ``(nnz, num_k_heads * head_size)``.
    cos_sin_cache : torch.Tensor
        Cosine and Sine cache tensor, shape: ``(max_seq_len, rotary_dim)``.
        Cosine is the first half and Sine is the second half on rotary_dim.
    is_neox : bool
        Whether to use Neox style RoPE, default: ``True``.

        * If ``True``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rorate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

        * If ``False``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

    backend : Literal["cuda", "cute-dsl"]
        Backend to use for the RoPE computation, default: ``"cuda"``.

        * ``"cuda"``: Use the CUDA C++ backend (default).
        * ``"cute-dsl"``: Use the CuTe-DSL backend (requires CuTe-DSL to be available).

    Note
    ----
    The rotary dimension is determined by the cosine cache and sine cache.
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    if backend == "cute-dsl":
        if not _is_cute_dsl_available():
            raise RuntimeError(
                "CuTe-DSL backend is not available. Please install CuTe-DSL."
            )
        from .kernels import apply_rope_with_cos_sin_cache_cute_dsl

        # CuTe-DSL expects 3D tensors: (nnz, num_heads, head_dim)
        # Input query is (nnz, num_qo_heads * head_dim), reshape to (nnz, num_qo_heads, head_dim)
        q_view = query.view(query.shape[0], -1, head_size)
        k_view = key.view(key.shape[0], -1, head_size)

        # Write directly to input tensors (no copy needed)
        apply_rope_with_cos_sin_cache_cute_dsl(
            q=q_view,
            k=k_view,
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions.view(-1),  # Flatten positions to 1D
            interleave=(not is_neox),
            q_rope=q_view,  # Write directly to input
            k_rope=k_view,  # Write directly to input
        )
        return

    # Default: CUDA C++ backend - pass q_rope and k_rope as q and k for inplace
    _apply_rope_pos_ids_cos_sin_cache(
        q=query.view(query.shape[0], -1, head_size),
        k=key.view(key.shape[0], -1, head_size),
        q_rope=query.view(query.shape[0], -1, head_size),
        k_rope=key.view(key.shape[0], -1, head_size),
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions,
        interleave=(not is_neox),
    )


# ============================================================================
# RoPE + Quantize APIs
# ============================================================================


@flashinfer_api
def mla_rope_quantize_fp8(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    is_neox: bool = True,
    quantize_dtype: Optional[torch.dtype] = None,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    q_rope_out: Optional[torch.Tensor] = None,
    k_rope_out: Optional[torch.Tensor] = None,
    q_nope_out: Optional[torch.Tensor] = None,
    k_nope_out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alias for rope_quantize_fp8. See rope_quantize_fp8 for documentation."""
    return rope_quantize_fp8(
        q_rope,
        k_rope,
        q_nope,
        k_nope,
        cos_sin_cache,
        pos_ids,
        is_neox,
        quantize_dtype,
        quant_scale_q,
        quant_scale_kv,
        q_rope_out,
        k_rope_out,
        q_nope_out,
        k_nope_out,
        enable_pdl,
    )


@flashinfer_api
def rope_quantize_fp8(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: Optional[torch.Tensor],
    k_nope: Optional[torch.Tensor],
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    is_neox: bool = True,
    quantize_dtype: Optional[torch.dtype] = None,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    q_rope_out: Optional[torch.Tensor] = None,
    k_rope_out: Optional[torch.Tensor] = None,
    q_nope_out: Optional[torch.Tensor] = None,
    k_nope_out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Apply RoPE (Rotary Positional Embeddings) and quantize to FP8 format.

    This function takes pre-split query/key tensors (rotary and non-rotary dimensions separated),
    applies RoPE to the rotary dimension tensors, and quantizes both rotary and non-rotary
    tensors to FP8 format. Supports MLA, GQA, and MHA architectures.

    Parameters
    ----------
    q_rope : torch.Tensor
        Query tensor (rotary dimensions), shape: ``(nnz, num_qo_heads, rope_dim)``.
        Must be float16 or bfloat16.
    k_rope : torch.Tensor
        Key tensor (rotary dimensions). For GQA/MHA: ``(nnz, num_kv_heads, rope_dim)``.
        For MLA: ``(nnz, rope_dim)``. Must be float16 or bfloat16.
    q_nope : Optional[torch.Tensor]
        Query tensor (non-rotary dimensions), shape: ``(nnz, num_qo_heads, no_rope_dim)``.
        If ``None``, treated as zero-dim: a size-0 tensor will be created internally.
    k_nope : Optional[torch.Tensor]
        Key tensor (non-rotary dimensions). For GQA/MHA: ``(nnz, num_kv_heads, no_rope_dim)``.
        For MLA: ``(nnz, no_rope_dim)``. If ``None``, treated as zero-dim and created internally.
    cos_sin_cache : torch.Tensor
        Precomputed cosine and sine values, shape: ``(max_seq_len, rope_dim)``.
        First half contains cosine values, second half contains sine values. Must be float32.
    pos_ids : torch.Tensor
        Position indices for each token, shape: ``(nnz,)``.
    is_neox : bool
        RoPE layout style. If ``True`` (default), use non-interleaved layout (first/second half).
        If ``False``, use interleaved layout (even/odd dimensions).
    quantize_dtype : Optional[torch.dtype]
        Target quantization dtype. If ``None``, inferred from output tensors or defaults to
        ``torch.float8_e4m3fn``. Must be ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.
    quant_scale_q : float
        Quantization scaling factor for query tensors, default: ``1.0``.
    quant_scale_kv : float
        Quantization scaling factor for key tensors, default: ``1.0``.
    q_rope_out : Optional[torch.Tensor]
        Pre-allocated output tensor for quantized query (rotary). If ``None``, allocated automatically.
    k_rope_out : Optional[torch.Tensor]
        Pre-allocated output tensor for quantized key (rotary). If ``None``, allocated automatically.
    q_nope_out : Optional[torch.Tensor]
        Pre-allocated output tensor for quantized query (non-rotary). If ``None``, allocated automatically.
    k_nope_out : Optional[torch.Tensor]
        Pre-allocated output tensor for quantized key (non-rotary). If ``None``, allocated automatically.
    enable_pdl : bool
        Whether to enable PDL (Programmatic Dependent Launch). Default: ``False``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Quantized tensors: (q_rope_out, k_rope_out, q_nope_out, k_nope_out).
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    # Allow None for nope tensors and normalize to size-0 tensors with correct shapes
    nnz = q_rope.shape[0]
    num_qo_heads = q_rope.shape[1]
    is_mla = k_rope.ndim == 2
    num_kv_heads = 1 if is_mla else k_rope.shape[1]
    if q_nope is None:
        q_nope = torch.empty(
            nnz, num_qo_heads, 0, dtype=q_rope.dtype, device=q_rope.device
        )
    if k_nope is None:
        if is_mla:
            k_nope = torch.empty(nnz, 0, dtype=k_rope.dtype, device=k_rope.device)
        else:
            k_nope = torch.empty(
                nnz, num_kv_heads, 0, dtype=k_rope.dtype, device=k_rope.device
            )

    # Infer quantize_dtype from output tensors or default to float8_e4m3fn
    if quantize_dtype is None:
        for out in (q_rope_out, k_rope_out, q_nope_out, k_nope_out):
            if out is not None:
                quantize_dtype = out.dtype
                break
        else:
            quantize_dtype = torch.float8_e4m3fn

    # Allocate output tensors if not provided
    q_rope_out = (
        q_rope_out
        if q_rope_out is not None
        else torch.empty_like(q_rope, dtype=quantize_dtype)
    )
    k_rope_out = (
        k_rope_out
        if k_rope_out is not None
        else torch.empty_like(k_rope, dtype=quantize_dtype)
    )
    q_nope_out = (
        q_nope_out
        if q_nope_out is not None
        else torch.empty_like(q_nope, dtype=quantize_dtype)
    )
    k_nope_out = (
        k_nope_out
        if k_nope_out is not None
        else torch.empty_like(k_nope, dtype=quantize_dtype)
    )

    _rope_quantize(
        q_rope,
        k_rope,
        q_nope,
        k_nope,
        cos_sin_cache,
        pos_ids,
        q_rope_out,
        k_rope_out,
        q_nope_out,
        k_nope_out,
        quant_scale_q,
        quant_scale_kv,
        not is_neox,  # interleave
        enable_pdl,
    )

    return q_rope_out, k_rope_out, q_nope_out, k_nope_out


@flashinfer_api
def rope_quantize_fp8_append_paged_kv_cache(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: Optional[torch.Tensor],
    k_nope: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    paged_kv_cache: Tuple[torch.Tensor, torch.Tensor],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool = True,
    quantize_dtype: Optional[torch.dtype] = None,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    page_size: int = 16,
    kv_layout: str = "NHD",
    q_rope_out: Optional[torch.Tensor] = None,
    q_nope_out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply RoPE (Rotary Positional Embeddings), quantize to FP8, and append K/V to paged cache.

    This fused function applies RoPE to query/key (Q/K) rotary dimension tensors, quantizes all Q/K tensors
    (and V for GQA/MHA) to FP8 format, and directly appends the quantized K/V to a paged KV cache.
    It returns quantized Q tensors for use in attention computation. Supports MLA, GQA, and MHA
    architectures with automatic detection based on input tensor shapes.

    Parameters
    ----------
    q_rope : torch.Tensor
        Query tensor (rotary dimensions), shape: ``(nnz, num_qo_heads, rope_dim)``.
        Must be float16 or bfloat16.
    k_rope : torch.Tensor
        Key tensor (rotary dimensions). For GQA/MHA: ``(nnz, num_kv_heads, rope_dim)``.
        For MLA: ``(nnz, rope_dim)``. Must be float16 or bfloat16.
    q_nope : torch.Tensor
        Query tensor (non-rotary dimensions), shape: ``(nnz, num_qo_heads, no_rope_dim)``.
        Must be float16 or bfloat16.
    k_nope : torch.Tensor
        Key tensor (non-rotary dimensions). For GQA/MHA: ``(nnz, num_kv_heads, no_rope_dim)``.
        For MLA: ``(nnz, no_rope_dim)``. Must be float16 or bfloat16.
    v : Optional[torch.Tensor]
        Value tensor for GQA/MHA: ``(nnz, num_kv_heads, head_dim)``. Must be float16 or bfloat16.
        For MLA: pass ``None`` (MLA does not use separate V; K non-RoPE acts as compressed KV).
    cos_sin_cache : torch.Tensor
        Precomputed cosine and sine values, shape: ``(max_seq_len, rope_dim)``.
        First half contains cosine values, second half contains sine values. Must be float32.
    pos_ids : torch.Tensor
        Position indices for each token, shape: ``(nnz,)``.
    paged_kv_cache : Tuple[torch.Tensor, torch.Tensor]
        For MLA: ``(ckv_cache, kpe_cache)`` where:
            - ckv_cache: ``(max_pages, page_size, no_rope_dim)`` in FP8
            - kpe_cache: ``(max_pages, page_size, rope_dim)`` in FP8
        For GQA/MHA: ``(k_cache, v_cache)`` where:
            - k_cache: ``(max_pages, page_size, num_kv_heads, head_dim)`` or
              ``(max_pages, num_kv_heads, page_size, head_dim)`` depending on layout, in FP8
            - v_cache: same shape as k_cache, in FP8
    kv_indices : torch.Tensor
        Page indices mapping, shape: ``(total_pages,)``. Typically ``torch.arange(total_pages)``.
    kv_indptr : torch.Tensor
        Page indptr array for each request, shape: ``(batch_size + 1,)``.
        ``kv_indptr[i]`` is the starting page index for request ``i``.
    batch_indices : torch.Tensor
        Batch index for each token, shape: ``(nnz,)``. Maps each token to its request.
    positions : torch.Tensor
        Position within each request's sequence for each token, shape: ``(nnz,)``.
    is_neox : bool
        RoPE layout style. If ``True`` (default), use non-interleaved layout (first/second half).
        If ``False``, use interleaved layout (even/odd dimensions).
    quantize_dtype : Optional[torch.dtype]
        Target quantization dtype. If ``None``, inferred from output tensors or defaults to
        ``torch.float8_e4m3fn``. Must be ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.
    quant_scale_q : float
        Quantization scaling factor for query tensors, default: ``1.0``.
    quant_scale_kv : float
        Quantization scaling factor for key/value tensors, default: ``1.0``.
    page_size : int
        Number of entries per page in the paged cache, default: ``16``.
    kv_layout : str
        Cache memory layout for GQA/MHA. Options: ``"NHD"`` (page, seq, head, dim) or
        ``"HND"`` (page, head, seq, dim). Default: ``"NHD"``. Ignored for MLA.
    q_rope_out : Optional[torch.Tensor]
        Pre-allocated output tensor for quantized query (rotary). If ``None``, allocated automatically.
    q_nope_out : Optional[torch.Tensor]
        Pre-allocated output tensor for quantized query (non-rotary). If ``None``, allocated automatically.
    enable_pdl : bool
        Whether to enable PDL (Programmatic Dependent Launch). Default: ``False``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Quantized query tensors: (q_rope_out, q_nope_out).
        K/V are written directly to the paged cache and not returned.
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    # Detect architecture
    is_mla = k_rope.ndim == 2

    # Allow None for nope tensors and normalize to size-0 tensors with correct shapes
    nnz = q_rope.shape[0]
    num_qo_heads = q_rope.shape[1]
    if q_nope is None:
        q_nope = torch.empty(
            nnz, num_qo_heads, 0, dtype=q_rope.dtype, device=q_rope.device
        )
    if k_nope is None:
        if is_mla:
            k_nope = torch.empty(nnz, 0, dtype=k_rope.dtype, device=k_rope.device)
        else:
            num_kv_heads = k_rope.shape[1]
            k_nope = torch.empty(
                nnz, num_kv_heads, 0, dtype=k_rope.dtype, device=k_rope.device
            )

    # Infer quantize_dtype from output tensors or default
    if quantize_dtype is None:
        if q_rope_out is not None:
            quantize_dtype = q_rope_out.dtype
        elif q_nope_out is not None:
            quantize_dtype = q_nope_out.dtype
        else:
            quantize_dtype = torch.float8_e4m3fn

    # Allocate Q output tensors if not provided
    if q_rope_out is None:
        q_rope_out = torch.empty_like(q_rope, dtype=quantize_dtype)
    if q_nope_out is None:
        q_nope_out = torch.empty_like(q_nope, dtype=quantize_dtype)

    # Handle MLA normalization and V (create empty dummy tensor, not used)
    if is_mla:
        # Normalize MLA K tensors to 3D (nnz, 1, dim) so C++ binding can always assume 3D
        if k_rope.ndim == 2:
            k_rope = k_rope.unsqueeze(1)
        if k_nope.ndim == 2:
            k_nope = k_nope.unsqueeze(1)
        if v is None:
            v = torch.empty(0, dtype=q_rope.dtype, device=q_rope.device)
        else:
            raise ValueError("MLA should not have V input (pass None)")

    # Unpack and validate cache tensors
    if len(paged_kv_cache) != 2:
        raise ValueError("paged_kv_cache must be a tuple of 2 tensors")

    cache_0, cache_1 = paged_kv_cache

    if is_mla:
        # MLA: Expect (ckv_cache, kpe_cache)
        ckv_cache = cache_0
        kpe_cache = cache_1
        if ckv_cache.dtype != quantize_dtype or kpe_cache.dtype != quantize_dtype:
            raise ValueError(
                f"MLA cache dtype mismatch: expected {quantize_dtype}, "
                f"got ckv={ckv_cache.dtype}, kpe={kpe_cache.dtype}"
            )
        if ckv_cache.ndim != 3 or kpe_cache.ndim != 3:
            raise ValueError(
                f"MLA cache must be 3D: (max_pages, page_size, dim), "
                f"got ckv={ckv_cache.ndim}D, kpe={kpe_cache.ndim}D"
            )
        # Create dummy tensors for GQA/MHA cache (not used)
        k_cache = torch.empty(0, dtype=quantize_dtype, device=q_rope.device)
        v_cache = torch.empty(0, dtype=quantize_dtype, device=q_rope.device)
    else:
        # GQA/MHA: Expect (k_cache, v_cache)
        k_cache = cache_0
        v_cache = cache_1
        # Validate V input is provided for GQA/MHA
        if v is None:
            raise ValueError(
                "GQA/MHA expects a V tensor, but got None. "
                "Only MLA uses None for V (compressed KV representation)."
            )
        if k_cache.dtype != quantize_dtype or v_cache.dtype != quantize_dtype:
            raise ValueError(
                f"GQA/MHA cache dtype mismatch: expected {quantize_dtype}, "
                f"got k={k_cache.dtype}, v={v_cache.dtype}"
            )
        if k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError(
                f"GQA/MHA cache must be 4D, got k={k_cache.ndim}D, v={v_cache.ndim}D"
            )
        # Create dummy tensors for MLA cache (not used)
        ckv_cache = torch.empty(0, dtype=quantize_dtype, device=q_rope.device)
        kpe_cache = torch.empty(0, dtype=quantize_dtype, device=q_rope.device)

    # Import TensorLayout enum
    from ..utils import TensorLayout

    kv_layout_code = TensorLayout[kv_layout].value

    batch_indices = batch_indices.int()
    positions = positions.int()
    kv_indices = kv_indices.int()
    kv_indptr = kv_indptr.int()

    # Call custom op
    _rope_quantize_fp8_append_paged_kv_cache(
        q_rope,
        k_rope,
        q_nope,
        k_nope,
        v,
        q_rope_out,
        q_nope_out,
        cos_sin_cache,
        pos_ids,
        k_cache,
        v_cache,
        ckv_cache,
        kpe_cache,
        kv_indices,
        kv_indptr,
        batch_indices,
        positions,
        kv_layout_code,
        page_size,
        quant_scale_q,
        quant_scale_kv,
        not is_neox,  # interleave
        enable_pdl,
    )

    return q_rope_out, q_nope_out
