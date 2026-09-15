"""
Copyright (c) 2024 by FlashInfer team.

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

import functools
from typing import Optional, Tuple

import torch

from .api_logging import flashinfer_api
from .trace.templates.rope import (
    apply_llama31_rope_inplace_trace,
    apply_llama31_rope_pos_ids_inplace_trace,
    apply_llama31_rope_pos_ids_trace,
    apply_llama31_rope_trace,
    apply_rope_inplace_trace,
    apply_rope_pos_ids_inplace_trace,
    apply_rope_pos_ids_trace,
    apply_rope_trace,
    apply_rope_with_cos_sin_cache_inplace_trace,
    apply_rope_with_cos_sin_cache_trace,
    mla_rope_quantize_fp8_trace,
    qk_rmsnorm_rope_store_hy3_trace_dispatch,
    rope_quantize_fp8_append_paged_kv_cache_trace,
    rope_quantize_fp8_trace,
)
from .jit.rope import gen_rope_module
from .jit.rope_hy3 import gen_qk_norm_rope_hy3_module
from .utils import register_custom_op, register_fake_op


@functools.cache
def get_rope_module():
    return gen_rope_module().build_and_load()


@functools.cache
def get_qk_norm_rope_hy3_module():
    return gen_qk_norm_rope_hy3_module().build_and_load()


@functools.cache
def _is_sm100_device(device_index: int) -> bool:
    """Cache the capability probe so it never enters the decode hot path."""

    return torch.cuda.get_device_capability(device_index) == (10, 0)


@functools.cache
def _qk_norm_rope_hy3_empty_placeholders(
    device_index: int, output_dtype: torch.dtype
) -> Tuple[torch.Tensor, ...]:
    """Return distinct zero-sized tensors for optional FFI arguments.

    Distinct storages avoid aliasing mutable and immutable custom-op arguments,
    while caching keeps placeholder creation out of the per-token hot path.
    """
    device = torch.device("cuda", device_index)
    return (
        torch.empty(0, dtype=torch.float32, device=device),  # q norm
        torch.empty(0, dtype=torch.float32, device=device),  # k norm
        torch.empty(0, dtype=torch.float32, device=device),  # k scale
        torch.empty(0, dtype=torch.float32, device=device),  # v scale
        torch.empty(0, dtype=torch.float32, device=device),  # q scale inverse
        torch.empty(0, dtype=torch.float32, device=device),  # output q scale
        torch.empty(0, dtype=torch.int32, device=device),  # split-k flag
        torch.empty(0, dtype=output_dtype, device=device),  # output k
        torch.empty(0, dtype=output_dtype, device=device),  # output v
    )


@register_custom_op("flashinfer::apply_rope", mutates_args=("q_rope", "k_rope"))
def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    get_rope_module().apply_rope(
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


@register_fake_op("flashinfer::apply_rope")
def _fake_apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    pass


@register_custom_op("flashinfer::apply_llama31_rope", mutates_args=("q_rope", "k_rope"))
def _apply_llama31_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    get_rope_module().apply_llama31_rope(
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
        old_context_len,
    )


@register_fake_op("flashinfer::apply_llama31_rope")
def _fake_apply_llama31_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    pass


@register_custom_op("flashinfer::apply_rope_pos_ids", mutates_args=("q_rope", "k_rope"))
def _apply_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    get_rope_module().apply_rope_pos_ids(
        q,
        k,
        q_rope,
        k_rope,
        pos_ids,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
    )


@register_fake_op("flashinfer::apply_rope_pos_ids")
def _fake_apply_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    pass


@register_custom_op(
    "flashinfer::rope_quantize",
    mutates_args=("q_rope_out", "k_rope_out", "q_nope_out", "k_nope_out"),
)
def _rope_quantize(
    q_rope_in: torch.Tensor,
    k_rope_in: torch.Tensor,
    q_nope_in: torch.Tensor,
    k_nope_in: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    q_rope_out: torch.Tensor,
    k_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope_out: torch.Tensor,
    quant_scale_q: float,
    quant_scale_kv: float,
    interleave: bool,
    enable_pdl: bool,
) -> None:
    r"""Custom operator that routes to the CUDA kernel implementation.

    Converts is_neox parameter to interleave format and dispatches to the underlying
    CUDA kernel via the JIT-compiled module.
    """
    get_rope_module().rope_quantize(
        q_rope_in,
        k_rope_in,
        q_nope_in,
        k_nope_in,
        q_rope_out,
        k_rope_out,
        q_nope_out,
        k_nope_out,
        cos_sin_cache,
        pos_ids,
        quant_scale_q,
        quant_scale_kv,
        interleave,
        enable_pdl,
    )


@register_fake_op("flashinfer::rope_quantize")
def _fake_rope_quantize(
    q_rope_in: torch.Tensor,
    k_rope_in: torch.Tensor,
    q_nope_in: torch.Tensor,
    k_nope_in: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    q_rope_out: torch.Tensor,
    k_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope_out: torch.Tensor,
    quant_scale_q: float,
    quant_scale_kv: float,
    interleave: bool,
    enable_pdl: bool,
) -> None:
    pass


@register_custom_op(
    "flashinfer::rope_quantize_append_paged_kv_cache",
    mutates_args=(
        "q_rope_out",
        "q_nope_out",
        "k_cache",
        "v_cache",
        "ckv_cache",
        "kpe_cache",
    ),
)
def _rope_quantize_fp8_append_paged_kv_cache(
    q_rope_in: torch.Tensor,
    k_rope_in: torch.Tensor,
    q_nope_in: torch.Tensor,
    k_nope_in: torch.Tensor,
    v_in: torch.Tensor,
    q_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    kv_layout_code: int,
    page_size: int,
    quant_scale_q: float,
    quant_scale_kv: float,
    interleave: bool,
    enable_pdl: bool,
) -> None:
    r"""Custom operator that routes to the CUDA kernel implementation.

    Fuses RoPE application, FP8 quantization, and paged KV cache append into a single kernel.

    Converts is_neox parameter to interleave format and dispatches to the underlying
    CUDA kernel via the JIT-compiled module.
    """
    get_rope_module().rope_quantize_append_paged_kv_cache(
        q_rope_in,
        k_rope_in,
        q_nope_in,
        k_nope_in,
        v_in,
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
        interleave,
        enable_pdl,
    )


@register_fake_op("flashinfer::rope_quantize_append_paged_kv_cache")
def _fake_rope_quantize_fp8_append_paged_kv_cache(
    q_rope_in: torch.Tensor,
    k_rope_in: torch.Tensor,
    q_nope_in: torch.Tensor,
    k_nope_in: torch.Tensor,
    v_in: torch.Tensor,
    q_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    kv_layout_code: int,
    page_size: int,
    quant_scale_q: float,
    quant_scale_kv: float,
    interleave: bool,
    enable_pdl: bool,
) -> None:
    pass


@register_custom_op(
    "flashinfer::qk_rmsnorm_rope_append_paged_kv_cache_hy3",
    mutates_args=(
        "output_q",
        "output_q_scale",
        "split_k_flag",
        "output_k",
        "output_v",
        "key_cache",
        "value_cache",
    ),
)
def _qk_rmsnorm_rope_append_paged_kv_cache_hy3(
    packed_qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    sequence_lengths: torch.Tensor,
    q_indptr: torch.Tensor,
    block_table: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q_scale_inverse: torch.Tensor,
    output_q: torch.Tensor,
    output_q_scale: torch.Tensor,
    split_k_flag: torch.Tensor,
    output_k: torch.Tensor,
    output_v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    is_prefill: bool,
    norm_policy: int,
    quant_policy: int,
    max_sequence_length: int,
    fp8_upper_bound: float,
    use_output_k: bool,
    use_output_v: bool,
    enable_sm100_uniform_decode: bool,
) -> None:
    get_qk_norm_rope_hy3_module().qk_rmsnorm_rope_append_paged_kv_cache_hy3(
        packed_qkv,
        cos_sin_cache,
        sequence_lengths,
        q_indptr,
        block_table,
        q_norm_weight,
        k_norm_weight,
        k_scale,
        v_scale,
        q_scale_inverse,
        output_q,
        output_q_scale,
        split_k_flag,
        output_k,
        output_v,
        key_cache,
        value_cache,
        is_prefill,
        norm_policy,
        quant_policy,
        max_sequence_length,
        fp8_upper_bound,
        use_output_k,
        use_output_v,
        enable_sm100_uniform_decode,
    )


@register_fake_op("flashinfer::qk_rmsnorm_rope_append_paged_kv_cache_hy3")
def _fake_qk_rmsnorm_rope_append_paged_kv_cache_hy3(
    packed_qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    sequence_lengths: torch.Tensor,
    q_indptr: torch.Tensor,
    block_table: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q_scale_inverse: torch.Tensor,
    output_q: torch.Tensor,
    output_q_scale: torch.Tensor,
    split_k_flag: torch.Tensor,
    output_k: torch.Tensor,
    output_v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    is_prefill: bool,
    norm_policy: int,
    quant_policy: int,
    max_sequence_length: int,
    fp8_upper_bound: float,
    use_output_k: bool,
    use_output_v: bool,
    enable_sm100_uniform_decode: bool,
) -> None:
    pass


@register_custom_op(
    "flashinfer::apply_rope_pos_ids_cos_sin_cache", mutates_args=("q_rope", "k_rope")
)
def _apply_rope_pos_ids_cos_sin_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    interleave: bool,
) -> None:
    get_rope_module().apply_rope_pos_ids_cos_sin_cache(
        q,
        k,
        q_rope,
        k_rope,
        cos_sin_cache,
        pos_ids,
        interleave,
    )


@register_fake_op("flashinfer::apply_rope_pos_ids_cos_sin_cache")
def _fake_apply_rope_pos_ids_cos_sin_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    interleave: bool,
) -> None:
    pass


@register_custom_op(
    "flashinfer::apply_llama31_rope_pos_ids", mutates_args=("q_rope", "k_rope")
)
def _apply_llama31_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    get_rope_module().apply_llama31_rope_pos_ids(
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
        old_context_len,
    )


@register_fake_op("flashinfer::apply_llama31_rope_pos_ids")
def _fake_apply_llama31_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    pass


@flashinfer_api(trace=apply_rope_inplace_trace)
def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
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

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size = 128
    >>> qkv_len = 1024
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> nnz = batch_size * qkv_len
    >>> qkv_packed = torch.randn(
    >>>    nnz,
    >>>    (num_qo_heads + 2 * num_kv_heads) * head_dim,
    >>>    dtype=torch.float16,
    >>>    device="cuda:0",
    >>> )
    >>> q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    >>> k = qkv_packed[
    ...    :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ... ].reshape(nnz, num_kv_heads, head_dim)
    >>> indptr = torch.tensor(
    ...    [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    >>> )
    >>> offsets = torch.full((batch_size,), 10, dtype=torch.int32, device="cuda:0")
    >>> flashinfer.apply_rope_inplace(q, k, indptr, offsets)

    See Also
    --------
    apply_rope
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_rope(
        q, k, q, k, indptr, offsets, rotary_dim, interleave, rope_scale, rope_theta
    )


@flashinfer_api(trace=apply_rope_pos_ids_inplace_trace)
def apply_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
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

    See Also
    --------
    apply_rope_pos_ids
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_rope_pos_ids(
        q, k, q, k, pos_ids, rotary_dim, interleave, rope_scale, rope_theta
    )


@flashinfer_api(trace=apply_llama31_rope_inplace_trace)
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

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size = 128
    >>> qkv_len = 1024
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> nnz = batch_size * qkv_len
    >>> qkv_packed = torch.randn(
    >>>    nnz,
    >>>    (num_qo_heads + 2 * num_kv_heads) * head_dim,
    >>>    dtype=torch.float16,
    >>>    device="cuda:0",
    >>> )
    >>> q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    >>> k = qkv_packed[
    ...    :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ... ].reshape(nnz, num_kv_heads, head_dim)
    >>> indptr = torch.tensor(
    ...    [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    >>> )
    >>> offsets = torch.full((batch_size,), 10, dtype=torch.int32, device="cuda:0")
    >>> flashinfer.apply_llama31_rope_inplace(q, k, indptr, offsets)

    See Also
    --------
    apply_llama31_rope
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)
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


@flashinfer_api(trace=apply_llama31_rope_pos_ids_inplace_trace)
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

    See Also
    --------
    apply_llama31_rope_pos_ids
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)
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


@flashinfer_api(trace=apply_rope_trace)
def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
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

    Returns
    -------
    q_rope : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k_rope : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads, head_dim)``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size = 128
    >>> qkv_len = 1024
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> nnz = batch_size * qkv_len
    >>> qkv_packed = torch.randn(
    >>>    nnz,
    >>>    (num_qo_heads + 2 * num_kv_heads) * head_dim,
    >>>    dtype=torch.float16,
    >>>    device="cuda:0",
    >>> )
    >>> q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    >>> k = qkv_packed[
    ...    :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ... ].reshape(nnz, num_kv_heads, head_dim)
    >>> indptr = torch.tensor(
    ...    [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    >>> )
    >>> offsets = torch.full((batch_size,), 10, dtype=torch.int32, device="cuda:0")
    >>> q_rope, k_rope = flashinfer.apply_rope(q, k, indptr, offsets)
    >>> q_rope.shape
    torch.Size([131072, 32, 128])
    >>> k_rope.shape
    torch.Size([131072, 32, 128])

    See Also
    --------
    apply_rope_inplace
    """
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    if rotary_dim is None:
        rotary_dim = q.size(-1)
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


@flashinfer_api(trace=apply_rope_pos_ids_trace)
def apply_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
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
    pos_ids : torch.Tensor
        Position indices, shape: ``(batch_size + 1)``.
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
    _apply_rope_pos_ids(
        q, k, q_rope, k_rope, pos_ids, rotary_dim, interleave, rope_scale, rope_theta
    )
    return q_rope, k_rope


@flashinfer_api(trace=apply_llama31_rope_trace)
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

    Returns
    -------
    q_rope : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k_rope : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads, head_dim)``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size = 128
    >>> qkv_len = 1024
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> nnz = batch_size * qkv_len
    >>> qkv_packed = torch.randn(
    >>>    nnz,
    >>>    (num_qo_heads + 2 * num_kv_heads) * head_dim,
    >>>    dtype=torch.float16,
    >>>    device="cuda:0",
    >>> )
    >>> q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    >>> k = qkv_packed[
    ...    :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ... ].reshape(nnz, num_kv_heads, head_dim)
    >>> indptr = torch.tensor(
    ...    [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    >>> )
    >>> offsets = torch.full((batch_size,), 10, dtype=torch.int32, device="cuda:0")
    >>> q_rope, k_rope = flashinfer.apply_llama31_rope(q, k, indptr, offsets)
    >>> q_rope.shape
    torch.Size([131072, 32, 128])
    >>> k_rope.shape
    torch.Size([131072, 32, 128])

    See Also
    --------
    apply_llama31_rope_inplace
    """
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    if rotary_dim is None:
        rotary_dim = q.size(-1)
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


@flashinfer_api(trace=apply_llama31_rope_pos_ids_trace)
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


@flashinfer_api(trace=apply_rope_with_cos_sin_cache_trace)
def apply_rope_with_cos_sin_cache(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
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
    head_size : int
        Per-head feature dimension. ``query`` and ``key`` are reshaped to
        ``(nnz, num_q_heads, head_size)`` and ``(nnz, num_k_heads, head_size)``
        respectively before applying RoPE. Must divide the trailing dimension
        of ``query`` and ``key``.
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


@flashinfer_api(trace=apply_rope_with_cos_sin_cache_inplace_trace)
def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
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
    head_size : int
        Per-head feature dimension. ``query`` and ``key`` are reshaped to
        ``(nnz, num_q_heads, head_size)`` and ``(nnz, num_k_heads, head_size)``
        respectively before applying RoPE. Must divide the trailing dimension
        of ``query`` and ``key``.
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
    Note
    ----
    The rotary dimension is determined by the cosine cache and sine cache.
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    # pass q_rope and k_rope as q and k to perform inplace operation
    _apply_rope_pos_ids_cos_sin_cache(
        q=query.view(query.shape[0], -1, head_size),
        k=key.view(key.shape[0], -1, head_size),
        q_rope=query.view(query.shape[0], -1, head_size),
        k_rope=key.view(key.shape[0], -1, head_size),
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions,
        interleave=(not is_neox),
    )


@flashinfer_api(trace=mla_rope_quantize_fp8_trace)
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
    r"""Apply RoPE and quantize to FP8 for MLA attention.

    Thin wrapper that forwards to :func:`rope_quantize_fp8` with the MLA
    tensor layout: ``k_rope`` / ``k_nope`` are 2-D (no KV-head axis).  All
    other parameters and behavior match :func:`rope_quantize_fp8`; see that
    function for full parameter documentation.

    Parameters
    ----------
    q_rope : torch.Tensor
        Query rotary portion, ``(nnz, num_qo_heads, rope_dim)``, fp16/bf16.
    k_rope : torch.Tensor
        Key rotary portion in MLA layout, ``(nnz, rope_dim)``, fp16/bf16.
    q_nope : torch.Tensor
        Query non-rotary portion, ``(nnz, num_qo_heads, no_rope_dim)``.
    k_nope : torch.Tensor
        Key non-rotary portion in MLA layout, ``(nnz, no_rope_dim)``.
    cos_sin_cache : torch.Tensor
        ``(max_seq_len, rope_dim)`` precomputed cos/sin cache (fp32).
    pos_ids : torch.Tensor
        Per-token position indices, ``(nnz,)``.
    is_neox : bool
        ``True`` for NeoX (split-half) layout, ``False`` for interleaved.
    quantize_dtype : torch.dtype, optional
        Target quantization dtype (``float8_e4m3fn`` or ``float8_e5m2``).
        Inferred from ``*_out`` tensors when ``None``.
    quant_scale_q : float
        Quantization scale applied to queries.
    quant_scale_kv : float
        Quantization scale applied to keys.
    q_rope_out, k_rope_out, q_nope_out, k_nope_out : torch.Tensor, optional
        Pre-allocated output tensors.  Allocated automatically when
        ``None``.
    enable_pdl : bool
        Whether to enable Programmatic Dependent Launch.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(q_rope_out, k_rope_out, q_nope_out, k_nope_out)``.
    """
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


@flashinfer_api(trace=rope_quantize_fp8_trace)
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
    backend: str = "cuda",
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
    backend : str
        Implementation backend. ``"cuda"`` (default) uses the fused CUDA kernel;
        ``"cutile"`` uses the cuda.tile Python kernel.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Quantized tensors: (q_rope_out, k_rope_out, q_nope_out, k_nope_out).
    """
    if backend not in ("cuda", "cutile"):
        raise ValueError(
            f"Unsupported backend for rope_quantize_fp8: {backend!r}; "
            "expected 'cuda' or 'cutile'."
        )
    if backend == "cutile":
        if is_neox:
            # The cuTile kernel implements only the interleaved (GPT-J) rotary
            # layout. is_neox defaults to True, so surface a clear error instead
            # of the kernel's bare AssertionError (which -O would also strip).
            raise NotImplementedError(
                "backend='cutile' supports is_neox=False (interleaved/GPT-J "
                "rotary) only; got is_neox=True."
            )
        if k_rope.ndim != 2:
            # The kernel addresses the key as 2D [tokens, rope_dim] (single
            # shared latent K head, the MLA use case). A 3D GQA/MHA key would
            # fail deep in autotune with an opaque TileTypeError.
            raise NotImplementedError(
                "backend='cutile' rope_quantize_fp8 supports MLA-style 2D key "
                f"tensors (single shared K head) only; got {k_rope.ndim}D key "
                "(GQA/MHA multi-head K is not supported)."
            )
        if cos_sin_cache.dtype != torch.float32:
            raise ValueError("cos_sin_cache should be float32")
        from .quantization.kernels.cutile.rope_quantize_fp8_cutile import (
            rope_quantize_fp8_cutile,
        )

        return rope_quantize_fp8_cutile(
            q_rope,
            k_rope,
            q_nope,
            k_nope,
            cos_sin_cache,
            pos_ids,
            is_neox=is_neox,
            quantize_dtype=quantize_dtype,
            quant_scale_q=quant_scale_q,
            quant_scale_kv=quant_scale_kv,
            q_rope_out=q_rope_out,
            k_rope_out=k_rope_out,
            q_nope_out=q_nope_out,
            k_nope_out=k_nope_out,
        )

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


@flashinfer_api(trace=qk_rmsnorm_rope_store_hy3_trace_dispatch)
def qk_rmsnorm_rope_append_paged_kv_cache_hy3(
    packed_qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    sequence_lengths: torch.Tensor,
    q_indptr: torch.Tensor,
    block_table: torch.Tensor,
    paged_kv_cache: Tuple[torch.Tensor, torch.Tensor],
    is_prefill: bool,
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
    norm_policy: int = 0,
    quant_policy: Optional[int] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    q_scale_inverse: Optional[torch.Tensor] = None,
    max_sequence_length: int = 0,
    fp8_upper_bound: float = 448.0,
    out_q: Optional[torch.Tensor] = None,
    out_q_scale: Optional[torch.Tensor] = None,
    split_k_flag: Optional[torch.Tensor] = None,
    out_k: Optional[torch.Tensor] = None,
    out_v: Optional[torch.Tensor] = None,
    uniform_one_token_decode: bool = False,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    r"""Fuse per-head Q/K RMSNorm, NeoX RoPE, and NHD paged-KV storage.

    This operator consumes a packed BF16 ``[Q, K, V]`` projection and writes
    the transformed query plus K/V cache data in one CUDA launch.  It preserves
    the cache-tail zero-fill side effect required by the originating serving
    pipeline and supports both BF16 and FP8 E4M3 output/cache paths.

    ``norm_policy`` selects the ordering: ``0`` disables normalization, ``1``
    applies RoPE then RMSNorm, and ``2`` applies RMSNorm then RoPE.  The two
    supported local-head configurations are ``8Q/1KV`` and ``64Q/8KV``, both
    with QK/V head dimension 128.  Norm weights are float32 by source contract:
    the originating CUDA entry receives ``const float*`` weights and performs
    the reduction/multiply in float32.  Prefill, decode, and multi-token
    prediction layouts are represented by ``q_indptr`` and
    ``sequence_lengths``.

    For FP8 caches, ``quant_policy=1`` uses a dynamic per-token/per-Q-head
    scale and returns it; ``quant_policy=2`` consumes ``q_scale_inverse``.
    K and V always consume scalar ``k_scale`` and ``v_scale``.  The cache dtype
    determines the output dtype and must be BF16 or ``float8_e4m3fn``.

    Passing ``out_k`` or ``out_v`` preserves the source behavior: that value is
    written to the contiguous output instead of the corresponding cache entry;
    final-page cache-tail clearing still occurs.  All output buffers may be
    preallocated for CUDA-graph-stable addresses.

    ``uniform_one_token_decode`` is a trusted scheduling contract, not a hint.
    Set it only when ``q_indptr`` is exactly ``[0, 1, ..., batch_size]`` and no
    graph-padding/ragged request is present.  On an SM100 device, the previously
    measured narrow FP8 dynamic-Q/norm2/64Q8KV/B>=256 case then maps each row
    directly to its request, reuses RoPE coefficients from registers, and
    clears the final cache-page tail inside the compute CTA.  It defaults to
    ``False``; all other shapes and non-SM100 devices use the source-faithful
    fallback.

    Parameters
    ----------
    packed_qkv : torch.Tensor
        Contiguous BF16 tensor of shape ``[num_rows, (Q + 2 * KV) * 128]``.
    cos_sin_cache : torch.Tensor
        Float32 NeoX cache of shape ``[max_position, 128]``; the first and
        second halves contain cosine and sine respectively.
    sequence_lengths : torch.Tensor
        Int32 final sequence length for each request, shape ``[batch]``.
    q_indptr : torch.Tensor
        Int32 row indptr, shape ``[batch + 1]``.  Absolute token positions are
        derived as ``row + sequence_lengths[request] - q_indptr[request + 1]``.
    block_table : torch.Tensor
        Int32 physical-page table, shape ``[batch, max_pages_per_request]``.
    paged_kv_cache : Tuple[torch.Tensor, torch.Tensor]
        Contiguous NHD key/value caches shaped
        ``[num_pages, page_size, num_kv_heads, 128]``.
    is_prefill : bool
        Whether rows use the prefill position and dynamic-scale layout.
    q_norm_weight : Optional[torch.Tensor]
        Float32 Q RMSNorm weight of shape ``[128]`` when normalization is
        enabled.
    k_norm_weight : Optional[torch.Tensor]
        Float32 K RMSNorm weight of shape ``[128]`` when normalization is
        enabled.
    norm_policy : int
        ``0`` disables normalization, ``1`` runs RoPE then RMSNorm, and ``2``
        runs RMSNorm then RoPE.
    quant_policy : Optional[int]
        ``0`` for BF16, ``1`` for dynamic-Q FP8, or ``2`` for static-Q FP8.
        The default is inferred from the cache dtype.
    k_scale : Optional[torch.Tensor]
        Scalar scale consumed when writing an FP8 K cache.
    v_scale : Optional[torch.Tensor]
        Scalar scale consumed when writing an FP8 V cache.
    q_scale_inverse : Optional[torch.Tensor]
        Scalar inverse Q scale required by static-Q FP8.
    max_sequence_length : int
        Maximum sequence length used to size dynamic-Q prefill scale output.
    fp8_upper_bound : float
        Saturation bound for FP8 conversion.
    out_q : Optional[torch.Tensor]
        Optional preallocated transformed-Q output.
    out_q_scale : Optional[torch.Tensor]
        Optional preallocated dynamic-Q FP8 scale output.
    split_k_flag : Optional[torch.Tensor]
        Optional preallocated per-request/per-KV-head completion flags for the
        FP8 path.
    out_k : Optional[torch.Tensor]
        Optional contiguous K output. When present, K is written here instead
        of to the paged cache.
    out_v : Optional[torch.Tensor]
        Optional contiguous V output. When present, V is written here instead
        of to the paged cache.
    uniform_one_token_decode : bool
        Enables the B200 direct row-to-request schedule. The caller must
        guarantee ``q_indptr == arange(batch_size + 1)``.

    Returns
    -------
    Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]
        ``(out_q, q_scale, split_k_flag, out_k, out_v)``.  Inapplicable outputs
        are returned as ``None``.
    """
    if not packed_qkv.is_cuda:
        raise ValueError("packed_qkv must be a CUDA tensor")
    if len(paged_kv_cache) != 2:
        raise ValueError("paged_kv_cache must be a (key_cache, value_cache) tuple")
    key_cache, value_cache = paged_kv_cache
    if packed_qkv.ndim != 2:
        raise ValueError("packed_qkv must have shape [num_rows, hidden]")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("key_cache and value_cache must be rank-4 NHD tensors")

    num_rows = packed_qkv.shape[0]
    batch_size = sequence_lengths.shape[0]
    num_kv_heads = key_cache.shape[2]
    qk_head_dim = key_cache.shape[3]
    v_head_dim = value_cache.shape[3]
    q_width = packed_qkv.shape[1] - num_kv_heads * (qk_head_dim + v_head_dim)
    if q_width < 0 or qk_head_dim <= 0 or q_width % qk_head_dim != 0:
        raise ValueError(
            "packed_qkv hidden dimension is incompatible with cache shapes"
        )
    num_q_heads = q_width // qk_head_dim
    if (num_q_heads, num_kv_heads) not in ((8, 1), (64, 8)):
        raise ValueError(
            "supported (num_q_heads, num_kv_heads) pairs are (8, 1) and (64, 8)"
        )
    if qk_head_dim != 128 or v_head_dim != 128:
        raise ValueError("qk_head_dim and v_head_dim must both be 128")
    if norm_policy not in (0, 1, 2):
        raise ValueError("norm_policy must be 0, 1, or 2")

    (
        empty_q_norm,
        empty_k_norm,
        empty_k_scale,
        empty_v_scale,
        empty_q_scale_inverse,
        empty_output_q_scale,
        empty_split_k_flag,
        empty_output_k,
        empty_output_v,
    ) = _qk_norm_rope_hy3_empty_placeholders(packed_qkv.get_device(), key_cache.dtype)
    if norm_policy == 0:
        if q_norm_weight is not None or k_norm_weight is not None:
            raise ValueError("norm weights must be None when norm_policy=0")
        q_norm_weight = empty_q_norm
        k_norm_weight = empty_k_norm
    elif q_norm_weight is None or k_norm_weight is None:
        raise ValueError(
            "q_norm_weight and k_norm_weight are required when norm_policy>0"
        )
    elif q_norm_weight.dtype != torch.float32 or k_norm_weight.dtype != torch.float32:
        raise ValueError("q_norm_weight and k_norm_weight must be float32")

    fp8 = key_cache.dtype == torch.float8_e4m3fn
    if quant_policy is None:
        quant_policy = 1 if fp8 else 0
    if fp8:
        if value_cache.dtype != torch.float8_e4m3fn:
            raise ValueError("both FP8 caches must have float8_e4m3fn dtype")
        if quant_policy not in (1, 2):
            raise ValueError("FP8 quant_policy must be 1 (dynamic Q) or 2 (static Q)")
        if k_scale is None or v_scale is None:
            raise ValueError("FP8 cache output requires k_scale and v_scale")
        if quant_policy == 2 and q_scale_inverse is None:
            raise ValueError("static-Q quantization requires q_scale_inverse")
        if quant_policy == 1 and q_scale_inverse is not None:
            raise ValueError("dynamic-Q quantization does not consume q_scale_inverse")
        if quant_policy == 1 and is_prefill and max_sequence_length <= 0:
            raise ValueError("dynamic-Q prefill requires max_sequence_length > 0")
    else:
        if key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16:
            raise ValueError("paged caches must be bfloat16 or float8_e4m3fn")
        if quant_policy != 0:
            raise ValueError("BF16 cache output requires quant_policy=0")
        if k_scale is not None or v_scale is not None or q_scale_inverse is not None:
            raise ValueError("BF16 cache output does not consume quantization scales")
        k_scale = empty_k_scale
        v_scale = empty_v_scale

    if q_scale_inverse is None:
        q_scale_inverse = empty_q_scale_inverse

    output_dtype = key_cache.dtype
    expected_q_shape = (num_rows, num_q_heads, qk_head_dim)
    if out_q is None:
        out_q = torch.empty(
            expected_q_shape, dtype=output_dtype, device=packed_qkv.device
        )
    elif tuple(out_q.shape) != expected_q_shape or out_q.dtype != output_dtype:
        raise ValueError(
            f"out_q must have shape {expected_q_shape} and dtype {output_dtype}"
        )

    q_scale_result: Optional[torch.Tensor]
    if fp8 and quant_policy == 1:
        expected_scale_shape: Tuple[int, ...]
        if is_prefill:
            aligned_length = (max_sequence_length + 127) // 128 * 128
            expected_scale_shape = (batch_size, num_q_heads, aligned_length)
        else:
            expected_scale_shape = (num_rows, num_q_heads)
        if out_q_scale is None:
            out_q_scale = torch.empty(
                expected_scale_shape, dtype=torch.float32, device=packed_qkv.device
            )
        elif (
            tuple(out_q_scale.shape) != expected_scale_shape
            or out_q_scale.dtype != torch.float32
        ):
            raise ValueError(
                f"out_q_scale must have shape {expected_scale_shape} and dtype float32"
            )
        q_scale_result = out_q_scale
    else:
        if out_q_scale is not None:
            raise ValueError("out_q_scale is only valid for dynamic-Q FP8")
        out_q_scale = empty_output_q_scale
        q_scale_result = None

    split_k_result: Optional[torch.Tensor]
    if fp8:
        expected_flag_shape = (batch_size, num_kv_heads)
        if split_k_flag is None:
            split_k_flag = torch.empty(
                expected_flag_shape, dtype=torch.int32, device=packed_qkv.device
            )
        elif (
            tuple(split_k_flag.shape) != expected_flag_shape
            or split_k_flag.dtype != torch.int32
        ):
            raise ValueError(
                f"split_k_flag must have shape {expected_flag_shape} and dtype int32"
            )
        split_k_result = split_k_flag
    else:
        if split_k_flag is not None:
            raise ValueError("split_k_flag is only valid for FP8")
        split_k_flag = empty_split_k_flag
        split_k_result = None

    expected_k_shape = (num_rows, num_kv_heads, qk_head_dim)
    expected_v_shape = (num_rows, num_kv_heads, v_head_dim)
    if out_k is not None and (
        tuple(out_k.shape) != expected_k_shape or out_k.dtype != output_dtype
    ):
        raise ValueError(
            f"out_k must have shape {expected_k_shape} and dtype {output_dtype}"
        )
    if out_v is not None and (
        tuple(out_v.shape) != expected_v_shape or out_v.dtype != output_dtype
    ):
        raise ValueError(
            f"out_v must have shape {expected_v_shape} and dtype {output_dtype}"
        )
    output_k_arg = out_k if out_k is not None else empty_output_k
    output_v_arg = out_v if out_v is not None else empty_output_v

    # Python only establishes the user-provided scheduling contract and device
    # capability here.  The C++ entry repeats the complete dtype/shape/policy
    # gate before selecting the specialized kernel, so duplicating it on every
    # decode step would add host-side launch latency without adding safety.
    enable_sm100_uniform_decode = bool(
        uniform_one_token_decode and _is_sm100_device(packed_qkv.get_device())
    )
    _qk_rmsnorm_rope_append_paged_kv_cache_hy3(
        packed_qkv,
        cos_sin_cache,
        sequence_lengths,
        q_indptr,
        block_table,
        q_norm_weight,
        k_norm_weight,
        k_scale,
        v_scale,
        q_scale_inverse,
        out_q,
        out_q_scale,
        split_k_flag,
        output_k_arg,
        output_v_arg,
        key_cache,
        value_cache,
        is_prefill,
        norm_policy,
        quant_policy,
        max_sequence_length,
        fp8_upper_bound,
        out_k is not None,
        out_v is not None,
        enable_sm100_uniform_decode,
    )
    return out_q, q_scale_result, split_k_result, out_k, out_v


@flashinfer_api(trace=rope_quantize_fp8_append_paged_kv_cache_trace)
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

    Notes
    -----
    - Architecture detection: Automatically distinguishes MLA (2D K tensors) from GQA/MHA (3D K tensors).
    - MLA writes K-RoPE to ``kpe_cache`` and K-noRoPE to ``ckv_cache``; V is not used.
    - GQA/MHA writes full K (RoPE+noRoPE) to ``k_cache`` and V to ``v_cache``.
    - The ``batch_indices`` and ``positions`` tensors are typically obtained from
      ``flashinfer.get_batch_indices_positions()``.
    - Cache tensors must already be allocated in the target FP8 dtype.
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
    from .utils import TensorLayout

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
