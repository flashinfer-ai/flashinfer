import torch
import math

import flashinfer.ops._kernels as _kernels
from .utils import RotaryMode, TensorLayout
from typing import Optional

_cache_buf = {}


def _get_cache_buf(name: str, bytes: int, device: torch.device):
    key = (name, device)
    buf = _cache_buf.get(key)
    if buf is None:
        buf = torch.empty(bytes, dtype=torch.uint8, device=device)
        _cache_buf[key] = buf
    return buf


def single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rotary_mode: str = "NONE",
    tensor_layout: str = "NHD",
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Single request decode with KV cache.

    Parameters
    ----------
    q : torch.Tensor
        Shape: [num_qo_heads, head_dim]
    k : torch.Tensor
        Shape: [kv_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, kv_len, head_dim] if HND
    v : torch.Tensor
        Shape: [kv_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, kv_len, head_dim] if HND
    rotary_mode : str
        Whether to apply rotary embeddings inside attention kernels, could be
        "NONE" or "LLAMA".
    tensor_layout : str
        The layout of the input k/v tensors, could be either "NHD" or "HND".
    sm_scale : Optional[float]
        The scale of softmax, if not provided, will be set to 1 / sqrt(head_dim)
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.
    """
    tmp = _get_cache_buf("single_decode_with_kv_cache_tmp", 8 * 1024 * 1024, q.device)
    if sm_scale is None:
        head_dim = q.shape[-1]
        sm_scale = 1.0 / math.sqrt(head_dim)
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    return _kernels.single_decode_with_kv_cache(
        q,
        k,
        v,
        tmp,
        getattr(RotaryMode, rotary_mode),
        getattr(TensorLayout, tensor_layout),
        sm_scale,
        rope_scale,
        rope_theta,
    )


def single_prefill_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    rotary_mode: str = "NONE",
    tensor_layout: str = "NHD",
    allow_fp16_qk_reduction: bool = False,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Single request prefill with KV cache kernel.

    Parameters
    ----------
    q : torch.Tensor
        Shape: [qo_len, num_qo_heads, head_dim] if NHD
               [num_qo_heads, qo_len, head_dim] if HND
    k : torch.Tensor
        Shape: [kv_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, kv_len, head_dim] if HND
    v : torch.Tensor
        Shape: [kv_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, kv_len, head_dim] if HND
    causal : bool
        Whether to apply causal mask to the attention matrix.
    rotary_mode : str
        Whether to apply rotary embeddings inside attention kernels, could be
        "NONE" or "LLAMA".
    tensor_layout : str
        The layout of the input k/v tensors, could be either "NHD" or "HND".
    allow_fp16_qk_reduction : bool
        Whether to use f16 for qk reduction (could be significantly faster for GeForce cards, at
        the cost of precision loss).
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.
    """
    tmp = _get_cache_buf("single_prefill_with_kv_cache_tmp", 8 * 1024 * 1024, q.device)
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    return _kernels.single_prefill_with_kv_cache(
        q,
        k,
        v,
        tmp,
        causal,
        getattr(TensorLayout, tensor_layout),
        getattr(RotaryMode, rotary_mode),
        allow_fp16_qk_reduction,
        rope_scale,
        rope_theta,
    )


def single_prefill_with_kv_cache_return_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    rotary_mode: str = "NONE",
    tensor_layout: str = "NHD",
    allow_fp16_qk_reduction: bool = False,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Single request prefill with KV cache kernel, return logsumexp value.

    Parameters
    ----------
    q : torch.Tensor
        Shape: [qo_len, num_qo_heads, head_dim] if NHD
               [num_qo_heads, qo_len, head_dim] if HND
    k : torch.Tensor
        Shape: [kv_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, kv_len, head_dim] if HND
    v : torch.Tensor
        Shape: [kv_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, kv_len, head_dim] if HND
    causal : bool
        Whether to apply causal mask to the attention matrix.
    rotary_mode : str
        Whether to apply rotary embeddings inside attention kernels, could be
        "NONE" or "LLAMA".
    tensor_layout : str
        The layout of the input k/v tensors, could be either "NHD" or "HND".
    allow_fp16_qk_reduction : bool
        Whether to use f16 for qk reduction (could be significantly faster for GeForce cards, at
        the cost of precision loss).
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.

    Returns
    -------
    V : torch.Tensor
        The attention output.
        Shape: [qo_len, num_qo_heads, head_dim] if NHD
               [num_qo_heads, qo_len, head_dim] if HND
    S : torch.Tensor
        The logsumexp value.
        Shape: [qo_len, num_qo_heads]
    """
    tmp = _get_cache_buf(
        "single_prefill_with_kv_cache_return_lse_tmp", 8 * 1024 * 1024, q.device
    )
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    return _kernels.single_prefill_with_kv_cache_return_lse(
        q,
        k,
        v,
        tmp,
        causal,
        getattr(TensorLayout, tensor_layout),
        getattr(RotaryMode, rotary_mode),
        allow_fp16_qk_reduction,
        rope_scale,
        rope_theta,
    )


def merge_state(
    v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
):
    r"""Merge the attention output (V) and the logsumexp value (S) from the two KV-segments.

    Parameters
    ----------
    v_a : torch.Tensor
        The attention output from the KV segment A.
        Shape: [batch_size, num_heads, head_dim]
    s_a : torch.Tensor
        The logsumexp value from the KV segment A.
        Shape: [batch_size, num_heads]
    v_b : torch.Tensor
        The attention output from the KV segment B.
        Shape: [batch_size, num_heads, head_dim]
    s_b : torch.Tensor
        The logsumexp value from the KV segment B.
        Shape: [batch_size, num_heads]

    Returns
    -------
    V : torch.Tensor
        The merged attention output (equivalent to attention with merged KV-segment [A: B]).
        Shape: [batch_size, num_heads, head_dim]
    S : torch.Tensor
        The logsumexp value from the merged KV-segment [A: B].
        Shape: [batch_size, num_heads]
    """
    return _kernels.merge_state(v_a, s_a, v_b, s_b)


def merge_states(v: torch.Tensor, s: torch.Tensor):
    r"""Merge the attention output (V) and the logsumexp value (S) from multiple KV-segments.

    Parameters
    ----------
    v : torch.Tensor
        The attention output from the KV segments.
        Shape: [num_kv_segments, batch_size, num_heads, head_dim]
    s : torch.Tensor
        The logsumexp value from the KV segments.
        Shape: [num_kv_segments, batch_size, num_heads]

    Returns
    -------
    V : torch.Tensor
        The merged attention output.
        Shape: [batch_size, num_heads, head_dim]
    S : torch.Tensor
        The logsumexp value from the merged KV-segments.
        Shape: [batch_size, num_heads]
    """
    return _kernels.merge_states(v, s)


def batch_decode_with_padded_kv_cache(
    q: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    tensor_layout: str = "NHD",
    rotary_mode: str = "NONE",
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Batch decode with padded KV cache.

    Parameters
    ----------
    q : torch.Tensor
        Shape: [batch_size, num_qo_heads, head_dim]
    k_padded : torch.Tensor
        Shape: [batch_size, padded_seq_len, num_kv_heads, head_dim] if NHD
               [batch_size, num_kv_heads, padded_seq_len, head_dim] if HND
    v_padded : torch.Tensor
        Shape: [batch_size, padded_seq_len, num_kv_heads, head_dim] if NHD
               [batch_size, num_kv_heads, padded_seq_len, head_dim] if HND

    Returns
    -------
    V : torch.Tensor
        Shape: [batch_size, num_heads, head_dim]
    """
    if sm_scale is None:
        head_dim = q.shape[-1]
        sm_scale = 1.0 / math.sqrt(head_dim)
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    return _kernels.batch_decode_with_padded_kv_cache(
        q,
        k_padded,
        v_padded,
        getattr(TensorLayout, tensor_layout),
        getattr(RotaryMode, rotary_mode),
        sm_scale,
        rope_scale,
        rope_theta,
    )


def batch_decode_with_padded_kv_cache_return_lse(
    q: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    tensor_layout: str = "NHD",
    rotary_mode: str = "NONE",
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""
    Parameters
    ----------
    q : torch.Tensor
        Shape: [batch_size, num_qo_heads, head_dim]
    k_padded : torch.Tensor
        Shape: [batch_size, padded_seq_len, num_kv_heads, head_dim] if NHD
               [batch_size, num_kv_heads, padded_seq_len, head_dim] if HND
    v_padded : torch.Tensor
        Shape: [batch_size, padded_seq_len, num_kv_heads, head_dim] if NHD
               [batch_size, num_kv_heads, padded_seq_len, head_dim] if HND
    tensor_layout: str
        The layout of the input k_padded/v_padded tensors, could be either
        "NHD" or "HND"
    rotary_mode: str
        Whether to apply rotary embeddings inside attention kernels, could be
        "NONE" or "LLAMA".
    sm_scale: Optional[float]
        The scale of softmax, if not provided, will be set to 1 / sqrt(head_dim)
    rope_scale: Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta: Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.

    Returns
    -------
    V : torch.Tensor
        Shape: [batch_size, num_heads, head_dim]
    S : torch.Tensor
        Shape: [batch_size, num_heads]
    """
    if sm_scale is None:
        head_dim = q.shape[-1]
        sm_scale = 1.0 / math.sqrt(head_dim)
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    return _kernels.batch_decode_with_padded_kv_cache_return_lse(
        q,
        k_padded,
        v_padded,
        getattr(TensorLayout, tensor_layout),
        getattr(RotaryMode, rotary_mode),
        sm_scale,
        rope_scale,
        rope_theta,
    )


def batch_decode_with_shared_prefix_padded_kv_cache(
    q: torch.Tensor,
    k_shared: torch.Tensor,
    v_shared: torch.Tensor,
    k_unique: torch.Tensor,
    v_unique: torch.Tensor,
    tensor_layout: str = "NHD",
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Batch decode with shared prefix padded KV cache.

    Parameters
    ----------
    q : torch.Tensor
        Shape: [batch_size, num_qo_heads, head_dim]
    k_shared : torch.Tensor
        Shape: [shared_prefix_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, shared_prefix_len, head_dim] if HND
    v_shared : torch.Tensor
        Shape: [shared_prefix_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, shared_prefix_len, head_dim] if HND
    k_unique : torch.Tensor
        Shape: [batch_size, unique_len, num_kv_heads, head_dim] if NHD
               [batch_size, num_kv_heads, unique_len, head_dim] if HND
    v_unique : torch.Tensor
        Shape: [batch_size, unique_len, num_kv_heads, head_dim] if NHD
               [batch_size, num_kv_heads, unique_len, head_dim] if HND
    tensor_layout : str
        The layout of the input k/v tensors, could be either "NHD" or "HND".
    sm_scale : Optional[float]
        The scale of softmax, if not provided, will be set to 1 / sqrt(head_dim)
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.

    Returns
    -------
    V : torch.Tensor
        Shape: [batch_size, num_heads, head_dim]
    """
    V_shared, S_shared = single_prefill_with_kv_cache_return_lse(
        q,
        k_shared,
        v_shared,
        causal=False,
        rotary_mode="NONE",
        tensor_layout=tensor_layout,
        allow_fp16_qk_reduction=False,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )
    V_unique, S_unique = batch_decode_with_padded_kv_cache_return_lse(
        q,
        k_unique,
        v_unique,
        tensor_layout=tensor_layout,
        rotary_mode="NONE",
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )
    return merge_state(V_shared, S_shared, V_unique, S_unique)[0]


def batch_prefill_with_paged_kv_cache(
    q: torch.Tensor,
    q_indptr: torch.Tensor,
    kv_data: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    page_size: int,
    casual: bool = True,
    layout: str = "NHD",
    rotary_mode: str = "NONE",
    allow_fp16_qk_reduction: bool = False,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    return _kernels.batch_prefill_with_paged_kv_cache(
        q,
        q_indptr,
        kv_data,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        page_size,
        casual,
        getattr(TensorLayout, layout),
        getattr(RotaryMode, rotary_mode),
        allow_fp16_qk_reduction,
        rope_scale,
        rope_theta,
    )
