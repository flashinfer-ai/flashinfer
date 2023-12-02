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
    return _kernels.merge_state(v_a, s_a, v_b, s_b)


def merge_states(v: torch.Tensor, s: torch.Tensor):
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
    r"""
    Parameters
    ----------
    q : torch.Tensor
        Shape: [batch_size, num_qo_heads, head_dim]
    k_padded : torch.Tensor
        Shape: [batch_size, padded_seq_len, num_kv_heads, head_dim]
    v_padded : torch.Tensor
        Shape: [batch_size, padded_seq_len, num_kv_heads, head_dim]

    Returns
    -------
    torch.Tensor
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
        Shape: [batch_size, padded_seq_len, num_kv_heads, head_dim] if NHD else [batch_size, num_kv_heads, padded_seq_len, head_dim]
    v_padded : torch.Tensor
        Shape: [batch_size, padded_seq_len, num_kv_heads, head_dim] if NHD else [batch_size, num_kv_heads, padded_seq_len, head_dim]

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        o shape: [batch_size, num_heads, head_dim]
        lse shape: [batch_size, num_heads]
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
    o_shared, lse_shared = single_prefill_with_kv_cache_return_lse(
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
    o_unique, lse_unique = batch_decode_with_padded_kv_cache_return_lse(
        q,
        k_unique,
        v_unique,
        tensor_layout=tensor_layout,
        rotary_mode="NONE",
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )
    o_merged = merge_state(o_shared, lse_shared, o_unique, lse_unique)
    return o_merged
