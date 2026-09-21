"""Backend for fused QK RMSNorm, RoPE, quantization, and paged append."""

import functools
import math
from typing import Optional, Tuple

import torch

from .jit import load_fused_qk_rope_append_module


@functools.cache
def _get_hpc_rope_module():
    return load_fused_qk_rope_append_module()


def _q_heads_from_packed_qkv(
    qkv: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor
) -> int:
    kv_heads = key_cache.shape[2]
    qk_dim = key_cache.shape[3]
    v_dim = value_cache.shape[3]
    q_width = qkv.shape[1] - kv_heads * (qk_dim + v_dim)
    if q_width <= 0 or q_width % qk_dim:
        raise ValueError("qkv width is inconsistent with the KV cache shape")
    return q_width // qk_dim


def fused_qk_norm_rope_append_paged_kv_cache(
    qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    q_indptr: torch.Tensor,
    page_indices: torch.Tensor,
    paged_kv_cache: Tuple[torch.Tensor, torch.Tensor],
    is_prefill: bool,
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
    qk_norm_policy: int = 0,
    out_q: Optional[torch.Tensor] = None,
    out_k: Optional[torch.Tensor] = None,
    out_v: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Fuse NeoX RoPE, optional Q/K RMSNorm, and paged BF16 KV append.

    ``qkv`` is packed as ``[Q heads, K heads, V heads]``. ``qk_norm_policy``
    is 0 (disabled), 1 (RoPE then RMSNorm), or 2 (RMSNorm then RoPE).
    The optimized hpc-ops kernel supports ``(q_heads, kv_heads)`` equal to
    ``(8, 1)`` or ``(64, 8)``, with Q/K/V head dimension 128 and NHD cache
    layout ``[page, page_size, kv_head, head_dim]``.

    ``seq_lens`` contains total sequence lengths after appending, ``q_indptr``
    maps packed input rows to requests, and ``page_indices`` is the dense
    per-request physical page table.
    """
    if len(paged_kv_cache) != 2:
        raise ValueError("paged_kv_cache must be (key_cache, value_cache)")
    key_cache, value_cache = paged_kv_cache
    q_heads = _q_heads_from_packed_qkv(qkv, key_cache, value_cache)
    if out_q is None:
        out_q = torch.empty(
            (qkv.shape[0], q_heads, key_cache.shape[3]),
            dtype=qkv.dtype,
            device=qkv.device,
        )
    _get_hpc_rope_module().hpc_rope_norm_store_kv(
        out_q,
        key_cache,
        value_cache,
        qkv,
        cos_sin_cache,
        seq_lens,
        q_indptr,
        page_indices,
        is_prefill,
        q_norm_weight,
        k_norm_weight,
        out_k,
        out_v,
        qk_norm_policy,
    )
    return out_q


def fused_qk_norm_rope_quantize_fp8_append_paged_kv_cache(
    qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    q_indptr: torch.Tensor,
    page_indices: torch.Tensor,
    paged_kv_cache: Tuple[torch.Tensor, torch.Tensor],
    is_prefill: bool,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    quant_policy: int,
    max_seqlen: int = 0,
    upper_max: float = 448.0,
    q_scale_inv: Optional[torch.Tensor] = None,
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
    qk_norm_policy: int = 0,
    out_q: Optional[torch.Tensor] = None,
    out_k: Optional[torch.Tensor] = None,
    out_v: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    split_k_flag: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Fused QK-Norm/RoPE, FP8 quantization, and paged KV append.

    ``quant_policy=1`` computes an exact per-token/per-Q-head scale. Decode
    returns scales shaped ``[num_rows, q_heads]``; prefill returns
    ``[num_requests, q_heads, round_up(max_seqlen, 128)]``.
    ``quant_policy=2`` uses scalar ``q_scale_inv`` and returns an empty scale
    tensor. K and V use scalar dequantization scales ``k_scale`` and ``v_scale``.
    Dynamic prefill requires ``max_seqlen`` to cover the longest request in
    ``q_indptr``. ``upper_max`` is the E4M3 finite magnitude and must be in
    ``(0, 448]``. Validation stays asynchronous: if ``q_indptr`` is malformed
    or any request exceeds ``max_seqlen``, the kernel leaves Q/K/V outputs
    untouched and returns ``split_k_flag`` filled with ``-1``.
    """
    if len(paged_kv_cache) != 2:
        raise ValueError("paged_kv_cache must be (key_cache, value_cache)")
    if quant_policy not in (1, 2):
        raise ValueError("quant_policy must be 1 (dynamic Q) or 2 (static Q)")
    if not math.isfinite(upper_max) or not 0.0 < upper_max <= 448.0:
        raise ValueError("upper_max must be finite and in the interval (0, 448]")
    if quant_policy == 2 and q_scale_inv is None:
        raise ValueError("q_scale_inv is required when quant_policy=2")
    if quant_policy == 1 and is_prefill and max_seqlen <= 0:
        raise ValueError("max_seqlen must be positive for dynamic prefill quantization")
    key_cache, value_cache = paged_kv_cache
    q_heads = _q_heads_from_packed_qkv(qkv, key_cache, value_cache)
    num_rows = qkv.shape[0]
    num_requests = seq_lens.shape[0]
    num_kv_heads = key_cache.shape[2]
    if out_q is None:
        out_q = torch.empty(
            (num_rows, q_heads, key_cache.shape[3]),
            dtype=torch.float8_e4m3fn,
            device=qkv.device,
        )
    if q_scale is None:
        if quant_policy == 1 and is_prefill:
            aligned = (max_seqlen + 127) // 128 * 128
            q_scale = torch.empty(
                (num_requests, q_heads, aligned),
                dtype=torch.float32,
                device=qkv.device,
            )
        elif quant_policy == 1:
            q_scale = torch.empty(
                (num_rows, q_heads), dtype=torch.float32, device=qkv.device
            )
        else:
            q_scale = torch.empty(0, dtype=torch.float32, device=qkv.device)
    if split_k_flag is None:
        split_k_flag = torch.zeros(
            (num_requests, num_kv_heads), dtype=torch.int32, device=qkv.device
        )
    else:
        split_k_flag.zero_()
    _get_hpc_rope_module().hpc_rope_norm_store_kv_fp8(
        out_q,
        q_scale,
        split_k_flag,
        key_cache,
        value_cache,
        qkv,
        cos_sin_cache,
        seq_lens,
        q_indptr,
        page_indices,
        is_prefill,
        k_scale,
        v_scale,
        quant_policy,
        max_seqlen,
        upper_max,
        q_scale_inv,
        q_norm_weight,
        k_norm_weight,
        out_k,
        out_v,
        qk_norm_policy,
    )
    return out_q, q_scale, split_k_flag
