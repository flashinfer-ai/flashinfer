# Copyright (c) 2026 by FlashInfer team.
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
"""Cake fused QK RMSNorm + NeoX RoPE + FP8 E4M3 quantization + paged KV append (one launch).

:func:`cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache` takes the
packed BF16 ``qkv`` projection of a decode or prefill step and, in a single
kernel launch,

1. unpacks Q, K and V (``[T, (Hq + 2 * Hkv) * 128]``, Q | K | V order),
2. applies an optional per-head RMSNorm to Q and K (f32 math, f32 weights,
   ``eps = 1e-6``; ``qk_norm_policy`` 0 = none, 1 = RoPE then norm,
   2 = norm then RoPE),
3. applies NeoX rotary embedding (dims ``(i, i + 64)``) from a caller-provided
   ``cos_sin_cache`` table indexed by absolute position,
4. quantizes Q to FP8 E4M3 (``quant_policy=1``: dynamic per-token / per-head
   scale ``max(|q|, 1e-6) / upper_max`` returned in ``q_scale``;
   ``quant_policy=2``: static ``q_scale_inv`` multiplier, empty ``q_scale``),
5. quantizes K and V with the scalar dequantization scales ``k_scale`` /
   ``v_scale`` (payload ``x / scale``) and appends them to an NHD FP8 paged
   cache ``[num_pages, page_size, Hkv, 128]`` (or into caller-owned
   ``out_k`` / ``out_v`` ``[T, Hkv, 128]``),
6. zeroes the unused rows ``[(seq_lens[b] - 1) % page_size + 1, page_size)`` of
   every request's last page in both caches, and
7. validates ``q_indptr`` on the device: a malformed table (``q_indptr[0] != 0``,
   non-monotone, ``q_indptr[-1] != T``) or, for dynamic prefill quantization, a
   request longer than ``max_seqlen`` leaves every output untouched and fills
   ``split_k_flag`` with ``-1``; a valid call sets every flag to 0 (empty
   requests included).  No host synchronisation is involved.

Positions: row ``r`` belongs to request ``b`` when ``q_indptr[b] <= r <
q_indptr[b + 1]`` and has position ``r + seq_lens[b] - q_indptr[b + 1]``
(``seq_lens`` is the total length *after* the append).  K/V of row ``r`` land at
``(page_indices[b, pos // page_size], pos % page_size, kv_head)``.

Supported: ``(Hq, Hkv) in {(8, 1), (64, 8)}``, head_dim 128, BF16 input, FP8
E4M3 output, SM90 (H100/H200), SM100 (B200) and SM103 (B300).  The kernel is
generated from the Cake schedule
``flashinfer_blackwell_fused_qk_rmsnorm_rope_fp8_paged_kv_append``; sources and
manifest live under ``csrc/cake_fused_qk_rope_fp8_append`` (JIT-only).  The call
is CUDA-graph capturable: every launch argument is a by-value scalar or a tensor
pointer, no host synchronisation, workspace or extra kernel is involved.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch

from .jit.cake_fused_qk_rope_fp8_append import (
    STAGES,
    arch_for,
    load_cake_fused_qk_rope_fp8_append_module,
)
from .utils import get_compute_capability

HEAD_DIM = 128
FP8_MAX = 448.0
SUPPORTED_HEADS = ((8, 1), (64, 8))
_STAGE = STAGES[0]


def launch_plan(num_rows: int, num_requests: int, num_kv_heads: int) -> dict[str, Any]:
    """Grid of the fused kernel (mirrors the Cake kernel's ``launch_plan``).

    One CTA per (packed row, kv head group) plus one CTA per (request, kv head)
    for the last-page tail clear and ``split_k_flag``.
    """
    return {"grid": (num_rows + num_requests, num_kv_heads, 1)}


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _heads_from_packed_qkv(
    qkv: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor
) -> Tuple[int, int]:
    _check(
        key_cache.dim() == 4 and value_cache.dim() == 4,
        "key_cache and value_cache must be NHD [num_pages, page_size, num_kv_heads, 128]",
    )
    num_kv_heads = int(key_cache.shape[2])
    width = int(qkv.shape[1])
    _check(
        width % HEAD_DIM == 0 and width // HEAD_DIM > 2 * num_kv_heads,
        "qkv must be [T, (num_q_heads + 2 * num_kv_heads) * 128]",
    )
    num_q_heads = width // HEAD_DIM - 2 * num_kv_heads
    _check(
        (num_q_heads, num_kv_heads) in SUPPORTED_HEADS,
        f"(num_q_heads, num_kv_heads) = ({num_q_heads}, {num_kv_heads}) is not supported; "
        f"expected one of {SUPPORTED_HEADS}",
    )
    return num_q_heads, num_kv_heads


def cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache(
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
    upper_max: float = FP8_MAX,
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
    r"""Fused Q/K RMSNorm + NeoX RoPE + FP8 quantization + paged K/V append in one launch.

    Parameters
    ----------
    qkv : torch.Tensor
        BF16 ``[T, (num_q_heads + 2 * num_kv_heads) * 128]``, packed Q | K | V.  The head
        configuration is derived from this width and the cache's kv-head count.
    cos_sin_cache : torch.Tensor
        F32 ``[max_position, 128]``; row ``p`` is ``cos(p, 0..63) | sin(p, 0..63)``.
    seq_lens : torch.Tensor
        I32 ``[B]`` total sequence length of every request *after* this append.
    q_indptr : torch.Tensor
        I32 ``[B + 1]``; rows ``[q_indptr[b], q_indptr[b + 1])`` belong to request ``b``.
    page_indices : torch.Tensor
        I32 ``[B, max_pages]`` dense physical page table (entries beyond a request's
        pages are ignored).
    paged_kv_cache : (torch.Tensor, torch.Tensor)
        FP8 E4M3 NHD ``(key_cache, value_cache)``, each
        ``[num_pages, page_size, num_kv_heads, 128]`` (written in place).
    is_prefill : bool
        Selects the ``q_scale`` layout of ``quant_policy=1`` (see below).
    k_scale, v_scale : torch.Tensor
        F32 one-element dequantization scales; the stored payload is ``x / scale``.
    quant_policy : int
        1 = dynamic per-token / per-Q-head scale ``max(|q|, 1e-6) / upper_max``;
        2 = static scale: payload ``q * q_scale_inv`` and an empty ``q_scale``.
    max_seqlen : int
        Dynamic prefill (``quant_policy=1`` and ``is_prefill``) only: must cover the
        longest request; ``q_scale`` is ``[B, num_q_heads, round_up(max_seqlen, 128)]``
        addressed by the token index inside its request.
    upper_max : float
        Finite E4M3 magnitude in ``(0, 448]`` used by the dynamic scale.
    q_scale_inv : torch.Tensor, optional
        F32 one-element static Q multiplier (required for ``quant_policy=2``).
    q_norm_weight, k_norm_weight : torch.Tensor, optional
        F32 ``[128]`` per-head RMSNorm weights (required unless ``qk_norm_policy == 0``).
    qk_norm_policy : int
        0 no norm, 1 RoPE then RMSNorm, 2 RMSNorm then RoPE.
    out_q : torch.Tensor, optional
        FP8 ``[T, num_q_heads, 128]`` output buffer (allocated when ``None``).
    out_k, out_v : torch.Tensor, optional
        FP8 ``[T, num_kv_heads, 128]``; when both are given, K / V rows are written there
        instead of the caches (the last-page tail clear still runs).
    q_scale : torch.Tensor, optional
        F32 output buffer: decode ``[T, num_q_heads]``, dynamic prefill
        ``[B, num_q_heads, round_up(max_seqlen, 128)]``, ``quant_policy=2`` empty.
    split_k_flag : torch.Tensor, optional
        I32 ``[B, num_kv_heads]`` output buffer; every element is written (0 valid,
        -1 invalid metadata), no host-side clear is needed.

    Returns
    -------
    (out_q, q_scale, split_k_flag)
        The caller-owned buffers when given, otherwise freshly allocated ones.
    """
    _check(len(paged_kv_cache) == 2, "paged_kv_cache must be (key_cache, value_cache)")
    key_cache, value_cache = paged_kv_cache
    _check(quant_policy in (1, 2), "quant_policy must be 1 (dynamic Q) or 2 (static Q)")
    _check(qk_norm_policy in (0, 1, 2), "qk_norm_policy must be 0, 1 or 2")
    _check(
        math.isfinite(upper_max) and 0.0 < upper_max <= FP8_MAX,
        "upper_max must be finite and in the interval (0, 448]",
    )
    _check(
        qkv.dim() == 2 and qkv.dtype == torch.bfloat16, "qkv must be a 2-D bf16 tensor"
    )
    _check(qkv.is_cuda, "qkv must be a CUDA tensor")
    device = qkv.device
    num_q_heads, num_kv_heads = _heads_from_packed_qkv(qkv, key_cache, value_cache)
    num_rows = int(qkv.shape[0])
    dyn_prefill = quant_policy == 1 and bool(is_prefill)
    _check(
        not dyn_prefill or max_seqlen > 0,
        "max_seqlen must be positive for dynamic prefill quantization",
    )
    _check(
        cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.dim() == 2
        and int(cos_sin_cache.shape[1]) == HEAD_DIM,
        "cos_sin_cache must be f32 [max_position, 128]",
    )
    for name, t in (
        ("seq_lens", seq_lens),
        ("q_indptr", q_indptr),
        ("page_indices", page_indices),
    ):
        _check(t.dtype == torch.int32, f"{name} must be int32")
    num_requests = int(seq_lens.shape[0])
    _check(
        seq_lens.dim() == 1
        and q_indptr.dim() == 1
        and int(q_indptr.shape[0]) == num_requests + 1,
        "q_indptr must be [B + 1] and seq_lens [B]",
    )
    _check(
        page_indices.dim() == 2 and int(page_indices.shape[0]) == num_requests,
        "page_indices must be [B, max_pages]",
    )
    for name, t in (("key_cache", key_cache), ("value_cache", value_cache)):
        _check(
            t.dtype == torch.float8_e4m3fn
            and int(t.shape[2]) == num_kv_heads
            and int(t.shape[3]) == HEAD_DIM,
            f"{name} must be float8_e4m3fn [num_pages, page_size, num_kv_heads, 128]",
        )
    _check(
        tuple(key_cache.shape) == tuple(value_cache.shape),
        "key_cache and value_cache must have equal shapes",
    )
    page_size = int(key_cache.shape[1])
    for name, t in (("k_scale", k_scale), ("v_scale", v_scale)):
        _check(
            t.dtype == torch.float32 and t.numel() == 1,
            f"{name} must be a one-element f32 tensor",
        )
    if quant_policy == 2:
        _check(q_scale_inv is not None, "q_scale_inv is required when quant_policy=2")
        _check(
            q_scale_inv.dtype == torch.float32 and q_scale_inv.numel() == 1,
            "q_scale_inv must be a one-element f32 tensor",
        )
    else:
        # Never read by the kernel under quant_policy=1: hand it an existing f32 scalar.
        q_scale_inv = k_scale
    if qk_norm_policy != 0:
        _check(
            q_norm_weight is not None and k_norm_weight is not None,
            "q_norm_weight and k_norm_weight are required when qk_norm_policy != 0",
        )
    if q_norm_weight is None or k_norm_weight is None:
        # qk_norm_policy == 0 never reads the weights: hand the kernel an existing f32 [128]
        # row instead of allocating and filling a tensor on every call (graph-safe).
        placeholder = cos_sin_cache[0, :HEAD_DIM]
        q_norm_weight = placeholder if q_norm_weight is None else q_norm_weight
        k_norm_weight = placeholder if k_norm_weight is None else k_norm_weight
    for name, w in (("q_norm_weight", q_norm_weight), ("k_norm_weight", k_norm_weight)):
        _check(
            w.dtype == torch.float32 and tuple(w.shape) == (HEAD_DIM,),
            f"{name} must be f32 [128]",
        )
    if out_q is None:
        out_q = torch.empty(
            (num_rows, num_q_heads, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
        )
    _check(
        out_q.dtype == torch.float8_e4m3fn
        and tuple(out_q.shape) == (num_rows, num_q_heads, HEAD_DIM),
        "out_q must be float8_e4m3fn [T, num_q_heads, 128]",
    )
    _check((out_k is None) == (out_v is None), "out_k and out_v must be given together")
    has_out_kv = out_k is not None
    for name, t in (("out_k", out_k), ("out_v", out_v)):
        if t is not None:
            _check(
                t.dtype == torch.float8_e4m3fn
                and tuple(t.shape) == (num_rows, num_kv_heads, HEAD_DIM),
                f"{name} must be float8_e4m3fn [T, num_kv_heads, 128]",
            )
    aligned = (max_seqlen + 127) // 128 * 128 if dyn_prefill else 0
    if dyn_prefill:
        q_scale_shape: Tuple[int, ...] = (num_requests, num_q_heads, aligned)
    elif quant_policy == 1:
        q_scale_shape = (num_rows, num_q_heads)
    else:
        q_scale_shape = (0,)
    if q_scale is None:
        q_scale = torch.empty(q_scale_shape, dtype=torch.float32, device=device)
    _check(
        q_scale.dtype == torch.float32 and tuple(q_scale.shape) == q_scale_shape,
        f"q_scale must be f32 {q_scale_shape} for this quant_policy / is_prefill",
    )
    if split_k_flag is None:
        split_k_flag = torch.empty(
            (num_requests, num_kv_heads), dtype=torch.int32, device=device
        )
    _check(
        split_k_flag.dtype == torch.int32
        and tuple(split_k_flag.shape) == (num_requests, num_kv_heads),
        "split_k_flag must be int32 [B, num_kv_heads]",
    )
    tensors = {
        "qkv": qkv,
        "cos_sin": cos_sin_cache,
        "seq_lens": seq_lens,
        "q_indptr": q_indptr,
        "page_indices": page_indices,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "q_scale_inv": q_scale_inv,
        "out_q": out_q,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "out_k": out_k if has_out_kv else key_cache,
        "out_v": out_v if has_out_kv else value_cache,
        "q_scale": q_scale,
        "split_k_flag": split_k_flag,
    }
    for name, t in tensors.items():
        _check(t.is_contiguous(), f"{name} must be contiguous")
        _check(t.device == device, f"{name} must be on {device}")

    arch = arch_for(get_compute_capability(device))
    module, record = load_cake_fused_qk_rope_fp8_append_module(_STAGE, arch)
    plan = launch_plan(num_rows, num_requests, num_kv_heads)
    scalars = {
        "num_rows": num_rows,
        "num_requests": num_requests,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "page_size": page_size,
        "max_pages_per_request": int(page_indices.shape[1]),
        "max_seqlen": int(max_seqlen),
        "max_seqlen_aligned": aligned,
        "quant_policy": int(quant_policy),
        "norm_policy": int(qk_norm_policy),
        "is_prefill": bool(is_prefill),
        "has_out_kv": has_out_kv,
        "upper_max": float(upper_max),
        "grid_x": plan["grid"][0],
        "grid_y": plan["grid"][1],
        "grid_z": plan["grid"][2],
    }
    args = []
    for kind, name in record["arg_plan"]:
        if name in tensors:
            args.append(tensors[name])
        elif name in scalars:
            args.append(scalars[name])
        else:
            raise RuntimeError(
                f"generated module expects unknown argument {name!r} ({kind})"
            )
    import tvm_ffi

    with torch.cuda.device(device), tvm_ffi.use_torch_stream():
        getattr(module, str(record["ffi_entry"]))(*args)
    return out_q, q_scale, split_k_flag


__all__ = [
    "cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache",
    "launch_plan",
]
