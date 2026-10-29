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
E4M3 output, SM90 (H100/H200), SM100 (B200) and SM103 (B300).  The kernel is a
Cake program; its generated sources, build manifest and loader live in the
self-contained package :mod:`flashinfer.cake_fused_qk_rope_fp8_append_backend`
(built on first use through FlashInfer's JIT, or loaded prebuilt).  Several
builds of the same schedule are shipped per architecture; the host picks one
from the architecture, the SM count and the problem's host scalars with the
Cake stage policy mirrored in
:func:`flashinfer.cake_fused_qk_rope_fp8_append_backend.stage_for` - every
build writes bit-identical outputs.  The call is CUDA-graph capturable: every
launch argument is a by-value scalar or a tensor pointer, no host
synchronisation, workspace or extra kernel is involved.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from . import cake_fused_qk_rope_fp8_append_backend as _backend
from .api_logging import flashinfer_api
from .cake_fused_qk_rope_fp8_append_backend import (  # noqa: F401
    FP8_MAX,
    HEAD_DIM,
    SUPPORTED_COMPUTE_CAPABILITIES,
    SUPPORTED_HEADS,
    launch_plan,
)

# The stable entry point is a real ``def`` here (public-API tooling reads the
# decorated signature of this module); its body lives in the self-contained
# backend package next to the generated CUDA sources.


@flashinfer_api
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
    return _backend.cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache(
        qkv,
        cos_sin_cache,
        seq_lens,
        q_indptr,
        page_indices,
        paged_kv_cache,
        is_prefill,
        k_scale,
        v_scale,
        quant_policy,
        max_seqlen=max_seqlen,
        upper_max=upper_max,
        q_scale_inv=q_scale_inv,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
        out_q=out_q,
        out_k=out_k,
        out_v=out_v,
        q_scale=q_scale,
        split_k_flag=split_k_flag,
    )


__all__ = [
    "cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache",
    "launch_plan",
]
