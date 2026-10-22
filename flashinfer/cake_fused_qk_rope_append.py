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
"""Cake fused QK RMSNorm + NeoX RoPE + BF16 Q output + paged KV append (one launch).

:func:`cake_fused_qk_rmsnorm_rope_append_paged_kv_cache` takes the packed BF16
``qkv`` projection of a decode or small-chunk prefill step and, in a single
kernel launch,

1. unpacks Q, K and V (``[T, (Hq + 2 * Hkv) * 128]``, Q | K | V order),
2. applies an optional per-head RMSNorm to Q and K (f32 math, f32 weights,
   ``eps`` default ``1e-6``; ``qk_norm_policy`` 0 = none, 1 = RoPE then norm,
   2 = norm then RoPE),
3. applies NeoX rotary embedding (dims ``(i, i + 64)``) from a caller-provided
   ``cos_sin`` table indexed by absolute position,
4. writes Q as BF16 ``[T, Hq, 128]`` and appends K/V to an NHD paged cache
   ``[num_pages, page_size, Hkv, 128]`` (or into caller-owned ``out_k`` /
   ``out_v`` ``[T, Hkv, 128]``),
5. zeroes the unused rows of every request's last page in both caches
   (``clear_unused_last_page_rows=True``), including when K/V go to
   caller-owned buffers.

Positions: row ``r`` belongs to request ``b`` when ``q_indptr[b] <= r <
q_indptr[b + 1]`` and has position ``r + seq_lens[b] - q_indptr[b + 1]``
(``seq_lens`` is the total length *after* the append).  Rows outside every
request or with a negative position produce no output.  K/V of row ``r`` land
at ``(page_indices[b, pos // page_size], pos % page_size, kv_head)``.

Supported: ``(Hq, Hkv) in {(8, 1), (64, 8)}``, head_dim 128, BF16 in/out, SM90
(H100/H200), SM100 (B200) and SM103 (B300).  The kernel is a Cake program; its
generated sources, build manifest and loader live in the self-contained package
:mod:`flashinfer.cake_fused_qk_rope_append_backend` (built on first use through
FlashInfer's JIT, or loaded prebuilt).  Several builds of the same schedule are
shipped per architecture (warps per row, request lookup, V placement, clearing
granularity and this round's levers); the host picks one from the architecture
and the problem shape with the Cake stage policy mirrored in
:func:`flashinfer.cake_fused_qk_rope_append_backend.stage_for` - every build
writes bit-identical outputs.  The call is CUDA-graph capturable: every launch
argument is a by-value scalar or a tensor pointer, no host synchronisation or
workspace is involved; the selected build depends only on shapes, so a captured
graph replays the same build.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import cake_fused_qk_rope_append_backend as _backend
from .api_logging import flashinfer_api
from .cake_fused_qk_rope_append_backend import (  # noqa: F401
    HEAD_DIM,
    SUPPORTED_COMPUTE_CAPABILITIES,
    launch_plan,
)

# The stable entry point is a real ``def`` here (public-API tooling reads the
# decorated signature of this module); its body lives in the self-contained
# backend package next to the generated CUDA sources.


@flashinfer_api
def cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
    qkv: torch.Tensor,
    cos_sin: torch.Tensor,
    seq_lens: torch.Tensor,
    q_indptr: torch.Tensor,
    page_indices: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    qk_norm_policy: int = 0,
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    out_q: Optional[torch.Tensor] = None,
    out_k: Optional[torch.Tensor] = None,
    out_v: Optional[torch.Tensor] = None,
    clear_unused_last_page_rows: bool = True,
) -> torch.Tensor:
    r"""Fused Q/K RMSNorm + NeoX RoPE + BF16 Q output + paged K/V append in one launch.

    Parameters
    ----------
    qkv : torch.Tensor
        BF16 ``[T, (num_q_heads + 2 * num_kv_heads) * 128]``, packed Q | K | V.
    cos_sin : torch.Tensor
        F32 ``[max_position, 128]``; row ``p`` is ``cos(p, 0..63) | sin(p, 0..63)``.
    seq_lens : torch.Tensor
        I32 ``[B]`` total sequence length of every request *after* this append.
    q_indptr : torch.Tensor
        I32 ``[B + 1]``; rows ``[q_indptr[b], q_indptr[b + 1])`` belong to request ``b``.
    page_indices : torch.Tensor
        I32 ``[B, max_pages]`` dense physical page table (entries beyond a request's
        pages are ignored).
    key_cache, value_cache : torch.Tensor
        BF16 NHD caches ``[num_pages, page_size, num_kv_heads, 128]`` (written in place).
    num_q_heads, num_kv_heads : int
        Head configuration; ``(8, 1)`` and ``(64, 8)`` are supported.
    qk_norm_policy : int
        0 no norm, 1 RoPE then RMSNorm, 2 RMSNorm then RoPE.
    q_norm_weight, k_norm_weight : torch.Tensor, optional
        F32 ``[128]`` per-head RMSNorm weights (required unless ``qk_norm_policy == 0``).
    eps : float
        RMSNorm epsilon (default ``1e-6``).
    out_q : torch.Tensor, optional
        BF16 ``[T, num_q_heads, 128]`` output buffer (allocated when ``None``).
    out_k, out_v : torch.Tensor, optional
        BF16 ``[T, num_kv_heads, 128]``; when given, K / V are written there instead
        of the caches (the last-page clear still runs).
    clear_unused_last_page_rows : bool
        Zero rows ``[(seq_lens[b] - 1) % page_size + 1, page_size)`` of every request's
        last page in both caches (all kv heads).

    Returns
    -------
    out_q : torch.Tensor
        BF16 ``[T, num_q_heads, 128]`` normalised and rotated queries (the caller's
        ``out_q`` when given).
    """
    return _backend.cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
        qkv,
        cos_sin,
        seq_lens,
        q_indptr,
        page_indices,
        key_cache,
        value_cache,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qk_norm_policy=qk_norm_policy,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        eps=eps,
        out_q=out_q,
        out_k=out_k,
        out_v=out_v,
        clear_unused_last_page_rows=clear_unused_last_page_rows,
    )


__all__ = ["cake_fused_qk_rmsnorm_rope_append_paged_kv_cache", "launch_plan"]
