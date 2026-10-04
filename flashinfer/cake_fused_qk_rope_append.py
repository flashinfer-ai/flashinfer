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
(H100/H200), SM100 (B200) and SM103 (B300).  The kernel is generated from the
Cake program ``fused_qk_rmsnorm_rope_paged_kv_append_bf16_v4``; sources
and manifest live under ``csrc/cake_fused_qk_rope_append`` (JIT-only).  Several
builds of the same schedule are shipped per architecture (warps per row, request
lookup, V placement, clearing granularity); the host picks one from the
architecture and the problem shape with the Cake policy mirrored in
:func:`flashinfer.jit.cake_fused_qk_rope_append.stage_for` - every build writes
bit-identical outputs.  The call is CUDA-graph capturable: every launch argument
is a by-value scalar or a tensor pointer, no host synchronisation or workspace
is involved; the selected build depends only on shapes, so a captured graph
replays the same build.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .jit.cake_fused_qk_rope_append import (
    arch_for,
    clear_units_per_request,
    load_cake_fused_qk_rope_append_module,
    stage_for,
)
from .utils import get_compute_capability

HEAD_DIM = 128
_THREADS = 128


def launch_plan(
    num_rows: int,
    num_requests: int,
    page_size: int,
    num_kv_heads: int,
    warps_per_row: int,
    clear_units_per_warp: int = 1,
) -> dict[str, Any]:
    """Grid and host-derived scalars (mirrors the Cake kernel's ``launch_plan``).

    ``warps_per_row`` / ``clear_units_per_warp`` are the selected stage's constexpr specialization.
    """
    rows_per_cta = max(1, 4 // warps_per_row)
    ctas_per_row = (warps_per_row + 3) // 4
    num_row_ctas = ((num_rows + rows_per_cta - 1) // rows_per_cta) * ctas_per_row
    units = clear_units_per_request(page_size, num_kv_heads)
    warps_per_request = (units + clear_units_per_warp - 1) // clear_units_per_warp
    num_clear_ctas = (num_requests * warps_per_request + 3) // 4
    return {
        "num_row_ctas": num_row_ctas,
        "clear_units_per_request": units,
        "grid": (num_row_ctas + num_clear_ctas, 1, 1),
    }


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


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
        BF16 ``[T, num_q_heads, 128]`` normalised and rotated queries.
    """
    _check(qk_norm_policy in (0, 1, 2), "qk_norm_policy must be 0, 1 or 2")
    _check(
        qkv.dim() == 2 and qkv.dtype == torch.bfloat16, "qkv must be a 2-D bf16 tensor"
    )
    _check(qkv.is_cuda, "qkv must be a CUDA tensor")
    device = qkv.device
    num_rows = int(qkv.shape[0])
    width = (num_q_heads + 2 * num_kv_heads) * HEAD_DIM
    _check(
        int(qkv.shape[1]) == width,
        f"qkv must have {width} columns for this head configuration",
    )
    _check(
        cos_sin.dtype == torch.float32
        and cos_sin.dim() == 2
        and int(cos_sin.shape[1]) == HEAD_DIM,
        "cos_sin must be f32 [max_position, 128]",
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
            t.dtype == torch.bfloat16
            and t.dim() == 4
            and int(t.shape[2]) == num_kv_heads
            and int(t.shape[3]) == HEAD_DIM,
            f"{name} must be bf16 [num_pages, page_size, num_kv_heads, 128]",
        )
    _check(
        tuple(key_cache.shape) == tuple(value_cache.shape),
        "key_cache and value_cache must have equal shapes",
    )
    page_size = int(key_cache.shape[1])
    if out_q is None:
        out_q = torch.empty(
            (num_rows, num_q_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
    _check(
        out_q.dtype == torch.bfloat16
        and tuple(out_q.shape) == (num_rows, num_q_heads, HEAD_DIM),
        "out_q must be bf16 [T, num_q_heads, 128]",
    )
    for name, t in (("out_k", out_k), ("out_v", out_v)):
        if t is not None:
            _check(
                t.dtype == torch.bfloat16
                and tuple(t.shape) == (num_rows, num_kv_heads, HEAD_DIM),
                f"{name} must be bf16 [T, num_kv_heads, 128]",
            )
    if qk_norm_policy != 0:
        _check(
            q_norm_weight is not None and k_norm_weight is not None,
            "q_norm_weight and k_norm_weight are required when qk_norm_policy != 0",
        )
    if q_norm_weight is None or k_norm_weight is None:
        # qk_norm_policy == 0 never reads the weights: hand the kernel an existing f32 [128] row
        # instead of allocating and filling a tensor on every call (no extra kernel, graph-safe).
        placeholder = cos_sin[0, :HEAD_DIM]
        q_norm_weight = placeholder if q_norm_weight is None else q_norm_weight
        k_norm_weight = placeholder if k_norm_weight is None else k_norm_weight
    for name, w in (("q_norm_weight", q_norm_weight), ("k_norm_weight", k_norm_weight)):
        _check(
            w.dtype == torch.float32 and tuple(w.shape) == (HEAD_DIM,),
            f"{name} must be f32 [128]",
        )
    tensors = {
        "qkv": qkv,
        "cos_sin": cos_sin,
        "q_indptr": q_indptr,
        "seq_lens": seq_lens,
        "page_table": page_indices,
        "k_cache": key_cache,
        "v_cache": value_cache,
        "out_q": out_q,
        "out_k": out_k if out_k is not None else key_cache,
        "out_v": out_v if out_v is not None else value_cache,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
    }
    for name, t in tensors.items():
        _check(t.is_contiguous(), f"{name} must be contiguous")
        _check(t.device == device, f"{name} must be on {device}")

    arch = arch_for(get_compute_capability(device))
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    stage = stage_for(
        num_q_heads,
        num_kv_heads,
        arch=arch,
        num_rows=num_rows,
        num_requests=num_requests,
        page_size=page_size,
        num_sms=num_sms,
    )
    module, record = load_cake_fused_qk_rope_append_module(stage, arch)
    specialization = dict(record["route"]).get("specialization", {})
    plan = launch_plan(
        num_rows,
        num_requests,
        page_size,
        num_kv_heads,
        int(specialization.get("WARPS_PER_ROW", 1)),
        int(specialization.get("CLEAR_UNITS_PER_WARP", 1)),
    )
    scalars = {
        "num_rows": num_rows,
        "num_requests": num_requests,
        "max_pages": int(page_indices.shape[1]),
        "page_size": page_size,
        "k_page_stride": int(key_cache.stride(0)),
        "v_page_stride": int(value_cache.stride(0)),
        "num_row_ctas": plan["num_row_ctas"],
        "clear_units_per_request": plan["clear_units_per_request"],
        "qk_norm_policy": int(qk_norm_policy),
        "eps": float(eps),
        "use_out_k": 0 if out_k is None else 1,
        "use_out_v": 0 if out_v is None else 1,
        "clear_last_page": 1 if clear_unused_last_page_rows else 0,
        "grid_x": plan["grid"][0],
        "grid_y": 1,
        "grid_z": 1,
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
    return out_q


__all__ = ["cake_fused_qk_rmsnorm_rope_append_paged_kv_cache", "launch_plan"]
