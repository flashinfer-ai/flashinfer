# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Masked top-k combine reduce (FlashInfer-owned shim).

Replaces the separate-reduce tail of the SM107 MegaMoE launch, i.e. the per-launch
clear of the invalid routes' combine slots (``masked_fill_`` over the whole
capacity) plus the vendored ``TopkReduce`` (BF16 combine over the whole capacity),
with ONE Triton launch over the staged rows only:

    y[t, :] = bf16( sum_k  valid(t, k) * score(t, k) * pre[t, k, :] )          (fp32 sum)

* ``valid(t, k)``: the staged route id is in ``[0, E)``. Invalid routes never get a
  combine slot written by the kernel; instead of zeroing those slots before every
  launch, every slot is loaded (no dependency on the id load) and an invalid one is
  selected to ``+0``, so the sum is bit-identical to TopkReduce over cleared slots
  (``+0`` terms, same fp32 order k = 0..K-1, same first-term initialisation,
  ``fma(v, s, acc)`` when scores are applied here).
"""

from __future__ import annotations

from typing import Optional

import torch

_KERNEL = None

# One token row x 1024 hidden columns per program, 4 warps, select loads. Standalone
# (L2 flushed) the reduce ran at 8-9 TB/s on VR200 for every block shape tried and
# for masked or select loads, so the simplest shape is kept.
_BLOCK_T = 1
_BLOCK_H = 1024
_NUM_WARPS = 4


def _use_pdl(device) -> bool:
    """Programmatic dependent launch behind the megakernel (SM90+)."""
    return torch.cuda.get_device_capability(device)[0] >= 9


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _masked_topk_reduce(
        pre_ptr,
        ids_ptr,
        score_ptr,
        out_ptr,
        num_tokens,
        num_experts,
        pre_stride_t,
        pre_stride_k,
        out_stride_t,
        H: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
        HAS_SCORE: tl.constexpr,
        LAUNCH_PDL: tl.constexpr,
    ):
        if LAUNCH_PDL:
            # Programmatic dependent launch: the launch overlaps the megakernel's
            # tail; every read below waits for the full primary grid.
            tl.extra.cuda.gdc_wait()
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        tmask = t < num_tokens
        hmask = h < H
        t64 = t.to(tl.int64)
        m2 = tmask[:, None] & hmask[None, :]
        acc = tl.zeros([BLOCK_T, BLOCK_H], dtype=tl.float32)
        for k in tl.static_range(TOPK):
            e = tl.load(ids_ptr + t64 * TOPK + k, mask=tmask, other=-1)
            valid = tmask & (e >= 0) & (e < num_experts)
            ptrs = pre_ptr + t64[:, None] * pre_stride_t + k * pre_stride_k + h[None, :]
            # Load every live slot (no dependency on the id load), then select:
            # an invalid route's stale slot (any bits, NaN included) becomes +0.
            v = tl.load(ptrs, mask=m2, other=0.0, eviction_policy="evict_first")
            v = tl.where(valid[:, None], v.to(tl.float32), 0.0)
            if HAS_SCORE:
                s = tl.load(score_ptr + t64 * TOPK + k, mask=tmask, other=0.0)
                if k == 0:
                    acc = v * s[:, None]
                else:
                    acc = tl.fma(v, s[:, None], acc)
            else:
                if k == 0:
                    acc = v
                else:
                    acc = acc + v
        y = acc.to(tl.bfloat16)
        tl.store(out_ptr + t64[:, None] * out_stride_t + h[None, :], y, mask=m2)

    _KERNEL = _masked_topk_reduce
    return _KERNEL


def masked_topk_reduce(
    pre_reduced: torch.Tensor,
    topk_idx: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    out: torch.Tensor,
    *,
    scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``out[:num_tokens] = reduce(pre_reduced[:num_tokens])``.

    ``pre_reduced``: BF16 ``(capacity, topk, >= hidden)`` combine slots (row stride may be
    padded); ``topk_idx``: contiguous int32 ``(capacity, topk)``; ``scores``: optional fp32
    ``(capacity, topk)`` router weights (only when they were not applied at FC1);
    ``out``: BF16 ``(>= num_tokens, hidden)`` with unit inner stride.
    Graph-safe (host ints only)."""
    if num_tokens <= 0:
        return out
    hidden = out.shape[1]
    topk = topk_idx.shape[1]
    if pre_reduced.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("masked_topk_reduce needs BF16 combine slots and output")
    if pre_reduced.stride(2) != 1 or out.stride(1) != 1 or not topk_idx.is_contiguous():
        raise ValueError("masked_topk_reduce needs unit inner strides / contiguous ids")
    if pre_reduced.shape[1] != topk or pre_reduced.shape[2] < hidden:
        raise ValueError(
            f"combine slots {tuple(pre_reduced.shape)} do not match topk {topk} / hidden {hidden}"
        )
    if scores is not None and (
        scores.dtype != torch.float32 or not scores.is_contiguous()
    ):
        raise ValueError("scores must be contiguous fp32")
    import triton

    pdl = _use_pdl(out.device)
    grid = (triton.cdiv(int(num_tokens), _BLOCK_T), triton.cdiv(hidden, _BLOCK_H))
    _get_kernel()[grid](
        pre_reduced,
        topk_idx,
        scores if scores is not None else topk_idx,
        out,
        int(num_tokens),
        int(num_experts),
        pre_reduced.stride(0),
        pre_reduced.stride(1),
        out.stride(0),
        H=hidden,
        TOPK=topk,
        BLOCK_T=_BLOCK_T,
        BLOCK_H=_BLOCK_H,
        HAS_SCORE=scores is not None,
        LAUNCH_PDL=pdl,
        num_warps=_NUM_WARPS,
        **({"launch_pdl": True} if pdl else {}),
    )
    return out


__all__ = ["masked_topk_reduce"]
