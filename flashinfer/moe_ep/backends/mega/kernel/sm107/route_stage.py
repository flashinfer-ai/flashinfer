"""One-launch routing staging for the SM107 MegaMoE backends (graph-safe).

``stage_inputs`` used ~16 small torch launches per forward for the routing plane: the
device-side route validation (sort, range / duplicate / finite checks, ``all()``
reductions) plus the int32 / fp32 copies and the capacity-tail ``-1`` fill. In a CUDA graph
those cost ~25-30 us per MoE layer (VR200, measured), ~10% of a decode-size MegaMoE step.

:func:`stage_routes` does all of it in one Triton launch: ids -> int32 staging, weights ->
fp32 staging, tail rows -> ``-1``, and (unless ``FLASHINFER_SM107_MEGA_CHECK_ROUTES=0``)
the same contract as ``validate_routing_values`` (ids in ``[-1, E)``, no duplicate valid id
in a row, finite weights) folded into one error flag + one ``torch._assert_async``.
Inputs this launch does not handle fall back to the torch staging, which always runs
``validate_routing_values``.
"""

from __future__ import annotations

import os

import torch

_KERNEL = None


def _check_routes() -> bool:
    return os.environ.get("FLASHINFER_SM107_MEGA_CHECK_ROUTES", "1") != "0"


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _stage_routes_kernel(
        ids_ptr,
        w_ptr,
        ids_out_ptr,
        w_out_ptr,
        flag_ptr,
        pad_ptr,
        num_tokens,
        capacity,
        num_experts,
        TOPK: tl.constexpr,
        KP: tl.constexpr,
        BLOCK_R: tl.constexpr,
        CHECK: tl.constexpr,
        HAS_PAD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = tl.arange(0, KP)
        in_cap = rows < capacity
        live = rows < num_tokens
        kmask = cols < TOPK
        m_live = live[:, None] & kmask[None, :]
        m_cap = in_cap[:, None] & kmask[None, :]
        offs = rows[:, None].to(tl.int64) * TOPK + cols[None, :]
        ids = tl.load(ids_ptr + offs, mask=m_live, other=-1).to(tl.int32)
        w = tl.load(w_ptr + offs, mask=m_live, other=0.0).to(tl.float32)
        routed = live
        if HAS_PAD:
            # The caller's padding mask (e.g. vLLM's is_padding rows with
            # VLLM_MOE_SKIP_PADDING) folded in: padding rows stage no routes.
            pad = tl.load(pad_ptr + rows, mask=live, other=0) != 0
            routed = live & ~pad
            ids = tl.where(routed[:, None], ids, -1)
        m_live = routed[:, None] & kmask[None, :]
        out_ids = tl.where(m_live, ids, -1)
        # Every staging store stays inside the capacity rows: the grid covers
        # cdiv(capacity, BLOCK_R) * BLOCK_R rows, so a live-only mask would
        # write past topk_weights_out when capacity % BLOCK_R != 0 and
        # num_tokens > capacity (the host wrapper also rejects that case before
        # launching).
        tl.store(ids_out_ptr + offs, out_ids, mask=m_cap)
        tl.store(w_out_ptr + offs, w, mask=m_cap & m_live)
        if CHECK:
            bad_range = m_live & ((ids < -1) | (ids >= num_experts))
            bad_w = m_live & ~((w - w) == 0.0)  # NaN and +-inf give NaN
            # duplicate valid ids within a row (TOPK <= KP small: pairwise compare)
            a = tl.reshape(ids, (BLOCK_R, KP, 1))
            b = tl.reshape(ids, (BLOCK_R, 1, KP))
            ii = tl.reshape(cols, (1, KP, 1))
            jj = tl.reshape(cols, (1, 1, KP))
            pair = (ii < jj) & (jj < TOPK)
            dup = (a == b) & (a >= 0) & pair & tl.reshape(live, (BLOCK_R, 1, 1))
            n_bad = (
                tl.sum(tl.sum(dup.to(tl.int32), axis=2), axis=1)
                + tl.sum(bad_range.to(tl.int32), axis=1)
                + tl.sum(bad_w.to(tl.int32), axis=1)
            )
            total = tl.sum(n_bad, axis=0)
            if total > 0:
                tl.atomic_add(flag_ptr, total)

    _KERNEL = _stage_routes_kernel
    return _KERNEL


def stage_routes(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    num_experts: int,
    pad_mask: torch.Tensor | None = None,
) -> bool:
    """Stage + validate the routing plane in one launch. ``False`` -> caller falls back
    (non-CUDA or non-contiguous tensors, unexpected dtypes or shapes, topk > 16).

    ``pad_mask`` (bool / uint8 ``[>= num_tokens]``, optional): rows where it is set stage
    no routes (all ids -1) and are left out of the route check."""
    tensors = (topk_ids, topk_weights, topk_idx_out, topk_weights_out)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        return False
    if pad_mask is not None:
        if not (pad_mask.is_cuda and pad_mask.is_contiguous() and pad_mask.dim() == 1):
            return False
        if pad_mask.shape[0] < topk_ids.shape[0]:
            return False
        if pad_mask.dtype == torch.bool:
            pad_mask = pad_mask.view(torch.uint8)
        elif pad_mask.dtype != torch.uint8:
            return False
    if topk_idx_out.dtype != torch.int32 or topk_weights_out.dtype != torch.float32:
        return False
    num_tokens, topk = topk_ids.shape
    capacity = topk_idx_out.shape[0]
    if (
        topk > 16
        or topk_idx_out.shape[1] != topk
        or topk_weights.shape != topk_ids.shape
    ):
        return False
    if topk_weights_out.shape != topk_idx_out.shape:
        return False
    if num_tokens > capacity:
        # Host ints only (graph-safe). The torch staging path fails on the same
        # shapes; never truncate the routing plane silently.
        raise ValueError(
            f"stage_routes: {num_tokens} tokens exceed the workspace capacity "
            f"{capacity}; allocate the MegaMoE workspace with a capacity >= the "
            "per-rank token count"
        )
    import triton

    kp = triton.next_power_of_2(topk)
    block_r = 64
    check = _check_routes() and num_tokens > 0
    flag = (
        torch.zeros(1, dtype=torch.int32, device=topk_ids.device)
        if check
        else topk_idx_out
    )
    grid = (triton.cdiv(max(capacity, 1), block_r),)
    _get_kernel()[grid](
        topk_ids,
        topk_weights,
        topk_idx_out,
        topk_weights_out,
        flag,
        pad_mask if pad_mask is not None else topk_idx_out,
        int(num_tokens),
        int(capacity),
        int(num_experts),
        TOPK=topk,
        KP=kp,
        BLOCK_R=block_r,
        CHECK=check,
        HAS_PAD=pad_mask is not None,
    )
    if check:
        torch._assert_async(
            flag[0] == 0,
            "SM107 routes require unique expert IDs in [0, E), or -1, and finite scores",
        )
    return True


__all__ = ["stage_routes"]
