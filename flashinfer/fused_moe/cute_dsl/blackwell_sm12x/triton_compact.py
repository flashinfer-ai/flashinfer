"""Triton kernel for compacting MoE routing IDs (ported from b12x).

Remaps global expert IDs to dense local indices (0, 1, 2, ...) for the
micro MoE kernel, which expects pre-compacted routing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_topk_ids_kernel(
    topk_ids_ptr,
    compact_topk_ids_ptr,
    weight_expert_ids_ptr,
    active_expert_count_ptr,
    total_pairs,
    BLOCK: tl.constexpr,
):
    pair_slots = tl.arange(0, BLOCK)
    valid = pair_slots < total_pairs
    ids = tl.load(topk_ids_ptr + pair_slots, mask=valid, other=-1).to(tl.int32)

    row_slots = pair_slots[:, None]
    col_slots = pair_slots[None, :]
    row_valid = valid[:, None]
    col_valid = valid[None, :]

    same_id = ids[:, None] == ids[None, :]
    prior_same = row_valid & col_valid & same_id & (col_slots < row_slots)

    first_flags = valid & (tl.sum(prior_same.to(tl.int32), axis=1) == 0)
    first_prefix = tl.cumsum(first_flags.to(tl.int32), axis=0)

    prior_slots = tl.where(prior_same, col_slots, BLOCK)
    first_match = tl.min(prior_slots, axis=1)
    first_slot = tl.where(first_match < BLOCK, first_match, pair_slots)
    first_slot_mask = col_slots == first_slot[:, None]
    compact_id = tl.sum(tl.where(first_slot_mask, first_prefix[None, :], 0), axis=1) - 1

    tl.store(compact_topk_ids_ptr + pair_slots, compact_id, mask=valid)
    tl.store(weight_expert_ids_ptr + compact_id, ids, mask=valid & first_flags)

    active_expert_count = tl.sum(first_flags.to(tl.int32), axis=0)
    tl.store(active_expert_count_ptr, active_expert_count)


def compact_topk_ids(
    topk_ids: torch.Tensor,
    compact_topk_ids: torch.Tensor,
    weight_expert_ids: torch.Tensor,
    active_expert_count: torch.Tensor,
) -> None:
    """Remap global expert IDs to dense contiguous local indices.

    Args:
        topk_ids: [total_pairs] int32 — flattened global expert IDs.
        compact_topk_ids: [total_pairs] int32 — output: dense local indices.
        weight_expert_ids: [>=total_pairs] int32 — output: local->global map.
        active_expert_count: [1] int32 — output: number of unique experts.
    """
    total_pairs = topk_ids.numel()
    if total_pairs == 0:
        active_expert_count.zero_()
        return
    if compact_topk_ids.numel() < total_pairs:
        raise ValueError("compact_topk_ids must have at least total_pairs elements")
    if weight_expert_ids.numel() < total_pairs:
        raise ValueError("weight_expert_ids must have at least total_pairs elements")
    if active_expert_count.numel() != 1:
        raise ValueError("active_expert_count must have shape [1]")

    block = triton.next_power_of_2(total_pairs)
    num_warps = 1 if block <= 16 else 2
    _compact_topk_ids_kernel[(1,)](
        topk_ids,
        compact_topk_ids,
        weight_expert_ids,
        active_expert_count,
        total_pairs,
        BLOCK=block,
        num_warps=num_warps,
    )
