"""W4A8-tier route packing: w4a16 semantics, group-granular padding, fast count.

Same output contract as ``pack_topk_routes_by_expert`` (packed_route_indices
slots carry token*topk + j flat route indices sorted by expert; padding slots
carry ``numel``; block_expert_ids carries -1 beyond the live blocks) with two
deliberate differences that this tier's GEMM wants:

- BLOCK granularity is ``group_rows`` = _BLOCKS_PER_CTA * 16 = 48: every
  expert run is padded to a whole number of 3-block GEMM groups, so every
  group is expert-uniform and the GEMM never takes its mixed-expert slow
  path. ``block_expert_ids`` is written PER GROUP (one id per 48 rows).
- The single-program count/prefix kernel of the w4a16 implementation is
  replaced by a multi-program atomic histogram + one tiny prefix program
  (the w4a16 prefix kernel measures ~595us at DS4 m=4096 routes; this path
  measures ~30us).

Capacity (host, static): ``numel + num_experts * (group_rows - 1)`` slots,
rounded up to whole groups -- callers size pri/beids once and prefill the
pri tail beyond ``capacity_slots`` with an invalid value (>= numel).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_COUNT_BLOCK_T = 1024
_SCATTER_BLOCK_T = 256
_POST_BLOCK_T = 256


def w4a8_route_capacity(
    numel: int, num_experts: int, group_rows: int
) -> tuple[int, int]:
    """(capacity_slots, capacity_groups) for worst-case per-expert padding."""
    slots = int(numel) + int(num_experts) * (int(group_rows) - 1)
    groups = (slots + int(group_rows) - 1) // int(group_rows)
    return groups * int(group_rows), groups


@triton.jit
def _w4a8_route_count_kernel(
    topk_ids,
    counts,
    numel,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = offs < numel
    ids = tl.load(topk_ids + offs, mask=mask, other=-1).to(tl.int32)
    valid = mask & (ids >= 0) & (ids < NUM_EXPERTS)
    tl.atomic_add(counts + ids, 1, sem="relaxed", mask=valid)


@triton.jit
def _w4a8_route_prefix_kernel(
    counts,
    expert_offsets,
    packed_route_count,
    GROUP_ROWS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    experts = tl.arange(0, BLOCK_E)
    mask = experts < NUM_EXPERTS
    counts_v = tl.load(counts + experts, mask=mask, other=0)
    padded = ((counts_v + GROUP_ROWS - 1) // GROUP_ROWS) * GROUP_ROWS
    padded = tl.where(mask, padded, 0)
    inclusive = tl.cumsum(padded, axis=0)
    prefix = inclusive - padded
    total = tl.sum(padded, axis=0)
    tl.store(expert_offsets + experts, prefix, mask=mask)
    tl.store(expert_offsets + NUM_EXPERTS, total)
    tl.store(packed_route_count, total)


@triton.jit
def _w4a8_route_post_kernel(
    packed_route_indices,
    block_expert_ids,
    expert_offsets,
    live_numel,
    GROUP_ROWS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    MAX_SLOTS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tl.store(packed_route_indices + offsets, live_numel, mask=offsets < MAX_SLOTS)

    group_rows0 = offsets * GROUP_ROWS
    group_experts = tl.full((BLOCK_T,), -1, dtype=tl.int32)
    valid_groups = offsets < MAX_GROUPS
    for expert in tl.range(0, NUM_EXPERTS):
        start = tl.load(expert_offsets + expert)
        end = tl.load(expert_offsets + expert + 1)
        active = valid_groups & (group_rows0 >= start) & (group_rows0 < end)
        group_experts = tl.where(active, expert, group_experts)
    tl.store(block_expert_ids + offsets, group_experts, mask=valid_groups)


@triton.jit
def _w4a8_route_scatter_kernel(
    topk_ids,
    packed_route_indices,
    expert_offsets,
    numel,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = offs < numel
    ids = tl.load(topk_ids + offs, mask=mask, other=-1).to(tl.int32)
    valid = mask & (ids >= 0) & (ids < NUM_EXPERTS)
    ranks = tl.atomic_add(expert_offsets + ids, 1, sem="relaxed", mask=valid)
    tl.store(packed_route_indices + ranks, offs, mask=valid)


def pack_routes_w4a8(
    topk_ids: torch.Tensor,
    num_experts: int,
    group_rows: int,
    *,
    packed_route_indices: torch.Tensor,
    block_expert_ids: torch.Tensor,
    packed_route_count: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_counts: torch.Tensor,
) -> None:
    """Pack ``topk_ids`` into expert-sorted, group-padded route slots.

    All outputs are caller-preallocated (see :func:`w4a8_route_capacity`);
    ``expert_counts`` is an i32 [num_experts] scratch buffer. Allocation-free
    and host-sync-free (CUDA-graph capturable).
    """
    numel = int(topk_ids.numel())
    cap_slots, cap_groups = w4a8_route_capacity(numel, num_experts, group_rows)
    expert_counts.zero_()
    _w4a8_route_count_kernel[(triton.cdiv(numel, _COUNT_BLOCK_T),)](
        topk_ids,
        expert_counts,
        numel,
        NUM_EXPERTS=num_experts,
        BLOCK_T=_COUNT_BLOCK_T,
        num_warps=4,
    )
    _w4a8_route_prefix_kernel[(1,)](
        expert_counts,
        expert_offsets,
        packed_route_count,
        GROUP_ROWS=group_rows,
        NUM_EXPERTS=num_experts,
        BLOCK_E=triton.next_power_of_2(num_experts),
        num_warps=4,
    )
    post_grid = (
        max(
            triton.cdiv(cap_slots, _POST_BLOCK_T),
            triton.cdiv(cap_groups, _POST_BLOCK_T),
        ),
    )
    _w4a8_route_post_kernel[post_grid](
        packed_route_indices,
        block_expert_ids,
        expert_offsets,
        numel,
        GROUP_ROWS=group_rows,
        NUM_EXPERTS=num_experts,
        MAX_SLOTS=cap_slots,
        MAX_GROUPS=cap_groups,
        BLOCK_T=_POST_BLOCK_T,
        num_warps=4,
    )
    _w4a8_route_scatter_kernel[(triton.cdiv(numel, _SCATTER_BLOCK_T),)](
        topk_ids,
        packed_route_indices,
        expert_offsets,
        numel,
        NUM_EXPERTS=num_experts,
        BLOCK_T=_SCATTER_BLOCK_T,
        num_warps=4,
    )


__all__ = ["pack_routes_w4a8", "w4a8_route_capacity"]
