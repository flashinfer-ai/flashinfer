"""Triton route-packing kernels for W4A16 MoE."""

from __future__ import annotations

import triton
import triton.language as tl
import torch


_COUNT_BLOCK_T = 256
_SORT_BLOCK_T = 256


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _max_packed_route_slots(numel: int, block_size: int, num_experts: int) -> int:
    max_packed = int(numel) + int(num_experts) * (int(block_size) - 1)
    if int(numel) < int(num_experts):
        max_packed = min(int(numel) * int(block_size), max_packed)
    return max_packed


def _workspace_slice(
    tensor: torch.Tensor | None,
    *,
    name: str,
    elements: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    elements = int(elements)
    if tensor is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{name} is not initialized for CUDA graph capture; "
                "provide a preallocated W4A16 route-packing workspace"
            )
        return torch.empty((elements,), dtype=dtype, device=device)
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if int(tensor.numel()) < elements:
        raise ValueError(
            f"{name} has {tensor.numel()} elements; need at least {elements}"
        )
    return tensor[:elements]


@triton.jit
def _pack_topk_routes_prefix_kernel(
    topk_ids,
    expert_map,
    packed_route_indices,
    block_expert_ids,
    packed_route_count,
    expert_offsets,
    NUMEL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    MAX_PACKED_ROUTES: tl.constexpr,
    MAX_ROUTE_BLOCKS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_ROUTE_INIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    experts = tl.arange(0, BLOCK_E)
    expert_mask = experts < NUM_EXPERTS
    counts = tl.zeros((BLOCK_E,), dtype=tl.int32)

    route_offsets = tl.arange(0, BLOCK_T)
    for start in tl.range(0, NUMEL, BLOCK_T):
        offsets = start + route_offsets
        raw_ids = tl.load(topk_ids + offsets, mask=offsets < NUMEL, other=-1).to(
            tl.int32
        )
        valid = (offsets < NUMEL) & (raw_ids >= 0) & (raw_ids < NUM_EXPERTS)
        ids = raw_ids
        if HAS_EXPERT_MAP:
            safe_ids = tl.minimum(tl.maximum(raw_ids, 0), NUM_EXPERTS - 1)
            ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(tl.int32)
            valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)

        matches = (
            (experts[:, None] == ids[None, :]) & expert_mask[:, None] & valid[None, :]
        )
        counts += tl.sum(matches.to(tl.int32), axis=1)

    padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padded = tl.where(expert_mask, padded, 0)
    inclusive = tl.cumsum(padded, axis=0)
    prefix = inclusive - padded
    total = tl.sum(padded, axis=0)

    tl.store(expert_offsets + experts, prefix, mask=expert_mask)
    tl.store(expert_offsets + NUM_EXPERTS, total)
    tl.store(packed_route_count, total)

    route_init_offsets = tl.arange(0, BLOCK_ROUTE_INIT)
    tl.store(
        packed_route_indices + route_init_offsets,
        NUMEL,
        mask=route_init_offsets < MAX_PACKED_ROUTES,
    )

    block_offsets = tl.arange(0, BLOCK_M)
    block_rows = block_offsets * BLOCK_SIZE
    active = (
        (block_offsets[None, :] < MAX_ROUTE_BLOCKS)
        & expert_mask[:, None]
        & (block_rows[None, :] >= prefix[:, None])
        & (block_rows[None, :] < inclusive[:, None])
    )
    block_experts = tl.max(tl.where(active, experts[:, None], -1), axis=0)
    tl.store(
        block_expert_ids + block_offsets,
        block_experts,
        mask=block_offsets < MAX_ROUTE_BLOCKS,
    )


@triton.jit
def _pack_topk_routes_sort_kernel(
    topk_ids,
    expert_map,
    packed_route_indices,
    expert_offsets,
    NUMEL: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    raw_ids = tl.load(topk_ids + offsets, mask=offsets < NUMEL, other=-1).to(tl.int32)
    valid = (offsets < NUMEL) & (raw_ids >= 0) & (raw_ids < NUM_EXPERTS)
    ids = raw_ids
    if HAS_EXPERT_MAP:
        safe_ids = tl.minimum(tl.maximum(raw_ids, 0), NUM_EXPERTS - 1)
        ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(tl.int32)
        valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)

    ranks = tl.atomic_add(expert_offsets + ids, 1, sem="relaxed", mask=valid)
    tl.store(packed_route_indices + ranks, offsets, mask=valid)


def pack_topk_routes_by_expert(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    expert_map: torch.Tensor | None = None,
    packed_route_indices: torch.Tensor | None = None,
    block_expert_ids: torch.Tensor | None = None,
    packed_route_count: torch.Tensor | None = None,
    expert_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numel = int(topk_ids.numel())
    max_packed_routes = _max_packed_route_slots(
        numel, int(block_size), int(num_experts)
    )
    max_route_blocks = (max_packed_routes + int(block_size) - 1) // int(block_size)

    packed_route_indices = _workspace_slice(
        packed_route_indices,
        name="packed_route_indices",
        elements=max_packed_routes,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    block_expert_ids = _workspace_slice(
        block_expert_ids,
        name="block_expert_ids",
        elements=max_route_blocks,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    packed_route_count = _workspace_slice(
        packed_route_count,
        name="packed_route_count",
        elements=1,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_offsets = _workspace_slice(
        expert_offsets,
        name="expert_offsets",
        elements=int(num_experts) + 1,
        dtype=torch.int32,
        device=topk_ids.device,
    )

    if numel == 0:
        packed_route_indices.fill_(0)
        block_expert_ids.fill_(-1)
        packed_route_count.zero_()
        return packed_route_indices, block_expert_ids, packed_route_count

    block_e = _next_power_of_2(num_experts)
    block_route_init = _next_power_of_2(max(max_packed_routes, 1))
    block_m = _next_power_of_2(max(max_route_blocks, 1))
    sort_grid = (triton.cdiv(numel, _SORT_BLOCK_T),)
    expert_map_tensor = expert_map if expert_map is not None else topk_ids

    _pack_topk_routes_prefix_kernel[(1,)](
        topk_ids,
        expert_map_tensor,
        packed_route_indices,
        block_expert_ids,
        packed_route_count,
        expert_offsets,
        NUMEL=numel,
        BLOCK_SIZE=int(block_size),
        NUM_EXPERTS=int(num_experts),
        MAX_PACKED_ROUTES=max_packed_routes,
        MAX_ROUTE_BLOCKS=max_route_blocks,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_E=block_e,
        BLOCK_T=_COUNT_BLOCK_T,
        BLOCK_ROUTE_INIT=block_route_init,
        BLOCK_M=block_m,
        num_warps=8,
    )
    _pack_topk_routes_sort_kernel[sort_grid](
        topk_ids,
        expert_map_tensor,
        packed_route_indices,
        expert_offsets,
        NUMEL=numel,
        NUM_EXPERTS=int(num_experts),
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_T=_SORT_BLOCK_T,
        num_warps=4,
    )
    return packed_route_indices, block_expert_ids, packed_route_count


__all__ = ["pack_topk_routes_by_expert"]
