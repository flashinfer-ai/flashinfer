"""Triton route-packing kernels for W4A16 MoE."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import triton
import triton.language as tl
import torch

from b12x._lib.compile_plan import launch_triton

from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity


_COUNT_BLOCK_T = 256
_SORT_BLOCK_T = 256
_POST_PREFIX_BLOCK_T = 256
_SMALL_PREFIX_MAX_PACKED_ROUTES = 4096
_SMALL_PREFIX_MAX_ROUTE_BLOCKS = 512


_FAST_COUNT_BLOCK_T = 1024


@triton.jit
def _w4a16_route_count_kernel(
    topk_ids,
    expert_map,
    counts,
    live_numel,
    NUM_EXPERTS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Parallel atomic histogram of routes per (mapped) expert.

    Same expert-id resolution as the sort kernel (expert_map aware).
    Writes ``counts[NUM_EXPERTS]``."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    raw_ids = tl.load(topk_ids + offsets, mask=offsets < live_numel, other=-1).to(
        tl.int32
    )
    valid = (offsets < live_numel) & (raw_ids >= 0) & (raw_ids < NUM_EXPERTS)
    ids = raw_ids
    if HAS_EXPERT_MAP:
        safe_ids = tl.minimum(tl.maximum(raw_ids, 0), NUM_EXPERTS - 1)
        ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(tl.int32)
        valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)
    tl.atomic_add(counts + ids, 1, sem="relaxed", mask=valid)


@triton.jit
def _w4a16_route_prefix_from_counts_kernel(
    counts,
    packed_route_count,
    expert_offsets,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Tiny over-experts block-padded prefix from precomputed counts.

    Emits the clean block-padded ``expert_offsets`` / ``packed_route_count``
    contract; the sort kernel later advances expert_offsets in place."""
    experts = tl.arange(0, BLOCK_E)
    mask = experts < NUM_EXPERTS
    counts_v = tl.load(counts + experts, mask=mask, other=0)
    padded = ((counts_v + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padded = tl.where(mask, padded, 0)
    inclusive = tl.cumsum(padded, axis=0)
    prefix = inclusive - padded
    total = tl.sum(padded, axis=0)
    tl.store(expert_offsets + experts, prefix, mask=mask)
    tl.store(expert_offsets + NUM_EXPERTS, total)
    tl.store(packed_route_count, total)


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _numel_capacity_for_route_workspace(
    packed_routes: int,
    route_blocks: int,
    block_size: int,
    num_experts: int,
) -> int:
    """Recover the largest live route count covered by a fixed workspace."""
    block_size = int(block_size)
    num_experts = int(num_experts)
    padded_slots = min(int(packed_routes), int(route_blocks) * block_size)
    fully_padded_experts = num_experts * block_size
    if padded_slots <= fully_padded_experts:
        return padded_slots // block_size
    return padded_slots - num_experts * (block_size - 1)


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
def _pack_topk_routes_post_prefix_kernel(
    packed_route_indices,
    block_expert_ids,
    expert_offsets,
    live_numel,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    MAX_PACKED_ROUTES: tl.constexpr,
    MAX_ROUTE_BLOCKS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    """Sentinel-fill the packed slots and resolve each block's owning expert.

    Must run after the prefix is stored and before the sort kernel advances
    ``expert_offsets`` in place. The rightmost expert whose prefix is <= the
    block's first row owns the block: empty experts share their successor's
    offset, so the binary search cannot land on them."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tl.store(
        packed_route_indices + offsets,
        live_numel,
        mask=offsets < MAX_PACKED_ROUTES,
    )

    total = tl.load(expert_offsets + NUM_EXPERTS)
    block_rows = offsets * BLOCK_SIZE
    valid_blocks = (offsets < MAX_ROUTE_BLOCKS) & (block_rows < total)
    low = tl.zeros((BLOCK_T,), dtype=tl.int32)
    high = tl.full((BLOCK_T,), NUM_EXPERTS, dtype=tl.int32)
    for _ in tl.static_range(0, SEARCH_STEPS):
        mid = (low + high) // 2
        mid_offset = tl.load(expert_offsets + mid, mask=valid_blocks, other=0)
        take = mid_offset <= block_rows
        low = tl.where(take, mid, low)
        high = tl.where(take, high, mid)
    block_experts = tl.where(valid_blocks, low, -1)
    tl.store(
        block_expert_ids + offsets,
        block_experts,
        mask=offsets < MAX_ROUTE_BLOCKS,
    )


@triton.jit
def _pack_topk_routes_small_prefix_kernel(
    topk_ids,
    expert_map,
    packed_route_indices,
    block_expert_ids,
    packed_route_count,
    expert_offsets,
    expert_counts,
    live_numel,
    NUMEL_CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    MAX_PACKED_ROUTES: tl.constexpr,
    MAX_ROUTE_BLOCKS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_ROUTE_INIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    COUNTS_ALIAS_PACKED: tl.constexpr,
):
    """Single-launch route packing whose cost scales with routed rows.

    Every expert-axis operation is one-dimensional: an atomic histogram of
    the routed ids into ``expert_counts``, one block-padded cumsum, and a
    per-route-block binary search of the stored prefix. Expert count only
    enters through the O(BLOCK_E) vector ops and the O(log BLOCK_E) search
    depth, so large-expert MoE decode does not regress the packing launch."""
    experts = tl.arange(0, BLOCK_E)
    expert_mask = experts < NUM_EXPERTS
    tl.store(expert_counts + experts, 0, mask=expert_mask)
    tl.debug_barrier()

    route_offsets = tl.arange(0, BLOCK_T)
    for start in tl.range(0, NUMEL_CAPACITY, BLOCK_T):
        offsets = start + route_offsets
        raw_ids = tl.load(topk_ids + offsets, mask=offsets < live_numel, other=-1).to(
            tl.int32
        )
        valid = (offsets < live_numel) & (raw_ids >= 0) & (raw_ids < NUM_EXPERTS)
        ids = raw_ids
        if HAS_EXPERT_MAP:
            safe_ids = tl.minimum(tl.maximum(raw_ids, 0), NUM_EXPERTS - 1)
            ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(tl.int32)
            valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)
        tl.atomic_add(expert_counts + ids, 1, sem="relaxed", mask=valid)
    tl.debug_barrier()

    counts = tl.load(expert_counts + experts, mask=expert_mask, other=0)
    padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padded = tl.where(expert_mask, padded, 0)
    inclusive = tl.cumsum(padded, axis=0)
    prefix = inclusive - padded
    total = tl.sum(padded, axis=0)

    tl.store(expert_offsets + experts, prefix, mask=expert_mask)
    tl.store(expert_offsets + NUM_EXPERTS, total)
    tl.store(packed_route_count, total)

    if COUNTS_ALIAS_PACKED:
        # The prefix has consumed the histogram. Make that read complete before
        # reusing the same storage for packed-route sentinels.
        tl.debug_barrier()

    route_init_offsets = tl.arange(0, BLOCK_ROUTE_INIT)
    tl.store(
        packed_route_indices + route_init_offsets,
        live_numel,
        mask=route_init_offsets < MAX_PACKED_ROUTES,
    )
    tl.debug_barrier()

    # Rightmost expert whose stored prefix is <= the block's first row. Empty
    # experts share their successor's offset, so the rightmost match is the
    # unique expert owning the block's padded segment.
    block_offsets = tl.arange(0, BLOCK_M)
    block_rows = block_offsets * BLOCK_SIZE
    valid_blocks = (block_offsets < MAX_ROUTE_BLOCKS) & (block_rows < total)
    low = tl.zeros((BLOCK_M,), dtype=tl.int32)
    high = tl.full((BLOCK_M,), NUM_EXPERTS, dtype=tl.int32)
    for _ in tl.static_range(0, SEARCH_STEPS):
        mid = (low + high) // 2
        mid_offset = tl.load(expert_offsets + mid, mask=valid_blocks, other=0)
        take = mid_offset <= block_rows
        low = tl.where(take, mid, low)
        high = tl.where(take, high, mid)
    block_experts = tl.where(valid_blocks, low, -1)
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
    live_numel,
    NUM_EXPERTS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    raw_ids = tl.load(topk_ids + offsets, mask=offsets < live_numel, other=-1).to(
        tl.int32
    )
    valid = (offsets < live_numel) & (raw_ids >= 0) & (raw_ids < NUM_EXPERTS)
    ids = raw_ids
    if HAS_EXPERT_MAP:
        safe_ids = tl.minimum(tl.maximum(raw_ids, 0), NUM_EXPERTS - 1)
        ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(tl.int32)
        valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)

    ranks = tl.atomic_add(expert_offsets + ids, 1, sem="relaxed", mask=valid)
    tl.store(packed_route_indices + ranks, offsets, mask=valid)


@dataclass(frozen=True)
class W4A16RoutePackLaunches:
    """Exact prepared Triton objects for one W4A16 packed-route geometry."""

    numel_capacity: int
    block_size: int
    num_experts: int
    max_packed_routes: int
    max_route_blocks: int
    use_small_prefix: bool
    programs: MappingProxyType

    def program(self, name: str, topk_ids: torch.Tensor, has_expert_map: bool):
        try:
            return self.programs[(name, topk_ids.dtype, bool(has_expert_map))]
        except KeyError:
            raise RuntimeError(
                "prepared W4A16 route programs are missing the runtime "
                f"{name}/{topk_ids.dtype}/mapped={bool(has_expert_map)} variant"
            ) from None

    def carriers(self) -> tuple[object, ...]:
        return tuple(self.programs.values())


def compile_w4a16_route_pack_launches(
    *,
    tokens: int,
    topk: int,
    block_size: int,
    num_experts: int,
    ordinal: int,
    bucket_tokens: bool = True,
) -> W4A16RoutePackLaunches:
    """Compile every legal route-pack helper for one planned W4A16 capacity."""
    numel_capacity, max_packed_routes, max_route_blocks = route_pack_capacity(
        int(tokens) * int(topk),
        int(block_size),
        int(num_experts),
        topk=int(topk),
        bucket_tokens=bucket_tokens,
    )
    max_packed_routes = max(int(max_packed_routes), 1)
    max_route_blocks = max(int(max_route_blocks), 1)
    block_e = _next_power_of_2(int(num_experts))
    block_route_init = _next_power_of_2(max_packed_routes)
    block_m = _next_power_of_2(max_route_blocks)
    use_small_prefix = (
        block_route_init <= _SMALL_PREFIX_MAX_PACKED_ROUTES
        and block_m <= _SMALL_PREFIX_MAX_ROUTE_BLOCKS
    )
    programs = {}
    # Avoid Triton's divisibility and value-one specialization of the live bound.
    live_numel = 3
    with torch.cuda.device(int(ordinal)):
        for ids_dtype in (torch.int32, torch.int64):
            for has_expert_map in (False, True):
                key = (ids_dtype, has_expert_map)
                expert_map_dtype = torch.int32 if has_expert_map else ids_dtype
                if use_small_prefix:
                    programs[("small_prefix", *key)] = (
                        _pack_topk_routes_small_prefix_kernel.warmup(
                            ids_dtype,
                            expert_map_dtype,
                            torch.int32,
                            torch.int32,
                            torch.int32,
                            torch.int32,
                            torch.int32,
                            live_numel,
                            NUMEL_CAPACITY=int(numel_capacity),
                            BLOCK_SIZE=int(block_size),
                            NUM_EXPERTS=int(num_experts),
                            MAX_PACKED_ROUTES=max_packed_routes,
                            MAX_ROUTE_BLOCKS=max_route_blocks,
                            HAS_EXPERT_MAP=has_expert_map,
                            BLOCK_E=block_e,
                            BLOCK_T=_COUNT_BLOCK_T,
                            BLOCK_ROUTE_INIT=block_route_init,
                            BLOCK_M=block_m,
                            SEARCH_STEPS=block_e.bit_length(),
                            COUNTS_ALIAS_PACKED=False,
                            num_warps=8,
                            grid=(1, 1, 1),
                        )
                    )
                if not use_small_prefix:
                    programs[("count", *key)] = _w4a16_route_count_kernel.warmup(
                        ids_dtype,
                        expert_map_dtype,
                        torch.int32,
                        live_numel,
                        NUM_EXPERTS=int(num_experts),
                        HAS_EXPERT_MAP=has_expert_map,
                        BLOCK_T=_FAST_COUNT_BLOCK_T,
                        num_warps=4,
                        grid=(
                            triton.cdiv(int(tokens) * int(topk), _FAST_COUNT_BLOCK_T),
                            1,
                            1,
                        ),
                    )
                programs[("sort", *key)] = _pack_topk_routes_sort_kernel.warmup(
                    ids_dtype,
                    expert_map_dtype,
                    torch.int32,
                    torch.int32,
                    live_numel,
                    NUM_EXPERTS=int(num_experts),
                    HAS_EXPERT_MAP=has_expert_map,
                    BLOCK_T=_SORT_BLOCK_T,
                    num_warps=4,
                    grid=(triton.cdiv(int(tokens) * int(topk), _SORT_BLOCK_T), 1, 1),
                )
        if not use_small_prefix:
            programs[("prefix", torch.int32, False)] = (
                _w4a16_route_prefix_from_counts_kernel.warmup(
                    torch.int32,
                    torch.int32,
                    torch.int32,
                    BLOCK_SIZE=int(block_size),
                    NUM_EXPERTS=int(num_experts),
                    BLOCK_E=block_e,
                    num_warps=4,
                    grid=(1, 1, 1),
                )
            )
            programs[("post_prefix", torch.int32, False)] = (
                _pack_topk_routes_post_prefix_kernel.warmup(
                    torch.int32,
                    torch.int32,
                    torch.int32,
                    live_numel,
                    BLOCK_SIZE=int(block_size),
                    NUM_EXPERTS=int(num_experts),
                    MAX_PACKED_ROUTES=max_packed_routes,
                    MAX_ROUTE_BLOCKS=max_route_blocks,
                    BLOCK_T=_POST_PREFIX_BLOCK_T,
                    SEARCH_STEPS=block_e.bit_length(),
                    num_warps=4,
                    grid=(
                        max(
                            triton.cdiv(max_packed_routes, _POST_PREFIX_BLOCK_T),
                            triton.cdiv(max_route_blocks, _POST_PREFIX_BLOCK_T),
                        ),
                        1,
                        1,
                    ),
                )
            )
    return W4A16RoutePackLaunches(
        numel_capacity=int(numel_capacity),
        block_size=int(block_size),
        num_experts=int(num_experts),
        max_packed_routes=max_packed_routes,
        max_route_blocks=max_route_blocks,
        use_small_prefix=use_small_prefix,
        programs=MappingProxyType(programs),
    )


def _launch_prepared(program: object, grid: tuple[int, ...], *args: object) -> object:
    # Compiled Triton launchers require the full signature, including constexprs.
    return program[tuple((*grid, 1, 1)[:3])](*args)


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
    expert_counts: torch.Tensor | None = None,
    launches: W4A16RoutePackLaunches | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numel = int(topk_ids.numel())
    topk = int(topk_ids.shape[-1]) if topk_ids.ndim >= 2 else 1
    numel_capacity, capacity_packed_routes, capacity_route_blocks = route_pack_capacity(
        numel,
        int(block_size),
        int(num_experts),
        topk=topk,
    )
    provided_routes = (
        None if packed_route_indices is None else int(packed_route_indices.numel())
    )
    provided_blocks = (
        None if block_expert_ids is None else int(block_expert_ids.numel())
    )
    if (
        provided_routes is not None
        and provided_blocks is not None
        and (
            provided_routes < capacity_packed_routes
            or provided_blocks < capacity_route_blocks
        )
    ):
        (
            exact_numel_capacity,
            exact_packed_routes,
            exact_route_blocks,
        ) = route_pack_capacity(
            numel,
            int(block_size),
            int(num_experts),
            topk=topk,
            bucket_tokens=False,
        )
        if (
            provided_routes >= exact_packed_routes
            and provided_blocks >= exact_route_blocks
        ):
            # A serving caller can own one fixed arena sized for its configured
            # maximum while a live prefill tail belongs to a larger power-of-two
            # bucket. Reuse the full caller capacity instead of specializing each
            # exact tail. The small-prefix and post-prefix kernels fill unused
            # route slots with ``live_numel`` and unused blocks with ``-1``. The
            # recovered live-route capacity also keeps the small-prefix loop bound
            # stable without changing the caller's allocation or route semantics.
            numel_capacity = _numel_capacity_for_route_workspace(
                provided_routes,
                provided_blocks,
                int(block_size),
                int(num_experts),
            )
            capacity_packed_routes = provided_routes
            capacity_route_blocks = provided_blocks
        else:
            numel_capacity = exact_numel_capacity
            capacity_packed_routes = exact_packed_routes
            capacity_route_blocks = exact_route_blocks
    max_packed_routes = capacity_packed_routes
    max_route_blocks = capacity_route_blocks
    max_packed_routes = max(max_packed_routes, 1)
    max_route_blocks = max(max_route_blocks, 1)
    if launches is not None:
        if (
            numel > launches.numel_capacity
            or int(block_size) != launches.block_size
            or int(num_experts) != launches.num_experts
        ):
            raise RuntimeError(
                "W4A16 live routes exceed or differ from the prepared geometry: "
                f"requested={(numel, block_size, num_experts)}, "
                f"prepared={(launches.numel_capacity, launches.block_size, launches.num_experts)}"
            )
        numel_capacity = launches.numel_capacity
        max_packed_routes = launches.max_packed_routes
        max_route_blocks = launches.max_route_blocks

    provided_routes = (
        None if packed_route_indices is None else int(packed_route_indices.numel())
    )
    provided_blocks = (
        None if block_expert_ids is None else int(block_expert_ids.numel())
    )
    if (
        provided_routes is not None
        and provided_blocks is not None
        and (provided_routes < max_packed_routes or provided_blocks < max_route_blocks)
    ):
        raise ValueError(
            "W4A16 route-packing workspace is too small: "
            f"topk_shape={tuple(topk_ids.shape)}, live_routes={numel}, "
            f"capacity_routes={numel_capacity}, block_size={int(block_size)}, "
            f"num_experts={int(num_experts)}, "
            f"packed_route_indices={provided_routes}/{max_packed_routes}, "
            f"block_expert_ids={provided_blocks}/{max_route_blocks}"
        )

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
    sort_grid = (triton.cdiv(numel, _SORT_BLOCK_T),)
    expert_map_tensor = expert_map if expert_map is not None else topk_ids

    block_route_init = _next_power_of_2(max(max_packed_routes, 1))
    block_m = _next_power_of_2(max(max_route_blocks, 1))
    use_small_prefix = (
        block_route_init <= _SMALL_PREFIX_MAX_PACKED_ROUTES
        and block_m <= _SMALL_PREFIX_MAX_ROUTE_BLOCKS
    )
    if launches is not None and launches.use_small_prefix != use_small_prefix:
        raise RuntimeError(
            "prepared W4A16 route-pack path does not match bound geometry"
        )
    if use_small_prefix:
        # Prepared small-prefix objects fix COUNTS_ALIAS_PACKED=False so their
        # storage ABI is stable. Serving scratch already owns expert_counts.
        counts_alias_packed = launches is None and (
            expert_counts is None
            and int(packed_route_indices.numel()) >= int(num_experts)
        )
        if counts_alias_packed:
            expert_counts = packed_route_indices[: int(num_experts)]
        else:
            expert_counts = _workspace_slice(
                expert_counts,
                name="expert_counts",
                elements=int(num_experts),
                dtype=torch.int32,
                device=topk_ids.device,
            )
        if launches is None:
            launch_triton(
                _pack_topk_routes_small_prefix_kernel,
                (1,),
                topk_ids,
                expert_map_tensor,
                packed_route_indices,
                block_expert_ids,
                packed_route_count,
                expert_offsets,
                expert_counts,
                numel,
                NUMEL_CAPACITY=numel_capacity,
                BLOCK_SIZE=int(block_size),
                NUM_EXPERTS=int(num_experts),
                MAX_PACKED_ROUTES=max_packed_routes,
                MAX_ROUTE_BLOCKS=max_route_blocks,
                HAS_EXPERT_MAP=expert_map is not None,
                BLOCK_E=block_e,
                BLOCK_T=_COUNT_BLOCK_T,
                BLOCK_ROUTE_INIT=block_route_init,
                BLOCK_M=block_m,
                SEARCH_STEPS=block_e.bit_length(),
                COUNTS_ALIAS_PACKED=counts_alias_packed,
                num_warps=8,
            )
        else:
            _launch_prepared(
                launches.program("small_prefix", topk_ids, expert_map is not None),
                (1,),
                topk_ids,
                expert_map_tensor,
                packed_route_indices,
                block_expert_ids,
                packed_route_count,
                expert_offsets,
                expert_counts,
                numel,
                numel_capacity,
                int(block_size),
                int(num_experts),
                max_packed_routes,
                max_route_blocks,
                expert_map is not None,
                block_e,
                _COUNT_BLOCK_T,
                block_route_init,
                block_m,
                block_e.bit_length(),
                counts_alias_packed,
            )
    else:
        # Parallel path for shapes above the single-launch caps: atomic
        # histogram over routed rows, one tiny 1-D prefix, and binary-search
        # block resolution. No stage scales with expert count beyond O(E)
        # vector work, so large-expert models keep prefill-scale packing
        # cheap. Capture-safe when the caller provides the routing
        # workspaces; ``_workspace_slice`` rejects capture without them.
        expert_counts = _workspace_slice(
            expert_counts,
            name="expert_counts",
            elements=int(num_experts),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        expert_counts.zero_()
        if launches is None:
            launch_triton(
                _w4a16_route_count_kernel,
                (triton.cdiv(numel, _FAST_COUNT_BLOCK_T),),
                topk_ids,
                expert_map_tensor,
                expert_counts,
                numel,
                NUM_EXPERTS=int(num_experts),
                HAS_EXPERT_MAP=expert_map is not None,
                BLOCK_T=_FAST_COUNT_BLOCK_T,
                num_warps=4,
            )
            launch_triton(
                _w4a16_route_prefix_from_counts_kernel,
                (1,),
                expert_counts,
                packed_route_count,
                expert_offsets,
                BLOCK_SIZE=int(block_size),
                NUM_EXPERTS=int(num_experts),
                BLOCK_E=block_e,
                num_warps=4,
            )
        else:
            _launch_prepared(
                launches.program("count", topk_ids, expert_map is not None),
                (triton.cdiv(numel, _FAST_COUNT_BLOCK_T),),
                topk_ids,
                expert_map_tensor,
                expert_counts,
                numel,
                int(num_experts),
                expert_map is not None,
                _FAST_COUNT_BLOCK_T,
            )
            _launch_prepared(
                launches.programs[("prefix", torch.int32, False)],
                (1,),
                expert_counts,
                packed_route_count,
                expert_offsets,
                int(block_size),
                int(num_experts),
                block_e,
            )
        post_prefix_grid = (
            max(
                triton.cdiv(max_packed_routes, _POST_PREFIX_BLOCK_T),
                triton.cdiv(max_route_blocks, _POST_PREFIX_BLOCK_T),
            ),
        )
        if launches is None:
            launch_triton(
                _pack_topk_routes_post_prefix_kernel,
                post_prefix_grid,
                packed_route_indices,
                block_expert_ids,
                expert_offsets,
                numel,
                BLOCK_SIZE=int(block_size),
                NUM_EXPERTS=int(num_experts),
                MAX_PACKED_ROUTES=max_packed_routes,
                MAX_ROUTE_BLOCKS=max_route_blocks,
                BLOCK_T=_POST_PREFIX_BLOCK_T,
                SEARCH_STEPS=block_e.bit_length(),
                num_warps=4,
            )
        else:
            _launch_prepared(
                launches.programs[("post_prefix", torch.int32, False)],
                post_prefix_grid,
                packed_route_indices,
                block_expert_ids,
                expert_offsets,
                numel,
                int(block_size),
                int(num_experts),
                max_packed_routes,
                max_route_blocks,
                _POST_PREFIX_BLOCK_T,
                block_e.bit_length(),
            )
    if launches is None:
        launch_triton(
            _pack_topk_routes_sort_kernel,
            sort_grid,
            topk_ids,
            expert_map_tensor,
            packed_route_indices,
            expert_offsets,
            numel,
            NUM_EXPERTS=int(num_experts),
            HAS_EXPERT_MAP=expert_map is not None,
            BLOCK_T=_SORT_BLOCK_T,
            num_warps=4,
        )
    else:
        _launch_prepared(
            launches.program("sort", topk_ids, expert_map is not None),
            sort_grid,
            topk_ids,
            expert_map_tensor,
            packed_route_indices,
            expert_offsets,
            numel,
            int(num_experts),
            expert_map is not None,
            _SORT_BLOCK_T,
        )
    return packed_route_indices, block_expert_ids, packed_route_count


__all__ = [
    "W4A16RoutePackLaunches",
    "compile_w4a16_route_pack_launches",
    "pack_topk_routes_by_expert",
]
