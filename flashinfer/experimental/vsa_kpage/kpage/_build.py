# ruff: noqa: SIM300
# Preserve upstream CuTe DSL constructs during the KPage extraction.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Host side of the KPage metadata build.
#
# Forward: every (head, Q block) walks 16-row K/V pages instead of 128-row
# blocks.  A KV block with vc valid rows contributes ceil(vc / 16) pages, so
# padding per block is at most 15 rows.  Pages are grouped eight per 128-row
# tile; fully valid pages come first, partial tails last, and each tile of the
# partial suffix carries four packed keep words that the softmax warps apply
# before the row max.  Q blocks with <= 64 valid rows go to the M64 x N256
# kernel, the rest to M128 x N128.  The Q and KV partitions are independent
# (q_sizes over nQ blocks, kv_sizes over nKV blocks); dense Q is q_sizes == 128.
#
# Sparse block size 64 (``build_walk_tables_blk64``, forward only) uses the same table layout with 4 pages per block and
# every Q block routed to the M64 kernel as a single tile.
#
# Backward: each CTA owns an 8-page K/V tile packed from consecutive KV
# blocks (full blocks tile-aligned, partial blocks next-fit, no block straddles
# a tile) and walks the union of the Q pages adjacent to any of those blocks.
# A 64-bit (kv page x q page) mask per step keeps only true adjacency.
#
# Tables depend on the sparse metadata only, never on Q/K/V values, so a
# layout can be reused across calls.  The forward build reads back three
# counts once; backward tables are built on first backward use.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from flashinfer.experimental.vsa_kpage.csrc.utils.kpage_utils import (
    BLOCK,
    FRAGS_PER_TILE,
    PAGE,
)

_BLOCK64 = 64
_Q64_HEAVY_ITER_THRESHOLD = 64
_Q128_HEAVY_ITER_THRESHOLD = 96

__all__ = [
    "Blk64ForwardTables",
    "PagedKVWalkTables",
    "WalkTables",
    "build_walk_tables",
    "build_walk_tables_blk64",
    "set_pad_q_stats",
    "zero_pad_forward_state",
    "zero_pad_rows",
]


# ---------------------------------------------------------------------------
# forward page walks: per (head, Q block) 16-row page lists, even tile counts
# ---------------------------------------------------------------------------


def _prepare_q_route_templates(
    q_sizes: Tensor,
    large_costs: Tensor,
    tile_offsets: Tensor,
    stream,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Enqueue per-Q route compaction and the shared capacity count."""
    n_q_blocks = q_sizes.numel()
    small_q = torch.empty(n_q_blocks, dtype=torch.int32, device=q_sizes.device)
    large_q = torch.empty_like(small_q)
    route_counts = torch.empty(3, dtype=torch.int32, device=q_sizes.device)
    from ._build_kernels import compact_q_routes

    compact_q_routes(
        q_sizes,
        tile_offsets,
        large_costs,
        small_q,
        large_q,
        route_counts,
        stream,
    )
    return small_q, large_q, route_counts


def _build_q_work_queues(
    device: torch.device,
    num_heads: int,
    large_costs: Tensor,
    small_costs: Tensor,
    small_q: Tensor,
    large_q: Tensor,
    n_small_q: int,
    n_large_q: int,
    stream,
) -> Tuple[Tensor, Tensor]:
    """Expand per-Q route templates into exact M128 and M64 queues."""
    small_work = torch.empty(num_heads * n_small_q, dtype=torch.int32, device=device)
    large_work = torch.empty(num_heads * n_large_q, dtype=torch.int32, device=device)
    from ._build_kernels import order_q_route_work

    if large_work.numel() > 0:
        order_q_route_work(
            large_q[:n_large_q],
            large_costs,
            large_work,
            heavy_threshold=_Q128_HEAVY_ITER_THRESHOLD,
            is_q64=False,
            stream=stream,
        )
    if small_work.numel() > 0:
        order_q_route_work(
            small_q[:n_small_q],
            small_costs,
            small_work,
            heavy_threshold=_Q64_HEAVY_ITER_THRESHOLD,
            is_q64=True,
            stream=stream,
        )
    return large_work, small_work


# ---------------------------------------------------------------------------
# backward walks: 8-page owned KV tiles, union Q-page walks, pair masks
# ---------------------------------------------------------------------------


@dataclass
class PagedKVWalkTables:
    """Backward walk tables."""

    tile_offsets: Tensor  # [1, H, G, nKVt] int32
    tile_counts: Tensor  # [1, H, G, nKVt] int32
    page_table: Tensor  # flat [total_steps * 8] int32 q page ids (~0 dummies)
    mask_table: Tensor  # flat [total_steps] int64 (bit 8j+i: kv page i x q page j)
    kv_page_table: Tensor  # [1, H, nKVt, 8] int32 K/V page ids (-1 dummies)
    kv_valid_table: Tensor  # [nKVt] int64, byte i = valid rows of K/V page slot i
    work_indices: Tensor  # [num_active, 2] int32 (head-group, KV tile)
    has_tail_pages: bool  # some block has a page with fewer than 16 valid rows


@dataclass
class _BwdWalkStats:
    """Device-side results of the backward stats phase; the four totals are read back by the caller."""

    num_kv_blocks: int
    num_heads: int
    num_groups: int
    config: Tensor
    q_page_counts: Tensor
    kv_page_table: Tensor
    kv_valid: Tensor
    kv_sizes: Tensor
    bitmap: Tensor
    step_counts: Tensor
    tile_offsets: Tensor
    entry_offsets: Tensor
    work_indices: Tensor
    active_count: Tensor  # 0-dim int64 (device)
    total_slots: Tensor  # 0-dim int64 (device)
    total_steps: Tensor  # 0-dim int64 (device)
    total_entries: Tensor  # 0-dim int64 (device)


def _bwd_bucket_blocks(
    num_q_blocks: int, num_heads: int, bwd_kv_groups: Optional[int]
) -> int:
    """Q blocks per backward walk group.

    Fewer groups reload each K/V tile less often; more groups shorten the
    serial Q walk of a CTA.  The group count also fixes how a tile's dK/dV
    sum is split across CTAs, so changing it moves results within fp32
    reduction-order noise.
    """
    if bwd_kv_groups is not None:
        if bwd_kv_groups < 1:
            raise ValueError(f"bwd_kv_groups must be >= 1, got {bwd_kv_groups}")
        return (num_q_blocks + bwd_kv_groups - 1) // bwd_kv_groups
    if num_q_blocks >= 4096 and num_heads <= 1:
        return 256
    if num_q_blocks >= 2048:
        # Six groups measured best for long multi-head sequences: fewer groups
        # lengthen the serial walk, more reload K/V and leave a one-CTA tail.
        return (num_q_blocks + 5) // 6
    return 384


def _bwd_paged_kv_stats(
    q2k_block_index: Tensor,  # [1, H, nQ, cap] int32
    q2k_block_nums: Tensor,  # [1, H, nQ] int32
    q_sizes: Tensor,  # [nQ] int32 valid rows per Q block
    kv_sizes: Tensor,  # [nKV] int32 valid rows per KV block
    num_heads: int,
    bwd_kv_groups: Optional[int] = None,
) -> _BwdWalkStats:
    """Stats phase of the backward tables.

    Packs K/V pages into 8-page owner tiles (full blocks first and aligned,
    partial blocks next-fit without straddling), then OR-s every q2k edge into
    a (walk row, q block) byte bitmap to get the union Q walk of each tile
    without materializing or sorting edges.  Everything is sized on device;
    the fill phase needs only the four totals read back by the caller.
    """
    num_q_blocks, num_kv_blocks = q_sizes.numel(), kv_sizes.numel()
    batch_size, metadata_heads = q2k_block_index.shape[:2]
    assert batch_size == 1 and metadata_heads == num_heads
    device = q2k_block_index.device
    blocks_per_group = _bwd_bucket_blocks(num_q_blocks, num_heads, bwd_kv_groups)
    num_groups = (num_q_blocks + blocks_per_group - 1) // blocks_per_group
    from ._build_kernels import (
        bwd_kv_bitmap_stats,
        bwd_kv_compact_work,
        bwd_kv_edge_bitmap,
        bwd_kv_nostraddle_pack,
        bwd_kv_page_table_fill,
    )

    q_page_counts = ((q_sizes + (PAGE - 1)) // PAGE).clamp(0, FRAGS_PER_TILE)
    page_counts = ((kv_sizes + (PAGE - 1)) // PAGE).clamp(0, FRAGS_PER_TILE)
    # Full blocks first (tile-aligned), then partial blocks from short to long;
    # packing similar sizes together keeps the union Q walks short.
    pack_key = page_counts & (FRAGS_PER_TILE - 1)
    order = torch.argsort(pack_key, stable=True)
    ordered_page_counts = page_counts[order].contiguous()
    ordered_start_slots = torch.empty_like(ordered_page_counts)
    bwd_kv_nostraddle_pack(ordered_page_counts, ordered_start_slots)
    start_slots = torch.empty_like(ordered_start_slots)
    start_slots[order] = ordered_start_slots
    first_tiles = start_slots // FRAGS_PER_TILE
    tile_slots = start_slots - first_tiles * FRAGS_PER_TILE

    # Allocate at the padded bound (one tile per KV block), then slice to num_kv_tiles * 8.
    kv_page_table = torch.full(
        (num_kv_blocks * FRAGS_PER_TILE,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    kv_valid = torch.zeros(
        num_kv_blocks * FRAGS_PER_TILE, dtype=torch.uint8, device=device
    )
    bwd_kv_page_table_fill(start_slots, page_counts, kv_sizes, kv_page_table, kv_valid)

    # One bitmap byte per (walk row, Q block in its group).
    words_per_bitmap_row = (blocks_per_group + 3) // 4
    num_padded_rows = num_heads * num_groups * num_kv_blocks
    config = torch.tensor(
        (
            num_q_blocks,
            num_groups,
            blocks_per_group,
            words_per_bitmap_row,
            num_kv_blocks,
        ),
        dtype=torch.int32,
        device=device,
    )
    bitmap = torch.zeros(
        num_padded_rows * words_per_bitmap_row,
        dtype=torch.int32,
        device=device,
    )
    capacity = q2k_block_index.shape[-1]
    block_index = q2k_block_index[0].reshape(num_heads * num_q_blocks, capacity)
    block_count = q2k_block_nums[0].reshape(-1).contiguous()
    bwd_kv_edge_bitmap(
        block_index,
        block_count,
        first_tiles,
        tile_slots,
        q_page_counts,
        page_counts,
        bitmap,
        config,
    )

    step_counts = torch.empty(num_padded_rows, dtype=torch.int32, device=device)
    entry_counts = torch.empty_like(step_counts)
    bwd_kv_bitmap_stats(bitmap, q_page_counts, step_counts, entry_counts, config)
    work_indices = torch.empty((num_padded_rows, 2), dtype=torch.int32, device=device)
    active_count = torch.empty(1, dtype=torch.int32, device=device)
    bwd_kv_compact_work(
        step_counts,
        config,
        work_indices,
        active_count,
    )
    tile_offsets = torch.cumsum(step_counts, 0, dtype=torch.int64) - step_counts
    entry_offsets = torch.zeros(num_padded_rows + 1, dtype=torch.int32, device=device)
    torch.cumsum(entry_counts, 0, dtype=torch.int32, out=entry_offsets[1:])
    return _BwdWalkStats(
        num_kv_blocks=num_kv_blocks,
        num_heads=num_heads,
        num_groups=num_groups,
        config=config,
        q_page_counts=q_page_counts,
        kv_page_table=kv_page_table,
        kv_valid=kv_valid,
        kv_sizes=kv_sizes,
        bitmap=bitmap,
        step_counts=step_counts,
        tile_offsets=tile_offsets,
        entry_offsets=entry_offsets,
        work_indices=work_indices,
        active_count=active_count[0].to(torch.int64),
        total_slots=(ordered_start_slots[-1] + ordered_page_counts[-1]).to(torch.int64),
        total_steps=tile_offsets[-1] + step_counts[-1],
        total_entries=entry_offsets[-1].to(torch.int64),
    )


def _bwd_paged_kv_fill(
    stats: _BwdWalkStats,
    total_slots: int,
    total_steps: int,
    total_entries: int,
    active_count: int,
    has_tail_pages: bool,
) -> PagedKVWalkTables:
    """Fill phase: allocations sized by the (already read back) totals."""
    device = stats.bitmap.device
    num_kv_blocks = stats.num_kv_blocks
    num_heads = stats.num_heads
    num_groups = stats.num_groups
    num_kv_tiles = max((total_slots + FRAGS_PER_TILE - 1) // FRAGS_PER_TILE, 1)
    total_steps = max(total_steps, 1)
    from ._build_kernels import (
        bwd_kv_bitmap_compact,
        bwd_kv_walk_fill,
    )

    q_blocks = torch.empty(max(total_entries, 1), dtype=torch.int32, device=device)
    adjacency_bytes = torch.empty_like(q_blocks)
    bwd_kv_bitmap_compact(
        stats.bitmap,
        stats.entry_offsets,
        q_blocks,
        adjacency_bytes,
        stats.config,
    )

    page_table = torch.full(
        (total_steps * FRAGS_PER_TILE,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    slot_bytes = torch.zeros(
        total_steps * FRAGS_PER_TILE, dtype=torch.uint8, device=device
    )
    tile_offsets = stats.tile_offsets.to(torch.int32)
    bwd_kv_walk_fill(
        stats.entry_offsets,
        q_blocks,
        adjacency_bytes,
        stats.q_page_counts,
        tile_offsets,
        page_table,
        slot_bytes,
    )

    return PagedKVWalkTables(
        tile_offsets=tile_offsets.view(num_heads, num_groups, num_kv_blocks)[
            :, :, :num_kv_tiles
        ]
        .contiguous()
        .view(1, num_heads, num_groups, num_kv_tiles),
        tile_counts=stats.step_counts.view(num_heads, num_groups, num_kv_blocks)[
            :, :, :num_kv_tiles
        ]
        .contiguous()
        .view(1, num_heads, num_groups, num_kv_tiles),
        page_table=page_table,
        mask_table=slot_bytes.view(torch.int64),  # byte j -> bits [8j, 8j+8)
        kv_page_table=stats.kv_page_table[: num_kv_tiles * FRAGS_PER_TILE]
        .view(1, 1, num_kv_tiles, FRAGS_PER_TILE)
        .expand(1, num_heads, num_kv_tiles, FRAGS_PER_TILE)
        .contiguous(),
        kv_valid_table=stats.kv_valid[: num_kv_tiles * FRAGS_PER_TILE].view(
            torch.int64
        ),
        work_indices=stats.work_indices[:active_count],
        has_tail_pages=has_tail_pages,
    )


@dataclass
class _BwdBuildMetadata:
    """Metadata kept until the backward tables are built."""

    q2k_block_index: Tensor
    q2k_block_nums: Tensor
    q_sizes: Tensor
    kv_sizes: Tensor
    H: int
    bwd_kv_groups: Optional[int]


@dataclass
class WalkTables:
    """Forward walk tables plus the deferred backward state."""

    page_table: Tensor  # [total_tiles, 8] int32 page ids (one row per N tile)
    tile_offsets: Tensor  # [1, H, nQ] int32 TILE offset of each walk
    tile_counts: Tensor  # [1, H, nQ] int32 EVEN tile counts
    full_tile_counts: Tensor  # [1, H, nQ] int32 unmasked N128 prefix length
    mask_table: Tensor  # [total_tiles, 4] int32 packed score keep masks
    work_indices: Tensor  # [num_active_q128] compact blk128 work queue
    bwd_kv_tables: Optional[PagedKVWalkTables]
    bwd_metadata: Optional[_BwdBuildMetadata] = None

    def resolve_bwd_tables(self) -> PagedKVWalkTables:
        """Build the backward tables on first use."""

        if self.bwd_kv_tables is None:
            metadata = self.bwd_metadata
            assert metadata is not None, "walk tables built without backward metadata"
            stats = _bwd_paged_kv_stats(
                metadata.q2k_block_index,
                metadata.q2k_block_nums,
                metadata.q_sizes,
                metadata.kv_sizes,
                metadata.H,
                metadata.bwd_kv_groups,
            )
            scalars = torch.stack(
                (
                    stats.total_slots,
                    stats.total_steps,
                    stats.total_entries,
                    stats.active_count,
                    (stats.kv_sizes % PAGE != 0).any().to(torch.int64),
                )
            ).cpu()
            total_slots, total_steps, total_entries, active_count, has_tail_pages = (
                int(x) for x in scalars
            )
            self.bwd_kv_tables = _bwd_paged_kv_fill(
                stats,
                total_slots,
                total_steps,
                total_entries,
                active_count,
                bool(has_tail_pages),
            )
            self.bwd_metadata = None
        return self.bwd_kv_tables


@dataclass
class Blk64ForwardTables:
    """Forward-only page walks for the M64 x N256 kernel."""

    page_table: Tensor  # [total_half_tiles, 8] int32 page ids
    tile_offsets: Tensor  # [1, H, 2 * nQ] int32, in 8-page half-tile units
    iter_counts: Tensor  # [1, H, 2 * nQ] int32, even N256 iterations
    full_iter_counts: Tensor  # [1, H, 2 * nQ] int32 unmasked N256 prefix
    mask_table: Tensor  # [total_half_tiles, 4] int32 packed N128 masks
    work_indices: Tensor  # [num_active_q64] int32 compact M64 work queue


def build_walk_tables(
    q2k_block_index: Tensor,  # [1, H, nQ, cap] int32
    q2k_block_nums: Tensor,  # [1, H, nQ] int32
    q_sizes: Tensor,  # [nQ] int32 valid rows per Q block
    kv_sizes: Tensor,  # [nKV] int32 valid rows per KV block; device kernels clamp on load
    H: int,
    bwd_kv_groups: Optional[int] = None,
) -> Tuple[WalkTables, Blk64ForwardTables]:
    """Forward page walks for both kernels, plus the deferred backward state.

    Both kernels read the same page stream.  A Q block with <= 64 valid rows
    rounds its tile count to a multiple of four (even N256 iterations), a
    larger one to a multiple of two (M128 kernel); each Q block lands in
    exactly one work queue.  Long walks are queued first to shorten the
    persistent-scheduler tail.  One readback of the route counts sizes the
    tables exactly.
    """

    dev = kv_sizes.device

    idx = q2k_block_index.view(-1, q2k_block_index.shape[-1])  # [H*nQ, cap] int32
    cnt = q2k_block_nums.view(-1)  # [H*nQ] int32
    assert idx.dtype == torch.int32 and cnt.dtype == torch.int32
    n_walks = idx.shape[0]

    shared_counts = torch.empty(n_walks, dtype=torch.int32, device=dev)
    from ._build_kernels import current_stream, walk_stats

    stream = current_stream()
    walk_stats(
        idx,
        cnt,
        q_sizes,
        kv_sizes,
        shared_counts,
        stream,
    )

    n_q_blocks = q_sizes.numel()
    tile_off = torch.cumsum(shared_counts, dim=0, dtype=torch.int32) - shared_counts
    q_route_state = _prepare_q_route_templates(
        q_sizes,
        shared_counts,
        tile_off,
        stream,
    )

    bwd_metadata = _BwdBuildMetadata(
        q2k_block_index=q2k_block_index,
        q2k_block_nums=q2k_block_nums,
        q_sizes=q_sizes,
        kv_sizes=kv_sizes,
        H=H,
        bwd_kv_groups=bwd_kv_groups,
    )

    n_small_q, n_large_q, total_tiles = (int(value) for value in q_route_state[2].cpu())

    # Dummy page ids are out of bounds for the TMA descriptor and load as zeros.
    nt = max(total_tiles, 1)
    page_table = torch.empty((nt, FRAGS_PER_TILE), dtype=torch.int32, device=dev)
    mask_table = torch.empty((nt, 4), dtype=torch.int32, device=dev)
    full_tile_counts = torch.empty(n_walks, dtype=torch.int32, device=dev)
    tile_offsets_q64 = torch.empty(2 * n_walks, dtype=torch.int32, device=dev)
    iter_counts_q64 = torch.empty_like(tile_offsets_q64)
    full_iter_counts_q64 = torch.empty_like(tile_offsets_q64)
    from ._build_kernels import walk_fill

    walk_fill(
        idx,
        cnt,
        q_sizes,
        kv_sizes,
        tile_off,
        shared_counts,
        full_tile_counts,
        tile_offsets_q64,
        iter_counts_q64,
        full_iter_counts_q64,
        page_table,
        mask_table,
        stream,
    )

    work_indices, work_indices_q64 = _build_q_work_queues(
        dev,
        H,
        shared_counts,
        iter_counts_q64,
        q_route_state[0],
        q_route_state[1],
        n_small_q,
        n_large_q,
        stream,
    )
    tables = WalkTables(
        page_table=page_table,
        tile_offsets=tile_off.view(1, H, n_q_blocks),
        tile_counts=shared_counts.view(1, H, n_q_blocks),
        full_tile_counts=full_tile_counts.view(1, H, n_q_blocks),
        mask_table=mask_table,
        work_indices=work_indices.contiguous(),
        bwd_kv_tables=None,
        bwd_metadata=bwd_metadata,
    )

    blk64_tables = Blk64ForwardTables(
        page_table=page_table,
        tile_offsets=tile_offsets_q64.view(1, H, 2 * n_q_blocks),
        iter_counts=iter_counts_q64.view(1, H, 2 * n_q_blocks),
        full_iter_counts=full_iter_counts_q64.view(1, H, 2 * n_q_blocks),
        mask_table=mask_table,
        work_indices=work_indices_q64.contiguous(),
    )
    return tables, blk64_tables


def build_walk_tables_blk64(
    q2k_block_index: Tensor,  # [1, H, nQ, cap] int32, sparse_block_size 64
    q2k_block_nums: Tensor,  # [1, H, nQ] int32
    kv_sizes: Tensor,  # [nKV] int32 valid rows per 64-row KV block; device kernels clamp on load
) -> Blk64ForwardTables:
    """Forward page walks for sparse_block_size=64 metadata, for the M64 x N256 kernel.

    A 64-row Q block is one M64 tile, so the tables are indexed by the Q block itself (``[1, H, nQ]``, not the two
    halves of a 128-row block) and the M128 route is never used.  Q padding is not skipped, as in the base blk64
    kernel.  One readback sizes the tables.
    """
    from ._build_kernels import current_stream, walk_fill, walk_stats

    dev = kv_sizes.device
    batch, num_heads, n_q_blocks, cap = q2k_block_index.shape
    assert batch == 1, "kpage blk64 runs a batch as independent heads; batch must be 1"
    idx = q2k_block_index.view(-1, cap)  # [H*nQ, cap]
    cnt = q2k_block_nums.view(-1)  # [H*nQ]
    assert (
        idx.dtype == torch.int32
        and cnt.dtype == torch.int32
        and kv_sizes.dtype == torch.int32
    )
    n_walks = idx.shape[0]
    q_sizes = torch.full((n_q_blocks,), _BLOCK64, dtype=torch.int32, device=dev)

    stream = current_stream()
    iter_tiles = torch.empty(n_walks, dtype=torch.int32, device=dev)
    walk_stats(idx, cnt, q_sizes, kv_sizes, iter_tiles, stream, block=_BLOCK64)
    tile_off = torch.cumsum(iter_tiles, dim=0, dtype=torch.int32) - iter_tiles
    total_tiles = int((tile_off[-1] + iter_tiles[-1]).cpu())

    # Dummy page ids are out of bounds for the TMA descriptor and load as zeros.
    nt = max(total_tiles, 1)
    page_table = torch.empty((nt, FRAGS_PER_TILE), dtype=torch.int32, device=dev)
    mask_table = torch.empty((nt, 4), dtype=torch.int32, device=dev)
    full_tile_counts = torch.empty(n_walks, dtype=torch.int32, device=dev)
    tile_offsets = torch.empty(n_walks, dtype=torch.int32, device=dev)
    iter_counts = torch.empty_like(tile_offsets)
    full_iter_counts = torch.empty_like(tile_offsets)
    walk_fill(
        idx,
        cnt,
        q_sizes,
        kv_sizes,
        tile_off,
        iter_tiles,
        full_tile_counts,
        tile_offsets,
        iter_counts,
        full_iter_counts,
        page_table,
        mask_table,
        stream,
        block=_BLOCK64,
    )

    # Every walk is queued, as on the blk128 route: an empty one (allow_empty_block_nums) takes the kernel's
    # empty-tile path and writes O = 0, LSE = -inf.  Long walks first, stable within each class.
    heavy = iter_counts > _Q64_HEAVY_ITER_THRESHOLD
    work_indices = torch.argsort(~heavy, stable=True).to(torch.int32).contiguous()
    return Blk64ForwardTables(
        page_table=page_table,
        tile_offsets=tile_offsets.view(1, num_heads, n_q_blocks),
        iter_counts=iter_counts.view(1, num_heads, n_q_blocks),
        full_iter_counts=full_iter_counts.view(1, num_heads, n_q_blocks),
        mask_table=mask_table,
        work_indices=work_indices,
    )


# ---------------------------------------------------------------------------
# padding rows
# ---------------------------------------------------------------------------


def zero_pad_forward_state(out: Tensor, lse: Tensor, q_sizes: Tensor) -> None:
    """Zero the padding rows of O and LSE (rows past the tensor end are skipped)."""
    B, H, T, D = out.shape
    assert B == 1 and lse.shape == out.shape[:-1]
    assert (T + BLOCK - 1) // BLOCK == q_sizes.numel() and D == 128
    from ._build_kernels import (
        zero_pad_forward_state as zero_forward_state_cute,
    )

    zero_forward_state_cute(out, lse, q_sizes)


def set_pad_q_stats(lse_log2: Tensor, dpsum: Tensor, q_sizes: Tensor) -> None:
    """Give padding Q rows LSE = +inf and dPsum = 0 (their P and dS then vanish in the backward kernel).

    The buffers are the backward's own, allocated at the 128-row-rounded length.
    """
    assert (
        lse_log2.shape == dpsum.shape
        and lse_log2.shape[0] == 1
        and lse_log2.shape[2] == q_sizes.numel() * BLOCK
    )
    from ._build_kernels import set_pad_q_stats as set_pad_q_stats_cute

    set_pad_q_stats_cute(lse_log2, dpsum, q_sizes)


def zero_pad_rows(
    dq: Tensor, dk: Tensor, dv: Tensor, q_sizes: Tensor, kv_sizes: Tensor
) -> None:
    """Zero rows ``>= q_sizes[i]`` of every Q block of dq and rows ``>= kv_sizes[j]`` of every KV block of dk/dv (one launch)."""
    B, H, T, D = dq.shape
    assert (
        B == 1
        and dk.shape == dv.shape
        and dk.shape[0] == 1
        and H % dk.shape[1] == 0
        and dk.shape[3] == D == 128
    )
    assert (T + BLOCK - 1) // BLOCK == q_sizes.numel() and (
        dk.shape[2] + BLOCK - 1
    ) // BLOCK == kv_sizes.numel()
    from ._build_kernels import zero_pad_rows as zero_pad_rows_cute

    zero_pad_rows_cute(dq, dk, dv, q_sizes, kv_sizes)
