# mypy: disable-error-code="attr-defined"
# Upstream CuTe launchers attach compile caches and mixin methods dynamically.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CuTe DSL kernels that build the KPage walk tables and clear padding rows.  The forward walk kernels take the sparse
# block size of the metadata (``block`` = 128 or 64); page ids are always ``block // PAGE`` consecutive Page16 ids per
# block.

from typing import Callable, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint8
import cuda.bindings.driver as cuda
import torch
from torch import Tensor

from flashinfer.experimental.vsa_kpage._cutlass_compat import SmemAllocator
from flashinfer.experimental.vsa_kpage.csrc.utils import copy_utils
from flashinfer.experimental.vsa_kpage.csrc.utils.cute_dsl_utils import (
    assume_tensor_aligned,
)
from flashinfer.experimental.vsa_kpage.csrc.utils.kernel_utils import warp_reduce
from flashinfer.experimental.vsa_kpage.csrc.utils.kpage_utils import (
    BLOCK,
    FRAGS_PER_TILE,
    HEAD_DIM,
    PAGE,
)
from flashinfer.experimental.vsa_kpage.kpage._compile import (
    compile_key as make_compile_key,
    to_cute_tensor,
)


class WalkStats:
    """Count the Page16 rows required by each forward walk."""

    def __init__(self, block: int = BLOCK):
        self.block = block
        self.num_threads = 256
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mQSizes: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mSharedCount: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_pages: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(
            mBlockIndex,
            mBlockCount,
            mQSizes,
            mBlockSizes,
            mSharedCount,
            SharedStorage,
        ).launch(
            grid=(mBlockIndex.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mQSizes: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mSharedCount: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        walk_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cap = mBlockIndex.shape[1]
        n_q_blocks = mQSizes.shape[0]
        n_blocks = mBlockSizes.shape[0]
        edge_count = max(min(mBlockCount[walk_idx], cap), 0)

        page_count = Int32(0)
        edge_idx = thread_idx
        while edge_idx < edge_count:
            block_idx = max(min(mBlockIndex[walk_idx, edge_idx], n_blocks - 1), 0)
            block_rows = max(min(mBlockSizes[block_idx], self.block), 0)
            page_count += (block_rows + PAGE - 1) // PAGE
            edge_idx += self.num_threads

        page_count = warp_reduce(page_count, lambda x, y: x + y)

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_pages = storage.warp_pages.get_tensor(cute.make_layout(self.num_warps))
        if lane_idx == 0:
            warp_pages[warp_idx] = page_count
        cute.arch.sync_threads()

        if warp_idx == 0:
            page_count = warp_pages[lane_idx] if lane_idx < self.num_warps else Int32(0)
            page_count = warp_reduce(page_count, lambda x, y: x + y)
            if lane_idx == 0:
                q_size = max(min(mQSizes[walk_idx % n_q_blocks], self.block), 0)
                shared_count = Int32(0)
                if q_size > 0:
                    if q_size <= 64:
                        num_iters = (page_count + 15) // 16
                        num_iters += num_iters & 1
                        shared_count = num_iters * 2
                    else:
                        shared_count = (page_count + 7) // 8
                        shared_count += shared_count & 1
                mSharedCount[walk_idx] = shared_count


class CompactQRoutes:
    """Build stable Q64/Q128 templates and the required tile count."""

    def __init__(self):
        self.num_threads = 1024
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mQSizes: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mTileCounts: cute.Tensor,
        mSmallQ: cute.Tensor,
        mLargeQ: cute.Tensor,
        mRouteCounts: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_small: cute.struct.MemRange[Int32, self.num_warps]
            warp_large: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(
            mQSizes,
            mTileOffsets,
            mTileCounts,
            mSmallQ,
            mLargeQ,
            mRouteCounts,
            SharedStorage,
        ).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQSizes: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mTileCounts: cute.Tensor,
        mSmallQ: cute.Tensor,
        mLargeQ: cute.Tensor,
        mRouteCounts: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n_q_blocks = mQSizes.shape[0]

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_small = storage.warp_small.get_tensor(cute.make_layout(self.num_warps))
        warp_large = storage.warp_large.get_tensor(cute.make_layout(self.num_warps))

        small_base = Int32(0)
        large_base = Int32(0)
        num_chunks = (n_q_blocks + self.num_threads - 1) // self.num_threads
        for chunk in cutlass.range(num_chunks):
            q_block = chunk * self.num_threads + thread_idx
            is_small = Int32(0)
            is_large = Int32(0)
            if q_block < n_q_blocks:
                q_size = max(min(mQSizes[q_block], BLOCK), 0)
                if q_size > 0:
                    if q_size <= 64:
                        is_small = Int32(1)
                    else:
                        is_large = Int32(1)

            small_rank = is_small
            large_rank = is_large
            for delta in (1, 2, 4, 8, 16):
                other_small = cute.arch.shuffle_sync_up(
                    small_rank, delta, mask_and_clamp=0
                )
                other_large = cute.arch.shuffle_sync_up(
                    large_rank, delta, mask_and_clamp=0
                )
                if lane_idx >= delta:
                    small_rank += other_small
                    large_rank += other_large

            if lane_idx == 31:
                warp_small[warp_idx] = small_rank
                warp_large[warp_idx] = large_rank
            cute.arch.sync_threads()

            small_rank += small_base
            large_rank += large_base
            for idx in cutlass.range(warp_idx):
                small_rank += warp_small[idx]
                large_rank += warp_large[idx]

            if is_small != 0:
                mSmallQ[small_rank - 1] = q_block
            if is_large != 0:
                mLargeQ[large_rank - 1] = q_block

            for idx in cutlass.range(self.num_warps):
                small_base += warp_small[idx]
                large_base += warp_large[idx]
            cute.arch.sync_threads()

        if thread_idx == 0:
            mRouteCounts[0] = small_base
            mRouteCounts[1] = large_base
            last_walk = mTileCounts.shape[0] - 1
            mRouteCounts[2] = mTileOffsets[last_walk] + mTileCounts[last_walk]


class WalkFill:
    """Fill Page16 walks, packed masks, and both M128/M64 metadata views.

    With ``block == 64`` every Q block is an M64 tile: the M64 views are indexed
    by the Q block itself (no second half) and the M128 view is unused.
    """

    def __init__(self, block: int = BLOCK):
        self.block = block
        self.pages_per_block = block // PAGE
        self.num_threads = 256
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mQSizes: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mTileCounts: cute.Tensor,
        mFullTiles: cute.Tensor,
        mQ64Offsets: cute.Tensor,
        mQ64IterCounts: cute.Tensor,
        mQ64FullIters: cute.Tensor,
        mPageTable: cute.Tensor,
        mMaskTable: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_pages: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(
            mBlockIndex,
            mBlockCount,
            mQSizes,
            mBlockSizes,
            mTileOffsets,
            mTileCounts,
            mFullTiles,
            mQ64Offsets,
            mQ64IterCounts,
            mQ64FullIters,
            mPageTable,
            mMaskTable,
            SharedStorage,
        ).launch(
            grid=(mBlockIndex.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mQSizes: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mTileCounts: cute.Tensor,
        mFullTiles: cute.Tensor,
        mQ64Offsets: cute.Tensor,
        mQ64IterCounts: cute.Tensor,
        mQ64FullIters: cute.Tensor,
        mPageTable: cute.Tensor,
        mMaskTable: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        walk_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cap = mBlockIndex.shape[1]
        n_q_blocks = mQSizes.shape[0]
        n_blocks = mBlockSizes.shape[0]
        edge_count = max(min(mBlockCount[walk_idx], cap), 0)
        q_size = max(min(mQSizes[walk_idx % n_q_blocks], self.block), 0)
        is_small = Int32(0)
        if q_size > 0:
            if q_size <= 64:
                is_small = Int32(1)
        requested_tiles = mTileCounts[walk_idx]
        tile_capacity = mPageTable.shape[0]
        offset_tiles = min(mTileOffsets[walk_idx], tile_capacity)
        room = max(tile_capacity - offset_tiles, 0)
        if is_small != 0:
            room = room & -4
        else:
            room = room & -2
        allocated_tiles = min(requested_tiles, room)
        allocated_pages = allocated_tiles * 8
        output_page_base = offset_tiles * 8

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_pages = storage.warp_pages.get_tensor(cute.make_layout(self.num_warps))

        # Keep every fully valid Page16 in one prefix. Partial Page16 tails
        # follow in stable edge order, then dummy pages fill the tile-aligned
        # suffix. This gives the attention kernel a contiguous unmasked phase.
        page_base = Int32(0)
        num_chunks = (edge_count + self.num_threads - 1) // self.num_threads
        for chunk in cutlass.range(num_chunks):
            edge_idx = chunk * self.num_threads + thread_idx
            page_count = Int32(0)
            block_idx = Int32(0)
            if edge_idx < edge_count:
                block_idx = max(min(mBlockIndex[walk_idx, edge_idx], n_blocks - 1), 0)
                block_rows = max(min(mBlockSizes[block_idx], self.block), 0)
                page_count = block_rows // PAGE

            page_rank = page_count
            for delta in (1, 2, 4, 8, 16):
                other = cute.arch.shuffle_sync_up(page_rank, delta, mask_and_clamp=0)
                if lane_idx >= delta:
                    page_rank += other

            if lane_idx == 31:
                warp_pages[warp_idx] = page_rank
            cute.arch.sync_threads()

            page_rank += page_base
            for idx in cutlass.range(warp_idx):
                page_rank += warp_pages[idx]
            page_pos = page_rank - page_count
            first_page = block_idx * self.pages_per_block
            for page_idx in cutlass.range_constexpr(self.pages_per_block):
                if page_count > page_idx and page_pos + page_idx < allocated_pages:
                    flat_slot = output_page_base + page_pos + page_idx
                    if flat_slot < mPageTable.shape[0] * 8:
                        mPageTable[flat_slot // 8, flat_slot % 8] = (
                            first_page + page_idx
                        )

            for idx in cutlass.range(self.num_warps):
                page_base += warp_pages[idx]
            cute.arch.sync_threads()

        full_pages = page_base
        for chunk in cutlass.range(num_chunks):
            edge_idx = chunk * self.num_threads + thread_idx
            page_count = Int32(0)
            block_idx = Int32(0)
            tail_page_idx = Int32(0)
            if edge_idx < edge_count:
                block_idx = max(min(mBlockIndex[walk_idx, edge_idx], n_blocks - 1), 0)
                block_rows = max(min(mBlockSizes[block_idx], self.block), 0)
                tail_page_idx = block_rows // PAGE
                if block_rows - tail_page_idx * PAGE > 0:
                    page_count = Int32(1)

            page_rank = page_count
            for delta in (1, 2, 4, 8, 16):
                other = cute.arch.shuffle_sync_up(page_rank, delta, mask_and_clamp=0)
                if lane_idx >= delta:
                    page_rank += other

            if lane_idx == 31:
                warp_pages[warp_idx] = page_rank
            cute.arch.sync_threads()

            page_rank += page_base
            for idx in cutlass.range(warp_idx):
                page_rank += warp_pages[idx]
            page_pos = page_rank - page_count
            if page_count > 0 and page_pos < allocated_pages:
                flat_slot = output_page_base + page_pos
                if flat_slot < mPageTable.shape[0] * 8:
                    mPageTable[flat_slot // 8, flat_slot % 8] = (
                        block_idx * self.pages_per_block + tail_page_idx
                    )

            for idx in cutlass.range(self.num_warps):
                page_base += warp_pages[idx]
            cute.arch.sync_threads()

        dummy_page = page_base + thread_idx
        zero_page = n_blocks * self.pages_per_block
        while dummy_page < allocated_pages:
            flat_slot = output_page_base + dummy_page
            mPageTable[flat_slot // 8, flat_slot % 8] = zero_page + flat_slot % 8
            dummy_page += self.num_threads
        cute.arch.sync_threads()

        valid_full_pages = min(full_pages, allocated_pages)
        first_mask_row = valid_full_pages // FRAGS_PER_TILE
        if is_small != 0:
            first_mask_row = (valid_full_pages // (2 * FRAGS_PER_TILE)) * 2

        # One int32 word covers two adjacent Page16 entries. Only rows in a
        # partial N128/N256 suffix are read by the forward kernels.
        mask_row = first_mask_row + thread_idx
        while mask_row < allocated_tiles:
            global_row = offset_tiles + mask_row
            for word_idx in cutlass.range_constexpr(4):
                word_mask = Int32(0)
                for half in cutlass.range_constexpr(2):
                    page_id = mPageTable[global_row, word_idx * 2 + half]
                    valid_rows = Int32(0)
                    if page_id < zero_page:
                        block_idx = page_id // self.pages_per_block
                        page_idx = page_id - block_idx * self.pages_per_block
                        block_rows = max(min(mBlockSizes[block_idx], self.block), 0)
                        valid_rows = max(min(block_rows - page_idx * PAGE, PAGE), 0)
                    page_mask = (Int32(1) << valid_rows) - 1
                    word_mask = word_mask | (page_mask << (half * PAGE))
                mMaskTable[global_row, word_idx] = word_mask
            mask_row += self.num_threads

        if thread_idx == 0:
            mTileOffsets[walk_idx] = offset_tiles
            mTileCounts[walk_idx] = allocated_tiles
            mFullTiles[walk_idx] = valid_full_pages // FRAGS_PER_TILE
            q64_idx = walk_idx if cutlass.const_expr(self.block == 64) else 2 * walk_idx
            mQ64Offsets[q64_idx] = offset_tiles
            mQ64IterCounts[q64_idx] = allocated_tiles // 2 if is_small != 0 else 0
            mQ64FullIters[q64_idx] = (
                valid_full_pages // (2 * FRAGS_PER_TILE) if is_small != 0 else 0
            )
            if cutlass.const_expr(self.block != 64):
                mQ64Offsets[q64_idx + 1] = 0
                mQ64IterCounts[q64_idx + 1] = 0
                mQ64FullIters[q64_idx + 1] = 0


class OrderQRouteWork:
    """Expand one Q route and stable-partition long walks first."""

    def __init__(self, heavy_threshold: int, is_q64: bool):
        self.heavy_threshold = heavy_threshold
        self.is_q64 = is_q64
        self.num_threads = 1024
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mQRoute: cute.Tensor,
        mCost: cute.Tensor,
        mWork: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_count: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(mQRoute, mCost, mWork, SharedStorage).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRoute: cute.Tensor,
        mCost: cute.Tensor,
        mWork: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n_route_q = mQRoute.shape[0]
        n_work = mWork.shape[0]
        num_heads = n_work // n_route_q
        n_q_blocks = mCost.shape[0] // num_heads
        if cutlass.const_expr(self.is_q64):
            n_q_blocks = n_q_blocks // 2

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_count = storage.warp_count.get_tensor(cute.make_layout(self.num_warps))

        local_heavy = Int32(0)
        work_idx = thread_idx
        while work_idx < n_work:
            head_idx = work_idx // n_route_q
            rank = work_idx - head_idx * n_route_q
            q_rank = rank
            if cutlass.const_expr(not self.is_q64):
                q_rank = n_route_q - 1 - rank
            q_block = mQRoute[q_rank]
            if cutlass.const_expr(self.is_q64):
                work = head_idx * (2 * n_q_blocks) + 2 * q_block
            else:
                work = head_idx * n_q_blocks + q_block
            if mCost[work] > self.heavy_threshold:
                local_heavy += 1
            work_idx += self.num_threads

        local_heavy = warp_reduce(local_heavy, lambda x, y: x + y)
        if lane_idx == 0:
            warp_count[warp_idx] = local_heavy
        cute.arch.sync_threads()
        heavy_total = Int32(0)
        for idx in cutlass.range_constexpr(self.num_warps):
            heavy_total += warp_count[idx]
        cute.arch.sync_threads()

        heavy_base = Int32(0)
        normal_base = heavy_total
        num_chunks = (n_work + self.num_threads - 1) // self.num_threads
        for chunk in cutlass.range(num_chunks):
            work_idx = chunk * self.num_threads + thread_idx
            work = Int32(0)
            is_heavy = Int32(0)
            if work_idx < n_work:
                head_idx = work_idx // n_route_q
                rank = work_idx - head_idx * n_route_q
                q_rank = rank
                if cutlass.const_expr(not self.is_q64):
                    q_rank = n_route_q - 1 - rank
                q_block = mQRoute[q_rank]
                if cutlass.const_expr(self.is_q64):
                    work = head_idx * (2 * n_q_blocks) + 2 * q_block
                else:
                    work = head_idx * n_q_blocks + q_block
                if mCost[work] > self.heavy_threshold:
                    is_heavy = Int32(1)
            is_normal = Int32(work_idx < n_work) - is_heavy

            heavy_rank = is_heavy
            normal_rank = is_normal
            for delta in (1, 2, 4, 8, 16):
                other_heavy = cute.arch.shuffle_sync_up(
                    heavy_rank, delta, mask_and_clamp=0
                )
                other_normal = cute.arch.shuffle_sync_up(
                    normal_rank, delta, mask_and_clamp=0
                )
                if lane_idx >= delta:
                    heavy_rank += other_heavy
                    normal_rank += other_normal

            if lane_idx == 31:
                warp_count[warp_idx] = heavy_rank
            cute.arch.sync_threads()
            heavy_rank += heavy_base
            for idx in cutlass.range(warp_idx):
                heavy_rank += warp_count[idx]
            chunk_heavy = Int32(0)
            for idx in cutlass.range_constexpr(self.num_warps):
                chunk_heavy += warp_count[idx]
            cute.arch.sync_threads()

            if lane_idx == 31:
                warp_count[warp_idx] = normal_rank
            cute.arch.sync_threads()
            normal_rank += normal_base
            for idx in cutlass.range(warp_idx):
                normal_rank += warp_count[idx]
            chunk_normal = Int32(0)
            for idx in cutlass.range_constexpr(self.num_warps):
                chunk_normal += warp_count[idx]
            cute.arch.sync_threads()

            if is_heavy != 0:
                mWork[heavy_rank - 1] = work
            if is_normal != 0:
                mWork[normal_rank - 1] = work
            heavy_base += chunk_heavy
            normal_base += chunk_normal


class BwdKvNoStraddlePack:
    """Pack Page16 groups without allowing a group to cross a Page8 tile."""

    @cute.jit
    def __call__(
        self,
        mPageCounts: cute.Tensor,
        mStartSlots: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mPageCounts, mStartSlots).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, mPageCounts: cute.Tensor, mStartSlots: cute.Tensor):
        start = Int32(0)
        remaining = Int32(8)
        for block_idx in cutlass.range(mPageCounts.shape[0]):
            page_count = mPageCounts[block_idx]
            if page_count > remaining:
                start += remaining
                remaining = Int32(8)
            mStartSlots[block_idx] = start
            start += page_count
            remaining -= page_count
            if remaining == 0:
                remaining = Int32(8)


class BwdKvPageTableFill:
    """Fill the owned-KV page ids and the valid row count of every slot."""

    def __init__(self):
        self.num_threads = 256

    @cute.jit
    def __call__(
        self,
        mStartSlots: cute.Tensor,
        mPageCounts: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mKvPageTable: cute.Tensor,
        mKvValid: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid = (
            (mPageCounts.shape[0] + self.num_threads - 1) // self.num_threads,
            1,
            1,
        )
        self.kernel(
            mStartSlots, mPageCounts, mBlockSizes, mKvPageTable, mKvValid
        ).launch(grid=grid, block=(self.num_threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        mStartSlots: cute.Tensor,
        mPageCounts: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mKvPageTable: cute.Tensor,
        mKvValid: cute.Tensor,
    ):
        block_idx = (
            cute.arch.block_idx()[0] * self.num_threads + cute.arch.thread_idx()[0]
        )
        if block_idx < mPageCounts.shape[0]:
            start = mStartSlots[block_idx]
            page_count = mPageCounts[block_idx]
            valid_rows = mBlockSizes[block_idx]
            for page_idx in cutlass.range_constexpr(8):
                if page_count > page_idx:
                    mKvPageTable[start + page_idx] = block_idx * 8 + page_idx
                    mKvValid[start + page_idx] = Uint8(
                        min(max(valid_rows - page_idx * PAGE, 0), PAGE)
                    )


class BwdKvEdgeBitmap:
    """Scatter forward edges into packed backward adjacency bytes."""

    def __init__(self):
        self.num_threads = 256

    @cute.jit
    def __call__(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mFirstTile: cute.Tensor,
        mTileSlot: cute.Tensor,
        mQPageCounts: cute.Tensor,
        mKvPageCounts: cute.Tensor,
        mBitmap: cute.Tensor,
        mConfig: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mBlockIndex,
            mBlockCount,
            mFirstTile,
            mTileSlot,
            mQPageCounts,
            mKvPageCounts,
            mBitmap,
            mConfig,
        ).launch(
            grid=(mBlockIndex.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mFirstTile: cute.Tensor,
        mTileSlot: cute.Tensor,
        mQPageCounts: cute.Tensor,
        mKvPageCounts: cute.Tensor,
        mBitmap: cute.Tensor,
        mConfig: cute.Tensor,
    ):
        walk_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        n_q_blocks = mConfig[0]
        num_groups = mConfig[1]
        bucket = mConfig[2]
        words_per_bucket = mConfig[3]
        n_kv_blocks = mConfig[4]
        cap = mBlockIndex.shape[1]
        edge_count = max(min(mBlockCount[walk_idx], cap), 0)
        head_idx = walk_idx // n_q_blocks
        q_block = walk_idx - head_idx * n_q_blocks
        q_pages = mQPageCounts[q_block]
        group_idx = q_block // bucket
        q_relative = q_block - group_idx * bucket

        edge_idx = thread_idx
        while edge_idx < edge_count:
            kv_block = max(min(mBlockIndex[walk_idx, edge_idx], n_kv_blocks - 1), 0)
            kv_pages = mKvPageCounts[kv_block]
            if q_pages > 0:
                if kv_pages > 0:
                    first_slot = mTileSlot[kv_block]
                    tile_idx = mFirstTile[kv_block]
                    adjacency = ((Int32(1) << kv_pages) - 1) << first_slot
                    row = (head_idx * num_groups + group_idx) * n_kv_blocks + tile_idx
                    word_idx = row * words_per_bucket + q_relative // 4
                    shift = (q_relative % 4) * 8
                    cute.arch.atomic_or(
                        ptr=(mBitmap.iterator + word_idx).llvm_ptr,
                        val=adjacency << shift,
                    )
            edge_idx += self.num_threads


class BwdKvBitmapStats:
    """Count backward walk steps and distinct Q blocks per bitmap row."""

    def __init__(self):
        self.num_threads = 256
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mBitmap: cute.Tensor,
        mQPageCounts: cute.Tensor,
        mSteps: cute.Tensor,
        mNnz: cute.Tensor,
        mConfig: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_full: cute.struct.MemRange[Int32, self.num_warps]
            warp_partial: cute.struct.MemRange[Int32, self.num_warps]
            warp_nnz: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(mBitmap, mQPageCounts, mSteps, mNnz, mConfig, SharedStorage).launch(
            grid=(mSteps.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mBitmap: cute.Tensor,
        mQPageCounts: cute.Tensor,
        mSteps: cute.Tensor,
        mNnz: cute.Tensor,
        mConfig: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        row_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n_q_blocks = mConfig[0]
        num_groups = mConfig[1]
        bucket = mConfig[2]
        words_per_bucket = mConfig[3]
        head_group = row_idx // mConfig[4]
        group_idx = head_group % num_groups
        q_base = group_idx * bucket

        full_count = Int32(0)
        partial_pages = Int32(0)
        nnz_count = Int32(0)
        word_in_bucket = thread_idx
        while word_in_bucket < words_per_bucket:
            word = mBitmap[row_idx * words_per_bucket + word_in_bucket]
            first_q = q_base + word_in_bucket * 4
            for byte_idx in cutlass.range_constexpr(4):
                adjacency = (word >> (byte_idx * 8)) & 0xFF
                if adjacency != 0:
                    q_block = first_q + byte_idx
                    if q_block < n_q_blocks:
                        q_pages = mQPageCounts[q_block]
                        if q_pages == 8:
                            full_count += 1
                        else:
                            partial_pages += q_pages
                        nnz_count += 1
            word_in_bucket += self.num_threads

        full_count = warp_reduce(full_count, lambda x, y: x + y)
        partial_pages = warp_reduce(partial_pages, lambda x, y: x + y)
        nnz_count = warp_reduce(nnz_count, lambda x, y: x + y)
        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_full = storage.warp_full.get_tensor(cute.make_layout(self.num_warps))
        warp_partial = storage.warp_partial.get_tensor(cute.make_layout(self.num_warps))
        warp_nnz = storage.warp_nnz.get_tensor(cute.make_layout(self.num_warps))
        if lane_idx == 0:
            warp_full[warp_idx] = full_count
            warp_partial[warp_idx] = partial_pages
            warp_nnz[warp_idx] = nnz_count
        cute.arch.sync_threads()
        if warp_idx == 0:
            full_count = warp_full[lane_idx] if lane_idx < self.num_warps else Int32(0)
            partial_pages = (
                warp_partial[lane_idx] if lane_idx < self.num_warps else Int32(0)
            )
            nnz_count = warp_nnz[lane_idx] if lane_idx < self.num_warps else Int32(0)
            full_count = warp_reduce(full_count, lambda x, y: x + y)
            partial_pages = warp_reduce(partial_pages, lambda x, y: x + y)
            nnz_count = warp_reduce(nnz_count, lambda x, y: x + y)
            if lane_idx == 0:
                mSteps[row_idx] = full_count + (partial_pages + 7) // 8
                mNnz[row_idx] = nnz_count


class BwdKvCompactWork:
    """Compact active backward rows while preserving their stable order."""

    def __init__(self):
        self.num_threads = 1024
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mSteps: cute.Tensor,
        mConfig: cute.Tensor,
        mWork: cute.Tensor,
        mWorkCount: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_count: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(
            mSteps,
            mConfig,
            mWork,
            mWorkCount,
            SharedStorage,
        ).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSteps: cute.Tensor,
        mConfig: cute.Tensor,
        mWork: cute.Tensor,
        mWorkCount: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        num_rows = mSteps.shape[0]
        n_kv_blocks = mConfig[4]
        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_count = storage.warp_count.get_tensor(cute.make_layout(self.num_warps))

        local_active = Int32(0)
        row_idx = thread_idx
        while row_idx < num_rows:
            local_active += Int32(mSteps[row_idx] > 0)
            row_idx += self.num_threads
        local_active = warp_reduce(local_active, lambda x, y: x + y)
        if lane_idx == 0:
            warp_count[warp_idx] = local_active
        cute.arch.sync_threads()
        active_total = Int32(0)
        for idx in cutlass.range_constexpr(self.num_warps):
            active_total += warp_count[idx]
        if thread_idx == 0:
            mWorkCount[0] = active_total
        cute.arch.sync_threads()

        active_base = Int32(0)
        num_chunks = (num_rows + self.num_threads - 1) // self.num_threads
        for chunk in cutlass.range(num_chunks):
            row_idx = chunk * self.num_threads + thread_idx
            step_count = mSteps[row_idx] if row_idx < num_rows else Int32(0)
            is_active = Int32(step_count > 0)

            active_rank = is_active
            for delta in (1, 2, 4, 8, 16):
                other = cute.arch.shuffle_sync_up(active_rank, delta, mask_and_clamp=0)
                if lane_idx >= delta:
                    active_rank += other
            if lane_idx == 31:
                warp_count[warp_idx] = active_rank
            cute.arch.sync_threads()
            active_rank += active_base
            for idx in cutlass.range(warp_idx):
                active_rank += warp_count[idx]
            chunk_active = Int32(0)
            for idx in cutlass.range_constexpr(self.num_warps):
                chunk_active += warp_count[idx]
            cute.arch.sync_threads()

            if is_active != 0:
                head_group = row_idx // n_kv_blocks
                mWork[active_rank - 1, 0] = head_group
                mWork[active_rank - 1, 1] = row_idx - head_group * n_kv_blocks
            active_base += chunk_active


class BwdKvBitmapCompact:
    """Compact nonzero bitmap bytes into stable Q-block entry lists."""

    def __init__(self):
        self.num_threads = 256
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mBitmap: cute.Tensor,
        mEntryOffsets: cute.Tensor,
        mQBlocks: cute.Tensor,
        mAdjacencyBytes: cute.Tensor,
        mConfig: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_count: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(
            mBitmap,
            mEntryOffsets,
            mQBlocks,
            mAdjacencyBytes,
            mConfig,
            SharedStorage,
        ).launch(
            grid=(mEntryOffsets.shape[0] - 1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mBitmap: cute.Tensor,
        mEntryOffsets: cute.Tensor,
        mQBlocks: cute.Tensor,
        mAdjacencyBytes: cute.Tensor,
        mConfig: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        row_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        num_groups = mConfig[1]
        bucket = mConfig[2]
        words_per_bucket = mConfig[3]
        head_group = row_idx // mConfig[4]
        group_idx = head_group % num_groups
        q_base = group_idx * bucket
        output_base = mEntryOffsets[row_idx]

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_count = storage.warp_count.get_tensor(cute.make_layout(self.num_warps))
        entry_base = Int32(0)
        num_chunks = (words_per_bucket + self.num_threads - 1) // self.num_threads
        for chunk in cutlass.range(num_chunks):
            word_in_bucket = chunk * self.num_threads + thread_idx
            word = Int32(0)
            if word_in_bucket < words_per_bucket:
                word = mBitmap[row_idx * words_per_bucket + word_in_bucket]
            byte0 = word & 0xFF
            byte1 = (word >> 8) & 0xFF
            byte2 = (word >> 16) & 0xFF
            byte3 = (word >> 24) & 0xFF
            entry_count = (
                Int32(byte0 != 0)
                + Int32(byte1 != 0)
                + Int32(byte2 != 0)
                + Int32(byte3 != 0)
            )
            entry_rank = entry_count
            for delta in (1, 2, 4, 8, 16):
                other = cute.arch.shuffle_sync_up(entry_rank, delta, mask_and_clamp=0)
                if lane_idx >= delta:
                    entry_rank += other
            if lane_idx == 31:
                warp_count[warp_idx] = entry_rank
            cute.arch.sync_threads()
            entry_rank += entry_base
            for idx in cutlass.range(warp_idx):
                entry_rank += warp_count[idx]
            output_idx = output_base + entry_rank - entry_count
            first_q = q_base + word_in_bucket * 4
            if byte0 != 0:
                mQBlocks[output_idx] = first_q
                mAdjacencyBytes[output_idx] = byte0
                output_idx += 1
            if byte1 != 0:
                mQBlocks[output_idx] = first_q + 1
                mAdjacencyBytes[output_idx] = byte1
                output_idx += 1
            if byte2 != 0:
                mQBlocks[output_idx] = first_q + 2
                mAdjacencyBytes[output_idx] = byte2
                output_idx += 1
            if byte3 != 0:
                mQBlocks[output_idx] = first_q + 3
                mAdjacencyBytes[output_idx] = byte3

            for idx in cutlass.range(self.num_warps):
                entry_base += warp_count[idx]
            cute.arch.sync_threads()


class BwdKvWalkFill:
    """Fill backward Q-page walks, full Q blocks before partial blocks."""

    def __init__(self):
        self.num_threads = 256
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mSegments: cute.Tensor,
        mQBlocks: cute.Tensor,
        mAdjacencyBytes: cute.Tensor,
        mQPageCounts: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mPageTable: cute.Tensor,
        mSlotBytes: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_pages: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(
            mSegments,
            mQBlocks,
            mAdjacencyBytes,
            mQPageCounts,
            mTileOffsets,
            mPageTable,
            mSlotBytes,
            SharedStorage,
        ).launch(
            grid=(mTileOffsets.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSegments: cute.Tensor,
        mQBlocks: cute.Tensor,
        mAdjacencyBytes: cute.Tensor,
        mQPageCounts: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mPageTable: cute.Tensor,
        mSlotBytes: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        row_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        entry_begin = mSegments[row_idx]
        entry_count = mSegments[row_idx + 1] - entry_begin
        output_base = mTileOffsets[row_idx] * 8

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_pages = storage.warp_pages.get_tensor(cute.make_layout(self.num_warps))
        page_base = Int32(0)
        num_chunks = (entry_count + self.num_threads - 1) // self.num_threads
        for pass_idx in cutlass.range_constexpr(2):
            for chunk in cutlass.range(num_chunks):
                entry_idx = chunk * self.num_threads + thread_idx
                q_block = Int32(0)
                adjacency = Int32(0)
                selected_pages = Int32(0)
                if entry_idx < entry_count:
                    q_block = mQBlocks[entry_begin + entry_idx]
                    adjacency = mAdjacencyBytes[entry_begin + entry_idx]
                    q_pages = mQPageCounts[q_block]
                    if pass_idx == 0:
                        if q_pages == 8:
                            selected_pages = Int32(8)
                    else:
                        if q_pages != 8:
                            selected_pages = q_pages

                page_rank = selected_pages
                for delta in (1, 2, 4, 8, 16):
                    other = cute.arch.shuffle_sync_up(
                        page_rank, delta, mask_and_clamp=0
                    )
                    if lane_idx >= delta:
                        page_rank += other
                if lane_idx == 31:
                    warp_pages[warp_idx] = page_rank
                cute.arch.sync_threads()
                page_rank += page_base
                for idx in cutlass.range(warp_idx):
                    page_rank += warp_pages[idx]
                page_pos = page_rank - selected_pages
                first_page = q_block * 8
                for page_idx in cutlass.range_constexpr(8):
                    if selected_pages > page_idx:
                        output_idx = output_base + page_pos + page_idx
                        mPageTable[output_idx] = first_page + page_idx
                        mSlotBytes[output_idx] = Uint8(adjacency)

                for idx in cutlass.range(self.num_warps):
                    page_base += warp_pages[idx]
                cute.arch.sync_threads()


class ZeroPadForwardState:
    """Clear padding Q rows of O and LSE before backward preprocess."""

    def __init__(self):
        self.num_threads = 128

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        mLse: cute.Tensor,
        mQSizes: cute.Tensor,
        stream: cuda.CUstream,
    ):
        mOut = assume_tensor_aligned(mOut)
        gmem_tiled_copy = copy_utils.tiled_copy_2d(
            mOut.element_type,
            threads_per_row=16,
            num_threads=self.num_threads,
            num_copy_elems=8,
        )
        chunks_per_block = BLOCK // PAGE
        self.kernel(mOut, mLse, mQSizes, gmem_tiled_copy).launch(
            grid=(
                mOut.shape[1] * mQSizes.shape[0] * chunks_per_block,
                1,
                1,
            ),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mOut: cute.Tensor,
        mLse: cute.Tensor,
        mQSizes: cute.Tensor,
        gmem_tiled_copy: cute.TiledCopy,
    ):
        block_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        chunks_per_block = BLOCK // PAGE
        head_block = block_idx // chunks_per_block
        chunk_idx = block_idx - head_block * chunks_per_block
        n_blocks = mQSizes.shape[0]
        head_idx = head_block // n_blocks
        logical_block = head_block - head_idx * n_blocks
        valid_rows = max(min(mQSizes[logical_block], BLOCK), 0)
        row_base = chunk_idx * PAGE
        rows_in_tensor = min(
            mOut.shape[2] - logical_block * BLOCK, BLOCK
        )  # a final block may end before row 128

        if row_base + PAGE > valid_rows:
            tile_idx = logical_block * chunks_per_block + chunk_idx
            gOut = cute.local_tile(
                mOut[0, head_idx, None, None],
                (PAGE, HEAD_DIM),
                (tile_idx, 0),
            )
            cOut = cute.make_identity_tensor((PAGE, HEAD_DIM))
            gmem_thr_copy = gmem_tiled_copy.get_slice(thread_idx)
            tOgOut = gmem_thr_copy.partition_D(gOut)
            tOcOut = gmem_thr_copy.partition_D(cOut)
            zeros = cute.make_fragment_like(tOgOut[None, 0, 0])
            zeros.fill(0.0)
            for rest_m in cutlass.range_constexpr(tOgOut.shape[1]):
                row = row_base + tOcOut[0, rest_m, 0][0]
                if row >= valid_rows and row < rows_in_tensor:
                    for rest_k in cutlass.range_constexpr(tOgOut.shape[2]):
                        cute.copy(
                            gmem_tiled_copy,
                            zeros,
                            tOgOut[None, rest_m, rest_k],
                        )

            if (
                thread_idx < PAGE
                and row_base + thread_idx >= valid_rows
                and row_base + thread_idx < rows_in_tensor
            ):
                row_idx = logical_block * BLOCK + row_base + thread_idx
                mLse[0, head_idx, row_idx] = cutlass.Float32(0.0)


class SetPadQStats:
    """Give padding Q rows LSE = +inf and dPsum = 0 so P and dS vanish on their columns."""

    def __init__(self):
        self.num_threads = 128

    @cute.jit
    def __call__(
        self,
        mLseLog2: cute.Tensor,  # [1, H, T] fp32
        mDPsum: cute.Tensor,  # [1, H, T] fp32
        mQSizes: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mLseLog2, mDPsum, mQSizes).launch(
            grid=(mLseLog2.shape[1] * mQSizes.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mLseLog2: cute.Tensor,
        mDPsum: cute.Tensor,
        mQSizes: cute.Tensor,
    ):
        block_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        n_blocks = mQSizes.shape[0]
        head_idx = block_idx // n_blocks
        logical_block = block_idx - head_idx * n_blocks
        valid_rows = max(min(mQSizes[logical_block], BLOCK), 0)
        if thread_idx >= valid_rows:
            row_idx = logical_block * BLOCK + thread_idx
            mLseLog2[0, head_idx, row_idx] = cutlass.Float32.inf
            mDPsum[0, head_idx, row_idx] = cutlass.Float32(0.0)


class ZeroPadRows:
    """Clear padding rows of dQ (Q partition) and dK/dV (KV partition, kv heads) with vectorized stores."""

    def __init__(self):
        self.num_threads = 128

    @cute.jit
    def __call__(
        self,
        mDq: cute.Tensor,
        mDk: cute.Tensor,
        mDv: cute.Tensor,
        mQSizes: cute.Tensor,
        mKvSizes: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mDq.element_type == mDk.element_type == mDv.element_type
        mDq = assume_tensor_aligned(mDq)
        mDk = assume_tensor_aligned(mDk)
        mDv = assume_tensor_aligned(mDv)
        gmem_tiled_copy = copy_utils.tiled_copy_2d(
            mDq.element_type,
            threads_per_row=16,
            num_threads=self.num_threads,
            num_copy_elems=8,
        )
        chunks_per_block = BLOCK // PAGE
        self.kernel(mDq, mDk, mDv, mQSizes, mKvSizes, gmem_tiled_copy).launch(
            grid=(
                mDq.shape[1]
                * max(mQSizes.shape[0], mKvSizes.shape[0])
                * chunks_per_block,
                1,
                1,
            ),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mDq: cute.Tensor,
        mDk: cute.Tensor,
        mDv: cute.Tensor,
        mQSizes: cute.Tensor,
        mKvSizes: cute.Tensor,
        gmem_tiled_copy: cute.TiledCopy,
    ):
        block_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        chunks_per_block = BLOCK // PAGE
        head_block = block_idx // chunks_per_block
        chunk_idx = block_idx - head_block * chunks_per_block
        n_q_blocks = mQSizes.shape[0]
        n_kv_blocks = mKvSizes.shape[0]
        n_blocks = max(n_q_blocks, n_kv_blocks)
        head_idx = head_block // n_blocks
        logical_block = head_block - head_idx * n_blocks
        row_base = chunk_idx * PAGE
        tile_idx = logical_block * chunks_per_block + chunk_idx
        gmem_thr_copy = gmem_tiled_copy.get_slice(thread_idx)
        cRows = gmem_thr_copy.partition_D(cute.make_identity_tensor((PAGE, HEAD_DIM)))

        valid_q = Int32(BLOCK)
        rows_q = Int32(0)
        if logical_block < n_q_blocks:
            valid_q = max(min(mQSizes[logical_block], BLOCK), 0)
            rows_q = min(
                mDq.shape[2] - logical_block * BLOCK, BLOCK
            )  # a final block may end before row 128
        if row_base + PAGE > valid_q:
            gDq = cute.local_tile(
                mDq[0, head_idx, None, None], (PAGE, HEAD_DIM), (tile_idx, 0)
            )
            tDqgDq = gmem_thr_copy.partition_D(gDq)
            zeros = cute.make_fragment_like(tDqgDq[None, 0, 0])
            zeros.fill(0.0)
            for rest_m in cutlass.range_constexpr(tDqgDq.shape[1]):
                row = row_base + cRows[0, rest_m, 0][0]
                if row >= valid_q and row < rows_q:
                    for rest_k in cutlass.range_constexpr(tDqgDq.shape[2]):
                        cute.copy(gmem_tiled_copy, zeros, tDqgDq[None, rest_m, rest_k])

        valid_kv = Int32(BLOCK)
        rows_kv = Int32(0)
        if logical_block < n_kv_blocks and head_idx < mDk.shape[1]:
            valid_kv = max(min(mKvSizes[logical_block], BLOCK), 0)
            rows_kv = min(mDk.shape[2] - logical_block * BLOCK, BLOCK)
        if row_base + PAGE > valid_kv:
            gDk = cute.local_tile(
                mDk[0, head_idx, None, None], (PAGE, HEAD_DIM), (tile_idx, 0)
            )
            gDv = cute.local_tile(
                mDv[0, head_idx, None, None], (PAGE, HEAD_DIM), (tile_idx, 0)
            )
            tDk_gDk = gmem_thr_copy.partition_D(gDk)
            tDv_gDv = gmem_thr_copy.partition_D(gDv)
            zeros = cute.make_fragment_like(tDk_gDk[None, 0, 0])
            zeros.fill(0.0)
            for rest_m in cutlass.range_constexpr(tDk_gDk.shape[1]):
                row = row_base + cRows[0, rest_m, 0][0]
                if row >= valid_kv and row < rows_kv:
                    for rest_k in cutlass.range_constexpr(tDk_gDk.shape[2]):
                        cute.copy(gmem_tiled_copy, zeros, tDk_gDk[None, rest_m, rest_k])
                        cute.copy(gmem_tiled_copy, zeros, tDv_gDv[None, rest_m, rest_k])


def _compile_tensors(tensors: Tuple[Tensor, ...]):
    converted = []
    for tensor in tensors:
        dynamic_modes = tuple(range(tensor.ndim)) if tensor.ndim <= 2 else (0, 1, 2)
        converted.append(
            to_cute_tensor(
                tensor,
                dynamic_modes=dynamic_modes,
                assumed_align=16 if tensor.ndim > 1 else 4,
            )
        )
    return tuple(converted)


def _run_padding_kernel(
    name: str,
    kernel,
    tensors: Tuple[Tensor, ...],
    cache: dict,
) -> None:
    compile_key = make_compile_key(
        name,
        (tuple(tensor.dtype for tensor in tensors), HEAD_DIM),
        tensors,
    )
    if compile_key not in cache:
        converted = _compile_tensors(tensors)
        cache[compile_key] = cute.compile(
            kernel,
            *converted,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cache[compile_key](*tensors, stream)


def _run_metadata_kernel(
    name: str,
    kernel,
    tensors: Tuple[Tensor, ...],
    cache: dict,
    stream: cuda.CUstream = None,
) -> None:
    compile_key = make_compile_key(
        name,
        tuple(tensor.dtype for tensor in tensors),
        tensors,
    )
    if compile_key not in cache:
        converted = _compile_tensors(tensors)
        cache[compile_key] = cute.compile(
            kernel,
            *converted,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
    if stream is None:
        stream = current_stream()
    cache[compile_key](*tensors, stream)


def current_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def walk_stats(
    block_index: Tensor,
    block_count: Tensor,
    q_sizes: Tensor,
    block_sizes: Tensor,
    shared_count: Tensor,
    stream: cuda.CUstream = None,
    block: int = BLOCK,
) -> None:
    _run_metadata_kernel(
        f"kpage_walk_stats_blk{block}",
        WalkStats(block),
        (block_index, block_count, q_sizes, block_sizes, shared_count),
        walk_stats.compile_cache,
        stream,
    )


walk_stats.compile_cache = {}


def compact_q_routes(
    q_sizes: Tensor,
    tile_offsets: Tensor,
    tile_counts: Tensor,
    small_q: Tensor,
    large_q: Tensor,
    route_counts: Tensor,
    stream: cuda.CUstream = None,
) -> None:
    _run_metadata_kernel(
        "kpage_compact_q_routes",
        CompactQRoutes(),
        (
            q_sizes,
            tile_offsets,
            tile_counts,
            small_q,
            large_q,
            route_counts,
        ),
        compact_q_routes.compile_cache,
        stream,
    )


compact_q_routes.compile_cache = {}


def walk_fill(
    block_index: Tensor,
    block_count: Tensor,
    q_sizes: Tensor,
    block_sizes: Tensor,
    tile_offsets: Tensor,
    tile_counts: Tensor,
    full_tiles: Tensor,
    q64_offsets: Tensor,
    q64_iter_counts: Tensor,
    q64_full_iters: Tensor,
    page_table: Tensor,
    mask_table: Tensor,
    stream: cuda.CUstream = None,
    block: int = BLOCK,
) -> None:
    _run_metadata_kernel(
        f"kpage_walk_fill_blk{block}",
        WalkFill(block),
        (
            block_index,
            block_count,
            q_sizes,
            block_sizes,
            tile_offsets,
            tile_counts,
            full_tiles,
            q64_offsets,
            q64_iter_counts,
            q64_full_iters,
            page_table,
            mask_table,
        ),
        walk_fill.compile_cache,
        stream,
    )


walk_fill.compile_cache = {}


def order_q_route_work(
    q_route: Tensor,
    cost: Tensor,
    work: Tensor,
    *,
    heavy_threshold: int,
    is_q64: bool,
    stream: cuda.CUstream = None,
) -> None:
    name = f"kpage_order_q_route_work_q{64 if is_q64 else 128}"
    compile_key = make_compile_key(
        name,
        (heavy_threshold, is_q64, *(tensor.dtype for tensor in (q_route, cost, work))),
        (q_route, cost, work),
    )
    if compile_key not in order_q_route_work.compile_cache:
        converted = _compile_tensors((q_route, cost, work))
        order_q_route_work.compile_cache[compile_key] = cute.compile(
            OrderQRouteWork(heavy_threshold, is_q64),
            *converted,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
    if stream is None:
        stream = current_stream()
    order_q_route_work.compile_cache[compile_key](q_route, cost, work, stream)


order_q_route_work.compile_cache = {}


def bwd_kv_nostraddle_pack(page_counts: Tensor, start_slots: Tensor) -> None:
    _run_metadata_kernel(
        "kpage_bwd_kv_nostraddle_pack",
        BwdKvNoStraddlePack(),
        (page_counts, start_slots),
        bwd_kv_nostraddle_pack.compile_cache,
    )


bwd_kv_nostraddle_pack.compile_cache = {}


def bwd_kv_page_table_fill(
    start_slots: Tensor,
    page_counts: Tensor,
    block_sizes: Tensor,
    kv_page_table: Tensor,
    kv_valid: Tensor,
) -> None:
    _run_metadata_kernel(
        "kpage_bwd_kv_page_table_fill",
        BwdKvPageTableFill(),
        (start_slots, page_counts, block_sizes, kv_page_table, kv_valid),
        bwd_kv_page_table_fill.compile_cache,
    )


bwd_kv_page_table_fill.compile_cache = {}


def bwd_kv_edge_bitmap(
    block_index: Tensor,
    block_count: Tensor,
    first_tile: Tensor,
    tile_slot: Tensor,
    q_page_counts: Tensor,
    kv_page_counts: Tensor,
    bitmap: Tensor,
    config: Tensor,
) -> None:
    _run_metadata_kernel(
        "kpage_bwd_kv_edge_bitmap",
        BwdKvEdgeBitmap(),
        (
            block_index,
            block_count,
            first_tile,
            tile_slot,
            q_page_counts,
            kv_page_counts,
            bitmap,
            config,
        ),
        bwd_kv_edge_bitmap.compile_cache,
    )


bwd_kv_edge_bitmap.compile_cache = {}


def bwd_kv_bitmap_stats(
    bitmap: Tensor,
    q_page_counts: Tensor,
    steps: Tensor,
    nnz: Tensor,
    config: Tensor,
) -> None:
    _run_metadata_kernel(
        "kpage_bwd_kv_bitmap_stats",
        BwdKvBitmapStats(),
        (bitmap, q_page_counts, steps, nnz, config),
        bwd_kv_bitmap_stats.compile_cache,
    )


bwd_kv_bitmap_stats.compile_cache = {}


def bwd_kv_compact_work(
    steps: Tensor,
    config: Tensor,
    work: Tensor,
    work_count: Tensor,
) -> None:
    name = "kpage_bwd_kv_compact_work"
    tensors = (steps, config, work, work_count)
    compile_key = make_compile_key(
        name,
        tuple(tensor.dtype for tensor in tensors),
        tensors,
    )
    if compile_key not in bwd_kv_compact_work.compile_cache:
        converted = _compile_tensors(tensors)
        bwd_kv_compact_work.compile_cache[compile_key] = cute.compile(
            BwdKvCompactWork(),
            *converted,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
    bwd_kv_compact_work.compile_cache[compile_key](*tensors, current_stream())


bwd_kv_compact_work.compile_cache = {}


def bwd_kv_bitmap_compact(
    bitmap: Tensor,
    entry_offsets: Tensor,
    q_blocks: Tensor,
    adjacency_bytes: Tensor,
    config: Tensor,
) -> None:
    _run_metadata_kernel(
        "kpage_bwd_kv_bitmap_compact",
        BwdKvBitmapCompact(),
        (bitmap, entry_offsets, q_blocks, adjacency_bytes, config),
        bwd_kv_bitmap_compact.compile_cache,
    )


bwd_kv_bitmap_compact.compile_cache = {}


def bwd_kv_walk_fill(
    segments: Tensor,
    q_blocks: Tensor,
    adjacency_bytes: Tensor,
    q_page_counts: Tensor,
    tile_offsets: Tensor,
    page_table: Tensor,
    slot_bytes: Tensor,
) -> None:
    _run_metadata_kernel(
        "kpage_bwd_kv_walk_fill",
        BwdKvWalkFill(),
        (
            segments,
            q_blocks,
            adjacency_bytes,
            q_page_counts,
            tile_offsets,
            page_table,
            slot_bytes,
        ),
        bwd_kv_walk_fill.compile_cache,
    )


bwd_kv_walk_fill.compile_cache = {}


def zero_pad_forward_state(out: Tensor, lse: Tensor, block_sizes: Tensor) -> None:
    _run_padding_kernel(
        "kpage_zero_pad_forward_state",
        ZeroPadForwardState(),
        (out, lse, block_sizes),
        zero_pad_forward_state.compile_cache,
    )


zero_pad_forward_state.compile_cache = {}


def zero_pad_rows(
    dq: Tensor, dk: Tensor, dv: Tensor, q_sizes: Tensor, kv_sizes: Tensor
) -> None:
    _run_padding_kernel(
        "kpage_zero_pad_rows",
        ZeroPadRows(),
        (dq, dk, dv, q_sizes, kv_sizes),
        zero_pad_rows.compile_cache,
    )


zero_pad_rows.compile_cache = {}


def set_pad_q_stats(lse_log2: Tensor, dpsum: Tensor, block_sizes: Tensor) -> None:
    _run_padding_kernel(
        "kpage_set_pad_q_stats",
        SetPadQStats(),
        (lse_log2, dpsum, block_sizes),
        set_pad_q_stats.compile_cache,
    )


set_pad_q_stats.compile_cache = {}
