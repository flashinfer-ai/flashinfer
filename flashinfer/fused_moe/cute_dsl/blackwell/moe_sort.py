# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""CuTe DSL MoE token-routing sort.

Algorithm:
  1. Flatten each ``(token, k)`` selection into an expanded pair and mask
     experts outside the local expert range.
  2. Count pairs per local expert, round counts to ``tile_size``, and prefix-sum
     the padded counts to form expert-major row segments.
  3. Scatter pairs into those segments and emit both mapping directions. This
     is a counting sort; the tiny-input path computes each pair's rank directly.

Tensor layouts (T=tokens, K=top-k):
  - ``selected_experts`` and ``final_scales`` are row-major [T, K]. Expanded
    pair ``(token, k)`` has index ``token * K + k``; scales remain unchanged.
  - ``tile_experts[max_tiles]`` stores a local expert ID per GEMM tile and
    ``tile_limits[max_tiles]`` stores each tile's exclusive valid row end.
  - ``expanded_to_permuted[T, K]`` maps pairs to expert-major rows; ``-1``
    marks non-local pairs. ``permuted_to_expanded[max_tiles * tile_size]`` is
    the reverse map, with padding rows unspecified.
  - ``total_padded[1]`` and ``num_tiles[1]`` describe the valid output prefix.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_memory
import cutlass.pipeline as pipeline
import torch
from cuda.bindings import driver as cuda


CONTIGUOUS_ROUTE_WINDOW_MIN_TOKENS = 65536


class MoeSortKernel:
    """Group selected token/expert pairs into tile-padded expert segments."""

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        local_expert_offset: int,
        num_local_experts: int,
        tile_size: int,
        *,
        use_pdl: bool,
    ):
        if num_tokens < 1:
            raise ValueError("num_tokens must be positive")
        if num_experts < 1 or num_experts > 1024:
            raise ValueError("num_experts must be in [1, 1024]")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if local_expert_offset < 0:
            raise ValueError("local_expert_offset must be non-negative")
        if num_local_experts < 1:
            raise ValueError("num_local_experts must be positive")
        if local_expert_offset + num_local_experts > num_experts:
            raise ValueError("local expert range exceeds num_experts")
        if tile_size < 1:
            raise ValueError("tile_size must be positive")

        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_pairs = num_tokens * top_k
        self.local_expert_offset = local_expert_offset
        self.num_local_experts = num_local_experts
        self.tile_size = tile_size
        self.use_pdl = use_pdl
        self.tiny_sort = self.num_pairs <= 4
        self.contiguous_windows = num_tokens >= CONTIGUOUS_ROUTE_WINDOW_MIN_TOKENS
        self.parallel_scan = num_tokens >= 16384

        self.threads_per_cta = 32 if self.tiny_sort else 128
        if not self.tiny_sort:
            while self.threads_per_cta < num_local_experts:
                self.threads_per_cta *= 2
            self.threads_per_cta = min(self.threads_per_cta, 512)
            if self.num_pairs > 8192:
                if num_experts <= 128:
                    self.threads_per_cta = (
                        512
                        if self.num_pairs <= 16384 or num_tokens >= 131072
                        else (
                            (1024 if num_tokens < 32768 else 896)
                            if self.parallel_scan
                            else 256
                        )
                    )
                else:
                    self.threads_per_cta = (
                        (1024 if num_tokens < 32768 else 896)
                        if self.parallel_scan and not self.contiguous_windows
                        else (1024 if num_tokens >= 131072 else 512)
                    )

        target_ctas = max(
            1,
            (
                self.num_pairs
                + self.threads_per_cta * (4 if self.num_pairs <= 8192 else 1)
                - 1
            )
            // (self.threads_per_cta * (4 if self.num_pairs <= 8192 else 1)),
        )
        if self.num_pairs <= 16384:
            max_ctas = 16
        elif self.num_tokens <= 8192:
            max_ctas = 128
        else:
            reserved_sms = 0 if self.parallel_scan else 8
            max_ctas = max(
                1,
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).multi_processor_count
                - reserved_sms,
            )
        self.num_ctas = 1
        while self.num_ctas < min(target_ctas, max_ctas):
            self.num_ctas *= 2
        self.num_ctas = min(self.num_ctas, max_ctas)
        self.cluster_sort = 1 < self.num_ctas <= 16
        self.warp_scan = self.num_local_experts <= self.threads_per_cta
        self.scan_warps = max(1, (self.num_local_experts + 31) // 32)
        self.pairs_per_cta = (
            (
                (self.num_pairs + self.num_ctas - 1) // self.num_ctas
                + self.threads_per_cta
                - 1
            )
            // self.threads_per_cta
            * self.threads_per_cta
        )
        self.routes_per_thread = (
            self.pairs_per_cta // self.threads_per_cta
            if self.contiguous_windows
            else (self.num_pairs + self.num_ctas * self.threads_per_cta - 1)
            // (self.num_ctas * self.threads_per_cta)
        )

    @cute.jit
    def __call__(
        self,
        mSelectedExperts: cute.Tensor,
        mFinalScales: cute.Tensor,
        mTileExperts: cute.Tensor,
        mTileLimits: cute.Tensor,
        mExpandedToPermuted: cute.Tensor,
        mPermutedToExpanded: cute.Tensor,
        mTotalPadded: cute.Tensor,
        mNumTiles: cute.Tensor,
        mGlobalCounts: cute.Tensor,
        mGlobalOffsets: cute.Tensor,
        mGridSync: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # Sorting preselected experts does not modify scales.
        _ = mFinalScales
        if cutlass.const_expr(self.tiny_sort):
            self.routing_indices_cluster_kernel(
                mSelectedExperts,
                mTileExperts,
                mTileLimits,
                mExpandedToPermuted,
                mPermutedToExpanded,
                mTotalPadded,
                mNumTiles,
            ).launch(
                grid=(1, 1, 1),
                block=(32, 1, 1),
                stream=stream,
                use_pdl=self.use_pdl,
            )
            return

        launch = self.routing_indices_coop_kernel(
            mSelectedExperts,
            mTileExperts,
            mTileLimits,
            mExpandedToPermuted,
            mPermutedToExpanded,
            mTotalPadded,
            mNumTiles,
            mGlobalCounts,
            mGlobalOffsets,
            mGridSync,
        )
        if cutlass.const_expr(self.cluster_sort):
            launch.launch(
                grid=(self.num_ctas, 1, 1),
                block=(self.threads_per_cta, 1, 1),
                cluster=(self.num_ctas, 1, 1),
                stream=stream,
                use_pdl=self.use_pdl,
            )
        else:
            launch.launch(
                grid=(self.num_ctas, 1, 1),
                block=(self.threads_per_cta, 1, 1),
                stream=stream,
                use_pdl=self.use_pdl,
                cooperative=self.num_ctas > 1,
            )

    @cute.kernel
    def routing_indices_cluster_kernel(
        self,
        gSelectedExperts: cute.Tensor,
        gTileExperts: cute.Tensor,
        gTileLimits: cute.Tensor,
        gExpandedToPermuted: cute.Tensor,
        gPermutedToExpanded: cute.Tensor,
        gTotalPadded: cute.Tensor,
        gNumTiles: cute.Tensor,
    ):
        # Tiny inputs use direct pairwise ranks and counts, avoiding global
        # histograms and grid synchronization.
        tidx = cute.arch.thread_idx()[0]
        selected_ptr = gSelectedExperts.iterator.raw_ptr()
        tile_expert_ptr = gTileExperts.iterator.raw_ptr()
        tile_limit_ptr = gTileLimits.iterator.raw_ptr()
        e2p_ptr = gExpandedToPermuted.iterator.raw_ptr()
        p2e_ptr = gPermutedToExpanded.iterator.raw_ptr()
        num_pairs = cutlass.Int32(gSelectedExperts.shape[0]) * cutlass.Int32(self.top_k)

        smem = cutlass_memory.SmemAllocator()
        sRoutes = smem.allocate_array(cutlass.Int32, self.num_pairs)

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        if tidx < num_pairs:
            local_expert = cutlass.Int32((selected_ptr + tidx).load()) - cutlass.Int32(
                self.local_expert_offset
            )
            if local_expert < cutlass.Int32(0) or local_expert >= cutlass.Int32(
                self.num_local_experts
            ):
                local_expert = cutlass.Int32(-1)
            (sRoutes + tidx).store(local_expert)

        pipeline.agent_sync(pipeline.Agent.ThreadBlock)
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        if tidx < num_pairs:
            local_expert = cutlass.Int32((sRoutes + tidx).load())
            if local_expert < cutlass.Int32(0):
                (e2p_ptr + tidx).store(cutlass.Int32(-1))
            else:
                local_rank = cutlass.Int32(0)
                pair = cutlass.Int32(0)
                while pair < tidx:
                    if cutlass.Int32((sRoutes + pair).load()) == local_expert:
                        local_rank = local_rank + cutlass.Int32(1)
                    pair = pair + cutlass.Int32(1)

                expert_count = local_rank
                pair = tidx
                while pair < num_pairs:
                    if cutlass.Int32((sRoutes + pair).load()) == local_expert:
                        expert_count = expert_count + cutlass.Int32(1)
                    pair = pair + cutlass.Int32(1)

                tile_base = cutlass.Int32(0)
                pair = cutlass.Int32(0)
                while pair < num_pairs:
                    other = cutlass.Int32((sRoutes + pair).load())
                    if other >= cutlass.Int32(0) and other < local_expert:
                        first = cutlass.Boolean(True)
                        prior = cutlass.Int32(0)
                        while prior < pair:
                            if cutlass.Int32((sRoutes + prior).load()) == other:
                                first = cutlass.Boolean(False)
                            prior = prior + cutlass.Int32(1)
                        if first:
                            other_count = cutlass.Int32(0)
                            scan = cutlass.Int32(0)
                            while scan < num_pairs:
                                if cutlass.Int32((sRoutes + scan).load()) == other:
                                    other_count = other_count + cutlass.Int32(1)
                                scan = scan + cutlass.Int32(1)
                            tile_base = tile_base + (
                                other_count + cutlass.Int32(self.tile_size - 1)
                            ) // cutlass.Int32(self.tile_size)
                    pair = pair + cutlass.Int32(1)

                padded_start = tile_base * cutlass.Int32(self.tile_size)
                permuted_idx = padded_start + local_rank
                (e2p_ptr + tidx).store(permuted_idx)
                (p2e_ptr + permuted_idx).store(tidx)

                if local_rank == cutlass.Int32(0):
                    expert_tiles = (
                        expert_count + cutlass.Int32(self.tile_size - 1)
                    ) // cutlass.Int32(self.tile_size)
                    expert_tile = cutlass.Int32(0)
                    while expert_tile < expert_tiles:
                        tile_idx = tile_base + expert_tile
                        tile_start = padded_start + expert_tile * cutlass.Int32(
                            self.tile_size
                        )
                        (tile_expert_ptr + tile_idx).store(local_expert)
                        (tile_limit_ptr + tile_idx).store(
                            cutlass.min(
                                tile_start + cutlass.Int32(self.tile_size),
                                padded_start + expert_count,
                            )
                        )
                        expert_tile = expert_tile + cutlass.Int32(1)

        if tidx == 0:
            total_tiles = cutlass.Int32(0)
            pair = cutlass.Int32(0)
            while pair < num_pairs:
                expert = cutlass.Int32((sRoutes + pair).load())
                if expert >= cutlass.Int32(0):
                    first = cutlass.Boolean(True)
                    prior = cutlass.Int32(0)
                    while prior < pair:
                        if cutlass.Int32((sRoutes + prior).load()) == expert:
                            first = cutlass.Boolean(False)
                        prior = prior + cutlass.Int32(1)
                    if first:
                        count = cutlass.Int32(0)
                        scan = cutlass.Int32(0)
                        while scan < num_pairs:
                            if cutlass.Int32((sRoutes + scan).load()) == expert:
                                count = count + cutlass.Int32(1)
                            scan = scan + cutlass.Int32(1)
                        total_tiles = total_tiles + (
                            count + cutlass.Int32(self.tile_size - 1)
                        ) // cutlass.Int32(self.tile_size)
                pair = pair + cutlass.Int32(1)
            gTotalPadded.iterator.raw_ptr().store(
                total_tiles * cutlass.Int32(self.tile_size)
            )
            gNumTiles.iterator.raw_ptr().store(total_tiles)

    @cute.jit
    def _pair_at(
        self,
        cta_rank: cutlass.Int32,
        tidx: cutlass.Int32,
        iteration: cutlass.Int32,
    ):
        if cutlass.const_expr(self.contiguous_windows):
            return (
                cta_rank * cutlass.Int32(self.pairs_per_cta)
                + iteration * cutlass.Int32(self.threads_per_cta)
                + tidx
            )
        return (
            cta_rank * cutlass.Int32(self.threads_per_cta)
            + tidx
            + iteration * cutlass.Int32(self.num_ctas * self.threads_per_cta)
        )

    @cute.jit
    def _inclusive_padded_scan(
        self,
        tidx: cutlass.Int32,
        lane_idx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        sCounts: cute.Pointer,
        sScan0: cute.Pointer,
        sScan1: cute.Pointer,
        sWarpTotals: cute.Pointer,
        sWarpBases: cute.Pointer,
        global_counts_ptr: cute.Pointer,
    ):
        # Inclusive scan of tile-rounded expert counts; the previous expert's
        # result is the current expert's starting row.
        local_expert = tidx
        while local_expert < self.num_local_experts:
            count = cutlass.Int32(0)
            if cutlass.const_expr(self.num_ctas == 1):
                count = cutlass.Int32((sCounts + local_expert).load())
            else:
                count = cutlass.Int32((global_counts_ptr + local_expert).load())
                (sCounts + local_expert).store(count)
            padded_count = (
                (count + cutlass.Int32(self.tile_size - 1))
                // cutlass.Int32(self.tile_size)
                * cutlass.Int32(self.tile_size)
            )
            (sScan0 + local_expert).store(padded_count)
            local_expert = local_expert + self.threads_per_cta

        pipeline.agent_sync(pipeline.Agent.ThreadBlock)

        if cutlass.const_expr(self.warp_scan):
            local_expert = tidx
            inclusive_rows = cutlass.Int32(0)
            if local_expert < self.num_local_experts:
                inclusive_rows = cutlass.Int32((sScan0 + local_expert).load())
            for scan_offset in (1, 2, 4, 8, 16):
                prior_rows = cute.arch.shuffle_sync_up(inclusive_rows, scan_offset)
                if lane_idx >= scan_offset:
                    inclusive_rows = inclusive_rows + prior_rows
            if warp_idx < self.scan_warps and lane_idx == cutlass.Int32(31):
                (sWarpTotals + warp_idx).store(inclusive_rows)

            pipeline.agent_sync(pipeline.Agent.ThreadBlock)
            if tidx == 0:
                warp_base = cutlass.Int32(0)
                metadata_warp = cutlass.Int32(0)
                while metadata_warp < self.scan_warps:
                    (sWarpBases + metadata_warp).store(warp_base)
                    warp_base = warp_base + cutlass.Int32(
                        (sWarpTotals + metadata_warp).load()
                    )
                    metadata_warp = metadata_warp + cutlass.Int32(1)
            pipeline.agent_sync(pipeline.Agent.ThreadBlock)

            if local_expert < self.num_local_experts:
                (sScan0 + local_expert).store(
                    cutlass.Int32((sWarpBases + warp_idx).load()) + inclusive_rows
                )
            pipeline.agent_sync(pipeline.Agent.ThreadBlock)
        else:
            scan_src = sScan0
            scan_dst = sScan1
            scan_stride = 1
            while scan_stride < self.num_local_experts:
                local_expert = tidx
                while local_expert < self.num_local_experts:
                    prefix = cutlass.Int32((scan_src + local_expert).load())
                    if local_expert >= cutlass.Int32(scan_stride):
                        prefix = prefix + cutlass.Int32(
                            (scan_src + local_expert - scan_stride).load()
                        )
                    (scan_dst + local_expert).store(prefix)
                    local_expert = local_expert + self.threads_per_cta
                pipeline.agent_sync(pipeline.Agent.ThreadBlock)
                scan_tmp = scan_src
                scan_src = scan_dst
                scan_dst = scan_tmp
                scan_stride *= 2

            local_expert = tidx
            while local_expert < self.num_local_experts:
                (sScan0 + local_expert).store((scan_src + local_expert).load())
                local_expert = local_expert + self.threads_per_cta
            pipeline.agent_sync(pipeline.Agent.ThreadBlock)

    @cute.jit
    def _reset_counts_after_load(
        self,
        tidx: cutlass.Int32,
        arrival_ptr: cute.Pointer,
        last_cta_ptr: cute.Pointer,
        global_counts_ptr: cute.Pointer,
    ):
        if tidx == 0:
            ticket = cute.arch.atomic_add(
                arrival_ptr,
                cutlass.Int32(1),
                sem="acq_rel",
                scope="gpu",
            )
            generation = ticket // cutlass.Int32(self.num_ctas)
            generation_offset = ticket - generation * cutlass.Int32(self.num_ctas)
            is_last = cutlass.Int32(0)
            if generation_offset == cutlass.Int32(self.num_ctas - 1):
                is_last = cutlass.Int32(1)
            last_cta_ptr.store(is_last)
        pipeline.agent_sync(pipeline.Agent.ThreadBlock)

        if cutlass.Int32(last_cta_ptr.load()) != cutlass.Int32(0):
            local_expert = tidx
            while local_expert < self.num_local_experts:
                (global_counts_ptr + local_expert).store(cutlass.Int32(0))
                local_expert = local_expert + self.threads_per_cta

    @cute.kernel
    def routing_indices_coop_kernel(
        self,
        gSelectedExperts: cute.Tensor,
        gTileExperts: cute.Tensor,
        gTileLimits: cute.Tensor,
        gExpandedToPermuted: cute.Tensor,
        gPermutedToExpanded: cute.Tensor,
        gTotalPadded: cute.Tensor,
        gNumTiles: cute.Tensor,
        gGlobalCounts: cute.Tensor,
        gGlobalOffsets: cute.Tensor,
        gGridSync: cute.Tensor,
    ):
        # Cooperative counting sort: histogram, padded scan and tile metadata,
        # then scatter expanded pairs and write both mapping directions.
        tidx = cute.arch.thread_idx()[0]
        cta_rank = cute.arch.block_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        selected_ptr = gSelectedExperts.iterator.raw_ptr()
        tile_expert_ptr = gTileExperts.iterator.raw_ptr()
        tile_limit_ptr = gTileLimits.iterator.raw_ptr()
        e2p_ptr = gExpandedToPermuted.iterator.raw_ptr()
        p2e_ptr = gPermutedToExpanded.iterator.raw_ptr()
        total_ptr = gTotalPadded.iterator.raw_ptr()
        num_tiles_ptr = gNumTiles.iterator.raw_ptr()
        global_counts_ptr = gGlobalCounts.iterator.raw_ptr()
        global_offsets_ptr = gGlobalOffsets.iterator.raw_ptr()
        grid_sync_ptr = gGridSync.iterator.raw_ptr()
        grid_generation = cutlass.Int32(0)

        smem = cutlass_memory.SmemAllocator()
        sCounts = smem.allocate_array(cutlass.Int32, self.num_local_experts)
        sScan0 = smem.allocate_array(cutlass.Int32, self.num_local_experts)
        sScan1 = smem.allocate_array(cutlass.Int32, self.num_local_experts)
        sWarpTotals = smem.allocate_array(cutlass.Int32, self.scan_warps)
        sWarpBases = smem.allocate_array(cutlass.Int32, self.scan_warps)

        expert_idx = tidx
        while expert_idx < self.num_local_experts:
            (sCounts + expert_idx).store(cutlass.Int32(0))
            expert_idx = expert_idx + self.threads_per_cta

        pipeline.agent_sync(pipeline.Agent.ThreadBlock)

        pair_end = cutlass.Int32(gSelectedExperts.shape[0]) * cutlass.Int32(self.top_k)
        if cutlass.const_expr(self.contiguous_windows):
            pair_end = cutlass.min(
                pair_end,
                (cta_rank + cutlass.Int32(1)) * cutlass.Int32(self.pairs_per_cta),
            )

        # Wait immediately before the first input load.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # Count local experts while caching each pair's CTA-local rank.
        cached_routes = cutlass.Array(cutlass.Int32, self.routes_per_thread)
        for iteration in cutlass.range_constexpr(self.routes_per_thread):
            cached_route = cutlass.Int32(-1)
            pair_idx = self._pair_at(cta_rank, tidx, cutlass.Int32(iteration))
            if pair_idx < pair_end:
                selected_expert = cutlass.Int32((selected_ptr + pair_idx).load())
                local_expert = selected_expert - cutlass.Int32(self.local_expert_offset)
                if local_expert >= cutlass.Int32(0) and local_expert < cutlass.Int32(
                    self.num_local_experts
                ):
                    local_rank = cute.arch.atomic_add(
                        sCounts + local_expert,
                        cutlass.Int32(1),
                        scope="cta",
                    )
                    cached_route = (local_rank << cutlass.Int32(10)) | local_expert
                else:
                    (e2p_ptr + pair_idx).store(cutlass.Int32(-1))
            cached_routes[iteration] = cached_route

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        pipeline.agent_sync(pipeline.Agent.ThreadBlock)

        if cutlass.const_expr(self.num_ctas > 1):
            expert_idx = tidx
            while expert_idx < self.num_local_experts:
                block_offset = cute.arch.atomic_add(
                    global_counts_ptr + expert_idx,
                    cutlass.Int32((sCounts + expert_idx).load()),
                    sem="relaxed",
                    scope="gpu",
                )
                (
                    global_offsets_ptr + cta_rank * self.num_local_experts + expert_idx
                ).store(block_offset)
                expert_idx = expert_idx + self.threads_per_cta
            pipeline.agent_sync(pipeline.Agent.ThreadBlock)

        if cutlass.const_expr(self.cluster_sort):
            pipeline.agent_sync(
                pipeline.Agent.ThreadBlockCluster,
                is_relaxed=True,
            )
        elif cutlass.const_expr(self.num_ctas > 1):
            if tidx == 0:
                ticket = cute.arch.atomic_add(
                    grid_sync_ptr,
                    cutlass.Int32(1),
                    sem="acq_rel",
                    scope="gpu",
                )
                grid_generation = ticket // cutlass.Int32(self.num_ctas)
                if cta_rank == cutlass.Int32(0):
                    target = (grid_generation + cutlass.Int32(1)) * cutlass.Int32(
                        self.num_ctas
                    )
                    completed = ticket + cutlass.Int32(1)
                    while completed < target:
                        completed = cutlass.Int32(
                            cute.arch.atomic_add(
                                grid_sync_ptr,
                                cutlass.Int32(0),
                                sem="acquire",
                                scope="gpu",
                            )
                        )
                    if cutlass.const_expr(self.parallel_scan):
                        cute.arch.atomic_add(
                            grid_sync_ptr + 1,
                            cutlass.Int32(1),
                            sem="release",
                            scope="gpu",
                        )
                elif cutlass.const_expr(self.parallel_scan):
                    released_generation = cutlass.Int32(
                        cute.arch.atomic_add(
                            grid_sync_ptr + 1,
                            cutlass.Int32(0),
                            sem="acquire",
                            scope="gpu",
                        )
                    )
                    while released_generation < grid_generation + cutlass.Int32(1):
                        released_generation = cutlass.Int32(
                            cute.arch.atomic_add(
                                grid_sync_ptr + 1,
                                cutlass.Int32(0),
                                sem="acquire",
                                scope="gpu",
                            )
                        )
            pipeline.agent_sync(pipeline.Agent.ThreadBlock)

        if cutlass.const_expr(self.parallel_scan):
            # Compute expert offsets in every CTA.
            self._inclusive_padded_scan(
                tidx,
                lane_idx,
                warp_idx,
                sCounts,
                sScan0,
                sScan1,
                sWarpTotals,
                sWarpBases,
                global_counts_ptr,
            )
        else:
            if cta_rank == cutlass.Int32(0):
                self._inclusive_padded_scan(
                    tidx,
                    lane_idx,
                    warp_idx,
                    sCounts,
                    sScan0,
                    sScan1,
                    sWarpTotals,
                    sWarpBases,
                    global_counts_ptr,
                )

        if cutlass.const_expr(self.parallel_scan):
            # The final loader clears counts for the next launch.
            self._reset_counts_after_load(
                tidx,
                grid_sync_ptr + 2,
                sWarpBases,
                global_counts_ptr,
            )

            local_expert = tidx
            while local_expert < self.num_local_experts:
                padded_start = cutlass.Int32(0)
                if local_expert > cutlass.Int32(0):
                    padded_start = cutlass.Int32((sScan0 + local_expert - 1).load())
                count = cutlass.Int32((sCounts + local_expert).load())

                expert_tiles = (
                    count + cutlass.Int32(self.tile_size - 1)
                ) // cutlass.Int32(self.tile_size)
                tile_start_idx = padded_start // cutlass.Int32(self.tile_size)
                expert_tile = cta_rank
                while expert_tile < expert_tiles:
                    tile_idx = tile_start_idx + expert_tile
                    tile_start = padded_start + expert_tile * cutlass.Int32(
                        self.tile_size
                    )
                    (tile_expert_ptr + tile_idx).store(local_expert)
                    (tile_limit_ptr + tile_idx).store(
                        cutlass.min(
                            tile_start + cutlass.Int32(self.tile_size),
                            padded_start + count,
                        )
                    )
                    expert_tile = expert_tile + self.num_ctas
                local_expert = local_expert + self.threads_per_cta
        else:
            if cta_rank == cutlass.Int32(0):
                local_expert = tidx
                while local_expert < self.num_local_experts:
                    padded_start = cutlass.Int32(0)
                    if local_expert > cutlass.Int32(0):
                        padded_start = cutlass.Int32((sScan0 + local_expert - 1).load())
                    count = cutlass.Int32((sCounts + local_expert).load())

                    if cutlass.const_expr(self.num_ctas > 1):
                        (
                            global_counts_ptr + self.num_local_experts + local_expert
                        ).store(padded_start)
                        (global_counts_ptr + local_expert).store(cutlass.Int32(0))

                    expert_tiles = (
                        count + cutlass.Int32(self.tile_size - 1)
                    ) // cutlass.Int32(self.tile_size)
                    tile_start_idx = padded_start // cutlass.Int32(self.tile_size)
                    expert_tile = cutlass.Int32(0)
                    while expert_tile < expert_tiles:
                        tile_idx = tile_start_idx + expert_tile
                        tile_start = padded_start + expert_tile * cutlass.Int32(
                            self.tile_size
                        )
                        (tile_expert_ptr + tile_idx).store(local_expert)
                        (tile_limit_ptr + tile_idx).store(
                            cutlass.min(
                                tile_start + cutlass.Int32(self.tile_size),
                                padded_start + count,
                            )
                        )
                        expert_tile = expert_tile + cutlass.Int32(1)
                    local_expert = local_expert + self.threads_per_cta

        if cta_rank == cutlass.Int32(0) and tidx == 0:
            total = cutlass.Int32((sScan0 + self.num_local_experts - 1).load())
            total_ptr.store(total)
            num_tiles_ptr.store(total // cutlass.Int32(self.tile_size))

        if not cutlass.const_expr(self.parallel_scan):
            if cutlass.const_expr(self.cluster_sort):
                pipeline.agent_sync(pipeline.Agent.ThreadBlock)
                pipeline.agent_sync(
                    pipeline.Agent.ThreadBlockCluster,
                    is_relaxed=True,
                )
            elif cutlass.const_expr(self.num_ctas > 1):
                pipeline.agent_sync(pipeline.Agent.ThreadBlock)
                if tidx == 0:
                    if cta_rank == cutlass.Int32(0):
                        cute.arch.atomic_add(
                            grid_sync_ptr + 1,
                            cutlass.Int32(1),
                            sem="release",
                            scope="gpu",
                        )
                    else:
                        metadata_generation = cutlass.Int32(
                            cute.arch.atomic_add(
                                grid_sync_ptr + 1,
                                cutlass.Int32(0),
                                sem="acquire",
                                scope="gpu",
                            )
                        )
                        while metadata_generation < grid_generation + cutlass.Int32(1):
                            metadata_generation = cutlass.Int32(
                                cute.arch.atomic_add(
                                    grid_sync_ptr + 1,
                                    cutlass.Int32(0),
                                    sem="acquire",
                                    scope="gpu",
                                )
                            )
                pipeline.agent_sync(pipeline.Agent.ThreadBlock)
                _ = cute.arch.atomic_add(
                    grid_sync_ptr + 1,
                    cutlass.Int32(0),
                    sem="acquire",
                    scope="gpu",
                )
            else:
                pipeline.agent_sync(pipeline.Agent.ThreadBlock)

        # Place each valid pair in its expert-major segment and record the inverse.
        for iteration in cutlass.range_constexpr(self.routes_per_thread):
            pair_idx = self._pair_at(cta_rank, tidx, cutlass.Int32(iteration))
            cached_route = cached_routes[iteration]
            if pair_idx < pair_end and cached_route >= cutlass.Int32(0):
                route_expert = cached_route & cutlass.Int32(0x3FF)
                local_rank = cached_route >> cutlass.Int32(10)
                if cutlass.const_expr(self.parallel_scan or self.num_ctas == 1):
                    expert_offset = cutlass.Int32(0)
                    if route_expert > cutlass.Int32(0):
                        expert_offset = cutlass.Int32(
                            (sScan0 + route_expert - 1).load()
                        )
                else:
                    expert_offset = cutlass.Int32(
                        (
                            global_counts_ptr + self.num_local_experts + route_expert
                        ).load()
                    )
                if cutlass.const_expr(self.num_ctas > 1):
                    expert_offset = expert_offset + cutlass.Int32(
                        (
                            global_offsets_ptr
                            + cta_rank * self.num_local_experts
                            + route_expert
                        ).load()
                    )
                permuted_idx = expert_offset + local_rank
                (e2p_ptr + pair_idx).store(permuted_idx)
                (p2e_ptr + permuted_idx).store(pair_idx)


__all__ = ["MoeSortKernel"]
