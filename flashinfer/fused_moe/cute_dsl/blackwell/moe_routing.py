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

"""Fused CuTe DSL top-k selection and MoE routing.

Algorithm:
  1. One warp per token finds each top-k expert in descending score order with
     a repeated argmax, then normalizes the selected scores with softmax.
  2. An atomic histogram counts routed pairs per expert.
  3. Tiled routing rounds each count to ``tile_size``, prefix-sums the padded
     counts, and performs a counting-sort-style scatter into expert-major rows.
     Fixed-slot routing instead appends directly to ``capacity`` slots per expert.

Tensor layouts (T=tokens, E=experts, K=top-k):
  - ``scores`` is row-major [T, E]. ``top_values`` and ``top_indices`` are
    row-major [T, K]. Expanded pair ``(token, k)`` has index ``token * K + k``.
  - Tiled outputs are ``tile_experts[max_tiles]``,
    ``tile_limits[max_tiles]`` (exclusive valid row end),
    ``permuted_to_expanded[padded_m]``, and ``num_tiles[1]``. Padding rows in
    ``permuted_to_expanded`` are unspecified.
  - The optional ``expanded_to_permuted[T, K]`` is the inverse map; ``-1``
    marks a non-local pair.
  - Fixed-slot outputs are ``token_ids[E, capacity]``, ``expert_counts[E]``,
    ``slot_to_expanded[E, capacity]``, and ``expanded_to_slot[T, K]``.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_memory
import cutlass.pipeline as pipeline
from cutlass._mlir.dialects import llvm

LOG2_E = 1.4426950408889634
MULTI_CTA_TOKEN_THRESHOLD = 32
CONTIGUOUS_ROUTE_WINDOW_MIN_TOKENS = 65536


def _half_bits(value):
    """Return the raw 16-bit representation of an fp16/bf16 scalar."""
    return cutlass.Uint32(llvm.bitcast(cutlass.Uint16.mlir_type, value.ir_value()))


class MoeRoutingKernel:
    """Select top-k experts and build one grouped-GEMM routing layout."""

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        *,
        padded_m: int | None = None,
        tile_size: int | None = None,
        capacity: int | None = None,
        use_pdl: bool = True,
        compact_topk: bool = False,
        emit_expanded_to_permuted: bool = False,
        max_ctas: int = 128,
    ):
        self.fixed_slot = capacity is not None
        self.emit_expanded_to_permuted = (
            emit_expanded_to_permuted and not self.fixed_slot
        )
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if not 1 <= top_k <= min(num_experts, 32):
            raise ValueError("top_k must be in [1, min(num_experts, 32)]")
        if not 1 <= num_experts <= 1024:
            raise ValueError("routing supports 1..1024 global experts")
        if max_ctas < 1:
            raise ValueError("max_ctas must be positive")
        if self.fixed_slot:
            if capacity <= 0 or padded_m is not None or tile_size is not None:
                raise ValueError("fixed routing requires only a positive capacity")
        elif (
            padded_m is None
            or tile_size is None
            or tile_size <= 0
            or padded_m % tile_size
        ):
            raise ValueError("tiled routing requires aligned padded_m and tile_size")

        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_pairs = num_tokens * top_k
        self.padded_m = padded_m or 0
        self.tile_size = tile_size or 0
        self.capacity = capacity or 0
        self.compact_topk = compact_topk
        self.threads_per_warp = 32
        self.single_token_direct = not self.fixed_slot and num_tokens == 1
        self.multi_cta = not self.fixed_slot and num_tokens > MULTI_CTA_TOKEN_THRESHOLD
        self.use_pdl = use_pdl
        if self.single_token_direct:
            self.warps_per_cta = 1
        elif self.multi_cta:
            self.warps_per_cta = (
                32
                if num_tokens > 1024 or num_tokens == 512
                else 16
                if num_tokens > 64
                else 8
            )
        elif not self.fixed_slot and num_tokens > 32:
            self.warps_per_cta = 32
        elif not self.fixed_slot and num_tokens > 8:
            self.warps_per_cta = 16
        elif not self.fixed_slot:
            required_warps = max(
                num_tokens,
                (num_experts + self.threads_per_warp - 1) // self.threads_per_warp,
            )
            self.warps_per_cta = 1
            while self.warps_per_cta < required_warps:
                self.warps_per_cta *= 2
        else:
            self.warps_per_cta = 8
        if self.multi_cta:
            self.warps_per_cta = max(
                self.warps_per_cta,
                (num_experts + self.threads_per_warp - 1) // self.threads_per_warp,
            )
        self.threads_per_cta = self.warps_per_cta * self.threads_per_warp
        self.scan_warps = max(
            1,
            (self.num_experts + self.threads_per_warp - 1) // self.threads_per_warp,
        )
        target_ctas = (num_tokens + self.warps_per_cta - 1) // self.warps_per_cta
        self.num_ctas = (
            min(
                (max_ctas if num_tokens > 1024 else 32),
                target_ctas,
            )
            if self.multi_cta
            else 1
        )
        self.contiguous_windows = self.num_tokens >= CONTIGUOUS_ROUTE_WINDOW_MIN_TOKENS
        self.rows_per_cta = (
            (
                (self.num_tokens + self.num_ctas - 1) // self.num_ctas
                + self.warps_per_cta
                - 1
            )
            // self.warps_per_cta
            * self.warps_per_cta
        )
        self.values_per_lane = (
            num_experts + self.threads_per_warp - 1
        ) // self.threads_per_warp
        self.parallel_scatter = self.multi_cta and self.num_pairs > 2048
        self.cta_base_scatter = self.parallel_scatter and self.num_tokens > 2048
        self.cluster_routing = (
            self.multi_cta and 128 <= self.num_tokens <= 512 and self.num_ctas <= 16
        )
        self.cache_routed_experts = not self.fixed_slot and (
            self.single_token_direct or self.parallel_scatter or self.num_tokens > 512
        )
        self.cached_expert_slots = (
            self.num_pairs if self.cache_routed_experts and not self.multi_cta else 1
        )

    @cute.jit
    def __call__(
        self,
        mScores: cute.Tensor,
        mTopValues: cute.Tensor,
        mTopIndices: cute.Tensor,
        mOutput0: cute.Tensor,
        mOutput1: cute.Tensor,
        mOutput2: cute.Tensor,
        mOutput3: cute.Tensor,
        mExpandedToPermuted: cute.Tensor,
        mGlobalCounts: cute.Tensor,
        mGlobalCursors: cute.Tensor,
        mGlobalRoutedExperts: cute.Tensor,
        mGridSync: cute.Tensor,
        stream: cuda.CUstream,
    ):
        launch = self.routing_kernel(
            mScores,
            mTopValues,
            mTopIndices,
            mOutput0,
            mOutput1,
            mOutput2,
            mOutput3,
            mExpandedToPermuted,
            mGlobalCounts,
            mGlobalCursors,
            mGlobalRoutedExperts,
            mGridSync,
        )
        if cutlass.const_expr(self.cluster_routing):
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
            )

    @cute.jit
    def _select_and_route_row(
        self,
        scores_ptr: cute.Pointer,
        values_ptr: cute.Pointer,
        indices_ptr: cute.Pointer,
        output0_ptr: cute.Pointer,
        output2_ptr: cute.Pointer,
        output3_ptr: cute.Pointer,
        expanded_to_permuted_ptr: cute.Pointer,
        sCounts: cute.Pointer,
        sRoutedExperts: cute.Pointer,
        row_idx: cutlass.Int32,
        lane_idx: cutlass.Int32,
    ):
        # Repeated warp argmax selects one expert at a time; ties prefer the
        # lower expert ID. Lane 0 then normalizes and records the routed pairs.
        row_offset = row_idx * self.num_experts
        local_values = cute.make_rmem_tensor((self.values_per_lane,), cutlass.Float32)
        local_indices = cute.make_rmem_tensor(
            (1 if self.compact_topk else self.values_per_lane,),
            cutlass.Int32,
        )
        local_keys = cute.make_rmem_tensor(
            (self.values_per_lane if self.compact_topk else 1,),
            cutlass.Uint32,
        )
        selected_indices = cute.make_rmem_tensor((self.top_k,), cutlass.Int32)
        selected_values = cute.make_rmem_tensor((self.top_k,), cutlass.Float32)

        for item_idx in cutlass.range_constexpr(self.values_per_lane):
            candidate_idx = lane_idx + item_idx * self.threads_per_warp
            candidate_value = cutlass.Float32(float("-inf"))
            candidate_key = cutlass.Uint32(0)
            if candidate_idx < self.num_experts:
                loaded_score = (scores_ptr + row_offset + candidate_idx).load()
                candidate_value = cutlass.Float32(loaded_score)
                if cutlass.const_expr(self.compact_topk):
                    score_bits = _half_bits(loaded_score)
                    ordered_bits = score_bits ^ cutlass.Uint32(0x8000)
                    if score_bits & cutlass.Uint32(0x8000):
                        ordered_bits = score_bits ^ cutlass.Uint32(0xFFFF)
                    candidate_key = (ordered_bits << cutlass.Uint32(10)) | (
                        cutlass.Uint32(1023) - cutlass.Uint32(candidate_idx)
                    )
            local_values[item_idx] = candidate_value
            if cutlass.const_expr(self.compact_topk):
                local_keys[item_idx] = candidate_key
            else:
                local_indices[item_idx] = cutlass.Int32(candidate_idx)

        for selected_idx in cutlass.range_constexpr(self.top_k):
            best_value = cutlass.Float32(float("-inf"))
            best_index = cutlass.Int32(0)
            if cutlass.const_expr(self.compact_topk):
                best_key = local_keys[0]
                for item_idx in cutlass.range_constexpr(1, self.values_per_lane):
                    best_key = cute.math.max(best_key, local_keys[item_idx])
                for offset in (16, 8, 4, 2, 1):
                    best_key = cute.math.max(
                        best_key,
                        cute.arch.shuffle_sync_bfly(best_key, offset=offset),
                    )
                best_index = cutlass.Int32(
                    cutlass.Uint32(1023) - (best_key & cutlass.Uint32(0x3FF))
                )
                lane_value = cutlass.Float32(float("-inf"))
                for item_idx in cutlass.range_constexpr(self.values_per_lane):
                    candidate_index = cutlass.Int32(
                        lane_idx + item_idx * self.threads_per_warp
                    )
                    if candidate_index == best_index:
                        lane_value = local_values[item_idx]
                best_value = cute.arch.shuffle_sync(
                    lane_value,
                    best_index & cutlass.Int32(31),
                )
            else:
                best_value = local_values[0]
                best_index = local_indices[0]
                for item_idx in cutlass.range_constexpr(1, self.values_per_lane):
                    candidate_value = local_values[item_idx]
                    candidate_index = local_indices[item_idx]
                    if candidate_value > best_value:
                        best_value = candidate_value
                        best_index = candidate_index
                    elif candidate_value == best_value:
                        if candidate_index < best_index:
                            best_index = candidate_index
                for offset in (16, 8, 4, 2, 1):
                    candidate_value = cute.arch.shuffle_sync_bfly(
                        best_value, offset=offset
                    )
                    candidate_index = cute.arch.shuffle_sync_bfly(
                        best_index, offset=offset
                    )
                    if candidate_value > best_value:
                        best_value = candidate_value
                        best_index = candidate_index
                    elif candidate_value == best_value:
                        if candidate_index < best_index:
                            best_index = candidate_index

            selected_indices[selected_idx] = best_index
            selected_values[selected_idx] = best_value
            if lane_idx == 0:
                output_idx = row_idx * self.top_k + selected_idx
                (indices_ptr + output_idx).store(best_index)
            for item_idx in cutlass.range_constexpr(self.values_per_lane):
                candidate_index = cutlass.Int32(
                    lane_idx + item_idx * self.threads_per_warp
                )
                if cutlass.const_expr(not self.compact_topk):
                    candidate_index = local_indices[item_idx]
                if candidate_index == best_index:
                    local_values[item_idx] = cutlass.Float32(float("-inf"))
                    if cutlass.const_expr(self.compact_topk):
                        local_keys[item_idx] = cutlass.Uint32(0)

        if lane_idx == 0:
            row_max = selected_values[0]
            row_sum = cutlass.Float32(0.0)
            log2_e = cutlass.Float32(LOG2_E)
            for selected_idx in cutlass.range_constexpr(self.top_k):
                row_sum = row_sum + cute.math.exp2(
                    (selected_values[selected_idx] - row_max) * log2_e,
                    fastmath=True,
                )
            inv_sum = cutlass.Float32(1.0) / row_sum
            for selected_idx in cutlass.range_constexpr(self.top_k):
                weight = (
                    cute.math.exp2(
                        (selected_values[selected_idx] - row_max) * log2_e,
                        fastmath=True,
                    )
                    * inv_sum
                )
                (values_ptr + row_idx * self.top_k + selected_idx).store(weight)

            for selected_idx in cutlass.range_constexpr(self.top_k):
                routed_expert = selected_indices[selected_idx]
                pair_idx = row_idx * self.top_k + selected_idx
                if routed_expert >= cutlass.Int32(0) and routed_expert < cutlass.Int32(
                    self.num_experts
                ):
                    expert_rank = cutlass.Int32(0)
                    if cutlass.const_expr(not self.single_token_direct):
                        expert_rank = cute.arch.atomic_add(
                            sCounts + routed_expert,
                            cutlass.Int32(1),
                            scope="cta",
                        )
                    if cutlass.const_expr(self.fixed_slot):
                        if expert_rank < cutlass.Int32(self.capacity):
                            slot_idx = (
                                routed_expert * cutlass.Int32(self.capacity)
                                + expert_rank
                            )
                            (output0_ptr + slot_idx).store(cutlass.Int64(row_idx))
                            (output2_ptr + slot_idx).store(pair_idx)
                            (output3_ptr + pair_idx).store(slot_idx)
                    elif cutlass.const_expr(self.cache_routed_experts):
                        cached_route = routed_expert
                        if cutlass.const_expr(self.parallel_scatter):
                            cached_route = (
                                expert_rank << cutlass.Int32(10)
                            ) | routed_expert
                        (sRoutedExperts + pair_idx).store(cached_route)
                else:
                    if cutlass.const_expr(self.cache_routed_experts):
                        (sRoutedExperts + pair_idx).store(routed_expert)
                    # Non-local pairs never reach the scatter phase, so
                    # mark them masked here
                    if cutlass.const_expr(
                        self.emit_expanded_to_permuted and not self.single_token_direct
                    ):
                        (expanded_to_permuted_ptr + pair_idx).store(cutlass.Int32(-1))

    @cute.jit
    def _scatter_pairs(
        self,
        indices_ptr: cute.Pointer,
        output2_ptr: cute.Pointer,
        expanded_to_permuted_ptr: cute.Pointer,
        padded_bases: cute.Pointer,
        routed_cursors: cute.Pointer,
        routed_experts: cute.Pointer,
        scatter_pair_idx: cutlass.Int32,
        scatter_stride: cutlass.Int32,
        num_pairs: cutlass.Int32,
    ):
        # Scatter expanded pairs into their expert-major, tile-padded segments.
        while scatter_pair_idx < num_pairs:
            routed_expert = cutlass.Int32(0)
            expert_rank = cutlass.Int32(0)
            if cutlass.const_expr(self.cache_routed_experts):
                cached_route = cutlass.Int32((routed_experts + scatter_pair_idx).load())
                if cutlass.const_expr(self.parallel_scatter):
                    routed_expert = cached_route & cutlass.Int32(0x3FF)
                    expert_rank = cached_route >> cutlass.Int32(10)
                else:
                    routed_expert = cached_route
            else:
                routed_expert = cutlass.Int32((indices_ptr + scatter_pair_idx).load())
            if routed_expert >= cutlass.Int32(0) and routed_expert < cutlass.Int32(
                self.num_experts
            ):
                dst_idx = cutlass.Int32(0)
                if cutlass.const_expr(self.parallel_scatter):
                    if cutlass.const_expr(self.cta_base_scatter):
                        token_idx = scatter_pair_idx // cutlass.Int32(self.top_k)
                        owner_cta = cutlass.Int32(0)
                        if cutlass.const_expr(self.contiguous_windows):
                            owner_cta = token_idx // cutlass.Int32(self.rows_per_cta)
                        else:
                            owner_cta = (
                                token_idx // cutlass.Int32(self.warps_per_cta)
                            ) % cutlass.Int32(self.num_ctas)
                        dst_idx = (
                            cutlass.Int32(
                                (padded_bases + self.num_experts + routed_expert).load()
                            )
                            + cutlass.Int32(
                                (
                                    routed_cursors
                                    + owner_cta * self.num_experts
                                    + routed_expert
                                ).load()
                            )
                            + expert_rank
                        )
                    else:
                        token_idx = scatter_pair_idx // cutlass.Int32(self.top_k)
                        owner_cta = token_idx // cutlass.Int32(self.warps_per_cta)
                        if cutlass.const_expr(
                            self.num_tokens > self.warps_per_cta * self.num_ctas
                        ):
                            owner_cta = owner_cta % cutlass.Int32(self.num_ctas)
                        dst_idx = (
                            cutlass.Int32(
                                (
                                    routed_cursors
                                    + owner_cta * self.num_experts
                                    + routed_expert
                                ).load()
                            )
                            + expert_rank
                        )
                else:
                    dst_idx = cute.arch.atomic_add(
                        routed_cursors + routed_expert,
                        cutlass.Int32(1),
                        scope="cta",
                    )
                (output2_ptr + dst_idx).store(scatter_pair_idx)
                if cutlass.const_expr(self.emit_expanded_to_permuted):
                    (expanded_to_permuted_ptr + scatter_pair_idx).store(dst_idx)
            scatter_pair_idx = scatter_pair_idx + scatter_stride

    @cute.jit
    def _route_single_token(
        self,
        scores_ptr: cute.Pointer,
        values_ptr: cute.Pointer,
        indices_ptr: cute.Pointer,
        output0_ptr: cute.Pointer,
        output1_ptr: cute.Pointer,
        output2_ptr: cute.Pointer,
        output3_ptr: cute.Pointer,
        expanded_to_permuted_ptr: cute.Pointer,
        routed_counts: cute.Pointer,
        routed_experts: cute.Pointer,
        pipeline_barriers: cute.Pointer,
        tidx: cutlass.Int32,
        lane_idx: cutlass.Int32,
        warp_idx: cutlass.Int32,
    ):
        # For one token, order the K selected experts directly and emit one
        # partially filled tile per expert instead of building a histogram.
        all_threads = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_cta
        )
        one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        selection_done_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=pipeline_barriers,
            num_stages=1,
            producer_group=all_threads,
            consumer_group=one_thread,
            defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()
        pipeline.agent_sync(pipeline.Agent.ThreadBlock)

        if warp_idx == cutlass.Int32(0):
            self._select_and_route_row(
                scores_ptr,
                values_ptr,
                indices_ptr,
                output0_ptr,
                output2_ptr,
                output3_ptr,
                expanded_to_permuted_ptr,
                routed_counts,
                routed_experts,
                cutlass.Int32(0),
                lane_idx,
            )
        selection_producer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        selection_done_pipeline.producer_commit(selection_producer)

        if tidx == cutlass.Int32(0):
            selection_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            selection_done_pipeline.consumer_wait(selection_consumer)
            if cutlass.const_expr(self.emit_expanded_to_permuted):
                # Mask every pair first; routed pairs are overwritten below.
                for pair_idx in cutlass.range_constexpr(self.top_k):
                    (expanded_to_permuted_ptr + pair_idx).store(cutlass.Int32(-1))
            previous_expert = cutlass.Int32(-1)
            num_tiles = cutlass.Int32(0)
            for _ in cutlass.range_constexpr(self.top_k):
                next_expert = cutlass.Int32(self.num_experts)
                next_pair = cutlass.Int32(-1)
                for pair_idx in cutlass.range_constexpr(self.top_k):
                    routed_expert = cutlass.Int32((routed_experts + pair_idx).load())
                    if (
                        routed_expert > previous_expert
                        and routed_expert < next_expert
                        and routed_expert < cutlass.Int32(self.num_experts)
                    ):
                        next_expert = routed_expert
                        next_pair = cutlass.Int32(pair_idx)
                if next_pair >= cutlass.Int32(0):
                    padded_start = num_tiles * cutlass.Int32(self.tile_size)
                    (output0_ptr + num_tiles).store(next_expert)
                    (output1_ptr + num_tiles).store(padded_start + cutlass.Int32(1))
                    (output2_ptr + padded_start).store(next_pair)
                    if cutlass.const_expr(self.emit_expanded_to_permuted):
                        (expanded_to_permuted_ptr + next_pair).store(padded_start)
                    num_tiles = num_tiles + cutlass.Int32(1)
                    previous_expert = next_expert
            output3_ptr.store(num_tiles)
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def routing_kernel(
        self,
        gScores: cute.Tensor,
        gTopValues: cute.Tensor,
        gTopIndices: cute.Tensor,
        gOutput0: cute.Tensor,
        gOutput1: cute.Tensor,
        gOutput2: cute.Tensor,
        gOutput3: cute.Tensor,
        gExpandedToPermuted: cute.Tensor,
        gGlobalCounts: cute.Tensor,
        gGlobalCursors: cute.Tensor,
        gGlobalRoutedExperts: cute.Tensor,
        gGridSync: cute.Tensor,
    ):
        # The tiled path runs top-k/histogram, padded prefix scan, metadata, and
        # pair scatter in one launch. Fixed-slot mode stops after direct append.
        tidx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_rank = cute.arch.block_idx()[0]

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        smem = cutlass_memory.SmemAllocator()
        sCounts = smem.allocate_array(cutlass.Int32, self.num_experts)
        sCursors = smem.allocate_array(
            cutlass.Int32,
            self.num_experts if not self.fixed_slot else 1,
        )
        sRoutedExperts = smem.allocate_array(cutlass.Int32, self.cached_expert_slots)
        sWarpTotals = smem.allocate_array(
            cutlass.Int32,
            self.scan_warps if not self.fixed_slot else 1,
        )
        sWarpBases = smem.allocate_array(
            cutlass.Int32,
            self.scan_warps if not self.fixed_slot else 1,
        )
        base_pipeline_barriers = 8 if self.multi_cta else 4
        sPipelineBarriers = smem.allocate_array(
            cutlass.Int64,
            base_pipeline_barriers
            + (6 if not self.fixed_slot else 0)
            + (4 if self.multi_cta else 0)
            + (2 if self.parallel_scatter else 0),
        )
        routed_counts = sCounts
        routed_cursors = sCursors
        routed_experts = sRoutedExperts
        if cutlass.const_expr(self.multi_cta):
            routed_experts = gGlobalRoutedExperts.iterator.raw_ptr()
        global_counts_ptr = gGlobalCounts.iterator.raw_ptr()
        global_cursors_ptr = gGlobalCursors.iterator.raw_ptr()
        grid_sync_ptr = gGridSync.iterator.raw_ptr()

        scores_ptr = gScores.iterator.raw_ptr()
        values_ptr = gTopValues.iterator.raw_ptr()
        indices_ptr = gTopIndices.iterator.raw_ptr()
        output0_ptr = gOutput0.iterator.raw_ptr()
        output1_ptr = gOutput1.iterator.raw_ptr()
        output2_ptr = gOutput2.iterator.raw_ptr()
        output3_ptr = gOutput3.iterator.raw_ptr()
        expanded_to_permuted_ptr = gExpandedToPermuted.iterator.raw_ptr()
        num_tokens = cutlass.Int32(gScores.shape[0])
        num_pairs = num_tokens * cutlass.Int32(self.top_k)

        if cutlass.const_expr(self.single_token_direct):
            self._route_single_token(
                scores_ptr,
                values_ptr,
                indices_ptr,
                output0_ptr,
                output1_ptr,
                output2_ptr,
                output3_ptr,
                expanded_to_permuted_ptr,
                routed_counts,
                routed_experts,
                sPipelineBarriers,
                tidx,
                lane_idx,
                warp_idx,
            )
            return

        is_initializer = cta_rank == cutlass.Int32(0)
        if cutlass.const_expr(self.fixed_slot):
            num_slots: cutlass.Constexpr = self.num_experts * self.capacity
            slot_idx = tidx
            while slot_idx < num_slots:
                if is_initializer:
                    (output0_ptr + slot_idx).store(cutlass.Int64(0))
                    (output2_ptr + slot_idx).store(cutlass.Int32(-self.top_k))
                slot_idx = slot_idx + self.threads_per_cta
            pair_idx = tidx
            while pair_idx < num_pairs:
                if is_initializer:
                    (output3_ptr + pair_idx).store(cutlass.Int32(-1))
                pair_idx = pair_idx + self.threads_per_cta
        # Tiled consumers mask mapping rows with output1's valid-end bound, so
        # padding entries are intentionally left unspecified.
        expert_idx = tidx
        while expert_idx < self.num_experts:
            (routed_counts + expert_idx).store(cutlass.Int32(0))
            if cutlass.const_expr(not self.fixed_slot):
                if is_initializer:
                    (routed_cursors + expert_idx).store(cutlass.Int32(0))
            else:
                (output1_ptr + expert_idx).store(cutlass.Int32(0))
            expert_idx = expert_idx + self.threads_per_cta
        if cutlass.const_expr(not self.fixed_slot):
            if tidx == 0:
                if is_initializer:
                    output3_ptr.store(cutlass.Int32(0))

        all_threads = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_cta
        )
        one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        if cutlass.const_expr(self.multi_cta):
            init_done_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=sPipelineBarriers,
                num_stages=1,
                producer_group=all_threads,
                consumer_group=all_threads,
                defer_sync=True,
            )
            count_done_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=sPipelineBarriers + 4,
                num_stages=1,
                producer_group=all_threads,
                consumer_group=all_threads,
                defer_sync=True,
            )
            metadata_ready_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=sPipelineBarriers + 6,
                num_stages=1,
                producer_group=one_thread,
                consumer_group=all_threads,
                defer_sync=True,
            )
        else:
            count_done_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=sPipelineBarriers,
                num_stages=1,
                producer_group=all_threads,
                consumer_group=one_thread,
                defer_sync=True,
            )
            metadata_ready_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=sPipelineBarriers + 2,
                num_stages=1,
                producer_group=one_thread,
                consumer_group=all_threads,
                defer_sync=True,
            )
        if cutlass.const_expr(not self.fixed_slot):
            scan_barrier_offset: cutlass.Constexpr = 8 if self.multi_cta else 4
            scan_done_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=sPipelineBarriers + scan_barrier_offset,
                num_stages=1,
                producer_group=all_threads,
                consumer_group=one_thread,
                defer_sync=True,
            )
            scan_ready_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=sPipelineBarriers + scan_barrier_offset + 2,
                num_stages=1,
                producer_group=one_thread,
                consumer_group=all_threads,
                defer_sync=True,
            )
            metadata_done_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=sPipelineBarriers + scan_barrier_offset + 4,
                num_stages=1,
                producer_group=all_threads,
                consumer_group=all_threads,
                defer_sync=True,
            )
            if cutlass.const_expr(self.multi_cta):
                publish_done_pipeline = pipeline.PipelineAsync.create(
                    barrier_storage=(sPipelineBarriers + scan_barrier_offset + 6),
                    num_stages=1,
                    producer_group=all_threads,
                    consumer_group=one_thread,
                    defer_sync=True,
                )
                reduce_done_pipeline = pipeline.PipelineAsync.create(
                    barrier_storage=(sPipelineBarriers + scan_barrier_offset + 8),
                    num_stages=1,
                    producer_group=all_threads,
                    consumer_group=all_threads,
                    defer_sync=True,
                )
                if cutlass.const_expr(self.parallel_scatter):
                    scatter_ready_pipeline = pipeline.PipelineAsync.create(
                        barrier_storage=(sPipelineBarriers + scan_barrier_offset + 10),
                        num_stages=1,
                        producer_group=one_thread,
                        consumer_group=all_threads,
                        defer_sync=True,
                    )
        cute.arch.mbarrier_init_fence()
        pipeline.agent_sync(pipeline.Agent.ThreadBlock)

        if cutlass.const_expr(self.multi_cta):
            init_done_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            init_done_pipeline.producer_commit(init_done_producer)
            init_done_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            init_done_pipeline.consumer_wait(init_done_consumer)

        # One warp handles each score row and contributes K pairs to the histogram.
        row_idx = cutlass.Int32(cta_rank) * self.warps_per_cta + cutlass.Int32(warp_idx)
        row_end = num_tokens
        row_stride = cutlass.Int32(self.warps_per_cta * self.num_ctas)
        if cutlass.const_expr(self.contiguous_windows):
            row_idx = cutlass.Int32(cta_rank) * cutlass.Int32(
                self.rows_per_cta
            ) + cutlass.Int32(warp_idx)
            row_end = cutlass.min(
                num_tokens,
                (cutlass.Int32(cta_rank) + cutlass.Int32(1))
                * cutlass.Int32(self.rows_per_cta),
            )
            row_stride = cutlass.Int32(self.warps_per_cta)
        while row_idx < row_end:
            self._select_and_route_row(
                scores_ptr,
                values_ptr,
                indices_ptr,
                output0_ptr,
                output2_ptr,
                output3_ptr,
                expanded_to_permuted_ptr,
                routed_counts,
                routed_experts,
                row_idx,
                lane_idx,
            )
            row_idx += row_stride
        count_done_producer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        count_done_pipeline.producer_commit(count_done_producer)
        if cutlass.const_expr(self.use_pdl and not self.multi_cta):
            cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.fixed_slot):
            if tidx == 0:
                fixed_count_consumer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, 1
                )
                count_done_pipeline.consumer_wait(fixed_count_consumer)
                expert_idx = cutlass.Int32(0)
                while expert_idx < self.num_experts:
                    (output1_ptr + expert_idx).store(
                        cutlass.Int32((sCounts + expert_idx).load())
                    )
                    expert_idx = expert_idx + cutlass.Int32(1)
        else:
            if cutlass.const_expr(self.multi_cta):
                publish_count_consumer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, 1
                )
                count_done_pipeline.consumer_wait(publish_count_consumer)
                publish_expert = cutlass.Int32(tidx)
                while publish_expert < self.num_experts:
                    if cutlass.const_expr(self.cta_base_scatter):
                        block_offset = cute.arch.atomic_add(
                            global_counts_ptr + publish_expert,
                            cutlass.Int32((sCounts + publish_expert).load()),
                            sem="relaxed",
                            scope="gpu",
                        )
                        (
                            global_cursors_ptr
                            + cta_rank * self.num_experts
                            + publish_expert
                        ).store(block_offset)
                    else:
                        (
                            global_counts_ptr
                            + cta_rank * self.num_experts
                            + publish_expert
                        ).store((sCounts + publish_expert).load())
                    publish_expert = publish_expert + self.threads_per_cta
                publish_done_producer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, 1
                )
                publish_done_pipeline.producer_commit(publish_done_producer)
                if cutlass.const_expr(self.cluster_routing):
                    pipeline.agent_sync(pipeline.Agent.ThreadBlock)
                    pipeline.agent_sync(
                        pipeline.Agent.ThreadBlockCluster,
                        is_relaxed=True,
                    )

            if tidx == 0:
                metadata_ready_producer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, 1
                )
                if cutlass.const_expr(not self.multi_cta):
                    metadata_count_consumer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, 1
                    )
                    count_done_pipeline.consumer_wait(metadata_count_consumer)
                if cutlass.const_expr(self.multi_cta):
                    if cutlass.const_expr(not self.cluster_routing):
                        publish_done_consumer = pipeline.make_pipeline_state(
                            pipeline.PipelineUserType.Consumer, 1
                        )
                        publish_done_pipeline.consumer_wait(publish_done_consumer)
                        ticket = cute.arch.atomic_add(
                            grid_sync_ptr,
                            cutlass.Int32(1),
                            sem="acq_rel",
                            scope="gpu",
                        )
                        generation = ticket // cutlass.Int32(self.num_ctas)
                        target = (generation + cutlass.Int32(1)) * cutlass.Int32(
                            self.num_ctas
                        )
                        if cta_rank == cutlass.Int32(0):
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
                        else:
                            if cutlass.const_expr(self.parallel_scatter):
                                metadata_generation = cutlass.Int32(
                                    cute.arch.atomic_add(
                                        grid_sync_ptr + 1,
                                        cutlass.Int32(0),
                                        sem="acquire",
                                        scope="gpu",
                                    )
                                )
                                while metadata_generation <= generation:
                                    metadata_generation = cutlass.Int32(
                                        cute.arch.atomic_add(
                                            grid_sync_ptr + 1,
                                            cutlass.Int32(0),
                                            sem="acquire",
                                            scope="gpu",
                                        )
                                    )
                                scatter_ready_producer = pipeline.make_pipeline_state(
                                    pipeline.PipelineUserType.Producer,
                                    1,
                                )
                                scatter_ready_pipeline.producer_commit(
                                    scatter_ready_producer
                                )
                if cta_rank == cutlass.Int32(0):
                    metadata_ready_pipeline.producer_commit(metadata_ready_producer)

            if cta_rank == cutlass.Int32(0):
                counts_ready_consumer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, 1
                )
                metadata_ready_pipeline.consumer_wait(counts_ready_consumer)
                if cutlass.const_expr(self.multi_cta):
                    reduce_expert = cutlass.Int32(tidx)
                    while reduce_expert < self.num_experts:
                        total = cutlass.Int32(0)
                        if cutlass.const_expr(self.cta_base_scatter):
                            total = cutlass.Int32(
                                (global_counts_ptr + reduce_expert).load()
                            )
                        else:
                            producer_cta = cutlass.Int32(0)
                            while producer_cta < self.num_ctas:
                                total = total + cutlass.Int32(
                                    (
                                        global_counts_ptr
                                        + producer_cta * self.num_experts
                                        + reduce_expert
                                    ).load()
                                )
                                producer_cta = producer_cta + cutlass.Int32(1)
                        (sCounts + reduce_expert).store(total)
                        reduce_expert = reduce_expert + self.threads_per_cta
                    reduce_done_producer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, 1
                    )
                    reduce_done_pipeline.producer_commit(reduce_done_producer)
                    reduce_done_consumer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, 1
                    )
                    reduce_done_pipeline.consumer_wait(reduce_done_consumer)

                # A padded inclusive scan assigns one contiguous row segment to
                # each expert; tile metadata records its valid (unpadded) ends.
                metadata_expert = cutlass.Int32(tidx)
                expert_count = cutlass.Int32(0)
                expert_rows = cutlass.Int32(0)
                if metadata_expert < self.num_experts:
                    expert_count = cutlass.Int32((sCounts + metadata_expert).load())
                    expert_rows = (
                        (expert_count + cutlass.Int32(self.tile_size - 1))
                        // cutlass.Int32(self.tile_size)
                        * cutlass.Int32(self.tile_size)
                    )

                inclusive_rows = expert_rows
                for scan_offset in (1, 2, 4, 8, 16):
                    prior_rows = cute.arch.shuffle_sync_up(inclusive_rows, scan_offset)
                    if lane_idx >= scan_offset:
                        inclusive_rows = inclusive_rows + prior_rows
                if warp_idx < self.scan_warps and lane_idx == self.threads_per_warp - 1:
                    (sWarpTotals + warp_idx).store(inclusive_rows)

                scan_done_producer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, 1
                )
                scan_done_pipeline.producer_commit(scan_done_producer)
                if tidx == 0:
                    scan_done_consumer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, 1
                    )
                    scan_ready_producer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, 1
                    )
                    scan_done_pipeline.consumer_wait(scan_done_consumer)
                    warp_base = cutlass.Int32(0)
                    metadata_warp = cutlass.Int32(0)
                    while metadata_warp < self.scan_warps:
                        (sWarpBases + metadata_warp).store(warp_base)
                        warp_base = warp_base + cutlass.Int32(
                            (sWarpTotals + metadata_warp).load()
                        )
                        metadata_warp = metadata_warp + cutlass.Int32(1)
                    output3_ptr.store(warp_base // cutlass.Int32(self.tile_size))
                    scan_ready_pipeline.producer_commit(scan_ready_producer)

                scan_ready_consumer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, 1
                )
                scan_ready_pipeline.consumer_wait(scan_ready_consumer)
                if metadata_expert < self.num_experts:
                    padded_start = (
                        cutlass.Int32((sWarpBases + warp_idx).load())
                        + inclusive_rows
                        - expert_rows
                    )
                    if cutlass.const_expr(self.parallel_scatter):
                        if cutlass.const_expr(self.cta_base_scatter):
                            (
                                global_counts_ptr + self.num_experts + metadata_expert
                            ).store(padded_start)
                            (global_counts_ptr + metadata_expert).store(
                                cutlass.Int32(0)
                            )
                        else:
                            cta_cursor = padded_start
                            cursor_cta = cutlass.Int32(0)
                            while cursor_cta < self.num_ctas:
                                (
                                    global_cursors_ptr
                                    + cursor_cta * self.num_experts
                                    + metadata_expert
                                ).store(cta_cursor)
                                cta_cursor = cta_cursor + cutlass.Int32(
                                    (
                                        global_counts_ptr
                                        + cursor_cta * self.num_experts
                                        + metadata_expert
                                    ).load()
                                )
                                cursor_cta = cursor_cta + cutlass.Int32(1)
                    else:
                        (routed_cursors + metadata_expert).store(padded_start)
                    expert_tiles = expert_rows // cutlass.Int32(self.tile_size)
                    expert_tile_idx = cutlass.Int32(0)
                    while expert_tile_idx < expert_tiles:
                        tile_idx = (
                            padded_start // cutlass.Int32(self.tile_size)
                            + expert_tile_idx
                        )
                        tile_start = padded_start + expert_tile_idx * cutlass.Int32(
                            self.tile_size
                        )
                        valid_end = cutlass.min(
                            tile_start + cutlass.Int32(self.tile_size),
                            padded_start + expert_count,
                        )
                        (output0_ptr + tile_idx).store(metadata_expert)
                        (output1_ptr + tile_idx).store(valid_end)
                        expert_tile_idx = expert_tile_idx + cutlass.Int32(1)

                metadata_done_producer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, 1
                )
                metadata_done_pipeline.producer_commit(metadata_done_producer)
                metadata_done_consumer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, 1
                )
                metadata_done_pipeline.consumer_wait(metadata_done_consumer)
                if cutlass.const_expr(self.use_pdl and self.multi_cta):
                    cute.arch.griddepcontrol_launch_dependents()
                if cutlass.const_expr(self.parallel_scatter):
                    if tidx == 0:
                        cute.arch.atomic_add(
                            grid_sync_ptr + 1,
                            cutlass.Int32(1),
                            sem="release",
                            scope="gpu",
                        )
                        scatter_ready_producer = pipeline.make_pipeline_state(
                            pipeline.PipelineUserType.Producer, 1
                        )
                        scatter_ready_pipeline.producer_commit(scatter_ready_producer)

            if cutlass.const_expr(self.parallel_scatter):
                if cutlass.const_expr(self.cluster_routing):
                    pipeline.agent_sync(pipeline.Agent.ThreadBlockCluster)
                else:
                    scatter_ready_consumer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, 1
                    )
                    scatter_ready_pipeline.consumer_wait(scatter_ready_consumer)
                self._scatter_pairs(
                    indices_ptr,
                    output2_ptr,
                    expanded_to_permuted_ptr,
                    global_counts_ptr,
                    global_cursors_ptr,
                    routed_experts,
                    cutlass.Int32(cta_rank) * self.threads_per_cta
                    + cutlass.Int32(tidx),
                    cutlass.Int32(self.threads_per_cta * self.num_ctas),
                    num_pairs,
                )
            else:
                if cta_rank == cutlass.Int32(0):
                    self._scatter_pairs(
                        indices_ptr,
                        output2_ptr,
                        expanded_to_permuted_ptr,
                        global_counts_ptr,
                        routed_cursors,
                        routed_experts,
                        cutlass.Int32(tidx),
                        cutlass.Int32(self.threads_per_cta),
                        num_pairs,
                    )


__all__ = ["MoeRoutingKernel"]
