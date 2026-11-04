# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from .block_scan import block_prefix_sum_kernel, fence_acq_rel_cta, warp_scan

"""
top-k varlen utils. could be used by prefill and decode phase.
"""


DETERMINISTIC_BITMAP_MAX_NUM_COLS = 262144


def resolve_deterministic_ordering(max_num_cols: int, deterministic: bool, ordering: str) -> str:
    """Resolve the internal canonical-ordering strategy for one specialization."""
    if ordering not in ("auto", "bitmap", "bitonic"):
        raise ValueError("deterministic_ordering must be 'auto', 'bitmap', or 'bitonic'")
    if not deterministic:
        if ordering != "auto":
            raise ValueError("deterministic_ordering requires deterministic=True")
        return "auto"
    if ordering == "auto":
        return "bitmap" if max_num_cols <= DETERMINISTIC_BITMAP_MAX_NUM_COLS else "bitonic"
    return ordering


@dsl_user_op
def atomicAdd(dst_ptr: cute.Pointer, val: cutlass.Int32, *, loc=None, ip=None) -> cutlass.Int32:
    """System-scope relaxed atomic add (drop-in for the deprecated
    ``cutlass.utils.distributed.atomicAdd``)."""
    return cute.arch.atomic_add(dst_ptr.llvm_ptr, val, sem="relaxed", scope="sys", loc=loc, ip=ip)


def half_as_ushort(half_val):
    """Interpret FP16 value as uint16 bit pattern"""
    return llvm.bitcast(cutlass.Uint16.mlir_type, half_val.ir_value())


def float_as_uint32(float_val):
    """Interpret FP32 value as uint32 bit pattern"""
    return llvm.bitcast(cutlass.Uint32.mlir_type, float_val.ir_value())


class IndexerTopKKernelVarlen:
    def __init__(
        self,
        dtype: cutlass.Numeric,
        max_num_cols: int,
        top_k: int,
        num_copy_bits: int = 256,
        return_val: bool = True,
        enable_multi_cta: bool = False,
        chunk_size_per_cta: int = 16384,
        num_ctas_per_row: int = 1,
        merge_blocks: bool = False,
        tie_break: int = 0,
        deterministic: bool = False,
        deterministic_ordering: str = "auto",
    ):
        self.dtype = dtype
        self.max_num_cols = max_num_cols
        self.top_k = top_k
        self.num_copy_bits = num_copy_bits
        self.enable_multi_cta = enable_multi_cta
        self.chunk_size_per_cta = chunk_size_per_cta
        self.num_ctas_per_row = num_ctas_per_row
        self.merge_blocks = merge_blocks
        self.tie_break = tie_break
        self.deterministic = deterministic
        self.deterministic_ordering = resolve_deterministic_ordering(max_num_cols, deterministic, deterministic_ordering)
        if tie_break not in (0, 1, 2):
            raise ValueError("tie_break must be 0 (none), 1 (small), or 2 (large)")
        if tie_break and merge_blocks:
            raise ValueError("tie_break requires original source-column order, not merged candidates")

        # Note: now we only support top_k <= 2048, we could change the code here to support larger top_k.
        self.indexer_topk_max_k = 2048
        # 8 bits for radix-based filter.
        self.radix = 256

        if cutlass.const_expr(self.dtype == cutlass.Float32):
            self.num_buffer_smem_input_idx = 2
        else:
            self.num_buffer_smem_input_idx = 1

        # 65536 columns (indices 0 through 65535) fit in UINT16.
        if cutlass.const_expr(enable_multi_cta):
            self.per_row_max_num_cols = chunk_size_per_cta * num_ctas_per_row
        else:
            self.per_row_max_num_cols = self.max_num_cols

        # UINT16 is sufficient through column 65535. Bitonic ordering also
        # needs one out-of-range sentinel, so only its exact 65536-column
        # boundary must widen to UINT32; bitmap ordering has no sentinel.
        needs_uint32_sentinel = self.deterministic and self.deterministic_ordering == "bitonic" and self.per_row_max_num_cols == 65536
        if cutlass.const_expr(self.per_row_max_num_cols <= 65536 and not needs_uint32_sentinel):
            self.index_type = cutlass.Uint16
            self.index_sentinel = 0xFFFF
            if cutlass.const_expr(self.num_buffer_smem_input_idx == 2):
                self.max_smem_input_size = 32 * 1024
            else:
                self.max_smem_input_size = 64 * 1024
        else:
            self.index_type = cutlass.Uint32
            self.index_sentinel = 0xFFFFFFFF
            if cutlass.const_expr(self.num_buffer_smem_input_idx == 2):
                self.max_smem_input_size = 16 * 1024
            else:
                self.max_smem_input_size = 32 * 1024

        self.indexer_topk_smem_input_size = min(self.max_smem_input_size, self.max_num_cols)

        if cutlass.const_expr(self.max_num_cols > self.indexer_topk_smem_input_size):
            self.enable_gmem_store = True
        else:
            self.enable_gmem_store = False

        self.return_val = return_val

        self.vec_size = num_copy_bits // dtype.width
        if cutlass.const_expr(dtype not in [cutlass.Float32, cute.BFloat16, cutlass.Float16]):
            raise ValueError(f"Unsupported dtype: {dtype}")

        if cutlass.const_expr(dtype == cutlass.Float32):
            if self.max_num_cols >= self.vec_size * 1024:
                self.num_threads_per_cta = 1024
            else:
                if cutlass.const_expr(self.max_num_cols > 2048 and self.max_num_cols < 8192):
                    self.num_threads_per_cta = 512
                else:
                    self.num_threads_per_cta = 256
        else:
            if self.max_num_cols >= 43008:
                self.num_threads_per_cta = 1024
            else:
                if cutlass.const_expr(self.max_num_cols > 4096 and self.max_num_cols < 43008):
                    self.num_threads_per_cta = 512
                else:
                    self.num_threads_per_cta = 256

        # radix-based filter parameters.
        if cutlass.const_expr(dtype == cutlass.Float32):
            self.ordered_type = cute.Uint32
            self.first_refine_shift = 24
            self.num_refine_rounds = 4
        elif cutlass.const_expr(dtype in [cutlass.Float16, cute.BFloat16]):
            self.ordered_type = cute.Uint16
            self.first_refine_shift = 0
            self.num_refine_rounds = 1

        self.deterministic_sort_size = 1 << (self.top_k - 1).bit_length()
        self.use_deterministic_bitmap = self.deterministic_ordering == "bitmap"

    @cute.jit
    def cta_sync(self, s_cta_mbarrier, sync_phase):
        """Synchronize the CTA with sanitizer-tracked ordering in deterministic mode."""
        if cutlass.const_expr(self.deterministic):
            phase = sync_phase[0]
            cute.arch.mbarrier_arrive(s_cta_mbarrier.iterator)
            cute.arch.mbarrier_wait(s_cta_mbarrier.iterator, phase)
            sync_phase[0] = phase ^ cutlass.Int32(1)
        else:
            cute.arch.barrier()

    @cute.jit
    def deterministic_block_prefix_sum(self, value, scratch, s_cta_mbarrier, sync_phase):
        """Inclusive CTA scan using the deterministic path's tracked mbarrier."""
        tidx, _, _ = cute.arch.thread_idx()
        warp_id = tidx // cutlass.Int32(32)
        lane_id = tidx % cutlass.Int32(32)
        num_warps = self.num_threads_per_cta // 32

        value = warp_scan(value, tidx, lane_id, num_threads_per_warp=32)
        if lane_id == 31:
            scratch[warp_id] = value
        self.cta_sync(s_cta_mbarrier, sync_phase)

        if warp_id == 0:
            if lane_id < num_warps:
                warp_value = scratch[lane_id]
                warp_value = warp_scan(warp_value, tidx, lane_id, num_threads_per_warp=num_warps)
                scratch[lane_id] = warp_value
        self.cta_sync(s_cta_mbarrier, sync_phase)

        if warp_id > 0:
            value += scratch[warp_id - 1]
        total = scratch[num_warps - 1]
        # Every warp must consume the prior scan before the next tile reuses
        # the same scratch slots.
        self.cta_sync(s_cta_mbarrier, sync_phase)
        return value, total

    @cute.jit
    def to_coarse_key(self, x):
        """Convert to coarse 8-bit key for histogram"""

        if cutlass.const_expr(self.dtype == cutlass.Float32):
            # Convert to FP16 and extract high 8 bits
            h = x.to(cutlass.Float16)
            bits = half_as_ushort(h)

            key = cutlass.Uint16(0)

            # extract the sign bit
            # key = (bits & 0x8000) ? bits : ~bits & 0x7fff;
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)

            # high 8 bits
            return cute.Uint8((key >> 8) & 0xFF)
        else:
            # For half/bfloat16, extract high 8 bits directly
            if cutlass.const_expr(self.dtype == cutlass.Float16):
                bits = half_as_ushort(x)
            else:  # BFloat16
                bits = half_as_ushort(x)

            key = cute.Uint16(0)
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
            # high 8 bits
            return cute.Uint8((key >> 8) & 0xFF)

    @cute.jit
    def to_ordered(self, x):
        """Convert to ordered integer for comparison"""
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            bits = float_as_uint32(x)

            key = cutlass.Uint32(0)
            if bits & 0x80000000:
                key = cutlass.Uint32(bits)
            else:
                key = (bits ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)
            return cute.Uint32(key)
        else:
            if cutlass.const_expr(self.dtype == cutlass.Float16):
                bits = half_as_ushort(x)
            else:  # BFloat16
                bits = half_as_ushort(x)

            key = cute.Uint16(0)
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cute.Uint16(0xFFFF)) & cute.Uint16(0x7FFF)
            return cute.Uint16(key)

    @cute.jit
    def collect_boundary_ties(
        self,
        score,
        row_start,
        length,
        remaining,
        s_indices,
        scan_scratch,
        s_cta_mbarrier,
        sync_phase,
    ):
        """Select cutoff ties in source-column order without sorting output slots."""
        tidx, _, _ = cute.arch.thread_idx()
        # The radix pass already found the exact cutoff and filled these slots.
        # Read its key before any thread replaces one of the provisional winners.
        pivot_index = cutlass.Int32(cutlass.Uint32(s_indices[self.top_k - 1]))
        pivot_key = self.to_ordered(score[pivot_index])
        self.cta_sync(s_cta_mbarrier, sync_phase)
        selected = cutlass.Int32(0)
        tile_start = cutlass.Int32(0)
        # Four adjacent items per thread amortize the block scan and barriers
        # when sparse cutoff ties require traversing a long row.
        items_per_thread = 4
        flags = cute.make_rmem_tensor((items_per_thread,), cutlass.Int32)
        while (tile_start < length) and (selected < remaining):
            count = cutlass.Int32(0)
            for item in cutlass.range(items_per_thread, unroll_full=True):
                offset = tile_start + tidx * items_per_thread + item
                index = row_start + offset
                if cutlass.const_expr(self.tie_break == 2):
                    index = row_start + length - 1 - offset
                flags[item] = 0
                if offset < length:
                    flags[item] = cutlass.Int32(self.to_ordered(score[index]) == pivot_key)
                count += flags[item]
            if cutlass.const_expr(self.deterministic):
                prefix, total = self.deterministic_block_prefix_sum(
                    count,
                    scan_scratch,
                    s_cta_mbarrier,
                    sync_phase,
                )
            else:
                prefix, total = block_prefix_sum_kernel(
                    count,
                    scan_scratch,
                    tidx,
                    self.num_threads_per_cta,
                    self.num_threads_per_cta // 32,
                    need_total_sum=True,
                )
            position = selected + prefix - count
            for item in cutlass.range(items_per_thread, unroll_full=True):
                position += flags[item]
                if flags[item] != 0 and position <= remaining:
                    index = row_start + tile_start + tidx * items_per_thread + item
                    if cutlass.const_expr(self.tie_break == 2):
                        index = row_start + length - 1 - (tile_start + tidx * items_per_thread + item)
                    s_indices[self.top_k - remaining + position - 1] = self.index_type(index)
            # All threads must consume the scan totals before reusing scratch.
            self.cta_sync(s_cta_mbarrier, sync_phase)
            selected += total
            tile_start += self.num_threads_per_cta * items_per_thread

    @cute.jit
    def sort_selected_indices(
        self,
        length,
        s_indices,
        scan_scratch,
        s_cta_mbarrier,
        sync_phase,
        s_index_bits,
    ):
        """Canonicalize selected local indices in ascending source order.

        The radix selection path uses shared-memory atomics to claim output
        slots.  The selected set is deterministic once boundary ties have
        been canonicalized, but those atomics do not provide a repeatable
        slot order.  Sort the selected local indices before the output copy,
        providing a stronger canonical ordering than set repeatability alone.
        """
        tidx, _, _ = cute.arch.thread_idx()
        valid_k = min(cutlass.Int32(self.top_k), length)
        if cutlass.const_expr(self.use_deterministic_bitmap):
            num_words = (length + cutlass.Int32(31)) // cutlass.Int32(32)
            for word in range(tidx, num_words, self.num_threads_per_cta):
                s_index_bits[word] = cutlass.Uint32(0)
            self.cta_sync(s_cta_mbarrier, sync_phase)

            for i in range(tidx, valid_k, self.num_threads_per_cta):
                index = cutlass.Uint32(s_indices[i])
                word = cutlass.Int32(index >> cutlass.Uint32(5))
                bit = cutlass.Uint32(1) << (index & cutlass.Uint32(31))
                cute.arch.atomic_or(
                    (s_index_bits.iterator + word).llvm_ptr,
                    bit,
                    sem="relaxed",
                    scope="cta",
                )
            self.cta_sync(s_cta_mbarrier, sync_phase)

            output_base = cutlass.Int32(0)
            word_base = cutlass.Int32(0)
            while word_base < num_words:
                word = word_base + tidx
                bits = cutlass.Uint32(0)
                if word < num_words:
                    bits = s_index_bits[word]
                count = cutlass.Int32(cute.arch.popc(bits))
                prefix, total = self.deterministic_block_prefix_sum(
                    count,
                    scan_scratch,
                    s_cta_mbarrier,
                    sync_phase,
                )
                slot = output_base + prefix - count
                for bit_idx in cutlass.range_constexpr(32):
                    if bits & (cutlass.Uint32(1) << bit_idx):
                        s_indices[slot] = self.index_type(word * cutlass.Int32(32) + bit_idx)
                        slot += 1
                output_base += total
                word_base += self.num_threads_per_cta
            self.cta_sync(s_cta_mbarrier, sync_phase)
        else:
            sentinel = self.index_type(self.index_sentinel)

            # s_indices is provisioned for INDEXER_TOPK_MAX_K (2048), so every
            # power-of-two padding slot needed by the in-place bitonic sort exists.
            i = tidx
            while i < self.deterministic_sort_size:
                if i >= valid_k:
                    s_indices[i] = sentinel
                i += self.num_threads_per_cta
            self.cta_sync(s_cta_mbarrier, sync_phase)

            width = cutlass.Int32(2)
            while width <= self.deterministic_sort_size:
                distance = width // cutlass.Int32(2)
                while distance > 0:
                    i = tidx
                    while i < self.deterministic_sort_size:
                        peer = i ^ distance
                        if peer > i:
                            lhs = s_indices[i]
                            rhs = s_indices[peer]
                            ascending = (i // width) % cutlass.Int32(2) == 0
                            descending = (i // width) % cutlass.Int32(2) == 1
                            swap = (ascending and lhs > rhs) or (descending and lhs < rhs)
                            if swap:
                                s_indices[i] = rhs
                                s_indices[peer] = lhs
                        i += self.num_threads_per_cta
                    self.cta_sync(s_cta_mbarrier, sync_phase)
                    distance = distance // cutlass.Int32(2)
                width = width * cutlass.Int32(2)

    @cute.jit
    def prefix_sum_and_find_threshold_coarse(
        self,
        tidx,
        s_histogram,
        s_warp_sums,
        num_warps,
        s_threshold_bin_id,
        s_num_input,
        s_counter,
        s_last_remain,
        topk_remaining,
        g_num_input,
        s_cta_mbarrier,
        sync_phase,
        s_num_input_idx=0,
    ):
        """Scan the coarse histogram and publish the selected radix bin."""
        if cutlass.const_expr(self.radix <= self.num_threads_per_cta):
            previous = 0
            if tidx < cutlass.Int32(self.radix):
                val = s_histogram[tidx]
                val, total_sum = block_prefix_sum_kernel(val, s_warp_sums, tidx, self.radix, num_warps, barrier_id=1)
                s_histogram[tidx] = val
                # sync among self.radix threads
                cute.arch.barrier(barrier_id=1, number_of_threads=self.radix)

                if tidx > 0:
                    previous = s_histogram[tidx - 1]
                if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                    s_threshold_bin_id[0] = tidx
                    s_num_input[s_num_input_idx] = 0
                    if cutlass.const_expr(self.enable_gmem_store):
                        g_num_input[s_num_input_idx] = 0
                    # TODO: the difference between 1 and 2.
                    s_counter[0] = 0
            # sync among all threads in a cta.
            self.cta_sync(s_cta_mbarrier, sync_phase)
        else:
            assert self.radix % self.num_threads_per_cta == 0
            previous_sum = 0
            val = 0
            total_sum = 0
            for i in range(tidx, self.radix, self.num_threads_per_cta):
                val = s_histogram[i]
                val, total_sum = block_prefix_sum_kernel(
                    val,
                    s_warp_sums,
                    tidx,
                    self.num_threads_per_cta,
                    num_warps,
                    barrier_id=2,
                    need_total_sum=True,
                )
                s_histogram[i] = val + previous_sum
                previous_sum = previous_sum + total_sum
            # sync among all threads in a cta.
            self.cta_sync(s_cta_mbarrier, sync_phase)

            previous = 0
            run_loop = True
            if tidx > 0:
                previous = s_histogram[tidx - 1]
            if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                s_threshold_bin_id[0] = tidx
                s_num_input[s_num_input_idx] = 0
                if cutlass.const_expr(self.enable_gmem_store):
                    g_num_input[s_num_input_idx] = 0
                # the difference between coarse and fine-grained.
                s_counter[0] = 0
                run_loop = False

            if run_loop:
                run_next_loop = True
                for i in range(
                    tidx + self.num_threads_per_cta,
                    self.radix,
                    self.num_threads_per_cta,
                ):
                    if run_next_loop:
                        previous = s_histogram[i - 1]
                        if previous <= topk_remaining and s_histogram[i] > topk_remaining:
                            s_threshold_bin_id[0] = i
                            s_num_input[s_num_input_idx] = 0
                            if cutlass.const_expr(self.enable_gmem_store):
                                g_num_input[s_num_input_idx] = 0
                            # the difference between coarse and fine-grained.
                            s_counter[0] = 0
                            run_next_loop = False
            # sync among all threads in a cta.
            self.cta_sync(s_cta_mbarrier, sync_phase)

    @cute.jit
    def prefix_sum_and_find_threshold_fine_grained(
        self,
        tidx,
        s_histogram,
        s_warp_sums,
        num_warps,
        s_threshold_bin_id,
        s_num_input,
        s_counter,
        s_last_remain,
        topk_remaining,
        g_num_input,
        s_cta_mbarrier,
        sync_phase,
        s_num_input_idx=0,
    ):
        """Scan the fine histogram and publish its bin and residual rank."""
        if cutlass.const_expr(self.radix <= self.num_threads_per_cta):
            previous = 0
            if tidx < cutlass.Int32(self.radix):
                val = s_histogram[tidx]
                val, total_sum = block_prefix_sum_kernel(val, s_warp_sums, tidx, self.radix, num_warps, barrier_id=1)
                s_histogram[tidx] = val
                # sync
                cute.arch.barrier(barrier_id=1, number_of_threads=self.radix)

                if tidx > 0:
                    previous = s_histogram[tidx - 1]
                if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                    s_threshold_bin_id[0] = tidx
                    s_num_input[s_num_input_idx] = 0
                    if cutlass.const_expr(self.enable_gmem_store):
                        g_num_input[s_num_input_idx] = 0
                    # the first difference between coarse and fine-grained.
                    s_last_remain[0] = topk_remaining - previous
            self.cta_sync(s_cta_mbarrier, sync_phase)
        else:
            assert self.radix % self.num_threads_per_cta == 0
            previous_sum = 0
            val = 0
            total_sum = 0
            for i in range(tidx, self.radix, self.num_threads_per_cta):
                val = s_histogram[i]
                val, total_sum = block_prefix_sum_kernel(
                    val,
                    s_warp_sums,
                    tidx,
                    self.num_threads_per_cta,
                    num_warps,
                    barrier_id=2,
                    need_total_sum=True,
                )
                s_histogram[i] = val + previous_sum
                previous_sum = previous_sum + total_sum
            # sync among all threads in a cta.
            self.cta_sync(s_cta_mbarrier, sync_phase)

            previous = 0
            run_loop = True
            if tidx > 0:
                previous = s_histogram[tidx - 1]
            if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                s_threshold_bin_id[0] = tidx
                s_num_input[s_num_input_idx] = 0
                if cutlass.const_expr(self.enable_gmem_store):
                    g_num_input[s_num_input_idx] = 0
                # the difference between coarse and fine-grained.
                s_last_remain[0] = topk_remaining - previous
                run_loop = False
            if run_loop:
                run_next_loop = True
                for i in range(
                    tidx + self.num_threads_per_cta,
                    self.radix,
                    self.num_threads_per_cta,
                ):
                    if run_next_loop:
                        previous = s_histogram[i - 1]
                        if previous <= topk_remaining and s_histogram[i] > topk_remaining:
                            s_threshold_bin_id[0] = i
                            s_num_input[s_num_input_idx] = 0
                            if cutlass.const_expr(self.enable_gmem_store):
                                g_num_input[s_num_input_idx] = 0
                            # the difference between coarse and fine-grained.
                            s_last_remain[0] = topk_remaining - previous
                            run_next_loop = False
            # sync among all threads in a cta.
            self.cta_sync(s_cta_mbarrier, sync_phase)

    @cute.jit
    def indexer_topk_kernel_per_row(
        self,
        input: cute.Tensor,
        # gmem, used for the merge blocks kernel.
        input_indices: cute.Tensor,
        extra_buffer: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
        tiler_mn: cute.Shape,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        row_start: int,
        length: int,
        bidx: int,
        s_histogram,
        s_counter,
        s_threshold_bin_id,
        s_num_input,
        g_num_input,
        s_indices,
        s_input_idx,
        s_last_remain,
        num_warps,
        s_warp_sums,
        s_cta_mbarrier,
        s_index_bits,
    ):
        """CuTe DSL implementation of TopK kernel based on radix-based filter algorithm."""
        # # Thread and block indexing
        tidx, _, _ = cute.arch.thread_idx()
        sync_phase = cute.make_rmem_tensor((1,), cutlass.Int32)
        sync_phase[0] = 0

        score = input[bidx, None]
        if cutlass.const_expr(self.merge_blocks):
            indices = input_indices[bidx, None]
        if cutlass.const_expr(self.enable_multi_cta):
            dst = output_indices
            if cutlass.const_expr(self.return_val):
                dst_values = output_values
        else:
            dst = output_indices[bidx, None]
            if cutlass.const_expr(self.return_val):
                dst_values = output_values[bidx, None]
        # Note, for multi-cta version, each ctas must have its own extra_buffer.
        if cutlass.const_expr(self.enable_gmem_store):
            if cutlass.const_expr(self.enable_multi_cta):
                _, grid_dim_y, _ = cute.arch.grid_dim()
                bidx_val, bidy_val, _ = cute.arch.block_idx()
                buffer_row_id = cutlass.Int64(bidx_val) * cutlass.Int64(grid_dim_y) + cutlass.Int64(bidy_val)
            else:
                buffer_row_id = cutlass.Int64(bidx)
            buffer = self._slice_row_64bit(extra_buffer, buffer_row_id)

        # for initial scalar load part.
        row_ptr = score.iterator + row_start
        row_addr_u64 = row_ptr.toint()

        # 256/8 = 32bytes
        align_bytes = self.num_copy_bits // 8
        # fp32: 4bytes
        elem_bytes = self.dtype.width // 8

        misalign = row_addr_u64 % align_bytes
        fix_bytes = cutlass.Int64(0)
        if misalign != 0:
            fix_bytes = align_bytes - misalign

        prologue_elems = cutlass.Int32(fix_bytes // elem_bytes)
        # A row shorter than the bytes up to the next aligned address is all prologue.
        if prologue_elems > length:
            prologue_elems = length

        remaining = length - prologue_elems
        aligned_size = (remaining // self.vec_size) * self.vec_size
        left_size = remaining - aligned_size

        vec_start = row_start + prologue_elems
        left_start = vec_start + aligned_size

        shape = input.shape

        idX = cute.make_identity_tensor((shape[0], aligned_size))
        input_ptr = input.iterator + vec_start
        input_addr_u64 = input_ptr.toint()
        input_ptr_aligned = cute.make_ptr(self.dtype, input_addr_u64, input.memspace, assumed_align=align_bytes)

        input_tensor = cute.make_tensor(
            input_ptr_aligned,
            cute.make_layout((shape[0], aligned_size), stride=input.stride),
        )

        # slice for CTAs
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, None)) for mT in (input_tensor, idX)]
        # Note, we use gX_aligned here to avoid the alignment issue when the input is not aligned.
        gX_aligned_ptr = cute.make_ptr(self.dtype, gX.iterator.toint(), gX.memspace, assumed_align=align_bytes)
        gX_aligned = cute.make_tensor(gX_aligned_ptr, cute.make_layout(gX.shape, stride=gX.stride))

        self.num_sub_tiles = gX.shape[2]

        thr_copy = tiled_copy.get_slice(tidx)

        tXgX = thr_copy.partition_S(gX_aligned)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None, None]
        tXrX = cute.make_fragment_like(tXgX[None, None, None, 0])

        tXcX_tile = thr_copy.partition_S(cX)

        # Trivial case: length <= top_k
        if length <= self.top_k:
            for i in range(tidx, self.top_k, self.num_threads_per_cta):
                # TODO: add multi-cta version support here.
                if i < length:
                    if cutlass.const_expr(self.enable_multi_cta):
                        dst[i] = i + row_start
                    elif cutlass.const_expr(self.merge_blocks):
                        dst[i] = indices[i]
                    else:
                        dst[i] = i
                    if cutlass.const_expr(self.return_val):
                        if cutlass.const_expr(self.enable_multi_cta):
                            dst_values[i] = score[i + row_start]
                        else:
                            dst_values[i] = score[i]
                else:
                    dst[i] = -1
                    if cutlass.const_expr(self.return_val):
                        dst_values[i] = dst_values.element_type(dst_values.element_type.inf * dst_values.element_type(-1.0))
        else:
            if cutlass.const_expr(self.deterministic):
                if tidx == 0:
                    cute.arch.mbarrier_init(s_cta_mbarrier.iterator, self.num_threads_per_cta)
                cute.arch.mbarrier_init_fence()
                # Publish initialization before any thread arrives on the mbarrier.
                cute.arch.barrier()

            topk_remaining = self.top_k

            val_one = cutlass.Int32(1)
            val_one_negative = cutlass.Int32(-1)

            # Stage 1: Coarse histogram.
            if tidx < self.radix + 1:
                s_histogram[tidx] = 0
            self.cta_sync(s_cta_mbarrier, sync_phase)

            # 1.1 Build histogram with vectorized loads
            vec_size = self.vec_size

            for tile_idx in range(self.num_sub_tiles):
                tXpX_tile = self.predicate_tile(
                    tXcX_tile[None, None, None, tile_idx],
                    cutlass.Int32(aligned_size),
                )
                cute.copy(
                    copy_atom,
                    tXgX[None, None, None, tile_idx],
                    tXrX,
                    pred=tXpX_tile[None, None, None],
                )
                self._fill_oob(
                    tXrX,
                    tXpX_tile[None, None, None],
                    -tXrX.element_type.inf,
                )

                cur_tXcX = tXcX[None, None, None, tile_idx]
                for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                    relative_idx = cutlass.Int32(cur_tXcX[i // vec_size][1] + i % vec_size)
                    # The predicated copy fills lanes outside aligned_size with
                    # -inf, but those lanes are not part of the input row and
                    # must not contribute to the radix histogram.
                    if relative_idx < aligned_size:
                        bin_val = self.to_coarse_key(tXrX[i])
                        atomicAdd(
                            s_histogram.iterator + cutlass.Int32(bin_val),
                            val_one,
                        )

            # for initial scalar load part.
            for j in range(tidx, prologue_elems, self.num_threads_per_cta):
                col_idx = cutlass.Int32(row_start + j)
                raw = score[col_idx]
                bin_val = self.to_coarse_key(raw)
                atomicAdd(
                    s_histogram.iterator + cutlass.Int32(bin_val),
                    val_one,
                )

            # for left part (left_size)
            for j in range(tidx, left_size, self.num_threads_per_cta):
                col_idx = cutlass.Int32(left_start + j)
                raw = score[col_idx]
                bin_val = self.to_coarse_key(raw)
                atomicAdd(
                    s_histogram.iterator + cutlass.Int32(bin_val),
                    val_one,
                )

            self.cta_sync(s_cta_mbarrier, sync_phase)

            # 1.2 and 1.3  Suffix sum to find threshold and find threshold bin
            self.prefix_sum_and_find_threshold_coarse(
                tidx,
                s_histogram,
                s_warp_sums,
                num_warps,
                s_threshold_bin_id,
                s_num_input,
                s_counter,
                s_last_remain,
                topk_remaining,
                g_num_input,
                s_cta_mbarrier,
                sync_phase,
                s_num_input_idx=0,
            )

            threshold_bin = s_threshold_bin_id[0]
            if threshold_bin > 0:
                topk_remaining -= s_histogram[threshold_bin - 1]

            # 1.4 Collect indices
            if topk_remaining == 0:
                # Collect indices where bin > threshold
                for tile_idx in range(self.num_sub_tiles):
                    tXpX_tile = self.predicate_tile(
                        tXcX_tile[None, None, None, tile_idx],
                        cutlass.Int32(aligned_size),
                    )
                    cute.copy(
                        copy_atom,
                        tXgX[None, None, None, tile_idx],
                        tXrX,
                        pred=tXpX_tile[None, None, None],
                    )
                    self._fill_oob(
                        tXrX,
                        tXpX_tile[None, None, None],
                        -tXrX.element_type.inf,
                    )
                    for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                        cur_tXcX = tXcX[None, None, None, tile_idx]
                        relative_idx = cutlass.Int32(cur_tXcX[i // vec_size][1] + i % vec_size)
                        if relative_idx < aligned_size:
                            bin_val = self.to_coarse_key(tXrX[i])
                            if bin_val < threshold_bin:
                                pos = atomicAdd(s_counter.iterator, val_one)
                                idx = self.index_type(relative_idx + vec_start)
                                s_indices[pos] = idx

                # for initial scalar load part.
                for j in range(tidx, prologue_elems, self.num_threads_per_cta):
                    col_idx = cutlass.Int32(row_start + j)
                    raw = score[col_idx]
                    bin_val = self.to_coarse_key(raw)
                    if bin_val < threshold_bin:
                        pos = atomicAdd(s_counter.iterator, val_one)
                        idx = self.index_type(col_idx)
                        s_indices[pos] = idx

                # for left part (left_size)
                for j in range(tidx, left_size, self.num_threads_per_cta):
                    col_idx = cutlass.Int32(left_start + j)
                    raw = score[col_idx]
                    bin_val = self.to_coarse_key(raw)
                    if bin_val < threshold_bin:
                        pos = atomicAdd(s_counter.iterator, val_one)
                        idx = self.index_type(col_idx)
                        s_indices[pos] = idx

                self.cta_sync(s_cta_mbarrier, sync_phase)

            else:
                # Reset histogram for refinement
                self.cta_sync(s_cta_mbarrier, sync_phase)
                if tidx < self.radix + 1:
                    s_histogram[tidx] = 0
                self.cta_sync(s_cta_mbarrier, sync_phase)

                # Filter and build refinement histogram
                for tile_idx in range(self.num_sub_tiles):
                    tXpX_tile = self.predicate_tile(
                        tXcX_tile[None, None, None, tile_idx],
                        cutlass.Int32(aligned_size),
                    )
                    cute.copy(
                        copy_atom,
                        tXgX[None, None, None, tile_idx],
                        tXrX,
                        pred=tXpX_tile[None, None, None],
                    )
                    self._fill_oob(
                        tXrX,
                        tXpX_tile[None, None, None],
                        -tXrX.element_type.inf,
                    )

                    for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                        cur_tXcX = tXcX[None, None, None, tile_idx]
                        relative_idx = cutlass.Int32(cur_tXcX[i // vec_size][1] + i % vec_size)
                        if relative_idx < aligned_size:
                            raw_input = tXrX[i]
                            bin_val = self.to_coarse_key(raw_input)
                            idx = self.index_type(relative_idx + vec_start)
                            if bin_val < threshold_bin:
                                pos = atomicAdd(s_counter.iterator, val_one)
                                s_indices[pos] = idx
                            elif bin_val == threshold_bin:
                                # pos = atomicAdd(s_num_input[0], 1)
                                pos = atomicAdd(s_num_input.iterator, val_one)
                                if cutlass.const_expr(self.enable_gmem_store):
                                    if pos < self.indexer_topk_smem_input_size:
                                        s_input_idx[0, pos] = idx
                                    else:
                                        buffer_pos = atomicAdd(
                                            g_num_input.iterator,
                                            val_one,
                                        )
                                        buffer[0, buffer_pos] = cutlass.Int32(cutlass.Uint32(idx))
                                    ordered = self.to_ordered(raw_input)
                                    sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                                    # atomicAdd(s_histogram[sub_bin], 1)
                                    atomicAdd(
                                        s_histogram.iterator + cutlass.Int32(sub_bin),
                                        val_one,
                                    )
                                else:
                                    if pos < self.indexer_topk_smem_input_size:
                                        s_input_idx[0, pos] = idx
                                    ordered = self.to_ordered(raw_input)
                                    sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                                    # atomicAdd(s_histogram[sub_bin], 1)
                                    atomicAdd(
                                        s_histogram.iterator + cutlass.Int32(sub_bin),
                                        val_one,
                                    )

                # for initial scalar load part.
                for j in range(tidx, prologue_elems, self.num_threads_per_cta):
                    col_idx = cutlass.Int32(row_start + j)
                    raw = score[col_idx]
                    bin_val = self.to_coarse_key(raw)
                    if bin_val < threshold_bin:
                        pos = atomicAdd(s_counter.iterator, val_one)
                        idx = self.index_type(col_idx)
                        s_indices[pos] = idx
                    elif bin_val == threshold_bin:
                        pos = atomicAdd(
                            s_num_input.iterator,
                            val_one,
                        )
                        # TODO: add gmem buffer here.
                        if cutlass.const_expr(self.enable_gmem_store):
                            if pos < self.indexer_topk_smem_input_size:
                                s_input_idx[0, pos] = self.index_type(col_idx)
                            else:
                                buffer_pos = atomicAdd(
                                    g_num_input.iterator,
                                    val_one,
                                )
                                buffer[0, buffer_pos] = cutlass.Int32(col_idx)
                            ordered = self.to_ordered(raw)
                            sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                            atomicAdd(
                                s_histogram.iterator + cutlass.Int32(sub_bin),
                                val_one,
                            )
                        else:
                            # TODO: how to handle the type of sub_bin and ordered?
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                ordered = cutlass.Uint32(0)
                                sub_bin = cutlass.Uint32(0)
                            else:
                                ordered = cutlass.Uint16(0)
                                sub_bin = cutlass.Int32(0)
                            if pos < self.indexer_topk_smem_input_size:
                                s_input_idx[0, pos] = self.index_type(col_idx)
                                ordered = self.to_ordered(raw)
                                sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                                atomicAdd(
                                    s_histogram.iterator + cutlass.Int32(sub_bin),
                                    val_one,
                                )

                # for left part
                for j in range(tidx, left_size, self.num_threads_per_cta):
                    col_idx = cutlass.Int32(left_start + j)
                    raw = score[col_idx]
                    bin_val = self.to_coarse_key(raw)
                    if bin_val < threshold_bin:
                        pos = atomicAdd(s_counter.iterator, val_one)
                        idx = self.index_type(col_idx)
                        s_indices[pos] = idx
                    elif bin_val == threshold_bin:
                        pos = atomicAdd(
                            s_num_input.iterator,
                            val_one,
                        )
                        # TODO: add gmem buffer here.
                        if cutlass.const_expr(self.enable_gmem_store):
                            if pos < self.indexer_topk_smem_input_size:
                                s_input_idx[0, pos] = self.index_type(col_idx)
                            else:
                                buffer_pos = atomicAdd(
                                    g_num_input.iterator,
                                    val_one,
                                )
                                buffer[0, buffer_pos] = cutlass.Int32(col_idx)
                            ordered = self.to_ordered(raw)
                            sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                            atomicAdd(
                                s_histogram.iterator + cutlass.Int32(sub_bin),
                                val_one,
                            )
                        else:
                            # TODO: how to handle the type of sub_bin and ordered?
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                ordered = cutlass.Uint32(0)
                                sub_bin = cutlass.Uint32(0)
                            else:
                                ordered = cutlass.Uint16(0)
                                sub_bin = cutlass.Int32(0)
                            if pos < self.indexer_topk_smem_input_size:
                                s_input_idx[0, pos] = self.index_type(col_idx)
                                ordered = self.to_ordered(raw)
                                sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                                atomicAdd(
                                    s_histogram.iterator + cutlass.Int32(sub_bin),
                                    val_one,
                                )
                fence_acq_rel_cta()
                self.cta_sync(s_cta_mbarrier, sync_phase)

                # Phase 2: Refinement rounds
                run_next_round = True
                for round in range(self.num_refine_rounds):
                    if run_next_round:
                        r_idx = round % 2

                        self.prefix_sum_and_find_threshold_fine_grained(
                            tidx,
                            s_histogram,
                            s_warp_sums,
                            num_warps,
                            s_threshold_bin_id,
                            s_num_input,
                            s_counter,
                            s_last_remain,
                            topk_remaining,
                            g_num_input,
                            s_cta_mbarrier,
                            sync_phase,
                            s_num_input_idx=r_idx ^ 1,
                        )
                        num_input = min(s_num_input[r_idx], self.indexer_topk_smem_input_size)
                        if cutlass.const_expr(self.enable_gmem_store):
                            cur_g_num_input = g_num_input[r_idx]

                        threshold = s_threshold_bin_id[0]
                        if threshold > 0:
                            topk_remaining -= s_histogram[threshold - 1]
                        offset = self.first_refine_shift - round * 8
                        is_last_round = round == self.num_refine_rounds - 1

                        if topk_remaining == 0:
                            for i in range(tidx, num_input, self.num_threads_per_cta):
                                idx = s_input_idx[r_idx, i]
                                idx = cutlass.Int32(cutlass.Uint32(idx))
                                bin_val = (self.to_ordered(score[idx]) >> offset) & 0xFF
                                if bin_val < threshold:
                                    pos = atomicAdd(s_counter.iterator, val_one)
                                    s_indices[pos] = self.index_type(idx)
                            if cutlass.const_expr(self.enable_gmem_store):
                                for i in range(
                                    tidx,
                                    cur_g_num_input,
                                    self.num_threads_per_cta,
                                ):
                                    idx = buffer[r_idx, i]
                                    bin_val = (self.to_ordered(score[idx]) >> offset) & 0xFF
                                    if bin_val < threshold:
                                        pos = atomicAdd(s_counter.iterator, val_one)
                                        s_indices[pos] = self.index_type(idx)
                            self.cta_sync(s_cta_mbarrier, sync_phase)
                            # break
                            run_next_round = False
                        else:
                            # Reset histogram
                            self.cta_sync(s_cta_mbarrier, sync_phase)
                            if tidx < self.radix + 1:
                                s_histogram[tidx] = 0
                            self.cta_sync(s_cta_mbarrier, sync_phase)

                            for i in range(tidx, num_input, self.num_threads_per_cta):
                                idx = s_input_idx[r_idx, i]
                                idx_int32 = cutlass.Int32(cutlass.Uint32(idx))
                                raw_input = score[idx_int32]
                                idx = self.index_type(idx_int32)
                                bin_val = (self.to_ordered(raw_input) >> offset) & 0xFF
                                if bin_val < threshold:
                                    pos = atomicAdd(s_counter.iterator, val_one)
                                    s_indices[pos] = idx
                                elif bin_val == threshold:
                                    if is_last_round:
                                        cur_pos = atomicAdd(
                                            s_last_remain.iterator,
                                            val_one_negative,
                                        )
                                        if cur_pos > 0:
                                            s_indices[self.top_k - cur_pos] = idx
                                    else:
                                        # pos = atomicAdd(s_num_input[r_idx ^ 1], 1)
                                        cur_pos = atomicAdd(
                                            s_num_input.iterator + (r_idx ^ 1),
                                            val_one,
                                        )
                                        # TODO: remove this if logic for gmem store?
                                        # num_input < filter_topk_smem_input_size
                                        if cutlass.const_expr(self.enable_gmem_store):
                                            if cur_pos < self.indexer_topk_smem_input_size:
                                                s_input_idx[r_idx ^ 1, cur_pos] = idx
                                            else:
                                                buffer_pos = atomicAdd(
                                                    g_num_input.iterator + (r_idx ^ 1),
                                                    val_one,
                                                )
                                                buffer[r_idx ^ 1, buffer_pos] = idx_int32
                                            bin32 = self.to_ordered(raw_input)
                                            sub_bin = (bin32 >> (offset - 8)) & 0xFF
                                            # atomicAdd(s_histogram[sub_bin], 1)
                                            atomicAdd(
                                                s_histogram.iterator + cutlass.Int32(sub_bin),
                                                val_one,
                                            )
                                        else:
                                            # TODO: how to handle the type of sub_bin and bin32?
                                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                                bin32 = cutlass.Uint32(0)
                                                sub_bin = cutlass.Uint32(0)
                                            else:
                                                bin32 = cutlass.Uint16(0)
                                                sub_bin = cutlass.Int32(0)
                                            if cur_pos < self.indexer_topk_smem_input_size:
                                                s_input_idx[r_idx ^ 1, cur_pos] = idx
                                                bin32 = self.to_ordered(raw_input)
                                                sub_bin = (bin32 >> (offset - 8)) & 0xFF
                                                # atomicAdd(s_histogram[sub_bin], 1)
                                                atomicAdd(
                                                    s_histogram.iterator + cutlass.Int32(sub_bin),
                                                    val_one,
                                                )

                            self.cta_sync(s_cta_mbarrier, sync_phase)
                            if cutlass.const_expr(self.enable_gmem_store):
                                for i in range(
                                    tidx,
                                    cur_g_num_input,
                                    self.num_threads_per_cta,
                                ):
                                    # int32
                                    idx = buffer[r_idx, i]
                                    raw_input = score[idx]
                                    bin_val = (self.to_ordered(raw_input) >> offset) & 0xFF
                                    if bin_val < threshold:
                                        pos = atomicAdd(
                                            s_counter.iterator,
                                            val_one,
                                        )
                                        s_indices[pos] = self.index_type(idx)
                                    elif bin_val == threshold:
                                        if is_last_round:
                                            cur_pos = atomicAdd(
                                                s_last_remain.iterator,
                                                val_one_negative,
                                            )
                                            if cur_pos > 0:
                                                s_indices[self.top_k - cur_pos] = self.index_type(idx)
                                        else:
                                            # pos = atomicAdd(s_num_input[r_idx ^ 1], 1)
                                            cur_pos = atomicAdd(
                                                s_num_input.iterator + (r_idx ^ 1),
                                                val_one,
                                            )
                                            if cutlass.const_expr(self.enable_gmem_store):
                                                if cur_pos < self.indexer_topk_smem_input_size:
                                                    s_input_idx[r_idx ^ 1, cur_pos] = self.index_type(idx)
                                                else:
                                                    buffer_pos = atomicAdd(
                                                        g_num_input.iterator + (r_idx ^ 1),
                                                        val_one,
                                                    )
                                                    buffer[r_idx ^ 1, buffer_pos] = idx
                                                bin32 = self.to_ordered(raw_input)
                                                sub_bin = (bin32 >> (offset - 8)) & 0xFF
                                                # atomicAdd(s_histogram[sub_bin], 1)
                                                atomicAdd(
                                                    s_histogram.iterator + cutlass.Int32(sub_bin),
                                                    val_one,
                                                )
                                            else:
                                                if cur_pos < self.indexer_topk_smem_input_size:
                                                    s_input_idx[r_idx ^ 1, cur_pos] = idx
                                                    bin32 = self.to_ordered(raw_input)
                                                    sub_bin = (bin32 >> (offset - 8)) & 0xFF
                                                    # atomicAdd(s_histogram[sub_bin], 1)
                                                    atomicAdd(
                                                        s_histogram.iterator + cutlass.Int32(sub_bin),
                                                        val_one,
                                                    )
                            fence_acq_rel_cta()
                            self.cta_sync(s_cta_mbarrier, sync_phase)
                            if cutlass.const_expr(self.deterministic or self.tie_break != 0):
                                if is_last_round:
                                    # Only an overfull exact-score boundary reaches this
                                    # branch. Reuse the now-dead histogram for scan scratch.
                                    self.collect_boundary_ties(
                                        score,
                                        row_start,
                                        length,
                                        topk_remaining,
                                        s_indices,
                                        s_histogram,
                                        s_cta_mbarrier,
                                        sync_phase,
                                    )

            if cutlass.const_expr(self.deterministic):
                self.sort_selected_indices(
                    length,
                    s_indices,
                    s_histogram,
                    s_cta_mbarrier,
                    sync_phase,
                    s_index_bits,
                )

            # Phase 3: Output phase
            vecsize_out = cutlass.const_expr(
                min(
                    self.top_k,
                    cute.ceil_div(self.top_k, self.num_threads_per_cta),
                    self.num_copy_bits // self.dtype.width,
                    # TODO: only tested for float32. need to check for other dtypes.
                    # Odd top_k cannot be tiled by width-2 vectorized stores (there is
                    # no scalar tail), so fall back to scalar stores in that case.
                    2 if self.top_k % 2 == 0 else 1,
                )
            )
            assert self.top_k % vecsize_out == 0

            nvec_per_thread = cutlass.const_expr(cute.ceil_div(self.top_k, vecsize_out * self.num_threads_per_cta))
            topk_vals = cute.make_rmem_tensor((vecsize_out, nvec_per_thread), self.dtype)
            topk_indices = cute.make_rmem_tensor((vecsize_out, nvec_per_thread), cutlass.Int32)

            stride = self.num_threads_per_cta * vecsize_out
            for i in cutlass.range(nvec_per_thread, unroll_full=True):
                idx = i * stride + tidx % self.num_threads_per_cta * vecsize_out
                if idx < self.top_k:
                    for v in cutlass.range(vecsize_out, unroll_full=True):
                        index_raw = s_indices[idx + v]
                        index = cutlass.Int32(cutlass.Uint32(index_raw))
                        if cutlass.const_expr(self.return_val):
                            topk_vals[v, i] = score[index]
                        if cutlass.const_expr(self.merge_blocks):
                            topk_indices[v, i] = indices[index]
                        else:
                            topk_indices[v, i] = index
            # [atom, rest_vec]
            mIndices_store = cute.tiled_divide(dst, (vecsize_out,))
            if cutlass.const_expr(self.return_val):
                mValues_store = cute.tiled_divide(dst_values, (vecsize_out,))
            # i represents the index of the vector in the output.
            for i in cutlass.range(cute.size(topk_vals.shape, [1]), unroll_full=True):
                col = i * self.num_threads_per_cta + tidx % self.num_threads_per_cta
                if col < self.top_k // vecsize_out:
                    cute.autovec_copy(topk_indices[None, i], mIndices_store[None, col])
                    if cutlass.const_expr(self.return_val):
                        cute.autovec_copy(topk_vals[None, i], mValues_store[None, col])

    def _slice_row_64bit(self, buffer, row_id):
        row_offset_elems = row_id * cute.size(buffer.stride[0])
        elem_bytes = buffer.element_type.width // 8
        row_offset_bytes = row_offset_elems * elem_bytes
        row_ptr = cute.make_ptr(
            buffer.element_type,
            buffer.iterator.toint() + row_offset_bytes,
            buffer.memspace,
            assumed_align=elem_bytes,
        )
        return cute.make_tensor(
            row_ptr,
            cute.make_layout(
                (buffer.shape[1], buffer.shape[2]),
                stride=(buffer.stride[1], buffer.stride[2]),
            ),
        )

    def _get_tiled_copy(self):
        threads_per_row = self.num_threads_per_cta
        tiler_mn = (
            1,
            self.vec_size * threads_per_row,
        )

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=self.num_copy_bits,
        )

        thr_layout = cute.make_ordered_layout(
            (1, threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, self.vec_size))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        return (
            copy_atom,
            tiled_copy,
            tiler_mn,
        )

    @cute.jit
    def predicate_tile(self, tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
        tApA = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    cute.size(tAcA, mode=[0, 1]),
                    cute.size(tAcA, mode=[1]),
                    cute.size(tAcA, mode=[2]),
                ),
                stride=(cute.size(tAcA, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tApA.shape[0]):
            for rest_k in range(tApA.shape[2]):
                tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
        return tApA

    @cute.jit
    def _fill_oob(self, tXrX: cute.Tensor, tXpX: cute.Tensor, fill_value: cute.Numeric) -> None:
        """Fill out-of-bounds values in register tensor.

        Args:
            tXrX: Register tensor to fill
            tXpX: Predicate tensor indicating valid elements
            fill_value: Value to fill OOB locations with
        """
        tXrX_fill = cute.make_fragment_like(tXrX[(None, 0), None, 0])
        tXrX_fill.fill(fill_value)
        for rest_v in range(tXrX.shape[0][1]):
            for rest_k in range(tXrX.shape[2]):
                if cutlass.const_expr(tXpX is not None):
                    if not tXpX[0, rest_v, rest_k]:
                        cute.autovec_copy(tXrX_fill, tXrX[(None, rest_v), None, rest_k])


def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
    pad_to_vec_size: bool = False,
    vec_size: int = 8,
) -> torch.Tensor:
    """Create random logits tensor for testing.

    Args:
        row_starts: Tensor of shape (num_rows,) indicating the start position of each row
        row_ends: Tensor of shape (num_rows,) indicating the end position (exclusive) of each row
        dtype: Data type for the logits tensor
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape (num_rows, max_row_length) with random values and -inf padding
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    num_rows = row_starts.shape[0]
    max_len = int(row_ends.max().item())
    if pad_to_vec_size:
        max_len = (max_len + vec_size - 1) // vec_size * vec_size

    # Generate random logits
    logits = torch.randn(num_rows, max_len, dtype=dtype, device="cuda")

    # Vectorized masking: set positions outside [row_start, row_end) to -inf
    col_indices = torch.arange(max_len, device="cuda").unsqueeze(0)  # (1, max_len)
    mask_lo = col_indices < row_starts.unsqueeze(1)  # positions before row_start
    mask_hi = col_indices >= row_ends.unsqueeze(1)  # positions at or after row_end
    mask = mask_lo | mask_hi  # positions outside valid range
    logits[mask] = float("-inf")

    return logits


def run_reference_top_k(logits, row_starts, row_ends, index_topk):
    # Run reference implementation
    torch_indices = logits.topk(min(index_topk, max(row_ends)), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    return torch_indices


def compare_top_k_results(
    logits: torch.Tensor,
    cuda_indices: torch.Tensor,
    torch_indices: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    top_k: int,
    tolerance: float = 1e-5,
) -> bool:
    """
    Compare results from CUDA top_k_per_row with torch.topk.
    Handles different shapes and -1 placeholders in cuda_indices.

    Args:
        logits: Input logits tensor [num_rows, vocab_size]
        cuda_indices: CUDA implementation output [num_rows, cuda_k], may contain -1
        torch_indices: PyTorch reference output [num_rows, torch_k], may contain -1
        row_starts: Start positions for each row [num_rows]
        row_ends: End positions for each row [num_rows]
        top_k: Target top-k value
        tolerance: Tolerance for floating point comparison

    Returns:
        True if results match within tolerance, False otherwise
    """
    num_rows = cuda_indices.shape[0]

    # Calculate valid lengths for each row (vectorized)
    row_lengths = row_ends - row_starts

    # For each row, compare only the valid indices (non -1)
    for row_idx in range(num_rows):
        row_len = row_lengths[row_idx].item()
        expected_valid = min(row_len, top_k)

        # Get valid indices from both implementations (filter out -1)
        cuda_row = cuda_indices[row_idx]
        torch_row = torch_indices[row_idx]

        # Filter out -1 (invalid) indices
        cuda_valid_mask = cuda_row != -1
        torch_valid_mask = torch_row != -1

        cuda_valid = cuda_row[cuda_valid_mask]
        torch_valid = torch_row[torch_valid_mask]

        # Check if the number of valid indices matches
        if cuda_valid.shape[0] != torch_valid.shape[0]:
            print(f"Row {row_idx}: Different number of valid indices - " f"CUDA: {cuda_valid.shape[0]}, PyTorch: {torch_valid.shape[0]}")
            return False

        if cuda_valid.shape[0] != expected_valid:
            print(f"Row {row_idx}: Expected {expected_valid} valid indices, got {cuda_valid.shape[0]}")
            return False

        # If no valid indices, continue
        if cuda_valid.shape[0] == 0:
            continue

        # Gather the corresponding logit values
        row_start = row_starts[row_idx].item()
        logits_row = logits[row_idx]

        # Adjust indices to absolute positions (add row_start offset)
        cuda_abs_indices = cuda_valid + row_start
        torch_abs_indices = torch_valid + row_start

        # Get logit values for the selected indices
        cuda_values = logits_row[cuda_abs_indices]
        torch_values = logits_row[torch_abs_indices]

        # Sort both value arrays in descending order
        cuda_values_sorted, _ = torch.sort(cuda_values, descending=True)
        torch_values_sorted, _ = torch.sort(torch_values, descending=True)

        # Compare sorted values
        if not torch.allclose(cuda_values_sorted, torch_values_sorted, rtol=tolerance, atol=tolerance):
            # Additional debug: check if sets are identical
            cuda_set = set(cuda_valid.cpu().tolist())
            torch_set = set(torch_valid.cpu().tolist())
            print(f"row_idx: {row_idx}, row_len: {row_len}, expected_valid: {expected_valid}")
            print(f"cuda_values_sorted: {cuda_values_sorted}")
            print(f"torch_values_sorted: {torch_values_sorted}")
            if cuda_set != torch_set:
                print("  Different indices selected:")
                print(f"    Only in CUDA: {cuda_set - torch_set}")
                print(f"    Only in Torch: {torch_set - cuda_set}")

            return False

    return True
