# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native H32 recovery over failed rows in four-row queued tasks."""

import cutlass
import cutlass.cute as cute
from .fp4_queue_recovery import FP4PrefillMQALogitsKernel as _Base
from .fp4_queue_recovery import (
    pipeline,
    indexer_memory,
    pdl_prologue,
    subset_barrier,
    finish_recovery,
)


class FP4PrefillMQALogitsKernel(_Base):
    supported_splits = (1, 4, 8, 16)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            self.num_heads == 32
            and self.top_k == 512
            and self.output_dtype == cutlass.BFloat16
            and self.native_lowp
        )
        self.block_q = 1
        self.N = 32
        self.w_stage_stride = (
            (self.N * self.weight_bytes + 127) // 128 * 128
        ) // self.weight_bytes
        self.mma_tiler_mn = (self.block_kv, self.N)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # KV: [block_kv, head_dim, num_kv_blocks] (FP4)
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # Q (L dim = num_q_blocks, FP4)
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,  # Weights TMA coord tensor [N, num_q_blocks]
        tma_atom_sf_kv: cute.CopyAtom,
        mSF_KV_tma: cute.Tensor,  # SF KV TMA coord tensor [block_kv, num_kv_blocks] int32
        tma_atom_sf_q: cute.CopyAtom,
        mSF_Q_tma: cute.Tensor,  # SF Q TMA coord tensor [N_padded, num_q_blocks] int32
        mLogits: cute.Tensor,  # [num_q_tokens, logits_stride]
        mCuSeqLenKStart: cute.Tensor,  # [num_q_tokens] int32
        mCuSeqLenKEnd: cute.Tensor,  # [num_q_tokens] int32
        thresholds: cute.Tensor,
        counts: cute.Tensor,
        candidates: cute.Tensor,
        output: cute.Tensor,
        recovery_counters: cute.Tensor,
        recovery_queue: cute.Tensor,
        recovery_sample_info: cute.Tensor,
        num_q_tokens: cutlass.Int32,
        num_q_blocks: cutlass.Int32,
        num_kv_tokens: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_tma_smem_layout: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        sf_kv_smem_layout_staged: cute.Layout,
        sf_q_smem_layout_staged: cute.Layout,
        sf_q_tma_smem_layout_staged: cute.Layout,
        tmem_sfa_layout: cute.Layout,
        tmem_sfb_layout: cute.Layout,
        sfa_chunk_smem_layout: cute.Layout,
        sfb_chunk_smem_layout: cute.Layout,
        copy_atom_s2t: cute.CopyAtom,
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
    ):
        pdl_prologue(self.pdl)
        task_splits = cutlass.Int32(self.splits)
        split_shift = cutlass.Int32(4 if self.splits == 16 else 3)
        tid = cute.arch.thread_idx()[0]
        partition = cute.arch.block_idx()[2] & (task_splits - 1)
        slot = cute.arch.block_idx()[2] >> split_shift
        queue_count = recovery_queue[0]
        smem = indexer_memory.SmemAllocator()
        hist = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(1538), byte_alignment=128
        )
        scalar = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(2), byte_alignment=16
        )
        recovery_samples = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout(768), byte_alignment=128
        )
        last = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(1), byte_alignment=16
        )
        holding = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(1), byte_alignment=16
        )
        if slot < queue_count:
            task_allocator = indexer_memory.TmemAllocator(
                holding.iterator,
                barrier_for_retrieve=pipeline.NamedBarrier(
                    barrier_id=15, num_threads=384
                ),
                allocator_warp_id=0,
                is_two_cta=False,
                arch=self.arch_str,
            )
            task_allocator.allocate(self.num_tmem_alloc_cols_total)
            task_allocator.wait_for_alloc()
            task_tmem_ptr = task_allocator.retrieve_ptr(self.acc_dtype)
            while slot < queue_count:
                queued_task = recovery_queue[slot + 1]
                for subrow in cutlass.range(4):
                    task = queued_task * 4 + subrow
                    if task < num_q_tokens and counts[task] < min(
                        self.top_k, mCuSeqLenKEnd[task] - mCuSeqLenKStart[task]
                    ):
                        row_base = task * self.block_q
                        delta = cutlass.Int64(
                            (cute.arch.block_idx()[2] // self.splits) * self.block_q
                            - row_base
                        )
                        task_logits = cute.make_tensor(
                            mLogits.iterator + delta * mLogits.stride[0], mLogits.layout
                        )
                        task_candidates = cute.make_tensor(
                            candidates.iterator + delta * candidates.stride[0],
                            candidates.layout,
                        )
                        if tid < self.block_q:
                            init_row = row_base + tid
                            if init_row < num_q_tokens:
                                for bank in cutlass.range(partition, 8, task_splits):
                                    task_candidates[init_row, bank] = cutlass.Int64(0)
                        cute.arch.fence_acq_rel_gpu()
                        subset_barrier(barrier_id=14, number_of_threads=384)
                        if cutlass.const_expr(self.splits > 8):
                            # Two partitions share each initialized candidate bank.
                            cute.arch.cluster_arrive()
                            cute.arch.cluster_wait()
                        first = mCuSeqLenKStart[row_base]
                        for token in cutlass.range_constexpr(1, self.block_q):
                            first = min(
                                first,
                                mCuSeqLenKStart[
                                    min(num_q_tokens - 1, row_base + token)
                                ],
                            )
                        last_row = min(num_q_tokens - 1, row_base + self.block_q - 1)
                        lo = (first // 256) * 2 + partition * 2
                        hi = (mCuSeqLenKEnd[last_row] + 127) // 128
                        if lo < hi:
                            self._gemm_body(
                                tiled_mma,
                                tma_atom_a,
                                mA_mkl,
                                tma_atom_b,
                                mB_nkl,
                                tma_atom_w,
                                mW_tma,
                                tma_atom_sf_kv,
                                mSF_KV_tma,
                                tma_atom_sf_q,
                                mSF_Q_tma,
                                task_logits,
                                mCuSeqLenKStart,
                                mCuSeqLenKEnd,
                                thresholds,
                                counts,
                                task_candidates,
                                output,
                                num_q_tokens,
                                task + 1,
                                num_kv_tokens,
                                cluster_layout_vmnk,
                                a_smem_layout_staged,
                                a_tma_smem_layout,
                                b_smem_layout_staged,
                                w_smem_layout_staged,
                                sf_kv_smem_layout_staged,
                                sf_q_smem_layout_staged,
                                sf_q_tma_smem_layout_staged,
                                tmem_sfa_layout,
                                tmem_sfb_layout,
                                sfa_chunk_smem_layout,
                                sfb_chunk_smem_layout,
                                copy_atom_s2t,
                                epi_tile,
                                task,
                                partition,
                                task_splits,
                                task_tmem_ptr,
                                recovery_sample_info,
                                SharedStorage,
                            )
                        cute.arch.fence_acq_rel_gpu()
                        subset_barrier(barrier_id=14, number_of_threads=384)
                        if tid == 0:
                            prior = cute.arch.atomic_add(
                                (recovery_counters.iterator + queued_task).llvm_ptr,
                                cutlass.Int32(1),
                                sem="acq_rel",
                                scope="gpu",
                            )
                            cute.arch.atomic_exch(
                                last.iterator.llvm_ptr,
                                cutlass.Int32(prior == task_splits - 1),
                                sem="release",
                                scope="cta",
                            )
                        # Dedicated full-CTA publication and reader-retirement phases.
                        # Keep the shared decision immutable until every warp snapshots it.
                        cute.arch.barrier(barrier_id=4, number_of_threads=384)
                        cute.arch.fence_acq_rel_gpu()
                        decision = cutlass.Int32(0)
                        if cute.arch.lane_idx() == 0:
                            decision = cute.arch.atomic_add(
                                last.iterator.llvm_ptr,
                                cutlass.Int32(0),
                                sem="acquire",
                                scope="cta",
                            )
                        is_last = cute.arch.shuffle_sync(decision, 0) != 0
                        cute.arch.barrier(barrier_id=4, number_of_threads=384)
                        if is_last:
                            if tid < 256:
                                for offset in cutlass.range(self.block_q):
                                    row = row_base + offset
                                    if row < num_q_tokens:
                                        required = min(
                                            self.top_k,
                                            mCuSeqLenKEnd[row] - mCuSeqLenKStart[row],
                                        )
                                        if counts[row] < required:
                                            finish_recovery(
                                                row,
                                                tid,
                                                task_candidates,
                                                counts,
                                                mCuSeqLenKStart,
                                                mCuSeqLenKEnd,
                                                task_logits,
                                                hist,
                                                scalar,
                                                output,
                                                thresholds,
                                                self.top_k,
                                                self.capacity,
                                                self.output_dtype,
                                                self.packed,
                                                recovery_samples,
                                            )
                            subset_barrier(barrier_id=14, number_of_threads=384)
                            if tid == 0:
                                recovery_counters[queued_task] = 0
                        subset_barrier(barrier_id=14, number_of_threads=384)
                        # Retire all readers before reusing row scratch and its task counter.
                        cute.arch.cluster_arrive()
                        cute.arch.cluster_wait()
                slot = slot + cute.arch.grid_dim()[2] // self.splits
            subset_barrier(barrier_id=14, number_of_threads=384)
            task_allocator.free(task_tmem_ptr)
            subset_barrier(barrier_id=14, number_of_threads=384)
        if cute.arch.warp_idx() == 0:
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
