"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

MSA q2k to k2q CSR builder for SM120/SM121: inverts the per-query top-K
selection into a KV-major CSR plus split-slot packing and a flat work schedule.
"""

import inspect

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import dsl_user_op

# nvvm.atomicrmw's signature comes from the cutlass-dsl libs wheel, not the pinned DSL
# version: libs-base needs a leading ``res`` (result type) arg, libs-cu13 infers it.
_ATOMICRMW_NEEDS_RES = "res" in inspect.signature(nvvm.atomicrmw).parameters


@dsl_user_op
def _atomic_add_i32(a, ptr: cute.Pointer, *, loc=None, ip=None) -> cutlass.Int32:
    """Global int32 atomic add; returns the OLD value (CUDA >= 13.1 nvvm API)."""
    av = cutlass.Int32(a).ir_value()
    if _ATOMICRMW_NEEDS_RES:
        return nvvm.atomicrmw(
            res=av.type, op=nvvm.AtomicOpKind.ADD, ptr=ptr.llvm_ptr, a=av
        )
    return nvvm.atomicrmw(op=nvvm.AtomicOpKind.ADD, ptr=ptr.llvm_ptr, a=av)


class BuildK2qCsrSm12x:
    def __init__(self, topk: int, has_schedule: bool):
        if topk not in (4, 8, 16, 32):
            raise ValueError(f"topk must be 4, 8, 16, or 32, got {topk}")
        self._topk = topk
        self._has_schedule = has_schedule

    @cute.jit
    def __call__(
        self,
        mQ2k: cute.Tensor,  # (H, S_Q, topk) int32
        mRowMap: cute.Tensor,  # (B, max_kv_blocks) int32, -1 = inactive
        mBatchOfQ: cute.Tensor,  # (S_Q,) int32
        mQlocOfQ: cute.Tensor,  # (S_Q,) int32
        mRowCoords: cute.Tensor,  # (total_rows, 2) int32 (batch, kv_block)
        mRowPtr: cute.Tensor,  # (H, total_rows + 1) int32 (out)
        mQIdx: cute.Tensor,  # (H, S_Q * topk) int32 (out)
        mQSplit: cute.Tensor,  # (H, S_Q * topk) int32 (out / dummy)
        mSplitCounts: cute.Tensor,  # (S_Q, H) int32 (out / dummy)
        mSched: cute.Tensor,  # (capacity, 6) int32 (out / dummy)
        mWorkCount: cute.Tensor,  # (1,) int32 (out / dummy)
        mTileCounts: cute.Tensor,  # (H, nchunks, total_rows) int32 (scratch, zeroed)
        mRowCounts: cute.Tensor,  # (H, total_rows) int32 (scratch)
        H: cutlass.Int32,
        S_Q: cutlass.Int32,
        total_rows: cutlass.Int32,
        max_kv_blocks: cutlass.Int32,
        nchunks: cutlass.Int32,
        q_per_chunk: cutlass.Int32,
        target_q_per_cta: cutlass.Int32,
        work_capacity: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._k_hist(
            mQ2k,
            mRowMap,
            mBatchOfQ,
            mTileCounts,
            H,
            S_Q,
            total_rows,
            max_kv_blocks,
            nchunks,
            q_per_chunk,
        ).launch(grid=(nchunks, H, 1), block=(32, 1, 1), stream=stream)

        self._k_prefix_chunks(mTileCounts, mRowCounts, total_rows, nchunks).launch(
            grid=(total_rows, H, 1), block=(32, 1, 1), stream=stream
        )

        self._k_row_ptr(mRowCounts, mRowPtr, total_rows).launch(
            grid=(H, 1, 1), block=(32, 1, 1), stream=stream
        )

        self._k_scatter(
            mQ2k,
            mRowMap,
            mBatchOfQ,
            mQlocOfQ,
            mRowPtr,
            mTileCounts,
            mQIdx,
            mQSplit,
            mSplitCounts,
            H,
            S_Q,
            total_rows,
            max_kv_blocks,
            nchunks,
            q_per_chunk,
        ).launch(grid=(nchunks, H, 1), block=(32, 1, 1), stream=stream)

        if cutlass.const_expr(self._has_schedule):
            self._k_scheduler(
                mRowPtr,
                mRowCoords,
                mSched,
                mWorkCount,
                H,
                total_rows,
                target_q_per_cta,
                work_capacity,
            ).launch(grid=(total_rows, H, 1), block=(1, 1, 1), stream=stream)

    # H: per-(head, chunk) histogram of row hits, one warp per (chunk, head).
    @cute.kernel
    def _k_hist(
        self,
        mQ2k: cute.Tensor,
        mRowMap: cute.Tensor,
        mBatchOfQ: cute.Tensor,
        mTileCounts: cute.Tensor,
        H: cutlass.Int32,
        S_Q: cutlass.Int32,
        total_rows: cutlass.Int32,
        max_kv_blocks: cutlass.Int32,
        nchunks: cutlass.Int32,
        q_per_chunk: cutlass.Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        c, h, _ = cute.arch.block_idx()

        q_start = c * q_per_chunk
        q_end = cutlass.min(q_start + q_per_chunk, S_Q)
        is_slot = lane < self._topk

        q = q_start
        while q < q_end:
            row = cutlass.Int32(-1)
            if is_slot:
                kvb = mQ2k[h, q, lane]
                if kvb >= 0 and kvb < max_kv_blocks:
                    row = mRowMap[mBatchOfQ[q], kvb]
            if row >= 0:
                # distinct slots of one query -> distinct rows, so within an
                # iteration the lanes touch distinct addresses; atomics make the
                # cross-iteration accumulation coherent.
                ptr = mTileCounts[h, c, None].iterator + row
                _atomic_add_i32(1, ptr)
            q += 1

    # PR-chunks: in-place exclusive prefix of tile_counts over the chunk axis, plus
    # per-row totals. One warp per (head, row); the 32 lanes scan the chunk axis in
    # 32-wide tiles with a Hillis-Steele warp scan.
    @cute.kernel
    def _k_prefix_chunks(
        self,
        mTileCounts: cute.Tensor,
        mRowCounts: cute.Tensor,
        total_rows: cutlass.Int32,
        nchunks: cutlass.Int32,
    ):
        row, h, _ = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        if row < total_rows:
            running = cutlass.Int32(0)
            c0 = cutlass.Int32(0)
            while c0 < nchunks:
                c = c0 + lane
                v = cutlass.Int32(0)
                if c < nchunks:
                    v = mTileCounts[h, c, row]
                # inclusive warp scan of this 32-wide tile (Hillis-Steele via
                # idx-shuffle from lane-off; guarded so lane<off keeps its value)
                x = v
                off = 1
                while off < 32:
                    src = lane - off
                    if src < 0:
                        src = cutlass.Int32(0)
                    nbr = cute.arch.shuffle_sync(x, src)
                    if lane >= off:
                        x += nbr
                    off <<= 1
                if c < nchunks:
                    mTileCounts[h, c, row] = running + x - v  # exclusive
                running += cute.arch.shuffle_sync(x, 31)  # tile total -> all lanes
                c0 += 32
            if lane == 0:
                mRowCounts[h, row] = running

    # PR-rows: inclusive prefix of row_counts over rows -> row_ptr[1:]. One warp per
    # head; the 32 lanes scan the row axis in 32-wide tiles (same scan as PR-chunks),
    # so long context (large total_rows) stays cheap.
    @cute.kernel
    def _k_row_ptr(
        self,
        mRowCounts: cute.Tensor,
        mRowPtr: cute.Tensor,
        total_rows: cutlass.Int32,
    ):
        h, _, _ = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        if lane == 0:
            mRowPtr[h, 0] = cutlass.Int32(0)
        running = cutlass.Int32(0)
        r0 = cutlass.Int32(0)
        while r0 < total_rows:
            row = r0 + lane
            v = cutlass.Int32(0)
            if row < total_rows:
                v = mRowCounts[h, row]
            x = v
            off = 1
            while off < 32:
                src = lane - off
                if src < 0:
                    src = cutlass.Int32(0)
                nbr = cute.arch.shuffle_sync(x, src)
                if lane >= off:
                    x += nbr
                off <<= 1
            if row < total_rows:
                mRowPtr[h, row + 1] = running + x  # inclusive prefix over rows
            running += cute.arch.shuffle_sync(x, 31)
            r0 += 32

    # S: scatter queries into CSR order, ascending within each row.
    @cute.kernel
    def _k_scatter(
        self,
        mQ2k: cute.Tensor,
        mRowMap: cute.Tensor,
        mBatchOfQ: cute.Tensor,
        mQlocOfQ: cute.Tensor,
        mRowPtr: cute.Tensor,
        mTileCounts: cute.Tensor,  # now holds per-chunk exclusive base offsets
        mQIdx: cute.Tensor,
        mQSplit: cute.Tensor,
        mSplitCounts: cute.Tensor,
        H: cutlass.Int32,
        S_Q: cutlass.Int32,
        total_rows: cutlass.Int32,
        max_kv_blocks: cutlass.Int32,
        nchunks: cutlass.Int32,
        q_per_chunk: cutlass.Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        c, h, _ = cute.arch.block_idx()

        q_start = c * q_per_chunk
        q_end = cutlass.min(q_start + q_per_chunk, S_Q)
        is_slot = lane < self._topk
        topk_mask = cutlass.Int32(-1) if self._topk == 32 else ((1 << self._topk) - 1)
        lane_lt = (1 << lane) - 1

        q = q_start
        while q < q_end:
            row = cutlass.Int32(-1)
            if is_slot:
                kvb = mQ2k[h, q, lane]
                if kvb >= 0 and kvb < max_kv_blocks:
                    row = mRowMap[mBatchOfQ[q], kvb]
            valid = row >= 0
            vmask = cute.arch.vote_ballot_sync(valid) & topk_mask
            if cutlass.const_expr(self._has_schedule):
                split_slot = cute.arch.popc(vmask & lane_lt)
                if lane == 0:
                    mSplitCounts[q, h] = cute.arch.popc(vmask)
            if valid:
                base_ptr = mTileCounts[h, c, None].iterator + row
                slot = _atomic_add_i32(1, base_ptr)
                pos = mRowPtr[h, row] + slot
                qloc = mQlocOfQ[q]
                mQIdx[h, pos] = qloc
                if cutlass.const_expr(self._has_schedule):
                    mQSplit[h, pos] = qloc | (split_slot << 24)
            q += 1

    # Scheduler: one thread per (kv-head, row); each nonzero row reserves its
    # work-item slots via one atomicAdd on the pre-zeroed global counter. Emission
    # order is irrelevant: the forward consumes work items independently.
    @cute.kernel
    def _k_scheduler(
        self,
        mRowPtr: cute.Tensor,
        mRowCoords: cute.Tensor,
        mSched: cute.Tensor,
        mWorkCount: cute.Tensor,
        H: cutlass.Int32,
        total_rows: cutlass.Int32,
        target_q_per_cta: cutlass.Int32,
        work_capacity: cutlass.Int32,
    ):
        row, h, _ = cute.arch.block_idx()
        if row < total_rows and h < H:
            cnt = mRowPtr[h, row + 1] - mRowPtr[h, row]
            if cnt > 0:
                nch = (cnt + target_q_per_cta - 1) // target_q_per_cta
                base = _atomic_add_i32(nch, mWorkCount.iterator)
                batch_idx = mRowCoords[row, 0]
                kv_block_idx = mRowCoords[row, 1]
                cc = cutlass.Int32(0)
                while cc < nch:
                    widx = base + cc
                    if widx < work_capacity:
                        q_begin = cc * target_q_per_cta
                        q_count = cutlass.min(target_q_per_cta, cnt - q_begin)
                        mSched[widx, 0] = h
                        mSched[widx, 1] = row
                        mSched[widx, 2] = q_begin
                        mSched[widx, 3] = q_count
                        mSched[widx, 4] = batch_idx
                        mSched[widx, 5] = kv_block_idx
                    cc += 1
