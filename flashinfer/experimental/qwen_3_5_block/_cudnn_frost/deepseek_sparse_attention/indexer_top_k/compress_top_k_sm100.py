# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTeDSL stage-2 top-k over the *compact* (compressed-logits) cand_buffer.

Stage 1 (the logits GEMM epilogue, in compress mode) writes only the valid
ratio-causal scores into a compact per-row variable-length buffer:

    cand_buffer[b * per_batch_floats + row_offset(r) + col]
        for col in [0, row_end_col(r)),
        row_end_col(r) = clamp((q_global_start + r + 1) // ratio, 0, seqlen_k),
        row_offset(r)  = G(q_global_start + r) - G(q_global_start),
        G(n)           = sum_{m=0}^{n} floor(m / ratio)
                       = ratio * Q*(Q-1)/2 + Q*(S+1)   (n = ratio*Q + S),
        per_batch_floats = row_offset(seqlen_q).

The implicit column index (position within a row's slab) IS the local KV id, so
no (value, index) packing is needed in stage 1.

Stage 2 (this kernel): one block per (row, batch) does a 3-level radix-select
(11+11+10 bits over a monotonic float key) and emits the per-row top-K as
**both** local KV indices (int32, -1 padded) and the selected logits (fp32,
-inf padded). It can also emit the fused Top-K softmax so the KL-loss backward
can reuse the probabilities instead of re-gathering / recomputing index scores.

The float→key twiddle and bin extraction:
    twiddle(v): bits = bitcast<u32>(v);  (bits & 0x80000000) ? ~bits : bits | 0x80000000
    bin0 = (key >> 21) & 0x7FF   # 2048 bins
    bin1 = (key >> 10) & 0x7FF   # 2048 bins
    bin2 =  key        & 0x3FF   # 1024 bins
so a larger float maps to a larger key and the top-K largest values are the
K highest keys.

Deterministic ties use a scalar minimum when one index is needed, otherwise
an index-ordered prefix scan of the original row. The unordered shrink buffer
cannot establish the smallest-index tie order.
"""

from __future__ import annotations

from typing import Optional

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, const_expr
from cutlass._mlir.dialects import llvm
from flashinfer.experimental.qwen_3_5_block._cudnn_frost._cutlass_compat import SmemAllocator
from .indexer_top_k_varlen_util import atomicAdd

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.compiler import compile_options
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.runtime import device_major, resolve_stream
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.tensor_conversion import (
    to_cute_tensor as _to_cute_tensor,
)

# Radix configuration.
_NUM_BINS_11 = 2048  # 11-bit levels (bin0, bin1)
_NUM_BINS_10 = 1024  # 10-bit level (bin2)
# Shrink-buffer capacity: candidates landing in the boundary bin0 (== thr_bin0)
# are staged here so passes 3/4 refine over this (small) set instead of
# re-scanning the full row.  If it overflows (rare; clustered distributions)
# the refinement passes fall back to a full-row re-scan.  16 KB smem (val+idx).
_SHRINK_MAX = 2048


def _make_i64_cand_buffer_compile_tensor():
    """Create the 1D candidate-buffer ABI with a 64-bit dynamic extent."""
    return cute.runtime.make_fake_tensor(
        dtype=cutlass.Float32,
        shape=(cute.sym_int64(symbol="cand_buffer_numel"),),
        stride=(1,),
        assumed_align=16,
    )


def _float_as_uint32(float_val):
    """Reinterpret an fp32 register's bits as uint32 (no value change)."""
    return llvm.bitcast(cutlass.Uint32.mlir_type, float_val.ir_value())


# ---------------------------------------------------------------------------
# Block-wide inclusive prefix sum (self-contained, no external dependency).
# ---------------------------------------------------------------------------
@cute.jit
def _warp_scan_inclusive(val: Int32, lane_id, num_threads_per_warp: cutlass.Constexpr):
    mask_val = const_expr(((1 << num_threads_per_warp) - 1) & 0xFFFFFFFF)
    iteration = cute.arch.log2_of_pow2_int(Int32(num_threads_per_warp))
    for i in cutlass.range(iteration, unroll_full=True):
        offset = 1 << i
        other = cute.arch.shuffle_sync_up(val, offset, mask=mask_val, mask_and_clamp=0)
        if lane_id >= offset:
            val = val + other
    return val


@cute.jit
def _block_scan_inclusive(
    val: Int32,
    s_warp_sums: cute.Tensor,
    tidx,
    num_threads,
    num_warps: cutlass.Constexpr,
):
    """Inclusive prefix sum of one int32 per thread across the whole block."""
    warp_id = tidx // 32
    lane_id = tidx % 32

    val = _warp_scan_inclusive(val, lane_id, 32)
    if lane_id == 31:
        s_warp_sums[warp_id] = val
    cute.arch.barrier(barrier_id=1, number_of_threads=num_threads)

    if warp_id == 0:
        if lane_id < num_warps:
            warp_val = s_warp_sums[lane_id]
            warp_val = _warp_scan_inclusive(warp_val, lane_id, num_warps)
            s_warp_sums[lane_id] = warp_val
    cute.arch.barrier(barrier_id=1, number_of_threads=num_threads)

    if warp_id > 0:
        val = val + s_warp_sums[warp_id - 1]
    return val


# ---------------------------------------------------------------------------
# Block-wide fp32 max / sum all-reduce (for the optional top-k softmax epilogue).
# ---------------------------------------------------------------------------
@cute.jit
def _warp_reduce_f32(val: Float32, is_max: cutlass.Constexpr) -> Float32:
    """Butterfly all-reduce of one fp32 per lane across a full 32-lane warp.

    ``is_max`` picks max vs. sum; the result is broadcast to every lane.
    """
    for i in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_bfly(val, 1 << i, mask=0xFFFFFFFF, mask_and_clamp=31)
        if const_expr(is_max):
            val = cute.arch.fmax(val, other)
        else:
            val = val + other
    return val


@cute.jit
def _block_reduce_f32(
    val: Float32,
    s_partials: cute.Tensor,
    s_bcast: cute.Tensor,
    tidx,
    num_warps: cutlass.Constexpr,
    is_max: cutlass.Constexpr,
) -> Float32:
    """Block-wide fp32 max/sum reduction; result broadcast to all threads.

    ``s_partials`` holds one fp32 per warp and ``s_bcast`` one fp32 for the final
    broadcast. Barriers are included so max and sum reductions can reuse them.
    """
    warp_id = tidx // 32
    lane_id = tidx % 32
    val = _warp_reduce_f32(val, is_max)
    if lane_id == 0:
        s_partials[warp_id] = val
    cute.arch.barrier()
    if tidx == 0:
        acc = s_partials[0]
        for w in cutlass.range_constexpr(1, num_warps):
            other = s_partials[w]
            if const_expr(is_max):
                acc = cute.arch.fmax(acc, other)
            else:
                acc = acc + other
        s_bcast[0] = acc
    cute.arch.barrier()
    return s_bcast[0]


class CompressTopkStage2:
    """Stage-2 radix top-k over the compact cand_buffer (BSHD).

    Compile-time attributes: ``topk`` (K), ``ratio``, ``block_threads``.
    """

    def __init__(
        self,
        topk: int,
        ratio: int,
        block_threads: int = 512,
        is_varlen: bool = False,
        deterministic: bool = False,
    ):
        assert block_threads % 32 == 0
        assert _NUM_BINS_11 % block_threads == 0 and _NUM_BINS_10 % block_threads == 0, "block_threads must divide both 2048 and 1024"
        self.topk = int(topk)
        self.ratio = int(ratio)
        self.block_threads = int(block_threads)
        self.num_warps = self.block_threads // 32
        # Varlen/THD: grid=(total_q,1), one block per global query token; the block
        # locates its batch from cu_seqlens and reads the tight compact slab via
        # cand_batch_offsets.  BSHD (default): grid=(seqlen_q, bs), uniform layout.
        self.is_varlen = bool(is_varlen)
        # Compile-time opt-in: exact-value ties at the K-th boundary select the
        # smallest local KV indices. The default path keeps its existing shared
        # memory footprint and scheduling-dependent tie-break.
        self.deterministic = bool(deterministic)

    # ----- compact-layout arithmetic (Int64) -----
    @cute.jit
    def _g_prefix(self, n: Int64) -> Int64:
        ratio = Int64(self.ratio)
        Q = n // ratio
        S = n - Q * ratio
        return ratio * Q * (Q - Int64(1)) // Int64(2) + Q * (S + Int64(1))

    @cute.jit
    def _row_offset(self, row: Int64, q_global_start: Int64) -> Int64:
        return self._g_prefix(q_global_start + row) - self._g_prefix(q_global_start)

    # ----- float key twiddle + bin extraction -----
    @cute.jit
    def _twiddle(self, v: Float32):
        bits = _float_as_uint32(v)
        key = cutlass.Uint32(0)
        if bits & 0x80000000:
            key = bits ^ cutlass.Uint32(0xFFFFFFFF)
        else:
            key = bits | cutlass.Uint32(0x80000000)
        return cutlass.Uint32(key)

    # ----- one radix threshold pass over a histogram (re-zeroes hist) -----
    @cute.jit
    def _find_threshold(self, s_hist, num_bins: cutlass.Constexpr, need: Int32, s_warp_sums, s_thr, tidx):
        """Find the bin (scanning high→low) where the cumulative count crosses
        ``need``.  Writes s_thr[0]=threshold_bin, s_thr[1]=count_taken_from_bin.
        Reads-and-zeroes every bin so the next pass starts clean."""
        ITEMS = const_expr(num_bins // self.block_threads)

        if tidx == 0:
            s_thr[0] = Int32(-1)
            s_thr[1] = Int32(0)
        cute.arch.barrier()

        # Read + zero this thread's bins (descending rank), accumulate local sum.
        local_sum = Int32(0)
        reg = [Int32(0) for _ in range(ITEMS)]
        for i in cutlass.range_constexpr(ITEMS):
            b = num_bins - 1 - (tidx * ITEMS + i)
            c = s_hist[b]
            s_hist[b] = Int32(0)
            reg[i] = c
            local_sum = local_sum + c

        incl = _block_scan_inclusive(local_sum, s_warp_sums, tidx, self.block_threads, self.num_warps)
        running = incl - local_sum  # exclusive prefix = count in strictly-higher bins

        for i in cutlass.range_constexpr(ITEMS):
            b = num_bins - 1 - (tidx * ITEMS + i)
            c = reg[i]
            if c > Int32(0) and running < need and running + c >= need:
                s_thr[0] = Int32(b)
                s_thr[1] = need - running
            running = running + c
        cute.arch.barrier()
        return s_thr[0], s_thr[1]

    @cute.kernel
    def kernel(
        self,
        mCand: cute.Tensor,  # 1D compact buffer FP32
        mIdx: cute.Tensor,  # BSHD (bs, seqlen_q, topk) / varlen (total_q, 1, topk) INT32
        mVal: cute.Tensor,  # same shape FP32
        seqlen_q: Int32,
        seqlen_k: Int32,
        mCuSeqlensQ: cute.Tensor | None = None,
        mCuSeqlensK: cute.Tensor | None = None,
        mCandBatchOffsets: cute.Tensor | None = None,
        mQCausalOffsets: cute.Tensor | None = None,
        mSoftmax: cute.Tensor | None = None,
        mKvLens: cute.Tensor | None = None,
    ):
        BT = const_expr(self.block_threads)
        K = const_expr(self.topk)
        ratio = const_expr(self.ratio)

        tidx = cute.arch.thread_idx()[0]
        # Resolve (batch b, local row q) and the output slot (ob, oq).  Varlen:
        # grid=(total_q,1), one block per global token t; scan cu_seqlens_q for its
        # batch (block-uniform), so EVERY block is a valid row (no padding-block
        # guard needed).  BSHD: grid=(seqlen_q, bs), block index IS (q, b).
        if const_expr(self.is_varlen):
            t = cute.arch.block_idx()[0]
            b = Int32(0)
            while mCuSeqlensQ[b + Int32(1)] <= t:
                b = b + Int32(1)
            offset_q = mCuSeqlensQ[b]
            q = t - offset_q
            seqlen_q_eff = mCuSeqlensQ[b + Int32(1)] - offset_q
            if const_expr(mKvLens is None):
                seqlen_k_eff = mCuSeqlensK[b + Int32(1)] - mCuSeqlensK[b]
            ob = t
            oq = Int32(0)
        else:
            q = cute.arch.block_idx()[0]
            b = cute.arch.block_idx()[1]
            seqlen_q_eff = seqlen_q
            if const_expr(mKvLens is None):
                seqlen_k_eff = seqlen_k
            ob = b
            oq = q
        # Paged K: the per-batch key count comes from kv_lens, not from cu_seqlens_k. It is
        # bounded by the static key extent, like the score kernel bounds its writes.
        if const_expr(mKvLens is not None):
            seqlen_k_eff = mKvLens[b]
            if seqlen_k_eff > seqlen_k:
                seqlen_k_eff = seqlen_k

        # Per-batch causal offset (q_causal_offset; default 0 when mQCausalOffsets is
        # None) — same convention as the dense indexer.
        if const_expr(mQCausalOffsets is None):
            q_global_start = Int32(0)
        else:
            q_global_start = mQCausalOffsets[b]
        # row_end_col = clamp((q_global_start + q + 1)//ratio, 0, seqlen_k_eff)
        row_end_col = (q_global_start + q + Int32(1)) // Int32(ratio)
        if row_end_col > seqlen_k_eff:
            row_end_col = seqlen_k_eff
        if row_end_col < Int32(0):
            row_end_col = Int32(0)
        seg_len = row_end_col

        qgs64 = Int64(q_global_start)
        row_off = self._row_offset(Int64(q), qgs64)
        if const_expr(mCandBatchOffsets is not None):
            # Per-batch slab base (prefix sum): varlen always, and BSHD when per-batch
            # offsets make per_batch_floats non-uniform.  See _compress_cand_batch_offsets.
            row_base = Int64(mCandBatchOffsets[b]) + row_off
        else:
            # BSHD uniform: batch_idx * this batch's per_batch_floats (qgs uniform).
            per_batch_floats = self._row_offset(Int64(seqlen_q), qgs64)
            row_base = Int64(b) * per_batch_floats + row_off

        # ---- shared memory ----
        smem = SmemAllocator()
        s_hist = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_ordered_layout((_NUM_BINS_11,), order=(0,)),
            byte_alignment=128,
        )
        s_warp_sums = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_ordered_layout((self.num_warps,), order=(0,)),
            byte_alignment=128,
        )
        s_thr = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_ordered_layout((2,), order=(0,)),
            byte_alignment=128,
        )
        s_ctl = smem.allocate_tensor(  # [0]=found [1]=eq_emitted [2]=shrink_cnt [3]=overflow
            element_type=Int32,
            layout=cute.make_ordered_layout((4,), order=(0,)),
            byte_alignment=128,
        )
        s_shrink_val = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_ordered_layout((_SHRINK_MAX,), order=(0,)),
            byte_alignment=128,
        )
        s_shrink_idx = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_ordered_layout((_SHRINK_MAX,), order=(0,)),
            byte_alignment=128,
        )
        if tidx == 0:
            s_ctl[0] = Int32(0)
            s_ctl[1] = Int32(0)
            s_ctl[2] = Int32(0)
            s_ctl[3] = Int32(0)
        # No barrier here: these block-init scalars are first read only in
        # Pass 2 / the refinement passes, each of which already sits behind the
        # Pass-1 hist-clear + accumulate + find_threshold barriers (which publish
        # them); the seg_len<=K path never reads them.

        neg_inf = -Float32.inf

        if seg_len <= Int32(K):
            # Fewer candidates than K: copy all, pad the rest.
            for slot in range(tidx, K, BT):
                if slot < seg_len:
                    mVal[ob, oq, slot] = mCand[row_base + Int64(slot)]
                    mIdx[ob, oq, slot] = Int32(slot)
                else:
                    mVal[ob, oq, slot] = neg_inf
                    mIdx[ob, oq, slot] = Int32(-1)
        else:
            # ---- Pass 1: bin0 histogram over the full row ----
            for i in range(tidx, _NUM_BINS_11, BT):
                s_hist[i] = Int32(0)
            cute.arch.barrier()

            for i in range(tidx, seg_len, BT):
                key = self._twiddle(mCand[row_base + Int64(i)])
                b0 = Int32((key >> 21) & cutlass.Uint32(0x7FF))
                atomicAdd(s_hist.iterator + b0, Int32(1))
            cute.arch.barrier()

            thr_bin0, thr_count0 = self._find_threshold(s_hist, _NUM_BINS_11, Int32(K), s_warp_sums, s_thr, tidx)

            # ---- Pass 2: emit bin0 winners + build bin1 histogram ----
            for i in range(tidx, seg_len, BT):
                v = mCand[row_base + Int64(i)]
                key = self._twiddle(v)
                b0 = Int32((key >> 21) & cutlass.Uint32(0x7FF))
                if thr_bin0 < Int32(0) or b0 > thr_bin0:
                    dst = atomicAdd(s_ctl.iterator + Int32(0), Int32(1))
                    if dst < Int32(K):
                        mVal[ob, oq, dst] = v
                        mIdx[ob, oq, dst] = Int32(i)
                elif b0 == thr_bin0:
                    b1 = Int32((key >> 10) & cutlass.Uint32(0x7FF))
                    atomicAdd(s_hist.iterator + b1, Int32(1))
                    # Stage the boundary candidate (value + local KV id) so the
                    # bin1/bin2 refinement passes scan this small set, not the row.
                    slot = atomicAdd(s_ctl.iterator + Int32(2), Int32(1))
                    if slot < Int32(_SHRINK_MAX):
                        s_shrink_val[slot] = v
                        s_shrink_idx[slot] = Int32(i)
                    else:
                        s_ctl[3] = Int32(1)
            cute.arch.barrier()

            if thr_count0 > Int32(0):
                thr_bin1, thr_count1 = self._find_threshold(s_hist, _NUM_BINS_11, thr_count0, s_warp_sums, s_thr, tidx)
                shrink_count = s_ctl[2]
                use_shrink = s_ctl[3] == Int32(0)

                # ---- Pass 3: emit bin1 winners + build bin2 histogram ----
                # Iterate the (small) shrink buffer when it didn't overflow, else
                # fall back to re-scanning the full row (filtered by bin0).
                if use_shrink:
                    for i in range(tidx, shrink_count, BT):
                        v = s_shrink_val[i]
                        idx = s_shrink_idx[i]
                        key = self._twiddle(v)
                        b1 = Int32((key >> 10) & cutlass.Uint32(0x7FF))
                        if thr_bin1 < Int32(0) or b1 > thr_bin1:
                            dst = atomicAdd(s_ctl.iterator + Int32(0), Int32(1))
                            if dst < Int32(K):
                                mVal[ob, oq, dst] = v
                                mIdx[ob, oq, dst] = idx
                        elif b1 == thr_bin1:
                            b2 = Int32(key & cutlass.Uint32(0x3FF))
                            atomicAdd(s_hist.iterator + b2, Int32(1))
                else:
                    for i in range(tidx, seg_len, BT):
                        v = mCand[row_base + Int64(i)]
                        key = self._twiddle(v)
                        b0 = Int32((key >> 21) & cutlass.Uint32(0x7FF))
                        if b0 == thr_bin0:
                            b1 = Int32((key >> 10) & cutlass.Uint32(0x7FF))
                            if thr_bin1 < Int32(0) or b1 > thr_bin1:
                                dst = atomicAdd(s_ctl.iterator + Int32(0), Int32(1))
                                if dst < Int32(K):
                                    mVal[ob, oq, dst] = v
                                    mIdx[ob, oq, dst] = Int32(i)
                            elif b1 == thr_bin1:
                                b2 = Int32(key & cutlass.Uint32(0x3FF))
                                atomicAdd(s_hist.iterator + b2, Int32(1))
                cute.arch.barrier()

                if thr_count1 > Int32(0):
                    thr_bin2, thr_count2 = self._find_threshold(s_hist, _NUM_BINS_10, thr_count1, s_warp_sums, s_thr, tidx)

                    if const_expr(self.deterministic):
                        if tidx == 0:
                            s_ctl[1] = seg_len
                        cute.arch.barrier()

                    # ---- Pass 4: emit bin2 winners (gt) + exactly thr_count2 (eq) ----
                    if use_shrink:
                        for i in range(tidx, shrink_count, BT):
                            v = s_shrink_val[i]
                            idx = s_shrink_idx[i]
                            key = self._twiddle(v)
                            b1 = Int32((key >> 10) & cutlass.Uint32(0x7FF))
                            if b1 == thr_bin1:
                                b2 = Int32(key & cutlass.Uint32(0x3FF))
                                if thr_bin2 < Int32(0) or b2 > thr_bin2:
                                    dst = atomicAdd(s_ctl.iterator + Int32(0), Int32(1))
                                    if dst < Int32(K):
                                        mVal[ob, oq, dst] = v
                                        mIdx[ob, oq, dst] = idx
                                elif b2 == thr_bin2:
                                    if const_expr(self.deterministic):
                                        if thr_count2 == Int32(1):
                                            cute.arch.atomic_min(s_ctl.iterator + Int32(1), idx, sem="relaxed", scope="cta")
                                    else:
                                        slot = atomicAdd(s_ctl.iterator + Int32(1), Int32(1))
                                        if slot < thr_count2:
                                            dst = atomicAdd(s_ctl.iterator + Int32(0), Int32(1))
                                            if dst < Int32(K):
                                                mVal[ob, oq, dst] = v
                                                mIdx[ob, oq, dst] = idx
                    else:
                        for i in range(tidx, seg_len, BT):
                            v = mCand[row_base + Int64(i)]
                            key = self._twiddle(v)
                            b0 = Int32((key >> 21) & cutlass.Uint32(0x7FF))
                            if b0 == thr_bin0:
                                b1 = Int32((key >> 10) & cutlass.Uint32(0x7FF))
                                if b1 == thr_bin1:
                                    b2 = Int32(key & cutlass.Uint32(0x3FF))
                                    if thr_bin2 < Int32(0) or b2 > thr_bin2:
                                        dst = atomicAdd(s_ctl.iterator + Int32(0), Int32(1))
                                        if dst < Int32(K):
                                            mVal[ob, oq, dst] = v
                                            mIdx[ob, oq, dst] = Int32(i)
                                    elif b2 == thr_bin2:
                                        if const_expr(self.deterministic):
                                            if thr_count2 == Int32(1):
                                                cute.arch.atomic_min(s_ctl.iterator + Int32(1), Int32(i), sem="relaxed", scope="cta")
                                        else:
                                            slot = atomicAdd(s_ctl.iterator + Int32(1), Int32(1))
                                            if slot < thr_count2:
                                                dst = atomicAdd(s_ctl.iterator + Int32(0), Int32(1))
                                                if dst < Int32(K):
                                                    mVal[ob, oq, dst] = v
                                                    mIdx[ob, oq, dst] = Int32(i)
                    cute.arch.barrier()
                    if const_expr(self.deterministic):
                        strict_count = s_ctl[0]
                        cute.arch.barrier()
                        if thr_count2 == Int32(1):
                            tie_idx = s_ctl[1]
                            if (tidx == 0) & (tie_idx < seg_len) & (strict_count < Int32(K)):
                                mVal[ob, oq, strict_count] = mCand[row_base + Int64(tie_idx)]
                                mIdx[ob, oq, strict_count] = tie_idx
                                s_ctl[0] = strict_count + Int32(1)
                        else:
                            threshold_key = (cutlass.Uint32(thr_bin0) << 21) | (cutlass.Uint32(thr_bin1) << 10) | cutlass.Uint32(thr_bin2)
                            tie_count = Int32(0)
                            tile_start = Int32(0)
                            while (tie_count < thr_count2) & (tile_start < seg_len):
                                column = tile_start + tidx
                                value = Float32(0.0)
                                matching = Int32(0)
                                if column < seg_len:
                                    value = mCand[row_base + Int64(column)]
                                    matching = Int32(self._twiddle(value) == threshold_key)
                                rank = _block_scan_inclusive(matching, s_warp_sums, tidx, BT, self.num_warps)
                                destination = strict_count + tie_count + rank - Int32(1)
                                if (matching != Int32(0)) & (tie_count + rank <= thr_count2) & (destination < Int32(K)):
                                    mVal[ob, oq, destination] = value
                                    mIdx[ob, oq, destination] = Int32(column)
                                tie_count = tie_count + s_warp_sums[self.num_warps - 1]
                                cute.arch.barrier()
                                tile_start = tile_start + Int32(BT)
                            if tidx == 0:
                                s_ctl[0] = strict_count + cutlass.min(tie_count, thr_count2)
                        cute.arch.barrier()

            # ---- Pad the unfilled tail ----
            total_found = s_ctl[0]
            if total_found > Int32(K):
                total_found = Int32(K)
            for slot in range(tidx, K, BT):
                if slot >= total_found:
                    mVal[ob, oq, slot] = neg_inf
                    mIdx[ob, oq, slot] = Int32(-1)

        # ---- Optional softmax over the K selected logits (fused epilogue) ----
        # The top-k-only path omits this work entirely when mSoftmax is None.
        if const_expr(mSoftmax is not None):
            s_freduce = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_ordered_layout((self.num_warps,), order=(0,)),
                byte_alignment=128,
            )
            s_fbcast = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )
            # Winners and padding have all been stored before they are read back.
            cute.arch.barrier()

            local_max = neg_inf
            for slot in range(tidx, K, BT):
                local_max = cute.arch.fmax(local_max, mVal[ob, oq, slot])
            row_max = _block_reduce_f32(local_max, s_freduce, s_fbcast, tidx, self.num_warps, True)

            if row_max == neg_inf:
                # A row with no candidates has an all-zero softmax.
                for slot in range(tidx, K, BT):
                    mSoftmax[ob, oq, slot] = Float32(0.0)
            else:
                local_sum = Float32(0.0)
                for slot in range(tidx, K, BT):
                    local_sum = local_sum + cute.math.exp(mVal[ob, oq, slot] - row_max, fastmath=True)
                row_sum = _block_reduce_f32(local_sum, s_freduce, s_fbcast, tidx, self.num_warps, False)
                inv_sum = Float32(1.0) / row_sum
                for slot in range(tidx, K, BT):
                    # -inf padding maps exactly to zero.
                    mSoftmax[ob, oq, slot] = cute.math.exp(mVal[ob, oq, slot] - row_max, fastmath=True) * inv_sum

    @cute.jit
    def __call__(
        self,
        mCand: cute.Tensor,
        mIdx: cute.Tensor,
        mVal: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        stream: cuda.CUstream,
        mCuSeqlensQ: cute.Tensor | None = None,
        mCuSeqlensK: cute.Tensor | None = None,
        mCandBatchOffsets: cute.Tensor | None = None,
        mQCausalOffsets: cute.Tensor | None = None,
        mSoftmax: cute.Tensor | None = None,
        mKvLens: cute.Tensor | None = None,
    ):
        if const_expr(self.is_varlen):
            # grid = (total_q, 1): one block per global query token.
            total_q = cute.size(mIdx.shape[0])
            grid = (total_q, 1, 1)
        else:
            bs = cute.size(mIdx.shape[0])
            sq = cute.size(mIdx.shape[1])
            grid = (sq, bs, 1)
        self.kernel(
            mCand,
            mIdx,
            mVal,
            seqlen_q,
            seqlen_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCandBatchOffsets,
            mQCausalOffsets,
            mSoftmax,
            mKvLens,
        ).launch(
            grid=grid,
            block=(self.block_threads, 1, 1),
            stream=stream,
        )


_compile_cache: dict[tuple, object] = {}


def per_batch_floats(seqlen_q: int, seqlen_k: int, ratio: int, q_causal_offset: int = 0) -> int:
    """Host-side compact per-batch float count = row_offset(seqlen_q) under the causal
    offset ``q_causal_offset`` (default 0 = top-left; matches the dense indexer).

    ``seqlen_k`` is accepted for call-compatibility but does not affect the layout
    (the offset does)."""
    if q_causal_offset < 0:
        raise ValueError(f"q_causal_offset must be >= 0, got {q_causal_offset}")

    def G(n: int) -> int:
        Q = n // ratio
        S = n - Q * ratio
        return ratio * Q * (Q - 1) // 2 + Q * (S + 1)

    return G(q_causal_offset + seqlen_q) - G(q_causal_offset)


def build_compact_buffer(dense_scores: torch.Tensor, ratio: int, q_causal_offset: int = 0) -> torch.Tensor:
    """Reference (torch) builder: pack a dense (bs, sq, sk) score tensor into the
    compact cand_buffer layout under causal offset ``q_causal_offset`` (default 0 =
    top-left, matching the dense indexer).
    Used by the standalone unit test to feed the stage-2 kernel without the GEMM."""
    assert dense_scores.ndim == 3
    bs, sq, sk = dense_scores.shape
    qgs = q_causal_offset
    if qgs < 0:
        raise ValueError("q_causal_offset must be >= 0")
    pbf = per_batch_floats(sq, sk, ratio, qgs)
    cand = torch.empty(bs * pbf, dtype=torch.float32, device=dense_scores.device)

    def G(n: int) -> int:
        Q = n // ratio
        S = n - Q * ratio
        return ratio * Q * (Q - 1) // 2 + Q * (S + 1)

    g0 = G(qgs)
    for r in range(sq):
        row_end = max(0, min(sk, (qgs + r + 1) // ratio))
        if row_end == 0:
            continue
        row_off = G(qgs + r) - g0
        for bb in range(bs):
            base = bb * pbf + row_off
            cand[base : base + row_end] = dense_scores[bb, r, :row_end].to(torch.float32)
    return cand


def _validate_out_buffer(t, name, out_shape, dtype, device):
    """Caller-provided stage-2 output buffer: validate (or allocate when None).

    The kernel writes via a leading-dim-dynamic cute layout (the last/topk dim must
    be contiguous) and on ``device``; a wrong-device or non-last-dim-contiguous
    buffer is undefined behaviour.  A view that is strided in the batch/row dims
    (e.g. a microbatch window slice ``out[:, rs:rs+mb, :]``) IS allowed — only
    ``stride(-1) == 1`` is required, not full contiguity."""
    if t is None:
        return torch.empty(out_shape, dtype=dtype, device=device)
    if tuple(t.shape) != out_shape or t.dtype != dtype or not t.is_cuda or t.device != device or t.stride(-1) != 1:
        raise ValueError(f"{name} must be a CUDA {dtype} tensor of shape {out_shape} on {device} " f"with a contiguous last (topk) dim")
    return t


def _validate_i32_vector(t, name, device):
    if not t.is_cuda or t.ndim != 1 or t.dtype != torch.int32 or t.stride(0) != 1 or t.device != device:
        raise ValueError(f"{name} must be a contiguous 1D int32 CUDA tensor on the cand device")


def compress_stage2_topk(
    cand_buffer: torch.Tensor,
    bs: int,
    seqlen_q: int,
    seqlen_k: int,
    topk: int,
    ratio: int,
    block_threads: int = 512,
    stream: Optional[cuda.CUstream] = None,
    out_indices: Optional[torch.Tensor] = None,
    out_logits: Optional[torch.Tensor] = None,
    cand_batch_offsets: Optional[torch.Tensor] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    out_softmax: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
    deterministic: bool = False,
    kv_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ...]:
    """Run the stage-2 radix top-k over a compact cand_buffer.

    ``q_causal_offsets`` ((bs,) int32, default None=0): per-batch causal offset (top-left
    convention, matching the dense indexer).  When given, also pass ``cand_batch_offsets``
    ((bs+1,) int64 prefix sum of per-batch compact sizes) so each batch's slab base is
    exact (per-batch pbf is non-uniform); both None ⇒ uniform offset-0 (slab base =
    b * per_batch_floats).

    ``kv_lens`` ((bs,) int32) replaces the uniform ``seqlen_k`` as each batch's visible
    key count (paged K, where the pool holds more rows than any sequence uses).

    When ``out_indices``/``out_logits`` are given the kernel writes into them
    directly (allocation-free / CUDA-graph friendly) instead of allocating; they
    must be ``(bs, seqlen_q, topk)`` int32/fp32 CUDA tensors and MAY be strided
    views of a larger output (e.g. a microbatch window ``out[:, rs:rs+mb, :]`` —
    the kernel writes via the tensor layout, so the per-window result lands in the
    full output with no temp + copy).  Returns those same tensors.

    When ``return_softmax`` is true, or ``out_softmax`` is supplied, the
    kernel also emits the FP32 softmax over the selected logits. Padding maps
    to zero. ``out_softmax`` follows the same buffer rules as ``out_logits``.

    When ``deterministic`` is true, exact-value ties at the K-th boundary select
    the smallest local KV indices, making the selected set reproducible. The
    within-row slot order remains unspecified. The default false path retains
    the faster scheduling-dependent tie-break.

    Returns (topk_indices (bs, sq, topk) int32 local KV ids; -1 padded,
             topk_logits  (bs, sq, topk) fp32; -inf padded), plus
    topk_softmax (bs, sq, topk) fp32 when requested.
    """
    if device_major() < 9:
        raise RuntimeError("CuTeDSL compress stage-2 topk requires SM90+")
    if not cand_buffer.is_cuda or cand_buffer.dtype != torch.float32:
        raise ValueError("cand_buffer must be a CUDA float32 tensor")
    if not cand_buffer.is_contiguous():
        # Read in place; .contiguous() would silently copy (extra allocation).
        raise ValueError("cand_buffer must be contiguous")
    cand_buffer = cand_buffer.view(-1)

    # Per-batch causal offsets make per_batch_floats non-uniform, so the slab base
    # for batch b must be the prefix sum (cand_batch_offsets), not b * pbf_b; the
    # uniform fallback would read the wrong slab.  Require them together (the main
    # indexer_fwd_compress_topk path always passes both).
    if q_causal_offsets is not None and cand_batch_offsets is None:
        raise ValueError(
            "compress_stage2_topk: q_causal_offsets requires cand_batch_offsets "
            "((bs+1,) int64 prefix sum of per-batch compact sizes, e.g. from "
            "compress_topk_cand_buffer_size / _bshd_cand_batch_offsets sized with the "
            "SAME offsets); without it the kernel assumes a uniform per-batch slab "
            "base (b * per_batch_floats) and reads the wrong slab when offsets differ."
        )

    device = cand_buffer.device
    if kv_lens is not None:
        # The kernel reads kv_lens[b] for every batch, so a short vector is an out-of-bounds read.
        _validate_i32_vector(kv_lens, "kv_lens", device)
        if kv_lens.numel() != bs:
            raise ValueError(f"kv_lens must have shape ({bs},), got {tuple(kv_lens.shape)}")
    out_shape = (bs, seqlen_q, topk)
    out_indices = _validate_out_buffer(out_indices, "out_indices", out_shape, torch.int32, device)
    out_logits = _validate_out_buffer(out_logits, "out_logits", out_shape, torch.float32, device)
    want_softmax = return_softmax or out_softmax is not None
    if want_softmax:
        out_softmax = _validate_out_buffer(out_softmax, "out_softmax", out_shape, torch.float32, device)

    # Outputs are written with per-element scalar stores (no vectorisation) and
    # may be strided slices of a larger tensor, so build them with element-size
    # alignment (4B) instead of assuming a 16B base pointer.
    def _out_cute(t):
        return _to_cute_tensor(t, assumed_align=4)

    softmax_cute = _out_cute(out_softmax) if want_softmax else None
    cbo_cute = _to_cute_tensor(cand_batch_offsets, leading_dim=0) if cand_batch_offsets is not None else None
    qco_cute = _to_cute_tensor(q_causal_offsets, leading_dim=0) if q_causal_offsets is not None else None
    kvl_cute = _to_cute_tensor(kv_lens, assumed_align=4, leading_dim=0) if kv_lens is not None else None

    stream = resolve_stream(stream)
    compile_key = (
        bs,
        seqlen_q,
        seqlen_k if kv_lens is None else None,  # paged: the key extent is a runtime bound
        topk,
        ratio,
        block_threads,
        cand_batch_offsets is not None,
        q_causal_offsets is not None,
        want_softmax,
        bool(deterministic),
    )
    if kv_lens is not None:
        compile_key = ("paged", *compile_key)
    if compile_key not in _compile_cache:
        kernel_obj = CompressTopkStage2(topk=topk, ratio=ratio, block_threads=block_threads, deterministic=deterministic)
        _compile_cache[compile_key] = cute.compile(
            kernel_obj,
            _to_cute_tensor(cand_buffer, leading_dim=0),
            _out_cute(out_indices),
            _out_cute(out_logits),
            cutlass.Int32(int(seqlen_q)),
            cutlass.Int32(int(seqlen_k)),
            stream,
            None,
            None,
            cbo_cute,
            qco_cute,
            softmax_cute,
            kvl_cute,
            options=compile_options("--opt-level 3"),
        )

    _compile_cache[compile_key](
        cand_buffer,
        out_indices,
        out_logits,
        cutlass.Int32(int(seqlen_q)),
        cutlass.Int32(int(seqlen_k)),
        stream,
        None,
        None,
        cand_batch_offsets,
        q_causal_offsets,
        out_softmax,
        kv_lens,
    )
    if want_softmax:
        return out_indices, out_logits, out_softmax
    return out_indices, out_logits


def compress_stage2_topk_varlen(
    cand_buffer: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor],
    cand_batch_offsets: torch.Tensor,
    total_q: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    topk: int,
    ratio: int,
    block_threads: int = 512,
    stream: Optional[cuda.CUstream] = None,
    out_indices: Optional[torch.Tensor] = None,
    out_logits: Optional[torch.Tensor] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    out_softmax: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
    deterministic: bool = False,
    kv_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ...]:
    """Native varlen/THD stage-2 top-k over the tight per-batch compact cand_buffer.

    ``q_causal_offsets`` ((bs,) int32, default None=0): per-batch causal offset (top-left
    convention, matching the dense indexer).  ``cand_batch_offsets`` must be the matching
    prefix sum (``compress_topk_cand_buffer_size_thd`` with the same offsets).

    ``kv_lens`` ((bs,) int32) carries each batch's visible key count for paged K; it
    replaces ``cu_seqlens_k``, which may then be ``None``.

    ONE launch (grid=(total_q, 1)); each block is a global query token, locates its
    batch by scanning ``cu_seqlens_q``, and reads its compact slab at
    ``cand_batch_offsets[b]`` (int64 (bs+1,) prefix sum).  No CPU sync, no per-batch
    launch — CUDA-graph friendly.

    Returns (idx (total_q, topk) int32 LOCAL KV ids in [0, seqlen_k_b); -1 padded,
             val (total_q, topk) fp32; -inf padded), plus topk_softmax
    (total_q, topk) fp32 when ``return_softmax`` is true or ``out_softmax``
    is supplied.

    ``deterministic=True`` applies the smallest-local-index tie-break described
    by :func:`compress_stage2_topk`.

    VALUE CONTRACT (caller-guaranteed; the kernel batch-scans cu_seqlens_q per token
    and would read past the end / mis-slab otherwise): cu_seqlens_q/k start at 0 and
    are monotonically non-decreasing, ``total_q == cu_seqlens_q[-1]``, and
    ``cand_batch_offsets`` are the prefix sums from
    ``compress_topk_cand_buffer_size_thd`` (consistent with cu_seqlens / ratio).  The
    interface validates metadata only and never reads these device values back to
    the host."""
    if device_major() < 9:
        raise RuntimeError("CuTeDSL compress stage-2 topk requires SM90+")
    if not cand_buffer.is_cuda or cand_buffer.dtype != torch.float32:
        raise ValueError("cand_buffer must be a CUDA float32 tensor")
    if not cand_buffer.is_contiguous():
        raise ValueError("cand_buffer must be contiguous")
    cand_buffer = cand_buffer.view(-1)
    device = cand_buffer.device
    bs = cu_seqlens_q.numel() - 1
    # int64 + contiguous so the .to(torch.int64) below is a guaranteed no-op (no
    # extra alloc / copy).
    if (
        not cand_batch_offsets.is_cuda
        or cand_batch_offsets.ndim != 1
        or cand_batch_offsets.numel() != bs + 1
        or cand_batch_offsets.dtype != torch.int64
        or not cand_batch_offsets.is_contiguous()
        or cand_batch_offsets.device != device
    ):
        raise ValueError(
            "cand_batch_offsets must be a contiguous 1D int64 CUDA tensor of length " "bs+1 on the cand device (use compress_topk_cand_buffer_size_thd)"
        )
    # Validate cu_seqlens rather than silently .to()-converting them: this is an
    # exported low-level API, and a silent cast would allocate/copy, breaking the
    # stage-2 no-extra-allocation / graph contract (the public THD entry already
    # enforces int32/contiguous/same-device before calling here).  cand_batch_offsets
    # is int64-validated above, so all three are used directly — no copy.
    if (cu_seqlens_k is None) == (kv_lens is None):
        raise ValueError("compress_stage2_topk_varlen takes exactly one of cu_seqlens_k and kv_lens")
    for t, name in ((cu_seqlens_q, "cu_seqlens_q"), (cu_seqlens_k, "cu_seqlens_k"), (kv_lens, "kv_lens")):
        if t is not None:
            _validate_i32_vector(t, name, device)
    # cu_seqlens_q/k must have the SAME length bs+1: bs is derived from
    # cu_seqlens_q, and the kernel reads mCuSeqlensK[b+1] for every such batch b,
    # so a shorter cu_seqlens_k is an out-of-bounds read.  (Structural, sync-free.)
    if bs < 1:
        raise ValueError(f"cu_seqlens_q must have length bs+1 >= 2; got numel {cu_seqlens_q.numel()}")
    if cu_seqlens_k is not None and cu_seqlens_k.numel() != cu_seqlens_q.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have equal length bs+1 (>= 2); got " f"numel {cu_seqlens_q.numel()} and {cu_seqlens_k.numel()}")
    if kv_lens is not None and kv_lens.numel() != bs:
        raise ValueError(f"kv_lens must have shape ({bs},), got {tuple(kv_lens.shape)}")
    out_shape = (total_q, topk)
    out_indices = _validate_out_buffer(out_indices, "out_indices", out_shape, torch.int32, device)
    out_logits = _validate_out_buffer(out_logits, "out_logits", out_shape, torch.float32, device)
    want_softmax = return_softmax or out_softmax is not None
    if want_softmax:
        out_softmax = _validate_out_buffer(out_softmax, "out_softmax", out_shape, torch.float32, device)
    # The kernel addresses the output as (total_q, 1, topk).
    idx3 = out_indices.view(total_q, 1, topk)
    val3 = out_logits.view(total_q, 1, topk)
    softmax3 = out_softmax.view(total_q, 1, topk) if want_softmax else None

    def _out_cute(t):
        return _to_cute_tensor(t, assumed_align=4)

    softmax3_cute = _out_cute(softmax3) if want_softmax else None

    stream = resolve_stream(stream)
    # Layout is fully dynamic: grid (total_q, 1) and all per-batch geometry
    # (cu_seqlens, cand_batch_offsets) are read at runtime (mark_layout_dynamic),
    # so one compile per (topk, ratio, block_threads) serves any total_q / bs /
    # output layout — hence total_q/bs are not in the compile key
    # (see test_indexer_fwd_dsl_compress_topk_thd_reuse).
    qco_cute = _to_cute_tensor(q_causal_offsets, leading_dim=0) if q_causal_offsets is not None else None
    cu_k_cute = _to_cute_tensor(cu_seqlens_k, leading_dim=0) if cu_seqlens_k is not None else None
    kvl_cute = _to_cute_tensor(kv_lens, assumed_align=4, leading_dim=0) if kv_lens is not None else None
    compile_key = (
        "varlen" if kv_lens is None else "varlen_paged",
        topk,
        ratio,
        block_threads,
        q_causal_offsets is not None,
        want_softmax,
        bool(deterministic),
    )
    if compile_key not in _compile_cache:
        kernel_obj = CompressTopkStage2(
            topk=topk,
            ratio=ratio,
            block_threads=block_threads,
            is_varlen=True,
            deterministic=deterministic,
        )
        _compile_cache[compile_key] = cute.compile(
            kernel_obj,
            _make_i64_cand_buffer_compile_tensor(),
            _out_cute(idx3),
            _out_cute(val3),
            cutlass.Int32(int(max_seqlen_q)),
            cutlass.Int32(int(max_seqlen_k)),
            stream,
            _to_cute_tensor(cu_seqlens_q, leading_dim=0),
            cu_k_cute,
            _to_cute_tensor(cand_batch_offsets, leading_dim=0),
            qco_cute,
            softmax3_cute,
            kvl_cute,
            options=compile_options("--opt-level 3"),
        )
    _compile_cache[compile_key](
        cand_buffer,
        idx3,
        val3,
        cutlass.Int32(int(max_seqlen_q)),
        cutlass.Int32(int(max_seqlen_k)),
        stream,
        cu_seqlens_q,
        cu_seqlens_k,
        cand_batch_offsets,
        q_causal_offsets,
        softmax3,
        kv_lens,
    )
    if want_softmax:
        return out_indices, out_logits, out_softmax
    return out_indices, out_logits


__all__ = [
    "CompressTopkStage2",
    "compress_stage2_topk",
    "compress_stage2_topk_varlen",
    "build_compact_buffer",
    "per_batch_floats",
]
