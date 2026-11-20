# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 CuTeDSL unified indexer-score kernel.

This class starts from the dense backward indexer-score kernel because that
path already computes both score and LSE.  A compile-time ``compute_lse`` key
lets the same kernel shape run either as:

  - forward score only: write score, skip LSE reduction/write
  - dense backward score: write score and LSE denom

The implementation lives under ``score_recompute``. Indexer forward imports
the same kernel and disables the optional LSE epilogue.
"""

import math

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
import cutlass.cute.nvgpu.tcgen05 as tcgen05

from .dense_score_recompute_sm100 import (
    DenseScoreRecomputeSm100,
    add_packed_f32x2,
    fma_packed_f32x2,
)


class IndexerScoreUnifiedSm100(DenseScoreRecomputeSm100):
    """Dense indexer-score kernel with optional LSE output.

    The packed M tile holds ``q_tokens_per_tile = m_block_size // qhead_per_kvhead`` query tokens of
    ``qhead_per_kvhead`` heads each; the epilogue reduces the ReLU(QK) * W products of every token
    independently and writes one score per (token, key). Head groups of 4 .. 64 are served: a 4-head
    group packs 8 tokens into a 32-column tile, a 64-head group 2 tokens into a 128-column one.
    """

    max_q_tokens_per_tile = 8

    def __init__(
        self,
        *args,
        compute_lse: bool = False,
        is_compressed_logits: bool = False,
        cand_2d: bool = False,
        **kwargs,
    ):
        # Reuse the dense-score base mainloop, but force its epilogue contract
        # to dense indexer score.
        kwargs["score_type"] = "indexer"
        super().__init__(*args, **kwargs)
        # Dense FE entry points remain the default. The compressed forward path
        # opts in and writes only ratio-causal candidates into a compact buffer.
        self.is_compressed_logits = is_compressed_logits
        # BSHD BF16 may expose that compact buffer as (batch, per_batch_floats)
        # so the hot epilogue carries an Int32 column offset instead of a flat
        # Int64 address. Varlen and non-uniform batches retain the flat path.
        self.cand_2d = cand_2d
        # LSE is independent of output layout: dense recompute requests it, and
        # compressed BF16 can request the per-row LSE.
        self.compute_lse = compute_lse

    @cute.jit
    def _g_prefix_i64(self, n):
        """Prefix sum of ratio-causal candidate counts through row ``n``."""
        ratio = Int64(self.ratio)
        quotient = n // ratio
        remainder = n - quotient * ratio
        return ratio * quotient * (quotient - Int64(1)) // Int64(2) + quotient * (remainder + Int64(1))

    @cute.jit
    def _row_offset_i64(self, row, q_global_start):
        return self._g_prefix_i64(q_global_start + row) - self._g_prefix_i64(q_global_start)

    @cute.jit
    def _epilogue_indexer_dense(
        self,
        tiled_mma_qk,
        tStS_ref,
        sW,
        sScoreAll,
        S_mbar_ptr,
        reduce_sync_mbar_ptr,
        mOut,
        mDenom: cute.Tensor | None,
        num_n_blocks_compute,
        seqlen_k,
        seqlen_q,
        max_seqlen_k,
        q_causal_offset,
        m_block,
        tidx,
        s_full_phase_bits,
        reduce_phase,
        softmax_scale,
        per_head_offset=None,
        batch_idx=None,
        cand_batch_offsets=None,
    ):
        """Indexer epilogue shared by score-only and score+LSE modes.

        ``mDenom`` is ``None`` when ``compute_lse`` is off; every read of it
        sits below the ``const_expr(self.compute_lse)`` early return.
        """
        tidx_wg = tidx % self.WARPGROUP_SIZE

        sW_off = Int32(0) if per_head_offset is None else per_head_offset
        qhpkv = self.qhead_per_kvhead
        q_tokens_per_tile = self.q_tokens_per_tile
        ratio = Int32(self.ratio)

        # Epilogue setup: map the packed M tile back to logical q-token rows,
        # compute each row's ratio-causal limit, and cache W in registers.
        # The head reduce walks the group's columns in pairs, W_ILP pairs per
        # unrolled step: 8 pairs for the 16+ head groups, the whole (even) group
        # for the 4- and 8-head ones so no column is left outside the loop.
        W_ILP = min(8, qhpkv // 2)
        sW_1d = cute.make_tensor(
            sW.iterator + sW_off,
            cute.make_layout((self.m_block_size,), stride=(1,)),
        )
        rW_all = cute.make_rmem_tensor((self.m_block_size,), Float32)

        warp_id_in_wg = tidx_wg // self.WARP_SIZE

        log2_e = Float32(math.log2(math.e))

        q_token_base = m_block * q_tokens_per_tile
        q_token_idxs = [q_token_base + Int32(qi) for qi in range(q_tokens_per_tile)]
        col_limits = [(q_causal_offset + q_token_idxs[qi] + Int32(1)) // ratio for qi in range(q_tokens_per_tile)]

        if cutlass.const_expr(self.is_compressed_logits and self.cand_2d):
            q_global_start = Int64(q_causal_offset)
            cand_row_cols = [Int32(self._row_offset_i64(Int64(q_token_idxs[qi]), q_global_start)) for qi in range(q_tokens_per_tile)]
        elif cutlass.const_expr(self.is_compressed_logits):
            q_global_start = Int64(q_causal_offset)
            if cutlass.const_expr(cand_batch_offsets is not None):
                cand_batch_base = Int64(cand_batch_offsets[batch_idx])
            else:
                cand_per_batch = self._row_offset_i64(Int64(seqlen_q), q_global_start)
                cand_batch_base = Int64(batch_idx) * cand_per_batch
            cand_row_offsets = [self._row_offset_i64(Int64(q_token_idxs[qi]), q_global_start) for qi in range(q_tokens_per_tile)]

        local_max = [-Float32.inf for _ in range(q_tokens_per_tile)]
        local_sum_exp = [Float32(0.0) for _ in range(q_tokens_per_tile)]
        first_block = Int32(1)

        n_blk = num_n_blocks_compute - Int32(1)
        while n_blk >= Int32(0):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tmem_repetition)),
                Float32,
            )
            thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_ref).get_slice(tidx_wg)
            thr_mma = tiled_mma_qk.get_slice(tidx_wg)
            cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
            tScS = thr_tmem_load.partition_D(thr_mma.partition_C(cS))
            tSrS_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(tStS_ref.shape)).shape
            tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

            # Consume one completed QK accumulator tile from TMEM, then release
            # the slot back to the mainloop before doing register-only work.
            slot = n_blk % Int32(self.num_tmem_slots)
            s_full_phase = self._phase_for_slot(s_full_phase_bits, slot)
            cute.arch.mbarrier_wait(S_mbar_ptr + 2 * slot, s_full_phase)

            if first_block == Int32(1):
                cute.autovec_copy(sW_1d, rW_all)
                first_block = Int32(0)

            tmem_ptr_cur = cute.make_ptr(
                Float32,
                slot * self.tmem_s_stride,
                mem_space=cute.AddressSpace.tmem,
                assumed_align=16,
            )
            tStS_cur = cute.make_tensor(tmem_ptr_cur, tStS_ref.layout)
            tStS_t2r_cur = thr_tmem_load.partition_S(tStS_cur)

            cute.copy(thr_tmem_load, tStS_t2r_cur, tSrS)
            cute.arch.fence_view_async_tmem_load()

            cute.arch.mbarrier_arrive(S_mbar_ptr + 2 * slot + 1)
            s_full_phase_bits = self._toggle_phase_for_slot(
                s_full_phase_bits,
                slot,
            )

            kv_offset = tScS[0][0]
            pos = kv_offset + n_blk * self.n_block_size

            # For each q token in the packed m tile, reduce ReLU(QK) * W over
            # heads and write one dense score column.  Independent partial sums
            # keep the FMA dependency chain short.  Invalid rows/columns are
            # skipped so the caller's prefilled invalid value remains intact.
            for qi in cutlass.range_constexpr(q_tokens_per_tile):
                q_token_idx = q_token_idxs[qi]
                col_limit = col_limits[qi]
                if q_token_idx < seqlen_q and pos < col_limit and pos < seqlen_k:
                    acc_score_0 = (Float32(0.0), Float32(0.0))
                    acc_score_1 = (Float32(0.0), Float32(0.0))
                    acc_score_2 = (Float32(0.0), Float32(0.0))
                    acc_score_3 = (Float32(0.0), Float32(0.0))
                    use_two_sums = not (qhpkv == 64 and self.ratio == 1 and self.compute_lse)

                    for ho in cutlass.range_constexpr(qhpkv // 2 // W_ILP):
                        for ci in cutlass.range_constexpr(W_ILP):
                            idx0 = qi * qhpkv + (ho * W_ILP + ci) * 2
                            idx1 = idx0 + 1
                            w_pair = (rW_all[idx0], rW_all[idx1])

                            val0 = tSrS[idx0]
                            val0 = val0 if val0 > Float32(0.0) else Float32(0.0)
                            val1 = tSrS[idx1]
                            val1 = val1 if val1 > Float32(0.0) else Float32(0.0)

                            if cutlass.const_expr((ci % 2 == 0) if use_two_sums else (ci < W_ILP // 4)):
                                acc_score_0 = fma_packed_f32x2(
                                    (val0, val1),
                                    w_pair,
                                    acc_score_0,
                                )
                            elif cutlass.const_expr(use_two_sums or ci < W_ILP // 2):
                                acc_score_1 = fma_packed_f32x2(
                                    (val0, val1),
                                    w_pair,
                                    acc_score_1,
                                )
                            elif cutlass.const_expr(ci < (W_ILP * 3) // 4):
                                acc_score_2 = fma_packed_f32x2((val0, val1), w_pair, acc_score_2)
                            else:
                                acc_score_3 = fma_packed_f32x2((val0, val1), w_pair, acc_score_3)

                    acc_score = add_packed_f32x2(acc_score_0, acc_score_1)
                    if cutlass.const_expr(not use_two_sums):
                        acc_score = add_packed_f32x2(acc_score, add_packed_f32x2(acc_score_2, acc_score_3))
                    score = (acc_score[0] + acc_score[1]) * Float32(softmax_scale)
                    if cutlass.const_expr(self.is_compressed_logits and self.cand_2d):
                        mOut[batch_idx, cand_row_cols[qi] + pos] = score
                    elif cutlass.const_expr(self.is_compressed_logits):
                        mOut[cand_batch_base + cand_row_offsets[qi] + Int64(pos)] = score
                    else:
                        mOut[q_token_idx, pos] = score

                    if cutlass.const_expr(self.compute_lse):
                        # Online log-sum-exp over the full score row. This runs
                        # for dense backward and compressed forward with LSE.
                        new_max = score if score > local_max[qi] else local_max[qi]
                        local_rescale = cute.math.exp2((local_max[qi] - new_max) * log2_e)
                        local_sum_exp[qi] = local_sum_exp[qi] * local_rescale + cute.math.exp2((score - new_max) * log2_e)
                        local_max[qi] = new_max
            n_blk = n_blk - 1

        if cutlass.const_expr(not self.compute_lse):
            return s_full_phase_bits, reduce_phase

        # Finish LSE by reducing each q token's online max/sum across the
        # epilogue warpgroup.  Keeping one logical row per reduction is a little
        # more work than the old dual-row fast path, but it avoids sharing
        # scratch state between independent rows and is stable under CuTeDSL
        # 4.5.0's stricter materialization/legalization rules.
        sScoreAll_sum = cute.make_tensor(
            sScoreAll.iterator + self.num_warps_in_epi_wg,
            cute.make_layout((self.num_warps_in_epi_wg,), stride=(1,)),
        )
        inv_log2_e = Float32(1.0 / math.log2(math.e))

        for qi in cutlass.range_constexpr(q_tokens_per_tile):
            global_max, reduce_phase = self._intra_inter_warp_reduce_max(
                sScoreAll,
                reduce_sync_mbar_ptr,
                reduce_phase,
                warp_id_in_wg,
                local_max[qi],
            )

            lse_val = -Float32.inf
            if global_max > Float32(-1e30):
                global_rescale = cute.math.exp2((local_max[qi] - global_max) * log2_e)
                adjusted_sum = local_sum_exp[qi] * global_rescale

                global_sum_exp, reduce_phase = self._intra_inter_warp_reduce_sum(
                    sScoreAll_sum,
                    reduce_sync_mbar_ptr,
                    reduce_phase,
                    warp_id_in_wg,
                    adjusted_sum,
                )

                lse_val = global_max + cute.math.log2(global_sum_exp) * inv_log2_e
            else:
                cute.arch.mbarrier_arrive(reduce_sync_mbar_ptr)
                cute.arch.mbarrier_wait(reduce_sync_mbar_ptr, reduce_phase)
                reduce_phase = reduce_phase ^ 1

            q_token_idx = q_token_base + qi
            if q_token_idx < seqlen_q:
                with cute.arch.elect_one():
                    mDenom[q_token_idx] = lse_val

        return s_full_phase_bits, reduce_phase
