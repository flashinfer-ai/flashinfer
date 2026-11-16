# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 indexer forward kernel - direct CuTeDSL port of the old C++ path.

This kernel intentionally keeps the SM90 C++ Q/K staging topology:
  * WG0 issues Q/K TMA loads and loads weights
  * consumer warpgroups run 64x64x16 WGMMA stages and the head-reduce epilogue

The algorithm computes the same score tensor as the C++ kernel:
    sm_scale * sum_h relu(Q_h @ K^T) * W_h
with ratio causal masking. q_causal_offsets defaults to 0 when omitted.

The FP8 path uses FP8 Q/K WGMMA with Hopper-style 1x128 FP32 descales. Q
descale is folded into the per-head W tile, and K descale is applied after the
head reduction. BF16 and FP8 both write reduced scores directly from registers
to global memory.
"""

import math
from typing import Optional, Type
import operator

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass import Float32, Int32, Boolean, const_expr
from flashinfer.experimental.qwen_3_5_block._cudnn_frost._cutlass_compat import LayoutEnum, OperandMajorMode, SmemAllocator

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils import copy as copy_utils
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.seqlen import SeqlenInfoQK
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.sm90 import mma as sm90_mma
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.sm90 import primitives as sm90_ops
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.sm90.mma import gemm_w_idx

LSE_BARRIER_STAGE0 = 10
LSE_BARRIER_STAGE1 = 11


def _mma_partition_fragment_AB(
    thr_mma: cute.ThrMma,
    sA: cute.Tensor,
    sB: cute.Tensor,
    swap_AB: bool,
):
    if const_expr(not swap_AB):
        return thr_mma.make_fragment_A(thr_mma.partition_A(sA)), thr_mma.make_fragment_B(thr_mma.partition_B(sB))
    return thr_mma.make_fragment_B(thr_mma.partition_B(sA)), thr_mma.make_fragment_A(thr_mma.partition_A(sB))


class IndexerForwardSm90:
    """Direct CuTeDSL translation of the SM90 forward kernel."""

    def __init__(
        self,
        qk_dtype: Type[cutlass.Numeric],
        w_dtype: Type[cutlass.Numeric],
        head_dim: int = 128,
        qhead_per_kvhead: int = 64,
        ratio: int = 4,
        is_varlen: bool = False,
        use_unchecked_qh64: bool = False,
        use_unchecked_qh64_masked: bool = False,
        compute_lse: bool = False,
    ):
        assert head_dim == 128, f"SM90 direct forward supports head_dim=128, got {head_dim}"
        supported_qhpkv = (32, 64) if qk_dtype == cutlass.Float8E4M3FN else (16, 32, 64)
        assert qhead_per_kvhead in supported_qhpkv, f"SM90 direct forward supports qhpkv in {supported_qhpkv}, got {qhead_per_kvhead}"
        assert ratio >= 1, f"ratio must be >=1, got {ratio}"
        self.qk_dtype = qk_dtype
        self.w_dtype = w_dtype
        self.use_fp8_scales = qk_dtype == cutlass.Float8E4M3FN
        self.head_dim = head_dim
        self.tile_hdim = int(math.ceil(head_dim / 16) * 16)
        self.qhead_per_kvhead = qhead_per_kvhead
        self.ratio = ratio
        self.is_varlen = is_varlen
        self.use_split_q_wg = self.use_fp8_scales
        self.use_fp8_prescaled_w = self.use_fp8_scales and w_dtype == Float32

        self.tile_m = 128
        self.mma_tile_n = 64
        self.tile_n = self.mma_tile_n
        self.q_stages = 2
        self.kv_stages = 2
        self.q_per_stage = self.tile_m // self.q_stages
        self.q_tokens_per_tile = self.tile_m // self.qhead_per_kvhead
        self.q_tokens_per_stage = self.q_tokens_per_tile // self.q_stages
        assert self.q_per_stage in (64, 128)
        assert self.q_tokens_per_stage * self.qhead_per_kvhead == self.q_per_stage
        # qh16 packs eight query tokens into one CTA. Their ratio-causal limits
        # can straddle a 64-column boundary, so both rightmost N blocks need
        # elementwise masking. Keep the existing one-block path unchanged for
        # qh32/qh64 so those specializations lower exactly as before.
        self.mask_two_rightmost_nblocks = self.qhead_per_kvhead == 16

        self.num_threads = 384 if self.use_split_q_wg else 256
        self.num_threads_per_warp_group = 128
        self.swap_AB = True
        self.num_acc_n_rows_per_thread = self.mma_tile_n // 32
        self.use_fast_qh64_epilogue = self.use_fp8_scales and self.use_split_q_wg and self.qhead_per_kvhead == 64 and self.q_tokens_per_stage == 1
        self.use_unchecked_qh64_full = use_unchecked_qh64 and self.use_fast_qh64_epilogue and not self.is_varlen
        self.use_unchecked_qh64_masked = use_unchecked_qh64_masked and self.use_fast_qh64_epilogue and not self.is_varlen
        self.compute_lse = compute_lse

    def _setup_attributes(self):
        self.sQ_layout_single = sm90_mma.make_smem_layout(self.qk_dtype, LayoutEnum.ROW_MAJOR, (self.q_per_stage, self.tile_hdim), stage=None)
        self.sQ_layout_staged = sm90_mma.make_smem_layout(self.qk_dtype, LayoutEnum.ROW_MAJOR, (self.q_per_stage, self.tile_hdim), stage=self.q_stages)
        self.sKV_layout_single = sm90_mma.make_smem_layout(self.qk_dtype, LayoutEnum.ROW_MAJOR, (self.mma_tile_n, self.tile_hdim), stage=None)
        self.sKV_layout_staged = sm90_mma.make_smem_layout(
            self.qk_dtype,
            LayoutEnum.ROW_MAJOR,
            (self.mma_tile_n, self.tile_hdim),
            stage=self.kv_stages,
        )

    def _get_tiled_mma(self):
        return sm90_utils_basic.make_trivial_tiled_mma(
            self.qk_dtype,
            self.qk_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(self.mma_tile_n, self.q_per_stage),
        )

    def _get_shared_storage_cls(self):
        sQ_struct = cute.struct.Align[cute.struct.MemRange[self.qk_dtype, cute.cosize(self.sQ_layout_staged)], 1024]
        sKV_struct = cute.struct.Align[cute.struct.MemRange[self.qk_dtype, cute.cosize(self.sKV_layout_staged)], 1024]
        sW_dtype = Float32 if self.use_fp8_scales else self.w_dtype
        sW_struct = cute.struct.Align[cute.struct.MemRange[sW_dtype, self.tile_m], 128]
        sKScale_struct = cute.struct.Align[cute.struct.MemRange[Float32, self.tile_n * self.kv_stages], 128]
        sLseReduce_struct = cute.struct.Align[cute.struct.MemRange[Float32, 32], 128]

        @cute.struct
        class SharedStorage:
            mbar_Q0: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_Q1: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_KV0: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_KV1: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_KVEmpty0: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_KVEmpty1: cute.struct.MemRange[cutlass.Int64, 2]
            sW: sW_struct
            sKScale: sKScale_struct
            sLseReduce: sLseReduce_struct
            sQ: sQ_struct
            sKV: sKV_struct

        return SharedStorage

    @cute.jit
    def _accumulate_lse_score(
        self,
        score: Float32,
        lse_max: cute.Tensor,
        lse_sum: cute.Tensor,
        qi: cutlass.Constexpr[int],
    ):
        LOG2_E = Float32(1.4426950408889634)
        old_max = lse_max[qi]
        new_max = cute.arch.fmax(old_max, score)
        lse_sum[qi] = lse_sum[qi] * cute.math.exp2((old_max - new_max) * LOG2_E, fastmath=True) + cute.math.exp2((score - new_max) * LOG2_E, fastmath=True)
        lse_max[qi] = new_max

    @cute.jit
    def _write_lse_from_local_accum(
        self,
        mLse: cute.Tensor,
        sLseReduce: cute.Tensor,
        lse_max: cute.Tensor,
        lse_sum: cute.Tensor,
        q_stage_idx: cutlass.Constexpr[int],
        lse_reduce_base: cutlass.Constexpr[int],
        lse_barrier_id: cutlass.Constexpr[int],
        m_block: Int32,
        seqlen_q: Int32,
        q_batch_offset: Int32,
        batch_idx: Int32,
        wg_tidx: Int32,
    ):
        warp_id = wg_tidx // Int32(32)
        LOG2_E = Float32(1.4426950408889634)

        for qi in cutlass.range_constexpr(self.q_tokens_per_stage):
            q_local = (self.q_stages * m_block + q_stage_idx) * self.q_tokens_per_stage + qi

            thread_max = lse_max[qi]
            warp_max = thread_max
            for i in cutlass.range_constexpr(5):
                warp_max = cute.arch.fmax(
                    warp_max,
                    cute.arch.shuffle_sync_bfly(warp_max, offset=1 << i),
                )
            with cute.arch.elect_one():
                sLseReduce[lse_reduce_base + qi * 8 + warp_id] = warp_max

            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=lse_barrier_id,
                number_of_threads=self.num_threads_per_warp_group,
            )

            reduce_base = lse_reduce_base + qi * 8
            global_max = cute.arch.fmax(
                cute.arch.fmax(sLseReduce[reduce_base + 0], sLseReduce[reduce_base + 1]),
                cute.arch.fmax(sLseReduce[reduce_base + 2], sLseReduce[reduce_base + 3]),
            )

            local_sum = Float32(0.0)
            if global_max > Float32(-1e30):
                local_sum = lse_sum[qi] * cute.math.exp2(
                    (thread_max - global_max) * LOG2_E,
                    fastmath=True,
                )

            warp_sum = cute.arch.warp_reduction_sum(local_sum)
            with cute.arch.elect_one():
                sLseReduce[reduce_base + 4 + warp_id] = warp_sum

            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=lse_barrier_id,
                number_of_threads=self.num_threads_per_warp_group,
            )

            if wg_tidx == 0 and q_local < seqlen_q:
                global_sum = sLseReduce[reduce_base + 4] + sLseReduce[reduce_base + 5]
                global_sum = global_sum + sLseReduce[reduce_base + 6] + sLseReduce[reduce_base + 7]
                lse = -Float32.inf
                if global_max > Float32(-1e30):
                    lse = sm90_ops.logf(global_sum) + global_max
                if const_expr(self.is_varlen):
                    mLse[q_batch_offset + q_local] = lse
                else:
                    mLse[batch_idx, q_local] = lse

    @cute.jit
    def _compute_n_blocks(
        self,
        m_block: Int32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        q_causal_offset: Int32,
    ) -> Int32:
        last_q_token = (m_block + Int32(1)) * Int32(self.q_tokens_per_tile) - Int32(1)
        kv_limit = (q_causal_offset + last_q_token + Int32(1)) // Int32(self.ratio)
        kv_limit = kv_limit if kv_limit < seqlen_k else seqlen_k
        kv_limit = kv_limit if kv_limit > Int32(0) else Int32(0)
        return cute.ceil_div(kv_limit, self.tile_n)

    @cute.jit
    def _iter_n_block(self, iter_idx: Int32, n_block_max: Int32) -> Int32:
        if const_expr(self.use_fp8_scales):
            return n_block_max - Int32(1) - iter_idx
        local_count = n_block_max if n_block_max < Int32(3) else Int32(3)
        return n_block_max - Int32(1) - iter_idx if iter_idx < local_count else iter_idx - local_count

    @cute.jit
    def _issue_tma(
        self,
        gmem_cur: cute.Tensor,
        smem_tile: cute.Tensor,
        block_idx: Int32,
        tma_atom: cute.CopyAtom,
        mbar_ptr,
        copy_bytes: cutlass.Constexpr[int],
        tile_rows: cutlass.Constexpr[int],
    ):
        g_tile = cute.local_tile(gmem_cur, (tile_rows, self.tile_hdim), (block_idx, 0))
        load_fn, _, _ = copy_utils.tma_get_copy_fn(tma_atom, 0, cute.make_layout(1), g_tile, smem_tile, single_stage=True)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, copy_bytes)
        load_fn(tma_bar_ptr=mbar_ptr)

    @cute.jit
    def _load_weights(
        self,
        mW_cur: cute.Tensor,
        mQScale_cur: Optional[cute.Tensor],
        sW: cute.Tensor,
        m_block: Int32,
        seqlen_q: Int32,
        sm_scale: Float32,
        lane_id: Int32,
    ):
        rows_per_thread = (self.tile_m + 31) // 32
        for ri in cutlass.range_constexpr(rows_per_thread):
            row = ri * 32 + lane_id
            if row < self.tile_m:
                m_packed = m_block * self.tile_m + row
                q_idx = m_packed // self.qhead_per_kvhead
                if q_idx < seqlen_q:
                    if const_expr(self.use_fp8_scales):
                        w = Float32(mW_cur[m_packed])
                        if const_expr(not self.use_fp8_prescaled_w):
                            w = w * Float32(mQScale_cur[m_packed])
                            w = w * sm_scale
                        sW[row] = w
                    else:
                        sW[row] = mW_cur[m_packed]
                else:
                    if const_expr(self.use_fp8_scales):
                        sW[row] = Float32(0.0)
                    else:
                        sW[row] = self.w_dtype(0.0)

    @cute.jit
    def _load_k_scales(
        self,
        mKScale_cur: cute.Tensor,
        sKScale: cute.Tensor,
        stage: Int32,
        n_block: Int32,
        seqlen_k: Int32,
        lane_id: Int32,
    ):
        rows_per_thread = (self.tile_n + 31) // 32
        for ri in cutlass.range_constexpr(rows_per_thread):
            row = ri * 32 + lane_id
            if row < self.tile_n:
                k_idx = n_block * self.tile_n + row
                if k_idx < seqlen_k:
                    sKScale[stage * self.tile_n + row] = Float32(mKScale_cur[k_idx])
                else:
                    sKScale[stage * self.tile_n + row] = Float32(0.0)

    @cute.jit
    def _epilogue_store_to_gmem(
        self,
        q_stage_idx: cutlass.Constexpr[int],
        acc_S: cute.Tensor,
        tWeights: cute.Tensor,
        rKScale: Optional[cute.Tensor],
        mOut: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        apply_causal_mask: Boolean,
        seqlen_q: Int32,
        seqlen_k: Int32,
        max_seqlen_k: Int32,
        q_batch_offset: Int32,
        batch_idx: Int32,
        q_causal_offset: Int32,
        sm_scale: Float32,
        kv_group_offset: Int32,
        lse_max: cute.Tensor = None,
        lse_sum: cute.Tensor = None,
    ):
        acc_mn = sm90_ops.make_acc_tensor_mn_view(acc_S)
        kNRows = cute.size(acc_mn, mode=[0])
        kNCols = cute.size(acc_mn, mode=[1])
        kColsPerQToken = kNCols // self.q_tokens_per_stage

        lane = cute.arch.lane_idx()
        t0 = lane % 4
        t1 = lane // 4
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        m_base = t1 + warp_idx_in_wg * kNRows * 8
        q_token_base = (self.q_stages * m_block + q_stage_idx) * self.q_tokens_per_stage

        ps = cute.make_rmem_tensor((kNRows, self.q_tokens_per_stage), Float32)
        ps.fill(Float32(0.0))

        for qi in cutlass.range_constexpr(self.q_tokens_per_stage):
            for hi in cutlass.range(kColsPerQToken, unroll_full=True):
                ni = qi * kColsPerQToken + hi
                w = Float32(tWeights[ni])
                for mi in cutlass.range(kNRows, unroll_full=True):
                    a = acc_mn[mi, ni]
                    ps[mi, qi] = ps[mi, qi] + cute.arch.fmax(a, Float32(0.0)) * w

        for mi in cutlass.range(kNRows, unroll_full=True):
            for qi in cutlass.range_constexpr(self.q_tokens_per_stage):
                ps[mi, qi] = sm90_ops.warp_reduce(ps[mi, qi], operator.add, width=4)
                if const_expr(not self.use_fp8_scales):
                    ps[mi, qi] = ps[mi, qi] * sm_scale

        if t0 == 0:
            for qi in cutlass.range_constexpr(self.q_tokens_per_stage):
                for mi in cutlass.range(kNRows, unroll_full=True):
                    kv_m = m_base + mi * 8
                    q_local = q_token_base + qi
                    k_local = n_block * self.tile_n + kv_group_offset + kv_m
                    score = ps[mi, qi]
                    should_store = Boolean(True)
                    should_accum_lse = Boolean(True)
                    if apply_causal_mask:
                        col_lim = (q_causal_offset + q_local + Int32(1)) // Int32(self.ratio)
                        if k_local >= seqlen_k or k_local >= col_lim:
                            should_store = Boolean(False)
                            should_accum_lse = Boolean(False)
                    elif k_local >= seqlen_k:
                        should_store = Boolean(False)
                        should_accum_lse = Boolean(False)
                    if const_expr(self.use_fp8_scales):
                        score = score * rKScale[mi]

                    if should_store and q_local < seqlen_q and k_local < max_seqlen_k:
                        if const_expr(self.is_varlen):
                            mOut[q_batch_offset + q_local, k_local] = score
                        else:
                            mOut[batch_idx, q_local, k_local] = score
                    if const_expr(self.compute_lse):
                        if should_accum_lse and q_local < seqlen_q and k_local < max_seqlen_k:
                            self._accumulate_lse_score(score, lse_max, lse_sum, qi)

    @cute.jit
    def _epilogue_store_to_gmem_qh64_full(
        self,
        q_stage_idx: cutlass.Constexpr[int],
        acc_S: cute.Tensor,
        tWeights: cute.Tensor,
        rKScale: cute.Tensor,
        mOut: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        seqlen_q: Int32,
        max_seqlen_k: Int32,
        q_batch_offset: Int32,
        batch_idx: Int32,
        lse_max: cute.Tensor = None,
        lse_sum: cute.Tensor = None,
    ):
        acc_mn = sm90_ops.make_acc_tensor_mn_view(acc_S)
        kNRows = cute.size(acc_mn, mode=[0])
        kNCols = cute.size(acc_mn, mode=[1])

        lane = cute.arch.lane_idx()
        t0 = lane % 4
        t1 = lane // 4
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        m_base = t1 + warp_idx_in_wg * kNRows * 8
        q_local = self.q_stages * m_block + q_stage_idx

        ps = cute.make_rmem_tensor((kNRows,), Float32)
        ps.fill(Float32(0.0))

        for hi in cutlass.range(kNCols, unroll_full=True):
            w = Float32(tWeights[hi])
            for mi in cutlass.range(kNRows, unroll_full=True):
                a = acc_mn[mi, hi]
                ps[mi] = ps[mi] + cute.arch.fmax(a, Float32(0.0)) * w

        for mi in cutlass.range(kNRows, unroll_full=True):
            ps[mi] = sm90_ops.warp_reduce(ps[mi], operator.add, width=4)

        mi_dyn = t0 % 2
        kv_m_pair = m_base + mi_dyn * 8
        k_local_pair = n_block * self.tile_n + kv_m_pair
        score_pair = ps[0] if mi_dyn == 0 else ps[1]
        score_pair = score_pair * rKScale[0]
        if const_expr(self.use_unchecked_qh64_full):
            if const_expr(self.is_varlen):
                mOut[q_batch_offset + q_local, k_local_pair] = score_pair
            else:
                mOut[batch_idx, q_local, k_local_pair] = score_pair
        else:
            if q_local < seqlen_q and k_local_pair < max_seqlen_k:
                if const_expr(self.is_varlen):
                    mOut[q_batch_offset + q_local, k_local_pair] = score_pair
                else:
                    mOut[batch_idx, q_local, k_local_pair] = score_pair
        if const_expr(self.compute_lse):
            if t0 < 2:
                if const_expr(self.use_unchecked_qh64_full):
                    self._accumulate_lse_score(score_pair, lse_max, lse_sum, 0)
                elif q_local < seqlen_q and k_local_pair < max_seqlen_k:
                    self._accumulate_lse_score(score_pair, lse_max, lse_sum, 0)

    @cute.jit
    def _epilogue_store_to_gmem_qh64_masked(
        self,
        q_stage_idx: cutlass.Constexpr[int],
        acc_S: cute.Tensor,
        tWeights: cute.Tensor,
        rKScale: cute.Tensor,
        mOut: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        max_seqlen_k: Int32,
        q_batch_offset: Int32,
        batch_idx: Int32,
        q_causal_offset: Int32,
        lse_max: cute.Tensor = None,
        lse_sum: cute.Tensor = None,
    ):
        acc_mn = sm90_ops.make_acc_tensor_mn_view(acc_S)
        kNRows = cute.size(acc_mn, mode=[0])
        kNCols = cute.size(acc_mn, mode=[1])

        lane = cute.arch.lane_idx()
        t0 = lane % 4
        t1 = lane // 4
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        m_base = t1 + warp_idx_in_wg * kNRows * 8
        q_local = self.q_stages * m_block + q_stage_idx
        col_lim = (q_causal_offset + q_local + Int32(1)) // Int32(self.ratio)

        ps = cute.make_rmem_tensor((kNRows,), Float32)
        ps.fill(Float32(0.0))

        for hi in cutlass.range(kNCols, unroll_full=True):
            w = Float32(tWeights[hi])
            for mi in cutlass.range(kNRows, unroll_full=True):
                a = acc_mn[mi, hi]
                ps[mi] = ps[mi] + cute.arch.fmax(a, Float32(0.0)) * w

        for mi in cutlass.range(kNRows, unroll_full=True):
            ps[mi] = sm90_ops.warp_reduce(ps[mi], operator.add, width=4)

        mi_dyn = t0 % 2
        kv_m_pair = m_base + mi_dyn * 8
        k_local_pair = n_block * self.tile_n + kv_m_pair
        score_pair = ps[0] if mi_dyn == 0 else ps[1]
        score_pair = score_pair * rKScale[0]
        should_store_pair = Boolean(True)
        should_accum_lse_pair = Boolean(True)
        if k_local_pair >= seqlen_k or k_local_pair >= col_lim:
            should_store_pair = Boolean(False)
            should_accum_lse_pair = Boolean(False)
        if const_expr(self.use_unchecked_qh64_masked):
            if should_store_pair:
                if const_expr(self.is_varlen):
                    mOut[q_batch_offset + q_local, k_local_pair] = score_pair
                else:
                    mOut[batch_idx, q_local, k_local_pair] = score_pair
        elif should_store_pair and q_local < seqlen_q and k_local_pair < max_seqlen_k:
            if const_expr(self.is_varlen):
                mOut[q_batch_offset + q_local, k_local_pair] = score_pair
            else:
                mOut[batch_idx, q_local, k_local_pair] = score_pair
        if const_expr(self.compute_lse):
            if t0 < 2:
                if const_expr(self.use_unchecked_qh64_masked):
                    if should_accum_lse_pair:
                        self._accumulate_lse_score(score_pair, lse_max, lse_sum, 0)
                elif should_accum_lse_pair and q_local < seqlen_q and k_local_pair < max_seqlen_k:
                    self._accumulate_lse_score(score_pair, lse_max, lse_sum, 0)

    @cute.jit
    def producer(
        self,
        mQ: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        mK: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        mW: cute.Tensor,
        mQScale: Optional[cute.Tensor],
        mKScale: Optional[cute.Tensor],
        sQ_0: cute.Tensor,
        sQ_1: cute.Tensor,
        sKV_0: cute.Tensor,
        sKV_1: cute.Tensor,
        sW: cute.Tensor,
        sKScale: cute.Tensor,
        mbar_Q0_ptr,
        mbar_Q1_ptr,
        mbar_KV0_ptr,
        mbar_KV1_ptr,
        mbar_KVEmpty0_ptr,
        mbar_KVEmpty1_ptr,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mQCausalOffsets: cute.Tensor,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        sm_scale: Float32,
        work_block_x: Int32,
        work_head_idx: Int32,
        work_batch_idx: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_wg = warp_idx % 4
        lane_id = cute.arch.lane_idx()
        block_x = work_block_x
        head_idx = work_head_idx
        batch_idx = work_batch_idx

        seqlen = SeqlenInfoQK.create(
            batch_idx,
            max_seqlen_q,
            max_seqlen_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            None,
            None,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
        num_m_blocks = cute.ceil_div(seqlen.seqlen_q * self.qhead_per_kvhead, self.tile_m)
        if block_x < num_m_blocks:
            m_block = num_m_blocks - Int32(1) - block_x
            n_block_max = self._compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)

            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx]
            mW_cur = seqlen.offset_batch_Q(mW, batch_idx, dim=2)[None, head_idx]
            if const_expr(self.use_fp8_scales):
                mQScale_cur = seqlen.offset_batch_Q(mQScale, batch_idx, dim=2)[None, head_idx]
                mKScale_cur = seqlen.offset_batch_K(mKScale, batch_idx, dim=2)[None, head_idx]

            q0_phase = Int32(0)
            q1_phase = Int32(0)
            kve0_phase = Int32(0)
            kve1_phase = Int32(0)

            if warp_idx_in_wg == 0:
                if const_expr(self.use_fp8_scales):
                    self._load_weights(mW_cur, mQScale_cur, sW, m_block, seqlen.seqlen_q, sm_scale, lane_id)
                else:
                    self._load_weights(mW_cur, None, sW, m_block, seqlen.seqlen_q, sm_scale, lane_id)
                cute.arch.fence_view_async_shared()
                self._issue_tma(mQ_cur, sQ_0, self.q_stages * m_block, tma_atom_Q, mbar_Q0_ptr, self.tma_copy_bytes_Q, self.q_per_stage)
                self._issue_tma(
                    mQ_cur,
                    sQ_1,
                    self.q_stages * m_block + Int32(1),
                    tma_atom_Q,
                    mbar_Q1_ptr,
                    self.tma_copy_bytes_Q,
                    self.q_per_stage,
                )

                iter_idx = Int32(0)
                if const_expr(self.use_fp8_scales):
                    while iter_idx < n_block_max:
                        n_block = self._iter_n_block(iter_idx, n_block_max)
                        if iter_idx >= Int32(2):
                            cute.arch.mbarrier_wait(mbar_KVEmpty0_ptr, kve0_phase)
                            kve0_phase = kve0_phase ^ Int32(1)
                        self._issue_tma(mK_cur, sKV_0, n_block, tma_atom_K, mbar_KV0_ptr, self.tma_copy_bytes_K, self.tile_n)
                        self._load_k_scales(mKScale_cur, sKScale, Int32(0), n_block, seqlen.seqlen_k, lane_id)
                        cute.arch.fence_view_async_shared()
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mbar_KV0_ptr)
                        iter_idx = iter_idx + Int32(2)
                else:
                    while iter_idx < n_block_max:
                        stage = iter_idx & Int32(1)
                        n_block = self._iter_n_block(iter_idx, n_block_max)
                        if iter_idx >= Int32(2):
                            if stage == Int32(0):
                                cute.arch.mbarrier_wait(mbar_KVEmpty0_ptr, kve0_phase)
                                kve0_phase = kve0_phase ^ Int32(1)
                            else:
                                cute.arch.mbarrier_wait(mbar_KVEmpty1_ptr, kve1_phase)
                                kve1_phase = kve1_phase ^ Int32(1)
                        if stage == Int32(0):
                            self._issue_tma(mK_cur, sKV_0, n_block, tma_atom_K, mbar_KV0_ptr, self.tma_copy_bytes_K, self.tile_n)
                        else:
                            self._issue_tma(mK_cur, sKV_1, n_block, tma_atom_K, mbar_KV1_ptr, self.tma_copy_bytes_K, self.tile_n)
                        iter_idx = iter_idx + Int32(1)

                if n_block_max >= Int32(1):
                    cute.arch.mbarrier_wait(mbar_KVEmpty0_ptr, kve0_phase)
                if const_expr(not self.use_fp8_scales):
                    if n_block_max >= Int32(2):
                        cute.arch.mbarrier_wait(mbar_KVEmpty1_ptr, kve1_phase)
            elif const_expr(self.use_fp8_scales):
                if warp_idx_in_wg == 2:
                    iter_idx = Int32(1)
                    while iter_idx < n_block_max:
                        n_block = self._iter_n_block(iter_idx, n_block_max)
                        if iter_idx >= Int32(2):
                            cute.arch.mbarrier_wait(mbar_KVEmpty1_ptr, kve1_phase)
                            kve1_phase = kve1_phase ^ Int32(1)
                        self._issue_tma(mK_cur, sKV_1, n_block, tma_atom_K, mbar_KV1_ptr, self.tma_copy_bytes_K, self.tile_n)
                        self._load_k_scales(mKScale_cur, sKScale, Int32(1), n_block, seqlen.seqlen_k, lane_id)
                        cute.arch.fence_view_async_shared()
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mbar_KV1_ptr)
                        iter_idx = iter_idx + Int32(2)

                    if n_block_max >= Int32(2):
                        cute.arch.mbarrier_wait(mbar_KVEmpty1_ptr, kve1_phase)

    @cute.jit
    def consumer(
        self,
        tiled_mma_QK: cute.TiledMma,
        sQ_0: cute.Tensor,
        sQ_1: cute.Tensor,
        sKV_staged: cute.Tensor,
        sW: cute.Tensor,
        sKScale: cute.Tensor,
        mOut: cute.Tensor,
        mLse: Optional[cute.Tensor],
        sLseReduce: cute.Tensor,
        mbar_Q0_ptr,
        mbar_Q1_ptr,
        mbar_KV0_ptr,
        mbar_KV1_ptr,
        mbar_KVEmpty0_ptr,
        mbar_KVEmpty1_ptr,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mQCausalOffsets: cute.Tensor,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        sm_scale: Float32,
        work_block_x: Int32,
        work_head_idx: Int32,
        work_batch_idx: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        wg_tidx = tidx % self.num_threads_per_warp_group
        block_x = work_block_x
        head_idx = work_head_idx
        batch_idx = work_batch_idx

        seqlen = SeqlenInfoQK.create(
            batch_idx,
            max_seqlen_q,
            max_seqlen_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            None,
            None,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
        num_m_blocks = cute.ceil_div(seqlen.seqlen_q * self.qhead_per_kvhead, self.tile_m)
        if block_x < num_m_blocks:
            m_block = num_m_blocks - Int32(1) - block_x
            n_block_max = self._compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)

            thr_mma = tiled_mma_QK.get_slice(wg_tidx)
            tSrQ0, tSrK = _mma_partition_fragment_AB(thr_mma, sQ_0, sKV_staged, self.swap_AB)
            tSrQ1, _ = _mma_partition_fragment_AB(thr_mma, sQ_1, sKV_staged, self.swap_AB)
            acc_shape = tiled_mma_QK.partition_shape_C((self.tile_n, self.q_per_stage))

            sW0_mma = cute.make_tensor(
                sW.iterator,
                cute.make_layout((self.q_per_stage, self.tile_n), stride=(1, 0)),
            )
            sW1_mma = cute.make_tensor(
                sW.iterator + self.q_per_stage,
                cute.make_layout((self.q_per_stage, self.tile_n), stride=(1, 0)),
            )
            if const_expr(self.swap_AB):
                sW0_mma = sm90_ops.transpose_view(sW0_mma)
                sW1_mma = sm90_ops.transpose_view(sW1_mma)
            weights_slice = (None, 0) if const_expr(not self.swap_AB) else (0, None)
            tWeights0 = sm90_ops.make_acc_tensor_mn_view(thr_mma.partition_C(sW0_mma))[weights_slice]
            tWeights1 = sm90_ops.make_acc_tensor_mn_view(thr_mma.partition_C(sW1_mma))[weights_slice]

            q0_phase = Int32(0)
            q1_phase = Int32(0)
            kv0_phase = Int32(0)
            kv1_phase = Int32(0)
            warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

            if const_expr(self.compute_lse):
                lse_max0 = cute.make_rmem_tensor((self.q_tokens_per_stage,), Float32)
                lse_sum0 = cute.make_rmem_tensor((self.q_tokens_per_stage,), Float32)
                lse_max1 = cute.make_rmem_tensor((self.q_tokens_per_stage,), Float32)
                lse_sum1 = cute.make_rmem_tensor((self.q_tokens_per_stage,), Float32)
                lse_max0.fill(-Float32.inf)
                lse_sum0.fill(Float32(0.0))
                lse_max1.fill(-Float32.inf)
                lse_sum1.fill(Float32(0.0))

            cute.arch.mbarrier_wait(mbar_Q0_ptr, q0_phase)
            cute.arch.mbarrier_wait(mbar_Q1_ptr, q1_phase)

            iter_idx = Int32(0)
            while iter_idx < n_block_max:
                stage = iter_idx & Int32(1)
                n_block = self._iter_n_block(iter_idx, n_block_max)
                if stage == Int32(0):
                    cute.arch.mbarrier_wait(mbar_KV0_ptr, kv0_phase)
                    kv0_phase = kv0_phase ^ Int32(1)
                else:
                    cute.arch.mbarrier_wait(mbar_KV1_ptr, kv1_phase)
                    kv1_phase = kv1_phase ^ Int32(1)

                acc0 = cute.make_rmem_tensor(acc_shape, Float32)
                acc1 = cute.make_rmem_tensor(acc_shape, Float32)
                gemm_w_idx(tiled_mma_QK, acc0, tSrQ0, tSrK, zero_init=Boolean(True), B_idx=stage, wg_wait=-1, swap_AB=self.swap_AB)
                gemm_w_idx(tiled_mma_QK, acc1, tSrQ1, tSrK, zero_init=Boolean(True), B_idx=stage, wg_wait=-1, swap_AB=self.swap_AB)
                warpgroup.wait_group(0)

                if const_expr(self.use_fp8_scales):
                    acc_mn_for_scale = sm90_ops.make_acc_tensor_mn_view(acc0)
                    kNRows_for_scale = cute.size(acc_mn_for_scale, mode=[0])
                    lane = cute.arch.lane_idx()
                    t0 = lane % 4
                    t1 = lane // 4
                    m_base = t1 + warp_idx_in_wg * kNRows_for_scale * 8
                    rKScale = cute.make_rmem_tensor((kNRows_for_scale,), Float32)
                    if t0 == 0:
                        for mi in cutlass.range(kNRows_for_scale, unroll_full=True):
                            kv_m = m_base + mi * 8
                            rKScale[mi] = sKScale[stage * self.tile_n + kv_m]

                if stage == Int32(0):
                    if const_expr(self.use_fp8_scales):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mbar_KVEmpty0_ptr, arrive_count=32)
                    else:
                        if warp_idx_in_wg == 0:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(mbar_KVEmpty0_ptr, arrive_count=128)
                else:
                    if const_expr(self.use_fp8_scales):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mbar_KVEmpty1_ptr, arrive_count=32)
                    else:
                        if warp_idx_in_wg == 0:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(mbar_KVEmpty1_ptr, arrive_count=128)

                apply_causal_mask = Boolean(iter_idx < Int32(2)) if const_expr(self.mask_two_rightmost_nblocks) else Boolean(iter_idx == Int32(0))
                if const_expr(self.use_fp8_scales):
                    self._epilogue_store_to_gmem(
                        0,
                        acc0,
                        tWeights0,
                        rKScale,
                        mOut,
                        m_block,
                        n_block,
                        apply_causal_mask,
                        seqlen.seqlen_q,
                        seqlen.seqlen_k,
                        max_seqlen_k,
                        seqlen.offset_q,
                        batch_idx,
                        q_causal_offset,
                        sm_scale,
                        Int32(0),
                        lse_max0 if const_expr(self.compute_lse) else None,
                        lse_sum0 if const_expr(self.compute_lse) else None,
                    )
                    self._epilogue_store_to_gmem(
                        1,
                        acc1,
                        tWeights1,
                        rKScale,
                        mOut,
                        m_block,
                        n_block,
                        apply_causal_mask,
                        seqlen.seqlen_q,
                        seqlen.seqlen_k,
                        max_seqlen_k,
                        seqlen.offset_q,
                        batch_idx,
                        q_causal_offset,
                        sm_scale,
                        Int32(0),
                        lse_max1 if const_expr(self.compute_lse) else None,
                        lse_sum1 if const_expr(self.compute_lse) else None,
                    )
                else:
                    self._epilogue_store_to_gmem(
                        0,
                        acc0,
                        tWeights0,
                        None,
                        mOut,
                        m_block,
                        n_block,
                        apply_causal_mask,
                        seqlen.seqlen_q,
                        seqlen.seqlen_k,
                        max_seqlen_k,
                        seqlen.offset_q,
                        batch_idx,
                        q_causal_offset,
                        sm_scale,
                        Int32(0),
                        lse_max0 if const_expr(self.compute_lse) else None,
                        lse_sum0 if const_expr(self.compute_lse) else None,
                    )
                    self._epilogue_store_to_gmem(
                        1,
                        acc1,
                        tWeights1,
                        None,
                        mOut,
                        m_block,
                        n_block,
                        apply_causal_mask,
                        seqlen.seqlen_q,
                        seqlen.seqlen_k,
                        max_seqlen_k,
                        seqlen.offset_q,
                        batch_idx,
                        q_causal_offset,
                        sm_scale,
                        Int32(0),
                        lse_max1 if const_expr(self.compute_lse) else None,
                        lse_sum1 if const_expr(self.compute_lse) else None,
                    )
                iter_idx = iter_idx + Int32(1)

            if const_expr(self.compute_lse):
                self._write_lse_from_local_accum(
                    mLse,
                    sLseReduce,
                    lse_max0,
                    lse_sum0,
                    0,
                    0,
                    LSE_BARRIER_STAGE0,
                    m_block,
                    seqlen.seqlen_q,
                    seqlen.offset_q,
                    batch_idx,
                    wg_tidx,
                )
                self._write_lse_from_local_accum(
                    mLse,
                    sLseReduce,
                    lse_max1,
                    lse_sum1,
                    1,
                    # The BF16 consumer handles both Q stages serially, so the
                    # 32-float reduction scratch can be reused.
                    0,
                    LSE_BARRIER_STAGE0,
                    m_block,
                    seqlen.seqlen_q,
                    seqlen.offset_q,
                    batch_idx,
                    wg_tidx,
                )

    @cute.jit
    def consumer_split_qstage(
        self,
        q_stage_idx: cutlass.Constexpr[int],
        tiled_mma_QK: cute.TiledMma,
        sQ: cute.Tensor,
        sKV_staged: cute.Tensor,
        sW: cute.Tensor,
        sKScale: cute.Tensor,
        mOut: cute.Tensor,
        mLse: Optional[cute.Tensor],
        sLseReduce: cute.Tensor,
        mbar_Q_ptr,
        mbar_KV0_ptr,
        mbar_KV1_ptr,
        mbar_KVEmpty0_ptr,
        mbar_KVEmpty1_ptr,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mQCausalOffsets: cute.Tensor,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        sm_scale: Float32,
        work_block_x: Int32,
        work_head_idx: Int32,
        work_batch_idx: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        wg_tidx = tidx % self.num_threads_per_warp_group
        block_x = work_block_x
        head_idx = work_head_idx
        batch_idx = work_batch_idx

        seqlen = SeqlenInfoQK.create(
            batch_idx,
            max_seqlen_q,
            max_seqlen_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            None,
            None,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
        num_m_blocks = cute.ceil_div(seqlen.seqlen_q * self.qhead_per_kvhead, self.tile_m)
        if block_x < num_m_blocks:
            m_block = num_m_blocks - Int32(1) - block_x
            n_block_max = self._compute_n_blocks(m_block, seqlen.seqlen_q, seqlen.seqlen_k, q_causal_offset)

            thr_mma = tiled_mma_QK.get_slice(wg_tidx)
            tSrQ, tSrK = _mma_partition_fragment_AB(thr_mma, sQ, sKV_staged, self.swap_AB)
            acc_shape = tiled_mma_QK.partition_shape_C((self.tile_n, self.q_per_stage))

            sW_mma = cute.make_tensor(
                sW.iterator + q_stage_idx * self.q_per_stage,
                cute.make_layout((self.q_per_stage, self.tile_n), stride=(1, 0)),
            )
            if const_expr(self.swap_AB):
                sW_mma = sm90_ops.transpose_view(sW_mma)
            weights_slice = (None, 0) if const_expr(not self.swap_AB) else (0, None)
            tWeights = sm90_ops.make_acc_tensor_mn_view(thr_mma.partition_C(sW_mma))[weights_slice]

            q_phase = Int32(0)
            kv0_phase = Int32(0)
            kv1_phase = Int32(0)
            warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

            if const_expr(self.compute_lse):
                lse_max = cute.make_rmem_tensor((self.q_tokens_per_stage,), Float32)
                lse_sum = cute.make_rmem_tensor((self.q_tokens_per_stage,), Float32)
                lse_max.fill(-Float32.inf)
                lse_sum.fill(Float32(0.0))

            cute.arch.mbarrier_wait(mbar_Q_ptr, q_phase)

            iter_idx = Int32(0)
            while iter_idx < n_block_max:
                stage = iter_idx & Int32(1)
                n_block = self._iter_n_block(iter_idx, n_block_max)
                if stage == Int32(0):
                    cute.arch.mbarrier_wait(mbar_KV0_ptr, kv0_phase)
                    kv0_phase = kv0_phase ^ Int32(1)
                else:
                    cute.arch.mbarrier_wait(mbar_KV1_ptr, kv1_phase)
                    kv1_phase = kv1_phase ^ Int32(1)

                acc = cute.make_rmem_tensor(acc_shape, Float32)
                gemm_w_idx(
                    tiled_mma_QK,
                    acc,
                    tSrQ,
                    tSrK,
                    zero_init=Boolean(True),
                    B_idx=stage,
                    wg_wait=-1,
                    swap_AB=self.swap_AB,
                )
                warpgroup.wait_group(0)

                acc_mn_for_scale = sm90_ops.make_acc_tensor_mn_view(acc)
                kNRows_for_scale = cute.size(acc_mn_for_scale, mode=[0])
                lane = cute.arch.lane_idx()
                t0 = lane % 4
                t1 = lane // 4
                m_base = t1 + warp_idx_in_wg * kNRows_for_scale * 8
                rKScale = cute.make_rmem_tensor((kNRows_for_scale,), Float32)
                if const_expr(self.use_fast_qh64_epilogue):
                    mi_dyn = t0 % 2
                    kv_m_pair_scale = m_base + mi_dyn * 8
                    rKScale[0] = sKScale[stage * self.tile_n + kv_m_pair_scale]
                elif t0 == 0:
                    for mi in cutlass.range(kNRows_for_scale, unroll_full=True):
                        kv_m_lane_scale = m_base + mi * 8
                        rKScale[mi] = sKScale[stage * self.tile_n + kv_m_lane_scale]

                if stage == Int32(0):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar_KVEmpty0_ptr, arrive_count=32)
                else:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar_KVEmpty1_ptr, arrive_count=32)

                if const_expr(self.use_fast_qh64_epilogue):
                    if iter_idx == Int32(0):
                        self._epilogue_store_to_gmem_qh64_masked(
                            q_stage_idx,
                            acc,
                            tWeights,
                            rKScale,
                            mOut,
                            m_block,
                            n_block,
                            seqlen.seqlen_q,
                            seqlen.seqlen_k,
                            max_seqlen_k,
                            seqlen.offset_q,
                            batch_idx,
                            q_causal_offset,
                            lse_max if const_expr(self.compute_lse) else None,
                            lse_sum if const_expr(self.compute_lse) else None,
                        )
                    else:
                        self._epilogue_store_to_gmem_qh64_full(
                            q_stage_idx,
                            acc,
                            tWeights,
                            rKScale,
                            mOut,
                            m_block,
                            n_block,
                            seqlen.seqlen_q,
                            max_seqlen_k,
                            seqlen.offset_q,
                            batch_idx,
                            lse_max if const_expr(self.compute_lse) else None,
                            lse_sum if const_expr(self.compute_lse) else None,
                        )
                else:
                    self._epilogue_store_to_gmem(
                        q_stage_idx,
                        acc,
                        tWeights,
                        rKScale,
                        mOut,
                        m_block,
                        n_block,
                        Boolean(iter_idx == Int32(0)),
                        seqlen.seqlen_q,
                        seqlen.seqlen_k,
                        max_seqlen_k,
                        seqlen.offset_q,
                        batch_idx,
                        q_causal_offset,
                        sm_scale,
                        Int32(0),
                        lse_max if const_expr(self.compute_lse) else None,
                        lse_sum if const_expr(self.compute_lse) else None,
                    )
                iter_idx = iter_idx + Int32(1)

            if const_expr(self.compute_lse):
                self._write_lse_from_local_accum(
                    mLse,
                    sLseReduce,
                    lse_max,
                    lse_sum,
                    q_stage_idx,
                    0 if const_expr(q_stage_idx == 0) else 16,
                    LSE_BARRIER_STAGE0 if const_expr(q_stage_idx == 0) else LSE_BARRIER_STAGE1,
                    m_block,
                    seqlen.seqlen_q,
                    seqlen.offset_q,
                    batch_idx,
                    wg_tidx,
                )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mW: cute.Tensor,
        mQScale: Optional[cute.Tensor],
        mKScale: Optional[cute.Tensor],
        mOut: cute.Tensor,
        mLse: Optional[cute.Tensor],
        n_heads_kv: Int32,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        sm_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mQCausalOffsets: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        is_varlen = mCuSeqlensQ is not None
        if const_expr(is_varlen):
            assert self.is_varlen
        else:
            assert not self.is_varlen
        assert mQ.element_type == self.qk_dtype
        assert mK.element_type == self.qk_dtype
        assert mW.element_type == self.w_dtype
        if const_expr(self.use_fp8_scales):
            assert mQScale is not None and mKScale is not None
            assert mQScale.element_type == Float32
            assert mKScale.element_type == Float32
        else:
            assert mQScale is None and mKScale is None
        if const_expr(self.compute_lse):
            assert mLse is not None
            assert mLse.element_type == Float32
        else:
            assert mLse is None

        def _assume_strides(t):
            divby = 128 // t.element_type.width
            new_strides = []
            for s in t.stride[:-1]:
                if const_expr(isinstance(s, int)):
                    new_strides.append(s)
                else:
                    new_strides.append(cute.assume(s, divby=divby))
            new_strides.append(t.stride[-1])
            return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=tuple(new_strides)))

        mQ = _assume_strides(mQ)
        mK = _assume_strides(mK)
        mW = _assume_strides(mW)
        mOut = _assume_strides(mOut)
        if const_expr(self.use_fp8_scales):
            mQScale = _assume_strides(mQScale)
            mKScale = _assume_strides(mKScale)

        mQ = sm90_ops.select(mQ, [0, 2, 1] if const_expr(is_varlen) else [1, 3, 2, 0])
        mK = sm90_ops.select(mK, [0, 2, 1] if const_expr(is_varlen) else [1, 3, 2, 0])
        mW = sm90_ops.select(mW, [0, 1] if const_expr(is_varlen) else [1, 2, 0])
        if const_expr(self.use_fp8_scales):
            mQScale = sm90_ops.select(mQScale, [0, 1] if const_expr(is_varlen) else [1, 2, 0])
            mKScale = sm90_ops.select(mKScale, [0, 1] if const_expr(is_varlen) else [1, 2, 0])

        qhpkv = self.qhead_per_kvhead
        num_head_kv = n_heads_kv
        mQ = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                ((qhpkv, mQ.shape[0]), mQ.shape[1], num_head_kv, *mQ.shape[3:]),
                stride=((mQ.stride[2], mQ.stride[0]), mQ.stride[1], mQ.stride[2] * qhpkv, *mQ.stride[3:]),
            ),
        )
        mW = cute.make_tensor(
            mW.iterator,
            cute.make_layout(
                ((qhpkv, mW.shape[0]), num_head_kv, *mW.shape[2:]),
                stride=((mW.stride[1], mW.stride[0]), mW.stride[1] * qhpkv, *mW.stride[2:]),
            ),
        )
        if const_expr(self.use_fp8_scales):
            mQScale = cute.make_tensor(
                mQScale.iterator,
                cute.make_layout(
                    ((qhpkv, mQScale.shape[0]), num_head_kv, *mQScale.shape[2:]),
                    stride=(
                        (mQScale.stride[1], mQScale.stride[0]),
                        mQScale.stride[1] * qhpkv,
                        *mQScale.stride[2:],
                    ),
                ),
            )

        self._setup_attributes()
        tiled_mma_QK = self._get_tiled_mma()
        SharedStorage = self._get_shared_storage_cls()

        self.tma_copy_bytes_Q = cute.size_in_bytes(self.qk_dtype, cute.select(self.sQ_layout_single, mode=[0, 1]))
        self.tma_copy_bytes_K = cute.size_in_bytes(self.qk_dtype, cute.select(self.sKV_layout_single, mode=[0, 1]))

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mQ, self.sQ_layout_single, (self.q_per_stage, self.tile_hdim))
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mK, self.sKV_layout_single, (self.mma_tile_n, self.tile_hdim))

        batch_size = cute.size(mCuSeqlensQ.shape[0]) - 1 if const_expr(is_varlen) else cute.size(mQ.shape[3])
        grid_x = cute.ceil_div(max_seqlen_q * self.qhead_per_kvhead, self.tile_m)
        grid = (grid_x, num_head_kv, batch_size)

        self.kernel(
            tma_tensor_Q,
            tma_atom_Q,
            tma_tensor_K,
            tma_atom_K,
            mW,
            mQScale,
            mKScale,
            mOut,
            mLse,
            self.sQ_layout_staged,
            self.sKV_layout_staged,
            tiled_mma_QK,
            SharedStorage,
            mCuSeqlensQ,
            mCuSeqlensK,
            mQCausalOffsets,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        mK: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        mW: cute.Tensor,
        mQScale: Optional[cute.Tensor],
        mKScale: Optional[cute.Tensor],
        mOut: cute.Tensor,
        mLse: Optional[cute.Tensor],
        sQ_layout_staged: cute.ComposedLayout,
        sKV_layout_staged: cute.ComposedLayout,
        tiled_mma_QK: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mQCausalOffsets: cute.Tensor,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        sm_scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)

        mbar_Q0_ptr = storage.mbar_Q0.data_ptr()
        mbar_Q1_ptr = storage.mbar_Q1.data_ptr()
        mbar_KV0_ptr = storage.mbar_KV0.data_ptr()
        mbar_KV1_ptr = storage.mbar_KV1.data_ptr()
        mbar_KVEmpty0_ptr = storage.mbar_KVEmpty0.data_ptr()
        mbar_KVEmpty1_ptr = storage.mbar_KVEmpty1.data_ptr()

        sQ_staged = storage.sQ.get_tensor(sQ_layout_staged.outer, swizzle=sQ_layout_staged.inner)
        sKV_staged = storage.sKV.get_tensor(sKV_layout_staged.outer, swizzle=sKV_layout_staged.inner)
        sQ_0 = sQ_staged[None, None, 0]
        sQ_1 = sQ_staged[None, None, 1]
        sKV_0 = sKV_staged[None, None, 0]
        sKV_1 = sKV_staged[None, None, 1]
        sW = storage.sW.get_tensor(cute.make_layout((self.tile_m,), stride=(1,)))
        sKScale = storage.sKScale.get_tensor(cute.make_layout((self.tile_n * self.kv_stages,), stride=(1,)))
        sLseReduce = storage.sLseReduce.get_tensor(cute.make_layout((32,), stride=(1,)))

        block_x, head_idx, batch_idx = cute.arch.block_idx()
        if warp_idx == 0:
            cute.arch.mbarrier_init(mbar_Q0_ptr, 1)
            cute.arch.mbarrier_init(mbar_Q1_ptr, 1)
            if const_expr(self.use_fp8_scales):
                cute.arch.mbarrier_init(mbar_KV0_ptr, 2)
                cute.arch.mbarrier_init(mbar_KV1_ptr, 2)
            else:
                cute.arch.mbarrier_init(mbar_KV0_ptr, 1)
                cute.arch.mbarrier_init(mbar_KV1_ptr, 1)
            if const_expr(self.use_split_q_wg):
                cute.arch.mbarrier_init(mbar_KVEmpty0_ptr, 256)
                cute.arch.mbarrier_init(mbar_KVEmpty1_ptr, 256)
            else:
                cute.arch.mbarrier_init(mbar_KVEmpty0_ptr, 128)
                cute.arch.mbarrier_init(mbar_KVEmpty1_ptr, 128)
        cute.arch.sync_threads()

        if warp_group_idx == 0:
            if warp_idx == 0:
                self.producer(
                    mQ,
                    tma_atom_Q,
                    mK,
                    tma_atom_K,
                    mW,
                    mQScale,
                    mKScale,
                    sQ_0,
                    sQ_1,
                    sKV_0,
                    sKV_1,
                    sW,
                    sKScale,
                    mbar_Q0_ptr,
                    mbar_Q1_ptr,
                    mbar_KV0_ptr,
                    mbar_KV1_ptr,
                    mbar_KVEmpty0_ptr,
                    mbar_KVEmpty1_ptr,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mQCausalOffsets,
                    max_seqlen_q,
                    max_seqlen_k,
                    sm_scale,
                    block_x,
                    head_idx,
                    batch_idx,
                )
            elif const_expr(self.use_fp8_scales):
                if warp_idx == 2:
                    self.producer(
                        mQ,
                        tma_atom_Q,
                        mK,
                        tma_atom_K,
                        mW,
                        mQScale,
                        mKScale,
                        sQ_0,
                        sQ_1,
                        sKV_0,
                        sKV_1,
                        sW,
                        sKScale,
                        mbar_Q0_ptr,
                        mbar_Q1_ptr,
                        mbar_KV0_ptr,
                        mbar_KV1_ptr,
                        mbar_KVEmpty0_ptr,
                        mbar_KVEmpty1_ptr,
                        mCuSeqlensQ,
                        mCuSeqlensK,
                        mQCausalOffsets,
                        max_seqlen_q,
                        max_seqlen_k,
                        sm_scale,
                        block_x,
                        head_idx,
                        batch_idx,
                    )
        else:
            if const_expr(self.use_split_q_wg):
                if warp_group_idx == 1:
                    self.consumer_split_qstage(
                        0,
                        tiled_mma_QK,
                        sQ_0,
                        sKV_staged,
                        sW,
                        sKScale,
                        mOut,
                        mLse,
                        sLseReduce,
                        mbar_Q0_ptr,
                        mbar_KV0_ptr,
                        mbar_KV1_ptr,
                        mbar_KVEmpty0_ptr,
                        mbar_KVEmpty1_ptr,
                        mCuSeqlensQ,
                        mCuSeqlensK,
                        mQCausalOffsets,
                        max_seqlen_q,
                        max_seqlen_k,
                        sm_scale,
                        block_x,
                        head_idx,
                        batch_idx,
                    )
                elif warp_group_idx == 2:
                    self.consumer_split_qstage(
                        1,
                        tiled_mma_QK,
                        sQ_1,
                        sKV_staged,
                        sW,
                        sKScale,
                        mOut,
                        mLse,
                        sLseReduce,
                        mbar_Q1_ptr,
                        mbar_KV0_ptr,
                        mbar_KV1_ptr,
                        mbar_KVEmpty0_ptr,
                        mbar_KVEmpty1_ptr,
                        mCuSeqlensQ,
                        mCuSeqlensK,
                        mQCausalOffsets,
                        max_seqlen_q,
                        max_seqlen_k,
                        sm_scale,
                        block_x,
                        head_idx,
                        batch_idx,
                    )
            else:
                self.consumer(
                    tiled_mma_QK,
                    sQ_0,
                    sQ_1,
                    sKV_staged,
                    sW,
                    sKScale,
                    mOut,
                    mLse,
                    sLseReduce,
                    mbar_Q0_ptr,
                    mbar_Q1_ptr,
                    mbar_KV0_ptr,
                    mbar_KV1_ptr,
                    mbar_KVEmpty0_ptr,
                    mbar_KVEmpty1_ptr,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mQCausalOffsets,
                    max_seqlen_q,
                    max_seqlen_k,
                    sm_scale,
                    block_x,
                    head_idx,
                    batch_idx,
                )
