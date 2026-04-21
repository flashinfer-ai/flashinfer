# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLAComputeRole — compute (softmax) warp role for MLA decode kernels.

Handles:
- Online softmax with row-max tracking, exp2, correction factor
- SM100 vs SM103 architecture dispatch for TMEM load (plain vs fused-reduce)
- SMEM exchange for row-max reduction across warps
- P quantization and SMEM store to feed the PV MMA stage
- Row-sum accumulation using packed f32x2 adds
- Tile scheduler loop with mask applied on the last tile
- Correction metadata exchange to TMEM for correction warps

Uses handle-passing pattern: state transitions (acquire_and_advance,
wait_and_advance) stay in run(), while ImmutableResourceHandles are passed
to sub-methods where side effects (commit, release) are co-located with
their paired fences.

Extracted from MLADecodeFP16Kernel.compute / softmax / exchange_p_cor_metadata.
"""

import math
from typing import Type

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass.pipeline import PipelineProducer, PipelineConsumer
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL
from types import SimpleNamespace

from ..mla_config import MLAConfig
from ..config import AttentionFusion
from ..scheduler.mla_persistent import (
    create_mla_static_tile_scheduler,
    MLAStaticTileSchedulerParams,
)

from ..compat import setmaxregister_increase as _setmaxregister_increase


class MLAComputeRole:
    """Compute (softmax) warp role for MLA decode kernels.

    Owns the tile-scheduler loop and performs online softmax, producing
    quantized P tiles in SMEM and correction metadata in TMEM.

    Optionally integrates AttentionVariant hooks (score_mod, update_statistics)
    via the fusion parameter, mirroring the prefill SoftmaxRole pattern.
    """

    def __init__(self, config: MLAConfig, fusion: AttentionFusion):
        self.acc_dtype = config.acc_dtype
        self.mma_qk_tiler = config.mma_qk_tiler
        self.mma_pv_tiler = config.mma_pv_tiler
        self.cluster_shape_mnk = config.cluster_shape_mnk
        self.warps_in_n = config.warps_in_n
        self.num_compute_warps = config.num_compute_warps
        self.threads_per_warp = 32
        self.mma_s_stage = config.mma_s_stage
        self.p_mma_stage = config.p_mma_stage
        self.p_cor_stage = config.p_cor_stage
        self.skip_correction_threshold = config.skip_correction_threshold
        self.tmem_o_offset = config.tmem_o_offset
        self.correction_factor_offset = config.correction_factor_offset
        self.is_var_split_kv = config.is_var_split_kv

        self.variant = fusion.variant
        self.has_score_mod = fusion.variant.has_score_mod
        self.has_statistics_update = fusion.variant.has_statistics_update
        self.has_params = fusion.has_params

        self.softmax_reg_num = 192
        self.softmax_exchange_sync_bar = None

    def set_dtypes(self, q_dtype: Type[cutlass.Numeric]) -> None:
        """Set tensor element types discovered at call time."""
        self.q_dtype: Type[cutlass.Numeric] = q_dtype

    def set_barriers(self, softmax_exchange_sync_bar):
        """Set named barriers owned by the kernel."""
        self.softmax_exchange_sync_bar = softmax_exchange_sync_bar

    # ------------------------------------------------------------------
    # Tile count helper
    # ------------------------------------------------------------------

    @cute.jit
    def _get_k_tile_count(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        blk_coord: cute.Coord,
    ) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        """Get k_index, k_tile_count, and local split_kv for a work tile."""
        K = cache_seqs[blk_coord[2]]
        if cutlass.const_expr(self.is_var_split_kv):
            split_kv = block_split_kvs[blk_coord[2]]

        k_tile_total = cute.ceil_div(K, self.mma_qk_tiler[1])
        k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
        k_index = blk_coord[3] * k_tile_per_cta
        k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)
        return k_index, k_tile_count, split_kv

    # ------------------------------------------------------------------
    # Correction metadata exchange
    # ------------------------------------------------------------------

    @cute.jit
    def exchange_p_cor_metadata(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        correction_factor: cutlass.Float32,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
        row_max_new: cutlass.Float32,
        tAcc: cute.Tensor,
        tidx: cutlass.Int32,
        p_cor_handle,
    ):
        """Write correction metadata to TMEM for the correction warps.

        Commits the p_cor handle after fence_view_async_tmem_store,
        co-locating the fence+commit pair. Returns the updated row_max_new.
        """
        no_correction = 0
        if (
            row_max_new - row_max
        ) * softmax_params.softmax_scale_log2 <= self.skip_correction_threshold:
            no_correction = 1
            row_max_new = row_max

        corr_layout = cute.make_layout(
            (tAcc.shape[0], (4, tAcc.shape[1][1]), self.mma_s_stage),
            stride=(tAcc.stride[0], (1, tAcc.stride[1][1]), 4),
        )
        tCor = cute.make_tensor(
            common_params.tmem_ptr + self.correction_factor_offset,
            corr_layout,
        )
        cCor = cute.make_identity_tensor(tCor.shape)
        corr_tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(4)), self.acc_dtype
        )
        corr_tmem_store_tiled_copy = tcgen05.make_tmem_copy(corr_tmem_store_atom, tCor)
        corr_tmem_store_thr_copy = corr_tmem_store_tiled_copy.get_slice(tidx)
        cCor_for_copy = corr_tmem_store_thr_copy.partition_S(cCor)
        tCor_for_copy = corr_tmem_store_thr_copy.partition_D(tCor)
        rCor = cute.make_fragment_like(
            cCor_for_copy[None, None, None, 0], self.acc_dtype
        )
        rCor_int = cute.make_tensor(
            cute.recast_ptr(rCor.iterator, dtype=cutlass.Int32), rCor.layout
        )
        rCor[0] = row_sum
        rCor[1] = row_max_new
        rCor[2] = correction_factor
        rCor_int[3] = no_correction

        cute.copy(
            corr_tmem_store_tiled_copy,
            rCor,
            tCor_for_copy[None, None, None, p_cor_handle.index],
        )
        cute.arch.fence_view_async_tmem_store()
        p_cor_handle.commit()
        return row_max_new

    # ------------------------------------------------------------------
    # Softmax (single tile)
    # ------------------------------------------------------------------

    @cute.jit
    def softmax(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        mma_s_handle,
        p_mma_handle,
        p_cor_handle,
        row_max: cutlass.Float32,
        row_sum: cutlass.Float32,
        correction_factor: cutlass.Float32,
        is_last_tile: bool,
        is_local_last_tile: cutlass.Boolean,
        params: cute.Tensor = None,
    ) -> tuple:
        """Online softmax for one k-tile.

        Contains the SM100 vs SM103 architecture dispatch for TMEM load,
        masking, exp2, row-max reduction with SMEM exchange, P quantization
        and SMEM store, row-sum accumulation.

        When an AttentionVariant is configured:
        - update_statistics runs before TMEM load (e.g. attention sink)
        - score_mod runs after TMEM load + masking, before row_max (e.g. ALiBi)

        Side effects co-located with their paired fences:
        - fence_view_async_shared → p_mma_handle.commit()
        - fence_view_async_tmem_store → p_cor_handle.commit() (via exchange)
        - fence_view_async_tmem_load → mma_s_handle.release()

        Returns (row_max_new, row_sum, correction_factor).
        """

        # load S from tmem
        tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged_fake = softmax_params.tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)
        tStS = tStS_staged[None, None, None, mma_s_handle.index]

        tAcc = tStS[(None, None), 0, 0]
        cta_qk_tiler = (
            self.mma_qk_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_qk_tiler[1],
            self.mma_qk_tiler[2],
        )
        cS = cute.make_identity_tensor(cute.select(cta_qk_tiler, mode=[0, 1]))

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)

        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc = tmem_thr_copy.partition_S(tAcc)
        tTR_tS = tmem_thr_copy.partition_D(cS)

        # Inject virtual tokens into running (m, d) before loading scores
        if cutlass.const_expr(self.has_statistics_update):
            if cutlass.const_expr(self.has_params):
                self.variant.params = params
            qo_head_idx_for_stats = tTR_tS[0][0] + common_params.cta_m_offset
            row_max, row_sum = self.variant.update_statistics(
                k_index,
                qo_head_idx_for_stats,
                row_max,
                row_sum,
                softmax_params.softmax_scale_log2,
            )

        tTR_rAcc = cute.make_fragment_like(tTR_tS, self.acc_dtype)

        row_max_new = row_max
        arch = BaseDSL._get_dsl().get_arch_enum()
        if cutlass.const_expr(arch >= Arch.sm_100 and arch <= Arch.sm_100f):
            cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)
            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                if is_last_tile:
                    tTR_rAcc[i] = (
                        tTR_rAcc[i]
                        if cute.elem_less(
                            tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index,
                            common_params.K,
                        )
                        else -self.acc_dtype.inf
                    )
            row_max_new = tTR_rAcc.load().reduce(cute.ReductionOp.MAX, row_max_new, 0)

        elif cutlass.const_expr(arch >= Arch.sm_103 and arch <= Arch.sm_103f):
            tmem_load_red_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition(64), redOp=tcgen05.TmemLoadRedOp.MAX
                ),
                self.acc_dtype,
            )
            tmem_red_tiled_copy = tcgen05.make_tmem_copy(tmem_load_red_atom, tAcc)
            tmem_red_thr_copy = tmem_red_tiled_copy.get_slice(tidx)
            tTR_tAcc_red = tmem_red_thr_copy.partition_S(tAcc)
            tTR_tS_red = tmem_red_thr_copy.partition_D(cS)
            tTR_rAcc_red = cute.make_fragment_like(tTR_tS_red, self.acc_dtype)
            tTR_rMax = cute.make_rmem_tensor(
                cute.make_layout((1, tTR_tS_red.shape[1], tTR_tS_red.shape[2])),
                self.acc_dtype,
            )
            cute.copy(
                tmem_red_tiled_copy,
                tTR_tAcc_red,
                (tTR_rAcc_red, tTR_rMax),
            )
            tTR_rAcc = cute.make_tensor(tTR_rAcc_red.iterator, tTR_rAcc.layout)
            if is_last_tile:
                for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                    tTR_rAcc[i] = (
                        tTR_rAcc[i]
                        if cute.elem_less(
                            tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index,
                            common_params.K,
                        )
                        else -self.acc_dtype.inf
                    )
                row_max_new = tTR_rAcc.load().reduce(
                    cute.ReductionOp.MAX, row_max_new, 0
                )
            else:
                row_max_new = cute.arch.fmax(row_max_new, tTR_rMax[0])

        # Per-element score modification (e.g. ALiBi bias, soft-capping).
        # Applied after masking, before row_max finalization.  When active,
        # row_max must be recomputed from the modified scores.
        if cutlass.const_expr(self.has_score_mod):
            if cutlass.const_expr(self.has_params and not self.has_statistics_update):
                self.variant.params = params
            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                qo_head_idx = tTR_tS[i][0] + common_params.cta_m_offset
                kv_idx = tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index
                tTR_rAcc[i] = self.variant.score_mod(
                    tTR_rAcc[i],
                    common_params.batch_idx,
                    common_params.qo_idx,
                    kv_idx,
                    qo_head_idx,
                    0,
                )
            # Re-apply masking: score_mod may map -inf to a finite value
            # (e.g. SoftCapping: cap*tanh(-inf/cap) = -cap).  Restore -inf
            # for out-of-bounds positions so they get zero softmax weight.
            if is_last_tile:
                for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                    if not cute.elem_less(
                        tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index,
                        common_params.K,
                    ):
                        tTR_rAcc[i] = -self.acc_dtype.inf
            row_max_new = row_max
            row_max_new = tTR_rAcc.load().reduce(cute.ReductionOp.MAX, row_max_new, 0)

        # reduce row_max across warps via SMEM exchange when warps_in_n == 2
        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[tidx] = row_max_new
            assert self.softmax_exchange_sync_bar is not None
            self.softmax_exchange_sync_bar.arrive_and_wait()
            row_max_new = cute.arch.fmax(
                row_max_new,
                common_params.smem_exchange[
                    (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ],
            )

        # correction factor
        correction_factor = cute.math.exp2(
            (row_max - row_max_new) * softmax_params.softmax_scale_log2, fastmath=True
        )
        # split kv case: exchange metadata before last tile
        if cutlass.const_expr(not is_local_last_tile):
            row_max_new = self.exchange_p_cor_metadata(
                common_params,
                softmax_params,
                correction_factor,
                row_sum,
                row_max,
                row_max_new,
                tAcc,
                tidx,
                p_cor_handle,
            )

        # exp2 + quantize
        fma_b = softmax_params.softmax_scale_log2
        fma_c = (0.0 - row_max_new) * softmax_params.softmax_scale_log2

        for i in cutlass.range(cute.size(tTR_rAcc), vectorize=True, unroll_full=True):
            tTR_rAcc[i] = tTR_rAcc[i] * fma_b + fma_c
            tTR_rAcc[i] = cute.math.exp2(tTR_rAcc[i], fastmath=True)

        tTR_rS = cute.make_fragment_like(tTR_tS, self.q_dtype)

        tTR_rS.store(tTR_rAcc.load().to(self.q_dtype))

        # store P to SMEM
        sP = softmax_params.sP[None, None, None, (None, p_mma_handle.index)]
        sP_mk_view = cute.make_tensor(
            sP.iterator,
            cute.make_layout(
                (
                    (sP.shape[0][0], sP.shape[1]),
                    (sP.shape[0][1], sP.shape[2], sP.shape[3]),
                ),
                stride=(
                    (sP.stride[0][0], sP.stride[1]),
                    (sP.stride[0][1], sP.stride[2], sP.stride[3]),
                ),
            ),
        )
        sP_wo_swizzle_iter = cute.recast_ptr(sP.iterator, swizzle_=None)
        swizzle_bits = (
            int(math.log2(self.mma_pv_tiler[2] * self.q_dtype.width // 8 // 32)) + 1
        )
        swizzle_base = 3 if self.q_dtype.width == 16 else 4
        sP_swizzle = cute.make_swizzle(swizzle_bits, swizzle_base, 3)
        sP_mk_view = cute.make_tensor(
            sP_wo_swizzle_iter,
            cute.make_composed_layout(sP_swizzle, 0, sP_mk_view.layout),
        )
        universal_copy_bits = 128
        smem_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.q_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        smem_tiled_copy = cute.make_tiled_copy_D(smem_copy_atom, tmem_tiled_copy)
        smem_thr_copy = smem_tiled_copy.get_slice(tidx)
        rP_copy_view = smem_thr_copy.retile(tTR_rS)
        sP_copy_view = smem_thr_copy.partition_D(sP_mk_view)
        cute.copy(smem_tiled_copy, rP_copy_view, sP_copy_view)

        cute.arch.fence_view_async_shared()
        p_mma_handle.commit()

        # row_sum accumulation using packed f32x2 to reduce instruction count
        row_sum = row_sum * correction_factor
        row_sum_vec = (0.0, 0.0)
        for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
            row_sum_vec = cute.arch.add_packed_f32x2(
                row_sum_vec, (tTR_rAcc[i], tTR_rAcc[i + 1])
            )
        row_sum = row_sum_vec[0] + row_sum_vec[1] + row_sum

        # split kv case: exchange metadata on last tile
        if cutlass.const_expr(is_local_last_tile):
            row_max_new = self.exchange_p_cor_metadata(
                common_params,
                softmax_params,
                correction_factor,
                row_sum,
                row_max,
                row_max_new,
                tAcc,
                tidx,
                p_cor_handle,
            )

        cute.arch.fence_view_async_tmem_load()
        mma_s_handle.release()

        return (
            row_max_new,
            row_sum,
            correction_factor,
        )

    # ------------------------------------------------------------------
    # run — top-level entry: pipeline init + tile scheduler loop
    #
    # State transitions (acquire_and_advance, wait_and_advance) live here.
    # ImmutableResourceHandles are passed to sub-methods where side
    # effects (commit, release) are co-located with their paired fences.
    # ------------------------------------------------------------------

    @cute.jit
    def run(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        tile_sched_params: MLAStaticTileSchedulerParams,
        tmem_ptr: cute.Pointer,
        mma_s_consumer: PipelineConsumer,
        p_mma_producer: PipelineProducer,
        p_cor_producer: PipelineProducer,
        softmax_smem_exchange: cute.Tensor,
        mAccO: cute.Tensor,
        mO: cute.Tensor,
        mCL: cute.Tensor,
        K: cutlass.Int32,
        L: cutlass.Int32,
        tiled_mma_qk: cute.TiledMma,
        sP: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        tmem,
        params: cute.Tensor = None,
    ):
        """Top-level entry for the compute warp role.

        Iterates the tile scheduler and runs softmax for each valid work
        tile. State transitions (acquire/wait) happen here; commit/release
        are co-located with fences inside softmax/exchange_p_cor_metadata.
        """
        _setmaxregister_increase(self.softmax_reg_num)

        tmem.wait_for_alloc()
        tmem_ptr_resolved = tmem.retrieve_ptr(self.acc_dtype)

        tidx, _, _ = cute.arch.thread_idx()

        tile_sched = create_mla_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        while work_tile.is_valid_tile:
            blk_coord = work_tile.tile_idx
            k_index, k_tile_count, local_split_kv = self._get_k_tile_count(
                split_kv, cache_seqs, block_split_kvs, blk_coord
            )
            if k_tile_count > 0:
                bidx, _, _ = cute.arch.block_idx()
                cta_m_offset = (bidx % self.cluster_shape_mnk[0]) * (
                    self.mma_qk_tiler[0] // self.cluster_shape_mnk[0]
                )
                compute_common_params = SimpleNamespace(
                    blk_coord=blk_coord,
                    split_kv=split_kv,
                    local_split_kv=local_split_kv,
                    smem_exchange=softmax_smem_exchange,
                    mAccO=mAccO,
                    mO=mO,
                    K=cache_seqs[blk_coord[2]],
                    L=L,
                    tmem_ptr=tmem_ptr_resolved,
                    tidx=tidx,
                    batch_idx=blk_coord[2],
                    qo_idx=blk_coord[1],
                    cta_m_offset=cta_m_offset,
                )
                compute_softmax_params = SimpleNamespace(
                    tiled_mma_qk=tiled_mma_qk,
                    sP=sP,
                    softmax_scale_log2=softmax_scale_log2,
                )
                k_tile_total = cute.ceil_div(
                    compute_common_params.K, self.mma_qk_tiler[1]
                )

                row_max = -self.acc_dtype.inf
                row_sum = self.acc_dtype(0)
                correction_factor = self.acc_dtype(1)
                p_cor_handle = p_cor_producer.acquire_and_advance()

                # unmasked tiles
                while k_tile_count > 1:
                    p_mma_handle = p_mma_producer.acquire_and_advance()
                    mma_s_handle = mma_s_consumer.wait_and_advance()

                    (
                        row_max,
                        row_sum,
                        correction_factor,
                    ) = self.softmax(
                        compute_common_params,
                        compute_softmax_params,
                        k_index,
                        mma_s_handle,
                        p_mma_handle,
                        p_cor_handle,
                        row_max,
                        row_sum,
                        correction_factor,
                        False,
                        False,
                        params,
                    )

                    p_cor_handle = p_cor_producer.acquire_and_advance()

                    k_index = k_index + 1
                    k_tile_count = k_tile_count - 1

                # last tile (masked)
                p_mma_handle = p_mma_producer.acquire_and_advance()
                mma_s_handle = mma_s_consumer.wait_and_advance()

                if cutlass.const_expr(mAccO is not None):
                    (
                        row_max,
                        row_sum,
                        correction_factor,
                    ) = self.softmax(
                        compute_common_params,
                        compute_softmax_params,
                        k_index,
                        mma_s_handle,
                        p_mma_handle,
                        p_cor_handle,
                        row_max,
                        row_sum,
                        correction_factor,
                        k_index == k_tile_total - 1,
                        True,
                        params,
                    )
                else:
                    (
                        row_max,
                        row_sum,
                        correction_factor,
                    ) = self.softmax(
                        compute_common_params,
                        compute_softmax_params,
                        k_index,
                        mma_s_handle,
                        p_mma_handle,
                        p_cor_handle,
                        row_max,
                        row_sum,
                        correction_factor,
                        True,
                        True,
                        params,
                    )

                # Trailing sync: acquire() without advance — back-pressure only,
                # no data produced. acquire_and_advance() would desync the pipeline
                # across persistent kernel work tiles.
                p_cor_producer.acquire()

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        p_cor_producer.tail()
