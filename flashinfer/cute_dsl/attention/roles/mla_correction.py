# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLACorrectionRole — rescale + epilogue warp role for MLA decode.

Owns the tile-scheduler loop for the correction warp, performing:
- Loading correction metadata (row_sum, row_max, correction_factor) from TMEM
- Rescaling partial O accumulator when row-max changes across KV tiles
- Final normalization, dtype conversion, and global memory write (O and LSE)
- Split-KV workspace output vs direct output path selection

Uses handle-passing pattern: state transitions (wait_and_advance) stay in
run(), while ImmutableResourceHandles are passed to sub-methods where side
effects (release) are co-located with their paired fences.

Extracted from the monolithic MLA decode kernel's correction warp section.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.pipeline import PipelineConsumer
from types import SimpleNamespace

from ..mla_config import MLAConfig
from ..config import AttentionFusion
from ..scheduler.mla_persistent import (
    create_mla_static_tile_scheduler,
    MLAStaticTileSchedulerParams,
)


class MLACorrectionRole:
    """Correction warp role for MLA decode kernels.

    Handles output rescaling across KV tiles and final epilogue (normalize,
    convert dtype, write O and LSE to global memory or split-KV workspace).

    Optionally integrates AttentionVariant.transform_output via the fusion
    parameter, allowing custom output normalization (e.g. AttentionWithSink).
    """

    def __init__(
        self,
        config: MLAConfig,
        fusion: AttentionFusion,
        v_dtype=None,
        o_dtype=None,
    ):
        self.acc_dtype = config.acc_dtype
        self.lse_dtype = config.lse_dtype
        self.mma_qk_tiler = config.mma_qk_tiler
        self.mma_pv_tiler = config.mma_pv_tiler
        self.cluster_shape_mnk = config.cluster_shape_mnk
        self.warps_in_n = config.warps_in_n
        self.num_compute_warps = config.num_compute_warps
        self.threads_per_warp = 32
        self.p_cor_stage = config.p_cor_stage
        self.mma_o_stage = config.mma_o_stage
        self.tmem_o_offset = config.tmem_o_offset
        self.correction_factor_offset = config.correction_factor_offset
        self.iterations_pv_n = config.iterations_pv_n
        self.is_var_split_kv = config.is_var_split_kv
        self.enable_pdl = config.enable_pdl
        self.per_iteration_mma_o = config.per_iteration_mma_o
        self.v_dtype = v_dtype
        self.o_dtype = o_dtype

        self.variant = fusion.variant
        self.has_output_transform = fusion.variant.has_output_transform
        self.has_params = fusion.has_params

        self.epilogue_exchange_sync_bar = None

    def set_barriers(self, epilogue_exchange_sync_bar):
        """Set named barriers owned by the kernel."""
        self.epilogue_exchange_sync_bar = epilogue_exchange_sync_bar

    @cute.jit
    def _get_k_tile_count(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        blk_coord: cute.Coord,
    ) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        K = cache_seqs[blk_coord[2]]
        if cutlass.const_expr(self.is_var_split_kv):
            split_kv = block_split_kvs[blk_coord[2]]

        k_tile_total = cute.ceil_div(K, self.mma_qk_tiler[1])
        k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
        k_index = blk_coord[3] * k_tile_per_cta
        k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)
        return k_index, k_tile_count, split_kv

    @cute.jit
    def _make_pv_tiled_mma(self):
        """Create an independent TiledMma for PV partition shape computation.

        The correction role needs TiledMma only for partition_shape_C and
        make_fragment_C — never for actual GEMM. Creating its own instance
        avoids sharing mutable state with the MMA role (which mutates TiledMma
        via .set(ACCUMULATE, ...)).
        """
        cta_group = tcgen05.CtaGroup.TWO
        p_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN
        return sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_pv_tiler[:2],
        )

    @cute.jit
    def _tmem_load_partition(
        self, common_params: SimpleNamespace, pv_tiled_mma: cute.TiledMma, iter_n: int
    ) -> tuple[
        cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma
    ]:
        """Create TMEM load partitions for rescale and epilogue.

        Computes the O accumulator TMEM view at tmem_o_offset, partitions it,
        and creates global memory output views for either mAccO (split-KV
        workspace) or mO (final output).
        """
        tOtO_shape = pv_tiled_mma.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        tOtO = pv_tiled_mma.make_fragment_C(tOtO_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.mma_pv_tiler[1] // self.warps_in_n,
            ),
        )
        tOtO = cute.make_tensor(
            common_params.tmem_ptr + self.tmem_o_offset, tOtO_layout
        )
        tOtO = tOtO[None, None, None, iter_n]

        tAcc = tOtO[(None, None), 0, 0]

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tmem_load_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)
        tmem_load_thr_copy = tmem_load_tiled_copy.get_slice(
            common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        )

        cta_pv_tiler = (
            self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_pv_tiler[1],
            self.mma_pv_tiler[2],
        )
        cta_pv_tiler_mn = cute.select(cta_pv_tiler, mode=[0, 1])

        gO = None
        if cutlass.const_expr(common_params.mAccO is not None):
            gO = cute.local_tile(
                common_params.mAccO[None, common_params.blk_coord[3], None, None, None],
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
            cO = cute.local_tile(
                cute.make_identity_tensor(
                    common_params.mAccO[
                        None, common_params.blk_coord[3], None, None, None
                    ].shape
                ),
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
        else:
            gO = cute.local_tile(
                common_params.mO,
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
            cO = cute.local_tile(
                cute.make_identity_tensor(common_params.mO.shape),
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
        tTR_tAcc = tmem_load_thr_copy.partition_S(tAcc)
        tTR_gO = tmem_load_thr_copy.partition_D(gO)
        tTR_cO = tmem_load_thr_copy.partition_D(cO)
        tTR_rAcc = cute.make_fragment_like(tTR_gO, self.acc_dtype)
        return tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc  # type: ignore[return-value]

    @cute.jit
    def get_correction_factor(
        self,
        common_params: SimpleNamespace,
        p_cor_handle,
    ) -> tuple[
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Int32,
    ]:
        """Load correction metadata from TMEM written by compute warps.

        Releases the p_cor handle after reading, co-locating the data
        consumption with its release.

        Returns (row_sum, row_max, correction_factor, no_correction).
        """
        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        _, tAcc, _, _, _, _ = self._tmem_load_partition(
            common_params, common_params.tiled_mma_pv, 0
        )
        corr_layout = cute.make_layout(
            (tAcc.shape[0], (4, tAcc.shape[1][1]), self.p_cor_stage),
            stride=(tAcc.stride[0], (1, tAcc.stride[1][1]), 4),
        )
        tCor = cute.make_tensor(
            common_params.tmem_ptr + self.correction_factor_offset, corr_layout
        )
        cCor = cute.make_identity_tensor(tCor.shape)
        corr_tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(4)), self.acc_dtype
        )
        corr_tmem_load_tiled_copy = tcgen05.make_tmem_copy(corr_tmem_load_atom, tCor)
        corr_tmem_load_thr_copy = corr_tmem_load_tiled_copy.get_slice(tidx)
        tCor_for_copy = corr_tmem_load_thr_copy.partition_S(tCor)
        cCor_for_copy = corr_tmem_load_thr_copy.partition_D(cCor)
        rCor = cute.make_fragment_like(
            cCor_for_copy[None, None, None, 0], self.acc_dtype
        )
        rCor_int = cute.make_tensor(
            cute.recast_ptr(rCor.iterator, dtype=cutlass.Int32), rCor.layout
        )
        cute.copy(
            corr_tmem_load_tiled_copy,
            tCor_for_copy[None, None, None, p_cor_handle.index],
            rCor,
        )
        row_sum = rCor[0]
        row_max = rCor[1]
        correction_factor = rCor[2]
        no_correction = rCor_int[3]

        p_cor_handle.release()
        return row_sum, row_max, correction_factor, no_correction

    @cute.jit
    def _rescale_one_iter(
        self,
        common_params: SimpleNamespace,
        correction_factor: cutlass.Float32,
        skip_correction: cutlass.Boolean,
        iter_n: int,
    ):
        """Rescale O accumulator for a single iter_n slice.

        Side-effect-only (TMEM load/store + fence). Pipeline ops stay in caller.
        """
        if not skip_correction:
            tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc = (
                self._tmem_load_partition(
                    common_params, common_params.tiled_mma_pv, iter_n
                )
            )
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
            )
            tmem_store_tiled_copy = tcgen05.make_tmem_copy(tmem_store_atom, tAcc)
            cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)
            for i in cutlass.range(
                cute.size(tTR_rAcc), vectorize=True, unroll_full=True
            ):
                tTR_rAcc[i] = tTR_rAcc[i] * correction_factor
            cute.copy(tmem_store_tiled_copy, tTR_rAcc, tTR_tAcc)

        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def rescale(
        self,
        common_params: SimpleNamespace,
        correction_factor: cutlass.Float32,
        no_correction: cutlass.Int32,
        mma_o_handle,
    ):
        """Rescale O accumulator in TMEM by correction_factor (FP16 single-handle path).

        Releases the mma_o handle after fence_view_async_tmem_store,
        co-locating the fence+release pair. Uses vote_all_sync to skip
        rescaling when all threads agree no correction is needed.
        """
        skip_correction = cute.arch.vote_all_sync(no_correction == 1)
        for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
            self._rescale_one_iter(
                common_params, correction_factor, skip_correction, iter_n
            )
        mma_o_handle.release()

    @cute.jit
    def _epilogue_one_iter(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
        tidx: cutlass.Int32,
        iter_n: int,
        params: cute.Tensor = None,
    ):
        """Epilogue for a single iter_n slice.

        Side-effect-only (TMEM load, global store, fence). Pipeline ops stay in caller.
        """
        tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc = (
            self._tmem_load_partition(common_params, common_params.tiled_mma_pv, iter_n)
        )

        cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)

        if cutlass.const_expr(not self.has_output_transform):
            for i in cutlass.range(
                cute.size(tTR_rAcc), vectorize=True, unroll_full=True
            ):
                tTR_rAcc[i] = (
                    tTR_rAcc[i]
                    * epilogue_params.output_scale
                    * cute.arch.rcp_approx(row_sum)
                )
        else:
            if cutlass.const_expr(self.has_params):
                self.variant.params = params
            rcp_d = (
                cute.arch.rcp_approx(row_sum) if row_max != -self.acc_dtype.inf else 0.0
            )
            for i in cutlass.range(
                cute.size(tTR_rAcc), vectorize=False, unroll_full=True
            ):
                qo_head_idx = tTR_cO[i][0] + common_params.cta_m_offset
                tTR_rAcc[i] = self.variant.transform_output(
                    tTR_rAcc[i],
                    common_params.blk_coord[2],
                    common_params.blk_coord[1],
                    qo_head_idx,
                    row_max,
                    rcp_d,
                    epilogue_params.output_scale,
                )

        tR2G_rO_src = None
        tR2G_rO_dst = tTR_gO
        if cutlass.const_expr(common_params.mAccO is None):
            tR2G_rO_src = cute.make_fragment_like(tTR_gO, self.o_dtype)
            tR2G_rO_src.store(tTR_rAcc.load().to(self.o_dtype))
        else:
            tR2G_rO_src = tTR_rAcc

        if cute.elem_less(tTR_cO[0][0], common_params.H):
            cute.autovec_copy(
                tR2G_rO_src,
                tR2G_rO_dst,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )

        cta_pv_tiler = (
            self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_pv_tiler[1],
            self.mma_pv_tiler[2],
        )
        gLSE = None
        cLSE = None
        if cutlass.const_expr(epilogue_params.mAccLSE is None):
            gLSE = cute.local_tile(
                epilogue_params.mLSE,
                (cta_pv_tiler[0], 1, 1),
                (
                    common_params.blk_coord[0],
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
                (1, 1, 1),
            )
            cLSE = cute.local_tile(
                cute.make_identity_tensor(epilogue_params.mLSE.shape),
                (cta_pv_tiler[0], 1, 1),
                (
                    common_params.blk_coord[0],
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
                (1, 1, 1),
            )
        else:
            gLSE = cute.local_tile(
                epilogue_params.mAccLSE[None, common_params.blk_coord[3], None, None],
                (cta_pv_tiler[0], 1, 1),
                (
                    common_params.blk_coord[0],
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
                (1, 1, 1),
            )
            cLSE = cute.local_tile(
                cute.make_identity_tensor(
                    epilogue_params.mAccLSE[
                        None, common_params.blk_coord[3], None, None
                    ].shape
                ),
                (cta_pv_tiler[0], 1, 1),
                (
                    common_params.blk_coord[0],
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
                (1, 1, 1),
            )
        lse = (
            cute.math.log2(row_sum, fastmath=True)
            + epilogue_params.softmax_scale_log2 * row_max
        )
        if cutlass.const_expr(self.warps_in_n == 2):
            if cute.elem_less(cLSE[tidx][0], common_params.H):
                gLSE[tidx] = lse

        cute.arch.fence_view_async_tmem_load()

    @cute.jit
    def epilogue(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
        mma_o_handle,
        params: cute.Tensor = None,
    ):
        """Final epilogue: normalize O, convert dtype, write O and LSE (FP16 single-handle path).

        Releases the mma_o handle after fence_view_async_tmem_load,
        co-locating the fence+release pair.
        """
        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)

        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[tidx] = row_sum
            assert self.epilogue_exchange_sync_bar is not None
            self.epilogue_exchange_sync_bar.arrive_and_wait()
            row_sum = (
                row_sum
                + common_params.smem_exchange[
                    (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ]
            )
        for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
            self._epilogue_one_iter(
                common_params,
                epilogue_params,
                row_sum,
                row_max,
                tidx,
                iter_n,
                params,
            )
        mma_o_handle.release()

    @cute.jit
    def run(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        tile_sched_params: MLAStaticTileSchedulerParams,
        tmem_ptr,
        p_cor_consumer: PipelineConsumer,
        mma_o_consumer: PipelineConsumer,
        compute_common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        params: cute.Tensor = None,
    ):
        """Tile-scheduler loop for the correction warp.

        For each work tile: loads correction factors, rescales O (if not the
        first KV tile), and runs the epilogue on the final KV tile.

        State transitions (wait_and_advance) happen here; release calls are
        co-located with fences inside get_correction_factor/rescale/epilogue.
        """
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
                pv_tiled_mma = self._make_pv_tiled_mma()
                common_params = SimpleNamespace(
                    blk_coord=blk_coord,
                    split_kv=split_kv,
                    local_split_kv=local_split_kv,
                    smem_exchange=compute_common_params.smem_exchange,
                    mAccO=compute_common_params.mAccO,
                    mO=compute_common_params.mO,
                    K=cache_seqs[blk_coord[2]],
                    L=compute_common_params.L,
                    H=compute_common_params.H,
                    cta_m_offset=compute_common_params.cta_m_offset,
                    tmem_ptr=tmem_ptr,
                    tidx=tidx,
                    tiled_mma_pv=pv_tiled_mma,
                )

                k_tile_count_init = k_tile_count
                while k_tile_count > 0:
                    p_cor_handle = p_cor_consumer.wait_and_advance()
                    row_sum, row_max, correction_factor, no_correction = (
                        self.get_correction_factor(common_params, p_cor_handle)
                    )

                    if k_tile_count_init != k_tile_count:
                        if cutlass.const_expr(self.per_iteration_mma_o):
                            skip_correction = cute.arch.vote_all_sync(
                                no_correction == 1
                            )
                            for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
                                mma_o_handle = mma_o_consumer.wait_and_advance()
                                self._rescale_one_iter(
                                    common_params,
                                    correction_factor,
                                    skip_correction,
                                    iter_n,
                                )
                                mma_o_handle.release()
                        else:
                            mma_o_handle = mma_o_consumer.wait_and_advance()
                            self.rescale(
                                common_params,
                                correction_factor,
                                no_correction,
                                mma_o_handle,
                            )

                    k_tile_count = k_tile_count - 1
                    if k_tile_count == 0:
                        if cutlass.const_expr(self.per_iteration_mma_o):
                            tidx = common_params.tidx % (
                                self.num_compute_warps * self.threads_per_warp
                            )
                            if cutlass.const_expr(self.warps_in_n == 2):
                                common_params.smem_exchange[tidx] = row_sum
                                assert self.epilogue_exchange_sync_bar is not None
                                self.epilogue_exchange_sync_bar.arrive_and_wait()
                                row_sum = (
                                    row_sum
                                    + common_params.smem_exchange[
                                        (tidx + 64)
                                        % (
                                            self.num_compute_warps
                                            * self.threads_per_warp
                                        )
                                    ]
                                )
                            for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
                                mma_o_handle = mma_o_consumer.wait_and_advance()
                                self._epilogue_one_iter(
                                    common_params,
                                    epilogue_params,
                                    row_sum,
                                    row_max,
                                    tidx,
                                    iter_n,
                                    params,
                                )
                                mma_o_handle.release()
                        else:
                            mma_o_handle = mma_o_consumer.wait_and_advance()
                            self.epilogue(
                                common_params,
                                epilogue_params,
                                row_sum,
                                row_max,
                                mma_o_handle,
                                params,
                            )

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
