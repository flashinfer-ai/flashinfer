# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLAEpilogueRole — final output write for MLA decode compute warps.

After all KV tiles are processed, reads the accumulated output from TMEM,
divides by row_sum, converts to output dtype, and writes to global memory.
Also writes the LSE (log-sum-exp) output.
"""

from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline

from .tmem_utils import tmem_load_partition


class MLAEpilogueRole:
    def __init__(self, config, mainloop, schedule, exchange_sync_bar):
        self.mma_pv_tiler = config.mma_pv_tiler
        self.cluster_shape_mnk = config.cluster_shape_mnk
        self.num_heads = config.num_heads
        self.warps_in_n = config.warps_in_n
        self.acc_dtype = config.acc_dtype
        self.iterations_pv_n = config.iterations_pv_n
        self.tmem_o_offset = mainloop.tmem_o_offset
        self.num_compute_warps = schedule.num_compute_warps
        self.threads_per_warp = schedule.threads_per_warp
        self.exchange_sync_bar = exchange_sync_bar
        self.o_dtype = None  # set after dtypes known

    @cute.jit
    def run(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        row_max: cutlass.Float32,
        row_sum: cutlass.Float32,
    ) -> pipeline.PipelineState:
        epilogue_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)

        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[common_params.tidx] = row_sum
            self.exchange_sync_bar.wait()
            row_sum = (
                row_sum
                + common_params.smem_exchange[
                    (common_params.tidx + 64)
                    % (self.num_compute_warps * self.threads_per_warp)
                ]
            )

        for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
            tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc = (
                tmem_load_partition(
                    common_params.tmem_ptr,
                    self.tmem_o_offset,
                    self.acc_dtype,
                    self.mma_pv_tiler,
                    self.cluster_shape_mnk,
                    self.warps_in_n,
                    self.num_compute_warps,
                    self.threads_per_warp,
                    common_params,
                    epilogue_params.tiled_mma_pv,
                    iter_n,
                )
            )

            cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)

            tTR_rAcc.store(
                tTR_rAcc.load()
                * epilogue_params.output_scale
                * cute.arch.rcp_approx(row_sum)
            )

            tR2G_rO_src = None
            tR2G_rO_dst = tTR_gO
            if cutlass.const_expr(common_params.mAccO is None):
                tR2G_rO_src = cute.make_fragment_like(tTR_gO, self.o_dtype)
                tR2G_rO_src.store(tTR_rAcc.load().to(self.o_dtype))
            else:
                tR2G_rO_src = tTR_rAcc
            if cute.elem_less(tTR_cO[0][0], self.num_heads):
                cute.autovec_copy(tR2G_rO_src, tR2G_rO_dst)

            cta_pv_tiler = (
                self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],
                self.mma_pv_tiler[1],
                self.mma_pv_tiler[2],
            )
            gLSE = None
            if cutlass.const_expr(epilogue_params.mAccLSE is None):
                gLSE = cute.local_tile(
                    epilogue_params.mLSE,
                    (cta_pv_tiler[0], 1, 1),
                    (
                        common_params.blk_coord[0],
                        common_params.blk_coord[1],
                        common_params.blk_coord[2],
                    ),
                    (1, None, 1),
                )
                cLSE = cute.local_tile(
                    cute.make_identity_tensor(epilogue_params.mLSE.shape),
                    (cta_pv_tiler[0], 1, 1),
                    (
                        common_params.blk_coord[0],
                        common_params.blk_coord[1],
                        common_params.blk_coord[2],
                    ),
                    (1, None, 1),
                )
            else:
                gLSE = cute.local_tile(
                    epilogue_params.mAccLSE[None, common_params.blk_coord[3], None],
                    (cta_pv_tiler[0], 1, 1),
                    (
                        common_params.blk_coord[0],
                        common_params.blk_coord[1],
                        common_params.blk_coord[2],
                    ),
                    (1, None, 1),
                )
                cLSE = cute.local_tile(
                    cute.make_identity_tensor(
                        epilogue_params.mAccLSE[
                            None, common_params.blk_coord[3], None
                        ].shape
                    ),
                    (cta_pv_tiler[0], 1, 1),
                    (
                        common_params.blk_coord[0],
                        common_params.blk_coord[1],
                        common_params.blk_coord[2],
                    ),
                    (1, None, 1),
                )
            lse = cute.math.log2(row_sum) + epilogue_params.softmax_scale_log2 * row_max
            if cutlass.const_expr(self.warps_in_n == 2):
                if cute.elem_less(cLSE[common_params.tidx][0], self.num_heads):
                    gLSE[common_params.tidx] = lse

        cute.arch.fence_view_async_tmem_load()
        epilogue_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
        mma_o_consumer_state.advance()

        return mma_o_consumer_state
