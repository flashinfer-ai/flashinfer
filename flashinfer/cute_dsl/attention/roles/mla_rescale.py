# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLARescaleRole — correction/rescaling for MLA decode compute warps.

When row-max changes across KV tiles, multiplies the partial output
accumulator in TMEM by the correction factor. Manages the mma_o pipeline
handshake.
"""

from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline

from .tmem_utils import tmem_load_partition


class MLARescaleRole:
    def __init__(self, config, mainloop, schedule):
        self.mma_pv_tiler = config.mma_pv_tiler
        self.cluster_shape_mnk = config.cluster_shape_mnk
        self.warps_in_n = config.warps_in_n
        self.acc_dtype = config.acc_dtype
        self.iterations_pv_n = config.iterations_pv_n
        self.tmem_o_offset = mainloop.tmem_o_offset
        self.num_compute_warps = schedule.num_compute_warps
        self.threads_per_warp = schedule.threads_per_warp

    @cute.jit
    def run(
        self,
        common_params: SimpleNamespace,
        rescale_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        correction_factor: cutlass.Float32,
    ) -> pipeline.PipelineState:
        rescale_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)

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
                    rescale_params.tiled_mma_pv,
                    iter_n,
                )
            )
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
            )
            tmem_store_tiled_copy = tcgen05.make_tmem_copy(tmem_store_atom, tAcc)

            cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)
            for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
                tTR_rAcc[i], tTR_rAcc[i + 1] = cute.arch.mul_packed_f32x2(
                    (tTR_rAcc[i], tTR_rAcc[i + 1]),
                    (correction_factor, correction_factor),
                )
            cute.copy(tmem_store_tiled_copy, tTR_rAcc, tTR_tAcc)

        cute.arch.fence_view_async_tmem_store()
        rescale_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
        mma_o_consumer_state.advance()

        return mma_o_consumer_state
