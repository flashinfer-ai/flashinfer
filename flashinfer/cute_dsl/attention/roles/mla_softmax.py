# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLASoftmaxRole — online softmax for MLA decode compute warps.

Reads QK scores from TMEM, computes row-max / row-sum / correction factor,
quantizes P to SMEM, and manages the mma_s / p_mma pipeline handshake.
"""

from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline

from .softmax_math import exp2_scale, packed_row_sum


class MLASoftmaxRole:
    def __init__(self, config, mainloop, schedule, exchange_sync_bar):
        self.mma_qk_tiler = config.mma_qk_tiler
        self.cluster_shape_mnk = config.cluster_shape_mnk
        self.warps_in_n = config.warps_in_n
        self.acc_dtype = config.acc_dtype
        self.mma_s_stage = mainloop.mma_s_stages
        self.num_compute_warps = schedule.num_compute_warps
        self.threads_per_warp = schedule.threads_per_warp
        self.exchange_sync_bar = exchange_sync_bar
        self.q_dtype = None  # set after dtypes known

    @cute.jit
    def dispatch_apply_mask(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_total: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        row_max: cutlass.Float32,
        row_sum: cutlass.Float32,
        correction_factor: cutlass.Float32,
    ) -> tuple[
        pipeline.PipelineState,
        pipeline.PipelineState,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
    ]:
        if k_index == k_tile_total - 1:
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.run(
                common_params,
                softmax_params,
                k_index,
                mma_s_consumer_state,
                p_mma_producer_state,
                row_max,
                row_sum,
                correction_factor,
                True,
            )
        else:
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.run(
                common_params,
                softmax_params,
                k_index,
                mma_s_consumer_state,
                p_mma_producer_state,
                row_max,
                row_sum,
                correction_factor,
                False,
            )
        return (
            mma_s_consumer_state,
            p_mma_producer_state,
            row_max,
            row_sum,
            correction_factor,
        )

    @cute.jit
    def run(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        row_max: cutlass.Float32,
        row_sum: cutlass.Float32,
        correction_factor: cutlass.Float32,
        is_last_tile: bool,
    ) -> tuple[
        pipeline.PipelineState,
        pipeline.PipelineState,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
    ]:
        softmax_params.p_mma_pipeline.producer_acquire(p_mma_producer_state)
        softmax_params.mma_s_pipeline.consumer_wait(mma_s_consumer_state)

        tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged_fake = softmax_params.tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)
        tStS = tStS_staged[None, None, None, mma_s_consumer_state.index]

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

        tTR_rAcc = cute.make_fragment_like(tTR_tS, self.acc_dtype)

        cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)

        row_max_new = row_max
        for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
            if cutlass.const_expr(is_last_tile):
                tTR_rAcc[i] = (
                    tTR_rAcc[i]
                    if cute.elem_less(
                        tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index,
                        common_params.K,
                    )
                    else -self.acc_dtype.inf
                )
            row_max_new = cute.arch.fmax(row_max_new, tTR_rAcc[i])

        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[tidx] = row_max_new
            self.exchange_sync_bar.wait()
            row_max_new = cute.arch.fmax(
                row_max_new,
                common_params.smem_exchange[
                    (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ],
            )

        correction_factor = cute.arch.exp2(
            (row_max - row_max_new) * softmax_params.softmax_scale_log2
        )
        row_max = row_max_new
        exp2_scale(tTR_rAcc, softmax_params.softmax_scale_log2, row_max_new)

        tTR_rS = cute.make_fragment_like(tTR_tS, self.q_dtype)

        tTR_rS.store(tTR_rAcc.load().to(self.q_dtype))

        sP = softmax_params.sP[None, None, None, (None, p_mma_producer_state.index)]
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
        swizzle_bits = 2 if self.q_dtype.width == 16 else 1
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

        row_sum = row_sum * correction_factor
        row_sum_vec = packed_row_sum(tTR_rAcc)
        row_sum = row_sum_vec[0] + row_sum_vec[1] + row_sum

        cute.arch.fence_view_async_tmem_load()
        cute.arch.fence_view_async_shared()

        softmax_params.mma_s_pipeline.consumer_release(mma_s_consumer_state)
        softmax_params.p_mma_pipeline.producer_commit(p_mma_producer_state)
        mma_s_consumer_state.advance()
        p_mma_producer_state.advance()

        return (
            mma_s_consumer_state,
            p_mma_producer_state,
            row_max,
            row_sum,
            correction_factor,
        )
