# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLAComputeRole — orchestrator for MLA decode compute warps.

Thin dispatcher that delegates to MLASoftmaxRole, MLARescaleRole, and
MLAEpilogueRole, matching the C++ CUTLASS collectives pattern of
separate concrete role types.
"""

from types import SimpleNamespace
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline

from .mla_softmax import MLASoftmaxRole
from .mla_rescale import MLARescaleRole
from .mla_epilogue import MLAEpilogueRole

from ..scheduler.mla_persistent import (
    MLAStaticTileSchedulerParams,
    create_mla_static_tile_scheduler,
)


class MLAComputeRole:
    def __init__(self, config, mainloop, schedule, exchange_sync_bar):
        self.mma_qk_tiler = config.mma_qk_tiler
        self.acc_dtype = config.acc_dtype
        self.is_var_split_kv = config.is_var_split_kv

        self.mma_s_stages = mainloop.mma_s_stages
        self.p_mma_stages = mainloop.p_mma_stages
        self.mma_o_stages = mainloop.mma_o_stages

        self.softmax_role = MLASoftmaxRole(
            config, mainloop, schedule, exchange_sync_bar
        )
        self.rescale_role = MLARescaleRole(config, mainloop, schedule)
        self.epilogue_role = MLAEpilogueRole(
            config, mainloop, schedule, exchange_sync_bar
        )

        self._q_dtype = None
        self._o_dtype = None

    @property
    def q_dtype(self):
        return self._q_dtype

    @q_dtype.setter
    def q_dtype(self, value):
        self._q_dtype = value
        self.softmax_role.q_dtype = value

    @property
    def o_dtype(self):
        return self._o_dtype

    @o_dtype.setter
    def o_dtype(self, value):
        self._o_dtype = value
        self.epilogue_role.o_dtype = value

    @cute.jit
    def _get_k_tile_count(self, split_kv, cache_seqs, block_split_kvs, blk_coord):
        K = cache_seqs[blk_coord[2]]
        if cutlass.const_expr(self.is_var_split_kv):
            split_kv = block_split_kvs[blk_coord[2]]
        k_tile_total = cute.ceil_div(K, self.mma_qk_tiler[1])
        k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
        k_index = blk_coord[3] * k_tile_per_cta
        k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)
        return k_index, k_tile_count, split_kv

    @cute.jit
    def run(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        sP: cute.Tensor,
        smem_exchange: cute.Tensor,
        mma_s_pipeline,
        p_mma_pipeline,
        mma_o_pipeline,
        tmem_holding_buf,
        tmem_dealloc_mbar_ptr,
        tmem_ptr_sync_bar,
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mAccO: Optional[cute.Tensor],
        mAccLSE: Optional[cute.Tensor],
        L: cutlass.Int32,
        cache_seqs: cute.Tensor,
        split_kv: cutlass.Int32,
        block_split_kvs: Optional[cute.Tensor],
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        tidx: cutlass.Int32,
        cta_rank_in_cluster: cutlass.Int32,
        tile_sched_params: MLAStaticTileSchedulerParams,
    ):
        """Compute warp orchestration loop for MLA decode.

        Owns the tile scheduler loop, constructs per-tile params,
        and delegates to compute() for each work tile.
        """
        mma_s_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_s_stages
        )
        p_mma_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.p_mma_stages
        )
        mma_o_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_o_stages
        )

        tmem_ptr_sync_bar.wait()

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            alignment=16,
            ptr_to_buffer_holding_addr=tmem_holding_buf,
        )

        tile_sched = create_mla_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        while work_tile.is_valid_tile:
            blk_coord = work_tile.tile_idx
            k_index, k_tile_count, local_split_kv = self._get_k_tile_count(
                split_kv, cache_seqs, block_split_kvs, blk_coord,
            )
            if k_tile_count > 0:
                compute_common_params = SimpleNamespace(
                    blk_coord=blk_coord,
                    split_kv=split_kv,
                    local_split_kv=local_split_kv,
                    smem_exchange=smem_exchange,
                    mAccO=mAccO,
                    mO=mO,
                    K=cache_seqs[blk_coord[2]],
                    L=L,
                    tmem_ptr=tmem_ptr,
                    tidx=tidx,
                )
                compute_softmax_params = SimpleNamespace(
                    tiled_mma_qk=tiled_mma_qk,
                    sP=sP,
                    mma_s_pipeline=mma_s_pipeline,
                    p_mma_pipeline=p_mma_pipeline,
                    softmax_scale_log2=softmax_scale_log2,
                )
                compute_rescale_params = SimpleNamespace(
                    tiled_mma_pv=tiled_mma_pv,
                    mma_o_pipeline=mma_o_pipeline,
                )
                compute_epilogue_params = SimpleNamespace(
                    tiled_mma_pv=tiled_mma_pv,
                    mma_o_pipeline=mma_o_pipeline,
                    output_scale=output_scale,
                    softmax_scale_log2=softmax_scale_log2,
                    mAccLSE=mAccLSE,
                    mLSE=mLSE,
                )
                mma_s_consumer_state, p_mma_producer_state, mma_o_consumer_state = (
                    self.compute(
                        compute_common_params,
                        compute_softmax_params,
                        compute_rescale_params,
                        compute_epilogue_params,
                        k_index=k_index,
                        k_tile_count=k_tile_count,
                        mma_s_consumer_state=mma_s_consumer_state,
                        p_mma_producer_state=p_mma_producer_state,
                        mma_o_consumer_state=mma_o_consumer_state,
                    )
                )
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1)

    @cute.jit
    def compute(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        rescale_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        mma_o_consumer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]:
        k_tile_total = cute.ceil_div(common_params.K, self.mma_qk_tiler[1])

        row_max = -self.acc_dtype.inf
        row_sum = self.acc_dtype(0)
        correction_factor = self.acc_dtype(1)
        k_index_init = k_index
        while k_tile_count > 0:
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.softmax_role.dispatch_apply_mask(
                common_params,
                softmax_params,
                k_index,
                k_tile_total,
                mma_s_consumer_state,
                p_mma_producer_state,
                row_max,
                row_sum,
                correction_factor,
            )
            if k_index > k_index_init:
                mma_o_consumer_state = self.rescale_role.run(
                    common_params,
                    rescale_params,
                    mma_o_consumer_state,
                    correction_factor,
                )
            k_index = k_index + 1
            k_tile_count = k_tile_count - 1

        mma_o_consumer_state = self.epilogue_role.run(
            common_params, epilogue_params, mma_o_consumer_state, row_max, row_sum
        )
        return mma_s_consumer_state, p_mma_producer_state, mma_o_consumer_state
