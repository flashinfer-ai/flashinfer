# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLAMmaRole — MMA operations for MLA decode.

Holds compile-time config for the MMA warp and provides @cute.jit
methods for Q*K^T and P*V computation.
"""

from types import SimpleNamespace
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline

from ..scheduler.mla_persistent import (
    MLAStaticTileSchedulerParams,
    create_mla_static_tile_scheduler,
)


class MLAMmaRole:
    def __init__(self, config, mainloop):
        self.mma_qk_tiler = config.mma_qk_tiler
        self.mma_pv_tiler = config.mma_pv_tiler
        self.iterations_qk = config.iterations_qk
        self.iterations_pv_k = config.iterations_pv_k
        self.iterations_pv_n = config.iterations_pv_n
        self.warps_in_n = config.warps_in_n
        self.use_2cta_instrs = config.use_2cta_instrs
        self.acc_dtype = config.acc_dtype
        self.is_var_split_kv = config.is_var_split_kv
        self.mma_s_stage = mainloop.mma_s_stages
        self.mma_o_stage = mainloop.mma_o_stages
        self.tmem_o_offset = mainloop.tmem_o_offset
        self.load_q_stages = mainloop.load_q_stages
        self.load_kv_stages = mainloop.load_kv_stages
        self.mma_s_stages = mainloop.mma_s_stages
        self.p_mma_stages = mainloop.p_mma_stages
        self.mma_o_stages = mainloop.mma_o_stages

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
        sQ: cute.Tensor,
        sKC: cute.Tensor,
        sP: cute.Tensor,
        sVC: cute.Tensor,
        load_q_pipeline,
        load_kv_pipeline,
        mma_s_pipeline,
        p_mma_pipeline,
        mma_o_pipeline,
        tmem_holding_buf,
        tmem_dealloc_mbar_ptr,
        tmem_ptr_sync_bar,
        L: cutlass.Int32,
        is_leader_cta,
        tile_sched_params: MLAStaticTileSchedulerParams,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: Optional[cute.Tensor],
    ):
        """MMA warp orchestration loop for MLA decode.

        Manages TMEM allocation/deallocation, tile scheduler loop,
        and delegates to mma() for each work tile.
        """
        cute.arch.alloc_tmem(
            cute.arch.SM100_TMEM_CAPACITY_COLUMNS,
            tmem_holding_buf,
            is_two_cta=self.use_2cta_instrs,
        )

        tmem_ptr_sync_bar.arrive()

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            alignment=16,
            ptr_to_buffer_holding_addr=tmem_holding_buf,
        )

        load_q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_q_stages
        )
        load_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_kv_stages
        )
        mma_s_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_s_stages
        )
        p_mma_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.p_mma_stages
        )
        mma_o_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_o_stages
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
                mma_common_params = SimpleNamespace(
                    blk_coord=blk_coord,
                    local_split_kv=local_split_kv,
                    load_q_pipeline=load_q_pipeline,
                    load_kv_pipeline=load_kv_pipeline,
                    tmem_ptr=tmem_ptr,
                    is_leader_cta=is_leader_cta,
                    L=L,
                )
                mma_qk_params = SimpleNamespace(
                    mma_s_pipeline=mma_s_pipeline,
                    sQ=sQ,
                    sKC=sKC,
                )
                mma_pv_params = SimpleNamespace(
                    p_mma_pipeline=p_mma_pipeline,
                    mma_o_pipeline=mma_o_pipeline,
                    sP=sP,
                    sVC=sVC,
                )
                (
                    tiled_mma_qk,
                    tiled_mma_pv,
                    load_q_consumer_state,
                    load_kv_consumer_state,
                    mma_s_producer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                ) = self.mma(
                    mma_common_params,
                    mma_qk_params,
                    mma_pv_params,
                    k_tile_count,
                    tiled_mma_qk,
                    tiled_mma_pv,
                    load_q_consumer_state,
                    load_kv_consumer_state,
                    mma_s_producer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                )
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        mma_s_pipeline.producer_tail(mma_s_producer_state)
        mma_o_pipeline.producer_tail(mma_o_producer_state)

        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=self.use_2cta_instrs)
        cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)

        cute.arch.dealloc_tmem(
            tmem_ptr,
            cute.arch.SM100_TMEM_CAPACITY_COLUMNS,
            is_two_cta=self.use_2cta_instrs,
        )

    @cute.jit
    def mma(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        pv_params: SimpleNamespace,
        k_tile_count: cutlass.Int32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        load_q_consumer_state: pipeline.PipelineState,
        load_kv_consumer_state: pipeline.PipelineState,
        mma_s_producer_state: pipeline.PipelineState,
        p_mma_consumer_state: pipeline.PipelineState,
        mma_o_producer_state: pipeline.PipelineState,
    ) -> tuple[
        cute.TiledMma,
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        tSrQ = tiled_mma_qk.make_fragment_A(qk_params.sQ)
        tSrKC = tiled_mma_qk.make_fragment_B(qk_params.sKC)
        tOrP = tiled_mma_pv.make_fragment_A(pv_params.sP)
        tOrVC = tiled_mma_pv.make_fragment_B(pv_params.sVC)

        tStS_shape = tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged_fake = tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)
        tOtO_shape = tiled_mma_pv.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        assert self.mma_o_stage == 1, (
            "mma O has 1 stage, otherwise the tmem usage exceeds the limit."
        )
        tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.mma_pv_tiler[1] // self.warps_in_n,
            ),
        )
        tOtO_staged = cute.make_tensor(
            tStS_staged.iterator + self.tmem_o_offset, tOtO_layout
        )

        qk_params.tSrQ = tSrQ
        qk_params.tSrKC = tSrKC
        qk_params.tStS_staged = tStS_staged
        pv_params.tOrP = tOrP
        pv_params.tOrVC = tOrVC
        pv_params.tOtO_staged = tOtO_staged

        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)
        load_q_pipeline = common_params.load_q_pipeline
        if common_params.is_leader_cta:
            load_q_release_state = load_q_consumer_state.clone()
            (
                tiled_mma_qk,
                load_q_consumer_state,
                load_kv_consumer_state,
                mma_s_producer_state,
            ) = self.mma_qk(
                common_params,
                qk_params,
                tiled_mma_qk,
                load_q_consumer_state,
                load_kv_consumer_state,
                mma_s_producer_state,
                wait_q=True,
            )
            k_tile_count -= 1

            while k_tile_count > 0:
                (
                    tiled_mma_qk,
                    load_q_consumer_state,
                    load_kv_consumer_state,
                    mma_s_producer_state,
                ) = self.mma_qk(
                    common_params,
                    qk_params,
                    tiled_mma_qk,
                    load_q_consumer_state,
                    load_kv_consumer_state,
                    mma_s_producer_state,
                    wait_q=False,
                )
                (
                    tiled_mma_pv,
                    load_kv_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                ) = self.mma_pv(
                    common_params,
                    pv_params,
                    tiled_mma_pv,
                    load_kv_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                )
                k_tile_count -= 1
            for _ in cutlass.range_constexpr(self.iterations_qk):
                load_q_pipeline.consumer_release(load_q_release_state)
                load_q_release_state.advance()
            (
                tiled_mma_pv,
                load_kv_consumer_state,
                p_mma_consumer_state,
                mma_o_producer_state,
            ) = self.mma_pv(
                common_params,
                pv_params,
                tiled_mma_pv,
                load_kv_consumer_state,
                p_mma_consumer_state,
                mma_o_producer_state,
            )

        return (
            tiled_mma_qk,
            tiled_mma_pv,
            load_q_consumer_state,
            load_kv_consumer_state,
            mma_s_producer_state,
            p_mma_consumer_state,
            mma_o_producer_state,
        )

    @cute.jit
    def mma_qk(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        tiled_mma_qk: cute.TiledMma,
        load_q_consumer_state: pipeline.PipelineState,
        load_kv_consumer_state: pipeline.PipelineState,
        mma_s_producer_state: pipeline.PipelineState,
        wait_q: bool,
    ) -> tuple[
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        tStS = qk_params.tStS_staged[None, None, None, mma_s_producer_state.index]

        qk_params.mma_s_pipeline.producer_acquire(mma_s_producer_state)
        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
        load_q_pipeline = common_params.load_q_pipeline
        load_kv_pipeline = common_params.load_kv_pipeline
        for q_stage in range(self.iterations_qk):
            if cutlass.const_expr(wait_q):
                load_q_pipeline.consumer_wait(load_q_consumer_state)
            load_kv_pipeline.consumer_wait(load_kv_consumer_state)
            kc_stage = load_kv_consumer_state.index
            for k_block in cutlass.range_constexpr(qk_params.tSrQ.shape[2]):
                cute.gemm(
                    tiled_mma_qk,
                    tStS,
                    qk_params.tSrQ[None, None, k_block, q_stage],
                    qk_params.tSrKC[None, None, k_block, kc_stage],
                    tStS,
                )
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
            load_kv_pipeline.consumer_release(load_kv_consumer_state)
            load_kv_consumer_state.advance()
            if cutlass.const_expr(wait_q):
                load_q_consumer_state.advance()
        qk_params.mma_s_pipeline.producer_commit(mma_s_producer_state)
        mma_s_producer_state.advance()
        return (
            tiled_mma_qk,
            load_q_consumer_state,
            load_kv_consumer_state,
            mma_s_producer_state,
        )

    @cute.jit
    def mma_pv(
        self,
        common_params: SimpleNamespace,
        pv_params: SimpleNamespace,
        tiled_mma_pv: cute.TiledMma,
        load_kv_consumer_state: pipeline.PipelineState,
        p_mma_consumer_state: pipeline.PipelineState,
        mma_o_producer_state: pipeline.PipelineState,
    ) -> tuple[
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        pv_params.mma_o_pipeline.producer_acquire(mma_o_producer_state)
        pv_params.p_mma_pipeline.consumer_wait(p_mma_consumer_state)
        load_kv_pipeline = common_params.load_kv_pipeline
        for p_stage in range(self.iterations_pv_k):
            accumulate_flag = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
            for acc_stage in range(self.iterations_pv_n):
                load_kv_pipeline.consumer_wait(load_kv_consumer_state)
                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, accumulate_flag)
                vc_stage = load_kv_consumer_state.index
                tOtO = pv_params.tOtO_staged[None, None, None, acc_stage]
                for k_block in cutlass.range_constexpr(pv_params.tOrP.shape[2]):
                    cute.gemm(
                        tiled_mma_pv,
                        tOtO,
                        pv_params.tOrP[
                            None,
                            None,
                            k_block,
                            (p_stage, p_mma_consumer_state.index),
                        ],
                        pv_params.tOrVC[None, None, k_block, vc_stage],
                        tOtO,
                    )
                    tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                load_kv_pipeline.consumer_release(load_kv_consumer_state)
                load_kv_consumer_state.advance()
        pv_params.p_mma_pipeline.consumer_release(p_mma_consumer_state)
        p_mma_consumer_state.advance()
        pv_params.mma_o_pipeline.producer_commit(mma_o_producer_state)
        mma_o_producer_state.advance()

        return (
            tiled_mma_pv,
            load_kv_consumer_state,
            p_mma_consumer_state,
            mma_o_producer_state,
        )
