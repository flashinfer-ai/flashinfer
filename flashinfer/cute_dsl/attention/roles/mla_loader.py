# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLALoaderRole — TMA load operations for MLA decode.

Holds the compile-time config subset needed for loading Q/K/V tiles
and provides @cute.jit methods for the load warp.
"""

from types import SimpleNamespace
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.pipeline as pipeline

from ..scheduler.mla_persistent import (
    MLAStaticTileSchedulerParams,
    create_mla_static_tile_scheduler,
)


class MLALoaderRole:
    def __init__(self, config):
        self.iterations_qk_latent = config.iterations_qk_latent
        self.iterations_qk_rope = config.iterations_qk_rope
        self.iterations_pv_k = config.iterations_pv_k
        self.iterations_pv_n = config.iterations_pv_n
        self.use_page_table = config.use_page_table
        self.is_var_split_kv = config.is_var_split_kv
        self.mma_qk_tiler = config.mma_qk_tiler
        self.mma_pv_tiler = config.mma_pv_tiler

    @cute.jit
    def run(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_atom_q_latent: cute.CopyAtom,
        mQL: cute.Tensor,
        tma_atom_q_rope: cute.CopyAtom,
        mQR: cute.Tensor,
        tma_atom_c_latent: cute.CopyAtom,
        mCL: cute.Tensor,
        tma_atom_c_rope: cute.CopyAtom,
        mKR: cute.Tensor,
        tma_atom_c_latent_transpose: cute.CopyAtom,
        mCLT: cute.Tensor,
        sQ: cute.Tensor,
        sKC: cute.Tensor,
        sVC: cute.Tensor,
        load_q_pipeline,
        load_kv_pipeline,
        load_q_stages: int,
        load_kv_stages: int,
        mPT: Optional[cute.Tensor],
        tile_sched_params: MLAStaticTileSchedulerParams,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: Optional[cute.Tensor],
    ):
        """Loader warp orchestration loop for MLA decode.

        Owns the tile scheduler loop, constructs per-tile params, and
        delegates to load_tma() for each work tile.
        """
        load_q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, load_q_stages
        )
        load_kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, load_kv_stages
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
                tma_common_params = SimpleNamespace(
                    blk_coord=blk_coord,
                    local_split_kv=local_split_kv,
                    load_q_pipeline=load_q_pipeline,
                    load_kv_pipeline=load_kv_pipeline,
                    mPT=mPT,
                )
                tma_qk_params = SimpleNamespace(
                    tiled_mma_qk=tiled_mma_qk,
                    tma_atom_q_latent=tma_atom_q_latent,
                    tma_atom_q_rope=tma_atom_q_rope,
                    tma_atom_c_latent=tma_atom_c_latent,
                    tma_atom_c_rope=tma_atom_c_rope,
                    mQL=mQL,
                    mQR=mQR,
                    mCL=mCL,
                    mKR=mKR,
                    sQ=sQ,
                    sKC=sKC,
                )
                tma_pv_params = SimpleNamespace(
                    tiled_mma_pv=tiled_mma_pv,
                    tma_atom_c_latent_transpose=tma_atom_c_latent_transpose,
                    mCL=mCL,
                    mKR=mKR,
                    mCLT=mCLT,
                    sVC=sVC,
                )
                load_q_producer_state, load_kv_producer_state = self.load_tma(
                    tma_common_params,
                    tma_qk_params,
                    tma_pv_params,
                    k_index,
                    k_tile_count,
                    load_q_producer_state,
                    load_kv_producer_state,
                )
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        load_q_pipeline.producer_tail(load_q_producer_state)
        load_kv_pipeline.producer_tail(load_kv_producer_state)

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
    def load_tma(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_q_producer_state: pipeline.PipelineState,
        load_kv_producer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        mPT = None
        if cutlass.const_expr(self.use_page_table):
            mPT = common_params.mPT[None, common_params.blk_coord[2]]

        mma_qk_tiler_mk = cute.select(self.mma_qk_tiler, mode=[0, 2])
        gQL = cute.flat_divide(qk_params.mQL, mma_qk_tiler_mk)
        gQR = cute.flat_divide(qk_params.mQR, mma_qk_tiler_mk)

        mma_qk_tiler_nk = cute.select(self.mma_qk_tiler, mode=[1, 2])
        gCL = cute.flat_divide(qk_params.mCL, mma_qk_tiler_nk)
        gKR = cute.flat_divide(qk_params.mKR, mma_qk_tiler_nk)

        thr_mma_qk = qk_params.tiled_mma_qk.get_slice(
            common_params.blk_coord[0] % cute.size(qk_params.tiled_mma_qk.thr_id)
        )
        tSgQL = thr_mma_qk.partition_A(gQL)
        tSgQR = thr_mma_qk.partition_A(gQR)

        tSgCL = thr_mma_qk.partition_B(gCL)
        tSgKR = thr_mma_qk.partition_B(gKR)

        tQsQ, tQLgQL_mkl = cpasync.tma_partition(
            qk_params.tma_atom_q_latent,
            0,
            cute.make_layout(1),
            cute.group_modes(qk_params.sQ, 0, 3),
            cute.group_modes(tSgQL, 0, 3),
        )

        _, tQRgQR_mkl = cpasync.tma_partition(
            qk_params.tma_atom_q_rope,
            0,
            cute.make_layout(1),
            cute.group_modes(qk_params.sQ, 0, 3),
            cute.group_modes(tSgQR, 0, 3),
        )

        tKCsKC, tCLgCL = cpasync.tma_partition(
            qk_params.tma_atom_c_latent,
            0,
            cute.make_layout(1),
            cute.group_modes(qk_params.sKC, 0, 3),
            cute.group_modes(tSgCL, 0, 3),
        )

        _, tKRgKR = cpasync.tma_partition(
            qk_params.tma_atom_c_rope,
            0,
            cute.make_layout(1),
            cute.group_modes(qk_params.sKC, 0, 3),
            cute.group_modes(tSgKR, 0, 3),
        )

        tQLgQL = tQLgQL_mkl[None, None, None, common_params.blk_coord[2]]
        tQRgQR = tQRgQR_mkl[None, None, None, common_params.blk_coord[2]]

        mma_pv_tiler_nk = cute.select(self.mma_pv_tiler, mode=[1, 2])
        gCLT = cute.flat_divide(v_params.mCLT, mma_pv_tiler_nk)

        thr_mma_pv = v_params.tiled_mma_pv.get_slice(
            common_params.blk_coord[0] % cute.size(v_params.tiled_mma_pv.thr_id)
        )
        tOgCLT = thr_mma_pv.partition_B(gCLT)

        tVCsVC, tCLTgCLT = cpasync.tma_partition(
            v_params.tma_atom_c_latent_transpose,
            0,
            cute.make_layout(1),
            cute.group_modes(v_params.sVC, 0, 3),
            cute.group_modes(tOgCLT, 0, 3),
        )

        common_params.mPT = mPT
        qk_params.tQLgQL = tQLgQL
        qk_params.tQRgQR = tQRgQR
        qk_params.tCLgCL = tCLgCL
        qk_params.tKRgKR = tKRgKR
        qk_params.tQsQ = tQsQ
        qk_params.tKCsKC = tKCsKC
        v_params.tCLTgCLT = tCLTgCLT
        v_params.tVCsVC = tVCsVC

        load_q_producer_state, load_kv_producer_state = self.load_tma_qk_one_k_tile(
            common_params,
            qk_params,
            k_index,
            k_tile_count,
            load_q_producer_state,
            load_kv_producer_state,
            load_q=True,
        )
        k_index += 1
        k_tile_count -= 1
        while k_tile_count > 0:
            load_q_producer_state, load_kv_producer_state = self.load_tma_qk_one_k_tile(
                common_params,
                qk_params,
                k_index,
                k_tile_count,
                load_q_producer_state,
                load_kv_producer_state,
                load_q=False,
            )
            load_kv_producer_state = self.load_tma_v_one_k_tile(
                common_params,
                v_params,
                k_index - 1,
                load_kv_producer_state,
            )
            k_index += 1
            k_tile_count -= 1

        load_kv_producer_state = self.load_tma_v_one_k_tile(
            common_params,
            v_params,
            k_index - 1,
            load_kv_producer_state,
        )
        return load_q_producer_state, load_kv_producer_state

    @cute.jit
    def load_tma_qk_one_k_tile(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_q_producer_state: pipeline.PipelineState,
        load_kv_producer_state: pipeline.PipelineState,
        load_q: bool,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        for i in cutlass.range_constexpr(self.iterations_qk_latent):
            if cutlass.const_expr(load_q):
                tma_bar_ptr = common_params.load_q_pipeline.producer_get_barrier(
                    load_q_producer_state
                )
                common_params.load_q_pipeline.producer_acquire(load_q_producer_state)
                cute.copy(
                    qk_params.tma_atom_q_latent,
                    qk_params.tQLgQL[None, 0, load_q_producer_state.index],
                    qk_params.tQsQ[None, load_q_producer_state.index],
                    tma_bar_ptr=tma_bar_ptr,
                )
                load_q_producer_state.advance()
            tma_bar_ptr = common_params.load_kv_pipeline.producer_get_barrier(
                load_kv_producer_state
            )
            common_params.load_kv_pipeline.producer_acquire(load_kv_producer_state)
            if cutlass.const_expr(self.use_page_table):
                cute.copy(
                    qk_params.tma_atom_c_latent,
                    qk_params.tCLgCL[None, 0, i, common_params.mPT[k_index]],
                    qk_params.tKCsKC[None, load_kv_producer_state.index],
                    tma_bar_ptr=tma_bar_ptr,
                )
            else:
                cute.copy(
                    qk_params.tma_atom_c_latent,
                    qk_params.tCLgCL[None, k_index, i, common_params.blk_coord[2]],
                    qk_params.tKCsKC[None, load_kv_producer_state.index],
                    tma_bar_ptr=tma_bar_ptr,
                )
            load_kv_producer_state.advance()

        for i in cutlass.range_constexpr(self.iterations_qk_rope):
            if cutlass.const_expr(load_q):
                tma_bar_ptr = common_params.load_q_pipeline.producer_get_barrier(
                    load_q_producer_state
                )
                common_params.load_q_pipeline.producer_acquire(load_q_producer_state)
                cute.copy(
                    qk_params.tma_atom_q_rope,
                    qk_params.tQRgQR[None, 0, i],
                    qk_params.tQsQ[None, i + self.iterations_qk_latent],
                    tma_bar_ptr=tma_bar_ptr,
                )
                load_q_producer_state.advance()
            tma_bar_ptr = common_params.load_kv_pipeline.producer_get_barrier(
                load_kv_producer_state
            )
            common_params.load_kv_pipeline.producer_acquire(load_kv_producer_state)
            if cutlass.const_expr(self.use_page_table):
                cute.copy(
                    qk_params.tma_atom_c_rope,
                    qk_params.tKRgKR[None, 0, i, common_params.mPT[k_index]],
                    qk_params.tKCsKC[None, load_kv_producer_state.index],
                    tma_bar_ptr=tma_bar_ptr,
                )
            else:
                cute.copy(
                    qk_params.tma_atom_c_rope,
                    qk_params.tKRgKR[None, k_index, i, common_params.blk_coord[2]],
                    qk_params.tKCsKC[None, load_kv_producer_state.index],
                    tma_bar_ptr=tma_bar_ptr,
                )
            load_kv_producer_state.advance()

        kPrefetchDistance = 1
        for i in cutlass.range_constexpr(self.iterations_qk_latent):
            if cutlass.const_expr(self.use_page_table):
                if k_tile_count > kPrefetchDistance:
                    cute.prefetch(
                        qk_params.tma_atom_c_latent,
                        qk_params.tCLgCL[
                            None,
                            k_index,
                            i,
                            common_params.mPT[k_index + kPrefetchDistance],
                        ],
                    )
            else:
                cute.prefetch(
                    qk_params.tma_atom_c_latent,
                    qk_params.tCLgCL[
                        None, k_index + kPrefetchDistance, i, common_params.blk_coord[2]
                    ],
                )

        for i in cutlass.range_constexpr(self.iterations_qk_rope):
            if cutlass.const_expr(self.use_page_table):
                if k_tile_count > kPrefetchDistance:
                    cute.prefetch(
                        qk_params.tma_atom_c_rope,
                        qk_params.tKRgKR[
                            None,
                            k_index,
                            i,
                            common_params.mPT[k_index + kPrefetchDistance],
                        ],
                    )
            else:
                cute.prefetch(
                    qk_params.tma_atom_c_rope,
                    qk_params.tKRgKR[
                        None, k_index + kPrefetchDistance, i, common_params.blk_coord[2]
                    ],
                )
        return load_q_producer_state, load_kv_producer_state

    @cute.jit
    def load_tma_v_one_k_tile(
        self,
        common_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        load_kv_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        for i in cutlass.range_constexpr(self.iterations_pv_k):
            for j in cutlass.range_constexpr(self.iterations_pv_n):
                tma_bar_ptr = common_params.load_kv_pipeline.producer_get_barrier(
                    load_kv_producer_state
                )
                common_params.load_kv_pipeline.producer_acquire(load_kv_producer_state)
                if cutlass.const_expr(self.use_page_table):
                    cute.copy(
                        v_params.tma_atom_c_latent_transpose,
                        v_params.tCLTgCLT[None, j, i, common_params.mPT[k_index]],
                        v_params.tVCsVC[None, load_kv_producer_state.index],
                        tma_bar_ptr=tma_bar_ptr,
                    )
                else:
                    cute.copy(
                        v_params.tma_atom_c_latent_transpose,
                        v_params.tCLTgCLT[
                            None,
                            j,
                            k_index * self.iterations_pv_k + i,
                            common_params.blk_coord[2],
                        ],
                        v_params.tVCsVC[None, load_kv_producer_state.index],
                        tma_bar_ptr=tma_bar_ptr,
                    )
                load_kv_producer_state.advance()
        return load_kv_producer_state
