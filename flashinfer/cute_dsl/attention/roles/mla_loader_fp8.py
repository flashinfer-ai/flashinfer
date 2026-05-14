# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""FP8 MLA Loader Roles — split K and V TMA loading for FP8 MLA decode.

FP8 replaces the unified load_kv + load_pt pipeline architecture with two
separate TMA loader warps:
- MLALoaderKRole: loads Q (latent+rope) and K (latent+rope) into SMEM.
  Page table indices are read directly from global memory (no SMEM staging).
- MLALoaderVRole: loads V (c_latent_transpose) into SMEM.
  Also reads page table from global memory directly.

This eliminates the page-table pipeline and dedicated PT loader warp.

All pipeline acquire/commit/tail calls happen directly in run().
Sub-methods only receive handles/indices and do pure TMA copies.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.pipeline import PipelineProducer
from types import SimpleNamespace

from ..mla_config import MLAConfig
from ..scheduler.mla_persistent import (
    create_mla_static_tile_scheduler,
    MLAStaticTileSchedulerParams,
    ceil_div,
)


class MLAFP8LoaderKRole:
    """Q/K loader warp for FP8 MLA decode.

    Loads Q latent/rope (once per work tile) and K latent/rope (per k-tile)
    into separate SMEM buffers. Page table indices are read directly from
    global memory — no CpAsync PT pipeline needed.
    """

    def __init__(self, config: MLAConfig):
        self.config = config
        self.mma_qk_tiler = config.mma_qk_tiler
        self.mma_qk_rope_tiler = config.mma_qk_rope_tiler
        self.page_size = config.page_size
        self.is_var_split_kv = config.is_var_split_kv
        self.iterations_qk_latent = config.iterations_qk_latent
        self.iterations_qk_rope = config.iterations_qk_rope

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
    def _setup_tma_partitions(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
    ):
        """Set up TMA partitions for Q latent/rope and K latent/rope.

        K-rope goes to separate sKC_rope SMEM (unlike FP16 which shares sKC).
        """
        mPT = common_params.mPT[None, common_params.blk_coord[2]]

        mma_qk_tiler_mk = cute.select(self.mma_qk_tiler, mode=[0, 2])
        gQL = cute.flat_divide(qk_params.mQL, mma_qk_tiler_mk)
        mma_qk_tiler_mk_rope = cute.select(self.mma_qk_rope_tiler, mode=[0, 2])
        gQR = cute.flat_divide(qk_params.mQR, mma_qk_tiler_mk_rope)

        thr_mma_qk = qk_params.tiled_mma_qk.get_slice(
            common_params.blk_coord[0] % cute.size(qk_params.tiled_mma_qk.thr_id)
        )
        tSgQL = thr_mma_qk.partition_A(gQL)
        tSgQR = thr_mma_qk.partition_A(gQR)

        cta_m = min(
            qk_params.tiled_mma_qk.op.shape_mnk[0]
            // qk_params.tiled_mma_qk.thr_id.shape,
            self.page_size,
        )
        page_tile_size = min(self.page_size, cta_m)
        gCL = cute.tiled_divide(qk_params.mCL, (page_tile_size, self.mma_qk_tiler[2]))
        tSgCL = (
            gCL[
                None,
                common_params.blk_coord[0] % qk_params.tiled_mma_qk.thr_id.shape,
                None,
                None,
            ]
            if cta_m < self.page_size
            else gCL[None, 0, None, None]
        )
        gKR = cute.tiled_divide(
            qk_params.mKR, (page_tile_size, self.mma_qk_rope_tiler[2])
        )
        tSgKR = (
            gKR[
                None,
                common_params.blk_coord[0] % qk_params.tiled_mma_qk.thr_id.shape,
                None,
                None,
            ]
            if cta_m < self.page_size
            else gKR[None, 0, None, None]
        )

        tQsQ, tQLgQL_mkl = cpasync.tma_partition(
            qk_params.tma_atom_q_latent,
            0,
            cute.make_layout(1),
            cute.group_modes(qk_params.sQ, 0, 3),
            cute.group_modes(tSgQL, 0, 3),
        )
        tQsQ_rope, tQRgQR_mkl = cpasync.tma_partition(
            qk_params.tma_atom_q_rope,
            0,
            cute.make_layout(1),
            cute.group_modes(qk_params.sQ_rope, 0, 3),
            cute.group_modes(tSgQR, 0, 3),
        )
        tKCsKC, tCLgCL = cpasync.tma_partition(
            qk_params.tma_atom_c_latent,
            0,
            cute.make_layout(1),
            qk_params.sKC,
            tSgCL,
        )
        tKCsKC_rope, tKRgKR = cpasync.tma_partition(
            qk_params.tma_atom_c_rope,
            0,
            cute.make_layout(1),
            qk_params.sKC_rope,
            tSgKR,
        )

        tQLgQL = tQLgQL_mkl[
            None, None, None, common_params.blk_coord[1], common_params.blk_coord[2]
        ]
        tQRgQR = tQRgQR_mkl[
            None, None, None, common_params.blk_coord[1], common_params.blk_coord[2]
        ]

        common_params.mPT = mPT
        qk_params.tQLgQL = tQLgQL
        qk_params.tQRgQR = tQRgQR
        qk_params.tCLgCL = tCLgCL
        qk_params.tKRgKR = tKRgKR
        qk_params.tQsQ = tQsQ
        qk_params.tQsQ_rope = tQsQ_rope
        qk_params.tKCsKC = tKCsKC
        qk_params.tKCsKC_rope = tKCsKC_rope

    @cute.jit
    def _load_q_tma(self, qk_params: SimpleNamespace, q_barrier):
        for i in cutlass.range_constexpr(self.iterations_qk_latent):
            cute.copy(
                qk_params.tma_atom_q_latent,
                qk_params.tQLgQL[None, 0, i],
                qk_params.tQsQ[None, (i, 0)],
                tma_bar_ptr=q_barrier,
            )
        for i in cutlass.range_constexpr(self.iterations_qk_rope):
            cute.copy(
                qk_params.tma_atom_q_rope,
                qk_params.tQRgQR[None, 0, i],
                qk_params.tQsQ_rope[None, i],
                tma_bar_ptr=q_barrier,
            )

    @cute.jit
    def _read_page_indices(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        k_index: cutlass.Int32,
    ):
        """Read page table indices directly from global memory."""
        page_per_tile = ceil_div(
            self.mma_qk_tiler[1] // self.page_size,
            qk_params.tiled_mma_qk.thr_id.shape,
        )
        k_idx = cute.make_rmem_tensor(cute.make_layout(page_per_tile), cutlass.Int32)
        for i in cutlass.range_constexpr(page_per_tile):
            k_idx[i] = (
                common_params.mPT[k_index]
                if self.mma_qk_tiler[1] // self.page_size == 1
                else common_params.mPT[
                    (
                        k_index * qk_params.tiled_mma_qk.thr_id.shape
                        + common_params.blk_coord[0]
                    )
                    * page_per_tile
                    + i
                ]
            )
        return k_idx

    @cute.jit
    def _load_k_one_tile(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_barrier,
        k_stage_index: cutlass.Int32,
    ):
        """Load one k-tile of K latent and K rope into SMEM."""
        page_per_tile = ceil_div(
            self.mma_qk_tiler[1] // self.page_size,
            qk_params.tiled_mma_qk.thr_id.shape,
        )
        k_idx = self._read_page_indices(common_params, qk_params, k_index)

        for i in range(self.iterations_qk_latent):
            for k in range(page_per_tile):
                cute.copy(
                    qk_params.tma_atom_c_latent,
                    qk_params.tCLgCL[None, i, k_idx[k]],
                    qk_params.tKCsKC[None, k, 0, (i, k_stage_index)],
                    tma_bar_ptr=k_barrier,
                )

        for i in cutlass.range_constexpr(self.iterations_qk_rope):
            for k in cutlass.range_constexpr(page_per_tile):
                cute.copy(
                    qk_params.tma_atom_c_rope,
                    qk_params.tKRgKR[None, i, k_idx[k]],
                    qk_params.tKCsKC_rope[None, k, 0, k_stage_index],
                    tma_bar_ptr=k_barrier,
                )

    @cute.jit
    def run(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        load_q_producer: PipelineProducer,
        load_k_producer: PipelineProducer,
        tile_sched_params: MLAStaticTileSchedulerParams,
    ):
        """Tile-scheduler loop for Q/K loader warp.

        For each work tile:
        1. Set up TMA partitions (creates fresh tile_params to avoid
           mutating common_params inside dynamic if)
        2. First tile: load Q + K
        3. Remaining tiles: load K only
        """
        tile_sched = create_mla_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        while work_tile.is_valid_tile:
            blk_coord = work_tile.tile_idx
            k_index, k_tile_count, local_split_kv = self._get_k_tile_count(
                split_kv,
                cache_seqs,
                block_split_kvs,
                blk_coord,
            )
            if k_tile_count > 0:
                tile_params = SimpleNamespace(
                    blk_coord=blk_coord,
                    local_split_kv=local_split_kv,
                    mPT=common_params.mPT,
                )
                self._setup_tma_partitions(tile_params, qk_params)

                k_tile_count_init = k_tile_count
                while k_tile_count > 0:
                    load_q = k_tile_count_init == k_tile_count
                    if load_q:
                        q_handle = load_q_producer.acquire_and_advance()
                        self._load_q_tma(qk_params, q_handle.barrier)

                    k_handle = load_k_producer.acquire_and_advance()
                    self._load_k_one_tile(
                        tile_params,
                        qk_params,
                        k_index,
                        k_handle.barrier,
                        k_handle.index,
                    )

                    k_index += 1
                    k_tile_count -= 1

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        load_q_producer.tail()
        load_k_producer.tail()


class MLAFP8LoaderVRole:
    """V loader warp for FP8 MLA decode.

    Loads V (c_latent_transpose) into separate SMEM buffer. Page table
    indices are read directly from global memory.
    """

    def __init__(self, config: MLAConfig):
        self.config = config
        self.mma_qk_tiler = config.mma_qk_tiler
        self.mma_pv_tiler = config.mma_pv_tiler
        self.page_size = config.page_size
        self.is_var_split_kv = config.is_var_split_kv
        self.iterations_pv_k = config.iterations_pv_k
        self.iterations_pv_n = config.iterations_pv_n

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
    def _setup_tma_partitions(
        self,
        common_params: SimpleNamespace,
        v_params: SimpleNamespace,
    ):
        """Set up TMA partitions for V (c_latent_transpose)."""
        mPT = common_params.mPT[None, common_params.blk_coord[2]]

        page_tile_size = min(self.page_size, self.mma_pv_tiler[2])
        gCLT = cute.flat_divide(v_params.mCLT, (self.mma_pv_tiler[1], page_tile_size))
        cta_n = self.mma_pv_tiler[1] // v_params.tiled_mma_pv.thr_id.shape
        gCLT = cute.logical_divide(gCLT, (cta_n,))[
            (None, common_params.blk_coord[0]), None, None, None, None
        ]
        tOgCLT = cute.tiled_divide(gCLT, (cta_n, page_tile_size))
        tOgCLT = tOgCLT[None, 0, 0, None, None, None]

        tVCsVC, tCLTgCLT = cpasync.tma_partition(
            v_params.tma_atom_c_latent_transpose,
            0,
            cute.make_layout(1),
            v_params.sVC,
            tOgCLT,
        )

        common_params.mPT = mPT
        v_params.tCLTgCLT = tCLTgCLT
        v_params.tVCsVC = tVCsVC

    @cute.jit
    def _load_v_one_tile(
        self,
        common_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        v_barrier,
        v_stage_index: cutlass.Int32,
    ):
        """Load one k-tile of V into SMEM."""
        page_per_tile = self.mma_pv_tiler[2] * self.iterations_pv_k // self.page_size
        page_per_subtile = ceil_div(page_per_tile, self.iterations_pv_k)
        k_idx = cute.make_rmem_tensor(cute.make_layout(page_per_tile), cutlass.Int32)
        for i in cutlass.range_constexpr(page_per_tile):
            k_idx[i] = (
                common_params.mPT[k_index]
                if page_per_tile == 1
                else common_params.mPT[k_index * page_per_tile + i]
            )

        for j in cutlass.range_constexpr(self.iterations_pv_n):
            for i in cutlass.range_constexpr(self.iterations_pv_k):
                if cutlass.const_expr(page_per_tile > 1):
                    for k in cutlass.range_constexpr(page_per_subtile):
                        k_idx_i = k_idx[k + i * page_per_subtile]
                        cute.copy(
                            v_params.tma_atom_c_latent_transpose,
                            v_params.tCLTgCLT[None, j, 0, k_idx_i],
                            v_params.tVCsVC[None, 0, k, ((j, i), v_stage_index)],
                            tma_bar_ptr=v_barrier,
                        )
                else:
                    cute.copy(
                        v_params.tma_atom_c_latent_transpose,
                        v_params.tCLTgCLT[None, j, i, k_idx[0]],
                        v_params.tVCsVC[None, 0, 0, ((j, i), v_stage_index)],
                        tma_bar_ptr=v_barrier,
                    )

    @cute.jit
    def run(
        self,
        common_params: SimpleNamespace,
        v_params: SimpleNamespace,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        load_v_producer: PipelineProducer,
        tile_sched_params: MLAStaticTileSchedulerParams,
    ):
        """Tile-scheduler loop for V loader warp."""
        tile_sched = create_mla_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        while work_tile.is_valid_tile:
            blk_coord = work_tile.tile_idx
            k_index, k_tile_count, local_split_kv = self._get_k_tile_count(
                split_kv,
                cache_seqs,
                block_split_kvs,
                blk_coord,
            )
            if k_tile_count > 0:
                tile_params = SimpleNamespace(
                    blk_coord=blk_coord,
                    local_split_kv=local_split_kv,
                    mPT=common_params.mPT,
                )
                self._setup_tma_partitions(tile_params, v_params)

                while k_tile_count > 0:
                    v_handle = load_v_producer.acquire_and_advance()
                    self._load_v_one_tile(
                        tile_params,
                        v_params,
                        k_index,
                        v_handle.barrier,
                        v_handle.index,
                    )
                    k_index += 1
                    k_tile_count -= 1

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        load_v_producer.tail()
