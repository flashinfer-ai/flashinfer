# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLALoaderRole — TMA load orchestration for MLA decode kernels.

Extracted from the monolithic mla_decode_fp16.py kernel. Owns:
- get_k_tile_count: compute per-CTA tile range from split-KV partitioning
- TMA copy helpers for Q, K (latent/rope), and V loads
- run(): tile-scheduler loop driving the full load warp lifetime

All pipeline acquire/wait/commit/release/tail calls happen directly in run(),
not in sub-methods, because CuTe DSL's MLIR compiler cannot track participant
state mutations across method boundaries inside dynamic loops.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.pipeline import PipelineProducer, PipelineConsumer
from types import SimpleNamespace

from ..mla_config import MLAConfig
from ..scheduler.mla_persistent import (
    create_mla_static_tile_scheduler,
    MLAStaticTileSchedulerParams,
    ceil_div,
)


class MLALoaderRole:
    """Loader warp for MLA decode kernels — TMA loads Q, K, V into SMEM.

    Created from MLAConfig in the kernel's __init__.
    """

    def __init__(self, config: MLAConfig):
        self.config = config
        self.mma_qk_tiler = config.mma_qk_tiler
        self.mma_qk_rope_tiler = config.mma_qk_rope_tiler
        self.mma_pv_tiler = config.mma_pv_tiler
        self.page_size = config.page_size
        self.is_var_split_kv = config.is_var_split_kv
        self.iterations_qk_latent = config.iterations_qk_latent
        self.iterations_qk_rope = config.iterations_qk_rope
        self.iterations_pv_k = config.iterations_pv_k
        self.iterations_pv_n = config.iterations_pv_n

    # =========================================================================
    #  Tile count computation
    # =========================================================================

    @cute.jit
    def _get_k_tile_count(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        blk_coord: cute.Coord,
    ) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        """Get the current k_index, k_tile_count, and local split_kv value.

        :param split_kv: Split_kv value
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: Per-block split_kv values tensor
        :type block_split_kvs: cute.Tensor
        :param blk_coord: Block coordinate
        :type blk_coord: cute.Coord
        :return: k_index, k_tile_count, split_kv
        :rtype: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        """
        K = cache_seqs[blk_coord[2]]
        if cutlass.const_expr(self.is_var_split_kv):
            split_kv = block_split_kvs[blk_coord[2]]

        k_tile_total = cute.ceil_div(K, self.mma_qk_tiler[1])
        k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
        k_index = blk_coord[3] * k_tile_per_cta
        k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)
        return k_index, k_tile_count, split_kv

    # =========================================================================
    #  TMA copy helpers (pure computation — no pipeline ops)
    # =========================================================================

    @cute.jit
    def _read_qk_page_indices(
        self,
        tile_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        page_table_stage: cutlass.Int32,
    ):
        """Read QK page table indices from SMEM into registers."""
        page_per_tile = ceil_div(
            self.mma_qk_tiler[1] // self.page_size,
            qk_params.tiled_mma_qk.thr_id.shape,
        )
        k_idx = cute.make_rmem_tensor(cute.make_layout(page_per_tile), cutlass.Int32)
        for i in cutlass.range_constexpr(page_per_tile):
            k_idx[i] = (
                tile_params.sPT[0, page_table_stage]
                if self.mma_qk_tiler[1] // self.page_size == 1
                else tile_params.sPT[
                    i + tile_params.blk_coord[0] * page_per_tile, page_table_stage
                ]
            )
        return k_idx

    @cute.jit
    def _load_q_tma(self, qk_params: SimpleNamespace, q_barrier):
        """Issue TMA copies for Q latent and Q rope fragments."""
        for i in cutlass.range(self.iterations_qk_latent):
            cute.copy(
                qk_params.tma_atom_q_latent,
                qk_params.tQLgQL[None, 0, i],
                qk_params.tQsQ[None, (i, 0)],
                tma_bar_ptr=q_barrier,
            )
        for i in cutlass.range(self.iterations_qk_rope):
            cute.copy(
                qk_params.tma_atom_q_rope,
                qk_params.tQRgQR[None, 0, i],
                qk_params.tQsQ_rope[None, i],
                tma_bar_ptr=q_barrier,
            )

    @cute.jit
    def _load_kv_latent_one_iter(
        self,
        qk_params: SimpleNamespace,
        k_idx,
        kv_barrier,
        kv_stage_index: cutlass.Int32,
        iteration: int,
    ):
        """Issue TMA copies for one K-latent iteration."""
        page_per_tile = ceil_div(
            self.mma_qk_tiler[1] // self.page_size,
            qk_params.tiled_mma_qk.thr_id.shape,
        )
        for k in cutlass.range(page_per_tile):
            cute.copy(
                qk_params.tma_atom_c_latent,
                qk_params.tCLgCL[None, iteration, k_idx[k]],
                qk_params.tKCsKC[None, k, 0, kv_stage_index],
                tma_bar_ptr=kv_barrier,
            )

    @cute.jit
    def _load_kv_rope_one_iter(
        self,
        qk_params: SimpleNamespace,
        k_idx,
        kv_barrier,
        kv_stage_index: cutlass.Int32,
        iteration: int,
    ):
        """Issue TMA copies for one K-rope iteration."""
        page_per_tile = ceil_div(
            self.mma_qk_tiler[1] // self.page_size,
            qk_params.tiled_mma_qk.thr_id.shape,
        )
        for k in cutlass.range(page_per_tile):
            cute.copy(
                qk_params.tma_atom_c_rope,
                qk_params.tKRgKR[None, iteration, k_idx[k]],
                qk_params.tKCsKC[None, k, 0, kv_stage_index],
                tma_bar_ptr=kv_barrier,
            )

    @cute.jit
    def _read_v_page_indices(
        self,
        tile_params: SimpleNamespace,
        page_table_stage: cutlass.Int32,
    ):
        """Read V page table indices from SMEM into registers."""
        page_per_tile = self.mma_pv_tiler[2] * self.iterations_pv_k // self.page_size
        k_idx = cute.make_rmem_tensor(cute.make_layout(page_per_tile), cutlass.Int32)
        for i in cutlass.range(page_per_tile):
            k_idx[i] = (
                tile_params.sPT[0, page_table_stage]
                if page_per_tile == 1
                else tile_params.sPT[i, page_table_stage]
            )
        return k_idx

    @cute.jit
    def _load_v_one_iter(
        self,
        v_params: SimpleNamespace,
        k_idx,
        kv_barrier,
        kv_stage_index: cutlass.Int32,
        i: int,
        j: int,
    ):
        """Issue TMA copies for one V iteration."""
        page_per_tile = self.mma_pv_tiler[2] * self.iterations_pv_k // self.page_size
        page_per_subtile = ceil_div(page_per_tile, self.iterations_pv_k)
        for k in cutlass.range(page_per_subtile):
            k_idx_i = k_idx[
                k
                + i // ceil_div(self.iterations_pv_k, page_per_tile) * page_per_subtile
            ]
            cute.copy(
                v_params.tma_atom_c_latent_transpose,
                v_params.tCLTgCLT[
                    None,
                    j,
                    i % ceil_div(self.iterations_pv_k, page_per_tile),
                    k_idx_i,
                ],
                v_params.tVCsVC[None, 0, k, kv_stage_index],
                tma_bar_ptr=kv_barrier,
            )

    # =========================================================================
    #  TMA partition setup (from load_tma lines 1621-1738)
    # =========================================================================

    @cute.jit
    def _setup_tma_partitions(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        v_params: SimpleNamespace,
    ):
        """Set up TMA partitions for Q, K, V tensors and store them into params.

        This is the partition logic from the monolithic load_tma method
        (lines 1621-1738), executed once per work tile before the per-k-tile
        load loops.
        """
        # page table
        mPT = common_params.mPT[None, common_params.blk_coord[2]]

        # Flatten divide and partition global tensors for QK TMA load
        # (bM, bK, rM, rK, rL)
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
        gKR = cute.tiled_divide(qk_params.mKR, (page_tile_size, self.mma_qk_tiler[2]))
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

        # tma partition for q, k latent/rope
        # smem: ((atom_v, rest_v), STAGE)
        # gmem: ((atom_v, rest_v), RestM, RestK, RestL)
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

        _, tKRgKR = cpasync.tma_partition(
            qk_params.tma_atom_c_rope,
            0,
            cute.make_layout(1),
            qk_params.sKC,
            tSgKR,
        )

        tQLgQL = tQLgQL_mkl[
            None, None, None, common_params.blk_coord[1], common_params.blk_coord[2]
        ]
        tQRgQR = tQRgQR_mkl[
            None, None, None, common_params.blk_coord[1], common_params.blk_coord[2]
        ]

        # Flatten divide and partition global tensors for V TMA load
        page_tile_size = min(self.page_size, self.mma_pv_tiler[2])
        gCLT = cute.flat_divide(v_params.mCLT, (self.mma_pv_tiler[1], page_tile_size))
        cta_n = self.mma_pv_tiler[1] // v_params.tiled_mma_pv.thr_id.shape
        gCLT = cute.logical_divide(gCLT, (cta_n,))[
            (None, common_params.blk_coord[0]), None, None, None, None
        ]
        tOgCLT = cute.tiled_divide(gCLT, (cta_n, page_tile_size))
        tOgCLT = tOgCLT[None, 0, 0, None, None, None]

        # tma partition for vc
        # smem: ((atom_v, rest_v), STAGE)
        # gmem: ((atom_v, rest_v), RestM, RestK, RestL)
        tVCsVC, tCLTgCLT = cpasync.tma_partition(
            v_params.tma_atom_c_latent_transpose,
            0,
            cute.make_layout(1),
            v_params.sVC,
            tOgCLT,
        )

        # set extra params
        common_params.mPT = mPT
        qk_params.tQLgQL = tQLgQL
        qk_params.tQRgQR = tQRgQR
        qk_params.tCLgCL = tCLgCL
        qk_params.tKRgKR = tKRgKR
        qk_params.tQsQ = tQsQ
        qk_params.tQsQ_rope = tQsQ_rope
        qk_params.tKCsKC = tKCsKC
        v_params.tCLTgCLT = tCLTgCLT
        v_params.tVCsVC = tVCsVC

    # =========================================================================
    #  run() — tile-scheduler loop driving the full load warp lifetime
    #
    #  All pipeline acquire/wait/commit/release/tail calls live here.
    #  Sub-methods only receive handles/indices and do pure TMA copies.
    # =========================================================================

    @cute.jit
    def run(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        v_params: SimpleNamespace,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        load_q_producer: PipelineProducer,
        load_kv_producer: PipelineProducer,
        load_pt_consumer: PipelineConsumer,
        tile_sched_params: MLAStaticTileSchedulerParams,
    ):
        """Tile-scheduler loop that orchestrates all TMA loads for the load warp.

        For each work tile produced by the tile scheduler:
        1. Compute k_index / k_tile_count via _get_k_tile_count
        2. Set up TMA partitions via _setup_tma_partitions
        3. Load first QK tile (with Q load)
        4. Loop remaining tiles: load QK + load V for previous tile
        5. Load final V tile

        After all work tiles are exhausted, calls tail() on both producer
        participants.
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
                    sPT=common_params.sPT,
                )

                self._setup_tma_partitions(tile_params, qk_params, v_params)

                # === First QK tile (with Q load) ===
                pt_handle = load_pt_consumer.wait_and_advance()
                k_idx_qk = self._read_qk_page_indices(
                    tile_params, qk_params, pt_handle.index
                )

                q_handle = load_q_producer.acquire_and_advance()
                self._load_q_tma(qk_params, q_handle.barrier)

                for i in cutlass.range(self.iterations_qk_latent):
                    kv_handle = load_kv_producer.acquire_and_advance()
                    self._load_kv_latent_one_iter(
                        qk_params,
                        k_idx_qk,
                        kv_handle.barrier,
                        kv_handle.index,
                        i,
                    )

                for i in cutlass.range(self.iterations_qk_rope):
                    kv_handle = load_kv_producer.acquire_and_advance()
                    self._load_kv_rope_one_iter(
                        qk_params,
                        k_idx_qk,
                        kv_handle.barrier,
                        kv_handle.index,
                        i,
                    )

                k_index += 1
                k_tile_count -= 1

                while k_tile_count > 0:
                    prev_pt_handle = pt_handle

                    # === Next QK tile (no Q load) ===
                    pt_handle = load_pt_consumer.wait_and_advance()
                    k_idx_qk = self._read_qk_page_indices(
                        tile_params, qk_params, pt_handle.index
                    )

                    for i in cutlass.range(self.iterations_qk_latent):
                        kv_handle = load_kv_producer.acquire_and_advance()
                        self._load_kv_latent_one_iter(
                            qk_params,
                            k_idx_qk,
                            kv_handle.barrier,
                            kv_handle.index,
                            i,
                        )

                    for i in cutlass.range(self.iterations_qk_rope):
                        kv_handle = load_kv_producer.acquire_and_advance()
                        self._load_kv_rope_one_iter(
                            qk_params,
                            k_idx_qk,
                            kv_handle.barrier,
                            kv_handle.index,
                            i,
                        )

                    # === V tile for previous k-tile ===
                    k_idx_v = self._read_v_page_indices(
                        tile_params, prev_pt_handle.index
                    )
                    prev_pt_handle.release()

                    for i in cutlass.range(self.iterations_pv_k):
                        for j in cutlass.range(self.iterations_pv_n):
                            kv_handle = load_kv_producer.acquire_and_advance()
                            self._load_v_one_iter(
                                v_params,
                                k_idx_v,
                                kv_handle.barrier,
                                kv_handle.index,
                                i,
                                j,
                            )

                    k_index += 1
                    k_tile_count -= 1

                # === Last V tile ===
                k_idx_v = self._read_v_page_indices(tile_params, pt_handle.index)
                pt_handle.release()

                for i in cutlass.range(self.iterations_pv_k):
                    for j in cutlass.range(self.iterations_pv_n):
                        kv_handle = load_kv_producer.acquire_and_advance()
                        self._load_v_one_iter(
                            v_params,
                            k_idx_v,
                            kv_handle.barrier,
                            kv_handle.index,
                            i,
                            j,
                        )

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        load_q_producer.tail()
        load_kv_producer.tail()
