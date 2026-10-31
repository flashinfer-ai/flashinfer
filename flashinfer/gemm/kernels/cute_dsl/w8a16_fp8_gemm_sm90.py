# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Structure after the CuTe-DSL Hopper persistent dense GEMM example
# (examples/python/CuTeDSL/hopper/dense_gemm_persistent.py, BSD-3-Clause) and
# CUTLASS's sm90_mma_tma_gmma_rs_warpspecialized_mixed_input mainloop.

"""Hopper (sm_90a) W8A16 GEMM: ``out[m, n] = act[m, K] @ (e4m3 weight[n, K] * scale).T
(+ bias[n])`` with the dequantization in the mainloop.

Operand mapping as the Blackwell kernel: the weight is MMA operand A (kernel M = weight
rows), the activation operand B (kernel N = activation rows), C[n, m] column-major is
``out`` row-major.

Persistent, warp-specialized: a producer warpgroup whose warp 0 TMA-loads the fp8
weight tile and the 16-bit activation tile of every k-tile, and one or two consumer
warpgroups.  A consumer warpgroup pulls its 64 weight rows of a k-block from smem into
registers, converts e4m3 -> bf16/fp16 (``w8a16_fp8_gemm_sm100_utils.convert_e4m3_fragment``)
and issues wgmma with A from registers and B from smem, software-pipelined over the
k-blocks of a stage as CUTLASS's register-sourced mixed-input mainloop does.  The
epilogue applies the per-tensor scale and the per-weight-row bias in fp32, converts and
TMA-stores (tile_m x 32) subtiles.

Tiles: (128, A) with two consumer warpgroups (the C++ cooperative tiles), (64, A) with one
(the C++ ping-pong tiles run as single-warpgroup tiles here).  The tile walk is the C++
grouped order: consecutive work items take the weight tiles of one activation tile;
``group_size`` activation tiles form an L2 group that walks all weight tiles together.

Split-K: the operands are 2-D views over the whole K and a work item's K-slice is a
k-tile offset; the fp32 partials are C[n, splits * m_padded] with m_padded a multiple of
the activation tile, so a slice's last (partial) tile never writes into the next slice
and the TMA store's own clipping handles the real bound.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync, warpgroup

from .w8a16_fp8_gemm_sm100_utils import convert_e4m3_fragment


class Sm90W8A16Fp8GemmKernel:
    def __init__(
        self,
        tile_shape_mn: tuple[int, int],
        has_bias: bool,
        group_size: int = 1,
        partial_output: bool = False,
        enable_pdl: bool = False,
    ):
        # tile_shape_mn: (weight rows, activation rows) per CTA tile; 128 weight rows run
        # on two consumer warpgroups, 64 on one.  Activation rows: a multiple of 8, at
        # most 256 (the wgmma N extent).
        tile_w, tile_a = tile_shape_mn
        if tile_w not in (64, 128):
            raise ValueError(
                f"W8A16 sm90: weight rows per tile must be 64 or 128, not {tile_w}"
            )
        if tile_a % 8 != 0 or not 8 <= tile_a <= 256:
            raise ValueError(
                f"W8A16 sm90: activation rows per tile {tile_a} not a multiple of 8 <= 256"
            )
        if group_size < 1:
            raise ValueError("group_size must be positive")
        self.tile_shape_mnk = (tile_w, tile_a, 64)
        self.has_bias = has_bias
        self.group_size = group_size
        # partial_output: split-K mode; C is fp32 partials and the epilogue applies no
        # scale / bias (the reduce kernel does).
        self.partial_output = partial_output
        self.enable_pdl = enable_pdl
        self.acc_dtype = cutlass.Float32

        self.atom_layout_mnk = (tile_w // 64, 1, 1)
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = tile_w // 64
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (
            self.num_dma_warp_groups + self.num_mma_warp_groups
        ) * self.num_threads_per_warp_group
        self.num_mma_threads = (
            self.num_mma_warp_groups * self.num_threads_per_warp_group
        )
        self.load_warp_id = 0
        self.epi_store_warp_id = self.num_dma_warp_groups * 4
        # Register budgets (the 64K register file): 128 x 40 + 256 x 232 = 64512.
        self.load_register_requirement = 40
        self.mma_register_requirement = 232 if self.num_mma_warp_groups == 2 else 240
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_mma_threads
        )

    # ------------------------------------------------------------- attributes
    def _setup_attributes(self):
        # The wgmma reads A (the converted weight) from registers and B from smem.
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.mma_dtype,
            self.mma_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
            a_source=warpgroup.OperandSource.RMEM,
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        assert self.tile_shape_mnk[2] % mma_inst_shape_k == 0
        self.num_k_blocks = self.tile_shape_mnk[2] // mma_inst_shape_k

        tile_m, tile_a, tile_k = self.tile_shape_mnk
        # Epilogue subtile: all weight rows of the tile x 32 activation rows.
        self.epi_tile = (tile_m, 32)
        self.epi_stage = 2
        self.ab_stage = self._compute_stages()

        # Weight (fp8, K-major): 64 bytes per row per stage -> the 64-byte swizzle atom;
        # activation (16-bit, K-major): 128 bytes per row -> 128-byte swizzle.
        a_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(self.a_layout, self.a_dtype, tile_k),
            self.a_dtype,
        )
        self.a_smem_layout_staged = cute.tile_to_shape(
            a_atom, (tile_m, tile_k, self.ab_stage), order=(0, 1, 2)
        )
        b_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(self.b_layout, self.b_dtype, tile_k),
            self.b_dtype,
        )
        self.b_smem_layout_staged = cute.tile_to_shape(
            b_atom, (tile_a, tile_k, self.ab_stage), order=(0, 1, 2)
        )
        # C is M-major (weight rows contiguous).
        c_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                self.c_layout, self.c_dtype, self.epi_tile[0]
            ),
            self.c_dtype,
        )
        self.epi_smem_layout_staged = cute.tile_to_shape(
            c_atom, (*self.epi_tile, self.epi_stage), order=(1, 0, 2)
        )

    def _compute_stages(self) -> int:
        tile_m, tile_a, tile_k = self.tile_shape_mnk
        ab_bytes = (
            tile_m * tile_k * self.a_dtype.width + tile_a * tile_k * self.b_dtype.width
        ) // 8
        epi_bytes = (
            self.epi_tile[0]
            * self.epi_tile[1]
            * self.c_dtype.width
            // 8
            * self.epi_stage
        )
        bias_bytes = 2 * 4 * tile_m
        mbar_bytes = 1024
        stages = (self.smem_capacity - epi_bytes - bias_bytes - mbar_bytes) // ab_bytes
        if stages < 2:
            raise ValueError(
                f"W8A16 sm90: tile {self.tile_shape_mnk} leaves fewer than 2 stages"
            )
        return stages

    # ------------------------------------------------------------------ entry
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        c: cute.Tensor,
        act_rows: cutlass.Int32,
        num_persistent: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.mma_dtype = self.b_dtype
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self._setup_attributes()

        tile_m, tile_a, tile_k = self.tile_shape_mnk
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), a, a_smem_layout, (tile_m, tile_k)
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), b, b_smem_layout, (tile_a, tile_k)
        )
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )
        self.tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            # The bias of the tile's weight rows, fp32, double-buffered by tile parity.
            bias_tile: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 2 * tile_m], 16
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            scale,
            bias,
            act_rows,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ).launch(
            grid=[num_persistent, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        act_rows: cutlass.Int32,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        tile_m, tile_a, tile_k = self.tile_shape_mnk
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdx, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # One pipeline for both operands: warp 0 of the producer warpgroup issues the two
        # TMA loads of a stage against one transaction barrier; every consumer warp
        # arrives on the empty barrier.
        ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warp_groups * 4
        )
        ab_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.ab_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=ab_producer_group,
            consumer_group=ab_consumer_group,
            tx_count=self.tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        pipeline.pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sBias = storage.bias_tile.get_tensor(
            cute.make_layout((tile_m, 2), stride=(1, tile_m))
        )

        # (bM, bK, RestM, RestK, L) / (bN, bK, RestN, RestK, L) / (bM, bN, RestM, RestN, L)
        gA_mkl = cute.local_tile(mA_mkl, (tile_m, tile_k), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, (tile_a, tile_k), (None, None, None))
        gC_mnl = cute.local_tile(mC_mnl, (tile_m, tile_a), (None, None, None))

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        # Work items: (weight tile, activation tile, K-slice); the grouped walk order
        # of _tile_at.  Every warp computes the same sequence: bidx, bidx + grid, ...
        weight_rows = cutlass.Int32(cute.size(mA_mkl.shape[0]))
        weight_tiles = cutlass.Int32(cute.ceil_div(weight_rows, tile_m))
        act_tiles = cutlass.Int32(cute.ceil_div(act_rows, tile_a))
        k_tiles_total = cutlass.Int32(cute.size(gA_mkl, mode=[3]))
        num_slices = cutlass.Int32(cute.size(mC_mnl.shape[1]) // (act_tiles * tile_a))
        if cutlass.const_expr(not self.partial_output):
            num_slices = cutlass.Int32(1)
        k_tile_cnt = k_tiles_total // num_slices
        total_work = weight_tiles * act_tiles * num_slices
        group = cutlass.min(cutlass.Int32(self.group_size), act_tiles)

        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        pipeline.pipeline_init_wait(cluster_shape_mn=(1, 1))

        # ---- producer warpgroup: warp 0 loads both operands of every k-tile.
        if warp_group_idx < self.num_dma_warp_groups:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            if warp_idx == self.load_warp_id:
                producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                lin = cutlass.Int32(bidx)
                while lin < total_work:
                    m_tile, n_tile, tile_l = self._tile_at(
                        lin, weight_tiles, act_tiles, group
                    )
                    k_base = tile_l * k_tile_cnt
                    for k in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        ab_pipeline.producer_acquire(producer_state)
                        cute.copy(
                            tma_atom_a,
                            tAgA[(None, m_tile, k_base + k, 0)],
                            tAsA[(None, producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                producer_state
                            ),
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, n_tile, k_base + k, 0)],
                            tBsB[(None, producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                producer_state
                            ),
                        )
                        ab_pipeline.producer_commit(producer_state)
                        producer_state.advance()
                    lin += gdx
                ab_pipeline.producer_tail(producer_state)

        # ---- consumer warpgroups: register-sourced wgmma mainloop, fused epilogue.
        if warp_group_idx >= self.num_dma_warp_groups:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            ctid = tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
            mma_wg_layout = cute.make_layout(
                self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
            )
            thr_mma = tiled_mma.get_slice(
                mma_wg_layout(warp_group_idx - self.num_dma_warp_groups)
            )

            tCsA = thr_mma.partition_A(sA)  # (MMA, MMA_M, MMA_K, STAGE) smem
            tCsB = thr_mma.partition_B(sB)
            tCrB = thr_mma.make_fragment_B(tCsB)  # smem descriptors
            # A in registers: the converted (16-bit) fragment the wgmma reads, and the
            # fp8 fragment of the same shape the smem copy fills.
            tCrA_mma = thr_mma.make_fragment_A(tCsA[(None, None, None, 0)])
            tCrA_load = cute.make_fragment_like(tCrA_mma, self.a_dtype)
            smem_copy_atom_a = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.a_dtype, num_bits_per_copy=16
            )
            smem_tiled_copy_a = cute.make_tiled_copy_A(smem_copy_atom_a, tiled_mma)
            smem_thr_copy_a = smem_tiled_copy_a.get_slice(ctid)
            tCsA_copy = smem_thr_copy_a.partition_S(sA)  # (CPY, CPY_M, CPY_K, STAGE)
            tCrA_copy_view = smem_thr_copy_a.retile(tCrA_load)  # (CPY, CPY_M, CPY_K)
            num_k_blocks = cute.size(tCrA_mma, mode=[2])
            k_wait_max = min(num_k_blocks - 1, 7)

            # The MMA slice is per warpgroup (wgmma operands and accumulator layout are
            # warpgroup-uniform), so partition_C of the tile's identity gives each element
            # the warpgroup's row base (+ the fragment's +8 row component); the thread's
            # own rows of the m64 accumulator are warp * 16 + lane / 4 (+ 8).
            tile_coords = cute.make_identity_tensor((tile_m, tile_a))
            tCcAcc = thr_mma.partition_C(tile_coords)
            thread_row = ((ctid % 128) // 32) * 16 + (ctid % 32) // 4
            accumulators = cute.make_rmem_tensor(tCcAcc.shape, self.acc_dtype)

            # Epilogue copies: registers -> smem (stmatrix for 16-bit C, plain stores for
            # fp32 partials), smem -> global by TMA.
            if cutlass.const_expr(self.c_dtype.width == 16):
                copy_atom_r2s = sm90_utils.get_smem_store_op(
                    self.c_layout, elem_ty_d=self.c_dtype, elem_ty_acc=self.acc_dtype
                )
                copy_atom_c = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
                    self.c_dtype,
                )
            else:
                copy_atom_r2s = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), self.c_dtype, num_bits_per_copy=32
                )
                copy_atom_c = copy_atom_r2s
            tiled_copy_c_atom = cute.make_tiled_copy_C_atom(copy_atom_c, tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_c_atom)
            thr_copy_r2s = tiled_copy_r2s.get_slice(ctid)
            tRS_sD = thr_copy_r2s.partition_D(sC)  # (R2S, R2S_M, R2S_N, EPI_STAGE)
            tRS_rAcc = tiled_copy_r2s.retile(
                accumulators
            )  # (R2S, R2S_M, R2S_N) full tile
            tRS_cAcc = tiled_copy_r2s.retile(tCcAcc)
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD = cute.make_rmem_tensor(rD_shape[:3], self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(rD_shape[:3], self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)
            m_subs = tile_m // self.epi_tile[0]
            n_subs = tile_a // self.epi_tile[1]
            # Subtile s of the retiled accumulator is its flat chunk [s * size, (s + 1) *
            # size) (n fast): the n8 blocks are the fragment's slowest mode.  For the
            # 16-bit stmatrix copy the chunk's order is the smem fragment's order; for
            # the 32-bit plain copy the fragment is (atom, rows 2, (col pair 2, n8 blocks))
            # while the accumulator chunk runs col pair, row, n8 block, so the partials are
            # written through that permutation.
            if cutlass.const_expr(self.c_dtype.width != 16):
                assert cute.size(rD_shape[0]) == 1 and cute.size(rD_shape[1]) == 2, (
                    rD_shape
                )
                assert cute.size(rD_shape[2]) * 2 == size_tRS_rD, rD_shape
                n8_blocks = cute.size(rD_shape[2]) // 2

            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.num_mma_threads
                ),
            )

            read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            tiles_done = cutlass.Int32(0)
            lin = cutlass.Int32(bidx)
            while lin < total_work:
                m_tile, n_tile, tile_l = self._tile_at(
                    lin, weight_tiles, act_tiles, group
                )

                # ---- mainloop
                accumulators.fill(0.0)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                read_state.reset_count()
                release_state.reset_count()

                # First k-tile: no wgmma in flight, so no wait before the converts.
                ab_pipeline.consumer_wait(read_state)
                self._load_a_kblock(
                    smem_tiled_copy_a, tCsA_copy, tCrA_copy_view, 0, read_state.index
                )
                if cutlass.const_expr(num_k_blocks > 1):
                    self._load_a_kblock(
                        smem_tiled_copy_a,
                        tCsA_copy,
                        tCrA_copy_view,
                        1,
                        read_state.index,
                    )
                self._convert_a_kblock(tCrA_load, tCrA_mma, 0)
                warpgroup.fence()
                for kb in cutlass.range_constexpr(num_k_blocks):
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        tCrA_mma[(None, None, kb)],
                        tCrB[(None, None, kb, read_state.index)],
                        accumulators,
                    )
                    warpgroup.commit_group()
                    if cutlass.const_expr(kb + 2 < num_k_blocks):
                        self._load_a_kblock(
                            smem_tiled_copy_a,
                            tCsA_copy,
                            tCrA_copy_view,
                            kb + 2,
                            read_state.index,
                        )
                    if cutlass.const_expr(kb + 1 < num_k_blocks):
                        self._convert_a_kblock(tCrA_load, tCrA_mma, kb + 1)
                warpgroup.fence()
                read_state.advance()

                # Steady state: wgmma of k-block kb issued, at most k_wait_max groups
                # pending, then the previous stage is released at the last k-block (all
                # its wgmma are done) and the registers of k-block kb + 1 are refilled
                # (their reader, k-block kb + 1 of the previous k-tile, is done).
                for _k_tile in cutlass.range(1, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.consumer_wait(read_state)
                    self._load_a_kblock(
                        smem_tiled_copy_a,
                        tCsA_copy,
                        tCrA_copy_view,
                        0,
                        read_state.index,
                    )
                    if cutlass.const_expr(num_k_blocks > 1):
                        self._load_a_kblock(
                            smem_tiled_copy_a,
                            tCsA_copy,
                            tCrA_copy_view,
                            1,
                            read_state.index,
                        )
                    warpgroup.wait_group(k_wait_max)
                    self._convert_a_kblock(tCrA_load, tCrA_mma, 0)
                    warpgroup.fence()
                    for kb in cutlass.range_constexpr(num_k_blocks):
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA_mma[(None, None, kb)],
                            tCrB[(None, None, kb, read_state.index)],
                            accumulators,
                        )
                        warpgroup.commit_group()
                        warpgroup.wait_group(k_wait_max)
                        if cutlass.const_expr(kb == num_k_blocks - 1):
                            ab_pipeline.consumer_release(release_state)
                            release_state.advance()
                        if cutlass.const_expr(kb + 2 < num_k_blocks):
                            self._load_a_kblock(
                                smem_tiled_copy_a,
                                tCsA_copy,
                                tCrA_copy_view,
                                kb + 2,
                                read_state.index,
                            )
                        if cutlass.const_expr(kb + 1 < num_k_blocks):
                            self._convert_a_kblock(tCrA_load, tCrA_mma, kb + 1)
                    warpgroup.fence()
                    read_state.advance()
                warpgroup.wait_group(0)
                ab_pipeline.consumer_release(release_state)
                release_state.advance()

                # ---- epilogue: D = scale * acc (+ bias[row]) for the output, acc for
                # split-K partials; TMA store per (tile_m x 32) subtile.
                scale_val = cutlass.Float32(scale[0])
                bias_buf = tiles_done % 2
                if cutlass.const_expr(self.has_bias and not self.partial_output):
                    if ctid < tile_m:
                        bias_row = m_tile * tile_m + ctid
                        bias_val = cutlass.Float32(0.0)
                        if bias_row < weight_rows:
                            bias_val = cutlass.Float32(bias[bias_row])
                        sBias[(ctid, bias_buf)] = bias_val
                    self.epilog_sync_barrier.arrive_and_wait()

                gC_tile = gC_mnl[(None, None, m_tile, n_tile + tile_l * act_tiles, 0)]
                tCgC_epi = cute.zipped_divide(gC_tile, self.epi_tile)
                bSG_sD, bSG_gD = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    tCgC_epi,
                )
                num_prev_subtiles = tiles_done * (m_subs * n_subs)
                for m_sub in cutlass.range_constexpr(m_subs):
                    for n_sub in cutlass.range_constexpr(n_subs):
                        base = (m_sub * n_subs + n_sub) * size_tRS_rD
                        if cutlass.const_expr(self.partial_output):
                            for b in cutlass.range_constexpr(n8_blocks):
                                for r in cutlass.range_constexpr(2):
                                    for c in cutlass.range_constexpr(2):
                                        tRS_rD[(0, r, c + 2 * b)] = tRS_rAcc[
                                            base + c + 2 * r + 4 * b
                                        ]
                        elif cutlass.const_expr(self.has_bias):
                            for v in cutlass.range_constexpr(size_tRS_rD):
                                tRS_rD[v] = (
                                    scale_val * tRS_rAcc[base + v]
                                    + sBias[
                                        (tRS_cAcc[base + v][0] + thread_row, bias_buf)
                                    ]
                                )
                        else:
                            for v in cutlass.range_constexpr(size_tRS_rD):
                                tRS_rD[v] = scale_val * tRS_rAcc[base + v]
                        tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))
                        epi_idx = m_sub * n_subs + n_sub
                        epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epi_store_warp_id:
                            cute.copy(
                                tma_atom_c,
                                bSG_sD[(None, epi_buffer)],
                                bSG_gD[(None, (m_sub, n_sub))],
                            )
                            tma_store_pipeline.producer_commit()
                            tma_store_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                tiles_done += 1
                lin += gdx
            tma_store_pipeline.producer_tail()

    # --------------------------------------------------------------- helpers
    @cute.jit
    def _tile_at(
        self,
        lin: cutlass.Int32,
        weight_tiles: cutlass.Int32,
        act_tiles: cutlass.Int32,
        group: cutlass.Int32,
    ):
        """Work item ``lin`` -> (weight tile, activation tile, K-slice).  The K-slice is
        the slowest index; within a slice the C++ grouped order: ``group`` activation
        tiles form a group whose weight tiles are walked together (weight tile fast
        within the group), the last group narrower when the count does not divide."""
        tiles_per_slice = weight_tiles * act_tiles
        tile_l = lin // tiles_per_slice
        t = lin % tiles_per_slice
        full_groups = act_tiles // group
        group_tiles = group * weight_tiles
        full_tiles = full_groups * group_tiles
        m_tile = cutlass.Int32(0)
        n_tile = cutlass.Int32(0)
        if t < full_tiles:
            group_idx = t // group_tiles
            tig = t % group_tiles
            m_tile = tig // group
            n_tile = group_idx * group + tig % group
        else:
            rem = act_tiles - full_groups * group
            t2 = t - full_tiles
            m_tile = t2 // rem
            n_tile = full_groups * group + t2 % rem
        return m_tile, n_tile, tile_l

    @cute.jit
    def _load_a_kblock(
        self,
        smem_tiled_copy_a: cute.TiledCopy,
        tCsA_copy: cute.Tensor,
        tCrA_copy_view: cute.Tensor,
        kb: cutlass.Constexpr,
        stage: cutlass.Int32,
    ):
        cute.copy(
            smem_tiled_copy_a,
            tCsA_copy[(None, None, kb, stage)],
            tCrA_copy_view[(None, None, kb)],
        )

    @cute.jit
    def _convert_a_kblock(
        self,
        tCrA_load: cute.Tensor,
        tCrA_mma: cute.Tensor,
        kb: cutlass.Constexpr,
    ):
        packed = cute.recast_tensor(tCrA_load[(None, None, kb)], cutlass.Uint8).load()
        tCrA_mma[(None, None, kb)].store(
            convert_e4m3_fragment(packed, self.mma_dtype, native_cvt=False)
        )

    # ----------------------------------------------------------------- launch
    @cute.jit
    def wrapper(
        self,
        weight_ptr: cute.Pointer,
        activation_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        n: cutlass.Int64,
        m: cutlass.Int64,
        k: cutlass.Int64,
        c_cols: cutlass.Int64,
        num_persistent: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Weight [n, k] fp8 and activation [m, k] 16-bit, K-major, over the whole K.
        Out [m, n] row-major is C[n, m, 1]; the split-K partials are
        C[n, splits * m_padded, 1] (``c_cols``), slice ``l`` being activation columns
        ``l * m_padded`` on (m_padded = m rounded up to the activation tile)."""
        weight = cute.make_tensor(
            weight_ptr, cute.make_ordered_layout((n, k, 1), order=(1, 0, 2))
        )
        activation = cute.make_tensor(
            activation_ptr, cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))
        )
        output = cute.make_tensor(
            output_ptr, cute.make_ordered_layout((n, c_cols, 1), order=(0, 1, 2))
        )
        scale = cute.make_tensor(scale_ptr, cute.make_layout((1,)))
        bias = cute.make_tensor(bias_ptr, cute.make_layout((n,)))
        return self(
            weight,
            activation,
            scale,
            bias,
            output,
            cutlass.Int32(m),
            num_persistent=num_persistent,
            stream=stream,
        )


__all__ = ["Sm90W8A16Fp8GemmKernel"]
