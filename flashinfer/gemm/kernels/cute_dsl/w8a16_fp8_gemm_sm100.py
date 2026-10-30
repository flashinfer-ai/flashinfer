# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""SM100 dense W8A16 GEMM: e4m3 weight x bf16 / fp16 activation, in CuTe-DSL.

    out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N])

The weight is the MMA's A operand (kernel M = N, the weight rows), the activation
its B operand (kernel N = M, the activation rows), so C[N, M] column-major is
``out`` row-major.  tcgen05.mma reads A from smem or TMEM, so the e4m3 -> 16-bit
conversion cannot ride in registers in front of the MMA: two transform warpgroups
convert the TMA-landed weight tile in registers
(``w8a16_fp8_gemm_sm100_utils.convert_e4m3_fragment``) and store the 16-bit copy to
TMEM, or to smem for the tile whose two accumulator stages take all of TMEM
(``a_in_smem``).  The MMA warp issues tcgen05.mma with A from there and B from smem.
The epilogue applies the per-tensor scale (fp32, read on device) and the optional
per-N bias in fp32 and stores through TMA (SIMT for a partial last activation tile).

Warp roles (16 warps): 0-3 epilogue, 4 MMA, 5 scheduler, 6 weight TMA, 7 activation
TMA, 8-15 transform.  Same structure as the C++ kernel in
``include/flashinfer/gemm/w8a16_sm100/`` and as FlashInfer's dense W4A16 kernel
(``dense_gemm_bf16_fp4_sm100.py``), from which this file is derived.
"""

from math import ceil, log2
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.mixed_input_helpers as mixed_input_utils
from cutlass.utils.mixed_input_helpers import TransformMode
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait

from .w8a16_fp8_gemm_sm100_utils import convert_e4m3_fragment


class Sm100W8A16Fp8GemmKernel:
    """Warp-specialized persistent dense W8A16 GEMM for SM100 / SM103."""

    def __init__(
        self,
        mma_tiler_mnk: tuple[int, int, int],
        use_2cta_instrs: bool,
        cluster_shape_mn: tuple[int, int],
        a_in_smem: bool,
        has_bias: bool,
        swizzle_size: int,
        raster_along_m: bool,
        transform_fragment_size: int,
        enable_pdl: bool,
        debug: bool = False,
    ):
        # debug: every warp role writes its tile count and a wait-site marker into the
        # `dbg` buffer (16 Int32 per CTA: slots 0-5 tile counts of scheduler, activation
        # TMA, weight TMA, transform, MMA, epilogue; slots 8-13 their markers).  Give it
        # pinned host memory and poll it to see where a hang sits.
        self.debug = debug
        self.acc_dtype = cutlass.Float32
        self.mma_tiler = mma_tiler_mnk
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.a_in_smem = a_in_smem
        self.has_bias = has_bias
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.transform_fragment_size = transform_fragment_size
        self.enable_pdl = enable_pdl
        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.schedule_warp_id = 5
        self.weight_tma_warp_id = 6
        self.activation_tma_warp_id = 7
        self.num_transform_warpgroups = 2
        self.transform_warp_id = tuple(range(8, 8 + 4 * self.num_transform_warpgroups))
        self.num_transform_warps = len(self.transform_warp_id)
        # Transform warps keep the 128-register launch default.
        self.num_regs_epilogue_warps = 176
        self.num_regs_generic_warps = 80
        self.threads_per_cta = 32 * (self.transform_warp_id[-1] + 1)

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            1, 32 * len(self.epilog_warp_id)
        )
        self.tmem_ptr_sync_barrier = pipeline.NamedBarrier(2, self.threads_per_cta)
        self.sched_sync_barrier = pipeline.NamedBarrier(5, 32)

        self.smem_buffer_align_bytes = 1024

    # ------------------------------------------------------------------ setup
    def _setup_attributes(self):
        """Derive the MMA, pipeline depths and smem / TMEM layouts."""
        self.transform_a_source = (
            tcgen05.OperandSource.SMEM
            if self.a_in_smem
            else mixed_input_utils.get_transform_a_source(self.a_major_mode)
        )
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.mma_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
            self.transform_a_source,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        (
            self.num_load2trans_stage,
            self.num_trans2mma_stage,
            self.num_acc_stage,
            self.num_c_stage,
            self.num_tile_info_stage,
            self.num_acc_tmem_cols,
            self.num_a_tmem_cols,
        ) = self._compute_stages_and_tmem_cols(
            tiled_mma,
            self.mma_tiler,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.c_layout,
            self.transform_a_source,
            self.smem_buffer_align_bytes,
        )

        # TMEM allocation requires power-of-two column alignment.
        self.num_tmem_alloc_cols = cute.round_up(
            self.num_acc_tmem_cols + self.num_a_tmem_cols,
            cute.arch.get_min_tmem_alloc_cols("sm_100"),
        )
        self.num_tmem_alloc_cols = 2 ** (ceil(log2(self.num_tmem_alloc_cols)))
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )
        (
            self.smem_layout_a,
            self.smem_layout_a_transform,
            self.smem_layout_b,
        ) = mixed_input_utils.compute_smem_layout(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.num_load2trans_stage,
            self.num_trans2mma_stage,
        )

    # ------------------------------------------------------------- transform
    @cute.jit
    def _setup_transform_partitions(
        self,
        tiled_mma: cute.TiledMma,
        copy_atom_a_input: cute.CopyAtom,
        copy_atom_a_transform: cute.CopyAtom,
        sA_input: cute.Tensor,
        a_smem_layout_transform: cute.ComposedLayout,
        sA_transform: Optional[cute.Tensor],
        accumulators: cute.Tensor,
        transform_tidx: cutlass.Int32,
        transform_group_idx: cutlass.Int32,
    ):
        if cutlass.const_expr(self.transform_a_source == tcgen05.OperandSource.TMEM):
            tmem_ptr_transform = cute.recast_ptr(
                accumulators.iterator + self.num_acc_tmem_cols,
                dtype=self.mma_dtype,
            )
            transform_destination = cute.make_tensor(
                tmem_ptr_transform,
                tiled_mma.make_fragment_A(a_smem_layout_transform.outer).layout,
            )
        else:
            transform_destination = sA_transform

        _, dst_copy_a, tAsA_input, tAsA_transform = (
            mixed_input_utils.transform_partition(
                self.transform_a_source,
                TransformMode.ConvertOnly,
                copy_atom_a_input,
                copy_atom_a_transform,
                sA_input,
                transform_destination,
                transform_tidx,
            )
        )
        # The two transform warpgroups split the A blocks of a stage between them.
        num_transform_blocks = cute.size(tAsA_input, mode=[3])
        assert num_transform_blocks % self.num_transform_warpgroups == 0, (
            "A blocks must divide evenly across transform warpgroups"
        )
        blocks_per_transform_group = (
            num_transform_blocks // self.num_transform_warpgroups
        )
        transform_group_tiler = (None, None, None, blocks_per_transform_group, None)
        transform_group_coord = (None, None, None, (None, transform_group_idx), None)
        tAsA_input = cute.logical_divide(tAsA_input, transform_group_tiler)[
            transform_group_coord
        ]
        tAsA_transform = cute.logical_divide(tAsA_transform, transform_group_tiler)[
            transform_group_coord
        ]

        a_stage_coord = (None,) * (cute.rank(tAsA_input) - 1) + (0,)
        tArA = cute.make_rmem_tensor(
            tAsA_input[a_stage_coord].shape, tAsA_input.element_type
        )
        tArA_transform = cute.make_rmem_tensor(
            tAsA_input[a_stage_coord].shape, self.mma_dtype
        )

        # Convert in fragments of transform_fragment_size bytes so the smem loads of one
        # fragment overlap the conversion of the previous one.
        transform_tiler_size = min(
            cute.size(cute.coalesce(tAsA_input.layout), mode=[0]),
            self.transform_fragment_size,
        )
        assert transform_tiler_size % 4 == 0, "fragments convert whole 32-bit words"
        transform_tiler = cute.make_layout(transform_tiler_size)
        tArA_load = cute.flat_divide(tArA, transform_tiler)
        tArA_load = cute.group_modes(tArA_load, 1, cute.rank(tArA_load))
        tArA_transform_store = cute.flat_divide(tArA_transform, transform_tiler)
        tArA_transform_store = cute.group_modes(
            tArA_transform_store, 1, cute.rank(tArA_transform_store)
        )
        return (
            dst_copy_a,
            tAsA_input,
            tAsA_transform,
            tArA_load,
            tArA_transform,
            tArA_transform_store,
            transform_tiler,
        )

    @cute.jit
    def _transform_tile(
        self,
        a_load2trans_pipeline: pipeline.PipelineTmaAsync,
        trans2mma_pipeline: pipeline.PipelineAsyncUmma,
        a_load2trans_consumer_state: pipeline.PipelineState,
        trans2mma_producer_state: pipeline.PipelineState,
        dst_copy_a: Optional[cute.CopyAtom],
        tAsA_input: cute.Tensor,
        tAsA_transform: cute.Tensor,
        tArA_load: cute.Tensor,
        tArA_transform: cute.Tensor,
        tArA_transform_store: cute.Tensor,
        transform_tiler: cute.Layout,
        k_tile_cnt: cutlass.Int32,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        a_load2trans_consumer_state.reset_count()
        peek_load2trans_full_status = cutlass.Boolean(1)
        if a_load2trans_consumer_state.count < k_tile_cnt:
            peek_load2trans_full_status = a_load2trans_pipeline.consumer_try_wait(
                a_load2trans_consumer_state
            )
        trans2mma_producer_state.reset_count()
        peek_trans2mma_empty_status = cutlass.Boolean(1)
        if trans2mma_producer_state.count < k_tile_cnt:
            peek_trans2mma_empty_status = trans2mma_pipeline.producer_try_acquire(
                trans2mma_producer_state
            )

        for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            a_load2trans_pipeline.consumer_wait(
                a_load2trans_consumer_state, peek_load2trans_full_status
            )
            a_stage_coord = (None,) * (cute.rank(tAsA_input) - 1) + (
                a_load2trans_consumer_state.index,
            )
            tAsA_input_slice = tAsA_input[a_stage_coord]
            tAsA_input_slice = cute.flat_divide(tAsA_input_slice, transform_tiler)
            tAsA_input_slice = cute.group_modes(
                tAsA_input_slice, 1, cute.rank(tAsA_input_slice)
            )
            trans2mma_pipeline.producer_acquire(
                trans2mma_producer_state, peek_trans2mma_empty_status
            )
            cur_a_load2trans_consumer_state = a_load2trans_consumer_state.clone()
            for idx in cutlass.range_constexpr(cute.size(tArA_load, mode=[1])):
                cute.autovec_copy(tAsA_input_slice[(None, idx)], tArA_load[(None, idx)])
                if cutlass.const_expr(idx == cute.size(tArA_load, mode=[1]) - 1):
                    a_load2trans_consumer_state.advance()
                    if a_load2trans_consumer_state.count < k_tile_cnt:
                        peek_load2trans_full_status = (
                            a_load2trans_pipeline.consumer_try_wait(
                                a_load2trans_consumer_state
                            )
                        )
                packed_fragment = cute.recast_tensor(
                    tArA_load[(None, idx)], cutlass.Uint8
                ).load()
                tArA_transform_store[(None, idx)].store(
                    convert_e4m3_fragment(packed_fragment, self.mma_dtype)
                )

            a_transform_stage_coord = (None,) * (cute.rank(tAsA_transform) - 1) + (
                trans2mma_producer_state.index,
            )
            mixed_input_utils.store_transformed_a(
                tArA_transform, tAsA_transform[a_transform_stage_coord], dst_copy_a
            )
            if cutlass.const_expr(
                self.transform_a_source == tcgen05.OperandSource.TMEM
            ):
                cute.arch.fence_view_async_tmem_store()
            else:
                cute.arch.fence_proxy("async.shared", space="cta")
            a_load2trans_pipeline.consumer_release(cur_a_load2trans_consumer_state)
            trans2mma_pipeline.producer_commit(trans2mma_producer_state)
            trans2mma_producer_state.advance()
            if trans2mma_producer_state.count < k_tile_cnt:
                peek_trans2mma_empty_status = trans2mma_pipeline.producer_try_acquire(
                    trans2mma_producer_state
                )

        return a_load2trans_consumer_state, trans2mma_producer_state

    # --------------------------------------------------------------- launch
    @cute.jit
    def wrapper(
        self,
        weight_ptr: cute.Pointer,
        activation_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        dbg_ptr: cute.Pointer,
        n: cutlass.Int64,
        m: cutlass.Int64,
        k: cutlass.Int64,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Build the logical tensors: weight [n, k] and activation [m, k] K-major,
        out [m, n] row-major seen as C[n, m] column-major; scale is one fp32; bias is
        [n] in the activation dtype (any pointer when ``has_bias`` is False); dbg is
        the debug buffer (any pointer when ``debug`` is False)."""
        weight = cute.make_tensor(
            weight_ptr, cute.make_ordered_layout((n, k, 1), order=(1, 0, 2))
        )
        activation = cute.make_tensor(
            activation_ptr, cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))
        )
        output = cute.make_tensor(
            output_ptr, cute.make_ordered_layout((n, m, 1), order=(0, 1, 2))
        )
        scale = cute.make_tensor(scale_ptr, cute.make_layout((1,)))
        bias = cute.make_tensor(bias_ptr, cute.make_layout((n,)))
        dbg = cute.make_tensor(dbg_ptr, cute.make_layout((16 * 4096,)))
        return self(
            weight, activation, scale, bias, output, dbg, max_active_clusters, stream
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        c: cute.Tensor,
        dbg: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype: type[cutlass.Numeric] = a.element_type
        self.b_dtype: type[cutlass.Numeric] = b.element_type
        self.c_dtype: type[cutlass.Numeric] = c.element_type
        self.mma_dtype = self.b_dtype
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.mma_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
            self.transform_a_source,
        )
        a_op = mixed_input_utils.get_tma_atom_kind(
            self.is_a_mcast, self.use_2cta_instrs, False
        )
        b_op = mixed_input_utils.get_tma_atom_kind(
            self.is_b_mcast, self.use_2cta_instrs, True
        )
        smem_layout_a_per_stage = cute.slice_(self.smem_layout_a, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            smem_layout_a_per_stage,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        smem_layout_b_per_stage = cute.slice_(self.smem_layout_b, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            smem_layout_b_per_stage,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        self.num_tma_load_bytes_a = cute.size_in_bytes(
            self.a_dtype, smem_layout_a_per_stage
        )
        self.num_tma_load_bytes_b = cute.size_in_bytes(
            self.b_dtype, smem_layout_b_per_stage
        ) * cute.size(tiled_mma.thr_id.shape)

        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.swizzle_size,
            self.raster_along_m,
        )

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )

        @cute.struct
        class SharedStorage:
            tile_info: cute.struct.MemRange[cutlass.Int32, 4 * self.num_tile_info_stage]
            a_load2trans_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_load2trans_stage
            ]
            a_load2trans_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_load2trans_stage
            ]
            a_trans2mma_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_trans2mma_stage
            ]
            a_trans2mma_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_trans2mma_stage
            ]
            b_load2mma_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_load2trans_stage
            ]
            b_load2mma_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_load2trans_stage
            ]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tile_info_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_info_stage
            ]
            tile_info_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_info_stage
            ]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            c,
            scale,
            bias,
            dbg,
            self.cluster_layout_vmnk,
            self.smem_layout_a,
            self.smem_layout_a_transform,
            self.smem_layout_b,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    # --------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tensor_c: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        dbg: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        a_smem_layout_transform: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        gdx, gdy, _ = cute.arch.grid_dim()
        dbg_base = (bidx + gdx * (bidy + gdy * bidz)) * 16
        if warp_idx == self.epilog_warp_id[0]:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipelines: weight TMA -> transform warps (a_load2trans), transform warps -> MMA
        # (trans2mma), activation TMA -> MMA (b_load2mma), MMA -> epilogue (acc), and the
        # scheduler warp -> everyone (tile_info).
        transform_thread_idx = (
            tidx - 32 * self.transform_warp_id[0]
            if tidx >= 32 * self.transform_warp_id[0]
            else tidx
        )
        a_load2trans_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.a_load2trans_full_mbar_ptr.data_ptr(),
            num_stages=self.num_load2trans_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mcast_ctas_a * self.num_transform_warps,
            ),
            tx_count=self.num_tma_load_bytes_a,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=transform_thread_idx,
            mcast_mode_mn=(1, 0),
            defer_sync=True,
        )
        cta_v_size = cute.size(cluster_layout_vmnk, mode=[0])
        trans2mma_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.a_trans2mma_full_mbar_ptr.data_ptr(),
            num_stages=self.num_trans2mma_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * self.num_transform_warps * cta_v_size
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        b_load2mma_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_load2mma_full_mbar_ptr.data_ptr(),
            num_stages=self.num_load2trans_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mcast_ctas_b
            ),
            tx_count=self.num_tma_load_bytes_b,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(0, 1),
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, cta_v_size * len(self.epilog_warp_id)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_full_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_info_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_cta - 32
            ),
            defer_sync=True,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_ptr_sync_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=self.smem_buffer_align_bytes,
            swizzle=c_smem_layout_staged.inner,
        )
        sA_input = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=self.smem_buffer_align_bytes,
            swizzle=a_smem_layout.inner,
        )
        sB_input = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=self.smem_buffer_align_bytes,
            swizzle=b_smem_layout.inner,
        )
        sA_transform = None
        if cutlass.const_expr(self.transform_a_source == tcgen05.OperandSource.SMEM):
            sA_transform = smem.allocate_tensor(
                element_type=self.mma_dtype,
                layout=a_smem_layout_transform.outer,
                byte_alignment=self.smem_buffer_align_bytes,
                swizzle=a_smem_layout_transform.inner,
            )
        sTile_info = storage.tile_info.get_tensor(
            cute.make_layout((4, self.num_tile_info_stage), stride=(1, 4))
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # (bM, bK, loopM, loopK, loopL) / (bN, bK, loopN, loopK, loopL) / (bM, bN, ...)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        gC_mnl_simt = cute.local_tile(
            tensor_c, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCgC_simt = thr_mma.partition_C(gC_mnl_simt)

        copy_atom_a_input = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.a_dtype, num_bits_per_copy=32
        )
        copy_atom_a_transform = mixed_input_utils.get_copy_atom_a_transform(
            self.mma_dtype,
            self.use_2cta_instrs,
            self.transform_a_source,
            a_smem_layout.outer.shape,
            self.a_dtype,
        )

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA_input, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB_input, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tCrB = tiled_mma.make_fragment_B(sB_input)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()

        # ---- scheduler warp: hands (cta_coord_m, activation offset, valid, rows left)
        # to every other warp through the tile_info pipeline.
        if warp_idx == self.schedule_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_generic_warps)
            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_info_stage
            )
            act_rows = cutlass.Int32(cute.size(tensor_c.shape[1]))
            act_tiles = cutlass.Int32(
                cute.ceil_div(act_rows, self.cta_tile_shape_mnk[1])
            )
            tile_sched = utils.StaticPersistentRuntimeTileScheduler.create(
                tile_sched_params,
                (bidx, bidy, bidz),
                cute.arch.grid_dim(),
                inner_mode=0 if self.raster_along_m else 1,
            )
            work_tile = tile_sched.initial_work_tile_info()
            not_last_tile = cutlass.Boolean(1)
            dbg_tiles = cutlass.Int32(0)
            while not_last_tile:
                cluster_tile_coord_mnl = work_tile.tile_idx
                cta_tile_coord_m = (
                    cluster_tile_coord_mnl[0] * self.cluster_shape_mn[0]
                    + block_in_cluster_coord_vmnk[1] * cute.size(tiled_mma.thr_id.shape)
                    + block_in_cluster_coord_vmnk[0]
                )
                act_tile_idx = (
                    cluster_tile_coord_mnl[1] * self.cluster_shape_mn[1]
                    + block_in_cluster_coord_vmnk[2]
                )
                # The runtime scheduler leaves its outer mode unbounded, so the
                # finite extent of the outer mode ends the walk.
                has_work = work_tile.is_valid_tile
                if cutlass.const_expr(self.raster_along_m):
                    has_work = has_work and (act_tile_idx < act_tiles)
                else:
                    has_work = has_work and (
                        cluster_tile_coord_mnl[0]
                        < tile_sched_params.problem_layout_ncluster_mnl.shape[0]
                    )
                is_dense_tile = has_work and (act_tile_idx < act_tiles)
                if is_dense_tile or not has_work:
                    if cutlass.const_expr(self.debug):
                        with cute.arch.elect_one():
                            dbg[dbg_base + 8] = cutlass.Int32(1)
                    tile_info_pipeline.producer_acquire(tile_info_producer_state)
                    if cutlass.const_expr(self.debug):
                        dbg_tiles += 1
                        with cute.arch.elect_one():
                            dbg[dbg_base + 8] = cutlass.Int32(2)
                            dbg[dbg_base + 0] = dbg_tiles
                    act_start = cutlass.Int32(-1)
                    group_idx = cutlass.Int32(1)  # 1 = no work (group_count is 1)
                    distance_to_boundary = cutlass.Int32(-1)
                    if is_dense_tile:
                        act_start = act_tile_idx * self.cta_tile_shape_mnk[1]
                        group_idx = cutlass.Int32(0)
                        distance_to_boundary = act_rows - act_start
                    cur_sTile_info = sTile_info[(None, tile_info_producer_state.index)]
                    with cute.arch.elect_one():
                        cur_sTile_info[0] = cta_tile_coord_m
                        cur_sTile_info[1] = act_start
                        cur_sTile_info[2] = group_idx
                        cur_sTile_info[3] = distance_to_boundary
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.sched_sync_barrier.arrive_and_wait()
                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                    tile_info_producer_state.advance()
                if has_work:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                else:
                    not_last_tile = cutlass.Boolean(0)
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        # ---- activation TMA warp (the only role that waits on the preceding grid).
        if warp_idx == self.activation_tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_generic_warps)
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            work_tile, tile_info_consumer_state = self._next_work_tile(
                tile_info_pipeline, tile_info_consumer_state, sTile_info
            )
            b_load2mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_load2trans_stage
            )
            if cutlass.const_expr(self.enable_pdl):
                griddepcontrol_wait()
            dbg_tiles = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                if cutlass.const_expr(self.debug):
                    with cute.arch.elect_one():
                        dbg[dbg_base + 9] = cutlass.Int32(1)
                coord_n_offset = (
                    (work_tile.coord_n, 0)
                    if cutlass.const_expr(
                        self.b_major_mode == cute.nvgpu.OperandMajorMode.MN
                    )
                    else (0, work_tile.coord_n)
                )
                tBgB_slice = cute.make_tensor(
                    (
                        tBgB.iterator[0] + coord_n_offset[0],
                        coord_n_offset[1] + tBgB.iterator[1],
                    ),
                    cute.slice_(tBgB.layout, (None, 0, None, 0)),
                )
                b_load2mma_producer_state.reset_count()
                for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    b_load2mma_pipeline.producer_acquire(b_load2mma_producer_state)
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, b_load2mma_producer_state.count)],
                        tBsB[(None, b_load2mma_producer_state.index)],
                        tma_bar_ptr=b_load2mma_pipeline.producer_get_barrier(
                            b_load2mma_producer_state
                        ),
                        mcast_mask=b_full_mcast_mask,
                    )
                    b_load2mma_pipeline.producer_commit(b_load2mma_producer_state)
                    b_load2mma_producer_state.advance()
                if cutlass.const_expr(self.debug):
                    dbg_tiles += 1
                    with cute.arch.elect_one():
                        dbg[dbg_base + 1] = dbg_tiles
                        dbg[dbg_base + 9] = cutlass.Int32(2)
                work_tile, tile_info_consumer_state = self._next_work_tile(
                    tile_info_pipeline, tile_info_consumer_state, sTile_info
                )
            b_load2mma_pipeline.producer_tail(b_load2mma_producer_state)

        # ---- weight TMA warp.
        if warp_idx == self.weight_tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_generic_warps)
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            work_tile, tile_info_consumer_state = self._next_work_tile(
                tile_info_pipeline, tile_info_consumer_state, sTile_info
            )
            a_load2trans_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_load2trans_stage
            )
            dbg_tiles = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                if cutlass.const_expr(self.debug):
                    with cute.arch.elect_one():
                        dbg[dbg_base + 10] = cutlass.Int32(1)
                tAgA_slice = tAgA[
                    (
                        None,
                        work_tile.cta_coord_m // cute.size(tiled_mma.thr_id.shape),
                        None,
                        0,
                    )
                ]
                a_load2trans_producer_state.reset_count()
                peek_load2trans_empty_status = cutlass.Boolean(1)
                if a_load2trans_producer_state.count < k_tile_cnt:
                    peek_load2trans_empty_status = (
                        a_load2trans_pipeline.producer_try_acquire(
                            a_load2trans_producer_state
                        )
                    )
                for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    a_load2trans_pipeline.producer_acquire(
                        a_load2trans_producer_state, peek_load2trans_empty_status
                    )
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, a_load2trans_producer_state.count)],
                        tAsA[(None, a_load2trans_producer_state.index)],
                        tma_bar_ptr=a_load2trans_pipeline.producer_get_barrier(
                            a_load2trans_producer_state
                        ),
                        mcast_mask=a_full_mcast_mask,
                    )
                    a_load2trans_pipeline.producer_commit(a_load2trans_producer_state)
                    a_load2trans_producer_state.advance()
                    if a_load2trans_producer_state.count < k_tile_cnt:
                        peek_load2trans_empty_status = (
                            a_load2trans_pipeline.producer_try_acquire(
                                a_load2trans_producer_state
                            )
                        )
                if cutlass.const_expr(self.debug):
                    dbg_tiles += 1
                    with cute.arch.elect_one():
                        dbg[dbg_base + 2] = dbg_tiles
                        dbg[dbg_base + 10] = cutlass.Int32(2)
                work_tile, tile_info_consumer_state = self._next_work_tile(
                    tile_info_pipeline, tile_info_consumer_state, sTile_info
                )
            a_load2trans_pipeline.producer_tail(a_load2trans_producer_state)

        # ---- transform warps.
        if warp_idx >= self.transform_warp_id[0]:
            transform_group_idx = (warp_idx - self.transform_warp_id[0]) // 4
            transform_local_tidx = tidx - 32 * (
                self.transform_warp_id[0] + transform_group_idx * 4
            )
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            accumulators = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            (
                dst_copy_a,
                tAsA_input,
                tAsA_transform,
                tArA_load,
                tArA_transform,
                tArA_transform_store,
                transform_tiler,
            ) = self._setup_transform_partitions(
                tiled_mma,
                copy_atom_a_input,
                copy_atom_a_transform,
                sA_input,
                a_smem_layout_transform,
                sA_transform,
                accumulators,
                transform_local_tidx,
                transform_group_idx,
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            work_tile, tile_info_consumer_state = self._next_work_tile(
                tile_info_pipeline, tile_info_consumer_state, sTile_info
            )
            a_load2trans_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_load2trans_stage
            )
            trans2mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_trans2mma_stage
            )
            dbg_tiles = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                if cutlass.const_expr(self.debug):
                    if warp_idx == self.transform_warp_id[0]:
                        with cute.arch.elect_one():
                            dbg[dbg_base + 11] = cutlass.Int32(1)
                a_load2trans_consumer_state, trans2mma_producer_state = (
                    self._transform_tile(
                        a_load2trans_pipeline,
                        trans2mma_pipeline,
                        a_load2trans_consumer_state,
                        trans2mma_producer_state,
                        dst_copy_a,
                        tAsA_input,
                        tAsA_transform,
                        tArA_load,
                        tArA_transform,
                        tArA_transform_store,
                        transform_tiler,
                        k_tile_cnt,
                    )
                )
                if cutlass.const_expr(self.debug):
                    dbg_tiles += 1
                    if warp_idx == self.transform_warp_id[0]:
                        with cute.arch.elect_one():
                            dbg[dbg_base + 3] = dbg_tiles
                            dbg[dbg_base + 11] = cutlass.Int32(2)
                work_tile, tile_info_consumer_state = self._next_work_tile(
                    tile_info_pipeline, tile_info_consumer_state, sTile_info
                )
            trans2mma_pipeline.producer_tail(trans2mma_producer_state)

        # ---- MMA warp.
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_generic_warps)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            accumulators = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            if cutlass.const_expr(
                self.transform_a_source == tcgen05.OperandSource.TMEM
            ):
                tmem_ptr_transform = cute.recast_ptr(
                    accumulators.iterator + self.num_acc_tmem_cols, dtype=self.mma_dtype
                )
                tCrA = cute.make_tensor(
                    tmem_ptr_transform,
                    tiled_mma.make_fragment_A(a_smem_layout_transform.outer).layout,
                )
            else:
                tCrA = tiled_mma.make_fragment_A(sA_transform)
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            work_tile, tile_info_consumer_state = self._next_work_tile(
                tile_info_pipeline, tile_info_consumer_state, sTile_info
            )
            trans2mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_trans2mma_stage
            )
            b_load2mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_load2trans_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            dbg_tiles = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                tCtAcc = accumulators[(None, None, None, acc_producer_state.index)]
                b_load2mma_consumer_state.reset_count()
                trans2mma_consumer_state.reset_count()
                peek_trans2mma_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    if trans2mma_consumer_state.count < k_tile_cnt:
                        peek_trans2mma_full_status = (
                            trans2mma_pipeline.consumer_try_wait(
                                trans2mma_consumer_state
                            )
                        )
                    if cutlass.const_expr(self.debug):
                        with cute.arch.elect_one():
                            dbg[dbg_base + 12] = cutlass.Int32(1)
                    acc_pipeline.producer_acquire(acc_producer_state)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        if cutlass.const_expr(self.debug):
                            with cute.arch.elect_one():
                                dbg[dbg_base + 12] = cutlass.Int32(2)
                        trans2mma_pipeline.consumer_wait(
                            trans2mma_consumer_state, peek_trans2mma_full_status
                        )
                        if cutlass.const_expr(self.debug):
                            with cute.arch.elect_one():
                                dbg[dbg_base + 12] = cutlass.Int32(3)
                        b_load2mma_pipeline.consumer_wait(b_load2mma_consumer_state)
                        if cutlass.const_expr(self.debug):
                            with cute.arch.elect_one():
                                dbg[dbg_base + 12] = cutlass.Int32(4)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord_a = (
                                None,
                                None,
                                kblock_idx,
                                trans2mma_consumer_state.index,
                            )
                            kblock_coord_b = (
                                None,
                                None,
                                kblock_idx,
                                b_load2mma_consumer_state.index,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord_a],
                                tCrB[kblock_coord_b],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        trans2mma_pipeline.consumer_release(trans2mma_consumer_state)
                        b_load2mma_pipeline.consumer_release(b_load2mma_consumer_state)
                        trans2mma_consumer_state.advance()
                        b_load2mma_consumer_state.advance()
                        peek_trans2mma_full_status = cutlass.Boolean(1)
                        if trans2mma_consumer_state.count < k_tile_cnt:
                            peek_trans2mma_full_status = (
                                trans2mma_pipeline.consumer_try_wait(
                                    trans2mma_consumer_state
                                )
                            )
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                if cutlass.const_expr(self.debug):
                    dbg_tiles += 1
                    with cute.arch.elect_one():
                        dbg[dbg_base + 4] = dbg_tiles
                        dbg[dbg_base + 12] = cutlass.Int32(5)
                work_tile, tile_info_consumer_state = self._next_work_tile(
                    tile_info_pipeline, tile_info_consumer_state, sTile_info
                )
            acc_pipeline.producer_tail(acc_producer_state)

        # ---- epilogue warps: D = scale * acc (+ bias[row]), cast, TMA store.
        if warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)
            epi_tidx = tidx
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            accumulators = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                mixed_input_utils.epilog_tmem_copy_and_partition(
                    self.cta_tile_shape_mnk,
                    self.c_layout,
                    self.c_dtype,
                    self.acc_dtype,
                    epi_tidx,
                    accumulators,
                    tCgC,
                    epi_tile,
                    self.use_2cta_instrs,
                )
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = (
                mixed_input_utils.epilog_smem_copy_and_partition(
                    self.c_layout,
                    self.c_dtype,
                    self.acc_dtype,
                    tiled_copy_t2r,
                    tTR_rC,
                    epi_tidx,
                    sC,
                )
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
                simt_atom,
                tTR_gC_partitioned,
            ) = mixed_input_utils.epilog_gmem_copy_and_partition(
                self.c_dtype,
                epi_tidx,
                tma_atom_c,
                tiled_copy_t2r,
                tCgC,
                tCgC_simt,
                epi_tile,
                sC,
            )
            # Per-element (row, col) within the CTA tile, for the bias row and the
            # SIMT-store predicate.
            thr_mapping = cute.make_identity_tensor(
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1])
            )
            thr_mapping_mn = cute.flat_divide(thr_mapping, epi_tile)
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            m_thr_offset = thr_copy_t2r.partition_D(thr_mapping_mn)
            m_thr_offset = cute.group_modes(m_thr_offset, 3, cute.rank(m_thr_offset))

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * len(self.epilog_warp_id)
                ),
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            work_tile, tile_info_consumer_state = self._next_work_tile(
                tile_info_pipeline, tile_info_consumer_state, sTile_info
            )
            weight_rows = cutlass.Int32(cute.size(tensor_c.shape[0]))
            num_prev_subtiles = cutlass.Int32(0)
            # Every tile here has 128 rows per CTA, so the TMEM load is the 32x32b
            # shape: one epilogue thread owns one output row for the whole tile, and
            # the bias of that row is a single register.
            assert self.cta_tile_shape_mnk[0] == 128, (
                "the per-thread bias register assumes the 32x32b TMEM load (128 rows)"
            )
            thread_row = m_thr_offset[(0, 0, 0, 0)][0]
            dbg_tiles = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                m_base = work_tile.cta_coord_m * self.cta_tile_shape_mnk[0]
                if cutlass.const_expr(self.debug):
                    if epi_tidx == 0:
                        dbg[dbg_base + 13] = cutlass.Int32(1)

                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        work_tile.cta_coord_m // cute.size(tiled_mma.thr_id.shape),
                        0,
                        0,
                    )
                ]
                tma_store_offset_coord = (
                    (work_tile.coord_n, 0)
                    if cutlass.const_expr(self.c_layout.is_n_major_c())
                    else (0, work_tile.coord_n)
                )
                bSG_gC = cute.make_tensor(
                    (
                        tma_store_offset_coord[0] + bSG_gC.iterator[0],
                        tma_store_offset_coord[1] + bSG_gC.iterator[1],
                    ),
                    bSG_gC.layout,
                )
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                tTR_gC = tTR_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        None,
                        None,
                        work_tile.cta_coord_m // cute.size(tiled_mma.thr_id.shape),
                        0,
                        0,
                    )
                ]
                tTR_gC = cute.make_tensor(
                    tTR_gC.iterator + (work_tile.coord_n * tensor_c.layout.stride[1]),
                    tTR_gC.layout,
                )
                tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                acc_pipeline.consumer_wait(acc_consumer_state)
                if cutlass.const_expr(self.debug):
                    if epi_tidx == 0:
                        dbg[dbg_base + 13] = cutlass.Int32(2)
                # Scale and bias are read after the accumulator wait so that under PDL
                # they are ordered behind the preceding grid too.
                scale_val = cutlass.Float32(scale[0])
                bias_val = cutlass.Float32(0.0)
                if cutlass.const_expr(self.has_bias):
                    bias_row = m_base + thread_row
                    if bias_row < weight_rows:
                        bias_val = cutlass.Float32(bias[bias_row])
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                for subtile_idx in cutlass.range(subtile_cnt):
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, subtile_idx)],
                        tTR_rAcc,
                    )
                    if subtile_idx == subtile_cnt - 1:
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()
                    m_thr_slice = m_thr_offset[(None, None, None, subtile_idx)]
                    # scale * acc (+ bias[row]) in fp32, on the whole fragment.
                    if cutlass.const_expr(self.has_bias):
                        tTR_rAcc.store(scale_val * tTR_rAcc.load() + bias_val)
                    else:
                        tTR_rAcc.store(scale_val * tTR_rAcc.load())

                    if work_tile.distance_to_boundary >= self.cta_tile_shape_mnk[1]:
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))
                        num_prev_subtiles += 1
                        c_buffer = num_prev_subtiles % self.num_c_stage
                        cute.copy(
                            tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)]
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                    else:
                        # Partial last activation tile: predicated SIMT store.
                        tTR_rC.store(tTR_rAcc.load().to(self.c_dtype))
                        tCpC = cute.make_rmem_tensor(
                            cute.make_layout(tTR_rC.shape), cutlass.Boolean
                        )
                        for i in cutlass.range(cute.size(tCpC), unroll_full=True):
                            tCpC[i] = (m_thr_slice[(i)][0] + m_base < weight_rows) and (
                                m_thr_slice[(i)][1] < work_tile.distance_to_boundary
                            )
                        cute.copy(
                            simt_atom,
                            cute.flatten(tTR_rC),
                            cute.flatten(tTR_gC[(None, None, None, subtile_idx)]),
                            pred=cute.flatten(tCpC),
                        )
                if cutlass.const_expr(self.debug):
                    dbg_tiles += 1
                    if epi_tidx == 0:
                        dbg[dbg_base + 5] = dbg_tiles
                        dbg[dbg_base + 13] = cutlass.Int32(3)
                work_tile, tile_info_consumer_state = self._next_work_tile(
                    tile_info_pipeline, tile_info_consumer_state, sTile_info
                )

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            c_pipeline.producer_tail()

        if cutlass.const_expr(self.enable_pdl):
            if warp_idx == self.mma_warp_id:
                with cute.arch.elect_one():
                    griddepcontrol_launch_dependents()

    @cute.jit
    def _next_work_tile(
        self,
        tile_info_pipeline: pipeline.PipelineAsync,
        tile_info_consumer_state: pipeline.PipelineState,
        sTile_info: cute.Tensor,
    ):
        """Consume the next tile_info record (one group, so group index 1 = done);
        returns it with the advanced consumer state."""
        tile_info_pipeline.consumer_wait(tile_info_consumer_state)
        work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
            1, sTile_info[(None, tile_info_consumer_state.index)]
        )
        cute.arch.fence_proxy("async.shared", space="cta")
        tile_info_pipeline.consumer_release(tile_info_consumer_state)
        tile_info_consumer_state.advance()
        # The advanced state is returned, not relied on as an in-place mutation: a
        # pipeline state handed to a @cute.jit helper is rebuilt from its MLIR values on
        # entry, so the caller's object does not see the advance (same reason
        # _transform_tile returns its states).
        return work_tile, tile_info_consumer_state

    # --------------------------------------------------------------- sizing
    @staticmethod
    def _compute_stages_and_tmem_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: tuple[int, int, int],
        cta_tile_shape_mnk: tuple[int, int, int],
        epi_tile: cute.Tile,
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        transform_a_source: tcgen05.OperandSource,
        smem_buffer_align_bytes: int,
    ) -> tuple[int, int, int, int, int, int, int]:
        """Fit the load, transform, accumulator and epilogue pipelines into smem and TMEM.

        Accumulator stages: two when the accumulator leaves room for them (so the
        epilogue of one tile overlaps the mainloop of the next), one otherwise.  With
        A in TMEM the transform stages take the TMEM columns the accumulators leave.
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        tCtAcc_stage1 = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        num_tmem_acc_col_per_stage = utils.get_num_tmem_alloc_cols(tCtAcc_stage1, True)
        sm100_tmem_columns = cute.arch.get_max_tmem_alloc_cols("sm_100")
        accumulator_stage_count = sm100_tmem_columns // num_tmem_acc_col_per_stage
        if transform_a_source == tcgen05.OperandSource.TMEM:
            accumulator_stage_count = 2 if num_tmem_acc_col_per_stage < 256 else 1
        num_elts_per_tmem_col = 32 // tiled_mma.op.a_dtype.width
        num_tmem_cols_a_per_stage = cute.round_up(
            (
                cta_tile_shape_mnk[2] // num_elts_per_tmem_col
                if transform_a_source == tcgen05.OperandSource.TMEM
                else 0
            ),
            4,
        )

        bytes_per_pipeline_stage = 16
        num_tile_info_stage = 3
        tile_info_bytes = (
            cute.size_in_bytes(cute.Int32, cute.make_layout((4, num_tile_info_stage)))
            + bytes_per_pipeline_stage * num_tile_info_stage
        )
        c_stage_count = 2
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype, c_layout, epi_tile, 1
        )
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * c_stage_count

        smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        carveout_smem_bytes = (
            bytes_per_pipeline_stage * accumulator_stage_count
            + c_bytes
            + tile_info_bytes
        )
        num_tmem_acc_cols = cute.round_up(
            accumulator_stage_count * num_tmem_acc_col_per_stage, 4
        )
        transform2mma_stage_count_a_source_tmem_potential = (
            (sm100_tmem_columns - num_tmem_acc_cols) // num_tmem_cols_a_per_stage
            if transform_a_source == tcgen05.OperandSource.TMEM
            else -1
        )
        if (
            transform_a_source == tcgen05.OperandSource.TMEM
            and transform2mma_stage_count_a_source_tmem_potential <= 0
        ):
            raise ValueError("Not enough TMEM capacity for selected tile size")
        a_load_bytes_per_stage = cute.round_up(
            cute.size_in_bytes(
                a_dtype,
                cute.make_layout((cta_tile_shape_mnk[0], cta_tile_shape_mnk[2])),
            ),
            smem_buffer_align_bytes,
        )
        b_load_bytes_per_stage = cute.round_up(
            cute.size_in_bytes(
                b_dtype,
                cute.make_layout(
                    (
                        cta_tile_shape_mnk[1] // cute.size(tiled_mma.thr_id),
                        cta_tile_shape_mnk[2],
                    )
                ),
            ),
            smem_buffer_align_bytes,
        )
        ab_load_bytes_per_stage = (
            a_load_bytes_per_stage
            + b_load_bytes_per_stage
            + 2 * bytes_per_pipeline_stage
        )
        a_transform_bytes_per_stage = (
            cute.round_up(
                cute.size_in_bytes(
                    tiled_mma.op.a_dtype,
                    cute.make_layout((cta_tile_shape_mnk[0], cta_tile_shape_mnk[2])),
                ),
                smem_buffer_align_bytes,
            )
            if transform_a_source == tcgen05.OperandSource.SMEM
            else 0
        ) + bytes_per_pipeline_stage
        transform2mma_stage_count_a_source_smem_potential = (
            smem_capacity - carveout_smem_bytes
        ) // (ab_load_bytes_per_stage + a_transform_bytes_per_stage)
        transform2mma_stage_count = (
            min(
                transform2mma_stage_count_a_source_tmem_potential,
                transform2mma_stage_count_a_source_smem_potential,
            )
            if transform_a_source == tcgen05.OperandSource.TMEM
            else transform2mma_stage_count_a_source_smem_potential
        )
        load2transform_stage_count = (
            smem_capacity
            - carveout_smem_bytes
            - (transform2mma_stage_count * a_transform_bytes_per_stage)
        ) // ab_load_bytes_per_stage
        if (
            load2transform_stage_count < 2
            or transform2mma_stage_count < 2
            or accumulator_stage_count < 1
        ):
            raise ValueError("Not enough SMEM or TMEM capacity for selected tile size")
        num_tmem_a_cols = transform2mma_stage_count * num_tmem_cols_a_per_stage
        c_stage_count += (
            smem_capacity
            - carveout_smem_bytes
            - load2transform_stage_count * ab_load_bytes_per_stage
            - transform2mma_stage_count * a_transform_bytes_per_stage
        ) // c_bytes_per_stage
        return (
            load2transform_stage_count,
            transform2mma_stage_count,
            accumulator_stage_count,
            c_stage_count,
            num_tile_info_stage,
            num_tmem_acc_cols,
            num_tmem_a_cols,
        )

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        swizzle_size: int,
        raster_along_m: bool,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        """Persistent scheduler over the CTA tiles of C.

        ``swizzle_size`` (the C++ kernel's L2 tile group, max_swizzle_size) is not
        applied yet: StaticPersistentRuntimeTileScheduler ignores the params'
        swizzled layout for the M raster and its termination test breaks on it for
        the N raster, so the grouped walk has to be done by the scheduler warp from
        the linear tile index.  Until then every variant walks the plain raster.
        """
        del swizzle_size
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            (*cluster_shape_mn, 1),
            raster_along_m=raster_along_m,
        )
        grid = (cluster_shape_mn[0], cluster_shape_mn[1], max_active_clusters)
        return tile_sched_params, grid
