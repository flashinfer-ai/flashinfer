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

from .w8a16_fp8_gemm_sm100_utils import (
    clock_lo,
    convert_e4m3_fragment,
    global_timer_lo,
)


def _slot_if(cond: cutlass.Boolean, slot: int) -> cutlass.Int32:
    """Debug timeline: ``slot`` when ``cond`` holds, else the scratch slot 15.  Folds the
    condition into the index so a timestamp store needs no nested staged ``if`` (a store
    through a tensor inside nested staged branches trips the tracer's value threading)."""
    return cutlass.Int32(15) - (15 - slot) * cond.to(cutlass.Int32)


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
        partial_output: bool = False,
    ):
        # partial_output: split-K mode.  The L mode of the operands is the K-slice,
        # C is fp32 partials [n, m, splits] and the epilogue applies no scale / bias
        # (the reduce kernel does, see w8a16_fp8_splitk_reduce.py).
        self.partial_output = partial_output
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
        # A stage's per-thread A is converted in two halves, one after the other, so a
        # half's registers (16 B of e4m3 per block, 32 B of 16-bit values) bound the
        # register use.  The partition is (copy, M-blocks, K-blocks, blocks, stages);
        # into TMEM the per-thread blocks sit in mode 3 (the 32x32b store repeats),
        # into smem the cotiled copy puts them in mode 1 and mode 3 is 1, so halve
        # whichever mode divides.  ``_transform_tile`` picks a half with a static index.
        split_mode = 3 if cute.size(tAsA_input, mode=[3]) % 2 == 0 else 1
        num_transform_blocks = cute.size(tAsA_input, mode=[split_mode])
        assert num_transform_blocks % 2 == 0, (
            "A blocks must split into two halves: partitioned A is "
            f"{tAsA_input.shape} (source {self.transform_a_source}, tile "
            f"{self.cta_tile_shape_mnk})"
        )
        transform_half_tiler = tuple(
            num_transform_blocks // 2 if i == split_mode else None for i in range(5)
        )
        tAsA_input = cute.logical_divide(tAsA_input, transform_half_tiler)
        tAsA_transform = cute.logical_divide(tAsA_transform, transform_half_tiler)
        half0 = self._transform_half_coord(split_mode, 0)
        a_stage_coord = (None,) * (cute.rank(tAsA_input[half0]) - 1) + (0,)
        tArA = cute.make_rmem_tensor(
            tAsA_input[half0][a_stage_coord].shape, tAsA_input.element_type
        )
        tArA_transform = cute.make_rmem_tensor(
            tAsA_input[half0][a_stage_coord].shape, self.mma_dtype
        )

        # Convert in fragments of transform_fragment_size bytes so the smem loads of one
        # fragment overlap the conversion of the previous one.
        transform_tiler_size = min(
            cute.size(cute.coalesce(tAsA_input[half0].layout), mode=[0]),
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
            split_mode,
        )

    @staticmethod
    def _transform_half_coord(split_mode: int, half: int):
        """Coordinate selecting half ``half`` of a stage's per-thread A blocks."""
        return tuple((None, half) if i == split_mode else None for i in range(5))

    def _dbg_clock(self, tag: int):
        """``%clock`` in debug mode, 0 otherwise (trace-time choice, no code when off)."""
        if self.debug:
            return clock_lo(tag)
        return cutlass.Int32(0)

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
        split_mode: cutlass.Constexpr,
        k_tile_cnt: cutlass.Int32,
        k_parity: cutlass.Int32,
        group_idx: cutlass.Int32,
        wait_a: cutlass.Int32,
        wait_stage: cutlass.Int32,
        cyc_convert: cutlass.Int32,
        cyc_store: cutlass.Int32,
        cyc_pre: cutlass.Int32,
        cyc_post: cutlass.Int32,
    ):
        """One tile's k-tiles for transform warpgroup ``group_idx``.

        The two transform warpgroups take alternate k-tiles of the kernel's k-tile
        sequence (group g the k-tiles whose global index has parity g; ``k_parity`` is
        the parity of this tile's first k-tile).  A k-tile's chain of smem loads,
        conversions, transformed store and fence is latency-bound, so with every warp
        on the same k-tile the MMA starved; alternating lets one group's chain overlap
        the other's.  Both pipeline states still step through every k-tile.

        Returns the advanced states and, in debug mode, the cycle accumulators grown:
        ``wait_a`` / ``wait_stage`` waiting for a raw A stage / a free transformed stage,
        ``cyc_convert`` / ``cyc_store`` the load-and-convert and store-and-fence phases,
        ``cyc_pre`` / ``cyc_post`` the rest before and after them."""
        a_load2trans_consumer_state.reset_count()
        trans2mma_producer_state.reset_count()
        for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            if (k_parity + k_tile) % 2 == group_idx:
                t_a0 = self._dbg_clock(50)
                a_load2trans_pipeline.consumer_wait(a_load2trans_consumer_state)
                t_a1 = self._dbg_clock(51)
                if cutlass.const_expr(self.debug):
                    wait_a += t_a1 - t_a0
                t_s0 = self._dbg_clock(52)
                trans2mma_pipeline.producer_acquire(trans2mma_producer_state)
                if cutlass.const_expr(self.debug):
                    wait_stage += clock_lo(53) - t_s0
                t_c0 = self._dbg_clock(60)
                if cutlass.const_expr(self.debug):
                    cyc_pre += t_c0 - t_a1
                for half in cutlass.range_constexpr(2):
                    half_coord = self._transform_half_coord(split_mode, half)
                    tAsA_input_half = tAsA_input[half_coord]
                    tAsA_transform_half = tAsA_transform[half_coord]
                    a_stage_coord = (None,) * (cute.rank(tAsA_input_half) - 1) + (
                        a_load2trans_consumer_state.index,
                    )
                    a_transform_stage_coord = (None,) * (
                        cute.rank(tAsA_transform_half) - 1
                    ) + (trans2mma_producer_state.index,)
                    tAsA_input_slice = tAsA_input_half[a_stage_coord]
                    tAsA_input_slice = cute.flat_divide(
                        tAsA_input_slice, transform_tiler
                    )
                    tAsA_input_slice = cute.group_modes(
                        tAsA_input_slice, 1, cute.rank(tAsA_input_slice)
                    )
                    for idx in cutlass.range_constexpr(cute.size(tArA_load, mode=[1])):
                        cute.autovec_copy(
                            tAsA_input_slice[(None, idx)], tArA_load[(None, idx)]
                        )
                        packed_fragment = cute.recast_tensor(
                            tArA_load[(None, idx)], cutlass.Uint8
                        ).load()
                        tArA_transform_store[(None, idx)].store(
                            convert_e4m3_fragment(packed_fragment, self.mma_dtype)
                        )
                    mixed_input_utils.store_transformed_a(
                        tArA_transform,
                        tAsA_transform_half[a_transform_stage_coord],
                        dst_copy_a,
                    )
                t_s1 = self._dbg_clock(61)
                if cutlass.const_expr(self.debug):
                    cyc_convert += t_s1 - t_c0
                # The raw stage goes back to the TMA producer once the converts have
                # consumed it; the transformed stores are fenced before the commit.
                a_load2trans_pipeline.consumer_release(a_load2trans_consumer_state)
                if cutlass.const_expr(
                    self.transform_a_source == tcgen05.OperandSource.TMEM
                ):
                    cute.arch.fence_view_async_tmem_store()
                else:
                    cute.arch.fence_proxy("async.shared", space="cta")
                t_s2 = self._dbg_clock(62)
                if cutlass.const_expr(self.debug):
                    cyc_store += t_s2 - t_s1
                trans2mma_pipeline.producer_commit(trans2mma_producer_state)
                if cutlass.const_expr(self.debug):
                    cyc_post += clock_lo(63) - t_s2
            a_load2trans_consumer_state.advance()
            trans2mma_producer_state.advance()

        return (
            a_load2trans_consumer_state,
            trans2mma_producer_state,
            wait_a,
            wait_stage,
            cyc_convert,
            cyc_store,
            cyc_pre,
            cyc_post,
        )

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
        splits: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Build the logical tensors.  ``k`` is one K-slice (the whole K when
        ``splits`` is 1).  Weight [n, K] and activation [m, K] are K-major 2-D
        tensors over the whole K (a size-1 batch mode); a tile's K-slice is its
        k-tile range offset by ``tile_l * k_tiles_per_slice``.  Out [m, n] row-major
        is C[n, m, 1]; the fp32 partials [splits, m, n] are C[n, splits * m, 1],
        slice ``l`` being activation rows ``l * m`` to ``(l + 1) * m``.

        Every tensor is 2-D on purpose: with the slice as a third (batch) mode the
        TMA loads and stores the DSL emits trap with an illegal instruction.
        """
        self.splits = splits
        k_total = k * splits
        weight = cute.make_tensor(
            weight_ptr, cute.make_ordered_layout((n, k_total, 1), order=(1, 0, 2))
        )
        activation = cute.make_tensor(
            activation_ptr, cute.make_ordered_layout((m, k_total, 1), order=(1, 0, 2))
        )
        output = cute.make_tensor(
            output_ptr, cute.make_ordered_layout((n, m * splits, 1), order=(0, 1, 2))
        )
        scale = cute.make_tensor(scale_ptr, cute.make_layout((1,)))
        bias = cute.make_tensor(bias_ptr, cute.make_layout((n,)))
        dbg = cute.make_tensor(dbg_ptr, cute.make_layout((64 * 4096,)))
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

        # Persistent grid: one CTA per SM slot (max_active_clusters clusters); the
        # scheduler warp walks the tile grid itself.
        grid = (self.cluster_shape_mn[0], self.cluster_shape_mn[1], max_active_clusters)

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )

        @cute.struct
        class SharedStorage:
            tile_info: cute.struct.MemRange[cutlass.Int32, 8 * self.num_tile_info_stage]
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
            # The bias of the current tile's weight rows, fp32, double-buffered by tile
            # parity so the next tile's staging never races the previous tile's reads.
            bias_tile: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 2 * self.cta_tile_shape_mnk[0]],
                16,
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
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        gdx, gdy, _ = cute.arch.grid_dim()
        dbg_base = (bidx + gdx * (bidy + gdy * bidz)) * 64
        if cutlass.const_expr(self.debug):
            # Slots 16..31: low 32 bits of %globaltimer at points of the first tile.
            # The tensor's iterator is materialized here, in straight-line code: the
            # DSL caches it on first use, and a first use inside a branch would leave
            # later uses outside that branch without a dominating definition.
            _ = dbg.iterator
            if tidx == 0:
                dbg[dbg_base + 16] = global_timer_lo(16)
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
        # This CTA's weight-tile row within the cluster tile and activation tile within
        # the cluster (used to derive the first work item without the scheduler).
        cta_m_in_cluster = (
            block_in_cluster_coord_vmnk[1] * cute.size(tiled_mma.thr_id.shape)
            + block_in_cluster_coord_vmnk[0]
        )
        act_in_cluster = block_in_cluster_coord_vmnk[2]
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
                self.num_mcast_ctas_a * (self.num_transform_warps // 2),
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
                pipeline.Agent.Thread, 32 * (self.num_transform_warps // 2) * cta_v_size
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
        sBias = storage.bias_tile.get_tensor(
            cute.make_layout((self.cta_tile_shape_mnk[0], 2))
        )
        sTile_info = storage.tile_info.get_tensor(
            cute.make_layout((8, self.num_tile_info_stage), stride=(1, 8))
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
        # k-tiles of one K-slice (the whole K when not split).
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]) // self.splits)

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

        if cutlass.const_expr(self.debug):
            if warp_idx == self.schedule_warp_id:
                with cute.arch.elect_one():
                    dbg[dbg_base + 32] = global_timer_lo(32)
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        if cutlass.const_expr(self.debug):
            if warp_idx == self.schedule_warp_id:
                with cute.arch.elect_one():
                    dbg[dbg_base + 33] = global_timer_lo(33)
        tmem.allocate(self.num_tmem_alloc_cols)
        if cutlass.const_expr(self.debug):
            if warp_idx == self.schedule_warp_id:
                with cute.arch.elect_one():
                    dbg[dbg_base + 34] = global_timer_lo(34)
        tmem.wait_for_alloc()
        if cutlass.const_expr(self.debug):
            if warp_idx == self.schedule_warp_id:
                with cute.arch.elect_one():
                    dbg[dbg_base + 30] = global_timer_lo(30)

        # ---- scheduler warp: hands (cta_coord_m, activation offset, valid, rows left)
        # to every other warp through the tile_info pipeline.
        if warp_idx == self.schedule_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_generic_warps)
            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_info_stage
            )
            (
                act_rows,
                act_tiles,
                major_clusters,
                minor_clusters,
                total_clusters,
                total_work,
            ) = self._grid_dims(tensor_c)
            _, _, num_persistent = cute.arch.grid_dim()
            # Every warp role derives work item `bidz` itself (_first_work_tile); the
            # scheduler publishes from the next one on, walking the linear work index
            # in steps of the launched cluster count.  A CTA without a first item has
            # no consumers waiting, so it publishes nothing (not even the sentinel).
            lin = cutlass.Int32(bidz)
            not_last_tile = lin < total_work
            lin += num_persistent
            dbg_tiles = cutlass.Int32(0)
            if cutlass.const_expr(self.debug):
                with cute.arch.elect_one():
                    dbg[dbg_base + 31] = global_timer_lo(31)
            while not_last_tile:
                has_work = lin < total_work
                cta_tile_coord_m, act_start_l, distance_l, tile_l = self._tile_at(
                    lin,
                    act_rows,
                    major_clusters,
                    minor_clusters,
                    total_clusters,
                    cta_m_in_cluster,
                    act_in_cluster,
                )
                act_tile_idx = act_start_l // self.cta_tile_shape_mnk[1]
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
                        act_start = act_start_l
                        group_idx = cutlass.Int32(0)
                        distance_to_boundary = distance_l
                    cur_sTile_info = sTile_info[(None, tile_info_producer_state.index)]
                    with cute.arch.elect_one():
                        cur_sTile_info[0] = cta_tile_coord_m
                        cur_sTile_info[1] = act_start
                        cur_sTile_info[2] = group_idx
                        cur_sTile_info[3] = distance_to_boundary
                        cur_sTile_info[4] = tile_l
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.sched_sync_barrier.arrive_and_wait()
                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                    tile_info_producer_state.advance()
                if has_work:
                    lin += num_persistent
                else:
                    not_last_tile = cutlass.Boolean(0)
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        # ---- activation TMA warp (the only role that waits on the preceding grid).
        if warp_idx == self.activation_tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_generic_warps)
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            work_tile, tile_l = self._first_work_tile(
                tensor_c, cta_m_in_cluster, act_in_cluster
            )
            b_load2mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_load2trans_stage
            )
            if cutlass.const_expr(self.enable_pdl):
                griddepcontrol_wait()
            dbg_tiles = cutlass.Int32(0)
            if cutlass.const_expr(self.debug):
                with cute.arch.elect_one():
                    dbg[dbg_base + 18] = global_timer_lo(18)
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
                tBgB_l = tBgB[(None, None, None, 0)]
                tBgB_slice = cute.make_tensor(
                    (
                        tBgB_l.iterator[0] + coord_n_offset[0],
                        coord_n_offset[1] + tBgB_l.iterator[1],
                    ),
                    cute.slice_(tBgB_l.layout, (None, 0, None)),
                )
                k_tile_base = tile_l * k_tile_cnt
                b_load2mma_producer_state.reset_count()
                for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    b_load2mma_pipeline.producer_acquire(b_load2mma_producer_state)
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[
                            (None, k_tile_base + b_load2mma_producer_state.count)
                        ],
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
                        dbg[dbg_base + _slot_if(dbg_tiles == 1, 19)] = global_timer_lo(
                            19
                        )
                work_tile, tile_info_consumer_state, tile_l = self._next_work_tile(
                    tile_info_pipeline, tile_info_consumer_state, sTile_info
                )
            b_load2mma_pipeline.producer_tail(b_load2mma_producer_state)

        # ---- weight TMA warp.
        if warp_idx == self.weight_tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_generic_warps)
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            work_tile, tile_l = self._first_work_tile(
                tensor_c, cta_m_in_cluster, act_in_cluster
            )
            a_load2trans_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_load2trans_stage
            )
            dbg_tiles = cutlass.Int32(0)
            if cutlass.const_expr(self.debug):
                with cute.arch.elect_one():
                    dbg[dbg_base + 20] = global_timer_lo(20)
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
                k_tile_base = tile_l * k_tile_cnt
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
                        tAgA_slice[
                            (None, k_tile_base + a_load2trans_producer_state.count)
                        ],
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
                work_tile, tile_info_consumer_state, tile_l = self._next_work_tile(
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
                split_mode,
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
            work_tile, tile_l = self._first_work_tile(
                tensor_c, cta_m_in_cluster, act_in_cluster
            )
            a_load2trans_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_load2trans_stage
            )
            trans2mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_trans2mma_stage
            )
            dbg_tiles = cutlass.Int32(0)
            xf_wait_a = cutlass.Int32(0)
            xf_wait_stage = cutlass.Int32(0)
            xf_cyc_convert = cutlass.Int32(0)
            xf_cyc_store = cutlass.Int32(0)
            xf_cyc_pre = cutlass.Int32(0)
            xf_cyc_post = cutlass.Int32(0)
            # Parity of the global index of this tile's first k-tile (see _transform_tile).
            xf_k_parity = cutlass.Int32(0)
            xf_clock0 = self._dbg_clock(54)
            if cutlass.const_expr(self.debug):
                if warp_idx == self.transform_warp_id[0]:
                    with cute.arch.elect_one():
                        dbg[dbg_base + 28] = global_timer_lo(28)
            while work_tile.is_valid_tile:
                if cutlass.const_expr(self.debug):
                    if warp_idx == self.transform_warp_id[0]:
                        with cute.arch.elect_one():
                            dbg[dbg_base + 11] = cutlass.Int32(1)
                (
                    a_load2trans_consumer_state,
                    trans2mma_producer_state,
                    xf_wait_a,
                    xf_wait_stage,
                    xf_cyc_convert,
                    xf_cyc_store,
                    xf_cyc_pre,
                    xf_cyc_post,
                ) = (
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
                        split_mode,
                        k_tile_cnt,
                        xf_k_parity,
                        transform_group_idx,
                        xf_wait_a,
                        xf_wait_stage,
                        xf_cyc_convert,
                        xf_cyc_store,
                        xf_cyc_pre,
                        xf_cyc_post,
                    )
                )
                xf_k_parity = (xf_k_parity + k_tile_cnt) % 2
                if cutlass.const_expr(self.debug):
                    dbg_tiles += 1
                    if warp_idx == self.transform_warp_id[0]:
                        with cute.arch.elect_one():
                            dbg[dbg_base + 3] = dbg_tiles
                            dbg[dbg_base + 11] = cutlass.Int32(2)
                            dbg[dbg_base + _slot_if(dbg_tiles == 1, 29)] = (
                                global_timer_lo(29)
                            )
                work_tile, tile_info_consumer_state, tile_l = self._next_work_tile(
                    tile_info_pipeline, tile_info_consumer_state, sTile_info
                )
            trans2mma_pipeline.producer_tail(trans2mma_producer_state)
            if cutlass.const_expr(self.debug):
                if warp_idx == self.transform_warp_id[0]:
                    with cute.arch.elect_one():
                        dbg[dbg_base + 43] = xf_wait_a
                        dbg[dbg_base + 44] = xf_wait_stage
                        dbg[dbg_base + 60] = xf_cyc_convert
                        dbg[dbg_base + 61] = xf_cyc_store
                        dbg[dbg_base + 62] = xf_cyc_pre
                        dbg[dbg_base + 63] = xf_cyc_post
                        dbg[dbg_base + 48] = clock_lo(55) - xf_clock0

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
            work_tile, tile_l = self._first_work_tile(
                tensor_c, cta_m_in_cluster, act_in_cluster
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
            # Debug wait accounting (cycles): inputs (transformed A + B stages), a free
            # accumulator stage, k-tiles seen, and the warp's total.
            mma_wait_in = cutlass.Int32(0)
            mma_wait_acc = cutlass.Int32(0)
            mma_ktiles = cutlass.Int32(0)
            mma_clock0 = self._dbg_clock(40)
            if cutlass.const_expr(self.debug):
                with cute.arch.elect_one():
                    dbg[dbg_base + 21] = global_timer_lo(21)
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
                    t_acc0 = self._dbg_clock(41)
                    acc_pipeline.producer_acquire(acc_producer_state)
                    if cutlass.const_expr(self.debug):
                        mma_wait_acc += clock_lo(42) - t_acc0
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        if cutlass.const_expr(self.debug):
                            with cute.arch.elect_one():
                                dbg[dbg_base + 12] = cutlass.Int32(2)
                        t_in0 = self._dbg_clock(43)
                        trans2mma_pipeline.consumer_wait(
                            trans2mma_consumer_state, peek_trans2mma_full_status
                        )
                        if cutlass.const_expr(self.debug):
                            with cute.arch.elect_one():
                                dbg[dbg_base + 12] = cutlass.Int32(3)
                        b_load2mma_pipeline.consumer_wait(b_load2mma_consumer_state)
                        if cutlass.const_expr(self.debug):
                            mma_wait_in += clock_lo(44) - t_in0
                            mma_ktiles += 1
                            with cute.arch.elect_one():
                                dbg[dbg_base + 12] = cutlass.Int32(4)
                                dbg[
                                    dbg_base
                                    + _slot_if(
                                        (dbg_tiles == 0).to(cutlass.Int32)
                                        * (_k_tile == 0).to(cutlass.Int32)
                                        == 1,
                                        22,
                                    )
                                ] = global_timer_lo(22)
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
                        dbg[dbg_base + _slot_if(dbg_tiles == 1, 23)] = global_timer_lo(
                            23
                        )
                        dbg[dbg_base + 27] = global_timer_lo(27)
                work_tile, tile_info_consumer_state, tile_l = self._next_work_tile(
                    tile_info_pipeline, tile_info_consumer_state, sTile_info
                )
            acc_pipeline.producer_tail(acc_producer_state)
            if cutlass.const_expr(self.debug):
                with cute.arch.elect_one():
                    dbg[dbg_base + 40] = mma_wait_in
                    dbg[dbg_base + 41] = mma_wait_acc
                    dbg[dbg_base + 42] = mma_ktiles
                    dbg[dbg_base + 47] = clock_lo(47) - mma_clock0

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
            work_tile, tile_l = self._first_work_tile(
                tensor_c, cta_m_in_cluster, act_in_cluster
            )
            weight_rows = cutlass.Int32(cute.size(tensor_c.shape[0]))
            act_rows = cutlass.Int32(cute.size(tensor_c.shape[1]) // self.splits)
            num_prev_subtiles = cutlass.Int32(0)
            assert self.cta_tile_shape_mnk[0] == 128, (
                "the bias staging uses the 128 epilogue threads, one per CTA-tile row"
            )
            dbg_tiles = cutlass.Int32(0)
            epi_tile_count = cutlass.Int32(0)
            epi_wait_acc = cutlass.Int32(0)
            epi_clock0 = self._dbg_clock(56)
            while work_tile.is_valid_tile:
                m_base = work_tile.cta_coord_m * self.cta_tile_shape_mnk[0]
                # The tile's activation offset within C: its K-slice's rows follow
                # the previous slices' (split-K partials).
                coord_n = work_tile.coord_n + tile_l * act_rows
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
                    (coord_n, 0)
                    if cutlass.const_expr(self.c_layout.is_n_major_c())
                    else (0, coord_n)
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
                    tTR_gC.iterator + (coord_n * tensor_c.layout.stride[1]),
                    tTR_gC.layout,
                )
                tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                t_e0 = self._dbg_clock(57)
                acc_pipeline.consumer_wait(acc_consumer_state)
                if cutlass.const_expr(self.debug):
                    epi_wait_acc += clock_lo(58) - t_e0
                    if epi_tidx == 0:
                        dbg[dbg_base + 13] = cutlass.Int32(2)
                        dbg[dbg_base + _slot_if(epi_tile_count == 0, 24)] = (
                            global_timer_lo(24)
                        )
                # Scale and bias are read after the accumulator wait so that under PDL
                # they are ordered behind the preceding grid too.
                scale_val = cutlass.Float32(scale[0])
                bias_buf = epi_tile_count % 2
                if cutlass.const_expr(self.has_bias):
                    # One fp32 bias per weight row of this tile, staged by the 128
                    # epilogue threads (rows past the weight read 0); the elements a
                    # thread holds span several rows, so each is indexed by its row.
                    bias_row = m_base + epi_tidx
                    bias_val = cutlass.Float32(0.0)
                    if bias_row < weight_rows:
                        bias_val = cutlass.Float32(bias[bias_row])
                    sBias[(epi_tidx, bias_buf)] = bias_val
                    self.epilog_sync_barrier.arrive_and_wait()
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                # Unrolled (the count is static).  Loading the next subtile's
                # accumulator ahead of the current one's store was tried and bought
                # nothing: the subtile's cost is the smem write, two named barriers
                # and the TMA store issue, not the TMEM load (GB200 timeline).
                for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, subtile_idx)],
                        tTR_rAcc,
                    )
                    if cutlass.const_expr(subtile_idx == subtile_cnt - 1):
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()
                    m_thr_slice = m_thr_offset[(None, None, None, subtile_idx)]
                    # scale * acc (+ bias[row]) in fp32: vectorized without the bias,
                    # per element (the row comes from the identity partition) with it.
                    if cutlass.const_expr(self.partial_output):
                        pass  # fp32 partials: the reduce applies scale and bias
                    elif cutlass.const_expr(self.has_bias):
                        for i in cutlass.range(cute.size(tTR_rAcc), unroll_full=True):
                            tTR_rAcc[i] = (
                                scale_val * tTR_rAcc[i]
                                + sBias[(m_thr_slice[(i)][0], bias_buf)]
                            )
                    else:
                        tTR_rAcc.store(scale_val * tTR_rAcc.load())
                    if cutlass.const_expr(self.debug and subtile_idx == 0):
                        if epi_tidx == 0:
                            dbg[dbg_base + _slot_if(epi_tile_count == 0, 6)] = (
                                global_timer_lo(6)
                            )

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
                        if cutlass.const_expr(self.debug and subtile_idx == 0):
                            if epi_tidx == 0:
                                dbg[dbg_base + _slot_if(epi_tile_count == 0, 7)] = (
                                    global_timer_lo(7)
                                )
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        if cutlass.const_expr(self.debug and subtile_idx == 0):
                            if epi_tidx == 0:
                                dbg[dbg_base + _slot_if(epi_tile_count == 0, 14)] = (
                                    global_timer_lo(14)
                                )
                        self.epilog_sync_barrier.arrive_and_wait()
                        if cutlass.const_expr(self.debug and subtile_idx == 0):
                            if epi_tidx == 0:
                                dbg[dbg_base + _slot_if(epi_tile_count == 0, 17)] = (
                                    global_timer_lo(17)
                                )
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
                epi_tile_count += 1
                if cutlass.const_expr(self.debug):
                    dbg_tiles += 1
                    if epi_tidx == 0:
                        dbg[dbg_base + _slot_if(epi_tile_count == 1, 25)] = (
                            global_timer_lo(25)
                        )
                        dbg[dbg_base + 26] = global_timer_lo(26)
                        dbg[dbg_base + 5] = dbg_tiles
                        dbg[dbg_base + 13] = cutlass.Int32(3)
                work_tile, tile_info_consumer_state, tile_l = self._next_work_tile(
                    tile_info_pipeline, tile_info_consumer_state, sTile_info
                )

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            c_pipeline.producer_tail()
            if cutlass.const_expr(self.debug):
                if epi_tidx == 0:
                    dbg[dbg_base + 45] = epi_wait_acc
                    dbg[dbg_base + 49] = clock_lo(59) - epi_clock0

        if cutlass.const_expr(self.enable_pdl):
            if warp_idx == self.mma_warp_id:
                with cute.arch.elect_one():
                    griddepcontrol_launch_dependents()

    @cute.jit
    def _grid_dims(self, tensor_c: cute.Tensor):
        """Cluster-tile grid of the problem: activation rows per K-slice, activation
        tiles, cluster counts along the fast (major) and slow (minor) raster axes, the
        cluster tiles of one slice and the total work items (cluster tiles x slices)."""
        act_rows = cutlass.Int32(cute.size(tensor_c.shape[1]) // self.splits)
        act_tiles = cutlass.Int32(cute.ceil_div(act_rows, self.cta_tile_shape_mnk[1]))
        weight_rows = cutlass.Int32(cute.size(tensor_c.shape[0]))
        weight_tiles = cutlass.Int32(
            cute.ceil_div(weight_rows, self.cta_tile_shape_mnk[0])
        )
        clusters_m = (
            weight_tiles + self.cluster_shape_mn[0] - 1
        ) // self.cluster_shape_mn[0]
        clusters_n = (
            act_tiles + self.cluster_shape_mn[1] - 1
        ) // self.cluster_shape_mn[1]
        if cutlass.const_expr(self.raster_along_m):
            major_clusters = clusters_m
            minor_clusters = clusters_n
        else:
            major_clusters = clusters_n
            minor_clusters = clusters_m
        total_clusters = clusters_m * clusters_n
        total_work = total_clusters * cutlass.Int32(self.splits)
        return (
            act_rows,
            act_tiles,
            major_clusters,
            minor_clusters,
            total_clusters,
            total_work,
        )

    @cute.jit
    def _tile_at(
        self,
        lin: cutlass.Int32,
        act_rows: cutlass.Int32,
        major_clusters: cutlass.Int32,
        minor_clusters: cutlass.Int32,
        total_clusters: cutlass.Int32,
        cta_m_in_cluster: cutlass.Int32,
        act_in_cluster: cutlass.Int32,
    ):
        """Work item ``lin`` of this CTA: (cta_coord_m, activation row offset, rows to
        the activation end, K-slice).  Slices are the slowest mode; within a slice the
        linear cluster-tile index maps to (major, minor), through the L2 group remap of
        CUTLASS's swizzle_and_rasterize (groups of swizzle_size clusters along the major
        axis, residual tiles not remapped), then to (m, n)."""
        tile_l = lin // total_clusters
        lin_mn = lin % total_clusters
        major = lin_mn % major_clusters
        minor = lin_mn // major_clusters
        swizzle = self.swizzle_size
        if cutlass.const_expr(swizzle > 1):
            major_groups = major_clusters // swizzle
            minor_groups = minor_clusters // swizzle
            first_residual_major = major_groups * swizzle
            first_residual_minor = minor_groups * swizzle
            if major < first_residual_major and minor < first_residual_minor:
                major_div = major // swizzle
                major_mod = major % swizzle
                minor_div = minor // swizzle
                minor_mod = minor % swizzle
                major = major_div + major_groups * minor_mod
                minor = major_mod + minor_div * swizzle
        if cutlass.const_expr(self.raster_along_m):
            cluster_m = major
            cluster_n = minor
        else:
            cluster_m = minor
            cluster_n = major
        cta_coord_m = cluster_m * self.cluster_shape_mn[0] + cta_m_in_cluster
        act_tile_idx = cluster_n * self.cluster_shape_mn[1] + act_in_cluster
        act_start = act_tile_idx * self.cta_tile_shape_mnk[1]
        distance_to_boundary = act_rows - act_start
        return cta_coord_m, act_start, distance_to_boundary, tile_l

    @cute.jit
    def _first_work_tile(
        self,
        tensor_c: cute.Tensor,
        cta_m_in_cluster: cutlass.Int32,
        act_in_cluster: cutlass.Int32,
    ):
        """The CTA's first work item, derived from its cluster index by every warp role
        itself (the scheduler warp publishes from the second item on): saves the
        scheduler-to-consumer hop at kernel start, ~0.6 us on GB200."""
        _, _, bidz = cute.arch.block_idx()
        lin = cutlass.Int32(bidz)
        (
            act_rows,
            act_tiles,
            major_clusters,
            minor_clusters,
            total_clusters,
            total_work,
        ) = self._grid_dims(tensor_c)
        cta_coord_m, act_start, distance_to_boundary, tile_l = self._tile_at(
            lin,
            act_rows,
            major_clusters,
            minor_clusters,
            total_clusters,
            cta_m_in_cluster,
            act_in_cluster,
        )
        group_idx = cutlass.Int32(1)  # 1 = no work (group_count is 1)
        if lin < total_work:
            group_idx = cutlass.Int32(0)
        work_tile = mixed_input_utils.ContiguousGroupWorkTileInfo(
            1, cta_coord_m, act_start, group_idx, distance_to_boundary
        )
        return work_tile, tile_l

    @cute.jit
    def _next_work_tile(
        self,
        tile_info_pipeline: pipeline.PipelineAsync,
        tile_info_consumer_state: pipeline.PipelineState,
        sTile_info: cute.Tensor,
    ):
        """Consume the next tile_info record (one group, so group index 1 = done);
        returns it with the advanced consumer state and the tile's K-slice."""
        tile_info_pipeline.consumer_wait(tile_info_consumer_state)
        work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
            1, sTile_info[(None, tile_info_consumer_state.index)]
        )
        tile_l = sTile_info[(4, tile_info_consumer_state.index)]
        cute.arch.fence_proxy("async.shared", space="cta")
        tile_info_pipeline.consumer_release(tile_info_consumer_state)
        tile_info_consumer_state.advance()
        # The advanced state is returned, not relied on as an in-place mutation: a
        # pipeline state handed to a @cute.jit helper is rebuilt from its MLIR values on
        # entry, so the caller's object does not see the advance (same reason
        # _transform_tile returns its states).
        return work_tile, tile_info_consumer_state, tile_l

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
        # One accumulator stage is the CTA tile's 128 lanes x N fp32 columns.  Counted
        # exactly, not rounded up to a power of two as the TMEM *allocation* is: the
        # 160- and 192-wide tiles would otherwise be charged 256 columns a stage and
        # fall back to a single stage, exposing the whole epilogue after every tile
        # (GB200: 2sm256x160 84 us vs 71 us for the C++, which keeps two stages).
        assert cta_tile_shape_mnk[0] == 128, "the accumulator spans all 128 TMEM lanes"
        num_tmem_acc_col_per_stage = cta_tile_shape_mnk[1]
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
            cute.size_in_bytes(cute.Int32, cute.make_layout((8, num_tile_info_stage)))
            + bytes_per_pipeline_stage * num_tile_info_stage
        )
        bias_bytes = 2 * 4 * cta_tile_shape_mnk[0]
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
            + bias_bytes
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
