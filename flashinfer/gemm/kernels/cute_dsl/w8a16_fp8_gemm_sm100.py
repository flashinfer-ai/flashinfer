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
TMEM, in the columns the accumulator stages leave.  The MMA warp issues tcgen05.mma
with A from TMEM and B from smem.
The epilogue applies the per-tensor scale (fp32, read on device) and the optional
per-N bias in fp32 and stores through TMA, which clips a partial last activation
tile of the final output; only a split-K partial tile that is not whole stores
through predicated SIMT (the slices' partials abut in memory).

Warp roles (16 warps): 0-3 epilogue, 4 MMA, 5 scheduler, 6 weight TMA, 7 activation
TMA, 8-15 transform.  Same structure as FlashInfer's dense W4A16 kernel
(``dense_gemm_bf16_fp4_sm100.py``), from which this file is derived, and as the
CUTLASS C++ mixed-input W8A16 kernel.
"""

from math import ceil, log2
from typing import NamedTuple, Optional

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
    convert_e4m3_fragment,
)


_FAST_DIVMOD_SHIFT = 40
# x * d below this bound makes the multiply-and-shift exact (see _fast_divmod_magic).
_FAST_DIVMOD_LIMIT = 1 << _FAST_DIVMOD_SHIFT
_INT64_MAX = (1 << 63) - 1


@cute.jit
def _fast_divmod_magic(d: cutlass.Int32, x_max: cutlass.Int32) -> cutlass.Int64:
    """The multiplier for ``_fast_divmod`` by ``d`` over dividends below ``x_max``:
    ceil(2^40 / d), with which x // d == (x * magic) >> 40 exactly whenever
    x * d < 2^40 (the rounding error is x * (magic * d - 2^40) / (d * 2^40), below
    one quotient step under that bound) and x * magic fits in 64 bits (magic is
    about 2^40 / d, so for small d the product overflows long before the first
    bound fails: at d == 1 from x = 2^23).  0 when either bound does not hold for
    every dividend: ``_fast_divmod`` then divides exactly."""
    d64 = d.to(cutlass.Int64)
    magic = (cutlass.Int64(_FAST_DIVMOD_LIMIT) + d64 - 1) // d64
    x_max64 = x_max.to(cutlass.Int64)
    overflow = x_max64 > cutlass.Int64(_INT64_MAX) // magic
    if x_max64 * d64 >= _FAST_DIVMOD_LIMIT or overflow:
        magic = cutlass.Int64(0)
    return magic


@cute.jit
def _fast_divmod(x: cutlass.Int32, d: cutlass.Int32, magic: cutlass.Int64):
    """(x // d, x % d) by a 64-bit multiply and shift instead of an integer division
    (~40 dependent instructions each on the GPU; the tile derivation runs it on every
    warp at kernel start and in the scheduler per tile).  ``magic == 0`` (the bound of
    ``_fast_divmod_magic`` not met) selects the exact division."""
    q = cutlass.Int32(0)
    if magic == 0:
        q = x // d
    else:
        q = ((x.to(cutlass.Int64) * magic) >> _FAST_DIVMOD_SHIFT).to(cutlass.Int32)
    return q, x - q * d


class _TmemBudget(NamedTuple):
    """TMEM split of one CTA (``_tmem_budget``)."""

    acc_stages: int  # accumulator stages (1 or 2)
    acc_cols: int  # TMEM columns of all accumulator stages
    a_cols_per_stage: int  # TMEM columns of one transformed-A stage
    transform_limit: int  # transform stages TMEM allows


def _aligned_tile_bytes(
    dtype: type[cutlass.Numeric], rows: int, cols: int, align: int
) -> int:
    """Bytes of a dense ``rows x cols`` tile of ``dtype`` in shared memory, rounded
    up to the buffer alignment."""
    return cute.round_up(
        cute.size_in_bytes(dtype, cute.make_layout((rows, cols))), align
    )


class _SmemBytes(NamedTuple):
    """Shared memory costs of one CTA (``_smem_stage_bytes``)."""

    carveout: int  # fixed: epilogue stages, tile-info ring, bias tile, barriers
    ab_load: int  # one raw-weight + activation load stage, with its barriers
    b_load: int  # one activation stage alone
    a_transform: int  # one transformed-A stage (barrier only when A is in TMEM)
    barrier_bytes: int
    c_stages: int
    tile_info_stages: int


class _Sm100Stages(NamedTuple):
    """Pipeline depths and TMEM columns of a Blackwell kernel variant."""

    load2trans: int  # raw weight + activation TMA stages feeding the transform
    trans2mma: int  # transformed-A stages feeding the MMA
    acc: int  # accumulator stages
    c: int  # epilogue smem stages
    b: int  # activation stages feeding the MMA (deeper than load2trans)
    tile_info: int  # scheduler ring depth
    acc_tmem_cols: int
    a_tmem_cols: int


class Sm100W8A16Fp8GemmKernel:
    """Warp-specialized persistent dense W8A16 GEMM for SM100 / SM103."""

    def __init__(
        self,
        mma_tiler_mnk: tuple[int, int, int],
        use_2cta_instrs: bool,
        cluster_shape_mn: tuple[int, int],
        has_bias: bool,
        swizzle_size: int,
        raster_along_m: bool,
        transform_fragment_size: int,
        enable_pdl: bool,
        partial_output: bool = False,
    ):
        # partial_output: split-K mode.  The L mode of the operands is the K-slice,
        # C is fp32 partials [n, m, splits] and the epilogue applies no scale / bias
        # (the reduce kernel does, see w8a16_fp8_splitk_reduce.py).
        self.partial_output = partial_output
        self.acc_dtype = cutlass.Float32
        self.mma_tiler = mma_tiler_mnk
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
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
        # Only the warps that touch TMEM wait for the allocation (epilogue, MMA,
        # transform); the TMA and scheduler warps start their work meanwhile.
        self.num_tmem_user_threads = 32 * (
            len(self.epilog_warp_id) + 1 + self.num_transform_warps
        )
        self.tmem_ptr_sync_barrier = pipeline.NamedBarrier(
            2, self.num_tmem_user_threads
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(5, 32)

        self.smem_buffer_align_bytes = 1024

    # ------------------------------------------------------------------ setup
    def _setup_attributes(self) -> cute.TiledMma:
        """Derive the MMA, pipeline depths and smem / TMEM layouts; returns the
        tiled MMA the launch builds its TMA atoms and kernel on."""
        # The transformed weight lives in TMEM: the kernel has no smem-A path, and the
        # helper agrees for the K-major weight the launcher always passes.
        self.transform_a_source = tcgen05.OperandSource.TMEM
        assert (
            mixed_input_utils.get_transform_a_source(self.a_major_mode)
            == tcgen05.OperandSource.TMEM
        ), "the transformed A operand must be TMEM-resident (K-major weight)"
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
        # 64 activation columns per epilogue subtile where the tile allows (the helper
        # picks 32 for a 16-bit C): half the subtiles, so half the named-barrier and
        # TMA-store round trips per tile, which is what the epilogue's time is.  The
        # smem per stage doubles and the stage count halves, same footprint.
        # (The helper's tile entries may be layouts rather than ints, hence cute.size.)
        if (
            self.c_dtype.width == 16
            and self.cta_tile_shape_mnk[1] % 64 == 0
            and cute.size(self.epi_tile[1]) < 64
        ):
            self.epi_tile = (self.epi_tile[0], 64)

        stages = self._compute_stages_and_tmem_cols(
            tiled_mma,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.c_layout,
            self.smem_buffer_align_bytes,
        )
        self.num_load2trans_stage = stages.load2trans
        self.num_trans2mma_stage = stages.trans2mma
        self.num_acc_stage = stages.acc
        self.num_c_stage = stages.c
        self.num_b_stage = stages.b
        self.num_tile_info_stage = stages.tile_info
        self.num_acc_tmem_cols = stages.acc_tmem_cols
        self.num_a_tmem_cols = stages.a_tmem_cols

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
            _,
        ) = mixed_input_utils.compute_smem_layout(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.num_load2trans_stage,
            self.num_trans2mma_stage,
        )
        # The B pipeline is deeper than the raw-weight one (see the stage computation).
        _, _, self.smem_layout_b = mixed_input_utils.compute_smem_layout(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.num_b_stage,
            self.num_trans2mma_stage,
        )
        return tiled_mma

    # ------------------------------------------------------------- transform
    @cute.jit
    def _setup_transform_partitions(
        self,
        tiled_mma: cute.TiledMma,
        copy_atom_a_input: cute.CopyAtom,
        copy_atom_a_transform: cute.CopyAtom,
        sA_input: cute.Tensor,
        a_smem_layout_transform: cute.ComposedLayout,
        accumulators: cute.Tensor,
        transform_tidx: cutlass.Int32,
    ):
        # The transformed A stages sit in TMEM right after the accumulator columns.
        tmem_ptr_transform = cute.recast_ptr(
            accumulators.iterator + self.num_acc_tmem_cols,
            dtype=self.mma_dtype,
        )
        transform_destination = cute.make_tensor(
            tmem_ptr_transform,
            tiled_mma.make_fragment_A(a_smem_layout_transform.outer).layout,
        )

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
        # the per-thread blocks sit in mode 3 (the 32x32b TMEM store repeats) or, when
        # that mode is 1, in mode 1, so halve whichever mode divides.
        # ``_transform_tile`` picks a half with a static index.
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
    ):
        """One tile's k-tiles for transform warpgroup ``group_idx``.

        The two transform warpgroups take alternate k-tiles of the kernel's k-tile
        sequence (group g the k-tiles whose global index has parity g; ``k_parity`` is
        the parity of this tile's first k-tile).  A k-tile's chain of smem loads,
        conversions, transformed store and fence is latency-bound, so with every warp
        on the same k-tile the MMA starved; alternating lets one group's chain overlap
        the other's.  Both pipeline states still step through every k-tile.

        Returns the advanced states."""
        a_load2trans_consumer_state.reset_count()
        trans2mma_producer_state.reset_count()
        for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            if (k_parity + k_tile) % 2 == group_idx:
                a_load2trans_pipeline.consumer_wait(a_load2trans_consumer_state)
                trans2mma_pipeline.producer_acquire(trans2mma_producer_state)
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
                # The raw stage goes back to the TMA producer once the converts have
                # consumed it; the transformed stores are fenced before the commit.
                a_load2trans_pipeline.consumer_release(a_load2trans_consumer_state)
                cute.arch.fence_view_async_tmem_store()
                trans2mma_pipeline.producer_commit(trans2mma_producer_state)
            a_load2trans_consumer_state.advance()
            trans2mma_producer_state.advance()

        return a_load2trans_consumer_state, trans2mma_producer_state

    @cute.jit
    def _create_pipelines(
        self,
        a_load2trans_bar: cute.Pointer,
        trans2mma_bar: cute.Pointer,
        b_load2mma_bar: cute.Pointer,
        acc_bar: cute.Pointer,
        tile_info_bar: cute.Pointer,
        cluster_layout_vmnk: cute.Layout,
        tidx: cutlass.Int32,
    ):
        """The five mbarrier pipelines of the kernel, on the shared-storage barriers:
        weight TMA -> transform warps, transform warps -> MMA, activation TMA -> MMA,
        MMA -> epilogue, scheduler warp -> everyone."""
        transform_thread_idx = (
            tidx - 32 * self.transform_warp_id[0]
            if tidx >= 32 * self.transform_warp_id[0]
            else tidx
        )
        a_load2trans_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=a_load2trans_bar,
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
            barrier_storage=trans2mma_bar,
            num_stages=self.num_trans2mma_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * (self.num_transform_warps // 2) * cta_v_size
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        b_load2mma_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=b_load2mma_bar,
            num_stages=self.num_b_stage,
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
            barrier_storage=acc_bar,
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, cta_v_size * len(self.epilog_warp_id)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=tile_info_bar,
            num_stages=self.num_tile_info_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_cta - 32
            ),
            defer_sync=True,
        )
        return (
            a_load2trans_pipeline,
            trans2mma_pipeline,
            b_load2mma_pipeline,
            acc_pipeline,
            tile_info_pipeline,
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
        return self(
            weight, activation, scale, bias, output, max_active_clusters, stream
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        c: cute.Tensor,
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

        tiled_mma = self._setup_attributes()

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
                cutlass.Int64, self.num_b_stage
            ]
            b_load2mma_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_b_stage
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

        # Magic multipliers for the two dynamic divisors of the tile derivation (see
        # _fast_divmod): computed once per launch, on the host.
        _, _, major_clusters, _, total_clusters, total_work = self._grid_dims(c)
        # Dividends: the work item index (below total_work) and the cluster tile
        # index within a slice (below total_clusters).
        fd_total = _fast_divmod_magic(total_clusters, total_work)
        fd_major = _fast_divmod_magic(major_clusters, total_clusters)
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
            fd_total,
            fd_major,
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
        fd_total: cutlass.Int64,
        fd_major: cutlass.Int64,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        a_smem_layout_transform: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, bidz = cute.arch.block_idx()
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
        (
            a_load2trans_pipeline,
            trans2mma_pipeline,
            b_load2mma_pipeline,
            acc_pipeline,
            tile_info_pipeline,
        ) = self._create_pipelines(
            storage.a_load2trans_full_mbar_ptr.data_ptr(),
            storage.a_trans2mma_full_mbar_ptr.data_ptr(),
            storage.b_load2mma_full_mbar_ptr.data_ptr(),
            storage.acc_full_mbar_ptr.data_ptr(),
            storage.tile_info_full_mbar_ptr.data_ptr(),
            cluster_layout_vmnk,
            tidx,
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

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        tmem.allocate(self.num_tmem_alloc_cols)
        uses_tmem = (
            (warp_idx != self.schedule_warp_id)
            & (warp_idx != self.weight_tma_warp_id)
            & (warp_idx != self.activation_tma_warp_id)
        )
        if uses_tmem:
            tmem.wait_for_alloc()
        # ---- scheduler warp: hands (cta_coord_m, activation offset, valid, rows left)
        if warp_idx == self.schedule_warp_id:
            self._scheduler_warp(
                act_in_cluster,
                bidz,
                cta_m_in_cluster,
                fd_major,
                fd_total,
                sTile_info,
                tensor_c,
                tile_info_pipeline,
            )

        # ---- activation TMA warp (the only role that waits on the preceding grid).
        if warp_idx == self.activation_tma_warp_id:
            self._activation_tma_warp(
                act_in_cluster,
                b_full_mcast_mask,
                b_load2mma_pipeline,
                cta_m_in_cluster,
                fd_major,
                fd_total,
                k_tile_cnt,
                sTile_info,
                tBgB,
                tBsB,
                tensor_c,
                tile_info_pipeline,
                tma_atom_b,
            )

        # ---- weight TMA warp.
        if warp_idx == self.weight_tma_warp_id:
            self._weight_tma_warp(
                a_full_mcast_mask,
                a_load2trans_pipeline,
                act_in_cluster,
                cta_m_in_cluster,
                fd_major,
                fd_total,
                k_tile_cnt,
                sTile_info,
                tAgA,
                tAsA,
                tensor_c,
                tile_info_pipeline,
                tiled_mma,
                tma_atom_a,
            )

        # ---- transform warps.
        if warp_idx >= self.transform_warp_id[0]:
            self._transform_warps(
                a_load2trans_pipeline,
                a_smem_layout_transform,
                act_in_cluster,
                copy_atom_a_input,
                copy_atom_a_transform,
                cta_m_in_cluster,
                fd_major,
                fd_total,
                k_tile_cnt,
                sA_input,
                sTile_info,
                tCtAcc_fake,
                tensor_c,
                tidx,
                tile_info_pipeline,
                tiled_mma,
                tmem,
                trans2mma_pipeline,
                warp_idx,
            )

        # ---- MMA warp.
        if warp_idx == self.mma_warp_id:
            self._mma_warp(
                a_smem_layout_transform,
                acc_pipeline,
                act_in_cluster,
                b_load2mma_pipeline,
                cta_m_in_cluster,
                fd_major,
                fd_total,
                is_leader_cta,
                k_tile_cnt,
                sTile_info,
                tCrB,
                tCtAcc_fake,
                tensor_c,
                tile_info_pipeline,
                tiled_mma,
                tmem,
                trans2mma_pipeline,
            )

        # ---- epilogue warps: D = scale * acc (+ bias[row]), cast, TMA store.
        if warp_idx < self.mma_warp_id:
            self._epilogue_warps(
                acc_pipeline,
                act_in_cluster,
                bias,
                cta_m_in_cluster,
                epi_tile,
                fd_major,
                fd_total,
                sBias,
                sC,
                sTile_info,
                scale,
                tCgC,
                tCgC_simt,
                tCtAcc_fake,
                tensor_c,
                tidx,
                tile_info_pipeline,
                tiled_mma,
                tma_atom_c,
                tmem,
                warp_idx,
            )

        if cutlass.const_expr(self.enable_pdl):
            if warp_idx == self.mma_warp_id:
                with cute.arch.elect_one():
                    griddepcontrol_launch_dependents()

    @cute.jit
    def _scheduler_warp(
        self,
        act_in_cluster: cutlass.Int32,
        bidz: cutlass.Int32,
        cta_m_in_cluster: cutlass.Int32,
        fd_major: cutlass.Int64,
        fd_total: cutlass.Int64,
        sTile_info: cute.Tensor,
        tensor_c: cute.Tensor,
        tile_info_pipeline: pipeline.PipelineAsync,
    ):
        """Scheduler warp: hands (cta_coord_m, activation offset, valid, rows left)
        to every other warp through the tile_info pipeline."""
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
                fd_total,
                fd_major,
            )
            act_tile_idx = act_start_l // self.cta_tile_shape_mnk[1]
            is_dense_tile = has_work and (act_tile_idx < act_tiles)
            if is_dense_tile or not has_work:
                tile_info_pipeline.producer_acquire(tile_info_producer_state)
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

    @cute.jit
    def _activation_tma_warp(
        self,
        act_in_cluster: cutlass.Int32,
        b_full_mcast_mask: Optional[cutlass.Int16],
        b_load2mma_pipeline: pipeline.PipelineTmaUmma,
        cta_m_in_cluster: cutlass.Int32,
        fd_major: cutlass.Int64,
        fd_total: cutlass.Int64,
        k_tile_cnt: cutlass.Int32,
        sTile_info: cute.Tensor,
        tBgB: cute.Tensor,
        tBsB: cute.Tensor,
        tensor_c: cute.Tensor,
        tile_info_pipeline: pipeline.PipelineAsync,
        tma_atom_b: cute.CopyAtom,
    ):
        """Activation TMA warp (the only role that waits on the preceding grid)."""
        cute.arch.setmaxregister_decrease(self.num_regs_generic_warps)
        tile_info_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
        )
        work_tile, tile_l = self._first_work_tile(
            tensor_c, cta_m_in_cluster, act_in_cluster, fd_total, fd_major
        )
        b_load2mma_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_b_stage
        )
        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_wait()
        while work_tile.is_valid_tile:
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
                # Peek first: a free stage then skips the blocking wait and the
                # acquire is just the transaction-count arrive.
                b_stage_free = b_load2mma_pipeline.producer_try_acquire(
                    b_load2mma_producer_state
                )
                # Tight poll (no suspend) until the stage is free: the MMA frees B
                # stages one k-tile at a time and the reload must go out at once,
                # or the MMA starves on B.
                while not b_stage_free:
                    b_stage_free = b_load2mma_pipeline.producer_try_acquire(
                        b_load2mma_producer_state
                    )
                b_load2mma_pipeline.producer_acquire(
                    b_load2mma_producer_state, b_stage_free
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, k_tile_base + b_load2mma_producer_state.count)],
                    tBsB[(None, b_load2mma_producer_state.index)],
                    tma_bar_ptr=b_load2mma_pipeline.producer_get_barrier(
                        b_load2mma_producer_state
                    ),
                    mcast_mask=b_full_mcast_mask,
                )
                b_load2mma_pipeline.producer_commit(b_load2mma_producer_state)
                b_load2mma_producer_state.advance()
            work_tile, tile_info_consumer_state, tile_l = self._next_work_tile(
                tile_info_pipeline, tile_info_consumer_state, sTile_info
            )
        b_load2mma_pipeline.producer_tail(b_load2mma_producer_state)

    @cute.jit
    def _weight_tma_warp(
        self,
        a_full_mcast_mask: Optional[cutlass.Int16],
        a_load2trans_pipeline: pipeline.PipelineTmaAsync,
        act_in_cluster: cutlass.Int32,
        cta_m_in_cluster: cutlass.Int32,
        fd_major: cutlass.Int64,
        fd_total: cutlass.Int64,
        k_tile_cnt: cutlass.Int32,
        sTile_info: cute.Tensor,
        tAgA: cute.Tensor,
        tAsA: cute.Tensor,
        tensor_c: cute.Tensor,
        tile_info_pipeline: pipeline.PipelineAsync,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
    ):
        """Weight TMA warp."""
        cute.arch.setmaxregister_decrease(self.num_regs_generic_warps)
        tile_info_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
        )
        work_tile, tile_l = self._first_work_tile(
            tensor_c, cta_m_in_cluster, act_in_cluster, fd_total, fd_major
        )
        a_load2trans_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_load2trans_stage
        )
        while work_tile.is_valid_tile:
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
                while not peek_load2trans_empty_status:
                    peek_load2trans_empty_status = (
                        a_load2trans_pipeline.producer_try_acquire(
                            a_load2trans_producer_state
                        )
                    )
                a_load2trans_pipeline.producer_acquire(
                    a_load2trans_producer_state, peek_load2trans_empty_status
                )
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, k_tile_base + a_load2trans_producer_state.count)],
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
            work_tile, tile_info_consumer_state, tile_l = self._next_work_tile(
                tile_info_pipeline, tile_info_consumer_state, sTile_info
            )
        a_load2trans_pipeline.producer_tail(a_load2trans_producer_state)

    @cute.jit
    def _transform_warps(
        self,
        a_load2trans_pipeline: pipeline.PipelineTmaAsync,
        a_smem_layout_transform: cute.ComposedLayout,
        act_in_cluster: cutlass.Int32,
        copy_atom_a_input: cute.CopyAtom,
        copy_atom_a_transform: cute.CopyAtom,
        cta_m_in_cluster: cutlass.Int32,
        fd_major: cutlass.Int64,
        fd_total: cutlass.Int64,
        k_tile_cnt: cutlass.Int32,
        sA_input: cute.Tensor,
        sTile_info: cute.Tensor,
        tCtAcc_fake: cute.Tensor,
        tensor_c: cute.Tensor,
        tidx: cutlass.Int32,
        tile_info_pipeline: pipeline.PipelineAsync,
        tiled_mma: cute.TiledMma,
        tmem: utils.TmemAllocator,
        trans2mma_pipeline: pipeline.PipelineAsyncUmma,
        warp_idx: cutlass.Int32,
    ):
        """Transform warps."""
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
            accumulators,
            transform_local_tidx,
        )
        tile_info_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
        )
        work_tile, _ = self._first_work_tile(
            tensor_c, cta_m_in_cluster, act_in_cluster, fd_total, fd_major
        )
        a_load2trans_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_load2trans_stage
        )
        trans2mma_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_trans2mma_stage
        )
        # Parity of the global index of this tile's first k-tile (see _transform_tile).
        xf_k_parity = cutlass.Int32(0)
        while work_tile.is_valid_tile:
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
                    split_mode,
                    k_tile_cnt,
                    xf_k_parity,
                    transform_group_idx,
                )
            )
            xf_k_parity = (xf_k_parity + k_tile_cnt) % 2
            work_tile, tile_info_consumer_state, _ = self._next_work_tile(
                tile_info_pipeline, tile_info_consumer_state, sTile_info
            )
        trans2mma_pipeline.producer_tail(trans2mma_producer_state)

    @cute.jit
    def _mma_warp(
        self,
        a_smem_layout_transform: cute.ComposedLayout,
        acc_pipeline: pipeline.PipelineUmmaAsync,
        act_in_cluster: cutlass.Int32,
        b_load2mma_pipeline: pipeline.PipelineTmaUmma,
        cta_m_in_cluster: cutlass.Int32,
        fd_major: cutlass.Int64,
        fd_total: cutlass.Int64,
        is_leader_cta: cutlass.Boolean,
        k_tile_cnt: cutlass.Int32,
        sTile_info: cute.Tensor,
        tCrB: cute.Tensor,
        tCtAcc_fake: cute.Tensor,
        tensor_c: cute.Tensor,
        tile_info_pipeline: pipeline.PipelineAsync,
        tiled_mma: cute.TiledMma,
        tmem: utils.TmemAllocator,
        trans2mma_pipeline: pipeline.PipelineAsyncUmma,
    ):
        """MMA warp."""
        cute.arch.setmaxregister_decrease(self.num_regs_generic_warps)
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        accumulators = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        tmem_ptr_transform = cute.recast_ptr(
            accumulators.iterator + self.num_acc_tmem_cols, dtype=self.mma_dtype
        )
        tCrA = cute.make_tensor(
            tmem_ptr_transform,
            tiled_mma.make_fragment_A(a_smem_layout_transform.outer).layout,
        )
        tile_info_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
        )
        work_tile, _ = self._first_work_tile(
            tensor_c, cta_m_in_cluster, act_in_cluster, fd_total, fd_major
        )
        trans2mma_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_trans2mma_stage
        )
        b_load2mma_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_b_stage
        )
        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_acc_stage
        )
        while work_tile.is_valid_tile:
            tCtAcc = accumulators[(None, None, None, acc_producer_state.index)]
            b_load2mma_consumer_state.reset_count()
            trans2mma_consumer_state.reset_count()
            peek_trans2mma_full_status = cutlass.Boolean(1)
            if is_leader_cta:
                if trans2mma_consumer_state.count < k_tile_cnt:
                    peek_trans2mma_full_status = trans2mma_pipeline.consumer_try_wait(
                        trans2mma_consumer_state
                    )
                acc_pipeline.producer_acquire(acc_producer_state)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    trans2mma_pipeline.consumer_wait(
                        trans2mma_consumer_state, peek_trans2mma_full_status
                    )
                    b_load2mma_pipeline.consumer_wait(b_load2mma_consumer_state)
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
            work_tile, tile_info_consumer_state, _ = self._next_work_tile(
                tile_info_pipeline, tile_info_consumer_state, sTile_info
            )
        acc_pipeline.producer_tail(acc_producer_state)

    @cute.jit
    def _epilogue_warps(
        self,
        acc_pipeline: pipeline.PipelineUmmaAsync,
        act_in_cluster: cutlass.Int32,
        bias: cute.Tensor,
        cta_m_in_cluster: cutlass.Int32,
        epi_tile: cute.Tile,
        fd_major: cutlass.Int64,
        fd_total: cutlass.Int64,
        sBias: cute.Tensor,
        sC: cute.Tensor,
        sTile_info: cute.Tensor,
        scale: cute.Tensor,
        tCgC: cute.Tensor,
        tCgC_simt: cute.Tensor,
        tCtAcc_fake: cute.Tensor,
        tensor_c: cute.Tensor,
        tidx: cutlass.Int32,
        tile_info_pipeline: pipeline.PipelineAsync,
        tiled_mma: cute.TiledMma,
        tma_atom_c: cute.CopyAtom,
        tmem: utils.TmemAllocator,
        warp_idx: cutlass.Int32,
    ):
        """Epilogue warps: D = scale * acc (+ bias[row]), cast, TMA store."""
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
            tensor_c, cta_m_in_cluster, act_in_cluster, fd_total, fd_major
        )
        weight_rows = cutlass.Int32(cute.size(tensor_c.shape[0]))
        act_rows = cutlass.Int32(cute.size(tensor_c.shape[1]) // self.splits)
        num_prev_subtiles = cutlass.Int32(0)
        assert self.cta_tile_shape_mnk[0] == 128, (
            "the bias staging uses the 128 epilogue threads, one per CTA-tile row"
        )
        epi_tile_count = cutlass.Int32(0)
        while work_tile.is_valid_tile:
            num_prev_subtiles, acc_consumer_state = self._epilogue_tile(
                work_tile,
                tile_l,
                epi_tile_count,
                num_prev_subtiles,
                epi_tidx,
                warp_idx,
                weight_rows,
                act_rows,
                acc_pipeline,
                acc_consumer_state,
                c_pipeline,
                tiled_mma,
                tiled_copy_t2r,
                tiled_copy_r2s,
                tma_atom_c,
                simt_atom,
                tensor_c,
                scale,
                bias,
                sBias,
                bSG_sC,
                bSG_gC_partitioned,
                tTR_gC_partitioned,
                tTR_tAcc_base,
                tTR_rAcc,
                tTR_rC,
                tRS_rC,
                tRS_sC,
                m_thr_offset,
            )
            epi_tile_count += 1
            work_tile, tile_info_consumer_state, tile_l = self._next_work_tile(
                tile_info_pipeline, tile_info_consumer_state, sTile_info
            )

        tmem.relinquish_alloc_permit()
        self.epilog_sync_barrier.arrive_and_wait()
        tmem.free(tmem_ptr)
        c_pipeline.producer_tail()

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
        fd_total: cutlass.Int64,
        fd_major: cutlass.Int64,
    ):
        """Work item ``lin`` of this CTA: (cta_coord_m, activation row offset, rows to
        the activation end, K-slice).  Slices are the slowest mode; within a slice the
        linear cluster-tile index maps to (major, minor), through the L2 group remap of
        CUTLASS's swizzle_and_rasterize (groups of swizzle_size clusters along the major
        axis, residual tiles not remapped), then to (m, n)."""
        tile_l, lin_mn = _fast_divmod(lin, total_clusters, fd_total)
        minor, major = _fast_divmod(lin_mn, major_clusters, fd_major)
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
        fd_total: cutlass.Int64,
        fd_major: cutlass.Int64,
    ):
        """The CTA's first work item, derived from its cluster index by every warp role
        itself (the scheduler warp publishes from the second item on): saves the
        scheduler-to-consumer hop at kernel start."""
        _, _, bidz = cute.arch.block_idx()
        lin = cutlass.Int32(bidz)
        (
            act_rows,
            _,
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
            fd_total,
            fd_major,
        )
        group_idx = cutlass.Int32(1)  # 1 = no work (group_count is 1)
        if lin < total_work:
            group_idx = cutlass.Int32(0)
        work_tile = mixed_input_utils.ContiguousGroupWorkTileInfo(
            1, cta_coord_m, act_start, group_idx, distance_to_boundary
        )
        return work_tile, tile_l

    @cute.jit
    def _epilogue_tile(
        self,
        work_tile: mixed_input_utils.ContiguousGroupWorkTileInfo,
        tile_l: cutlass.Int32,
        epi_tile_count: cutlass.Int32,
        num_prev_subtiles: cutlass.Int32,
        epi_tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        weight_rows: cutlass.Int32,
        act_rows: cutlass.Int32,
        acc_pipeline: pipeline.PipelineUmmaAsync,
        acc_consumer_state: pipeline.PipelineState,
        c_pipeline: pipeline.PipelineTmaStore,
        tiled_mma: cute.TiledMma,
        tiled_copy_t2r: cute.TiledCopy,
        tiled_copy_r2s: cute.TiledCopy,
        tma_atom_c: cute.CopyAtom,
        simt_atom: cute.CopyAtom,
        tensor_c: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        sBias: cute.Tensor,
        bSG_sC: cute.Tensor,
        bSG_gC_partitioned: cute.Tensor,
        tTR_gC_partitioned: cute.Tensor,
        tTR_tAcc_base: cute.Tensor,
        tTR_rAcc: cute.Tensor,
        tTR_rC: cute.Tensor,
        tRS_rC: cute.Tensor,
        tRS_sC: cute.Tensor,
        m_thr_offset: cute.Tensor,
    ):
        """One output tile of the epilogue warps: wait for its accumulator stage, scale
        (and bias) in fp32, cast, and store it subtile by subtile through smem and the
        TMA (a partial last activation tile of split-K partials goes through the
        predicated SIMT store).  Returns the advanced subtile count and accumulator
        consumer state (a pipeline state is passed into a jit helper by value, so
        the caller must take it back)."""
        m_base = work_tile.cta_coord_m * self.cta_tile_shape_mnk[0]
        # The tile's activation offset within C: its K-slice's rows follow
        # the previous slices' (split-K partials).
        coord_n = work_tile.coord_n + tile_l * act_rows
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

        acc_pipeline.consumer_wait(acc_consumer_state)
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
        # Unrolled (the count is static); the accumulator load is not pipelined
        # across subtiles: a subtile's cost is the smem write, the two named
        # barriers and the TMA store issue, not the TMEM load.
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
                        scale_val * tTR_rAcc[i] + sBias[(m_thr_slice[(i)][0], bias_buf)]
                    )
            else:
                tTR_rAcc.store(scale_val * tTR_rAcc.load())
            # A partial last activation tile goes through the TMA store as well
            # when the output is the final one: the TMA clips the rows past the
            # tensor's extent.  Only split-K partials, whose slices abut in
            # memory, need the predicated SIMT store (a clipped TMA store would
            # run into the next slice).  The SIMT path is much slower per tile, so
            # it is kept to that case.
            tma_store_ok = work_tile.distance_to_boundary >= self.cta_tile_shape_mnk[1]
            if cutlass.const_expr(self.splits == 1):
                tma_store_ok = cutlass.Boolean(True)
            if tma_store_ok:
                acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                tRS_rC.store(acc_vec.to(self.c_dtype))
                num_prev_subtiles += 1
                c_buffer = num_prev_subtiles % self.num_c_stage
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                cute.arch.fence_proxy("async.shared", space="cta")
                # One CTA barrier per subtile: the storing warp first waits for
                # the previous TMA store to finish reading its buffer (the one
                # the next subtile writes); after the barrier everyone may write
                # it and this subtile's store is issued.
                if warp_idx == self.epilog_warp_id[0]:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                self.epilog_sync_barrier.arrive_and_wait()
                if warp_idx == self.epilog_warp_id[0]:
                    cute.copy(
                        tma_atom_c,
                        bSG_sC[(None, c_buffer)],
                        bSG_gC[(None, subtile_idx)],
                    )
                    c_pipeline.producer_commit()
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
        return num_prev_subtiles, acc_consumer_state

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
    def _tmem_budget(
        tiled_mma: cute.TiledMma,
        cta_tile_shape_mnk: tuple[int, int, int],
    ) -> "_TmemBudget":
        """Accumulator stages and TMEM columns: two accumulator stages (the epilogue
        of one tile then overlaps the mainloop of the next); the transform stages
        take the columns the accumulators leave."""
        # One accumulator stage is the CTA tile's 128 lanes x N fp32 columns.  Counted
        # exactly, not rounded up to a power of two as the TMEM *allocation* is: the
        # 160- and 192-wide tiles would otherwise be charged 256 columns a stage and
        # fall back to a single stage, exposing the whole epilogue after every tile.
        assert cta_tile_shape_mnk[0] == 128, "the accumulator spans all 128 TMEM lanes"
        acc_cols_per_stage = cta_tile_shape_mnk[1]
        tmem_columns = cute.arch.get_max_tmem_alloc_cols("sm_100")
        acc_stages = 2
        elts_per_col = 32 // tiled_mma.op.a_dtype.width
        a_cols_per_stage = cute.round_up(cta_tile_shape_mnk[2] // elts_per_col, 4)
        acc_cols = cute.round_up(acc_stages * acc_cols_per_stage, 4)
        transform_limit = (tmem_columns - acc_cols) // a_cols_per_stage
        if transform_limit <= 0:
            raise ValueError("Not enough TMEM capacity for selected tile size")
        return _TmemBudget(acc_stages, acc_cols, a_cols_per_stage, transform_limit)

    @staticmethod
    def _smem_stage_bytes(
        tiled_mma: cute.TiledMma,
        cta_tile_shape_mnk: tuple[int, int, int],
        epi_tile: cute.Tile,
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        smem_buffer_align_bytes: int,
        acc_stages: int,
    ) -> "_SmemBytes":
        """Shared memory per pipeline stage of each kind, and the fixed carve-out
        (epilogue stages, tile-info ring, bias tile, accumulator barriers)."""
        barrier_bytes = 16
        tile_info_stages = 3
        tile_info_bytes = (
            cute.size_in_bytes(cute.Int32, cute.make_layout((8, tile_info_stages)))
            + barrier_bytes * tile_info_stages
        )
        bias_bytes = 2 * 4 * cta_tile_shape_mnk[0]
        c_stages = 2
        c_layout_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
        c_bytes = cute.size_in_bytes(c_dtype, c_layout_one) * c_stages
        carveout = barrier_bytes * acc_stages + c_bytes + tile_info_bytes + bias_bytes
        # Per-stage operand tiles: the whole weight tile (rows x K) as loaded, and this
        # CTA's share of the activation tile (the MMA's CTA pair splits the activation
        # rows).  The transformed weight lives in TMEM: its stage costs smem only
        # the barrier.
        weight_rows, act_rows, tile_k = cta_tile_shape_mnk
        act_rows_per_cta = act_rows // cute.size(tiled_mma.thr_id)
        align = smem_buffer_align_bytes
        a_load = _aligned_tile_bytes(a_dtype, weight_rows, tile_k, align)
        b_load = _aligned_tile_bytes(b_dtype, act_rows_per_cta, tile_k, align)
        ab_load = a_load + b_load + 2 * barrier_bytes
        a_transform = barrier_bytes
        return _SmemBytes(
            carveout,
            ab_load,
            b_load,
            a_transform,
            barrier_bytes,
            c_stages,
            tile_info_stages,
        )

    @staticmethod
    def _compute_stages_and_tmem_cols(
        tiled_mma: cute.TiledMma,
        cta_tile_shape_mnk: tuple[int, int, int],
        epi_tile: cute.Tile,
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        smem_buffer_align_bytes: int,
    ) -> "_Sm100Stages":
        """Fit the load, transform, accumulator and epilogue pipelines into smem and
        TMEM: the TMEM budget, the per-stage smem costs, then the stage counts.  The
        transform pipeline takes as many stages as TMEM and smem allow, the raw load
        pipeline what smem then leaves; both need two.  Smem left after that deepens
        the activation (B) pipeline alone: the TMA latency of a first-touch k-tile
        exceeds several stages' worth of MMA time, while two epilogue stages are all
        the epilogue overlaps."""
        tmem = Sm100W8A16Fp8GemmKernel._tmem_budget(tiled_mma, cta_tile_shape_mnk)
        smem = Sm100W8A16Fp8GemmKernel._smem_stage_bytes(
            tiled_mma,
            cta_tile_shape_mnk,
            epi_tile,
            a_dtype,
            b_dtype,
            c_dtype,
            c_layout,
            smem_buffer_align_bytes,
            tmem.acc_stages,
        )
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        free = smem_capacity - smem.carveout
        trans2mma_by_smem = free // (smem.ab_load + smem.a_transform)
        trans2mma_stages = min(tmem.transform_limit, trans2mma_by_smem)
        load2trans_stages = (free - trans2mma_stages * smem.a_transform) // smem.ab_load
        if load2trans_stages < 2 or trans2mma_stages < 2 or tmem.acc_stages < 1:
            raise ValueError("Not enough SMEM or TMEM capacity for selected tile size")
        leftover = (
            free
            - load2trans_stages * smem.ab_load
            - trans2mma_stages * smem.a_transform
        )
        b_stages = load2trans_stages + leftover // (
            smem.b_load + 2 * smem.barrier_bytes
        )
        return _Sm100Stages(
            load2trans_stages,
            trans2mma_stages,
            tmem.acc_stages,
            smem.c_stages,
            b_stages,
            smem.tile_info_stages,
            tmem.acc_cols,
            trans2mma_stages * tmem.a_cols_per_stage,
        )
