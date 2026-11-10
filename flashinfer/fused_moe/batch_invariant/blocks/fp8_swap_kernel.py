# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The swapped body of the fp8 per-tensor MoE classes: the bf16 swapped body (:mod:`.swap_bf16`) on e4m3 operands with the W8A8
chains in the epilogue -- ``C^T`` per expert, the expert's weight rows on the MMA's M side (a 128- or 64-row tile streamed by TMA
with the expert as the batch coordinate), the group's tokens on the N side as a narrow token tile (16 or 32 rows: ``kind::f8f6f4``
takes N >= 16) read as a TMA box from the dense permuted e4m3 rows at the group's row. One TMA pipeline carries the weight tile and
the token box of one k-tile per stage (eight K32 steps: two swizzle atoms of K), the dense ``mma_loop`` folds them, and the
transposed group-masked epilogue applies the W8A8 chain (the per-expert alpha, the fused e4m3 quantization of GEMM1, the scaled
conversion of GEMM2) and stores token ``j`` of the tile as output row ``row0 + j``.

The same class: the tcgen05 step is symmetric in its operands, so every output element's fold is the dense fp8 class's fold on the
same operands; the chain per element is the grouped fp8 body's. Two TMEM layouts: the 128-row tile gives a thread one weight row and
its token columns (the gate accumulators of the interleaved weight's up rows come from lane ``+8``), the 64-row tile gives a thread
the rows ``r`` and ``r + 8`` -- the up / gate pair itself -- at every fourth token column (the 16-lane data paths), so the pairing
needs no shuffle; both are placements (the row a thread owns and the register that holds its gate), the term per element unchanged.
The 64-row weight tile doubles the CTAs of a small weight at decode.

Operands: ``rows (R, K, 1)`` K-major e4m3 (the dense permuted rows; ``R >= Tk``, with ``R >= Tk + token_tile`` the last tiles'
boxes stay inside the buffer, else they are zero-filled past it), ``w (N, K, E)`` K-major e4m3, ``out (Tk, Nout)`` row-major (the
uint8 bytes of e4m3 for the GEMM1 chains, bf16 / f32 for GEMM2), ``counts (E,)`` int32, ``alpha (E,)`` fp32 (``s_w[e]``), ``sa (1,)``
fp32 (``s_x`` / ``s_inter``), ``si (1,)`` fp32 (``m_inter``: GEMM1's ``inv``, read as given).
"""

from __future__ import annotations

from typing import List

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Float32, Int32, Uint8
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum
from cutlass.utils.blackwell_helpers import get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....batch_invariant.quantization.cute_ext import select_f32
from ....batch_invariant.quantization.cvt_fp8 import cvt_f32x2_to_fp8x2
from ....gemm.batch_invariant.blocks.mainloop import mma_loop
from ....gemm.batch_invariant.blocks.operands import (
    ACC_DTYPE,
    BAR_TMEM_ALLOC,
    IN_DT,
    StaticPlan,
)
from ....gemm.batch_invariant.generator import TILES_N_NARROW, Knobs, MainloopBlock
from ..spec_fp8 import w8a8_alpha
from .epilogue import BAR_EPI_JOIN, GROUP, MoeF32, fragment_offsets
from ..spec_fp8 import pertoken_elem_scale
from .fp8_epilogue import DEAD_ACC, Fp8Chain, _chain_elem
from .kernel import GROUPED_K_BLOCKS
from .scheduler import MAX_E, TABLE_SMEM_BYTES, GroupScheduler, build_group_tables
from .swap_bf16 import plan_knobs_of, swap_tma_producer_loop

IN_FMT = "e4m3"
EPILOGUE_WARPS = (
    0,
    1,
    2,
    3,
)  # one warpgroup: the tile's weight rows (one per thread, or the (r, r + 8) pair)
MMA_WARP = 4
TMA_WARP = 5
THREADS = 32 * (TMA_WARP + 1)
#: The swapped records' weight tiles (the MMA's M): the 128-row tile, or the 64-row one (twice the CTAs of a small weight).
WEIGHT_TILES = (128, 64)
#: The swapped records' token tiles: the narrow dense tiles the fp8 instruction admits (N >= 16).
TOKEN_TILES = tuple(t for t in TILES_N_NARROW if t >= 16)


def swap_knobs(
    token_tile: int = TOKEN_TILES[0],
    raster: str = "m",
    weight_tile: int = WEIGHT_TILES[0],
) -> Knobs:
    """The fp8 classes' swapped record with ``token_tile`` tokens on the MMA's N side and ``weight_tile`` weight rows on its M
    side (1-CTA, no cluster)."""
    assert token_tile in TOKEN_TILES and weight_tile in WEIGHT_TILES, (
        token_tile,
        weight_tile,
    )
    return Knobs(
        (weight_tile, token_tile), (1, 1), False, True, raster, 1, swap_ab=True
    )


def swapped_knobs() -> List[Knobs]:
    """The swapped record space of the fp8 classes: one record per (weight tile, token tile)."""
    return [swap_knobs(tn, weight_tile=wt) for wt in WEIGHT_TILES for tn in TOKEN_TILES]


def is_swapped_record(knobs: Knobs) -> bool:
    """A record the fp8 swapped body takes: the swapped 1-CTA record at one of the weight tiles and token tiles, no cluster,
    swizzle 1, no in-cluster split, no in-CTA chunk fold, the TMA store flag of every record."""
    return (
        knobs.swap_ab
        and knobs.tile_m in WEIGHT_TILES
        and knobs.tile_n in TOKEN_TILES
        and knobs.cluster_shape_mn == (1, 1)
        and not knobs.use_2cta_instrs
        and knobs.swizzle_size == 1
        and knobs.cluster_k == 1
        and not knobs.cta_chunks
        and knobs.use_tma_store
    )


class SwappedGroupedGemmFp8:
    """``out[p, n] = chain(sum_k rows[p, k] w[n, k, e(p)], alpha = mul(sa[0], alpha[e(p)]))`` for every dense row ``p`` of every
    group, the weights as the MMA's A and the group's rows as a narrow TMA box on its B side. The persistent grid
    (``max_active_clusters``, in CTAs) is a dynamic launch argument: which CTA computes a tile, never the bits."""

    def __init__(self, knobs: Knobs, chain: Fp8Chain):
        assert isinstance(chain, Fp8Chain), chain
        assert is_swapped_record(knobs), (
            "the fp8 swapped body takes the swapped records (128 or 64 weight rows x 16 / 32 tokens, 1-CTA)",
            knobs,
        )
        self.knobs, self.chain = knobs, chain
        self.token_tile, self.weight_tile = knobs.tile_n, knobs.tile_m

    @cute.jit
    def __call__(
        self,
        rows: cute.Tensor,
        w: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        alpha: cute.Tensor,
        sa: cute.Tensor,
        si: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(rows.element_type != w.element_type):
            raise TypeError(
                f"rows and w must have one type: {rows.element_type} != {w.element_type}"
            )
        assert rows.element_type is IN_DT[IN_FMT], rows.element_type
        st = StaticPlan(
            plan_knobs_of(self.knobs),
            w.element_type,
            rows.element_type,
            ACC_DTYPE,
            LayoutEnum.from_tensor(w).mma_major_mode(),
            LayoutEnum.from_tensor(rows).mma_major_mode(),
            LayoutEnum.ROW_MAJOR,
            c_in_smem=False,
            extra_smem_bytes=TABLE_SMEM_BYTES,
            k_blocks=GROUPED_K_BLOCKS,
            in_fmt=IN_FMT,
        )
        self.st = st
        self.block = MainloopBlock(IN_FMT, inner_steps=GROUPED_K_BLOCKS)
        self.lowering = self.block.lowering()
        assert st.mma_tiler[2] == self.block.k_block, (st.mma_tiler, self.block.k_block)
        assert (
            st.cta_tile_shape_mnk[0] == self.weight_tile
            and st.cta_tile_shape_mnk[1] == self.token_tile
        ), st.cta_tile_shape_mnk
        # the weights as the A operand: (M = N, K, L = E), TMA with the expert as the batch coordinate
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        a_one = cute.select(
            st.a_smem_layout_staged,
            mode=list(range(cute.rank(st.a_smem_layout_staged) - 1)),
        )
        atom_w, tensor_w = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w, a_one, st.mma_tiler, st.tiled_mma, st.cluster_layout_vmnk.shape
        )
        # the token rows as the B operand: a box of token_tile rows x one k-tile, re-addressed per tile at the group's row
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        b_one = cute.select(
            st.b_smem_layout_staged,
            mode=list(range(cute.rank(st.b_smem_layout_staged) - 1)),
        )
        atom_tok, tensor_tok = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, rows, b_one, st.mma_tiler, st.tiled_mma, st.cluster_layout_vmnk.shape
        )
        self.num_tma_load_bytes = cute.size_in_bytes(
            st.a_dtype, a_one
        ) + cute.size_in_bytes(st.b_dtype, b_one)
        self.kernel(
            st.tiled_mma,
            atom_w,
            tensor_w,
            atom_tok,
            tensor_tok,
            out,
            counts,
            alpha,
            sa,
            si,
            st.cluster_layout_vmnk,
            st.a_smem_layout_staged,
            st.b_smem_layout_staged,
            st.epi_tile,
        ).launch(
            grid=(1, 1, max_active_clusters),
            block=[THREADS, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_w: cute.CopyAtom,
        mW_mkl: cute.Tensor,
        tma_atom_tok: cute.CopyAtom,
        mTok_nkl: cute.Tensor,
        mOut: cute.Tensor,
        counts: cute.Tensor,
        mAlpha: cute.Tensor,
        mSa: cute.Tensor,
        mSi: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
    ):
        st = self.st
        tn, wt = self.token_tile, self.weight_tile
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_tok)
        # programmatic dependent launch: as the bf16 swapped body -- the wait for the predecessor grid sits in each warp before
        # its first read of the predecessor's output (the token rows); the weight loads and the group tables do not wait; the
        # trigger placement is the chain's (``pdl.TRIGGER``)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        bidx, _, _ = cute.arch.block_idx()
        is_leader_cta = (
            bidx % cute.size(tiled_mma.thr_id.shape) == 0
        )  # every CTA (the 1-CTA instruction)
        tidx, _, _ = cute.arch.thread_idx()
        E = counts.shape[0]
        N = mW_mkl.shape[0]
        Tk = mOut.shape[0]

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            group_off: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
            group_pre: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # one TMA pipeline: the weight tile and the token box of one k-tile per stage
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=st.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=st.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(EPILOGUE_WARPS)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=BAR_TMEM_ALLOC, num_threads=32 * (len(EPILOGUE_WARPS) + 1)
        )
        epi_join = pipeline.NamedBarrier(
            barrier_id=BAR_EPI_JOIN, num_threads=32 * len(EPILOGUE_WARPS)
        )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=EPILOGUE_WARPS[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        s_off = storage.group_off.get_tensor(cute.make_layout((MAX_E + 1,)))
        s_pre = storage.group_pre.get_tensor(cute.make_layout((MAX_E + 1,)))
        if warp_idx == EPILOGUE_WARPS[0]:
            build_group_tables(
                counts, E, tn, s_off, s_pre
            )  # dense offsets; the "m-block" is the token tile
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sW = smem.allocate_tensor(
            element_type=st.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sTok = smem.allocate_tensor(
            element_type=st.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        gW_mkl = cute.local_tile(
            mW_mkl, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )  # (bM, bK, RestM, RestK, RestL)
        k_tile_cnt = cute.size(gW_mkl, mode=[3])
        thr_mma = tiled_mma.get_slice(0)
        tCgW = thr_mma.partition_A(gW_mkl)
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sW, 0, 3),
            cute.group_modes(tCgW, 0, 3),
        )
        tCrW = tiled_mma.make_fragment_A(sW)
        tCrTok = tiled_mma.make_fragment_B(sTok)
        acc_shape = tiled_mma.partition_shape_C(st.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, st.num_acc_stage)
        )
        pipeline_init_wait(cluster_shape_mn=(1, 1))
        cute.arch.barrier()  # the group tables are complete

        raster_m = self.knobs.raster_along == "m"
        # the scheduler's "cluster rows" are the token tile, its "N tiles" the weight tiles
        if warp_idx == TMA_WARP:
            tile_sched = GroupScheduler.create(
                s_off, s_pre, counts, E, N, tn, wt, 1, raster_m, 0, 0
            )
            swap_tma_producer_loop(
                st,
                k_tile_cnt,
                tile_sched,
                ab_producer,
                tma_atom_w,
                tma_atom_tok,
                tWgW,
                tWsW,
                mTok_nkl,
                sTok,
                thr_mma,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
            )
        if warp_idx == MMA_WARP:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = GroupScheduler.create(
                s_off, s_pre, counts, E, N, tn, wt, 1, raster_m, 0, 0
            )
            mma_loop(
                self.lowering,
                st.num_acc_stage,
                k_tile_cnt,
                is_leader_cta,
                tile_sched,
                tiled_mma,
                ab_consumer,
                acc_pipeline,
                tCrW,
                tCrTok,
                tCtAcc_base,
            )
        if warp_idx < len(EPILOGUE_WARPS):
            tmem.allocate(st.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = GroupScheduler.create(
                s_off, s_pre, counts, E, N, tn, wt, 1, raster_m, 0, 0
            )
            cute.arch.griddepcontrol_wait()  # the output rows were the predecessor chain's reads
            swapped_epilogue_loop_fp8(
                st,
                self.chain,
                tidx,
                tile_sched,
                tCtAcc_base,
                thr_mma,
                mOut,
                Tk,
                N,
                epi_tile,
                acc_pipeline,
                mAlpha,
                mSa,
                mSi,
                wt,
            )
            tmem.relinquish_alloc_permit()
            # the allocator warp alone frees the columns: the epilogue warpgroup's warps join here after their last TMEM loads
            epi_join.arrive_and_wait()
            tmem.free(tmem_ptr)


@cute.jit
def swapped_epilogue_loop_fp8(
    st,
    chain: cutlass.Constexpr,
    tidx: Int32,
    tile_sched,
    tCtAcc_base: cute.Tensor,
    thr_mma,
    mOut: cute.Tensor,
    Tk: Int32,
    N: Int32,
    epi_tile: cute.Tile,
    acc_pipeline,
    mAlpha: cute.Tensor,
    mSa: cute.Tensor,
    mSi: cute.Tensor,
    weight_tile: cutlass.Constexpr,
):
    """The transposed group-masked epilogue with the W8A8 chains: the accumulator tile is (weight rows, tokens). The 128-row
    TMEM layout gives thread ``t`` weight row ``t`` of the tile and its token columns; the gate accumulators of the up rows (row
    % 16 < 8) come from lane ``+8`` (``shfl.sync.down``). The 64-row layout gives a thread the rows ``r`` and ``r + 8`` -- the
    interleaved weight's up / gate pair -- at every fourth token column, so the gate is the thread's own register. The up rows
    compute the gated chain and store output column ``(n // 16) * 8 + n % 8``; the ungated and plain chains store every row at
    column ``n``. Token ``j`` of the tile is dense row ``row0 + j``, stored iff ``< row_end``. ``alpha_e = mul(sa, alpha[e])`` per
    tile, ``inv = si[0]`` the intermediate multiplier as given; the e4m3 outputs are converted in pairs (two token columns of one
    weight row; each value's conversion is independent of its pairing) and stored as bytes, one element per store (the token
    columns of a thread's weight row are not contiguous in the output, so this body's stores are scalars by construction).
    Every output element's chain is computed by one thread from the same accumulator value in the same fp32 sequence as the
    grouped fp8 body's epilogue."""
    pair = chain.pair
    e4m3_out = chain.out_fmt == "e4m3"
    tiler_mn = cute.slice_(st.mma_tiler, (None, None, 0))
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    copy_atom_t2r = get_tmem_load_op(
        st.cta_tile_shape_mnk, st.c_layout, st.c_dtype, ACC_DTYPE, epi_tile, False
    )
    tAcc_epi = cute.flat_divide(tCtAcc, epi_tile)  # (EPI_M, EPI_N, SUB_M, SUB_N, STAGE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
    )
    thr_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc_base = thr_t2r.partition_S(tAcc_epi)
    idF = cute.make_identity_tensor(
        (st.cta_tile_shape_mnk[0], st.cta_tile_shape_mnk[1], 1)
    )  # the tile's own (weight row, token) coordinates
    cF = cute.local_tile(idF, tiler_mn, (None, None, None))
    tCcF = transform_partitioned_tensor_layout(thr_mma.partition_C(cF))
    cF_epi = cute.flat_divide(tCcF, epi_tile)
    tTR_cF = thr_t2r.partition_D(cF_epi)
    frag = tTR_cF[(None, None, None, 0, 0, 0, 0, 0)]
    offs = fragment_offsets(frag.layout)
    nv = len(offs)
    # the thread's weight rows: one (the 128-row tile) or the (r, r + 8) pair (the 64-row tile's lane split)
    row_offs = sorted({m for m, _ in offs})
    assert row_offs in ([0], [0, GROUP]), (
        "the swapped epilogue's TMEM layouts: one weight row per thread, or the (r, r + 8) pair",
        offs,
    )
    two_rows = len(row_offs) == 2
    index = {o: v for v, o in enumerate(offs)}
    partner = [
        index.get((m + GROUP, n)) for m, n in offs
    ]  # the gate register of an up register (two_rows)
    ups = [
        v for v in range(nv) if offs[v][0] == 0
    ]  # the registers of the thread's (first) weight row
    assert not two_rows or all(partner[v] is not None for v in ups), offs
    cols = [
        n - offs[0][1] for _, n in offs
    ]  # the register's token column inside the subtile
    rows = [m for m, _ in offs]  # the register's weight row inside the thread's pair
    tTR_rAcc = cute.make_rmem_tensor(frag.shape, ACC_DTYPE)
    tTR_cF = cute.group_modes(tTR_cF, 3, cute.rank(tTR_cF))
    inv = None
    if cutlass.const_expr(e4m3_out):
        inv = mSi[0]
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, st.num_acc_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        row0, row_end, n_tile, e = (
            work_tile.row0,
            work_tile.row_end,
            work_tile.n_tile,
            work_tile.e,
        )
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        rows_left = cutlass.min(row_end, Tk) - row0  # valid tokens of this tile
        alpha_e = None
        if cutlass.const_expr(not chain.pertoken):
            alpha_e = w8a8_alpha(MoeF32, mSa[0], mAlpha[e])
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_state.index)]
        acc_pipeline.consumer_wait(acc_state)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for s in cutlass.range_constexpr(subtile_cnt):
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
            crd0 = tTR_cF[(None, None, None, s)][(0, 0, 0)]
            wrow = crd0[0]  # this thread's (first) weight row in the tile
            col0 = crd0[1] - Int32(offs[0][1])  # the subtile's first token column
            n_glob = n_tile * Int32(weight_tile) + wrow
            # a dead token column's accumulators (past the group's live tokens, never stored) are replaced by a benign constant and
            # its chain runs with the live columns' (a select per register, not a region per register); the stores carry the predicate
            for v in cutlass.range_constexpr(nv):
                tTR_rAcc[v] = select_f32(
                    col0 + Int32(cols[v]) < rows_left, tTR_rAcc[v], Float32(DEAD_ACC)
                )
            # the gate accumulators of the up rows: lane + 8 (the 128-row layout) or the thread's second row (the 64-row one)
            gate = None
            if cutlass.const_expr(pair):
                if cutlass.const_expr(two_rows):
                    gate = [
                        tTR_rAcc[partner[v]] if partner[v] is not None else tTR_rAcc[v]
                        for v in range(nv)
                    ]
                else:
                    gate = [
                        cute.arch.shuffle_sync_down(tTR_rAcc[v], offset=8)
                        for v in range(nv)
                    ]
                is_up = (wrow % Int32(2 * GROUP)) < Int32(GROUP)
                out_col = (
                    n_tile * Int32(weight_tile // 2)
                    + (wrow // Int32(2 * GROUP)) * Int32(GROUP)
                    + wrow % Int32(GROUP)
                )
                store_ok = is_up & (n_glob < N)
            else:
                out_col = n_glob
                store_ok = n_glob < N
            # the per-token class: the thread's weight rows' channel scales (the gate row's: GROUP rows below the up row, or the thread's
            # second row) times the token column's scale (``mSa``: the per-row scales of the token rows)
            scale = [alpha_e] * nv
            scale_gate = [None] * nv
            if cutlass.const_expr(chain.pertoken):
                ch = {
                    m: mAlpha[(e, cutlass.min(n_glob + Int32(m), N - Int32(1)))]
                    for m in row_offs
                }
                chg = (
                    ch[GROUP]
                    if two_rows
                    else mAlpha[(e, cutlass.min(n_glob + Int32(GROUP), N - Int32(1)))]
                )
                r_pts = mSa.shape[0]
                tok = [
                    mSa[cutlass.min(row0 + col0 + Int32(cols[v]), r_pts - Int32(1))]
                    for v in range(nv)
                ]
                scale = [
                    pertoken_elem_scale(MoeF32, tok[v], ch[rows[v]]) for v in range(nv)
                ]
                if cutlass.const_expr(pair):
                    scale_gate = [
                        pertoken_elem_scale(MoeF32, tok[v], chg) for v in range(nv)
                    ]
            if cutlass.const_expr(e4m3_out):
                outs = ups if cutlass.const_expr(pair) else list(range(nv))
                assert len(outs) % 2 == 0, offs
                for q in cutlass.range_constexpr(0, len(outs), 2):
                    v0, v1 = outs[q], outs[q + 1]
                    j0 = col0 + Int32(cols[v0])
                    j1 = col0 + Int32(cols[v1])
                    y0 = _chain_elem(
                        chain,
                        tTR_rAcc[v0],
                        gate[v0] if cutlass.const_expr(pair) else None,
                        alpha_e,
                        inv,
                    )
                    y1 = _chain_elem(
                        chain,
                        tTR_rAcc[v1],
                        gate[v1] if cutlass.const_expr(pair) else None,
                        alpha_e,
                        inv,
                    )
                    pr = cvt_f32x2_to_fp8x2(y0, y1, "e4m3")
                    if store_ok & (j0 < rows_left) & (n_glob + Int32(rows[v0]) < N):
                        mOut[(row0 + j0, out_col + Int32(rows[v0]))] = (pr & 0xFF).to(
                            Uint8
                        )
                    if store_ok & (j1 < rows_left) & (n_glob + Int32(rows[v1]) < N):
                        mOut[(row0 + j1, out_col + Int32(rows[v1]))] = (pr >> 8).to(
                            Uint8
                        )
            else:
                for v in cutlass.range_constexpr(nv):
                    if cutlass.const_expr(pair and rows[v] != 0):
                        continue  # a gate register (two_rows): folded into its up register's chain
                    j = col0 + Int32(cols[v])
                    val = _chain_elem(
                        chain,
                        tTR_rAcc[v],
                        gate[v] if cutlass.const_expr(pair) else None,
                        scale[v],
                        inv,
                        s_gate=scale_gate[v],
                    )
                    if store_ok & (j < rows_left) & (n_glob + Int32(rows[v]) < N):
                        mOut[(row0 + j, out_col + Int32(rows[v]))] = val
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_state)
        acc_state.advance()
