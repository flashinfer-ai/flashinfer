# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The swapped body of the MoE classes: ``C^T`` per expert, with the expert's weight rows on the MMA's M side (a 128- or
64-row tile streamed by TMA with the expert as the batch coordinate) and the group's tokens on the N side as a **narrow
token tile** (8, 16 or 32 rows: the swapped records of :mod:`..schedule`; ``kind::f8f6f4`` takes 16 or 32), read as a TMA
box from the dense permuted activations at the group's row (``x_perm`` for GEMM1, the dense intermediate for GEMM2) -- the
same dense rows the grouped body reads, so no second packing. One body for the bf16 and the fp8 per-tensor classes, as
:mod:`.kernel`: the operand block (:class:`.kernel.DenseOperands`, keyed by the chain's operand format) and the epilogue
block (the chain: the bf16 chains, or the W8A8 chains with their scale operands) are resolved at trace time. Rows past the group's end (the next group's rows or the buffer's tail) are computed and
never stored: the MMA is token-local on N. One TMA pipeline carries the weight tile and the token box of one k-tile per
stage, two swizzle atoms of K per stage (the grouped records' step), the dense ``mma_loop`` folds them, and the
transposed group-masked epilogue stores token ``j`` of the tile as output row ``row0 + j``.

The same class: the tcgen05 step is symmetric in its operands -- ``D[m, n] += sum_k A[m, k] B[n, k]`` folds the same K
sequence per output element whichever operand carries the token -- so every output element's fold is the dense class's
fold on the same operands (the per-expert agreement with the grouped body is the test). The gated chain's pairing moves
from two registers of one thread to two lanes of one warp (rows ``16q + r`` (up) and ``16q + 8 + r`` (gate) of the
interleaved weight are TMEM lanes ``l`` and ``l + 8``: a ``shfl.sync.down 8`` of the gate accumulators); the term per
element is unchanged. Everything else is placement: which CTA owns which (expert, token tile, weight tile), the TMA
pipeline, the store pattern.

Per tile the operand bytes are the weight tile plus ``token_tile`` rows of tokens instead of the grouped body's 64- or
128-row token tile: at a few rows per active expert the GEMM traffic becomes the weights read once, through twice as many
weight rows per CTA. The dispatcher ranks the swapped records with the grouped ones (:func:`..schedule.rank_grouped`).

Operands: ``rows (R, K, 1)`` K-major (the dense permuted rows; ``R >= Tk``: with ``R >= Tk + token_tile`` the last tiles'
boxes stay inside the buffer, else they are zero-filled past it), ``w (N, K, E)`` K-major, ``out (Tk, Nout)`` row-major
(``Nout = N`` plain and ungated, ``N / 2`` gated; bf16 / f16 / f32, or the uint8 bytes of e4m3 for the fp8 classes'
GEMM1), ``counts (E,)`` int32; for a chain with scale operands ``alpha (E,)``, ``sa (1,)``, ``si (1,)`` fp32 as for the
grouped body (:meth:`SwappedGroupedGemm.with_scales`).
The gather form (``gather=True``, :meth:`SwappedGroupedGemm.gathered`): the token tile's rows gathered by the TMA's gather4
mode from the unpermuted ``x (T, K)`` through the plan's ``src`` (row ``p`` reads ``x[src[p] // k_div]``; rows past the
group's end repeat its last row and are never stored), one swizzle atom of K per stage (the gather4 box is the atom): GEMM1
without the gather launch and the permuted tensor, the same bytes in the stage, the same fold. Its predecessor is the plan's
launch, so every warp waits for it before the group tables.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Int32, Uint8
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.blackwell_helpers import get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ....batch_invariant.quantization.cvt_fp8 import cvt_f32x2_to_fp8x2
from ....gemm.batch_invariant.blocks.mainloop import mma_loop
from ....gemm.batch_invariant.blocks.operands import ACC_DTYPE, BAR_TMEM_ALLOC
from ....gemm.batch_invariant.generator import TILES_N_NARROW, Knobs
from ..spec_fp8 import w8a8_alpha
from .epilogue import (
    BAR_EPI_JOIN,
    GROUP,
    GroupedChain,
    GroupedEpilogueBlock,
    MoeF32,
    fragment_offsets,
)
from .kernel import DenseOperands
from .scheduler import MAX_E, TABLE_SMEM_BYTES, GroupScheduler, build_group_tables
from .. import pdl

EPILOGUE_WARPS = (
    0,
    1,
    2,
    3,
)  # one warpgroup: the weight rows of the tile (one per thread, or the (r, r + 8) pair of the 64-row tile)
MMA_WARP = 4
TMA_WARP = 5
THREADS = 32 * (TMA_WARP + 1)
WEIGHT_TILE = (
    128  # weight rows per tile (the MMA's M: the 1-CTA instruction's 128 rows)
)
WEIGHT_TILES = (
    WEIGHT_TILE,
    64,
)  # the weight tiles: 128 rows, or 64 (the two-row TMEM layout: twice the CTAs of a small weight)
#: The narrowest token tile of each operand format's instruction: ``kind::f16`` admits N >= 8, ``kind::f8f6f4`` N >= 16.
MIN_TOKEN_TILE: Dict[str, int] = {"bf16": 8, "e4m3": 16}
#: The 64-row weight tile takes token tiles from 16 (a 64 x 8 tile holds too little work per CTA).
MIN_TOKEN_TILE_64 = 16


def token_tiles_of(weight_tile: int, in_fmt: str = "bf16") -> Tuple[int, ...]:
    """The token tiles a weight tile takes for an operand format: the narrow dense tiles from the format's narrowest, the
    64-row tile from 16."""
    lo = max(MIN_TOKEN_TILE[in_fmt], MIN_TOKEN_TILE_64 if weight_tile == 64 else 0)
    return tuple(t for t in TILES_N_NARROW if t >= lo)


TOKEN_TILES = TILES_N_NARROW  # every swapped record's token tile is one of the narrow dense tiles (8, 16, 32)
# the token tiles a weight tile takes (the bf16 classes)
TOKEN_TILES_OF_WEIGHT = {wt: token_tiles_of(wt, "bf16") for wt in WEIGHT_TILES}


def swap_knobs(
    token_tile: int = None,
    raster: str = "m",
    weight_tile: int = WEIGHT_TILE,
    in_fmt: str = "bf16",
) -> Knobs:
    """The classes' swapped record with ``token_tile`` tokens (the format's narrowest by default) on the MMA's N side and
    ``weight_tile`` (128 or 64) weight rows on its M side, 1-CTA, no cluster."""
    tiles = token_tiles_of(weight_tile, in_fmt)
    token_tile = tiles[0] if token_tile is None else token_tile
    assert token_tile in tiles, (token_tile, weight_tile, in_fmt)
    return Knobs(
        (weight_tile, token_tile), (1, 1), False, True, raster, 1, swap_ab=True
    )


def swapped_knobs(in_fmt: str = "bf16") -> List[Knobs]:
    """The swapped record space of a format's classes: one record per (weight tile, token tile) -- 128 x 8 / 16 / 32 and
    64 x 16 / 32 for bf16; 128 x 16 / 32 and 64 x 16 / 32 for e4m3."""
    return [
        swap_knobs(tn, weight_tile=wt, in_fmt=in_fmt)
        for wt in WEIGHT_TILES
        for tn in token_tiles_of(wt, in_fmt)
    ]


def is_swapped_record(knobs: Knobs, in_fmt: str = "bf16") -> bool:
    """A record the swapped body takes for the format: the swapped 1-CTA record at one of the weight tiles and its narrow
    token tiles, no cluster, swizzle 1, no in-cluster split, no in-CTA chunk fold, the TMA store flag of every record."""
    return (
        knobs.swap_ab
        and knobs.tile_m in WEIGHT_TILES
        and knobs.tile_n in token_tiles_of(knobs.tile_m, in_fmt)
        and knobs.cluster_shape_mn == (1, 1)
        and not knobs.use_2cta_instrs
        and knobs.swizzle_size == 1
        and knobs.cluster_k == 1
        and not knobs.cta_chunks
        and knobs.use_tma_store
    )


def plan_knobs_of(knobs: Knobs) -> Knobs:
    """The dense twin that sizes the static plan (tile, cluster, stages): the swapped flag names the body, not a tile shape."""
    return dataclasses.replace(knobs, swap_ab=False)


class SwappedGroupedGemm:
    """``out[p, n] = chain(sum_k rows[p, k] w[n, k, e(p)])`` for every dense row ``p`` of every group (with the fp8 classes'
    chains ``alpha = mul(sa[0], alpha[e(p)])`` scales the accumulators), the weights as the MMA's A and the group's rows as a
    narrow TMA box on its B side. The persistent grid (``max_active_clusters``, in CTAs) is a dynamic launch argument: which
    CTA computes a tile, never the bits. The chain's operand format selects the operand block and the record space; a chain
    with scale operands launches through :meth:`with_scales`; the gather form belongs to the bf16 classes."""

    def __init__(
        self,
        knobs: Knobs,
        chain: Union[str, GroupedEpilogueBlock],
        gather: bool = False,
    ):
        if isinstance(chain, str):
            chain = GroupedChain.legacy(chain, "bf16")
        assert isinstance(chain, GroupedEpilogueBlock), chain
        self.in_fmt = chain.in_fmt
        self.operands = DenseOperands(self.in_fmt)
        assert is_swapped_record(knobs, self.in_fmt), (
            "the swapped body takes the narrow swapped records of the format (128 or 64 weight rows x its narrow token "
            "tiles, 1-CTA)",
            knobs,
            self.in_fmt,
        )
        assert not gather or self.in_fmt == "bf16", (
            "the gather form is the bf16 classes'",
            self.in_fmt,
        )
        self.knobs, self.chain, self.gather = knobs, chain, gather
        self.token_tile, self.weight_tile = knobs.tile_n, knobs.tile_m
        # the box form reads the group counts before its wait (the kernel body): a placement that triggers before its wait
        # would let this grid launch before the plan completed (pdl.BOX_FORM_TRIGGERS)
        assert gather or pdl.TRIGGER in pdl.BOX_FORM_TRIGGERS, (
            "the swapped box form reads the plan's counts before its PDL wait: it compiles under the trigger placements "
            f"{pdl.BOX_FORM_TRIGGERS} only",
            pdl.TRIGGER,
        )

    @cute.jit
    def __call__(
        self,
        rows: cute.Tensor,
        w: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The TMA-box form of the chains without scale operands: ``rows (R, K, 1)`` the dense permuted rows."""
        assert not self.gather and not self.chain.scale_operands
        self._launch(
            rows,
            w,
            out,
            counts,
            max_active_clusters,
            stream,
            None,
            Int32(1),
            None,
            None,
            None,
        )

    @cute.jit
    def with_scales(
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
        """The TMA-box form of the chains with scale operands (the fp8 per-tensor classes): ``alpha (E,)`` the per-expert
        weight scales, ``sa (1,)`` the A operand's scale, ``si (1,)`` the intermediate multiplier."""
        assert not self.gather and self.chain.scale_operands
        self._launch(
            rows,
            w,
            out,
            counts,
            max_active_clusters,
            stream,
            None,
            Int32(1),
            alpha,
            sa,
            si,
        )

    @cute.jit
    def gathered(
        self,
        x: cute.Tensor,
        src: cute.Tensor,
        k_div: Int32,
        w: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The gather form: ``x (T, K)`` the unpermuted rows, ``src (>= Tk,)`` int32 the plan's slot of every expanded row."""
        assert self.gather
        self._launch(
            x, w, out, counts, max_active_clusters, stream, src, k_div, None, None, None
        )

    def _launch(
        self,
        rows,
        w,
        out,
        counts,
        max_active_clusters,
        stream,
        src,
        k_div,
        alpha,
        sa,
        si,
    ):
        if cutlass.const_expr(rows.element_type != w.element_type):
            raise TypeError(
                f"rows and w must have one type: {rows.element_type} != {w.element_type}"
            )
        # the gather form's slab is one swizzle atom of K wide (the gather4 box is the atom): it keeps the format's default stage
        body = "gather" if self.gather else "swapped"
        # the weights are the plan's A operand, the token rows its B operand
        st = self.operands.plan(
            plan_knobs_of(self.knobs), w, rows, TABLE_SMEM_BYTES, body
        )
        self.st = st
        self.block = self.operands.mainloop(body)
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
        b_one = cute.select(
            st.b_smem_layout_staged,
            mode=list(range(cute.rank(st.b_smem_layout_staged) - 1)),
        )
        tile_k = st.cta_tile_shape_mnk[2]
        if cutlass.const_expr(self.gather):
            # the token rows gathered by the TMA's gather4 mode over the 2-D x: a (token_tile, k-tile) box whose rows come
            # from the index tensor, written to the swizzled stage as the dense box writes it (the atom carries the swizzle)
            slab = cute.make_composed_layout(
                st.b_smem_layout_staged.inner,
                0,
                cute.make_layout((self.token_tile, tile_k), stride=(tile_k, 1)),
            )
            gI = cute.make_tensor(
                src.iterator,
                cute.make_layout((src.shape[0], rows.shape[1]), stride=(1, 0)),
            )
            atom_tok, tensor_tok = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensor2DGather4G2SOp(tcgen05.CtaGroup.ONE),
                rows,
                slab,
                (self.token_tile, tile_k),
                gmem_coord_tensor=gI,
            )
            m_src = src
        else:
            # the token rows as the B operand: a box of token_tile rows x one k-tile, re-addressed per tile at the group's row
            b_op = utils.sm100.cluster_shape_to_tma_atom_B(
                st.cluster_shape_mn, st.tiled_mma.thr_id
            )
            atom_tok, tensor_tok = cute.nvgpu.make_tiled_tma_atom_B(
                b_op,
                rows,
                b_one,
                st.mma_tiler,
                st.tiled_mma,
                st.cluster_layout_vmnk.shape,
            )
            m_src = counts  # unread
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
            m_src,
            k_div,
            st.cluster_layout_vmnk,
            st.a_smem_layout_staged,
            st.b_smem_layout_staged,
            st.epi_tile,
            alpha,
            sa,
            si,
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
        mSrc: cute.Tensor,
        k_div: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        mAlpha: cute.Tensor = None,
        mSa: cute.Tensor = None,
        mSi: cute.Tensor = None,
    ):
        st = self.st
        tn, wt = self.token_tile, self.weight_tile
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_tok)
        # programmatic dependent launch: every warp waits for the preceding grid before the group tables (the tables read
        # ``counts``, the predecessor chain's writes: a no-op when nothing before triggered early, and the rule that no kernel
        # of the chain triggers before its own wait -- ``pdl.BOX_FORM_TRIGGERS``, asserted at the builder -- is the margin);
        # the producer waits once more before its first read of the predecessor's output (the token rows), the epilogue
        # before its first store; the trigger placement is the chain's (``pdl.TRIGGER``).
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
        cute.arch.griddepcontrol_wait()  # the tables read the predecessor chain's counts
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
            if cutlass.const_expr(self.gather):
                swap_gather_producer_loop(
                    st,
                    k_tile_cnt,
                    tile_sched,
                    ab_producer,
                    tma_atom_w,
                    tma_atom_tok,
                    tWgW,
                    tWsW,
                    mTok_nkl,
                    mSrc,
                    k_div,
                    sTok,
                    Tk,
                    tn,
                    st.cta_tile_shape_mnk[2],
                )
            else:
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
            swapped_epilogue_loop(
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
                mAlpha=mAlpha,
                mSa=mSa,
                mSi=mSi,
            )
            tmem.relinquish_alloc_permit()
            # the allocator warp alone frees the columns: the epilogue warpgroup's warps join here after their last TMEM loads
            epi_join.arrive_and_wait()
            tmem.free(tmem_ptr)


@cute.jit
def swap_tma_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    tma_producer,
    tma_atom_w: cute.CopyAtom,
    tma_atom_tok: cute.CopyAtom,
    tWgW: cute.Tensor,
    tWsW: cute.Tensor,
    mTok_nkl: cute.Tensor,
    sTok: cute.Tensor,
    thr_mma,
    b_cta_coord,
    b_cta_layout,
):
    """The TMA producer: per tile the expert's weight tile ``n_tile`` (128 or 64 rows; the expert as the batch coordinate) and
    the token box re-addressed at the group's row ``row0`` (``token_tile`` rows: the rows past the group's end are computed and never
    stored, the rows past the buffer are zero-filled); k-tiles ascending, one ring slot holds both. The weights are inputs of
    the whole chain, the token rows the predecessor's output: the first tile's weight stages (as many as the ring holds) are
    issued before the programmatic-dependent-launch wait and their token boxes right after it, so the ring fills while the
    predecessor drains; the stage's transaction barrier completes when both copies have landed, whatever their order."""
    work_tile = tile_sched.initial_work_tile_info()
    n_pre = cutlass.min(Int32(st.num_ab_stage), k_tile_cnt)
    if not work_tile.is_valid_tile:
        n_pre = Int32(0)  # a CTA without tiles issues nothing
    tWgW_slice = tWgW[(None, work_tile.n_tile, None, work_tile.e)]
    real_tok = cute.domain_offset((work_tile.row0, 0, 0), mTok_nkl)
    gTok = cute.local_tile(
        real_tok, cute.slice_(st.mma_tiler, (0, None, None)), (None, None, None)
    )
    tCgTok = thr_mma.partition_B(gTok)
    tTsTok, tTgTok = cpasync.tma_partition(
        tma_atom_tok,
        b_cta_coord,
        b_cta_layout,
        cute.group_modes(sTok, 0, 3),
        cute.group_modes(tCgTok, 0, 3),
    )
    tTgTok_slice = tTgTok[(None, 0, None, 0)]
    tma_producer.reset()
    tok_producer = (
        tma_producer.clone()
    )  # the same stages, walked again for the token boxes
    for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
        handle = tma_producer.acquire_and_advance()
        cute.copy(
            tma_atom_w,
            tWgW_slice[(None, handle.count)],
            tWsW[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
        )
    cute.arch.griddepcontrol_wait()  # the token rows are the predecessor's
    for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
        handle = tok_producer.current_handle()
        cute.copy(
            tma_atom_tok,
            tTgTok_slice[(None, handle.count)],
            tTsTok[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
        )
        tok_producer.advance()
    k_start = n_pre
    while work_tile.is_valid_tile:
        tWgW_slice = tWgW[(None, work_tile.n_tile, None, work_tile.e)]
        real_tok = cute.domain_offset((work_tile.row0, 0, 0), mTok_nkl)
        gTok = cute.local_tile(
            real_tok, cute.slice_(st.mma_tiler, (0, None, None)), (None, None, None)
        )
        tCgTok = thr_mma.partition_B(gTok)
        tTsTok, tTgTok = cpasync.tma_partition(
            tma_atom_tok,
            b_cta_coord,
            b_cta_layout,
            cute.group_modes(sTok, 0, 3),
            cute.group_modes(tCgTok, 0, 3),
        )
        tTgTok_slice = tTgTok[(None, 0, None, 0)]
        peek_empty = cutlass.Boolean(1)
        if k_start < k_tile_cnt:
            peek_empty = tma_producer.try_acquire()
        for _k_tile in cutlass.range(k_start, k_tile_cnt, 1, unroll=1):
            handle = tma_producer.acquire_and_advance(peek_empty)
            cute.copy(
                tma_atom_w,
                tWgW_slice[(None, handle.count)],
                tWsW[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            cute.copy(
                tma_atom_tok,
                tTgTok_slice[(None, handle.count)],
                tTsTok[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = tma_producer.try_acquire()
        tma_producer.reset()  # the next tile's k-tiles count from 0
        k_start = Int32(0)
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    pdl.trigger_late()  # the CTA's loads are all issued: the dependents' prologues may start
    tma_producer.tail()


@cute.jit
def swap_gather_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    tma_producer,
    tma_atom_w: cute.CopyAtom,
    tma_atom_tok: cute.CopyAtom,
    tWgW: cute.Tensor,
    tWsW: cute.Tensor,
    mX: cute.Tensor,
    mSrc: cute.Tensor,
    k_div: Int32,
    sTok: cute.Tensor,
    Tk: Int32,
    tn: cutlass.Constexpr,
    tile_k: cutlass.Constexpr,
):
    """The TMA producer of the gather form: per tile the expert's weight tile (128 or 64 rows; the expert as the batch
    coordinate) and the token tile gathered by the TMA's gather4 mode from the unpermuted rows -- the tile's ``tn`` token indices (``src[p] //
    k_div`` for the group's rows, the group's last row past its end: a valid box whose extra rows are never stored) go into
    registers once per tile and every k-tile's gather issues from them; both copies count on the stage's transaction barrier.
    The kernel has waited for the predecessor (the plan's launch) before its tables, so the wait here returns at once; the
    loop keeps the box form's shape (the first tile's weight stages, the wait, the token gathers)."""
    stage_elems = tn * tile_k
    sTok_plain = cute.make_tensor(
        sTok.iterator,
        cute.make_layout(
            (tn, tile_k, st.num_ab_stage), stride=(tile_k, 1, stage_elems)
        ),
    )  # the atom applies the swizzle
    sTok_plain = cute.group_modes(sTok_plain, 0, 2)
    idx = cute.make_rmem_tensor(cute.make_layout((tn,)), Int32)
    work_tile = tile_sched.initial_work_tile_info()
    n_pre = cutlass.min(Int32(st.num_ab_stage), k_tile_cnt)
    if not work_tile.is_valid_tile:
        n_pre = Int32(0)
    tWgW_slice = tWgW[(None, work_tile.n_tile, None, work_tile.e)]
    tma_producer.reset()
    tok_producer = tma_producer.clone()
    for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
        handle = tma_producer.acquire_and_advance()
        cute.copy(
            tma_atom_w,
            tWgW_slice[(None, handle.count)],
            tWsW[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
        )
    cute.arch.griddepcontrol_wait()  # the plan's row map is the predecessor's
    row0 = work_tile.row0
    last = cutlass.max(cutlass.min(work_tile.row_end, Tk) - Int32(1), Int32(0))
    for j in cutlass.range_constexpr(tn):
        idx[j] = mSrc[cutlass.min(row0 + Int32(j), last)] // k_div
    gX = cute.local_tile(
        mX, (tn, tile_k), (0, None)
    )  # (tn, tile_k, n_kt): the K tiles of the gathered rows
    n_kt = cute.size(gX, mode=[2])
    gI = cute.make_tensor(
        idx.iterator, cute.make_layout((tn, tile_k, n_kt), stride=(1, 0, 0))
    )
    tTsTok, tTgX, tTgI = cpasync.tma_partition(
        tma_atom_tok,
        0,
        cute.make_layout(1),
        sTok_plain,
        [cute.group_modes(gX, 0, 2), cute.group_modes(gI, 0, 2)],
    )
    for _k_tile in cutlass.range(0, n_pre, 1, unroll=1):
        handle = tok_producer.current_handle()
        cute.copy(
            tma_atom_tok,
            [tTgX[(None, handle.count)], tTgI[(None, handle.count)]],
            tTsTok[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
        )
        tok_producer.advance()
    k_start = n_pre
    while work_tile.is_valid_tile:
        tWgW_slice = tWgW[(None, work_tile.n_tile, None, work_tile.e)]
        row0 = work_tile.row0
        last = cutlass.max(cutlass.min(work_tile.row_end, Tk) - Int32(1), Int32(0))
        for j in cutlass.range_constexpr(tn):
            idx[j] = mSrc[cutlass.min(row0 + Int32(j), last)] // k_div
        gI = cute.make_tensor(
            idx.iterator, cute.make_layout((tn, tile_k, n_kt), stride=(1, 0, 0))
        )
        tTsTok, tTgX, tTgI = cpasync.tma_partition(
            tma_atom_tok,
            0,
            cute.make_layout(1),
            sTok_plain,
            [cute.group_modes(gX, 0, 2), cute.group_modes(gI, 0, 2)],
        )
        peek_empty = cutlass.Boolean(1)
        if k_start < k_tile_cnt:
            peek_empty = tma_producer.try_acquire()
        for _k_tile in cutlass.range(k_start, k_tile_cnt, 1, unroll=1):
            handle = tma_producer.acquire_and_advance(peek_empty)
            cute.copy(
                tma_atom_w,
                tWgW_slice[(None, handle.count)],
                tWsW[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            cute.copy(
                tma_atom_tok,
                [tTgX[(None, handle.count)], tTgI[(None, handle.count)]],
                tTsTok[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = tma_producer.try_acquire()
        tma_producer.reset()
        k_start = Int32(0)
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    pdl.trigger_late()
    tma_producer.tail()


@cute.jit
def swapped_epilogue_loop(
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
    mAlpha: cute.Tensor = None,
    mSa: cute.Tensor = None,
    mSi: cute.Tensor = None,
):
    """The transposed group-masked epilogue: the accumulator tile is (weight rows, tokens); the 128-row TMEM layout gives
    thread ``t`` weight row ``t`` of the tile and its token columns (the record's N tile). Gated chain: the gate accumulators
    come from lane ``+8`` (``shfl.sync.down``); the up lanes (row % 16 < 8) compute the chain and store output column
    ``(n // 16) * 8 + n % 8``; plain and ungated chains: every lane stores column ``n``. Token ``j`` of the tile is dense row
    ``row0 + j``, stored iff ``< row_end``. Every output element's chain is computed by one thread from the same accumulator
    value in the same fp32 sequence as the grouped body's epilogue (one element per store: the token columns of a thread's
    weight row are not contiguous in the output, so this body's stores are scalars by construction). The chains with scale
    operands (``mAlpha`` / ``mSa`` / ``mSi``, as in the grouped epilogue) form ``alpha_e = mul(sa, alpha[e])`` per tile and
    read ``inv = si[0]`` as given; the e4m3 outputs are converted in pairs (two token columns of one weight row; each value's
    conversion is independent of its pairing) and stored as bytes."""
    pair = chain.pair
    paired_cvt = chain.paired_cvt
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
    tTR_tAcc_base = thr_t2r.partition_S(
        tAcc_epi
    )  # (T2R, T2R_M, T2R_N, SUB_M, SUB_N, STAGE)
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
    tile_m = st.cta_tile_shape_mnk[0]
    # the thread's weight rows: one (the 128-row tile: lane = row, the gate row GROUP lanes below) or two (the 64-row tile's
    # lane split: rows r and r + GROUP of one thread, so the gated pair sits in the thread's own registers)
    row_offs = sorted({m for m, _ in offs})
    assert row_offs in ([0], [0, GROUP]), (
        "the swapped epilogue's TMEM layouts: one weight row per thread, or the (r, r + 8) pair",
        offs,
        frag.layout,
    )
    two_rows = len(row_offs) == 2
    index = {o: v for v, o in enumerate(offs)}
    partner = [
        index.get((m + GROUP, n)) for m, n in offs
    ]  # the gate register of an up register (two_rows)
    ups = [
        v for v in range(nv) if offs[v][0] == 0
    ]  # the registers that hold an up row (all, with one row)
    assert not two_rows or all(partner[v] is not None for v in ups), offs
    cols = [
        n - offs[0][1] for _, n in offs
    ]  # the register's token column inside the subtile
    rows = [m for m, _ in offs]  # the register's weight row inside the thread's pair
    tTR_rAcc = cute.make_rmem_tensor(frag.shape, ACC_DTYPE)
    tTR_cF = cute.group_modes(tTR_cF, 3, cute.rank(tTR_cF))
    inv = None
    if cutlass.const_expr(paired_cvt):
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
        if cutlass.const_expr(chain.scaled):
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
            n_glob = n_tile * Int32(tile_m) + wrow
            # the gate accumulators of the up rows: lane + 8 (rows 16q + 8 + r sit 8 lanes below rows 16q + r in the 128-row
            # layout), or the thread's own second row (the 64-row layout)
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
                    n_tile * Int32(tile_m // 2)
                    + (wrow // Int32(2 * GROUP)) * Int32(GROUP)
                    + wrow % Int32(GROUP)
                )
                store_ok = is_up & (n_glob < N)
            else:
                out_col = n_glob
                store_ok = n_glob < N
            # the chain of a token column is computed only where the column is a live token of the group (a warp-uniform
            # branch: every lane of the warp holds the same column), so a tile with few live tokens pays few chains
            if cutlass.const_expr(paired_cvt):
                outs = ups if cutlass.const_expr(pair) else list(range(nv))
                assert len(outs) % 2 == 0, offs
                for q in cutlass.range_constexpr(0, len(outs), 2):
                    v0, v1 = outs[q], outs[q + 1]
                    j0 = col0 + Int32(cols[v0])
                    j1 = col0 + Int32(cols[v1])
                    if (j0 < rows_left) | (j1 < rows_left):
                        y0 = chain.elem(
                            MoeF32,
                            tTR_rAcc[v0],
                            gate[v0] if cutlass.const_expr(pair) else None,
                            alpha_e,
                            inv,
                        )
                        y1 = chain.elem(
                            MoeF32,
                            tTR_rAcc[v1],
                            gate[v1] if cutlass.const_expr(pair) else None,
                            alpha_e,
                            inv,
                        )
                        pr = cvt_f32x2_to_fp8x2(y0, y1, "e4m3")
                        if store_ok & (j0 < rows_left) & (n_glob + Int32(rows[v0]) < N):
                            mOut[(row0 + j0, out_col + Int32(rows[v0]))] = (
                                pr & 0xFF
                            ).to(Uint8)
                        if store_ok & (j1 < rows_left) & (n_glob + Int32(rows[v1]) < N):
                            mOut[(row0 + j1, out_col + Int32(rows[v1]))] = (pr >> 8).to(
                                Uint8
                            )
            else:
                for v in cutlass.range_constexpr(nv):
                    if cutlass.const_expr(pair and rows[v] != 0):
                        continue  # a gate register (two_rows): folded into its up register's chain
                    j = col0 + Int32(cols[v])
                    if j < rows_left:
                        val = chain.elem(
                            MoeF32,
                            tTR_rAcc[v],
                            gate[v] if cutlass.const_expr(pair) else None,
                            alpha_e,
                            inv,
                        )
                        if store_ok & (n_glob + Int32(rows[v]) < N):
                            mOut[(row0 + j, out_col + Int32(rows[v]))] = val
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_state)
        acc_state.advance()
