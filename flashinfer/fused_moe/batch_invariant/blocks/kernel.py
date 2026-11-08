# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The grouped GEMM kernel body on the dense blocks (bf16 operands): the dense blocks (operands, TMA ring, the
per-instruction MMA fold, TMEM lifecycle) under the group scheduler, with per-tile TMA coordinates (A offset by the
group's rows through ``cute.domain_offset``, B indexed by the expert as the batch coordinate) and the group-masked
epilogue.

Operands: A ``(Tk, K, 1)`` K-major (the permuted activations), B ``(N, K, E)`` K-major (the stacked expert weights,
``l`` = expert), out ``(Tk, Nout)`` row-major (``Nout = N`` for the plain and the ungated chains, ``N / 2`` for the
gated chain; bf16 / f16 / f32), ``counts (E,)`` int32 (tokens per expert, in group order). The group's rows are
``[off_e, off_e + counts[e])`` with ``off`` the exclusive prefix of ``counts`` (the dense packing).

The gather variant (``GatherGroupedBlockGemm``) takes the unpermuted ``x (T, K)`` and the plan's ``src (Tk,)`` instead of
the permuted A: one gather warp per 32-row slab of the CTA's A tile issues the TMA's ``gather4`` loads of the slab's rows
(``x[src[p] // k_div]``; rows past a group's end repeat its last row and are never stored) into the same swizzled stage the
dense box fills, counted on the same stage barrier; the MMA fold, the epilogue and the bits are the dense variant's.
"""

from __future__ import annotations

from typing import Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum

from ....gemm.batch_invariant.blocks.mainloop import mma_loop
from ....gemm.batch_invariant.blocks.operands import (
    ACC_DTYPE,
    BAR_TMEM_ALLOC,
    IN_DT,
    StaticPlan,
    TmaSetup,
)
from ....gemm.batch_invariant.generator import Knobs, MainloopBlock
from .epilogue import (
    GroupedChain,
    grouped_epilogue_loop,
    staged_store,
    staging_bytes,
    staging_layout,
)
from .pipelines import PipelineCpAsyncUmma
from .scheduler import (
    EPI_WARPGROUPS_DENSE,
    MAX_E,
    TABLE_SMEM_BYTES,
    GroupScheduler,
    build_group_tables,
    warp_roles,
)

CHAINS = ("plain", "swiglu", "relu2")
EPILOGUE_WARPS, MMA_WARP, TMA_WARP, THREADS = warp_roles(EPI_WARPGROUPS_DENSE)
SLAB_ROWS = 32  # the gather variant: A rows gathered per gather warp per k-tile (8 TMA gather4 instructions of 4 rows)
A_PRODUCERS = (
    "gather4",
    "ldgsts",
)  # the gather variant's A producer: the TMA's gather4 mode or 16-byte cp.async chunks (the LSU)
LDGSTS_WARPS = 4  # the LDGSTS producer: one warpgroup, 16 rows x 8 chunks per pass
A_CHUNK_BYTES = 16  # one cp.async of the LDGSTS producer
GATHER_SMEM_ALIGN = 512  # the gather variants' extra barriers: the ring's budget also pays their arrays and this alignment margin


class GroupedBlockGemm:
    """``out[p, n] = chain(sum_k A[p, k] B[n, k, e(p)])`` for every row ``p`` of every group. The persistent grid
    (``max_active_clusters``, in clusters) is a dynamic launch argument: which cluster computes a tile, never the bits."""

    def __init__(
        self,
        knobs: Knobs,
        chain: Union[str, GroupedChain],
        out_fmt: str = "bf16",
        gather: bool = False,
        a_producer: str = "gather4",
    ):
        if isinstance(chain, str):  # the bf16 class's names
            chain = GroupedChain.legacy(chain, out_fmt)
        assert isinstance(chain, GroupedChain), chain
        assert a_producer in A_PRODUCERS, a_producer
        self.knobs, self.chain, self.out_fmt = knobs, chain, chain.out_fmt
        self.cluster_shape_mn = knobs.cluster_shape_mn
        self.gather, self.a_producer = gather, a_producer
        epi, mma, tma, threads = warp_roles(EPI_WARPGROUPS_DENSE)
        self.gather_warps = ()
        if gather:
            # the gather warps sit between the epilogue warpgroups and the MMA warp: one per 32-row slab of the CTA's A
            # tile for the TMA gather4 producer, one warpgroup (16 rows per pass of 16-byte chunks) for the LDGSTS producer
            cta_rows = knobs.mma_tiler_mn[0] // (2 if knobs.use_2cta_instrs else 1)
            assert cta_rows % SLAB_ROWS == 0, cta_rows
            n_gather = LDGSTS_WARPS if a_producer == "ldgsts" else cta_rows // SLAB_ROWS
            self.gather_warps = tuple(range(len(epi), len(epi) + n_gather))
            mma, tma = self.gather_warps[-1] + 1, self.gather_warps[-1] + 2
            threads = 32 * (tma + 1)
        self.epilogue_warps, self.mma_warp, self.tma_warp, self.threads = (
            epi,
            mma,
            tma,
            threads,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._launch(a, b, out, counts, max_active_clusters, stream)

    # host side (trace time)
    def _launch(
        self, a, b, out, counts, max_active_clusters, stream, src=None, k_div=None
    ):
        """``src`` / ``k_div`` (the gather variant): ``a`` is then the unpermuted ``x [T, K]`` and expanded row ``p`` takes
        ``x[src[p] // k_div]`` through the TMA's gather4 mode (rows past a group's end repeat its last row, never stored);
        otherwise ``a`` is the permuted ``[Tk, K, 1]`` read by dense TMA boxes."""
        if cutlass.const_expr(a.element_type != b.element_type):
            raise TypeError(
                f"A and B must have one type: {a.element_type} != {b.element_type}"
            )
        assert a.element_type is IN_DT["bf16"], a.element_type
        assert (src is not None) == self.gather
        # the accumulator-side "C" of the static plan is fp32 row-major: it only sizes the epilogue tile; the ring leaves room
        # for the group tables and the epilogue staging buffers
        self.staged = staged_store(self.chain.out_fmt)

        def plan(extra: int) -> StaticPlan:
            return StaticPlan(
                self.knobs,
                a.element_type,
                b.element_type,
                ACC_DTYPE,
                LayoutEnum.from_tensor(a).mma_major_mode(),
                LayoutEnum.from_tensor(b).mma_major_mode(),
                LayoutEnum.ROW_MAJOR,
                c_in_smem=False,
                extra_smem_bytes=TABLE_SMEM_BYTES + extra,
            )

        st = plan(0)
        if self.staged:
            st = plan(
                staging_bytes(out.element_type, st.epi_tile, EPI_WARPGROUPS_DENSE)
            )
        if self.gather:
            # the gather variants' two extra barrier arrays (the gather and the sync pipelines, 2 x stages x 8 bytes each)
            # come off the operand ring's budget: re-plan with their bytes at this ring depth (monotone: the ring only shrinks)
            reserve = 2 * 2 * st.num_ab_stage * 8 + GATHER_SMEM_ALIGN
            st = plan(
                reserve
                + (
                    staging_bytes(out.element_type, st.epi_tile, EPI_WARPGROUPS_DENSE)
                    if self.staged
                    else 0
                )
            )
        self.st = st
        self.c_smem_layout_staged = (
            staging_layout(out.element_type, st.epi_tile, EPI_WARPGROUPS_DENSE)
            if self.staged
            else st.a_smem_layout_staged
        )
        # the generator's mainloop block for the format: its lowering (blocked issue order, accumulator reset) is what
        # mma_loop executes
        self.block = MainloopBlock("bf16")
        self.lowering = self.block.lowering()
        assert st.mma_tiler[2] == self.block.k_block, (st.mma_tiler, self.block.k_block)
        if cutlass.const_expr(self.gather):
            # B on the dense TMA atom (the expert as the batch coordinate); A on the TMA's gather4 mode over the 2-D x: a
            # 32-row slab per issuing warp, the row indices from the warp's registers, the swizzled stage written as the dense
            # box writes it; the stage's transaction count is A + B as for the dense boxes (each CTA gathers its own rows)
            b_op = utils.sm100.cluster_shape_to_tma_atom_B(
                st.cluster_shape_mn, st.tiled_mma.thr_id
            )
            b_one = cute.select(
                st.b_smem_layout_staged,
                mode=list(range(cute.rank(st.b_smem_layout_staged) - 1)),
            )
            atom_b, tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, b, b_one, st.mma_tiler, st.tiled_mma, st.cluster_layout_vmnk.shape
            )
            a_one = cute.select(
                st.a_smem_layout_staged,
                mode=list(range(cute.rank(st.a_smem_layout_staged) - 1)),
            )
            tile_k = st.cta_tile_shape_mnk[2]
            K = a.shape[1]
            cta_group = (
                tcgen05.CtaGroup.TWO if st.atom_thr_size == 2 else tcgen05.CtaGroup.ONE
            )
            slab_layout = cute.make_composed_layout(
                st.a_smem_layout_staged.inner,
                0,
                cute.make_layout((SLAB_ROWS, tile_k), stride=(tile_k, 1)),
            )
            gI = cute.make_tensor(
                src.iterator, cute.make_layout((src.shape[0], K), stride=(1, 0))
            )
            atom_a, tensor_a = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensor2DGather4G2SOp(cta_group),
                a,
                slab_layout,
                (SLAB_ROWS, tile_k),
                gmem_coord_tensor=gI,
            )
            a_bytes = (  # the LDGSTS producer's bytes land on the gather barrier
                cute.size_in_bytes(st.a_dtype, a_one)
                if self.a_producer == "gather4"
                else 0
            )
            self.num_tma_load_bytes = (
                a_bytes + cute.size_in_bytes(st.b_dtype, b_one)
            ) * st.atom_thr_size
            m_src, kd = src, k_div
        else:
            tma = TmaSetup(st, a, b, None)
            atom_a, tensor_a, atom_b, tensor_b = (
                tma.atom_a,
                tma.tensor_a,
                tma.atom_b,
                tma.tensor_b,
            )
            self.num_tma_load_bytes = tma.tx_bytes
            m_src, kd = counts, Int32(1)  # unread
        cm, cn = self.cluster_shape_mn
        grid = (cm, cn, max_active_clusters)
        self.kernel(
            st.tiled_mma,
            atom_a,
            tensor_a,
            atom_b,
            tensor_b,
            a,
            out,
            counts,
            m_src,
            kd,
            st.cluster_layout_vmnk,
            st.a_smem_layout_staged,
            st.b_smem_layout_staged,
            st.epi_tile,
            self.c_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads, 1, 1],
            cluster=(cm, cn, 1),
            stream=stream,
            use_pdl=True,
        )

    # device side
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mX: cute.Tensor,
        mOut: cute.Tensor,
        counts: cute.Tensor,
        mSrc: cute.Tensor,
        k_div: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        c_smem_layout_staged,
    ):
        st = self.st
        TMA_WARP, MMA_WARP, EPILOGUE_WARPS = (
            self.tma_warp,
            self.mma_warp,
            self.epilogue_warps,
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        # programmatic dependent launch: wait for the predecessor grid before any dependent read (counts, the operands), and
        # let the successor start its prologue at once (this grid is one persistent wave)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        use_2cta_instrs = atom_thr_size == 2
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % atom_thr_size
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()
        E = counts.shape[0]
        N = mB_nkl.shape[0]
        Tk = mOut.shape[0]
        tile_m, tile_k = st.cta_tile_shape_mnk[0], st.cta_tile_shape_mnk[2]

        # --- scaffolding: barriers, pipelines, TMEM allocator, the group tables
        if cutlass.const_expr(self.gather):
            # the gather variants: the gather and the sync pipelines' barriers too
            @cute.struct
            class SharedStorage:
                ab_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_ab_stage * 2
                ]
                a_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_ab_stage * 2
                ]
                a_sync_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_ab_stage * 2
                ]
                acc_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_acc_stage * 2
                ]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_holding_buf: cutlass.Int32
                group_off: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
                group_pre: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
        else:

            @cute.struct
            class SharedStorage:
                ab_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_ab_stage * 2
                ]
                acc_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_acc_stage * 2
                ]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_holding_buf: cutlass.Int32
                group_off: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]
                group_pre: cute.struct.MemRange[cutlass.Int32, MAX_E + 1]

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)
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
        a_pipeline = a_sync = None
        ldgsts = self.gather and self.a_producer == "ldgsts"
        if cutlass.const_expr(self.gather):
            # the gather warps' handshake on the stage: acquire on its empty barrier (released, as the TMA pipeline's, by
            # the MMA's commit); the gather4 producer arrives plainly on this full barrier (its bytes count on the TMA
            # barrier; a cp.async arrive with no cp.async to track is not reliably performed), the LDGSTS producer's arrive
            # fires when its cp.async chunks landed (the bytes of A are on this barrier then)
            a_pipeline = PipelineCpAsyncUmma.create(
                barrier_storage=storage.a_full_mbar_ptr.data_ptr(),
                num_stages=st.num_ab_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * len(self.gather_warps)
                ),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
                producer_op=pipeline.PipelineOp.AsyncLoad
                if ldgsts
                else pipeline.PipelineOp.AsyncThread,
            )
        if cutlass.const_expr(ldgsts and use_2cta_instrs):
            # the peer's rows: the non-leader's MMA warp arrives on the leader's barrier once its CTA's chunks landed
            a_sync = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.a_sync_mbar_ptr.data_ptr(),
                num_stages=st.num_ab_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=st.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(EPILOGUE_WARPS) * (2 if use_2cta_instrs else 1),
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=BAR_TMEM_ALLOC, num_threads=32 * (len(EPILOGUE_WARPS) + 1)
        )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=EPILOGUE_WARPS[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        s_off = storage.group_off.get_tensor(cute.make_layout((MAX_E + 1,)))
        s_pre = storage.group_pre.get_tensor(cute.make_layout((MAX_E + 1,)))
        cluster_rows = st.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0]
        if warp_idx == EPILOGUE_WARPS[0]:
            build_group_tables(counts, E, cluster_rows, s_off, s_pre)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # --- operand staging: smem rings, multicast masks, partitions
        sA = smem.allocate_tensor(
            element_type=st.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=st.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sC = None
        if cutlass.const_expr(self.staged):  # one output subtile per epilogue warpgroup
            if cutlass.const_expr(
                isinstance(c_smem_layout_staged, cute.ComposedLayout)
            ):
                sC = smem.allocate_tensor(
                    element_type=mOut.element_type,
                    layout=c_smem_layout_staged.outer,
                    byte_alignment=128,
                    swizzle=c_smem_layout_staged.inner,
                )
            else:
                sC = smem.allocate_tensor(
                    element_type=mOut.element_type,
                    layout=c_smem_layout_staged,
                    byte_alignment=128,
                )
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(st.is_a_mcast or st.is_b_mcast or use_2cta_instrs):
            if cutlass.const_expr(
                not self.gather
            ):  # the gathered A is per CTA (no multicast)
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(st.mma_tiler, (0, None, None)), (None, None, None)
        )  # (bN, bK, RestN, RestK, RestL)
        k_tile_cnt = cute.size(gB_nkl, mode=[3])
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgB = thr_mma.partition_B(gB_nkl)
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(st.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, st.num_acc_stage)
        )
        mma_m_in_cluster = block_in_cluster_coord_vmnk[1]
        cta_n_in_cluster = block_in_cluster_coord_vmnk[2]
        # the CTA's rows inside the cluster tile (the gather variant): the M coordinate's block of ``tile_m * atom_thr_size``
        # rows, then the pair member's half (a 2-CTA record's peer gathers the rows its epilogue stores)
        cta_row_off = mma_m_in_cluster * Int32(
            tile_m * atom_thr_size
        ) + mma_tile_coord_v * Int32(tile_m)
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        cute.arch.barrier()  # the group tables are complete

        raster_m = self.knobs.raster_along == "m"
        tn, cn = st.mma_tiler[1], self.cluster_shape_mn[1]
        # --- roles
        if warp_idx == TMA_WARP:
            tile_sched = GroupScheduler.create(
                s_off,
                s_pre,
                counts,
                E,
                N,
                cluster_rows,
                tn,
                cn,
                raster_m,
                mma_m_in_cluster,
                cta_n_in_cluster,
            )
            if cutlass.const_expr(self.gather):
                b_only_producer_loop(
                    k_tile_cnt,
                    tile_sched,
                    ab_producer,
                    tma_atom_b,
                    tBgB,
                    tBsB,
                    b_full_mcast_mask,
                )
            else:
                grouped_tma_producer_loop(
                    st,
                    k_tile_cnt,
                    tile_sched,
                    ab_producer,
                    tma_atom_a,
                    tma_atom_b,
                    mA_mkl,
                    tBgB,
                    sA,
                    tBsB,
                    thr_mma,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    mma_m_in_cluster,
                    a_full_mcast_mask,
                    b_full_mcast_mask,
                )
        if cutlass.const_expr(self.gather):
            if warp_idx >= self.gather_warps[0] and warp_idx <= self.gather_warps[-1]:
                tile_sched = GroupScheduler.create(
                    s_off,
                    s_pre,
                    counts,
                    E,
                    N,
                    cluster_rows,
                    tn,
                    cn,
                    raster_m,
                    mma_m_in_cluster,
                    cta_n_in_cluster,
                )
                if cutlass.const_expr(ldgsts):
                    ldgsts_dense_producer_loop(
                        st,
                        k_tile_cnt,
                        tile_sched,
                        a_pipeline,
                        mX,
                        mSrc,
                        k_div,
                        sA,
                        tidx - 32 * self.gather_warps[0],
                        cta_row_off,
                        Tk,
                        mX.shape[1],
                        tile_m,
                        tile_k,
                    )
                else:
                    gather4_dense_producer_loop(
                        st,
                        k_tile_cnt,
                        tile_sched,
                        a_pipeline,
                        storage.ab_full_mbar_ptr.data_ptr(),
                        tma_atom_a,
                        mA_mkl,
                        mSrc,
                        k_div,
                        sA,
                        tidx - 32 * self.gather_warps[0],
                        cta_row_off,
                        Tk,
                        tile_m,
                        tile_k,
                    )
        if warp_idx == MMA_WARP:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = GroupScheduler.create(
                s_off,
                s_pre,
                counts,
                E,
                N,
                cluster_rows,
                tn,
                cn,
                raster_m,
                mma_m_in_cluster,
                cta_n_in_cluster,
            )
            if cutlass.const_expr(ldgsts and use_2cta_instrs):
                if is_leader_cta:
                    gather_dense_mma_loop(
                        self.lowering,
                        st.num_acc_stage,
                        st.num_ab_stage,
                        k_tile_cnt,
                        is_leader_cta,
                        tile_sched,
                        tiled_mma,
                        a_pipeline,
                        ab_consumer,
                        acc_pipeline,
                        tCrA,
                        tCrB,
                        tCtAcc_base,
                        a_sync,
                    )
                else:
                    gather_relay_loop(
                        st.num_acc_stage,
                        st.num_ab_stage,
                        k_tile_cnt,
                        tile_sched,
                        a_pipeline,
                        a_sync,
                        acc_pipeline,
                    )
            elif cutlass.const_expr(self.gather):
                gather_dense_mma_loop(
                    self.lowering,
                    st.num_acc_stage,
                    st.num_ab_stage,
                    k_tile_cnt,
                    is_leader_cta,
                    tile_sched,
                    tiled_mma,
                    a_pipeline,
                    ab_consumer,
                    acc_pipeline,
                    tCrA,
                    tCrB,
                    tCtAcc_base,
                    None,
                )
            else:
                mma_loop(
                    self.lowering,
                    st.num_acc_stage,
                    k_tile_cnt,
                    is_leader_cta,
                    tile_sched,
                    tiled_mma,
                    ab_consumer,
                    acc_pipeline,
                    tCrA,
                    tCrB,
                    tCtAcc_base,
                )
        if warp_idx < len(EPILOGUE_WARPS):
            tmem.allocate(st.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = GroupScheduler.create(
                s_off,
                s_pre,
                counts,
                E,
                N,
                cluster_rows,
                tn,
                cn,
                raster_m,
                mma_m_in_cluster,
                cta_n_in_cluster,
            )
            grouped_epilogue_loop(
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
                mma_m_in_cluster,
                EPI_WARPGROUPS_DENSE,
                sC=sC,
                cta_v=mma_tile_coord_v,
            )
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


class GatherGroupedBlockGemm(GroupedBlockGemm):
    """The bf16 class's grouped kernel with the A rows gathered by the TMA (``x [T, K]`` unpermuted, ``src [Tk]`` the
    permutation, ``k_div`` the slots per token): no permuted activation tensor; the same bits (the rows land in the stage
    as the dense box lands them)."""

    def __init__(
        self,
        knobs: Knobs,
        chain: Union[str, GroupedChain],
        out_fmt: str = "bf16",
        a_producer: str = "gather4",
    ):
        super().__init__(knobs, chain, out_fmt, gather=True, a_producer=a_producer)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        src: cute.Tensor,
        k_div: cutlass.Int32,
        b: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._launch(
            x, b, out, counts, max_active_clusters, stream, src=src, k_div=k_div
        )


@cute.jit
def grouped_tma_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    ab_producer,
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tBgB: cute.Tensor,
    sA: cute.Tensor,
    tBsB: cute.Tensor,
    thr_mma,
    a_cta_coord,
    a_cta_layout,
    mma_m_in_cluster,
    a_mcast_mask,
    b_mcast_mask,
):
    """The dense producer with per-tile coordinates: A is re-partitioned from the group-offset view (rows ``row0 + ...``,
    an element offset, so unaligned group starts are fine), B is the dense partition indexed by the expert as the batch
    coordinate. k-tiles ascending, one slot each."""
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        real_a = cute.domain_offset((work_tile.row0, 0, 0), mA_mkl)
        gA = cute.local_tile(
            real_a, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )  # (bM, bK, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            a_cta_coord,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tAgA_slice = tAgA[(None, mma_m_in_cluster, None, 0)]
        tBgB_slice = tBgB[(None, work_tile.n_tile, None, work_tile.e)]
        ab_producer.reset()
        peek_empty = ab_producer.try_acquire()
        for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            handle = ab_producer.acquire_and_advance(peek_empty)
            cute.copy(
                tma_atom_a,
                tAgA_slice[(None, handle.count)],
                tAsA[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=a_mcast_mask,
            )
            cute.copy(
                tma_atom_b,
                tBgB_slice[(None, handle.count)],
                tBsB[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=b_mcast_mask,
            )
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = ab_producer.try_acquire()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    ab_producer.tail()


@cute.jit
def b_only_producer_loop(
    k_tile_cnt: Int32,
    tile_sched,
    ab_producer,
    tma_atom_b: cute.CopyAtom,
    tBgB: cute.Tensor,
    tBsB: cute.Tensor,
    b_mcast_mask,
):
    """The TMA producer of B alone (the gather variant: A comes through the gather warps' instructions, counted on the
    same stage barrier, whose transaction count this producer sets for A + B): the expert as the batch coordinate,
    k-tiles ascending."""
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        tBgB_slice = tBgB[(None, work_tile.n_tile, None, work_tile.e)]
        ab_producer.reset()
        peek_empty = ab_producer.try_acquire()
        for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            handle = ab_producer.acquire_and_advance(peek_empty)
            cute.copy(
                tma_atom_b,
                tBgB_slice[(None, handle.count)],
                tBsB[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=b_mcast_mask,
            )
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = ab_producer.try_acquire()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    ab_producer.tail()


@cute.jit
def gather4_dense_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    a_pipeline,
    ab_full_bar0: cute.Pointer,
    tma_atom_a: cute.CopyAtom,
    mX: cute.Tensor,
    mSrc: cute.Tensor,
    k_div: Int32,
    sA: cute.Tensor,
    gidx: Int32,
    cta_row_off: Int32,
    Tk: Int32,
    tile_m: cutlass.Constexpr,
    tile_k: cutlass.Constexpr,
):
    """The gather4 producer of A (the gather warps, one per 32-row slab of the CTA's A tile). Warp ``w`` owns tile rows
    ``32 w .. 32 w + 31``: per tile every lane loads the slab's token indices into registers (``src[p] // k_div`` for the
    group's rows, -1 past its end: those rows are zero-filled and never stored), per k-tile the warp issues the slab's
    eight TMA ``gather4`` instructions (4 rows x ``tile_k`` each, the swizzled stage written as the dense box writes it,
    the K tail zero-filled), their bytes counted on the TMA pipeline's full barrier of the stage (whose transaction count
    the TMA warp sets for A + B). The stage is acquired on the gather pipeline's empty barrier, released (as the TMA
    pipeline's) by the MMA's commit; the gather pipeline's arrive marks the issue."""
    warp = cute.arch.make_warp_uniform(gidx // Int32(32))
    slab_elems = SLAB_ROWS * tile_k
    a_stage_elems = tile_m * tile_k
    slab_off = cute.assume(warp * Int32(slab_elems), divby=1024)
    sA_slab = cute.make_tensor(
        sA.iterator + slab_off,
        cute.make_layout(
            (SLAB_ROWS, tile_k, st.num_ab_stage), stride=(tile_k, 1, a_stage_elems)
        ),
    )
    sA_slab = cute.group_modes(sA_slab, 0, 2)
    idx = cute.make_rmem_tensor(cute.make_layout((SLAB_ROWS,)), Int32)
    a_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, st.num_ab_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        row_base = work_tile.row0 + cta_row_off
        row_end = cutlass.min(work_tile.row_end, Tk)
        slab_base = row_base + warp * Int32(SLAB_ROWS)
        last = row_end - Int32(1)
        for j in cutlass.range_constexpr(SLAB_ROWS):
            # rows past the group's end read the group's last row (never stored): a valid box, where an out-of-bounds
            # box costs the TMA a long fill
            idx[j] = mSrc[cutlass.min(slab_base + Int32(j), last)] // k_div
        gA = cute.local_tile(
            cute.domain_offset((slab_base, 0), mX), (SLAB_ROWS, tile_k), (0, None)
        )
        n_kt = cute.size(gA, mode=[2])
        gI = cute.make_tensor(
            idx.iterator,
            cute.make_layout((SLAB_ROWS, tile_k, n_kt), stride=(1, 0, 0)),
        )
        tAsA, tAgA, tAgI = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sA_slab,
            [cute.group_modes(gA, 0, 2), cute.group_modes(gI, 0, 2)],
        )
        a_state.reset_count()
        peek = cutlass.Boolean(1)
        if a_state.count < k_tile_cnt:
            peek = a_pipeline.producer_try_acquire(a_state)
        for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            a_pipeline.producer_acquire(a_state, peek)
            kt = a_state.count
            stage = a_state.index
            cute.copy(
                tma_atom_a,
                [tAgA[(None, kt)], tAgI[(None, kt)]],
                tAsA[(None, stage)],
                tma_bar_ptr=ab_full_bar0 + stage,
            )
            a_pipeline.producer_commit(a_state)
            a_state.advance()
            peek = cutlass.Boolean(1)
            if a_state.count < k_tile_cnt:
                peek = a_pipeline.producer_try_acquire(a_state)
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    a_pipeline.producer_tail(a_state)


@cute.jit
def gather_dense_mma_loop(
    lowering,
    num_acc_stage: cutlass.Constexpr,
    num_ab_stage: cutlass.Constexpr,
    k_tile_cnt: Int32,
    is_leader_cta,
    tile_sched,
    tiled_mma: cute.TiledMma,
    a_pipeline,
    ab_consumer,
    acc_pipeline,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtAcc_base: cute.Tensor,
    a_sync,
):
    """``mma_loop`` with the stage behind two producers: the TMA pipeline's full barrier (A's gathered bytes and B's box)
    and the gather pipeline's (the gather warps issued), both waited per k-tile, both released by the MMA's commit (the
    leader's commit releases the pair's barriers on a 2-CTA record; the peer's MMA warp only keeps the accumulator state
    in step). The same instruction sequence as the dense producer's loop: the lowering's blocks and inner steps
    ascending."""
    assert cute.size(tCrA, mode=[2]) == lowering.inner, (
        "one smem stage must hold exactly the lowering's inner steps"
    )
    use_sync = a_sync is not None
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, num_acc_stage
    )
    a_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, num_ab_stage
    )
    sync_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, num_ab_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
        ab_consumer.reset()
        a_state.reset_count()
        peek_full = cutlass.Boolean(1)
        if is_leader_cta:
            peek_full = ab_consumer.try_wait()
            acc_pipeline.producer_acquire(acc_state)
        for k_tile in range(k_tile_cnt):
            if is_leader_cta:
                a_pipeline.consumer_wait(a_state)
                if cutlass.const_expr(use_sync):
                    a_sync.consumer_wait(sync_state)
                handle = ab_consumer.wait_and_advance(peek_full)
                for j in lowering.inner_steps():
                    tiled_mma.set(
                        tcgen05.Field.ACCUMULATE, lowering.accumulate(k_tile, j)
                    )
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, j, handle.index)],
                        tCrB[(None, None, j, handle.index)],
                        tCtAcc,
                    )
                handle.release()
                a_pipeline.consumer_release(a_state)
                a_state.advance()
                if cutlass.const_expr(use_sync):
                    a_sync.consumer_release(sync_state)
                    sync_state.advance()
                peek_full = cutlass.Boolean(1)
                if handle.count + 1 < k_tile_cnt:
                    peek_full = ab_consumer.try_wait()
        if is_leader_cta:
            acc_pipeline.producer_commit(acc_state)
        acc_state.advance()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    acc_pipeline.producer_tail(acc_state)


@cute.jit
def ldgsts_dense_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    a_pipeline,
    mX: cute.Tensor,
    mSrc: cute.Tensor,
    k_div: Int32,
    sA: cute.Tensor,
    gidx: Int32,
    cta_row_off: Int32,
    Tk: Int32,
    K: Int32,
    tile_m: cutlass.Constexpr,
    tile_k: cutlass.Constexpr,
):
    """The LDGSTS producer of A (one warpgroup): thread ``gidx`` owns 16-byte chunk ``gidx % 8`` of rows ``gidx // 8 + 16 i``
    of the CTA's A tile per k-tile, copied by ``cp.async`` from ``x[src[p] // k_div]`` into the swizzled stage (the smem
    pointer applies the swizzle: the chunk lands where the dense box lands it). Rows past the group's end and chunks past
    K are predicated off (zero-filled, never stored). The stage is acquired on the gather pipeline's empty barrier (released
    by the MMA's commit); the commit is the ``cp.async`` arrive that fires when the thread's chunks landed."""
    chunk_elems = A_CHUNK_BYTES // (mX.element_type.width // 8)
    chunks_per_row = tile_k // chunk_elems
    rows_per_pass = (32 * LDGSTS_WARPS) // chunks_per_row
    passes = tile_m // rows_per_pass
    atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        mX.element_type,
        num_bits_per_copy=8 * A_CHUNK_BYTES,
    )
    row_in_pass = gidx // Int32(chunks_per_row)
    chunk = gidx % Int32(chunks_per_row)
    a_stage_elems = tile_m * tile_k
    tok_row = cute.make_rmem_tensor(cute.make_layout((passes,)), Int32)
    row_ok = cute.make_rmem_tensor(cute.make_layout((passes,)), cutlass.Boolean)
    pred1 = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Boolean)
    a_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, st.num_ab_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        row_base = work_tile.row0 + cta_row_off
        row_end = cutlass.min(work_tile.row_end, Tk)
        last = row_end - Int32(1)
        for i in cutlass.range_constexpr(passes):
            p = row_base + row_in_pass + Int32(i * rows_per_pass)
            row_ok[i] = p < row_end
            tok_row[i] = mSrc[cutlass.min(p, last)] // k_div
        a_state.reset_count()
        peek = cutlass.Boolean(1)
        if a_state.count < k_tile_cnt:
            peek = a_pipeline.producer_try_acquire(a_state)
        for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            a_pipeline.producer_acquire(a_state, peek)
            kt = a_state.count
            stage = a_state.index
            # the K tail of a partial k-tile (B is zero there)
            k_chunk_ok = kt * Int32(tile_k) + chunk * Int32(chunk_elems) < K
            for i in cutlass.range_constexpr(passes):
                r = row_in_pass + Int32(i * rows_per_pass)
                src_off = cute.assume(
                    tok_row[i] * K + kt * Int32(tile_k) + chunk * Int32(chunk_elems),
                    divby=chunk_elems,
                )
                dst_off = cute.assume(
                    stage * Int32(a_stage_elems)
                    + r * Int32(tile_k)
                    + chunk * Int32(chunk_elems),
                    divby=chunk_elems,
                )
                src = cute.make_tensor(
                    mX.iterator + src_off, cute.make_layout((chunk_elems,))
                )
                dst = cute.make_tensor(
                    sA.iterator + dst_off, cute.make_layout((chunk_elems,))
                )
                pred1[0] = row_ok[i] & k_chunk_ok
                cute.copy_atom_call(atom, src, dst, pred=pred1)
            a_pipeline.producer_commit(a_state)
            a_state.advance()
            peek = cutlass.Boolean(1)
            if a_state.count < k_tile_cnt:
                peek = a_pipeline.producer_try_acquire(a_state)
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    a_pipeline.producer_tail(a_state)


@cute.jit
def gather_relay_loop(
    num_acc_stage: cutlass.Constexpr,
    num_ab_stage: cutlass.Constexpr,
    k_tile_cnt: Int32,
    tile_sched,
    a_pipeline,
    a_sync,
    acc_pipeline,
):
    """The non-leader CTA's MMA warp of a 2-CTA record with the LDGSTS producer: per k-tile it waits for its CTA's chunks
    (the local gather pipeline's full barrier) and arrives on the leader's sync barrier of the stage (the arrive lands in
    the leading CTA), so the leader's MMA reads landed rows in both CTAs. The stage's release is the leader's commit (both
    CTAs' empty barriers). The accumulator pipeline's state advances per tile as the leader's."""
    a_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, num_ab_stage
    )
    sync_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, num_ab_stage
    )
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, num_acc_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        for _k_tile in range(k_tile_cnt):
            a_pipeline.consumer_wait(a_state)
            a_sync.producer_commit(sync_state)
            a_state.advance()
            sync_state.advance()
        acc_state.advance()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    a_sync.producer_tail(sync_state)
    acc_pipeline.producer_tail(acc_state)
