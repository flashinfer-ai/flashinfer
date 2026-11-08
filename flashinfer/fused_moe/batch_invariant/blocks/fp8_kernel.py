# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The grouped GEMM kernel body of the fp8 per-tensor MoE classes: the bf16 grouped body (:mod:`.kernel`) on e4m3 operands
(the dense fp8 class's ``kind::f8f6f4`` instruction and 32-wide K step through ``MainloopBlock("e4m3")``), with the per-expert
weight scales, the A operand's scale and the intermediate multiplier as further operands and the W8A8 chains in the
epilogue (:mod:`.fp8_epilogue`).

Operands: A ``(Tk, K, 1)`` K-major e4m3 (the permuted activations), B ``(N, K, E)`` K-major e4m3 (the stacked expert weights),
out ``(Tk, Nout)`` row-major (the uint8 bytes of e4m3 for GEMM1, bf16 for GEMM2, fp32 for the oracle; ``Nout = N`` or ``N / 2``
for the gated chain), ``counts (E,)`` int32, ``alpha (E,)`` fp32 (``s_w[e]``), ``sa (1,)`` fp32 (``s_x`` / ``s_inter``), ``si (1,)``
fp32 (``m_inter``). The persistent grid is a dynamic launch argument, as for the bf16 body. The gather variant
(``ScaledGatherGroupedBlockGemmFp8``) takes the unpermuted ``xq (T, K)`` and the plan's ``src (Tk,)`` in place of the permuted
A: the bf16 body's gather warps (:mod:`.kernel`) issue the TMA's ``gather4`` loads of the rows into the same stage.
"""

from __future__ import annotations

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
from .fp8_epilogue import (
    Fp8Chain,
    grouped_epilogue_loop_fp8,
    staged_store_fp8,
    staging_bytes,
    staging_layout,
)
from .kernel import (
    A_PRODUCERS,
    GATHER_SMEM_ALIGN,
    LDGSTS_WARPS,
    SLAB_ROWS,
    b_only_producer_loop,
    gather4_dense_producer_loop,
    gather_dense_mma_loop,
    gather_relay_loop,
    grouped_tma_producer_loop,
    ldgsts_dense_producer_loop,
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

IN_FMT = "e4m3"
EPILOGUE_WARPS, MMA_WARP, TMA_WARP, THREADS = warp_roles(EPI_WARPGROUPS_DENSE)


class ScaledGroupedBlockGemmFp8:
    """``out[p, n] = chain(sum_k A[p, k] B[n, k, e(p)], alpha = mul(sa[0], alpha[e(p)]))`` for every row ``p`` of every group."""

    def __init__(
        self,
        knobs: Knobs,
        chain: Fp8Chain,
        gather: bool = False,
        a_producer: str = "gather4",
    ):
        assert isinstance(chain, Fp8Chain), chain
        assert a_producer in A_PRODUCERS, a_producer
        self.knobs, self.chain, self.out_fmt = knobs, chain, chain.out_fmt
        self.cluster_shape_mn = knobs.cluster_shape_mn
        self.gather, self.a_producer = gather, a_producer
        epi, mma, tma, threads = warp_roles(EPI_WARPGROUPS_DENSE)
        self.gather_warps = ()
        if gather:
            # the gather warps sit between the epilogue warpgroups and the MMA warp (:mod:`.kernel`)
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
        alpha: cute.Tensor,
        sa: cute.Tensor,
        si: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._launch(a, b, out, counts, alpha, sa, si, max_active_clusters, stream)

    # host side (trace time)
    def _launch(
        self,
        a,
        b,
        out,
        counts,
        alpha,
        sa,
        si,
        max_active_clusters,
        stream,
        src=None,
        k_div=None,
    ):
        """``src`` / ``k_div`` (the gather variant): ``a`` is then the unpermuted ``xq [T, K]`` and expanded row ``p`` takes
        ``xq[src[p] // k_div]`` through the TMA's gather4 mode; otherwise ``a`` is the permuted ``[Tk, K, 1]``."""
        if cutlass.const_expr(a.element_type != b.element_type):
            raise TypeError(
                f"A and B must have one type: {a.element_type} != {b.element_type}"
            )
        assert a.element_type is IN_DT[IN_FMT], a.element_type
        assert (src is not None) == self.gather
        self.staged = staged_store_fp8(self.chain.out_fmt)

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
                in_fmt=IN_FMT,
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
        self.block = MainloopBlock(IN_FMT)
        self.lowering = self.block.lowering()
        assert st.mma_tiler[2] == self.block.k_block, (st.mma_tiler, self.block.k_block)
        if cutlass.const_expr(self.gather):
            # B on the dense TMA atom; A on the TMA's gather4 mode over the 2-D xq (the bf16 body's construction)
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
            alpha,
            sa,
            si,
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
        mAlpha: cute.Tensor,
        mSa: cute.Tensor,
        mSi: cute.Tensor,
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
        # programmatic dependent launch: wait for the predecessor grid before any dependent read (counts, the operands, the
        # scales), and let the successor start its prologue at once
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
            # the gather warps' stage handshake (:mod:`.kernel`): a plain "issued" arrive for the gather4 producer, the
            # cp.async arrive for the LDGSTS producer
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
        if cutlass.const_expr(self.staged):
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
        )
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
        # the CTA's rows inside the cluster tile (the gather variant; :mod:`.kernel`)
        cta_row_off = mma_m_in_cluster * Int32(
            tile_m * atom_thr_size
        ) + mma_tile_coord_v * Int32(tile_m)
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        cute.arch.barrier()

        raster_m = self.knobs.raster_along == "m"
        tn, cn = st.mma_tiler[1], self.cluster_shape_mn[1]
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
            grouped_epilogue_loop_fp8(
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
                mAlpha,
                mSa,
                mSi,
                EPI_WARPGROUPS_DENSE,
                sC=sC,
                cta_v=mma_tile_coord_v,
            )
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


class ScaledGatherGroupedBlockGemmFp8(ScaledGroupedBlockGemmFp8):
    """:class:`ScaledGroupedBlockGemmFp8` with the A rows gathered by the TMA (``xq [T, K]`` unpermuted, ``src [Tk]`` the
    permutation, ``k_div`` the slots per token): no permuted activation tensor; the same bits."""

    def __init__(self, knobs: Knobs, chain: Fp8Chain, a_producer: str = "gather4"):
        super().__init__(knobs, chain, gather=True, a_producer=a_producer)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        src: cute.Tensor,
        k_div: cutlass.Int32,
        b: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        alpha: cute.Tensor,
        sa: cute.Tensor,
        si: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._launch(
            x,
            b,
            out,
            counts,
            alpha,
            sa,
            si,
            max_active_clusters,
            stream,
            src=src,
            k_div=k_div,
        )
