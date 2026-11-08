# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The grouped GEMM kernel body of the fp8 per-tensor MoE classes: the bf16 grouped body (:mod:`.kernel`) on e4m3 operands
(the dense fp8 class's ``kind::f8f6f4`` instruction and 32-wide K step through ``MainloopBlock("e4m3")``), with the per-expert
weight scales, the A operand's scale and the intermediate multiplier as further operands and the W8A8 chains in the
epilogue (:mod:`.fp8_epilogue`).

Operands: A ``(Tk, K, 1)`` K-major e4m3 (the permuted activations), B ``(N, K, E)`` K-major e4m3 (the stacked expert weights),
out ``(Tk, Nout)`` row-major (the uint8 bytes of e4m3 for GEMM1, bf16 for GEMM2, fp32 for the oracle; ``Nout = N`` or ``N / 2``
for the gated chain), ``counts (E,)`` int32, ``alpha (E,)`` fp32 (``s_w[e]``), ``sa (1,)`` fp32 (``s_x`` / ``s_inter``), ``si (1,)``
fp32 (``m_inter``). The persistent grid is a dynamic launch argument, as for the bf16 body.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync
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
from .kernel import grouped_tma_producer_loop
from .scheduler import (
    EPI_WARPGROUPS_DENSE,
    MAX_E,
    TABLE_SMEM_BYTES,
    GroupScheduler,
    build_group_tables,
    warp_roles,
)
from .. import pdl

IN_FMT = "e4m3"
EPILOGUE_WARPS, MMA_WARP, TMA_WARP, THREADS = warp_roles(EPI_WARPGROUPS_DENSE)


class ScaledGroupedBlockGemmFp8:
    """``out[p, n] = chain(sum_k A[p, k] B[n, k, e(p)], alpha = mul(sa[0], alpha[e(p)]))`` for every row ``p`` of every group."""

    def __init__(self, knobs: Knobs, chain: Fp8Chain):
        assert isinstance(chain, Fp8Chain), chain
        self.knobs, self.chain, self.out_fmt = knobs, chain, chain.out_fmt
        self.cluster_shape_mn = knobs.cluster_shape_mn

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
    def _launch(self, a, b, out, counts, alpha, sa, si, max_active_clusters, stream):
        if cutlass.const_expr(a.element_type != b.element_type):
            raise TypeError(
                f"A and B must have one type: {a.element_type} != {b.element_type}"
            )
        assert a.element_type is IN_DT[IN_FMT], a.element_type
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
        self.st = st
        self.c_smem_layout_staged = (
            staging_layout(out.element_type, st.epi_tile, EPI_WARPGROUPS_DENSE)
            if self.staged
            else st.a_smem_layout_staged
        )
        self.block = MainloopBlock(IN_FMT)
        self.lowering = self.block.lowering()
        assert st.mma_tiler[2] == self.block.k_block, (st.mma_tiler, self.block.k_block)
        tma = TmaSetup(st, a, b, None)
        self.num_tma_load_bytes = tma.tx_bytes
        cm, cn = self.cluster_shape_mn
        grid = (cm, cn, max_active_clusters)
        self.kernel(
            st.tiled_mma,
            tma.atom_a,
            tma.tensor_a,
            tma.atom_b,
            tma.tensor_b,
            out,
            counts,
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
            block=[THREADS, 1, 1],
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
        mOut: cute.Tensor,
        counts: cute.Tensor,
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
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        # programmatic dependent launch: wait for the predecessor grid before any dependent read (counts, the operands, the
        # scales), and let the successor start its prologue at once
        cute.arch.griddepcontrol_wait()
        pdl.trigger_early()
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
        Tk = mA_mkl.shape[0]

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
        if warp_idx < MMA_WARP:
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
