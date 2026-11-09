# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The kernel body: scaffolding (smem, pipelines, barriers, TMEM), the role dispatch and the composition of
the blocks. ``BlockGemm`` is the unscaled body (the engine's fold in the mainloop, ``cvt.rn`` in the epilogue; bf16
inputs, or fp8 inputs storing the fp32 partials of the workspace split); ``ScaledBlockGemm`` the fp8 body with the
per-tensor chain (``mul.rn alpha`` then ``cvt.rn``), ``alpha = mul.rn(a_scale, b_scale)`` formed once per CTA on the
device from the two per-tensor scales.
``BiasBlockGemm`` is the bf16 body with the biased chain (``add.rn bias`` of the output column's bias before the
``cvt.rn``; ``spec.BIAS_NOTE``), which takes the (N,) bias as one more operand.
"""

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline

# FlashInfer's copy of the DSL's persistent tile scheduler (the DSL's own module is deprecated and warns per use)
from flashinfer.cute_dsl.attention.fmha.helpers import (
    static_persistent_tile_scheduler as tile_scheduler,
)
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum

from ....batch_invariant.fp_interface.cute import CuteF32
from ..generator import Knobs, MainloopBlock
from ..spec import EpilogueChain
from .epilogue import epilogue_loop
from .mainloop import mma_loop
from .operands import (
    ACC_DTYPE,
    BAR_TMEM_ALLOC,
    EPILOGUE_WARPS,
    IN_DT,
    MMA_WARP,
    OUT_DT,
    THREADS,
    TMA_WARP,
    StaticPlan,
    TmaSetup,
    build_grid,
)
from .producer import tma_producer_loop


class BlockGemm:
    """``C[m, n, l] = cvt(sum_k A[m, k, l] B[n, k, l])`` for the knobs; M, N, K (and L) dynamic. ``chunks > 1`` with
    ``knobs.cta_chunks``: the in-CTA chunk fold of the split class (A and B as the split class's ``(rows, Kc, C)`` views,
    C with ``L = 1``; the chunk partials folded in order through the TMEM running sum). ``in_fmt`` is the operand format
    (``bf16`` / ``e4m3`` / ``e5m2``); the chain is the specification's object for the output format (unscaled and
    unbiased here; the subclasses set ``scaled`` / ``biased``)."""

    scaled = False
    biased = False

    def __init__(self, knobs: Knobs, in_fmt: str, out_fmt: str, chunks: int = 1):
        self.knobs = knobs
        self.in_fmt = in_fmt
        self.chunks = chunks if knobs.cta_chunks else 1
        assert self.chunks == 1 or chunks > 1, (knobs, chunks)
        self.cluster_shape_mn = knobs.cluster_shape_mn
        self.out_fmt = out_fmt
        # the chain is the spec's object; the epilogue block applies it (the fp32 ops, then the vector cvt)
        self.chain = EpilogueChain.for_class(out_fmt, self.scaled, self.biased)
        assert self.chain.scaled == self.scaled and self.chain.out_fmt == out_fmt
        assert self.chain.biased == self.biased
        assert not self.biased or out_fmt != "f32", (
            "the split's partials are bias-free: the fixup adds the bias"
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self._launch(a, b, c, max_active_clusters, stream, None, None, None)

    # host side (trace time)
    def _launch(self, a, b, c, max_active_clusters, stream, a_scale, b_scale, bias):
        if cutlass.const_expr(a.element_type != b.element_type):
            raise TypeError(
                f"A and B must have one type: {a.element_type} != {b.element_type}"
            )
        st = StaticPlan(
            self.knobs,
            a.element_type,
            b.element_type,
            c.element_type,
            LayoutEnum.from_tensor(a).mma_major_mode(),
            LayoutEnum.from_tensor(b).mma_major_mode(),
            LayoutEnum.from_tensor(c),
            chunks=self.chunks,
            in_fmt=self.in_fmt,
        )
        self.st = st
        assert OUT_DT[self.out_fmt] is c.element_type, (self.out_fmt, c.element_type)
        assert a.element_type is IN_DT[self.in_fmt], (self.in_fmt, a.element_type)
        # the generator's mainloop block for this format: its lowering is what mma_loop executes, and one smem stage holds
        # exactly one block of it
        self.block = MainloopBlock(self.in_fmt)
        self.lowering = self.block.lowering()
        assert st.mma_tiler[2] == self.block.k_block, (st.mma_tiler, self.block.k_block)
        tma = TmaSetup(st, a, b, c)
        self.num_tma_load_bytes = tma.tx_bytes
        params, grid = build_grid(st, c, max_active_clusters)
        self.kernel(
            st.tiled_mma,
            tma.atom_a,
            tma.tensor_a,
            tma.atom_b,
            tma.tensor_b,
            tma.atom_c,
            tma.tensor_c,
            st.cluster_layout_vmnk,
            st.a_smem_layout_staged,
            st.b_smem_layout_staged,
            st.c_smem_layout_staged,
            st.epi_tile,
            params,
            a_scale,
            b_scale,
            bias,
        ).launch(
            grid=grid,
            block=[THREADS, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
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
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged,
        epi_tile: cute.Tile,
        tile_sched_params: tile_scheduler.PersistentTileSchedulerParams,
        a_scale: Optional[cute.Tensor],
        b_scale: Optional[cute.Tensor],
        mBias: Optional[cute.Tensor],
    ):
        st = self.st
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        alpha = None
        if cutlass.const_expr(self.scaled):
            alpha = CuteF32.mul(
                a_scale[0], b_scale[0]
            )  # the chain's alpha: one .rn multiply per CTA on the device
        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)
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

        # --- scaffolding: barriers, pipelines, TMEM allocator
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, st.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

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
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # --- operand staging: smem rings, multicast masks, global tiles and partitions
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
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(st.is_a_mcast or st.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )  # (bM, bK, RestM, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(st.mma_tiler, (0, None, None)), (None, None, None)
        )  # (bN, bK, RestN, RestK, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(st.mma_tiler, (None, None, 0)), (None, None, None)
        )  # (bM, bN, RestM, RestN, RestL)
        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCcC = None
        if cutlass.const_expr(self.biased):
            # the output coordinates, tiled and partitioned exactly like C: the epilogue reads the bias of each
            # element's column through them (and skips the columns of a partial tile that the store clips)
            cC_mnl = cute.local_tile(
                cute.make_identity_tensor(mC_mnl.shape),
                cute.slice_(st.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
            tCcC = thr_mma.partition_C(cC_mnl)
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
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
        tCrA = tiled_mma.make_fragment_A(sA)  # (MMA, MMA_M, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)  # (MMA, MMA_N, MMA_K, STAGE)
        acc_shape = tiled_mma.partition_shape_C(st.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, st.num_tmem_stages)
        )  # + the running sum of the in-CTA chunk fold
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # --- roles
        if warp_idx == TMA_WARP:
            tile_sched = tile_scheduler.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            tma_producer_loop(
                atom_thr_size,
                k_tile_cnt,
                tile_sched,
                ab_producer,
                tma_atom_a,
                tma_atom_b,
                tAgA,
                tBgB,
                tAsA,
                tBsB,
                a_full_mcast_mask,
                b_full_mcast_mask,
                st.cta_chunks,
            )
        if warp_idx == MMA_WARP:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = tile_scheduler.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
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
                st.cta_chunks,
            )
        sC = smem.allocate_tensor(
            element_type=st.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )
        if warp_idx < MMA_WARP:
            tmem.allocate(st.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = tile_scheduler.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            epilogue_loop(
                st,
                self.chain,
                atom_thr_size,
                tidx,
                warp_idx,
                tile_sched,
                tma_atom_c,
                tCtAcc_base,
                sC,
                tCgC,
                epi_tile,
                acc_pipeline,
                alpha,
                st.cta_chunks,
                mBias,
                tCcC,
            )
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


class ScaledBlockGemm(BlockGemm):
    """``C = cvt(mul.rn(sum_k A B, mul.rn(a_scale, b_scale)))``: the fp8 classes' body; the scales are 1-element fp32
    tensors."""

    scaled = True

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        a_scale: cute.Tensor,
        b_scale: cute.Tensor,
    ):
        self._launch(a, b, c, max_active_clusters, stream, a_scale, b_scale, None)


class BiasBlockGemm(BlockGemm):
    """``C = cvt(add(sum_k A B, bias[n]))``: the bf16 body with the biased chain; ``bias`` is the (N,) bf16 vector, read
    in the epilogue (the unsplit class and the in-CTA chunk fold; the split's partial kernel never takes it)."""

    biased = True

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        bias: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self._launch(a, b, c, max_active_clusters, stream, None, None, bias)
