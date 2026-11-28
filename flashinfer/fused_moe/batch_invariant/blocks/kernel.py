# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The grouped GEMM kernel body on the dense and the block-scaled blocks: the operand block (:mod:`.operands`) under the
group scheduler, with per-tile TMA coordinates (A offset by the group's rows through ``cute.domain_offset``, B indexed by
the expert as the batch coordinate) and the group-masked epilogue.

One body for the bf16, the fp8 per-tensor and the NVFP4 classes: the operand block (:class:`.operands.DenseOperands`, keyed
by the chain's operand format: the format's dtype, alignment, K-steps per stage, mainloop block and static plan; or
:class:`.operands.BlockScaledOperands`: the block-scaled plan of the instruction class, a scale-factor ring beside each
operand ring, the scale factors staged to TMEM per k-tile, the group tables at the 128-row scale atom with the dense packing's
second table) and the epilogue block (the chain, :class:`.epilogue.GroupedEpilogueBlock`: the bf16 chains, the W8A8 chains
with their scale operands, the W4A4 chains with the per-row scale and the static class's fused quantization) are resolved at
trace time, so every format-dependent choice is a ``const_expr`` branch and each format compiles to its own kernel.

Operands: A ``(Tk, K, 1)`` K-major (the permuted activations; for the 4-bit formats the byte-packed ``(rows, K / 2, 1)``
viewed as fp4), B ``(N, K, E)`` K-major (the stacked expert weights, ``l`` = expert), out ``(Tk, Nout)`` row-major (``Nout =
N`` for the plain and the ungated chains, ``N / 2`` for the gated chain; bf16 / f16 / f32, the uint8 bytes of e4m3 for the fp8
classes' GEMM1, the packed e2m1 codes for the static NVFP4 class's GEMM1), ``counts (E,)`` int32 (tokens per expert, in group
order); for a chain with scale operands ``alpha (E,)`` fp32 (``s_w[e]``), ``sa (1,)`` fp32 (``s_x`` / ``s_inter``), ``si
(1,)`` fp32 (``m_inter``); for the block-scaled chains SFA / SFB the flat uint8 128x4 swizzled scale buffers (SFB: ``E``
consecutive buffers), ``alpha (E,)`` the experts' alphas, ``pts (rows_sf,)`` the per-row scales at the padded positions,
``aux (E,)`` the static class's per-expert scales and ``out_sf`` its scale-byte output. The group's rows are ``[off_e, off_e
+ counts[e])`` with ``off`` the exclusive prefix of ``counts`` (the dense packing); the block-scaled bodies' scale factors
start at a 128-row atom per group (the dense packing's second offset table, or the padded packing's row offsets).
"""

from __future__ import annotations

from typing import Optional, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blockscaled_layout as bs_layout
from cutlass import Int32
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ....gemm.batch_invariant.blocks.blockscaled import blockscaled_mma_loop
from ....gemm.batch_invariant.blocks.mainloop import mma_loop
from ....gemm.batch_invariant.blocks.operands import (
    ACC_DTYPE,
    BAR_TMEM_ALLOC,
    TmaSetup,
)
from ....gemm.batch_invariant.generator import Knobs
from ....gemm.batch_invariant.spec import SF_ATOM_ROWS
from .epilogue import (
    BAR_EPI_JOIN,
    GroupedChain,
    GroupedEpilogueBlock,
    grouped_epilogue_loop,
    staged_store,
    staging_bytes,
    staging_layout,
)
from .operands import (  # noqa: F401  (re-exported: the operand blocks' home moved here from this module)
    GROUPED_K_BLOCKS,
    K_BLOCKS_OF,
    BlockScaledOperands,
    DenseOperands,
    operands_for,
)
from .scheduler import (
    EPI_WARPGROUPS_DENSE,
    MAX_E,
    TABLE_SMEM_BYTES,
    GroupScheduler,
    build_group_tables,
    warp_roles,
)
from .. import pdl

CHAINS = ("plain", "swiglu", "relu2")
#: The dense bodies' warp roles (the block-scaled body takes its own epilogue warpgroup count from its operand block).
EPILOGUE_WARPS, MMA_WARP, TMA_WARP, THREADS = warp_roles(EPI_WARPGROUPS_DENSE)
#: The group alignment of the block-scaled classes' scale factors (rows): one scale-factor atom.
GROUP_ALIGN = SF_ATOM_ROWS


class GroupedBlockGemm:
    """``out[p, n] = chain(sum_k A[p, k] B[n, k, e(p)])`` for every row ``p`` of every group (with the fp8 classes' chains
    ``alpha = mul(sa[0], alpha[e(p)])`` scales the accumulators; with the block-scaled classes' chains the block scales are
    applied inside the instruction and the W4A4 row scale ``mul(alpha[e(p)], pts[p])`` by the chain). The persistent grid
    (``max_active_clusters``, in clusters) is a dynamic launch argument: which cluster computes a tile, never the bits. The
    chain's operand format selects the operand block; a chain with scale operands launches through :meth:`with_scales`, the
    block-scaled chains through :meth:`block_scaled` / :meth:`block_scaled_static`."""

    def __init__(
        self,
        knobs: Knobs,
        chain: Union[str, GroupedEpilogueBlock],
        out_fmt: str = "bf16",
        *,
        packing: str = "dense",
    ):
        if isinstance(chain, str):  # the bf16 class's names
            chain = GroupedChain.legacy(chain, out_fmt)
        assert isinstance(chain, GroupedEpilogueBlock), chain
        self.knobs, self.chain, self.out_fmt = knobs, chain, chain.out_fmt
        self.in_fmt = chain.in_fmt
        self.operands = operands_for(self.in_fmt, packing)
        self.cluster_shape_mn = knobs.cluster_shape_mn
        self.epi_warpgroups = self.operands.epi_warpgroups
        self.epilogue_warps, self.mma_warp, self.tma_warp, self.threads = warp_roles(
            self.epi_warpgroups
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
        """The entry of the chains without scale operands (the bf16 classes)."""
        assert not self.chain.operands and not self.operands.block_scaled, self.chain
        self._launch(a, b, out, counts, max_active_clusters, stream)

    @cute.jit
    def with_scales(
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
        """The entry of the chains with scale operands (the fp8 per-tensor classes): ``alpha (E,)`` the per-expert weight
        scales, ``sa (1,)`` the A operand's scale, ``si (1,)`` the intermediate multiplier."""
        assert self.chain.scale_operands, self.chain
        self._launch(
            a, b, out, counts, max_active_clusters, stream, alpha=alpha, sa=sa, si=si
        )

    @cute.jit
    def block_scaled(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        alpha: cute.Tensor,
        pts: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The entry of the block-scaled chains with the W4A4 row scale: ``sfa`` / ``sfb`` the flat 128x4 scale-factor buffers,
        ``alpha (E,)`` the experts' alphas as given, ``pts (rows_sf,)`` the per-row scales at the (padded) row positions."""
        assert self.operands.block_scaled and not self.chain.needs_aux, self.chain
        self._launch(
            a,
            b,
            out,
            counts,
            max_active_clusters,
            stream,
            sfa=sfa,
            sfb=sfb,
            alpha=alpha,
            pts=pts,
            rows_sf=pts.shape[0],
        )

    @cute.jit
    def block_scaled_plain(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        rows_sf: Int32,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The entry of the block-scaled chains without further operands (the MXFP4 classes: the e8m0 scales of both operands
        are applied inside the instruction, no global or per-row scale): ``rows_sf`` the rows of the A operand's scale-factor
        buffer (the padded row count of the dense packing)."""
        assert self.operands.block_scaled and not self.chain.operands, self.chain
        self._launch(
            a,
            b,
            out,
            counts,
            max_active_clusters,
            stream,
            sfa=sfa,
            sfb=sfb,
            rows_sf=rows_sf,
        )

    @cute.jit
    def block_scaled_static(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        out: cute.Tensor,
        out_sf: cute.Tensor,
        counts: cute.Tensor,
        alpha: cute.Tensor,
        pts: cute.Tensor,
        aux: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """The static class's chains: GEMM1 (``out_fmt == "nvfp4"``: ``out`` the packed codes uint8 ``[rows, I/2]``, ``out_sf`` the
        flat 128x4 scale bytes at the (padded) row positions, ``aux[e] = gs2[e]``) and GEMM2 (``scale == "w4a4s"``: ``aux[e] =
        inv2[e]``, ``out_sf`` unread)."""
        assert self.operands.block_scaled and self.chain.needs_aux, self.chain
        self._launch(
            a,
            b,
            out,
            counts,
            max_active_clusters,
            stream,
            sfa=sfa,
            sfb=sfb,
            alpha=alpha,
            pts=pts,
            rows_sf=pts.shape[0],
            out_sf=out_sf,
            aux=aux,
        )

    # host side (trace time)
    def _launch(
        self,
        a,
        b,
        out,
        counts,
        max_active_clusters,
        stream,
        *,
        sfa=None,
        sfb=None,
        alpha=None,
        sa=None,
        si=None,
        pts=None,
        rows_sf=None,
        out_sf=None,
        aux=None,
    ):
        ops = self.operands
        if cutlass.const_expr(a.element_type != b.element_type):
            raise TypeError(
                f"A and B must have one type: {a.element_type} != {b.element_type}"
            )
        # the ring leaves room for the group tables and the epilogue staging buffers
        self.staged = staged_store(self.chain.out_fmt)
        sfa_t = sfb_t = None
        if cutlass.const_expr(ops.block_scaled):
            a = ops.view_elems(a)
            b = ops.view_elems(b)
            sfa_t = ops.sf_view_rows(sfa, rows_sf, a.shape[1])  # the scale rows: padded
            sfb_t = ops.sf_view_experts(sfb, b.shape[0], b.shape[1], b.shape[2])
            st = ops.plan(
                self.knobs,
                lambda st: (
                    ops.table_bytes
                    + (
                        staging_bytes(
                            out.element_type, st.epi_tile, self.epi_warpgroups
                        )
                        if self.staged
                        else 0
                    )
                ),
                "grouped",
            )
            self.block = st.block
            assert self.block.in_fmt == self.in_fmt, (self.block.in_fmt, self.in_fmt)
            assert st.cluster_k == 1 and not self.knobs.swap_ab and not st.breuse, (
                "the grouped body takes the dense records"
            )
        else:

            def plan(extra: int):
                return ops.plan(self.knobs, a, b, TABLE_SMEM_BYTES + extra, "grouped")

            st = plan(0)
            if self.staged:
                st = plan(
                    staging_bytes(out.element_type, st.epi_tile, self.epi_warpgroups)
                )
            self.block = ops.mainloop("grouped")
        self.st = st
        self.c_smem_layout_staged = (
            staging_layout(out.element_type, st.epi_tile, self.epi_warpgroups)
            if self.staged
            else st.a_smem_layout_staged
        )
        self.lowering = self.block.lowering()
        assert st.mma_tiler[2] == self.block.k_block, (st.mma_tiler, self.block.k_block)
        cm, cn = self.cluster_shape_mn
        grid = (cm, cn, max_active_clusters)
        if cutlass.const_expr(ops.block_scaled):
            tma = ops.tma(st, a, b, sfa_t, sfb_t)
            self.num_tma_load_bytes = tma.tx_bytes
            self.kernel(
                st.tiled_mma,
                tma.atom_a,
                tma.tensor_a,
                tma.atom_b,
                tma.tensor_b,
                out,
                counts,
                st.cluster_layout_vmnk,
                st.a_smem_layout_staged,
                st.b_smem_layout_staged,
                st.epi_tile,
                self.c_smem_layout_staged,
                alpha,
                sa,
                si,
                st.tiled_mma_sfb,
                tma.atom_sfa,
                tma.tensor_sfa,
                tma.atom_sfb,
                tma.tensor_sfb,
                st.cluster_layout_sfb_vmnk,
                st.sfa_smem_layout_staged,
                st.sfb_smem_layout_staged,
                pts,
                out_sf,
                aux,
            ).launch(
                grid=grid,
                block=[self.threads, 1, 1],
                cluster=(cm, cn, 1),
                stream=stream,
                use_pdl=True,
            )
        else:
            tma = TmaSetup(st, a, b, None)
            self.num_tma_load_bytes = tma.tx_bytes
            self.kernel(
                st.tiled_mma,
                tma.atom_a,
                tma.tensor_a,
                tma.atom_b,
                tma.tensor_b,
                out,
                counts,
                st.cluster_layout_vmnk,
                st.a_smem_layout_staged,
                st.b_smem_layout_staged,
                st.epi_tile,
                self.c_smem_layout_staged,
                alpha,
                sa,
                si,
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
        mOut: cute.Tensor,
        counts: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        c_smem_layout_staged,
        mAlpha: cute.Tensor = None,
        mSa: cute.Tensor = None,
        mSi: cute.Tensor = None,
        tiled_mma_sfb: Optional[cute.TiledMma] = None,
        tma_atom_sfa: Optional[cute.CopyAtom] = None,
        mSFA_mkl: Optional[cute.Tensor] = None,
        tma_atom_sfb: Optional[cute.CopyAtom] = None,
        mSFB_nkl: Optional[cute.Tensor] = None,
        cluster_layout_sfb_vmnk: Optional[cute.Layout] = None,
        sfa_smem_layout_staged: Optional[cute.Layout] = None,
        sfb_smem_layout_staged: Optional[cute.Layout] = None,
        mPts: Optional[cute.Tensor] = None,
        mOutSf: Optional[cute.Tensor] = None,
        mAux: Optional[cute.Tensor] = None,
    ):
        st = self.st
        ops = self.operands
        bs = ops.block_scaled
        epilogue_warps, mma_warp, tma_warp = (
            self.epilogue_warps,
            self.mma_warp,
            self.tma_warp,
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(bs):
                cpasync.prefetch_descriptor(tma_atom_sfa)
                cpasync.prefetch_descriptor(tma_atom_sfb)
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
        block_in_cluster_coord_sfb_vmnk = None
        if cutlass.const_expr(bs):
            block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
                cta_rank_in_cluster
            )
        tidx, _, _ = cute.arch.thread_idx()
        E = counts.shape[0]
        N = mB_nkl.shape[0]
        Tk = mA_mkl.shape[0]

        # --- scaffolding: barriers, pipelines, TMEM allocator, the group tables
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
                len(epilogue_warps) * (2 if use_2cta_instrs else 1),
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=BAR_TMEM_ALLOC, num_threads=32 * (len(epilogue_warps) + 1)
        )
        epi_join = pipeline.NamedBarrier(
            barrier_id=BAR_EPI_JOIN, num_threads=32 * len(epilogue_warps)
        )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=epilogue_warps[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            **ops.tmem_alloc_kwargs(st),
        )
        s_off = storage.group_off.get_tensor(cute.make_layout((MAX_E + 1,)))
        s_pre = storage.group_pre.get_tensor(cute.make_layout((MAX_E + 1,)))
        s_off_pad = None
        if cutlass.const_expr(bs and ops.dense_a):
            # the padded (atom-aligned) group offsets: the scale-factor and row-scale positions of the dense rows
            s_off_pad = smem.allocate_tensor(
                Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
            )
            s_pre_pad = smem.allocate_tensor(
                Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
            )
        cluster_rows = st.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0]
        # programmatic dependent launch: the prologue above (descriptor prefetch, barrier init, the pipelines) depends on no
        # predecessor; the wait sits before the first dependent read (counts: the group tables, then the operands and the
        # scales), and the trigger placement (pdl.py) follows it
        cute.arch.griddepcontrol_wait()
        pdl.trigger_early()
        if warp_idx == epilogue_warps[0]:
            if cutlass.const_expr(bs and ops.dense_a):
                build_group_tables(counts, E, cluster_rows, s_off, s_pre, 1)
                build_group_tables(
                    counts, E, cluster_rows, s_off_pad, s_pre_pad, ops.group_align
                )
            elif cutlass.const_expr(bs):
                build_group_tables(
                    counts, E, cluster_rows, s_off, s_pre, ops.group_align
                )
            else:
                build_group_tables(counts, E, cluster_rows, s_off, s_pre)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # --- operand staging: smem rings (A, B swizzled; the scale factors plain), multicast masks, partitions
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
        sSFA = sSFB = None
        if cutlass.const_expr(bs):
            sSFA = smem.allocate_tensor(
                element_type=st.sf_dtype,
                layout=sfa_smem_layout_staged,
                byte_alignment=128,
            )
            sSFB = smem.allocate_tensor(
                element_type=st.sf_dtype,
                layout=sfb_smem_layout_staged,
                byte_alignment=128,
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
        sfb_full_mcast_mask = None
        if cutlass.const_expr(st.is_a_mcast or st.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            if cutlass.const_expr(bs):
                sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_sfb_vmnk,
                    block_in_cluster_coord_sfb_vmnk,
                    mcast_mode=1,
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
        tBsSFB = tBgSFB = None
        tCtSFA_layout = tCtSFB_layout = None
        if cutlass.const_expr(bs):
            gSFB_nkl = cute.local_tile(
                mSFB_nkl,
                cute.slice_(st.mma_tiler_sfb, (0, None, None)),
                (None, None, None),
            )
            thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
            tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
            sfb_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
            )
            tBsSFB, tBgSFB = cpasync.tma_partition(
                tma_atom_sfb,
                block_in_cluster_coord_sfb_vmnk[1],
                sfb_cta_layout,
                cute.group_modes(sSFB, 0, 3),
                cute.group_modes(tCgSFB, 0, 3),
            )
            tBsSFB = cute.filter_zeros(tBsSFB)
            tBgSFB = cute.filter_zeros(tBgSFB)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(st.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, st.num_acc_stage)
        )
        if cutlass.const_expr(bs):
            tCtSFA_layout = bs_layout.make_tmem_layout_sfa(
                tiled_mma,
                st.mma_tiler,
                st.sf_vec,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB_layout = bs_layout.make_tmem_layout_sfb(
                tiled_mma,
                st.mma_tiler,
                st.sf_vec,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
        mma_m_in_cluster = block_in_cluster_coord_vmnk[1]
        cta_n_in_cluster = block_in_cluster_coord_vmnk[2]
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        cute.arch.barrier()  # the group tables are complete

        raster_m = self.knobs.raster_along == "m"
        tn, cn = st.mma_tiler[1], self.cluster_shape_mn[1]
        # --- roles
        if warp_idx == tma_warp:
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
            if cutlass.const_expr(bs):
                grouped_blockscaled_producer_loop(
                    st,
                    k_tile_cnt,
                    tile_sched,
                    ab_producer,
                    tma_atom_a,
                    tma_atom_b,
                    tma_atom_sfa,
                    tma_atom_sfb,
                    mA_mkl,
                    mSFA_mkl,
                    tBgB,
                    tBgSFB,
                    sA,
                    sSFA,
                    tBsB,
                    tBsSFB,
                    thr_mma,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    mma_m_in_cluster,
                    a_full_mcast_mask,
                    b_full_mcast_mask,
                    sfb_full_mcast_mask,
                    s_off,
                    s_off_pad,
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
        if warp_idx == mma_warp:
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
            if cutlass.const_expr(bs):
                tCtSFA = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + st.num_acc_tmem_cols, dtype=st.sf_dtype),
                    tCtSFA_layout,
                )
                tCtSFB = cute.make_tensor(
                    cute.recast_ptr(
                        tmem_ptr + st.num_acc_tmem_cols + st.num_sfa_tmem_cols,
                        dtype=st.sf_dtype,
                    ),
                    tCtSFB_layout,
                )
                blockscaled_mma_loop(
                    self.lowering,
                    st,
                    k_tile_cnt,
                    is_leader_cta,
                    tile_sched,
                    tiled_mma,
                    ab_consumer,
                    acc_pipeline,
                    tCrA,
                    tCrB,
                    tCtAcc_base,
                    tmem_ptr,
                    tCtSFA,
                    tCtSFB,
                    sSFA,
                    sSFB,
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
        if warp_idx < mma_warp:
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
                self.epi_warpgroups,
                sC=sC,
                cta_v=mma_tile_coord_v,
                mAlpha=mAlpha,
                mSa=mSa,
                mSi=mSi,
                mPts=mPts,
                mAux=mAux,
                mOutSf=mOutSf,
                s_off=s_off,
                s_off_pad=s_off_pad,
            )
            tmem.relinquish_alloc_permit()
            # the allocator warp alone frees the columns: every epilogue warpgroup joins here after its last TMEM load (the
            # staged store's barriers join one group only)
            epi_join.arrive_and_wait()
            tmem.free(tmem_ptr)


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
    pdl.trigger_late()  # the CTA's loads are all issued: the dependents' prologues may start
    ab_producer.tail()


@cute.jit
def grouped_blockscaled_producer_loop(
    st,
    k_tile_cnt: Int32,
    tile_sched,
    ab_producer,
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    tma_atom_sfa: cute.CopyAtom,
    tma_atom_sfb: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    mSFA_mkl: cute.Tensor,
    tBgB: cute.Tensor,
    tBgSFB: cute.Tensor,
    sA: cute.Tensor,
    sSFA: cute.Tensor,
    tBsB: cute.Tensor,
    tBsSFB: cute.Tensor,
    thr_mma,
    a_cta_coord,
    a_cta_layout,
    mma_m_in_cluster,
    a_mcast_mask,
    b_mcast_mask,
    sfb_mcast_mask,
    s_off=None,
    s_off_pad=None,
):
    """The block-scaled producer with per-tile coordinates: A and SFA are re-partitioned from the group-offset view (the
    scale-factor row a multiple of the atom, so the swizzled SFA box is addressed exactly; the dense packing shifts it by
    ``off_pad[e] - off[e]``), B and SFB are the dense partitions indexed by the expert as the batch coordinate. k-tiles
    ascending, one ring slot holds A, B, SFA and SFB of a k-tile. A 64-wide N tile reads half of a 128-row SFB atom; the
    192-wide tile's addressing lives in the SFB coordinate tensor."""
    tn = st.cta_tile_shape_mnk[1]
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        real_a = cute.domain_offset((work_tile.row0, 0, 0), mA_mkl)
        gA = cute.local_tile(
            real_a, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )
        tCgA = thr_mma.partition_A(gA)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            a_cta_coord,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tAgA_slice = tAgA[(None, mma_m_in_cluster, None, 0)]
        row0_sf = work_tile.row0
        if cutlass.const_expr(s_off_pad is not None):
            row0_sf = work_tile.row0 + (s_off_pad[work_tile.e] - s_off[work_tile.e])
        real_sfa = cute.domain_offset((row0_sf, 0, 0), mSFA_mkl)
        gSFA = cute.local_tile(
            real_sfa, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )
        tCgSFA = thr_mma.partition_A(gSFA)
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            a_cta_coord,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tAgSFA_slice = tAgSFA[(None, mma_m_in_cluster, None, 0)]
        n_tile, e = work_tile.n_tile, work_tile.e
        tBgB_slice = tBgB[(None, n_tile, None, e)]
        sfb_n = n_tile
        if cutlass.const_expr(tn < 128):
            sfb_n = n_tile // (SF_ATOM_ROWS // tn)  # 64-wide: half an atom
        tBgSFB_slice = tBgSFB[(None, sfb_n, None, e)]
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
            cute.copy(
                tma_atom_sfa,
                tAgSFA_slice[(None, handle.count)],
                tAsSFA[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=a_mcast_mask,
            )
            cute.copy(
                tma_atom_sfb,
                tBgSFB_slice[(None, handle.count)],
                tBsSFB[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=sfb_mcast_mask,
            )
            peek_empty = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_empty = ab_producer.try_acquire()
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    pdl.trigger_late()  # the CTA's loads are all issued: the dependents' prologues may start
    ab_producer.tail()
