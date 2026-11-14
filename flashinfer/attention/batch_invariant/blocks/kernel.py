# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The attention kernel body: scaffolding (smem, pipelines, TMEM) and the composition of the role
blocks. ``grid`` CTAs walk the work units persistently
(``plan.py``); warps: ``n_wg`` softmax warpgroups (all on every tile, warpgroup k holding key slice k), the
correction warpgroup (which also owns the rows' statistics chain), the MMA warp, the load warp and empty
warps completing the last warpgroup (``operands.Roles``). A split unit whose group folds in-kernel
(``F_GROUP >= 0``) counts its arrival on the group's counter after its partial rows are visible; the last
arriver runs the group's chunk-order fold (``fixup.fold_row``) and writes the output rows.

Every floating-point op outside the two MMAs and ``ex2`` goes through ``CuteF32``; the reduction
trees come from ``canon`` plans; the arithmetic functions are ``spec.py``'s.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ....batch_invariant import canon
from ..plan import (
    F_CHUNK_HI,
    F_CHUNK_LO,
    F_GROUP,
    F_KVHEAD,
    F_KVLEN,
    F_MODE,
    F_QBASE,
    F_QLEN,
    F_QTOK0,
    F_REQ,
    F_ROWS,
    F_SLOT,
    F_TILE_HI,
    F_TILE_LO,
)
from .correction import correction_loop, correction_loop_q2, fold_group
from .mma import mma_loop, mma_loop_q2
from .operands import (
    ACC_DTYPE,
    BAR_CORRECTION,
    BAR_TMEM_ALLOC,
    IN_DTYPE,
    Q_ROWS,
    REGS_OTHER,
    REGS_SOFTMAX,
    TMEM_COLS,
    StaticPlan,
    TmaSetup,
)
from .producer import load_loop, load_loop_q2
from .softmax import softmax_loop


def _thread_group(n):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, n)


@cute.jit
def tmem_views(
    st,
    qk_thr_mma: cute.TiledMma,
    pv_thr_mma: cute.TiledMma,
    p_tmem_layout: cute.ComposedLayout,
    tmem_ptr: cute.Pointer,
):
    """S, O_c, O_m accumulators and the PV A-operand view of P at the static TMEM column offsets (``O_m`` is ``None`` when it
    lives in the global scratch: D = 256)."""
    s_shape = qk_thr_mma.partition_shape_C((Q_ROWS, st.T))
    tStS = cute.make_tensor(
        tmem_ptr + st.tmem_s, qk_thr_mma.make_fragment_C(s_shape).layout
    )
    o_shape = pv_thr_mma.partition_shape_C((Q_ROWS, st.D))
    o_layout = pv_thr_mma.make_fragment_C(o_shape).layout
    tOtOc = cute.make_tensor(tmem_ptr + st.tmem_oc, o_layout)
    tOtOm = None
    if cutlass.const_expr(st.om_tmem):
        tOtOm = cute.make_tensor(tmem_ptr + st.tmem_om, o_layout)
    tP = cute.make_tensor(tStS.iterator, p_tmem_layout.outer)
    tOrP_ = pv_thr_mma.make_fragment_A(tP)[None, None, None, 0]
    tOrP = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + st.tmem_p, dtype=tOrP_.dtype), tOrP_.layout
    )
    return tStS, tOtOc, tOtOm, tOrP


class AttentionKernel:
    """``AttentionKernel(D, g, page, kv_slots, n_wg)`` compiled once per static tuple; shapes, strides, the
    page table, the work units and the scale are runtime arguments."""

    def __init__(
        self,
        D: int,
        g: int,
        page: int,
        kv_slots: int,
        n_wg: int = 1,
        q2: bool = False,
        ragged: bool = False,
    ):
        self.D, self.g, self.page, self.kv_slots, self.n_wg, self.q2, self.ragged = (
            D,
            g,
            page,
            kv_slots,
            n_wg,
            q2,
            ragged,
        )

    # ------------------------------------------------------------------ host (trace time)
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mWs: cute.Tensor,
        mScratch: cute.Tensor,
        mGroups: cute.Tensor,
        mCounters: cute.Tensor,
        mDesc: cute.Tensor,
        mKvIndptr: cute.Tensor,
        mKvIndices: cute.Tensor,
        c: Float32,
        nc: Float32,
        window_left: Int32,
        num_units: Int32,
        grid: Int32,
        stream: cuda.CUstream,
    ):
        """``grid`` CTAs walk the ``num_units`` work units persistently (unit ``u`` on CTA ``u % grid``): a launch
        parameter, never a quantity that reaches the bits (I10). ``mScratch`` is the per-CTA ``O_m`` scratch (``grid x 128 x D``
        fp32) when ``O_m`` is not in TMEM (D = 256), else any one-element tensor."""
        st = StaticPlan(
            self.D, self.g, self.page, self.kv_slots, self.n_wg, self.q2, self.ragged
        )
        self.st = st
        tma = TmaSetup(st, mQ, mK, mV)

        @cute.struct
        class SharedStorage:
            q_mbar: cute.struct.MemRange[Int64, 2 * st.q_stages]
            kv_mbar: cute.struct.MemRange[Int64, 2 * st.n_slots]
            s_mbar: cute.struct.MemRange[Int64, 2 * st.s_stages]
            p_mbar: cute.struct.MemRange[Int64, 2 * max(st.s_stages, st.n_q)]
            o_mbar: cute.struct.MemRange[Int64, 2 * st.n_q]
            stats_mbar: cute.struct.MemRange[Int64, 2 * st.stats_slots]
            tmem_dealloc_mbar: Int64
            tmem_holding_buf: Int32
            s_fold: Int32
            s_alpha: cute.struct.MemRange[Float32, st.stats_slots * Q_ROWS]
            s_m: cute.struct.MemRange[Float32, st.stats_slots * Q_ROWS]
            s_r: cute.struct.MemRange[
                Float32, st.stats_slots * 4 * st.n_slices * Q_ROWS
            ]
            s_p: cute.struct.Align[
                cute.struct.MemRange[Float32, self.n_wg * 3 * 32 * 16], 16
            ]
            sOm: cute.struct.Align[
                cute.struct.MemRange[
                    Float32, st.n_q * Q_ROWS * st.D if st.om_smem else 4
                ],
                16,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[IN_DTYPE, st.q_tiles * st.q_tile_elems], 1024
            ]
            sKV: cute.struct.Align[
                cute.struct.MemRange[IN_DTYPE, st.n_slots * st.kv_slot_elems], 1024
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma.atom_q,
            tma.tensor_q,
            tma.atom_q_rep,
            tma.tensor_q_rep,
            tma.atom_k,
            tma.tensor_k,
            tma.atom_v,
            tma.tensor_v,
            mO,
            mWs,
            mScratch,
            mGroups,
            mCounters,
            mDesc,
            mKvIndptr,
            mKvIndices,
            c,
            nc,
            window_left,
            num_units,
            st.qk_tiled_mma,
            st.qks_tiled_mma,
            st.pv_tiled_mma,
            st.q_smem_layout,
            st.q_view_layout,
            st.q_view_rep_layout,
            st.k_smem_layout,
            st.v_smem_layout,
            st.ring_view_layout,
            st.p_tmem_layout,
        ).launch(
            grid=[grid, 1, 1],
            block=[st.roles.threads, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ------------------------------------------------------------------ device
    @cute.kernel
    def kernel(
        self,
        atom_q: cute.CopyAtom,
        tQ: cute.Tensor,
        atom_q_rep: cute.CopyAtom,
        tQr: cute.Tensor,
        atom_k: cute.CopyAtom,
        tK: cute.Tensor,
        atom_v: cute.CopyAtom,
        tV: cute.Tensor,
        mO: cute.Tensor,
        mWs: cute.Tensor,
        mScratch: cute.Tensor,
        mGroups: cute.Tensor,
        mCounters: cute.Tensor,
        mDesc: cute.Tensor,
        mKvIndptr: cute.Tensor,
        mKvIndices: cute.Tensor,
        c: Float32,
        nc: Float32,
        window_left: Int32,
        num_units: Int32,
        qk_tiled_mma: cute.TiledMma,
        qks_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        q_smem_layout: cute.ComposedLayout,
        q_view_layout: cute.ComposedLayout,
        q_view_rep_layout: cute.ComposedLayout,
        k_smem_layout: cute.ComposedLayout,
        v_smem_layout: cute.ComposedLayout,
        ring_view_layout: cute.ComposedLayout,
        p_tmem_layout: cute.ComposedLayout,
    ):
        st = self.st
        R = st.roles
        n_soft = st.softmax_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        grid_size, _, _ = cute.arch.grid_dim()
        if warp_idx == R.load_warp:
            cpasync.prefetch_descriptor(atom_q)
            cpasync.prefetch_descriptor(atom_q_rep)
            cpasync.prefetch_descriptor(atom_k)
            cpasync.prefetch_descriptor(atom_v)
        Hq = tK.shape[2] * st.g  # heads of Q = kv heads x group

        # ---- scaffolding
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        q_prod, q_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=st.q_stages,
            producer_group=_thread_group(1),
            consumer_group=_thread_group(1),
            tx_count=st.q_bytes,
            defer_sync=True,
        ).make_participants()
        kv_prod, kv_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.kv_mbar.data_ptr(),
            num_stages=st.n_slots,
            producer_group=_thread_group(1),
            consumer_group=_thread_group(1),
            tx_count=st.kv_slot_bytes,
            defer_sync=True,
        ).make_participants()
        n_stage = (
            128 if st.q2 else n_soft
        )  # threads on a tile's stage: one warpgroup under ping-pong / pairing, all otherwise
        s_pipe = pipeline.PipelineUmmaAsync.create(  # S stages (one, or two with P over S: operands.tmem_plan_for)
            barrier_storage=storage.s_mbar.data_ptr(),
            num_stages=st.s_stages,
            producer_group=_thread_group(1),
            consumer_group=_thread_group(n_stage),
            defer_sync=True,
        )
        s_prod, s_cons = s_pipe.make_participants()
        p_pipe = pipeline.PipelineAsyncUmma.create(  # P stages: one per S stage (tile n uses stage n % s_stages of both)
            barrier_storage=storage.p_mbar.data_ptr(),
            num_stages=2 if st.q2 else st.s_stages,
            producer_group=_thread_group(n_stage),
            consumer_group=_thread_group(1),
            defer_sync=True,
        )
        p_prod, p_cons = p_pipe.make_participants()
        stats_pipe = pipeline.PipelineAsync.create(
            barrier_storage=storage.stats_mbar.data_ptr(),
            num_stages=st.stats_slots,
            producer_group=_thread_group(n_stage),
            consumer_group=_thread_group(128),
            defer_sync=True,
        )
        stats_prod, stats_cons = stats_pipe.make_participants()
        o_prod, o_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.o_mbar.data_ptr(),
            num_stages=1,
            producer_group=_thread_group(1),
            consumer_group=_thread_group(128),
            defer_sync=True,
        ).make_participants()
        o_prod_b = o_prod
        o_cons_b = o_cons
        if cutlass.const_expr(st.q2):  # pairing: Q tile B's own one-stage O pipeline
            o_prod_b, o_cons_b = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.o_mbar.data_ptr() + 2,
                num_stages=1,
                producer_group=_thread_group(1),
                consumer_group=_thread_group(128),
                defer_sync=True,
            ).make_participants()
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=BAR_TMEM_ALLOC, num_threads=R.n_tmem_users
        )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=R.correction_warps[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(is_relaxed=True)

        # ---- smem views: the MMA's (library layouts) and the producer's (page slabs) of the same bytes
        sQ = storage.sQ.get_tensor(q_smem_layout.outer, swizzle=q_smem_layout.inner)
        sQ_view = storage.sQ.get_tensor(
            q_view_layout.outer, swizzle=q_view_layout.inner
        )
        sQ_rep_view = storage.sQ.get_tensor(
            q_view_rep_layout.outer, swizzle=q_view_rep_layout.inner
        )
        sK = storage.sKV.get_tensor(k_smem_layout.outer, swizzle=k_smem_layout.inner)
        sV = storage.sKV.get_tensor(v_smem_layout.outer, swizzle=v_smem_layout.inner)
        sKV_view = storage.sKV.get_tensor(
            ring_view_layout.outer, swizzle=ring_view_layout.inner
        )
        sKV_flat = storage.sKV.get_tensor(
            cute.make_layout((st.n_slots * st.kv_slot_elems,))
        )
        s_alpha = storage.s_alpha.get_tensor(
            cute.make_layout((st.stats_slots, Q_ROWS))
        )  # softmax -> correction: alpha (early)
        s_m = storage.s_m.get_tensor(
            cute.make_layout((st.stats_slots, Q_ROWS))
        )  # ... m' and the slices' row-sum partials (late)
        s_r = storage.s_r.get_tensor(
            cute.make_layout((st.stats_slots, 4 * st.n_slices, Q_ROWS))
        )
        s_p_all = storage.s_p.get_tensor(
            cute.make_layout((st.n_wg, 3, 32, 16))
        )  # per warpgroup: the P slices of its warps 1..3
        qk_thr_mma = qk_tiled_mma.get_slice(0)
        qks_thr_mma = qks_tiled_mma.get_slice(0)
        pv_thr_mma = pv_tiled_mma.get_slice(0)
        tSrQ = qks_thr_mma.make_fragment_A(sQ)  # (MMA, MMA_M, MMA_K, Q_STAGE)
        tSrK = qks_thr_mma.make_fragment_B(
            sK
        )  # (MMA, MMA_N, MMA_K, SLOT): one ring slot of K
        tOrV = pv_thr_mma.make_fragment_B(
            sV
        )  # (MMA, MMA_N, MMA_K, SLOT): one ring slot of V (K = slot_keys)
        pipeline_init_wait()
        # programmatic dependent launch: every working role issues griddepcontrol.launch_dependents AFTER its unit walk, so
        # the fixup kernel's blocks are scheduled only when the grid's last CTAs drain (its griddepcontrol.wait still orders every
        # workspace read behind the whole grid). Issued at the kernel start, the blocks sat resident on the SMs of early-retired CTAs
        # while the persistent walk still needed them: slower at every multi-wave fold.

        # ---- roles: every role walks the same units in the same order (u = bidx, bidx + grid, ...)
        if warp_idx == R.load_warp:
            cute.arch.setmaxregister_decrease(REGS_OTHER)
            if cutlass.const_expr(st.q2):
                for u in range(bidx, num_units, grid_size):
                    ua = 2 * u
                    ub = ua + 1
                    q_prod, kv_prod = load_loop_q2(
                        st,
                        atom_q,
                        tQ,
                        atom_q_rep,
                        tQr,
                        atom_k,
                        tK,
                        atom_v,
                        tV,
                        sQ_view,
                        sQ_rep_view,
                        sKV_view,
                        q_prod,
                        kv_prod,
                        mKvIndptr,
                        mKvIndices,
                        mDesc[(ua, F_REQ)],
                        mDesc[(ua, F_KVHEAD)],
                        mDesc[(ua, F_QBASE)] + mDesc[(ua, F_QTOK0)],
                        mDesc[(ua, F_ROWS)],
                        mDesc[(ua, F_TILE_LO)],
                        mDesc[(ua, F_TILE_HI)],
                        mDesc[(ub, F_QBASE)] + mDesc[(ub, F_QTOK0)],
                        mDesc[(ub, F_ROWS)],
                        mDesc[(ub, F_TILE_LO)],
                        mDesc[(ub, F_TILE_HI)],
                    )
            if cutlass.const_expr(not st.q2):
                for u in range(bidx, num_units, grid_size):
                    q_prod, kv_prod = load_loop(
                        st,
                        atom_q,
                        tQ,
                        atom_q_rep,
                        tQr,
                        atom_k,
                        tK,
                        atom_v,
                        tV,
                        sQ_view,
                        sQ_rep_view,
                        sKV_view,
                        q_prod,
                        kv_prod,
                        mKvIndptr,
                        mKvIndices,
                        mDesc[(u, F_REQ)],
                        mDesc[(u, F_QBASE)] + mDesc[(u, F_QTOK0)],
                        mDesc[(u, F_KVHEAD)],
                        mDesc[(u, F_ROWS)],
                        mDesc[(u, F_TILE_LO)],
                        mDesc[(u, F_TILE_HI)],
                    )
            cute.arch.griddepcontrol_launch_dependents()
        if warp_idx == R.mma_warp:
            cute.arch.setmaxregister_decrease(REGS_OTHER)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tStS, tOtOc, tOtOm, tOrP = tmem_views(
                st, qk_thr_mma, pv_thr_mma, p_tmem_layout, tmem_ptr
            )
            # the S accumulator per ring slot of K: the whole S (one slot per tile) or its two 64-column halves (half-tile slots); stage 0
            # (a tile's stage adds stage * tmem_s_stride columns: mma.py)
            ss_shape = qks_thr_mma.partition_shape_C((Q_ROWS, st.slot_keys))
            ss_layout = qks_thr_mma.make_fragment_C(ss_shape).layout
            tStS_slots = tuple(
                cute.make_tensor(tmem_ptr + st.tmem_s + sl * st.slot_keys, ss_layout)
                for sl in range(st.slots_per_tile)
            )
            qk_low = canon.BlockedFold(
                "dstep", st.d_steps
            )  # one block of D/16 steps, reset on the first
            pv_low = canon.BlockedFold(
                "kstep", st.k_steps
            )  # blocks = the tiles of a chunk, reset on the chunk's first step
            if cutlass.const_expr(st.q2):
                s_prod_b, _ = s_pipe.make_participants()
                _, p_cons_b = p_pipe.make_participants()
                s_prod_b.advance(1)  # Q tile B's S / P stage is stage 1
                p_cons_b.advance(1)
                tOtOc_b = cute.make_tensor(
                    tOtOc.iterator + st.tmem_q2_stride, tOtOc.layout
                )
                for u in range(bidx, num_units, grid_size):
                    ua = 2 * u
                    ub = ua + 1
                    (
                        q_cons,
                        kv_cons,
                        s_prod,
                        s_prod_b,
                        p_cons,
                        p_cons_b,
                        o_prod,
                        o_prod_b,
                        qks_tiled_mma,
                        pv_tiled_mma,
                    ) = mma_loop_q2(
                        st,
                        qks_tiled_mma,
                        pv_tiled_mma,
                        qk_low,
                        pv_low,
                        tStS_slots[0],
                        tOtOc,
                        tOtOc_b,
                        tOrP,
                        tmem_ptr,
                        tSrQ,
                        tSrK,
                        tOrV,
                        sKV_flat,
                        q_cons,
                        kv_cons,
                        s_prod,
                        s_prod_b,
                        p_cons,
                        p_cons_b,
                        o_prod,
                        o_prod_b,
                        mDesc[(ua, F_TILE_LO)],
                        mDesc[(ua, F_TILE_HI)],
                        mDesc[(ub, F_TILE_LO)],
                        mDesc[(ub, F_TILE_HI)],
                        mDesc[(ua, F_KVLEN)],
                    )
            if cutlass.const_expr(not st.q2):
                for u in range(bidx, num_units, grid_size):
                    (
                        q_cons,
                        kv_cons,
                        s_prod,
                        p_cons,
                        o_prod,
                        qks_tiled_mma,
                        pv_tiled_mma,
                    ) = mma_loop(
                        st,
                        qks_tiled_mma,
                        pv_tiled_mma,
                        qk_low,
                        pv_low,
                        tStS_slots,
                        tOtOc,
                        tOrP,
                        tmem_ptr,
                        tSrQ,
                        tSrK,
                        tOrV,
                        sKV_flat,
                        q_cons,
                        kv_cons,
                        s_prod,
                        p_cons,
                        o_prod,
                        mDesc[(u, F_TILE_LO)],
                        mDesc[(u, F_TILE_HI)],
                        mDesc[(u, F_KVLEN)],
                    )
            cute.arch.griddepcontrol_launch_dependents()
        if (
            warp_idx < R.correction_warps[0]
        ):  # the softmax warpgroups: one code copy, warpgroup k = key slice k (dynamic)
            k = cute.arch.make_warp_uniform(warp_idx // 4)
            cute.arch.setmaxregister_increase(REGS_SOFTMAX)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tStS, tOtOc, tOtOm, tOrP = tmem_views(
                st, qk_thr_mma, pv_thr_mma, p_tmem_layout, tmem_ptr
            )
            if cutlass.const_expr(st.q2):
                # pairing: warpgroup k owns Q tile k -- S / P stage k and stats entries 2k, 2k + 1 of every four (it skips the other's)
                s_cons.advance(k)
                p_prod.advance(k)
                stats_prod.advance(2 * k)
                for u in range(bidx, num_units, grid_size):
                    uk = 2 * u + k
                    s_cons, p_prod, stats_prod = softmax_loop(
                        st,
                        k,
                        qk_thr_mma,
                        tStS,
                        s_cons,
                        p_prod,
                        stats_prod,
                        s_alpha,
                        s_m,
                        s_r,
                        s_p_all,
                        c,
                        nc,
                        window_left,
                        mDesc[(uk, F_ROWS)],
                        mDesc[(uk, F_QTOK0)],
                        mDesc[(uk, F_KVLEN)],
                        mDesc[(uk, F_QLEN)],
                        mDesc[(uk, F_TILE_LO)],
                        mDesc[(uk, F_TILE_HI)],
                    )
            if cutlass.const_expr(not st.q2):
                for u in range(bidx, num_units, grid_size):
                    s_cons, p_prod, stats_prod = softmax_loop(
                        st,
                        k,
                        qk_thr_mma,
                        tStS,
                        s_cons,
                        p_prod,
                        stats_prod,
                        s_alpha,
                        s_m,
                        s_r,
                        s_p_all,
                        c,
                        nc,
                        window_left,
                        mDesc[(u, F_ROWS)],
                        mDesc[(u, F_QTOK0)],
                        mDesc[(u, F_KVLEN)],
                        mDesc[(u, F_QLEN)],
                        mDesc[(u, F_TILE_LO)],
                        mDesc[(u, F_TILE_HI)],
                    )
            cute.arch.griddepcontrol_launch_dependents()
        if warp_idx >= R.correction_warps[0] and warp_idx < R.mma_warp:
            if cutlass.const_expr(R.regs_correction > R.regs_launch):
                cute.arch.setmaxregister_increase(R.regs_correction)
            else:
                cute.arch.setmaxregister_decrease(R.regs_correction)
            tmem.allocate(TMEM_COLS)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tStS, tOtOc, tOtOm, tOrP = tmem_views(
                st, qk_thr_mma, pv_thr_mma, p_tmem_layout, tmem_ptr
            )
            s_fold = storage.s_fold
            if cutlass.const_expr(st.q2):
                _, stats_cons_b = stats_pipe.make_participants()
                stats_cons_b.advance(2)  # Q tile B's stats entries: 2, 3 of every four
                for u in range(bidx, num_units, grid_size):
                    ua = 2 * u
                    ub = ua + 1
                    o_cons, o_cons_b, stats_cons, stats_cons_b = correction_loop_q2(
                        st,
                        pv_thr_mma,
                        tOtOc,
                        storage.sOm.get_tensor(
                            cute.make_layout((st.n_q * Q_ROWS * st.D,))
                        ),
                        o_cons,
                        o_cons_b,
                        stats_cons,
                        stats_cons_b,
                        s_alpha,
                        s_m,
                        s_r,
                        mO,
                        mWs,
                        c,
                        nc,
                        Hq,
                        mDesc[(ua, F_QBASE)],
                        mDesc[(ua, F_KVHEAD)],
                        mDesc[(ua, F_ROWS)],
                        mDesc[(ua, F_QTOK0)],
                        mDesc[(ua, F_CHUNK_LO)],
                        mDesc[(ua, F_CHUNK_HI)],
                        mDesc[(ua, F_TILE_LO)],
                        mDesc[(ua, F_TILE_HI)],
                        mDesc[(ua, F_MODE)],
                        mDesc[(ua, F_SLOT)],
                        mDesc[(ub, F_ROWS)],
                        mDesc[(ub, F_QTOK0)],
                        mDesc[(ub, F_CHUNK_LO)],
                        mDesc[(ub, F_CHUNK_HI)],
                        mDesc[(ub, F_TILE_LO)],
                        mDesc[(ub, F_TILE_HI)],
                        mDesc[(ub, F_MODE)],
                        mDesc[(ub, F_SLOT)],
                    )
                    fold_group(
                        st,
                        mWs,
                        mO,
                        mGroups,
                        mCounters,
                        s_fold,
                        mDesc[(ua, F_GROUP)],
                        c,
                        nc,
                        window_left,
                        Hq,
                    )
                    if mDesc[(ub, F_TILE_HI)] > mDesc[(ub, F_TILE_LO)]:
                        fold_group(
                            st,
                            mWs,
                            mO,
                            mGroups,
                            mCounters,
                            s_fold,
                            mDesc[(ub, F_GROUP)],
                            c,
                            nc,
                            window_left,
                            Hq,
                        )
            if cutlass.const_expr(not st.q2):
                for u in range(bidx, num_units, grid_size):
                    o_cons, stats_cons = correction_loop(
                        st,
                        pv_thr_mma,
                        tOtOc,
                        tOtOm,
                        o_cons,
                        stats_cons,
                        s_alpha,
                        s_m,
                        s_r,
                        mO,
                        mWs,
                        mScratch,
                        bidx,
                        c,
                        nc,
                        mDesc[(u, F_ROWS)],
                        mDesc[(u, F_QBASE)],
                        mDesc[(u, F_QTOK0)],
                        Hq,
                        mDesc[(u, F_KVHEAD)],
                        mDesc[(u, F_CHUNK_LO)],
                        mDesc[(u, F_CHUNK_HI)],
                        mDesc[(u, F_TILE_LO)],
                        mDesc[(u, F_TILE_HI)],
                        mDesc[(u, F_MODE)],
                        mDesc[(u, F_SLOT)],
                    )
                    fold_group(
                        st,
                        mWs,
                        mO,
                        mGroups,
                        mCounters,
                        s_fold,
                        mDesc[(u, F_GROUP)],
                        c,
                        nc,
                        window_left,
                        Hq,
                    )
            cute.arch.griddepcontrol_launch_dependents()
            tmem.relinquish_alloc_permit()
            pipeline.NamedBarrier(
                barrier_id=BAR_CORRECTION, num_threads=128
            ).arrive_and_wait()
            tmem.free(tmem_ptr)
        if warp_idx >= R.empty_warps[0]:
            cute.arch.setmaxregister_decrease(REGS_OTHER)
