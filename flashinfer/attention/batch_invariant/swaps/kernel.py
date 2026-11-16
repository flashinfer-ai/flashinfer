# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Swaps-AB attention kernel body: the class's term for work units of at most 32 live rows with the
keys on the MMA M dimension. Scaffolding (smem, pipelines, TMEM) and the composition of the role blocks: TWO softmax warpgroups on alternate
tiles (ping-pong; thread = key; warpgroup k takes the tiles of parity k and the running row maximum crosses between them through a one-entry
hand-off per direction), the correction warpgroup (thread = dim lane, owning the row chains for ``t < n_pad``), the MMA warp, the load warp
and two empty warps; ``grid`` CTAs walk the work units persistently (``plan.py``: the SAME descriptors as the row body). A split unit whose
group folds in-kernel counts its arrival and the last arriver runs the group's chunk-order fold (``blocks.correction.fold_group``).

Every floating-point op outside the two MMAs and ``ex2`` goes through ``CuteF32``; the key reductions are ``canon`` plans; the arithmetic
is ``spec.py``'s.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ....batch_invariant import canon
from ..blocks.correction import fold_group
from ..blocks.operands import ACC_DTYPE
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
from .correction import correction_loop
from .mma import mma_loop
from .operands import (
    BAR_CORRECTION,
    BAR_TMEM_ALLOC,
    N_SOFTMAX_WGS,
    STATS_SLOTS_PP,
    TMEM_COLS_SWAPS,
    Roles,
    SwapsPlan,
    SwapsTma,
    shared_storage_struct,
)
from .producer import load_loop
from .softmax import unit_loop


def _thread_group(n):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, n)


class SwapsKernel:
    """``SwapsKernel(D, g, page, n_pad)`` compiled once per static tuple; shapes, strides, the page table, the work units and the scale are
    runtime arguments. Every unit must have at most ``n_pad`` live rows (the API selects ``n_pad`` per launch)."""

    def __init__(
        self, D: int, g: int, page: int, n_pad: int, ragged: bool = False, cap: int = 0
    ):
        self.D, self.g, self.page, self.n_pad, self.ragged = D, g, page, n_pad, ragged
        self.cap = cap  # the shared-memory capacity the ring is sized for (0: the current device's)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mWs: cute.Tensor,
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
        st = SwapsPlan(self.D, self.g, self.page, self.n_pad, self.ragged, self.cap)
        self.st = st
        tma = SwapsTma(st, mQ, mK, mV)

        self.shared_storage = shared_storage_struct(
            st.q_stages,
            st.n_slots,
            st.s_stages,
            st.p_stages,
            st.q_stage_elems,
            st.p_stage_elems,
            st.kv_slot_elems,
        )
        self.kernel(
            tma.atom_q,
            tma.tensor_q,
            tma.atom_k,
            tma.tensor_k,
            tma.atom_v,
            tma.tensor_v,
            mO,
            mWs,
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
            st.pv_tiled_mma,
            st.q_smem_layout,
            st.q_view_layout,
            st.k_smem_layout,
            st.v_smem_layout,
            st.p_smem_layout,
            st.ring_view_layout,
        ).launch(
            grid=[grid, 1, 1],
            block=[Roles.threads, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        atom_q: cute.CopyAtom,
        tQ: cute.Tensor,
        atom_k: cute.CopyAtom,
        tK: cute.Tensor,
        atom_v: cute.CopyAtom,
        tV: cute.Tensor,
        mO: cute.Tensor,
        mWs: cute.Tensor,
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
        pv_tiled_mma: cute.TiledMma,
        q_smem_layout: cute.ComposedLayout,
        q_view_layout: cute.ComposedLayout,
        k_smem_layout: cute.ComposedLayout,
        v_smem_layout: cute.ComposedLayout,
        p_smem_layout: cute.ComposedLayout,
        ring_view_layout: cute.ComposedLayout,
    ):
        st = self.st
        R = Roles
        NP = st.n_pad
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        grid_size, _, _ = cute.arch.grid_dim()
        if warp_idx == R.load_warp:
            cpasync.prefetch_descriptor(atom_q)
            cpasync.prefetch_descriptor(atom_k)
            cpasync.prefetch_descriptor(atom_v)
        Hq = tK.shape[2] * st.g

        # ---- scaffolding
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        q_prod, q_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=st.q_stages,
            producer_group=_thread_group(1),
            consumer_group=_thread_group(1),
            tx_count=st.q_tx_bytes,
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
        s_prod, s_cons = (
            pipeline.PipelineUmmaAsync.create(  # S^T stages: stage k = softmax warpgroup k's (MMA -> warpgroup)
                barrier_storage=storage.s_mbar.data_ptr(),
                num_stages=st.s_stages,
                producer_group=_thread_group(1),
                consumer_group=_thread_group(128),
                defer_sync=True,
            ).make_participants()
        )
        p_prod, p_cons = (
            pipeline.PipelineAsyncUmma.create(  # P^T stages (smem): stage k = warpgroup k's (warpgroup -> MMA)
                barrier_storage=storage.p_mbar.data_ptr(),
                num_stages=st.p_stages,
                producer_group=_thread_group(128),
                consumer_group=_thread_group(1),
                defer_sync=True,
            ).make_participants()
        )
        stats_prod, stats_cons = (
            pipeline.PipelineAsync.create(  # 2 entries per tile in tile order; warpgroup k writes tiles of parity k
                barrier_storage=storage.stats_mbar.data_ptr(),
                num_stages=STATS_SLOTS_PP,
                producer_group=_thread_group(128),
                consumer_group=_thread_group(128),
                defer_sync=True,
            ).make_participants()
        )
        hm_prod, hm_cons = (
            pipeline.PipelineAsync.create(  # the running maximum's hand-off: entry k written by warpgroup k, read by 1 - k
                barrier_storage=storage.hm_mbar.data_ptr(),
                num_stages=N_SOFTMAX_WGS,
                producer_group=_thread_group(128),
                consumer_group=_thread_group(128),
                defer_sync=True,
            ).make_participants()
        )
        o_prod, o_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.o_mbar.data_ptr(),
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

        # ---- smem views
        sQ = storage.sQ.get_tensor(q_smem_layout.outer, swizzle=q_smem_layout.inner)
        sQ_view = storage.sQ.get_tensor(
            q_view_layout.outer, swizzle=q_view_layout.inner
        )
        sK = storage.sKV.get_tensor(k_smem_layout.outer, swizzle=k_smem_layout.inner)
        sV = storage.sKV.get_tensor(v_smem_layout.outer, swizzle=v_smem_layout.inner)
        sKV_view = storage.sKV.get_tensor(
            ring_view_layout.outer, swizzle=ring_view_layout.inner
        )
        sKV_flat = storage.sKV.get_tensor(
            cute.make_layout((st.n_slots * st.kv_slot_elems,))
        )
        sP = storage.sP.get_tensor(p_smem_layout.outer, swizzle=p_smem_layout.inner)
        sP_e = storage.sP.get_tensor(
            cute.make_layout((st.p_stages * st.p_stage_elems,))
        )
        s_alpha = storage.s_alpha.get_tensor(cute.make_layout((STATS_SLOTS_PP, 128)))
        s_m = storage.s_m.get_tensor(cute.make_layout((STATS_SLOTS_PP, 128)))
        s_r = storage.s_r.get_tensor(cute.make_layout((STATS_SLOTS_PP, 4, 128)))
        s_x_all = storage.s_x.get_tensor(cute.make_layout((N_SOFTMAX_WGS, 2, 4, 32)))
        s_hm = storage.s_hm.get_tensor(cute.make_layout((N_SOFTMAX_WGS, 32)))
        s_ab = storage.s_ab.get_tensor(cute.make_layout((2, 128)))
        s_inv = storage.s_inv.get_tensor(cute.make_layout((128,)))
        qk_thr_mma = qk_tiled_mma.get_slice(0)
        pv_thr_mma = pv_tiled_mma.get_slice(0)
        tSrK = qk_thr_mma.make_fragment_A(sK)  # (MMA, MMA_M, MMA_K, SLOT): one K slot
        tSrQ = qk_thr_mma.make_fragment_B(
            sQ
        )  # (MMA, MMA_N, MMA_K, STAGE): the unit's Q rows
        tOrV = pv_thr_mma.make_fragment_A(
            sV
        )  # (MMA, MMA_M, MMA_K, SLOT): one V slot as (dims x keys)
        tOrP = pv_thr_mma.make_fragment_B(sP)  # (MMA, MMA_N, MMA_K, STAGE): P^T
        pipeline_init_wait()
        # programmatic dependent launch: every working role issues griddepcontrol.launch_dependents AFTER its unit walk, so
        # the fixup kernel's blocks are scheduled only when the grid's last CTAs drain (its griddepcontrol.wait still orders every
        # workspace read behind the whole grid). Issued at the kernel start, the blocks sat resident on the SMs of early-retired CTAs
        # while the persistent walk still needed them: slower at every multi-wave fold.

        # ---- TMEM views (built after the allocation by every TMEM user)
        s_shape = qk_thr_mma.partition_shape_C((st.T, NP))
        s_layout = qk_thr_mma.make_fragment_C(s_shape).layout
        o_shape = pv_thr_mma.partition_shape_C((st.M_pv, NP))
        o_layout = pv_thr_mma.make_fragment_C(o_shape).layout

        # ---- roles: every role walks the same units in the same order (u = bidx, bidx + grid, ...)
        if warp_idx == R.load_warp:
            cute.arch.setmaxregister_decrease(R.regs_other)
            for u in range(bidx, num_units, grid_size):
                q_prod, kv_prod = load_loop(
                    st,
                    atom_q,
                    tQ,
                    atom_k,
                    tK,
                    atom_v,
                    tV,
                    sQ_view,
                    sKV_view,
                    q_prod,
                    kv_prod,
                    mKvIndptr,
                    mKvIndices,
                    mDesc[(u, F_REQ)],
                    mDesc[(u, F_QBASE)] + mDesc[(u, F_QTOK0)],
                    mDesc[(u, F_KVHEAD)],
                    mDesc[(u, F_TILE_LO)],
                    mDesc[(u, F_TILE_HI)],
                )
            cute.arch.griddepcontrol_launch_dependents()
        if warp_idx == R.mma_warp:
            cute.arch.setmaxregister_decrease(R.regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tStS = cute.make_tensor(tmem_ptr + st.tmem_s, s_layout)
            tOtO_slots = tuple(
                cute.make_tensor(tmem_ptr + st.tmem_o + sl * NP, o_layout)
                for sl in range(st.slots_per_tile)
            )
            qk_low = canon.BlockedFold(
                "dstep", st.d_steps
            )  # one block of D/16 steps, reset on the first
            pv_low = canon.BlockedFold(
                "kstep", st.k_steps
            )  # blocks = the tiles of a chunk, reset on the chunk's first step
            for u in range(bidx, num_units, grid_size):
                q_cons, kv_cons, s_prod, p_cons, o_prod, qk_tiled_mma, pv_tiled_mma = (
                    mma_loop(
                        st,
                        qk_tiled_mma,
                        pv_tiled_mma,
                        qk_low,
                        pv_low,
                        tStS,
                        tOtO_slots,
                        tSrQ,
                        tSrK,
                        tOrV,
                        tOrP,
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
                )
            cute.arch.griddepcontrol_launch_dependents()
        if (
            warp_idx < R.correction_warps[0]
        ):  # the softmax warpgroups: one code copy, warpgroup k = the tiles of parity k (dynamic)
            k = cute.arch.make_warp_uniform(warp_idx // 4)
            cute.arch.setmaxregister_increase(R.regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tStS = cute.make_tensor(tmem_ptr + st.tmem_s, s_layout)
            s_x = cute.make_tensor(
                s_x_all.iterator + k * (2 * 4 * 32), cute.make_layout((2, 4, 32))
            )
            # this warpgroup's stages: S / P stage k, stats entries 2k, 2k + 1 (mod 8), the hand-off entry it writes (k) and reads (1 - k)
            s_cons.advance(k)
            p_prod.advance(k)
            stats_prod.advance(2 * k)
            hm_prod.advance(k)
            hm_cons.advance(1 - k)
            for u in range(bidx, num_units, grid_size):
                s_cons, p_prod, stats_prod, hm_prod, hm_cons = unit_loop(
                    st,
                    k,
                    qk_thr_mma,
                    tStS,
                    s_cons,
                    p_prod,
                    stats_prod,
                    hm_prod,
                    hm_cons,
                    s_alpha,
                    s_m,
                    s_r,
                    s_x,
                    s_hm,
                    sP_e,
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
            tmem.allocate(TMEM_COLS_SWAPS)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tOtO_slots = tuple(
                cute.make_tensor(tmem_ptr + st.tmem_o + sl * NP, o_layout)
                for sl in range(st.slots_per_tile)
            )
            tOtOm_slots = tuple(
                cute.make_tensor(tmem_ptr + st.tmem_om + sl * NP, o_layout)
                for sl in range(st.slots_per_tile)
            )
            s_fold = storage.s_fold
            for u in range(bidx, num_units, grid_size):
                o_cons, stats_cons = correction_loop(
                    st,
                    pv_thr_mma,
                    tOtO_slots,
                    tOtOm_slots,
                    o_cons,
                    stats_cons,
                    s_alpha,
                    s_m,
                    s_r,
                    s_ab,
                    s_inv,
                    mO,
                    mWs,
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
            cute.arch.setmaxregister_decrease(R.regs_other)
