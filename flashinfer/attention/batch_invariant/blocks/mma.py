# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The two MMA folds (the MMA warp), issued per instruction through ``canon.BlockedFold``:

* ``S = Q K^T`` per tile: ``D/16`` K-steps ascending, ``ACCUMULATE`` off on the first (the GEMM's
  ``kstep`` lowering with one block); one ``cute.gemm`` is one ``tcgen05.mma`` K-step (opaque);
* ``O_c += P V`` per tile: ``T/16`` key-steps ascending; the chunk's first instruction has ``ACCUMULATE``
  off (the per-chunk fold starts at ``+0``; the correction warpgroup has already applied the
  ``mul(O, alpha)`` of the spec's step to the TMEM accumulator for every later tile).

Issue order per unit (a placement): with one S stage, QK(0) then per tile n: QK(n + 1), PV(n) -- QK one tile ahead so
the softmax of tile n overlaps the tensor core; with two S stages (``st.s_stages == 2``, P written over its own S stage),
QK(0), QK(1) then per tile n: PV(n), QK(n + 2) into the stage PV(n) has just consumed -- the tensor core executes a CTA's
MMAs in order, so QK(n + 2) overwrites P(n) only after PV(n) has read it, and the softmax always finds its next S ready.
Either way tile n's PV consumes tile n's P after the correction applied tile n's alpha, and the accumulator is reset at
every chunk start.

Before the PV of a tile that reaches past the request's live length, the V rows beyond it are zeroed
in smem (through the SW128 swizzle: 16-byte granule ``^= row % 8``), so the fold's identity step holds
whatever the pool holds there.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.nvgpu import tcgen05

from .operands import HALF, IN_DTYPE


@cute.jit
def qk_tile(
    st,
    qk_mma: cute.TiledMma,
    low,
    tStS_slots,
    tSrQ: cute.Tensor,
    tSrK: cute.Tensor,
    q_stage: Int32,
    kv_cons,
    s_prod,
):
    """``S = Q K^T`` of one tile: one ``D/16``-step fold per ring slot of K (one slot = the whole tile at D <= 128; two 64-key slots at
    D = 256, each written to its own S columns by a 128 x 64 instruction -- the same per-element fold, ``ACCUMULATE`` off on its
    first step); the slot is released as soon as its MMAs are issued (the release is the MMAs' completion)."""
    sh = None
    for sl in cutlass.range_constexpr(st.slots_per_tile):
        kh = kv_cons.wait_and_advance()
        if cutlass.const_expr(sl == 0):
            sh = s_prod.acquire_and_advance()  # the softmax warpgroups have copied the stage's previous S out (and its P was consumed)
        tS = cute.make_tensor(
            tStS_slots[sl].iterator + sh.index * st.tmem_s_stride, tStS_slots[sl].layout
        )  # this tile's S stage
        for j in cutlass.range_constexpr(low.inner):
            qk_mma.set(tcgen05.Field.ACCUMULATE, low.accumulate(0, j))
            cute.gemm(
                qk_mma,
                tS,
                tSrQ[(None, None, j, q_stage)],
                tSrK[(None, None, j, kh.index)],
                tS,
            )
        kh.release()
    sh.commit()
    return kv_cons, s_prod, qk_mma


@cute.jit
def zero_v_tail(
    st,
    sKV_flat: cute.Tensor,
    zeros8: cute.Tensor,
    key0: Int32,
    slot: Int32,
    kv_len: Int32,
):
    """Rows ``[kv_len - key0, slot_keys)`` of the V slot ``slot`` (first key ``key0``) are past the live length: +0 them."""
    SK = st.slot_keys
    lane = cute.arch.lane_idx()
    first = kv_len - key0
    if first < 0:
        first = Int32(0)
    if first < SK:
        base = slot * st.kv_slot_elems
        n_gran = (SK - first) * 8
        for idx in range((n_gran + 31) // 32):
            e = lane + idx * 32
            if e < n_gran:
                r = first + e // 8
                gran = e % 8
                for half in cutlass.range_constexpr(st.halves):
                    phys = base + half * (SK * HALF) + r * HALF + ((gran ^ (r % 8)) * 8)
                    cute.autovec_copy(
                        zeros8,
                        cute.make_tensor(
                            sKV_flat.iterator + phys, cute.make_layout((8,))
                        ),
                    )
        cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_warp()


@cute.jit
def pv_tile(
    st,
    pv_mma: cute.TiledMma,
    low,
    tOtOc: cute.Tensor,
    tOrP: cute.Tensor,
    tmem_ptr: cute.Pointer,
    tOrV: cute.Tensor,
    sKV_flat: cute.Tensor,
    zeros8: cute.Tensor,
    t: Int32,
    blk: Int32,
    kv_len: Int32,
    kv_cons,
    p_cons,
    o_prod,
):
    ph = None
    oh = None
    vh = None
    tOrP_k = tOrP
    KS = st.k_steps_per_slot
    for sl in cutlass.range_constexpr(st.slots_per_tile):
        vh = kv_cons.wait_and_advance()
        zero_v_tail(
            st, sKV_flat, zeros8, t * st.T + sl * st.slot_keys, vh.index, kv_len
        )
        if cutlass.const_expr(sl == 0):
            ph = p_cons.wait_and_advance()  # P(t) stored by the softmax warpgroups
            if cutlass.const_expr(
                st.s_stages > 1
            ):  # the tile's P stage (P_k over S_k: stage k of the S pipeline)
                tOrP_k = cute.make_tensor(
                    cute.recast_ptr(
                        tmem_ptr + st.tmem_p + ph.index * st.tmem_p_stride,
                        dtype=IN_DTYPE,
                    ),
                    tOrP.layout,
                )
            oh = (
                o_prod.acquire_and_advance()
            )  # O_c rescaled (or merged away) by the correction warpgroup
        for jj in cutlass.range_constexpr(KS):
            j = (
                sl * KS + jj
            )  # the tile's key step: ascending across the slots (the same fold)
            pv_mma.set(tcgen05.Field.ACCUMULATE, low.accumulate(blk, j))
            cute.gemm(
                pv_mma,
                tOtOc,
                tOrP_k[(None, None, j)],
                tOrV[(None, None, jj, vh.index)],
                tOtOc,
            )
        if cutlass.const_expr(sl < st.slots_per_tile - 1):
            vh.release()  # the slot's MMAs are issued: its release is their completion
    oh.commit()
    ph.release()
    vh.release()
    return kv_cons, p_cons, o_prod, pv_mma


@cute.jit
def chunk_blk(t: Int32, TPC: cutlass.Constexpr, tile_lo: Int32):
    """Position of tile ``t`` among its chunk's processed tiles (0 = the chunk's first PV: accumulator reset)."""
    start = t - t % TPC
    if start < tile_lo:
        start = tile_lo
    return t - start


@cute.jit
def mma_loop(
    st,
    qk_mma: cute.TiledMma,
    pv_mma: cute.TiledMma,
    qk_low,
    pv_low,
    tStS,
    tOtOc: cute.Tensor,
    tOrP: cute.Tensor,
    tmem_ptr: cute.Pointer,
    tSrQ: cute.Tensor,
    tSrK: cute.Tensor,
    tOrV: cute.Tensor,
    sKV_flat: cute.Tensor,
    q_cons,
    kv_cons,
    s_prod,
    p_cons,
    o_prod,
    tile_lo: Int32,
    tile_hi: Int32,
    kv_len: Int32,
):
    """``tStS`` is the tuple of the S accumulator's per-slot views of stage 0 (one view = the whole S at D <= 128)."""
    TPC = st.tpc
    zeros8 = cute.make_rmem_tensor(cute.make_layout((8,)), IN_DTYPE)
    zeros8.fill(0.0)
    qh = q_cons.wait_and_advance()
    qs = qh.index
    N = tile_hi - tile_lo
    # the Q stage is released right after the unit's LAST QK is issued (the release is a tcgen05.commit: the stage is free once those MMAs
    # complete), so the load warp fetches the next unit's Q while this unit's last PVs and epilogue run -- with one Q stage (D = 256) the
    # fetch would otherwise sit between two units in full (a placement: the order of the issue and the release, no value involved)
    kv_cons, s_prod, qk_mma = qk_tile(
        st, qk_mma, qk_low, tStS, tSrQ, tSrK, qs, kv_cons, s_prod
    )  # QK(0)
    if N == 1:
        qh.release()
    if cutlass.const_expr(st.qk_ahead == 2):
        if N > 1:
            kv_cons, s_prod, qk_mma = qk_tile(
                st, qk_mma, qk_low, tStS, tSrQ, tSrK, qs, kv_cons, s_prod
            )  # QK(1)
            if N == 2:
                qh.release()
        for n in range(N):
            t = tile_lo + n
            kv_cons, p_cons, o_prod, pv_mma = pv_tile(
                st,
                pv_mma,
                pv_low,
                tOtOc,
                tOrP,
                tmem_ptr,
                tOrV,
                sKV_flat,
                zeros8,
                t,
                chunk_blk(t, TPC, tile_lo),
                kv_len,
                kv_cons,
                p_cons,
                o_prod,
            )
            if n + 2 < N:
                kv_cons, s_prod, qk_mma = qk_tile(
                    st, qk_mma, qk_low, tStS, tSrQ, tSrK, qs, kv_cons, s_prod
                )
                if n + 3 == N:
                    qh.release()
    else:
        for n in range(N):
            t = tile_lo + n
            if n + 1 < N:
                kv_cons, s_prod, qk_mma = qk_tile(
                    st, qk_mma, qk_low, tStS, tSrQ, tSrK, qs, kv_cons, s_prod
                )
                if n + 2 == N:
                    qh.release()
            kv_cons, p_cons, o_prod, pv_mma = pv_tile(
                st,
                pv_mma,
                pv_low,
                tOtOc,
                tOrP,
                tmem_ptr,
                tOrV,
                sKV_flat,
                zeros8,
                t,
                chunk_blk(t, TPC, tile_lo),
                kv_len,
                kv_cons,
                p_cons,
                o_prod,
            )
    return q_cons, kv_cons, s_prod, p_cons, o_prod, qk_mma, pv_mma


@cute.jit
def qk_q2(
    st,
    qk_mma: cute.TiledMma,
    low,
    tStS0,
    tSrQ: cute.Tensor,
    tSrK: cute.Tensor,
    q_tile: Int32,
    k_slot: Int32,
    s_prod,
):
    """The paired placement's ``S_k = Q_k K^T`` into S stage k (its own: the participant skips the other tile's stage)."""
    sh = s_prod.acquire_and_advance()
    s_prod.advance(1)
    tS = cute.make_tensor(tStS0.iterator + sh.index * st.tmem_s_stride, tStS0.layout)
    for j in cutlass.range_constexpr(low.inner):
        qk_mma.set(tcgen05.Field.ACCUMULATE, low.accumulate(0, j))
        cute.gemm(
            qk_mma, tS, tSrQ[(None, None, j, q_tile)], tSrK[(None, None, j, k_slot)], tS
        )
    sh.commit()
    return s_prod, qk_mma


@cute.jit
def pv_q2(
    st,
    pv_mma: cute.TiledMma,
    low,
    tOtOc: cute.Tensor,
    tOrP: cute.Tensor,
    tmem_ptr: cute.Pointer,
    tOrV: cute.Tensor,
    v_slot: Int32,
    blk: Int32,
    p_cons,
    o_prod,
):
    """The paired placement's ``O_c,k += P_k V`` (P_k over S stage k; O_c,k rescaled or merged away by the correction first)."""
    ph = p_cons.wait_and_advance()
    p_cons.advance(1)
    tOrP_k = cute.make_tensor(
        cute.recast_ptr(
            tmem_ptr + st.tmem_p + ph.index * st.tmem_p_stride, dtype=IN_DTYPE
        ),
        tOrP.layout,
    )
    oh = o_prod.acquire_and_advance()  # this Q tile's own O pipeline (one stage)
    for j in cutlass.range_constexpr(st.k_steps):
        pv_mma.set(tcgen05.Field.ACCUMULATE, low.accumulate(blk, j))
        cute.gemm(
            pv_mma, tOtOc, tOrP_k[(None, None, j)], tOrV[(None, None, j, v_slot)], tOtOc
        )
    oh.commit()
    ph.release()
    return p_cons, o_prod, pv_mma


@cute.jit
def mma_loop_q2(
    st,
    qk_mma: cute.TiledMma,
    pv_mma: cute.TiledMma,
    qk_low,
    pv_low,
    tStS0,
    tOtOc_a: cute.Tensor,
    tOtOc_b: cute.Tensor,
    tOrP: cute.Tensor,
    tmem_ptr: cute.Pointer,
    tSrQ: cute.Tensor,
    tSrK: cute.Tensor,
    tOrV: cute.Tensor,
    sKV_flat: cute.Tensor,
    q_cons,
    kv_cons,
    s_prod_a,
    s_prod_b,
    p_cons_a,
    p_cons_b,
    o_prod_a,
    o_prod_b,
    lo_a: Int32,
    hi_a: Int32,
    lo_b: Int32,
    hi_b: Int32,
    kv_len: Int32,
):
    """The paired placement's MMA warp: one K / V stream over the union of the two tile ranges, each K and V tile consumed by both Q
    tiles' folds. Issue order (a placement): ``QK_A(lo), QK_B(lo)``, then per tile t: ``QK_A(t+1), QK_B(t+1), PV_A(t), PV_B(t)`` -- QK one
    tile ahead (P has its own TMEM columns, so S(t+1) only waits for the softmax's copy-out of S(t)); a fold is issued only for a tile in
    its Q tile's range (a union tile outside both is loaded and released unused). Every fold is the one-tile body's: ``QK`` one
    ``D / 16``-step block, ``PV`` reset at its Q tile's chunk starts (``chunk_blk`` against that tile's own ``tile_lo``)."""
    TPC = st.tpc
    zeros8 = cute.make_rmem_tensor(cute.make_layout((8,)), IN_DTYPE)
    zeros8.fill(0.0)
    qh = q_cons.wait_and_advance()
    qa = 2 * qh.index
    qb = qa + 1
    has_b = hi_b > lo_b
    lo = lo_a
    hi = hi_a
    if has_b:
        if lo_b < lo:
            lo = lo_b
        if hi_b > hi:
            hi = hi_b
    if hi > lo:
        kh = kv_cons.wait_and_advance()
        if (lo >= lo_a) & (lo < hi_a):
            s_prod_a, qk_mma = qk_q2(
                st, qk_mma, qk_low, tStS0, tSrQ, tSrK, qa, kh.index, s_prod_a
            )
        if (lo >= lo_b) & (lo < hi_b):
            s_prod_b, qk_mma = qk_q2(
                st, qk_mma, qk_low, tStS0, tSrQ, tSrK, qb, kh.index, s_prod_b
            )
        kh.release()
        for t in range(lo, hi):
            if t + 1 < hi:
                kh2 = kv_cons.wait_and_advance()
                if (t + 1 >= lo_a) & (t + 1 < hi_a):
                    s_prod_a, qk_mma = qk_q2(
                        st, qk_mma, qk_low, tStS0, tSrQ, tSrK, qa, kh2.index, s_prod_a
                    )
                if (t + 1 >= lo_b) & (t + 1 < hi_b):
                    s_prod_b, qk_mma = qk_q2(
                        st, qk_mma, qk_low, tStS0, tSrQ, tSrK, qb, kh2.index, s_prod_b
                    )
                kh2.release()
            vh = kv_cons.wait_and_advance()
            zero_v_tail(st, sKV_flat, zeros8, t * st.T, vh.index, kv_len)
            if (t >= lo_a) & (t < hi_a):
                p_cons_a, o_prod_a, pv_mma = pv_q2(
                    st,
                    pv_mma,
                    pv_low,
                    tOtOc_a,
                    tOrP,
                    tmem_ptr,
                    tOrV,
                    vh.index,
                    chunk_blk(t, TPC, lo_a),
                    p_cons_a,
                    o_prod_a,
                )
            if (t >= lo_b) & (t < hi_b):
                p_cons_b, o_prod_b, pv_mma = pv_q2(
                    st,
                    pv_mma,
                    pv_low,
                    tOtOc_b,
                    tOrP,
                    tmem_ptr,
                    tOrV,
                    vh.index,
                    chunk_blk(t, TPC, lo_b),
                    p_cons_b,
                    o_prod_b,
                )
            vh.release()
    qh.release()
    return (
        q_cons,
        kv_cons,
        s_prod_a,
        s_prod_b,
        p_cons_a,
        p_cons_b,
        o_prod_a,
        o_prod_b,
        qk_mma,
        pv_mma,
    )
