# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The two MMA folds of the Swaps-AB body (the MMA warp), issued per instruction through ``canon.BlockedFold`` exactly as the row body's:

* ``S^T = K Q^T`` per tile: ``D / 16`` K-steps ascending over the slots' dims (``ACCUMULATE`` off on the first): A = the K slot
  (M = 128 keys), B = the unit's Q rows (N = n_pad); one ``cute.gemm`` is one ``tcgen05.mma`` K-step (opaque);
* ``O^T[slot] += V^T P^T`` per tile and dims slot: ``T / 16`` key-steps ascending, ``ACCUMULATE`` off on a chunk's first step; A = the V slot as
  ``(dims x keys)`` (MN-major), B = ``P^T`` in smem (N = n_pad rows, K = keys).

Issue order per unit (a placement): ``QK(0)`` then per tile n: ``QK(n + 1)``, ``PV(n)`` (QK one tile ahead). The S^T and P^T stage of tile n is
its parity ``n % 2`` = the softmax warpgroup that works on it (ping-pong, operands.N_SOFTMAX_WGS): the two-stage S / P pipelines advance one
stage per tile; after a unit with an odd tile count the phantom tile's S^T stage is committed and its P^T stage consumed (empty), so
every barrier phase is matched by an arrival and every unit's tile 0 is warpgroup 0's stage. Before the PV of
a tile that reaches past the request's live length the V rows beyond it are zeroed in smem (``blocks.mma.zero_v_tail``: the same slot layout).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.nvgpu import tcgen05

from ..blocks.mma import chunk_blk, zero_v_tail
from ..blocks.operands import IN_DTYPE


# P^T in smem, QK(n + 2) needs only warpgroup n % 2's copy-out of S^T(n) (early in its tile),
# not P(n + 1) of the OTHER warpgroup (QK one ahead chained the two warpgroups); a placement


@cute.jit
def qk_tile(
    st,
    qk_mma: cute.TiledMma,
    low,
    tStS: cute.Tensor,
    tSrQ: cute.Tensor,
    tSrK: cute.Tensor,
    q_stage: Int32,
    kv_cons,
    s_prod,
):
    """``S^T = K Q^T`` of one tile into the next S^T stage: ``slots_per_tile`` K slots x ``d_steps_slot`` K-steps, one ascending fold."""
    sh = (
        s_prod.acquire_and_advance()
    )  # the softmax has copied this stage's previous S^T out
    tS = cute.make_tensor(tStS.iterator + sh.index * st.tmem_s_stride, tStS.layout)
    for sl in cutlass.range_constexpr(st.slots_per_tile):
        kh = kv_cons.wait_and_advance()
        for j in cutlass.range_constexpr(st.d_steps_slot):
            jg = sl * st.d_steps_slot + j
            qk_mma.set(tcgen05.Field.ACCUMULATE, low.accumulate(0, jg))
            cute.gemm(
                qk_mma,
                tS,
                tSrK[(None, None, j, kh.index)],
                tSrQ[(None, None, jg, q_stage)],
                tS,
            )
        kh.release()  # the slot's MMAs are issued: its release is their completion
    sh.commit()
    return kv_cons, s_prod, qk_mma


@cute.jit
def pv_tile(
    st,
    pv_mma: cute.TiledMma,
    low,
    tOtO_slots,
    tOrV: cute.Tensor,
    tOrP: cute.Tensor,
    sKV_flat: cute.Tensor,
    zeros8: cute.Tensor,
    t: Int32,
    blk: Int32,
    kv_len: Int32,
    kv_cons,
    p_cons,
    o_prod,
):
    ph = p_cons.wait_and_advance()  # P^T(t) stored by the softmax warpgroup
    oh = (
        o_prod.acquire_and_advance()
    )  # O^T rescaled (or merged away) by the correction warpgroup
    for sl in cutlass.range_constexpr(st.slots_per_tile):
        vh = kv_cons.wait_and_advance()
        zero_v_tail(st, sKV_flat, zeros8, t * st.T, vh.index, kv_len)
        for j in cutlass.range_constexpr(st.k_steps):
            pv_mma.set(tcgen05.Field.ACCUMULATE, low.accumulate(blk, j))
            cute.gemm(
                pv_mma,
                tOtO_slots[sl],
                tOrV[(None, None, j, vh.index)],
                tOrP[(None, None, j, ph.index)],
                tOtO_slots[sl],
            )
        vh.release()
    oh.commit()
    ph.release()
    return kv_cons, p_cons, o_prod, pv_mma


@cute.jit
def mma_loop(
    st,
    qk_mma: cute.TiledMma,
    pv_mma: cute.TiledMma,
    qk_low,
    pv_low,
    tStS: cute.Tensor,
    tOtO_slots,
    tSrQ: cute.Tensor,
    tSrK: cute.Tensor,
    tOrV: cute.Tensor,
    tOrP: cute.Tensor,
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
    TPC = st.tpc
    zeros8 = cute.make_rmem_tensor(cute.make_layout((8,)), IN_DTYPE)
    zeros8.fill(0.0)
    qh = q_cons.wait_and_advance()
    qs = qh.index
    N = tile_hi - tile_lo
    kv_cons, s_prod, qk_mma = qk_tile(
        st, qk_mma, qk_low, tStS, tSrQ, tSrK, qs, kv_cons, s_prod
    )  # QK(0)
    if N == 1:
        qh.release()  # the Q stage is free once the unit's last QK completes (a tcgen05.commit)
    if N > 1:
        kv_cons, s_prod, qk_mma = qk_tile(
            st, qk_mma, qk_low, tStS, tSrQ, tSrK, qs, kv_cons, s_prod
        )  # QK(1)
        if N == 2:
            qh.release()
    for n in range(N):
        t = tile_lo + n
        if n + 2 < N:
            kv_cons, s_prod, qk_mma = qk_tile(
                st, qk_mma, qk_low, tStS, tSrQ, tSrK, qs, kv_cons, s_prod
            )
            if n + 3 == N:
                qh.release()
        kv_cons, p_cons, o_prod, pv_mma = pv_tile(
            st,
            pv_mma,
            pv_low,
            tOtO_slots,
            tOrV,
            tOrP,
            sKV_flat,
            zeros8,
            t,
            chunk_blk(t, TPC, tile_lo),
            kv_len,
            kv_cons,
            p_cons,
            o_prod,
        )
    if N % 2 == 1:
        # an odd unit: warpgroup 1's phantom tile. Its S^T stage is committed (no MMA: the commit arrives once the issued MMAs complete) and its
        # P^T stage waited for and released (warpgroup 1 commits it empty), so every barrier's phase stays matched with real arrivals -- a
        # parity wait cannot skip a phase -- and the next unit's tile 0 is stage 0 again
        sh = s_prod.acquire_and_advance()
        sh.commit()
        ph = p_cons.wait_and_advance()
        ph.release()
    return q_cons, kv_cons, s_prod, p_cons, o_prod, qk_mma, pv_mma
