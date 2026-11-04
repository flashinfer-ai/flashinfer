# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The correction warpgroup of the Swaps-AB body: thread ``t`` owns TMEM lane ``t``, i.e. dim ``t`` of every ``O^T`` slot (dims ``t`` and
``128 + t`` at D = 256; at D = 64 the 64 x n_pad accumulator's 16-lane layout, read through the library's partition), and -- for
``t < n_pad`` -- the statistics chain of row ``t`` (exactly the row body's thread-per-row chain: ``l = fma(l, alpha, r_t)`` with ``r_t`` the
pairwise add of the four warp partials in warp order, the chunk merges' ``(a1, a2)`` from ``spec.merge``'s scalar part, the normaliser).
The per-row coefficients every thread needs for its columns (``alpha`` from the stats ring; ``a1, a2`` and ``inv`` published by the owning
thread into smem, one named barrier) are applied column-wise: ``O^T[d, n] *= alpha[n]`` (``mul.rn``, skipped when no live row's maximum
moved -- ``mul(O, 1.0)`` is an identity), ``O_m^T[d, n] = fma(O_c^T[d, n], a2[n], mul(O_m^T[d, n], a1[n]))`` in TMEM at every D, and the
epilogue ``cvt_bf16(add(mul(O, inv[n]), +0))`` to the output rows (mode 0) or the raw fp32 ``O^T`` columns and ``(m, l)`` to the workspace
rows (mode 1, folded by ``fixup.py`` or the in-kernel last arriver).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import tcgen05

from ....batch_invariant.fp_interface.cute import CuteF32
from ..blocks.correction import merge_coeffs, row_sum
from ..blocks.operands import ACC_DTYPE, IN_DTYPE
from .operands import BAR_COEF

_NEG_INF = float("-inf")


def copy_atoms(st):
    """The TMEM load / store atoms of one ``O^T`` slot: 32 lanes x n_pad columns at M = 128; the 16-lane shape at M = 64 (D = 64)."""
    if st.M_pv == 128:
        return (
            cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(st.n_pad)), ACC_DTYPE
            ),
            cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(st.n_pad)), ACC_DTYPE
            ),
        )
    return (
        cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(st.n_pad // 8)), ACC_DTYPE
        ),
        cute.make_copy_atom(
            tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(st.n_pad // 8)), ACC_DTYPE
        ),
    )


@cute.jit
def load_coefs(src: cute.Tensor, idx: Int32, tLDc: cute.Tensor, out):
    """``out[i] = src[(idx, n_i)]`` for every register ``i`` of the thread's fragment (``n_i`` = its column = the row index)."""
    for i in cutlass.range_constexpr(cute.size(out)):
        out[i] = src[(idx, tLDc[i][1])]


@cute.jit
def correction_loop(
    st,
    pv_thr_mma: cute.TiledMma,
    tOtO_slots,
    tOtOm_slots,
    o_cons,
    stats_cons,
    s_alpha: cute.Tensor,
    s_m: cute.Tensor,
    s_r: cute.Tensor,
    s_ab: cute.Tensor,
    s_inv: cute.Tensor,
    mO: cute.Tensor,
    mWs: cute.Tensor,
    c: Float32,
    nc: Float32,
    rows_valid: Int32,
    q_base: Int32,
    q_tok0: Int32,
    Hq: Int32,
    kv_head: Int32,
    chunk_lo: Int32,
    chunk_hi: Int32,
    tile_lo: Int32,
    tile_hi: Int32,
    mode: Int32,
    slot_base: Int32,
):
    D, g, TPC, NP, WS = st.D, st.g, st.tpc, st.n_pad, st.ws_row
    SPT = st.slots_per_tile
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % 128
    cO = cute.make_identity_tensor((st.M_pv, NP))
    tOcO = pv_thr_mma.partition_C(cO)
    ld_atom, st_atom = copy_atoms(st)
    tiled_ld = tcgen05.make_tmem_copy(ld_atom, tOtO_slots[0])
    tiled_st = tcgen05.make_tmem_copy(st_atom, tOtO_slots[0])
    thr_ld = tiled_ld.get_slice(thread_idx)
    thr_st = tiled_st.get_slice(thread_idx)
    tLD = []
    tST = []
    tLDm = []
    tSTm = []
    for sl in cutlass.range_constexpr(SPT):
        tLD.append(thr_ld.partition_S(tOtO_slots[sl]))
        tST.append(thr_st.partition_D(tOtO_slots[sl]))
        tLDm.append(thr_ld.partition_S(tOtOm_slots[sl]))
        tSTm.append(thr_st.partition_D(tOtOm_slots[sl]))
    tLDc = thr_ld.partition_D(tOcO)  # (dim, row) coordinates of the thread's registers
    rO = cute.make_rmem_tensor(tLDc.shape, ACC_DTYPE)
    rM = cute.make_rmem_tensor(tLDc.shape, ACC_DTYPE)
    n_el = cute.size(rO)
    rA = cute.make_rmem_tensor(
        cute.make_layout((n_el,)), ACC_DTYPE
    )  # per-register coefficients (alpha / a1 / a2 / inv of the register's row)
    rB = cute.make_rmem_tensor(cute.make_layout((n_el,)), ACC_DTYPE)
    own = thread_idx < NP  # this thread owns row thread_idx's statistics chain
    multi = (mode == 0) & (chunk_hi - chunk_lo > 1)
    N = tile_hi - tile_lo
    m = Float32(_NEG_INF)
    l = Float32(0.0)
    m_r = Float32(_NEG_INF)
    l_r = Float32(0.0)
    a1 = Float32(1.0)
    a2 = Float32(0.0)
    first_merge = cutlass.Boolean(True)
    for n_t in range(N):
        t = tile_lo + n_t
        chunk_first = ((t % TPC) == 0) | (t == tile_lo)
        chunk_last = ((t % TPC) == TPC - 1) | (t == tile_hi - 1)
        cont = ((t % TPC) != 0) & (t != tile_lo)  # a previous PV of this chunk exists
        # ---- alpha of every row (the stats ring), the rescale of O^T once the previous PV is complete
        sth = stats_cons.wait_and_advance()
        my_alpha = s_alpha[(sth.index, thread_idx)]
        load_coefs(s_alpha, sth.index, tLDc, rA)
        any_rescale = cutlass.Boolean(False)
        for n in cutlass.range_constexpr(NP):
            if n < rows_valid:
                any_rescale = any_rescale | (s_alpha[(sth.index, n)] != Float32(1.0))
        sth.release()
        if cont:
            oh = o_cons.wait_and_advance()  # PV(n-1) complete
            if any_rescale:
                for sl in cutlass.range_constexpr(SPT):
                    cute.copy(tiled_ld, tLD[sl], rO)
                    for e in cutlass.range_constexpr(
                        0, n_el, 2
                    ):  # mul(O, alpha[row]) two elements per mul.rn.f32x2
                        y0, y1 = CuteF32.mul2((rO[e], rO[e + 1]), (rA[e], rA[e + 1]))
                        rO[e] = y0
                        rO[e + 1] = y1
                    cute.copy(tiled_st, rO, tST[sl])
                cute.arch.fence_view_async_tmem_store()
            oh.release()
        # ---- the statistics chain of the owned row: (m', r_warps) from the softmax warpgroup
        sth = stats_cons.wait_and_advance()
        if own:
            m_new = s_m[(sth.index, thread_idx)]
            r_t = row_sum(
                s_r, sth.index, thread_idx, 4
            )  # the top levels of the rowsum tree: the four warps' partials in warp order
            if chunk_first:
                l = Float32(0.0)
            l = CuteF32.fma(l, my_alpha, r_t)
            m = m_new
        sth.release()
        if chunk_last:
            if multi:
                if own:
                    m_r, l_r, a1, a2 = merge_coeffs(m_r, l_r, m, l, c, nc)
                    s_ab[(0, thread_idx)] = a1
                    s_ab[(1, thread_idx)] = a2
                cute.arch.barrier(barrier_id=BAR_COEF, number_of_threads=128)
                if t != tile_hi - 1:  # an interior chunk end: fold O_c^T into O_m^T now
                    load_coefs(s_ab, Int32(0), tLDc, rA)
                    load_coefs(s_ab, Int32(1), tLDc, rB)
                    oh = o_cons.wait_and_advance()
                    for sl in cutlass.range_constexpr(SPT):
                        cute.copy(tiled_ld, tLD[sl], rO)
                        if first_merge:
                            for e in cutlass.range_constexpr(n_el):
                                rM[e] = CuteF32.fma(rO[e], rB[e], Float32(0.0))
                        else:
                            cute.copy(tiled_ld, tLDm[sl], rM)
                            for e in cutlass.range_constexpr(n_el):
                                rM[e] = CuteF32.fma(
                                    rO[e], rB[e], CuteF32.mul(rM[e], rA[e])
                                )
                        cute.copy(tiled_st, rM, tSTm[sl])
                    cute.arch.fence_view_async_tmem_store()
                    cute.arch.fence_view_async_tmem_load()
                    oh.release()
                    first_merge = cutlass.Boolean(False)
            else:
                if own:
                    m_r = m
                    l_r = l
    if (
        N % 2 == 1
    ):  # an odd unit: the phantom tile's two (empty) entries, committed by softmax warpgroup 1
        sth = stats_cons.wait_and_advance()
        sth.release()
        sth = stats_cons.wait_and_advance()
        sth.release()
    # ---- epilogue: the normaliser (mode 0) or (m, l) of the partial rows (mode 1) from the owning threads, then every thread's columns
    if own:
        inv = CuteF32.div(Float32(1.0), l_r)
        s_inv[thread_idx] = inv
        if (mode != 0) & (thread_idx < rows_valid):
            mWs[(slot_base + thread_idx).to(Int64) * WS + D] = m_r
            mWs[(slot_base + thread_idx).to(Int64) * WS + D + 1] = l_r
    cute.arch.barrier(barrier_id=BAR_COEF, number_of_threads=128)
    load_coefs(s_ab, Int32(0), tLDc, rA)
    load_coefs(s_ab, Int32(1), tLDc, rB)
    oh = (
        o_cons.wait_and_advance()
    )  # the unit's last PV complete; held until the reads below are done
    for sl in cutlass.range_constexpr(SPT):
        cute.copy(tiled_ld, tLD[sl], rO)
        if mode == 0:
            if multi:  # the last chunk's merge, as the interior one
                if first_merge:
                    for e in cutlass.range_constexpr(n_el):
                        rM[e] = CuteF32.fma(rO[e], rB[e], Float32(0.0))
                else:
                    cute.copy(tiled_ld, tLDm[sl], rM)
                    for e in cutlass.range_constexpr(n_el):
                        rM[e] = CuteF32.fma(rO[e], rB[e], CuteF32.mul(rM[e], rA[e]))
            else:
                for e in cutlass.range_constexpr(n_el):
                    rM[e] = rO[e]
            for e in cutlass.range_constexpr(n_el):
                n = tLDc[e][1]
                d = tLDc[e][0] + sl * st.slot_dims
                if n < rows_valid:
                    out_row = (q_base + q_tok0 + n // g) * Hq + kv_head * g + n % g
                    mO[(out_row.to(Int64), d)] = CuteF32.add(
                        CuteF32.mul(rM[e], s_inv[n]), Float32(0.0)
                    ).to(IN_DTYPE)
        else:
            for e in cutlass.range_constexpr(n_el):
                n = tLDc[e][1]
                d = tLDc[e][0] + sl * st.slot_dims
                if n < rows_valid:
                    mWs[(slot_base + n).to(Int64) * WS + d] = rO[e]
    cute.arch.fence_view_async_tmem_load()
    oh.release()
    return o_cons, stats_cons
