# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The correction warpgroup block: one thread per Q row owns the TMEM accumulator of its row **and the
row's statistics chain**. Per tile it receives from the softmax warpgroups ``alpha`` (early) and
``(m', r_0 .. r_{n_wg-1})`` (late: the tile maximum and the row-sum partials of the key slices) and

* rescales ``O_c *= alpha`` (``mul.rn`` per element) when any row of the warp has ``alpha != 1``
  (``mul(O, 1.0)`` is an identity, so the skip is a placement), once the previous PV is complete;
* completes the ``rowsum`` tree -- its top level is the pairwise ``add`` of the slices' partials in slice
  order (``spec.rowsum_reduction`` under ``key: [(T/n_wg, reg), (n_wg, warp)]``: the same canonical tree) --
  and advances ``l = fma(l, alpha, r_t)``, ``m = m'``;
* at an interior chunk end (several chunks, mode 0): ``spec.merge``'s scalar part gives ``(a1, a2)`` from
  ``(m_r, m)`` (``(1, 0)`` for a chunk absent for the row), ``l_r = fma(l, a2, mul(l_r, a1))`` and
  ``O_m = fma(O_c, a2, mul(O_m, a1))`` in TMEM (the unit's first merge takes ``mul(+0, a1) = +0``: the init state);
* the epilogue, holding the accumulator until its reads are done (the next unit's first PV overwrites
  it): the unit's last chunk merged the same way, then ``cvt_bf16(add(mul(O, div(1, l)), +0))`` to the
  output rows (mode 0), or the raw fp32 ``O_c`` rows and ``(m, l)`` to the workspace slot (mode 1, folded
  by ``fixup.py``).

TMEM traffic goes through ``tcgen05.copy.Ld32x32bOp`` / ``St32x32bOp`` in ``CT``-column pieces (16: the
warpgroup runs at 96 registers when there are two softmax warpgroups).

The running merged state ``O_m`` lives in TMEM at D <= 128 (``S | P | O_c | O_m`` <= 448 columns) and in a per-CTA
global scratch at D = 256 (``st.om_tmem`` False: TMEM holds ``S | P | O_c`` = 448; the scratch row of this CTA and
TMEM lane is ``gOm``, touched once per chunk end, L2-resident) -- the placement the TS body's two-tile build uses.
The merge arithmetic (``merge_piece``) is one code path; only the source / sink of the ``O_m`` piece differs.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import tcgen05

from ....batch_invariant.fp_interface.cute import CuteF32
from ..fixup import VEC, fold_row
from ..plan import (
    G_CHUNK_LO,
    G_KVHEAD,
    G_KVLEN,
    G_NCHUNKS,
    G_QBASE,
    G_QLEN,
    G_QTOK0,
    G_ROWS,
    G_SLOT,
)
from ..spec import chunk_len
from .operands import ACC_DTYPE, BAR_FOLD, IN_DTYPE, OM_VEC, Q_ROWS, om_scratch_blocks


CT = 16  # TMEM columns per load / store with two softmax warpgroups (96 registers); 32 with one (StaticPlan.ct)
MERGE_DEPTH = 4  # O_m pieces of the global-scratch merge in flight (register buffers): the loads of pieces i+1 .. i+3 are issued before piece i
# is merged, so the eight dependent L2 round trips of the D = 256 merge become ~two (the correction runs at 208 registers)


def om_piece(gOm: cute.Tensor, i: int, CT: int) -> cute.Tensor:
    """Piece ``i`` (columns ``[i CT, (i + 1) CT)``) of this row's ``O_m`` in the global scratch, as a ``(OM_VEC, CT / OM_VEC)`` tensor whose flat
    (column-major) order is the column order, over the row-interleaved layout of ``operands.OM_VEC``: ``gOm`` is the row's base (``+ row * OM_VEC``)
    with strides ``(1, OM_VEC * 128)``. Trace-time (static ``i``), so the copies vectorise to 16 bytes per lane and a warp's lanes are contiguous."""
    return cute.make_tensor(
        gOm.iterator + i * (CT // OM_VEC) * (OM_VEC * Q_ROWS),
        cute.make_layout((OM_VEC, CT // OM_VEC), stride=(1, OM_VEC * Q_ROWS)),
    )


_NEG_INF = float("-inf")


@cute.jit
def rescale_tile(
    stats_cons,
    s_alpha: cute.Tensor,
    row: Int32,
    warp_active,
    cont,
    o_cons,
    tiled_ld,
    tiled_st,
    tLDc: cute.Tensor,
    tSTc: cute.Tensor,
    rO: cute.Tensor,
    n_pieces: cutlass.Constexpr,
    CT: cutlass.Constexpr,
):
    """One tile's rescale: take its alpha; when the tile continues a chunk (``cont``: a previous PV of the chunk
    exists), wait for that PV and rescale the rows whose maximum moved."""
    sth = stats_cons.wait_and_advance()
    alpha = s_alpha[(sth.index, row)]
    sth.release()
    if cont:
        oh = o_cons.wait_and_advance()  # PV(n-1) complete
        if warp_active & cute.arch.vote_any_sync(alpha != Float32(1.0)):
            for i in cutlass.range_constexpr(n_pieces):
                cute.copy(
                    tiled_ld, cute.make_tensor(tLDc.iterator + i * CT, tLDc.layout), rO
                )
                for e in cutlass.range_constexpr(
                    0, cute.size(rO), 2
                ):  # mul(O, alpha) two elements per mul.rn.f32x2
                    y0, y1 = CuteF32.mul2((rO[e], rO[e + 1]), (alpha, alpha))
                    rO[e] = y0
                    rO[e + 1] = y1
                cute.copy(
                    tiled_st, rO, cute.make_tensor(tSTc.iterator + i * CT, tSTc.layout)
                )
        cute.arch.fence_view_async_tmem_store()
        oh.release()
    return stats_cons, o_cons, alpha


def _pairwise_add(vals):
    """Balanced pairwise ``add`` tree over a static list of values (plain Python: trace-time structure)."""
    while len(vals) > 1:
        vals = [CuteF32.add(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    return vals[0]


@cute.jit
def row_sum(
    s_r: cute.Tensor, slot: Int32, row: Int32, n_slices: cutlass.Constexpr
) -> Float32:
    """The top levels of the rowsum tree: the pairwise add of the slices' partials in slice (= key) order."""
    parts = []
    for k in cutlass.range_constexpr(n_slices):
        parts.append(s_r[(slot, k, row)])
    return _pairwise_add(parts)


@cute.jit
def merge_coeffs(
    m_r: Float32, l_r: Float32, m: Float32, l: Float32, c: Float32, nc: Float32
):
    """``spec.merge``'s scalar part: the running ``(m_r, l_r)`` with the chunk's ``(m, l)``; ``(1, 0)`` drops an
    absent chunk (``m == -inf``). Returns ``(m_r, l_r, a1, a2)``."""
    a1 = Float32(1.0)
    a2 = Float32(0.0)
    if m != Float32(_NEG_INF):
        mm = CuteF32.fmaxc(m_r, m)
        nmm = CuteF32.mul(mm, nc)
        a1 = CuteF32.ex2(CuteF32.add(CuteF32.mul(m_r, c), nmm))
        a2 = CuteF32.ex2(CuteF32.add(CuteF32.mul(m, c), nmm))
        l_r = CuteF32.fma(l, a2, CuteF32.mul(l_r, a1))
        m_r = mm
    return m_r, l_r, a1, a2


@cute.jit
def merge_piece(
    i: cutlass.Constexpr,
    first_merge,
    a1: Float32,
    a2: Float32,
    om_tmem: cutlass.Constexpr,
    tiled_ld,
    tLDm,
    gOm,
    rO: cute.Tensor,
    rM: cute.Tensor,
    CT: cutlass.Constexpr,
):
    """``rM = fma(O_c piece (in rO), a2, mul(O_m piece, a1))`` with the O_m piece from TMEM (``om_tmem``) or from this row's
    global scratch ``gOm`` (``+0`` on the unit's first merge: the init state)."""
    if first_merge:
        for e in cutlass.range_constexpr(cute.size(rO)):
            rM[e] = CuteF32.fma(rO[e], a2, Float32(0.0))
    else:
        if cutlass.const_expr(om_tmem):
            cute.copy(
                tiled_ld, cute.make_tensor(tLDm.iterator + i * CT, tLDm.layout), rM
            )
        else:
            cute.autovec_copy(om_piece(gOm, i, CT), rM)
        for e in cutlass.range_constexpr(cute.size(rO)):
            rM[e] = CuteF32.fma(rO[e], a2, CuteF32.mul(rM[e], a1))


@cute.jit
def store_om_piece(
    i: cutlass.Constexpr,
    om_tmem: cutlass.Constexpr,
    tiled_st,
    tSTm,
    gOm,
    rM: cute.Tensor,
    CT: cutlass.Constexpr,
):
    """The merged piece back to ``O_m``'s home: TMEM or the global scratch row."""
    if cutlass.const_expr(om_tmem):
        cute.copy(tiled_st, rM, cute.make_tensor(tSTm.iterator + i * CT, tSTm.layout))
    else:
        cute.autovec_copy(rM, om_piece(gOm, i, CT))


@cute.jit
def merge_global_pipelined(
    first_merge,
    a1: Float32,
    a2: Float32,
    inv: Float32,
    tiled_ld,
    tLDc: cute.Tensor,
    gOm,
    rO: cute.Tensor,
    rMs,
    rOut: cute.Tensor,
    mO: cute.Tensor,
    out_row: Int32,
    row_valid,
    n_pieces: cutlass.Constexpr,
    CT: cutlass.Constexpr,
    to_output: cutlass.Constexpr,
    D: cutlass.Constexpr,
):
    """The chunk merge with ``O_m`` in the global scratch, software-pipelined ``len(rMs)`` deep: the ``O_m`` pieces ``i + 1 .. i + depth - 1``
    are in flight (one register buffer each) while piece ``i`` is merged -- ``fma(O_c, a2, mul(O_m, a1))`` per element, exactly
    ``merge_piece``'s arithmetic -- and written back to the scratch (interior chunk end) or normalised to the output row (``to_output``:
    the fused unit's epilogue). A placement: the dependent L2 round trips of the D = 256 merge overlap each other."""
    depth = len(rMs)
    if first_merge:
        rMa = rMs[0]
        for i in cutlass.range_constexpr(n_pieces):
            cute.copy(
                tiled_ld, cute.make_tensor(tLDc.iterator + i * CT, tLDc.layout), rO
            )
            for e in cutlass.range_constexpr(cute.size(rO)):
                rMa[e] = CuteF32.fma(rO[e], a2, Float32(0.0))
            if cutlass.const_expr(to_output):
                for e in cutlass.range_constexpr(cute.size(rO)):
                    rOut[e] = CuteF32.add(CuteF32.mul(rMa[e], inv), Float32(0.0)).to(
                        IN_DTYPE
                    )
                if row_valid:
                    cute.autovec_copy(
                        rOut,
                        cute.make_tensor(
                            mO.iterator
                            + cute.assume(out_row.to(Int64) * D + i * CT, divby=8),
                            cute.make_layout((cute.size(rO),)),
                        ),
                    )
            else:
                cute.autovec_copy(rMa, om_piece(gOm, i, CT))
    else:
        for i in cutlass.range_constexpr(
            min(depth - 1, n_pieces)
        ):  # prologue: the first depth - 1 pieces in flight
            cute.autovec_copy(om_piece(gOm, i, CT), rMs[i % depth])
        for i in cutlass.range_constexpr(n_pieces):
            cur = rMs[i % depth]
            if cutlass.const_expr(i + depth - 1 < n_pieces):
                nxt = rMs[(i + depth - 1) % depth]
                cute.autovec_copy(om_piece(gOm, i + depth - 1, CT), nxt)
            cute.copy(
                tiled_ld, cute.make_tensor(tLDc.iterator + i * CT, tLDc.layout), rO
            )
            for e in cutlass.range_constexpr(cute.size(rO)):
                cur[e] = CuteF32.fma(rO[e], a2, CuteF32.mul(cur[e], a1))
            if cutlass.const_expr(to_output):
                for e in cutlass.range_constexpr(cute.size(rO)):
                    rOut[e] = CuteF32.add(CuteF32.mul(cur[e], inv), Float32(0.0)).to(
                        IN_DTYPE
                    )
                if row_valid:
                    cute.autovec_copy(
                        rOut,
                        cute.make_tensor(
                            mO.iterator
                            + cute.assume(out_row.to(Int64) * D + i * CT, divby=8),
                            cute.make_layout((cute.size(rO),)),
                        ),
                    )
            else:
                cute.autovec_copy(cur, om_piece(gOm, i, CT))


@cute.jit
def merge_chunk(
    a1: Float32,
    a2: Float32,
    warp_active,
    first_merge,
    o_cons,
    tiled_ld,
    tiled_st,
    tLDc: cute.Tensor,
    tLDm,
    tSTm,
    gOm,
    rO: cute.Tensor,
    rM: cute.Tensor,
    rMs,
    n_pieces: cutlass.Constexpr,
    CT: cutlass.Constexpr,
    om_tmem: cutlass.Constexpr,
):
    """An interior chunk end: the chunk's last PV is complete; fold its accumulator into the running state."""
    oh = o_cons.wait_and_advance()
    if warp_active:
        if cutlass.const_expr(om_tmem):
            for i in cutlass.range_constexpr(n_pieces):
                cute.copy(
                    tiled_ld, cute.make_tensor(tLDc.iterator + i * CT, tLDc.layout), rO
                )
                merge_piece(
                    i, first_merge, a1, a2, om_tmem, tiled_ld, tLDm, gOm, rO, rM, CT
                )
                store_om_piece(i, om_tmem, tiled_st, tSTm, gOm, rM, CT)
            cute.arch.fence_view_async_tmem_store()
        else:
            merge_global_pipelined(
                first_merge,
                a1,
                a2,
                Float32(1.0),
                tiled_ld,
                tLDc,
                gOm,
                rO,
                rMs,
                rM,
                rM,
                Int32(0),
                False,
                n_pieces,
                CT,
                False,
                1,
            )
    cute.arch.fence_view_async_tmem_load()
    oh.release()
    return o_cons


@cute.jit
def epilogue(
    a1: Float32,
    a2: Float32,
    inv: Float32,
    m_r: Float32,
    l_r: Float32,
    row: Int32,
    row_valid,
    warp_active,
    multi,
    first_merge,
    mode,
    o_cons,
    tiled_ld,
    tLDc: cute.Tensor,
    tLDm,
    gOm,
    rO: cute.Tensor,
    rM: cute.Tensor,
    rMs,
    rOut: cute.Tensor,
    mO: cute.Tensor,
    mWs: cute.Tensor,
    out_row: Int32,
    slot_base: Int32,
    D: cutlass.Constexpr,
    n_pieces: cutlass.Constexpr,
    CT: cutlass.Constexpr,
    om_tmem: cutlass.Constexpr,
    WS: cutlass.Constexpr,
):
    oh = (
        o_cons.wait_and_advance()
    )  # the unit's last PV complete; held until the reads below are done
    if mode == 0:
        if warp_active:
            if cutlass.const_expr(not om_tmem):
                if multi:  # the last chunk's merge from the global scratch, pipelined, into the output
                    merge_global_pipelined(
                        first_merge,
                        a1,
                        a2,
                        inv,
                        tiled_ld,
                        tLDc,
                        gOm,
                        rO,
                        rMs,
                        rOut,
                        mO,
                        out_row,
                        row_valid,
                        n_pieces,
                        CT,
                        True,
                        D,
                    )
                else:
                    for i in cutlass.range_constexpr(n_pieces):
                        cute.copy(
                            tiled_ld,
                            cute.make_tensor(tLDc.iterator + i * CT, tLDc.layout),
                            rO,
                        )
                        for e in cutlass.range_constexpr(cute.size(rO)):
                            rOut[e] = CuteF32.add(
                                CuteF32.mul(rO[e], inv), Float32(0.0)
                            ).to(IN_DTYPE)
                        if row_valid:
                            cute.autovec_copy(
                                rOut,
                                cute.make_tensor(
                                    mO.iterator
                                    + cute.assume(
                                        out_row.to(Int64) * D + i * CT, divby=8
                                    ),
                                    cute.make_layout((cute.size(rO),)),
                                ),
                            )
            else:
                for i in cutlass.range_constexpr(n_pieces):
                    cute.copy(
                        tiled_ld,
                        cute.make_tensor(tLDc.iterator + i * CT, tLDc.layout),
                        rO,
                    )
                    if multi:  # the last chunk's merge, as merge_chunk's
                        merge_piece(
                            i,
                            first_merge,
                            a1,
                            a2,
                            om_tmem,
                            tiled_ld,
                            tLDm,
                            gOm,
                            rO,
                            rM,
                            CT,
                        )
                    else:
                        for e in cutlass.range_constexpr(cute.size(rO)):
                            rM[e] = rO[e]
                    for e in cutlass.range_constexpr(cute.size(rO)):
                        rOut[e] = CuteF32.add(CuteF32.mul(rM[e], inv), Float32(0.0)).to(
                            IN_DTYPE
                        )
                    if row_valid:
                        cute.autovec_copy(
                            rOut,
                            cute.make_tensor(
                                mO.iterator
                                + cute.assume(out_row.to(Int64) * D + i * CT, divby=8),
                                cute.make_layout((cute.size(rO),)),
                            ),
                        )
    else:
        if warp_active:
            for i in cutlass.range_constexpr(n_pieces):
                cute.copy(
                    tiled_ld, cute.make_tensor(tLDc.iterator + i * CT, tLDc.layout), rO
                )
                if row_valid:
                    cute.autovec_copy(
                        rO,
                        cute.make_tensor(
                            mWs.iterator
                            + cute.assume(
                                (slot_base + row).to(Int64) * WS + i * CT, divby=4
                            ),
                            cute.make_layout((cute.size(rO),)),
                        ),
                    )
            if row_valid:
                mWs[(slot_base + row).to(Int64) * WS + D] = m_r
                mWs[(slot_base + row).to(Int64) * WS + D + 1] = l_r
    cute.arch.fence_view_async_tmem_load()
    oh.release()
    return o_cons


@cute.jit
def correction_loop(
    st,
    pv_thr_mma: cute.TiledMma,
    tOtOc: cute.Tensor,
    tOtOm,
    o_cons,
    stats_cons,
    s_alpha: cute.Tensor,
    s_m: cute.Tensor,
    s_r: cute.Tensor,
    mO: cute.Tensor,
    mWs: cute.Tensor,
    mScratch: cute.Tensor,
    cta: Int32,
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
    """``tOtOm`` is the TMEM ``O_m`` (``None`` at D = 256, whose ``O_m`` is row ``(cta, lane)`` of ``mScratch``)."""
    D, g, TPC = st.D, st.g, st.tpc
    CT = st.ct
    om_tmem = st.om_tmem
    WS = st.ws_row
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % 128
    cO = cute.make_identity_tensor((Q_ROWS, D))
    tOcO = pv_thr_mma.partition_C(cO)
    tOtOc_i = cute.make_tensor(
        tOtOc.iterator, cute.composition(tOtOc.layout, cute.make_layout((Q_ROWS, CT)))
    )
    tOcO_i = cute.make_tensor(
        tOcO.iterator, cute.composition(tOcO.layout, cute.make_layout((Q_ROWS, CT)))
    )
    ld_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(CT)), ACC_DTYPE
    )
    st_atom = cute.make_copy_atom(
        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(CT)), ACC_DTYPE
    )
    tiled_ld = tcgen05.make_tmem_copy(ld_atom, tOtOc_i)
    tiled_st = tcgen05.make_tmem_copy(st_atom, tOtOc_i)
    thr_ld = tiled_ld.get_slice(thread_idx)
    thr_st = tiled_st.get_slice(thread_idx)
    tLDc = thr_ld.partition_S(tOtOc_i)
    tSTc = thr_st.partition_D(tOtOc_i)
    tLDm = tSTm = None
    if cutlass.const_expr(om_tmem):
        tOtOm_i = cute.make_tensor(
            tOtOm.iterator,
            cute.composition(tOtOm.layout, cute.make_layout((Q_ROWS, CT))),
        )
        tLDm = thr_ld.partition_S(tOtOm_i)
        tSTm = thr_st.partition_D(tOtOm_i)
    tLDcO = thr_ld.partition_D(tOcO_i)
    rO = cute.make_rmem_tensor(tLDcO.shape, ACC_DTYPE)
    rM = cute.make_rmem_tensor(tLDcO.shape, ACC_DTYPE)
    # the O_m piece buffers of the pipelined global merge (D = 256): MERGE_DEPTH of them (one at D <= 128, where the merge is in TMEM)
    rMs = tuple(
        cute.make_rmem_tensor(tLDcO.shape, ACC_DTYPE)
        for _ in range(MERGE_DEPTH if not om_tmem else 1)
    )
    n_el = cute.size(rO)
    rOut = cute.make_rmem_tensor(cute.make_layout((n_el,)), IN_DTYPE)
    n_pieces = D // CT
    row = tLDcO[
        0
    ][
        0
    ]  # = the TMEM lane; in the replicated-row placement the live rows are 0..31 (warp 0)
    # this (CTA, row)'s O_m when not in TMEM: the row-interleaved block layout of operands.OM_VEC (om_piece), base = block + row * OM_VEC
    gOm = cute.make_tensor(
        mScratch.iterator
        + cute.assume(
            cta * (om_scratch_blocks(D) * Q_ROWS * D) + row * OM_VEC, divby=OM_VEC
        ),
        cute.make_layout((OM_VEC, D // OM_VEC), stride=(1, OM_VEC * Q_ROWS)),
    )  # divby: 16-byte vector copies per lane
    row_valid = row < rows_valid
    warp_active = (
        (thread_idx // 32) * 32
    ) < rows_valid  # warp-uniform: any stored row in this warp's 32 rows
    token = q_tok0 + row // g
    multi = (mode == 0) & (chunk_hi - chunk_lo > 1)
    N = tile_hi - tile_lo
    m = Float32(_NEG_INF)
    l = Float32(0.0)
    m_r = Float32(_NEG_INF)
    l_r = Float32(0.0)
    a1 = Float32(1.0)
    a2 = Float32(0.0)
    first_merge = cutlass.Boolean(True)
    for n in range(N):
        t = tile_lo + n
        chunk_first = ((t % TPC) == 0) | (t == tile_lo)
        chunk_last = ((t % TPC) == TPC - 1) | (t == tile_hi - 1)
        cont = ((t % TPC) != 0) & (t != tile_lo)  # a previous PV of this chunk exists
        stats_cons, o_cons, alpha = rescale_tile(
            stats_cons,
            s_alpha,
            row,
            warp_active,
            cont,
            o_cons,
            tiled_ld,
            tiled_st,
            tLDc,
            tSTc,
            rO,
            n_pieces,
            CT,
        )
        # ---- the statistics chain: (m', r_slices) from the softmax warpgroups
        sth = stats_cons.wait_and_advance()
        m_new = s_m[(sth.index, row)]
        r_t = row_sum(s_r, sth.index, row, st.n_slices)
        if cutlass.const_expr(st.rep_ok):
            if rows_valid <= st.rep_rows:  # the replicated-row placement: 4 n_wg slices
                r_t = row_sum(s_r, sth.index, row, st.slices_rep)
        sth.release()
        if chunk_first:
            l = Float32(0.0)
        l = CuteF32.fma(l, alpha, r_t)
        m = m_new
        if chunk_last:
            if multi:
                m_r, l_r, a1, a2 = merge_coeffs(m_r, l_r, m, l, c, nc)
                if t != tile_hi - 1:  # an interior chunk end: fold O_c into O_m now
                    o_cons = merge_chunk(
                        a1,
                        a2,
                        warp_active,
                        first_merge,
                        o_cons,
                        tiled_ld,
                        tiled_st,
                        tLDc,
                        tLDm,
                        tSTm,
                        gOm,
                        rO,
                        rM,
                        rMs,
                        n_pieces,
                        CT,
                        om_tmem,
                    )
                    first_merge = cutlass.Boolean(False)
            else:
                m_r = m
                l_r = l
    inv = CuteF32.div(Float32(1.0), l_r)
    out_row = (q_base + token) * Hq + kv_head * g + row % g
    o_cons = epilogue(
        a1,
        a2,
        inv,
        m_r,
        l_r,
        row,
        row_valid,
        warp_active,
        multi,
        first_merge,
        mode,
        o_cons,
        tiled_ld,
        tLDc,
        tLDm,
        gOm,
        rO,
        rM,
        rMs,
        rOut,
        mO,
        mWs,
        out_row,
        slot_base,
        D,
        n_pieces,
        CT,
        om_tmem,
        WS,
    )
    return o_cons, stats_cons


@cute.jit
def fold_group(
    st,
    mWs: cute.Tensor,
    mO: cute.Tensor,
    mGroups: cute.Tensor,
    mCounters: cute.Tensor,
    s_fold,
    grp: Int32,
    c: Float32,
    nc: Float32,
    window_left: Int32,
    Hq: Int32,
):
    """The in-kernel chunk fold of a split group (``grp >= 0``: this unit's group is folded in-kernel). After the unit's partial rows
    are written (``epilogue``, mode 1) every correction thread fences them to GPU scope and the warpgroup counts its arrival on the
    group's counter; the LAST arriver (``n_chunks - 1`` units before it) fences, folds every row of the group in chunk order with the
    fixup's ``fold_row`` (thread ``t``: rows ``4 p + t // 32`` over the passes ``p``, output columns ``4 (t % 32) ..``) and writes the
    output rows, then resets the counter for the next launch. The order in which units arrive never reaches the bits: the fold
    reads the workspace slots in chunk order whatever CTA wrote them (``RLK.Attn.splitKernel_eq_fused``). ``PF`` partial rows are
    loaded ahead per step (4 at the 96-register budget of the two-warpgroup build, 8 otherwise)."""
    if grp >= 0:
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % 128
        n_chunks = mGroups[
            (grp, G_NCHUNKS)
        ]  # the group's fields are plan constants: read before the hand-off
        rows = mGroups[(grp, G_ROWS)]
        slot = mGroups[(grp, G_SLOT)]
        chunk_lo = mGroups[(grp, G_CHUNK_LO)]
        q_tok0 = mGroups[(grp, G_QTOK0)]
        kv_head = mGroups[(grp, G_KVHEAD)]
        kv_len = mGroups[(grp, G_KVLEN)]
        q_len = mGroups[(grp, G_QLEN)]
        q_base = mGroups[(grp, G_QBASE)]
        cute.arch.barrier(barrier_id=BAR_FOLD, number_of_threads=128)
        if thread_idx == 0:
            s_fold[0] = cute.arch.atomic_add(
                mCounters.iterator + grp, Int32(1), sem="acq_rel", scope="gpu"
            )
        cute.arch.barrier(barrier_id=BAR_FOLD, number_of_threads=128)
        arrived = s_fold[0]
        if arrived == n_chunks - 1:
            thr_per_row = st.D // VEC
            rows_per_pass = 128 // thr_per_row
            d0 = (thread_idx % thr_per_row) * VEC
            for r0 in range(0, rows, rows_per_pass):
                r = r0 + thread_idx // thr_per_row
                if r < rows:
                    fold_row(
                        mWs,
                        mO,
                        slot,
                        rows,
                        chunk_lo,
                        n_chunks,
                        q_tok0,
                        kv_head,
                        kv_len,
                        q_len,
                        q_base,
                        c,
                        nc,
                        window_left,
                        Hq,
                        r,
                        d0,
                        st.D,
                        st.g,
                        chunk_len(st.D),
                        8 if st.n_wg == 1 else 4,
                        st.ws_row,
                    )
            if thread_idx == 0:
                mCounters[grp] = Int32(
                    0
                )  # clean for the next launch (no other unit of this group is left)


@cute.jit
def corr_tile_q2(
    st,
    t: Int32,
    lo: Int32,
    hi: Int32,
    multi,
    row: Int32,
    rows_valid: Int32,
    warp_active,
    stats_cons,
    o_cons,
    s_alpha: cute.Tensor,
    s_m: cute.Tensor,
    s_r: cute.Tensor,
    tiled_ld,
    tiled_st,
    tLDc: cute.Tensor,
    tSTc: cute.Tensor,
    tLDm: cute.Tensor,
    tSTm: cute.Tensor,
    rO: cute.Tensor,
    rM: cute.Tensor,
    rMs,
    gOm,
    m: Float32,
    l: Float32,
    m_r: Float32,
    l_r: Float32,
    a1: Float32,
    a2: Float32,
    first_merge,
    c: Float32,
    nc: Float32,
):
    """One KV tile of one Q tile in the paired placement: ``correction_loop``'s tile step verbatim (alpha and the O_c rescale, the
    statistics chain, the interior chunk merge into O_m in TMEM). The two-stage stats ring holds both tiles' entries interleaved (two
    per tile per Q tile): this Q tile's two, then the other's two are skipped."""
    TPC = st.tpc
    CT = st.ct
    n_pieces = st.D // CT
    chunk_first = ((t % TPC) == 0) | (t == lo)
    chunk_last = ((t % TPC) == TPC - 1) | (t == hi - 1)
    cont = ((t % TPC) != 0) & (t != lo)
    stats_cons, o_cons, alpha = rescale_tile(
        stats_cons,
        s_alpha,
        row,
        warp_active,
        cont,
        o_cons,
        tiled_ld,
        tiled_st,
        tLDc,
        tSTc,
        rO,
        n_pieces,
        CT,
    )
    sth = stats_cons.wait_and_advance()
    m_new = s_m[(sth.index, row)]
    r_t = row_sum(s_r, sth.index, row, st.n_slices)
    if cutlass.const_expr(st.rep_ok):
        if rows_valid <= st.rep_rows:
            r_t = row_sum(s_r, sth.index, row, st.slices_rep)
    sth.release()
    stats_cons.advance(2)
    if chunk_first:
        l = Float32(0.0)
    l = CuteF32.fma(l, alpha, r_t)
    m = m_new
    if chunk_last:
        if multi:
            m_r, l_r, a1, a2 = merge_coeffs(m_r, l_r, m, l, c, nc)
            if t != hi - 1:
                o_cons = merge_chunk(
                    a1,
                    a2,
                    warp_active,
                    first_merge,
                    o_cons,
                    tiled_ld,
                    tiled_st,
                    tLDc,
                    tLDm,
                    tSTm,
                    gOm,
                    rO,
                    rM,
                    rMs,
                    n_pieces,
                    CT,
                    st.om_tmem,
                )
                first_merge = cutlass.Boolean(False)
        else:
            m_r = m
            l_r = l
    return stats_cons, o_cons, m, l, m_r, l_r, a1, a2, first_merge


@cute.jit
def correction_loop_q2(
    st,
    pv_thr_mma: cute.TiledMma,
    tOtOc: cute.Tensor,
    sOm: cute.Tensor,
    o_cons_a,
    o_cons_b,
    stats_cons_a,
    stats_cons_b,
    s_alpha: cute.Tensor,
    s_m: cute.Tensor,
    s_r: cute.Tensor,
    mO: cute.Tensor,
    mWs: cute.Tensor,
    c: Float32,
    nc: Float32,
    Hq: Int32,
    q_base: Int32,
    kv_head: Int32,
    rows_a: Int32,
    tok_a: Int32,
    clo_a: Int32,
    chi_a: Int32,
    lo_a: Int32,
    hi_a: Int32,
    mode_a: Int32,
    slot_a: Int32,
    rows_b: Int32,
    tok_b: Int32,
    clo_b: Int32,
    chi_b: Int32,
    lo_b: Int32,
    hi_b: Int32,
    mode_b: Int32,
    slot_b: Int32,
):
    """The paired placement's correction warpgroup: both Q tiles' statistics chains, O_c (Q tile B's at ``tmem_q2_stride`` columns past
    A's) and O_m (smem ``sOm``: tile k's 128 x D fp32 block at ``k 128 D``, the row-interleaved layout of ``operands.OM_VEC``), stepped A then B per tile of the union of their ranges (the MMA warp's PV_A(t), PV_B(t) order: no wait of one
    tile's step on the other's later tiles), then A's epilogue and B's. Every per-tile computation is ``correction_loop``'s."""
    D, g = st.D, st.g
    CT = st.ct
    WS = st.ws_row
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % 128
    cO = cute.make_identity_tensor((Q_ROWS, D))
    tOcO = pv_thr_mma.partition_C(cO)
    tOcO_i = cute.make_tensor(
        tOcO.iterator, cute.composition(tOcO.layout, cute.make_layout((Q_ROWS, CT)))
    )
    ld_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(CT)), ACC_DTYPE
    )
    st_atom = cute.make_copy_atom(
        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(CT)), ACC_DTYPE
    )
    tOtOc_i = cute.make_tensor(
        tOtOc.iterator, cute.composition(tOtOc.layout, cute.make_layout((Q_ROWS, CT)))
    )
    tiled_ld = tcgen05.make_tmem_copy(ld_atom, tOtOc_i)
    tiled_st = tcgen05.make_tmem_copy(st_atom, tOtOc_i)
    thr_ld = tiled_ld.get_slice(thread_idx)
    thr_st = tiled_st.get_slice(thread_idx)
    QS = st.tmem_q2_stride
    tLDc_a = thr_ld.partition_S(tOtOc_i)
    tSTc_a = thr_st.partition_D(tOtOc_i)
    tLDc_b = cute.make_tensor(tLDc_a.iterator + QS, tLDc_a.layout)
    tSTc_b = cute.make_tensor(tSTc_a.iterator + QS, tSTc_a.layout)
    tLDm_a = tSTm_a = tLDm_b = tSTm_b = None
    tLDcO = thr_ld.partition_D(tOcO_i)
    rO = cute.make_rmem_tensor(tLDcO.shape, ACC_DTYPE)
    rM = cute.make_rmem_tensor(tLDcO.shape, ACC_DTYPE)
    rMs = (
        cute.make_rmem_tensor(tLDcO.shape, ACC_DTYPE),
        cute.make_rmem_tensor(tLDcO.shape, ACC_DTYPE),
    )
    n_el = cute.size(rO)
    rOut = cute.make_rmem_tensor(cute.make_layout((n_el,)), IN_DTYPE)
    n_pieces = D // CT
    row = tLDcO[0][0]
    om_layout = cute.make_layout((OM_VEC, D // OM_VEC), stride=(1, OM_VEC * Q_ROWS))
    gOm_a = cute.make_tensor(
        sOm.iterator + cute.assume(row * OM_VEC, divby=OM_VEC), om_layout
    )
    gOm_b = cute.make_tensor(
        sOm.iterator + cute.assume(Q_ROWS * D + row * OM_VEC, divby=OM_VEC), om_layout
    )
    has_b = hi_b > lo_b
    rv_a = row < rows_a
    rv_b = row < rows_b
    wa_a = ((thread_idx // 32) * 32) < rows_a
    wa_b = ((thread_idx // 32) * 32) < rows_b
    multi_a = (mode_a == 0) & (chi_a - clo_a > 1)
    multi_b = (mode_b == 0) & (chi_b - clo_b > 1)
    m_a = Float32(_NEG_INF)
    l_a = Float32(0.0)
    mr_a = Float32(_NEG_INF)
    lr_a = Float32(0.0)
    a1_a = Float32(1.0)
    a2_a = Float32(0.0)
    fm_a = cutlass.Boolean(True)
    m_b = Float32(_NEG_INF)
    l_b = Float32(0.0)
    mr_b = Float32(_NEG_INF)
    lr_b = Float32(0.0)
    a1_b = Float32(1.0)
    a2_b = Float32(0.0)
    fm_b = cutlass.Boolean(True)
    lo = lo_a
    hi = hi_a
    if has_b:
        if lo_b < lo:
            lo = lo_b
        if hi_b > hi:
            hi = hi_b
    for t in range(lo, hi):
        if (t >= lo_a) & (t < hi_a):
            stats_cons_a, o_cons_a, m_a, l_a, mr_a, lr_a, a1_a, a2_a, fm_a = (
                corr_tile_q2(
                    st,
                    t,
                    lo_a,
                    hi_a,
                    multi_a,
                    row,
                    rows_a,
                    wa_a,
                    stats_cons_a,
                    o_cons_a,
                    s_alpha,
                    s_m,
                    s_r,
                    tiled_ld,
                    tiled_st,
                    tLDc_a,
                    tSTc_a,
                    tLDm_a,
                    tSTm_a,
                    rO,
                    rM,
                    rMs,
                    gOm_a,
                    m_a,
                    l_a,
                    mr_a,
                    lr_a,
                    a1_a,
                    a2_a,
                    fm_a,
                    c,
                    nc,
                )
            )
        if (t >= lo_b) & (t < hi_b):
            stats_cons_b, o_cons_b, m_b, l_b, mr_b, lr_b, a1_b, a2_b, fm_b = (
                corr_tile_q2(
                    st,
                    t,
                    lo_b,
                    hi_b,
                    multi_b,
                    row,
                    rows_b,
                    wa_b,
                    stats_cons_b,
                    o_cons_b,
                    s_alpha,
                    s_m,
                    s_r,
                    tiled_ld,
                    tiled_st,
                    tLDc_b,
                    tSTc_b,
                    tLDm_b,
                    tSTm_b,
                    rO,
                    rM,
                    rMs,
                    gOm_b,
                    m_b,
                    l_b,
                    mr_b,
                    lr_b,
                    a1_b,
                    a2_b,
                    fm_b,
                    c,
                    nc,
                )
            )
    if hi_a > lo_a:
        inv_a = CuteF32.div(Float32(1.0), lr_a)
        out_a = (q_base + tok_a + row // g) * Hq + kv_head * g + row % g
        o_cons_a = epilogue(
            a1_a,
            a2_a,
            inv_a,
            mr_a,
            lr_a,
            row,
            rv_a,
            wa_a,
            multi_a,
            fm_a,
            mode_a,
            o_cons_a,
            tiled_ld,
            tLDc_a,
            tLDm_a,
            gOm_a,
            rO,
            rM,
            rMs,
            rOut,
            mO,
            mWs,
            out_a,
            slot_a,
            D,
            n_pieces,
            CT,
            st.om_tmem,
            WS,
        )
    if has_b:
        inv_b = CuteF32.div(Float32(1.0), lr_b)
        out_b = (q_base + tok_b + row // g) * Hq + kv_head * g + row % g
        o_cons_b = epilogue(
            a1_b,
            a2_b,
            inv_b,
            mr_b,
            lr_b,
            row,
            rv_b,
            wa_b,
            multi_b,
            fm_b,
            mode_b,
            o_cons_b,
            tiled_ld,
            tLDc_b,
            tLDm_b,
            gOm_b,
            rO,
            rM,
            rMs,
            rOut,
            mO,
            mWs,
            out_b,
            slot_b,
            D,
            n_pieces,
            CT,
            st.om_tmem,
            WS,
        )
    return o_cons_a, o_cons_b, stats_cons_a, stats_cons_b
