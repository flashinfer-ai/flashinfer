# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""TMA producer block (the load warp): Q once per unit (one box per 64-wide half, 2-stage buffer so the
next unit's Q is prefetched), then the page boxes of K and of V of every tile of the unit into the smem
ring, in the order the MMA warp consumes them: K one tile ahead of V (``K_0, K_1, V_0, K_2, V_1, ...``),
because the MMA warp issues QK(n + 1) before PV(n) so that the softmax of tile n overlaps the next QK.
The order carries no arithmetic; absent pages (past the request's page count) load a clamped valid
page whose rows the MMA warp zeroes / the softmax masks.

The tiles of a unit are a flat sequence ``t = tile_lo + n`` (chunk boundaries are predicates on ``t``,
the same integer logic in every role).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.nvgpu import cpasync

from .operands import HALF, Q_ROWS, REP_ROWS


@cute.jit
def issue_slot(
    st,
    key0: Int32,
    kv_prod,
    atom: cute.CopyAtom,
    tg: cute.Tensor,
    ts: cute.Tensor,
    mKvIndices: cute.Tensor,
    kv_ptr0: Int32,
    n_pages: Int32,
    kv_head: Int32,
):
    """One ring slot (``slot_keys`` keys of K or V from absolute key ``key0``): acquire it, issue ``boxes_per_slot x halves`` TMA
    boxes (a box is ``box_rows = min(page, slot_keys)`` keys of one page).
    Under ``st.ragged`` the pool is one page of ``total_kv`` tokens and the box starts at the request's token offset plus the key
    (the domain shifted by ``kv_indptr[req]``, a token offset, in ``load_loop``); coordinates past the pool are zero-filled by the TMA and keys past the
    request's length are masked / zeroed exactly as the junk of a paged pool's pages is."""
    page = st.page
    h = kv_prod.acquire_and_advance()
    for i in cutlass.range_constexpr(st.boxes_per_slot):
        k0 = key0 + i * st.box_rows
        if cutlass.const_expr(st.ragged):
            for half in cutlass.range_constexpr(st.halves):
                cute.copy(
                    atom,
                    tg[(None, (half, k0 // st.box_rows, kv_head, 0))],
                    ts[(None, i, half, h.index)],
                    tma_bar_ptr=h.barrier,
                )
        else:
            pslot = k0 // page
            if cutlass.const_expr(page <= st.slot_keys):
                box_in_page = Int32(0)
            else:
                box_in_page = (k0 % page) // st.box_rows
            if pslot > n_pages - 1:
                pslot = n_pages - 1
            page_id = mKvIndices[kv_ptr0 + pslot]
            for half in cutlass.range_constexpr(st.halves):
                cute.copy(
                    atom,
                    tg[(None, (half, box_in_page, kv_head, page_id))],
                    ts[(None, i, half, h.index)],
                    tma_bar_ptr=h.barrier,
                )
    return kv_prod


@cute.jit
def issue_tile(
    st,
    t: Int32,
    kv_prod,
    atom: cute.CopyAtom,
    tg: cute.Tensor,
    ts: cute.Tensor,
    mKvIndices: cute.Tensor,
    kv_ptr0: Int32,
    n_pages: Int32,
    kv_head: Int32,
):
    """One K or V tile = ``slots_per_tile`` consecutive ring slots in key order."""
    for sl in cutlass.range_constexpr(st.slots_per_tile):
        kv_prod = issue_slot(
            st,
            t * st.T + sl * st.slot_keys,
            kv_prod,
            atom,
            tg,
            ts,
            mKvIndices,
            kv_ptr0,
            n_pages,
            kv_head,
        )
    return kv_prod


@cute.jit
def load_loop(
    st,
    atom_q: cute.CopyAtom,
    tQ: cute.Tensor,
    atom_q_rep: cute.CopyAtom,
    tQr: cute.Tensor,
    atom_k: cute.CopyAtom,
    tK: cute.Tensor,
    atom_v: cute.CopyAtom,
    tV: cute.Tensor,
    sQ_view: cute.Tensor,
    sQ_rep_view: cute.Tensor,
    sKV_view: cute.Tensor,
    q_prod,
    kv_prod,
    mKvIndptr: cute.Tensor,
    mKvIndices: cute.Tensor,
    req: Int32,
    tok_abs: Int32,
    kv_head: Int32,
    rows_valid: Int32,
    tile_lo: Int32,
    tile_hi: Int32,
):
    LA = st.qk_ahead  # K tiles issued ahead of V: 1 (the MMA issues QK(n+1), PV(n)) or 2 (PV(n), QK(n+2))
    kv_ptr0 = mKvIndptr[req]
    n_pages = mKvIndptr[req + 1] - kv_ptr0
    # Q: the tile's tokens start at the packed row tok_abs; the offset is applied to the TMA tensor's domain
    tQ_off = cute.domain_offset((0, 0, tok_abs), tQ)
    gQ = cute.zipped_divide(tQ_off, st.q_box)  # ((64, g, tpt), (halves, Hq/g, rest))
    tQsQ, tQgQ = cpasync.tma_partition(
        atom_q, 0, cute.make_layout(1), cute.group_modes(sQ_view, 0, 3), gQ
    )  # smem ((box), halves, stages)
    # the replicated load (units of <= 32 live rows): the first 32 rows into every 32-row quadrant
    tQr_off = cute.domain_offset((0, 0, tok_abs), tQr)
    gQr = cute.zipped_divide(
        tQr_off, st.q_box_rep
    )  # ((64, g, 32/g), (halves, Hq/g, rest))
    tQsQr, tQgQr = cpasync.tma_partition(
        atom_q_rep, 0, cute.make_layout(1), cute.group_modes(sQ_rep_view, 0, 3), gQr
    )  # smem ((box), quadrants, halves, stages)
    if cutlass.const_expr(st.ragged):
        # the ragged pool: shift the K / V domains by the request's token offset, so the boxes below address the request's own keys
        # (box i of the shifted domain = its keys [i box_rows, (i + 1) box_rows))
        tK = cute.domain_offset((0, kv_ptr0, 0, 0), tK)
        tV = cute.domain_offset((0, kv_ptr0, 0, 0), tV)
    gK = cute.zipped_divide(
        tK, (HALF, st.box_rows, 1, 1)
    )  # ((64, rows, 1, 1), (halves, P/rows, Hkv, pages))
    gV = cute.zipped_divide(tV, (HALF, st.box_rows, 1, 1))
    sKVg = cute.group_modes(sKV_view, 0, 4)  # ((64, rows, 1, 1), boxes, halves, slots)
    tKsK, tKgK = cpasync.tma_partition(atom_k, 0, cute.make_layout(1), sKVg, gK)
    tVsV, tVgV = cpasync.tma_partition(atom_v, 0, cute.make_layout(1), sKVg, gV)
    # the transaction count of this unit's Q load: the replicated boxes move 4 x rep_rows rows, the full-tile box rows_per_tile rows
    # (both equal 128 rows only when g divides 32); armed per acquire so that the barrier completes for every GQA group
    q_tx = Int32(st.q_bytes_full)
    if cutlass.const_expr(st.rep_ok):
        if rows_valid <= st.rep_rows:
            q_tx = Int32(st.q_bytes_rep)
    qh = q_prod.acquire_and_advance(expected_tx=q_tx)
    if cutlass.const_expr(st.rep_ok):
        if rows_valid <= st.rep_rows:
            for quad in cutlass.range_constexpr(Q_ROWS // REP_ROWS):
                for half in cutlass.range_constexpr(st.halves):
                    cute.copy(
                        atom_q_rep,
                        tQgQr[(None, (half, kv_head, 0))],
                        tQsQr[(None, quad, half, qh.index)],
                        tma_bar_ptr=qh.barrier,
                    )
        else:
            for half in cutlass.range_constexpr(st.halves):
                cute.copy(
                    atom_q,
                    tQgQ[(None, (half, kv_head, 0))],
                    tQsQ[(None, half, qh.index)],
                    tma_bar_ptr=qh.barrier,
                )
    else:
        for half in cutlass.range_constexpr(st.halves):
            cute.copy(
                atom_q,
                tQgQ[(None, (half, kv_head, 0))],
                tQsQ[(None, half, qh.index)],
                tma_bar_ptr=qh.barrier,
            )
    N = tile_hi - tile_lo
    # the ring is consumed in the MMA warp's order: K_0 [K_1] then, per tile n, (K_{n+1}, V_n) with QK one tile ahead or (V_n, K_{n+2}) with two
    kv_prod = issue_tile(
        st, tile_lo, kv_prod, atom_k, tKgK, tKsK, mKvIndices, kv_ptr0, n_pages, kv_head
    )
    if cutlass.const_expr(LA == 2):
        if N > 1:
            kv_prod = issue_tile(
                st,
                tile_lo + 1,
                kv_prod,
                atom_k,
                tKgK,
                tKsK,
                mKvIndices,
                kv_ptr0,
                n_pages,
                kv_head,
            )
        for n in range(N):
            kv_prod = issue_tile(
                st,
                tile_lo + n,
                kv_prod,
                atom_v,
                tVgV,
                tVsV,
                mKvIndices,
                kv_ptr0,
                n_pages,
                kv_head,
            )
            if n + 2 < N:
                kv_prod = issue_tile(
                    st,
                    tile_lo + n + 2,
                    kv_prod,
                    atom_k,
                    tKgK,
                    tKsK,
                    mKvIndices,
                    kv_ptr0,
                    n_pages,
                    kv_head,
                )
    else:
        for n in range(N):
            if n + 1 < N:
                kv_prod = issue_tile(
                    st,
                    tile_lo + n + 1,
                    kv_prod,
                    atom_k,
                    tKgK,
                    tKsK,
                    mKvIndices,
                    kv_ptr0,
                    n_pages,
                    kv_head,
                )
            kv_prod = issue_tile(
                st,
                tile_lo + n,
                kv_prod,
                atom_v,
                tVgV,
                tVsV,
                mKvIndices,
                kv_ptr0,
                n_pages,
                kv_head,
            )
    return q_prod, kv_prod


@cute.jit
def load_q_tile(
    st,
    atom_q: cute.CopyAtom,
    tQ: cute.Tensor,
    atom_q_rep: cute.CopyAtom,
    tQr: cute.Tensor,
    sQ_view: cute.Tensor,
    sQ_rep_view: cute.Tensor,
    qh,
    q_tile: Int32,
    tok_abs: Int32,
    kv_head: Int32,
    rows_valid: Int32,
):
    """One Q tile into smem tile ``q_tile`` on the stage barrier of ``qh``: the full-tile boxes, or the replicated boxes for a unit of at
    most ``rep_rows`` live rows (``load_loop``'s two patterns)."""
    tQ_off = cute.domain_offset((0, 0, tok_abs), tQ)
    gQ = cute.zipped_divide(tQ_off, st.q_box)
    tQsQ, tQgQ = cpasync.tma_partition(
        atom_q, 0, cute.make_layout(1), cute.group_modes(sQ_view, 0, 3), gQ
    )
    tQr_off = cute.domain_offset((0, 0, tok_abs), tQr)
    gQr = cute.zipped_divide(tQr_off, st.q_box_rep)
    tQsQr, tQgQr = cpasync.tma_partition(
        atom_q_rep, 0, cute.make_layout(1), cute.group_modes(sQ_rep_view, 0, 3), gQr
    )
    if cutlass.const_expr(st.rep_ok):
        if rows_valid <= st.rep_rows:
            for quad in cutlass.range_constexpr(Q_ROWS // REP_ROWS):
                for half in cutlass.range_constexpr(st.halves):
                    cute.copy(
                        atom_q_rep,
                        tQgQr[(None, (half, kv_head, 0))],
                        tQsQr[(None, quad, half, q_tile)],
                        tma_bar_ptr=qh.barrier,
                    )
        else:
            for half in cutlass.range_constexpr(st.halves):
                cute.copy(
                    atom_q,
                    tQgQ[(None, (half, kv_head, 0))],
                    tQsQ[(None, half, q_tile)],
                    tma_bar_ptr=qh.barrier,
                )
    else:
        for half in cutlass.range_constexpr(st.halves):
            cute.copy(
                atom_q,
                tQgQ[(None, (half, kv_head, 0))],
                tQsQ[(None, half, q_tile)],
                tma_bar_ptr=qh.barrier,
            )


@cute.jit
def q_tx_bytes(st, rows_valid: Int32) -> Int32:
    q_tx = Int32(st.q_bytes_full)
    if cutlass.const_expr(st.rep_ok):
        if rows_valid <= st.rep_rows:
            q_tx = Int32(st.q_bytes_rep)
    return q_tx


@cute.jit
def load_loop_q2(
    st,
    atom_q: cute.CopyAtom,
    tQ: cute.Tensor,
    atom_q_rep: cute.CopyAtom,
    tQr: cute.Tensor,
    atom_k: cute.CopyAtom,
    tK: cute.Tensor,
    atom_v: cute.CopyAtom,
    tV: cute.Tensor,
    sQ_view: cute.Tensor,
    sQ_rep_view: cute.Tensor,
    sKV_view: cute.Tensor,
    q_prod,
    kv_prod,
    mKvIndptr: cute.Tensor,
    mKvIndices: cute.Tensor,
    req: Int32,
    kv_head: Int32,
    tok_a: Int32,
    rows_a: Int32,
    lo_a: Int32,
    hi_a: Int32,
    tok_b: Int32,
    rows_b: Int32,
    lo_b: Int32,
    hi_b: Int32,
):
    """The paired placement's load warp: Q tiles A and B of the unit into tiles ``2 s``, ``2 s + 1`` of one Q stage (one barrier, armed
    with the bytes of the tiles present; B is absent when its tile range is empty), then K and V of the union of the two tile ranges in
    the MMA warp's order ``K_lo``, then per tile ``t``: ``K_{t+1}``, ``V_t`` (each K / V tile is consumed by both Q tiles' MMAs)."""
    kv_ptr0 = mKvIndptr[req]
    n_pages = mKvIndptr[req + 1] - kv_ptr0
    has_b = hi_b > lo_b
    q_tx = q_tx_bytes(st, rows_a)
    if has_b:
        q_tx = q_tx + q_tx_bytes(st, rows_b)
    qh = q_prod.acquire_and_advance(expected_tx=q_tx)
    load_q_tile(
        st,
        atom_q,
        tQ,
        atom_q_rep,
        tQr,
        sQ_view,
        sQ_rep_view,
        qh,
        2 * qh.index,
        tok_a,
        kv_head,
        rows_a,
    )
    if has_b:
        load_q_tile(
            st,
            atom_q,
            tQ,
            atom_q_rep,
            tQr,
            sQ_view,
            sQ_rep_view,
            qh,
            2 * qh.index + 1,
            tok_b,
            kv_head,
            rows_b,
        )
    lo = lo_a
    hi = hi_a
    if has_b:
        if lo_b < lo:
            lo = lo_b
        if hi_b > hi:
            hi = hi_b
    if cutlass.const_expr(st.ragged):
        # the ragged pool: shift the K / V domains by the request's token offset, so the boxes below address the request's own keys
        # (box i of the shifted domain = its keys [i box_rows, (i + 1) box_rows))
        tK = cute.domain_offset((0, kv_ptr0, 0, 0), tK)
        tV = cute.domain_offset((0, kv_ptr0, 0, 0), tV)
    gK = cute.zipped_divide(tK, (HALF, st.box_rows, 1, 1))
    gV = cute.zipped_divide(tV, (HALF, st.box_rows, 1, 1))
    sKVg = cute.group_modes(sKV_view, 0, 4)
    tKsK, tKgK = cpasync.tma_partition(atom_k, 0, cute.make_layout(1), sKVg, gK)
    tVsV, tVgV = cpasync.tma_partition(atom_v, 0, cute.make_layout(1), sKVg, gV)
    if hi > lo:
        kv_prod = issue_tile(
            st, lo, kv_prod, atom_k, tKgK, tKsK, mKvIndices, kv_ptr0, n_pages, kv_head
        )
        for t in range(lo, hi):
            if t + 1 < hi:
                kv_prod = issue_tile(
                    st,
                    t + 1,
                    kv_prod,
                    atom_k,
                    tKgK,
                    tKsK,
                    mKvIndices,
                    kv_ptr0,
                    n_pages,
                    kv_head,
                )
            kv_prod = issue_tile(
                st,
                t,
                kv_prod,
                atom_v,
                tVgV,
                tVsV,
                mKvIndices,
                kv_ptr0,
                n_pages,
                kv_head,
            )
    return q_prod, kv_prod
