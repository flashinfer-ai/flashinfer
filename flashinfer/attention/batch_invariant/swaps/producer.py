# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""TMA producer block of the Swaps-AB body (the load warp): the unit's Q rows once per unit (one box of ``(64 d, g heads, tokens_box
tokens)`` per 64-wide half of D into the ``n_pad x D`` B-operand stage; two stages, so the next unit's Q is prefetched), then the page boxes
of K and of V of every tile into the ring in the MMA warp's order (K one tile ahead of V: ``K_0, K_1, V_0, K_2, V_1, ...``). At D = 256 a tile
is two slots (dims 0-127, 128-255), each ``boxes_per_slot x 2`` boxes. The order carries no arithmetic; absent pages (past the request's
page count) load a clamped valid page whose rows the MMA warp zeroes / the softmax masks.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.nvgpu import cpasync

from ..blocks.operands import HALF


@cute.jit
def issue_slot(
    st,
    key0: Int32,
    sl: cutlass.Constexpr,
    kv_prod,
    atom: cute.CopyAtom,
    tg: cute.Tensor,
    ts: cute.Tensor,
    mKvIndices: cute.Tensor,
    kv_ptr0: Int32,
    n_pages: Int32,
    kv_head: Int32,
):
    """One ring slot: 128 keys from absolute key ``key0``, dims ``[sl slot_dims, (sl + 1) slot_dims)``: acquire it, issue
    ``boxes_per_slot x halves`` TMA boxes (a box is ``box_rows = min(page, 128)`` keys of one page, one 64-wide half).
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
                    tg[(None, (sl * st.halves + half, k0 // st.box_rows, kv_head, 0))],
                    ts[(None, i, half, h.index)],
                    tma_bar_ptr=h.barrier,
                )
        else:
            pslot = k0 // page
            if cutlass.const_expr(page <= st.T):
                box_in_page = Int32(0)
            else:
                box_in_page = (k0 % page) // st.box_rows
            if pslot > n_pages - 1:
                pslot = n_pages - 1
            page_id = mKvIndices[kv_ptr0 + pslot]
            for half in cutlass.range_constexpr(st.halves):
                cute.copy(
                    atom,
                    tg[(None, (sl * st.halves + half, box_in_page, kv_head, page_id))],
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
    """One K or V tile = ``slots_per_tile`` consecutive ring slots (the dims slices of the same 128 keys)."""
    for sl in cutlass.range_constexpr(st.slots_per_tile):
        kv_prod = issue_slot(
            st,
            t * st.T,
            sl,
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
    atom_k: cute.CopyAtom,
    tK: cute.Tensor,
    atom_v: cute.CopyAtom,
    tV: cute.Tensor,
    sQ_view: cute.Tensor,
    sKV_view: cute.Tensor,
    q_prod,
    kv_prod,
    mKvIndptr: cute.Tensor,
    mKvIndices: cute.Tensor,
    req: Int32,
    tok_abs: Int32,
    kv_head: Int32,
    tile_lo: Int32,
    tile_hi: Int32,
):
    kv_ptr0 = mKvIndptr[req]
    n_pages = mKvIndptr[req + 1] - kv_ptr0
    # Q: the unit's tokens start at the packed row tok_abs (tokens past the request are padding rows, never stored; past total_q the TMA zero-fills)
    tQ_off = cute.domain_offset((0, 0, tok_abs), tQ)
    gQ = cute.zipped_divide(
        tQ_off, st.q_box
    )  # ((64, g, tokens_box), (halves, Hq/g, rest))
    tQsQ, tQgQ = cpasync.tma_partition(
        atom_q, 0, cute.make_layout(1), cute.group_modes(sQ_view, 0, 3), gQ
    )  # smem ((box), halves, stages)
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
    qh = q_prod.acquire_and_advance(expected_tx=Int32(st.q_tx_bytes))
    for half in cutlass.range_constexpr(st.halves_tile):
        cute.copy(
            atom_q,
            tQgQ[(None, (half, kv_head, 0))],
            tQsQ[(None, half, qh.index)],
            tma_bar_ptr=qh.barrier,
        )
    N = tile_hi - tile_lo
    kv_prod = issue_tile(
        st, tile_lo, kv_prod, atom_k, tKgK, tKsK, mKvIndices, kv_ptr0, n_pages, kv_head
    )
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
