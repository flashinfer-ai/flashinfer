# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Split-KV fixup: fold the chunk partials of a row in chunk order with the spec's ``merge`` from the
init state ``(-inf, +0, +0)`` and apply the epilogue (``cvt_bf16(add(mul(O, div(1, l)), +0))``): the
``chunk`` fold digit of the class, ONE implementation (``fold_row``) executed either by the fixup kernel
(one thread per (row, 4 output columns), a second graph node) or in-kernel by the correction warpgroup of
the unit that completes a group (the "last arriver": ``blocks/correction.py``, a placement chosen by the
planner per group, ``plan.py``). The present chunks of a row are recomputed from its position (a function
of fixed dims and the row only), so absent chunks (sliding window) are never merged. All math through
``CuteF32``; one kernel per D.

Placement: the partials of ``PF`` consecutive chunks are loaded before their merges (the loads of
chunk ``j + 1`` do not depend on the merge of chunk ``j``, so ``PF`` L2 round trips overlap instead of
one per chunk: with ``C = 512`` a 32K row has 64 chunks and the one-load-per-merge chain was longer than
the attention kernel itself at B = 1). The fold order and every operation are unchanged.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from ...batch_invariant import jit
from ...batch_invariant.fp_interface.cute import CuteF32
from .plan import (
    G_CHUNK_LO,
    G_KVHEAD,
    G_KVLEN,
    G_NCHUNKS,
    G_QBASE,
    G_QLEN,
    G_QTOK0,
    G_ROWS,
    G_SLOT,
    GROUP_FIELDS,
    ws_row,
)
from .spec import chunk_len

VEC = 4
THREADS = 128
PF = 32  # chunks whose partials are loaded ahead of their merges in the fixup KERNEL (a placement: the merges stay in chunk order;
# 32: two load batches for a 64-chunk row -- the fixup node was a sizeable share of B = 1 decode at D = 256;
# 192 registers of partials at 128 threads per block; the in-kernel fold uses 8 / 4 at the correction warpgroup's budget, blocks/correction.py)
PDL = True  # the fixup kernel is launched with programmatic dependent launch: its blocks are scheduled while the attention kernel's
# last CTAs drain (every working role of the attention kernel issues griddepcontrol.launch_dependents AFTER its unit
# walk -- issued at the kernel start, the resident blocks slowed every multi-wave fold) and griddepcontrol.wait
# before the first workspace read orders every partial row behind the whole attention grid (a placement: the second graph
# node's launch gap overlaps the drain, measured faster at small batches)
_NEG_INF = float("-inf")


@cute.jit
def fold_row_state(
    mWs: cute.Tensor,
    slot: Int32,
    rows: Int32,
    chunk_lo: Int32,
    n_chunks: Int32,
    pos: Int32,
    c: Float32,
    nc: Float32,
    window_left: Int32,
    r: Int32,
    d0: Int32,
    o: cute.Tensor,
    D: cutlass.Constexpr,
    C: cutlass.Constexpr,
    PF: cutlass.Constexpr,
    WS: cutlass.Constexpr,
    layout: cutlass.Constexpr = "row",
    lane_rows: cutlass.Constexpr = 0,
    cells: cutlass.Constexpr = None,
    grid: cutlass.Constexpr = None,
):
    """The ``chunk`` fold digit for row ``r`` (query position ``pos``) of a split group over output columns ``[d0, d0 + VEC)``: the
    row's present chunks folded in chunk order with ``spec.merge`` from the init state ``(-inf, +0, +0)`` into ``o`` (``VEC`` fp32,
    caller-allocated) and the returned ``(m, l)``. ``PF`` partial rows are loaded ahead of their merges; ``WS`` is the workspace row
    pitch (``plan.ws_row(D)`` for the attention classes: 68 / 132 / 260 at D = 64 / 128 / 256; ``D + 4`` for MLA's 512-wide rows). One implementation for every caller: the fixup
    kernels of both classes and the attention body's in-kernel last-arriver fold.

    ``layout`` (a placement of the private partials, no arithmetic): ``"row"`` -- partial row ``r`` of chunk ``j`` is the ``WS`` fp32 at
    ``(slot + (j - chunk_lo) * rows + r) * WS`` (``O[0..D)``, ``m``, ``l``); ``"lane"`` -- the chunk's region of ``lane_rows`` rows (a static
    count >= ``rows``) is lane-major: region ``(slot + (j - chunk_lo) * lane_rows) * WS``, element ``d``
    of row ``r`` at ``region + ((d // 4) * lane_rows + r) * 4 + d % 4`` with ``m, l`` at ``d = D, D + 1`` (``d0`` a multiple of 4, ``VEC`` = 4).

    ``cells`` / ``grid`` (the MLA class; additive: the attention classes pass neither and keep the fixed ``C``
    arithmetic below, bit for bit): a device function ``cells(grid, lo, L) -> (j_lo, j_hi)`` giving the first and one-past-last
    chunk (cell) of the row's visible range on the class's absolute chunk grid ``grid``;
    the regions stay one per chunk index, so the fold below is unchanged.
    """
    L = pos + 1
    lo = Int32(0)
    if window_left >= 0:
        lo = pos - window_left
        if lo < 0:
            lo = Int32(0)
    if cutlass.const_expr(cells is None):
        j_lo = lo // C
        j_hi = (L + C - 1) // C
    else:
        j_lo, j_hi = cells(grid, lo, L)
    if j_lo < chunk_lo:
        j_lo = chunk_lo
    if j_hi > chunk_lo + n_chunks:
        j_hi = chunk_lo + n_chunks
    m = Float32(_NEG_INF)
    l = Float32(0.0)
    o.fill(0.0)
    # the regions to fold, in order: one per present chunk (region r = chunk chunk_lo + r)
    r_lo = j_lo - chunk_lo
    r_hi = j_hi - chunk_lo
    # PF partial rows (m, l, O[d0 .. d0 + VEC)) per step, loaded before the step's merges; a region past r_hi is
    # loaded from the last present region's row (a valid address) and skipped
    pml = cute.make_rmem_tensor(
        cute.make_layout((PF, 2), stride=(1, PF)), Float32
    )  # [i] -> m at i, l at PF + i (an (m, l) pair is one 8-byte load)
    pm = cute.make_tensor(pml.iterator, cute.make_layout((PF,)))
    pl = cute.make_tensor(pml.iterator + PF, cute.make_layout((PF,)))
    po = cute.make_rmem_tensor(
        cute.make_layout((PF, VEC), stride=(VEC, 1)), Float32
    )  # row-major: chunk i's VEC values are contiguous (the 16-byte load)
    for r0 in range(r_lo, r_hi, PF):
        for i in cutlass.range_constexpr(PF):
            rg = r0 + i
            if rg > r_hi - 1:
                rg = r_hi - 1
            if cutlass.const_expr(layout == "row"):
                base = (slot + rg * rows + r).to(
                    Int64
                ) * WS  # 64-bit: the workspace may exceed 2^31 fp32
                pm[i] = mWs[base + D]
                pl[i] = mWs[base + D + 1]
                for e in cutlass.range_constexpr(VEC):
                    po[(i, e)] = mWs[base + d0 + e]
            else:
                # the chunk's lane-major region (64-bit); (m, l) as one 8-byte load, the O granule as one 16-byte load: with six scalar loads per
                # chunk the compiler kept only part of a batch's loads in flight and serialised the rest
                region = (slot + rg * lane_rows).to(Int64) * WS
                ml = cute.assume(region + ((D // VEC) * lane_rows + r) * VEC, divby=4)
                og = cute.assume(region + ((d0 // VEC) * lane_rows + r) * VEC, divby=4)
                cute.autovec_copy(
                    cute.make_tensor(mWs.iterator + ml, cute.make_layout((2,))),
                    cute.make_tensor(
                        pm.iterator + i, cute.make_layout((2,), stride=(PF,))
                    ),
                )
                cute.autovec_copy(
                    cute.make_tensor(mWs.iterator + og, cute.make_layout((VEC,))),
                    cute.make_tensor(po.iterator + i * VEC, cute.make_layout((VEC,))),
                )  # po is (PF, VEC) with VEC contiguous
        for i in cutlass.range_constexpr(PF):
            if r0 + i < r_hi:
                m2 = pm[i]
                l2 = pl[i]
                mm = CuteF32.fmaxc(m, m2)
                nmm = CuteF32.mul(mm, nc)
                a1 = CuteF32.ex2(CuteF32.add(CuteF32.mul(m, c), nmm))
                a2 = CuteF32.ex2(CuteF32.add(CuteF32.mul(m2, c), nmm))
                l = CuteF32.fma(l2, a2, CuteF32.mul(l, a1))
                for e in cutlass.range_constexpr(VEC):
                    o[e] = CuteF32.fma(po[(i, e)], a2, CuteF32.mul(o[e], a1))
                m = mm
    return m, l


@cute.jit
def fold_row(
    mWs: cute.Tensor,
    mO: cute.Tensor,
    slot: Int32,
    rows: Int32,
    chunk_lo: Int32,
    n_chunks: Int32,
    q_tok0: Int32,
    kv_head: Int32,
    kv_len: Int32,
    q_len: Int32,
    q_base: Int32,
    c: Float32,
    nc: Float32,
    window_left: Int32,
    Hq: Int32,
    r: Int32,
    d0: Int32,
    D: cutlass.Constexpr,
    g: cutlass.Constexpr,
    C: cutlass.Constexpr,
    PF: cutlass.Constexpr,
    WS: cutlass.Constexpr,
):
    """Row ``r`` of a split group, output columns ``[d0, d0 + VEC)``: the chunk-order fold of the row's present chunks from the
    init state (``fold_row_state``, the single implementation), then the epilogue to the output row. ``WS`` is the workspace row
    pitch of the head dim (``plan.ws_row(D)``: 68 / 132 / 260 at D = 64 / 128 / 256)."""
    token = q_tok0 + r // g
    pos = kv_len - q_len + token
    o = cute.make_rmem_tensor(cute.make_layout((VEC,)), Float32)
    m, l = fold_row_state(
        mWs,
        slot,
        rows,
        chunk_lo,
        n_chunks,
        pos,
        c,
        nc,
        window_left,
        r,
        d0,
        o,
        D,
        C,
        PF,
        WS,
    )
    inv = CuteF32.div(Float32(1.0), l)
    out_row = (q_base + token) * Hq + kv_head * g + r % g
    for e in cutlass.range_constexpr(VEC):
        mO[(out_row.to(Int64), d0 + e)] = CuteF32.add(
            CuteF32.mul(o[e], inv), Float32(0.0)
        ).to(mO.element_type)


WIDE_PF = 32  # the fixup kernel's batch of chunks loaded before their merges (the wide thread mapping below)


def wide_vec(D: int) -> int:
    """Output columns per thread in the fixup KERNEL: one row spread over up to 128 threads (1 / 1 / 2 columns at D = 64 / 128 / 256)."""
    return max(1, D // 128)


@cute.jit
def fold_row_state_wide(
    mWs: cute.Tensor,
    slot: Int32,
    rows: Int32,
    chunk_lo: Int32,
    n_chunks: Int32,
    pos: Int32,
    c: Float32,
    nc: Float32,
    window_left: Int32,
    r: Int32,
    d0: Int32,
    o: cute.Tensor,
    D: cutlass.Constexpr,
    C: cutlass.Constexpr,
    PF: cutlass.Constexpr,
    WS: cutlass.Constexpr,
    V: cutlass.Constexpr,
):
    """``fold_row_state`` for the fixup kernel's wide thread mapping (``V`` output columns per thread, ``"row"`` workspace layout): the
    same present chunks, the same fold order and the same operations on the same operands, scheduled for latency: a batch's partials
    are all loaded first, then the running-maximum chain and the batch's ``a1`` / ``a2`` exponentials (independent of the ``O`` chain),
    then the ``l`` and ``O`` chains. Bit for bit ``fold_row_state``: only the instruction schedule differs."""
    L = pos + 1
    lo = Int32(0)
    if window_left >= 0:
        lo = pos - window_left
        if lo < 0:
            lo = Int32(0)
    j_lo = lo // C
    j_hi = (L + C - 1) // C
    if j_lo < chunk_lo:
        j_lo = chunk_lo
    if j_hi > chunk_lo + n_chunks:
        j_hi = chunk_lo + n_chunks
    m = Float32(_NEG_INF)
    l = Float32(0.0)
    o.fill(0.0)
    r_lo = j_lo - chunk_lo
    r_hi = j_hi - chunk_lo
    pm = cute.make_rmem_tensor(cute.make_layout((PF,)), Float32)
    pl = cute.make_rmem_tensor(cute.make_layout((PF,)), Float32)
    pa1 = cute.make_rmem_tensor(cute.make_layout((PF,)), Float32)
    pa2 = cute.make_rmem_tensor(cute.make_layout((PF,)), Float32)
    po = cute.make_rmem_tensor(cute.make_layout((PF, V), stride=(V, 1)), Float32)
    for r0 in range(r_lo, r_hi, PF):
        for i in cutlass.range_constexpr(PF):
            rg = r0 + i
            if rg > r_hi - 1:
                rg = r_hi - 1
            base = (slot + rg * rows + r).to(Int64) * WS
            pm[i] = mWs[base + D]
            pl[i] = mWs[base + D + 1]
            for e in cutlass.range_constexpr(V):
                po[(i, e)] = mWs[base + d0 + e]
        for i in cutlass.range_constexpr(PF):
            if r0 + i < r_hi:
                m2 = pm[i]
                mm = CuteF32.fmaxc(m, m2)
                nmm = CuteF32.mul(mm, nc)
                pa1[i] = CuteF32.ex2(CuteF32.add(CuteF32.mul(m, c), nmm))
                pa2[i] = CuteF32.ex2(CuteF32.add(CuteF32.mul(m2, c), nmm))
                m = mm
        for i in cutlass.range_constexpr(PF):
            if r0 + i < r_hi:
                l = CuteF32.fma(pl[i], pa2[i], CuteF32.mul(l, pa1[i]))
                for e in cutlass.range_constexpr(V):
                    o[e] = CuteF32.fma(po[(i, e)], pa2[i], CuteF32.mul(o[e], pa1[i]))
    return m, l


COOP_NCB = (
    64  # chunks per batch of the cooperative fold (two per lane of the scan warp)
)


@cute.jit
def _warp_inclusive_max(v: Float32, lane: Int32) -> Float32:
    """Inclusive prefix ``fmaxc`` over the warp's lanes in lane order (Hillis-Steele with ``shfl.up``). ``max.f32`` returns one of its
    operands, so the prefix of any order equals the sequential running maximum (non-NaN operands; a zero's sign may differ, which the
    merge's ``a1`` / ``a2`` cannot see: ``add(+-0, x)`` and ``ex2(+-0)`` are sign-blind there)."""
    for off in cutlass.range_constexpr(5):
        t = cute.arch.shuffle_sync_up(v, 1 << off)
        if lane >= (1 << off):
            v = CuteF32.fmaxc(t, v)
    return v


@cute.jit
def fold_row_coop(
    mWs: cute.Tensor,
    sA1: cute.Tensor,
    sA2: cute.Tensor,
    sL: cute.Tensor,
    slot: Int32,
    rows: Int32,
    chunk_lo: Int32,
    n_chunks: Int32,
    pos: Int32,
    c: Float32,
    nc: Float32,
    window_left: Int32,
    r: Int32,
    d0: Int32,
    o: cute.Tensor,
    is_scan_warp,
    lane: Int32,
    D: cutlass.Constexpr,
    C: cutlass.Constexpr,
    WS: cutlass.Constexpr,
    V: cutlass.Constexpr,
):
    """The fixup kernel's cooperative fold of row ``r`` (every thread of the row's thread group calls it; ``is_scan_warp`` marks the
    row's first warp). Per batch of up to ``COOP_NCB`` present chunks: every thread loads its ``V`` columns of every chunk's partial;
    the scan warp loads the chunks' ``(m, l)``, forms the running maxima (``_warp_inclusive_max``) and every chunk's
    ``a1 = ex2(add(mul(m, c), mul(mm, nc)))`` / ``a2 = ex2(add(mul(m2, c), mul(mm, nc)))`` in parallel into shared memory; then every
    thread runs the chunk-order ``l`` and ``O`` chains ``fma(x2, a2, mul(x, a1))``. The operations and their order per output are
    exactly ``fold_row_state``'s; only who computes the scalars, and when, differs. Returns ``l`` (the running ``m`` is not an output)."""
    L = pos + 1
    lo = Int32(0)
    if window_left >= 0:
        lo = pos - window_left
        if lo < 0:
            lo = Int32(0)
    j_lo = lo // C
    j_hi = (L + C - 1) // C
    if j_lo < chunk_lo:
        j_lo = chunk_lo
    if j_hi > chunk_lo + n_chunks:
        j_hi = chunk_lo + n_chunks
    r_lo = j_lo - chunk_lo
    r_hi = j_hi - chunk_lo
    l = Float32(0.0)
    o.fill(0.0)
    m_run = Float32(_NEG_INF)
    po = cute.make_rmem_tensor(cute.make_layout((COOP_NCB, V), stride=(V, 1)), Float32)
    # batches over the GROUP's chunk range (uniform for the block: the barriers below are reached by every thread), each row
    # folding only its present chunks [r_lo, r_hi)
    for r0 in range(0, n_chunks, COOP_NCB):
        for i in cutlass.range_constexpr(COOP_NCB):
            rg = r0 + i
            if rg > r_hi - 1:
                rg = r_hi - 1
            if rg < r_lo:
                rg = r_lo
            base = (slot + rg * rows + r).to(Int64) * WS
            for e in cutlass.range_constexpr(V):
                po[(i, e)] = mWs[base + d0 + e]
        if is_scan_warp:
            ka = r0 + lane
            kb = r0 + 32 + lane
            ca = ka
            if ca > r_hi - 1:
                ca = r_hi - 1
            if ca < r_lo:
                ca = r_lo
            cb = kb
            if cb > r_hi - 1:
                cb = r_hi - 1
            if cb < r_lo:
                cb = r_lo
            base_a = (slot + ca * rows + r).to(Int64) * WS
            base_b = (slot + cb * rows + r).to(Int64) * WS
            ma = mWs[base_a + D]
            la = mWs[base_a + D + 1]
            mb = mWs[base_b + D]
            lb = mWs[base_b + D + 1]
            if ka > r_hi - 1 or ka < r_lo:
                ma = Float32(_NEG_INF)
            if kb > r_hi - 1 or kb < r_lo:
                mb = Float32(_NEG_INF)
            inc_a = CuteF32.fmaxc(m_run, _warp_inclusive_max(ma, lane))
            exc_a = cute.arch.shuffle_sync_up(inc_a, 1)
            if lane == 0:
                exc_a = m_run
            tot_a = cute.arch.shuffle_sync(inc_a, 31)
            inc_b = CuteF32.fmaxc(tot_a, _warp_inclusive_max(mb, lane))
            exc_b = cute.arch.shuffle_sync_up(inc_b, 1)
            if lane == 0:
                exc_b = tot_a
            m_run = cute.arch.shuffle_sync(inc_b, 31)
            na = CuteF32.mul(inc_a, nc)
            nb = CuteF32.mul(inc_b, nc)
            sA1[lane] = CuteF32.ex2(CuteF32.add(CuteF32.mul(exc_a, c), na))
            sA2[lane] = CuteF32.ex2(CuteF32.add(CuteF32.mul(ma, c), na))
            sL[lane] = la
            sA1[32 + lane] = CuteF32.ex2(CuteF32.add(CuteF32.mul(exc_b, c), nb))
            sA2[32 + lane] = CuteF32.ex2(CuteF32.add(CuteF32.mul(mb, c), nb))
            sL[32 + lane] = lb
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(COOP_NCB):
            if r0 + i < r_hi and r0 + i >= r_lo:
                a1 = sA1[i]
                a2 = sA2[i]
                l = CuteF32.fma(sL[i], a2, CuteF32.mul(l, a1))
                for e in cutlass.range_constexpr(V):
                    o[e] = CuteF32.fma(po[(i, e)], a2, CuteF32.mul(o[e], a1))
        cute.arch.sync_threads()
    return l


class SplitKVFixup:
    def __init__(self, D: int, g: int, pdl: bool = PDL):
        self.D, self.g, self.pdl = D, g, pdl
        self.C = chunk_len(D)
        self.WS = ws_row(D)

    @cute.jit
    def __call__(
        self,
        mWs: cute.Tensor,
        mO: cute.Tensor,
        mGroups: cute.Tensor,
        c: Float32,
        nc: Float32,
        window_left: Int32,
        Hq: Int32,
        num_groups: Int32,
        grid_y: Int32,
        max_rows: Int32,
        stream,
    ):
        # grid: (row blocks of a group, min(groups, MAX_GRID_Y)); a block handles 128 / (D / V) rows, V = wide_vec(D) output columns
        # per thread, and strides over the groups beyond grid.y
        rows_per_block = THREADS // (self.D // wide_vec(self.D))
        self.kernel(mWs, mO, mGroups, c, nc, window_left, Hq, num_groups).launch(
            grid=[cute.ceil_div(max_rows, rows_per_block), grid_y, 1],
            block=[THREADS, 1, 1],
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mWs: cute.Tensor,
        mO: cute.Tensor,
        mGroups: cute.Tensor,
        c: Float32,
        nc: Float32,
        window_left: Int32,
        Hq: Int32,
        num_groups: Int32,
    ):
        D, g, C = self.D, self.g, self.C
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()  # every partial row of the attention grid is complete and visible before any read
        tidx, _, _ = cute.arch.thread_idx()
        bx, grp0, _ = cute.arch.block_idx()
        _, grid_y, _ = cute.arch.grid_dim()
        V = wide_vec(D)
        thr_per_row = D // V
        rows_per_block = THREADS // thr_per_row
        ri = tidx // thr_per_row
        r = bx * rows_per_block + ri
        d0 = (tidx % thr_per_row) * V
        lane = tidx % 32
        is_scan_warp = (tidx % thr_per_row) < 32
        smem = cutlass.utils.SmemAllocator()
        sbuf = smem.allocate_tensor(
            Float32, cute.make_layout((3, rows_per_block, COOP_NCB)), byte_alignment=16
        )
        sA1 = sbuf[0, ri, None]
        sA2 = sbuf[1, ri, None]
        sL = sbuf[2, ri, None]
        for grp in cutlass.range(grp0, num_groups, grid_y):
            rows = mGroups[(grp, G_ROWS)]
            # every thread of the block reaches the fold's barriers: a row past the group's rows folds a clamped (valid) row and stores nothing
            live = r < rows
            rr = r
            if rr > rows - 1:
                rr = rows - 1
            q_tok0 = mGroups[(grp, G_QTOK0)]
            kv_len = mGroups[(grp, G_KVLEN)]
            q_len = mGroups[(grp, G_QLEN)]
            token = q_tok0 + rr // g
            pos = kv_len - q_len + token
            o = cute.make_rmem_tensor(cute.make_layout((V,)), Float32)
            l = fold_row_coop(
                mWs,
                sA1,
                sA2,
                sL,
                mGroups[(grp, G_SLOT)],
                rows,
                mGroups[(grp, G_CHUNK_LO)],
                mGroups[(grp, G_NCHUNKS)],
                pos,
                c,
                nc,
                window_left,
                rr,
                d0,
                o,
                is_scan_warp,
                lane,
                D,
                C,
                self.WS,
                V,
            )
            if live:
                inv = CuteF32.div(Float32(1.0), l)
                out_row = (
                    (mGroups[(grp, G_QBASE)] + token) * Hq
                    + mGroups[(grp, G_KVHEAD)] * g
                    + rr % g
                )
                for e in cutlass.range_constexpr(V):
                    mO[(out_row.to(Int64), d0 + e)] = CuteF32.add(
                        CuteF32.mul(o[e], inv), Float32(0.0)
                    ).to(mO.element_type)
            cute.arch.sync_threads()  # the block's smem rows are reused by the next group


MAX_GRID_Y = (
    65535  # the hardware's grid.y limit; the fixup strides over the groups beyond it
)


@functools.cache
def get_fixup(D: int, g: int):
    name = jit.sanitize_symbol_name(f"attn_fixup_d{D}_g{g}_pdl{int(PDL)}")

    def compile_fn():
        f32, i32, bf16 = cutlass.Float32, cutlass.Int32, cutlass.BFloat16
        ws_fake = cute.runtime.make_fake_tensor(
            f32, (cute.sym_int(64),), (1,), assumed_align=16
        )  # > 2^31 fp32 of partials is legal
        o_fake = cute.runtime.make_fake_tensor(
            bf16,
            (cute.sym_int(32), D),
            (cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        groups_fake = cute.runtime.make_fake_tensor(
            i32, (cute.sym_int(32), GROUP_FIELDS), (GROUP_FIELDS, 1), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            SplitKVFixup(D, g),
            ws_fake,
            o_fake,
            groups_fake,
            Float32(1.0),
            Float32(-1.0),
            Int32(-1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load("batch_invariant_attention", name, compile_fn)
