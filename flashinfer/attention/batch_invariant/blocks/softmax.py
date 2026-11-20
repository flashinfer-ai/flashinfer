# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The softmax block: the spec's tile step on the engine's plans, executed by ``n_wg`` softmax warpgroups
that all work on every tile and split its **keys**: warpgroup ``k`` holds keys ``[k T/n_wg, (k+1) T/n_wg)``
of every row (the warps of the warpgroups share their TMEM lanes, so a row's keys are dealt out to
``n_wg`` threads). Per tile a thread

* copies the **whole** row of ``S`` out of TMEM for the ``rowmax`` plan (the canonical 128-leaf pairwise
  tree, recomputed by every slice: a redundant tree costs less than an exchange) and its own slice of
  the row for the exponentials, masks both by absolute key (``select``; only on boundary tiles),
* forms ``m' = fmax(m, m_t)``, ``alpha`` exactly as ``spec.alpha_term`` and hands ``alpha`` to the correction
  warpgroup at once (its O rescale overlaps the exponentials),
* forms the slice's probabilities ``p_k = EX2(fma(s_k, c, mul(m', nc)))``, their bf16 words and the ``rowsum``
  plan restricted to the slice -- the canonical tree's subtree over the slice's keys -- as one stream in key
  order (``streamed_tile_probs``), and hands ``(m', r_slice)`` to the correction warpgroup, which owns the
  row's ``l`` chain, the chunk merges and the normaliser (``correction.py``: the top level of the
  ``rowsum`` tree is its ``add`` of the slices' partials, in slice order),
* stores its own ``P`` columns to TMEM, in key order.

A tile with no visible key for the row is a *null step* (``alpha = 1``, ``P`` row ``+0``, ``m`` untouched). Padding
rows (beyond the unit's live rows) are neither masked nor stored: whatever they compute stays in their
own TMEM lanes and payload slots.
Units of at most 32 live rows (every decode unit) take the **replicated-row** placement: the Q tile was
loaded with its first 32 rows replicated into all four 32-row quadrants (``producer.py``), the logical
row is ``lane``, warp ``wq`` of warpgroup ``k`` takes key slice ``4k + wq`` of ``4 n_wg`` (so a decode tile's
exponentials spread over every warp), the warpgroup's four slices of ``P`` are gathered in its warp 0
through smem (a warpgroup-local named barrier; with two P stages a second one after warp 0's store keeps
warps 1..3 from overwriting the exchange buffer with the next tile's slices) before the TMEM store, and
the correction reads rows 0..31. Padding rows compute values that are never stored (no predicated regions).
With the two-stage S / P plan (``operands.tmem_plan_for``) tile ``n`` reads ``S`` from and writes ``P`` into stage
``n % 2`` (a dynamic TMEM column offset; ``P`` over the first ``T/2`` columns of its own ``S`` stage).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

from ....batch_invariant import canon
from ....batch_invariant.fp_interface.cute import CuteF32, fmax3


from .. import spec
from .operands import ACC_DTYPE, BAR_SOFTMAX_ALL, BAR_XCHG, IN_DTYPE, Q_ROWS

_NEG_INF = float("-inf")
MASK_SELECT_MAX_D = 128  # the key mask as per-key selects (``masked_score``) serves the one-stage TMEM plan; the two-stage body at D = 256
# keeps the per-key compare (the select form spills past its launch budget)


@dsl_user_op
def masked_score(
    s: Float32, lim_lo: Int32, lim_hi: Int32, col: int, *, loc=None, ip=None
) -> Float32:
    """The spec's mask of one score: ``-inf`` when its key -- static column ``col`` of the tile -- lies outside the row's visible range
    ``[lim_lo, lim_hi)`` (tile-relative bounds), else ``s`` unchanged. A select, not an arithmetic op: two compares of the bounds against
    the immediate column and one ``selp`` per key, with no dependence between keys (the general per-key ``if`` lowered to a predicate
    chain over uniform-register key sums, several times the instructions)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(s).ir_value(loc=loc, ip=ip),
                Int32(lim_lo).ir_value(loc=loc, ip=ip),
                Int32(lim_hi).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .pred p; setp.gt.s32 p, $2, %d; setp.le.or.s32 p, $3, %d, p; selp.f32 $0, 0fFF800000, $1, p; }"
            % (col, col),
            "=f,f,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class _RegLoader:
    """``canon.init_state`` leaf loader: key ``k`` -> the value holding column ``k`` (a plain callable
    object, not a closure: the DSL forbids captured closures)."""

    def __init__(self, vals):
        self.vals = vals

    def __call__(self, key):
        return self.vals[key[0]]


def streamed_tile_probs(ss, rP_e, c, nms, fp, in_dtype):
    """The slice's exponentials, their bf16 conversion and the rowsum tree's subtree over the slice, emitted in KEY order as a stream:
    for every pair of keys ``p = EX2(fma(s, c, nms))`` (the two fmas as one ``fma.rn.f32x2``), the pair's bf16 words, and the
    pair's ``add`` -- the first level of the engine's adjacent pairwise tree (``canon._pairwise``: low index first, level by level) --
    with every higher level's ``add`` issued as soon as both of its operands exist (a binary counter over the levels: the same
    tree, the same operand order). Written this way the live values at any point are the unprocessed scores, the finished P words
    and at most ``log2(keys)`` partial sums, so the P words can take the registers of the scores they came from; emitting every
    exponential before the first conversion kept all of them live next to the P block and cost the register allocator a copy of
    most of them per tile. ``ss[i]`` is the score of key ``i`` of the slice (register ``i``). Plain Python: trace-time structure."""
    assert len(ss) % 2 == 0 and len(ss) & (len(ss) - 1) == 0, len(ss)
    stack = []  # (level, partial sum) of the completed subtrees, highest level first
    for i in range(0, len(ss), 2):
        x0, x1 = fp.fma2((ss[i], ss[i + 1]), (c, c), (nms, nms))
        p0 = fp.ex2(x0)
        p1 = fp.ex2(x1)
        rP_e[i] = p0.to(in_dtype)
        rP_e[i + 1] = p1.to(in_dtype)
        v = fp.add(p0, p1)
        level = 1
        while stack and stack[-1][0] == level:
            v = fp.add(
                stack.pop()[1], v
            )  # the lower keys' subtree first (the tree's operand order)
            level += 1
        stack.append((level, v))
    assert len(stack) == 1, [lv for lv, _ in stack]
    return stack[0][1]


def packed_pairwise_add(vals, fp):
    """The engine's adjacent pairwise ``add`` tree (``canon._pairwise``: low index first, level by level) over
    ``vals`` (a power of two of them, in key order), issuing two adjacent adds of a level as one ``fp.add2``
    (``add.rn.f32x2``) and the odd last add of a level as ``fp.add``: the same adds of the same operands, so
    the same bits -- a packing placement of the canonical tree. Plain Python: trace-time structure."""
    while len(vals) > 1:
        nxt = []
        n = len(vals) // 2
        for j in range(0, n - 1, 2):
            s0, s1 = fp.add2(
                (vals[2 * j], vals[2 * j + 2]), (vals[2 * j + 1], vals[2 * j + 3])
            )
            nxt.append(s0)
            nxt.append(s1)
        if n % 2 == 1:
            nxt.append(fp.add(vals[2 * n - 2], vals[2 * n - 1]))
        vals = nxt
    return vals[0]


@cute.jit
def reduce_vals(plan, regs, reg_of_key, n_keys: cutlass.Constexpr) -> Float32:
    """Execute a register-level engine plan over ``n_keys`` keys (``regs[i]`` = the value held in register i;
    ``plan`` and ``reg_of_key`` are trace-time objects)."""
    vals = []
    for k in cutlass.range_constexpr(n_keys):
        vals.append(regs[reg_of_key[k]])
    state, start = canon.init_state(plan, CuteF32, _RegLoader(vals))
    return canon.run(plan, state, start, canon.Peers(), CuteF32)


def max3_tree(vals):
    """The maximum of a static list as a balanced 3-ary tree of three-input ``max.f32`` (a trailing pair takes the two-input one)."""
    while len(vals) > 1:
        nxt = []
        i = 0
        while i < len(vals):
            if i + 2 < len(vals):
                nxt.append(fmax3(vals[i], vals[i + 1], vals[i + 2]))
                i += 3
            elif i + 1 < len(vals):
                nxt.append(CuteF32.fmaxc(vals[i], vals[i + 1]))
                i += 2
            else:
                nxt.append(vals[i])
                i += 1
        vals = nxt
    return vals[0]


@cute.jit
def unit_loop(
    st,
    k: Int32,
    rep: cutlass.Constexpr,
    qk_thr_mma: cute.TiledMma,
    tStS: cute.Tensor,
    s_cons,
    p_prod,
    stats_prod,
    s_alpha: cute.Tensor,
    s_m: cute.Tensor,
    s_r: cute.Tensor,
    s_p: cute.Tensor,
    s_mx: cute.Tensor,
    c: Float32,
    nc: Float32,
    window_left: Int32,
    rows_valid: Int32,
    q_tok0: Int32,
    kv_len: Int32,
    q_len: Int32,
    tile_lo: Int32,
    tile_hi: Int32,
):
    """One unit for softmax warpgroup ``k``, in the full-row placement (its slice = keys ``[k kpt, (k+1) kpt)``) or the
    replicated-row placement (warp ``wq`` of the warpgroup takes slice ``4k + wq`` of ``4 n_wg``; the warpgroup's four
    slices are gathered in its warp 0 through smem before the P store). ``k`` is dynamic: the warpgroups run one copy
    of this code (the slice index only moves TMEM column offsets and smem indices)."""
    T, g, TPC = st.T, st.g, st.tpc
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % 128  # thread within the warpgroup (= TMEM lane)
    wq = cute.arch.make_warp_uniform(thread_idx // 32)
    lane = thread_idx % 32
    OWN2 = st.q2  # stage k of two-stage S / P / stats rings belongs to warpgroup k (skip the other's)
    ks = k  # this warpgroup's key slice (0 under ping-pong / pairing: every warpgroup holds whole rows)
    if cutlass.const_expr(OWN2):
        ks = Int32(0)
    if cutlass.const_expr(rep):
        kpt = st.kpt_rep  # keys per thread (the slice)
        row = lane  # the logical row: the warps hold replicas
        slice_id = 4 * ks + wq
        col_base = slice_id * kpt
    else:
        kpt = st.kpt
        row = thread_idx
        slice_id = ks
        col_base = ks * kpt
    # TMEM -> registers: the whole row for the maximum, the slice for the exponentials (the MMA's C partition of the
    # full (128, T) tile composed down to the slice's columns, as the correction block does for its pieces)
    cS = cute.make_identity_tensor((Q_ROWS, T))
    tScS = qk_thr_mma.partition_C(cS)
    HWM = st.hw_rowmax
    if cutlass.const_expr(HWM):
        # the load returns each 32-column piece's maximum in an extra register (tcgen05.ld.red::max): the unmasked tile's row maximum
        # is the max.f32 join of the pieces -- the same exact maximum as the register tree
        ld_full = cute.make_copy_atom(
            tcgen05.copy.LdRed32x32bOp(tcgen05.copy.Repetition(32)), ACC_DTYPE
        )
    else:
        ld_full = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), ACC_DTYPE
        )
    tiled_full = tcgen05.make_tmem_copy(ld_full, tStS)
    thr_full = tiled_full.get_slice(thread_idx)
    tFtS = thr_full.partition_S(tStS)
    tFcS = thr_full.partition_D(tScS)
    rF = cute.make_rmem_tensor(tFcS.shape, ACC_DTYPE)
    assert cute.size(rF) == T
    rRed = None
    if cutlass.const_expr(HWM):
        rRed = cute.make_rmem_tensor(((1, 1), *tFcS.shape[1:]), ACC_DTYPE)
        assert cute.size(rRed) == T // 32, cute.size(rRed)
    one_slice = kpt == T  # one warpgroup: the slice is the row, loaded once
    # two warpgroups on whole rows: each loads ONLY its key slice, forms the slice's maximum and exchanges it with the other
    # warpgroup through shared memory (fmax is exact and order-free, so the maximum of the two partials is the row maximum: a
    # placement); the replicated-row placement keeps the full-row load (its slices are spread over the warps of a warpgroup)
    exchange = (not one_slice) and (not rep)
    if cutlass.const_expr(one_slice):
        tiled_slice, tStS_part, tScS_part, rS = tiled_full, tFtS, tFcS, rF
    else:
        tScS_s = cute.make_tensor(
            tScS.iterator,
            cute.composition(tScS.layout, cute.make_layout((Q_ROWS, kpt))),
        )
        tStS_s = cute.make_tensor(
            tStS.iterator + col_base,
            cute.composition(tStS.layout, cute.make_layout((Q_ROWS, kpt))),
        )
        ld_slice = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(min(32, kpt))), ACC_DTYPE
        )
        tiled_slice = tcgen05.make_tmem_copy(ld_slice, tStS_s)
        thr_slice = tiled_slice.get_slice(thread_idx)
        tStS_part = thr_slice.partition_S(tStS_s)
        tScS_part = thr_slice.partition_D(tScS_s)
        rS = cute.make_rmem_tensor(tScS_part.shape, ACC_DTYPE)
    assert cute.size(rS) == kpt
    # P (bf16 pairs in 32-bit TMEM columns): the warpgroup's columns (its slice, or its four slices gathered in warp 0)
    p_words = st.kpt // 2
    tStS_P = cute.make_tensor(
        tStS.iterator + (st.tmem_p - st.tmem_s) + ks * p_words,
        cute.composition(tStS.layout, cute.make_layout((Q_ROWS, p_words))),
    )
    tScS_P = cute.make_tensor(
        tScS.iterator,
        cute.composition(tScS.layout, cute.make_layout((Q_ROWS, p_words))),
    )
    # one tcgen05.st covers the warpgroup's P columns (Repetition(p_words), 64 at the full tile): the packed bf16 pairs are
    # written from a single register block
    st_atom = cute.make_copy_atom(
        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(p_words)), ACC_DTYPE
    )
    tiled_st = tcgen05.make_tmem_copy(st_atom, tStS_P)
    thr_st = tiled_st.get_slice(thread_idx)
    tPtS = thr_st.partition_D(tStS_P)
    tPcS = thr_st.partition_S(tScS_P)
    rP = cute.make_rmem_tensor(tPcS.shape, ACC_DTYPE)
    assert cute.size(rP) == p_words
    for i in cutlass.range_constexpr(p_words):
        assert int(tPcS[i][1]) == i, (
            "P store fragment: register i must be column i (keys 2i, 2i+1 of the slice)"
        )
    rP_e = cute.make_tensor(
        cute.recast_ptr(rP.iterator, dtype=IN_DTYPE), cute.make_layout((2 * p_words,))
    )
    # the static column of every register and the key -> register maps (the engine's leaf order)
    cols_f = []
    for i in cutlass.range_constexpr(T):
        cols_f.append(tFcS[i][1])
    reg_f = {}
    for i in cutlass.range_constexpr(T):
        reg_f[int(cols_f[i])] = i
    cols_s = []
    for i in cutlass.range_constexpr(kpt):
        cols_s.append(tScS_part[i][1])
    reg_s = {}
    for i in cutlass.range_constexpr(kpt):
        reg_s[int(cols_s[i])] = i
    assert len(reg_f) == T and len(reg_s) == kpt, (
        "every key must be owned exactly once by the thread"
    )
    token = q_tok0 + row // g
    pos = kv_len - q_len + token
    L_row = pos + 1
    lo_row = Int32(0)
    if window_left >= 0:
        lo_row = pos - window_left
        if lo_row < 0:
            lo_row = Int32(0)
    neg_inf = Float32(_NEG_INF)
    m = Float32(_NEG_INF)
    N = tile_hi - tile_lo

    for n in range(N):
        t = tile_lo + n
        chunk_first = ((t % TPC) == 0) | (t == tile_lo)
        sh = s_cons.wait_and_advance()
        if cutlass.const_expr(OWN2):
            s_cons.advance(1)  # the other warpgroup's stage
        s_off = sh.index * st.tmem_s_stride  # this tile's S stage (0 with one stage)
        if cutlass.const_expr(not exchange):
            if cutlass.const_expr(HWM):
                cute.copy(
                    tiled_full,
                    cute.make_tensor(tFtS.iterator + s_off, tFtS.layout),
                    (rF, rRed),
                )
            else:
                cute.copy(
                    tiled_full,
                    cute.make_tensor(tFtS.iterator + s_off, tFtS.layout),
                    rF,
                )
        if cutlass.const_expr(not one_slice):
            cute.copy(
                tiled_slice,
                cute.make_tensor(tStS_part.iterator + s_off, tStS_part.layout),
                rS,
            )
        cute.arch.fence_view_async_tmem_load()
        sh.release()
        if cutlass.const_expr(st.p_over_s and st.n_slices > 1 and not exchange):
            # P is written over this stage's S columns: with two softmax warpgroups sharing TMEM lanes, no thread may store its P
            # slice before EVERY softmax thread has copied S out (the other warpgroup reads the whole row for its maximum)
            cute.arch.barrier(
                barrier_id=BAR_SOFTMAX_ALL, number_of_threads=st.softmax_threads
            )
        k0 = t * T
        # mask only the tiles that need it: a boundary of a live row's visible range inside the tile. Interior tiles
        # are fully visible and padding rows (row >= rows_valid) are never masked: their values are never stored (a
        # warp-uniform decision, no arithmetic involved)
        need_mask = cute.arch.vote_any_sync(
            ((k0 < lo_row) | (k0 + T > L_row)) & (row < rows_valid)
        )
        if need_mask:
            # the row's visible range in tile-relative columns: key k0 + col is masked iff col < lim_lo or col >= lim_hi (the slice's
            # columns offset by col_base)
            if cutlass.const_expr(st.D <= MASK_SELECT_MAX_D):
                lim_lo = lo_row - k0
                lim_hi = L_row - k0
                if cutlass.const_expr(not exchange):
                    for i in cutlass.range_constexpr(T):
                        rF[i] = masked_score(rF[i], lim_lo, lim_hi, int(cols_f[i]))
                if cutlass.const_expr(not one_slice):
                    lim_lo_s = lim_lo - col_base
                    lim_hi_s = lim_hi - col_base
                    for i in cutlass.range_constexpr(kpt):
                        rS[i] = masked_score(rS[i], lim_lo_s, lim_hi_s, int(cols_s[i]))
            else:
                # the two-stage body keeps the per-key compare: the select form raises its register pressure past the launch budget
                if cutlass.const_expr(not exchange):
                    for i in cutlass.range_constexpr(T):
                        key = k0 + cols_f[i]
                        if (key < lo_row) | (key >= L_row):
                            rF[i] = neg_inf
                if cutlass.const_expr(not one_slice):
                    for i in cutlass.range_constexpr(kpt):
                        key = k0 + col_base + cols_s[i]
                        if (key < lo_row) | (key >= L_row):
                            rS[i] = neg_inf
        if cutlass.const_expr(exchange):
            sp = []
            for i in cutlass.range_constexpr(kpt):
                sp.append(rS[i])
            m_part = max3_tree(sp)  # this slice's maximum
            par = n % 2
            s_mx[(par, ks, row)] = m_part
            cute.arch.barrier(
                barrier_id=BAR_SOFTMAX_ALL, number_of_threads=st.softmax_threads
            )
            m_t = CuteF32.fmaxc(m_part, s_mx[(par, 1 - ks, row)])
        else:
            sf = []
            for i in cutlass.range_constexpr(T):
                sf.append(rF[i])
            if cutlass.const_expr(HWM):
                sr = []
                for i in cutlass.range_constexpr(T // 32):
                    sr.append(rRed[i])
                m_t = max3_tree(
                    sr
                )  # the pieces' hardware maxima joined (an unmasked tile)
                if need_mask:  # the maximum of the masked values: the register tree
                    m_t = max3_tree(sf)
            else:
                m_t = max3_tree(
                    sf
                )  # the same value as the rowmax plan (max is exact and order-free)
        if chunk_first:
            m = neg_inf
        null = m_t == neg_inf  # no visible key for this row: a null step
        m_new = CuteF32.fmaxc(m, m_t)
        m_base = m_new
        if null:
            m_base = Float32(0.0)
        nms = CuteF32.mul(m_base, nc)
        alpha = spec.alpha_term(
            CuteF32, m, m_base, c, nc
        )  # EX2(add(mul(m, c), mul(m_new, nc)))
        if null:
            alpha = Float32(1.0)
        m = m_new
        # ---- alpha to the correction warpgroup at once (its O rescale overlaps the exponentials)
        sth = stats_prod.acquire_and_advance()
        s_alpha[(sth.index, row)] = alpha
        sth.commit()
        ss = []
        for kk in cutlass.range_constexpr(kpt):
            ss.append(rS[reg_s[kk]])  # the slice's scores in key order
        # p_k = EX2(fma(s_k, c, mul(m_new, nc))); masked -> +0. Whole or sliced rows under one softmax warp per scheduler: the
        # exponentials, their bf16 words and the rowsum tree as one stream in key order (streamed_tile_probs: the P words take the
        # scores' registers). The paired placement (two warpgroups on whole rows) and the replicated rows: the exponentials two keys
        # per fma.rn.f32x2, the subtree with packed adds, the conversion after -- measured faster there, where two softmax warps
        # share a scheduler and the fewer instructions of the packed adds matter more than the register copies
        if cutlass.const_expr(rep):
            rPq = cute.make_rmem_tensor(cute.make_layout((kpt // 2,)), ACC_DTYPE)
            rPq_e = cute.make_tensor(
                cute.recast_ptr(rPq.iterator, dtype=IN_DTYPE), cute.make_layout((kpt,))
            )
        if cutlass.const_expr(not rep and st.streamed):
            r_s = streamed_tile_probs(ss, rP_e, c, nms, CuteF32, IN_DTYPE)
        else:
            ps = [None] * kpt
            for i in cutlass.range_constexpr(0, kpt, 2):
                x0, x1 = CuteF32.fma2((ss[i], ss[i + 1]), (c, c), (nms, nms))
                ps[i] = CuteF32.ex2(x0)
                ps[i + 1] = CuteF32.ex2(x1)
            r_s = packed_pairwise_add(ps, CuteF32)
            if cutlass.const_expr(rep):
                for i in cutlass.range_constexpr(kpt):
                    rPq_e[i] = ps[i].to(IN_DTYPE)
            else:
                for i in cutlass.range_constexpr(kpt):
                    rP_e[i] = ps[i].to(IN_DTYPE)
        # ---- (m', r_slice) to the correction warpgroup, which owns the l chain and the merges
        sth = stats_prod.acquire_and_advance()
        s_m[(sth.index, row)] = m_new
        s_r[(sth.index, slice_id, row)] = r_s
        sth.commit()
        if cutlass.const_expr(OWN2):
            stats_prod.advance(2)  # the other warpgroup's two entries of the next tile
        # ---- P to TMEM (bf16, cvt.rn per element), the warpgroup's columns in key order
        ph = p_prod.acquire_and_advance()
        if cutlass.const_expr(OWN2):
            p_prod.advance(1)
        tPtS_k = cute.make_tensor(
            tPtS.iterator + ph.index * st.tmem_p_stride, tPtS.layout
        )  # this tile's P stage (= its S stage)
        if cutlass.const_expr(rep):
            # every warp converted its slice above; warps 1..3 hand theirs to warp 0 through smem, which stores the warpgroup's columns
            if wq > 0:
                for i in cutlass.range_constexpr(kpt // 2):
                    s_p[(k, wq - 1, lane, i)] = rPq[i]
            cute.arch.barrier(
                barrier_id=BAR_XCHG + k, number_of_threads=128
            )  # this warpgroup's four warps
            if wq == 0:
                for i in cutlass.range_constexpr(kpt // 2):
                    rP[i] = rPq[i]
                for sq in cutlass.range_constexpr(3):
                    for i in cutlass.range_constexpr(kpt // 2):
                        rP[(sq + 1) * (kpt // 2) + i] = s_p[(k, sq, lane, i)]
                cute.copy(tiled_st, rP, tPtS_k)
                cute.arch.fence_view_async_tmem_store()
            if cutlass.const_expr(st.s_stages > 1):
                # with two P stages the P acquire no longer serialises the warps on PV(n): warps 1..3 may not overwrite the exchange
                # buffer with tile n + 1's slices before warp 0 has read tile n's (a WAR race seen as an intermittent bit mismatch).
                # Placed after warp 0's store so that no register state lives across the barrier (register spills otherwise)
                cute.arch.barrier(barrier_id=BAR_XCHG + k, number_of_threads=128)
        else:
            cute.copy(tiled_st, rP, tPtS_k)
            cute.arch.fence_view_async_tmem_store()
        ph.commit()
    return s_cons, p_prod, stats_prod


@cute.jit
def softmax_loop(
    st,
    k: Int32,
    qk_thr_mma: cute.TiledMma,
    tStS: cute.Tensor,
    s_cons,
    p_prod,
    stats_prod,
    s_alpha: cute.Tensor,
    s_m: cute.Tensor,
    s_r: cute.Tensor,
    s_p: cute.Tensor,
    s_mx: cute.Tensor,
    c: Float32,
    nc: Float32,
    window_left: Int32,
    rows_valid: Int32,
    q_tok0: Int32,
    kv_len: Int32,
    q_len: Int32,
    tile_lo: Int32,
    tile_hi: Int32,
):
    """One unit for softmax warpgroup ``k``: the replicated-row placement when the unit has at most ``rep_rows = (32 // g) g``
    live rows (``g <= 32``), the full-row placement otherwise. Returns the advanced pipelines."""
    if cutlass.const_expr(st.rep_ok):
        if rows_valid <= st.rep_rows:
            s_cons, p_prod, stats_prod = unit_loop(
                st,
                k,
                True,
                qk_thr_mma,
                tStS,
                s_cons,
                p_prod,
                stats_prod,
                s_alpha,
                s_m,
                s_r,
                s_p,
                s_mx,
                c,
                nc,
                window_left,
                rows_valid,
                q_tok0,
                kv_len,
                q_len,
                tile_lo,
                tile_hi,
            )
        else:
            s_cons, p_prod, stats_prod = unit_loop(
                st,
                k,
                False,
                qk_thr_mma,
                tStS,
                s_cons,
                p_prod,
                stats_prod,
                s_alpha,
                s_m,
                s_r,
                s_p,
                s_mx,
                c,
                nc,
                window_left,
                rows_valid,
                q_tok0,
                kv_len,
                q_len,
                tile_lo,
                tile_hi,
            )
    else:
        s_cons, p_prod, stats_prod = unit_loop(
            st,
            k,
            False,
            qk_thr_mma,
            tStS,
            s_cons,
            p_prod,
            stats_prod,
            s_alpha,
            s_m,
            s_r,
            s_p,
            s_mx,
            c,
            nc,
            window_left,
            rows_valid,
            q_tok0,
            kv_len,
            q_len,
            tile_lo,
            tile_hi,
        )
    return s_cons, p_prod, stats_prod
