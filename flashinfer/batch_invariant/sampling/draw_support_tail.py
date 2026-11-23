# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""K5's list tail (docs/sampling_plan.md 21.3, 21.8): once the kept set is on the compaction list WITH its positions, the
kept-set sum, the final mass and the draw's crossing are computed on the list by one warp -- the canonical tree over the kept
entries with the identity leaves exact (``add(x, +0) = x`` for the positive kept weights, the absent leaves being ``+0``), the
crossing's prefixes as the add chain of the left-sibling nodes along each entry's slot address, bit for bit the dense sweeps'
values (the CPU prototypes of 2026-11-17: 300 + 400 random kept sets, no mismatch).

Tree order (``reference.row_sum``, ``draw_reference.slot_folds`` / ``outer_prefix``): the leaf of element ``g`` is at slot
``s = (lane << kp_log) | block`` (``block = g >> 8``, ``lane = (g >> 3) & 31``) and position ``e = g & 7`` within it; the slot
fold adds from ``+0`` in ``e`` order; the pairwise levels combine the slot index's bits from bit 0 up (the block bits, then the
lane bits); the outer prefix of a leaf is the chain, ascending in the level, of the left-sibling node values where the slot
index has a 1 bit, from ``+0``; the inclusive prefix is ``add(outer, running)`` with ``running`` the slot fold through the
element. The crossing: the first leaf in tree order with a positive weight whose inclusive prefix reaches ``mul(u, mass)``,
else the last positive leaf (``draw_kernel._crossing_pass``).

The placement (lever C, 2026-11-17): the entries stay in registers at their SORTED index -- two per lane, a bitonic sort of
the packed ``(tree address, list index)`` keys over shuffles -- and every level looks up ONE sibling entry by index: the node
containing entry ``i`` at level ``L`` is the run of entries ``[rs, re]`` with the same ``slot >> L``, its sibling the adjacent
run (``re + 1`` or ``rs - 1``) whose ``slot >> L`` is the sibling id; the node value becomes ``add(left, right)``, the run
bounds merge, and a LEFT sibling's value joins the entry's outer prefix. No node tables, no compaction, no shared-memory round
trips: the previous form (sorted node tables in shared memory, an O(64^2) rank, 15 levels of dependent loads) cost 10-19k cycles
per call. One ``@cute.jit`` over the list tensors (no Python object crosses its dynamic branch): warp 0 computes, the other warps
wait at the barriers, the three results are broadcast through a shared slot.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32
from .cute_ext import CuteF32
from .draw_kernel import _imax, _imin, _ld_shared_cluster_i32
from .kernel import _map_shared_rank, _select
from .schedule import LANES
from .support_kernel import CAP, _bitcast_i32_to_f32, _f32_bits, _popc

TAIL_WORDS = 4  # the broadcast slot: mass bits, first token, last token, (spare)
SENTINEL = 0x7FFFFFC0  # a non-participant's key (above every participant's: the tree address is below 2^25)
assert CAP == 2 * LANES  # two list entries per lane


def tail_smem_bytes() -> int:
    """The tail's shared memory: the broadcast slot."""
    return TAIL_WORDS * 4 + 16


def alloc_tail(smem):
    """The broadcast slot."""
    return smem.allocate_tensor(
        Int32, cute.make_layout((TAIL_WORDS,)), byte_alignment=16
    )


def _add(a: Float32, b: Float32) -> Float32:
    """``add.rn.f32`` as a plain function (a method call on the backend class inside the warp branch would carry the class)."""
    return CuteF32.add(a, b)


def _mul(a: Float32, b: Float32) -> Float32:
    return CuteF32.mul(a, b)


def _shf(v: Int32, src: Int32) -> Int32:
    """``shfl.idx`` from lane ``src`` (a dynamic index, clamped by the caller)."""
    return cute.arch.shuffle_sync(v, offset=src)


def _pick(lo: Int32, hi: Int32, idx: Int32) -> Int32:
    """The value of sorted index ``idx`` (0..63) from the two per-lane registers: ``lo`` holds indices 0..31, ``hi`` 32..63."""
    src = idx & Int32(LANES - 1)
    a = _shf(lo, src)
    b = _shf(hi, src)
    return _select(idx < Int32(LANES), a, b)


def _cmpx(k: Int32, p: Int32, keep_min: Boolean) -> Int32:
    return _select(keep_min, _imin(k, p), _imax(k, p))


@cute.jit
def list_tail(
    cand: cute.Tensor,
    cpos: cute.Tensor,
    n_all: Int32,
    floor_bits: Int32,
    u: Float32,
    draw: Boolean,
    warp: Int32,
    lane: Int32,
    out: cute.Tensor,
    kp_log: cutlass.Constexpr[int],
):
    """The kept entries of ``cand`` / ``cpos`` (the first ``n_all <= CAP`` slots) with pattern ``>= floor_bits`` and positive:
    their canonical sum tree, and with ``draw`` the crossing at ``mul(u, sum)``. Returns ``(sum, first token, last token)``
    as every thread's values (``-1`` tokens when nothing is kept or ``draw`` is off)."""
    levels = kp_log + 5
    cute.arch.barrier()  # the lists are complete (the caller's gather)
    if warp == Int32(0):
        i0 = lane
        i1 = lane + Int32(LANES)
        b0 = cand[_select(i0 < n_all, i0, Int32(CAP))]
        b1 = cand[_select(i1 < n_all, i1, Int32(CAP))]
        g0 = cpos[_select(i0 < n_all, i0, Int32(CAP))]
        g1 = cpos[_select(i1 < n_all, i1, Int32(CAP))]
        p0 = (
            (i0 < n_all) & (b0 >= floor_bits) & (b0 > Int32(0))
        )  # positive: the count tail's list may hold zero weights
        p1 = (i1 < n_all) & (b1 >= floor_bits) & (b1 > Int32(0))
        # the packed sort key: the tree address ((lane << kp_log) | block) << 3 | e, then the list index (6 bits)
        a0 = ((((g0 >> 3) & Int32(LANES - 1)) << kp_log) | (g0 >> 8)) << 3 | (
            g0 & Int32(7)
        )
        a1 = ((((g1 >> 3) & Int32(LANES - 1)) << kp_log) | (g1 >> 8)) << 3 | (
            g1 & Int32(7)
        )
        k0 = _select(p0, (a0 << 6) | i0, Int32(SENTINEL) | i0)
        k1 = _select(p1, (a1 << 6) | i1, Int32(SENTINEL) | i1)
        m = _popc(cute.arch.vote_ballot_sync(p0)) + _popc(
            cute.arch.vote_ballot_sync(p1)
        )
        # the bitonic sort of the 64 keys, ascending: the partner of index i is i ^ j; the lower index keeps the minimum when
        # (i & k) == 0; j == 32 pairs a lane's two registers, the other steps pair lanes of one register (xor shuffles)
        for k in cutlass.range_constexpr(1, 7):  # k = 2 .. 64
            kk = 1 << k
            for jj in cutlass.range_constexpr(k):  # j = kk / 2 .. 1
                j = kk >> (jj + 1)
                if cutlass.const_expr(j == LANES):
                    lo = _imin(k0, k1)
                    hi = _imax(k0, k1)
                    k0 = lo  # (i & 64) == 0 for every i: ascending
                    k1 = hi
                else:
                    q0 = cute.arch.shuffle_sync_bfly(k0, offset=j)
                    q1 = cute.arch.shuffle_sync_bfly(k1, offset=j)
                    lower = (lane & Int32(j)) == Int32(0)
                    asc0 = (i0 & Int32(kk)) == Int32(0)
                    asc1 = (i1 & Int32(kk)) == Int32(0)
                    k0 = _cmpx(k0, q0, lower == asc0)
                    k1 = _cmpx(k1, q1, lower == asc1)
        # the sorted entries: index i0 / i1 of the sorted order holds key k0 / k1
        q0 = i0 < m
        q1 = i1 < m
        o0 = k0 & Int32(63)
        o1 = k1 & Int32(63)
        s0 = k0 >> 9  # the slot (the address >> 3)
        s1 = k1 >> 9
        w0 = _bitcast_i32_to_f32(cand[_select(q0, o0, Int32(CAP))])
        w1 = _bitcast_i32_to_f32(cand[_select(q1, o1, Int32(CAP))])
        t0 = cpos[_select(q0, o0, Int32(CAP))]
        t1 = cpos[_select(q1, o1, Int32(CAP))]
        # the slot fold from +0 over the same-slot predecessors in order (at most 7, consecutive in the sorted order), and the
        # run bounds [rs, re] of every entry's slot
        same0 = []
        same1 = []
        pw0 = []
        pw1 = []
        for d in cutlass.range_constexpr(1, 8):
            up = _imax(lane - Int32(d), Int32(0))
            cross = lane + Int32(LANES - d)
            ps0 = _shf(s0, up)
            ps1 = _select(lane >= Int32(d), _shf(s1, up), _shf(s0, cross))
            pw0.append(_shf(_f32_bits(w0), up))
            pw1.append(
                _select(
                    lane >= Int32(d),
                    _shf(_f32_bits(w1), up),
                    _shf(_f32_bits(w0), cross),
                )
            )
            same0.append((lane >= Int32(d)) & (ps0 == s0))
            same1.append(ps1 == s1)
        cons0 = []
        cons1 = []
        c0 = Boolean(True)
        c1 = Boolean(True)
        for d in cutlass.range_constexpr(7):
            c0 = c0 & same0[d]
            c1 = c1 & same1[d]
            cons0.append(c0)
            cons1.append(c1)
        run0 = Float32(0.0)
        run1 = Float32(0.0)
        for d in cutlass.range_constexpr(6, -1, -1):
            run0 = _select(cons0[d], _add(run0, _bitcast_i32_to_f32(pw0[d])), run0)
            run1 = _select(cons1[d], _add(run1, _bitcast_i32_to_f32(pw1[d])), run1)
        run0 = _add(run0, w0)
        run1 = _add(run1, w1)
        n0 = Int32(0)
        n1 = Int32(0)
        for d in cutlass.range_constexpr(7):
            n0 = n0 + _select(cons0[d], Int32(1), Int32(0))
            n1 = n1 + _select(cons1[d], Int32(1), Int32(0))
        rs0 = i0 - n0
        rs1 = i1 - n1
        e0 = Boolean(True)
        e1 = Boolean(True)
        m0 = Int32(0)
        m1 = Int32(0)
        for d in cutlass.range_constexpr(1, 8):
            dn = _imin(lane + Int32(d), Int32(LANES - 1))
            cross = lane + Int32(d - LANES)  # index i0 + d in the high register
            ns0 = _select(
                lane + Int32(d) < Int32(LANES),
                _shf(s0, dn),
                _shf(s1, _imax(cross, Int32(0))),
            )
            ns1 = _shf(s1, dn)
            e0 = e0 & (ns0 == s0)
            e1 = e1 & (lane + Int32(d) < Int32(LANES)) & (ns1 == s1)
            m0 = m0 + _select(e0, Int32(1), Int32(0))
            m1 = m1 + _select(e1, Int32(1), Int32(0))
        re0 = i0 + m0
        re1 = i1 + m1
        # the slot node's value: the fold through the run's last entry
        v0 = _bitcast_i32_to_f32(_pick(_f32_bits(run0), _f32_bits(run1), re0))
        v1 = _bitcast_i32_to_f32(_pick(_f32_bits(run0), _f32_bits(run1), re1))
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        for L in cutlass.range_constexpr(levels):
            my0 = s0 >> L
            my1 = s1 >> L
            bit0 = (my0 & Int32(1)) == Int32(1)
            bit1 = (my1 & Int32(1)) == Int32(1)
            j0 = _select(bit0, rs0 - Int32(1), re0 + Int32(1))
            j1 = _select(bit1, rs1 - Int32(1), re1 + Int32(1))
            jv0 = (j0 >= Int32(0)) & (j0 < Int32(CAP))
            jv1 = (j1 >= Int32(0)) & (j1 < Int32(CAP))
            jc0 = _select(jv0, j0, Int32(0))
            jc1 = _select(jv1, j1, Int32(0))
            sj0 = _pick(s0, s1, jc0)
            sj1 = _pick(s0, s1, jc1)
            has0 = q0 & jv0 & ((sj0 >> L) == (my0 ^ Int32(1)))
            has1 = q1 & jv1 & ((sj1 >> L) == (my1 ^ Int32(1)))
            vj0 = _bitcast_i32_to_f32(_pick(_f32_bits(v0), _f32_bits(v1), jc0))
            vj1 = _bitcast_i32_to_f32(_pick(_f32_bits(v0), _f32_bits(v1), jc1))
            rsj0 = _pick(rs0, rs1, jc0)
            rsj1 = _pick(rs0, rs1, jc1)
            rej0 = _pick(re0, re1, jc0)
            rej1 = _pick(re0, re1, jc1)
            nv0 = _select(bit0, _add(vj0, v0), _add(v0, vj0))
            nv1 = _select(bit1, _add(vj1, v1), _add(v1, vj1))
            acc0 = _select(has0 & bit0, _add(acc0, vj0), acc0)
            acc1 = _select(has1 & bit1, _add(acc1, vj1), acc1)
            v0 = _select(has0, nv0, v0)
            v1 = _select(has1, nv1, v1)
            rs0 = _select(has0, _imin(rs0, rsj0), rs0)
            rs1 = _select(has1, _imin(rs1, rsj1), rs1)
            re0 = _select(has0, _imax(re0, rej0), re0)
            re1 = _select(has1, _imax(re1, rej1), re1)
        total = _select(
            m > Int32(0),
            _bitcast_i32_to_f32(_shf(_f32_bits(v0), Int32(0))),
            Float32(0.0),
        )
        # the crossing over the kept entries in tree order
        thr = _mul(u, total)
        f0 = _add(acc0, run0)
        f1 = _add(acc1, run1)
        hit0 = q0 & (f0 >= thr)
        hit1 = q1 & (f1 >= thr)
        first = _imin(_select(hit0, i0, Int32(CAP)), _select(hit1, i1, Int32(CAP)))
        for mask in cutlass.range_constexpr(5):
            first = _imin(first, cute.arch.shuffle_sync_bfly(first, offset=1 << mask))
        last = m - Int32(1)
        tok_first = _pick(t0, t1, _select(first < Int32(CAP), first, Int32(0)))
        tok_last = _pick(t0, t1, _imax(last, Int32(0)))
        if lane == Int32(0):
            out[0] = _f32_bits(total)
            out[1] = _select(draw & (first < Int32(CAP)), tok_first, Int32(-1))
            out[2] = _select(draw & (m > Int32(0)), tok_last, Int32(-1))
    cute.arch.barrier()
    total = _bitcast_i32_to_f32(out[0])
    first_tok = out[1]
    last_tok = out[2]
    cute.arch.barrier()  # every thread has read the slot before the next tail overwrites it
    return total, first_tok, last_tok


@cute.jit
def reset_box(cbox: cute.Tensor, tid: Int32):
    """The compaction counter cleared by one thread (the caller places the barrier)."""
    if tid == Int32(0):
        cbox[0] = Int32(0)


@cute.jit
def gather_lists(
    cand: cute.Tensor,
    cpos: cute.Tensor,
    cbox: cute.Tensor,
    tid: Int32,
    c_rank: Int32,
    C: cutlass.Constexpr[int],
    nthreads: cutlass.Constexpr[int],
) -> Int32:
    """After every CTA compacted its share: the cluster's lists (patterns and positions) appended to this CTA's through
    ``mapa`` (a candidate past the capacity to the dummy slot), so every CTA holds the same list in the same order (own
    entries, then the peers' in rank order); returns the cluster-wide count (CTA-uniform). The loads stop at the capacity:
    a peer's count may exceed it (the caller then takes the dense path), and a read past its list ran past the shared
    memory window once the tail's scratch shrank (2026-11-17: the top-p launch failures at 152064)."""
    n_all = cbox[0]
    if cutlass.const_expr(C > 1):
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        off = n_all
        for sid in cutlass.range_constexpr(C):
            cnt_s = _ld_shared_cluster_i32(_map_shared_rank(cbox.iterator, Int32(sid)))
            cnt_s = _select(Int32(sid) == c_rank, Int32(0), cnt_s)
            base_c = _map_shared_rank(cand.iterator, Int32(sid))
            base_p = _map_shared_rank(cpos.iterator, Int32(sid))
            for s in cutlass.range_constexpr(-(-CAP // nthreads)):
                j = tid + Int32(s * nthreads)
                ok = (
                    (j < cnt_s) & (j < Int32(CAP))
                )  # a peer's count beyond the capacity (the dense cases): its list holds CAP
                v = _ld_shared_cluster_i32(base_c + _select(ok, j, Int32(0)) * 4)
                g = _ld_shared_cluster_i32(base_p + _select(ok, j, Int32(0)) * 4)
                dst = off + j
                slot = _select(ok & (dst < Int32(CAP)), dst, Int32(CAP))
                cand[slot] = v
                cpos[slot] = g
            off = off + cnt_s
        n_all = off
        cute.arch.barrier()
    return n_all
