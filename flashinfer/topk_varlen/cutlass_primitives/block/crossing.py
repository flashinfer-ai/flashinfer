"""Rank crossings over a histogram: which bin holds the element of a given rank from the top.

Bins ascend with value.  For a target rank ``t`` (1 = the largest element) the crossing bin
``b`` is the one where the count in bins strictly above ``b`` is below ``t`` and including
``b`` reaches it:  ``above < t <= above + count[b]``.  Every phase of a selection kernel asks
this question: the sample asks it to aim a threshold, the verdict asks it at rank k over the
survivor histogram, the radix fallback asks it once per digit.  If ``t`` exceeds the total
the answer is bin 0 with ``above`` = everything above bin 0 (the caller sees ``above +
count < t`` and knows the histogram ran out).

Three shapes for three sizes of question:

* ``crossing_256_warp``: one warp, 256 bins, two targets, no barrier.  Lane l owns bins
  8l..8l+7 (two 16-byte loads), a warp scan gives each lane the count above its span, and
  the owning lane walks its 8 bins.  Optionally zeroes the histogram on the way out so the
  next phase's clear is free.  Cheaper than a 256-thread scan for this size (measured:
  0.73 us behind one barrier against 1.3 us for 32 warps each redundantly scanning).
* ``crossing_256_block``: 256 threads, one bin each, one named barrier among themselves and
  one full barrier; the form used inside the radix rounds where 256 threads already hold
  the bins.
* ``crossing_wide_pair``: all threads over ``bins`` bins (4096 or 8192), two targets, two
  barriers; for the coarse histograms of the register-resident and radix kernels.

Warp forms return their answer on every lane of the warp.  Block forms publish to a shared
result array and end with a block barrier so every thread can read it.
"""

import cutlass
import cutlass.cute as cute

from ..device.memory import clear_shared_16, load_shared_16
from ..device.warp import (
    warp_broadcast,
    warp_inclusive_scan_add,
    warp_max_u32,
    warp_sum,
)

__all__ = [
    "crossing_256_warp",
    "crossing_256_block",
    "crossing_wide_pair",
    "crossing_wide_warp",
]


@cute.jit
def _bin_word(v0, v1, v2, v3, v4, v5, v6, v7, j: cutlass.Constexpr):
    return (v0, v1, v2, v3, v4, v5, v6, v7)[j]


@cute.jit
def crossing_256_warp(
    s_hist,
    target_a,
    target_b,
    lane,
    clear: cutlass.Constexpr,
    copies: cutlass.Constexpr = 1,
    s_cursor=None,
    cursors: cutlass.Constexpr = False,
):
    """Crossing bins of two target ranks over a 256-bin Int32 histogram, by one warp.

    Contract: exactly one warp calls this with all 32 lanes; ``s_hist`` is 16-byte aligned
    and complete (a barrier separates the last increment from this call).  With ``copies``
    above one the histogram is the sum of ``copies`` consecutive 256-bin arrays (a privatized
    histogram; the lanes sum them while loading).  With ``clear`` the histogram (every copy) is
    zero on return (the caller's next barrier publishes the zeros).  With ``cursors``,
    ``s_cursor[b]`` (256 Int32) receives the count in bins strictly above ``b`` for every bin:
    the rank-ordered start of bin ``b``'s members, so an emit can place each candidate with an
    atomic on its own bin's cursor instead of one cursor shared by every winner (the walk
    already has the number; eight shared stores per lane).
    Returns ``(bin_a, above_a, count_a, bin_b, above_b, count_b)`` on every lane.
    Cost: two 16-byte shared loads per copy, one warp scan, 8 steps of compares, three shuffles.
    """
    base_bin = lane * 8
    addr = s_hist.toint() + lane * 32
    v0, v1, v2, v3 = load_shared_16(addr)
    v4, v5, v6, v7 = load_shared_16(addr + 16)
    for c in cutlass.range_constexpr(1, copies):
        u0, u1, u2, u3 = load_shared_16(addr + c * 1024)
        u4, u5, u6, u7 = load_shared_16(addr + c * 1024 + 16)
        v0, v1, v2, v3 = v0 + u0, v1 + u1, v2 + u2, v3 + u3
        v4, v5, v6, v7 = v4 + u4, v5 + u5, v6 + u6, v7 + u7
    mine = (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7).to(cutlass.Int32)
    incl = warp_inclusive_scan_add(mine, lane)
    total = warp_broadcast(incl, 31)
    above = total - incl  # count in bins above this lane's span
    hit_a = cutlass.Int32(0)
    hit_b = cutlass.Int32(0)
    above_a = cutlass.Int32(0)
    above_b = cutlass.Int32(0)
    count_a = cutlass.Int32(0)
    count_b = cutlass.Int32(0)
    for j in cutlass.range_constexpr(7, -1, -1):
        c = _bin_word(v0, v1, v2, v3, v4, v5, v6, v7, j).to(cutlass.Int32)
        b = base_bin + j
        if cutlass.const_expr(cursors):
            s_cursor[b] = above
        if above < target_a:
            if (above + c >= target_a) | (b == 0):
                hit_a = b
                above_a = above
                count_a = c
        if above < target_b:
            if (above + c >= target_b) | (b == 0):
                hit_b = b
                above_b = above
                count_b = c
        above = above + c
    if cutlass.const_expr(clear):
        for c in cutlass.range_constexpr(copies):
            clear_shared_16(addr + c * 1024)
            clear_shared_16(addr + c * 1024 + 16)
    # exactly one lane hit each target (or lane 0 by the bin-0 rule); the bin doubles as the
    # lane address for the broadcast
    bin_a = warp_max_u32(cutlass.Uint32(hit_a)).to(cutlass.Int32)
    bin_b = warp_max_u32(cutlass.Uint32(hit_b)).to(cutlass.Int32)
    return (
        bin_a,
        warp_broadcast(above_a, bin_a >> 3),
        warp_broadcast(count_a, bin_a >> 3),
        bin_b,
        warp_broadcast(above_b, bin_b >> 3),
        warp_broadcast(count_b, bin_b >> 3),
    )


@cute.jit
def crossing_256_block(s_hist, total, target, s_slots, s_result, tidx):
    """Crossing bin of one target rank over a 256-bin Int32 histogram, by threads 0..255.

    ``total`` is the histogram sum (known to the caller from the count that built it).
    ``s_slots``: Int32 shared array of 8 entries.  Publishes ``s_result[0..2] = (bin, above,
    count)`` and ends with a full block barrier; requires at least 256 threads and that every
    thread of the block calls it.  Cost: one warp scan, one named barrier, one full barrier.
    """
    lane = tidx % 32
    warp = tidx // 32
    if tidx < 256:
        c = s_hist[tidx]
        incl = warp_inclusive_scan_add(c, lane)
        if lane == 31:
            s_slots[warp] = incl
        cute.arch.barrier(barrier_id=1, number_of_threads=256)
        lower = cutlass.Int32(0)
        if lane < 8:
            if lane < warp:
                lower = s_slots[lane]
        incl = incl + warp_sum(lower)  # count in bins <= mine
        above = total - incl
        if above < target:
            if (above + c >= target) | (tidx == 0):
                s_result[0] = tidx
                s_result[1] = above
                s_result[2] = c
    cute.arch.barrier()


@cute.jit
def crossing_wide_warp(s_hist, bins: cutlass.Constexpr, target, lane):
    """Crossing bin of one target rank over a ``bins``-bin Int32 histogram (1024 to 4096), by
    one warp in two levels, no barrier.

    Level 1: lane l sums bins ``l * span .. (l + 1) * span`` (``span = bins / 32``, 16-byte
    loads), a warp scan gives the count above each span, and the span holding the target is
    found.  Level 2: the warp re-reads that span, ``span / 32`` bins per lane, scans again and
    the owning lane walks its bins.  Identical answer to ``crossing_wide_pair``; costs the
    lane loads (8 to 32 per lane) and two warp scans instead of two block barriers, so the
    other warps of the block are free and the caller needs one barrier to publish.  A target
    above the total is clamped to it; an empty histogram answers bin 0.  Returns ``(bin,
    above, count)`` on every lane.

    Measured worse than ``crossing_wide_pair`` in the register kernel on B200 (2026-09-18:
    1K b=1 2.22 -> 2.50 us, 16K b=1 5.14 -> 6.08, 16K b=148 5.39 -> 6.33): one warp reading
    8 to 32 vectors per lane serially is slower than 32 warps reading one each behind two
    barriers.  Kept for histograms where the block is busy elsewhere; not used by any kernel.
    """
    span = cutlass.const_expr(bins // 32)
    sub = cutlass.const_expr(span // 32)
    addr = s_hist.toint() + lane * (span * 4)
    mine = cutlass.Uint32(0)
    for q in cutlass.range_constexpr(span // 4):
        a, b, c, d = load_shared_16(addr + q * 16)
        mine = mine + a + b + c + d
    mine_i = mine.to(cutlass.Int32)
    incl = warp_inclusive_scan_add(mine_i, lane)
    total = warp_broadcast(incl, 31)
    above = total - incl
    t = target
    if t > total:
        t = total
    hit = cutlass.Int32(0)
    if (above < t) & (above + mine_i >= t):
        hit = lane
    hit_lane = warp_max_u32(cutlass.Uint32(hit)).to(
        cutlass.Int32
    )  # lane 0 when nothing crosses
    above_span = warp_broadcast(above, hit_lane)
    # level 2 over the hit span: ``sub`` consecutive bins per lane
    base2 = s_hist.toint() + (hit_lane * span + lane * sub) * 4
    vals: list = []
    if cutlass.const_expr(sub == 4):
        a, b, c, d = load_shared_16(base2)
        vals.extend(
            (
                a.to(cutlass.Int32),
                b.to(cutlass.Int32),
                c.to(cutlass.Int32),
                d.to(cutlass.Int32),
            )
        )
    else:
        for i in cutlass.range_constexpr(sub):
            vals.append(s_hist[hit_lane * span + lane * sub + i])
    mine2 = cutlass.Int32(0)
    for i in cutlass.range_constexpr(sub):
        mine2 = mine2 + vals[i]
    incl2 = warp_inclusive_scan_add(mine2, lane)
    span_total = warp_broadcast(incl2, 31)
    above2 = above_span + (span_total - incl2)  # count above this lane's bins
    hit_bin = cutlass.Int32(0)
    hit_above = cutlass.Int32(0)
    hit_count = cutlass.Int32(0)
    for i in cutlass.range_constexpr(sub - 1, -1, -1):
        c = vals[i]
        b = hit_lane * span + lane * sub + i
        if above2 < t:
            if (above2 + c >= t) | (b == 0):
                hit_bin = b
                hit_above = above2
                hit_count = c
        above2 = above2 + c
    bin_out = warp_max_u32(cutlass.Uint32(hit_bin)).to(cutlass.Int32)
    owner = (bin_out - hit_lane * span) // cutlass.Int32(sub)
    return bin_out, warp_broadcast(hit_above, owner), warp_broadcast(hit_count, owner)


@cute.jit
def crossing_wide_pair(
    s_hist,
    bins: cutlass.Constexpr,
    target_a,
    target_b,
    s_slots,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
):
    """Crossing bins of two target ranks over a ``bins``-bin Int32 histogram, by the whole block.

    Thread t owns bins ``t * items .. (t + 1) * items`` with ``items = bins // threads``.
    ``s_slots``: Int32 shared array of ``warps`` entries.  Targets above the total are clamped
    to it.  Publishes ``s_result[0..2]`` for ``target_a`` and ``s_result[3..5]`` for
    ``target_b`` as (bin, above, count); ends with a block barrier.  Two barriers, one
    16-byte shared load per four bins (values kept for the second pass).
    """
    items = cutlass.const_expr(bins // threads)
    warps = cutlass.const_expr(threads // 32)
    lane = tidx % 32
    warp = tidx // 32
    # the thread's bins in one 16-byte load per four (a scalar load per bin is a four-way
    # bank conflict at this stride), kept in registers for the second pass
    vals: list = []
    if cutlass.const_expr(items % 4 == 0):
        for q in cutlass.range_constexpr(items // 4):
            v0, v1, v2, v3 = load_shared_16(s_hist.toint() + (tidx * items + 4 * q) * 4)
            vals.extend(
                (
                    v0.to(cutlass.Int32),
                    v1.to(cutlass.Int32),
                    v2.to(cutlass.Int32),
                    v3.to(cutlass.Int32),
                )
            )
    else:
        for i in cutlass.range_constexpr(items):
            vals.append(s_hist[tidx * items + i])
    mine = cutlass.Int32(0)
    for i in cutlass.range_constexpr(items):
        mine = mine + vals[i]
    incl = warp_inclusive_scan_add(mine, lane)
    if lane == 31:
        s_slots[warp] = incl
    cute.arch.barrier()
    slot = cutlass.Int32(0)
    lower = cutlass.Int32(0)
    if lane < cutlass.Int32(warps):
        slot = s_slots[lane]
        if lane < warp:
            lower = slot
    total = warp_sum(slot)
    below = warp_sum(lower) + (incl - mine)  # count in bins below this thread's span
    ta = target_a
    if ta > total:
        ta = total
    tb = target_b
    if tb > total:
        tb = total
    for i in cutlass.range_constexpr(items):
        c = vals[i]
        below = below + c
        above = total - below
        if above < ta:
            if above + c >= ta:
                s_result[0] = tidx * items + i
                s_result[1] = above
                s_result[2] = c
        if above < tb:
            if above + c >= tb:
                s_result[3] = tidx * items + i
                s_result[4] = above
                s_result[5] = c
    cute.arch.barrier()
