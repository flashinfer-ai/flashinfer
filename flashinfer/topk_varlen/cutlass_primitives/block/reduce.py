"""Block-wide reductions and scans built from warp collectives and one round of shared slots.

Pattern shared by every function here: each warp reduces in registers, lane 0 (or lane 31 for
scans) writes the warp's partial to its own slot, one block barrier, then warp-level code
folds the ``warps`` partials.  The slot array is owned by the caller and needs no zeroing:
only slots below the warp count are written, and reads are guarded to that count (at 512
threads lanes 16-31 would otherwise read stale slots, the bug recorded in the conventions).

``threads`` is a compile-time parameter; ``warps = threads // 32``.  All threads of the block
must call the block functions (they contain block barriers).  ``warp_reserve`` is the
barrier-free cousin: a warp-aggregated ticket on a shared counter.
"""

import cutlass
import cutlass.cute as cute

from ..device.warp import (
    warp_broadcast,
    warp_inclusive_scan_add,
    warp_max_u32,
    warp_sum,
)

__all__ = [
    "block_max_min_u32",
    "block_sum_i32",
    "block_exclusive_scan_i32",
    "warp_reserve",
]


@cute.jit
def block_max_min_u32(vmax, vmin, s_slots, tidx, threads: cutlass.Constexpr):
    """Maximum of ``vmax`` and minimum of ``vmin`` (both Uint32) over the block, to every thread.

    ``s_slots``: Uint32 shared array of ``2 * warps`` entries, caller-owned scratch, dead on
    exit.  One block barrier.  The minimum rides the same instruction as the maximum by
    reducing the complement (``max(~x) == ~min(x)``), which is why the two travel together:
    the sample phase wants both ends of the key range for one barrier.
    """
    warps = cutlass.const_expr(threads // 32)
    lane = tidx % 32
    warp = tidx // 32
    wmax = warp_max_u32(vmax)
    wnmin = warp_max_u32(~vmin)
    if lane == 0:
        s_slots[warp] = wmax
        s_slots[warps + warp] = wnmin
    cute.arch.barrier()
    pmax = cutlass.Uint32(0)
    pnmin = cutlass.Uint32(0)
    if lane < cutlass.Int32(warps):
        pmax = cutlass.Uint32(s_slots[lane])
        pnmin = cutlass.Uint32(s_slots[warps + lane])
    return warp_max_u32(pmax), ~warp_max_u32(pnmin)


@cute.jit
def block_sum_i32(val, s_slots, tidx, threads: cutlass.Constexpr):
    """Sum of ``val`` (Int32) over the block, to every thread.

    ``s_slots``: Int32 shared array of ``warps`` entries, dead on exit.  One block barrier.
    """
    warps = cutlass.const_expr(threads // 32)
    lane = tidx % 32
    warp = tidx // 32
    wsum = warp_sum(val)
    if lane == 0:
        s_slots[warp] = wsum
    cute.arch.barrier()
    part = cutlass.Int32(0)
    if lane < cutlass.Int32(warps):
        part = s_slots[lane]
    return warp_sum(part)


@cute.jit
def block_exclusive_scan_i32(val, s_slots, tidx, threads: cutlass.Constexpr):
    """Exclusive prefix sum of ``val`` (Int32) in thread order, and the block total, to
    every thread: thread t receives (val[0] + ... + val[t-1], total).

    ``s_slots``: Int32 shared array of ``warps`` entries.  One block barrier: the warp partials
    go out, then every warp reads all of them and sums the ones below it (one warp reduction
    each for the prefix and the total) instead of waiting for warp 0 to scan them in place
    behind a second barrier (v0.1.27; the second barrier cost about 0.1 us per use on the
    register kernels, whose classify is one of the uses).  ``s_slots`` is still being read
    when this returns: the caller must not write it before its next block barrier.  This is
    how a block assigns output positions without one serialized atomic per element.
    """
    warps = cutlass.const_expr(threads // 32)
    lane = tidx % 32
    warp = tidx // 32
    incl = warp_inclusive_scan_add(val, lane)
    if lane == 31:
        s_slots[warp] = incl
    cute.arch.barrier()
    part = cutlass.Int32(0)
    lower = cutlass.Int32(0)
    if lane < cutlass.Int32(warps):
        part = s_slots[lane]
        if lane < warp:
            lower = part
    total = warp_sum(part)
    before = warp_sum(lower)
    return before + incl - val, total


@cute.jit
def warp_reserve(count, lane, s_counter):
    """Reserve ``count`` consecutive slots for this lane from a shared counter with one atomic
    per warp; returns the lane's first slot.

    Inclusive scan of the counts, lane 31 adds the warp total to ``s_counter`` (skipped when
    zero) and broadcasts the old value; lane i's range starts at that base plus the sum of
    lower lanes.  All 32 lanes must participate.  Cost: 5 shuffles, one atomic, one shuffle.
    Used by the filter pass to stage survivors: with a few survivors per warp per tile, one
    atomic replaces up to 32.
    """
    incl = warp_inclusive_scan_add(count, lane)
    base = cutlass.Int32(0)
    if lane == 31:
        if incl != 0:
            base = cute.arch.atomic_add(s_counter, incl, sem="relaxed", scope="cta")
    return warp_broadcast(base, 31) + (incl - count)
