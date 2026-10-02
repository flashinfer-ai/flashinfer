"""Slab merge: a row split over CTAs that cannot share memory (no clusters, or more than 8).

Each CTA publishes its stage to its own segment of a global slab, bin-major, with a table of
per-bin prefixes, then adds one to the row's arrival counter.  The CTA that sees the count reach
``splits - 1`` is last: everyone else's data is visible to it (release before the add, acquire
after), and it alone merges the histograms, finds the rank-k bin, copies winners and ties by
range, selects the ties, and resets the counter for the next launch.  No CTA ever waits, so
the grid may run in as many waves as it likes.

Bin-major publish is what makes the last arriver cheap: with every segment sorted into bins
by the prefix table, "everything above bin B" and "everything in bin B" are contiguous ranges,
copied without classifying a single candidate again.  Classifying at the last arriver measured
2.5-3 us per row at 1M elements in the FlashInfer kernel before the change.

Limits of this form: the verdict is known only to the last arriver, after every other CTA has
left, so an undershoot cannot be repaired by a re-walk; the dispatcher pairs the slab merge
with the wide aim, and an undershoot or a stage overflow goes to the exact fallback.

Segment layout per row (Int32 words): ``splits x capacity`` keys, ``splits x capacity``
indices, ``splits x TABLE_WORDS`` tables (``[0]`` staged count or -1 on overflow, ``[1 + b]``
exclusive prefix of bin ``b`` for b in 0..256).  One Int32 arrival counter per row, zero
between launches.  The last arriver rebuilds the merged histogram from prefix differences
(2 x splits L2 loads per bin, all independent); accumulating a row histogram with atomics
instead measured 3-4% worse at 1M and 256K b=8 (docs/measured-worse.md).
"""

import cutlass
import cutlass.cute as cute

from ...block.crossing import crossing_256_warp
from ...device.atomics import global_add_acq_rel, shared_add
from ...device.memory import load_global_l2_i32
from ...device.timers import read_clock64
from ...device.warp import warp_inclusive_scan_add

from .binning import survivor_bin
from .filter_pass import stage_pair
from .resolve import _select_ties

__all__ = ["TABLE_WORDS", "slab_words_per_row", "publish_and_arrive", "merge_slab"]

TABLE_WORDS = 258


def slab_words_per_row(splits: int, capacity: int) -> int:
    return splits * (2 * capacity + TABLE_WORDS)


@cute.jit
def publish_and_arrive(
    elems,
    rank,
    splits: cutlass.Constexpr,
    bar,
    scale,
    capacity: cutlass.Constexpr,
    s_count,
    s_stage,
    s_hist,
    s_cursor,
    s_result,
    slab_keys,
    slab_idx,
    tables,
    counter,
    tidx,
    threads: cutlass.Constexpr,
):
    """Publish this CTA's stage bin-major into its slab segment and arrive; return 1 on the
    last arriver, 0 elsewhere (block-uniform).

    ``s_stage``: the (bits, index) pair stage (``filter_pass``).  ``s_cursor``: 256 Int32
    scratch (per-bin write cursors).  ``slab_keys``/``slab_idx``:
    row base pointers (this CTA writes ``[rank * capacity, ...)``); ``tables``: row base;
    ``counter``: pointer to the row's arrival word.  Two block barriers plus the fence.
    """
    local = s_count[0]
    seg = rank * cutlass.Int32(capacity)
    table = tables + rank * TABLE_WORDS
    if tidx < 32:  # exclusive prefix of the 256 bins: lane l owns bins 8l..8l+7
        mine = cutlass.Int32(0)
        for j in cutlass.range_constexpr(8):
            mine = mine + s_hist[tidx * 8 + j]
        incl = warp_inclusive_scan_add(mine, tidx)
        run = incl - mine
        for j in cutlass.range_constexpr(8):
            s_cursor[tidx * 8 + j] = run
            table[1 + tidx * 8 + j] = run
            run = run + s_hist[tidx * 8 + j]
        if tidx == 31:
            table[1 + 256] = incl
    if tidx == 0:
        staged = local
        if local > cutlass.Int32(capacity):
            staged = cutlass.Int32(-1)  # overflow: the last arriver takes the fallback
        table[0] = staged
    cute.arch.barrier()
    stage_base = s_stage.toint()
    if local <= cutlass.Int32(capacity):
        for t in range(tidx, local, threads):
            bits, idx = stage_pair(stage_base, t)
            b = survivor_bin(elems.value(bits), bar, scale)
            pos = seg + shared_add(s_cursor + b, 1)
            slab_keys[pos] = bits.bitcast(cutlass.Int32)
            slab_idx[pos] = idx
    cute.arch.barrier()
    if tidx == 0:
        # one acq_rel atomic is the whole handshake: release for this CTA's segment, acquire
        # for the last arriver over everyone else's (the block barrier above orders the other
        # threads' slab stores before this thread's release at CTA scope, the release lifts
        # them to GPU scope)
        arrived = global_add_acq_rel(counter, 1)
        s_result[10] = cutlass.Int32(0)
        if arrived == cutlass.Int32(splits - 1):
            s_result[10] = cutlass.Int32(1)
    cute.arch.barrier()
    return s_result[10]


@cute.jit
def _table_word(tables, r, w):
    return load_global_l2_i32((tables + r * TABLE_WORDS + w).toint())


@cute.jit
def _prefix(s_pref, s_staged, r, b):
    """Segment ``r``'s exclusive prefix of bin ``b`` in 0..256 from the shared copies: bins
    0..255 from the table copy, 256 (one past the last bin) is the segment's staged count."""
    v = s_staged[r]
    if b < 256:
        v = s_pref[r * 256 + b]
    return v


@cute.jit
def _slab_source(
    s_scratch,
    offsets: cutlass.Constexpr,
    sources: cutlass.Constexpr,
    g,
    splits: cutlass.Constexpr,
):
    """Slab word of output position ``g`` of a concatenated range: the segment whose output
    offset (``s_scratch[offsets + r]``, ascending) is the last at or below ``g``, plus its
    distance into that segment's source range (``s_scratch[sources + r]``)."""
    r = cutlass.Int32(0)
    for rr in cutlass.range_constexpr(1, splits):
        if g >= s_scratch[offsets + rr]:
            r = cutlass.Int32(rr)
    return s_scratch[sources + r] + (g - s_scratch[offsets + r])


@cute.jit
def merge_slab(
    elems,
    k: cutlass.Constexpr,
    splits: cutlass.Constexpr,
    capacity: cutlass.Constexpr,
    out_row,
    s_scratch,
    s_pref,
    s_merged,
    s_tie_keys,
    s_tie_idx,
    tie_capacity: cutlass.Constexpr,
    ballot_limit: cutlass.Constexpr,
    s_slots,
    s_result,
    slab_keys,
    slab_idx,
    tables,
    counter,
    tidx,
    threads: cutlass.Constexpr,
    telemetry: cutlass.Constexpr = False,
):
    """On the last arriver: merge the row and write its k winners; return 1, or 0 when any
    segment overflowed, the row undershot k, or the ties overflowed the stage.

    ``s_scratch``: >= 768 Int32 (the dead key stage: staged counts, range descriptors, then
    the radix select's histogram); ``s_pref``: >= 256 * splits Int32 (the dead index stage:
    the segments' prefix tables); ``s_merged``: 256 Int32; ``s_result``: 16 Int32.  Resets
    the arrival counter.  Reads slab words with L1-bypassing loads.  Barriers: four plus the
    tie select's.  With ``telemetry`` the sub-phase clocks (gather, ranges, copy, select) land
    in ``s_result[12..15]``.
    """
    mark = cutlass.Int64(0)
    if cutlass.const_expr(telemetry):
        mark = read_clock64()
    # The arrival atomic (acq_rel) was the acquire; the other threads of this block read the
    # slab after the block barrier that followed it, which is enough at CTA scope.
    # Segment prefix tables: thread t loads prefix[t] of every segment (one independent L2
    # load per segment) into shared memory, where both the merged histogram and, after the
    # crossing, the per-segment ranges read them; a second L2 round trip for the ranges
    # measured 0.58 us at 1M b=8.  (An earlier copy of the tables issued 4-8 dependent loads
    # per thread and measured worse; these loads are independent.)
    if (
        tidx < 256
    ):  # prefix[b] of every segment; prefix[256] is the segment's staged count
        for r in cutlass.range_constexpr(splits):
            s_pref[r * 256 + tidx] = _table_word(tables, r, 1 + tidx)
    if tidx < 32:  # per-segment staged counts (lane r) and the overflow / total verdict
        staged = cutlass.Int32(0)
        if tidx < cutlass.Int32(splits):
            staged = _table_word(tables, tidx, 0)
        bad = cutlass.Int32(0)
        if staged < 0:
            bad = cutlass.Int32(1)
            staged = cutlass.Int32(0)
        s_scratch[tidx] = staged
        total = warp_inclusive_scan_add(staged, tidx)
        nbad = warp_inclusive_scan_add(bad, tidx)
        if tidx == 31:
            s_result[8] = total
            s_result[9] = nbad
    cute.arch.barrier()
    if tidx < 256:  # merged histogram: sum over segments of prefix[b+1] - prefix[b]
        total = cutlass.Int32(0)
        for r in cutlass.range_constexpr(splits):
            total = (
                total + _prefix(s_pref, s_scratch, r, tidx + 1) - s_pref[r * 256 + tidx]
            )
        s_merged[tidx] = total
    cute.arch.barrier()
    if cutlass.const_expr(telemetry):
        if tidx == 0:
            s_result[12] = (read_clock64() - mark).to(cutlass.Int32)
        mark = read_clock64()
    ok = cutlass.Int32(1)
    if (s_result[9] != 0) | (s_result[8] < cutlass.Int32(k)):
        ok = cutlass.Int32(0)
    if ok == 1:
        if tidx < 32:
            b, above, _c, _b2, _a2, _c2 = crossing_256_warp(
                s_merged, cutlass.Int32(k), cutlass.Int32(k), tidx, False
            )
            # per-segment ranges: winners [prefix[b+1], staged), ties [prefix[b], prefix[b+1])
            win_src = cutlass.Int32(0)
            win_cnt = cutlass.Int32(0)
            tie_src = cutlass.Int32(0)
            tie_cnt = cutlass.Int32(0)
            if tidx < cutlass.Int32(splits):
                p_b = s_pref[tidx * 256 + b]
                p_b1 = _prefix(s_pref, s_scratch, tidx, b + 1)
                win_src = p_b1
                win_cnt = s_scratch[tidx] - p_b1
                tie_src = p_b
                tie_cnt = p_b1 - p_b
            win_incl = warp_inclusive_scan_add(win_cnt, tidx)
            tie_incl = warp_inclusive_scan_add(tie_cnt, tidx)
            s_scratch[32 + tidx] = (
                win_incl - win_cnt
            )  # output offset of segment r's winners
            s_scratch[64 + tidx] = tidx * cutlass.Int32(capacity) + win_src
            s_scratch[96 + tidx] = (
                tie_incl - tie_cnt
            )  # tie-stage offset of segment r's ties
            s_scratch[128 + tidx] = tidx * cutlass.Int32(capacity) + tie_src
            if tidx == 0:
                s_result[0] = b
                s_result[1] = above
            if tidx == 31:
                s_result[2] = win_incl
                s_result[3] = tie_incl
        cute.arch.barrier()
        if cutlass.const_expr(telemetry):
            if tidx == 0:
                s_result[13] = (read_clock64() - mark).to(cutlass.Int32)
            mark = read_clock64()
        above = s_result[1]
        n_win = s_result[2]
        n_tie = s_result[3]
        # Each thread's first winner and first tie entry are loaded together before either is
        # stored: the two copies are independent L2 round trips (about 0.6 us each on B200) and
        # the winners rarely need more than one step (n_win < k <= threads on the default
        # configurations), so issuing both loads first overlaps them.  Further steps, when the
        # tie bin is wider than the block, follow in the plain loops.
        has_win = tidx < n_win
        has_tie = tidx < n_tie
        win_idx = cutlass.Int32(0)
        tie_key = cutlass.Int32(0)
        tie_idx = cutlass.Int32(0)
        if has_win:
            win_idx = load_global_l2_i32(
                (slab_idx + _slab_source(s_scratch, 32, 64, tidx, splits)).toint()
            )
        if has_tie:
            src = _slab_source(s_scratch, 96, 128, tidx, splits)
            tie_key = load_global_l2_i32((slab_keys + src).toint())
            tie_idx = load_global_l2_i32((slab_idx + src).toint())
        if has_win & (tidx < cutlass.Int32(k)):
            out_row[tidx] = win_idx
        if has_tie & (tidx < cutlass.Int32(tie_capacity)):
            s_tie_keys[tidx] = elems.key(cutlass.Uint32(tie_key))
            s_tie_idx[tidx] = tie_idx
        for g in range(
            tidx + cutlass.Int32(threads), n_win, threads
        ):  # one winner index per thread per step
            if g < cutlass.Int32(k):
                out_row[g] = load_global_l2_i32(
                    (slab_idx + _slab_source(s_scratch, 32, 64, g, splits)).toint()
                )
        for g in range(tidx + cutlass.Int32(threads), n_tie, threads):
            src = _slab_source(s_scratch, 128 - 32, 128, g, splits)
            if g < cutlass.Int32(tie_capacity):
                s_tie_keys[g] = elems.key(
                    cutlass.Uint32(load_global_l2_i32((slab_keys + src).toint()))
                )
                s_tie_idx[g] = load_global_l2_i32((slab_idx + src).toint())
        cute.arch.barrier()
        if cutlass.const_expr(telemetry):
            if tidx == 0:
                s_result[14] = (read_clock64() - mark).to(cutlass.Int32)
            mark = read_clock64()
        ok = _select_ties(
            elems,
            k,
            above,
            n_tie,
            out_row,
            s_scratch + 256,
            s_tie_keys,
            s_tie_idx,
            tie_capacity,
            ballot_limit,
            s_slots,
            s_result,
            tidx,
            threads,
        )
        if cutlass.const_expr(telemetry):
            if tidx == 0:
                s_result[15] = (read_clock64() - mark).to(cutlass.Int32)
    if tidx == 0:
        # a plain store: nobody reads the counter again in this launch, and the kernel boundary
        # orders it before the next one.  The release store used before waited for every
        # outstanding output store to drain first: 1.0 us per row at 1M b=8.
        counter[0] = cutlass.Int32(0)
    return ok
