"""Exact select over a whole row: the fallback every other path relies on (kernel 5).

Entry points:

* ``exact_select_row``: the census first (two vector passes; it emits everything above the
  rank-k coarse bin and stages the bin's ties), then, only when the bin held more ties than
  the stage, the radix select restricted to the bin's exact key range, which the census
  gathered in the same pass that would have staged.  This is the path for constant, two-value
  and one-bin rows, where the census always overflows.
* ``radix_select_in_range``: the radix select over the elements whose exact key lies in a
  known range, for a kernel that found the bin from registers and can produce the range there.
* ``radix_select_row``: the whole row, range found by one extra pass.

Members are tested by exact key range, not by recomputing the coarse bin: every element whose
key lies between the smallest and largest key of a coarse bin's members is in that bin (the
binning is monotone), and a key compare costs a few integer instructions where the bin costs a
quarter-rate float conversion.  The radix select first checks for equal extremes (every member
an exact tie: fill any ``need`` of them in one pass, no digit round), otherwise starts the
digit rounds at the first byte where the extremes differ.  Each round histograms the current
digit of the members whose higher digits match the prefix, takes the crossing at the remaining
rank, extends the prefix; the final pass emits every member above the k-th key and fills the
rest from the members equal to it.  All passes stream four vectors per iteration.
"""

import cutlass
import cutlass.cute as cute

from ...block.crossing import crossing_256_block
from ...block.reduce import block_exclusive_scan_i32, block_max_min_u32
from ...device.atomics import shared_add, shared_count
from ...device.timers import read_clock64

from .census import census_select_row, element_bits
from .row_scan import index_of_element, load_quad, quad_stride

__all__ = ["exact_select_row", "radix_select_in_range", "radix_select_row"]


@cute.jit
def _radix_in_range(
    elems,
    row_ptr,
    length,
    need,
    out_row,
    kmin,
    kmax,
    s_hist,
    s_slots,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
    telemetry: cutlass.Constexpr = False,
):
    """Write the ``need`` largest elements with exact key in [kmin, kmax] to ``out_row[0 ..
    need)``.  Precondition: at least ``need`` such elements.  ``s_hist``: 256 Int32;
    ``s_slots``: 8 Int32; ``s_result``: 5 Int32 (``[14]`` gets the clock with telemetry).
    Vector passes: one (all keys equal) or one per differing key byte plus one.
    """
    mark = cutlass.Int64(0)
    if cutlass.const_expr(telemetry):
        mark = read_clock64()
    elements = cutlass.const_expr(16 * elems.per_word)  # per 16-word quad
    n_vectors = (length + cutlass.Int32(elems.per_vector - 1)) >> cutlass.Int32(
        elems.log2_per_vector
    )
    last_vector = n_vectors - 1
    stride = cutlass.const_expr(quad_stride(threads))
    base = row_ptr.toint()

    if kmin == kmax:
        # every member is an exact tie: any ``need`` of them, positions from one block scan
        mine = cutlass.Int32(0)
        for v in range(tidx, n_vectors, stride):
            words = load_quad(base, v, threads, last_vector)
            for e in cutlass.range_constexpr(elements):
                key = elems.key(element_bits(elems, words, e))
                mine = mine + (
                    cutlass.Int32(key == kmin)
                    & cutlass.Int32(index_of_element(elems, v, threads, e) < length)
                )
        pos, _total = block_exclusive_scan_i32(mine, s_slots, tidx, threads)
        if pos < need:  # threads whose whole share lies past ``need`` skip the pass
            for v in range(tidx, n_vectors, stride):
                words = load_quad(base, v, threads, last_vector)
                for e in cutlass.range_constexpr(elements):
                    idx = index_of_element(elems, v, threads, e)
                    if (elems.key(element_bits(elems, words, e)) == kmin) & (
                        idx < length
                    ):
                        if pos < need:
                            out_row[pos] = idx
                        pos = pos + 1
    else:
        rounds = cutlass.const_expr(len(elems.key_shifts))
        prefix = cutlass.Uint32(0)
        prefix_mask = cutlass.Uint32(0)
        remaining = need
        in_play = cutlass.Int32(0)
        first = cutlass.Int32(1)
        hist_base = s_hist.toint()
        for r in cutlass.range_constexpr(rounds):
            shift = cutlass.const_expr(elems.key_shifts[r])
            digit_mask = cutlass.Uint32(0xFF) << cutlass.Uint32(shift)
            same = ((kmin ^ kmax) & digit_mask) == cutlass.Uint32(0)
            if same & (first == 1):
                # both extremes agree on this byte, so every member does: the prefix grows for free
                prefix = prefix | (kmin & digit_mask)
                prefix_mask = prefix_mask | digit_mask
            else:
                if tidx < 256:
                    s_hist[tidx] = cutlass.Int32(0)
                cute.arch.barrier()
                for v in range(tidx, n_vectors, stride):
                    words = load_quad(base, v, threads, last_vector)
                    for e in cutlass.range_constexpr(elements):
                        key = elems.key(element_bits(elems, words, e))
                        member = (
                            (key >= kmin)
                            & (key <= kmax)
                            & (index_of_element(elems, v, threads, e) < length)
                        )
                        if member & ((key & prefix_mask) == prefix):
                            shared_count(
                                hist_base
                                + (
                                    (key >> cutlass.Uint32(shift))
                                    & cutlass.Uint32(0xFF)
                                ).to(cutlass.Int32)
                                * 4
                            )
                cute.arch.barrier()
                if first == 1:  # the first histogram's total is the member count
                    total = cutlass.Int32(0)
                    for b in cutlass.range_constexpr(0, 256, 4):
                        total = (
                            total
                            + s_hist[b]
                            + s_hist[b + 1]
                            + s_hist[b + 2]
                            + s_hist[b + 3]
                        )
                    in_play = total
                    first = cutlass.Int32(0)
                crossing_256_block(s_hist, in_play, remaining, s_slots, s_result, tidx)
                bucket = s_result[0]
                prefix = prefix | (cutlass.Uint32(bucket) << cutlass.Uint32(shift))
                prefix_mask = prefix_mask | digit_mask
                remaining = remaining - s_result[1]
                in_play = s_result[2]
        if tidx == 0:
            s_result[3] = cutlass.Int32(0)  # members above the k-th key
            s_result[4] = cutlass.Int32(0)  # members equal to it
        cute.arch.barrier()
        winners_above = need - remaining
        for v in range(tidx, n_vectors, stride):
            words = load_quad(base, v, threads, last_vector)
            for e in cutlass.range_constexpr(elements):
                idx = index_of_element(elems, v, threads, e)
                key = elems.key(element_bits(elems, words, e))
                if (key >= kmin) & (key <= kmax) & (idx < length):
                    if key > prefix:
                        out_row[shared_add(s_result + 3, 1)] = idx
                    else:
                        if key == prefix:
                            c = shared_add(s_result + 4, 1)
                            if c < remaining:
                                out_row[winners_above + c] = idx
    if cutlass.const_expr(telemetry):
        cute.arch.barrier()
        if tidx == 0:
            s_result[14] = (read_clock64() - mark).to(cutlass.Int32)


@cute.jit
def _key_range(elems, row_ptr, length, s_slots_u32, tidx, threads: cutlass.Constexpr):
    """Minimum and maximum exact key over the row (one pass, one block reduction)."""
    elements = cutlass.const_expr(16 * elems.per_word)
    n_vectors = (length + cutlass.Int32(elems.per_vector - 1)) >> cutlass.Int32(
        elems.log2_per_vector
    )
    last_vector = n_vectors - 1
    stride = cutlass.const_expr(quad_stride(threads))
    base = row_ptr.toint()
    kmax = cutlass.Uint32(0)
    kmin = cutlass.Uint32(0xFFFFFFFF)
    for v in range(tidx, n_vectors, stride):
        words = load_quad(base, v, threads, last_vector)
        for e in cutlass.range_constexpr(elements):
            if index_of_element(elems, v, threads, e) < length:
                key = elems.key(element_bits(elems, words, e))
                if key > kmax:
                    kmax = key
                if key < kmin:
                    kmin = key
    kmax, kmin = block_max_min_u32(kmax, kmin, s_slots_u32, tidx, threads)
    return kmin, kmax


@cute.jit
def exact_select_row(
    elems,
    row_ptr,
    length,
    k: cutlass.Constexpr,
    out_row,
    s_bins,
    s_tie_keys,
    s_tie_idx,
    tie_capacity: cutlass.Constexpr,
    ballot_limit: cutlass.Constexpr,
    s_slots,
    s_slots_u32,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
    check_constant: cutlass.Constexpr = True,
    telemetry: cutlass.Constexpr = False,
):
    """Write the k winners of ``row[0, length)``: with ``check_constant`` one pass for the
    row's key range first (a constant row is finished right there), then the census, then the
    radix select over the rank-k bin's key range if the bin overflowed the tie stage.  Returns
    the arm taken (1 census, 3 radix or constant).

    ``check_constant`` is for callers reaching this as a fallback (degenerate sample, tie
    overflow), where constant rows are the common case and one pass saves the census's worst
    case (every element into one bin: 64K same-address atomics, 13 us).  A short-row arm on
    ordinary data leaves it off; the pass cost it 8% at 16K b=64.

    Precondition: ``k < length``.  ``s_bins``: 4096 Int32 (the census histogram; its first 256
    words become the radix histogram); ``s_slots``: warps Int32; ``s_slots_u32``: 2 x warps
    Uint32; ``s_result``: 8 Int32 (16 with telemetry: ``[13]`` range pass, ``[12]`` census,
    ``[14]`` radix).  All threads call it.
    """
    mark = cutlass.Int64(0)
    if cutlass.const_expr(telemetry):
        mark = read_clock64()
    kmin = cutlass.Uint32(0)
    kmax = cutlass.Uint32(1)
    if cutlass.const_expr(check_constant):
        kmin, kmax = _key_range(elems, row_ptr, length, s_slots_u32, tidx, threads)
    if cutlass.const_expr(telemetry):
        if tidx == 0:
            s_result[13] = (read_clock64() - mark).to(cutlass.Int32)
            s_result[12] = cutlass.Int32(0)
            s_result[14] = cutlass.Int32(0)
        mark = read_clock64()
    ok = cutlass.Int32(1)
    arm = cutlass.Int32(3)
    if kmin == kmax:
        for i in range(tidx, k, threads):  # all equal: any k indices are exact
            out_row[i] = cutlass.Int32(i)
    else:
        ok = census_select_row(
            elems,
            row_ptr,
            length,
            k,
            out_row,
            s_bins,
            s_tie_keys,
            s_tie_idx,
            tie_capacity,
            ballot_limit,
            s_slots,
            s_slots_u32,
            s_result,
            tidx,
            threads,
        )
        arm = cutlass.Int32(1)
        if cutlass.const_expr(telemetry):
            if tidx == 0:
                s_result[12] = (read_clock64() - mark).to(cutlass.Int32)
            mark = read_clock64()
    if ok == 0:
        above = s_result[1]
        kmin = cutlass.Uint32(s_result[3])
        kmax = cutlass.Uint32(s_result[4])
        cute.arch.barrier()  # everyone has read the census results before the radix reuses them
        _radix_in_range(
            elems,
            row_ptr,
            length,
            cutlass.Int32(k) - above,
            out_row + above,
            kmin,
            kmax,
            s_bins,
            s_slots,
            s_result,
            tidx,
            threads,
            telemetry,
        )
        arm = cutlass.Int32(3)
    return arm


@cute.jit
def radix_select_in_range(
    elems,
    row_ptr,
    length,
    need,
    out_row,
    kmin,
    kmax,
    s_hist,
    s_slots,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
):
    """Write the ``need`` largest elements of ``row[0, length)`` with exact key in [kmin, kmax]
    to ``out_row[0 .. need)``, for a kernel whose own histogram found the bin, produced the
    members' key range, and already wrote everything above it.  Scratch as
    ``radix_select_row``."""
    _radix_in_range(
        elems,
        row_ptr,
        length,
        need,
        out_row,
        kmin,
        kmax,
        s_hist,
        s_slots,
        s_result,
        tidx,
        threads,
    )


@cute.jit
def radix_select_row(
    elems,
    row_ptr,
    length,
    k: cutlass.Constexpr,
    out_row,
    s_hist,
    s_slots,
    s_slots_u32,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
):
    """Write the indices of the k largest of ``row[0, length)`` to ``out_row`` by the radix
    select over the whole row (no census).  Precondition: ``k < length``.  ``s_hist``: 256
    Int32; ``s_slots``: 8 Int32; ``s_slots_u32``: 2 x warps Uint32; ``s_result``: 5 Int32."""
    kmin, kmax = _key_range(elems, row_ptr, length, s_slots_u32, tidx, threads)
    _radix_in_range(
        elems,
        row_ptr,
        length,
        cutlass.Int32(k),
        out_row,
        kmin,
        kmax,
        s_hist,
        s_slots,
        s_result,
        tidx,
        threads,
    )
