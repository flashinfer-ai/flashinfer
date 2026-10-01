"""Census select: two passes over a short row through a 4096-bin coarse histogram.

For rows of a few thousand to a few tens of thousands of elements the sampled pipeline's fixed
cost (three-barrier sample, aim, repair rungs) is most of the time, and the exact radix select
pays five passes.  The census pays two: count every element into a coarse bin, find the rank-k
bin, then emit everything above it and stage the bin's members for the exact tie select.  It
is also the first thing the fallback tries, because it fails cheaply: a bin holding more ties
than the stage (constant rows, one-bin rows) is known after the crossing; the second pass then
skips staging and instead records the members' exact key range, which is all the radix select
needs to take over.

Coarse bin: the top 12 bits of the fp16 ordered key of the value (sign, exponent, 6 mantissa
bits: 64 bins per octave).  fp32 values are rounded to fp16 first; the rounding is monotone, so
bins keep the order and the exact 32-bit keys settle the bin's ties.  An 8-bit top-byte
histogram was tried in FlashInfer's kernel and measured worse: it is an exponent histogram, so
randn rows put thousands of ties in the crossing bin and always overflowed.

Bins are computed two elements at a time (one ``cvt.rn.f16x2`` per fp32 pair, the twiddle on
both halves at once); a scalar conversion per element is a quarter-rate instruction and made
each pass over 64K elements cost 6.5 us.  Both passes stream the row four vectors per
iteration (``row_scan``).
"""

import cutlass
import cutlass.cute as cute

from ...block.crossing import crossing_wide_pair
from ...block.reduce import block_max_min_u32
from ...device.atomics import shared_add, shared_count
from ...device.keys import f16x2_of_f32_pair, key_of_f16_bits, key_of_f16x2_bits

from .resolve import _select_ties
from .row_scan import index_of_element, load_quad, quad_stride

__all__ = [
    "COARSE_BINS",
    "coarse_bin",
    "pair_key16",
    "pair_coarse_bins",
    "census_select_row",
]

COARSE_BINS = 4096


@cute.jit
def coarse_bin(elems, bits):
    """Coarse bin in [0, 4096) of an element bit pattern; monotone in the value, NaN on top."""
    if cutlass.const_expr(elems.dtype is cutlass.Float16):
        key16 = key_of_f16_bits(bits)
    else:
        half = elems.value(bits).to(cutlass.Float16)
        key16 = key_of_f16_bits(half.bitcast(cutlass.Uint16).to(cutlass.Uint32))
    return (key16 >> cutlass.Uint32(4)).to(cutlass.Int32)


@cute.jit
def pair_key16(elems, words, p: cutlass.Constexpr):
    """Packed 16-bit ordered keys of elements 2p and 2p+1 of a word tuple (low half first).

    fp32: two words through one paired conversion; bf16: the two halves widened by a shift and
    paired; fp16: the word itself.  Then the twiddle on both halves in 32-bit arithmetic.
    """
    if cutlass.const_expr(elems.is_f32):
        packed = f16x2_of_f32_pair(words[2 * p], words[2 * p + 1])
    elif cutlass.const_expr(elems.is_bf16):
        w = words[p]
        packed = f16x2_of_f32_pair(
            w << cutlass.Uint32(16), w & cutlass.Uint32(0xFFFF0000)
        )
    else:
        packed = words[p]
    return key_of_f16x2_bits(packed)


@cute.jit
def pair_coarse_bins(elems, words, p: cutlass.Constexpr):
    """Coarse bins (Int32) of elements 2p and 2p+1 of a word tuple."""
    bins = (pair_key16(elems, words, p) >> cutlass.Uint32(4)) & cutlass.Uint32(
        0x0FFF0FFF
    )
    return (bins & cutlass.Uint32(0xFFFF)).to(cutlass.Int32), (
        bins >> cutlass.Uint32(16)
    ).to(cutlass.Int32)


@cute.jit
def element_bits(elems, words, e: cutlass.Constexpr):
    """Bit pattern of element ``e`` of a word tuple (fp32: word e; 16-bit: half e % 2 of word e // 2)."""
    if cutlass.const_expr(elems.is_f32):
        return words[e]
    else:
        return elems.bits(words[e // 2], e % 2)


@cute.jit
def census_select_row(
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
):
    """Write the k winners of ``row[0, length)``; return 1, or 0 if the rank-k bin held more
    ties than the tie stage.  In that case the winners above the bin are already written,
    ``s_result[0..1]`` hold the bin and the count above it, and ``s_result[3..4]`` hold the
    minimum and maximum exact key (as Int32 bit patterns) of the bin's members, so the caller's
    radix select can take the bin by key range.

    Precondition: ``k < length``.  ``s_bins``: 4096 Int32 (dead on exit); ``s_slots``: warps
    Int32; ``s_slots_u32``: 2 x warps Uint32; ``s_result``: 8 Int32.  Two passes over the
    row, one wide crossing, the tie select's cost.  Barriers: five plus the crossing's two.
    """
    pairs = cutlass.const_expr(8 * elems.per_word)  # element pairs per 16-word quad
    n_vectors = (length + cutlass.Int32(elems.per_vector - 1)) >> cutlass.Int32(
        elems.log2_per_vector
    )
    last_vector = n_vectors - 1
    stride = cutlass.const_expr(quad_stride(threads))
    base = row_ptr.toint()
    bins_base = s_bins.toint()

    for i in range(tidx, COARSE_BINS, threads):
        s_bins[i] = cutlass.Int32(0)
    if tidx == 0:
        s_result[6] = cutlass.Int32(0)  # winner cursor
        s_result[7] = cutlass.Int32(0)  # tie cursor
    cute.arch.barrier()
    # plain increments: a warp-aggregated count (match.any, one atomic per group) measured
    # worse on every input (docs/measured-worse.md), so low-entropy rows pay same-address
    # atomics here and the fallback catches the constant case before reaching this
    for v in range(tidx, n_vectors, stride):
        words = load_quad(base, v, threads, last_vector)
        for p in cutlass.range_constexpr(pairs):
            lo, hi = pair_coarse_bins(elems, words, p)
            if index_of_element(elems, v, threads, 2 * p) < length:
                shared_count(bins_base + lo * 4)
            if index_of_element(elems, v, threads, 2 * p + 1) < length:
                shared_count(bins_base + hi * 4)
    cute.arch.barrier()
    crossing_wide_pair(
        s_bins,
        COARSE_BINS,
        cutlass.Int32(k),
        cutlass.Int32(k),
        s_slots,
        s_result,
        tidx,
        threads,
    )
    cut_bin = s_result[0]
    above = s_result[1]
    in_bin = s_result[2]
    # when the bin cannot fit the stage, staging is pointless and its one-address atomics
    # (one per member) are skipped; the members' exact key range is gathered instead
    stage_ties = cutlass.Int32(in_bin <= cutlass.Int32(tie_capacity))
    kmax = cutlass.Uint32(0)
    kmin = cutlass.Uint32(0xFFFFFFFF)

    for v in range(tidx, n_vectors, stride):
        words = load_quad(base, v, threads, last_vector)
        for p in cutlass.range_constexpr(pairs):
            lo, hi = pair_coarse_bins(elems, words, p)
            for side in cutlass.range_constexpr(2):
                b = hi
                if cutlass.const_expr(side == 0):
                    b = lo
                e = cutlass.const_expr(2 * p + side)
                idx = index_of_element(elems, v, threads, e)
                if idx < length:
                    if b > cut_bin:
                        out_row[shared_add(s_result + 6, 1)] = idx
                    else:
                        if b == cut_bin:
                            key = elems.key(element_bits(elems, words, e))
                            if stage_ties == 1:
                                t = shared_add(s_result + 7, 1)
                                s_tie_keys[t] = key
                                s_tie_idx[t] = idx
                            else:
                                if key > kmax:
                                    kmax = key
                                if key < kmin:
                                    kmin = key
    cute.arch.barrier()
    if stage_ties == 0:
        kmax, kmin = block_max_min_u32(kmax, kmin, s_slots_u32, tidx, threads)
        if tidx == 0:
            s_result[3] = kmin.bitcast(cutlass.Int32)
            s_result[4] = kmax.bitcast(cutlass.Int32)
        cute.arch.barrier()
    return _select_ties(
        elems,
        k,
        above,
        in_bin,
        out_row,
        s_bins,
        s_tie_keys,
        s_tie_idx,
        tie_capacity,
        ballot_limit,
        s_slots,
        s_result,
        tidx,
        threads,
    )
