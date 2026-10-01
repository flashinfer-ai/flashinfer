"""Sample phase: from ``threads * vectors`` vectors of the row to a walk threshold, in three
barriers.

1. Every thread loads one or two 16-byte vectors at a row-uniform stride and folds their keys
   to a maximum and a minimum in registers; the block reduces both (barrier 1, which also
   publishes the zeroed histogram).
2. The sampled values are binned into 256 equal-width bins over [min, max] and counted
   (barrier 2).
3. Warp 0 finds the crossing bins of two ranks scaled from the aim and its floor and zeroes
   the histogram on the way out for the filter pass (barrier 3).

The threshold is the lower edge of the aim's crossing bin; the survivor scale maps
[bar, max * span_ext] onto 255 bins so the ~N/4096 elements above the sample maximum do not
all pile into bin 255 (one such pile-up turned a 53 us batch into 320 us before the extension).

Two vectors per thread (``sample_vectors=2``) double the sample and halve the variance of the
survivor count around the aim: the survivor count above a sample-derived bar is roughly
normal with variance ``aim * length / samples``, so at 1M fp32 rows with a 4096 sample the
spread is +-1250 survivors per sigma against a stage of 8192 (parts with a 99 KB carveout), and
one row in ten overflowed and re-walked (L40S 1M b=142 k=2048: median 1006 us, phase sum 847).
The second vector is the pick's neighbour (same 32-byte sector, no extra memory traffic) and
costs eight bin increments per thread.

Replica-deterministic: every CTA of a split row samples the same positions and gets the same
threshold, so split verdicts agree without communication.

Ragged rows (v0.1.24): the probe is issued for the full row buffer before the length is known,
and a shorter row keeps it, masking the vectors that fall past its end, instead of re-sampling
at length-derived positions.  The re-sample was one dependent global round trip on every
ragged row: B200 64K b=256 k=2048, every row four elements short of full (the same work as
full rows), 16.8 -> 17.8 us; the ledger's ragged batch, whose longest rows set its time, sat
at that 17.6 against gvr_2's 15.7 while a batch of full rows took 16.8.  The masked sample is
smaller for shorter rows (half the vectors at half the length); ``valid_samples`` gives the
count the aim and the ranks scale by, and the count-crossing verdict keeps the result exact
whatever the sample.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute

from ...block.crossing import crossing_256_warp, crossing_wide_pair
from ...block.reduce import block_max_min_u32
from ...device.atomics import shared_count
from ...device.memory import load_global_readonly_16
from ...device.timers import read_clock64

__all__ = ["Threshold", "sample_probe", "sample_threshold", "valid_samples"]


class Threshold(NamedTuple):
    """Output of the sample phase.  ``degenerate`` is 1 when the sample cannot give a
    threshold (constant, NaN or infinite span); the caller then takes the exact fallback."""

    bar: object
    scale: object
    floor_bar: object
    floor_scale: object
    degenerate: object
    sample_max: object


@cute.jit
def _word(w0, w1, w2, w3, j: cutlass.Constexpr):
    return (w0, w1, w2, w3)[j]


@cute.jit
def _scaled_rank(aim, samples, length):
    """Rank among the ``samples`` sampled elements corresponding to ``aim`` survivors in the
    row, in [1, samples]."""
    # 32-bit: aim <= 3/4 of the stage x splits (< 2^18) and samples <= 16K, so the product fits
    r = (cutlass.Int32(aim) * samples) // length
    if r < 1:
        r = cutlass.Int32(1)
    if r > samples:
        r = samples
    return r


@cute.jit
def _pick(length, slot, slots: cutlass.Constexpr, log2_per_vector: cutlass.Constexpr):
    """Sample vector ``slot`` of ``slots`` in a row of ``length`` elements: a row-uniform stride."""
    n_vectors = length >> cutlass.Int32(log2_per_vector)
    # 32-bit: slots <= 1024 and rows hold at most 2^18 vectors (1M fp32 / 2M 16-bit), so the
    # product stays below 2^28 (a 64-bit divide here cost about 0.1 us per CTA on Rubin)
    pick = (slot * n_vectors) // cutlass.Int32(slots)
    if pick > n_vectors - 1:
        pick = n_vectors - 1
    return pick


@cute.jit
def _probe_pick(
    length,
    tidx,
    threads: cutlass.Constexpr,
    vectors: cutlass.Constexpr,
    log2_per_vector: cutlass.Constexpr,
):
    """First of the thread's ``vectors`` adjacent sample vectors in a row of ``length``."""
    pick = _pick(length, tidx, threads, log2_per_vector)
    n_vectors = length >> cutlass.Int32(log2_per_vector)
    if pick > n_vectors - cutlass.Int32(vectors):
        pick = n_vectors - cutlass.Int32(vectors)
    if pick < 0:
        pick = cutlass.Int32(0)
    return pick


@cute.jit
def _load_probe(
    elems, row_ptr, length, tidx, threads: cutlass.Constexpr, vectors: cutlass.Constexpr
):
    # the extra vectors are the ones ADJACENT to the thread's pick: a 16-byte load already
    # fetches a 32-byte sector, so the neighbour costs no memory traffic (a second strided
    # vector doubled the sample phase: L40S 1M b=142 21.8 -> 33.6 us, B200 64K b=8 2.1 -> 2.9)
    pick = _probe_pick(length, tidx, threads, vectors, elems.log2_per_vector)
    out: list = []
    for v in cutlass.range_constexpr(vectors):
        out.append(
            load_global_readonly_16(row_ptr.toint() + cutlass.Int64(pick + v) * 16)
        )
    return tuple(out)


@cute.jit
def valid_samples(
    n_cols: cutlass.Constexpr,
    length,
    threads: cutlass.Constexpr,
    vectors: cutlass.Constexpr,
    log2_per_vector: cutlass.Constexpr,
):
    """Elements of the full-row probe (``sample_probe`` over ``n_cols``) that lie inside a row
    of ``length``: for each of the ``vectors`` adjacent vectors, the threads whose pick plus
    its offset is below the row's vector count.  Picks are ``floor(t * NV / threads)`` clamped
    to ``NV - vectors``, so the valid threads for offset v are ``min(threads, ceil(m * threads
    / NV))`` with ``m = row vectors - v``, and every thread once ``m`` exceeds the clamp."""
    per_vector = cutlass.const_expr(1 << log2_per_vector)
    nv = cutlass.const_expr(n_cols >> log2_per_vector)
    n_row = length >> cutlass.Int32(log2_per_vector)
    total = cutlass.Int32(0)
    for v in cutlass.range_constexpr(vectors):
        m = n_row - cutlass.Int32(v)
        count = cutlass.Int32(0)
        if m > cutlass.Int32(nv - vectors):
            count = cutlass.Int32(threads)
        elif m > 0:
            count = (
                m * cutlass.Int32(threads) + cutlass.Int32(nv - 1)
            ) // cutlass.Int32(nv)  # < 2^28, see _pick
            if count > cutlass.Int32(threads):
                count = cutlass.Int32(threads)
        total = total + count
    return total * cutlass.Int32(per_vector)


@cute.jit
def sample_probe(
    elems,
    row_ptr,
    n_cols: cutlass.Constexpr,
    tidx,
    threads: cutlass.Constexpr,
    vectors: cutlass.Constexpr = 1,
):
    """The thread's ``vectors`` sample vectors for a full-length row, loaded before the row's
    length is known.  Every vector of the row buffer is in-bounds memory whatever the length,
    so the kernel issues these loads first and reads the length while they are in flight; a
    row shorter than the buffer re-samples inside ``sample_threshold`` (extra loads, ragged
    rows only).  Measured: the dependent length load ahead of the sample cost about 0.4 us per
    row (B200 64K b=8: 7.87 -> 7.46 us)."""
    return _load_probe(elems, row_ptr, cutlass.Int32(n_cols), tidx, threads, vectors)


@cute.jit
def _fold_max_min(elems, vec, kmax, kmin):
    w0, w1, w2, w3 = vec
    for j in cutlass.range_constexpr(4):
        for h in cutlass.range_constexpr(elems.per_word):
            key = elems.key(elems.bits(_word(w0, w1, w2, w3, j), h))
            if key > kmax:
                kmax = key
            if key < kmin:
                kmin = key
    return kmax, kmin


@cute.jit
def _bin_vector(elems, vec, smin, to_bin, hist_base, bins: cutlass.Constexpr = 256):
    w0, w1, w2, w3 = vec
    for j in cutlass.range_constexpr(4):
        for h in cutlass.range_constexpr(elems.per_word):
            v = elems.value(elems.bits(_word(w0, w1, w2, w3, j), h))
            b = ((v - smin) * to_bin).to(cutlass.Int32)
            if b < 0:
                b = cutlass.Int32(0)
            if b > cutlass.Int32(bins - 1):
                b = cutlass.Int32(bins - 1)
            shared_count(hist_base + b * 4)


@cute.jit
def sample_threshold(
    elems,
    row_ptr,
    length,
    n_cols: cutlass.Constexpr,
    probe,
    aim,
    floor_multiple: cutlass.Constexpr,
    span_ext: cutlass.Constexpr,
    s_hist,
    s_slots,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
    telemetry: cutlass.Constexpr = False,
    vectors: cutlass.Constexpr = 1,
    bins: cutlass.Constexpr = 256,
    s_wide=None,
    s_slots_i32=None,
    probe_stale=None,
    samples=None,
):
    """Sample the row and return ``(bar, scale, floor_bar, floor_scale, degenerate, max)``.

    Contract: ``length >= per_vector``; ``probe`` is what ``sample_probe`` returned for this
    thread with the same ``vectors`` over the full row buffer (its vectors past ``length`` are
    masked, not reloaded; ``samples`` is ``valid_samples`` for that, or the full count after a
    reload for another row); ``s_hist`` (256
    Int32, 16-byte aligned) is zero on exit; ``s_slots`` (2 * warps Uint32) and ``s_result``
    (16 Int32) are scratch.  All threads call it.  Cost: ``vectors`` loads per thread (already
    in flight for full rows), three barriers, one warp crossing.  With ``telemetry`` the three
    sub-phase clocks (load+fold+barrier 1, bin+barrier 2, crossing+barrier 3) land in
    ``s_result[12..14]``.  (Privatizing the sample histogram into 2 or 4 lane-interleaved
    copies measured no gain on B200, L40S or RTX 5080: the increments are not contention
    bound; see docs/measured-worse.md.)

    ``bins`` above 256: the sample histogram lives in ``s_wide`` (``bins`` Int32, a dead
    stage) and the crossing is the block-wide ``crossing_wide_pair`` (``s_slots_i32``: warps
    Int32), two barriers instead of one warp crossing.  Finer bins quantize the threshold
    finer: the bar is a bin's lower edge, so the survivors include everything in the crossing
    bin, up to one bin's population above the aim.
    """
    mark = cutlass.Int64(0)
    if cutlass.const_expr(telemetry):
        mark = read_clock64()
    per_vector = cutlass.const_expr(elems.per_vector)
    full_samples = cutlass.const_expr(threads * per_vector * vectors)
    wide = cutlass.const_expr(bins > 256)
    vecs = probe
    # the probe's vectors that lie inside this row: all of them for a full row (the common case
    # pays nothing here) or for a permuted CTA's fresh sample; a shorter row masks the vectors
    # past its end.  ``valid`` is a per-thread bit mask over the probe's vectors.
    valid = cutlass.Int32((1 << vectors) - 1)
    if cutlass.const_expr(samples is None):
        samples = cutlass.Int32(full_samples)
    stale = cutlass.Int32(0)
    if cutlass.const_expr(probe_stale is not None):
        stale = probe_stale
    if (
        stale != 0
    ):  # the probe was issued for another row (a permuted CTA): sample this one
        vecs = _load_probe(elems, row_ptr, length, tidx, threads, vectors)
        samples = cutlass.Int32(full_samples)
    elif length != cutlass.Int32(n_cols):
        pick = _probe_pick(
            cutlass.Int32(n_cols), tidx, threads, vectors, elems.log2_per_vector
        )
        n_row_vectors = length >> cutlass.Int32(elems.log2_per_vector)
        valid = cutlass.Int32(0)
        for v in cutlass.range_constexpr(vectors):
            if pick + cutlass.Int32(v) < n_row_vectors:
                valid = valid | cutlass.Int32(1 << v)
        if cutlass.const_expr(samples is None):
            samples = valid_samples(
                n_cols, length, threads, vectors, elems.log2_per_vector
            )

    # (the neutral pair for the block reduction is (0, 0xFFFFFFFF))
    kmax = cutlass.Uint32(0)
    kmin = cutlass.Uint32(0xFFFFFFFF)
    for v in cutlass.range_constexpr(vectors):
        if (valid & cutlass.Int32(1 << v)) != 0:
            kmax, kmin = _fold_max_min(elems, vecs[v], kmax, kmin)
    if tidx < 256:
        s_hist[tidx] = cutlass.Int32(0)
    if cutlass.const_expr(wide):
        for i in range(tidx, bins, threads):
            s_wide[i] = cutlass.Int32(0)
    kmax, kmin = block_max_min_u32(kmax, kmin, s_slots, tidx, threads)  # barrier 1
    if cutlass.const_expr(telemetry):
        if tidx == 0:
            s_result[12] = (read_clock64() - mark).to(cutlass.Int32)
        mark = read_clock64()

    smin = elems.value_of_key(kmin)
    smax = elems.value_of_key(kmax)
    span = smax - smin
    degenerate = cutlass.Int32(1)
    if (span > cutlass.Float32(0.0)) & (
        span <= cutlass.Float32(3.0e38)
    ):  # finite, non-empty
        degenerate = cutlass.Int32(0)
    if degenerate == 0:
        to_bin = cutlass.Float32(float(bins)) / span
        if cutlass.const_expr(wide):
            hist_base = s_wide.toint()
        else:
            hist_base = s_hist.toint()
        for v in cutlass.range_constexpr(vectors):
            if (valid & cutlass.Int32(1 << v)) != 0:
                _bin_vector(elems, vecs[v], smin, to_bin, hist_base, bins)
    cute.arch.barrier()  # barrier 2
    if cutlass.const_expr(telemetry):
        if tidx == 0:
            s_result[13] = (read_clock64() - mark).to(cutlass.Int32)
        mark = read_clock64()

    rank_aim = _scaled_rank(aim, samples, length)
    rank_floor = rank_aim * cutlass.Int32(floor_multiple)
    if rank_floor > samples:
        rank_floor = samples
    if cutlass.const_expr(wide):
        # block-wide crossing over the wide histogram: publishes (bin, above, count) for the
        # aim at s_result[0..2] and for the floor at [3..5], ends with a barrier
        crossing_wide_pair(
            s_wide, bins, rank_aim, rank_floor, s_slots_i32, s_result, tidx, threads
        )
        bin_aim_w = s_result[0]
        bin_floor_w = s_result[3]
    else:
        if tidx < 32:
            bin_aim, _a, _c, bin_floor, _a2, _c2 = crossing_256_warp(
                s_hist, rank_aim, rank_floor, tidx, True
            )
            if tidx == 0:
                s_result[0] = bin_aim
                s_result[1] = bin_floor
        cute.arch.barrier()  # barrier 3: bins published, histogram zeroed
        bin_aim_w = s_result[0]
        bin_floor_w = s_result[1]
    if cutlass.const_expr(telemetry):
        if tidx == 0:
            s_result[14] = (read_clock64() - mark).to(cutlass.Int32)

    bin_width = span / cutlass.Float32(float(bins))
    bar = smin + bin_aim_w.to(cutlass.Float32) * bin_width
    floor_bar = smin + bin_floor_w.to(cutlass.Float32) * bin_width
    scale = cutlass.Float32(0.0)
    span_above = (smax - bar) * cutlass.Float32(span_ext)
    if span_above > cutlass.Float32(0.0):
        scale = cutlass.Float32(255.0) / span_above
    floor_scale = cutlass.Float32(0.0)
    span_floor = (smax - floor_bar) * cutlass.Float32(span_ext)
    if span_floor > cutlass.Float32(0.0):
        floor_scale = cutlass.Float32(255.0) / span_floor
    return bar, scale, floor_bar, floor_scale, degenerate, smax
