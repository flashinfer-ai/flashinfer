"""Register-resident row: load once, histogram from registers, classify from registers.

For rows that fit the block's registers the whole row is read from memory exactly once.  Each
thread keeps ``words`` 32-bit words (4, 8 or 16: one to four 16-byte vectors) in registers
and every later step reads them from there.  This is gvr_2's register family, the design that
wins below 16K rows: no sample, no aim, no second pass over the row.

Layout: thread t owns vectors ``t + j * threads`` for j in 0..V-1, so consecutive threads
load consecutive 16-byte vectors and the loads coalesce.  Element ``e`` of a thread (fp32:
e = word; 16-bit: e = 2 * word + half) has row index
``(t + (e // (4 * per_word)) * threads) * per_vector + e % (4 * per_word)``.  Vectors past the
row's last one are clamped to it (a harmless duplicate load) and their elements masked by
index, so the load is unconditional and the words stay plain named values.

The per-element code is issue-bound (all of it runs on every element of the row), so its
instruction count is the whole story.  Three measured rules:

* Bins are computed for two elements at a time.  fp32 pairs go through one ``cvt.rn.f16x2``
  and the twiddle runs on both halves in 32-bit arithmetic (five instructions for two keys).
  A scalar fp32 -> fp16 conversion takes a quarter-rate slot per element.
* No ``if`` per element: sixteen traced ``if`` regions in a row serialize the independent
  per-element work.  Validity is arithmetic (out-of-range elements count into a trash bin),
  and it runs only for the vectors at or past the row's tail, behind one uniform branch per
  vector (v0.1.19: 16K b=64 on B200 measured the arithmetic on every element at a quarter of
  the count phase, the whole of the 5% gap to walk-first's register kernel).
* Bins are kept, packed two per register, from the count to the classify, so the classify is
  an unpack and two compares per element.  The block runs one CTA per SM (64 registers), which
  measured better than two CTAs per SM at 32 registers anyway (8.7 -> 7.3 us at 16K b=64).
* Output positions come from one block scan of per-thread counts, not a shared cursor (1024
  same-address returning atomics measured about 2 us per row), and winners are emitted by a
  bit-walk over the mask (about one iteration per thread).

The coarse bin is the census bin (top 12 bits of the fp16 ordered key, 64 bins per octave).
"""

import cutlass
import cutlass.cute as cute

from ...block.reduce import block_exclusive_scan_i32, block_max_min_u32
from ...device.atomics import shared_count
from ...device.cluster import peer_store_i32
from ...device.memory import load_global_readonly_16

from .census import element_bits, pair_key16

__all__ = [
    "load_row_words",
    "zero_bins",
    "words_after_barrier",
    "count_coarse_bins",
    "classify_from_registers",
    "classify_from_registers_cluster",
    "key_range_in_bin",
]

# The bin count is a compile-time parameter (1024 .. 4096): bin = key16 >> (16 - log2 bins),
# so fewer bins are coarser (fewer per octave) but cost less to zero and to scan.  Expected
# ties in the crossing bin scale with N / bins, so the dispatcher keeps N / bins about 4.
# Out-of-range elements get bin -1: they count into the slot just below the histogram (the
# caller allocates it) and can neither exceed nor equal any crossing bin, so the classify needs
# no extra test.  Packed bins are therefore unpacked with a sign-extending shift.


@cute.jit
def _clamped_vector(row_ptr, v, last_vector):
    vc = v
    if vc > last_vector:
        vc = last_vector
    return load_global_readonly_16(row_ptr.toint() + cutlass.Int64(vc) * 16)


@cute.jit
def load_row_words(
    row_ptr,
    length,
    tidx,
    threads: cutlass.Constexpr,
    words: cutlass.Constexpr,
    log2_per_vector: cutlass.Constexpr,
):
    """The thread's ``words // 4`` vectors of the row as ``words`` Uint32 values.

    ``length >= 1``.  Sixteen-byte read-only loads; vectors past the last one re-load it.
    """
    last_vector = (length - 1) >> cutlass.Int32(log2_per_vector)
    out: list = []
    for j in cutlass.range_constexpr(words // 4):
        w0, w1, w2, w3 = _clamped_vector(row_ptr, tidx + j * threads, last_vector)
        out.extend((w0, w1, w2, w3))
    return tuple(out)


@cute.jit
def _element_index(
    tidx,
    threads: cutlass.Constexpr,
    per_vector: cutlass.Constexpr,
    e: cutlass.Constexpr,
):
    """Row index of the thread's element ``e`` (compile-time e)."""
    return (tidx + cutlass.Int32((e // per_vector) * threads)) * cutlass.Int32(
        per_vector
    ) + cutlass.Int32(e % per_vector)


@cute.jit
def _index_of_bit(tidx, threads: cutlass.Constexpr, per_vector: cutlass.Constexpr, e):
    """Row index of the thread's element ``e`` (dynamic e)."""
    j = e // cutlass.Int32(per_vector)
    return (tidx + j * cutlass.Int32(threads)) * cutlass.Int32(per_vector) + (
        e - j * cutlass.Int32(per_vector)
    )


@cute.jit
def zero_bins(s_bins_all, count: cutlass.Constexpr, tidx, threads: cutlass.Constexpr):
    """Zero ``count`` Int32 of ``s_bins_all`` (unrolled: a rolled loop costs three instructions
    per iteration on every warp, 12 per warp at 4096 bins)."""
    for z in cutlass.range_constexpr(count // threads):
        s_bins_all[tidx + z * threads] = cutlass.Int32(0)
    if cutlass.const_expr(count % threads != 0):
        if tidx < cutlass.Int32(count % threads):
            s_bins_all[tidx + (count // threads) * threads] = cutlass.Int32(0)


@cute.jit
def words_after_barrier(wordvals, s_zeroed):
    """The row's words made dependent on a shared load of ``s_zeroed`` (an Int32 the caller
    zeroed before its barrier), so no use of the words can be scheduled above that barrier.

    Without it the compiler hoists the first pair conversion above the histogram-zeroing
    barrier (it is register-only work), so a warp reaches the barrier only once its loads have
    landed: the barrier becomes a rendezvous after the load latency and all 32 warps then
    issue their 16 shared atomics at once.  Stall-sampled on L40S, A100 and H100 at 16K b=64,
    that burst throttled the shared-memory pipe (680, 1302 and 391 MIO samples) where
    walk-first's count, entered by each warp as its own loads land, showed none.  One LDS and
    one OR per word.
    """
    zero = cutlass.Uint32(s_zeroed[0])
    return tuple(w | zero for w in wordvals)


@cute.jit
def _raw_pair_bins(elems, wordvals, p: cutlass.Constexpr, bins: cutlass.Constexpr):
    """Coarse bins of elements 2p and 2p+1 (low half = element 2p), validity not applied."""
    shift = cutlass.const_expr(16 - {1024: 10, 2048: 11, 4096: 12}[int(bins)])
    half_mask = cutlass.const_expr((int(bins) - 1) | ((int(bins) - 1) << 16))
    packed = (pair_key16(elems, wordvals, p) >> cutlass.Uint32(shift)) & cutlass.Uint32(
        half_mask
    )
    lo = (packed & cutlass.Uint32(0xFFFF)).to(cutlass.Int32)
    hi = (packed >> cutlass.Uint32(16)).to(cutlass.Int32)
    return lo, hi


@cute.jit
def _tail_bin(
    b,
    e: cutlass.Constexpr,
    length,
    tidx,
    threads: cutlass.Constexpr,
    per_vector: cutlass.Constexpr,
):
    """Bin ``b`` of the thread's element ``e``, or -1 when the element lies past the row's end."""
    valid = cutlass.Int32(_element_index(tidx, threads, per_vector, e) < length)
    return (b + 1) * valid - 1


@cute.jit
def _vector_bins(
    elems,
    wordvals,
    j: cutlass.Constexpr,
    length,
    full_vectors,
    tidx,
    threads: cutlass.Constexpr,
    bins: cutlass.Constexpr,
):
    """Coarse bins of the thread's vector ``j`` as (lo, hi) per pair: two pairs (fp32) or four
    (16-bit).  The validity test runs only on vectors at or past the row's tail: a vector
    below ``full_vectors`` is entirely inside the row (uniform across a warp except at the tail
    itself, so the branch is a cheap skip; per-element validity arithmetic measured 1.5
    instructions per element on every element of every row, a quarter of the count phase)."""
    ppv = cutlass.const_expr(2 * elems.per_word)  # pairs per 16-byte vector
    pv = cutlass.const_expr(elems.per_vector)
    l0, h0 = _raw_pair_bins(elems, wordvals, j * ppv, bins)
    l1, h1 = _raw_pair_bins(elems, wordvals, j * ppv + 1, bins)
    l2 = cutlass.Int32(0)
    h2 = cutlass.Int32(0)
    l3 = cutlass.Int32(0)
    h3 = cutlass.Int32(0)
    if cutlass.const_expr(ppv == 4):
        l2, h2 = _raw_pair_bins(elems, wordvals, j * ppv + 2, bins)
        l3, h3 = _raw_pair_bins(elems, wordvals, j * ppv + 3, bins)
    if tidx + cutlass.Int32(j * threads) >= full_vectors:
        e0 = cutlass.const_expr(j * pv)
        l0 = _tail_bin(l0, e0, length, tidx, threads, pv)
        h0 = _tail_bin(h0, e0 + 1, length, tidx, threads, pv)
        l1 = _tail_bin(l1, e0 + 2, length, tidx, threads, pv)
        h1 = _tail_bin(h1, e0 + 3, length, tidx, threads, pv)
        if cutlass.const_expr(ppv == 4):
            l2 = _tail_bin(l2, e0 + 4, length, tidx, threads, pv)
            h2 = _tail_bin(h2, e0 + 5, length, tidx, threads, pv)
            l3 = _tail_bin(l3, e0 + 6, length, tidx, threads, pv)
            h3 = _tail_bin(h3, e0 + 7, length, tidx, threads, pv)
    if cutlass.const_expr(ppv == 4):
        return (l0, h0, l1, h1, l2, h2, l3, h3)
    else:
        return (l0, h0, l1, h1)


@cute.jit
def count_coarse_bins(
    elems,
    wordvals,
    length,
    s_bins,
    tidx,
    threads: cutlass.Constexpr,
    words: cutlass.Constexpr,
    bins: cutlass.Constexpr,
):
    """Add every element to its coarse bin in ``s_bins`` (``bins`` Int32 zeroed by the caller,
    with one writable slot just below index 0 for out-of-range elements) and return the bins
    packed two per Int32 (element 2p in the low half).  No barrier; one uniform branch per
    vector (``_vector_bins``)."""
    base = s_bins.toint()
    full_vectors = length >> cutlass.Int32(
        elems.log2_per_vector
    )  # vectors entirely inside the row
    packed = []
    for j in cutlass.range_constexpr(words // 4):
        vec = _vector_bins(
            elems, wordvals, j, length, full_vectors, tidx, threads, bins
        )
        for q in cutlass.range_constexpr(len(vec) // 2):
            lo = vec[2 * q]
            hi = vec[2 * q + 1]
            shared_count(base + lo * 4)
            shared_count(base + hi * 4)
            packed.append((lo & cutlass.Int32(0xFFFF)) | (hi << 16))
    return tuple(packed)


_element_bits = element_bits


@cute.jit
def key_range_in_bin(
    elems,
    wordvals,
    packed_bins,
    cut_bin,
    s_slots_u32,
    tidx,
    threads: cutlass.Constexpr,
    words: cutlass.Constexpr,
):
    """(kmin, kmax): the exact key range of this CTA's elements in ``cut_bin``, from registers.

    For the overflow path: the radix select over the bin needs only its key range.  One block
    reduction; ``s_slots_u32``: 2 x warps Uint32.  A CTA with no member returns (0xFFFFFFFF, 0),
    the neutral pair for a further min/max merge.
    """
    pairs = cutlass.const_expr(words * elems.per_word // 2)
    kmax = cutlass.Uint32(0)
    kmin = cutlass.Uint32(0xFFFFFFFF)
    for p in cutlass.range_constexpr(pairs):
        lo = (packed_bins[p] << 16) >> 16
        hi = packed_bins[p] >> 16
        if lo == cut_bin:
            key = elems.key(element_bits(elems, wordvals, 2 * p))
            if key > kmax:
                kmax = key
            if key < kmin:
                kmin = key
        if hi == cut_bin:
            key = elems.key(element_bits(elems, wordvals, 2 * p + 1))
            if key > kmax:
                kmax = key
            if key < kmin:
                kmin = key
    kmax, kmin = block_max_min_u32(kmax, kmin, s_slots_u32, tidx, threads)
    return kmin, kmax


@cute.jit
def _masks(packed_bins, cut_bin, pairs: cutlass.Constexpr):
    """Winner and tie bit masks over the thread's elements from the packed bins."""
    win_mask = cutlass.Int32(0)
    tie_mask = cutlass.Int32(0)
    for p in cutlass.range_constexpr(pairs):
        lo = (
            packed_bins[p] << 16
        ) >> 16  # sign-extending: -1 marks an out-of-range element
        hi = packed_bins[p] >> 16
        win_mask = (
            win_mask
            | (cutlass.Int32(lo > cut_bin) << (2 * p))
            | (cutlass.Int32(hi > cut_bin) << (2 * p + 1))
        )
        tie_mask = (
            tie_mask
            | (cutlass.Int32(lo == cut_bin) << (2 * p))
            | (cutlass.Int32(hi == cut_bin) << (2 * p + 1))
        )
    return win_mask, tie_mask


@cute.jit
def _emit_winners(
    elems,
    win_mask,
    win_pos,
    index_offset,
    out_row,
    tidx,
    threads: cutlass.Constexpr,
    elements: cutlass.Constexpr = 0,
):
    """Write the row indices of the mask's elements at ``out_row[win_pos ...]``.  ``index_offset``
    is the slice's start in the row.  With ``elements`` (the thread's element count, a
    compile-time constant) of at most 4, the walk is unrolled as one predicated store per
    element slot: no lowest-bit arithmetic, and the warp does not iterate to its slowest lane
    (v0.1.27: 1K b=1 2.22 -> 2.11 us, 4K b=1 3.17 -> 3.09, 4K b=256 4.64 -> 4.52 on B200).
    Above 4 slots the predicated form loses (16 slots: 16K b=1 5.14 -> 5.36, the winners are
    sparse and every slot still pays its test), so those keep the bit-walk (about 17
    instructions per winner, about five rounds per warp at k=2048)."""
    if cutlass.const_expr(0 < elements <= 4):
        for e in cutlass.range_constexpr(elements):
            if ((win_mask >> cutlass.Int32(e)) & cutlass.Int32(1)) == 1:
                out_row[win_pos] = index_offset + _element_index(
                    tidx, threads, elems.per_vector, e
                )
                win_pos = win_pos + 1
    else:
        while win_mask != 0:
            e = cutlass.Int32(
                cute.arch.popc(
                    (win_mask & (cutlass.Int32(0) - win_mask)) - cutlass.Int32(1)
                )
            )
            win_mask = win_mask & (win_mask - cutlass.Int32(1))
            out_row[win_pos] = index_offset + _index_of_bit(
                tidx, threads, elems.per_vector, e
            )
            win_pos = win_pos + 1


@cute.jit
def classify_from_registers(
    elems,
    wordvals,
    packed_bins,
    cut_bin,
    out_row,
    s_tie_keys,
    s_tie_idx,
    tie_capacity: cutlass.Constexpr,
    s_slots,
    tidx,
    threads: cutlass.Constexpr,
    words: cutlass.Constexpr,
    bins: cutlass.Constexpr,
):
    """Write every element above ``cut_bin`` to ``out_row`` and stage every element in
    ``cut_bin`` (exact key, index); return ``(winners, ties)`` totals to every thread.

    ``packed_bins`` is what ``count_coarse_bins`` returned.  Positions come from one block
    scan of per-thread (winner, tie) counts packed in one Int32 (16 bits each; at most 32
    elements per thread).  ``s_slots``: warps Int32 scratch.  Two barriers, inside the scan;
    the caller barriers before reading the tie stage.  (Warp-aggregated tickets on two shared
    cursors instead of the scan measured 0.15 us slower at 16K b=64 on B200: the two extra
    warp scans and atomics cost more than the two barriers they remove; tickets for the winners
    alone with per-tie atomics gained 0.07 us at 16K b=64 but lost 0.1-0.2 us at 4K b=256 and
    8K b=64.)
    """
    pairs = cutlass.const_expr(words * elems.per_word // 2)
    win_mask, tie_mask = _masks(packed_bins, cut_bin, pairs)
    packed = (cutlass.Int32(cute.arch.popc(win_mask)) << 16) | cutlass.Int32(
        cute.arch.popc(tie_mask)
    )
    before, total = block_exclusive_scan_i32(packed, s_slots, tidx, threads)
    _emit_winners(
        elems,
        win_mask,
        before >> 16,
        cutlass.Int32(0),
        out_row,
        tidx,
        threads,
        2 * pairs,
    )
    tie_pos = before & cutlass.Int32(0xFFFF)
    if tie_mask != 0:  # rare; the exact key needs the word, so test each position
        for e in cutlass.range_constexpr(2 * pairs):
            if ((tie_mask >> e) & 1) == 1:
                if tie_pos < cutlass.Int32(tie_capacity):
                    s_tie_keys[tie_pos] = elems.key(_element_bits(elems, wordvals, e))
                    s_tie_idx[tie_pos] = _element_index(
                        tidx, threads, elems.per_vector, e
                    )
                tie_pos = tie_pos + 1
    return total >> 16, total & cutlass.Int32(0xFFFF)


@cute.jit
def classify_from_registers_cluster(
    elems,
    wordvals,
    packed_bins,
    cut_bin,
    slice_start,
    out_row,
    tie_keys_root,
    tie_idx_root,
    win_before,
    tie_before,
    tie_capacity: cutlass.Constexpr,
    s_slots,
    tidx,
    threads: cutlass.Constexpr,
    words: cutlass.Constexpr,
):
    """Cluster form: this CTA's winners go to ``out_row`` from position ``win_before`` and its
    ties to rank 0's tie stage from ``tie_before`` (the lower-ranked peers' counts, from
    ``cluster_crossing``), so the output order is by rank and no CTA reserves anything.

    ``tie_keys_root``, ``tie_idx_root``: mapped DSMEM addresses of rank 0's tie arrays.  Two
    block barriers (the scan's).  Row indices are ``slice_start`` plus the position in the
    slice.  (v0.1.23: the previous form reserved positions with one remote atomic per CTA on
    rank 0's cursors, and every warp then waited at a barrier for that round trip.)
    """
    pairs = cutlass.const_expr(words * elems.per_word // 2)
    win_mask, tie_mask = _masks(packed_bins, cut_bin, pairs)
    packed = (cutlass.Int32(cute.arch.popc(win_mask)) << 16) | cutlass.Int32(
        cute.arch.popc(tie_mask)
    )
    before, _total = block_exclusive_scan_i32(packed, s_slots, tidx, threads)
    _emit_winners(
        elems,
        win_mask,
        win_before + (before >> 16),
        slice_start,
        out_row,
        tidx,
        threads,
        2 * pairs,
    )
    tie_pos = tie_before + (before & cutlass.Int32(0xFFFF))
    if tie_mask != 0:
        for e in cutlass.range_constexpr(2 * pairs):
            if ((tie_mask >> e) & 1) == 1:
                if tie_pos < cutlass.Int32(tie_capacity):
                    peer_store_i32(
                        tie_keys_root + tie_pos * 4,
                        elems.key(_element_bits(elems, wordvals, e)).bitcast(
                            cutlass.Int32
                        ),
                    )
                    peer_store_i32(
                        tie_idx_root + tie_pos * 4,
                        slice_start
                        + _element_index(tidx, threads, elems.per_vector, e),
                    )
                tie_pos = tie_pos + 1
