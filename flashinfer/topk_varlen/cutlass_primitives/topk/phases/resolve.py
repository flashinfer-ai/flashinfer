"""Resolution: from a usable survivor set to the k output indices.

The crossing at rank k over the survivor histogram splits the stage into three: bins above
the crossing are winners outright, the crossing bin holds the ties for the last
``k - above`` slots, bins below are discarded.  One pass over the stage emits winners at a
shared cursor and copies the ties with their exact keys to the tie stage; then the tie select
(ballot for small sets, byte radix otherwise) fills the remainder.

Emit positions come from a cursor per histogram bin (v0.1.29): the crossing warp writes the
count above each bin into a 256-word array and a candidate of bin b takes ``cursor[b]++``, so
winners land in bin order within [0, above) and the crossing bin's members fill the tie stage.
Before that every winner took one shared cursor: at one 1024-thread CTA per SM the hardware
coalesced those same-address atomics well enough (two warp-aggregated forms, shuffle scan and
ballot + popcount, were slower: 1.1 -> 2.0 and 1.5 us at 64K, k=2048), but with two
512-thread CTAs sharing an SM's atomic unit they serialized (the resolution cost 0.74 us per
thousand candidates).  Per-bin cursors: B200 wide batches 3% (64K b=256 k=2048 16.94 -> 16.40
us, k=512 12.72 -> 12.39, 16K b=256 7.17 -> 6.97), one-CTA-per-SM rows 1% (docs/kernels/
streaming.md).  The cluster form gets a private cursor array per CTA from the merged
histogram's "above" plus the lower-ranked peers' counts (``merge_histograms_256_ranked``), so
its emit does no remote atomic at all.
"""

import cutlass
import cutlass.cute as cute

from ...block.cluster_merge import merge_histograms_256_ranked
from ...block.crossing import crossing_256_warp
from ...block.reduce import block_exclusive_scan_i32
from ...block.tie_select import tie_select_ballot, tie_select_radix
from ...device.atomics import shared_add
from ...device.cluster import cluster_sync, peer_shared_address, peer_store_i32

from .binning import survivor_bin
from .filter_pass import stage_pair

__all__ = ["emit_and_select", "emit_and_select_cluster"]


@cute.jit
def emit_and_select(
    elems,
    k: cutlass.Constexpr,
    survivors,
    bar,
    scale,
    out_row,
    s_stage,
    s_hist,
    s_tie_keys,
    s_tie_idx,
    tie_capacity: cutlass.Constexpr,
    ballot_limit: cutlass.Constexpr,
    s_slots,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
    scan_emit: cutlass.Constexpr = False,
    s_cursor=None,
    bin_cursors: cutlass.Constexpr = True,
):
    """Write the k winners of a usable stage to ``out_row``; return 1, or 0 if the crossing
    bin overflowed the tie stage (the caller then takes the exact fallback).

    Preconditions: ``k <= survivors <= capacity``; ``s_hist`` is the histogram the stage was
    built with, under the same ``bar`` and ``scale``.  ``s_stage``: the (bits, index) pair
    stage of ``filter_pass`` (an Int32 array of >= 512 words, dead after the emit and reused as
    the radix select's histogram).  ``s_cursor``: 256 Int32 scratch
    for the per-bin emit cursors (required unless ``scan_emit``).  ``s_result``: 8 Int32
    scratch; ``s_slots``: warps Int32.  Barriers: two, plus the radix select's when taken.

    ``scan_emit``: positions from one block scan of per-thread (winner, tie) counts packed in
    one word, instead of a same-address shared atomic per candidate.  The atomic form won at
    one 1024-thread CTA per SM (module docstring); with two 512-thread CTAs sharing an SM's
    atomic unit at k=2048 the candidates' atomics serialize (measured here: the per-row cost
    grows 2.7 us per thousand staged survivors), so the wide-batch policy selects the scan.
    """
    if tidx < 32:
        b, above, c, _b2, _a2, _c2 = crossing_256_warp(
            s_hist,
            cutlass.Int32(k),
            cutlass.Int32(k),
            tidx,
            False,
            s_cursor=s_cursor,
            cursors=bin_cursors and not scan_emit,
        )
        if tidx == 0:
            s_result[0] = b
            s_result[1] = above
            s_result[2] = c  # members of the crossing bin: the tie count
            s_result[6] = cutlass.Int32(0)  # single-cursor form: winner cursor
            s_result[7] = cutlass.Int32(0)  # single-cursor form: tie cursor
    cute.arch.barrier()
    cut_bin = s_result[0]
    above = s_result[1]
    ties = cutlass.Int32(0)
    stage_base = s_stage.toint()
    if cutlass.const_expr(scan_emit):
        # pass 1: this thread's winner and tie counts (packed: winners in the high half)
        mine = cutlass.Int32(0)
        for t in range(tidx, survivors, threads):
            bits, _idx = stage_pair(stage_base, t)
            b = survivor_bin(elems.value(bits), bar, scale)
            if b > cut_bin:
                mine = mine + cutlass.Int32(65536)
            else:
                if b == cut_bin:
                    mine = mine + cutlass.Int32(1)
        before, total = block_exclusive_scan_i32(mine, s_slots, tidx, threads)
        wpos = before >> cutlass.Int32(16)
        tpos = before & cutlass.Int32(0xFFFF)
        ties = total & cutlass.Int32(0xFFFF)
        # pass 2: write at the scanned positions (the stage is 8 KB per thousand candidates:
        # the second read comes from shared memory)
        for t in range(tidx, survivors, threads):
            bits, idx = stage_pair(stage_base, t)
            b = survivor_bin(elems.value(bits), bar, scale)
            if b > cut_bin:
                if wpos < cutlass.Int32(k):
                    out_row[wpos] = idx
                wpos = wpos + cutlass.Int32(1)
            else:
                if b == cut_bin:
                    if tpos < cutlass.Int32(tie_capacity):
                        s_tie_keys[tpos] = elems.key(bits)
                        s_tie_idx[tpos] = idx
                    tpos = tpos + cutlass.Int32(1)
        cute.arch.barrier()
    elif cutlass.const_expr(bin_cursors):
        # per-bin cursors (v0.1.29): candidate in bin b takes position cursor[b]++, where the
        # cursor starts at the count above b (the crossing wrote it), so winners land in
        # rank order of their bins within [0, above) and the ties at [above, above + ties).
        # One shared cursor for all winners was 2048 same-address atomics per row at k=2048:
        # the resolution cost 0.74 us per thousand candidates on the wide batches (B200,
        # 512 threads x 2 per SM; 2.55 us at k=2048 against 1.26 at k=512).
        for t in range(tidx, survivors, threads):
            bits, idx = stage_pair(stage_base, t)
            b = survivor_bin(elems.value(bits), bar, scale)
            if b >= cut_bin:
                p = shared_add(s_cursor + b, 1)
                if b > cut_bin:
                    out_row[p] = idx  # p < above <= k by construction
                else:
                    e = p - above
                    if e < cutlass.Int32(tie_capacity):
                        s_tie_keys[e] = elems.key(bits)
                        s_tie_idx[e] = idx
        cute.arch.barrier()
        ties = s_result[2]
    else:
        # one shared cursor each for winners and ties: the SM80 form (device fact
        # ``bin_cursors``; its shared atomic unit handles the same address better than 256)
        for t in range(tidx, survivors, threads):
            bits, idx = stage_pair(stage_base, t)
            b = survivor_bin(elems.value(bits), bar, scale)
            if b > cut_bin:
                p = shared_add(s_result + 6, 1)
                if p < cutlass.Int32(k):
                    out_row[p] = idx
            else:
                if b == cut_bin:
                    e = shared_add(s_result + 7, 1)
                    if e < cutlass.Int32(tie_capacity):
                        s_tie_keys[e] = elems.key(bits)
                        s_tie_idx[e] = idx
        cute.arch.barrier()
        ties = s_result[7]
    return _select_ties(
        elems,
        k,
        above,
        ties,
        out_row,
        s_stage,
        s_tie_keys,
        s_tie_idx,
        tie_capacity,
        ballot_limit,
        s_slots,
        s_result,
        tidx,
        threads,
    )


@cute.jit
def _select_ties(
    elems,
    k: cutlass.Constexpr,
    above,
    ties,
    out_row,
    s_scratch,
    s_tie_keys,
    s_tie_idx,
    tie_capacity: cutlass.Constexpr,
    ballot_limit: cutlass.Constexpr,
    s_slots,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
):
    """Fill ``out_row[above, k)`` from the tie stage; 0 if the ties overflowed it."""
    remaining = cutlass.Int32(k) - above
    ok = cutlass.Int32(1)
    if (remaining < 0) | (remaining > ties) | (ties > cutlass.Int32(tie_capacity)):
        ok = cutlass.Int32(0)
    if ok == 1:
        if remaining > 0:
            if ties <= cutlass.Int32(ballot_limit):
                tie_select_ballot(
                    s_tie_keys,
                    s_tie_idx,
                    ties,
                    remaining,
                    out_row,
                    above,
                    tidx,
                    threads,
                )
            else:
                tie_select_radix(
                    s_tie_keys,
                    s_tie_idx,
                    ties,
                    remaining,
                    out_row,
                    above,
                    s_scratch,
                    s_slots,
                    s_result,
                    tidx,
                    threads,
                    elems.key_shifts,
                    tie_capacity // threads,
                )
    return ok


@cute.jit
def emit_and_select_cluster(
    elems,
    k: cutlass.Constexpr,
    rank,
    splits: cutlass.Constexpr,
    bar,
    scale,
    out_row,
    s_stage,
    s_hist,
    s_merged,
    s_tie_keys,
    s_tie_idx,
    tie_capacity: cutlass.Constexpr,
    ballot_limit: cutlass.Constexpr,
    s_count,
    s_slots,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
):
    """Cluster form of the resolution: the row's stage is spread over ``splits`` CTAs.

    Every CTA merges the peers' survivor histograms over DSMEM (identical result everywhere)
    and, alongside, the sum over the lower-ranked peers; the rank-k crossing over the merged
    histogram gives every bin its start in rank order, and ``above[b] + lower[b]`` is where
    this CTA's members of bin b begin: a private per-bin cursor array per CTA (v0.1.29; before,
    every winner took a DSMEM atomic on one cursor in rank 0).  Each CTA classifies its own
    stage: winners go straight to the output at their cursor, ties are stored into rank 0's
    tie stage at theirs.  After the cluster barrier rank 0 alone selects the ties.  Returns
    ``ok`` (meaningful on rank 0; other ranks return 1 and must not run the fallback).
    Preconditions: the verdict's cluster barrier has passed (every peer's histogram is
    complete).  Peers use their own idle tie-index stage as the lower-sum scratch (rank 0's
    is the cluster's tie stage and rank 0's lower sum is zero).  Two cluster barriers, three
    block barriers.
    """
    merge_histograms_256_ranked(s_hist, s_merged, s_tie_idx, rank, splits, tidx)
    cute.arch.barrier()
    if tidx < 32:
        b, above, c, _b2, _a2, _c2 = crossing_256_warp(
            s_merged,
            cutlass.Int32(k),
            cutlass.Int32(k),
            tidx,
            False,
            s_cursor=s_merged,
            cursors=True,
        )
        if tidx == 0:
            s_result[0] = b
            s_result[1] = above
            s_result[2] = (
                c  # the crossing bin's members across the cluster: the tie count
            )
    cute.arch.barrier()
    if tidx < 256:  # cursor[b] = above[b] + this CTA's lower-rank sum of bin b
        if rank != 0:
            s_merged[tidx] = s_merged[tidx] + s_tie_idx[tidx]
    cute.arch.barrier()
    cut_bin = s_result[0]
    above = s_result[1]
    root = cutlass.Int32(0)
    tie_keys_root = peer_shared_address(s_tie_keys.toint(), root)
    tie_idx_root = peer_shared_address(s_tie_idx.toint(), root)
    local = s_count[0]
    stage_base = s_stage.toint()
    for t in range(tidx, local, threads):
        bits, idx = stage_pair(stage_base, t)
        b = survivor_bin(elems.value(bits), bar, scale)
        if b >= cut_bin:
            p = shared_add(s_merged + b, 1)
            if b > cut_bin:
                out_row[p] = idx  # p < above <= k by construction
            else:
                e = (
                    p - above
                )  # this tie's index in the cluster-wide tie stage on rank 0
                if e < cutlass.Int32(tie_capacity):
                    peer_store_i32(
                        tie_keys_root + e * 4, elems.key(bits).bitcast(cutlass.Int32)
                    )
                    peer_store_i32(tie_idx_root + e * 4, idx)
    cluster_sync()  # every CTA's winners and ties are in; peers may stop touching rank 0 now
    ok = cutlass.Int32(1)
    if rank == 0:
        ok = _select_ties(
            elems,
            k,
            above,
            s_result[2],
            out_row,
            s_stage,
            s_tie_keys,
            s_tie_idx,
            tie_capacity,
            ballot_limit,
            s_slots,
            s_result,
            tidx,
            threads,
        )
    return ok
