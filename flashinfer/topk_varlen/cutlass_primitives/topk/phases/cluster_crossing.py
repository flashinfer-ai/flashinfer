"""Rank-k crossing over a histogram that is spread across a cluster, without merging all of it.

Every CTA holds a local ``bins``-bin histogram of its slice.  Merging all bins over DSMEM is
``bins x splits`` remote loads per CTA (3 to 8 us at 4096 bins on B200, latency-bound).  The
crossing only needs the merged counts of the bins above the answer and of the answer's bin,
so it is found in two levels:

1. Each CTA sums its histogram into 256 groups of ``bins / 256`` consecutive bins (a local
   pass over shared memory).  The groups are merged over DSMEM (256 x splits remote loads per
   CTA, one per thread) and the crossing found among them: group G, ``above`` = the merged
   count in groups above G.
2. The ``bins / 256`` fine bins of group G are merged (one thread each) and the crossing is
   found among them by one thread.  The result is identical to a crossing over the fully
   merged histogram.

Every CTA computes the same answer from the same peer data, so no broadcast is needed.
Precondition: a cluster barrier separates the peers' histogram and group writes from this call.

The same remote loads give each CTA its output offsets (v0.1.23): the winners of a peer are
its counts above the crossing bin, so the peer values a thread loaded for the merge, kept in
registers, summed over the lower-ranked peers once the crossing is known, are the number of
winners (and ties) written before this CTA's.  One block reduction replaces the remote atomic
on rank 0's cursors that every CTA used to wait on between two barriers (B200 64K b=8 k=1024:
the stall samples of that barrier were the largest single difference to gvr_2's kernel).
"""

import cutlass
import cutlass.cute as cute

from ...block.crossing import crossing_256_warp
from ...device.atomics import shared_add
from ...device.cluster import peer_load_i32, peer_shared_address
from ...device.memory import load_shared_16
from ...device.warp import warp_sum

__all__ = ["summarize_groups_256", "cluster_crossing"]


@cute.jit
def summarize_groups_256(s_bins, s_groups, bins: cutlass.Constexpr, tidx):
    """``s_groups[g]`` = sum of the ``bins / 256`` bins of group g, by threads 0..255.  Sixteen-
    byte shared loads; ``s_bins`` 16-byte aligned.  No barrier (the caller's cluster barrier
    publishes the groups to the peers)."""
    per_group = cutlass.const_expr(bins // 256)
    if tidx < 256:
        base = s_bins.toint() + tidx * (per_group * 4)
        total = cutlass.Uint32(0)
        for q in cutlass.range_constexpr(per_group // 4):
            a, b, c, d = load_shared_16(base + q * 16)
            total = total + a + b + c + d
        s_groups[tidx] = total.to(cutlass.Int32)


@cute.jit
def cluster_crossing(
    s_bins,
    s_groups,
    s_merged_groups,
    s_fine,
    s_result,
    k: cutlass.Constexpr,
    bins: cutlass.Constexpr,
    splits: cutlass.Constexpr,
    tidx,
    rank,
):
    """Publish the rank-k crossing over the cluster's merged histogram to ``s_result[0..2]``
    as ``(bin, above, count in the bin)``, and this CTA's output offsets to ``s_result[4..6]``
    as ``(winners before this CTA's, ties before this CTA's, ties in the row)``: the counts
    above and in the crossing bin of the peers with a lower ``rank`` (output order is by rank).

    ``s_merged_groups``: 256 Int32 scratch; ``s_fine``: ``bins / 256`` Int32 scratch;
    ``s_result``: 8 Int32.  Remote loads: ``splits`` per thread for threads 0..255, then
    ``splits`` for threads 0..per_group-1 (the peer values stay in registers for the offsets).
    Five block barriers.  Requires 256 or more threads.
    """
    per_group = cutlass.const_expr(bins // 256)
    if tidx == 0:
        s_result[4] = cutlass.Int32(0)
        s_result[5] = cutlass.Int32(0)
        s_result[6] = cutlass.Int32(0)
    below_groups = cutlass.Int32(0)  # this group's count over the lower-ranked peers
    if tidx < 256:
        addr = s_groups.toint() + tidx * 4
        total = cutlass.Int32(0)
        for r in cutlass.range_constexpr(splits):
            v = peer_load_i32(peer_shared_address(addr, cutlass.Int32(r)))
            total = total + v
            below_groups = below_groups + v * cutlass.Int32(cutlass.Int32(r) < rank)
        s_merged_groups[tidx] = total
    cute.arch.barrier()
    if tidx < 32:
        g, above_g, _c, _g2, _a2, _c2 = crossing_256_warp(
            s_merged_groups, cutlass.Int32(k), cutlass.Int32(k), tidx, False
        )
        if tidx == 0:
            s_result[2] = g
            s_result[3] = above_g
    cute.arch.barrier()
    group = s_result[2]
    fine_total = cutlass.Int32(
        0
    )  # this fine bin's count over every peer, and over the lower-ranked ones
    fine_below = cutlass.Int32(0)
    if tidx < per_group:
        addr = s_bins.toint() + (group * per_group + tidx) * 4
        for r in cutlass.range_constexpr(splits):
            v = peer_load_i32(peer_shared_address(addr, cutlass.Int32(r)))
            fine_total = fine_total + v
            fine_below = fine_below + v * cutlass.Int32(cutlass.Int32(r) < rank)
        s_fine[tidx] = fine_total
    cute.arch.barrier()
    if tidx == 0:  # walk the group's fine bins from the top
        above = s_result[3]
        found = cutlass.Int32(0)
        for j in cutlass.range_constexpr(per_group - 1, -1, -1):
            c = s_fine[j]
            if found == 0:
                if (above < cutlass.Int32(k)) & (
                    (above + c >= cutlass.Int32(k)) | (j == 0)
                ):
                    s_result[0] = group * per_group + j
                    s_result[1] = above
                    s_result[2] = c
                    found = cutlass.Int32(1)
                else:
                    above = above + c
    cute.arch.barrier()
    # the offsets: lower-ranked peers' counts in the groups above the crossing group, plus in
    # the fine bins above the crossing bin; ties in the crossing bin itself
    cut_bin = s_result[0]
    win_before = cutlass.Int32(0)
    tie_before = cutlass.Int32(0)
    ties = cutlass.Int32(0)
    if tidx < 256:
        if tidx > group:
            win_before = below_groups
    if tidx < per_group:
        b = group * per_group + tidx
        if b > cut_bin:
            win_before = win_before + fine_below
        if b == cut_bin:
            tie_before = fine_below
            ties = fine_total
    if tidx < 256:  # eight warps: one shared add per warp and value
        win_before = warp_sum(win_before)
        tie_before = warp_sum(tie_before)
        ties = warp_sum(ties)
        if tidx % 32 == 0:
            shared_add(s_result + 4, win_before)
            shared_add(s_result + 5, tie_before)
            shared_add(s_result + 6, ties)
    cute.arch.barrier()
