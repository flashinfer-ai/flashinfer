"""Merging per-CTA results across a cluster through distributed shared memory.

A row split over a cluster leaves each CTA with its own survivor count and 256-bin survivor
histogram.  Instead of publishing them to global memory and electing a last arriver, every
CTA reads its peers' shared memory directly.  The reads are replicated: every CTA computes
the same merged values, so verdicts derived from them are cluster-uniform and barriers
inside the guarded code are safe.  Phase telemetry on the streaming kernel put the global
memory route at a flat ~1.8 us histogram gather plus ~1 us slab emit plus arrival skew;
the DSMEM route removes all three.

Preconditions for every function here: the launch is a cluster of ``size`` CTAs, and a
``cluster_sync`` separates the peers' last writes from the reads.
"""

import cutlass
import cutlass.cute as cute

from ..device.cluster import peer_load_16, peer_load_i32, peer_shared_address
from ..device.memory import store_shared_16

__all__ = [
    "cluster_sum_i32",
    "merge_histograms_256",
    "merge_histograms_256_ranked",
    "merge_histograms_wide",
]


@cute.jit
def cluster_sum_i32(s_value, size: cutlass.Constexpr):
    """Sum of the Int32 at the same shared address in every CTA of the cluster.

    ``s_value`` is a local shared pointer to one Int32.  Every calling thread gets the sum;
    no barrier.  Cost: ``size`` remote loads.
    """
    addr = s_value.toint()
    total = cutlass.Int32(0)
    for r in cutlass.range_constexpr(size):
        total = total + peer_load_i32(peer_shared_address(addr, cutlass.Int32(r)))
    return total


@cute.jit
def merge_histograms_wide(
    s_hist,
    s_merged,
    bins: cutlass.Constexpr,
    size: cutlass.Constexpr,
    tidx,
    threads: cutlass.Constexpr,
):
    """Sum ``bins``-bin Int32 histograms of all CTAs in the cluster into ``s_merged``, four
    bins per 16-byte remote load (``bins / 4 / threads`` quads per thread, ``size`` loads
    each).  Both arrays 16-byte aligned; ``s_hist`` stays intact for the peers.  No barrier
    inside.  Measured against 4-byte loads at 4096 bins on B200: 3.45 -> ? us at 4 CTAs (see
    the register-cluster design note)."""
    for q in range(tidx, bins // 4, threads):
        addr = s_hist.toint() + q * 16
        t0 = cutlass.Uint32(0)
        t1 = cutlass.Uint32(0)
        t2 = cutlass.Uint32(0)
        t3 = cutlass.Uint32(0)
        for r in cutlass.range_constexpr(size):
            a, b, c, d = peer_load_16(peer_shared_address(addr, cutlass.Int32(r)))
            t0 = t0 + a
            t1 = t1 + b
            t2 = t2 + c
            t3 = t3 + d
        store_shared_16(s_merged.toint() + q * 16, t0, t1, t2, t3)


@cute.jit
def merge_histograms_256_ranked(
    s_hist, s_merged, s_lower, rank, size: cutlass.Constexpr, tidx
):
    """``merge_histograms_256`` that also leaves, in ``s_lower[b]``, the sum over the CTAs of
    lower rank than ``rank`` (zero on rank 0; ``s_lower`` is not written there).

    With the merged histogram's "count above bin b" from the crossing, ``above[b] +
    lower[b]`` is where this CTA's members of bin b start in the cluster-wide rank order: a
    private per-bin cursor per CTA, so the clustered emit needs no atomic on a peer (the
    single remote cursor cost every winner a DSMEM atomic; docs/kernels/streaming.md,
    "Per-bin emit cursors").  Same loads as the plain merge; no barrier inside.
    """
    if tidx < 256:
        addr = s_hist.toint() + tidx * 4
        total = cutlass.Int32(0)
        lower = cutlass.Int32(0)
        for r in cutlass.range_constexpr(size):
            v = peer_load_i32(peer_shared_address(addr, cutlass.Int32(r)))
            total = total + v
            if cutlass.Int32(r) < rank:
                lower = lower + v
        s_merged[tidx] = total
        if rank != 0:
            s_lower[tidx] = lower


@cute.jit
def merge_histograms_256(s_hist, s_merged, size: cutlass.Constexpr, tidx):
    """Sum the 256-bin Int32 histograms of all CTAs in the cluster into ``s_merged``.

    Thread t < 256 owns bin t and adds the peers' values (``size`` remote loads).  ``s_hist``
    stays intact because peers may still be reading it; the sum lands in the separate
    ``s_merged``.  No barrier inside: the caller's next block barrier publishes ``s_merged``.
    """
    if tidx < 256:
        addr = s_hist.toint() + tidx * 4
        total = cutlass.Int32(0)
        for r in cutlass.range_constexpr(size):
            total = total + peer_load_i32(peer_shared_address(addr, cutlass.Int32(r)))
        s_merged[tidx] = total
