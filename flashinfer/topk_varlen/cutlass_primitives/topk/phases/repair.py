"""Verdict and repair: is the staged survivor set usable, and if not, can one more pass fix it?

Three outcomes of a filter pass, read from block-uniform shared values so every branch here
is safe to put barriers in:

* ``k <= count <= capacity``: usable.
* ``count < k`` (undershoot): the aim was too tight for this row.  Re-run the pass at the
  floor threshold (rank ``floor_multiple x`` the aim among the samples), which undershoots
  with negligible probability.
* ``count > capacity`` (overflow): the survivor histogram is complete regardless of staging,
  so the crossing at rank ``capacity - offset`` names a tighter bar with fewer survivors
  than the stage holds.  Re-run there if that rank still covers k.  Before this rung a 1.7x
  sample overshoot (3 of 256 fp32 rows at 1M, k=1024) cost a full exact fallback per row.
  The offset (64) absorbs bin-edge rounding of the re-run.

A degenerate sample skips everything: the caller takes the exact fallback.
"""

import cutlass
import cutlass.cute as cute

from ...block.cluster_merge import cluster_sum_i32
from ...block.crossing import crossing_256_warp
from ...device.cluster import cluster_sync

from .filter_pass import filter_pass

__all__ = ["verdict_and_repair", "verdict_and_repair_cluster"]


@cute.jit
def verdict_and_repair(
    elems,
    row_ptr,
    start,
    count,
    k: cutlass.Constexpr,
    bar,
    scale,
    floor_bar,
    floor_scale,
    degenerate,
    packed: cutlass.Constexpr,
    capacity: cutlass.Constexpr,
    overflow_offset: cutlass.Constexpr,
    s_count,
    s_hist,
    s_stage,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
    walk_width: cutlass.Constexpr = 1,
):
    """Return ``(bar, scale, survivors, ok, passes)`` after at most two repair passes.

    ``bar`` and ``scale`` are rebound to whatever the last pass used, so the emit classifies
    with exactly the arithmetic that built the histogram.  ``ok`` is 1 when
    ``k <= survivors <= capacity`` and the sample was not degenerate.  ``passes`` counts the
    filter passes taken (1 to 3), for the status word.  ``s_result``: 2 Int32 scratch.  All
    threads call it.
    """
    survivors = s_count[0]
    passes = cutlass.Int32(1)
    if degenerate == 0:
        if survivors < cutlass.Int32(k):
            bar = floor_bar
            scale = floor_scale
            filter_pass(
                elems,
                row_ptr,
                start,
                count,
                bar,
                scale,
                packed,
                capacity,
                s_count,
                s_hist,
                s_stage,
                tidx,
                threads,
                unroll,
                walk_width,
            )
            survivors = s_count[0]
            passes = passes + cutlass.Int32(1)
        if survivors > cutlass.Int32(capacity):
            target = cutlass.Int32(capacity - overflow_offset)
            if tidx < 32:
                b, above, _c, _b2, _a2, _c2 = crossing_256_warp(
                    s_hist, target, target, tidx, False
                )
                if tidx == 0:
                    s_result[0] = b
                    s_result[1] = above
            cute.arch.barrier()
            cut_bin = s_result[0]
            cut_above = s_result[1]
            if (cut_above >= cutlass.Int32(k)) & (cut_bin < 254):
                # raise the bar to the cut bin's upper edge; keep 255 bins over [bar, max]
                edge = (cut_bin + 1).to(cutlass.Float32)
                bar = bar + edge / scale
                scale = scale * cutlass.Float32(255.0) / (cutlass.Float32(255.0) - edge)
                filter_pass(
                    elems,
                    row_ptr,
                    start,
                    count,
                    bar,
                    scale,
                    packed,
                    capacity,
                    s_count,
                    s_hist,
                    s_stage,
                    tidx,
                    threads,
                    unroll,
                    walk_width,
                )
                survivors = s_count[0]
                passes = passes + cutlass.Int32(1)
    ok = cutlass.Int32(0)
    if (
        (degenerate == 0)
        & (survivors >= cutlass.Int32(k))
        & (survivors <= cutlass.Int32(capacity))
    ):
        ok = cutlass.Int32(1)
    return bar, scale, survivors, ok, passes


@cute.jit
def _publish_local(s_count, s_result, capacity: cutlass.Constexpr, tidx):
    """Local survivor count and overflow flag into the two shared words peers will read."""
    local = s_count[0]
    if tidx == 0:
        s_result[8] = local
        s_result[9] = cutlass.Int32(0)
        if local > cutlass.Int32(capacity):
            s_result[9] = cutlass.Int32(1)


@cute.jit
def verdict_and_repair_cluster(
    elems,
    row_ptr,
    start,
    count,
    k: cutlass.Constexpr,
    bar,
    scale,
    floor_bar,
    floor_scale,
    degenerate,
    packed: cutlass.Constexpr,
    capacity: cutlass.Constexpr,
    splits: cutlass.Constexpr,
    s_count,
    s_hist,
    s_stage,
    s_result,
    tidx,
    threads: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
    walk_width: cutlass.Constexpr = 1,
):
    """Cluster form of the verdict: the row's survivors are the sum over the ``splits`` CTAs.

    Every CTA publishes its local count and overflow flag (``s_result[8..9]``), the cluster
    synchronizes, and every CTA reads all peers' words, so the verdict is identical on every
    CTA and the re-walk below is cluster-uniform.  Undershoot re-walks at the floor; overflow
    of any CTA's stage is not repaired here (rare with the aim capped at 3/4 of the combined
    stage) and hands the row to the fallback.  Returns ``(bar, scale, survivors, ok)``; the
    caller's next DSMEM traffic may start right away, the closing ``cluster_sync`` is inside.
    """
    _publish_local(s_count, s_result, capacity, tidx)
    cluster_sync()
    survivors = cluster_sum_i32(s_result + 8, splits)
    overflow = cluster_sum_i32(s_result + 9, splits)
    passes = cutlass.Int32(1)
    if degenerate == 0:
        if (survivors < cutlass.Int32(k)) & (overflow == 0):
            bar = floor_bar
            scale = floor_scale
            filter_pass(
                elems,
                row_ptr,
                start,
                count,
                bar,
                scale,
                packed,
                capacity,
                s_count,
                s_hist,
                s_stage,
                tidx,
                threads,
                unroll,
                walk_width,
            )
            _publish_local(s_count, s_result, capacity, tidx)
            cluster_sync()
            survivors = cluster_sum_i32(s_result + 8, splits)
            overflow = cluster_sum_i32(s_result + 9, splits)
            passes = passes + cutlass.Int32(1)
    ok = cutlass.Int32(0)
    if (degenerate == 0) & (survivors >= cutlass.Int32(k)) & (overflow == 0):
        ok = cutlass.Int32(1)
    return bar, scale, survivors, ok, passes
