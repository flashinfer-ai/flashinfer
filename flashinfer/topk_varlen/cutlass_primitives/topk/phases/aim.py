"""Aim policies: how many survivors the sample should aim for.

The sample knows only ``samples`` values out of ``length``.  Aiming the threshold at exactly
k survivors would undershoot on half the rows (each undershoot is a full second pass), so the
aim adds a margin.  Two policies with one signature, selected at Python level:

* ``aim_tight``: ``k + max(k / 2, length / 256)``, plus a statistical floor for wide grids.
  The survivor count above a sample-derived bar is roughly normal with variance
  ``aim * length / samples``; with 32 or more rows in flight the batch waits for its slowest
  row, so the margin is raised to 3.5 sigma wherever the fixed one is below 2.5 sigma
  (k=2048 at 256K to 512K rows).  Measured: removed a bimodal 35/51 us tail at 256K,
  b=148, k=2048 without touching the cells that did not need it.
* ``aim_wide``: ``max(2k, length / 128)``, gvr_2's choice.  Fewer undershoots, more
  candidates to stage and classify.  Kept to measure against.

Both cap the aim at three quarters of the stage so a typical overshoot still fits, and when the
room above the aim is under three sigma of the survivor count they fall back to the balanced
aim ``(k + stage) / 2`` (see ``_capped``).
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath

__all__ = ["aim_tight", "aim_wide"]


@cute.jit
def _capped(aim, k: cutlass.Constexpr, cap: cutlass.Constexpr, length, samples):
    """Keep the aim clear of the stage: when the overshoot room above the aim is under three
    sigma of the survivor count, balance the two tails instead.

    ``cap = 3/4 stage``.  The row's survivor count spreads with sigma ``sqrt(aim * length /
    samples)``; at 1M fp32 rows on an 8K stage (99 KB parts) the tight aim sits at 6144 and
    the 2048 of room is 1.7 sigma with a 4096 sample, 2.3 with 8192: one row in a hundred
    overflows and re-walks, and a batch of 142 rows nearly always waits for one (L40S 1M b=142
    k=2048: median 1000 us against an 850 us phase sum).  The balanced aim ``(k + stage) / 2``
    puts the undershoot margin and the overshoot room at the same distance (3072 each at
    k=2048: 3.8 sigma with the 8192 sample).  Rows whose room is comfortable keep their aim.
    """
    stage = cutlass.const_expr(4 * cap // 3)
    room = (cutlass.Int32(stage) - aim).to(cutlass.Float32)
    # rcp_approx: one MUFU instead of a full division sequence; a statistical margin needs no
    # more (``samples`` is a runtime count since v0.1.24, so this is no longer a constant)
    variance = aim.to(cutlass.Float32) * (
        length.to(cutlass.Float32) * cute.arch.rcp_approx(samples.to(cutlass.Float32))
    )
    if (aim > cutlass.Int32(cap)) | (room * room < cutlass.Float32(9.0) * variance):
        aim = cutlass.Int32((k + stage) // 2)
    return aim


@cute.jit
def aim_tight(
    k: cutlass.Constexpr,
    length,
    rows,
    samples,
    cap: cutlass.Constexpr,
    margin_frac: cutlass.Constexpr = 0.125,
    z: cutlass.Constexpr = 3.5,
    floor_rows: cutlass.Constexpr = 32,
    margin_small: cutlass.Constexpr = 0.5,
):
    """Target survivor count for a row of ``length`` in a grid of ``rows`` rows.

    Grids of ``floor_rows`` rows or more: ``k + max(margin_frac * k, length / 256)`` raised to
    the ``z``-sigma statistical floor (the batch waits for its slowest row, so an undershoot
    re-walk anywhere costs the whole batch; at 3.5 sigma one row in 4000 re-walks).  The floor
    decides on these grids, so the fixed margin is small (k / 8: 64K b=148 k=2048 10.54 ->
    10.22 us on B200 against the earlier k / 2, which exceeded the floor there).  Below
    ``floor_rows`` there is no floor and ``margin_small`` (k / 2) alone covers the spread: with
    a 4096 sample at 64K, k=2048 the k / 8 margin was 1.3 sigma and one row in ten re-walked.
    A lower sigma is not a trade worth making: z=3.0 gained 2% and a single re-walk in a 256-row
    batch costs 30% (measured z=2.5: 12.5 -> 16.5 us).
    """
    margin = cutlass.Int32(
        int(k * margin_small)
    )  # assigned before the branch: the DSL drops names first defined inside one
    if rows >= cutlass.Int32(floor_rows):
        margin = cutlass.Int32(int(k * margin_frac))
    by_length = length >> cutlass.Int32(8)
    if by_length > margin:
        margin = by_length
    aim = cutlass.Int32(k) + margin
    if rows >= cutlass.Int32(floor_rows):
        per_sample = length.to(cutlass.Float32) * cute.arch.rcp_approx(
            samples.to(cutlass.Float32)
        )
        zq = cutlass.Float32(z) * cmath.sqrt(per_sample)
        root = (zq + cmath.sqrt(zq * zq + cutlass.Float32(4.0 * k))) * cutlass.Float32(
            0.5
        )
        floor = (root * root).to(cutlass.Int32) + cutlass.Int32(1)
        if floor > aim:
            aim = floor
    return _capped(aim, k, cap, length, samples)


@cute.jit
def aim_wide(
    k: cutlass.Constexpr,
    length,
    rows,
    samples,
    cap: cutlass.Constexpr,
    margin_frac: cutlass.Constexpr = 0.125,
    z: cutlass.Constexpr = 3.5,
    floor_rows: cutlass.Constexpr = 32,
    margin_small: cutlass.Constexpr = 0.5,
):
    """Target survivor count: twice k, or length / 128 if larger (the tight aim's knobs unused)."""
    aim = cutlass.Int32(2 * k)
    by_length = length >> cutlass.Int32(7)
    if by_length > aim:
        aim = by_length
    return _capped(aim, k, cap, length, samples)
