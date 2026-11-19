# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The host planner: FlashInfer-style request metadata -> work units for the kernel and fixup groups
for the split-KV merge.

A work unit is one CTA: (request, kv head, Q tile of ``tokens_per_tile`` tokens x ``g`` heads) and a
chunk range that is either ALL present chunks of its rows (mode 0: the CTA folds the chunks in TMEM
and writes the final output) or exactly ONE chunk (mode 1: the CTA writes the chunk's partial
``(m, l, O)`` rows to a workspace slot and the fixup kernel folds the slots in chunk order). Any other
range would merge a prefix-of-chunks state with a later state, which the (non-associative) merge
does not allow. Which of the two kinds a Q tile gets is the planner's choice (it may use the SM
count); every choice computes the same bits (R3).

Positions are bottom-right causal: token ``i`` of request ``b`` (``0 <= i < q_len``) sits at
``p = kv_len - q_len + i`` and sees keys ``[max(0, p - window_left), p]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .spec import TILE_KEYS, chunk_len

DESC_FIELDS = 16
# descriptor fields (int32); F_QBASE = qo_indptr[req], the packed token index of the request's first token;
# F_TILE_LO / F_TILE_HI are the unit's OWN tiles (the row block's visible range clipped to the unit's chunks): every role
# of the kernel walks exactly that range and derives the chunk boundaries from the absolute tile index
(
    F_REQ,
    F_QTOK0,
    F_ROWS,
    F_KVHEAD,
    F_CHUNK_LO,
    F_CHUNK_HI,
    F_MODE,
    F_SLOT,
    F_KVLEN,
    F_QLEN,
    F_TILE_LO,
    F_TILE_HI,
    F_QBASE,
    F_GROUP,
) = range(14)
# F_GROUP: the unit's split group (an index into Plan.groups) when the group's chunk fold runs IN-KERNEL (the unit that completes the
# group folds it: blocks/correction.py), -1 otherwise (mode 0, or a group left to the fixup kernel)
INKERNEL_MAX_ROWS = 32  # merge == "auto": groups of at most this many rows (every decode unit) and at most INKERNEL_MAX_CHUNKS chunks fold
INKERNEL_MAX_CHUNKS = 2  # in-kernel (measured faster for small decode batches, within noise at B = 1); wider groups (prefill split
FUSE2_MIN_ROWS = 8  # D = 256 two-chunk rule (build_plan): fused decode-sized units only from this many live rows (g x q_len) per unit
# ---- the unit kinds and the launch grid of a Swaps-body batch (``small_swap``: every unit at most ``swap_max_rows`` rows): a makespan
# model decides per batch (``swaps_unit_kinds``); every choice computes the same bits (R3). Costs are in units of one 128-key tile of
# D = 128 with every SM streaming at the HBM rate; a D-wide tile costs D / 128 of it. Empirical constants of the Swaps body
STREAM_SAT_FRACTION = {
    64: 0.78,
    128: 0.42,
    256: 0.30,
}  # per head dim: the fraction of the SMs whose CTAs, each walking one unit, together reach the HBM rate (the HBM rate over a lone
# CTA's stream rate, over the SM count): a wave of fewer CTAs is bound by the per-CTA rate, a fuller wave by the HBM rate
GRID_SEARCH_ARCHES = (
    (10, 7),
)  # the parts whose stream rates the launch-grid model was fitted on: elsewhere the fused walk keeps the
# full grid and the model decides only fuse-or-split (``build_plan(grid_search=...)``)
GRID_MIN_FRACTION = 0.45  # a multi-wave fused walk keeps at least this fraction of the SMs (and the saturating fraction): above it
# the HBM rate holds and fewer concurrent streams run the HBM more efficiently
GRID_LONG_TILES = 16  # units of fewer tiles keep the default grid unless its last wave is under a third full: their boundary
# pauses outweigh what integral waves save
FIRST_WAVE_RAMP = 2.0  # the first wave's loads contend (every CTA's opening burst of tiles): a cost growing with the CTAs of the
# first wave and the tile bytes
UNIT_BUBBLE = {
    64: 0.6,
    128: 2.0,
    256: 2.0,
}  # per head dim: a CTA's stream pauses at every unit boundary (the next unit's opening tiles and the epilogue hand-off), a fixed
# cost per unit on the per-CTA-bound side, hidden while the wave is bound by the HBM rate
FIXUP_BASE_COST = 2.9  # the fixup kernel's cost on the critical path beyond the attention kernel's end: a fixed part ...
FIXUP_ROW_COST = 0.0012  # ... plus this per live row of its groups
INKERNEL_FOLD_COST = 1.4  # the in-kernel last-arriver fold of a group of at most INKERNEL_MAX_CHUNKS chunks, per wave of groups
WIDE_SPLIT_SM_FRACTION = 4  # merge-agnostic unit-kind rule of ``"auto"`` for units wider than INKERNEL_MAX_ROWS rows (prefill tiles): they
# split only when the fused units would occupy fewer than 1 / this of the SMs. A wide split unit costs its 128 x D fp32 partial rows
# and the fixup's fold of them, so a split that merely balances a part-filled machine loses to the fused unit; the narrow (decode)
# units keep the 2 x SM rule below
INKERNEL_MAX_CHUNKS_OF = {
    64: INKERNEL_MAX_CHUNKS,
    128: INKERNEL_MAX_CHUNKS,
    256: 0,
}  # D = 256: the fixup kernel folds every group (the fixup placement measured faster at B = 1 for 256-wide rows;
# the in-kernel fold is 64 threads per row there)
# units: 128 rows x up to 8 chunks per last arriver) and longer chains take the fixup kernel: the fold runs in
# the last arriver's correction warpgroup and stalls that CTA's MMA and softmax through the stats ring, and every
# group waiting on a chunk of that CTA completes later -- a large slowdown with 16-chunk groups, and with 64-chunk groups
# the fold alone took as long as the kernel; a fold by the idle warps is the
# open item
GROUP_FIELDS = 10
# fixup group fields (int32)
(
    G_SLOT,
    G_ROWS,
    G_CHUNK_LO,
    G_NCHUNKS,
    G_QTOK0,
    G_KVHEAD,
    G_KVLEN,
    G_QLEN,
    G_QBASE,
    G_REQ,
) = range(10)
COUNT_FIELDS = 8
# the counts of a plan under a fixed launch (int32), read at entry by the kernels compiled for it: the units the attention grid
# walks, the split groups, the groups the fixup kernel folds, and the CTAs that walk the units (the CTAs of the launch grid beyond
# them idle). Those kernels take the counts from this table in place of launch arguments, so a launch captured in a CUDA graph
# -- one grid for every plan -- follows a new plan written into the same buffers
(
    N_UNITS,
    N_GROUPS,
    N_FIXUP_GROUPS,
    N_WALK,
) = range(4)
WS_ROW = 132  # fp32 per partial row AT D = 128: O[0..D) then m at [D], l at [D+1] (padded to a 16-byte multiple).
# Contract: the kernel writes exactly [0, D+2) of every partial row of a present chunk and the fixup
# reads exactly those; the padding [D+2, WS_ROW) is never written nor read (the workspace is the
# caller's uninitialised buffer, so its padding is not a function of the inputs)


def ws_row(D: int) -> int:
    """fp32 per partial row of the split workspace at head dim ``D``: ``D + 2`` rounded up to a multiple of 4 (16 bytes):
    68 / 132 / 260 for D = 64 / 128 / 256. ``WS_ROW`` is this value at D = 128 (kept for the D = 128 call sites)."""
    return -(-(D + 2) // 4) * 4


assert ws_row(128) == WS_ROW


@dataclass
class Plan:
    desc: List[List[int]] = field(default_factory=list)  # per unit
    groups: List[List[int]] = field(
        default_factory=list
    )  # per split group (ALL groups: the tests read every group's partials)
    group_inkernel: List[bool] = field(
        default_factory=list
    )  # per group: folded in-kernel by its last arriver (else by the fixup kernel)
    ws_rows: int = 0  # partial rows in the workspace
    tokens_per_tile: int = 128
    g: int = 1
    ws_row: int = WS_ROW  # fp32 per partial row (``ws_row(D)``)
    num_launch: Optional[int] = (
        None  # the paired placement's launch count (pairs of units), else None
    )
    grid: int = 0  # the launch grid the planner chose for a Swaps-body batch (``swaps_unit_kinds``), 0 = min(units, SMs); a launch
    # parameter (I10): an explicit ``AttnConfig.grid`` overrides it

    @property
    def num_units(self) -> int:
        return len(self.desc)

    @property
    def fixup_groups(self) -> List[List[int]]:
        """The groups the fixup kernel folds (the others are folded in-kernel)."""
        return [
            gr
            for gr, ik in zip(self.groups, self.group_inkernel, strict=True)
            if not ik
        ]

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def num_fixup_groups(self) -> int:
        return len(self.group_inkernel) - sum(self.group_inkernel)


def om_in_tmem(D: int) -> bool:
    """Whether the body keeps the merged chunk state ``O_m`` in TMEM next to ``S | P | O_c`` (``128 + 64 + 2 D <= 512``: D <= 128). At D = 256
    it lives in a per-CTA global scratch, so a fused multi-chunk unit pays one L2 round trip of its 128 x D fp32 state per chunk end -- which
    is why the planner's ``auto`` splits decode-sized units there (``build_plan``). The body (``blocks/operands.py``) reads the same rule."""
    return 128 + 64 + 2 * D <= 512


def kv_lengths(
    kv_indptr: Sequence[int], kv_last_page_len: Sequence[int], page: int
) -> List[int]:
    out = []
    for b in range(len(kv_indptr) - 1):
        n = kv_indptr[b + 1] - kv_indptr[b]
        out.append(0 if n == 0 else (n - 1) * page + kv_last_page_len[b])
    return out


def chunk_tile_counts(
    tile_lo: int, tile_hi: int, chunk_lo: int, chunk_hi: int, tpc: int
) -> tuple:
    """A unit's chunks as ``(n_chunks, first_tiles, last_tiles)``: every chunk but the first and the last holds ``tpc`` tiles, the
    first ``min(tile_hi, (chunk_lo + 1) tpc) - tile_lo`` and the last ``tile_hi - max(tile_lo, (chunk_hi - 1) tpc)`` (one chunk:
    its own tiles). The planner prices the split kind from these three numbers instead of listing every chunk."""
    n = chunk_hi - chunk_lo
    if n <= 1:
        return n, tile_hi - tile_lo, tile_hi - tile_lo
    return (
        n,
        min(tile_hi, (chunk_lo + 1) * tpc) - tile_lo,
        tile_hi - max(tile_lo, (chunk_hi - 1) * tpc),
    )


def _swaps_lengths(
    units: Sequence[tuple], split: Sequence[bool], inkernel_max_chunks: int, tpc: int
) -> tuple:
    """The multiset of walk-unit lengths (tiles) of a Swaps-body batch as ``{length: count}``, with the fixup rows and the in-kernel
    fold groups the split units add: ``units[i] = (tiles, (n_chunks, first_tiles, last_tiles), rows)``, split into one unit per
    chunk when ``split[i]``."""
    counts: Dict[int, int] = {}
    fixup_rows = 0
    inkernel_groups = 0
    for (tiles, (n_chunks, first, last), rows), s in zip(units, split, strict=True):
        if s and n_chunks > 1:
            counts[first] = counts.get(first, 0) + 1
            counts[last] = counts.get(last, 0) + 1
            if n_chunks > 2:
                counts[tpc] = counts.get(tpc, 0) + n_chunks - 2
            if n_chunks > inkernel_max_chunks:
                fixup_rows += rows
            else:
                inkernel_groups += 1
        else:
            counts[tiles] = counts.get(tiles, 0) + 1
    return counts, fixup_rows, inkernel_groups


def _swaps_walk_cost(
    counts: Dict[int, int],
    fixup_rows: int,
    inkernel_groups: int,
    grid: int,
    sm_count: int,
    D: int,
    relax_last_wave: bool,
) -> float:
    """``swaps_kind_cost`` over a length multiset: the walk takes the units longest first in waves of ``grid``; a wave costs its longest
    unit's tiles at the rate of its active CTAs (the HBM rate from ``STREAM_SAT_FRACTION`` of the SMs up; below it the per-CTA rate
    plus the unit boundary's bubble), the first wave adds its opening burst's contention, the split groups add the fixup kernel's cost
    or the in-kernel fold's per wave of folding units. The waves are walked over the runs of equal lengths, so the cost of a batch
    is linear in its distinct lengths and its waves, never in its chunks."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    scale = D / 128
    grid = max(1, min(grid, total))
    sat = STREAM_SAT_FRACTION[D] * sm_count
    runs = sorted(counts.items(), reverse=True)  # (length, count), longest first
    n_waves = -(-total // grid)
    cost = 0.0
    r = 0
    left = runs[0][1]  # units left in the current run
    for i in range(n_waves):
        width = min(grid, total - i * grid)
        top = runs[r][0]
        streamed = top * scale * width / sm_count
        if relax_last_wave and n_waves > 1 and i == n_waves - 1:
            cost += streamed
        else:
            cost += max(top * scale * sat / sm_count + UNIT_BUBBLE[D], streamed)
        take = width
        while take > 0:
            c = min(take, left)
            take -= c
            left -= c
            if left == 0 and r + 1 < len(runs):
                r += 1
                left = runs[r][1]
    cost += FIRST_WAVE_RAMP * scale * grid / sm_count
    if fixup_rows:
        cost += FIXUP_BASE_COST + FIXUP_ROW_COST * fixup_rows
    if inkernel_groups:
        cost += INKERNEL_FOLD_COST * -(-inkernel_groups // grid)
    return cost


def swaps_kind_cost(
    units: Sequence[tuple],
    split: Sequence[bool],
    grid: int,
    sm_count: int,
    D: int,
    inkernel_max_chunks: int,
    relax_last_wave: bool = False,
) -> float:
    """The predicted makespan of a Swaps-body batch whose unit ``i`` -- ``units[i] = (tiles, (n_chunks, first_tiles, last_tiles),
    rows)``, the chunks as ``chunk_tile_counts`` gives them -- is split into one unit per chunk when ``split[i]``, walked by ``grid``
    persistent CTAs (``_swaps_walk_cost``). ``relax_last_wave``: a multi-wave walk's last wave is charged at the HBM rate whatever its
    width -- the form used off the parts the per-CTA rate was fitted on, so that a part-filled last wave alone never turns a fused
    batch into a split one."""
    counts, fixup_rows, inkernel_groups = _swaps_lengths(
        units, split, inkernel_max_chunks, chunk_len(D) // TILE_KEYS
    )
    return _swaps_walk_cost(
        counts, fixup_rows, inkernel_groups, grid, sm_count, D, relax_last_wave
    )


def swaps_fused_grids(
    num_units: int, sm_count: int, D: int, tiles: int = GRID_LONG_TILES
) -> List[int]:
    """The launch grids a fused walk of ``num_units`` equal units of ``tiles`` tiles may take: ``min(num_units, sm_count)`` and, for
    every wave count, the grid that spreads the units over integral waves (``ceil(units / waves)``) while keeping ``GRID_MIN_FRACTION``
    of the SMs and the saturating CTA count, so that no part-filled last wave of long units streams below the HBM rate; units under
    ``GRID_LONG_TILES`` tiles get the alternatives only when the default grid's last wave is under a third full."""
    grids = [min(num_units, sm_count)]
    waves = -(-num_units // sm_count)
    if tiles < GRID_LONG_TILES and 3 * (num_units - (waves - 1) * sm_count) >= sm_count:
        return grids
    # the floor is at least one CTA: on a part with one or two SMs the integral-wave grid is one CTA, reached at ``num_units`` waves
    floor = max(1.0, sm_count * max(GRID_MIN_FRACTION, STREAM_SAT_FRACTION[D]))
    while waves <= num_units:
        g = -(-num_units // waves)
        if g < floor:
            break
        if g <= sm_count and g not in grids:
            grids.append(g)
        waves += 1
    return grids


def swaps_unit_kinds(
    units: Sequence[tuple],
    sm_count: int,
    D: int,
    inkernel_max_chunks: int,
    grid_search: bool = True,
) -> tuple:
    """The unit kinds and the launch grid of a Swaps-body batch under ``"auto"`` (``units`` as in ``swaps_kind_cost``): ``(split, grid)``
    with ``split[i]`` whether unit ``i`` splits into its chunks. The candidates are every unit fused on each grid of
    ``swaps_fused_grids`` and every multi-chunk unit split on the full grid; the cheapest wins (among equal fused grids the one of
    most waves: the fewest CTAs that still stream at the HBM rate). ``grid_search`` False keeps the fused walk on the full grid
    (the parts whose stream rates the grid model was not fitted on: ``GRID_SEARCH_ARCHES``); the fuse-or-split choice stays.
    A placement: every choice computes the same bits."""
    n = len(units)
    if n == 0:
        return [], 0
    tpc = chunk_len(D) // TILE_KEYS
    split = [u[1][0] > 1 for u in units]
    split_all, best_grid = swaps_decide(
        n,
        max(u[0] for u in units),
        _swaps_lengths(units, [False] * n, inkernel_max_chunks, tpc),
        _swaps_lengths(units, split, inkernel_max_chunks, tpc) if any(split) else None,
        sm_count,
        D,
        grid_search,
    )
    return (split, 0) if split_all else ([False] * n, best_grid)


def swaps_decide(
    n: int,
    max_tiles: int,
    fused_lengths: tuple,
    split_lengths: Optional[tuple],
    sm_count: int,
    D: int,
    grid_search: bool = True,
) -> tuple:
    """``swaps_unit_kinds`` over the length summaries of the batch (``_swaps_lengths``: every unit fused, and every multi-chunk
    unit split -- ``None`` when no unit has a second chunk): ``(split, grid)``, whether the multi-chunk units split and, when
    they do not, the fused walk's grid. ``n`` units, the longest of ``max_tiles`` tiles."""
    f_counts, f_fixup_rows, f_inkernel = fused_lengths
    best_grid, best_cost = 0, float("inf")
    if grid_search:
        grids = swaps_fused_grids(n, sm_count, D, max_tiles)
    else:
        # off the fitted parts: the full grid and the even-waves grid (the units spread over the full grid's wave count: no fitted
        # constant), so that a part-filled last wave does not stream alone
        grids = [min(n, sm_count)]
        waves = -(-n // sm_count)
        even = -(-n // waves)
        thin_tail = 3 * (n - (waves - 1) * sm_count) < sm_count
        if even not in grids and (max_tiles >= GRID_LONG_TILES or thin_tail):
            grids.append(even)
    for g in grids:
        c = _swaps_walk_cost(f_counts, f_fixup_rows, f_inkernel, g, sm_count, D, False)
        if c <= best_cost:
            best_grid, best_cost = g, c
    if split_lengths is not None:
        s_counts, s_fixup_rows, s_inkernel = split_lengths
        c = _swaps_walk_cost(
            s_counts, s_fixup_rows, s_inkernel, sm_count, sm_count, D, False
        )
        fused_cmp = (
            best_cost
            if grid_search
            else _swaps_walk_cost(
                f_counts, f_fixup_rows, f_inkernel, best_grid, sm_count, D, True
            )
        )
        if c < fused_cmp:
            return True, 0
    return False, best_grid


def budget_unit_kinds(
    fused: Sequence[bool],
    tiles: Sequence[int],
    chunks: Sequence[int],
    rows: Sequence[int],
    unit_budget: int = 0,
    row_budget: int = -1,
) -> List[bool]:
    """The unit kinds of a plan held to a fixed capacity: ``fused[i]`` is the kind the planner chose for Q tile ``i`` (``tiles[i]``
    key tiles in ``chunks[i]`` chunks, ``rows[i]`` live rows); the result keeps a split tile split only while the plan stays within
    ``unit_budget`` units (0: no bound) and ``row_budget`` partial rows of workspace (-1: no bound), and fuses the rest. The split
    tiles are taken longest first, a whole class of equal length at a time, and the first class that does not fit ends the walk:
    a split shortens the walk only when no longer unit stays fused. Within the budgets the kinds are the planner's own. A
    placement: every choice computes the same bits."""
    out = [True] * len(fused)
    classes: Dict[int, List[int]] = {}
    for i, f in enumerate(fused):
        if not f:
            classes.setdefault(tiles[i], []).append(i)
    units, ws_rows = len(fused), 0
    for length in sorted(classes, reverse=True):
        members = classes[length]
        extra = sum(chunks[i] - 1 for i in members)
        need = sum(chunks[i] * rows[i] for i in members)
        if (unit_budget > 0 and units + extra > unit_budget) or (
            row_budget >= 0 and ws_rows + need > row_budget
        ):
            break
        units += extra
        ws_rows += need
        for i in members:
            out[i] = False
    return out


def build_plan(
    qo_indptr: Sequence[int],
    kv_indptr: Sequence[int],
    kv_last_page_len: Sequence[int],
    Hq: int,
    Hkv: int,
    D: int,
    page: int,
    window_left: int = -1,
    split_policy: str = "auto",
    sm_count: int = 204,
    q_rows: int = 128,
    merge: str = "auto",
    small_swap: bool = False,
    swap_max_rows: int = INKERNEL_MAX_ROWS,
    grid: int = 0,
    grid_search: bool = True,
    unit_budget: int = 0,
    row_budget: int = -1,
    null_requests: bool = False,
) -> Plan:
    """``split_policy``: ``"auto"`` chooses the unit kinds. A Swaps-body batch (``small_swap``: every unit at most ``swap_max_rows``
    live rows) takes the kinds and the launch grid of ``swaps_unit_kinds`` (a makespan model over the SM count: every unit fused on
    the grid of integral waves, or every multi-chunk unit split). Otherwise a multi-chunk tile of at most ``INKERNEL_MAX_ROWS`` rows
    (a decode unit) splits when the fused unit count would leave the machine underfilled (``< 2 x SM``), a wider tile (a prefill
    unit) only when the fused units would fill under ``1 / WIDE_SPLIT_SM_FRACTION`` of the SMs -- and, at head dims whose merged state
    is not in TMEM (D = 256), every multi-chunk tile of at most ``INKERNEL_MAX_ROWS`` rows (decode units) whatever the count: a fused
    decode unit there merges a 128 x 256 fp32 state through the global scratch at every chunk end, and the split kind measured
    faster at long KV lengths -- except the TWO-chunk decode-sized units of at least ``FUSE2_MIN_ROWS`` live rows in a batch whose
    fused unit count exceeds half the SM count, which stay fused (one in-kernel merge against two unit ramps and the fixup node);
    ``"split"`` / ``"fused"`` force one kind (R3: same bits).
    ``merge``: where a split group's chunk fold runs -- ``"auto"`` (in-kernel for groups of at most
    ``INKERNEL_MAX_ROWS`` rows, the fixup kernel otherwise), ``"fixup"`` (always the kernel), ``"inkernel"``
    (always in-kernel); the same fold in the same order either way (R3: same bits).
    ``grid``: the persistent grid the launch will use (0: the planner's ``Plan.grid`` if it chose one, else ``min(units, sm_count)``),
    so that the unit order's waves (``snake_order``) are the waves the CTAs actually walk.
    ``unit_budget`` / ``row_budget``: the fixed capacity of a launch captured in a CUDA graph (``budget_unit_kinds``: at most
    ``unit_budget`` units and ``row_budget`` partial rows; the tiles beyond stay fused). ``null_requests``: a request with query tokens and no keys gets no unit (its output rows are
    the caller's) instead of being refused."""
    assert merge in ("auto", "fixup", "inkernel"), merge
    assert Hq % Hkv == 0, (Hq, Hkv)
    g = Hq // Hkv
    assert 1 <= g <= q_rows, g
    tpt = q_rows // g
    T, C = TILE_KEYS, chunk_len(D)
    B = len(qo_indptr) - 1
    kv_lens = kv_lengths(kv_indptr, kv_last_page_len, page)
    tiles = []  # (b, h, tok0, ntok, chunk_lo, chunk_hi, tile_lo, tile_hi, kv_len, q_len)
    for b in range(B):
        q_len = qo_indptr[b + 1] - qo_indptr[b]
        kv_len = kv_lens[b]
        if q_len == 0 or (null_requests and kv_len == 0):
            continue
        assert kv_len >= q_len, (
            f"request {b}: kv_len {kv_len} < q_len {q_len} (bottom-right causal needs kv_len >= q_len)"
        )
        for tok0 in range(0, q_len, tpt):
            ntok = min(tpt, q_len - tok0)
            p_first = kv_len - q_len + tok0
            p_last = p_first + ntok - 1
            L_max = p_last + 1
            lo_min = max(0, p_first - window_left) if window_left >= 0 else 0
            chunk_lo, chunk_hi = lo_min // C, -(-L_max // C)
            tile_lo, tile_hi = lo_min // T, -(-L_max // T)
            for h in range(Hkv):
                tiles.append(
                    (
                        b,
                        h,
                        tok0,
                        ntok,
                        chunk_lo,
                        chunk_hi,
                        tile_lo,
                        tile_hi,
                        kv_len,
                        q_len,
                    )
                )
    if split_policy == "split":
        do_split_narrow = do_split_wide = True
    elif split_policy == "fused":
        do_split_narrow = do_split_wide = False
    else:
        do_split_narrow = len(tiles) < 2 * sm_count
        do_split_wide = WIDE_SPLIT_SM_FRACTION * len(tiles) < sm_count
    # the merged chunk state's home decides the D = 256 rules below: the row body keeps it in a global scratch at D = 256, the Swaps-AB
    # kernel (every unit of the batch <= 32 rows under small = "swap") in TMEM at every D -- those batches take the D = 128 rules
    swap_batch = (
        small_swap and bool(tiles) and all(t[3] * g <= swap_max_rows for t in tiles)
    )  # the batch takes the Swaps body (api.SWAPS_MAX_ROWS)
    om_tmem = om_in_tmem(D) or swap_batch
    split_small = (
        split_policy == "auto" and not om_tmem
    )  # D = 256: decode-sized multi-chunk units always split (see the docstring) ...
    # ... except two-chunk units of >= FUSE2_MIN_ROWS live rows when the fused units alone fill more than half the machine: there the split
    # kind pays two unit ramps and the fixup node (whose cost grows with the rows) against one in-kernel merge (a fixed 128 x D scratch pass
    # whatever the live rows), measured fused faster with several live rows per unit and slower with one live row -- D = 256;
    # with fewer units the split kind's parallelism wins (B = 1). A placement (R3)
    fuse_two_chunk = (
        split_policy == "auto" and not om_tmem and 2 * len(tiles) > sm_count
    )
    inkernel_max_chunks = INKERNEL_MAX_CHUNKS if om_tmem else INKERNEL_MAX_CHUNKS_OF[D]
    plan = Plan(tokens_per_tile=tpt, g=g, ws_row=ws_row(D))
    tpc = C // T
    swaps_kinds: Optional[List[bool]] = (
        None  # per tile: split? (Swaps-body batches under "auto")
    )
    if split_policy == "auto" and swap_batch:
        swaps_kinds, plan.grid = swaps_unit_kinds(
            [
                (thi - tlo, chunk_tile_counts(tlo, thi, clo, chi, tpc), ntok * g)
                for _, _, _, ntok, clo, chi, tlo, thi, _, _ in tiles
            ],
            sm_count,
            D,
            inkernel_max_chunks,
            grid_search,
        )
        if plan.grid == min(len(tiles), sm_count):
            plan.grid = 0  # the default launch grid
    kinds: List[bool] = []  # per tile: fused?
    for i, tile in enumerate(tiles):
        ntok, clo, chi = tile[3:6]
        rows = ntok * g
        if swaps_kinds is not None:
            fused_unit = chi - clo == 1 or not swaps_kinds[i]
        else:
            two_chunk_fused = (
                fuse_two_chunk
                and chi - clo == 2
                and FUSE2_MIN_ROWS <= rows <= INKERNEL_MAX_ROWS
            )
            do_split = do_split_narrow if rows <= INKERNEL_MAX_ROWS else do_split_wide
            fused_unit = (
                chi - clo == 1
                or two_chunk_fused
                or not (do_split or (split_small and rows <= INKERNEL_MAX_ROWS))
            )
        kinds.append(fused_unit)
    if unit_budget > 0 or row_budget >= 0:
        kinds = budget_unit_kinds(
            kinds,
            [t[7] - t[6] for t in tiles],
            [t[5] - t[4] for t in tiles],
            [t[3] * g for t in tiles],
            unit_budget,
            row_budget,
        )
    for (b, h, tok0, ntok, clo, chi, tlo, thi, kv_len, q_len), fused_unit in zip(
        tiles, kinds, strict=True
    ):
        rows = ntok * g
        if fused_unit:
            plan.desc.append(
                [
                    b,
                    tok0,
                    rows,
                    h,
                    clo,
                    chi,
                    0,
                    0,
                    kv_len,
                    q_len,
                    tlo,
                    thi,
                    qo_indptr[b],
                    -1,
                    0,
                    0,
                ]
            )
            continue
        base = plan.ws_rows
        n = chi - clo
        ik = merge == "inkernel" or (
            merge == "auto" and rows <= INKERNEL_MAX_ROWS and n <= inkernel_max_chunks
        )
        grp = len(plan.groups) if ik else -1
        plan.groups.append(
            [base, rows, clo, n, tok0, h, kv_len, q_len, qo_indptr[b], b]
        )
        plan.group_inkernel.append(ik)
        for j in range(clo, chi):
            tl, th = (
                max(tlo, j * tpc),
                min(thi, (j + 1) * tpc),
            )  # the chunk's tiles present for the row block
            plan.desc.append(
                [
                    b,
                    tok0,
                    rows,
                    h,
                    j,
                    j + 1,
                    1,
                    base + (j - clo) * rows,
                    kv_len,
                    q_len,
                    tl,
                    th,
                    qo_indptr[b],
                    grp,
                    0,
                    0,
                ]
            )
        plan.ws_rows += n * rows
    # longest units first, then the waves of ``grid`` units alternate direction (``snake_order``): the persistent CTAs walk units
    # u = bidx, bidx + grid, ..., so CTA j takes the j-th unit of every wave and the alternation pairs a wave's long units with the
    # next wave's short ones. A placement: the order of units reaches no value (every unit restarts its state)
    plan.desc.sort(key=lambda d: -(d[F_TILE_HI] - d[F_TILE_LO]))
    plan.desc = snake_order(
        plan.desc,
        min(len(plan.desc), grid if grid > 0 else (plan.grid or sm_count)),
    )
    return plan


def snake_order(units: list, grid: int) -> list:
    """``units`` (longest first) in the order the persistent walk ``u = bidx, bidx + grid, ...`` balances best without a per-CTA
    table: consecutive waves of ``grid`` units alternate direction, so the CTA that takes the longest unit of one wave takes the
    shortest of the next (a boustrophedon over the LPT order). With a descending order alone, CTA 0 takes the longest unit of EVERY
    wave and finishes last. A placement: the order of units reaches no value."""
    if grid <= 0:
        return list(units)
    out = []
    for w, start in enumerate(range(0, len(units), grid)):
        wave = units[start : start + grid]
        out.extend(wave[::-1] if w % 2 else wave)
    return out


class ArrayPlan:
    """A plan held as int32 arrays (``build_decode_plan``): the tables of ``Plan`` -- ``desc_array`` ``(units, DESC_FIELDS)``,
    ``groups_array`` and ``fixup_array`` ``(groups, GROUP_FIELDS)`` -- ready to be copied to the device; ``desc`` / ``groups`` /
    ``fixup_groups`` / ``group_inkernel`` give ``Plan``'s lists."""

    num_launch: Optional[int] = (
        None  # the pairs of the paired placement's table, set by the launch that builds it
    )
    launch_array: Optional[np.ndarray] = None  # that table
    walk = 0  # ... and the CTAs that walk it
    shape: Optional[tuple] = (
        None  # what the tables depend on besides the KV lengths written in them (``build_array_plan``)
    )

    def __init__(
        self,
        desc: np.ndarray,
        groups: np.ndarray,
        inkernel: np.ndarray,
        ws_rows: int,
        tokens_per_tile: int,
        g: int,
        row: int,
        grid: int = 0,
    ):
        self.desc_array, self.groups_array, self.inkernel_array = desc, groups, inkernel
        self.grid = grid
        self.fixup_array = groups[~inkernel]
        self.ws_rows, self.tokens_per_tile, self.g, self.ws_row = (
            ws_rows,
            tokens_per_tile,
            g,
            row,
        )

    def same_shape(self, shape: tuple) -> bool:
        """Whether a batch of this ``shape`` (the planner's arguments and every Q tile's tile and chunk range) has this
        plan's tables up to the KV lengths."""
        mine = self.shape
        return (
            mine is not None
            and mine[0] == shape[0]
            and all(
                a.shape == b.shape and bool((a == b).all())
                for a, b in zip(mine[1:], shape[1:], strict=True)
            )
        )

    def with_kv_lens(self, kv_lens: np.ndarray) -> "ArrayPlan":
        """This plan for other KV lengths of the same shape: the same units, groups and order, the lengths rewritten."""
        desc = self.desc_array.copy()
        desc[:, F_KVLEN] = kv_lens[desc[:, F_REQ]]
        groups = self.groups_array.copy()
        groups[:, G_KVLEN] = kv_lens[groups[:, G_REQ]]
        plan = ArrayPlan(
            desc,
            groups,
            self.inkernel_array,
            self.ws_rows,
            self.tokens_per_tile,
            self.g,
            self.ws_row,
            self.grid,
        )
        plan.shape = self.shape
        if self.launch_array is not None:
            plan.launch_array = self.launch_array.copy()
            plan.launch_array[:, F_KVLEN] = kv_lens[plan.launch_array[:, F_REQ]]
            plan.num_launch, plan.walk = self.num_launch, self.walk
        return plan

    @property
    def num_units(self) -> int:
        return self.desc_array.shape[0]

    @property
    def num_groups(self) -> int:
        return self.groups_array.shape[0]

    @property
    def num_fixup_groups(self) -> int:
        return self.fixup_array.shape[0]

    @property
    def desc(self) -> List[List[int]]:
        return self.desc_array.tolist()

    @property
    def groups(self) -> List[List[int]]:
        return self.groups_array.tolist()

    @property
    def fixup_groups(self) -> List[List[int]]:
        return self.fixup_array.tolist()

    @property
    def group_inkernel(self) -> List[bool]:
        return self.inkernel_array.tolist()


def _length_counts(lengths: np.ndarray) -> Dict[int, int]:
    values, counts = np.unique(lengths, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist(), strict=True))


def build_decode_plan(
    kv_lens: np.ndarray,
    q_len: int,
    Hq: int,
    Hkv: int,
    D: int,
    window_left: int = -1,
    split_policy: str = "auto",
    sm_count: int = 204,
    **kw,
) -> "ArrayPlan":
    """``build_array_plan`` for a decode batch: ``q_len`` query tokens per request, a request without keys a null request."""
    kv_lens = np.asarray(kv_lens, dtype=np.int64)
    return build_array_plan(
        np.full(kv_lens.size, q_len, dtype=np.int64),
        kv_lens,
        Hq,
        Hkv,
        D,
        window_left,
        split_policy,
        sm_count,
        null_requests=True,
        **kw,
    )


def build_array_plan(
    q_lens: np.ndarray,
    kv_lens: np.ndarray,
    Hq: int,
    Hkv: int,
    D: int,
    window_left: int = -1,
    split_policy: str = "auto",
    sm_count: int = 204,
    q_rows: int = 128,
    merge: str = "auto",
    small_swap: bool = False,
    swap_max_rows: int = INKERNEL_MAX_ROWS,
    grid: int = 0,
    grid_search: bool = True,
    unit_budget: int = 0,
    row_budget: int = -1,
    null_requests: bool = False,
    previous: Optional[ArrayPlan] = None,
) -> ArrayPlan:
    """``build_plan`` computed over arrays: request ``b`` has ``q_lens[b]`` query tokens over ``kv_lens[b]`` keys and its packed
    rows start at ``sum(q_lens[:b])``. The tables and the grid are those of ``build_plan`` with the same arguments, entry for
    entry (the planner a replan before every replay of a captured graph runs: its cost must not grow with a Python loop over
    the units). A request without query tokens gets no unit; under ``null_requests`` neither does one without keys.
    ``previous``: the last plan of the caller; when this batch differs from that one in its KV lengths alone, with every
    Q tile on the same key tiles and chunks -- a decode loop between two tile boundaries --, the tables are that plan's with
    the lengths rewritten (``ArrayPlan.with_kv_lens``), the same tables as planning afresh."""
    assert merge in ("auto", "fixup", "inkernel"), merge
    assert Hq % Hkv == 0, (Hq, Hkv)
    g = Hq // Hkv
    assert 1 <= g <= q_rows, g
    tpt = q_rows // g
    T, C = TILE_KEYS, chunk_len(D)
    tpc = C // T
    q_lens = np.asarray(q_lens, dtype=np.int64)
    kv_lens = np.asarray(kv_lens, dtype=np.int64)
    q_base = np.cumsum(q_lens) - q_lens
    keep = q_lens > 0
    if null_requests:
        keep &= kv_lens > 0
    live = np.flatnonzero(keep)
    short = live[kv_lens[live] < q_lens[live]]
    assert not short.size, (
        f"request {int(short[0]) if short.size else 0}: kv_len < q_len (bottom-right causal needs kv_len >= q_len)"
    )
    # the Q tiles in the planner's order: request, first token, KV head
    tiles_r = -(-q_lens[live] // tpt)
    req = np.repeat(np.arange(live.size), tiles_r)  # per (request, Q tile)
    tok0 = (np.arange(req.size) - (np.cumsum(tiles_r) - tiles_r)[req]) * tpt
    ntok = np.minimum(tpt, q_lens[live][req] - tok0)
    p_first = kv_lens[live][req] - q_lens[live][req] + tok0
    l_max = p_first + ntok
    lo_min = (
        np.maximum(0, p_first - window_left)
        if window_left >= 0
        else np.zeros_like(p_first)
    )
    shape = (
        (Hq, Hkv, D, window_left, split_policy, sm_count, q_rows, merge)
        + (small_swap, swap_max_rows, grid, grid_search, unit_budget, row_budget)
        + (null_requests,),
        q_lens,
        live,
        lo_min // T,
        -(-l_max // T),
    )
    if previous is not None and previous.same_shape(shape):
        return previous.with_kv_lens(kv_lens)
    n_tiles = req.size * Hkv

    def tiles(a: np.ndarray) -> np.ndarray:
        return np.repeat(a, Hkv)

    b_t, tok_t = tiles(live[req]), tiles(tok0)
    h_t = np.tile(np.arange(Hkv, dtype=np.int64), req.size)
    kv_t, qlen_t, qbase_t = (
        tiles(kv_lens[live][req]),
        tiles(q_lens[live][req]),
        tiles(q_base[live][req]),
    )
    rows_t = tiles(ntok) * g
    clo_t, chi_t = tiles(lo_min // C), tiles(-(-l_max // C))
    tlo_t, thi_t = tiles(lo_min // T), tiles(-(-l_max // T))
    n_t = chi_t - clo_t
    len_t = thi_t - tlo_t
    multi = n_t > 1
    # the unit kinds: build_plan's rules
    swap_batch = small_swap and n_tiles > 0 and int(rows_t.max()) <= swap_max_rows
    om_tmem = om_in_tmem(D) or swap_batch
    inkernel_max_chunks = INKERNEL_MAX_CHUNKS if om_tmem else INKERNEL_MAX_CHUNKS_OF[D]
    narrow = rows_t <= INKERNEL_MAX_ROWS
    plan_grid = 0
    if split_policy == "split":
        fused = ~multi
    elif split_policy == "fused" or n_tiles == 0:
        fused = np.ones(n_tiles, dtype=bool)
    elif swap_batch:
        split_lengths = None
        if multi.any():
            first = np.minimum(thi_t, (clo_t + 1) * tpc) - tlo_t
            last = thi_t - np.maximum(tlo_t, (chi_t - 1) * tpc)
            counts = _length_counts(
                np.concatenate([first[multi], last[multi], len_t[~multi]])
            )
            middle = int((n_t[multi] - 2).sum())
            if middle:
                counts[tpc] = counts.get(tpc, 0) + middle
            folds_inkernel = multi & (n_t <= inkernel_max_chunks)
            split_lengths = (
                counts,
                int(rows_t[multi & ~folds_inkernel].sum()),
                int(folds_inkernel.sum()),
            )
        split_all, plan_grid = swaps_decide(
            n_tiles,
            int(len_t.max()),
            (_length_counts(len_t), 0, 0),
            split_lengths,
            sm_count,
            D,
            grid_search,
        )
        if plan_grid == min(n_tiles, sm_count):
            plan_grid = 0  # the default launch grid
        fused = ~multi if split_all else np.ones(n_tiles, dtype=bool)
    else:
        do_split = np.where(
            narrow, n_tiles < 2 * sm_count, WIDE_SPLIT_SM_FRACTION * n_tiles < sm_count
        )
        two_chunk_fused = (
            (not om_tmem and 2 * n_tiles > sm_count)
            & (n_t == 2)
            & (rows_t >= FUSE2_MIN_ROWS)
            & narrow
        )
        fused = ~multi | two_chunk_fused | ~(do_split | ((not om_tmem) & narrow))
    # the fixed capacity: budget_unit_kinds
    want = ~fused
    if want.any() and (unit_budget > 0 or row_budget >= 0):
        lengths, cls = np.unique(len_t[want], return_inverse=True)
        extra = np.bincount(cls, weights=n_t[want] - 1, minlength=lengths.size)
        need = np.bincount(cls, weights=(n_t * rows_t)[want], minlength=lengths.size)
        units = n_tiles + np.cumsum(extra[::-1])  # the classes longest first
        ws = np.cumsum(need[::-1])
        fits = np.ones(lengths.size, dtype=bool)
        if unit_budget > 0:
            fits &= units <= unit_budget
        if row_budget >= 0:
            fits &= ws <= row_budget
        kept = lengths.size if fits.all() else int(np.argmin(fits))
        if kept:
            fused = ~(want & (len_t >= lengths[::-1][kept - 1]))
        else:
            fused = np.ones(n_tiles, dtype=bool)
    # the tables: one unit per fused tile, one per chunk of a split tile and a group for it, in tile order
    split_t = ~fused
    group_tiles = np.flatnonzero(split_t)
    g_rows, g_n = rows_t[group_tiles], n_t[group_tiles]
    g_size = g_n * g_rows
    g_base = np.cumsum(g_size) - g_size
    if merge == "inkernel":
        inkernel = np.ones(group_tiles.size, dtype=bool)
    elif merge == "fixup":
        inkernel = np.zeros(group_tiles.size, dtype=bool)
    else:
        inkernel = (g_rows <= INKERNEL_MAX_ROWS) & (g_n <= inkernel_max_chunks)
    groups = np.empty((group_tiles.size, GROUP_FIELDS), dtype=np.int32)
    groups[:, G_SLOT] = g_base
    groups[:, G_ROWS] = g_rows
    groups[:, G_CHUNK_LO] = clo_t[group_tiles]
    groups[:, G_NCHUNKS] = g_n
    groups[:, G_QTOK0] = tok_t[group_tiles]
    groups[:, G_KVHEAD] = h_t[group_tiles]
    groups[:, G_KVLEN] = kv_t[group_tiles]
    groups[:, G_QLEN] = qlen_t[group_tiles]
    groups[:, G_QBASE] = qbase_t[group_tiles]
    groups[:, G_REQ] = b_t[group_tiles]
    base_t = np.zeros(n_tiles, dtype=np.int64)
    base_t[group_tiles] = g_base
    grp_t = np.full(n_tiles, -1, dtype=np.int64)
    grp_t[group_tiles] = np.where(inkernel, np.arange(group_tiles.size), -1)
    units_t = np.where(fused, 1, n_t)
    tile = np.repeat(np.arange(n_tiles), units_t)
    num_units = tile.size
    split_u = split_t[tile]
    chunk = clo_t[tile] + np.arange(num_units) - (np.cumsum(units_t) - units_t)[tile]
    tile_lo = np.where(split_u, np.maximum(tlo_t[tile], chunk * tpc), tlo_t[tile])
    tile_hi = np.where(split_u, np.minimum(thi_t[tile], (chunk + 1) * tpc), thi_t[tile])
    desc = np.zeros((num_units, DESC_FIELDS), dtype=np.int32)
    desc[:, F_REQ] = b_t[tile]
    desc[:, F_QTOK0] = tok_t[tile]
    desc[:, F_ROWS] = rows_t[tile]
    desc[:, F_KVHEAD] = h_t[tile]
    desc[:, F_CHUNK_LO] = np.where(split_u, chunk, clo_t[tile])
    desc[:, F_CHUNK_HI] = np.where(split_u, chunk + 1, chi_t[tile])
    desc[:, F_MODE] = split_u
    desc[:, F_SLOT] = np.where(
        split_u, base_t[tile] + (chunk - clo_t[tile]) * rows_t[tile], 0
    )
    desc[:, F_KVLEN] = kv_t[tile]
    desc[:, F_QLEN] = qlen_t[tile]
    desc[:, F_TILE_LO] = tile_lo
    desc[:, F_TILE_HI] = tile_hi
    desc[:, F_QBASE] = qbase_t[tile]
    desc[:, F_GROUP] = np.where(split_u, grp_t[tile], -1)
    # longest first (stable), then the waves of the launch grid in alternating direction: build_plan's order
    if num_units:
        order = np.argsort(tile_lo - tile_hi, kind="stable")
        waves = min(num_units, grid if grid > 0 else (plan_grid or sm_count))
        pos = np.arange(num_units)
        start = pos // waves * waves
        back = start + np.minimum(waves, num_units - start) - 1 - (pos - start)
        desc = desc[order[np.where(pos // waves % 2 == 1, back, pos)]]
    plan = ArrayPlan(
        desc, groups, inkernel, int(g_size.sum()), tpt, g, ws_row(D), plan_grid
    )
    plan.shape = shape
    return plan


def _empty_partner(units: np.ndarray) -> np.ndarray:
    """The empty B of a unit alone in its pair: no rows, an empty tile range, no group."""
    empty = units.copy()
    empty[:, F_ROWS] = 0
    empty[:, F_TILE_HI] = empty[:, F_TILE_LO]
    empty[:, F_GROUP] = -1
    return empty


def lone_units_array(desc: np.ndarray) -> np.ndarray:
    """The paired placement's table (rows ``2 u`` and ``2 u + 1`` of pair ``u``) with every unit of ``desc`` alone in its pair,
    in ``desc``'s order: what the paired kernel walks when the batch is not worth pairing."""
    table = np.empty((2 * desc.shape[0], DESC_FIELDS), dtype=np.int32)
    table[0::2] = desc
    table[1::2] = _empty_partner(desc)
    return table


def pair_units_array(desc: np.ndarray, grid: int = 0) -> np.ndarray:
    """``engine.pair_units`` over arrays, row for row: two units pair when they share the request, the KV head, the chunk
    range and the kind -- consecutive Q tiles in token order; an odd one out gets an empty B; the pairs longest first in
    waves of ``grid`` of alternating direction."""
    n = desc.shape[0]
    if n == 0:
        return np.empty((0, DESC_FIELDS), dtype=np.int32)
    keys = [desc[:, f] for f in (F_MODE, F_CHUNK_HI, F_CHUNK_LO, F_KVHEAD, F_REQ)]
    by_key = np.lexsort(keys)
    new = np.zeros(n, dtype=bool)
    new[0] = True
    for key in keys:
        new[1:] |= key[by_key][1:] != key[by_key][:-1]
    # buckets in the order of their first unit, a bucket's units in token order
    first = np.empty(n, dtype=np.int64)
    first[by_key] = np.minimum.reduceat(by_key, np.flatnonzero(new))[np.cumsum(new) - 1]
    bucket = first
    order = np.lexsort((desc[:, F_QTOK0], first))
    sorted_bucket = bucket[order]
    start = np.flatnonzero(np.r_[True, sorted_bucket[1:] != sorted_bucket[:-1]])
    run = np.cumsum(np.r_[True, sorted_bucket[1:] != sorted_bucket[:-1]]) - 1
    is_a = (np.arange(n) - start[run]) % 2 == 0
    a_pos = np.flatnonzero(is_a)
    has_b = (a_pos + 1 < n) & ~is_a[np.minimum(a_pos + 1, n - 1)]
    a = desc[order[a_pos]]
    b = _empty_partner(a)
    b[has_b] = desc[order[a_pos[has_b] + 1]]
    length = np.maximum(
        a[:, F_TILE_HI] - a[:, F_TILE_LO], b[:, F_TILE_HI] - b[:, F_TILE_LO]
    )
    rank = np.argsort(-length, kind="stable")
    pairs = rank.size
    waves = min(pairs, grid)
    if waves > 0:
        pos = np.arange(pairs)
        begin = pos // waves * waves
        back = begin + np.minimum(waves, pairs - begin) - 1 - (pos - begin)
        rank = rank[np.where(pos // waves % 2 == 1, back, pos)]
    table = np.empty((2 * pairs, DESC_FIELDS), dtype=np.int32)
    table[0::2] = a[rank]
    table[1::2] = b[rank]
    return table
