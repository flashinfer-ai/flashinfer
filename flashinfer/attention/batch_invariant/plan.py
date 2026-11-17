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
from typing import List, Optional, Sequence

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


def swaps_kind_cost(
    units: Sequence[tuple],
    split: Sequence[bool],
    grid: int,
    sm_count: int,
    D: int,
    inkernel_max_chunks: int,
) -> float:
    """The predicted makespan of a Swaps-body batch whose unit ``i`` -- ``units[i] = (tiles, chunk_tiles, rows)`` with ``chunk_tiles``
    the tiles of each of its chunks -- is split into one unit per chunk when ``split[i]``, walked by ``grid`` persistent CTAs. The
    walk takes the units longest first in waves of ``grid``; a wave costs its longest unit's tiles at the rate of its active CTAs (the
    HBM rate from ``STREAM_SAT_FRACTION`` of the SMs up; below it the per-CTA rate plus the unit boundary's bubble), the first wave
    adds its opening burst's contention, the split groups add the fixup kernel's cost or the in-kernel fold's per wave of folding
    units."""
    scale = D / 128
    lengths: List[int] = []
    fixup_rows = 0
    inkernel_groups = 0
    for (tiles, chunk_tiles, rows), s in zip(units, split, strict=True):
        if s and len(chunk_tiles) > 1:
            lengths.extend(chunk_tiles)
            if len(chunk_tiles) > inkernel_max_chunks:
                fixup_rows += rows
            else:
                inkernel_groups += 1
        else:
            lengths.append(tiles)
    if not lengths:
        return 0.0
    lengths.sort(reverse=True)
    grid = max(1, min(grid, len(lengths)))
    sat = STREAM_SAT_FRACTION[D] * sm_count
    cost = 0.0
    for w in range(0, len(lengths), grid):
        wave = lengths[w : w + grid]
        cost += max(
            wave[0] * scale * sat / sm_count + UNIT_BUBBLE[D],
            wave[0] * scale * len(wave) / sm_count,
        )
    cost += FIRST_WAVE_RAMP * scale * grid / sm_count
    if fixup_rows:
        cost += FIXUP_BASE_COST + FIXUP_ROW_COST * fixup_rows
    if inkernel_groups:
        cost += INKERNEL_FOLD_COST * -(-inkernel_groups // grid)
    return cost


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
    units: Sequence[tuple], sm_count: int, D: int, inkernel_max_chunks: int
) -> tuple:
    """The unit kinds and the launch grid of a Swaps-body batch under ``"auto"`` (``units`` as in ``swaps_kind_cost``): ``(split, grid)``
    with ``split[i]`` whether unit ``i`` splits into its chunks. The candidates are every unit fused on each grid of
    ``swaps_fused_grids`` and every multi-chunk unit split on the full grid; the cheapest wins (among equal fused grids the one of
    most waves: the fewest CTAs that still stream at the HBM rate). A placement: every choice computes the same bits."""
    n = len(units)
    if n == 0:
        return [], 0
    fused = [False] * n
    best_grid, best_cost = 0, float("inf")
    for g in swaps_fused_grids(n, sm_count, D, max(u[0] for u in units)):
        c = swaps_kind_cost(units, fused, g, sm_count, D, inkernel_max_chunks)
        if c <= best_cost:
            best_grid, best_cost = g, c
    split = [len(u[1]) > 1 for u in units]
    if any(split):
        c = swaps_kind_cost(units, split, sm_count, sm_count, D, inkernel_max_chunks)
        if c < best_cost:
            return split, 0
    return fused, best_grid


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
    so that the unit order's waves (``snake_order``) are the waves the CTAs actually walk."""
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
        if q_len == 0:
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
                (
                    thi - tlo,
                    [
                        min(thi, (j + 1) * tpc) - max(tlo, j * tpc)
                        for j in range(clo, chi)
                    ],
                    ntok * g,
                )
                for _, _, _, ntok, clo, chi, tlo, thi, _, _ in tiles
            ],
            sm_count,
            D,
            inkernel_max_chunks,
        )
        if plan.grid == min(len(tiles), sm_count):
            plan.grid = 0  # the default launch grid
    for i, (b, h, tok0, ntok, clo, chi, tlo, thi, kv_len, q_len) in enumerate(tiles):
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
