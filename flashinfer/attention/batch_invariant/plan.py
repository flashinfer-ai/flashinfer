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
SWAPS_FUSE_MAX_CHUNKS = 4  # Swaps-body batches (small_swap): units of at most this many chunks stay FUSED when the batch has at least
SWAPS_FUSE_MIN_UNITS = 48  # SWAPS_FUSE_MIN_UNITS of them (the split kind's second
# ramp + fold cost more than 1-3 extra chunks of streaming -- measured fused faster at B = 8 / 64 with short rows, every D;
# at B = 1 (4-8 units) the split's parallelism wins from 2 chunks up and the batch stays below the unit floor)
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
) -> Plan:
    """``split_policy``: ``"auto"`` splits every multi-chunk tile when the fused unit count would leave
    the machine underfilled (``< 2 x SM``) -- and, at head dims whose merged state is not in TMEM (D = 256), every multi-chunk
    tile of at most ``INKERNEL_MAX_ROWS`` rows (decode units) whatever the count: a fused decode unit there merges a 128 x 256 fp32
    state through the global scratch at every chunk end, and the split kind measured faster at long KV lengths
    -- except the TWO-chunk decode-sized units of at least ``FUSE2_MIN_ROWS`` live rows in a batch whose fused unit count exceeds
    half the SM count, which stay fused (one in-kernel merge against two unit ramps and the fixup node); ``"split"`` / ``"fused"``
    force one kind (R3: same bits).
    ``merge``: where a split group's chunk fold runs -- ``"auto"`` (in-kernel for groups of at most
    ``INKERNEL_MAX_ROWS`` rows, the fixup kernel otherwise), ``"fixup"`` (always the kernel), ``"inkernel"``
    (always in-kernel); the same fold in the same order either way (R3: same bits)."""
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
        do_split = True
    elif split_policy == "fused":
        do_split = False
    else:
        do_split = len(tiles) < 2 * sm_count
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
    swaps_fuse = (
        swap_batch and split_policy == "auto" and len(tiles) >= SWAPS_FUSE_MIN_UNITS
    )  # the Swaps body's short-unit rule
    for b, h, tok0, ntok, clo, chi, tlo, thi, kv_len, q_len in tiles:
        rows = ntok * g
        two_chunk_fused = (
            fuse_two_chunk
            and chi - clo == 2
            and FUSE2_MIN_ROWS <= rows <= INKERNEL_MAX_ROWS
        )
        swaps_fused = swaps_fuse and chi - clo <= SWAPS_FUSE_MAX_CHUNKS
        if (
            chi - clo == 1
            or two_chunk_fused
            or swaps_fused
            or not (do_split or (split_small and rows <= INKERNEL_MAX_ROWS))
        ):
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
        tpc = C // T
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
    # longest units first (stable): the persistent CTAs walk units u = bidx, bidx + grid, ..., so a descending order
    # balances their loads (LPT). A placement: the order of units reaches no value (every unit restarts its state)
    plan.desc.sort(key=lambda d: -(d[F_TILE_HI] - d[F_TILE_LO]))
    return plan
