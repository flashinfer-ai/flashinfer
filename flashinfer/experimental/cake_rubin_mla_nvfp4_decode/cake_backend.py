"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Host backend of the Cake dense NVFP4 MLA decode on Rubin (SM107): the host
plan, the caller-owned workspace, the prepared launcher of the decode (and
the split-KV merge behind it) and the BF16 -> NVFP4 query quantizer.

Operands (absorbed multi-head latent attention decode, Kimi-K3 / DeepSeek-V3
geometry: 512 latent + 64 rope channels per key, 512-wide value) over the
NVFP4 paged latent cache of flashinfer-ai/flashinfer#4676:

* ``ckv_cache``    uint8          ``[num_pages, 64, 256]``  packed E2M1 (even element low nibble)
* ``ckv_sf_cache`` float8_e4m3fn  ``[num_pages, 64, 32]``   one block scale per 16 latent channels
* ``kpe_cache``    float8_e4m3fn  ``[num_pages, 64, 64]``   rope channels at the cache scale ``kpe_scale``

with a decoded key ``e2m1 * sf * ckv_scale | fp8 * kpe_scale``; the three may
be strided views of one allocation (the 352-byte-per-token
``[.., 256 | 64 | 32]`` buffer): the last axis contiguous, the token stride,
the page stride (64 token strides) and the base 16-byte multiples.  The page
size is 64.  The query is NVFP4 too (:func:`quantize_mla_nvfp4_query`): per
(token, head) row 256 packed E2M1 bytes ``q_nope``, 32 E4M3 block scales
``q_sf``, 64 E4M3 rope channels ``q_rope`` and one FP32 row scale ``q_scale``
chosen so the rope product lands in the latent logit unit; the logits are
``sm_scale * ckv_scale * q_scale * (QN . KN + QR . KR)``.  Rows are
``token * H + head`` per request, token-major; a dense query is ``[B, q_len,
H, .]``, a packed variable-length one ``[total_q, H, .]`` with its
``cum_seq_lens_q``.  Every request's row count ``q_len * H`` is a multiple of
16 (the epilogue's 16-row bulk stores; ``H`` a multiple of 16 covers every
query length).  ``out`` is BF16 ``q.shape[:-1] + (512,)``; ``lse`` (optional)
FP32 ``q.shape[:-1]`` in natural log.

The host plan (:func:`plan`) is a pure function of host scalars and host
length lists: this rank's KV lengths ``seq_lens_host``, the query lengths,
``num_heads``, the device's ``units`` (clusters of four it holds at once) and
``pairs`` (pair clusters on the SMs the fours leave), and for decode context
parallelism ``cp_world`` / ``cp_rank`` / ``cp_interleave`` with the global
lengths ``kv_len_global_host``.  :func:`mla2_plan` cuts every (request,
256-row tile) into key ranges -- the pair of split counts (full tiles, half
tiles) whose modelled makespan over the clusters and pairs is least -- and
emits the work items the decode walks (``item_table``: request, row tile,
first 128-key tile, tiles, split index, split count; the full-tile items
first, ``n_full`` of them).  ``row_limit[row]`` is how many of the request's
local keys the row sees (bottom-right causal; under DCP the rank's keys whose
global position ``((k // I) W + r) I + k % I`` is within the causal bound).
``units`` / ``pairs`` shape the cost model and the item order only;
correctness never depends on them.

Workspace: one caller-owned uint8 buffer of :func:`workspace_bytes` (exact for
a plan) or :func:`max_workspace_bytes` (a bound) bytes holding, on 256-byte
slots, ``partial_o`` BF16 ``[rows, max_splits, 512]``, ``partial_lse`` FP32
``[rows, max_splits]``, an FP32 ``[rows]`` LSE scratch the decode writes when
the caller asks for no ``lse`` (the kernel always stores a row's LSE) and the
int32 plan block ``item_table | q_indptr | row_limit``, uploaded once when the
runner is prepared (one pinned host-to-device copy).  ``launch()`` allocates
nothing, synchronizes nothing and reads no device memory, so it is
CUDA-graph capturable (a captured graph replays through this workspace: the
runner outlives the graph); ``prepare`` is the host-side step (it reads the
device length tensors when their host lists are not given, a documented
synchronization).  Every launch goes through a ``launcher(record, values,
grid=, device_index=, enable_pdl=)`` (the decode and the merge pass
``enable_pdl``, the quantizer does not), ``cake_jit.run_module`` by default
and injectable for a diagnostic harness.  Repeated calls on the same inputs
give bitwise the same outputs: the kernels reduce in a fixed order.

Programmatic dependent launch is a per-plan switch (:func:`pdl_for`, reported
as ``route_metadata["pdl"]``): when the plan has a split merge and its decode
is one hardware wave (``items <= units + pairs``: the clusters of four plus
the pairs the device holds resident), the decode and the merge launch
with the programmatic attribute, so the decode's prologue overlaps the
quantizer's tail and the merge's the decode's (the quantizer triggers its
dependents once its rows are stored, the decode once a role runs out of
items; each syncs on its dependency before reading the predecessor's output).
A plan with more items than resident clusters, or without a merge, launches
all three plain: an early-launched dependent would take SMs from the decode's
second wave, and a merge under the attribute behind a decode that never
triggers starts later than a plain launch.
"""

from __future__ import annotations

import dataclasses
import functools
import heapq
import math
from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch

from . import cake_jit
from .cake_jit import MAX_SPLITS, STAGES

LATENT = 512
ROPE = 64
QK_DIM = LATENT + ROPE  # 576 BF16 query channels before quantization
V_DIM = LATENT
SF_VEC = 16  # latent channels per E4M3 block scale
CKV_BYTES = LATENT // 2  # 256 packed E2M1 bytes per token / query row
SF_BYTES = LATENT // SF_VEC  # 32 block scales per token / query row
PAGE = 64  # tokens per page: the decode's TMA boxes are one page
ROWS_TILE = 256  # query rows per work-item tile (the cta_group::2 M)
KEY_TILE = 2 * PAGE  # keys per 128-key tile of the decode
CLUSTER = 4  # CTAs per work item in the launch grid
ITEM_WORDS = (
    6  # an item_table row: request, row tile, first key tile, tiles, z, split count
)
ROWS_GRANULE = (
    16  # every request's q_len * H is a multiple of this (the epilogue's 16-row boxes)
)
E2M1_MAX = 6.0
E4M3_MAX = 448.0
LOG2E = math.log2(math.e)
ALIGN = 256  # every workspace slot starts on a 256 B boundary (TMA needs 16 B; 256 B also serves .view)
QUANTIZE_ROWS_PER_CTA = 4  # query quantizer: one warp per (token, head) row
WORKSPACE_SLOTS = ("partial_o", "partial_lse", "lse_scratch", "plan_block")

# The decode cost model's constants (:func:`mla2_plan`).
MLA2_ITEM_COST = 7  # the mixed-cluster kernel's ramp and epilogue per item in 128-key tile times (fitted on sweeps)
# A half tile's time per 128-key tile in a cluster's, when one request holds one full and one half tile (fitted
# on forced plans: the class's half items pace the wave; 0.6 elsewhere, where the full items do).
MLA2_HALF_ONE_FULL_HALF = 0.75
MLA2_SHAVE_TILES = 48  # a shaved one-wave plan's short range per full tile, in 128-key tiles (:func:`mla2_plan`)

# Clusters of four the GPU holds at once and pair clusters on the SMs the fours leave, by SM count, for devices
# whose occupancy cannot be queried (:func:`device_units_pairs`): R200 boards measured 50 fours and 4 pairs at 208 SMs
# and 46 fours and 14 pairs at 212 SMs.
UNITS_PAIRS_BY_SM_COUNT: dict[int, tuple[int, int]] = {208: (50, 4), 212: (46, 14)}

# A launcher takes a record name, the logical argument values, the plan grid, the device index and, for the decode
# and the merge, the per-call PDL switch ``enable_pdl`` (0 or 1), and runs the program (``cake_jit.run_module``).  The
# runners hold one so a diagnostic harness can run the same composition against another build of the same kernels;
# the package itself always launches through ``cake_jit``.
Launcher = Callable[..., None]


# ----------------------------------------------------------------------------------------------------------------
# Host plan (pure functions of the host-known scalars and length lists)


def mla2_plan(
    kv_lens,
    q_lens,
    heads,
    units,
    pairs=0,
    item_cost=MLA2_ITEM_COST,
    merge_rows_per_tile=3900,
    slack=0.03,
    pair_cost=1.4,
    split=0,
    shave=MLA2_SHAVE_TILES,
):
    """The work items of the mixed-cluster decode (128-key tiles). A (request, row tile) whose
    rows past the first 128 are padding (H96 x 4 = 384 rows: the second tile) is a half tile; its items are double
    items, which a cluster of four runs as two pairs over the two halves of the key range and a pair cluster takes
    one range of, each writing a partial (``2 z + range`` of ``2 split``), at 0.6 of the four-CTA body's time per
    tile (the pair body's 1.4 over half the keys less its padding rows: Q4 B1 KV64K measured 24.5 us with 16 double
    ranges against 26.0 with 20, where 0.7 chose 20). Full tiles are cut into ``s_f`` key ranges, half tiles
    into ``s_h`` equal ones; the items go full tiles first, then the double items, so a cluster changes body once.
    The first wave takes items in index order, the ``units`` clusters of four items ``[0, units)`` and the
    ``pairs // 2`` couples of pair clusters the next ones, as 128-row halves at ``pair_cost`` of a cluster's time:
    so when full items reach the couples, each row tile's ranges come in two lengths, the couples' ``1 / pair_cost``
    of the clusters', spread over the row tiles and placed at the couples' indices (the couples then finish with
    the clusters instead of ``pair_cost`` behind them; a double item costs a pair what it costs a cluster, so
    those stay equal). ``split`` is given (an int for both counts, a pair ``(s_f, s_h)``) or the pair of counts
    whose cost is least: the makespan of the items over the clusters and the couples taking items in order as they
    free, each item its tiles plus ``item_cost`` (ramp and epilogue), plus the partials written and merged (``2 n +
    1`` rows of 1 KB per query row with ``n`` partials, at ``merge_rows_per_tile`` rows per tile-time; the merge
    variant is chosen to load every row in one round, so no second-batch term); the
    plan within ``slack`` of the least cost: for a one-tile request and one request of a full and a half tile the one
    that fills the wave (``units + pairs // 2`` items, 52 on 208 SMs, which the half tile's grid of counts misses),
    then equal lengths, else two, then the smallest counts, and at those counts the couples' scale the model puts at
    least 1 % under equal lengths. One wave of at least ``0.8 units`` full tiles
    of 1024 key tiles or more also prices the shaved plan: every tile in a long range and a short one of ``shave``
    tiles, the long ranges emitted first (the clusters' wave) and the short ones after them, which the clusters the
    wave leaves and the couples take as they free (48 tiles on 50 clusters and 2 couples otherwise leave 16 of 208
    SMs idle for the whole decode); it replaces the choice when the model puts it under that by more than ``slack``.
    Returns the items (page ranges ``(request, row tile, first page, end page, split, split count, flags, 0)``,
    ``flags & 1`` a direct item that writes ``O`` / ``LSE``), the kernel's ``item_table`` rows ``(request, row tile,
    first 128-key tile, tiles, z, split count)``, ``max_splits`` (a row of a full tile has ``s_f`` partials, of a
    half tile ``2 s_h``; the kernel voids the slots a row does not use) and ``n_full``, the count of the full
    tiles' items, which lead the table: the kernel tells a double item by its index."""
    tile = 2 * PAGE
    # A request without keys on this rank is planned as one tile, every key masked (its rows' limits are 0): the
    # ordinary path then writes O = 0 and LSE = -inf, and no item has zero tiles.
    tiles_of = [max(1, -(-kv // tile)) for kv in kv_lens]
    tiles = []  # (request, row tile, half)
    for b, ql in enumerate(q_lens):
        rows = ql * heads
        for m in range(-(-rows // ROWS_TILE)):
            tiles.append((b, m, rows - m * ROWS_TILE <= ROWS_TILE // 2))
    full_tiles = [t for t in tiles if not t[2]]
    half_tiles = [t for t in tiles if t[2]]
    rows_full = sum(
        min(ROWS_TILE, q_lens[b] * heads - m * ROWS_TILE) for b, m, _ in full_tiles
    )
    rows_half = sum(
        min(ROWS_TILE, q_lens[b] * heads - m * ROWS_TILE) for b, m, _ in half_tiles
    )
    k_max = max(tiles_of, default=0)
    couples = pairs // 2 if pairs > 1 else 0
    # A one-wave plan: full tiles only, no more of them than clusters (every item starts at once).
    one_wave = not half_tiles and 0 < len(full_tiles) <= units
    one_tile = one_wave and len(full_tiles) == 1
    # One request of a full tile and a half tile (257-384 rows: H96 x 4 at B1): the candidates below add its
    # wave-filling pair of counts (``units + couples`` items, 52 on 208 SMs) and the choice prefers that pair, as
    # it does a one-tile request's.  Measured on 208 SMs; a batch of two such requests is not in the class.
    one_full_half = len(q_lens) == 1 and len(full_tiles) == 1 and len(half_tiles) == 1

    def makespan(costs):
        """Greedy list scheduling of ``costs`` (cluster time, double flag) in order over the clusters of four and
        the couples, each taking the next item as it frees."""
        free = [(0.0, 0, r) for r in range(units)] + [
            (0.0, 1, r) for r in range(couples)
        ]
        heapq.heapify(free)
        span = 0.0
        for c, dbl in costs:
            t, kind, r = heapq.heappop(free)
            if kind == 0 or dbl:
                t += c
            elif one_tile:
                # A one-tile request's couple ranges are short (at most 20 tiles at 128K): the couple takes pair_cost
                # of a cluster's time over the range and the ramp and epilogue are the item's own (52 items beat 43 by
                # 3.5 % on a one-tile 64K row, which pair_cost on the whole item priced 6 % slower); longer ranges
                # keep the whole-item costing (a 4-tile row at 13 splits measured 0.9 % slower than at 12).
                t += pair_cost * (c - item_cost) + item_cost
            else:
                t += pair_cost * c
            span = max(span, t)
            heapq.heappush(free, (t, kind, r))
        return span

    def full_ranges(s_f, scale):
        """Each full tile's key ranges ``(lo, n, couple)`` in key order and the items' emission order ``(tile,
        range)``: with ``scale != 1`` the couples' ranges are ``scale`` times the others (``1 / pair_cost`` where
        the couples' items are their only ones, ``pair_cost`` where the clusters take a second round and the
        couples' slack carries more), spread over the tiles and emitted at indices ``[units, units + couples)``."""
        n_tiles = len(full_tiles)
        n_p = (
            min(couples, max(0, n_tiles * s_f - units))
            if scale != 1.0 and n_tiles
            else 0
        )
        per_tile = [
            n_p // n_tiles + (1 if t < n_p % n_tiles else 0) for t in range(n_tiles)
        ]
        ranges = []
        for t, (b, m, _) in enumerate(full_tiles):  # noqa: B007  (the planner as written)
            k = tiles_of[b]
            p_t = min(per_tile[t], s_f)
            q_t = s_f - p_t
            if p_t:
                long = k / (q_t + p_t * scale)
                lens = [long] * q_t + [long * scale] * p_t
            else:
                lens = [k / s_f] * s_f
            base = [int(x) for x in lens]
            for i in sorted(range(s_f), key=lambda i: -(lens[i] - base[i]))[
                : k - sum(base)
            ]:
                base[i] += 1
            lo, rs = 0, []
            for i, n in enumerate(base):
                rs.append((lo, n, i >= q_t))
                lo += n
            ranges.append(rs)
        q_list = [
            (t, i) for t in range(n_tiles) for i, r in enumerate(ranges[t]) if not r[2]
        ]
        p_list = [
            (t, i) for t in range(n_tiles) for i, r in enumerate(ranges[t]) if r[2]
        ]
        return ranges, q_list[:units] + p_list + q_list[units:]

    def mixed_ranges(s_t):
        """The first ``units`` full tiles whole (one direct item each, the clusters' first wave) and the tiles past
        them in ``s_t`` equal ranges, which the couples take from the start and the clusters as they free: more
        full tiles than clusters at short key counts otherwise cost a second round of whole tiles (H128 x Q8 x B16 at
        8K keys per rank: 64 tiles over 46 clusters measured 79.6 us whole, where a two-CTA tile takes one wave)."""
        ranges, order = [], []
        for t, (b, m, _) in enumerate(full_tiles):  # noqa: B007  (the planner as written)
            k = tiles_of[b]
            if t < units:
                ranges.append([(0, k, False)])
                continue
            base = [k // s_t + (1 if i < k % s_t else 0) for i in range(s_t)]
            lo, rs = 0, []
            for n in base:
                rs.append((lo, n, False))
                lo += n
            ranges.append(rs)
        order = [(t, 0) for t in range(min(units, len(full_tiles)))]
        order += [
            (t, i) for t in range(units, len(full_tiles)) for i in range(len(ranges[t]))
        ]
        return ranges, order

    def shaved_ranges(x):
        """Each full tile in a long range and a short one of ``x`` tiles, the long ranges emitted first (the
        clusters' wave) and the short ones after them, for the clusters the wave leaves and the couples to take as
        they free."""
        ranges = [
            [(0, tiles_of[b] - x, False), (tiles_of[b] - x, x, False)]
            for b, _m, _h in full_tiles
        ]
        n_tiles = len(full_tiles)
        return ranges, [(t, 0) for t in range(n_tiles)] + [
            (t, 1) for t in range(n_tiles)
        ]

    def cost(s_f, s_h, scale, s_t=0, x=0):
        ranges, order = (
            shaved_ranges(x)
            if x
            else full_ranges(s_f, scale)
            if not s_t
            else mixed_ranges(s_t)
        )
        tps_h = -(-k_max // s_h) if half_tiles else 0
        costs = [(ranges[t][i][1] + item_cost, False) for t, i in order]
        phi_h = MLA2_HALF_ONE_FULL_HALF if one_full_half else 0.6
        costs += [(phi_h * tps_h + item_cost, True)] * (len(half_tiles) * s_h)
        merge = 0.0
        if full_tiles and s_t:
            rows_tail = sum(
                min(ROWS_TILE, q_lens[b] * heads - m * ROWS_TILE)
                for b, m, _ in full_tiles[units:]
            )
            merge += (2 * s_t + 1) * rows_tail / merge_rows_per_tile
        elif full_tiles and s_f > 1:
            merge += (2 * s_f + 1) * rows_full / merge_rows_per_tile
        if half_tiles:
            merge += (4 * s_h + 1) * rows_half / merge_rows_per_tile
        return makespan(costs) + merge

    scales = (1.0, 1.0 / pair_cost, pair_cost)
    s_t = 0  # a mixed plan's tail split (0: every full tile cut alike)
    x_s = 0  # a shaved plan's short range (0: none)
    if isinstance(split, tuple) and split and split[0] == "mixed":
        s_f, s_h, s_t, scale = 1, 1, int(split[1]), 1.0
    elif split:
        s_f, s_h = (split, split) if isinstance(split, int) else tuple(split)
        scale = min(scales, key=lambda sc: (cost(s_f, s_h, sc), sc != 1.0))
    else:
        cap = min(max(k_max, 1), 64)
        cands = set()
        for s_f in range(1, cap + 1):
            if not half_tiles:
                cands.add((s_f, 1))
                continue
            for f in (0.5, 0.6, 0.7, 0.85, 1.0, 1.2):
                s_h = max(1, round(s_f * f))
                # A half tile's items write two partials per split, and the merge takes at most MAX_SPLITS slots
                # per row: the one bound this port adds to the kernel-tests planner (a 1-token request at 64K keys
                # would otherwise plan 86 slots).
                if s_h <= cap and 2 * s_h <= MAX_SPLITS:
                    cands.add((s_f if full_tiles else 1, s_h))
            if one_full_half:
                # The pair of counts that fills the wave (``units + couples`` items: ``fill_cap`` below), which the
                # ``f`` grid misses (31 x 0.7 rounds to 22, 32 x 0.6 to 19); the choice below takes the pair when
                # the model puts it within the slack, at the counts ``MLA2_HALF_ONE_FULL_HALF`` sets (forced
                # 52-item plans on 208 SMs, the package call with the decode one wave):
                # Q4 H96 B1, 208 SMs, package call: 52-item forced plans against the 50-item pin (us, 64K / 128K):
                # 32 / 20 -0.21 / -0.48, 31 / 21 -0.33 / -0.80, 30 / 22 -0.05 / -0.98,
                # 29 / 23 -0.46 / -1.53, 28 / 24 +0.63 / +0.30;
                # 0.75 puts the choice at 29 / 23 on both tables (50 / 4, 49 / 6); the pin's 0.6 reaches 32 / 20.
                s_h = units + couples - s_f
                if 1 <= s_h <= cap and 2 * s_h <= MAX_SPLITS:
                    cands.add((s_f, s_h))
            if s_f <= 2:
                cands |= {(s_f if full_tiles else 1, 1), (s_f if full_tiles else 1, 2)}
        if half_tiles and len(full_tiles) > units + couples:
            # Direct full items beside double items lose to two ranges (Q4 B64 KV64K: 631 against 592 us measured, the
            # model within its slack): a full tile keeps at least two ranges where its items exceed the first wave.
            cands = {c for c in cands if c[0] >= 2}
        costs = {(c, sc): cost(*c, sc) for c in cands for sc in scales}
        best = min(costs.values())
        # Within the slack (the model's accuracy across split counts): a one-tile request, and one request of a full
        # and a half tile, fill the wave first (up to the clusters and couples: 52 items on 208 SMs; two-tile rows at
        # 25 splits measured +0.9 % on one board and -1.0 % on another, and three-tile rows at 17-18 splits 0.86 and
        # 0.70 of 16, so more tiles keep the old order), then equal lengths, then fewer splits; then, at those
        # counts, a couple scale the model puts at least 1 % under equal lengths.
        fill_cap = units + couples

        def wave_fill(k):
            n = k[0][0] + (k[0][1] if half_tiles else 0)
            return n if (one_tile or one_full_half) and n <= fill_cap else 0

        (s_f, s_h), scale = min(
            (k for k, v in costs.items() if v <= (1 + slack) * best),
            key=lambda k: (-wave_fill(k), k[1] != 1.0, k[0][0] + k[0][1], k[0]),
        )
        if scale == 1.0:
            alt = min(
                (sc for sc in scales if sc != 1.0),
                key=lambda sc: costs[((s_f, s_h), sc)],
            )
            if costs[((s_f, s_h), alt)] < 0.99 * costs[((s_f, s_h), 1.0)]:
                scale = alt
        # More full tiles than clusters: a mixed plan (the first wave whole, the tail split) when the model puts it
        # under the uniform choice by more than the slack.
        if len(full_tiles) > units:
            s_h_m = s_h if half_tiles else 1
            mixed = {st: cost(1, s_h_m, 1.0, st) for st in (2, 3, 4, 6, 8) if st <= cap}
            st_best = min(mixed, key=lambda st: (mixed[st], st))
            if mixed[st_best] < costs[((s_f, s_h), scale)] / (1 + slack):
                s_f, s_h, s_t, scale = 1, s_h_m, st_best, 1.0
        # One wave of nearly as many full tiles as clusters at long key counts (1024 key tiles, 128K, and up: the rows
        # the family was measured on): the shaved plan, each tile a long range and a short one of ``shave`` tiles for
        # the clusters the wave leaves and the couples, when the model puts it under the chosen plan by more than the
        # slack (it is not held to the equal-lengths order above).
        if (
            one_wave
            and shave
            and len(full_tiles) >= 0.8 * units
            and min(tiles_of[b] for b, _m, _h in full_tiles) >= max(1024, 2 * shave)
        ):
            shaved = cost(2, 1, 1.0, x=shave)
            if shaved < costs[((s_f, s_h), scale)] / (1 + slack):
                s_f, s_h, scale, x_s = 2, 1, 1.0, shave
    ranges, order = (
        shaved_ranges(x_s)
        if x_s
        else full_ranges(s_f, scale)
        if not s_t
        else mixed_ranges(s_t)
    )
    tps_h = -(-k_max // s_h)
    n_f = (s_t or s_f) if full_tiles else 0
    n_h = 2 * s_h if half_tiles else 0
    items, table = [], []
    for t, i in order:
        b, m, _ = full_tiles[t]
        pages = -(-kv_lens[b] // PAGE)
        lo, n, _ = ranges[t][i]
        s_tile = len(
            ranges[t]
        )  # a mixed plan's first-wave tiles are whole (direct) beside the tail's ranges
        items.append(
            (
                b,
                m,
                2 * lo,
                min(2 * (lo + n), pages),
                i,
                s_tile,
                1 if s_tile == 1 else 0,
                0,
            )
        )
        table.append((b, m, lo, n, i, s_tile))
    for b, m, _ in half_tiles:
        pages = -(-kv_lens[b] // PAGE)
        k_tiles = tiles_of[b]
        for z in range(s_h):
            lo, hi = min(z * tps_h, k_tiles), min(k_tiles, (z + 1) * tps_h)
            mid = lo + (hi - lo) // 2
            items.append((b, m, 2 * lo, min(2 * mid, pages), 2 * z, 2 * s_h, 0, 0))
            items.append((b, m, 2 * mid, min(2 * hi, pages), 2 * z + 1, 2 * s_h, 0, 0))
            table.append((b, m, lo, hi - lo, z, 2 * s_h))
    return items, table, max(n_f, n_h), len(order)


def visible_keys(
    bound: int, kv_len: int, world: int, rank: int, interleave: int
) -> int:
    """The decode's masking rule: how many of rank ``rank``'s ``kv_len`` local keys (held in blocks of
    ``interleave`` at the global positions ``((k // I) W + r) I + k % I``) have a global position at most
    ``bound``.  With ``world == 1`` it is ``clamp(bound + 1, 0, kv_len)``."""
    bnd = max(int(bound), -1)
    blocks_to = (bnd + 1) // interleave - rank - 1
    n_full = blocks_to // world + 1 if blocks_to >= 0 else 0
    g0 = (n_full * world + rank) * interleave
    partial = min(max(bnd - g0 + 1, 0), interleave)
    return min(n_full * interleave + partial, int(kv_len))


def row_limits(
    kv_lens: Sequence[int],
    q_lens: Sequence[int],
    heads: int,
    kv_lens_global: Sequence[int],
    cp_world: int = 1,
    cp_rank: int = 0,
    cp_interleave: int = 1,
) -> list[int]:
    """``row_limit`` of every packed (token, head) row: the local keys ``[0, limit)`` the row sees.  Token ``t`` of
    request ``b`` sees the global positions up to ``kv_lens_global[b] - q_lens[b] + t`` (bottom-right causal
    against the request's last keys); the rank's share of them follows :func:`visible_keys`."""
    limits: list[int] = []
    for b, (kv, ql) in enumerate(zip(kv_lens, q_lens, strict=True)):
        for t in range(ql):
            n = visible_keys(
                t + (kv_lens_global[b] - ql), kv, cp_world, cp_rank, cp_interleave
            )
            limits.extend([n] * heads)
    return limits


def units_pairs_for_sm_count(sm_count: int) -> tuple[int, int]:
    """``(units, pairs)`` of a device by its SM count: the measured table (:data:`UNITS_PAIRS_BY_SM_COUNT`), else
    ``(sm_count // 4, 0)``."""
    sm = int(sm_count)
    if sm < 1:
        raise ValueError(f"sm_count must be at least 1, got {sm_count}")
    return UNITS_PAIRS_BY_SM_COUNT.get(sm, (max(1, sm // 4), 0))


@functools.cache
def device_units_pairs(device_index: int) -> tuple[int, int]:
    """``(units, pairs)`` of one CUDA device: the clusters of four it holds at once (``MAC(4)``, the maximum active
    clusters of size four at one CTA per SM) and the pair clusters on the SMs the fours leave (``MAC(2) - 2
    MAC(4)``), read once per process through the occupancy query when one is available, else
    :func:`units_pairs_for_sm_count`.  The values shape the plan's cost model and item order only."""
    sm = cake_jit.sm_count(device_index)
    try:
        from ...attention.prims_ts._hardware import get_max_active_clusters

        with torch.cuda.device(device_index):
            mac4 = int(get_max_active_clusters(4))
            mac2 = int(get_max_active_clusters(2))
    except Exception:  # noqa: BLE001  (no query on this installation: the table)
        return units_pairs_for_sm_count(sm)
    units, pairs = min(mac4, sm // 4), max(0, mac2 - 2 * mac4)
    if units < 1 or 4 * units + 2 * pairs > sm:
        return units_pairs_for_sm_count(sm)
    return units, pairs


@dataclasses.dataclass(frozen=True)
class Region:
    """One workspace slot: its name, byte offset, byte length, dtype and view shape."""

    name: str
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Plan:
    """The host plan of one decode call: the problem's lengths, the work items, the per-row key limits, the records
    of the decode and the merge, their grids and the workspace layout.

    ``item_table`` is the flat int32 ``[items * 6]`` the decode walks (full-tile items first, ``n_full`` of them),
    ``q_indptr`` the ``[B + 1]`` query prefix sums, ``row_limit`` the ``[rows_total]`` per-row key limits;
    together they are the plan block uploaded once per runner.  ``max_splits`` is the partial-slot count per row
    (1: every item direct, no merge).  ``decode_record`` is ``decode_full_pairpipe`` when the plan scalars lie in
    its registry boxes, else ``decode_full`` when ``n_full == items``, else ``decode``; ``merge_record`` the merge
    variant for the row count and partials per row, ``None`` when ``max_splits == 1``.
    """

    batch: int
    num_heads: int
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    kv_lens_global: tuple[int, ...]
    cp_world: int
    cp_rank: int
    cp_interleave: int
    units: int
    pairs: int
    rows_total: int
    items: int
    n_full: int
    max_splits: int
    item_table: tuple[int, ...]
    q_indptr: tuple[int, ...]
    row_limit: tuple[int, ...]
    decode_record: str
    merge_record: str | None
    grid_decode: tuple[int, int, int]
    grid_merge: tuple[int, int, int] | None

    @property
    def plan_block(self) -> tuple[int, ...]:
        """The int32 words of the plan block: ``item_table | q_indptr | row_limit``."""
        return self.item_table + self.q_indptr + self.row_limit

    @property
    def stages(self) -> tuple[str, ...]:
        """The stages of the complete call this plan launches (``merge`` only with more than one split)."""
        return tuple(s for s in STAGES if s != "merge" or self.merge_record is not None)

    def layout(self) -> tuple[Region, ...]:
        """The workspace slots in order, each on a 256-byte boundary."""
        return _layout(self.rows_total, self.max_splits, len(self.plan_block))

    @property
    def workspace_bytes(self) -> int:
        return workspace_bytes(self)


def pdl_for(plan: Plan) -> bool:
    """Whether the plan's decode and merge launch with programmatic dependent launch.

    On when the plan has a split merge (``max_splits > 1``) and the decode is one hardware wave (``items <= units +
    pairs``: the clusters of four and the pairs the device holds resident at once, 50 + 4 on a 208-SM board): the
    merge's prologue then overlaps the decode's tail and the decode's the quantizer's.  Off otherwise: without a
    merge there is nothing behind the decode to overlap, and a decode of more items than the device holds resident
    would lose SMs of its second wave to the early-launched merge.  Off launches every program plain, the merge
    included (a merge under the attribute behind a decode that never triggers starts later than a plain launch).
    The plans of 51-52 items (204-208 CTAs, still one hardware wave) run 1-4 % faster with the attribute on R200
    (52 items: 24.7 against 25.6 us), and the 53-54-item plans, whose second wave holds one or two items, 0.7-2 %
    faster still (31.5 against 32.1, 33.8 against 34.0 us): the bound stays ``units + pairs``.
    """
    return plan.max_splits > 1 and plan.items <= plan.units + plan.pairs


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _layout(rows: int, max_splits: int, plan_words: int) -> tuple[Region, ...]:
    specs = (
        (
            "partial_o",
            torch.bfloat16,
            (rows, max_splits, V_DIM),
            rows * max_splits * V_DIM * 2,
        ),
        ("partial_lse", torch.float32, (rows, max_splits), rows * max_splits * 4),
        ("lse_scratch", torch.float32, (rows,), rows * 4),
        ("plan_block", torch.int32, (plan_words,), plan_words * 4),
    )
    regions = []
    offset = 0
    for name, dtype, shape, nbytes in specs:
        regions.append(Region(name, offset, nbytes, dtype, shape))
        offset += _round_up(nbytes, ALIGN)
    return tuple(regions)


def _layout_bytes(regions: tuple[Region, ...]) -> int:
    last = regions[-1]
    return _round_up(last.offset + last.nbytes, ALIGN)


def workspace_bytes(plan: Plan) -> int:
    """Bytes of ``workspace_buffer`` ``plan`` needs: its partials, the LSE scratch and the plan block."""
    return _layout_bytes(plan.layout())


def max_items(rows_max: int, batch_max: int) -> int:
    """An upper bound of a plan's work items for at most ``rows_max`` rows in ``batch_max`` requests: every
    (request, row tile) is cut into at most ``MAX_SPLITS`` ranges, and the tiles number at most
    ``rows_max // 256 + batch_max`` (each request's last tile may be partial)."""
    return MAX_SPLITS * (int(rows_max) // ROWS_TILE + int(batch_max) + 1)


def max_workspace_bytes(
    rows_max: int,
    batch_max: int,
    items_max: int | None = None,
    *,
    max_splits: int = MAX_SPLITS,
) -> int:
    """Upper bound of :func:`workspace_bytes` over every plan of at most ``rows_max`` rows, ``batch_max`` requests
    and ``items_max`` work items (:func:`max_items` when omitted) with at most ``max_splits`` partial slots per row
    (the merge's ``MAX_SPLITS`` = 64 by default; a caller that bounds its plans' splits tighter may pass less)."""
    if not 1 <= int(max_splits) <= MAX_SPLITS:
        raise ValueError(f"max_splits must be in [1, {MAX_SPLITS}], got {max_splits}")
    items = max_items(rows_max, batch_max) if items_max is None else int(items_max)
    words = items * ITEM_WORDS + int(batch_max) + 1 + int(rows_max)
    return _layout_bytes(_layout(int(rows_max), int(max_splits), words))


@functools.lru_cache(maxsize=512)
def _plan_cached(
    kv_lens: tuple[int, ...],
    q_lens: tuple[int, ...],
    num_heads: int,
    units: int,
    pairs: int,
    cp_world: int,
    cp_rank: int,
    cp_interleave: int,
    kv_lens_global: tuple[int, ...],
    split: Any,
) -> Plan:
    _items, table, max_splits, n_full = mla2_plan(
        kv_lens, q_lens, num_heads, units, pairs, split=0 if split is None else split
    )
    if not 1 <= max_splits <= MAX_SPLITS:
        raise ValueError(
            f"the plan has {max_splits} partial slots per row; the merge takes at most {MAX_SPLITS} "
            f"(kv_lens={kv_lens}, q_lens={q_lens}, units={units}, pairs={pairs}, split={split})"
        )
    items = len(table)
    rows_total = sum(q_lens) * num_heads
    q_indptr = [0]
    for ql in q_lens:
        q_indptr.append(q_indptr[-1] + ql)
    limits = row_limits(
        kv_lens, q_lens, num_heads, kv_lens_global, cp_world, cp_rank, cp_interleave
    )
    decode_record = cake_jit.select_record(
        "decode", rows=rows_total, n_full=n_full, items=items, splits=max_splits
    )
    merge_record = (
        cake_jit.select_record(
            "merge", rows=rows_total, n_full=n_full, items=items, splits=max_splits
        )
        if max_splits > 1
        else None
    )
    return Plan(
        batch=len(kv_lens),
        num_heads=num_heads,
        q_lens=q_lens,
        kv_lens=kv_lens,
        kv_lens_global=kv_lens_global,
        cp_world=cp_world,
        cp_rank=cp_rank,
        cp_interleave=cp_interleave,
        units=units,
        pairs=pairs,
        rows_total=rows_total,
        items=items,
        n_full=n_full,
        max_splits=max_splits,
        item_table=tuple(int(w) for row in table for w in row),
        q_indptr=tuple(q_indptr),
        row_limit=tuple(limits),
        decode_record=decode_record,
        merge_record=merge_record,
        grid_decode=cake_jit.launch_grid(decode_record, items=items, rows=rows_total),
        grid_merge=(
            cake_jit.launch_grid(merge_record, items=items, rows=rows_total)
            if merge_record is not None
            else None
        ),
    )


def plan(
    *,
    seq_lens_host: Sequence[int],
    num_heads: int,
    q_len: int | None = None,
    cum_seq_lens_q_host: Sequence[int] | None = None,
    units: int,
    pairs: int,
    cp_world: int = 1,
    cp_rank: int = 0,
    cp_interleave: int = 1,
    kv_len_global_host: Sequence[int] | None = None,
    split: int | tuple | None = None,
) -> Plan:
    """The host plan of one call (cached per distinct argument tuple).

    ``seq_lens_host`` are this rank's local KV lengths per request (0 allowed: a request without keys on this
    rank gives ``O = 0``, ``LSE = -inf``), the query lengths one ``q_len`` for every request (a dense ``[B,
    q_len, H, .]`` query) or the prefix sums ``cum_seq_lens_q_host`` (``[B + 1]``, first 0) of a packed one;
    ``units`` / ``pairs`` the device's (:func:`device_units_pairs`); under decode context parallelism
    ``cp_world`` ranks, this one ``cp_rank``, local keys in blocks of ``cp_interleave`` and the requests' global
    key counts ``kv_len_global_host`` (the local lengths when omitted or with ``cp_world == 1``).  ``split``
    forces the planner's split counts (an int for both, a pair ``(s_f, s_h)``); ``None`` lets the cost model
    choose.  Every request's ``q_len * num_heads`` must be a multiple of 16.
    """
    kv_lens = tuple(int(k) for k in seq_lens_host)
    batch = len(kv_lens)
    if batch < 1:
        raise ValueError("seq_lens_host must name at least one request")
    if any(k < 0 for k in kv_lens):
        raise ValueError(f"seq_lens_host must be non-negative, got {kv_lens}")
    heads = int(num_heads)
    if heads < 1:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if (q_len is None) == (cum_seq_lens_q_host is None):
        raise ValueError(
            "give exactly one of q_len (dense query) and cum_seq_lens_q_host (packed query)"
        )
    if q_len is not None:
        if int(q_len) < 1:
            raise ValueError(f"q_len must be positive, got {q_len}")
        q_lens = (int(q_len),) * batch
    else:
        assert cum_seq_lens_q_host is not None
        indptr = [int(v) for v in cum_seq_lens_q_host]
        if (
            len(indptr) != batch + 1
            or indptr[0] != 0
            or any(b > a for a, b in zip(indptr[1:], indptr[:-1], strict=True))
        ):
            raise ValueError(
                f"cum_seq_lens_q_host must be {batch + 1} non-decreasing entries starting at 0, got {indptr}"
            )
        q_lens = tuple(b - a for a, b in zip(indptr[:-1], indptr[1:], strict=True))
        if any(ql < 1 for ql in q_lens):
            raise ValueError(
                f"every request needs at least one query token, got q_lens={q_lens}"
            )
    bad = [b for b, ql in enumerate(q_lens) if (ql * heads) % ROWS_GRANULE]
    if bad:
        raise ValueError(
            f"every request's q_len * num_heads must be a multiple of {ROWS_GRANULE} (the epilogue's 16-row "
            f"stores); requests {bad} have q_lens {[q_lens[b] for b in bad]} at num_heads={heads}"
        )
    world, rank, interleave = int(cp_world), int(cp_rank), int(cp_interleave)
    if world < 1 or not 0 <= rank < world or interleave < 1:
        raise ValueError(
            f"need cp_world >= 1, 0 <= cp_rank < cp_world and cp_interleave >= 1, got {cp_world}, {cp_rank}, "
            f"{cp_interleave}"
        )
    if kv_len_global_host is None:
        kv_global = kv_lens
    else:
        kv_global = tuple(int(k) for k in kv_len_global_host)
        if len(kv_global) != batch or any(g < 0 for g in kv_global):
            raise ValueError(
                f"kv_len_global_host must be {batch} non-negative entries, got {kv_global}"
            )
    if int(units) < 1 or int(pairs) < 0:
        raise ValueError(
            f"units must be at least 1 and pairs at least 0, got {units}, {pairs}"
        )
    if split is not None and not isinstance(split, int):
        split = tuple(split)
    return _plan_cached(
        kv_lens,
        q_lens,
        heads,
        int(units),
        int(pairs),
        world,
        rank,
        interleave,
        kv_global,
        split,
    )


# ----------------------------------------------------------------------------------------------------------------
# Device facts, workspace carve and operand checks


def _device_index(device: torch.device) -> int:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            f"the Cake Rubin NVFP4 MLA decode runs on CUDA devices, not {device}"
        )
    return device.index if device.index is not None else torch.cuda.current_device()


def _carve(workspace: torch.Tensor, plan: Plan) -> dict[str, torch.Tensor]:
    if (
        workspace.device.type != "cuda"
        or workspace.dtype != torch.uint8
        or not workspace.is_contiguous()
    ):
        raise ValueError("workspace_buffer must be a contiguous uint8 CUDA tensor")
    raw = workspace.reshape(-1)
    need = workspace_bytes(plan)
    if raw.numel() < need:
        raise ValueError(
            f"workspace_buffer needs at least {need} bytes for this Cake NVFP4 MLA decode plan "
            f"({plan.rows_total} rows, {plan.max_splits} partial slots, {plan.items} items), got {raw.numel()}"
        )
    if raw.data_ptr() % 16:
        raise ValueError("workspace_buffer must be 16-byte aligned")
    views = {}
    for region in plan.layout():
        views[region.name] = (
            raw[region.offset : region.offset + region.nbytes]
            .view(region.dtype)
            .view(region.shape)
        )
    return views


def _check_cache(name: str, t: torch.Tensor, width: int, dtype: torch.dtype) -> None:
    if t.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {t.dtype}")
    if t.ndim != 3 or int(t.shape[-1]) != width:
        raise ValueError(
            f"{name} must be [num_pages, page_size, {width}], got shape {tuple(t.shape)}"
        )
    page_size = int(t.shape[1])
    if page_size != PAGE:
        raise ValueError(
            f"page_size must be {PAGE} (the Rubin decode's TMA box is one page), got {page_size}"
        )
    if t.stride(-1) != 1 or t.stride(-2) % 16 or t.data_ptr() % 16:
        raise ValueError(
            f"{name}: the last dim must be contiguous and the token stride and base 16-byte aligned"
        )
    if t.stride(0) != page_size * t.stride(1):
        raise ValueError(
            f"{name}: the page stride must be {page_size} token strides (one [pages * {page_size}, {width}] "
            f"view), got strides {tuple(t.stride())}"
        )


def _words_view(u8_2d: torch.Tensor) -> tuple[torch.Tensor, int]:
    """A ``[rows, bytes]`` uint8 view (possibly a column slice of a wider buffer) as the int32 words from its first
    element onward and its row stride in words, for a kernel that reads it with 32-bit loads."""
    off = u8_2d.storage_offset()
    nbytes = u8_2d.untyped_storage().nbytes()
    if off % 4 or u8_2d.stride(0) % 4 or u8_2d.stride(1) != 1:
        raise ValueError(
            "a cache view must start and stride on 4-byte boundaries with a contiguous last axis"
        )
    flat = torch.as_strided(u8_2d, (nbytes // 4 * 4,), (1,), storage_offset=0)
    return flat.view(torch.int32)[off // 4 :], u8_2d.stride(0) // 4


def _host_list(
    name: str, host: Sequence[int] | None, device_tensor: torch.Tensor | None
) -> list[int]:
    """``host`` as ints, else the device tensor's contents (a host synchronization, documented)."""
    if host is not None:
        return [int(v) for v in host]
    if device_tensor is None:
        raise ValueError(f"{name}: give the host list or the device tensor")
    return [int(v) for v in device_tensor.reshape(-1).tolist()]


def _check_int32_vector(
    name: str, t: torch.Tensor, numel: int, device: torch.device
) -> None:
    if t.dtype != torch.int32 or not t.is_contiguous() or int(t.numel()) != numel:
        raise ValueError(
            f"{name} must be a contiguous int32 tensor with {numel} entries"
        )
    if t.device != device:
        raise ValueError(f"{name} must be on {device}, got {t.device}")


# ----------------------------------------------------------------------------------------------------------------
# Query quantizer


def query_scale_constants(ckv_scale: float, kpe_scale: float) -> tuple[float, float]:
    """``(c_nope, c_rope)`` of the quantizer in float32 arithmetic: ``1 / (6 * 448)`` and
    ``kpe_scale * rcp(448 * ckv_scale)``."""
    f32 = torch.float32
    c_nope = torch.tensor(1.0 / (E2M1_MAX * E4M3_MAX), dtype=f32)
    c_rope = torch.tensor(float(kpe_scale), dtype=f32) * torch.reciprocal(
        torch.tensor(E4M3_MAX * float(ckv_scale), dtype=f32)
    )
    return float(c_nope), float(c_rope)


# q_nope + q_sf + q_rope + q_scale per query row
_QUERY_ROW_BYTES = CKV_BYTES + SF_BYTES + ROPE + 4


def query_workspace_bytes(rows: int) -> int:
    """Bytes of the caller-owned uint8 buffer :func:`mla_nvfp4_query_buffers` carves for ``rows`` query rows."""
    return int(rows) * _QUERY_ROW_BYTES


def mla_nvfp4_query_buffers(
    workspace: torch.Tensor, lead: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
    """``(q_nope, q_sf, q_rope, q_scale)`` views of the caller's uint8 CUDA ``workspace`` for query rows of
    leading shape ``lead`` (at least ``query_workspace_bytes(prod(lead))`` bytes; nothing is allocated)."""
    rows = math.prod(lead)
    if (
        workspace.device.type != "cuda"
        or workspace.dtype != torch.uint8
        or not workspace.is_contiguous()
    ):
        raise ValueError("the query workspace must be a contiguous uint8 CUDA tensor")
    raw = workspace.reshape(-1)
    if raw.numel() < query_workspace_bytes(rows):
        raise ValueError(
            f"the query workspace needs at least {query_workspace_bytes(rows)} bytes for {rows} rows, got "
            f"{raw.numel()}"
        )
    off = 0
    views = []
    for width, dtype in (
        (CKV_BYTES, torch.uint8),
        (SF_BYTES, torch.float8_e4m3fn),
        (ROPE, torch.float8_e4m3fn),
    ):
        views.append(raw[off : off + rows * width].view(dtype).view(*lead, width))
        off += rows * width
    views.append(raw[off : off + rows * 4].view(torch.float32).view(lead))
    return tuple(views)


class CakeMlaNvfp4QueryQuantize:
    """Prepared BF16 -> NVFP4 query quantizer: ``launch`` allocates nothing.

    ``query`` is a contiguous BF16 tensor ``[.., 576]`` (512 latent + 64 rope channels per (token, head) row after
    the up-projection / absorption); ``out`` holds the four query operands of :class:`CakeMlaNvfp4PagedDecode`
    (``mla_nvfp4_query_buffers`` carves them from a caller workspace).  Per row ``q_scale = max(amax(nope) /
    (6 * 448), amax(rope) * kpe_scale / (448 * ckv_scale))`` (1 for an all-zero row), per 16-channel block
    ``sf = e4m3(amax_block / (6 q_scale))`` and ``codes = e2m1(x / (sf q_scale))``, rope ``e4m3(x * kpe_scale /
    (q_scale * ckv_scale))``, bit for bit the torch reference quantizer.
    """

    def __init__(
        self,
        *,
        query: torch.Tensor,
        ckv_scale: float,
        kpe_scale: float,
        out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        launcher: Launcher | None = None,
    ):
        if query.device.type != "cuda":
            raise ValueError("query must be a CUDA tensor")
        if (
            query.dtype != torch.bfloat16
            or query.shape[-1] != QK_DIM
            or not query.is_contiguous()
        ):
            raise ValueError(
                f"query must be a contiguous bfloat16 tensor with last dim {QK_DIM}"
            )
        if not (ckv_scale > 0.0 and kpe_scale > 0.0):
            raise ValueError("ckv_scale and kpe_scale must be positive")
        q_nope, q_sf, q_rope, q_scale = out
        rows = query.numel() // QK_DIM
        if rows < 1:
            raise ValueError("query must hold at least one row")
        lead = tuple(query.shape[:-1])
        for name, t, shape, dtype in (
            ("q_nope", q_nope, (*lead, CKV_BYTES), torch.uint8),
            ("q_sf", q_sf, (*lead, SF_BYTES), torch.float8_e4m3fn),
            ("q_rope", q_rope, (*lead, ROPE), torch.float8_e4m3fn),
            ("q_scale", q_scale, lead, torch.float32),
        ):
            if (
                tuple(t.shape) != shape
                or t.dtype != dtype
                or not t.is_contiguous()
                or t.device != query.device
            ):
                raise ValueError(
                    f"{name} must be a contiguous {dtype} tensor of shape {shape} on the query's device"
                )
        self.device_index = _device_index(query.device)
        self.arch = cake_jit.device_arch(self.device_index)
        self.launcher: Launcher = cake_jit.run_module if launcher is None else launcher
        self.rows = int(rows)
        self.out = out
        self.record = cake_jit.select_record(
            "quantize", rows=self.rows, n_full=0, items=0
        )
        self.grid = cake_jit.launch_grid(self.record, items=0, rows=self.rows)
        c_nope, c_rope = query_scale_constants(float(ckv_scale), float(kpe_scale))
        self.values: dict[str, Any] = {
            "q": query.view(-1, QK_DIM),
            "q_nope": q_nope.view(torch.uint32).view(-1, CKV_BYTES // 4),
            "q_sf": q_sf.view(torch.uint8).view(-1, SF_BYTES),
            "q_rope": q_rope.view(torch.uint32).view(-1, ROPE // 4),
            "q_scale": q_scale.view(-1),
            "rows": self.rows,
            "c_nope": c_nope,
            "c_rope": c_rope,
            "kpe_scale": float(kpe_scale),
            "ckv_scale": float(ckv_scale),
        }
        assert tuple(self.values) == cake_jit.QUANTIZE_PARAMETERS

    def load(self) -> None:
        """Load (or JIT-build) the quantizer program; call once before timing."""
        cake_jit.load(self.record, self.arch)

    def launch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize on the current stream; no allocation, no host synchronization."""
        self.launcher(
            self.record, self.values, grid=self.grid, device_index=self.device_index
        )
        return self.out

    __call__ = launch


def quantize_mla_nvfp4_query(
    query: torch.Tensor,
    ckv_scale: float,
    kpe_scale: float,
    *,
    out: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-shot query quantization: ``(q_nope, q_sf, q_rope, q_scale)`` of a BF16 query ``[.., 576]`` into the
    caller-owned ``out`` (``mla_nvfp4_query_buffers`` carves it from a workspace)."""
    if out is None:
        raise ValueError(
            "out is required: the caller owns the query buffers (mla_nvfp4_query_buffers carves them from a "
            "uint8 workspace of query_workspace_bytes(rows) bytes)"
        )
    return CakeMlaNvfp4QueryQuantize(
        query=query, ckv_scale=ckv_scale, kpe_scale=kpe_scale, out=out
    ).launch()


# ----------------------------------------------------------------------------------------------------------------
# Prepared decode launcher


class CakeMlaNvfp4PagedDecode:
    """Prepared launcher: validation, the host plan, the workspace carve and the plan-block upload at
    construction; no allocation, synchronization or device read at ``launch``.

    Query operands are ``[B, q_len, H, .]`` or ``[total_q, H, .]`` (with ``cum_seq_lens_q`` /
    ``cum_seq_lens_q_host``): ``q_nope`` uint8 ``[.., 256]``, ``q_sf`` float8_e4m3fn ``[.., 32]``, ``q_rope``
    float8_e4m3fn ``[.., 64]``, ``q_scale`` float32 ``[..]`` (contiguous).  Cache: ``ckv_cache`` uint8 ``[pages,
    64, 256]``, ``ckv_sf_cache`` float8_e4m3fn ``[pages, 64, 32]``, ``kpe_cache`` float8_e4m3fn ``[pages, 64,
    64]`` (strided views of one allocation are fine: the last axis contiguous, the token stride, the page stride
    (64 token strides) and the base 16-byte multiples); the page size is 64.  ``out`` BF16 ``q.shape[:-1] +
    (512,)``; ``lse`` optional FP32 ``q.shape[:-1]`` (natural log).  ``seq_lens`` int32 ``[B]`` this rank's local
    KV lengths (``seq_lens_host`` the same as a host list: without it the constructor reads the device tensor,
    a synchronization), ``block_tables`` int32 ``[B, max_pages]`` (entry ``[b, 0]`` is read even for a request
    without local keys: keep it a valid page).  Variable-length / MTP queries pack the tokens of request ``b`` at
    rows ``cum_seq_lens_q[b] .. cum_seq_lens_q[b + 1]``, bottom-right causal against the request's last keys.
    DCP: ``cp_world`` / ``cp_rank`` / ``cp_interleave`` with ``kv_len_global`` int32 ``[B]`` (or
    ``kv_len_global_host``): rank ``r`` holds the global positions ``((k // I) W + r) I + k % I``.  ``units`` /
    ``pairs`` override the device's cluster counts; ``split`` forces the planner's split counts.
    ``workspace_buffer`` is a uint8 CUDA buffer of at least ``workspace_bytes(plan)`` bytes
    (``max_workspace_bytes(rows, batch)`` covers every plan of a row and request count); never share one
    workspace between two live runners.  ``pdl`` (also ``route_metadata["pdl"]``) is the plan's programmatic
    dependent launch switch (:func:`pdl_for`): the decode and the merge launch with the attribute when it is set.
    """

    def __init__(
        self,
        *,
        q_nope: torch.Tensor,
        q_sf: torch.Tensor,
        q_rope: torch.Tensor,
        q_scale: torch.Tensor,
        ckv_cache: torch.Tensor,
        ckv_sf_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        out: torch.Tensor,
        workspace_buffer: torch.Tensor,
        sm_scale: float,
        ckv_scale: float,
        o_scale: float = 1.0,
        lse: Optional[torch.Tensor] = None,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        cp_world: int = 1,
        cp_rank: int = 0,
        kv_len_global: Optional[torch.Tensor] = None,
        seq_lens_host: Optional[Sequence[int]] = None,
        kv_len_global_host: Optional[Sequence[int]] = None,
        cum_seq_lens_q_host: Optional[Sequence[int]] = None,
        cp_interleave: int = 1,
        units: Optional[int] = None,
        pairs: Optional[int] = None,
        split: int | tuple | None = None,
        launcher: Launcher | None = None,
    ):
        device = q_nope.device
        if device.type != "cuda":
            raise ValueError("query operands must be CUDA tensors")
        u8, e4m3 = torch.uint8, torch.float8_e4m3fn
        if q_nope.dtype != u8:
            raise ValueError(f"q_nope must be uint8, got {q_nope.dtype}")
        if q_sf.dtype != e4m3 or q_rope.dtype != e4m3:
            raise ValueError("q_sf / q_rope must be float8_e4m3fn")
        if q_scale.dtype != torch.float32:
            raise ValueError("q_scale must be float32")
        if (
            q_nope.shape[-1] != CKV_BYTES
            or q_sf.shape[-1] != SF_BYTES
            or q_rope.shape[-1] != ROPE
        ):
            raise ValueError(
                f"query operand widths must be {CKV_BYTES} / {SF_BYTES} / {ROPE}"
            )
        _check_cache("ckv_cache", ckv_cache, CKV_BYTES, u8)
        _check_cache("ckv_sf_cache", ckv_sf_cache, SF_BYTES, e4m3)
        _check_cache("kpe_cache", kpe_cache, ROPE, e4m3)
        if (
            ckv_sf_cache.shape[:2] != ckv_cache.shape[:2]
            or kpe_cache.shape[:2] != ckv_cache.shape[:2]
        ):
            raise ValueError("cache tensors must share [num_pages, page_size]")
        num_pages = int(ckv_cache.shape[0])
        if (
            block_tables.dtype != torch.int32
            or block_tables.ndim != 2
            or not block_tables.is_contiguous()
        ):
            raise ValueError(
                "block_tables must be a contiguous int32 [batch, max_pages_per_seq] tensor"
            )
        if q_nope.ndim == 4:
            batch, q_len, num_heads, _ = (int(v) for v in q_nope.shape)
            if cum_seq_lens_q is not None or cum_seq_lens_q_host is not None:
                raise ValueError(
                    "a dense [B, q_len, H, .] query takes no cum_seq_lens_q"
                )
            indptr = [q_len * b for b in range(batch + 1)]
        elif q_nope.ndim == 3:
            if cum_seq_lens_q is None and cum_seq_lens_q_host is None:
                raise ValueError(
                    "a packed [total_q, H, .] query needs cum_seq_lens_q (or cum_seq_lens_q_host)"
                )
            indptr = _host_list("cum_seq_lens_q", cum_seq_lens_q_host, cum_seq_lens_q)
            batch = len(indptr) - 1
            num_heads = int(q_nope.shape[1])
            if cum_seq_lens_q is not None:
                _check_int32_vector("cum_seq_lens_q", cum_seq_lens_q, batch + 1, device)
            q_len = None
        else:
            raise ValueError("query operands must be 3D or 4D")
        if batch < 1:
            raise ValueError("the query must hold at least one request")
        if int(block_tables.shape[0]) != batch:
            raise ValueError(
                f"block_tables must have {batch} rows, got {tuple(block_tables.shape)}"
            )
        lead = tuple(q_nope.shape[:-1])
        for name, t, shape in (
            ("q_sf", q_sf, (*lead, SF_BYTES)),
            ("q_rope", q_rope, (*lead, ROPE)),
            ("q_scale", q_scale, lead),
            ("out", out, (*lead, V_DIM)),
        ):
            if tuple(t.shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {tuple(t.shape)}"
                )
        for name, t in (
            ("q_nope", q_nope),
            ("q_sf", q_sf),
            ("q_rope", q_rope),
            ("q_scale", q_scale),
            ("out", out),
        ):
            if not t.is_contiguous():
                raise ValueError(
                    f"{name} must be contiguous (a strided query view is refused, never copied)"
                )
        if out.dtype != torch.bfloat16:
            raise ValueError("out must be bfloat16")
        _check_int32_vector("seq_lens", seq_lens, batch, device)
        if lse is not None and (
            lse.dtype != torch.float32
            or tuple(lse.shape) != lead
            or not lse.is_contiguous()
        ):
            raise ValueError(
                "lse must be a contiguous float32 tensor of shape query.shape[:-1]"
            )
        if (
            type(cp_world) is not int
            or type(cp_rank) is not int
            or cp_world < 1
            or not 0 <= cp_rank < cp_world
        ):
            raise ValueError("need int cp_world >= 1 and 0 <= cp_rank < cp_world")
        if type(cp_interleave) is not int or cp_interleave < 1:
            raise ValueError("need int cp_interleave >= 1")
        if cp_world > 1:
            if kv_len_global is None and kv_len_global_host is None:
                raise ValueError(
                    "DCP (cp_world > 1) needs kv_len_global (or kv_len_global_host)"
                )
            if kv_len_global is not None:
                _check_int32_vector("kv_len_global", kv_len_global, batch, device)
            kv_global_host: list[int] | None = _host_list(
                "kv_len_global", kv_len_global_host, kv_len_global
            )
        elif kv_len_global is not None or kv_len_global_host is not None:
            raise ValueError("kv_len_global requires cp_world > 1")
        else:
            kv_global_host = None
        if not (sm_scale > 0.0 and ckv_scale > 0.0):
            raise ValueError("sm_scale and ckv_scale must be positive")
        tensors = [q_nope, q_sf, q_rope, q_scale, ckv_cache, ckv_sf_cache, kpe_cache, block_tables, seq_lens, out,
                   workspace_buffer]  # fmt: skip
        tensors += [t for t in (cum_seq_lens_q, lse, kv_len_global) if t is not None]
        if not all(t.device == device for t in tensors):
            raise ValueError("Expected all tensors on one CUDA device")
        self.device_index = _device_index(device)
        self.arch = cake_jit.device_arch(self.device_index)
        self.launcher: Launcher = cake_jit.run_module if launcher is None else launcher
        kv_host = _host_list("seq_lens", seq_lens_host, seq_lens)
        if len(kv_host) != batch:
            raise ValueError(
                f"seq_lens_host must have {batch} entries, got {len(kv_host)}"
            )
        max_pages = int(block_tables.shape[1])
        if max_pages < 1:
            raise ValueError("block_tables must have at least one page column")
        if any(k > max_pages * PAGE for k in kv_host):
            raise ValueError(
                f"a request's seq_len exceeds block_tables' {max_pages} pages x {PAGE} tokens: seq_lens={kv_host}"
            )
        if max_seq_len is not None and max(kv_host) > int(max_seq_len):
            raise ValueError(
                f"max_seq_len={max_seq_len} is below the longest seq_len {max(kv_host)}"
            )
        if units is None or pairs is None:
            dev_units, dev_pairs = device_units_pairs(self.device_index)
            units = dev_units if units is None else units
            pairs = dev_pairs if pairs is None else pairs
        self.plan = plan(
            seq_lens_host=kv_host,
            num_heads=num_heads,
            q_len=q_len,
            cum_seq_lens_q_host=indptr if q_len is None else None,
            units=int(units),
            pairs=int(pairs),
            cp_world=cp_world,
            cp_rank=cp_rank,
            cp_interleave=cp_interleave,
            kv_len_global_host=kv_global_host,
            split=split,
        )
        p = self.plan
        rows = q_nope.numel() // CKV_BYTES
        if rows != p.rows_total:
            raise ValueError(
                f"query holds {rows} rows, the plan {p.rows_total} (cum_seq_lens_q[-1] = {p.q_indptr[-1]} tokens "
                f"x {num_heads} heads)"
            )
        if max_q_len is not None and max(p.q_lens) > int(max_q_len):
            raise ValueError(
                f"max_q_len={max_q_len} is below the longest query {max(p.q_lens)}"
            )
        self.batch = batch
        self.num_heads = num_heads
        self.max_q_len = max(p.q_lens)
        self.max_pages = max_pages
        self.num_pages = num_pages
        self.rows_total = p.rows_total
        self.max_splits = p.max_splits
        self.pdl = pdl_for(p)
        self.logit_scale = float(sm_scale) * LOG2E * float(ckv_scale)
        self.o_factor = float(o_scale) * float(ckv_scale)
        self.out = out
        self.lse = lse
        self.workspace_bytes = workspace_bytes(p)
        views = _carve(workspace_buffer, p)
        self.partial_o = views["partial_o"]
        self.partial_lse = views["partial_lse"]
        self._plan_block = views["plan_block"]
        lse_rows = lse.view(-1) if lse is not None else views["lse_scratch"]
        # The plan block, uploaded once: one pinned host-to-device copy on the current stream.  The pinned source
        # stays referenced until the runner is dropped (the copy is asynchronous).
        self._plan_block_host = torch.tensor(
            p.plan_block, dtype=torch.int32
        ).pin_memory()
        self._plan_block.copy_(self._plan_block_host, non_blocking=True)
        words = len(p.item_table)
        item_table = self._plan_block[:words].view(p.items, ITEM_WORDS)
        q_indptr = self._plan_block[words : words + batch + 1]
        row_limit = self._plan_block[words + batch + 1 :]
        ckv_2d = ckv_cache.view(num_pages * PAGE, CKV_BYTES)
        cks_2d = ckv_sf_cache.view(u8).view(num_pages * PAGE, SF_BYTES)
        kpe_2d = kpe_cache.view(u8).view(num_pages * PAGE, ROPE)
        ckv_words, ckv_stride_words = _words_view(ckv_2d)
        cks_words, cks_stride_words = _words_view(cks_2d)
        o_rows = out.view(-1, V_DIM)
        self.decode_values: dict[str, Any] = {
            "QN": q_nope.view(-1, CKV_BYTES),
            "QS": q_sf.view(u8).view(-1, SF_BYTES),
            "QR": q_rope.view(u8).view(-1, ROPE),
            "CKV": ckv_2d,
            "CKS": cks_2d,
            "KPE": kpe_2d,
            "ckv_words": ckv_words,
            "cks_words": cks_words,
            "O": o_rows,
            "LSE": lse_rows,
            "partial_o": self.partial_o,
            "partial_lse": self.partial_lse,
            "q_scale": q_scale.view(-1),
            "item_table": item_table,
            "page_table": block_tables,
            "seq_lens": seq_lens,
            "q_indptr": q_indptr,
            "row_limit": row_limit,
            "num_heads": num_heads,
            "max_pages": max_pages,
            "items": p.items,
            "max_splits": p.max_splits,
            "ckv_stride_words": ckv_stride_words,
            "cks_stride_words": cks_stride_words,
            "logit_scale": self.logit_scale,
            "o_factor": self.o_factor,
            "O_T": o_rows,
            "PO_T": self.partial_o,
            "O_T16": o_rows,
            "PO_T16": self.partial_o,
            "n_full": p.n_full,
        }
        assert tuple(self.decode_values) == cake_jit.DECODE_PARAMETERS
        self.merge_values: dict[str, Any] | None = None
        if p.merge_record is not None:
            self.merge_values = {
                "partial_o": self.partial_o,
                "partial_lse": self.partial_lse,
                "O": o_rows,
                "lse": lse_rows,
                "rows": p.rows_total,
                "num_split": p.max_splits,
                "has_lse": int(lse is not None),
            }
            assert tuple(self.merge_values) == cake_jit.MERGE_PARAMETERS
        self.route_metadata = dict(
            backend="cake",
            arch=self.arch,
            decode_record=p.decode_record,
            merge_record=p.merge_record,
            pdl=self.pdl,
            items=p.items,
            n_full=p.n_full,
            max_splits=p.max_splits,
            units=p.units,
            pairs=p.pairs,
            grid_decode=p.grid_decode,
            grid_merge=p.grid_merge,
        )

    def records(self) -> dict[str, str]:
        """Stage -> the record this plan launches."""
        p = self.plan
        out = {"decode": p.decode_record}
        if p.merge_record is not None:
            out["merge"] = p.merge_record
        return out

    def load(self) -> None:
        """Load (or JIT-build) every program this plan launches; call once before timing."""
        for name in self.records().values():
            cake_jit.load(name, self.arch)

    def launch(self) -> torch.Tensor:
        """Enqueue the decode (and the split merge) on the current stream, both with programmatic dependent
        launch when the plan's switch is set (``self.pdl``); no allocation, no synchronization."""
        p = self.plan
        enable_pdl = int(self.pdl)
        self.launcher(
            p.decode_record,
            self.decode_values,
            grid=p.grid_decode,
            device_index=self.device_index,
            enable_pdl=enable_pdl,
        )
        if self.merge_values is not None:
            self.launcher(
                p.merge_record,
                self.merge_values,
                grid=p.grid_merge,
                device_index=self.device_index,
                enable_pdl=enable_pdl,
            )
        return self.out

    __call__ = launch


def prepare_cake_mla_nvfp4_paged_decode(
    q_nope: torch.Tensor,
    q_sf: torch.Tensor,
    q_rope: torch.Tensor,
    q_scale: torch.Tensor,
    ckv_cache: torch.Tensor,
    ckv_sf_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    sm_scale: float,
    ckv_scale: float,
    o_scale: float = 1.0,
    lse: Optional[torch.Tensor] = None,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    cp_world: int = 1,
    cp_rank: int = 0,
    kv_len_global: Optional[torch.Tensor] = None,
    backend: str = "cake",
    seq_lens_host: Optional[Sequence[int]] = None,
    kv_len_global_host: Optional[Sequence[int]] = None,
    cum_seq_lens_q_host: Optional[Sequence[int]] = None,
    cp_interleave: int = 1,
    units: Optional[int] = None,
    pairs: Optional[int] = None,
    split: int | tuple | None = None,
    launcher: Launcher | None = None,
) -> CakeMlaNvfp4PagedDecode:
    """Plan and bind one decode problem: the returned runner's ``launch()`` allocates nothing and synchronizes
    nothing (the graph-safe path); ``.plan`` and ``.workspace_bytes`` describe the plan.  See
    :class:`CakeMlaNvfp4PagedDecode`."""
    if backend != "cake":
        raise ValueError("Cake NVFP4 MLA decode supports backend='cake'")
    return CakeMlaNvfp4PagedDecode(
        q_nope=q_nope,
        q_sf=q_sf,
        q_rope=q_rope,
        q_scale=q_scale,
        ckv_cache=ckv_cache,
        ckv_sf_cache=ckv_sf_cache,
        kpe_cache=kpe_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        out=out,
        workspace_buffer=workspace_buffer,
        sm_scale=sm_scale,
        ckv_scale=ckv_scale,
        o_scale=o_scale,
        lse=lse,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        max_seq_len=max_seq_len,
        cp_world=cp_world,
        cp_rank=cp_rank,
        kv_len_global=kv_len_global,
        seq_lens_host=seq_lens_host,
        kv_len_global_host=kv_len_global_host,
        cum_seq_lens_q_host=cum_seq_lens_q_host,
        cp_interleave=cp_interleave,
        units=units,
        pairs=pairs,
        split=split,
        launcher=launcher,
    )


def cake_mla_nvfp4_paged_decode(
    q_nope: torch.Tensor,
    q_sf: torch.Tensor,
    q_rope: torch.Tensor,
    q_scale: torch.Tensor,
    ckv_cache: torch.Tensor,
    ckv_sf_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    sm_scale: float,
    ckv_scale: float,
    o_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    cp_world: int = 1,
    cp_rank: int = 0,
    kv_len_global: Optional[torch.Tensor] = None,
    backend: str = "cake",
    seq_lens_host: Optional[Sequence[int]] = None,
    kv_len_global_host: Optional[Sequence[int]] = None,
    cum_seq_lens_q_host: Optional[Sequence[int]] = None,
    cp_interleave: int = 1,
    units: Optional[int] = None,
    pairs: Optional[int] = None,
):
    """One-shot dense NVFP4 MLA decode (prepare + one launch) into the caller-owned ``out`` (and ``lse`` with
    ``return_lse=True``).  Without ``seq_lens_host`` (and ``cum_seq_lens_q_host`` / ``kv_len_global_host`` where
    those tensors are given) the call reads the device length tensors on the host, a synchronization; the
    prepared runner of :func:`prepare_cake_mla_nvfp4_paged_decode` is the graph-safe path.  Returns ``out`` or
    ``(out, lse)``.
    """
    if backend != "cake":
        raise ValueError("Cake NVFP4 MLA decode supports backend='cake'")
    if out is None or (return_lse and lse is None):
        raise ValueError(
            "out (and lse with return_lse=True) are required: the caller owns the result buffers"
        )
    runner = CakeMlaNvfp4PagedDecode(
        q_nope=q_nope,
        q_sf=q_sf,
        q_rope=q_rope,
        q_scale=q_scale,
        ckv_cache=ckv_cache,
        ckv_sf_cache=ckv_sf_cache,
        kpe_cache=kpe_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        out=out,
        workspace_buffer=workspace_buffer,
        sm_scale=sm_scale,
        ckv_scale=ckv_scale,
        o_scale=o_scale,
        lse=lse,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        max_seq_len=max_seq_len,
        cp_world=cp_world,
        cp_rank=cp_rank,
        kv_len_global=kv_len_global,
        seq_lens_host=seq_lens_host,
        kv_len_global_host=kv_len_global_host,
        cum_seq_lens_q_host=cum_seq_lens_q_host,
        cp_interleave=cp_interleave,
        units=units,
        pairs=pairs,
    )
    runner.launch()
    return (out, lse) if return_lse else out


def is_supported(device: torch.device | str | int | None = None) -> bool:
    """Whether ``device`` (the current CUDA device when ``None``) is an exported target of this package."""
    if not torch.cuda.is_available():
        return False
    if device is None:
        index = torch.cuda.current_device()
    elif isinstance(device, int):
        index = device
    else:
        d = torch.device(device)
        if d.type != "cuda":
            return False
        index = d.index if d.index is not None else torch.cuda.current_device()
    return (
        cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(index)))
        is not None
    )


__all__ = [
    "ALIGN",
    "CKV_BYTES",
    "CLUSTER",
    "CakeMlaNvfp4PagedDecode",
    "CakeMlaNvfp4QueryQuantize",
    "ITEM_WORDS",
    "KEY_TILE",
    "LATENT",
    "LOG2E",
    "Launcher",
    "MAX_SPLITS",
    "MLA2_ITEM_COST",
    "PAGE",
    "Plan",
    "QK_DIM",
    "QUANTIZE_ROWS_PER_CTA",
    "ROPE",
    "ROWS_GRANULE",
    "ROWS_TILE",
    "Region",
    "SF_BYTES",
    "SF_VEC",
    "UNITS_PAIRS_BY_SM_COUNT",
    "V_DIM",
    "WORKSPACE_SLOTS",
    "cake_mla_nvfp4_paged_decode",
    "device_units_pairs",
    "is_supported",
    "max_items",
    "max_workspace_bytes",
    "mla2_plan",
    "mla_nvfp4_query_buffers",
    "pdl_for",
    "plan",
    "prepare_cake_mla_nvfp4_paged_decode",
    "quantize_mla_nvfp4_query",
    "query_scale_constants",
    "query_workspace_bytes",
    "row_limits",
    "units_pairs_for_sm_count",
    "visible_keys",
    "workspace_bytes",
]
