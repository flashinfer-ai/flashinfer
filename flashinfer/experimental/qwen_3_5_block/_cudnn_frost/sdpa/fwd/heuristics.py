# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDPA-forward family's proposals: which cells, with which knob sets.

:func:`recommend` is the family's ENTIRE heuristic surface — the PURE core:
``(kind, facts, offered) -> [PlanConfig]``. Backend-blind, graph-blind,
import-light. For every offered cell whose capability row admits the facts,
the cell's rule emits an ORDERED list of COMPLETE knob assignments (every
axis the row declares a domain for carries a concrete value; ``None`` only on
undeclared axes), each re-validated through ``mismatch(caps, facts, knobs)``
— a set is honored or never listed. The same engine appears once per
surviving set. Standalone callers (wrappers, autotuners) invoke this directly
with a hand-built :class:`~cudnn.sdpa.graph_analyzer.SdpaGraphFacts`; nothing
here touches the backend, a graph object, or heuristic modes.

``kind`` is ``"A"`` (candidates worth timing, best guess first, runners-up
behind for a caller that autotunes) or ``"FALLBACK"`` (the config expected to
build where A's choice may not — nothing chosen for speed).

Everything else about the final list — mode blocks, the backend's entries,
the delegating entry, dedup, the mode strip — is PLACEMENT and lives once in
``engines/heuristics._assemble``. The one placement opinion this family holds
is WHERE the backend's block goes, and :func:`propose` (the manifest hook)
states it per measured shard through ``placement.place``: ours first where
FROST is timed ahead of the backend, the backend's block first where it is
not. With ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES`` set the caller asked for
FROST and ours lead everywhere.

Cross-ENGINE order within a proposal batch is ``ENGINE_SPECS`` declaration
order. Today that is unambiguous in practice — co-eligible cells are the
envelope-overlap family, which all lower to the same kernel — and the seam
for a real ranking, when one is measured, is a score stage here in
:func:`recommend`, not a new layer.

To add a rule for a cell: write a generator (:func:`_sm120_tiles` is the
worked example), register the cell in ``_TILE_RULE_CELLS`` (or grow a new
axis via the five-part checklist in ``engines.py``), put the measurement in
the commit. A cell with no rule runs its row's sole point per axis, which is
the honest answer while nobody has timed it.
"""

from __future__ import annotations

from dataclasses import replace
from math import gcd
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.engines.base import PlanConfig
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.constants import (
    DTYPE_BF16,
    DTYPE_E4M3,
    DTYPE_E5M2,
    DTYPE_FP16,
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import (
    d256_mxfp8_pair_leg,
    TemplateParams as Sm100TemplateParams,
    cga_tile_m,
    cga_ctas,
    d192_square_br_as_tl,
    d256_square_br_as_tl,
    decode_d256_q_tile,
    decode_d256_q_units,
    pack_gqa_group_size,
    pack_gqa_supported,
    supports_paged_d256_pack_gqa,
    supports_paged_prefill_cga1,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm120 import (
    D512_FLAVOR,
    FP8_HEAD_TILE_GRANULE,
    HEAD_TILE_GRANULE,
    SMEM_CAPACITY_BYTES,
    head_tiles,
    kv_tile_preference,
    pick_flavor,
    smem_bytes,
    tile_domain,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.engines import (
    ENGINE_SPECS,
    Capabilities,
    EngineSpec,
    SdpaFwdKnobs,
    paged_thd_split_domain,
    thd_split_domain,
    _selected_d_shape,
    _synth_kv_padding,
    _thd_decode_leg,
    d256_decode_tile_selected,
    decode_d256_q_tile_for_row,
    effective_cgas,
    effective_sched_policies,
    launch_grid_mismatch,
    mismatch,
    pack_gqa_partial,
    rubin_dense_d128_shared_leg,
)

# Cells whose (tile_m, tile_n) choice _sm120_tiles makes.
_TILE_RULE_CELLS = frozenset({"sdpa_fwd_prefill_sm120", "sdpa_fwd_prefill_sm120_fp8", "sdpa_fwd_prefill_sm120_mxfp8"})

# Cap on complete knob sets emitted per engine per kind. The combiner grows
# Σ|axis runners|, never the cartesian product; this bound keeps the plan list
# legible and an autotune-ALL pass affordable even as axes accumulate.
_MAX_SETS_PER_ENGINE = 6

# The causal-balancing budget for the CLC/static LPT_L2 policy on SM100/SM120:
# LPT_L2's block-cyclic head grouping only pays when ONE head's K+V working set
# can actually stay L2-resident.
_SM100_L2_BUDGET_BYTES = 50 * 1024 * 1024
# Below this per-head K+V footprint the SM100 d128 / d64 f16 and per-tensor FP8
# causal walks take plain LPT (see _sched_points): 8 MiB sits between the measured
# S=8K bf16 (4 MiB, LPT ahead) and S=32K bf16 (16 MiB, LPT_L2 ahead) points.
_SM100_D128_LPT_L2_MIN_BYTES = 8 * 1024 * 1024
# Rubin, causal, NO GQA (h_q == h_kv): the LPT-vs-NATURAL crossover in grid WAVES
# (work items per persistent 2-CTA cluster).  Perf node, kernel-level, d192x128
# FP8 H128 causal, NATURAL = 1.00: LPT 1.00 / 1.16 / 0.93 / 0.87 / 0.92 and
# LPT_L2 0.90 / - / 1.00 / - / 0.99 at S = 2K / 4K / 8K / 16K / 32K, i.e. LPT
# pays at 10-19 waves and costs from 39 waves on, LPT_L2 never pays there.
# 256 = the 2-CTA flavors' q rows per cluster; measured on the 212-SM part (106 clusters).
_SM107_CGA_Q_ROWS = 256
_SM107_MEASURED_SMS = 212
_SM107_NO_GQA_LPT_MAX_WAVES = 24
# cc 10.7 half, the shared d128 DECODE tile (sm100/decode_d128_f16.py compiled for sm_107a, issue #1472) carrying a
# packed GQA group: plain LPT leads NATURAL from a 2k cache on.  MEASURED on a 216-SM cc 10.7 board (cuDNN 9.26.0.51
# and 9.27.0.28, CUDA-graph replay, kernel time, every plan of the unified list timed round-robin; the dense sink decode /
# verify tables, bottom-right causal, no window): LPT / NATURAL median 0.989 over 224 (cell, policy) pairs at KV 2k-16k
# (LPT more than 3 % ahead in 58, NATURAL in 1), 0.944 at 32k (8 pairs; b1 / b4 64/8 q1 / q8: 153 -> 135 us), parity
# below 2k (26 pairs, median 1.000, 4 past 3 % each way; b8 64/8 q1 KV 1k 9.3 -> 13.1 us) and NATURAL under a sliding
# window (GPT-OSS d64 SWA128: 1.05-1.16).  Per-cell exceptions exist both ways (b8 32/4 q8 KV 2k: LPT 1.32x slower;
# b32 32/8 q8 KV 8k f16: 0.80x), so the other policy stays listed as the runner.  The unpacked (MHA) decode tile
# measured parity (b128 64/8 unpacked 0.997-1.025) and keeps the Rubin no-GQA wave rule.
_SM107_DECODE_TILE_LPT_MIN_KV = 2048
# cc 10.7 D128 nonpaged THD split: the shortest cache it is proposed for (the SM100 line keeps 2048; 1024-token caches
# lost up to 1.29x warm), and below 2048 the cache must hold this many query rows' worth (Q256 over 1-2k caches lost up
# to 1.39x to the backend).
_SM107_THD_SPLIT_MIN_KV = 1280
_SM107_THD_SPLIT_SHORT_KV_RATIO = 12
# Dense d128 half past one decode tile: one cga2 wave (512-row clusters on SMs / 2 slots) costs this many single-CTA
# decode-tile waves (128-row CTAs on every SM); the narrower tile leads while its grid quantizes into no more than that.
# Fit on cc 10.7 (216 SMs) and B300 (148 SMs) bottom-right causal prefill / chunk cells (issue #1518).
_D128_CGA2_WAVE_COST = 1.6
# ... below this cache: past it the cga2 pair's shared K/V reads win (B200 / cc 10.7 chunks at 32k-128k KV ran 0.86-0.97
# on the decode tile without it; with it the 68 changed first plans run 0.98x or faster).
_D128_CGA1_MAX_KV = 16384

# The SM80 kernels' L2 grouping budget is a per-flavor MiB table fed to the
# template (sched_l2_mib); the adapter owns that table. For POINT ORDERING all
# that matters here is that SM80's measured primary for causal is LPT_L2.


# ---------------------------------------------------------------------------
# axis generators — each returns an ORDERED candidate list, best first
# ---------------------------------------------------------------------------


def _sole(values):
    """The only value on an axis, or None where the row declares no domain."""
    return next(iter(values)) if len(values) == 1 else None


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# --- KV split (see choose_split_kv) ----------------------------------------
# A split thinner than this is prologue/epilogue dominated.
_SPLIT_KV_MIN_TILES = 2
# What a CTA-tile costs beyond its KV loop (Q load, prologue, epilogue), in
# units of one KV tile. Empirical: re-measure if the per-tile fixed cost moves.
_SPLIT_KV_CTA_COST = 21.0
# What ONE split's partials cost the combine pass, per wave of combine blocks,
# in units of one KV tile of main-kernel work. The combine's own occupancy
# (blocks/SM of sm100/split_combine) is ABSORBED into this coefficient: it is
# one fixed kernel, so blocks/SM is a constant, and folding it in keeps a
# cuOccupancy query -- which would need a compiled CUfunction -- off the
# planning path. Empirical: re-measure if sm100/split_combine changes.
#
# Re-measured for the fp32 partials the SM100 split kernels now write (was 0.2,
# fitted when partials were half). Widening them turned out to cost the combine
# almost nothing -- 1.05x on a 148-row x 512-split sweep, because the pass is
# not purely bandwidth-bound -- so the move is NOT a consequence of the extra
# bytes. It corrects a coefficient that was too large for the shapes this model
# is asked about: the split kernel also stopped staging O through SMEM and TMA,
# which made splitting cheaper on the main-kernel side.
#
# Two independent fits agree. Timing the combine directly against one KV tile of
# main-kernel work, with the partials large enough to live in HBM rather than
# L2, gives 0.037 (f16) / 0.039 (f32). Minimising regret over the end-to-end
# sweep in test_split_kv_heuristic._B300_FIT gives an optimum PLATEAU of
# [0.04, 0.155] -- every value in it makes the same 12 choices. 0.1 is that
# plateau's midpoint, so it is the value furthest from flipping either way.
_SPLIT_KV_COMBINE_COST = 0.1
# What ONE partial costs a combine block that has nothing to hide behind, in
# units of one KV tile of main-kernel work. sm100/split_combine runs one block
# per output row and walks the split axis three times per block with a
# dependent global load per step (``unroll=1``), so a block is a latency chain
# ~3 s loads long. The per-WAVE coefficient above absorbed the blocks/SM that
# overlap such chains -- which presumes enough rows to fill them. A launch with
# FEWER rows than the machine holds at once (decode: S_q * H_q * B in the tens
# or hundreds, at most one block per SM) overlaps nothing, and its one round
# costs what a lone block costs however the wave count reads. Measured on B200
# (d128 bf16 paged, kernel time along a split ladder whose grid stays inside
# one wave, so only the loop and combine terms move): 0.60 us per partial
# against 1.69 us per KV tile at b=1 h=64/4 S_q=1 S_kv=32k (splits 8/16/32:
# 69.5/47.2/43.2 us) and 0.64 against 1.86 us at b=1 h=16/2 (splits 16/32/64:
# 44.5/39.8/52.7 us) -- 0.35 and 0.34 KV tiles. The floor and the per-wave
# price meet at 3.5 combine waves (~520 rows on 148 SMs); the fitted sweep
# (_B300_FIT) starts at 2048 rows, so the floor moves none of its choices.
_SPLIT_KV_COMBINE_FLOOR = 0.35
# SM120 (RTX PRO 6000, 188 SMs): a lone CTA walks a KV tile in ~3.7 us at d128, so thinner splits keep
# paying and a partial costs ~0.05 tile. Fitted on a 161-case decode split ladder (2026-10-08): mean
# regret 5.3% -> 0.5%, worst 43% -> 21%; any floor in [0.02, 0.1] makes the same choices.
_SM120_SPLIT_KV_MIN_TILES = 1
_SM120_SPLIT_KV_COMBINE_FLOOR = 0.1


class _SplitKvLaunch(NamedTuple):
    q_tiles: int
    heads_q: int
    kv_tiles: int
    ctas_per_tile: int


def split_kv_candidates(*, sm_count: int, kv_tiles: int, min_tiles: Optional[int] = None) -> List[int]:
    """The splits worth scoring on this device, ascending, always starting at 1.

    THE single split-KV list -- what a row can BUILD is a separate boolean
    (``Capabilities.split_kv_supported``), so this is free to be device-derived
    rather than a hand-maintained per-row literal.

    Powers of two from 1 up to ``2**ceil(log2(sm_count))``: you never need more
    CTA-tiles than the machine has SMs, so that is where the occupancy argument
    for splitting runs out. Rounding UP rather than down offers the first
    over-subscribing point and lets the cost model reject it on the wave term,
    instead of the bound pre-judging it. Powers of two because ``split_kv`` is a
    TemplateParams field and so a kernel-module cache key -- an unrestricted
    choice mints a compiled specialization per shape.

    Also bounded by ``kv_tiles // _SPLIT_KV_MIN_TILES``. The chunking hands the
    remainder to the LEADING splits, so the thinnest gets ``floor(kv_tiles/s)``
    tiles, and ``floor(kv_tiles/s) >= m`` is exactly ``s <= floor(kv_tiles/m)``.
    On a short KV that bound binds first.

    No workspace bound here: the partial slabs grow with s, but so does the
    combine term in :func:`choose_split_kv`, and it grows with the Q rows --
    which is what makes the slabs big in the first place. The model self-limits;
    a caller needing a hard ceiling has ``deselect_workspace_greater_than``.
    """
    if sm_count <= 0 or kv_tiles <= 0:
        return [1]
    hi = 1 << max(0, (sm_count - 1).bit_length())  # 2**ceil(log2(sm_count))
    hi = min(hi, max(1, kv_tiles // (min_tiles or _SPLIT_KV_MIN_TILES)))
    out, s = [], 1
    while s <= hi:
        out.append(s)
        s <<= 1
    return out


def choose_split_kv(
    *,
    q_tiles: int,
    heads_q: int,
    batch: int,
    kv_tiles: int,
    sm_count: int,
    combine_rows: int,
    ctas_per_tile: int = 1,
    candidates: Optional[List[int]] = None,
    unsplit_launch: Optional[_SplitKvLaunch] = None,
    min_tiles: Optional[int] = None,
    combine_floor: Optional[float] = None,
) -> int:
    """How many KV chunks to cut each Q tile into; 1 = do not split.

    A prefill launch is ``q_tiles * heads_q * batch`` independent tiles, each
    walking the whole KV loop.  When that product is below the SM count the chip
    idles however long the loop is; splitting multiplies the tile count by ``s``
    and divides each tile's KV work by it, then pays one reduction over the
    partials.

    Splitting runs TWO kernels, so the model is two LATENCIES summed -- each one
    (sequential rounds) x (what one round costs).  Both terms must be latency:
    mixing in an aggregate-work term would double-count the parallelism the wave
    factor has already divided out.

        waves(s)      = ceil(base_ctas * s / sm_count)      # main grid
        combine_waves = ceil(combine_rows / sm_count)       # combine grid, NO s
        per_partial   = max(combine_waves * COMBINE_COST, COMBINE_FLOOR)
        cost(1)       = waves(1)      * (kv_tiles + CTA_COST)        # one kernel
        cost(s > 1)   = waves(s)      * (ceil(kv_tiles / s) + CTA_COST)
                      + s * per_partial

    CTA_COST is what a tile re-pays whatever its loop length, so it sits INSIDE
    the wave term -- once per CTA-tile, not once per split.  COMBINE_COST is
    outside it: the combine is a separate launch whose grid is ``(S_q, H, B)``
    (sm100/split_combine), one block per output row and independent of ``s`` --
    only the per-block work grows with ``s``, since each block reduces ``s``
    partials.  Hence ``combine_rows`` (= S_q * H_q * B) and not ``base_ctas``.
    COMBINE_FLOOR is the least a partial can cost: what one block pays for one
    step of its serial split walk when there are too few rows for the blocks
    to hide each other's latency (_SPLIT_KV_COMBINE_FLOOR).  Without it a
    few-unit launch -- b=1, a handful of KV heads, a long KV -- read the
    combine as nearly free and split until the wave was full: b=1 h=16/2
    S_q=1 S_kv=32k at cga1 (2 units, 256 KV tiles, 16 rows) took 64 splits at
    50.8 us where 32 measure 39.7 us (B200); b=1 h=64/4 S_q=1 S_kv=4k took 16
    at 22.3 us where 8 measure 20.3 us.  The floor prices those extra partials
    at what they cost and leaves every launch with 3.5 or more combine waves
    exactly as it was.
    The unsplit leg runs the classic single-pass kernel and NO combine, so it
    carries no combine term at all.  It used to be charged one (``s = 1`` in
    the formula above), which under-priced the combine a split adds by exactly
    one combine wave-set -- invisible on the fitted shapes (512 KV tiles, where
    that is < 3% of the loop term), decisive on a 4k KV with many output rows:
    b=8 h=32/8 S_q=64 S_kv=4096 paged, 16384 combine rows, split 2 modelled
    8% cheaper than unsplit and measured 11% slower (B200, 75.0 vs 67.6 us);
    dense b=4 h=32/8 S_q=128 S_kv=4096 causal, split 2 measured 155.5 us
    against 120.2 us unsplit.  Where the combine is one wave (decode: b=8
    h=64/4 S_q=1, 512 rows) the term is 0.1 tile and nothing moves: cga1 with
    SPLIT_KV=4 stays at 27.5 us against 63.6 us unsplit.

    Why the combine term matters: ``s`` reaches the first term ONLY through
    ``waves(s)``, a step function.  Between wave boundaries a larger split is
    free there while the loop term keeps falling, so without a second term the
    model always takes the largest split that fits the current wave.  That is
    harmless while the candidate list stops at 4 and a runaway once it does not.

    What falls out: an under-full launch splits until the wave is full; an
    over-full one with a partial-wave tail splits FINER to smooth it, even past
    the SM count; an exactly balanced one (base_ctas = k * sm_count) has no tail
    and never splits; and a long-S_q chunk splits less than a short one, because
    its combine has more rows to reduce.

    ``q_tiles`` / ``heads_q`` / ``kv_tiles`` / ``ctas_per_tile`` describe the
    split leg. ``unsplit_launch`` may describe a different complete no-split
    assignment, as happens when D192 uses CGA1 unsplit but requires CGA2 for a
    split. It defaults to the split geometry, preserving the ordinary case.

    Returns 1 when there is nothing to split or nothing beats not splitting.
    ``candidates`` defaults to :func:`split_kv_candidates` for the device.
    """
    split_launch = _SplitKvLaunch(q_tiles, heads_q, kv_tiles, ctas_per_tile)
    if unsplit_launch is None:
        unsplit_launch = split_launch
    if min(*split_launch, *unsplit_launch, batch, sm_count) <= 0:
        return 1
    if kv_tiles <= 1:
        return 1
    if candidates is None:
        candidates = split_kv_candidates(sm_count=sm_count, kv_tiles=kv_tiles, min_tiles=min_tiles)
    # The combine reads every partial of every output row, so its grid is sized
    # by the rows; max(1, ...) because a decode-shaped launch has fewer rows
    # than SMs and still pays one wave.
    combine_waves = max(1, _ceil_div(max(0, combine_rows), sm_count))
    # ... and that one wave costs no less than a lone block's serial walk of
    # its partials: with fewer rows than the machine holds at once there is
    # nothing to hide the chain behind (_SPLIT_KV_COMBINE_FLOOR).
    per_partial = max(combine_waves * _SPLIT_KV_COMBINE_COST, _SPLIT_KV_COMBINE_FLOOR if combine_floor is None else combine_floor)

    best_split, best_cost = 1, None
    for split in candidates:
        if split < 1 or split > kv_tiles:
            continue
        # Every split must stay thick enough to amortise its own prologue and
        # epilogue. The chunking hands the remainder to the leading splits, so
        # the THINNEST gets floor(kv_tiles / split) -- that is what must clear.
        if split > 1 and kv_tiles // split < (min_tiles or _SPLIT_KV_MIN_TILES):
            continue
        launch = unsplit_launch if split == 1 else split_launch
        base_ctas = launch.q_tiles * launch.heads_q * batch * launch.ctas_per_tile
        waves = _ceil_div(base_ctas * split, sm_count)
        # No combine runs unsplit: the single-pass kernel writes O itself.
        combine = split * per_partial if split > 1 else 0.0
        cost = waves * (_ceil_div(launch.kv_tiles, split) + _SPLIT_KV_CTA_COST) + combine
        if best_cost is None or cost < best_cost:
            best_split, best_cost = split, cost
    return best_split


# --- KV split on the d256 decode tile (see choose_decode_tile_split_kv) ------
# sm100/decode_d256_f16.py is not the machine choose_split_kv was fitted on: one
# cta_group::1 CTA per (KV-head group, batch, split) unit streams 128-key K/V
# tiles (128 KiB each at d256 half) with about one tile of fixed cost, and the
# kernel is HBM-bound rather than MMA-bound. Its costs, in units of one KV tile
# streamed by a lone CTA (1.75 us on B200):
#
# The share of the device's SMs whose concurrent K/V streams saturate HBM. Past
# it every CTA slows in proportion to the stream count (the bandwidth is
# shared), which for an HBM-bound kernel is the same statement as "waves".
# Fitted on B200 (148 SMs): 64 CTAs x 32 tiles ran 56.9 us (1.78 us/tile,
# per-CTA-bound), 128 CTAs x 16 tiles 45 us (1.57x per tile), 256 CTAs x 32
# tiles 184 us (3.2x per tile) -- all consistent with ~81 CTAs.
_DECODE_TILE_SATURATION_FRACTION = 0.55
# What a CTA costs beyond its KV loop (Q^T load, prologue, a 16/32-row epilogue).
_DECODE_TILE_CTA_COST = 1.0
# A KV tile's cost to a lone CTA on the 32-column tile (S_q x G in (16, 32]:
# two softmax column groups over the same 128 TMEM lanes, issue-bound where
# the 16-column tile is bandwidth-bound): 64 CTAs x 32 tiles ran 90.2 us
# against the 16-column tile's 57.1 us (B200). HBM saturates at the same byte
# rate, so a slower stream also needs proportionally more CTAs to reach it.
# That tile is compiled but not routed on the Blackwell line (the adapter keeps
# S_q x G in (16, 32] on the prefill tile there,
# config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS); the Rubin line routes it, in up to
# two TOKEN UNITS for a packed MTP step (config_sm107.decode_d256_q_tile), and
# this B200 fit is the cost its units are modelled with until a cc 10.7 refit.
_DECODE_TILE_WIDE_Q_TILE_COST = 1.6
# The shared sm100/split_combine pass over a decode-shaped grid: one combine
# wave, ~6 us measured at b=32 x 32 heads.
_DECODE_TILE_COMBINE_COST = 3.5
# What EVERY partial adds to that pass, in KV-tile units.  A decode-shaped grid
# is one wave of warps with nothing to hide a serial split walk behind, so the
# combine grows linearly with the split count: measured on the Rubin d256 tile
# (212 SMs, 2376 MHz lock; batch 1 x 2 KV heads, pack 12:1, kv 4K / 32K / 128K
# and batch 4 x 128K; splits 4 .. 64 pinned) 3.0-3.4 us + 0.17-0.18 us per
# split with the grouped-load walk of sm100/split_combine (0.63-0.67 before it,
# when the pass read one split per memory round trip), against 1.03 us per KV
# tile for a lone CTA there.  Without this term the model took split 64 at
# batch 1 wherever the KV was long (the per-wave term is flat in s) and paid a
# 40 us combine for a 12 us kernel; with it the batch-1 optima measured at
# kv 4K / 32K / 128K (16 / 32 / 64) and the batch-4 x 128K one (16) all fall
# out, and every B200 pin below holds (the term is 0.3-2.7 tiles there).
_DECODE_TILE_COMBINE_PER_PARTIAL = 0.17
# What the two-launch split path costs an EAGER caller on the host per
# graph.execute beyond the one-launch path: a second CuTe-DSL launch plus the
# partial-slab carving. Measured 62 -> 92 us on the b=32 x 2 KV heads x 4096
# keys serving shape (Python launch path, B200 host), against a 6 us GPU
# saving. A CUDA-graph replay pays none of it, but a plan cannot know whether
# it will be captured unless the caller says so (pygraph(
# is_cuda_graph_replay_expected=True) -> facts.cuda_graph_replay), so the
# default CHARGES it -- a split leads only where its GPU saving also covers
# the eager caller's extra host time -- and the captured caller's optimum
# (this term at 0) is listed as the runner-up plan; a caller that declared
# the replay gets the two in the other order.
_DECODE_TILE_SPLIT_LAUNCH_COST = 17.0


def _decode_tile_cost(split: int, *, units: int, kv_tiles: int, sm_count: int, tile_cost: float, launch_cost: float) -> float:
    streams = units * split
    per_tile = max(tile_cost, streams / (_DECODE_TILE_SATURATION_FRACTION * sm_count))
    cost = (_ceil_div(kv_tiles, split) + _DECODE_TILE_CTA_COST) * per_tile
    if split > 1:
        cost += _DECODE_TILE_COMBINE_COST + split * _DECODE_TILE_COMBINE_PER_PARTIAL + launch_cost
    return cost


def choose_decode_tile_split_kv(
    *,
    units: int,
    kv_tiles: int,
    sm_count: int,
    q_tile: int = 16,
    launch_cost: float = _DECODE_TILE_SPLIT_LAUNCH_COST,
) -> int:
    """How many KV chunks the d256 decode tile cuts each unit into; 1 = do not split.

    ``units`` is the launch's CTA count before splitting -- batch x KV-head
    groups (a packed group is one unit, an unpacked head one each) x the token
    units a group is cut into (:func:`config_sm100.decode_d256_q_units`: one,
    or two for the Rubin MTP step); ``kv_tiles`` the 128-key tiles of the
    declared S_kv; ``q_tile`` the tile's N extent (16 or 32 packed Q rows,
    :func:`config_sm100.decode_d256_q_tile`).

        streams(s)  = units * s                                   # concurrent K/V streams
        per_tile(s) = max(TILE_COST(q_tile), streams(s) / (SATURATION * sm_count))
        cost(s)     = (ceil(kv_tiles / s) + CTA_COST) * per_tile(s)
                    + [s > 1] * (COMBINE_COST + s * COMBINE_PER_PARTIAL + launch_cost)

    A lone CTA streams a KV tile at TILE_COST (1 on the 16-column tile, 1.6 on
    the issue-bound 32-column one); once the streams saturate HBM every CTA
    slows with their count instead.  What falls out: an under-full launch
    splits until its streams saturate HBM; a launch already past it never
    splits (the bytes are the same, only the fixed costs grow); in between,
    the split must SAVE more than it ADDS -- and every partial it adds costs
    the one-wave combine a serial step (COMBINE_PER_PARTIAL), which is what
    stops a few-unit launch from splitting past the point where the combine
    outweighs the kernel.  With the default ``launch_cost``
    the added part includes the eager caller's second host launch, so on the
    16-column tile the b=32 x 2 KV heads x 4096-key serving shape (6 us of GPU
    saving for 30 us of host time) stays unsplit while a small batch or a
    long KV (b=8; s_kv=16384 at b=32) still splits, and the 32-column tile --
    whose lone CTA is slow enough that the same shape saves ~32 us -- would
    split (routed on cc 10.7 only; see _DECODE_TILE_WIDE_Q_TILE_COST).
    ``launch_cost=0`` is the optimum of a caller replaying a captured CUDA
    graph (the LEADING entry of a graph created with
    ``is_cuda_graph_replay_expected=True``, see :func:`_split_points`).

    Candidates come from :func:`split_kv_candidates`; ties go to the smaller
    split. Returns 1 for degenerate inputs.
    """
    if min(units, kv_tiles, sm_count) <= 0:
        return 1
    tile_cost = _DECODE_TILE_WIDE_Q_TILE_COST if q_tile > 16 else 1.0
    kw = dict(units=units, kv_tiles=kv_tiles, sm_count=sm_count, tile_cost=tile_cost, launch_cost=launch_cost)
    best, best_cost = 1, _decode_tile_cost(1, **kw)
    for split in split_kv_candidates(sm_count=sm_count, kv_tiles=kv_tiles):
        if split <= 1:
            continue
        cost = _decode_tile_cost(split, **kw)
        if cost < best_cost:
            best, best_cost = split, cost
    return best


def _sm120_envelope(facts) -> Tuple[int, int]:
    """(D_QK, D_V) rounded up to the head-tile granule: the ENVELOPE tiles the
    SM120 kernels stage, so a fit check must use them or a tile offered here
    declines at build (and a knob-carried tile skips the adapter's fallback)."""
    return head_tiles(facts.d_qk, facts.d_v, FP8_HEAD_TILE_GRANULE if (facts.is_fp8 or facts.is_mxfp8) else HEAD_TILE_GRANULE)


def _sm120_fits_smem(facts, tile_m: int, tile_n: int) -> bool:
    """Whether a (tile_m, tile_n) CTA fits SM120 SMEM. FP8 stages a byte per KV
    element but still writes O in half, so the two SMEM terms size differently
    (config_sm120.smem_bytes); MXFP8 adds its K/V scale tiles."""
    quantized = facts.is_fp8 or facts.is_mxfp8
    d_qp, d_vp = _sm120_envelope(facts)
    return smem_bytes(d_qp, d_vp, tile_m, tile_n, 1 if quantized else 2, 2, mxfp8=facts.is_mxfp8) <= SMEM_CAPACITY_BYTES


def _sm120_tiles(caps: Capabilities, facts) -> Tuple[int, int]:
    """(tile_m, tile_n) for the SM120 SDPA-forward prefill cell.

    ``tile_m=64`` when the grid cannot fill the machine AND each CTA has enough
    KV tiles to amortize the extra Q-tile loop; a causal mask counts as a
    halved grid because it halves the work per CTA. ``tile_n`` is the first of
    ``config_sm120.kv_tile_preference`` that fits SMEM: 128 is fastest, but a
    wide head has no room for it (D>=208 in half, further out in FP8 -- its KV
    tile is a byte per element).

    Fit is a KERNEL property, so ``config_sm120.smem_bytes`` is the one
    implementation and the adapter's check calls it too. So is the tuning: an
    earlier revision of this template staged P through SMEM and wanted
    ``tile_n=64``. Re-measure when a kernel changes -- the sweeps are in PR #528
    (f16: regret 1.009 geomean, 1.054 worst) and PR #509 (fp8: 1.0046 geomean,
    1.039 worst over 30 seeded cells).

    Read those worst cases with care. Most cells here are within the ~1%
    run-to-run floor of each other, so a single sweep's worst cell is often
    noise: an unseeded run of the SAME code reported 1.155 at B1xH16xS2048
    causal, which the seeded repeat shows as a tie. What survives repetition is
    that the misses cluster on CAUSAL shapes, where the triangular mask shifts
    the per-CTA balance in a way `grid` alone does not capture.
    """
    sm_count = facts.device_sm_count or 0
    grid = -(-facts.s_q // 128) * facts.h_q * facts.b
    if facts.causal:
        grid //= 2
    kv_tiles = -(-facts.s_kv // 128)
    fine = sm_count > 0 and (grid * 2 <= sm_count or (grid * 2 <= 3 * sm_count and kv_tiles >= 12))
    preferred_tile_m = 64 if fine else 128
    kv_order = kv_tile_preference(facts.d_qk, mxfp8=facts.is_mxfp8)
    # Prefer the selected query tile, then larger query tiles and faster KV tiles.
    candidate_tiles = sorted(
        tile_domain(facts.d_qk, facts.d_v, facts.is_fp8 or facts.is_mxfp8),
        key=lambda tile: (tile[0] == preferred_tile_m, tile[0], -kv_order.index(tile[1])),
        reverse=True,
    )
    for m, n in candidate_tiles:
        if m not in caps.tile_ms or n not in caps.tile_ns:
            continue
        if _sm120_fits_smem(facts, m, n):
            return m, n
    # Nothing fits: fall back to the smallest tile the kernel table admits for
    # the row, so the decline comes from the SMEM check rather than at build.
    admitted = [(m, n) for m, n in candidate_tiles if m in caps.tile_ms and n in caps.tile_ns] or candidate_tiles
    return min(admitted, key=lambda tile: (tile[1], tile[0]))


def _tile_points(spec: EngineSpec, facts) -> List[Tuple[Optional[int], Optional[int]]]:
    """Ordered (tile_m, tile_n) candidates: the rule's best guess first, then
    the rest of the SMEM-fitting domain for a caller that autotunes. Configs
    the kernel cannot fit are not runners-up — they would sit in the list only
    to decline at build."""
    caps = spec.capabilities
    if spec.name not in _TILE_RULE_CELLS:
        # No rule measured for this cell: its capability row has one point per
        # axis, so there is nothing to choose between anyway.
        return [(_sole(caps.tile_ms), _sole(caps.tile_ns))]
    best = _sm120_tiles(caps, facts)
    tile_choices = tile_domain(facts.d_qk, facts.d_v, facts.is_fp8 or facts.is_mxfp8)
    domain = [(m, n) for m in caps.tile_ms for n in caps.tile_ns if (m, n) in tile_choices and _sm120_fits_smem(facts, m, n)]
    return sorted(domain or [best], key=lambda mn: (mn != best, mn[1] != best[1], -mn[0]))


def _prefer_paged_d256_lpt(facts) -> bool:
    """Qualified full-prefill envelopes; current lengths may change after capture."""
    return (
        facts.device_cc in ((10, 0), (10, 7))
        and facts.thd
        and facts.has_paged_kv
        and facts.bottom_right
        and facts.causal
        and facts.window_left is None
        and (facts.right_bound or 0) == 0
        and facts.dtype == cudnn.data_type.BFLOAT16
        and (facts.d_qk, facts.d_v) == (256, 256)
        and facts.b == 1
        and (facts.h_q, facts.h_kv) in ((8, 1), (16, 2))
        and (4096 if facts.h_q == 8 else 2048) <= facts.s_q <= 16384
        and facts.s_q == facts.s_kv
        and facts.page_size in (16, 128)
        and not (facts.has_sink or facts.has_epilogue_gate)
    )


def _sched_points(caps: Capabilities, facts) -> List[Optional[int]]:
    """Ordered scheduler-policy candidates.

    The PRIMARY follows the measured preference for the graph's shape; the
    remaining domain follows for autotune. This is the one causal LPT/LPT_L2 oracle on
    the graph path — the adapters keep a None-input derivation only for
    standalone wrapper users who bypass ranking.
    """
    # The FLAVOR's domain, not the row-wide floor: a `sched_policies_by_d_shape`
    # claim (the Rubin f16 / FP8 (256, 256) LPT entries) must reach the ranking,
    # or LPT is only ever honoured when a caller REQUESTS the knob and is never
    # proposed for the first plan -- which is the whole point of claiming it.
    domain = effective_sched_policies(caps, facts)
    if len(domain) <= 1:
        return [_sole(domain)]
    if facts.thd and SCHED_NATURAL in domain:
        # A ragged batch walks LIVE units through batch_remap over a
        # machine-sized grid. Only flavors with a THD policy decoder can tune
        # its ordering; the dense rectangular LPT decoder cannot serve it.
        # D64/D128/D256 half THD implements policy ordering within the live list.
        # Expose alternatives for tuning and prefer LPT only
        # for the measured prefill families below.
        if 100 <= caps.sm_lo < 120 and not (facts.is_fp8 or facts.is_mxfp8) and _selected_d_shape(caps, facts) in ((64, 64), (128, 128), (256, 256)):
            primary = SCHED_NATURAL
            if (
                SCHED_LPT in domain
                and (_prefer_thd_pack_gqa(caps, facts) or (caps.sm_lo == 100 and (facts.d_qk, facts.d_v) == (64, 64) and facts.causal))
                and facts.window_left is None
                and not facts.right_band_widening
                and not (facts.has_sink and not _sm107_paged_half(caps, facts))
            ):
                # Packing does not remove the causal load imbalance: order the
                # live token tiles by their GPU-resident lengths. The decoder
                # still uses current lengths when a cached full-prefill plan
                # replays a prefix chunk, including tiny Q and low TP heads.
                # An attention sink changes nothing in that walk: on cc 10.7
                # paged THD (216 SMs, cuDNN 9.26 / 9.27, CUDA-graph replay) LPT
                # beat NATURAL on every sink cell measured -- 64/8 b128 q 1 / 4 /
                # 8 KV 2k 363 -> 269 us at cga2, 223 -> 178 us at cga1, b24 mixed
                # 61 -> 57 / 43 -> 41 us; 64/4 b128 225 -> 170 / 153 -> 129 us --
                # the same ranking as the sink-free twins.  The SM100 line keeps
                # the exclusion until it is measured there.
                primary = SCHED_LPT
            elif SCHED_LPT in domain and _prefer_paged_d256_lpt(facts):
                primary = SCHED_LPT
            return [primary] + sorted(domain - {primary})
        return [SCHED_NATURAL]
    causal_ish = facts.causal or facts.right_band_widening
    if caps.sm_hi == 80:
        # SM80's measured choices (see the adapter's flavor table): causal
        # always groups for L2; pure SWA prefers plain LPT only in the band
        # where the window walk is long enough to imbalance rows.
        if causal_ish:
            primary = SCHED_LPT_L2
        elif facts.window_left is not None and facts.window_left >= 0:
            primary = SCHED_LPT if 1024 <= facts.s_kv <= 16384 else SCHED_NATURAL
        else:
            primary = SCHED_NATURAL
    elif causal_ish and _sm120_d512_windowed(caps, facts):
        # Sliding window on the d512 flavor: a unit's K/V working set is a few
        # tiles whichever head it belongs to, so LPT_L2's per-KV-group batching
        # has nothing to protect in L2 and its row order only costs; the plain
        # heads-fastest LPT walk is the faster one for packed and unpacked units.
        primary = SCHED_LPT
    elif causal_ish and 107 <= caps.sm_lo < 120 and int(facts.h_q) == int(facts.h_kv):
        # Rubin (cc 10.7-11.x) without GQA: every KV head is read by exactly one Q head, so
        # LPT_L2 has no K/V sharing to group -- and it is not free (-10 % at
        # S=2K, see the constants above).  Plain LPT balances the triangle
        # while the grid is a few waves and costs at many, so choose by wave
        # count.  GQA shapes keep the L2 rule below (llama d128 H64/8 on the
        # same node: LPT_L2 +4..8 % over NATURAL at every S).  Rubin only until
        # the SM100 line is measured the same way.
        # The bound is the arch LINE, not `>= 107`: the SM120 rows sit at sm_lo=120 and
        # keep the SM100/SM120 L2-budget rule below (the wave-count constants above
        # are Rubin's cluster count and CGA rows, unmeasured on GeForce Blackwell).
        waves = (int(facts.b) * int(facts.h_q) * -(-int(facts.s_q) // _SM107_CGA_Q_ROWS)) / ((facts.device_sm_count or _SM107_MEASURED_SMS) // 2)
        primary = SCHED_LPT if waves <= _SM107_NO_GQA_LPT_MAX_WAVES else SCHED_NATURAL
    elif causal_ish and rubin_dense_d128_shared_leg(caps, facts) and _q_clusters_per_unit(caps, facts, None) == 1:
        # The cc 10.7 half row's shared d128 bodies (issue #1472) measured the one-cluster band their own way, so they
        # take this arm instead of the SM100 rule below.  Decode tile (S_q x PACK_G <= 128), a packed group, no window
        # and a cache of at least _SM107_DECODE_TILE_LPT_MIN_KV keys: plain LPT first (the measured 1-6 % median lead,
        # 11 % at 32k).  Everything else in the band keeps NATURAL first: a short cache (parity; the cells past 3 %
        # favour NATURAL), a sliding window (every unit's walk IS the window: GPT-OSS d64 SWA128 NATURAL 1.05-1.16
        # ahead), an unpacked unit (parity), and the packed cga2 prefill body one two-CTA cluster wide (128 < S_q x G
        # <= 512: 30 (cell, policy) pairs, median 1.013, LPT more than 3 % slower on 9 and never ahead; b128 64/4
        # q16 KV 1k 63 -> 67 us).  The other policy stays listed as the runner for autotune.
        if (
            _d128_decode_tile_fits(caps, facts)
            and _sm100_pack_g(caps, facts) > 1
            and facts.window_left is None
            and int(facts.s_kv) >= _SM107_DECODE_TILE_LPT_MIN_KV
        ):
            primary = SCHED_LPT
        else:
            primary = SCHED_NATURAL
    elif causal_ish and rubin_dense_d128_shared_leg(caps, facts) and _sm100_pack_g(caps, facts) > 1 and facts.window_left is not None:
        # Past one cluster, a sliding window on the packed shared prefill body (a GQA group under a band packs first,
        # _sm100_banded_gqa_packs): every packed unit walks the same window-bounded range, and the heads-fastest LPT
        # walk scatters the units that share a K/V window over the KV heads, so NATURAL's adjacent-tile order keeps
        # the window L2-resident -- MEASURED on the cc 10.7 board (216 SMs, cuDNN 9.26.0.51, CUDA-graph replay, kernel
        # time): b2 64/8 S4096 window 1024 NATURAL 214 us vs LPT 281 (1.31x; with a sink 215 vs 282), b8 64/8 S2048
        # window 512 373 vs 385, b1 64/8 S8192 window 2048 379 vs 393.  The unpacked Rubin body (the runner) keeps the
        # L2-budget rule below.
        primary = SCHED_NATURAL
    elif causal_ish and _d128_f16_flavor(caps, facts) and _q_clusters_per_unit(caps, facts, None) == 1:
        # One Q cluster per (batch, packed head) unit on the SM100 f16 row's
        # d128 flavor -- S_q * PACK_G <= 512: the decode tile's band
        # (S_q * PACK_G <= 128, one 128-row tile per unit -- decode and MTP)
        # and the one-cga2-cluster band above it (a chunk or speculative burst
        # of up to 512 / PACK_G tokens on the prefill tile).  Every unit walks
        # the same per-batch KV range with the same static tile weight, so LPT
        # has nothing to balance; LPT_L2's head grouping groups nothing when
        # the packed head IS the KV head (only the G / PACK_G packed heads of a
        # partially packed group).  Measured on B200 (S_kv=4096 paged, bf16,
        # bottom-right causal, kernel time): b=32 H=64/4 S_q=4 on the prefill
        # tile NATURAL 120.0 us vs LPT_L2 125.4 us (the decode tile keeps the
        # order); b=8 H=32/8 S_q=128 (512 rows, one cga2 cluster) NATURAL 62.9
        # us vs LPT_L2 65.3 us.  One row more restores the L2-budget rule
        # below.  The LPT variants stay behind as runners for autotune, as on
        # every causal graph.  (d256 32/2 packed at cga2 flips sign between
        # S_q=4 and 8 -- LPT_L2 / LPT / NATURAL 65.9 / 63.3 / 62.1 vs 64.8 /
        # 65.0 / 66.6 us -- so that flavor keeps the rule below until it is
        # measured on its own.)  On the cc 10.7 half row the shared d128 legs
        # take the arms above; its Rubin prefill body (the pre-folded scale)
        # keeps this rule -- NATURAL 1221 us vs LPT 1232 us at B128 64/8 Q8
        # KV2056 unpacked at cga2, parity, the LPT runner stays for autotune.
        primary = SCHED_NATURAL
    elif (
        causal_ish
        and caps.sm_lo == 100
        and not (facts.is_fp8 or facts.is_mxfp8)
        and _selected_d_shape(caps, facts) == (64, 64)
        and 1 in effective_cgas(caps, facts)
        and _d128_decode_tile_fits(caps, facts)
    ):
        # The d64 f16 decode tile (bottom-right causal MTP, S_q * PACK_G <= 128,
        # api_dsl._D64_DECODE_TILE_ROWS): one Q tile per (packed head, batch),
        # so every unit walks the same per-batch KV range and LPT has nothing
        # to balance -- the d128 decode band's reasoning above, on the flavor
        # whose prefill runs 256-row cga1 CTAs rather than cga2 clusters; its
        # band above the decode tile is unmeasured and keeps the L2-budget rule.
        primary = SCHED_NATURAL
    elif causal_ish and caps.sm_lo == 100 and facts.is_mxfp8 and _selected_d_shape(caps, facts) == (128, 128):
        # SM100 d128 MXFP8 (sm100/prefill_d128_mxfp8.py): plain LPT beats the L2-budget
        # arm's LPT_L2 on EVERY measured causal shape -- MEASURED on B200 (cc 10.0,
        # 148 SMs, 2026-09-22; CUPTI device time, L2 flushed per launch, 3 rounds,
        # one process per slot, controls <= 0.35 %): LPT over LPT_L2 +5.1 / +7.2 /
        # +5.0 / +4.3 / +7.9 / +10.4 % at B1 H24/8 S16K / S8K / S32K, B1 H128/128
        # S16K, B1 H8/8 S4K, B4 H24/8 S4K (GQA 1 and 3, 1.7-111 waves, 1-8 MiB per
        # head), and on three of the six LPT_L2 is slower than NATURAL.  O / Stats /
        # Amax_O are bitwise identical across the three policies, so this moves
        # only the PROPOSAL: LPT_L2 stays the first runner through order[SCHED_LPT]
        # for autotune and the row's domain is untouched.  The per-tensor FP8 d128
        # row measured SHAPE-DEPENDENT on the same node (LPT +2.9..+4.0 % at S=4K,
        # LPT_L2 +0.5..+0.9 % at S=16K, S=8K / 32K inside 1.5x their control) and
        # keeps the L2-budget arm below, unchanged; the other MXFP8 flavors
        # (d192x128 / d256 / d512) and the Rubin rows are unmeasured here and unchanged.
        # Row-keyed (``caps.sm_lo == 100`` names a cc RANGE, not a device): this also
        # proposes LPT on cc 10.3, where LPT vs LPT_L2 is unmeasured; it follows the
        # existing B200-measured precedent of the decode-tile arm above, keyed the same way.
        primary = SCHED_LPT
    elif causal_ish:
        # SM100/SM120: balance the triangular load; pick the LPT variant by
        # whether one head's K+V working set fits the L2 budget.
        elem = 1 if (facts.is_fp8 or facts.is_mxfp8) else 2
        one_head_bytes = int(facts.s_kv) * (int(facts.d_qk) + int(facts.d_v)) * elem
        primary = SCHED_LPT_L2 if one_head_bytes <= _SM100_L2_BUDGET_BYTES else SCHED_LPT
        if (
            caps.sm_lo == 100
            and not (facts.is_fp8 or facts.is_mxfp8)
            and _selected_d_shape(caps, facts) == (192, 128)
            and int(facts.h_q) == int(facts.h_kv)
            and facts.window_left is None
        ):
            # SM100 d192x128 f16/bf16 without GQA (DeepSeek-V3 / Kimi layers):
            # every KV head is read by one Q head, so LPT_L2 has nothing to
            # group, and at these head counts the causal triangle spans dozens
            # of waves, so LPT has little to balance -- the natural walk keeps
            # a head's K/V hot in L2.  MEASURED (cuDNN 9.30 yardstick, profiler
            # kernel sums, 2026-09-28) NATURAL / LPT_L2 / LPT of cuDNN: B300
            # 128 heads S=2K 1.06 / 1.09 / 1.09x, S=8K 1.08 / 1.12 / 1.25x;
            # B200 S=2K 1.03 / 1.05 / 1.06x, S=8K 1.02 / 1.02 / 1.12x, 64 heads
            # S=4K 1.02 / 1.03 / 1.05x.  GQA d192 graphs keep the L2 rule.
            primary = SCHED_NATURAL
        elif caps.sm_lo == 100 and _selected_d_shape(caps, facts) in ((128, 128), (64, 64)) and one_head_bytes < _SM100_D128_LPT_L2_MIN_BYTES:
            # The SM100 d128 / d64 f16 and per-tensor FP8 kernels: below a few
            # MiB per head the K/V of a head stays L2-resident under ANY walk, so
            # LPT_L2's head grouping only costs balance.  MEASURED on B200
            # (cuDNN 9.30 yardstick, profiler kernel sums, 2026-09-28), LPT vs
            # LPT_L2: llama 64/8 bf16 S=2K unpacked 1.14x vs 1.23x, S=8K packed
            # 1.04x vs 1.07x, S=32K packed 1.11x vs 1.07x (16 MiB/head: L2 wins);
            # e4m3 64/64 S=2K cga1 1.19x vs 1.33x, 64/8 S=2K/8K packed equal.
            primary = SCHED_LPT
    else:
        primary = SCHED_NATURAL
    order = {SCHED_LPT_L2: (SCHED_LPT, SCHED_NATURAL), SCHED_LPT: (SCHED_LPT_L2, SCHED_NATURAL), SCHED_NATURAL: (SCHED_LPT, SCHED_LPT_L2)}
    # The primary may be outside a row's DOMAIN (the SM107 f16 rows and the
    # d256 / d512 flavors carry no SCHED_LPT_L2 -- their kernels do not thread
    # its decode inputs).  Fall back along the
    # SAME preference order rather than to NATURAL: dropping straight to NATURAL
    # cost a causal Rubin FP8 graph the LPT load-balancing win, and listed
    # NATURAL twice ([0, 1, 0]), burning an autotune slot on a duplicate plan.
    # When the primary IS in domain this is byte-identical to the old form
    # (order[primary] never contains primary).
    chosen = next((p for p in (primary, *order[primary]) if p in domain), SCHED_NATURAL)
    runners = [p for p in order[primary] if p in domain and p != chosen]
    # A mask-free graph gains nothing from either LPT remap — the grid is
    # already balanced — so don't spend autotune slots on them.
    if not causal_ish and facts.window_left is None:
        runners = []
    return [chosen, *runners]


def select_d192_auto_knobs(
    params: Sm100TemplateParams,
    *,
    pertensor: bool,
    s_q: int,
    s_kv: int,
) -> tuple[int, int]:
    """Select the measured D192 scheduler and CGA defaults.

    This function is shared by graph heuristics and standalone callers. It
    chooses public knobs only; lowering canonicalizations and private codegen
    fields are derived separately in ``config_sm100``.
    """

    if params.split_kv > 1:
        return SCHED_NATURAL, 2

    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    mxfp8 = fp8 and not pertensor
    window_left = params.window_left
    window_right = params.window_right
    top_left = not params.bottom_right or d192_square_br_as_tl(params, s_q=s_q, s_kv=s_kv)

    mx_dense_mid_causal_cga1 = (
        mxfp8
        and not params.thd_varlen
        and window_left is None
        and window_right == 0
        and top_left
        and 4096 < s_kv <= 8192
        and (params.dtype_qkv == DTYPE_E5M2 or s_q >= 4096)
    )
    sched_policy = SCHED_NATURAL if mx_dense_mid_causal_cga1 else params.sched_policy

    pt_cga1 = (
        pertensor
        and not params.thd_varlen
        and (
            window_left is not None
            or (params.dtype_qkv == DTYPE_E5M2 and window_right is None)
            or (params.dtype_qkv == DTYPE_E4M3 and params.dtype_o in (DTYPE_E4M3, DTYPE_E5M2) and window_left is None and window_right == 0 and top_left)
        )
    )

    mx_cga1 = False
    if mxfp8:
        masked = window_right is not None
        sliding = window_left is not None
        if params.thd_varlen:
            if params.dtype_qkv == DTYPE_E5M2 and not masked:
                mx_cga1 = True
            elif masked and sliding:
                min_s_kv = 4096 if params.dtype_qkv == DTYPE_E4M3 else 2048
                mx_cga1 = s_kv >= min_s_kv
            elif masked:
                mx_cga1 = s_kv >= 2048
        elif masked:
            mx_cga1 = params.dtype_qkv == DTYPE_E4M3 or sliding or s_kv <= 4096
    mx_cga1 = mx_cga1 or mx_dense_mid_causal_cga1
    return sched_policy, 1 if pt_cga1 or mx_cga1 else 2


def select_d256_auto_knobs(
    params: Sm100TemplateParams,
    *,
    pertensor: bool,
    s_q: int,
    s_kv: int,
    device_cc: Optional[tuple[int, int]] = None,
    paged: bool = False,
) -> tuple[int, int]:
    """Select the measured D256 scheduler and CTA geometry: the 2-CTA pair for f16 and for the cc 10.7 MXFP8 dense
    unsplit legs (config_sm100.d256_mxfp8_pair_leg), one CTA for the other quantized legs."""

    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    if not fp8:
        return params.sched_policy, 2
    if params.split_kv > 1 or params.thd_varlen:
        return SCHED_NATURAL, 1
    cga = 2 if d256_mxfp8_pair_leg(device_cc=device_cc, pertensor=pertensor, thd=params.thd_varlen, paged=paged, split_kv=params.split_kv) else 1

    no_mask = params.window_left is None and params.window_right is None and not params.seq_kv_lens_present
    if no_mask:
        return SCHED_NATURAL, cga

    top_left = not params.bottom_right or d256_square_br_as_tl(params, s_q=s_q, s_kv=s_kv)
    pt_lpt_l2 = (
        pertensor
        and params.window_left is None
        and params.window_right is not None
        and not params.seq_kv_lens_present
        and top_left
        and _ceil_div(s_q, 256) >= 16
    )
    if pt_lpt_l2:
        return params.sched_policy, cga
    return SCHED_LPT, cga


def select_d512_auto_knobs(params: Sm100TemplateParams) -> tuple[int, int]:
    """Select the measured D512 scheduler and fixed MXFP8 CTA1 geometry."""

    if params.thd_varlen:
        return SCHED_NATURAL, 1
    if params.window_right is not None:
        return SCHED_LPT, 1
    return params.sched_policy, 1


# --- d128 decode-shaped launches (SM100-line f16/bf16) ----------------------
#
# The SM100 f16 row's (128, 128) flavor has two tiles behind TILE_CGA_M
# (_d128_decode_tile_fits below): cga1 IS the decode tile -- one independent
# 128-row CTA per (batch, packed head) unit, sm100/decode_d128_f16.py -- and
# cga2 the prefill pipeline, 512 rows per cluster.  The rules here measure a
# unit by its live rows, S_q * PACK_G -- the kernel's packed subgroup (the
# whole GQA group when it divides the tile, its largest divisor that does
# under partial PackGQA, 96/8 -> 4; 1 unpacked), not the raw ratio G:
#
#   * width: cga1 while one decode tile covers the unit (select_d128_auto_cga,
#     the one rule behind _d128_decode_tile_fits on the graph path and the
#     standalone adapter's default in api_dsl.SdpaFwdDslSm100.template_params);
#   * scheduler: NATURAL first while one cga2 cluster covers the unit -- the
#     decode tile's band and the band above it up to 512 rows (_sched_points;
#     the cc 10.7 shared legs re-measured that band and lead the packed decode
#     tile with plain LPT from a 2k cache, _SM107_DECODE_TILE_LPT_MIN_KV);
#   * split: the wave-cost model sees the true CTA count (ctas_per_tile=1 at
#     cga1), so a small-batch decode splits finer than the cga2 plan did --
#     b=8 h=64/4 S_q=1 S_kv=4096 paged bf16, B200 kernel time: SPLIT_KV=4 at
#     27.5 us where the cga2 plan split in two at 39.1 us -- and choose_split_kv
#     carries two corrections that band exposed at its edges.  The unsplit leg
#     pays no combine, so a many-row chunk no longer splits (b=8 h=32/8 S_q=64
#     paged bottom-right, 16384 combine rows: split 2 measured 75.0 us against
#     67.6 us unsplit; dense b=4 h=32/8 S_q=128 causal at cga2: 155.5 vs 120.2
#     us), and a lone combine block's serial walk over its partials is priced
#     at its latency floor, so a few-unit b=1 launch stops one power of two
#     short of the full wave (h=16/2 S_kv=32768: split 32 at 39.7 us where 64
#     measured 50.8; h=64/4 S_kv=4096: split 8 at 20.3 us where 16 measured
#     22.3).  Neither correction is d128's alone: the floor moves the d256
#     paged b=1 h=64/4 S_q=1 S_kv=4096 lead from split 16 (29.8 us) to 8
#     (24.5 us), and the unsplit-leg accounting the d256 / d192x128 2k-KV chunk
#     leads (choose_split_kv's docstring).
#
# Provenance: the split-model numbers above were measured on the d128 prefill
# kernel's cga1 configuration (one 256-row Q/O-aliased CTA per unit), the
# width the graph path led with before the decode tile took over TILE_CGA_M=1
# (PR #1094).  The model's inputs are unchanged on the decode tile (one CTA
# per unit, 128-row KV tiles, the shared sm100/split_combine), so the choices
# carry over; the floor is expressed in KV tiles of MAIN-kernel time, and a
# decode-tile KV tile may run faster than the prefill kernel's, so a
# re-measurement on the decode tile could only raise it.
_D128_SHAPE = (128, 128)


def _d128_f16_flavor(caps: Capabilities, facts) -> bool:
    """The d128 flavor of the half-precision rows whose (128, 128) cga domain
    holds two tiles (cga1 the decode tile, cga2 the prefill pipeline): the SM100
    f16 row and, since issue #1472, the cc 10.7 f16 row (the shared tile compiled
    for cc 10.7); the fp8 / mxfp8 rows keep d128 on the row-wide cga2
    (``caps.sm_lo == 100`` names the cc 10.0-10.6 range, 107 the cc 10.7+ line)."""
    return (
        caps.sm_lo in (100, 107)
        and _selected_d_shape(caps, facts) == _D128_SHAPE
        and not (facts.is_fp8 or facts.is_mxfp8)
        and any(shape == _D128_SHAPE for shape, _ in caps.cgas_by_d_shape)
    )


def _sm100_pack_g(caps: Capabilities, facts) -> int:
    """The heads a packed set folds into one Q tile row-group -- the kernel's
    ``Cfg.PACK_G`` (:func:`_pack_gqa_group`: the whole ratio G when it divides
    the tile, its largest divisor that does under partial PackGQA, 96/8 -> 4);
    1 when the row cannot pack this graph (MHA, THD, an epilogue gate, or a
    ratio with no factor in common with the tile). A (batch, packed head) unit
    holds ``S_q * PACK_G`` live rows, so this -- not the raw ratio -- is what
    the one-cluster scheduler rule measures the unit by."""
    return _pack_gqa_group(caps, facts, 128, _pack_gqa_eligible(caps, facts, 128))


def _q_clusters_per_unit(caps: Capabilities, facts, cga: Optional[int]) -> int:
    """Q clusters one (batch, packed head) unit launches at cluster width ``cga``
    (``None`` = the flavor's default width, the cga2 prefill cluster)."""
    return _ceil_div(facts.s_q * _sm100_pack_g(caps, facts), _pack_gqa_tile_q(caps, facts, 128, cga))


def select_d128_auto_cga(*, s_q: int, pack_g: int, thd: bool, thd_decode_leg: bool = False) -> int:
    """The d128 f16 cluster width for a unit of ``S_q * pack_g`` live rows:
    cga1 -- the decode tile -- when one of its 128-row tiles covers them
    (``cga_tile_m(128, 1)``, config_sm100.CfgD128Decode), cga2 -- the prefill
    pipeline -- otherwise.  A ragged (THD) graph keeps cga2 unless it rides the
    decode tile's ragged-Q leg (``thd_decode_leg``: S_q(max) == 1 over paged
    KV, engines._thd_decode_leg); the prefill tile's THD_VARLEN leg is the only
    other ragged form, and engines.mismatch / api_dsl.check_support decline a
    pinned cga1 there.  The one rule behind the graph path
    (:func:`_d128_decode_tile_fits`, which derives ``pack_g`` from the
    candidate's packing) and the standalone adapter's default width
    (api_dsl.SdpaFwdDslSm100.template_params)."""
    if thd and not thd_decode_leg:
        return 2
    return 1 if s_q * pack_g <= cga_tile_m(128, 1) else 2


def _sm100_params_from_facts(facts, *, split_kv: int, sched_policy: int) -> Sm100TemplateParams:
    dtype_codes = {
        cudnn.data_type.FP8_E4M3: DTYPE_E4M3,
        cudnn.data_type.FP8_E5M2: DTYPE_E5M2,
        cudnn.data_type.BFLOAT16: DTYPE_BF16,
        cudnn.data_type.HALF: DTYPE_FP16,
    }
    return Sm100TemplateParams(
        dtype_qkv=dtype_codes[facts.dtype],
        dtype_o=dtype_codes.get(facts.dtype_o, -1),
        window_left=facts.window_left,
        window_right=(facts.right_bound if facts.right_band_widening else 0 if facts.causal else None),
        bottom_right=facts.bottom_right,
        seq_kv_lens_present=facts.padded,
        seq_q_lens_present=facts.seq_q_trim,
        sched_policy=sched_policy,
        thd_varlen=facts.thd,
        split_kv=split_kv,
    )


# The d128 decode tile's Q rows per CTA (config_sm100.CfgD128Decode: TILES_Q=1 x
# TILE_M=128 x CTA_MMA=1).  The SM100 f16 row's (128, 128) flavor has two
# tiles behind TILE_CGA_M: cga2 is the prefill pipeline (512 rows per
# cluster), cga1 the decode tile (sm100/decode_d128_f16.py).
_D128_DECODE_TILE_ROWS = 128
# The d64 f16 decode tile's Q rows (api_dsl._D64_DECODE_TILE_ROWS): the one d64 leg
# narrower than the flavor's 256-row cga1 prefill, selected by the adapter once
# S_q x packed heads fit it (and the cga knob is unset or 1).
_D64_DECODE_TILE_ROWS = 128


def _d128_dense_two_widths(caps: Capabilities, facts) -> bool:
    """A dense d128 half graph both widths serve at any S_q: the SM100 f16 row and the cc 10.7 shared leg."""
    sm100_dense = _sm100_f16(caps, facts) and _d128_f16_flavor(caps, facts) and not facts.thd and not facts.has_paged_kv
    return sm100_dense or rubin_dense_d128_shared_leg(caps, facts)


def d128_dense_waves(caps: Capabilities, facts, pack_gqa: Optional[bool] = None) -> Optional[Tuple[int, int]]:
    """(decode-tile waves, cga2 waves) of a dense d128 half graph both widths serve; ``None`` elsewhere."""
    if not _d128_dense_two_widths(caps, facts):
        return None
    if pack_gqa is None:
        pack_gqa = _pack_gqa_eligible(caps, facts, _D128_DECODE_TILE_ROWS)
    pack_g = _pack_gqa_group(caps, facts, _D128_DECODE_TILE_ROWS, pack_gqa)
    units, rows = facts.b * (facts.h_q // pack_g), facts.s_q * pack_g
    sms = facts.device_sm_count or (_SM107_MEASURED_SMS if caps.sm_lo == 107 else 148)
    return (
        _ceil_div(units * _ceil_div(rows, _D128_DECODE_TILE_ROWS), sms),
        _ceil_div(units * _ceil_div(rows, 4 * _D128_DECODE_TILE_ROWS), sms // 2),
    )


def _d128_dense_cga1_waves_win(caps: Capabilities, facts, pack_gqa: Optional[bool] = None) -> bool:
    """Whether the dense d128 decode tile beats the cga2 prefill cluster on wave count (_D128_CGA2_WAVE_COST)."""
    waves = d128_dense_waves(caps, facts, pack_gqa)
    return waves is not None and facts.s_kv < _D128_CGA1_MAX_KV and waves[0] <= _D128_CGA2_WAVE_COST * waves[1]


def _d128_decode_tile_fits(caps: Capabilities, facts, pack_gqa: Optional[bool] = None) -> bool:
    """Whether one d128 decode tile covers a KV head's live Q rows.

    ``S_q * pack_g <= 128`` with ``pack_g`` the CANDIDATE's own packing:
    ``pack_gqa=True`` is the packed leg (one unit carries the packed group
    ``p = Cfg.PACK_G`` -- the whole GQA group ``G`` when it divides the tile,
    else its largest divisor that does (partial PackGQA: 96/8 packs 4) --
    ``p`` rows per token), ``False`` the unpacked one (one head, ``S_q`` rows),
    and ``None`` -- the graph-level question -- reads as the packed leg when
    the row can pack this graph, which is the leg a decode-shaped graph
    proposes first.  The fit is a property of the candidate, not the graph:
    at ``S_q * p > 128 >= S_q`` the packed leg keeps the prefill tile while
    the unpacked runner-up rides the decode tile.  THD: the decode tile has no
    THD_VARLEN leg; a ragged graph rides it only as the ragged-Q-over-paged-KV
    leg (``engines._thd_decode_leg``: S_q(max) == 1, page pools, ragged Stats)
    and keeps the prefill tile otherwise.  Measured on B200 (b=32, H=64/4,
    d128, S_kv=4096, page 16, bf16): the prefill tile at cga2 119 us, the
    decode tile 49 us -- see the kernel docstring; at S_q * G > 128 the prefill
    tile's second sub-tile is live and cga2's collective MMA halves per-CTA K/V
    traffic, so the rule stops there rather than at a measured crossover.
    """
    thd_decode_leg = bool(facts.thd and _thd_decode_leg(caps, facts))
    if facts.thd and not thd_decode_leg:
        return False
    if pack_gqa is None:
        pack_gqa = _pack_gqa_eligible(caps, facts, _D128_DECODE_TILE_ROWS)
    # The kernel's HEADS_PER_TILE for this leg (1 unpacked): the launch the
    # split model sees, and the rows one unit really carries.
    pack_g = _pack_gqa_group(caps, facts, _D128_DECODE_TILE_ROWS, pack_gqa)
    return select_d128_auto_cga(s_q=facts.s_q, pack_g=pack_g, thd=facts.thd, thd_decode_leg=thd_decode_leg) == 1


# Longest average per-tile K/V walk (keys) at which the packed MXFP8 d128 leg still takes the single 256-row CTA
# (causal: half of S_kv; sliding window: the window plus one tile).  Causal S=2K / S=4K and window 1024 sit at or
# below it, causal S=8K and above keep the cga2 pair -- see _auto_sched_cga.
_SM100_MXFP8_D128_CGA1_MAX_WALK = 2048


def _in_flavor_sched_domain(caps: Capabilities, facts, selected: int, seed: int) -> int:
    """``selected`` when the selected flavor's scheduler domain honours it, else ``seed`` (drawn from that
    domain by the caller), else the domain's lowest policy.  The measured D192 / D256 / D512 pickers answer
    for the SM100 kernels; a row whose kernel of that flavor threads no LPT inputs (the cc 10.7 MXFP8 row at
    (256, 256)) claims NATURAL only, and a set outside the domain is dropped by recommend()'s re-validation --
    which cost every masked d256 MXFP8 graph on cc 10.7 its FALLBACK entry and its leading A set (only the
    NATURAL runner survived)."""
    domain = effective_sched_policies(caps, facts)
    if selected in domain:
        return selected
    return seed if seed in domain else min(domain)


def _auto_sched_cga(spec: EngineSpec, facts, *, split_kv: int, sched_policy: int, pack_gqa: Optional[bool] = None) -> tuple[int, Optional[int]]:
    """``pack_gqa`` is the candidate's packing where the width depends on it
    (the d128 f16 SM100 flavor, :func:`_d128_decode_tile_fits`); ``None`` asks
    the graph-level question.  The other flavors' rules ignore it."""
    caps = spec.capabilities
    domain = effective_cgas(caps, facts, split_kv)
    selected_shape = _selected_d_shape(caps, facts)
    if selected_shape == (192, 128) and (split_kv or 1) > 1 and domain == frozenset({1}):
        # The packed THD split ABI uses the single-Q tile. The existing dense
        # D192 split remains a two-CTA lowering.
        return sched_policy, 1
    if selected_shape == (64, 64) and domain == frozenset({1, 2}) and not (facts.is_fp8 or facts.is_mxfp8):
        # d64 runs cga1 on BOTH legs -- it is the prefill width (the narrow
        # slabs need no collective MMA, and a 512-row cga2 cluster wastes most
        # of a narrow diagonal band) and the decode tile's width. The two are
        # told apart by TemplateParams.decode_tile, not by this knob.
        return sched_policy, 1
    if selected_shape == (128, 128) and facts.is_fp8 and not facts.is_mxfp8 and 1 in domain and not facts.thd and not facts.has_paged_kv:
        # Per-tensor FP8 d128 on a dense graph: one 256-row CTA (cga1) for the
        # unsplit leg, the geometry cuDNN's own fp8 kernel runs.  MEASURED on
        # B200 (cuDNN 9.30 yardstick, 2026-09-28): llama 64/8 e4m3 causal S=2K
        # 1.18x -> 1.14x, S=8K 1.09x -> 1.07x, 64/64 S=2K 1.49x -> 1.33x, AR-DiT
        # no-split 1.07x -> 1.05x, dense S=2K unchanged.  The split leg keeps
        # cga2 (split_cgas_by_d_shape); the THD and paged legs are cga2-only.
        return sched_policy, 1
    if selected_shape == (128, 128) and facts.is_mxfp8 and caps.sm_lo == 100 and 1 in domain and not facts.thd and not facts.has_paged_kv:
        # MXFP8 d128 on a dense graph.  cga1 = one 256-row CTA of the prefill body (the geometry cuDNN's own MXFP8
        # d128 kernel runs; no pair handshake at the tile boundary, but no K/V multicast either).  MEASURED 2026-10-07
        # against the cga2 pair in the benchmark regime (same-node A/B, cuDNN 9.27; O bit-identical between the
        # widths): the PACKED GQA leg wins at cga1 while its K/V walk is short -- llama 64/8 causal S=2K 0.97x (B200)
        # / 0.97x (B300) with the three-stage cga1 ring, S=4K 1.00x / 0.99x, sliding window 1024 at S=2K 0.96x /
        # 0.95x -- and loses once it is long (causal S=8K 1.01x / 1.00x, S=16K 1.07x on B300), where the pair's
        # shared K/V stream pays; an unpacked leg (dense GQA, any MHA) and the split leg keep the pair (dense S=2K
        # +3 % at cga1 on B200; AR-DiT split +2..3 %).
        packed = pack_gqa if pack_gqa is not None else _sm100_banded_gqa_packs(caps, facts)
        if (split_kv or 1) > 1 or not packed:
            return sched_policy, 2
        if facts.window_left is not None:
            kv_walk = min(facts.s_kv, facts.window_left + 128)
        else:
            kv_walk = facts.s_kv // 2 if facts.causal else facts.s_kv
        return sched_policy, 1 if kv_walk <= _SM100_MXFP8_D128_CGA1_MAX_WALK else 2
    if facts.device_cc == (10, 7) and supports_paged_prefill_cga1(
        (facts.d_qk, facts.d_v),
        device_cc=facts.device_cc,
        fp8=facts.is_fp8 or facts.is_mxfp8,
        thd=facts.thd,
        paged=facts.has_paged_kv,
        split_kv=split_kv,
        max_q=facts.s_q,
    ):
        # A packed query fits the two-slab single-CTA tile at 256 rows.
        # Prefer it only when the two-CTA tile would need another grid wave.
        # Split/decode keeps its separate 128-row tile and existing choice.
        sm_count = facts.device_sm_count or 0
        group = facts.h_q // facts.h_kv if facts.h_kv else 0
        units = facts.b * facts.h_kv
        # On cc 10.7 paged THD the sink, GQA16 and single-token packed batches take the same choice (216 SMs,
        # cuDNN 9.26 / 9.27, CUDA-graph replay, kernel time; the sink cells' ranking matched their sink-free
        # twins'): 64/8 b128 q 1 / 4 / 8 KV 2k two-slab cga1 LPT 178-180 us against the cga2 plan's 269-276
        # (the wave rule below prefers cga1: 1024 units), b24 mixed 41 vs 57 us (192 units); 64/4 (GQA16) b128
        # q 4 / 8 129-131 vs 170-172 us, and b24 mixed (96 units: one wave either way) keeps cga2 at 40 vs 42 us.
        # SM100 automatic single-CTA paged prefill selection remains disabled pending separate qualification: the
        # device guard on this branch keeps the preference on cc 10.7 while the capability helper admits the two-slab
        # body on SM100 for explicit candidates (#1493; an explicit cga=1 pin is honoured there regardless).
        rubin_paged = _sm107_paged_half(caps, facts)
        prefer = (
            pack_gqa is not False
            and _prefer_thd_pack_gqa(caps, facts)
            and facts.dtype == cudnn.data_type.BFLOAT16
            and facts.bottom_right
            and facts.window_left is None
            and not facts.right_band_widening
            and (rubin_paged or not facts.has_sink)
            and not facts.has_epilogue_gate
            and group in ((4, 8, 16) if rubin_paged else (4, 8))
            and (rubin_paged or 1 < facts.s_q)
            and facts.s_q * group <= 256
            and 2048 <= facts.s_kv <= 32768
            and facts.s_kv >= 4 * facts.s_q
            and sm_count >= 2
            and _ceil_div(units, sm_count) < _ceil_div(units, sm_count // 2)
        )
        return sched_policy, 1 if prefer else 2
    if selected_shape == (128, 128) and domain == frozenset({1, 2}) and not (facts.is_fp8 or facts.is_mxfp8):
        # The f16 SM100 and cc 10.7 rows: cga1 = the decode tile when one of its 128-row
        # tiles covers the head's Q rows, else the cga2 prefill pipeline.
        return sched_policy, (1 if _d128_decode_tile_fits(caps, facts, pack_gqa) or _d128_dense_cga1_waves_win(caps, facts, pack_gqa) else 2)
    if selected_shape == (256, 256) and any(shape == selected_shape for shape, _ in caps.cgas_by_d_shape):
        params = _sm100_params_from_facts(facts, split_kv=split_kv, sched_policy=sched_policy)
        selected_sched, selected_cga = select_d256_auto_knobs(
            params, pertensor=facts.is_fp8, s_q=facts.s_q, s_kv=facts.s_kv, device_cc=facts.device_cc, paged=facts.has_paged_kv
        )
        if selected_cga not in domain:
            raise ValueError(f"D256 heuristic selected cga={selected_cga} outside the declared domain {sorted(domain)}")
        return _in_flavor_sched_domain(caps, facts, selected_sched, sched_policy), selected_cga
    if selected_shape == (512, 512) and facts.is_mxfp8 and caps.sm_lo == 100:
        params = _sm100_params_from_facts(facts, split_kv=split_kv, sched_policy=sched_policy)
        selected_sched, selected_cga = select_d512_auto_knobs(params)
        if selected_cga not in domain:
            raise ValueError(f"D512 heuristic selected cga={selected_cga} outside the declared domain {sorted(domain)}")
        return _in_flavor_sched_domain(caps, facts, selected_sched, sched_policy), selected_cga
    if selected_shape != (192, 128) or not any(shape == (192, 128) for shape, _ in caps.cgas_by_d_shape):
        return sched_policy, _sole(domain)
    params = _sm100_params_from_facts(facts, split_kv=split_kv, sched_policy=sched_policy)
    selected_sched, selected_cga = select_d192_auto_knobs(params, pertensor=facts.is_fp8, s_q=facts.s_q, s_kv=facts.s_kv)
    if selected_cga not in domain:
        raise ValueError(f"D192 heuristic selected cga={selected_cga} outside the declared domain {sorted(domain)}")
    return _in_flavor_sched_domain(caps, facts, selected_sched, sched_policy), selected_cga


# --- pack_gqa (GQA head packing) --------------------------------------------


def _pack_gqa_wins(facts, tile_q: int) -> bool:
    """Pack when the Q sequence does not fill a single tile, then we can further
    apply split_kv on top of GQA packing.

    TODO: we may enhance this heuristic logic in the future by considering more
    factors such as the device SM count.
    """
    return facts.s_q < tile_q


def _sm100_banded_gqa_packs(caps: Capabilities, facts) -> bool:
    """The SM100 prefill rows pack a GQA group under a diagonal band (causal,
    bottom-right causal, sliding window) at ANY S_q, not only on decode shapes.

    A packed unit holds ``tile_m / G`` tokens of every head in the group, so
    under a band its K/V walk is bounded by those few tokens' diagonal instead
    of a whole 128-token tile's, and the group's heads share every K/V tile the
    unit streams.  MEASURED on B200 (llama 3.1 layer, B=2, H=64/8, d=128,
    S=2048 top-left causal, profiler kernel sums, cuDNN 9.30 as the yardstick):
    bf16 unpacked 0.155 ms (1.23x cuDNN) -> packed 0.125 ms (0.99x); e4m3
    unpacked 0.150 ms (1.39x) -> packed 0.127 ms (1.18x); S=8192 e4m3 causal
    1.17x -> 1.09x.  Dense (no band) graphs are unmoved (e4m3 0.184 vs 0.183
    ms), so the plain decode rule keeps them.  SM120 keeps its own row's rules.

    The cc 10.7 half row's shared dense d128 leg (:func:`rubin_dense_d128_shared_leg`:
    the SM100 prefill body under PackGQA, issue #1472) takes the same rule -- MEASURED
    on a 216-SM cc 10.7 board (cuDNN 9.26.0.51, CUDA-graph replay, kernel time,
    bottom-right causal, every listed plan timed): b2 64/8 S=2048 packed 66.7 us vs
    unpacked 76.8 (backend 70.6), b4 64/4 S=1024 50.8 vs 60.8 (54.2), b1 32/8 S=4096
    60.5 vs 64.0 (61.0), b1 64/4 S=512 15.3 vs 16.3 (17.0), with and without a sink;
    the unpacked Rubin tile stays listed as the runner-up."""
    banded = not facts.thd and (facts.causal or facts.window_left is not None) and facts.h_q != facts.h_kv
    return banded and ((caps.sm_lo == 100 and caps.sm_hi < 107) or rubin_dense_d128_shared_leg(caps, facts))


def _pack_gqa_tile_q(caps: Capabilities, facts, tile_m: Optional[int], cga: Optional[int] = None, *, split_kv: int = 1) -> int:
    """The Q rows one grid tile covers, for :func:`_pack_gqa_wins`.

    The SM100 family runs CGA tiles. D192 accepts CGA1 and CGA2, so callers must
    pass the CGA of the complete assignment they are evaluating. SM90 and SM120
    launch one CTA per tile, so it is ``tile_m`` itself.
    """
    if caps.sm_lo == 90 and caps.sm_hi == 90:
        # One 64-row CTA per grid tile; the cluster chain below would answer 128.
        return tile_m or _sole(caps.tile_ms)
    if caps.sm_lo >= 120 and caps.sm_hi < 130:
        return tile_m or 128
    if facts.d_qk <= 128 and facts.d_v <= 128:
        if _selected_d_shape(caps, facts) == (64, 64):
            if cga == 1 and split_kv > 1 and thd_split_domain(caps, facts):
                return _D64_DECODE_TILE_ROWS
            # The native d64 flavor is a TILES_Q=2 prefill on every row that has
            # it (f16, per-tensor FP8, MXFP8): 256 rows at cga1, the width its
            # rows run (config_sm100.CfgD64; an unset knob means that width, not
            # CfgD128's cga2 default).  Only its f16 decode tile is narrower (128
            # rows): _split_launch keys that off the packed row count, and the
            # pack decision is the same either way (S_q x G <= 128 is < 256).
            return cga_tile_m(64, 1 if cga is None else cga)
        if (facts.is_fp8 or facts.is_mxfp8) and cga == 1:
            # Quantized d128 prefill at cga1 keeps TILES_Q=2 (256 rows),
            # unlike the half decode/split tile that cga_tile_m models.
            return 256
        if cga == 1 and supports_paged_prefill_cga1(
            (facts.d_qk, facts.d_v),
            device_cc=facts.device_cc,
            fp8=facts.is_fp8 or facts.is_mxfp8,
            thd=facts.thd,
            paged=facts.has_paged_kv,
            split_kv=split_kv,
            max_q=facts.s_q,
        ):
            return 256
        return cga_tile_m(128, cga)
    if facts.d_qk <= 192 and facts.d_v <= 128:
        if cga == 1 and _sm100_f16(caps, facts) and facts.thd and not facts.has_paged_kv:
            return _D128_DECODE_TILE_ROWS
        return cga_tile_m(192, cga)
    if facts.d_qk <= 256 and facts.d_v <= 256:
        return cga_tile_m(256, cga)
    return cga_tile_m(512, cga)


def _sm120_d512_windowed(caps: Capabilities, facts) -> bool:
    """A sliding-window graph on the SM120 d512 flavor.

    Two rules key on it. Pack the GQA group into the Q tile: a packed unit holds
    ``tile_m / G`` tokens, so its key span is that many tokens plus the window
    instead of ``tile_m`` plus the window, fewer K/V tiles through the L2->SMEM
    path and less masked-out MMA at the same DRAM bytes. And walk the units with
    plain LPT (see :func:`_sched_points`). Without a window the decode rule
    alone decides the packing. The FP8 and MXFP8 flavors take only the LPT
    walk: their 64-key tile covers a 64-token unit's window in as many tiles as
    a packed one-token unit, so packing saves no MMA there (measured 8-11%
    slower on FP8, 1-3% on MXFP8).
    """
    return caps.sm_lo >= 120 and caps.sm_hi < 130 and facts.window_left is not None and pick_flavor(facts.d_qk, facts.d_v, fp8=facts.is_fp8) == D512_FLAVOR


def _d256_decode_tile_selected(caps: Capabilities, facts, pack_g: int, split_kv: Optional[int] = None) -> bool:
    """Whether the f16/bf16 row lowers this graph onto the d256 decode tile
    (sm100/decode_d256_f16.py on the Blackwell row, sm107/decode_d256_f16.py on
    the Rubin row) -- the twin of ``SdpaFwdDslSm100._decode_q_tile``: the
    (256, 256) flavor, half inputs, dense (not THD), no pre-folded scale,
    S_q x packed heads within the tile's N extent, ``pack_g`` being the DECODE
    tile's group (:func:`_decode_tile_pack_g`); a GATED graph rides it only
    through a split (``split_kv`` >= 2: the gate moves into the split combine;
    None = some plan).  ONE definition for the three consumers:
    ``engines.d256_decode_tile_selected``."""
    return d256_decode_tile_selected(caps, facts, pack_g, split_kv)


def _decode_tile_pack_g(facts, pack_g: int) -> int:
    """The heads the DECODE tile packs per token for a set whose prefill-tile
    group is ``pack_g`` (:func:`_pack_gqa_group`; 1 = unpacked): the whole GQA
    ratio.  sm100/decode_d256_f16.py packs ``HEADS_PER_TILE = QH_PER_KH`` --
    one unit per (KV head, batch) with every head of the group in its 16-column
    tile (96/8: 12 live rows, four zero-filled) -- and has no partial form, so
    the prefill tile's ``gcd(G, 128)`` (4 for 96/8, partial PackGQA) is not
    this launch's geometry: fed that, S_q = 2 x 96/8 (24 packed rows, the
    prefill tile's graph -- the adapter counts S_q x G) would read as an
    8-row decode launch and be costed with the decode model."""
    return (facts.h_q // facts.h_kv) if pack_g > 1 else 1


def _pack_gqa_eligible(caps: Capabilities, facts, tile_m: int, split_kv: Optional[int] = None, cga: Optional[int] = None) -> bool:
    """Whether a packed set can be built at ``tile_m``: the row offers packing,
    the graph carries no fused epilogue gate (its per-head
    gate tile cannot address a packed tile's interleaved rows -- mismatch()
    declines the same pair) -- unless the set rides the d256 decode tile's
    SPLIT, whose combine applies the gate (``split_kv``: the set's split, None =
    some plan) -- there is a group to pack and the ratio divides
    the tile -- or, on a flavor with partial PackGQA, shares a factor with it
    (96/8 packs 4 of its 12 heads; 24/8 has nothing to pack and stays unpacked).
    THD prefill packs only on a flavor advertising token-unit worklists and
    packed-head Stats stores; the decode tile's ragged-Q leg remains separate.
    cc 10.7 half packs dense D128 on the shared SM100 bodies (issue #1472) and dense d256
    on the decode tile (below); its other nonpaged half graphs stay unpacked.
    The row's per-flavor wiring (``pack_gqa_d_shapes``) is honoured here as
    ``mismatch()`` honours it: a packed proposal on a flavor the row keeps
    unpacked would only be declined there, and when the base leg is a split the
    unpacked alternative is never emitted, so the engine would offer NOTHING
    (the Rubin half row at d256, once its paged decode-shaped graphs reached
    the heuristics).

    The d256 DECODE tile is the one packing that ignores ``tile_m``: it packs the
    WHOLE group into its 16-row Q tile (``HEADS_PER_TILE = QH_PER_KH``; 24/2 = 12
    live rows + 4 zero tail rows), dense or paged.  On the Rubin half row the
    dense d256 prefill kernel runs unpacked, so a packed d256 proposal there is
    eligible exactly when the whole group rides the tile -- the predicate
    ``mismatch()`` reads (``d256_decode_tile_selected``) -- or when the graph is
    the paged half THD prefill's own d256 packing (CGA2, unsplit;
    ``config_sm100.supports_paged_d256_pack_gqa``, the route the row served
    before the tile; ``cga`` is the set's CGA span, None = some plan); the
    dense-cache exclusion of the Rubin half row applies to neither.
    The SM100 line keeps its rules: its partial PackGQA already admits such
    groups on the prefill tile."""
    half = not (facts.is_fp8 or facts.is_mxfp8)
    group = (facts.h_q // facts.h_kv) if facts.h_kv else 0
    rubin_decode_tile = caps.sm_lo == 107 and half and group > 1 and facts.h_q % facts.h_kv == 0 and _d256_decode_tile_selected(caps, facts, group, split_kv)
    # The paged half THD prefill pipeline packs d256 too (CGA2, unsplit) -- the route the Rubin row served before the
    # decode tile existed; the tile's d256 rule exempts it exactly as engines.mismatch does.
    rubin_paged_thd_d256 = (
        caps.sm_lo == 107
        and half
        and supports_paged_d256_pack_gqa(
            (facts.d_qk, facts.d_v), device_cc=facts.device_cc, fp8=not half, thd=facts.thd, paged=facts.has_paged_kv, cga=cga, split_kv=split_kv or 1
        )
    )
    return (
        True in caps.pack_gqas
        and (caps.pack_gqa_d_shapes is None or _selected_d_shape(caps, facts) in caps.pack_gqa_d_shapes)
        and not (caps.sm_lo == 107 and half and not facts.has_paged_kv and not rubin_decode_tile and not rubin_dense_d128_shared_leg(caps, facts))
        and not (caps.sm_lo == 107 and half and _selected_d_shape(caps, facts) == (256, 256) and not rubin_decode_tile and not rubin_paged_thd_d256)
        and not (facts.thd and not _thd_decode_leg(caps, facts) and (facts.d_qk, facts.d_v) not in caps.thd_pack_gqa_d_shapes)
        and not (facts.has_epilogue_gate and not rubin_decode_tile)
        and facts.h_q != facts.h_kv
        and (rubin_decode_tile or pack_gqa_supported(facts.h_q, facts.h_kv, tile_m, partial=pack_gqa_partial(caps, facts)))
    )


def _pack_gqa_group(caps: Capabilities, facts, tile_m: Optional[int], packed: Optional[bool], split_kv: Optional[int] = None) -> int:
    """The heads one packed Q tile row-group holds for a ``pack_gqa=packed``
    set: 1 unpacked, else ``Cfg.PACK_G`` -- the whole ratio G when it divides
    the tile, its largest divisor that does on a partial-PackGQA flavor (96/8
    -> 4).  The launch geometry the wave-cost model must see is the PACKED
    one: ``h_q // p`` packed heads of ``s_q * p`` rows each -- feeding it G
    where the kernel packs p (96/8: 8 heads instead of 24) shrinks the
    apparent grid 3x and over-proposes the split at mid batch sizes.

    When NO prefill tile can pack the group (the full contract at a ratio that
    does not divide the tile: the Rubin half row's 24/2, G = 12 at tile_m 128,
    no partial form) but the whole group rides the d256 decode tile, that
    tile's ``HEADS_PER_TILE = QH_PER_KH`` is the launch geometry -- the group is
    G, not 0 (which :func:`_decode_tile_pack_g` would read as unpacked and
    cost the launch at G times its unit count)."""
    if not packed:
        return 1
    g = facts.h_q // facts.h_kv
    p = pack_gqa_group_size(g, tile_m or 128, partial=pack_gqa_partial(caps, facts))
    if p == 0 and _d256_decode_tile_selected(caps, facts, g, split_kv):
        return g
    return p


def _sm107_paged_half(caps: Capabilities, facts) -> bool:
    """The cc 10.7 half row over paged K/V (the shared SM100 paged bodies compiled for sm_107a): the family whose
    THD plan ordering was measured with and without an attention sink (issue #1472's paged serving contract)."""
    return caps.sm_lo == 107 and facts.has_paged_kv and facts.dtype in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16)


def _prefer_thd_pack_gqa(caps: Capabilities, facts) -> bool:
    """The measured native-half THD causal family, separate from decode."""
    native_half = _sm100_f16(caps, facts) or _sm107_paged_half(caps, facts)
    return (
        native_half
        and (facts.d_qk, facts.d_v) == (128, 128)
        and facts.thd
        and not _thd_decode_leg(caps, facts)
        and facts.causal
        and not facts.has_epilogue_gate
        and (facts.d_qk, facts.d_v) in caps.thd_pack_gqa_d_shapes
        # Nonpaged SM100 also packs GQA16 (unpacked 1.04-2.58x the backend, packed 0.53-0.99) and GQA2
        # (packed matched or beat unpacked on all 72 measured cases, e.g. 1.58 -> 0.94). So does cc 10.7
        # paged THD with GQA16 (216 SMs, cuDNN 9.26 / 9.27, CUDA-graph replay, with or without a sink): unpacked
        # cga2 2.8 ms against the packed set's 0.22 ms at b128 64/4 q 1 / 4 / 8 KV 2k (page 16 and 128), 184 vs 43 us at b24 mixed.
        and facts.h_q // facts.h_kv
        in ((2, 4, 8, 16) if _sm100_f16(caps, facts) and not facts.has_paged_kv else (4, 8, 16) if _sm107_paged_half(caps, facts) else (4, 8))
    )


def _pack_gqa_points(caps: Capabilities, facts, tile_m: int, cga: Optional[int] = None, split_kv: Optional[int] = None) -> Tuple[bool, ...]:
    """The pack_gqa axis, best first: ``(True, False)`` when packing wins,
    ``(False, True)`` when it is only eligible, ``(False,)`` when it is not.
    ``split_kv`` is the set's split (None = some plan): a GATED d256 graph packs
    only on the decode tile's split, whose combine applies the gate."""
    if not _pack_gqa_eligible(caps, facts, tile_m, split_kv, cga):
        return (False,)
    if facts.thd and not _thd_decode_leg(caps, facts):
        # On the admitted d128 half prefill tile, packing shortens the token
        # span along the causal diagonal and shares KV across query heads.
        # Keep the default bounded to the measured GQA4/GQA8 family; other
        # supported groups remain explicit tuning candidates. This also
        # covers a long declared envelope replayed with short live lengths.
        return (True, False) if _prefer_thd_pack_gqa(caps, facts) else (False, True)
    tile_q = _pack_gqa_tile_q(caps, facts, tile_m, cga)
    if caps.sm_lo == 90 and caps.sm_hi == 90:
        # SM90 runs one CTA per (Q tile, head, batch) and each walks the whole KV
        # range, so cost is the CTA COUNT: pack when it needs fewer grid tiles.
        g = _pack_gqa_group(caps, facts, tile_m, True)
        wins = _ceil_div(facts.s_q * g, tile_q) < _ceil_div(facts.s_q, tile_q) * g
    else:
        wins = _pack_gqa_wins(facts, tile_q)
        if (
            wins
            and rubin_dense_d128_shared_leg(caps, facts)
            and not (facts.causal or facts.window_left is not None)
            and facts.s_q <= _D128_DECODE_TILE_ROWS
            and not _d128_decode_tile_fits(caps, facts, True)
        ):
            # Mask-free cc 10.7 dense d128 GQA whose UNPACKED unit fits the shared decode tile while the packed one
            # overflows it (S_q <= 128 < S_q x G): the unpacked decode tile leads -- MEASURED on the 216-SM cc 10.7
            # board (cuDNN 9.26.0.51, CUDA-graph replay, kernel time, 2026-10-08): b2 64/8 q128 KV 4k unpacked
            # decode tile 32.0 us against the packed cga2 body's 90.9 (split 2) / 98.1 (unsplit) and the backend's
            # 103.7; b2 64/8 q64 KV 4k 31.2 vs 31.9 (the packed body's split-4 arm) and b8 64/4 q128 KV 2k 97.2 vs
            # 97.3 at parity.  Packing keeps the lead where no unpacked tile fits (b2 64/8 q256 KV 4k: packed 93.9 vs
            # unpacked 191.2) and under a band (_sm100_banded_gqa_packs below).  The packed set stays listed.
            wins = False
    if wins or (_sm120_d512_windowed(caps, facts) and not (facts.is_fp8 or facts.is_mxfp8)) or _sm100_banded_gqa_packs(caps, facts):
        return (True, False)
    return (False, True)


def _sm100_f16(caps: Capabilities, facts) -> bool:
    """The Blackwell datacenter f16 rows below Rubin (the row's capability range,
    not the device: ``sdpa_fwd_prefill_sm100`` spans cc 10.0-10.6, so B200 and
    GB300 both take this geometry)."""
    return 100 <= caps.sm_lo and caps.sm_hi < 107 and facts.dtype in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16)


def _swa_kv_tiles(facts, *, token_span: int, tile_n: int) -> int:
    """Maximum issued KV range of this candidate's Q clusters under SWA.

    Mirrors the range, not the element mask: the last Q cluster retains its
    full token span, including padding, just as compute_kv_loop_bounds does.
    Distinct Q-start residues repeat after tile_n/gcd(token_span,tile_n)
    clusters. Within one residue, both unclipped bounds move by whole tiles;
    their clipped intersection is largest at one of the two integer points
    nearest its center. Thus planning never scans an unbounded Q sequence.
    """
    kv_tiles = _ceil_div(facts.s_kv, tile_n)
    if facts.window_left is None:
        return kv_tiles
    causal = facts.causal or facts.right_band_widening
    right = facts.right_bound if facts.right_band_widening else 0
    if facts.padded and facts.bottom_right:
        # Device KV lengths change the diagonal's tile residue. Bound every
        # possible alignment without reading a device value on the host.
        return min(kv_tiles, _ceil_div(token_span + facts.window_left + right + tile_n - 1, tile_n)) if causal else kv_tiles
    diagonal = facts.s_kv - facts.s_q if facts.bottom_right else 0
    if not causal:
        return max(0, kv_tiles - max(0, (diagonal - facts.window_left) // tile_n))
    q_tiles = _ceil_div(facts.s_q, token_span)
    period = tile_n // gcd(token_span, tile_n)
    step = period * token_span // tile_n
    longest = 0
    for residue in range(min(period, q_tiles)):
        lo = (residue * token_span + diagonal - facts.window_left) // tile_n
        hi = _ceil_div(residue * token_span + diagonal + token_span + right, tile_n)
        last = (q_tiles - 1 - residue) // period
        center = max(0, min(last, (kv_tiles - lo - hi) // (2 * step)))
        for index in (center, min(last, center + 1)):
            longest = max(longest, min(kv_tiles, hi + index * step) - max(0, lo + index * step))
    return longest


def _split_launch(caps: Capabilities, facts, tile_m, tile_n, cga, pack_g: int, *, physical: bool = True, split_kv: int = 1) -> _SplitKvLaunch:
    """The launch the wave-cost model sees. ``physical=False`` counts the MMA
    width per cluster (the count the model's constants were fitted with);
    ``physical=True`` counts every CTA the cluster launches (D512: 4 for 2)."""
    rows = _pack_gqa_tile_q(caps, facts, tile_m, cga, split_kv=split_kv)
    if (
        _sm100_f16(caps, facts)
        and not facts.thd
        and cga in (None, 1)
        and _selected_d_shape(caps, facts) == (64, 64)
        and facts.s_q * pack_g <= _D64_DECODE_TILE_ROWS
    ):
        # The d64 f16 decode tile (api_dsl._d64_decode_tile): one 128-row Q tile
        # per CTA once S_q x packed heads fit it, in place of the 256-row prefill.
        rows = _D64_DECODE_TILE_ROWS
    kv_tiles = _ceil_div(facts.s_kv, tile_n or 128)
    ctas = cga or 1
    if _sm100_f16(caps, facts):
        kv_tiles = _swa_kv_tiles(facts, token_span=rows // pack_g, tile_n=tile_n or 128)
        if physical:
            ctas = cga_ctas(_selected_d_shape(caps, facts)[0], cga)
    return _SplitKvLaunch(_ceil_div(facts.s_q * pack_g, rows), facts.h_q // pack_g, kv_tiles, ctas)


def _split_points(
    caps: Capabilities,
    facts,
    tile_m: Optional[int],
    tile_n: Optional[int],
    cga: Optional[int],
    pack_g: int = 1,
    *,
    unsplit_knobs: Optional[SdpaFwdKnobs] = None,
) -> List[Optional[int]]:
    """Ordered split-KV candidates for the chosen tile geometry.

    ``pack_g`` is the packed group of the set the split rides (1 = unpacked;
    :func:`_pack_gqa_group` -- the kernel's ``Cfg.PACK_G``, which is the GQA
    ratio G when it divides the tile and a proper divisor of it under partial
    PackGQA): packing multiplies each packed head's Q rows by ``pack_g`` and
    divides the head count by it, so the wave-cost model must see the PACKED
    launch — the packed grid is smaller, which is exactly when splitting pays.

    The value comes from :func:`choose_split_kv`'s wave-cost model, fed the
    EXACT launch geometry via :func:`_pack_gqa_tile_q` — the Q rows one grid
    tile covers, which on SM100 is the cluster's ``TILES_Q*TILE_M*CTA_MMA``.
    ``unsplit_knobs`` lets the no-split candidate carry a different complete
    geometry; D192 requires this because split-KV is CGA2-only while its tuned
    unsplit assignment may use CGA1. The generator respects structural limits
    (dense-only, no sink — mismatch() enforces the same, so an emitted >1
    never reaches a kernel that cannot honor it).

    A split the model asks for LEADS, with no-split behind it as the runner-up
    — so a plain ``build_plans()`` runs the split, and autotune / select_plan
    can still reach the unsplit plan. Emitting it the other way round meant the
    default build never used the split the model had just computed.
    """
    no_split = 1
    if not caps.split_kv_supported:
        return [no_split]
    # The decode tile's ragged-Q leg (cga=1 on a THD-over-paged-KV graph at
    # S_q(max) == 1): the combine places the ragged O / Stats rows, so the
    # split path is the ONLY path -- the leading entry is at least 2 and there
    # is no unsplit runner-up.  On the cga=2 prefill tile the same graph rides
    # the THD leg, which packs its own flat grid and cannot split.
    ragged_decode = facts.thd and cga == 1 and _thd_decode_leg(caps, facts)
    # split_kv_supported is a ROW-wide flag; split_d_shapes narrows it to the
    # flavors that actually wire SplitHelpers. mismatch() already honours that
    # for an explicitly REQUESTED split, but this function proposes one on its
    # own, so it has to consult the same set or it hands back a knob the
    # lowering will reject (d64 wires no split; see config_sm100.CfgD64).
    if caps.split_d_shapes is not None and _selected_d_shape(caps, facts) not in caps.split_d_shapes:
        return [no_split]
    # Paged KV is padded by construction and the split composes with the
    # per-batch lengths (it IS the decode lever there) — see mismatch().
    if (facts.thd and not ragged_decode) or facts.has_sink or (facts.padded and not facts.has_paged_kv) or facts.seq_q_trim:
        return [no_split]
    decode_pack_g = _decode_tile_pack_g(facts, pack_g)
    # A gated graph's split exists on the d256 decode tile only (its combine applies
    # the gate: engines.d256_decode_tile_selected's split_kv arm) -- asked for the
    # SPLIT form, since the tile's unsplit form never carries a gate.
    gated_decode_tile = facts.has_epilogue_gate and _d256_decode_tile_selected(caps, facts, decode_pack_g, 2)
    if facts.has_epilogue_gate and not gated_decode_tile:
        # The fused O * sigmoid(G) epilogue lives in the unsplit kernel; the
        # combine would write the un-gated O (mismatch declines the same pair,
        # so this is hygiene: never PROPOSE a knob the row cannot honour).
        return [no_split]
    if _synth_kv_padding(caps, facts):
        # D128 scalar-tail splitting is available explicitly. Keep automatic
        # selection conservative until this tail domain is independently tuned;
        # paged graphs continue to use their existing live-length split rule.
        return [no_split]
    # A quantized O is a legal split target: the partials stay WIDER than the
    # O dtype whatever it is, and the combine performs the only cast down to it.
    sm_count = facts.device_sm_count or 0
    if sm_count <= 0:
        # The ragged-Q leg has no unsplit form, nor has a gated graph on the decode
        # tile: keep a legal split without the SM count.
        return [2] if (ragged_decode or gated_decode_tile) else [no_split]
    if _d256_decode_tile_selected(caps, facts, decode_pack_g, 2):
        # The decode tile is a different machine from the one the prefill model
        # below was fitted on (one cta_group::1 CTA per (KV-head group, batch,
        # split) unit, HBM-bound, ~1 tile of fixed cost -- not cga2 clusters
        # with a 21-tile cost), so it has its own model.  The LEADING entry is
        # the choice that also pays for the split path's second host launch
        # (charged as if serialized with the GPU work -- SUPPORT_MATRIX_TRACKER.md
        # footnote d has the measured eager and replay numbers); the captured
        # caller's optimum, when it differs, is the runner-up -- the other way
        # round for a graph whose caller declared CUDA-graph replay
        # (facts.cuda_graph_replay: the second launch is paid once at capture);
        # no-split closes the list as usual.
        # The row's N extent (16, or the 32-column tile on cc 10.7) and the TOKEN UNITS it cuts a
        # (head group, batch) into: every unit is one more CTA streaming the KV range (the MTP step at
        # 24/2 = two units of two tokens), so the model sees them as streams; the KV range of a unit
        # spans its tokens (token_span = tokens per unit).
        decode_q_tile = decode_d256_q_tile_for_row(caps, facts.s_q, decode_pack_g)
        tok_units = decode_d256_q_units(facts.s_q, decode_pack_g, decode_q_tile)
        geometry = dict(
            units=facts.b * (facts.h_q // decode_pack_g) * tok_units,
            kv_tiles=_swa_kv_tiles(facts, token_span=decode_q_tile // decode_pack_g, tile_n=tile_n or 128),
            sm_count=sm_count,
            q_tile=decode_q_tile,
        )
        eager = choose_decode_tile_split_kv(**geometry)
        captured = choose_decode_tile_split_kv(**geometry, launch_cost=0.0)
        if gated_decode_tile:
            # The gate rides the split COMBINE: the tile's unsplit form cannot carry
            # it, so the model's "do not split" becomes the smallest split (its
            # combine is the gate's only home on the tile).  The unsplit entry that
            # closes the list is the d256 PREFILL kernel's fused epilogue -- a
            # different kernel, admissible on a dense cache only (the paged prefill
            # kernel has no gate), so a paged gated graph lists splits alone.
            eager, captured = max(eager, 2), max(captured, 2)
        lead, runner = (captured, eager) if facts.cuda_graph_replay else (eager, captured)
        points = [lead]
        closers = (runner,) if (gated_decode_tile and facts.has_paged_kv) else (runner, no_split)
        for split in closers:
            if split not in points:
                points.append(split)
        return points
    unsplit_pack_g = None
    if unsplit_knobs is not None:
        unsplit_pack_g = _pack_gqa_group(caps, facts, unsplit_knobs.tile_m, unsplit_knobs.pack_gqa)

    def _choose(physical: bool) -> int:
        split_launch = _split_launch(caps, facts, tile_m, tile_n, cga, pack_g, physical=physical, split_kv=2)
        unsplit_launch = None
        if unsplit_knobs is not None:
            unsplit_launch = _split_launch(caps, facts, unsplit_knobs.tile_m, unsplit_knobs.tile_n, unsplit_knobs.cga, unsplit_pack_g, physical=physical)
        return choose_split_kv(
            q_tiles=split_launch.q_tiles,
            heads_q=split_launch.heads_q,
            batch=facts.b,
            kv_tiles=split_launch.kv_tiles,
            sm_count=sm_count,
            # The combine's grid is (S_q, H, B) — the REAL head count, not the
            # packed one: packing folds heads into Q rows for the main kernel, but
            # the combine still reduces one block per (row, head, batch) of the
            # graph's own output.
            combine_rows=facts.s_q * facts.h_q * facts.b,
            ctas_per_tile=split_launch.ctas_per_tile,
            unsplit_launch=unsplit_launch,
            **(
                dict(min_tiles=_SM120_SPLIT_KV_MIN_TILES, combine_floor=_SM120_SPLIT_KV_COMBINE_FLOOR)
                if caps.sm_lo == 120 and not (caps.is_fp8 or caps.is_mxfp8)
                else {}
            ),
        )

    split = _choose(physical=True)
    if _sm100_f16(caps, facts) and cga_ctas(_selected_d_shape(caps, facts)[0], cga) != (cga or 1):
        # The physical CTA count is a one-sided correction: every measured win of
        # counting D512's role CTAs is a LOWER split (decode-shaped D512: 64 -> 32,
        # 8 -> 4 on B200); the regime where it asks for MORE splits than the
        # MMA-width count the constants were fitted with (D512, S_q <= 128, long
        # KV) is unmeasured and timed slower on an SM100 board, so keep the
        # fitted answer there.
        split = min(split, _choose(physical=False))
    if ragged_decode:
        # No unsplit runner-up: the ragged final rows exist only through the combine.
        return [max(split, 2)]
    if split <= 1:
        return [no_split]
    return [split, no_split]


# NOTE: the softmax accumulator precision is NOT a knob axis. It changes
# numerics (the Rubin f16x2 exponent arm), so it is the
# sdpa(softmax_precision=) op attribute: a graph FACT that engines.mismatch
# gates against Capabilities.softmax_precisions. Heuristics never propose it —
# auto-proposing it was the CUDNN_SOFTMAX_PRECISION environment-knob failure
# mode this vocabulary exists to avoid.


# ---------------------------------------------------------------------------
# the combiner — complete assignments, Σ growth, never the cartesian product
# ---------------------------------------------------------------------------


def paged_d256_prefix_launch(caps: Capabilities, facts) -> Optional[_SplitKvLaunch]:
    """Qualified two-CTA prefix geometry, shared by placement and split choice."""
    if not (
        caps.sm_lo == 107
        and (facts.d_qk, facts.d_v) == (256, 256)
        and paged_thd_split_domain(caps, facts)
        and getattr(cudnn._pybind_module._SdpaThdBinder, "supports_paged_d256_packed_split", False)
        and facts.dtype in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16)
        and facts.b >= 1
        and 4 <= facts.h_q <= 64
        and facts.h_kv > 0
        and facts.h_q % facts.h_kv == 0
        and facts.h_q // facts.h_kv in (1, 2, 4, 8, 16)
        and facts.page_size in (16, 128)
        and facts.causal
        and facts.bottom_right
        and not facts.right_band_widening
        and facts.window_left is None
        and 64 <= facts.s_q <= 1024
        and 2048 <= facts.s_kv <= 32768
        and facts.s_kv >= 4 * facts.s_q
        and facts.device_sm_count
    ):
        return None
    launch = _split_launch(caps, facts, 128, 128, 2, 1, split_kv=2)
    units = facts.b * launch.heads_q * launch.q_tiles
    resident = facts.device_sm_count // launch.ctas_per_tile
    return launch if units <= resident else None


def _paged_d256_thd_split_choice(caps: Capabilities, facts) -> int:
    launch = paged_d256_prefix_launch(caps, facts)
    if launch is None:
        return 1
    units = facts.b * launch.heads_q * launch.q_tiles
    resident = facts.device_sm_count // launch.ctas_per_tile
    # Budget four KV tiles per partition on average, bounding partial traffic.
    # A declared envelope is conservative: never read live lengths to refine it.
    budget = min(16, max(1, resident // units), max(1, launch.kv_tiles // 4))
    return _ceil_div(launch.kv_tiles, _ceil_div(launch.kv_tiles, budget))


def _d128_thd_split_units(facts, pack_g: int) -> int:
    """Bound actual token tiles, including a partial tile per live sequence."""
    token_rows = 128 // pack_g
    tiles = facts.b * _ceil_div(facts.s_q, token_rows)
    total = facts.max_total_seq_len_q
    if total is not None:
        # Each nonempty sequence buys its first tile with one token; every
        # additional tile needs token_rows more. This uses declared capacity,
        # never runtime device lengths, and remains valid for any ragged split.
        nonempty = min(facts.b, max(0, total))
        tiles = min(tiles, nonempty + max(0, total - nonempty) // token_rows)
    return tiles * (facts.h_q // pack_g)


def _d128_thd_split_wave_choice(facts, *, extra_waves: int = 1) -> Tuple[int, bool]:
    """Score the physical packed/unpacked grids, preserving first-wave wins."""
    kv_tiles = _ceil_div(facts.s_kv, 128)
    sm_count = facts.device_sm_count or 128
    for waves in ((1, extra_waves) if extra_waves > 1 else (1,)):
        choices = []
        for pack in ((False, True) if facts.h_q != facts.h_kv else (False,)):
            group = facts.h_q // facts.h_kv if pack else 1
            units = _d128_thd_split_units(facts, group)
            if not units:
                continue
            budget = min(16, waves * sm_count // units, kv_tiles // 4)
            for splits in range(2, budget + 1):
                work = _ceil_div(units * splits, sm_count) * _ceil_div(kv_tiles, splits)
                choices.append((work, splits, pack))
        if choices:
            _, splits, pack = min(choices)
            return splits, pack
    return 1, False


def paged_thd_split_choice(caps: Capabilities, facts) -> Tuple[int, bool]:
    """Measured bounded-graph (split count, packing); one keeps the existing plan.

    Include batch in the grid estimate so multi-request chunks do not receive
    the split budget of an underfilled single request. Rubin qualification
    retains the first-wave budget for larger batches and multi-tile queries.
    Small-batch one-tile prefixes can use up to three waves when no first-wave
    split fits. Blackwell also admits
    GQA16 and KV lengths through 32K with the same physical-grid score and
    bounded partial workspace, excluding splits with more waves than partitions.
    """
    if (facts.d_qk, facts.d_v) == (256, 256):
        return _paged_d256_thd_split_choice(caps, facts), False
    if not (
        paged_thd_split_domain(caps, facts)
        and (facts.d_qk, facts.d_v) == (128, 128)
        and getattr(cudnn._pybind_module._SdpaThdBinder, "supports_paged_packed_split", False)
        and facts.dtype == cudnn.data_type.BFLOAT16
        and 1 <= facts.b <= (64 if caps.sm_lo == 107 else 4)
        and 4 <= facts.h_q <= 64
        and facts.h_kv > 0
        and facts.h_q % facts.h_kv == 0
        and facts.h_q // facts.h_kv in ((1, 2, 4, 8, 16) if caps.sm_lo == 100 else (1, 2, 4, 8))
        and facts.page_size == 16
        and facts.causal
        and facts.bottom_right
        and facts.window_left is None
        and 64 <= facts.s_q <= 1024
        and 2048 <= facts.s_kv <= 32768
        and facts.k_t is not None
        and facts.k_t.get_stride()[2] < facts.k_t.get_stride()[1]
    ):
        return 1, False
    if caps.sm_lo == 100:
        # Packed tiles can remove a partial-wave tail. Preserve first-wave
        # wins; otherwise compare up to three waves. Short loops stay on
        # their first-wave policy to amortize setup and combine.
        splits, pack = _d128_thd_split_wave_choice(facts, extra_waves=3 if facts.s_kv >= 4096 else 1)
        if splits > 1 and (facts.s_kv > 16384 or facts.h_q // facts.h_kv == 16):
            group = facts.h_q // facts.h_kv if pack else 1
            waves = _ceil_div(_d128_thd_split_units(facts, group) * splits, facts.device_sm_count or 128)
            # The newly admitted family must not stretch a split over more
            # waves than partitions. Keep useful three-wave/four-way splits,
            # but avoid three-wave/two-way tails on smaller GPUs.
            if waves > splits:
                return 1, False
        return splits, pack
    # Retain Rubin's first-wave budget, using the same declared packed-token
    # capacity bound as Blackwell. Never read runtime sequence lengths here.
    kv_tiles = _ceil_div(facts.s_kv, 128)
    sm_count = facts.device_sm_count or 128
    choices = []
    for pack in (False, True):
        group = facts.h_q // facts.h_kv if pack else 1
        units = _d128_thd_split_units(facts, group)
        if not units:
            continue
        # Keep four KV tiles per partition to amortize setup/combine.
        budget = min(16, max(1, sm_count // units), max(1, kv_tiles // 4))
        loop_tiles = _ceil_div(kv_tiles, budget)
        splits = _ceil_div(kv_tiles, loop_tiles)
        if splits > 1:
            work = _ceil_div(units * splits, sm_count) * loop_tiles
            # Equal loop work prefers fewer partials, then fewer physical CTAs.
            choices.append((work, splits, units, pack))
    if choices:
        _, splits, _, pack = min(choices)
        return splits, pack
    if caps.sm_lo == 107 and facts.b <= 4 and facts.s_q <= 128 and facts.s_kv >= 8192:
        # Preserve first-wave choices. One-tile queries can use more waves;
        # longer queries can already use an efficient backend prefill tile.
        return _d128_thd_split_wave_choice(facts, extra_waves=3)
    return 1, False


def nonpaged_thd_split_choice(caps: Capabilities, facts) -> Tuple[int, bool]:
    """(split count, packing) for the underfilled nonpaged tile; one keeps the old choice.

    B200 / released cuDNN 9.26, BF16 THD, Hq=Hkv, Q64..1024/KV2K..32K:
    the smaller tile plus splitting beats the wide unsplit tile and backend
    while the launch is underfilled. Bounded overrides use their declared
    envelope; full prefill and other graph features keep their existing policy.
    Bottom-right prefixes are at least three quarters KV, so the unmasked loop
    bounds their work closely. B200 / released cuDNN 9.27 also qualifies exact
    D128 FP16/BF16 with integral GQA1..16 on Blackwell, fixed or bounded, with
    or without packed Stats; the same first-wave budget avoids splitting
    already-filled/full-prefill grids. Rubin reuses this budget for its native
    packed D128 and MLA paths, with the device's actual SM count. Blackwell and
    Rubin D128 count the CTAs of the actual candidate: a packed CTA holds
    128 / (H_q/H_kv) tokens of one KV head's group, and a declared
    ``max_total_seq_len_q`` bounds ragged batches. It also admits Q8..63, where
    the backend ran 2-16x slower than the split, and may fill a second wave
    when that shortens waves x KV loop.
    """
    d128 = (facts.d_qk, facts.d_v) == (128, 128)
    if d128:
        # Reuse the MLA launch budget for native half ragged prefixes.
        if caps.sm_lo not in (100, 107) or facts.h_kv <= 0 or facts.h_q % facts.h_kv or facts.h_q // facts.h_kv not in (1, 2, 4, 8, 16):
            return 1, False
    elif (facts.d_qk, facts.d_v) != (192, 128) or facts.h_q != facts.h_kv:
        return 1, False
    counted_d128 = d128 and caps.sm_lo in (100, 107)
    if not (
        thd_split_domain(caps, facts)
        and not facts.has_paged_kv
        and getattr(cudnn._pybind_module._SdpaThdBinder, "supports_nonpaged_d128_packed_split" if d128 else "supports_nonpaged_packed_split", False)
        and (facts.dtype == cudnn.data_type.BFLOAT16 or (d128 and facts.dtype == cudnn.data_type.HALF))
        and 1 <= facts.b <= 4
        and 4 <= facts.h_q <= 64
        and (8 if counted_d128 else 64) <= facts.s_q <= 1024
        and (_SM107_THD_SPLIT_MIN_KV if counted_d128 and caps.sm_lo == 107 else 2048) <= facts.s_kv <= 32768
        and (facts.s_kv >= 2048 or _SM107_THD_SPLIT_SHORT_KV_RATIO * facts.s_q <= facts.s_kv)
        and 4 * facts.s_q <= facts.s_kv
        and (not facts.causal or facts.bottom_right)
        and not facts.right_band_widening
        and facts.window_left is None
        and facts.device_sm_count
    ):
        return 1, False
    # Do not overfill the wave budget (Blackwell/Rubin D128 two, else one): beyond it the extra
    # partials/combine usually cost more than the shorter loop saves. Four KV
    # tiles per partition amortize that overhead. Reuse the power-of-two
    # specialization set; selection uses host graph facts only, never live
    # device lengths.
    kv_tiles = _ceil_div(facts.s_kv, 128)
    group = facts.h_q // facts.h_kv
    budget = facts.device_sm_count * (2 if counted_d128 else 1)
    choices = []
    for pack in (False, True) if counted_d128 and group > 1 else (False,):
        rows = 128 // group if pack else 128
        q_tiles = facts.b * _ceil_div(facts.s_q, rows)
        if counted_d128 and facts.max_total_seq_len_q:
            # Each sequence leaves at most one partial tile.
            q_tiles = min(q_tiles, (facts.max_total_seq_len_q + facts.b * (rows - 1)) // rows)
        units = q_tiles * (facts.h_q // group if pack else facts.h_q)
        for s in split_kv_candidates(sm_count=facts.device_sm_count, kv_tiles=kv_tiles):
            if counted_d128 and caps.sm_lo == 107 and s > 16:
                break  # Rubin D128 was measured at splits 2-16
            if s > 1 and units * s <= budget and kv_tiles // s >= 4:
                # Fewest waves x loop, then fewer partials, then unpacked (packed on Rubin).
                prefer = not pack if caps.sm_lo == 107 else pack
                choices.append((_ceil_div(units * s, facts.device_sm_count) * _ceil_div(kv_tiles, s), s, prefer, pack))
    if not choices:
        return 1, False
    _, splits, _, pack = min(choices)
    return splits, pack


def _knob_sets(spec: EngineSpec, facts) -> List[SdpaFwdKnobs]:
    """The cell's ordered COMPLETE knob assignments.

    The baseline takes the best value on every axis; runners-up deviate on ONE
    geometry at a time (tiles, sched, pack_gqa, split), recomputing split for
    SM100 f16 geometry runners, capped at ``_MAX_SETS_PER_ENGINE``. Two
    axes carry a structural coupling: a packed set rides the largest tile
    that admits the ratio, and a split set rides the plain scheduler. Axis
    interactions the kernels cannot serve are the generators'/mismatch's job
    — nothing here multiplies domains together.
    """
    caps = spec.capabilities
    tiles = _tile_points(spec, facts)
    generic_scheds = _sched_points(caps, facts)
    unpacked_pack = False if True in caps.pack_gqas else _sole(caps.pack_gqas)
    # A split set rides the plain scheduler: the SM120 config bars a split under
    # the LPT remaps, and in the underfilled regime a split targets, LPT
    # balancing is moot — the split itself levels the grid. The coupling is
    # structural, so it binds whichever leg leads; it cannot live only on the
    # runner-up loop or a leading split would inherit the derived LPT policy.
    plain_sched = SCHED_NATURAL if SCHED_NATURAL in caps.sched_policies else generic_scheds[0]

    unsplit_sched, _ = _auto_sched_cga(spec, facts, split_kv=1, sched_policy=generic_scheds[0])
    scheds = [unsplit_sched] + [policy for policy in generic_scheds if policy != unsplit_sched]

    def _pack_choice(cga: Optional[int], split_value: Optional[int] = None):
        # A packed set rides the largest fitting tile that admits the ratio, on
        # the rule's KV tile where it can. The
        # decision uses this leg's actual CGA span; D192 CGA1 and CGA2 cover a
        # different number of Q rows.  The leg's split is part of the question on
        # a GATED d256 graph: it packs on the decode tile's split only (the gate
        # rides that split's combine), never on the unsplit fused-gate kernel.
        pack_tile = next(
            (
                (m, n)
                for m, n in sorted(tiles, key=lambda mn: (-(mn[0] or 0), mn[1] != tiles[0][1]))
                if True in _pack_gqa_points(caps, facts, m or 128, cga, split_value)
            ),
            None,
        )
        packed_first = pack_tile is not None and _pack_gqa_points(caps, facts, pack_tile[0] or 128, cga, split_value)[0]
        return (pack_tile if packed_first else tiles[0]), packed_first, pack_tile

    def _leg(split_value: int) -> SdpaFwdKnobs:
        seed_sched = plain_sched if split_value > 1 else scheds[0]
        sched_policy, cga = _auto_sched_cga(spec, facts, split_kv=split_value, sched_policy=seed_sched)
        base_tile, packed_first, _ = _pack_choice(cga, split_value)
        pack_gqa = True if packed_first else unpacked_pack
        if pack_gqa is not True:
            # The width above was judged on the packed leg's rows (the graph-
            # level question); a leg that runs unpacked is re-judged on one
            # head's S_q rows -- the d128 decode-tile fit belongs to the
            # candidate, not the graph.  _pack_choice does not move under the
            # new width: it depends on cga only through _pack_gqa_wins, which
            # is settled the same way at either width for any S_q this can
            # change (an eligible group that does not win has S_q >= 512).
            sched_policy, cga = _auto_sched_cga(spec, facts, split_kv=split_value, sched_policy=seed_sched, pack_gqa=False)
        return SdpaFwdKnobs(
            sched_policy=sched_policy,
            tile_m=base_tile[0],
            tile_n=base_tile[1],
            cga=cga,
            pack_gqa=pack_gqa,
            split_kv=split_value,
        )

    unsplit_leg = _leg(1)
    split_leg = _leg(2)
    split_pack_g = _pack_gqa_group(caps, facts, split_leg.tile_m, split_leg.pack_gqa, split_kv=2)
    splits = _split_points(
        caps,
        facts,
        split_leg.tile_m,
        split_leg.tile_n,
        split_leg.cga,
        pack_g=split_pack_g,
        unsplit_knobs=unsplit_leg,
    )

    def _resplit(knobs: SdpaFwdKnobs) -> SdpaFwdKnobs:
        # A runner's changed packing/tile/CGA is a different launch, not just
        # a label on the baseline's split. Keep other architecture policies
        # unchanged until they have their own geometry validation.
        if not _sm100_f16(caps, facts):
            return knobs

        def leg(split):
            policy, auto_cga = _auto_sched_cga(spec, facts, split_kv=split, sched_policy=plain_sched if split > 1 else scheds[0], pack_gqa=knobs.pack_gqa)
            cga = knobs.cga if knobs.cga in effective_cgas(caps, facts, split) else auto_cga
            return replace(knobs, sched_policy=policy, cga=cga, split_kv=split)

        unsplit, split = leg(1), leg(2)
        points = _split_points(
            caps,
            facts,
            split.tile_m,
            split.tile_n,
            split.cga,
            pack_g=_pack_gqa_group(caps, facts, split.tile_m, split.pack_gqa),
            unsplit_knobs=unsplit,
        )
        return leg(points[0])

    base = _leg(splits[0])
    out = [base]
    for tile_m, tile_n in tiles[1:]:
        # A packed baseline's tile runners keep the packing, so tiles the
        # ratio cannot ride are skipped — emitted, they would only spend
        # set-cap slots for mismatch to drop.
        if base.pack_gqa is True and True not in _pack_gqa_points(caps, facts, tile_m or 128, base.cga, base.split_kv):
            continue
        out.append(_resplit(replace(base, tile_m=tile_m, tile_n=tile_n)))
    # No CGA-width runner here: the width follows each leg's own rows (_auto_sched_cga); the dense d128 other-width
    # runner is added below.
    # Scheduler runners ride an UNSPLIT leg: a split set is pinned to the plain
    # scheduler above, so an LPT runner is only a candidate without one.
    sched_host = unsplit_leg
    for policy in scheds[1:]:
        out.append(replace(sched_host, sched_policy=policy))
    # The opposite pack_gqa leg, riding its own tile (packed: the largest
    # admitting tile; unpacked: the tile rule's best) and its own CGA width:
    # on the d128 f16 SM100 flavor a packed leg that overflows one 128-row
    # tile (S_q * G > 128 >= S_q) keeps the prefill tile while its unpacked
    # runner-up fits the decode tile.  The other flavors' width rules do not
    # read the packing, so this returns base.cga there.
    _, _, pack_tile = _pack_choice(base.cga, base.split_kv)
    if pack_tile is not None:
        if base.pack_gqa is True:
            _, unpacked_cga = _auto_sched_cga(spec, facts, split_kv=base.split_kv, sched_policy=base.sched_policy, pack_gqa=False)
            out.append(_resplit(replace(base, pack_gqa=unpacked_pack, tile_m=tiles[0][0], tile_n=tiles[0][1], cga=unpacked_cga)))
        else:
            _, packed_cga = _auto_sched_cga(spec, facts, split_kv=base.split_kv, sched_policy=base.sched_policy, pack_gqa=True)
            out.append(_resplit(replace(base, pack_gqa=True, tile_m=pack_tile[0], tile_n=pack_tile[1], cga=packed_cga)))
    for split in splits[1:]:
        out.append(_leg(split))
    if _d128_dense_two_widths(caps, facts) and not _d128_decode_tile_fits(caps, facts, base.pack_gqa):
        # Past one decode tile both widths serve a dense d128 graph; the wave rule picks the lead.
        out.insert(1, _resplit(replace(base, cga=3 - base.cga)))
    seen, unique = set(), []
    for knobs in out:
        if knobs not in seen:
            seen.add(knobs)
            unique.append(knobs)
    splits, packed = paged_thd_split_choice(caps, facts)
    if splits > 1:
        unique.insert(
            0,
            replace(
                base,
                cga=_sole(effective_cgas(caps, facts, splits)),
                pack_gqa=packed,
                split_kv=splits,
                sched_policy=SCHED_NATURAL if caps.sm_lo == 107 else SCHED_LPT,
            ),
        )
    nonpaged_splits, nonpaged_packed = nonpaged_thd_split_choice(caps, facts)
    if nonpaged_splits > 1:
        unique.insert(
            0, replace(base, cga=1, pack_gqa=nonpaged_packed, split_kv=nonpaged_splits, sched_policy=SCHED_NATURAL if caps.sm_lo == 107 else SCHED_LPT)
        )
    return unique[:_MAX_SETS_PER_ENGINE]


def _fallback_knobs(spec: EngineSpec, facts) -> SdpaFwdKnobs:
    """The config expected to build where the tuned choice may not.

    Today this is the smallest tile the row admits with the plain scheduler
    and no split — the config that asks least of the device, which is the one
    thing a fallback must be.
    """
    caps = spec.capabilities
    sched_policy = SCHED_NATURAL if SCHED_NATURAL in caps.sched_policies else _sole(caps.sched_policies)
    pack_gqa = False if False in caps.pack_gqas else _sole(caps.pack_gqas)
    # An unpacked fallback is judged on one head's rows (the d128 decode tile
    # is also the least-demanding width: one CTA, no cluster); a row that only
    # packs leaves the graph-level question to the rule.
    sched_policy, cga = _auto_sched_cga(spec, facts, split_kv=1, sched_policy=sched_policy, pack_gqa=False if pack_gqa is False else None)
    return SdpaFwdKnobs(
        sched_policy=sched_policy,
        tile_m=min(caps.tile_ms, default=None),
        tile_n=min(caps.tile_ns, default=None),
        cga=cga,
        pack_gqa=pack_gqa,
        split_kv=1,  # the fallback never splits: least-demanding means one kernel, no partial workspace
    )


def _eligible(facts, offered: Dict[str, int]) -> Iterator[Tuple[int, EngineSpec]]:
    """(engine_id, spec) for each offered cell whose capability row admits ``facts``."""
    for spec in ENGINE_SPECS:
        engine_id = offered.get(spec.name)
        if engine_id is not None and mismatch(spec.capabilities, facts, None) is None:
            yield engine_id, spec


# ---------------------------------------------------------------------------
# recommend — the pure, backend-blind core (also the standalone entry point)
# ---------------------------------------------------------------------------


def recommend(kind: str, facts, offered: Dict[str, int]) -> List[PlanConfig]:
    """Ordered candidate plans for ``facts`` — no backend, no graph, no modes.

    ``kind`` is ``"A"`` (candidates worth timing, best guess first) or
    ``"FALLBACK"`` (least-demanding configs). Every returned entry carries a
    complete knob assignment validated through ``mismatch(caps, facts, knobs)``
    — honored-or-never-listed — and NO mode. Standalone callers (wrappers,
    autotuners) use this directly: build a ``SdpaGraphFacts``, pass the
    family's ``offered_ids()``, run or time the sets in order.
    """
    out: List[PlanConfig] = []
    for engine_id, spec in _eligible(facts, offered):
        caps = spec.capabilities
        sets = _knob_sets(spec, facts) if kind == "A" else [_fallback_knobs(spec, facts)]
        for knobs in sets:
            if mismatch(caps, facts, knobs) is not None:
                if knobs.split_kv != 1 or launch_grid_mismatch(caps, facts, knobs) is None:
                    continue
                # Past 65535 heads or batch the NATURAL grid cannot launch; the flat LPT grid serves the same set (#1573).
                knobs = next((lpt for lpt in (replace(knobs, sched_policy=p) for p in (SCHED_LPT, SCHED_LPT_L2)) if mismatch(caps, facts, lpt) is None), None)
            if knobs is not None and PlanConfig(engine_id, knobs) not in out:
                out.append(PlanConfig(engine_id, knobs))
    return out


def propose(kind: str, facts, offered: Dict[str, int]) -> List[PlanConfig]:
    """The manifest hook: :func:`recommend`'s proposals with the backend's block
    placed per the measured shard (:func:`placement.place`) — ours first where
    FROST is timed ahead of the backend, the backend's block first where it is
    not. ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES`` set means the caller asked for
    FROST (the ``cudnn_oss`` benchmark lane, the FROST suites): ours lead
    everywhere. An empty proposal list stays empty — nothing to place."""
    ours = recommend(kind, facts, offered)
    if not ours:
        return ours
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.engines.heuristics import BACKEND
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.engines.manifest import opt_in_engines_enabled

    from .placement import LEAD, place

    if opt_in_engines_enabled():
        return ours + [BACKEND]
    spec = next(s for s in ENGINE_SPECS if offered.get(s.name) == ours[0].engine_id)
    return ours + [BACKEND] if place(spec, facts) == LEAD else [BACKEND] + ours


# Mode blocks, the delegating entry, dedup, the mode strip: placement bookkeeping
# that happens once for every family in ``engines/heuristics._assemble``.
