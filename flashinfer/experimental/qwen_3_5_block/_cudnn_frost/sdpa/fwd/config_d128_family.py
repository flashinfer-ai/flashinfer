# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""The arch-NEUTRAL half of the f16/bf16 d128-FILE prefill configuration: the small-tile seams the Rubin and Blackwell
d128-file kernels share (``sm107/prefill_d128_f16.py``, ``sm100/prefill_d128_f16.py``) -- the additive bias / kv_mask arms,
the (n, n+1) pair item, the streamed ring, the row-max forms, the padded tail and dead-sub-tile forms, the double-buffered S,
Q in TMEM, the third softmax warp group -- as ONE field block, ONE TMEM map formula, ONE set of budgets and ONE set of
validators, each taking the arch's BUDGET (:class:`D128FamilyArch`) as a parameter.

Why a shared module and not a widening of either config: ``config_sm107`` and ``config_sm100`` are different kernel
LINEAGES (their module docstrings say why sharing their geometry is dangerous), but the small-tile seams are the SAME
body feature on both archs behind the same ``const_expr`` Cfg fields, and what differs between the archs is exactly three
numbers -- the usable SMEM carveout, the TMEM column total and whether the TMEM allocation needs ``is_exclusive`` -- plus
the K / V ring depth and the SM count the picks read.  A resident / ring budget is ONE arithmetic per arch BUDGET, never a
Rubin threshold carried to SM100: at 227 KiB the D 64 @128 cga1 forms hold NO bias at STAGES_KV 4 while Rubin holds 128 /
384 columns there.  Every function below takes the arch (or the depth) explicitly; nothing reads a module-level arch constant.

Each arch config keeps its OWN sweep seams (``_STAGES_KV_OVERRIDE_FOR_SWEEPS`` and friends are module globals the tests
``monkeypatch.setattr`` on that module) and its own measured pick tables; ``config_sm107`` re-exports every lifted name
under its old spelling so the kernels, the heuristics, the engine rows, the adapter and the tests import nothing new.

All configuration errors raise :class:`ValueError` whose message names the failure's SIGNATURE (sdpa-invariants s 6,
frost-kernels s 5 "bind the couplings"); reaching one from a user-built graph means an eligibility gate has a gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.constants import DTYPE_BF16 as _DTYPE_BF16, DTYPE_E5M2 as _DTYPE_E5M2, DTYPE_FP16 as _DTYPE_FP16, MASK_PADDED, SCHED_NATURAL

# The per-CTA register POOL a `setmaxnreg` split redistributes (the launch's entry count x the warps, NOT the 2048-register file): the
# backward config's helper, the ONE arithmetic behind the d128-file predicate `SUM(role regs x warps) <= pool` (frost-kernels.md s2: a
# split over the pool parks the last INCREASE warp forever -- a hang at 100 % SM utilization at every shape, no fault).
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.bwd.config_sm107 import reg_entry_pool as _reg_entry_pool

__all__ = [
    "D128FamilyArch",
    "D128FamilySmallTileFields",
    "D128FamilyTmemLayout",
    "d128_family_tmem_layout",
    "BIAS_MODE_NONE",
    "BIAS_MODE_RESIDENT",
    "BIAS_MODE_STREAMED",
    "BIAS_MODE_STREAMED_MULTICAST",
    "BIAS_MODES",
    "BIAS_STREAMED_MODES",
    "BIAS_MCAST_CTAS",
    "BIAS_RING_STAGES",
    "KV_MASK_NONE",
    "KV_MASK_BOOL_WORD",
    "KV_MASK_ADDITIVE_16",
    "KV_MASK_ADDITIVE_F32",
    "KV_MASK_FORMS",
    "BIAS_MAX_SOFTWARE",
    "BIAS_MAX_HW_UPPER_BOUND",
    "BIAS_MAX_FORMS",
    "KV_MASK_ADDITIVE_GRANULE_ELEMS",
    "BIAS_SWZ_BYTES",
    "BIAS_PRELOAD_CHUNKS_DOMAIN",
    "PADDED_TAIL_FORMS",
    "DEAD_SUBTILE_QUIET_FORMS",
    "SMALL_D_STAGES_KV",
    "SMALL_D_TILE_MS",
    "SMALL_D_SOFTMAX_WGS",
    "SMALL_D_S_SLOTS",
    "SMALL_D_Q_IN_TMEM",
    "TCGEN05_V0_ADDR_LIMIT",
    "sdpa_smem_bytes",
    "q_swz_bytes",
    "v_swz_bytes",
    "o_swz_bytes",
    "bias_block_rows",
    "bias_pair_form",
    "bias_resident_tiles",
    "bias_ring_stages",
    "operand_region_end_bytes",
    "bias_smem_tiles",
    "bias_resident_block_bytes",
    "d128_total_warps",
    "d128_read_tile_arrivers",
    "D128_REG_SPLIT",
    "validate_regs",
    "validate_warp_layout",
    "validate_swizzles",
    "validate_smem",
    "validate_head_dim_tile",
    "validate_tile_n",
    "validate_bias",
    "validate_q_in_tmem",
    "validate_softmax_wgs",
    "validate_small_tile_masks",
]


# ---------------------------------------------------------------------------
# The arch budget: the three numbers (and one flag) the small-tile seams depend on
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class D128FamilyArch:
    """The per-arch BUDGET every shared budget / validator takes as a parameter.

    ``smem_usable_bytes``: the carveout the SMEM predicate checks against (Rubin: 320 KiB = the 327 KiB oversized cap minus
    reserves -- the d192 overflow that read 50 % zeros with an exact LSE sits behind the number; SM100: the 227 KiB opt-in
    cap, no oversized mode).  ``smem_fixed_overhead``: barriers + the scheduler ring + the TMEM pointer slack (2 KiB on both:
    480 B measured plus 16 B per bias barrier row).  ``tmem_total_cols``: 576 on Rubin (``tmem_alloc(..., is_exclusive=True)``),
    512 on SM100 (NO ``is_exclusive`` -- the 4.7.0 floor rejects the kwarg; a 512-column allocation must never trace it).
    ``name`` is what the messages print ("Rubin" / "SM100")."""

    name: str
    smem_usable_bytes: int
    smem_fixed_overhead: int
    tmem_total_cols: int
    tmem_exclusive_alloc: bool


# ---------------------------------------------------------------------------
# The small-tile vocabulary (every d128-file body folds on these; one definition for both archs)
# ---------------------------------------------------------------------------

# ``TemplateParams.bias_mode`` / ``kv_mask_form`` (config_sm100): the f16/bf16 d128 FILE carries the graph's additive bias as a
# RESIDENT SMEM row-block -- the TMA-LDG warp fills ``BIAS_BLOCK_ROWS = TILES_Q * TILE_M`` bias rows x the work item's whole KV
# range once per item, the softmax warps read their row slice with 128-B-swizzled lane loads and add it pre-scale (``s + bias *
# inv_scale``, one FFMA2 per pair) before the band mask and the software row max -- and the per-key boolean ``kv_mask`` as one
# 32-bit keep word per 32 kv columns (a warp ballot over byte loads), folded into the bit-word mask.  The block is sized from what
# the carveout leaves after Q / O / K / V at the flavor's ring depth, in whole TILE_N-column tiles; a graph whose KV range needs
# more tiles than fit takes the STREAMED form (``bias_mode = 2``): the TMA-LDG warp streams ONE ``BIAS_BLOCK_ROWS x TILE_N`` bias
# tile per kv tile through a ``STAGES_BIAS``-deep SMEM ring (2 / 3 stages, the deepest the carveout holds) that the softmax warps
# release per kv tile -- same swizzle, boxes and lane loads, any S_kv.  A width where neither form fits is a TYPED decline on the
# row AND in the adapter.  The kv_mask ALONE stages nothing in SMEM (byte loads + warp ballots per kv tile) -- no block, no S_kv
# budget: served at any S_kv.
BIAS_MODE_NONE, BIAS_MODE_RESIDENT, BIAS_MODE_STREAMED = 0, 1, 2
# The streamed ring's MULTICAST form (``bias_mode = 3``, the (n, n+1) PAIR item only): the kernel launches a cluster of TWO
# INDEPENDENT CTAs along the grid's chunk axis (cluster (1, 1, 2); CTA_MMA / CGA_M / CGA_N stay 1 -- no collective MMA, no shared K / V),
# which decode the SAME (q tile, head) and ADJACENT chunks (c, c + 1) and therefore read the SAME bias tile sequence; the rank-0 CTA
# issues every kv tile's bias boxes ONCE as a cta_group::1 ``.multicast::cluster`` TMA load whose bytes land at the same SMEM offset in
# both CTAs and complete each CTA's OWN ``mb_bias_ring_full`` (every CTA arms its own), and every stage is released to BOTH CTAs'
# ``mb_bias_ring_empty`` (init 2 x BIAS_EMPTY_ARRIVERS).  The L2 -> SMEM bias feed of the pair item is halved.  Served with GLOBAL
# lengths only (no per-row lengths: the two CTAs' kv-tile sequences must be identical) under the NATURAL scheduler; a request-only
# form until its A/B says otherwise (the Rubin pick never picks it by itself).
BIAS_MODE_STREAMED_MULTICAST = 3
BIAS_MODES = frozenset({BIAS_MODE_NONE, BIAS_MODE_RESIDENT, BIAS_MODE_STREAMED, BIAS_MODE_STREAMED_MULTICAST})
# The two forms that stream ONE bias tile per kv tile through the STAGES_BIAS-deep ring (the multicast form is the ring with a
# cluster-wide issue): every "is this the ring" seam reads this set, never ``== BIAS_MODE_STREAMED``.
BIAS_STREAMED_MODES = frozenset({BIAS_MODE_STREAMED, BIAS_MODE_STREAMED_MULTICAST})
# The multicast ring's cluster extent along the chunk axis = its TMA multicast fan-out = the factor on the ring's release count.
BIAS_MCAST_CTAS = 2
# The streamed ring's depth domain: 2 or 3 stages (plan: "a 2-3-stage bias ring").  Deeper buys nothing the K/V ring's own
# lookahead does not already bound (the pair item streams 2 kv tiles ahead); shallower than 2 is no ring.
BIAS_RING_STAGES = (2, 3)
# ``kv_mask`` forms (TemplateParams.kv_mask_form): 1 = the BOOLEAN mask as a bit word (a masked key is -inf, a fully masked row
# reads O = 0 / Stats = -inf); 2 / 3 = the ADDITIVE per-key vector in the io dtype (bf16 / fp16 pairs) / in fp32, folded into the
# raw scores pre-scale like the bias (``s + v * inv_scale``, one packed FFMA per pair) from a warp-uniform 16-byte GMEM load --
# the models' finite -1e9 fold preserved exactly (a fully -1e9 row is the uniform average, NOT the empty-row contract).  Neither
# additive form stages anything in SMEM or adds a barrier row: the bool word's byte loads become 16-byte granule loads.
KV_MASK_NONE, KV_MASK_BOOL_WORD, KV_MASK_ADDITIVE_16, KV_MASK_ADDITIVE_F32 = 0, 1, 2, 3
KV_MASK_FORMS = frozenset({KV_MASK_NONE, KV_MASK_BOOL_WORD, KV_MASK_ADDITIVE_16, KV_MASK_ADDITIVE_F32})
# The ROW-MAX FORM of the bias / kv_mask softmax body (``TemplateParams.bias_max_form`` -> ``CFG.BIAS_MAX_FORM``): 0 = the SOFTWARE row max
# over the biased, -inf-masked scores (the S13' body as built); 1 = the HARDWARE UPPER BOUND -- the fused ``tcgen05.ld.red...max`` of the plain
# body over the RAW scores plus the thread's per-ROW bias-tile max (packed ``max.bf16x2`` / ``max.f16x2`` over the words the FFMA reads), the
# online softmax on ``max_k S + max_k bias >= max_k (S + bias)``.  Per ROW, never the block scalar: the per-row bound's slack is the row's RAW
# score range inside the tile (bias range irrelevant); a block scalar's slack is ``max_block bias - max_row bias``, unbounded for a row-structured
# (masked) bias.  Served with the bit-word kv_mask or none: the additive vector (forms 2 / 3) has no cheap per-tile max -> a typed decline.
BIAS_MAX_SOFTWARE, BIAS_MAX_HW_UPPER_BOUND = 0, 1
BIAS_MAX_FORMS = frozenset({BIAS_MAX_SOFTWARE, BIAS_MAX_HW_UPPER_BOUND})
# The additive vector's extent and strides are whole 16-byte granules (8 halves / 4 floats; the 16-bit form is the stricter one,
# applied to both so the rule reads the same for every dtype): the tail granule is CLAMPED to the last in-bounds one in-kernel.
KV_MASK_ADDITIVE_GRANULE_ELEMS = 8
# The bias block's TMA / lane swizzle: the 128-B atom.  BOTH swizzle jobs force it -- the TMA descriptor writes the
# block under it (job D) and the softmax lanes read one 128-B row slice each (job L: a per-lane stride that is a multiple
# of the 128-B bank cycle is a 32-way conflict unswizzled; the 16-B chunk index XOR (row % 8) spreads 8 rows over 8 bank
# groups -> 4 wavefronts per warp-wide 16-B load = ideal).  ``Swizzle(3, 4, 3)``: MBase + SShift = 7 = log2(128 B).
BIAS_SWZ_BYTES = 128
# The kernel's domain of the pre-load constant (``prefill_d128_f16`` raises outside it): 0 / 1 / 2 chunks ahead of the ``bmm1_done`` wait.
BIAS_PRELOAD_CHUNKS_DOMAIN = frozenset({0, 1, 2})
# The padded TAIL forms of the PLAIN padded body on the small tiles (TemplateParams.padded_tail_form): 0 = the legacy second
# (software-max) body for the tail kv tile, 1 = the PRE-MASKED tail (one HW-max body; the tail tile's out-of-range S columns
# rewritten as -inf in TMEM before the fused max load).
PADDED_TAIL_FORMS = frozenset({0, 1})
# The dead-sub-tile forms of the one-batch-entry item (TemplateParams.dead_subtile_quiet): 0 = the full body on a dead sub-tile,
# 1 = the QUIET body (the live body's barrier sequence, no math).
DEAD_SUBTILE_QUIET_FORMS = frozenset({0, 1})
# The K / V ring depths the small-tile body is validated at (SMEM fit at every served width, the ring-wrap axis n_kv = depth /
# depth + 1 / two wraps of the fresh-process census).  The validator pins cfg.STAGES_KV to this set on the small tiles.
SMALL_D_STAGES_KV = frozenset({2, 3, 4})
# The Q tile heights the small-tile body is validated at (the TILE_M = 64 lever: the half-idle reference lane form).  The validator
# pins cfg.TILE_M to this set on the small tiles; every other flavor keeps 128.
SMALL_D_TILE_MS = frozenset({64, 128})
# The softmax warp-group counts the d128-file body is written for: 2 (every path) and 3 (the (n, n+1) pair item at TILE_N 64 -- the one
# form whose roles loop over the sub-tiles; the one-batch-entry paths hand-unroll two).  The validator pins the serving set.
SMALL_D_SOFTMAX_WGS = frozenset({2, 3})
# The S / P slot counts the small-tile body is validated at: 1 (the classic one slot per sub-tile) and 2 (the double-buffered S at
# TILE_N 64 -- the TMEM map 2 sub-tiles x 2 slots x 64 + O + STATS fits either arch's columns; at TILE_N 128 it fits neither, a
# 3-slot S fits nowhere: 3 x 2 x 128 = 768).
SMALL_D_S_SLOTS = frozenset({1, 2})
# The Q-operand forms the small-tile body is validated at: 0 (Q from SMEM, every committed record) and 1 (Q in TMEM).
SMALL_D_Q_IN_TMEM = frozenset({0, 1})

# 14-bit ``start_address`` of a version-0 tcgen05 SMEM descriptor: every
# descriptor-fed slab (MMA operand or UTCCP scale-factor source) of a module
# that pins DESC_VERSION=0 must START under this, or it wraps to offset 0 and
# the MMA/UTCCP reads the bottom of SMEM (rules/mma-tma-matrix.md S6).
TCGEN05_V0_ADDR_LIMIT = 256 * 1024

_SMEM_SLAB_ALIGN = 1024  # every kernel slab is a 1024-B-aligned cutlass.Array


def _align_slab(nbytes: int) -> int:
    return -(-nbytes // _SMEM_SLAB_ALIGN) * _SMEM_SLAB_ALIGN


def _check(preds) -> None:
    for ok, msg in preds:
        if not ok:
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Tile-geometry helpers both configs read (pure arithmetic; identical on both archs)
# ---------------------------------------------------------------------------


def q_swz_bytes(tile_k: int, bpe_val: int) -> int:
    return 128 if (tile_k * bpe_val) % 128 == 0 else 64


def v_swz_bytes(tile_o: int, cta_mma: int, bpe_val: int) -> int:
    """V is split along d_v under cga2, so its swizzle follows the PER-CTA inner
    stride; Q/K/O use the full-tile inner extent."""
    inner = (tile_o // cta_mma) * bpe_val
    if inner % 128 == 0:
        return 128
    if inner % 64 == 0:
        return 64
    if inner % 32 == 0:
        return 32
    raise ValueError(f"V inner bytes {inner} is not a multiple of 32/64/128")


def o_swz_bytes(tile_o: int, bpe_o: int, pack_div: int = 1) -> int:
    return 128 if (tile_o * bpe_o // pack_div) % 128 == 0 else 64


def sdpa_smem_bytes(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    tile_o: int,
    tiles_q: int,
    stages_kv: int,
    cta_mma: int,
    bpe_qkv: int,
    bpe_o: int,
    qo_alias: bool = False,
    gate_bpe: int = 0,
    sf_bytes: int = 0,
) -> int:
    """Q/K/V/O (+ GATE) (+ MXFP8 scale-factor slabs) SMEM footprint for the
    classic prefill pipeline.

    K (seq rows) and V (d_v cols) shrink by CTA_MMA; Q/O are per-CTA full.
    ``qo_alias`` = O reuses Q's slab in the epilogue (max instead of sum).
    ``gate_bpe`` = bytes per element of the fused epilogue gate's staging tile
    (0 = no gate).  The gate tile is one O-shaped slab per CTA -- full TILE_O
    width and NOT divided by CTA_MMA, because under cga2 each CTA of the pair
    owns a distinct 128-row Q block and gates its own full-width O rows.
    ``sf_bytes`` = the block-scale kernels' scale-factor slabs (Q, K ring, P,
    V ring; ``d256_mxfp8_sf_smem_bytes``), 0 on the f16 / per-tensor kernels.
    """
    s_q = tiles_q * (tile_m * tile_k) * bpe_qkv
    s_o = tiles_q * (tile_m * tile_o) * bpe_o
    s_k = stages_kv * (tile_n * tile_k // cta_mma) * bpe_qkv
    s_v = stages_kv * (tile_o * tile_n // cta_mma) * bpe_qkv
    s_gate = tiles_q * (tile_m * tile_o) * gate_bpe
    return (max(s_q, s_o) if qo_alias else s_q + s_o) + s_k + s_v + s_gate + sf_bytes


# ---------------------------------------------------------------------------
# The bias block's geometry and the per-arch budgets (ONE arithmetic: the row, the heuristics, the adapter, the factory)
# ---------------------------------------------------------------------------


def bias_block_rows(tiles_q: int = 2, tile_m: int = 128, *, pair: bool = False) -> int:
    """Rows of the resident bias block = the Q rows one work item's softmax sub-tiles READ: ``tiles_q x tile_m`` on the
    one-batch-entry item (both sub-tiles are rows of one 256-row super-tile), ``tile_m`` on the (n, n+1) PAIR item
    (``TemplateParams.n_chunk >= 1``: the two sub-tiles are rows n / n + 1 of the SAME 128-row q tile and share the block)."""
    return tile_m if pair else tiles_q * tile_m


def bias_pair_form(bias_batch_extent: int, b_flat: int, cta_mma: int) -> bool:
    """Whether a BIAS plan runs the (n, n+1) PAIR work item (``TemplateParams.n_chunk >= 1``) rather than the
    one-batch-entry item (the 256-row super-tile form): the bias must be batch-BROADCAST (two rows of one chunk read ONE block; a per-batch bias
    ``[B, ., S_q, S_kv]`` keeps the 256-row item), the flattened batch must hold at least two rows (a single row would run a
    dead sub-tile on every item) and the plan runs cga1 (a collective MMA cannot span two rows).  ONE predicate for the
    engine row, the heuristics and the adapter (rule 8b')."""
    return int(bias_batch_extent) == 1 and int(b_flat) >= 2 and int(cta_mma) == 1


def bias_resident_tiles(
    arch: D128FamilyArch,
    tile_k: int,
    cta_mma: int,
    *,
    stages_kv: int,
    tile_m: int,
    bpe_val: int = 2,
    tile_n: int = 128,
    tiles_q: int = 2,
    pair: bool = False,
    qo_alias: bool = False,
) -> int:
    """How many ``tile_n``-column bias tiles (``BIAS_BLOCK_ROWS x tile_n`` elements each) the arch's carveout holds beside
    the d128 family's Q / O slabs and K / V rings at ``tile_k = tile_o``, the given ring depth and Q tile height -- the SAME
    budget :func:`validate_smem` enforces (``arch.smem_usable_bytes`` minus ``arch.smem_fixed_overhead``).  0 = no resident
    block fits.  Derived, never a literal.  Rubin at 4 stages: 3 at the 32-wide tile (cga1), 1 / 2 at the 64-wide one (cga1 /
    cga2), 0 at d128 on the 256-row item; the PAIR item's 128-row block holds twice as many -- 6 / 3.  SM100 (227 KiB) at 4
    stages: 2 / 4 (one-batch-entry / pair) at the 32-wide tile, **0 at the 64-wide tile's cga1 on either item** (its 192 KiB of
    operands leave no 32 KiB tile), 1 at its cga2 one-batch-entry item -- the arch is a parameter so that the SM100 numbers come
    out of the arithmetic and never out of a Rubin threshold."""
    operands = sdpa_smem_bytes(tile_m, tile_n, tile_k, tile_k, tiles_q, stages_kv, cta_mma, bpe_val, bpe_val, qo_alias=qo_alias)
    spare = arch.smem_usable_bytes - arch.smem_fixed_overhead - operands
    tile_bytes = bias_block_rows(tiles_q, tile_m, pair=pair) * tile_n * bpe_val
    return max(0, spare // tile_bytes)


def bias_ring_stages(
    arch: D128FamilyArch,
    tile_k: int,
    cta_mma: int,
    *,
    stages_kv: int,
    tile_m: int,
    bpe_val: int = 2,
    tile_n: int = 128,
    tiles_q: int = 2,
    pair: bool = False,
    qo_alias: bool = False,
) -> int:
    """The deepest STREAMED bias ring (``BIAS_MODE_STREAMED``) the arch's carveout holds beside the operands at this width:
    ``min(max(BIAS_RING_STAGES), spare // tile_bytes)`` in whole ``BIAS_BLOCK_ROWS x tile_n`` tiles, **0 when fewer than
    ``min(BIAS_RING_STAGES)`` fit** (no ring).  The SAME spare the resident budget reads (one arithmetic, rule 8b')."""
    tiles = bias_resident_tiles(
        arch, tile_k, cta_mma, stages_kv=stages_kv, tile_m=tile_m, bpe_val=bpe_val, tile_n=tile_n, tiles_q=tiles_q, pair=pair, qo_alias=qo_alias
    )
    stages = min(max(BIAS_RING_STAGES), tiles)
    return stages if stages >= min(BIAS_RING_STAGES) else 0


def operand_region_end_bytes(cfg) -> int:
    """Where the descriptor-fed region of a d128-family kernel ENDS: sQ, sO, sK, sV in the body's declaration order, each a
    1024-B-aligned slab -- the roots every tcgen05 SMEM descriptor is built from.  The bias block that lanes read directly is declared
    AFTER them, so with this below ``TCGEN05_V0_ADDR_LIMIT`` every descriptor root stays inside the version-0 window
    whatever the block's size (the block itself may sit past 256 KiB: no descriptor touches it)."""
    s_q = _align_slab(cfg.TILES_Q * cfg.TILE_M * cfg.TILE_K * cfg.BPE)
    s_o = _align_slab(cfg.TILES_Q * cfg.TILE_M * cfg.TILE_O * cfg.BPE_O)
    s_k = _align_slab(cfg.STAGES_KV * (cfg.TILE_N * cfg.TILE_K // cfg.CTA_MMA) * cfg.BPE)
    s_v = _align_slab(cfg.STAGES_KV * (cfg.TILE_O * cfg.TILE_N // cfg.CTA_MMA) * cfg.BPE)
    return s_q + s_o + s_k + s_v


def bias_smem_tiles(cfg) -> int:
    """The bias block's SMEM extent in ``BIAS_BLOCK_ROWS x TILE_N`` tiles, by FORM: ``BIAS_RESIDENT_TILES`` on the resident block
    (BIAS_MODE 1), ``STAGES_BIAS`` on the streamed ring (BIAS_MODE 2) and its multicast form (BIAS_MODE 3: the same per-CTA ring), 0
    without the bias.  The kernel hands this to ``tile_dsl.bias.bias_block_geometry(n_tiles=)`` so the TMA box count, the expect_tx
    and the per-thread read offsets stay one arithmetic."""
    if cfg.BIAS_MODE == BIAS_MODE_RESIDENT:
        return cfg.BIAS_RESIDENT_TILES
    if cfg.BIAS_MODE in BIAS_STREAMED_MODES:
        return cfg.STAGES_BIAS
    return 0


def bias_resident_block_bytes(cfg) -> int:
    """SMEM bytes of the bias block in its FORM (0 without the bias): :func:`bias_smem_tiles` tiles of ``BIAS_BLOCK_ROWS x TILE_N``
    elements in the io dtype -- the resident block's tiles or the streamed ring's stages (the name predates the ring; the
    ``validate_smem`` term it feeds names the form)."""
    return bias_smem_tiles(cfg) * cfg.BIAS_BLOCK_ROWS * cfg.TILE_N * cfg.BPE


# ---------------------------------------------------------------------------
# The field block every d128-file Cfg carries (the OFF defaults trace the committed instruction stream byte-for-byte)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class D128FamilySmallTileFields:
    """The small-tile seams of the f16/bf16 d128-FILE bodies as Cfg fields.  Every seam in a body folds on
    ``cutlass.const_expr(CFG.<FIELD>)``, so the defaults below trace the pre-seam instruction stream byte-for-byte (the committed
    cubin records pin it on both archs).  Mixed into ``config_sm107._CfgSm107`` and ``config_sm100.CfgD128`` (dataclass
    inheritance: these fields precede the arch's own; every construction site is keyword-only and no cache key hashes a Cfg)."""

    # The padded tail form (TemplateParams.padded_tail_form; the f16/bf16 small tiles' PLAIN padded body, MASK_FLAGS == MASK_PADDED
    # alone): 0 = the legacy second (software-max) body for the tail kv tile, 1 = the pre-masked tail (one HW-max body; the tail
    # tile's out-of-range S columns rewritten as -inf in TMEM before the fused max load).  Every kernel seam folds on
    # cutlass.const_expr(CFG.PADDED_TAIL_FORM == 1); the factory refuses 1 off the eligible set.
    PADDED_TAIL_FORM: int = 0
    # The dead-sub-tile quiet body (TemplateParams.dead_subtile_quiet; the one-batch-entry item of the f16/bf16 small tiles, no
    # bias): 1 = a softmax sub-tile with no live row (its first row >= the batch entry's S_q) runs the kv range through the QUIET
    # body (the live body's barrier sequence, no math) and the live arms see an empty range; 0 = it runs the full body.  Every
    # kernel seam folds on cutlass.const_expr(CFG.DEAD_SUBTILE_QUIET == 1); the factory refuses 1 off the eligible set.
    DEAD_SUBTILE_QUIET: int = 0
    # --- additive bias (resident row-block) + per-key boolean mask; d128-file f16/bf16 only.  Every seam in the body
    # folds on cutlass.const_expr(CFG.BIAS_MODE) / (CFG.KV_MASK_FORM), so 0 / 0 traces the pre-bias instruction stream
    # byte-for-byte.  BIAS_RESIDENT_TILES / BIAS_BLOCK_ROWS size the SMEM block (bias_resident_tiles -- what the
    # carveout leaves after the operands, in whole TILE_N-column tiles); BIAS_EMPTY_ARRIVERS is the mb_bias_empty init
    # count = the softmax warps (one elected arrive each per work item).
    BIAS_MODE: int = 0
    BIAS_RESIDENT_TILES: int = 0
    BIAS_BLOCK_ROWS: int = 0
    BIAS_SWZ_BYTES: int = 128
    BIAS_BOXES_PER_TILE: int = 0
    BIAS_EMPTY_ARRIVERS: int = 0
    KV_MASK_FORM: int = 0
    # The (n, n+1) PAIR work item (TemplateParams.n_chunk): 0 = the one-batch-entry item; n >= 1 = a work item is (a 128-row
    # q tile, a head, a chunk of up to n consecutive flattened-batch rows), the two sub-tiles rows (nb, nb+1) of that q tile
    # sharing the 128-row bias block (BIAS_BLOCK_ROWS = TILE_M).  Every seam folds on cutlass.const_expr(CFG.N_CHUNK > 0).
    N_CHUNK: int = 0
    # The STREAMED bias ring's depth (BIAS_MODE == 2; TemplateParams.bias_stages): 2 / 3 stages of BIAS_BLOCK_ROWS x TILE_N bias
    # tiles, one tile per kv tile, released per kv tile (mb_bias_ring_full / mb_bias_ring_empty, STAGES_BIAS stages each);
    # 0 on BIAS_MODE 0 / 1 (the resident block sizes itself by BIAS_RESIDENT_TILES).  A pure latency knob: no accumulator
    # parity is indexed from this ring.
    STAGES_BIAS: int = 0
    # The row-max form of the -inf body (TemplateParams.bias_max_form): 0 = the software max tree, 1 = the hardware upper bound
    # (BIAS_MAX_HW_UPPER_BOUND).  Pure instruction-stream axis: no barrier, no SMEM, no accumulator parity depends on it.
    BIAS_MAX_FORM: int = 0
    # The S / P TMEM slot count per softmax sub-tile (TemplateParams.s_slots; d128-file f16/bf16 small tiles at TILE_N 64 only): 1 = the
    # classic one slot per sub-tile (every seam folds on cutlass.const_expr(CFG.S_SLOTS == 2), so 1 traces the one-slot instruction
    # stream byte-for-byte), 2 = the double-buffered S -- the TMEM map gains the slot axis (d128_family_tmem_layout: S[qs][p] at
    # S_OFF[qs] + p * S_SLOT_STRIDE), mb_bmm1_done / mb_bmm2_ready gain a x 2 stage axis (make_classic_bars(s_stages=TILES_Q x S_SLOTS)),
    # and every role walks a depth-2 PipelineState cursor over the slots, advanced once per non-empty kv tile.  A pure overlap knob:
    # the same MMAs and softmax steps in the same order -- O and Stats are BITWISE the one-slot kernel's.
    S_SLOTS: int = 1
    # The streamed ring's MULTICAST form (BIAS_MODE == 3): the launch's cluster extent along the grid's CHUNK axis (cluster (1, 1, 2)) =
    # the bias TMA's multicast fan-out = the factor on mb_bias_ring_empty's release count (each CTA's ring is released by BOTH CTAs'
    # softmax warps).  1 on every other BIAS_MODE (cluster (CTA_MMA, 1, 1), the collective geometry).  The kernel's CGA_SIZE (the
    # scheduler credit fan-out and the mb_read_tile_id count) multiplies by it; CGA_M / CGA_N / CTA_MMA stay 1 on this form.
    BIAS_MCAST_CTAS: int = 1
    # Q resident in TMEM (TemplateParams.q_in_tmem; the f16/bf16 32- / 64-wide d128-file tiles at cga1 / TILE_M 128 / two sub-tiles): 0 = BMM1
    # reads Q from SMEM through its descriptor on every kv tile (every seam folds on cutlass.const_expr(CFG.Q_IN_TMEM == 1), so 0 traces the
    # committed instruction stream byte-for-byte), 1 = the MMA warp copies each sub-tile's Q tile SMEM -> TMEM once per n-step (tcgen05.cp
    # 128x128b, Q_COLS / 4 calls through the same descriptor, one elect block) whose commit releases the Q slab (mb_q_empty) at the START of
    # the step, and issues every BMM1 as an A-from-TMEM MMA.  The TMEM map gains Q_COLS = TILE_K x BPE / 4 columns per sub-tile after the
    # last O slot (d128_family_tmem_layout.Q_OFF); no barrier object, no SMEM byte and no init count moves (validate_q_in_tmem).
    Q_IN_TMEM: int = 0


# ---------------------------------------------------------------------------
# The TMEM column map, derived from the Cfg (ONE formula for the body and the validator)
# ---------------------------------------------------------------------------

_D128_FAMILY_STATS_REGION_COLS = 32


@dataclass(frozen=True)
class D128FamilyTmemLayout:
    """Column offsets (fp32 TMEM columns, 128 lanes) of the classic two-sub-tile pipeline, see :func:`d128_family_tmem_layout`."""

    TOTAL_COLS: int
    S_OFF: Tuple[int, ...]
    P_OFF: Tuple[int, ...]
    O_OFF: Tuple[int, ...]
    STATS_OFF: int
    STATS_COLS_PER_SUBTILE: int = 8
    # The S / P slot axis (Cfg.S_SLOTS; 1 on every rendering but the double-buffered small tiles): slot p of sub-tile qs sits at
    # S_OFF[qs] + p * S_SLOT_STRIDE (its P at P_OFF[qs] + p * S_SLOT_STRIDE); S_OFF / P_OFF name slot 0.
    S_SLOTS: int = 1
    S_SLOT_STRIDE: int = 0
    # The A-from-TMEM Q slot of the TS arm (Cfg.Q_IN_TMEM): Q_COLS = TILE_K x BPE / 4 fp32 columns per sub-tile (two 16-bit elements per
    # column: 16 / 32 at TILE_K 32 / 64), Q_OFF[qs] after the LAST O slot; computed on every record, read by the body and validated for the
    # fit (Q end <= STATS_OFF) under the arm only -- STATS_OFF does not move, so every record without the arm keeps its map.
    Q_OFF: Tuple[int, ...] = ()
    Q_COLS: int = 0


def d128_family_tmem_layout(cfg, *, total_cols: int, stats_cols_per_subtile: int = 8) -> D128FamilyTmemLayout:
    """Derive the d128-family TMEM map from the Cfg: ``S[qs] = qs * TILE_N`` (the fp32 S accumulator, TILE_N columns per
    sub-tile); ``P[qs]`` in the TAIL of its own S slot -- the packed BMM2 A operand is ``TILE_N * BPE / 4`` columns wide
    (bf16 / f16 pack two probabilities per fp32 cell: 64 columns at TILE_N = 128, so P sits at +64; a 4-byte dtype would
    alias the full slot) and BMM1 has finished with S before P is written; ``O[qs] = TILES_Q * TILE_N + qs * TILE_O`` (the
    fp32 O accumulator, TILE_O columns per sub-tile: 128 / 64 / 32); STATS at the top of the allocation.  Every term is a
    Cfg field, never a literal copied from the d128 map (a literal O1 offset of 384 at TILE_O = 32 would leave 352 unused
    columns and no error; a literal P offset of 64 at a TILE_N of 64 would overlap the live S columns).

    ``total_cols``: the arch's TMEM column total (576 on Rubin, 512 on SM100): STATS_OFF = total - 32 (544 / 480) at the
    default stats width, so the same Cfg maps to each arch's allocation and the fit predicate reads the arch's number.

    ``stats_cols_per_subtile``: 8 on every rendering today (alpha; the final (max, sum); the score side outputs' true max in
    columns 2-3).  The 64-row tile's quad-fragment form (``TILE_M64_LANE_FORM = 1``) publishes each value 8 wide through the
    ``16x256b .x1`` store and takes 24; the region at the top grows by the extra columns per sub-tile, so the default keeps
    STATS_OFF at total - 32 and every committed record byte-identical."""
    if stats_cols_per_subtile < 8 or stats_cols_per_subtile % 8:
        raise ValueError(f"d128_family_tmem_layout: stats_cols_per_subtile must be a multiple of 8, at least 8 (got {stats_cols_per_subtile})")
    stats_region = _D128_FAMILY_STATS_REGION_COLS + cfg.TILES_Q * (stats_cols_per_subtile - 8)
    # The slot axis (Cfg.S_SLOTS, 1 on every record but the double-buffered small tiles; read through getattr so the quantized
    # siblings' Cfg classes, which carry no such field, keep their map): sub-tile qs owns S_SLOTS consecutive TILE_N-column slots,
    # S[qs][p] = (qs * S_SLOTS + p) * TILE_N, each with its own P tail; O follows the LAST slot.  At S_SLOTS = 1 every value is the map
    # the kernels always had.
    n_slots = int(getattr(cfg, "S_SLOTS", 1))
    s_off = tuple(qs * n_slots * cfg.TILE_N for qs in range(cfg.TILES_Q))
    p_width = (cfg.TILE_N * cfg.BPE) // 4
    p_off = tuple(s + cfg.TILE_N - p_width for s in s_off)
    o_base = cfg.TILES_Q * n_slots * cfg.TILE_N
    o_off = tuple(o_base + qs * cfg.TILE_O for qs in range(cfg.TILES_Q))
    # The TS arm's Q slots (Cfg.Q_IN_TMEM; read through getattr so the quantized siblings' Cfg classes keep their map): TILE_K x BPE / 4
    # columns per sub-tile after the last O slot -- the d512 f16 kernel's Q_TMEM_COLS == TILE_K / 2 at BPE 2.
    q_cols = (cfg.TILE_K * cfg.BPE) // 4
    q_base = o_off[-1] + cfg.TILE_O
    q_off = tuple(q_base + qs * q_cols for qs in range(cfg.TILES_Q))
    return D128FamilyTmemLayout(
        TOTAL_COLS=int(total_cols),
        S_OFF=s_off,
        P_OFF=p_off,
        O_OFF=o_off,
        STATS_OFF=int(total_cols) - stats_region,
        STATS_COLS_PER_SUBTILE=stats_cols_per_subtile,
        S_SLOTS=n_slots,
        S_SLOT_STRIDE=cfg.TILE_N if n_slots > 1 else 0,
        Q_OFF=q_off,
        Q_COLS=q_cols,
    )


# ---------------------------------------------------------------------------
# The d128-file warp table as a function of the softmax WARP-GROUP count
# ---------------------------------------------------------------------------
#
# Warp map: 8 softmax (2 warpgroups) + 4 correction + MMA + TMALDG + TMASTG + scheduler.  15 of the 16 warps arrive on the
# scheduler ring (all but the scheduler warp itself).  2 = the 16-warp body; 3 = the 20-warp body of the TILE_N-64 pair arm
# (TemplateParams.softmax_wgs): 4 warps per softmax group, 4 correction warps, then the MMA / TMA-LDG / TMA-STG / scheduler
# warps -- the kernel's role dispatch reads the Cfg ids and derives WG2's first warp as SOFTMAX_WG0_BASE + 2 x SOFTMAX_WG_WARPS.
D128_SOFTMAX_WG_WARPS = 4
D128_CORRECTION_WARPS = 4
D128_OTHER_WARPS = 4


def d128_total_warps(softmax_wgs: int) -> int:
    return softmax_wgs * D128_SOFTMAX_WG_WARPS + D128_CORRECTION_WARPS + D128_OTHER_WARPS


def d128_read_tile_arrivers(softmax_wgs: int) -> int:
    """The per-CTA ``read_tile_id_arrive`` callers of the d128-file body: every softmax warp, every correction warp, the MMA, the TMA-LDG
    and the TMA-STG warp (the kernel's READ_TILE_ARRIVERS_TOTAL at CGA_SIZE 1: ``(WGS x 4 + 4 + 1 + 1) x CGA_SIZE + CGA_M / CTA_MMA``, the
    last term the STG warp): 15 at two groups, 19 at three.  A count the body cannot reach is a hang at EVERY shape (the config_sm107
    module docstring's d256 story)."""
    return softmax_wgs * D128_SOFTMAX_WG_WARPS + D128_CORRECTION_WARPS + 1 + 1 + 1


# The per-role register split (SOFTMAX_REGS, CORRECTION_REGS) of the d128-file body by (softmax warp groups, head-dim tile); the four
# 40-register roles are fixed.  SUM(role regs x warps) EQUALS the launch's entry pool (reg_entry_pool: 128 x 16 = 2048; 96 x 20 = 1920),
# as every CUTLASS warp-specialized split does: 8 x 192 + 4 x 88 + 4 x 40 = 2048; 12 x 128 + 4 x 56 + 4 x 40 = 1920 at the 32-wide tile
# (the correction holds a 32-column O row); 12 x 120 + 4 x 80 + 4 x 40 = 1920 at the 64-wide one (a 64-column O row needs 80 -- 56 spills
# -- so the softmax gives up 8).  The 128-register softmax of the 3-WG arm holds ONE 64-column chunk: reg_S 64 + the packed P chunk 32 +
# the loop state, no bias pre-load (the kernel derives _BIAS_PRELOAD_EFF 0 there).  frost-kernels.md s2.  The sm100 d128 FLAVOR runs its
# own 2 x 184 + 88 + 56 (its MMA warp holds 16 Q descriptors) -- a d128-flavor fact, not a small-tile one; the small tiles run this table.
D128_REG_SPLIT = {2: (192, 88), (3, 32): (128, 56), (3, 64): (120, 80)}


# ---------------------------------------------------------------------------
# Shared validators (each takes the arch BUDGET where a number of the arch enters)
# ---------------------------------------------------------------------------


def validate_regs(cfg, flavor: str) -> None:
    """The four hardware register constraints. Equality and alignment are HW
    requirements, not style: violating them gives wrong results or a crash, with
    no diagnostic."""
    _check(
        [
            (
                cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS,
                f"{flavor}: MMA/TMALDG/TMASTG/SCHEDULER regs must be equal",
            ),
            (
                cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512,
                f"{flavor}: register budget over 512",
            ),
            (
                all(r % 8 == 0 for r in (cfg.MMA_REGS, cfg.CORRECTION_REGS, cfg.SOFTMAX_REGS, cfg.OTHER_REGS)),
                f"{flavor}: every per-role register count must be a multiple of 8",
            ),
            (
                all(24 <= r <= 256 for r in (cfg.MMA_REGS, cfg.SOFTMAX_REGS) if r),
                f"{flavor}: per-warp register counts must be within 24..256",
            ),
        ]
    )


def validate_warp_layout(cfg, flavor: str, *, expect_total: int, expect_arrivers: int) -> None:
    """Pin the warp population against the count the kernel BODY implements.

    This is the check whose absence hung the d256 FP8 port: the body's role
    dispatch and its ``read_tile_id_arrive`` call sites are fixed, so a config
    that disagrees produces an mbarrier init count nothing can reach."""
    _check(
        [
            (cfg.TOTAL_WARPS == expect_total, f"{flavor}: kernel body has {expect_total} warps, config says {cfg.TOTAL_WARPS}"),
            (cfg.THREADS_PER_CTA == cfg.TOTAL_WARPS * 32, f"{flavor}: THREADS_PER_CTA must be TOTAL_WARPS*32"),
            (
                cfg.READ_TILE_ARRIVERS == expect_arrivers,
                f"{flavor}: READ_TILE_ARRIVERS must be {expect_arrivers} for this body (got {cfg.READ_TILE_ARRIVERS}); "
                f"a wrong count is an unreachable scheduler mbarrier, i.e. a hang at EVERY shape",
            ),
        ]
    )


def validate_swizzles(cfg, flavor: str) -> None:
    _check(
        [
            (cfg.Q_SWZ_BYTES in (64, 128), f"{flavor}: Q swizzle must be 64/128 B"),
            (cfg.K_SWZ_BYTES in (64, 128), f"{flavor}: K swizzle must be 64/128 B"),
            (cfg.V_SWZ_BYTES in (32, 64, 128), f"{flavor}: V swizzle must be 32/64/128 B"),
            (cfg.O_SWZ_BYTES in (64, 128), f"{flavor}: O swizzle must be 64/128 B"),
        ]
    )


def validate_smem(cfg, flavor: str, *, arch: D128FamilyArch, qo_alias: bool, sf_bytes: int = 0, bias_bytes: int = 0) -> None:
    """``sf_bytes``: the block-scale kernel's scale-factor slabs (0 on the f16 /
    per-tensor flavors); ``bias_bytes``: the resident bias row-block of the f16 d128-file
    tiles (0 without the bias); both named in the message so a reader sees every term.  ``arch``: the budget
    (``smem_usable_bytes`` / ``smem_fixed_overhead``) and the name the message prints."""
    gate_bpe = cfg.GATE_BPE if getattr(cfg, "EPILOGUE_GATE", 0) else 0
    gate_kib = (cfg.TILES_Q * cfg.TILE_M * cfg.TILE_O * gate_bpe) // 1024
    used = (
        sdpa_smem_bytes(
            cfg.TILE_M,
            cfg.TILE_N,
            cfg.TILE_K,
            cfg.TILE_O,
            cfg.TILES_Q,
            cfg.STAGES_KV,
            cfg.CTA_MMA,
            cfg.BPE,
            cfg.BPE_O,
            qo_alias=qo_alias,
            gate_bpe=gate_bpe,
            sf_bytes=sf_bytes,
        )
        + bias_bytes
        + arch.smem_fixed_overhead
    )
    if used > arch.smem_usable_bytes:
        gate_term = f" + {gate_kib} KiB GATE staging tile" if gate_bpe else ""
        sf_term = (
            f" + {sf_bytes // 1024} KiB MXFP8 scale-factor slabs (Q 1 | K {cfg.STAGES_KV}x1 | P 1 (512 B padded) | V {cfg.STAGES_KV}x1)" if sf_bytes else ""
        )
        if bias_bytes:
            _streamed = getattr(cfg, "BIAS_MODE", 0) in BIAS_STREAMED_MODES
            gate_term += (
                f" + {bias_bytes // 1024} KiB {'streamed bias ring' if _streamed else 'resident bias block'} "
                f"({getattr(cfg, 'STAGES_BIAS', 0) if _streamed else getattr(cfg, 'BIAS_RESIDENT_TILES', 0)} {'stages' if _streamed else 'tiles'} x "
                f"{getattr(cfg, 'BIAS_BLOCK_ROWS', 0)} rows x {cfg.TILE_N} cols x {cfg.BPE} B)"
            )
        raise ValueError(
            f"{flavor}: SMEM {used // 1024} KiB (Q/K/V/O + {arch.smem_fixed_overhead // 1024} KiB fixed{gate_term}{sf_term}) exceeds the "
            f"{arch.smem_usable_bytes // 1024} KiB usable {arch.name} carveout at STAGES_KV={cfg.STAGES_KV}, CTA_MMA={cfg.CTA_MMA}"
            f"{', EPILOGUE_GATE=1' if gate_bpe else ''}. "
            f"Overflowing it does NOT fail the launch -- it clobbers the last buffer allocated (measured: O came back "
            f"50 % zeros with an exact LSE), so this must raise here"
        )


def validate_head_dim_tile(cfg, flavor: str, *, arch: D128FamilyArch) -> None:
    """The head-dim tile and everything coupled to it (frost-kernels.md s5 "bind the couplings" / s6).

    One Cfg field, ``TILE_K`` / ``TILE_O``, sets a TMA box, an MMA descriptor swizzle, the epilogue's SMEM store swizzle,
    the TMEM O slot width and the BMM1 k-step count at once; moving it without the rest raises HERE with the failure it
    would otherwise produce, instead of a numerically perfect slow kernel or a cos ~ 0.006 one."""
    lay = d128_family_tmem_layout(cfg, total_cols=arch.tmem_total_cols)
    q_swz, v_swz, o_swz = q_swz_bytes(cfg.TILE_K, cfg.BPE), v_swz_bytes(cfg.TILE_O, cfg.CTA_MMA, cfg.BPE), o_swz_bytes(cfg.TILE_O, cfg.BPE_O, cfg.O_PACK_DIV)
    _check(
        [
            (
                cfg.TILE_K in (32, 64, 128, 192) and cfg.TILE_O in (32, 64, 128),
                f"{flavor}: TILE_K must be one of 32/64/128/192 and TILE_O one of 32/64/128 (got {cfg.TILE_K}/{cfg.TILE_O})",
            ),
            (
                cfg.TILE_K == cfg.TILE_O or (cfg.TILE_K, cfg.TILE_O) == (192, 128),
                f"{flavor}: the d128 family serves TILE_K == TILE_O (32/64/128) or the d192xd128 pair (got {cfg.TILE_K}/{cfg.TILE_O})",
            ),
            (
                cfg.TILE_M in SMALL_D_TILE_MS,
                f"{flavor}: TILE_M must be one of {sorted(SMALL_D_TILE_MS)} (got {cfg.TILE_M}): the softmax / correction lane mapping and the "
                f"tcgen05.mma datapath layout (M = 128: one lane per row; M = 64: 16 live rows per 32-lane TMEM sub-partition) exist for these two only",
            ),
            (
                cfg.TILE_M == 128 or (cfg.TILE_K <= 64 and cfg.DTYPE_QKV in (_DTYPE_BF16, _DTYPE_FP16)),
                f"{flavor}: the 64-row Q tile is wired on the f16/bf16 32- / 64-wide tiles only (TILE_K={cfg.TILE_K}, DTYPE_QKV={cfg.DTYPE_QKV}); the "
                f"d128 / d192 renderings keep their committed records and the quantized siblings' k_dim / SF geometry is tied to the 128-row tile",
            ),
            (
                cfg.TILE_M == 128 or cfg.CTA_MMA == 1,
                f"{flavor}: the 64-row Q tile runs cga1 only (CTA_MMA={cfg.CTA_MMA}): a cta_group::2 MMA with a 64-row per-CTA tile has a different "
                f"TMEM datapath layout than the one the warp-lane -> row mapping was derived for",
            ),
            (
                cfg.TILE_M == 128 or not (cfg.THD_VARLEN or cfg.SPLIT_KV > 1 or cfg.PACK_GQA or getattr(cfg, "EPILOGUE_GATE", 0)),
                f"{flavor}: the 64-row Q tile serves dense, unsplit, non-PackGQA, ungated graphs only (the small-tile rows' own gates)",
            ),
            (
                cfg.TILE_K > 64 or cfg.STAGES_KV in SMALL_D_STAGES_KV,
                f"{flavor}: the 32- / 64-wide tiles' K / V ring is validated at STAGES_KV in {sorted(SMALL_D_STAGES_KV)} (got {cfg.STAGES_KV}); the body is "
                f"depth-generic but the SMEM fit and the ring-wrap census hold for these depths only -- a depth of 1 serialises every kv tile behind "
                f"its own BMM1 and a depth above 4 was never run",
            ),
            (
                cfg.TILE_K % cfg.TILE_K_HW_BMM1 == 0 and cfg.TILE_O % 16 == 0,
                f"{flavor}: TILE_K must be whole BMM1 k-steps of {cfg.TILE_K_HW_BMM1} and TILE_O a multiple of the correction warps' 16-column O chunk",
            ),
            (
                cfg.Q_SWZ_BYTES == q_swz and cfg.K_SWZ_BYTES == q_swz,
                f"{flavor}: Q/K swizzle {cfg.Q_SWZ_BYTES}/{cfg.K_SWZ_BYTES} B must follow the tile width ({q_swz} B rows at TILE_K={cfg.TILE_K}, BPE={cfg.BPE}): "
                f"the TMA descriptor and the MMA descriptor read the same atom, and a mismatch is cos ~ 0.006 on every output, not a slow kernel",
            ),
            (
                cfg.V_SWZ_BYTES == v_swz,
                f"{flavor}: V swizzle {cfg.V_SWZ_BYTES} B must follow the per-CTA d_v extent ({v_swz} B at TILE_O={cfg.TILE_O}, CTA_MMA={cfg.CTA_MMA})",
            ),
            (
                cfg.O_SWZ_BYTES == o_swz,
                f"{flavor}: O swizzle {cfg.O_SWZ_BYTES} B must follow the O row ({o_swz} B at TILE_O={cfg.TILE_O}): the epilogue's swizzled SMEM store and the O TMA descriptor are ONE unit",
            ),
            (
                (cfg.TILE_K * cfg.BPE) % cfg.Q_SWZ_BYTES == 0
                and (cfg.TILE_O * cfg.BPE) % cfg.V_SWZ_BYTES == 0
                and (cfg.TILE_O * cfg.BPE_O // cfg.O_PACK_DIV) % cfg.O_SWZ_BYTES == 0,
                f"{flavor}: every TMA inner extent must be whole swizzle atoms (TMA_QK/VO/O iteration counts are integers)",
            ),
            (
                cfg.N_BMM2_CHUNKS * cfg.BMM2_CHUNK_SIZE == cfg.TILE_N,
                f"{flavor}: the softmax chunking is TILE_N-derived (the KV axis), never TILE_O's: N_BMM2_CHUNKS x BMM2_CHUNK_SIZE must equal TILE_N "
                f"(got {cfg.N_BMM2_CHUNKS} x {cfg.BMM2_CHUNK_SIZE} vs {cfg.TILE_N}); a TILE_O-scaled chunk reads half of S and signals half the bmm2_ready barriers -- a silent NaN or a hang",
            ),
            (
                lay.O_OFF[-1] + cfg.TILE_O <= lay.STATS_OFF,
                f"{flavor}: TMEM map S {lay.S_OFF} | P {lay.P_OFF} | O {lay.O_OFF} (+{cfg.TILE_O}) must end at or before the STATS region (column {lay.STATS_OFF} of {lay.TOTAL_COLS}); "
                f"an overlap accumulates the next KV tile's O under the softmax stats",
            ),
            (
                all(o % 32 == 0 for o in lay.O_OFF)
                and all(sv % 32 == 0 for sv in lay.S_OFF)
                and all(pv % 8 == 0 for pv in lay.P_OFF)
                and lay.STATS_OFF % 8 == 0,
                f"{flavor}: accumulator slot bases must be 32-column aligned and P / STATS bases 8-column aligned (S {lay.S_OFF}, P {lay.P_OFF}, O {lay.O_OFF}, STATS {lay.STATS_OFF})",
            ),
        ]
    )


def validate_tile_n(cfg, flavor: str, *, arch: D128FamilyArch) -> None:
    """The KV tile width and what it couples to (M-2: TILE_N as a Cfg field).  The S accumulator takes TILE_N fp32 columns
    per sub-tile and P half of them in its tail; with two sub-tiles the TMEM map must still end under the STATS region, so
    256 is served only at ONE sub-tile -- a predicate, not a comment."""
    lay = d128_family_tmem_layout(cfg, total_cols=arch.tmem_total_cols)
    _check(
        [
            (cfg.TILE_N in (64, 128, 256), f"{flavor}: TILE_N must be one of 64 / 128 / 256 (got {cfg.TILE_N})"),
            (
                cfg.TILE_N != 256 or cfg.TILES_Q == 1,
                f"{flavor}: TILE_N = 256 needs ONE softmax sub-tile: {cfg.TILES_Q} sub-tiles x 256 S columns + the O slots + STATS exceed the "
                f"{lay.TOTAL_COLS}-column TMEM allocation (the body indexes its S / P parity slots per sub-tile)",
            ),
            (
                cfg.N_BMM2_CHUNKS * cfg.BMM2_CHUNK_SIZE == cfg.TILE_N and cfg.BMM2_CHUNK_SIZE == 64,
                f"{flavor}: the softmax chunking is TILE_N / 64 chunks of 64 (got {cfg.N_BMM2_CHUNKS} x {cfg.BMM2_CHUNK_SIZE} vs TILE_N={cfg.TILE_N})",
            ),
            (
                getattr(cfg, "S_SLOTS", 1) in SMALL_D_S_SLOTS,
                f"{flavor}: the S / P slot count per sub-tile is validated at S_SLOTS in {sorted(SMALL_D_S_SLOTS)} (got {getattr(cfg, 'S_SLOTS', 1)}): a 3-slot S "
                f"fits no TMEM map at two sub-tiles (3 x 2 x 128 = 768 > {lay.TOTAL_COLS}) and the body walks a depth-S_SLOTS cursor validated at 1 / 2",
            ),
            (
                getattr(cfg, "S_SLOTS", 1) == 1 or cfg.TILE_N == 64,
                f"{flavor}: the double-buffered S (S_SLOTS = {getattr(cfg, 'S_SLOTS', 1)}) is the TILE_N 64 form: at TILE_N {cfg.TILE_N} the map "
                f"{cfg.TILES_Q} sub-tiles x {getattr(cfg, 'S_SLOTS', 1)} slots x {cfg.TILE_N} S columns + the O slots ends at {lay.O_OFF[-1] + cfg.TILE_O}, past the "
                f"STATS region at {lay.STATS_OFF} of the {lay.TOTAL_COLS}-column allocation -- the next kv tile's BMM1 would overwrite the softmax stats",
            ),
            (
                getattr(cfg, "S_SLOTS", 1) == 1
                or ((cfg.TILE_K <= 64 or (arch.name == "Rubin" and cfg.TILE_K == cfg.TILE_O == 128)) and cfg.DTYPE_QKV > _DTYPE_E5M2),
                f"{flavor}: the double-buffered S requires f16/bf16 small tiles or native Rubin D128 (TILE_K={cfg.TILE_K}, DTYPE_QKV={cfg.DTYPE_QKV}): the "
                f"quantized d128 files keep one slot per sub-tile",
            ),
            (
                getattr(cfg, "S_SLOTS", 1) == 1 or getattr(cfg, "SOFTMAX_WARPGROUPS", 2) == 2,
                f"{flavor}: the double-buffered S (S_SLOTS = {getattr(cfg, 'S_SLOTS', 1)}) is served with two softmax warp groups only (got "
                f"SOFTMAX_WARPGROUPS={getattr(cfg, 'SOFTMAX_WARPGROUPS', 2)}): the two-slot MMA issue order (BMM1(k) of both sub-tiles ahead of BMM2(k - 1) of "
                f"both) hand-unrolls the two-sub-tile body and its barrier audit is the two-group one; the third softmax warp group keeps one slot per sub-tile",
            ),
            (
                lay.O_OFF[-1] + cfg.TILE_O <= lay.STATS_OFF,
                f"{flavor}: TILES_Q x S_SLOTS x TILE_N + TILES_Q x TILE_O = {lay.O_OFF[-1] + cfg.TILE_O} TMEM columns must end at or before STATS ({lay.STATS_OFF})",
            ),
        ]
    )


def validate_bias(cfg, flavor: str, *, arch: D128FamilyArch) -> None:
    """The resident bias block and the kv_mask word: every coupling of the block to the body, as predicates whose
    messages name the failure (frost-kernels s5 "bind the couplings"; sdpa-invariants s6)."""
    if not cfg.BIAS_MODE and not cfg.KV_MASK_FORM:
        _check(
            [
                (
                    cfg.BIAS_RESIDENT_TILES == 0 and cfg.BIAS_BLOCK_ROWS == 0 and cfg.STAGES_BIAS == 0 and cfg.BIAS_MCAST_CTAS == 1,
                    f"{flavor}: bias geometry fields must be 0 (BIAS_MCAST_CTAS 1) without BIAS_MODE",
                ),
                (
                    cfg.BIAS_MAX_FORM == BIAS_MAX_SOFTWARE,
                    f"{flavor}: BIAS_MAX_FORM={cfg.BIAS_MAX_FORM} names a row-max form of the bias / kv_mask body; this config has neither arm",
                ),
            ]
        )
        return
    preds = [
        (
            cfg.BIAS_MODE in BIAS_MODES,
            f"{flavor}: BIAS_MODE must be 0 (none), 1 (resident row-block), 2 (streamed ring) or 3 (streamed ring, cluster multicast); got {cfg.BIAS_MODE}",
        ),
        (
            cfg.KV_MASK_FORM in KV_MASK_FORMS,
            f"{flavor}: KV_MASK_FORM must be 0 (none), 1 (bit word), 2 (additive io-dtype vector) or 3 (additive fp32 vector); got {cfg.KV_MASK_FORM}",
        ),
        (
            cfg.BIAS_EMPTY_ARRIVERS == cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_WG_WARPS,
            f"{flavor}: BIAS_EMPTY_ARRIVERS must equal the softmax warp count {cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_WG_WARPS} (one elected arrive per "
            f"softmax warp per work item; got {cfg.BIAS_EMPTY_ARRIVERS}) -- an unreachable mbarrier count is a hang at EVERY shape",
        ),
        (
            cfg.DTYPE_QKV in (_DTYPE_BF16, _DTYPE_FP16),
            f"{flavor}: the bias / kv_mask arms are wired on the f16/bf16 kernel only (dtype_qkv={cfg.DTYPE_QKV})",
        ),
        (
            not (cfg.THD_VARLEN or cfg.SPLIT_KV > 1 or cfg.PACK_GQA or getattr(cfg, "EPILOGUE_GATE", 0)),
            f"{flavor}: the bias / kv_mask arms serve dense, unsplit, unpacked, ungated graphs only",
        ),
        (
            cfg.BIAS_MAX_FORM in BIAS_MAX_FORMS,
            f"{flavor}: BIAS_MAX_FORM must be 0 (the software row max) or 1 (the hardware upper bound); got {cfg.BIAS_MAX_FORM}",
        ),
        (
            cfg.BIAS_MAX_FORM != BIAS_MAX_HW_UPPER_BOUND or cfg.KV_MASK_FORM in (KV_MASK_NONE, KV_MASK_BOOL_WORD),
            f"{flavor}: the hardware upper-bound row max (BIAS_MAX_FORM=1) bounds the bias tile and ignores the bit-word mask; an additive kv_mask "
            f"vector (KV_MASK_FORM={cfg.KV_MASK_FORM}) is unbounded by it -- the body would run a max BELOW the true row max: P > 1, an overflowing "
            f"exp2 at a large positive vector",
        ),
    ]
    if cfg.BIAS_MODE:
        tile_bytes = cfg.BIAS_BLOCK_ROWS * cfg.TILE_N * cfg.BPE
        preds += [
            (
                cfg.BIAS_MODE != BIAS_MODE_RESIDENT or (cfg.BIAS_RESIDENT_TILES >= 1 and cfg.STAGES_BIAS == 0),
                f"{flavor}: a resident bias block needs at least one {cfg.TILE_N}-column tile and no ring (BIAS_RESIDENT_TILES={cfg.BIAS_RESIDENT_TILES}, "
                f"STAGES_BIAS={cfg.STAGES_BIAS})",
            ),
            (
                cfg.BIAS_MODE not in BIAS_STREAMED_MODES or (cfg.STAGES_BIAS in BIAS_RING_STAGES and cfg.BIAS_RESIDENT_TILES == 0),
                f"{flavor}: the streamed bias ring is {sorted(BIAS_RING_STAGES)} stages of one {cfg.TILE_N}-column tile and holds no resident block "
                f"(STAGES_BIAS={cfg.STAGES_BIAS}, BIAS_RESIDENT_TILES={cfg.BIAS_RESIDENT_TILES}); the TMA-LDG side pre-arms STAGES_BIAS fills and the "
                f"softmax side advances the same count -- the two sides read ONE Cfg field",
            ),
            # The MULTICAST ring (BIAS_MODE 3): a cluster of two independent CTAs along the chunk axis that share one bias tile sequence.
            (
                (cfg.BIAS_MCAST_CTAS == BIAS_MCAST_CTAS) == (cfg.BIAS_MODE == BIAS_MODE_STREAMED_MULTICAST) and cfg.BIAS_MCAST_CTAS in (1, BIAS_MCAST_CTAS),
                f"{flavor}: BIAS_MCAST_CTAS must be {BIAS_MCAST_CTAS} on the multicast ring (BIAS_MODE 3) and 1 on every other form (got BIAS_MCAST_CTAS="
                f"{cfg.BIAS_MCAST_CTAS}, BIAS_MODE={cfg.BIAS_MODE}): it is the launch's cluster extent along the chunk axis, the TMA multicast fan-out "
                f"and the factor on the ring's release count at once -- a mismatch launches a cluster the body does not credit (a hang at EVERY shape)",
            ),
            (
                cfg.BIAS_MODE != BIAS_MODE_STREAMED_MULTICAST or cfg.N_CHUNK > 0,
                f"{flavor}: the multicast bias ring is the (n, n+1) PAIR item's (N_CHUNK={cfg.N_CHUNK}): only the pair item's two CTAs of a cluster read "
                f"the SAME 128-row bias tile sequence (same q tile, same head, adjacent chunks); the one-batch-entry item's CTAs read different rows",
            ),
            (
                cfg.BIAS_MODE != BIAS_MODE_STREAMED_MULTICAST or (cfg.CTA_MMA == 1 and cfg.CGA_M == 1 and cfg.CGA_N == 1),
                f"{flavor}: the multicast bias ring runs two INDEPENDENT CTAs (CTA_MMA={cfg.CTA_MMA}, CGA_M={cfg.CGA_M}, CGA_N={cfg.CGA_N} must be 1): no "
                f"collective MMA, no shared K / V ring -- every collective-geometry count stays its cga1 value",
            ),
            (
                cfg.BIAS_MODE != BIAS_MODE_STREAMED_MULTICAST or cfg.SCHEDULER_POLICY == SCHED_NATURAL,
                f"{flavor}: the multicast bias ring pairs chunks (c, c + 1) through the NATURAL payload's batch field (SCHEDULER_POLICY="
                f"{cfg.SCHEDULER_POLICY}); the LPT decodes carry no cluster-rank arm",
            ),
            (
                cfg.BIAS_MODE != BIAS_MODE_STREAMED_MULTICAST or (cfg.SEQ_KV_LENS_PRESENT == 0 and cfg.SEQ_Q_LENS_PRESENT == 0),
                f"{flavor}: the multicast bias ring streams ONE kv-tile sequence to both CTAs of a cluster, so their step bounds must be the same "
                f"function of (q tile, S_q, S_kv): per-row lengths (SEQ_KV_LENS_PRESENT={cfg.SEQ_KV_LENS_PRESENT}, SEQ_Q_LENS_PRESENT="
                f"{cfg.SEQ_Q_LENS_PRESENT}) would make the two chunks' sequences differ -- served with global lengths only",
            ),
            (
                cfg.BIAS_BLOCK_ROWS == (cfg.TILE_M if cfg.N_CHUNK > 0 else cfg.TILES_Q * cfg.TILE_M),
                f"{flavor}: BIAS_BLOCK_ROWS must be the Q rows the work item's sub-tiles READ: TILE_M = {cfg.TILE_M} on the (n, n+1) pair item "
                f"(N_CHUNK={cfg.N_CHUNK} > 0: both sub-tiles are rows of ONE q tile and share the block -- a 256-row block would feed row n's bias "
                f"into row n + 1's sub-tile, finite and wrong, no crash), TILES_Q x TILE_M = {cfg.TILES_Q * cfg.TILE_M} on the one-batch-entry item "
                f"(got {cfg.BIAS_BLOCK_ROWS})",
            ),
            (
                cfg.N_CHUNK >= 0,
                f"{flavor}: N_CHUNK must be >= 0 (0 = the one-batch-entry item; got {cfg.N_CHUNK})",
            ),
            (
                cfg.N_CHUNK == 0 or cfg.CTA_MMA == 1,
                f"{flavor}: the (n, n+1) pair item runs cga1 only (CTA_MMA={cfg.CTA_MMA}): a cta_group::2 collective MMA shares K across both CTAs' rows, "
                f"and rows n / n + 1 are DIFFERENT batch entries with different K / V",
            ),
            (
                cfg.N_CHUNK == 0 or cfg.TILES_Q == cfg.SOFTMAX_WARPGROUPS,
                f"{flavor}: the pair item runs one softmax warp group per sub-tile (TILES_Q={cfg.TILES_Q}, SOFTMAX_WARPGROUPS={cfg.SOFTMAX_WARPGROUPS}): "
                f"sub-tile qs is row n + qs of the chunk",
            ),
            (
                cfg.N_CHUNK == 0 or not cfg.BOTTOM_RIGHT,
                f"{flavor}: the pair item's per-step KV range is the UNION of two rows' top-left bands (monotone in the KV length); the bottom-right "
                f"diagonal differs per row -- not served (the graph contract refuses bottom-right causal with a bias anyway)",
            ),
            (
                cfg.N_CHUNK == 0 or not (cfg.THD_VARLEN or cfg.SPLIT_KV > 1 or cfg.PACK_GQA or getattr(cfg, "EPILOGUE_GATE", 0)),
                f"{flavor}: the pair item is the dense, unsplit, non-PackGQA, ungated arm (the bias arm's own gates)",
            ),
            (
                cfg.BIAS_SWZ_BYTES == BIAS_SWZ_BYTES and cfg.BIAS_BOXES_PER_TILE == (cfg.TILE_N * cfg.BPE) // BIAS_SWZ_BYTES,
                f"{flavor}: the bias block is TMA-written AND lane-read under the 128-B swizzle (BIAS_SWZ_BYTES={cfg.BIAS_SWZ_BYTES}); a tile is "
                f"TILE_N x BPE / 128 = {(cfg.TILE_N * cfg.BPE) // BIAS_SWZ_BYTES} boxes of 64 columns (got {cfg.BIAS_BOXES_PER_TILE})",
            ),
            (
                cfg.BIAS_BLOCK_ROWS <= 256,
                f"{flavor}: a TMA box holds at most 256 rows; the block's {cfg.BIAS_BLOCK_ROWS} rows need a per-sub-tile box",
            ),
            (
                operand_region_end_bytes(cfg) <= TCGEN05_V0_ADDR_LIMIT,
                f"{flavor}: the descriptor-fed sQ / sO / sK / sV region ends at {operand_region_end_bytes(cfg)} B, past the 256 KiB version-0 tcgen05 "
                f"descriptor window: a root at or above 256 KiB wraps to offset 0 (the accumulator is EXACTLY zero); the bias block is placed after "
                f"them and may straddle the line, the operands may not",
            ),
            (
                bias_smem_tiles(cfg) * tile_bytes <= arch.smem_usable_bytes,
                f"{flavor}: {bias_smem_tiles(cfg)} bias tiles x {tile_bytes} B exceed the carveout on their own",
            ),
        ]
    else:
        preds += [
            (
                cfg.STAGES_BIAS == 0 and cfg.BIAS_RESIDENT_TILES == 0 and cfg.BIAS_MCAST_CTAS == 1,
                f"{flavor}: bias geometry fields must be 0 (BIAS_MCAST_CTAS 1) without BIAS_MODE",
            )
        ]
    _check(preds)


def validate_q_in_tmem(cfg, flavor: str, *, arch: D128FamilyArch) -> None:
    """Q resident in TMEM (Cfg.Q_IN_TMEM) and everything the d128-file body couples to it (frost-kernels.md s 3 / s 6, sdpa-invariants s 6): the
    Q slots' place in the TMEM map, the ``tcgen05.cp 128x128b`` geometry the MMA warp issues (``Q_COLS / 4`` calls of 16 B per lane, each a
    4-column destination), the dtype the column count assumes, and the one body geometry the arm was audited on -- cga1, the 128-row tile,
    two sub-tiles.  Every message names the failure the predicate prevents."""
    q = int(getattr(cfg, "Q_IN_TMEM", 0))
    lay = d128_family_tmem_layout(cfg, total_cols=arch.tmem_total_cols)
    preds = [
        (q in SMALL_D_Q_IN_TMEM, f"{flavor}: Q_IN_TMEM must be 0 (Q from SMEM, every committed record) or 1 (Q resident in TMEM); got {q}"),
    ]
    if q:
        preds += [
            (
                cfg.DTYPE_QKV in (_DTYPE_BF16, _DTYPE_FP16),
                f"{flavor}: Q in TMEM packs two 16-bit elements per fp32 column (Q_COLS = TILE_K x BPE / 4 = {lay.Q_COLS}); a 4-byte Q is a different column "
                f"count and a different A-operand k-step advance (DTYPE_QKV={cfg.DTYPE_QKV}); Rubin has no TF32 prefill kernel",
            ),
            (
                cfg.TILE_K <= 64,
                f"{flavor}: Q in TMEM is wired on the f16/bf16 32- / 64-wide tiles only (TILE_K={cfg.TILE_K}): at TILE_K 128 the Q slots end at column "
                f"{lay.Q_OFF[-1] + lay.Q_COLS}, past the STATS region at {lay.STATS_OFF} (the committed d128 rendering keeps its record)",
            ),
            (
                cfg.TILE_M == 128,
                f"{flavor}: the tcgen05.cp 128x128b shape copies 128 rows per call; the {cfg.TILE_M}-row Q tile needs the 64x128b shape, which no lane has "
                f"pinned -- Q in TMEM is declined on the 64-row tile until it is",
            ),
            (
                cfg.CTA_MMA == 1,
                f"{flavor}: a tcgen05.cp.cta_group::2 is the cluster SELF-FILL form (both CTAs' MMA warps issue it, each copies its own SMEM); the arm was "
                f"audited at cga1 only (CTA_MMA={cfg.CTA_MMA})",
            ),
            (
                cfg.TILES_Q == 2 and cfg.SOFTMAX_WARPGROUPS == 2,
                f"{flavor}: Q in TMEM serves the two-sub-tile body (TILES_Q={cfg.TILES_Q}, SOFTMAX_WARPGROUPS={cfg.SOFTMAX_WARPGROUPS}): the third softmax "
                f"warp group's third Q slot and third UTCCP block are outside the arm's census",
            ),
            (
                int(getattr(cfg, "QO_ALIAS", 0)) == 0,
                f"{flavor}: under the Q u O alias the TMA-STG warp gates the Q reload (mb_q_o_alias), not the MMA commit the early release moves; the alias "
                f"is not served with Q in TMEM",
            ),
            (
                not (cfg.THD_VARLEN or cfg.SPLIT_KV > 1 or cfg.PACK_GQA or getattr(cfg, "EPILOGUE_GATE", 0)),
                f"{flavor}: Q in TMEM serves dense, unsplit, non-PackGQA, ungated graphs only (the small-tile rows' own gates)",
            ),
            (
                (cfg.TILE_K * cfg.BPE) % 16 == 0 and lay.Q_COLS % 4 == 0 and (cfg.TILE_M * cfg.TILE_K * cfg.BPE) % (128 * 16) == 0,
                f"{flavor}: the Q tile must be whole 128x128b UTCCP calls (16 B per lane x 128 lanes = 4 fp32 columns each): TILE_K x BPE = {cfg.TILE_K * cfg.BPE} B "
                f"per row, Q_COLS={lay.Q_COLS}",
            ),
            (
                lay.Q_OFF[-1] + lay.Q_COLS <= lay.STATS_OFF,
                f"{flavor}: the Q slots {lay.Q_OFF} (+{lay.Q_COLS}) must end at or before the STATS region (column {lay.STATS_OFF} of {lay.TOTAL_COLS}): a Q slot under "
                f"the softmax stats is the alpha / (max, sum) store of a sub-tile overwriting BMM1's own operand",
            ),
            (
                all(off % 8 == 0 for off in lay.Q_OFF),
                f"{flavor}: the A-from-TMEM operand advances 8 columns per 16-wide k-step, so every Q slot base must be 8-column aligned (got {lay.Q_OFF})",
            ),
        ]
    _check(preds)


def validate_softmax_wgs(cfg, flavor: str) -> None:
    """The softmax warp-group count and everything the d128-file body couples to it (frost-kernels.md s2 / s4, sdpa-invariants s6): one
    group per Q sub-tile, the warp table, the scheduler credit count, the register split against the launch's entry pool, and the
    serving set of the third group (the (n, n+1) pair item at TILE_N 64 on the f16/bf16 32- / 64-wide tiles at cga1 -- the one form whose
    roles loop over the sub-tiles; the one-batch-entry paths hand-unroll two).  Each predicate names the failure it prevents."""
    wgs = cfg.SOFTMAX_WARPGROUPS
    split_sum = wgs * cfg.SOFTMAX_WG_WARPS * cfg.SOFTMAX_REGS + cfg.CORRECTION_WARPS * cfg.CORRECTION_REGS + D128_OTHER_WARPS * cfg.OTHER_REGS
    pool = _reg_entry_pool(cfg.TOTAL_WARPS)
    _check(
        [
            (wgs in SMALL_D_SOFTMAX_WGS, f"{flavor}: SOFTMAX_WARPGROUPS must be one of {sorted(SMALL_D_SOFTMAX_WGS)} (got {wgs})"),
            (
                cfg.TILES_Q == wgs,
                f"{flavor}: one softmax warp group per Q sub-tile (TILES_Q={cfg.TILES_Q}, SOFTMAX_WARPGROUPS={wgs}): the body indexes its S / P / O "
                f"slots, its per-sub-tile rings and its warp-group bases by the same index",
            ),
            (
                cfg.SOFTMAX_WG_WARPS == D128_SOFTMAX_WG_WARPS and cfg.CORRECTION_WARPS == D128_CORRECTION_WARPS and cfg.OTHER_WARPS == D128_OTHER_WARPS,
                f"{flavor}: the d128-file body is 4 warps per softmax group, 4 correction warps and 4 single-warp roles",
            ),
            (
                cfg.TOTAL_WARPS == d128_total_warps(wgs) and cfg.THREADS_PER_CTA == cfg.TOTAL_WARPS * 32,
                f"{flavor}: {wgs} softmax warp groups are a {d128_total_warps(wgs)}-warp body (got TOTAL_WARPS={cfg.TOTAL_WARPS}, THREADS_PER_CTA={cfg.THREADS_PER_CTA})",
            ),
            (
                cfg.SOFTMAX_WG1_BASE == cfg.SOFTMAX_WG0_BASE + cfg.SOFTMAX_WG_WARPS
                and cfg.CORR_WARP_BASE == cfg.SOFTMAX_WG0_BASE + wgs * cfg.SOFTMAX_WG_WARPS
                and cfg.MMA_WARP_ID == cfg.CORR_WARP_BASE + cfg.CORRECTION_WARPS
                and (cfg.TMALDG_WARP_ID, cfg.TMASTG_WARP_ID, cfg.SCHED_WARP_ID) == (cfg.MMA_WARP_ID + 1, cfg.MMA_WARP_ID + 2, cfg.MMA_WARP_ID + 3)
                and cfg.SCHED_WARP_ID == cfg.TOTAL_WARPS - 1,
                f"{flavor}: the role table must be contiguous softmax groups (WG0 at {cfg.SOFTMAX_WG0_BASE}, 4 warps each; the kernel derives WG2's first "
                f"warp as WG0 + 8), then correction, MMA, TMA-LDG, TMA-STG, scheduler (got WG1 {cfg.SOFTMAX_WG1_BASE}, corr {cfg.CORR_WARP_BASE}, MMA "
                f"{cfg.MMA_WARP_ID}, LDG {cfg.TMALDG_WARP_ID}, STG {cfg.TMASTG_WARP_ID}, sched {cfg.SCHED_WARP_ID} of {cfg.TOTAL_WARPS}) -- a wrong id "
                f"puts a warp into another role's body",
            ),
            (
                cfg.READ_TILE_ARRIVERS == d128_read_tile_arrivers(wgs),
                f"{flavor}: READ_TILE_ARRIVERS must be {d128_read_tile_arrivers(wgs)} for {wgs} softmax warp groups (got {cfg.READ_TILE_ARRIVERS}); "
                f"an unreachable scheduler mbarrier count is a hang at EVERY shape",
            ),
            (
                split_sum <= pool,
                f"{flavor}: the per-role register split {wgs} x {cfg.SOFTMAX_WG_WARPS} x {cfg.SOFTMAX_REGS} + {cfg.CORRECTION_WARPS} x {cfg.CORRECTION_REGS} + "
                f"{D128_OTHER_WARPS} x {cfg.OTHER_REGS} = {split_sum} exceeds the {cfg.TOTAL_WARPS}-warp launch's entry pool {pool}: setmaxnreg.inc can "
                f"only draw what the .dec warps released, so the last INCREASE warp parks forever -- a HANG at 100 % SM utilization at every shape",
            ),
            (
                wgs == 2
                or (
                    cfg.N_CHUNK > 0
                    and cfg.TILE_N == 64
                    and cfg.CTA_MMA == 1
                    and cfg.TILE_K <= 64
                    and cfg.DTYPE_QKV in (_DTYPE_BF16, _DTYPE_FP16)
                    and cfg.BIAS_MODE != BIAS_MODE_NONE
                    and cfg.TILE_M == 128
                ),
                f"{flavor}: the third softmax warp group serves the (n, n+1) pair item (N_CHUNK > 0) of the bias arm at TILE_N 64 on the f16/bf16 32- / "
                f"64-wide tiles at cga1 and the 128-row Q tile only (got N_CHUNK={cfg.N_CHUNK}, TILE_N={cfg.TILE_N}, CTA_MMA={cfg.CTA_MMA}, TILE_K={cfg.TILE_K}, TILE_M={cfg.TILE_M}, "
                f"DTYPE_QKV={cfg.DTYPE_QKV}, BIAS_MODE={cfg.BIAS_MODE}): the one-batch-entry paths hand-unroll two sub-tiles and the 128-register "
                f"softmax holds one 64-column chunk",
            ),
        ]
    )


def validate_small_tile_masks(cfg, flavor: str) -> None:
    """The two mask-arm couplings of the small-tile forms (the dead-sub-tile quiet body, the pre-masked padded tail): each needs the ONE
    body geometry it was built on, and a form kept on another silently changes a barrier sequence or a masked segment."""
    _check(
        [
            (
                not (
                    int(getattr(cfg, "DEAD_SUBTILE_QUIET", 0)) == 1
                    and (int(getattr(cfg, "N_CHUNK", 0)) or cfg.PACK_GQA or cfg.THD_VARLEN or int(getattr(cfg, "EPILOGUE_GATE", 0)))
                ),
                f"{flavor}: DEAD_SUBTILE_QUIET=1 (the dead-sub-tile quiet body) needs the one-batch-entry item without PackGQA, THD or the gated "
                f"epilogue -- the quiet body replicates its live body's barrier sequence (the plain row mapping's on the Rubin file, the -inf body's "
                f"with the ring handshake on the SM100 file; each factory admits the half its file renders) (got N_CHUNK={getattr(cfg, 'N_CHUNK', 0)}, "
                f"PACK_GQA={cfg.PACK_GQA}, THD_VARLEN={cfg.THD_VARLEN}, EPILOGUE_GATE={getattr(cfg, 'EPILOGUE_GATE', 0)})",
            ),
            (
                not (int(getattr(cfg, "PADDED_TAIL_FORM", 0)) == 1 and (cfg.MASK_FLAGS != MASK_PADDED or int(getattr(cfg, "KV_MASK_FORM", 0)))),
                f"{flavor}: PADDED_TAIL_FORM=1 (the pre-masked tail) needs MASK_FLAGS == MASK_PADDED alone and no kv_mask -- the plain body's single "
                f"HW-max loop and the bias body's MASK_NONE -inf loop have no masked segment for a band, and a kv_mask folds the length into its keep "
                f"words already (got MASK_FLAGS={cfg.MASK_FLAGS}, KV_MASK_FORM={getattr(cfg, 'KV_MASK_FORM', 0)}); a band's masked tail silently kept "
                f"would read -inf on no column and the row max would see the padding",
            ),
        ]
    )
