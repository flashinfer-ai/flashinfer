# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel configuration for the Frost SM107 (Rubin) DSL SDPA-forward flavors.

**Why this is a separate module from** :mod:`cudnn.sdpa.fwd.config_sm100`, and
not a widening of it: the Rubin kernels are a different kernel *lineage*, not
the Blackwell kernels recompiled. They implement different pipelines, so they
need different warp populations, stage depths and MMA k-steps. The two module's
flavors overlap in NAME (``d128`` / ``d256`` / ``d512``) and in almost nothing
else, and the overlap is what makes sharing dangerous rather than economical.

That is not a theoretical concern — it cost a debugging session. The SM107
``d256`` per-tensor FP8 kernel was routed through ``make_cfg_d256`` (Blackwell),
which derives its whole warp layout from a *split-P* switch that only the
Blackwell d256 kernel implements::

    SOFTMAX_WARPGROUPS   rubin-body=1     blackwell-cfg=2
    TOTAL_WARPS          rubin-body=12    blackwell-cfg=16
    THREADS_PER_CTA      rubin-body=384   blackwell-cfg=512
    MMA_WARP_ID          rubin-body=8     blackwell-cfg=12
    READ_TILE_ARRIVERS   rubin-body=11    blackwell-cfg=15

The last line is the fatal one: ``READ_TILE_ARRIVERS`` is the init count of the
scheduler's tile-handoff mbarrier, so a count of 15 against 11 actual arrivers
can never be reached and the *first* tile handoff never completes. The kernel
hung at every shape, down to ``B=1 H=1 S=128`` — 100 % GPU, no fault, no wrong
number, nothing for a sanitizer to find. Keeping Rubin's geometry in its own
module is what makes that class of mismatch impossible rather than merely
unlikely.

Provenance: every value here is the one the Rubin kernel BODIES were written and
validated against (the pre-upstream ``sdpa_config_{llama,dsv3,qwen,
dsv4}`` flavors, keyed here by op geometry per the engine contract, not by model
name). ``test_sdpa_fwd_config_sm107.py`` pins the ones a kernel body would
deadlock or corrupt on if they drifted.

Shared with SM100 by design: :class:`TemplateParams` (the engine-contract record
the loader injects) and the mask / dtype / scheduler vocabulary. Those are the
interface; everything below is the Rubin geometry behind it.

All configuration errors raise :class:`ValueError`. Anything a *user-built
graph* could trip must be rejected earlier, by ``graph_analyzer.probe`` /
``api_dsl.SdpaFwdDslSm100.check_support`` — a ``ValueError`` from here means
those support checks have a gap.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclasses_replace
from typing import Optional, Tuple


from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.constants import (
    MASK_CAUSAL,
    MASK_NONE,
    MASK_PADDED,
    MASK_SWA,
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
)

# The engine-contract record is deliberately SHARED: one lowering builds it, and
# an arch-specific copy would let the two drift in ways the loader cannot see.
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import TemplateParams, bshd_compact, bshd_zero_copy_stride

# The d512 2x2-DATAPATH record is shared with the SM100 line too (CfgD512X2 + its arch-neutral geometry checks); this
# module adds only the Rubin arch facts to it (3-deep sub-chunk rings, DESC_VERSION derived from the layout, the 320 KiB
# usable budget) -- see make_cfg_d512_2x2 below.
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import (
    CfgD512X2,
    _pack_g,
    _validate_params as _validate_params_sm100,
    d512_2x2_geometry_checks,
    d512_2x2_p_ring_start_bytes,
    d512_2x2_smem_bytes,
)

# The d256 DECODE record (the swap-AB tile's CfgD256Decode + its arch-neutral predicates) is shared the same way: the
# Rubin twin below re-validates the SM100 factory's record against the Rubin SMEM carveout and the version-0 tcgen05
# descriptor window -- see make_cfg_d256_decode at the end of this module.
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import CfgD256Decode, make_cfg_d256_decode as _make_cfg_d256_decode_record
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import D256_DECODE_MAX_Q_ROWS, decode_d256_q_units

# The per-CTA register POOL a `setmaxnreg` split redistributes (the launch's entry count x the warps, NOT the 2048-register file): the
# backward config's helper, the ONE arithmetic behind the d128-file predicate `SUM(role regs x warps) <= pool` (frost-kernels.md s2: a
# split over the pool parks the last INCREASE warp forever -- a hang at 100 % SM utilization at every shape, no fault).
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.bwd.config_sm107 import reg_entry_pool as _reg_entry_pool

# The arch-NEUTRAL half of the d128-file small-tile configuration (the bias / kv_mask / pair / ring / max-form / tail / quiet /
# S_SLOTS / Q_IN_TMEM / softmax-WG field block, the TMEM map formula, the budgets and the validators, each taking the arch BUDGET as a
# parameter) lives in config_d128_family; every lifted name is re-exported here under its old spelling so the kernel, the heuristics,
# the engine rows, the adapter and the tests import nothing new, and every `sm107_*` budget below is the shared arithmetic at SM107_ARCH.
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_d128_family import (  # noqa: F401 -- re-exports
    BIAS_MAX_FORMS,
    BIAS_MAX_HW_UPPER_BOUND,
    BIAS_MAX_SOFTWARE,
    BIAS_MCAST_CTAS,
    BIAS_MODE_NONE,
    BIAS_MODE_RESIDENT,
    BIAS_MODE_STREAMED,
    BIAS_MODE_STREAMED_MULTICAST,
    BIAS_MODES,
    BIAS_PRELOAD_CHUNKS_DOMAIN,
    BIAS_RING_STAGES,
    BIAS_STREAMED_MODES,
    BIAS_SWZ_BYTES,
    D128_REG_SPLIT as _D128_REG_SPLIT,
    D128_CORRECTION_WARPS as _D128_CORRECTION_WARPS,
    D128_OTHER_WARPS as _D128_OTHER_WARPS,
    D128_SOFTMAX_WG_WARPS as _D128_SOFTMAX_WG_WARPS,
    D128FamilyArch,
    D128FamilySmallTileFields,
    D128FamilyTmemLayout,
    DEAD_SUBTILE_QUIET_FORMS,
    KV_MASK_ADDITIVE_16,
    KV_MASK_ADDITIVE_F32,
    KV_MASK_ADDITIVE_GRANULE_ELEMS,
    KV_MASK_BOOL_WORD,
    KV_MASK_FORMS,
    KV_MASK_NONE,
    PADDED_TAIL_FORMS,
    SMALL_D_Q_IN_TMEM,
    SMALL_D_S_SLOTS,
    SMALL_D_SOFTMAX_WGS,
    SMALL_D_STAGES_KV,
    SMALL_D_TILE_MS,
    TCGEN05_V0_ADDR_LIMIT,
    _align_slab,
    _check,
    bias_block_rows as _bias_block_rows,
    bias_pair_form as _bias_pair_form,
    bias_resident_block_bytes,
    bias_resident_tiles as _bias_resident_tiles,
    bias_smem_tiles,
    d128_family_tmem_layout as _d128_family_tmem_layout,
    d128_read_tile_arrivers as _d128_read_tile_arrivers,
    d128_total_warps as _d128_total_warps,
    o_swz_bytes,
    operand_region_end_bytes,
    q_swz_bytes,
    sdpa_smem_bytes,
    v_swz_bytes,
    validate_bias as _validate_bias_shared,
    validate_head_dim_tile as _validate_head_dim_tile_shared,
    validate_q_in_tmem as _validate_q_in_tmem_shared,
    validate_regs as _validate_regs,
    validate_smem as _validate_smem_shared,
    validate_small_tile_masks as _validate_small_tile_masks,
    validate_softmax_wgs as _validate_softmax_wgs,
    validate_swizzles as _validate_swizzles,
    validate_tile_n as _validate_tile_n_shared,
    validate_warp_layout as _validate_warp_layout,
)

__all__ = [
    "TemplateParams",
    "resolve_dtype_o",
    "TmaIters",
    "CfgD32",
    "CfgD64",
    "CfgD128",
    "CfgD192",
    "CfgD256",
    "CfgD512",
    "CfgD512X2",
    "make_cfg_d32",
    "make_cfg_d64",
    "make_cfg_d128",
    "make_cfg_d128_mxfp8",
    "make_cfg_d192",
    "make_cfg_d192_mxfp8",
    "make_cfg_d256",
    "make_cfg_d256_mxfp8",
    "make_cfg_d512",
    "make_cfg_d512_mxfp8",
    "make_cfg_d512_2x2",
    "CfgD256Decode",
    "make_cfg_d256_decode",
    "d256_decode_smem_layout",
    "d256_decode_desc_version",
    "D256_DECODE_SINGLE_UNIT_Q_TILES",
    "D256_DECODE_ROUTED_MAX_Q_ROWS",
    "D256_DECODE_ROUTED_MAX_TOKEN_UNITS",
    "D256_DECODE_MAX_Q_ROWS_SM107",
    "decode_d256_wide_tile_admits",
    "decode_d256_route_widths",
    "decode_d256_single_unit_q_tile",
    "decode_d256_q_tile",
    "SMEM_CAP_BYTES",
    "SMEM_USABLE_BYTES",
    "SM107_FP8_THD_SHAPES",
    "SM107_FP8_SPLIT_SHAPES",
    "SM107_FP8_PACK_GQA_SHAPES",
    "SM107_F16_THD_SHAPES",
    "SM107_EPILOGUE_GATE_SHAPES",
    "SM107_SCORE_STATS_SHAPES",
    "SM107_HEAD_DIM_TILE_WIDTHS",
    "SM107_F16_SMALL_D_SHAPES",
    "SM107_SMALL_D_CGAS",
    "SMALL_D_STAGES_KV",
    "SMALL_D_TILE_MS",
    "SMALL_D_SOFTMAX_WGS",
    "SM107_THREE_WG_S_KV_MIN",
    "SM107_THREE_WG_B_FLAT_MAX",
    "sm107_softmax_wgs_pick",
    "SMALL_D_S_SLOTS",
    "sm107_s_slots_pick",
    "SMALL_D_Q_IN_TMEM",
    "sm107_q_in_tmem_pick",
    "sm107_f16_tile_width",
    "sm107_small_tile_leg",
    "D128FamilyTmemLayout",
    "d128_family_tmem_layout",
    "epilogue_gate_layout_declarable",
    "TCGEN05_V0_ADDR_LIMIT",
    "d256_mxfp8_sf_smem_bytes",
    "d256_mxfp8_last_sf_tile_start",
    "CfgD256Sparse",
    "make_cfg_d256_sparse",
    "d256_sparse_smem_layout",
    "sparse_entry_regs",
    "SPARSE_D256_WIRED_ARMS",
    "SMEM_STANDARD_CARVEOUT_BYTES",
    "CfgD256SparseLean",
    "TemplateParamsLean",
    "make_cfg_d256_sparse_lean",
    "lean_sparse_n_sub",
    "lean_sparse_split",
    "lean_sparse_n_mt",
    "lean_sparse_smem_bytes",
    "LEAN_SPARSE_N_KEYS",
    "LEAN_SPARSE_MAX_SUB_TILES_PER_CTA",
    "LEAN_SPARSE_MAX_SEQLEN_Q",
]


# ---------------------------------------------------------------------------
# Rubin architecture constants
# ---------------------------------------------------------------------------

# Rubin SM10.7 oversized SMEM cap.  Reaching past the standard 227 KiB requires
# the launcher to set ALLOW_OVERSIZED_SHARED_MEMORY; the d128 FP8 9-stage ring
# and the d256/d512 flavors all depend on it.
SMEM_CAP_BYTES = 327 * 1024

# ...but the CAPACITY is not the budget. The shipped sm107/prefill_d128_fp8.py
# sizes its own guard against 320 KiB ("327 KiB capacity minus reserves"), and
# the kernels additionally spend ~2 KiB on barriers, the scheduler ring and the
# TMEM pointer. Validating Q/K/V/O against the raw 327 KiB waves through an
# allocation that silently overflows: d192 f16 at STAGES_KV=4 sums to exactly
# 320 KiB of Q/K/V/O, passed a 327 KiB check, and produced an output that was
# 50 % ZEROS with a numerically EXACT LSE (BMM1 and the softmax are fine; only
# the clobbered O half is wrong). Check against this, not the capacity.
SMEM_USABLE_BYTES = 320 * 1024
_SMEM_FIXED_OVERHEAD = 2 * 1024  # barriers + scheduler + tmem-ptr slack

# TMEM columns.  Rubin raises the Blackwell 512 to 576, which needs
# `tmem_alloc(..., is_exclusive=True)` (see tile_dsl/tmem.py).
TMEM_TOTAL_COLS = 576

# The Rubin BUDGET the shared d128-family budgets / validators take as a parameter (config_d128_family): the three numbers above and
# the exclusive-allocation flag, named "Rubin" in every message exactly as before the lift.
SM107_ARCH = D128FamilyArch(
    name="Rubin", smem_usable_bytes=SMEM_USABLE_BYTES, smem_fixed_overhead=_SMEM_FIXED_OVERHEAD, tmem_total_cols=TMEM_TOTAL_COLS, tmem_exclusive_alloc=True
)

_DTYPE_E4M3, _DTYPE_E5M2, _DTYPE_BF16, _DTYPE_FP16 = 0, 1, 2, 3
# Output-only block-scaled codes (per-tensor FP8 d128 epilogue; see config_sm100):
# 4 = E2M1 data + E4M3 scale per 16 d, 5 = E4M3 data + UE8M0 scale per 32 d.
_DTYPE_O_NVFP4, _DTYPE_O_MXFP8 = 4, 5
_O_BLOCK_SCALE_BY_DTYPE = {_DTYPE_O_NVFP4: 16, _DTYPE_O_MXFP8: 32}

# Head-dim shapes whose Rubin PER-TENSOR FP8 kernel carries the THD/varlen leg.
#
# ONE definition, consumed by BOTH the engine row (`engines._sm100_fp8_spec`'s
# ``thd_d_shapes`` on the Rubin arm) and the standalone adapter's THD gate
# (``api_dsl.SdpaFwdDslSm100.check_support``).  They are two enforcement points
# for one fact, and contract rule 8b' exists because keeping two copies in step
# by hand does not work: widen only the row and a graph enters the ranked list
# then dies with a bare NotImplementedError in check_support; widen only the
# wrapper and the row still declines.  Sharing the constant makes disagreement
# unrepresentable rather than merely tested.
#
# Membership rule: the shape's body must carry the FROST THD contract -- the
# 14-arg build_thd_meta_o_descs_kernel, 4B+4 metadata, (b+3) O-descriptor slots,
# the persistent claim-counter scheduler, the dead-unit O-store guard and the
# packed-total-clamped runtime K/V descriptors.
#
# All four qualify as of 2026-09-09.  d192xd128 came free -- it IS the shipped
# d128 body, only the config factory differs -- and d256 / d512 were moved onto
# the contract (an internal porting script); they previously called the
# setup kernel with the pre-upstream 7-arg signature and allocated 3B+2 where
# the shared decode reads 4B+4.  Confirmed on a Rubin dev node: 43 passed / 0
# failed across the per-tensor FP8 THD suite.
SM107_FP8_THD_SHAPES = frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})
# Bodies with FP32 partial stores: shared by factory, admission and prepared workspace.
SM107_FP8_SPLIT_SHAPES = frozenset({(64, 64), (128, 128), (192, 128), (256, 256)})
# Packing and splitting are separate domains; both graph and adapter read this one.
SM107_FP8_PACK_GQA_SHAPES = frozenset({(64, 64), (128, 128), (192, 128), (256, 256)})


def supports_fp8_d256_padded_split(d_shape, *, device_cc, pertensor, thd, paged):
    """Native dense D256 FP8 partials honor runtime Q/KV lengths and tails."""
    return device_cc == (10, 7) and pertensor and d_shape == (256, 256) and not thd and not paged


# f16/bf16 flavor names whose kernel body HAS been ported to the FROST
# setup-kernel contract (the 14-arg build_thd_meta_o_descs_kernel + the 4B+4
# metadata the shared decode reads).  Keyed by the `flavor` string
# `_validate_params` already receives, so adding a ported flavor is one entry.
_F16_THD_FLAVORS = frozenset({"sm107 d128", "sm107 d192xd128", "sm107 d256", "sm107 d512"})

# Paged KV (TemplateParams.paged_kv / page_size) on the Rubin line uses the native half D128 and MXFP8 D128/D256
# K/V TMA-LDG sites. Half D256 prefill and the single-Q THD split retain their shared bodies. The page id is the TMA
# batch coordinate; MXFP8 also indexes its SF descriptor by the page and tile-in-page. _validate_params declines
# every other flavor, and each unwired MXFP8 kernel file refuses paged_kv at module scope as well. MXFP8 pages hold
# whole 128-row F8_128x4 SF atom sets. Half pages instead use boxes of at least eight rows within each KV tile.
# Two cc 10.7 HALF kernels read pools natively and validate their OWN records (neither passes through _validate_params,
# so this set does not name them): sm107/decode_d256_f16.py (make_cfg_d256_decode -- the decode-shaped dense-Q d256
# route, the SM100 decode record re-validated for Rubin) and sm107/sparse_d256_f16.py (make_cfg_d256_sparse -- the
# index-list core's gather warps read K / V from the pools through the (B, max_pages) table; page_size a multiple of
# the 4-token block).
_PAGED_KV_FLAVORS_SM107 = frozenset({"sm107 d128", "sm107 d128 mxfp8", "sm107 d256 mxfp8"})

# Head-dim shapes whose Rubin f16/bf16 kernel carries the THD/varlen leg -- the
# same one-definition-two-consumers arrangement as SM107_FP8_THD_SHAPES above
# (engine row + standalone adapter gate; contract rule 8b').  Must stay in step
# with _F16_THD_FLAVORS, which is the same fact keyed by config-flavor name.
SM107_F16_THD_SHAPES = frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})

# MXFP8 flavor names whose kernel body carries the FROST THD contract (the 14-arg
# build_thd_meta_o_descs_kernel setup launch, the 4B+4 metadata, the persistent
# claim-counter scheduler, the dead-unit O-store guard, the packed-total-clamped
# runtime K/V descriptors, the three Stats arms) together with the PACKED
# per-sequence-TILE-padded scale-factor layout (SF descriptors at B = 1 over the
# bound SF tensor's tile extent, every SF load offset by the per-sequence SF-tile
# prefix base).  Keyed by the `flavor` string `_validate_params` receives.  The
# d128 / d192xd128 / d512 MXFP8 bodies still carry the pre-upstream arm (7-arg
# setup call, 3B+2 metadata, static K/V maps, `total_*_sf_tiles` host kwargs the
# prepared pointer host never passes) and are declined here as well as by the
# row -- reaching the raise below is an engine-row bug, not a user error.
_MXFP8_THD_FLAVORS = frozenset({"sm107 d256 mxfp8"})

# Head-dim shapes whose Rubin MXFP8 kernel carries the THD/varlen leg -- the same
# one-definition-many-consumers arrangement as the two sets above (the engine
# row's `thd_d_shapes`, the standalone adapter's Rubin THD gate and its
# `_can_prepare_mxfp8`, the MXFP8 suite's per-shape Rubin skip; contract rule
# 8b').  Must stay in step with _MXFP8_THD_FLAVORS, the same fact keyed by
# config-flavor name.  d256 ONLY: the row serves d256 MXFP8 at cga1, the width
# the THD arm is validated at.
SM107_MXFP8_THD_SHAPES = frozenset({(256, 256)})

# Head-dim shapes whose Rubin kernels carry the FUSED EPILOGUE GATE
# (O := O * sigmoid(G), TemplateParams.epilogue_gate).  ONE named constant with
# three consumers -- the engine rows' ``epilogue_gate_d_shapes``, the standalone
# adapter's rule-8b twin in ``api_dsl.SdpaFwdDslSm100.check_support`` and the
# gated-attention block's geometry pin -- so the three cannot drift (rule 8b').
# EXACT dims: the envelope flavor that pads d=200 up to (256, 256) is NOT
# claimed (the gate tile is TMA'd at the kernel's TILE_O, and nothing has
# validated a padded G).  Membership rule: the flavor's f16/bf16, per-tensor
# FP8 AND block-scale MXFP8 bodies all carry the gate seams (sGate SmemTile,
# mb_gate_full/empty, the TMA-LDG issue after the KV loop, the packed tanh
# epilogue).  The MXFP8 kernel stages a bf16 G like the FP8 one and writes a
# gated e4m3 O UNSCALED (it has no per-tensor scale_o: block scales dequantize
# in-MMA), and its gate tile is declared AFTER the scale-factor slabs so the
# version-0 SF descriptors stay under 256 KiB (see d256_mxfp8_last_sf_tile_start).
SM107_EPILOGUE_GATE_SHAPES = frozenset({(256, 256)})

# Native half flavors whose Rubin PREFILL kernel stores the score side outputs (``score_max`` /
# ``score_sum_exp``, see config_sm100.SM100_SCORE_STATS_SHAPES for the contract): the d128 file
# (``sm107/prefill_d128_f16.py``) from its correction epilogue.  The engine row's
# ``score_stats_d_shapes`` and the adapter's check_support read THIS set (rule 8b').
SM107_SCORE_STATS_SHAPES = frozenset({(128, 128)})

# Head-dim TILE widths of the Rubin f16/bf16 d128-family prefill kernel (``sm107/prefill_d128_f16.py`` at
# ``TemplateParams.d_flavor``): ``TILE_K = TILE_O`` in {32, 64, 128} is a Cfg field, not a fork.  A graph's head dim rides
# the SMALLEST width covering max(d_qk, d_v): the TMA load box is the tile while the tensor extent is D, so the load
# zero-fills the missing columns (exact zero terms in every QK^T dot product and in P.V's padded V columns), BMM1 runs
# TILE_K / 16 k-steps instead of 8, BMM2's N is TILE_O, the O store clips at the true d_v, and the expect_tx byte count is
# the FULL box (an overshooting TMA box delivers every byte of the box, zero-filled).  D % 8 == 0 is the TMA 16-byte
# global-stride rule at 2 B/elem.  The two small shapes are the ROW's native flavors next to d128 (one constant for the
# engine row's d_shapes / cga domain, the adapter's kernel-file map and the tests -- rule 8b'); D > 128 belongs to the
# d192 / d256 / d512 rows.
#
# The small tiles carry the row's DENSE leg only (causal / bottom-right / right band / SWA / padded / sink / GQA / Stats, at
# the cga widths of SM107_SMALL_D_CGAS).  A graph that asks for a leg beyond it -- THD, paged KV, split-KV, PackGQA, the
# score side outputs (``sm107_small_tile_leg``) -- or for a cga outside the tile's domain is NOT declined: the pick walks
# PAST the small tiles to the next covering width, i.e. the d128 flavor the graph rode before the tiles existed (zero-filled
# to 128, where those legs are wired and validated), so nothing the row served before the tiles is withdrawn.  The engine
# row says it as ``Capabilities.dense_only_d_shapes``; the adapter's flavor pool and ``sm107_f16_tile_width(leg=, cga=)``
# are the twins (rule 8b').  Serving a leg natively on a small tile = validating it there and taking the tile out of that set.
SM107_HEAD_DIM_TILE_WIDTHS = (32, 64, 128)
SM107_F16_SMALL_D_SHAPES = frozenset({(32, 32), (64, 64)})
# cga widths the small tiles are VALIDATED at (the engine row's ``cgas_by_d_shape`` reads this; the factory refuses a width
# outside it).  cga1 is the default: the narrow slabs need no collective MMA to halve K/V, and a 512-row cga2 cluster wastes
# 25-50 % of its Q rows at the S = 256-384 crops these tiles exist for.  At TILE_O = 32 a cga2 V operand would be 16 columns
# = a 32-byte MN-major swizzle atom, which no shipped kernel runs -- declined until measured (a cga2 request at D <= 32 lands
# on the 64-wide tile, the smallest covering one validated at that width).
SM107_SMALL_D_CGAS = {(32, 32): frozenset({1}), (64, 64): frozenset({1, 2})}


def supports_native_d128_narrow_tile(d_shape, *, device_cc, half_precision, thd, paged, cga, split_kv=1, has_bias=False, has_kv_mask=False):
    """Whether TILE_N=64 selects the native D128 two-score-slot pipeline.

    Dense CGA1, paged and ragged split plans use different bodies. Keep the
    explicit knob domain shared by graph validation and standalone lowering.
    """
    return (
        device_cc == (10, 7)
        and half_precision
        and d_shape == (128, 128)
        and not paged
        and not has_bias
        and not has_kv_mask
        and ((thd and split_kv == 1 and cga in (None, 1, 2)) or (not thd and cga in (None, 2)))
    )


def supports_native_d128_dense_partial_pack(d_shape, *, device_cc, fp8, thd, paged, cga):
    """Dense half graphs whose selected flavor uses native partial PackGQA.

    The two-CTA D128 prefill body packs a divisor of the head ratio, including
    envelope and split plans. The single-CTA dense route is a different kernel.
    Share this admission boundary between graph and standalone lowering.
    """
    return device_cc == (10, 7) and not fp8 and not thd and not paged and d_shape == (128, 128) and cga in (None, 2)


def supports_native_d512_dense_pack_split(d_shape, *, device_cc, fp8, thd, paged, cga):
    """Dense half D512 uses the native 2x2 body for packed and split plans."""
    return device_cc == (10, 7) and not fp8 and not thd and not paged and d_shape == (512, 512) and cga in (None, 2)


def sm107_small_tile_leg(
    *,
    thd: bool = False,
    paged: bool = False,
    split_kv: Optional[int] = 1,
    pack_gqa: Optional[bool] = False,
    score_stats: bool = False,
    prefolded: bool = False,
) -> Optional[str]:
    """The leg a graph asks for BEYOND the dense one the Rubin small head-dim tiles carry -- ``"THD"``, ``"paged KV"``,
    ``"split-KV"``, ``"PackGQA"``, ``"score side outputs"`` or ``"pre-folded attention scale"`` (the row serves
    ``sdpa(attn_scale_prefolded=True)`` on its d128+ flavors; the small tiles' bias / kv_mask bodies carry no pre-folded arm) --
    or ``None`` for a graph on the dense leg.  A non-None leg
    routes the graph past the small tiles onto the d128 flavor (the pre-tile route); the engine row spells the same predicate
    on its graph facts and knobs (``engines._selected_d_shape`` over ``Capabilities.dense_only_d_shapes``), the adapter calls
    this one, and the tests assert the two agree on every (D, leg) cell (rule 8b')."""
    if thd:
        return "THD"
    if paged:
        return "paged KV"
    if (split_kv or 1) > 1:
        return "split-KV"
    if pack_gqa:
        return "PackGQA"
    if score_stats:
        return "score side outputs"
    if prefolded:
        return "pre-folded attention scale"
    return None


def sm107_f16_tile_width(d_qk: int, d_v: int, *, leg: Optional[str] = None, cga: Optional[int] = None) -> Optional[int]:
    """The head-dim tile width the Rubin f16/bf16 row serves ``(d_qk, d_v)`` on: the smallest of
    :data:`SM107_HEAD_DIM_TILE_WIDTHS` covering both -- 8 / 16 / 24 -> 32, 40 / 48 / 56 -> 64, 72..120 -> 128, and 32 / 64 /
    128 native -- or ``None`` when no width covers them (D > 128: the d192 / d256 / d512 rows) or a dim is not a multiple of 8
    (the TMA 16-byte global-stride rule at 2 B/elem: a typed decline, never a padded copy).

    ``leg`` (:func:`sm107_small_tile_leg`) and ``cga`` make the pick feature-aware: a small tile is taken only for a graph on
    the dense leg (``leg is None``) at a cga inside :data:`SM107_SMALL_D_CGAS` (or unset); otherwise the walk continues to the
    next covering width -- a split-KV / score-output graph at D <= 64 lands on 128 (its pre-tile route), a cga2 request at
    D <= 32 on 64.  The engine row (``d_shapes`` + ``dense_only_d_shapes``, the envelope walk of ``engines._selected_d_shape``)
    and the adapter (``api_dsl._pick_flavor`` over its Rubin pool) implement the SAME pick; the tests assert the three agree for
    every served D and every leg (rule 8b')."""
    d = max(int(d_qk), int(d_v))
    if min(int(d_qk), int(d_v)) <= 0 or int(d_qk) % 8 or int(d_v) % 8:
        return None
    for width in SM107_HEAD_DIM_TILE_WIDTHS:
        if d <= width:
            small = (width, width) in SM107_F16_SMALL_D_SHAPES
            if small and (leg is not None or (cga is not None and cga not in SM107_SMALL_D_CGAS[(width, width)])):
                continue
            return width
    return None


# ---------------------------------------------------------------------------
# Additive bias (resident row-block) + per-key boolean mask on the d128-file tiles
# ---------------------------------------------------------------------------
#
# The bias / kv_mask / ring / row-max vocabulary (BIAS_MODE_*, BIAS_MODES, BIAS_STREAMED_MODES, BIAS_MCAST_CTAS, BIAS_RING_STAGES,
# KV_MASK_*, BIAS_MAX_*, KV_MASK_ADDITIVE_GRANULE_ELEMS) is defined ONCE in config_d128_family (its docstrings carry the forms) and
# re-exported above; the Rubin-MEASURED picks that read it stay in this module.
# The KV tile width (TemplateParams.tile_n) each native flavor of the sm107 f16/bf16 row renders: the small tiles take 64 or 128
# (the one-chunk arm the softmax body carries, validated and trace-compiled at both), the d128 / d192 / d256 / d512 renderings keep
# 128; 256 is refused by _validate_tile_n at two sub-tiles (2 x 256 S columns + the O slots + STATS exceed the 576-column TMEM).
# The engine row's per-shape knob domain (Capabilities.tile_ns_by_d_shape) and the adapter's check_support read this ONE constant.
SM107_TILE_NS_BY_D_SHAPE = (((32, 32), frozenset({64, 128})), ((64, 64), frozenset({64, 128})))
# The KV tile width an fp32 ADDITIVE kv_mask vector TOGETHER WITH a bias is served at on the small tiles: 64, and only 64.  The
# 128-wide rendering of that pair spills on the softmax role (MEASURED, sm_107a, cutlass-dsl 4.8.0: STL / LDL 23 / 27 causal, 15 / 15
# padded, 16 / 16 at d64 cga2 -- the sixteen 16-byte granules of a 64-column fp32 chunk are 64 registers beside the second chunk's
# 128 live scores; BIAS_PRELOAD_CHUNKS = 0 leaves the counts unchanged, a streamed granule fold makes them worse), where the io-dtype
# vector packs two values per register and stays at the bit word's count and the 64-wide tile (64 live scores) fits again.  Read by
# the heuristics' pick, the engine row's gate (a requested 128 is a typed decline, never a degraded plan) and the adapter's default.
SM107_FP32_VECTOR_WITH_BIAS_TILE_N = 64
# The KV tile width the small tiles RUN a graph at when no knob asks for one: the MEASURED table of the knob sweep (bf16, D 32 and
# 64, S_q = S_kv = S in 128..2048 and the padding leg 200..600, B x S flattened rows at B 1 / 4, the bias batch-broadcast; a 212-SM
# cc 10.7 node at the 2376 MHz lock, 100 x 3 graph-replay rounds with a control twin, every ratio same-process, the kernel-only
# CUPTI time where the replay is launch-bound).  The half tile pays where the 128-wide body is over the instruction-cache cliff
# (a bias AND a kv_mask: +9..+55 % at every D / B / S measured through S 768 and d64 through S 2048; at d32 S 1024 / 2048 B = 1 the
# pair ring at the capped chunk is +2.4 / +1.4 % while the wave-quantised n32 / n64 rings tied or lost), at one or two kv tiles with a bias
# (nothing for the wider tile to amortise: +7..+37 %), and on the padded dense / bias body whose 128-wide rendering is
# pathologically slow (S_kv % 128 != 0: plain +16..+68 % at d32, +36 % at d64 S 200; bias +17..+38 %) -- a workaround for a
# code-generation effect of the padded dense arm, to retire with its fix.  Everywhere else the half tile doubles the per-kv-tile
# fixed work and LOSES: plain / kv_mask-only at S % 128 == 0 -3..-20 %, bias-only at S % 128 == 0 >= 384 -3..-13 %, the
# kv_mask-only padded body a wash or a loss (except one kv tile), the d64 padded plain body above 256 -4 %.
# Read by the heuristics' pick, the engine row's form evaluation and the adapter's default (one arithmetic, rule 8b'); a
# REQUESTED width is honoured or declined by the row / the adapter, never degraded.
SM107_SMALL_TILE_N_DEFAULT, SM107_SMALL_TILE_N_NARROW = 128, 64
# the d64 bias-only graph and the d64 padded body take the narrow tile up to this S_kv (one or two 128-wide kv tiles)
SM107_NARROW_TILE_S_KV_MAX = 256
# How many of a kv step's 64-column bias chunks the softmax lanes pre-load BEFORE the bmm1_done wait (the kernel's
# BIAS_PRELOAD_CHUNKS), per RENDERING, keyed on the KV tile width -- the sign of the pre-load is a property of the rendering, not of
# the kernel (MEASURED, bf16 D 32 / 64, a 212-SM cc 10.7 node at the 2376 MHz lock, 100 x 3 graph-replay rounds: a base / 0 / base
# cross-process A/B/A over 27 cells with the base drift <= 0.23 %, then a base / applied / base confirm over 12 cells with control
# floors <= 0.18 % and base drift <= 0.2 %).  The sign is PER FORM, not per width.  On the 128-wide tile loading NOTHING ahead of the
# wait is +1.8..+3.2 % on the RESIDENT bias forms at S_kv >= 384 (the pair item +1.9 / +2.0 %, the one-batch-entry item +3.2 %; with
# a kv_mask +0.9..+2.1 % on the pair forms, -0.7 % on the one-batch-entry at S 384 -- a second node of the class read the same S 384
# d32 forms +1.7..+1.9 / +3.1 / +1.2 %, two chunks ahead -1.0 %), +0.0..+0.6 % on the STREAMED ring (S 768) and
# -1.2..+0.7 % at ONE kv tile (S 128, 5 of 6 forms a loss or noise -- a width the heuristics never pick for a bias graph below
# S_kv 384): the row read hides in place on the issue-bound body and the 32 packed words it kept live across the wait are freed.
# The confirm of the applied constant: +2.1 % (d32 S 384, pair resident) and +0.3 / +0.5 / +0.3 % on the three 128-wide streamed-ring
# renderings (d32 S 1024 one-batch-entry; d64 S 512 / 1024 pair) -- above the replay control floors and the base drift, inside the
# CUPTI control spread of the d32 processes (1.12 / 1.12 / 1.08 %) and of the d64 variant process (0.93 %), above the two d64 base
# processes' (0.14 / 0.02 %): at the edge of resolution.  On the 64-wide tile the signs are MIXED: the resident bias + kv_mask
# body (the heuristics' pick for a bias with any mask) LOSES 2.5..2.7 % without the pre-load (S 128 / 384; its one 64-column chunk
# has nothing to overlap the row read with), the resident bias-only forms gain +0.3..+1.4 %, the streamed ring loses 2.6 % without
# a mask and gains 1.4 % with one (S 768, one cell each) -- so that width keeps one chunk ahead of the wait for its served masked
# body; two chunks ahead is -1.0 % (64 words live across the wait).  Read by the kernel module at render time (one arithmetic, rule
# 8b'); the domain {0, 1, 2} is the kernel's.  RE-MEASURED per FORM on the training crops (S 256 / 384 / 512 / 768, B 1 / 4, live fractions
# 1.0 / 0.75 / 0.5; the three values of one form interleaved in ONE process through the sweep seam below, a 212-SM cc 10.7 node at the 2376
# lock, control twins 0.00-0.53 %): every 128-wide form loses with a chunk ahead of the wait (the pair resident -0.9..-1.7 %, the
# one-batch-entry ring -0.5..-1.0 %, two chunks -0.8..-5.9 %; the 128-wide pair RING +0.1 % with one chunk, -1.7 % with two), the 64-wide
# bias + kv_mask resident body loses WITHOUT its chunk (-1.5..-3.3 %; two chunks -0.6..+0.5 %), the 64-wide bias RING (D 64 S 768, the
# per-form split the first table left open) loses without it too (-1.8 % at 0.75 of the keys live, -0.9 % at 0.5), the padded 64-wide
# resident body is within noise either way (-0.6..+0.8 %), the three-warp-group body derives its effective pre-load 0 whatever the
# constant, and the 64-wide bias-only RESIDENT pair reads +1.1 / +0.5 / -0.6 % without the chunk where it is NOT the pick (D 32 S 256 /
# 384, D 64 S 384) and -0.4 / +0.1 % on its two pick cells (D 64 S 256, 256 / 1024 rows) -- so the per-width table stands as the default
# of every training form and gains no per-form key.
SM107_BIAS_PRELOAD_CHUNKS_BY_TILE_N = {SM107_SMALL_TILE_N_DEFAULT: 0, SM107_SMALL_TILE_N_NARROW: 1}
# The kernel's domain of the constant (``prefill_d128_f16_sm107`` raises outside it): 0 / 1 / 2 chunks ahead of the ``bmm1_done`` wait.
# PRIVATE, process-local sweep seam (the ``_N_CHUNK_OVERRIDE_FOR_SWEEPS`` rules: ``monkeypatch.setattr`` in a test or a sweep driver that
# resets it to None; never a plan path, an import or a config): pins the constant every rendering of the process gets -- the per-FORM
# A/B arm of the table above (0 / 1 / 2 against the width's value, same process, slots interleaved).  It moves a MODULE constant
# without moving the TemplateParams record, so it is part of :func:`sweep_seam_state` (the template-module cache key + the
# compiled-plan digest): two renderings that differ only by the pre-load never share a module or an artifact.
_BIAS_PRELOAD_CHUNKS_OVERRIDE_FOR_SWEEPS: Optional[int] = None


def sm107_bias_preload_chunks(tile_n: int) -> int:
    """The kernel's ``BIAS_PRELOAD_CHUNKS`` for a rendering at KV tile width ``tile_n`` (:data:`SM107_BIAS_PRELOAD_CHUNKS_BY_TILE_N`);
    a width outside the small tiles' domain (the d128 / d192 / d256 / d512 renderings, which carry no bias arm) keeps the one-chunk
    default so an unrelated rendering's instruction stream is unchanged.  The sweep seam ``_BIAS_PRELOAD_CHUNKS_OVERRIDE_FOR_SWEEPS``
    (validated to the kernel's domain) wins over the table."""
    if _BIAS_PRELOAD_CHUNKS_OVERRIDE_FOR_SWEEPS is not None:
        want = int(_BIAS_PRELOAD_CHUNKS_OVERRIDE_FOR_SWEEPS)
        if want not in BIAS_PRELOAD_CHUNKS_DOMAIN:
            raise ValueError(
                f"_BIAS_PRELOAD_CHUNKS_OVERRIDE_FOR_SWEEPS={want}: the kernel's BIAS_PRELOAD_CHUNKS domain is {sorted(BIAS_PRELOAD_CHUNKS_DOMAIN)}"
            )
        return want
    return SM107_BIAS_PRELOAD_CHUNKS_BY_TILE_N.get(int(tile_n), 1)


# The padded TAIL form of the PLAIN padded body on the small tiles (TemplateParams.padded_tail_form; MASK_FLAGS == MASK_PADDED alone):
# 0 = the legacy form -- the tail kv tile runs the SOFTWARE-max masked body as a SECOND loop beside the HW-max body, so the 128-wide
# rendering carries two full softmax bodies per warp group (4139 vs 2776 SASS instructions on the d32 bf16 rendering, HOST
# trace-compile for sm_107a) and outgrows the per-SM instruction cache -- the padding leg's measured pathology (S 217 plain 25.9 us vs
# 15.4 for the DENSE S 256 cell and 19.6 for the same shape WITH a kv_mask, whose -inf body is one body); 1 = the PRE-MASKED tail --
# one loop, the HW-max body on every tile, and on the tail tile the softmax warp first rewrites its sub-tile's out-of-range S
# columns as -inf in TMEM (a plain load, the bit-word PADDED mask, a store, one tcgen05.wait::st) so the fused max load observes
# them masked.  MEASURED (212-SM cc 10.7 perf node, 2376 MHz lock, same-process interleaved arms, 100 launches x 3 rounds, control
# twins within 0.4 %): at TILE_N 128 form 1 wins EVERY padded plain cell -- d32 B 1 S 200 / 217 / 300 / 450 / 600 +39 / +39 / +43 /
# +29 / +17 %, d32 B 4 S 217 / 300 +81 / +56 %, d64 B 1 S 200 / 300 / 600 +37 / +37 / +13 % over the legacy 128-wide body -- so the
# 128-wide default is 1; at TILE_N 64 (two half bodies under the cliff already) it is within noise with a per-cell sign (d32
# -0.4..-2.2 %, d64 +1.2..+2.3 %), so the 64-wide default stays the legacy form.  A request or the sweep seam pins either form at
# either width.
SM107_PADDED_TAIL_FORM_DEFAULT = 1  # the 128-wide tile
SM107_PADDED_TAIL_FORM_DEFAULT_NARROW = 0  # the 64-wide tile
# PRIVATE, process-local sweep seam (the ``_N_CHUNK_OVERRIDE_FOR_SWEEPS`` rules: ``monkeypatch.setattr`` in a test or a sweep driver
# that resets it to None; never a plan path, an import or a config): pins the padded tail form a knob-less ELIGIBLE plan gets.
_PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS: Optional[int] = None


def sm107_padded_tail_form_pick(
    d_flavor: int,
    *,
    padded: bool,
    has_bias: bool,
    has_kv_mask: bool,
    causal: bool = False,
    swa: bool = False,
    request: Optional[int] = None,
    tile_n: int = SM107_SMALL_TILE_N_DEFAULT,
) -> int:
    """The ``TemplateParams.padded_tail_form`` of a plan on the Rubin small tiles: 0 off the eligible set (a dense S_kv at a tile
    multiple, a bias or a kv_mask -- whose -inf body is ONE body already -- a causal / SWA band, the d128 flavor); on it ``request``
    when the caller made one (the adapter's ``padded_tail_form=``, validated there: honoured or declined, never degraded), else the
    sweep seam, else the measured default of the plan's KV tile width ``tile_n`` (appended, default 128):
    :data:`SM107_PADDED_TAIL_FORM_DEFAULT` on the 128-wide tile, :data:`SM107_PADDED_TAIL_FORM_DEFAULT_NARROW` on the 64-wide one."""
    if not padded or has_bias or has_kv_mask or causal or swa or int(d_flavor) not in (32, 64):
        return 0
    if request is not None:
        return int(request)
    if _PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS is not None:
        return int(_PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS)
    return SM107_PADDED_TAIL_FORM_DEFAULT_NARROW if int(tile_n) == SM107_SMALL_TILE_N_NARROW else SM107_PADDED_TAIL_FORM_DEFAULT


# The DEAD-SUB-TILE QUIET body (TemplateParams.dead_subtile_quiet; the one-batch-entry item of the small tiles, no bias): a softmax
# sub-tile whose first row is at or past its batch entry's S_q has no live row -- sub-tile 1 of the last 256-row super-tile when
# S_q % 256 is in 1..128 (S 384: a quarter of the item's softmax work; S 300 / 600 on the padding leg), and under cga2 the non-leader
# CTA's sub-tiles past S_q.  0 = the legacy form: that warp group runs the full softmax body on rows nobody stores (exp2 / max / sum
# on values the correction trims or the TMA store clips); 1 = the QUIET body: the same barrier sequence as the live body (the
# bmm1_done wait, alpha = 1 and the stat_full arrive, the bmm2_ready arrives, the warp-group ping-pong, the stat_empty wait), no TMEM
# load and no math, so the live sibling warp group runs the ping-pong alone.  The barrier table is unchanged by construction (the same
# sites fire the same lanes); the MMA still runs the dead sub-tile's BMM1 / BMM2 (its P is TMEM residue, its O rows are the trimmed /
# clipped rows).  MEASURED (212-SM cc 10.7 perf node, 2376 MHz lock, every process's busy clock median 2364 MHz, counter deltas 0,
# same-process interleaved arms, control twins within 0.13 %, 100 launches x 3 rounds, graph replay): the quiet body wins every
# PLAIN cell with a dead sub-tile -- d32 B 1 S 384 / 300 / 600 / 100 +7.5 / +1.0 / +3.9 / +6.0 %, d64 B 1 S 384 / 300 / 600 +4.2 /
# +0.5 / +0.7 % -- and the kv_mask cells on the dense and the 64-wide renderings (S 384 +5.8 %, S 100 +5.3 %), ties the no-dead-tile
# controls, and LOSES on the kv_mask + PADDED 128-wide rendering (d32 S 300 / 600 -2.7 / -4.5 %: the heaviest small-tile rendering,
# the quiet loop's code beside its hot body), so that one rendering keeps the full body (``sm107_dead_subtile_quiet_pick``).
SM107_DEAD_SUBTILE_QUIET_DEFAULT = 1
SM107_DEAD_SUBTILE_QUIET_DEFAULT_KV_MASK_PADDED_WIDE = 0  # the kv_mask + padded body at TILE_N 128
# PRIVATE, process-local sweep seam (the ``_N_CHUNK_OVERRIDE_FOR_SWEEPS`` rules): pins the quiet form a knob-less ELIGIBLE plan gets.
_DEAD_SUBTILE_QUIET_OVERRIDE_FOR_SWEEPS: Optional[int] = None


def sm107_dead_subtile_quiet_pick(
    d_flavor: int,
    *,
    has_bias: bool,
    pair_item: bool = False,
    request: Optional[int] = None,
    has_kv_mask: bool = False,
    padded: bool = False,
    tile_n: int = SM107_SMALL_TILE_N_DEFAULT,
    s_q: Optional[int] = None,
    cga: int = 1,
    tile_m: int = 128,
) -> int:
    """The ``TemplateParams.dead_subtile_quiet`` of a plan on the Rubin small tiles: 0 off the eligible set (a bias -- its warp groups
    also wait and release the bias barriers, which the quiet body does not replicate -- the pair item, whose dead sub-tile is the
    odd chunk tail recomputing row nb0, the d128 flavor); on it ``request`` when the caller made one (the adapter's
    ``dead_subtile_quiet=``, validated there: honoured or declined, never degraded), else the sweep seam, else the measured default of
    the plan's rendering (``has_kv_mask`` / ``padded`` / ``tile_n``, appended): :data:`SM107_DEAD_SUBTILE_QUIET_DEFAULT_KV_MASK_PADDED_WIDE`
    on the kv_mask + padded 128-wide rendering, :data:`SM107_DEAD_SUBTILE_QUIET_DEFAULT` everywhere else -- and 0 when the plan's S_q (``s_q``
    with its cluster width ``cga`` and Q tile height ``tile_m``) leaves NO dead sub-tile: the one-batch-entry item's cluster tile is
    2 x tile_m x cga rows (256 x cga at the default height), a dead sub-tile exists iff ``s_q % rows`` is in 1..(rows - tile_m); without
    one the quiet rendering only costs its code (S 217: -1.6 %)."""
    if has_bias or pair_item or int(d_flavor) not in (32, 64):
        return 0
    if request is not None:
        return int(request)
    if _DEAD_SUBTILE_QUIET_OVERRIDE_FOR_SWEEPS is not None:
        return int(_DEAD_SUBTILE_QUIET_OVERRIDE_FOR_SWEEPS)
    if s_q is not None:
        rows = 2 * int(tile_m) * max(int(cga), 1)
        rem = int(s_q) % rows
        if rem == 0 or rem > rows - int(tile_m):
            return 0  # every softmax sub-tile of every super-tile holds a live row: nothing to keep quiet
    if has_kv_mask and padded and int(tile_n) == SM107_SMALL_TILE_N_DEFAULT:
        return SM107_DEAD_SUBTILE_QUIET_DEFAULT_KV_MASK_PADDED_WIDE
    return SM107_DEAD_SUBTILE_QUIET_DEFAULT


# The knob sweep's INTERIOR rows (bf16, H 4, B x S flattened rows, the bias [1, H, S, S], the kv_mask [B_flat, 1, 1, S] bool, a padding
# mask with per-batch KV lengths of 0.75 S; a 212-SM cc 10.7 node at the 2376 MHz lock, CUDA-graph replay 100 x 3 with a control twin
# per process, the kernel-only CUPTI time where the replay is launch-bound): the measured record of the rows that sit INSIDE a constant
# run of the picks below -- test_sdpa_heuristics_small_tile_sm107.py pins the rows at which the picks turn, these document WHY the
# picks are what they are at the points in between.  Columns: (D, S, B, form, padded) -> (TILE_N, TILE_CGA_M, n_chunk, bias_mode,
# bias_stages) | the best measured arm | the margin % over the pre-table pick (POSITIVE = faster) | the metric.
#   (32, 256, 1, 'mask', False) -> (128, 1, 0, 0, 0) | mask | +0.0 % | CUPTI
#   (32, 384, 1, 'mask', False) -> (128, 1, 0, 0, 0) | mask | +0.0 % | replay
#   (32, 512, 1, 'plain', False) -> (128, 1, 0, 0, 0) | plain | +0.0 % | replay
#   (32, 512, 1, 'mask', False) -> (128, 1, 0, 0, 0) | mask | +0.0 % | replay
#   (32, 768, 1, 'plain', False) -> (128, 1, 0, 0, 0) | plain | +0.0 % | replay
#   (32, 1024, 1, 'plain', False) -> (128, 1, 0, 0, 0) | plain | +0.0 % | replay
#   (32, 1024, 1, 'bias', False) -> (128, 1, 0, 2, 2) | one-batch-entry ring, 2 stages | +1.5 % | replay
#   (32, 2048, 1, 'plain', False) -> (128, 1, 0, 0, 0) | plain | +0.0 % | replay
#   (32, 2048, 1, 'both', False) -> (64, 1, 8, 2, 3) | pair n8 ring / TILE_N 64 (n16 -1.5 %, the one-batch-entry 64-wide ring -4 %) | +1.4 % | replay
#   (32, 2048, 1, 'mask', False) -> (128, 1, 0, 0, 0) | mask | +0.0 % | replay
#   (32, 2048, 1, 'bias', False) -> (128, 1, 0, 2, 2) | one-batch-entry ring, 2 stages | +2.1 % | replay
#   (32, 128, 4, 'mask', False) -> (64, 1, 0, 0, 0) | mask@TILE_N 64 (+10.2 % CUPTI, whose twin spread was 18.6 % on this sub-20 us cell; the replay margin
#      carries it) | +8.5 % | replay
#   (32, 512, 4, 'both', False) -> (64, 1, 8, 1, 0) | pair n8 resident / TILE_N 64 | +17.7 % | replay
#   (32, 768, 4, 'bias', False) -> (128, 1, 0, 2, 2) | one-batch-entry ring, 2 stages | +2.1 % | replay
#   (32, 1024, 4, 'bias', False) -> (128, 1, 0, 2, 2) | one-batch-entry ring, 2 stages | +1.6 % | replay
#   (32, 2048, 4, 'bias', False) -> (128, 1, 0, 2, 2) | one-batch-entry ring, 2 stages | +2.0 % | replay
#   (32, 2048, 4, 'plain', True) -> (128, 1, 0, 0, 0) | plain with per-batch KV lengths (TILE_N 64 -19 %) | +0.0 % | replay
#   (32, 200, 1, 'plain', False) -> (128, 1, 0, 0, 0) | plain / TILE_N 128 + the pre-masked tail (the half tile +1.8 % behind) | +39.2 % | replay
#   (32, 217, 1, 'plain', False) -> (128, 1, 0, 0, 0) | plain / TILE_N 128 + the pre-masked tail (the half tile +3.9 % behind) | +39.2 % | replay
#   (32, 600, 1, 'plain', False) -> (128, 1, 0, 0, 0) | plain / TILE_N 128 + the pre-masked tail (the half tile +13.7 % behind) | +17.1 % | replay
#   (32, 600, 1, 'both', False) -> (64, 1, 8, 1, 0) | pair / TILE_N 64 (n16 measured +23.2 %) | +23.2 % | replay
#   (64, 256, 1, 'plain', False) -> (128, 1, 0, 0, 0) | plain, cga1 (cga2 -31 %) | +0.0 % | CUPTI
#   (64, 384, 1, 'plain', False) -> (128, 1, 0, 0, 0) | plain, cga1 (cga2 a tie) | +0.0 % | replay
#   (64, 512, 1, 'plain', False) -> (128, 1, 0, 0, 0) | plain, cga1 (cga2 a tie) | +0.0 % | replay
#   (64, 768, 1, 'bias', False) -> (128, 1, 8, 2, 3) | pair ring n8 (n4 +3.8 %, n8 +3.0 % over the rule's n16) | +3.0 % | replay
#   (64, 768, 1, 'both', False) -> (64, 1, 8, 2, 3) | pair n8 ring / TILE_N 64 | +37.8 % | replay
#   (64, 1024, 1, 'bias', False) -> (128, 1, 8, 2, 3) | pair ring (n16 +1.0 % over the rule's n32; the cga2 one-batch-entry ring +4.0 % is an isolated point,
#      not a rule) | +1.0 % | replay
#   (64, 1024, 1, 'both', False) -> (64, 1, 8, 2, 3) | pair ring / TILE_N 64 (n32 measured +39 %) | +39.0 % | replay
#   (64, 2048, 1, 'plain', False) -> (128, 1, 0, 0, 0) | plain, cga1 (cga2 -2.7 % replay, -2.4 % CUPTI) | +0.0 % | replay
#   (64, 2048, 1, 'bias', False) -> (128, 1, 8, 2, 3) | pair ring n8 (+3.8 % over the rule's n128; n4 / n16 within 0.3 %) | +3.8 % | replay
#   (64, 2048, 1, 'both', False) -> (64, 1, 8, 2, 3) | pair n8 ring / TILE_N 64 | +50.8 % | replay
#   (64, 128, 4, 'mask', False) -> (64, 1, 0, 0, 0) | mask@TILE_N 64 (the additive vector alone +31 % too) | +17.7 % | CUPTI
#   (64, 1024, 4, 'both', False) -> (64, 1, 8, 2, 3) | pair ring / TILE_N 64 (n128 measured +31.6 %) | +31.6 % | replay
#   (64, 2048, 4, 'bias', False) -> (128, 1, 8, 2, 3) | pair ring n8 (+4.3 % over the rule's n512, 8192 rows; lower-bound process) | +4.3 % | replay
#   (64, 2048, 4, 'both', False) -> (64, 1, 8, 2, 3) | pair ring / TILE_N 64 (n512 measured +39 %; lower-bound process) | +39.1 % | replay
#   (32, 256, 1, 'plain', True) -> (128, 1, 0, 0, 0) | plain with per-batch KV lengths 0.75 S (TILE_N 64 +94.4 %: kv_len 192 ends mid-tile, the padded dense
#      body's pathology -- the pick sees S_kv = 256, a tile multiple, and cannot follow the lengths; an open kernel item) | +94.4 % | CUPTI
#   (32, 512, 1, 'plain', True) -> (128, 1, 0, 0, 0) | plain with per-batch KV lengths 0.75 S (TILE_N 64 -11.5 %) | +0.0 % | replay
#   (32, 1024, 1, 'plain', True) -> (128, 1, 0, 0, 0) | plain with per-batch KV lengths 0.75 S (TILE_N 64 -16.3 %) | +0.0 % | replay
#   (32, 2048, 1, 'bias', True) -> (64, 1, 8, 2, 3) | bias + per-batch KV lengths 0.75 S: one-batch-entry / ring 3 st / TILE_N 64 +14.3 % (the table's pair n8
#      ring at 64 unmeasured here) | +14.3 % | replay
#   (32, 256, 4, 'plain', True) -> (128, 1, 0, 0, 0) | plain with per-batch KV lengths 0.75 S (TILE_N 64 +107.0 %: kv_len 192 ends mid-tile, the padded dense
#      body's pathology -- the pick sees S_kv = 256, a tile multiple, and cannot follow the lengths; an open kernel item) | +107.0 % | replay
#   (32, 512, 4, 'plain', True) -> (128, 1, 0, 0, 0) | plain with per-batch KV lengths 0.75 S (TILE_N 64 -13.1 %) | +0.0 % | replay
#   (32, 1024, 4, 'plain', True) -> (128, 1, 0, 0, 0) | plain with per-batch KV lengths 0.75 S (TILE_N 64 -16.6 %) | +0.0 % | replay
#   (64, 256, 1, 'plain', True) -> (128, 1, 0, 0, 0) | plain with per-batch KV lengths 0.75 S (TILE_N 64 +66.6 %: kv_len 192 ends mid-tile, the padded dense
#      body's pathology -- the pick sees S_kv = 256, a tile multiple, and cannot follow the lengths; an open kernel item) | +66.6 % | CUPTI
#   (64, 512, 1, 'plain', True) -> (128, 1, 0, 0, 0) | plain with per-batch KV lengths 0.75 S (TILE_N 64 -12.8 %) | +0.0 % | replay
#   (64, 1024, 1, 'bias', True) -> (64, 1, 8, 2, 3) | bias + per-batch KV lengths 0.75 S: the table's pair n8 / ring 3 st / TILE_N 64 +29.8 %; the pair n2 /
#      ring 3 st / TILE_N 64 +31.6 % -- a 2-row chunk 1.8 points ahead of the capped n8 here (one process; the cap at 8 stays) | +31.6 % | replay
#   (64, 1024, 1, 'plain', True) -> (128, 1, 0, 0, 0) | plain with per-batch KV lengths 0.75 S (TILE_N 64 -17.6 %) | +0.0 % | replay
#   (64, 256, 4, 'plain', True) -> (128, 1, 0, 0, 0) | plain with per-batch KV lengths 0.75 S (TILE_N 64 +86.4 %: kv_len 192 ends mid-tile, the padded dense
#      body's pathology -- the pick sees S_kv = 256, a tile multiple, and cannot follow the lengths; an open kernel item) | +86.4 % | replay
#   (64, 512, 4, 'plain', True) -> (128, 1, 0, 0, 0) | plain with per-batch KV lengths 0.75 S (TILE_N 64 -14.9 %) | +0.0 % | replay
#   (64, 1024, 4, 'bias', True) -> (64, 1, 8, 2, 3) | bias + per-batch KV lengths 0.75 S: pair n8 / ring 3 st / TILE_N 64 +38.5 % | +38.5 % | replay
#   (64, 1024, 4, 'plain', True) -> (128, 1, 0, 0, 0) | plain with per-batch KV lengths 0.75 S (TILE_N 64 -17.7 %) | +0.0 % | replay


def sm107_tile_n_pick(d_flavor: int, s_kv: int, *, has_bias: bool, has_kv_mask: bool, padded: bool = False, request: Optional[int] = None, domain=None) -> int:
    """The ``TemplateParams.tile_n`` a knob-less plan on the Rubin 32- / 64-wide tiles gets (the table above): ``request`` when the
    caller made one (the ``TILE_N`` knob / the adapter's ``tile_n=``: honoured or declined by the row / adapter, never degraded);
    else :data:`SM107_SMALL_TILE_N_NARROW` for (a) a bias TOGETHER WITH a mask -- a kv_mask of any form (the fp32 vector's only served
    width, :data:`SM107_FP32_VECTOR_WITH_BIAS_TILE_N`, agrees) or the kernel's padding-mask arm (per-batch KV lengths, or
    ``s_kv % 128 != 0``: with per-batch lengths of 0.75 S +11..+73 % over the pre-table pick at S 256 / 512 / 1024 / 2048, both D, B 1 / 4 --
    13 of 13 cells with the table's form measured, the d32 B 1 S 2048 cell through the one-batch-entry ring at 64 (+14 %); +17..+38 % on the
    ragged leg) -- the combined body is the one over the instruction-cache cliff; (b) a bias alone at one 128-wide kv tile (the pair n4 / n2: +26 / +24 % at d32 / d64 with 128 rows, +35 /
    +22 % with 512) and at two tiles on the 64-wide tile (the pair n2 / n8: +24 / +4 %) -- at two tiles on the 32-wide tile the
    one-batch-entry item's streamed ring at 128 is the measured best (+6 / +9 % with 256 / 1024 rows), so 128 stays there; (c) a kv_mask
    alone at one 128-wide kv tile (+7..+18 %; a wash or a loss above it); (d) a plain graph whose S_kv ends in the FIRST half of a 128-wide
    tile (``s_kv % 128`` in 1..64, both D): the half tile skips 64 dead columns the 128-wide tile would mask -- d32 S 300 B 1 / 4 +4.2 /
    +6.2 % over the 128-wide pre-masked tail; d64 S 300 the half tile's LEGACY tail (the 64-wide default) is 0.8..1.1 % BEHIND the
    128-wide pre-masked tail -- 48.14 vs 47.63 / 47.74 us, two pairs of one process -- inside the per-S rule, which follows ``s_kv % 128``
    alone (the half tile WITH the pre-masked tail would tie it at 47.54 us, but at TILE_N 64 that form is within noise with a per-cell
    sign, d32 -2.2..+1.3 % and d64 -0.7..+3.9 %, and is not the default); where the tail holds MORE than 64 columns the 128-wide tile WITH
    the pre-masked tail (``padded_tail_form`` 1, the 128-wide default) wins: d32 S 200 / 217 / 450 / 600 +1.8 / +3.9 / +11.5 / +13.7 %,
    d32 B 4 S 217 +5.4 %, d64 S 200 / 600 +7.1 / +20 % over the half tile (212-SM cc 10.7 perf node, 2376 MHz lock, same process) --
    before the pre-masked tail the 128-wide padded body was the instruction-cache pathology and the half tile served every d32 S and
    d64 up to 256; not for per-batch lengths at a tile multiple (128 wins there: the narrow tile is -11..-18 % at S 512 / 1024 and -19 /
    -20 % at d32 / d64 S 2048 with 0.75 S keys).  The pick sees ``s_kv``, never the lengths: where the LENGTHS end mid-tile (0.75 S at
    S 256 = 192 keys) the half tile measured +58..+119 % and this rule cannot follow it; else :data:`SM107_SMALL_TILE_N_DEFAULT`.
    ``domain`` = the flavor's served widths (:data:`SM107_TILE_NS_BY_D_SHAPE`); a flavor without the narrow width keeps the default."""
    if request is not None:
        return int(request)
    dom = (
        frozenset(domain) if domain is not None else dict(SM107_TILE_NS_BY_D_SHAPE).get((int(d_flavor), int(d_flavor)), frozenset({SM107_SMALL_TILE_N_DEFAULT}))
    )
    if SM107_SMALL_TILE_N_NARROW not in dom:
        return SM107_SMALL_TILE_N_DEFAULT
    padded_body = bool(padded) or int(s_kv) % SM107_SMALL_TILE_N_DEFAULT != 0  # the kernel's padding-mask arm
    if has_bias and (has_kv_mask or padded_body):
        return SM107_SMALL_TILE_N_NARROW
    if has_bias and (int(s_kv) <= SM107_SMALL_TILE_N_DEFAULT or (int(d_flavor) == 64 and int(s_kv) <= SM107_NARROW_TILE_S_KV_MAX)):
        return SM107_SMALL_TILE_N_NARROW
    if has_kv_mask and int(s_kv) <= SM107_SMALL_TILE_N_DEFAULT:
        return SM107_SMALL_TILE_N_NARROW
    if padded_body and not has_kv_mask and 0 < int(s_kv) % SM107_SMALL_TILE_N_DEFAULT <= SM107_SMALL_TILE_N_NARROW:
        return SM107_SMALL_TILE_N_NARROW
    return SM107_SMALL_TILE_N_DEFAULT


# The bias block's TMA / lane swizzle: the 128-B atom.  BOTH swizzle jobs force it -- the TMA descriptor writes the
# block under it (job D) and the softmax lanes read one 128-B row slice each (job L: a per-lane stride that is a multiple
# of the 128-B bank cycle is a 32-way conflict unswizzled; the 16-B chunk index XOR (row % 8) spreads 8 rows over 8 bank
# groups -> 4 wavefronts per warp-wide 16-B load = ideal).  ``Swizzle(3, 4, 3)``: MBase + SShift = 7 = log2(128 B).
# (BIAS_SWZ_BYTES = 128 is defined in config_d128_family and re-exported above.)


def sm107_bias_block_rows(tiles_q: int = 2, tile_m: int = 128, *, pair: bool = False) -> int:
    """Rows of the resident bias block = the Q rows one work item's softmax sub-tiles READ (``config_d128_family.bias_block_rows``:
    ``tiles_q x tile_m`` on the one-batch-entry item, ``tile_m`` on the (n, n+1) PAIR item)."""
    return _bias_block_rows(tiles_q, tile_m, pair=pair)


def sm107_bias_pair_form(bias_batch_extent: int, b_flat: int, cta_mma: int) -> bool:
    """Whether a BIAS plan runs the (n, n+1) PAIR work item (``config_d128_family.bias_pair_form``: a batch-BROADCAST bias on >= 2
    flattened rows at cga1).  ONE predicate for the engine row, the heuristics and the adapter (rule 8b')."""
    return _bias_pair_form(bias_batch_extent, b_flat, cta_mma)


def sm107_bias_resident_tiles(
    tile_k: int, cta_mma: int, *, dtype_qkv: int = _DTYPE_BF16, tile_n: int = 128, tiles_q: int = 2, tile_m: Optional[int] = None, pair: bool = False
) -> int:
    """How many ``tile_n``-column bias tiles (``BIAS_BLOCK_ROWS x tile_n`` elements each) the Rubin carveout holds beside
    the d128 family's Q / O slabs and K / V rings at ``tile_k = tile_o`` and the flavor's ring depth (``_stages_kv_d128``)
    -- the SAME budget ``_validate_smem`` enforces (320 KiB usable minus the fixed overhead): ``config_d128_family.bias_resident_tiles``
    at ``SM107_ARCH``.  0 = no resident block fits.  Derived, never a literal: 3 at the 32-wide tile (cga1), 1 / 2 at the 64-wide one
    (cga1 / cga2), 0 at d128 on the 256-row item; the PAIR item's 128-row block (``pair=True``) holds twice as many -- 6 at the 32-wide
    tile, 3 at the 64-wide one (cga1)."""
    stages = _stages_kv_d128(dtype_qkv, cta_mma, mxfp8=False, tile_k=tile_k)
    # The Q tile height: the seam's (the TILE_M = 64 lever halves the Q / O slabs AND the block's rows) unless the caller pinned one.
    tile_m = _tile_m_d128(tile_k) if tile_m is None else int(tile_m)
    return _bias_resident_tiles(SM107_ARCH, tile_k, cta_mma, stages_kv=stages, tile_m=tile_m, bpe_val=bpe(dtype_qkv), tile_n=tile_n, tiles_q=tiles_q, pair=pair)


def sm107_bias_resident_s_kv_max(d_flavor: int, cta_mma: int, *, dtype_qkv: int = _DTYPE_BF16, tile_n: int = 128, pair: bool = False, tiles_q: int = 2) -> int:
    """The largest S_kv (in whole tiles) whose bias row-block the resident form serves at this width and cga: the work
    item's KV range is at most ceil(S_kv / tile_n) tiles.  0 = the width serves no bias.  ``pair``: the (n, n+1) item's
    128-row block (``sm107_bias_pair_form``); ``tiles_q``: the body's Q sub-tiles = softmax warp groups (3 on the third-warp-group
    arm: one more Q and O slab each, so fewer bias tiles fit).  Read by the engine row's mismatch, the adapter's check_support and
    the config factory (one arithmetic, rule 8b')."""
    return sm107_bias_resident_tiles(d_flavor, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, tiles_q=tiles_q, pair=pair) * tile_n


def sm107_bias_ring_stages(
    tile_k: int, cta_mma: int, *, dtype_qkv: int = _DTYPE_BF16, tile_n: int = 128, tiles_q: int = 2, tile_m: Optional[int] = None, pair: bool = False
) -> int:
    """The deepest STREAMED bias ring (``BIAS_MODE_STREAMED``) the Rubin carveout holds beside the operands at this width:
    ``min(max(BIAS_RING_STAGES), spare // tile_bytes)`` in whole ``BIAS_BLOCK_ROWS x tile_n`` tiles, **0 when fewer than
    ``min(BIAS_RING_STAGES)`` fit** (no ring).  The SAME spare the resident budget reads (one arithmetic, rule 8b').  Derived:
    3 at the 32-wide tile (both item forms, cga1), 3 at the 64-wide tile's pair item (32 KiB stages), 0 at its one-batch-entry
    item at cga1 (two 64 KiB stages exceed the 126 KiB spare) and 2 at cga2."""
    tiles = sm107_bias_resident_tiles(tile_k, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, tiles_q=tiles_q, tile_m=tile_m, pair=pair)
    stages = min(max(BIAS_RING_STAGES), tiles)
    return stages if stages >= min(BIAS_RING_STAGES) else 0


# PRIVATE, process-local TEST seams beside ``_N_CHUNK_OVERRIDE_FOR_SWEEPS`` (same rules: ``monkeypatch.setattr`` in a test or a
# sweep driver that resets them to None; never a plan path, an import or a config): ``_BIAS_MODE_OVERRIDE_FOR_SWEEPS`` pins the
# bias FORM a knob-less plan gets (1 = the resident block, declining above its budget -- the A/B's resident arm; 2 = the streamed
# ring wherever it fits, INCLUDING where the resident block would also fit -- the A/B's streamed arm); ``_BIAS_STAGES_OVERRIDE_FOR_SWEEPS``
# pins the ring depth (2 / 3).  A set value changes the plan the graph BUILDS and its ELIGIBILITY (the pick feeds the engine row's
# mismatch, the heuristics' width pick and the adapter), so neither is configuration nor part of the public surface.
_BIAS_MODE_OVERRIDE_FOR_SWEEPS: Optional[int] = None
_BIAS_STAGES_OVERRIDE_FOR_SWEEPS: Optional[int] = None
# ``_STAGES_KV_OVERRIDE_FOR_SWEEPS`` (same rules) pins the K / V ring depth of the 32- / 64-wide f16/bf16 tiles (``_stages_kv_d128``,
# the small-tile branch; 2 / 3 / 4 = :data:`SMALL_D_STAGES_KV`): the A/B arm of the ring-depth lever.  Read by the ONE helper every
# budget and the factory call, so a pinned depth moves the bias budgets with it (a shallower ring frees SMEM for the resident block /
# a deeper bias ring).  None = the derived depth (4).
_STAGES_KV_OVERRIDE_FOR_SWEEPS: Optional[int] = None
# The K / V ring depths the small-tile body is validated at (SMEM fit at every width in SM107_SMALL_D_CGAS, the ring-wrap axis
# n_kv = depth / depth + 1 / two wraps of the fresh-process census).  The validator pins cfg.STAGES_KV to this set on the small tiles.
# ``_TILE_M_OVERRIDE_FOR_SWEEPS`` (same rules) pins the Q TILE HEIGHT of the 32- / 64-wide f16/bf16 tiles: 64 or 128 (:data:`SMALL_D_TILE_MS`);
# None = 128, the height every rendering has had.  Read by the ONE helper (``_tile_m_d128``) the config factory, the bias budgets
# (``sm107_bias_resident_tiles`` / ``sm107_bias_ring_stages``: the Q / O slabs and the bias block's rows both halve) and the kernel's
# TMEM / grid derivations share (rule 8b').  A 64-row tile doubles the work items of a given q extent (the parallelism lever at small S)
# and halves the MMA / softmax work per kv step; the kernel body renders it as a lane-mapping arm (``TILE_M64_LANE_FORM`` in
# ``sm107/prefill_d128_f16.py``).  An A/B arm, never a plan path: the engine row's ``tile_ms`` stays {128} until the arm has a measured win.
_TILE_M_OVERRIDE_FOR_SWEEPS: Optional[int] = None
# The Q tile heights the small-tile body is validated at.  The validator pins cfg.TILE_M to this set on the small tiles; every other
# flavor keeps 128 (the d128 / d192 renderings and their committed records, the quantized siblings' SF geometry).


def sweep_seam_state() -> Optional[tuple]:
    """The private sweep seams' values as ONE hashable, ``None`` when every seam is unset (every plan path).  ``api_dsl`` folds it into
    the template-module cache key and the source digest (``template_loader.load_template(extra_key=)``): the module cache keys on the
    TemplateParams record, and a seam moves ``Cfg`` fields (the ring depth, the Q tile height) or a module constant (the bias pre-load
    count) WITHOUT touching the record -- so two renderings that differ only by a seam would otherwise share one module in a process
    (the second request silently running the first rendering) and one compiled-plan artifact on disk.  ``_S_SLOTS_OVERRIDE_FOR_SWEEPS``
    is NOT in the tuple on purpose: its pick lands IN the record (``TemplateParams.s_slots``, ``sm107_s_slots_pick``), so the record
    alone already keys the two renderings -- a seam belongs here only when it moves ``Cfg`` (or a module constant) without moving the
    record."""
    state = (
        _N_CHUNK_OVERRIDE_FOR_SWEEPS,
        _BIAS_MODE_OVERRIDE_FOR_SWEEPS,
        _BIAS_STAGES_OVERRIDE_FOR_SWEEPS,
        _STAGES_KV_OVERRIDE_FOR_SWEEPS,
        _TILE_M_OVERRIDE_FOR_SWEEPS,
        _BIAS_PRELOAD_CHUNKS_OVERRIDE_FOR_SWEEPS,
    )
    return None if all(v is None for v in state) else ("sm107_sweep_seams",) + tuple(None if v is None else int(v) for v in state)


def _tile_m_d128(tile_k: int) -> int:
    """The Q tile height of a d128-family rendering: 128 everywhere but the 32- / 64-wide f16/bf16 tiles, where the private seam
    ``_TILE_M_OVERRIDE_FOR_SWEEPS`` may pin 64 (the TILE_M = 64 lever).  Read by the factory AND the bias budgets so the Q / O slabs,
    the bias block rows and the resident / ring budgets move together (one arithmetic, rule 8b')."""
    if tile_k > 64:
        return 128
    if _TILE_M_OVERRIDE_FOR_SWEEPS is not None:
        want = int(_TILE_M_OVERRIDE_FOR_SWEEPS)
        if want not in SMALL_D_TILE_MS:
            raise ValueError(
                f"_TILE_M_OVERRIDE_FOR_SWEEPS={want}: the 32- / 64-wide tiles' Q tile height is validated at {sorted(SMALL_D_TILE_MS)} rows "
                f"(the warp-lane -> row mapping of the softmax / correction groups and the M = 64 tcgen05.mma datapath layout exist for these two only)"
            )
        return want
    return 128


# The row-max FORM seam (same rules): pins ``TemplateParams.bias_max_form`` of a knob-less plan (0 = software, 1 = the hardware upper bound);
# the A/B driver's ``hw`` / ``sw`` arms.  Read by :func:`sm107_bias_max_form_pick` only.  NOT part of :func:`sweep_seam_state`: unlike the
# ring-depth / tile-height seams it moves a RECORD field (the adapter writes the pick into ``TemplateParams.bias_max_form``), so two
# renderings that differ only by this seam already key different template modules and compiled-plan artifacts.
_BIAS_MAX_FORM_OVERRIDE_FOR_SWEEPS: Optional[int] = None


def sm107_bias_max_form_default(bias_mode: int, kv_mask_form: int) -> int:
    """The MEASURED default row-max form of a bias / kv_mask plan that requests none -- per FORM, never one constant:

    * the kv_mask-ONLY form (the bit word, no bias) runs the HARDWARE UPPER BOUND: there the bound is the fused raw max alone, nothing is
      added per step, and it recovers a third of the -inf body's cost over the plain kernel -- **+6.1..+14.8 %** on eight cells (bf16, D 32 at
      S 256 / 384 / 512 / 768 / 1024 with one batch entry and S 384 with four, D 64 at S 384 / 768; +11.4 % at the 384 cell, +14.8 % at 1024),
      every cell outside its control-pair spread by more than ten times; a 212-SM cc 10.7 perf node at the 2376 MHz lock (sampled busy median
      0.5 % under it, no power-cap event), 100 x 3 CUDA-graph replay, the two forms interleaved in ONE process, the fp64 oracle on both.
      The trees behind the claim (a perf number carries its base): the eight cells were measured on the arm over ``deb8d0e6e`` -- the kv
      tile body BEFORE the dead-sub-tile quiet default (``3cdb717d4``), which the tree this function ships in renders at S 384 on the
      kv_mask-only plan (``dead_subtile_quiet`` 1: S_q % 256 = 128) and nowhere at S % 256 = 0 (there the S 256 / 512 / 768 / 1024 cells
      render the SAME kv tile body on every tree: one cubin md5 for the kv_mask-only rendering across the four bases rendered).  The
      S 384 cells were therefore RE-MEASURED on the landed tree ``b302e46b9`` (a second node of the same class, the quiet rendering attested
      per process, same protocol): D 32 **+12.0 %** with one batch entry (62.18 -> 55.52 us; the arm's cost over the plain kernel +44.8 % ->
      +29.3 %), **+15.2 %** with four (224.00 -> 194.50 us; +55.1 % -> +34.7 %), D 64 **+10.2 %** (71.94 -> 65.28 us; +40.4 % -> +27.4 %;
      a duty-cycled process whose busy-clock median sat 1.0 % under the lock -- a lower bound on the absolutes, the ratio same-process); the
      rebased tree ``111c061f8`` (the legacy body) read +11.3 % at S 384 and +14.9 % at S 1024.  Against the 6-op softmax ALU floor
      (212 SMs x 2.364 GHz, the binding term at D 32) the plain kernel runs at 53 / 60 % on the landed S 384 cells (one / four entries) and the
      kv_mask-only arm moves from 36 -> 40 % and 38 -> 44 % (CUPTI kernel time);
    * every form WITH a bias keeps the SOFTWARE tree: the per-step packed per-row bias-tile max (63 ``max.bf16x2`` per 128 columns per thread
      plus the early second read of the non-pre-loaded chunk) costs about twice what the removed tree saves -- **-4.9..-12.2 %** on nine
      cells (both item forms, the resident block and the streamed ring, with and without the kv_mask, D 32 / 64, S 384 / 768 / 1024).  The
      bias row max has to come from OUTSIDE the per-step path (a per-item reduction over the resident block, or a once-per-execute row-max
      pre-pass over the bias) before those arms can collect the kv_mask-only gain; recorded, not built.  The trees behind THIS half: the
      nine cells were measured on the same arm over ``deb8d0e6e`` -- whose 128-wide bias arms pre-load ONE bias chunk and render the legacy
      dead-sub-tile body -- and two of them again on ``111c061f8`` (the same constants: the S 384 resident pair -8.4 %, the S 1024
      one-batch-entry ring -12.1 %); the tree this function ships in renders the 128-wide bias arms with NO pre-loaded chunk
      (``9022d1a47``) and the quiet body at S 384, where the hardware-vs-software sign of the bias forms is UNMEASURED -- the software
      default is the pre-existing kernel either way, so only a missed win is at stake there; the per-item / pre-pass follow-up re-measures
      from that tree;
    * a plan without either arm has no -inf body: the software value (the field is meaningless there and the factory pins it to 0).

    The adapter's pick (:func:`sm107_bias_max_form_pick`) applies this unless a request or the sweep seam overrides it; the record field
    itself (``TemplateParams.bias_max_form``) still defaults to 0, so a record without the field renders byte-identically."""
    if kv_mask_form == KV_MASK_BOOL_WORD and not bias_mode:
        return BIAS_MAX_HW_UPPER_BOUND
    return BIAS_MAX_SOFTWARE


def sm107_bias_max_form_pick(bias_mode: int, kv_mask_form: int, *, request: Optional[int] = None) -> int:
    """The ``TemplateParams.bias_max_form`` of a bias / kv_mask plan: ``request`` when the caller made one (honoured here, validated by the
    factory / the adapter's check_support: an additive kv_mask or a plan without either arm is a typed decline, never a degraded form); else
    the sweep seam (``_BIAS_MAX_FORM_OVERRIDE_FOR_SWEEPS``, DEGRADED to the software form where the hardware bound is not served -- no arm, an
    additive vector -- because a pinned default must never produce an invalid record); else the measured per-form default
    (:func:`sm107_bias_max_form_default`)."""
    if request is not None:
        return int(request)
    if _BIAS_MAX_FORM_OVERRIDE_FOR_SWEEPS is not None:
        want = int(_BIAS_MAX_FORM_OVERRIDE_FOR_SWEEPS)
        if want == BIAS_MAX_HW_UPPER_BOUND and (not (bias_mode or kv_mask_form) or kv_mask_form in (KV_MASK_ADDITIVE_16, KV_MASK_ADDITIVE_F32)):
            return BIAS_MAX_SOFTWARE
        return want
    return sm107_bias_max_form_default(bias_mode, kv_mask_form)


# ``_SOFTMAX_WGS_OVERRIDE_FOR_SWEEPS`` (the same rules) pins the number of softmax WARP GROUPS a knob-less BIAS plan on the small tiles
# RECORDS (``TemplateParams.softmax_wgs``: 2 = the classic 16-warp body, 3 = the third softmax warp group of the TILE_N-64 pair item --
# the plan's "third softmax warpgroup" experiment).  It feeds the RECORD through :func:`sm107_softmax_wgs_pick`, never a Cfg behind an
# unchanged record: the kernel-module cache (``frost.template_loader``) is keyed by the TemplateParams, so a Cfg-only seam hands a second
# in-process build the FIRST build's module (verified 2026-10-09).  None = 2.
_SOFTMAX_WGS_OVERRIDE_FOR_SWEEPS: Optional[int] = None
# The softmax warp-group counts the d128-file body is written for: 2 (every path) and 3 (the (n, n+1) pair item at TILE_N 64 -- the one
# form whose roles loop over the sub-tiles; the one-batch-entry paths hand-unroll two).  The validator pins the serving set.


# The third softmax warp group is the DEFAULT of ONE form: a bias TOGETHER WITH a kv_mask on the (n, n+1) pair item at TILE_N 64 (the heuristics'
# pick for that graph) at cga1, dense KV lengths, S_kv at or above this -- MEASURED (bf16 B 1 H 4, a 212-SM cc 10.7 node at the 2376 MHz lock,
# the 3-WG arm interleaved with its 2-WG base in ONE process, 100 x 3 graph replays, control twins <= 0.14 %, speed-up = t(2 WGs) / t(3 WGs) - 1 at
# EQUAL live row-computations -- the 3-WG body at its chunk 6 against the pair's pick n8, each body at its own chunk): D 32 S 640 / 768 / 896 /
# 1024 +8.1 / +7.9 / +3.3 / +2.9 %, D 64 S 768 / 1024 +3.5 / +8.7 %; a confirm of the applied default on the knob-less path D 32 S 640 / 768 /
# 1024 +6.6 / +7.3 / +3.3 %, D 64 S 768 / 1024 +5.4 / +10.7 %; and at S 512 only +0.8 % (D 32) / -2.5 % (D 64), S 384 +0.5 / +0.7 % (two
# processes) -- inside the protocol's resolution or a loss, so the two-group body keeps those.  Every OTHER form loses or ties with three groups:
# the bias alone at TILE_N 64 -1.0..-1.6 % at S 384 / 512 (D 32), -9.8..-17.2 % at D 64 (and its heuristics' pick is TILE_N 128 anyway), the
# pair's own even chunk (n4 / n8) on the 3-WG body -2.5..-28.5 % (one or two DEAD rows recomputed per chunk).  The padded (per-batch KV
# length) form is unmeasured on three groups and keeps two.  A chunk on the 3-WG body is a multiple of THREE (`rows_per_step`): 6 = the
# largest multiple of 3 under the pair's cap of 8 (n9 within -0.5..+1.5 % of n6 from S 640, -2.9..-3.8 % at S 512: not a default).
SM107_THREE_WG_S_KV_MIN = 640
# ... and up to this many FLATTENED batch rows (the Q tensor's batch): the chunk 6 of the 3-WG body makes 4 / 3 the items of the pair's n8,
# and at 4096 rows (D 32 S 1024, B 4) the per-item cost outran the body's gain (-2.4 % where B 1 at the same S won +3.3 %); 2560 / 3072 rows
# (B 4 at S 640 / 768) kept +7.2 / +7.4 % at D 32 and +0.8 % at D 64 S 768, so the bound is the largest measured winning flattened batch.
# A caller that does not know the flattened batch gets two groups.
SM107_THREE_WG_B_FLAT_MAX = 3072


def sm107_softmax_wgs_pick(
    request=None,
    *,
    d_flavor: Optional[int] = None,
    s_kv: Optional[int] = None,
    tile_n: int = 128,
    pair: bool = False,
    has_bias: bool = False,
    has_kv_mask: bool = False,
    padded: bool = False,
    cga: int = 1,
    b_flat: Optional[int] = None,
    s_slots: int = 1,
) -> int:
    """The ``TemplateParams.softmax_wgs`` a plan records: the adapter's ``softmax_wgs=`` request, else the sweep seam
    ``_SOFTMAX_WGS_OVERRIDE_FOR_SWEEPS``, else the MEASURED rule above -- 3 for a bias + kv_mask graph on the (n, n+1) pair item at TILE_N 64 /
    cga 1 with dense KV lengths, ``s_kv >= SM107_THREE_WG_S_KV_MIN`` and at most ``SM107_THREE_WG_B_FLAT_MAX`` flattened batch rows (the 32- /
    64-wide tiles), 2 everywhere else (a caller that does not know the flattened batch gets 2; a two-slot record -- ``s_slots`` 2 -- gets 2,
    the only group count the double-buffered S is served with).  Validity of a REQUEST --
    TILE_N 64, the pair item, cga1, the f16/bf16 32- / 64-wide tiles -- is :func:`_validate_softmax_wgs`'s, declined typed by the adapter's
    ``check_support`` before the factory is reached.  ONE arithmetic for the engine row's budget, the adapter and the sweep (rule 8b')."""
    if request is not None:
        return int(request)
    if _SOFTMAX_WGS_OVERRIDE_FOR_SWEEPS is not None:
        return int(_SOFTMAX_WGS_OVERRIDE_FOR_SWEEPS)
    if int(s_slots) == 2:
        # The double-buffered S (``TemplateParams.s_slots`` 2) is served with TWO softmax warp groups only (its MMA issue order hand-unrolls two
        # sub-tiles; ``_validate_tile_n``): a measured DEFAULT must never produce an invalid record, so a two-slot plan that requests no group
        # count takes two groups where the rule below would pick three.  A REQUEST of three (or the seam) passes verbatim and the factory declines it typed.
        return 2
    if (
        pair
        and int(cga) == 1
        and int(tile_n) == 64
        and has_bias
        and has_kv_mask
        and not padded
        and d_flavor in (32, 64)
        and s_kv is not None
        and int(s_kv) >= SM107_THREE_WG_S_KV_MIN
        and b_flat is not None
        and int(b_flat) <= SM107_THREE_WG_B_FLAT_MAX
    ):
        return 3
    return 2


# ``_S_SLOTS_OVERRIDE_FOR_SWEEPS`` (same rules as the three seams above) pins the S / P TMEM slot count per softmax sub-tile of the
# 32- / 64-wide f16/bf16 tiles at TILE_N 64 (``TemplateParams.s_slots``; :data:`SMALL_D_S_SLOTS`): 2 = the double-buffered S (the MMA
# warp issues BMM1(k + 1) into the free slot ahead of BMM2(k)), None = the derived count (the measured rule of :func:`sm107_s_slots_pick`:
# 2 on the plain body at TILE_N 64, 1 everywhere else).  Read by :func:`sm107_s_slots_pick`, which
# the adapter calls when it builds the record -- the value lands IN the TemplateParams, so the template-module cache (keyed on the
# record alone) and the compiled-plan cache both see two kernels for the two counts; a pinned value is returned VERBATIM, so a plan
# outside the arm's domain (TILE_N 128, d128, a quantized dtype) is a typed factory raise, never a silent one-slot rendering under a
# two-slot label (an A/B whose two arms share one record compares the first-built kernel with itself).
_S_SLOTS_OVERRIDE_FOR_SWEEPS: Optional[int] = None
# The S / P slot counts the small-tile body is validated at: 1 (the classic one slot per sub-tile) and 2 (the double-buffered S at
# TILE_N 64 -- the TMEM map 2 sub-tiles x 2 slots x 64 + O + STATS fits the 576 columns; at TILE_N 128 it does not, a 3-slot S fits
# nowhere: 3 x 2 x 128 = 768).


def sm107_s_slots_pick(
    d_flavor: int, tile_n: int, *, dtype_qkv: int = _DTYPE_BF16, request: Optional[int] = None, has_bias: bool = False, has_kv_mask: bool = False
) -> int:
    """The ``TemplateParams.s_slots`` a knob-less plan gets: ``request`` when the caller made one (returned verbatim; the factory
    validates it), else the sweep seam ``_S_SLOTS_OVERRIDE_FOR_SWEEPS`` (verbatim, see its note), else the MEASURED rule: **2 on the
    PLAIN body (no bias, no kv_mask) at TILE_N 64** of the f16/bf16 32- / 64-wide tiles, 1 everywhere else.  Measured (bf16, B 1, a 212-SM
    cc 10.7 perf node at the 2376 MHz lock, 100 x 3 graph-replay rounds, both slot counts in ONE process with a control twin, the kernel-only
    CUPTI time agreeing on every cell; speed-up = t(1 slot) / t(2 slots) - 1): the plain body WINS on every TILE_N 64 cell -- the served
    pick (``sm107_tile_n_pick`` (d): ``s_kv % 128`` in 1..64) at S 300: d32 +4.33 % (40.11 -> 38.45 us), d64 +6.63 % (47.10 -> 44.18 us,
    duty-cycled); through the public TILE_N 64 knob at S 384 / 1024 (d32) +9.53 / +8.44 % and S 384 / 512 (d64) +11.88 / +11.46 % -- the
    second slot lets the MMA warp issue BMM1(k + 1) of both sub-tiles ahead of BMM2(k), and the plain softmax body is the one short enough
    for the tensor pipe to be the wait (the gain GROWS on the dead-sub-tile quiet body: S 384 d32 +4.11 % on the kv tile body before it ->
    +9.53 % on it).  The bias + kv_mask TILE_N 64 forms LOSE on every cell (d32 S 384 / 1024 -1.67 / -5.93 %, d64 S 384 / 512 -0.69 /
    -0.98 %: the biased softmax body is the longer stage, so the earlier BMM1 only competes with BMM2 for the tensor pipe and the slot
    cursor's extra ring stages cost their waits), bias alone is MIXED (d32 S 384 -0.19 % inside the band, S 1024 -2.87 %; d64 S 384
    +1.41 %), a kv_mask alone is unmeasured: those keep one slot.  The causal and per-batch-length plain forms at TILE_N 64 run the
    two-slot body too (bitwise the one-slot kernel on O and Stats, verified; their perf is unmeasured -- the rule follows the FORM, which
    is what the MMA issue order sees).  ``has_bias`` / ``has_kv_mask`` are the adapter's graph facts; a caller that does not pass them
    (the sweep) gets the plain rule."""
    if request is not None:
        return int(request)
    if _S_SLOTS_OVERRIDE_FOR_SWEEPS is not None:
        return int(_S_SLOTS_OVERRIDE_FOR_SWEEPS)
    if int(tile_n) == SM107_SMALL_TILE_N_NARROW and int(d_flavor) in (32, 64) and int(dtype_qkv) > _DTYPE_E5M2 and not has_bias and not has_kv_mask:
        return 2
    return 1


# ``_Q_IN_TMEM_OVERRIDE_FOR_SWEEPS`` (the ``_S_SLOTS_OVERRIDE_FOR_SWEEPS`` rules: ``monkeypatch.setattr`` in a test or a sweep driver that resets
# it to None; never a plan path, an import or a config) pins ``TemplateParams.q_in_tmem`` of a knob-less plan on the f16/bf16 32- / 64-wide
# tiles: 1 = Q resident in TMEM (the MMA warp copies each sub-tile's Q SMEM -> TMEM once per n-step and runs BMM1 as an A-from-TMEM MMA; the
# copy's commit releases the Q slab at the start of the step), None = the derived value (:func:`sm107_q_in_tmem_pick`: the measured rule).
# Read by :func:`sm107_q_in_tmem_pick`, which the adapter calls when it builds the record -- the value lands IN the TemplateParams, so the
# template-module cache (keyed on the record alone) and the compiled-plan cache both see two kernels for the two forms; a pinned value is
# returned VERBATIM, so a plan outside the arm's domain (d128, a quantized dtype, cga2, the 64-row tile, three warp groups) is a typed
# factory raise, never a silent SMEM-Q rendering under a TMEM-Q label.  NOT in :func:`sweep_seam_state` (it moves a record field).
_Q_IN_TMEM_OVERRIDE_FOR_SWEEPS: Optional[int] = None
# The Q-operand forms the small-tile body is validated at: 0 (Q from SMEM, every committed record) and 1 (Q in TMEM).


def sm107_q_in_tmem_pick(
    d_flavor: int,
    *,
    dtype_qkv: int = _DTYPE_BF16,
    request: Optional[int] = None,
    tile_n: Optional[int] = None,
    has_bias: Optional[bool] = None,
    has_kv_mask: Optional[bool] = None,
    softmax_wgs: Optional[int] = None,
    cta_mma: Optional[int] = None,
    tile_m: Optional[int] = None,
    seq_kv_lens_present: Optional[bool] = None,
    dense_unsplit: Optional[bool] = None,
) -> int:
    """The ``TemplateParams.q_in_tmem`` a knob-less plan gets: ``request`` when the caller made one (returned verbatim; the factory validates
    it), else the sweep seam ``_Q_IN_TMEM_OVERRIDE_FOR_SWEEPS`` (verbatim, see its note), else the MEASURED rule on the record's own geometry
    (the keyword facts; a caller without them -- the sweep's bare call -- gets 0, the SMEM-Q form of every committed record):

    * **1 on the PLAIN body (no bias, no kv_mask) at TILE_N 128**, the one-batch-entry item: +3.1 / +4.3 / +3.3 / +3.5 % at d32 S 384 / 512 / 640 / 768
      with S flattened rows (43.30 -> 42.02, 72.51 -> 69.50, 140.67 -> 136.22, 205.47 -> 198.54 us), +6.8 / +4.5 % at S 256 / 384 with 4 S rows (48.00 ->
      44.96, 144.70 -> 138.53 us); d64 +2.8 / +2.6 / +3.1 / +2.8 % and +6.7 / +4.3 % on the same cells -- 14 of 14 cells a win.
    * **1 on the BIAS body without a kv_mask at TILE_N 64** (the pair item's resident block or streamed ring: the per-row ``kv_lengths`` leg of the
      training crops, and the dense cells the width rule puts on the 64-wide tile): +1.5 / +2.4 / +1.9 / +2.0 % at d32 S 384 / 512 / 640 / 768 with
      kv_lengths = 0.75 S (57.28 -> 56.45, 102.21 -> 99.82, 198.72 -> 195.12, 299.87 -> 293.89 us) and +1.6 / +2.5 / +1.9 / +1.8 % at 0.5 S; d64 +2.4..+2.9 %
      (0.75 S) and +1.7..+3.8 % (0.5 S), the dense d64 S 256 pair +3.8 % with 4 S rows -- 27 of 30 cells a win; the two NEUTRAL are d64 S 256 dense -0.2 %
      (inside its 0.71 % control spread) and d32 S 256 at 0.75 S +0.4 % (above its 0.26 % spread but under the +1 % win threshold), the one LOSS -0.4 %
      (d32 S 256, 0.5 S, two of four kv tiles).  The same body at **TILE_N 128 LOSES** (12 cells, geomean -0.45 %): -2.2 / -1.1 / -0.5 / -0.15 % at d32
      S 384 / 512 / 256 / 768 (55.39 -> 56.64, 102.30 -> 103.39, 23.87 -> 24.00, 295.10 -> 295.55 us), -1.1 % at d32 S 384 with 4 S rows (179.26 ->
      181.28 us), -0.4 % at d64 S 384 / 512; the other five are NEUTRAL (-0.05..+0.5 %: three inside their control spread, d64 S 640 / 768 +0.1 / +0.5 %
      above it but under the +1 % threshold) -- it keeps Q in SMEM.
    * **1 on the BIAS + kv_mask body at TILE_N 64 of the 64-wide tile WITHOUT per-row lengths**: +1.8 / +1.3 % at d64 S 384 / 512 (76.72 -> 75.36,
      160.98 -> 158.96 us) and +7.1 / +3.2 % at S 256 / 384 with 4 S rows (98.14 -> 91.65, 263.23 -> 255.09), S 256 inside the spread; the same body
      with ``kv_lengths`` is a wash (-1.4..+3.2 %, geomean +0.2 %) and at the 32-wide tile a wash (-0.6..+1.5 %, geomean +0.24 %) -- both keep Q in SMEM.
    * 0 on a kv_mask without a bias and on the plain TILE_N 64 body (unmeasured), and off the arm's validated domain (cga2, the 64-row tile, three
      softmax warp groups, a quantized dtype, THD / split-KV / PackGQA / the epilogue gate -- the factory would decline typed).

    Measured bf16, H 4, a 212-SM cc 10.7 node at the 2376 MHz lock (sampled 2328..2364 MHz, the power-cap counter unmoved), 100 x 3 CUDA-graph
    replay rounds, both forms in ONE process (two template modules) with a control twin (spreads 0.00..0.71 %), the kernel-only CUPTI time agreeing
    in sign on every cell above 20 us; speed-up = t(Q in SMEM) / t(Q in TMEM) - 1.  The gain is the TMA-LDG warp's earlier Q / K / V prefetch
    (the Q slab is released by the copy's commit at the start of the step, not after the step's last BMM1), so it grows where steps are short
    (few kv tiles per step: the 64-wide tile, the padded crops, four batch rows) and vanishes where the 128-wide biased softmax body is the long
    stage.  O and Stats are BITWISE between the two forms (the same MMAs over the same operands in the same k-step order).  The facts are the
    adapter's (``api_dsl``'s ``_q_in_tmem``): the record's TILE_N, the graph's bias / kv_mask, the softmax warp-group count, the cluster width,
    the Q tile height, the per-row-lengths flag and the dense / unsplit / non-PackGQA / ungated gates -- the same facts ``_validate_q_in_tmem`` pins."""
    if request is not None:
        return int(request)
    if _Q_IN_TMEM_OVERRIDE_FOR_SWEEPS is not None:
        return int(_Q_IN_TMEM_OVERRIDE_FOR_SWEEPS)
    facts = (tile_n, has_bias, has_kv_mask, softmax_wgs, cta_mma, tile_m, seq_kv_lens_present, dense_unsplit)
    if any(f is None for f in facts):
        return 0
    in_domain = (
        int(d_flavor) in (32, 64)
        and int(dtype_qkv) > _DTYPE_E5M2
        and int(cta_mma) == 1
        and int(tile_m) == 128
        and int(softmax_wgs) == 2
        and bool(dense_unsplit)
    )
    if not in_domain:
        return 0
    if not has_bias and not has_kv_mask:
        return 1 if int(tile_n) == SM107_SMALL_TILE_N_DEFAULT else 0
    if has_bias and not has_kv_mask:
        return 1 if int(tile_n) == SM107_SMALL_TILE_N_NARROW else 0
    if has_bias and has_kv_mask:
        return 1 if (int(tile_n) == SM107_SMALL_TILE_N_NARROW and int(d_flavor) == 64 and not seq_kv_lens_present) else 0
    return 0


# The ONE-BATCH-ENTRY item at the 32-wide tile's cga1 STREAMS its bias wherever its ring fits, resident budget or not: the 256-row
# item's block fill is exposed per item (nothing amortises it) and the ring overlaps the tile loads with the kv loop -- MEASURED
# (bf16, a 212-SM cc 10.7 node at the 2376 MHz lock, 100 x 3 graph-replay rounds, a control twin per process, the kernel-only CUPTI
# time at S 256) the 2-stage ring over the resident block: +6.4 % at S_q = S_kv = 256 with 256 flattened rows (22.11 -> 20.78 us),
# +10.0 % with 1024 rows (67.47 -> 61.34), +9.9 / +12.9 % at S 384 with 384 / 1536 rows (70.24 -> 63.90, 256.00 -> 226.74; the 3-stage
# ring +8.5 / +9.6 %).  The PAIR item keeps "resident where it fits" (its block IS amortised over the chunk: the 2-stage pair ring is
# -0.3 / -1.7 % at d32 S 384 / 768, a tie at d64), and so does the 64-wide tile (its one-batch-entry item has no cga1 ring; the cga2
# resident-vs-ring pick is unmeasured).  Measured at two and three kv tiles; at ONE tile the block stays (a ring has nothing to overlap).
SM107_ONE_BATCH_ENTRY_STREAMING_D_FLAVORS = frozenset({32})


def sm107_bias_mode_pick(
    d_flavor: int,
    cta_mma: int,
    s_kv: int,
    *,
    dtype_qkv: int = _DTYPE_BF16,
    tile_n: int = 128,
    pair: bool = False,
    request=None,
    tiles_q: int = 2,
    global_lengths: bool = False,
    natural: bool = False,
) -> int:
    """The ``TemplateParams.bias_mode`` a BIAS plan gets at this width: ``BIAS_MODE_RESIDENT`` when the resident block holds
    ``s_kv`` (:func:`sm107_bias_resident_s_kv_max`) -- except the one-batch-entry item at a width in
    :data:`SM107_ONE_BATCH_ENTRY_STREAMING_D_FLAVORS` (cga1), which takes ``BIAS_MODE_STREAMED`` whenever its ring fits (the measured
    pick above) -- else ``BIAS_MODE_STREAMED`` when a ring of >= 2 stages fits (:func:`sm107_bias_ring_stages`), else **0 = no form at
    this width** (a typed decline).  ``request`` (the adapter's ``bias_mode=`` or the test seam) pins the form and returns 0 when THAT
    form does not fit here -- honoured or declined, never degraded.  ONE arithmetic for the engine row, the heuristics' width pick and
    the adapter (rule 8b').  The PAIR item's ring takes its cluster-MULTICAST form (``BIAS_MODE_STREAMED_MULTICAST``) at a width in
    :data:`SM107_PAIR_RING_MULTICAST_D_FLAVORS` when the caller vouches for the two facts the form needs -- ``global_lengths`` (no
    per-batch KV / Q lengths; a dense S_kv tail the lowering serves through SYNTHESIZED per-batch lengths, ``engines._synth_kv_padding``,
    is NOT global either: the adapter sees ``seq_kv_lens_present`` and the row reads the same predicate, so a dense mask-free bias graph
    whose S_kv is not a KV-tile multiple keeps the cga1 ring) and ``natural`` (the NATURAL scheduler pairs the cluster's chunks) -- both
    False by default, so a caller that cannot vouch gets the cga1 ring as before (the adapter and the engine row pass them)."""
    want = request if request is not None else _BIAS_MODE_OVERRIDE_FOR_SWEEPS
    resident_ok = s_kv <= sm107_bias_resident_s_kv_max(d_flavor, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, pair=pair, tiles_q=tiles_q)
    ring_ok = sm107_bias_ring_stages(d_flavor, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, tiles_q=tiles_q, pair=pair) > 0
    if want is not None:
        want = int(want)
        if want == BIAS_MODE_RESIDENT:
            return BIAS_MODE_RESIDENT if resident_ok else BIAS_MODE_NONE
        if want == BIAS_MODE_STREAMED:
            return BIAS_MODE_STREAMED if ring_ok else BIAS_MODE_NONE
        if want == BIAS_MODE_STREAMED_MULTICAST:
            # The multicast ring is the PAIR item's ring at cga1 (two independent CTAs sharing one bias tile sequence): the same
            # SMEM ring must fit, the item must be the pair (the one-batch-entry item's two CTAs would read different bias rows).
            return BIAS_MODE_STREAMED_MULTICAST if (ring_ok and pair and int(cta_mma) == 1) else BIAS_MODE_NONE
        return BIAS_MODE_NONE
    if ring_ok and not pair and int(cta_mma) == 1 and int(d_flavor) in SM107_ONE_BATCH_ENTRY_STREAMING_D_FLAVORS and int(s_kv) > SM107_SMALL_TILE_N_DEFAULT:
        return BIAS_MODE_STREAMED  # measured at 2 and 3 kv tiles; one tile keeps the block (nothing to overlap)
    if resident_ok:
        return BIAS_MODE_RESIDENT
    if ring_ok and pair and int(cta_mma) == 1 and int(d_flavor) in SM107_PAIR_RING_MULTICAST_D_FLAVORS and global_lengths and natural:
        return BIAS_MODE_STREAMED_MULTICAST  # the pair ring's MEASURED default (the record above SM107_PAIR_RING_MULTICAST_D_FLAVORS)
    return BIAS_MODE_STREAMED if ring_ok else BIAS_MODE_NONE


# The PAIR item's streamed ring runs in its cluster-MULTICAST form (bias_mode 3: two chunk-adjacent CTAs of a (1, 1, 2) cluster share
# one bias tile sequence through one .multicast::cluster TMA load) by default at the 32-wide tile, where the pair's resident block
# does not hold S_kv -- MEASURED on the 212-SM Rubin perf-node class (lock 2376 per job, 100 x 3 graph replay, the control pair in
# the same process, the fp64 oracle on every timed cell): on the cells the heuristics serve with the pair's cga1 ring (S_q % 256 != 0,
# S_kv above the block's budget) the multicast ring is +3.38 % over the knob-less pick at D 32 S 896 (B 1; +3.16 / +3.29 % over the
# n4 / n8 ring, +5.36 % over the 2-stage ring) and the same arm-vs-arm delta holds on the pair ring at S 640 / 768 / 1024, B 1 and
# B 2 (+3.09 / +2.05 / +3.07 / +2.31 / +2.79 % at n4; +0.71 % at S 512 B 2, NEUTRAL at S 512 B 1).  NOT the 64-wide tile: -6.3 /
# -10.1 % at S 1024 / 512 (its wider bias row feeds two CTAs no better than one) -- the frozenset holds the measured flavor only.
# Where the pair's RESIDENT block holds S_kv the block stays (the multicast ring loses to it: -4.70 % at S 384, -1.87 % at S 640);
# the one-batch-entry item (S_q % 256 == 0) keeps its own ring, 5..9 % ahead of every pair form at S 512 / 768 / 1024.  The class is
# the pair ring under GLOBAL lengths: a dense S_kv tail (S_kv % 128 != 0 with no mask covering it) runs the kernel's padded path with
# synthesized per-batch lengths and keeps the cga1 ring -- unmeasured for the multicast form (every measured cell was a 128-multiple).
SM107_PAIR_RING_MULTICAST_D_FLAVORS = frozenset({32})

# The one-batch-entry item's streamed ring runs 2 stages, not the deepest that fits: its 64 KiB stage x 3 buys a lookahead the
# 2-sub-tile item never uses -- MEASURED +3.6 / +2.0 / +1.5 % at d32 S 512 / 768 / 1024 (B = 1), +3.2 / +2.1 / +1.6 % at B = 4,
# +2.0 / +1.2 % with a kv_mask, 6 of 6 processes (the sweep's node and protocol, see sm107_tile_n_pick).  The PAIR item's ring keeps
# the deepest: 2 stages are a tie there (d64 +0.3..+0.8 %) or a loss (d32 -0.3 / -1.7 %).
SM107_ONE_BATCH_ENTRY_RING_STAGES = 2


def sm107_bias_stages_pick(
    d_flavor: int, cta_mma: int, *, dtype_qkv: int = _DTYPE_BF16, tile_n: int = 128, pair: bool = False, request: int = 0, tiles_q: int = 2
) -> int:
    """The ``STAGES_BIAS`` of a STREAMED plan: the request (``TemplateParams.bias_stages`` / the adapter's ``bias_stages=`` /
    the test seam) when it is in ``BIAS_RING_STAGES`` and fits, else the derived deepest ring for the PAIR item and
    ``min(deepest, SM107_ONE_BATCH_ENTRY_RING_STAGES)`` for the one-batch-entry item; 0 = no ring fits (the factory raises, the row /
    adapter decline).  A request that does not fit is 0 too (declined typed, never degraded)."""
    deepest = sm107_bias_ring_stages(d_flavor, cta_mma, dtype_qkv=dtype_qkv, tile_n=tile_n, tiles_q=tiles_q, pair=pair)
    want = int(request) if request else (_BIAS_STAGES_OVERRIDE_FOR_SWEEPS or 0)
    if want:
        return want if (want in BIAS_RING_STAGES and want <= deepest) else 0
    return deepest if pair else min(deepest, SM107_ONE_BATCH_ENTRY_RING_STAGES)


def sm107_bias_auto_cga(
    d_flavor: int, s_kv: int, cgas, *, dtype_qkv: int = _DTYPE_BF16, tile_n: int = 128, pair: bool = False, tiles_q: int = 2
) -> Optional[int]:
    """The NARROWEST cga in ``cgas`` whose resident block holds ``s_kv`` (``sm107_bias_resident_s_kv_max``), or ``None``
    when no width does.  The heuristics' auto pick for a BIAS graph (a kv_mask alone stages no block and keeps the flavor's
    own width rule), the width the standalone adapter lowers
    with when none is requested, and the engine row's knob-less eligibility all read it (rule 8b'): a d64 graph at S_kv
    256 lands on cga2 (two tiles) without a knob, S_kv <= 128 keeps the tuned cga1.  ``pair`` = the graph qualifies for the
    (n, n+1) item (a batch-broadcast bias on >= 2 rows): the 128-row block is evaluated at cga1 only (the pair runs cga1), the
    256-row one at every other width.  Second pass (the STREAMED fallback): when no width's resident block holds ``s_kv``, the
    narrowest width whose ring fits (:func:`sm107_bias_mode_pick` != 0) -- a per-batch bias at the 64-wide tile above S_kv 256
    lands on cga2 streamed, the pair item stays cga1.  ``None`` when neither form fits anywhere."""
    for cga in sorted(cgas):
        if sm107_bias_mode_pick(d_flavor, cga, s_kv, dtype_qkv=dtype_qkv, tile_n=tile_n, pair=pair and cga == 1, tiles_q=tiles_q) == BIAS_MODE_RESIDENT:
            return cga
    for cga in sorted(cgas):
        if sm107_bias_mode_pick(d_flavor, cga, s_kv, dtype_qkv=dtype_qkv, tile_n=tile_n, pair=pair and cga == 1, tiles_q=tiles_q) != BIAS_MODE_NONE:
            return cga
    return None


def sm107_n_chunk_rule(b_flat: int, h_q: int, s_q: int, sm_count: int, *, tile_m: int = 128, waves: int = 3, rows_per_step: int = 2) -> int:
    """The default chunk length of the (n, n+1) PAIR item: the largest power of two ``<= h_q * ceil(s_q / tile_m) * b_flat /
    (waves * sm_count)`` (the ">= 3 waves" reading of the plan's parallelism-vs-amortisation trade), clamped to
    ``[rows_per_step, b_flat]`` and a MULTIPLE OF ``rows_per_step`` -- EVEN on the pair (2 rows per step: an odd chunk runs a dead
    sub-tile on its last step), a multiple of three with the third softmax warp group (``softmax_wgs`` 3: a chunk that is not one ends in
    one or two DEAD rows the body recomputes).  ``sm_count`` 0 / None = 148.  Returns 0 (the one-batch-entry item) when ``b_flat < 2``."""
    if b_flat < 2:
        return 0
    rps = max(1, int(rows_per_step))
    sms = int(sm_count) if sm_count else 148
    items = h_q * (-(-s_q // tile_m)) * b_flat
    n = 1
    while n * 2 <= items // (waves * sms):
        n *= 2
    n = max(rps, min(n, b_flat))
    n -= n % rps
    return max(rps, n)


# A PRIVATE, process-local TEST seam (None = the rule above; an int pins :func:`sm107_n_chunk_pick`'s result, 0 = the
# one-batch-entry item): the (n, n+1) item's chunk length is a performance axis of the frontend band (an autotune knob,
# append-only in knobs.h) once BOTH pybind extensions carry the enum member; until that rebuild the graph path cannot carry
# it, so a sweep driver or a test pins the pick here -- through ``monkeypatch.setattr`` (or a driver that resets it to None),
# never from a plan path or at import.  A set value changes the plan the graph BUILDS and its ELIGIBILITY (the pick feeds
# :func:`sm107_pair_item_active`, i.e. the engine row's mismatch, the heuristics and the adapter), so it is not configuration
# and is not part of the module's public surface.
_N_CHUNK_OVERRIDE_FOR_SWEEPS: Optional[int] = None


# The (n, n+1) pair item's chunk is capped at 8: the 3-wave rule's larger picks LOSE wherever they were measured against n8 --
# the RESIDENT block n16 -> n8 +3.1 / +3.7 % at 1536 rows (d32 / d64), n32 -> n8 +2.9 % at 2048 rows (d32, with a kv_mask), the
# earlier S = 768 B = 1 miss (16 vs 8, +3.9 %), n64 -> n16 +5.8 % at 3072 rows (n8 unmeasured there); the streamed RING (nothing to
# amortise over the chunk) n16 / n32 -> n4 / n8 at d64 S_kv 640 / 768 / 896 (+2.4 / +3.8 / +5.5 % for n4, n8 within 1 % of n4, two
# GPUs), n32 -> n16 +1.0 % at 1024.  n8 itself is the measured best at 512 rows and stays (see sm107_tile_n_pick for the sweep).
SM107_PAIR_N_CHUNK_MAX = 8
# At ONE 128-wide kv tile the per-item fixed cost (the block fill, the item barriers) dominates and a 4-row chunk amortises it -- as
# long as the chunking leaves the grid under-filled: n4 beats the clamp-2 pick by +8..+16 % at 128 flattened rows (every D, both
# widths) while at 512 rows (4.8 waves of 2-row items) n2 wins back by +8 % on the 64-wide tile.  The floor applies while the 2-row
# items number at most this many waves.  CONFIRMED on the training crops (same node class, same protocol, same-process arm vs pick): at
# S 256 with 256 rows the clamp-2 pick wins at every live fraction (1.0 / 0.75 / 0.5 of the keys -- the 0.5 crop is ONE live 128-wide
# tile, where the floor does not reach because the 2-row items fill 4.8 waves: n4 -3..-10 %, n8 -17..-30 %, both D), the cap at 8 wins
# at 1024 / 1536 rows (n16 -8.6..-16.3 % at 1024 rows on every form at every fraction; at 1536 rows -5.8..-8.7 % on the 64-wide forms
# and -0.7 / -1.6 % (D 32 / D 64) on the 128-wide bias pair, where the +3.1 / +3.7 % of the record above did not reproduce -- n8 still
# ahead, the D 32 margin under the 1 % floor; n4 -0.3..-2.5 % on every other B 4 cell except -8.7 % against the one-batch-entry pick
# at D 32 S 256 with 1024 rows and +1.3..+1.6 % on three DENSE 64-wide cells at D 64 B 4 -- a single-fraction reading, not a default),
# the D 64 pair ring's n8 is within 1 % of n4 at S 640 / 768.  The one reproducible exception: on the padded S 384 cells with 384 rows
# the pair n2 beats the n4 pick at 0.75 of the keys live (+3.0 / +3.2 % D 32, +1.7 / +1.5 % D 64; +2.7..+3.9 % with a kv_mask on top)
# and not at 0.5 (+1.0 / +1.0 % D 32, -0.7 / +0.2 % D 64) or
# at the full length (-0.9 / -1.6 %) -- the pick sees S_kv, never the lengths, and a single-fraction win is recorded, not folded.
SM107_ONE_TILE_N_CHUNK_FLOOR, SM107_ONE_TILE_FLOOR_MAX_WAVES = 4, 2


def sm107_n_chunk_refine(n: int, b_flat: int, h_q: int, s_q: int, sm_count: int, *, s_kv: int, tile_m: int = 128, rows_per_step: int = 2) -> int:
    """The knob sweep's corrections to :func:`sm107_n_chunk_rule`'s 3-wave pick for the (n, n+1) pair item (the constants above
    carry the numbers): the pick is capped at :data:`SM107_PAIR_N_CHUNK_MAX`; at one 128-wide kv tile
    (``s_kv <= SM107_SMALL_TILE_N_DEFAULT``) it is floored at :data:`SM107_ONE_TILE_N_CHUNK_FLOOR` while the 2-row items fill at most
    :data:`SM107_ONE_TILE_FLOOR_MAX_WAVES` waves.  The result stays a multiple of ``rows_per_step`` (even on the pair) inside
    ``[rows_per_step, b_flat]`` (:func:`sm107_n_chunk_rule`'s clamp); with the third softmax warp group (``rows_per_step`` 3) the cap and the
    floor are rounded DOWN to multiples of three (8 -> 6: n9 measured -2.7..-3.6 % against n6; 4 -> 3)."""
    rps = max(1, int(rows_per_step))
    sms = int(sm_count) if sm_count else 148
    n = min(n, max(rps, SM107_PAIR_N_CHUNK_MAX - SM107_PAIR_N_CHUNK_MAX % rps))
    items_at_2 = h_q * (-(-s_q // tile_m)) * (-(-b_flat // 2))
    if s_kv <= SM107_SMALL_TILE_N_DEFAULT and items_at_2 <= SM107_ONE_TILE_FLOOR_MAX_WAVES * sms:
        n = max(n, max(rps, SM107_ONE_TILE_N_CHUNK_FLOOR - SM107_ONE_TILE_N_CHUNK_FLOOR % rps))
    n = max(rps, min(n, b_flat))
    n -= n % rps
    return max(rps, n)


def sm107_n_chunk_pick(
    bias_batch_extent: int,
    b_flat: int,
    h_q: int,
    s_q: int,
    sm_count: int,
    cta_mma: int,
    *,
    s_kv: Optional[int] = None,
    d_flavor: Optional[int] = None,
    tile_n: int = 128,
    rows_per_step: int = 2,
) -> int:
    """The ``TemplateParams.n_chunk`` a knob-less BIAS plan gets: 0 unless :func:`sm107_bias_pair_form` holds, else the
    test seam ``_N_CHUNK_OVERRIDE_FOR_SWEEPS`` (0 = the one-batch-entry item, for its A/B arm) or :func:`sm107_n_chunk_rule` refined by
    :func:`sm107_n_chunk_refine` -- behind ONE measured guard:
    at ``s_q % 256 == 0`` with ``s_kv`` inside the one-batch-entry block's budget the 256-row super-tile wastes no Q rows and its
    two sub-tiles SHARE every K / V tile, where the pair streams them twice, so the one-batch-entry item is the faster form
    (S_q = S_kv = 256, d32: 25.44 vs 26.50 us at n_chunk 2, 26.66 at 4, 31.26 at 8 -- a 212-SM cc 10.7 node at the 2376 MHz lock,
    sampled 2364 MHz, one process, control pair 0.24 %); the pair wins where a super-tile would be partly empty (S_q 128: +30.6 %,
    384: +30.0 %, each against the one-batch-entry item in the same process) or where only its block holds S_kv (512+).
    ``s_kv`` / ``d_flavor`` None skips the guard (callers that pin the form through the override or a request never reach it).  The guard
    covers every ``s_kv`` the one-batch-entry item serves at the width (its block or its streamed ring): +8.6 % over the pair's resident block
    at S 768, +6..7 % over the pair's ring at 1024 / 2048 / 4096 (d32).  **The guard holds at ``tile_n`` 128 only** (the plan's KV tile
    width, :func:`sm107_tile_n_pick`): on the 64-wide tile the pair's resident block beats every one-batch-entry form -- n2 at S 256
    (+2.8 %), n8 at S 512 / 768 with a kv_mask (+12 / +13 % over the one-batch-entry ring at the same width), n4 / n8 at S 256 with
    1024 rows (+7 %); at S 1024 the pair ring and the one-batch-entry ring tie there."""
    if not sm107_bias_pair_form(bias_batch_extent, b_flat, cta_mma):
        return 0
    if _N_CHUNK_OVERRIDE_FOR_SWEEPS is not None:
        return max(0, int(_N_CHUNK_OVERRIDE_FOR_SWEEPS))
    if s_kv is not None and d_flavor is not None and s_q % 256 == 0 and int(tile_n) == SM107_SMALL_TILE_N_DEFAULT:
        # A full 256-row super-tile wastes no Q rows and SHARES every K / V tile across its two sub-tiles, where the pair streams them
        # twice; the pair's one advantage -- one resident block amortised over the chunk -- does not outweigh it: the one-batch-entry item
        # wins wherever a FORM of it fits at this width (its resident block -- the guard's first measured shape, S_q = S_kv = 256 above;
        # its streamed ring -- MEASURED on a 212-SM cc 10.7 node at the 2376 lock, one process per cell, control pairs <= 0.11 %:
        # S_q = S_kv = 768, d32, sampled 2364 MHz: the
        # one-batch-entry ring 303.86 us vs the pair's RESIDENT block 330.10 (+8.6 %) vs the pair's ring 332.86; 1024: 675.84 vs the pair's
        # ring 725.49 (+7.3 %); 2048: 5040.62 vs 5389.50 (+6.9 %); 4096 (both at a capped 2160 MHz): 44823.51 vs 47563.59 (+6.1 %)).  At the
        # 64-wide tile's cga1 the one-batch-entry item has no ring (sm107_bias_ring_stages = 0) above its 128-column block, so the pair
        # streams there.
        if sm107_bias_mode_pick(d_flavor, cta_mma, s_kv, pair=False) != BIAS_MODE_NONE:
            return 0
    n = sm107_n_chunk_rule(b_flat, h_q, s_q, sm_count, rows_per_step=rows_per_step)
    if s_kv is None:
        return n
    return sm107_n_chunk_refine(n, b_flat, h_q, s_q, sm_count, s_kv=s_kv, rows_per_step=rows_per_step)


def sm107_pair_item_active(
    bias_batch_extent: int,
    b_flat: int,
    h_q: int,
    s_q: int,
    sm_count: int,
    cta_mma: int,
    n_chunk_request=None,
    *,
    s_kv: Optional[int] = None,
    d_flavor: Optional[int] = None,
    tile_n: int = 128,
) -> bool:
    """Whether a BIAS plan RUNS the (n, n+1) pair item -- the form predicate AND a chunk length of at least 1 (the request when
    the caller made one, else :func:`sm107_n_chunk_pick` at the plan's KV tile width ``tile_n``; 0 from either = the one-batch-entry
    item).  THE predicate the engine row, the heuristics and the adapter evaluate the budget of (rule 8b'): a plan must never be
    admitted on the 128-row block's budget and built on the 256-row block."""
    if not sm107_bias_pair_form(bias_batch_extent, b_flat, cta_mma):
        return False
    n = (
        int(n_chunk_request)
        if n_chunk_request is not None
        else sm107_n_chunk_pick(bias_batch_extent, b_flat, h_q, s_q, sm_count, cta_mma, s_kv=s_kv, d_flavor=d_flavor, tile_n=tile_n)
    )
    return n >= 1


# The same fact keyed by the config-flavor string `_validate_params` receives
# (`make_cfg_d256` and `make_cfg_d256_mxfp8`).  Must stay in step with
# SM107_EPILOGUE_GATE_SHAPES: every dtype family of a listed shape carries the
# seams, and only those flavors may load with epilogue_gate=True.
_EPILOGUE_GATE_FLAVORS = frozenset({"sm107 d256", "sm107 d256 mxfp8"})

# Softmax levers (TemplateParams.softmax_f16 / softmax_scale_prefolded), per FLAVOR: a kernel port flips its
# flavor in, and _validate_params declines everything else so an unlisted flavor can never trace the default
# chain under the request (a silent no-op -- the failure the backstop exists for).
#   softmax_f16 -- the f16x2 exponent arm (softmax_precision=HALF): a QUANTIZED-kernel specialization; the listed
#     flavors carry it for FP8 / MXFP8 inputs and never for half inputs (the half kernels run the f32 exponent).
#   softmax_scale_prefolded -- the caller pre-multiplied Q by attn_scale * log2 e and the kernel traces no per-score
#     scale: wired in the MXFP8 and half bodies of the listed flavors.  Per-tensor FP8 keeps folding
#     descale_q * descale_k into the softmax scale in-kernel, so the fold is declined there whatever the flavor.
SM107_SOFTMAX_F16_FLAVORS = frozenset(
    {
        "sm107 d64",
        "sm107 d128",
        "sm107 d192xd128",
        "sm107 d256",
        "sm107 d512",
        "sm107 d128 mxfp8",
        "sm107 d192xd128 mxfp8",
        "sm107 d256 mxfp8",
        "sm107 d512 mxfp8",
    }
)
SM107_SCALE_PREFOLDED_FLAVORS = frozenset(
    {"sm107 d128", "sm107 d192xd128", "sm107 d256", "sm107 d512", "sm107 d128 mxfp8", "sm107 d192xd128 mxfp8", "sm107 d256 mxfp8", "sm107 d512 mxfp8"}
)


def epilogue_gate_layout_declarable(shape_bhsd: tuple, stride_bhsd: tuple, elem_bytes: int = 2) -> bool:
    """Whether a logical BHSD gate ``G`` binds ZERO-COPY on the Rubin d256 gated
    kernels: BSHD-compact, or a token-major declaration TMA can express
    (``config_sm100.bshd_zero_copy_stride``: D innermost-contiguous, then heads,
    then tokens; seq/head strides 16-byte multiples; covering).

    The gate has NO normalisation-copy fallback (a hidden per-execute repack is
    what the engine contract forbids), so this predicate is STRICTER than
    ``graph_analyzer.dense_layout_ok`` -- which admits any B/H/S stride order
    and any 16-byte-unaligned stride because the Q/K/V/O path copies those.  It
    is the ONE rule both sides of rule 8b consume: ``engines.mismatch`` applies
    it to G before admitting a gated graph, and
    ``api_dsl.SdpaFwdDslSm100.check_support`` raises on exactly the layouts it
    rejects, so a plan that entered the ranked list can never die on G's layout
    in the lowering.  ``elem_bytes`` defaults to the 2 B every Rubin gate kernel
    stores (``_CfgSm107.GATE_BPE``: Q-dtype on the half kernels, bf16 on the
    quantized one).
    """
    return bshd_compact(shape_bhsd, stride_bhsd) or bshd_zero_copy_stride(shape_bhsd, stride_bhsd, elem_bytes) is not None


# ---------------------------------------------------------------------------
# Derivation helpers (Rubin values — several differ from the SM100 twins)
# ---------------------------------------------------------------------------


def bpe(dtype: int) -> int:
    """Bytes per STORAGE element. FP8/MXFP8 = 1 (the block-scaled O codes are
    byte containers too), BF16/FP16 = 2."""
    return 1 if (dtype <= _DTYPE_E5M2 or dtype in (_DTYPE_O_NVFP4, _DTYPE_O_MXFP8)) else 2


def o_pack_div(dtype_o: int) -> int:
    """Logical O elements per storage byte: 2 for E2M1, else 1."""
    return 2 if dtype_o == _DTYPE_O_NVFP4 else 1


def resolve_dtype_o(params: TemplateParams) -> int:
    """``TemplateParams.dtype_o`` uses -1 as "inherit dtype_qkv"; every consumer
    must resolve it before use. Passing the raw -1 into ``bpe()`` would silently
    yield BPE_O=1 (an FP8-sized O buffer) for a half-precision output."""
    return params.dtype_qkv if params.dtype_o < 0 else params.dtype_o


def tile_k_hw(dtype_qkv: int) -> int:
    """HW k-step in elements.

    **This is the single most arch-sensitive value in the file.** Rubin runs the
    dense 2xFP8 MMA at K=64 per instruction and pairs it with idesc ``k_dim=1``;
    Blackwell has no K=64 FP8 path and uses K=32 with ``k_dim=0``. The pairing is
    arch-OPPOSITE, and a mismatch does not fail — it silently scrambles rows of
    the accumulator (see rules/mma-tma-matrix.md S1). The kernel bodies here
    hardcode ``k_dim=1``, so FP8/MXFP8 must be 64.

    BF16/FP16 use 16 on both arches: the 2-chunk (K=32) f16 path is silently
    wrong on SM10x.
    """
    return 64 if dtype_qkv <= _DTYPE_E5M2 else 16


def rescale_threshold(dtype_qkv: int) -> float:
    """4.0 for FP8/MXFP8 (low precision, rescale aggressively); 8.0 for
    BF16/FP16 (defer rescale to amortize the alpha == 1 fast path)."""
    return 4.0 if dtype_qkv <= _DTYPE_E5M2 else 8.0


# 14-bit ``start_address`` of a version-0 tcgen05 SMEM descriptor: every
# descriptor-fed slab (MMA operand or UTCCP scale-factor source) of a module
# that pins DESC_VERSION=0 must START under this, or it wraps to offset 0 and
# the MMA/UTCCP reads the bottom of SMEM (rules/mma-tma-matrix.md S6).

_MX_BLOCK = 32  # MXFP8: 32 elements share one E8M0 scale byte


def _d256_mxfp8_sf_sizes(tile_m: int, tile_n: int, tile_k: int, tile_o: int) -> Tuple[int, int, int, int]:
    """(Q, K-per-stage, P, V-per-stage) scale-factor slab sizes in bytes, the
    kernel's own formulas (``sm107/prefill_d256_mxfp8.py`` SF_SMEM_SIZE_*): a
    128-row tile x ceil(D/128)*128 / 32 for the rowwise operands, TILE_N/32 x
    the 128-padded non-K extent for the BMM2 pair.  Full-size regardless of
    CTA_MMA (the cga2 peers multicast halves into the same slab)."""
    k128 = -(-tile_k // 128) * 128
    return (
        tile_m * k128 // _MX_BLOCK,
        tile_n * k128 // _MX_BLOCK,
        (-(-tile_m // 128) * 128) * tile_n // _MX_BLOCK,
        (-(-tile_o // 128) * 128) * tile_n // _MX_BLOCK,
    )


def d256_mxfp8_sf_smem_bytes(tile_m: int, tile_n: int, tile_k: int, tile_o: int, stages_kv: int) -> int:
    """SMEM the d256 MXFP8 kernel spends on its four scale-factor slabs, as the
    allocator lays them out (declaration order sQ_SF | sK_SF[stages] | sP_SF |
    sV_SF[stages], each slab 1024-B aligned -- the 512 B P slab is padded to the
    next KiB).  At d256, STAGES_KV=2: 1 + 2 + 1 + 2 = 6 KiB."""
    sf_q, sf_k, sf_p, sf_v = _d256_mxfp8_sf_sizes(tile_m, tile_n, tile_k, tile_o)
    return _align_slab(sf_q) + _align_slab(stages_kv * sf_k) + _align_slab(sf_p) + _align_slab(stages_kv * sf_v)


def d256_mxfp8_last_sf_tile_start(cfg) -> int:
    """Byte offset of the LAST scale-factor stage (sV_SF[STAGES_KV-1]) of the
    d256 MXFP8 kernel for ``cfg`` -- the highest address any of its version-0
    UTCCP descriptors names.  Config-side twin of the kernel's own
    ``_last_sf_tile_start`` (a test pins that the two agree), so a depth or an
    O dtype that pushes the scale factors past the 256 KiB window is a typed
    ``ValueError`` at config build rather than LSE = +inf / O = NaN at run time.

    Layout (declaration order in ``_kernel``, each slab 1024-B aligned):
    sQO alias (byte-max of Q@BPE and O@BPE_O -- a bf16 O doubles it) | sK ring |
    sV ring | sQ_SF | sK_SF ring | sP_SF | sV_SF ring.  The epilogue gate tile
    is declared AFTER sV_SF and is not in this sum; declared before it, a bf16 O
    at STAGES_KV=2 would put sQ_SF at exactly 262144.
    """
    sf_q, sf_k, sf_p, sf_v = _d256_mxfp8_sf_sizes(cfg.TILE_M, cfg.TILE_N, cfg.TILE_K, cfg.TILE_O)
    off = 0
    for nbytes in (
        max(cfg.TILE_M * cfg.TILE_K * cfg.BPE, cfg.TILE_M * cfg.TILE_O * cfg.BPE_O),  # sQO alias
        cfg.STAGES_KV * (cfg.TILE_N * cfg.TILE_K // cfg.CTA_MMA) * cfg.BPE,  # sK ring
        cfg.STAGES_KV * (cfg.TILE_O * cfg.TILE_N // cfg.CTA_MMA) * cfg.BPE,  # sV ring
        sf_q,  # sQ_SF
        cfg.STAGES_KV * sf_k,  # sK_SF ring
        sf_p,  # sP_SF
    ):
        off = _align_slab(off + nbytes)
    return off + (cfg.STAGES_KV - 1) * sf_v


@dataclass(frozen=True)
class TmaIters:
    """TMA inner-loop iteration counts and per-iteration element granularity."""

    QK_ITERS: int
    VO_ITERS: int
    QK_GRANU_ELEMS: int
    VO_GRANU_ELEMS: int


def _tma_iters_for(d_elems: int, bpe_val: int, swz_b: int) -> int:
    """TMA inner-loop iters when d*BPE exceeds the swizzle atom: each TMA-tile
    load expands into inner_bytes/swz_b sub-loads stepped by swz_b/BPE elems."""
    inner_bytes = d_elems * bpe_val
    if inner_bytes % swz_b != 0:
        raise ValueError(f"TMA inner bytes {inner_bytes} is not a multiple of swizzle {swz_b}")
    return inner_bytes // swz_b


def _tma_iters(cfg) -> TmaIters:
    qk = _tma_iters_for(cfg.TILE_K, cfg.BPE, cfg.Q_SWZ_BYTES)
    vo = _tma_iters_for(cfg.TILE_O, cfg.BPE, cfg.V_SWZ_BYTES)
    return TmaIters(QK_ITERS=qk, VO_ITERS=vo, QK_GRANU_ELEMS=cfg.TILE_K // qk, VO_GRANU_ELEMS=cfg.TILE_O // vo)


def _mask_flags_from(params: TemplateParams) -> int:
    """Kernel-facing MASK bits, derived from the band + padding. The kernels fold
    trace-time branches on these bits; the band VALUES ride WINDOW_LEFT /
    WINDOW_RIGHT (0 when the corresponding bit is unset)."""
    flags = MASK_NONE
    if params.window_right is not None:
        flags |= MASK_CAUSAL
    if params.window_left is not None:
        flags |= MASK_SWA
    if params.thd_varlen or params.seq_kv_lens_present or params.kv_tail_mask:
        flags |= MASK_PADDED
    return flags


def _validate_params(flavor: str, k: TemplateParams, *, split_wired: bool = False, block_scaled_o_wired: bool = False) -> None:
    """Guard the TemplateParams a Rubin flavor can express. Every rejection here
    must also be a Capabilities decline — reaching this is an engine-row bug.

    ``split_wired`` identifies kernels with both SplitHelpers and FP32
    partial-output stores: half and per-tensor FP8 D128/D192, plus per-tensor FP8 D256. The engine
    row and standalone adapter expose the same flavor domain.

    ``block_scaled_o_wired`` says whether THIS flavor's kernel carries the
    block-scaled O epilogue (DTYPE_O 4 = NVFP4 / 5 = MXFP8 output). The two d128
    kernels (per-tensor FP8 and MXFP8) do; the d192xd128 siblings share the d128
    config family but accept DTYPE_O 0..3 only, so a flavor-name test
    ("d128" in flavor) would let them through to a specialization error."""
    if k.decode_q_tile:
        # The loader routes a decode_q_tile record to sm107/decode_d256_f16.py
        # (make_cfg_d256_decode at the end of this module); a Rubin prefill template
        # must never consume one (the config_sm100 twin makes the same refusal).
        raise ValueError(f"{flavor}: decode_q_tile={k.decode_q_tile} selects the d256 decode tile; the prefill templates do not consume it")
    # Fused epilogue gate FIRST, so an interaction decline names the feature the
    # caller asked for (`TemplateParams(epilogue_gate=True, split_kv=2)` reads
    # "epilogue_gate is dense, unsplit ...", not the generic Rubin split
    # message below).  Only the flavors whose BODIES carry the gate seams may
    # load with it -- an unlisted flavor would trace an ungated epilogue and
    # silently return O instead of O * sigmoid(G).  The feature interactions
    # are declined here as well as in the rows/adapter twin: a KV split would
    # gate the partials the combine then re-normalizes, paged KV and PackGQA
    # are simply not wired through the gate TMA coordinates.  THD (packed
    # sequences) x gate is wired on the f16/bf16 "sm107 d256" body ONLY: its
    # host maps G in the packed [1, T, H, D] form Q / O take and its TMA-LDG
    # warp's gate load already runs on the THD coordinates; the quantized
    # bodies (per-tensor FP8 on the same flavor string, MXFP8) build the dense
    # gate map and their prepared THD launch carries no gate.
    if k.qsa_block_topk or k.qsa_block_size or k.qsa_softmax_groups:
        # An index-list record (qsa_block_topk != 0, or its softmax-split knob) belongs to sm107/sparse_d256_f16.py (make_cfg_d256_sparse); a dense
        # template would read K/V densely and silently ignore the block list.  The selection is the FUNCTION -- a decline,
        # never a knob fallback (the sm100 twin's _validate_params / make_cfg_d256_decode decline the same way).
        raise ValueError(
            f"{flavor}: qsa_block_topk={k.qsa_block_topk} / qsa_block_size={k.qsa_block_size} / qsa_softmax_groups={k.qsa_softmax_groups} "
            f"select the index-list sparse kernel; the dense templates do not consume them"
        )
    if k.epilogue_gate:
        if flavor not in _EPILOGUE_GATE_FLAVORS:
            raise ValueError(f"{flavor}: epilogue_gate is wired on {sorted(_EPILOGUE_GATE_FLAVORS)} only")
        if (k.split_kv or 1) > 1 or k.pack_gqa or k.paged_kv:
            raise ValueError(f"{flavor}: epilogue_gate is unsplit, unpaged, non-PackGQA only")
        if k.thd_varlen and not (flavor == "sm107 d256" and k.dtype_qkv in (_DTYPE_BF16, _DTYPE_FP16)):
            raise ValueError(
                f"{flavor}: epilogue_gate under THD (packed sequences) is wired on the f16/bf16 'sm107 d256' body only "
                f"(the quantized bodies build a dense gate map); got dtype_qkv={k.dtype_qkv}"
            )
    # Paged KV: the two MXFP8 flavors whose TMA-LDG warp walks the block tables (_PAGED_KV_FLAVORS_SM107); every
    # other Rubin flavor refuses here, before its module body could trace dense K/V descriptors over a page pool.
    # Dense queries only (THD queries over pools are the stage-2 port: the quantized THD setup clamps the K/V
    # descriptors to a packed total), unsplit and unpacked, and a page is a whole number of 128-row F8_128x4 SF atom
    # sets (the SF pools page with K/V; a 64-row page cannot hold one).
    if k.paged_kv:
        if flavor not in _PAGED_KV_FLAVORS_SM107:
            raise ValueError(
                f"{flavor}: paged_kv is not wired on this Rubin flavor (the cc 10.7 PAGED_KV specialization lives in "
                f"sm107/prefill_d128_f16.py, sm107/prefill_d128_mxfp8.py and sm107/prefill_d256_mxfp8.py; served: {sorted(_PAGED_KV_FLAVORS_SM107)})"
            )
        if not k.seq_kv_lens_present:
            raise ValueError(f"{flavor}: paged_kv requires seq_kv_lens_present (the per-batch KV length bounds the block-table walk)")
        if flavor == "sm107 d128":
            if k.dtype_qkv not in (_DTYPE_BF16, _DTYPE_FP16) or k.d_flavor != 128:
                raise ValueError(f"{flavor}: paged_kv is not wired for this dtype/tile; native paged half prefill requires the D128 half body")
            p = k.page_size
            if p < 8 or p % 8 or (p < 128 and 128 % p) or (p > 128 and p % 128):
                raise ValueError(f"{flavor}: page_size must be a multiple of 8 dividing 128, or a multiple of 128; got {p}")
        else:
            if k.thd_varlen:
                raise ValueError(
                    f"{flavor}: THD queries over MXFP8 pools are not wired on cc 10.7 (the quantized THD setup clamps K/V descriptors to a packed total)"
                )
            if k.pack_gqa or (k.split_kv or 1) > 1:
                raise ValueError(f"{flavor}: paged_kv serves unsplit, unpacked plans only")
            if k.page_size <= 0 or k.page_size % 128 != 0:
                raise ValueError(
                    f"{flavor}: paged MXFP8 KV needs page_size to be a positive multiple of 128 (whole F8_128x4 SF atom sets per page); got {k.page_size}"
                )
    elif k.page_size:
        raise ValueError(f"{flavor}: page_size requires paged_kv=True")
    if k.dtype_qkv not in (_DTYPE_E4M3, _DTYPE_E5M2, _DTYPE_BF16, _DTYPE_FP16):
        raise ValueError(f"{flavor}: dtype_qkv must be 0=E4M3/1=E5M2/2=BF16/3=FP16 (got {k.dtype_qkv}); Rubin has no TF32 prefill kernel")
    dtype_o = resolve_dtype_o(k)
    if dtype_o not in (_DTYPE_E4M3, _DTYPE_E5M2, _DTYPE_BF16, _DTYPE_FP16, _DTYPE_O_NVFP4, _DTYPE_O_MXFP8):
        raise ValueError(f"{flavor}: dtype_o must be 0..5 (got {k.dtype_o})")
    if dtype_o in (_DTYPE_O_NVFP4, _DTYPE_O_MXFP8):
        if k.dtype_qkv > _DTYPE_E5M2:
            raise ValueError(f"{flavor}: block-scaled O (dtype_o {dtype_o}) requires FP8 inputs")
        if not block_scaled_o_wired:
            raise ValueError(f"{flavor}: block-scaled O (dtype_o {dtype_o}) is wired in the d128 kernels (per-tensor FP8, MXFP8) only")
        if k.thd_varlen or k.seq_q_lens_present or (k.split_kv or 1) > 1 or k.pack_gqa or k.paged_kv:
            raise ValueError(f"{flavor}: block-scaled O (dtype_o {dtype_o}) serves dense, unsplit, unpacked, unpaged graphs only")
    if k.dtype_qkv > _DTYPE_E5M2 and dtype_o != k.dtype_qkv:
        raise ValueError(f"{flavor}: half input (BF16/FP16) requires dtype_o == dtype_qkv; got dtype_o={dtype_o}")
    quantized = k.dtype_qkv in (_DTYPE_E4M3, _DTYPE_E5M2)
    if k.softmax_f16:
        if not quantized:
            raise ValueError(f"{flavor}: softmax_f16 is a quantized-kernel (FP8 / MXFP8) specialization (half inputs run the f32 exponent)")
        if flavor not in SM107_SOFTMAX_F16_FLAVORS:
            raise ValueError(f"{flavor}: softmax_f16 is wired on {sorted(SM107_SOFTMAX_F16_FLAVORS)} only")
    if k.softmax_scale_prefolded:
        if quantized and not flavor.endswith(" mxfp8"):
            raise ValueError(
                f"{flavor}: softmax_scale_prefolded is not served on per-tensor FP8 (the kernel folds descale_q * descale_k into the softmax scale)"
            )
        if flavor not in SM107_SCALE_PREFOLDED_FLAVORS:
            raise ValueError(f"{flavor}: softmax_scale_prefolded is wired on {sorted(SM107_SCALE_PREFOLDED_FLAVORS)} only")
    if k.sched_policy not in (None, SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2):
        raise ValueError(f"{flavor}: sched_policy must be NATURAL/LPT/LPT_L2 or None (got {k.sched_policy})")
    if k.qh_per_kh < 1:
        raise ValueError(f"{flavor}: qh_per_kh ({k.qh_per_kh}) must be >= 1")
    if k.split_kv and k.split_kv > 1 and not split_wired:
        raise ValueError(f"{flavor}: split_kv > 1 is not wired in this SM107 kernel (no SplitHelpers)")
    # THD/varlen is per-FLAVOR on the Rubin line, not per-dtype.  Every
    # QUANTIZED flavor carries it; on the f16/bf16 side only the flavors whose
    # BODY has been ported to the FROST setup-kernel contract do -- the rest
    # still call it with the pre-upstream 7-arg signature against a 14-arg
    # helper, and allocate the 3B+2 metadata buffer where the SHARED decode
    # (_common_blackwell._thd_decode) reads 4B+4 with a batch_remap.  That
    # mismatch is a HANG or a wrong batch, not an arity error, so it is declined
    # here rather than left to fail deep in a trace.
    if k.thd_varlen and k.dtype_qkv not in (_DTYPE_E4M3, _DTYPE_E5M2) and flavor not in _F16_THD_FLAVORS:
        raise ValueError(
            f"{flavor}: THD/varlen on the SM107 f16/bf16 line is served by {sorted(_F16_THD_FLAVORS)} only "
            f"(got dtype_qkv={k.dtype_qkv}); the other flavors' setup-kernel call sites are not ported"
        )
    # The MXFP8 twin of the gate above (the flavor string carries the family: the
    # per-tensor FP8 d256 config is "sm107 d256", the MXFP8 one "sm107 d256 mxfp8").
    # A pre-upstream MXFP8 body loaded with THD_VARLEN=1 would call the 14-arg setup
    # helper with 7 arguments and size its SF descriptors from host kwargs the
    # prepared host never passes -- a trace error at best, a zero-extent descriptor
    # (cudaErrorIllegalInstruction) at worst -- so it is declined here, where the
    # row (`thd_d_shapes = SM107_MXFP8_THD_SHAPES`) already declines it.
    if k.thd_varlen and k.dtype_qkv in (_DTYPE_E4M3, _DTYPE_E5M2) and flavor.endswith(" mxfp8") and flavor not in _MXFP8_THD_FLAVORS:
        raise ValueError(
            f"{flavor}: THD/varlen on the SM107 MXFP8 line is served by {sorted(_MXFP8_THD_FLAVORS)} only; "
            f"the other MXFP8 bodies keep the pre-upstream THD arm (7-arg setup call, 3B+2 metadata, static K/V maps)"
        )
    if k.seq_q_lens_present:
        if k.thd_varlen:
            raise ValueError(f"{flavor}: SEQ_Q_LENS_PRESENT is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
        if not k.seq_kv_lens_present:
            raise ValueError(f"{flavor}: SEQ_Q_LENS_PRESENT requires SEQ_KV_LENS_PRESENT (padding mask)")


def _band_fields(params: TemplateParams) -> Tuple[int, int, int, int, int]:
    """(MASK_FLAGS, WINDOW_LEFT, WINDOW_RIGHT, BOTTOM_RIGHT, HAS_SINK).

    WINDOW_* are 0 when their mask bit is unset. Passing window_left without
    window_right is silently wrong rather than partial — the KV-tile trim already
    uses WINDOW_RIGHT, so the element-level mask must receive it too."""
    return (
        _mask_flags_from(params),
        params.window_left or 0,
        params.window_right or 0,
        int(params.bottom_right),
        int(params.has_sink),
    )


# ---------------------------------------------------------------------------
# Shared field surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CfgSm107(D128FamilySmallTileFields):
    """Fields every Rubin prefill flavor carries.

    Per-flavor factories set every field explicitly; the defaults exist so the
    dataclass is constructible in a doc/test context, not as a fallback.
    """

    # --- tile geometry
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 128  # d_qk
    TILE_O: int = 128  # d_v

    # --- dtypes
    DTYPE_QKV: int = _DTYPE_FP16  # 0=E4M3 1=E5M2 2=BF16 3=FP16
    DTYPE_O: int = _DTYPE_FP16
    BPE: int = 2
    BPE_O: int = 2
    # Block-scaled O (per-tensor FP8 d128 only): scale block along d (0 = plain
    # O) and logical O elements per storage byte (2 for E2M1).
    O_BLOCK_SCALE: int = 0
    O_PACK_DIV: int = 1

    # --- cluster
    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2
    SPLIT_PIPELINE: int = 0

    # --- swizzles
    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128

    # --- MMA k-step (arch-sensitive; see tile_k_hw)
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16

    # --- pipeline depth
    TILES_Q: int = 2
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 4

    # --- warp roles
    SOFTMAX_WARPGROUPS: int = 2
    CORRECTION_WARPS: int = 4
    SOFTMAX_REGS: int = 192
    CORRECTION_REGS: int = 88
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    RESCALE_THRESHOLD: float = 8.0

    # --- graph-derived features
    MASK_FLAGS: int = MASK_NONE
    # (PADDED_TAIL_FORM / DEAD_SUBTILE_QUIET and the bias / kv_mask / pair / ring / max-form / S_SLOTS / BIAS_MCAST_CTAS / Q_IN_TMEM field
    # block are the inherited D128FamilySmallTileFields of config_d128_family -- ONE field block for the Rubin and the SM100 d128-file
    # bodies; every seam folds on them, so their defaults trace the committed instruction stream byte-for-byte.)
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    BOTTOM_RIGHT: int = 0
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    PACK_GQA: int = 0
    PACK_G: int = 0  # 0 preserves whole-group packing on flavors without an explicit group.
    QH_PER_KH: int = 1
    SPLIT_KV: int = 1
    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0
    THD_VARLEN: int = 0
    # Paged KV cache (TemplateParams.paged_kv / page_size): D128 half and D128/D256 MXFP8 factories set them.
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL

    N_BMM2_CHUNKS: int = 2
    BMM2_CHUNK_SIZE: int = 64

    # --- warp population (READ_TILE_ARRIVERS must equal the number of warps that
    # actually call read_tile_id_arrive; a mismatch is an unreachable mbarrier
    # count, i.e. a hang at every shape)
    TOTAL_WARPS: int = 16
    THREADS_PER_CTA: int = 16 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4
    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4
    CORR_WARP_BASE: int = 8
    MMA_WARP_ID: int = 12
    TMALDG_WARP_ID: int = 13
    TMASTG_WARP_ID: int = 14
    SCHED_WARP_ID: int = 15

    # --- mbarrier lane constants
    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_LANES: int = 128
    CORR_LANES: int = 128
    SOFTMAX_PLUS_CORR: int = 256
    READ_TILE_ARRIVERS: int = 15

    # --- fused epilogue gate (O := O * sigmoid(G)); every seam in the kernel
    # bodies folds on cutlass.const_expr(CFG.EPILOGUE_GATE), so 0 traces the
    # pre-gate instruction stream byte-for-byte.  Only the d256 factory ever
    # sets it (config_sm107._EPILOGUE_GATE_FLAVORS); d128/d192/d512 inherit 0.
    EPILOGUE_GATE: int = 0
    # Bytes per gate element: the gate is Q-dtype on the half kernels and bf16
    # on the quantized one -- both 2 B.  Every gate constant in a kernel (TMA
    # box, swizzle, subtile walk) derives from THIS, never from BPE_O: with an
    # e4m3 O the two geometries differ, and a gate walk borrowed from O's
    # passes every bf16-O test and fails only e4m3-O.
    GATE_BPE: int = 2


@dataclass(frozen=True)
class CfgD128(_CfgSm107):
    """d_qk = d_v = 128 (the classic 2-warpgroup pipeline)."""

    QO_ALIAS: int = 0


@dataclass(frozen=True)
class CfgD64(CfgD128):
    """d_qk = d_v = 64 on the d128 body: the same 16-warp pipeline with ``TILE_K = TILE_O = 64`` (BMM1 in 4 k-steps, a
    64-wide O accumulator, half the Q/K/V/O SMEM).  Serves every f16/bf16 head dim in (32, 64] with D % 8 == 0 by TMA zero
    fill (the SM100 ``CfgD64`` precedent, which measured the zero-filled d128 tile taking the SAME time at d = 64)."""

    TILE_K: int = 64
    TILE_O: int = 64


@dataclass(frozen=True)
class CfgD32(CfgD128):
    """d_qk = d_v = 32 on the d128 body: ``TILE_K = TILE_O = 32`` (BMM1 in 2 k-steps, a 32-wide O accumulator, a quarter of
    the d128 SMEM; 64-byte Q/K/O swizzle atoms).  Serves every f16/bf16 head dim <= 32 with D % 8 == 0 by TMA zero fill.
    At this width the MMA is a quarter of d128's per score element while the softmax's exp / FMA work per element is unchanged,
    so the kernel's floor is the softmax issue rate, not the tensor core."""

    TILE_K: int = 32
    TILE_O: int = 32


@dataclass(frozen=True)
class CfgD192(CfgD128):
    """d_qk = 192, d_v = 128. Same pipeline as d128; Q/K swizzle drops to 64 B at
    FP8 because TILE_K * 1 = 192 is not a multiple of 128."""


@dataclass(frozen=True)
class CfgD256(_CfgSm107):
    """d_qk = d_v = 256. Q-union-O SMEM alias, single softmax warpgroup, a
    Q.K(i+1) -> S.V(i) lookahead MMA stream with two parity S_acc TMEM slots."""


@dataclass(frozen=True)
class CfgD512(_CfgSm107):
    """d_qk = d_v = 512, cga4x1 role-split: the cluster's 4 CTAs form two 2-CTA
    sub-groups (sg0 = BMM1 + softmax + ship, sg1 = recv + correction + BMM2)."""

    XFER_STAGES: int = 2
    BMM2_N_PER_CALL: int = 256
    BMM2_LOOP_N_BLOCKS: int = 2


# ---------------------------------------------------------------------------
# The d128-family TMEM column map -- ONE formula for the kernel body and the validator
# ---------------------------------------------------------------------------

# The top of the allocation holds the softmax stats: 8 columns per sub-tile (alpha; the final (max, sum); the score side
# outputs' true max in columns 2-3) for the two sub-tiles, plus 16 spare.  Keeping STATS at the TOP (TOTAL_COLS - 32 = 544)
# rather than right after the last O slot leaves the d128 map exactly what the kernel always had (S 0 / 128, O 256 / 384,
# STATS 544), so the d128 rendering is byte-identical to the committed records; at the small widths the gap between the
# last O slot and STATS simply grows.


def d128_family_tmem_layout(cfg, *, stats_cols_per_subtile: int = 8) -> D128FamilyTmemLayout:
    """The d128-family TMEM map of a Rubin Cfg: ``config_d128_family.d128_family_tmem_layout`` at the 576-column allocation (STATS at the
    top, 544 at the default stats width).  At d128 this is the map the kernel always had (0 / 128 | 64 / 192 | 256 / 384 | 544)."""
    return _d128_family_tmem_layout(cfg, total_cols=TMEM_TOTAL_COLS, stats_cols_per_subtile=stats_cols_per_subtile)


# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------


def _validate_head_dim_tile(cfg, flavor: str) -> None:
    """The head-dim tile and everything coupled to it (``config_d128_family.validate_head_dim_tile`` at ``SM107_ARCH``)."""
    _validate_head_dim_tile_shared(cfg, flavor, arch=SM107_ARCH)


def _validate_dtype_k_step(cfg, flavor: str) -> None:
    """FP8/MXFP8 must ride Rubin's K=64 dense path (the bodies hardcode idesc
    k_dim=1); f16/bf16 must stay 1-chunk at 16."""
    want = tile_k_hw(cfg.DTYPE_QKV)
    _check(
        [
            (
                cfg.TILE_K_HW_BMM1 == want and cfg.TILE_K_HW_BMM2 == want,
                f"{flavor}: TILE_K_HW must be {want} at DTYPE_QKV={cfg.DTYPE_QKV} on Rubin "
                f"(got {cfg.TILE_K_HW_BMM1}/{cfg.TILE_K_HW_BMM2}); a mismatch against the body's "
                f"idesc k_dim silently scrambles accumulator rows",
            ),
        ]
    )


def _validate_smem(cfg, flavor: str, *, qo_alias: bool, sf_bytes: int = 0, bias_bytes: int = 0) -> None:
    """The SMEM fit against the Rubin budget (``config_d128_family.validate_smem`` at ``SM107_ARCH``: 320 KiB usable minus the 2 KiB fixed
    overhead; ``sf_bytes`` / ``bias_bytes`` named in the message)."""
    _validate_smem_shared(cfg, flavor, arch=SM107_ARCH, qo_alias=qo_alias, sf_bytes=sf_bytes, bias_bytes=bias_bytes)


def _validate_masks(cfg, flavor: str) -> None:
    _check(
        [
            (
                not (cfg.BOTTOM_RIGHT and not (cfg.MASK_FLAGS & MASK_CAUSAL)),
                f"{flavor}: bottom-right alignment requires a causal band",
            ),
            (
                not (cfg.THD_VARLEN and not (cfg.MASK_FLAGS & MASK_PADDED)),
                f"{flavor}: THD/varlen implies per-sequence padded masking",
            ),
            (
                not (cfg.THD_VARLEN and not cfg.SEQ_KV_LENS_PRESENT),
                f"{flavor}: THD/varlen must force SEQ_KV_LENS_PRESENT=1, or every sequence attends the whole pack",
            ),
        ]
    )
    # The small-tile mask-arm couplings (the dead-sub-tile quiet body, the pre-masked padded tail): the shared predicates.
    _validate_small_tile_masks(cfg, flavor)


# ---------------------------------------------------------------------------
# d128 / d192 -- the classic 2-warpgroup pipeline (16 warps)
# ---------------------------------------------------------------------------
#
# Warp map: 8 softmax (2 warpgroups) + 4 correction + MMA + TMALDG + TMASTG +
# scheduler.  15 of the 16 warps arrive on the scheduler ring (all but the
# scheduler warp itself).


_D128_TOTAL_WARPS = 16
_D128_READ_TILE_ARRIVERS = 15
# The d128-file warp table as a function of the softmax WARP-GROUP count (2 = the 16-warp body above; 3 = the 20-warp body of the
# TILE_N-64 pair arm, TemplateParams.softmax_wgs): 4 warps per softmax group, 4 correction warps, then the MMA / TMA-LDG / TMA-STG /
# scheduler warps -- the kernel's role dispatch reads the Cfg ids and derives WG2's first warp as SOFTMAX_WG0_BASE + 2 x SOFTMAX_WG_WARPS.


# The per-role register split (SOFTMAX_REGS, CORRECTION_REGS) of the d128-file body by (softmax warp groups, head-dim tile); the four
# 40-register roles are fixed.  SUM(role regs x warps) EQUALS the launch's entry pool (reg_entry_pool: 128 x 16 = 2048; 96 x 20 = 1920),
# as every CUTLASS warp-specialized split does: 8 x 192 + 4 x 88 + 4 x 40 = 2048; 12 x 128 + 4 x 56 + 4 x 40 = 1920 at the 32-wide tile
# (the correction holds a 32-column O row); 12 x 120 + 4 x 80 + 4 x 40 = 1920 at the 64-wide one (a 64-column O row needs 80 -- 56 spills
# -- so the softmax gives up 8).  The 128-register softmax of the 3-WG arm holds ONE 64-column chunk: reg_S 64 + the packed P chunk 32 +
# the loop state, no bias pre-load (the kernel derives _BIAS_PRELOAD_EFF 0 there).  frost-kernels.md s2.


def _stages_kv_d128(dtype_qkv: int, cta_mma: int, *, mxfp8: bool, tile_k: int) -> int:
    """Rubin d128-family ring depth.

    Per-tensor FP8 at d128/cga2 runs a **9-stage** ring: a Rubin-specific tuning
    carried by the shipped ``sm107/prefill_d128_fp8.py`` (which previously spelled
    it as a post-hoc ``dataclasses.replace``), and the reason that kernel needs
    the oversized-SMEM launch mode.

    It is deliberately narrow. At cga1 the per-CTA K/V slabs double and 9 stages
    need 352 KiB, past the 327 KiB cap; at d192 (``tile_k=192``) the Q/K slabs
    grow and the tuning was never measured. Both fall back to 4, which is what
    the kernel bodies were validated with.
    """
    if dtype_qkv <= _DTYPE_E5M2:
        if not mxfp8 and tile_k == 128 and cta_mma == 2:
            return 9
        return 4
    # d192's wider Q/K slabs do not leave room for a 4-deep ring: Q/K/V/O alone
    # sum to exactly 320 KiB, which overflows the usable carveout once barriers
    # are counted. Measured on Rubin at n_kv=2: depth 4 -> 50 % of O zeroed,
    # cos = 0.0015; depth 2 -> cos = 1.000000. d128 at depth 4 sums to 256 KiB
    # and passes every n_kv, so this is a d192 constraint, not a family one.
    if tile_k == 192:
        return 2
    if tile_k <= 64:
        # The small head-dim tiles (TILE_K = TILE_O in {32, 64}): a quarter / half of d128's slabs, so a 4-deep ring is
        # 96 / 192 KiB at cga1 (64 / 128 KiB at cga2), inside the carveout at either width, and 4 is the depth the SM100 d64
        # flavor runs (its ncu read long_scoreboard as the dominant stall).  A perf lever, not a body constraint (the ring
        # is depth-generic; the body indexes no accumulator parity from it; ``PipelineState.advance`` runs the index and the
        # phase over any depth >= 1 on both the TMA-LDG and the MMA side, the pair item's two tiles per kv step included).
        # ``_STAGES_KV_OVERRIDE_FOR_SWEEPS`` (2 / 3 / 4, the depths the SMEM validator and the body were validated at) pins
        # the depth for the A/B of the small-tile knob sweep; it is read HERE so the bias budgets (``sm107_bias_resident_tiles``
        # / ``sm107_bias_ring_stages``, which call this helper) and the factory see ONE depth (rule 8b').  Measured on a
        # 212-SM cc 10.7 node at the 2376 MHz lock (bf16, graph replay, control pair; the knob sweep's A/B/A): see the
        # STAGES_KV row of the small-tile heuristics notes before moving the default.
        if _STAGES_KV_OVERRIDE_FOR_SWEEPS is not None:
            want = int(_STAGES_KV_OVERRIDE_FOR_SWEEPS)
            if want not in SMALL_D_STAGES_KV:
                raise ValueError(
                    f"_STAGES_KV_OVERRIDE_FOR_SWEEPS={want}: the 32- / 64-wide tiles' K / V ring is validated at {sorted(SMALL_D_STAGES_KV)} stages "
                    f"(SMEM fit + the ring-wrap axis of the fresh-process census); a depth outside it is unvalidated, not merely slow"
                )
            return want
        return 4
    return 2 * cta_mma


# The flavors that serve the bias / kv_mask at SOME validated cga -- DERIVED from the resident budget (the d128 tile has no
# spare for a block at either cga: its Q / O / K / V sum to 256 KiB).  The engine row's ``bias_d_shapes`` and the adapter
# read this set (rule 8b').  Defined here, after the ring-depth rule the budget reads.
SM107_BIAS_D_SHAPES = frozenset(
    shape
    for shape in SM107_F16_SMALL_D_SHAPES | {(128, 128)}
    if any(sm107_bias_resident_tiles(shape[0], cga) > 0 for cga in SM107_SMALL_D_CGAS.get(shape, {2}))
)


def _validate_tile_n(cfg, flavor: str) -> None:
    """The KV tile width and what it couples to (``config_d128_family.validate_tile_n`` at ``SM107_ARCH``: the 576-column map)."""
    _validate_tile_n_shared(cfg, flavor, arch=SM107_ARCH)


def _validate_bias(cfg, flavor: str) -> None:
    """The resident bias block and the kv_mask word, every coupling as a predicate (``config_d128_family.validate_bias`` at ``SM107_ARCH``)."""
    _validate_bias_shared(cfg, flavor, arch=SM107_ARCH)


def _validate_q_in_tmem(cfg, flavor: str) -> None:
    """Q resident in TMEM and everything the body couples to it (``config_d128_family.validate_q_in_tmem`` at ``SM107_ARCH``)."""
    _validate_q_in_tmem_shared(cfg, flavor, arch=SM107_ARCH)


def _make_cfg_d128_family(params: TemplateParams, *, flavor: str, tile_k: int, tile_o: int, mxfp8: bool):
    # Per-tensor FP8 D64/D128/D192 and half D128/D192 share FP32 split partials.
    # MXFP8 has no split loop yet.
    split_shapes = SM107_FP8_SPLIT_SHAPES if params.dtype_qkv <= _DTYPE_E5M2 else {(128, 128), (192, 128)}
    split_wired = not mxfp8 and (tile_k, tile_o) in split_shapes
    # The block-scaled O epilogue lives in the two d128 kernels (per-tensor fp8 and
    # mxfp8); the d192xd128 siblings share this config family but not the epilogue.
    block_scaled_o_wired = tile_k == 128 and tile_o == 128
    _validate_params(flavor, params, split_wired=split_wired, block_scaled_o_wired=block_scaled_o_wired)
    # softmax_f16 / softmax_scale_prefolded: per-flavor backstops in _validate_params (SM107_*_FLAVORS).
    cta_mma = params.cta_mma
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    o_div = o_pack_div(dtype_o)
    # TILE_N is a Cfg field (TemplateParams.tile_n): the quantized siblings keep 128 (their SF geometry is tied to the 128-column
    # tile), the f16/bf16 small tiles take 64 / 128 (SM107_TILE_NS_BY_D_SHAPE; the 64-wide rendering is trace-compiled and served
    # through the TILE_N knob), and the half D128 rendering takes 64 / 128; 256 is refused by _validate_tile_n.
    tile_n = int(getattr(params, "tile_n", 128))
    if tile_n != 128 and (params.dtype_qkv <= _DTYPE_E5M2 or tile_k > 128 or (tile_k == 128 and tile_n != 64)):
        raise ValueError(f"{flavor}: TILE_N = {tile_n} requires a supported half-precision width: D128 permits 64/128; quantized renderings keep 128")
    # The S / P slot count (TemplateParams.s_slots, appended; getattr so a record without the field renders one slot): 2 = the
    # double-buffered S of the f16/bf16 small tiles at TILE_N 64 -- _validate_tile_n pins the couplings (the TMEM fit, the width, the dtype).
    s_slots = int(getattr(params, "s_slots", 1))
    if s_slots != 1 and (params.dtype_qkv <= _DTYPE_E5M2 or mxfp8 or tile_k > 128):
        raise ValueError(f"{flavor}: s_slots = {s_slots} requires f16/bf16 TILE_K <= 128; quantized and wider renderings keep one slot")
    # Q resident in TMEM (TemplateParams.q_in_tmem, appended; getattr so a record without the field renders Q from SMEM): the f16/bf16
    # small tiles at cga1 / TILE_M 128 / two sub-tiles -- _validate_q_in_tmem pins every coupling (the TMEM fit, the UTCCP geometry, the dtype).
    q_in_tmem = int(getattr(params, "q_in_tmem", 0))
    if q_in_tmem not in SMALL_D_Q_IN_TMEM:
        raise ValueError(f"{flavor}: q_in_tmem must be 0 (Q from SMEM) or 1 (Q resident in TMEM); got {q_in_tmem}")
    if q_in_tmem and (params.dtype_qkv <= _DTYPE_E5M2 or mxfp8 or tile_k > 64):
        raise ValueError(
            f"{flavor}: q_in_tmem=1 (Q resident in TMEM) is wired on the f16/bf16 32- / 64-wide tiles only (the quantized and d128 renderings keep Q in SMEM: "
            f"at TILE_K 128 the Q slots would end at column {2 * 128 + 2 * 128 + 2 * 64} of the 576, past the STATS region)"
        )
    bias_mode = int(getattr(params, "bias_mode", 0))
    kv_mask_form = int(getattr(params, "kv_mask_form", 0))
    if bias_mode or kv_mask_form:
        if params.dtype_qkv <= _DTYPE_E5M2 or mxfp8:
            raise ValueError(f"{flavor}: the additive bias / kv_mask are wired on the f16/bf16 prefill kernel only (dtype_qkv={params.dtype_qkv})")
        if params.thd_varlen or (params.split_kv or 1) > 1 or params.pack_gqa or params.paged_kv or params.epilogue_gate:
            raise ValueError(f"{flavor}: the additive bias / kv_mask serve dense, unsplit, unpaged, non-PackGQA, ungated graphs only")
        if bias_mode not in BIAS_MODES or kv_mask_form not in KV_MASK_FORMS:
            raise ValueError(
                f"{flavor}: bias_mode must be 0 / 1 (resident row-block) / 2 (streamed ring) / 3 (streamed ring, cluster multicast) and kv_mask_form "
                f"0 / 1 (bit word) / 2 / 3 (additive io-dtype / fp32 vector); got {bias_mode} / {kv_mask_form}"
            )
    stages_kv = _stages_kv_d128(params.dtype_qkv, cta_mma, mxfp8=mxfp8, tile_k=tile_k)
    # The (n, n+1) pair item (TemplateParams.n_chunk, appended; read through getattr so a record without the field renders
    # the one-batch-entry item): a 128-row block, cga1, the bias arm only -- _validate_bias pins the couplings.
    n_chunk = int(getattr(params, "n_chunk", 0))
    pair = n_chunk > 0
    if pair and bias_mode == BIAS_MODE_NONE:
        raise ValueError(f"{flavor}: n_chunk={n_chunk} is the (n, n+1) pair item of the BIAS arm; it needs bias_mode=1 / 2 (got {bias_mode})")
    bias_stages_req = int(getattr(params, "bias_stages", 0))
    if bias_stages_req and bias_mode not in BIAS_STREAMED_MODES:
        raise ValueError(f"{flavor}: bias_stages={bias_stages_req} is the STREAMED ring's depth; it needs bias_mode=2 / 3 (got {bias_mode})")
    # The Q tile height (TILE_M): 128 on every flavor but the f16/bf16 small tiles, where the private seam may pin 64 (the TILE_M = 64
    # lever; ``_tile_m_d128`` is the ONE helper the budgets below read too, so the Q / O slabs, the block rows and the budgets agree).
    tile_m = _tile_m_d128(tile_k) if (tile_k <= 64 and params.dtype_qkv > _DTYPE_E5M2 and not mxfp8) else 128
    # The row-max form (appended; a record without the field renders the software tree): the hardware upper bound needs an arm whose body it
    # replaces and a kv_mask it can bound (none or the bit word) -- the additive vector is a typed decline here and in the adapter.
    bias_max_form = int(getattr(params, "bias_max_form", 0))
    if bias_max_form not in BIAS_MAX_FORMS:
        raise ValueError(f"{flavor}: bias_max_form must be 0 (the software row max) or 1 (the hardware upper bound); got {bias_max_form}")
    if bias_max_form == BIAS_MAX_HW_UPPER_BOUND:
        if not (bias_mode or kv_mask_form):
            raise ValueError(
                f"{flavor}: bias_max_form=1 is the row-max form of the bias / kv_mask body; this record has neither arm (bias_mode=0, kv_mask_form=0)"
            )
        if kv_mask_form in (KV_MASK_ADDITIVE_16, KV_MASK_ADDITIVE_F32):
            raise ValueError(
                f"{flavor}: bias_max_form=1 (the hardware upper-bound row max) is served with the bit-word kv_mask or none; the additive kv_mask vector "
                f"(kv_mask_form={kv_mask_form}) has no per-tile max to bound -- keep bias_max_form=0 with it"
            )
    # The softmax warp-group count (TemplateParams.softmax_wgs, appended; read through getattr so a record without the field renders the
    # classic two-group body): 3 = the third softmax warp group of the TILE_N-64 pair item -- the backstops below mirror
    # _validate_softmax_wgs (the adapter's check_support declines typed first); the warp table, the split and the arriver counts derive
    # from it so no two of them can disagree.
    softmax_wgs = int(getattr(params, "softmax_wgs", 2))
    if softmax_wgs not in SMALL_D_SOFTMAX_WGS:
        raise ValueError(f"{flavor}: softmax_wgs must be one of {sorted(SMALL_D_SOFTMAX_WGS)}; got {softmax_wgs}")
    if softmax_wgs == 3:
        if params.dtype_qkv <= _DTYPE_E5M2 or mxfp8 or tile_k > 64:
            raise ValueError(f"{flavor}: softmax_wgs=3 (the third softmax warp group) is wired on the f16/bf16 32- / 64-wide tiles only")
        if tile_n != 64 or not pair or cta_mma != 1 or tile_m != 128:
            raise ValueError(
                f"{flavor}: softmax_wgs=3 serves the (n, n+1) pair item at TILE_N 64, cga1 and the 128-row Q tile only (got tile_n={tile_n}, n_chunk={n_chunk}, cta_mma={cta_mma}, tile_m={tile_m}): "
                f"its 128-register softmax holds one 64-column chunk and the one-batch-entry paths hand-unroll two sub-tiles"
            )
    softmax_regs, correction_regs = _D128_REG_SPLIT[2 if softmax_wgs == 2 else (softmax_wgs, tile_k)]
    total_warps = _d128_total_warps(softmax_wgs)
    corr_warp_base = softmax_wgs * _D128_SOFTMAX_WG_WARPS
    mma_warp_id = corr_warp_base + _D128_CORRECTION_WARPS
    bias_tiles = 0
    stages_bias = 0
    if bias_mode == BIAS_MODE_RESIDENT:
        bias_tiles = sm107_bias_resident_tiles(tile_k, cta_mma, dtype_qkv=params.dtype_qkv, tile_n=tile_n, tile_m=tile_m, tiles_q=softmax_wgs, pair=pair)
        if bias_tiles == 0:
            raise ValueError(
                f"{flavor}: no resident bias block fits beside the Q / O slabs and the {stages_kv}-deep K / V ring at TILE_K={tile_k}, cta_mma={cta_mma} "
                f"({SMEM_USABLE_BYTES // 1024} KiB usable); the bias is served on the widths of SM107_BIAS_D_SHAPES={sorted(SM107_BIAS_D_SHAPES)}"
            )
    elif bias_mode in BIAS_STREAMED_MODES:
        stages_bias = sm107_bias_stages_pick(
            tile_k, cta_mma, dtype_qkv=params.dtype_qkv, tile_n=tile_n, pair=pair, request=bias_stages_req, tiles_q=softmax_wgs
        )
        if stages_bias == 0:
            raise ValueError(
                f"{flavor}: no {sorted(BIAS_RING_STAGES)}-stage streamed bias ring of {sm107_bias_block_rows(softmax_wgs, tile_m, pair=pair)} x {tile_n} tiles "
                f"{'(requested bias_stages=' + str(bias_stages_req) + ') ' if bias_stages_req else ''}fits beside the Q / O slabs and the {stages_kv}-deep "
                f"K / V ring at TILE_K={tile_k}, cta_mma={cta_mma} ({SMEM_USABLE_BYTES // 1024} KiB usable)"
            )
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)
    # The padded tail form (TemplateParams.padded_tail_form, appended; read through getattr so a record without the field renders the
    # legacy tail body): the pre-masked tail serves the f16/bf16 small tiles' PLAIN padded body only -- a band has masked segments on
    # both sides, the bias / kv_mask arms run the -inf body (one body already), the d128 flavor keeps its own rendering.
    padded_tail_form = int(getattr(params, "padded_tail_form", 0))
    if padded_tail_form not in PADDED_TAIL_FORMS:
        raise ValueError(f"{flavor}: padded_tail_form must be 0 (the legacy tail body) or 1 (the pre-masked tail); got {padded_tail_form}")
    if padded_tail_form == 1 and (mask_flags != MASK_PADDED or bias_mode or kv_mask_form or tile_k > 64 or mxfp8 or params.dtype_qkv <= _DTYPE_E5M2):
        raise ValueError(
            f"{flavor}: padded_tail_form=1 (the pre-masked tail) is wired on the f16/bf16 32- / 64-wide tiles' PLAIN padded body only "
            f"(MASK_FLAGS == MASK_PADDED alone, no bias / kv_mask); got MASK_FLAGS={mask_flags}, bias_mode={bias_mode}, kv_mask_form={kv_mask_form}, "
            f"TILE_K={tile_k}, dtype_qkv={params.dtype_qkv}"
        )
    # The dead-sub-tile quiet body (TemplateParams.dead_subtile_quiet, appended; read through getattr so a record without the field
    # renders the full body): the one-batch-entry item of the f16/bf16 small tiles without a bias -- the bias arms also wait and
    # release the bias barriers per item, the pair item's dead sub-tile is the odd chunk tail, the d128 flavor keeps its rendering.
    dead_subtile_quiet = int(getattr(params, "dead_subtile_quiet", 0))
    if dead_subtile_quiet not in DEAD_SUBTILE_QUIET_FORMS:
        raise ValueError(f"{flavor}: dead_subtile_quiet must be 0 (the full body on a dead sub-tile) or 1 (the quiet body); got {dead_subtile_quiet}")
    if dead_subtile_quiet == 1 and (
        bias_mode or pair or tile_k > 64 or mxfp8 or params.dtype_qkv <= _DTYPE_E5M2 or params.pack_gqa or params.thd_varlen or params.epilogue_gate
    ):
        raise ValueError(
            f"{flavor}: dead_subtile_quiet=1 (the dead-sub-tile quiet body) is wired on the f16/bf16 32- / 64-wide tiles' one-batch-entry item "
            f"without a bias, PackGQA, THD or the gated epilogue only; got bias_mode={bias_mode}, n_chunk={n_chunk}, TILE_K={tile_k}, "
            f"dtype_qkv={params.dtype_qkv}, pack_gqa={params.pack_gqa}, thd_varlen={params.thd_varlen}, epilogue_gate={params.epilogue_gate}"
        )
    cls = {192: CfgD192, 128: CfgD128, 64: CfgD64, 32: CfgD32}[tile_k]
    if tile_k <= 64:
        # Per-tensor FP8 can issue one Rubin K=64 MMA at the native D64 width.
        # Block-scaled FP8 and D32 retain their existing larger envelope.
        if params.dtype_qkv <= _DTYPE_E5M2 and (mxfp8 or tile_k != 64):
            raise ValueError(f"{flavor}: quantized small head-dim tiles require per-tensor FP8 D64")
        if cta_mma not in SM107_SMALL_D_CGAS[(tile_k, tile_o)]:
            raise ValueError(
                f"{flavor}: cta_mma={cta_mma} is outside the validated widths {sorted(SM107_SMALL_D_CGAS[(tile_k, tile_o)])} for TILE_O={tile_o}; "
                f"at cga2 the per-CTA V operand is {tile_o // 2} columns = a {tile_o // 2 * b}-byte MN-major swizzle atom"
            )

    cfg = cls(
        TILE_M=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_o,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=cta_mma,
        CGA_N=1,
        CTA_MMA=cta_mma,
        SPLIT_PIPELINE=0,
        QO_ALIAS=0,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_o, cta_mma, b),
        O_SWZ_BYTES=o_swz_bytes(tile_o, b_o, o_div),
        O_BLOCK_SCALE=_O_BLOCK_SCALE_BY_DTYPE.get(dtype_o, 0),
        O_PACK_DIV=o_div,
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=softmax_wgs,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        SOFTMAX_WARPGROUPS=softmax_wgs,
        CORRECTION_WARPS=_D128_CORRECTION_WARPS,
        SOFTMAX_REGS=softmax_regs,
        CORRECTION_REGS=correction_regs,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        PADDED_TAIL_FORM=padded_tail_form,
        DEAD_SUBTILE_QUIET=dead_subtile_quiet,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        STATS_LOG2=int(params.stats_log2),
        PACK_GQA=int(params.pack_gqa),
        PACK_G=_pack_g(params, tile_m, partial=params.dtype_qkv > _DTYPE_E5M2),
        QH_PER_KH=params.qh_per_kh,
        SPLIT_KV=params.split_kv or 1,
        SEQ_KV_LENS_PRESENT=1 if params.thd_varlen else int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        PAGED_KV=int(bool(params.paged_kv)),
        PAGE_SIZE=int(params.page_size) if params.paged_kv else 0,
        SCHEDULER_POLICY=params.sched_policy if params.sched_policy is not None else SCHED_NATURAL,
        N_BMM2_CHUNKS=tile_n // 64,
        BMM2_CHUNK_SIZE=64,
        TOTAL_WARPS=total_warps,
        THREADS_PER_CTA=total_warps * 32,
        SOFTMAX_WG_WARPS=_D128_SOFTMAX_WG_WARPS,
        OTHER_WARPS=_D128_OTHER_WARPS,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=_D128_SOFTMAX_WG_WARPS,
        CORR_WARP_BASE=corr_warp_base,
        MMA_WARP_ID=mma_warp_id,
        TMALDG_WARP_ID=mma_warp_id + 1,
        TMASTG_WARP_ID=mma_warp_id + 2,
        SCHED_WARP_ID=mma_warp_id + 3,
        READ_TILE_ARRIVERS=_d128_read_tile_arrivers(softmax_wgs),
        BIAS_MODE=bias_mode,
        BIAS_RESIDENT_TILES=bias_tiles,
        BIAS_BLOCK_ROWS=sm107_bias_block_rows(softmax_wgs, tile_m, pair=pair) if bias_mode else 0,
        BIAS_SWZ_BYTES=BIAS_SWZ_BYTES,
        BIAS_BOXES_PER_TILE=(tile_n * b) // BIAS_SWZ_BYTES if bias_mode else 0,
        BIAS_EMPTY_ARRIVERS=softmax_wgs * _D128_SOFTMAX_WG_WARPS if (bias_mode or kv_mask_form) else 0,
        KV_MASK_FORM=kv_mask_form,
        N_CHUNK=n_chunk,
        STAGES_BIAS=stages_bias,
        BIAS_MAX_FORM=bias_max_form,
        S_SLOTS=s_slots,
        Q_IN_TMEM=q_in_tmem,
        # The multicast ring's cluster extent along the chunk axis (its fan-out); 1 everywhere else -- _validate_bias pins the coupling.
        BIAS_MCAST_CTAS=BIAS_MCAST_CTAS if bias_mode == BIAS_MODE_STREAMED_MULTICAST else 1,
    )

    _validate_regs(cfg, flavor)
    _validate_warp_layout(cfg, flavor, expect_total=_d128_total_warps(softmax_wgs), expect_arrivers=_d128_read_tile_arrivers(softmax_wgs))
    _validate_softmax_wgs(cfg, flavor)
    _validate_dtype_k_step(cfg, flavor)
    _validate_swizzles(cfg, flavor)
    _validate_head_dim_tile(cfg, flavor)
    _validate_tile_n(cfg, flavor)
    _validate_q_in_tmem(cfg, flavor)
    _validate_masks(cfg, flavor)
    _validate_bias(cfg, flavor)
    _validate_smem(cfg, flavor, qo_alias=bool(cfg.QO_ALIAS), bias_bytes=bias_resident_block_bytes(cfg))
    return cfg, _tma_iters(cfg)


def make_cfg_d32(params: TemplateParams) -> Tuple[CfgD32, TmaIters]:
    """The 32-wide head-dim tile of ``sm107/prefill_d128_f16.py`` (``TemplateParams.d_flavor=32``): every f16/bf16 head dim
    <= 32 with D % 8 == 0 (8 / 16 / 24 / 32) by TMA zero fill; cga1 (SM107_SMALL_D_CGAS); dense / causal / bottom-right /
    SWA / padded / sink / GQA like d128.  THD, split-KV, paged KV, PackGQA and the score side outputs are NOT wired on the
    tile: the row routes a graph asking for them onto the d128 flavor (``engines.Capabilities.dense_only_d_shapes``), and this
    factory refuses the THD record for the flavor."""
    return _make_cfg_d128_family(params, flavor="sm107 d32", tile_k=32, tile_o=32, mxfp8=False)


def make_cfg_d64(params: TemplateParams) -> Tuple[CfgD64, TmaIters]:
    """The 64-wide head-dim tile of ``sm107/prefill_d128_f16.py`` (``TemplateParams.d_flavor=64``): every f16/bf16 head dim in
    (32, 64] with D % 8 == 0 (40 / 48 / 56 / 64) by TMA zero fill; the same feature set and routing as ``make_cfg_d32`` (cga1
    and cga2). Per-tensor FP8 uses the same native width in ``prefill_d128_fp8.py``,
    including dense PackGQA and split-KV; its THD and paged forms remain unsupported."""
    if params.dtype_qkv <= _DTYPE_E5M2 and params.thd_varlen:
        raise ValueError("sm107 d64: per-tensor FP8 THD is not wired at the native D64 width")
    return _make_cfg_d128_family(params, flavor="sm107 d64", tile_k=64, tile_o=64, mxfp8=False)


def make_cfg_d128(params: TemplateParams) -> Tuple[CfgD128, TmaIters]:
    return _make_cfg_d128_family(params, flavor="sm107 d128", tile_k=128, tile_o=128, mxfp8=False)


def make_cfg_d128_mxfp8(params: TemplateParams) -> Tuple[CfgD128, TmaIters]:
    return _make_cfg_d128_family(params, flavor="sm107 d128 mxfp8", tile_k=128, tile_o=128, mxfp8=True)


def make_cfg_d192(params: TemplateParams) -> Tuple[CfgD192, TmaIters]:
    """d_qk=192 / d_v=128, f16 and per-tensor FP8.

    Same family as ``make_cfg_d128`` with a wider K: the pre-upstream base kernel
    was flavor-generic (one body served d128 and d192xd128 by swapping the config),
    and ``CfgD192`` only widens ``TILE_K``.
    """
    return _make_cfg_d128_family(params, flavor="sm107 d192xd128", tile_k=192, tile_o=128, mxfp8=False)


def make_cfg_d192_mxfp8(params: TemplateParams) -> Tuple[CfgD192, TmaIters]:
    """d_qk=192 / d_v=128, block-scale MXFP8.

    CGA2 ONLY, and that is a DESCRIPTOR constraint rather than a tuning choice.
    At ``cta_mma=1`` the K/V rings are not halved, so the four scale-factor tiles
    -- allocated last -- start at 256/258/276/278 KiB, i.e. past the **256 KiB
    version-0 tcgen05 descriptor window**.  A version-0 SF descriptor there wraps
    to offset 0 and the UTCCP copies Q DATA bytes into the SF TMEM columns:
    ``LSE = +inf`` and ``O = NaN`` on 100 % of cells, at every shape (the exact
    d512 MXFP8 failure in rules/mma-tma-matrix.md S6).  At ``cta_mma=2`` the
    highest slab sits at 200 KiB and version 0 is provably safe.

    So the Rubin MXFP8 engine row deliberately declares NO ``cgas_by_d_shape``
    entry for (192, 128), leaving it on the row default ``cgas={2}``.  Lifting
    that needs ``DESC_VERSION`` derived from the layout AND the version-1 SF path
    validated on Rubin -- which is NOT a free widening: setting
    ``desc_version=1`` on the d128/d256 MXFP8 tiles turned 21 green tests red
    (2026-09-08), so the bit is not a transparent superset.
    """
    return _make_cfg_d128_family(params, flavor="sm107 d192xd128 mxfp8", tile_k=192, tile_o=128, mxfp8=True)


# ---------------------------------------------------------------------------
# d256 -- Q-union-O alias, ONE softmax warpgroup (12 warps)
# ---------------------------------------------------------------------------
#
# Warp map: 4 softmax + 4 correction + MMA + TMALDG + TMASTG + scheduler.
# The arriver count is per-CTA-group: ((SOFTMAX_WG*4) + CORR + 2) * CGA_SIZE
# + (CGA_M / CTA_MMA).

_D256_TOTAL_WARPS = 12


def _d256_read_tile_arrivers(cta_mma: int) -> int:
    return ((1 * 4) + 4 + 2) * (cta_mma * 1) + (cta_mma // cta_mma)


def _make_cfg_d256_family(params: TemplateParams, *, flavor: str, mxfp8: bool):
    """The cc 10.7 d256 config (per-tensor FP8 or MXFP8): tile geometry, ring depths, warp layout and the SMEM / descriptor-window
    checks for the one-CTA and the 2-CTA-pair builds (``params.cta_mma``); raises with the failing tally on an impossible layout."""
    split_wired = not mxfp8 and params.dtype_qkv in (_DTYPE_E4M3, _DTYPE_E5M2)
    _validate_params(flavor, params, split_wired=split_wired)
    if (params.split_kv or 1) > 1 and params.thd_varlen:
        raise ValueError(f"{flavor}: per-tensor FP8 split-KV is dense-only")
    if params.pack_gqa and not mxfp8:
        if params.thd_varlen:
            raise ValueError(f"{flavor}: per-tensor FP8 PackGQA is dense-only")
        if 128 % params.qh_per_kh:
            raise ValueError(f"{flavor}: PackGQA requires a whole head group dividing TILE_M=128; got {params.qh_per_kh}")
    cta_mma = params.cta_mma
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    tile_k = tile_o = 256
    tile_n = 128
    # STAGES_KV is a TUNING KNOB again (2..4), and the note that said otherwise
    # was WRONG about why.
    #
    # It used to be pinned to 2, blaming a body that "conflates the KV ring
    # index with the 2-slot S_acc parity". That diagnosis does not survive
    # reading the body: every parity site derives parity from the ABSOLUTE
    # `kv_loop & 1`, and K/V ride separate PipelineStates advanced in lockstep
    # with the TMA warp, so the body is depth-agnostic.
    #
    # The real cause was the >256 KiB tcgen05 descriptor wrap. Buffers are laid
    # out sQO | sK[stages] | sV[stages]; at depth 4 the last two V stages start
    # at 256 KiB and 288 KiB, and a VERSION-0 descriptor truncates
    # `start_address` to 14 bits, so those stages alias the bottom of SMEM. The
    # answer goes wrong from the KV iteration that first touches a wrapped
    # stage -- which is why it looked like a ring bug and why it only shows up
    # at S_kv > 256.
    #
    # Measured on Rubin, d256 f16, cos vs an fp32 reference, n_kv = 2/3/4/8:
    #   depth 4, desc v0:  1.0000 / 0.6806 / 0.4980 / 0.4640   <- the old data
    #   depth 4, desc v1:  1.0000 / 1.0000 / 1.0000 / 1.0000   dense AND causal
    #   depth 3, desc v0:  1.0000 / 1.0000 / 1.0000 / 1.0000   (stays under)
    # `prefill_d256_f16.DESC_VERSION` is now DERIVED from the layout, so any
    # depth that fits SMEM is correct by construction. Do not re-literal it.
    # (The per-tensor FP8 sibling `prefill_d256_fp8.py` carried a literal 0
    # until the epilogue-gate PR ported the same `_needs_desc_v1` derivation;
    # at e4m3-O depth 4 its last V stage starts at exactly 262144, so the
    # literal was one depth away from the same wrap.)
    # `is not None`, NOT a truth test: a truthiness check maps an explicit
    # stages_kv=0 onto the default 2, which is a knob SUBSTITUTION -- the one
    # thing the engine contract forbids (honored or ineligible). Out-of-domain
    # values must reach the 2..4 check below and raise there.
    stages_kv = params.stages_kv if getattr(params, "stages_kv", None) is not None else 2
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)
    arrivers = _d256_read_tile_arrivers(cta_mma)
    # The four single-warp roles (MMA / TMA-LDG / TMA-STG / scheduler; the HW
    # equality constraint keeps them one number) run at 40 on every d256
    # module.  MEASURED DEAD END (2026-09-15, gated MXFP8): its sm_107a SASS
    # carries 25 STL/LDL against the ungated 6 (the SR_CTAID.{x,y,z} tile
    # coordinates, spilled at kernel entry and refilled in the TMA-LDG /
    # TMA-STG / scheduler regions), and raising these four roles to 48 left the
    # count at 25.  The refill sites carry MOV.SPILL UR -> R (5 vs 0 ungated):
    # the pressure starts in the UNIFORM register file, which setmaxnreg does
    # not govern, so this vector budget is not the lever.  The split itself IS
    # applied (USETMAXREG.TRY_ALLOC / DEALLOC present in the sm_107a SASS).  Do
    # not retry the bump; see the REGISTER NOTE in
    # sm107/prefill_d256_mxfp8.py's TMA-LDG warp for the levers not yet tried.
    other_regs = 40

    cfg = CfgD256(
        TILE_M=128,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_o,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=cta_mma,
        CGA_N=1,
        CTA_MMA=cta_mma,
        SPLIT_PIPELINE=0,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_o, cta_mma, b),
        O_SWZ_BYTES=o_swz_bytes(tile_o, b_o),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=1,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        # The body is a SINGLE-warpgroup pipeline. Two warpgroups is the
        # Blackwell d256 split-P design and does not exist here.
        SOFTMAX_WARPGROUPS=1,
        CORRECTION_WARPS=4,
        SOFTMAX_REGS=240,
        CORRECTION_REGS=96,
        MMA_REGS=other_regs,
        TMALDG_REGS=other_regs,
        TMASTG_REGS=other_regs,
        SCHEDULER_REGS=other_regs,
        OTHER_REGS=other_regs,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        STATS_LOG2=int(params.stats_log2),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=params.qh_per_kh,
        SPLIT_KV=params.split_kv or 1,
        SEQ_KV_LENS_PRESENT=1 if params.thd_varlen else int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        PAGED_KV=int(bool(params.paged_kv)),
        PAGE_SIZE=int(params.page_size) if params.paged_kv else 0,
        SCHEDULER_POLICY=params.sched_policy if params.sched_policy is not None else SCHED_NATURAL,
        N_BMM2_CHUNKS=tile_n // 64,
        BMM2_CHUNK_SIZE=64,
        TOTAL_WARPS=_D256_TOTAL_WARPS,
        THREADS_PER_CTA=_D256_TOTAL_WARPS * 32,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=0,  # unused at SOFTMAX_WARPGROUPS=1
        CORR_WARP_BASE=4,
        MMA_WARP_ID=8,
        TMALDG_WARP_ID=9,
        TMASTG_WARP_ID=10,
        SCHED_WARP_ID=11,
        READ_TILE_ARRIVERS=arrivers,
        # Fused epilogue gate (validated per flavor in _validate_params). The gate
        # is 2 B/elem on BOTH the half kernel (Q dtype) and the FP8 kernel (bf16).
        EPILOGUE_GATE=int(params.epilogue_gate),
        GATE_BPE=2,
    )

    _validate_regs(cfg, flavor)
    _validate_warp_layout(cfg, flavor, expect_total=_D256_TOTAL_WARPS, expect_arrivers=arrivers)
    _validate_dtype_k_step(cfg, flavor)
    _validate_swizzles(cfg, flavor)
    _validate_masks(cfg, flavor)
    # The STAGES_KV upper bound differs per dtype family, and the message says
    # which: the f16 and per-tensor FP8 bodies DERIVE their tcgen05 descriptor
    # version from the layout (`_needs_desc_v1`), so for them every depth that
    # fits SMEM is correct; the MXFP8 body PINS DESC_VERSION=0 (version 1
    # regressed 21 green d128/d256 MXFP8 tests, rules/mma-tma-matrix.md S6), so
    # for it the bound is the 256 KiB version-0 window on the LAST scale-factor
    # slab -- checked below with its own diagnostic.
    _depth_bound = (
        f"the {SMEM_USABLE_BYTES // 1024} KiB Rubin carveout, and the kernel derives its tcgen05 descriptor version from the resulting "
        f"layout so every depth in range is correct"
        if not mxfp8
        else f"the {SMEM_USABLE_BYTES // 1024} KiB Rubin carveout AND the {TCGEN05_V0_ADDR_LIMIT // 1024} KiB version-0 tcgen05 descriptor "
        f"window on the last scale-factor slab (this kernel pins DESC_VERSION=0; see d256_mxfp8_last_sf_tile_start)"
    )
    _check(
        [
            (cfg.TILES_Q == 1, f"{flavor}: the d256 pipeline mandates TILES_Q == 1"),
            (cfg.SOFTMAX_WARPGROUPS == 1, f"{flavor}: the d256 pipeline mandates SOFTMAX_WARPGROUPS == 1"),
            (
                2 <= cfg.STAGES_KV <= 4,
                f"{flavor}: STAGES_KV must be in 2..4 (got {cfg.STAGES_KV}); the upper bound is {_depth_bound}",
            ),
        ]
    )
    sf_bytes = 0
    if mxfp8:
        # MXFP8: the four scale-factor slabs are declared after sQO | sK | sV
        # and read through VERSION-0 UTCCP descriptors, so their START must sit
        # under the 14-bit window.  A bf16 O at STAGES_KV=3 (or an e4m3 O at 4)
        # puts sQ_SF at or past 262144: the descriptor wraps to offset 0 and the
        # UTCCP copies Q DATA bytes into the SF TMEM columns -- LSE = +inf and
        # O = NaN on 100 % of cells at every shape, the SF values provably
        # irrelevant (the d512 MXFP8 failure).  Latent until 2026-09-15 (depth 3
        # fit the SMEM check); raised HERE, before the kernel's own backstop.
        # The epilogue gate tile is declared AFTER the SF slabs (PR-B D7) and is
        # not in this sum.
        last_sf = d256_mxfp8_last_sf_tile_start(cfg)
        _check(
            [
                # The gated MXFP8 module runs at both widths: the dense unsplit legs take the 2-CTA pair since
                # d256_mxfp8_pair_leg (K/V rings halved, SF slabs full-size, one gate tile per CTA -- the tallies
                # below bound that layout too), validated by the MXFP8 gate tests at the pair's default width.
                (
                    last_sf < TCGEN05_V0_ADDR_LIMIT,
                    f"{flavor}: the last scale-factor slab starts at {last_sf} B, at or past the {TCGEN05_V0_ADDR_LIMIT} B version-0 tcgen05 "
                    f"descriptor window (STAGES_KV={cfg.STAGES_KV}, BPE_O={cfg.BPE_O}, CTA_MMA={cfg.CTA_MMA}); a version-0 SF descriptor there "
                    f"wraps to offset 0 and the UTCCP copies Q data as scale factors (LSE = +inf, O = NaN on every cell).  Reduce STAGES_KV or "
                    f"write an e4m3 O -- the kernel pins DESC_VERSION=0 (version 1 is not a transparent widening, rules/mma-tma-matrix.md S6)",
                ),
            ]
        )
        sf_bytes = d256_mxfp8_sf_smem_bytes(cfg.TILE_M, cfg.TILE_N, cfg.TILE_K, cfg.TILE_O, cfg.STAGES_KV)
    # d256 always Q-union-O aliases.  With the epilogue gate on, the 64 KiB
    # staging tile is added here: depth 2 fits at both CTA_MMA widths (258 KiB
    # f16 cga2 / 226 KiB fp8 e4m3-O / 258 KiB fp8 bf16-O; MXFP8 adds its 6 KiB of
    # scale-factor slabs: 232 KiB e4m3-O / 264 KiB bf16-O), depth 3 does not
    # (322 KiB) -- the raise names every term so the reader sees why a depth
    # that fits ungated no longer does.
    _validate_smem(cfg, flavor, qo_alias=True, sf_bytes=sf_bytes)
    return cfg, _tma_iters(cfg)


def make_cfg_d256(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    return _make_cfg_d256_family(params, flavor="sm107 d256", mxfp8=False)


def make_cfg_d256_mxfp8(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    return _make_cfg_d256_family(params, flavor="sm107 d256 mxfp8", mxfp8=True)


# ---------------------------------------------------------------------------
# d512 -- cga4x1 role split (8 warps)
# ---------------------------------------------------------------------------
#
# Warp map: 4 compute (softmax on sg0 / correction on sg1) + MMA + TMALDG +
# TMASTG + scheduler.  READ_TILE_ARRIVERS counts across the whole cluster:
#   compute warps (4/CTA, both sgs)          = 4 * CGA_M * CGA_N          = 16
#   TMALDG        (1/CTA, both sgs)          = 1 * CGA_M * CGA_N          =  4
#   sg0 MMA leader (lead CTA only)           = (CGA_M*CGA_N)/(2*CTA_MMA)  =  1
#   sg1 MMA all   (P12 forwarder + leader)   = (CGA_M*CGA_N)/2            =  2
#   sg1 TMASTG    (sg1 CTAs only)            = CTA_MMA * CGA_N            =  2
#                                                                   total = 25

_D512_TOTAL_WARPS = 8
_D512_CGA_M, _D512_CGA_N, _D512_CTA_MMA = 4, 1, 2
_D512_READ_TILE_ARRIVERS = (
    ((1 * 4) + 1) * (_D512_CGA_M * _D512_CGA_N)
    + (_D512_CGA_M * _D512_CGA_N) // (2 * _D512_CTA_MMA)
    + (_D512_CGA_M * _D512_CGA_N) // 2
    + _D512_CTA_MMA * _D512_CGA_N
)


def _make_cfg_d512_family(params: TemplateParams, *, flavor: str, mxfp8: bool):
    _validate_params(flavor, params)
    if params.cta_mma != _D512_CTA_MMA:
        raise ValueError(f"{flavor}: the role-split pipeline is cga4x1 / CTA_MMA=2 only (got cta_mma={params.cta_mma})")
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    tile_k = tile_o = 512
    tile_n = 128
    is_fp8 = params.dtype_qkv <= _DTYPE_E5M2
    # FP8 (BPE=1) fits a deeper K/V ring and a 4-deep cross-sg P ring; f16/bf16
    # (BPE=2) are SMEM-cap driven down to 2/2.
    stages_kv = 3 if is_fp8 else 2
    xfer_stages = 4 if is_fp8 else 2
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)

    cfg = CfgD512(
        TILE_M=128,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_o,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=_D512_CGA_M,
        CGA_N=_D512_CGA_N,
        CTA_MMA=_D512_CTA_MMA,
        SPLIT_PIPELINE=1,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_o, _D512_CTA_MMA, b),
        O_SWZ_BYTES=o_swz_bytes(tile_o, b_o),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=1,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        XFER_STAGES=xfer_stages,
        # One 4-warp compute group; correction is fused into it, sub-group
        # conditional, so there are no separate correction warps or registers.
        SOFTMAX_WARPGROUPS=1,
        CORRECTION_WARPS=0,
        SOFTMAX_REGS=240,
        CORRECTION_REGS=0,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        STATS_LOG2=int(params.stats_log2),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=params.qh_per_kh,
        SPLIT_KV=params.split_kv or 1,
        SEQ_KV_LENS_PRESENT=1 if params.thd_varlen else int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SCHEDULER_POLICY=params.sched_policy if params.sched_policy is not None else SCHED_NATURAL,
        BMM2_N_PER_CALL=256,
        BMM2_LOOP_N_BLOCKS=tile_o // 256,
        N_BMM2_CHUNKS=2,
        BMM2_CHUNK_SIZE=256,
        TOTAL_WARPS=_D512_TOTAL_WARPS,
        THREADS_PER_CTA=_D512_TOTAL_WARPS * 32,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=0,  # unused under the role split
        CORR_WARP_BASE=0,  # correction fused with the compute group
        MMA_WARP_ID=4,
        TMALDG_WARP_ID=5,
        TMASTG_WARP_ID=6,
        SCHED_WARP_ID=7,
        READ_TILE_ARRIVERS=_D512_READ_TILE_ARRIVERS,
    )

    _validate_regs(cfg, flavor)
    _validate_warp_layout(cfg, flavor, expect_total=_D512_TOTAL_WARPS, expect_arrivers=_D512_READ_TILE_ARRIVERS)
    _validate_dtype_k_step(cfg, flavor)
    _validate_swizzles(cfg, flavor)
    _validate_masks(cfg, flavor)
    _check([(cfg.SPLIT_PIPELINE == 1, f"{flavor}: the d512 flavor is role-split (SPLIT_PIPELINE=1)")])
    return cfg, _tma_iters(cfg)


def make_cfg_d512(params: TemplateParams):
    # The 2x2-datapath record (TemplateParams.mma_2x2, appended, default False) is read through getattr so a
    # record built before the field existed takes the role-split arm unchanged (the config_sm100 dispatch, mirrored).
    if getattr(params, "mma_2x2", False):
        return make_cfg_d512_2x2(params)
    return _make_cfg_d512_family(params, flavor="sm107 d512", mxfp8=False)


def make_cfg_d512_mxfp8(params: TemplateParams) -> Tuple[CfgD512, TmaIters]:
    return _make_cfg_d512_family(params, flavor="sm107 d512 mxfp8", mxfp8=True)


# ---------------------------------------------------------------------------
# d512 flavor on the 2x2 DATAPATH -- d_qk = d_v = 512, Rubin, sm107/prefill_d512_f16_2x2.py
#
# The SM100 CfgD512X2 record (config_sm100: one pipeline per CTA on the
# cta_group::2 M=128 atom, 64 Q rows per CTA, 12 warps, a (CGA_M, 1, 1)
# cluster of twin pairs sharing K/V by TMA multicast) with the Rubin deltas of
# the design (section 13):
#   * 3-deep K and V sub-chunk rings (STAGES_K_SUB = STAGES_V_SUB = 3: one
#     extra KV iteration of TMA allowance; the O staging keeps aliasing the V
#     ring).  K 4 / V 3 would put 320 KiB of data over the usable line.
#   * DESC_VERSION derived from the LAYOUT: sQ 64 KiB | sK 3 x 32 | sV 3 x 32 u
#     sO puts the P ring at exactly 262144 B = TCGEN05_V0_ADDR_LIMIT, so every
#     SmemTile of the kernel carries desc_version=1 (the 2026-09-04 silent-zero
#     class; test_sm107_descriptor_version_matches_the_smem_budget).
#   * The budget is checked against SMEM_USABLE_BYTES (320 KiB), never the
#     327 KiB capacity (the d192 f16 50 %-zeros lesson above).
#   * TMEM: 512 columns (388 used); the 576-col is_exclusive allocation is NOT
#     needed until a 3rd / 4th S parity is added.
#   * TILE_K_HW from tile_k_hw(dtype) = 16 for f16/bf16 (the K=64 2-chunk form
#     is the FP8 QMMA path).
# ---------------------------------------------------------------------------

_D512_2X2_STAGES_K_SUB = 3
_D512_2X2_STAGES_V_SUB = 3


def _d512_2x2_desc_version_for(cfg: CfgD512X2) -> int:
    """The tcgen05 SMEM-descriptor version the 2x2 layout NEEDS: 1 as soon as the last MMA-operand tile (the P ring,
    the highest slab) reaches the 256 KiB window of a version-0 descriptor, else 0."""
    p_end = d512_2x2_p_ring_start_bytes(cfg) + cfg.XFER_STAGES * cfg.TILE_M * cfg.TILE_N * cfg.BPE
    return 1 if p_end > TCGEN05_V0_ADDR_LIMIT else 0


def _validate_cfg_d512_2x2_sm107(cfg: CfgD512X2, flavor: str) -> None:
    """The arch-neutral 2x2 geometry checks plus the Rubin arch facts."""
    smem = d512_2x2_smem_bytes(cfg)
    want_desc = _d512_2x2_desc_version_for(cfg)
    _check(
        d512_2x2_geometry_checks(cfg)
        + (
            (
                cfg.STAGES_K_SUB == _D512_2X2_STAGES_K_SUB and cfg.STAGES_V_SUB == _D512_2X2_STAGES_V_SUB,
                f"{flavor}: the Rubin arm runs 3-deep K/V sub-chunk rings (got {cfg.STAGES_K_SUB}/{cfg.STAGES_V_SUB})",
            ),
            (
                cfg.DESC_VERSION == want_desc,
                f"{flavor}: DESC_VERSION={cfg.DESC_VERSION} but the layout puts the P ring at {d512_2x2_p_ring_start_bytes(cfg)} B "
                f"(version-0 window ends at {TCGEN05_V0_ADDR_LIMIT}); it needs version {want_desc}",
            ),
            (
                cfg.SMEM_CAP_BYTES == SMEM_USABLE_BYTES,
                f"{flavor}: the SMEM budget is checked against the {SMEM_USABLE_BYTES // 1024} KiB usable carveout, got {cfg.SMEM_CAP_BYTES}",
            ),
            (
                smem["total"] <= SMEM_USABLE_BYTES,
                f"{flavor}: SMEM {smem['total']} B (incl. {cfg.SMEM_ALIGN_PAD} B pad) exceeds the {SMEM_USABLE_BYTES // 1024} KiB usable Rubin "
                f"carveout: {smem}.  Overflowing it does NOT fail the launch -- it clobbers the last buffer allocated",
            ),
            (cfg.TMEM_COLS == 512, f"{flavor}: 512 TMEM columns (388 used); 576 + is_exclusive only with a 3rd/4th S parity"),
            (
                cfg.O_EMPTY_ARRIVERS == cfg.ONE_WARP * cfg.KV_SHARE,
                f"{flavor}: the O u V alias gate is PAIR-WIDE -- mb_o_empty init must be ONE_WARP x KV_SHARE = {cfg.ONE_WARP * cfg.KV_SHARE} "
                f"(got {cfg.O_EMPTY_ARRIVERS}); a twin's multicast V(t+1) lands in BOTH twins' sVO",
            ),
        )
    )
    _validate_dtype_k_step(cfg, flavor)


def make_cfg_d512_2x2(params: TemplateParams, *, cga_m: int = 4) -> Tuple[CfgD512X2, TmaIters]:
    """The Rubin 2x2-datapath d512 configuration (``TemplateParams.mma_2x2``).

    ``cga_m`` = 4 (two twin pairs sharing K/V by multicast, the shipped arm) or 2 (the bring-up / bitwise-twin arm:
    one pair, own-bit loads).  Not a knob: tests load the template with the arm they want through the module constant
    the kernel file reads (FROST_D512_2X2_CGA_M)."""
    flavor = "sm107 d512 2x2"
    if not getattr(params, "mma_2x2", False):
        raise ValueError(f"{flavor}: make_cfg_d512_2x2 needs a TemplateParams record with mma_2x2=True (make_cfg_d512 dispatches on it)")
    # The Rubin record guard keys its THD allowlist by FLAVOR NAME (_F16_THD_FLAVORS holds "sm107 d512"): the 2x2 sibling
    # carries the same ported setup-kernel call sites as the role-split d512 body, so it is validated under that name.
    _validate_params("sm107 d512", params, split_wired=True)
    # Share the d512/half/unpaged guards with SM100. This native twin additionally
    # addresses packed subgroups and supports the pre-folded scale.
    _validate_params_sm100("d512", params, scale_prefolded_wired=True, d512_partial_pack_wired=True)
    if params.dtype_qkv not in (_DTYPE_BF16, _DTYPE_FP16):
        raise ValueError(f"{flavor}: BF16/FP16 inputs only")
    if params.cta_mma != 2:
        raise ValueError(f"{flavor}: cta_group::2 pairs only (got cta_mma={params.cta_mma})")
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    n_pairs = cga_m // CfgD512X2.CTA_MMA
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)
    base = CfgD512X2(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=cga_m,
        KV_SHARE=n_pairs,
        Q_SUPERS_PER_CLUSTER=cga_m,
        ROWS_PER_CLUSTER=cga_m * CfgD512X2.TILE_M,
        READ_TILE_ARRIVERS=10 * cga_m + n_pairs,
        KV_EMPTY_ARRIVERS=n_pairs,
        # PAIR-WIDE O u V alias gate (fix-lane FATAL-1): every twin's TMA-STG warp arrives on both twins' mb_o_empty.
        O_EMPTY_ARRIVERS=CfgD512X2.ONE_WARP * n_pairs,
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        STAGES_K_SUB=_D512_2X2_STAGES_K_SUB,
        STAGES_V_SUB=_D512_2X2_STAGES_V_SUB,
        STAGES_KV=max(_D512_2X2_STAGES_K_SUB, _D512_2X2_STAGES_V_SUB),
        SMEM_CAP_BYTES=SMEM_USABLE_BYTES,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        STATS_LOG2=int(params.stats_log2),
        SCHEDULER_POLICY=params.sched_policy if params.sched_policy is not None else SCHED_NATURAL,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv or 1),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        PACK_G=_pack_g(params, CfgD512X2.TILE_M, partial=True),
    )
    # DESC_VERSION is DERIVED from the layout the fields above describe, never set by hand.
    cfg = dataclasses_replace(base, DESC_VERSION=_d512_2x2_desc_version_for(base))
    _validate_cfg_d512_2x2_sm107(cfg, flavor)
    return cfg, _tma_iters(cfg)


# ---------------------------------------------------------------------------
# d256 index-list SPARSE prefill -- swap-AB tile over gathered K/V (sm107/sparse_d256_f16.py)
# ---------------------------------------------------------------------------
#
# The kernel this config describes (its header carries the full tables): one CTA per SM, cga1, persistent; a work item is
# (query token, KV head) -- the G query heads of the token on the MMA N axis (N_Q = 16), 128-key tiles of 32 four-token blocks
# on the M axis, gathered by GATHER_WARPS warps with tma_gather4 from the token's block list.  Warp map of the ONE-column-group
# record (16 warps; the DEFAULT record adds a second 4-warp softmax group = 20 warps -- the kernel header's table is the authority),
# every warpgroup role-homogeneous because setmaxnreg is warpgroup-collective:
#
#     WG0  warps 0-3    softmax + epilogue (128 lanes = 128 key lanes of S^T / P^T = 128 d lanes of O^T)
#     WG1-2 warps 4-11  gather issuers (blocks 4w..4w+3 of every K and V tile; one arrive_expect_tx per operand stage)
#     WG3  warps 12-15  MMA / TMA-LDG / scheduler / spare, ONE register count
#
# Every arrival count below is derived from the warp counts, and every one is a mbarrier init the body cannot reach if the
# config disagrees (an unreachable init is a HANG at every shape, a stale ring slot a SILENT wrong answer):
#     KV_FULL_ARRIVERS   = GATHER_WARPS                       (one expect_tx per gather warp per stage; bytes sum to the stage)
#     IDS_EMPTY_ARRIVERS = GATHER_WARPS + SOFTMAX_WARPS        (one elected lane per warp that reads the staged list)
#     READ_TILE_ARRIVERS = SOFTMAX_WARPS + 1 (MMA) + 1 (TMA) + GATHER_WARPS   (the scheduler warp and the spare never credit)
#
# Registers: a 640-thread launch is capped at 65536 / 640 -> 96 per thread FLAT, a 512-thread one at 128 (the DSL emits .reqntid, no .maxnreg, so every
# setmaxnreg is dropped); the declared split for when .maxnreg lands must balance the ENTRY pool, SUM(role regs x warps) ==
# entry x warps, or the last INCREASE warp parks forever.  Host trace-compile of the swap-AB decode body under a forced
# 512-thread rendering (sm_107a): REG 104, STL 0, LDL 0.  The DEFAULT record is the two-column-group body (20 warps at 8 gather
# warps, 96 entry registers, 0 spills on the host renderings but one register pair in the gated 16-head-group prologue); the
# one-group body is 16 warps / 128; gather_warps=4 is the named fallback (12 warps / 168 at one group, 16 / 128 at two).

_SPARSE_FLAVOR = "sm107 d256 sparse"
# The standard per-CTA carveout (L1 keeps its 100 KiB); past it the launcher needs ALLOW_OVERSIZED_SHARED_MEMORY and the
# budget becomes SMEM_USABLE_BYTES (320 KiB, never the 327 KiB capacity).
SMEM_STANDARD_CARVEOUT_BYTES = 227 * 1024
_SPARSE_N_Q = 16  # one softmax column group (4 warps x 16 columns); 32 would be two groups
_SPARSE_BLOCK_SIZE = 4  # tokens per block = the four rows of ONE tma_gather4
_SPARSE_SOFTMAX_WARPS = 4  # at ONE column group (the default body)
_SPARSE_SOFTMAX_WARPS_PER_GROUP = 4  # 128 lanes = the 128 key rows of a tile, per column group
_SPARSE_SOFTMAX_GROUPS = (1, 2)  # the softmax SPLIT (TemplateParams.qsa_softmax_groups): 4 x 16 columns, or 2 x (4 x 8)
_SPARSE_AUX_WARPS = 4  # MMA, TMA-LDG, scheduler, spare: one complete warpgroup at one register count
_SPARSE_TOPK_MIN, _SPARSE_TOPK_MAX = 4, 512
_SPARSE_IDS_SLOT_PAD_BYTES = 64  # 16 reserved words per staged list (unread in v1; keeps the slot a 16-B multiple at every top_k)
_SPARSE_SMEM_MISC_BYTES = 512  # tmem_ptr 16 B + 15 mbarrier arrays (29 stages, 16-B padded) 272 B + the 2-slot payload ring 64 B = 352 B (+ the two 1-stage gate arrays, 32 B, on the EPILOGUE_GATE rendering), reserved as 512
# The S^T / P^T ring depth: the S^T TMEM slots (S_ACC_OFF), the sP buffers and the s_full / s_empty / p_full / bmm2_done rings are all
# STAGES_S deep, and the column-max exchange scratch holds one slot per ring parity + the epilogue's lane sum.  The kernel reads the
# depth ONCE (`STAGES_S = CFG.STAGES_S`) and spells it at every ring site -- never a literal 2 (frost-kernels.md section 6: a depth
# changed at some sites and not others is a stale S^T slot read from the first reuse, a silent wrong answer).
_SPARSE_STAGES_S = 2
_SPARSE_RED_SLOTS = _SPARSE_STAGES_S + 1  # the column-max exchange scratch: one slot per S ring parity + the epilogue's lane sum
_GATHER4_ROWS = 4  # rows per gather4 issue
_GATHER4_BOX_BYTES = 128  # one SW128 span per row per issue
_REG_FILE_PER_CTA = 65536
# Arms of the kernel whose BODY carries them.  A record that asks for an arm outside this set is declined at config build (the
# same backstop every other Rubin flavor applies to THD / the gate / split_kv); each arm's landing commit adds its name here and
# flips the adapter's claims record in the same commit.  Names: "epilogue_gate", "thd_varlen", "paged_kv", "split_kv",
# "list_per_sequence", "bottom_right", "pure_list" (qsa_include_open_block=False), "seq_q_lens".
# "paged_kv": the gather warps read K / V from page pools through a (B, max_pages) block table -- one table lookup per
# 4-token block (PAGE_SIZE % 4 == 0, so a block never straddles a page); the visible range comes from the per-batch KV
# lengths, which the paged form therefore REQUIRES (SEQ_KV_LENS_PRESENT is forced to 1 below).
# "thd_varlen": packed sequences through the persistent claim-counter scheduler, the shared THD metadata layout (seq_kv_lens |
# cu_q | cu_k | remap | live | ctr) in a caller workspace, sequence-relative block ids (the gather row carries cu_k[b]).
# "epilogue_gate": the item's gate tile (O's shape, the token's G heads) aliased into its freed Q^T slot after the item's last
# BMM1 (GATE_TX_BYTES = the box), O gated in place after the dead-row SELECT, the LSE untouched; it touches the Q^T slot ring
# and the epilogue only, so it composes with either of the two arms above (the gate is O-shaped on every arm).
# The THD and paged arms are wired ONE AT A TIME: a record asking for both is refused by the validator below (the packed
# sequence's K / V row offset cu_k[b] composes with a dense [1, T_kv, H_kv, D] tensor only, never with a page pool) until a
# serving stack asks for the combination -- it then lands with its own accept cells.
# "bottom_right": the row's key-space position is tok + (eff_seqlen_kv_b - eff_seqlen_q_b) (the dense tiles' bottom-right
# diagonal); the Q / O / LSE coordinate stays the row index; pos < 0 is a dead row.  Dense BSHD only (not under THD).
# "list_per_sequence": the decode / MTP form -- block_ids [B, BLOCK_TOPK] and block_lens [B], ONE list per sequence read by every
# item of the sequence (every KV head, split and token), the count and the appended tail anchored at the STEP-0 position
# pos_0 = eff_seqlen_kv_b - eff_seqlen_q_b (0, 1 or 2 tail ids at S_q <= 4: every token from the step-0 tail start to the row's
# own position).  Requires BOTTOM_RIGHT (a top-left per-sequence list has no serving meaning); the S_q <= 4 cap is the adapter's
# shape decline.  Not under THD.
# "split_kv": SPLIT_KV > 1 cuts every item's tiles into SPLIT_KV contiguous chunks, each its own work item (grid x = tok + S_q x
# split) writing fp32 O / natural-log LSE partials into the split-major (b + s x B) workspace sm100/split_combine.py reduces; an
# empty chunk is a dead item (one clamped tile, O = 0 / LSE = -inf = the combine's identity).  SPLIT_KV <= MAX_TILES_PER_ITEM (a
# larger split has an empty chunk on EVERY item); never with the kernel's own EPILOGUE_GATE (the gate rides the combine) and never
# under THD (the persistent claim-counter form has no split axis).
SPARSE_D256_WIRED_ARMS: frozenset = frozenset({"paged_kv", "thd_varlen", "epilogue_gate", "split_kv", "list_per_sequence", "bottom_right"})


def sparse_entry_regs(total_warps: int) -> int:
    """ptxas's per-thread ENTRY register count for a ``total_warps``-warp launch: the register file over the launch, rounded
    DOWN to the 8-register allocation granule (128 at 16 warps, 168 at 12, 255 at <= 8).  ``entry x total_warps`` is the pool a
    ``setmaxnreg`` split redistributes -- never 65536 / 32."""
    return min(255, (_REG_FILE_PER_CTA // (32 * total_warps)) // 8 * 8)


def _align16(nbytes: int) -> int:
    return -(-nbytes // 16) * 16


@dataclass(frozen=True)
class CfgD256Sparse:
    """d_qk = d_v = 256 index-list sparse prefill: the swap-AB tile (keys on the MMA M axis, the token's query heads on N) over
    K/V tiles gathered block by block.  Every field is set by ``make_cfg_d256_sparse``; the defaults are the ONE-column-group
    16-warp record at the 24/2 geometry and top_k = 512 (the factory's default renders the two-group 20-warp record) so the
    dataclass is constructible in a doc / test context, not a fallback."""

    # --- tile geometry: TILE_N keys per tile (the MMA M axis = the 128-lane TMEM layout), TILE_K / TILE_O the head dims,
    # N_Q the MMA N axis (the token's query heads, zero-padded to 16)
    TILE_N: int = 128
    TILE_K: int = 256
    TILE_O: int = 256
    N_Q: int = _SPARSE_N_Q
    BLOCK_SIZE: int = _SPARSE_BLOCK_SIZE
    BLOCKS_PER_TILE: int = 32
    BLOCK_TOPK: int = 512  # TemplateParams.qsa_block_topk: the list width; every ids byte count is BLOCK_TOPK x 4
    MAX_TILES_PER_ITEM: int = 17  # ceil((BLOCK_TOPK + 1 open block) / BLOCKS_PER_TILE)

    # --- dtypes (half only; the output inherits)
    DTYPE_QKV: int = _DTYPE_BF16
    DTYPE_O: int = _DTYPE_BF16
    BPE: int = 2
    BPE_O: int = 2
    GATE_BPE: int = 2

    # --- swizzles / MMA k-step
    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    P_SWZ_BYTES: int = 32  # P^T rows are N_Q half values = 32 B: the 32-B swizzle atom
    TILE_K_HW: int = 16

    # --- rings
    STAGES_KV: int = 3  # K(t), V(t), K(t+1) resident: 3 x 64 KiB
    STAGES_S: int = _SPARSE_STAGES_S  # S^T slots in TMEM (len(S_ACC_OFF)) = sP buffers = the s_full / s_empty / p_full / bmm2_done ring depth
    STAGES_Q: int = 2  # Q^T of item i and i+1 (the gate of item i aliases its freed slot)
    STAGES_IDS: int = 2  # the list of item i and i+1 (the one-item-ahead prefetch)
    STAGES_GATE: int = 1
    SCHEDULER_STAGES: int = 2

    # --- warp population
    SOFTMAX_WARPS: int = _SPARSE_SOFTMAX_WARPS
    GATHER_WARPS: int = 8
    AUX_WARPS: int = _SPARSE_AUX_WARPS
    TOTAL_WARPS: int = 16
    THREADS_PER_CTA: int = 16 * 32
    SOFTMAX_WARP_BASE: int = 0
    GATHER_WARP_BASE: int = 4
    MMA_WARP_ID: int = 12
    TMALDG_WARP_ID: int = 13
    SCHED_WARP_ID: int = 14
    SPARE_WARP_ID: int = 15

    # --- mbarrier arrival counts (every one the EXACT per-phase sum of the body's arrive sites)
    ONE_LANE: int = 1
    SOFTMAX_LANES: int = 128
    KV_FULL_ARRIVERS: int = 8
    IDS_EMPTY_ARRIVERS: int = 12
    READ_TILE_ARRIVERS: int = 14
    BAR_TMEM_THREADS: int = 160  # named barrier 1: the MMA warp + the softmax warps
    BAR_SOFTMAX_THREADS: int = 128  # named barrier 2: the softmax warps

    # --- gather geometry (per operand stage)
    BLOCKS_PER_WARP: int = 4
    GATHER_BOXES: int = 4  # 128-B column boxes per 256-wide row
    GATHER_BOX_ELEMS: int = 64
    GATHER_ISSUE_BYTES: int = 512  # 4 rows x 128 B
    GATHERS_PER_WARP_PER_STAGE: int = 16
    KV_TX_BYTES_PER_WARP: int = 8192  # the per-warp expect_tx
    KV_STAGE_BYTES: int = 65536

    # --- the Q^T / gate box (one token per item: Q_BOX_ROWS = the token's query heads)
    Q_BOX_TOKENS: int = 1
    Q_BOX_ROWS: int = 12
    Q_SLOT_BYTES: int = 8192  # N_Q x TILE_K x BPE
    Q_TX_BYTES: int = 6144  # Q_BOX_ROWS x TILE_K x BPE = the BOX bytes, never the slot
    GATE_TX_BYTES: int = 6144  # Q_BOX_ROWS x TILE_O x GATE_BPE

    # --- the staged block list
    IDS_TX_BYTES: int = 2048  # BLOCK_TOPK x 4 = the bulk-copy length = the expect_tx
    IDS_SLOT_BYTES: int = 2112  # IDS_TX_BYTES + the 64-B reserve

    # --- TMEM (64 columns: two S^T slots, two O^T d-blocks)
    TMEM_COLS: int = 64
    S_ACC_OFF: Tuple[int, int] = (0, 16)
    O_OFF: Tuple[int, int] = (32, 48)

    # --- registers (the entry count is what the launch gives every warp; the split is for when .maxnreg lands)
    ENTRY_REGS: int = 128
    SOFTMAX_REGS: int = 240
    GATHER_REGS: int = 96
    AUX_REGS: int = 80
    REG_SPLIT_DECLARED: int = 1

    # --- SMEM (derived by d256_sparse_smem_layout; pinned here so the body can read them)
    SMEM_TOTAL_BYTES: int = 226688
    SMEM_CARVEOUT_BYTES: int = SMEM_STANDARD_CARVEOUT_BYTES
    DESC_VERSION: int = 0

    # --- graph-derived features / arms
    INCLUDE_OPEN_BLOCK: int = 1
    EPILOGUE_GATE: int = 0
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0
    THD_VARLEN: int = 0
    BOTTOM_RIGHT: int = 0
    SPLIT_KV: int = 1
    LIST_PER_SEQUENCE: int = 0
    PACK_GQA: int = 1
    QH_PER_KH: int = 12
    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0
    STATS_LOG2: int = 0
    SCHEDULER_POLICY: int = SCHED_NATURAL
    MASK_FLAGS: int = MASK_NONE  # the selection IS the mask: no band bits ever
    HAS_SINK: int = 0
    CTA_MMA: int = 1
    CGA_M: int = 1
    CGA_N: int = 1

    # --- the softmax role's column groups (APPENDED; the softmax split of the body tuning): SOFTMAX_GROUPS groups of
    # SOFTMAX_WARPS_PER_GROUP warps, each owning COLS_PER_GROUP = N_Q / SOFTMAX_GROUPS columns of S^T / P^T / O^T over all 128 lanes;
    # SOFTMAX_WARPS = SOFTMAX_WARPS_PER_GROUP x SOFTMAX_GROUPS and every softmax-fired barrier init (SOFTMAX_LANES, the named
    # TMEM barrier, IDS_EMPTY_ARRIVERS, READ_TILE_ARRIVERS) follows it; the exchange barrier counts ONE group's lanes.
    SOFTMAX_GROUPS: int = 1
    SOFTMAX_WARPS_PER_GROUP: int = _SPARSE_SOFTMAX_WARPS_PER_GROUP
    COLS_PER_GROUP: int = _SPARSE_N_Q


def d256_sparse_smem_layout(cfg: CfgD256Sparse) -> dict:
    """The sparse kernel's SMEM byte table as the allocator lays it out: declaration order = address order, the descriptor-read
    operands FIRST (``sQ`` | ``sP`` | ``sKV``, each a 1024-B-aligned ``cutlass.Array``) so every MMA operand starts below the 256 KiB
    version-0 descriptor window, then the register-addressed buffers (``sIds`` | ``red`` | misc, 16-B aligned) so no alignment padding is spent.
    Returns every start, the start of the LAST K/V stage (what the descriptor version is derived from), the total and the carveout
    the total needs."""
    starts = {}
    off = 0
    starts["sQ"] = off
    off = _align_slab(off + cfg.STAGES_Q * cfg.N_Q * cfg.TILE_K * cfg.BPE)
    starts["sP"] = off
    off = _align_slab(off + cfg.STAGES_S * cfg.TILE_N * cfg.N_Q * cfg.BPE)
    starts["sKV"] = off
    kv_stage = cfg.TILE_N * cfg.TILE_K * cfg.BPE
    starts["sKV_last"] = off + (cfg.STAGES_KV - 1) * kv_stage
    off = _align_slab(off + cfg.STAGES_KV * kv_stage)
    starts["sIds"] = off
    off = _align16(off + cfg.STAGES_IDS * cfg.IDS_SLOT_BYTES)
    starts["red"] = off
    off = _align16(
        off + (cfg.STAGES_S + 1) * cfg.SOFTMAX_WARPS * cfg.COLS_PER_GROUP * 4
    )  # [slot][warp][column]: one slot per S ring parity + the epilogue's lane sum = 768 B at either group count
    starts["misc"] = off
    off += _SPARSE_SMEM_MISC_BYTES
    total = off
    if total <= SMEM_STANDARD_CARVEOUT_BYTES:
        carveout = SMEM_STANDARD_CARVEOUT_BYTES
    else:
        carveout = SMEM_USABLE_BYTES  # the oversized mode (ALLOW_OVERSIZED_SHARED_MEMORY); may still not fit -- the validator says
    return {"starts": starts, "total": total, "carveout": carveout, "desc_version": 1 if starts["sKV_last"] >= TCGEN05_V0_ADDR_LIMIT else 0}


def _validate_cfg_d256_sparse(cfg: CfgD256Sparse, flavor: str = _SPARSE_FLAVOR) -> None:
    """Every geometry claim of the kernel header's tables, as a raising predicate naming what the failure looks like."""
    lay = d256_sparse_smem_layout(cfg)
    st = lay["starts"]
    _check(
        [
            (cfg.DTYPE_QKV in (_DTYPE_BF16, _DTYPE_FP16), f"{flavor}: f16/bf16 inputs only (got DTYPE_QKV={cfg.DTYPE_QKV}); the gathered tile has no FP8 arm"),
            (cfg.DTYPE_O == cfg.DTYPE_QKV, f"{flavor}: half input requires DTYPE_O == DTYPE_QKV (got {cfg.DTYPE_O} vs {cfg.DTYPE_QKV})"),
            (cfg.BPE == 2 and cfg.BPE_O == 2 and cfg.GATE_BPE == 2, f"{flavor}: BPE / BPE_O / GATE_BPE are 2 on the half kernel"),
            (
                cfg.TILE_N == 128 and cfg.TILE_N == cfg.BLOCKS_PER_TILE * cfg.BLOCK_SIZE,
                f"{flavor}: the KV tile is the 128-lane TMEM layout = BLOCKS_PER_TILE x BLOCK_SIZE (got {cfg.TILE_N})",
            ),
            (
                cfg.BLOCK_SIZE == _SPARSE_BLOCK_SIZE and cfg.BLOCKS_PER_TILE == 32,
                f"{flavor}: a block is the four rows of one gather4 and a tile holds 32 of them",
            ),
            (cfg.TILE_K == 256 and cfg.TILE_O == 256, f"{flavor}: d_qk = d_v = 256 only (the gather box count and the 4-subtile Q box are derived for 256)"),
            (cfg.N_Q == _SPARSE_N_Q, f"{flavor}: N_Q must be 16 (one softmax column group); got {cfg.N_Q}"),
            # warp population: role-homogeneous warpgroups
            (
                cfg.SOFTMAX_GROUPS in _SPARSE_SOFTMAX_GROUPS
                and cfg.SOFTMAX_WARPS_PER_GROUP == _SPARSE_SOFTMAX_WARPS_PER_GROUP
                and cfg.SOFTMAX_WARPS == cfg.SOFTMAX_WARPS_PER_GROUP * cfg.SOFTMAX_GROUPS
                and cfg.AUX_WARPS == _SPARSE_AUX_WARPS,
                f"{flavor}: 4 softmax warps per column group x SOFTMAX_GROUPS in {{1, 2}} (got SOFTMAX_WARPS={cfg.SOFTMAX_WARPS}, "
                f"SOFTMAX_GROUPS={cfg.SOFTMAX_GROUPS}, SOFTMAX_WARPS_PER_GROUP={cfg.SOFTMAX_WARPS_PER_GROUP}) and the 4-warp MMA / TMA / scheduler / spare group",
            ),
            (
                cfg.N_Q % cfg.SOFTMAX_GROUPS == 0
                and cfg.COLS_PER_GROUP == cfg.N_Q // cfg.SOFTMAX_GROUPS
                and cfg.COLS_PER_GROUP % 2 == 0
                and cfg.COLS_PER_GROUP * cfg.BPE >= 16,
                f"{flavor}: COLS_PER_GROUP must be N_Q / SOFTMAX_GROUPS, even, and at least one 16-B P^T chunk per lane row (got {cfg.COLS_PER_GROUP} at "
                f"SOFTMAX_GROUPS={cfg.SOFTMAX_GROUPS}); a narrower group would store a partial swizzled chunk",
            ),
            (
                cfg.GATHER_WARPS >= 1 and 32 % cfg.GATHER_WARPS == 0,
                f"{flavor}: 32 blocks per tile must divide across the gather warps (got {cfg.GATHER_WARPS})",
            ),
            (
                cfg.GATHER_WARPS % 4 == 0,
                f"{flavor}: the gather warps must fill whole warpgroups (setmaxnreg is warpgroup-collective); got {cfg.GATHER_WARPS} -- a partial warpgroup would carry two roles at one register count",
            ),
            (
                cfg.TOTAL_WARPS == cfg.SOFTMAX_WARPS + cfg.GATHER_WARPS + cfg.AUX_WARPS and cfg.TOTAL_WARPS % 4 == 0,
                f"{flavor}: TOTAL_WARPS must be SOFTMAX_WARPS + GATHER_WARPS + 4 (= {cfg.SOFTMAX_WARPS} + {cfg.GATHER_WARPS} + 4) and a multiple of 4 (got {cfg.TOTAL_WARPS})",
            ),
            (cfg.THREADS_PER_CTA == 32 * cfg.TOTAL_WARPS, f"{flavor}: THREADS_PER_CTA must be 32 x TOTAL_WARPS"),
            (cfg.SOFTMAX_WARP_BASE == 0 and cfg.GATHER_WARP_BASE == cfg.SOFTMAX_WARPS, f"{flavor}: the gather warps follow the softmax warpgroup(s)"),
            (
                cfg.MMA_WARP_ID == cfg.GATHER_WARP_BASE + cfg.GATHER_WARPS
                and cfg.TMALDG_WARP_ID == cfg.MMA_WARP_ID + 1
                and cfg.SCHED_WARP_ID == cfg.MMA_WARP_ID + 2
                and cfg.SPARE_WARP_ID == cfg.MMA_WARP_ID + 3 == cfg.TOTAL_WARPS - 1,
                f"{flavor}: the last warpgroup is MMA / TMA-LDG / scheduler / spare in that order (a wrong warp takes a role)",
            ),
            # arrival counts == the body's arrive sites
            (cfg.ONE_LANE == 1 and cfg.SOFTMAX_LANES == 32 * cfg.SOFTMAX_WARPS, f"{flavor}: ONE_LANE is 1 and SOFTMAX_LANES is 32 x SOFTMAX_WARPS"),
            (
                cfg.KV_FULL_ARRIVERS == cfg.GATHER_WARPS,
                f"{flavor}: mb_kv_full's init is one expect_tx per gather warp (got {cfg.KV_FULL_ARRIVERS} vs {cfg.GATHER_WARPS}); a mismatch is a stage that never fills or over-arrives",
            ),
            (
                cfg.IDS_EMPTY_ARRIVERS == cfg.GATHER_WARPS + cfg.SOFTMAX_WARPS,
                f"{flavor}: mb_ids_empty's init is one elected lane per list-reading warp = GATHER_WARPS + SOFTMAX_WARPS (got {cfg.IDS_EMPTY_ARRIVERS})",
            ),
            (
                cfg.READ_TILE_ARRIVERS == cfg.SOFTMAX_WARPS + 1 + 1 + cfg.GATHER_WARPS,
                f"{flavor}: READ_TILE_ARRIVERS must be softmax + MMA + TMA + gather = {cfg.SOFTMAX_WARPS + 2 + cfg.GATHER_WARPS} (got {cfg.READ_TILE_ARRIVERS}); "
                f"a wrong count is an unreachable scheduler mbarrier, i.e. a hang at EVERY shape",
            ),
            (
                cfg.BAR_TMEM_THREADS == 32 * (cfg.SOFTMAX_WARPS + 1) and cfg.BAR_SOFTMAX_THREADS == 32 * cfg.SOFTMAX_WARPS_PER_GROUP,
                f"{flavor}: the named barriers count the MMA warp + EVERY softmax warp (160 at one column group, 288 at two) and ONE column group's "
                f"four warps (128) -- got {cfg.BAR_TMEM_THREADS} / {cfg.BAR_SOFTMAX_THREADS}",
            ),
            # gather bytes: the per-warp expect_tx x warps == the stage
            (cfg.BLOCKS_PER_WARP == cfg.BLOCKS_PER_TILE // cfg.GATHER_WARPS, f"{flavor}: BLOCKS_PER_WARP must be 32 / GATHER_WARPS"),
            (
                cfg.GATHER_BOXES == (cfg.TILE_K * cfg.BPE) // _GATHER4_BOX_BYTES and cfg.GATHER_BOX_ELEMS == _GATHER4_BOX_BYTES // cfg.BPE,
                f"{flavor}: a 256-wide bf16 row is 4 boxes of 64 elements",
            ),
            (cfg.GATHER_ISSUE_BYTES == _GATHER4_ROWS * _GATHER4_BOX_BYTES, f"{flavor}: one gather4 moves 4 rows x 128 B = 512 B"),
            (
                cfg.GATHERS_PER_WARP_PER_STAGE == cfg.BLOCKS_PER_WARP * cfg.GATHER_BOXES,
                f"{flavor}: gathers per warp per stage = blocks per warp x boxes per row",
            ),
            (
                cfg.KV_TX_BYTES_PER_WARP == cfg.GATHERS_PER_WARP_PER_STAGE * cfg.GATHER_ISSUE_BYTES,
                f"{flavor}: the per-warp expect_tx must equal its gather4 bytes (got {cfg.KV_TX_BYTES_PER_WARP}); a consumer waiting on bytes that never come is a hang",
            ),
            (
                cfg.KV_TX_BYTES_PER_WARP * cfg.GATHER_WARPS == cfg.KV_STAGE_BYTES == cfg.TILE_N * cfg.TILE_K * cfg.BPE,
                f"{flavor}: the gather warps' expect_tx must sum to the 64 KiB stage ({cfg.KV_TX_BYTES_PER_WARP} x {cfg.GATHER_WARPS} vs {cfg.KV_STAGE_BYTES})",
            ),
            # the Q^T / gate box: one token per item
            (cfg.PACK_GQA == 1, f"{flavor}: the work item IS the token's packed query-head group (PACK_GQA must be 1)"),
            (
                cfg.Q_BOX_TOKENS == 1,
                f"{flavor}: one query token per work item (Q_BOX_TOKENS must be 1; got {cfg.Q_BOX_TOKENS}) -- a multi-token item is a different kernel form",
            ),
            (
                1 <= cfg.QH_PER_KH <= cfg.N_Q,
                f"{flavor}: the token's query heads must fit the N_Q = {cfg.N_Q} tile (got qh_per_kh={cfg.QH_PER_KH}); above it the adapter declines",
            ),
            (cfg.Q_BOX_ROWS == cfg.Q_BOX_TOKENS * cfg.QH_PER_KH, f"{flavor}: Q_BOX_ROWS must be Q_BOX_TOKENS x QH_PER_KH (got {cfg.Q_BOX_ROWS})"),
            (cfg.Q_SLOT_BYTES == cfg.N_Q * cfg.TILE_K * cfg.BPE, f"{flavor}: a Q^T slot is N_Q x TILE_K x BPE = 8 KiB"),
            (
                cfg.Q_TX_BYTES == cfg.Q_BOX_ROWS * cfg.TILE_K * cfg.BPE,
                f"{flavor}: Q_TX_BYTES must be the Q BOX's bytes (Q_BOX_ROWS x TILE_K x BPE = {cfg.Q_BOX_ROWS * cfg.TILE_K * cfg.BPE}, got {cfg.Q_TX_BYTES}); "
                f"an expect_tx of the 8 KiB slot against a {cfg.Q_BOX_ROWS}-row delivery never completes -- a hang at every shape",
            ),
            (
                cfg.GATE_TX_BYTES == cfg.Q_BOX_ROWS * cfg.TILE_O * cfg.GATE_BPE,
                f"{flavor}: GATE_TX_BYTES must be the gate BOX's bytes (Q_BOX_ROWS x TILE_O x GATE_BPE = {cfg.Q_BOX_ROWS * cfg.TILE_O * cfg.GATE_BPE}, got {cfg.GATE_TX_BYTES})",
            ),
            (cfg.Q_TX_BYTES <= cfg.Q_SLOT_BYTES and cfg.GATE_TX_BYTES <= cfg.Q_SLOT_BYTES, f"{flavor}: the Q / gate box must fit the aliased slot"),
            # swizzles and the k-step
            (cfg.Q_SWZ_BYTES == cfg.K_SWZ_BYTES == cfg.V_SWZ_BYTES == 128, f"{flavor}: Q / K / V operands are SW128 (the gather box is one 128-B span)"),
            (
                cfg.P_SWZ_BYTES == cfg.N_Q * cfg.BPE and cfg.P_SWZ_BYTES in (32, 64),
                f"{flavor}: P^T row bytes must be the 32 B or 64 B swizzle atom (N_Q x BPE); got {cfg.P_SWZ_BYTES}",
            ),
            (cfg.TILE_K_HW == 16, f"{flavor}: TILE_K_HW must be 16 (1-chunk f16 on SM10x; 2-chunk is silently wrong)"),
            # the staged list
            (
                _SPARSE_TOPK_MIN <= cfg.BLOCK_TOPK <= _SPARSE_TOPK_MAX and cfg.BLOCK_TOPK % 4 == 0,
                f"{flavor}: BLOCK_TOPK must be a multiple of 4 in [{_SPARSE_TOPK_MIN}, {_SPARSE_TOPK_MAX}] (got {cfg.BLOCK_TOPK}); the ids row stride is a bulk-copy length (16-B multiples)",
            ),
            (cfg.IDS_TX_BYTES == cfg.BLOCK_TOPK * 4 and cfg.IDS_TX_BYTES % 16 == 0, f"{flavor}: IDS_TX_BYTES must be BLOCK_TOPK x 4 (got {cfg.IDS_TX_BYTES})"),
            (
                cfg.IDS_SLOT_BYTES == cfg.IDS_TX_BYTES + _SPARSE_IDS_SLOT_PAD_BYTES and cfg.IDS_SLOT_BYTES % 16 == 0,
                f"{flavor}: IDS_SLOT_BYTES must be IDS_TX_BYTES + the 64-B reserve",
            ),
            (cfg.MAX_TILES_PER_ITEM == -(-(cfg.BLOCK_TOPK + 1) // cfg.BLOCKS_PER_TILE), f"{flavor}: MAX_TILES_PER_ITEM is ceil((BLOCK_TOPK + 1) / 32)"),
            (cfg.STAGES_KV >= 2, f"{flavor}: the K/V ring needs at least K(t) and V(t) resident (got {cfg.STAGES_KV})"),
            (
                cfg.STAGES_Q == 2 and cfg.STAGES_IDS == 2 and cfg.STAGES_GATE == 1 and cfg.SCHEDULER_STAGES == 2,
                f"{flavor}: Q / ids rings are 2-deep (the one-item-ahead prefetch), the gate single-slot, the scheduler 2-deep",
            ),
            (
                cfg.STAGES_S == len(cfg.S_ACC_OFF) == 2,
                f"{flavor}: STAGES_S (got {cfg.STAGES_S}) must be the number of S^T TMEM slots (len(S_ACC_OFF) = {len(cfg.S_ACC_OFF)}) = 2: the s_full / "
                f"s_empty / p_full / bmm2_done rings, the sP buffers and the column-max scratch's parity slots are all that deep -- a depth the TMEM "
                f"map does not hold is a stale S^T slot read from the first reuse (a silent wrong answer, no crash)",
            ),
            # SMEM: the byte table with alignment modelled, against the carveout the config claims
            (
                st["sQ"] % 1024 == 0 and st["sP"] % 1024 == 0 and st["sKV"] % 1024 == 0,
                f"{flavor}: every descriptor-read / SW128 buffer must start 1024-B aligned (sQ {st['sQ']}, sP {st['sP']}, sKV {st['sKV']})",
            ),
            (st["sIds"] % 16 == 0, f"{flavor}: the bulk-copy destination sIds must start 16-B aligned (got {st['sIds']})"),
            (cfg.SMEM_TOTAL_BYTES == lay["total"], f"{flavor}: SMEM_TOTAL_BYTES ({cfg.SMEM_TOTAL_BYTES}) disagrees with the layout ({lay['total']})"),
            (
                cfg.SMEM_CARVEOUT_BYTES in (SMEM_STANDARD_CARVEOUT_BYTES, SMEM_USABLE_BYTES),
                f"{flavor}: SMEM_CARVEOUT_BYTES is the 227 KiB standard carveout or the 320 KiB usable oversized one",
            ),
            (
                lay["total"] <= SMEM_USABLE_BYTES,
                f"{flavor}: SMEM {lay['total']} B (sQ {cfg.STAGES_Q} x {cfg.N_Q * cfg.TILE_K * cfg.BPE} + sP {cfg.STAGES_S} x {cfg.TILE_N * cfg.N_Q * cfg.BPE} + sKV {cfg.STAGES_KV} x "
                f"{cfg.TILE_N * cfg.TILE_K * cfg.BPE} + sIds {cfg.STAGES_IDS} x {cfg.IDS_SLOT_BYTES} + red {(cfg.STAGES_S + 1) * cfg.SOFTMAX_WARPS * cfg.COLS_PER_GROUP * 4} + misc {_SPARSE_SMEM_MISC_BYTES}, "
                f"1024-B aligned) exceeds the {SMEM_USABLE_BYTES // 1024} KiB usable Rubin carveout; overflowing it does NOT fail the launch, it clobbers the last buffer",
            ),
            (
                cfg.SMEM_CARVEOUT_BYTES == lay["carveout"],
                f"{flavor}: SMEM {lay['total']} B needs the {lay['carveout'] // 1024} KiB carveout (standard 227 KiB = L1 intact; oversized = L1 to 8 kB), the config claims {cfg.SMEM_CARVEOUT_BYTES // 1024} KiB",
            ),
            (
                cfg.DESC_VERSION == lay["desc_version"],
                f"{flavor}: the last K/V stage starts at {st['sKV_last']} B, so the tcgen05 descriptor version must be {lay['desc_version']} (got {cfg.DESC_VERSION}); "
                f"a version-0 descriptor at or past {TCGEN05_V0_ADDR_LIMIT} wraps to offset 0 and the MMA multiplies the bottom of SMEM",
            ),
            # TMEM
            (
                cfg.TMEM_COLS == 64 and cfg.TMEM_COLS >= (cfg.STAGES_S + len(cfg.O_OFF)) * cfg.N_Q and (cfg.TMEM_COLS & (cfg.TMEM_COLS - 1)) == 0,
                f"{flavor}: TMEM is (STAGES_S + 2 O^T d-blocks) x N_Q = 4 x N_Q fp32 columns rounded to a power of two = 64",
            ),
            (
                cfg.S_ACC_OFF == (0, cfg.N_Q) and cfg.O_OFF == (2 * cfg.N_Q, 3 * cfg.N_Q),
                f"{flavor}: TMEM map is S^T slots at [0, N_Q), [N_Q, 2 N_Q) and O^T d-blocks at [2 N_Q, 3 N_Q), [3 N_Q, 4 N_Q)",
            ),
            # registers
            (
                cfg.ENTRY_REGS == sparse_entry_regs(cfg.TOTAL_WARPS),
                f"{flavor}: ENTRY_REGS must be the launch's per-thread count for {cfg.TOTAL_WARPS} warps = {sparse_entry_regs(cfg.TOTAL_WARPS)} (got {cfg.ENTRY_REGS})",
            ),
            (
                all(r % 8 == 0 and 24 <= r <= 256 for r in (cfg.SOFTMAX_REGS, cfg.GATHER_REGS, cfg.AUX_REGS, cfg.ENTRY_REGS)),
                f"{flavor}: every per-role register count must be a multiple of 8 in 24..256",
            ),
            (
                (not cfg.REG_SPLIT_DECLARED)
                or cfg.SOFTMAX_WARPS * cfg.SOFTMAX_REGS + cfg.GATHER_WARPS * cfg.GATHER_REGS + cfg.AUX_WARPS * cfg.AUX_REGS == cfg.ENTRY_REGS * cfg.TOTAL_WARPS,
                f"{flavor}: the register split must balance the ENTRY pool: {cfg.SOFTMAX_WARPS} x {cfg.SOFTMAX_REGS} + {cfg.GATHER_WARPS} x {cfg.GATHER_REGS} + {cfg.AUX_WARPS} x {cfg.AUX_REGS} = "
                f"{cfg.SOFTMAX_WARPS * cfg.SOFTMAX_REGS + cfg.GATHER_WARPS * cfg.GATHER_REGS + cfg.AUX_WARPS * cfg.AUX_REGS} != {cfg.ENTRY_REGS} x {cfg.TOTAL_WARPS} = {cfg.ENTRY_REGS * cfg.TOTAL_WARPS}; "
                f"a sum above the pool parks the last INCREASE warp forever (a hang at 100 % SM utilisation at every shape)",
            ),
            # features
            (
                cfg.MASK_FLAGS == MASK_NONE,
                f"{flavor}: the selection IS the mask -- no band / padded mask bits (got MASK_FLAGS={cfg.MASK_FLAGS}); per-batch lengths ride SEQ_KV_LENS_PRESENT",
            ),
            (cfg.HAS_SINK == 0, f"{flavor}: no sink"),
            (cfg.CTA_MMA == 1 and cfg.CGA_M == 1 and cfg.CGA_N == 1, f"{flavor}: one cga1 CTA per SM"),
            (
                not cfg.PAGED_KV or (cfg.PAGE_SIZE > 0 and cfg.PAGE_SIZE % cfg.BLOCK_SIZE == 0),
                f"{flavor}: page_size must be a positive multiple of BLOCK_SIZE = {cfg.BLOCK_SIZE} (got {cfg.PAGE_SIZE}) or a block straddles two pages",
            ),
            (
                not cfg.PAGED_KV or cfg.SEQ_KV_LENS_PRESENT == 1,
                f"{flavor}: the paged form must force SEQ_KV_LENS_PRESENT=1 -- a page pool has no dense extent, so the visible range (and the "
                f"gather rows past it, which read as -1) can only come from the per-batch KV lengths",
            ),
            (
                cfg.PAGED_KV == 1 or cfg.PAGE_SIZE == 0,
                f"{flavor}: PAGE_SIZE is 0 exactly when the kernel is not paged (the body divides by it only under PAGED_KV)",
            ),
            (cfg.SPLIT_KV >= 1, f"{flavor}: split_kv must be >= 1"),
            (
                cfg.SPLIT_KV <= cfg.MAX_TILES_PER_ITEM,
                f"{flavor}: split_kv ({cfg.SPLIT_KV}) must not exceed MAX_TILES_PER_ITEM ({cfg.MAX_TILES_PER_ITEM}) -- a larger split has an EMPTY chunk on "
                f"every item (a dead work item that gathers 128 KiB of -1 rows and writes an identity partial)",
            ),
            (
                not (cfg.SPLIT_KV > 1 and cfg.EPILOGUE_GATE),
                f"{flavor}: split_kv > 1 with EPILOGUE_GATE is not served -- a split's partials are gate-free and the combine applies sigmoid(G) to the "
                f"fp32 merged value (the adapter compiles the ungated kernel and the gated combine)",
            ),
            (
                not (cfg.SPLIT_KV > 1 and cfg.THD_VARLEN),
                f"{flavor}: split_kv > 1 under THD is not served -- the persistent claim-counter scheduler hands out (token, KV head) units with no split axis",
            ),
            (
                not (cfg.LIST_PER_SEQUENCE and cfg.THD_VARLEN),
                f"{flavor}: list_per_sequence under THD is not served -- the packed form's block_ids are one row per packed token",
            ),
            (
                not cfg.LIST_PER_SEQUENCE or cfg.BOTTOM_RIGHT == 1,
                f"{flavor}: list_per_sequence requires BOTTOM_RIGHT -- the shared list is anchored at the step-0 position kv_len - S_q (the decode / MTP "
                f"form); a top-left per-sequence list has no serving meaning",
            ),
            (
                not (cfg.BOTTOM_RIGHT and cfg.THD_VARLEN),
                f"{flavor}: bottom_right under THD is not served -- the packed form's position is the token's offset in its sequence (top-left)",
            ),
            (
                not cfg.THD_VARLEN or cfg.SEQ_KV_LENS_PRESENT == 1,
                f"{flavor}: THD/varlen must force SEQ_KV_LENS_PRESENT=1, or every sequence attends the whole pack",
            ),
            (
                not (cfg.THD_VARLEN and cfg.PAGED_KV),
                f"{flavor}: THD x paged is not served -- the packed sequence's K / V row offset (cu_k[b]) composes with a dense [1, T_kv, H_kv, D] "
                f"tensor only, never with a page pool; one of THD_VARLEN / PAGED_KV per record (the adapter declines the combination by name)",
            ),
            (not cfg.SEQ_Q_LENS_PRESENT or cfg.SEQ_KV_LENS_PRESENT == 1, f"{flavor}: SEQ_Q_LENS_PRESENT requires SEQ_KV_LENS_PRESENT"),
            (cfg.SCHEDULER_POLICY == SCHED_NATURAL, f"{flavor}: the item scheduler is NATURAL in v1 (LPT needs per-item costs)"),
        ]
    )


def _sparse_arms_requested(params: TemplateParams) -> Tuple[str, ...]:
    arms = []
    if params.epilogue_gate:
        arms.append("epilogue_gate")
    if params.thd_varlen:
        arms.append("thd_varlen")
    if params.paged_kv:
        arms.append("paged_kv")
    if (params.split_kv or 1) > 1:
        arms.append("split_kv")
    if params.qsa_list_per_sequence:
        arms.append("list_per_sequence")
    if params.bottom_right:
        arms.append("bottom_right")
    if not params.qsa_include_open_block:
        arms.append("pure_list")
    if params.seq_q_lens_present:
        arms.append("seq_q_lens")
    return tuple(arms)


def make_cfg_d256_sparse(params: TemplateParams, *, gather_warps: int = 8, softmax_groups: int = 0) -> Tuple[CfgD256Sparse, TmaIters]:
    """Config for sm107/sparse_d256_f16.py from the adapter's TemplateParams (``qsa_block_topk`` = the list width, ``qh_per_kh`` =
    the token's query heads on the N axis).  ``gather_warps`` selects the gather population: 8 (the default) or 4 (the named
    fallback; -25 % gather issue rate).  The warp total is 4 x groups softmax + gather + 4 auxiliary warps, the entry register count
    the register file over it rounded down to the 8-register granule: 20 warps / 96 at the default (8 gather, two groups), 16 / 128
    at one group with 8 gather warps or two groups with 4, 12 / 168 at one group with 4.
    ``softmax_groups`` (appended) selects the softmax SPLIT: 0 = the record's ``qsa_softmax_groups`` (itself 0 = the flavor default,
    TWO 4-warp groups of 8 columns: 20 warps at 8 gather warps, 96 entry registers flat; 16 warps / 128 at the 4-gather-warp fallback),
    1 = ONE 4-warp column group of 16 columns (the pre-split body, 16 warps) -- a performance knob: the same function, bitwise, at either value.

    Backstop only: every rejection here must also be a decline of the adapter's claims record (reaching a ValueError from a
    served record is a claims bug, not a user error).  Returns ``(cfg, TmaIters)`` like every sibling factory."""
    flavor = _SPARSE_FLAVOR
    if not params.qsa_block_topk:
        raise ValueError(f"{flavor}: not an index-list record (qsa_block_topk=0); the dense templates serve it")
    if params.qsa_block_size != _SPARSE_BLOCK_SIZE:
        raise ValueError(f"{flavor}: qsa_block_size must be {_SPARSE_BLOCK_SIZE} (one tma_gather4 = four rows); got {params.qsa_block_size}")
    if params.dtype_qkv not in (_DTYPE_BF16, _DTYPE_FP16):
        raise ValueError(f"{flavor}: f16/bf16 inputs only (got dtype_qkv={params.dtype_qkv}); the gathered tile has no FP8 arm")
    dtype_o = resolve_dtype_o(params)
    if dtype_o != params.dtype_qkv:
        raise ValueError(f"{flavor}: half input requires dtype_o == dtype_qkv; got dtype_o={dtype_o}")
    if params.cta_mma != 1:
        raise ValueError(f"{flavor}: one cga1 CTA per SM (cta_mma must be 1; got {params.cta_mma})")
    if not params.pack_gqa:
        raise ValueError(f"{flavor}: the work item is the token's packed query-head group (pack_gqa must be True)")
    if not 1 <= params.qh_per_kh <= _SPARSE_N_Q:
        raise ValueError(f"{flavor}: qh_per_kh must be in 1..{_SPARSE_N_Q} (the token's query heads form the N axis); got {params.qh_per_kh}")
    if params.window_left is not None or params.window_right is not None:
        raise ValueError(f"{flavor}: the selection IS the mask; a diagonal band (window_left / window_right) is not part of this kernel")
    if params.has_sink:
        raise ValueError(f"{flavor}: no sink")
    if (
        params.decode_q_tile
        or params.decode_tile
        or params.ragged_q
        or params.mma_2x2
        or params.pv_bf16
        or params.softmax_f16
        or params.softmax_scale_prefolded
        or params.exp2_fma_split
        # the dense line's own compile-time options: the sparse body has no field for any of them, so a record that sets one would
        # render a kernel that silently ignores it (negate_scores: the signed scale is applied in-kernel instead)
        or params.negate_scores
        or params.kv_tail_mask
        or params.thd_batch_one
        or params.lpt_head_group != 1
        or params.lpt_q_tiles
        or params.lpt_l2_size_mib
    ):
        raise ValueError(
            f"{flavor}: decode_q_tile / decode_tile / ragged_q / mma_2x2 / pv_bf16 / softmax_f16 / softmax_scale_prefolded / exp2_fma_split / "
            f"negate_scores / kv_tail_mask / thd_batch_one / lpt_head_group / lpt_q_tiles / lpt_l2_size_mib are other kernels' specializations"
        )
    if params.sched_policy not in (None, SCHED_NATURAL):
        raise ValueError(f"{flavor}: sched_policy must be NATURAL or None in v1 (got {params.sched_policy})")
    if params.paged_kv and (params.page_size <= 0 or params.page_size % _SPARSE_BLOCK_SIZE != 0):
        raise ValueError(f"{flavor}: page_size must be a positive multiple of {_SPARSE_BLOCK_SIZE} (got {params.page_size}) or a block straddles two pages")
    if params.seq_q_lens_present and not params.seq_kv_lens_present:
        raise ValueError(f"{flavor}: SEQ_Q_LENS_PRESENT requires SEQ_KV_LENS_PRESENT")
    missing = [a for a in _sparse_arms_requested(params) if a not in SPARSE_D256_WIRED_ARMS]
    if missing:
        raise ValueError(
            f"{flavor}: the kernel body does not carry the {', '.join(missing)} arm(s) yet (wired: {sorted(SPARSE_D256_WIRED_ARMS) or 'none'}); the adapter declines them"
        )
    if gather_warps not in (4, 8):
        raise ValueError(f"{flavor}: gather_warps must be 8 (the default) or 4 (the named fallback); got {gather_warps}")
    # 0 = the record's value, itself 0 = the flavor default: TWO column groups since the A/B/A on a 212-SM Rubin GPU (the split +20..+29 %
    # over the one-group body, bitwise); softmax_groups=1 renders the one-group body (the measured base, 16 warps).
    groups = int(softmax_groups) or int(getattr(params, "qsa_softmax_groups", 0) or 0) or 2
    if groups not in _SPARSE_SOFTMAX_GROUPS:
        raise ValueError(
            f"{flavor}: softmax_groups must be 1 (one 4-warp column group of 16 columns) or 2 (two 4-warp groups of 8 columns, the softmax split); got {groups}"
        )

    b = bpe(params.dtype_qkv)
    tile_n, tile_k, tile_o, n_q = 128, 256, 256, _SPARSE_N_Q
    softmax_warps, aux_warps = _SPARSE_SOFTMAX_WARPS_PER_GROUP * groups, _SPARSE_AUX_WARPS
    total_warps = softmax_warps + gather_warps + aux_warps
    blocks_per_tile = tile_n // _SPARSE_BLOCK_SIZE
    blocks_per_warp = blocks_per_tile // gather_warps
    gather_boxes = (tile_k * b) // _GATHER4_BOX_BYTES
    gathers_per_warp = blocks_per_warp * gather_boxes
    issue_bytes = _GATHER4_ROWS * _GATHER4_BOX_BYTES
    q_box_tokens = 1
    q_box_rows = q_box_tokens * params.qh_per_kh
    ids_tx = params.qsa_block_topk * 4
    entry = sparse_entry_regs(total_warps)
    # The split for when .maxnreg lands: softmax 240 and gather 96 on both one-group populations; at two column groups (half the
    # per-warp softmax state) 136 / 72 on the 20-warp pool (96 x 20 = 1920) and 160 / 96 on the 16-warp fallback (128 x 16 = 2048);
    # the aux group takes the remainder of the ENTRY pool so SUM(role regs x warps) == entry x warps by construction
    # (80 at 16 warps, 168 at 12; 64 at 20; 96 on the two-group 16-warp fallback).
    softmax_regs, gather_regs = (240, 96) if groups == 1 else ((136, 72) if gather_warps == 8 else (160, 96))
    aux_regs = (entry * total_warps - softmax_warps * softmax_regs - gather_warps * gather_regs) // aux_warps
    probe = CfgD256Sparse(
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_o,
        N_Q=n_q,
        BLOCK_SIZE=_SPARSE_BLOCK_SIZE,
        BLOCKS_PER_TILE=blocks_per_tile,
        BLOCK_TOPK=params.qsa_block_topk,
        MAX_TILES_PER_ITEM=-(-(params.qsa_block_topk + 1) // blocks_per_tile),
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=bpe(dtype_o),
        GATE_BPE=2,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_o, 1, b),
        P_SWZ_BYTES=n_q * b,
        TILE_K_HW=tile_k_hw(params.dtype_qkv),
        STAGES_KV=3,
        STAGES_S=_SPARSE_STAGES_S,
        STAGES_Q=2,
        STAGES_IDS=2,
        STAGES_GATE=1,
        SCHEDULER_STAGES=2,
        SOFTMAX_WARPS=softmax_warps,
        GATHER_WARPS=gather_warps,
        AUX_WARPS=aux_warps,
        TOTAL_WARPS=total_warps,
        THREADS_PER_CTA=32 * total_warps,
        SOFTMAX_WARP_BASE=0,
        GATHER_WARP_BASE=softmax_warps,
        MMA_WARP_ID=softmax_warps + gather_warps,
        TMALDG_WARP_ID=softmax_warps + gather_warps + 1,
        SCHED_WARP_ID=softmax_warps + gather_warps + 2,
        SPARE_WARP_ID=softmax_warps + gather_warps + 3,
        ONE_LANE=1,
        SOFTMAX_LANES=32 * softmax_warps,
        KV_FULL_ARRIVERS=gather_warps,
        IDS_EMPTY_ARRIVERS=gather_warps + softmax_warps,
        READ_TILE_ARRIVERS=softmax_warps + 1 + 1 + gather_warps,
        BAR_TMEM_THREADS=32 * (softmax_warps + 1),
        BAR_SOFTMAX_THREADS=32 * _SPARSE_SOFTMAX_WARPS_PER_GROUP,  # ONE column group's four warps
        BLOCKS_PER_WARP=blocks_per_warp,
        GATHER_BOXES=gather_boxes,
        GATHER_BOX_ELEMS=_GATHER4_BOX_BYTES // b,
        GATHER_ISSUE_BYTES=issue_bytes,
        GATHERS_PER_WARP_PER_STAGE=gathers_per_warp,
        KV_TX_BYTES_PER_WARP=gathers_per_warp * issue_bytes,
        KV_STAGE_BYTES=tile_n * tile_k * b,
        Q_BOX_TOKENS=q_box_tokens,
        Q_BOX_ROWS=q_box_rows,
        Q_SLOT_BYTES=n_q * tile_k * b,
        Q_TX_BYTES=q_box_rows * tile_k * b,
        GATE_TX_BYTES=q_box_rows * tile_o * 2,
        IDS_TX_BYTES=ids_tx,
        IDS_SLOT_BYTES=ids_tx + _SPARSE_IDS_SLOT_PAD_BYTES,
        TMEM_COLS=64,
        S_ACC_OFF=tuple(i * n_q for i in range(_SPARSE_STAGES_S)),  # one N_Q-column S^T slot per ring parity
        O_OFF=tuple((_SPARSE_STAGES_S + i) * n_q for i in range(2)),  # the two O^T d-blocks after them
        ENTRY_REGS=entry,
        SOFTMAX_REGS=softmax_regs,
        GATHER_REGS=gather_regs,
        AUX_REGS=aux_regs,
        REG_SPLIT_DECLARED=1,
        SMEM_TOTAL_BYTES=0,  # filled from the layout below
        SMEM_CARVEOUT_BYTES=SMEM_STANDARD_CARVEOUT_BYTES,
        DESC_VERSION=0,
        INCLUDE_OPEN_BLOCK=int(params.qsa_include_open_block),
        EPILOGUE_GATE=int(params.epilogue_gate),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size) if params.paged_kv else 0,
        THD_VARLEN=int(params.thd_varlen),
        BOTTOM_RIGHT=int(params.bottom_right),
        SPLIT_KV=int(params.split_kv or 1),
        LIST_PER_SEQUENCE=int(params.qsa_list_per_sequence),
        SOFTMAX_GROUPS=groups,
        SOFTMAX_WARPS_PER_GROUP=_SPARSE_SOFTMAX_WARPS_PER_GROUP,
        COLS_PER_GROUP=n_q // groups,
        PACK_GQA=1,
        QH_PER_KH=int(params.qh_per_kh),
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.paged_kv) else int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        STATS_LOG2=int(params.stats_log2),
        SCHEDULER_POLICY=SCHED_NATURAL,
        MASK_FLAGS=MASK_NONE,
        HAS_SINK=0,
        CTA_MMA=1,
        CGA_M=1,
        CGA_N=1,
    )
    lay = d256_sparse_smem_layout(probe)
    cfg = dataclasses_replace(probe, SMEM_TOTAL_BYTES=lay["total"], SMEM_CARVEOUT_BYTES=lay["carveout"], DESC_VERSION=lay["desc_version"])
    _validate_cfg_d256_sparse(cfg, flavor)
    return cfg, _tma_iters(cfg)


# ---------------------------------------------------------------------------
# d256 DECODE flavor on Rubin -- the swap-AB decode tile (sm107/decode_d256_f16.py)
# ---------------------------------------------------------------------------
# The decode tile's body is arch-agnostic (cta_group::1, 6 / 10 / 18 warps at N_Q = 16 / 32 /
# 64, f16/bf16 TILE_K_HW = 16 on both arch lines, 4 x N_Q TMEM columns) and its record is the
# SM100 CfgD256Decode built by config_sm100.make_cfg_d256_decode -- with the Rubin width set
# (_D256_DECODE_Q_TILES_SM107 below: the SM100 pair plus the 64-column tile) -- re-validated
# here against the two Rubin facts a port can get wrong silently:
#   * the SMEM budget is checked against SMEM_USABLE_BYTES (320 KiB), never the 327 KiB
#     capacity (the d192 f16 50 %-zeros lesson above).  The tile's slabs -- Q^T, the
#     3-slot 64 KiB K/V ring, two P^T tiles, the reduction scratch -- sum to ~209 KiB at
#     N_Q = 16 and ~226 KiB at N_Q = 32, so those two widths fit the STANDARD 227 KiB
#     carveout (no oversized-SMEM launch, no L1 shrink); the 64-column tile's 258 KiB (264,080 B) take
#     the OVERSIZED carveout (L1 -> 8 kB, which costs this TMA-fed tile nothing it uses) --
#     the one width in _D256_DECODE_OVERSIZED_Q_TILES, admitted after its launch was VERIFIED
#     on cc 10.7 (the DSL sets the dynamic-SMEM attribute from the module's own total, the
#     way the 320 KiB d512 bodies launch).
#   * DESC_VERSION is DERIVED from the layout: every MMA-operand slab (Q^T, the K/V ring,
#     P^T) must START under TCGEN05_V0_ADDR_LIMIT for the version-0 tcgen05 descriptor the
#     kernel pins; the highest start is P^T at Q + ring = 200 / 208 / 224 KiB, and at N_Q = 64
#     the slab ENDS exactly at the line (224 + 32 KiB), so every address a version-0
#     descriptor forms stays inside the window -- the answer is 0 at every admitted record,
#     asserted on the start AND the end, not assumed (the 2026-09-04 silent-zero class).

_D256_DECODE_FLAVOR = "sm107 d256 decode"

_D256_DECODE_MAX_RED_SLOTS = 3  # the kernel's per-tile column-max slots: [3 rotating slots][column group][16 columns] of Int32 keys (red.shared.max)
_D256_DECODE_COLS = 16  # S^T / O^T columns one 4-warp softmax group owns; the epilogue's sum scratch is [softmax warp][16 columns] fp32
_D256_DECODE_BARRIER_WORDS = 1 + 2 * 3 + 2 * 4 + 1  # q_full, kv_full/empty x STAGES (3), s_full/s_empty/p_full/bmm2_done x 2, tmem_dealloc -- Int64 each
# The widths the Rubin factory COMPILES (the N extent of one token unit): the SM100 pair and the 64-column tile -- four
# 16-column softmax groups (16 softmax warps + the MMA warp + the TMA warp = 18 warps / 576 threads), ONE unit of up to 64
# packed rows, so an MTP step of four tokens x 12 / 16 heads (48 / 64 rows) streams the KV range ONCE where the 32-column
# tile cuts it into two token units (each a second read of the whole KV).  Its budgets (the barrier / SMEM / TMEM / register
# tables written with the change): SMEM 258 KiB = 264,080 B (the oversized carveout), TMEM 256 columns, the barrier inits scale through SOFTMAX_LANES
# (512), and ptxas caps the 18-warp CTA at 96 registers per thread (16384 per SM sub-partition / (5 warps x 32),
# 8-granular -- the per-thread work is width-independent and the sm_107a rendering carries no spill).  Whether the ROUTE
# sends a step to it is the route table's decision below (D256_DECODE_SINGLE_UNIT_Q_TILES names the widths it routes); a
# width in this tuple is compiled and template-tested.  The SM100 factory keeps (16, 32): the 64-column tile does not fit
# the 227 KiB Blackwell carveout.
_D256_DECODE_Q_TILES_SM107 = (16, 32, 64)
D256_DECODE_MAX_Q_ROWS_SM107 = max(
    _D256_DECODE_Q_TILES_SM107
)  # the widest tile the Rubin line compiles (64); the SM100 line's is config_sm100.D256_DECODE_MAX_Q_ROWS
# The widths whose SMEM exceeds the standard 227 KiB carveout and launch in the oversized mode (L1 -> 8 kB).  A width is
# added here only after its launch ran on cc 10.7 (the carveout is a launch attribute the DSL derives, never a knob this
# config sets; the validator below refuses a record past 227 KiB that is not in this tuple).
_D256_DECODE_OVERSIZED_Q_TILES = (64,)
# The FORMS the 64-column tile is admitted for -- the ones whose sm_107a rendering stays SPILL-FREE at ptxas's 96-register
# cap (MEASURED by host trace-compile over 30 arms, cuobjdump -res-usage STACK 0 / nvdisasm STL+LDL 0, 2026-10-15): a PACKED
# group of at least _D256_DECODE_WIDE_PACK_MIN heads (2 .. 16 measured; the unpacked 64-token step carries 16 per-column
# mask thresholds and spills 236 STL/LDL), and over a PAGED cache the causal / window mask (the serving MTP step: bottom-right
# causal; the padded-only paged arms spill 8-16 B of stack).  A dense cache compiles clean with or without a mask.  Everything
# else keeps the 32-column tile (two token units) or the prefill tile.  Pinned by the host spill probe in
# test_sdpa_fwd_decode_d256_sm107.py; a width or form added here is added there.
_D256_DECODE_WIDE_PACK_MIN = 2


def decode_d256_wide_tile_admits(pack_g: int, paged: bool, masked: bool) -> bool:
    """THE form predicate of the oversized widths (_D256_DECODE_OVERSIZED_Q_TILES: the 64-column tile) -- the facts the record
    validator (``_validate_cfg_d256_decode_sm107``) checks on the compiled record, read here on a DECLARATION so the route
    never names a width the record declines: a PACKED group of at least :data:`_D256_DECODE_WIDE_PACK_MIN` heads (``pack_g``
    = the heads packed per token; 1 = unpacked / MHA), and over a PAGED cache the causal / window mask (``masked``: the record's
    MASK_CAUSAL or MASK_SWA bit -- a right band (causal) or a left window declared; a dense cache takes any mask).  The
    adapter reads it on its own declaration (``SdpaFwdDslSm100._decode_q_tile_for``), the facts twins on the graph facts
    (``engines.decode_d256_q_tile_for_row``), the gated attention block on its always paged, always bottom-right-causal DECODE
    form; the route functions below take the same three facts.  Two readings of one rule: the validator keeps its own
    wording and ``test_sdpa_fwd_decode_d256_sm107.py`` pins their agreement over the form matrix."""
    return int(pack_g) >= _D256_DECODE_WIDE_PACK_MIN and (not bool(paged) or bool(masked))


# The Rubin ROUTE onto the decode tile (the cc 10.7 twin of config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS /
# decode_d256_q_tile; read by SdpaFwdDslSm100._decode_q_tile_for, engines.d256_decode_tile_selected and the
# heuristics' decode geometry through engines.decode_d256_q_tile_for_row -- keep the three in lockstep), TABLE-DRIVEN
# (2026-10-15):
#   * D256_DECODE_SINGLE_UNIT_Q_TILES is THE table -- the tile's N extents this line routes in ONE token unit, ordered
#     narrow to wide: a step whose packed rows S_q x G fit a width rides the narrowest such width, the KV streamed ONCE
#     per (sequence, KV head).  The 32-column tile IS routed here (rows in (16, 32]: 24/2 at S_q = 2 = 24 live rows,
#     32/2 at S_q = 2, an MHA step of 17-32 tokens), unlike the Blackwell line; the 64-column tile takes a packed step of
#     up to 64 rows in ONE unit (24/2 at S_q = 3 .. 5 = 36 / 48 / 60 rows, 32/2 at S_q = 3 / 4 = 48 / 64, 8/1 at S_q 5 .. 8,
#     64/2 at S_q = 2) wherever its form predicate admits the step (decode_d256_wide_tile_admits: a packed group of >= 2
#     heads; over pages the causal / window mask) -- the forms it declines (an unpacked step of 33-64 tokens, a mask-free
#     paged packed step) keep the (16, 32) table and its arms below, as before the width landed.  The rule reads the table
#     through decode_d256_route_widths and nothing else, and the gated attention block's decode route
#     (gated_attention_block.decode_route) reads the same table.
#   * past the table a PACKED group may be cut into up to D256_DECODE_ROUTED_MAX_TOKEN_UNITS TOKEN UNITS of the WIDEST
#     admitted width (the kernel's Q_TOKEN_UNITS axis: S_q = 6 .. 8 at 24/2 = two units of up to five tokens x 12 heads on
#     the 64-column tile, each streaming the KV range once -- 2x the KV bytes of a single stream; S_q = 3 / 4 at 24/2 ran
#     two units of the 32-column tile before the 64-column width landed, the measured MTP form) -- the ADAPTER's route for
#     a dense paged graph on this line, where the d256 prefill kernel serves no dense paged query.
#   * ``max_q_tile`` (every route function): a caller's CAP on the table -- the gated attention block's ``decode_q_tile_max``
#     and the adapter's ``decode_q_tile_max`` -- so a declaration can hold the route at the MEASURED widths (32: the route
#     before the 64-column width landed, value for value) beside the default; a performance-only knob (the same function
#     under every width: the SDPA suite's cross-form bitwise pins).
# MEASURED on cc 10.7 (212 SMs) at the 24/2 geometry ONLY -- G = 12: the one-unit 32-column form at S_q = 2 and
# the two-unit form at S_q = 3 / 4, against the prefill body and against the 16-column tile in single-token
# units, B in {1, 4, 32}, S_kv 2K..128K.  The envelope is WIDER than that: an unpacked / MHA step of 17-32
# tokens and a packed group with 16 < S_q x G <= 32 at G != 12 ride one unit of the 32-column tile, a packed
# group with 32 < S_q x G <= 64 one unit of the 64-column tile (UNMEASURED on a perf GPU: the width is compiled,
# template-tested and bitwise the two-unit form on the SDPA suite's cross-form pin), and a packed group with
# ceil(S_q / (64 // G)) = 2 two units of it -- all routed by the same rule because each is the same tile with the same
# stream count per (head group, batch, split) as the measured form (the byte model does not see G), and each class has
# a GPU correctness cell in test_sdpa_fwd_decode_d256_sm107.py (the rule, the forms' agreement, the two-unit forms, an
# MHA step of 24 tokens, the 64-column one-unit forms).  A perf cell per class is owed before any of them is quoted.
# Past the envelope (a third token unit, an unpacked step above 32 tokens, a group wider than the tile) the d256
# prefill kernel serves the graph as before.
D256_DECODE_SINGLE_UNIT_Q_TILES: Tuple[int, ...] = (16, 32, 64)
D256_DECODE_ROUTED_MAX_Q_ROWS = D256_DECODE_SINGLE_UNIT_Q_TILES[-1]
D256_DECODE_ROUTED_MAX_TOKEN_UNITS = 2
# The table is sorted, positive, a subset of the widths this line COMPILES (_D256_DECODE_Q_TILES_SM107), never wider than
# D256_DECODE_MAX_Q_ROWS_SM107, and its oversized widths sit at the wide end (the admitted table of a form is a PREFIX): a
# new width lands in both places or this tripwire names the compile-side bound.
assert (
    list(D256_DECODE_SINGLE_UNIT_Q_TILES) == sorted(set(D256_DECODE_SINGLE_UNIT_Q_TILES)) and D256_DECODE_SINGLE_UNIT_Q_TILES[0] > 0
), D256_DECODE_SINGLE_UNIT_Q_TILES
assert all(
    w in _D256_DECODE_Q_TILES_SM107 for w in D256_DECODE_SINGLE_UNIT_Q_TILES
), f"the single-unit table {D256_DECODE_SINGLE_UNIT_Q_TILES} routes a width the Rubin factory does not compile ({_D256_DECODE_Q_TILES_SM107})"
assert D256_DECODE_ROUTED_MAX_Q_ROWS <= D256_DECODE_MAX_Q_ROWS_SM107, (
    f"the single-unit table's widest width {D256_DECODE_ROUTED_MAX_Q_ROWS} exceeds the width the decode tile compiles for, "
    f"D256_DECODE_MAX_Q_ROWS_SM107 = {D256_DECODE_MAX_Q_ROWS_SM107}"
)
assert all(w > n for w in _D256_DECODE_OVERSIZED_Q_TILES for n in D256_DECODE_SINGLE_UNIT_Q_TILES if n not in _D256_DECODE_OVERSIZED_Q_TILES), (
    D256_DECODE_SINGLE_UNIT_Q_TILES,
    _D256_DECODE_OVERSIZED_Q_TILES,
)


def decode_d256_route_widths(pack_g: int, *, paged: bool = False, masked: bool = True, max_q_tile: Optional[int] = None) -> Tuple[int, ...]:
    """The single-unit table a FORM may ride, narrow to wide: :data:`D256_DECODE_SINGLE_UNIT_Q_TILES` without the oversized
    widths wherever :func:`decode_d256_wide_tile_admits` declines the form (``pack_g`` the heads packed per token, ``paged``
    the cache kind, ``masked`` the causal / window mask), and without the widths past ``max_q_tile`` (a caller's cap; ``None``
    = the whole table).  Always a prefix of the table (the oversized widths are its wide end)."""
    cap = D256_DECODE_ROUTED_MAX_Q_ROWS if max_q_tile is None else int(max_q_tile)
    wide_ok = decode_d256_wide_tile_admits(pack_g, paged, masked)
    return tuple(w for w in D256_DECODE_SINGLE_UNIT_Q_TILES if w <= cap and (wide_ok or w not in _D256_DECODE_OVERSIZED_Q_TILES))


def decode_d256_single_unit_q_tile(s_q: int, pack_g: int, *, paged: bool = False, masked: bool = True, max_q_tile: Optional[int] = None) -> int:
    """THE single-unit rule: the narrowest width of the form's table (:func:`decode_d256_route_widths`) holding ``s_q x pack_g``
    packed rows (``pack_g`` = the heads packed per token, 1 = unpacked / MHA) in ONE token unit -- the KV streamed once per
    (sequence, KV head) -- or 0 when the step is past the table (the callers decide: the adapter's token-unit arm below, the
    gated attention block's prefill route).  Reads the table and nothing else."""
    s_q, pack_g = int(s_q), int(pack_g)
    if s_q <= 0 or pack_g <= 0:
        return 0
    rows = s_q * pack_g
    for width in decode_d256_route_widths(pack_g, paged=paged, masked=masked, max_q_tile=max_q_tile):
        if rows <= width:
            return width
    return 0


def decode_d256_q_tile(
    s_q: int, pack_g: int, max_token_units: Optional[int] = None, *, paged: bool = False, masked: bool = True, max_q_tile: Optional[int] = None
) -> int:
    """The Rubin ADAPTER's route: the decode tile's N extent for ``s_q`` tokens packed ``pack_g`` heads per token (1 =
    unpacked), or 0 when the d256 prefill kernel serves the graph.  The single-unit rule first
    (:func:`decode_d256_single_unit_q_tile`); past the table a PACKED group rides the WIDEST width of the form's table in
    ``ceil(s_q / (width // pack_g))`` token units (:func:`config_sm100.decode_d256_q_units`: each unit one more KV stream)
    when that count is within ``max_token_units`` (default :data:`D256_DECODE_ROUTED_MAX_TOKEN_UNITS`; 1 = the single-unit
    rule alone) and the group fits the width; an unpacked / MHA step takes one unit only.  ``paged`` / ``masked`` /
    ``max_q_tile`` select the form's table (:func:`decode_d256_route_widths`); with ``max_q_tile=32`` this is, value for value,
    the route before the 64-column width landed (the former shared rule ``config_sm100.decode_d256_q_tile`` with the Rubin
    constants), pinned by the block's MTP route tests."""
    s_q, pack_g = int(s_q), int(pack_g)
    width = decode_d256_single_unit_q_tile(s_q, pack_g, paged=paged, masked=masked, max_q_tile=max_q_tile)
    if width or s_q <= 0 or pack_g <= 1:
        return width
    cap = D256_DECODE_ROUTED_MAX_TOKEN_UNITS if max_token_units is None else int(max_token_units)
    widths = decode_d256_route_widths(pack_g, paged=paged, masked=masked, max_q_tile=max_q_tile)
    if not widths or pack_g > widths[-1]:
        return 0
    wide = widths[-1]
    units = decode_d256_q_units(s_q, pack_g, wide)
    return wide if 0 < units <= cap else 0


def d256_decode_smem_layout(cfg: CfgD256Decode) -> dict:
    """Byte offsets of the decode tile's SMEM slabs in the kernel's DECLARATION order
    (``sQ_raw | sKV_raw | sP_raw | mred_smem | red_smem | tmem_ptr | mbarriers``), every slab
    1024-aligned like the kernel's ``cutlass.Array(..., alignment=1024)`` (the small
    trailing arrays at their 16-B alignment), plus the total, the standard-carveout
    verdict and the carveout the launch needs (227 KiB standard, else the 320 KiB
    oversized one).  The MMA-operand slabs are the first three; ``p_start`` is the
    highest operand start and is what the descriptor-version derivation reads;
    ``p_end`` is the last operand byte + 1 (the version-0 window check reads both)."""
    q_bytes = cfg.N_Q * cfg.TILE_K * cfg.BPE
    kv_bytes = cfg.STAGES_KV * cfg.TILE_N * cfg.TILE_K * cfg.BPE
    p_bytes = 2 * (cfg.TILE_N * cfg.N_Q // 2) * 4  # two P^T tiles of TILE_N key rows x N_Q/2 packed-half Int32 words
    mred_bytes = _D256_DECODE_MAX_RED_SLOTS * (cfg.N_Q // _D256_DECODE_COLS) * _D256_DECODE_COLS * 4  # the max-key slots, one per group per rotating slot
    red_bytes = cfg.SOFTMAX_WARPS * _D256_DECODE_COLS * 4  # the epilogue's per-warp column sums
    q_start = 0
    kv_start = _align_slab(q_start + q_bytes)
    p_start = _align_slab(kv_start + kv_bytes)
    mred_start = _align_slab(p_start + p_bytes)
    red_start = -(-(mred_start + mred_bytes) // 16) * 16
    tmem_ptr_start = -(-(red_start + red_bytes) // 16) * 16
    bars_start = -(-(tmem_ptr_start + 4) // 16) * 16
    total = bars_start + 8 * _D256_DECODE_BARRIER_WORDS
    return {
        "q_start": q_start,
        "kv_start": kv_start,
        "p_start": p_start,
        "mred_start": mred_start,
        "red_start": red_start,
        "tmem_ptr_start": tmem_ptr_start,
        "bars_start": bars_start,
        "p_end": p_start + p_bytes,
        "total": total,
        "fits_standard_carveout": total <= SMEM_STANDARD_CARVEOUT_BYTES,
        "carveout": SMEM_STANDARD_CARVEOUT_BYTES if total <= SMEM_STANDARD_CARVEOUT_BYTES else SMEM_USABLE_BYTES,
    }


def d256_decode_desc_version(cfg: CfgD256Decode) -> int:
    """The tcgen05 SMEM-descriptor version the decode tile's layout NEEDS: 1 as soon as an
    MMA-operand slab (the last one declared is P^T) starts at or past the 256 KiB window of a
    version-0 descriptor, else 0.  The kernel module reads this ONCE into its DESC_VERSION
    constant and wires it into every SmemTile (test_sm107_decode_descriptor_version_*)."""
    return 1 if d256_decode_smem_layout(cfg)["p_start"] >= TCGEN05_V0_ADDR_LIMIT else 0


def _validate_cfg_d256_decode_sm107(cfg: CfgD256Decode, flavor: str) -> None:
    """The Rubin arch facts on top of config_sm100._validate_cfg_d256_decode (which the SM100
    factory already ran on this record)."""
    smem = d256_decode_smem_layout(cfg)
    want_k = tile_k_hw(cfg.DTYPE_QKV)
    _check(
        [
            (
                cfg.TILE_K_HW == want_k,
                f"{flavor}: TILE_K_HW must be {want_k} at DTYPE_QKV={cfg.DTYPE_QKV} on Rubin (got {cfg.TILE_K_HW}); the body's "
                f"f16 idesc leaves k_dim at its default, so a wider k-step silently scrambles accumulator rows",
            ),
            (
                smem["total"] <= SMEM_USABLE_BYTES,
                f"{flavor}: SMEM {smem['total']} B (Q^T + {cfg.STAGES_KV}-slot K/V ring + 2 P^T + reduction scratch + barriers) exceeds the "
                f"{SMEM_USABLE_BYTES // 1024} KiB usable Rubin carveout at N_Q={cfg.N_Q}, STAGES_KV={cfg.STAGES_KV}: {smem}.  Overflowing it "
                f"does NOT fail the launch -- it clobbers the last buffer allocated (measured on the prefill line: O came back 50 % zeros "
                f"with an exact LSE), so this must raise here",
            ),
            (
                cfg.N_Q in _D256_DECODE_Q_TILES_SM107,
                f"{flavor}: N_Q must be one of {_D256_DECODE_Q_TILES_SM107} (the widths this line compiles); got {cfg.N_Q}",
            ),
            # The two FORM checks below are decode_d256_wide_tile_admits read on the compiled record (the route reads it on the
            # declaration); their wording is pinned by the config backstop test, their agreement with the predicate by the form-matrix test.
            (
                cfg.N_Q not in _D256_DECODE_OVERSIZED_Q_TILES or (cfg.PACK_GQA == 1 and cfg.QH_PER_KH >= _D256_DECODE_WIDE_PACK_MIN),
                f"{flavor}: the {cfg.N_Q}-column tile serves PACKED groups of at least {_D256_DECODE_WIDE_PACK_MIN} heads only (got pack_gqa="
                f"{cfg.PACK_GQA}, qh_per_kh={cfg.QH_PER_KH}): its unpacked rendering spills at ptxas's register cap for 18 warps; an unpacked"
                f" step of more than 32 tokens rides the 32-column tile or the prefill kernel",
            ),
            (
                cfg.N_Q not in _D256_DECODE_OVERSIZED_Q_TILES or not cfg.PAGED_KV or (cfg.MASK_FLAGS & (MASK_CAUSAL | MASK_SWA)) != 0,
                f"{flavor}: the {cfg.N_Q}-column tile over a PAGED cache needs the causal / window mask (the MTP step's bottom-right causal"
                f" mask; got MASK_FLAGS={cfg.MASK_FLAGS}): its padded-only paged rendering spills at ptxas's register cap for 18 warps; a"
                f" mask-free paged step of that width rides the 32-column tile in two token units",
            ),
            (
                smem["fits_standard_carveout"] or cfg.N_Q in _D256_DECODE_OVERSIZED_Q_TILES,
                f"{flavor}: SMEM {smem['total']} B no longer fits the standard 227 KiB per-CTA carveout at N_Q={cfg.N_Q}, "
                f"STAGES_KV={cfg.STAGES_KV}, and this width is not in _D256_DECODE_OVERSIZED_Q_TILES ({_D256_DECODE_OVERSIZED_Q_TILES}: the "
                f"widths whose oversized-carveout launch (L1 -> 8 kB) was verified on cc 10.7); verify the launch before admitting it",
            ),
            (
                smem["p_start"] < TCGEN05_V0_ADDR_LIMIT and smem["p_end"] <= TCGEN05_V0_ADDR_LIMIT,
                f"{flavor}: the P^T operand slab spans [{smem['p_start']}, {smem['p_end']}) B, reaching past the {TCGEN05_V0_ADDR_LIMIT} B version-0 "
                f"tcgen05 descriptor window (N_Q={cfg.N_Q}, STAGES_KV={cfg.STAGES_KV}); the kernel derives DESC_VERSION from the slab START "
                f"(d256_decode_desc_version) and the descriptor's per-k-step advances must stay inside the window too, so an operand past "
                f"the line must be validated under version 1 before the record is admitted",
            ),
        ]
    )


def make_cfg_d256_decode(params: TemplateParams) -> Tuple[CfgD256Decode, TmaIters]:
    """Config for sm107/decode_d256_f16.py: the SM100 decode record (config_sm100.make_cfg_d256_decode,
    whose predicates are arch-neutral, built here over the Rubin width set _D256_DECODE_Q_TILES_SM107 --
    the SM100 pair plus the 64-column tile) re-validated against the Rubin SMEM carveout (standard or
    oversized) and the version-0 tcgen05 descriptor window.

    Backstop only (see TemplateParams): the adapter routes decode-shaped f16/bf16 d256 graphs here
    (``_decode_q_tile``) and keeps every other graph on the prefill tile, so a ValueError is an adapter
    gap, not a user error.  ``cta_mma`` / ``sched_policy`` are accepted and unused (one cta_group::1
    CTA per unit, nothing to schedule).  Not served by this tile, declined here so a mis-routed record
    cannot trace a silently incomplete kernel: the fused epilogue gate (no gate seams on the decode
    tile; the Rubin d256 PREFILL kernel serves gated graphs), the pre-folded softmax scale (the tile
    applies the scale in-kernel), the d512 2x2 / d64 decode / ragged-Q records of other templates.
    """
    flavor = _D256_DECODE_FLAVOR
    if params.epilogue_gate:
        raise ValueError(f"{flavor}: the fused epilogue gate is not wired on the decode tile (the Rubin d256 prefill kernel serves gated graphs)")
    if params.softmax_scale_prefolded:
        raise ValueError(f"{flavor}: softmax_scale_prefolded is not wired on the decode tile (it applies the scale in-kernel)")
    if getattr(params, "mma_2x2", False) or params.decode_tile or params.ragged_q:
        raise ValueError(f"{flavor}: mma_2x2 / decode_tile / ragged_q select other templates; the d256 decode tile does not consume them")
    cfg, tma = _make_cfg_d256_decode_record(params, _D256_DECODE_Q_TILES_SM107)
    _validate_cfg_d256_decode_sm107(cfg, flavor)
    return cfg, tma


# ---------------------------------------------------------------------------
# d256 index-list sparse DECODE tile, the LEAN form on Rubin (sm107/sparse_lean_d256_f16.py): mma.sync + cp.async, no TMEM / TMA / CLC
# ---------------------------------------------------------------------------
# The second body of the sparse function (the first is the tcgen05 tile above, make_cfg_d256_sparse): one CTA of 128 x N_MT threads per
# (sequence, KV head, chunk of SUB_TILES_PER_CTA 32-key sub-tiles) holding ALL S_q x G query rows of the sequence's KV head (S_q 1..4
# through TemplateParamsLean.decode_seqlen_q; N_MT = ceil(S_q x G / 16) m16 tiles of four warps each), every warp owning a 64-wide
# d-slice of its m-tile (BMM1 split-K over the 4 warps + an 8-KiB SMEM exchange per m-tile, the softmax redundant per warp, BMM2 on the
# warp's own 16 x 64 O slice), the one-shot zero-filled cp.async gather of the paged / dense rows the block list names -- ONE gathered K / V
# sub-tile per CTA shared by every m-tile (the tcgen05 tile re-gathers the list once per row) --, the SAME fp32 split-major partial slab +
# LSE as the tcgen05 tile's split store (sm100/split_combine.py consumes it unchanged).
# Geometry the body ASSUMES and this config PINS (sdpa-invariants section 6: the body is the truth): THREADS == 128 x N_MT (the loader's
# `row = 8 w + p` map over the FIRST m-tile's four warps, the exchange's [N_MT][4 warps] slot count, bar.sync 0 = the launch's threads),
# N_KEYS == 32 (one 32-bit mask word per ROW per sub-tile, 8 blocks per sub-tile), SMEM = 2 x STAGES x N_KEYS x 512 + N_MT x 8192 under the
# standard carveout, 1 <= SUB_TILES_PER_CTA <= 8 (the body is range_constexpr-unrolled over it), 1 <= S_Q <= 4 (the shared list's bound).
_LEAN_FLAVOR = "sm107 d256 sparse lean"
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_tile_rules import (  # noqa: E402 -- the lean host arithmetic's DSL-free home
    _LEAN_N_TAIL_MAX,
    LEAN_SPARSE_MAX_SEQLEN_Q,
    LEAN_SPARSE_MAX_SUB_TILES_PER_CTA,
    LEAN_SPARSE_N_KEYS,
    LEAN_SPARSE_SERVED_MAX_S_Q,
    LEAN_SPARSE_TOPK_MIN,
    lean_sparse_n_sub,
    lean_sparse_split,
)

LEAN_SPARSE_BLOCKS_PER_SUB = LEAN_SPARSE_N_KEYS // _SPARSE_BLOCK_SIZE
LEAN_SPARSE_THREADS_PER_MT = 128  # four warps per m16 tile: the d-slices of the split-D plan
LEAN_SPARSE_ROWS_PER_MT = 16  # the m16n8k16 M extent: one m-tile = 16 query rows (token-major: row R = s x G + g)
assert 1 <= LEAN_SPARSE_SERVED_MAX_S_Q <= LEAN_SPARSE_MAX_SEQLEN_Q  # the route never serves more rows than the kernel renders (the tripwire)
_LEAN_S_EXCHANGE_BYTES = 4 * 4 * 32 * 16  # sS[4 warps][4 n-blocks][32 lanes][16 B] = 8 KiB per m-tile


@dataclass(frozen=True)
class TemplateParamsLean(TemplateParams):
    """The lean tile's record: the shared ``TemplateParams`` plus the serving step's query rows per sequence.  A SEPARATE record on
    purpose (the ``TemplateParams2x2`` pattern): the shared record gains no field, so every other template's digest and compiled-plan
    cache entries are what they were; the adapter builds this record on the lean route only.  Frozen + hashable like its base: it is
    the template-module cache key and the source digest."""

    # S_q, the rows of the step per sequence (1..LEAN_SPARSE_MAX_SEQLEN_Q): a COMPILE-TIME axis of this kernel -- the m-tile count
    # N_MT = ceil(S_q x G / 16), the thread count 128 x N_MT and the S exchange's extent fold on it (the tcgen05 tile takes S_q at
    # runtime; this tile holds every row of the sequence in one CTA).  Plan-time only: the declared Q shape's S_q.
    decode_seqlen_q: int = 1
    # k, the 32-key sub-tiles per CTA (1..LEAN_SPARSE_MAX_SUB_TILES_PER_CTA): the resolved pin the adapter carries into the record, so an
    # explicit `sub_tiles_per_cta` survives template lowering (`ceil(n_sub / split)` is NOT invertible: top_k 32 gives split 2 for k 3 and
    # k 4) and two pins are two cache identities.  0 = unset (the doc-default record): the builder derives k from `split_kv` as before.
    sub_tiles_per_cta: int = 0


def lean_sparse_n_mt(seqlen_q: int, qh_per_kh: int, rows_per_mt: int = LEAN_SPARSE_ROWS_PER_MT) -> int:
    """The m16 tiles that hold the sequence's ``S_q x G`` query rows (token-major): 1 at S_q 1 / G 12, 2 at S_q 2, 3 at S_q 3..4 (G 12), 4 at
    S_q 4 / G 16, 1 at G 1 (four rows, one per token)."""
    return -(-(int(seqlen_q) * int(qh_per_kh)) // int(rows_per_mt))


def lean_sparse_smem_bytes(stages: int, n_mt: int = 1, n_keys: int = LEAN_SPARSE_N_KEYS, bpe_val: int = 2, d: int = 256) -> int:
    """sK[stages] + sV[stages] (n_keys rows x d x bpe each) + the S exchange (8 KiB per m-tile): 40,960 B at one stage, 73,728 B at two."""
    return 2 * int(stages) * int(n_keys) * int(d) * int(bpe_val) + int(n_mt) * _LEAN_S_EXCHANGE_BYTES


@dataclass(frozen=True)
class CfgD256SparseLean:
    """The lean decode tile's record; every field is set by ``make_cfg_d256_sparse_lean`` (the defaults are the 24/2, top_k 512, k 1,
    dense bf16 rendering so the dataclass is constructible in a doc / test context -- not a fallback)."""

    # --- tile geometry
    D: int = 256  # d_qk = d_v (exact)
    N_KEYS: int = LEAN_SPARSE_N_KEYS
    BLOCK_SIZE: int = _SPARSE_BLOCK_SIZE
    BLOCKS_PER_SUB: int = LEAN_SPARSE_BLOCKS_PER_SUB
    BLOCK_TOPK: int = 512
    N_SUB: int = 65  # lean_sparse_n_sub(BLOCK_TOPK)
    SUB_TILES_PER_CTA: int = 1  # k: the chunk size (a template axis; range_constexpr-unrolled)
    SPLIT: int = 65  # ceil(N_SUB / k): the grid's x extent = the combine's partial count
    STAGES: int = 1  # min(2, k): the K / V ring depth
    # --- the CTA
    N_MT: int = 1  # m16 tiles: ceil(S_q x G / 16); 1 at S_q 1 / G 12, 3 at S_q 4 / G 12, 4 at S_q 4 / G 16
    S_Q: int = 1  # the step's rows per sequence (TemplateParamsLean.decode_seqlen_q), 1..LEAN_SPARSE_MAX_SEQLEN_Q
    WARPS: int = 4  # 4 x N_MT
    THREADS: int = LEAN_SPARSE_THREADS_PER_MT  # 128 x N_MT
    QH_PER_KH: int = 12
    # --- dtypes
    DTYPE_QKV: int = _DTYPE_BF16
    BPE: int = 2
    # --- SMEM (bytes; declaration order sK | sV | sS, 128-B aligned rows, no descriptor)
    SMEM_K_BYTES: int = 16384
    SMEM_V_BYTES: int = 16384
    SMEM_S_BYTES: int = _LEAN_S_EXCHANGE_BYTES
    SMEM_TOTAL_BYTES: int = 40960
    SMEM_CARVEOUT_BYTES: int = SMEM_STANDARD_CARVEOUT_BYTES
    # --- arms (the decode form is forced; the paged read is the one optional arm)
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0
    SEQ_KV_LENS_PRESENT: int = 1
    BOTTOM_RIGHT: int = 1
    LIST_PER_SEQUENCE: int = 1
    INCLUDE_OPEN_BLOCK: int = 1


def _validate_cfg_d256_sparse_lean(cfg: CfgD256SparseLean, flavor: str = _LEAN_FLAVOR) -> None:
    """Every geometry claim of the lean kernel's header tables as a raising predicate naming what the failure looks like."""
    _check(
        [
            (
                cfg.DTYPE_QKV in (_DTYPE_BF16, _DTYPE_FP16) and cfg.BPE == 2,
                f"{flavor}: f16/bf16 inputs only (got DTYPE_QKV={cfg.DTYPE_QKV}); mma.sync m16n8k16 bf16/f16",
            ),
            (cfg.D == 256, f"{flavor}: d_qk = d_v = 256 only (the four 64-wide d-slices, the 512-B rows); got {cfg.D}"),
            (
                cfg.N_KEYS == LEAN_SPARSE_N_KEYS and cfg.BLOCK_SIZE == _SPARSE_BLOCK_SIZE and cfg.BLOCKS_PER_SUB == LEAN_SPARSE_BLOCKS_PER_SUB,
                f"{flavor}: a sub-tile is 32 keys = 8 four-token blocks (ONE 32-bit valid word, the 8 w + p loader row map); got N_KEYS={cfg.N_KEYS}",
            ),
            (
                LEAN_SPARSE_TOPK_MIN <= cfg.BLOCK_TOPK <= _SPARSE_TOPK_MAX and cfg.BLOCK_TOPK % 4 == 0,
                f"{flavor}: top_k must be a multiple of 4 in [{LEAN_SPARSE_TOPK_MIN}, {_SPARSE_TOPK_MAX}] (got {cfg.BLOCK_TOPK}): 16-B aligned ids quads; below 8 "
                f"the list is one sub-tile and the tcgen05 tile serves it",
            ),
            (
                cfg.N_SUB == lean_sparse_n_sub(cfg.BLOCK_TOPK),
                f"{flavor}: N_SUB must be ceil((top_k + 2) x 4 / 32) = {lean_sparse_n_sub(cfg.BLOCK_TOPK)} (got {cfg.N_SUB})",
            ),
            (
                1 <= cfg.SUB_TILES_PER_CTA <= min(LEAN_SPARSE_MAX_SUB_TILES_PER_CTA, cfg.N_SUB),
                f"{flavor}: SUB_TILES_PER_CTA (k) must be in [1, {min(LEAN_SPARSE_MAX_SUB_TILES_PER_CTA, cfg.N_SUB)}] (got {cfg.SUB_TILES_PER_CTA}): the sub-tile loop is "
                f"unrolled at trace time and a chunk never exceeds the list",
            ),
            (
                cfg.SPLIT == -(-cfg.N_SUB // cfg.SUB_TILES_PER_CTA),
                f"{flavor}: SPLIT must be ceil(N_SUB / k) = {-(-cfg.N_SUB // cfg.SUB_TILES_PER_CTA)} (got {cfg.SPLIT})",
            ),
            (cfg.STAGES == min(2, cfg.SUB_TILES_PER_CTA), f"{flavor}: STAGES must be min(2, k) (got {cfg.STAGES}): the body's wait_group immediates assume it"),
            (
                1 <= cfg.S_Q <= LEAN_SPARSE_MAX_SEQLEN_Q,
                f"{flavor}: S_q must be in 1..{LEAN_SPARSE_MAX_SEQLEN_Q} (got {cfg.S_Q}): the serving step's rows share ONE list (two tail blocks at most); "
                f"a longer step is the tcgen05 tile's",
            ),
            (
                1 <= cfg.QH_PER_KH <= 16,
                f"{flavor}: the token's query heads must fit the m16 tile (got qh_per_kh={cfg.QH_PER_KH}); above 16 the adapter declines",
            ),
            (
                cfg.N_MT == lean_sparse_n_mt(cfg.S_Q, cfg.QH_PER_KH),
                f"{flavor}: N_MT must be ceil(S_q x G / 16) = {lean_sparse_n_mt(cfg.S_Q, cfg.QH_PER_KH)} (got {cfg.N_MT}): the m16 tiles that hold the S_q x G rows",
            ),
            (
                cfg.WARPS == 4 * cfg.N_MT and cfg.THREADS == 32 * cfg.WARPS == LEAN_SPARSE_THREADS_PER_MT * cfg.N_MT,
                f"{flavor}: THREADS must be 128 x N_MT = {LEAN_SPARSE_THREADS_PER_MT * cfg.N_MT} (got {cfg.THREADS}): the loader's row map `8 w + p` over the first "
                f"m-tile's warps and the exchange's [N_MT][4 warps] slot count assume four warps per m-tile, and bar.sync 0 counts the launch's threads",
            ),
            (
                cfg.SMEM_K_BYTES == cfg.SMEM_V_BYTES == cfg.STAGES * cfg.N_KEYS * cfg.D * cfg.BPE,
                f"{flavor}: sK / sV are STAGES x N_KEYS x 512 B each (got {cfg.SMEM_K_BYTES} / {cfg.SMEM_V_BYTES})",
            ),
            (cfg.SMEM_S_BYTES == cfg.N_MT * _LEAN_S_EXCHANGE_BYTES, f"{flavor}: the S exchange is 8 KiB per m-tile (got {cfg.SMEM_S_BYTES})"),
            (
                cfg.SMEM_TOTAL_BYTES
                == cfg.SMEM_K_BYTES + cfg.SMEM_V_BYTES + cfg.SMEM_S_BYTES
                == lean_sparse_smem_bytes(cfg.STAGES, cfg.N_MT, cfg.N_KEYS, cfg.BPE, cfg.D),
                f"{flavor}: SMEM_TOTAL_BYTES ({cfg.SMEM_TOTAL_BYTES}) disagrees with sK + sV + sS = {lean_sparse_smem_bytes(cfg.STAGES, cfg.N_MT, cfg.N_KEYS, cfg.BPE, cfg.D)}; "
                f"an overflow does NOT fail the launch, it clobbers the last buffer",
            ),
            (
                cfg.SMEM_TOTAL_BYTES <= SMEM_STANDARD_CARVEOUT_BYTES and cfg.SMEM_CARVEOUT_BYTES == SMEM_STANDARD_CARVEOUT_BYTES,
                f"{flavor}: SMEM {cfg.SMEM_TOTAL_BYTES} B must fit the standard 227 KiB carveout (L1 kept; 3 CTAs per SM at 72 KiB) -- the oversized mode is a measured lever, not a default",
            ),
            (
                not cfg.PAGED_KV or (cfg.PAGE_SIZE > 0 and cfg.PAGE_SIZE % cfg.BLOCK_SIZE == 0),
                f"{flavor}: page_size must be a positive multiple of BLOCK_SIZE = {cfg.BLOCK_SIZE} (got {cfg.PAGE_SIZE}) or a block straddles two pages",
            ),
            (cfg.PAGED_KV == 1 or cfg.PAGE_SIZE == 0, f"{flavor}: PAGE_SIZE is 0 exactly when the kernel is not paged"),
            (
                cfg.SEQ_KV_LENS_PRESENT == 1 and cfg.BOTTOM_RIGHT == 1 and cfg.LIST_PER_SEQUENCE == 1 and cfg.INCLUDE_OPEN_BLOCK == 1,
                f"{flavor}: the DECODE form is the only body (per-batch KV lengths, bottom-right rows, one list per sequence, the open tail block): "
                f"got SEQ_KV_LENS_PRESENT={cfg.SEQ_KV_LENS_PRESENT} BOTTOM_RIGHT={cfg.BOTTOM_RIGHT} LIST_PER_SEQUENCE={cfg.LIST_PER_SEQUENCE} INCLUDE_OPEN_BLOCK={cfg.INCLUDE_OPEN_BLOCK}",
            ),
        ]
    )


def make_cfg_d256_sparse_lean(params: TemplateParams, *, n_keys: int = LEAN_SPARSE_N_KEYS, sub_tiles_per_cta: Optional[int] = None) -> CfgD256SparseLean:
    """Config for sm107/sparse_lean_d256_f16.py from the adapter's record -- ``TemplateParamsLean`` (its ``decode_seqlen_q`` = S_q in
    1..4 sizes the CTA: N_MT = ceil(S_q x G / 16) m-tiles of 128 threads) or the shared ``TemplateParams`` (read as S_q 1).  The decode
    form is REQUIRED (``bottom_right``, ``qsa_list_per_sequence``, ``seq_kv_lens_present``); ``split_kv`` is the LEAN split ``ceil(n_sub /
    k)`` and selects ``k`` (a ``split_kv`` that is not a lean split of this top_k is declined by name; ``split_kv`` 1 / unset takes
    ``sub_tiles_per_cta``, default 1); ``n_keys`` is 32 only (N 64 is a measured lever, not wired).  Every other arm is a typed decline
    naming the feature.  Backstop only: every rejection here must also be a decline of the adapter's claims record."""
    flavor = _LEAN_FLAVOR
    if not params.qsa_block_topk:
        raise ValueError(f"{flavor}: not an index-list record (qsa_block_topk=0); the dense templates serve it")
    s_q = int(getattr(params, "decode_seqlen_q", 1))
    if not 1 <= s_q <= LEAN_SPARSE_MAX_SEQLEN_Q:
        raise ValueError(
            f"{flavor}: decode_seqlen_q (S_q) must be in 1..{LEAN_SPARSE_MAX_SEQLEN_Q} (got {s_q}): the serving step's rows share ONE list (two tail blocks at "
            f"most); a longer step is the tcgen05 tile's"
        )
    if params.qsa_block_size != _SPARSE_BLOCK_SIZE:
        raise ValueError(f"{flavor}: qsa_block_size must be {_SPARSE_BLOCK_SIZE} (four-token blocks); got {params.qsa_block_size}")
    if n_keys != LEAN_SPARSE_N_KEYS:
        raise ValueError(f"{flavor}: n_keys must be {LEAN_SPARSE_N_KEYS} (one 32-bit valid word per sub-tile); N 64 is not wired")
    if params.dtype_qkv not in (_DTYPE_BF16, _DTYPE_FP16):
        raise ValueError(f"{flavor}: f16/bf16 inputs only (got dtype_qkv={params.dtype_qkv}); the e4m3 pools arm is not wired")
    if resolve_dtype_o(params) != params.dtype_qkv:
        raise ValueError(f"{flavor}: half input requires dtype_o == dtype_qkv; got dtype_o={resolve_dtype_o(params)}")
    if not (params.bottom_right and params.qsa_list_per_sequence and params.seq_kv_lens_present):
        raise ValueError(
            f"{flavor}: the DECODE form only -- bottom_right, qsa_list_per_sequence and seq_kv_lens_present must all be set (got "
            f"{params.bottom_right} / {params.qsa_list_per_sequence} / {params.seq_kv_lens_present}); the prefill forms are the tcgen05 tile's"
        )
    if not params.qsa_include_open_block:
        raise ValueError(f"{flavor}: the pure-list arm (qsa_include_open_block=False) is not wired; the open tail block is always attended")
    if params.thd_varlen:
        raise ValueError(f"{flavor}: thd_varlen (packed sequences) is not wired; the tcgen05 tile serves THD")
    if params.epilogue_gate:
        raise ValueError(f"{flavor}: epilogue_gate is not wired -- under the lean split the gate rides the combine")
    if params.has_sink:
        raise ValueError(f"{flavor}: no sink")
    if params.stats_log2:
        raise ValueError(f"{flavor}: stats_log2 is not wired; the partial LSE is natural-log (the combine's contract)")
    if params.seq_q_lens_present:
        raise ValueError(f"{flavor}: seq_q_lens_present (the padded-Q trim) is not wired; every sequence has the record's S_q rows (decode_seqlen_q)")
    if params.window_left is not None or params.window_right is not None:
        raise ValueError(f"{flavor}: the selection IS the mask; a diagonal band (window_left / window_right) is not part of this kernel")
    if not params.pack_gqa:
        raise ValueError(f"{flavor}: the work item is the token's packed query-head group (pack_gqa must be True)")
    if params.cta_mma != 1:
        raise ValueError(f"{flavor}: one CTA per work item (cta_mma must be 1; got {params.cta_mma})")
    if not 1 <= params.qh_per_kh <= 16:
        raise ValueError(f"{flavor}: qh_per_kh must be in 1..16 (the token's query heads fill the m16 tile); got {params.qh_per_kh}")
    if (
        params.decode_q_tile
        or params.decode_tile
        or params.ragged_q
        or params.mma_2x2
        or params.pv_bf16
        or params.softmax_f16
        or params.softmax_scale_prefolded
        or params.exp2_fma_split
        or params.negate_scores
        or params.kv_tail_mask
        or params.thd_batch_one
        or params.lpt_head_group != 1
        or params.lpt_q_tiles
        or params.lpt_l2_size_mib
        or params.qsa_softmax_groups
    ):
        raise ValueError(
            f"{flavor}: decode_q_tile / decode_tile / ragged_q / mma_2x2 / pv_bf16 / softmax_f16 / softmax_scale_prefolded / exp2_fma_split / "
            f"negate_scores / kv_tail_mask / thd_batch_one / lpt_head_group / lpt_q_tiles / lpt_l2_size_mib / qsa_softmax_groups are other kernels' specializations"
        )
    if params.sched_policy not in (None, SCHED_NATURAL):
        raise ValueError(f"{flavor}: sched_policy must be NATURAL or None (the grid is the work list; got {params.sched_policy})")
    if not LEAN_SPARSE_TOPK_MIN <= params.qsa_block_topk <= _SPARSE_TOPK_MAX or params.qsa_block_topk % 4 != 0:
        raise ValueError(
            f"{flavor}: top_k must be a multiple of 4 in [{LEAN_SPARSE_TOPK_MIN}, {_SPARSE_TOPK_MAX}] (got {params.qsa_block_topk}); below 8 the list is one sub-tile and "
            f"the tcgen05 tile serves it"
        )
    if params.paged_kv and (params.page_size <= 0 or params.page_size % _SPARSE_BLOCK_SIZE != 0):
        raise ValueError(f"{flavor}: page_size must be a positive multiple of {_SPARSE_BLOCK_SIZE} (got {params.page_size}) or a block straddles two pages")
    if not params.paged_kv and params.page_size:
        raise ValueError(f"{flavor}: page_size requires paged_kv=True")
    n_sub = lean_sparse_n_sub(params.qsa_block_topk, _SPARSE_BLOCK_SIZE, n_keys)
    split_req = int(params.split_kv or 1)
    k_rec = int(getattr(params, "sub_tiles_per_cta", 0) or 0)  # the record's k (the adapter's resolved pin); 0 = unset -> derived from the split
    if k_rec > 0:
        k = k_rec
        if split_req > 1 and lean_sparse_split(params.qsa_block_topk, k) != split_req:
            raise ValueError(
                f"{flavor}: the record's sub_tiles_per_cta={k} and split_kv={split_req} disagree: the lean split of k {k} at top_k {params.qsa_block_topk} is "
                f"ceil({n_sub} / {k}) = {lean_sparse_split(params.qsa_block_topk, k)}"
            )
        if sub_tiles_per_cta is not None and int(sub_tiles_per_cta) != k:
            raise ValueError(f"{flavor}: the record carries sub_tiles_per_cta={k}, but sub_tiles_per_cta={sub_tiles_per_cta} was given")
    elif split_req > 1:
        k = -(-n_sub // split_req)
        if sub_tiles_per_cta is not None and int(sub_tiles_per_cta) != k:
            raise ValueError(
                f"{flavor}: split_kv={split_req} selects k = ceil({n_sub} / {split_req}) = {k} sub-tiles per CTA, but sub_tiles_per_cta={sub_tiles_per_cta} was given"
            )
    else:
        k = 1 if sub_tiles_per_cta is None else int(sub_tiles_per_cta)
    if not 1 <= k <= min(LEAN_SPARSE_MAX_SUB_TILES_PER_CTA, n_sub):
        raise ValueError(
            f"{flavor}: sub_tiles_per_cta (k) must be in [1, {min(LEAN_SPARSE_MAX_SUB_TILES_PER_CTA, n_sub)}] at top_k {params.qsa_block_topk} (n_sub {n_sub}); got {k}"
        )
    split = lean_sparse_split(params.qsa_block_topk, k)
    if split_req > 1 and split != split_req:
        raise ValueError(
            f"{flavor}: split_kv={split_req} is not a lean split of top_k {params.qsa_block_topk}: the lean tile's split is ceil(n_sub / k) = ceil({n_sub} / k) "
            f"for k in 1..{min(LEAN_SPARSE_MAX_SUB_TILES_PER_CTA, n_sub)} ({sorted({lean_sparse_split(params.qsa_block_topk, kk) for kk in range(1, min(LEAN_SPARSE_MAX_SUB_TILES_PER_CTA, n_sub) + 1)}, reverse=True)})"
        )
    stages = min(2, k)
    b = bpe(params.dtype_qkv)
    n_mt = lean_sparse_n_mt(s_q, params.qh_per_kh)
    cfg = CfgD256SparseLean(
        D=256,
        N_KEYS=n_keys,
        BLOCK_SIZE=_SPARSE_BLOCK_SIZE,
        BLOCKS_PER_SUB=n_keys // _SPARSE_BLOCK_SIZE,
        BLOCK_TOPK=int(params.qsa_block_topk),
        N_SUB=n_sub,
        SUB_TILES_PER_CTA=k,
        SPLIT=split,
        STAGES=stages,
        N_MT=n_mt,
        S_Q=s_q,
        WARPS=4 * n_mt,
        THREADS=LEAN_SPARSE_THREADS_PER_MT * n_mt,
        QH_PER_KH=int(params.qh_per_kh),
        DTYPE_QKV=int(params.dtype_qkv),
        BPE=b,
        SMEM_K_BYTES=stages * n_keys * 256 * b,
        SMEM_V_BYTES=stages * n_keys * 256 * b,
        SMEM_S_BYTES=n_mt * _LEAN_S_EXCHANGE_BYTES,
        SMEM_TOTAL_BYTES=lean_sparse_smem_bytes(stages, n_mt, n_keys, b, 256),
        SMEM_CARVEOUT_BYTES=SMEM_STANDARD_CARVEOUT_BYTES,
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size) if params.paged_kv else 0,
        SEQ_KV_LENS_PRESENT=1,
        BOTTOM_RIGHT=1,
        LIST_PER_SEQUENCE=1,
        INCLUDE_OPEN_BLOCK=1,
    )
    _validate_cfg_d256_sparse_lean(cfg, flavor)
    return cfg
