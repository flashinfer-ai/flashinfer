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
from typing import Tuple

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
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import D256_DECODE_MAX_Q_ROWS, decode_d256_q_tile as _decode_d256_q_tile_rule

__all__ = [
    "TemplateParams",
    "resolve_dtype_o",
    "TmaIters",
    "CfgD128",
    "CfgD192",
    "CfgD256",
    "CfgD512",
    "CfgD512X2",
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
    "D256_DECODE_ROUTED_MAX_Q_ROWS",
    "D256_DECODE_ROUTED_MAX_TOKEN_UNITS",
    "decode_d256_q_tile",
    "SMEM_CAP_BYTES",
    "SMEM_USABLE_BYTES",
    "SM107_FP8_THD_SHAPES",
    "SM107_F16_THD_SHAPES",
    "SM107_EPILOGUE_GATE_SHAPES",
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
# the contract (frost_dev/port_thd_contract.py); they previously called the
# setup kernel with the pre-upstream 7-arg signature and allocated 3B+2 where
# the shared decode reads 4B+4.  Confirmed on w2u1g-lc-0030: 43 passed / 0
# failed across the per-tensor FP8 THD suite.
SM107_FP8_THD_SHAPES = frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})

# f16/bf16 flavor names whose kernel body HAS been ported to the FROST
# setup-kernel contract (the 14-arg build_thd_meta_o_descs_kernel + the 4B+4
# metadata the shared decode reads).  Keyed by the `flavor` string
# `_validate_params` already receives, so adding a ported flavor is one entry.
_F16_THD_FLAVORS = frozenset({"sm107 d128", "sm107 d192xd128", "sm107 d256", "sm107 d512"})

# Paged KV (TemplateParams.paged_kv / page_size) on the Rubin line: wired through the K/V TMA-LDG sites of the MXFP8
# d128 / d256 siblings only -- the SM100 PAGED_KV specialization ported (page id = TMA batch coordinate of the K/V and
# SF descriptors, tile-in-page = SF tile coordinate).  engines' paged_d_shapes and the adapter's Rubin clause name the
# same two shapes; half paged graphs never load an sm107 file (api_dsl._load_sm100_kernel_module routes them to the
# SM100 bodies).  _validate_params declines every other flavor, and each unwired MXFP8 kernel file refuses paged_kv at
# module scope as well.  A page holds whole 128-row F8_128x4 SF atom sets, so page_size is a multiple of the KV tile.
# Two cc 10.7 HALF kernels read pools natively and validate their OWN records (neither passes through _validate_params,
# so this set does not name them): sm107/decode_d256_f16.py (make_cfg_d256_decode -- the decode-shaped dense-Q d256
# route, the SM100 decode record re-validated for Rubin) and sm107/sparse_d256_f16.py (make_cfg_d256_sparse -- the
# index-list core's gather warps read K / V from the pools through the (B, max_pages) table; page_size a multiple of
# the 4-token block).
_PAGED_KV_FLAVORS_SM107 = frozenset({"sm107 d128 mxfp8", "sm107 d256 mxfp8"})

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
    {"sm107 d128", "sm107 d192xd128", "sm107 d256", "sm107 d512", "sm107 d128 mxfp8", "sm107 d192xd128 mxfp8", "sm107 d256 mxfp8", "sm107 d512 mxfp8"}
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


def rescale_threshold(dtype_qkv: int) -> float:
    """4.0 for FP8/MXFP8 (low precision, rescale aggressively); 8.0 for
    BF16/FP16 (defer rescale to amortize the alpha == 1 fast path)."""
    return 4.0 if dtype_qkv <= _DTYPE_E5M2 else 8.0


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


# 14-bit ``start_address`` of a version-0 tcgen05 SMEM descriptor: every
# descriptor-fed slab (MMA operand or UTCCP scale-factor source) of a module
# that pins DESC_VERSION=0 must START under this, or it wraps to offset 0 and
# the MMA/UTCCP reads the bottom of SMEM (rules/mma-tma-matrix.md S6).
TCGEN05_V0_ADDR_LIMIT = 256 * 1024

_SMEM_SLAB_ALIGN = 1024  # every kernel slab is a 1024-B-aligned cutlass.Array
_MX_BLOCK = 32  # MXFP8: 32 elements share one E8M0 scale byte


def _align_slab(nbytes: int) -> int:
    return -(-nbytes // _SMEM_SLAB_ALIGN) * _SMEM_SLAB_ALIGN


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
    partial-output stores: half and per-tensor FP8 D128 and D192/V128. The engine
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
                f"sm107/prefill_d128_mxfp8.py and sm107/prefill_d256_mxfp8.py; served: {sorted(_PAGED_KV_FLAVORS_SM107)})"
            )
        if not k.seq_kv_lens_present:
            raise ValueError(f"{flavor}: paged_kv requires seq_kv_lens_present (the per-batch KV length bounds the block-table walk)")
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
class _CfgSm107:
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
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    BOTTOM_RIGHT: int = 0
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    PACK_GQA: int = 0
    QH_PER_KH: int = 1
    SPLIT_KV: int = 1
    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0
    THD_VARLEN: int = 0
    # Paged KV cache (TemplateParams.paged_kv / page_size); only the d128 / d256 MXFP8 factories set them.
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
# Shared validators
# ---------------------------------------------------------------------------


def _check(preds) -> None:
    for ok, msg in preds:
        if not ok:
            raise ValueError(msg)


def _validate_regs(cfg, flavor: str) -> None:
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


def _validate_warp_layout(cfg, flavor: str, *, expect_total: int, expect_arrivers: int) -> None:
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


def _validate_smem(cfg, flavor: str, *, qo_alias: bool, sf_bytes: int = 0) -> None:
    """``sf_bytes``: the block-scale kernel's scale-factor slabs (0 on the f16 /
    per-tensor flavors); named in the message so a reader sees every term."""
    gate_bpe = cfg.GATE_BPE if cfg.EPILOGUE_GATE else 0
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
        + _SMEM_FIXED_OVERHEAD
    )
    if used > SMEM_USABLE_BYTES:
        gate_term = f" + {gate_kib} KiB GATE staging tile" if gate_bpe else ""
        sf_term = (
            f" + {sf_bytes // 1024} KiB MXFP8 scale-factor slabs (Q 1 | K {cfg.STAGES_KV}x1 | P 1 (512 B padded) | V {cfg.STAGES_KV}x1)" if sf_bytes else ""
        )
        raise ValueError(
            f"{flavor}: SMEM {used // 1024} KiB (Q/K/V/O + {_SMEM_FIXED_OVERHEAD // 1024} KiB fixed{gate_term}{sf_term}) exceeds the "
            f"{SMEM_USABLE_BYTES // 1024} KiB usable Rubin carveout at STAGES_KV={cfg.STAGES_KV}, CTA_MMA={cfg.CTA_MMA}"
            f"{', EPILOGUE_GATE=1' if gate_bpe else ''}. "
            f"Overflowing it does NOT fail the launch -- it clobbers the last buffer allocated (measured: O came back "
            f"50 % zeros with an exact LSE), so this must raise here"
        )


def _validate_swizzles(cfg, flavor: str) -> None:
    _check(
        [
            (cfg.Q_SWZ_BYTES in (64, 128), f"{flavor}: Q swizzle must be 64/128 B"),
            (cfg.K_SWZ_BYTES in (64, 128), f"{flavor}: K swizzle must be 64/128 B"),
            (cfg.V_SWZ_BYTES in (32, 64, 128), f"{flavor}: V swizzle must be 32/64/128 B"),
            (cfg.O_SWZ_BYTES in (64, 128), f"{flavor}: O swizzle must be 64/128 B"),
        ]
    )


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


# ---------------------------------------------------------------------------
# d128 / d192 -- the classic 2-warpgroup pipeline (16 warps)
# ---------------------------------------------------------------------------
#
# Warp map: 8 softmax (2 warpgroups) + 4 correction + MMA + TMALDG + TMASTG +
# scheduler.  15 of the 16 warps arrive on the scheduler ring (all but the
# scheduler warp itself).


_D128_TOTAL_WARPS = 16
_D128_READ_TILE_ARRIVERS = 15


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
    return 2 * cta_mma


def _make_cfg_d128_family(params: TemplateParams, *, flavor: str, tile_k: int, tile_o: int, mxfp8: bool):
    # Half and per-tensor FP8 D128/D192 use the shared split protocol.
    # MXFP8 has no split loop yet.
    split_wired = not mxfp8 and tile_k in (128, 192) and tile_o == 128
    # The block-scaled O epilogue lives in the two d128 kernels (per-tensor fp8 and
    # mxfp8); the d192xd128 siblings share this config family but not the epilogue.
    block_scaled_o_wired = tile_k == 128 and tile_o == 128
    _validate_params(flavor, params, split_wired=split_wired, block_scaled_o_wired=block_scaled_o_wired)
    # softmax_f16 / softmax_scale_prefolded: per-flavor backstops in _validate_params (SM107_*_FLAVORS).
    cta_mma = params.cta_mma
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    o_div = o_pack_div(dtype_o)
    tile_n = 128
    stages_kv = _stages_kv_d128(params.dtype_qkv, cta_mma, mxfp8=mxfp8, tile_k=tile_k)
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)
    cls = CfgD192 if tile_k == 192 else CfgD128

    cfg = cls(
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
        QO_ALIAS=0,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_o, cta_mma, b),
        O_SWZ_BYTES=o_swz_bytes(tile_o, b_o, o_div),
        O_BLOCK_SCALE=_O_BLOCK_SCALE_BY_DTYPE.get(dtype_o, 0),
        O_PACK_DIV=o_div,
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=2,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        SOFTMAX_WARPGROUPS=2,
        CORRECTION_WARPS=4,
        SOFTMAX_REGS=192,
        CORRECTION_REGS=88,
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
        TOTAL_WARPS=_D128_TOTAL_WARPS,
        THREADS_PER_CTA=_D128_TOTAL_WARPS * 32,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=4,
        CORR_WARP_BASE=8,
        MMA_WARP_ID=12,
        TMALDG_WARP_ID=13,
        TMASTG_WARP_ID=14,
        SCHED_WARP_ID=15,
        READ_TILE_ARRIVERS=_D128_READ_TILE_ARRIVERS,
    )

    _validate_regs(cfg, flavor)
    _validate_warp_layout(cfg, flavor, expect_total=_D128_TOTAL_WARPS, expect_arrivers=_D128_READ_TILE_ARRIVERS)
    _validate_dtype_k_step(cfg, flavor)
    _validate_swizzles(cfg, flavor)
    _validate_masks(cfg, flavor)
    _validate_smem(cfg, flavor, qo_alias=bool(cfg.QO_ALIAS))
    return cfg, _tma_iters(cfg)


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
    _validate_params(flavor, params)
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
    _validate_params("sm107 d512", params)
    # The 2x2 record's own domain (d512 only, half inputs, unpaged, whole-group PackGQA) is stated ONCE, in config_sm100;
    # the pre-folded scale is this twin's own arm (the cc 10.0 line keeps declining it).
    _validate_params_sm100("d512", params, scale_prefolded_wired=True)
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
        PACK_G=_pack_g(params, CfgD512X2.TILE_M, partial=False),
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
# The decode tile's body is arch-agnostic (cta_group::1, 6 warps at N_Q = 16, f16/bf16
# TILE_K_HW = 16 on both arch lines, 64 TMEM columns) and its record is the SM100
# CfgD256Decode built by config_sm100.make_cfg_d256_decode, re-validated here against the
# two Rubin facts a port can get wrong silently:
#   * the SMEM budget is checked against SMEM_USABLE_BYTES (320 KiB), never the 327 KiB
#     capacity (the d192 f16 50 %-zeros lesson above).  The tile's slabs -- Q^T, the
#     3-slot 64 KiB K/V ring, two P^T tiles, the reduction scratch -- sum to ~209 KiB at
#     N_Q = 16 and ~226 KiB at N_Q = 32, so BOTH widths fit the STANDARD 227 KiB
#     carveout: no ALLOW_OVERSIZED_SHARED_MEMORY launch, no L1 shrink, on this tile.
#   * DESC_VERSION is DERIVED from the layout: every MMA-operand slab (Q^T, the K/V ring,
#     P^T) must START under TCGEN05_V0_ADDR_LIMIT for the version-0 tcgen05 descriptor the
#     kernel pins; the highest start is P^T at Q + ring = 200 / 208 KiB, so the answer is 0
#     at every admitted record -- asserted, not assumed (the 2026-09-04 silent-zero class).

_D256_DECODE_FLAVOR = "sm107 d256 decode"

# The Rubin ROUTE onto the decode tile (the cc 10.7 twin of config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS /
# decode_d256_q_tile; read by SdpaFwdDslSm100._decode_q_tile_for, engines.d256_decode_tile_selected and the
# heuristics' decode geometry through engines.decode_d256_q_tile_for_row -- keep the three in lockstep):
#   * the 32-column tile IS routed here -- rows in (16, 32] (24/2 at S_q = 2 = 24 live rows, 32/2 at S_q = 2,
#     an MHA step of 17-32 tokens) ride one unit of it;
#   * a PACKED group may be cut into up to two TOKEN UNITS of the 32-column tile (the kernel's Q_TOKEN_UNITS
#     axis: S_q = 4 at 24/2 = two units of two tokens x 12 heads, each streaming the KV range once -- 2x the
#     KV bytes of a single stream), the MTP step of the 24/2 and 32/2 geometries (S_q <= 4).
# MEASURED on cc 10.7 (212 SMs) at the 24/2 geometry ONLY -- G = 12: the one-unit 32-column form at S_q = 2 and
# the two-unit form at S_q = 3 / 4, against the prefill body and against the 16-column tile in single-token
# units, B in {1, 4, 32}, S_kv 2K..128K.  The envelope is WIDER than that: an unpacked / MHA step of 17-32
# tokens and a packed group with 16 < S_q x G <= 32 at G != 12 ride one unit of the 32-column tile, and a packed
# group with ceil(S_q / (32 // G)) = 2 at G in {2, 4, 6, 8, 16, 32} two units of it -- all UNMEASURED on a perf
# GPU, routed by the same rule because each is the same tile with the same stream count per (head group, batch,
# split) as the measured form (the byte model does not see G), and each class has a GPU correctness cell in
# test_sdpa_fwd_decode_d256_sm107.py (the rule, the forms' agreement, the two-unit forms at G = 8 and at G = 6
# with tail rows, an MHA step of 24 tokens).  A perf cell per class is owed before any of them is quoted.  Past
# the envelope (a third token unit, an unpacked step above 32 tokens, a group wider than the tile) the d256
# prefill kernel serves the graph as before.
D256_DECODE_ROUTED_MAX_Q_ROWS = D256_DECODE_MAX_Q_ROWS
D256_DECODE_ROUTED_MAX_TOKEN_UNITS = 2


def decode_d256_q_tile(s_q: int, pack_g: int) -> int:
    """The Rubin decode tile's N extent for ``s_q`` tokens packed ``pack_g`` heads per token
    (1 = unpacked), or 0 when the d256 prefill kernel serves the graph: the shared rule
    (config_sm100.decode_d256_q_tile) with the Rubin constants above -- the 32-column tile
    routed, two token units for a packed group, one for an unpacked / MHA step."""
    units = D256_DECODE_ROUTED_MAX_TOKEN_UNITS if int(pack_g) > 1 else 1
    return _decode_d256_q_tile_rule(s_q, pack_g, D256_DECODE_ROUTED_MAX_Q_ROWS, units)


_D256_DECODE_RED_SLOTS = 3  # the kernel's cross-warp reduction scratch: [2 tile parities + 1 epilogue][softmax warp][16 columns]
_D256_DECODE_COLS = 16  # S^T / O^T columns one 4-warp softmax group owns
_D256_DECODE_BARRIER_WORDS = 1 + 2 * 3 + 2 * 4 + 1  # q_full, kv_full/empty x STAGES (3), s_full/s_empty/p_full/bmm2_done x 2, tmem_dealloc -- Int64 each


def d256_decode_smem_layout(cfg: CfgD256Decode) -> dict:
    """Byte offsets of the decode tile's SMEM slabs in the kernel's DECLARATION order
    (``sQ_raw | sKV_raw | sP_raw | red_smem | tmem_ptr | mbarriers``), every slab
    1024-aligned like the kernel's ``cutlass.Array(..., alignment=1024)`` (the small
    trailing arrays at their 16-B alignment), plus the total and the standard-carveout
    verdict.  The MMA-operand slabs are the first three; ``p_start`` is the highest
    operand start and is what the descriptor-version derivation reads."""
    q_bytes = cfg.N_Q * cfg.TILE_K * cfg.BPE
    kv_bytes = cfg.STAGES_KV * cfg.TILE_N * cfg.TILE_K * cfg.BPE
    p_bytes = 2 * (cfg.TILE_N * cfg.N_Q // 2) * 4  # two P^T tiles of TILE_N key rows x N_Q/2 packed-half Int32 words
    red_bytes = _D256_DECODE_RED_SLOTS * cfg.SOFTMAX_WARPS * _D256_DECODE_COLS * 4
    q_start = 0
    kv_start = _align_slab(q_start + q_bytes)
    p_start = _align_slab(kv_start + kv_bytes)
    red_start = _align_slab(p_start + p_bytes)
    tmem_ptr_start = -(-(red_start + red_bytes) // 16) * 16
    bars_start = -(-(tmem_ptr_start + 4) // 16) * 16
    total = bars_start + 8 * _D256_DECODE_BARRIER_WORDS
    return {
        "q_start": q_start,
        "kv_start": kv_start,
        "p_start": p_start,
        "red_start": red_start,
        "tmem_ptr_start": tmem_ptr_start,
        "bars_start": bars_start,
        "total": total,
        "fits_standard_carveout": total <= 227 * 1024,
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
                smem["fits_standard_carveout"],
                f"{flavor}: SMEM {smem['total']} B no longer fits the standard 227 KiB per-CTA carveout at N_Q={cfg.N_Q}, "
                f"STAGES_KV={cfg.STAGES_KV}; this tile is launched WITHOUT the oversized-SMEM mode (no L1 shrink), so a record past "
                f"227 KiB needs the launcher's ALLOW_OVERSIZED_SHARED_MEMORY arm before it can be admitted",
            ),
            (
                smem["p_start"] < TCGEN05_V0_ADDR_LIMIT,
                f"{flavor}: the P^T operand slab starts at {smem['p_start']} B, at or past the {TCGEN05_V0_ADDR_LIMIT} B version-0 tcgen05 "
                f"descriptor window (N_Q={cfg.N_Q}, STAGES_KV={cfg.STAGES_KV}); the kernel derives DESC_VERSION from this layout "
                f"(d256_decode_desc_version), so an operand past the line must be validated under version 1 before the record is admitted",
            ),
        ]
    )


def make_cfg_d256_decode(params: TemplateParams) -> Tuple[CfgD256Decode, TmaIters]:
    """Config for sm107/decode_d256_f16.py: the SM100 decode record (config_sm100.make_cfg_d256_decode,
    whose predicates are arch-neutral) re-validated against the Rubin SMEM carveout and the version-0
    tcgen05 descriptor window.

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
    cfg, tma = _make_cfg_d256_decode_record(params)
    _validate_cfg_d256_decode_sm107(cfg, flavor)
    return cfg, tma
