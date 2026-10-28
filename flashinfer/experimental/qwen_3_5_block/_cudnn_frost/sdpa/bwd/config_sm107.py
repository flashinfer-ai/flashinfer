# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel configuration for the FROST SM107 (Rubin) SDPA-backward d256 flavors.

Three kernel BODIES share this module, one per dtype family, and each is its
own pipeline rather than a dtype arm of the other:

* ``kernels/sm107/bprop_d256_f16.py`` (bf16 / fp16): natural MMA order
  ``Q.K -> dO.V -> P.dO``, P written back INTO the S accumulator (no P ring, no
  ``s_acc_empty``), BMM1 K-split (the K back-half is UTCCP'd into the spare
  TMEM columns and the freed SMEM hosts the second ``dO_dv`` stage).
* ``kernels/sm107/bprop_d256_fp8.py`` (per-tensor FP8 E4M3): lookahead MMA
  order ``Q.K[i+1]`` ahead of ``S.dO[i]``, a 2-stage fp8 P ring in TMEM, three
  3-deep Q / dO rings.
* ``kernels/sm107/bprop_d256_mxfp8.py`` (MXFP8: E4M3 payloads + E8M0 block scale
  factors, ``FAMILY_MXFP8``): the fp8 body's pipeline -- the 2-stage e4m3 P ring
  in TMEM [512, 576), BMM2 = ``mma_ts`` -- with every UTCCP'd scale-factor atom
  ALIASED into the P-ring slot that is dead in the current q iteration (no
  dedicated SF columns; the slot-relative ``TmemLayout.SF_*_OFF``) and ONE added
  commit ring ``mb_p_sf_consumed`` closing the softmax-store-vs-SF-read race;
  six SF slabs ride the existing ``_full`` mbarriers; dS under ``DS_SF_POLICY``.

Both compute dV in-kernel and store dS to a GMEM workspace; dK and dQ are the
``bprop_matmul_blackwell`` GEMMs over that workspace.  The FP8 chain writes its
dS as **E4M3** (``dS_q = e4m3(dS * scale_dP)``, the ``DTYPE_DS`` default) and
the GEMMs render the template's fp8 K64 arm, whose epilogue undoes ``scale_dP``;
``api_dsl_sm107.FP8_DS_DTYPE = DTYPE_BF16`` selects the bf16-dS twin (bf16 GEMM
renderings over exact upcasts) for A/B.  GQA is folded by
``bprop_chain_common.dkv_reduce``.

**Why this is a separate module from** :mod:`cudnn.sdpa.bwd.config_sm100`: the
same reason the forward has one (:mod:`cudnn.sdpa.fwd.config_sm107`).  The
Blackwell d512 backward is a cga4x1 role split with 8 warps; these Rubin bodies
are a single cga2 pair with 12 warps, a different TMEM map (576 columns) and a
327 KiB SMEM layout.  The two overlap in the NAME of a few fields and in
nothing else, and routing one body through the other's ``make_cfg`` is how the
forward d256 FP8 port got a scheduler-ring init count no warp population could
reach -- a 100 %-GPU hang at every shape with no fault and nothing for a
sanitizer to find.  A config value is a claim about a body; the body cannot
check it at run time, so the claims are pinned here, per body.

Provenance: every value is the one the two bodies were written and validated
against (dV / dK / dQ ``cos = 1.0`` on Rubin), keyed here by op geometry
(``d256``) per the engine contract.  **Field names are the pre-port bodies'
own, verbatim** (``STAGES_dO``, ``SWA_WINDOW``, ``CAUSAL_BOTTOM_RIGHT``, ...),
so a kernel port can reference ``CFG.<name>`` without a rename pass; the
FROST-only additions carry NEW names (``IS_FP8``, ``DTYPE_DS`` / ``BPE_DS``,
``IDESC_K_DIM``, ``K_SPLIT_UTCCP``, ``XFER_STAGES``, ``STATS_STAGES``,
``READ_TILE_ARRIVERS_TOT``, ``SOFT_X_CTA_MMA``, ``MMA_COMMIT_ARRIVES``,
``SEQ_KV_LENS_PRESENT``; the MXFP8 family's ``IS_MXFP8``, ``SF_*``,
``*_SF_TX``, ``P_SCALE_LOG2`` / ``P_SF_BYTE``,
``DS_SF_POLICY`` / ``DS_PAYLOADS`` / ``DS_SF_ATOMS`` (``sf_workspace_bytes``), ``SCALED_FP8_PACK``,
``MASK_Q_PAD`` -- every one of them 0 on the other two families).  Two dead
knobs of the pre-port bodies
(``SPLIT_PIPELINE``, ``V2_PIPELINE``) and the unused ``dK_SWZ_BYTES`` are NOT
carried -- the ports delete their (already dead) uses.

**Dtype codes are ``tile_dsl.constants.DTYPE_*``** (E4M3 = 0, E5M2 = 1,
BF16 = 2, FP16 = 3), NOT the pre-port bodies' private 0/1/2.  The f16 body's
``STORAGE_DTYPE = Float16 if DTYPE_QKV == 0 else BFloat16`` dispatch and the
``DTYPE_O in (1, 2)`` grad-dtype dispatch must be rewritten against
``DTYPE_FP16`` / ``DTYPE_BF16`` in the port; the fp8 body's ``DTYPE_QKV == 0``
happens to coincide with ``DTYPE_E4M3`` and stays correct.

What this module is the single source of truth for (never re-literal these in
a kernel):

1. the SMEM buffer table in DECLARATION ORDER (:func:`smem_layout`) -> the byte
   tally the launcher uses (:func:`kernel_smem_bytes`), the per-CTA cap
   predicate, and the tcgen05 descriptor version (:func:`desc_version`, from
   the ``build()`` ROOTS of that table -- rules/mma-tma-matrix.md S6);
2. the TMEM column map (:func:`tmem_layout`, 576 columns, ``is_exclusive``);
3. the 12-warp register split and warp ids;
4. the mbarrier lane constants and the scheduler-ring arriver count
   (:func:`read_tile_arrivers_tot`, derived from the body's call sites);
5. the ring depths the bodies were validated at.

All configuration errors raise :class:`ValueError`.  Anything a *user-built
graph* could trip must be rejected earlier, by the engine row's
``Capabilities`` / the adapter's ``check_support`` -- a ``ValueError`` from here
means those have a gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.constants import (
    DTYPE_BF16,
    DTYPE_E4M3,
    DTYPE_E5M2,
    DTYPE_FP16,
    DTYPE_FP32,
    MASK_CAUSAL,
    MASK_NONE,
    MASK_PADDED,
    MASK_SWA,
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
)

# The per-graph record is SHARED with the SM100 backward so the two adapters
# spell dtype / band / padding / schedule identically; the Rubin body's extra
# knobs are appended below (see TemplateParams).
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.bwd.config_sm100 import TemplateParams as _BwdTemplateParams

__all__ = [
    "TemplateParams",
    "CfgBwdD256",
    "BufferElems",
    "TmemLayout",
    "SmemSlab",
    "FAMILY_F16",
    "FAMILY_FP8",
    "FAMILY_MXFP8",
    "DS_SF_NONE",
    "DS_SF_P_A",
    "DS_SF_P_B",
    "DS_SF_P_C",
    "DS_SF_POLICY_DEFAULT",
    "MX_BLOCK",
    "SF_ATOM_BYTES",
    "SF_ATOM_COLS",
    "SF_TMEM_COLS_PER_ATOM",
    "MXFP8_P_SCALE_LOG2",
    "sf_smem_bytes",
    "sf_tmem_cols",
    "SMEM_CAP_BYTES",
    "SMEM_SCAFFOLD_BYTES",
    "SMEM_USABLE_BYTES",
    "TCGEN05_V0_ADDR_LIMIT",
    "TMEM_TOTAL_COLS",
    "TMEM_IS_EXCLUSIVE",
    "REG_BUDGET_PER_CTA",
    "reg_entry_pool",
    "make_cfg_d256_bwd",
    "bpe",
    "buffer_elems",
    "tmem_layout",
    "smem_layout",
    "smem_bytes",
    "kernel_smem_bytes",
    "desc_roots",
    "desc_version",
    "mbar_stage_counts",
    "scaffold_bytes_declared",
    "read_tile_arrivers_tot",
    "validate_head_chunk",
    "q_pad_rows",
    "kv_pad_rows",
    "ds_workspace_bytes",
    "sf_workspace_bytes",
    "launch_grid",
    "thd_units_upper_bound",
]


# ---------------------------------------------------------------------------
# Rubin architecture constants
# ---------------------------------------------------------------------------

# Rubin SM10.7 oversized per-CTA SMEM cap (``CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_
# SHARED_MEMORY_PER_BLOCK`` = 334848 B, CUDA 13.4+).  Reaching past 227 KiB
# requires the launcher to set ALLOW_OVERSIZED_SHARED_MEMORY; both bodies do
# (the f16 layout is 322 KiB of slabs).
SMEM_CAP_BYTES = 327 * 1024

# Everything that is NOT one of the 1024-B-aligned operand / staging slabs:
# the ~45 mbarriers (16-B-aligned Int64 arrays), the scheduler's two rings +
# 16-B response slots, the TMEM-pointer word.  The DECLARED tally
# (:func:`scaffold_bytes_declared`) is ~0.6 KiB on both bodies; 2 KiB is the
# budget it must stay under, so an added mbarrier ring surfaces here as a
# raise instead of as the last slab silently overflowing the cap.
SMEM_SCAFFOLD_BYTES = 2 * 1024

# What the slabs may use.  This is deliberately NOT the forward's 320 KiB
# ``SMEM_USABLE_BYTES``: that figure is one kernel's self-imposed guard, and
# the f16 backward body is a MEASURED 322 KiB-of-slabs layout that launched
# and validated on Rubin -- a 320 KiB ceiling would reject a kernel that runs.
# The honest bound is the device cap minus the declared scaffolding.
SMEM_USABLE_BYTES = SMEM_CAP_BYTES - SMEM_SCAFFOLD_BYTES

# 14-bit ``start_address`` window of a version-0 tcgen05 SMEM descriptor.  Any
# ``SmemTile.desc()`` ROOT (a ring stage start, or a ``.shifted()`` UTCCP
# source) at or past this wraps to offset 0 and the MMA multiplies the bottom
# of SMEM -- exactly-zero accumulator, no crash (rules/mma-tma-matrix.md S6).
TCGEN05_V0_ADDR_LIMIT = 256 * 1024

# TMEM columns.  Rubin raises the Blackwell 512 to 576, which needs
# ``tmem_alloc(..., is_exclusive=True)`` (tile_dsl/tmem.py); alloc and dealloc
# must both pass ``TmemLayout.TOTAL_COLS``.
TMEM_TOTAL_COLS = 576
TMEM_IS_EXCLUSIVE = True

# Register file: 65 536 x 32-bit per SM, one CTA per SM, so the sum of the
# per-warp counts over the LAUNCHED warps must fit 65536 / 32.
REG_BUDGET_PER_CTA = 65536 // 32  # 2048


def reg_entry_pool(total_warps: int) -> int:
    """The per-CTA register POOL that ``setmaxnreg`` redistributes = what the LAUNCH allocated: ptxas's entry
    count (65536 / 32 / total_warps, rounded DOWN to the 8-register allocation granule) times the warps -- NOT
    the 65536 / 32 register file.  ``setmaxnreg.inc`` can only draw what the ``.dec`` warps released into that
    pool, so a split whose per-warp sum exceeds it leaves the last INCREASE warp parked forever: a HANG at 100 %
    SM utilization at EVERY shape, no fault, no message, and no SASS pin sees it (the split is present and the
    counts look fine).  2026-09-23: the f16 body's 8 x 232 + 4 x 48 = 2048 > 12 x 168 = 2016 hung the very first
    Rubin launch of the port (1 cluster, 1 tile); the pre-port 8 x 232 + 4 x 40 = 2016 balanced exactly, as
    every CUTLASS warp-specialized split does.  Same rule as the "`.maxnreg 128` HANGS" dead end on d512."""
    return (REG_BUDGET_PER_CTA // total_warps) // 8 * 8 * total_warps


_SMEM_SLAB_ALIGN = 1024  # every slab is a 1024-B-aligned cutlass.Array

FAMILY_F16 = "f16"
FAMILY_FP8 = "fp8"
FAMILY_MXFP8 = "mxfp8"
_FAMILIES = (FAMILY_F16, FAMILY_FP8, FAMILY_MXFP8)

_FLAVOR = {FAMILY_F16: "sm107 bwd d256 f16", FAMILY_FP8: "sm107 bwd d256 fp8", FAMILY_MXFP8: "sm107 bwd d256 mxfp8"}


# ---------------------------------------------------------------------------
# MXFP8 block-scaling vocabulary (the OCP MX / cuDNN F8_128x4 convention the
# forward's block-scale kernels and ``tile_dsl.mma`` share)
# ---------------------------------------------------------------------------

# 32 consecutive K elements share one E8M0 scale byte.
MX_BLOCK = 32
# A scale-factor ATOM covers 128 operand rows x 4 K-groups (= 128 K elements):
# 512 B in SMEM (the TMA box / UTCCP source unit), 4 TMEM columns once UTCCP'd
# (``SF_REGISTERS_PER_BLOCK`` in ``sm107/prefill_d256_mxfp8.py``: 16 swizzled
# K bytes x 8 bit / 32).
SF_ATOM_ROWS = 128
SF_ATOM_K = 4 * MX_BLOCK
SF_ATOM_BYTES = SF_ATOM_ROWS * SF_ATOM_K // MX_BLOCK  # 512
SF_ATOM_COLS = SF_ATOM_K // MX_BLOCK  # 4 scale columns (K-groups) per atom row; == tile_dsl.sf_layout.SF_ATOM_COLS
SF_TMEM_COLS_PER_ATOM = 4
# P's ONE fixed power-of-two quantization scale, cuDNN's MXFP8 backward
# convention (the SM100 chain's ``p_scale_log2`` default): P_q = e4m3(P * 2^8),
# descaled inside BMM2 by the constant E8M0 byte 127 - 8 = 119.  A compile-time
# family constant, never a knob (numerics-changing).
MXFP8_P_SCALE_LOG2 = 8

# dS scale-factor policy.  A FAMILY CONSTANT fixed at ``make_cfg`` time from
# ``TemplateParams.ds_sf_policy`` (numerics-changing, so a per-graph compile-time
# constant and never a knob).  Decides the dS workspace dtype, the dS ring depth
# and the payload / SF staging slabs (the slabs after ``sP`` in ``smem_layout``).
DS_SF_NONE = 0  # the f16 / per-tensor fp8 bodies: no block scale factors at all
# P-a: 32x32 TILE scale -- one e4m3 payload ring at the fp8 body's 3-deep ring, TWO SF atoms staged per stage (the one tile
# byte expanded once per GEMM orientation: sf_ds_dk kv-row-major, sf_ds_dq q-row-major).  Optional validation only -- not a
# target, never the default.
DS_SF_P_A = 1
# P-b: exact 1x32 both ways -- TWO e4m3 payload rings (ds_dk: dS scaled per 32-q block, the dK GEMM's A; ds_dq: dS scaled per
# 32-kv block, the dQ GEMM's A) at a 2-deep ring, two SF atoms staged per stage (sf_ds_dk kv-row-major, sf_ds_dq q-row-major).
# The along-kv scale is a warp redux (32 lanes = the 32 kv rows of one block); its column scales stay in registers, so no
# 2 KiB rcp gather slab is declared (``BufferElems.rcpGatherElems`` stays 0).  The SHIPPED policy (``DS_SF_POLICY_DEFAULT``), flipped
# after the Rubin A/B against the bf16-dS chain (whole row +22.0 % dense / +17.1 % causal at B=1 H=128/128 S=8192).
DS_SF_P_B = 2
# P-c: bf16 dS (no dS quantization) -- a 2-stage bf16 ring, no SF staging.  The oracle twin (the kernel's bring-up arm), KEPT
# as a built, selectable arm (``api_dsl_sm107.MXFP8_DS_SF_POLICY = DS_SF_P_C``); its accept cells keep the bf16 recipe on dK / dQ.
DS_SF_P_C = 3
_DS_SF_POLICIES = (DS_SF_P_A, DS_SF_P_B, DS_SF_P_C)
_DS_SF_POLICY_NAME = {DS_SF_NONE: "none", DS_SF_P_A: "P-a", DS_SF_P_B: "P-b", DS_SF_P_C: "P-c"}
# The -1 inherit of ``TemplateParams.ds_sf_policy`` on the MXFP8 family: the SHIPPED policy is P-b -- exact 1x32 block-scaled
# e4m3 dS both ways into the block-scale stage-3 GEMM arm (no dequant pass).  The kernel was brought up under P-c (bf16 dS, the
# oracle twin), which stays a built, selectable arm; P-a is optional validation only.  Flipped to DS_SF_P_B once the P-b chain
# beat the bf16-dS chain on Rubin (cc 10.7, 212 SMs, SM clock 2376 MHz; B=1 H=128/128 S=8192: whole row +22.0 % dense / +17.1 %
# causal, every P-b accept cell within the fp8 recipe, the dS payloads bit-exact against the oracle's 1x32 quantization); never
# P-a.  The adapter still names the policy explicitly per graph (numerics-changing, so a graph fact, never a knob).  Pinned by
# test_mxfp8_family_default_policy_is_the_shipped_p_b.
DS_SF_POLICY_DEFAULT = DS_SF_P_B


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def sf_smem_bytes(rows: int, k: int) -> int:
    """Bytes of one scale-factor slab (one ring stage) for a ``rows x k`` MMA
    operand: whole F8_128x4 atoms, ``ceil(rows/128) * ceil(k/128) * 512`` (=
    ``ceil128(rows) * ceil128(k) / 32``, the forward's ``SF_SMEM_SIZE_*`` in
    ``sm107/prefill_d256_mxfp8.py`` / ``fwd/config_sm107._d256_mxfp8_sf_sizes``).
    Full-size regardless of CTA_MMA: the cga2 peers multicast their halves into
    the same slab."""
    return _ceil_div(rows, SF_ATOM_ROWS) * _ceil_div(k, SF_ATOM_K) * SF_ATOM_BYTES


def sf_tmem_cols(rows: int, k: int) -> int:
    """TMEM columns the UTCCP'd scale factors of a ``rows x k`` operand take:
    ``4 x ceil(rows/128) x ceil(k/128)`` (the forward's ``SF_TMEM_COLS_*``).
    BOTH extents count -- ``4 * ceil(TILE_N/128)`` for P coincides with this only
    while ``TILE_M == TILE_N``."""
    return SF_TMEM_COLS_PER_ATOM * _ceil_div(rows, SF_ATOM_ROWS) * _ceil_div(k, SF_ATOM_K)


# ---------------------------------------------------------------------------
# Per-graph record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateParams(_BwdTemplateParams):
    """The SM100 backward record plus the two knobs only these bodies read.

    Inherited (spelled exactly as the SM100 adapter spells them): ``dtype_qkv``,
    ``window_left`` / ``window_right`` (``window_right`` set => causal,
    ``window_left`` set => SWA), ``bottom_right``, ``seq_kv_lens_present``,
    ``seq_q_lens_present``, ``thd_varlen``, ``sched_policy``, ``xfer_halves``.
    ``xfer_halves`` is a d512 role-split tuning knob with no counterpart in
    these bodies and is INERT here.  ``seq_q_lens_present`` is rejected (no
    body threads a per-batch Q length on the DENSE path); ``thd_varlen`` is
    served by all three bodies (packed ``[1, T, H, D]`` operands, a kv-blocked
    dS workspace, per-sequence lengths from the metadata buffer -- the module
    docs of ``sm107/bprop_d256_f16.py`` / ``bprop_d256_fp8.py`` /
    ``bprop_d256_mxfp8.py``; the MXFP8 body over packed per-sequence-tile-padded
    scale factors).

    A plain :class:`cudnn.sdpa.bwd.config_sm100.TemplateParams` is also
    accepted by :func:`make_cfg_d256_bwd` (the two extras default).
    """

    # Gradient (dV; and via the quantize pass dK / dQ) storage dtype.  -1 =
    # inherit: the io dtype on the f16 family, E4M3 on the fp8 family (the
    # cuDNN ``sdpa_fp8_backward`` contract: FP8 grads + amax).  BF16 / FP16 on
    # the fp8 family is the pre-quantization output used for the bitwise A/B
    # against the pre-port kernel.
    dtype_o: int = -1
    # dS GMEM-workspace storage dtype.  -1 = inherit: the io dtype on the f16
    # family, E4M3 on the fp8 family (dS_q = e4m3(dS * scale_dP), the cuDNN
    # ``sdpa_fp8_backward`` recipe -- the stage-3 GEMMs then run the fp8 K64
    # arm over the e4m3 payloads).  BF16 on the fp8 family is the pre-quantized
    # dS the bf16 stage-3 renderings consume unchanged (the A/B twin,
    # ``api_dsl_sm107.FP8_DS_DTYPE``); the f16 family takes only its io dtype.
    dtype_ds: int = -1
    # Informational only: the main kernel is sink-agnostic (dQ/dK/dV are already
    # sink-correct through the sink-aware LSE the forward wrote); it gates the
    # separate dsink reduction launch in the adapter.  No body effect.
    has_sink: bool = False
    # --- MXFP8 family only (append-only; REJECTED, not ignored, when set on the f16 / fp8 families) ---
    # The Rubin fused scale-and-pack ``cvt.rn.satfinite.scaled::n1::ue8m0.e4m3x2.f32``
    # (assembles for sm_107a only) for the P and dS quantizers; False = the
    # bit-identical FMUL-then-pack arm.  Set by the adapter from
    # ``compute_capability == (10, 7)``, never by a device query in the kernel
    # (the ``_EXP2_FMA_SPLIT_CC`` idiom).
    scaled_fp8_pack: bool = False
    # The ``q < seqlen_q_real`` band in the transposed mask (P select-zero on
    # the q pad rows), set by the adapter iff ``S_q % 128 != 0`` and folded out
    # of the dense specialization otherwise.  The MXFP8 body's Q-side SF pad
    # rows are producer-defined bytes, so a masked P is what keeps a 0xFF pad
    # byte (E8M0 NaN) out of dV.
    mask_q_pad: bool = False
    # dS scale-factor policy: ``DS_SF_P_A`` / ``DS_SF_P_B`` / ``DS_SF_P_C``, -1 =
    # ``DS_SF_POLICY_DEFAULT`` (P-b, the shipped block-scaled chain; P-c is the
    # bf16-dS oracle twin, kept selectable; P-a is optional validation only).
    # Decides ``DTYPE_DS`` (-1 -> e4m3 under P-a / P-b, bf16
    # under P-c), the dS ring depth and the payload / SF staging slabs.
    # Numerics-changing: a per-graph compile-time constant, never a knob.
    ds_sf_policy: int = -1
    # --- the 2x2-datapath body (``kernels/bprop_d256_2x2_f16.py``, config ``bwd/config_d256_2x2``) ---------------
    # 0 = the shipped 4x1 bodies (this module's ``make_cfg_d256_bwd``; REJECTED != 0 there -- a 4x1 body never
    # reads it, so a non-zero value would be a claim it cannot honour); 1 = the SM100 profile (one 64-row sub-block per
    # CTA, the ``sdpa_bwd_sm100_d256`` row); 2 = the Rubin interleaved profile (two sub-blocks per CTA, the
    # ``api_dsl_sm107.BWD_D256_2X2`` twin).  Appended last so every positional caller keeps working; it rides
    # ``repr(params)`` into the template digest, so the 4x1 renderings' PTX is unchanged while their compiled-plan
    # cache key moves once.
    datapath_2x2_profile: int = 0


def bpe(dtype: int) -> int:
    """Bytes per storage element: FP8 codes 1, BF16 / FP16 2, the output-only DTYPE_FP32 4."""
    if dtype == DTYPE_FP32:
        return 4
    return 1 if dtype <= DTYPE_E5M2 else 2


def _is_fp8_code(dtype: int) -> bool:
    return dtype <= DTYPE_E5M2


# ---------------------------------------------------------------------------
# The Cfg record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfgBwdD256:
    """Geometry of the Rubin d256 backward bodies (d_qk = d_v = 256, cga2).

    Defaults exist so the dataclass is constructible in a doc / test context;
    :func:`make_cfg_d256_bwd` sets every field explicitly and validates.
    """

    # --- Tile shape ---------------------------------------------------------
    TILE_M: int = 128  # kv rows per CTA (M of every collective MMA is TILE_M * CTA_MMA = 256)
    TILE_N: int = 128  # q cols per q-tile (inner loop)
    TILE_K: int = 256  # d_qk
    TILE_O: int = 256  # d_v

    # --- Dtype (tile_dsl.constants.DTYPE_* codes) ---------------------------
    DTYPE_QKV: int = DTYPE_BF16
    DTYPE_O: int = DTYPE_BF16  # dV (in-kernel) grad storage dtype
    BPE: int = 2
    BPE_O: int = 2
    # FROST-only: 1 on the fp8-CLASS families (E4M3 payloads: the per-tensor fp8 body AND the MXFP8 one, which share the
    # K64 / k_dim=1 / 3-deep-ring lookahead pipeline), 0 on f16 (the bodies' dtype dispatch).  IS_MXFP8 tells the two apart:
    # route per-tensor-fp8-ONLY behaviour (scale_s / descale_dP, the E4M3 dV quantize, the amax folds and atomics the MXFP8
    # family compiles OUT) on ``IS_FP8 and not IS_MXFP8``, never on IS_FP8 alone -- a bare ``CFG.IS_FP8`` guard copied from
    # the per-tensor body traces the per-tensor arm into the MXFP8 one (descaled dS / a quantized dV where the graph expects
    # half gradients, no crash).
    IS_FP8: int = 0
    # FROST-only: dS GMEM-workspace dtype.  f16: the io dtype (the stage-3 GEMMs
    # read the workspace as the io dtype -- a mismatch is silent garbage).  fp8:
    # E4M3 by default (dS_q = e4m3(dS * scale_dP); the stage-3 GEMMs run the fp8
    # K64 arm with a descale epilogue), or BF16 (the pre-quantized dS the bf16
    # GEMM renderings consume unchanged -- the A/B twin).  EVERY dS-ring
    # constant (SMEM bytes, TMA box, subtile count, swizzle row) derives from
    # BPE_DS, never from BPE: with an e4m3 io and a bf16 dS the two geometries
    # differ, and a dS walk borrowed from BPE passes every f16 test and fails
    # only fp8.
    DTYPE_DS: int = DTYPE_BF16
    BPE_DS: int = 2

    # --- Cluster (single cga2 pair; both CTAs collaborate on every MMA) ------
    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2

    # --- Per-tensor swizzle (all 128 B at d=256) ----------------------------
    Q_SWZ_BYTES: int = 128
    dO_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    P_SWZ_BYTES: int = 128  # the dS ring's store_swizzled pattern (Swizzle(3,4,3))
    dS_SWZ_BYTES: int = 128  # the dS ring's TMA-store descriptor swizzle
    dV_SWZ_BYTES: int = 128  # dV epilogue staging + TMA-store descriptor swizzle

    # --- MMA k-step (arch-sensitive) + idesc k_dim --------------------------
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16
    # FROST-only: the ``Tcgen05InstrDesc.build(k_dim=)`` value every idesc in
    # the body must pass.  Rubin dense FP8 = K64 + k_dim=1 (arch-OPPOSITE of
    # Blackwell); f16 = K16 + the default 0.  A mismatch scrambles accumulator
    # ROWS silently (rules/mma-tma-matrix.md S1).
    IDESC_K_DIM: int = 0
    # FROST-only: 1 = the f16 BMM1 K-split (K back-half UTCCP'd to TMEM, the
    # freed SMEM aliased by dO_dv stage 1).  Decides the TMEM RSVD purpose and
    # the combined [sdOdv_s0 | K | V] SMEM slab.
    K_SPLIT_UTCCP: int = 0

    # --- Pipeline depths ----------------------------------------------------
    STAGES_Q: int = 2
    STAGES_dO: int = 2  # sdO -- BMM1 dP B-operand ring
    STAGES_dO_DV: int = 2  # sdO_dv -- BMM2 dV B-operand ring (f16: stage 1 = K back-half alias)
    STAGES_KV: int = 1  # K, V loaded ONCE per kv-block
    STAGES_TMEM_S: int = 1  # S / dP accumulators single-buffered
    STAGES_TMEM_P: int = 1  # f16: P inside S_acc; fp8 / mxfp8: 2-stage e4m3 P ring in TMEM [512..576) (mxfp8: its dead slot hosts the SF atoms)
    # FROST-only (module constants in the pre-port bodies): dS SMEM ring depth
    # and the lse / do_dot prefetch ring depth.
    XFER_STAGES: int = 1
    STATS_STAGES: int = 2
    TILES_Q: int = 1
    SCHEDULER_STAGES: int = 2

    # --- Warp specialization (8 compute warps = 2 wg x 4, lane = kv row) ----
    SOFTMAX_WARPGROUPS: int = 2
    SOFTMAX_WG_WARPS: int = 4
    CORRECTION_WARPS: int = 0

    # --- Register budget ----------------------------------------------------
    SOFTMAX_REGS: int = 232
    CORRECTION_REGS: int = 0
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    # --- Mask / sink (compile-time; the TRANSPOSE of the forward) -----------
    # lane = kv row, inner loop = q tile: a fixed kv-block bounds WHICH q-tiles
    # attend and the per-cell mask zeroes P on masked (kv, q) cells; P = 0 makes
    # dV / dS / dK / dQ inherit the mask.
    MASK_FLAGS: int = MASK_NONE
    SWA_WINDOW: int = 0  # window_left (keys q - W .. q); 0 when MASK_SWA is unset
    CAUSAL_BOTTOM_RIGHT: int = 0
    HAS_SINK: int = 0  # informational only (no main-kernel effect)
    # FROST-only: per-batch kv lengths are threaded (MASK_PADDED).
    SEQ_KV_LENS_PRESENT: int = 0
    # FROST-only: THD / varlen (all three bodies; the MXFP8 body also stages packed per-sequence-tile-padded
    # scale factors).  Packed [1, T, H, D] operands addressed through
    # packed-total-clamped runtime descriptors, per-sequence lengths and the kv-blocked dS
    # workspace row offsets from the metadata buffer, a device claim counter in place of CLC,
    # the q-pad band in the transposed mask (MASK_PADDED is set).  Folds out at 0.
    THD_VARLEN: int = 0
    # The kv-blocked THD workspace's row granularity = the pair's kv block (TILE_M * CTA_MMA):
    # every dS / dV byte a (sequence, kv block, head) unit writes lands in rows its sequence
    # owns, so no store needs a skip predicate (the setup launch's ``ws_gran`` / ``cga_tile_m``).
    WS_BLOCK_ROWS: int = 256

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL  # 0/1/2 = natural-3D / lpt / lpt_l2 (flat 1-D grid)

    # --- Warp layout (per CTA) ----------------------------------------------
    #   0..3  softmax wg0 (q[0:64])   4..7  softmax wg1 (q[64:128])
    #   8 MMA   9 TMALDG   10 TMASTG (dS store + stats prefetch)   11 scheduler
    TOTAL_WARPS: int = 12
    THREADS_PER_CTA: int = 12 * 32
    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4
    MMA_WARP_ID: int = 8
    TMALDG_WARP_ID: int = 9
    TMASTG_WARP_ID: int = 10
    SCHED_WARP_ID: int = 11

    # --- mbarrier lane constants (P3) ---------------------------------------
    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_WG_LANES: int = 128  # one wg
    SOFTMAX_LANES: int = 256  # both wgs (every compute lane of one CTA)
    # FROST-only: every compute lane of BOTH CTAs arriving on the pair leader
    # (the LEADER-scope ``s_acc_empty`` / ``dp_empty`` / ``p_ready`` /
    # ``dv_acc_empty`` / ``tmem_dealloc`` counts).
    SOFT_X_CTA_MMA: int = 512
    # FROST-only: one ``tcgen05.commit`` multicast = one arrive per target CTA.
    MMA_COMMIT_ARRIVES: int = 1
    # FROST-only: per-CTA init count of the scheduler's ``mb_read_tile_id``;
    # see read_tile_arrivers_tot() for the derivation naming the body.
    READ_TILE_ARRIVERS_TOT: int = 21

    N_BMM2_CHUNKS: int = 2
    BMM2_CHUNK_SIZE: int = 64

    # --- MXFP8 block scaling (FROST-only; EVERY field 0 on the f16 / fp8 families, pinned) ----
    IS_MXFP8: int = 0
    SF_BLOCK: int = 0  # 32 K elements per E8M0 byte (MX_BLOCK)
    SF_BLOCKS_PER_STEP: int = 0  # MmaDesc(sf_blocks_per_step=): TILE_K_HW // SF_BLOCK = 2 at K64
    P_SCALE_LOG2: int = 0  # P_q = e4m3(P * 2^P_SCALE_LOG2) -- 8
    P_SF_BYTE: int = 0  # the constant E8M0 descale byte BMM2 applies to P: 127 - P_SCALE_LOG2 = 119
    # Bytes of one SF slab / ring stage, the F8_128x4 atom form (sf_smem_bytes(rows, K) of the operand it scales).
    SF_SMEM_K: int = 0  # BMM1 S   A = K    [TILE_M kv  x TILE_K]
    SF_SMEM_V: int = 0  # BMM1 dP  A = V    [TILE_M kv  x TILE_O]
    SF_SMEM_Q: int = 0  # BMM1 S   B = Q    [TILE_N q   x TILE_K]  (the full N tile in both CTAs)
    SF_SMEM_dO: int = 0  # BMM1 dP  B = dO   [TILE_N q   x TILE_O]
    SF_SMEM_P: int = 0  # BMM2 dV  A = P    [TILE_M kv  x TILE_N q]  (constant byte P_SF_BYTE, filled once, UTCCP'd per q iteration)
    SF_SMEM_dOT: int = 0  # BMM2 dV  B = dO_T [TILE_O d_v x TILE_N q]  (columnwise SF; the full N=256 in both CTAs)
    SF_SMEM_dS: int = 0  # ONE dS SF atom [TILE_M x TILE_N] (512 B); DS_SF_ATOMS of them are staged per dS ring stage for the TMA store
    # TMEM columns each operand's UTCCP'd scale factors take: sf_tmem_cols(rows, K) = 4 x ceil(rows/128) x ceil(K/128).  They
    # alias the dead P-ring slot of the current q iteration (TmemLayout: slot-relative SF_*_OFF), never dedicated columns.
    SF_TMEM_COLS_K: int = 0
    SF_TMEM_COLS_V: int = 0
    SF_TMEM_COLS_Q: int = 0
    SF_TMEM_COLS_dO: int = 0
    SF_TMEM_COLS_P: int = 0
    SF_TMEM_COLS_dOT: int = 0
    # expect_tx growth per SF load = SF_SMEM x CTA_MMA: a cga2 tensor TMA delivers BOTH peers' bytes to the LEADER's
    # mbar (P9) -- Q / dO / dO_T-SF: 2 CTAs x 512 B x 2 destinations, K / V-SF: 2 CTAs x 1024 B self-multicast.
    K_SF_TX: int = 0
    V_SF_TX: int = 0
    Q_SF_TX: int = 0
    dO_SF_TX: int = 0
    dOT_SF_TX: int = 0
    # dS scale-factor policy (DS_SF_*) and the two slab counts it decides.
    DS_SF_POLICY: int = DS_SF_NONE
    DS_PAYLOADS: int = 1  # dS payload rings in SMEM (and workspace tensors): 2 under P-b (along q + along kv), else 1
    # dS SF atoms staged per dS ring stage: P-a 2 (the ONE tile byte expanded once per GEMM orientation -- the dK atom is
    # kv-row-major [kv rows x q groups], the dQ atom q-row-major [q rows x kv groups]: two distinct F8_128x4 atoms, the
    # workspace tensors sf_ds_dk / sf_ds_dq), P-b 2 (one per payload), P-c 0 (bf16 dS, no scale factors).
    DS_SF_ATOMS: int = 0
    # Trace-time 0/1 flags from TemplateParams (see there).
    SCALED_FP8_PACK: int = 0
    # MASK_Q_PAD is a SEPARATE 0/1 field, NOT a bit of MASK_FLAGS (the CAUSAL_BOTTOM_RIGHT precedent: a const_expr arm, not
    # a mask bit).  The transposed mask's ``hi = min(hi, seqlen_q_real)``
    # band reads ``cutlass.const_expr(CFG.MASK_Q_PAD)``: a mask dispatch keyed on MASK_FLAGS alone folds the band out at
    # S_q % 128 != 0 and lets a 0xFF Q-side SF pad byte (E8M0 NaN) reach BMM2 (dV NaN on every kv row).
    MASK_Q_PAD: int = 0


# Every MXFP8-only Cfg field that must read 0 on the f16 / per-tensor fp8 bodies (IS_MXFP8 first, so a message names it).
_MXFP8_ZERO_FIELDS = (
    "IS_MXFP8",
    "SF_BLOCK",
    "SF_BLOCKS_PER_STEP",
    "P_SCALE_LOG2",
    "P_SF_BYTE",
    "SF_SMEM_K",
    "SF_SMEM_V",
    "SF_SMEM_Q",
    "SF_SMEM_dO",
    "SF_SMEM_P",
    "SF_SMEM_dOT",
    "SF_SMEM_dS",
    "SF_TMEM_COLS_K",
    "SF_TMEM_COLS_V",
    "SF_TMEM_COLS_Q",
    "SF_TMEM_COLS_dO",
    "SF_TMEM_COLS_P",
    "SF_TMEM_COLS_dOT",
    "K_SF_TX",
    "V_SF_TX",
    "Q_SF_TX",
    "dO_SF_TX",
    "dOT_SF_TX",
    "DS_SF_POLICY",
    "DS_SF_ATOMS",
    "SCALED_FP8_PACK",
    "MASK_Q_PAD",
)


# ---------------------------------------------------------------------------
# Derived element counts / TMA geometry -- the pre-port bodies' module constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferElems:
    """The pre-port bodies' module-level buffer / TMA constants, by name.

    Bind them at module level in a kernel (``_B = buffer_elems(CFG)``,
    ``qBufferElems = _B.qBufferElems`` ...) instead of re-deriving.  Every dS
    figure is BPE_DS-driven (see ``CfgBwdD256.DTYPE_DS``).
    """

    _M_PER_CTA: int  # per-CTA q rows of the Q / dO N-split (TILE_N // CTA_MMA)
    qBufferElems: int  # one Q ring stage (per CTA): _M_PER_CTA x TILE_K
    dOBufferElems: int  # one dO ring stage (per CTA): _M_PER_CTA x TILE_O (== TILE_N x TILE_O / CTA_MMA for sdO_dv)
    kBufferElems: int  # K (per CTA): TILE_M x TILE_K
    vBufferElems: int  # V (per CTA): TILE_M x TILE_O
    pBufferElems: int  # a [kv, q] tile: TILE_M x TILE_N
    dSBufferElems: int  # one dS ring stage: TILE_M x TILE_N (elements of DTYPE_DS)
    dVBufferElems: int  # dV epilogue staging: TILE_M x TILE_O (elements of DTYPE_O)
    K_BACK_OFF_ELEMS: int  # f16 K-split: the K back-half (d_qk[128:256]) offset inside sK
    Q_BACK_OFF_ELEMS: int  # f16 K-split: the Q back-half offset inside a sQ stage
    _DODV_STAGE_ELEMS: int  # sdO_dv ring stride (f16: dOBufferElems + K_BACK_OFF_ELEMS so stage 1 == sK_back)
    STATS_SLOT_ELEMS: int  # fp32 elems per stats ring slot: lse[TILE_N] | do_dot[TILE_N]
    STATS_LSE_OFF: int
    STATS_DOT_OFF: int
    _SMX_CHUNK: int  # q cols per softmax wg (TILE_N // SOFTMAX_WARPGROUPS)
    _LDTM_NUM: int  # tcgen05.ld.32x32b.xN granularity of the S / dP reads
    _DSOFT_CHUNK: int  # dSoftmax register-pressure chunk
    _KV_BLOCK_ROWS: int  # kv rows per cga2 pair (TILE_M * CTA_MMA)
    CGA_SIZE: int
    # TMA subtile walks (inner bytes / swizzle bytes).
    TMA_QK_ITERS: int
    TMA_VO_ITERS: int
    TMA_QK_GRANU_ELEMS: int
    TMA_VO_GRANU_ELEMS: int
    TMA_QK_SG1_ITERS: int  # the BT=true dO_dv / Q views (d-split across the pair)
    TMA_VO_SG1_ITERS: int
    TMA_QK_SG1_GRANU_ELEMS: int
    TMA_VO_SG1_GRANU_ELEMS: int
    DV_D_BLOCK: int  # dV store subtile width in d_v elems (dV_SWZ_BYTES / BPE_O)
    TMA_DV_ITERS: int
    TMA_DV_GRANU_ELEMS: int
    DV_BLOCK_SLAB: int  # TILE_M * DV_D_BLOCK
    P_TMA_ITERS: int  # dS store subtiles per row: TILE_N * BPE_DS / dS_SWZ_BYTES
    P_D_BLOCK: int  # q cols per dS store subtile
    P_BLOCK_BYTES: int  # (pre-port misnomer, kept) ELEMS per dS subtile = TILE_M * P_D_BLOCK
    pXferBytes: int  # bytes of one dS ring stage = TILE_M * TILE_N * BPE_DS
    # Byte transactions the expect_tx sites arm (cga2 tensor TMAs route both
    # peers' bytes to the LEADER's mbar -> x CTA_MMA; the dV store is per-CTA).
    qTmaTransactionBytes: int
    dOTmaTransactionBytes: int
    kTmaTransactionBytes: int
    vTmaTransactionBytes: int
    dVTmaTransactionBytes: int
    # tcgen05 SMEM descriptor leading / stride byte offsets.
    LEADING_BYTE_OFFSET_QK: int
    STRIDE_BYTE_OFFSET_QK: int
    LEADING_BYTE_OFFSET_dO: int
    STRIDE_BYTE_OFFSET_dO: int
    LEADING_BYTE_OFFSET_P: int
    STRIDE_BYTE_OFFSET_P: int
    LEADING_BYTE_OFFSET_dS: int
    STRIDE_BYTE_OFFSET_dS: int
    LEADING_BYTE_OFFSET_dV: int
    STRIDE_BYTE_OFFSET_dV: int
    LEADING_BYTE_OFFSET_Q_SG1: int  # BT=true B operand: K x swz (B_PC_COLS // 8 > 8 heuristic)
    STRIDE_BYTE_OFFSET_Q_SG1: int
    LEADING_BYTE_OFFSET_dO_SG1: int
    STRIDE_BYTE_OFFSET_dO_SG1: int
    # SmemTile.layout codes (128 B -> 2, 64 B -> 4, 32 B -> 6).
    SMEM_LAYOUT_Q: int
    SMEM_LAYOUT_dO: int
    SMEM_LAYOUT_K: int
    SMEM_LAYOUT_V: int
    SMEM_LAYOUT_P: int
    SMEM_LAYOUT_dS: int
    SMEM_LAYOUT_dV: int
    # --- MXFP8 (0 on the f16 / fp8 families) ---------------------------------------------------------------------------
    # RETIRED, 0 on every family: the MXFP8 body's e4m3 P ring was an SMEM ring of TILE_M x TILE_N stages until it went back
    # to the fp8 body's TMEM ring (whose dead slot hosts the UTCCP'd SF atoms).  Kept so the siblings' byte-identical layout
    # pins need no edit.
    pRingStageBytes: int = 0
    # dS SF staging per dS ring stage: DS_SF_ATOMS x SF_SMEM_dS (TMA-stored next to the dS slot; no descriptor reads it).
    dSSfStagingBytes: int = 0
    # A P-b rcp gather slab (fp32 x compute warps x _SMX_CHUNK, an SMEM exchange of the along-kv column scales) -- 0 on every policy: the kernel's
    # along-kv consumer keeps the warp-uniform column scales in registers (no SMEM exchange), so no slab is declared.  The
    # field stays so an SMEM-gather twin can re-add the row without moving the others.
    rcpGatherElems: int = 0


_SWZ_ENUM = {128: 2, 64: 4, 32: 6}


def buffer_elems(cfg: CfgBwdD256) -> BufferElems:
    m_per_cta = cfg.TILE_N // cfg.CTA_MMA
    q_elems = m_per_cta * cfg.TILE_K
    do_elems = m_per_cta * cfg.TILE_O
    k_elems = cfg.TILE_M * cfg.TILE_K
    v_elems = cfg.TILE_M * cfg.TILE_O
    k_back = k_elems // 2
    tma_qk_iters = (cfg.TILE_K * cfg.BPE) // cfg.Q_SWZ_BYTES
    tma_vo_iters = (cfg.TILE_O * cfg.BPE) // cfg.dO_SWZ_BYTES
    tma_qk_sg1_iters = ((cfg.TILE_K // cfg.CTA_MMA) * cfg.BPE) // cfg.Q_SWZ_BYTES
    tma_vo_sg1_iters = ((cfg.TILE_O // cfg.CTA_MMA) * cfg.BPE) // cfg.dO_SWZ_BYTES
    dv_d_block = cfg.dV_SWZ_BYTES // cfg.BPE_O
    p_tma_iters = (cfg.TILE_N * cfg.BPE_DS) // cfg.dS_SWZ_BYTES
    p_d_block = cfg.TILE_N // p_tma_iters
    return BufferElems(
        _M_PER_CTA=m_per_cta,
        qBufferElems=q_elems,
        dOBufferElems=do_elems,
        kBufferElems=k_elems,
        vBufferElems=v_elems,
        pBufferElems=cfg.TILE_M * cfg.TILE_N,
        dSBufferElems=cfg.TILE_M * cfg.TILE_N,
        dVBufferElems=cfg.TILE_M * cfg.TILE_O,
        K_BACK_OFF_ELEMS=k_back,
        Q_BACK_OFF_ELEMS=q_elems // 2,
        _DODV_STAGE_ELEMS=(do_elems + k_back) if cfg.K_SPLIT_UTCCP else do_elems,
        STATS_SLOT_ELEMS=2 * cfg.TILE_N,
        STATS_LSE_OFF=0,
        STATS_DOT_OFF=cfg.TILE_N,
        _SMX_CHUNK=cfg.TILE_N // cfg.SOFTMAX_WARPGROUPS,
        _LDTM_NUM=64,
        _DSOFT_CHUNK=64,
        _KV_BLOCK_ROWS=cfg.TILE_M * cfg.CTA_MMA,
        CGA_SIZE=cfg.CGA_M * cfg.CGA_N,
        TMA_QK_ITERS=tma_qk_iters,
        TMA_VO_ITERS=tma_vo_iters,
        TMA_QK_GRANU_ELEMS=cfg.TILE_K // tma_qk_iters,
        TMA_VO_GRANU_ELEMS=cfg.TILE_O // tma_vo_iters,
        TMA_QK_SG1_ITERS=tma_qk_sg1_iters,
        TMA_VO_SG1_ITERS=tma_vo_sg1_iters,
        TMA_QK_SG1_GRANU_ELEMS=(cfg.TILE_K // cfg.CTA_MMA) // tma_qk_sg1_iters,
        TMA_VO_SG1_GRANU_ELEMS=(cfg.TILE_O // cfg.CTA_MMA) // tma_vo_sg1_iters,
        DV_D_BLOCK=dv_d_block,
        TMA_DV_ITERS=(cfg.TILE_O * cfg.BPE_O) // cfg.dV_SWZ_BYTES,
        TMA_DV_GRANU_ELEMS=dv_d_block,
        DV_BLOCK_SLAB=cfg.TILE_M * dv_d_block,
        P_TMA_ITERS=p_tma_iters,
        P_D_BLOCK=p_d_block,
        P_BLOCK_BYTES=cfg.TILE_M * p_d_block,
        pXferBytes=cfg.TILE_M * cfg.TILE_N * cfg.BPE_DS,
        qTmaTransactionBytes=q_elems * cfg.BPE * cfg.CTA_MMA,
        dOTmaTransactionBytes=do_elems * cfg.BPE * cfg.CTA_MMA,
        kTmaTransactionBytes=k_elems * cfg.BPE * cfg.CTA_MMA,
        vTmaTransactionBytes=v_elems * cfg.BPE * cfg.CTA_MMA,
        dVTmaTransactionBytes=cfg.TILE_M * cfg.TILE_O * cfg.BPE_O,
        LEADING_BYTE_OFFSET_QK=0,
        STRIDE_BYTE_OFFSET_QK=8 * cfg.Q_SWZ_BYTES,
        LEADING_BYTE_OFFSET_dO=0,
        STRIDE_BYTE_OFFSET_dO=8 * cfg.dO_SWZ_BYTES,
        LEADING_BYTE_OFFSET_P=0,
        STRIDE_BYTE_OFFSET_P=8 * cfg.P_SWZ_BYTES,
        LEADING_BYTE_OFFSET_dS=0,
        STRIDE_BYTE_OFFSET_dS=8 * cfg.dS_SWZ_BYTES,
        LEADING_BYTE_OFFSET_dV=0,
        STRIDE_BYTE_OFFSET_dV=8 * cfg.dV_SWZ_BYTES,
        LEADING_BYTE_OFFSET_Q_SG1=cfg.TILE_N * cfg.Q_SWZ_BYTES,
        STRIDE_BYTE_OFFSET_Q_SG1=8 * cfg.Q_SWZ_BYTES,
        LEADING_BYTE_OFFSET_dO_SG1=cfg.TILE_N * cfg.dO_SWZ_BYTES,
        STRIDE_BYTE_OFFSET_dO_SG1=8 * cfg.dO_SWZ_BYTES,
        SMEM_LAYOUT_Q=_SWZ_ENUM[cfg.Q_SWZ_BYTES],
        SMEM_LAYOUT_dO=_SWZ_ENUM[cfg.dO_SWZ_BYTES],
        SMEM_LAYOUT_K=_SWZ_ENUM[cfg.K_SWZ_BYTES],
        SMEM_LAYOUT_V=_SWZ_ENUM[cfg.V_SWZ_BYTES],
        SMEM_LAYOUT_P=_SWZ_ENUM[cfg.P_SWZ_BYTES],
        SMEM_LAYOUT_dS=_SWZ_ENUM[cfg.dS_SWZ_BYTES],
        SMEM_LAYOUT_dV=_SWZ_ENUM[cfg.dV_SWZ_BYTES],
        pRingStageBytes=0,
        dSSfStagingBytes=cfg.DS_SF_ATOMS * cfg.SF_SMEM_dS,
        rcpGatherElems=0,  # no rcp gather slab: P-b's column scales stay in registers (BufferElems.rcpGatherElems)
    )


# ---------------------------------------------------------------------------
# TMEM column map (one alloc per CTA, all 576 columns, is_exclusive=True)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TmemLayout:
    """Column map of the pre-port ``KernelTmemLayout``, by field name.

    Both families: ``S_acc`` [0, 128) fp32 [kv, q]; ``dP`` [128, 256) fp32;
    ``dV_acc`` [256, 512) fp32 [kv, d_v] persistent over the q loop.

    * f16: P (the ``mma_ts`` A operand, ``TILE_N * BPE / 4 = 64`` cols) is
      written back INTO ``S_acc`` at [P_OFF=32, 96) -- wg0 [32, 64), wg1
      [64, 96), each inside its own S-read q-half; [512, 576) = RSVD holds the
      UTCCP'd K back-half (d_qk[128:256], ``(TILE_K / 2) * BPE / 4 = 64`` cols).
    * fp8: P is a ``STAGES_TMEM_P``-deep ring of ``TILE_N * BPE / 4 = 32`` cols
      at [512, 576); RSVD_COLS = 0.
    * mxfp8: the fp8 map (the 2-stage e4m3 P ring at [512, 576), RSVD 0) and
      ZERO dedicated SF columns: every UTCCP'd scale-factor atom ALIASES the
      P-ring slot that is dead in the current q iteration -- with n the
      kernel-global q-iteration count (the ``mb_p_ready`` PipelineState), slot
      ``n % 2`` holds P[n] and slot ``(n + 1) % 2`` the scale factors.  The
      ``SF_*_OFF`` are therefore SLOT-RELATIVE column offsets, added to
      ``P_OFF + sf_slot * P_COLS``: the BMM1 band ``K | V | Q | dO`` =
      ``0 | 8 | 16 | 24`` (``SF_BMM1_COLS = 32 <= P_COLS``, the whole slot) and
      the BMM2 band ``P | dOT`` = ``0 | 4`` (``SF_BMM2_COLS = 12``), written over
      the dead BMM1 columns after the BMM1s of the iteration were issued.
    """

    TOTAL_COLS: int
    S_OFF: int
    S_COLS: int
    dP_OFF: int
    dP_COLS: int
    dV_OFF: int
    dV_COLS: int
    P_OFF: int
    P_COLS: int
    RSVD_OFF: int
    RSVD_COLS: int
    # MXFP8 scale-factor columns (UTCCP targets, ``tmem_sf_a`` / ``tmem_sf_b`` of the block-scale MMAs); 0 on the
    # other families.  OFF is SLOT-RELATIVE (the kernel adds ``P_OFF + sf_slot * P_COLS``); COLS mirror
    # ``CfgBwdD256.SF_TMEM_COLS_*`` so the map is self-contained for the kernel.
    SF_K_OFF: int = 0
    SF_K_COLS: int = 0
    SF_V_OFF: int = 0
    SF_V_COLS: int = 0
    SF_Q_OFF: int = 0
    SF_Q_COLS: int = 0
    SF_dO_OFF: int = 0
    SF_dO_COLS: int = 0
    SF_P_OFF: int = 0
    SF_P_COLS: int = 0
    SF_dOT_OFF: int = 0
    SF_dOT_COLS: int = 0
    # The two alias bands inside ONE P slot (0 on the other families): BMM1 = K | V | Q | dO, BMM2 = P | dOT.
    SF_BMM1_COLS: int = 0
    SF_BMM2_COLS: int = 0


def tmem_layout(cfg: CfgBwdD256) -> TmemLayout:
    s_cols = cfg.TILE_N
    dp_cols = cfg.TILE_N
    dv_cols = cfg.TILE_O
    p_cols = (cfg.TILE_N * cfg.BPE) // 4
    dv_off = s_cols + dp_cols
    tail = dv_off + dv_cols  # 512
    if cfg.IS_MXFP8:
        # The fp8 map (the P ring owns the tail) + the SF alias: every atom lands in the DEAD P slot of the current q
        # iteration at a slot-relative column.  Two bands share the slot in time: the BMM1 band (K | V | Q | dO, one slot
        # exactly) and, once the BMM1s were issued, the BMM2 band (P | dOT) over the dead K / V columns.
        sf_fields = {}
        col = 0
        for name, cols in (("K", cfg.SF_TMEM_COLS_K), ("V", cfg.SF_TMEM_COLS_V), ("Q", cfg.SF_TMEM_COLS_Q), ("dO", cfg.SF_TMEM_COLS_dO)):
            sf_fields[f"SF_{name}_OFF"] = col
            sf_fields[f"SF_{name}_COLS"] = cols
            col += cols
        bmm1_cols = col
        col = 0
        for name, cols in (("P", cfg.SF_TMEM_COLS_P), ("dOT", cfg.SF_TMEM_COLS_dOT)):
            sf_fields[f"SF_{name}_OFF"] = col
            sf_fields[f"SF_{name}_COLS"] = cols
            col += cols
        return TmemLayout(
            TOTAL_COLS=TMEM_TOTAL_COLS,
            S_OFF=0,
            S_COLS=s_cols,
            dP_OFF=s_cols,
            dP_COLS=dp_cols,
            dV_OFF=dv_off,
            dV_COLS=dv_cols,
            P_OFF=tail,
            P_COLS=p_cols,
            RSVD_OFF=TMEM_TOTAL_COLS,
            RSVD_COLS=0,
            SF_BMM1_COLS=bmm1_cols,
            SF_BMM2_COLS=col,
            **sf_fields,
        )
    if cfg.IS_FP8:
        # fp8 P ring owns the tail; nothing reserved.
        return TmemLayout(
            TOTAL_COLS=TMEM_TOTAL_COLS,
            S_OFF=0,
            S_COLS=s_cols,
            dP_OFF=s_cols,
            dP_COLS=dp_cols,
            dV_OFF=dv_off,
            dV_COLS=dv_cols,
            P_OFF=tail,
            P_COLS=p_cols,
            RSVD_OFF=TMEM_TOTAL_COLS,
            RSVD_COLS=0,
        )
    # f16: P inside S_acc, base column = one wg's P half (P_COLS // SOFTMAX_WARPGROUPS)
    # so that wg1's half starts exactly at the S-read q-half boundary.
    return TmemLayout(
        TOTAL_COLS=TMEM_TOTAL_COLS,
        S_OFF=0,
        S_COLS=s_cols,
        dP_OFF=s_cols,
        dP_COLS=dp_cols,
        dV_OFF=dv_off,
        dV_COLS=dv_cols,
        P_OFF=p_cols // cfg.SOFTMAX_WARPGROUPS,
        P_COLS=p_cols,
        RSVD_OFF=tail,
        RSVD_COLS=((cfg.TILE_K // 2) * cfg.BPE) // 4 if cfg.K_SPLIT_UTCCP else 0,
    )


# ---------------------------------------------------------------------------
# SMEM buffer table in DECLARATION ORDER -- the kernel must declare its
# ``cutlass.Array(space=smem)`` slabs in exactly this order, or the desc-root
# tally below describes a layout the kernel does not have.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmemSlab:
    """One 1024-B-aligned ``cutlass.Array`` of the kernel's SharedStorage.

    ``roots`` are the tcgen05 ``build()`` roots inside it -- every ring stage
    start of a descriptor-read tile, plus the UTCCP ``.shifted()`` source --
    each as ``(label, absolute byte offset)``.  Lane-written / TMA-stored
    staging (dS ring, stats ring, the dV alias) has none: no descriptor reads
    it, so it may sit past the 256 KiB line under a version-0 descriptor.
    """

    name: str
    offset: int
    nbytes: int
    roots: Tuple[Tuple[str, int], ...] = ()


def _align_slab(nbytes: int) -> int:
    return (nbytes + _SMEM_SLAB_ALIGN - 1) // _SMEM_SLAB_ALIGN * _SMEM_SLAB_ALIGN


def smem_layout(cfg: CfgBwdD256) -> Tuple[SmemSlab, ...]:
    """The SharedStorage in declaration order, with every descriptor root.

    fp8 body (``sQ | sdO | sdOdv | sExcl[K | V] | sStats | sdS``):
      Q ring 3 x 16 = 48 KiB | dO ring 48 | dO_dv ring 48 | K 32 + V 32 = 64
      (the dV staging ALIASES it post-loop: max(K + V, dV @ BPE_O)) | stats 2 |
      dS ring 3 x 16 (e4m3, the shipped DTYPE_DS) = 48  -> 258 KiB; the bf16-dS
      twin's ring is 3 x 32 = 96 -> 306 KiB.  Every root < 208 KiB.  At
      ``DTYPE_O = FP32`` (the GQA fold's per-Q-head dV partial) the alias slab is
      the 128 KiB fp32 dV staging: 322 KiB with the e4m3 ring (1 KiB under the
      325 KiB usable), 370 with the bf16-dS twin's -- which is why the twin keeps
      bf16 partials; the roots are unchanged (K / V sit at the slab's start).

    f16 body (``sQ | sdO | sCombined[sdOdv_s0 | K | V] | sStats | sdS``):
      Q ring 2 x 32 = 64 | dO ring 64 | dO_dv stage 0 32 + K 64 + V 64 = 160
      (dO_dv stage 1 aliases the K back-half freed by the UTCCP -- the ring
      stride is ``dOBufferElems + K_BACK_OFF_ELEMS``; dV staging aliases
      K + V post-loop) | stats 2 | dS ring 1 x 32 = 32  -> 322 KiB.  The last
      root is V at 224 KiB; sStats (288) and sdS (290) sit past the 256 KiB
      line but nothing descriptor-reads them.

    mxfp8 body (``sQ | sdO | sdOdv | sExcl[K | V] | sStats | sK_SF | sV_SF |
    sP_SF | sQ_SF | sdO_SF | sdOT_SF | sdS_SF | sdS`` [P-b: ``| sdS_kv``]): the
    fp8 body's first five slabs (258 - 48 = 210 KiB), then EVERY descriptor-fed
    slab BEFORE ``sdS`` -- the six SF slabs (UTCCP sources: K 1 | V 1 | P 1 (512 B
    padded) | Q 3 x 1 | dO 3 x 1 | dO_T 3 x 1 = 12 KiB) -- so every root stays
    under the 256 KiB version-0 window (highest ``sdOT_SF[2]`` at 226304 B = 221
    KiB); the e4m3 P ring is in TMEM (no slab, no root); the dS SF staging (P-a:
    3 stages x 2 atoms x 512 B = 3 KiB -- one atom per GEMM orientation) and the
    dS ring(s) carry no root.  P-a 273 KiB of slabs (275 with the scaffold, 52 KiB
    free); P-b (the DS_SF_POLICY_DEFAULT arm): two 2-stage e4m3 payload rings
    (sdS = ds_dk 32 + sdS_kv = ds_dq 32) + 2 KiB staging (2 stages x 2 atoms x
    512 B) = 288 (290, 37 KiB free; no rcp gather slab is
    declared -- BufferElems.rcpGatherElems); P-c (the bf16-dS oracle twin): one
    2-stage bf16 ring 64, no SF staging = 286 (288, 39 KiB free).
    """
    b = buffer_elems(cfg)
    slabs = []
    off = 0

    def _push(name, nbytes, roots=()):
        nonlocal off
        slab = SmemSlab(name, off, _align_slab(nbytes), tuple((lbl, off + r) for lbl, r in roots))
        slabs.append(slab)
        off += slab.nbytes

    q_stage = b.qBufferElems * cfg.BPE
    do_stage = b.dOBufferElems * cfg.BPE
    k_bytes = b.kBufferElems * cfg.BPE
    v_bytes = b.vBufferElems * cfg.BPE
    dv_bytes = b.dVBufferElems * cfg.BPE_O
    kv_alias = max(k_bytes + v_bytes, dv_bytes)

    _push("sQ", cfg.STAGES_Q * q_stage, [(f"sQ[{s}]", s * q_stage) for s in range(cfg.STAGES_Q)])
    _push("sdO", cfg.STAGES_dO * do_stage, [(f"sdO[{s}]", s * do_stage) for s in range(cfg.STAGES_dO)])
    if cfg.K_SPLIT_UTCCP:
        # [sdOdv_s0 | K | V]; sdO_dv[1] == sK_back (alias), UTCCP source = sK_back.
        k_off = do_stage
        k_back_off = k_off + b.K_BACK_OFF_ELEMS * cfg.BPE
        roots = [("sdO_dv[0]", 0), ("sK", k_off), ("sK_back(UTCCP src)", k_back_off), ("sV", k_off + k_bytes)]
        if cfg.STAGES_dO_DV > 1:
            roots.append(("sdO_dv[1](=sK_back alias)", b._DODV_STAGE_ELEMS * cfg.BPE))
        _push("sCombined[sdOdv_s0|K|V](+sdV alias)", do_stage + kv_alias, roots)
    else:
        _push("sdOdv", cfg.STAGES_dO_DV * do_stage, [(f"sdO_dv[{s}]", s * do_stage) for s in range(cfg.STAGES_dO_DV)])
        _push("sExcl[K|V](+sdV alias)", kv_alias, [("sK", 0), ("sV", k_bytes)])
    _push("sStats", cfg.STATS_STAGES * b.STATS_SLOT_ELEMS * 4)
    if cfg.IS_MXFP8:
        # Every UTCCP-descriptor source BEFORE sdS (rules/mma-tma-matrix.md S6): the six SF slabs (the P ring is in TMEM).
        _push("sK_SF", cfg.SF_SMEM_K, [("sK_SF", 0)])
        _push("sV_SF", cfg.SF_SMEM_V, [("sV_SF", 0)])
        _push("sP_SF", cfg.SF_SMEM_P, [("sP_SF", 0)])
        _push("sQ_SF", cfg.STAGES_Q * cfg.SF_SMEM_Q, [(f"sQ_SF[{s}]", s * cfg.SF_SMEM_Q) for s in range(cfg.STAGES_Q)])
        _push("sdO_SF", cfg.STAGES_dO * cfg.SF_SMEM_dO, [(f"sdO_SF[{s}]", s * cfg.SF_SMEM_dO) for s in range(cfg.STAGES_dO)])
        _push("sdOT_SF", cfg.STAGES_dO_DV * cfg.SF_SMEM_dOT, [(f"sdOT_SF[{s}]", s * cfg.SF_SMEM_dOT) for s in range(cfg.STAGES_dO_DV)])
        if b.dSSfStagingBytes:
            _push("sdS_SF", cfg.XFER_STAGES * b.dSSfStagingBytes)
    _push("sdS", cfg.XFER_STAGES * b.dSBufferElems * cfg.BPE_DS)
    if cfg.IS_MXFP8:
        if cfg.DS_PAYLOADS > 1:
            # P-b: the second e4m3 payload ring (ds_dq: dS quantized per 32-kv block, the dQ GEMM's A operand; sdS itself is
            # ds_dk, per 32-q block); TMA-stored only, the same 128-B-row Swizzle(3,4,3) geometry as sdS.
            _push("sdS_kv", cfg.XFER_STAGES * b.dSBufferElems * cfg.BPE_DS)
        if b.rcpGatherElems:
            _push("sRcp", b.rcpGatherElems * 4)
    return tuple(slabs)


def smem_bytes(cfg: CfgBwdD256) -> int:
    """Slab bytes per CTA (the ``cutlass.Array`` allocations), scaffolding excluded."""
    return sum(s.nbytes for s in smem_layout(cfg))


def kernel_smem_bytes(cfg: CfgBwdD256) -> int:
    """Per-CTA SMEM the launcher must provision: slabs + the scaffold budget."""
    return smem_bytes(cfg) + SMEM_SCAFFOLD_BYTES


def desc_roots(cfg: CfgBwdD256) -> Tuple[Tuple[str, int], ...]:
    """Every tcgen05 SMEM-descriptor ``build()`` root, ``(label, byte offset)``."""
    return tuple(r for s in smem_layout(cfg) for r in s.roots)


def _needs_desc_v1(cfg: CfgBwdD256) -> bool:
    """True when any descriptor root starts at or past the 14-bit v0 window."""
    return any(off >= TCGEN05_V0_ADDR_LIMIT for _, off in desc_roots(cfg))


def desc_version(cfg: CfgBwdD256) -> int:
    """The ``desc_version=`` every ``SmemTile`` in the kernel must take (bind it
    ONCE as the module constant ``DESC_VERSION``; never a per-tile literal)."""
    return 1 if _needs_desc_v1(cfg) else 0


# ---------------------------------------------------------------------------
# Scaffolding: the mbarrier inventory + scheduler + TMEM pointer
# ---------------------------------------------------------------------------


def mbar_stage_counts(cfg: CfgBwdD256) -> Dict[str, int]:
    """``Bars`` field -> stage count, in the pre-port bodies' order.

    The kernel's ``Bars`` must allocate exactly these (the barrier-inspector
    audits init counts against this list).  ``mb_s_acc_empty`` exists only on
    the fp8-class bodies (per-tensor fp8 and MXFP8; the f16 body's in-order MMA
    covers the S WAR); ``mb_k_utccp_done`` only on the f16 body (the K-split
    alias seam).  The MXFP8 body adds ONE mbarrier, ``mb_p_sf_consumed`` (1
    stage; an MMA_COMMIT multicast after every P.dO, waited by every softmax lane
    of each CTA before its P store): its scale-factor atoms alias the P-ring
    slot the softmax writes NEXT, and the in-order tcgen05 stream cannot order
    another thread's ``tcgen05.st`` after the MMA that reads those columns.  Its
    SF loads ride the ``_full`` bars (the tx grows), and ``mb_p_ready`` keeps the
    fp8 body's ``STAGES_TMEM_P`` stages with no ``p_empty`` -- slot reuse is
    ordered by ``mb_s_acc_full[i+2]`` as on the fp8 body AND by
    ``mb_p_sf_consumed``.  The 24-ring inventory of the SMEM-P-ring predecessor
    was MEASURED on Rubin 2026-09-30 (48 / 0 / 0 exit 0 / 124 / other over 12
    fresh processes x n_q_tiles 1 / 2 / 3 / 4); the 25-ring one awaits its own
    fresh-process count (sdpa-invariants S6: agreement with a sibling config is
    transcription, not evidence).
    """
    rings = {
        "mb_q_full": cfg.STAGES_Q,
        "mb_q_empty": cfg.STAGES_Q,
        "mb_do_full": cfg.STAGES_dO,
        "mb_do_empty": cfg.STAGES_dO,
        "mb_dodv_full": cfg.STAGES_dO_DV,
        "mb_dodv_empty": cfg.STAGES_dO_DV,
        "mb_k_full": cfg.STAGES_KV,
        "mb_k_empty": cfg.STAGES_KV,
        "mb_v_full": cfg.STAGES_KV,
        "mb_v_empty": cfg.STAGES_KV,
    }
    if cfg.K_SPLIT_UTCCP:
        rings["mb_k_utccp_done"] = 1
    rings["mb_s_acc_full"] = cfg.STAGES_TMEM_S
    if cfg.IS_FP8:
        rings["mb_s_acc_empty"] = cfg.STAGES_TMEM_S
    rings["mb_dp_full"] = cfg.STAGES_TMEM_S
    rings["mb_dp_empty"] = cfg.STAGES_TMEM_S
    rings["mb_p_ready"] = cfg.STAGES_TMEM_P
    if cfg.IS_MXFP8:
        # softmax's P store of iteration n+1 may not enter slot (n+1)%2 before P.dO[n] finished reading its SF atoms there.
        rings["mb_p_sf_consumed"] = 1
    rings.update(
        {
            "mb_stats_full": cfg.STATS_STAGES,
            "mb_stats_empty": cfg.STATS_STAGES,
            "mb_ds_smem_full": cfg.XFER_STAGES,
            "mb_ds_smem_empty": cfg.XFER_STAGES,
            "mb_dv_ready": 1,
            "mb_dv_acc_empty": 1,
            "mb_dv_stg_full": 1,
            "mb_dv_stg_empty": 1,
            "mb_tmem_dealloc": 1,
        }
    )
    return rings


_MBAR_ARRAY_ALIGN = 16  # cutlass.Array(Int64, n, alignment=16)
_SCHED_RESPONSE_WORDS = 8  # Int32 words per scheduler stage (16-B CLC response + slack)
_TMEM_PTR_BYTES = 16  # cutlass.Array(Int32, 1, alignment=16)


def scaffold_bytes_declared(cfg: CfgBwdD256) -> int:
    """Bytes the non-slab allocations declare: mbarriers (8 B each, arrays
    16-B aligned), the scheduler's two rings + response slots, the TMEM
    pointer.  An upper bound on what the DSL lays out; pinned under
    ``SMEM_SCAFFOLD_BYTES`` by the validator."""

    def _arr(n_qwords):
        return (8 * n_qwords + _MBAR_ARRAY_ALIGN - 1) // _MBAR_ARRAY_ALIGN * _MBAR_ARRAY_ALIGN

    mbars = sum(_arr(n) for n in mbar_stage_counts(cfg).values())
    sched = 2 * _arr(cfg.SCHEDULER_STAGES) + cfg.SCHEDULER_STAGES * _SCHED_RESPONSE_WORDS * 4
    return mbars + sched + _TMEM_PTR_BYTES


# ---------------------------------------------------------------------------
# Scheduler-ring arriver count -- derived from the body's call sites
# ---------------------------------------------------------------------------


def read_tile_arrivers_tot(cfg: CfgBwdD256) -> int:
    """Per-CTA init count of ``mb_read_tile_id``.

    ``read_tile_id_arrive(mb, CGA_SIZE)`` lands exactly ONE arrive on EVERY
    CTA of the cluster per calling warp, so each CTA's count is the number of
    warps calling it cluster-wide.  In BOTH bodies the persistent-loop warps
    that call it are: the ``SOFTMAX_WARPGROUPS * SOFTMAX_WG_WARPS`` compute
    warps (``_softmax_warp_group``), the TMA-LDG warp (``_tmaldg_warp``) and
    the TMA-STG warp (``_tmastg_warp``) -- on every CTA -- plus the MMA warp
    on the pair LEADER only (``_mma_warp``; the follower runs
    ``_mma_warp_quiet``, which has no persistent loop and no arrive).  The
    scheduler warp never arrives on its own ring.

        leader   : 8 + MMA + TMALDG + TMASTG = 11
        follower : 8 +       TMALDG + TMASTG = 10
        cluster  : 21 == CTA_MMA * (softmax + 2) + 1

    Too LOW is the dangerous direction: the ring completes before every role
    has read the payload and advances a tile early; too HIGH is an
    unreachable count = a hang at every shape.  Both are silent (P3).
    """
    compute = cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_WG_WARPS
    return cfg.CTA_MMA * (compute + 2) + 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check(preds) -> None:
    for ok, msg in preds:
        if not ok:
            raise ValueError(msg)


def _validate_params(flavor: str, family: str, params: _BwdTemplateParams, datapath_2x2: bool = False) -> None:
    """Guard the record a Rubin d256 backward body can express.  Every raise
    here must also be a Capabilities decline -- reaching it is an engine-row bug.
    ``datapath_2x2`` is passed by ``config_d256_2x2.make_cfg_d256_2x2`` alone (it validates the profile itself); the
    4x1 families reject any non-zero ``datapath_2x2_profile``."""
    dtype_o = getattr(params, "dtype_o", -1)
    dtype_ds = getattr(params, "dtype_ds", -1)
    ds_sf_policy = getattr(params, "ds_sf_policy", -1)
    profile_2x2 = getattr(params, "datapath_2x2_profile", 0)
    if not datapath_2x2 and profile_2x2 != 0:
        raise ValueError(
            f"{flavor}: datapath_2x2_profile={profile_2x2} selects the 2x2-datapath body (kernels/bprop_d256_2x2_f16.py, config_d256_2x2.make_cfg_d256_2x2); "
            f"the 4x1 bodies render profile 0 only and would silently ignore it"
        )
    if params.dtype_qkv not in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"{flavor}: dtype_qkv must be a tile_dsl DTYPE_* code (E4M3=0 E5M2=1 BF16=2 FP16=3); got {params.dtype_qkv}")
    if family != FAMILY_MXFP8:
        # The MXFP8-only record fields are rejected, not ignored: a flag the body does not read is a claim it cannot honour.
        if ds_sf_policy not in (-1, DS_SF_NONE):
            raise ValueError(
                f"{flavor}: ds_sf_policy is the MXFP8 family's (dS block scale factors, DS_SF_P_A/P_B/P_C); the f16 / per-tensor fp8 bodies "
                f"write no dS scale factors -- got {ds_sf_policy}"
            )
        if getattr(params, "scaled_fp8_pack", False):
            raise ValueError(
                f"{flavor}: scaled_fp8_pack is the MXFP8 body's Rubin fused cvt arm (fp32_to_fp8_pack_scaled); the f16 / per-tensor fp8 bodies "
                f"do not call it, so the flag would be silently ignored"
            )
        if getattr(params, "mask_q_pad", False):
            raise ValueError(
                f"{flavor}: mask_q_pad is the MXFP8 body's q-pad band (P select-zero on q >= seqlen_q_real against the Q-side SF pad bytes); the "
                f"f16 / per-tensor fp8 bodies have no such arm -- their q pad rows are LSE +inf-padded by the adapter instead"
            )
    if family == FAMILY_MXFP8:
        if params.dtype_qkv != DTYPE_E4M3:
            raise ValueError(
                f"{flavor}: the MXFP8 body takes E4M3 payloads with F8_128x4 E8M0 scale factors (dtype_qkv={DTYPE_E4M3}); got {params.dtype_qkv}"
                + (
                    " -- E5M2 payloads are not implemented in this body"
                    if params.dtype_qkv == DTYPE_E5M2
                    else " -- a half-precision io belongs to the f16 body"
                )
            )
        if dtype_o not in (-1, DTYPE_BF16, DTYPE_FP16):
            raise ValueError(
                f"{flavor}: dtype_o must be -1 (inherit -> BF16), DTYPE_BF16 or DTYPE_FP16 -- the MXFP8 backward graph's gradients are half precision "
                f"(the adapter passes the graph's dK / dV dtype; no quantized gradient, no amax); got {dtype_o}"
            )
        if ds_sf_policy not in (-1,) + _DS_SF_POLICIES:
            raise ValueError(
                f"{flavor}: ds_sf_policy must be -1 (DS_SF_POLICY_DEFAULT = {_DS_SF_POLICY_NAME[DS_SF_POLICY_DEFAULT]}, the shipped block-scaled "
                f"chain; P-c is the bf16-dS oracle twin) or one of DS_SF_P_A/P_B/P_C ({DS_SF_P_A}/{DS_SF_P_B}/{DS_SF_P_C}): "
                f"32x32 tile scale (optional validation only) / exact 1x32 both ways (two payloads) / bf16 dS; got {ds_sf_policy}"
            )
        policy = DS_SF_POLICY_DEFAULT if ds_sf_policy < 0 else ds_sf_policy
        want_ds = DTYPE_BF16 if policy == DS_SF_P_C else DTYPE_E4M3
        if dtype_ds not in (-1, want_ds):
            what = (
                "the bf16 dS the bf16 renderings read over dequantized Q_T / K_T (no dS quantization)"
                if policy == DS_SF_P_C
                else "an e4m3 dS payload + E8M0 scale factors for the block-scale stage-3 GEMM arm"
            )
            raise ValueError(
                f"{flavor}: dtype_ds must be -1 or {'DTYPE_BF16' if want_ds == DTYPE_BF16 else 'DTYPE_E4M3'} ({want_ds}) under {_DS_SF_POLICY_NAME[policy]}, "
                f"which writes {what}; got {dtype_ds}"
            )
    elif family == FAMILY_FP8:
        if dtype_ds not in (-1, DTYPE_E4M3, DTYPE_BF16):
            raise ValueError(
                f"{flavor}: dtype_ds must be -1 (inherit -> E4M3: dS_q = e4m3(dS * scale_dP) for the fp8 K64 GEMM arm), DTYPE_E4M3 or "
                f"DTYPE_BF16 (the pre-quantized dS the bf16 GEMM renderings read unchanged -- the A/B twin); got {dtype_ds}"
            )
        if params.dtype_qkv != DTYPE_E4M3:
            raise ValueError(
                f"{flavor}: the fp8 body is E4M3-only (dtype_qkv={DTYPE_E4M3}); got {params.dtype_qkv}"
                + (" -- E5M2 is not implemented in this body" if params.dtype_qkv == DTYPE_E5M2 else " -- a half-precision io belongs to the f16 body")
            )
        if dtype_o not in (-1, DTYPE_E4M3, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32):
            raise ValueError(
                f"{flavor}: dtype_o must be -1 (inherit -> E4M3, the fp8 graph contract), DTYPE_E4M3, DTYPE_BF16 or DTYPE_FP16 "
                f"(the pre-quantization output for the bitwise A/B) or DTYPE_FP32 (the per-Q-head dV_true partial a GQA fold sums in fp32 and "
                f"rounds ONCE; fits the 327 KiB cap with the e4m3 dS ring only); got {dtype_o}"
            )
    else:
        if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
            raise ValueError(
                f"{flavor}: the f16 body takes DTYPE_BF16 ({DTYPE_BF16}) or DTYPE_FP16 ({DTYPE_FP16}); got {params.dtype_qkv} -- an FP8 io belongs to the fp8 body"
            )
        if dtype_o not in (-1, DTYPE_BF16, DTYPE_FP16):
            raise ValueError(f"{flavor}: dtype_o must be -1 (inherit the io dtype), DTYPE_BF16 or DTYPE_FP16 on the f16 body; got {dtype_o}")
        if dtype_ds not in (-1, params.dtype_qkv):
            raise ValueError(
                f"{flavor}: dtype_ds must be -1 or the io dtype ({params.dtype_qkv}) on the f16 body -- its bf16 / fp16 stage-3 GEMMs read the workspace "
                f"as the io dtype; got {dtype_ds}"
            )
    # --- band -----------------------------------------------------------------
    if params.window_left is not None and params.window_left <= 0:
        raise ValueError(
            f"{flavor}: SWA requires window_left > 0 (got {params.window_left}); the transposed q-tile trim widens the "
            f"q range by SWA_WINDOW and a zero or negative window is not a sliding window"
        )
    if params.window_right is not None and params.window_right != 0:
        raise ValueError(
            f"{flavor}: window_right must be 0 when set (plain causal diagonal); got {params.window_right}. Right-band widening "
            f"is not implemented in this body -- its q-tile trim and per-cell mask know only kv <= q (+ the bottom-right shift)"
        )
    if params.bottom_right and params.window_right is None:
        raise ValueError(f"{flavor}: bottom_right alignment requires a causal band (window_right set)")
    # --- padding / THD ------------------------------------------------------------
    if params.seq_q_lens_present:
        raise ValueError(f"{flavor}: seq_q_lens_present is not implemented -- the body threads only the per-batch kv length (seq_kv_lens)")
    # thd_varlen is admitted for every family: all three bodies carry the THD arm (packed operands, the metadata buffer,
    # the device claim counter; the MXFP8 body also the packed per-sequence-tile-padded scale factors).
    if params.thd_varlen and (params.seq_kv_lens_present or params.seq_q_lens_present):
        raise ValueError(
            f"{flavor}: thd_varlen is mutually exclusive with seq_kv_lens_present / seq_q_lens_present -- THD carries its per-sequence "
            f"lengths in the metadata buffer (two sources of truth for the same fact drift apart)"
        )
    if params.thd_varlen and params.sched_policy != SCHED_NATURAL:
        raise ValueError(
            f"{flavor}: thd_varlen requires sched_policy NATURAL (0): the THD work list is a device claim counter over "
            f"(sequence, kv block, head) units walked longest-sequence-first by the metadata's batch_remap; the LPT / LPT_L2 flat-grid "
            f"decodes address the dense envelope; got {params.sched_policy}"
        )
    # --- schedule -----------------------------------------------------------------
    if params.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2):
        raise ValueError(f"{flavor}: sched_policy must be one of NATURAL/LPT/LPT_L2 (0/1/2); got {params.sched_policy}")


def _mask_flags_from(params: _BwdTemplateParams) -> int:
    flags = MASK_NONE
    if params.window_right is not None:
        flags |= MASK_CAUSAL
    if params.window_left is not None:
        flags |= MASK_SWA
    if params.seq_kv_lens_present or params.thd_varlen:
        # THD is padded BY CONSTRUCTION: every sequence has its own S_q / S_kv, so the kv tail
        # rows (row_dead) and the q pad columns (the THD q band) take the per-cell mask exactly
        # as a padding mask does; the lengths come from the metadata buffer instead.
        flags |= MASK_PADDED
    return flags


def _validate_cfg_d256_bwd(cfg: CfgBwdD256, flavor: str) -> None:
    """Every predicate the bodies hardcode.  Each message names the failure
    SIGNATURE so the next reader recognises it."""
    b = buffer_elems(cfg)
    tm = tmem_layout(cfg)
    compute_warps = cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_WG_WARPS
    reg_total = (
        compute_warps * cfg.SOFTMAX_REGS + cfg.CORRECTION_WARPS * cfg.CORRECTION_REGS + cfg.MMA_REGS + cfg.TMALDG_REGS + cfg.TMASTG_REGS + cfg.SCHEDULER_REGS
    )
    # --- register split: all four are hardware constraints (12-warp form) -----
    _check(
        [
            (
                cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS,
                f"{flavor}: MMA/TMALDG/TMASTG/SCHEDULER regs must be equal (HW equality constraint); got "
                f"{cfg.MMA_REGS}/{cfg.TMALDG_REGS}/{cfg.TMASTG_REGS}/{cfg.SCHEDULER_REGS} -- a violation is wrong results or a crash with no diagnostic",
            ),
            (
                reg_total <= reg_entry_pool(cfg.TOTAL_WARPS),
                f"{flavor}: register split {reg_total} over the {cfg.TOTAL_WARPS}-warp ENTRY pool {reg_entry_pool(cfg.TOTAL_WARPS)} "
                f"(= {reg_entry_pool(cfg.TOTAL_WARPS) // cfg.TOTAL_WARPS}/thread at launch; the 65536/32 = {REG_BUDGET_PER_CTA} register file is NOT the bound) "
                f"({compute_warps} softmax @ {cfg.SOFTMAX_REGS} + 4 service @ {cfg.MMA_REGS}) -- setmaxnreg.inc can only take what .dec released: "
                f"the last softmax warp parks forever = a HANG at 100 % SM utilization at every shape, no fault, no message",
            ),
            (
                all(r % 8 == 0 for r in (cfg.MMA_REGS, cfg.SOFTMAX_REGS, cfg.OTHER_REGS, cfg.CORRECTION_REGS)),
                f"{flavor}: every per-role register count must be a multiple of 8 (setmaxnreg granularity)",
            ),
            (
                all(24 <= r <= 256 for r in (cfg.MMA_REGS, cfg.SOFTMAX_REGS, cfg.OTHER_REGS)),
                f"{flavor}: per-warp register counts must be within 24..256",
            ),
            (
                cfg.OTHER_REGS == cfg.MMA_REGS,
                f"{flavor}: OTHER_REGS is the service-warp count the body's setmaxregister(DECREASE) uses; it must equal MMA_REGS",
            ),
        ]
    )
    # --- warp population + scheduler ring ---------------------------------------
    _check(
        [
            (
                cfg.SOFTMAX_WARPGROUPS == 2 and cfg.SOFTMAX_WG_WARPS == 4 and cfg.CORRECTION_WARPS == 0,
                f"{flavor}: the body is 8 compute warps in 2 warpgroups of 4 (q-halves) and no correction warps; got "
                f"{cfg.SOFTMAX_WARPGROUPS}x{cfg.SOFTMAX_WG_WARPS} + {cfg.CORRECTION_WARPS}",
            ),
            (
                cfg.TOTAL_WARPS == compute_warps + cfg.CORRECTION_WARPS + 4,
                f"{flavor}: TOTAL_WARPS must be compute + correction + 4 service warps; got {cfg.TOTAL_WARPS}",
            ),
            (cfg.THREADS_PER_CTA == cfg.TOTAL_WARPS * 32, f"{flavor}: THREADS_PER_CTA must be TOTAL_WARPS*32"),
            (
                cfg.SOFTMAX_WG0_BASE == 0
                and cfg.SOFTMAX_WG1_BASE == cfg.SOFTMAX_WG_WARPS
                and cfg.MMA_WARP_ID == compute_warps + cfg.CORRECTION_WARPS
                and cfg.TMALDG_WARP_ID == cfg.MMA_WARP_ID + 1
                and cfg.TMASTG_WARP_ID == cfg.MMA_WARP_ID + 2
                and cfg.SCHED_WARP_ID == cfg.MMA_WARP_ID + 3 == cfg.TOTAL_WARPS - 1,
                f"{flavor}: warp ids must be wg0 @0, wg1 @{cfg.SOFTMAX_WG_WARPS}, MMA/TMALDG/TMASTG/SCHED consecutive after the compute warps "
                f"(the body's role dispatch is `warp_idx < compute` then == each id)",
            ),
            (
                cfg.READ_TILE_ARRIVERS_TOT == read_tile_arrivers_tot(cfg),
                f"{flavor}: READ_TILE_ARRIVERS_TOT must be {read_tile_arrivers_tot(cfg)} for this body (got {cfg.READ_TILE_ARRIVERS_TOT}) = "
                f"CTA_MMA * (compute warps + TMALDG + TMASTG) + the leader-only MMA warp; a wrong count is an unreachable (hang at EVERY shape) "
                f"or early-completing (scheduler advances a tile early, wedges past ~12 ring wraps) mbarrier",
            ),
            (
                cfg.CGA_M * cfg.CGA_N == cfg.CTA_MMA,
                f"{flavor}: a single cga2 pair -- the cluster IS the MMA pair (CGA_M*CGA_N == CTA_MMA), no sub-group role split",
            ),
        ]
    )
    # --- mbarrier lane constants (P3) -------------------------------------------
    _check(
        [
            (cfg.ONE_LANE == 1 and cfg.ONE_WARP == 32, f"{flavor}: ONE_LANE/ONE_WARP are 1/32"),
            (cfg.SOFTMAX_WG_LANES == cfg.SOFTMAX_WG_WARPS * 32, f"{flavor}: SOFTMAX_WG_LANES must be SOFTMAX_WG_WARPS*32 (one warpgroup's lanes)"),
            (
                cfg.SOFTMAX_LANES == compute_warps * 32,
                f"{flavor}: SOFTMAX_LANES must be every compute lane of one CTA ({compute_warps}*32); got {cfg.SOFTMAX_LANES}",
            ),
            (
                cfg.SOFT_X_CTA_MMA == cfg.SOFTMAX_LANES * cfg.CTA_MMA,
                f"{flavor}: SOFT_X_CTA_MMA (the LEADER-scope init count) must be SOFTMAX_LANES*CTA_MMA -- every compute lane of BOTH CTAs arrives on the leader",
            ),
            (cfg.MMA_COMMIT_ARRIVES == 1, f"{flavor}: one tcgen05.commit multicast is ONE arrive per target CTA (from one elected lane)"),
        ]
    )
    # --- geometry the bodies hardcode ----------------------------------------------
    _check(
        [
            (cfg.TILE_M == 128 and cfg.TILE_N == 128, f"{flavor}: TILE_M = TILE_N = 128 (kv rows per CTA, q cols per q-tile)"),
            (
                cfg.TILE_K == 256 and cfg.TILE_O == 256,
                f"{flavor}: d_qk = d_v = 256 (got TILE_K={cfg.TILE_K}, TILE_O={cfg.TILE_O}); the body is exact-d, no envelope",
            ),
            (cfg.CGA_M == 2 and cfg.CGA_N == 1 and cfg.CTA_MMA == 2, f"{flavor}: single cga2 sub-group (CGA_M=2, CGA_N=1, CTA_MMA=2) only"),
            (cfg.TILES_Q == 1, f"{flavor}: TILES_Q == 1"),
            (cfg.SCHEDULER_STAGES == 2, f"{flavor}: SCHEDULER_STAGES == 2 (the CLC response ring depth the scheduler warp loop assumes)"),
            (cfg.N_BMM2_CHUNKS * cfg.BMM2_CHUNK_SIZE == cfg.TILE_N, f"{flavor}: N_BMM2_CHUNKS*BMM2_CHUNK_SIZE must equal TILE_N"),
            (
                cfg.TILE_N % cfg.SOFTMAX_WARPGROUPS == 0 and b._SMX_CHUNK % b._LDTM_NUM == 0,
                f"{flavor}: the softmax q-half (TILE_N/SOFTMAX_WARPGROUPS) must be a multiple of the LDTM x{b._LDTM_NUM} granule",
            ),
            (cfg.TILE_N % b._DSOFT_CHUNK == 0, f"{flavor}: TILE_N must be a multiple of the dSoftmax chunk ({b._DSOFT_CHUNK})"),
        ]
    )
    # --- dtype / k-step / idesc -----------------------------------------------------
    _check(
        [
            (cfg.IS_FP8 == int(_is_fp8_code(cfg.DTYPE_QKV)), f"{flavor}: IS_FP8 must follow DTYPE_QKV"),
            (
                cfg.BPE == bpe(cfg.DTYPE_QKV) and cfg.BPE_O == bpe(cfg.DTYPE_O) and cfg.BPE_DS == bpe(cfg.DTYPE_DS),
                f"{flavor}: BPE/BPE_O/BPE_DS must match their dtypes",
            ),
        ]
    )
    # --- MXFP8 bookkeeping: the SF fields are the MXFP8 body's alone ------------------------------------------------------
    if not cfg.IS_MXFP8:
        nonzero = [f for f in _MXFP8_ZERO_FIELDS if getattr(cfg, f) != 0] + (["DS_PAYLOADS"] if cfg.DS_PAYLOADS != 1 else [])
        if nonzero:
            raise ValueError(
                f"{flavor}: the f16 / per-tensor fp8 bodies load no block scale factors and write no dS scale factors: every MXFP8 field must be 0 "
                f"(IS_MXFP8, SF_*, *_SF_TX, P_SCALE_LOG2 / P_SF_BYTE, DS_SF_POLICY / DS_SF_ATOMS, SCALED_FP8_PACK, MASK_Q_PAD) and "
                f"DS_PAYLOADS 1 -- a nonzero value is a claim about scale factors the body does not honour; got nonzero {nonzero}"
            )
    if cfg.IS_FP8:
        # The fp8-CLASS predicates: E4M3 payloads on Rubin's K64 path, the lookahead pipeline's 3-deep rings.  Shared by
        # the per-tensor fp8 body and the MXFP8 one (everything the MXFP8 body does not change is the per-tensor pipeline).
        _check(
            [
                (cfg.DTYPE_QKV == DTYPE_E4M3, f"{flavor}: the fp8-class bodies are E4M3-only (no E5M2 arm); got DTYPE_QKV={cfg.DTYPE_QKV}"),
                (
                    cfg.TILE_K_HW_BMM1 == 64 and cfg.TILE_K_HW_BMM2 == 64 and cfg.IDESC_K_DIM == 1,
                    f"{flavor}: Rubin dense FP8 runs the K=64 path and EVERY idesc must pass k_dim=1 (got TILE_K_HW {cfg.TILE_K_HW_BMM1}/{cfg.TILE_K_HW_BMM2}, "
                    f"IDESC_K_DIM={cfg.IDESC_K_DIM}); a mismatch silently scrambles accumulator ROWS (some exact, some 0.5x-2x, sign flips), passes a 1-tile "
                    f"shape and fails wide K (rules/mma-tma-matrix.md S1 -- arch-OPPOSITE of Blackwell's K32/k_dim=0)",
                ),
                (
                    cfg.BMM2_CHUNK_SIZE == cfg.TILE_K_HW_BMM2,
                    f"{flavor}: the fp8 BMM2 P chunk must be the K=64 hardware k-step (BMM2_CHUNK_SIZE == TILE_K_HW_BMM2)",
                ),
                (
                    b.P_D_BLOCK % b._SMX_CHUNK == 0,
                    f"{flavor}: a dS store subtile ({b.P_D_BLOCK} q cols = {cfg.dS_SWZ_BYTES} B at BPE_DS={cfg.BPE_DS}) must be whole warpgroup q halves "
                    f"({b._SMX_CHUNK} cols): each warpgroup store_swizzled's its half at col_in_blk = q_half % P_D_BLOCK inside the 128-B swizzle row",
                ),
                (
                    cfg.K_SPLIT_UTCCP == 0,
                    f"{flavor}: the fp8 body has no BMM1 K-split, and neither has the MXFP8 one (their tail TMEM columns hold the e4m3 P ring, whose "
                    f"dead slot also hosts the MXFP8 scale-factor atoms); K_SPLIT_UTCCP must be 0",
                ),
                # ring depths the body was validated at (the lookahead needs Q[i], Q[i+1] and a prefetch in flight)
                (
                    cfg.STAGES_Q == 3 and cfg.STAGES_dO == 3,
                    f"{flavor}: Q / dO rings are 3-deep in this body (got {cfg.STAGES_Q}/{cfg.STAGES_dO}) -- the Q.K[i+1] lookahead keeps two stages live plus one prefetch; "
                    f"another depth was never validated on this body",
                ),
                (cfg.STAGES_dO_DV == cfg.STAGES_dO, f"{flavor}: this fp8-class body drives the dO_dv ring at STAGES_dO (got STAGES_dO_DV={cfg.STAGES_dO_DV})"),
            ]
        )
        if cfg.IS_MXFP8:
            _validate_cfg_mxfp8(cfg, flavor, b, tm)
        else:
            _check(
                [
                    (
                        cfg.DTYPE_DS in (DTYPE_E4M3, DTYPE_BF16),
                        f"{flavor}: the fp8 chain's dS workspace is E4M3 (dS_q = e4m3(dS * scale_dP); the stage-3 GEMMs render the fp8 K64 arm with the "
                        f"descale_dP epilogue, api_dsl_sm107 FP8_DS_DTYPE) or BF16 (the pre-quantized dS the bf16 renderings read unchanged); got "
                        f"DTYPE_DS={cfg.DTYPE_DS} -- a workspace dtype the GEMM arm does not read is silent garbage gradients",
                    ),
                    (
                        cfg.DTYPE_O in (DTYPE_E4M3, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32),
                        f"{flavor}: DTYPE_O must be E4M3 (the fp8 graph contract), BF16/FP16 (pre-quantization output) or FP32 (the GQA fold's "
                        f"per-Q-head dV_true partial, rounded once by the fold); got {cfg.DTYPE_O}",
                    ),
                    (
                        cfg.STAGES_TMEM_P == 2 and tm.P_OFF + cfg.STAGES_TMEM_P * tm.P_COLS == tm.TOTAL_COLS,
                        f"{flavor}: the fp8 P ring is exactly 2 stages of {tm.P_COLS} cols filling TMEM [{tm.P_OFF}, {tm.TOTAL_COLS}) (got STAGES_TMEM_P={cfg.STAGES_TMEM_P}); "
                        f"the body indexes P_OFF + slot*P_COLS by the mb_p_ready PipelineState, so a deeper ring reads past the allocation and a shallower one breaks "
                        f"the 'store P[i+1] before dSoftmax[i]' overlap the missing p_empty relies on",
                    ),
                    (cfg.XFER_STAGES == 3, f"{flavor}: the fp8 dS SMEM ring is 3-deep in this body (got {cfg.XFER_STAGES}); other depths were never validated"),
                ]
            )
    else:
        _check(
            [
                (cfg.DTYPE_QKV in (DTYPE_BF16, DTYPE_FP16), f"{flavor}: the f16 body takes BF16 or FP16; got DTYPE_QKV={cfg.DTYPE_QKV}"),
                (
                    cfg.TILE_K_HW_BMM1 == 16 and cfg.TILE_K_HW_BMM2 == 16 and cfg.IDESC_K_DIM == 0,
                    f"{flavor}: f16/bf16 TILE_K_HW must be 16 with the default idesc k_dim=0 (got {cfg.TILE_K_HW_BMM1}/{cfg.TILE_K_HW_BMM2}, IDESC_K_DIM={cfg.IDESC_K_DIM}); "
                    f"the 2-chunk (K=32) f16 path is silently wrong on SM10x (rules/mma-tma-matrix.md S2)",
                ),
                (
                    cfg.DTYPE_DS == cfg.DTYPE_QKV,
                    f"{flavor}: the dS workspace dtype IS the io dtype (got DTYPE_DS={cfg.DTYPE_DS}, DTYPE_QKV={cfg.DTYPE_QKV}); the stage-3 GEMMs read the workspace "
                    f"as the io dtype and a mismatch is silent garbage gradients",
                ),
                (cfg.DTYPE_O in (DTYPE_BF16, DTYPE_FP16), f"{flavor}: DTYPE_O must be BF16 or FP16 on the f16 body; got {cfg.DTYPE_O}"),
                (cfg.K_SPLIT_UTCCP == 1, f"{flavor}: the f16 body is the BMM1 K-split body (K back-half UTCCP'd to TMEM); K_SPLIT_UTCCP must be 1"),
                # ring depths the body was validated at (327 KiB cap: 2-deep Q/dO, 1-stage dS)
                (
                    cfg.STAGES_Q == 2 and cfg.STAGES_dO == 2,
                    f"{flavor}: Q / dO rings are 2-deep in this body (got {cfg.STAGES_Q}/{cfg.STAGES_dO}); a third 32 KiB stage of either does not fit the 327 KiB cap",
                ),
                (
                    cfg.STAGES_dO_DV == 2 and b.dOBufferElems == b.K_BACK_OFF_ELEMS,
                    f"{flavor}: the dO_dv ring is 2-deep with stage 1 ALIASING the K back-half freed by the UTCCP (got STAGES_dO_DV={cfg.STAGES_dO_DV}); the alias is exact "
                    f"only when one dO_dv stage ({b.dOBufferElems} elems) equals half of K ({b.K_BACK_OFF_ELEMS} elems)",
                ),
                (
                    cfg.STAGES_TMEM_P == 1 and tm.P_OFF + tm.P_COLS <= tm.S_COLS,
                    f"{flavor}: f16 P is a single buffer INSIDE S_acc ([{tm.P_OFF}, {tm.P_OFF + tm.P_COLS}) must fit [0, {tm.S_COLS})); got STAGES_TMEM_P={cfg.STAGES_TMEM_P}",
                ),
                (
                    tm.P_OFF + tm.P_COLS // cfg.SOFTMAX_WARPGROUPS == cfg.TILE_N // cfg.SOFTMAX_WARPGROUPS,
                    f"{flavor}: wg1's P half must begin exactly at the S-read q-half boundary (col {cfg.TILE_N // cfg.SOFTMAX_WARPGROUPS}) so each wg's P write stays inside its own S read",
                ),
                (
                    tm.RSVD_OFF + tm.RSVD_COLS == tm.TOTAL_COLS and tm.RSVD_COLS == ((cfg.TILE_K // 2) * cfg.BPE) // 4,
                    f"{flavor}: TMEM [{tm.RSVD_OFF}, {tm.TOTAL_COLS}) must hold exactly the UTCCP'd K back-half ((TILE_K/2)*BPE/4 = {((cfg.TILE_K // 2) * cfg.BPE) // 4} cols); got RSVD_COLS={tm.RSVD_COLS}",
                ),
                (
                    cfg.XFER_STAGES == 1,
                    f"{flavor}: the f16 dS SMEM ring is 1-deep in this body (got {cfg.XFER_STAGES}); a second 32 KiB stage does not fit the 327 KiB cap",
                ),
            ]
        )
    # --- ring depths common to both -----------------------------------------------------
    _check(
        [
            (cfg.STAGES_KV == 1, f"{flavor}: K and V are loaded ONCE per kv-block (STAGES_KV == 1); got {cfg.STAGES_KV}"),
            (
                cfg.STAGES_TMEM_S == 1,
                f"{flavor}: S and dP accumulators are single-buffered (STAGES_TMEM_S == 1); cross-iteration overlap is the MMA order, not a parity ring",
            ),
            (
                cfg.STATS_STAGES == 2,
                f"{flavor}: the lse/do_dot prefetch ring is 2-deep (got {cfg.STATS_STAGES}); the TMA-STG warp's prefetch-one-ahead assumes it",
            ),
        ]
    )
    # --- TMEM -----------------------------------------------------------------------------
    _check(
        [
            (
                tm.TOTAL_COLS == TMEM_TOTAL_COLS,
                f"{flavor}: TMEM alloc is the full {TMEM_TOTAL_COLS}-column Rubin carve (is_exclusive=True); got {tm.TOTAL_COLS}",
            ),
            (
                tm.S_OFF == 0 and tm.dP_OFF == tm.S_COLS and tm.dV_OFF == tm.dP_OFF + tm.dP_COLS,
                f"{flavor}: TMEM must be S | dP | dV back to back from column 0",
            ),
            (
                tm.S_COLS + tm.dP_COLS + tm.dV_COLS + _tmem_tail_cols(cfg, tm) == tm.TOTAL_COLS,
                f"{flavor}: TMEM column map must sum to {TMEM_TOTAL_COLS}: S {tm.S_COLS} + dP {tm.dP_COLS} + dV {tm.dV_COLS} + {_tmem_tail_desc(cfg, tm)}",
            ),
            (
                tm.P_COLS == (cfg.TILE_N * cfg.BPE) // 4,
                f"{flavor}: the P A-operand is TILE_N*BPE/4 = {(cfg.TILE_N * cfg.BPE) // 4} TMEM cols on every family; got {tm.P_COLS}",
            ),
        ]
    )
    # --- swizzles: one unit with the descriptors that read them -----------------------------
    _check(
        [
            (
                cfg.Q_SWZ_BYTES == 128 and cfg.K_SWZ_BYTES == 128 and cfg.dO_SWZ_BYTES == 128 and cfg.V_SWZ_BYTES == 128,
                f"{flavor}: Q/K/dO/V swizzle must be 128 B (d=256 rows are 256 B fp8 / 512 B f16, and the bodies' subtile walks derive from 128)",
            ),
            (
                cfg.P_SWZ_BYTES == cfg.dS_SWZ_BYTES == 128,
                f"{flavor}: the dS ring's store_swizzled pattern (P_SWZ_BYTES) and its TMA-store descriptor swizzle (dS_SWZ_BYTES) are ONE unit and must both be 128 B; "
                f"bytes stored under one swizzle and read back under another give cos ~ 0.006 with no crash",
            ),
            (cfg.dV_SWZ_BYTES == 128, f"{flavor}: the dV staging store_swizzled and its TMA-store descriptor are 128 B"),
            ((cfg.TILE_N * cfg.BPE_DS) % cfg.dS_SWZ_BYTES == 0, f"{flavor}: a dS row (TILE_N*BPE_DS bytes) must be whole swizzle atoms"),
            ((cfg.TILE_O * cfg.BPE_O) % cfg.dV_SWZ_BYTES == 0, f"{flavor}: a dV row (TILE_O*BPE_O bytes) must be whole swizzle atoms"),
        ]
    )
    # --- masks ------------------------------------------------------------------------------
    _check(
        [
            (not (cfg.CAUSAL_BOTTOM_RIGHT and not (cfg.MASK_FLAGS & MASK_CAUSAL)), f"{flavor}: bottom-right alignment requires a causal band (MASK_CAUSAL)"),
            (
                bool(cfg.MASK_FLAGS & MASK_SWA) == (cfg.SWA_WINDOW > 0),
                f"{flavor}: MASK_SWA <=> SWA_WINDOW > 0 (got MASK_FLAGS={cfg.MASK_FLAGS}, SWA_WINDOW={cfg.SWA_WINDOW})",
            ),
            (
                bool(cfg.MASK_FLAGS & MASK_PADDED) == bool(cfg.SEQ_KV_LENS_PRESENT or cfg.THD_VARLEN),
                f"{flavor}: MASK_PADDED <=> (SEQ_KV_LENS_PRESENT or THD_VARLEN) (got MASK_FLAGS={cfg.MASK_FLAGS}, SEQ_KV_LENS_PRESENT={cfg.SEQ_KV_LENS_PRESENT}, "
                f"THD_VARLEN={cfg.THD_VARLEN}); a padded mask without the per-batch kv lengths masks against the padded total and every sequence attends "
                f"the whole pad",
            ),
            (
                not (cfg.THD_VARLEN and cfg.SEQ_KV_LENS_PRESENT),
                f"{flavor}: THD_VARLEN and SEQ_KV_LENS_PRESENT are mutually exclusive (THD reads its lengths from the metadata buffer)",
            ),
            (
                not cfg.THD_VARLEN or cfg.SCHEDULER_POLICY == SCHED_NATURAL,
                f"{flavor}: THD_VARLEN requires SCHEDULER_POLICY NATURAL (the device claim counter walks the metadata's unit list)",
            ),
            (
                cfg.WS_BLOCK_ROWS == cfg.TILE_M * cfg.CTA_MMA,
                f"{flavor}: WS_BLOCK_ROWS ({cfg.WS_BLOCK_ROWS}) must be the pair's kv block (TILE_M * CTA_MMA = {cfg.TILE_M * cfg.CTA_MMA}): the THD "
                f"dS / dV stores carry no skip predicate because every unit's 256-row store box lies wholly inside its sequence's block",
            ),
            (cfg.SCHEDULER_POLICY in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2), f"{flavor}: SCHEDULER_POLICY must be 0/1/2"),
            (
                (cfg.TILE_M * cfg.CTA_MMA) % cfg.TILE_N == 0 and (cfg.TILE_M * cfg.CTA_MMA) // cfg.TILE_N >= 1,
                f"{flavor}: the kv block (TILE_M * CTA_MMA = {cfg.TILE_M * cfg.CTA_MMA}) must be whole q tiles (TILE_N = {cfg.TILE_N}): the bodies round a "
                f"kv block's q range OUTWARD to that many tiles so the stage-3 GEMMs' K-trim (causal_gran = the kv block) never reads an unwritten tile",
            ),
        ]
    )
    # --- SMEM: the per-CTA cap, and the scaffold budget -----------------------------------------
    slabs = smem_layout(cfg)
    used = smem_bytes(cfg)
    if used > SMEM_USABLE_BYTES:
        tally = " | ".join(f"{s.name} {s.nbytes // 1024} KiB @{s.offset // 1024}" for s in slabs)
        raise ValueError(
            f"{flavor}: SMEM slabs {used // 1024} KiB + {SMEM_SCAFFOLD_BYTES // 1024} KiB scaffolding exceed the {SMEM_CAP_BYTES // 1024} KiB Rubin oversized "
            f"per-CTA cap ({tally}). Overflowing it does NOT fail cleanly -- the last slab declared clobbers the scaffolding / wraps -- so it must raise here"
        )
    declared = scaffold_bytes_declared(cfg)
    if declared > SMEM_SCAFFOLD_BYTES:
        raise ValueError(
            f"{flavor}: declared scaffolding ({declared} B of mbarriers + scheduler + tmem ptr) exceeds the {SMEM_SCAFFOLD_BYTES} B budget the SMEM cap check "
            f"reserves; raise SMEM_SCAFFOLD_BYTES (and re-check the slabs) rather than letting the tally lie"
        )
    for lbl, off in desc_roots(cfg):
        if not (0 <= off < used):
            raise ValueError(f"{flavor}: descriptor root {lbl} at {off} B lies outside the {used} B slab layout -- the declaration-order model is inconsistent")
    if cfg.IS_MXFP8 and desc_version(cfg) != 0:
        worst_lbl, worst_off = max(desc_roots(cfg), key=lambda r: r[1])
        raise ValueError(
            f"{flavor}: every UTCCP / MMA descriptor root must stay under the {TCGEN05_V0_ADDR_LIMIT} B version-0 window (highest: {worst_lbl} at "
            f"{worst_off} B) -- an SF slab past the line makes the UTCCP copy P/dS DATA into the SF columns (LSE = +inf / O = NaN on 100 % of cells, the "
            f"d512 MXFP8 forward's signature; rules/mma-tma-matrix.md S6) and an MMA-operand root past it wraps the operand to offset 0 (exactly-zero "
            f"accumulator, no crash); declare the descriptor-fed slabs before sdS rather than switching the module to desc_version=1 (that turned 21 green "
            f"MXFP8 tests red)"
        )


def _tmem_tail_cols(cfg: CfgBwdD256, tm: TmemLayout) -> int:
    """The columns past S | dP | dV that the family's map accounts for: the fp8-class P ring (per-tensor fp8 AND MXFP8, whose
    SF atoms alias the ring's dead slot and take no columns of their own) or the f16 RSVD (K back-half)."""
    if cfg.IS_FP8:
        return cfg.STAGES_TMEM_P * tm.P_COLS
    return tm.RSVD_COLS


def _tmem_tail_desc(cfg: CfgBwdD256, tm: TmemLayout) -> str:
    if cfg.IS_MXFP8:
        return f"P ring {cfg.STAGES_TMEM_P}x{tm.P_COLS} (the SF atoms alias its dead slot)"
    if cfg.IS_FP8:
        return f"P ring {cfg.STAGES_TMEM_P}x{tm.P_COLS}"
    return f"RSVD {tm.RSVD_COLS}"


def _validate_cfg_mxfp8(cfg: CfgBwdD256, flavor: str, b: BufferElems, tm: TmemLayout) -> None:
    """The MXFP8 family's own predicates (TMEM map, SMEM table, barrier inventory, dS policy, mask arm), after the
    fp8-class ones passed.  Each message names the failure SIGNATURE."""
    policy = cfg.DS_SF_POLICY
    pname = _DS_SF_POLICY_NAME.get(policy, str(policy))
    want_ds = DTYPE_BF16 if policy == DS_SF_P_C else DTYPE_E4M3
    want_xfer = 3 if policy == DS_SF_P_A else 2
    want_payloads = 2 if policy == DS_SF_P_B else 1
    want_atoms = {DS_SF_P_A: 2, DS_SF_P_B: 2, DS_SF_P_C: 0}.get(policy, -1)
    sf_smem_want = {
        "K": sf_smem_bytes(cfg.TILE_M, cfg.TILE_K),
        "V": sf_smem_bytes(cfg.TILE_M, cfg.TILE_O),
        "Q": sf_smem_bytes(cfg.TILE_N, cfg.TILE_K),
        "dO": sf_smem_bytes(cfg.TILE_N, cfg.TILE_O),
        "P": sf_smem_bytes(cfg.TILE_M, cfg.TILE_N),
        "dO_T": sf_smem_bytes(cfg.TILE_O, cfg.TILE_N),
        "dS": sf_smem_bytes(cfg.TILE_M, cfg.TILE_N),
    }
    sf_smem_got = {
        "K": cfg.SF_SMEM_K,
        "V": cfg.SF_SMEM_V,
        "Q": cfg.SF_SMEM_Q,
        "dO": cfg.SF_SMEM_dO,
        "P": cfg.SF_SMEM_P,
        "dO_T": cfg.SF_SMEM_dOT,
        "dS": cfg.SF_SMEM_dS,
    }
    sf_cols_want = {
        "K": sf_tmem_cols(cfg.TILE_M, cfg.TILE_K),
        "V": sf_tmem_cols(cfg.TILE_M, cfg.TILE_O),
        "Q": sf_tmem_cols(cfg.TILE_N, cfg.TILE_K),
        "dO": sf_tmem_cols(cfg.TILE_N, cfg.TILE_O),
        "P": sf_tmem_cols(cfg.TILE_M, cfg.TILE_N),
        "dO_T": sf_tmem_cols(cfg.TILE_O, cfg.TILE_N),
    }
    sf_cols_got = {
        "K": cfg.SF_TMEM_COLS_K,
        "V": cfg.SF_TMEM_COLS_V,
        "Q": cfg.SF_TMEM_COLS_Q,
        "dO": cfg.SF_TMEM_COLS_dO,
        "P": cfg.SF_TMEM_COLS_P,
        "dO_T": cfg.SF_TMEM_COLS_dOT,
    }
    sf_tx_want = {"K": cfg.SF_SMEM_K, "V": cfg.SF_SMEM_V, "Q": cfg.SF_SMEM_Q, "dO": cfg.SF_SMEM_dO, "dO_T": cfg.SF_SMEM_dOT}
    sf_tx_got = {"K": cfg.K_SF_TX, "V": cfg.V_SF_TX, "Q": cfg.Q_SF_TX, "dO": cfg.dO_SF_TX, "dO_T": cfg.dOT_SF_TX}
    # The two alias bands, each contiguous from slot-relative column 0 inside ONE P slot (the kernel adds P_OFF + sf_slot * P_COLS).
    bmm1_band = ("K", "V", "Q", "dO")
    bmm2_band = ("P", "dOT")

    def _band_cols(names):
        col = 0
        for n in names:
            if getattr(tm, f"SF_{n}_OFF") != col:
                return -1
            col += getattr(tm, f"SF_{n}_COLS")
        return col

    bmm1_cols = _band_cols(bmm1_band)
    bmm2_cols = _band_cols(bmm2_band)
    _check(
        [
            (
                cfg.SF_BLOCK == MX_BLOCK,
                f"{flavor}: 32 elements share one E8M0 byte (SF_BLOCK == MX_BLOCK, the F8_128x4 SF tensors' and the block-scale MMA's block); got {cfg.SF_BLOCK}",
            ),
            (
                cfg.SF_BLOCK > 0 and cfg.SF_BLOCKS_PER_STEP == cfg.TILE_K_HW_BMM1 // cfg.SF_BLOCK == 2 and cfg.TILE_K_HW_BMM1 == cfg.TILE_K_HW_BMM2,
                f"{flavor}: MmaDesc(sf_blocks_per_step=) is the K64 hardware k-step over 32-element blocks = 2 (tile_dsl/mma.py cycles sf_id = (k*2) % 4 on it); "
                f"got SF_BLOCKS_PER_STEP={cfg.SF_BLOCKS_PER_STEP} at TILE_K_HW {cfg.TILE_K_HW_BMM1}/{cfg.TILE_K_HW_BMM2}",
            ),
            (
                0 <= cfg.P_SCALE_LOG2 <= 126 and cfg.P_SF_BYTE == 127 - cfg.P_SCALE_LOG2,
                f"{flavor}: P is quantized with ONE fixed power-of-two scale P_q = e4m3(P * 2^P_SCALE_LOG2) (0 <= P_SCALE_LOG2 <= 126) and descaled in BMM2 by "
                f"the constant E8M0 byte 127 - P_SCALE_LOG2 (cuDNN's MXFP8 backward convention: 8 -> 119); a byte that disagrees with the scale rescales dV by "
                f"2^k silently; got P_SCALE_LOG2={cfg.P_SCALE_LOG2}, P_SF_BYTE={cfg.P_SF_BYTE}",
            ),
            (
                cfg.STAGES_TMEM_P == 2 and tm.P_OFF + cfg.STAGES_TMEM_P * tm.P_COLS == tm.TOTAL_COLS and tm.RSVD_COLS == 0,
                f"{flavor}: the MXFP8 P ring is the fp8 body's: exactly 2 stages of {tm.P_COLS} e4m3 cols filling TMEM [{tm.P_OFF}, {tm.TOTAL_COLS}) with NO "
                f"dedicated scale-factor columns (got STAGES_TMEM_P={cfg.STAGES_TMEM_P}, RSVD_COLS={tm.RSVD_COLS}).  The SF alias needs exactly ONE dead slot per q "
                f"iteration -- slot n%2 holds P[n], slot (n+1)%2 the atoms every MMA of iteration n reads -- so a deeper ring puts the slot rule and the "
                f"mb_p_ready PipelineState out of step (an MMA reads P or scale factors the softmax is overwriting: garbage dV, no crash) and a shallower "
                f"one has no dead slot at all",
            ),
            (
                sf_smem_got == sf_smem_want,
                f"{flavor}: every SF slab is the operand's F8_128x4 atom count x 512 B, sf_smem_bytes(rows, K) (K / V: TILE_M x TILE_K / TILE_O; Q / dO: TILE_N x "
                f"TILE_K / TILE_O; P and the dS atom: TILE_M x TILE_N; dO_T: TILE_O x TILE_N) -- a short slab lets the SF TMA overshoot into the next slab and the "
                f"UTCCP read past it; got {' '.join(f'{k} {v}' for k, v in sf_smem_got.items())}, want {' '.join(f'{k} {v}' for k, v in sf_smem_want.items())}",
            ),
            (
                sf_cols_got == sf_cols_want,
                f"{flavor}: every SF TMEM count is 4 x ceil(rows/128) x ceil(K/128) (sf_tmem_cols; rows x K-chunks of the operand it scales) and in particular "
                f"SF_TMEM_COLS_P is 4 x ceil(TILE_M/128) x ceil(TILE_N/128), NOT 4 x ceil(TILE_N/128) -- the two coincide only while TILE_M == TILE_N; got "
                f"{' '.join(f'{k} {v}' for k, v in sf_cols_got.items())}, want {' '.join(f'{k} {v}' for k, v in sf_cols_want.items())}",
            ),
            (
                bmm1_cols == tm.SF_BMM1_COLS and 0 < bmm1_cols <= tm.P_COLS,
                f"{flavor}: the BMM1 scale-factor band must pack K | V | Q | dO contiguously from slot-relative column 0 and fit ONE P slot "
                f"({tm.P_COLS} cols): the four operands' atoms are UTCCP'd into the dead P slot of the iteration before dO.V / Q.K read them there; got "
                f"offsets {[getattr(tm, f'SF_{n}_OFF') for n in bmm1_band]} x cols {[getattr(tm, f'SF_{n}_COLS') for n in bmm1_band]}, SF_BMM1_COLS={tm.SF_BMM1_COLS}",
            ),
            (
                bmm2_cols == tm.SF_BMM2_COLS and 0 < bmm2_cols <= tm.P_COLS,
                f"{flavor}: the BMM2 scale-factor band must pack P | dOT contiguously from slot-relative column 0 and fit the same P slot: it is "
                f"UTCCP'd over the dead BMM1 columns after Q.K[i+1] / dO.V[i] were issued and before P.dO[i] reads it; got offsets "
                f"{[getattr(tm, f'SF_{n}_OFF') for n in bmm2_band]} x cols {[getattr(tm, f'SF_{n}_COLS') for n in bmm2_band]}, SF_BMM2_COLS={tm.SF_BMM2_COLS}",
            ),
            (
                sf_tx_got == {k: v * cfg.CTA_MMA for k, v in sf_tx_want.items()},
                f"{flavor}: the TMA-LDG warp grows each _full expect_tx by the SF bytes of BOTH CTAs (SF_SMEM x CTA_MMA = x{cfg.CTA_MMA}: cga2 tensor TMAs route both "
                f"peers' bytes to the leader's mbar, P9; Q / dO / dO_T-SF = 2 CTAs x 512 B x 2 destinations, K / V-SF = 2 CTAs x 1024 B self-multicast); a short count "
                f"completes the phase before the scale factors land (stale SF, wrong dV / dS, no crash), a long one hangs; got "
                f"{' '.join(f'{k} {v}' for k, v in sf_tx_got.items())}.  The x{cfg.CTA_MMA} is the P9 routing rule applied to the SF boxes; MEASURED on the "
                f"MXFP8 body 2026-09-30 (a HAND lane ledger of the barrier table, re-checked row by row in two reviews: 26/26 bars balance, "
                f"every expect_tx == the delivered bytes; on Rubin, cc 10.7: 12 shapes dense / kv tail / q tail / causal / bottom-right / SWA / GQA / B3 x "
                f"n_q_tiles 1 within the oracle gate, 48 / 0 / 0 fresh processes at n_q_tiles 1-4, 0 hangs)",
            ),
            (
                cfg.DTYPE_O in (DTYPE_BF16, DTYPE_FP16),
                f"{flavor}: the MXFP8 backward's dV is half precision (dK / dV / dQ are the graph's bf16 / fp16 gradients; no quantized gradient and no "
                f"amax, so the body has no atomics); got DTYPE_O={cfg.DTYPE_O}",
            ),
            (
                policy in _DS_SF_POLICIES,
                f"{flavor}: DS_SF_POLICY must be DS_SF_P_A / P_B / P_C ({DS_SF_P_A}/{DS_SF_P_B}/{DS_SF_P_C}) on the MXFP8 body -- the dS scale-factor policy is a family "
                f"constant fixed at make_cfg time (numerics-changing, never a knob); got {policy}",
            ),
            (
                cfg.DTYPE_DS == want_ds,
                f"{flavor}: under {pname} the dS workspace is "
                + (
                    "bf16 (no dS quantization: the bf16 stage-3 renderings over dequantized Q_T / K_T)"
                    if policy == DS_SF_P_C
                    else "e4m3 + E8M0 scale factors (the block-scale stage-3 GEMM arm)"
                )
                + f"; got DTYPE_DS={cfg.DTYPE_DS} -- a workspace dtype the GEMM arm does not read is silent garbage gradients",
            ),
            (
                cfg.XFER_STAGES == want_xfer,
                (
                    f"{flavor}: the P-a dS ring is 3-deep (the fp8 body's validated depth: one e4m3 payload, 3 x 16 KiB, 20 KiB of SMEM left); got XFER_STAGES={cfg.XFER_STAGES}"
                    if policy == DS_SF_P_A
                    else f"{flavor}: a 2-deep dS ring: "
                    + ("P-b's second e4m3 payload ring" if policy == DS_SF_P_B else "the bf16 dS ring")
                    + " does not fit at 3 stages (P-b 2 x 3 x 16 KiB, P-c 3 x 32 KiB = 350 KiB of slabs vs the 325 KiB usable), so P-b / P-c run "
                    f"XFER_STAGES=2 -- a depth the ds_full / ds_empty PipelineState and the TMA-STG's per-slot wait(0) were only ever run 3-deep on the fp8 body; "
                    f"validated on the MXFP8 body at n_q_tiles 1/2/3/4 on 2026-09-30 (Rubin, cc 10.7: 48 / 0 / 0 exit 0 / 124 / other over 12 fresh processes x n_q_tiles 1 / 2 / 3 / 4; dS exact to one bf16 rounding on the "
                    f"oracle at every ladder shape); got XFER_STAGES={cfg.XFER_STAGES}"
                ),
            ),
            (
                cfg.DS_PAYLOADS == want_payloads,
                f"{flavor}: P-b writes TWO dS payload rings (dS quantized along q for dK and along kv for dQ: the 32-lane column amax needs the second payload), "
                f"every other policy one; got DS_PAYLOADS={cfg.DS_PAYLOADS} under {pname}",
            ),
            (
                cfg.DS_SF_ATOMS == want_atoms,
                f"{flavor}: dS SF atoms staged per ring stage: P-a 2 (the ONE 32x32 tile byte, expanded once per GEMM orientation -- the dK atom is kv-row-major, "
                f"the dQ atom q-row-major: two distinct F8_128x4 atoms; one staged atom overruns into sdS[0] on the third stage), P-b 2 (one per payload), P-c 0 "
                f"(bf16 dS); got {cfg.DS_SF_ATOMS} under {pname}",
            ),
            (
                cfg.SCALED_FP8_PACK in (0, 1) and cfg.MASK_Q_PAD in (0, 1),
                f"{flavor}: SCALED_FP8_PACK / MASK_Q_PAD are 0/1 trace-time constants (const_expr arms of the pack and the transposed mask); got "
                f"{cfg.SCALED_FP8_PACK} / {cfg.MASK_Q_PAD}",
            ),
            # --- P-b body geometry: the atom byte offsets the compute lanes write are derived from THESE couplings -------------
            (
                policy != DS_SF_P_B or (cfg.TILE_N // cfg.SF_BLOCK == SF_ATOM_COLS and (cfg.TILE_N // cfg.SOFTMAX_WARPGROUPS) == 2 * cfg.SF_BLOCK),
                f"{flavor}: P-b's sf_ds_dk atom columns are the q-blocks of the 128-q tile: TILE_N / SF_BLOCK must be the F8_128x4 atom's {SF_ATOM_COLS} "
                f"columns and each warpgroup's q half exactly TWO blocks (a lane's two E8M0 bytes sit at columns 2 * wg, 2 * wg + 1 and land as ONE "
                f"st.u16); got TILE_N={cfg.TILE_N}, SF_BLOCK={cfg.SF_BLOCK}, SOFTMAX_WARPGROUPS={cfg.SOFTMAX_WARPGROUPS} -- a wider half writes past "
                f"column 3 into the next atom row: the dK GEMM dequantizes with a neighbour row's scale (wrong dK, no crash)",
            ),
            (
                policy != DS_SF_P_B or (cfg.SOFTMAX_WG_WARPS * 32 == cfg.TILE_M and cfg.TILE_M // cfg.SF_BLOCK == SF_ATOM_COLS),
                f"{flavor}: P-b's along-kv scale is ONE warp redux per q column: a compute warp's 32 lanes must be exactly one 32-kv block and the "
                f"warpgroup's SOFTMAX_WG_WARPS warps the CTA's TILE_M rows = the sf_ds_dq atom's {SF_ATOM_COLS} columns (c4 = warp index); got "
                f"SOFTMAX_WG_WARPS={cfg.SOFTMAX_WG_WARPS}, TILE_M={cfg.TILE_M}, SF_BLOCK={cfg.SF_BLOCK} -- a warp that is not one kv block scales "
                f"a column by the wrong rows' amax (wrong dQ, no crash)",
            ),
            (
                policy != DS_SF_P_B
                or (cfg.SF_SMEM_dS == SF_ATOM_BYTES and b.dSSfStagingBytes == cfg.DS_SF_ATOMS * SF_ATOM_BYTES and b.P_D_BLOCK == cfg.TILE_N),
                f"{flavor}: P-b stages ONE {SF_ATOM_BYTES}-B atom per orientation per dS ring stage (sf_ds_dk at +0, sf_ds_dq at +{SF_ATOM_BYTES}) next "
                f"to two e4m3 payload rings whose 128-B row holds BOTH warpgroups' q halves (P_D_BLOCK == TILE_N at BPE_DS 1); got "
                f"SF_SMEM_dS={cfg.SF_SMEM_dS}, staging {b.dSSfStagingBytes} B per stage, P_D_BLOCK={b.P_D_BLOCK} -- an atom offset off by a slab "
                f"lands the dq bytes in the dk atom and both stage-3 GEMMs dequantize with the other orientation's scale (no crash)",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_cfg_d256_bwd(params: _BwdTemplateParams, dtype_family: str) -> CfgBwdD256:
    """Build and validate the Cfg for one body.

    ``dtype_family`` names the BODY (``FAMILY_F16`` / ``FAMILY_FP8`` /
    ``FAMILY_MXFP8``), passed by the kernel template itself; a record whose
    ``dtype_qkv`` belongs to another body raises, so a template loaded against
    the wrong family fails at load time instead of tracing the wrong dtype
    dispatch.
    """
    if dtype_family not in _FAMILIES:
        raise ValueError(f"sm107 bwd d256: dtype_family must be one of {_FAMILIES}; got {dtype_family!r}")
    flavor = _FLAVOR[dtype_family]
    _validate_params(flavor, dtype_family, params)
    is_mxfp8 = dtype_family == FAMILY_MXFP8
    # IS_FP8 = E4M3 payloads = the fp8-CLASS pipeline (K64 + k_dim=1, 3-deep rings, the lookahead MMA order): the
    # per-tensor fp8 body AND the MXFP8 one.
    is_fp8 = dtype_family in (FAMILY_FP8, FAMILY_MXFP8)
    tile_m, tile_n, tile_k, tile_o = 128, 128, 256, 256
    cta_mma = 2  # the single cga2 pair: every collective MMA and every cga2 tensor TMA spans both CTAs
    ds_sf_policy = getattr(params, "ds_sf_policy", -1)
    if is_mxfp8:
        ds_sf_policy = DS_SF_POLICY_DEFAULT if ds_sf_policy < 0 else ds_sf_policy
    else:
        ds_sf_policy = DS_SF_NONE
    dtype_o = getattr(params, "dtype_o", -1)
    if dtype_o < 0:
        # inherit: fp8 -> E4M3 (the sdpa_fp8_backward contract); mxfp8 -> BF16 (half-precision gradients; the adapter
        # passes the graph's dK / dV dtype -- this default serves the doc / test construction); f16 -> the io dtype.
        dtype_o = DTYPE_BF16 if is_mxfp8 else (DTYPE_E4M3 if is_fp8 else params.dtype_qkv)
    dtype_ds = getattr(params, "dtype_ds", -1)
    if dtype_ds < 0:
        if is_mxfp8:
            dtype_ds = DTYPE_BF16 if ds_sf_policy == DS_SF_P_C else DTYPE_E4M3
        else:
            dtype_ds = DTYPE_E4M3 if is_fp8 else params.dtype_qkv
    mask_flags = _mask_flags_from(params)
    tile_k_hw = 64 if is_fp8 else 16
    # MXFP8: the SF slabs / TMEM columns / tx in the general rows x K-chunks form of the operand each scales (sf_smem_bytes /
    # sf_tmem_cols), P's fixed scale, and the dS policy's slabs; the P ring is the fp8 body's TMEM ring (STAGES_TMEM_P).
    if is_mxfp8:
        sf_smem = dict(
            SF_SMEM_K=sf_smem_bytes(tile_m, tile_k),
            SF_SMEM_V=sf_smem_bytes(tile_m, tile_o),
            SF_SMEM_Q=sf_smem_bytes(tile_n, tile_k),
            SF_SMEM_dO=sf_smem_bytes(tile_n, tile_o),
            SF_SMEM_P=sf_smem_bytes(tile_m, tile_n),
            SF_SMEM_dOT=sf_smem_bytes(tile_o, tile_n),
            SF_SMEM_dS=sf_smem_bytes(tile_m, tile_n),
        )
        mxfp8_fields = dict(
            IS_MXFP8=1,
            SF_BLOCK=MX_BLOCK,
            SF_BLOCKS_PER_STEP=tile_k_hw // MX_BLOCK,
            P_SCALE_LOG2=MXFP8_P_SCALE_LOG2,
            P_SF_BYTE=127 - MXFP8_P_SCALE_LOG2,
            **sf_smem,
            SF_TMEM_COLS_K=sf_tmem_cols(tile_m, tile_k),
            SF_TMEM_COLS_V=sf_tmem_cols(tile_m, tile_o),
            SF_TMEM_COLS_Q=sf_tmem_cols(tile_n, tile_k),
            SF_TMEM_COLS_dO=sf_tmem_cols(tile_n, tile_o),
            SF_TMEM_COLS_P=sf_tmem_cols(tile_m, tile_n),
            SF_TMEM_COLS_dOT=sf_tmem_cols(tile_o, tile_n),
            K_SF_TX=sf_smem["SF_SMEM_K"] * cta_mma,
            V_SF_TX=sf_smem["SF_SMEM_V"] * cta_mma,
            Q_SF_TX=sf_smem["SF_SMEM_Q"] * cta_mma,
            dO_SF_TX=sf_smem["SF_SMEM_dO"] * cta_mma,
            dOT_SF_TX=sf_smem["SF_SMEM_dOT"] * cta_mma,
            DS_SF_POLICY=ds_sf_policy,
            DS_PAYLOADS=2 if ds_sf_policy == DS_SF_P_B else 1,
            DS_SF_ATOMS={DS_SF_P_A: 2, DS_SF_P_B: 2, DS_SF_P_C: 0}[ds_sf_policy],
            SCALED_FP8_PACK=int(bool(getattr(params, "scaled_fp8_pack", False))),
            MASK_Q_PAD=int(bool(getattr(params, "mask_q_pad", False))),
        )
        # P-a keeps the fp8 body's 3-deep e4m3 dS ring; P-b (two payloads) and P-c (bf16) fit only at 2 stages (350 KiB of
        # slabs at 3 stages vs the 325 KiB usable -- see smem_layout).
        xfer_stages = 3 if ds_sf_policy == DS_SF_P_A else 2
    else:
        mxfp8_fields = {}
        xfer_stages = 3 if is_fp8 else 1
    # Register split.  The pool setmaxnreg redistributes is the LAUNCH allocation, 12 x 168 = 2016 (reg_entry_pool),
    # so the per-warp sum must not exceed it -- 8 x 232 + 4 x 48 = 2048 (the whole register file) HUNG the first
    # Rubin launch (2026-09-23): the four DEALLOCs release 4 x 120 = 480, the eight ALLOCs ask 8 x 64 = 512, the
    # last softmax warp parks forever.  The pre-port 8 x 232 + 4 x 40 = 2016 balanced exactly but the bf16 f16 builds
    # then spill ONE 4-byte slot in the 40-register MMA warp (the per-thread shared-storage base the sleeping
    # wait-retry loops reload: 1 STL / 8-11 LDL on sm_107a; the fp16 build fits).  f16 therefore moves 8 registers
    # from the softmax warps to the service warps: 8 x 224 + 4 x 56 = 2016, spill-free on both sides (SASS pin
    # test_sm107_register_split_spills_and_drains_sass_pins).  fp8 keeps the pre-port 232 / 40: its dense AND causal
    # sm_107a builds are spill-free (0 STL / 0 LDL, the same pin's fp8-dense / fp8-causal rows, 2026-09-24).
    # The MXFP8 body starts from the fp8 split unchanged (its softmax is LIGHTER per lane; the f16
    # split 224 / 56 is the named fallback if the MMA warp's six SF descriptors spill -- arbiter: the SASS spill pins).
    # MEASURED on the MXFP8 body 2026-09-30: 232 / 40 holds -- REG 168, STL 0 / LDL 0, USETMAXREG 5 on all seven sm_107a arms (host trace-
    # compile, test_sdpa_bwd_mxfp8_sm107.py spill pins) and on the Rubin cubin (SWA arm); the six SF descriptors + the sP descriptor
    # fit the 40-register MMA warp without desc_opaque anchors; the f16 224 / 56 fallback was not needed.  (Measured on the SMEM-P-ring
    # predecessor; the TMEM-alias body trades the sP descriptor for a 32-bit slot base -- its spill pins are the same 0 / 0.)
    softmax_regs = 232 if is_fp8 else 224
    service_regs = 40 if is_fp8 else 56
    cfg = CfgBwdD256(
        TILE_M=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_o,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=bpe(params.dtype_qkv),
        BPE_O=bpe(dtype_o),
        IS_FP8=int(is_fp8),
        DTYPE_DS=dtype_ds,
        BPE_DS=bpe(dtype_ds),
        CGA_M=cta_mma,
        CGA_N=1,
        CTA_MMA=cta_mma,
        Q_SWZ_BYTES=128,
        dO_SWZ_BYTES=128,
        K_SWZ_BYTES=128,
        V_SWZ_BYTES=128,
        P_SWZ_BYTES=128,
        dS_SWZ_BYTES=128,
        dV_SWZ_BYTES=128,
        TILE_K_HW_BMM1=tile_k_hw,
        TILE_K_HW_BMM2=tile_k_hw,
        IDESC_K_DIM=1 if is_fp8 else 0,
        K_SPLIT_UTCCP=0 if is_fp8 else 1,
        STAGES_Q=3 if is_fp8 else 2,
        STAGES_dO=3 if is_fp8 else 2,
        STAGES_dO_DV=3 if is_fp8 else 2,
        STAGES_KV=1,
        STAGES_TMEM_S=1,
        STAGES_TMEM_P=2 if is_fp8 else 1,
        XFER_STAGES=xfer_stages,
        STATS_STAGES=2,
        TILES_Q=1,
        SCHEDULER_STAGES=2,
        SOFTMAX_WARPGROUPS=2,
        SOFTMAX_WG_WARPS=4,
        CORRECTION_WARPS=0,
        SOFTMAX_REGS=softmax_regs,
        CORRECTION_REGS=0,
        MMA_REGS=service_regs,
        TMALDG_REGS=service_regs,
        TMASTG_REGS=service_regs,
        SCHEDULER_REGS=service_regs,
        OTHER_REGS=service_regs,
        MASK_FLAGS=mask_flags,
        SWA_WINDOW=params.window_left or 0,
        CAUSAL_BOTTOM_RIGHT=int(params.bottom_right),
        HAS_SINK=int(getattr(params, "has_sink", False)),
        SEQ_KV_LENS_PRESENT=int(params.seq_kv_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        WS_BLOCK_ROWS=tile_m * cta_mma,
        L2_SIZE_MIB=60,
        SCHEDULER_POLICY=params.sched_policy,
        TOTAL_WARPS=12,
        THREADS_PER_CTA=12 * 32,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=4,
        MMA_WARP_ID=8,
        TMALDG_WARP_ID=9,
        TMASTG_WARP_ID=10,
        SCHED_WARP_ID=11,
        ONE_LANE=1,
        ONE_WARP=32,
        SOFTMAX_WG_LANES=4 * 32,
        SOFTMAX_LANES=2 * 4 * 32,
        SOFT_X_CTA_MMA=2 * 4 * 32 * 2,
        MMA_COMMIT_ARRIVES=1,
        READ_TILE_ARRIVERS_TOT=2 * (2 * 4 + 2) + 1,
        N_BMM2_CHUNKS=128 // 64,
        BMM2_CHUNK_SIZE=64,
        **mxfp8_fields,
    )
    _validate_cfg_d256_bwd(cfg, flavor)
    return cfg


# ---------------------------------------------------------------------------
# Shape-time helpers for the adapter / the kernel's compile()
# ---------------------------------------------------------------------------


def validate_head_chunk(h_q: int, h_kv: int, qh_chunk: int, allow_subgroups: bool = False) -> None:
    """The dS workspace is head-chunked (``qh_chunk`` heads per launch, runtime
    ``head_base``).  A chunk must be a divisor of ``H_q`` AND whole GQA groups
    (a multiple of ``H_q // H_kv``), or the per-Q-head dK/dV partial fold over a
    chunk is incomplete and a KV head's gradient is summed from a partial
    group. A dense chain that writes global per-Q-head partials and folds only
    after ALL chunks may explicitly allow a chunk that evenly divides one group.
    Other chains retain the whole-group requirement."""
    if h_kv < 1 or h_q < 1 or h_q % h_kv != 0:
        raise ValueError(f"sm107 bwd d256: H_q ({h_q}) must be a positive multiple of H_kv ({h_kv})")
    group = h_q // h_kv
    if qh_chunk < 1 or h_q % qh_chunk != 0:
        raise ValueError(f"sm107 bwd d256: qh_chunk ({qh_chunk}) must be a positive divisor of H_q ({h_q})")
    if qh_chunk % group != 0 and not (allow_subgroups and group % qh_chunk == 0):
        raise ValueError(
            f"sm107 bwd d256: qh_chunk ({qh_chunk}) must be a multiple of the GQA group H_q/H_kv = {group}, or the dK/dV fold over a chunk is partial"
        )


def q_pad_rows(cfg: CfgBwdD256) -> int:
    """S_q must be padded to the q-tile (the inner loop walks whole q-tiles)."""
    return cfg.TILE_N


def kv_pad_rows(cfg: CfgBwdD256) -> int:
    """S_kv must be padded to the cga2 pair's kv block (TILE_M * CTA_MMA)."""
    return cfg.TILE_M * cfg.CTA_MMA


def q_write_tiles(cfg: CfgBwdD256) -> int:
    """The q tiles a kv block's q range is rounded OUTWARD to before the body walks it: ``kv_pad_rows / TILE_N`` (2).

    The bodies' ``_q_loop_bounds`` round the mask band's ``[q_lo, q_hi)`` to this many tiles so that stage 2 writes its
    dS in ``kv_pad_rows``-row q PAIRS -- the granularity the stage-3 GEMMs trim at (``MatmulTemplateParams.causal_gran``,
    which the adapter sets to ``kv_pad_rows``; ``bprop_matmul_blackwell._causal_k_range``).  A dQ cluster M tile spans a
    whole pair, so a kv block that wrote one tile of the pair wrote the other: the GEMM never reads a tile only its
    neighbour's band covered, and the workspace needs no zero-fill.  The cost is at most one fully-masked (P = 0 -> dS = 0,
    dV += 0) q tile per side per kv block; under plain top-left causal none (the diagonal's tile is already even).
    """
    return kv_pad_rows(cfg) // cfg.TILE_N


def ds_workspace_bytes(cfg: CfgBwdD256, batch: int, qh_chunk: int, s_q_pad: int, s_kv_pad: int) -> int:
    """Bytes of the dS PAYLOAD workspace one launch writes: ``DS_PAYLOADS`` tensors of
    ``[batch, qh_chunk, S_kv, S_q] x BPE_DS`` (kv-major: the layout that lets dK = dS.Q
    read it un-permuted).  One payload on the f16 / fp8 bodies and under P-a / P-c; TWO
    under P-b (dS quantized along q for dK and along kv for dQ).  The dS
    SCALE-FACTOR tensors (``sf_ds_dk`` /
    ``sf_ds_dq``) are separate workspace rows, not counted here."""
    if s_q_pad % q_pad_rows(cfg) or s_kv_pad % kv_pad_rows(cfg):
        raise ValueError(
            f"sm107 bwd d256: workspace extents must be padded to q {q_pad_rows(cfg)} / kv {kv_pad_rows(cfg)} rows; got S_q={s_q_pad}, S_kv={s_kv_pad}"
        )
    return cfg.DS_PAYLOADS * batch * qh_chunk * s_kv_pad * s_q_pad * cfg.BPE_DS


def sf_workspace_bytes(cfg: CfgBwdD256, batch: int, qh_chunk: int, s_q_pad: int, s_kv_pad: int) -> int:
    """Bytes of ONE dS scale-factor workspace tensor a launch writes -- ``DS_SF_ATOMS`` of them exist (P-a / P-b: ``sf_ds_dk``
    ``[batch, qh_chunk, S_kv/128, S_q/128, 512]`` and ``sf_ds_dq`` ``[batch, qh_chunk, S_q/128, S_kv/128, 512]``, one F8_128x4
    atom per (kv tile, q tile); P-c: none), each ``batch * qh_chunk * (S_kv/128) * (S_q/128) * SF_ATOM_BYTES`` = the payload's
    bytes / 32 (1 / 32 of ``ds_workspace_bytes`` per payload).  0 when the policy stages no atoms.  Same padding contract as
    :func:`ds_workspace_bytes`; the two tensors differ only in the ORDER of their tile axes (the byte count is the same)."""
    if s_q_pad % q_pad_rows(cfg) or s_kv_pad % kv_pad_rows(cfg):
        raise ValueError(
            f"sm107 bwd d256: workspace extents must be padded to q {q_pad_rows(cfg)} / kv {kv_pad_rows(cfg)} rows; got S_q={s_q_pad}, S_kv={s_kv_pad}"
        )
    if cfg.DS_SF_ATOMS == 0:
        return 0
    return batch * qh_chunk * (s_kv_pad // SF_ATOM_ROWS) * (s_q_pad // SF_ATOM_ROWS) * SF_ATOM_BYTES


def launch_grid(cfg: CfgBwdD256, batch: int, qh_chunk: int, s_kv_pad: int, thd_units: int = 0) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """``(grid, cluster)`` for one launch: natural = ``(kv_blocks * CGA_M, qh_chunk, B)``;
    LPT / LPT_L2 = the flat 1-D grid the body's tile decode expects; THD (``cfg.THD_VARLEN``,
    ``thd_units`` appended) = the occupancy-sized persistent grid ``(thd_units * CGA_M, 1, 1)``
    the device claim counter hands units out to (:func:`thd_units_upper_bound`)."""
    if cfg.THD_VARLEN:
        if thd_units < 1:
            raise ValueError(f"sm107 bwd d256: a THD launch needs thd_units >= 1 (the persistent grid's cluster count); got {thd_units}")
        return (thd_units * cfg.CGA_M, 1, 1), (cfg.CGA_M, cfg.CGA_N, 1)
    kv_blocks = -(-s_kv_pad // kv_pad_rows(cfg))
    if cfg.SCHEDULER_POLICY == SCHED_NATURAL:
        grid = (kv_blocks * cfg.CGA_M, qh_chunk, batch)
    else:
        grid = (kv_blocks * qh_chunk * batch * cfg.CGA_M, 1, 1)
    return grid, (cfg.CGA_M, cfg.CGA_N, 1)


def thd_units_upper_bound(cfg: CfgBwdD256, t_kv_cap: int, batch: int, qh_chunk: int) -> int:
    """An UPPER bound on the THD launch's live units: ``(ceil(t_kv_cap / kv block) + B) * qh_chunk`` -- every sequence's
    kv blocks (its own ceil can add at most one block per sequence over the packed total's) times the heads of one
    launch.  The grid is ``min(this, the device's cluster count)``; a cluster whose first unit is at or past the device
    live total runs ONE forced fully-masked tile (dead initial units are legal: the body's degenerate-input contract)."""
    if t_kv_cap < 0 or batch < 1 or qh_chunk < 1:
        raise ValueError(f"sm107 bwd d256: thd_units_upper_bound needs t_kv_cap >= 0, batch >= 1, qh_chunk >= 1; got {t_kv_cap}, {batch}, {qh_chunk}")
    return max(1, (-(-t_kv_cap // kv_pad_rows(cfg)) + batch) * qh_chunk)
