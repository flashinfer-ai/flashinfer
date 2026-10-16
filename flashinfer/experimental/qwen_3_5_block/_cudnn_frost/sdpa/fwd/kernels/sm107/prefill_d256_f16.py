# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""pre-upstream DSL qwen prefill SDPA kernel — d_qk = d_v = 256, FP16/BF16.

TILES_Q=1, SOFTMAX_WARPGROUPS=1, 12 warps total.  Q∪O SMEM alias with
extra barriers ``mb_q_o_alias`` and ``mb_tmastg_go``.  Q·K(i+1) → S·V(i)
lookahead stream order with two parity-keyed S_acc TMEM slots; P_cast
aliases the tail of each.  Stats parked at col 544 (outside S_acc/O).

THD / varlen (``CFG.THD_VARLEN=1``, the pre-upstream packed-varlen entry point)
is supported (f16/bf16) via the shared mechanism (packed ``[1,T,H,D]`` +
``cu_seqlens`` coord offset, per-batch O descriptor array, packed ``[1,QH,T]``
LSE); the dense ``[B,S,H,D]`` path is byte-identical.

Fused epilogue gate (``CFG.EPILOGUE_GATE`` <- ``TemplateParams.epilogue_gate``):
``O := O * sigmoid(G)`` with G an O-shaped Q-dtype tensor, TMA-staged by the
load warp AFTER the KV loop into a 64 KiB SMEM tile and consumed in the
correction epilogue as the packed ``h*tanh(g/2) + h`` (shared hook in
``_common_blackwell``; every splice carries an ``EPILOGUE_FUSION_SEAM`` token).
Gate-off traces byte-identically except for one aliased, unread
``tma_gate_desc`` GridConstant.  Dense or THD (packed), unsplit, unpaged: under
``CFG.THD_VARLEN`` the host maps G in the same packed ``[1, T, H_q, D_v]`` form
as Q / O (``sdpa_gate_tensor(..., thd=)``) and the TMA-LDG warp's gate load
already takes the THD coordinates (``q_row_base + q_seq_off``, ``tma_batch``) --
the kernel body is the dense gated body, byte for byte.
"""

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
import os
import sys
from functools import lru_cache
from typing import Callable, Optional, Tuple


from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import vote_sync, VoteSync
from cutlass._mlir.dialects import arith

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental.cuda import tensor_map as tmap
import cuda.bindings.driver as _cuda_driver  # noqa: F401

from dataclasses import dataclass
from typing import NamedTuple

# Config comes from the FROST template loader, NOT an env var: the pre-upstream
# kernels picked a flavor with an env var + a sdpa_config_<flavor> module, which the
# FROST engine contract forbids ("no environment variables for configuration" --
# parameters travel as typed dataclasses). The loader injects
# FROST_TEMPLATE_PARAMS before this body runs; the default keeps a plain
# `import` usable as a standalone driver.
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import TemplateParams, make_cfg_d256

PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
CFG, _TMA = make_cfg_d256(PARAMS)
if PARAMS.paged_kv:
    raise ValueError(
        "prefill_d256_f16_sm107: paged_kv is not wired on this kernel (the PAGED_KV specialization lives in sm100/prefill_d128_f16, sm100/prefill_d256_f16 and sm100/prefill_d128_fp8)"
    )

# tcgen05 SMEM-descriptor version for EVERY SmemTile in this module -- ONE
# decision point, wired into every construction below rather than repeated as a
# per-tile literal.  A version-0 descriptor's ``start_address`` is 14 bits = a
# 256 KiB window; Rubin raises the per-CTA SMEM cap to 327 KiB, so an operand
# buffer at or above 256 KiB wraps to offset 0 and the MMA multiplies whatever
# sits at the bottom of SMEM.
#
# DERIVED, not a literal, because STAGES_KV moves the last operand's offset:
# the buffers are laid out sQO | sK[stages] | sV[stages], so the LAST V stage
# starts at (qBufferElems + stages*kBufferElems + (stages-1)*vBufferElems)*BPE
# and crosses 256 KiB at STAGES_KV=4.  A hardcoded 0 there is a SILENT wrong
# answer, and this was shipped: `config_sm107` pinned STAGES_KV to 2 and blamed
# a ring/parity conflation in the body, but the body derives parity from the
# absolute `kv_loop & 1` and is depth-agnostic.  Measured on Rubin (cos vs an
# fp32 reference, n_kv = 2/3/4/8): depth 4 with v0 gives 1.0000/0.6806/0.4980/
# 0.4640 -- wrong exactly from the KV iteration that first touches the wrapped
# V stage -- and depth 4 with v1 gives 1.0000 across the board, dense AND
# causal.  Depth 2 and 3 stay under the line and are correct at v0.
#
# Do NOT re-literal this at a call site: the d512 MXFP8 sibling shipped NaN on
# 100% of cells because its scale-factor tiles were declared UNDER a comment
# claiming "every operand tile here carries desc_version=1" -- without the
# kwarg.  A single constant makes that class of drift impossible, and
# test_sm107_descriptor_version_matches_the_smem_budget asserts it.
_TCGEN05_V0_ADDR_LIMIT = 262144  # 14-bit start_address window


def _needs_desc_v1(cfg) -> bool:
    """True when any MMA-operand buffer starts at or past the v0 address limit.

    Uses the START of the last V stage, not the total: a buffer that merely
    ENDS past the line is fine, it is the start address that gets truncated.
    """
    last_v_start = (
        cfg.TILE_M * cfg.TILE_K + cfg.STAGES_KV * (cfg.TILE_N * cfg.TILE_K // cfg.CTA_MMA) + (cfg.STAGES_KV - 1) * (cfg.TILE_O * cfg.TILE_N // cfg.CTA_MMA)
    ) * cfg.BPE
    return last_v_start >= _TCGEN05_V0_ADDR_LIMIT


DESC_VERSION: int = 1 if _needs_desc_v1(CFG) else 0
# Retry form of the per-KV-iteration RING waits (k/v/q _full/_empty, bmm1_done / bmm2_ready / bmm2_done, stat_* / xfer_*,
# s_acc / o_empty, empty_mainloop): every such site is spelled ``.wait(..., spin=SPIN_RING_WAITS)``; the waits a warp
# parks in for a whole tile (scheduler payload, tmem_dealloc, the TMA-STG's O-ready wait) and the end-of-kernel drains
# keep the default sleeping form.  ``spin=True`` is the hint-less uniform spin (tile_dsl.barrier.wait); the decision is
# a MEASURED per-kernel fact (Rubin node locked at 2376 MHz, A/B/A x3, control pair, TFLOPS ratios vs the sleeping
# form), so it lives here once, like DESC_VERSION, and never as a literal at a call site
# (test_sm107_ring_waits_take_the_module_spin_constant).  Flipping this constant IS the whole experiment.
# Measured: neutral on the 12-warp d256 layout (+0.1 % @ S=2K, +0.6 % @32K, -0.3..+0.5 % @8K with every wait spun, all
# inside the control pair), so this kernel keeps the sleeping form its bf16 SDPA stage in the gated block was measured with.
SPIN_RING_WAITS: bool = False
Cfg = type(CFG)
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS

# DTYPE_QKV ∈ {2: BF16, 3: FP16, 4: TF32} — FP8 lives in the d256_fp8 kernel.
if CFG.DTYPE_QKV not in (2, 3, 4):
    raise ValueError(
        f"prefill_sdpa_d256_f16: DTYPE_QKV must be 2 (BF16), 3 (FP16), or " f"4 (TF32); got {CFG.DTYPE_QKV}.  Use prefill_sdpa_d256_fp8.py for FP8/E5M2."
    )

TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE_O
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import (
    PipelineState,
    advance,
    cga_arrive,
    cga_wait,
    # `wait` (free fn) — still used for sched.mb_* (Sched not in Bars).
    wait,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.scheduler import (
    Sched,
    scheduler_warp_loop,
    scheduler_warp_loop_persistent,
    read_tile_id_arrive,
    read_clc_payload,
    SCHED_NATURAL,
    SCHED_LPT,
    SCHED_LPT_L2,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import (
    tmem_load_max_reduction_tile,
    row_reduction_pair,
    row_max_reduction,
    vec_scale_pair,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.regtile import RegTile, vec_concat
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.mma import mma_ss, mma_ts_step
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import tma_load_tile, tma_store_tile, tma_store_commit, tma_store_wait, tma_tensormap_acquire
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma, tma_slice_runtime_desc
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.mask import (
    apply_mask_chunk,
    MASK_NONE,
    MASK_PADDED,
    MASK_CAUSAL,
    MASK_SWA,
)

if CFG.DTYPE_QKV == 2:
    STORAGE_DTYPE = cutlass.BFloat16
    P_STORAGE_DTYPE = cutlass.BFloat16
    MMA_KIND = nvvm.Tcgen05MMAKind.F16
    IS_TF32 = False
elif CFG.DTYPE_QKV == 3:
    STORAGE_DTYPE = cutlass.Float16
    P_STORAGE_DTYPE = cutlass.Float16
    MMA_KIND = nvvm.Tcgen05MMAKind.F16
    IS_TF32 = False
elif CFG.DTYPE_QKV == 4:
    # TF32 — see prefill_sdpa_f16.py for the three deltas (V SWIZZLE_128B_ATOM_32B
    # SmemDesc+TMA, 1:1 P TMEM packing, BPE=4).  qwen already Q∪O-aliases, so
    # the only flavor-specific bits here are STAGES_KV=2 + TILE_N=64 (config).
    STORAGE_DTYPE = cutlass.Float32
    P_STORAGE_DTYPE = cutlass.Float32
    MMA_KIND = nvvm.Tcgen05MMAKind.TF32
    IS_TF32 = True
else:
    raise ValueError(f"prefill_sdpa_d256_f16: DTYPE_QKV={CFG.DTYPE_QKV} not supported " f"(expected 2=BF16, 3=FP16, or 4=TF32)")

if CFG.DTYPE_O != CFG.DTYPE_QKV:
    raise NotImplementedError(f"prefill_sdpa_d256_f16: DTYPE_O={CFG.DTYPE_O} != DTYPE_QKV=" f"{CFG.DTYPE_QKV} not yet supported.")
OUT_STORAGE_DTYPE = STORAGE_DTYPE

# softmax_scale_prefolded (TemplateParams.softmax_scale_prefolded <- the graph's attn_scale_prefolded): the caller
# multiplied Q by attn_scale * log2(e), so the raw QK^T already sits in the log2 domain.  The softmax then takes the
# RAW row max and shifts with one FADD2 per pair (reg_S - m) in place of the FFMA2 (reg_S * scale_log2 - m); the
# ``scale_log2`` runtime argument stays in every signature as a dead operand (the adapter pins it to exactly 1.0 and
# the kernel never reads it under the fold).  The published Stats are unchanged: the running max and the shifted
# scores live in the same log2 domain as when the kernel applies the scale, so the correction's LSE / sink fold /
# inv_sum consume the same values.  Numerically neutral on its own (the per-score multiply by the scale becomes the
# caller's one rounding of Q); the fused shift+convert it enables is a quantized-kernel arm this body does not carry.
# Mask sentinel: a fully-masked tile leaves the raw max exactly at the finite mask sentinel (== NEG_INF) under the fold, at
# the scaled sentinel otherwise.  Ahead of the row's first live key such a tile is DEAD: running_max_step_finite_sentinel
# (the shared running-max step) keeps total_max at NEG_INF, publishes alpha = 1 and shifts the tile by 0 so P = 0, and the
# first live tile starts the online softmax through is_first -- the sentinel never becomes the running max (scaled by
# scale_log2 > 1 it overflowed to -inf and the shift read -inf - (-inf) = NaN; below that P = 1 per masked column).  A
# keyless row ends at (NEG_INF, 0) and is overridden by the correction's _kv_empty select.  The softmax body is inlined four
# times (dense / left-masked / unmasked interior / right-masked); every copy carries both arms at its max site and its
# shift site -- the source pin in test_sdpa_fwd_dsl_sm107.py counts them.
SCALE_PREFOLDED = int(PARAMS.softmax_scale_prefolded)
# softmax_precision=HALF (TemplateParams.softmax_f16) is a quantized-kernel specialization: on half inputs the f16x2
# exponent would be a new numerics class (exp input ulp 2^-8 at the rescale threshold; bf16 ex2 is 2^-7), so the config
# backstop (config_sm107.py _validate_params) declines it before this body loads and the fused arm never exists here.
# The constants are kept so the cc 10.7 forward kernels expose one lever vocabulary (tests read _k_mod._FUSED_SHIFT_CVT).
SOFTMAX_F16 = int(PARAMS.softmax_f16)
if SOFTMAX_F16:
    raise ValueError("prefill_d256_f16: softmax_f16 is a quantized-kernel (FP8 / MXFP8) specialization; half inputs run the f32 exponent")
_FUSED_SHIFT_CVT = False


from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels._common_blackwell import (
    sdpa_gate_tensor,
    sdpa_operand_tensors,
    D256Bars as Bars,
    KvLoopBounds,
    make_d256_bars,
    compute_kv_loop_bounds,
    lpt_tile_coords,
    make_sdpa_helpers,
    running_max_step_finite_sentinel,
    gate_geometry,
    issue_gate_load,
    gate_chunk_smem_offset,
    load_gate_chunk,
    gate_epilogue_pairs,
    gate_inv_sum,
    gate_half_opaque,
)

CGA_SIZE = CFG.CGA_M * CFG.CGA_N

CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2 if CFG.CTA_MMA == 2 else nvvm.CTAGroup.CTA_1

qBufferElems = CFG.TILE_M * CFG.TILE_K
kBufferElems = CFG.TILE_N * CFG.TILE_K // CFG.CTA_MMA
vBufferElems = CFG.TILE_O * CFG.TILE_N // CFG.CTA_MMA
oBufferElems = CFG.TILE_M * CFG.TILE_O
qoAliasBytes = max(qBufferElems * CFG.BPE, oBufferElems * CFG.BPE_O)

qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
kTmaTransactionBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA
vTmaTransactionBytes = vBufferElems * CFG.BPE * CFG.CTA_MMA

N_O_CHUNKS = (CFG.TILE_O * CFG.BPE_O + 127) // 128

CGA_TILE_M = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA

# lpt_q_tiles_in_cga_units=True is REQUIRED, not optional: under a non-NATURAL
# policy the LPT linearization needs q_tiles in CGA units, i.e. n_q_supers //
# CTA_MMA. Without it the decode walks a row range CTA_MMA times too large, no
# valid tile is ever claimed, and the kernel writes NOTHING -- cosine 0.0000 at
# every shape, dense and causal, which reads like a dead kernel rather than a
# scheduling bug.
#
# Every SM100 kernel passes it. The SM107 port dropped it on most flavors, and
# `sdpa/fwd/engines.py` then narrowed the whole Rubin row to
# sched_policies={SCHED_NATURAL} and blamed the ported decode -- while noting
# that the d128 FP8 kernel "DOES honor LPT" and loses the plan anyway. That
# kernel is one of the only two SM107 flavors that kept this argument, which is
# the actual explanation.
_sdpa_h = make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units=True)
_decode_initial = _sdpa_h.decode_initial
_decode_payload = _sdpa_h.decode_payload
# qtrim variant: collapses the KV loop for CGA tiles entirely past the
# per-batch actual Q length (SEQ_Q_LENS_PRESENT; folds to plain bounds otherwise).
_bounds_for_tile = _sdpa_h.bounds_for_tile_qtrim
_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv
# Q-side per-sequence length.  Under THD `seqlen_q` is the PACKED TOTAL, so
# every BOTTOM_RIGHT diagonal needs the sequence's own S_q_b instead.
_resolve_seqlen_q = _sdpa_h.resolve_seqlen_q

# THD / varlen — flat-grid decode + tma-offset closures (CFG-bound) from the
# factory; O-descriptor builder + TENSOR_MAP_QWORDS from the shared
# the shared pre-upstream THD helper.  Gated by CFG.THD_VARLEN (folds out otherwise).
# seq_kv_lens overloaded as the THD metadata buffer (int32 len 3B+2):
#   [0..B-1]=seq_kv_lens  [B..2B]=cu_q(B+1)  [2B+1..3B+1]=cu_k(B+1)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels.thd_helpers import build_thd_meta_o_descs_kernel as _build_thd_meta_o_descs_kernel, TENSOR_MAP_QWORDS, THD_SETUP_THREADS

_TENSOR_MAP_QWORDS = TENSOR_MAP_QWORDS
_dispatch_decode_initial = _sdpa_h.dispatch_decode_initial
_dispatch_decode_payload = _sdpa_h.dispatch_decode_payload
_thd_tma_offsets = _sdpa_h.thd_tma_offsets


@dataclass(frozen=True)
class KernelTmemLayout:
    """Column offsets for the qwen 2-parity-slot SDPA pipeline (FP16 d=256)."""

    TOTAL_COLS: int = 576

    S_ACC_EVEN_OFF: int = 0
    S_ACC_ODD_OFF: int = 128

    # P_cast aliases tail of same-parity S_acc (FP16 2:1 packing).
    P_EVEN_OFF: int = 64
    P_ODD_OFF: int = 192

    O_OFF: int = 256

    # Stats parked at 544 (outside S_acc/O range — DSL-only vs C++ col 512).
    STATS_OFF: int = 544


LAYOUT = KernelTmemLayout()


_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
_SWZ_ATOM_32B = 1  # SWIZZLE_128B_ATOM_32B — TF32 transposed BMM2 operand (V)
SMEM_LAYOUT_Q = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_K = _SWZ_ENUM[CFG.K_SWZ_BYTES]
# V is the B-transposed BMM2 operand → TF32 needs SWIZZLE_128B_ATOM_32B.
SMEM_LAYOUT_V = _SWZ_ATOM_32B if IS_TF32 else _SWZ_ENUM[CFG.V_SWZ_BYTES]
SMEM_LAYOUT_O = _SWZ_ENUM[CFG.O_SWZ_BYTES]
SMEM_LAYOUT_QKO = SMEM_LAYOUT_Q

_O_SWZ_B = {128: 3, 64: 2, 32: 1}[CFG.O_SWZ_BYTES]
_O_SMEM_SWIZZLE = cutlass.Swizzle(_O_SWZ_B, 4, 3)

# == EPILOGUE_FUSION_SEAM(cfg) ==
# Fused epilogue gate (O := O * sigmoid(G)), CFG.EPILOGUE_GATE.  The gate is
# Q-dtype on this kernel; its staging geometry derives from GATE_BPE -- never
# from O's BPE_O / O_SWZ_BYTES (they coincide here; they do NOT on the FP8
# sibling with e4m3 O).  _GG is built for every module (pure host arithmetic)
# so the gate-off body never references an unbound name; every USE folds on
# cutlass.const_expr(CFG.EPILOGUE_GATE).
GATE_STORAGE_DTYPE = STORAGE_DTYPE
if CFG.EPILOGUE_GATE and CFG.BPE != CFG.GATE_BPE:
    # Structural invariant of THIS kernel: the gate is stored in Q's dtype, so
    # its byte width must be the one the geometry (and the config's SMEM tally)
    # were derived from.  TF32 Q (BPE=4) with a gate is not wired.
    raise NotImplementedError(f"{__name__}: the epilogue gate is Q-dtype here ({CFG.BPE} B/elem) but CFG.GATE_BPE={CFG.GATE_BPE}; a TF32 gate is not wired")
_GG = gate_geometry(CFG, gate_bpe=CFG.GATE_BPE, swz_enum=_SWZ_ENUM)
gateBufferElems = _GG.buffer_elems
# cta_group=1 (shared::cta) -> every byte lands on THIS CTA's mbar, so there is
# NO `* CFG.CTA_MMA` here -- contrast qTmaTransactionBytes above (P9's leader
# routing is a property of the cta_group::2 tensor form only).
gateTmaTransactionBytes = _GG.tx_bytes
SMEM_LAYOUT_GATE = _GG.layout_enum
_GATE_SMEM_SWIZZLE = _GG.smem_swizzle

LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES

# leading_byte_offset = 0 when (TILE_O/CTA_MMA)/8 <= 8 else TILE_N * V_SWZ_BYTES
# TF32 ATOM32 override (see prefill_sdpa_f16.py): leading = TILE_N*128, stride = 512.
_CORE_MATRIX_ROWS = 8
_V_PC_COLS = CFG.TILE_O // CFG.CTA_MMA
_LEADING_PV_STD = 0 if (_V_PC_COLS // _CORE_MATRIX_ROWS) <= 8 else CFG.TILE_N * CFG.V_SWZ_BYTES
LEADING_BYTE_OFFSET_PV = (CFG.TILE_N * 128) if IS_TF32 else _LEADING_PV_STD
STRIDE_BYTE_OFFSET_PV = 512 if IS_TF32 else (8 * CFG.V_SWZ_BYTES)

NUM_KPHASES_PV = CFG.TILE_N // CFG.TILE_K_HW_BMM2
NUM_KPHASES_PV_PER_CHUNK = NUM_KPHASES_PV // CFG.N_BMM2_CHUNKS


# === Kernel entry ===


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_o_desc: cutlass.GridConstant[tmap.TensorMap],
    # == EPILOGUE_FUSION_SEAM(kernel_params) ==
    # Gate LOAD descriptor.  Always a valid tensor map: the gate-off compile
    # aliases it to tma_o_desc rather than passing None, so the GridConstant
    # never has to be Optional and CFG.EPILOGUE_GATE stays the single fold flag
    # (gate-off residual = this one unread parameter).
    tma_gate_desc: cutlass.GridConstant[tmap.TensorMap],
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    o_desc_words: cute.Tensor,
    # Unread in the body (tma_gate_desc carries the address); present so the
    # None-specialised gate-off signature mirrors _host's and the gate-on
    # artifact binds G at the tvm-ffi boundary (shape/dtype checked per launch).
    gate_tensor: Optional[cute.Tensor],
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    n_q_supers: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    qh_per_kh: cutlass.Int32,
    scale_softmax_log2: cutlass.Float32,
    seq_q_lens_addr: cutlass.Int64 = 0,
) -> None:

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # Q∪O alias — single buffer (DTYPE_O == DTYPE_QKV).
    sQO_raw = cutlass.Array(STORAGE_DTYPE, qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * kBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * vBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)

    sQ = SmemTile(
        base=sQO_raw,
        elems_per_stage=qBufferElems,
        stages=1,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_QK_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sK = SmemTile(
        base=sK_raw,
        elems_per_stage=kBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=(CFG.TILE_N // CFG.CTA_MMA) * TMA_QK_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sV = SmemTile(
        base=sV_raw,
        elems_per_stage=vBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_PV,
        stride_byte_offset=STRIDE_BYTE_OFFSET_PV,
        layout=SMEM_LAYOUT_V,
        tma_loads_per_tile=TMA_VO_ITERS // CFG.CTA_MMA,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_N * TMA_VO_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    # sO shares backing with sQ (Q∪O alias).
    sO = SmemTile(
        base=sQO_raw,
        elems_per_stage=oBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_O,
        tma_loads_per_tile=TMA_O_ITERS_HOST,
        tma_granu_elems=TMA_O_GRANU_ELEMS_HOST,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_O_GRANU_ELEMS_HOST,
        desc_version=DESC_VERSION,
    )

    # == EPILOGUE_FUSION_SEAM(smem) ==
    # ---- Fused epilogue gate: staging tile + its two barriers -----------------
    # BARRIER TABLE (the two rows CFG.EPILOGUE_GATE adds; the rest are make_d256_bars')
    #
    #   field            mb_gate_full                             mb_gate_empty
    #   producer kind    Producer.TMA_LOAD (arrive_expect_tx)     Producer.THREAD (plain arrive)
    #   producer lanes   TMA-LDG warp (CFG.TMALDG_WARP_ID), one   4 correction warps x 32 lanes, bare
    #                    warp, pred=nvvm.elect_sync()             .arrive() once per tile after the last
    #                    -> 1 issuing lane x 1 call/tile = 1      gate LDS -> 128 lanes x 1 call/tile = 128
    #   guard form       value predicate (P16)                    bare, all lanes -- THREAD has no pred=
    #                                                             path (barrier.py raises); same form as
    #                                                             mb_o_full
    #   byte delivery    tma_load_tile(sGate, tma_gate(0, head,   n/a
    #                    q_row_base + q_seq_off, tma_batch),
    #                    mb_gate_full.smem_ptr, cta_group=1)
    #                    -> cp.async.bulk.tensor.shared::cta,
    #                    _GG.tma_iters (4) subtiles x (TILE_M x
    #                    _GG.tma_granu_elems (64) x GATE_BPE (2))
    #                    = gateTmaTransactionBytes = 64 KiB, ALL
    #                    to THIS CTA's mbar; tma_load_tile elects
    #                    per subtile internally; ONE expect_tx
    #                    covers all four (same shape as sQ)
    #   consumer         correction warpgroup, all 128 lanes      TMA-LDG warp, ONE wait per tile at the
    #                    wait once per tile before the first      TOP of its tile iteration (right after
    #                    gate LDS                                 the mb_q_o_alias wait + flip)
    #   init count       CFG.ONE_LANE = 1                         CFG.CORR_LANES = 128
    #   phase init       consumer gate_full_phase = Int32(0)      consumer gate_empty_phase = Int32(1)
    #                    (wait-then-arrive, P2)                   (PRE-ARMED, P5b: first wait returns,
    #                                                             bar at 0 != 1, no bootstrap arrive)
    #   scope            Scope.LOCAL (per CTA)                    Scope.LOCAL
    #   init site        inside the warp-0 / elect_sync() init block, right after
    #                    bars.mb_tmem_dealloc.init() (P4: one warp, one lane)
    #   P14 balance      the gate load is issued on BOTH arms of the empty-mainloop branch
    #                    (empty arm AND after the kv loop) and the epilogue -- which waits
    #                    mb_stat_full for EVERY tile -- runs on every tile incl. empty-KV and
    #                    padded-Q-trimmed ones -> exactly one wait and one arrive per tile on
    #                    both bars at every shape
    #   P15 drain        none owed (no cross-CTA arrive on either bar); the mb_tmem_dealloc
    #                    exit path is untouched
    #
    # Why per-CTA, full width, cta_group=1 even under cga2 (this kernel runs
    # CTA_MMA=2 by default): the decode folds cta_in_pair into q_super_idx and
    # q_row_global has no cta_in_pair term, so each CTA of the pair already owns
    # a distinct 128-row Q block at the FULL TILE_O width (only K rows / V cols
    # split per CTA).  Hence gateTmaTransactionBytes = gateBufferElems * GATE_BPE
    # with NO * CFG.CTA_MMA (contrast qTmaTransactionBytes, P9), and the shared
    # hook never takes cta_group=CFG.CTA_MMA.
    #
    # SMEM BUFFER ROW  sGate
    #   dtype / elems    GATE_STORAGE_DTYPE (= STORAGE_DTYPE, the Q dtype), gateBufferElems
    #                    = TILES_Q * TILE_M * TILE_O = 32768 elems = 64 KiB, 1 stage
    #   writer           TMA, cta_group=1, _GG.tma_iters (4) subtiles of TILE_M x
    #                    _GG.tma_granu_elems (64), subtile stride TILE_M * 64 elems;
    #                    descriptor box _GG.box_dims = (1, TILE_M, 1, 64),
    #                    swizzle _tma_swz(_GG.swz_bytes = 128)
    #   reader           correction lanes, one q row per lane:
    #                    load_swizzled(_GATE_SMEM_SWIZZLE, 32, count=O_CHUNK) at
    #                    gate_chunk_smem_offset(chunk, O_CHUNK, _GG, tid_in_wg) =
    #                    ((c*16)//64)*(TILE_M*64) + (c*16)%64 + tid_in_wg*64
    #   per-lane stride  _GG.d_block * GATE_BPE = 64 elems x 2 B = 128 B = the bank cycle
    #                    -> an unswizzled tile would be a 32-way conflict
    #   swizzle + why    cutlass.Swizzle(3, 4, 3) (MBase + SShift = 7 = log2(128 B)),
    #                    layout enum _SWZ_ENUM[128] = 2.  BOTH jobs (frost-kernels S5): the
    #                    TMA descriptor WRITES it (s128b) AND lanes READ it directly.
    #                    Derived from GATE_BPE, never from BPE_O / O_SWZ_BYTES.
    #   allocation       declared AFTER sV_raw and BEFORE the bars' _alloc()s, tmem_ptr_i32
    #                    and the sched arrays -> starts at 192 KiB (STAGES_KV=2), ends at
    #                    256 KiB, inside the 327 KiB Rubin carveout (config._validate_smem
    #                    tallies it: 258 KiB at depth 2, RAISES at depth 3).  Pushing the
    #                    barrier / tmem-ptr / sched arrays up by 64 KiB is harmless (none is
    #                    an MMA/UTCCP operand); sGate itself is never an MMA / UTCCP
    #                    operand, so DESC_VERSION (_needs_desc_v1) is unaffected
    #                    (mma-tma-matrix S6).
    sGate_raw = (
        cutlass.Array(GATE_STORAGE_DTYPE, gateBufferElems, alignment=1024, space=cutlass.AddressSpace.smem) if cutlass.const_expr(CFG.EPILOGUE_GATE) else None
    )
    sGate = (
        SmemTile(
            base=sGate_raw,
            elems_per_stage=gateBufferElems,
            stages=1,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=SMEM_LAYOUT_GATE,
            tma_loads_per_tile=_GG.tma_iters,
            tma_granu_elems=_GG.tma_granu_elems,
            tma_subtile_stride_elems=CFG.TILE_M * _GG.tma_granu_elems,
            desc_version=DESC_VERSION,
        )
        if cutlass.const_expr(CFG.EPILOGUE_GATE)
        else None
    )

    # == EPILOGUE_FUSION_SEAM(bars) ==
    bars = make_d256_bars(CFG, N_O_CHUNKS=N_O_CHUNKS, epilogue_gate=bool(CFG.EPILOGUE_GATE))

    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)

    sched = Sched(
        **{
            "mb_scheduler": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "mb_read_tile_id": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "tile_id_smem": cutlass.Array(cutlass.Int32, CFG.SCHEDULER_STAGES * 8, alignment=16, space=cutlass.AddressSpace.smem),
            "bidx_init": bidx,
            "bidy_init": bidy,
            "bidz_init": bidz,
        }
    )

    READ_TILE_ARRIVERS_TOTAL = CFG.READ_TILE_ARRIVERS

    if warp_idx == 0:
        if nvvm.elect_sync():
            # Init counts baked into make_d256_bars(...); kernel calls .init()
            # per stage.  range_constexpr keeps loop vars Python int (required
            # for mb_bmm2_ready tuple-init lookup).
            bars.mb_q_full.init()
            bars.mb_q_o_alias.init()
            bars.mb_tmastg_go.init()
            for p in cutlass.range_constexpr(2):
                bars.mb_bmm1_done[p].init()
                bars.mb_bmm2_done[p].init()
                for c in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                    bars.mb_bmm2_ready[p * CFG.N_BMM2_CHUNKS + c].init()
            bars.mb_stat_full.init()
            bars.mb_stat_empty.init()
            for chunk in cutlass.range_constexpr(N_O_CHUNKS):
                bars.mb_o_full[chunk].init()
            bars.mb_o_empty.init()
            for ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_k_full[ks].init()
                bars.mb_k_empty[ks].init()
                bars.mb_v_full[ks].init()
                bars.mb_v_empty[ks].init()
            for s in range(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), CFG.ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS_TOTAL)
            bars.mb_empty_mainloop.init()
            bars.mb_tmem_dealloc.init()
            # == EPILOGUE_FUSION_SEAM(init) ==
            if cutlass.const_expr(CFG.EPILOGUE_GATE):
                bars.mb_gate_full.init()
                bars.mb_gate_empty.init()

            # bootstrap mb_q_o_alias so first TMA wait passes (no prior O to flush)
            bars.mb_q_o_alias.arrive()

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # P4 cluster fence after mbar init — required before any cga2 cross-CTA arrive
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        cga_arrive()
        cga_wait()

    cta_id_x = cute.arch.block_idx_in_cluster() if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    cta_in_pair = (cta_id_x & cutlass.Int32(1)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    leader_cta_id = (cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    mcast_mask = (cutlass.Int32(3) << leader_cta_id) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    tma_mcast_mask = (cutlass.Int16(1) << cta_in_pair) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int16(0)
    is_leader = cta_in_pair == cutlass.Int32(0)

    # Warp layout: 0-3 softmax wg / 4-7 correction / 8 MMA / 9 TMALDG / 10 TMASTG / 11 scheduler
    if warp_idx >= CFG.SOFTMAX_WG0_BASE and warp_idx < CFG.SOFTMAX_WG0_BASE + CFG.SOFTMAX_WG_WARPS:
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_softmax_log2,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            sched=sched,
            lse_tensor=lse_tensor,
            sinks_tensor=sinks_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
        )

    elif warp_idx >= CFG.CORR_WARP_BASE and warp_idx < CFG.CORR_WARP_BASE + CFG.CORRECTION_WARPS:
        nvvm.setmaxregister(CFG.CORRECTION_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _correction_warp_group(
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            sO=sO,
            tmem_ptr_i32=tmem_ptr_i32,
            tidx=tidx,
            bars=bars,
            sched=sched,
            lse_tensor=lse_tensor,
            sinks_tensor=sinks_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            cta_id_x=cta_id_x,
            sGate=sGate,  # == EPILOGUE_FUSION_SEAM(dispatch_corr) == (gate bars ride `bars`)
        )

    elif warp_idx == CFG.MMA_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(CFG.CTA_MMA == 2):
            if is_leader:
                _mma_warp_group(
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    sQ=sQ,
                    sK=sK,
                    sV=sV,
                    tmem_ptr_i32=tmem_ptr_i32,
                    bars=bars,
                    sched=sched,
                    seq_kv_lens_tensor=seq_kv_lens_tensor,
                    seq_q_lens_addr=seq_q_lens_addr,
                    n_q_supers=n_q_supers,
                    n_qh=n_qh,
                    n_batch=n_batch,
                    mcast_mask=mcast_mask,
                    cta_in_pair=cta_in_pair,
                )
            else:
                _mma_warp_quiet(tmem_ptr_i32, bars)
        else:
            _mma_warp_group(
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                sQ=sQ,
                sK=sK,
                sV=sV,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                seq_q_lens_addr=seq_q_lens_addr,
                n_q_supers=n_q_supers,
                n_qh=n_qh,
                n_batch=n_batch,
                mcast_mask=mcast_mask,
                cta_in_pair=cta_in_pair,
            )

    elif warp_idx == CFG.TMALDG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        # == EPILOGUE_FUSION_SEAM(dispatch_tmaldg) ==
        if cutlass.const_expr(CFG.EPILOGUE_GATE):
            nvvm.prefetch_tensormap(tma_gate_desc.get_ptr())
        _tmaldg_warp_group(
            tma_q_desc=tma_q_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            o_desc_words=o_desc_words,
            sQ=sQ,
            sK=sK,
            sV=sV,
            tma_gate_desc=tma_gate_desc,
            sGate=sGate,
            bars=bars,
            sched=sched,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            qh_per_kh=qh_per_kh,
            is_leader=is_leader,
            cta_in_pair=cta_in_pair,
            tma_mcast_mask=tma_mcast_mask,
        )

    elif warp_idx == CFG.TMASTG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _tmastg_warp_group(
            tma_o_desc=tma_o_desc,
            sO=sO,
            bars=bars,
            sched=sched,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            cta_in_pair=cta_in_pair,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            o_desc_words=o_desc_words,
        )

    else:  # warp_idx == CFG.SCHED_WARP_ID
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        is_cga_first_cta = cta_id_x == cutlass.Int32(0)
        if cutlass.const_expr(CFG.THD_VARLEN):
            # THD: persistent grid + the device claim counter in the 4B+4
            # metadata's last two slots.  The dense CLC path hands out tiles
            # against a grid that is the persistent cluster count, not the work
            # list, so every tile->unit mapping is off without this.
            scheduler_warp_loop_persistent(
                sched,
                CFG.SCHEDULER_STAGES,
                is_cga_first_cta,
                seq_kv_lens_tensor,
                cutlass.Int32(4) * n_batch + cutlass.Int32(3),
                cutlass.Int32(4) * n_batch + cutlass.Int32(2),
                CGA_SIZE,
                CFG.CGA_M,
            )
        else:
            scheduler_warp_loop(sched, CFG.SCHEDULER_STAGES, is_cga_first_cta, CGA_SIZE)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# === TMA-LDG warp group ===


@cute.jit
def _tmaldg_warp_group(
    tma_q_desc,
    tma_k_desc,
    tma_v_desc,
    o_desc_words,
    sQ,
    sK,
    sV,
    bars,
    sched,
    seqlen_q,
    seqlen_kv,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    n_q_supers,
    n_qh,
    n_batch,
    qh_per_kh,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
    # Fused epilogue gate (appended; the call is by keyword).  None = gate off.
    tma_gate_desc=None,
    sGate=None,
):
    """Qwen TMA-LDG warp — single Q load gated on mb_q_o_alias; fires
    mb_tmastg_go each tile (covers empty-mainloop branch)."""

    q_o_alias_phase = cutlass.Int32(0)
    kv_state = PipelineState.start(phase=1)  # bootstrap pre-armed at phase 1
    # == EPILOGUE_FUSION_SEAM(tmaldg_state) ==
    # mb_gate_empty consumer phase, PRE-ARMED at 1 (P5b): the first tile's
    # staging buffer is untouched, so the first wait must return with no
    # producer arrive.  Loop-carried like q_o_alias_phase.
    gate_empty_phase = cutlass.Int32(1)

    tma_q = GmemTileTma(tma_q_desc)
    tma_gate = GmemTileTma(tma_gate_desc) if cutlass.const_expr(CFG.EPILOGUE_GATE) else None
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD: K/V ride the setup kernel's packed-total-CLAMPED runtime
        # descriptors (o_desc_words slots n_batch+1 / n_batch+2).  The
        # plan-time descriptors describe the buffer's CAPACITY, so the last
        # sequence's tile-tail would read caller-owned bytes past cu_k[B]; a
        # NaN there wipes the whole tile through BMM2 (0 * NaN == NaN).  Through
        # the clamped descriptors those rows are TMA-OOB and land as EXACT
        # ZEROS.  Same closure shape as GmemTileTma, so load sites stay
        # branch-free.
        _k_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(1)) * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        _v_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(2)) * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        # Setup publishes immutable packed-total maps for this launch.
        # Each loader warp acquires both maps before consuming any tile.
        tma_tensormap_acquire(_k_rt_ptr)
        tma_tensormap_acquire(_v_rt_ptr)
        tma_k = lambda *coords: tma_slice_runtime_desc(_k_rt_ptr, *coords)  # noqa: E731
        tma_v = lambda *coords: tma_slice_runtime_desc(_v_rt_ptr, *coords)  # noqa: E731
    else:
        tma_k = GmemTileTma(tma_k_desc)
        tma_v = GmemTileTma(tma_v_desc)

    q_super_idx, head_idx, batch_idx = _dispatch_decode_initial(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
    )
    kv_head_idx = cute.arch.make_warp_uniform(head_idx // qh_per_kh)
    q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M))
    q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)

    if cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds_init = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)
    V_COL_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        # Wait prior tile's O drained before any potential Q∪O clobber.
        bars.mb_q_o_alias.wait(q_o_alias_phase, spin=SPIN_RING_WAITS)
        q_o_alias_phase = q_o_alias_phase ^ cutlass.Int32(1)

        # == EPILOGUE_FUSION_SEAM(tmaldg_issue) ==
        # (a) Gate staging-buffer reuse: the PRE-ARMED wait stays at the TOP of
        # the tile iteration -- a pre-armed wait placed mid-iter crash-failed with
        # cudaErrorLaunchFailure on the pre-upstream toolchain (a JIT interaction
        # with mbarrier.try_wait.parity, never root-caused).  Free in practice: the mb_q_o_alias
        # wait above already parks this warp behind the previous tile's O store,
        # which follows the epilogue's gate consume.  (b) and (c) below issue the
        # load itself on BOTH arms of the empty-mainloop branch (P14).
        if cutlass.const_expr(CFG.EPILOGUE_GATE):
            bars.mb_gate_empty.wait(gate_empty_phase, spin=SPIN_RING_WAITS)
            gate_empty_phase = gate_empty_phase ^ cutlass.Int32(1)

        if cutlass.const_expr(CFG.MASK_FLAGS != 0) and (kv_right <= kv_left):
            # Empty mainloop — skip Q/K/V loads; mb_tmastg_go still fires below.
            # (b) The gate tile is still consumed: this tile still runs an epilogue
            # (writing the selected zeros) and still waits mb_gate_full.  Folds to
            # nothing when the gate is compiled out.
            issue_gate_load(sGate, tma_gate, bars.mb_gate_full, gateTmaTransactionBytes, head_idx, q_row_base + q_seq_off, tma_batch)
        else:
            # P9 — leader-only arrive_expect_tx (TMA byte routing delivers all bytes to leader under cga2)
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                bars.mb_q_full.arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            else:
                bars.mb_q_full.arrive(n_bytes=qTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(
                sQ[0],
                tma_q(cutlass.Int32(0), head_idx, q_row_base + q_seq_off, tma_batch),
                bars.mb_q_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )

            for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
                kv_row_base = kv_loop * cutlass.Int32(CFG.TILE_N)

                bars.mb_k_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sK[kv_state.idx],
                    tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, tma_batch),
                    bars.mb_k_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                    acquire=not CFG.THD_VARLEN,
                )

                bars.mb_v_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sV[kv_state.idx],
                    tma_v(V_COL_OFFSET_PEER, kv_head_idx, kv_row_base + kv_seq_off, tma_batch),
                    bars.mb_v_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                    acquire=not CFG.THD_VARLEN,
                )

                kv_state = advance(kv_state, CFG.STAGES_KV)

            # (c) Gate load AFTER the kv loop, queued ahead of nothing: the TMA
            # engine serves requests in order and the gate is the one operand
            # with slack (consumed only in the epilogue), while K/V have none
            # (the after-Q position measured -2.4 % on the fused kernel).
            issue_gate_load(sGate, tma_gate, bars.mb_gate_full, gateTmaTransactionBytes, head_idx, q_row_base + q_seq_off, tma_batch)

        # Both arms fire mb_tmastg_go (covers empty-mainloop too)
        if nvvm.elect_sync():
            bars.mb_tmastg_go.arrive()
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_super_idx, head_idx, batch_idx = _dispatch_decode_payload(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        kv_head_idx = cute.arch.make_warp_uniform(head_idx // qh_per_kh)
        q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M))
        q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
            bounds_next = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx)
            kv_left = bounds_next.left
            kv_right = bounds_next.right

    # cga2 drain — trailing empty mbar arrives from leader's multicast commits
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
            bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
            bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
            kv_state = advance(kv_state, CFG.STAGES_KV)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


# === TMA-STG warp group ===


@cute.jit
def _tmastg_warp_group(
    tma_o_desc,
    sO,
    bars,
    sched,
    n_q_supers,
    n_qh,
    n_batch,
    cta_in_pair,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    o_desc_words,
):
    """Qwen TMA-STG warp — waits mb_tmastg_go then per-chunk mb_o_full;
    fires mb_q_o_alias at end of tile so TMA can clobber Q∪O."""

    tmastg_go_phase = cutlass.Int32(0)
    o_full_phase = cutlass.Int32(0)

    tma_o = GmemTileTma(tma_o_desc)

    q_super_idx, head_idx, batch_idx = _dispatch_decode_initial(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        bars.mb_tmastg_go.wait(tmastg_go_phase)
        tmastg_go_phase = tmastg_go_phase ^ cutlass.Int32(1)

        for chunk in cutlass.range_constexpr(N_O_CHUNKS):
            bars.mb_o_full[chunk].wait(o_full_phase)

        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)

        if cutlass.const_expr(CFG.THD_VARLEN):
            # THD: store through this batch's pre-built descriptor (base already
            # at the sequence's packed row, seq extent = S_q_b → box past S_q_b
            # OOB-clipped).  q_row_coord is sequence-local; batch coord → 0.
            # DEAD unit (batch_idx == n_batch, the over-launched persistent grid's
            # sentinel): no O rows exist and descriptor slot n_batch is never built,
            # so skip only the store -- the barrier protocol below still runs.
            if batch_idx < n_batch:
                o_desc_ptr = (o_desc_words.iterator.raw_ptr() + batch_idx * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
                o_slice = tma_slice_runtime_desc(o_desc_ptr, cutlass.Int32(0), head_idx, q_row_coord, cutlass.Int32(0))
                tma_store_tile(sO[0], o_slice)
        else:
            tma_store_tile(
                sO[0],
                tma_o(cutlass.Int32(0), head_idx, q_row_coord, batch_idx),
            )
        tma_store_commit()
        tma_store_wait(0)

        bars.mb_o_empty.arrive()
        if nvvm.elect_sync():
            bars.mb_q_o_alias.arrive()

        o_full_phase = o_full_phase ^ cutlass.Int32(1)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        q_super_idx, head_idx, batch_idx = _dispatch_decode_payload(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


# === MMA warp group (+ quiet warp for cga2 non-leader) ===


@cute.jit
def _mma_warp_quiet(tmem_ptr_i32, bars):
    """Quiet MMA-warp body for cga2 non-leader CTA — alloc then dealloc."""
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=True)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _mma_warp_group(
    seqlen_q,
    seqlen_kv,
    sQ,
    sK,
    sV,
    tmem_ptr_i32,
    bars,
    sched,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    n_q_supers,
    n_qh,
    n_batch,
    mcast_mask,
    cta_in_pair,
):
    """Qwen MMA warp — Q*K(i+1) → S*V(i) lookahead stream order with
    prologue / mainloop (kv = lo..hi-2) / epilogue split."""
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=True)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        k_dim=1,
        a_negate=int(PARAMS.negate_scores),
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_O,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        b_major=1,
        k_dim=1,
    )
    bmm1_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=CFG.TILE_K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_qk,
        kind=MMA_KIND,
    )
    bmm2_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_O,
        K=CFG.TILE_N,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM2,
        btranspose=True,
        k_subtile=CFG.V_SWZ_BYTES // CFG.BPE,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_pv,
        kind=MMA_KIND,
    )

    desc_Q = sQ[0].desc()

    if cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    else:
        q_super_idx, _hd, batch_idx = _dispatch_decode_initial(
            sched.bidx_init,
            sched.bidy_init,
            sched.bidz_init,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds_init = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    q_full_phase = cutlass.Int32(0)
    kv_state_K = PipelineState.start(phase=0)
    kv_state_V = PipelineState.start(phase=0)
    # bit p of bmm2_ready_phase_pair holds parity-p's next-wait phase
    bmm2_ready_phase_pair = cutlass.Int32(0)
    empty_mainloop_phase = cutlass.Int32(0)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(CFG.MASK_FLAGS != 0) and (kv_right <= kv_left):
            bars.mb_empty_mainloop.wait(empty_mainloop_phase, spin=SPIN_RING_WAITS)
            empty_mainloop_phase = empty_mainloop_phase ^ cutlass.Int32(1)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        else:
            bars.mb_q_full.wait(q_full_phase, spin=SPIN_RING_WAITS)
            q_full_phase = q_full_phase ^ cutlass.Int32(1)

            # Prologue: Q*K(lo) → S_acc[lo&1]
            lo_parity_runtime = kv_left & cutlass.Int32(1)
            parity_lo_is_even = lo_parity_runtime == cutlass.Int32(0)
            tmem_S_acc_lo_addr = cutlass.Int32(
                arith.select(
                    parity_lo_is_even.ir_value(),
                    cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(),
                    cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value(),
                )
            )

            bars.mb_k_full[kv_state_K.idx].wait(kv_state_K.phase, spin=SPIN_RING_WAITS)
            desc_K = sK[kv_state_K.idx].desc()
            mma_ss(bmm1_desc, desc_Q, desc_K, (tmem_raw.subview(tmem_S_acc_lo_addr)))
            elect_p = nvvm.elect_sync()
            bars.mb_bmm1_done[lo_parity_runtime].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_k_empty[kv_state_K.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

            # Mainloop: kv = lo..hi-2 — Q*K(kv+1) then S*V(kv)
            k_per_chunk = NUM_KPHASES_PV_PER_CHUNK

            for kv_loop in cutlass.range(kv_left, kv_right - cutlass.Int32(1), 1, unroll=1):
                parity_cur_rt = kv_loop & cutlass.Int32(1)
                parity_next_rt = (kv_loop + cutlass.Int32(1)) & cutlass.Int32(1)
                cur_is_even = parity_cur_rt == cutlass.Int32(0)
                next_is_even = parity_next_rt == cutlass.Int32(0)

                tmem_S_acc_next_addr = cutlass.Int32(
                    arith.select(
                        next_is_even.ir_value(),
                        cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(),
                        cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value(),
                    )
                )
                tmem_P_cur_addr = cutlass.Int32(
                    arith.select(
                        cur_is_even.ir_value(),
                        cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(),
                        cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value(),
                    )
                )
                bmm2_ready_phase_cur = (bmm2_ready_phase_pair >> parity_cur_rt) & cutlass.Int32(1)

                # Q*K(kv+1) → S_acc[parity_next]
                bars.mb_k_full[kv_state_K.idx].wait(kv_state_K.phase, spin=SPIN_RING_WAITS)
                desc_K = sK[kv_state_K.idx].desc()
                mma_ss(bmm1_desc, desc_Q, desc_K, (tmem_raw.subview(tmem_S_acc_next_addr)))
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[parity_next_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_k_empty[kv_state_K.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

                # S*V(kv) — chunked manual k-loop with bmm2_ready gates
                bars.mb_v_full[kv_state_V.idx].wait(kv_state_V.phase, spin=SPIN_RING_WAITS)
                desc_V = sV[kv_state_V.idx].desc()

                # scaleC=False on kv_lo first k-step overwrites O acc; True after
                scaleC = cutlass.Boolean(kv_loop != kv_left)
                accum_b2 = scaleC
                for k in cutlass.range_constexpr(NUM_KPHASES_PV):
                    if k % k_per_chunk == 0:
                        chunk_id = k // k_per_chunk
                        bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(
                            bmm2_ready_phase_cur, spin=SPIN_RING_WAITS
                        )
                    mma_ts_step(bmm2_desc, (tmem_raw.subview(tmem_P_cur_addr)), desc_V, (tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF))), k, accum_b2)
                    accum_b2 = cutlass.Boolean(True)
                elect_p = nvvm.elect_sync()
                bars.mb_bmm2_done[parity_cur_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_v_empty[kv_state_V.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_cur_rt)
                kv_state_V = advance(kv_state_V, CFG.STAGES_KV)

            # Epilogue: S*V(hi-1) → O
            kv_last = kv_right - cutlass.Int32(1)
            parity_last_rt = kv_last & cutlass.Int32(1)
            last_is_even = parity_last_rt == cutlass.Int32(0)
            tmem_P_last_addr = cutlass.Int32(
                arith.select(
                    last_is_even.ir_value(),
                    cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(),
                    cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value(),
                )
            )
            bmm2_ready_phase_last = (bmm2_ready_phase_pair >> parity_last_rt) & cutlass.Int32(1)

            bars.mb_v_full[kv_state_V.idx].wait(kv_state_V.phase, spin=SPIN_RING_WAITS)
            desc_V = sV[kv_state_V.idx].desc()
            n_kv_eff = kv_right - kv_left
            scaleC_epi = cutlass.Boolean(n_kv_eff != cutlass.Int32(1))
            accum_b2 = scaleC_epi
            for k in cutlass.range_constexpr(NUM_KPHASES_PV):
                if k % k_per_chunk == 0:
                    chunk_id = k // k_per_chunk
                    bars.mb_bmm2_ready[parity_last_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(
                        bmm2_ready_phase_last, spin=SPIN_RING_WAITS
                    )
                mma_ts_step(bmm2_desc, (tmem_raw.subview(tmem_P_last_addr)), desc_V, (tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF))), k, accum_b2)
                accum_b2 = cutlass.Boolean(True)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[parity_last_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_v_empty[kv_state_V.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)
            kv_state_V = advance(kv_state_V, CFG.STAGES_KV)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0):
            _nq, _nh, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
            is_valid_tile = nxt_v & cutlass.Int32(1)
        else:
            nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
            nxt_q = cute.arch.make_warp_uniform(nxt_q)
            nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
            nxt_v = cute.arch.make_warp_uniform(nxt_v)
            q_super_idx, _hd, batch_idx = _dispatch_decode_payload(
                nxt_q,
                nxt_hb,
                cta_in_pair,
                n_q_supers,
                n_qh,
                n_batch,
                seq_kv_lens_tensor,
            )
            is_valid_tile = nxt_v & cutlass.Int32(1)
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
            bounds_next = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx)
            kv_left = bounds_next.left
            kv_right = bounds_next.right
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


# === Softmax warp group (single wg — qwen TILES_Q=1) ===


@cute.jit
def _softmax_warp_group(
    seqlen_q,
    seqlen_kv,
    scale_log2: cutlass.Float32,
    tmem_ptr_i32,
    bars,
    sched,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_in_pair,
):
    """Qwen softmax — single warpgroup, parity-keyed (kv & 1) S_acc/P
    selectors; epilogue publishes (total_max, total_sum) and fires
    one extra mb_stat_full to balance correction's epilogue wait."""
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    bmm1_done_phase_pair = cutlass.Int32(0)  # bit p = next-wait phase for bmm1_done[p]
    stat_empty_phase = cutlass.Int32(1)  # bootstrap pre-armed so first wait passes
    epilogue_state = cutlass.Int32(1)

    NEG_INF = cutlass.Float32(-3.4028235e38)

    q_super_idx, head_idx, batch_idx = _dispatch_decode_initial(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
    bounds = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx)

    tid_in_wg = cute.arch.thread_idx()[0] - cutlass.Int32(CFG.SOFTMAX_WG0_BASE * 32)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        total_max = NEG_INF
        total_sum = cutlass.Vector.from_elements(
            (cutlass.Float32(0.0), cutlass.Float32(0.0)),
            cutlass.Float32,
        )
        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)
        q_abs = q_row_coord + tid_in_wg

        bars.mb_o_empty.wait(epilogue_state, spin=SPIN_RING_WAITS)
        bars.mb_stat_empty.wait(stat_empty_phase, spin=SPIN_RING_WAITS)
        stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
        epilogue_state = epilogue_state ^ cutlass.Int32(1)

        CHUNK = 64
        # fp16/bf16 pack 2 probs per FP32 TMEM cell; TF32 is 1:1.
        P_COLS_PER_CHUNK = CHUNK if IS_TF32 else CHUNK // 2
        N_CHUNKS = CFG.N_BMM2_CHUNKS
        RESCALE_THRESHOLD = cutlass.Float32(CFG.RESCALE_THRESHOLD)

        # 3-segment dispatch — body inlined so the DSL tracer can dispatch
        # through cutlass.range without tripping the closure check.
        if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
            for kv_loop in cutlass.range(bounds.left, bounds.right, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase, spin=SPIN_RING_WAITS)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)

                tmem_base = tmem_ptr_i32.load()
                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF)

                reg_S_tile, current_max_unscaled = tmem_load_max_reduction_tile(
                    s_addr_base,
                    num_elems=CFG.TILE_N,
                )
                reg_S = RegTile(reg_S_tile.vec, size=CFG.TILE_N)
                if cutlass.const_expr(SCALE_PREFOLDED):
                    current_max = current_max_unscaled  # raw-domain max (Q carries attn_scale * log2 e); sentinel note at SCALE_PREFOLDED
                else:
                    current_max = current_max_unscaled * scale_log2

                # Running max / alpha / exp2 shift; a tile fully masked ahead of the row's first live key is selected out of the
                # state (total_max stays NEG_INF, alpha = 1, shift 0 -> P = 0) -- the sentinel never becomes the running max.
                total_max, alpha, new_total_max = running_max_step_finite_sentinel(
                    current_max_unscaled, current_max, total_max, NEG_INF, RESCALE_THRESHOLD, masked=CFG.MASK_FLAGS != MASK_NONE
                )
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()

                if cutlass.const_expr(SCALE_PREFOLDED):
                    reg_S = reg_S - new_total_max  # FADD2 per pair; the scale is the caller's
                else:
                    reg_S = reg_S * scale_log2 - new_total_max

                # Chunk 1 cast/store/sum folds out at trace time when N_CHUNKS == 1
                chunk_S_0 = reg_S[0:CHUNK].vec
                chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
                hoisted_sum = row_reduction_pair(chunk_P_0)
                chunk_P_0_fp16 = chunk_P_0.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0_fp16)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_P_1 = None
                if cutlass.const_expr(N_CHUNKS == 2):
                    chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
                    deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
                    chunk_P_1_fp16 = deferred_P_1.to(STORAGE_DTYPE)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32), chunk_P_1_fp16)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2):
                    new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair

                bars.mb_stat_empty.wait(stat_empty_phase, spin=SPIN_RING_WAITS)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
        else:
            # LEFT-masked
            for kv_loop in cutlass.range(bounds.left, bounds.unmasked_lo, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase, spin=SPIN_RING_WAITS)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                tmem_base = tmem_ptr_i32.load()
                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF)
                kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)
                raw_chunks = [
                    nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32), num=CHUNK) for c in range(N_CHUNKS)
                ]
                # Bottom-right causal: runtime SKV-SQ diagonal offset (folds out when
                # CFG.BOTTOM_RIGHT is 0 — top-left masking is unchanged).
                causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
                chunks_S = [
                    apply_mask_chunk(
                        raw_chunks[c],
                        q_abs,
                        kv_col_base + cutlass.Int32(c * CHUNK),
                        eff_seqlen_kv,
                        CFG.WINDOW_LEFT,
                        CFG.MASK_FLAGS,
                        N=CHUNK,
                        bottom_right=CFG.BOTTOM_RIGHT,
                        causal_diag=causal_diag,
                        window_right=CFG.WINDOW_RIGHT,
                    )
                    for c in range(N_CHUNKS)
                ]
                chunks_max = [row_max_reduction(chunks_S[c]) for c in range(N_CHUNKS)]
                reg_S_vec = vec_concat(chunks_S)
                current_max_unscaled = chunks_max[0]
                for m in chunks_max[1:]:
                    current_max_unscaled = cute.math.max(current_max_unscaled, m)
                reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
                if cutlass.const_expr(SCALE_PREFOLDED):
                    current_max = current_max_unscaled  # raw-domain max (Q carries attn_scale * log2 e); sentinel note at SCALE_PREFOLDED
                else:
                    current_max = current_max_unscaled * scale_log2

                # Running max / alpha / exp2 shift; a tile fully masked ahead of the row's first live key is selected out of the
                # state (total_max stays NEG_INF, alpha = 1, shift 0 -> P = 0) -- the sentinel never becomes the running max.
                total_max, alpha, new_total_max = running_max_step_finite_sentinel(
                    current_max_unscaled, current_max, total_max, NEG_INF, RESCALE_THRESHOLD, masked=CFG.MASK_FLAGS != MASK_NONE
                )
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()
                if cutlass.const_expr(SCALE_PREFOLDED):
                    reg_S = reg_S - new_total_max  # FADD2 per pair; the scale is the caller's
                else:
                    reg_S = reg_S * scale_log2 - new_total_max

                chunk_S_0 = reg_S[0:CHUNK].vec
                chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
                hoisted_sum = row_reduction_pair(chunk_P_0)
                chunk_P_0_fp16 = chunk_P_0.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0_fp16)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_P_1 = None
                if cutlass.const_expr(N_CHUNKS == 2):
                    chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
                    deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
                    chunk_P_1_fp16 = deferred_P_1.to(STORAGE_DTYPE)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32), chunk_P_1_fp16)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2):
                    new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair
                bars.mb_stat_empty.wait(stat_empty_phase, spin=SPIN_RING_WAITS)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
            # UNMASKED interior
            for kv_loop in cutlass.range(bounds.unmasked_lo, bounds.unmasked_hi, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase, spin=SPIN_RING_WAITS)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                tmem_base = tmem_ptr_i32.load()
                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF)
                reg_S_tile, current_max_unscaled = tmem_load_max_reduction_tile(
                    s_addr_base,
                    num_elems=CFG.TILE_N,
                )
                reg_S = RegTile(reg_S_tile.vec, size=CFG.TILE_N)
                if cutlass.const_expr(SCALE_PREFOLDED):
                    current_max = current_max_unscaled  # raw-domain max (Q carries attn_scale * log2 e); sentinel note at SCALE_PREFOLDED
                else:
                    current_max = current_max_unscaled * scale_log2

                # Running max / alpha / exp2 shift; a tile fully masked ahead of the row's first live key is selected out of the
                # state (total_max stays NEG_INF, alpha = 1, shift 0 -> P = 0) -- the sentinel never becomes the running max.
                total_max, alpha, new_total_max = running_max_step_finite_sentinel(
                    current_max_unscaled, current_max, total_max, NEG_INF, RESCALE_THRESHOLD, masked=CFG.MASK_FLAGS != MASK_NONE
                )
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()
                if cutlass.const_expr(SCALE_PREFOLDED):
                    reg_S = reg_S - new_total_max  # FADD2 per pair; the scale is the caller's
                else:
                    reg_S = reg_S * scale_log2 - new_total_max

                chunk_S_0 = reg_S[0:CHUNK].vec
                chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
                hoisted_sum = row_reduction_pair(chunk_P_0)
                chunk_P_0_fp16 = chunk_P_0.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0_fp16)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_P_1 = None
                if cutlass.const_expr(N_CHUNKS == 2):
                    chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
                    deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
                    chunk_P_1_fp16 = deferred_P_1.to(STORAGE_DTYPE)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32), chunk_P_1_fp16)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2):
                    new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair
                bars.mb_stat_empty.wait(stat_empty_phase, spin=SPIN_RING_WAITS)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
            # RIGHT-masked
            for kv_loop in cutlass.range(bounds.unmasked_hi, bounds.right, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase, spin=SPIN_RING_WAITS)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                tmem_base = tmem_ptr_i32.load()
                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF)
                kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)
                raw_chunks = [
                    nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32), num=CHUNK) for c in range(N_CHUNKS)
                ]
                # Bottom-right causal: runtime SKV-SQ diagonal offset (folds out when
                # CFG.BOTTOM_RIGHT is 0 — top-left masking is unchanged).
                causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
                chunks_S = [
                    apply_mask_chunk(
                        raw_chunks[c],
                        q_abs,
                        kv_col_base + cutlass.Int32(c * CHUNK),
                        eff_seqlen_kv,
                        CFG.WINDOW_LEFT,
                        CFG.MASK_FLAGS,
                        N=CHUNK,
                        bottom_right=CFG.BOTTOM_RIGHT,
                        causal_diag=causal_diag,
                        window_right=CFG.WINDOW_RIGHT,
                    )
                    for c in range(N_CHUNKS)
                ]
                chunks_max = [row_max_reduction(chunks_S[c]) for c in range(N_CHUNKS)]
                reg_S_vec = vec_concat(chunks_S)
                current_max_unscaled = chunks_max[0]
                for m in chunks_max[1:]:
                    current_max_unscaled = cute.math.max(current_max_unscaled, m)
                reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
                if cutlass.const_expr(SCALE_PREFOLDED):
                    current_max = current_max_unscaled  # raw-domain max (Q carries attn_scale * log2 e); sentinel note at SCALE_PREFOLDED
                else:
                    current_max = current_max_unscaled * scale_log2

                # Running max / alpha / exp2 shift; a tile fully masked ahead of the row's first live key is selected out of the
                # state (total_max stays NEG_INF, alpha = 1, shift 0 -> P = 0) -- the sentinel never becomes the running max.
                total_max, alpha, new_total_max = running_max_step_finite_sentinel(
                    current_max_unscaled, current_max, total_max, NEG_INF, RESCALE_THRESHOLD, masked=CFG.MASK_FLAGS != MASK_NONE
                )
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()
                if cutlass.const_expr(SCALE_PREFOLDED):
                    reg_S = reg_S - new_total_max  # FADD2 per pair; the scale is the caller's
                else:
                    reg_S = reg_S * scale_log2 - new_total_max

                chunk_S_0 = reg_S[0:CHUNK].vec
                chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
                hoisted_sum = row_reduction_pair(chunk_P_0)
                chunk_P_0_fp16 = chunk_P_0.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0_fp16)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_P_1 = None
                if cutlass.const_expr(N_CHUNKS == 2):
                    chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
                    deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
                    chunk_P_1_fp16 = deferred_P_1.to(STORAGE_DTYPE)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32), chunk_P_1_fp16)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2):
                    new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair
                bars.mb_stat_empty.wait(stat_empty_phase, spin=SPIN_RING_WAITS)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)

        # Publish final (total_max, total_sum) — correction reads for LSE / inv_sum
        total_sum_scalar = total_sum[0] + total_sum[1]
        stats_addr_epi = tmem_ptr_i32.load() + cutlass.Int32(LAYOUT.STATS_OFF)
        stats_vec_epi = cutlass.Vector.from_elements((total_max, total_sum_scalar), cutlass.Float32)
        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32), stats_vec_epi)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        bars.mb_stat_full.arrive()

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_super_idx, head_idx, batch_idx = _dispatch_decode_payload(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx)


# === Correction warp group ===


@cute.jit
def _correction_warp_group(
    seqlen_q,
    seqlen_kv,
    sO,
    tmem_ptr_i32,
    tidx,
    bars,
    sched,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_in_pair,
    cta_id_x,
    # Fused epilogue gate staging tile (appended; the call is by keyword); the
    # gate barriers ride `bars`.  None = gate off.
    sGate=None,
):
    """Qwen correction warp — bootstrap-only-lo-parity bmm2_ready arrive,
    per-parity bmm2_done phases, LSE / sink fold + write in epilogue."""
    nvvm.barrier_cta_sync(barrier_id=2, thread_count=32 * (CFG.CORRECTION_WARPS + 1))

    tid_raw = cute.arch.thread_idx()[0]
    tid_in_wg = tid_raw - cutlass.Int32(CFG.CORR_WARP_BASE * 32)

    bmm2_done_phase_pair = cutlass.Int32(0)  # bit p = next-wait phase for bmm2_done[p]
    stat_mbar_state = cutlass.Int32(0)
    gate_full_phase = cutlass.Int32(0)  # == EPILOGUE_FUSION_SEAM(corr_state) == wait-then-arrive consumer (P2)
    epilogue_state = cutlass.Int32(1)  # bootstrap pre-armed so first wait passes

    q_super_idx, head_idx, batch_idx = _dispatch_decode_initial(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
    bounds = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx)

    O_CHUNK = 16
    N_CHUNKS_O = CFG.TILE_O // O_CHUNK
    TMA_O_ITERS_LOCAL = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES
    D_BLOCK_SIZE = CFG.TILE_O // TMA_O_ITERS_LOCAL
    TMA_O_GRANU_ELEMS_LOCAL = CFG.TILE_M * D_BLOCK_SIZE

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if bounds.right > bounds.left:
            # Bootstrap-only-lo-parity — fire ONE parity = lo & 1
            lo_parity_rt = bounds.left & cutlass.Int32(1)
            bars.mb_bmm2_ready[lo_parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            # Iter-0: consume start-of-tile stat_full (no rescale)
            bars.mb_stat_full.wait(stat_mbar_state, spin=SPIN_RING_WAITS)
            bars.mb_stat_empty.arrive()
            stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)
        else:
            # Empty mainloop — corr arms MMA's mb_empty_mainloop wait
            bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        for kv_loop in cutlass.range(bounds.left + cutlass.Int32(1), bounds.right, 1, unroll=1):
            parity_prev_rt = (kv_loop - cutlass.Int32(1)) & cutlass.Int32(1)
            parity_cur_rt = kv_loop & cutlass.Int32(1)
            tmem_base_iter = tmem_ptr_i32.load()

            bars.mb_stat_full.wait(stat_mbar_state, spin=SPIN_RING_WAITS)

            stats_addr = tmem_base_iter + cutlass.Int32(LAYOUT.STATS_OFF)
            stats_vec = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(stats_addr, cutlass.Float32),
                num=2,
            )
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            alpha = stats_vec[0]

            alpha_is_one = alpha == cutlass.Float32(1.0)
            all_alpha_one = vote_sync(0xFFFFFFFF, alpha_is_one, VoteSync.ALL)

            bars.mb_stat_empty.arrive()

            bmm2_done_phase_prev = (bmm2_done_phase_pair >> parity_prev_rt) & cutlass.Int32(1)
            bars.mb_bmm2_done[parity_prev_rt].wait(bmm2_done_phase_prev, spin=SPIN_RING_WAITS)
            bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_prev_rt)

            if ~all_alpha_one:
                for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                    o_addr = tmem_base_iter + cutlass.Int32(LAYOUT.O_OFF + chunk_idx * O_CHUNK)
                    o_chunk = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                        num=O_CHUNK,
                    )
                    o_scaled = vec_scale_pair(o_chunk, alpha, O_CHUNK)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), o_scaled)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

            bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)

        tmem_base_epi = tmem_ptr_i32.load()

        # DSL if-staging requires names used after the conditional to be bound on every path
        total_max_scaled = cutlass.Float32(0.0)
        total_sum = cutlass.Float32(0.0)
        # Consumed for EVERY tile: the softmax publishes the end-of-tile stats and waits stat_empty at the next
        # tile start unconditionally, so skipping this on an empty range (zero KV, or a Q-trim-collapsed tile)
        # leaves one stat_full unconsumed and deadlocks the next tile of a persistent worker. The empty tile's
        # (-FLT_MAX, 0) stats fall into the _kv_empty selects below.
        bars.mb_stat_full.wait(stat_mbar_state, spin=SPIN_RING_WAITS)
        stats_addr_epi = tmem_base_epi + cutlass.Int32(LAYOUT.STATS_OFF)
        stats_vec_epi = nvvm.tcgen05_ld(
            "32x32b",
            nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32),
            num=2,
        )
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        total_max_scaled = stats_vec_epi[0]
        total_sum = stats_vec_epi[1]
        bars.mb_stat_empty.arrive()
        stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)

        LN2 = cutlass.Float32(0.6931471805599453)
        total_max_nat = total_max_scaled * LN2
        lse_val = cutlass.Float32(0.0)
        inv_sum = cutlass.Float32(0.0)
        if cutlass.const_expr(CFG.HAS_SINK):
            sinks_arr = cutlass.make_array_view(sinks_tensor)
            sink_logit = sinks_arr[head_idx]
            new_max = cute.math.max(total_max_nat, sink_logit)
            scale = cute.math.exp(total_max_nat - new_max, fastmath=True)
            new_sum = total_sum * scale + cute.math.exp(sink_logit - new_max, fastmath=True)
            lse_val = new_max + cute.math.log(new_sum, fastmath=True)
            inv_sum = scale / new_sum
        else:
            lse_val = total_max_nat + cute.math.log(cute.math.max(total_sum, cutlass.Float32(1e-30)), fastmath=True)
            inv_sum = cutlass.Float32(1.0) / cute.math.max(total_sum, cutlass.Float32(1e-30))

        # --- empty KV range (zero-length sequence under the padding mask) ---
        # Same guard as sm107/prefill_d256_fp8.py: with bounds.right <=
        # bounds.left the mainloop never ran, so total_max/total_sum are still 0
        # and the 1e-30 denominator floor yields LSE = log(1e-30) = -69.08 and
        # inv_sum = 1/1e-30 -> +inf, making O = (TMEM residue) * inf.
        #
        # O looks fine in a casual test only because the residue happens to be
        # zero; with a poisoned accumulator it is NaN, so zero it with a SELECT
        # rather than a multiply (NaN * 0 = NaN).
        q_row_global = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M) + tid_in_wg
        _kv_empty = bounds.right <= bounds.left
        # A row with no live key inside a live tile -- bottom-right rows above the diagonal, a left band
        # past the last key, a zero KV length -- read off the mask geometry: these kernels mask with a
        # finite sentinel, so a keyless row's softmax sum is N, not 0, and cannot tell itself apart.
        if cutlass.const_expr(CFG.MASK_FLAGS != 0):
            _diag = (eff_seqlen_kv - eff_seqlen_q) if cutlass.const_expr(CFG.BOTTOM_RIGHT) else cutlass.Int32(0)
            _last_k = eff_seqlen_kv - cutlass.Int32(1)
            if cutlass.const_expr(CFG.MASK_FLAGS & MASK_CAUSAL):
                _last_k = cute.math.min(_last_k, q_row_global + _diag + cutlass.Int32(CFG.WINDOW_RIGHT))
            _first_k = cutlass.Int32(0)
            if cutlass.const_expr(CFG.MASK_FLAGS & MASK_SWA):
                _first_k = cute.math.max(_first_k, q_row_global + _diag - cutlass.Int32(CFG.WINDOW_LEFT))
            _kv_empty = _kv_empty | (_first_k > _last_k)
        if cutlass.const_expr(not CFG.HAS_SINK):
            # A sink leaves the row with mass, and the branch above already
            # gives LSE = sink_logit there; without one, LSE is -inf.
            lse_val = cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(float("-inf")).ir_value(), lse_val.ir_value()))
        else:
            # A keyless row with a sink holds the sink's mass alone: LSE = sink_logit. The finite mask
            # sentinel, scaled, can overflow to -inf and NaN the sink fold, so the value is selected, not
            # computed; O is zeroed by the same _kv_empty select below.
            lse_val = cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(sink_logit).ir_value(), lse_val.ir_value()))

        if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT):
            # Dense padded-Q trim: q rows >= seq_len_q[b] write O := 0 / LSE := -inf
            # (after the sink branch: a trimmed row is dead even with a sink); folded
            # into _kv_empty so the O cast's select zeroes the row, NaN residue or not.
            _sq_arr = cute.make_tensor(cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24))
            row_trim = q_row_global >= cutlass.Int32(_sq_arr[cutlass.Int32(batch_idx)])
            lse_val = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(float("-inf")).ir_value(), lse_val.ir_value()))
            inv_sum = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(0.0).ir_value(), inv_sum.ir_value()))
            _kv_empty = _kv_empty | row_trim
        # Convert only the final Stats; -inf padding stays -inf.
        if cutlass.const_expr(CFG.STATS_LOG2):
            lse_val = lse_val * cutlass.Float32(1.4426950408889634)
        if cutlass.const_expr(CFG.THD_VARLEN):
            # THD: q_row_global is sequence-local; LSE is packed [1,QH,T] →
            # index [0, head, cu_q[b] + local], bound by per-sequence Q len S_q_b.
            _cu = cutlass.make_array_view(seq_kv_lens_tensor)
            _cu_q_b = cutlass.Int32(_cu[n_batch + batch_idx])
            _s_q_b = cutlass.Int32(_cu[n_batch + batch_idx + cutlass.Int32(1)]) - _cu_q_b
            if cutlass.const_expr(lse_tensor is not None):
                if q_row_global < _s_q_b:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    # Written in the CALLER's layout, picked by the STATIC rank
                    # compile() baked in: token-major rank-2 [T, QH] (the DEFAULT)
                    # or head-major rank-3 [1, QH, head_stride].  Serving only the
                    # rank-3 arm transposes every LSE on the common path.
                    if cutlass.const_expr(len(lse_tensor.shape) == 2):
                        lse_arr[_cu_q_b + q_row_global, head_idx] = lse_val
                    else:
                        if cutlass.const_expr(len(lse_tensor.shape) == 4):
                            # rank-4 = per-batch padded Stats (B, QH, s_max, 1) in the declared strides, no ragged offsets
                            lse_arr[batch_idx, head_idx, q_row_global, 0] = lse_val
                        else:
                            lse_arr[cutlass.Int32(0), head_idx, _cu_q_b + q_row_global] = lse_val
        else:
            if cutlass.const_expr(lse_tensor is not None):
                if q_row_global < seqlen_q:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    lse_arr[batch_idx, head_idx, q_row_global] = lse_val

        # n_kv > 0 → parity_last = (right - 1) & 1; n_kv == 0 → parity_last = 0
        parity_last_rt = cutlass.Int32(0)
        if bounds.right > bounds.left:
            parity_last_rt = (bounds.right - cutlass.Int32(1)) & cutlass.Int32(1)
        bmm2_done_phase_last = (bmm2_done_phase_pair >> parity_last_rt) & cutlass.Int32(1)
        bars.mb_bmm2_done[parity_last_rt].wait(bmm2_done_phase_last, spin=SPIN_RING_WAITS)
        # P14 catch-up flip — bmm2_done_phase_pair ^= (1u << parity_last) AFTER epilogue wait
        bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)

        # "block" = 64/sizeof(ElementO) elems; mb_o_full fires every 2 blocks
        O_EPI_BLK = 64 // CFG.BPE_O
        N_BLOCKS_EPI = CFG.TILE_O // O_EPI_BLK
        CHUNKS_PER_BLK = O_EPI_BLK // O_CHUNK

        sO_base = sO[0].base
        epi_o_full_block_idx = 0

        # == EPILOGUE_FUSION_SEAM(corr_prologue) ==
        # Gate math setup (gate_epilogue_pairs): fold sigmoid's 1/2 into inv_sum
        # so `o_chunk * _gate_inv_sum` below IS h, and hoist the opaque 0.5 once
        # per tile.  inv_sum is already the padded-Q-trimmed value (selected to
        # 0 above); the per-element dead-row SELECT still runs AFTER the gate
        # fma in the chunk loop (sdpa-invariants S2).  Then ONE wait per tile on
        # the staged gate tile -- issued by the TMA-LDG warp after this tile's
        # kv loop, so it has had the whole epilogue tail to land.
        _gate_inv_sum = gate_inv_sum(inv_sum) if cutlass.const_expr(CFG.EPILOGUE_GATE) else inv_sum
        _half_opaque = gate_half_opaque() if cutlass.const_expr(CFG.EPILOGUE_GATE) else cutlass.Float32(0.0)
        if cutlass.const_expr(CFG.EPILOGUE_GATE):
            bars.mb_gate_full.wait(gate_full_phase, spin=SPIN_RING_WAITS)
            gate_full_phase = gate_full_phase ^ cutlass.Int32(1)
        sGate_base = sGate[0].base if cutlass.const_expr(CFG.EPILOGUE_GATE) else None

        for block_idx in cutlass.range_constexpr(N_BLOCKS_EPI):
            for sub in cutlass.range_constexpr(CHUNKS_PER_BLK):
                chunk_idx_total = block_idx * CHUNKS_PER_BLK + sub
                o_addr = tmem_base_epi + cutlass.Int32(LAYOUT.O_OFF + chunk_idx_total * O_CHUNK)
                o_chunk = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                    num=O_CHUNK,
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                # _gate_inv_sum carries sigmoid's 1/2 when gating (o_scaled is h);
                # ungated it is the plain inv_sum.
                o_scaled = o_chunk * _gate_inv_sum
                # == EPILOGUE_FUSION_SEAM(corr_chunk) ==
                # One swizzled LDS of this lane's O_CHUNK gate elements (GATE
                # geometry _GG, never O's) + the packed h*tanh(g/2) + h.  The first
                # gate LDS precedes the mb_o_empty wait below -- a different buffer,
                # no hazard.  A tail tile's rows past seqlen_q are TMA zero-filled
                # (the expect_tx is the FULL box) and the O store clips them anyway.
                _vals = (
                    gate_epilogue_pairs(
                        o_scaled,
                        load_gate_chunk(sGate_base, gate_chunk_smem_offset(chunk_idx_total, O_CHUNK, _GG, tid_in_wg), _GG, O_CHUNK),
                        _half_opaque,
                        O_CHUNK,
                    )
                    if cutlass.const_expr(CFG.EPILOGUE_GATE)
                    else o_scaled
                )
                # SELECT the zero for an empty KV range -- see _kv_empty above.  Per
                # element and AFTER the gate fma: the TMEM residue behind h can be a
                # NaN bit pattern, and NaN * 0 = NaN (sdpa-invariants S2).
                # NB: a plain `for` statement, not a comprehension -- the DSL
                # preprocessor only rewrites statement-level range_constexpr
                # loops ("range_constexpr should be preprocessed by preprocessor").
                _o_elems = []
                for _i in cutlass.range_constexpr(O_CHUNK):
                    _o_elems.append(cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(0.0).ir_value(), _vals[_i].ir_value())))
                o_out = cutlass.Vector.from_elements(tuple(_o_elems), cutlass.Float32).to(OUT_STORAGE_DTYPE)

                col_offset_const = (chunk_idx_total * O_CHUNK) % D_BLOCK_SIZE
                block_offset_const = ((chunk_idx_total * O_CHUNK) // D_BLOCK_SIZE) * TMA_O_GRANU_ELEMS_LOCAL
                smem_offset = cutlass.Int32(block_offset_const + col_offset_const) + tid_in_wg * cutlass.Int32(D_BLOCK_SIZE)
                smem_ptr = sO_base.subview(smem_offset).data_ptr()

                # Gate first sub-chunk on mb_o_empty so sO is reusable
                if block_idx == 0 and sub == 0:
                    bars.mb_o_empty.wait(epilogue_state, spin=SPIN_RING_WAITS)
                smem_ptr.store_swizzled(o_out, alignment=64, swizzle=_O_SMEM_SWIZZLE)

            fire_now = (block_idx % 2 == 1) or (CFG.TILE_O == O_EPI_BLK)
            if cutlass.const_expr(fire_now):
                # fence_proxy needed before TMASTG reads SMEM written by store_swizzled
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_o_full[(block_idx // 2)].arrive()

        # == EPILOGUE_FUSION_SEAM(corr_release) ==
        # Release the gate staging tile: a BARE arrive from every correction lane
        # (128 == CORR_LANES; THREAD has no pred= path), exactly like mb_o_full
        # above.  Generic LDS -> arrive -> producer wait -> TMA write needs no
        # proxy fence.
        if cutlass.const_expr(CFG.EPILOGUE_GATE):
            bars.mb_gate_empty.arrive()

        epilogue_state = epilogue_state ^ cutlass.Int32(1)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        q_super_idx, head_idx, batch_idx = _dispatch_decode_payload(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx)

    if cutlass.const_expr(CFG.CTA_MMA == 2):
        peer_cta = cta_id_x ^ cutlass.Int32(1)
        bars.mb_tmem_dealloc.arrive_on_peer(peer_cta)
    bars.mb_tmem_dealloc.arrive()


# === Host launcher ===


def _require_gate_presence_matches_cfg(gate_tensor) -> None:
    """Trace-time guard for _host (plain Python, runs during the trace): the
    gate tensor's presence must equal the module's CFG.EPILOGUE_GATE.
    compile() builds the fake iff the flag is set, so only a direct
    cute.compile(_host, ...) caller can trip this."""
    if (gate_tensor is not None) != bool(CFG.EPILOGUE_GATE):
        raise ValueError(f"{__name__}: gate_tensor presence ({gate_tensor is not None}) must match CFG.EPILOGUE_GATE={CFG.EPILOGUE_GATE}")


@cute.jit
def _host(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    lse_ptr: Optional[cute.Pointer],
    sinks_ptr: cute.Pointer,
    meta_ptr: cute.Pointer,
    o_desc_ptr: cute.Pointer,
    problem_size: Tuple[int, int, int, int, int, int],
    q_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    k_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    v_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_ext: cutlass.Int64,
    scale_softmax_log2: cutlass.Float32,
    n_thd_units: cutlass.Int32,
    seq_q_lens_addr: cutlass.Int64,
    thd_q_lens_ptr: Optional[cute.Pointer],
    thd_kv_lens_ptr: Optional[cute.Pointer],
    thd_lens_form: Optional[cutlass.Int32],
    # == EPILOGUE_FUSION_SEAM(host_desc) ==
    # Fused epilogue gate, [B, S, H_q, D_v] like O: a pointer + runtime (batch, seq, head) strides;
    # None iff CFG.EPILOGUE_GATE == 0 (sdpa_gate_tensor folds the slot out).  Under THD the gate is the
    # packed [1, T, H_q, D_v] form like Q / O (batch extent 1, the token stride bound there, never stepped).
    gate_ptr: Optional[cute.Pointer],
    gate_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    lse_kind: cutlass.Constexpr[str],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Host entry: device pointers, runtime extents and strides in, TMA encodes and launches out.

    Operands are ``[B, S, H, D]`` with the head dim innermost; ``*_strides`` carry the
    (seq, head) element strides. ``problem_size`` = (B, QH, KH, SQ, SKV, 0); under THD
    SQ/SKV are the packed token totals. Dense batch strides are ``S * seq_stride``; a
    packed THD operand has batch extent 1 and binds the seq stride there. Every stride
    leaf is Int64 (the ``compile()`` fakes fix the width): a 16-bit operand with
    S * H * D >= 2^27 elements would wrap the Int32 TMA-unit scaling.
    ``lse_kind``: "dense" (B, QH, SQ) in ``lse_strides``; "token" (SQ, QH) packed;
    "head" (1, QH, lse_ext); "padded" (B, QH, lse_ext, 1) in ``lse_strides``. None
    pointers compile their paths out."""
    B, QH, KH, SQ, SKV, _ = problem_size
    (
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        sinks_tensor,
        seq_kv_lens_tensor,
        o_desc_words,
        thd_q_lens_tensor,
        thd_kv_lens_tensor,
        o_partial_f32,
        block_table_tensor,
        block_table_v_tensor,
    ) = sdpa_operand_tensors(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        sinks_ptr,
        meta_ptr,
        o_desc_ptr,
        problem_size,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
        lse_strides,
        lse_ext,
        thd_q_lens_ptr,
        thd_kv_lens_ptr,
        thd_lens_form,
        None,
        d_qk=d_qk,
        d_v=d_v,
        lse_kind=lse_kind,
        thd=CFG.THD_VARLEN,
        split_kv=1,  # the SM107 f16 hosts carry no split
        tensor_map_qwords=_TENSOR_MAP_QWORDS,
    )
    gate_tensor = sdpa_gate_tensor(gate_ptr, problem_size, d_v, gate_strides, thd=bool(CFG.THD_VARLEN))
    _require_gate_presence_matches_cfg(gate_tensor)
    stride_order = (3, 2, 1, 0)
    _O_GRANU_ELEMS = CFG.O_SWZ_BYTES // CFG.BPE_O
    qk_box_q = (1, CFG.TILE_M, 1, TMA_QK_GRANU_ELEMS)
    qk_box_k = (1, CFG.TILE_N // CFG.CTA_MMA, 1, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, CFG.TILE_N, 1, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M, 1, _O_GRANU_ELEMS)

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    tma_q_desc = tmap.create_tensor_map_tiled_from_view(
        q_tensor,
        box_dims=qk_box_q,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.Q_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    tma_k_desc = tmap.create_tensor_map_tiled_from_view(
        k_tensor,
        box_dims=qk_box_k,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.K_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    # V TMA: TF32 transposed BMM2 operand needs SWIZZLE_128B_ATOM_32B (same box).
    _v_tma_swz = tmap.TensorMapSwizzle.s128b_atom_32b if IS_TF32 else _tma_swz(CFG.V_SWZ_BYTES)
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_tensor,
        box_dims=vo_box_v,
        stride_order=stride_order,
        swizzle=_v_tma_swz,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    tma_o_desc = tmap.create_tensor_map_tiled_from_view(
        o_tensor,
        box_dims=vo_box_o,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.O_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    # Gate LOAD descriptor in GATE geometry (_GG: box + swizzle from GATE_BPE).
    # Gate-off aliases tma_o_desc rather than passing None: a GridConstant
    # tensor map is always a valid one, and CFG.EPILOGUE_GATE stays the single
    # fold flag (the aliased descriptor is never read).
    tma_gate_desc = (
        tmap.create_tensor_map_tiled_from_view(
            gate_tensor,
            box_dims=_GG.box_dims,
            stride_order=stride_order,
            swizzle=_tma_swz(_GG.swz_bytes),
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )
        if cutlass.const_expr(gate_tensor is not None)
        else tma_o_desc
    )

    rows_per_cluster = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
    q_clusters = (SQ + rows_per_cluster - 1) // rows_per_cluster
    grid_q_supers = q_clusters * CFG.CTA_MMA
    q_supers = grid_q_supers
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD: build the per-batch O descriptor array (reuse tma_o_desc over the
        # packed [1,T,QH,D_v] O as base), then launch the exact flat
        # batch-outermost grid (n_thd_units = Σ_b ceil(S_q_b/CGA_TILE_M)*QH,
        # host-computed); grid_x = n_thd_units * CGA_M.
        _build_thd_meta_o_descs_kernel(
            o_tensor,
            tma_o_desc,
            tma_k_desc,
            tma_v_desc,
            o_desc_words,
            seq_kv_lens_tensor,
            thd_q_lens_tensor,
            thd_kv_lens_tensor,
            thd_lens_form,
            cutlass.Int32(QH),
            cutlass.Int32(B),
            cutlass.Int64(o_tensor.stride[1]),
            cutlass.Int32(CGA_TILE_M),
            n_thd_units,
        ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
        grid_shape = (n_thd_units * cutlass.Int32(CFG.CGA_M), cutlass.Int32(1), cutlass.Int32(1))
    else:
        grid_shape = (grid_q_supers, QH, B) if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL) else (grid_q_supers * QH * B, 1, 1)
    _kernel(
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        tma_o_desc,
        tma_gate_desc,  # == EPILOGUE_FUSION_SEAM(host_launch) ==
        lse_tensor,
        sinks_tensor,
        seq_kv_lens_tensor,
        o_desc_words,
        gate_tensor,
        cutlass.Int32(SQ),
        cutlass.Int32(SKV),
        cutlass.Int32(q_supers),
        cutlass.Int32(QH),
        cutlass.Int32(B),
        cutlass.Int32(QH // KH),
        scale_softmax_log2,
        seq_q_lens_addr,
    ).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(CFG.CTA_MMA, 1, 1),
        stream=stream,
    )


EXPLICIT_ABI = True  # pointer/int host entry; the adapter builds the argument list itself
LSE_KINDS = ("dense", "token", "head", "padded")


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
    has_lse: bool = True,
    lse_kind: str = "dense",
    # == EPILOGUE_FUSION_SEAM(compile) ==
    # The gate is a MODULE specialization (TemplateParams.epilogue_gate -> CFG.EPILOGUE_GATE);
    # its strides are runtime host arguments, so nothing of it is in the compile key.
) -> Callable:
    """Compile the host entry for one layout kind: every extent and stride is a runtime
    argument of the artifact (see ``_host``), so the key is only what specializes the
    traced code — the head-dim envelope where the template has one, whether the LSE
    store exists, the Stats layout kind, and the template's own constexpr flags."""
    _cache_key = _template_key(globals(), locals(), "compile")
    if not (0 < d_qk <= CFG.TILE_K and 0 < d_v <= CFG.TILE_O):
        raise ValueError(f"envelope: need 0 < d_qk <= {CFG.TILE_K} and 0 < d_v <= {CFG.TILE_O}; got ({d_qk}, {d_v})")
    if (d_qk * CFG.BPE) % 16 != 0 or (d_v * CFG.BPE_O) % 16 != 0:
        raise ValueError(f"envelope: d_qk*BPE and d_v*BPE must be 16-byte multiples (TMA global-stride rule); got ({d_qk}, {d_v})")
    if lse_kind not in LSE_KINDS:
        raise ValueError(f"lse_kind must be one of {LSE_KINDS}; got {lse_kind!r}")
    if has_lse and (lse_kind == "dense") == bool(CFG.THD_VARLEN):
        raise ValueError("lse_kind 'dense' is the dense form; 'token' / 'head' / 'padded' are the THD forms")
    gmem = cute.AddressSpace.gmem

    def P(dtype, align=16):
        return cute.runtime.make_ptr(dtype, 16, gmem, assumed_align=align)  # fake: type only

    i32 = cutlass.Int32(0)
    i64_3 = (cutlass.Int64(0),) * 3  # stride slots: Int64 leaves, see _host
    thd = bool(CFG.THD_VARLEN)
    return _compile_cached(
        _host,
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(OUT_STORAGE_DTYPE),
        P(cutlass.Float32, 4) if has_lse else None,
        P(cutlass.Float32),
        P(cutlass.Int32),
        P(cutlass.Int64),
        (0, 0, 0, 0, 0, 0),
        i64_3,
        i64_3,
        i64_3,
        i64_3,
        i64_3,
        cutlass.Int64(0),
        cutlass.Float32(0.0),
        i32,
        cutlass.Int64(0),
        P(cutlass.Int32, 4) if thd else None,
        P(cutlass.Int32, 4) if thd else None,
        i32 if thd else None,
        P(GATE_STORAGE_DTYPE) if CFG.EPILOGUE_GATE else None,
        i64_3,
        d_qk,
        d_v,
        lse_kind,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd",
    )


def _main():
    """Minimal compile-check / perf CLI."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--iters", type=int, default=0)
    args = parser.parse_args()

    lse_kind = "token" if CFG.THD_VARLEN else "dense"
    print(f"[d256_f16] compile lse_kind={lse_kind}", flush=True)
    fn = compile(lse_kind=lse_kind)
    print(f"[d256_f16] compile OK: {fn}", flush=True)
    if args.validate:
        print("[d256_f16] --validate is not implemented in this CLI; use the FE test suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
