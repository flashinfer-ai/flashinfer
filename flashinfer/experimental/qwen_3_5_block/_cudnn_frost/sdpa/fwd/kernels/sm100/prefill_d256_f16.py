# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from functools import lru_cache
from typing import Callable, Optional, Tuple

from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import vote_sync, VoteSync  # noqa: F401
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith, llvm

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver  # noqa: F401

from dataclasses import dataclass
from typing import NamedTuple

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d256

# The per-graph params are injected as a module global by the loader
# (api._load_kernel_module) before this body executes; a plain import gets
# the all-defaults config (dense fp16), which is what the standalone
# `python sm100/prefill_d256_f16.py` benchmark path uses.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
CFG, _TMA = make_cfg_d256(PARAMS)
Cfg = type(CFG)
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS

TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE_O
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import (
    PipelineState,
    advance,
    cga_arrive,
    cga_wait,
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
elif CFG.DTYPE_QKV == 3:
    STORAGE_DTYPE = cutlass.Float16
    P_STORAGE_DTYPE = cutlass.Float16
    MMA_KIND = nvvm.Tcgen05MMAKind.F16

if CFG.DTYPE_O != CFG.DTYPE_QKV:
    raise NotImplementedError(f"prefill_sdpa_d256_f16: DTYPE_O={CFG.DTYPE_O} != DTYPE_QKV={CFG.DTYPE_QKV} not yet supported.")
OUT_STORAGE_DTYPE = STORAGE_DTYPE


from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels._common_blackwell import (
    sdpa_operand_tensors,
    make_split_helpers,
    store_fp32_partial_tile as _store_fp32_partial_tile,
    D256Bars as Bars,
    KvLoopBounds,
    make_d256_bars,
    compute_kv_loop_bounds,
    row_max_for_exp2,
    lpt_tile_coords,
    make_sdpa_helpers,
    sdpa_gate_tensor,
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
CTA_MMA = CFG.CTA_MMA
# THD uses a persistent grid + device-bounded claim counter (not the CLC
# envelope). The adapter reads both this flag and CTA_MMA to size the launch.
THD_PERSISTENT = True

_sdpa_h = make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units=True)
_decode_initial = _sdpa_h.decode_initial
_decode_payload = _sdpa_h.decode_payload
# qtrim variant: collapses the KV loop for CGA tiles entirely past the
# per-batch actual Q length (SEQ_Q_LENS_PRESENT; folds to plain bounds otherwise).
_bounds_for_tile = _sdpa_h.bounds_for_tile_qtrim
_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv
_resolve_seqlen_q = _sdpa_h.resolve_seqlen_q


from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels.thd_origins import build_thd_origins, origin_tensors, origin_tma_slice
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import thd_port_origin
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels.thd_helpers import build_thd_meta_o_descs_kernel as _build_thd_meta_o_descs_kernel, TENSOR_MAP_QWORDS, THD_SETUP_THREADS

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import exit_if_dead_thd_cluster
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels._thd_split_host import host as _host_thd_split, compile_host as _compile_thd_split_host

_TENSOR_MAP_QWORDS = TENSOR_MAP_QWORDS
# The setup kernel builds the THD metadata buffer DEVICE-side from the
# caller's length tensors (issue #552) — no length ever reaches the host — and
# also publishes the live unit total + claim counter the persistent scheduler
# reads, so the adapter launches a MACHINE-sized grid rather than the plan-time
# envelope (issue #618). Metadata layout, in int32 words:
#   [0..B-1]=seq_kv_lens  [B..2B]=cu_q(B+1)  [2B+1..3B+1]=cu_k(B+1)
#   [3B+2..4B+1]=batch_remap(B)  [4B+2]=live units  [4B+3]=claim counter
_dispatch_decode_initial = _sdpa_h.dispatch_decode_initial
_dispatch_decode_payload = _sdpa_h.dispatch_decode_payload
_thd_tma_offsets = _sdpa_h.thd_tma_offsets

# === PackGQA ===
#
# HEADS_PER_TILE = CFG.PACK_G heads share one token row-group of the Q tile
# (row r <-> token r // PACK_G, head r % PACK_G): the whole GQA group when it
# divides TILE_M, else its largest divisor that does (partial PackGQA -- 96/8
# packs 4 of its 12 heads, so the grid's head axis holds QH / 4 packed heads
# and PACKED_HEADS_PER_KV = 3 of them read the same KV head).  CFG.QH_PER_KH
# stays the graph's GQA ratio.  The bounds helpers take HEADS_PER_TILE (the
# packed group), never QH_PER_KH.
HEADS_PER_TILE = CFG.PACK_G if CFG.PACK_GQA else 1
TOKENS_PER_TILE = CFG.TILE_M // HEADS_PER_TILE
PACKED_HEADS_PER_KV = CFG.QH_PER_KH // HEADS_PER_TILE if CFG.PACK_GQA else 1

# === KV split ===
#
# Mechanics live in _common_blackwell.make_split_helpers, shared with the other
# SM100 prefill flavors: each Q tile's KV loop range is cut into SPLIT_KV
# contiguous chunks, each run as its own persistent tile, and each writing a
# normalized partial O + its own LSE into a split-major workspace that
# sm100/split_combine folds with the exact log-sum-exp identity.  At
# SPLIT_KV == 1 every closure folds away and this is the classic kernel.
_split_h = make_split_helpers(
    CFG,
    bounds_for_tile=_bounds_for_tile,
    dispatch_decode_initial=_dispatch_decode_initial,
    dispatch_decode_payload=_dispatch_decode_payload,
)
SPLIT_KV = _split_h.SPLIT_KV
# A split writes fp32 partials, replacing the SMEM/TMA O path rather than
# widening it; the combine performs the only cast to O's dtype.
_FP32_PARTIALS = SPLIT_KV > 1
MAY_BE_EMPTY = _split_h.MAY_BE_EMPTY
_decode_initial_split = _split_h.decode_initial_split
_decode_payload_split = _split_h.decode_payload_split
_bounds_for_tile_split = _split_h.bounds_for_tile_split
_nomask_range_split = _split_h.nomask_range_split
_partial_batch = _split_h.partial_batch

# === Paged KV (see prefill_d128_f16_sm100.py for the design notes) ===
#
# A K tile is TILE_N/CTA_MMA rows per CTA, a V tile the full TILE_N rows; each
# is loaded as a stack of page-sized row boxes (or one box inside a taller
# page), so no box ever straddles a page.
PAGED_KV = bool(CFG.PAGED_KV)
PAGE_SIZE = CFG.PAGE_SIZE if PAGED_KV else 0
_K_TILE_ROWS = CFG.TILE_N // CFG.CTA_MMA
K_BOX_ROWS = min(PAGE_SIZE, _K_TILE_ROWS) if PAGED_KV else _K_TILE_ROWS
V_BOX_ROWS = min(PAGE_SIZE, CFG.TILE_N) if PAGED_KV else CFG.TILE_N
K_BOXES = _K_TILE_ROWS // K_BOX_ROWS
V_BOXES = CFG.TILE_N // V_BOX_ROWS


@dataclass(frozen=True)
class KernelTmemLayout:
    TOTAL_COLS: int = 512

    S_ACC_EVEN_OFF: int = 0
    S_ACC_ODD_OFF: int = 128

    P_EVEN_OFF: int = 64
    P_ODD_OFF: int = 192

    O_OFF: int = 256

    STATS_EVEN_OFF: int = 0
    STATS_ODD_OFF: int = 128
    STATS_EPI_OFF: int = 0


LAYOUT = KernelTmemLayout()


_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
SMEM_LAYOUT_Q = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_K = _SWZ_ENUM[CFG.K_SWZ_BYTES]
SMEM_LAYOUT_V = _SWZ_ENUM[CFG.V_SWZ_BYTES]
SMEM_LAYOUT_O = _SWZ_ENUM[CFG.O_SWZ_BYTES]
SMEM_LAYOUT_QKO = SMEM_LAYOUT_Q

_O_SWZ_B = {128: 3, 64: 2, 32: 1}[CFG.O_SWZ_BYTES]
_O_SMEM_SWIZZLE = cutlass.Swizzle(_O_SWZ_B, 4, 3)

LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES

_CORE_MATRIX_ROWS = 8
_V_PC_COLS = CFG.TILE_O // CFG.CTA_MMA
LEADING_BYTE_OFFSET_PV = 0 if (_V_PC_COLS // _CORE_MATRIX_ROWS) <= 8 else CFG.TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_PV = 8 * CFG.V_SWZ_BYTES

NUM_KPHASES_PV = CFG.TILE_N // CFG.TILE_K_HW_BMM2
NUM_KPHASES_PV_PER_CHUNK = NUM_KPHASES_PV // CFG.N_BMM2_CHUNKS

# === Fused epilogue gate (O := O * sigmoid(G)), CFG.EPILOGUE_GATE <- TemplateParams.epilogue_gate ===
#
# The sm107 d256 body's gate, on THIS body's paged / THD / PackGQA arms (the serving EXTEND form's SDPA is this body
# compiled for sm_107a): G is an O-shaped Q-dtype tensor, TMA-staged by the TMA-LDG warp AFTER the KV loop into a
# 64 KiB SMEM tile (sGate, declared after the V ring: 192..256 KiB, so every MMA operand keeps its version-0
# descriptor root under 256 KiB -- the tile is LDS-read only) and consumed in the correction epilogue as the packed
# ``h * tanh(g/2) + h`` (the shared hook in ``_common_blackwell``).  Two LOCAL barriers ride ``bars``: mb_gate_full
# (TMA_LOAD, init ONE_LANE: one elected lane of the TMA-LDG warp arms 64 KiB per tile, cta_group=1 -- each CTA of the
# pair owns a distinct 128-row Q block at the full TILE_O width, so no CTA_MMA factor) and mb_gate_empty (THREAD, init
# CORR_LANES: every correction lane arrives once per tile after its last gate LDS; the TMA-LDG warp waits it at the
# TOP of its tile iteration, pre-armed).  The gate tile's TMA box is the Q / O box's PACKED-ROW convention
# ``(1, TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, 64)``, so under PackGQA SMEM row r = packed row r = the thread that
# reads it; the load coordinates are the Q load's (q_head_idx, q_row_base + q_seq_off, tma_batch) under every arm --
# dense, THD (the packed [1, T, H, D] map Q / O take) and paged (the gate is a Q-side operand).  Every seam folds on
# ``cutlass.const_expr(EPILOGUE_GATE)`` and every new kernel argument is ``Optional[...] = None``, so the ungated
# renderings trace byte for byte (the render gate: the sm_100a AND sm_107a cubins of the dense / THD / paged / PackGQA
# renderings are md5-pinned).  The split arm (fp32 partials) carries no gate: config_sm100 refuses the pair.
EPILOGUE_GATE = bool(CFG.EPILOGUE_GATE)
GATE_STORAGE_DTYPE = STORAGE_DTYPE
if EPILOGUE_GATE and CFG.BPE != CFG.GATE_BPE:
    # Structural invariant of THIS body: the gate is stored in Q's dtype, so its byte width must be the one the geometry
    # (and the config's SMEM tally) were derived from.
    raise NotImplementedError(f"{__name__}: the epilogue gate is Q-dtype here ({CFG.BPE} B/elem) but CFG.GATE_BPE={CFG.GATE_BPE}")
if EPILOGUE_GATE and SPLIT_KV > 1:
    raise ValueError(f"{__name__}: the epilogue gate has no arm on the split (fp32 partials) path; config_sm100._validate_params refuses the pair")
# Built for every module (pure host arithmetic) so the gate-off body never references an unbound name.
_GG = gate_geometry(CFG, gate_bpe=CFG.GATE_BPE, swz_enum=_SWZ_ENUM)
gateBufferElems = _GG.buffer_elems
# cta_group=1 (shared::cta): every byte lands on THIS CTA's mbar -- no ``* CFG.CTA_MMA`` (contrast qTmaTransactionBytes).
gateTmaTransactionBytes = _GG.tx_bytes
SMEM_LAYOUT_GATE = _GG.layout_enum


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_o_desc: cutlass.GridConstant[tmap.TensorMap],
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    o_desc_words: cute.Tensor,
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    n_q_supers: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    qh_per_kh: cutlass.Int32,
    scale_softmax_log2: cutlass.Float32,
    # Dense padded-Q trim: separate (B,)-int32 per-batch Q lengths (mirrors
    # cuDNN's SEQLEN_Q pointer / FA's seqused_q). None unless
    # CFG.SEQ_Q_LENS_PRESENT — the DSL specializes on None, so the flag-off
    # ABI is unchanged.
    seq_q_lens_addr: cutlass.Int64 = 0,
    o_partial_f32: Optional[cute.Tensor] = None,
    # Paged KV: [B, max_pages] int32 page ids per batch for K and for V; None
    # (folded out of the ABI) unless CFG.PAGED_KV. paged_hnd: HND pool (row
    # stride below head stride) -> descriptor dims (D, row, H_kv, page).
    block_table_tensor: Optional[cute.Tensor] = None,
    block_table_v_tensor: Optional[cute.Tensor] = None,
    paged_hnd: cutlass.Constexpr[bool] = False,
    # Fused epilogue gate: the gate tile's LOAD descriptor; None (dropped from the specialised signature, so the ungated
    # cubin does not move) unless CFG.EPILOGUE_GATE.  Appended LAST.
    tma_gate_desc: Optional[cutlass.GridConstant[tmap.TensorMap]] = None,
    thd_origins: Optional[cute.Tensor] = None,
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    if cutlass.const_expr(CFG.THD_VARLEN and (SPLIT_KV > 1 or thd_origins is not None)):
        exit_if_dead_thd_cluster(seq_kv_lens_tensor, n_batch, CFG.CGA_M)

    sQO_raw = cutlass.Array(STORAGE_DTYPE, qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * kBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * vBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # Gate staging tile: declared right AFTER the V ring (starts at 192 KiB at STAGES_KV=2, ends at 256 KiB) and BEFORE the
    # bars' _alloc()s, tmem_ptr_i32 and the sched arrays -- none of those is an MMA / UTCCP operand, and sGate itself is LDS-read
    # only, so every MMA operand root stays under the version-0 descriptor window (mma-tma-matrix S6).  SW128 swizzle has BOTH
    # jobs (frost-kernels S5): the TMA descriptor writes it, the correction lanes read it one packed row (128 B) per lane.
    sGate_raw = (
        cutlass.Array(GATE_STORAGE_DTYPE, gateBufferElems, alignment=1024, space=cutlass.AddressSpace.smem) if cutlass.const_expr(EPILOGUE_GATE) else None
    )

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
    )
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
        )
        if cutlass.const_expr(EPILOGUE_GATE)
        else None
    )

    bars = make_d256_bars(CFG, N_O_CHUNKS=N_O_CHUNKS, epilogue_gate=EPILOGUE_GATE)
    # Half fused schedule, per CTA: stat_full has 128 softmax arrivals for
    # shared alpha/max, stat_empty has 128 correction arrivals after S-hi is
    # loaded. P0 aliases S[64:96] and waits on stat_empty; P1 aliases S[96:128]
    # and follows stat_full, whose producer completes all S reads. Both edges
    # have tcgen05 completion plus before/after-thread-sync fences (Rule S3).
    # Across the CTA pair, ready[chunk0] still has 256 softmax + 256 correction
    # arrivals; ready[chunk1] now has 256 correction arrivals instead of softmax.
    # Counts, scheduler credits and phase cadence are unchanged.
    softmax_exchange = None
    if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
        softmax_exchange = cutlass.Array(cutlass.Float32, 4 * CFG.TILE_M, alignment=16, space=cutlass.AddressSpace.smem)

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
            if cutlass.const_expr(EPILOGUE_GATE):
                bars.mb_gate_full.init()
                bars.mb_gate_empty.init()

            bars.mb_q_o_alias.arrive()

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    if cutlass.const_expr(CFG.CTA_MMA == 2):
        cga_arrive()
        cga_wait()

    cta_id_x = cute.arch.block_idx_in_cluster() if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    cta_in_pair = (cta_id_x & cutlass.Int32(1)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    leader_cta_id = (cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    mcast_mask = (cutlass.Int32(3) << leader_cta_id) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    tma_mcast_mask = (cutlass.Int16(1) << cta_in_pair) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int16(0)
    is_leader = cta_in_pair == cutlass.Int32(0)

    if warp_idx >= CFG.SOFTMAX_WG0_BASE and warp_idx < CFG.SOFTMAX_WG0_BASE + CFG.SOFTMAX_WG_WARPS:
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            softmax_exchange=softmax_exchange,
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
            qh_per_kh=qh_per_kh,
        )

    elif warp_idx >= CFG.CORR_WARP_BASE and warp_idx < CFG.CORR_WARP_BASE + CFG.CORRECTION_WARPS:
        if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
            nvvm.setmaxregister(CFG.CORRECTION_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        else:
            nvvm.setmaxregister(CFG.CORRECTION_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _correction_warp_group(
            softmax_exchange=softmax_exchange,
            scale_log2=scale_softmax_log2,
            thd_origins=thd_origins,
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
            qh_per_kh=qh_per_kh,
            o_partial_f32=o_partial_f32,
            sGate=sGate,
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
                    qh_per_kh=qh_per_kh,
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
                qh_per_kh=qh_per_kh,
            )

    elif warp_idx == CFG.TMALDG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        if cutlass.const_expr(EPILOGUE_GATE):
            nvvm.prefetch_tensormap(tma_gate_desc.get_ptr())
        _tmaldg_warp_group(
            thd_origins=thd_origins,
            tma_q_desc=tma_q_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            sQ=sQ,
            sK=sK,
            sV=sV,
            bars=bars,
            sched=sched,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            o_desc_words=o_desc_words,
            qh_per_kh=qh_per_kh,
            is_leader=is_leader,
            cta_in_pair=cta_in_pair,
            tma_mcast_mask=tma_mcast_mask,
            block_table_tensor=block_table_tensor,
            block_table_v_tensor=block_table_v_tensor,
            paged_hnd=paged_hnd,
            tma_gate_desc=tma_gate_desc,
            sGate=sGate,
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
            o_desc_words=o_desc_words,
            qh_per_kh=qh_per_kh,
            seqlen_kv=seqlen_kv,
        )

    else:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        is_cga_first_cta = cta_id_x == cutlass.Int32(0)
        if cutlass.const_expr(CFG.THD_VARLEN):
            # THD: persistent grid + device-bounded claim counter, so no unit
            # past the live total is ever handed out (the CLC path would need
            # the grid to BE the work list, i.e. the plan-time envelope).
            # n_batch is a kernel argument -- do NOT re-derive it from the
            # metadata tensor's layout.
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


# Only the common dense CLC path has a single scheduler publisher.
# THD/predecode retains local schedulers and broadcast credits.
if not CFG.THD_VARLEN:
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.scheduler import read_tile_id_arrive_on_leader as read_tile_id_arrive


@cute.jit
def _paged_load_tile(
    smem_tile,
    tma,
    block_table_tensor,
    batch_idx,
    kv_head_idx,
    n_pages_b,
    kv_tile,
    row_off,
    d_coord,
    mbar,
    tma_mcast_mask,
    box_rows: cutlass.Constexpr[int],
    n_boxes: cutlass.Constexpr[int],
    granu_elems: cutlass.Constexpr[int],
):
    """One paged K or V tile as ``n_boxes`` row boxes through the block table
    (same contract as prefill_d128_f16_sm100._paged_load_tile: boxes past the
    batch's live pages take page -1 = TMA-OOB zeros, bytes still credited)."""
    bt = cutlass.make_array_view(block_table_tensor)
    last_live = cute.math.max(n_pages_b - cutlass.Int32(1), cutlass.Int32(0))
    for j in cutlass.range_constexpr(n_boxes):
        g = kv_tile * cutlass.Int32(CFG.TILE_N) + row_off + cutlass.Int32(j * box_rows)
        slot = g // cutlass.Int32(PAGE_SIZE)
        row_in_page = g % cutlass.Int32(PAGE_SIZE)
        page_live = cutlass.Int32(bt[batch_idx, cute.math.min(slot, last_live)])
        in_range = slot < n_pages_b
        page = cutlass.Int32(arith.select(in_range.ir_value(), page_live.ir_value(), cutlass.Int32(-1).ir_value()))
        tma_load_tile(
            smem_tile.shifted(j * box_rows * granu_elems),
            tma(d_coord, kv_head_idx, row_in_page, page),
            mbar,
            cta_group=CFG.CTA_MMA,
            mcast_mask=tma_mcast_mask,
        )


@cute.jit
def _tmaldg_warp_group(
    tma_q_desc,
    tma_k_desc,
    tma_v_desc,
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
    o_desc_words,
    qh_per_kh,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
    block_table_tensor=None,
    block_table_v_tensor=None,
    paged_hnd: cutlass.Constexpr[bool] = False,
    # Fused epilogue gate (appended; the call is by keyword).  None = gate off.
    tma_gate_desc=None,
    sGate=None,
    thd_origins=None,
):
    q_o_alias_phase = cutlass.Int32(0)
    kv_state = PipelineState.start(phase=1)
    # mb_gate_empty consumer phase, PRE-ARMED at 1 (P5b): the first tile's staging buffer is untouched, so the first wait
    # must return with no producer arrive.  Loop-carried like q_o_alias_phase.
    gate_empty_phase = cutlass.Int32(1)

    if cutlass.const_expr(thd_origins is not None):
        tma_q = lambda d, h, r, b: origin_tma_slice(o_desc_words, 1, n_batch, b, d, h, r)
    else:
        tma_q = GmemTileTma(tma_q_desc)
    tma_gate = GmemTileTma(tma_gate_desc) if cutlass.const_expr(EPILOGUE_GATE) else None
    if cutlass.const_expr(thd_origins is not None and not PAGED_KV):
        tma_k = lambda d, h, r, b: origin_tma_slice(o_desc_words, 2, n_batch, b, d, h, r)
        tma_v = lambda d, h, r, b: origin_tma_slice(o_desc_words, 3, n_batch, b, d, h, r)
    elif cutlass.const_expr(CFG.THD_VARLEN and not PAGED_KV):
        # THD: K/V ride the setup kernel's packed-total-clamped runtime
        # descriptors (o_desc_words slots n_batch+1 / n_batch+2), so the last
        # sequence's tile-tail lands as exact zeros instead of reading the
        # buffer's capacity tail (issue #624). Same closure shape as the dense
        # GmemTileTma, so every load site below stays branch-free.
        _k_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(1)) * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        _v_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(2)) * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        # Setup publishes immutable packed-total descriptors before this kernel.
        # Every loader warp, including both CTAs, acquires each map once.
        tma_tensormap_acquire(_k_rt_ptr)
        tma_tensormap_acquire(_v_rt_ptr)
        tma_k = lambda *coords: tma_slice_runtime_desc(_k_rt_ptr, *coords)  # noqa: E731
        tma_v = lambda *coords: tma_slice_runtime_desc(_v_rt_ptr, *coords)  # noqa: E731
    elif cutlass.const_expr(PAGED_KV and paged_hnd):
        # HND page pools: descriptor dims are (D, row, H_kv, page); every load
        # site keeps the (d, head, row, page) vocabulary, the swap lives here.
        _tk, _tv = GmemTileTma(tma_k_desc), GmemTileTma(tma_v_desc)
        tma_k = lambda d, h, r, p: _tk(d, r, h, p)  # noqa: E731
        tma_v = lambda d, h, r, p: _tv(d, r, h, p)  # noqa: E731
    else:
        tma_k = GmemTileTma(tma_k_desc)
        tma_v = GmemTileTma(tma_v_desc)

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        qh_per_kh,
        seqlen_kv,
    )
    # GQA: K/V are indexed by kv-head, not Q-head; with PackGQA the decoded
    # head_idx is the PACKED head (Q head base = head_idx * PACK_G), the KV head
    # is packed head // PACKED_HEADS_PER_KV (== head_idx when the whole group
    # packs) and q_row_base is in TOKEN units (rows // PACK_G).
    q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
    kv_head_idx = cute.arch.make_warp_uniform(
        (head_idx if cutlass.const_expr(PACKED_HEADS_PER_KV == 1) else head_idx // cutlass.Int32(PACKED_HEADS_PER_KV))
        if cutlass.const_expr(CFG.PACK_GQA)
        else head_idx // qh_per_kh
    )
    q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE))
    q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)

    if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    elif cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    # Paged KV: this batch's live page count bounds the block-table walk
    # (MASK_PADDED is mandatory under PAGED_KV, so eff_seqlen_kv is defined).
    n_pages_b = cutlass.Int32(0)
    if cutlass.const_expr(PAGED_KV):
        n_pages_b = (eff_seqlen_kv + cutlass.Int32(PAGE_SIZE - 1)) // cutlass.Int32(PAGE_SIZE)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)
    V_COL_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)

    while is_valid_tile > cutlass.Int32(0):
        if cutlass.const_expr(thd_origins is not None):
            # Sequence-local maps have independent caller-port bases and clip
            # every interior tail. Acquire again when a persistent unit changes.
            stride = (n_batch + cutlass.Int32(1)) * cutlass.Int32(TENSOR_MAP_QWORDS)
            base = o_desc_words.iterator.raw_ptr() + batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
            q_map = (base + stride).tospace(cutlass.AddressSpace.generic)
            tma_tensormap_acquire(q_map)
            if cutlass.const_expr(not PAGED_KV):
                k_map = (base + 2 * stride).tospace(cutlass.AddressSpace.generic)
                v_map = (base + 3 * stride).tospace(cutlass.AddressSpace.generic)
                tma_tensormap_acquire(k_map)
                tma_tensormap_acquire(v_map)
            q_seq_off, kv_seq_off, tma_batch = cutlass.Int32(0), cutlass.Int32(0), batch_idx
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        bars.mb_q_o_alias.wait(q_o_alias_phase)
        q_o_alias_phase = q_o_alias_phase ^ cutlass.Int32(1)

        # Gate staging-buffer reuse: ONE pre-armed wait per tile at the TOP of the iteration (the mb_q_o_alias wait above already
        # parks this warp behind the previous tile's O store, which follows the epilogue's gate consume).  Folds away ungated.
        if cutlass.const_expr(EPILOGUE_GATE):
            bars.mb_gate_empty.wait(gate_empty_phase)
            gate_empty_phase = gate_empty_phase ^ cutlass.Int32(1)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            # P14: the epilogue runs on every tile and waits mb_gate_full each time, so an empty tile still consumes one gate
            # load (folds to nothing when the gate is compiled out; MAY_BE_EMPTY itself is False on the gated rendering).
            issue_gate_load(sGate, tma_gate, bars.mb_gate_full, gateTmaTransactionBytes, q_head_idx, q_row_base + q_seq_off, tma_batch)
        else:
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                bars.mb_q_full.arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            else:
                bars.mb_q_full.arrive(n_bytes=qTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(
                sQ[0],
                tma_q(cutlass.Int32(0), q_head_idx, q_row_base + q_seq_off, tma_batch),
                bars.mb_q_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )

            for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
                kv_row_base = kv_loop * cutlass.Int32(CFG.TILE_N)

                bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=nvvm.elect_sync())
                if cutlass.const_expr(PAGED_KV):
                    _paged_load_tile(
                        sK[kv_state.idx],
                        tma_k,
                        block_table_tensor,
                        batch_idx,
                        kv_head_idx,
                        n_pages_b,
                        kv_loop,
                        K_ROW_OFFSET_PEER,
                        cutlass.Int32(0),
                        bars.mb_k_full[kv_state.idx].smem_ptr,
                        tma_mcast_mask,
                        K_BOX_ROWS,
                        K_BOXES,
                        TMA_QK_GRANU_ELEMS,
                    )
                else:
                    tma_load_tile(
                        sK[kv_state.idx],
                        tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, tma_batch),
                        bars.mb_k_full[kv_state.idx].smem_ptr,
                        cta_group=CFG.CTA_MMA,
                        mcast_mask=tma_mcast_mask,
                        acquire=not (CFG.THD_VARLEN and not PAGED_KV),
                    )

                bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=nvvm.elect_sync())
                if cutlass.const_expr(PAGED_KV):
                    _paged_load_tile(
                        sV[kv_state.idx],
                        tma_v,
                        block_table_v_tensor,
                        batch_idx,
                        kv_head_idx,
                        n_pages_b,
                        kv_loop,
                        cutlass.Int32(0),
                        V_COL_OFFSET_PEER,
                        bars.mb_v_full[kv_state.idx].smem_ptr,
                        tma_mcast_mask,
                        V_BOX_ROWS,
                        V_BOXES,
                        TMA_VO_GRANU_ELEMS,
                    )
                else:
                    tma_load_tile(
                        sV[kv_state.idx],
                        tma_v(V_COL_OFFSET_PEER, kv_head_idx, kv_row_base + kv_seq_off, tma_batch),
                        bars.mb_v_full[kv_state.idx].smem_ptr,
                        cta_group=CFG.CTA_MMA,
                        mcast_mask=tma_mcast_mask,
                        acquire=not (CFG.THD_VARLEN and not PAGED_KV),
                    )

                kv_state = advance(kv_state, CFG.STAGES_KV)

            # Gate load AFTER the kv loop: the TMA engine serves requests in order and the gate is the one operand with slack
            # (consumed only in the epilogue), while K/V have none.  The Q load's coordinates: the packed head base, the token
            # row base (+ the THD sequence offset), the batch -- one 64 KiB tile per Q tile, cta_group=1, ONE expect_tx.
            issue_gate_load(sGate, tma_gate, bars.mb_gate_full, gateTmaTransactionBytes, q_head_idx, q_row_base + q_seq_off, tma_batch)

        if nvvm.elect_sync():
            bars.mb_tmastg_go.arrive()
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
        kv_head_idx = cute.arch.make_warp_uniform(
            (head_idx if cutlass.const_expr(PACKED_HEADS_PER_KV == 1) else head_idx // cutlass.Int32(PACKED_HEADS_PER_KV))
            if cutlass.const_expr(CFG.PACK_GQA)
            else head_idx // qh_per_kh
        )
        q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE))
        q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV > 1):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        elif cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
            bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)
            kv_left = bounds_next.left
            kv_right = bounds_next.right
            if cutlass.const_expr(PAGED_KV):
                n_pages_b = (eff_seqlen_kv + cutlass.Int32(PAGE_SIZE - 1)) // cutlass.Int32(PAGE_SIZE)

    if cutlass.const_expr(CFG.CTA_MMA == 2):
        for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
            bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
            bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
            kv_state = advance(kv_state, CFG.STAGES_KV)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


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
    o_desc_words,
    seqlen_kv,
    qh_per_kh,
):
    tmastg_go_phase = cutlass.Int32(0)
    o_full_phase = cutlass.Int32(0)

    tma_o = GmemTileTma(tma_o_desc)

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        qh_per_kh,
        seqlen_kv,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    # Each per-sequence O map is immutable for this launch.
    last_o_batch = cutlass.Int32(-1)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        bars.mb_tmastg_go.wait(tmastg_go_phase)
        tmastg_go_phase = tmastg_go_phase ^ cutlass.Int32(1)

        for chunk in cutlass.range_constexpr(N_O_CHUNKS):
            bars.mb_o_full[chunk].wait(o_full_phase)

        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE)
        q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
        # KV split: partials are stacked split-major on the workspace BATCH
        # axis (extent B*SPLIT_KV), so the store needs no new descriptor —
        # only a shifted batch coord.  Folds to batch_idx at SPLIT_KV == 1.
        o_batch = _partial_batch(batch_idx, split_idx, n_batch)

        # fp32 partials wrote the workspace directly, so nothing is staged to
        # copy.  Skip ONLY the store: the arrives below, the QO_ALIAS
        # handshake and the phase flips must all still run.
        if cutlass.const_expr(not _FP32_PARTIALS):
            if cutlass.const_expr(CFG.THD_VARLEN):
                # DEAD unit (batch == n_batch, over-launched grid — issue #552):
                # no O rows exist and descriptor slot n_batch is never built, so
                # skip the store; the barrier protocol below still runs.
                if batch_idx < n_batch:
                    o_desc_ptr = (o_desc_words.iterator.raw_ptr() + batch_idx * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
                    if batch_idx != last_o_batch:
                        tma_tensormap_acquire(o_desc_ptr)
                        last_o_batch = batch_idx
                    o_slice = tma_slice_runtime_desc(o_desc_ptr, cutlass.Int32(0), q_head_idx, q_row_coord, cutlass.Int32(0))
                    tma_store_tile(sO[0], o_slice, acquire=False)
            else:
                tma_store_tile(
                    sO[0],
                    tma_o(cutlass.Int32(0), q_head_idx, q_row_coord, o_batch),
                )
            tma_store_commit()
            tma_store_wait(0)

        bars.mb_o_empty.arrive()
        if nvvm.elect_sync():
            bars.mb_q_o_alias.arrive()

        o_full_phase = o_full_phase ^ cutlass.Int32(1)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


@cute.jit
def _mma_warp_quiet(tmem_ptr_i32, bars):
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
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
    qh_per_kh,
):
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
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

    if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    else:
        q_super_idx, _hd, batch_idx, split_idx = _decode_initial_split(
            sched.bidx_init,
            sched.bidy_init,
            sched.bidz_init,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        if cutlass.const_expr(CFG.MASK_FLAGS == 0):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        else:
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
            bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)
            kv_left = bounds_init.left
            kv_right = bounds_init.right

    q_full_phase = cutlass.Int32(0)
    kv_state_K = PipelineState.start(phase=0)
    kv_state_V = PipelineState.start(phase=0)
    bmm2_ready_phase_pair = cutlass.Int32(0)
    empty_mainloop_phase = cutlass.Int32(0)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            bars.mb_empty_mainloop.wait(empty_mainloop_phase)
            empty_mainloop_phase = empty_mainloop_phase ^ cutlass.Int32(1)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        else:
            bars.mb_q_full.wait(q_full_phase)
            q_full_phase = q_full_phase ^ cutlass.Int32(1)

            lo_parity_runtime = kv_left & cutlass.Int32(1)
            parity_lo_is_even = lo_parity_runtime == cutlass.Int32(0)
            tmem_S_acc_lo_addr = cutlass.Int32(
                arith.select(
                    parity_lo_is_even.ir_value(),
                    cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(),
                    cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value(),
                )
            )

            bars.mb_k_full[kv_state_K.idx].wait(kv_state_K.phase)
            desc_K = sK[kv_state_K.idx].desc()
            mma_ss(bmm1_desc, desc_Q, desc_K, (tmem_raw.subview(tmem_S_acc_lo_addr)))
            elect_p = nvvm.elect_sync()
            bars.mb_bmm1_done[lo_parity_runtime].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_k_empty[kv_state_K.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

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

                bars.mb_k_full[kv_state_K.idx].wait(kv_state_K.phase)
                desc_K = sK[kv_state_K.idx].desc()
                mma_ss(bmm1_desc, desc_Q, desc_K, (tmem_raw.subview(tmem_S_acc_next_addr)))
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[parity_next_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_k_empty[kv_state_K.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

                bars.mb_v_full[kv_state_V.idx].wait(kv_state_V.phase)
                desc_V = sV[kv_state_V.idx].desc()

                scaleC = cutlass.Boolean(kv_loop != kv_left)
                accum_b2 = scaleC
                for k in cutlass.range_constexpr(NUM_KPHASES_PV):
                    if k % k_per_chunk == 0:
                        chunk_id = k // k_per_chunk
                        bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase_cur)
                        if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                            _fused_after_sync()
                    mma_ts_step(bmm2_desc, (tmem_raw.subview(tmem_P_cur_addr)), desc_V, (tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF))), k, accum_b2)
                    accum_b2 = cutlass.Boolean(True)
                elect_p = nvvm.elect_sync()
                bars.mb_bmm2_done[parity_cur_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_v_empty[kv_state_V.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_cur_rt)
                kv_state_V = advance(kv_state_V, CFG.STAGES_KV)

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

            bars.mb_v_full[kv_state_V.idx].wait(kv_state_V.phase)
            desc_V = sV[kv_state_V.idx].desc()
            n_kv_eff = kv_right - kv_left
            scaleC_epi = cutlass.Boolean(n_kv_eff != cutlass.Int32(1))
            accum_b2 = scaleC_epi
            for k in cutlass.range_constexpr(NUM_KPHASES_PV):
                if k % k_per_chunk == 0:
                    chunk_id = k // k_per_chunk
                    bars.mb_bmm2_ready[parity_last_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase_last)
                    if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                        _fused_after_sync()
                mma_ts_step(bmm2_desc, (tmem_raw.subview(tmem_P_last_addr)), desc_V, (tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF))), k, accum_b2)
                accum_b2 = cutlass.Boolean(True)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[parity_last_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_v_empty[kv_state_V.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)
            kv_state_V = advance(kv_state_V, CFG.STAGES_KV)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
            _nq, _nh, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
            is_valid_tile = nxt_v & cutlass.Int32(1)
        else:
            nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
            nxt_q = cute.arch.make_warp_uniform(nxt_q)
            nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
            nxt_v = cute.arch.make_warp_uniform(nxt_v)
            q_super_idx, _hd, batch_idx, split_idx = _decode_payload_split(
                nxt_q,
                nxt_hb,
                cta_in_pair,
                n_q_supers,
                n_qh,
                n_batch,
                seq_kv_lens_tensor,
                qh_per_kh,
                seqlen_kv,
            )
            is_valid_tile = nxt_v & cutlass.Int32(1)
            if cutlass.const_expr(CFG.MASK_FLAGS == 0):
                kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
            else:
                eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
                eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
                bounds_next = _bounds_for_tile_split(
                    q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE
                )
                kv_left = bounds_next.left
                kv_right = bounds_next.right
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _fused_before_sync():
    llvm.inline_asm(None, [], "tcgen05.fence::before_thread_sync;", "", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def _fused_after_sync():
    llvm.inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


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
    qh_per_kh,
    softmax_exchange=None,
):
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    bmm1_done_phase_pair = cutlass.Int32(0)
    stat_empty_phase = cutlass.Int32(1)
    epilogue_state = cutlass.Int32(1)

    NEG_INF = cutlass.Float32(float("-inf"))

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        qh_per_kh,
        seqlen_kv,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)

    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)

    tid_in_wg = cute.arch.thread_idx()[0] - cutlass.Int32(CFG.SOFTMAX_WG0_BASE * 32)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        total_max = NEG_INF
        total_max_safe = NEG_INF
        total_sum = cutlass.Vector.from_elements(
            (cutlass.Float32(0.0), cutlass.Float32(0.0)),
            cutlass.Float32,
        )
        # PackGQA: q_abs is the row's TOKEN index (row // PACK_G): every mask
        # predicate downstream is a token-space compare, and all PACK_G rows of
        # one token share it.
        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE)
        q_abs = q_row_coord + (tid_in_wg // cutlass.Int32(HEADS_PER_TILE))

        bars.mb_o_empty.wait(epilogue_state)
        bars.mb_stat_empty.wait(stat_empty_phase)
        stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
        epilogue_state = epilogue_state ^ cutlass.Int32(1)

        CHUNK = 64
        P_COLS_PER_CHUNK = CHUNK // 2
        N_CHUNKS = CFG.N_BMM2_CHUNKS
        RESCALE_THRESHOLD = cutlass.Float32(CFG.RESCALE_THRESHOLD)

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
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    _fused_after_sync()
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)

                tmem_base = tmem_ptr_i32.load()
                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + s_off_rt

                raw_chunks = [
                    nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32), num=CHUNK) for c in range(N_CHUNKS)
                ]
                chunks_max = [row_max_reduction(raw_chunks[c]) for c in range(N_CHUNKS)]
                current_max_unscaled = chunks_max[0]
                for _m in chunks_max[1:]:
                    current_max_unscaled = cute.math.max(current_max_unscaled, _m)
                reg_S = RegTile(vec_concat(raw_chunks), size=CFG.TILE_N)
                current_max = current_max_unscaled * scale_log2  # -inf when the whole iteration is masked

                # total_max starts at -inf: a live iteration always clears the
                # threshold (real - (-inf) = +inf), a fully-masked one never does
                # (-inf - x = -inf or NaN; ordered > is false for both).
                update_cond = (current_max - total_max) > RESCALE_THRESHOLD
                total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                # Canonical 0-substitution at point of use, on BOTH alpha operands;
                # min(., 0) guards the dead->alive drop of the safe max (total_sum
                # is still 0 there).  total_max_safe starts at -inf: iter-0 alpha = 0.
                new_total_max_safe = row_max_for_exp2(total_max)
                alpha = cute.math.exp2(cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)), fastmath=True)
                total_max_safe = new_total_max_safe
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                    softmax_exchange.subview(exchange_base + tid_in_wg).store(alpha)
                    softmax_exchange.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).store(total_max_safe)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    _fused_before_sync()
                else:
                    alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()

                reg_S = reg_S * scale_log2 - total_max_safe

                chunk_S_0 = reg_S[0:CHUNK].vec
                chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
                hoisted_sum = row_reduction_pair(chunk_P_0)
                chunk_P_0_fp16 = chunk_P_0.to(STORAGE_DTYPE)
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    # P0 aliases S[64:96]. Correction releases after its S-hi
                    # load completes; pair its producer fence with this acquire.
                    bars.mb_stat_empty.wait(stat_empty_phase)
                    _fused_after_sync()
                    stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0_fp16)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    _fused_before_sync()
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_P_1 = None
                if cutlass.const_expr(N_CHUNKS == 2 and not CFG.HALF_FUSED_CORR_SPLIT_P):
                    chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
                    deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
                    chunk_P_1_fp16 = deferred_P_1.to(STORAGE_DTYPE)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32), chunk_P_1_fp16)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2 and not CFG.HALF_FUSED_CORR_SPLIT_P):
                    new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair

                if cutlass.const_expr(not CFG.HALF_FUSED_CORR_SPLIT_P):
                    bars.mb_stat_empty.wait(stat_empty_phase)
                    stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
        else:
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
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    _fused_after_sync()
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                tmem_base = tmem_ptr_i32.load()
                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + s_off_rt
                kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)
                raw_chunks = [
                    nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32), num=CHUNK) for c in range(N_CHUNKS)
                ]
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
                        mask_value=float("-inf"),
                    )
                    for c in range(N_CHUNKS)
                ]
                chunks_max = [row_max_reduction(chunks_S[c]) for c in range(N_CHUNKS)]
                reg_S_vec = vec_concat(chunks_S)
                current_max_unscaled = chunks_max[0]
                for m in chunks_max[1:]:
                    current_max_unscaled = cute.math.max(current_max_unscaled, m)
                reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
                current_max = current_max_unscaled * scale_log2  # -inf when the whole iteration is masked

                # total_max starts at -inf: a live iteration always clears the
                # threshold (real - (-inf) = +inf), a fully-masked one never does
                # (-inf - x = -inf or NaN; ordered > is false for both).
                update_cond = (current_max - total_max) > RESCALE_THRESHOLD
                total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                # Canonical 0-substitution at point of use, on BOTH alpha operands;
                # min(., 0) guards the dead->alive drop of the safe max (total_sum
                # is still 0 there).  total_max_safe starts at -inf: iter-0 alpha = 0.
                new_total_max_safe = row_max_for_exp2(total_max)
                alpha = cute.math.exp2(cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)), fastmath=True)
                total_max_safe = new_total_max_safe
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                    softmax_exchange.subview(exchange_base + tid_in_wg).store(alpha)
                    softmax_exchange.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).store(total_max_safe)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    _fused_before_sync()
                else:
                    alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()
                reg_S = reg_S * scale_log2 - total_max_safe

                chunk_S_0 = reg_S[0:CHUNK].vec
                chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
                hoisted_sum = row_reduction_pair(chunk_P_0)
                chunk_P_0_fp16 = chunk_P_0.to(STORAGE_DTYPE)
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    # P0 aliases S[64:96]. Correction releases after its S-hi
                    # load completes; pair its producer fence with this acquire.
                    bars.mb_stat_empty.wait(stat_empty_phase)
                    _fused_after_sync()
                    stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0_fp16)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    _fused_before_sync()
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_P_1 = None
                if cutlass.const_expr(N_CHUNKS == 2 and not CFG.HALF_FUSED_CORR_SPLIT_P):
                    chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
                    deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
                    chunk_P_1_fp16 = deferred_P_1.to(STORAGE_DTYPE)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32), chunk_P_1_fp16)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2 and not CFG.HALF_FUSED_CORR_SPLIT_P):
                    new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair
                if cutlass.const_expr(not CFG.HALF_FUSED_CORR_SPLIT_P):
                    bars.mb_stat_empty.wait(stat_empty_phase)
                    stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
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
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    _fused_after_sync()
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                tmem_base = tmem_ptr_i32.load()
                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + s_off_rt
                raw_chunks = [
                    nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32), num=CHUNK) for c in range(N_CHUNKS)
                ]
                chunks_max = [row_max_reduction(raw_chunks[c]) for c in range(N_CHUNKS)]
                current_max_unscaled = chunks_max[0]
                for _m in chunks_max[1:]:
                    current_max_unscaled = cute.math.max(current_max_unscaled, _m)
                reg_S = RegTile(vec_concat(raw_chunks), size=CFG.TILE_N)
                current_max = current_max_unscaled * scale_log2  # -inf when the whole iteration is masked

                # total_max starts at -inf: a live iteration always clears the
                # threshold (real - (-inf) = +inf), a fully-masked one never does
                # (-inf - x = -inf or NaN; ordered > is false for both).
                update_cond = (current_max - total_max) > RESCALE_THRESHOLD
                total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                # Canonical 0-substitution at point of use, on BOTH alpha operands;
                # min(., 0) guards the dead->alive drop of the safe max (total_sum
                # is still 0 there).  total_max_safe starts at -inf: iter-0 alpha = 0.
                new_total_max_safe = row_max_for_exp2(total_max)
                alpha = cute.math.exp2(cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)), fastmath=True)
                total_max_safe = new_total_max_safe
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                    softmax_exchange.subview(exchange_base + tid_in_wg).store(alpha)
                    softmax_exchange.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).store(total_max_safe)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    _fused_before_sync()
                else:
                    alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()
                reg_S = reg_S * scale_log2 - total_max_safe

                chunk_S_0 = reg_S[0:CHUNK].vec
                chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
                hoisted_sum = row_reduction_pair(chunk_P_0)
                chunk_P_0_fp16 = chunk_P_0.to(STORAGE_DTYPE)
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    # P0 aliases S[64:96]. Correction releases after its S-hi
                    # load completes; pair its producer fence with this acquire.
                    bars.mb_stat_empty.wait(stat_empty_phase)
                    _fused_after_sync()
                    stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0_fp16)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    _fused_before_sync()
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_P_1 = None
                if cutlass.const_expr(N_CHUNKS == 2 and not CFG.HALF_FUSED_CORR_SPLIT_P):
                    chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
                    deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
                    chunk_P_1_fp16 = deferred_P_1.to(STORAGE_DTYPE)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32), chunk_P_1_fp16)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2 and not CFG.HALF_FUSED_CORR_SPLIT_P):
                    new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair
                if cutlass.const_expr(not CFG.HALF_FUSED_CORR_SPLIT_P):
                    bars.mb_stat_empty.wait(stat_empty_phase)
                    stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
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
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    _fused_after_sync()
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                tmem_base = tmem_ptr_i32.load()
                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + s_off_rt
                kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)
                raw_chunks = [
                    nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32), num=CHUNK) for c in range(N_CHUNKS)
                ]
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
                        mask_value=float("-inf"),
                    )
                    for c in range(N_CHUNKS)
                ]
                chunks_max = [row_max_reduction(chunks_S[c]) for c in range(N_CHUNKS)]
                reg_S_vec = vec_concat(chunks_S)
                current_max_unscaled = chunks_max[0]
                for m in chunks_max[1:]:
                    current_max_unscaled = cute.math.max(current_max_unscaled, m)
                reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
                current_max = current_max_unscaled * scale_log2  # -inf when the whole iteration is masked

                # total_max starts at -inf: a live iteration always clears the
                # threshold (real - (-inf) = +inf), a fully-masked one never does
                # (-inf - x = -inf or NaN; ordered > is false for both).
                update_cond = (current_max - total_max) > RESCALE_THRESHOLD
                total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                # Canonical 0-substitution at point of use, on BOTH alpha operands;
                # min(., 0) guards the dead->alive drop of the safe max (total_sum
                # is still 0 there).  total_max_safe starts at -inf: iter-0 alpha = 0.
                new_total_max_safe = row_max_for_exp2(total_max)
                alpha = cute.math.exp2(cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)), fastmath=True)
                total_max_safe = new_total_max_safe
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    exchange_base = parity_rt * cutlass.Int32(2 * CFG.TILE_M)
                    softmax_exchange.subview(exchange_base + tid_in_wg).store(alpha)
                    softmax_exchange.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).store(total_max_safe)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    _fused_before_sync()
                else:
                    alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()
                reg_S = reg_S * scale_log2 - total_max_safe

                chunk_S_0 = reg_S[0:CHUNK].vec
                chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
                hoisted_sum = row_reduction_pair(chunk_P_0)
                chunk_P_0_fp16 = chunk_P_0.to(STORAGE_DTYPE)
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    # P0 aliases S[64:96]. Correction releases after its S-hi
                    # load completes; pair its producer fence with this acquire.
                    bars.mb_stat_empty.wait(stat_empty_phase)
                    _fused_after_sync()
                    stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0_fp16)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
                    _fused_before_sync()
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_P_1 = None
                if cutlass.const_expr(N_CHUNKS == 2 and not CFG.HALF_FUSED_CORR_SPLIT_P):
                    chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
                    deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
                    chunk_P_1_fp16 = deferred_P_1.to(STORAGE_DTYPE)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32), chunk_P_1_fp16)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2 and not CFG.HALF_FUSED_CORR_SPLIT_P):
                    new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair
                if cutlass.const_expr(not CFG.HALF_FUSED_CORR_SPLIT_P):
                    bars.mb_stat_empty.wait(stat_empty_phase)
                    stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)

        total_sum_scalar = total_sum[0] + total_sum[1]
        stats_addr_epi = tmem_ptr_i32.load() + cutlass.Int32(LAYOUT.STATS_EPI_OFF)
        stats_vec_epi = cutlass.Vector.from_elements((total_max_safe, total_sum_scalar), cutlass.Float32)
        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32), stats_vec_epi)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
            _fused_before_sync()
        bars.mb_stat_full.arrive()

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)


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
    qh_per_kh,
    o_partial_f32=None,
    # Fused epilogue gate staging tile (appended; the call is by keyword); the gate barriers ride `bars`.  None = gate off.
    sGate=None,
    thd_origins=None,
    softmax_exchange=None,
    scale_log2=None,
):
    nvvm.barrier_cta_sync(barrier_id=2, thread_count=32 * (CFG.CORRECTION_WARPS + 1))

    tid_raw = cute.arch.thread_idx()[0]
    tid_in_wg = tid_raw - cutlass.Int32(CFG.CORR_WARP_BASE * 32)

    bmm1_done_phase_pair = cutlass.Int32(0)
    bmm2_done_phase_pair = cutlass.Int32(0)
    stat_mbar_state = cutlass.Int32(0)
    epilogue_state = cutlass.Int32(1)
    gate_full_phase = cutlass.Int32(0)  # wait-then-arrive consumer of mb_gate_full (P2)

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        qh_per_kh,
        seqlen_kv,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)

    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)

    O_CHUNK = 16
    N_CHUNKS_O = CFG.TILE_O // O_CHUNK
    TMA_O_ITERS_LOCAL = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES
    D_BLOCK_SIZE = CFG.TILE_O // TMA_O_ITERS_LOCAL
    TMA_O_GRANU_ELEMS_LOCAL = CFG.TILE_M * D_BLOCK_SIZE

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
            high_sum = cutlass.Vector.from_elements((cutlass.Float32(0.0), cutlass.Float32(0.0)), cutlass.Float32)
            q_abs = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE) + tid_in_wg // cutlass.Int32(HEADS_PER_TILE)
            if bounds.right <= bounds.left:
                bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            for kv_loop in cutlass.range(bounds.left, bounds.right, 1, unroll=1):
                parity_cur_rt = kv_loop & cutlass.Int32(1)
                parity_prev_rt = (kv_loop - cutlass.Int32(1)) & cutlass.Int32(1)
                even = parity_cur_rt == cutlass.Int32(0)
                s_off = cutlass.Int32(
                    arith.select(even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off = cutlass.Int32(arith.select(even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value()))
                phase = (bmm1_done_phase_pair >> parity_cur_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_cur_rt].wait(phase)
                _fused_after_sync()
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_cur_rt)
                tmem_base_iter = tmem_ptr_i32.load()
                raw_hi = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(tmem_base_iter + s_off + cutlass.Int32(64), cutlass.Float32), num=64)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                if (kv_loop < bounds.unmasked_lo) | (kv_loop >= bounds.unmasked_hi):
                    raw_hi = apply_mask_chunk(
                        raw_hi,
                        q_abs,
                        kv_loop * cutlass.Int32(CFG.TILE_N) + cutlass.Int32(64),
                        eff_seqlen_kv,
                        CFG.WINDOW_LEFT,
                        CFG.MASK_FLAGS,
                        N=64,
                        bottom_right=CFG.BOTTOM_RIGHT,
                        causal_diag=None,
                        window_right=CFG.WINDOW_RIGHT,
                        mask_value=float("-inf"),
                    )
                bars.mb_stat_full.wait(stat_mbar_state)
                _fused_after_sync()
                exchange_base = parity_cur_rt * cutlass.Int32(2 * CFG.TILE_M)
                alpha = softmax_exchange.subview(exchange_base + tid_in_wg).load()
                max_safe = softmax_exchange.subview(exchange_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).load()
                # Release the aliased S tail only after tcgen05.wait::ld and reads.
                _fused_before_sync()
                bars.mb_stat_empty.arrive()
                stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)

                if kv_loop != bounds.left:
                    all_alpha_one = vote_sync(0xFFFFFFFF, alpha == cutlass.Float32(1.0), VoteSync.ALL)
                    previous_phase = (bmm2_done_phase_pair >> parity_prev_rt) & cutlass.Int32(1)
                    bars.mb_bmm2_done[parity_prev_rt].wait(previous_phase)
                    _fused_after_sync()
                    bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_prev_rt)
                    if ~all_alpha_one:
                        for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                            o_addr = tmem_base_iter + cutlass.Int32(LAYOUT.O_OFF + chunk_idx * O_CHUNK)
                            o_chunk = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), num=O_CHUNK)
                            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                            o_scaled = vec_scale_pair(o_chunk, alpha, O_CHUNK)
                            nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), o_scaled)
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                _fused_before_sync()
                bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                scaled = RegTile(raw_hi, size=64) * scale_log2 - max_safe
                high_p = cute.math.exp2(scaled.vec, fastmath=True)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(tmem_base_iter + p_off + cutlass.Int32(32), cutlass.Float32), high_p.to(STORAGE_DTYPE))
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                _fused_before_sync()
                bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(1)].arrive(
                    leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA
                )
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                high_sum = high_sum * alpha_pair + row_reduction_pair(high_p)

        else:
            if bounds.right > bounds.left:
                lo_parity_rt = bounds.left & cutlass.Int32(1)
                bars.mb_bmm2_ready[lo_parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                bars.mb_stat_full.wait(stat_mbar_state)
                bars.mb_stat_empty.arrive()
                stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)
            else:
                bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            for kv_loop in cutlass.range(bounds.left + cutlass.Int32(1), bounds.right, 1, unroll=1):
                parity_prev_rt = (kv_loop - cutlass.Int32(1)) & cutlass.Int32(1)
                parity_cur_rt = kv_loop & cutlass.Int32(1)
                tmem_base_iter = tmem_ptr_i32.load()

                bars.mb_stat_full.wait(stat_mbar_state)

                stats_off_cur = cutlass.Int32(
                    arith.select(
                        (parity_cur_rt == cutlass.Int32(0)).ir_value(),
                        cutlass.Int32(LAYOUT.STATS_EVEN_OFF).ir_value(),
                        cutlass.Int32(LAYOUT.STATS_ODD_OFF).ir_value(),
                    )
                )
                stats_addr = tmem_base_iter + stats_off_cur
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
                bars.mb_bmm2_done[parity_prev_rt].wait(bmm2_done_phase_prev)
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

        total_max_scaled = cutlass.Float32(0.0)
        total_sum = cutlass.Float32(0.0)
        # UNCONDITIONAL epilogue stat handshake (mirrors d128): the softmax group
        # publishes its epilogue stats every tile — including empty/inverted KV
        # ranges from compute_kv_loop_bounds (SWA window past the padded KV tail).
        # Guarding this behind right > left desyncs the mb_stat_full/mb_stat_empty
        # phase counts and the next tile's softmax wait spins forever (the
        # rectangular causal+SWA+padding hang). On an empty tile the published
        # stats are the reset values (max=-inf, sum=0) and the dead-row override
        # below emits O=0 / LSE=-inf.
        bars.mb_stat_full.wait(stat_mbar_state)
        if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
            _fused_after_sync()
        stats_addr_epi = tmem_base_epi + cutlass.Int32(LAYOUT.STATS_EPI_OFF)
        stats_vec_epi = nvvm.tcgen05_ld(
            "32x32b",
            nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32),
            num=2,
        )
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        total_max_scaled = stats_vec_epi[0]
        total_sum = stats_vec_epi[1]
        if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
            total_sum = total_sum + high_sum[0] + high_sum[1]
        if cutlass.const_expr(CFG.HALF_FUSED_CORR_SPLIT_P):
            _fused_before_sync()
        bars.mb_stat_empty.arrive()
        stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)

        LN2 = cutlass.Float32(0.6931471805599453)
        total_max_nat = total_max_scaled * LN2
        lse_val = cutlass.Float32(0.0)
        inv_sum = cutlass.Float32(0.0)
        q_row_global = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE) + (tid_in_wg // cutlass.Int32(HEADS_PER_TILE))
        row_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE) + (tid_in_wg % cutlass.Int32(HEADS_PER_TILE))
        # Under a KV split the fold moves to the combine pass (split_combine), which adds the sink column exactly once; the partials stay sink-free.
        if cutlass.const_expr(CFG.HAS_SINK and SPLIT_KV == 1):
            sinks_arr = cutlass.make_array_view(sinks_tensor)
            sink_logit = cutlass.Float32(sinks_arr[row_head_idx])
            # Keyless row (total_sum == 0; the softmax masks with -inf and publishes a
            # 0-substituted max): the sink is the row's whole mass, O := 0 / LSE := sink.
            # Select new_max := sink and scale := 0 for it instead of computing the fold
            # (exp(sink - 0) underflows to a zero denominator for sink <= -104 -> O = NaN,
            # LSE = -inf; exp(0 - sink) can overflow, 0 * inf is NaN) -- the arithmetic is
            # spelled out at the d128 kernel's sink fold.  Rows with keys are unchanged.
            kv_empty = total_sum <= cutlass.Float32(0.0)
            new_max = cutlass.Float32(arith.select(kv_empty.ir_value(), sink_logit.ir_value(), cute.math.max(total_max_nat, sink_logit).ir_value()))
            scale = cutlass.Float32(
                arith.select(kv_empty.ir_value(), cutlass.Float32(0.0).ir_value(), cute.math.exp(total_max_nat - new_max, fastmath=True).ir_value())
            )
            new_sum = total_sum * scale + cute.math.exp(sink_logit - new_max, fastmath=True)
            lse_val = new_max + cute.math.log(new_sum, fastmath=True)
            inv_sum = scale / new_sum
        else:
            lse_val = total_max_nat + cute.math.log(cute.math.max(total_sum, cutlass.Float32(1e-30)), fastmath=True)
            inv_sum = cutlass.Float32(1.0) / cute.math.max(total_sum, cutlass.Float32(1e-30))
            # Dead row (no valid KV column, incl. empty tiles where total_sum defaults 0):
            #   O := 0, LSE := -inf. total_sum >= 1 for any alive row so this never fires
            #   spuriously. Skipped under sink (the sink path defines a finite LSE).
            row_dead = total_sum <= cutlass.Float32(0.0)
            neg_inf_lse = cutlass.Float32(float("-inf"))
            lse_val = cutlass.Float32(arith.select(row_dead.ir_value(), neg_inf_lse.ir_value(), lse_val.ir_value()))
            inv_sum = cutlass.Float32(arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), inv_sum.ir_value()))

        if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT):
            # Dense padded-Q trim (cuDNN >= 9.14): q rows >= seq_len_q[b] write
            # O := 0 / LSE := -inf.  Applied AFTER the sink branch on purpose —
            # a trimmed row is dead even with a sink.  Per-batch q lens come in
            # via the dedicated seq_q_lens_addr parameter.
            _sq_arr = cute.make_tensor(cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24))
            _q_len_b = cutlass.Int32(_sq_arr[batch_idx])
            row_trim = q_row_global >= _q_len_b
            neg_inf_trim = cutlass.Float32(float("-inf"))
            lse_val = cutlass.Float32(arith.select(row_trim.ir_value(), neg_inf_trim.ir_value(), lse_val.ir_value()))
            inv_sum = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(0.0).ir_value(), inv_sum.ir_value()))
        # Base-2 Stats (stats_use_log2): natural LSE * log2(e); -inf stays -inf.
        if cutlass.const_expr(CFG.STATS_LOG2 and SPLIT_KV == 1):
            lse_val = lse_val * cutlass.Float32(1.4426950408889634)
        if cutlass.const_expr(lse_tensor is None):
            pass  # has_lse=False: the Stats store is compiled out
        elif cutlass.const_expr(CFG.THD_VARLEN):
            _cu = cutlass.make_array_view(seq_kv_lens_tensor)
            _cu_q_b = cutlass.Int32(_cu[n_batch + batch_idx])
            _s_q_b = cutlass.Int32(_cu[n_batch + batch_idx + cutlass.Int32(1)]) - _cu_q_b
            if q_row_global < _s_q_b:
                if cutlass.const_expr(thd_origins is not None):
                    _stats_base = thd_port_origin(thd_origins, 4, batch_idx, _cu_q_b)
                else:
                    _stats_base = _cu_q_b
                lse_arr = cutlass.make_array_view(lse_tensor)
                if cutlass.const_expr(SPLIT_KV > 1):
                    lse_arr[split_idx, row_head_idx, _stats_base + q_row_global] = lse_val
                elif cutlass.const_expr(len(lse_tensor.shape) == 2):
                    # token-major packed (T, H)
                    lse_arr[_stats_base + q_row_global, row_head_idx] = lse_val
                else:
                    # head-major packed (1, QH, head_stride)
                    if cutlass.const_expr(len(lse_tensor.shape) == 4):
                        # rank-4 = per-batch padded Stats (B, QH, s_max, 1) in the declared strides, no ragged offsets
                        lse_arr[batch_idx, row_head_idx, q_row_global, 0] = lse_val
                    else:
                        lse_arr[cutlass.Int32(0), row_head_idx, _stats_base + q_row_global] = lse_val
        else:
            if q_row_global < seqlen_q:
                lse_arr = cutlass.make_array_view(lse_tensor)
                # This chunk's LSE goes to its own split-major slot, matching where
                # TMA-STG put the chunk's O.  The pair (O_s, lse_s) is everything
                # the combine needs.
                lse_batch = _partial_batch(batch_idx, split_idx, n_batch)
                lse_arr[lse_batch, row_head_idx, q_row_global] = lse_val

        parity_last_rt = cutlass.Int32(0)
        if bounds.right > bounds.left:
            parity_last_rt = (bounds.right - cutlass.Int32(1)) & cutlass.Int32(1)
        bmm2_done_phase_last = (bmm2_done_phase_pair >> parity_last_rt) & cutlass.Int32(1)
        bars.mb_bmm2_done[parity_last_rt].wait(bmm2_done_phase_last)
        bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)

        O_EPI_BLK = 64 // CFG.BPE_O
        N_BLOCKS_EPI = CFG.TILE_O // O_EPI_BLK
        CHUNKS_PER_BLK = O_EPI_BLK // O_CHUNK

        sO_base = sO[0].base
        if cutlass.const_expr(_FP32_PARTIALS):
            # fp32 partials: accumulator straight to the workspace, no SMEM
            # staging and no TMA.  The staged path's barrier traffic still has
            # to happen: it consumes mb_o_empty once and publishes every
            # mb_o_full chunk, and the store warp waits on all of them.
            partial_batch = _partial_batch(batch_idx, split_idx, n_batch)
            partial_row = q_row_global
            partial_live = q_row_global < seqlen_q
            if cutlass.const_expr(CFG.THD_VARLEN):
                partial_cu = cutlass.make_array_view(seq_kv_lens_tensor)
                partial_q_base = cutlass.Int32(partial_cu[n_batch + batch_idx])
                partial_q_len = cutlass.Int32(partial_cu[n_batch + batch_idx + cutlass.Int32(1)]) - partial_q_base
                partial_batch = split_idx
                partial_row = partial_q_base + q_row_global
                partial_live = q_row_global < partial_q_len
            _store_fp32_partial_tile(
                o_partial_f32,
                tmem_base_epi,
                LAYOUT.O_OFF,
                inv_sum,
                row_dead,
                partial_live,
                partial_batch,
                partial_row,
                row_head_idx,
                CFG.TILE_O,
                O_CHUNK,
            )
            bars.mb_o_empty.wait(epilogue_state)
            for _chunk in cutlass.range_constexpr(N_O_CHUNKS):
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_o_full[_chunk].arrive()
        else:
            epi_o_full_block_idx = 0

            # Gate math setup (gate_epilogue_pairs): fold sigmoid's 1/2 into inv_sum so `o_chunk * _gate_inv_sum` IS h, hoist the
            # opaque 0.5 once per tile, then ONE wait per tile on the staged gate tile (issued by the TMA-LDG warp after this
            # tile's kv loop).  inv_sum is already 0 on a dead / trimmed / keyless row (the selects above); the per-element SELECT
            # below runs AFTER the gate fma (sdpa-invariants S2: the TMEM residue behind h can be NaN, and NaN * 0 = NaN).
            # Everything here folds away ungated: _gate_inv_sum IS inv_sum and the chunk loop is the pre-gate one byte for byte.
            _gate_inv_sum = gate_inv_sum(inv_sum) if cutlass.const_expr(EPILOGUE_GATE) else inv_sum
            _half_opaque = gate_half_opaque() if cutlass.const_expr(EPILOGUE_GATE) else cutlass.Float32(0.0)
            if cutlass.const_expr(EPILOGUE_GATE):
                _row_zero = inv_sum == cutlass.Float32(0.0)
                bars.mb_gate_full.wait(gate_full_phase)
                gate_full_phase = gate_full_phase ^ cutlass.Int32(1)
            sGate_base = sGate[0].base if cutlass.const_expr(EPILOGUE_GATE) else None

            for block_idx in cutlass.range_constexpr(N_BLOCKS_EPI):
                for sub in cutlass.range_constexpr(CHUNKS_PER_BLK):
                    chunk_idx_total = block_idx * CHUNKS_PER_BLK + sub
                    o_out = cutlass.Vector.from_elements(
                        tuple(OUT_STORAGE_DTYPE(0.0) for _ in range(O_CHUNK)),
                        OUT_STORAGE_DTYPE,
                    )
                    if cutlass.const_expr(not MAY_BE_EMPTY) or (bounds.right > bounds.left):
                        o_addr = tmem_base_epi + cutlass.Int32(LAYOUT.O_OFF + chunk_idx_total * O_CHUNK)
                        o_chunk = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                            num=O_CHUNK,
                        )
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                        o_scaled = o_chunk * _gate_inv_sum
                        if cutlass.const_expr(EPILOGUE_GATE):
                            # One swizzled LDS of this thread's O_CHUNK gate elements (GATE geometry _GG, the packed row tid_in_wg)
                            # + the packed h*tanh(g/2) + h; then the dead-row SELECT per element.  A plain `for` + append:
                            # range_constexpr is statement-only (frost-gotchas).
                            _vals = gate_epilogue_pairs(
                                o_scaled,
                                load_gate_chunk(sGate_base, gate_chunk_smem_offset(chunk_idx_total, O_CHUNK, _GG, tid_in_wg), _GG, O_CHUNK),
                                _half_opaque,
                                O_CHUNK,
                            )
                            _o_elems = []
                            for _i in cutlass.range_constexpr(O_CHUNK):
                                _o_elems.append(cutlass.Float32(arith.select(_row_zero.ir_value(), cutlass.Float32(0.0).ir_value(), _vals[_i].ir_value())))
                            o_out = cutlass.Vector.from_elements(tuple(_o_elems), cutlass.Float32).to(OUT_STORAGE_DTYPE)
                        else:
                            o_out = o_scaled.to(OUT_STORAGE_DTYPE)

                    col_offset_const = (chunk_idx_total * O_CHUNK) % D_BLOCK_SIZE
                    block_offset_const = ((chunk_idx_total * O_CHUNK) // D_BLOCK_SIZE) * TMA_O_GRANU_ELEMS_LOCAL
                    smem_offset = cutlass.Int32(block_offset_const + col_offset_const) + tid_in_wg * cutlass.Int32(D_BLOCK_SIZE)
                    smem_ptr = sO_base.subview(smem_offset).data_ptr()

                    if block_idx == 0 and sub == 0:
                        bars.mb_o_empty.wait(epilogue_state)
                    smem_ptr.store_swizzled(o_out, alignment=64, swizzle=_O_SMEM_SWIZZLE)

                fire_now = (block_idx % 2 == 1) or (CFG.TILE_O == O_EPI_BLK)
                if cutlass.const_expr(fire_now):
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_o_full[(block_idx // 2)].arrive()

            # Release the gate staging tile: a BARE arrive from every correction lane (128 == CORR_LANES), exactly like
            # mb_o_full above.  Generic LDS -> arrive -> producer wait -> TMA write needs no proxy fence.
            if cutlass.const_expr(EPILOGUE_GATE):
                bars.mb_gate_empty.arrive()

        epilogue_state = epilogue_state ^ cutlass.Int32(1)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)

    if cutlass.const_expr(CFG.CTA_MMA == 2):
        peer_cta = cta_id_x ^ cutlass.Int32(1)
        bars.mb_tmem_dealloc.arrive_on_peer(peer_cta)
    bars.mb_tmem_dealloc.arrive()


def _require_gate_presence_matches_cfg(gate_tensor) -> None:
    """Trace-time guard for _host (plain Python, runs during the trace): the gate tensor's presence must equal the module's
    CFG.EPILOGUE_GATE.  compile() builds the fake iff the flag is set, so only a direct cute.compile(_host, ...) caller can trip this."""
    if (gate_tensor is not None) != EPILOGUE_GATE:
        raise ValueError(f"{__name__}: gate_tensor presence ({gate_tensor is not None}) must match CFG.EPILOGUE_GATE={int(EPILOGUE_GATE)}")


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
    o_partial_ptr: Optional[cute.Pointer],
    block_table_ptr: Optional[cute.Pointer],
    block_table_v_ptr: Optional[cute.Pointer],
    table_strides: Tuple[cutlass.Int64, cutlass.Int64],
    n_pages: cutlass.Int32,
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    lse_kind: cutlass.Constexpr[str],
    paged_hnd: cutlass.Constexpr[bool],
    # Fused epilogue gate, [B, S, H_q, D_v] like O: a pointer + runtime (batch, seq, head) strides.  gate_ptr is None iff
    # CFG.EPILOGUE_GATE == 0 (a constexpr: sdpa_gate_tensor folds the slot out of the specialised signature); gate_strides is a
    # RUNTIME Int64 triple on EVERY rendering (compile() specialises it like the other stride slots, gated or not), so the no-gate
    # launch passes `gate_ptr=None, gate_strides=(0, 0, 0)` -- the sm107 d256 body's convention.  Appended, and REQUIRED like every
    # other slot of this ABI: a Python default cannot stand in for the pair (a None default reaches the compiled tuple slot as a
    # type mismatch; a non-None default is one the compiled-plan cache cannot carry, and the raw compiled callable then hides its
    # parameter names from a keyword caller).  The prepared binders fill both BY NAME, the THD-split wrapper passes the no-gate pair
    # (its `gate_slots`), direct launches pass it themselves.  Under THD the gate is the packed [1, T, H_q, D_v] form like Q / O.
    gate_ptr: Optional[cute.Pointer],
    gate_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    thd_origin_ptrs=None,
    thd_origin_specs: cutlass.Constexpr = None,
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Host entry: device pointers, runtime extents and strides in, TMA encodes and launches out.

    Operands are ``[B, S, H, D]`` with the head dim innermost (element stride 1);
    ``*_strides`` carry the (seq, head) element strides, K/V additionally the
    outer stride, which is the page stride of a paged pool and unused otherwise.
    Every stride leaf is Int64 (the ``compile()`` fakes fix the width): a 16-bit
    operand with S * H * D >= 2^27 elements would wrap the Int32 TMA-unit scaling.
    ``problem_size`` = (B, QH, KH, SQ, SKV, 0); under THD SQ/SKV are the packed
    token totals, under paged KV SKV is ``max_pages * PAGE_SIZE``. The batch
    stride of a dense operand is ``S * seq_stride``; a packed THD operand
    has batch extent 1 and binds the seq stride there (never stepped).

    ``lse_kind``: "dense" (B*SPLIT_KV, QH, SQ) in ``lse_strides``; "token" (SQ, QH)
    packed; "head" (1, QH, lse_ext) with lse_ext the head-row stride; "padded"
    (B, QH, lse_ext, 1) in ``lse_strides``. ``lse_ptr`` None compiles the store out.
    Unused slots (no THD, no paged KV, no split, no gate) are passed as None / zeros -- the gate's absence is
    ``gate_ptr`` None with ``gate_strides`` (0, 0, 0), passed by every gate-less caller."""
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
        o_partial_ptr,
        d_qk=d_qk,
        d_v=d_v,
        lse_kind=lse_kind,
        thd=CFG.THD_VARLEN,
        split_kv=SPLIT_KV,
        tensor_map_qwords=_TENSOR_MAP_QWORDS,
        paged=PAGED_KV,
        page_size=PAGE_SIZE,
        block_table_ptr=block_table_ptr,
        block_table_v_ptr=block_table_v_ptr,
        table_strides=table_strides,
        n_pages=n_pages,
    )
    # The fused epilogue gate G over the caller's (batch, seq, head) strides -- the packed [1, T, H_q, D_v] map under THD, like
    # Q / O -- or None (the ungated rendering: nothing of it is traced).
    gate_tensor = sdpa_gate_tensor(gate_ptr, problem_size, d_v, gate_strides, thd=bool(CFG.THD_VARLEN)) if cutlass.const_expr(gate_ptr is not None) else None
    _require_gate_presence_matches_cfg(gate_tensor)
    # Paged KV: HND storage viewed as [page, row, H_kv, D] has the row stride
    # BELOW the head stride, so its descriptor lists dims innermost-first as
    # (D, row, H_kv, page) and the TMA-LDG warp swaps its (head, row) coords;
    # NHD is the dense BSHD order with batch -> page.
    stride_order = (3, 2, 1, 0)
    if cutlass.const_expr(paged_hnd):
        kv_stride_order = (3, 1, 2, 0)
    else:
        kv_stride_order = stride_order
    _O_GRANU_ELEMS = CFG.O_SWZ_BYTES // CFG.BPE_O
    qk_box_q = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, TMA_QK_GRANU_ELEMS)
    qk_box_k = (1, K_BOX_ROWS, 1, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, V_BOX_ROWS, 1, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, _O_GRANU_ELEMS)

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    def _create_tma_desc(tensor: cute.Tensor, box_dims, swizzle, order=stride_order):
        # CUTLASS DSL 4.8's `create_tensor_map_tiled_from_view` scales dynamic strides in i32.
        # Explicitly widen the strides to 64-bit integers to avoid overflow.
        return tmap.create_tensor_map_tiled(
            global_address=tensor.iterator.toint(),
            dtype=tensor.element_type,
            global_dims=tuple(tensor.shape[i] for i in order),
            global_strides=tuple(cutlass.Int64(tensor.stride[i]) * tensor.element_type.width // 128 for i in order[1:]),
            box_dims=tuple(box_dims[i] for i in order),
            swizzle=swizzle,
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )

    tma_q_desc = _create_tma_desc(q_tensor, qk_box_q, _tma_swz(CFG.Q_SWZ_BYTES))
    tma_k_desc = _create_tma_desc(k_tensor, qk_box_k, _tma_swz(CFG.K_SWZ_BYTES), kv_stride_order)
    tma_v_desc = _create_tma_desc(v_tensor, vo_box_v, _tma_swz(CFG.V_SWZ_BYTES), kv_stride_order)
    _o_box = list(vo_box_o)
    if _FP32_PARTIALS:
        # fp32 is 4 bytes; scale the box by the O element's own width so
        # the inner dimension stays inside the swizzle's byte limit.
        _o_box[-1] = max(1, _o_box[-1] * CFG.BPE_O // 4)
    tma_o_desc = _create_tma_desc(o_tensor, tuple(_o_box), _tma_swz(CFG.O_SWZ_BYTES))
    # Gate LOAD descriptor in GATE geometry (_GG: 64-elem subtiles, s128b): the Q / O box's PACKED-ROW convention, so under PackGQA
    # the SMEM row order is the packed row order the correction threads hold.  None on the ungated rendering (dropped from _kernel's
    # specialised signature).
    gate_box = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, _GG.tma_granu_elems)
    tma_gate_desc = _create_tma_desc(gate_tensor, gate_box, _tma_swz(_GG.swz_bytes)) if cutlass.const_expr(gate_tensor is not None) else None

    # PackGQA: SQ*PACK_G packed rows per packed head, and QH/PACK_G packed heads.
    rows_per_cluster = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
    q_clusters = (SQ * HEADS_PER_TILE + rows_per_cluster - 1) // rows_per_cluster
    grid_q_supers = q_clusters * CFG.CTA_MMA
    q_supers = grid_q_supers
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD setup launch: build the [kv|cu_q|cu_k|remap|live|ctr] metadata
        # buffer DEVICE-side from the caller's length tensors (no host cumsum,
        # no H2D — issue #552), then the per-batch O descriptor array. Main
        # grid: the PERSISTENT cluster count — the adapter hands down
        # n_thd_units already capped to what the device holds resident,
        # min(plan-time envelope, SMs / CTA_MMA), NOT the envelope itself
        # (issue #618). It doubles as the claim counter's seed: cluster c runs
        # unit c off its blockIdx, then pulls from the counter, so the grid and
        # the seed must be the same number. Dispatching past the live total
        # stays safe either way — such a unit decodes the batch == n_batch
        # sentinel and drains without loads or stores.
        # ENVELOPE: the packed-O row stride is QH * ACTUAL d_v (o_tensor's
        # static inner extent), not QH * TILE_O — the per-batch descriptor
        # bases must step in real rows or every batch >= 1 lands OOB.
        if cutlass.const_expr(thd_origin_ptrs is not None):
            o_desc_words, origins, offsets = origin_tensors(o_desc_ptr, thd_origin_ptrs, cutlass.Int32(B))
            build_thd_origins(
                (q_tensor, k_tensor, v_tensor, o_tensor),
                tma_q_desc,
                tma_k_desc,
                tma_v_desc,
                tma_o_desc,
                o_desc_words,
                origins,
                offsets,
                thd_origin_specs,
                seq_kv_lens_tensor,
                thd_q_lens_tensor,
                thd_kv_lens_tensor,
                thd_lens_form,
                cutlass.Int32(B),
                cutlass.Int32(QH // HEADS_PER_TILE),
                cutlass.Int32(CGA_TILE_M // HEADS_PER_TILE),
                n_thd_units,
                paged=PAGED_KV,
                rank_by_kv=PAGED_KV and bool(CFG.BOTTOM_RIGHT) and (CFG.MASK_FLAGS & MASK_SWA) == 0,
            ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
        else:
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
                cutlass.Int32(QH // HEADS_PER_TILE),
                cutlass.Int32(B),
                cutlass.Int64(o_tensor.stride[1]),
                cutlass.Int32(CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA // HEADS_PER_TILE),
                n_thd_units,  # persistent cluster count; also seeds the claim counter
                not PAGED_KV,  # clamp_kv: paged pools have no packed KV total to clamp to
                SPLIT_KV,
                SPLIT_KV > 1 and bool(CFG.BOTTOM_RIGHT) and (CFG.MASK_FLAGS & MASK_SWA) == 0,
            ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
        grid_shape = (n_thd_units * cutlass.Int32(CFG.CGA_M), cutlass.Int32(1), cutlass.Int32(1))
    else:
        # KV split rides the BATCH axis: z = batch + split*B.  The decode
        # already recovers the batch coord on both the blockIdx and the
        # scheduler-handout paths, so the split travels with it for free.
        grid_shape = (
            (grid_q_supers, (QH // HEADS_PER_TILE), B * SPLIT_KV)
            if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL)
            else (grid_q_supers * (QH // HEADS_PER_TILE) * B * SPLIT_KV, 1, 1)
        )
    _kernel(
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        tma_o_desc,
        lse_tensor,
        sinks_tensor,
        seq_kv_lens_tensor,
        o_desc_words,
        cutlass.Int32(SQ),
        cutlass.Int32(SKV),
        cutlass.Int32(q_supers),
        cutlass.Int32(QH // HEADS_PER_TILE),
        cutlass.Int32(B),
        cutlass.Int32(QH // KH),
        scale_softmax_log2,
        seq_q_lens_addr,
        o_partial_f32,
        block_table_tensor,
        block_table_v_tensor,
        paged_hnd,
        tma_gate_desc,
        thd_origins=origins if cutlass.const_expr(thd_origin_ptrs is not None) else None,
    ).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(CFG.CTA_MMA, 1, 1),
        min_blocks_per_mp=1 if CFG.HALF_FUSED_CORR_SPLIT_P else 0,
        stream=stream,
    )


EXPLICIT_ABI = True  # pointer/int host entry; the adapter builds the argument list itself
LSE_KINDS = ("dense", "token", "head", "padded")


@lru_cache(maxsize=None)
def compile_thd_split(*, has_lse: bool = True, lse_kind: str = "head", paged_hnd: bool = False) -> Callable:
    """Compile only from plan facts; every token capacity and stride is dynamic."""
    if not (CFG.THD_VARLEN and SPLIT_KV > 1 and CFG.TILE_O == 256 and CFG.TILE_K == 256 and not CFG.PACK_GQA):
        raise ValueError("packed D256 split requires THD without PackGQA")
    cache_key = _template_key(globals(), locals(), "compile_thd_split")
    # gate_slots: this body's host ends in the two required epilogue-gate slots; the split wrapper passes the no-gate pair.
    return _compile_thd_split_host(
        _host, CFG, STORAGE_DTYPE, cache_key, has_lse=has_lse, lse_kind=lse_kind, paged_hnd=paged_hnd, ragged_q_slots=False, gate_slots=True
    )


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
    has_lse: bool = True,
    lse_kind: str = "dense",
    paged_hnd: bool = False,
    thd_origin_specs=None,
) -> Callable:
    """Compile the host entry for one layout kind.

    Every extent and stride is a runtime argument of the artifact (see ``_host``),
    so the key is only what specializes the traced code: the head-dim ENVELOPE
    (``d_qk`` / ``d_v``: the TMA descriptors carry the real extents while the
    tile box stays the compile-time TILE geometry — box columns past d_qk / d_v
    zero-fill on load, O columns past d_v clip on store), whether the LSE store
    exists, the Stats layout kind, and for paged pools whether the in-page
    layout is HND (row stride below head stride).

    Constraint: every non-innermost TMA global stride must be a 16-byte
    multiple; the compact BSHD H-stride is d * BPE, so d must be a multiple of
    8 at 2 bytes/elem — checked here for the envelope, by the adapter for
    declared strides."""
    if thd_origin_specs is not None:
        if not CFG.THD_VARLEN or SPLIT_KV != 1:
            raise ValueError("physical THD origins require the unsplit half prefill body")
        if len(thd_origin_specs) != 5:
            raise ValueError("physical THD origins require five port declarations")
    _cache_key = _template_key(globals(), locals(), "compile")
    if not (0 < d_qk <= CFG.TILE_K and 0 < d_v <= CFG.TILE_O):
        raise ValueError(f"d256 envelope: need 0 < d_qk <= {CFG.TILE_K} and 0 < d_v <= {CFG.TILE_O}; got ({d_qk}, {d_v})")
    if (d_qk * CFG.BPE) % 16 != 0 or (d_v * CFG.BPE_O) % 16 != 0:
        raise ValueError(f"d256 envelope: d_qk*BPE and d_v*BPE must be 16-byte multiples (TMA global-stride rule); got ({d_qk}, {d_v}) at BPE={CFG.BPE}")
    if SPLIT_KV > 1 and not has_lse:
        raise ValueError("split_kv > 1 requires has_lse=True (the per-split LSE drives the combine)")
    if lse_kind not in LSE_KINDS:
        raise ValueError(f"lse_kind must be one of {LSE_KINDS}; got {lse_kind!r}")
    if has_lse and (lse_kind == "dense") == bool(CFG.THD_VARLEN):
        raise ValueError("lse_kind 'dense' is the dense form; 'token' / 'head' / 'padded' are the THD forms")
    if paged_hnd and not PAGED_KV:
        raise ValueError("paged_hnd is a paged-KV specialization")
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
        P(cutlass.Float32 if _FP32_PARTIALS else OUT_STORAGE_DTYPE),
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
        P(cutlass.Float32) if _FP32_PARTIALS else None,
        P(cutlass.Int32, 4) if PAGED_KV else None,
        P(cutlass.Int32, 4) if PAGED_KV else None,
        (cutlass.Int64(0), cutlass.Int64(0)),
        i32,
        d_qk,
        d_v,
        lse_kind,
        paged_hnd,
        P(GATE_STORAGE_DTYPE) if EPILOGUE_GATE else None,  # gate_ptr: the gate is a MODULE specialization; its strides are runtime
        i64_3,
        thd_origin_ptrs=(
            tuple(P(cutlass.Int64 if spec[3] == 64 else cutlass.Int32, spec[3] // 8) if spec is not None else None for spec in thd_origin_specs)
            if thd_origin_specs is not None
            else None
        ),
        thd_origin_specs=thd_origin_specs,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd",
    )


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--iters", type=int, default=0)
    args = parser.parse_args()

    lse_kind = "token" if CFG.THD_VARLEN else "dense"
    print(f"[d256_f16_sm100] compile lse_kind={lse_kind}", flush=True)
    fn = compile(lse_kind=lse_kind)
    print(f"[d256_f16_sm100] compile OK: {fn}", flush=True)
    if args.validate:
        print("[d256_f16_sm100] compiled — run validation via the frost SDPA test suite (see test/python/frost/sdpa/).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
