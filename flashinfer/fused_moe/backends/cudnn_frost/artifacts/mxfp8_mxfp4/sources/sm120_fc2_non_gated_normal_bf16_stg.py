# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""sm120 (GeForce/consumer Blackwell, CC 12.x) MoE grouped block-scale matmul
fwd (nvfp4 / mxfp4 / mxfp8): grouped persistent scheduler + warp-level
block-scaled MMA.

Per routed group ``g``: ``out[fto[g]:fto[g+1]] = dequant(token[range]) @
dequant(weight[g % E]).T`` with one scale factor per ``block_size`` K elements
applied inside the tensor core (``mma.sync.aligned ... .kind::mx*.block_scale``).

Derived from ``sm120_block_scale_matmul.py`` the way
``sm120_moe_grouped_matmul_fwd.py`` is derived from ``sm120_matmul.py``: the
mainloop (TMA -> swizzled packed SMEM + SF words -> ldmatrix -> block-scaled
``mma.sync``) and the transposed-STG epilogue are the dense kernel's; the tile
source is the grouped persistent scheduler (global atomic tile counter + warp
prefix scan over ``first_token_offset``) of the sm100 MoE kernels, without the
cluster broadcast layer (CC 12.x has no clusters).

Scale-factor layouts (the ``CompiledMoeBlockScaleGemm`` contract):

* ``sfb`` is F8_128x4-reordered PER EXPERT: one ``[256 fp16][k4][n blocks]``
  blob per expert, addressed by the routed expert as B's batch coordinate.
* ``sfa`` is F8_128x4-reordered and padded to whole 128-row blocks PER GROUP,
  then concatenated in group order: group ``g``'s scales start at SF block
  ``sum(ceil(rows_g' / 128) for g' < g)``. That prefix is what the scheduler
  carries alongside the tile record (``start_sf_block_m``), so it walks the
  groups in INDEX order (no expert-major reordering here). A tile's SF rows are
  GROUP-LOCAL: block ``start_sf_block_m + (tile_m * cta_m) // 128``, in-block
  offset ``(tile_m * cta_m) % 128`` -- never the global token row.

KEEP IN SYNC WITH ``sm120_block_scale_matmul.py`` / ``sm120_moe_grouped_matmul_fwd.py``
(this tree) and ``../../sm100/kernel_templates/sm100_moe_grouped_block_scale_matmul_fwd.py``.

What is deliberately NOT here (vs. the sm100 MoE block-scale kernel)
-------------------------------------------------------------------
* No per-group TMA descriptor replacement, for A or for SFA. Both are addressed
  by COORDINATE on ONE global descriptor each: A on the flat ``[K, S]`` token
  matrix (row ``group_begin + tile_m * cta_m``), SFA on the whole segmented
  blob whose block count is the graph-time capacity the compiler already
  requires of the buffer (``min(S, G) + (S - min(S, G)) // 128`` blocks), so a
  ragged-tail box past a group's blocks reads the next group's blocks or is
  zero-filled at the end -- into accumulator rows the epilogue never stores
  (``row < group_end``). No ``tensormap.replace``, no proxy fences, no per-CTA
  descriptor scratch: the workspace is the scheduler counter alone
  (``moe_desc_slots = 0``).
* No TMA-store epilogue.

Per-side operand forms (as the sm100 kernel's per-side SMEM / TMA facts)
------------------------------------------------------------------------
* Mixed MXFP4 x MXFP8 (e2m1 x e4m3 / e5m2, either order; E8M0 / 32): the warp
  MMA is ``kind::mxf8f6f4`` m16n8k32 with per-side types, reading e2m1 from
  bits [5:2] of an 8-bit container. The e2m1 operand stays packed: TMA
  ``.b4x16`` (any K a multiple of 32, 16-byte aligned rows) into a 64-byte
  ``s64b`` row per 128-value K-tile -- the parent's native packed form, not its
  ``.b4x16_p64`` one, whose TMA needs K a multiple of 128 and a 32-byte grid
  (gpt-oss's K = 2880 is not) -- and each lane builds its fragment registers
  from two packed words (``_fp4_frag_regs``: one PRMT picks the lane's two
  16-bit halves, two shift-and-mask steps and two PRMTs spread the nibbles to
  bits [5:2] of the fp8 fragment layout's bytes). The fp8 side keeps its
  128-byte rows and ldmatrix loads.
* One-sided dequant (a raw FP8 operand against a block-scaled one, normalized
  to a fake dequant by the analyzer): the raw side has no scale-factor tensor.
  sm100 seeds its TMEM scale region with the identity word; here the scale is
  a register operand of ``mma.sync``, so its MMAs read the constant
  ``sf_one_word`` and the side keeps no SF descriptor, prefetch, TMA box, SMEM
  ring or GATHER repack (``num_sfa_operands`` / ``num_sfb_operands`` = 0).

FP4 scale corners (the sm100 kernel's idesc scale_format / scale_vec fields)
---------------------------------------------------------------------------
* fp4 + E8M0 scales at block 16 is ``kind::mxf4nvf4`` scale_vec::4X ``.ue8m0``
  (PTX ISA 9.1), which the DSL's ``nvvm.mma.block_scale`` verifier rejects:
  that one form goes out as inline PTX with the same operands
  (``_MMA_INLINE_PTX``).
* fp4 + E4M3 scales at block 32 has no warp-MMA form (``.ue4m3`` is 4X only,
  one scale per 16 K): it rides the nvfp4 4X ``.ue4m3`` instruction, each
  k-block reading a word rebuilt from bytes [b, b, b+1, b+1] of its loaded SF
  word (``_SF_DUP == 2``, one PRMT per word and k-block) -- every block-32
  scale in both of its block-16 slots, which is the block-32 product exactly.
  The SF boxes, TMA and GATHER repack are mxfp4's.

Multi-GEMM (e.g. SwiGLU: ``silu(A @ B0) * (A @ B1)``)
-------------------------------------------------------
As on ``sm120_moe_grouped_matmul_fwd.py``: one SMEM tile per DISTINCT operand
per stage -- here its packed K-tile plus its SF box, an operand's scales
travelling with its data -- and one register accumulator per GEMM
(``gemm_a_idx`` / ``gemm_b_idx`` pick each GEMM's operands, and with them the
SFA / SFB words its block-scaled MMA reads). Every GEMM's accumulators stay
resident for the whole tile (``num_gemms * _ACC_REGS`` fp32 per lane); past the
compute warp's register grant ptxas spills them to local memory -- a perf
trade-off, never a reject. The one hard gate is SMEM: the renderer sizes the
ring with the epilogue staging taken off in bytes
(``Sm120KernelTemplate.multi_gemm_reject`` / ``.multi_gemm_ab_stages`` in
kernel_registry). Both drains stage the GEMMs' fragments through the same
warp-private buffer one after another and hand the fused epilogue one fp32
vector per GEMM (``vec_f32``, ``vec_f32_1``, ...).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cutlass.experimental.primitives as nvvm
# Inlined from cudnn.gemm.frost.tile_helpers
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Tile-level helpers shared by the SM100 and SM120 kernel templates.

A template is RENDERED (its `@@INJECT_*@@` blocks become module-level
constants) and then exec'd from the kernel cache under a synthetic module
name, so it cannot use relative imports and this module is never rendered.
Everything here therefore takes what it needs as ARGUMENTS -- a helper that
reads an injected constant (`mma_size_m`, `tile_swizzle_n`, `ab_dtype`, ...)
has to stay in the template, or receive it explicitly.

Scheduling and gather helpers serve both families. The tcgen05 wrappers are
used only by the SM100 family and retain its instruction-specific contracts.
"""


import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass._mlir.dialects import llvm






TENSOR_MAP_QWORDS = 16


def moe_swizzle_tile(t, nt_m, nt_n, swizzle_w):
    """Group-local linear tile index -> (m, n) under an N-super-block walk.
    ``swizzle_w == nt_n`` reproduces the plain n-fast split; ``1`` gives m-fast.
    """
    blk = cutlass.max(nt_m * swizzle_w, cutlass.Int32(1))
    sb = t // blk
    off = t - sb * blk
    base_n = sb * swizzle_w
    cur_S = cutlass.min(cutlass.Int32(swizzle_w), nt_n - base_n)
    tile_m = off // cur_S
    tile_n = base_n + off - tile_m * cur_S
    return tile_m, tile_n


















@cute.jit
def moe_gather_row(token_index, row, group_end, source_rows):
    # Never read beyond the routed group. An out-of-range source row asks TMA
    # to zero-fill the padding while still completing the expected byte count.
    src = cutlass.Int32(source_rows)
    if row < group_end:
        src = cutlass.Int32(token_index[row])
    return src










@cute.jit
def tma_gather4(dst, desc, k, r0, r1, r2, r3, mbar, mask=None, cta_group: cutlass.Constexpr = 1):
    """Gather four rows from a rank-2 tensor map with box_dims[1] == 1.

    No mask selects shared::cta (SM120); a mask selects SM100 cluster multicast.
    The experimental DSL TMA wrapper currently validates two coordinates for
    this five-coordinate instruction, so issue the PTX directly.
    """
    bar_addr = mbar.data_ptr().toint(dtype=cutlass.Int32)
    if cutlass.const_expr(cta_group == 2):
        bar_addr = bar_addr & cutlass.Int32(0xFEFFFFFF)
    args = [
        dst.data_ptr().toint(dtype=cutlass.Int32).ir_value(),
        desc.toint().ir_value(),
        cutlass.Int32(k).ir_value(),
        cutlass.Int32(r0).ir_value(),
        cutlass.Int32(r1).ir_value(),
        cutlass.Int32(r2).ir_value(),
        cutlass.Int32(r3).ir_value(),
        bar_addr.ir_value(),
    ]
    if cutlass.const_expr(mask is None):
        llvm.inline_asm(
            None,
            args,
            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4." "mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, $4, $5, $6}], [$7];",
            "r,l,r,r,r,r,r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        llvm.inline_asm(
            None,
            args + [cutlass.Int16(mask).ir_value()],
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4."
            "mbarrier::complete_tx::bytes.multicast::cluster."
            f"cta_group::{cta_group} [$0], [$1, {{$2, $3, $4, $5, $6}}], [$7], $8;",
            "r,l,r,r,r,r,r,r,h,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )











_moe_swizzle_tile = moe_swizzle_tile
import cutlass.experimental.cuda.tensor_map as _tma
from cutlass import apply_swizzle as _apply_smem_swizzle
import cutlass
# Inlined from cudnn.gemm.frost.kernel_templates.dynamic_scheduler_counter_initialization
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cutlass
import cutlass.cute as cute


@cute.kernel
def dynamic_scheduler_counter_initialization(workspace: cute.Tensor, counter_qword: cutlass.Int32):
    counter = cute.make_tensor(
        cute.recast_ptr(workspace.iterator + counter_qword, dtype=cutlass.Int32),
        cute.make_layout(1),
    )
    counter[0] = cutlass.Int32(0)


dynamic_scheduler_counter_initialization.set_name_prefix("cudnn", remove_cutlass_symbol=True)

_dynamic_scheduler_counter_initialization = dynamic_scheduler_counter_initialization
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_tensor
from cutlass.cute.runtime import make_fake_stream
from cutlass.cutlass_dsl import T as _T
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.dialects.nvvm import BlockScaleFormat as _BlockScaleFormat
from cutlass._mlir.dialects.nvvm import MMABlockScaleKind as _MMABlockScaleKind
from cutlass._mlir.dialects.nvvm import MMATypes as _MMATypes
from cutlass._mlir.dialects.nvvm import ScaleVecSize as _ScaleVecSize
from cuda.bindings import driver as _cuda

# @@FROST_GEOMETRY@@

@cute.jit
def red_part_seed(_red_part):
    pass


@cute.jit
def red_part_flush_tile(_red_part, lane, M, N, _red_part_ctx, _red_part_args):
    pass


@cute.jit
def red_part_flush_cta(_red_part, lane, M, N, _red_part_ctx, _red_part_args):
    pass

if a_is_m_major:
    raise NotImplementedError(f"{__name__}: the MoE token is K-major only (the grouped A walk is a K-major TMA box)")
# GATHER repacks the tile's routed token scales into the F8_128x4 SFA stage in
# warp-wide passes of 32 SF words (the TMA warp's repack, as in
# tile_helpers.moe_gather_scales): the token tile must hold a whole number of
# them (Sm120KernelTemplate.gather_sf_reject declines it upstream; this is the
# backstop). A raw-FP8 token has no scales to repack.
if moe_gather and num_sfa_operands and (cta_tile_mnk[0] * sf_tma_box_k) % 32:
    raise NotImplementedError(
        f"{__name__}: GATHER needs cta_tile_m * sf_k4 to be a multiple of 32 SF words "
        f"(got cta_tile_m={cta_tile_mnk[0]}, sf_k4={sf_tma_box_k}); pick a taller M tile"
    )

# A TMA tensormap is 128 bytes = 16 int64 qwords. The workspace is laid out as
# grid_ctas * moe_desc_slots tensormap slots followed by the scheduler counter;
# this kernel patches no descriptor, so its slot count is zero and the counter
# sits at the start of the buffer (the compiler carves it the same way; the host
# below zeroes it).
TENSOR_MAP_QWORDS = 16
moe_desc_slots = 0

# Scheduler ring depth and the i32 words per record.
SCHED_STAGES = 2
SCHED_SLOT_WORDS = 8

# Programmatic Dependent Launch (PDL, sm_90+; supported on sm_120).
USE_PDL = True

# Named barrier id for cross-warp sync of the compute warps (unused by the STG
# epilogue, kept for parity with the dense kernel's constants).
EPI_SYNC_BAR_ID = 1

# Compute-warp grid over the CTA tile (warp_row x warp_col), derived from the
# injected geometry: one warp tile = mma_size x the 16x16 warp-MMA pair, so
# the grid is cta_tile / warp_tile per axis.
WARPS_M = cta_tile_mnk[0] // (mma_size_m * mma_inst_shape_mnk[0])
WARPS_N = cta_tile_mnk[1] // (mma_size_n * mma_inst_shape_mnk[1])
NUM_COMPUTE_WARPS = WARPS_M * WARPS_N
# GATHER: the token tile's 4-row GATHER4 groups per TMA-warp lane.
_GATHER_LANE_GROUPS = (cta_tile_mnk[0] // 4 + 31) // 32

TMA_WARP_ID = NUM_COMPUTE_WARPS
SCHEDULER_WARP_ID = NUM_COMPUTE_WARPS + 1
NUM_WARPS = threads_per_cta // 32

# Scheduler-ring consumers: every compute warp + the TMA producer each arrive
# once (elected) per consumed record.
NUM_SCHED_CONSUMER_WARPS = NUM_COMPUTE_WARPS + 1

EPI_REG_COUNT = 232
# One budget for the whole producer warpgroup (TMA, scheduler and donor warps).
# PTX setmaxnreg is .sync.aligned over the warpgroup: it is issued once, before
# the per-warp roles diverge, so the count cannot differ per warp.
PROD_REG_COUNT = 24
assert NUM_COMPUTE_WARPS % 4 == 0, "the producer warps must form whole warpgroups"

# ---------------------------------------------------------------------------
# Geometry derived from the injected tile constants (all plain Python ints --
# resolved at render/import time, traced as constants). Verbatim
# sm120_block_scale_matmul.
#
# K is kept in TWO units: NATIVE elements (fp4 / fp8; what the TMA descriptor
# and the problem's K count in) and PACKED bytes (what SMEM and the fragment
# loads count in). Each side's SMEM dtype is one byte wide (fp8, or packed
# Float4E2M1FNx2 pairs) and each side has its own K row
# (a_packed_per_row / b_packed_per_row bytes): 128 B, except the e2m1
# operand of a mixed MXFP4 x MXFP8 pair -- its K-tile runs at the fp8 rate
# (128 values), packed into a 64-byte row.
# ---------------------------------------------------------------------------

_CTA_K_ELEMS = cta_tile_mnk[2]  # native K elements per tile
# One k-block = the K extent of one warp MMA: m16n8k64 for packed fp4 x fp4,
# m16n8k32 for fp8 and the mixed pairs.
_K_BLK_ELEMS = mma_inst_shape_mnk[2]
_NUM_K_BLOCKS = _CTA_K_ELEMS // _K_BLK_ELEMS
assert _NUM_K_BLOCKS * _K_BLK_ELEMS == _CTA_K_ELEMS, (_CTA_K_ELEMS, _K_BLK_ELEMS)
assert a_smem_dtype.width == 8 and b_smem_dtype.width == 8, f"the packed carriers must be one byte wide, got {a_smem_dtype} / {b_smem_dtype}"
_A_ROW_BYTES = a_packed_per_row  # packed bytes per K row of A in SMEM
_B_ROW_BYTES = b_packed_per_row
# Packed bytes per k-block: 32, or 16 for the e2m1 side of a mixed pair.
_A_KBLK_BYTES = _A_ROW_BYTES // _NUM_K_BLOCKS
_B_KBLK_BYTES = _B_ROW_BYTES // _NUM_K_BLOCKS
_ELEMS_16B = 16  # packed bytes per 16-byte ldmatrix row
# The e2m1 operand of a mixed pair: kind::mxf8f6f4 reads e2m1 from bits [5:2]
# of an 8-bit container (the fp8 fragment layout), but SMEM holds it packed
# (TMA .b4x16, two values a byte, any K a multiple of 32): its fragment
# registers are built from the packed words (_fp4_frag_regs), not by ldmatrix.
_A_FP4_TO_BYTES = mma_block_scale_kind == "MXF8F6F4" and mma_a_ptx_type == "e2m1"
_B_FP4_TO_BYTES = mma_block_scale_kind == "MXF8F6F4" and mma_b_ptx_type == "e2m1"
assert (_A_KBLK_BYTES, _B_KBLK_BYTES) == (16 if _A_FP4_TO_BYTES else 32, 16 if _B_FP4_TO_BYTES else 32), (_A_ROW_BYTES, _B_ROW_BYTES, _NUM_K_BLOCKS)

_WARP_TILE_M = cta_tile_mnk[0] // WARPS_M
_WARP_TILE_N = cta_tile_mnk[1] // WARPS_N
_M_FRAGS = _WARP_TILE_M // 16
_N_FRAGS = _WARP_TILE_N // 8
_N_FRAG_PAIRS = _N_FRAGS // 2
_ACC_REGS = _M_FRAGS * _N_FRAGS * 4

# SMEM K-row swizzle per side: the K-row width IS the swizzle span. The TMA
# s128b pattern == cutlass.Swizzle(3, 4, 3) on a 128-byte row, s64b ==
# cutlass.Swizzle(2, 4, 3) on a mixed pair's 64-byte e2m1 row; the fragment
# loads below apply the same XOR.
_A_SWIZZLE = cutlass.Swizzle((_A_ROW_BYTES // 16).bit_length() - 1, 4, 3)
_B_SWIZZLE = cutlass.Swizzle((_B_ROW_BYTES // 16).bit_length() - 1, 4, 3)
_TMA_SWIZZLE_BY_ROW = {128: _tma.TensorMapSwizzle.s128b, 64: _tma.TensorMapSwizzle.s64b}
assert (
    _TMA_SWIZZLE_BY_ROW.get(_A_ROW_BYTES) == a_tma_swizzle and _TMA_SWIZZLE_BY_ROW.get(_B_ROW_BYTES) == b_tma_swizzle
), "block-scale K rows are one swizzle span: 128 B, 64 B for a mixed pair's e2m1 side (validate_block_scale_config_sm120)"
# The packed e2m1 side folds its rows' s64b XOR into lane constants
# (fp4_lane_words): 64-byte rows, and every stage a whole number of 512-byte
# swizzle periods (the rings themselves are 1024-aligned).
assert not _A_FP4_TO_BYTES or (_A_ROW_BYTES == 64 and sA_packed_elems % 512 == 0), (_A_ROW_BYTES, sA_packed_elems)
assert not _B_FP4_TO_BYTES or (_B_ROW_BYTES == 64 and sB_packed_elems % 512 == 0), (_B_ROW_BYTES, sB_packed_elems)

# ---- Transposed STG epilogue staging (fort "Sheet3" scheme) -----------------
_STG_EPI_LANE_QUAD = 4  # one STS.128 = 4 x 32-bit acc regs per lane
_STG_EPI_PAD = 4  # 16B skew after each 128-element batch (sheet's X cells)
_STG_EPI_BATCH_STRIDE = 32 * _STG_EPI_LANE_QUAD + _STG_EPI_PAD  # 132
_STG_EPI_GROUP_FRAGS = 4  # fragments (= STS batches) per 32-column group
_STG_EPI_WARP_ELEMS = _STG_EPI_GROUP_FRAGS * _STG_EPI_BATCH_STRIDE  # 528
_STG_EPI_NGRP = (_N_FRAGS + _STG_EPI_GROUP_FRAGS - 1) // _STG_EPI_GROUP_FRAGS
_STG_V = (vec_bytes_epi * 8) // cd_dtype.width

# Row-per-lane drain (block-scale quantize chains; the row_per_lane_drain tile
# constant picks it over the lane-run drain). A 32-row slab is two 16-row m-fragments staged
# in two sub-passes; a warp tile of one 16-row fragment gives a 16-row slab
# owned by lanes 0..15.
_RPL_SUBPASSES = 2 if _M_FRAGS % 2 == 0 else 1
_RPL_SLABS = _M_FRAGS // _RPL_SUBPASSES


def _rpl_swz(r):
    """XOR key of the 16-byte granules of row-major staging row ``r``: keeps
    the drain's STS.64 writes and LDS.128 reads of a 16x32 fp32 tile at the
    two-words-per-bank minimum (an unswizzled tile hits eight)."""
    rr = r % 8
    return ((rr & 3) << 1) | (rr >> 2)


# One AB stage = the packed tile + SF box of every distinct A and B operand.
# The injected ab_stages already has the STG staging taken off the budget in
# bytes, so it is used as is.
assert ab_stages >= 1, f"{__name__}: the renderer emitted an empty AB ring"

# ---------------------------------------------------------------------------
# Scale-factor geometry. The F8_128x4 blob is 512-byte atoms: 128 rows x 4
# K-scales, row r at byte (r%32)*16 + (r//32)*4, K-scale kk at byte +kk -- so
# the 32-bit word of a row holds its 4 consecutive K-scales. A K-tile has
# sf_k4 such words per row; the TMA box lands them as [mn_block][k_word][512 B],
# which is [mn_block][k_word][128 words] in the Int32 SMEM arrays below.
# ---------------------------------------------------------------------------

_SF_WORDS = sf_tma_box_k  # 32-bit SF words per row per K-tile (= sf_k4)
_SF_SPI = sf_scales_per_inst  # scales one MMA consumes along K: 1 (1X), 2 (2X), 4 (4X)
_SF_ATOM_WORDS = 128  # 512 B
_SF_BLOCK_WORDS = _SF_WORDS * _SF_ATOM_WORDS  # one 128-row block of the stage
_SFA_STAGE_WORDS = sfa_smem_bytes // 4
_SFB_STAGE_WORDS = sfb_smem_bytes // 4
assert _SF_WORDS * 4 == _NUM_K_BLOCKS * _SF_SPI, (_SF_WORDS, _NUM_K_BLOCKS, _SF_SPI)
# k-block j reads SF word (j*spi)//4 at byte_id (j*spi)%4 (a 2X MMA also reads
# the byte after it, a 4X MMA the whole word).
_SF_WORD_BY_KBLK = tuple((j * _SF_SPI) // 4 for j in range(_NUM_K_BLOCKS))
_SF_BYTE_BY_KBLK = tuple((j * _SF_SPI) % 4 for j in range(_NUM_K_BLOCKS))
# Instruction scale-vector slots each consumed scale fills: 1, except fp4 with
# e4m3 scales at block 32. That combo has no warp-MMA form of its own (.ue4m3
# is 4X only: one slot per 16 K) and rides the 4X .ue4m3 instruction with its
# 2 block-32 scales per MMA, each filling both of its block-16 slots -- the
# k-block's word is bytes [b, b, b+1, b+1] of the loaded one (one PRMT), read
# whole (byte_id 0). A block-32 scale applied to both 16-element halves of its
# block is the block-32 product exactly.
_SF_DUP = {"X1": 1, "X2": 2, "X4": 4}[mma_scale_vec_size] // _SF_SPI
assert _SF_DUP in (1, 2) and _SF_DUP * _SF_SPI == {"X1": 1, "X2": 2, "X4": 4}[mma_scale_vec_size], (mma_scale_vec_size, _SF_SPI)
# The byte_id immediate k-block j's MMA passes with its word, and (duplicated
# scales) the PRMT selector that rebuilds the word: bytes [b, b, b+1, b+1] of
# the loaded one, b = _SF_BYTE_BY_KBLK[j] -- scale b in slots 0-1, b+1 in 2-3.
_SF_MMA_BYTE_BY_KBLK = _SF_BYTE_BY_KBLK if _SF_DUP == 1 else (0,) * _NUM_K_BLOCKS
_SF_DUP_PRMT_BY_KBLK = tuple(b | (b << 4) | ((b + 1) << 8) | ((b + 1) << 12) for b in _SF_BYTE_BY_KBLK)
# One SFA word per PAIR of m-frags (thread_id_a picks the frag), one SFB word
# per QUAD of n-frags (thread_id_b picks the frag).
_SFA_GROUPS = -(-_M_FRAGS // 2)
_SFB_GROUPS = -(-_N_FRAGS // 4)
# The tile's first row / column sits at a 128-block boundary iff the CTA tile
# is a multiple of 128; otherwise (tile is a divisor of 128) the box holds the
# whole block and the tile's offset inside it is a runtime value.
_M_BLOCK_ALIGNED = cta_tile_mnk[0] % 128 == 0
_N_BLOCK_ALIGNED = cta_tile_mnk[1] % 128 == 0

# ---------------------------------------------------------------------------
# The block-scaled warp MMA, resolved from the injected block-scale constants
# (see sm120_block_scale_matmul.py for the instruction forms and the
# {byte_id, thread_id} scale-operand contract).
# ---------------------------------------------------------------------------

_MMA_KIND = getattr(_MMABlockScaleKind, mma_block_scale_kind)
_MMA_SCALE_VEC = getattr(_ScaleVecSize, mma_scale_vec_size)
_MMA_SF_FORMAT = getattr(_BlockScaleFormat, mma_sf_format)
_MMA_A_TYPE = getattr(_MMATypes, mma_a_ptx_type)
_MMA_B_TYPE = getattr(_MMATypes, mma_b_ptx_type)
_MMA_SHAPE_ATTR = f"#nvvm.shape<m = 16, n = 8, k = {_K_BLK_ELEMS}>"
assert mma_c_dtype == cutlass.Float32, "the block-scaled warp MMA accumulates in fp32"
# kind::mxf4nvf4 scale_vec::4X with a .ue8m0 scale (fp4 + e8m0 at block 16; PTX
# ISA 9.1, SASS OMMA.SF.16864.F32.E2M1.E2M1.E8.4X) is a legal sm_120a instruction
# the DSL's nvvm.mma.block_scale verifier rejects ("unsupported ScaleVecSize and
# BlockScaleFormat attributes"). That one form goes out as inline PTX with the
# NVVM op's operands (_mma_16x8_bs) and every other form keeps the op. The
# module's PTX ISA version is 9.3 under the DSL's cu13 toolchain (CUDA 13.x),
# above the 9.1 the form needs; its cu12 toolchain (CUDA 12.9, PTX ISA 8.8)
# cannot assemble it, and Sm120KernelTemplate.block_scale_toolchain_reject
# declines the graph upstream there.
_MMA_INLINE_PTX = mma_scale_vec_size == "X4" and mma_sf_format == "UE8M0"
_MMA_PTX = (
    f"mma.sync.aligned.m16n8k{_K_BLK_ELEMS}.row.col.kind::{mma_block_scale_kind.lower()}.block_scale"
    f".scale_vec::{mma_scale_vec_size[1:]}X.f32.{mma_a_ptx_type}.{mma_b_ptx_type}.f32.{mma_sf_format.lower()} "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13}, $14, {%d, %d}, $15, {%d, %d};"
)


def _iv(x):
    return x.ir_value() if hasattr(x, "ir_value") else x


def _mma_16x8_bs(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sfa, byte_id_a, thread_id_a, sfb, byte_id_b, thread_id_b):
    """One warp-wide block-scaled mma.sync on a (16, 8, 32-byte-K) fragment.

    A carrier: 4x b32 regs (ldmatrix.x4 of a [16 x 32B] K-major SMEM region).
    B carrier: 2x b32 regs. D/C: 4 fp32 accumulator regs. ``sfa`` / ``sfb``:
    one 32-bit SF word per lane; the ``{byte_id, thread_id}`` immediates select
    the byte and the lane group the instruction reads.
    """
    if _MMA_INLINE_PTX:
        # The same instruction as inline PTX (sm120_matmul.py's style), its
        # {byte_id, thread_id} immediates formatted into the string.
        return cute.arch.inline_ptx(
            _MMA_PTX % (byte_id_a, thread_id_a, byte_id_b, thread_id_b),
            write_only_types=[cutlass.Float32] * 4,
            read_only_args=[a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sfa, sfb],
        )
    i16 = _T.i16()
    res = nvvm.mma_block_scale(
        _llvm.StructType.get_literal([_T.f32()] * 4),
        _ir.Attribute.parse(_MMA_SHAPE_ATTR),
        _MMA_SCALE_VEC,
        _MMA_SF_FORMAT,
        _MMA_KIND,
        [_iv(a0), _iv(a1), _iv(a2), _iv(a3)],
        [_iv(b0), _iv(b1)],
        [_iv(c0), _iv(c1), _iv(c2), _iv(c3)],
        _iv(sfa),
        _arith.constant(i16, byte_id_a),
        _arith.constant(i16, thread_id_a),
        _iv(sfb),
        _arith.constant(i16, byte_id_b),
        _arith.constant(i16, thread_id_b),
        multiplicand_a_ptx_type=_MMA_A_TYPE,
        multiplicand_b_ptx_type=_MMA_B_TYPE,
    )
    return tuple(cutlass.Float32(_llvm.extractvalue(_T.f32(), res, [i])) for i in range(4))


def _sf_kblk_words(words):
    """The SF word each k-block's MMA reads (``_NUM_K_BLOCKS`` of them), from
    the ``_SF_WORDS`` words one row / column holds for the K-tile: the loaded
    word itself, its byte picked by the MMA's byte_id -- or, for duplicated
    scales (``_SF_DUP == 2``), bytes [b, b, b+1, b+1] of it (one PRMT each)."""
    if _SF_DUP == 1:
        return [words[w] for w in _SF_WORD_BY_KBLK]
    return [nvvm.prmt(words[w], sel, nvvm.PermuteMode.DEFAULT, hi=0) for w, sel in zip(_SF_WORD_BY_KBLK, _SF_DUP_PRMT_BY_KBLK)]


_FP4_IN_BYTE = 0x3C3C3C3C  # bits [5:2] of every byte: where kind::mxf8f6f4 reads e2m1


def _fp4_frag_regs(words, sel):
    """Two kind::mxf8f6f4 fragment registers of the packed e2m1 operand of a
    mixed pair: the lane's K elements 4t..4t+3 and 16+4t..16+4t+3 (t = lane % 4)
    of its row's 32-element k-block, element 4t+i in bits [5:2] of byte i --
    the bytes an fp8 operand's ldmatrix hands the lane (a0 / a2 of an A row,
    b0 / b1 of a B column). ``words``: the (swizzled) 32-bit SMEM address of
    word t // 2 of the row's 16-byte k-block chunk (32 values, element 2j in
    the low nibble of byte j); the lane needs half t % 2 of that word and of the
    word two ahead, which ``sel`` picks (0x5410 for even t, 0x7632 for odd).
    Every other bit of the registers is zero."""
    # Bytes (2t, 2t+1, 8+2t, 9+2t) of the chunk: the nibbles of K 4t..4t+3 and 16+4t..16+4t+3.
    z = nvvm.prmt(words.load(), sel, nvvm.PermuteMode.DEFAULT, hi=(words + 2).load())
    even = (z << 2) & _FP4_IN_BYTE  # low nibbles (K 4t, 4t+2, 16+4t, 18+4t) to bits [5:2]
    odd = (z >> 2) & _FP4_IN_BYTE  # high nibbles (K 4t+1, 4t+3, 17+4t, 19+4t) to bits [5:2]
    return (
        nvvm.prmt(even, 0x5140, nvvm.PermuteMode.DEFAULT, hi=odd),
        nvvm.prmt(even, 0x7362, nvvm.PermuteMode.DEFAULT, hi=odd),
    )


def _sf_word_offset(r, r_in_block):
    """Int32 word offset (within one stage's SF box) of row/column ``r`` of the
    tile's first 128-block, ``r_in_block`` = ``r`` plus the tile's offset inside
    its block. Block ``r_in_block // 128`` of the box, F8_128x4 atom position
    ``(r%32)*16 + ((r//32)%4)*4`` bytes."""
    return (r_in_block // 128) * _SF_BLOCK_WORDS + (r_in_block % 32) * 4 + (r_in_block // 32) % 4


# ---------------------------------------------------------------------------
# Group-local swizzle width depends on this template's injected constants.
# Tile mapping is shared through tile_helpers.
# ---------------------------------------------------------------------------


@cute.jit
def _moe_auto_swizzle_w(group_rows, n, k, nt_n):
    """N-super-block width for one routed group, resolved per group.

    Same rule as the dense path, but the "M side" is THIS group's token slice, not the
    whole token tensor: block along the shorter of (group tokens, expert weight), capped
    by what L2 can hold onto. ``k`` is the NATIVE element count; the row is packed
    (a mixed pair is budgeted at its wider operand's row).
    """
    if cutlass.const_expr(tile_swizzle_n > 0):
        return tile_swizzle_n
    budget = cutlass.Int64(swizzle_l2_budget_bytes)
    row_bytes = (cutlass.Int64(ab_max_data_bits) * k) // 8
    cap = cutlass.max(budget // (row_bytes * cgrp_tile_mnk[1]), cutlass.Int64(1))
    w = cutlass.min(cutlass.Int64(nt_n), cap)
    rows = cutlass.Int64(group_rows)
    if cutlass.min(rows, n) * row_bytes <= budget and rows <= n:
        w = cutlass.Int64(1)
    return cutlass.Int32(w)


@cute.kernel
def frost_template_kernel(
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    num_experts: cutlass.Int32,
    num_groups: cutlass.Int32,
    first_token_offset: cute.Tensor,
    a_tma_workspace: cute.Tensor,
    tma_a_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_b_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_sfa_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_sfb_desc_0: cutlass.GridConstant[_tma.TensorMap],
    mC_tap_0: cute.Tensor,
    out_stride_m_0: cutlass.Int64,
    out_stride_n_0: cutlass.Int64,
    out_stride_l_0: cutlass.Int64,
    alpha: cute.Tensor,
) -> None:
    tma_a_descs = [tma_a_desc_0]
    tma_b_descs = [tma_b_desc_0]
    tma_sfa_descs = [tma_sfa_desc_0]
    tma_sfb_descs = [tma_sfb_desc_0]

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    elect_one = nvvm.elect_sync()

    tidx = cute.arch.thread_idx()[0]

    # The dynamic tile scheduler's global counter: the one word this kernel
    # keeps in the workspace (no descriptor slots precede it).
    sched_counter_ptr = cute.make_ptr(
        cutlass.Int32,
        (a_tma_workspace.iterator.raw_ptr() + grid_num_clusters * moe_desc_slots * TENSOR_MAP_QWORDS).toint(),
        mem_space=cute.AddressSpace.generic,
    )

    if warp_idx == TMA_WARP_ID:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
        for _i in cutlass.range_constexpr(num_sfa_operands):
            nvvm.prefetch_tensormap(tma_sfa_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())
        for _j in cutlass.range_constexpr(num_sfb_operands):
            nvvm.prefetch_tensormap(tma_sfb_descs[_j].get_ptr())

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)

    # Scheduler ring: one record per stage --
    # [0] expert (B's batch coordinate), [1] tile_m within the group, [2] tile_n,
    # [3] valid, [4] group_begin, [5] group_end, [6] the group's first SFA
    # 128-row block in the segmented blob, [7] routed group index.
    sched_storage = cutlass.Array(
        cutlass.Int32,
        SCHED_STAGES * SCHED_SLOT_WORDS,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    sched_full_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    sched_empty_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)

    sA_elems = sA_packed_elems
    sB_elems = sB_packed_elems
    smem_a_list = [
        cutlass.Array(
            a_smem_dtype,
            sA_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_b_list = [
        cutlass.Array(
            b_smem_dtype,
            sB_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_b_operands)
    ]
    # Scale factors, one TMA box per operand per stage, addressed in 32-bit
    # words (one word = the 4 K-scales of one row of one atom). An SF box is
    # unswizzled, so its TMA destination needs 128-byte alignment only; 512
    # (the smallest box, one mxfp8 128-row block) packs the rings back to back,
    # leaving at most one 512-B pad before the 1024-aligned staging below. A
    # 1024-aligned ring would pad each odd-stage mxfp8 ring by 512 B, which the
    # renderer's linear stage accounting (2048-B reserve) does not budget: with
    # several operands the CTA then asks for more than the device's SMEM. A
    # one-sided graph's raw-FP8 side has no ring (num_sf*_operands = 0).
    smem_sfa_list = [
        cutlass.Array(
            cutlass.Int32,
            _SFA_STAGE_WORDS * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=512,
        )
        for _ in range(num_sfa_operands)
    ]
    smem_sfb_list = [
        cutlass.Array(
            cutlass.Int32,
            _SFB_STAGE_WORDS * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=512,
        )
        for _ in range(num_sfb_operands)
    ]

    # Per-compute-warp staging stream for the transposed STG epilogue (raw
    # accumulator dtype; 4 batches x (128 elems + 16B pad) = 528 elems, one
    # 32-column group of one m-frag at a time). Slices are warp-private, so
    # the round trip only needs bar.warp syncs -- no CTA barrier.
    smem_stg_epi = cutlass.Array(
        mma_c_dtype,
        _STG_EPI_WARP_ELEMS * NUM_COMPUTE_WARPS,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )

    # ab full: one producer-elected arrive_expect_tx per stage (data + SF bytes).
    # ab empty: one elected arrive per compute warp per stage.
    # sched full: one elected arrive by the scheduler warp per record.
    # sched empty: one elected arrive per consumer warp (compute + TMA) per record.
    # GATHER: one linear-scale scratch + barrier per token operand with scales
    # (a raw-FP8 token gathers none); each repack is one more ab-full arrival.
    sf_gather_scratch = [
        cutlass.Array(cutlass.Uint8, cta_tile_mnk[0] * 32, space=cutlass.AddressSpace.smem, alignment=128) for _ in range(num_sfa_operands if moe_gather else 0)
    ]
    sf_gather_mbar = [cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem) for _ in range(num_sfa_operands if moe_gather else 0)]
    if warp_idx == 0:
        for _bar in sf_gather_mbar:
            if elect_one:
                nvvm.mbarrier_init(_bar, 1)
        for i in range(ab_stages):
            if elect_one:
                nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1 + (num_sfa_operands if moe_gather else 0))
            if elect_one:
                nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), NUM_COMPUTE_WARPS)
        for i in range(SCHED_STAGES):
            if elect_one:
                nvvm.mbarrier_init(sched_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(sched_empty_mbar_ptr.subview(i), NUM_SCHED_CONSUMER_WARPS)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync(0)

    # TMA completion counts GMEM payload bytes per stage: each operand's data box
    # (every side lands as stored, a mixed pair's packed e2m1 one included, so
    # its bytes are its SMEM tile) plus the SF box of every side with scales --
    # the GATHER repack releases its own SFA arrival.
    num_tma_copy_bytes = (
        num_a_operands * sA_tma_bytes
        + num_b_operands * sB_tma_bytes
        + (0 if moe_gather else num_sfa_operands * sfa_smem_bytes)
        + num_sfb_operands * sfb_smem_bytes
    )

    gC_tap_0_ptr = mC_tap_0.iterator.raw_ptr()
    VEC_BYTES_TAP_0 = vec_bytes_tap_0
    _red_part_args = ()

    VEC_BYTES = vec_bytes_epi
    vsize = epi_chunk_elems

    M = m
    N = n
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    # Every group is cut into the same N tiling; only its M tiling is its own.
    tiles_along_n = cute.ceil_div(cutlass.Int32(N), cgrp_tile_mnk[1])
    first_token_arr = cutlass.make_array_view(first_token_offset)

    # -- Producer warpgroup ---------------------------------------------------
    # Warps NUM_COMPUTE_WARPS.. (TMA, scheduler, donors) are whole warpgroups:
    # release their registers once, warpgroup-uniformly, before the roles below
    # diverge per warp. Donor warps have no further role and simply exit.
    if warp_idx >= NUM_COMPUTE_WARPS:
        nvvm.setmaxregister(PROD_REG_COUNT, nvvm.SetMaxRegisterAction.DECREASE)

    # -- Grouped scheduler warp ------------------------------------------------
    # Claim the next GLOBAL linear tile index off the counter, locate the group
    # it falls in (a warp-parallel prefix scan over the group sizes in INDEX
    # order -- the segmented SFA blob is laid out in that order -- carrying the
    # SF-block prefix alongside the tile prefix, resumed from the last hit),
    # split the group-local index into (tile_m, tile_n) under the group's own
    # L2 raster, and publish.
    if warp_idx == SCHEDULER_WARP_ID:
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        full_warp_mask = 0xFFFFFFFF
        shfl_idx_clamp = 0x1F
        shfl_up_clamp = 0
        lane = cute.arch.lane_idx()
        sched_stage = cutlass.Int32(0)
        sched_empty_phase = cutlass.Int32(1)
        linear_idx = cutlass.Int32(0)
        start_linear_idx = cutlass.Int32(0)
        total_tiles = cutlass.Int32(0)
        start_sf_block_m = cutlass.Int32(0)
        scan_base = cutlass.Int32(0)
        group_idx = cutlass.Int32(0)
        group_begin = cutlass.Int32(0)
        group_end = cutlass.Int32(0)
        is_tile_valid = cutlass.Int32(1)

        while is_tile_valid != 0:
            # Dynamic tile assignment: the CTAs live at any instant sit in one
            # contiguous window of tile space and share L2. No cluster, so the
            # claimed index is broadcast within the warp only.
            claimed = cutlass.Int32(0)
            if lane == 0:
                claimed = nvvm.atomicrmw(
                    "add",
                    sched_counter_ptr,
                    cutlass.Int32(1),
                    mem_order="relaxed",
                    syncscope="gpu",
                )
            linear_idx = nvvm.shfl_sync(full_warp_mask, claimed, 0, shfl_idx_clamp, nvvm.Shfl.IDX)
            if linear_idx >= start_linear_idx + total_tiles:
                is_search_live = cutlass.Int32(1)
                while is_search_live != 0:
                    my_group = scan_base + lane
                    my_begin = cutlass.Int32(0)
                    my_end = cutlass.Int32(0)
                    my_tiles = cutlass.Int32(0)
                    my_sf_blocks = cutlass.Int32(0)
                    if my_group < num_groups:
                        if my_group != 0:
                            my_begin = cutlass.Int32(first_token_arr[my_group])
                        my_end = cutlass.Int32(first_token_arr[my_group + 1])
                        my_tiles = cute.ceil_div(my_end - my_begin, cgrp_tile_mnk[0]) * tiles_along_n
                        my_sf_blocks = cute.ceil_div(my_end - my_begin, 128)
                    prefix_tiles = my_tiles
                    prefix_sf = my_sf_blocks
                    for delta in (1, 2, 4, 8, 16):
                        prefix_delta = nvvm.shfl_sync(
                            full_warp_mask,
                            prefix_tiles,
                            delta,
                            shfl_up_clamp,
                            nvvm.Shfl.UP,
                        )
                        prefix_sf_delta = nvvm.shfl_sync(
                            full_warp_mask,
                            prefix_sf,
                            delta,
                            shfl_up_clamp,
                            nvvm.Shfl.UP,
                        )
                        if lane >= delta:
                            prefix_tiles += prefix_delta
                            prefix_sf += prefix_sf_delta
                    my_start = start_linear_idx + prefix_tiles - my_tiles
                    my_sf_start = start_sf_block_m + prefix_sf - my_sf_blocks
                    thread_succeed = nvvm.vote_sync(
                        full_warp_mask,
                        linear_idx < my_start + my_tiles,
                        nvvm.VoteSync.BALLOT,
                    )
                    if thread_succeed != 0:
                        winning_lane = cutlass.Int32(31) - cute.arch.bfind(cute.arch.brev(thread_succeed)).to(cutlass.Int32)
                        scan_base = nvvm.shfl_sync(full_warp_mask, my_group, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        group_idx = scan_base
                        group_begin = nvvm.shfl_sync(full_warp_mask, my_begin, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        group_end = nvvm.shfl_sync(full_warp_mask, my_end, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        start_linear_idx = nvvm.shfl_sync(full_warp_mask, my_start, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        total_tiles = nvvm.shfl_sync(full_warp_mask, my_tiles, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        start_sf_block_m = nvvm.shfl_sync(full_warp_mask, my_sf_start, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        is_search_live = cutlass.Int32(0)
                    else:
                        start_linear_idx = nvvm.shfl_sync(
                            full_warp_mask,
                            my_start + my_tiles,
                            31,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        start_sf_block_m = nvvm.shfl_sync(
                            full_warp_mask,
                            my_sf_start + my_sf_blocks,
                            31,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        scan_base += 32
                        if scan_base >= num_groups:
                            is_tile_valid = cutlass.Int32(0)
                            is_search_live = cutlass.Int32(0)

            coord_expert = cutlass.Int32(0)
            tile_m = cutlass.Int32(0)
            tile_n = cutlass.Int32(0)
            if is_tile_valid != 0:
                local_linear_idx = linear_idx - start_linear_idx
                group_nt_m = total_tiles // tiles_along_n
                tile_m, tile_n = _moe_swizzle_tile(
                    local_linear_idx,
                    group_nt_m,
                    tiles_along_n,
                    _moe_auto_swizzle_w(group_nt_m * cgrp_tile_mnk[0], N, k, tiles_along_n),
                )
                coord_expert = group_idx % num_experts

            while not nvvm.mbarrier_try_wait_parity(
                sched_empty_mbar_ptr.subview(sched_stage),
                sched_empty_phase,
                time_limit=10_000_000,
            ):
                pass
            if lane == 0:
                slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
                (slot.subview(0)).store(coord_expert)
                (slot.subview(1)).store(tile_m)
                (slot.subview(2)).store(tile_n)
                (slot.subview(3)).store(is_tile_valid)
                (slot.subview(4)).store(group_begin)
                (slot.subview(5)).store(group_end)
                (slot.subview(6)).store(start_sf_block_m)
                (slot.subview(7)).store(group_idx)
                nvvm.mbarrier_arrive(sched_full_mbar_ptr.subview(sched_stage))

            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_empty_phase = sched_empty_phase ^ 1

    # -- TMA producer warp ----------------------------------------------------
    if warp_idx == TMA_WARP_ID:
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        ab_empty_phase_bit = cutlass.Int32(1)
        ab_iter = cutlass.Int32(0)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        while is_valid != 0:
            while not nvvm.mbarrier_try_wait_parity(
                sched_full_mbar_ptr.subview(sched_stage),
                sched_full_phase,
                time_limit=10_000_000,
            ):
                pass
            slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
            coord_expert = (slot.subview(0)).load()
            tile_m = (slot.subview(1)).load()
            tile_n = (slot.subview(2)).load()
            is_valid = (slot.subview(3)).load()
            group_begin = (slot.subview(4)).load()
            group_end = (slot.subview(5)).load()
            start_sf_block_m = (slot.subview(6)).load()
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                # The tile's rows in the flat token matrix: the group's first row
                # plus its m-tile offset. The ONE global A descriptor clips at S;
                # rows past group_end are loaded and never stored.
                coord_m_local = tile_m * cgrp_tile_mnk[0]
                coord_m = group_begin + coord_m_local
                coord_n = tile_n * cgrp_tile_mnk[1]
                # SFA is GROUP-LOCAL in the segmented blob: the group's first block
                # plus the tile's block offset within the group. SFB is per expert.
                sfa_m_block = start_sf_block_m + coord_m_local // 128
                sfb_n_block = coord_n // 128

                if cutlass.const_expr(moe_gather):
                    # Lane l owns the tile's 4-row GATHER groups l, l + 32, ...:
                    # it reads their source rows once per tile and issues their
                    # GATHER4s itself, so no lane holds the whole tile's rows
                    # (one lane holding them all spilled once the producer warps
                    # drop to PROD_REG_COUNT registers).
                    gather_lane = cute.arch.lane_idx()
                    gather_rows = []
                    for _lg in cutlass.range_constexpr(_GATHER_LANE_GROUPS):
                        _row = coord_m + (_lg * 32 + gather_lane) * 4
                        gather_rows.append([moe_gather_row(token_index, _row + _j, group_end, source_rows) for _j in range(4)])

                for k_tile_idx in range(num_k_tiles):
                    stage = ab_iter % ab_stages
                    if stage == 0 and ab_iter != 0:
                        ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(stage), ab_empty_phase_bit, time_limit=10_000_000):
                        pass

                    coord_k = k_tile_idx * cta_tile_mnk[2]
                    coord_sf_k = k_tile_idx * sf_tma_box_k
                    # One elected lane only: the barrier's arrival count is 1, and
                    # the TMA copies deliver exactly num_tma_copy_bytes once.
                    if elect_one:
                        nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)
                    if cutlass.const_expr(moe_gather):
                        # Every lane's GATHER4 completes on this stage's barrier:
                        # order them after the elected lane's expect_tx.
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                    # K-major token: one TMA box [K_tile, cta_m] at (k, m, 0) on the
                    # NATIVE dtype (fp4 packs via B4X16); OOB rows/cols are hardware
                    # zero-filled (K tails contribute 0).
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        if cutlass.const_expr(moe_gather):
                            for _lg in cutlass.range_constexpr(_GATHER_LANE_GROUPS):
                                _gr = _lg * 32 + gather_lane
                                if _gr < cta_tile_mnk[0] // 4:
                                    tma_gather4(
                                        smem_a_list[_ai].subview(sA_elems * stage + _gr * 4 * _A_ROW_BYTES),
                                        tma_a_descs[_ai].get_ptr(),
                                        coord_k,
                                        gather_rows[_lg][0],
                                        gather_rows[_lg][1],
                                        gather_rows[_lg][2],
                                        gather_rows[_lg][3],
                                        ab_full_mbar_ptr.subview(stage),
                                    )
                            # tile_helpers.moe_gather_scales with the same lane split: each
                            # lane gathers its rows' linear token scales into the scratch,
                            # then the warp repacks the tile into the F8_128x4 stage and
                            # one lane releases it on the stage barrier. A raw-FP8 token
                            # (one-sided dequant) has no scales: nothing to gather.
                            if cutlass.const_expr(not fake_dequant_a):
                                if elect_one:
                                    nvvm.mbarrier_arrive_expect_tx(sf_gather_mbar[_ai], cta_tile_mnk[0] * 16)
                                nvvm.bar_warp_sync(0xFFFFFFFF)
                                scale_k = coord_k // block_size
                                for _lg in cutlass.range_constexpr(_GATHER_LANE_GROUPS):
                                    _gr = _lg * 32 + gather_lane
                                    if _gr < cta_tile_mnk[0] // 4:
                                        tma_gather4(
                                            sf_gather_scratch[_ai].subview(_gr * 128),
                                            tma_sfa_descs[_ai].get_ptr(),
                                            (scale_k // 16) * 16,
                                            gather_rows[_lg][0],
                                            gather_rows[_lg][1],
                                            gather_rows[_lg][2],
                                            gather_rows[_lg][3],
                                            sf_gather_mbar[_ai],
                                        )
                                while not nvvm.mbarrier_try_wait_parity(sf_gather_mbar[_ai], ab_iter % 2, time_limit=10_000_000):
                                    pass
                                sf_src_words = cutlass.inttoptr(sf_gather_scratch[_ai].data_ptr().toint(), 3, cutlass.Uint32)
                                sf_dst_words = cutlass.inttoptr(smem_sfa_list[_ai].subview(_SFA_STAGE_WORDS * stage).data_ptr().toint(), 3, cutlass.Uint32)
                                for _i in cutlass.range(cta_tile_mnk[0] * (cta_tile_mnk[2] // block_size // 4) // 32, unroll_full=True):
                                    sf_word = gather_lane + _i * 32
                                    sf_row = sf_word // (cta_tile_mnk[2] // block_size // 4)
                                    sf_kw = sf_word % (cta_tile_mnk[2] // block_size // 4)
                                    sf_packed = (
                                        (sf_row // 128) * (128 * (cta_tile_mnk[2] // block_size // 4)) + sf_kw * 128 + (sf_row % 32) * 4 + (sf_row % 128) // 32
                                    )
                                    (sf_dst_words + sf_packed).store(
                                        (sf_src_words + (sf_row // 4) * 32 + (sf_row % 4) * 4 + (scale_k % 16) // 4 + sf_kw).load()
                                    )
                                cute.arch.fence_view_async_shared()
                                nvvm.bar_warp_sync(0xFFFFFFFF)
                                if elect_one:
                                    nvvm.mbarrier_arrive(ab_full_mbar_ptr.subview(stage))
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cta_global(
                                    smem_a_list[_ai].subview(sA_elems * stage),
                                    tma_a_descs[_ai].get_ptr(),
                                    (coord_k, coord_m, cutlass.Int32(0)),
                                    ab_full_mbar_ptr.subview(stage),
                                )
                            # SFA: [256 fp16 (= one 512 B atom), sf_k4 atoms, m blocks, 1]
                            # -- the whole K-tile of scales for the tile's M blocks (none
                            # for a raw-FP8 token).
                            if cutlass.const_expr(not fake_dequant_a):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cta_global(
                                        smem_sfa_list[_ai].subview(_SFA_STAGE_WORDS * stage),
                                        tma_sfa_descs[_ai].get_ptr(),
                                        (0, coord_sf_k, sfa_m_block, cutlass.Int32(0)),
                                        ab_full_mbar_ptr.subview(stage),
                                    )
                    # The expert's weight is B's batch coordinate. K-major B: box
                    # [K_tile, cta_n] at (k, n, e); N-major B (fp8) walks N in
                    # b_tma_group_elems-wide groups (same row bytes as a K-major row).
                    for _bj in cutlass.range_constexpr(num_b_operands):
                        if cutlass.const_expr(b_is_n_major):
                            for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cta_global(
                                        smem_b_list[_bj].subview(sB_elems * stage + n_group * b_tma_group_elems * _B_ROW_BYTES),
                                        tma_b_descs[_bj].get_ptr(),
                                        (coord_n + n_group * b_tma_group_elems, coord_k, coord_expert),
                                        ab_full_mbar_ptr.subview(stage),
                                    )
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cta_global(
                                    smem_b_list[_bj].subview(sB_elems * stage),
                                    tma_b_descs[_bj].get_ptr(),
                                    (coord_k, coord_n, coord_expert),
                                    ab_full_mbar_ptr.subview(stage),
                                )
                        # SFB (none for a raw-FP8 weight).
                        if cutlass.const_expr(not fake_dequant_b):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cta_global(
                                    smem_sfb_list[_bj].subview(_SFB_STAGE_WORDS * stage),
                                    tma_sfb_descs[_bj].get_ptr(),
                                    (0, coord_sf_k, sfb_n_block, coord_expert),
                                    ab_full_mbar_ptr.subview(stage),
                                )
                    ab_iter += 1

    # -- Compute warps: block-scaled mma.sync mainloop + epilogue ------------
    if warp_idx < NUM_COMPUTE_WARPS:
        nvvm.setmaxregister(EPI_REG_COUNT, nvvm.SetMaxRegisterAction.INCREASE)
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")

        lane = tidx % 32
        lane_div4 = lane // 4
        lane_mod4 = lane % 4
        warp_row = warp_idx % WARPS_M
        warp_col = warp_idx // WARPS_M

        # ldmatrix lane->address maps (see PTX ldmatrix; addresses are 16B rows).
        # A x4 tile order = (rows 0-7, rows 8-15) x (16B col 0, 16B col 1) --
        # matching the a0..a3 fragment order of mma.sync.
        a_ldm_row = (lane % 8) + 8 * ((lane // 8) % 2)
        a_ldm_col16 = lane // 16
        # B x4 covers TWO 8-col n-frags: (n rows 0-7, n rows 8-15) each split
        # over (16B col 0, 16B col 1) -> regs (b0,b1) frag0 + (b0,b1) frag1.
        b_ldm_pair_row = (lane % 8) + 8 * (lane // 16)
        b_ldm_pair_col16 = (lane // 8) % 2
        # B x2 tail: one n-frag (rows 0-7 x two 16B cols; lanes 16-31 unused).
        b_ldm_tail_row = lane % 8
        b_ldm_tail_col16 = (lane // 8) % 2
        # A mixed pair's packed e2m1 side (_fp4_frag_regs): lane (g, t) =
        # (lane // 4, lane % 4) reads words t // 2 and t // 2 + 2 of the 16-byte
        # k-block chunk of a fragment's row g (A: and row g + 8) and keeps half
        # t % 2 of each. A fragment starts at a multiple of 8 rows of 64 B and a
        # stage at a multiple of 512 B, so the s64b XOR of the lane's row is
        # (g // 2) % 4 in every fragment and stage: the lane's word offset inside
        # a fragment is a constant per k-block.
        fp4_lane_words = [
            lane_div4 * 16 + ((cutlass.Int32(kb * 16) ^ (((lane_div4 // 2) % 4) * 16)) + (lane_mod4 // 2) * 4) // 4 for kb in range(_NUM_K_BLOCKS)
        ]
        fp4_sel = cutlass.Int32(0x5410) + (lane_mod4 % 2) * 0x2222

        # Scale-factor lane roles (see sm120_block_scale_matmul.py). Lane L
        # supplies A row (L>>2) + 8*(L&1) of the m-frag its lane group ((L>>1)&1)
        # owns, and B column L>>2 of the n-frag its lane group (L&3) owns.
        sf_lane_row = lane_div4 + 8 * (lane % 2)
        sfa_lane_sel = (lane // 2) % 2
        sfb_lane_sel = lane_mod4
        # The identity scale a one-sided graph's raw-FP8 side reads: 1.0 in every
        # byte, so any {byte_id, thread_id} selects it.
        sf_one = cutlass.Int32(sf_one_word)

        # One register accumulator per GEMM of the chain (multi-GEMM: past the
        # warp's register grant ptxas spills -- a perf trade-off, never a gate).
        acc_list = [cutlass.Array(mma_c_dtype, _ACC_REGS, alignment=16) for _g in range(num_gemms)]

        ab_full_phase_bit = cutlass.Int32(0)
        ab_iter = cutlass.Int32(0)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)
        # The routed output is one flat (S, N) surface: no batch term.
        tile_l = cutlass.Int32(0)

        while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr.subview(sched_stage), sched_full_phase, time_limit=10_000_000):
            pass
        _slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
        tile_m = (_slot.subview(1)).load()
        tile_n = (_slot.subview(2)).load()
        is_valid = (_slot.subview(3)).load()
        group_begin = (_slot.subview(4)).load()
        group_end = (_slot.subview(5)).load()
        start_sf_block_m = (_slot.subview(6)).load()
        group_idx = (_slot.subview(7)).load()
        nvvm.bar_warp_sync(0xFFFFFFFF)
        if elect_one:
            nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
        sched_stage += 1
        if sched_stage == SCHED_STAGES:
            sched_stage = cutlass.Int32(0)
            sched_full_phase = sched_full_phase ^ 1

        # Tile-invariant (scalar) reductions fold every epilogue vector into one
        # register slot per thread; the red_part_* helpers, rendered with the tile
        # constants (empty when the chain has none), seed the slots here and flush
        # them -- one warp shuffle and one atomic per warp -- after each tile's
        # drain and after the tile loop, where all 32 lanes are converged.
        _red_part = cute.make_rmem_tensor(red_part_slots, cutlass.Float32)
        red_part_seed(_red_part)
        while is_valid != 0:
            coord_m_local = tile_m * cgrp_tile_mnk[0]
            coord_m = group_begin + coord_m_local
            coord_n = tile_n * cgrp_tile_mnk[1]

            # Per-lane SF word offsets inside a stage's SF box, one per m-frag
            # pair / n-frag quad. They depend on the tile only through its
            # offset inside its 128-block (zero for a block-aligned tile). The A
            # side is GROUP-LOCAL: the segmented blob restarts at a block
            # boundary for every group, so the offset is the tile's, not the
            # global token row's.
            if cutlass.const_expr(_M_BLOCK_ALIGNED or moe_gather):
                m_in_block = cutlass.Int32(0)
            else:
                m_in_block = coord_m_local % 128
            if cutlass.const_expr(_N_BLOCK_ALIGNED):
                n_in_block = cutlass.Int32(0)
            else:
                n_in_block = coord_n % 128
            sfa_word_off = []
            for g in cutlass.range_constexpr(_SFA_GROUPS):
                # Lane group 1 of the last (odd) pair duplicates frag 2g.
                mf_sel = cutlass.min(cutlass.Int32(2 * g) + sfa_lane_sel, cutlass.Int32(_M_FRAGS - 1))
                r = warp_row * _WARP_TILE_M + mf_sel * 16 + sf_lane_row
                sfa_word_off.append(_sf_word_offset(r, m_in_block + r))
            sfb_word_off = []
            for g in cutlass.range_constexpr(_SFB_GROUPS):
                nf_sel = cutlass.min(cutlass.Int32(4 * g) + sfb_lane_sel, cutlass.Int32(_N_FRAGS - 1))
                c = warp_col * _WARP_TILE_N + nf_sel * 8 + lane_div4
                sfb_word_off.append(_sf_word_offset(c, n_in_block + c))

            for _g in cutlass.range_constexpr(num_gemms):
                _acc_g = acc_list[_g]
                for _z in cutlass.range_constexpr(_ACC_REGS):
                    _acc_g[_z] = mma_c_dtype(0)

            for k_tile_idx in range(num_k_tiles):
                stage = ab_iter % ab_stages
                if stage == 0 and ab_iter != 0:
                    ab_full_phase_bit = ab_full_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(ab_full_mbar_ptr.subview(stage), ab_full_phase_bit, time_limit=10_000_000):
                    pass

                sA_ptrs = [smem_a_list[_ai].subview(sA_elems * stage).data_ptr() for _ai in range(num_a_operands)]
                sB_ptrs = [smem_b_list[_bj].subview(sB_elems * stage).data_ptr() for _bj in range(num_b_operands)]
                sSFA_ptrs = [smem_sfa_list[_ai].subview(_SFA_STAGE_WORDS * stage).data_ptr() for _ai in range(num_sfa_operands)]
                sSFB_ptrs = [smem_sfb_list[_bj].subview(_SFB_STAGE_WORDS * stage).data_ptr() for _bj in range(num_sfb_operands)]

                # The K-tile's SF words of every distinct operand, one per (frag
                # group, K word): word w of a row sits one 512 B atom (128 words)
                # after word w-1. The word offsets are the tile's, shared by all
                # operands of one side. A one-sided graph's raw-FP8 side reads
                # the identity scale register instead.
                if cutlass.const_expr(fake_dequant_a):
                    sfa_words_list = [[[sf_one] * _NUM_K_BLOCKS for _g in range(_SFA_GROUPS)] for _ai in range(num_a_operands)]
                else:
                    sfa_words_list = []
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        sfa_words = []
                        for g in cutlass.range_constexpr(_SFA_GROUPS):
                            _ws = []
                            for w in cutlass.range_constexpr(_SF_WORDS):
                                _ws.append((sSFA_ptrs[_ai] + sfa_word_off[g] + w * _SF_ATOM_WORDS).load())
                            sfa_words.append(_sf_kblk_words(_ws))
                        sfa_words_list.append(sfa_words)
                if cutlass.const_expr(fake_dequant_b):
                    sfb_words_list = [[[sf_one] * _NUM_K_BLOCKS for _g in range(_SFB_GROUPS)] for _bj in range(num_b_operands)]
                else:
                    sfb_words_list = []
                    for _bj in cutlass.range_constexpr(num_b_operands):
                        sfb_words = []
                        for g in cutlass.range_constexpr(_SFB_GROUPS):
                            _ws = []
                            for w in cutlass.range_constexpr(_SF_WORDS):
                                _ws.append((sSFB_ptrs[_bj] + sfb_word_off[g] + w * _SF_ATOM_WORDS).load())
                            sfb_words.append(_sf_kblk_words(_ws))
                        sfb_words_list.append(sfb_words)

                for k_blk in cutlass.range_constexpr(_NUM_K_BLOCKS):
                    _sf_byte = _SF_MMA_BYTE_BY_KBLK[k_blk]
                    a_frags_list = []
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        sA_ptr = sA_ptrs[_ai]
                        a_frags = []
                        if cutlass.const_expr(_A_FP4_TO_BYTES):
                            # A mixed pair's packed e2m1 token: rows g (a0, a2) and
                            # g + 8 (a1, a3) of each m-frag, built in registers.
                            sA_words = cutlass.inttoptr(sA_ptr.toint(), 3, cutlass.Uint32)
                            for mf in cutlass.range_constexpr(_M_FRAGS):
                                a_fp4_words = sA_words + (warp_row * _WARP_TILE_M + mf * 16) * (_A_ROW_BYTES // 4) + fp4_lane_words[k_blk]
                                a0, a2 = _fp4_frag_regs(a_fp4_words, fp4_sel)
                                a1, a3 = _fp4_frag_regs(a_fp4_words + 8 * (_A_ROW_BYTES // 4), fp4_sel)
                                a_frags.append((a0, a1, a2, a3))
                        else:
                            for mf in cutlass.range_constexpr(_M_FRAGS):
                                a_row = warp_row * _WARP_TILE_M + mf * 16 + a_ldm_row
                                a_off = a_row * _A_ROW_BYTES + k_blk * _A_KBLK_BYTES + a_ldm_col16 * _ELEMS_16B
                                a_frags.append(
                                    nvvm.ldmatrix(
                                        _apply_smem_swizzle(sA_ptr + a_off, _A_SWIZZLE),
                                        4,
                                        nvvm.MMALayout.ROW,
                                    )
                                )
                        a_frags_list.append(a_frags)
                    for _bj in cutlass.range_constexpr(num_b_operands):
                        sB_ptr = sB_ptrs[_bj]
                        b_frags = []
                        if cutlass.const_expr(b_is_n_major):
                            # fp8 only. ldmatrix.m16n16.x2.trans.b8 per n-frag pair: the
                            # tile's 16 transposed columns span n-frags (2p, 2p+1), so the
                            # result regs are [b0(2p), b0(2p+1), b1(2p), b1(2p+1)].
                            # Addresses: k = k_blk * 32 + lane, no lane map. (_N_FRAGS is
                            # even here -- asserted at render.)
                            for npair in cutlass.range_constexpr(_N_FRAG_PAIRS):
                                b_n = warp_col * _WARP_TILE_N + npair * 16
                                b_off = (
                                    (b_n // b_tma_group_elems) * (b_tma_group_elems * _B_ROW_BYTES)
                                    + (k_blk * _B_KBLK_BYTES + lane) * b_tma_group_elems
                                    + b_n % b_tma_group_elems
                                )
                                bv = nvvm.ldmatrix(
                                    _apply_smem_swizzle(sB_ptr + b_off, _B_SWIZZLE),
                                    4,
                                    nvvm.MMALayout.COL,
                                    shape=nvvm.LoadShape.M16N16,
                                    src_format=nvvm.LoadSrcFormat.B8,
                                )
                                b_frags.append((bv[0], bv[2]))
                                b_frags.append((bv[1], bv[3]))
                        elif cutlass.const_expr(_B_FP4_TO_BYTES):
                            # A mixed pair's packed e2m1 weight: column g of each n-frag
                            # (b0, b1), built in registers.
                            sB_words = cutlass.inttoptr(sB_ptr.toint(), 3, cutlass.Uint32)
                            for nf in cutlass.range_constexpr(_N_FRAGS):
                                b_frags.append(
                                    _fp4_frag_regs(sB_words + (warp_col * _WARP_TILE_N + nf * 8) * (_B_ROW_BYTES // 4) + fp4_lane_words[k_blk], fp4_sel)
                                )
                        else:
                            for npair in cutlass.range_constexpr(_N_FRAG_PAIRS):
                                b_row = warp_col * _WARP_TILE_N + npair * 16 + b_ldm_pair_row
                                b_off = b_row * _B_ROW_BYTES + k_blk * _B_KBLK_BYTES + b_ldm_pair_col16 * _ELEMS_16B
                                bv = nvvm.ldmatrix(
                                    _apply_smem_swizzle(sB_ptr + b_off, _B_SWIZZLE),
                                    4,
                                    nvvm.MMALayout.ROW,
                                )
                                b_frags.append((bv[0], bv[1]))
                                b_frags.append((bv[2], bv[3]))
                            if cutlass.const_expr(_N_FRAGS % 2 == 1):
                                b_row = warp_col * _WARP_TILE_N + (_N_FRAGS - 1) * 8 + b_ldm_tail_row
                                b_off = b_row * _B_ROW_BYTES + k_blk * _B_KBLK_BYTES + b_ldm_tail_col16 * _ELEMS_16B
                                bt = nvvm.ldmatrix(
                                    _apply_smem_swizzle(sB_ptr + b_off, _B_SWIZZLE),
                                    2,
                                    nvvm.MMALayout.ROW,
                                )
                                b_frags.append((bt[0], bt[1]))

                        # Every GEMM fed by this B operand (trace-time selection); its
                        # A operand's fragments and SFA words come with it.
                        for _g in cutlass.range_constexpr(num_gemms):
                            if cutlass.const_expr(gemm_b_idx[_g] == _bj):
                                _acc_g = acc_list[_g]
                                a_frags = a_frags_list[gemm_a_idx[_g]]
                                sfa_words = sfa_words_list[gemm_a_idx[_g]]
                                sfb_words = sfb_words_list[_bj]
                                for mf in cutlass.range_constexpr(_M_FRAGS):
                                    av = a_frags[mf]
                                    _sfa = sfa_words[mf // 2][k_blk]
                                    for nf in cutlass.range_constexpr(_N_FRAGS):
                                        b0, b1 = b_frags[nf]
                                        _sfb = sfb_words[nf // 4][k_blk]
                                        _o = (mf * _N_FRAGS + nf) * 4
                                        _acc_g[_o:4] = _mma_16x8_bs(
                                            av[0],
                                            av[1],
                                            av[2],
                                            av[3],
                                            b0,
                                            b1,
                                            _acc_g[_o + 0],
                                            _acc_g[_o + 1],
                                            _acc_g[_o + 2],
                                            _acc_g[_o + 3],
                                            _sfa,
                                            _sf_byte,
                                            mf % 2,
                                            _sfb,
                                            _sf_byte,
                                            nf % 4,
                                        )
                nvvm.bar_warp_sync(0xFFFFFFFF)
                cute.arch.fence_proxy("async.shared", space="cta")
                if elect_one:
                    nvvm.mbarrier_arrive(ab_empty_mbar_ptr.subview(stage))
                ab_iter += 1

            # -- Epilogue: accumulators are already in registers ------------------

            _aux_alpha_ptr = alpha.iterator.raw_ptr()
            _aux_alpha_pre = (_aux_alpha_ptr + group_idx).load()

            _stg_stage = smem_stg_epi.subview(warp_idx * _STG_EPI_WARP_ELEMS)
            if cutlass.const_expr(not row_per_lane_drain):
                for mf in cutlass.range_constexpr(_M_FRAGS):
                    for grp in cutlass.range_constexpr(_STG_EPI_NGRP):
                        _nf0 = grp * _STG_EPI_GROUP_FRAGS
                        _grp_frags = min(_STG_EPI_GROUP_FRAGS, _N_FRAGS - _nf0)
                        # -- STS_128: reg-index-order dump, one batch per fragment; then
                        # LDS: 16 contiguous elems = both row-halves of one frag. One GEMM
                        # at a time through the same warp-private buffer (multi-GEMM keeps
                        # every GEMM's segment in registers, not extra SMEM).
                        _seg_list = []
                        for _g in cutlass.range_constexpr(num_gemms):
                            _acc_g = acc_list[_g]
                            for b in cutlass.range_constexpr(_grp_frags):
                                _o = (mf * _N_FRAGS + _nf0 + b) * 4
                                _s_off = b * _STG_EPI_BATCH_STRIDE + lane * _STG_EPI_LANE_QUAD
                                (_stg_stage.data_ptr() + _s_off).store(_acc_g[_o:4], alignment=16)
                            nvvm.bar_warp_sync(0xFFFFFFFF)
                            _seg_list.append((_stg_stage.data_ptr() + lane_mod4 * _STG_EPI_BATCH_STRIDE + lane_div4 * 16).load(alignment=16, count=16))
                            if cutlass.const_expr(_g + 1 < num_gemms):
                                nvvm.bar_warp_sync(0xFFFFFFFF)  # every lane has its segment before the buffer is reused
                        # Short tail group: trailing lanes own no fragment there
                        # (True at trace time for full groups -- no guard emitted).
                        _lane_active = True if _grp_frags == _STG_EPI_GROUP_FRAGS else lane_mod4 < _grp_frags
                        if _lane_active:
                            for half in cutlass.range_constexpr(2):
                                row_in_cta = warp_row * _WARP_TILE_M + mf * 16 + half * 8 + lane_div4
                                row = coord_m + row_in_cta
                                # The ragged tail of a group: rows at or past group_end
                                # hold the next group's tokens (or zero-fill) scaled by
                                # padding SF words, and are not this expert's output.
                                if row < group_end:
                                    _row_list = []
                                    for _g in cutlass.range_constexpr(num_gemms):
                                        _seg = _seg_list[_g]
                                        _row = cutlass.Array(mma_c_dtype, 8, alignment=16)
                                        for sj in cutlass.range_constexpr(4):
                                            _row[2 * sj] = _seg[4 * sj + 2 * half]
                                            _row[2 * sj + 1] = _seg[4 * sj + 2 * half + 1]
                                        _row_list.append(_row)
                                    for sv in cutlass.range_constexpr(8 // _STG_V):
                                        col = coord_n + warp_col * _WARP_TILE_N + (_nf0 + lane_mod4) * 8 + sv * _STG_V
                                        col_j = col
                                        if col_j + vsize <= N:
                                            # One epilogue input vector per GEMM: GEMM 0 is the
                                            # template's `vec_f32`, GEMMs > 0 are bound as
                                            # `vec_f32_<g>` by the injected STG bindings below.
                                            # NB: Array slices are [start:COUNT], not
                                            # [start:stop] (matches _acc_g[_o:4] above).
                                            c_rmem_vecs = []
                                            for _g in cutlass.range_constexpr(num_gemms):
                                                _vec = _row_list[_g][sv * _STG_V : _STG_V]
                                                if cutlass.const_expr(acc_widen_to_fp32):
                                                    _pf = _vec.to(cutlass.Float32)
                                                    c_rmem_vecs.append(_pf + cutlass.full_like(_pf, 0.0))
                                                else:
                                                    c_rmem_vecs.append(_vec)
                                            vec_f32 = c_rmem_vecs[0]
                                            # Every store carries its own output's offset: no template-level
                                            # linear_idx, so a reduction-only chain (no output 0) renders.

                                            pass

                                            _c_0_a = (vec_f32).to(cutlass.Float32)
                                            _op_0 = _c_0_a * (cutlass.full_like(_c_0_a, _aux_alpha_pre.to(cutlass.Float32)))
                                            _r_0 = (_op_0).to(cutlass.BFloat16)
                                            _tap_0 = (_r_0).to(cutlass.BFloat16)
                                            (gC_tap_0_ptr + (row * out_stride_m_0 + col_j)).store(_tap_0, alignment=VEC_BYTES_TAP_0)
                        nvvm.bar_warp_sync(0xFFFFFFFF)
            else:
                # Row-per-lane drain: re-read the warp's accumulators row-major through the
                # warp-private staging buffer so lane l owns row l of a 32-row slab (lanes
                # 0..15 of a 16-row slab when the warp tile is one m-fragment) and a
                # contiguous run of the slab's 32-column group -- the sm100 TMEM-drain view
                # (thread = row, warp = 32 consecutive rows) the block-quantize emitters are
                # written for. Each 16x32 sub-pass uses 512 of the warp's 528 staging floats.
                for slab in cutlass.range_constexpr(_RPL_SLABS):
                    for grp in cutlass.range_constexpr(_STG_EPI_NGRP):
                        _nf0 = grp * _STG_EPI_GROUP_FRAGS
                        _grp_frags = min(_STG_EPI_GROUP_FRAGS, _N_FRAGS - _nf0)
                        _run_list = []
                        for _g in cutlass.range_constexpr(num_gemms):
                            _acc_g = acc_list[_g]
                            _run = cutlass.Array(mma_c_dtype, 32, alignment=16)
                            for _sp in cutlass.range_constexpr(_RPL_SUBPASSES):
                                _mf = slab * _RPL_SUBPASSES + _sp
                                # STS.64: regs (0,1) are row lane//4, regs (2,3) row lane//4 + 8,
                                # columns 8b + 2*(lane%4) of n-fragment b.
                                for b in cutlass.range_constexpr(_grp_frags):
                                    for _hr in cutlass.range_constexpr(2):
                                        _rr = lane_div4 + 8 * _hr
                                        _off = _rr * 32 + ((2 * b + lane_mod4 // 2) ^ _rpl_swz(_rr)) * 4 + (lane_mod4 % 2) * 2
                                        _o = (_mf * _N_FRAGS + _nf0 + b) * 4 + 2 * _hr
                                        (_stg_stage.data_ptr() + _off).store(_acc_g[_o:2], alignment=8)
                                nvvm.bar_warp_sync(0xFFFFFFFF)
                                # LDS.128: lanes 16*_sp .. 16*_sp + 15 take this fragment's rows.
                                if lane // 16 == _sp:
                                    _r = lane % 16
                                    for _j in cutlass.range_constexpr(2 * _grp_frags):
                                        _run[_j * 4 : 4] = (_stg_stage.data_ptr() + _r * 32 + (_j ^ _rpl_swz(_r)) * 4).load(alignment=16, count=4)
                                nvvm.bar_warp_sync(0xFFFFFFFF)  # the next sub-pass / GEMM reuses the buffer
                            _run_list.append(_run)
                        # Lanes 16..31 own no row when the slab is a single 16-row fragment.
                        _rows_owned = True if _RPL_SUBPASSES == 2 else lane < 16
                        if _rows_owned:
                            row_in_cta = warp_row * _WARP_TILE_M + slab * (16 * _RPL_SUBPASSES) + lane
                            row = coord_m + row_in_cta
                            # A col quant reduces a block's amax across the lanes of a
                            # (half-)warp, so under rpl_bounded_rows every lane reaches it:
                            # the guard is always taken and each store carries
                            # row < group_end itself (generate(..., bounded_rows=True)).
                            if cutlass.const_expr(rpl_bounded_rows):
                                _row_live = True
                            else:
                                _row_live = row < group_end
                            if _row_live:
                                for _jv in cutlass.range_constexpr((_grp_frags * 8) // vsize):
                                    col = coord_n + warp_col * _WARP_TILE_N + _nf0 * 8 + _jv * vsize
                                    col_j = col
                                    if col_j + vsize <= N:
                                        # One vsize-wide epilogue vector per GEMM (GEMM 0 is
                                        # `vec_f32`; the injected bindings name the others).
                                        c_rmem_vecs = []
                                        for _g in cutlass.range_constexpr(num_gemms):
                                            _vec = _run_list[_g][_jv * vsize : vsize]
                                            if cutlass.const_expr(acc_widen_to_fp32):
                                                _pf = _vec.to(cutlass.Float32)
                                                c_rmem_vecs.append(_pf + cutlass.full_like(_pf, 0.0))
                                            else:
                                                c_rmem_vecs.append(_vec)
                                        vec_f32 = c_rmem_vecs[0]

                                        pass

                                        _c_0_a = (vec_f32).to(cutlass.Float32)
                                        _op_0 = _c_0_a * (cutlass.full_like(_c_0_a, _aux_alpha_pre.to(cutlass.Float32)))
                                        _r_0 = (_op_0).to(cutlass.BFloat16)
                                        _tap_0 = (_r_0).to(cutlass.BFloat16)
                                        (gC_tap_0_ptr + (row * out_stride_m_0 + col_j)).store(_tap_0, alignment=VEC_BYTES_TAP_0)
            red_part_flush_tile(_red_part, lane, M, N, (group_idx, group_begin, group_end), _red_part_args)

            # Next record.
            while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr.subview(sched_stage), sched_full_phase, time_limit=10_000_000):
                pass
            _slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
            tile_m = (_slot.subview(1)).load()
            tile_n = (_slot.subview(2)).load()
            is_valid = (_slot.subview(3)).load()
            group_begin = (_slot.subview(4)).load()
            group_end = (_slot.subview(5)).load()
            start_sf_block_m = (_slot.subview(6)).load()
            group_idx = (_slot.subview(7)).load()
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

        red_part_flush_cta(_red_part, lane, M, N, (group_idx, group_begin, group_end), _red_part_args)
        # No more tiles for this CTA: all its global A/B reads have been issued.
        if cutlass.const_expr(USE_PDL):
            if warp_idx == 0:
                if elect_one:
                    nvvm.griddepcontrol("launch_dependents")


frost_template_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    problem_size: tuple,
    first_token_offset: cute.Tensor,
    a_tma_workspace: cute.Tensor,
    a_0: cute.Tensor,
    b_0: cute.Tensor,
    sfa_0: cute.Tensor,
    sfb_0: cute.Tensor,
    c_tap_0: cute.Tensor,
    alpha: cute.Tensor,
    stream: _cuda.CUstream,
) -> None:
    _a_operands = [a_0]
    _b_operands = [b_0]
    _sfa_operands = [sfa_0]
    _sfb_operands = [sfb_0]

    m = problem_size[0]
    n = problem_size[1]
    k_sym = problem_size[2]
    num_experts = problem_size[3]
    num_groups = problem_size[4]
    _stride_idx = 5
    _a_stride_sets = []
    for _ in cutlass.range_constexpr(num_a_operands):
        _a_stride_sets.append(
            (
                problem_size[_stride_idx],
                problem_size[_stride_idx + 1],
                problem_size[_stride_idx + 2],
            )
        )
        _stride_idx += 3
    _b_stride_sets = []
    for _ in cutlass.range_constexpr(num_b_operands):
        _b_stride_sets.append(
            (
                problem_size[_stride_idx],
                problem_size[_stride_idx + 1],
                problem_size[_stride_idx + 2],
            )
        )
        _stride_idx += 3

    out_stride_m_0 = problem_size[_stride_idx]
    out_stride_n_0 = problem_size[_stride_idx + 1]
    out_stride_l_0 = problem_size[_stride_idx + 2]
    _stride_idx += 3

    # SF blobs: F8_128x4, i.e. [mn_block][k4][512 B]; viewed as fp16 so the
    # 512-byte atom is one 256-element TMA inner box.
    rest_k = ((k_sym // block_size) + 3) // 4
    # The segmented SFA blob holds, for S rows split across at most num_groups
    # non-empty groups, the graph-time worst case of independently 128-row
    # padded segments: min(S, G) + (S - min(S, G)) // 128 blocks
    # (fusion_ir.segmented_row_scale_capacity_rows). The compiler requires the
    # buffer to be exactly that large, so the descriptor's M extent IS the
    # buffer: a ragged-tail box past the last group is hardware zero-filled,
    # never an out-of-bounds read.
    sf_active = cutlass.min(m, num_groups)
    rest_m_sf = sf_active + (m - sf_active) // 128
    rest_n = (n + 127) // 128

    # ONE global A descriptor over the flat token matrix: K-major box
    # [K_tile, cta_m] on the NATIVE dtype (fp4 packs via B4X16) in A's swizzle
    # (s64b for a mixed pair's 64-byte e2m1 rows); the kernel offsets its row
    # coordinate per routed group. One global SFA descriptor over the whole
    # segmented blob; the kernel offsets its block coordinate. Strides arrive in
    # one-byte storage elements (a_fake_dtype). The data and SF lists are built
    # separately: a raw-FP8 token (one-sided dequant) has no SFA operand.
    # The A and B data descriptors carry CUTLASS's default TMA L2 promotion
    # (128 B). Their box rows are 64 B for the packed e2m1 side of a mixed MXFP4 x
    # MXFP8 pair (a 128-element fp8 K-tile holds 128 e2m1 values), and without the
    # hint each 64-byte row is its own DRAM request: on an RTX PRO 6000 (SM120) the
    # weight stream then tops out near 0.85 TB/s against 1.19 TB/s with it, which
    # was the whole gap between the W4A8 and NVFP4 kernels of one tile. 128-byte
    # rows (fp8, and e2m1 at the fp4 K-tile) measure the same with or without it.
    tma_a_desc_list = []
    for _a_idx, _a_op in enumerate(_a_operands):
        a_stride_m, a_stride_k, a_stride_l = _a_stride_sets[_a_idx]
        if cutlass.const_expr(moe_gather):
            a_dims, a_strides, a_box = [k_sym, _a_op.shape[0]], [a_stride_m * a_fake_dtype.width // 128], [cta_tile_mnk[2], 1]
        else:
            a_dims, a_strides, a_box = (
                [k_sym, m, 1],
                [a_stride_m * a_fake_dtype.width // 128, a_stride_l * a_fake_dtype.width // 128],
                [cta_tile_mnk[2], cta_tile_mnk[0], 1],
            )
        tma_a_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_a_op.iterator.toint(),
                dtype=a_tma_desc_dtype,
                global_dims=a_dims,
                global_strides=a_strides,
                box_dims=a_box,
                swizzle=a_tma_swizzle,
                tma_format=a_tma_format,
                l2_promotion=_tma.TensorMapL2Promotion.l2_128b,
            )
        )
    tma_sfa_desc_list = []
    for _sfa_op in _sfa_operands:
        if cutlass.const_expr(moe_gather):
            tma_sfa_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_sfa_op.iterator.toint(),
                    dtype=cutlass.Uint8,
                    global_dims=[k_sym // block_size, _sfa_op.shape[0]],
                    global_strides=[_sfa_op.stride[0] // 16],
                    box_dims=[16, 1],
                    swizzle=_tma.TensorMapSwizzle.none,
                )
            )
        else:
            sfa_fp16_tensor = cute.make_tensor(
                cute.recast_ptr(_sfa_op.iterator, dtype=cutlass.Float16),
                cute.make_layout(
                    (256, rest_k, rest_m_sf, 1),
                    stride=(
                        1,
                        256,
                        cute.assume(256 * rest_k, 8),
                        cute.assume(256 * rest_k * rest_m_sf, 8),
                    ),
                ),
            )
            tma_sfa_desc_list.append(
                _tma.create_tensor_map_tiled_from_view(
                    sfa_fp16_tensor,
                    dtype=cutlass.Uint16,
                    box_dims=(256, sf_tma_box_k, sfa_tma_box_mn, 1),
                    stride_order=(0, 1, 2, 3),
                    swizzle=_tma.TensorMapSwizzle.none,
                )
            )
    # B and SFB are batched by expert. K-major B: box [K_tile, cta_n]. N-major
    # (fp8): [group_elems, K_tile] boxes, one per N group. As for A, the SF list
    # is its own loop (a raw-FP8 weight has no SFB operand).
    tma_b_desc_list = []
    for _b_idx, _b_op in enumerate(_b_operands):
        b_stride_n, b_stride_k, b_stride_l = _b_stride_sets[_b_idx]
        if cutlass.const_expr(b_is_n_major):
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=b_tma_desc_dtype,
                    global_dims=[n, k_sym, num_experts],
                    global_strides=[
                        b_stride_k * b_fake_dtype.width // 128,
                        b_stride_l * b_fake_dtype.width // 128,
                    ],
                    box_dims=[b_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=b_tma_swizzle,
                    tma_format=b_tma_format,
                    l2_promotion=_tma.TensorMapL2Promotion.l2_128b,
                )
            )
        else:
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=b_tma_desc_dtype,
                    global_dims=[k_sym, n, num_experts],
                    global_strides=[
                        b_stride_n * b_fake_dtype.width // 128,
                        b_stride_l * b_fake_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[1], 1],
                    swizzle=b_tma_swizzle,
                    tma_format=b_tma_format,
                    l2_promotion=_tma.TensorMapL2Promotion.l2_128b,
                )
            )
    tma_sfb_desc_list = []
    for _sfb_op in _sfb_operands:
        sfb_fp16_tensor = cute.make_tensor(
            cute.recast_ptr(_sfb_op.iterator, dtype=cutlass.Float16),
            cute.make_layout(
                (256, rest_k, rest_n, num_experts),
                stride=(
                    1,
                    256,
                    cute.assume(256 * rest_k, 8),
                    cute.assume(_sfb_op.stride[2] // 2, 8),
                ),
            ),
        )
        tma_sfb_desc_list.append(
            _tma.create_tensor_map_tiled_from_view(
                sfb_fp16_tensor,
                dtype=cutlass.Uint16,
                box_dims=(256, sf_tma_box_k, sfb_tma_box_mn, 1),
                stride_order=(0, 1, 2, 3),
                swizzle=_tma.TensorMapSwizzle.none,
            )
        )

    # Persistent grid: as many CTAs as the device co-schedules; every CTA pulls
    # tiles off the global counter until the group space is exhausted. No
    # cluster launch on sm120 (CC 12.x has no thread-block clusters).
    grid_shape = (grid_num_clusters, 1, 1)
    # Zero the scheduler counter on the launch stream. The PDL main kernel
    # below reads it only after griddepcontrol.wait, i.e. once this has landed.
    counter_qword = grid_num_clusters * moe_desc_slots * TENSOR_MAP_QWORDS
    _dynamic_scheduler_counter_initialization(a_tma_workspace, cutlass.Int32(counter_qword)).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
    frost_template_kernel(
        problem_size[0],
        problem_size[1],
        problem_size[2],
        cutlass.Int32(num_experts),
        cutlass.Int32(num_groups),
        first_token_offset,
        a_tma_workspace,
        tma_a_desc_list[0],
        tma_b_desc_list[0],
        tma_sfa_desc_list[0],
        tma_sfb_desc_list[0],
        c_tap_0,
        out_stride_m_0,
        out_stride_n_0,
        out_stride_l_0,
        alpha,
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        use_pdl=USE_PDL,
        stream=stream,
        # .minnctapersm 1. With .reqntid alone ptxas cannot tell the register
        # count at entry, so it drops every setmaxnreg above (C7508): the compute
        # warps never get EPI_REG_COUNT, and ptxas may also size every warp for
        # several CTAs per SM (the dual MoE kernels fell to 40 registers). The
        # SMEM ring is sized to the per-CTA budget, so one CTA per SM is all that
        # fits anyway -- except the two smallest catalog tiles (32x64 / 64x32 at
        # a 32-byte K, whose ring caps at 16 stages), which now run one per SM
        # instead of two.
        min_blocks_per_mp=1,
    )


@lru_cache(maxsize=None)
def compile() -> Callable:
    out_vec_elems = vec_bytes_epi // (cd_dtype.width // 8)
    # Strides arrive in one-byte storage elements on either side (fp8, packed fp4 pairs).
    a_stride_elems = 128 // a_fake_dtype.width
    b_stride_elems = 128 // b_fake_dtype.width
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=out_vec_elems)
    # K tails are supported: the K loop is ceil_div and the TMA descriptor's global K
    # extent makes a partial box HW zero-filled. The only real K rule is the 16-byte
    # TMA contiguous-extent one, already gated by _tma_alignment_reject.
    sym_k = cute.sym_int64()
    # Packed K extent: same reasoning as sym_k -- no CTA-tile multiple is required.
    # One symbol for both operands (the call then checks token and weight agree on
    # K); a mixed pair packs K per side (fp8 K bytes against e2m1 K / 2).
    sym_kp = cute.sym_int64()
    sym_kp_b = sym_kp if (a_tma_format is None) == (b_tma_format is None) else cute.sym_int64()
    sym_e = cute.sym_int64()
    sym_g = cute.sym_int64()
    sym_source_m = cute.sym_int64() if moe_gather else sym_m

    def _make_fake_a():
        return make_fake_compact_tensor(
            a_fake_dtype,
            (sym_source_m, sym_kp, 1),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            b_fake_dtype,
            (sym_n, sym_kp_b, sym_e),
            stride_order=(0, 1, 2) if b_is_n_major else (1, 0, 2),
            assumed_align=16,
        )

    # The host rebuilds blocked row/K geometry from problem_size. Keep SF
    # layouts dynamic so SFB's runtime expert stride and GATHER token SF's
    # row shape/stride remain available without specializing the artifact.
    def _make_fake_sfa():
        return make_fake_tensor(
            sf_cutlass_dtype,
            (cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            assumed_align=16,
        )

    def _make_fake_sfb():
        return make_fake_tensor(
            sf_cutlass_dtype,
            (cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            assumed_align=16,
        )

    fake_first_token_offset = make_fake_compact_tensor(
        offset_cutlass_dtype,
        (cute.sym_int64(),),
        stride_order=(0,),
        assumed_align=offset_cutlass_dtype.width // 8,
    )
    # The compiler carves grid_ctas * moe_desc_slots tensormap slots plus one
    # counter slot (16 int64 each); with no descriptor slots that is the one
    # counter slot.
    fake_a_tma_workspace = make_fake_compact_tensor(
        cutlass.Int64,
        (grid_num_clusters * moe_desc_slots * TENSOR_MAP_QWORDS + TENSOR_MAP_QWORDS,),
        stride_order=(0,),
        assumed_align=128,
    )

    def _sym_operand_strides(is_mn_major: bool, stride_elems: int) -> tuple:
        # Operand is permuted to (M|N, K, L): the unit stride is mode 0 when MN-major, mode 1 when K-major, and never reaches TMA.
        unit = 0 if is_mn_major else 1
        return tuple(cute.sym_int64() if i == unit else cute.sym_int64(divisibility=stride_elems) for i in range(3))

    sym_a_strides = []
    for _ in range(num_a_operands):
        sym_a_strides.extend(_sym_operand_strides(False, a_stride_elems))
    sym_b_strides = []
    for _ in range(num_b_operands):
        sym_b_strides.extend(_sym_operand_strides(b_is_n_major, b_stride_elems))
    sym_out_stride_m_0 = cute.sym_int64()
    sym_out_stride_n_0 = cute.sym_int64()
    sym_out_stride_l_0 = cute.sym_int64()
    fake_a_0 = _make_fake_a()
    fake_b_0 = _make_fake_b()
    fake_sfa_0 = _make_fake_sfa()
    fake_sfb_0 = _make_fake_sfb()
    fake_c_tap_0 = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (sym_m, sym_n, 1),
        stride=(cute.sym_int64(), 1, cute.sym_int64()),
        assumed_align=16,
    )
    problem_size = (
        sym_m,
        sym_n,
        sym_k,
        sym_e,
        sym_g,
        *sym_a_strides,
        *sym_b_strides,
        sym_out_stride_m_0,
        sym_out_stride_n_0,
        sym_out_stride_l_0,
    )
    fake_alpha = cute.runtime.make_fake_tensor(cutlass.Float32, (sym_g, 1, 1), stride=(1, 1, 1), assumed_align=4)
    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    return cute.compile(_host, problem_size, fake_first_token_offset, fake_a_tma_workspace, fake_a_0, fake_b_0, fake_sfa_0, fake_sfb_0, fake_c_tap_0, fake_alpha, stream=_fake_stream, options=frost_compile_options)
