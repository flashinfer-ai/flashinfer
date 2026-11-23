# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Sm120 moe grouped block scale matmul fwd swap ab."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cutlass.experimental.primitives as nvvm
# Inlined from cudnn.gemm.frost.tile_helpers
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Tile-level helpers shared by the SM100 and SM120 kernel templates."""


import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cutlass._mlir_helpers.vector as _cvec
from cutlass._mlir.dialects import llvm






TENSOR_MAP_QWORDS = 16
CLUSTER_K_REDUCE_BARRIER_ID = 4






def moe_swizzle_tile(t, nt_m, nt_n, swizzle_w):
    """Group-local linear tile index -> (m, n) under an N-super-block walk."""
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
    src = cutlass.Int32(source_rows)
    if row < group_end:
        src = cutlass.Int32(token_index[row])
    return src
















@cute.jit
def tma_gather4(dst, desc, k, r0, r1, r2, r3, mbar, mask=None, cta_group: cutlass.Constexpr = 1):
    """Gather four rows from a rank-2 tensor map with box_dims[1] == 1."""
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

if b_is_n_major:
    raise NotImplementedError(f"{__name__}: the MoE token (B after swap-AB) is K-major only (the grouped token walk is a K-major TMA box)")
if moe_gather and num_sfb_operands and (cta_tile_mnk[1] * sf_tma_box_k) % 32:
    raise NotImplementedError(
        f"{__name__}: GATHER needs cta_tile_n * sf_k4 to be a multiple of 32 SF words "
        f"(got cta_tile_n={cta_tile_mnk[1]}, sf_k4={sf_tma_box_k}); pick a wider N tile"
    )

TENSOR_MAP_QWORDS = 16
moe_desc_slots = 0

# Scheduler ring depth and the i32 words per record.
SCHED_STAGES = 2
SCHED_SLOT_WORDS = 8

# Programmatic Dependent Launch (PDL, sm_90+; supported on sm_120).
USE_PDL = True

EPI_SYNC_BAR_ID = 1

WARPS_M = cta_tile_mnk[0] // (mma_size_m * mma_inst_shape_mnk[0])
WARPS_N = cta_tile_mnk[1] // (mma_size_n * mma_inst_shape_mnk[1])
NUM_COMPUTE_WARPS = WARPS_M * WARPS_N
# GATHER: the token tile's 4-row GATHER4 groups per TMA-warp lane.
_GATHER_LANE_GROUPS = (cta_tile_mnk[1] // 4 + 31) // 32

TMA_WARP_ID = NUM_COMPUTE_WARPS
SCHEDULER_WARP_ID = NUM_COMPUTE_WARPS + 1
NUM_WARPS = threads_per_cta // 32

NUM_SCHED_CONSUMER_WARPS = NUM_COMPUTE_WARPS + 1

EPI_REG_COUNT = 232
PROD_REG_COUNT = 24
assert NUM_COMPUTE_WARPS % 4 == 0, "the producer warps must form whole warpgroups"


_CTA_K_ELEMS = cta_tile_mnk[2]  # native K elements per tile
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
_A_FP4_TO_BYTES = mma_block_scale_kind == "MXF8F6F4" and mma_a_ptx_type == "e2m1"
_B_FP4_TO_BYTES = mma_block_scale_kind == "MXF8F6F4" and mma_b_ptx_type == "e2m1"
_A_FP4_PADDED = _A_FP4_TO_BYTES and a_tma_format is _tma.TensorMapDataFormat.B4X16_P64
_B_FP4_PADDED = _B_FP4_TO_BYTES and b_tma_format is _tma.TensorMapDataFormat.B4X16_P64
_A_FP4_PACKED = _A_FP4_TO_BYTES and not _A_FP4_PADDED
_B_FP4_PACKED = _B_FP4_TO_BYTES and not _B_FP4_PADDED
assert (_A_KBLK_BYTES, _B_KBLK_BYTES) == (16 if _A_FP4_PACKED else 32, 16 if _B_FP4_PACKED else 32), (_A_ROW_BYTES, _B_ROW_BYTES, _NUM_K_BLOCKS)
# A padded side: one-byte containers in 128-byte rows, half of them GMEM payload.
assert not _A_FP4_PADDED or (_A_ROW_BYTES == 128 and a_smem_dtype is cutlass.Uint8 and sA_tma_bytes * 2 == sA_packed_elems), (
    _A_ROW_BYTES,
    a_smem_dtype,
    sA_tma_bytes,
    sA_packed_elems,
)
assert not _B_FP4_PADDED or (_B_ROW_BYTES == 128 and b_smem_dtype is cutlass.Uint8 and sB_tma_bytes * 2 == sB_packed_elems), (
    _B_ROW_BYTES,
    b_smem_dtype,
    sB_tma_bytes,
    sB_packed_elems,
)

_WARP_TILE_M = cta_tile_mnk[0] // WARPS_M
_WARP_TILE_N = cta_tile_mnk[1] // WARPS_N
_M_FRAGS = _WARP_TILE_M // 16
_N_FRAGS = _WARP_TILE_N // 8
_N_FRAG_PAIRS = _N_FRAGS // 2
_ACC_REGS = _M_FRAGS * _N_FRAGS * 4

_A_SWIZZLE = cutlass.Swizzle((_A_ROW_BYTES // 16).bit_length() - 1, 4, 3)
_B_SWIZZLE = cutlass.Swizzle((_B_ROW_BYTES // 16).bit_length() - 1, 4, 3)
_TMA_SWIZZLE_BY_ROW = {128: _tma.TensorMapSwizzle.s128b, 64: _tma.TensorMapSwizzle.s64b}
assert (
    _TMA_SWIZZLE_BY_ROW.get(_A_ROW_BYTES) == a_tma_swizzle and _TMA_SWIZZLE_BY_ROW.get(_B_ROW_BYTES) == b_tma_swizzle
), "block-scale K rows are one swizzle span: 128 B, 64 B for a mixed pair's packed e2m1 side (validate_block_scale_config_sm120)"
assert not _A_FP4_PACKED or (_A_ROW_BYTES == 64 and sA_packed_elems % 512 == 0), (_A_ROW_BYTES, sA_packed_elems)
assert not _B_FP4_PACKED or (_B_ROW_BYTES == 64 and sB_packed_elems % 512 == 0), (_B_ROW_BYTES, sB_packed_elems)

_STG_EPI_LANE_QUAD = 4  # one STS.128 = 4 x 32-bit acc regs per lane
_STG_EPI_PAD = 4  # 16B skew after each 128-element batch (sheet's X cells)
_STG_EPI_BATCH_STRIDE = 32 * _STG_EPI_LANE_QUAD + _STG_EPI_PAD  # 132
_STG_EPI_GROUP_FRAGS = 4  # fragments (= STS batches) per 32-column group
_STG_EPI_WARP_ELEMS = _STG_EPI_GROUP_FRAGS * _STG_EPI_BATCH_STRIDE  # 528
_STG_EPI_NGRP = (_N_FRAGS + _STG_EPI_GROUP_FRAGS - 1) // _STG_EPI_GROUP_FRAGS
_STG_V = (vec_bytes_epi * 8) // cd_dtype.width

_RPL_SUBPASSES = 2 if _M_FRAGS % 2 == 0 else 1
_RPL_SLABS = _M_FRAGS // _RPL_SUBPASSES


def _rpl_swz(r):
    """Return the row-major staging swizzle key."""
    rr = r % 8
    return ((rr & 3) << 1) | (rr >> 2)


assert ab_stages >= 1, f"{__name__}: the renderer emitted an empty AB ring"


_SF_WORDS = sf_tma_box_k  # 32-bit SF words per row per K-tile (= sf_k4)
_SF_SPI = sf_scales_per_inst  # scales one MMA consumes along K: 1 (1X), 2 (2X), 4 (4X)
_SF_ATOM_WORDS = 128  # 512 B
_SF_BLOCK_WORDS = _SF_WORDS * _SF_ATOM_WORDS  # one 128-row block of the stage
_SFA_STAGE_WORDS = sfa_smem_bytes // 4
_SFB_STAGE_WORDS = sfb_smem_bytes // 4
assert _SF_WORDS * 4 == _NUM_K_BLOCKS * _SF_SPI, (_SF_WORDS, _NUM_K_BLOCKS, _SF_SPI)
_SF_WORD_BY_KBLK = tuple((j * _SF_SPI) // 4 for j in range(_NUM_K_BLOCKS))
_SF_BYTE_BY_KBLK = tuple((j * _SF_SPI) % 4 for j in range(_NUM_K_BLOCKS))
_SF_DUP = {"X1": 1, "X2": 2, "X4": 4}[mma_scale_vec_size] // _SF_SPI
assert _SF_DUP in (1, 2) and _SF_DUP * _SF_SPI == {"X1": 1, "X2": 2, "X4": 4}[mma_scale_vec_size], (mma_scale_vec_size, _SF_SPI)
_SF_MMA_BYTE_BY_KBLK = _SF_BYTE_BY_KBLK if _SF_DUP == 1 else (0,) * _NUM_K_BLOCKS
_SF_DUP_PRMT_BY_KBLK = tuple(b | (b << 4) | ((b + 1) << 8) | ((b + 1) << 12) for b in _SF_BYTE_BY_KBLK)
_SFA_GROUPS = -(-_M_FRAGS // 2)
_SFB_GROUPS = -(-_N_FRAGS // 4)
_M_BLOCK_ALIGNED = cta_tile_mnk[0] % 128 == 0
_N_BLOCK_ALIGNED = cta_tile_mnk[1] % 128 == 0


_MMA_KIND = getattr(_MMABlockScaleKind, mma_block_scale_kind)
_MMA_SCALE_VEC = getattr(_ScaleVecSize, mma_scale_vec_size)
_MMA_SF_FORMAT = getattr(_BlockScaleFormat, mma_sf_format)
_MMA_A_TYPE = getattr(_MMATypes, mma_a_ptx_type)
_MMA_B_TYPE = getattr(_MMATypes, mma_b_ptx_type)
_MMA_SHAPE_ATTR = f"#nvvm.shape<m = 16, n = 8, k = {_K_BLK_ELEMS}>"
assert mma_c_dtype == cutlass.Float32, "the block-scaled warp MMA accumulates in fp32"
_MMA_INLINE_PTX = mma_scale_vec_size == "X4" and mma_sf_format == "UE8M0"
_MMA_PTX = (
    f"mma.sync.aligned.m16n8k{_K_BLK_ELEMS}.row.col.kind::{mma_block_scale_kind.lower()}.block_scale"
    f".scale_vec::{mma_scale_vec_size[1:]}X.f32.{mma_a_ptx_type}.{mma_b_ptx_type}.f32.{mma_sf_format.lower()} "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13}, $14, {%d, %d}, $15, {%d, %d};"
)


def _iv(x):
    return x.ir_value() if hasattr(x, "ir_value") else x


def _mma_16x8_bs(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sfa, byte_id_a, thread_id_a, sfb, byte_id_b, thread_id_b):
    """One warp-wide block-scaled mma.sync on a (16, 8, 32-byte-K) fragment."""
    if _MMA_INLINE_PTX:
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
    """Select the scale word used by each MMA K block."""
    if _SF_DUP == 1:
        return [words[w] for w in _SF_WORD_BY_KBLK]
    return [nvvm.prmt(words[w], sel, nvvm.PermuteMode.DEFAULT, hi=0) for w, sel in zip(_SF_WORD_BY_KBLK, _SF_DUP_PRMT_BY_KBLK)]


_FP4_IN_BYTE = 0x3C3C3C3C  # bits [5:2] of every byte: where kind::mxf8f6f4 reads e2m1


def _fp4_frag_regs(words, sel):
    """Unpack an E2M1 operand fragment into MMA registers."""
    # Bytes (2t, 2t+1, 8+2t, 9+2t) of the chunk: the nibbles of K 4t..4t+3 and 16+4t..16+4t+3.
    z = nvvm.prmt(words.load(), sel, nvvm.PermuteMode.DEFAULT, hi=(words + 2).load())
    even = (z << 2) & _FP4_IN_BYTE  # low nibbles (K 4t, 4t+2, 16+4t, 18+4t) to bits [5:2]
    odd = (z >> 2) & _FP4_IN_BYTE  # high nibbles (K 4t+1, 4t+3, 17+4t, 19+4t) to bits [5:2]
    return (
        nvvm.prmt(even, 0x5140, nvvm.PermuteMode.DEFAULT, hi=odd),
        nvvm.prmt(even, 0x7362, nvvm.PermuteMode.DEFAULT, hi=odd),
    )


def _ldm_frags(ptr, num, padded):
    """Load byte-operand fragments with ldmatrix."""
    if padded:
        regs = nvvm.ldmatrix(ptr, num, nvvm.MMALayout.ROW, shape=nvvm.LoadShape.M8N16, src_format=nvvm.LoadSrcFormat.B4X16_P64)
        return tuple(regs[i] << 2 for i in range(num))
    regs = nvvm.ldmatrix(ptr, num, nvvm.MMALayout.ROW)
    return tuple(regs[i] for i in range(num))


def _sf_word_offset(r, r_in_block):
    """Return the scale-word offset within one shared-memory stage."""
    return (r_in_block // 128) * _SF_BLOCK_WORDS + (r_in_block % 32) * 4 + (r_in_block // 32) % 4


@cute.jit
def _moe_auto_swizzle_w(group_rows, n, k, nt_n):
    """N-super-block width for one routed group, resolved per group."""
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
    tma_workspace: cute.Tensor,
    tma_a_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_a_desc_1: cutlass.GridConstant[_tma.TensorMap],
    tma_b_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_sfa_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_sfa_desc_1: cutlass.GridConstant[_tma.TensorMap],
    tma_sfb_desc_0: cutlass.GridConstant[_tma.TensorMap],
    mC_tap_0: cute.Tensor,
    out_stride_m_0: cutlass.Int64,
    out_stride_n_0: cutlass.Int64,
    out_stride_l_0: cutlass.Int64,
    gate_alpha: cute.Tensor,
    up_alpha: cute.Tensor,
# @@FROST_ACTIVATION_0@@
    scale: cute.Tensor,
) -> None:
    tma_a_descs = [tma_a_desc_0, tma_a_desc_1]
    tma_b_descs = [tma_b_desc_0]
    tma_sfa_descs = [tma_sfa_desc_0, tma_sfa_desc_1]
    tma_sfb_descs = [tma_sfb_desc_0]

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    elect_one = nvvm.elect_sync()

    tidx = cute.arch.thread_idx()[0]

    sched_counter_ptr = cute.make_ptr(
        cutlass.Int32,
        (tma_workspace.iterator.raw_ptr() + grid_num_clusters * moe_desc_slots * TENSOR_MAP_QWORDS).toint(),
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

    smem_stg_epi = cutlass.Array(
        mma_c_dtype,
        _STG_EPI_WARP_ELEMS * NUM_COMPUTE_WARPS,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )

    sf_gather_scratch = [
        cutlass.Array(cutlass.Uint8, cta_tile_mnk[1] * 32, space=cutlass.AddressSpace.smem, alignment=128) for _ in range(num_sfb_operands if moe_gather else 0)
    ]
    sf_gather_mbar = [cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem) for _ in range(num_sfb_operands if moe_gather else 0)]
    if warp_idx == 0:
        for _bar in sf_gather_mbar:
            if elect_one:
                nvvm.mbarrier_init(_bar, 1)
        for i in range(ab_stages):
            if elect_one:
                nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1 + (num_sfb_operands if moe_gather else 0))
            if elect_one:
                nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), NUM_COMPUTE_WARPS)
        for i in range(SCHED_STAGES):
            if elect_one:
                nvvm.mbarrier_init(sched_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(sched_empty_mbar_ptr.subview(i), NUM_SCHED_CONSUMER_WARPS)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync(0)

    num_tma_copy_bytes = (
        num_a_operands * sA_tma_bytes
        + num_b_operands * sB_tma_bytes
        + num_sfa_operands * sfa_smem_bytes
        + (0 if moe_gather else num_sfb_operands * sfb_smem_bytes)
    )

    gC_tap_0_ptr = mC_tap_0.iterator.raw_ptr()
    VEC_BYTES_TAP_0 = vec_bytes_tap_0
    _red_part_args = ()

    VEC_BYTES = vec_bytes_epi
    vsize = epi_chunk_elems

    M = m
    N = n
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    tiles_along_m = cute.ceil_div(cutlass.Int32(M), cgrp_tile_mnk[0])
    first_token_arr = cutlass.make_array_view(first_token_offset)

    if warp_idx >= NUM_COMPUTE_WARPS:
        nvvm.setmaxregister(PROD_REG_COUNT, nvvm.SetMaxRegisterAction.DECREASE)

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
        start_sf_block_n = cutlass.Int32(0)
        scan_base = cutlass.Int32(0)
        group_idx = cutlass.Int32(0)
        group_begin = cutlass.Int32(0)
        group_end = cutlass.Int32(0)
        is_tile_valid = cutlass.Int32(1)

        while is_tile_valid != 0:
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
                        my_tiles = cute.ceil_div(my_end - my_begin, cgrp_tile_mnk[1]) * tiles_along_m
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
                    my_sf_start = start_sf_block_n + prefix_sf - my_sf_blocks
                    thread_succeed = nvvm.vote_sync(
                        full_warp_mask,
                        linear_idx < my_start + my_tiles,
                        nvvm.VoteSync.BALLOT,
                    )
                    if thread_succeed != 0:
                        # The lowest succeeding lane, found on the Uint32 ballot: on the Int32 one bfind is bfind.s32,
                        # which returns 0xFFFFFFFF for an all-lanes ballot (brev == -1), i.e. lane 32.
                        winning_lane = cutlass.Int32(31) - cute.arch.bfind(cute.arch.brev(cutlass.Uint32(thread_succeed))).to(cutlass.Int32)
                        scan_base = nvvm.shfl_sync(full_warp_mask, my_group, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        group_idx = scan_base
                        group_begin = nvvm.shfl_sync(full_warp_mask, my_begin, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        group_end = nvvm.shfl_sync(full_warp_mask, my_end, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        start_linear_idx = nvvm.shfl_sync(full_warp_mask, my_start, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        total_tiles = nvvm.shfl_sync(full_warp_mask, my_tiles, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        start_sf_block_n = nvvm.shfl_sync(full_warp_mask, my_sf_start, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        is_search_live = cutlass.Int32(0)
                    else:
                        start_linear_idx = nvvm.shfl_sync(
                            full_warp_mask,
                            my_start + my_tiles,
                            31,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        start_sf_block_n = nvvm.shfl_sync(
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
                group_nt_n = total_tiles // tiles_along_m
                tile_m, tile_n = _moe_swizzle_tile(
                    local_linear_idx,
                    tiles_along_m,
                    group_nt_n,
                    _moe_auto_swizzle_w(M, group_nt_n * cgrp_tile_mnk[1], k, group_nt_n),
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
                (slot.subview(6)).store(start_sf_block_n)
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
            start_sf_block_n = (slot.subview(6)).load()
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                coord_m = tile_m * cgrp_tile_mnk[0]
                coord_n_local = tile_n * cgrp_tile_mnk[1]
                coord_n = group_begin + coord_n_local
                sfa_m_block = coord_m // 128
                sfb_n_block = start_sf_block_n + coord_n_local // 128

                if cutlass.const_expr(moe_gather):
                    gather_lane = cute.arch.lane_idx()
                    gather_rows = []
                    for _lg in cutlass.range_constexpr(_GATHER_LANE_GROUPS):
                        _row = coord_n + (_lg * 32 + gather_lane) * 4
                        gather_rows.append([moe_gather_row(token_index, _row + _j, group_end, source_rows) for _j in range(4)])

                for k_tile_idx in range(num_k_tiles):
                    stage = ab_iter % ab_stages
                    if stage == 0 and ab_iter != 0:
                        ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(stage), ab_empty_phase_bit, time_limit=10_000_000):
                        pass

                    coord_k = k_tile_idx * cta_tile_mnk[2]
                    coord_sf_k = k_tile_idx * sf_tma_box_k
                    if elect_one:
                        nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)
                    if cutlass.const_expr(moe_gather):
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        if cutlass.const_expr(a_is_m_major):
                            for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cta_global(
                                        smem_a_list[_ai].subview(sA_elems * stage + m_group * a_tma_group_elems * _A_ROW_BYTES),
                                        tma_a_descs[_ai].get_ptr(),
                                        (coord_m + m_group * a_tma_group_elems, coord_k, coord_expert),
                                        ab_full_mbar_ptr.subview(stage),
                                    )
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cta_global(
                                    smem_a_list[_ai].subview(sA_elems * stage),
                                    tma_a_descs[_ai].get_ptr(),
                                    (coord_k, coord_m, coord_expert),
                                    ab_full_mbar_ptr.subview(stage),
                                )
                        if cutlass.const_expr(not fake_dequant_a):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cta_global(
                                    smem_sfa_list[_ai].subview(_SFA_STAGE_WORDS * stage),
                                    tma_sfa_descs[_ai].get_ptr(),
                                    (0, coord_sf_k, sfa_m_block, coord_expert),
                                    ab_full_mbar_ptr.subview(stage),
                                )
                    for _bj in cutlass.range_constexpr(num_b_operands):
                        if cutlass.const_expr(moe_gather):
                            for _lg in cutlass.range_constexpr(_GATHER_LANE_GROUPS):
                                _gr = _lg * 32 + gather_lane
                                if _gr < cta_tile_mnk[1] // 4:
                                    tma_gather4(
                                        smem_b_list[_bj].subview(sB_elems * stage + _gr * 4 * _B_ROW_BYTES),
                                        tma_b_descs[_bj].get_ptr(),
                                        coord_k,
                                        gather_rows[_lg][0],
                                        gather_rows[_lg][1],
                                        gather_rows[_lg][2],
                                        gather_rows[_lg][3],
                                        ab_full_mbar_ptr.subview(stage),
                                    )
                            if cutlass.const_expr(not fake_dequant_b):
                                if elect_one:
                                    nvvm.mbarrier_arrive_expect_tx(sf_gather_mbar[_bj], cta_tile_mnk[1] * 16)
                                nvvm.bar_warp_sync(0xFFFFFFFF)
                                scale_k = coord_k // block_size
                                for _lg in cutlass.range_constexpr(_GATHER_LANE_GROUPS):
                                    _gr = _lg * 32 + gather_lane
                                    if _gr < cta_tile_mnk[1] // 4:
                                        tma_gather4(
                                            sf_gather_scratch[_bj].subview(_gr * 128),
                                            tma_sfb_descs[_bj].get_ptr(),
                                            (scale_k // 16) * 16,
                                            gather_rows[_lg][0],
                                            gather_rows[_lg][1],
                                            gather_rows[_lg][2],
                                            gather_rows[_lg][3],
                                            sf_gather_mbar[_bj],
                                        )
                                while not nvvm.mbarrier_try_wait_parity(sf_gather_mbar[_bj], ab_iter % 2, time_limit=10_000_000):
                                    pass
                                sf_src_words = cutlass.inttoptr(sf_gather_scratch[_bj].data_ptr().toint(), 3, cutlass.Uint32)
                                sf_dst_words = cutlass.inttoptr(smem_sfb_list[_bj].subview(_SFB_STAGE_WORDS * stage).data_ptr().toint(), 3, cutlass.Uint32)
                                for _i in cutlass.range(cta_tile_mnk[1] * (cta_tile_mnk[2] // block_size // 4) // 32, unroll_full=True):
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
                                    smem_b_list[_bj].subview(sB_elems * stage),
                                    tma_b_descs[_bj].get_ptr(),
                                    (coord_k, coord_n, cutlass.Int32(0)),
                                    ab_full_mbar_ptr.subview(stage),
                                )
                            if cutlass.const_expr(not fake_dequant_b):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cta_global(
                                        smem_sfb_list[_bj].subview(_SFB_STAGE_WORDS * stage),
                                        tma_sfb_descs[_bj].get_ptr(),
                                        (0, coord_sf_k, sfb_n_block, cutlass.Int32(0)),
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

        a_ldm_row = (lane % 8) + 8 * ((lane // 8) % 2)
        a_ldm_col16 = lane // 16
        b_ldm_pair_row = (lane % 8) + 8 * (lane // 16)
        b_ldm_pair_col16 = (lane // 8) % 2
        # B x2 tail: one n-frag (rows 0-7 x two 16B cols; lanes 16-31 unused).
        b_ldm_tail_row = lane % 8
        b_ldm_tail_col16 = (lane // 8) % 2
        fp4_lane_words = [
            lane_div4 * 16 + ((cutlass.Int32(kb * 16) ^ (((lane_div4 // 2) % 4) * 16)) + (lane_mod4 // 2) * 4) // 4 for kb in range(_NUM_K_BLOCKS)
        ]
        fp4_sel = cutlass.Int32(0x5410) + (lane_mod4 % 2) * 0x2222

        sf_lane_row = lane_div4 + 8 * (lane % 2)
        sfa_lane_sel = (lane // 2) % 2
        sfb_lane_sel = lane_mod4
        sf_one = cutlass.Int32(sf_one_word)
        # A row's ceil(K / block_size) K-scales end in its F8_128x4 word sf_tail_word. Unless that count is
        # a multiple of 4, the word's last 1-3 bytes are blob padding: the SF box still loads them (TMA
        # zero-fills only the words past rest_k), and the MMAs of the zero-filled K blocks behind them use
        # them as scales, so a NaN pad (UE8M0 0xFF, UE4M3 0x7F) would make a whole output row or column
        # NaN. sf_tail_keep is the AND mask that keeps only that word's live bytes.
        sf_last = cutlass.Int32((k + (block_size - 1)) // block_size) - 1
        sf_tail_word = sf_last >> 2
        sf_tail_keep = ~(cutlass.Int32(-256) << ((sf_last & 3) * 8))

        acc_list = [cutlass.Array(mma_c_dtype, _ACC_REGS, alignment=16) for _g in range(num_gemms)]

        ab_full_phase_bit = cutlass.Int32(0)
        ab_iter = cutlass.Int32(0)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)
        # The routed output is one flat (N, S) view of the (S, N) buffer: no batch term.
        tile_l = cutlass.Int32(0)

        while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr.subview(sched_stage), sched_full_phase, time_limit=10_000_000):
            pass
        _slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
        tile_m = (_slot.subview(1)).load()
        tile_n = (_slot.subview(2)).load()
        is_valid = (_slot.subview(3)).load()
        group_begin = (_slot.subview(4)).load()
        group_end = (_slot.subview(5)).load()
        start_sf_block_n = (_slot.subview(6)).load()
        group_idx = (_slot.subview(7)).load()
        nvvm.bar_warp_sync(0xFFFFFFFF)
        if elect_one:
            nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
        sched_stage += 1
        if sched_stage == SCHED_STAGES:
            sched_stage = cutlass.Int32(0)
            sched_full_phase = sched_full_phase ^ 1

        # Routed rows the groups cover (the final offset, at most the capacity N): the count an
        # ungrouped AVG over tokens divides by. At least 1, so an empty routing flushes 0, not NaN.
        moe_used_rows = cutlass.max(cutlass.min(cutlass.Int64(first_token_arr[num_groups]), N), cutlass.Int64(1))
        _red_part = cute.make_rmem_tensor(red_part_slots, cutlass.Float32)
        red_part_seed(_red_part)
        while is_valid != 0:
            coord_m = tile_m * cgrp_tile_mnk[0]
            coord_n_local = tile_n * cgrp_tile_mnk[1]
            coord_n = group_begin + coord_n_local

            if cutlass.const_expr(_M_BLOCK_ALIGNED):
                m_in_block = cutlass.Int32(0)
            else:
                m_in_block = coord_m % 128
            if cutlass.const_expr(_N_BLOCK_ALIGNED or moe_gather):
                n_in_block = cutlass.Int32(0)
            else:
                n_in_block = coord_n_local % 128
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

                # sf_tail_keep on the word that holds a row's last K-scale, all ones on every other word.
                _sf_keep = [
                    cutlass.Int32(cutlass.select_(cutlass.Int32(k_tile_idx) * _SF_WORDS + w == sf_tail_word, sf_tail_keep, cutlass.Int32(-1)))
                    for w in range(_SF_WORDS)
                ]
                if cutlass.const_expr(fake_dequant_a):
                    sfa_words_list = [[[sf_one] * _NUM_K_BLOCKS for _g in range(_SFA_GROUPS)] for _ai in range(num_a_operands)]
                else:
                    sfa_words_list = []
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        sfa_words = []
                        for g in cutlass.range_constexpr(_SFA_GROUPS):
                            _ws = []
                            for w in cutlass.range_constexpr(_SF_WORDS):
                                _ws.append((sSFA_ptrs[_ai] + sfa_word_off[g] + w * _SF_ATOM_WORDS).load() & _sf_keep[w])
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
                                _ws.append((sSFB_ptrs[_bj] + sfb_word_off[g] + w * _SF_ATOM_WORDS).load() & _sf_keep[w])
                            sfb_words.append(_sf_kblk_words(_ws))
                        sfb_words_list.append(sfb_words)

                for k_blk in cutlass.range_constexpr(_NUM_K_BLOCKS):
                    _sf_byte = _SF_MMA_BYTE_BY_KBLK[k_blk]
                    a_frags_list = []
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        sA_ptr = sA_ptrs[_ai]
                        a_frags = []
                        if cutlass.const_expr(a_is_m_major):
                            for mf in cutlass.range_constexpr(_M_FRAGS):
                                a_m = warp_row * _WARP_TILE_M + mf * 16
                                a_off = (
                                    (a_m // a_tma_group_elems) * (a_tma_group_elems * _A_ROW_BYTES)
                                    + (k_blk * _A_KBLK_BYTES + lane) * a_tma_group_elems
                                    + a_m % a_tma_group_elems
                                )
                                a_frags.append(
                                    nvvm.ldmatrix(
                                        _apply_smem_swizzle(sA_ptr + a_off, _A_SWIZZLE),
                                        4,
                                        nvvm.MMALayout.COL,
                                        shape=nvvm.LoadShape.M16N16,
                                        src_format=nvvm.LoadSrcFormat.B8,
                                    )
                                )
                        elif cutlass.const_expr(_A_FP4_PACKED):
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
                                a_frags.append(_ldm_frags(_apply_smem_swizzle(sA_ptr + a_off, _A_SWIZZLE), 4, _A_FP4_PADDED))
                        a_frags_list.append(a_frags)
                    for _bj in cutlass.range_constexpr(num_b_operands):
                        sB_ptr = sB_ptrs[_bj]
                        b_frags = []
                        if cutlass.const_expr(_B_FP4_PACKED):
                            sB_words = cutlass.inttoptr(sB_ptr.toint(), 3, cutlass.Uint32)
                            for nf in cutlass.range_constexpr(_N_FRAGS):
                                b_frags.append(
                                    _fp4_frag_regs(sB_words + (warp_col * _WARP_TILE_N + nf * 8) * (_B_ROW_BYTES // 4) + fp4_lane_words[k_blk], fp4_sel)
                                )
                        else:
                            for npair in cutlass.range_constexpr(_N_FRAG_PAIRS):
                                b_row = warp_col * _WARP_TILE_N + npair * 16 + b_ldm_pair_row
                                b_off = b_row * _B_ROW_BYTES + k_blk * _B_KBLK_BYTES + b_ldm_pair_col16 * _ELEMS_16B
                                bv = _ldm_frags(_apply_smem_swizzle(sB_ptr + b_off, _B_SWIZZLE), 4, _B_FP4_PADDED)
                                b_frags.append((bv[0], bv[1]))
                                b_frags.append((bv[2], bv[3]))
                            if cutlass.const_expr(_N_FRAGS % 2 == 1):
                                b_row = warp_col * _WARP_TILE_N + (_N_FRAGS - 1) * 8 + b_ldm_tail_row
                                b_off = b_row * _B_ROW_BYTES + k_blk * _B_KBLK_BYTES + b_ldm_tail_col16 * _ELEMS_16B
                                bt = _ldm_frags(_apply_smem_swizzle(sB_ptr + b_off, _B_SWIZZLE), 2, _B_FP4_PADDED)
                                b_frags.append((bt[0], bt[1]))

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

            _stg_stage = smem_stg_epi.subview(warp_idx * _STG_EPI_WARP_ELEMS)
            if cutlass.const_expr(not row_per_lane_drain):
                for mf in cutlass.range_constexpr(_M_FRAGS):
                    for grp in cutlass.range_constexpr(_STG_EPI_NGRP):
                        _nf0 = grp * _STG_EPI_GROUP_FRAGS
                        _grp_frags = min(_STG_EPI_GROUP_FRAGS, _N_FRAGS - _nf0)
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
                        _lane_active = True if _grp_frags == _STG_EPI_GROUP_FRAGS else lane_mod4 < _grp_frags
                        if _lane_active:
                            for half in cutlass.range_constexpr(2):
                                row_in_cta = warp_row * _WARP_TILE_M + mf * 16 + half * 8 + lane_div4
                                row = coord_m + row_in_cta
                                # Weight rows past M are the TMA zero-fill of the last M tile.
                                if row < M:
                                    _aux_gate_alpha_ptr = gate_alpha.iterator.raw_ptr()
                                    _aux_gate_alpha_pre = (_aux_gate_alpha_ptr + group_idx).load()
                                    _aux_up_alpha_ptr = up_alpha.iterator.raw_ptr()
                                    _aux_up_alpha_pre = (_aux_up_alpha_ptr + group_idx).load()
# @@FROST_ACTIVATION_1@@
                                    _aux_scale_ptr = scale.iterator.raw_ptr()
                                    _aux_scale_pre = (_aux_scale_ptr + 0).load()
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
                                        if col_j < group_end:
                                            c_rmem_vecs = []
                                            for _g in cutlass.range_constexpr(num_gemms):
                                                _vec = _row_list[_g][sv * _STG_V : _STG_V]
                                                if cutlass.const_expr(acc_widen_to_fp32):
                                                    _pf = _vec.to(cutlass.Float32)
                                                    c_rmem_vecs.append(_pf + cutlass.full_like(_pf, 0.0))
                                                else:
                                                    c_rmem_vecs.append(_vec)
                                            vec_f32 = c_rmem_vecs[0]

                                            vec_f32_1 = c_rmem_vecs[1]

                                            _c_0_a = (vec_f32).to(cutlass.Float32)
                                            _op_0 = _c_0_a * (cutlass.full_like(_c_0_a, _aux_gate_alpha_pre.to(cutlass.Float32)))
                                            _c_1_a = (vec_f32_1).to(cutlass.Float32)
                                            _op_1 = _c_1_a * (cutlass.full_like(_c_1_a, _aux_up_alpha_pre.to(cutlass.Float32)))
                                            _c_2_a = (_op_0).to(cutlass.Float32)
# @@FROST_ACTIVATION_2@@
                                            _c_3_a = (_op_2).to(cutlass.Float32)
# @@FROST_ACTIVATION_3@@
                                            if (row < M) & (col_j + 0 < group_end):
                                                (gC_tap_0_ptr + row * out_stride_m_0 + (col_j + 0) * out_stride_n_0).store(_tap_0[0 : 0 + 1], alignment=2)
                        nvvm.bar_warp_sync(0xFFFFFFFF)
            else:
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
                            if row < M:
                                _aux_gate_alpha_ptr = gate_alpha.iterator.raw_ptr()
                                _aux_gate_alpha_pre = (_aux_gate_alpha_ptr + group_idx).load()
                                _aux_up_alpha_ptr = up_alpha.iterator.raw_ptr()
                                _aux_up_alpha_pre = (_aux_up_alpha_ptr + group_idx).load()
# @@FROST_ACTIVATION_4@@
                                _aux_scale_ptr = scale.iterator.raw_ptr()
                                _aux_scale_pre = (_aux_scale_ptr + 0).load()
                                for _jv in cutlass.range_constexpr((_grp_frags * 8) // vsize):
                                    col = coord_n + warp_col * _WARP_TILE_N + _nf0 * 8 + _jv * vsize
                                    col_j = col
                                    if col_j < group_end:
                                        c_rmem_vecs = []
                                        for _g in cutlass.range_constexpr(num_gemms):
                                            _vec = _run_list[_g][_jv * vsize : vsize]
                                            if cutlass.const_expr(acc_widen_to_fp32):
                                                _pf = _vec.to(cutlass.Float32)
                                                c_rmem_vecs.append(_pf + cutlass.full_like(_pf, 0.0))
                                            else:
                                                c_rmem_vecs.append(_vec)
                                        vec_f32 = c_rmem_vecs[0]

                                        vec_f32_1 = c_rmem_vecs[1]

                                        _c_0_a = (vec_f32).to(cutlass.Float32)
                                        _op_0 = _c_0_a * (cutlass.full_like(_c_0_a, _aux_gate_alpha_pre.to(cutlass.Float32)))
                                        _c_1_a = (vec_f32_1).to(cutlass.Float32)
                                        _op_1 = _c_1_a * (cutlass.full_like(_c_1_a, _aux_up_alpha_pre.to(cutlass.Float32)))
                                        _c_2_a = (_op_0).to(cutlass.Float32)
# @@FROST_ACTIVATION_5@@
                                        _c_3_a = (_op_2).to(cutlass.Float32)
# @@FROST_ACTIVATION_6@@
                                        if (row < M) & (col_j + 0 < group_end):
                                            (gC_tap_0_ptr + row * out_stride_m_0 + (col_j + 0) * out_stride_n_0).store(_tap_0[0 : 0 + 1], alignment=2)
            red_part_flush_tile(_red_part, lane, M, N, (group_idx, group_begin, group_end, moe_used_rows), _red_part_args)

            # Next record.
            while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr.subview(sched_stage), sched_full_phase, time_limit=10_000_000):
                pass
            _slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
            tile_m = (_slot.subview(1)).load()
            tile_n = (_slot.subview(2)).load()
            is_valid = (_slot.subview(3)).load()
            group_begin = (_slot.subview(4)).load()
            group_end = (_slot.subview(5)).load()
            start_sf_block_n = (_slot.subview(6)).load()
            group_idx = (_slot.subview(7)).load()
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

        red_part_flush_cta(_red_part, lane, M, N, (group_idx, group_begin, group_end, moe_used_rows), _red_part_args)
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
    tma_workspace: cute.Tensor,
    a_0: cute.Tensor,
    a_1: cute.Tensor,
    b_0: cute.Tensor,
    sfa_0: cute.Tensor,
    sfa_1: cute.Tensor,
    sfb_0: cute.Tensor,
    c_tap_0: cute.Tensor,
    gate_alpha: cute.Tensor,
    up_alpha: cute.Tensor,
# @@FROST_ACTIVATION_7@@
    scale: cute.Tensor,
    stream: _cuda.CUstream,
) -> None:
    _a_operands = [a_0, a_1]
    _b_operands = [b_0]
    _sfa_operands = [sfa_0, sfa_1]
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

    rest_k = ((k_sym // block_size) + 3) // 4
    # SFA is the weight's per-expert blob: the expert's M rows in whole 128-row blocks.
    rest_m = (m + 127) // 128
    sf_active = cutlass.min(n, num_groups)
    rest_n_sf = sf_active + (n - sf_active) // 128

    tma_a_desc_list = []
    for _a_idx, _a_op in enumerate(_a_operands):
        a_stride_m, a_stride_k, a_stride_l = _a_stride_sets[_a_idx]
        if cutlass.const_expr(a_is_m_major):
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=a_tma_desc_dtype,
                    global_dims=[m, k_sym, num_experts],
                    global_strides=[
                        a_stride_k * a_fake_dtype.width // 128,
                        a_stride_l * a_fake_dtype.width // 128,
                    ],
                    box_dims=[a_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=a_tma_swizzle,
                    tma_format=a_tma_format,
                    l2_promotion=_tma.TensorMapL2Promotion.l2_128b,
                )
            )
        else:
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=a_tma_desc_dtype,
                    global_dims=[k_sym, m, num_experts],
                    global_strides=[
                        a_stride_m * a_fake_dtype.width // 128,
                        a_stride_l * a_fake_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[0], 1],
                    swizzle=a_tma_swizzle,
                    tma_format=a_tma_format,
                    l2_promotion=_tma.TensorMapL2Promotion.l2_128b,
                )
            )
    tma_sfa_desc_list = []
    for _sfa_op in _sfa_operands:
        sfa_fp16_tensor = cute.make_tensor(
            cute.recast_ptr(_sfa_op.iterator, dtype=cutlass.Float16),
            cute.make_layout(
                (256, rest_k, rest_m, num_experts),
                stride=(
                    1,
                    256,
                    cute.assume(256 * rest_k, 8),
                    cute.assume(_sfa_op.stride[2] // 2, 8),
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
    tma_b_desc_list = []
    for _b_idx, _b_op in enumerate(_b_operands):
        b_stride_n, b_stride_k, b_stride_l = _b_stride_sets[_b_idx]
        if cutlass.const_expr(moe_gather):
            b_dims, b_strides, b_box = [k_sym, _b_op.shape[0]], [b_stride_n * b_fake_dtype.width // 128], [cta_tile_mnk[2], 1]
        else:
            b_dims, b_strides, b_box = (
                [k_sym, n, 1],
                [b_stride_n * b_fake_dtype.width // 128, b_stride_l * b_fake_dtype.width // 128],
                [cta_tile_mnk[2], cta_tile_mnk[1], 1],
            )
        tma_b_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_b_op.iterator.toint(),
                dtype=b_tma_desc_dtype,
                global_dims=b_dims,
                global_strides=b_strides,
                box_dims=b_box,
                swizzle=b_tma_swizzle,
                tma_format=b_tma_format,
                l2_promotion=_tma.TensorMapL2Promotion.l2_128b,
            )
        )
    # The SF list is its own loop: a raw-FP8 token (one-sided dequant) has no SFB operand.
    tma_sfb_desc_list = []
    for _sfb_op in _sfb_operands:
        if cutlass.const_expr(moe_gather):
            tma_sfb_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_sfb_op.iterator.toint(),
                    dtype=cutlass.Uint8,
                    global_dims=[k_sym // block_size, _sfb_op.shape[0]],
                    global_strides=[_sfb_op.stride[0] // 16],
                    box_dims=[16, 1],
                    swizzle=_tma.TensorMapSwizzle.none,
                )
            )
        else:
            sfb_fp16_tensor = cute.make_tensor(
                cute.recast_ptr(_sfb_op.iterator, dtype=cutlass.Float16),
                cute.make_layout(
                    (256, rest_k, rest_n_sf, 1),
                    stride=(
                        1,
                        256,
                        cute.assume(256 * rest_k, 8),
                        cute.assume(256 * rest_k * rest_n_sf, 8),
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

    grid_shape = (grid_num_clusters, 1, 1)
    # A launch started at zero leaves the scheduler counter (the int32 at counter_qword) at its tile count + grid_num_clusters (one failing claim per CTA).
    # With external_counter_reset True the runtime must zero it before every launch, visible at griddepcontrol.wait (which waits only for the previous grid);
    # one zero for FC1 and FC2 on one workspace makes FC2 silently skip its tiles below FC1's final count.
    counter_qword = grid_num_clusters * moe_desc_slots * TENSOR_MAP_QWORDS
    if cutlass.const_expr(not external_counter_reset):
        _dynamic_scheduler_counter_initialization(tma_workspace, cutlass.Int32(counter_qword)).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
    frost_template_kernel(
        problem_size[0],
        problem_size[1],
        problem_size[2],
        cutlass.Int32(num_experts),
        cutlass.Int32(num_groups),
        first_token_offset,
        tma_workspace,
        tma_a_desc_list[0],
        tma_a_desc_list[1],
        tma_b_desc_list[0],
        tma_sfa_desc_list[0],
        tma_sfa_desc_list[1],
        tma_sfb_desc_list[0],
        c_tap_0,
        out_stride_m_0,
        out_stride_n_0,
        out_stride_l_0,
        gate_alpha,
        up_alpha,
# @@FROST_ACTIVATION_8@@
        scale,
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        use_pdl=USE_PDL,
        stream=stream,
        min_blocks_per_mp=1,
    )


@lru_cache(maxsize=None)
def compile() -> Callable:
    out_vec_elems = vec_bytes_epi // (cd_dtype.width // 8)
    # Strides arrive in one-byte storage elements on either side (fp8, packed fp4 pairs).
    a_stride_elems = 128 // a_fake_dtype.width
    b_stride_elems = 128 // b_fake_dtype.width
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=min(out_vec_elems, moe_token_alignment))
    sym_k = cute.sym_int64()
    sym_kp = cute.sym_int64()
    sym_kp_b = sym_kp if (a_tma_format is None) == (b_tma_format is None) else cute.sym_int64()
    sym_e = cute.sym_int64()
    sym_g = cute.sym_int64()
    sym_source_n = cute.sym_int64() if moe_gather else sym_n

    def _make_fake_a():
        return make_fake_compact_tensor(
            a_fake_dtype,
            (sym_m, sym_kp, sym_e),
            stride_order=(0, 1, 2) if a_is_m_major else (1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            b_fake_dtype,
            (sym_source_n, sym_kp_b, 1),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )

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
    fake_tma_workspace = make_fake_compact_tensor(
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
        sym_a_strides.extend(_sym_operand_strides(a_is_m_major, a_stride_elems))
    sym_b_strides = []
    for _ in range(num_b_operands):
        sym_b_strides.extend(_sym_operand_strides(False, b_stride_elems))
    sym_out_stride_m_0 = cute.sym_int64()
    sym_out_stride_n_0 = cute.sym_int64()
    sym_out_stride_l_0 = cute.sym_int64()
    fake_a_0 = _make_fake_a()
    fake_a_1 = _make_fake_a()
    fake_b_0 = _make_fake_b()
    fake_sfa_0 = _make_fake_sfa()
    fake_sfa_1 = _make_fake_sfa()
    fake_sfb_0 = _make_fake_sfb()
    fake_c_tap_0 = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (sym_m, sym_n, 1),
        stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
        assumed_align=2,
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
    fake_gate_alpha = cute.runtime.make_fake_tensor(cutlass.Float32, (sym_g, 1, 1), stride=(1, 1, 1), assumed_align=4)
    fake_up_alpha = cute.runtime.make_fake_tensor(cutlass.Float32, (sym_g, 1, 1), stride=(1, 1, 1), assumed_align=4)
# @@FROST_ACTIVATION_9@@
    fake_scale = cute.runtime.make_fake_tensor(cutlass.Float32, (1, 1, 1), stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()), assumed_align=4)
    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
# @@FROST_ACTIVATION_10@@
