# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/paged/forward_extend_generic.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Historical generic paged extend kernels kept in a separate file.

This uses the exact host planner worklists with the
literal tensor-core inner path that we actually ship:
- staged paged K/V ingress,
- literal QK/PV MMA for BF16 and FP8 KV,
- base-2 LSE storage for direct extend output.
"""

from __future__ import annotations
import os
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.core import make_swizzle
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic

from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cutlass_dsl import Int64, T, dsl_user_op
from flashinfer.experimental.sm12x.attention._shared.cute import copy as cute_copy
from flashinfer.experimental.sm12x.attention._shared.cute import (
    pipeline as cute_pipeline,
)
from flashinfer.experimental.sm12x.attention._shared.cute import ops as attention_ops
from flashinfer.experimental.sm12x._lib.intrinsics import (
    get_ptr_as_int64,
    shared_ptr_to_u32,
)
from flashinfer.experimental.sm12x._lib.smem import make_smem_memrange_alias
from flashinfer.experimental.sm12x._lib.intrinsics import (
    bf16_mma_m16n16k16_f32,
    bf16_rowsum_m16k16_f32,
    bfloat2_mul,
    bfloat2_to_float2_scaled,
    broadcast_f32_to_bfloat2,
    cvt_bf16x2_to_e4m3x2,
    fp8x4_e4m3_to_bfloat2x2,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x4_left_half_b16,
    ldmatrix_m8n8x4_right_half_b16,
    ldmatrix_m8n8x4_trans_b16,
    ldmatrix_m8n8x4_trans_left_half_b16,
    ldmatrix_m8n8x4_trans_right_half_b16,
    mxfp8_mma_m16n8k32_f32_e4m3,
    pack_f32x2_to_bfloat2,
    frag_layout_swizzle_16b_to_8b,
    frag_layout_swizzle_16b_to_8b_trans,
    st_global_v4_u32,
    st_shared_v4_u32,
)

from .traits import PagedForwardTraits


def _assume_strides_aligned(t: cute.Tensor):
    divby = 128 // t.element_type.width
    strides = tuple(
        s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1]
    )
    return (*strides, t.stride[-1])


def _assume_tensor_aligned(t: cute.Tensor | None):
    if t is None:
        return None
    return cute.make_tensor(
        t.iterator, cute.make_layout(t.shape, stride=_assume_strides_aligned(t))
    )


def _assume_paged_kv_tma_source_aligned(t: cute.Tensor):
    divby = 128 // t.element_type.width
    strides = []
    for dim, stride in enumerate(t.stride):
        if dim == 1 or isinstance(stride, int):
            strides.append(stride)
        else:
            strides.append(cute.assume(stride, divby=divby))
    return cute.make_tensor(
        t.iterator, cute.make_layout(t.shape, stride=tuple(strides))
    )


def _make_payload_ptr(payload_u8: cute.Tensor, dtype, offset_bytes: int = 0):
    # Preserve 128-bit shared alignment when carving typed aliases out of the
    # byte payload.
    ptr = (
        payload_u8.iterator if offset_bytes == 0 else payload_u8.iterator + offset_bytes
    )
    return cute.recast_ptr(ptr.align(16), dtype=dtype)


def _make_payload_tensor(payload_u8: cute.Tensor, dtype, offset_bytes: int, layout):
    return cute.make_tensor(_make_payload_ptr(payload_u8, dtype, offset_bytes), layout)


def _make_payload_memrange(
    payload_u8: cute.Tensor, dtype, offset_bytes: int, num_elems: int
):
    # Rebuild a MemRange alias over the payload slice so CuTe can lower swizzled
    # shared-memory pointers the same way it does for typed struct fields.
    return make_smem_memrange_alias(
        dtype,
        num_elems,
        _make_payload_ptr(payload_u8, dtype, offset_bytes),
    )


def _get_memrange_tensor(memrange, layout):
    if hasattr(layout, "outer") and hasattr(layout, "inner"):
        return memrange.get_tensor(layout.outer, swizzle=layout.inner)
    return memrange.get_tensor(layout)


@dsl_user_op
def _cp_async_bulk_tensor_2d(
    dst_smem_addr: Int32,
    tensor_map_ptr: Int64,
    coord0: Int32,
    coord1: Int32,
    mbar_smem_addr: Int32,
    *,
    loc=None,
    ip=None,
):
    raise RuntimeError("raw tensor-map TMA issue is disabled; use CuTe atom TMA")


@cute.jit
def _dump_tma_stage_rows(
    mDst: cute.Tensor,
    sSrc: cute.Tensor,
    tidx,
    num_rows,
    head_dim,
    num_threads,
    max_rows,
):
    dump_rows = cutlass.select_(max_rows < num_rows, max_rows, num_rows)
    dst_rows = mDst.shape[0] * mDst.shape[1]
    dump_rows = cutlass.select_(dump_rows < dst_rows, dump_rows, dst_rows)
    linear = tidx
    dump_elems = dump_rows * head_dim
    while linear < dump_elems:
        row = linear // head_dim
        col = linear - row * head_dim
        dst_q = row // mDst.shape[1]
        dst_h = row - dst_q * mDst.shape[1]
        mDst[dst_q, dst_h, col] = sSrc[row, col, 0]
        linear += num_threads


@cute.jit
def _dump_s_frag_tile(
    mDst: cute.Tensor,
    s_frag: cute.Tensor,
    lane,
    warp_q_idx,
    warp_kv_idx,
    num_mma_q,
    num_mma_kv,
    packed_tile_rows,
    tile_tokens,
):
    lane_group = lane // 4
    lane_pair_base = 2 * (lane % 4)
    for mma_q in cutlass.range_constexpr(num_mma_q):
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            for reg_id in cutlass.range_constexpr(8):
                row_slot = (reg_id % 4) // 2
                row = (
                    warp_q_idx * num_mma_q * 16 + mma_q * 16 + lane_group + 8 * row_slot
                )
                col = (
                    warp_kv_idx * num_mma_kv * 16
                    + mma_kv * 16
                    + lane_pair_base
                    + 8 * (reg_id // 4)
                    + (reg_id % 2)
                )
                if row < packed_tile_rows and col < tile_tokens:
                    dst_linear = row * tile_tokens + col
                    dst_q = dst_linear // (mDst.shape[1] * mDst.shape[2])
                    dst_rem = dst_linear - dst_q * (mDst.shape[1] * mDst.shape[2])
                    dst_h = dst_rem // mDst.shape[2]
                    dst_col = dst_rem - dst_h * mDst.shape[2]
                    mDst[dst_q, dst_h, dst_col] = cutlass.BFloat16(
                        s_frag[mma_q, mma_kv, reg_id]
                    )


@cute.jit
def _dump_pv_copyfrag_regs(
    mDst: cute.Tensor,
    tOrVt: cute.Tensor,
    tOsVt: cute.Tensor,
    smem_thr_copy_V: cute.TiledCopy,
    lane,
    num_mma_d_vo,
):
    tCrV_copy_view = smem_thr_copy_V.retile(tOrVt)
    cute.copy(smem_thr_copy_V, tOsVt[None, None, 0], tCrV_copy_view[None, None, 0])
    for mma_d in cutlass.range_constexpr(num_mma_d_vo):
        if const_expr(mma_d < num_mma_d_vo - 1):
            cute.copy(
                smem_thr_copy_V,
                tOsVt[None, None, mma_d + 1],
                tCrV_copy_view[None, None, mma_d + 1],
            )
        b_regs = cute.flatten(cute.recast_tensor(tOrVt[None, None, mma_d], Uint32))
        if lane == Int32(0):
            for reg_id in cutlass.range_constexpr(cute.size(b_regs.shape)):
                v0, v1 = bfloat2_to_float2_scaled(b_regs[reg_id], Float32(1.0))
                mDst[mma_d * 8 + reg_id * 2 + 0] = cutlass.BFloat16(v0)
                mDst[mma_d * 8 + reg_id * 2 + 1] = cutlass.BFloat16(v1)


@cute.jit
def _dump_pv_copyfrag_regs_raw(
    mDst: cute.Tensor,
    tOrVt: cute.Tensor,
    tOsVt: cute.Tensor,
    smem_thr_copy_V: cute.TiledCopy,
    lane,
    num_mma_d_vo,
):
    tCrV_copy_view = smem_thr_copy_V.retile(tOrVt)
    cute.copy(smem_thr_copy_V, tOsVt[None, None, 0], tCrV_copy_view[None, None, 0])
    lane_words = num_mma_d_vo * 4
    dst_words = cute.size(mDst.shape)
    for mma_d in cutlass.range_constexpr(num_mma_d_vo):
        if const_expr(mma_d < num_mma_d_vo - 1):
            cute.copy(
                smem_thr_copy_V,
                tOsVt[None, None, mma_d + 1],
                tCrV_copy_view[None, None, mma_d + 1],
            )
        b_regs = cute.flatten(cute.recast_tensor(tOrVt[None, None, mma_d], Uint32))
        dst_idx = lane * lane_words + mma_d * 4
        if dst_idx + 0 < dst_words:
            mDst[dst_idx + 0] = b_regs[0]
        if dst_idx + 1 < dst_words:
            mDst[dst_idx + 1] = b_regs[1]
        if dst_idx + 2 < dst_words:
            mDst[dst_idx + 2] = b_regs[2]
        if dst_idx + 3 < dst_words:
            mDst[dst_idx + 3] = b_regs[3]


@cute.jit
def _dump_flat_u32_words(
    mDst: cute.Tensor,
    sSrc: cute.Tensor,
    tidx,
    num_threads,
):
    flat = cute.flatten(sSrc)
    dst_words = cute.size(mDst.shape)
    src_words = cute.size(flat.shape)
    dump_words = cutlass.select_(src_words < dst_words, src_words, dst_words)
    linear = tidx
    while linear < dump_words:
        mDst[linear] = flat[linear]
        linear += num_threads


@cute.jit
def _dump_flat_u32_words_offset(
    mDst: cute.Tensor,
    sSrc: cute.Tensor,
    dst_word_offset,
    tidx,
    num_threads,
):
    flat = cute.flatten(sSrc)
    dst_words = cute.size(mDst.shape)
    src_words = cute.size(flat.shape)
    dump_words = cutlass.select_(
        src_words < (dst_words - dst_word_offset),
        src_words,
        (dst_words - dst_word_offset),
    )
    linear = tidx
    while linear < dump_words:
        mDst[dst_word_offset + linear] = flat[linear]
        linear += num_threads


@cute.jit
def _issue_paged_kv_tma_copy_2planes_fp8_raw_impl(
    mDescPtrsFlat: cute.Tensor,
    kv_head_idx,
    kv_tma_plane_head_dim,
    sStageBytes: cute.Tensor,
    stage_plane_offset,
    kv_plane_total_bytes,
    producer_state,
    mbar_ptr,
    expected_bytes,
    mPageTable: cute.Tensor,
    request_idx,
    tile_token_base,
    page_size,
):
    page_idx = tile_token_base // page_size
    page_row_offset = tile_token_base - page_idx * page_size
    page_id = (
        Int32(0)
        if const_expr(
            os.environ.get("FLASHINFER_EXP_SM12X_PAGED_KV_TMA_FORCE_PAGE0", "0") == "1"
        )
        else mPageTable[request_idx, page_idx]
    )
    page_row_base = page_id * page_size + page_row_offset
    desc_ptr = Int64(mDescPtrsFlat[kv_head_idx])
    full_mbar_ptr = mbar_ptr + producer_state.index
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(
            full_mbar_ptr,
            expected_bytes,
        )
        tma_bar_addr = shared_ptr_to_u32(full_mbar_ptr)
        plane0_dst = shared_ptr_to_u32(
            sStageBytes.iterator + stage_plane_offset + Int32(0 * kv_plane_total_bytes)
        )
        plane1_dst = shared_ptr_to_u32(
            sStageBytes.iterator + stage_plane_offset + Int32(1 * kv_plane_total_bytes)
        )
        _cp_async_bulk_tensor_2d(
            plane0_dst,
            desc_ptr,
            Int32(0),
            page_row_base,
            tma_bar_addr,
        )
        _cp_async_bulk_tensor_2d(
            plane1_dst,
            desc_ptr,
            Int32(kv_tma_plane_head_dim),
            page_row_base,
            tma_bar_addr,
        )


@cute.jit
def _async_copy_q_tile_permuted_128b_impl(
    mQBytes: cute.Tensor,
    q_start,
    packed_tile_start,
    packed_tile_rows,
    kv_head_idx,
    group_size,
    num_q_heads,
    row_bytes,
    token_stride_bytes,
    head_stride_bytes,
    sQBytes: cute.Tensor,
    lane,
    warp_q_idx,
    num_mma_q,
    num_mma_d_qk,
    upcast_stride_q,
):
    lane_row = lane // 8
    lane_col = lane % 8
    warp_row_base = Int32(warp_q_idx * num_mma_q * 16)
    for mma_q in cutlass.range_constexpr(num_mma_q):
        for row_iter in cutlass.range_constexpr(4):
            packed_q_idx = Int32(
                packed_tile_start + warp_row_base + mma_q * 16 + lane_row + row_iter * 4
            )
            row_valid = packed_q_idx < (packed_tile_start + packed_tile_rows)
            q_row_local = packed_q_idx // group_size
            q_group_lane = packed_q_idx - q_row_local * group_size
            q_head_idx = Int32(kv_head_idx * group_size + q_group_lane)
            q_row_idx = Int32(q_start + q_row_local)
            row_byte_base = Int64(q_row_idx) * Int64(token_stride_bytes) + Int64(
                q_head_idx
            ) * Int64(head_stride_bytes)
            row_idx = Int32(warp_row_base + mma_q * 16 + lane_row + row_iter * 4)
            for mma_do in cutlass.range_constexpr(num_mma_d_qk // 4):
                vec_idx = Int32(lane_col + mma_do * 8)
                src_byte_idx = row_byte_base + vec_idx * 16
                dst_byte_idx = (
                    _permuted_offset_128b(row_idx, vec_idx, upcast_stride_q) * 16
                )
                _cp_async_load_128b_pred(
                    shared_ptr_to_u32(sQBytes.iterator + dst_byte_idx),
                    get_ptr_as_int64(mQBytes, src_byte_idx),
                    Int32(row_valid),
                )


@cute.jit
def _dump_plane_stage_words_u32(
    mDebugU32: cute.Tensor,
    sStageBytes: cute.Tensor,
    stage_idx,
    kv_plane_stage_bytes,
    kv_plane_total_bytes,
    kv_tma_plane_count,
    tidx,
    num_threads,
):
    plane_words = kv_plane_stage_bytes // 4
    plane0_u32 = cute.make_tensor(
        cute.recast_tensor(
            cute.make_tensor(
                sStageBytes.iterator
                + Int32(stage_idx * kv_plane_stage_bytes + 0 * kv_plane_total_bytes),
                cute.make_layout((kv_plane_stage_bytes,), stride=(1,)),
            ),
            cutlass.Uint32,
        ).iterator,
        cute.make_layout((plane_words,), stride=(1,)),
    )
    plane1_u32 = cute.make_tensor(
        cute.recast_tensor(
            cute.make_tensor(
                sStageBytes.iterator
                + Int32(stage_idx * kv_plane_stage_bytes + 1 * kv_plane_total_bytes),
                cute.make_layout((kv_plane_stage_bytes,), stride=(1,)),
            ),
            cutlass.Uint32,
        ).iterator,
        cute.make_layout((plane_words,), stride=(1,)),
    )
    _dump_flat_u32_words_offset(
        mDebugU32,
        plane0_u32,
        Int32(0),
        tidx,
        num_threads,
    )
    _dump_flat_u32_words_offset(
        mDebugU32,
        plane1_u32,
        Int32(plane_words),
        tidx,
        num_threads,
    )
    if const_expr(kv_tma_plane_count > 2):
        plane2_u32 = cute.make_tensor(
            cute.recast_tensor(
                cute.make_tensor(
                    sStageBytes.iterator
                    + Int32(
                        stage_idx * kv_plane_stage_bytes + 2 * kv_plane_total_bytes
                    ),
                    cute.make_layout((kv_plane_stage_bytes,), stride=(1,)),
                ),
                cutlass.Uint32,
            ).iterator,
            cute.make_layout((plane_words,), stride=(1,)),
        )
        plane3_u32 = cute.make_tensor(
            cute.recast_tensor(
                cute.make_tensor(
                    sStageBytes.iterator
                    + Int32(
                        stage_idx * kv_plane_stage_bytes + 3 * kv_plane_total_bytes
                    ),
                    cute.make_layout((kv_plane_stage_bytes,), stride=(1,)),
                ),
                cutlass.Uint32,
            ).iterator,
            cute.make_layout((plane_words,), stride=(1,)),
        )
        _dump_flat_u32_words_offset(
            mDebugU32,
            plane2_u32,
            Int32(plane_words * 2),
            tidx,
            num_threads,
        )
        _dump_flat_u32_words_offset(
            mDebugU32,
            plane3_u32,
            Int32(plane_words * 3),
            tidx,
            num_threads,
        )


@cute.jit
def _dump_p_frag_regs_raw(
    mDst: cute.Tensor,
    p_frag: cute.Tensor,
    lane,
):
    p_regs = cute.flatten(p_frag)
    dst_words = cute.size(mDst.shape)
    lane_words = cute.size(p_regs.shape)
    dst_idx = lane * lane_words
    for reg_id in cutlass.range_constexpr(cute.size(p_regs.shape)):
        if dst_idx + reg_id < dst_words:
            mDst[dst_idx + reg_id] = p_regs[reg_id]


@cute.jit
def _dump_s_frag_regs_raw(
    mDst: cute.Tensor,
    s_frag: cute.Tensor,
    lane,
):
    s_regs = cute.flatten(cute.recast_tensor(s_frag, cutlass.Uint32))
    dst_words = cute.size(mDst.shape)
    lane_words = cute.size(s_regs.shape)
    dst_idx = lane * lane_words
    for reg_id in cutlass.range_constexpr(cute.size(s_regs.shape)):
        if dst_idx + reg_id < dst_words:
            mDst[dst_idx + reg_id] = s_regs[reg_id]


@cute.jit
def _dump_s_frag_regs_raw_offset(
    mDst: cute.Tensor,
    s_frag: cute.Tensor,
    lane,
    dst_word_offset,
):
    s_regs = cute.flatten(cute.recast_tensor(s_frag, cutlass.Uint32))
    dst_words = cute.size(mDst.shape)
    lane_words = cute.size(s_regs.shape)
    dst_idx = dst_word_offset + lane * lane_words
    for reg_id in cutlass.range_constexpr(cute.size(s_regs.shape)):
        if dst_idx + reg_id < dst_words:
            mDst[dst_idx + reg_id] = s_regs[reg_id]


@cute.jit
def _permute_rowmajor_tile_in_place_to_permuted_128b(
    sStageBytes: cute.Tensor,
    stage_byte_offset,
    lane,
    warp_linear_idx,
    valid_rows,
    upcast_stride,
    total_warps,
):
    stage_u32 = cute.make_tensor(
        cute.recast_tensor(sStageBytes, cutlass.Uint32).iterator,
        cute.make_layout((cute.size(sStageBytes.shape) // 4,), stride=(1,)),
    )
    lane_row = lane // 8
    lane_col = lane % 8
    stage_word_offset = stage_byte_offset // 4
    for tile_iter in cutlass.range_constexpr(4):
        row_idx = Int32(warp_linear_idx * 4 + lane_row + tile_iter * total_warps * 4)
        row_word_base = stage_word_offset + row_idx * (upcast_stride * 4)
        for vec_iter in cutlass.range_constexpr(upcast_stride // 8):
            vec_idx = Int32(lane_col + vec_iter * 8)
            word_idx = row_word_base + vec_idx * 4
            if row_idx < valid_rows:
                swap_mask = Int32(row_idx % 8)
                partner_vec = vec_idx ^ swap_mask
                if vec_idx < partner_vec:
                    partner_word_idx = row_word_base + partner_vec * 4
                    a0 = stage_u32[word_idx + 0]
                    a1 = stage_u32[word_idx + 1]
                    a2 = stage_u32[word_idx + 2]
                    a3 = stage_u32[word_idx + 3]
                    b0 = stage_u32[partner_word_idx + 0]
                    b1 = stage_u32[partner_word_idx + 1]
                    b2 = stage_u32[partner_word_idx + 2]
                    b3 = stage_u32[partner_word_idx + 3]
                    stage_u32[word_idx + 0] = b0
                    stage_u32[word_idx + 1] = b1
                    stage_u32[word_idx + 2] = b2
                    stage_u32[word_idx + 3] = b3
                    stage_u32[partner_word_idx + 0] = a0
                    stage_u32[partner_word_idx + 1] = a1
                    stage_u32[partner_word_idx + 2] = a2
                    stage_u32[partner_word_idx + 3] = a3
            else:
                stage_u32[word_idx + 0] = Uint32(0)
                stage_u32[word_idx + 1] = Uint32(0)
                stage_u32[word_idx + 2] = Uint32(0)
                stage_u32[word_idx + 3] = Uint32(0)


@cute.jit
def _permute_rowmajor_tile_in_place_to_permuted_128b_vec128(
    sStageBytes: cute.Tensor,
    stage_byte_offset,
    lane,
    warp_linear_idx,
    valid_rows,
    upcast_stride,
    total_warps,
):
    lane_row = lane // 8
    lane_col = lane % 8
    for tile_iter in cutlass.range_constexpr(4):
        row_idx = Int32(warp_linear_idx * 4 + lane_row + tile_iter * total_warps * 4)
        for vec_iter in cutlass.range_constexpr(upcast_stride // 8):
            vec_idx = Int32(lane_col + vec_iter * 8)
            vec_byte_idx = stage_byte_offset + (row_idx * upcast_stride + vec_idx) * 16
            vec_addr = shared_ptr_to_u32(sStageBytes.iterator + vec_byte_idx)
            if row_idx < valid_rows:
                swap_mask = Int32(row_idx % 8)
                partner_vec = vec_idx ^ swap_mask
                if vec_idx < partner_vec:
                    partner_vec_byte_idx = (
                        stage_byte_offset + (row_idx * upcast_stride + partner_vec) * 16
                    )
                    partner_vec_addr = shared_ptr_to_u32(
                        sStageBytes.iterator + partner_vec_byte_idx
                    )
                    a0, a1, a2, a3 = ld_shared_v4_u32(vec_addr)
                    b0, b1, b2, b3 = ld_shared_v4_u32(partner_vec_addr)
                    st_shared_v4_u32(vec_addr, b0, b1, b2, b3)
                    st_shared_v4_u32(partner_vec_addr, a0, a1, a2, a3)
            else:
                st_shared_v4_u32(vec_addr, Uint32(0), Uint32(0), Uint32(0), Uint32(0))


@dsl_user_op
def _cp_async_load_128b_pred(
    smem_addr: Int32,
    gmem_addr: Int64,
    predicate: Int32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [
            Int32(predicate).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, $0, 0;\n"
        " @p cp.async.cg.shared.global.L2::128B [$1], [$2], 16;\n"
        "}",
        "r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _cp_async_load_128b_zfill(
    smem_addr: Int32,
    gmem_addr: Int64,
    src_bytes: Int32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
            Int32(src_bytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::128B [$0], [$1], 16, $2;",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _exp2_approx_ftz_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _apply_attention_sink_after_lse_scale(
    o_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    mAttentionSinkBias: cute.Tensor,
    mma_q,
    row_slot,
    q_head_idx: Int32,
    row_valid: Int32,
    causal_k_limit: Int32,
    chunk_start: Int32,
    chunk_end: Int32,
    warp_kv_idx: Int32,
    num_mma_d_vo,
    softmax_scale_log2: Float32,
    has_attention_sink_bias,
    split_kv,
):
    if m_frag[mma_q, row_slot] != -Float32.inf:
        m_frag[mma_q, row_slot] = Float32(m_frag[mma_q, row_slot] * softmax_scale_log2)
    if const_expr(has_attention_sink_bias):
        sink_owner = (row_valid != Int32(0)) and (warp_kv_idx == Int32(0))
        if const_expr(split_kv):
            if sink_owner:
                sink_owner = (chunk_start <= causal_k_limit) and (
                    causal_k_limit < chunk_end
                )
        if sink_owner:
            old_m = m_frag[mma_q, row_slot]
            sink_m = Float32(mAttentionSinkBias[q_head_idx] * attention_ops.LOG2_E)
            new_m = attention_ops.fmax(old_m, sink_m)
            old_scale = (
                Float32(0.0)
                if old_m == -Float32.inf
                else _exp2_approx_ftz_f32(old_m - new_m)
            )
            sink_scale = _exp2_approx_ftz_f32(sink_m - new_m)
            d_frag[mma_q, row_slot] = Float32(
                d_frag[mma_q, row_slot] * old_scale + sink_scale
            )
            for mma_d in cutlass.range_constexpr(num_mma_d_vo):
                reg_base = row_slot * 2
                o_frag[mma_q, mma_d, reg_base + 0] = Float32(
                    o_frag[mma_q, mma_d, reg_base + 0] * old_scale
                )
                o_frag[mma_q, mma_d, reg_base + 1] = Float32(
                    o_frag[mma_q, mma_d, reg_base + 1] * old_scale
                )
                o_frag[mma_q, mma_d, reg_base + 4] = Float32(
                    o_frag[mma_q, mma_d, reg_base + 4] * old_scale
                )
                o_frag[mma_q, mma_d, reg_base + 5] = Float32(
                    o_frag[mma_q, mma_d, reg_base + 5] * old_scale
                )
            m_frag[mma_q, row_slot] = Float32(new_m)


@cute.jit
def _apply_relative_attention_bias(
    frag_s: cute.Tensor,
    mRelativeAttentionBias: cute.Tensor,
    q_row_idx_frag: cute.Tensor,
    q_head_idx_frag: cute.Tensor,
    causal_k_limit: cute.Tensor,
    tile_key_base: Int32,
    warp_kv_base: Int32,
    lane_pair_base: Int32,
    inverse_softmax_scale: Float32,
    num_mma_q,
    num_mma_kv,
):
    relative_extent = mRelativeAttentionBias.shape[2]
    for mma_q in cutlass.range_constexpr(num_mma_q):
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            for reg_id in cutlass.range_constexpr(8):
                row_slot = (reg_id % 4) // 2
                if frag_s[mma_q, mma_kv, reg_id] != -Float32.inf:
                    key_local = (
                        warp_kv_base
                        + mma_kv * 16
                        + lane_pair_base
                        + 8 * (reg_id // 4)
                        + (reg_id % 2)
                    )
                    distance = (
                        causal_k_limit[mma_q, row_slot] - tile_key_base - key_local
                    )
                    if distance >= Int32(0) and distance < relative_extent:
                        bias = Float32(
                            mRelativeAttentionBias[
                                q_row_idx_frag[mma_q, row_slot],
                                q_head_idx_frag[mma_q, row_slot],
                                distance,
                            ]
                        )
                        frag_s[mma_q, mma_kv, reg_id] += bias * inverse_softmax_scale


@dsl_user_op
def _exit_thread(
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [],
        "exit;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _permuted_offset_128b(row_idx, vec_idx, stride_128b):
    return row_idx * stride_128b + (vec_idx ^ (row_idx % 8))


@cute.jit
def _smem_addr_from_b128_offset(base_addr: Int32, offset_128b):
    return base_addr + Int32(offset_128b * 16)


@cute.jit
def _advance_offset_by_row_128b(offset_128b, step_size, row_stride_128b):
    return offset_128b + step_size * row_stride_128b


@cute.jit
def _advance_offset_by_column_128b_2(offset_128b, step_idx):
    xor_term = Int32(0x2) + (Int32(0x4) if const_expr(step_idx % 2 == 1) else Int32(0))
    extra = Int32(8) if const_expr(step_idx % 4 == 3) else Int32(0)
    return (offset_128b ^ xor_term) + extra


@cute.jit
def _smem_addr_from_split_planes_128b(
    plane0_base_addr: Int32,
    plane1_base_addr: Int32,
    full_offset_128b,
    full_stride_128b,
):
    plane_stride_128b = full_stride_128b // Int32(2)
    row = full_offset_128b // full_stride_128b
    col = full_offset_128b - row * full_stride_128b
    plane_idx = col // plane_stride_128b
    local_col = col - plane_idx * plane_stride_128b
    local_offset = row * plane_stride_128b + local_col
    plane_base_addr = cutlass.select_(
        plane_idx == Int32(0), plane0_base_addr, plane1_base_addr
    )
    return _smem_addr_from_b128_offset(plane_base_addr, local_offset)


def _transpose_view(a: cute.Tensor) -> cute.Tensor:
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def _convert_layout_acc_mn(
    acc_layout: cute.Layout, transpose: bool = False
) -> cute.Layout:
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),
        *acc_layout_col_major.stride[3:],
    )
    if transpose:
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


def _reshape_acc_to_mn(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(
        acc.iterator, _convert_layout_acc_mn(acc.layout, transpose=transpose)
    )


@cute.jit
def _convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    if const_expr(cute.rank(acc_layout.shape[0]) == 3):
        div = 2 if const_expr(acc_layout.shape[0][2] % 2 == 0) else 1
        l = cute.logical_divide(acc_layout, ((None, None, div), None, None))
        return cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
                l.shape[1],
                (l.shape[0][2][1], l.shape[2]),
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                l.stride[1],
                (l.stride[0][2][1], l.stride[2]),
            ),
        )
    l = cute.logical_divide(acc_layout, (None, None, 2))
    return cute.make_layout(
        (
            (l.shape[0], l.shape[2][0]),
            l.shape[1],
            l.shape[2][1],
        ),
        stride=(
            (l.stride[0], l.stride[2][0]),
            l.stride[1],
            l.stride[2][1],
        ),
    )


def _reshape_acc_to_frgA(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, _convert_layout_acc_frgA(acc.layout))


@cute.jit
def _mask_donor_acc_s_tma(
    acc_S: cute.Tensor,
    tScS_mn: cute.Tensor,
    t0ScS_mn: cute.Tensor,
    packed_tile_rows,
    tile_tokens,
    tile_base,
    cache_len,
    qo_len,
    group_size,
):
    acc_S_mn = _reshape_acc_to_mn(acc_S)
    thr_col_offset = tScS_mn[0][1]
    for r in cutlass.range_constexpr(cute.size(tScS_mn.shape[0])):
        row_idx = tScS_mn[r, 0][0]
        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
            col_idx = t0ScS_mn[0, c][1] + thr_col_offset
            valid = row_idx < packed_tile_rows and col_idx < tile_tokens
            if valid:
                q_token_local = row_idx // group_size
                causal_k_limit = q_token_local + cache_len - qo_len
                valid = (tile_base + col_idx) <= causal_k_limit
            if not valid:
                acc_S_mn[r, c] = -Float32.inf


@cute.jit
def _donor_update_mdo_states_fp32_pack_p(
    acc_S: cute.Tensor,
    o_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    sm_scale_log2: Float32,
    num_mma_d_vo,
    dtype_p: cutlass.Constexpr,
):
    acc_S_mn = _reshape_acc_to_mn(acc_S)
    for row_slot in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
        m_prev = Float32(m_frag[0, row_slot])
        m_new = Float32(m_prev)
        for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
            m_new = attention_ops.fmax(m_new, acc_S_mn[row_slot, c])
        m_new = cute.arch.warp_reduction_max(m_new, threads_in_group=4)

        scale_term = (
            Float32(1.0)
            if m_new == -Float32.inf
            else _exp2_approx_ftz_f32((m_prev - m_new) * sm_scale_log2)
        )
        d_frag[0, row_slot] = Float32(d_frag[0, row_slot] * scale_term)
        for mma_d in cutlass.range_constexpr(num_mma_d_vo):
            o_frag[0, mma_d, row_slot * 2 + 0] *= scale_term
            o_frag[0, mma_d, row_slot * 2 + 1] *= scale_term
            o_frag[0, mma_d, row_slot * 2 + 4] *= scale_term
            o_frag[0, mma_d, row_slot * 2 + 5] *= scale_term

        for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
            acc_S_mn[row_slot, c] = (
                Float32(0.0)
                if m_new == -Float32.inf
                else _exp2_approx_ftz_f32(
                    (acc_S_mn[row_slot, c] - m_new) * sm_scale_log2
                )
            )
        m_frag[0, row_slot] = Float32(m_new)

    rP = cute.make_fragment_like(acc_S, dtype_p)
    rP.store(acc_S.load().to(dtype_p))
    return cute.recast_tensor(_reshape_acc_to_frgA(rP), Uint32)


@cute.jit
def _acc_mn_to_frag_s(
    frag_S: cute.Tensor,
    acc_S_mn: cute.Tensor,
    tScS_mn: cute.Tensor,
    t0ScS_mn: cute.Tensor,
    lane,
    warp_q_idx,
    warp_kv_idx,
    num_mma_q,
    num_mma_kv,
):
    lane_group = lane // 4
    lane_pair_base = 2 * (lane % 4)
    thr_col_offset = tScS_mn[0][1]
    for r in cutlass.range_constexpr(cute.size(tScS_mn.shape[0])):
        row = tScS_mn[r, 0][0]
        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
            col = t0ScS_mn[0, c][1] + thr_col_offset
            val = acc_S_mn[r, c]
            for mma_q in cutlass.range_constexpr(num_mma_q):
                for mma_kv in cutlass.range_constexpr(num_mma_kv):
                    for reg_id in cutlass.range_constexpr(8):
                        row_slot = (reg_id % 4) // 2
                        target_row = (
                            warp_q_idx * num_mma_q * 16
                            + mma_q * 16
                            + lane_group
                            + 8 * row_slot
                        )
                        target_col = (
                            warp_kv_idx * num_mma_kv * 16
                            + mma_kv * 16
                            + lane_pair_base
                            + 8 * (reg_id // 4)
                            + (reg_id % 2)
                        )
                        if row == target_row and col == target_col:
                            frag_S[mma_q, mma_kv, reg_id] = val


@cute.jit
def _warp_mma_gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
    A_in_regs: cutlass.Constexpr = False,
    B_in_regs: cutlass.Constexpr = False,
):
    tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    if const_expr(not A_in_regs):
        cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
    if const_expr(not B_in_regs):
        cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
        if k < cute.size(tCsA.shape[2]) - 1:
            if const_expr(not A_in_regs):
                cute.copy(
                    smem_thr_copy_A,
                    tCsA[None, None, k + 1],
                    tCrA_copy_view[None, None, k + 1],
                )
            if const_expr(not B_in_regs):
                cute.copy(
                    smem_thr_copy_B,
                    tCsB[None, None, k + 1],
                    tCrB_copy_view[None, None, k + 1],
                )
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def _warp_mma_gemm_rs(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_B: cute.TiledCopy,
):
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if const_expr(k < cute.size(tCrA.shape[2]) - 1):
            cute.copy(
                smem_thr_copy_B,
                tCsB[None, None, k + 1],
                tCrB_copy_view[None, None, k + 1],
            )
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def _literal_qk_mma_into_sfrag(
    s_frag: cute.Tensor,
    q_base_addr: Int32,
    k_base_addr: Int32,
    lane,
    warp_q_idx,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_qk,
    upcast_stride_q,
    upcast_stride_k,
):
    for mma_d in cutlass.range_constexpr(num_mma_d_qk):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        for mma_q in cutlass.range_constexpr(num_mma_q):
            q_row = warp_q_idx * num_mma_q * 16 + mma_q * 16 + lane % 16
            q_col = mma_d * 2 + lane // 16
            q_offset = _permuted_offset_128b(q_row, q_col, upcast_stride_q)
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(q_base_addr, q_offset)
            )
            a_regs[mma_q, 0] = a0
            a_regs[mma_q, 1] = a1
            a_regs[mma_q, 2] = a2
            a_regs[mma_q, 3] = a3

        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            k_row = (
                row_base
                + warp_kv_idx * num_mma_kv * 16
                + mma_kv * 16
                + 8 * (lane // 16)
                + lane % 8
            )
            k_col = mma_d * 2 + (lane % 16) // 8
            k_offset = _permuted_offset_128b(k_row, k_col, upcast_stride_k)
            b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset)
            )

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    s_frag[mma_q, mma_kv, 0],
                    s_frag[mma_q, mma_kv, 1],
                    s_frag[mma_q, mma_kv, 2],
                    s_frag[mma_q, mma_kv, 3],
                    s_frag[mma_q, mma_kv, 4],
                    s_frag[mma_q, mma_kv, 5],
                    s_frag[mma_q, mma_kv, 6],
                    s_frag[mma_q, mma_kv, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0,
                    b1,
                    b2,
                    b3,
                )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7


@cute.jit
def _literal_qk_mma_into_sfrag_plane_bf16(
    s_frag: cute.Tensor,
    q_base_addr: Int32,
    k_plane0_base_addr: Int32,
    k_plane1_base_addr: Int32,
    k_plane2_base_addr: Int32,
    k_plane3_base_addr: Int32,
    lane,
    warp_q_idx,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_qk,
    upcast_stride_q,
    upcast_stride_plane,
):
    for mma_d in cutlass.range_constexpr(num_mma_d_qk):
        plane_idx = mma_d // 4
        mma_d_local = mma_d - plane_idx * 4
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        for mma_q in cutlass.range_constexpr(num_mma_q):
            q_row = warp_q_idx * num_mma_q * 16 + mma_q * 16 + lane % 16
            q_col = mma_d * 2 + lane // 16
            q_offset = _permuted_offset_128b(q_row, q_col, upcast_stride_q)
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(q_base_addr, q_offset)
            )
            a_regs[mma_q, 0] = a0
            a_regs[mma_q, 1] = a1
            a_regs[mma_q, 2] = a2
            a_regs[mma_q, 3] = a3

        if const_expr(plane_idx == 0):
            k_plane_base_addr = k_plane0_base_addr
        elif const_expr(plane_idx == 1):
            k_plane_base_addr = k_plane1_base_addr
        elif const_expr(plane_idx == 2):
            k_plane_base_addr = k_plane2_base_addr
        else:
            k_plane_base_addr = k_plane3_base_addr

        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            k_row = (
                row_base
                + warp_kv_idx * num_mma_kv * 16
                + mma_kv * 16
                + 8 * (lane // 16)
                + lane % 8
            )
            k_col = mma_d_local * 2 + (lane % 16) // 8
            k_offset = _permuted_offset_128b(k_row, k_col, upcast_stride_plane)
            b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(k_plane_base_addr, k_offset)
            )

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    s_frag[mma_q, mma_kv, 0],
                    s_frag[mma_q, mma_kv, 1],
                    s_frag[mma_q, mma_kv, 2],
                    s_frag[mma_q, mma_kv, 3],
                    s_frag[mma_q, mma_kv, 4],
                    s_frag[mma_q, mma_kv, 5],
                    s_frag[mma_q, mma_kv, 6],
                    s_frag[mma_q, mma_kv, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0,
                    b1,
                    b2,
                    b3,
                )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7


@cute.jit
def _literal_qk_mma_into_sfrag_fp8_raw(
    s_frag: cute.Tensor,
    q_base_addr: Int32,
    k_base_addr: Int32,
    lane,
    warp_q_idx,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_qk,
    upcast_stride_q,
    upcast_stride_k,
):
    q_offset = _permuted_offset_128b(
        warp_q_idx * num_mma_q * 16 + lane % 16,
        lane // 16,
        upcast_stride_q,
    )
    k_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + 8 * (lane // 16) + lane % 8,
        (lane % 16) // 8,
        upcast_stride_k,
    )
    for mma_d in cutlass.range_constexpr(num_mma_d_qk):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        q_offset_cur = q_offset
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(q_base_addr, q_offset_cur)
            )
            a_regs[mma_q, 0] = a0
            a_regs[mma_q, 1] = a1
            a_regs[mma_q, 2] = a2
            a_regs[mma_q, 3] = a3
            q_offset_cur = _advance_offset_by_row_128b(
                q_offset_cur, 16, upcast_stride_q
            )
        q_offset = _advance_offset_by_column_128b_2(q_offset_cur, mma_d) - Int32(
            num_mma_q * 16 * upcast_stride_q
        )

        k_offset_cur = k_offset
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            if const_expr(mma_d % 2 == 0):
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_left_half_b16(
                    _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
                )
            else:
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_right_half_b16(
                    _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
                )
            b_f8_0 = frag_layout_swizzle_16b_to_8b(b_f8_0)
            b_f8_1 = frag_layout_swizzle_16b_to_8b(b_f8_1)
            b0, b1 = fp8x4_e4m3_to_bfloat2x2(b_f8_0)
            b2, b3 = fp8x4_e4m3_to_bfloat2x2(b_f8_1)
            k_offset_cur = _advance_offset_by_row_128b(
                k_offset_cur, 16, upcast_stride_k
            )

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    s_frag[mma_q, mma_kv, 0],
                    s_frag[mma_q, mma_kv, 1],
                    s_frag[mma_q, mma_kv, 2],
                    s_frag[mma_q, mma_kv, 3],
                    s_frag[mma_q, mma_kv, 4],
                    s_frag[mma_q, mma_kv, 5],
                    s_frag[mma_q, mma_kv, 6],
                    s_frag[mma_q, mma_kv, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0,
                    b1,
                    b2,
                    b3,
                )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7

        if const_expr(mma_d % 2 == 1):
            k_offset = _advance_offset_by_column_128b_2(
                k_offset_cur, mma_d // 2
            ) - Int32(num_mma_kv * 16 * upcast_stride_k)
        else:
            k_offset = k_offset_cur - Int32(num_mma_kv * 16 * upcast_stride_k)


@cute.jit
def _literal_qk_mma_into_sfrag_fp8_raw_paired(
    s_frag: cute.Tensor,
    q_base_addr: Int32,
    k_base_addr: Int32,
    lane,
    warp_q_idx,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_qk,
    upcast_stride_q,
    upcast_stride_k,
):
    q_offset = _permuted_offset_128b(
        warp_q_idx * num_mma_q * 16 + lane % 16,
        lane // 16,
        upcast_stride_q,
    )
    k_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + 8 * (lane // 16) + lane % 8,
        (lane % 16) // 8,
        upcast_stride_k,
    )
    for mma_pair in cutlass.range_constexpr(num_mma_d_qk // 2):
        a_regs_k0 = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        a_regs_k1 = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )

        q_offset_cur = q_offset
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(q_base_addr, q_offset_cur)
            )
            a_regs_k0[mma_q, 0] = a0
            a_regs_k0[mma_q, 1] = a1
            a_regs_k0[mma_q, 2] = a2
            a_regs_k0[mma_q, 3] = a3
            q_offset_cur = _advance_offset_by_row_128b(
                q_offset_cur, 16, upcast_stride_q
            )

        mma_d0 = mma_pair * 2
        q_offset_mid = _advance_offset_by_column_128b_2(q_offset_cur, mma_d0) - Int32(
            num_mma_q * 16 * upcast_stride_q
        )
        q_offset_cur = q_offset_mid
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(q_base_addr, q_offset_cur)
            )
            a_regs_k1[mma_q, 0] = a0
            a_regs_k1[mma_q, 1] = a1
            a_regs_k1[mma_q, 2] = a2
            a_regs_k1[mma_q, 3] = a3
            q_offset_cur = _advance_offset_by_row_128b(
                q_offset_cur, 16, upcast_stride_q
            )
        q_offset = _advance_offset_by_column_128b_2(q_offset_cur, mma_d0 + 1) - Int32(
            num_mma_q * 16 * upcast_stride_q
        )

        k_offset_cur = k_offset
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            b_f8_0_l, b_f8_1_l = ldmatrix_m8n8x4_left_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b_f8_0_r, b_f8_1_r = ldmatrix_m8n8x4_right_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b_f8_0_l = frag_layout_swizzle_16b_to_8b(b_f8_0_l)
            b_f8_1_l = frag_layout_swizzle_16b_to_8b(b_f8_1_l)
            b_f8_0_r = frag_layout_swizzle_16b_to_8b(b_f8_0_r)
            b_f8_1_r = frag_layout_swizzle_16b_to_8b(b_f8_1_r)
            bl0, bl1 = fp8x4_e4m3_to_bfloat2x2(b_f8_0_l)
            bl2, bl3 = fp8x4_e4m3_to_bfloat2x2(b_f8_1_l)
            br0, br1 = fp8x4_e4m3_to_bfloat2x2(b_f8_0_r)
            br2, br3 = fp8x4_e4m3_to_bfloat2x2(b_f8_1_r)
            k_offset_cur = _advance_offset_by_row_128b(
                k_offset_cur, 16, upcast_stride_k
            )

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    s_frag[mma_q, mma_kv, 0],
                    s_frag[mma_q, mma_kv, 1],
                    s_frag[mma_q, mma_kv, 2],
                    s_frag[mma_q, mma_kv, 3],
                    s_frag[mma_q, mma_kv, 4],
                    s_frag[mma_q, mma_kv, 5],
                    s_frag[mma_q, mma_kv, 6],
                    s_frag[mma_q, mma_kv, 7],
                    a_regs_k0[mma_q, 0],
                    a_regs_k0[mma_q, 1],
                    a_regs_k0[mma_q, 2],
                    a_regs_k0[mma_q, 3],
                    bl0,
                    bl1,
                    bl2,
                    bl3,
                )
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    d0,
                    d1,
                    d2,
                    d3,
                    d4,
                    d5,
                    d6,
                    d7,
                    a_regs_k1[mma_q, 0],
                    a_regs_k1[mma_q, 1],
                    a_regs_k1[mma_q, 2],
                    a_regs_k1[mma_q, 3],
                    br0,
                    br1,
                    br2,
                    br3,
                )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7

        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            num_mma_kv * 16 * upcast_stride_k
        )


@cute.jit
def _literal_qk_mma_into_sfrag_plane_fp8_raw(
    s_frag: cute.Tensor,
    q_base_addr: Int32,
    k_plane0_base_addr: Int32,
    k_plane1_base_addr: Int32,
    lane,
    warp_q_idx,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_qk,
    upcast_stride_q,
    upcast_stride_plane,
):
    upcast_stride_full = upcast_stride_plane * Int32(2)
    q_offset = _permuted_offset_128b(
        warp_q_idx * num_mma_q * 16 + lane % 16,
        lane // 16,
        upcast_stride_q,
    )
    k_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + 8 * (lane // 16) + lane % 8,
        (lane % 16) // 8,
        upcast_stride_full,
    )
    for mma_d in cutlass.range_constexpr(num_mma_d_qk):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        q_offset_cur = q_offset
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(q_base_addr, q_offset_cur)
            )
            a_regs[mma_q, 0] = a0
            a_regs[mma_q, 1] = a1
            a_regs[mma_q, 2] = a2
            a_regs[mma_q, 3] = a3
            q_offset_cur = _advance_offset_by_row_128b(
                q_offset_cur, 16, upcast_stride_q
            )
        q_offset = _advance_offset_by_column_128b_2(q_offset_cur, mma_d) - Int32(
            num_mma_q * 16 * upcast_stride_q
        )

        k_offset_cur = k_offset
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            k_addr = _smem_addr_from_split_planes_128b(
                k_plane0_base_addr,
                k_plane1_base_addr,
                k_offset_cur,
                upcast_stride_full,
            )
            if const_expr(mma_d % 2 == 0):
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_left_half_b16(k_addr)
            else:
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_right_half_b16(k_addr)
            b_f8_0 = frag_layout_swizzle_16b_to_8b(b_f8_0)
            b_f8_1 = frag_layout_swizzle_16b_to_8b(b_f8_1)
            b0, b1 = fp8x4_e4m3_to_bfloat2x2(b_f8_0)
            b2, b3 = fp8x4_e4m3_to_bfloat2x2(b_f8_1)
            k_offset_cur = _advance_offset_by_row_128b(
                k_offset_cur, 16, upcast_stride_full
            )

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    s_frag[mma_q, mma_kv, 0],
                    s_frag[mma_q, mma_kv, 1],
                    s_frag[mma_q, mma_kv, 2],
                    s_frag[mma_q, mma_kv, 3],
                    s_frag[mma_q, mma_kv, 4],
                    s_frag[mma_q, mma_kv, 5],
                    s_frag[mma_q, mma_kv, 6],
                    s_frag[mma_q, mma_kv, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0,
                    b1,
                    b2,
                    b3,
                )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7

        if const_expr(mma_d % 2 == 1):
            k_offset = _advance_offset_by_column_128b_2(
                k_offset_cur, mma_d // 2
            ) - Int32(num_mma_kv * 16 * upcast_stride_full)
        else:
            k_offset = k_offset_cur - Int32(num_mma_kv * 16 * upcast_stride_full)


@cute.jit
def _literal_qk_mma_into_sfrag_mxfp8_raw(
    s_frag: cute.Tensor,
    q_base_addr: Int32,
    k_base_addr: Int32,
    lane,
    warp_q_idx,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_qk,
    upcast_stride_q,
    upcast_stride_k,
):
    unit_scale = Uint32(0x7F7F7F7F)
    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)
    q_offset = _permuted_offset_128b(
        warp_q_idx * num_mma_q * 16 + lane % 16,
        lane // 16,
        upcast_stride_q,
    )
    k_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + 8 * (lane // 16) + lane % 8,
        (lane % 16) // 8,
        upcast_stride_k,
    )
    for mma_pair in cutlass.range_constexpr(num_mma_d_qk // 2):
        a_regs_k0 = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        a_regs_k1 = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )

        q_offset_cur = q_offset
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(q_base_addr, q_offset_cur)
            )
            a_regs_k0[mma_q, 0] = a0
            a_regs_k0[mma_q, 1] = a1
            a_regs_k0[mma_q, 2] = a2
            a_regs_k0[mma_q, 3] = a3
            q_offset_cur = _advance_offset_by_row_128b(
                q_offset_cur, 16, upcast_stride_q
            )

        mma_d0 = mma_pair * 2
        q_offset_mid = _advance_offset_by_column_128b_2(q_offset_cur, mma_d0) - Int32(
            num_mma_q * 16 * upcast_stride_q
        )
        q_offset_cur = q_offset_mid
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(q_base_addr, q_offset_cur)
            )
            a_regs_k1[mma_q, 0] = a0
            a_regs_k1[mma_q, 1] = a1
            a_regs_k1[mma_q, 2] = a2
            a_regs_k1[mma_q, 3] = a3
            q_offset_cur = _advance_offset_by_row_128b(
                q_offset_cur, 16, upcast_stride_q
            )
        q_offset = _advance_offset_by_column_128b_2(q_offset_cur, mma_d0 + 1) - Int32(
            num_mma_q * 16 * upcast_stride_q
        )

        k_offset_cur = k_offset
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            b0_k0, b1_k0 = ldmatrix_m8n8x4_left_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_right_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
            b1_k0 = frag_layout_swizzle_16b_to_8b(b1_k0)
            b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
            b1_k1 = frag_layout_swizzle_16b_to_8b(b1_k1)
            k_offset_cur = _advance_offset_by_row_128b(
                k_offset_cur, 16, upcast_stride_k
            )

            for mma_q in cutlass.range_constexpr(num_mma_q):
                qa0 = (cvt_bf16x2_to_e4m3x2(a_regs_k0[mma_q, 0]) & mask16) | (
                    (cvt_bf16x2_to_e4m3x2(a_regs_k1[mma_q, 0]) & mask16) << shift16
                )
                qa1 = (cvt_bf16x2_to_e4m3x2(a_regs_k0[mma_q, 1]) & mask16) | (
                    (cvt_bf16x2_to_e4m3x2(a_regs_k1[mma_q, 1]) & mask16) << shift16
                )
                qa2 = (cvt_bf16x2_to_e4m3x2(a_regs_k0[mma_q, 2]) & mask16) | (
                    (cvt_bf16x2_to_e4m3x2(a_regs_k1[mma_q, 2]) & mask16) << shift16
                )
                qa3 = (cvt_bf16x2_to_e4m3x2(a_regs_k0[mma_q, 3]) & mask16) | (
                    (cvt_bf16x2_to_e4m3x2(a_regs_k1[mma_q, 3]) & mask16) << shift16
                )

                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    s_frag[mma_q, mma_kv, 0],
                    s_frag[mma_q, mma_kv, 1],
                    s_frag[mma_q, mma_kv, 2],
                    s_frag[mma_q, mma_kv, 3],
                    qa0,
                    qa1,
                    qa2,
                    qa3,
                    b0_k0,
                    b0_k1,
                    unit_scale,
                    unit_scale,
                )
                d4, d5, d6, d7 = mxfp8_mma_m16n8k32_f32_e4m3(
                    s_frag[mma_q, mma_kv, 4],
                    s_frag[mma_q, mma_kv, 5],
                    s_frag[mma_q, mma_kv, 6],
                    s_frag[mma_q, mma_kv, 7],
                    qa0,
                    qa1,
                    qa2,
                    qa3,
                    b1_k0,
                    b1_k1,
                    unit_scale,
                    unit_scale,
                )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7

        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            num_mma_kv * 16 * upcast_stride_k
        )


@cute.jit
def _literal_pv_mma_into_ofrag_bf16_packed(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    lane,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_vo,
    upcast_stride_v,
    v_scale,
    debug_regs: cute.Tensor | None = None,
):
    v_scale_bf2 = broadcast_f32_to_bfloat2(v_scale)
    v_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + lane % 16,
        lane // 16,
        upcast_stride_v,
    )
    for mma_kv in cutlass.range_constexpr(num_mma_kv):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a_regs[mma_q, 0] = bfloat2_mul(p_frag[mma_q, mma_kv, 0], v_scale_bf2)
            a_regs[mma_q, 1] = bfloat2_mul(p_frag[mma_q, mma_kv, 1], v_scale_bf2)
            a_regs[mma_q, 2] = bfloat2_mul(p_frag[mma_q, mma_kv, 2], v_scale_bf2)
            a_regs[mma_q, 3] = bfloat2_mul(p_frag[mma_q, mma_kv, 3], v_scale_bf2)

        v_offset_cur = v_offset
        for mma_d in cutlass.range_constexpr(num_mma_d_vo):
            b0, b1, b2, b3 = ldmatrix_m8n8x4_trans_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_cur)
            )
            if const_expr(debug_regs is not None):
                lane_words = num_mma_kv * num_mma_d_vo * 4
                dst_words = cute.size(debug_regs.shape)
                dst_idx = lane * lane_words + (mma_kv * num_mma_d_vo + mma_d) * 4
                if dst_idx + 0 < dst_words:
                    debug_regs[dst_idx + 0] = b0
                if dst_idx + 1 < dst_words:
                    debug_regs[dst_idx + 1] = b1
                if dst_idx + 2 < dst_words:
                    debug_regs[dst_idx + 2] = b2
                if dst_idx + 3 < dst_words:
                    debug_regs[dst_idx + 3] = b3
            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    o_frag[mma_q, mma_d, 0],
                    o_frag[mma_q, mma_d, 1],
                    o_frag[mma_q, mma_d, 2],
                    o_frag[mma_q, mma_d, 3],
                    o_frag[mma_q, mma_d, 4],
                    o_frag[mma_q, mma_d, 5],
                    o_frag[mma_q, mma_d, 6],
                    o_frag[mma_q, mma_d, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0,
                    b1,
                    b2,
                    b3,
                )
                o_frag[mma_q, mma_d, 0] = d0
                o_frag[mma_q, mma_d, 1] = d1
                o_frag[mma_q, mma_d, 2] = d2
                o_frag[mma_q, mma_d, 3] = d3
                o_frag[mma_q, mma_d, 4] = d4
                o_frag[mma_q, mma_d, 5] = d5
                o_frag[mma_q, mma_d, 6] = d6
                o_frag[mma_q, mma_d, 7] = d7
            v_offset_cur = _advance_offset_by_column_128b_2(v_offset_cur, mma_d)
        v_offset = _advance_offset_by_row_128b(
            v_offset_cur, 16, upcast_stride_v
        ) - Int32(2 * num_mma_d_vo)
    v_offset -= Int32(16 * num_mma_kv * upcast_stride_v)


@cute.jit
def _literal_pv_mma_into_ofrag_plane_bf16_packed(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_plane0_base_addr: Int32,
    v_plane1_base_addr: Int32,
    v_plane2_base_addr: Int32,
    v_plane3_base_addr: Int32,
    lane,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_vo,
    upcast_stride_plane,
    v_scale,
    debug_regs: cute.Tensor | None = None,
):
    v_scale_bf2 = broadcast_f32_to_bfloat2(v_scale)
    for mma_kv in cutlass.range_constexpr(num_mma_kv):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a_regs[mma_q, 0] = bfloat2_mul(p_frag[mma_q, mma_kv, 0], v_scale_bf2)
            a_regs[mma_q, 1] = bfloat2_mul(p_frag[mma_q, mma_kv, 1], v_scale_bf2)
            a_regs[mma_q, 2] = bfloat2_mul(p_frag[mma_q, mma_kv, 2], v_scale_bf2)
            a_regs[mma_q, 3] = bfloat2_mul(p_frag[mma_q, mma_kv, 3], v_scale_bf2)

        v_row = row_base + warp_kv_idx * num_mma_kv * 16 + mma_kv * 16 + lane % 16
        for mma_d in cutlass.range_constexpr(num_mma_d_vo):
            plane_idx = mma_d // 4
            mma_d_local = mma_d - plane_idx * 4
            if const_expr(plane_idx == 0):
                v_plane_base_addr = v_plane0_base_addr
            elif const_expr(plane_idx == 1):
                v_plane_base_addr = v_plane1_base_addr
            elif const_expr(plane_idx == 2):
                v_plane_base_addr = v_plane2_base_addr
            else:
                v_plane_base_addr = v_plane3_base_addr
            v_col = mma_d_local * 2 + lane // 16
            v_offset = _permuted_offset_128b(v_row, v_col, upcast_stride_plane)
            b0, b1, b2, b3 = ldmatrix_m8n8x4_trans_b16(
                _smem_addr_from_b128_offset(v_plane_base_addr, v_offset)
            )
            if const_expr(debug_regs is not None):
                lane_words = num_mma_kv * num_mma_d_vo * 4
                dst_words = cute.size(debug_regs.shape)
                dst_idx = lane * lane_words + (mma_kv * num_mma_d_vo + mma_d) * 4
                if dst_idx + 0 < dst_words:
                    debug_regs[dst_idx + 0] = b0
                if dst_idx + 1 < dst_words:
                    debug_regs[dst_idx + 1] = b1
                if dst_idx + 2 < dst_words:
                    debug_regs[dst_idx + 2] = b2
                if dst_idx + 3 < dst_words:
                    debug_regs[dst_idx + 3] = b3
            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    o_frag[mma_q, mma_d, 0],
                    o_frag[mma_q, mma_d, 1],
                    o_frag[mma_q, mma_d, 2],
                    o_frag[mma_q, mma_d, 3],
                    o_frag[mma_q, mma_d, 4],
                    o_frag[mma_q, mma_d, 5],
                    o_frag[mma_q, mma_d, 6],
                    o_frag[mma_q, mma_d, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0,
                    b1,
                    b2,
                    b3,
                )
                o_frag[mma_q, mma_d, 0] = d0
                o_frag[mma_q, mma_d, 1] = d1
                o_frag[mma_q, mma_d, 2] = d2
                o_frag[mma_q, mma_d, 3] = d3
                o_frag[mma_q, mma_d, 4] = d4
                o_frag[mma_q, mma_d, 5] = d5
                o_frag[mma_q, mma_d, 6] = d6
                o_frag[mma_q, mma_d, 7] = d7


@cute.jit
def _literal_pv_mma_into_ofrag_tma_bf16_copyfrag(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    tOrVt: cute.Tensor,
    tOsVt: cute.Tensor,
    smem_thr_copy_V: cute.TiledCopy,
    num_mma_q,
    num_mma_d_vo,
    v_scale,
):
    v_scale_bf2 = broadcast_f32_to_bfloat2(v_scale)
    tCrV_copy_view = smem_thr_copy_V.retile(tOrVt)
    cute.copy(smem_thr_copy_V, tOsVt[None, None, 0], tCrV_copy_view[None, None, 0])
    for mma_d in cutlass.range_constexpr(num_mma_d_vo):
        if const_expr(mma_d < num_mma_d_vo - 1):
            cute.copy(
                smem_thr_copy_V,
                tOsVt[None, None, mma_d + 1],
                tCrV_copy_view[None, None, mma_d + 1],
            )
        b_regs = cute.flatten(cute.recast_tensor(tOrVt[None, None, mma_d], Uint32))
        b0 = b_regs[0]
        b1 = b_regs[1]
        b2 = b_regs[2]
        b3 = b_regs[3]
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0 = bfloat2_mul(p_frag[mma_q, 0, 0], v_scale_bf2)
            a1 = bfloat2_mul(p_frag[mma_q, 0, 1], v_scale_bf2)
            a2 = bfloat2_mul(p_frag[mma_q, 0, 2], v_scale_bf2)
            a3 = bfloat2_mul(p_frag[mma_q, 0, 3], v_scale_bf2)
            d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                o_frag[mma_q, mma_d, 0],
                o_frag[mma_q, mma_d, 1],
                o_frag[mma_q, mma_d, 2],
                o_frag[mma_q, mma_d, 3],
                o_frag[mma_q, mma_d, 4],
                o_frag[mma_q, mma_d, 5],
                o_frag[mma_q, mma_d, 6],
                o_frag[mma_q, mma_d, 7],
                a0,
                a1,
                a2,
                a3,
                b0,
                b1,
                b2,
                b3,
            )
            o_frag[mma_q, mma_d, 0] = d0
            o_frag[mma_q, mma_d, 1] = d1
            o_frag[mma_q, mma_d, 2] = d2
            o_frag[mma_q, mma_d, 3] = d3
            o_frag[mma_q, mma_d, 4] = d4
            o_frag[mma_q, mma_d, 5] = d5
            o_frag[mma_q, mma_d, 6] = d6
            o_frag[mma_q, mma_d, 7] = d7


@cute.jit
def _literal_pv_mma_into_ofrag_fp8_raw(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    lane,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_vo,
    upcast_stride_v,
    v_scale,
    debug_regs: cute.Tensor | None = None,
):
    v_scale_bf2 = broadcast_f32_to_bfloat2(v_scale)
    v_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + lane % 16,
        lane // 16,
        upcast_stride_v,
    )
    for mma_kv in cutlass.range_constexpr(num_mma_kv):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a_regs[mma_q, 0] = bfloat2_mul(p_frag[mma_q, mma_kv, 0], v_scale_bf2)
            a_regs[mma_q, 1] = bfloat2_mul(p_frag[mma_q, mma_kv, 1], v_scale_bf2)
            a_regs[mma_q, 2] = bfloat2_mul(p_frag[mma_q, mma_kv, 2], v_scale_bf2)
            a_regs[mma_q, 3] = bfloat2_mul(p_frag[mma_q, mma_kv, 3], v_scale_bf2)

        v_offset_cur = v_offset
        for mma_d in cutlass.range_constexpr(num_mma_d_vo):
            if const_expr(mma_d % 2 == 0):
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_trans_left_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_cur)
                )
            else:
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_trans_right_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_cur)
                )
            b_f8_0 = frag_layout_swizzle_16b_to_8b_trans(b_f8_0)
            b_f8_1 = frag_layout_swizzle_16b_to_8b_trans(b_f8_1)
            b0, b1 = fp8x4_e4m3_to_bfloat2x2(b_f8_0)
            b2, b3 = fp8x4_e4m3_to_bfloat2x2(b_f8_1)
            tmp = b1
            b1 = b2
            b2 = tmp
            if const_expr(debug_regs is not None):
                lane_words = num_mma_kv * num_mma_d_vo * 4
                dst_words = cute.size(debug_regs.shape)
                dst_idx = lane * lane_words + (mma_kv * num_mma_d_vo + mma_d) * 4
                if dst_idx + 0 < dst_words:
                    debug_regs[dst_idx + 0] = b0
                if dst_idx + 1 < dst_words:
                    debug_regs[dst_idx + 1] = b1
                if dst_idx + 2 < dst_words:
                    debug_regs[dst_idx + 2] = b2
                if dst_idx + 3 < dst_words:
                    debug_regs[dst_idx + 3] = b3
            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    o_frag[mma_q, mma_d, 0],
                    o_frag[mma_q, mma_d, 1],
                    o_frag[mma_q, mma_d, 2],
                    o_frag[mma_q, mma_d, 3],
                    o_frag[mma_q, mma_d, 4],
                    o_frag[mma_q, mma_d, 5],
                    o_frag[mma_q, mma_d, 6],
                    o_frag[mma_q, mma_d, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0,
                    b1,
                    b2,
                    b3,
                )
                o_frag[mma_q, mma_d, 0] = d0
                o_frag[mma_q, mma_d, 1] = d1
                o_frag[mma_q, mma_d, 2] = d2
                o_frag[mma_q, mma_d, 3] = d3
                o_frag[mma_q, mma_d, 4] = d4
                o_frag[mma_q, mma_d, 5] = d5
                o_frag[mma_q, mma_d, 6] = d6
                o_frag[mma_q, mma_d, 7] = d7
            if const_expr(mma_d % 2 == 1):
                v_offset_cur = _advance_offset_by_column_128b_2(
                    v_offset_cur, mma_d // 2
                )
        v_offset = _advance_offset_by_row_128b(
            v_offset_cur, 16, upcast_stride_v
        ) - Int32(num_mma_d_vo)
    v_offset -= Int32(16 * num_mma_kv * upcast_stride_v)


@cute.jit
def _literal_pv_mma_into_ofrag_plane_fp8_raw(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_plane0_base_addr: Int32,
    v_plane1_base_addr: Int32,
    lane,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_vo,
    upcast_stride_plane,
    v_scale,
    debug_regs: cute.Tensor | None = None,
):
    v_scale_bf2 = broadcast_f32_to_bfloat2(v_scale)
    upcast_stride_full = upcast_stride_plane * Int32(2)
    v_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + lane % 16,
        lane // 16,
        upcast_stride_full,
    )
    for mma_kv in cutlass.range_constexpr(num_mma_kv):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a_regs[mma_q, 0] = bfloat2_mul(p_frag[mma_q, mma_kv, 0], v_scale_bf2)
            a_regs[mma_q, 1] = bfloat2_mul(p_frag[mma_q, mma_kv, 1], v_scale_bf2)
            a_regs[mma_q, 2] = bfloat2_mul(p_frag[mma_q, mma_kv, 2], v_scale_bf2)
            a_regs[mma_q, 3] = bfloat2_mul(p_frag[mma_q, mma_kv, 3], v_scale_bf2)

        v_offset_cur = v_offset
        for mma_d in cutlass.range_constexpr(num_mma_d_vo):
            v_addr = _smem_addr_from_split_planes_128b(
                v_plane0_base_addr,
                v_plane1_base_addr,
                v_offset_cur,
                upcast_stride_full,
            )
            if const_expr(mma_d % 2 == 0):
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_trans_left_half_b16(v_addr)
            else:
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_trans_right_half_b16(v_addr)
            b_f8_0 = frag_layout_swizzle_16b_to_8b_trans(b_f8_0)
            b_f8_1 = frag_layout_swizzle_16b_to_8b_trans(b_f8_1)
            b0, b1 = fp8x4_e4m3_to_bfloat2x2(b_f8_0)
            b2, b3 = fp8x4_e4m3_to_bfloat2x2(b_f8_1)
            tmp = b1
            b1 = b2
            b2 = tmp
            if const_expr(debug_regs is not None):
                lane_words = num_mma_kv * num_mma_d_vo * 4
                dst_words = cute.size(debug_regs.shape)
                dst_idx = lane * lane_words + (mma_kv * num_mma_d_vo + mma_d) * 4
                if dst_idx + 0 < dst_words:
                    debug_regs[dst_idx + 0] = b0
                if dst_idx + 1 < dst_words:
                    debug_regs[dst_idx + 1] = b1
                if dst_idx + 2 < dst_words:
                    debug_regs[dst_idx + 2] = b2
                if dst_idx + 3 < dst_words:
                    debug_regs[dst_idx + 3] = b3
            if const_expr(mma_d % 2 == 1):
                v_offset_cur = _advance_offset_by_column_128b_2(
                    v_offset_cur, mma_d // 2
                )
            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    o_frag[mma_q, mma_d, 0],
                    o_frag[mma_q, mma_d, 1],
                    o_frag[mma_q, mma_d, 2],
                    o_frag[mma_q, mma_d, 3],
                    o_frag[mma_q, mma_d, 4],
                    o_frag[mma_q, mma_d, 5],
                    o_frag[mma_q, mma_d, 6],
                    o_frag[mma_q, mma_d, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0,
                    b1,
                    b2,
                    b3,
                )
                o_frag[mma_q, mma_d, 0] = d0
                o_frag[mma_q, mma_d, 1] = d1
                o_frag[mma_q, mma_d, 2] = d2
                o_frag[mma_q, mma_d, 3] = d3
                o_frag[mma_q, mma_d, 4] = d4
                o_frag[mma_q, mma_d, 5] = d5
                o_frag[mma_q, mma_d, 6] = d6
                o_frag[mma_q, mma_d, 7] = d7
        v_offset = _advance_offset_by_row_128b(
            v_offset_cur, 16, upcast_stride_full
        ) - Int32(num_mma_d_vo)
    v_offset -= Int32(16 * num_mma_kv * upcast_stride_full)


@cute.jit
def _literal_pv_mma_into_ofrag_mxfp8_raw(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    lane,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_vo,
    upcast_stride_v,
    v_scale,
):
    unit_scale = Uint32(0x7F7F7F7F)
    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)
    v_scale_bf2 = broadcast_f32_to_bfloat2(v_scale)
    v_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + lane % 16,
        lane // 16,
        upcast_stride_v,
    )
    for mma_pair in cutlass.range_constexpr(num_mma_kv // 2):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        mma_kv0 = mma_pair * 2
        mma_kv1 = mma_kv0 + 1
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a_regs[mma_q, 0] = (
                cvt_bf16x2_to_e4m3x2(
                    bfloat2_mul(p_frag[mma_q, mma_kv0, 0], v_scale_bf2)
                )
                & mask16
            ) | (
                (
                    cvt_bf16x2_to_e4m3x2(
                        bfloat2_mul(p_frag[mma_q, mma_kv1, 0], v_scale_bf2)
                    )
                    & mask16
                )
                << shift16
            )
            a_regs[mma_q, 1] = (
                cvt_bf16x2_to_e4m3x2(
                    bfloat2_mul(p_frag[mma_q, mma_kv0, 1], v_scale_bf2)
                )
                & mask16
            ) | (
                (
                    cvt_bf16x2_to_e4m3x2(
                        bfloat2_mul(p_frag[mma_q, mma_kv1, 1], v_scale_bf2)
                    )
                    & mask16
                )
                << shift16
            )
            a_regs[mma_q, 2] = (
                cvt_bf16x2_to_e4m3x2(
                    bfloat2_mul(p_frag[mma_q, mma_kv0, 2], v_scale_bf2)
                )
                & mask16
            ) | (
                (
                    cvt_bf16x2_to_e4m3x2(
                        bfloat2_mul(p_frag[mma_q, mma_kv1, 2], v_scale_bf2)
                    )
                    & mask16
                )
                << shift16
            )
            a_regs[mma_q, 3] = (
                cvt_bf16x2_to_e4m3x2(
                    bfloat2_mul(p_frag[mma_q, mma_kv0, 3], v_scale_bf2)
                )
                & mask16
            ) | (
                (
                    cvt_bf16x2_to_e4m3x2(
                        bfloat2_mul(p_frag[mma_q, mma_kv1, 3], v_scale_bf2)
                    )
                    & mask16
                )
                << shift16
            )

        v_offset_k0 = v_offset
        v_offset_k1 = _advance_offset_by_row_128b(v_offset, 16, upcast_stride_v)
        for mma_d in cutlass.range_constexpr(num_mma_d_vo):
            if const_expr(mma_d % 2 == 0):
                b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_left_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
                )
                b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_left_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
                )
            else:
                b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_right_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
                )
                b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_right_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
                )
            b0_k0 = frag_layout_swizzle_16b_to_8b_trans(b0_k0)
            b1_k0 = frag_layout_swizzle_16b_to_8b_trans(b1_k0)
            b0_k1 = frag_layout_swizzle_16b_to_8b_trans(b0_k1)
            b1_k1 = frag_layout_swizzle_16b_to_8b_trans(b1_k1)

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    o_frag[mma_q, mma_d, 0],
                    o_frag[mma_q, mma_d, 1],
                    o_frag[mma_q, mma_d, 2],
                    o_frag[mma_q, mma_d, 3],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0_k0,
                    b0_k1,
                    unit_scale,
                    unit_scale,
                )
                d4, d5, d6, d7 = mxfp8_mma_m16n8k32_f32_e4m3(
                    o_frag[mma_q, mma_d, 4],
                    o_frag[mma_q, mma_d, 5],
                    o_frag[mma_q, mma_d, 6],
                    o_frag[mma_q, mma_d, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b1_k0,
                    b1_k1,
                    unit_scale,
                    unit_scale,
                )
                o_frag[mma_q, mma_d, 0] = d0
                o_frag[mma_q, mma_d, 1] = d1
                o_frag[mma_q, mma_d, 2] = d2
                o_frag[mma_q, mma_d, 3] = d3
                o_frag[mma_q, mma_d, 4] = d4
                o_frag[mma_q, mma_d, 5] = d5
                o_frag[mma_q, mma_d, 6] = d6
                o_frag[mma_q, mma_d, 7] = d7
            if const_expr(mma_d % 2 == 1):
                v_offset_k0 = _advance_offset_by_column_128b_2(v_offset_k0, mma_d // 2)
                v_offset_k1 = _advance_offset_by_column_128b_2(v_offset_k1, mma_d // 2)

        v_offset = _advance_offset_by_row_128b(v_offset, 32, upcast_stride_v)


@cute.jit
def _literal_update_mdo_states_fp32_pack_p(
    s_frag: cute.Tensor,
    o_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    p_frag: cute.Tensor,
    sm_scale_log2: Float32,
    num_mma_q,
    num_mma_kv,
    num_mma_d_vo,
    p_frag_scalar: cute.Tensor | None = None,
):
    for mma_q in cutlass.range_constexpr(num_mma_q):
        for row_slot in cutlass.range_constexpr(2):
            m_prev = Float32(m_frag[mma_q, row_slot])
            m_new = Float32(m_prev)
            for mma_kv in cutlass.range_constexpr(num_mma_kv):
                m_local = attention_ops.fmax(
                    attention_ops.fmax(
                        s_frag[mma_q, mma_kv, row_slot * 2 + 0],
                        s_frag[mma_q, mma_kv, row_slot * 2 + 1],
                    ),
                    attention_ops.fmax(
                        s_frag[mma_q, mma_kv, row_slot * 2 + 4],
                        s_frag[mma_q, mma_kv, row_slot * 2 + 5],
                    ),
                )
                m_new = attention_ops.fmax(m_new, m_local)
            m_new = attention_ops.fmax(
                m_new, cute.arch.shuffle_sync_bfly(m_new, offset=2)
            )
            m_new = attention_ops.fmax(
                m_new, cute.arch.shuffle_sync_bfly(m_new, offset=1)
            )

            scale_term = (
                Float32(1.0)
                if m_new == -Float32.inf
                else _exp2_approx_ftz_f32((m_prev - m_new) * sm_scale_log2)
            )
            d_frag[mma_q, row_slot] = Float32(d_frag[mma_q, row_slot] * scale_term)
            for mma_d in cutlass.range_constexpr(num_mma_d_vo):
                o_frag[mma_q, mma_d, row_slot * 2 + 0] *= scale_term
                o_frag[mma_q, mma_d, row_slot * 2 + 1] *= scale_term
                o_frag[mma_q, mma_d, row_slot * 2 + 4] *= scale_term
                o_frag[mma_q, mma_d, row_slot * 2 + 5] *= scale_term

            for mma_kv in cutlass.range_constexpr(num_mma_kv):
                p0 = (
                    Float32(0.0)
                    if m_new == -Float32.inf
                    else _exp2_approx_ftz_f32(
                        (s_frag[mma_q, mma_kv, row_slot * 2 + 0] - m_new)
                        * sm_scale_log2
                    )
                )
                p1 = (
                    Float32(0.0)
                    if m_new == -Float32.inf
                    else _exp2_approx_ftz_f32(
                        (s_frag[mma_q, mma_kv, row_slot * 2 + 1] - m_new)
                        * sm_scale_log2
                    )
                )
                p2 = (
                    Float32(0.0)
                    if m_new == -Float32.inf
                    else _exp2_approx_ftz_f32(
                        (s_frag[mma_q, mma_kv, row_slot * 2 + 4] - m_new)
                        * sm_scale_log2
                    )
                )
                p3 = (
                    Float32(0.0)
                    if m_new == -Float32.inf
                    else _exp2_approx_ftz_f32(
                        (s_frag[mma_q, mma_kv, row_slot * 2 + 5] - m_new)
                        * sm_scale_log2
                    )
                )
                p_frag[mma_q, mma_kv, row_slot + 0] = pack_f32x2_to_bfloat2(p0, p1)
                p_frag[mma_q, mma_kv, row_slot + 2] = pack_f32x2_to_bfloat2(p2, p3)
                if const_expr(p_frag_scalar is not None):
                    p_frag_scalar[mma_q, mma_kv, row_slot * 2 + 0] = cutlass.BFloat16(
                        p0
                    )
                    p_frag_scalar[mma_q, mma_kv, row_slot * 2 + 1] = cutlass.BFloat16(
                        p1
                    )
                    p_frag_scalar[mma_q, mma_kv, row_slot * 2 + 4] = cutlass.BFloat16(
                        p2
                    )
                    p_frag_scalar[mma_q, mma_kv, row_slot * 2 + 5] = cutlass.BFloat16(
                        p3
                    )

            m_frag[mma_q, row_slot] = Float32(m_new)


class PagedForwardKernel:
    def __init__(
        self,
        dtype_q: Type[cutlass.Numeric],
        dtype_kv: Type[cutlass.Numeric],
        dtype_kv_storage: Type[cutlass.Numeric],
        dtype_o: Type[cutlass.Numeric],
        *,
        traits: PagedForwardTraits,
        use_native_fp8_qk: bool = False,
        use_native_fp8_pv: bool = False,
        enable_paged_kv_tma: bool = False,
        window_left: int = -1,
        has_attention_sink_bias: bool = False,
        has_relative_attention_bias: bool = False,
        msa_block_sparse: bool = False,
        msa_union_tile: bool = False,
        page_size: int = 64,
    ):
        self.dtype_q = dtype_q
        self.dtype_kv = dtype_kv
        self.dtype_kv_storage = dtype_kv_storage
        self.dtype_o = dtype_o
        self.traits = traits
        self.split_kv = False
        self.window_left = int(window_left)
        self.has_attention_sink_bias = bool(has_attention_sink_bias)
        self.has_relative_attention_bias = bool(has_relative_attention_bias)
        self.msa_block_sparse = bool(msa_block_sparse)
        self.msa_union_tile = bool(msa_union_tile)
        self.MSA_BLOCK_TOKENS = 128
        self.MSA_TOPK = 16
        self.MSA_UNION_TOKENS_PER_TILE = 8
        self.page_size = int(page_size)
        if self.page_size not in (64, 128):
            raise ValueError(
                f"paged extend kernel supports page_size 64 or 128, got {self.page_size}"
            )
        # Number of page-table entries that span one 128-token MSA block.
        self.entries_per_block = self.MSA_BLOCK_TOKENS // self.page_size
        self.kv_is_fp8 = dtype_kv == cutlass.Float8E4M3FN
        self.vec_size = traits.head_dim_vo // 32
        self.total_warps = traits.num_warps_q * traits.num_warps_kv
        self.stage_tile_rows = traits.cta_tile_kv
        q_stage_bytes = traits.cta_tile_q * traits.head_dim_qk * (dtype_q.width // 8)
        kv_stage_bytes = (
            self.stage_tile_rows
            * (traits.head_dim_qk + traits.head_dim_vo)
            * (dtype_kv_storage.width // 8)
        )
        self.num_stages = (
            1
            if traits.num_warps_kv > 1 or self.kv_is_fp8
            else (
                2
                if q_stage_bytes + 2 * kv_stage_bytes <= traits.max_smem_per_threadblock
                else 1
            )
        )
        # The extend TMA path indexes one 64-row source tile per page-table
        # entry. Page-128 uses the byte-addressed copy path until extend TMA
        # gets the decode kernel's page-entry flattening.
        base_use_paged_kv_tma_extend = (
            enable_paged_kv_tma
            and os.environ.get("FLASHINFER_EXP_SM12X_PAGED_KV_TMA", "1") != "0"
            and dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
            and traits.head_dim_qk == 256
            and traits.head_dim_vo == 256
            and self.num_stages == 1
            and traits.num_warps_kv > 1
            and traits.num_warps_q == 1
            and self.stage_tile_rows == 64
            and traits.cta_tile_q == 16
            and traits.num_mma_q == 1
            and traits.num_mma_kv == 1
            and self.page_size == 64
        )
        self.use_paged_kv_tma_exact_plane_bf16_layout = base_use_paged_kv_tma_extend
        self.use_paged_k_tma = self.use_paged_kv_tma_exact_plane_bf16_layout
        self.use_paged_v_tma = self.use_paged_kv_tma_exact_plane_bf16_layout
        self.use_paged_kv_tma = self.use_paged_kv_tma_exact_plane_bf16_layout
        self.use_paged_kv_tma_fp8_raw_issue = False
        tma_debug_dump = os.environ.get(
            "FLASHINFER_EXP_SM12X_PAGED_KV_TMA_DEBUG_DUMP", ""
        )
        paged_debug_dump = os.environ.get(
            "FLASHINFER_EXP_SM12X_PAGED_KV_DEBUG_DUMP", ""
        )
        self.debug_dump_paged_kv_tma_k = self.use_paged_kv_tma and tma_debug_dump == "K"
        self.debug_dump_paged_kv_tma_s = self.use_paged_kv_tma and tma_debug_dump == "S"
        self.debug_dump_paged_kv_tma_v = self.use_paged_kv_tma and tma_debug_dump == "V"
        self.debug_dump_paged_kv_pvregs = (
            paged_debug_dump == "PVREGS"
            and traits.num_warps_kv > 1
            and traits.num_warps_q == 1
            and dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
        )
        self.debug_dump_paged_kv_pregs = (
            paged_debug_dump == "PREGS"
            and traits.num_warps_kv > 1
            and traits.num_warps_q == 1
            and dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
        )
        self.debug_dump_paged_kv_sregs = (
            paged_debug_dump == "SREGS"
            and traits.num_warps_kv > 1
            and traits.num_warps_q == 1
            and dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
        )
        self.debug_dump_paged_kv_svwords = (
            paged_debug_dump == "SVWORDS"
            and traits.num_warps_kv > 1
            and traits.num_warps_q == 1
            and dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
            and not self.kv_is_fp8
        )
        self.debug_dump_paged_kv_planewords = (
            paged_debug_dump == "PLANEWORDS"
            and traits.num_warps_kv > 1
            and traits.num_warps_q == 1
            and dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
            and self.use_paged_v_tma
            and self.use_paged_kv_tma_exact_plane_bf16_layout
        )
        self.debug_dump_paged_kv_extend_state = (
            paged_debug_dump == "EXTEND_STATE"
            and self.kv_is_fp8
            and dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
        )
        self.debug_dump_paged_kv_extend_pregs = (
            paged_debug_dump == "EXTEND_PREGS"
            and self.kv_is_fp8
            and dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
        )
        self.debug_dump_paged_kv_extend_pscalars = (
            paged_debug_dump == "EXTEND_PSCALARS"
            and self.kv_is_fp8
            and dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
        )
        self.debug_dump_paged_kv_extend_partials = (
            paged_debug_dump == "EXTEND_PARTIALS"
            and self.kv_is_fp8
            and dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
            and traits.num_warps_kv > 1
        )
        self.debug_dump_paged_kv_extend_sregs = (
            paged_debug_dump == "EXTEND_SREGS"
            and self.kv_is_fp8
            and dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
        )
        self.debug_dump_paged_kv_extend_kwords = (
            paged_debug_dump == "EXTEND_KWORDS"
            and self.kv_is_fp8
            and traits.num_warps_kv == 1
            and dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
            and self.use_paged_k_tma
        )
        self.debug_dump_paged_kv_extend_vwords = (
            paged_debug_dump == "EXTEND_VWORDS"
            and self.kv_is_fp8
            and traits.num_warps_kv == 1
            and dtype_q == cutlass.BFloat16
            and dtype_o == cutlass.BFloat16
            and self.use_paged_v_tma
        )
        self.kv_tma_plane_head_dim = 128 if self.kv_is_fp8 else 64
        self.kv_tma_plane_mem_dtype = (
            cutlass.Uint8 if self.kv_is_fp8 else self.dtype_kv_storage
        )
        self.kv_tma_internal_type = None
        self.kv_tma_plane_count = (
            (2 if self.kv_is_fp8 else 4)
            if self.use_paged_kv_tma_exact_plane_bf16_layout
            else 1
        )
        self.kv_tma_copy_bytes_k = (
            self.stage_tile_rows * traits.head_dim_qk * (dtype_kv_storage.width // 8)
        )
        self.kv_tma_copy_bytes_v = (
            self.stage_tile_rows * traits.head_dim_vo * (dtype_kv_storage.width // 8)
        )
        self.kv_tma_desc_words_per_head = 16
        self.use_native_fp8_qk_mma = (
            use_native_fp8_qk
            and self.kv_is_fp8
            and dtype_q == cutlass.BFloat16
            and traits.head_dim_qk % 32 == 0
            and traits.num_mma_d_qk % 2 == 0
        )
        self.use_native_fp8_pv_mma = (
            use_native_fp8_pv
            and self.kv_is_fp8
            and dtype_q == cutlass.BFloat16
            and traits.num_warps_kv == 1
            and traits.num_mma_kv % 2 == 0
        )
        self.softmax_scale_log2 = Float32(
            (traits.head_dim_qk**-0.5) * attention_ops.LOG2_E
        )
        self.inverse_softmax_scale = Float32(traits.head_dim_qk**0.5)

    def _get_shared_storage_cls(self):
        class SharedStorage:
            pass

        if self.traits.num_warps_kv > 1:
            if self.use_paged_kv_tma:
                mbar_struct = cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages]
                SharedStorage.__annotations__ = {
                    "mbar_ptr_K": mbar_struct,
                    "mbar_ptr_V": mbar_struct,
                }
            payload_alignment = (
                1024 if self.use_paged_kv_tma_exact_plane_bf16_layout else 128
            )
            payload_struct = cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8,
                    int(self.traits.shared_storage_bytes),
                ],
                payload_alignment,
            ]
            SharedStorage.__annotations__["payload"] = payload_struct
        else:
            q_struct = cute.struct.Align[
                cute.struct.MemRange[
                    self.dtype_q,
                    int(self.traits.cta_tile_q * self.traits.head_dim_qk),
                ],
                128,
            ]
            if self.use_paged_kv_tma:
                payload_bytes = int(
                    self.num_stages
                    * self.stage_tile_rows
                    * (self.traits.upcast_stride_k + self.traits.upcast_stride_v)
                    * 16
                )
                SharedStorage.__annotations__ = {
                    "mbar_ptr_K": cute.struct.MemRange[
                        cutlass.Int64, 2 * self.num_stages
                    ],
                    "mbar_ptr_V": cute.struct.MemRange[
                        cutlass.Int64, 2 * self.num_stages
                    ],
                    "sQ": q_struct,
                    "payload": cute.struct.Align[
                        cute.struct.MemRange[cutlass.Uint8, payload_bytes],
                        1024,
                    ],
                }
            else:
                kv_storage_bytes = self.dtype_kv_storage.width // 8
                k_smem_row_elems = self.traits.upcast_stride_k * 16 // kv_storage_bytes
                v_smem_row_elems = self.traits.upcast_stride_v * 16 // kv_storage_bytes
                k_struct = cute.struct.Align[
                    cute.struct.MemRange[
                        self.dtype_kv_storage,
                        int(self.num_stages * self.stage_tile_rows * k_smem_row_elems),
                    ],
                    128,
                ]
                v_struct = cute.struct.Align[
                    cute.struct.MemRange[
                        self.dtype_kv_storage,
                        int(self.num_stages * self.stage_tile_rows * v_smem_row_elems),
                    ],
                    128,
                ]
                SharedStorage.__annotations__ = {
                    "sQ": q_struct,
                    "sK": k_struct,
                    "sV": v_struct,
                }

        return cute.struct(SharedStorage)

    def _get_paged_kv_tma_plane_layout(self):
        plane_swizzle = os.environ.get(
            "FLASHINFER_EXP_SM12X_PAGED_KV_TMA_PLANE_SWIZZLE", ""
        )
        if plane_swizzle == "none":
            return cute.make_layout(
                (self.stage_tile_rows, self.kv_tma_plane_head_dim),
                stride=(self.kv_tma_plane_head_dim, 1),
            )
        if plane_swizzle:
            mbase, bbits, sshift = [int(part) for part in plane_swizzle.split(",")]
            swizzle = make_swizzle(mbase, bbits, sshift)
        else:
            swizzle = make_swizzle(3, 4, 3)
        return cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (self.stage_tile_rows, self.kv_tma_plane_head_dim),
                stride=(self.kv_tma_plane_head_dim, 1),
            ),
        )

    def _get_paged_kv_tma_plane_stage_layout(self):
        return cute.tile_to_shape(
            self._get_paged_kv_tma_plane_layout(),
            (self.stage_tile_rows, self.kv_tma_plane_head_dim, self.num_stages),
            (0, 1, 2),
        )

    def _get_paged_kv_tma_layout(self, head_dim: int):
        if self.dtype_kv_storage.width == 16:
            layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR,
                    self.dtype_kv_storage,
                    head_dim,
                ),
                self.dtype_kv_storage,
            )
            return cute.tile_to_shape(
                layout_atom,
                (self.stage_tile_rows, head_dim),
                (0, 1),
            )
        swizzle = (
            make_swizzle(3, 4, 4)
            if self.dtype_kv_storage.width == 8
            else make_swizzle(3, 3, 5)
        )
        return cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (self.stage_tile_rows, head_dim),
                stride=(head_dim, 1),
            ),
        )

    def _get_paged_kv_tma_stage_layout(self, head_dim: int):
        return cute.tile_to_shape(
            self._get_paged_kv_tma_layout(head_dim),
            (self.stage_tile_rows, head_dim, self.num_stages),
            (0, 1, 2),
        )

    @cute.jit
    def _msa_selected_token_count(
        self,
        mQ2KIndices: cute.Tensor,
        kv_head_idx,
        q_row_idx,
        visible_len,
    ):
        visible_len = cutlass.select_(visible_len > Int32(0), visible_len, Int32(1))
        block_count = (visible_len + Int32(self.MSA_BLOCK_TOKENS - 1)) // Int32(
            self.MSA_BLOCK_TOKENS
        )
        block_count = cutlass.select_(
            block_count < Int32(self.MSA_TOPK), block_count, Int32(self.MSA_TOPK)
        )
        block_count = cutlass.select_(block_count > Int32(0), block_count, Int32(1))
        if const_expr(self.page_size == 128):
            # Whole-page walk at page_size=128: the planner counts full
            # 128-token pages, so the kernel must not tail-compact.
            return block_count * Int32(self.MSA_BLOCK_TOKENS)
        last_j = block_count - Int32(1)
        local_block_id = mQ2KIndices[kv_head_idx, q_row_idx, last_j]
        tail_tokens = visible_len - local_block_id * Int32(self.MSA_BLOCK_TOKENS)
        tail_pages = (tail_tokens + Int32(63)) // Int32(64)
        tail_pages = cutlass.select_(tail_pages < Int32(1), Int32(1), tail_pages)
        tail_pages = cutlass.select_(tail_pages > Int32(2), Int32(2), tail_pages)
        return ((block_count - Int32(1)) * Int32(2) + tail_pages) * Int32(64)

    @cute.jit
    def _msa_union_selected_token_count(
        self,
        mMSAUnionBlocks: cute.Tensor,
        mMSAUnionCounts: cute.Tensor,
        work_idx,
        kv_head_idx,
        cache_len,
        qo_len,
        tile_first_token,
        tile_token_count,
    ):
        block_count = mMSAUnionCounts[work_idx, kv_head_idx]
        block_count = cutlass.select_(block_count > Int32(0), block_count, Int32(1))
        if const_expr(self.page_size == 128):
            # Whole-page walk at page_size=128 (no tail compaction).
            return block_count * Int32(self.MSA_BLOCK_TOKENS)
        last_block_id = mMSAUnionBlocks[work_idx, kv_head_idx, block_count - Int32(1)]
        max_visible_len = cache_len - qo_len + tile_first_token + tile_token_count
        max_visible_len = cutlass.select_(
            max_visible_len > Int32(0), max_visible_len, Int32(1)
        )
        tail_tokens = max_visible_len - last_block_id * Int32(self.MSA_BLOCK_TOKENS)
        tail_pages = (tail_tokens + Int32(63)) // Int32(64)
        tail_pages = cutlass.select_(tail_pages < Int32(1), Int32(1), tail_pages)
        tail_pages = cutlass.select_(tail_pages > Int32(2), Int32(2), tail_pages)
        return ((block_count - Int32(1)) * Int32(2) + tail_pages) * Int32(64)

    @cute.jit
    def _tile_page_idx(
        self,
        mQ2KIndices: cute.Tensor | None,
        mMSAUnionBlocks: cute.Tensor | None,
        work_idx,
        kv_head_idx,
        q_row_idx,
        tile_token_base,
        page_size,
    ):
        if const_expr(self.msa_block_sparse):
            block_j = tile_token_base // Int32(self.MSA_BLOCK_TOKENS)
            if const_expr(self.entries_per_block == 1):
                block_id = (
                    mMSAUnionBlocks[work_idx, kv_head_idx, block_j]
                    if const_expr(self.msa_union_tile)
                    else mQ2KIndices[kv_head_idx, q_row_idx, block_j]
                )
                return cutlass.select_(
                    block_id >= Int32(0),
                    block_id,
                    Int32(0),
                )
            block_offset = tile_token_base - block_j * Int32(self.MSA_BLOCK_TOKENS)
            subpage = block_offset // page_size
            block_id = (
                mMSAUnionBlocks[work_idx, kv_head_idx, block_j]
                if const_expr(self.msa_union_tile)
                else mQ2KIndices[kv_head_idx, q_row_idx, block_j]
            )
            return cutlass.select_(
                block_id >= Int32(0),
                block_id * Int32(self.entries_per_block) + subpage,
                Int32(0),
            )
        return tile_token_base // page_size

    @cute.jit
    def _tile_key_base(
        self,
        mQ2KIndices: cute.Tensor | None,
        mMSAUnionBlocks: cute.Tensor | None,
        work_idx,
        kv_head_idx,
        q_row_idx,
        tile_token_base,
    ):
        if const_expr(self.msa_block_sparse):
            block_j = tile_token_base // Int32(self.MSA_BLOCK_TOKENS)
            block_offset = tile_token_base - block_j * Int32(self.MSA_BLOCK_TOKENS)
            block_id = (
                mMSAUnionBlocks[work_idx, kv_head_idx, block_j]
                if const_expr(self.msa_union_tile)
                else mQ2KIndices[kv_head_idx, q_row_idx, block_j]
            )
            return cutlass.select_(
                block_id >= Int32(0),
                block_id * Int32(self.MSA_BLOCK_TOKENS) + block_offset,
                Int32(0x7FFFFFFF),
            )
        return tile_token_base

    @cute.jit
    def _msa_union_row_has_block(
        self,
        mMSAUnionMasks: cute.Tensor | None,
        work_idx,
        kv_head_idx,
        tile_token_base,
        q_token_local,
        tile_first_token,
    ):
        if const_expr(self.msa_union_tile):
            block_j = tile_token_base // Int32(self.MSA_BLOCK_TOKENS)
            local_token = q_token_local - tile_first_token
            mask_word = mMSAUnionMasks[work_idx, kv_head_idx, block_j]
            return ((mask_word >> local_token) & Int32(1)) != Int32(0)
        return True

    @cute.jit
    def _async_copy_paged_tile_permuted_128b(
        self,
        mCacheBytes: cute.Tensor,
        mPageTable: cute.Tensor,
        request_idx,
        tile_token_base,
        page_idx,
        kv_head_idx,
        num_kv_heads,
        row_bytes,
        page_stride_bytes,
        token_stride_bytes,
        head_stride_bytes,
        sStageBytes: cute.Tensor,
        stage_byte_offset,
        lane,
        warp_linear_idx,
        valid_rows,
        upcast_stride,
        fill_zero: cutlass.Constexpr,
    ):
        page_size = Int32(self.page_size)
        lane_row = lane // 8
        lane_col = lane % 8
        for tile_iter in cutlass.range_constexpr(
            self.traits.num_mma_kv * 4 // self.traits.num_warps_q
        ):
            row_idx = Int32(
                warp_linear_idx * 4 + lane_row + tile_iter * self.total_warps * 4
            )
            token_idx = Int32(tile_token_base + row_idx)
            page_iter = (
                page_idx
                if const_expr(self.msa_block_sparse)
                else token_idx // page_size
            )
            entry_idx = (
                (tile_token_base - (tile_token_base // page_size) * page_size) + row_idx
                if const_expr(self.msa_block_sparse)
                else token_idx - page_iter * page_size
            )
            page_id = mPageTable[request_idx, page_iter]
            row_valid = row_idx < valid_rows
            row_byte_base = (
                Int64(page_id) * Int64(page_stride_bytes)
                + Int64(entry_idx) * Int64(token_stride_bytes)
                + Int64(kv_head_idx) * Int64(head_stride_bytes)
            )
            for vec_iter in cutlass.range_constexpr((row_bytes + 127) // 128):
                vec_idx = Int32(lane_col + vec_iter * 8)
                src_byte_idx = row_byte_base + vec_idx * 16
                dst_byte_idx = (
                    stage_byte_offset
                    + _permuted_offset_128b(row_idx, vec_idx, upcast_stride) * 16
                )
                vec_valid = row_valid and (vec_idx * Int32(16) < row_bytes)
                if const_expr(fill_zero):
                    _cp_async_load_128b_zfill(
                        shared_ptr_to_u32(sStageBytes.iterator + dst_byte_idx),
                        get_ptr_as_int64(mCacheBytes, src_byte_idx),
                        cutlass.select_(vec_valid, Int32(16), Int32(0)),
                    )
                else:
                    _cp_async_load_128b_pred(
                        shared_ptr_to_u32(sStageBytes.iterator + dst_byte_idx),
                        get_ptr_as_int64(mCacheBytes, src_byte_idx),
                        Int32(vec_valid),
                    )

    @cute.jit
    def _issue_paged_kv_tma_copy_planes(
        self,
        load_tma0,
        load_tma1,
        load_tma2,
        load_tma3,
        pipeline_tma,
        producer_state,
        mPageTable: cute.Tensor,
        request_idx,
        tile_token_base,
        page_size,
    ):
        page_idx = tile_token_base // page_size
        page_id = (
            Int32(0)
            if const_expr(
                os.environ.get("FLASHINFER_EXP_SM12X_PAGED_KV_TMA_FORCE_PAGE0", "0")
                == "1"
            )
            else mPageTable[request_idx, page_idx]
        )
        pipeline_tma.producer_acquire(producer_state)
        load_tma0(src_idx=page_id, producer_state=producer_state)
        load_tma1(src_idx=page_id, producer_state=producer_state)
        load_tma2(src_idx=page_id, producer_state=producer_state)
        load_tma3(src_idx=page_id, producer_state=producer_state)

    @cute.jit
    def _issue_paged_kv_tma_copy_2planes(
        self,
        load_tma0,
        load_tma1,
        pipeline_tma,
        producer_state,
        mPageTable: cute.Tensor,
        request_idx,
        tile_token_base,
        page_size,
    ):
        page_idx = tile_token_base // page_size
        page_id = (
            Int32(0)
            if const_expr(
                os.environ.get("FLASHINFER_EXP_SM12X_PAGED_KV_TMA_FORCE_PAGE0", "0")
                == "1"
            )
            else mPageTable[request_idx, page_idx]
        )
        pipeline_tma.producer_acquire(producer_state)
        load_tma0(src_idx=page_id, producer_state=producer_state)
        load_tma1(src_idx=page_id, producer_state=producer_state)

    @cute.jit
    def _issue_paged_kv_tma_copy_2planes_fp8_raw(
        self,
        mDescPtrsFlat: cute.Tensor,
        kv_head_idx,
        sStageBytes: cute.Tensor,
        stage_plane_offset,
        kv_plane_total_bytes,
        producer_state,
        mbar_ptr,
        expected_bytes,
        mPageTable: cute.Tensor,
        request_idx,
        tile_token_base,
        page_size,
    ):
        _issue_paged_kv_tma_copy_2planes_fp8_raw_impl(
            mDescPtrsFlat,
            kv_head_idx,
            Int32(self.kv_tma_plane_head_dim),
            sStageBytes,
            stage_plane_offset,
            kv_plane_total_bytes,
            producer_state,
            mbar_ptr,
            expected_bytes,
            mPageTable,
            request_idx,
            tile_token_base,
            page_size,
        )

    @cute.jit
    def _async_copy_q_tile_permuted_128b(
        self,
        mQBytes: cute.Tensor,
        q_start,
        packed_tile_start,
        packed_tile_rows,
        kv_head_idx,
        group_size,
        num_q_heads,
        row_bytes,
        token_stride_bytes,
        head_stride_bytes,
        sQBytes: cute.Tensor,
        lane,
        warp_q_idx,
    ):
        lane_row = lane // 8
        lane_col = lane % 8
        warp_row_base = Int32(warp_q_idx * self.traits.num_mma_q * 16)
        for mma_q in cutlass.range_constexpr(self.traits.num_mma_q):
            for row_iter in cutlass.range_constexpr(4):
                packed_q_idx = Int32(
                    packed_tile_start
                    + warp_row_base
                    + mma_q * 16
                    + lane_row
                    + row_iter * 4
                )
                row_valid = packed_q_idx < (packed_tile_start + packed_tile_rows)
                q_row_local = packed_q_idx // group_size
                q_group_lane = packed_q_idx - q_row_local * group_size
                q_head_idx = Int32(kv_head_idx * group_size + q_group_lane)
                q_row_idx = Int32(q_start + q_row_local)
                row_byte_base = Int64(q_row_idx) * Int64(token_stride_bytes) + Int64(
                    q_head_idx
                ) * Int64(head_stride_bytes)
                row_idx = Int32(warp_row_base + mma_q * 16 + lane_row + row_iter * 4)
                for mma_do in cutlass.range_constexpr(self.traits.num_mma_d_qk // 4):
                    vec_idx = Int32(lane_col + mma_do * 8)
                    src_byte_idx = row_byte_base + vec_idx * 16
                    dst_byte_idx = (
                        _permuted_offset_128b(
                            row_idx, vec_idx, self.traits.upcast_stride_q
                        )
                        * 16
                    )
                    _cp_async_load_128b_pred(
                        shared_ptr_to_u32(sQBytes.iterator + dst_byte_idx),
                        get_ptr_as_int64(mQBytes, src_byte_idx),
                        Int32(row_valid),
                    )

    @staticmethod
    def can_implement(
        dtype_q: Type[cutlass.Numeric],
        dtype_kv: Type[cutlass.Numeric],
        dtype_kv_storage: Type[cutlass.Numeric],
        dtype_o: Type[cutlass.Numeric],
        *,
        traits: PagedForwardTraits,
        split_kv: bool,
    ) -> bool:
        del split_kv
        if dtype_q not in (cutlass.Float16, cutlass.BFloat16):
            return False
        if dtype_kv not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN):
            return False
        if dtype_kv_storage not in (cutlass.Float16, cutlass.BFloat16, cutlass.Uint8):
            return False
        if dtype_o not in (cutlass.Float16, cutlass.BFloat16):
            return False
        if traits.head_dim_qk % 32 != 0 or traits.head_dim_vo % 32 != 0:
            return False
        if traits.num_threads != 128:
            return False
        if traits.cta_tile_q not in (16, 64, 128):
            return False
        return True

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mVCache: cute.Tensor,
        mPageTable: cute.Tensor,
        mCacheSeqlens: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mRequestIndices: cute.Tensor,
        mQoTileIndices: cute.Tensor,
        mKvTileIndices: cute.Tensor,
        mOIndptr: cute.Tensor,
        mKvChunkSizePtr: cute.Tensor,
        mKvWindowStartTokens: cute.Tensor,
        mBlockValidMask: cute.Tensor,
        mQ2KIndices: cute.Tensor | None,
        mMSAUnionBlocks: cute.Tensor | None,
        mMSAUnionMasks: cute.Tensor | None,
        mMSAUnionCounts: cute.Tensor | None,
        mAttentionSinkBias: cute.Tensor,
        mRelativeAttentionBias: cute.Tensor | None,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mKDescale: cute.Tensor | None,
        mVDescale: cute.Tensor | None,
        mKTmaDescPtrs: cute.Tensor,
        mVTmaDescPtrs: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if const_expr(len(mQ.shape) != 3):
            raise ValueError("mQ must have shape (total_q, q_heads, head_dim)")
        if const_expr(len(mKCache.shape) != 4 or len(mVCache.shape) != 4):
            raise ValueError(
                "mKCache and mVCache must have shape (num_pages, page_size, kv_heads, head_dim)"
            )
        if const_expr(len(mPageTable.shape) != 2):
            raise ValueError("mPageTable must have shape (batch, max_pages)")
        if const_expr(len(mCacheSeqlens.shape) != 1 or len(mCuSeqlensQ.shape) != 1):
            raise ValueError("mCacheSeqlens and mCuSeqlensQ must be rank-1")
        if const_expr(
            len(mRequestIndices.shape) != 1
            or len(mQoTileIndices.shape) != 1
            or len(mKvTileIndices.shape) != 1
        ):
            raise ValueError("worklist tensors must be rank-1")
        if const_expr(
            len(mOIndptr.shape) != 1
            or len(mKvChunkSizePtr.shape) != 1
            or len(mKvWindowStartTokens.shape) != 1
            or len(mBlockValidMask.shape) != 1
        ):
            raise ValueError(
                "mOIndptr, mKvChunkSizePtr, mKvWindowStartTokens, and mBlockValidMask must be rank-1"
            )
        if const_expr(len(mO.shape) != 3 or len(mLSE.shape) != 2):
            raise ValueError("mO must be rank-3 and mLSE must be rank-2")
        if const_expr(mKDescale is not None and len(mKDescale.shape) not in (1, 2)):
            raise ValueError("mKDescale must have shape (batch,) or (batch, kv_heads)")
        if const_expr(mVDescale is not None and len(mVDescale.shape) not in (1, 2)):
            raise ValueError("mVDescale must have shape (batch,) or (batch, kv_heads)")
        if const_expr(self.msa_block_sparse and mQ2KIndices is None):
            raise ValueError("MSA block-sparse paged extend requires mQ2KIndices")
        if const_expr(
            self.has_relative_attention_bias and len(mRelativeAttentionBias.shape) != 3
        ):
            raise ValueError(
                "mRelativeAttentionBias must have shape "
                "(total_q_capacity, q_heads, relative_extent)"
            )
        if const_expr(
            self.msa_union_tile
            and (
                mMSAUnionBlocks is None
                or mMSAUnionMasks is None
                or mMSAUnionCounts is None
            )
        ):
            raise ValueError(
                "MSA union-tile paged extend requires union metadata tensors"
            )
        if const_expr(self.msa_block_sparse and self.window_left >= 0):
            raise ValueError(
                "MSA block-sparse paged extend does not support window_left"
            )
        if const_expr(
            self.msa_block_sparse
            and (
                self.traits.head_dim_qk != 128
                or self.traits.head_dim_vo != 128
                or (
                    self.traits.cta_tile_q != 16
                    and not (self.msa_union_tile and self.traits.cta_tile_q == 128)
                )
                or self.stage_tile_rows > 64
                or 64 % self.stage_tile_rows != 0
            )
        ):
            raise ValueError(
                "MSA block-sparse extend requires cta_tile_q=16 or union cta_tile_q=128, page_size in (64, 128), cta_tile_kv dividing 64, and head_dim=128"
            )
        if const_expr(mQ.element_type != self.dtype_q):
            raise TypeError("mQ dtype must match dtype_q")
        if const_expr(
            mKCache.element_type != self.dtype_kv_storage
            or mVCache.element_type != self.dtype_kv_storage
        ):
            raise TypeError("mKCache/mVCache dtype must match dtype_kv_storage")
        if const_expr(mO.element_type != self.dtype_o):
            raise TypeError("mO dtype must match dtype_o")
        if const_expr(mLSE.element_type != Float32):
            raise TypeError("mLSE must be Float32")
        if const_expr(
            not self.can_implement(
                self.dtype_q,
                self.dtype_kv,
                self.dtype_kv_storage,
                self.dtype_o,
                traits=self.traits,
                split_kv=self.split_kv,
            )
        ):
            raise TypeError("paged forward kernel configuration is not supported")

        mQ = _assume_tensor_aligned(mQ)
        mKCache = _assume_tensor_aligned(mKCache)
        mVCache = _assume_tensor_aligned(mVCache)
        mO = _assume_tensor_aligned(mO)

        mKCacheT = cute.make_tensor(
            mKCache.iterator, cute.select(mKCache.layout, mode=[1, 3, 2, 0])
        )
        mVCacheT = cute.make_tensor(
            mVCache.iterator, cute.select(mVCache.layout, mode=[1, 3, 2, 0])
        )
        if const_expr(self.use_paged_k_tma):
            mKCacheT = _assume_paged_kv_tma_source_aligned(mKCacheT)
        if const_expr(self.use_paged_v_tma):
            mVCacheT = _assume_paged_kv_tma_source_aligned(mVCacheT)
        tma_tensor_K = mKCacheT
        tma_tensor_V = mVCacheT
        tma_atom_K = None
        tma_atom_V = None
        if const_expr(
            (self.use_paged_k_tma or self.use_paged_v_tma)
            and not self.use_paged_kv_tma_fp8_raw_issue
        ):
            gmem_tiled_copy_kv = cpasync.CopyBulkTensorTileG2SOp()
            k_tma_source = (
                cute.recast_tensor(mKCacheT, self.kv_tma_plane_mem_dtype)
                if const_expr(self.kv_is_fp8)
                else mKCacheT
            )
            v_tma_source = mVCacheT
            if const_expr(self.kv_is_fp8):
                v_tma_source = cute.recast_tensor(
                    v_tma_source, self.kv_tma_plane_mem_dtype
                )
            if const_expr(self.use_paged_k_tma):
                tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                    gmem_tiled_copy_kv,
                    k_tma_source,
                    self._get_paged_kv_tma_plane_layout(),
                    (self.stage_tile_rows, self.kv_tma_plane_head_dim),
                    1,
                    internal_type=self.kv_tma_internal_type,
                )
            if const_expr(self.use_paged_v_tma):
                tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                    gmem_tiled_copy_kv,
                    v_tma_source,
                    self._get_paged_kv_tma_plane_layout(),
                    (self.stage_tile_rows, self.kv_tma_plane_head_dim),
                    1,
                    internal_type=self.kv_tma_internal_type,
                )

        self.kernel(
            mQ,
            mKCache,
            mVCache,
            tma_tensor_K,
            tma_tensor_V,
            mPageTable,
            mCacheSeqlens,
            mCuSeqlensQ,
            mRequestIndices,
            mQoTileIndices,
            mKvTileIndices,
            mOIndptr,
            mKvChunkSizePtr,
            mKvWindowStartTokens,
            mBlockValidMask,
            mQ2KIndices,
            mMSAUnionBlocks,
            mMSAUnionMasks,
            mMSAUnionCounts,
            mAttentionSinkBias,
            mRelativeAttentionBias,
            mO,
            mLSE,
            mKDescale,
            mVDescale,
            mKTmaDescPtrs,
            mVTmaDescPtrs,
            tma_atom_K,
            tma_atom_V,
        ).launch(
            grid=(mBlockValidMask.shape[0], mKCache.shape[2], 1),
            block=[32, self.traits.num_warps_q, self.traits.num_warps_kv],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mVCache: cute.Tensor,
        mKCacheT: cute.Tensor,
        mVCacheT: cute.Tensor,
        mPageTable: cute.Tensor,
        mCacheSeqlens: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mRequestIndices: cute.Tensor,
        mQoTileIndices: cute.Tensor,
        mKvTileIndices: cute.Tensor,
        mOIndptr: cute.Tensor,
        mKvChunkSizePtr: cute.Tensor,
        mKvWindowStartTokens: cute.Tensor,
        mBlockValidMask: cute.Tensor,
        mQ2KIndices: cute.Tensor | None,
        mMSAUnionBlocks: cute.Tensor | None,
        mMSAUnionMasks: cute.Tensor | None,
        mMSAUnionCounts: cute.Tensor | None,
        mAttentionSinkBias: cute.Tensor,
        mRelativeAttentionBias: cute.Tensor | None,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mKDescale: cute.Tensor | None,
        mVDescale: cute.Tensor | None,
        mKTmaDescPtrs: cute.Tensor,
        mVTmaDescPtrs: cute.Tensor,
        tma_atom_K: cute.CopyAtom | None,
        tma_atom_V: cute.CopyAtom | None,
    ):
        lane, warp_q_idx, warp_kv_idx = cute.arch.thread_idx()
        work_idx, kv_head_idx, _ = cute.arch.block_idx()
        block_valid = mBlockValidMask[work_idx]
        if block_valid == Int32(0):
            _exit_thread()
        valid_work = True
        request_idx = mRequestIndices[work_idx]
        qo_tile_idx = mQoTileIndices[work_idx]
        kv_tile_idx = mKvTileIndices[work_idx]
        q_start = mCuSeqlensQ[request_idx]
        q_end = mCuSeqlensQ[request_idx + 1]
        qo_len = q_end - q_start
        cache_len = mCacheSeqlens[request_idx]
        group_size = mQ.shape[1] // mKCache.shape[2]
        packed_qo_len = qo_len * group_size
        packed_tile_start = qo_tile_idx * self.traits.cta_tile_q
        packed_tile_limit = packed_tile_start + self.traits.cta_tile_q
        packed_tile_end = cutlass.select_(
            packed_tile_limit < packed_qo_len, packed_tile_limit, packed_qo_len
        )
        packed_tile_rows = packed_tile_end - packed_tile_start
        kv_chunk_size = mKvChunkSizePtr[0]

        if const_expr(self.msa_block_sparse):
            kv_window_start = Int32(0)
            msa_tile_first_token = packed_tile_start // group_size
            msa_tile_token_count = (
                packed_tile_rows + group_size - Int32(1)
            ) // group_size
            msa_q_row_idx = q_start + msa_tile_first_token
            msa_visible_len = cache_len - qo_len + msa_tile_first_token + Int32(1)
            msa_tile_max_visible_len = (
                cache_len - qo_len + msa_tile_first_token + msa_tile_token_count
            )
            selected_token_count = (
                self._msa_union_selected_token_count(
                    mMSAUnionBlocks,
                    mMSAUnionCounts,
                    work_idx,
                    kv_head_idx,
                    cache_len,
                    qo_len,
                    msa_tile_first_token,
                    msa_tile_token_count,
                )
                if const_expr(self.msa_union_tile)
                else self._msa_selected_token_count(
                    mQ2KIndices,
                    kv_head_idx,
                    msa_q_row_idx,
                    msa_visible_len,
                )
            )
            chunk_start = (
                kv_tile_idx * kv_chunk_size if const_expr(self.split_kv) else Int32(0)
            )
            chunk_end = (
                cutlass.select_(
                    (kv_tile_idx + 1) * kv_chunk_size < selected_token_count,
                    (kv_tile_idx + 1) * kv_chunk_size,
                    selected_token_count,
                )
                if const_expr(self.split_kv)
                else selected_token_count
            )
        else:
            kv_window_start = (
                mKvWindowStartTokens[request_idx]
                if const_expr(self.window_left >= 0)
                else Int32(0)
            )
            msa_q_row_idx = Int32(0)
            msa_visible_len = Int32(0)
            msa_tile_first_token = Int32(0)
            msa_tile_token_count = Int32(0)
            msa_tile_max_visible_len = Int32(0)
            chunk_start = (
                kv_window_start + kv_tile_idx * kv_chunk_size
                if const_expr(self.split_kv)
                else kv_window_start
            )
            chunk_end = (
                cutlass.select_(
                    kv_window_start + (kv_tile_idx + 1) * kv_chunk_size < cache_len,
                    kv_window_start + (kv_tile_idx + 1) * kv_chunk_size,
                    cache_len,
                )
                if const_expr(self.split_kv)
                else cache_len
            )
        request_partial_start = mOIndptr[request_idx]
        request_partial_end = mOIndptr[request_idx + 1]
        num_chunks_kv = (
            (request_partial_end - request_partial_start) // qo_len
            if const_expr(self.split_kv)
            else 1
        )
        page_size = mKCache.shape[1]
        stage_tile_rows = self.stage_tile_rows
        q_bytes = self.traits.q_smem_bytes
        kv_storage_bytes = self.dtype_kv_storage.width // 8
        k_smem_row_bytes = self.traits.upcast_stride_k * 16
        v_smem_row_bytes = self.traits.upcast_stride_v * 16
        k_smem_row_elems = k_smem_row_bytes // kv_storage_bytes
        v_smem_row_elems = v_smem_row_bytes // kv_storage_bytes
        k_bytes = self.num_stages * stage_tile_rows * k_smem_row_bytes
        v_bytes = self.num_stages * stage_tile_rows * v_smem_row_bytes
        kv_plane_stage_bytes = (
            stage_tile_rows
            * self.kv_tma_plane_head_dim
            * (self.dtype_kv_storage.width // 8)
        )
        kv_plane_total_bytes = self.num_stages * kv_plane_stage_bytes
        warp_linear_idx = warp_kv_idx * self.traits.num_warps_q + warp_q_idx
        tidx = lane + 32 * (warp_q_idx + self.traits.num_warps_q * warp_kv_idx)

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._get_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        if const_expr(
            (self.use_paged_k_tma or self.use_paged_v_tma)
            and not self.use_paged_kv_tma_fp8_raw_issue
        ):
            if warp_q_idx == Int32(0) and warp_kv_idx == Int32(0):
                if const_expr(self.use_paged_k_tma):
                    cpasync.prefetch_descriptor(tma_atom_K)
                if const_expr(self.use_paged_v_tma):
                    cpasync.prefetch_descriptor(tma_atom_V)
        if const_expr(self.traits.num_warps_kv > 1):
            if const_expr(self.use_paged_k_tma):
                mbar_ptr_K = storage.mbar_ptr_K.data_ptr()
            else:
                mbar_ptr_K = None
            if const_expr(self.use_paged_v_tma):
                mbar_ptr_V = storage.mbar_ptr_V.data_ptr()
            else:
                mbar_ptr_V = None
            if const_expr(self.use_paged_kv_tma_fp8_raw_issue):
                if tidx < self.num_stages:
                    if const_expr(self.use_paged_k_tma):
                        cute.arch.mbarrier_init(mbar_ptr_K + tidx, Int32(1))
                    if const_expr(self.use_paged_v_tma):
                        cute.arch.mbarrier_init(mbar_ptr_V + tidx, Int32(1))
                cute.arch.sync_threads()
            payload_u8 = storage.payload.get_tensor(
                cute.make_layout((self.traits.shared_storage_bytes,), stride=(1,))
            )
            sQ = _make_payload_tensor(
                payload_u8,
                self.dtype_q,
                0,
                cute.make_layout(
                    (self.traits.cta_tile_q, self.traits.head_dim_qk),
                    stride=(self.traits.head_dim_qk, 1),
                ),
            )
            sQTile = sQ
            sK = _make_payload_tensor(
                payload_u8,
                self.dtype_kv_storage,
                q_bytes,
                cute.make_layout(
                    (stage_tile_rows, self.traits.head_dim_qk, self.num_stages),
                    stride=(k_smem_row_elems, 1, stage_tile_rows * k_smem_row_elems),
                ),
            )
            sKStageBytes = _make_payload_tensor(
                payload_u8,
                cutlass.Uint8,
                q_bytes,
                cute.make_layout((k_bytes,), stride=(1,)),
            )
            sV = _make_payload_tensor(
                payload_u8,
                self.dtype_kv_storage,
                q_bytes + k_bytes,
                cute.make_layout(
                    (stage_tile_rows, self.traits.head_dim_vo, self.num_stages),
                    stride=(v_smem_row_elems, 1, stage_tile_rows * v_smem_row_elems),
                ),
            )
            sVStageBytes = _make_payload_tensor(
                payload_u8,
                cutlass.Uint8,
                q_bytes + k_bytes,
                cute.make_layout((v_bytes,), stride=(1,)),
            )
            sKTma = None
            sVTma = None
            if const_expr(self.use_paged_kv_tma_exact_plane_bf16_layout):
                sKPlane0 = _get_memrange_tensor(
                    _make_payload_memrange(
                        payload_u8,
                        self.kv_tma_plane_mem_dtype,
                        q_bytes + 0 * kv_plane_total_bytes,
                        self.num_stages * stage_tile_rows * self.kv_tma_plane_head_dim,
                    ),
                    self._get_paged_kv_tma_plane_stage_layout(),
                )
                sKPlane1 = _get_memrange_tensor(
                    _make_payload_memrange(
                        payload_u8,
                        self.kv_tma_plane_mem_dtype,
                        q_bytes + 1 * kv_plane_total_bytes,
                        self.num_stages * stage_tile_rows * self.kv_tma_plane_head_dim,
                    ),
                    self._get_paged_kv_tma_plane_stage_layout(),
                )
                sKPlane2 = (
                    _get_memrange_tensor(
                        _make_payload_memrange(
                            payload_u8,
                            self.kv_tma_plane_mem_dtype,
                            q_bytes + 2 * kv_plane_total_bytes,
                            self.num_stages
                            * stage_tile_rows
                            * self.kv_tma_plane_head_dim,
                        ),
                        self._get_paged_kv_tma_plane_stage_layout(),
                    )
                    if const_expr(self.kv_tma_plane_count > 2)
                    else None
                )
                sKPlane3 = (
                    _get_memrange_tensor(
                        _make_payload_memrange(
                            payload_u8,
                            self.kv_tma_plane_mem_dtype,
                            q_bytes + 3 * kv_plane_total_bytes,
                            self.num_stages
                            * stage_tile_rows
                            * self.kv_tma_plane_head_dim,
                        ),
                        self._get_paged_kv_tma_plane_stage_layout(),
                    )
                    if const_expr(self.kv_tma_plane_count > 2)
                    else None
                )
                sVPlane0 = _get_memrange_tensor(
                    _make_payload_memrange(
                        payload_u8,
                        self.kv_tma_plane_mem_dtype,
                        q_bytes + k_bytes + 0 * kv_plane_total_bytes,
                        self.num_stages * stage_tile_rows * self.kv_tma_plane_head_dim,
                    ),
                    self._get_paged_kv_tma_plane_stage_layout(),
                )
                sVPlane1 = _get_memrange_tensor(
                    _make_payload_memrange(
                        payload_u8,
                        self.kv_tma_plane_mem_dtype,
                        q_bytes + k_bytes + 1 * kv_plane_total_bytes,
                        self.num_stages * stage_tile_rows * self.kv_tma_plane_head_dim,
                    ),
                    self._get_paged_kv_tma_plane_stage_layout(),
                )
                sVPlane2 = (
                    _get_memrange_tensor(
                        _make_payload_memrange(
                            payload_u8,
                            self.kv_tma_plane_mem_dtype,
                            q_bytes + k_bytes + 2 * kv_plane_total_bytes,
                            self.num_stages
                            * stage_tile_rows
                            * self.kv_tma_plane_head_dim,
                        ),
                        self._get_paged_kv_tma_plane_stage_layout(),
                    )
                    if const_expr(self.kv_tma_plane_count > 2)
                    else None
                )
                sVPlane3 = (
                    _get_memrange_tensor(
                        _make_payload_memrange(
                            payload_u8,
                            self.kv_tma_plane_mem_dtype,
                            q_bytes + k_bytes + 3 * kv_plane_total_bytes,
                            self.num_stages
                            * stage_tile_rows
                            * self.kv_tma_plane_head_dim,
                        ),
                        self._get_paged_kv_tma_plane_stage_layout(),
                    )
                    if const_expr(self.kv_tma_plane_count > 2)
                    else None
                )
            else:
                sKPlane0 = None
                sKPlane1 = None
                sKPlane2 = None
                sKPlane3 = None
                sVPlane0 = None
                sVPlane1 = None
                sVPlane2 = None
                sVPlane3 = None
        else:
            sQTile = None
            if const_expr(self.use_paged_kv_tma):
                if const_expr(self.use_paged_k_tma):
                    mbar_ptr_K = storage.mbar_ptr_K.data_ptr()
                else:
                    mbar_ptr_K = None
                if const_expr(self.use_paged_v_tma):
                    mbar_ptr_V = storage.mbar_ptr_V.data_ptr()
                else:
                    mbar_ptr_V = None
                if const_expr(self.use_paged_kv_tma_fp8_raw_issue):
                    if tidx < self.num_stages:
                        if const_expr(self.use_paged_k_tma):
                            cute.arch.mbarrier_init(mbar_ptr_K + tidx, Int32(1))
                        if const_expr(self.use_paged_v_tma):
                            cute.arch.mbarrier_init(mbar_ptr_V + tidx, Int32(1))
                    cute.arch.sync_threads()
                payload_u8 = storage.payload.get_tensor(
                    cute.make_layout((k_bytes + v_bytes,), stride=(1,))
                )
                sKStageBytes = _make_payload_tensor(
                    payload_u8,
                    cutlass.Uint8,
                    0,
                    cute.make_layout((k_bytes,), stride=(1,)),
                )
                sVStageBytes = _make_payload_tensor(
                    payload_u8,
                    cutlass.Uint8,
                    k_bytes,
                    cute.make_layout((v_bytes,), stride=(1,)),
                )
                sK = cute.make_tensor(
                    cute.recast_tensor(sKStageBytes, self.dtype_kv_storage).iterator,
                    cute.make_layout(
                        (stage_tile_rows, self.traits.head_dim_qk, self.num_stages),
                        stride=(
                            k_smem_row_elems,
                            1,
                            stage_tile_rows * k_smem_row_elems,
                        ),
                    ),
                )
                sV = cute.make_tensor(
                    cute.recast_tensor(sVStageBytes, self.dtype_kv_storage).iterator,
                    cute.make_layout(
                        (stage_tile_rows, self.traits.head_dim_vo, self.num_stages),
                        stride=(
                            v_smem_row_elems,
                            1,
                            stage_tile_rows * v_smem_row_elems,
                        ),
                    ),
                )
                sKTma = None
                sVTma = None
                if const_expr(self.use_paged_kv_tma_exact_plane_bf16_layout):
                    sKPlane0 = _get_memrange_tensor(
                        _make_payload_memrange(
                            payload_u8,
                            self.kv_tma_plane_mem_dtype,
                            0 * kv_plane_total_bytes,
                            self.num_stages
                            * stage_tile_rows
                            * self.kv_tma_plane_head_dim,
                        ),
                        self._get_paged_kv_tma_plane_stage_layout(),
                    )
                    sKPlane1 = _get_memrange_tensor(
                        _make_payload_memrange(
                            payload_u8,
                            self.kv_tma_plane_mem_dtype,
                            1 * kv_plane_total_bytes,
                            self.num_stages
                            * stage_tile_rows
                            * self.kv_tma_plane_head_dim,
                        ),
                        self._get_paged_kv_tma_plane_stage_layout(),
                    )
                    sKPlane2 = None
                    sKPlane3 = None
                    sVPlane0 = _get_memrange_tensor(
                        _make_payload_memrange(
                            payload_u8,
                            self.kv_tma_plane_mem_dtype,
                            k_bytes + 0 * kv_plane_total_bytes,
                            self.num_stages
                            * stage_tile_rows
                            * self.kv_tma_plane_head_dim,
                        ),
                        self._get_paged_kv_tma_plane_stage_layout(),
                    )
                    sVPlane1 = _get_memrange_tensor(
                        _make_payload_memrange(
                            payload_u8,
                            self.kv_tma_plane_mem_dtype,
                            k_bytes + 1 * kv_plane_total_bytes,
                            self.num_stages
                            * stage_tile_rows
                            * self.kv_tma_plane_head_dim,
                        ),
                        self._get_paged_kv_tma_plane_stage_layout(),
                    )
                    sVPlane2 = None
                    sVPlane3 = None
                else:
                    sKPlane0 = None
                    sKPlane1 = None
                    sKPlane2 = None
                    sKPlane3 = None
                    sVPlane0 = None
                    sVPlane1 = None
                    sVPlane2 = None
                    sVPlane3 = None
            else:
                mbar_ptr_K = None
                mbar_ptr_V = None
                sK = storage.sK.get_tensor(
                    cute.make_layout(
                        (stage_tile_rows, self.traits.head_dim_qk, self.num_stages),
                        stride=(
                            k_smem_row_elems,
                            1,
                            stage_tile_rows * k_smem_row_elems,
                        ),
                    )
                )
                sV = storage.sV.get_tensor(
                    cute.make_layout(
                        (stage_tile_rows, self.traits.head_dim_vo, self.num_stages),
                        stride=(
                            v_smem_row_elems,
                            1,
                            stage_tile_rows * v_smem_row_elems,
                        ),
                    )
                )
                sKStageBytes = cute.make_tensor(
                    cute.recast_tensor(sK, cutlass.Uint8).iterator,
                    cute.make_layout((k_bytes,), stride=(1,)),
                )
                sVStageBytes = cute.make_tensor(
                    cute.recast_tensor(sV, cutlass.Uint8).iterator,
                    cute.make_layout((v_bytes,), stride=(1,)),
                )
                sKTma = None
                sVTma = None
                sKPlane0 = None
                sKPlane1 = None
                sKPlane2 = None
                sKPlane3 = None
                sVPlane0 = None
                sVPlane1 = None
                sVPlane2 = None
                sVPlane3 = None
        if const_expr(
            (self.use_paged_k_tma or self.use_paged_v_tma)
            and not self.use_paged_kv_tma_fp8_raw_issue
        ):
            pipeline_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, self.total_warps
            )
            pipeline_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread
            )
            pipeline_k = (
                cute_pipeline.PipelineTmaAsync.create(
                    barrier_storage=mbar_ptr_K,
                    num_stages=self.num_stages,
                    producer_group=pipeline_kv_producer_group,
                    consumer_group=pipeline_kv_consumer_group,
                    tx_count=self.kv_tma_copy_bytes_k,
                    defer_sync=True,
                )
                if const_expr(self.use_paged_k_tma)
                else None
            )
            pipeline_v = (
                cute_pipeline.PipelineTmaAsync.create(
                    barrier_storage=mbar_ptr_V,
                    num_stages=self.num_stages,
                    producer_group=pipeline_kv_producer_group,
                    consumer_group=pipeline_kv_consumer_group,
                    tx_count=self.kv_tma_copy_bytes_v,
                    defer_sync=False,
                )
                if const_expr(self.use_paged_v_tma)
                else None
            )
        else:
            pipeline_k = None
            pipeline_v = None
        if const_expr(self.traits.num_warps_kv > 1):
            sQ = cute.make_tensor(
                cute.recast_tensor(
                    cute.make_tensor(
                        payload_u8.iterator,
                        cute.make_layout((q_bytes,), stride=(1,)),
                    ),
                    self.dtype_q,
                ).iterator,
                cute.make_layout(
                    (self.traits.cta_tile_q * self.traits.head_dim_qk,), stride=(1,)
                ),
            )
            sKTC = None
            sVTC = None
        else:
            sQ = storage.sQ.get_tensor(
                cute.make_layout(
                    (self.traits.cta_tile_q * self.traits.head_dim_qk,), stride=(1,)
                )
            )
            sKTC = None
            sVTC = None
        k_row_bytes = self.traits.head_dim_qk * (self.dtype_kv_storage.width // 8)
        v_row_bytes = self.traits.head_dim_vo * (self.dtype_kv_storage.width // 8)
        k_page_stride_bytes = mKCache.stride[0] * kv_storage_bytes
        k_token_stride_bytes = mKCache.stride[1] * kv_storage_bytes
        k_head_stride_bytes = mKCache.stride[2] * kv_storage_bytes
        v_page_stride_bytes = mVCache.stride[0] * kv_storage_bytes
        v_token_stride_bytes = mVCache.stride[1] * kv_storage_bytes
        v_head_stride_bytes = mVCache.stride[2] * kv_storage_bytes
        q_storage_bytes = self.dtype_q.width // 8
        q_token_stride_bytes = mQ.stride[0] * q_storage_bytes
        q_head_stride_bytes = mQ.stride[1] * q_storage_bytes
        k_stage_bytes = stage_tile_rows * k_smem_row_bytes
        v_stage_bytes = stage_tile_rows * v_smem_row_bytes
        mQBytes = cute.flatten(cute.recast_tensor(mQ, cutlass.Uint8))
        mKBytes = cute.flatten(cute.recast_tensor(mKCache, cutlass.Uint8))
        mVBytes = cute.flatten(cute.recast_tensor(mVCache, cutlass.Uint8))
        mKTmaDescFlat = cute.flatten(mKTmaDescPtrs)
        mVTmaDescFlat = cute.flatten(mVTmaDescPtrs)
        if const_expr(self.use_paged_k_tma or self.use_paged_v_tma):
            mKCacheTHead = (
                mKCacheT[None, None, kv_head_idx, None]
                if const_expr(not self.use_paged_kv_tma_fp8_raw_issue)
                else None
            )
            mVCacheTHead = (
                mVCacheT[None, None, kv_head_idx, None]
                if const_expr(not self.use_paged_kv_tma_fp8_raw_issue)
                else None
            )
            if const_expr(
                self.use_paged_k_tma and not self.use_paged_kv_tma_fp8_raw_issue
            ):
                gKTma0 = cute.local_tile(
                    mKCacheTHead,
                    (self.stage_tile_rows, self.kv_tma_plane_head_dim),
                    (0, 0, None),
                )
                gKTma1 = cute.local_tile(
                    mKCacheTHead,
                    (self.stage_tile_rows, self.kv_tma_plane_head_dim),
                    (0, 1, None),
                )
                gKTma2 = (
                    cute.local_tile(
                        mKCacheTHead,
                        (self.stage_tile_rows, self.kv_tma_plane_head_dim),
                        (0, 2, None),
                    )
                    if const_expr(self.kv_tma_plane_count > 2)
                    else None
                )
                gKTma3 = (
                    cute.local_tile(
                        mKCacheTHead,
                        (self.stage_tile_rows, self.kv_tma_plane_head_dim),
                        (0, 3, None),
                    )
                    if const_expr(self.kv_tma_plane_count > 2)
                    else None
                )
                load_K_tma0, _, _ = cute_copy.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gKTma0, sKPlane0
                )
                load_K_tma1, _, _ = cute_copy.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gKTma1, sKPlane1
                )
                load_K_tma2, _, _ = (
                    cute_copy.tma_get_copy_fn(
                        tma_atom_K, 0, cute.make_layout(1), gKTma2, sKPlane2
                    )
                    if const_expr(self.kv_tma_plane_count > 2)
                    else (None, None, None)
                )
                load_K_tma3, _, _ = (
                    cute_copy.tma_get_copy_fn(
                        tma_atom_K, 0, cute.make_layout(1), gKTma3, sKPlane3
                    )
                    if const_expr(self.kv_tma_plane_count > 2)
                    else (None, None, None)
                )
                load_K_tma0 = cute_copy.tma_producer_copy_fn(load_K_tma0, pipeline_k)
                load_K_tma1 = cute_copy.tma_producer_copy_fn(load_K_tma1, pipeline_k)
                load_K_tma2 = (
                    cute_copy.tma_producer_copy_fn(load_K_tma2, pipeline_k)
                    if const_expr(self.kv_tma_plane_count > 2)
                    else None
                )
                load_K_tma3 = (
                    cute_copy.tma_producer_copy_fn(load_K_tma3, pipeline_k)
                    if const_expr(self.kv_tma_plane_count > 2)
                    else None
                )
                load_K_tma = None
            else:
                load_K_tma = None
                load_K_tma0 = None
                load_K_tma1 = None
                load_K_tma2 = None
                load_K_tma3 = None
            if const_expr(
                self.use_paged_v_tma and not self.use_paged_kv_tma_fp8_raw_issue
            ):
                gVTma0 = cute.local_tile(
                    mVCacheTHead,
                    (self.stage_tile_rows, self.kv_tma_plane_head_dim),
                    (0, 0, None),
                )
                gVTma1 = cute.local_tile(
                    mVCacheTHead,
                    (self.stage_tile_rows, self.kv_tma_plane_head_dim),
                    (0, 1, None),
                )
                gVTma2 = (
                    cute.local_tile(
                        mVCacheTHead,
                        (self.stage_tile_rows, self.kv_tma_plane_head_dim),
                        (0, 2, None),
                    )
                    if const_expr(self.kv_tma_plane_count > 2)
                    else None
                )
                gVTma3 = (
                    cute.local_tile(
                        mVCacheTHead,
                        (self.stage_tile_rows, self.kv_tma_plane_head_dim),
                        (0, 3, None),
                    )
                    if const_expr(self.kv_tma_plane_count > 2)
                    else None
                )
                load_V_tma0, _, _ = cute_copy.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gVTma0, sVPlane0
                )
                load_V_tma1, _, _ = cute_copy.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gVTma1, sVPlane1
                )
                load_V_tma2, _, _ = (
                    cute_copy.tma_get_copy_fn(
                        tma_atom_V, 0, cute.make_layout(1), gVTma2, sVPlane2
                    )
                    if const_expr(self.kv_tma_plane_count > 2)
                    else (None, None, None)
                )
                load_V_tma3, _, _ = (
                    cute_copy.tma_get_copy_fn(
                        tma_atom_V, 0, cute.make_layout(1), gVTma3, sVPlane3
                    )
                    if const_expr(self.kv_tma_plane_count > 2)
                    else (None, None, None)
                )
                load_V_tma0 = cute_copy.tma_producer_copy_fn(load_V_tma0, pipeline_v)
                load_V_tma1 = cute_copy.tma_producer_copy_fn(load_V_tma1, pipeline_v)
                load_V_tma2 = (
                    cute_copy.tma_producer_copy_fn(load_V_tma2, pipeline_v)
                    if const_expr(self.kv_tma_plane_count > 2)
                    else None
                )
                load_V_tma3 = (
                    cute_copy.tma_producer_copy_fn(load_V_tma3, pipeline_v)
                    if const_expr(self.kv_tma_plane_count > 2)
                    else None
                )
                load_V_tma = None
            else:
                load_V_tma = None
                load_V_tma0 = None
                load_V_tma1 = None
                load_V_tma2 = None
                load_V_tma3 = None
        sKU8 = (
            sK
            if const_expr(self.kv_is_fp8 and self.dtype_kv_storage == cutlass.Uint8)
            else (
                cute.recast_tensor(sK, cutlass.Uint8)
                if const_expr(self.kv_is_fp8)
                else None
            )
        )
        sVU8 = (
            sV
            if const_expr(self.kv_is_fp8 and self.dtype_kv_storage == cutlass.Uint8)
            else (
                cute.recast_tensor(sV, cutlass.Uint8)
                if const_expr(self.kv_is_fp8)
                else None
            )
        )
        if const_expr(self.traits.num_warps_kv > 1):
            sync_payload = cute.recast_tensor(
                payload_u8,
                Float32,
            )
            sync_o_elems = (
                self.traits.num_warps_kv
                * self.traits.cta_tile_q
                * self.traits.head_dim_vo
            )
            sSyncO = cute.make_tensor(
                sync_payload.iterator,
                cute.make_layout(
                    (
                        self.traits.num_warps_kv,
                        self.traits.cta_tile_q,
                        self.traits.head_dim_vo,
                    ),
                    stride=(
                        self.traits.cta_tile_q * self.traits.head_dim_vo,
                        self.traits.head_dim_vo,
                        1,
                    ),
                ),
            )
            sSyncMD = cute.make_tensor(
                sync_payload.iterator + Int32(sync_o_elems),
                cute.make_layout(
                    (self.traits.num_warps_kv, self.traits.cta_tile_q, 2),
                    stride=(self.traits.cta_tile_q * 2, 2, 1),
                ),
            )
            sFinalStage = cute.make_tensor(
                cute.recast_tensor(sync_payload, self.dtype_o).iterator,
                cute.make_layout(
                    (
                        self.traits.num_warps_kv,
                        self.traits.cta_tile_q,
                        self.traits.head_dim_vo * 2,
                    ),
                    stride=(
                        self.traits.cta_tile_q * self.traits.head_dim_vo * 2,
                        self.traits.head_dim_vo * 2,
                        1,
                    ),
                ),
            )
            sFinalStageU32 = cute.recast_tensor(sFinalStage, cutlass.Uint32)
        else:
            sync_payload = cute.make_tensor(
                cute.recast_tensor(cute.flatten(sQ), Float32).iterator,
                cute.make_layout((4,), stride=(1,)),
            )
            sSyncO = cute.make_tensor(
                sync_payload.iterator,
                cute.make_layout((1, 1, 1), stride=(1, 1, 1)),
            )
            sSyncMD = cute.make_tensor(
                sync_payload.iterator,
                cute.make_layout((1, 1, 2), stride=(2, 2, 1)),
            )
        merged_store_v128 = const_expr(
            self.traits.num_warps_kv > 1 and self.dtype_o == cutlass.BFloat16
        )
        split_store_v128 = const_expr(
            self.split_kv
            and self.traits.num_warps_kv == 1
            and self.dtype_o == cutlass.BFloat16
        )
        final_store_v128 = const_expr(
            not self.split_kv
            and self.traits.num_warps_kv == 1
            and self.dtype_o == cutlass.BFloat16
        )
        sOStage = cute.make_tensor(
            sQ.iterator,
            cute.make_layout(
                (self.traits.cta_tile_q, self.traits.head_dim_vo),
                stride=(self.traits.head_dim_vo, 1),
            ),
        )
        sOStageU32 = cute.recast_tensor(sOStage, cutlass.Uint32)
        mOFlat = cute.flatten(mO)

        tc_upcast_elems_qk = 16 // (self.dtype_q.width // 8)
        tc_upcast_stride_qk = self.traits.head_dim_qk // tc_upcast_elems_qk
        tc_upcast_elems_vo = 16 // (self.dtype_q.width // 8)
        tc_upcast_elems_plane = 16 // (self.dtype_kv_storage.width // 8)
        tc_upcast_stride_vo = self.traits.head_dim_vo // tc_upcast_elems_vo
        tc_upcast_stride_plane = self.kv_tma_plane_head_dim // tc_upcast_elems_plane
        if const_expr(self.traits.num_warps_kv > 1):
            sQBytes = cute.flatten(cute.recast_tensor(sQ, cutlass.Uint8))
            if warp_kv_idx == Int32(0):
                self._async_copy_q_tile_permuted_128b(
                    mQBytes,
                    q_start,
                    packed_tile_start,
                    packed_tile_rows,
                    kv_head_idx,
                    group_size,
                    mQ.shape[1],
                    self.traits.head_dim_qk * (self.dtype_q.width // 8),
                    q_token_stride_bytes,
                    q_head_stride_bytes,
                    sQBytes,
                    lane,
                    warp_q_idx,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        else:
            sQBytes = cute.flatten(cute.recast_tensor(sQ, cutlass.Uint8))
            self._async_copy_q_tile_permuted_128b(
                mQBytes,
                q_start,
                packed_tile_start,
                packed_tile_rows,
                kv_head_idx,
                group_size,
                mQ.shape[1],
                self.traits.head_dim_qk * (self.dtype_q.width // 8),
                q_token_stride_bytes,
                q_head_stride_bytes,
                sQBytes,
                lane,
                warp_q_idx,
            )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

        k_scale = (
            mKDescale[request_idx]
            if const_expr(mKDescale is not None and len(mKDescale.shape) == 1)
            else (
                mKDescale[request_idx, kv_head_idx]
                if const_expr(mKDescale is not None)
                else Float32(1.0)
            )
        )
        v_scale = (
            mVDescale[request_idx]
            if const_expr(mVDescale is not None and len(mVDescale.shape) == 1)
            else (
                mVDescale[request_idx, kv_head_idx]
                if const_expr(mVDescale is not None)
                else Float32(1.0)
            )
        )
        num_mma_q = self.traits.num_mma_q
        num_mma_kv = self.traits.num_mma_kv
        num_mma_d_vo = self.traits.num_mma_d_vo
        warp_row_base = warp_q_idx * num_mma_q * 16
        warp_kv_base = warp_kv_idx * num_mma_kv * 16
        lane_group = lane // 4
        lane_pair_base = 2 * (lane % 4)
        row_local_idx = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 2), stride=(2, 1)), Int32
        )
        row_valid = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 2), stride=(2, 1)), Int32
        )
        q_token_local = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 2), stride=(2, 1)), Int32
        )
        q_head_idx_frag = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 2), stride=(2, 1)), Int32
        )
        q_row_idx_frag = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 2), stride=(2, 1)), Int32
        )
        causal_k_limit = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 2), stride=(2, 1)), Int32
        )
        frag_s_layout = cute.make_layout(
            (num_mma_q, num_mma_kv, 8), stride=(num_mma_kv * 8, 8, 1)
        )
        frag_p_layout = cute.make_layout(
            (num_mma_q, num_mma_kv, 4), stride=(num_mma_kv * 4, 4, 1)
        )
        frag_o_layout = cute.make_layout(
            (num_mma_q, num_mma_d_vo, 8), stride=(num_mma_d_vo * 8, 8, 1)
        )
        s_frag = cute.make_rmem_tensor(
            frag_s_layout,
            Float32,
        )
        tSsQ_tma = None
        tSrQ_tma = None
        acc_shape_S_tma = None
        tiled_mma_qk_tma = None
        tiled_mma_pv_tma = None
        thr_mma_qk_tma = None
        thr_mma_pv_tma = None
        smem_thr_copy_K_tma = None
        smem_thr_copy_V_tma = None
        tScS_mn_tma = None
        t0ScS_mn_tma = None
        o_frag = cute.make_rmem_tensor(
            frag_o_layout,
            Float32,
        )
        m_frag = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 2), stride=(2, 1)), Float32
        )
        d_frag = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 2), stride=(2, 1)), Float32
        )
        p_frag = cute.make_rmem_tensor(
            frag_p_layout,
            Uint32,
        )
        p_frag_scalar_debug = cute.make_rmem_tensor(
            frag_s_layout,
            cutlass.BFloat16,
        )
        q_smem_base_addr = shared_ptr_to_u32(sQ.iterator)

        for mma_q in cutlass.range_constexpr(num_mma_q):
            for row_slot in cutlass.range_constexpr(2):
                packed_row_local = (
                    warp_row_base + mma_q * 16 + lane_group + 8 * row_slot
                )
                row_local_idx[mma_q, row_slot] = Int32(packed_row_local)
                valid_row = packed_row_local < packed_tile_rows
                row_valid[mma_q, row_slot] = Int32(valid_row)
                if valid_row:
                    packed_q_idx = packed_tile_start + packed_row_local
                    token_local = packed_q_idx // group_size
                    q_group_lane = packed_q_idx - token_local * group_size
                    q_token_local[mma_q, row_slot] = Int32(token_local)
                    q_head_idx_frag[mma_q, row_slot] = Int32(
                        kv_head_idx * group_size + q_group_lane
                    )
                    q_row_idx_frag[mma_q, row_slot] = Int32(q_start + token_local)
                    causal_k_limit[mma_q, row_slot] = Int32(
                        token_local + cache_len - qo_len
                    )
                else:
                    q_token_local[mma_q, row_slot] = Int32(0)
                    q_head_idx_frag[mma_q, row_slot] = Int32(0)
                    q_row_idx_frag[mma_q, row_slot] = Int32(0)
                    causal_k_limit[mma_q, row_slot] = Int32(-1)

        for mma_q in cutlass.range_constexpr(num_mma_q):
            for mma_d in cutlass.range_constexpr(num_mma_d_vo):
                for reg_id in cutlass.range_constexpr(8):
                    o_frag[mma_q, mma_d, reg_id] = Float32(0.0)
            for row_slot in cutlass.range_constexpr(2):
                m_frag[mma_q, row_slot] = Float32(-Float32.inf)
                d_frag[mma_q, row_slot] = Float32(1.0)

        prefetch_base = chunk_start
        preload_count = 0
        preload_stage_idx = Int32(0)
        if const_expr(self.use_paged_k_tma or self.use_paged_v_tma):
            kv_producer_state = cute_pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.num_stages
            )
            kv_consumer_state = cute_pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, self.num_stages
            )
        else:
            kv_producer_state = None
            kv_consumer_state = None
        if const_expr(self.use_paged_k_tma or self.use_paged_v_tma):
            if prefetch_base < chunk_end:
                tile_limit = cutlass.select_(
                    prefetch_base + stage_tile_rows < chunk_end,
                    prefetch_base + stage_tile_rows,
                    chunk_end,
                )
                tile_tokens = tile_limit - prefetch_base
                prefetch_page_idx = self._tile_page_idx(
                    mQ2KIndices,
                    mMSAUnionBlocks,
                    work_idx,
                    kv_head_idx,
                    msa_q_row_idx,
                    prefetch_base,
                    page_size,
                )
                if warp_linear_idx == Int32(0):
                    if const_expr(self.use_paged_k_tma):
                        if const_expr(self.use_paged_kv_tma_fp8_raw_issue):
                            self._issue_paged_kv_tma_copy_2planes_fp8_raw(
                                mKTmaDescFlat,
                                kv_head_idx,
                                sKStageBytes,
                                Int32(preload_stage_idx * kv_plane_stage_bytes),
                                kv_plane_total_bytes,
                                kv_producer_state,
                                mbar_ptr_K,
                                self.kv_tma_copy_bytes_k,
                                mPageTable,
                                request_idx,
                                prefetch_base,
                                page_size,
                            )
                        elif const_expr(self.kv_tma_plane_count > 2):
                            self._issue_paged_kv_tma_copy_planes(
                                load_K_tma0,
                                load_K_tma1,
                                load_K_tma2,
                                load_K_tma3,
                                pipeline_k,
                                kv_producer_state,
                                mPageTable,
                                request_idx,
                                prefetch_base,
                                page_size,
                            )
                        else:
                            self._issue_paged_kv_tma_copy_2planes(
                                load_K_tma0,
                                load_K_tma1,
                                pipeline_k,
                                kv_producer_state,
                                mPageTable,
                                request_idx,
                                prefetch_base,
                                page_size,
                            )
                    if const_expr(self.use_paged_v_tma):
                        if const_expr(self.use_paged_kv_tma_fp8_raw_issue):
                            self._issue_paged_kv_tma_copy_2planes_fp8_raw(
                                mVTmaDescFlat,
                                kv_head_idx,
                                sVStageBytes,
                                Int32(preload_stage_idx * kv_plane_stage_bytes),
                                kv_plane_total_bytes,
                                kv_producer_state,
                                mbar_ptr_V,
                                self.kv_tma_copy_bytes_v,
                                mPageTable,
                                request_idx,
                                prefetch_base,
                                page_size,
                            )
                        elif const_expr(self.kv_tma_plane_count > 2):
                            self._issue_paged_kv_tma_copy_planes(
                                load_V_tma0,
                                load_V_tma1,
                                load_V_tma2,
                                load_V_tma3,
                                pipeline_v,
                                kv_producer_state,
                                mPageTable,
                                request_idx,
                                prefetch_base,
                                page_size,
                            )
                        else:
                            self._issue_paged_kv_tma_copy_2planes(
                                load_V_tma0,
                                load_V_tma1,
                                pipeline_v,
                                kv_producer_state,
                                mPageTable,
                                request_idx,
                                prefetch_base,
                                page_size,
                            )
                if const_expr(not self.use_paged_k_tma):
                    self._async_copy_paged_tile_permuted_128b(
                        mKBytes,
                        mPageTable,
                        request_idx,
                        prefetch_base,
                        prefetch_page_idx,
                        kv_head_idx,
                        mKCache.shape[2],
                        k_row_bytes,
                        k_page_stride_bytes,
                        k_token_stride_bytes,
                        k_head_stride_bytes,
                        sKStageBytes,
                        Int32(preload_stage_idx * k_stage_bytes),
                        lane,
                        warp_linear_idx,
                        tile_tokens,
                        self.traits.upcast_stride_k,
                        False,
                    )
                    cute.arch.cp_async_commit_group()
                if const_expr(not self.use_paged_v_tma):
                    self._async_copy_paged_tile_permuted_128b(
                        mVBytes,
                        mPageTable,
                        request_idx,
                        prefetch_base,
                        prefetch_page_idx,
                        kv_head_idx,
                        mVCache.shape[2],
                        v_row_bytes,
                        v_page_stride_bytes,
                        v_token_stride_bytes,
                        v_head_stride_bytes,
                        sVStageBytes,
                        Int32(preload_stage_idx * v_stage_bytes),
                        lane,
                        warp_linear_idx,
                        tile_tokens,
                        self.traits.upcast_stride_v,
                        True,
                    )
                    cute.arch.cp_async_commit_group()
                kv_producer_state.advance()
                prefetch_base += stage_tile_rows
                preload_count = 1
        else:
            while preload_count < self.num_stages and prefetch_base < chunk_end:
                tile_limit = cutlass.select_(
                    prefetch_base + stage_tile_rows < chunk_end,
                    prefetch_base + stage_tile_rows,
                    chunk_end,
                )
                tile_tokens = tile_limit - prefetch_base
                prefetch_page_idx = self._tile_page_idx(
                    mQ2KIndices,
                    mMSAUnionBlocks,
                    work_idx,
                    kv_head_idx,
                    msa_q_row_idx,
                    prefetch_base,
                    page_size,
                )
                self._async_copy_paged_tile_permuted_128b(
                    mKBytes,
                    mPageTable,
                    request_idx,
                    prefetch_base,
                    prefetch_page_idx,
                    kv_head_idx,
                    mKCache.shape[2],
                    k_row_bytes,
                    k_page_stride_bytes,
                    k_token_stride_bytes,
                    k_head_stride_bytes,
                    sKStageBytes,
                    Int32(preload_stage_idx * k_stage_bytes),
                    lane,
                    warp_linear_idx,
                    tile_tokens,
                    self.traits.upcast_stride_k,
                    False,
                )
                cute.arch.cp_async_commit_group()
                self._async_copy_paged_tile_permuted_128b(
                    mVBytes,
                    mPageTable,
                    request_idx,
                    prefetch_base,
                    prefetch_page_idx,
                    kv_head_idx,
                    mVCache.shape[2],
                    v_row_bytes,
                    v_page_stride_bytes,
                    v_token_stride_bytes,
                    v_head_stride_bytes,
                    sVStageBytes,
                    Int32(preload_stage_idx * v_stage_bytes),
                    lane,
                    warp_linear_idx,
                    tile_tokens,
                    self.traits.upcast_stride_v,
                    True,
                )
                cute.arch.cp_async_commit_group()
                prefetch_base += stage_tile_rows
                preload_count += 1
                if const_expr(self.num_stages == 2):
                    preload_stage_idx = Int32(1) - preload_stage_idx

        if const_expr(
            self.debug_dump_paged_kv_planewords
            and self.kv_is_fp8
            and not self.use_paged_k_tma
            and self.use_paged_v_tma
        ):
            if const_expr(self.use_paged_kv_tma_fp8_raw_issue):
                cute.arch.mbarrier_wait(mbar_ptr_V + Int32(0), phase=Int32(0))
            else:
                pipeline_v.consumer_wait(
                    kv_consumer_state,
                    pipeline_v.consumer_try_wait(kv_consumer_state),
                )
            cute.arch.sync_threads()
            if work_idx == Int32(0) and kv_head_idx == Int32(0):
                mDebugU32 = cute.flatten(cute.recast_tensor(mO, cutlass.Uint32))
                _dump_plane_stage_words_u32(
                    mDebugU32,
                    sVStageBytes,
                    Int32(0),
                    kv_plane_stage_bytes,
                    kv_plane_total_bytes,
                    self.kv_tma_plane_count,
                    tidx,
                    self.traits.num_threads,
                )
            _exit_thread()

        consume_stage_idx = Int32(0)
        tile_base = chunk_start
        while tile_base < chunk_end:
            tile_limit = cutlass.select_(
                tile_base + stage_tile_rows < chunk_end,
                tile_base + stage_tile_rows,
                chunk_end,
            )
            tile_key_base = self._tile_key_base(
                mQ2KIndices,
                mMSAUnionBlocks,
                work_idx,
                kv_head_idx,
                msa_q_row_idx,
                tile_base,
            )
            tile_tokens = tile_limit - tile_base
            if const_expr(self.msa_block_sparse):
                live_limit = (
                    msa_tile_max_visible_len
                    if const_expr(self.msa_union_tile)
                    else msa_visible_len
                )
                live_tokens = live_limit - tile_key_base
                live_tokens = cutlass.select_(
                    live_tokens < Int32(0), Int32(0), live_tokens
                )
                tile_tokens = cutlass.select_(
                    live_tokens < tile_tokens, live_tokens, tile_tokens
                )
            if const_expr(self.use_paged_k_tma):
                if const_expr(self.use_paged_kv_tma_fp8_raw_issue):
                    cute.arch.mbarrier_wait(
                        mbar_ptr_K + kv_consumer_state.index,
                        phase=kv_consumer_state.phase,
                    )
                else:
                    pipeline_k.consumer_wait(
                        kv_consumer_state,
                        pipeline_k.consumer_try_wait(kv_consumer_state),
                    )
            elif const_expr(self.traits.num_warps_kv > 1):
                cute.arch.cp_async_wait_group(1 if self.kv_is_fp8 else 0)
            else:
                cute.arch.cp_async_wait_group(1)
            cute.arch.sync_threads()

            if const_expr(
                self.debug_dump_paged_kv_tma_k or self.debug_dump_paged_kv_tma_v
            ):
                if work_idx == Int32(0) and kv_head_idx == Int32(0):
                    _dump_tma_stage_rows(
                        mO,
                        sKTma if const_expr(self.debug_dump_paged_kv_tma_k) else sVTma,
                        tidx,
                        stage_tile_rows,
                        self.traits.head_dim_qk
                        if const_expr(self.debug_dump_paged_kv_tma_k)
                        else self.traits.head_dim_vo,
                        self.traits.num_threads,
                        Int32(24),
                    )
                _exit_thread()

            if const_expr(
                self.debug_dump_paged_kv_extend_kwords
                or self.debug_dump_paged_kv_extend_vwords
            ):
                if (
                    work_idx == Int32(0)
                    and kv_head_idx == Int32(0)
                    and tile_base == chunk_start
                ):
                    mDebugU32 = cute.flatten(cute.recast_tensor(mO, cutlass.Uint32))
                    _dump_plane_stage_words_u32(
                        mDebugU32,
                        sKStageBytes
                        if const_expr(self.debug_dump_paged_kv_extend_kwords)
                        else sVStageBytes,
                        consume_stage_idx,
                        kv_plane_stage_bytes,
                        kv_plane_total_bytes,
                        self.kv_tma_plane_count,
                        tidx,
                        self.traits.num_threads,
                    )
                _exit_thread()

            subtile_base = (
                Int32(0) if const_expr(self.traits.num_warps_kv == 1) else warp_kv_base
            )
            for _ in cutlass.range_constexpr(1):
                p_frag.fill(Uint32(0))
                if const_expr(self.use_native_fp8_qk_mma):
                    k_smem_base_addr = shared_ptr_to_u32(
                        sKStageBytes.iterator + Int32(consume_stage_idx * k_stage_bytes)
                    )
                    frag_S = cute.make_rmem_tensor(
                        cute.make_layout(
                            (num_mma_q, num_mma_kv, 8),
                            stride=(num_mma_kv * 8, 8, 1),
                        ),
                        Float32,
                    )
                    frag_S.fill(0.0)
                    _literal_qk_mma_into_sfrag_mxfp8_raw(
                        frag_S,
                        q_smem_base_addr,
                        k_smem_base_addr,
                        lane,
                        warp_q_idx,
                        warp_kv_idx,
                        Int32(0)
                        if const_expr(self.traits.num_warps_kv > 1)
                        else subtile_base,
                        num_mma_q,
                        num_mma_kv,
                        self.traits.num_mma_d_qk,
                        tc_upcast_stride_qk,
                        self.traits.upcast_stride_k,
                    )
                    for mma_q in cutlass.range_constexpr(num_mma_q):
                        for mma_kv in cutlass.range_constexpr(num_mma_kv):
                            for reg_id in cutlass.range_constexpr(8):
                                row_slot = (reg_id % 4) // 2
                                key_local = (
                                    warp_kv_base
                                    + mma_kv * 16
                                    + lane_pair_base
                                    + 8 * (reg_id // 4)
                                    + (reg_id % 2)
                                )
                                valid = row_valid[mma_q, row_slot] != 0
                                if valid:
                                    valid = valid and key_local < tile_tokens
                                if valid:
                                    key_pos = tile_key_base + key_local
                                    if const_expr(self.msa_union_tile):
                                        valid = valid and self._msa_union_row_has_block(
                                            mMSAUnionMasks,
                                            work_idx,
                                            kv_head_idx,
                                            tile_base,
                                            q_token_local[mma_q, row_slot],
                                            msa_tile_first_token,
                                        )
                                    valid = (
                                        valid
                                        and key_pos <= causal_k_limit[mma_q, row_slot]
                                    )
                                    if const_expr(self.window_left >= 0):
                                        window_start = causal_k_limit[
                                            mma_q, row_slot
                                        ] - Int32(self.window_left)
                                        window_start = cutlass.select_(
                                            window_start > Int32(0),
                                            window_start,
                                            Int32(0),
                                        )
                                        valid = valid and key_pos >= window_start
                                if valid:
                                    frag_S[mma_q, mma_kv, reg_id] = (
                                        frag_S[mma_q, mma_kv, reg_id] * k_scale
                                    )
                                else:
                                    frag_S[mma_q, mma_kv, reg_id] = Float32(
                                        -Float32.inf
                                    )
                    if const_expr(self.has_relative_attention_bias):
                        _apply_relative_attention_bias(
                            frag_S,
                            mRelativeAttentionBias,
                            q_row_idx_frag,
                            q_head_idx_frag,
                            causal_k_limit,
                            tile_key_base,
                            warp_kv_base,
                            lane_pair_base,
                            self.inverse_softmax_scale,
                            num_mma_q,
                            num_mma_kv,
                        )
                    p_frag_scalar = (
                        p_frag_scalar_debug
                        if const_expr(self.debug_dump_paged_kv_extend_pscalars)
                        else None
                    )
                    if const_expr(self.debug_dump_paged_kv_extend_pscalars):
                        p_frag_scalar.fill(cutlass.BFloat16(0.0))
                    if (
                        const_expr(self.traits.num_warps_kv == 1)
                        or warp_kv_base < tile_tokens
                    ):
                        _literal_update_mdo_states_fp32_pack_p(
                            frag_S,
                            o_frag,
                            m_frag,
                            d_frag,
                            p_frag,
                            self.softmax_scale_log2,
                            num_mma_q,
                            num_mma_kv,
                            num_mma_d_vo,
                            p_frag_scalar,
                        )
                elif const_expr(self.kv_is_fp8):
                    k_smem_base_addr = shared_ptr_to_u32(
                        sKStageBytes.iterator + Int32(consume_stage_idx * k_stage_bytes)
                    )
                    frag_S = cute.make_rmem_tensor(
                        cute.make_layout(
                            (num_mma_q, num_mma_kv, 8),
                            stride=(num_mma_kv * 8, 8, 1),
                        ),
                        Float32,
                    )
                    frag_S.fill(0.0)
                    if const_expr(self.use_paged_k_tma):
                        k_stage_plane_offset = Int32(
                            consume_stage_idx * kv_plane_stage_bytes
                        )
                        _literal_qk_mma_into_sfrag_plane_fp8_raw(
                            frag_S,
                            q_smem_base_addr,
                            shared_ptr_to_u32(
                                sKStageBytes.iterator
                                + k_stage_plane_offset
                                + Int32(0 * kv_plane_total_bytes)
                            ),
                            shared_ptr_to_u32(
                                sKStageBytes.iterator
                                + k_stage_plane_offset
                                + Int32(1 * kv_plane_total_bytes)
                            ),
                            lane,
                            warp_q_idx,
                            warp_kv_idx,
                            Int32(0)
                            if const_expr(self.traits.num_warps_kv > 1)
                            else subtile_base,
                            num_mma_q,
                            num_mma_kv,
                            self.traits.num_mma_d_qk,
                            tc_upcast_stride_qk,
                            tc_upcast_stride_plane,
                        )
                    else:
                        _literal_qk_mma_into_sfrag_fp8_raw_paired(
                            frag_S,
                            q_smem_base_addr,
                            k_smem_base_addr,
                            lane,
                            warp_q_idx,
                            warp_kv_idx,
                            Int32(0)
                            if const_expr(self.traits.num_warps_kv > 1)
                            else subtile_base,
                            num_mma_q,
                            num_mma_kv,
                            self.traits.num_mma_d_qk,
                            tc_upcast_stride_qk,
                            self.traits.upcast_stride_k,
                        )
                    for mma_q in cutlass.range_constexpr(num_mma_q):
                        for mma_kv in cutlass.range_constexpr(num_mma_kv):
                            for reg_id in cutlass.range_constexpr(8):
                                row_slot = (reg_id % 4) // 2
                                key_local = (
                                    warp_kv_base
                                    + mma_kv * 16
                                    + lane_pair_base
                                    + 8 * (reg_id // 4)
                                    + (reg_id % 2)
                                )
                                valid = row_valid[mma_q, row_slot] != 0
                                if valid:
                                    valid = valid and key_local < tile_tokens
                                if valid:
                                    key_pos = tile_key_base + key_local
                                    if const_expr(self.msa_union_tile):
                                        valid = valid and self._msa_union_row_has_block(
                                            mMSAUnionMasks,
                                            work_idx,
                                            kv_head_idx,
                                            tile_base,
                                            q_token_local[mma_q, row_slot],
                                            msa_tile_first_token,
                                        )
                                    valid = (
                                        valid
                                        and key_pos <= causal_k_limit[mma_q, row_slot]
                                    )
                                    if const_expr(self.window_left >= 0):
                                        window_start = causal_k_limit[
                                            mma_q, row_slot
                                        ] - Int32(self.window_left)
                                        window_start = cutlass.select_(
                                            window_start > Int32(0),
                                            window_start,
                                            Int32(0),
                                        )
                                        valid = valid and key_pos >= window_start
                                if valid:
                                    frag_S[mma_q, mma_kv, reg_id] = (
                                        frag_S[mma_q, mma_kv, reg_id] * k_scale
                                    )
                                else:
                                    frag_S[mma_q, mma_kv, reg_id] = Float32(
                                        -Float32.inf
                                    )
                    if const_expr(self.has_relative_attention_bias):
                        _apply_relative_attention_bias(
                            frag_S,
                            mRelativeAttentionBias,
                            q_row_idx_frag,
                            q_head_idx_frag,
                            causal_k_limit,
                            tile_key_base,
                            warp_kv_base,
                            lane_pair_base,
                            self.inverse_softmax_scale,
                            num_mma_q,
                            num_mma_kv,
                        )
                    p_frag_scalar = (
                        p_frag_scalar_debug
                        if const_expr(self.debug_dump_paged_kv_extend_pscalars)
                        else None
                    )
                    if const_expr(self.debug_dump_paged_kv_extend_pscalars):
                        p_frag_scalar.fill(cutlass.BFloat16(0.0))
                    if (
                        const_expr(self.traits.num_warps_kv == 1)
                        or warp_kv_base < tile_tokens
                    ):
                        _literal_update_mdo_states_fp32_pack_p(
                            frag_S,
                            o_frag,
                            m_frag,
                            d_frag,
                            p_frag,
                            self.softmax_scale_log2,
                            num_mma_q,
                            num_mma_kv,
                            num_mma_d_vo,
                            p_frag_scalar,
                        )
                else:
                    literal_key_base = (
                        Int32(0)
                        if const_expr(self.traits.num_warps_kv > 1)
                        else subtile_base
                    )
                    k_smem_base_addr = shared_ptr_to_u32(
                        sKStageBytes.iterator + Int32(consume_stage_idx * k_stage_bytes)
                    )
                    p_frag_scalar = None
                    if const_expr(self.use_paged_k_tma):
                        frag_S = cute.make_rmem_tensor(
                            frag_s_layout,
                            Float32,
                        )
                        frag_S.fill(0.0)
                        k_stage_plane_offset = Int32(
                            consume_stage_idx * kv_plane_stage_bytes
                        )
                        _literal_qk_mma_into_sfrag_plane_bf16(
                            frag_S,
                            q_smem_base_addr,
                            shared_ptr_to_u32(
                                sKStageBytes.iterator
                                + k_stage_plane_offset
                                + Int32(0 * kv_plane_total_bytes)
                            ),
                            shared_ptr_to_u32(
                                sKStageBytes.iterator
                                + k_stage_plane_offset
                                + Int32(1 * kv_plane_total_bytes)
                            ),
                            shared_ptr_to_u32(
                                sKStageBytes.iterator
                                + k_stage_plane_offset
                                + Int32(2 * kv_plane_total_bytes)
                            ),
                            shared_ptr_to_u32(
                                sKStageBytes.iterator
                                + k_stage_plane_offset
                                + Int32(3 * kv_plane_total_bytes)
                            ),
                            lane,
                            warp_q_idx,
                            warp_kv_idx,
                            literal_key_base,
                            num_mma_q,
                            num_mma_kv,
                            self.traits.num_mma_d_qk,
                            tc_upcast_stride_qk,
                            tc_upcast_stride_plane,
                        )
                    else:
                        frag_S = cute.make_rmem_tensor(
                            frag_s_layout,
                            Float32,
                        )
                        frag_S.fill(0.0)
                        _literal_qk_mma_into_sfrag(
                            frag_S,
                            q_smem_base_addr,
                            k_smem_base_addr,
                            lane,
                            warp_q_idx,
                            warp_kv_idx,
                            literal_key_base,
                            num_mma_q,
                            num_mma_kv,
                            self.traits.num_mma_d_qk,
                            tc_upcast_stride_qk,
                            tc_upcast_stride_qk,
                        )
                    for mma_q in cutlass.range_constexpr(num_mma_q):
                        for mma_kv in cutlass.range_constexpr(num_mma_kv):
                            for reg_id in cutlass.range_constexpr(8):
                                row_slot = (reg_id % 4) // 2
                                key_local = (
                                    warp_kv_base
                                    + mma_kv * 16
                                    + lane_pair_base
                                    + 8 * (reg_id // 4)
                                    + (reg_id % 2)
                                )
                                valid = row_valid[mma_q, row_slot] != 0
                                if valid:
                                    valid = valid and key_local < tile_tokens
                                if valid:
                                    key_pos = tile_key_base + key_local
                                    if const_expr(self.msa_union_tile):
                                        valid = valid and self._msa_union_row_has_block(
                                            mMSAUnionMasks,
                                            work_idx,
                                            kv_head_idx,
                                            tile_base,
                                            q_token_local[mma_q, row_slot],
                                            msa_tile_first_token,
                                        )
                                    valid = (
                                        valid
                                        and key_pos <= causal_k_limit[mma_q, row_slot]
                                    )
                                    if const_expr(self.window_left >= 0):
                                        window_start = causal_k_limit[
                                            mma_q, row_slot
                                        ] - Int32(self.window_left)
                                        window_start = cutlass.select_(
                                            window_start > Int32(0),
                                            window_start,
                                            Int32(0),
                                        )
                                        valid = valid and key_pos >= window_start
                                # Keep each score selection scalar.  CUTLASS DSL 4.6
                                # otherwise combines the unrolled branches into b64
                                # selects, extending pair lifetimes and adding three
                                # registers to this prefill specialization.
                                frag_S[mma_q, mma_kv, reg_id] = cutlass.select_(
                                    valid,
                                    frag_S[mma_q, mma_kv, reg_id],
                                    Float32(-Float32.inf),
                                )

                    if const_expr(self.debug_dump_paged_kv_tma_s):
                        if work_idx == Int32(0) and kv_head_idx == Int32(0):
                            _dump_s_frag_tile(
                                mO,
                                frag_S,
                                lane,
                                warp_q_idx,
                                warp_kv_idx,
                                num_mma_q,
                                num_mma_kv,
                                packed_tile_rows,
                                tile_tokens,
                            )
                        _exit_thread()

                    if const_expr(self.has_relative_attention_bias):
                        _apply_relative_attention_bias(
                            frag_S,
                            mRelativeAttentionBias,
                            q_row_idx_frag,
                            q_head_idx_frag,
                            causal_k_limit,
                            tile_key_base,
                            warp_kv_base,
                            lane_pair_base,
                            self.inverse_softmax_scale,
                            num_mma_q,
                            num_mma_kv,
                        )
                    p_frag_scalar = (
                        p_frag_scalar_debug
                        if const_expr(self.debug_dump_paged_kv_extend_pscalars)
                        else None
                    )
                    if const_expr(self.debug_dump_paged_kv_extend_pscalars):
                        p_frag_scalar.fill(cutlass.BFloat16(0.0))
                    _literal_update_mdo_states_fp32_pack_p(
                        frag_S,
                        o_frag,
                        m_frag,
                        d_frag,
                        p_frag,
                        self.softmax_scale_log2,
                        num_mma_q,
                        num_mma_kv,
                        num_mma_d_vo,
                        p_frag_scalar,
                    )
                if const_expr(self.debug_dump_paged_kv_pregs):
                    if (
                        work_idx == Int32(0)
                        and kv_head_idx == Int32(0)
                        and warp_q_idx == Int32(0)
                        and warp_kv_idx == Int32(0)
                    ):
                        mDebugU32 = cute.flatten(cute.recast_tensor(mO, cutlass.Uint32))
                        _dump_p_frag_regs_raw(
                            mDebugU32,
                            p_frag,
                            lane,
                        )
                    _exit_thread()
                if const_expr(self.debug_dump_paged_kv_sregs):
                    if (
                        work_idx == Int32(0)
                        and kv_head_idx == Int32(0)
                        and warp_q_idx == Int32(0)
                        and warp_kv_idx == Int32(0)
                    ):
                        mDebugU32 = cute.flatten(cute.recast_tensor(mO, cutlass.Uint32))
                        _dump_s_frag_regs_raw(
                            mDebugU32,
                            frag_S,
                            lane,
                        )
                    _exit_thread()
                if const_expr(self.debug_dump_paged_kv_extend_sregs):
                    if (
                        work_idx == Int32(0)
                        and kv_head_idx == Int32(0)
                        and tile_base == chunk_start
                    ):
                        mDebugU32 = cute.flatten(cute.recast_tensor(mO, cutlass.Uint32))
                        lane_words = num_mma_q * num_mma_kv * 8
                        dst_word_offset = (
                            (warp_q_idx * Int32(self.traits.num_warps_kv) + warp_kv_idx)
                            * Int32(32)
                            * lane_words
                        )
                        _dump_s_frag_regs_raw_offset(
                            mDebugU32,
                            frag_S,
                            lane,
                            dst_word_offset,
                        )
                    _exit_thread()
                for mma_q in cutlass.range_constexpr(num_mma_q):
                    for mma_kv in cutlass.range_constexpr(num_mma_kv):
                        d0, d1 = bf16_rowsum_m16k16_f32(
                            d_frag[mma_q, 0],
                            d_frag[mma_q, 1],
                            p_frag[mma_q, mma_kv, 0],
                            p_frag[mma_q, mma_kv, 1],
                            p_frag[mma_q, mma_kv, 2],
                            p_frag[mma_q, mma_kv, 3],
                        )
                        d_frag[mma_q, 0] = d0
                        d_frag[mma_q, 1] = d1
                if const_expr(self.kv_is_fp8 and self.traits.num_warps_kv > 1):
                    if warp_kv_base >= tile_tokens:
                        for mma_q in cutlass.range_constexpr(num_mma_q):
                            for row_slot in cutlass.range_constexpr(2):
                                m_frag[mma_q, row_slot] = Float32(-Float32.inf)
                                d_frag[mma_q, row_slot] = Float32(1.0)
                            for mma_d in cutlass.range_constexpr(num_mma_d_vo):
                                for reg_id in cutlass.range_constexpr(8):
                                    o_frag[mma_q, mma_d, reg_id] = Float32(0.0)

                if const_expr(self.debug_dump_paged_kv_extend_state):
                    if (
                        work_idx == Int32(0)
                        and kv_head_idx == Int32(0)
                        and tile_base == chunk_start
                    ):
                        dump_base = (
                            warp_q_idx * Int32(self.traits.num_warps_kv) + warp_kv_idx
                        ) * Int32(4)
                        if lane == Int32(0):
                            mLSE[dump_base + 0, 0] = m_frag[0, 0]
                            mLSE[dump_base + 1, 0] = d_frag[0, 0]
                            mLSE[dump_base + 2, 0] = m_frag[0, 1]
                            mLSE[dump_base + 3, 0] = d_frag[0, 1]
                    _exit_thread()
                if const_expr(self.debug_dump_paged_kv_extend_pregs):
                    if (
                        work_idx == Int32(0)
                        and kv_head_idx == Int32(0)
                        and tile_base == chunk_start
                    ):
                        mDebugU32 = cute.flatten(cute.recast_tensor(mO, cutlass.Uint32))
                        lane_words = num_mma_kv * 4
                        dst_idx = (
                            (warp_q_idx * Int32(self.traits.num_warps_kv) + warp_kv_idx)
                            * Int32(32)
                            + lane
                        ) * lane_words
                        for mma_kv in cutlass.range_constexpr(num_mma_kv):
                            word_base = dst_idx + mma_kv * 4
                            mDebugU32[word_base + 0] = p_frag[0, mma_kv, 0]
                            mDebugU32[word_base + 1] = p_frag[0, mma_kv, 1]
                            mDebugU32[word_base + 2] = p_frag[0, mma_kv, 2]
                            mDebugU32[word_base + 3] = p_frag[0, mma_kv, 3]
                    _exit_thread()
                if const_expr(self.debug_dump_paged_kv_extend_pscalars):
                    if (
                        work_idx == Int32(0)
                        and kv_head_idx == Int32(0)
                        and tile_base == chunk_start
                    ):
                        mDebugU32 = cute.flatten(cute.recast_tensor(mO, cutlass.Uint32))
                        lane_words = num_mma_q * num_mma_kv * 4
                        dst_word_offset = (
                            (warp_q_idx * Int32(self.traits.num_warps_kv) + warp_kv_idx)
                            * Int32(32)
                            * lane_words
                        )
                        _dump_flat_u32_words_offset(
                            mDebugU32,
                            cute.flatten(
                                cute.recast_tensor(p_frag_scalar_debug, cutlass.Uint32)
                            ),
                            dst_word_offset,
                            lane,
                            Int32(32),
                        )
                    _exit_thread()

                if const_expr(
                    self.use_paged_k_tma and not self.use_paged_kv_tma_fp8_raw_issue
                ):
                    pipeline_k.consumer_release(kv_consumer_state)

                next_tile_base = prefetch_base
                next_tile_tokens = Int32(0)
                next_page_idx = Int32(0)
                if const_expr(not self.use_paged_k_tma):
                    if const_expr(self.traits.num_warps_kv > 1):
                        if next_tile_base < chunk_end:
                            next_tile_limit = cutlass.select_(
                                next_tile_base + stage_tile_rows < chunk_end,
                                next_tile_base + stage_tile_rows,
                                chunk_end,
                            )
                            next_tile_tokens = next_tile_limit - next_tile_base
                            next_page_idx = self._tile_page_idx(
                                mQ2KIndices,
                                mMSAUnionBlocks,
                                work_idx,
                                kv_head_idx,
                                msa_q_row_idx,
                                next_tile_base,
                                page_size,
                            )
                            self._async_copy_paged_tile_permuted_128b(
                                mKBytes,
                                mPageTable,
                                request_idx,
                                next_tile_base,
                                next_page_idx,
                                kv_head_idx,
                                mKCache.shape[2],
                                k_row_bytes,
                                k_page_stride_bytes,
                                k_token_stride_bytes,
                                k_head_stride_bytes,
                                sKStageBytes,
                                Int32(consume_stage_idx * k_stage_bytes),
                                lane,
                                warp_linear_idx,
                                next_tile_tokens,
                                self.traits.upcast_stride_k,
                                False,
                            )
                            cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(1)
                        cute.arch.sync_threads()
                    elif const_expr(self.traits.num_warps_kv == 1):
                        if next_tile_base < chunk_end:
                            next_tile_limit = cutlass.select_(
                                next_tile_base + stage_tile_rows < chunk_end,
                                next_tile_base + stage_tile_rows,
                                chunk_end,
                            )
                            next_tile_tokens = next_tile_limit - next_tile_base
                            next_page_idx = self._tile_page_idx(
                                mQ2KIndices,
                                mMSAUnionBlocks,
                                work_idx,
                                kv_head_idx,
                                msa_q_row_idx,
                                next_tile_base,
                                page_size,
                            )
                            self._async_copy_paged_tile_permuted_128b(
                                mKBytes,
                                mPageTable,
                                request_idx,
                                next_tile_base,
                                next_page_idx,
                                kv_head_idx,
                                mKCache.shape[2],
                                k_row_bytes,
                                k_page_stride_bytes,
                                k_token_stride_bytes,
                                k_head_stride_bytes,
                                sKStageBytes,
                                Int32(consume_stage_idx * k_stage_bytes),
                                lane,
                                warp_linear_idx,
                                next_tile_tokens,
                                self.traits.upcast_stride_k,
                                False,
                            )
                            cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(1)
                        cute.arch.sync_threads()

                if const_expr(self.use_paged_v_tma):
                    if const_expr(self.use_paged_kv_tma_fp8_raw_issue):
                        cute.arch.mbarrier_wait(
                            mbar_ptr_V + kv_consumer_state.index,
                            phase=kv_consumer_state.phase,
                        )
                    else:
                        pipeline_v.consumer_wait(
                            kv_consumer_state,
                            pipeline_v.consumer_try_wait(kv_consumer_state),
                        )
                elif const_expr(self.use_paged_k_tma):
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()

                if const_expr(self.debug_dump_paged_kv_pvregs):
                    if (
                        work_idx == Int32(0)
                        and kv_head_idx == Int32(0)
                        and warp_q_idx == Int32(0)
                        and warp_kv_idx == Int32(0)
                    ):
                        mDebugU32 = cute.flatten(cute.recast_tensor(mO, cutlass.Uint32))
                        pv_row_base = (
                            Int32(0)
                            if const_expr(self.traits.num_warps_kv > 1)
                            else subtile_base
                        )
                        if const_expr(self.kv_is_fp8):
                            if const_expr(self.use_paged_v_tma):
                                v_stage_plane_offset = Int32(
                                    consume_stage_idx * kv_plane_stage_bytes
                                )
                                _literal_pv_mma_into_ofrag_plane_fp8_raw(
                                    o_frag,
                                    p_frag,
                                    shared_ptr_to_u32(
                                        sVStageBytes.iterator
                                        + v_stage_plane_offset
                                        + Int32(0 * kv_plane_total_bytes)
                                    ),
                                    shared_ptr_to_u32(
                                        sVStageBytes.iterator
                                        + v_stage_plane_offset
                                        + Int32(1 * kv_plane_total_bytes)
                                    ),
                                    lane,
                                    warp_kv_idx,
                                    pv_row_base,
                                    num_mma_q,
                                    num_mma_kv,
                                    num_mma_d_vo,
                                    tc_upcast_stride_plane,
                                    v_scale,
                                    mDebugU32,
                                )
                            else:
                                _literal_pv_mma_into_ofrag_fp8_raw(
                                    o_frag,
                                    p_frag,
                                    shared_ptr_to_u32(
                                        sVStageBytes.iterator
                                        + Int32(consume_stage_idx * v_stage_bytes)
                                    ),
                                    lane,
                                    warp_kv_idx,
                                    pv_row_base,
                                    num_mma_q,
                                    num_mma_kv,
                                    num_mma_d_vo,
                                    tc_upcast_stride_vo,
                                    v_scale,
                                    mDebugU32,
                                )
                        elif const_expr(self.use_paged_v_tma):
                            v_stage_plane_offset = Int32(
                                consume_stage_idx * kv_plane_stage_bytes
                            )
                            _literal_pv_mma_into_ofrag_plane_bf16_packed(
                                o_frag,
                                p_frag,
                                shared_ptr_to_u32(
                                    sVStageBytes.iterator
                                    + v_stage_plane_offset
                                    + Int32(0 * kv_plane_total_bytes)
                                ),
                                shared_ptr_to_u32(
                                    sVStageBytes.iterator
                                    + v_stage_plane_offset
                                    + Int32(1 * kv_plane_total_bytes)
                                ),
                                shared_ptr_to_u32(
                                    sVStageBytes.iterator
                                    + v_stage_plane_offset
                                    + Int32(2 * kv_plane_total_bytes)
                                ),
                                shared_ptr_to_u32(
                                    sVStageBytes.iterator
                                    + v_stage_plane_offset
                                    + Int32(3 * kv_plane_total_bytes)
                                ),
                                lane,
                                warp_kv_idx,
                                pv_row_base,
                                num_mma_q,
                                num_mma_kv,
                                num_mma_d_vo,
                                tc_upcast_stride_plane,
                                v_scale,
                                mDebugU32,
                            )
                        else:
                            _literal_pv_mma_into_ofrag_bf16_packed(
                                o_frag,
                                p_frag,
                                shared_ptr_to_u32(
                                    sVStageBytes.iterator
                                    + Int32(consume_stage_idx * v_stage_bytes)
                                ),
                                lane,
                                warp_kv_idx,
                                pv_row_base,
                                num_mma_q,
                                num_mma_kv,
                                num_mma_d_vo,
                                tc_upcast_stride_vo,
                                v_scale,
                                mDebugU32,
                            )
                    _exit_thread()

                if const_expr(self.debug_dump_paged_kv_svwords):
                    if work_idx == Int32(0) and kv_head_idx == Int32(0):
                        mDebugU32 = cute.flatten(cute.recast_tensor(mO, cutlass.Uint32))
                        _dump_flat_u32_words(
                            mDebugU32,
                            cute.recast_tensor(
                                sV[None, None, consume_stage_idx],
                                cutlass.Uint32,
                            ),
                            tidx,
                            self.traits.num_threads,
                        )
                    _exit_thread()

                if const_expr(self.debug_dump_paged_kv_planewords):
                    if work_idx == Int32(0) and kv_head_idx == Int32(0):
                        mDebugU32 = cute.flatten(cute.recast_tensor(mO, cutlass.Uint32))
                        _dump_plane_stage_words_u32(
                            mDebugU32,
                            sVStageBytes,
                            consume_stage_idx,
                            kv_plane_stage_bytes,
                            kv_plane_total_bytes,
                            self.kv_tma_plane_count,
                            tidx,
                            self.traits.num_threads,
                        )
                    _exit_thread()

                if const_expr(self.use_native_fp8_pv_mma):
                    v_smem_base_addr = shared_ptr_to_u32(
                        sVStageBytes.iterator + Int32(consume_stage_idx * v_stage_bytes)
                    )
                    _literal_pv_mma_into_ofrag_mxfp8_raw(
                        o_frag,
                        p_frag,
                        v_smem_base_addr,
                        lane,
                        warp_kv_idx,
                        Int32(0)
                        if const_expr(self.traits.num_warps_kv > 1)
                        else subtile_base,
                        num_mma_q,
                        num_mma_kv,
                        num_mma_d_vo,
                        self.traits.upcast_stride_v,
                        v_scale,
                    )
                elif const_expr(self.kv_is_fp8):
                    v_smem_base_addr = shared_ptr_to_u32(
                        sVStageBytes.iterator + Int32(consume_stage_idx * v_stage_bytes)
                    )
                    if const_expr(self.use_paged_v_tma):
                        v_stage_plane_offset = Int32(
                            consume_stage_idx * kv_plane_stage_bytes
                        )
                        _literal_pv_mma_into_ofrag_plane_fp8_raw(
                            o_frag,
                            p_frag,
                            shared_ptr_to_u32(
                                sVStageBytes.iterator
                                + v_stage_plane_offset
                                + Int32(0 * kv_plane_total_bytes)
                            ),
                            shared_ptr_to_u32(
                                sVStageBytes.iterator
                                + v_stage_plane_offset
                                + Int32(1 * kv_plane_total_bytes)
                            ),
                            lane,
                            warp_kv_idx,
                            Int32(0)
                            if const_expr(self.traits.num_warps_kv > 1)
                            else subtile_base,
                            num_mma_q,
                            num_mma_kv,
                            num_mma_d_vo,
                            tc_upcast_stride_plane,
                            v_scale,
                        )
                    else:
                        _literal_pv_mma_into_ofrag_fp8_raw(
                            o_frag,
                            p_frag,
                            v_smem_base_addr,
                            lane,
                            warp_kv_idx,
                            Int32(0)
                            if const_expr(self.traits.num_warps_kv > 1)
                            else subtile_base,
                            num_mma_q,
                            num_mma_kv,
                            num_mma_d_vo,
                            self.traits.upcast_stride_v,
                            v_scale,
                        )
                else:
                    v_smem_base_addr = shared_ptr_to_u32(
                        sVStageBytes.iterator + Int32(consume_stage_idx * v_stage_bytes)
                    )
                    if const_expr(self.use_paged_v_tma):
                        v_stage_plane_offset = Int32(
                            consume_stage_idx * kv_plane_stage_bytes
                        )
                        _literal_pv_mma_into_ofrag_plane_bf16_packed(
                            o_frag,
                            p_frag,
                            shared_ptr_to_u32(
                                sVStageBytes.iterator
                                + v_stage_plane_offset
                                + Int32(0 * kv_plane_total_bytes)
                            ),
                            shared_ptr_to_u32(
                                sVStageBytes.iterator
                                + v_stage_plane_offset
                                + Int32(1 * kv_plane_total_bytes)
                            ),
                            shared_ptr_to_u32(
                                sVStageBytes.iterator
                                + v_stage_plane_offset
                                + Int32(2 * kv_plane_total_bytes)
                            ),
                            shared_ptr_to_u32(
                                sVStageBytes.iterator
                                + v_stage_plane_offset
                                + Int32(3 * kv_plane_total_bytes)
                            ),
                            lane,
                            warp_kv_idx,
                            Int32(0)
                            if const_expr(self.traits.num_warps_kv > 1)
                            else subtile_base,
                            num_mma_q,
                            num_mma_kv,
                            num_mma_d_vo,
                            tc_upcast_stride_plane,
                            v_scale,
                        )
                    else:
                        _literal_pv_mma_into_ofrag_bf16_packed(
                            o_frag,
                            p_frag,
                            v_smem_base_addr,
                            lane,
                            warp_kv_idx,
                            Int32(0)
                            if const_expr(self.traits.num_warps_kv > 1)
                            else subtile_base,
                            num_mma_q,
                            num_mma_kv,
                            num_mma_d_vo,
                            tc_upcast_stride_vo,
                            v_scale,
                        )

                if const_expr(
                    self.use_paged_v_tma and not self.use_paged_kv_tma_fp8_raw_issue
                ):
                    pipeline_v.consumer_release(kv_consumer_state)
                if const_expr(self.use_paged_k_tma or self.use_paged_v_tma):
                    kv_consumer_state.advance()
                    if next_tile_base < chunk_end:
                        if warp_linear_idx == Int32(0):
                            if const_expr(self.use_paged_k_tma):
                                if const_expr(self.use_paged_kv_tma_fp8_raw_issue):
                                    self._issue_paged_kv_tma_copy_2planes_fp8_raw(
                                        mKTmaDescFlat,
                                        kv_head_idx,
                                        sKStageBytes,
                                        Int32(consume_stage_idx * kv_plane_stage_bytes),
                                        kv_plane_total_bytes,
                                        kv_producer_state,
                                        mbar_ptr_K,
                                        self.kv_tma_copy_bytes_k,
                                        mPageTable,
                                        request_idx,
                                        next_tile_base,
                                        page_size,
                                    )
                                elif const_expr(self.kv_tma_plane_count > 2):
                                    self._issue_paged_kv_tma_copy_planes(
                                        load_K_tma0,
                                        load_K_tma1,
                                        load_K_tma2,
                                        load_K_tma3,
                                        pipeline_k,
                                        kv_producer_state,
                                        mPageTable,
                                        request_idx,
                                        next_tile_base,
                                        page_size,
                                    )
                                else:
                                    self._issue_paged_kv_tma_copy_2planes(
                                        load_K_tma0,
                                        load_K_tma1,
                                        pipeline_k,
                                        kv_producer_state,
                                        mPageTable,
                                        request_idx,
                                        next_tile_base,
                                        page_size,
                                    )
                            if const_expr(self.use_paged_v_tma):
                                if const_expr(self.use_paged_kv_tma_fp8_raw_issue):
                                    self._issue_paged_kv_tma_copy_2planes_fp8_raw(
                                        mVTmaDescFlat,
                                        kv_head_idx,
                                        sVStageBytes,
                                        Int32(consume_stage_idx * kv_plane_stage_bytes),
                                        kv_plane_total_bytes,
                                        kv_producer_state,
                                        mbar_ptr_V,
                                        self.kv_tma_copy_bytes_v,
                                        mPageTable,
                                        request_idx,
                                        next_tile_base,
                                        page_size,
                                    )
                                elif const_expr(self.kv_tma_plane_count > 2):
                                    self._issue_paged_kv_tma_copy_planes(
                                        load_V_tma0,
                                        load_V_tma1,
                                        load_V_tma2,
                                        load_V_tma3,
                                        pipeline_v,
                                        kv_producer_state,
                                        mPageTable,
                                        request_idx,
                                        next_tile_base,
                                        page_size,
                                    )
                                else:
                                    self._issue_paged_kv_tma_copy_2planes(
                                        load_V_tma0,
                                        load_V_tma1,
                                        pipeline_v,
                                        kv_producer_state,
                                        mPageTable,
                                        request_idx,
                                        next_tile_base,
                                        page_size,
                                    )
                        if const_expr(not self.use_paged_k_tma):
                            self._async_copy_paged_tile_permuted_128b(
                                mKBytes,
                                mPageTable,
                                request_idx,
                                next_tile_base,
                                next_page_idx,
                                kv_head_idx,
                                mKCache.shape[2],
                                k_row_bytes,
                                k_page_stride_bytes,
                                k_token_stride_bytes,
                                k_head_stride_bytes,
                                sKStageBytes,
                                Int32(consume_stage_idx * k_stage_bytes),
                                lane,
                                warp_linear_idx,
                                next_tile_tokens,
                                self.traits.upcast_stride_k,
                                False,
                            )
                            cute.arch.cp_async_commit_group()
                        if const_expr(not self.use_paged_v_tma):
                            self._async_copy_paged_tile_permuted_128b(
                                mVBytes,
                                mPageTable,
                                request_idx,
                                next_tile_base,
                                next_page_idx,
                                kv_head_idx,
                                mVCache.shape[2],
                                v_row_bytes,
                                v_page_stride_bytes,
                                v_token_stride_bytes,
                                v_head_stride_bytes,
                                sVStageBytes,
                                Int32(consume_stage_idx * v_stage_bytes),
                                lane,
                                warp_linear_idx,
                                next_tile_tokens,
                                self.traits.upcast_stride_v,
                                True,
                            )
                            cute.arch.cp_async_commit_group()
                        kv_producer_state.advance()
                        prefetch_base += stage_tile_rows
                elif const_expr(self.traits.num_warps_kv > 1):
                    if next_tile_base < chunk_end:
                        self._async_copy_paged_tile_permuted_128b(
                            mVBytes,
                            mPageTable,
                            request_idx,
                            next_tile_base,
                            next_page_idx,
                            kv_head_idx,
                            mVCache.shape[2],
                            v_row_bytes,
                            v_page_stride_bytes,
                            v_token_stride_bytes,
                            v_head_stride_bytes,
                            sVStageBytes,
                            Int32(consume_stage_idx * v_stage_bytes),
                            lane,
                            warp_linear_idx,
                            next_tile_tokens,
                            self.traits.upcast_stride_v,
                            True,
                        )
                        cute.arch.cp_async_commit_group()
                        prefetch_base += stage_tile_rows
                elif const_expr(self.traits.num_warps_kv == 1):
                    if next_tile_base < chunk_end:
                        self._async_copy_paged_tile_permuted_128b(
                            mVBytes,
                            mPageTable,
                            request_idx,
                            next_tile_base,
                            next_page_idx,
                            kv_head_idx,
                            mVCache.shape[2],
                            v_row_bytes,
                            v_page_stride_bytes,
                            v_token_stride_bytes,
                            v_head_stride_bytes,
                            sVStageBytes,
                            Int32(consume_stage_idx * v_stage_bytes),
                            lane,
                            warp_linear_idx,
                            next_tile_tokens,
                            self.traits.upcast_stride_v,
                            True,
                        )
                        cute.arch.cp_async_commit_group()
                        prefetch_base += stage_tile_rows

            cute.arch.sync_threads()
            if const_expr(self.num_stages == 2):
                consume_stage_idx = Int32(1) - consume_stage_idx
            tile_base += stage_tile_rows

        if const_expr(self.use_paged_k_tma or self.use_paged_v_tma):
            if warp_linear_idx == Int32(0):
                if const_expr(
                    self.use_paged_k_tma and not self.use_paged_kv_tma_fp8_raw_issue
                ):
                    pipeline_k.producer_tail(kv_producer_state.clone())
                if const_expr(
                    self.use_paged_v_tma and not self.use_paged_kv_tma_fp8_raw_issue
                ):
                    pipeline_v.producer_tail(kv_producer_state.clone())

        if const_expr(not self.has_attention_sink_bias):
            for mma_q in cutlass.range_constexpr(num_mma_q):
                for row_slot in cutlass.range_constexpr(2):
                    if m_frag[mma_q, row_slot] != -Float32.inf:
                        m_frag[mma_q, row_slot] = Float32(
                            m_frag[mma_q, row_slot] * self.softmax_scale_log2
                        )
        else:
            for mma_q in cutlass.range_constexpr(num_mma_q):
                for row_slot in cutlass.range_constexpr(2):
                    _apply_attention_sink_after_lse_scale(
                        o_frag,
                        m_frag,
                        d_frag,
                        mAttentionSinkBias,
                        mma_q,
                        row_slot,
                        q_head_idx_frag[mma_q, row_slot],
                        row_valid[mma_q, row_slot],
                        causal_k_limit[mma_q, row_slot],
                        chunk_start,
                        chunk_end,
                        warp_kv_idx,
                        num_mma_d_vo,
                        self.softmax_scale_log2,
                        self.has_attention_sink_bias,
                        self.split_kv,
                    )

        if const_expr(self.traits.num_warps_kv > 1):
            for mma_q in cutlass.range_constexpr(num_mma_q):
                for row_slot in cutlass.range_constexpr(2):
                    packed_row_local = row_local_idx[mma_q, row_slot]
                    if row_valid[mma_q, row_slot] != 0 and lane_pair_base == 0:
                        sSyncMD[warp_kv_idx, packed_row_local, 0] = m_frag[
                            mma_q, row_slot
                        ]
                        sSyncMD[warp_kv_idx, packed_row_local, 1] = d_frag[
                            mma_q, row_slot
                        ]
                    for mma_d in cutlass.range_constexpr(num_mma_d_vo):
                        dim_low = mma_d * 16 + lane_pair_base
                        dim_high = dim_low + 8
                        reg_base = row_slot * 2
                        if row_valid[mma_q, row_slot] != 0:
                            sSyncO[warp_kv_idx, packed_row_local, dim_low + 0] = o_frag[
                                mma_q, mma_d, reg_base + 0
                            ]
                            sSyncO[warp_kv_idx, packed_row_local, dim_low + 1] = o_frag[
                                mma_q, mma_d, reg_base + 1
                            ]
                            sSyncO[warp_kv_idx, packed_row_local, dim_high + 0] = (
                                o_frag[mma_q, mma_d, reg_base + 4]
                            )
                            sSyncO[warp_kv_idx, packed_row_local, dim_high + 1] = (
                                o_frag[mma_q, mma_d, reg_base + 5]
                            )
            cute.arch.sync_threads()
            if const_expr(self.debug_dump_paged_kv_extend_partials):
                if (
                    work_idx == Int32(0)
                    and kv_head_idx == Int32(0)
                    and warp_q_idx == Int32(0)
                    and warp_kv_idx == Int32(0)
                    and lane == Int32(0)
                ):
                    debug_idx = Int32(0)
                    for packed_row_local_dump in cutlass.range_constexpr(
                        self.traits.cta_tile_q
                    ):
                        packed_q_idx = packed_tile_start + Int32(packed_row_local_dump)
                        token_local_dump = packed_q_idx // group_size
                        q_group_lane_dump = packed_q_idx - token_local_dump * group_size
                        q_head_idx_dump = kv_head_idx * group_size + q_group_lane_dump
                        q_row_idx_dump = q_start + token_local_dump
                        valid_row_dump = packed_row_local_dump < packed_tile_rows
                        mOFlat[debug_idx + 0] = (
                            Float32(q_row_idx_dump).to(self.dtype_o)
                            if valid_row_dump
                            else self.dtype_o(0.0)
                        )
                        mOFlat[debug_idx + 1] = (
                            Float32(q_head_idx_dump).to(self.dtype_o)
                            if valid_row_dump
                            else self.dtype_o(0.0)
                        )
                        debug_idx += 2
                        for kv_warp_dump in cutlass.range_constexpr(
                            self.traits.num_warps_kv
                        ):
                            part_m = sSyncMD[kv_warp_dump, packed_row_local_dump, 0]
                            part_d = sSyncMD[kv_warp_dump, packed_row_local_dump, 1]
                            mOFlat[debug_idx + 0] = (
                                part_m.to(self.dtype_o)
                                if valid_row_dump
                                else self.dtype_o(0.0)
                            )
                            mOFlat[debug_idx + 1] = (
                                part_d.to(self.dtype_o)
                                if valid_row_dump
                                else self.dtype_o(0.0)
                            )
                            debug_idx += 2
                            for dim_dump in cutlass.range_constexpr(8):
                                part_o = sSyncO[
                                    kv_warp_dump, packed_row_local_dump, dim_dump
                                ]
                                mOFlat[debug_idx + dim_dump] = (
                                    part_o.to(self.dtype_o)
                                    if valid_row_dump
                                    else self.dtype_o(0.0)
                                )
                            debug_idx += 8
                _exit_thread()

        store_enabled = valid_work and warp_kv_idx == 0
        packed_row_local = Int32(0)
        q_head_idx = Int32(0)
        q_row_idx = Int32(0)
        token_local = Int32(0)
        partial_row_idx = Int32(0)
        if const_expr(self.traits.num_warps_kv > 1):
            if store_enabled:
                for mma_q in cutlass.range_constexpr(num_mma_q):
                    for row_slot in cutlass.range_constexpr(2):
                        packed_row_local = row_local_idx[mma_q, row_slot]
                        q_head_idx = q_head_idx_frag[mma_q, row_slot]
                        q_row_idx = q_row_idx_frag[mma_q, row_slot]
                        token_local = q_token_local[mma_q, row_slot]
                        valid_row_store = row_valid[mma_q, row_slot] != 0
                        merged_m = Float32(-Float32.inf)
                        merged_d = Float32(1.0)
                        inv_d = Float32(0.0)
                        merge_scale = cute.make_rmem_tensor(
                            cute.make_layout((self.traits.num_warps_kv,), stride=(1,)),
                            Float32,
                        )
                        merge_scale.fill(0.0)
                        if valid_row_store:
                            for kv_warp in cutlass.range_constexpr(
                                self.traits.num_warps_kv
                            ):
                                part_m = sSyncMD[kv_warp, packed_row_local, 0]
                                part_d = sSyncMD[kv_warp, packed_row_local, 1]
                                if merged_m == -Float32.inf:
                                    merged_m = part_m
                                    merged_d = part_d
                                elif part_m != -Float32.inf:
                                    new_m = attention_ops.fmax(merged_m, part_m)
                                    merged_d = Float32(
                                        merged_d
                                        * _exp2_approx_ftz_f32(merged_m - new_m)
                                        + part_d * _exp2_approx_ftz_f32(part_m - new_m)
                                    )
                                    merged_m = new_m
                            if merged_m != -Float32.inf:
                                inv_d = cute.arch.rcp_approx(merged_d)
                                for kv_warp in cutlass.range_constexpr(
                                    self.traits.num_warps_kv
                                ):
                                    part_m = sSyncMD[kv_warp, packed_row_local, 0]
                                    merge_scale[kv_warp] = (
                                        Float32(0.0)
                                        if part_m == -Float32.inf
                                        else _exp2_approx_ftz_f32(part_m - merged_m)
                                    )

                        for mma_d in cutlass.range_constexpr(num_mma_d_vo):
                            dim_low = mma_d * 16 + lane_pair_base
                            dim_high = dim_low + 8
                            out_low0 = Float32(0.0)
                            out_low1 = Float32(0.0)
                            out_high0 = Float32(0.0)
                            out_high1 = Float32(0.0)
                            if valid_row_store and merged_m != -Float32.inf:
                                acc_low0 = Float32(0.0)
                                acc_low1 = Float32(0.0)
                                acc_high0 = Float32(0.0)
                                acc_high1 = Float32(0.0)
                                for kv_warp in cutlass.range_constexpr(
                                    self.traits.num_warps_kv
                                ):
                                    scale = merge_scale[kv_warp]
                                    acc_low0 += (
                                        sSyncO[kv_warp, packed_row_local, dim_low + 0]
                                        * scale
                                    )
                                    acc_low1 += (
                                        sSyncO[kv_warp, packed_row_local, dim_low + 1]
                                        * scale
                                    )
                                    acc_high0 += (
                                        sSyncO[kv_warp, packed_row_local, dim_high + 0]
                                        * scale
                                    )
                                    acc_high1 += (
                                        sSyncO[kv_warp, packed_row_local, dim_high + 1]
                                        * scale
                                    )
                                out_low0 = acc_low0 * inv_d
                                out_low1 = acc_low1 * inv_d
                                out_high0 = acc_high0 * inv_d
                                out_high1 = acc_high1 * inv_d

                            if valid_row_store:
                                if const_expr(self.dtype_o == cutlass.BFloat16):
                                    sFinalStageU32[
                                        0, packed_row_local, dim_low // 2
                                    ] = pack_f32x2_to_bfloat2(out_low0, out_low1)
                                    sFinalStageU32[
                                        0, packed_row_local, dim_high // 2
                                    ] = pack_f32x2_to_bfloat2(out_high0, out_high1)
                                elif split_store_v128:
                                    sOStageU32[packed_row_local, dim_low // 2] = (
                                        pack_f32x2_to_bfloat2(out_low0, out_low1)
                                    )
                                    sOStageU32[packed_row_local, dim_high // 2] = (
                                        pack_f32x2_to_bfloat2(out_high0, out_high1)
                                    )
                                elif final_store_v128:
                                    sOStageU32[packed_row_local, dim_low // 2] = (
                                        pack_f32x2_to_bfloat2(out_low0, out_low1)
                                    )
                                    sOStageU32[packed_row_local, dim_high // 2] = (
                                        pack_f32x2_to_bfloat2(out_high0, out_high1)
                                    )
                                elif const_expr(self.split_kv):
                                    partial_row_idx = (
                                        request_partial_start
                                        + token_local * num_chunks_kv
                                        + kv_tile_idx
                                    )
                                    mO[partial_row_idx, q_head_idx, dim_low + 0] = (
                                        out_low0.to(self.dtype_o)
                                    )
                                    mO[partial_row_idx, q_head_idx, dim_low + 1] = (
                                        out_low1.to(self.dtype_o)
                                    )
                                    mO[partial_row_idx, q_head_idx, dim_high + 0] = (
                                        out_high0.to(self.dtype_o)
                                    )
                                    mO[partial_row_idx, q_head_idx, dim_high + 1] = (
                                        out_high1.to(self.dtype_o)
                                    )
                                else:
                                    mO[q_row_idx, q_head_idx, dim_low + 0] = (
                                        out_low0.to(self.dtype_o)
                                    )
                                    mO[q_row_idx, q_head_idx, dim_low + 1] = (
                                        out_low1.to(self.dtype_o)
                                    )
                                    mO[q_row_idx, q_head_idx, dim_high + 0] = (
                                        out_high0.to(self.dtype_o)
                                    )
                                    mO[q_row_idx, q_head_idx, dim_high + 1] = (
                                        out_high1.to(self.dtype_o)
                                    )
                        if valid_row_store and (lane_pair_base == 0 or self.kv_is_fp8):
                            row_lse = (
                                Float32(-Float32.inf)
                                if merged_m == -Float32.inf
                                else Float32(
                                    merged_m + cute.math.log2(merged_d, fastmath=True)
                                )
                            )
                            if const_expr(self.split_kv):
                                partial_row_idx = (
                                    request_partial_start
                                    + token_local * num_chunks_kv
                                    + kv_tile_idx
                                )
                                mLSE[partial_row_idx, q_head_idx] = row_lse
                            else:
                                mLSE[q_head_idx, q_row_idx] = row_lse
        else:
            for mma_q in cutlass.range_constexpr(num_mma_q):
                for row_slot in cutlass.range_constexpr(2):
                    packed_row_local = row_local_idx[mma_q, row_slot]
                    q_head_idx = q_head_idx_frag[mma_q, row_slot]
                    q_row_idx = q_row_idx_frag[mma_q, row_slot]
                    token_local = q_token_local[mma_q, row_slot]
                    valid_row_store = row_valid[mma_q, row_slot] != 0
                    merged_m = Float32(-Float32.inf)
                    merged_d = Float32(1.0)
                    inv_d = Float32(0.0)
                    if store_enabled and valid_row_store:
                        merged_m = m_frag[mma_q, row_slot]
                        merged_d = d_frag[mma_q, row_slot]
                        if merged_m != -Float32.inf:
                            inv_d = cute.arch.rcp_approx(merged_d)

                    for mma_d in cutlass.range_constexpr(num_mma_d_vo):
                        dim_low = mma_d * 16 + lane_pair_base
                        dim_high = dim_low + 8
                        reg_base = row_slot * 2
                        out_low0 = Float32(0.0)
                        out_low1 = Float32(0.0)
                        out_high0 = Float32(0.0)
                        out_high1 = Float32(0.0)
                        if (
                            store_enabled
                            and valid_row_store
                            and merged_m != -Float32.inf
                        ):
                            out_low0 = o_frag[mma_q, mma_d, reg_base + 0] * inv_d
                            out_low1 = o_frag[mma_q, mma_d, reg_base + 1] * inv_d
                            out_high0 = o_frag[mma_q, mma_d, reg_base + 4] * inv_d
                            out_high1 = o_frag[mma_q, mma_d, reg_base + 5] * inv_d

                        if store_enabled and valid_row_store:
                            if split_store_v128:
                                sOStageU32[packed_row_local, dim_low // 2] = (
                                    pack_f32x2_to_bfloat2(out_low0, out_low1)
                                )
                                sOStageU32[packed_row_local, dim_high // 2] = (
                                    pack_f32x2_to_bfloat2(out_high0, out_high1)
                                )
                            elif final_store_v128:
                                sOStageU32[packed_row_local, dim_low // 2] = (
                                    pack_f32x2_to_bfloat2(out_low0, out_low1)
                                )
                                sOStageU32[packed_row_local, dim_high // 2] = (
                                    pack_f32x2_to_bfloat2(out_high0, out_high1)
                                )
                            elif const_expr(self.split_kv):
                                partial_row_idx = (
                                    request_partial_start
                                    + token_local * num_chunks_kv
                                    + kv_tile_idx
                                )
                                mO[partial_row_idx, q_head_idx, dim_low + 0] = (
                                    out_low0.to(self.dtype_o)
                                )
                                mO[partial_row_idx, q_head_idx, dim_low + 1] = (
                                    out_low1.to(self.dtype_o)
                                )
                                mO[partial_row_idx, q_head_idx, dim_high + 0] = (
                                    out_high0.to(self.dtype_o)
                                )
                                mO[partial_row_idx, q_head_idx, dim_high + 1] = (
                                    out_high1.to(self.dtype_o)
                                )
                            else:
                                mO[q_row_idx, q_head_idx, dim_low + 0] = out_low0.to(
                                    self.dtype_o
                                )
                                mO[q_row_idx, q_head_idx, dim_low + 1] = out_low1.to(
                                    self.dtype_o
                                )
                                mO[q_row_idx, q_head_idx, dim_high + 0] = out_high0.to(
                                    self.dtype_o
                                )
                                mO[q_row_idx, q_head_idx, dim_high + 1] = out_high1.to(
                                    self.dtype_o
                                )
                    if (
                        store_enabled
                        and valid_row_store
                        and (lane_pair_base == 0 or self.kv_is_fp8)
                    ):
                        row_lse = (
                            Float32(-Float32.inf)
                            if merged_m == -Float32.inf
                            else Float32(
                                merged_m + cute.math.log2(merged_d, fastmath=True)
                            )
                        )
                        if const_expr(self.split_kv):
                            partial_row_idx = (
                                request_partial_start
                                + token_local * num_chunks_kv
                                + kv_tile_idx
                            )
                            mLSE[partial_row_idx, q_head_idx] = row_lse
                        else:
                            mLSE[q_head_idx, q_row_idx] = row_lse

        if const_expr(merged_store_v128):
            if valid_work:
                cute.arch.sync_threads()
                merged_chunks_per_row = self.traits.head_dim_vo // 8
                merged_chunk_linear_idx = tidx
                merged_total_chunks = packed_tile_rows * merged_chunks_per_row
                while merged_chunk_linear_idx < merged_total_chunks:
                    packed_row_local = merged_chunk_linear_idx // merged_chunks_per_row
                    chunk_idx = (
                        merged_chunk_linear_idx
                        - packed_row_local * merged_chunks_per_row
                    )
                    packed_q_idx = packed_tile_start + packed_row_local
                    token_local = packed_q_idx // group_size
                    q_group_lane = packed_q_idx - token_local * group_size
                    q_head_idx = kv_head_idx * group_size + q_group_lane
                    q_row_idx = q_start + token_local
                    u32_idx = chunk_idx * 4
                    if const_expr(self.split_kv):
                        partial_row_idx = (
                            request_partial_start
                            + token_local * num_chunks_kv
                            + kv_tile_idx
                        )
                        gmem_elem_offset = (
                            (partial_row_idx * mO.shape[1] + q_head_idx)
                            * self.traits.head_dim_vo
                        ) + chunk_idx * 8
                    else:
                        gmem_elem_offset = (
                            (q_row_idx * mO.shape[1] + q_head_idx)
                            * self.traits.head_dim_vo
                        ) + chunk_idx * 8
                    st_global_v4_u32(
                        get_ptr_as_int64(mOFlat, gmem_elem_offset),
                        sFinalStageU32[0, packed_row_local, u32_idx + 0],
                        sFinalStageU32[0, packed_row_local, u32_idx + 1],
                        sFinalStageU32[0, packed_row_local, u32_idx + 2],
                        sFinalStageU32[0, packed_row_local, u32_idx + 3],
                    )
                    merged_chunk_linear_idx += self.traits.num_threads

        if split_store_v128:
            if valid_work:
                cute.arch.sync_threads()
                split_chunks_per_row = self.traits.head_dim_vo // 8
                split_chunk_linear_idx = tidx
                split_total_chunks = packed_tile_rows * split_chunks_per_row
                while split_chunk_linear_idx < split_total_chunks:
                    packed_row_local = split_chunk_linear_idx // split_chunks_per_row
                    chunk_idx = (
                        split_chunk_linear_idx - packed_row_local * split_chunks_per_row
                    )
                    packed_q_idx = packed_tile_start + packed_row_local
                    token_local = packed_q_idx // group_size
                    q_group_lane = packed_q_idx - token_local * group_size
                    q_head_idx = kv_head_idx * group_size + q_group_lane
                    partial_row_idx = (
                        request_partial_start
                        + token_local * num_chunks_kv
                        + kv_tile_idx
                    )
                    u32_idx = chunk_idx * 4
                    gmem_elem_offset = (
                        (partial_row_idx * mO.shape[1] + q_head_idx)
                        * self.traits.head_dim_vo
                    ) + chunk_idx * 8
                    st_global_v4_u32(
                        get_ptr_as_int64(mOFlat, gmem_elem_offset),
                        sOStageU32[packed_row_local, u32_idx + 0],
                        sOStageU32[packed_row_local, u32_idx + 1],
                        sOStageU32[packed_row_local, u32_idx + 2],
                        sOStageU32[packed_row_local, u32_idx + 3],
                    )
                    split_chunk_linear_idx += self.traits.num_threads

        if final_store_v128:
            if valid_work:
                cute.arch.sync_threads()
                final_chunks_per_row = self.traits.head_dim_vo // 8
                final_chunk_linear_idx = tidx
                final_total_chunks = packed_tile_rows * final_chunks_per_row
                while final_chunk_linear_idx < final_total_chunks:
                    packed_row_local = final_chunk_linear_idx // final_chunks_per_row
                    chunk_idx = (
                        final_chunk_linear_idx - packed_row_local * final_chunks_per_row
                    )
                    packed_q_idx = packed_tile_start + packed_row_local
                    token_local = packed_q_idx // group_size
                    q_group_lane = packed_q_idx - token_local * group_size
                    q_head_idx = kv_head_idx * group_size + q_group_lane
                    q_row_idx = q_start + token_local
                    u32_idx = chunk_idx * 4
                    gmem_elem_offset = (
                        (q_row_idx * mO.shape[1] + q_head_idx) * self.traits.head_dim_vo
                    ) + chunk_idx * 8
                    st_global_v4_u32(
                        get_ptr_as_int64(mOFlat, gmem_elem_offset),
                        sOStageU32[packed_row_local, u32_idx + 0],
                        sOStageU32[packed_row_local, u32_idx + 1],
                        sOStageU32[packed_row_local, u32_idx + 2],
                        sOStageU32[packed_row_local, u32_idx + 3],
                    )
                    final_chunk_linear_idx += self.traits.num_threads


PagedExtendForwardKernel = PagedForwardKernel


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float8_e4m3fn:
        return cutlass.Float8E4M3FN
    raise TypeError(f"unsupported extend dtype {dtype}")


def _torch_to_cutlass_storage_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.float8_e4m3fn:
        return cutlass.Uint8
    return _torch_to_cutlass_dtype(dtype)


def build_extend_forward_kernel(
    traits: PagedForwardTraits,
    use_native_fp8_qk: bool,
    use_native_fp8_pv: bool,
    *,
    window_left: int = -1,
    has_attention_sink_bias: bool = False,
    has_relative_attention_bias: bool = False,
    msa_block_sparse: bool = False,
    msa_union_tile: bool = False,
    page_size: int = 64,
):
    enable_paged_kv_tma = (
        os.environ.get("FLASHINFER_EXP_SM12X_PAGED_KV_TMA", "1") != "0"
    )
    return PagedExtendForwardKernel(
        _torch_to_cutlass_dtype(traits.q_dtype),
        _torch_to_cutlass_dtype(traits.kv_dtype),
        _torch_to_cutlass_storage_dtype(traits.kv_dtype),
        _torch_to_cutlass_dtype(traits.o_dtype),
        traits=traits,
        use_native_fp8_qk=use_native_fp8_qk,
        use_native_fp8_pv=use_native_fp8_pv,
        enable_paged_kv_tma=enable_paged_kv_tma,
        window_left=window_left,
        has_attention_sink_bias=has_attention_sink_bias,
        has_relative_attention_bias=has_relative_attention_bias,
        msa_block_sparse=msa_block_sparse,
        msa_union_tile=msa_union_tile,
        page_size=page_size,
    )
