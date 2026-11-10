# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM100-family weight-by-token GATHER with independent token/SF producer warps.

One template specializes for 1CTA and 2CTA MMA. Weight A/SFA uses TMA while
four B-copy, four SFB-copy and four SFB-pack warps feed the operand rings.
Multicast clusters share work items for both MMA modes; narrow token tiles
zero-pad their SF atoms without reserving linear scratch for the padding rows.
Each CTA copies its own B slice; its SFB image spans the pair's whole token
tile. The 2CTA relay publishes local B completion to the pair leader.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cutlass.experimental.primitives as nvvm
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
# Inlined from cudnn.gemm.frost.tile_helpers
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Tile-level helpers shared by the SM100 and SM120 kernel templates."""


import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cutlass._mlir_helpers.vector as _cvec
from cutlass._mlir.dialects import llvm




def epi_subtile_spans(cols, epi_n=32):
    """Power-of-two column spans the epilogue drains a tile in (host-side)."""
    spans = []
    off = 0
    while off < cols:
        w = epi_n
        while w > cols - off:
            w //= 2
        spans.append((off, w))
        off += w
    return spans


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
def replace_tensormap_global_dim_0(desc_ptr, new_dim) -> None:
    llvm.inline_asm(
        None,
        [desc_ptr.data_ptr().toint(dtype=cutlass.Int32).ir_value(), cutlass.Int32(new_dim).ir_value()],
        "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [$0], 0, $1;",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def replace_tensormap_global_dim_1(desc_ptr, new_dim) -> None:
    nvvm.tensormap_replace(
        nvvm.TensormapField.GLOBAL_DIM,
        desc_ptr,
        new_value=cutlass.Int32(new_dim),
        ord=1,
    )


@cute.jit
def replace_tensormap_global_dim_2(desc_ptr, new_dim) -> None:
    nvvm.tensormap_replace(
        nvvm.TensormapField.GLOBAL_DIM,
        desc_ptr,
        new_value=cutlass.Int32(new_dim),
        ord=2,
    )


@cute.jit
def replace_tensormap_global_address(desc_ptr, new_address) -> None:
    nvvm.tensormap_replace(
        nvvm.TensormapField.GLOBAL_ADDRESS,
        desc_ptr,
        new_value=cutlass.Int64(new_address),
    )


@cute.jit
def fence_tensormap_release() -> None:
    nvvm.fence_proxy_release(
        nvvm.MemScope.GPU,
        from_proxy=nvvm.Proxy.GENERIC,
        to_proxy=nvvm.Proxy.TENSORMAP,
    )


@cute.jit
def fence_tensormap_acquire(desc_ptr) -> None:
    nvvm.fence_proxy_acquire(
        nvvm.MemScope.GPU,
        desc_ptr,
        TENSOR_MAP_QWORDS * 8,
        from_proxy=nvvm.Proxy.GENERIC,
        to_proxy=nvvm.Proxy.TENSORMAP,
    )




@cute.jit
def copy_tensormap_to_workspace(src_desc_ptr, dst_i64_ptr) -> None:
    """Copy the 128-byte A tensormap into ``dst_i64_ptr`` (seeds the SMEM copy)."""
    src_words = cute.make_ptr(cutlass.Int64, src_desc_ptr.toint(), mem_space=cute.AddressSpace.generic)
    for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
        dst_i64_ptr.subview(i).store((src_words + i).load())


@cute.jit
def moe_gather_row(token_index, row, group_end, source_rows):
    src = cutlass.Int32(source_rows)
    if row < group_end:
        src = cutlass.Int32(token_index[row])
    return src


@cute.jit
def moe_combine_add(ptr, value, dtype: cutlass.Constexpr):
    """Add one weighted contribution in the declared output precision."""
    if cutlass.const_expr(dtype == cutlass.Float32):
        llvm.inline_asm(
            None,
            [ptr.toint().ir_value(), value.ir_value()],
            "red.global.add.f32 [$0], $1;",
            "l,f,~{memory}",
            has_side_effects=True,
        )
    else:
        instruction = "red.global.add.noftz.bf16 [$0], $1;" if cutlass.const_expr(dtype == cutlass.BFloat16) else "red.global.add.noftz.f16 [$0], $1;"
        llvm.inline_asm(
            None,
            [ptr.toint().ir_value(), value.to(dtype).bitcast(cutlass.Int16).ir_value()],
            instruction,
            "l,h,~{memory}",
            has_side_effects=True,
        )








@cute.jit
def moe_combine_swap_bulk(
    staging,
    output,
    token_index,
    token_ks,
    scores,
    feature,
    route_begin,
    group_end,
    columns,
    feature_stride,
    output_row_stride,
    tid,
    top_k: cutlass.Constexpr,
    width: cutlass.Constexpr,
    features: cutlass.Constexpr,
):
    """Publish a transposed BF16 tile and reduce one feature segment per token."""
    cute.arch.fence_view_async_shared()
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=128)
    if tid < width:
        route = route_begin + tid
        if route < group_end:
            token = token_index[route]
            slot = token_ks[route]
            if (token >= 0) & (token < scores.shape[0]) & (slot >= 0) & (slot < top_k):
                destination = output + cutlass.Int64(token) * output_row_stride + cutlass.Int64(feature) * feature_stride
                source = staging + tid * features
                if (feature + features <= columns) & (feature_stride == 1) & (destination.toint() % 16 == 0):
                    llvm.inline_asm(
                        None,
                        [destination.toint().ir_value(), cutlass.Int32(source.toint()).ir_value(), cutlass.Int32(features * 2).ir_value()],
                        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [$0], [$1], $2;",
                        "l,r,r,~{memory}",
                        has_side_effects=True,
                    )
                    nvvm.cp_async_bulk_commit_group()
                    nvvm.cp_async_bulk_wait_group(0, read=True)
                else:
                    for f in range(features):
                        if feature + f < columns:
                            value = (source + f).load().to(cutlass.Float32)
                            moe_combine_add(destination + cutlass.Int64(f) * feature_stride, value, cutlass.BFloat16)
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=128)








def tcgen05_alloc(tmem_ptr, num_cols, *, is_exclusive=False, group=None):
    if is_exclusive:
        nvvm.tcgen05_alloc(tmem_ptr, num_cols, is_exclusive=True, group=group)
    else:
        nvvm.tcgen05_alloc(tmem_ptr, num_cols, group=group)


def tcgen05_dealloc(tmem_ptr, num_cols, *, is_exclusive=False, group=None):
    if is_exclusive:
        nvvm.tcgen05_dealloc(tmem_ptr, num_cols, is_exclusive=True, group=group)
    else:
        nvvm.tcgen05_dealloc(tmem_ptr, num_cols, group=group)




def tcgen05_mma_block_scale(mma_kind, cta_group, d, a, b, idesc, *, enable_input_d, scale_a, scale_b, scale_vec_size, collector_op=None, b_collector_op=None):
    if b_collector_op is None:
        nvvm.tcgen05_mma_block_scale(
            mma_kind,
            cta_group,
            d,
            a,
            b,
            idesc,
            enable_input_d=enable_input_d,
            scale_a=scale_a,
            scale_b=scale_b,
            scale_vec_size=scale_vec_size,
            collector_op=collector_op,
        )
    else:
        nvvm.tcgen05_mma_block_scale(
            mma_kind,
            cta_group,
            d,
            a,
            b,
            idesc,
            enable_input_d=enable_input_d,
            scale_a=scale_a,
            scale_b=scale_b,
            scale_vec_size=scale_vec_size,
            collector_op=collector_op,
            b_collector_op=b_collector_op,
        )

_copy_tensormap_to_workspace = copy_tensormap_to_workspace
_epi_subtile_spans = epi_subtile_spans
_fence_tensormap_acquire = fence_tensormap_acquire
_fence_tensormap_release = fence_tensormap_release
_moe_swizzle_tile = moe_swizzle_tile
_replace_tensormap_global_address = replace_tensormap_global_address
_replace_tensormap_global_dim_0 = replace_tensormap_global_dim_0
_replace_tensormap_global_dim_1 = replace_tensormap_global_dim_1
_replace_tensormap_global_dim_2 = replace_tensormap_global_dim_2
_tcgen05_alloc = tcgen05_alloc
_tcgen05_dealloc = tcgen05_dealloc
_tcgen05_mma_block_scale = tcgen05_mma_block_scale
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda



@cute.jit
def _issue_gather_scales(
    scratch, source, token_indices, scale_k, row_bytes: cutlass.Constexpr, row_start, warp_rows: cutlass.Constexpr, rows: cutlass.Constexpr, source_rows
):
    """Warp-cooperative linear SF loads, packed into the same MMA 128x4 atoms.

    Aligned 16-byte windows cover narrow K tiles too. Clip each copy to the
    declared SF extent rather than reading row-stride padding. Invalid tokens
    use a valid base pointer with zero source bytes. The scratch belongs to
    this producer warp and is reused only after every lane finishes packing.
    """
    lane = cute.arch.lane_idx()
    window = scale_k // 16 * 16
    for part in cutlass.range_constexpr((warp_rows + 31) // 32):
        row = row_start + lane + part * 32
        if cutlass.const_expr(rows % 128 == 0) or row < rows:
            token = token_indices.subview(row).load()
            live = (token >= 0) & (token < source.shape[0]) & (token < source_rows)
            safe_token = cutlass.Int32(0)
            if live:
                safe_token = token
            for chunk in cutlass.range_constexpr(row_bytes // 16):
                column = window + chunk * 16
                size = cutlass.Int32(0)
                safe_column = cutlass.Int32(0)
                if live & (column < source.shape[1]):
                    safe_column = cutlass.Int32(column)
                    size = cutlass.min(cutlass.Int32(16), cutlass.Int32(source.shape[1] - column))
                address = source.iterator.toint() + cutlass.Int64(safe_token) * source.stride[0] + cutlass.Int64(safe_column)
                src = cutlass.inttoptr(address, 1, cutlass.Uint8)
                nvvm.cp_async_shared_global(scratch.subview(row * row_bytes + chunk * 16), src, 16, nvvm.LoadCacheModifier.CG, cp_size=size)


@cute.jit
def _pack_gather_scales(
    dst, scratch, ready_bar, scale_k, scales: cutlass.Constexpr, row_bytes: cutlass.Constexpr, row_start, warp_rows: cutlass.Constexpr, rows: cutlass.Constexpr
):
    lane = cute.arch.lane_idx()
    nvvm.bar_warp_sync(4294967295)
    src_words = cutlass.inttoptr(scratch.data_ptr().toint(), 3, cutlass.Uint32)
    dst_words = cutlass.inttoptr(dst.data_ptr().toint(), 3, cutlass.Uint32)
    for part in cutlass.range((warp_rows * (scales // 4) + 31) // 32, unroll_full=True):
        word = lane + part * 32
        if word < warp_rows * (scales // 4):
            row = row_start + word // (scales // 4)
            kw = word % (scales // 4)
            packed = row // 128 * (128 * (scales // 4)) + kw * 128 + row % 32 * 4 + row % 128 // 32
            value = cutlass.Uint32(0)
            if cutlass.const_expr(rows % 128 == 0) or row < rows:
                value = (src_words + row * (row_bytes // 4) + scale_k % 16 // 4 + kw).load()
            (dst_words + packed).store(value)
    cute.arch.fence_view_async_shared()
    nvvm.bar_warp_sync(4294967295)
    with cute.arch.elect_one():
        if cutlass.const_expr(cta_group == 2):
            leader = cute.arch.block_idx_in_cluster() & ~1
            # Only local shared scale stores are published to the pair.
            nvvm.inline_ptx("fence.release.sync_restrict::shared::cta.cluster;")
            nvvm.mbarrier_arrive(nvvm.mapa(ready_bar, leader), scope=nvvm.MemScope.CLUSTER, relaxed=True)
        else:
            nvvm.mbarrier_arrive(ready_bar)


@cute.jit
def _load_bcast_ack(slot):
    return nvvm.inline_ptx(
        "ld.acquire.cluster.shared::cta.u32 {$r0}, [{$r1}];", write_only_types=[cutlass.Uint32], read_only_args=[cutlass.Uint32(slot.data_ptr().toint())]
    )


@cute.jit
def _publish_bcast_ack(slot):
    nvvm.inline_ptx("red.release.cluster.shared::cluster.add.u32 [{$r0}], 1;", read_only_args=[cutlass.Uint32(slot.data_ptr().toint())])


# @@FROST_GEOMETRY@@

moe_desc_slots = num_b_operands + num_sfb_operands + n_tma_outputs
_CTA_GROUP = nvvm.CTAGroup.CTA_2 if cta_group == 2 else nvvm.CTAGroup.CTA_1
_OPERAND_BARRIER_SCOPE = nvvm.MBarrierScope.CLUSTER if cta_group == 2 else None
_sf_words_resident = sf_word_stride_cols > 0
_sf_word_slots = num_sf_atoms if _sf_words_resident else 1
_sfa_word_stride = 0 if fake_dequant_a else sf_word_stride_cols
_sfb_word_stride = 0 if fake_dequant_b else sf_word_stride_cols
# GATHER produces tile-local SF rows directly in the staging ring.
# Redistribute within the CTA's 96 * 640 entry-register pool. Wider token tiles
# and distinct operands borrow registers from the SF-copy warpgroup.
_gather_cached_rows = num_b_operands * cta_tile_mnk[1] * b_packed_per_row * (b_smem_dtype.width // 8) // (128 * 16)
_gather_copy_regs = (128 if num_b_operands == 1 else 112) if _gather_cached_rows > 8 else 96
_gather_sf_copy_regs = 176 - _gather_copy_regs
# The unused token TMA argument describes packed global storage; padding is
# produced by the independent copy warps without a stronger token alignment.
_gather_tma_dtype = cutlass.Float4E2M1FN if b_gather_copy_bytes == 8 else b_tma_desc_dtype
_gather_tma_format = _tma.TensorMapDataFormat.B4X16 if b_gather_copy_bytes == 8 else b_tma_format
_gather_tma_swizzle = _tma.TensorMapSwizzle.s64b if b_gather_copy_bytes == 8 else b_tma_swizzle

if use_acc_overlap and any(_w != epi_n for _, _w in _epi_subtile_spans(epi_cols_per_mma_m, epi_n)):
    raise NotImplementedError(f"{__name__}: acc overlap reverses subtiles by index, which needs a uniform drain width")


# Per-CTA scheduler ring (replaces CLC): 2 stages, 8 int32 slot words.
SCHED_STAGES = 2
SCHED_BCAST_STAGES = 2
SCHED_SLOT_WORDS = 8

USE_PDL = True
EPI_SMEM_STAGES = 2
EPI_SYNC_BAR_ID = 1
TMEM_ALLOC_BARRIER_ID = 2
TMEM_SCALE_ONE_BARRIER_ID = 3


@cute.jit
def _moe_auto_swizzle_w(m, group_tokens, k, nt_n):
    """N-super-block width for one routed group."""
    if cutlass.const_expr(tile_swizzle_n > 0):
        return tile_swizzle_n
    budget = cutlass.Int64(swizzle_l2_budget_bytes)
    row_bytes = (cutlass.Int64(ab_max_data_bits) * k) // 8
    cap = cutlass.max(budget // (row_bytes * cgrp_tile_mnk[1]), cutlass.Int64(1))
    w = cutlass.min(cutlass.Int64(nt_n), cap)
    rows = cutlass.Int64(m)
    if cutlass.min(rows, group_tokens) * row_bytes <= budget and rows <= group_tokens:
        w = cutlass.Int64(1)
    return cutlass.Int32(w)


def _a_collector_op(g):
    if cutlass.const_expr(num_gemms == 1 or num_a_operands != 1 or mma_size_m != 1):
        return None
    if cutlass.const_expr(g == 0):
        return nvvm.Tcgen05MMACollectorOp.FILL
    if cutlass.const_expr(g == num_gemms - 1):
        return nvvm.Tcgen05MMACollectorOp.LASTUSE
    return nvvm.Tcgen05MMACollectorOp.USE


def _b_collector_op(mi):
    if cutlass.const_expr(not b_collector_ok or mma_size_m == 1):
        return None
    if cutlass.const_expr(mi == 0):
        return nvvm.Tcgen05MMACollectorOp.FILL
    if cutlass.const_expr(mi == mma_size_m - 1):
        return nvvm.Tcgen05MMACollectorOp.LASTUSE
    return nvvm.Tcgen05MMACollectorOp.USE


@cute.jit
def _fill_scale_one(tmem_base, num_cols):
    """Seed one 32-row SF image in the issuing warp's TMEM partition."""
    one = cutlass.Uint32(sf_one_word)
    one_vec = cutlass.Vector.from_elements((one,), cutlass.Uint32)
    for col in cutlass.range_constexpr(num_cols):
        nvvm.tcgen05_st(
            "32x32b",
            nvvm.make_tmem_ptr(tmem_base + col, cutlass.Uint32),
            one_vec,
        )
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)


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
    mB_0: cute.Tensor,
    b_stride_n_0: cutlass.Int64,
    token_index: cute.Tensor,
    mSFB_0: cute.Tensor,
    out_stride_m_0: cutlass.Int64,
    out_stride_n_0: cutlass.Int64,
    out_stride_l_0: cutlass.Int64,
    gate_alpha: cute.Tensor,
    up_alpha: cute.Tensor,
# @@FROST_ACTIVATION_0@@
    scale: cute.Tensor,
    tma_c_desc_0: cutlass.GridConstant[_tma.TensorMap],
) -> None:
    """Run weight-by-token MoE GEMM, including the qualified bulk COMBINE tile."""
    tma_a_descs = [tma_a_desc_0, tma_a_desc_1]
    tma_b_descs = [tma_b_desc_0]
    tma_sfa_descs = [tma_sfa_desc_0, tma_sfa_desc_1]
    tma_sfb_descs = [tma_sfb_desc_0]

    mB_list = [mB_0]
    b_stride_n_list = [b_stride_n_0]
    mSFB_list = [mSFB_0]
    tma_c_descs = [tma_c_desc_0]
    tma_c_m_major = (True,)

    mma_warp_id = 4
    tma_warp_id = 5
    scheduler_warp_id = 6
    unused_warp_id = 7
    num_epilogue_warps = 4
    epi_reg_count = 216
    prod_reg_count = 64

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    elect_one = nvvm.elect_sync()

    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    gridx = cute.arch.grid_dim()[0]

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    cluster_size = cluster_m * cluster_n * cluster_shape_mnk[2]

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    m_rank = cta_rank_in_cluster % cluster_m
    n_rank = cta_rank_in_cluster // cluster_m
    if cutlass.const_expr(cta_group == 2):
        pair_member = m_rank % cta_group
        pair_m_idx = m_rank // cta_group
        is_pair_leader = pair_member == 0
        pair_leader_rank = pair_m_idx * cta_group + n_rank * cluster_m
    else:
        pair_member = 0
        pair_m_idx = m_rank
        is_pair_leader = True
        pair_leader_rank = cta_rank_in_cluster

    if cutlass.const_expr(cta_group == 2):
        sched_counter_ptr = cute.make_ptr(
            cutlass.Int32,
            (tma_workspace.iterator.raw_ptr() + grid_num_clusters * cluster_m * cluster_n * moe_desc_slots * TENSOR_MAP_QWORDS).toint(),
            mem_space=cute.AddressSpace.generic,
        )

    if warp_idx == mma_warp_id:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
        for _i in cutlass.range_constexpr(num_sfa_operands):
            nvvm.prefetch_tensormap(tma_sfa_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())
        for _j in cutlass.range_constexpr(num_sfb_operands):
            nvvm.prefetch_tensormap(tma_sfb_descs[_j].get_ptr())

        for _ci in cutlass.range_constexpr(n_tma_outputs):
            nvvm.prefetch_tensormap(tma_c_descs[_ci].get_ptr())

    if cutlass.const_expr(cta_group == 1):
        sched_counter_ptr = cute.make_ptr(
            cutlass.Int32,
            (tma_workspace.iterator.raw_ptr() + grid_num_clusters * cluster_m * cluster_n * moe_desc_slots * TENSOR_MAP_QWORDS).toint(),
            mem_space=cute.AddressSpace.generic,
        )

    a_pattern = 0
    for n_idx in cutlass.range_constexpr(cluster_n):
        a_pattern = a_pattern | (1 << (n_idx * cluster_m))
    if cutlass.const_expr(cta_group == 1):
        b_pattern = (1 << cluster_m) - 1
    else:
        b_pattern = 0
        for pm_idx in cutlass.range_constexpr(cluster_m // 2):
            b_pattern = b_pattern | (1 << (pm_idx * 2))

    if cutlass.const_expr(multicast_a):
        if cutlass.const_expr(cta_group == 1):
            tma_mcast_mask_a = cutlass.Int16(a_pattern) << m_rank
        else:
            tma_mcast_mask_a = cutlass.Int16(a_pattern << m_rank)
    else:
        if cutlass.const_expr(cta_group == 1):
            tma_mcast_mask_a = cutlass.Int16(1) << cta_rank_in_cluster
        else:
            tma_mcast_mask_a = cutlass.Int16(1 << cta_rank_in_cluster)
    if cutlass.const_expr(cta_group == 1):
        if cutlass.const_expr(multicast_b):
            tma_mcast_mask_b = cutlass.Int16(b_pattern) << (n_rank * cluster_m)
        else:
            tma_mcast_mask_b = cutlass.Int16(1) << cta_rank_in_cluster

        a_part_arrive = cutlass.Int16(a_pattern) << m_rank
        b_part_arrive = cutlass.Int16(b_pattern) << (n_rank * cluster_m)
        if cutlass.const_expr(ab_empty_full_mask):
            ab_empty_arrive_mask = cutlass.Int16((1 << cluster_size) - 1)
        else:
            ab_empty_arrive_mask = a_part_arrive | b_part_arrive
    else:
        if cutlass.const_expr(multicast_b):
            tma_mcast_mask_b = cutlass.Int16((b_pattern << pair_member) << (n_rank * cluster_m))
        else:
            tma_mcast_mask_b = cutlass.Int16(1 << cta_rank_in_cluster)

    _smem_sys_reserved = cutlass.Array(cutlass.Int8, 1024, space=cutlass.AddressSpace.smem, alignment=1)
    if cutlass.const_expr(moe_combine_swap_bulk_enabled):
        combine_staging = cutlass.Array(cutlass.BFloat16, 128 * mma_size_m * epi_n, space=cutlass.AddressSpace.smem, alignment=16)

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    sf_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)

    tmem_dealloc_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    sched_storage = cutlass.Array(
        cutlass.Int32,
        SCHED_STAGES * SCHED_SLOT_WORDS,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    sched_full_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    sched_empty_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    sched_bcast_slot = cutlass.Array(cutlass.Int32, SCHED_BCAST_STAGES, space=cutlass.AddressSpace.smem, alignment=16)
    sched_bcast_full_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_BCAST_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    bcast_ack_count = cutlass.Array(cutlass.Uint32, SCHED_BCAST_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    # Retain token descriptor storage to preserve the compiler-modeled SMEM layout.
    tma_b_desc_smem_list = [
        cutlass.Array(
            cutlass.Int64,
            TENSOR_MAP_QWORDS,
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        for _ in range(num_b_operands)
    ]
    tma_sfb_desc_smem_list = [
        cutlass.Array(
            cutlass.Int64,
            TENSOR_MAP_QWORDS,
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        for _ in range(num_sfb_operands)
    ]

    epi_subtile_elems = epi_stage_rows * epi_row_elems * epi_slot_widen
    smem_d_ptr = cutlass.Array(
        cd_dtype,
        epi_subtile_elems * EPI_SMEM_STAGES,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    tma_c_desc_smem = cutlass.Array(
        cutlass.Int64,
        TENSOR_MAP_QWORDS * n_tma_outputs,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )

    sA_elems = sA_packed_elems
    sB_elems = sB_packed_elems
    sf_gather_scratch = [
        cutlass.Array(cutlass.Uint8, ((gather_token_rows * gather_sf_row_bytes + 127) // 128) * 128 * ab_stages, space=cutlass.AddressSpace.smem, alignment=128)
        for _ in range(num_sfb_operands if gather_scratch_before_rings else 0)
    ]
    smem_sfa_list = [
        cutlass.Array(
            cutlass.Uint8,
            sfa_smem_bytes * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_sfa_operands)
    ]

    smem_sfb_list = [
        cutlass.Array(
            cutlass.Uint8,
            sfb_smem_bytes * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_sfb_operands)
    ]

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
    # Keep scratch low when descriptor roots permit it; large graphs place
    # it after the aligned operand rings to preserve their address range.
    if cutlass.const_expr(not gather_scratch_before_rings):
        sf_gather_scratch = [
            cutlass.Array(
                cutlass.Uint8, ((gather_token_rows * gather_sf_row_bytes + 127) // 128) * 128 * ab_stages, space=cutlass.AddressSpace.smem, alignment=128
            )
            for _ in range(num_sfb_operands)
        ]

    if cutlass.const_expr(cta_group == 2):
        acc_empty_count = num_epilogue_warps * 2
    if cutlass.const_expr(ab_empty_full_mask):
        if cutlass.const_expr(cta_group == 1):
            ab_empty_count = cluster_size
        else:
            ab_empty_count = cluster_size // cta_group
    else:
        if cutlass.const_expr(cta_group == 1):
            ab_empty_count = cluster_m + cluster_n - 1
        else:
            ab_empty_count = (cluster_m // cta_group) + cluster_n - 1
    sched_empty_count = 1 + 1 + num_epilogue_warps + 4 + (8 if num_sfb_operands else 0) + (1 if cta_group == 2 else 0)
    if cutlass.const_expr(cta_group == 2):
        b_local_ready = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem, alignment=8)
    sf_local_ready = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem, alignment=8)
    if cutlass.const_expr(num_sfb_operands > 0):
        sf_token_indices = cutlass.Array(cutlass.Int32, gather_token_rows, space=cutlass.AddressSpace.smem, alignment=128)
    if warp_idx == 0:
        if cutlass.const_expr(cta_group == 2):
            if cutlass.const_expr(use_acc_overlap):
                if elect_one:
                    nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, num_epilogue_warps)
            else:
                if elect_one:
                    nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, 32)
        else:
            for i in range(ab_stages):
                if elect_one:
                    nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1 + (2 if cta_group == 2 else 128))
                if elect_one:
                    nvvm.mbarrier_init(sf_full_mbar_ptr.subview(i), 1 + cta_group * num_sfb_operands * 4)
                if elect_one:
                    nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
            for i in range(acc_stages):
                if elect_one:
                    nvvm.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(acc_empty_mbar_ptr.subview(i), num_epilogue_warps)
            if cutlass.const_expr(use_acc_overlap):
                if elect_one:
                    nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, num_epilogue_warps)
        if cutlass.const_expr(cta_group == 2):
            for i in range(ab_stages):
                if elect_one:
                    nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1 + (2 if cta_group == 2 else 128))
                if elect_one:
                    nvvm.mbarrier_init(sf_full_mbar_ptr.subview(i), 1 + cta_group * num_sfb_operands * 4)
                if elect_one:
                    nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
            for i in range(acc_stages):
                if elect_one:
                    nvvm.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(acc_empty_mbar_ptr.subview(i), acc_empty_count)
        for i in range(SCHED_STAGES):
            if elect_one:
                nvvm.mbarrier_init(sched_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(sched_empty_mbar_ptr.subview(i), sched_empty_count)
        for i in range(SCHED_BCAST_STAGES):
            if elect_one:
                nvvm.mbarrier_init(sched_bcast_full_mbar_ptr.subview(i), 1)
            if elect_one:
                bcast_ack_count.subview(i).store(cutlass.Uint32(0))
    if warp_idx == 0:
        if elect_one:
            for stage_init in cutlass.range_constexpr(ab_stages):
                nvvm.mbarrier_init(sf_local_ready.subview(stage_init), 128)
                if cutlass.const_expr(cta_group == 2):
                    nvvm.mbarrier_init(b_local_ready.subview(stage_init), 128)
    nvvm.fence_mbarrier_init()
    if cutlass.const_expr(cta_group == 1):

        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            nvvm.barrier_cluster_arrive_relaxed()
            nvvm.barrier_cluster_wait()
        else:
            nvvm.barrier_cta_sync(0)
    else:
        nvvm.barrier_cluster_arrive_relaxed()

    sA_bytes = sA_elems * (a_smem_dtype.width // 8)
    sB_bytes = sB_elems * (b_smem_dtype.width // 8)
    # The pair leader issues ONE expect_tx for both CTAs, so it counts twice.
    ab_only_copy_bytes = num_a_operands * sA_tma_bytes * cta_group
    sf_only_copy_bytes = (num_sfa_operands * sfa_smem_bytes + num_sfb_operands * sfb_smem_bytes - (num_sfb_operands * sfb_smem_bytes)) * cta_group
    if cutlass.const_expr(cta_group == 2):
        pair_n_size = cgrp_tile_mnk[1] // cluster_n
    epi_rows_per_mma_m = cta_tile_mnk[0] // mma_size_m
    tmem_alloc_bar_count = (num_epilogue_warps + 1) * 32
    if cutlass.const_expr(cta_group == 2):

        nvvm.barrier_cluster_wait()
        nvvm.barrier_cta_sync(0)

    pass

    vsize = epi_chunk_elems

    M = m
    N = n
    clusters_along_m = cute.ceil_div(cutlass.Int32(M), cgrp_tile_mnk[0])
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    first_token_arr = cutlass.make_array_view(first_token_offset)

    if warp_idx == scheduler_warp_id:
        if cutlass.const_expr(not moe_combine_swap_bulk_enabled):
            nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        full_warp_mask = 0xFFFFFFFF
        shfl_idx_clamp = 0x1F
        shfl_up_clamp = 0
        lane = cute.arch.lane_idx()
        gemm_s = cutlass.Int32(first_token_arr[num_groups])
        sched_stage = cutlass.Int32(0)
        sched_empty_phase = cutlass.Int32(1)
        bcast_stage = cutlass.Int32(0)
        bcast_full_phase = cutlass.Int32(0)
        bcast_empty_phase = cutlass.Int32(1)
        last_bcast_stage = cutlass.Int32(0)
        last_bcast_ack_expected = cutlass.Uint32(0)
        bcast_round = cutlass.Uint32(0)
        linear_idx = cutlass.Int32(0)
        start_linear_idx = cutlass.Int32(0)
        total_tiles = cutlass.Int32(0)
        start_sf_block_n = cutlass.Int32(0)
        total_sf_blocks_n = cutlass.Int32(0)
        group_idx = cutlass.Int32(0)
        is_tile_valid = cutlass.Int32(1)
        cached_next_end = cutlass.Int32(0)
        if lane < num_groups:
            cached_next_end = cutlass.Int32(first_token_arr[lane + 1])
        else:
            cached_next_end = gemm_s
        tile_lower_bound = nvvm.shfl_sync(full_warp_mask, cached_next_end, 1, shfl_up_clamp, nvvm.Shfl.UP)
        cached_next_begin = cutlass.Int32(0)
        if lane != 0:
            cached_next_begin = tile_lower_bound

        while is_tile_valid != 0:
            # Weight multicast needs cluster-wide work agreement for both
            # singleton and paired MMA instructions.
            if cutlass.const_expr(cluster_size > 1):
                if cta_rank_in_cluster == 0:
                    while _load_bcast_ack(bcast_ack_count.subview(bcast_stage)) != bcast_round * cluster_size:
                        pass
                    claimed = cutlass.Int32(0)
                    if lane == 0:
                        claimed = nvvm.atomicrmw(
                            "add",
                            sched_counter_ptr,
                            cutlass.Int32(1),
                            mem_order="relaxed",
                            syncscope="gpu",
                        )
                    claimed = nvvm.shfl_sync(full_warp_mask, claimed, 0, shfl_idx_clamp, nvvm.Shfl.IDX)
                    if lane < cluster_size:
                        (nvvm.mapa(sched_bcast_slot.subview(bcast_stage), lane)).store(claimed)
                        nvvm.mbarrier_arrive(nvvm.mapa(sched_bcast_full_mbar_ptr.subview(bcast_stage), lane))
                while not nvvm.mbarrier_try_wait_parity(
                    sched_bcast_full_mbar_ptr.subview(bcast_stage),
                    bcast_full_phase,
                    time_limit=10_000_000,
                    scope=_OPERAND_BARRIER_SCOPE,
                ):
                    pass
                linear_idx = (sched_bcast_slot.subview(bcast_stage)).load()
                # Finish every lane's slot reads before the elected release.
                nvvm.bar_warp_sync(0xFFFFFFFF)
                # Release orders the mailbox reads before the producer may reuse it.
                if lane == 0:
                    _publish_bcast_ack(nvvm.mapa(bcast_ack_count.subview(bcast_stage), 0))
                if cutlass.const_expr(cluster_size > 1):
                    last_bcast_stage = bcast_stage
                    last_bcast_ack_expected = (bcast_round + 1) * cluster_size
                bcast_stage += 1
                if bcast_stage == SCHED_BCAST_STAGES:
                    bcast_stage = cutlass.Int32(0)
                    bcast_full_phase = bcast_full_phase ^ 1
                    bcast_round += 1

            else:
                claimed = cutlass.Int32(0)
                if lane == 0:
                    claimed = nvvm.atomicrmw("add", sched_counter_ptr, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu")
                linear_idx = nvvm.shfl_sync(full_warp_mask, claimed, 0, shfl_idx_clamp, nvvm.Shfl.IDX)

            group_begin = cached_next_begin
            group_end = cached_next_end

            if linear_idx >= start_linear_idx + total_tiles:
                group_idx += lane
                is_search_live = cutlass.Int32(1)
                while is_search_live != 0:
                    cached_group_begin = cached_next_begin
                    cached_group_end = cached_next_end
                    tile_start_idx = nvvm.shfl_sync(
                        full_warp_mask,
                        cached_next_end,
                        31,
                        shfl_idx_clamp,
                        nvvm.Shfl.IDX,
                    )
                    next_end_group = group_idx + 32 + 1
                    if next_end_group <= num_groups:
                        cached_next_end = cutlass.Int32(first_token_arr[next_end_group])
                    else:
                        cached_next_end = gemm_s
                    tile_lower_bound = nvvm.shfl_sync(
                        full_warp_mask,
                        cached_next_end,
                        1,
                        shfl_up_clamp,
                        nvvm.Shfl.UP,
                    )
                    if lane != 0:
                        cached_next_begin = tile_lower_bound
                    else:
                        cached_next_begin = tile_start_idx

                    group_tokens = cached_group_end - cached_group_begin
                    total_tiles = cute.ceil_div(group_tokens, cgrp_tile_mnk[1]) * clusters_along_m
                    total_sf_blocks_n = cute.ceil_div(group_tokens, 128)
                    prefix_tiles = total_tiles
                    prefix_sf = total_sf_blocks_n
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
                    start_linear_idx += prefix_tiles - total_tiles
                    start_sf_block_n += prefix_sf - total_sf_blocks_n
                    thread_succeed = nvvm.vote_sync(
                        full_warp_mask,
                        linear_idx < start_linear_idx + total_tiles,
                        nvvm.VoteSync.BALLOT,
                    )
                    if thread_succeed != 0:
                        winning_lane = cutlass.Int32(31) - cute.arch.bfind(cute.arch.brev(thread_succeed)).to(cutlass.Int32)
                        group_idx = nvvm.shfl_sync(
                            full_warp_mask,
                            group_idx,
                            winning_lane,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        start_linear_idx = nvvm.shfl_sync(
                            full_warp_mask,
                            start_linear_idx,
                            winning_lane,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        total_tiles = nvvm.shfl_sync(
                            full_warp_mask,
                            total_tiles,
                            winning_lane,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        start_sf_block_n = nvvm.shfl_sync(
                            full_warp_mask,
                            start_sf_block_n,
                            winning_lane,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        tile_start_idx = nvvm.shfl_sync(
                            full_warp_mask,
                            cached_group_begin,
                            winning_lane,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        group_end_idx = group_idx + lane + 1
                        if group_end_idx <= num_groups:
                            cached_next_end = cutlass.Int32(first_token_arr[group_end_idx])
                        else:
                            cached_next_end = gemm_s
                        tile_lower_bound = nvvm.shfl_sync(
                            full_warp_mask,
                            cached_next_end,
                            1,
                            shfl_up_clamp,
                            nvvm.Shfl.UP,
                        )
                        if lane != 0:
                            cached_next_begin = tile_lower_bound
                        else:
                            cached_next_begin = tile_start_idx
                        group_begin = cached_next_begin
                        group_end = cached_next_end
                        is_search_live = cutlass.Int32(0)
                    else:
                        group_idx += 32
                        first_lane_group = nvvm.shfl_sync(
                            full_warp_mask,
                            group_idx,
                            0,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        if first_lane_group >= num_groups:
                            is_tile_valid = cutlass.Int32(0)
                            is_search_live = cutlass.Int32(0)
                        else:
                            next_start_linear_idx = start_linear_idx + total_tiles
                            start_linear_idx = nvvm.shfl_sync(
                                full_warp_mask,
                                next_start_linear_idx,
                                31,
                                shfl_idx_clamp,
                                nvvm.Shfl.IDX,
                            )
                            next_start_sf = start_sf_block_n + total_sf_blocks_n
                            start_sf_block_n = nvvm.shfl_sync(
                                full_warp_mask,
                                next_start_sf,
                                31,
                                shfl_idx_clamp,
                                nvvm.Shfl.IDX,
                            )

            coord_expert = cutlass.Int32(0)
            cluster_tile_m = cutlass.Int32(0)
            coord_n = cutlass.Int32(0)
            if is_tile_valid != 0:
                local_linear_idx = linear_idx - start_linear_idx
                group_nt_n = total_tiles // clusters_along_m
                cluster_tile_m, coord_n = _moe_swizzle_tile(
                    local_linear_idx,
                    clusters_along_m,
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
                (slot.subview(1)).store(cluster_tile_m)
                (slot.subview(2)).store(coord_n)
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

        if cutlass.const_expr(cluster_size > 1):
            if cta_rank_in_cluster == 0:
                while _load_bcast_ack(bcast_ack_count.subview(last_bcast_stage)) != last_bcast_ack_expected:
                    pass

    if warp_idx == tma_warp_id:
        if cutlass.const_expr(not moe_combine_swap_bulk_enabled):
            nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        ab_empty_phase_bit = cutlass.Int32(1)
        ab_iter = cutlass.Int32(0)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        if cutlass.const_expr(cta_group == 2):
            logical_cta_tile_n = cgrp_tile_mnk[1] // cluster_n

        lane = tidx % 32
        block_linear = bidx + bidy * gridx
        sfb_block_bytes = 512 * (((k // block_size) + 3) // 4)
        nvvm.bar_warp_sync(0xFFFFFFFF)

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
            # Finish every lane's slot reads before the elected release.
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                coord_m_weight = tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
                coord_n_group = tile_n * cgrp_tile_mnk[1] + n_rank * (cta_tile_mnk[1] * cta_group)
                coord_n_desc = group_begin + coord_n_group
                coord_n_per_cta = coord_n_desc + pair_member * cta_tile_mnk[1]
                sfa_m_block = coord_m_weight // 128
                if cutlass.const_expr(moe_combine_swap_bulk_enabled):
                    sfb_n_block = start_sf_block_n + coord_n_group // 128
                else:
                    sfb_n_block = coord_n_desc // 128

                for k_tile_idx in range(num_k_tiles):
                    stage = ab_iter % ab_stages
                    if stage == 0 and ab_iter != 0:
                        ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(
                        ab_empty_mbar_ptr.subview(stage),
                        ab_empty_phase_bit,
                        time_limit=10_000_000,
                        scope=_OPERAND_BARRIER_SCOPE,
                    ):
                        pass

                    coord_k = k_tile_idx * cta_tile_mnk[2]
                    coord_sf_k = k_tile_idx * sf_tma_box_k
                    if cutlass.const_expr(cta_group == 1):
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), ab_only_copy_bytes)
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(sf_full_mbar_ptr.subview(stage), sf_only_copy_bytes)
                    else:

                        if is_pair_leader:
                            if elect_one:
                                nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), ab_only_copy_bytes)
                            if elect_one:
                                nvvm.mbarrier_arrive_expect_tx(sf_full_mbar_ptr.subview(stage), sf_only_copy_bytes)
                    a_issue = (not multicast_a) or (n_rank == 0)
                    if cutlass.const_expr(a_mcast_slices > 1):
                        a_data_issue = True
                        _a_off = n_rank * (cta_tile_mnk[0] // a_mcast_slices)
                    else:
                        a_data_issue = a_issue
                        _a_off = 0
                    b_issue = (not multicast_b) or (pair_m_idx == 0)
                    if cutlass.const_expr(b_mcast_slices > 1):
                        b_data_issue = True
                        _b_off = pair_m_idx * (cta_tile_mnk[1] // b_mcast_slices)
                    else:
                        b_data_issue = b_issue
                        _b_off = 0
                    if a_issue:
                        for _ai in cutlass.range_constexpr(num_sfa_operands):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    smem_sfa_list[_ai].subview(sfa_smem_bytes * stage),
                                    tma_sfa_descs[_ai].get_ptr(),
                                    (0, coord_sf_k, sfa_m_block, coord_expert),
                                    sf_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=_CTA_GROUP,
                                )

                    if a_data_issue:
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            sA_stage = smem_a_list[_ai].subview(sA_elems * stage)
                            if cutlass.const_expr(a_is_m_major):
                                for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sA_stage.subview(m_group * a_tma_group_elems * cta_tile_mnk[2]),
                                            tma_a_descs[_ai].get_ptr(),
                                            (coord_m_weight + m_group * a_tma_group_elems, coord_k, coord_expert),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_a,
                                            group=_CTA_GROUP,
                                        )
                            else:
                                for _am in cutlass.range_constexpr(cta_tile_mnk[0] // a_mcast_slices // a_tma_box_m):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sA_stage.subview(_a_off * a_packed_per_row + _am * a_tma_box_m * a_packed_per_row),
                                            tma_a_descs[_ai].get_ptr(),
                                            (coord_k, coord_m_weight + _a_off + _am * a_tma_box_m, coord_expert),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_a,
                                            group=_CTA_GROUP,
                                        )
                    ab_iter += 1

        tail_stage = ab_iter % ab_stages
        tail_phase = ab_empty_phase_bit
        if tail_stage == 0 and ab_iter != 0:
            tail_phase = tail_phase ^ 1
        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            for _ in range(ab_stages):
                while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(tail_stage), tail_phase, time_limit=10_000_000, scope=_OPERAND_BARRIER_SCOPE):
                    pass
                tail_stage = tail_stage + 1
                if tail_stage == ab_stages:
                    tail_stage = cutlass.Int32(0)
                    tail_phase = tail_phase ^ 1

    if cutlass.const_expr(cta_group == 2):
        pair_mask = cutlass.Int16(3) << pair_leader_rank
        a_arrive_pattern = 0
        for n_idx in cutlass.range_constexpr(cluster_n):
            a_arrive_pattern = a_arrive_pattern | (1 << (n_idx * cluster_m))
        b_arrive_pattern = 0
        for m_idx in cutlass.range_constexpr(cluster_m):
            b_arrive_pattern = b_arrive_pattern | (1 << m_idx)
        a_part = a_arrive_pattern << m_rank
        a_part = a_part | (a_part << 1)
        b_part = b_arrive_pattern << (n_rank * cluster_m)
        if cutlass.const_expr(ab_empty_full_mask):
            ab_empty_arrive_mask = cutlass.Int16((1 << cluster_size) - 1)
        else:
            ab_empty_arrive_mask = cutlass.Int16(a_part | b_part)
    if warp_idx == mma_warp_id:
        if cutlass.const_expr(not moe_combine_swap_bulk_enabled):
            nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        _tcgen05_alloc(
            tmem_ptr_i32,
            cutlass.Int32(num_tmem_alloc_cols),
            is_exclusive=tmem_alloc_exclusive,
            group=_CTA_GROUP,
        )
        nvvm.bar_warp_sync(0xFFFFFFFF)
        nvvm.barrier_cta_arrive(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        if cutlass.const_expr(fake_dequant_a or fake_dequant_b):
            nvvm.barrier_cta_sync(
                barrier_id=TMEM_SCALE_ONE_BARRIER_ID,
                thread_count=tmem_alloc_bar_count,
            )
        if cutlass.const_expr(cta_group == 1):
            ab_full_phase_bit = cutlass.Int32(0)
            ab_iter = cutlass.Int32(0)
            acc_empty_phase_bit = cutlass.Int32(1)
            tile_iter = cutlass.Int32(0)
            is_valid = cutlass.Int32(1)
            sched_stage = cutlass.Int32(0)
            sched_full_phase = cutlass.Int32(0)
            acc_stage = cutlass.Int32(0)

            if cutlass.const_expr(idesc_is_omma):
                idesc_by_j = [
                    cutlass.experimental.primitives.Tcgen05MxOmmaInstrDesc.build(
                        a_dtype=idesc_a_dtype,
                        b_dtype=idesc_b_dtype,
                        scale_format=sf_scale_format,
                        n_dim=mma_n_dim,
                        m_dim=mma_m_dim,
                        a_major=mma_a_major,
                        b_major=mma_b_major,
                        a_sf_id=j * sf_scales_per_inst,
                        b_sf_id=j * sf_scales_per_inst,
                        k_dim=mma_k_dim_mode,
                    )
                    for j in range(sf_insts_per_atom)
                ]
            else:
                idesc_by_j = [
                    cutlass.experimental.primitives.Tcgen05MxInstrDesc.build(
                        a_dtype=idesc_a_dtype,
                        b_dtype=idesc_b_dtype,
                        scale_format=sf_scale_format,
                        n_dim=mma_n_dim,
                        m_dim=mma_m_dim,
                        a_major=mma_a_major,
                        b_major=mma_b_major,
                        a_sf_id=j * sf_scales_per_inst,
                        b_sf_id=j * sf_scales_per_inst,
                        k_dim=mma_k_dim_mode,
                    )
                    for j in range(sf_insts_per_atom)
                ]

            sfa_tmem_bases = [(base_row_id << 16) | (base_col_id_root + sfa_col_bases[i]) for i in range(num_a_operands)]
            sfb_tmem_bases = [(base_row_id << 16) | (base_col_id_root + sfb_col_bases[j]) for j in range(num_b_operands)]
            s2t_shape, s2t_multicast = nvvm.S2TCopyMode.S2T_32x128b_WARPX4
            sfb_scale_ptrs = [[nvvm.make_tmem_ptr(b + _sfb_word_stride * w, cutlass.Float32) for b in sfb_tmem_bases] for w in range(_sf_word_slots)]
            sfa_dst_ptrs = [
                [
                    [
                        [
                            nvvm.make_tmem_ptr(sfa_tmem_bases[i] + _sfa_word_stride * w + m * registers_per_block + a * registers_per_atom, cutlass.Float32)
                            for a in range(word_atoms)
                        ]
                        for m in range(mma_size_m)
                    ]
                    for i in range(num_a_operands)
                ]
                for w in range(_sf_word_slots)
            ]
            sfb_dst_ptrs = [
                [
                    [
                        [
                            nvvm.make_tmem_ptr(sfb_tmem_bases[j] + _sfb_word_stride * w + (a * num_blocks_n + m) * registers_per_atom, cutlass.Float32)
                            for a in range(word_atoms)
                        ]
                        for m in range(num_blocks_n)
                    ]
                    for j in range(num_b_operands)
                ]
                for w in range(_sf_word_slots)
            ]
            desc_a_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_a_list[i],
                    leading_byte_offset=a_smem_desc_leading_byte_offset,
                    stride_byte_offset=a_smem_desc_stride_byte_offset,
                    layout=a_smem_swizzle,
                )
                for i in range(num_a_operands)
            ]
            desc_b_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_b_list[j],
                    leading_byte_offset=b_smem_desc_leading_byte_offset,
                    stride_byte_offset=b_smem_desc_stride_byte_offset,
                    layout=b_smem_swizzle,
                )
                for j in range(num_b_operands)
            ]
            desc_sfa_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_sfa_list[i],
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    layout=cutlass.experimental.primitives.Tcgen05SmemSwizzle.NONE,
                )
                for i in range(num_sfa_operands)
            ]
            desc_sfb_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_sfb_list[j],
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    layout=cutlass.experimental.primitives.Tcgen05SmemSwizzle.NONE,
                )
                for j in range(num_sfb_operands)
            ]
            while is_valid != 0:
                while not nvvm.mbarrier_try_wait_parity(
                    sched_full_mbar_ptr.subview(sched_stage),
                    sched_full_phase,
                    time_limit=10_000_000,
                ):
                    pass
                is_valid = (sched_storage.subview(sched_stage * SCHED_SLOT_WORDS).subview(3)).load()
                # Finish every lane's slot reads before the elected release.
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if elect_one:
                    nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
                sched_stage += 1
                if sched_stage == SCHED_STAGES:
                    sched_stage = cutlass.Int32(0)
                    sched_full_phase = sched_full_phase ^ 1

                if is_valid != 0:
                    acc_stage = tile_iter % acc_stages
                    if acc_stage == 0 and tile_iter != 0:
                        acc_empty_phase_bit = acc_empty_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(
                        acc_empty_mbar_ptr.subview(acc_stage),
                        acc_empty_phase_bit,
                        time_limit=10_000_000,
                        scope=_OPERAND_BARRIER_SCOPE,
                    ):
                        pass

                    if cutlass.const_expr(use_acc_overlap):
                        acc_base_col = base_col_id_root + (tile_iter % 2) * acc_stage_stride
                    else:
                        acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                    acc_tmem_ptrs = [
                        [
                            nvvm.make_tmem_ptr(
                                (base_row_id << 16) | (acc_base_col + g * acc_gemm_stride + mi * epi_cols_per_mma_m),
                                cutlass.Float32,
                            )
                            for mi in range(mma_size_m)
                        ]
                        for g in range(num_gemms)
                    ]

                    scale_d = cutlass.Boolean(False)
                    for k_tile_idx in range(num_k_tiles):
                        stage = ab_iter % ab_stages
                        if stage == 0 and ab_iter != 0:
                            ab_full_phase_bit = ab_full_phase_bit ^ 1

                        desc_a_bases = [desc_a_roots[i].advance_start_address(sA_bytes * stage) for i in range(num_a_operands)]
                        desc_b_bases = [desc_b_roots[j].advance_start_address(sB_bytes * stage) for j in range(num_b_operands)]
                        desc_sfa_bases = [desc_sfa_roots[i].advance_start_address(sfa_smem_bytes * stage) for i in range(num_sfa_operands)]
                        desc_sfb_bases = [desc_sfb_roots[j].advance_start_address(sfb_smem_bytes * stage) for j in range(num_sfb_operands)]

                        while not nvvm.mbarrier_try_wait_parity(
                            sf_full_mbar_ptr.subview(stage),
                            ab_full_phase_bit,
                            time_limit=10_000_000,
                            scope=_OPERAND_BARRIER_SCOPE,
                        ):
                            pass

                        # Disjoint words let all SF copies precede the K-tile's MMAs.
                        for sf_load_word in cutlass.range_constexpr(num_sf_atoms if _sf_words_resident else 0):
                            for _bj in cutlass.range_constexpr(num_sfb_operands):
                                for block_n in cutlass.range_constexpr(num_blocks_n):
                                    for _a in cutlass.range_constexpr(word_atoms):
                                        if elect_one:
                                            nvvm.tcgen05_cp(
                                                s2t_shape,
                                                sfb_dst_ptrs[sf_load_word][_bj][block_n][_a],
                                                desc_sfb_bases[_bj] + (sf_atom_desc_stride * (sf_load_word * word_atoms + _a) + sf_block_desc_stride * block_n),
                                                group=_CTA_GROUP,
                                                multicast=s2t_multicast,
                                            )
                            for _ai in cutlass.range_constexpr(num_sfa_operands):
                                for mma_m in cutlass.range_constexpr(mma_size_m):
                                    for _a in cutlass.range_constexpr(word_atoms):
                                        if elect_one:
                                            nvvm.tcgen05_cp(
                                                s2t_shape,
                                                sfa_dst_ptrs[sf_load_word][_ai][mma_m][_a],
                                                desc_sfa_bases[_ai] + (sf_atom_desc_stride * (sf_load_word * word_atoms + _a) + sf_block_desc_stride * mma_m),
                                                group=_CTA_GROUP,
                                                multicast=s2t_multicast,
                                            )

                        for sf_word in cutlass.range_constexpr(num_sf_atoms):
                            _sf_slot = sf_word if _sf_words_resident else 0
                            for _bj in cutlass.range_constexpr(0 if _sf_words_resident else num_sfb_operands):
                                for block_n in cutlass.range_constexpr(num_blocks_n):
                                    for _a in cutlass.range_constexpr(word_atoms):
                                        if elect_one:
                                            nvvm.tcgen05_cp(
                                                s2t_shape,
                                                sfb_dst_ptrs[0][_bj][block_n][_a],
                                                desc_sfb_bases[_bj] + (sf_atom_desc_stride * (sf_word * word_atoms + _a) + sf_block_desc_stride * block_n),
                                                group=_CTA_GROUP,
                                                multicast=s2t_multicast,
                                            )
                            if cutlass.const_expr(sf_word == 0):
                                while not nvvm.mbarrier_try_wait_parity(
                                    ab_full_mbar_ptr.subview(stage),
                                    ab_full_phase_bit,
                                    time_limit=10_000_000,
                                    scope=_OPERAND_BARRIER_SCOPE,
                                ):
                                    pass
                            for mma_k_in_word in cutlass.range_constexpr(sf_insts_per_atom):
                                mma_k = sf_word * sf_insts_per_atom + mma_k_in_word
                                idesc_k = idesc_by_j[mma_k_in_word]
                                for gemm_i in cutlass.range_constexpr(num_gemms):
                                    _ai = gemm_a_idx[gemm_i]
                                    _bj = gemm_b_idx[gemm_i]
                                    desc_a_k = desc_a_bases[_ai].advance_start_address(a_smem_k_step_bytes * mma_k)
                                    desc_b = desc_b_bases[_bj].advance_start_address(b_smem_k_step_bytes * mma_k)
                                    for mma_m in cutlass.range_constexpr(mma_size_m):
                                        if cutlass.const_expr(
                                            not _sf_words_resident and not fake_dequant_a and mma_k_in_word == 0 and _ai not in gemm_a_idx[:gemm_i]
                                        ):
                                            for _a in cutlass.range_constexpr(word_atoms):
                                                if elect_one:
                                                    nvvm.tcgen05_cp(
                                                        s2t_shape,
                                                        sfa_dst_ptrs[0][_ai][mma_m][_a],
                                                        desc_sfa_bases[_ai]
                                                        + (sf_atom_desc_stride * (sf_word * word_atoms + _a) + sf_block_desc_stride * mma_m),
                                                        group=_CTA_GROUP,
                                                        multicast=s2t_multicast,
                                                    )
                                        desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mma_m)
                                        if elect_one:
                                            _tcgen05_mma_block_scale(
                                                mma_block_scale_kind,
                                                _CTA_GROUP,
                                                acc_tmem_ptrs[gemm_i][mma_m],
                                                desc_a,
                                                desc_b,
                                                idesc_k,
                                                enable_input_d=scale_d,
                                                scale_a=sfa_dst_ptrs[_sf_slot][_ai][mma_m][0],
                                                scale_b=sfb_scale_ptrs[_sf_slot][_bj],
                                                scale_vec_size=scale_vec_size,
                                                collector_op=_a_collector_op(gemm_i),
                                                b_collector_op=_b_collector_op(mma_m),
                                            )
                                scale_d = cutlass.Boolean(True)

                        if elect_one:
                            nvvm.tcgen05_commit(
                                ab_empty_mbar_ptr.subview(stage),
                                multicast_mask=ab_empty_arrive_mask,
                                group=_CTA_GROUP,
                            )
                        ab_iter += 1

                    if elect_one:
                        nvvm.tcgen05_commit(
                            acc_full_mbar_ptr.subview(acc_stage),
                            group=_CTA_GROUP,
                        )
                    tile_iter += 1

            if cutlass.const_expr(USE_PDL):
                nvvm.griddepcontrol("launch_dependents")

            nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
            if tile_iter != 0:
                tail_stage = acc_stage
                tail_phase = acc_empty_phase_bit
                for _ in range(acc_stages):
                    tail_stage = tail_stage + 1
                    if tail_stage == acc_stages:
                        tail_stage = cutlass.Int32(0)
                        tail_phase = tail_phase ^ 1
                    while not nvvm.mbarrier_try_wait_parity(
                        acc_empty_mbar_ptr.subview(tail_stage),
                        tail_phase,
                        time_limit=10_000_000,
                        scope=_OPERAND_BARRIER_SCOPE,
                    ):
                        pass
                if cutlass.const_expr(use_acc_overlap):
                    while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                        pass

            nvvm.bar_warp_sync(0xFFFFFFFF)
            alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
            _tcgen05_dealloc(
                alloc_ptr,
                cutlass.Int32(num_tmem_alloc_cols),
                is_exclusive=tmem_alloc_exclusive,
                group=_CTA_GROUP,
            )
        else:
            peer_cta_rank = cta_rank_in_cluster ^ 1
            if is_pair_leader:
                ab_full_phase_bit = cutlass.Int32(0)
                ab_iter = cutlass.Int32(0)
                acc_empty_phase_bit = cutlass.Int32(1)
                tile_iter = cutlass.Int32(0)
                is_valid = cutlass.Int32(1)
                sched_stage = cutlass.Int32(0)
                sched_full_phase = cutlass.Int32(0)
                acc_stage = cutlass.Int32(0)
                if cutlass.const_expr(idesc_is_omma):
                    idesc_by_j = [
                        cutlass.experimental.primitives.Tcgen05MxOmmaInstrDesc.build(
                            a_dtype=idesc_a_dtype,
                            b_dtype=idesc_b_dtype,
                            scale_format=sf_scale_format,
                            n_dim=mma_n_dim,
                            m_dim=mma_m_dim,
                            a_major=mma_a_major,
                            b_major=mma_b_major,
                            a_sf_id=j * sf_scales_per_inst,
                            b_sf_id=j * sf_scales_per_inst,
                            k_dim=mma_k_dim_mode,
                        )
                        for j in range(sf_insts_per_atom)
                    ]
                else:
                    idesc_by_j = [
                        cutlass.experimental.primitives.Tcgen05MxInstrDesc.build(
                            a_dtype=idesc_a_dtype,
                            b_dtype=idesc_b_dtype,
                            scale_format=sf_scale_format,
                            n_dim=mma_n_dim,
                            m_dim=mma_m_dim,
                            a_major=mma_a_major,
                            b_major=mma_b_major,
                            a_sf_id=j * sf_scales_per_inst,
                            b_sf_id=j * sf_scales_per_inst,
                            k_dim=mma_k_dim_mode,
                        )
                        for j in range(sf_insts_per_atom)
                    ]
                sfa_tmem_bases = [(base_row_id << 16) | (base_col_id_root + sfa_col_bases[i]) for i in range(num_a_operands)]
                sfb_tmem_bases = [(base_row_id << 16) | (base_col_id_root + sfb_col_bases[j]) for j in range(num_b_operands)]
                s2t_shape, s2t_multicast = nvvm.S2TCopyMode.S2T_32x128b_WARPX4
                sfb_scale_ptrs = [[nvvm.make_tmem_ptr(b + _sfb_word_stride * w, cutlass.Float32) for b in sfb_tmem_bases] for w in range(_sf_word_slots)]
                sfa_dst_ptrs = [
                    [
                        [
                            [
                                nvvm.make_tmem_ptr(sfa_tmem_bases[i] + _sfa_word_stride * w + m * registers_per_block + a * registers_per_atom, cutlass.Float32)
                                for a in range(word_atoms)
                            ]
                            for m in range(mma_size_m)
                        ]
                        for i in range(num_a_operands)
                    ]
                    for w in range(_sf_word_slots)
                ]
                sfb_dst_ptrs = [
                    [
                        [
                            [
                                nvvm.make_tmem_ptr(sfb_tmem_bases[j] + _sfb_word_stride * w + (a * num_blocks_n + m) * registers_per_atom, cutlass.Float32)
                                for a in range(word_atoms)
                            ]
                            for m in range(num_blocks_n)
                        ]
                        for j in range(num_b_operands)
                    ]
                    for w in range(_sf_word_slots)
                ]
                desc_a_roots = [
                    cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                        start_address=smem_a_list[i],
                        leading_byte_offset=a_smem_desc_leading_byte_offset,
                        stride_byte_offset=a_smem_desc_stride_byte_offset,
                        layout=a_smem_swizzle,
                    )
                    for i in range(num_a_operands)
                ]
                desc_b_roots = [
                    cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                        start_address=smem_b_list[j],
                        leading_byte_offset=b_smem_desc_leading_byte_offset,
                        stride_byte_offset=b_smem_desc_stride_byte_offset,
                        layout=b_smem_swizzle,
                    )
                    for j in range(num_b_operands)
                ]
                desc_sfa_roots = [
                    cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                        start_address=smem_sfa_list[i],
                        leading_byte_offset=16,
                        stride_byte_offset=128,
                        layout=cutlass.experimental.primitives.Tcgen05SmemSwizzle.NONE,
                    )
                    for i in range(num_sfa_operands)
                ]
                desc_sfb_roots = [
                    cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                        start_address=smem_sfb_list[j],
                        leading_byte_offset=16,
                        stride_byte_offset=128,
                        layout=cutlass.experimental.primitives.Tcgen05SmemSwizzle.NONE,
                    )
                    for j in range(num_sfb_operands)
                ]
                while is_valid != 0:
                    while not nvvm.mbarrier_try_wait_parity(
                        sched_full_mbar_ptr.subview(sched_stage),
                        sched_full_phase,
                        time_limit=10_000_000,
                    ):
                        pass
                    is_valid = (sched_storage.subview(sched_stage * SCHED_SLOT_WORDS).subview(3)).load()
                    # Finish every lane's slot reads before the elected release.
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
                    sched_stage += 1
                    if sched_stage == SCHED_STAGES:
                        sched_stage = cutlass.Int32(0)
                        sched_full_phase = sched_full_phase ^ 1

                    if is_valid != 0:
                        acc_stage = tile_iter % acc_stages
                        if acc_stage == 0 and tile_iter != 0:
                            acc_empty_phase_bit = acc_empty_phase_bit ^ 1

                        while not nvvm.mbarrier_try_wait_parity(
                            acc_empty_mbar_ptr.subview(acc_stage),
                            acc_empty_phase_bit,
                            time_limit=10_000_000,
                            scope=_OPERAND_BARRIER_SCOPE,
                        ):
                            pass

                        if cutlass.const_expr(use_acc_overlap):
                            acc_base_col = base_col_id_root + (tile_iter % 2) * acc_stage_stride
                        else:
                            acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                        acc_tmem_ptrs = [
                            [
                                nvvm.make_tmem_ptr(
                                    (base_row_id << 16) | (acc_base_col + g * acc_gemm_stride + mi * epi_cols_per_mma_m),
                                    cutlass.Float32,
                                )
                                for mi in range(mma_size_m)
                            ]
                            for g in range(num_gemms)
                        ]

                        scale_d = cutlass.Boolean(False)
                        for k_tile_idx in range(num_k_tiles):
                            stage = ab_iter % ab_stages
                            if stage == 0 and ab_iter != 0:
                                ab_full_phase_bit = ab_full_phase_bit ^ 1

                            desc_a_bases = [desc_a_roots[i].advance_start_address(sA_bytes * stage) for i in range(num_a_operands)]
                            desc_b_bases = [desc_b_roots[j].advance_start_address(sB_bytes * stage) for j in range(num_b_operands)]
                            desc_sfa_bases = [desc_sfa_roots[i].advance_start_address(sfa_smem_bytes * stage) for i in range(num_sfa_operands)]
                            desc_sfb_bases = [desc_sfb_roots[j].advance_start_address(sfb_smem_bytes * stage) for j in range(num_sfb_operands)]

                            while not nvvm.mbarrier_try_wait_parity(
                                sf_full_mbar_ptr.subview(stage),
                                ab_full_phase_bit,
                                time_limit=10_000_000,
                                scope=_OPERAND_BARRIER_SCOPE,
                            ):
                                pass

                            # Disjoint words let all SF copies precede the K-tile's MMAs.
                            for sf_load_word in cutlass.range_constexpr(num_sf_atoms if _sf_words_resident else 0):
                                for _bj in cutlass.range_constexpr(num_sfb_operands):
                                    for block_n in cutlass.range_constexpr(num_blocks_n):
                                        for _a in cutlass.range_constexpr(word_atoms):
                                            if elect_one:
                                                nvvm.tcgen05_cp(
                                                    s2t_shape,
                                                    sfb_dst_ptrs[sf_load_word][_bj][block_n][_a],
                                                    desc_sfb_bases[_bj]
                                                    + (sf_atom_desc_stride * (sf_load_word * word_atoms + _a) + sf_block_desc_stride * block_n),
                                                    group=_CTA_GROUP,
                                                    multicast=s2t_multicast,
                                                )
                                for _ai in cutlass.range_constexpr(num_sfa_operands):
                                    for mma_m in cutlass.range_constexpr(mma_size_m):
                                        for _a in cutlass.range_constexpr(word_atoms):
                                            if elect_one:
                                                nvvm.tcgen05_cp(
                                                    s2t_shape,
                                                    sfa_dst_ptrs[sf_load_word][_ai][mma_m][_a],
                                                    desc_sfa_bases[_ai]
                                                    + (sf_atom_desc_stride * (sf_load_word * word_atoms + _a) + sf_block_desc_stride * mma_m),
                                                    group=_CTA_GROUP,
                                                    multicast=s2t_multicast,
                                                )

                            for sf_word in cutlass.range_constexpr(num_sf_atoms):
                                _sf_slot = sf_word if _sf_words_resident else 0
                                for _bj in cutlass.range_constexpr(0 if _sf_words_resident else num_sfb_operands):
                                    for block_n in cutlass.range_constexpr(num_blocks_n):
                                        for _a in cutlass.range_constexpr(word_atoms):
                                            if elect_one:
                                                nvvm.tcgen05_cp(
                                                    s2t_shape,
                                                    sfb_dst_ptrs[0][_bj][block_n][_a],
                                                    desc_sfb_bases[_bj] + (sf_atom_desc_stride * (sf_word * word_atoms + _a) + sf_block_desc_stride * block_n),
                                                    group=_CTA_GROUP,
                                                    multicast=s2t_multicast,
                                                )
                                if cutlass.const_expr(sf_word == 0):
                                    while not nvvm.mbarrier_try_wait_parity(
                                        ab_full_mbar_ptr.subview(stage),
                                        ab_full_phase_bit,
                                        time_limit=10_000_000,
                                        scope=_OPERAND_BARRIER_SCOPE,
                                    ):
                                        pass
                                for mma_k_in_word in cutlass.range_constexpr(sf_insts_per_atom):
                                    mma_k = sf_word * sf_insts_per_atom + mma_k_in_word
                                    idesc_k = idesc_by_j[mma_k_in_word]
                                    for gemm_i in cutlass.range_constexpr(num_gemms):
                                        _ai = gemm_a_idx[gemm_i]
                                        _bj = gemm_b_idx[gemm_i]
                                        desc_a_k = desc_a_bases[_ai].advance_start_address(a_smem_k_step_bytes * mma_k)
                                        desc_b = desc_b_bases[_bj].advance_start_address(b_smem_k_step_bytes * mma_k)
                                        for mma_m in cutlass.range_constexpr(mma_size_m):
                                            if cutlass.const_expr(
                                                not _sf_words_resident and not fake_dequant_a and mma_k_in_word == 0 and _ai not in gemm_a_idx[:gemm_i]
                                            ):
                                                for _a in cutlass.range_constexpr(word_atoms):
                                                    if elect_one:
                                                        nvvm.tcgen05_cp(
                                                            s2t_shape,
                                                            sfa_dst_ptrs[0][_ai][mma_m][_a],
                                                            desc_sfa_bases[_ai]
                                                            + (sf_atom_desc_stride * (sf_word * word_atoms + _a) + sf_block_desc_stride * mma_m),
                                                            group=_CTA_GROUP,
                                                            multicast=s2t_multicast,
                                                        )
                                            desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mma_m)
                                            if elect_one:
                                                _tcgen05_mma_block_scale(
                                                    mma_block_scale_kind,
                                                    _CTA_GROUP,
                                                    acc_tmem_ptrs[gemm_i][mma_m],
                                                    desc_a,
                                                    desc_b,
                                                    idesc_k,
                                                    enable_input_d=scale_d,
                                                    scale_a=sfa_dst_ptrs[_sf_slot][_ai][mma_m][0],
                                                    scale_b=sfb_scale_ptrs[_sf_slot][_bj],
                                                    scale_vec_size=scale_vec_size,
                                                    collector_op=_a_collector_op(gemm_i),
                                                    b_collector_op=_b_collector_op(mma_m),
                                                )
                                    scale_d = cutlass.Boolean(True)

                            if elect_one:
                                nvvm.tcgen05_commit(
                                    ab_empty_mbar_ptr.subview(stage),
                                    multicast_mask=ab_empty_arrive_mask,
                                    group=_CTA_GROUP,
                                )
                            ab_iter += 1

                        if elect_one:
                            nvvm.tcgen05_commit(
                                acc_full_mbar_ptr.subview(acc_stage),
                                multicast_mask=pair_mask,
                                group=_CTA_GROUP,
                            )
                        tile_iter += 1

                if cutlass.const_expr(USE_PDL):
                    nvvm.griddepcontrol("launch_dependents")

                if tile_iter != 0:
                    tail_stage = acc_stage
                    tail_phase = acc_empty_phase_bit
                    for _ in range(acc_stages):
                        tail_stage = tail_stage + 1
                        if tail_stage == acc_stages:
                            tail_stage = cutlass.Int32(0)
                            tail_phase = tail_phase ^ 1
                        while not nvvm.mbarrier_try_wait_parity(
                            acc_empty_mbar_ptr.subview(tail_stage),
                            tail_phase,
                            time_limit=10_000_000,
                            scope=_OPERAND_BARRIER_SCOPE,
                        ):
                            pass
                nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
                peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
                while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                    pass
                if cutlass.const_expr(not use_acc_overlap):
                    nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
                _tcgen05_dealloc(
                    alloc_ptr,
                    cutlass.Int32(num_tmem_alloc_cols),
                    is_exclusive=tmem_alloc_exclusive,
                    group=_CTA_GROUP,
                )
            else:
                is_valid = cutlass.Int32(1)
                sched_stage = cutlass.Int32(0)
                sched_full_phase = cutlass.Int32(0)
                while is_valid != 0:
                    while not nvvm.mbarrier_try_wait_parity(
                        sched_full_mbar_ptr.subview(sched_stage),
                        sched_full_phase,
                        time_limit=10_000_000,
                    ):
                        pass
                    is_valid = (sched_storage.subview(sched_stage * SCHED_SLOT_WORDS).subview(3)).load()
                    # Finish every lane's slot reads before the elected release.
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
                    sched_stage += 1
                    if sched_stage == SCHED_STAGES:
                        sched_stage = cutlass.Int32(0)
                        sched_full_phase = sched_full_phase ^ 1

                if cutlass.const_expr(USE_PDL):
                    nvvm.griddepcontrol("launch_dependents")

                nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
                peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
                if cutlass.const_expr(not use_acc_overlap):
                    nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                    pass
                alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
                _tcgen05_dealloc(
                    alloc_ptr,
                    cutlass.Int32(num_tmem_alloc_cols),
                    is_exclusive=tmem_alloc_exclusive,
                    group=_CTA_GROUP,
                )

    if warp_idx < num_epilogue_warps:
        if cutlass.const_expr(not moe_combine_swap_bulk_enabled):
            nvvm.setmaxregister(epi_reg_count, nvvm.SetMaxRegisterAction.INCREASE)
        nvvm.barrier_cta_sync(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16

        if cutlass.const_expr(fake_dequant_a):
            for i in cutlass.range_constexpr(num_a_operands):
                _fill_scale_one((base_row_id << 16) | (base_col_id_root + sfa_col_bases[i]), sfa_tmem_cols)
        if cutlass.const_expr(fake_dequant_b):
            for j in cutlass.range_constexpr(num_b_operands):
                _fill_scale_one((base_row_id << 16) | (base_col_id_root + sfb_col_bases[j]), sfb_tmem_cols)
        if cutlass.const_expr(fake_dequant_a or fake_dequant_b):
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            nvvm.barrier_cta_sync(
                barrier_id=TMEM_SCALE_ONE_BARRIER_ID,
                thread_count=tmem_alloc_bar_count,
            )

        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")

        tile_iter = cutlass.Int32(0)
        acc_full_phase_bit = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)

        # AVG uses routed rows in use; clamp empty routing to a finite divisor.
        moe_used_rows = cutlass.max(cutlass.min(cutlass.Int64(first_token_arr[num_groups]), N), cutlass.Int64(1))

        # @@EPILOGUE_SETUP:BEGIN@@
        row_id_with_warp_offset = base_row_id + warp_idx * 32

        epi_spans = _epi_subtile_spans(epi_cols_per_mma_m, epi_n)
        subtile_cnt = len(epi_spans)
        shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
        lane = tidx % 32
        # @@EPILOGUE_SETUP:END@@

        epi_stage_idx = cutlass.Int32(EPI_SMEM_STAGES - 1)
        tile_l = cutlass.Int32(0)
        epi_block_linear = bidx + bidy * gridx
        d_desc_base_list = [
            tma_workspace.iterator.raw_ptr() + (epi_block_linear * moe_desc_slots + num_b_operands + num_sfb_operands + _di) * TENSOR_MAP_QWORDS
            for _di in range(n_tma_outputs)
        ]
        d_desc_ptr_list = [cute.make_ptr(cutlass.Int64, _b.toint(), mem_space=cute.AddressSpace.generic) for _b in d_desc_base_list]
        previous_group_end = cutlass.Int32(-1)
        if warp_idx == 0 and cutlass.const_expr(not moe_aligned_offsets and not moe_scatter):
            for _di in cutlass.range_constexpr(n_tma_outputs):
                if elect_one:
                    _copy_tensormap_to_workspace(tma_c_descs[_di].get_ptr(), tma_c_desc_smem.subview(_di * TENSOR_MAP_QWORDS))
            nvvm.bar_warp_sync(0xFFFFFFFF)

        while is_valid != 0:
            while not nvvm.mbarrier_try_wait_parity(
                sched_full_mbar_ptr.subview(sched_stage),
                sched_full_phase,
                time_limit=10_000_000,
            ):
                pass
            if cutlass.const_expr(cta_group == 1):
                _slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
                tile_m = (_slot.subview(1)).load()
                tile_n = (_slot.subview(2)).load()
                is_valid = (_slot.subview(3)).load()
                group_begin = (_slot.subview(4)).load()
                group_end = (_slot.subview(5)).load()
                start_sf_block_n = (_slot.subview(6)).load()
                group_idx = (_slot.subview(7)).load()
            else:
                slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
                tile_m = (slot.subview(1)).load()
                tile_n = (slot.subview(2)).load()
                is_valid = (slot.subview(3)).load()
                group_begin = (slot.subview(4)).load()
                group_end = (slot.subview(5)).load()
                start_sf_block_n = (slot.subview(6)).load()
                group_idx = (slot.subview(7)).load()
            nvvm.bar_warp_sync(0xFFFFFFFF)
            sched_stage = cute.arch.make_warp_uniform(sched_stage)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                coord_m_tile = tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
                if warp_idx == 0 and cutlass.const_expr(not moe_aligned_offsets and not moe_scatter):
                    if group_end != previous_group_end:
                        previous_group_end = group_end
                        nvvm.cp_async_bulk_wait_group(0, read=True)
                        for _di in cutlass.range_constexpr(n_tma_outputs):
                            _scratch = tma_c_desc_smem.subview(_di * TENSOR_MAP_QWORDS)
                            _fence_tensormap_acquire(d_desc_ptr_list[_di])
                            if elect_one:
                                if cutlass.const_expr(tma_c_m_major[_di]):
                                    _replace_tensormap_global_dim_1(_scratch, group_end)
                                else:
                                    _replace_tensormap_global_dim_0(_scratch, group_end)
                            nvvm.bar_warp_sync(0xFFFFFFFF)
                            if lane < TENSOR_MAP_QWORDS:
                                (d_desc_base_list[_di] + lane).store((_scratch.subview(lane)).load())
                            nvvm.bar_warp_sync(0xFFFFFFFF)
                        _fence_tensormap_release()
                # @@EPILOGUE_DRAIN:BEGIN@@
                coord_n_c = group_begin + tile_n * cgrp_tile_mnk[1] + n_rank * (cta_tile_mnk[1] * cta_group)

                acc_stage = tile_iter % acc_stages
                if acc_stage == 0 and tile_iter != 0:
                    acc_full_phase_bit = acc_full_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(acc_full_mbar_ptr.subview(acc_stage), acc_full_phase_bit, time_limit=10_000_000):
                    pass

                if cutlass.const_expr(use_acc_overlap):
                    acc_buf_parity = tile_iter % 2
                    acc_base_col = base_col_id_root + acc_buf_parity * acc_stage_stride
                else:
                    acc_buf_parity = cutlass.Int32(0)
                    acc_base_col = base_col_id_root + acc_stage * acc_region_cols

                if cutlass.const_expr(moe_combine_swap_bulk_enabled):
                    for combine_subtile in cutlass.range_constexpr(subtile_cnt):
                        combine_col = coord_n_c + combine_subtile * epi_n
                        combine_weights = []
                        for combine_batch in cutlass.range_constexpr(epi_n // 32):
                            combine_route = combine_col + combine_batch * 32 + lane
                            combine_weight = cutlass.Float32(0.0)
                            if combine_route < group_end:
                                combine_token = token_index[combine_route]
                                combine_slot = token_ks[combine_route]
                                if (combine_token >= 0) & (combine_token < top_k_scores.shape[0]) & (combine_slot >= 0) & (combine_slot < moe_top_k):
                                    combine_weight = top_k_scores[combine_token, combine_slot]
                            combine_weights.append(combine_weight)
                        for combine_mi in cutlass.range_constexpr(mma_size_m):
                            for combine_chunk in cutlass.range_constexpr(epi_n // 64):
                                combine_tmem_col = acc_base_col + combine_mi * epi_cols_per_mma_m + combine_subtile * epi_n + combine_chunk * 64
                                combine_tmem_addr = (row_id_with_warp_offset << 16) | combine_tmem_col
                                combine_tmem = cutlass.inttoptr(combine_tmem_addr, 6, mma_c_dtype)
                                combine_values = nvvm.tcgen05_ld(shape, combine_tmem, num=64)
                                if cutlass.const_expr(combine_mi == mma_size_m - 1 and combine_subtile == subtile_cnt - 1 and combine_chunk == epi_n // 64 - 1):
                                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                                    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                    if elect_one:
                                        nvvm.mbarrier_arrive(
                                            nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank),
                                            scope=nvvm.MemScope.CLUSTER,
                                            relaxed=True,
                                        )
                                for combine_j in cutlass.range_constexpr(64):
                                    combine_logical_j = combine_chunk * 64 + combine_j
                                    combine_score = cute.arch.shuffle_sync(combine_weights[combine_logical_j // 32], cutlass.Int32(combine_logical_j % 32))
                                    combine_value = combine_values[combine_j] * combine_score
                                    combine_offset = combine_logical_j * (128 * mma_size_m) + combine_mi * 128 + tidx
                                    combine_staging.data_ptr(combine_offset).store(combine_value.to(cutlass.BFloat16))
                        moe_combine_swap_bulk(
                            combine_staging.data_ptr(0),
                            gC_tap_0_ptr,
                            token_index,
                            token_ks,
                            top_k_scores,
                            coord_m_tile,
                            combine_col,
                            group_end,
                            M,
                            out_stride_m_0,
                            out_stride_n_0,
                            tidx,
                            moe_top_k,
                            epi_n,
                            128 * mma_size_m,
                        )
                else:
                    for mi in cutlass.range_constexpr(mma_size_m):
                        if cutlass.const_expr(use_acc_overlap and mma_size_m > 1):
                            _mi = mi + (1 - acc_buf_parity) * (mma_size_m - 1 - 2 * mi)
                        else:
                            _mi = mi
                        coord_m = coord_m_tile + _mi * epi_rows_per_mma_m
                        mi_col_base = acc_base_col + _mi * epi_cols_per_mma_m
                        tmem_col_addr_gemms = [(row_id_with_warp_offset << 16) | (mi_col_base + g * acc_gemm_stride) for g in range(num_gemms)]

                        row = coord_m + tidx
                        row_active = True

                        _aux_gate_alpha_ptr = gate_alpha.iterator.raw_ptr()
                        _aux_gate_alpha_pre = (_aux_gate_alpha_ptr + group_idx).load()
                        _aux_up_alpha_ptr = up_alpha.iterator.raw_ptr()
                        _aux_up_alpha_pre = (_aux_up_alpha_ptr + group_idx).load()
# @@FROST_ACTIVATION_1@@
                        _aux_scale_ptr = scale.iterator.raw_ptr()
                        _aux_scale_pre = (_aux_scale_ptr + 0).load()

                        for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                            if cutlass.const_expr(use_acc_overlap):
                                _sub = subtile_idx + (1 - acc_buf_parity) * (subtile_cnt - 1 - 2 * subtile_idx)
                                subtile_col_offset = _sub * epi_n
                                subtile_w = epi_n
                            else:
                                subtile_col_offset, subtile_w = epi_spans[subtile_idx]
                            c_rmem_vecs = []
                            for g in cutlass.range_constexpr(num_gemms):
                                subtile_tmem_addr = tmem_col_addr_gemms[g] + subtile_col_offset
                                tmem = cutlass.inttoptr(subtile_tmem_addr, 6, mma_c_dtype)
                                _cv = nvvm.tcgen05_ld(shape, tmem, num=subtile_w)
                                c_rmem_vecs.append(_cv)
                            c_rmem_vec = c_rmem_vecs[0]

                            if cutlass.const_expr(not use_acc_overlap):
                                if cutlass.const_expr(mi == mma_size_m - 1 and subtile_idx == subtile_cnt - 1):
                                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                                    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                    if elect_one:
                                        if cutlass.const_expr(cta_group == 2):
                                            nvvm.mbarrier_arrive(
                                                nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank),
                                                scope=nvvm.MemScope.CLUSTER,
                                                relaxed=True,
                                            )
                                        else:
                                            nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage))

                            if use_acc_overlap and mi * subtile_cnt + subtile_idx == acc_overlap_subtiles - 1:
                                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                if elect_one:
                                    if cutlass.const_expr(cta_group == 2):
                                        nvvm.mbarrier_arrive(
                                            nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank),
                                            scope=nvvm.MemScope.CLUSTER,
                                            relaxed=True,
                                        )
                                    else:
                                        nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage))

                            col = coord_n_c + subtile_col_offset

                            vec_f32 = c_rmem_vec
                            col_j = col

                            vec_f32_1 = c_rmem_vecs[1]
                            _c_0_a = (vec_f32).to(cutlass.Float32)
                            _op_0 = _c_0_a * (cutlass.full_like(_c_0_a, _aux_gate_alpha_pre.to(cutlass.Float32)))
                            _c_1_a = (vec_f32_1).to(cutlass.Float32)
                            _op_1 = _c_1_a * (cutlass.full_like(_c_1_a, _aux_up_alpha_pre.to(cutlass.Float32)))
                            _c_2_a = (_op_0).to(cutlass.Float32)
# @@FROST_ACTIVATION_2@@
                            _c_3_a = (_op_2).to(cutlass.Float32)
# @@FROST_ACTIVATION_3@@


                            # @@FROST_TMA_STORE@@


                # The M-major TMA path loads its accumulator inside the store loop, so its release cannot move up.
                # @@EPILOGUE_DRAIN:END@@
                tile_iter += 1

        if cutlass.const_expr(use_acc_overlap):
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if elect_one:
                nvvm.mbarrier_arrive(tmem_dealloc_mbar_ptr)

    if warp_idx == unused_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(cta_group == 2):
            if cutlass.const_expr(USE_PDL):
                nvvm.griddepcontrol("wait")
            relay_iter = cutlass.Int32(0)
            relay_full_phase = cutlass.Int32(0)
            relay_sched_stage = cutlass.Int32(0)
            relay_sched_phase = cutlass.Int32(0)
            relay_valid = cutlass.Int32(1)
            while relay_valid != 0:
                while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr.subview(relay_sched_stage), relay_sched_phase, time_limit=10_000_000):
                    pass
                relay_valid = sched_storage.subview(relay_sched_stage * SCHED_SLOT_WORDS + 3).load()
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if elect_one:
                    nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(relay_sched_stage))
                relay_sched_stage += 1
                if relay_sched_stage == SCHED_STAGES:
                    relay_sched_stage = cutlass.Int32(0)
                    relay_sched_phase = relay_sched_phase ^ 1
                if relay_valid != 0:
                    for relay_k_tile in range(num_k_tiles):
                        relay_stage = relay_iter % ab_stages
                        if relay_stage == 0 and relay_iter != 0:
                            relay_full_phase = relay_full_phase ^ 1
                        while not nvvm.mbarrier_try_wait_parity(b_local_ready.subview(relay_stage), relay_full_phase, time_limit=10_000_000):
                            pass
                        if elect_one:
                            # The local acquire above completed every A cp.async.
                            # Publish shared A without fencing unrelated global accesses.
                            nvvm.inline_ptx("fence.release.sync_restrict::shared::cta.cluster;")
                            nvvm.mbarrier_arrive(nvvm.mapa(ab_full_mbar_ptr.subview(relay_stage), pair_leader_rank), scope=nvvm.MemScope.CLUSTER, relaxed=True)
                        relay_iter += 1

    # Inactive SF producers must still return their entry registers before the
    # epilogue's register increase; otherwise a fake-dequant side can deadlock.
    if cutlass.const_expr(num_sfb_operands == 0):
        if warp_idx >= 8 and warp_idx < 12:
            nvvm.setmaxregister(_gather_sf_copy_regs, nvvm.SetMaxRegisterAction.DECREASE)
        if warp_idx >= 16 and warp_idx < 20:
            nvvm.setmaxregister(24, nvvm.SetMaxRegisterAction.DECREASE)

    if cutlass.const_expr(num_sfb_operands > 0):
        if warp_idx >= 8 and (warp_idx < 12):
            nvvm.setmaxregister(_gather_sf_copy_regs, nvvm.SetMaxRegisterAction.DECREASE)
            lane = tidx % 32
            if cutlass.const_expr(USE_PDL):
                nvvm.griddepcontrol("wait")
            sf_warp_rank = warp_idx - 8
            sf_rows_per_warp = gather_sf_rows // 4
            sf_ab_iter = cutlass.Int32(0)
            sf_empty_phase = cutlass.Int32(1)
            sf_sched_stage = cutlass.Int32(0)
            sf_sched_phase = cutlass.Int32(0)
            sf_is_valid = cutlass.Int32(1)
            while sf_is_valid != 0:
                while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr.subview(sf_sched_stage), sf_sched_phase, time_limit=10_000_000):
                    pass
                sf_slot = sched_storage.subview(sf_sched_stage * SCHED_SLOT_WORDS)
                sf_tile_n = sf_slot.subview(2).load()
                sf_is_valid = sf_slot.subview(3).load()
                sf_group_begin = sf_slot.subview(4).load()
                sf_group_end = sf_slot.subview(5).load()
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if elect_one:
                    nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sf_sched_stage))
                sf_sched_stage += 1
                if sf_sched_stage == SCHED_STAGES:
                    sf_sched_stage = cutlass.Int32(0)
                    sf_sched_phase = sf_sched_phase ^ 1
                if sf_is_valid != 0:
                    sf_coord_n = sf_tile_n * cgrp_tile_mnk[1] + n_rank * (cta_tile_mnk[1] * cta_group)
                    sf_row_start = sf_warp_rank * sf_rows_per_warp
                    for sf_part in cutlass.range_constexpr(sf_rows_per_warp // 32):
                        sf_row = sf_row_start + lane + sf_part * 32
                        if cutlass.const_expr(gather_sf_rows == gather_token_rows) or sf_row < gather_token_rows:
                            sf_token_indices.subview(sf_row).store(
                                moe_gather_row(token_index, sf_group_begin + sf_coord_n + sf_row, sf_group_end, cutlass.Int32(-1))
                            )
                    nvvm.bar_warp_sync(0xFFFFFFFF)

                    for sf_k_tile in range(num_k_tiles):
                        sf_stage = sf_ab_iter % ab_stages
                        if sf_stage == 0 and sf_ab_iter != 0:
                            sf_empty_phase = sf_empty_phase ^ 1
                        while not nvvm.mbarrier_try_wait_parity(
                            ab_empty_mbar_ptr.subview(sf_stage), sf_empty_phase, time_limit=10_000_000, scope=_OPERAND_BARRIER_SCOPE
                        ):
                            pass
                        for sf_operand in cutlass.range_constexpr(num_sfb_operands):
                            _issue_gather_scales(
                                sf_gather_scratch[sf_operand].subview(sf_stage * gather_token_rows * gather_sf_row_bytes),
                                mSFB_list[sf_operand],
                                sf_token_indices,
                                sf_k_tile * cta_tile_mnk[2] // block_size,
                                gather_sf_row_bytes,
                                sf_row_start,
                                sf_rows_per_warp,
                                gather_token_rows,
                                mB_list[sf_operand].shape[0],
                            )
                        nvvm.cp_async_mbarrier_arrive(sf_local_ready.subview(sf_stage), noinc=True)
                        sf_ab_iter += 1

            nvvm.cp_async_commit_group()
            nvvm.cp_async_wait_group(0)

    if warp_idx >= 12 and warp_idx < 16:
        if cutlass.const_expr(_gather_copy_regs > 96):
            nvvm.setmaxregister(_gather_copy_regs, nvvm.SetMaxRegisterAction.INCREASE)
        else:
            nvvm.setmaxregister(_gather_copy_regs, nvvm.SetMaxRegisterAction.DECREASE)
        lane = tidx % 32
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        b_thread = (warp_idx - 12) * 32 + lane
        b_row_bytes = b_packed_per_row * (b_smem_dtype.width // 8)
        b_chunks_per_row = b_row_bytes // 16
        b_rows_per_round = 128 // b_chunks_per_row
        b_rows_per_thread = (cta_tile_mnk[1] + b_rows_per_round - 1) // b_rows_per_round
        b_chunk = b_thread % b_chunks_per_row
        b_bases = [cute.make_rmem_tensor(cute.make_layout((b_rows_per_thread,)), cutlass.Int64) for _ in range(num_b_operands)]
        b_live = [cute.make_rmem_tensor(cute.make_layout((b_rows_per_thread,)), cutlass.Int32) for _ in range(num_b_operands)]
        b_ab_iter = cutlass.Int32(0)
        b_empty_phase = cutlass.Int32(1)
        b_sched_stage = cutlass.Int32(0)
        b_sched_phase = cutlass.Int32(0)
        b_is_valid = cutlass.Int32(1)
        while b_is_valid != 0:
            while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr.subview(b_sched_stage), b_sched_phase, time_limit=10_000_000):
                pass
            b_slot = sched_storage.subview(b_sched_stage * SCHED_SLOT_WORDS)
            b_tile_n = b_slot.subview(2).load()
            b_is_valid = b_slot.subview(3).load()
            b_group_begin = b_slot.subview(4).load()
            b_group_end = b_slot.subview(5).load()
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(b_sched_stage))
            b_sched_stage += 1
            if b_sched_stage == SCHED_STAGES:
                b_sched_stage = cutlass.Int32(0)
                b_sched_phase = b_sched_phase ^ 1
            if b_is_valid != 0:
                b_coord_n = b_tile_n * cgrp_tile_mnk[1] + n_rank * (cta_tile_mnk[1] * cta_group) + pair_member * cta_tile_mnk[1]
                # Cache every distinct operand's row base and validity. Sources
                # may have different pointers, row pitches and token extents.
                for b_cache_operand in cutlass.range_constexpr(num_b_operands):
                    # Routing indices are signed int32. Capping the extent at
                    # 2**31 lets one unsigned comparison also reject negatives.
                    b_source_rows = cutlass.Uint32(cutlass.min(mB_list[b_cache_operand].shape[0], cutlass.Int64(1 << 31)))
                    for b_gather_part in cutlass.range_constexpr(b_rows_per_thread):
                        b_row = b_thread // b_chunks_per_row + b_gather_part * b_rows_per_round
                        b_token = moe_gather_row(token_index, b_group_begin + b_coord_n + b_row, b_group_end, mB_list[b_cache_operand].shape[0])
                        b_token_live = cutlass.Uint32(b_token) < b_source_rows
                        b_live[b_cache_operand][b_gather_part] = cutlass.Int32(b_token_live)
                        b_safe_token = cutlass.Int32(0)
                        if b_token_live:
                            b_safe_token = b_token
                        b_bases[b_cache_operand][b_gather_part] = mB_list[b_cache_operand].iterator.toint() + cutlass.Int64(b_safe_token) * b_stride_n_list[
                            b_cache_operand
                        ] * (b_dtype.width // 8)
                for b_k_tile in range(num_k_tiles):
                    b_stage = b_ab_iter % ab_stages
                    if b_stage == 0 and b_ab_iter != 0:
                        b_empty_phase = b_empty_phase ^ 1
                    while not nvvm.mbarrier_try_wait_parity(
                        ab_empty_mbar_ptr.subview(b_stage), b_empty_phase, time_limit=10_000_000, scope=_OPERAND_BARRIER_SCOPE
                    ):
                        pass
                    for b_operand in cutlass.range_constexpr(num_b_operands):
                        for b_gather_part in cutlass.range_constexpr(b_rows_per_thread):
                            b_row = b_thread // b_chunks_per_row + b_gather_part * b_rows_per_round
                            b_safe_col = cutlass.Int32(0)
                            b_size = cutlass.Int32(0)
                            b_col = (b_k_tile * b_chunks_per_row + b_chunk) * b_gather_copy_bytes
                            b_source_bytes = mB_list[b_operand].shape[1] * (b_dtype.width // 8)
                            if (b_live[b_operand][b_gather_part] != 0) & (b_col < b_source_bytes):
                                b_safe_col = cutlass.Int32(b_col)
                                b_size = cutlass.min(cutlass.Int32(b_gather_copy_bytes), cutlass.Int32(b_source_bytes - b_col))
                            b_src_addr = b_bases[b_operand][b_gather_part] + b_safe_col
                            b_src = cutlass.inttoptr(b_src_addr, 1, cutlass.Uint8)
                            b_dst = smem_b_list[b_operand].data_ptr(sB_elems * b_stage + b_row * b_packed_per_row + b_chunk * (16 // (b_smem_dtype.width // 8)))
                            b_dst = cutlass.apply_swizzle(b_dst, cutlass.Swizzle(b_gather_swizzle_bits, 4, 3))
                            if (
                                cutlass.const_expr(cta_tile_mnk[1] % b_rows_per_round == 0) or b_row < cta_tile_mnk[1]
                            ) and b_group_begin + b_coord_n + b_row < b_group_end:
                                if cutlass.const_expr(b_gather_copy_bytes == 16):
                                    nvvm.cp_async_shared_global(b_dst, b_src, 16, nvvm.LoadCacheModifier.CG, cp_size=b_size)
                                else:
                                    # Preserve the .b4x16_p64 storage required by
                                    # the mixed-width K32 MMA, including padding.
                                    nvvm.cp_async_shared_global(b_dst, b_src, 8, nvvm.LoadCacheModifier.CA, cp_size=b_size)
                                    b_padding = cutlass.inttoptr(b_dst.toint() + 8, 3, cutlass.Uint8)
                                    nvvm.cp_async_shared_global(b_padding, b_src, 8, nvvm.LoadCacheModifier.CA, cp_size=0)
                    if cutlass.const_expr(cta_group == 2):
                        nvvm.cp_async_mbarrier_arrive(b_local_ready.subview(b_stage), noinc=True)
                    else:
                        nvvm.cp_async_mbarrier_arrive(ab_full_mbar_ptr.subview(b_stage), noinc=True)
                    b_ab_iter += 1

        nvvm.cp_async_commit_group()
        nvvm.cp_async_wait_group(0)

    if cutlass.const_expr(num_sfb_operands > 0):
        if warp_idx >= 16 and (warp_idx < 20):
            nvvm.setmaxregister(24, nvvm.SetMaxRegisterAction.DECREASE)
            if cutlass.const_expr(USE_PDL):
                nvvm.griddepcontrol("wait")
            tf_row_start = (warp_idx - 16) * (gather_sf_rows // 4)
            tf_iter = cutlass.Int32(0)
            tf_full_phase = cutlass.Int32(0)
            tf_sched_stage = cutlass.Int32(0)
            tf_sched_phase = cutlass.Int32(0)
            tf_valid = cutlass.Int32(1)
            while tf_valid != 0:
                while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr.subview(tf_sched_stage), tf_sched_phase, time_limit=10_000_000):
                    pass
                tf_slot = sched_storage.subview(tf_sched_stage * SCHED_SLOT_WORDS)
                tf_valid = tf_slot.subview(3).load()
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if elect_one:
                    nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(tf_sched_stage))
                tf_sched_stage += 1
                if tf_sched_stage == SCHED_STAGES:
                    tf_sched_stage = cutlass.Int32(0)
                    tf_sched_phase = tf_sched_phase ^ 1
                if tf_valid != 0:
                    for tf_k_tile in range(num_k_tiles):
                        tf_stage = tf_iter % ab_stages
                        if tf_stage == 0 and tf_iter != 0:
                            tf_full_phase = tf_full_phase ^ 1
                        while not nvvm.mbarrier_try_wait_parity(sf_local_ready.subview(tf_stage), tf_full_phase, time_limit=10_000_000):
                            pass
                        for tf_operand in cutlass.range_constexpr(num_sfb_operands):
                            _pack_gather_scales(
                                smem_sfb_list[tf_operand].subview(sfb_smem_bytes * tf_stage),
                                sf_gather_scratch[tf_operand].subview(tf_stage * gather_token_rows * gather_sf_row_bytes),
                                sf_full_mbar_ptr.subview(tf_stage),
                                tf_k_tile * cta_tile_mnk[2] // block_size,
                                cta_tile_mnk[2] // block_size,
                                gather_sf_row_bytes,
                                tf_row_start,
                                gather_sf_rows // 4,
                                gather_token_rows,
                            )
                        tf_iter += 1


frost_template_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    problem_size: tuple,
    first_token_offset: cute.Tensor,
    tma_workspace: cute.Tensor,
    token_index: cute.Tensor,
    a_0: cute.Tensor,
    a_1: cute.Tensor,
    b_0: cute.Tensor,
    sfa_0: cute.Tensor,
    sfa_1: cute.Tensor,
    sfb_0: cute.Tensor,
    gate_alpha: cute.Tensor,
    up_alpha: cute.Tensor,
# @@FROST_ACTIVATION_4@@
    scale: cute.Tensor,
    c_0: cute.Tensor,
    stream: _cuda.CUstream,
) -> None:
    """Bind expert weights and routed token maps for the swapped GEMM launch."""
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

    _tma_c_outputs = [c_0]
    _c0 = _tma_c_outputs[0]
    tma_c_desc_0 = _tma.create_tensor_map_tiled(
        global_address=_c0.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[m, n],
        global_strides=[
            out_stride_n_0 * 16 // 128,
        ],
        box_dims=[64, FROST_EPI_N],
        swizzle=_tma.TensorMapSwizzle.s128b,
    )
    tma_c_desc_list = [tma_c_desc_0]

    tma_a_desc_list = []
    for _a_idx, _a_op in enumerate(_a_operands):
        a_stride_m, a_stride_k, a_stride_l = _a_stride_sets[_a_idx]
        if cutlass.const_expr(a_is_m_major):
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=a_tma_desc_dtype,
                    global_dims=[m, k_sym, num_experts],
                    global_strides=[a_stride_k * a_dtype.width // 128, a_stride_l * a_dtype.width // 128],
                    box_dims=[a_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=a_tma_swizzle,
                    tma_format=a_tma_format,
                )
            )
        else:
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=a_tma_desc_dtype,
                    global_dims=[k_sym, m, num_experts],
                    global_strides=[a_stride_m * a_dtype.width // 128, a_stride_l * a_dtype.width // 128],
                    box_dims=[cta_tile_mnk[2], a_tma_box_m, 1],
                    swizzle=a_tma_swizzle,
                    tma_format=a_tma_format,
                )
            )
    tma_b_desc_list = []
    for _b_idx, _b_op in enumerate(_b_operands):
        b_stride_n, b_stride_k, b_stride_l = _b_stride_sets[_b_idx]
        token_dims = [k_sym, _b_op.shape[0]]
        token_strides = [b_stride_n * b_dtype.width // 128]
        token_box = [cta_tile_mnk[2], 1]
        tma_b_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_b_op.iterator.toint(),
                dtype=_gather_tma_dtype,
                global_dims=token_dims,
                global_strides=token_strides,
                box_dims=token_box,
                swizzle=_gather_tma_swizzle,
                tma_format=_gather_tma_format,
            )
        )
    rest_k = ((k_sym // block_size) + 3) // 4
    rest_m = (m + 127) // 128
    if cutlass.const_expr(moe_combine_swap_bulk_enabled):
        # Sum ceil(group_rows / 128), bounded using at most min(n, G) nonempty groups.
        nonempty = cutlass.min(n, num_groups)
        rest_n = nonempty + (n - nonempty) // 128
    else:
        rest_n = (n + 127) // 128 + num_groups
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
    tma_sfb_desc_list = []
    for _sfb_op in _sfb_operands:
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

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    grid_shape = (grid_num_clusters * cluster_m, cluster_n, 1)
    counter_qword = grid_num_clusters * cluster_m * cluster_n * moe_desc_slots * TENSOR_MAP_QWORDS
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
        b_0,
        _b_stride_sets[0][0],
        token_index,
        _sfb_operands[0],
        out_stride_m_0,
        out_stride_n_0,
        out_stride_l_0,
        gate_alpha,
        up_alpha,
# @@FROST_ACTIVATION_5@@
        scale,
        tma_c_desc_list[0],
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        min_blocks_per_mp=1,
        cluster=cluster_shape_mnk,
        use_pdl=USE_PDL,
        stream=stream,
    )


@lru_cache(maxsize=None)
def compile() -> Callable:
    out_vec_elems = vec_bytes_epi // (cd_dtype.width // 8)
    a_stride_elems = 16 // (a_dtype.width // 8)
    b_stride_elems = 16 // (b_dtype.width // 8)
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=min(out_vec_elems, moe_token_alignment))
    sym_k = cute.sym_int64()
    # Packed K extent: same reasoning as sym_k -- no CTA-tile multiple is required.
    sym_akp = cute.sym_int64()
    sym_bkp = cute.sym_int64()
    sym_e = cute.sym_int64()
    sym_g = cute.sym_int64()
    sym_source_n = cute.sym_int64()

    def _make_fake_a():
        return make_fake_compact_tensor(
            a_fake_dtype,
            (sym_m, sym_akp, sym_e),
            stride_order=(0, 1, 2) if a_is_m_major else (1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            b_fake_dtype,
            (sym_source_n, sym_bkp, 1),
            stride_order=(0, 1, 2) if b_is_n_major else (1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_sfa():
        return cute.runtime.make_fake_tensor(
            sf_cutlass_dtype,
            (cute.sym_int64(), cute.sym_int64(), sym_e),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            assumed_align=16,
        )

    def _make_fake_sfb():
        return cute.runtime.make_fake_tensor(
            sf_cutlass_dtype,
            (cute.sym_int64(), cute.sym_int64(), 1),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            assumed_align=16,
        )

    fake_first_token_offset = make_fake_compact_tensor(
        offset_cutlass_dtype,
        (cute.sym_int64(),),
        stride_order=(0,),
        assumed_align=offset_cutlass_dtype.width // 8,
    )
    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    grid_ctas = grid_num_clusters * cluster_m * cluster_n
    fake_tma_workspace = make_fake_compact_tensor(
        cutlass.Int64,
        (grid_ctas * moe_desc_slots * 16 + 16,),
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
        sym_b_strides.extend(_sym_operand_strides(b_is_n_major, b_stride_elems))

    sym_out_stride_m_0 = cute.sym_int64()
    sym_out_stride_n_0 = cute.sym_int64()
    sym_out_stride_l_0 = cute.sym_int64()

    fake_a_0 = _make_fake_a()
    fake_a_1 = _make_fake_a()
    fake_b_0 = _make_fake_b()
    fake_sfa_0 = _make_fake_sfa()
    fake_sfa_1 = _make_fake_sfa()
    fake_sfb_0 = _make_fake_sfb()
    fake_token_index = make_fake_compact_tensor(cutlass.Int32, (sym_n,), stride_order=(0,), assumed_align=4)


    def _make_fake_c(_dt, _div, _mm):
        return make_fake_compact_tensor(
            _dt,
            (sym_m, sym_n // _div, 1),
            stride_order=(0, 1, 2) if _mm else (1, 0, 2),
            assumed_align=16,
        )

    fake_c_0 = _make_fake_c(cutlass.BFloat16, 1, True)

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
# @@FROST_ACTIVATION_6@@
    fake_scale = cute.runtime.make_fake_tensor(cutlass.Float32, (1, 1, 1), stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()), assumed_align=4)

    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
# @@FROST_ACTIVATION_7@@
