# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM100-family non-block-scale MoE GATHER with independent token copies.

Four copy warps gather K-major token rows with cp.async while a dedicated TMA
warp loads expert weights. The compiler specializes the same pipeline for
independent 1CTA and paired 2CTA MMA. Each CTA gathers its own token slice; the
weight operand retains cluster multicast. Paired MMA uses a local completion
barrier and a relay warp to publish token readiness to the pair leader.
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
def moe_group_at(visit_idx, num_groups, num_experts):
    """Visitation index -> routed group index."""
    per_expert = num_groups // cutlass.max(num_experts, cutlass.Int32(1))
    group = visit_idx
    if per_expert > 1 and per_expert * num_experts == num_groups:
        group = (visit_idx % per_expert) * num_experts + (visit_idx // per_expert)
    return group


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


def tcgen05_mma(mma_kind, cta_group, d, a, b, idesc, scale_d, *, collector_op=None, b_collector_op=None):
    if b_collector_op is None:
        nvvm.tcgen05_mma(
            mma_kind,
            cta_group,
            d,
            a,
            b,
            idesc,
            scale_d,
            collector_op=collector_op,
        )
    else:
        nvvm.tcgen05_mma(
            mma_kind,
            cta_group,
            d,
            a,
            b,
            idesc,
            scale_d,
            collector_op=collector_op,
            b_collector_op=b_collector_op,
        )



_copy_tensormap_to_workspace = copy_tensormap_to_workspace
_epi_subtile_spans = epi_subtile_spans
_fence_tensormap_acquire = fence_tensormap_acquire
_fence_tensormap_release = fence_tensormap_release
_moe_group_at = moe_group_at
_moe_swizzle_tile = moe_swizzle_tile
_replace_tensormap_global_address = replace_tensormap_global_address
_replace_tensormap_global_dim_0 = replace_tensormap_global_dim_0
_replace_tensormap_global_dim_1 = replace_tensormap_global_dim_1
_tcgen05_alloc = tcgen05_alloc
_tcgen05_dealloc = tcgen05_dealloc
_tcgen05_mma = tcgen05_mma
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda


# A TMA tensormap is 128 bytes = 16 int64 qwords.
@cute.jit
def _load_bcast_ack(slot):
    return nvvm.inline_ptx(
        "ld.acquire.cluster.shared::cta.u32 {$r0}, [{$r1}];", write_only_types=[cutlass.Uint32], read_only_args=[cutlass.Uint32(slot.data_ptr().toint())]
    )


@cute.jit
def _publish_bcast_ack(slot):
    nvvm.inline_ptx("red.release.cluster.shared::cluster.add.u32 [{$r0}], 1;", read_only_args=[cutlass.Uint32(slot.data_ptr().toint())])


# @@FROST_GEOMETRY@@

moe_desc_slots = num_a_operands + n_tma_outputs
_CTA_GROUP = nvvm.CTAGroup.CTA_2 if cta_group == 2 else nvvm.CTAGroup.CTA_1
_OPERAND_BARRIER_SCOPE = nvvm.MBarrierScope.CLUSTER if cta_group == 2 else None
_gather_chunks_per_row = cta_tile_mnk[2] * (ab_dtype.width // 8) // 16
_gather_rows_per_round = 128 // _gather_chunks_per_row
_gather_rows_per_thread = (gather_token_rows + _gather_rows_per_round - 1) // _gather_rows_per_round
_gather_cache_bases = _gather_rows_per_thread * num_token_operands <= 16
_gather_copy_regs = 96 if _gather_rows_per_thread <= 16 else 128


# Scheduler ring depth.
SCHED_STAGES = 2
SCHED_BCAST_STAGES = 2

# Number of i32 slots per scheduler ring stage.
SCHED_SLOT_WORDS = 8

# Programmatic Dependent Launch (PDL, sm_90+).
USE_PDL = True

# Double-buffer for the TMA-store epilogue path.
EPI_SMEM_STAGES = 2

# Named barrier id for the 4-warp epilogue handoff around the TMA store.
EPI_SYNC_BAR_ID = 1

# Named barrier id for the TMEM-alloc handoff.
TMEM_ALLOC_BARRIER_ID = 2


@cute.jit
def _moe_auto_swizzle_w(group_rows, n, k, nt_n):
    """N-super-block width for one routed group, resolved per group."""
    if cutlass.const_expr(tile_swizzle_n > 0):
        return tile_swizzle_n
    budget = cutlass.Int64(swizzle_l2_budget_bytes)
    row_bytes = (cutlass.Int64(ab_dtype.width) * k) // 8
    cap = cutlass.max(budget // (row_bytes * cgrp_tile_mnk[1]), cutlass.Int64(1))
    w = cutlass.min(cutlass.Int64(nt_n), cap)
    rows = cutlass.Int64(group_rows)
    if cutlass.min(rows, n) * row_bytes <= budget and rows <= n:
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
    tma_b_desc_1: cutlass.GridConstant[_tma.TensorMap],
    mA_0: cute.Tensor,
    a_stride_m_0: cutlass.Int64,
    token_index: cute.Tensor,
    mC_tap_0: cute.Tensor,
    out_stride_m_0: cutlass.Int64,
    out_stride_n_0: cutlass.Int64,
    out_stride_l_0: cutlass.Int64,
# @@FROST_ACTIVATION_0@@
    scale: cute.Tensor,
) -> None:
    tma_a_descs = [tma_a_desc_0]
    tma_b_descs = [tma_b_desc_0, tma_b_desc_1]
    mA_list = [mA_0]
    a_stride_m_list = [a_stride_m_0]

    mma_warp_id = 4
    tma_warp_id = 5
    scheduler_warp_id = 6
    unused_warp_id = 7
    num_epilogue_warps = 4
    epi_reg_count = 232
    prod_reg_count = 24

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

    sched_counter_ptr = cute.make_ptr(
        cutlass.Int32,
        (a_tma_workspace.iterator.raw_ptr() + grid_num_clusters * cluster_m * cluster_n * moe_desc_slots * TENSOR_MAP_QWORDS).toint(),
        mem_space=cute.AddressSpace.generic,
    )

    if warp_idx == mma_warp_id:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())


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
        ab_empty_arrive_mask = b_part_arrive
    else:
        if cutlass.const_expr(multicast_b):
            tma_mcast_mask_b = cutlass.Int16((b_pattern << pair_member) << (n_rank * cluster_m))
        else:
            tma_mcast_mask_b = cutlass.Int16(1 << cta_rank_in_cluster)

    _smem_sys_reserved = cutlass.Array(cutlass.Int8, 1024, space=cutlass.AddressSpace.smem, alignment=1)

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    if cutlass.const_expr(cta_group == 2):
        token_local_ready = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem, alignment=8)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    if cutlass.const_expr(cta_group == 2):
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
    if cutlass.const_expr(cluster_size > 1):
        bcast_ack_count = cutlass.Array(cutlass.Uint32, SCHED_BCAST_STAGES, space=cutlass.AddressSpace.smem, alignment=8)

    sA_elems = cta_tile_mnk[0] * cta_tile_mnk[2]
    sB_elems = cta_tile_mnk[1] * cta_tile_mnk[2]
    smem_a_list = [
        cutlass.Array(
            ab_dtype,
            sA_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_b_list = [
        cutlass.Array(
            ab_dtype,
            sB_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_b_operands)
    ]

    # Retain token descriptor storage to preserve the compiler-modeled SMEM layout.
    tma_a_desc_smem_list = [
        cutlass.Array(
            cutlass.Int64,
            TENSOR_MAP_QWORDS,
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        for _ in range(num_a_operands)
    ]


    if cutlass.const_expr(cta_group == 2):
        ab_empty_count = cluster_m // cta_group
        acc_empty_count = num_epilogue_warps * 2
    else:
        ab_empty_count = cluster_m
        acc_empty_count = num_epilogue_warps
    sched_empty_count = 1 + 1 + num_epilogue_warps + 4 + (1 if cta_group == 2 else 0)
    if warp_idx == 0:
        if cutlass.const_expr(cta_group == 2):
            if elect_one:
                nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, 32)
        for i in range(ab_stages):
            if elect_one:
                nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1 + (2 if cta_group == 2 else 128))
                if cutlass.const_expr(cta_group == 2):
                    nvvm.mbarrier_init(token_local_ready.subview(i), 128)
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
                if cutlass.const_expr(cluster_size > 1):
                    bcast_ack_count.subview(i).store(cutlass.Uint32(0))
    nvvm.fence_mbarrier_init()
    if cutlass.const_expr(cta_group == 1):
        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            nvvm.barrier_cluster_arrive_relaxed()
            nvvm.barrier_cluster_wait()
        else:
            nvvm.barrier_cta_sync(0)
    else:
        nvvm.barrier_cluster_arrive_relaxed()

    sA_bytes = sA_elems * (ab_dtype.width // 8)
    sB_bytes = sB_elems * (ab_dtype.width // 8)
    num_tma_copy_bytes = num_b_operands * sB_bytes * cta_group

    idesc = cutlass.experimental.primitives.Tcgen05InstrDesc.build(
        a_dtype=mma_a_dtype,
        b_dtype=mma_b_dtype,
        c_dtype=mma_c_dtype,
        n_dim=mma_inst_shape_mnk[1],
        m_dim=mma_inst_shape_mnk[0],
        k_dim=mma_k_dim,
        a_major=mma_a_major,
        b_major=mma_b_major,
    )

    if cutlass.const_expr(cta_group == 1):
        epi_cols_per_mma_m = cta_tile_mnk[1]
    else:
        pair_n_size = cgrp_tile_mnk[1] // cluster_n
        # Per-CTA output rows one MMA-M block covers (the pair splits M).
        epi_rows_per_mma_m = cta_tile_mnk[0] // mma_size_m
        if cutlass.const_expr(epi_rows_per_mma_m == 64):
            # cluster-MMA m=128: the pair also splits N, so each CTA drains N/2.
            epi_cols_per_mma_m = pair_n_size // 2
        else:
            epi_cols_per_mma_m = pair_n_size
        # N is NOT a sub-block axis (the CTA tile is never split along N).
    cols_per_acc_stage = mma_size_m * epi_cols_per_mma_m
    acc_region_cols = num_gemms * cols_per_acc_stage
    if cutlass.const_expr(cta_group == 1):
        epi_rows_per_mma_m = cta_tile_mnk[0] // mma_size_m
    tmem_alloc_bar_count = (num_epilogue_warps + 1) * 32
    if cutlass.const_expr(cta_group == 2):

        nvvm.barrier_cluster_wait()
        nvvm.barrier_cta_sync(0)

    gC_tap_0_ptr = mC_tap_0.iterator.raw_ptr()
    VEC_BYTES_TAP_0 = vec_bytes_tap_0

    vsize = epi_chunk_elems

    M = m
    N = n
    clusters_along_n = cute.ceil_div(cutlass.Int32(N), cgrp_tile_mnk[1])
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    num_k_blocks = cta_tile_mnk[2] // mma_inst_shape_mnk[2]
    first_token_arr = cutlass.make_array_view(first_token_offset)

    if warp_idx == scheduler_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        full_warp_mask = 0xFFFFFFFF
        shfl_idx_clamp = 0x1F
        shfl_up_clamp = 0
        lane = cute.arch.lane_idx()
        sched_stage = cutlass.Int32(0)
        sched_empty_phase = cutlass.Int32(1)
        bcast_stage = cutlass.Int32(0)
        bcast_full_phase = cutlass.Int32(0)
        last_bcast_stage = cutlass.Int32(0)
        if cutlass.const_expr(cluster_size > 1):
            last_bcast_ack_expected = cutlass.Uint32(0)
            bcast_round = cutlass.Uint32(0)
        linear_idx = cutlass.Int32(0)
        start_linear_idx = cutlass.Int32(0)
        total_tiles = cutlass.Int32(0)
        scan_base = cutlass.Int32(0)
        group_idx = cutlass.Int32(0)
        group_begin = cutlass.Int32(0)
        group_end = cutlass.Int32(0)
        is_tile_valid = cutlass.Int32(1)

        while is_tile_valid != 0:
            # Every multicast cluster must agree on the same work item, even
            # when its MMAs are independent 1-CTA instructions.
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

            if linear_idx >= start_linear_idx + total_tiles:
                is_search_live = cutlass.Int32(1)
                while is_search_live != 0:
                    visit_idx = scan_base + lane
                    my_group = _moe_group_at(visit_idx, num_groups, num_experts)
                    my_begin = cutlass.Int32(0)
                    my_end = cutlass.Int32(0)
                    my_tiles = cutlass.Int32(0)
                    if visit_idx < num_groups:
                        if my_group != 0:
                            my_begin = cutlass.Int32(first_token_arr[my_group])
                        my_end = cutlass.Int32(first_token_arr[my_group + 1])
                        my_tiles = cute.ceil_div(my_end - my_begin, cgrp_tile_mnk[0]) * clusters_along_n
                    prefix_tiles = my_tiles
                    for delta in (1, 2, 4, 8, 16):
                        prefix_delta = nvvm.shfl_sync(
                            full_warp_mask,
                            prefix_tiles,
                            delta,
                            shfl_up_clamp,
                            nvvm.Shfl.UP,
                        )
                        if lane >= delta:
                            prefix_tiles += prefix_delta
                    my_start = start_linear_idx + prefix_tiles - my_tiles
                    thread_succeed = nvvm.vote_sync(
                        full_warp_mask,
                        linear_idx < my_start + my_tiles,
                        nvvm.VoteSync.BALLOT,
                    )
                    if thread_succeed != 0:
                        winning_lane = cutlass.Int32(31) - cute.arch.bfind(cute.arch.brev(thread_succeed)).to(cutlass.Int32)
                        scan_base = nvvm.shfl_sync(full_warp_mask, visit_idx, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        group_idx = nvvm.shfl_sync(full_warp_mask, my_group, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        group_begin = nvvm.shfl_sync(full_warp_mask, my_begin, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        group_end = nvvm.shfl_sync(full_warp_mask, my_end, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        start_linear_idx = nvvm.shfl_sync(full_warp_mask, my_start, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        total_tiles = nvvm.shfl_sync(full_warp_mask, my_tiles, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        is_search_live = cutlass.Int32(0)
                    else:
                        start_linear_idx = nvvm.shfl_sync(
                            full_warp_mask,
                            my_start + my_tiles,
                            31,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        scan_base += 32
                        if scan_base >= num_groups:
                            is_tile_valid = cutlass.Int32(0)
                            is_search_live = cutlass.Int32(0)

            coord_expert = cutlass.Int32(0)
            cluster_tile_m = cutlass.Int32(0)
            coord_n = cutlass.Int32(0)
            if is_tile_valid != 0:
                local_linear_idx = linear_idx - start_linear_idx
                group_nt_m = total_tiles // clusters_along_n
                cluster_tile_m, coord_n = _moe_swizzle_tile(
                    local_linear_idx,
                    group_nt_m,
                    clusters_along_n,
                    _moe_auto_swizzle_w(group_nt_m * cgrp_tile_mnk[0], N, k, clusters_along_n),
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
            # Finish every lane's slot reads before the elected release.
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                coord_m_group = tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
                if cutlass.const_expr(cta_group == 1):
                    coord_n_per_cta = tile_n * cgrp_tile_mnk[1] + n_rank * cta_tile_mnk[1]
                else:
                    coord_n_per_cta = tile_n * cgrp_tile_mnk[1] + n_rank * logical_cta_tile_n + pair_member * cta_tile_mnk[1]

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
                    if cutlass.const_expr(cta_group == 1):
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)

                    if cutlass.const_expr(cta_group == 2):
                        if is_pair_leader:
                            if elect_one:
                                nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)
                    b_issue = (not multicast_b) or (pair_m_idx == 0)
                    if cutlass.const_expr(b_mcast_slices > 1):
                        b_data_issue = True
                        _b_off = pair_m_idx * (cta_tile_mnk[1] // b_mcast_slices)
                    else:
                        b_data_issue = b_issue
                        _b_off = 0
                    if b_data_issue:
                        for _bj in cutlass.range_constexpr(num_b_operands):
                            sB_stage = smem_b_list[_bj].subview(sB_elems * stage)
                            if cutlass.const_expr(b_is_n_major):
                                for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sB_stage.subview(n_group * b_tma_group_elems * cta_tile_mnk[2]),
                                            tma_b_descs[_bj].get_ptr(),
                                            (
                                                coord_n_per_cta + n_group * b_tma_group_elems,
                                                coord_k,
                                                coord_expert,
                                            ),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_b,
                                            group=_CTA_GROUP,
                                        )
                            else:
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage.subview(_b_off * cta_tile_mnk[2]),
                                        tma_b_descs[_bj].get_ptr(),
                                        (coord_k, coord_n_per_cta + _b_off, coord_expert),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
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
        ab_empty_arrive_mask = cutlass.Int16(((1 << cluster_m) - 1) << (n_rank * cluster_m))
    if warp_idx == mma_warp_id:
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
        if cutlass.const_expr(cta_group == 1):
            ab_full_phase_bit = cutlass.Int32(0)
            ab_iter = cutlass.Int32(0)
            acc_empty_phase_bit = cutlass.Int32(1)
            tile_iter = cutlass.Int32(0)
            is_valid = cutlass.Int32(1)
            sched_stage = cutlass.Int32(0)
            sched_full_phase = cutlass.Int32(0)
            acc_stage = cutlass.Int32(0)
            desc_a_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_a_list[i],
                    leading_byte_offset=a_smem_desc_leading_byte_offset,
                    stride_byte_offset=a_smem_desc_stride_byte_offset,
                    layout=ab_smem_swizzle,
                )
                for i in range(num_a_operands)
            ]
            desc_b_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_b_list[j],
                    leading_byte_offset=b_smem_desc_leading_byte_offset,
                    stride_byte_offset=b_smem_desc_stride_byte_offset,
                    layout=ab_smem_swizzle,
                )
                for j in range(num_b_operands)
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
                    ):
                        pass

                    acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                    tmem_addr_mmas = [
                        [
                            cutlass.inttoptr(
                                (base_row_id << 16) | (acc_base_col + g * cols_per_acc_stage + mi * epi_cols_per_mma_m),
                                6,
                                cutlass.Int32,
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

                        while not nvvm.mbarrier_try_wait_parity(
                            ab_full_mbar_ptr.subview(stage),
                            ab_full_phase_bit,
                            time_limit=10_000_000,
                            scope=_OPERAND_BARRIER_SCOPE,
                        ):
                            pass

                        for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                            for g in cutlass.range_constexpr(num_gemms):
                                desc_a_k = desc_a_roots[gemm_a_idx[g]].advance_start_address(sA_bytes * stage + a_smem_k_step_bytes * k_block_idx)
                                desc_b_k = desc_b_roots[gemm_b_idx[g]].advance_start_address(sB_bytes * stage + b_smem_k_step_bytes * k_block_idx)
                                for mi in cutlass.range_constexpr(mma_size_m):
                                    desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mi)
                                    if elect_one:
                                        _tcgen05_mma(
                                            mma_kind,
                                            _CTA_GROUP,
                                            tmem_addr_mmas[g][mi],
                                            desc_a,
                                            desc_b_k,
                                            idesc,
                                            scale_d,
                                            collector_op=_a_collector_op(g),
                                            b_collector_op=_b_collector_op(mi),
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
                    ):
                        pass

            nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
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
                desc_a_roots = [
                    cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                        start_address=smem_a_list[i],
                        leading_byte_offset=a_smem_desc_leading_byte_offset,
                        stride_byte_offset=a_smem_desc_stride_byte_offset,
                        layout=ab_smem_swizzle,
                    )
                    for i in range(num_a_operands)
                ]
                desc_b_roots = [
                    cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                        start_address=smem_b_list[j],
                        leading_byte_offset=b_smem_desc_leading_byte_offset,
                        stride_byte_offset=b_smem_desc_stride_byte_offset,
                        layout=ab_smem_swizzle,
                    )
                    for j in range(num_b_operands)
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
                        ):
                            pass

                        acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                        tmem_addr_mmas = [
                            [
                                cutlass.inttoptr(
                                    (base_row_id << 16) | (acc_base_col + g * cols_per_acc_stage + mi * epi_cols_per_mma_m),
                                    6,
                                    cutlass.Int32,
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

                            while not nvvm.mbarrier_try_wait_parity(
                                ab_full_mbar_ptr.subview(stage),
                                ab_full_phase_bit,
                                time_limit=10_000_000,
                                scope=_OPERAND_BARRIER_SCOPE,
                            ):
                                pass

                            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                                for g in cutlass.range_constexpr(num_gemms):
                                    desc_a_k = desc_a_roots[gemm_a_idx[g]].advance_start_address(sA_bytes * stage + a_smem_k_step_bytes * k_block_idx)
                                    desc_b = desc_b_roots[gemm_b_idx[g]].advance_start_address(sB_bytes * stage + b_smem_k_step_bytes * k_block_idx)
                                    for mi in cutlass.range_constexpr(mma_size_m):
                                        desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mi)
                                        if elect_one:
                                            _tcgen05_mma(
                                                mma_kind,
                                                _CTA_GROUP,
                                                tmem_addr_mmas[g][mi],
                                                desc_a,
                                                desc_b,
                                                idesc,
                                                scale_d,
                                                collector_op=_a_collector_op(g),
                                                b_collector_op=_b_collector_op(mi),
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
                        ):
                            pass
                nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
                peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
                while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                    pass
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
        nvvm.setmaxregister(epi_reg_count, nvvm.SetMaxRegisterAction.INCREASE)
        nvvm.barrier_cta_sync(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        tile_iter = cutlass.Int32(0)
        acc_full_phase_bit = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)

        # AVG uses routed rows in use; clamp empty routing to a finite divisor.
        moe_used_rows = cutlass.max(cutlass.min(cutlass.Int64(first_token_arr[num_groups]), M), cutlass.Int64(1))

        # @@EPILOGUE_SETUP:BEGIN@@
        if cutlass.const_expr(epi_packed_lanes):
            row_id_with_warp_offset = base_row_id
        else:
            row_id_with_warp_offset = base_row_id + warp_idx * 32

        epi_spans = _epi_subtile_spans(epi_cols_per_mma_m, epi_n)
        subtile_cnt = len(epi_spans)
        if cutlass.const_expr(epi_packed_lanes):
            shape = nvvm.Tcgen05LdStShape.SHAPE_16X32BX2
            ld_half_off = 0
        else:
            shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
            ld_half_off = None
        lane = tidx % 32
        # @@EPILOGUE_SETUP:END@@


        while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr.subview(sched_stage), sched_full_phase, time_limit=10_000_000):
            pass
        _slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
        tile_m = (_slot.subview(1)).load()
        tile_n = (_slot.subview(2)).load()
        is_valid = (_slot.subview(3)).load()
        group_begin = (_slot.subview(4)).load()
        group_end = (_slot.subview(5)).load()
        group_idx = (_slot.subview(7)).load()
        nvvm.bar_warp_sync(0xFFFFFFFF)
        sched_stage = cute.arch.make_warp_uniform(sched_stage)
        if elect_one:
            nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
        sched_stage += 1
        if sched_stage == SCHED_STAGES:
            sched_stage = cutlass.Int32(0)
            sched_full_phase = sched_full_phase ^ 1

        while is_valid != 0:
            coord_m_tile = group_begin + tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
            # @@EPILOGUE_DRAIN:BEGIN@@
            coord_n_c = tile_n * cgrp_tile_mnk[1] + n_rank * (cta_tile_mnk[1] * cta_group)
            if cutlass.const_expr(epi_dp22):
                coord_n_c = coord_n_c + (warp_idx // 2) * epi_cols_per_mma_m

            acc_stage = tile_iter % acc_stages
            if acc_stage == 0 and tile_iter != 0:
                acc_full_phase_bit = acc_full_phase_bit ^ 1

            while not nvvm.mbarrier_try_wait_parity(acc_full_mbar_ptr.subview(acc_stage), acc_full_phase_bit, time_limit=10_000_000):
                pass

            acc_base_col = base_col_id_root + acc_stage * acc_region_cols

            for mi in cutlass.range_constexpr(mma_size_m):
                coord_m = coord_m_tile + mi * epi_rows_per_mma_m
                mi_col_base = acc_base_col + mi * epi_cols_per_mma_m
                tmem_col_addr_gemms = [(row_id_with_warp_offset << 16) | (mi_col_base + g * cols_per_acc_stage) for g in range(num_gemms)]

                if cutlass.const_expr(epi_packed_lanes):
                    row = coord_m + warp_idx * 16 + lane
                    row_active = lane < 16
                elif cutlass.const_expr(epi_dp22):
                    row = coord_m + (warp_idx % 2) * 32 + lane
                    row_active = True
                else:
                    row = coord_m + tidx
                    row_active = True

# @@FROST_ACTIVATION_1@@
                _aux_scale_ptr = scale.iterator.raw_ptr()
                _aux_scale_pre = (_aux_scale_ptr + 0).load()

                for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                    subtile_col_offset, subtile_w = epi_spans[subtile_idx]
                    c_rmem_vecs = []
                    for g in cutlass.range_constexpr(num_gemms):
                        subtile_tmem_addr = tmem_col_addr_gemms[g] + subtile_col_offset
                        tmem = cutlass.inttoptr(subtile_tmem_addr, 6, mma_c_dtype)
                        _cv = nvvm.tcgen05_ld(shape, tmem, num=subtile_w, offset=ld_half_off)
                        # INT8 int32 accumulate → widen to fp32 (skipped for int32 output).
                        if cutlass.const_expr(acc_widen_to_fp32):
                            _accf = _cv.to(cutlass.Float32)
                            # `+ 0.0` forces a fresh fp32 register so int32->fp32 isn't folded into an invalid int32->fp8 cast.
                            _cv = _accf + cutlass.full_like(_accf, 0.0)
                        c_rmem_vecs.append(_cv)
                    c_rmem_vec = c_rmem_vecs[0]

                    if mi == mma_size_m - 1 and subtile_idx == subtile_cnt - 1:
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


                    if row_active and row < group_end:
                        for j in cutlass.range_constexpr(subtile_w // vsize):
                            col_j = col + j * vsize
                            if col_j + vsize <= N:
                                vec_f32 = c_rmem_vec[j * vsize : (j + 1) * vsize]

                                vec_f32_1 = c_rmem_vecs[1][j * vsize : (j + 1) * vsize]

                                _c_0_a = (vec_f32).to(cutlass.Float32)
# @@FROST_ACTIVATION_2@@
                                _c_1_a = (_op_0).to(cutlass.Float32)
# @@FROST_ACTIVATION_3@@
                                (gC_tap_0_ptr + (row * out_stride_m_0 + col_j)).store(_tap_0, alignment=VEC_BYTES_TAP_0)

            # The M-major TMA path loads its accumulator inside the store loop, so its release cannot move up.
            # @@EPILOGUE_DRAIN:END@@
            tile_iter += 1

            while not nvvm.mbarrier_try_wait_parity(
                sched_full_mbar_ptr.subview(sched_stage),
                sched_full_phase,
                time_limit=10_000_000,
            ):
                pass
            _slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
            tile_m = (_slot.subview(1)).load()
            tile_n = (_slot.subview(2)).load()
            is_valid = (_slot.subview(3)).load()
            group_begin = (_slot.subview(4)).load()
            group_end = (_slot.subview(5)).load()
            group_idx = (_slot.subview(7)).load()
            nvvm.bar_warp_sync(0xFFFFFFFF)
            sched_stage = cute.arch.make_warp_uniform(sched_stage)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

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
                        while not nvvm.mbarrier_try_wait_parity(token_local_ready.subview(relay_stage), relay_full_phase, time_limit=10_000_000):
                            pass
                        if elect_one:
                            nvvm.inline_ptx("fence.release.sync_restrict::shared::cta.cluster;")
                            nvvm.mbarrier_arrive(nvvm.mapa(ab_full_mbar_ptr.subview(relay_stage), pair_leader_rank), scope=nvvm.MemScope.CLUSTER, relaxed=True)
                        relay_iter += 1

    if warp_idx >= 8 and warp_idx < 12:
        # A twelve-warp CTA enters with 168 registers per thread. Both copy
        # budgets are below that entry allocation, including the large tile.
        nvvm.setmaxregister(_gather_copy_regs, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        copy_thread = (warp_idx - 8) * 32 + tidx % 32
        copy_chunk = copy_thread % _gather_chunks_per_row
        copy_tokens = cute.make_rmem_tensor(cute.make_layout((_gather_rows_per_thread,)), cutlass.Int32)
        if cutlass.const_expr(_gather_cache_bases):
            copy_bases = [cute.make_rmem_tensor(cute.make_layout((_gather_rows_per_thread,)), cutlass.Int64) for _ in range(num_token_operands)]
            copy_live = [cute.make_rmem_tensor(cute.make_layout((_gather_rows_per_thread,)), cutlass.Int32) for _ in range(num_token_operands)]
        copy_iter = cutlass.Int32(0)
        copy_empty_phase = cutlass.Int32(1)
        copy_sched_stage = cutlass.Int32(0)
        copy_sched_phase = cutlass.Int32(0)
        copy_valid = cutlass.Int32(1)
        while copy_valid != 0:
            while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr.subview(copy_sched_stage), copy_sched_phase, time_limit=10_000_000):
                pass
            copy_slot = sched_storage.subview(copy_sched_stage * SCHED_SLOT_WORDS)
            copy_tile = copy_slot.subview(1).load()
            copy_valid = copy_slot.subview(3).load()
            copy_group_begin = copy_slot.subview(4).load()
            copy_group_end = copy_slot.subview(5).load()
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(copy_sched_stage))
            copy_sched_stage += 1
            if copy_sched_stage == SCHED_STAGES:
                copy_sched_stage = cutlass.Int32(0)
                copy_sched_phase = copy_sched_phase ^ 1
            if copy_valid != 0:
                copy_coord = copy_tile * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
                for copy_part in cutlass.range_constexpr(_gather_rows_per_thread):
                    copy_row = copy_thread // _gather_chunks_per_row + copy_part * _gather_rows_per_round
                    copy_token = cutlass.Int32(-1)
                    if cutlass.const_expr(gather_token_rows % _gather_rows_per_round == 0) or copy_row < gather_token_rows:
                        copy_token = moe_gather_row(token_index, copy_group_begin + copy_coord + copy_row, copy_group_end, cutlass.Int32(-1))
                    copy_tokens[copy_part] = copy_token
                if cutlass.const_expr(_gather_cache_bases):
                    for cache_operand in cutlass.range_constexpr(num_token_operands):
                        cache_source_rows = cutlass.Uint32(cutlass.min(mA_list[cache_operand].shape[0], cutlass.Int64(1 << 31)))
                        for cache_part in cutlass.range_constexpr(_gather_rows_per_thread):
                            cache_token = copy_tokens[cache_part]
                            cache_is_live = cutlass.Uint32(cache_token) < cache_source_rows
                            copy_live[cache_operand][cache_part] = cutlass.Int32(cache_is_live)
                            cache_safe_token = cutlass.Int32(0)
                            if cache_is_live:
                                cache_safe_token = cache_token
                            copy_bases[cache_operand][cache_part] = mA_list[cache_operand].iterator.toint() + cutlass.Int64(cache_safe_token) * a_stride_m_list[
                                cache_operand
                            ] * (ab_dtype.width // 8)
                for copy_k_tile in range(num_k_tiles):
                    copy_stage = copy_iter % ab_stages
                    if copy_stage == 0 and copy_iter != 0:
                        copy_empty_phase = copy_empty_phase ^ 1
                    while not nvvm.mbarrier_try_wait_parity(
                        ab_empty_mbar_ptr.subview(copy_stage), copy_empty_phase, time_limit=10_000_000, scope=_OPERAND_BARRIER_SCOPE
                    ):
                        pass
                    copy_column = (copy_k_tile * _gather_chunks_per_row + copy_chunk) * 16
                    for copy_operand in cutlass.range_constexpr(num_token_operands):
                        copy_source_bytes = mA_list[copy_operand].shape[1] * (ab_dtype.width // 8)
                        for copy_part in cutlass.range_constexpr(_gather_rows_per_thread):
                            copy_row = copy_thread // _gather_chunks_per_row + copy_part * _gather_rows_per_round
                            if cutlass.const_expr(gather_token_rows % _gather_rows_per_round == 0) or copy_row < gather_token_rows:
                                if cutlass.const_expr(_gather_cache_bases):
                                    copy_base = copy_bases[copy_operand][copy_part]
                                    copy_is_live = copy_live[copy_operand][copy_part] != 0
                                else:
                                    copy_token = copy_tokens[copy_part]
                                    copy_source_rows = cutlass.Uint32(cutlass.min(mA_list[copy_operand].shape[0], cutlass.Int64(1 << 31)))
                                    copy_is_live = cutlass.Uint32(copy_token) < copy_source_rows
                                    copy_safe_token = cutlass.Int32(0)
                                    if copy_is_live:
                                        copy_safe_token = copy_token
                                    copy_base = mA_list[copy_operand].iterator.toint() + cutlass.Int64(copy_safe_token) * a_stride_m_list[copy_operand] * (
                                        ab_dtype.width // 8
                                    )
                                copy_size = cutlass.Int32(0)
                                copy_safe_column = cutlass.Int32(0)
                                if copy_is_live & (copy_column < copy_source_bytes):
                                    copy_safe_column = cutlass.Int32(copy_column)
                                    copy_size = cutlass.min(cutlass.Int32(16), cutlass.Int32(copy_source_bytes - copy_column))
                                copy_src = cutlass.inttoptr(copy_base + copy_safe_column, 1, cutlass.Uint8)
                                copy_dst = smem_a_list[copy_operand].data_ptr(
                                    sA_elems * copy_stage + copy_row * cta_tile_mnk[2] + copy_chunk * (16 // (ab_dtype.width // 8))
                                )
                                copy_dst = cutlass.apply_swizzle(copy_dst, cutlass.Swizzle(gather_swizzle_bits, 4, 3))
                                nvvm.cp_async_shared_global(copy_dst, copy_src, 16, nvvm.LoadCacheModifier.CG, cp_size=copy_size)
                    if cutlass.const_expr(cta_group == 2):
                        nvvm.cp_async_mbarrier_arrive(token_local_ready.subview(copy_stage), noinc=True)
                    else:
                        nvvm.cp_async_mbarrier_arrive(ab_full_mbar_ptr.subview(copy_stage), noinc=True)
                    copy_iter += 1
        nvvm.cp_async_commit_group()
        nvvm.cp_async_wait_group(0)


frost_template_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    problem_size: tuple,
    first_token_offset: cute.Tensor,
    a_tma_workspace: cute.Tensor,
    token_index: cute.Tensor,
    a_0: cute.Tensor,
    b_0: cute.Tensor,
    b_1: cute.Tensor,
    c_tap_0: cute.Tensor,
# @@FROST_ACTIVATION_4@@
    scale: cute.Tensor,
    stream: _cuda.CUstream,
) -> None:
    _a_operands = [a_0]
    _b_operands = [b_0, b_1]

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


    tma_a_desc_list = []
    for _a_idx, _a_op in enumerate(_a_operands):
        a_stride_m, a_stride_k, a_stride_l = _a_stride_sets[_a_idx]
        a_dims = [k_sym, _a_op.shape[0]]
        a_strides = [a_stride_m * ab_dtype.width // 128]
        a_box = [cta_tile_mnk[2], 1]
        tma_a_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_a_op.iterator.toint(),
                dtype=ab_tma_dtype,
                global_dims=a_dims,
                global_strides=a_strides,
                box_dims=a_box,
                swizzle=ab_tma_swizzle,
            )
        )
    tma_b_desc_list = []
    for _b_idx, _b_op in enumerate(_b_operands):
        b_stride_n, b_stride_k, b_stride_l = _b_stride_sets[_b_idx]
        if cutlass.const_expr(b_is_n_major):
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[n, k_sym, num_experts],
                    global_strides=[
                        b_stride_k * ab_dtype.width // 128,
                        b_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[b_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=ab_tma_swizzle,
                )
            )
        else:
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[k_sym, n, num_experts],
                    global_strides=[
                        b_stride_n * ab_dtype.width // 128,
                        b_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[1] // b_mcast_slices, 1],
                    swizzle=ab_tma_swizzle,
                )
            )

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    grid_shape = (grid_num_clusters * cluster_m, cluster_n, 1)
    counter_qword = grid_num_clusters * cluster_m * cluster_n * moe_desc_slots * TENSOR_MAP_QWORDS
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
        tma_b_desc_list[1],
        a_0,
        _a_stride_sets[0][0],
        token_index,
        c_tap_0,
        out_stride_m_0,
        out_stride_n_0,
        out_stride_l_0,
# @@FROST_ACTIVATION_5@@
        scale,
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
    ab_stride_elems = 16 // (ab_dtype.width // 8)
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=out_vec_elems)
    sym_k = cute.sym_int64()
    sym_e = cute.sym_int64()
    sym_g = cute.sym_int64()
    sym_source_m = cute.sym_int64()

    def _make_fake_a():
        return make_fake_compact_tensor(
            mma_a_dtype,
            (sym_source_m, sym_k, 1),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            mma_b_dtype,
            (sym_n, sym_k, sym_e),
            stride_order=(0, 1, 2) if b_is_n_major else (1, 0, 2),
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
    fake_a_tma_workspace = make_fake_compact_tensor(
        cutlass.Int64,
        (grid_ctas * moe_desc_slots * 16 + 16,),
        stride_order=(0,),
        assumed_align=128,
    )

    def _sym_operand_strides(is_mn_major: bool) -> tuple:
        # Operand is permuted to (M|N, K, L): the unit stride is mode 0 when MN-major, mode 1 when K-major, and never reaches TMA.
        unit = 0 if is_mn_major else 1
        return tuple(cute.sym_int64() if i == unit else cute.sym_int64(divisibility=ab_stride_elems) for i in range(3))

    sym_a_strides = []
    for _ in range(num_a_operands):
        sym_a_strides.extend(_sym_operand_strides(a_is_m_major))
    sym_b_strides = []
    for _ in range(num_b_operands):
        sym_b_strides.extend(_sym_operand_strides(b_is_n_major))
    sym_out_stride_m_0 = cute.sym_int64()
    sym_out_stride_n_0 = cute.sym_int64()
    sym_out_stride_l_0 = cute.sym_int64()
    fake_a_0 = _make_fake_a()
    fake_b_0 = _make_fake_b()
    fake_b_1 = _make_fake_b()
    fake_token_index = make_fake_compact_tensor(cutlass.Int32, (sym_m,), stride_order=(0,), assumed_align=4)
    fake_c_tap_0 = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (sym_m, sym_n, 1),
        stride=(cute.sym_int64(), 1, cute.sym_int64()),
        assumed_align=32,
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
# @@FROST_ACTIVATION_6@@
    fake_scale = cute.runtime.make_fake_tensor(cutlass.Float32, (1, 1, 1), stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()), assumed_align=4)
    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
# @@FROST_ACTIVATION_7@@
