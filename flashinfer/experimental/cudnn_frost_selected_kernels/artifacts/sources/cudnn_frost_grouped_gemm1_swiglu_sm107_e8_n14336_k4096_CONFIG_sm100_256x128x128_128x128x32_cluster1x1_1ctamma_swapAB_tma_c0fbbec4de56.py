# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM100 MoE weight-by-token matmul with grouped N scheduling."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cutlass.experimental.primitives as nvvm
# Inlined from cudnn.gemm.frost.sm100.kernel_templates._tile_helpers
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Tile-level helpers shared by the rendered kernel templates.

A template is RENDERED (its `@@INJECT_*@@` blocks become module-level
constants) and then exec'd from the kernel cache under a synthetic module
name, so it cannot use relative imports and this module is never rendered.
Everything here therefore takes what it needs as ARGUMENTS -- a helper that
reads an injected constant (`mma_size_m`, `tile_swizzle_n`, `ab_dtype`, ...)
has to stay in the template, or be re-signed to receive it.
"""


import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass._mlir.dialects import llvm


@cute.jit
def l2_swizzle_tile(raw_m, raw_n, nt_m, nt_n, swizzle_w, identity=False):
    """N-direction super-block rasterization of the (m, n) cgrp-tile coord, for
    L2 reuse. ``identity=True`` compiles out the general mapping when the caller
    knows that ``swizzle_w == 1``.
    """
    if cutlass.const_expr(identity):
        return raw_m, raw_n
    t = raw_n * nt_m + raw_m
    blk = nt_m * swizzle_w
    sb = t // blk
    off = t - sb * blk
    base_n = sb * swizzle_w
    cur_S = cutlass.min(cutlass.Int32(swizzle_w), nt_n - base_n)
    log_m = off // cur_S
    log_n = base_n + off - log_m * cur_S
    return log_m, log_n


def epi_subtile_spans(cols, epi_n=32):
    """Power-of-two column spans the epilogue drains a tile in (host-side).
    Starts at ``epi_n`` and halves to fit the remainder, so any 8-multiple N is
    covered whatever the widest span is."""
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
def replace_tensormap_global_dim_0(desc_ptr, new_dim) -> None:
    # Public DSL/NVVM builds reject ordinal 0 despite PTX using zero-based
    # dimensions. Patch the shared-memory descriptor directly with legal PTX.
    # The write must stay ordered before the caller copies/fences the descriptor.
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
def moe_group_at(visit_idx, num_groups, num_experts):
    """Visitation index -> routed group index.

    ``num_groups == num_experts`` (or a non-multiple) walks groups in order. Batched MoE
    (``num_groups == B * num_experts``) walks expert-major -- the B groups sharing expert
    ``g % E`` become consecutive, so the expert weight is fetched once instead of B times.
    """
    per_expert = num_groups // cutlass.max(num_experts, cutlass.Int32(1))
    group = visit_idx
    if per_expert > 1 and per_expert * num_experts == num_groups:
        group = (visit_idx % per_expert) * num_experts + (visit_idx // per_expert)
    return group


@cute.jit
def copy_tensormap_to_workspace(src_desc_ptr, dst_i64_ptr) -> None:
    """Copy the 128-byte A tensormap into ``dst_i64_ptr`` (seeds the SMEM copy).

    The trip count is a compile-time constant, so this is a constexpr loop --
    the templates had drifted into two spellings of the same fully-unrolled
    copy (`range_constexpr` vs `range(..., unroll_full=True)`).
    """
    src_words = cute.make_ptr(cutlass.Int64, src_desc_ptr.toint(), mem_space=cute.AddressSpace.generic)
    for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
        dst_i64_ptr.subview(i).store((src_words + i).load())


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
# Tile config: CONFIG_sm100_256x128x128_128x128x32_cluster1x1_1ctamma_swapAB
mma_inst_shape_mnk = (128, 128, 16)
mma_k_dim = 0
cta_group = 1
cgrp_tile_mnk = (256, 128, 64)
cta_tile_mnk = (256, 128, 64)
epi_tile_mn = (128, 32)
threads_per_cta = 256
cluster_shape_mnk = (1, 1, 1)
matmul_a_batch = 1
matmul_b_batch = 1
a_is_m_major = False
b_is_n_major = False
mma_a_major = 0
mma_b_major = 0
ab_stages = 6
b_collector_ok = True
multicast_a = False
multicast_b = False
a_mcast_slices = 1
a_tma_box_m = 256
b_mcast_slices = 1
ab_empty_full_mask = False
ab_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.SWIZZLE_128B
a_smem_desc_leading_byte_offset = 16
a_smem_desc_stride_byte_offset = 1024
a_smem_k_step_bytes = 32
a_smem_m_step_bytes = 16384
a_tma_group_elems = 1
b_smem_desc_leading_byte_offset = 16
b_smem_desc_stride_byte_offset = 1024
b_smem_k_step_bytes = 32
b_tma_group_elems = 1
mma_size_m = 2
mma_size_n = 1
mma_size_k = 4
ab_tma_swizzle = _tma.TensorMapSwizzle.s128b

# Dtype family: A=bf16->MMAbf16, B=bf16->MMAbf16, out=bf16 (K_BYTES=128)
ab_dtype = cutlass.BFloat16
cd_dtype = cutlass.BFloat16
epi_store_dtype = cutlass.BFloat16
mma_a_dtype = cutlass.BFloat16
mma_b_dtype = cutlass.BFloat16
mma_c_dtype = cutlass.Float32
acc_widen_to_fp32 = False
ab_tma_dtype = cutlass.BFloat16
mma_kind = nvvm.Tcgen05MMAKind.F16
epi_n = 32
epi_row_elems = 32
tile_swizzle_n = 0
swizzle_l2_budget_bytes = 44040192
num_gemms = 2
num_a_operands = 2
num_b_operands = 1
gemm_a_idx = (0, 1)
gemm_b_idx = (0, 0)
num_tmem_alloc_cols = 576
tmem_alloc_exclusive = True
acc_stages = 1  # 512 acc cols/stage
grid_num_clusters = 216
offset_cutlass_dtype = cutlass.Int32
vec_bytes_epi = 2
split_k_slices = 1
frost_compile_options = '--enable-tvm-ffi --gpu-arch sm_107a'
n_tma_outputs = 1
moe_aligned_offsets = False
moe_token_alignment = 1
epi_slot_widen = 1
epi_packed_lanes = False
epi_dp22 = False
epi_stage_rows = 128
epi_chunk_elems = 32
ab_stages = 3  # SMEM-D 16400B fixed + cast LOAD 0B/stage + multi-GEMM 32768B/stage
fallback_cluster_shape_mnk = None
mixed_a_pattern_pref = 1
mixed_b_pattern_pref = 1
mixed_a_pattern_fb = 1
mixed_b_pattern_fb = 1

# Tensormap workspace slots per CTA: the B operands, plus the output descriptor
# when the TMA-store epilogue re-dimensions it per routed group.
moe_desc_slots = num_b_operands + n_tma_outputs
_CTA_GROUP = nvvm.CTAGroup.CTA_2 if cta_group == 2 else nvvm.CTAGroup.CTA_1


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
    """N-super-block width for one routed token group."""
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
def frost_sm100_moe_grouped_matmul_fwd_swap_ab_256x128x128_128x128x32_cluster1x1_1ctamma_swapAB(
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
    mB_0: cute.Tensor,
    b_stride_n_0: cutlass.Int64,
    out_stride_m_0: cutlass.Int64,
    out_stride_n_0: cutlass.Int64,
    out_stride_l_0: cutlass.Int64,
    scale: cute.Tensor,
    tma_c_desc_0: cutlass.GridConstant[_tma.TensorMap],
) -> None:
    tma_a_descs = [tma_a_desc_0, tma_a_desc_1]
    tma_b_descs = [tma_b_desc_0]
    mB_list = [mB_0]
    b_stride_n_list = [b_stride_n_0]
    tma_c_descs = [tma_c_desc_0]
    tma_c_m_major = (True,)

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
        (tma_workspace.iterator.raw_ptr() + grid_num_clusters * cluster_m * cluster_n * moe_desc_slots * TENSOR_MAP_QWORDS).toint(),
        mem_space=cute.AddressSpace.generic,
    )

    if warp_idx == mma_warp_id:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())

        for _ci in cutlass.range_constexpr(n_tma_outputs):
            nvvm.prefetch_tensormap(tma_c_descs[_ci].get_ptr())

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

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
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
    sched_bcast_empty_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_BCAST_STAGES, space=cutlass.AddressSpace.smem, alignment=8)

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

    tma_b_desc_smem_list = [
        cutlass.Array(
            cutlass.Int64,
            TENSOR_MAP_QWORDS,
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        for _ in range(num_b_operands)
    ]

    # One epilogue subtile = one MMA-M block x 32 cols; the M blocks reuse it.
    # The ring slot is indexed by `tidx`, so its row count is the EPILOGUE THREAD
    # count -- which is epi_tile_mn[0] only when the MMA M block is 128.
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

    # One block per MMA mode, every count defined in both: the 2-CTA pair
    # releases per PAIR (so ab_empty counts pairs and acc_empty counts both
    # CTAs' epilogue warps), the 1-CTA one per CTA.
    if cutlass.const_expr(cta_group == 2):
        ab_empty_count = cluster_size // cta_group if cutlass.const_expr(ab_empty_full_mask) else (cluster_m // cta_group) + cluster_n - 1
        acc_empty_count = num_epilogue_warps * 2
    else:
        ab_empty_count = cluster_size if cutlass.const_expr(ab_empty_full_mask) else cluster_m + cluster_n - 1
        acc_empty_count = num_epilogue_warps
    sched_empty_count = 1 + 1 + num_epilogue_warps
    if warp_idx == 0:
        if cutlass.const_expr(cta_group == 2):
            if elect_one:
                nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, 32)
        for i in range(ab_stages):
            if elect_one:
                nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
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
                nvvm.mbarrier_init(sched_bcast_empty_mbar_ptr.subview(i), cluster_size)
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
    if cutlass.const_expr(cta_group == 1):
        num_tma_copy_bytes = num_a_operands * sA_bytes + num_b_operands * sB_bytes
    else:
        num_tma_copy_bytes = (num_a_operands * sA_bytes + num_b_operands * sB_bytes) * 2

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
        # TMEM accumulator layout, per acc stage: gemm g, M block mi, N block ni
        # -> columns [g*cols_per_acc_stage + mi*epi_cols_per_mma_m + ni*mma_inst_n, +N),
        # all at TMEM lane base 0.
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

    pass

    vsize = epi_chunk_elems

    M = m
    N = n
    clusters_along_m = cute.ceil_div(cutlass.Int32(M), cgrp_tile_mnk[0])
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    num_k_blocks = cta_tile_mnk[2] // mma_inst_shape_mnk[2]
    first_token_arr = cutlass.make_array_view(first_token_offset)

    if warp_idx == scheduler_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        full_warp_mask = 0xFFFFFFFF
        shfl_idx_clamp = 0x1F
        shfl_up_clamp = 0
        lane = cute.arch.lane_idx()
        gemm_s = cutlass.Int32(N)
        sched_stage = cutlass.Int32(0)
        sched_empty_phase = cutlass.Int32(1)
        bcast_stage = cutlass.Int32(0)
        bcast_full_phase = cutlass.Int32(0)
        bcast_empty_phase = cutlass.Int32(1)
        last_bcast_stage = cutlass.Int32(0)
        last_bcast_empty_done_phase = cutlass.Int32(0)
        linear_idx = cutlass.Int32(0)
        start_linear_idx = cutlass.Int32(0)
        total_tiles = cutlass.Int32(0)
        scan_base = cutlass.Int32(0)
        group_idx = cutlass.Int32(0)
        group_begin = cutlass.Int32(0)
        group_end = cutlass.Int32(0)
        is_tile_valid = cutlass.Int32(1)

        while is_tile_valid != 0:
            # Dynamic tile assignment: the cluster leader claims the next GLOBAL
            # tile index and broadcasts it, so the clusters live at any instant
            # sit in one contiguous window of tile space and share L2. Static
            # striding lets them drift apart and share nothing.
            if cta_rank_in_cluster == 0:
                while not nvvm.mbarrier_try_wait_parity(
                    sched_bcast_empty_mbar_ptr.subview(bcast_stage),
                    bcast_empty_phase,
                    time_limit=10_000_000,
                ):
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
            ):
                pass
            linear_idx = (sched_bcast_slot.subview(bcast_stage)).load()
            if lane == 0:
                nvvm.mbarrier_arrive(nvvm.mapa(sched_bcast_empty_mbar_ptr.subview(bcast_stage), 0))
            if cutlass.const_expr(cluster_size > 1):
                # The final invalid broadcast has no next reuse of this stage to
                # wait for its cluster-wide acknowledgements, so retain its exact
                # stage and completion parity for the scheduler-warp drain below.
                last_bcast_stage = bcast_stage
                last_bcast_empty_done_phase = bcast_empty_phase ^ 1
            bcast_stage += 1
            if bcast_stage == SCHED_BCAST_STAGES:
                bcast_stage = cutlass.Int32(0)
                bcast_full_phase = bcast_full_phase ^ 1
                bcast_empty_phase = bcast_empty_phase ^ 1
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
                        if my_group + 1 < num_groups:
                            my_end = cutlass.Int32(first_token_arr[my_group + 1])
                        else:
                            my_end = gemm_s
                        my_tiles = cute.ceil_div(my_end - my_begin, cgrp_tile_mnk[1]) * clusters_along_m
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
                (slot.subview(7)).store(group_idx)
                nvvm.mbarrier_arrive(sched_full_mbar_ptr.subview(sched_stage))

            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_empty_phase = sched_empty_phase ^ 1

        # Every multi-CTA cluster emits one invalid scheduler record before
        # leaving the loop. Keep the leader CTA's DSM alive until every
        # scheduler warp has consumed that final broadcast and acknowledged the
        # leader slot. A singleton cluster has no remote DSM lifetime to drain.
        if cutlass.const_expr(cluster_size > 1):
            if cta_rank_in_cluster == 0:
                # Each acknowledgement flips the saved pre-wrap empty phase, so
                # pre-wrap ``bcast_empty_phase ^ 1`` is the completed parity.
                while not nvvm.mbarrier_try_wait_parity(
                    sched_bcast_empty_mbar_ptr.subview(last_bcast_stage),
                    last_bcast_empty_done_phase,
                    time_limit=10_000_000,
                ):
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

        lane = tidx % 32
        block_linear = bidx + bidy * gridx
        cta_desc_base_list = [tma_workspace.iterator.raw_ptr() + (block_linear * moe_desc_slots + _bj) * TENSOR_MAP_QWORDS for _bj in range(num_b_operands)]
        b_desc_tma_ptr_list = [
            cute.make_ptr(
                cutlass.Int64,
                cta_desc_base_list[_bj].toint(),
                mem_space=cute.AddressSpace.generic,
            )
            for _bj in range(num_b_operands)
        ]
        previous_group_begin = cutlass.Int32(-1)
        if cutlass.const_expr(moe_aligned_offsets):
            b_desc_load_list = [tma_b_descs[_bj].get_ptr() for _bj in range(num_b_operands)]
        else:
            b_desc_load_list = b_desc_tma_ptr_list
        if elect_one and cutlass.const_expr(not moe_aligned_offsets):
            for _bj in cutlass.range_constexpr(num_b_operands):
                _copy_tensormap_to_workspace(tma_b_descs[_bj].get_ptr(), tma_b_desc_smem_list[_bj])
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
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                coord_m_desc = tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
                coord_n_group = tile_n * cgrp_tile_mnk[1] + n_rank * (cta_tile_mnk[1] * cta_group)
                if cutlass.const_expr(moe_aligned_offsets):
                    coord_n_desc = group_begin + coord_n_group
                else:
                    coord_n_desc = coord_n_group
                coord_n_per_cta = coord_n_desc + pair_member * cta_tile_mnk[1]

                if group_begin != previous_group_begin and cutlass.const_expr(not moe_aligned_offsets):
                    previous_group_begin = group_begin
                    for _bj in cutlass.range_constexpr(num_b_operands):
                        _fence_tensormap_acquire(b_desc_tma_ptr_list[_bj])
                    for _bj in cutlass.range_constexpr(num_b_operands):
                        if elect_one:
                            row_base = mB_list[_bj].iterator.raw_ptr() + group_begin * b_stride_n_list[_bj]
                            _replace_tensormap_global_address(tma_b_desc_smem_list[_bj], row_base.toint())
                            _replace_tensormap_global_dim_1(tma_b_desc_smem_list[_bj], group_end - group_begin)
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        if lane < TENSOR_MAP_QWORDS:
                            (cta_desc_base_list[_bj] + lane).store((tma_b_desc_smem_list[_bj].subview(lane)).load())
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        _fence_tensormap_release()

                for k_tile_idx in range(num_k_tiles):
                    stage = ab_iter % ab_stages
                    if stage == 0 and ab_iter != 0:
                        ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(
                        ab_empty_mbar_ptr.subview(stage),
                        ab_empty_phase_bit,
                        time_limit=10_000_000,
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
                    a_issue = (not multicast_a) or (n_rank == 0)
                    if cutlass.const_expr(a_mcast_slices > 1):
                        a_data_issue = True
                        _a_off = n_rank * (cta_tile_mnk[0] // a_mcast_slices)
                    else:
                        a_data_issue = a_issue
                        _a_off = 0
                    if a_data_issue:
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            sA_stage = smem_a_list[_ai].subview(sA_elems * stage)
                            if cutlass.const_expr(a_is_m_major):
                                for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sA_stage.subview(m_group * a_tma_group_elems * cta_tile_mnk[2]),
                                            tma_a_descs[_ai].get_ptr(),
                                            (
                                                coord_m_desc + m_group * a_tma_group_elems,
                                                coord_k,
                                                coord_expert,
                                            ),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_a,
                                            group=_CTA_GROUP,
                                        )
                            else:
                                for _am in cutlass.range_constexpr(cta_tile_mnk[0] // a_mcast_slices // a_tma_box_m):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sA_stage.subview(_a_off * cta_tile_mnk[2] + _am * a_tma_box_m * cta_tile_mnk[2]),
                                            tma_a_descs[_ai].get_ptr(),
                                            (coord_k, coord_m_desc + _a_off + _am * a_tma_box_m, coord_expert),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_a,
                                            group=_CTA_GROUP,
                                        )
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
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sB_stage.subview(_b_off * cta_tile_mnk[2]),
                                    b_desc_load_list[_bj],
                                    (coord_k, coord_n_per_cta + _b_off, cutlass.Int32(0)),
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
                while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(tail_stage), tail_phase, time_limit=10_000_000):
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
            # Per-group TMA replacement changes the GMEM source, not these invariant
            # MMA-side SMEM descriptor roots.
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
                    # One accumulator per (gemm, M block). Column arithmetic stays on
                    # the encoded (row << 16) | col integer.
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
                        ):
                            pass

                        for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                            for g in cutlass.range_constexpr(num_gemms):
                                desc_a_k = desc_a_roots[gemm_a_idx[g]].advance_start_address(sA_bytes * stage + a_smem_k_step_bytes * k_block_idx)
                                desc_b_k = desc_b_roots[gemm_b_idx[g]].advance_start_address(sB_bytes * stage + b_smem_k_step_bytes * k_block_idx)
                                for mi in cutlass.range_constexpr(mma_size_m):
                                    # The M sub-block offset is a whole SMEM swizzle atom,
                                    # so the descriptor's swizzle phase is preserved.
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
                            # Every accumulator sees scale_d=False on exactly the first
                            # k_block of the tile, so the flip stays outside mi.
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
                # Per-group TMA replacement changes the GMEM source, not these
                # invariant MMA-side SMEM descriptor roots.
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
                        # One accumulator per (gemm, M block). Column arithmetic stays on the
                        # encoded (row << 16) | col integer.
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
                            ):
                                pass

                            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                                for g in cutlass.range_constexpr(num_gemms):
                                    desc_a_k = desc_a_roots[gemm_a_idx[g]].advance_start_address(sA_bytes * stage + a_smem_k_step_bytes * k_block_idx)
                                    desc_b = desc_b_roots[gemm_b_idx[g]].advance_start_address(sB_bytes * stage + b_smem_k_step_bytes * k_block_idx)
                                    for mi in cutlass.range_constexpr(mma_size_m):
                                        # The M sub-block offset is a whole SMEM swizzle atom, so the
                                        # descriptor's swizzle phase is preserved. B is shared.
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
                                # Every accumulator sees scale_d=False on exactly the first
                                # k_block of the tile, so the flip stays outside mi/ni.
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

        # The drain layout follows the MMA INSTRUCTION's M: at hardware M=64 the
        # accumulator occupies data paths 0-15 of each sub-partition, so the
        # 32-data-path LDTM layout cannot be used to read it.
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

        epi_stage_idx = cutlass.Int32(EPI_SMEM_STAGES - 1)
        # The routed output is one flat (N, S) TMA surface.
        tile_l = cutlass.Int32(0)
        epi_block_linear = bidx + bidy * gridx
        d_desc_base_list = [
            tma_workspace.iterator.raw_ptr() + (epi_block_linear * moe_desc_slots + num_b_operands + _di) * TENSOR_MAP_QWORDS for _di in range(n_tma_outputs)
        ]
        d_desc_ptr_list = [cute.make_ptr(cutlass.Int64, _b.toint(), mem_space=cute.AddressSpace.generic) for _b in d_desc_base_list]
        previous_group_end = cutlass.Int32(-1)
        if warp_idx == 0 and cutlass.const_expr(not moe_aligned_offsets):
            for _di in cutlass.range_constexpr(n_tma_outputs):
                if elect_one:
                    _copy_tensormap_to_workspace(tma_c_descs[_di].get_ptr(), tma_c_desc_smem.subview(_di * TENSOR_MAP_QWORDS))
            nvvm.bar_warp_sync(0xFFFFFFFF)

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
            coord_m_tile = tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
            # Re-dimension D to this group's last column so the hardware clips the
            # ragged tail; the base stays put, so the store coords are global.
            # Under `moe_aligned_offsets` no tile crosses `group_end` in the
            # first place -- `cgrp_tile_n` divides every group -- so the clip
            # has nothing to clip and the original descriptor serves.
            if warp_idx == 0 and cutlass.const_expr(not moe_aligned_offsets):
                if group_end != previous_group_end:
                    previous_group_end = group_end
                    # One drain retires the in-flight stores of EVERY descriptor,
                    # so it does not repeat per output.
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

                    vec_f32 = c_rmem_vec
                    col_j = col
                    linear_idx = tile_l * out_stride_l_0 + row * out_stride_m_0 + col_j * out_stride_n_0

                    vec_f32_1 = c_rmem_vecs[1]
                    _c_0_a = (vec_f32).to(cutlass.Float32)
                    _op_0 = _c_0_a * cute.math.rcp(cutlass.full_like(_c_0_a, cutlass.Float32(1.0)) + cute.math.exp2(-_c_0_a * cutlass.full_like(_c_0_a, cutlass.Float32(1.4426950408889634)), fastmath=True), approx=True, ftz=True)
                    _c_1_a = (_op_0).to(cutlass.Float32)
                    _c_1_b = (vec_f32_1).to(cutlass.Float32)
                    _op_1 = _c_1_a * _c_1_b
                    _c_2_a = (_op_1).to(cutlass.Float32)
                    _op_2 = _c_2_a * (cutlass.full_like(_c_2_a, _aux_scale_pre.to(cutlass.Float32)))
                    _r_2 = (_op_2).to(cutlass.BFloat16)
                    vec_out = (_r_2).to(cutlass.BFloat16)
                    

                    epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES
                    _tsv_0 = cutlass.Array(base=smem_d_ptr.data_ptr(epi_stage_idx * epi_subtile_elems), shape=4096, dtype=cutlass.BFloat16)
                    _mrow_0 = tidx % 64
                    _mblk_0 = (tidx // 64) * 2048
                    _mx0_0 = _mrow_0 ^ 0
                    _mx1_0 = _mrow_0 ^ 8
                    _mx2_0 = _mrow_0 ^ 16
                    _mx3_0 = _mrow_0 ^ 24
                    _mx4_0 = _mrow_0 ^ 32
                    _mx5_0 = _mrow_0 ^ 40
                    _mx6_0 = _mrow_0 ^ 48
                    _mx7_0 = _mrow_0 ^ 56
                    _tsv_0.data_ptr(_mblk_0 + 0 + _mx0_0).store(vec_out[0 : 1], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 64 + _mx1_0).store(vec_out[1 : 2], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 128 + _mx2_0).store(vec_out[2 : 3], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 192 + _mx3_0).store(vec_out[3 : 4], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 256 + _mx4_0).store(vec_out[4 : 5], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 320 + _mx5_0).store(vec_out[5 : 6], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 384 + _mx6_0).store(vec_out[6 : 7], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 448 + _mx7_0).store(vec_out[7 : 8], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 512 + _mx0_0).store(vec_out[8 : 9], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 576 + _mx1_0).store(vec_out[9 : 10], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 640 + _mx2_0).store(vec_out[10 : 11], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 704 + _mx3_0).store(vec_out[11 : 12], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 768 + _mx4_0).store(vec_out[12 : 13], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 832 + _mx5_0).store(vec_out[13 : 14], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 896 + _mx6_0).store(vec_out[14 : 15], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 960 + _mx7_0).store(vec_out[15 : 16], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1024 + _mx0_0).store(vec_out[16 : 17], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1088 + _mx1_0).store(vec_out[17 : 18], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1152 + _mx2_0).store(vec_out[18 : 19], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1216 + _mx3_0).store(vec_out[19 : 20], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1280 + _mx4_0).store(vec_out[20 : 21], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1344 + _mx5_0).store(vec_out[21 : 22], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1408 + _mx6_0).store(vec_out[22 : 23], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1472 + _mx7_0).store(vec_out[23 : 24], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1536 + _mx0_0).store(vec_out[24 : 25], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1600 + _mx1_0).store(vec_out[25 : 26], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1664 + _mx2_0).store(vec_out[26 : 27], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1728 + _mx3_0).store(vec_out[27 : 28], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1792 + _mx4_0).store(vec_out[28 : 29], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1856 + _mx5_0).store(vec_out[29 : 30], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1920 + _mx6_0).store(vec_out[30 : 31], alignment=2)
                    _tsv_0.data_ptr(_mblk_0 + 1984 + _mx7_0).store(vec_out[31 : 32], alignment=2)
                    cute.arch.fence_view_async_shared()
                    nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)
                    if warp_idx == 0:
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_global_shared_cta(
                                d_desc_ptr_list[0],
                                _tsv_0.data_ptr(0),
                                (coord_m + 0, col),
                            )
                    if warp_idx == 0:
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_global_shared_cta(
                                d_desc_ptr_list[0],
                                _tsv_0.data_ptr(2048),
                                (coord_m + 64, col),
                            )
                        if elect_one:
                            nvvm.cp_async_bulk_commit_group()
                        nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)
                    nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)


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


frost_sm100_moe_grouped_matmul_fwd_swap_ab_256x128x128_128x128x32_cluster1x1_1ctamma_swapAB.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    problem_size: tuple,
    first_token_offset: cute.Tensor,
    tma_workspace: cute.Tensor,
    a_0: cute.Tensor,
    a_1: cute.Tensor,
    b_0: cute.Tensor,
    scale: cute.Tensor,
    c_0: cute.Tensor,
    stream: _cuda.CUstream,
) -> None:
    _a_operands = [a_0, a_1]
    _b_operands = [b_0]

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
        box_dims=[64, 32],
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
                    dtype=ab_tma_dtype,
                    global_dims=[m, k_sym, num_experts],
                    global_strides=[a_stride_k * ab_dtype.width // 128, a_stride_l * ab_dtype.width // 128],
                    box_dims=[a_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=ab_tma_swizzle,
                )
            )
        else:
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[k_sym, m, num_experts],
                    global_strides=[a_stride_m * ab_dtype.width // 128, a_stride_l * ab_dtype.width // 128],
                    box_dims=[cta_tile_mnk[2], a_tma_box_m, 1],
                    swizzle=ab_tma_swizzle,
                )
            )
    tma_b_desc_list = []
    for _b_idx, _b_op in enumerate(_b_operands):
        b_stride_n, b_stride_k, b_stride_l = _b_stride_sets[_b_idx]
        tma_b_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_b_op.iterator.toint(),
                dtype=ab_tma_dtype,
                global_dims=[k_sym, n, 1],
                global_strides=[b_stride_n * ab_dtype.width // 128, b_stride_l * ab_dtype.width // 128],
                box_dims=[cta_tile_mnk[2], cta_tile_mnk[1] // b_mcast_slices, 1],
                swizzle=ab_tma_swizzle,
            )
        )

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    grid_shape = (grid_num_clusters * cluster_m, cluster_n, 1)
    frost_sm100_moe_grouped_matmul_fwd_swap_ab_256x128x128_128x128x32_cluster1x1_1ctamma_swapAB(
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
        b_0,
        _b_stride_sets[0][0],
        out_stride_m_0,
        out_stride_n_0,
        out_stride_l_0,
        scale,
        tma_c_desc_list[0],
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        cluster=cluster_shape_mnk,
        use_pdl=USE_PDL,
        stream=stream,
    )


@lru_cache(maxsize=None)
def compile() -> Callable:
    out_vec_elems = vec_bytes_epi // (cd_dtype.width // 8)
    ab_stride_elems = 16 // (ab_dtype.width // 8)
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=min(out_vec_elems, moe_token_alignment))
    # K tails are supported: the K loop is ceil_div and the TMA descriptor's global K
    # extent makes a partial box HW zero-filled. The only real K rule is the 16-byte
    # TMA contiguous-extent one, already gated by _tma_alignment_reject.
    sym_k = cute.sym_int64()
    sym_e = cute.sym_int64()
    sym_g = cute.sym_int64()

    def _make_fake_a():
        return make_fake_compact_tensor(
            mma_a_dtype,
            (sym_m, sym_k, sym_e),
            stride_order=(0, 1, 2) if a_is_m_major else (1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            mma_b_dtype,
            (sym_n, sym_k, 1),
            stride_order=(0, 1, 2) if b_is_n_major else (1, 0, 2),
            assumed_align=16,
        )

    fake_first_token_offset = make_fake_compact_tensor(
        offset_cutlass_dtype,
        (sym_g,),
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
    fake_a_1 = _make_fake_a()
    fake_b_0 = _make_fake_b()

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
    fake_scale = cute.runtime.make_fake_tensor(cutlass.Float32, (1, 1, 1), stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()), assumed_align=4)
    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    return cute.compile(_host, problem_size, fake_first_token_offset, fake_tma_workspace, fake_a_0, fake_a_1, fake_b_0, fake_scale, fake_c_0, stream=_fake_stream, options=frost_compile_options)
