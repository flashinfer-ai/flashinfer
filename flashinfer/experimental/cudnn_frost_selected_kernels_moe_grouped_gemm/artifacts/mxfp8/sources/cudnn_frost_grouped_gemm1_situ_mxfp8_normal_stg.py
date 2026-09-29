# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""sm100 MoE grouped block-scale matmul fwd (nvfp4 / mxfp4 / mxfp8,
including mixed MXFP8/MXFP4 on baseline SM100 K32 and Rubin K64).

A one-sided-dequant graph keeps the ordinary block-scale MMA.  Its raw FP8
side has no runtime SF/SMEM/TMA path; four epilogue warps seed the fake side's
32-row TMEM image across the four warp partitions before MMA starts.

Serves both MMA modes; ``cta_group`` is an injected tile constant.

  cta_group=1  single-CTA MMA — every CTA runs its own MMA on its own
               SMEM/TMEM, no leader-follower pair.
  cta_group=2  2-CTA MMA cluster pair — the leader CTA issues the MMA while the
               follower consumes the scheduler ring only. SFB is loaded FULL per
               CTA, so the pair's accumulator region spans the full tile N.

``mma_inst_k_bytes`` selects the MMA-inst K width (64 is SM 10.7+ silicon, gated
by validate_block_scale_config); at 32 every SF word is one utccp atom.

Blocks that genuinely differ between the two MMA modes are expressed with
``cutlass.const_expr(cta_group == N)`` or, where an arm is not statement-shaped,
with ``@@CTA{1,2}_ONLY@@`` marker regions that the renderer strips.
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

# A TMA tensormap is 128 bytes = 16 int64 qwords. The per-group A descriptor
# replacement keeps a per-CTA SMEM copy, patches base/M-dim there, then publishes
# it to the per-CTA GMEM workspace the TMA reads.
# @@FROST_GEOMETRY@@

# Tensormap workspace slots per CTA: every A operand, each real SFA, plus the
# output descriptors that are re-dimensioned per routed group.
moe_desc_slots = num_a_operands + num_sfa_operands + n_tma_outputs
_CTA_GROUP = nvvm.CTAGroup.CTA_2 if cta_group == 2 else nvvm.CTAGroup.CTA_1

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
def _moe_auto_swizzle_w(group_rows, n, k, nt_n):
    """N-super-block width for one routed group, resolved per group.

    Same rule as the dense path, but the "M side" is THIS group's token slice, not the
    whole token tensor: block along the shorter of (group tokens, expert weight), capped
    by what L2 can hold onto. A group spanning one m-tile makes both orders identical.
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
    a_tma_workspace: cute.Tensor,
    tma_a_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_b_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_b_desc_1: cutlass.GridConstant[_tma.TensorMap],
    tma_sfa_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_sfb_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_sfb_desc_1: cutlass.GridConstant[_tma.TensorMap],
    mA_0: cute.Tensor,
    a_stride_m_0: cutlass.Int64,
    mSFA_0: cute.Tensor,
    mC_tap_0: cute.Tensor,
    out_stride_m_0: cutlass.Int64,
    out_stride_n_0: cutlass.Int64,
    out_stride_l_0: cutlass.Int64,
    gate_scale: cute.Tensor,
    linear_scale: cute.Tensor,
    scale: cute.Tensor,
) -> None:
    tma_a_descs = [tma_a_desc_0]
    tma_b_descs = [tma_b_desc_0, tma_b_desc_1]
    tma_sfa_descs = [tma_sfa_desc_0]
    tma_sfb_descs = [tma_sfb_desc_0, tma_sfb_desc_1]

    mA_list = [mA_0]
    a_stride_m_list = [a_stride_m_0]
    mSFA_list = [mSFA_0]

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

    if cutlass.const_expr(cta_group == 2):
        sched_counter_ptr = cute.make_ptr(
            cutlass.Int32,
            (a_tma_workspace.iterator.raw_ptr() + grid_num_clusters * cluster_m * cluster_n * moe_desc_slots * TENSOR_MAP_QWORDS).toint(),
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


    if cutlass.const_expr(cta_group == 1):
        sched_counter_ptr = cute.make_ptr(
            cutlass.Int32,
            (a_tma_workspace.iterator.raw_ptr() + grid_num_clusters * cluster_m * cluster_n * moe_desc_slots * TENSOR_MAP_QWORDS).toint(),
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
    sched_bcast_empty_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_BCAST_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    tma_a_desc_smem_list = [
        cutlass.Array(
            cutlass.Int64,
            TENSOR_MAP_QWORDS,
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        for _ in range(num_a_operands)
    ]
    tma_sfa_desc_smem_list = [
        cutlass.Array(
            cutlass.Int64,
            TENSOR_MAP_QWORDS,
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        for _ in range(num_sfa_operands)
    ]


    sA_elems = sA_packed_elems
    sB_elems = sB_packed_elems
    # Declaration order IS the SMEM layout, and here it is load-bearing.  Every
    # ring ROOT feeds `Tcgen05SmemDesc.build(start_address=...)`, whose lowering
    # (cutlass-dsl experimental/primitives/descriptors.py:513-522, the
    # non-versioned `_tcgen05_mma_smem_desc` intrinsic) keeps only 14 bits of
    # `addr >> 4`: a root at or above 262144 B wraps to the bottom of SMEM with
    # no error, and the SF UTCCP then copies A-operand bytes into the SF TMEM
    # columns -> NaN/inf on every output.  Reachable on sm107 only, whose 327 KiB
    # carveout lets the AB ring run past 256 KiB.  `advance_start_address`
    # (descriptors.py:413-425) is a plain encoded add and carries the per-stage
    # and per-k-step offsets past the line correctly (the d512 SDPA kernels
    # already rely on it), so the SMALL scale-factor rings are declared FIRST and
    # the big A/B rings -- whose roots then stay far below the line -- follow.
    # The compiler models these roots (`_block_scale_smem_desc_roots`), trims
    # `ab_stages` when a deeper ring would still put one past the line (the
    # MoE template at sm107 512x128), and refuses a layout a single stage cannot
    # fit; the CPU test test_block_scale_smem_layout_sm107.py pins this order.
    # Do not reorder.
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
    sched_empty_count = 1 + 1 + num_epilogue_warps
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
                    nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(sf_full_mbar_ptr.subview(i), 1)
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
                    nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(sf_full_mbar_ptr.subview(i), 1)
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

    sA_bytes = sA_elems * (a_smem_dtype.width // 8)
    sB_bytes = sB_elems * (b_smem_dtype.width // 8)
    # The pair leader issues ONE expect_tx for both CTAs, so it counts twice.
    ab_only_copy_bytes = (num_a_operands * sA_tma_bytes + num_b_operands * sB_tma_bytes) * cta_group
    sf_only_copy_bytes = (num_sfa_operands * sfa_smem_bytes + num_sfb_operands * sfb_smem_bytes) * cta_group
    if cutlass.const_expr(cta_group == 2):
        pair_n_size = cgrp_tile_mnk[1] // cluster_n
    # Per-CTA output rows one MMA-M block covers. The pair splits M, so this is
    # the per-CTA mma_tile_m — half the instruction's hardware M.
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
    first_token_arr = cutlass.make_array_view(first_token_offset)

    if warp_idx == scheduler_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        full_warp_mask = 0xFFFFFFFF
        shfl_idx_clamp = 0x1F
        shfl_up_clamp = 0
        lane = cute.arch.lane_idx()
        gemm_s = cutlass.Int32(M)
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
        start_sf_block_m = cutlass.Int32(0)
        total_sf_blocks_m = cutlass.Int32(0)
        group_idx = cutlass.Int32(0)
        is_tile_valid = cutlass.Int32(1)
        cached_next_end = cutlass.Int32(0)
        if lane + 1 < num_groups:
            cached_next_end = cutlass.Int32(first_token_arr[lane + 1])
        else:
            cached_next_end = gemm_s
        tile_lower_bound = nvvm.shfl_sync(full_warp_mask, cached_next_end, 1, shfl_up_clamp, nvvm.Shfl.UP)
        cached_next_begin = cutlass.Int32(0)
        if lane != 0:
            cached_next_begin = tile_lower_bound

        while is_tile_valid != 0:
            # Dynamic tile assignment: the cluster leader claims the next GLOBAL
            # tile index and broadcasts it, so the clusters that are live at any
            # instant sit in one contiguous window of tile space and share L2.
            # Static striding lets them drift apart and share nothing.
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
            # Finish every lane's slot reads before the elected release.
            nvvm.bar_warp_sync(0xFFFFFFFF)
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
                    if next_end_group < num_groups:
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

                    group_m = cached_group_end - cached_group_begin
                    total_tiles = cute.ceil_div(group_m, cgrp_tile_mnk[0]) * clusters_along_n
                    total_sf_blocks_m = cute.ceil_div(group_m, 128)
                    prefix_tiles = total_tiles
                    prefix_sf = total_sf_blocks_m
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
                    start_sf_block_m += prefix_sf - total_sf_blocks_m
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
                        start_sf_block_m = nvvm.shfl_sync(
                            full_warp_mask,
                            start_sf_block_m,
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
                        if group_end_idx < num_groups:
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
                            next_start_sf = start_sf_block_m + total_sf_blocks_m
                            start_sf_block_m = nvvm.shfl_sync(
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
                (slot.subview(6)).store(start_sf_block_m)
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
        if cutlass.const_expr(cta_group == 2):
            logical_cta_tile_n = cgrp_tile_mnk[1] // cluster_n

        lane = tidx % 32
        block_linear = bidx + bidy * gridx
        cta_desc_base_list = [a_tma_workspace.iterator.raw_ptr() + (block_linear * moe_desc_slots + _ai) * TENSOR_MAP_QWORDS for _ai in range(num_a_operands)]
        a_desc_tma_ptr_list = [
            cute.make_ptr(
                cutlass.Int64,
                cta_desc_base_list[_ai].toint(),
                mem_space=cute.AddressSpace.generic,
            )
            for _ai in range(num_a_operands)
        ]
        sfa_desc_base_list = [
            a_tma_workspace.iterator.raw_ptr() + (block_linear * moe_desc_slots + num_a_operands + _ai) * TENSOR_MAP_QWORDS for _ai in range(num_sfa_operands)
        ]
        sfa_desc_tma_ptr_list = [
            cute.make_ptr(
                cutlass.Int64,
                sfa_desc_base_list[_ai].toint(),
                mem_space=cute.AddressSpace.generic,
            )
            for _ai in range(num_sfa_operands)
        ]
        sfa_block_bytes = 512 * (((k // block_size) + 3) // 4)
        previous_group_begin = cutlass.Int32(-1)
        if cutlass.const_expr(moe_aligned_offsets):
            a_desc_load_list = [tma_a_descs[_ai].get_ptr() for _ai in range(num_a_operands)]
            sfa_desc_load_list = [tma_sfa_descs[_ai].get_ptr() for _ai in range(num_sfa_operands)]
        else:
            a_desc_load_list = a_desc_tma_ptr_list
            sfa_desc_load_list = sfa_desc_tma_ptr_list
        if elect_one and cutlass.const_expr(not moe_aligned_offsets):
            for _ai in cutlass.range_constexpr(num_a_operands):
                _copy_tensormap_to_workspace(tma_a_descs[_ai].get_ptr(), tma_a_desc_smem_list[_ai])
            for _ai in cutlass.range_constexpr(num_sfa_operands):
                _copy_tensormap_to_workspace(tma_sfa_descs[_ai].get_ptr(), tma_sfa_desc_smem_list[_ai])
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
            start_sf_block_m = (slot.subview(6)).load()
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
                    coord_n_pair = tile_n * cgrp_tile_mnk[1] + n_rank * logical_cta_tile_n
                if cutlass.const_expr(moe_aligned_offsets):
                    coord_m_desc = group_begin + coord_m_group
                else:
                    coord_m_desc = coord_m_group
                sfa_m_block = coord_m_desc // 128
                if cutlass.const_expr(cta_group == 1):
                    sfb_n_block = coord_n_per_cta // 128
                else:
                    sfb_n_block = coord_n_pair // 128

                if group_begin != previous_group_begin and cutlass.const_expr(not moe_aligned_offsets):
                    previous_group_begin = group_begin
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        _fence_tensormap_acquire(a_desc_tma_ptr_list[_ai])
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        if elect_one:
                            row_base = mA_list[_ai].iterator.raw_ptr().toint() + ((group_begin * a_stride_m_list[_ai] * a_dtype.width) >> 3)
                            _replace_tensormap_global_address(tma_a_desc_smem_list[_ai], row_base)
                            _replace_tensormap_global_dim_1(tma_a_desc_smem_list[_ai], group_end - group_begin)
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        if lane < TENSOR_MAP_QWORDS:
                            (cta_desc_base_list[_ai] + lane).store((tma_a_desc_smem_list[_ai].subview(lane)).load())
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        _fence_tensormap_release()
                    for _ai in cutlass.range_constexpr(num_sfa_operands):
                        _fence_tensormap_acquire(sfa_desc_tma_ptr_list[_ai])
                    for _ai in cutlass.range_constexpr(num_sfa_operands):
                        if elect_one:
                            sfa_base = mSFA_list[_ai].iterator.raw_ptr().toint() + start_sf_block_m * sfa_block_bytes
                            _replace_tensormap_global_address(tma_sfa_desc_smem_list[_ai], sfa_base)
                            _replace_tensormap_global_dim_2(tma_sfa_desc_smem_list[_ai], cute.ceil_div(group_end - group_begin, 128))
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        if lane < TENSOR_MAP_QWORDS:
                            (sfa_desc_base_list[_ai] + lane).store((tma_sfa_desc_smem_list[_ai].subview(lane)).load())
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
                                    sfa_desc_load_list[_ai],
                                    (0, coord_sf_k, sfa_m_block, cutlass.Int32(0)),
                                    sf_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=_CTA_GROUP,
                                )
                    if b_issue:
                        for _bj in cutlass.range_constexpr(num_sfb_operands):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    smem_sfb_list[_bj].subview(sfb_smem_bytes * stage),
                                    tma_sfb_descs[_bj].get_ptr(),
                                    (0, coord_sf_k, sfb_n_block, coord_expert),
                                    sf_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=_CTA_GROUP,
                                )

                    if a_data_issue:
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            for _am in cutlass.range_constexpr(cta_tile_mnk[0] // a_mcast_slices // a_tma_box_m):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        smem_a_list[_ai].subview(sA_elems * stage + _a_off * a_packed_per_row + _am * a_tma_box_m * a_packed_per_row),
                                        a_desc_load_list[_ai],
                                        (coord_k, coord_m_desc + _a_off + _am * a_tma_box_m, cutlass.Int32(0)),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_a,
                                        group=_CTA_GROUP,
                                    )
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
                                        sB_stage.subview(_b_off * b_packed_per_row),
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

            # fp4 packs its K-mode into the OMMA descriptor's 2-bit split field; fp8
            # keeps the MX descriptor's 1-bit one. Both are built once, outside the
            # loops — the fields depend only on j (the scale id within a word).
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
            sfb_scale_ptrs = [nvvm.make_tmem_ptr(b, cutlass.Float32) for b in sfb_tmem_bases]
            # utccp destination per (MN-block, atom within the scale word). SFB is
            # atom-MAJOR across the N-blocks because ONE instruction walks all of
            # them; SFA is block-major because one instruction covers exactly one
            # 128-row block, so that word has to be contiguous. Both collapse to
            # the same addresses at a single block, and to sm100's layout at
            # word_atoms == 1.
            sfa_dst_ptrs = [
                [
                    [nvvm.make_tmem_ptr(sfa_tmem_bases[i] + m * registers_per_block + a * registers_per_atom, cutlass.Float32) for a in range(word_atoms)]
                    for m in range(mma_size_m)
                ]
                for i in range(num_a_operands)
            ]
            sfb_dst_ptrs = [
                [
                    [nvvm.make_tmem_ptr(sfb_tmem_bases[j] + (a * num_blocks_n + m) * registers_per_atom, cutlass.Float32) for a in range(word_atoms)]
                    for m in range(num_blocks_n)
                ]
                for j in range(num_b_operands)
            ]
            # Per-group TMA replacement changes the GMEM source, not these invariant
            # MMA-side SMEM descriptor roots.
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
                    ):
                        pass

                    if cutlass.const_expr(use_acc_overlap):
                        acc_base_col = base_col_id_root + (tile_iter % 2) * acc_stage_stride
                    else:
                        acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                    # One accumulator per (gemm, M block); M block mi sits
                    # epi_cols_per_mma_m columns further into its GEMM's region and
                    # reads SF word block mi (SF words are one per 128 rows).
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

                        # One SF word per group of MMAs, refreshed right before they
                        # read it. A word spans word_atoms consecutive K-atoms in SMEM.
                        while not nvvm.mbarrier_try_wait_parity(
                            sf_full_mbar_ptr.subview(stage),
                            ab_full_phase_bit,
                            time_limit=10_000_000,
                        ):
                            pass

                        for sf_word in cutlass.range_constexpr(num_sf_atoms):
                            for _bj in cutlass.range_constexpr(num_sfb_operands):
                                for block_n in cutlass.range_constexpr(num_blocks_n):
                                    for _a in cutlass.range_constexpr(word_atoms):
                                        if elect_one:
                                            nvvm.tcgen05_cp(
                                                s2t_shape,
                                                sfb_dst_ptrs[_bj][block_n][_a],
                                                desc_sfb_bases[_bj] + (sf_atom_desc_stride * (sf_word * word_atoms + _a) + sf_block_desc_stride * block_n),
                                                group=_CTA_GROUP,
                                                multicast=s2t_multicast,
                                            )
                            if cutlass.const_expr(sf_word == 0):
                                while not nvvm.mbarrier_try_wait_parity(
                                    ab_full_mbar_ptr.subview(stage),
                                    ab_full_phase_bit,
                                    time_limit=10_000_000,
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
                                        if cutlass.const_expr(not fake_dequant_a and mma_k_in_word == 0 and _ai not in gemm_a_idx[:gemm_i]):
                                            for _a in cutlass.range_constexpr(word_atoms):
                                                if elect_one:
                                                    nvvm.tcgen05_cp(
                                                        s2t_shape,
                                                        sfa_dst_ptrs[_ai][mma_m][_a],
                                                        desc_sfa_bases[_ai]
                                                        + (sf_atom_desc_stride * (sf_word * word_atoms + _a) + sf_block_desc_stride * mma_m),
                                                        group=_CTA_GROUP,
                                                        multicast=s2t_multicast,
                                                    )
                                        # The M sub-block offset is a whole SMEM swizzle atom, so
                                        # the descriptor's swizzle phase is preserved. B and its SF
                                        # are shared; A's SF word block follows the M block.
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
                                                scale_a=sfa_dst_ptrs[_ai][mma_m][0],
                                                scale_b=sfb_scale_ptrs[_bj],
                                                scale_vec_size=scale_vec_size,
                                                collector_op=_a_collector_op(gemm_i),
                                                b_collector_op=_b_collector_op(mma_m),
                                            )
                                # Every accumulator sees scale_d=False on exactly the first
                                # k_block of the tile, so the flip stays outside mma_m.
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
                # fp4 packs its K-mode into the OMMA descriptor's 2-bit split field; fp8
                # keeps the MX descriptor's 1-bit one. Both are built once, outside the
                # loops — the fields depend only on j (the scale id within a word).
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
                sfb_scale_ptrs = [nvvm.make_tmem_ptr(b, cutlass.Float32) for b in sfb_tmem_bases]
                # utccp destination per (MN-block, atom within the scale word). SFB
                # is atom-MAJOR across the N-blocks because ONE instruction walks
                # all of them; SFA is block-major because one instruction covers
                # exactly one 128-row block, so that word has to be contiguous.
                # Both collapse to the same addresses at a single block, and to
                # sm100's layout at word_atoms == 1.
                sfa_dst_ptrs = [
                    [
                        [nvvm.make_tmem_ptr(sfa_tmem_bases[i] + m * registers_per_block + a * registers_per_atom, cutlass.Float32) for a in range(word_atoms)]
                        for m in range(mma_size_m)
                    ]
                    for i in range(num_a_operands)
                ]
                sfb_dst_ptrs = [
                    [
                        [nvvm.make_tmem_ptr(sfb_tmem_bases[j] + (a * num_blocks_n + m) * registers_per_atom, cutlass.Float32) for a in range(word_atoms)]
                        for m in range(num_blocks_n)
                    ]
                    for j in range(num_b_operands)
                ]
                # Per-group TMA replacement changes the GMEM source, not these
                # invariant MMA-side SMEM descriptor roots.
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
                        ):
                            pass

                        if cutlass.const_expr(use_acc_overlap):
                            acc_base_col = base_col_id_root + (tile_iter % 2) * acc_stage_stride
                        else:
                            acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                        # One accumulator per (gemm, M block); M block mi sits
                        # epi_cols_per_mma_m columns further into its GEMM's region and
                        # reads SF word block mi (SF words are one per 128 rows).
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

                            # One SF word per group of MMAs, refreshed right before they
                            # read it. A word spans word_atoms consecutive K-atoms in SMEM.
                            while not nvvm.mbarrier_try_wait_parity(
                                sf_full_mbar_ptr.subview(stage),
                                ab_full_phase_bit,
                                time_limit=10_000_000,
                            ):
                                pass

                            for sf_word in cutlass.range_constexpr(num_sf_atoms):
                                for _bj in cutlass.range_constexpr(num_sfb_operands):
                                    for block_n in cutlass.range_constexpr(num_blocks_n):
                                        for _a in cutlass.range_constexpr(word_atoms):
                                            if elect_one:
                                                nvvm.tcgen05_cp(
                                                    s2t_shape,
                                                    sfb_dst_ptrs[_bj][block_n][_a],
                                                    desc_sfb_bases[_bj] + (sf_atom_desc_stride * (sf_word * word_atoms + _a) + sf_block_desc_stride * block_n),
                                                    group=_CTA_GROUP,
                                                    multicast=s2t_multicast,
                                                )
                                if cutlass.const_expr(sf_word == 0):
                                    while not nvvm.mbarrier_try_wait_parity(
                                        ab_full_mbar_ptr.subview(stage),
                                        ab_full_phase_bit,
                                        time_limit=10_000_000,
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
                                            if cutlass.const_expr(not fake_dequant_a and mma_k_in_word == 0 and _ai not in gemm_a_idx[:gemm_i]):
                                                for _a in cutlass.range_constexpr(word_atoms):
                                                    if elect_one:
                                                        nvvm.tcgen05_cp(
                                                            s2t_shape,
                                                            sfa_dst_ptrs[_ai][mma_m][_a],
                                                            desc_sfa_bases[_ai]
                                                            + (sf_atom_desc_stride * (sf_word * word_atoms + _a) + sf_block_desc_stride * mma_m),
                                                            group=_CTA_GROUP,
                                                            multicast=s2t_multicast,
                                                        )
                                            # The M sub-block offset is a whole SMEM swizzle atom, so
                                            # the descriptor's swizzle phase is preserved. B and its SF
                                            # are shared; A's SF word block follows the M block.
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
                                                    scale_a=sfa_dst_ptrs[_ai][mma_m][0],
                                                    scale_b=sfb_scale_ptrs[_bj],
                                                    scale_vec_size=scale_vec_size,
                                                    collector_op=_a_collector_op(gemm_i),
                                                    b_collector_op=_b_collector_op(mma_m),
                                                )
                                    # Every accumulator sees scale_d=False on exactly the first
                                    # k_block of the tile, so the flip stays outside mma_m.
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

        # @@EPILOGUE_SETUP:BEGIN@@
        row_id_with_warp_offset = base_row_id + warp_idx * 32

        epi_spans = _epi_subtile_spans(epi_cols_per_mma_m, epi_n)
        subtile_cnt = len(epi_spans)
        shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
        lane = tidx % 32
        # @@EPILOGUE_SETUP:END@@


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
                start_sf_block_m = (_slot.subview(6)).load()
                group_idx = (_slot.subview(7)).load()
            else:
                slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
                tile_m = (slot.subview(1)).load()
                tile_n = (slot.subview(2)).load()
                is_valid = (slot.subview(3)).load()
                group_begin = (slot.subview(4)).load()
                group_end = (slot.subview(5)).load()
                start_sf_block_m = (slot.subview(6)).load()
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
                coord_m_tile = group_begin + tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
                # @@EPILOGUE_DRAIN:BEGIN@@
                coord_n_c = tile_n * cgrp_tile_mnk[1] + n_rank * (cta_tile_mnk[1] * cta_group)

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

                    _aux_gate_scale_ptr = gate_scale.iterator.raw_ptr()
                    _aux_gate_scale_pre = (_aux_gate_scale_ptr + 0).load()
                    _aux_linear_scale_ptr = linear_scale.iterator.raw_ptr()
                    _aux_linear_scale_pre = (_aux_linear_scale_ptr + 0).load()
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


                        if row_active and row < group_end:
                            for j in cutlass.range_constexpr(subtile_w // vsize):
                                col_j = col + j * vsize
                                if col_j + vsize <= N:
                                    vec_f32 = c_rmem_vec[j * vsize : (j + 1) * vsize]

                                    vec_f32_1 = c_rmem_vecs[1][j * vsize : (j + 1) * vsize]

                                    _c_0_a = (vec_f32).to(cutlass.Float32)
                                    _op_0 = _c_0_a / (cutlass.full_like(_c_0_a, _aux_gate_scale_pre.to(cutlass.Float32)))
                                    _c_1_a = (_op_0).to(cutlass.Float32)
                                    _op_1 = (cutlass.full_like(_c_1_a, cutlass.Float32(1.0)) - cutlass.full_like(_c_1_a, cutlass.Float32(2.0)) * cute.math.rcp(cute.math.exp2(_c_1_a * cutlass.full_like(_c_1_a, cutlass.Float32(2.8853900817779268)), fastmath=True) + cutlass.full_like(_c_1_a, cutlass.Float32(1.0)), approx=True, ftz=True))
                                    _c_2_a = (_op_1).to(cutlass.Float32)
                                    _op_2 = (cutlass.full_like(_c_2_a, _aux_gate_scale_pre.to(cutlass.Float32))) * _c_2_a
                                    _c_3_a = (vec_f32).to(cutlass.Float32)
                                    _op_3 = cute.math.rcp(cutlass.full_like(_c_3_a, cutlass.Float32(1.0)) + cute.math.exp2(-_c_3_a * cutlass.full_like(_c_3_a, cutlass.Float32(1.4426950408889634)), fastmath=True), approx=True, ftz=True)
                                    _c_4_a = (_op_2).to(cutlass.Float32)
                                    _c_4_b = (_op_3).to(cutlass.Float32)
                                    _op_4 = _c_4_a * _c_4_b
                                    _c_5_a = (vec_f32_1).to(cutlass.Float32)
                                    _op_5 = _c_5_a / (cutlass.full_like(_c_5_a, _aux_linear_scale_pre.to(cutlass.Float32)))
                                    _c_6_a = (_op_5).to(cutlass.Float32)
                                    _op_6 = (cutlass.full_like(_c_6_a, cutlass.Float32(1.0)) - cutlass.full_like(_c_6_a, cutlass.Float32(2.0)) * cute.math.rcp(cute.math.exp2(_c_6_a * cutlass.full_like(_c_6_a, cutlass.Float32(2.8853900817779268)), fastmath=True) + cutlass.full_like(_c_6_a, cutlass.Float32(1.0)), approx=True, ftz=True))
                                    _c_7_a = (_op_6).to(cutlass.Float32)
                                    _op_7 = (cutlass.full_like(_c_7_a, _aux_linear_scale_pre.to(cutlass.Float32))) * _c_7_a
                                    _c_8_a = (_op_4).to(cutlass.Float32)
                                    _c_8_b = (_op_7).to(cutlass.Float32)
                                    _op_8 = _c_8_a * _c_8_b
                                    _c_9_a = (_op_8).to(cutlass.Float32)
                                    _op_9 = _c_9_a * (cutlass.full_like(_c_9_a, _aux_scale_pre.to(cutlass.Float32)))
                                    _r_9 = (_op_9).to(cutlass.BFloat16)
                                    _tap_0 = (_r_9).to(cutlass.BFloat16)
                                    (gC_tap_0_ptr + (row * out_stride_m_0 + col_j)).store(_tap_0, alignment=VEC_BYTES_TAP_0)

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


frost_template_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    problem_size: tuple,
    first_token_offset: cute.Tensor,
    a_tma_workspace: cute.Tensor,
    a_0: cute.Tensor,
    b_0: cute.Tensor,
    b_1: cute.Tensor,
    sfa_0: cute.Tensor,
    sfb_0: cute.Tensor,
    sfb_1: cute.Tensor,
    c_tap_0: cute.Tensor,
    gate_scale: cute.Tensor,
    linear_scale: cute.Tensor,
    scale: cute.Tensor,
    stream: _cuda.CUstream,
) -> None:
    _a_operands = [a_0]
    _b_operands = [b_0, b_1]
    _sfa_operands = [sfa_0]
    _sfb_operands = [sfb_0, sfb_1]

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
        tma_a_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_a_op.iterator.toint(),
                dtype=a_tma_desc_dtype,
                global_dims=[k_sym, m, 1],
                global_strides=[
                    a_stride_m * a_dtype.width // 128,
                    a_stride_l * a_dtype.width // 128,
                ],
                box_dims=[cta_tile_mnk[2], a_tma_box_m, 1],
                swizzle=a_tma_swizzle,
                tma_format=a_tma_format,
            )
        )
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
                        b_stride_k * b_dtype.width // 128,
                        b_stride_l * b_dtype.width // 128,
                    ],
                    box_dims=[b_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=b_tma_swizzle,
                    tma_format=b_tma_format,
                )
            )
        else:
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=b_tma_desc_dtype,
                    global_dims=[k_sym, n, num_experts],
                    global_strides=[
                        b_stride_n * b_dtype.width // 128,
                        b_stride_l * b_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[1] // b_mcast_slices, 1],
                    swizzle=b_tma_swizzle,
                    tma_format=b_tma_format,
                )
            )
    rest_k = ((k_sym // block_size) + 3) // 4
    rest_m = (m + 127) // 128 + num_groups
    rest_n = (n + 127) // 128
    tma_sfa_desc_list = []
    for _sfa_op in _sfa_operands:
        sfa_fp16_tensor = cute.make_tensor(
            cute.recast_ptr(_sfa_op.iterator, dtype=cutlass.Float16),
            cute.make_layout(
                (256, rest_k, rest_m, 1),
                stride=(
                    1,
                    256,
                    cute.assume(256 * rest_k, 8),
                    cute.assume(256 * rest_k * rest_m, 8),
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
        sfb_fp16_tensor = cute.make_tensor(
            cute.recast_ptr(_sfb_op.iterator, dtype=cutlass.Float16),
            cute.make_layout(
                (256, rest_k, rest_n, num_experts),
                stride=(
                    1,
                    256,
                    cute.assume(256 * rest_k, 8),
                    cute.assume(256 * rest_k * rest_n, 8),
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
        tma_sfa_desc_list[0],
        tma_sfb_desc_list[0],
        tma_sfb_desc_list[1],
        a_0,
        _a_stride_sets[0][0],
        _sfa_operands[0],
        c_tap_0,
        out_stride_m_0,
        out_stride_n_0,
        out_stride_l_0,
        gate_scale,
        linear_scale,
        scale,
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
    a_stride_elems = 16 // (a_dtype.width // 8)
    b_stride_elems = 16 // (b_dtype.width // 8)
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=out_vec_elems)
    # K tails are supported: the K loop is ceil_div and the TMA descriptor's global K
    # extent makes a partial box HW zero-filled. The only real K rule is the 16-byte
    # TMA contiguous-extent one, already gated by _tma_alignment_reject.
    sym_k = cute.sym_int64()
    # Packed K extent: same reasoning as sym_k -- no CTA-tile multiple is required.
    sym_akp = cute.sym_int64()
    sym_bkp = cute.sym_int64()
    sym_e = cute.sym_int64()
    sym_g = cute.sym_int64()

    def _make_fake_a():
        return make_fake_compact_tensor(
            a_fake_dtype,
            (sym_m, sym_akp, 1),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            b_fake_dtype,
            (sym_n, sym_bkp, sym_e),
            stride_order=(0, 1, 2) if b_is_n_major else (1, 0, 2),
            assumed_align=16,
        )

    # SF reaches the kernel as a base pointer only; the host rebuilds the
    # F8_128x4 view from problem_size. Modes 0/1 and all strides carry no
    # contract; mode 2 keeps its literal plane count (1 for sfa, sym_e for sfb).
    def _make_fake_sfa():
        return cute.runtime.make_fake_tensor(
            sf_cutlass_dtype,
            (cute.sym_int64(), cute.sym_int64(), 1),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            assumed_align=16,
        )

    def _make_fake_sfb():
        return cute.runtime.make_fake_tensor(
            sf_cutlass_dtype,
            (cute.sym_int64(), cute.sym_int64(), sym_e),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
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
    fake_a_tma_workspace = make_fake_compact_tensor(
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
    fake_b_0 = _make_fake_b()
    fake_b_1 = _make_fake_b()
    fake_sfa_0 = _make_fake_sfa()
    fake_sfb_0 = _make_fake_sfb()
    fake_sfb_1 = _make_fake_sfb()

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

    fake_gate_scale = cute.runtime.make_fake_tensor(cutlass.Float32, (1, 1, 1), stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()), assumed_align=4)
    fake_linear_scale = cute.runtime.make_fake_tensor(cutlass.Float32, (1, 1, 1), stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()), assumed_align=4)
    fake_scale = cute.runtime.make_fake_tensor(cutlass.Float32, (1, 1, 1), stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()), assumed_align=4)

    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    return cute.compile(_host, problem_size, fake_first_token_offset, fake_a_tma_workspace, fake_a_0, fake_b_0, fake_b_1, fake_sfa_0, fake_sfb_0, fake_sfb_1, fake_c_tap_0, fake_gate_scale, fake_linear_scale, fake_scale, stream=_fake_stream, options=frost_compile_options)
