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

from __future__ import annotations

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


@cute.jit
def moe_gather_row(token_index, row, group_end, source_rows):
    # Never read beyond the routed group. An out-of-range source row asks TMA
    # to zero-fill the padding while still completing the expected byte count.
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
def moe_combine_add_vector(ptr, values, dtype: cutlass.Constexpr, count: cutlass.Constexpr):
    """At most 16 bytes of adjacent output elements, with scalar alignment ABI.

    The full destination address includes the routed token's runtime row
    stride. Halve the access when it is not naturally aligned; this preserves
    support for shifted pointers and odd/padded row strides on plan reuse.
    Atomicity remains per element, exactly as in the scalar path.
    """
    if cutlass.const_expr(count == 1):
        moe_combine_add(ptr, values[0], dtype)
    else:
        if ptr.toint() % (count * dtype.width // 8) == 0:
            if cutlass.const_expr(dtype == cutlass.Float32):
                words = values
                nwords = count
                instruction = f"red.global.add.v{count}.f32"
                constraint = "f"
            else:
                words = values.to(dtype).bitcast(cutlass.Uint32)
                nwords = count // 2
                scalar_type = "bf16x2" if cutlass.const_expr(dtype == cutlass.BFloat16) else "f16x2"
                vector_type = "" if cutlass.const_expr(count == 2) else f"v{count // 2}."
                instruction = f"red.global.add.noftz.{vector_type}{scalar_type}"
                constraint = "r"
            operands = ", ".join(f"${i + 1}" for i in range(nwords))
            operands = operands if cutlass.const_expr(nwords == 1) else "{" + operands + "}"
            llvm.inline_asm(
                None,
                [ptr.toint().ir_value()] + [words[i].ir_value() for i in range(nwords)],
                instruction + " [$0], " + operands + ";",
                "l," + ",".join([constraint] * nwords) + ",~{memory}",
                has_side_effects=True,
            )
        else:
            moe_combine_add_vector(ptr, values[: count // 2], dtype, count // 2)
            moe_combine_add_vector(ptr + count // 2, values[count // 2 :], dtype, count // 2)


@cute.jit
def moe_scatter_row(token_index, token_ks, row, group_end, output_rows, top_k: cutlass.Constexpr):
    """Map a routed row to its token/top-k slot; clip padding with an OOB row."""
    dst = cutlass.Int32(output_rows)
    if row < group_end:
        token = cutlass.Int32(token_index[row])
        slot = cutlass.Int32(token_ks[row])
        if token >= 0 and token < output_rows // top_k and slot >= 0 and slot < top_k:
            dst = token * top_k + slot
    return dst


@cute.jit
def tma_scatter4(desc, src, col, r0, r1, r2, r3):
    """Store four indexed rows through a rank-2 map with box_dims[1] == 1."""
    llvm.inline_asm(
        None,
        [
            desc.toint().ir_value(),
            src.toint(dtype=cutlass.Int32).ir_value(),
            cutlass.Int32(col).ir_value(),
            cutlass.Int32(r0).ir_value(),
            cutlass.Int32(r1).ir_value(),
            cutlass.Int32(r2).ir_value(),
            cutlass.Int32(r3).ir_value(),
        ],
        "cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group [$0, {$2, $3, $4, $5, $6}], [$1];",
        "l,r,r,r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


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


@cute.jit
def moe_gather_scales(
    dst,
    scratch,
    load_bar,
    ready_bar,
    desc,
    token_index,
    row_begin,
    group_end,
    source_rows,
    scale_k,
    phase,
    rows: cutlass.Constexpr,
    scales: cutlass.Constexpr,
    cta_group: cutlass.Constexpr = 1,
):
    """TMA-load linear source SF, then publish an MMA 128x4 atom per stage.

    A 16-byte SF window serves narrow K tiles too. Each four-row transfer uses
    a 128-byte-aligned scratch slot (64 payload bytes). Scratch is producer-private
    and reused only after all lanes have read it. The MMA stage barrier accounts
    for a release arrival after packing, separately from the TMA scratch bytes.
    """
    lane = cute.arch.lane_idx()
    with cute.arch.elect_one():
        nvvm.mbarrier_arrive_expect_tx(load_bar, rows * 16)
    for r in cutlass.range(rows // 4, unroll_full=True):
        with cute.arch.elect_one():
            row = row_begin + r * 4
            r0 = moe_gather_row(token_index, row, group_end, source_rows)
            r1 = moe_gather_row(token_index, row + 1, group_end, source_rows)
            r2 = moe_gather_row(token_index, row + 2, group_end, source_rows)
            r3 = moe_gather_row(token_index, row + 3, group_end, source_rows)
            tma_gather4(scratch.subview(r * 128), desc, (scale_k // 16) * 16, r0, r1, r2, r3, load_bar)
    while not nvvm.mbarrier_try_wait_parity(load_bar, phase, time_limit=10_000_000):
        pass
    src_words = cutlass.inttoptr(scratch.data_ptr().toint(), 3, cutlass.Uint32)
    dst_words = cutlass.inttoptr(dst.data_ptr().toint(), 3, cutlass.Uint32)
    for i in cutlass.range(rows * (scales // 4) // 32, unroll_full=True):
        word = lane + i * 32
        row = word // (scales // 4)
        kw = word % (scales // 4)
        packed = (row // 128) * (128 * (scales // 4)) + kw * 128 + (row % 32) * 4 + (row % 128) // 32
        (dst_words + packed).store((src_words + (row // 4) * 32 + (row % 4) * 4 + (scale_k % 16) // 4 + kw).load())
    cute.arch.fence_view_async_shared()
    nvvm.bar_warp_sync(0xFFFFFFFF)
    with cute.arch.elect_one():
        if cutlass.const_expr(cta_group == 2):
            leader = cute.arch.block_idx_in_cluster() & ~1
            nvvm.mbarrier_arrive(nvvm.mapa(ready_bar, leader), scope=nvvm.MemScope.CLUSTER)
        else:
            nvvm.mbarrier_arrive(ready_bar)


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
