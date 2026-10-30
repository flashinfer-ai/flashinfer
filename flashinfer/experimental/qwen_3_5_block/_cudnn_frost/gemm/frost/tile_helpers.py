# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Tile-level helpers shared by the SM100 and SM120 kernel templates."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cutlass._mlir_helpers.vector as _cvec
from cutlass._mlir.dialects import llvm


@cute.jit
def l2_swizzle_tile(raw_m, raw_n, nt_m, nt_n, swizzle_w, identity=False):
    """N-direction super-block rasterization of the (m, n) cgrp-tile coord, for L2 reuse."""
    if cutlass.const_expr(identity):
        return raw_m, raw_n
    log_m, log_n = raw_m, raw_n
    if swizzle_w != 1:
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


@cute.jit
def cluster_split_k_reduce(
    partial, mailbox, ready, free, iteration, k_rank, root_rank, plane_ctas: cutlass.Constexpr, slices: cutlass.Constexpr, epilogue_threads: cutlass.Constexpr
):
    """Bulk-copy one fragment per peer to reusable slots on its K-rank-zero CTA."""
    tid = cute.arch.thread_idx()[0]
    warp = cute.arch.warp_idx()
    elected = nvvm.elect_sync()
    values = partial.to_elements()
    chunk_elems = epilogue_threads * len(values)
    phase = iteration & 1
    result = partial
    if k_rank != 0:
        staging = mailbox.subview((slices - 1) * chunk_elems)
        for q in cutlass.range_constexpr(len(values) // 4):
            staging.data_ptr(tid * len(values) + q * 4).store(partial[q * 4 : q * 4 + 4], alignment=16)
        cute.arch.fence_proxy("async.shared", space="cta")
        nvvm.barrier_cta_sync(barrier_id=CLUSTER_K_REDUCE_BARRIER_ID, thread_count=epilogue_threads)
        if warp == 0:
            remote_ready = nvvm.mapa(ready, root_rank)
            if elected:
                nvvm.mbarrier_arrive_expect_tx(remote_ready, chunk_elems * 4, scope=nvvm.MemScope.CLUSTER)
            if elected:
                nvvm.cp_async_bulk_shared_cluster_shared_cta(
                    nvvm.mapa(mailbox.subview((k_rank - 1) * chunk_elems), root_rank), staging, remote_ready, chunk_elems * 4
                )
        while not nvvm.mbarrier_try_wait_parity(free, phase, time_limit=10_000_000, scope=nvvm.MBarrierScope.CLUSTER):
            pass
    else:
        while not nvvm.mbarrier_try_wait_parity(ready, phase, time_limit=10_000_000, scope=nvvm.MBarrierScope.CLUSTER):
            pass
        summed = list(values)
        for peer in cutlass.range_constexpr(1, slices):
            for q in cutlass.range_constexpr(len(values) // 4):
                incoming = mailbox.load((peer - 1) * chunk_elems + tid * len(values) + q * 4, vector_size=4, alignment=16)
                for e in cutlass.range_constexpr(4):
                    summed[q * 4 + e] = summed[q * 4 + e] + incoming[e]
        result = _cvec.Vector.from_elements(tuple(summed), cutlass.Float32)
        nvvm.barrier_cta_sync(barrier_id=CLUSTER_K_REDUCE_BARRIER_ID, thread_count=epilogue_threads)
        if warp == 0:
            for peer in cutlass.range_constexpr(1, slices):
                if elected:
                    nvvm.mbarrier_arrive(nvvm.mapa(free, root_rank + peer * plane_ctas), scope=nvvm.MemScope.CLUSTER)
    return result


@cute.jit
def extract_block_scale_sfb(stage, n_start, tile_n: cutlass.Constexpr, k_atoms: cutlass.Constexpr, stage_bytes: cutlass.Constexpr):
    """Extract a tcgen05 N sub-tile in place from one or two F8_128x4 blocks."""
    words = cutlass.Array(base=stage.data_ptr(), shape=stage_bytes // 4, dtype=cutlass.Uint32)
    lane = cute.arch.lane_idx()
    for atom in cutlass.range_constexpr(k_atoms):
        values = []
        for col in cutlass.range_constexpr((tile_n + 31) // 32):
            row = n_start + cutlass.min(lane + col * 32, tile_n - 1)
            src = (row // 128) * (128 * k_atoms) + atom * 128 + (row % 32) * 4 + (row % 128) // 32
            values.append(words.data_ptr(src).load())
        nvvm.bar_warp_sync(0xFFFFFFFF)
        for col in cutlass.range_constexpr((tile_n + 31) // 32):
            words.data_ptr(atom * 128 + lane * 4 + col).store(values[col])
    cute.arch.fence_proxy("async.shared", space="cta")
    nvvm.bar_warp_sync(0xFFFFFFFF)


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
    """At most 16 bytes of adjacent output elements, with scalar alignment ABI."""
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
def moe_combine_stage(
    staging,
    values,
    output,
    token_index,
    token_ks,
    scores,
    row,
    group_end,
    tile_col,
    col,
    columns,
    output_row_stride,
    top_k: cutlass.Constexpr,
    tile_width: cutlass.Constexpr,
    width: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
):
    """Stage weighted row fragments, or atomically reduce unsupported tails."""
    if row < group_end:
        token = token_index[row]
        slot = token_ks[row]
        if (token >= 0) & (token < scores.shape[0]) & (slot >= 0) & (slot < top_k):
            weighted = values * cutlass.full_like(values, scores[token, slot])
            row_base = output + cutlass.Int64(token) * output_row_stride
            if ((row_base + tile_col).toint() % 16 == 0) & (tile_col + tile_width <= columns):
                staging.store(weighted.to(dtype), alignment=16)
            else:
                destination = row_base + cutlass.Int64(col)
                elements: cutlass.Constexpr = 128 // dtype.width
                for start in cutlass.range_constexpr(0, width, elements):
                    if col + start + elements <= columns:
                        moe_combine_add_vector(destination + start, weighted[start : start + elements], dtype, elements)
                    else:
                        for element in cutlass.range_constexpr(elements):
                            if col + start + element < columns:
                                moe_combine_add(destination + start + element, weighted[start + element], dtype)


@cute.jit
def moe_combine_bulk(
    staging,
    output,
    token_index,
    token_ks,
    scores,
    row,
    group_end,
    col,
    columns,
    output_row_stride,
    top_k: cutlass.Constexpr,
    width: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
):
    """Reduce a complete staged feature tile, then release this thread's row."""
    if row < group_end:
        token = token_index[row]
        slot = token_ks[row]
        if (token >= 0) & (token < scores.shape[0]) & (slot >= 0) & (slot < top_k):
            destination = output + cutlass.Int64(token) * output_row_stride + cutlass.Int64(col)
            if (destination.toint() % 16 == 0) & (col + width <= columns):
                cute.arch.fence_view_async_shared()
                if cutlass.const_expr(dtype == cutlass.Float32):
                    instruction = "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;"
                elif cutlass.const_expr(dtype == cutlass.BFloat16):
                    instruction = "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [$0], [$1], $2;"
                else:
                    instruction = "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [$0], [$1], $2;"
                llvm.inline_asm(
                    None,
                    [destination.toint().ir_value(), cutlass.Int32(staging.toint()).ir_value(), cutlass.Int32(width * dtype.width // 8).ir_value()],
                    instruction,
                    "l,r,r,~{memory}",
                    has_side_effects=True,
                )
                nvvm.cp_async_bulk_commit_group()
                nvvm.cp_async_bulk_wait_group(0, read=True)


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
    """TMA-load linear source SF, then publish an MMA 128x4 atom per stage."""
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
    for i in cutlass.range((rows * (scales // 4) + 31) // 32, unroll_full=True):
        word = lane + i * 32
        if word < rows * (scales // 4):
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
