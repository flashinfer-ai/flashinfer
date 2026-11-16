# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Vectorized owner-local reduction, called after last-contributor acquire.

The original compute warps reduce two independent hidden fragments at once.
Source-generated plans encode top-k slots, not physical pool rows: the latter
must still be resolved from this replay's owner-local route map. No readiness,
completion, or epoch ownership is transferred to these arithmetic helpers.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, Int32, Int64, Float32, dsl_user_op

from .config import RANK_COMBINE_PLAN_COUNT_SHIFT, RANK_COMBINE_PLAN_SLOT_BITS


@dsl_user_op
def _load128(addr: Int64, *, loc=None, ip=None):
    value = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l", has_side_effects=True,
        asm_dialect=0, loc=loc, ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), value, [i])) for i in range(4))


@dsl_user_op
def _store128(addr: Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [x.ir_value(loc=loc, ip=ip) for x in (addr, a, b, c, d)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@cute.jit
def _load_fragment(route_words, offset, h, dst):
    words = cute.recast_tensor(dst, Int32)
    for part in cutlass.range_constexpr(2):
        # 16 lanes x 8 contiguous BF16 values per vector transaction.
        data = _load128((route_words + offset + (h + part * 128) // 4).toint())
        for i in cutlass.range_constexpr(4):
            words[part * 4 + i] = data[i]


@cute.jit
def _store_fragment(peer_words, h, values):
    packed = cute.make_rmem_tensor((16,), cutlass.BFloat16)
    words = cute.recast_tensor(packed, Int32)
    for i in cutlass.range_constexpr(16):
        packed[i] = values[i].to(cutlass.BFloat16)
    for part in cutlass.range_constexpr(2):
        _store128(
            (peer_words + (h + part * 128) // 4).toint(),
            words[part * 4], words[part * 4 + 1],
            words[part * 4 + 2], words[part * 4 + 3],
        )


@cute.jit
def _fixed_reduce(route_words, peer_words, first, second, chunk_base, lane,
                  chunk_hidden: cutlass.Constexpr, count: cutlass.Constexpr):
    source = cute.make_rmem_tensor((16, count, 2), cutlass.BFloat16)
    values = cute.make_rmem_tensor((16, 2), Float32)
    for offset in cutlass.range(0, chunk_hidden, 512):
        # Issue both independent fragments before the first reduce/store.
        for b in cutlass.range_constexpr(2):
            h = chunk_base + offset + b * 256 + (lane % 16) * 8
            for c in cutlass.range_constexpr(count):
                src_offset = first
                if cutlass.const_expr(c == 1):
                    src_offset = second
                _load_fragment(route_words, src_offset, h, source[None, c, b])
        values.fill(Float32(0.0))
        for b in cutlass.range_constexpr(2):
            for c in cutlass.range_constexpr(count):
                for i in cutlass.range_constexpr(16):
                    values[i, b] += Float32(source[i, c, b])
            h = chunk_base + offset + b * 256 + (lane % 16) * 8
            _store_fragment(peer_words, h, values[None, b])


@cute.jit
def _dense_reduce(route_output, peer_row, mapping, key, lane, chunk_base,
                  hidden: cutlass.Constexpr, pool_capacity: cutlass.Constexpr,
                  topk: cutlass.Constexpr, chunk_hidden: cutlass.Constexpr):
    # Keep original top-k order and zero additions for count >= 3.
    offset_dtype: cutlass.Constexpr = (
        Int32 if pool_capacity * (hidden // 4) < (1 << 31) else Int64
    )
    offsets = cute.make_rmem_tensor((topk,), offset_dtype)
    for slot in cutlass.range_constexpr(topk):
        offsets[slot] = offset_dtype(mapping[key, slot]) * offset_dtype(hidden // 4)
    route_words = cute.recast_ptr(route_output.iterator, dtype=Int64)
    peer_words = cute.recast_ptr(peer_row.iterator, dtype=Int64)
    values = cute.make_rmem_tensor((16, 2), Float32)
    source = cute.make_rmem_tensor((16, 2, 2), cutlass.BFloat16)
    for offset in cutlass.range(0, chunk_hidden, 512):
        values.fill(Float32(0.0))
        for pair in cutlass.range_constexpr((topk + 1) // 2):
            source.fill(cutlass.BFloat16(0.0))
            for b in cutlass.range_constexpr(2):
                h = chunk_base + offset + b * 256 + (lane % 16) * 8
                for item in cutlass.range_constexpr(2):
                    if cutlass.const_expr(pair * 2 + item < topk):
                        if offsets[pair * 2 + item] >= 0:
                            _load_fragment(route_words, offsets[pair * 2 + item],
                                           h, source[None, item, b])
            for b in cutlass.range_constexpr(2):
                for item in cutlass.range_constexpr(2):
                    if cutlass.const_expr(pair * 2 + item < topk):
                        for i in cutlass.range_constexpr(16):
                            values[i, b] += Float32(source[i, item, b])
        for b in cutlass.range_constexpr(2):
            h = chunk_base + offset + b * 256 + (lane % 16) * 8
            _store_fragment(peer_words, h, values[None, b])


@cute.jit
def reduce_owner_partial(route_output, peer_row, mapping, plans, key, lane,
                         chunk_base, hidden: cutlass.Constexpr,
                         pool_capacity: cutlass.Constexpr, topk: cutlass.Constexpr,
                         chunk_hidden: cutlass.Constexpr):
    # Called only for a last-contributor winner, hence count >= 1. Source
    # count release/admission and the existing completion acquire order all
    # plan/map/payload reads. Bases/rows are 16-byte aligned; host layout
    # selection guarantees complete 512-channel steps.
    plan = plans[key]
    count = plan >> RANK_COMBINE_PLAN_COUNT_SHIFT
    offset_dtype: cutlass.Constexpr = (
        Int32 if pool_capacity * (hidden // 4) < (1 << 31) else Int64
    )
    if count <= 2:
        first = offset_dtype(mapping[key, plan & 7]) * offset_dtype(hidden // 4)
        second = first
        if count == 2:
            second = offset_dtype(mapping[key, (plan >> RANK_COMBINE_PLAN_SLOT_BITS) & 7]) * offset_dtype(hidden // 4)
        route_words = cute.recast_ptr(route_output.iterator, dtype=Int64)
        peer_words = cute.recast_ptr(peer_row.iterator, dtype=Int64)
        if count == 1:
            _fixed_reduce(route_words, peer_words, first, second, chunk_base, lane,
                          chunk_hidden, 1)
        else:
            _fixed_reduce(route_words, peer_words, first, second, chunk_base, lane,
                          chunk_hidden, 2)
    else:
        _dense_reduce(route_output, peer_row, mapping, key, lane, chunk_base,
                      hidden, pool_capacity, topk, chunk_hidden)
