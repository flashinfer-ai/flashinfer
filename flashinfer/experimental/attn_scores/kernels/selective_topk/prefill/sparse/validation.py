# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Warp-local candidate validation before sparse KV gathers."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def load_four(ptr, offset, *, loc=None, ip=None):
    address = cutlass.Uint64(ptr.toint()) + cutlass.Uint64(offset) * 4
    result = llvm.inline_asm(
        llvm.StructType.get_literal([cutlass.Int32.mlir_type] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.global.v4.s32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Int32(
            llvm.extractvalue(cutlass.Int32.mlir_type, result, [i], loc=loc, ip=ip)
        )
        for i in range(4)
    )


@cute.jit
def validate_candidate_row(blocks, row, end, keys):
    lane = cute.arch.lane_idx()
    count = min(2048, (max(0, end) + 7) // 8)
    bad = cutlass.Int32(end < 0 or end > keys)
    tail = cutlass.Int32(-1)
    if count == 2048:
        for base in cutlass.range(0, 2048, 128, unroll=4):
            a, b, c, d = load_four(blocks.iterator, row * 2048 + base + lane * 4)
            previous = cute.arch.shuffle_sync_up(d, 1, mask=-1, mask_and_clamp=0)
            previous = tail if lane == 0 else previous
            tail = cute.arch.shuffle_sync(d, 31)
            bad = bad | cutlass.Int32(a <= previous or b <= a or c <= b or d <= c)
        bad = bad | cutlass.Int32(tail != (end - 1) // 8)
    elif end > 0:
        for base in cutlass.range(0, 2048, 128, unroll=4):
            col = base + lane * 4
            a, b, c, d = load_four(blocks.iterator, row * 2048 + col)
            previous = cute.arch.shuffle_sync_up(d, 1, mask=-1, mask_and_clamp=0)
            previous = tail if lane == 0 else previous
            tail = cute.arch.shuffle_sync(d, 31)
            values = (a, b, c, d)
            prior = (previous, a, b, c)
            for j in cutlass.range_constexpr(4):
                valid = (
                    values[j] > prior[j] if col + j < count else values[j] == 2147483647
                )
                if col + j == count - 1:
                    valid = valid & (values[j] == (end - 1) // 8)
                bad = bad | cutlass.Int32(not valid)
    return cute.arch.warp_redux_sync(bad, "max")
