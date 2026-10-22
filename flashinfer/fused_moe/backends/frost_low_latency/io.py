# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scale packing and ordered route reduction for the public MoE contract."""

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass import cute
from .quantization import _sf_index


@cute.jit
def _fma(a, b, c):
    return cutlass.Float32(
        llvm.inline_asm(
            a.ir_value().type,
            [a.ir_value(), b.ir_value(), c.ir_value()],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
        )
    )


@cute.kernel
def pack_scale_kernel(
    linear: cute.Tensor,
    packed: cute.Tensor,
    mid: cute.Tensor,
    groups: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    token, _, _ = cute.arch.block_idx()
    for index in range(tid, 128 * groups, 256):
        # Each token is a batch of N=1. Initialize the TMA padding as well.
        packed[token * 128 * groups + index] = cutlass.Uint8(0)
    if token == 0:
        for index in range(tid, mid.shape[0], 256):
            mid[index] = cutlass.Uint8(0)
    cute.arch.barrier()
    for group in range(tid, groups, 256):
        packed[token * 128 * groups + _sf_index(0, group, groups)] = linear[
            token, group
        ]


@cute.jit
def pack_scale(linear, packed, mid, stream):
    pack_scale_kernel(linear, packed, mid, linear.shape[1]).launch(
        grid=(linear.shape[0], 1, 1), block=(256, 1, 1), stream=stream
    )


@cute.kernel
def combine_kernel(
    routes: cute.Tensor,
    weights: cute.Tensor,
    output: cute.Tensor,
    top_k: cutlass.Constexpr,
    narrow_weights: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    tile, token, _ = cute.arch.block_idx()
    col = tile * 256 + tid
    if col < output.shape[1]:
        value = cutlass.Float32(0)
        for slot in cutlass.range_constexpr(top_k):
            weight = weights[token, slot]
            if cutlass.const_expr(narrow_weights):
                weight = weight.to(cutlass.BFloat16)
            value = _fma(
                routes[token * top_k + slot, col].to(cutlass.Float32),
                weight.to(cutlass.Float32),
                value,
            )
        output[token, col] = value.to(cutlass.BFloat16)


@cute.jit
def combine(routes, weights, output, narrow_weights: cutlass.Constexpr, stream):
    combine_kernel(routes, weights, output, weights.shape[1], narrow_weights).launch(
        grid=(cute.ceil_div(output.shape[1], 256), output.shape[0], 1),
        block=(256, 1, 1),
        stream=stream,
    )
