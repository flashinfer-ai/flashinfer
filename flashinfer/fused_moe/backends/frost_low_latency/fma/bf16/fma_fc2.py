# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""BF16/FP16 routed FC2 using FP32 FMA and slot-ordered output combination."""

import cutlass
from cutlass import cute, utils
from cutlass._mlir.dialects import llvm

from flashinfer.fused_moe.backends.frost_low_latency.fma.bf16.fma_fc1 import (
    _quad,
    _load_values,
    _reduce_values,
)
from flashinfer.fused_moe.backends.frost_low_latency.fma.fp4_weight import (
    scaled_values,
    weight_scale,
)


@cute.jit
def _mul(a: cutlass.Float32, b: cutlass.Float32):
    # Match the standalone finalize's separate multiplication and addition.
    return cutlass.Float32(
        llvm.inline_asm(
            a.ir_value().type,
            [a.ir_value(), b.ir_value()],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
        )
    )


@cute.kernel
def _kernel(
    mid: cute.Tensor,
    weights: cute.Tensor,
    ids: cute.Tensor,
    scores: cute.Tensor,
    out: cute.Tensor,
    narrow: cutlass.Constexpr = False,
    finalize: cutlass.Constexpr = True,
    outputs: cutlass.Constexpr = 4,
    dequant=None,
    block_scales=None,
    native_fp4: cutlass.Constexpr = False,
    scale_block_size: cutlass.Constexpr = 16,
    integer_weights: cutlass.Constexpr = False,
    values_per_lane: cutlass.Constexpr = 4,
):
    bid, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    lane = cute.arch.lane_idx()
    slot = cute.arch.warp_idx()
    h = cute.size(out, mode=[1])
    i = cute.size(mid, mode=[1])
    k = cute.size(ids, mode=[1])
    token = bid // (h // outputs)
    row = (bid % (h // outputs)) * outputs
    route = token * k + slot
    # Large expert tables can exceed signed 32-bit word indexing.
    expert = ids[token, slot].to(cutlass.Int64)
    acc = cute.make_rmem_tensor((values_per_lane, outputs), cutlass.Float32)
    acc.fill(0.0)
    shared = utils.SmemAllocator().allocate_tensor(
        cutlass.Float32, cute.make_layout((outputs, k)), byte_alignment=16
    )
    if expert >= 0 and expert < cute.size(weights, mode=[0]):
        word_type = (
            cutlass.Uint32
            if mid.element_type == cutlass.Float8E4M3FN
            else cutlass.Uint64
        )
        xp = cute.recast_ptr(mid.iterator, dtype=word_type)
        weight_word_type = (
            (cutlass.Uint16 if values_per_lane == 4 else cutlass.Uint32)
            if weights.element_type == cutlass.Uint8
            else (
                cutlass.Uint32
                if weights.element_type == cutlass.Float8E4M3FN
                else cutlass.Uint64
            )
        )
        wp = cute.recast_ptr(weights.iterator, dtype=weight_word_type)
        for group in range(cute.ceil_div(i, 32 * values_per_lane)):
            col = group * 32 + lane
            if cutlass.const_expr(values_per_lane == 4) or col < i // values_per_lane:
                xv = _load_values(
                    xp,
                    route * (i // 4) + col * (values_per_lane // 4),
                    mid.element_type,
                    values_per_lane,
                )
                for o in cutlass.range_constexpr(outputs):
                    word = wp[(expert * h + row + o) * (i // values_per_lane) + col]
                    if cutlass.const_expr(weights.element_type == cutlass.Uint8):
                        scale = weight_scale(
                            block_scales[
                                expert,
                                row + o,
                                col // (scale_block_size // values_per_lane),
                            ],
                            scale_block_size,
                            integer_weights,
                        )
                        wv = scaled_values(
                            word,
                            scale,
                            native_fp4,
                            scale_block_size,
                            integer_weights,
                            values_per_lane,
                        )
                    else:
                        wv = _quad(word, weights.element_type)
                    for c in cutlass.range_constexpr(values_per_lane):
                        acc[c, o] = acc[c, o] + xv[c] * wv[c].to(cutlass.Float32)
    for o in cutlass.range_constexpr(outputs):
        value = _reduce_values(acc, o, values_per_lane)
        if lane == 0:
            if cutlass.const_expr(
                weights.element_type in (cutlass.Float8E4M3FN, cutlass.Uint8)
            ):
                if expert >= 0 and expert < cute.size(weights, mode=[0]):
                    value = _mul(value, dequant[expert])
            if cutlass.const_expr(finalize):
                weighted = cutlass.Float32(0.0)
                if expert >= 0 and expert < cute.size(weights, mode=[0]):
                    value = value.to(out.element_type).to(cutlass.Float32)
                    score = scores[token, slot].to(cutlass.Float32)
                    if cutlass.const_expr(narrow):
                        score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                    weighted = _mul(value, score)
                shared[o, slot] = weighted
            else:
                out[route, row + o] = value.to(out.element_type)
    if cutlass.const_expr(finalize):
        cute.arch.sync_threads()
        if tid < outputs:
            total = cutlass.Float32(0.0)
            for s in cutlass.range_constexpr(k):
                total = total + shared[tid, s]
            out[token, row + tid] = total.to(out.element_type)


@cute.jit
def _launch(mid, weights, ids, scores, out, stream):
    _kernel(mid, weights, ids, scores, out).launch(
        grid=(cute.size(out) // 4, 1, 1),
        block=(32 * cute.size(ids, mode=[1]), 1, 1),
        stream=stream,
    )


_kernel.set_name_prefix("flashinfer_bf16_fma_fc2", remove_cutlass_symbol=True)
