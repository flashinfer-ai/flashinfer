# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Routed BF16/FP16 FC1 using FP32 FMA, with a fused activation."""

import cutlass
from cutlass import cute
from flashinfer.cute_dsl.fp4_common import cvt_e4m3x4_to_f32x4, cvt_f32_to_e4m3
from flashinfer.fused_moe.backends.frost_low_latency.fma.activations import (
    apply_activation,
)
from flashinfer.fused_moe.backends.frost_low_latency.fma.fp4_weight import (
    scaled_values,
    weight_scale,
)

# Overridden by the generated host specialization before compilation.
_ACTIVATION = "swiglu"
_GATED = True


@cute.jit
def _quad(word, dtype: cutlass.Constexpr):
    if cutlass.const_expr(dtype == cutlass.Float8E4M3FN):
        return cvt_e4m3x4_to_f32x4(word)
    return (
        word.to(cutlass.Uint16).bitcast(dtype).to(cutlass.Float32),
        (word >> 16).to(cutlass.Uint16).bitcast(dtype).to(cutlass.Float32),
        (word >> 32).to(cutlass.Uint16).bitcast(dtype).to(cutlass.Float32),
        (word >> 48).to(cutlass.Uint16).bitcast(dtype).to(cutlass.Float32),
    )


@cute.jit
def _load_values(
    pointer, offset, dtype: cutlass.Constexpr, values_per_lane: cutlass.Constexpr
):
    first = _quad(pointer[offset], dtype)
    if cutlass.const_expr(values_per_lane == 4):
        return first
    second = _quad(pointer[offset + 1], dtype)
    return (
        first[0],
        first[1],
        first[2],
        first[3],
        second[0],
        second[1],
        second[2],
        second[3],
    )


@cute.jit
def _reduce_values(values, column, values_per_lane: cutlass.Constexpr):
    if cutlass.const_expr(values_per_lane == 4):
        return cute.arch.warp_reduction_sum(
            values[0, column]
            + values[1, column]
            + values[2, column]
            + values[3, column]
        )
    return cute.arch.warp_reduction_sum(
        (
            (values[0, column] + values[1, column])
            + (values[2, column] + values[3, column])
        )
        + (
            (values[4, column] + values[5, column])
            + (values[6, column] + values[7, column])
        )
    )


@cute.jit
def _single_route_body(
    x: cute.Tensor,
    weights: cute.Tensor,
    ids: cute.Tensor,
    mid: cute.Tensor,
    outputs: cutlass.Constexpr,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr = None,
    dequant=None,
    mid_scale=None,
    block_scales=None,
    native_fp4: cutlass.Constexpr = False,
    scale_block_size: cutlass.Constexpr = 16,
    integer_weights: cutlass.Constexpr = False,
    values_per_lane: cutlass.Constexpr = 4,
):
    bid, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    h = cute.size(x, mode=[1])
    i = cute.size(mid, mode=[1])
    k = cute.size(ids, mode=[1])
    tiles = i // (8 * outputs)
    route = bid // tiles
    row = (bid % tiles) * 8 * outputs + warp * outputs
    # Widen before forming the expert stride in 64-bit weight words.
    expert = ids[route // k, route % k].to(cutlass.Int64)
    up = cute.make_rmem_tensor((values_per_lane, outputs), cutlass.Float32)
    branches = 2 if gated else 1
    gate = cute.make_rmem_tensor((values_per_lane, outputs), cutlass.Float32)
    up.fill(0.0)
    gate.fill(0.0)
    if expert >= 0 and expert < cute.size(weights, mode=[0]):
        word_type = (
            cutlass.Uint32 if x.element_type == cutlass.Float8E4M3FN else cutlass.Uint64
        )
        xp = cute.recast_ptr(x.iterator, dtype=word_type)
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
        # Adjacent values per lane give coalesced word loads.
        # Accumulate in FP32; each format's rounding boundary is applied below.
        for group in range(cute.ceil_div(h, 32 * values_per_lane)):
            col = group * 32 + lane
            if cutlass.const_expr(values_per_lane == 4) or col < h // values_per_lane:
                xv = _load_values(
                    xp,
                    (route // k) * (h // 4) + col * (values_per_lane // 4),
                    x.element_type,
                    values_per_lane,
                )
                for o in cutlass.range_constexpr(outputs):
                    offset = (expert * branches * i + row + o) * (
                        h // values_per_lane
                    ) + col
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
                        uv = scaled_values(
                            wp[offset],
                            scale,
                            native_fp4,
                            scale_block_size,
                            integer_weights,
                            values_per_lane,
                        )
                        if cutlass.const_expr(gated):
                            gate_scale = weight_scale(
                                block_scales[
                                    expert,
                                    row + o + i,
                                    col // (scale_block_size // values_per_lane),
                                ],
                                scale_block_size,
                                integer_weights,
                            )
                            gv = scaled_values(
                                wp[offset + i * (h // values_per_lane)],
                                gate_scale,
                                native_fp4,
                                scale_block_size,
                                integer_weights,
                                values_per_lane,
                            )
                    else:
                        uv = _quad(wp[offset], weights.element_type)
                        if cutlass.const_expr(gated):
                            gv = _quad(
                                wp[offset + i * (h // values_per_lane)],
                                weights.element_type,
                            )
                    for c in cutlass.range_constexpr(values_per_lane):
                        up[c, o] = up[c, o] + xv[c] * uv[c].to(cutlass.Float32)
                        if cutlass.const_expr(gated):
                            gate[c, o] = gate[c, o] + xv[c] * gv[c].to(cutlass.Float32)
    for o in cutlass.range_constexpr(outputs):
        u = _reduce_values(up, o, values_per_lane)
        g = cutlass.Float32(0.0)
        if cutlass.const_expr(gated):
            g = _reduce_values(gate, o, values_per_lane)
        if lane == 0:
            if cutlass.const_expr(weights.element_type == cutlass.Uint8):
                if expert >= 0 and expert < cute.size(weights, mode=[0]):
                    u = u * dequant[expert, 0]
                    if cutlass.const_expr(gated):
                        g = g * dequant[expert, 1]
                    # MXINT4 retains FP32 preactivation; FP4 formats round
                    # to the activation dtype before applying the nonlinearity.
                    if cutlass.const_expr(not integer_weights):
                        u = u.to(x.element_type).to(cutlass.Float32)
                        g = g.to(x.element_type).to(cutlass.Float32)
            if cutlass.const_expr(weights.element_type == cutlass.Float8E4M3FN):
                if expert >= 0 and expert < cute.size(weights, mode=[0]):
                    u = u * dequant[expert]
                    g = g * dequant[expert]
                    # Quantized FP8 retains its BF16 FC1 boundary. Weight-only
                    # inputs round to their activation dtype before nonlinearity.
                    if cutlass.const_expr(x.element_type == cutlass.Float16):
                        u = u.to(cutlass.Float16).to(cutlass.Float32)
                        g = g.to(cutlass.Float16).to(cutlass.Float32)
                    else:
                        u = u.to(cutlass.BFloat16).to(cutlass.Float32)
                        g = g.to(cutlass.BFloat16).to(cutlass.Float32)
            value = apply_activation(u, g, activation, parameters=parameters)
            if cutlass.const_expr(mid.element_type == cutlass.Float8E4M3FN):
                # Preserve the checkpoint's static intermediate multiplier.
                value = (
                    cvt_f32_to_e4m3(value * mid_scale[0])
                    .to(cutlass.Uint8)
                    .bitcast(cutlass.Float8E4M3FN)
                )
            mid[route, row + o] = value.to(mid.element_type)


@cute.kernel
def _kernel(
    x: cute.Tensor,
    weights: cute.Tensor,
    ids: cute.Tensor,
    mid: cute.Tensor,
    outputs: cutlass.Constexpr,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr = None,
    dequant=None,
    mid_scale=None,
    block_scales=None,
    native_fp4: cutlass.Constexpr = False,
    scale_block_size: cutlass.Constexpr = 16,
    integer_weights: cutlass.Constexpr = False,
    values_per_lane: cutlass.Constexpr = 4,
):
    _single_route_body(
        x,
        weights,
        ids,
        mid,
        outputs,
        activation,
        gated,
        parameters,
        dequant,
        mid_scale,
        block_scales,
        native_fp4,
        scale_block_size,
        integer_weights,
        values_per_lane,
    )


@cute.jit
def _launch(x, weights, ids, mid, stream):
    outputs = 2 if cute.size(x, mode=[0]) == 1 else 4
    _kernel(x, weights, ids, mid, outputs, _ACTIVATION, _GATED).launch(
        grid=(cute.size(mid) // (8 * outputs), 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


# Activation constants can contain scientific notation such as e+38. Keep
# those values in the compilation key, but not in the NVVM symbol identifier.
_kernel.set_name_prefix(
    "flashinfer_bf16_fma_fc1", keep_mangled_name=False, remove_cutlass_symbol=True
)
