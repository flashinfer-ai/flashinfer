# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routed block-FP8 FMA with FP32 preactivation and dynamic E4M3 intermediate."""

import cutlass
from cutlass import cute, utils
from flashinfer.fused_moe.backends.frost_low_latency.fma.bf16.fma_fc1 import _quad
from flashinfer.fused_moe.backends.frost_low_latency.fma.bf16.fma_fc2 import _mul
from flashinfer.fused_moe.backends.frost_low_latency.fma.activations import (
    apply_activation,
)
from flashinfer.cute_dsl.fp4_common import cvt_f32_to_e4m3, fdiv_rn


@cute.kernel
def fc1(
    x,
    xs,
    w,
    ws,
    ids,
    raw,
    columns: cutlass.Constexpr,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
):
    bid, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    h = cute.size(x, mode=[1])
    i = cute.size(raw, mode=[1])
    branches = 2 if gated else 1
    k = cute.size(ids, mode=[1])
    tiles = i // (8 * columns)
    route = bid // tiles
    row = (bid % tiles) * 8 * columns + warp * columns
    expert = ids[route // k, route % k].to(cutlass.Int64)
    uacc = cute.make_rmem_tensor(columns, cutlass.Float32)
    gacc = cute.make_rmem_tensor(columns, cutlass.Float32)
    uacc.fill(0.0)
    gacc.fill(0.0)
    if expert >= 0 and expert < cute.size(w, mode=[0]):
        xp = cute.recast_ptr(x.iterator, dtype=cutlass.Uint32)
        wp = cute.recast_ptr(w.iterator, dtype=cutlass.Uint32)
        for group in range(h // 128):
            col = group * 32 + lane
            xv = _quad(xp[(route // k) * (h // 4) + col], x.element_type)
            for o in cutlass.range_constexpr(columns):
                offset = (expert * branches * i + row + o) * (h // 4) + col
                uv = _quad(wp[offset], w.element_type)
                if cutlass.const_expr(gated):
                    gv = _quad(wp[offset + i * (h // 4)], w.element_type)
                u = cutlass.Float32(0.0)
                g = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(4):
                    u = u + xv[c] * uv[c]
                    if cutlass.const_expr(gated):
                        g = g + xv[c] * gv[c]
                us = xs[group, route // k] * ws[expert, (row + o) // 128, group]
                uacc[o] = uacc[o] + u * us
                if cutlass.const_expr(gated):
                    gs = xs[group, route // k] * ws[expert, (row + o + i) // 128, group]
                    gacc[o] = gacc[o] + g * gs
    for o in cutlass.range_constexpr(columns):
        u = cute.arch.warp_reduction_sum(uacc[o])
        g = cutlass.Float32(0.0)
        if cutlass.const_expr(gated):
            g = cute.arch.warp_reduction_sum(gacc[o])
        if lane == 0:
            raw[route, row + o] = apply_activation(
                u, g, activation, parameters=parameters
            )


@cute.kernel
def quantize(raw, mid, ms):
    bid, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    groups = cute.size(raw, mode=[1]) // 128
    tile = bid * 8 + warp
    route = tile // groups
    group = tile % groups
    if route < cute.size(raw, mode=[0]):
        values = cute.make_rmem_tensor(4, cutlass.Float32)
        maximum = cutlass.Float32(0.0)
        for c in cutlass.range_constexpr(4):
            values[c] = raw[route, group * 128 + lane * 4 + c].to(cutlass.Float32)
            if abs(values[c]) > maximum:
                maximum = abs(values[c])
        maximum = cute.arch.warp_reduction_max(maximum)
        if maximum < 1.0e-4:
            maximum = cutlass.Float32(1.0e-4)
        scale = _mul(maximum, cutlass.Float32(1.0 / 448.0))
        for c in cutlass.range_constexpr(4):
            mid[route, group * 128 + lane * 4 + c] = (
                cvt_f32_to_e4m3(fdiv_rn(values[c], scale))
                .to(cutlass.Uint8)
                .bitcast(cutlass.Float8E4M3FN)
            )
        if lane == 0:
            ms[group, route] = scale


@cute.kernel
def fc2(
    mid,
    ms,
    w,
    ws,
    ids,
    scores,
    out,
    columns: cutlass.Constexpr,
    narrow: cutlass.Constexpr,
    finalize: cutlass.Constexpr,
):
    bid, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    lane = cute.arch.lane_idx()
    slot = cute.arch.warp_idx()
    h = cute.size(out, mode=[1])
    i = cute.size(mid, mode=[1])
    k = cute.size(ids, mode=[1])
    token = bid // (h // columns)
    row = (bid % (h // columns)) * columns
    route = token * k + slot
    expert = ids[token, slot].to(cutlass.Int64)
    acc = cute.make_rmem_tensor(columns, cutlass.Float32)
    acc.fill(0.0)
    if cutlass.const_expr(finalize):
        shared = utils.SmemAllocator().allocate_tensor(
            cutlass.Float32, cute.make_layout((columns, k)), byte_alignment=16
        )
    if expert >= 0 and expert < cute.size(w, mode=[0]):
        xp = cute.recast_ptr(mid.iterator, dtype=cutlass.Uint32)
        wp = cute.recast_ptr(w.iterator, dtype=cutlass.Uint32)
        for group in range(i // 128):
            col = group * 32 + lane
            xv = _quad(xp[route * (i // 4) + col], mid.element_type)
            for o in cutlass.range_constexpr(columns):
                wv = _quad(wp[(expert * h + row + o) * (i // 4) + col], w.element_type)
                part = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(4):
                    part = part + xv[c] * wv[c]
                scale = ms[group, route] * ws[expert, (row + o) // 128, group]
                acc[o] = acc[o] + part * scale
    for o in cutlass.range_constexpr(columns):
        value = cute.arch.warp_reduction_sum(acc[o])
        if lane == 0:
            value = value.to(cutlass.BFloat16).to(cutlass.Float32)
            if cutlass.const_expr(finalize):
                weighted = cutlass.Float32(0.0)
                if expert >= 0 and expert < cute.size(w, mode=[0]):
                    score = scores[token, slot].to(cutlass.Float32)
                    if cutlass.const_expr(narrow):
                        score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                    weighted = _mul(value, score)
                shared[o, slot] = weighted
            else:
                out[route, row + o] = value.to(cutlass.BFloat16)
    if cutlass.const_expr(finalize):
        cute.arch.sync_threads()
        if tid < columns:
            total = cutlass.Float32(0.0)
            for s in cutlass.range_constexpr(k):
                total = total + shared[tid, s]
            out[token, row + tid] = total.to(cutlass.BFloat16)


@cute.jit
def launch(
    x,
    qx,
    xs,
    w1,
    s1,
    w2,
    s2,
    ids,
    scores,
    raw,
    mid,
    ms,
    out,
    first: cutlass.Constexpr,
    second: cutlass.Constexpr,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
    narrow: cutlass.Constexpr,
    finalize: cutlass.Constexpr,
    stream,
):
    if cutlass.const_expr(x.element_type == cutlass.BFloat16):
        quantize(x, qx, xs).launch(
            grid=((cute.size(x) // 128 + 7) // 8, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )
    fc1(qx, xs, w1, s1, ids, raw, first, activation, gated, parameters).launch(
        grid=(cute.size(raw) // (8 * first), 1, 1), block=(256, 1, 1), stream=stream
    )
    quantize(raw, mid, ms).launch(
        grid=((cute.size(raw) // 128 + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream
    )
    fc2(mid, ms, w2, s2, ids, scores, out, second, narrow, finalize).launch(
        grid=(cute.size(ids, mode=[0]) * cute.size(w2, mode=[1]) // second, 1, 1),
        block=(32 * cute.size(ids, mode=[1]), 1, 1),
        stream=stream,
    )
