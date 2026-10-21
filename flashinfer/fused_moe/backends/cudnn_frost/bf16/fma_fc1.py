# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Routed BF16 FC1 using FP32 FMA, with a fused activation."""

import cutlass
from cutlass import cute
from flashinfer.fused_moe.backends.cudnn_frost.fma_activations import (
    apply_activation,
)

# Overridden by the generated host specialization before compilation.
_ACTIVATION = "swiglu"
_GATED = True


@cute.jit
def _bf16x4(word):
    return (
        word.to(cutlass.Uint16).bitcast(cutlass.BFloat16).to(cutlass.Float32),
        (word >> 16).to(cutlass.Uint16).bitcast(cutlass.BFloat16).to(cutlass.Float32),
        (word >> 32).to(cutlass.Uint16).bitcast(cutlass.BFloat16).to(cutlass.Float32),
        (word >> 48).to(cutlass.Uint16).bitcast(cutlass.BFloat16).to(cutlass.Float32),
    )


@cute.kernel
def _kernel(
    x: cute.Tensor,
    weights: cute.Tensor,
    ids: cute.Tensor,
    mid: cute.Tensor,
    outputs: cutlass.Constexpr,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
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
    expert = ids[route // k, route % k]
    up = cute.make_rmem_tensor((4, outputs), cutlass.Float32)
    branches = 2 if gated else 1
    gate = cute.make_rmem_tensor((4, outputs), cutlass.Float32)
    up.fill(0.0)
    gate.fill(0.0)
    if expert >= 0 and expert < cute.size(weights, mode=[0]):
        xp = cute.recast_ptr(x.iterator, dtype=cutlass.Uint64)
        wp = cute.recast_ptr(weights.iterator, dtype=cutlass.Uint64)
        # Four adjacent BF16 values per lane give coalesced 64-bit loads.
        # Accumulate in FP32: long BF16 dot products must not round per FMA.
        for group in range(h // 128):
            col = group * 32 + lane
            xv = _bf16x4(xp[(route // k) * (h // 4) + col])
            for o in cutlass.range_constexpr(outputs):
                offset = (expert * branches * i + row + o) * (h // 4) + col
                uv = _bf16x4(wp[offset])
                if cutlass.const_expr(gated):
                    gv = _bf16x4(wp[offset + i * (h // 4)])
                for c in cutlass.range_constexpr(4):
                    up[c, o] = up[c, o] + xv[c] * uv[c]
                    if cutlass.const_expr(gated):
                        gate[c, o] = gate[c, o] + xv[c] * gv[c]
    for o in cutlass.range_constexpr(outputs):
        u = cute.arch.warp_reduction_sum(up[0, o] + up[1, o] + up[2, o] + up[3, o])
        g = cutlass.Float32(0.0)
        if cutlass.const_expr(gated):
            g = cute.arch.warp_reduction_sum(
                gate[0, o] + gate[1, o] + gate[2, o] + gate[3, o]
            )
        if lane == 0:
            value = apply_activation(u, g, activation)
            mid[route, row + o] = value.to(cutlass.BFloat16)


@cute.jit
def _launch(x, weights, ids, mid, stream):
    outputs = 2 if cute.size(x, mode=[0]) == 1 else 4
    _kernel(x, weights, ids, mid, outputs, _ACTIVATION, _GATED).launch(
        grid=(cute.size(mid) // (8 * outputs), 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


_kernel.set_name_prefix("flashinfer_bf16_fma_fc1", remove_cutlass_symbol=True)
