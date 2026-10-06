# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Routed MXFP8 FC1 with FP8/FP4 weights, activation and block32 quantization."""

import cutlass
from cutlass import cute, utils
from flashinfer.cute_dsl.fp4_common import cvt_f32_to_e4m3
from flashinfer.quantization.quantization_cute_dsl_utils import float_to_ue8m0_fast
from flashinfer.fused_moe.backends.cudnn_frost.fma_activations import (
    apply_activation,
)
from flashinfer.fused_moe.backends.cudnn_frost.mxfp8.fma_common import (
    _fp8x4,
    _inverse_scale,
    _mul,
    _scale_dot,
    _sf_index,
    _weight_quad,
)

# Overridden by the generated host specialization before compilation.
_ACTIVATION = "swiglu"
_GATED = True


@cute.kernel
def _kernel(
    x: cute.Tensor,
    xsf: cute.Tensor,
    weights: cute.Tensor,
    weight_sf: cute.Tensor,
    ids: cute.Tensor,
    mid: cute.Tensor,
    mid_sf: cute.Tensor,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
):
    bid, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    h = cute.size(x, mode=[1])
    i = cute.size(mid, mode=[1])
    cols = h // 32
    k = cute.size(ids, mode=[1])
    experts = cute.size(weights, mode=[0])
    mixed = weights.element_type == cutlass.Uint8
    route = bid // (i // 32)
    row = (bid % (i // 32)) * 32 + warp * 4
    token = route // k
    expert = ids[token, route % k]
    xp = cute.recast_ptr(x.iterator, dtype=cutlass.Uint32)
    wp = cute.recast_ptr(weights.iterator, dtype=cutlass.Uint32)
    ws = cute.recast_ptr(weight_sf.iterator, dtype=cutlass.Uint8)
    branches = 2 if gated else 1
    accum = cute.make_rmem_tensor((4, branches), cutlass.Float32)
    accum.fill(0.0)
    shared = utils.SmemAllocator().allocate_tensor(
        cutlass.Float32, cute.make_layout(32), byte_alignment=16
    )
    if expert >= 0 and expert < experts:
        for group in range((cols + 31) // 32):
            col = group * 32 + lane
            if col < cols:
                if cutlass.const_expr(
                    cute.size(xsf, mode=[0]) == cute.size(x, mode=[0])
                    and cute.size(xsf, mode=[1]) == cols
                ):
                    sx = xsf[token, col]
                else:
                    sx = xsf[0, _sf_index(token, col, cols)]
                partial = cute.make_rmem_tensor((4, 4, branches), cutlass.Float32)
                partial.fill(0.0)
                for part in cutlass.range_constexpr(8):
                    xv = _fp8x4(xp[(token * cols + col) * 8 + part])
                    for o in cutlass.range_constexpr(4):
                        for branch in cutlass.range_constexpr(branches):
                            block = (
                                expert * branches * i + branch * i + row + o
                            ) * cols + col
                            wv = _weight_quad(wp, block, part, mixed)
                            for c in cutlass.range_constexpr(4):
                                partial[c, o, branch] = (
                                    partial[c, o, branch] + xv[c] * wv[c]
                                )
                for o in cutlass.range_constexpr(4):
                    for branch in cutlass.range_constexpr(branches):
                        sw = ws[
                            (branch * experts + expert) * i * cols
                            + _sf_index(row + o, col, cols)
                        ]
                        dot = (
                            partial[0, o, branch]
                            + partial[1, o, branch]
                            + partial[2, o, branch]
                            + partial[3, o, branch]
                        )
                        accum[o, branch] = accum[o, branch] + _scale_dot(dot, sx, sw)
    for o in cutlass.range_constexpr(4):
        up = cute.arch.warp_reduction_sum(accum[o, 0])
        gate = cutlass.Float32(0.0)
        if cutlass.const_expr(gated):
            gate = cute.arch.warp_reduction_sum(accum[o, 1])
        if lane == 0:
            value = apply_activation(up, gate, activation)
            shared[warp * 4 + o] = value.to(cutlass.BFloat16).to(cutlass.Float32)
    cute.arch.sync_threads()
    # One CTA produces one complete 32-element microscaling block.
    if tid < 32:
        value = shared[tid]
        maximum = cute.arch.warp_reduction_max(abs(value))
        scale = float_to_ue8m0_fast(_mul(maximum, cutlass.Float32(1.0 / 448.0)))
        inverse = _inverse_scale(scale)
        column = (bid % (i // 32)) * 32 + tid
        mid[route, column] = (
            cvt_f32_to_e4m3(_mul(value, inverse))
            .to(cutlass.Uint8)
            .bitcast(cutlass.Float8E4M3FN)
        )
        if tid == 0:
            mid_sf[route, column // 32] = scale.to(cutlass.Uint8)


@cute.jit
def _launch(x, xsf, weights, weight_sf, ids, mid, mid_sf, stream):
    _kernel(x, xsf, weights, weight_sf, ids, mid, mid_sf, _ACTIVATION, _GATED).launch(
        grid=(cute.size(mid) // 32, 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


_kernel.set_name_prefix("flashinfer_mxfp8_fma_fc1", remove_cutlass_symbol=True)
