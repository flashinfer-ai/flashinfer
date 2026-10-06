# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Routed MXFP8 FC2 with FP8/FP4 weights and slot-ordered finalization."""

import cutlass
from cutlass import cute, utils
from flashinfer.fused_moe.backends.cudnn_frost.mxfp8.fma_common import (
    _fp8x4,
    _mul,
    _scale_dot,
    _sf_index,
    _weight_quad,
)


@cute.kernel
def _kernel(
    mid: cute.Tensor,
    mid_sf: cute.Tensor,
    weights: cute.Tensor,
    weight_sf: cute.Tensor,
    ids: cute.Tensor,
    scores: cute.Tensor,
    out: cute.Tensor,
):
    bid, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    lane = cute.arch.lane_idx()
    slot = cute.arch.warp_idx()
    h = cute.size(out, mode=[1])
    i = cute.size(mid, mode=[1])
    cols = i // 32
    k = cute.size(ids, mode=[1])
    mixed = weights.element_type == cutlass.Uint8
    token = bid // (h // 4)
    row = (bid % (h // 4)) * 4
    route = token * k + slot
    expert = ids[token, slot]
    xp = cute.recast_ptr(mid.iterator, dtype=cutlass.Uint32)
    wp = cute.recast_ptr(weights.iterator, dtype=cutlass.Uint32)
    ws = cute.recast_ptr(weight_sf.iterator, dtype=cutlass.Uint8)
    accum = cute.make_rmem_tensor((4,), cutlass.Float32)
    accum.fill(0.0)
    shared = utils.SmemAllocator().allocate_tensor(
        cutlass.Float32, cute.make_layout((4, k)), byte_alignment=16
    )
    if expert >= 0 and expert < cute.size(weights, mode=[0]):
        for group in range((cols + 31) // 32):
            col = group * 32 + lane
            if col < cols:
                sx = mid_sf[route, col]
                partial = cute.make_rmem_tensor((4, 4), cutlass.Float32)
                partial.fill(0.0)
                for part in cutlass.range_constexpr(8):
                    xv = _fp8x4(xp[(route * cols + col) * 8 + part])
                    for o in cutlass.range_constexpr(4):
                        wv = _weight_quad(
                            wp, (expert * h + row + o) * cols + col, part, mixed
                        )
                        for c in cutlass.range_constexpr(4):
                            partial[c, o] = partial[c, o] + xv[c] * wv[c]
                for o in cutlass.range_constexpr(4):
                    sw = ws[expert * h * cols + _sf_index(row + o, col, cols)]
                    dot = partial[0, o] + partial[1, o] + partial[2, o] + partial[3, o]
                    accum[o] = accum[o] + _scale_dot(dot, sx, sw)
    for o in cutlass.range_constexpr(4):
        value = cute.arch.warp_reduction_sum(accum[o])
        if lane == 0:
            weighted = cutlass.Float32(0.0)
            if expert >= 0 and expert < cute.size(weights, mode=[0]):
                value = value.to(cutlass.BFloat16).to(cutlass.Float32)
                weighted = _mul(value, scores[token, slot])
            shared[o, slot] = weighted
    cute.arch.sync_threads()
    if tid < 4:
        total = cutlass.Float32(0.0)
        for s in cutlass.range_constexpr(k):
            total = total + shared[tid, s]
        out[token, row + tid] = total.to(cutlass.BFloat16)


@cute.jit
def _launch(mid, mid_sf, weights, weight_sf, ids, scores, out, stream):
    _kernel(mid, mid_sf, weights, weight_sf, ids, scores, out).launch(
        grid=(cute.size(out) // 4, 1, 1),
        block=(32 * cute.size(ids, mode=[1]), 1, 1),
        stream=stream,
    )


_kernel.set_name_prefix("flashinfer_mxfp8_fma_fc2", remove_cutlass_symbol=True)
