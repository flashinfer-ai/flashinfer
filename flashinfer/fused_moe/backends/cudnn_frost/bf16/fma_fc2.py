# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""BF16 routed FC2 using FP32 FMA and slot-ordered output combination."""

import cutlass
from cutlass import cute, utils
from cutlass._mlir.dialects import llvm

from flashinfer.fused_moe.backends.cudnn_frost.bf16.fma_fc1 import (
    _bf16x4,
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
):
    bid, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    lane = cute.arch.lane_idx()
    slot = cute.arch.warp_idx()
    h = cute.size(out, mode=[1])
    i = cute.size(mid, mode=[1])
    k = cute.size(ids, mode=[1])
    token = bid // (h // 4)
    row = (bid % (h // 4)) * 4
    route = token * k + slot
    expert = ids[token, slot]
    acc = cute.make_rmem_tensor((4, 4), cutlass.Float32)
    acc.fill(0.0)
    shared = utils.SmemAllocator().allocate_tensor(
        cutlass.Float32, cute.make_layout((4, k)), byte_alignment=16
    )
    if expert >= 0 and expert < cute.size(weights, mode=[0]):
        xp = cute.recast_ptr(mid.iterator, dtype=cutlass.Uint64)
        wp = cute.recast_ptr(weights.iterator, dtype=cutlass.Uint64)
        for group in range(i // 128):
            col = group * 32 + lane
            xv = _bf16x4(xp[route * (i // 4) + col])
            for o in cutlass.range_constexpr(4):
                wv = _bf16x4(wp[(expert * h + row + o) * (i // 4) + col])
                for c in cutlass.range_constexpr(4):
                    acc[c, o] = acc[c, o] + xv[c] * wv[c]
    for o in cutlass.range_constexpr(4):
        value = cute.arch.warp_reduction_sum(
            acc[0, o] + acc[1, o] + acc[2, o] + acc[3, o]
        )
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
def _launch(mid, weights, ids, scores, out, stream):
    _kernel(mid, weights, ids, scores, out).launch(
        grid=(cute.size(out) // 4, 1, 1),
        block=(32 * cute.size(ids, mode=[1]), 1, 1),
        stream=stream,
    )


_kernel.set_name_prefix("flashinfer_bf16_fma_fc2", remove_cutlass_symbol=True)
