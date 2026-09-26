# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""FMA routed FC2 with CTA-local, slot-ordered BF16 output combination."""

import cutlass
from cutlass.cute.typing import Float16
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import vector
from cutlass import cute, utils
from cutlass._mlir.dialects import llvm


@cute.jit
def _mul(a: cutlass.Float32, b: cutlass.Float32):
    return cutlass.Float32(
        llvm.inline_asm(
            a.ir_value().type,
            [a.ir_value(), b.ir_value()],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _add(a: cutlass.Float32, b: cutlass.Float32):
    return cutlass.Float32(
        llvm.inline_asm(
            a.ir_value().type,
            [a.ir_value(), b.ir_value()],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _fp4_quad(packed, *, loc=None, ip=None):
    decoded = cute.arch.cvt_f4e2m1x4_to_f16x4(
        packed.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    return cutlass.Vector.from_elements(
        (
            Float16(vector.extract(decoded, [], [0], loc=loc, ip=ip)),
            Float16(vector.extract(decoded, [], [1], loc=loc, ip=ip)),
            Float16(vector.extract(decoded, [], [2], loc=loc, ip=ip)),
            Float16(vector.extract(decoded, [], [3], loc=loc, ip=ip)),
        ),
        cutlass.Float16,
    )


@cute.kernel
def _kernel(
    qmid: cute.Tensor,
    smid: cute.Tensor,
    weights: cute.Tensor,
    weight_sf: cute.Tensor,
    alpha: cute.Tensor,
    ids: cute.Tensor,
    scores: cute.Tensor,
    out: cute.Tensor,
):
    bid, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    H = cute.size(out, mode=[1])
    I = cute.size(weights, mode=[2]) * 2
    C = I // 16
    token = bid // (H // 4)
    row = (bid - token * (H // 4)) * 4
    topk = cute.size(ids, mode=[1])
    slot = warp
    route = token * topk + slot
    expert = cutlass.Int32(-1)
    if slot < topk:
        expert = ids[token, slot]
    shared = utils.SmemAllocator().allocate_tensor(
        cutlass.Float32, cute.make_layout(topk * 4), byte_alignment=16
    )
    acc0 = cutlass.Float32(0.0)
    acc1 = cutlass.Float32(0.0)
    acc2 = cutlass.Float32(0.0)
    acc3 = cutlass.Float32(0.0)
    if expert >= 0 and expert < cute.size(weights, mode=[0]):
        x64 = cute.recast_ptr(qmid.iterator, dtype=cutlass.Uint64)
        w64 = cute.recast_ptr(weights.iterator, dtype=cutlass.Uint64)
        ws = cute.recast_ptr(weight_sf.iterator, dtype=cutlass.Uint8)
        for group in range((C + 31) // 32):
            sc = lane + group * 32
            if sc < C:
                sx = (
                    smid[route * 128 * C + (sc // 4) * 512 + sc % 4]
                    .bitcast(cutlass.Float8E4M3FN)
                    .to(cutlass.Float32)
                )
                sfidx = (
                    expert * H * C
                    + (row // 128) * 128 * C
                    + (sc // 4) * 512
                    + (row % 32) * 16
                    + ((row % 128) // 32) * 4
                    + sc % 4
                )
                xv = x64[route * C + sc]
                sw0 = ws[sfidx + 0].bitcast(cutlass.Float8E4M3FN).to(cutlass.Float32)
                w0 = w64[(expert * H + row + 0) * C + sc]
                g0 = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                sw1 = ws[sfidx + 16].bitcast(cutlass.Float8E4M3FN).to(cutlass.Float32)
                w1 = w64[(expert * H + row + 1) * C + sc]
                g1 = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                sw2 = ws[sfidx + 32].bitcast(cutlass.Float8E4M3FN).to(cutlass.Float32)
                w2 = w64[(expert * H + row + 2) * C + sc]
                g2 = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                sw3 = ws[sfidx + 48].bitcast(cutlass.Float8E4M3FN).to(cutlass.Float32)
                w3 = w64[(expert * H + row + 3) * C + sc]
                g3 = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                for b in range(4):
                    shift = b * 16
                    x = _fp4_quad((xv >> shift).to(cutlass.Uint16))
                    a0 = _fp4_quad((w0 >> shift).to(cutlass.Uint16))
                    g0 = g0 + x * a0
                    a1 = _fp4_quad((w1 >> shift).to(cutlass.Uint16))
                    g1 = g1 + x * a1
                    a2 = _fp4_quad((w2 >> shift).to(cutlass.Uint16))
                    g2 = g2 + x * a2
                    a3 = _fp4_quad((w3 >> shift).to(cutlass.Uint16))
                    g3 = g3 + x * a3
                acc0 += g0.to(cutlass.Float32).reduce("add") * (sx * sw0)
                acc1 += g1.to(cutlass.Float32).reduce("add") * (sx * sw1)
                acc2 += g2.to(cutlass.Float32).reduce("add") * (sx * sw2)
                acc3 += g3.to(cutlass.Float32).reduce("add") * (sx * sw3)
        acc0 = cute.arch.warp_reduction_sum(acc0)
        acc1 = cute.arch.warp_reduction_sum(acc1)
        acc2 = cute.arch.warp_reduction_sum(acc2)
        acc3 = cute.arch.warp_reduction_sum(acc3)
        if lane == 0:
            v0 = (acc0 * alpha[expert]).to(cutlass.BFloat16).to(cutlass.Float32)
            shared[warp * 4 + 0] = _mul(v0, scores[token, slot])
            v1 = (acc1 * alpha[expert]).to(cutlass.BFloat16).to(cutlass.Float32)
            shared[warp * 4 + 1] = _mul(v1, scores[token, slot])
            v2 = (acc2 * alpha[expert]).to(cutlass.BFloat16).to(cutlass.Float32)
            shared[warp * 4 + 2] = _mul(v2, scores[token, slot])
            v3 = (acc3 * alpha[expert]).to(cutlass.BFloat16).to(cutlass.Float32)
            shared[warp * 4 + 3] = _mul(v3, scores[token, slot])
    else:
        if lane == 0:
            shared[warp * 4 + 0] = cutlass.Float32(0.0)
            shared[warp * 4 + 1] = cutlass.Float32(0.0)
            shared[warp * 4 + 2] = cutlass.Float32(0.0)
            shared[warp * 4 + 3] = cutlass.Float32(0.0)
    cute.arch.sync_threads()
    if tid == 0:
        total0 = cutlass.Float32(0.0)
        total1 = cutlass.Float32(0.0)
        total2 = cutlass.Float32(0.0)
        total3 = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(topk):
            total0 = _add(total0, shared[j * 4 + 0])
            total1 = _add(total1, shared[j * 4 + 1])
            total2 = _add(total2, shared[j * 4 + 2])
            total3 = _add(total3, shared[j * 4 + 3])
        out[token, row + 0] = total0.to(cutlass.BFloat16)
        out[token, row + 1] = total1.to(cutlass.BFloat16)
        out[token, row + 2] = total2.to(cutlass.BFloat16)
        out[token, row + 3] = total3.to(cutlass.BFloat16)


@cute.jit
def _launch(qmid, smid, weights, weight_sf, alpha, ids, scores, out, stream):
    _kernel(qmid, smid, weights, weight_sf, alpha, ids, scores, out).launch(
        grid=(cute.size(out) // 4, 1, 1),
        block=(32 * cute.size(ids, mode=[1]), 1, 1),
        stream=stream,
    )


_kernel.set_name_prefix("flashinfer_nvfp4_fma", remove_cutlass_symbol=True)
