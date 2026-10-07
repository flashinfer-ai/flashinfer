# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Register decoding and E8M0 arithmetic shared by the MX FMA kernels."""

import cutlass
from cutlass import cute
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import dsl_user_op
from flashinfer.cute_dsl.fp4_common import cvt_e4m3_to_f32_via_f16


@cute.jit
def _mul(a: cutlass.Float32, b: cutlass.Float32):
    return cutlass.Float32(
        llvm.inline_asm(
            a.ir_value().type,
            [a.ir_value(), b.ir_value()],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
        )
    )


@cute.jit
def _sf_index(row, col, cols):
    return (
        (row // 128) * 128 * cols
        + (col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + col % 4
    )


@cute.jit
def _fp8x4(word):
    return (
        cvt_e4m3_to_f32_via_f16(word & 255),
        cvt_e4m3_to_f32_via_f16((word >> 8) & 255),
        cvt_e4m3_to_f32_via_f16((word >> 16) & 255),
        cvt_e4m3_to_f32_via_f16(word >> 24),
    )


@dsl_user_op
def _fp4x4(word, *, loc=None, ip=None):
    values = cute.arch.cvt_f4e2m1x4_to_f16x4(
        word.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    return tuple(
        cutlass.Float16(vector.extract(values, [], [j], loc=loc, ip=ip)).to(
            cutlass.Float32
        )
        for j in range(4)
    )


@cute.jit
def _weight_quad(words, block, part, mixed: cutlass.Constexpr):
    if cutlass.const_expr(mixed):
        word = words[block * 4 + part // 2] >> ((part % 2) * 16)
        result = _fp4x4(word.to(cutlass.Uint16))
    else:
        result = _fp8x4(words[block * 8 + part])
    return result


@cute.jit
def _pow2(exponent):
    bits = (exponent + 127).to(cutlass.Uint32) << 23
    if exponent < -126:
        bits = cutlass.Uint32(1) << (exponent + 149)
    return bits.bitcast(cutlass.Float32)


@cute.jit
def _scale_dot(dot, a, b):
    # CuTe's byte loads may sign-extend when cast directly to Int32. E8M0
    # codes are unsigned, including the upper half of the exponent range.
    a = (a.to(cutlass.Uint32) & 255).to(cutlass.Int32)
    b = (b.to(cutlass.Uint32) & 255).to(cutlass.Int32)
    # Combine exponents before multiplying: forming two E8M0 scales' product
    # first can underflow/overflow even when the scaled dot is representable.
    exponent = a + b - 254
    first = exponent
    if exponent < -126:
        first = cutlass.Int32(-126)
    elif exponent > 127:
        first = cutlass.Int32(127)
    result = _mul(_mul(dot, _pow2(first)), _pow2(exponent - first))
    if a == 255 or b == 255:
        result = cutlass.Uint32(0x7FC00000).bitcast(cutlass.Float32)
    return result


@cute.jit
def _inverse_scale(code):
    bits = (cutlass.Uint32(254) - code) << 23
    if code >= 254:
        bits = bits | cutlass.Uint32(1 << 22)
    return bits.bitcast(cutlass.Float32)
