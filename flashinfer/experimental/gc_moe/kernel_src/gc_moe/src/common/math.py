"""Scalar conversions, quantization scales and warp reductions."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint16, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def preweight_routed_sf(value: Float32, secondary: Float32, *, loc=None, ip=None) -> Float32:
    """Preserve both FP32 RN boundaries: (value *64), then *secondary."""
    product = llvm.inline_asm(
        T.f32(), [value.ir_value(loc=loc, ip=ip), secondary.ir_value(loc=loc, ip=ip)],
        '{ .reg .f32 original; mul.rn.f32 original, $1, 0f42800000; '
        'mul.rn.f32 $0, original, $2; }', '=f,f,f',
        has_side_effects=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(product)


@dsl_user_op
def round_bf16_f32(value: Float32, *, loc=None, ip=None) -> Float32:
    rounded = llvm.inline_asm(
        T.f32(), [value.ir_value(loc=loc, ip=ip)],
        '{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }',
        '=f,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(rounded)


@dsl_user_op
def float_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    bits = llvm.inline_asm(
        T.i32(), [value.ir_value(loc=loc, ip=ip)],
        'mov.b32 $0, $1;', '=r,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(bits)


@dsl_user_op
def bits_float(bits: Uint32, *, loc=None, ip=None) -> Float32:
    value = llvm.inline_asm(
        T.f32(), [bits.ir_value(loc=loc, ip=ip)],
        'mov.b32 $0, $1;', '=f,r', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(value)


@dsl_user_op
def fast_rcp(value: Float32, *, loc=None, ip=None) -> Float32:
    reciprocal = llvm.inline_asm(
        T.f32(), [value.ir_value(loc=loc, ip=ip)],
        'rcp.approx.ftz.f32 $0, $1;', '=f,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(reciprocal)


@dsl_user_op
def abs_f32(value: Float32, *, loc=None, ip=None) -> Float32:
    absolute = llvm.inline_asm(
        T.f32(), [value.ir_value(loc=loc, ip=ip)],
        'abs.f32 $0, $1;', '=f,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(absolute)


@dsl_user_op
def f32_to_bf16_bits(value: Float32, *, loc=None, ip=None) -> Uint16:
    bits = llvm.inline_asm(
        T.i16(), [value.ir_value(loc=loc, ip=ip)],
        'cvt.rn.bf16.f32 $0, $1;', '=h,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint16(bits)


@dsl_user_op
def f32x2_to_bf16x2_bits(low: Float32, high: Float32,
                         *, loc=None, ip=None) -> Uint32:
    """Reference ``__float22bfloat162_rn``: low/high FP32 to BF16x2."""
    bits = llvm.inline_asm(
        T.i32(), [high.ir_value(loc=loc, ip=ip),
                  low.ir_value(loc=loc, ip=ip)],
        'cvt.rn.bf16x2.f32 $0, $1, $2;', '=r,f,f',
        has_side_effects=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(bits)


@dsl_user_op
def bf16x2_min(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    value = llvm.inline_asm(
        T.i32(), [a.ir_value(loc=loc, ip=ip),
                  b.ir_value(loc=loc, ip=ip)],
        'min.bf16x2 $0, $1, $2;', '=r,r,r',
        has_side_effects=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(value)


@dsl_user_op
def bf16x2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    value = llvm.inline_asm(
        T.i32(), [a.ir_value(loc=loc, ip=ip),
                  b.ir_value(loc=loc, ip=ip)],
        'max.bf16x2 $0, $1, $2;', '=r,r,r',
        has_side_effects=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(value)


@dsl_user_op
def f32x2_to_e4m3x2_bits(low: Float32, high: Float32,
                         *, loc=None, ip=None) -> Uint16:
    """CUDA FP8x2 constructor: low byte from `low`, high from `high`."""
    pair = llvm.inline_asm(
        T.i16(), [low.ir_value(loc=loc, ip=ip),
                  high.ir_value(loc=loc, ip=ip)],
        '{ cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1; }',
        '=h,f,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint16(pair)


@dsl_user_op
def fast_exp_neg_f32(sigmoid_arg: Float32, *, loc=None, ip=None) -> Float32:
    """Evaluate the fast negative-gate exponential for the W1 epilogue."""
    result = llvm.inline_asm(
        T.f32(), [sigmoid_arg.ir_value(loc=loc, ip=ip)],
        '{ .reg .f32 exponent; '
        'mul.f32 exponent, $1, 0fBFB8AA3B; '
        'ex2.approx.f32 $0, exponent; }',
        '=f,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(result)


@dsl_user_op
def bf16x2_high_f32_bits(packed: Uint32, *, loc=None, ip=None) -> Uint32:
    """Place the high BF16 lane in FP32 bits with one byte permutation."""
    bits = llvm.inline_asm(
        T.i32(), [packed.ir_value(loc=loc, ip=ip)],
        'prmt.b32 $0, $1, 0, 0x3254;', '=r,r',
        has_side_effects=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(bits)


@cute.jit
def ue8m0_fp8_scale_exp(amax: Float32) -> Uint32:
    """Compute the UE8M0 scale exponent for FP8 E4M3 quantization."""
    bits = float_bits(amax)
    # 0x600000 is 1.75 in FP32 mantissa units; min exponent 105 keeps
    # E4M3 values finite. Preserve the uint32 addition/shift order.
    rounded_exp = (bits + Uint32(0x7fffff) - Uint32(0x600000)) >> 23
    rounded_exp = rounded_exp if rounded_exp > Uint32(113) else Uint32(113)
    return rounded_exp - Uint32(8)


@cute.jit
def ue8m0_fp8_inv(sf_exp: Uint32) -> Float32:
    return bits_float((Uint32(254) - sf_exp) << 23)


@cute.jit
def pack_bf16_pair(first: Float32, second: Float32) -> Uint32:
    """Reference cast_into_bf16_and_pack: paired RN conversion."""
    return f32x2_to_bf16x2_bits(first, second)


@cute.jit
def fmax_f32(a: Float32, b: Float32) -> Float32:
    # Source fmaxf semantics, including the reduction's NaN selection.
    return Float32(max_f32(a, b))


@dsl_user_op
def max_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    result = llvm.inline_asm(
        T.f32(), [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        'max.f32 $0, $1, $2;', '=f,f,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(result)


@cute.jit
def warp_group4_max(value: Float32) -> Float32:
    # Reduce within each lane%4 group.
    for offset in cutlass.range_constexpr(3):
        value = fmax_f32(value, cute.arch.shuffle_sync_bfly(
            value, 4 << offset))
    return value


@cute.jit
def warp_inclusive_sum(value: Uint32, lane: Uint32) -> Uint32:
    """Full-warp inclusive sum for one expert-count wave."""
    for shift in cutlass.range_constexpr(5):
        offset = 1 << shift
        neighbor = cute.arch.shuffle_sync_up(value, offset, mask_and_clamp=0)
        if lane >= Uint32(offset):
            value += neighbor
    return value
