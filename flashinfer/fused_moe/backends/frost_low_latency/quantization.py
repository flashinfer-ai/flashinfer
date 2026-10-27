# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.

"""MXFP8 activation quantization; round each block scale upward to UE8M0."""

import cutlass
from cutlass import cute, Uint32, Float32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T


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
def _inverse_scale(code):
    bits = (cutlass.Uint32(254) - code) << 23
    if code >= 254:
        bits = bits | cutlass.Uint32(1 << 22)
    return bits.bitcast(cutlass.Float32)


@dsl_user_op
def cvt_f32_to_e4m3(a: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert float32 to E4M3 using native cvt.rn.satfinite.e4m3x2.f32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .f32 zero;
                mov.f32 zero, 0f00000000;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def float_to_ue8m0_fast(value: Float32, *, loc=None, ip=None) -> Uint32:
    """
    Convert float to UE8M0 format using exact IEEE 754 bit manipulation.

    Matches the hardware __nv_cvt_float_to_e8m0(value, __NV_SATFINITE, cudaRoundPosInf):
    - Extract biased exponent from IEEE 754 float
    - If mantissa is nonzero, add 1 (round towards +inf / ceil behavior)
    - Clamp to [0, 254] (255 = NaN in E8M0)
    - Return 0 for zero/negative input
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;
                .reg .u32 bits, exp_biased, mantissa, bump, result;

                setp.le.f32 p_zero, $1, 0f00000000;

                mov.b32 bits, $1;
                shr.b32 exp_biased, bits, 23;
                and.b32 exp_biased, exp_biased, 255;
                and.b32 mantissa, bits, 0x7FFFFF;

                setp.ne.u32 p_has_mant, mantissa, 0;
                selp.u32 bump, 1, 0, p_has_mant;
                setp.eq.u32 p_exp_zero, exp_biased, 0;
                setp.le.u32 p_tiny_sub, mantissa, 0x400000;
                and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;
                @p_tiny_sub mov.u32 bump, 0;
                add.u32 result, exp_biased, bump;

                setp.gt.u32 p_ovf, result, 254;
                selp.u32 result, 254, result, p_ovf;
                selp.u32 $0, 0, result, p_zero;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
