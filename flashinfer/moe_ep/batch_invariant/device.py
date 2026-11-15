# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Pinned device interpretation of the BF16 MegaMoE arithmetic specification.

The approximate instructions are opaque, architecture-scoped operations. In
particular, MegaMoE uses ``ex2.approx.ftz.f32`` and
``rcp.approx.ftz.f32``; neither is a correctly rounded exponential or division.
Packed operations apply the same scalar function independently to two lanes.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, T, dsl_user_op


@dsl_user_op
def _constant(bits: str, *, loc=None, ip=None) -> Float32:
    # The DSL's Python-float constant interning can merge +0.0 and -0.0.
    # Materialize the canonical binary32 bits rather than a numeric literal.
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [],
            "mov.b32 $0, 0x" + bits + ";",
            "=f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _unary(instruction: str, a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            instruction + " $0, $1;",
            "=f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _binary(instruction: str, a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            instruction + " $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


class ScalarF32:
    """Scalar FP32 operations with explicit rounding and denormal behavior."""

    @staticmethod
    def const(value):
        from .spec import f32_bits

        return _constant(f32_bits(value))

    @staticmethod
    def add(a, b):
        return _binary("add.rn.f32", a, b)

    @staticmethod
    def mul(a, b):
        return _binary("mul.rn.f32", a, b)

    @staticmethod
    def fma(a, b, c):
        return _fma(a, b, c)

    @staticmethod
    def ex2(a):
        return _unary("ex2.approx.ftz.f32", a)

    @staticmethod
    def rcp(a):
        return _unary("rcp.approx.ftz.f32", a)

    @staticmethod
    def fmin_nan(a, b):
        return _binary("min.NaN.f32", a, b)

    @staticmethod
    def fmax_nan(a, b):
        return _binary("max.NaN.f32", a, b)

    @staticmethod
    def cvt_out(a, fmt):
        if fmt == "bf16":
            return a.to(cutlass.BFloat16)
        if fmt == "f32":
            return Float32(a)
        raise ValueError(f"unsupported MegaMoE output format: {fmt}")


class PackedF32:
    """Two independent scalar chains using explicit RN, non-FTZ packed ALU ops."""

    @staticmethod
    def const(value):
        scalar = ScalarF32.const(value)
        return (scalar, scalar)

    @staticmethod
    def add(a, b):
        return cute.arch.add_packed_f32x2(a, b, rnd="rn", ftz=False)

    @staticmethod
    def mul(a, b):
        return cute.arch.mul_packed_f32x2(a, b, rnd="rn", ftz=False)

    @staticmethod
    def fma(a, b, c):
        return cute.arch.fma_packed_f32x2(a, b, c, rnd="rn", ftz=False)

    @staticmethod
    def ex2(a):
        return (ScalarF32.ex2(a[0]), ScalarF32.ex2(a[1]))

    @staticmethod
    def rcp(a):
        return (ScalarF32.rcp(a[0]), ScalarF32.rcp(a[1]))

    @staticmethod
    def fmin_nan(a, b):
        return (ScalarF32.fmin_nan(a[0], b[0]), ScalarF32.fmin_nan(a[1], b[1]))

    @staticmethod
    def fmax_nan(a, b):
        return (ScalarF32.fmax_nan(a[0], b[0]), ScalarF32.fmax_nan(a[1], b[1]))

    @staticmethod
    def cvt_out(a, fmt):
        return (ScalarF32.cvt_out(a[0], fmt), ScalarF32.cvt_out(a[1], fmt))
