# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Device interpretation of the ordered NVFP4 MegaMoE scalar chains."""

from __future__ import annotations

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int32, T, dsl_user_op

from .device import PackedF32 as _PackedF32
from .device import ScalarF32 as _ScalarF32
from .device import _binary, _unary


@dsl_user_op
def _abs_bits(a: Float32, *, loc=None, ip=None) -> Float32:
    bits = Int32(llvm.bitcast(T.i32(), Float32(a).ir_value(loc=loc, ip=ip)))
    return Float32(llvm.bitcast(T.f32(), (bits & Int32(0x7FFFFFFF)).ir_value()))


class ScalarF32(_ScalarF32):
    @staticmethod
    def fmin(a, b):
        return _binary("min.f32", a, b)

    @staticmethod
    def fmax(a, b):
        return _binary("max.f32", a, b)

    @staticmethod
    def neg(a):
        return _unary("neg.f32", a)

    @staticmethod
    def min_xorsign_abs(a, b):
        return _binary("min.xorsign.abs.f32", a, b)

    @staticmethod
    def max_xorsign_abs(a, b):
        return _binary("max.xorsign.abs.f32", a, b)

    @staticmethod
    def abs_bits(a):
        return _abs_bits(a)

    @staticmethod
    def cvt_out(a, fmt):
        if fmt == "e4m3":
            return a.to(cutlass.Float8E4M3FN)
        if fmt == "e2m1":
            raise ValueError(
                "E2M1 device output requires the vector conversion after quantize_block_precast"
            )
        return _ScalarF32.cvt_out(a, fmt)

    @staticmethod
    def widen_e4m3(a):
        return Float32(a)


class PackedF32(_PackedF32):
    @staticmethod
    def fmin(a, b):
        return (ScalarF32.fmin(a[0], b[0]), ScalarF32.fmin(a[1], b[1]))

    @staticmethod
    def fmax(a, b):
        return (ScalarF32.fmax(a[0], b[0]), ScalarF32.fmax(a[1], b[1]))

    @staticmethod
    def neg(a):
        return (ScalarF32.neg(a[0]), ScalarF32.neg(a[1]))

    @staticmethod
    def min_xorsign_abs(a, b):
        return (
            ScalarF32.min_xorsign_abs(a[0], b[0]),
            ScalarF32.min_xorsign_abs(a[1], b[1]),
        )

    @staticmethod
    def max_xorsign_abs(a, b):
        return (
            ScalarF32.max_xorsign_abs(a[0], b[0]),
            ScalarF32.max_xorsign_abs(a[1], b[1]),
        )

    @staticmethod
    def abs_bits(a):
        return (ScalarF32.abs_bits(a[0]), ScalarF32.abs_bits(a[1]))

    @staticmethod
    def cvt_out(a, fmt):
        return (ScalarF32.cvt_out(a[0], fmt), ScalarF32.cvt_out(a[1], fmt))

    @staticmethod
    def widen_e4m3(a):
        return (ScalarF32.widen_e4m3(a[0]), ScalarF32.widen_e4m3(a[1]))
