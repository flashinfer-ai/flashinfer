# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Pinned MXFP4/MXFP8 device arithmetic, including the UE8M0 scale path."""

from __future__ import annotations

import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass.cutlass_dsl import Float32, Int32, T, dsl_user_op

from .nvfp4_device import PackedF32 as _PackedF32
from .nvfp4_device import ScalarF32 as _ScalarF32


@dsl_user_op
def _ue8m0_roundtrip(value: Float32, *, loc=None, ip=None):
    packed = llvm.inline_asm(
        T.i32(),
        [Float32(value).ir_value(loc=loc, ip=ip)],
        "{\n .reg .b16 sf;\n cvt.rp.satfinite.ue8m0x2.f32 sf, 0f00000000, $1;\n cvt.rn.bf16x2.ue8m0x2 $0, sf;\n}",
        "=r,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    pair = llvm.bitcast(ir.Type.parse("vector<2xbf16>"), packed, loc=loc, ip=ip)
    low = vector.extract(pair, [], [0], loc=loc, ip=ip)
    return Float32(arith.extf(T.f32(), low, loc=loc, ip=ip))


@dsl_user_op
def _cvt_ue8m0(value: Float32, *, loc=None, ip=None):
    packed = Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "{\n .reg .b16 sf;\n cvt.rp.satfinite.ue8m0x2.f32 sf, 0f00000000, $1;\n cvt.u32.u16 $0, sf;\n}",
            "=r,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )
    # Generic FP32.to(UE8M0) does not encode this non-IEEE type reliably.
    return cutlass.Uint8(packed).bitcast(cutlass.Float8E8M0FNU)


class ScalarF32(_ScalarF32):
    @staticmethod
    def ue8m0_roundtrip(value):
        return _ue8m0_roundtrip(value)

    @staticmethod
    def cvt_out(value, fmt):
        if fmt == "ue8m0":
            return _cvt_ue8m0(value)
        return _ScalarF32.cvt_out(value, fmt)


class PackedF32(_PackedF32):
    @staticmethod
    def ue8m0_roundtrip(value):
        return (
            ScalarF32.ue8m0_roundtrip(value[0]),
            ScalarF32.ue8m0_roundtrip(value[1]),
        )

    @staticmethod
    def cvt_out(value, fmt):
        return (ScalarF32.cvt_out(value[0], fmt), ScalarF32.cvt_out(value[1], fmt))
