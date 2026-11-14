# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Pinned interpretation of the quantized-combine scalar arithmetic."""

import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass.cutlass_dsl import Float32, T, dsl_user_op

from .mxfp8_device import ScalarF32 as _ScalarF32


@dsl_user_op
def _widen_ue8m0(value, *, loc=None, ip=None):
    code = cutlass.Int32(value.bitcast(cutlass.Uint8))
    pair = llvm.inline_asm(
        T.i32(),
        [code.ir_value(loc=loc, ip=ip)],
        "{\n .reg .b16 sf;\n cvt.u16.u32 sf, $1;\n cvt.rn.bf16x2.ue8m0x2 $0, sf;\n}",
        "=r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    values = llvm.bitcast(ir.Type.parse("vector<2xbf16>"), pair, loc=loc, ip=ip)
    low = vector.extract(values, [], [0], loc=loc, ip=ip)
    return Float32(arith.extf(T.f32(), low, loc=loc, ip=ip))


class ScalarF32(_ScalarF32):
    @staticmethod
    def widen(value):
        return Float32(value)

    @staticmethod
    def widen_ue8m0(value):
        return _widen_ue8m0(value)
