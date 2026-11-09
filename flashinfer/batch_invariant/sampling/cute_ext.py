# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The only floating-point primitives the batch-invariant sampling kernels may use.

Each rounded one lowers to an NVVM intrinsic with an explicit ``.rn`` rounding mode (``add.rn.f32`` etc.). Unlike
``a + b`` on ``Float32`` (which lowers to ``arith`` ops that the NVPTX backend is free to contract into FMAs), these are
opaque to the compiler: no contraction, no reassociation, no flush-to-zero. The two transcendental primitives are the
declared opaque MUFU instructions, pinned as inline PTX so no other approximation can be substituted.

The class mirrors the interface of :class:`flashinfer.batch_invariant.sampling.sterm.TermAlgebra` (symbolic), so the
specification's chains are written once and run under both backends.
"""

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Float32
from cutlass._mlir.dialects import llvm
from cutlass.cute.math import RoundingMode
from cutlass.cutlass_dsl import T, dsl_user_op

RN = RoundingMode.NEAREST_EVEN


class CuteF32:
    @staticmethod
    def const(value: float):
        from ..fp_interface import exact as fp32

        return Float32(fp32.to_f32(value))

    @staticmethod
    def add(a, b):
        """``add.rn.f32``."""
        return cute.math.add(a, b, rounding=RN)

    @staticmethod
    def mul(a, b):
        """``mul.rn.f32``."""
        return cute.math.mul(a, b, rounding=RN)

    @staticmethod
    def div(a, b):
        """``div.rn.f32``."""
        return cute.math.div(a, b, rounding=RN)

    @staticmethod
    def fmaxc(a, b):
        """``nvvm.fmax`` (PTX ``max.f32``): exact, no rounding; a NaN operand yields the other operand and zeros of
        opposite sign give ``+0`` whatever the order (bitwise commutative on non-NaN inputs; the symbolic algebra
        normalises the operand order)."""
        return cute.arch.fmax(a, b)

    @staticmethod
    def neg(a):
        """``neg.f32``: exact; folding a negation into a neighbouring op cannot change bits."""
        return cute.math.neg(a)

    @staticmethod
    def ex2(a):
        """``ex2.approx.ftz.f32``, pinned as inline asm: the class's opaque exponential."""
        return ex2_approx_ftz(a)

    @staticmethod
    def lg2(a):
        """``lg2.approx.ftz.f32``, pinned as inline asm: the class's opaque base-2 logarithm."""
        return lg2_approx_ftz(a)

    @staticmethod
    def cvt_out(a, fmt_name: str):
        """``cvt.rn.<fmt>.f32`` (``arith.truncf``, RNE); the identity for ``f32``."""
        from ..fp_interface.cute import get_cutlass_dtype

        return a if fmt_name == "f32" else a.to(get_cutlass_dtype(fmt_name))


@dsl_user_op
def ex2_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """PTX ``ex2.approx.ftz.f32`` (MUFU.EX2), deterministic per architecture; ``ex2(+0) = ex2(-0) = 1``,
    ``ex2(-inf) = +0``, results below ``2^-126`` flush to ``+0``."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def lg2_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """PTX ``lg2.approx.ftz.f32`` (MUFU.LG2), deterministic per architecture. Like ``ex2`` it has no CPU model: the
    device tabulates its values for the exact evaluator (``mufu_device.py``)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
