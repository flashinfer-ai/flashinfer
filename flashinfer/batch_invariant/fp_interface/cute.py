# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The only floating-point primitives a batch-invariant CuTe-DSL kernel may use.

Each one lowers to an NVVM intrinsic with an explicit ``.rn`` rounding mode (``add.rn.f32``, ``mul.rn.f32``). Unlike
``a + b`` on ``Float32`` (which lowers to ``arith`` ops that the NVPTX backend is free to contract into
FMAs), these are opaque to the compiler: no contraction, no reassociation, no flush-to-zero. The GEMM's
tensor-core step is not here: it is the kernel's opaque ``cute.gemm`` on the accumulator, below which
nothing is reordered.

The class mirrors the interface of :class:`flashinfer.batch_invariant.fp_interface.term.TermAlgebra` (symbolic), so
the specification's chains can be written once and run under both backends.
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
        from .term import to_f32

        return Float32(to_f32(value))

    @staticmethod
    def add(a, b):
        """``add.rn.f32``."""
        return cute.math.add(a, b, rounding=RN)

    @staticmethod
    def mul(a, b):
        """``mul.rn.f32``."""
        return cute.math.mul(a, b, rounding=RN)

    @staticmethod
    def fma(a, b, c):
        """``fma.rn.f32``."""
        return cute.math.fma(a, b, c, rounding=RN)

    @staticmethod
    def div(a, b):
        """``div.rn.f32``."""
        return cute.math.div(a, b, rounding=RN)

    @staticmethod
    def fmaxc(a, b):
        """``nvvm.fmax`` (PTX ``max.f32``, FMNMX): exact, no rounding, the maximumNumber NaN rule; the attention class's
        commutative join (the symbolic algebra normalises the operand order)."""
        return cute.arch.fmax(a, b)

    # -- packed pairs (sm_100+ ``f32x2`` instructions): two independent ``.rn`` ops in one instruction, bit for bit the
    #    scalar ops above -- a packing placement, never a different function
    @staticmethod
    def fma2(a, b, c):
        """``fma.rn.f32x2``: ``(fma(a0, b0, c0), fma(a1, b1, c1))``."""
        return cute.arch.fma_packed_f32x2(a, b, c, rnd="rn")

    @staticmethod
    def mul2(a, b):
        """``mul.rn.f32x2``: ``(mul(a0, b0), mul(a1, b1))``."""
        return cute.arch.mul_packed_f32x2(a, b, rnd="rn")

    @staticmethod
    def add2(a, b):
        """``add.rn.f32x2``: ``(add(a0, b0), add(a1, b1))``."""
        return cute.arch.add_packed_f32x2(a, b, rnd="rn")

    @staticmethod
    def ex2(a):
        """``ex2.approx.ftz.f32``, pinned as inline asm: the attention class's opaque exponential."""
        return ex2_approx_ftz(a)

    @staticmethod
    def cvt_out(a, fmt_name: str):
        """``cvt.rn.<fmt>.f32`` (``arith.truncf``, RNE); the identity for ``f32``."""
        return a if fmt_name == "f32" else a.to(get_cutlass_dtype(fmt_name))

    @staticmethod
    def widen(a):
        """A bf16 / f16 value as binary32 (``arith.extf``): exact, a bit movement rather than a rounding step; how a
        narrower input (the bias) enters the fp32 chain."""
        return a.to(Float32)


@dsl_user_op
def ex2_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """The one ``exp2`` a batch-invariant kernel may use: PTX ``ex2.approx.ftz.f32`` (MUFU.EX2), deterministic per
    architecture; ``ex2(+0) = ex2(-0) = 1``, ``ex2(-inf) = +0``, results below ``2^-126`` flush to ``+0``."""
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
def fmax3(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """PTX ``max.f32 d, a, b, c`` (sm_100+, one FMNMX3): exact, no rounding -- the maximum of three is the same value
    whatever the grouping (``max`` is order-free on the class's non-NaN inputs), so a 3-ary tree over a row gives the
    2-ary tree's bits."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "max.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# Name <-> CuTe dtype table shared by the kernels (kept out of the API modules so that compiled-kernel cache keys do not depend
# on them).
def get_cutlass_dtype(name: str):
    import cutlass

    return {
        "bf16": cutlass.BFloat16,
        "f16": cutlass.Float16,
        "f32": cutlass.Float32,
        "e4m3": cutlass.Float8E4M3FN,
        "e5m2": cutlass.Float8E5M2,
    }[name]
