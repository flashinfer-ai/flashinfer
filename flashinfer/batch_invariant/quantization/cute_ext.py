"""The only floating-point primitives an RL-ready CuTe-DSL kernel may use.

Each one lowers to an NVVM intrinsic with an explicit ``.rn`` rounding mode
(``add.rn.f32`` etc.). Unlike ``a + b`` on ``Float32`` (which lowers to ``arith``
ops that the NVPTX backend is free to contract into FMAs), these are opaque to the
compiler: no contraction, no reassociation, no flush-to-zero.

The class mirrors the interface of :class:`flashinfer.batch_invariant.term.TermAlgebra` (symbolic) and the
functions in :mod:`flashinfer.batch_invariant.fp_interface.exact` (exact CPU), so schedule code can be written once
against an ``fp`` object and run under all three backends.
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
        return cute.math.add(a, b, rounding=RN)

    @staticmethod
    def mul(a, b):
        return cute.math.mul(a, b, rounding=RN)

    @staticmethod
    def fma(a, b, c):
        return cute.math.fma(a, b, c, rounding=RN)

    @staticmethod
    def div(a, b):
        return cute.math.div(a, b, rounding=RN)

    @staticmethod
    def sqrt(a):
        return cute.math.sqrt(a, rounding=RN)

    @staticmethod
    def fmax(a, b):
        """``nvvm.fmax`` (PTX ``max.f32``, FMNMX): exact, no rounding; maximumNumber NaN rule; on
        sm_107 ``max(+0, -0) = max(-0, +0) = +0`` (bitwise commutative on non-NaN inputs)."""
        return cute.arch.fmax(a, b)

    @staticmethod
    def fmaxc(a, b):
        """The attention class's commutative ``max.f32`` join: the same instruction as ``fmax`` (the
        symbolic algebra normalises the operand order; the CPU model uses the probed zero rule)."""
        return cute.arch.fmax(a, b)

    # -- packed pairs (sm_100+ ``f32x2`` instructions): two independent ``.rn`` ops in one instruction, bit for bit the
    #    scalar ops above -- a packing placement, never a different function
    @staticmethod
    def fma2(a, b, c):
        """``fma.rn.f32x2``: ``(fma(a0, b0, c0), fma(a1, b1, c1))``."""
        return cute.arch.fma_packed_f32x2(a, b, c, rnd="rn")

    @staticmethod
    def add2(a, b):
        """``add.rn.f32x2``: ``(add(a0, b0), add(a1, b1))``."""
        return cute.arch.add_packed_f32x2(a, b, rnd="rn")

    @staticmethod
    def mul2(a, b):
        """``mul.rn.f32x2``: ``(mul(a0, b0), mul(a1, b1))``."""
        return cute.arch.mul_packed_f32x2(a, b, rnd="rn")

    @staticmethod
    def ex2(a):
        """``ex2.approx.ftz.f32``, pinned as inline asm: the attention spec's opaque exponential."""
        return ex2_approx_ftz(a)

    @staticmethod
    def lg2(a):
        """``lg2.approx.ftz.f32``, pinned as inline asm: the MLA class's opaque base-2 logarithm (the LSE member)."""
        return lg2_approx_ftz(a)

    # -- exact (rounding-free) primitives, added for the MoE family -------------
    @staticmethod
    def neg(a):
        """``neg.f32``: exact; folding a negation into a neighbouring op cannot change bits."""
        return cute.math.neg(a)

    @staticmethod
    def fmin(a, b):
        """``min.f32`` (no ftz, no NaN propagation flag): exact."""
        return cute.arch.fmin(a, b)

    @staticmethod
    def cvt_out(a, fmt_name: str):
        """``cvt.rn.<fmt>.f32`` (``arith.truncf``, RNE); the identity for ``f32``."""
        from ..fp_interface.cute import get_cutlass_dtype

        return a if fmt_name == "f32" else a.to(get_cutlass_dtype(fmt_name))

    # -- exact primitives added for the NVFP4 quantization family ---------------
    @staticmethod
    def fabs(a):
        """``abs.f32`` (``nvvm.fabs`` without ftz): clears the sign bit, exact."""
        from cutlass._mlir.dialects import nvvm

        return Float32(nvvm.fabs(Float32(a).ir_value(), ftz=False))

    # ``fmaxc`` is the attention class's commutative max above (same ``max.f32``); the quantizer's
    # amax trees use it on ``fabs(x) >= +0`` operands.

    @staticmethod
    def nzor(a, b):
        """``a`` if ``a != 0`` else ``b``: ``setp.ne.f32`` + ``selp.f32`` (``arith.select`` on the exact
        comparison; no arithmetic)."""
        from cutlass._mlir.dialects import arith

        a, b = Float32(a), Float32(b)
        cond = (a != Float32(0.0)).ir_value()
        return Float32(arith.select(cond, a.ir_value(), b.ir_value()))

    @staticmethod
    def exp2i(k):
        """``2^k`` for an integer-valued f32 ``k`` in ``[-126, 127]``: the exponent field is built
        from the integer (``cvt.rzi`` is exact on integer values) and moved into an f32 register
        with ``mov.b32``; no floating-point rounding is involved."""
        from cutlass import Int32

        n = k.to(Int32)
        return bitcast_i32_to_f32((n + Int32(127)) << 23)


def bitcast_i32_to_f32(bits):
    """``mov.b32 f, r``: reinterpret an Int32 as a Float32 (no conversion)."""
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T
    from cutlass import Int32

    res = llvm.inline_asm(
        T.f32(),
        [Int32(bits).ir_value()],
        "mov.b32 $0, $1;",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(res)


@dsl_user_op
def ex2_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """The one ``exp2`` an RL-ready kernel may use: PTX ``ex2.approx.ftz.f32`` (MUFU.EX2), deterministic
    per architecture; ``ex2(+0) = ex2(-0) = 1``, ``ex2(-inf) = +0``, results below ``2^-126`` flush to
    ``+0`` (probed on sm_107, 2026-10-27; same bits as ``cute.math.exp2(fastmath=True)``)."""
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
    """The one ``log2`` an RL-ready kernel may use: PTX ``lg2.approx.ftz.f32`` (MUFU.LG2), deterministic per architecture;
    the MLA class's LSE member ``add(mul(m, c), lg2(l))`` (``rlk/attention/mla/spec.py``). Like ``ex2`` it has no CPU model: a
    device helper tabulates its values for the exact evaluator."""
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
