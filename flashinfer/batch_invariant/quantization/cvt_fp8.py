"""The declared fp8 conversion primitive, shared by the kernels that quantize to e4m3 / e5m2:
``cvt.rn.satfinite.{e4m3,e5m2}x2.f32`` (RNE, saturating to the largest finite value; assumption A2 of
docs/requirements.md; exact CPU model ``flashinfer.batch_invariant.fp_interface.exact.to_fp8``). The row-norm kernel (``rlk/rmsnorm/kernel.py``)
carries the same inline-PTX primitive; it is kept there unchanged so that this file's existence does not
touch the row-norm cache closure (consolidating the two is a lead's refactor: it recompiles the row norms)."""

from __future__ import annotations

import cutlass
from cutlass import Float32, Uint16, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def cvt_f32x2_to_fp8x2(
    a: Float32, b: Float32, kind: cutlass.Constexpr[str], *, loc=None, ip=None
) -> Uint16:
    """``cvt.rn.satfinite.{e4m3,e5m2}x2.f32``: a -> low byte, b -> high byte."""
    res = llvm.inline_asm(
        T.i16(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        f"cvt.rn.satfinite.{kind}x2.f32 $0, $2, $1;",
        "=h,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Uint16(res)


@dsl_user_op
def cvt_f32_to_e4m3(a: Float32, *, loc=None, ip=None):
    """``cvt.rn.satfinite.e4m3x2.f32`` of one value. Returns ``(byte, value)``: the e4m3 byte (in a ``Uint32``) and the
    byte's value widened back to f32 exactly (``cvt.rn.f16x2.e4m3x2`` then ``cvt.f32.f16``: every e4m3 value is an f16
    value is an f32 value). The NVFP4 quantizer's block-scale conversion, shared with the MoE epilogues that quantize the
    intermediate in place."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [Float32(a).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 pair, lo, hi;
            .reg .b32 h2;
            .reg .f32 zero;
            mov.f32 zero, 0f00000000;
            cvt.rn.satfinite.e4m3x2.f32 pair, zero, $2;
            cvt.rn.f16x2.e4m3x2 h2, pair;
            mov.b32 {lo, hi}, h2;
            cvt.f32.f16 $1, lo;
            cvt.u32.u16 $0, pair;
        }
        """,
        "=r,=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_f32x8_to_e2m1x8(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Four ``cvt.rn.satfinite.e2m1x2.f32``: element ``2j`` in the low nibble of byte ``j``, the four bytes packed
    little-endian into one 32-bit word (byte 0 = elements 0, 1). Each value's conversion is independent of its pairing."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v).ir_value(loc=loc, ip=ip)
                for v in (v0, v1, v2, v3, v4, v5, v6, v7)
            ],
            """
            {
                .reg .b8 b0, b1, b2, b3;
                cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;
                cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;
                cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;
                cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;
                mov.b32 $0, {b0, b1, b2, b3};
            }
            """,
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
