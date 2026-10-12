"""PTX helpers for FP8 activations with prepared E2M1 weights.

Prepared K32 scales are decoded to FP32 and applied after each unscaled MMA.
Weight nibbles are rearranged into native E2M1 containers without quantizing them.
The global scale correction reverses the BF16 preparation multiplier.
"""

from __future__ import annotations

from typing import Tuple

from cutlass import Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


GLOBAL_FP8_FIXUP = 2.0**-119  # g_fp8 = g_prep * 2**-119  (bf16 params_dtype)


@dsl_user_op
def f32x4_to_e4m3x4(v0, v1, v2, v3, *, loc=None, ip=None) -> Uint32:
    """4 x f32 -> 4 E4M3 bytes (RN, satfinite); v0 -> byte 0 ... v3 -> byte 3."""
    from cutlass import Float32

    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
                Float32(v2).ir_value(loc=loc, ip=ip),
                Float32(v3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 lo, hi;
                cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;
                cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;
                mov.b32 $0, {lo, hi};
            }
            """,
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e2m1x8_to_e2m1c_nblk(
    q: Uint32, shift: int = 2, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Exact-weight variant (native .e2m1 B for kind::f8f6f4): spread 8 nibbles (b12x order) to
    two registers of 8-bit containers, nibble at bits [shift+3:shift] (ISA: [5:2] -> shift=2).

    out0 = (n0, n4, n1, n5), out1 = (n2, n6, n3, n7): the same k slots as the E4M3 routes.
    x = prmt(q,0,0x2020) = [B0,B2,B0,B2]; low nibbles -> bytes 0,1, high nibbles -> bytes 2,3.
    """
    shift = int(shift)
    if shift == 2:
        body = """
            prmt.b32 x, $2, 0, 0x2020;
            shl.b32 lo, x, 2;  and.b32 lo, lo, 0x00003C3C;
            shr.b32 hi, x, 2;  and.b32 hi, hi, 0x3C3C0000;
            or.b32 $0, lo, hi;
            prmt.b32 x, $2, 0, 0x3131;
            shl.b32 lo, x, 2;  and.b32 lo, lo, 0x00003C3C;
            shr.b32 hi, x, 2;  and.b32 hi, hi, 0x3C3C0000;
            or.b32 $1, lo, hi;"""
    elif shift == 0:
        body = """
            prmt.b32 x, $2, 0, 0x2020;
            and.b32 lo, x, 0x00000F0F;
            shr.b32 hi, x, 4;  and.b32 hi, hi, 0x0F0F0000;
            or.b32 $0, lo, hi;
            prmt.b32 x, $2, 0, 0x3131;
            and.b32 lo, x, 0x00000F0F;
            shr.b32 hi, x, 4;  and.b32 hi, hi, 0x0F0F0000;
            or.b32 $1, lo, hi;"""
    else:
        raise ValueError(f"unsupported e2m1 container shift {shift}")
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(q).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 x, lo, hi;" + body + " }",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    b0 = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    b1 = llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)
    return Uint32(b0), Uint32(b1)


@dsl_user_op
def e2m1x8_to_e2m1c_nblk6(q: Uint32, *, loc=None, ip=None) -> Tuple[Uint32, Uint32]:
    """bit-identical output to e2m1x8_to_e2m1c_nblk(q, shift=2) in 6 ops
    instead of 10.  y = (q<<2)&0x3C3C3C3C holds the low nibbles (n0,n2,n4,n6)<<2, z =
    (q>>2)&0x3C3C3C3C the high nibbles (n1,n3,n5,n7)<<2 (the mask keeps bits 2..5 of every
    byte, so no bit crosses a byte); out0 = (n0,n4,n1,n5) = prmt(y,z,0x6420), out1 =
    (n2,n6,n3,n7) = prmt(y,z,0x7531).  Exhaustively checked against the 10-op form
    with exhaustive nibble and randomized word checks."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(q).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 y, z;
            shl.b32 y, $2, 2;
            and.b32 y, y, 0x3C3C3C3C;
            shr.b32 z, $2, 2;
            and.b32 z, z, 0x3C3C3C3C;
            prmt.b32 $0, y, z, 0x6420;
            prmt.b32 $1, y, z, 0x7531;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    b0 = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    b1 = llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)
    return Uint32(b0), Uint32(b1)


@dsl_user_op
def prep_scale_byte_to_f32_v2(packed: Uint32, byte_idx: int, *, loc=None, ip=None):
    """same bits as prep_scale_byte_to_f32 ((b + 840) << 20, b == 0 -> 0.0)
    as prmt (zero-extended byte) + mad (b * 2**20 + 0x34800000) + setp/selp."""
    from cutlass import Float32

    sel = 0x4440 + int(byte_idx)
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(packed).ir_value(loc=loc, ip=ip)],
            f"""
            {{
                .reg .b32 b, r;
                .reg .pred z;
                prmt.b32 b, $1, 0, {sel:#06x};
                mad.lo.u32 r, b, 1048576, 880803840;
                setp.eq.u32 z, b, 0;
                selp.b32 r, 0, r, z;
                mov.b32 $0, r;
            }}
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16x2_bits_to_f32x2(packed: Uint32, *, loc=None, ip=None):
    """one u32 of two bf16 (element 2i in the low half) -> two exact f32
    (bf16 -> f32 is exact: the bf16 bits are the top half of the f32)."""
    from cutlass import Float32

    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            shl.b32 lo, $2, 16;
            and.b32 hi, $2, 0xFFFF0000;
            mov.b32 $0, lo;
            mov.b32 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def pack_f32x2_to_bf16x2_rn(x0, x1, *, loc=None, ip=None) -> Uint32:
    """(x0 -> low half, x1 -> high half), RN, NO satfinite -- the same
    rounding as the scalar cvt.rn.bf16.f32 of the stock top-k sum (inf stays inf)."""
    from cutlass import Float32

    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(x0).ir_value(loc=loc, ip=ip),
                Float32(x1).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def prep_scale_byte_to_f32(packed: Uint32, byte_idx: int, *, loc=None, ip=None):
    """Byte `byte_idx` of 4 prepared E5M3 scales -> exact f32 2**(E5-22) * (1 + M/8).

    f32 bits = (b + 840) << 20  (E5 lands in the exponent with bias 127-22=105=840/8, M in the
    top 3 mantissa bits); b == 0 (flushed) -> 0.0.
    """
    from cutlass import Float32

    shift = 8 * int(byte_idx)
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(packed).ir_value(loc=loc, ip=ip)],
            f"""
            {{
                .reg .b32 b, r;
                .reg .pred z;
                shr.u32 b, $1, {shift};
                and.b32 b, b, 0xFF;
                add.u32 r, b, 840;
                shl.b32 r, r, 20;
                setp.eq.u32 z, b, 0;
                selp.b32 r, 0, r, z;
                mov.b32 $0, r;
            }}
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f8f6f4_mma_m16n8k32_f32(
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    btype: str = "e4m3",
    *,
    loc=None,
    ip=None,
):
    """Unscaled `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.<btype>.f32` with C = 0.

    Returns the k32 partial sum (D) so the caller can apply a per-column f32 scale (exact
    variant).  btype: "e4m3" or "e2m1" (8-bit containers, value at bits [5:2])."""
    from cutlass import Float32

    if btype not in ("e4m3", "e2m1"):
        raise ValueError(btype)
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(x).ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)],
        f"""
        {{
            .reg .f32 z;
            mov.f32 z, 0f00000000;
            mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.{btype}.f32
            {{$0, $1, $2, $3}}, {{$4, $5, $6, $7}}, {{$8, $9}}, {{z, z, z, z}};
        }}
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)
    )
