"""Order-preserving integer keys for floating-point values.

Selection kernels compare and bin values millions of times.  Comparing floats directly is
slow and ambiguous (signed zeros, NaN).  Instead every value is mapped once to an unsigned
integer *key* such that ``a < b`` for the floats implies ``key(a) < key(b)`` for the
integers.  All later work (histograms, radix digits, tie ranking) is done on keys.

The mapping is the classic "twiddle": for a non-negative float set the sign bit, for a
negative float flip every bit.  It is a bijection on bit patterns, so a key can be turned
back into the exact value.  Properties that the kernels rely on:

* monotone over all finite values and +-inf;
* -0.0 and +0.0 map to distinct adjacent keys (-0.0 below +0.0);
* NaN with the sign bit clear maps above +inf (largest keys), NaN with the sign bit set
  maps below -inf; kernels canonicalize NaN to positive so "NaN on top" is the rule;
* key 0 is the smallest possible value (a negative NaN pattern), key 0xFFFFFFFF the largest.

16-bit floats (fp16, bf16) use the same twiddle on 16 bits and occupy the low half of a
32-bit word, so one set of histogram and radix code serves both widths.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = [
    "key_of_f32_bits",
    "f32_bits_of_key",
    "key_of_f16_bits",
    "f16_bits_of_key",
    "key_of_f32",
    "f32_of_key",
    "f16x2_of_f32_pair",
    "key_of_f16x2_bits",
    "pack_threshold_16",
    "below_or_equal_mask_16x8",
]


@cute.jit
def key_of_f32_bits(bits):
    """Ordered key of an fp32 bit pattern held in a Uint32.

    Contract: pure, one value in, one out; no memory, no barriers.
    Cost: a shift, a negate, an or and an xor (4 instructions).
    """
    sign = bits >> cutlass.Uint32(31)  # 1 for negative patterns
    xor_mask = (cutlass.Uint32(0) - sign) | cutlass.Uint32(0x80000000)
    return bits ^ xor_mask


@cute.jit
def f32_bits_of_key(key):
    """Inverse of ``key_of_f32_bits``: ordered key back to the fp32 bit pattern.

    A key with its top bit set came from a non-negative float (undo by clearing the bit);
    otherwise every bit was flipped (undo by flipping again).  Same cost as the forward map.
    """
    positive = key >> cutlass.Uint32(31)  # 1 iff the original float was non-negative
    xor_mask = (cutlass.Uint32(0) - (cutlass.Uint32(1) - positive)) | cutlass.Uint32(
        0x80000000
    )
    return key ^ xor_mask


@cute.jit
def key_of_f16_bits(bits):
    """Ordered key of a 16-bit float pattern (fp16 or bf16) held in the low half of a Uint32.

    Returns a key in [0, 0xFFFF]; the high half is zero.  Pure, 5 instructions.
    """
    sign = bits >> cutlass.Uint32(15)
    xor_mask = ((cutlass.Uint32(0) - sign) & cutlass.Uint32(0xFFFF)) | cutlass.Uint32(
        0x8000
    )
    return (bits ^ xor_mask) & cutlass.Uint32(0xFFFF)


@cute.jit
def f16_bits_of_key(key):
    """Inverse of ``key_of_f16_bits``: 16-bit ordered key back to the float pattern."""
    positive = key >> cutlass.Uint32(15)
    xor_mask = (
        (cutlass.Uint32(0) - (cutlass.Uint32(1) - positive)) & cutlass.Uint32(0xFFFF)
    ) | cutlass.Uint32(0x8000)
    return (key ^ xor_mask) & cutlass.Uint32(0xFFFF)


@cute.jit
def key_of_f32(x):
    """Ordered key of an fp32 value (bitcast, then twiddle)."""
    return key_of_f32_bits(x.bitcast(cutlass.Uint32))


@cute.jit
def f32_of_key(key):
    """fp32 value of an ordered key (inverse twiddle, then bitcast)."""
    return f32_bits_of_key(key).bitcast(cutlass.Float32)


@dsl_user_op
def f16x2_of_f32_pair(
    low_bits: cutlass.Uint32, high_bits: cutlass.Uint32, *, loc=None, ip=None
):
    """Two fp32 values (as bit patterns) rounded to fp16 and packed, ``low`` in the low half.

    ``cvt.rn.f16x2.f32``: one instruction converts both, where two scalar conversions would
    each take a quarter-rate slot.  Rounding to nearest is monotone, so the packed halves keep
    the values' order for binning.  SM80 and newer.
    """
    result = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Uint32(high_bits).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(low_bits).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .f32 a, b;\nmov.b32 a, $1;\nmov.b32 b, $2;\ncvt.rn.f16x2.f32 $0, a, b;\n}",
        "=r,r,r",
        has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(result)


@cute.jit
def key_of_f16x2_bits(pair):
    """Ordered 16-bit keys of both halves of a packed fp16/bf16 pair, in place.

    The twiddle applied to two halves at once with 32-bit arithmetic: each half's mask is
    0xFFFF if its sign bit is set, else 0x8000.  Pure, 5 instructions for two keys.
    """
    signs = (pair >> cutlass.Uint32(15)) & cutlass.Uint32(0x00010001)
    xor_mask = (signs * cutlass.Uint32(0xFFFF)) | cutlass.Uint32(0x80008000)
    return pair ^ xor_mask


@dsl_user_op
def pack_threshold_16(threshold: cutlass.Float32, bf16: bool, *, loc=None, ip=None):
    """Both halves of a Uint32 set to the largest 16-bit value not above ``threshold``.

    The packed compare below asks ``x <= t16`` for 16-bit grid values x.  Rounding the fp32
    threshold DOWN to the grid (``cvt.rm``) makes that equal to ``f32(x) <= threshold``
    for every grid value, so the 16-bit walk classifies exactly like the fp32 one.
    ``bf16`` selects the format at trace time.  Pure, 2 instructions.
    """
    fmt = "bf16" if bf16 else "f16"
    result = llvm.inline_asm(
        T.i32(),
        [cutlass.Float32(threshold).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.rm." + fmt + ".f32 h, $1;\nmov.b32 $0, {h, h};\n}",
        "=r,f",
        has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def below_or_equal_mask_16x8(
    mask: cutlass.Int32,
    w0: cutlass.Uint32,
    w1: cutlass.Uint32,
    w2: cutlass.Uint32,
    w3: cutlass.Uint32,
    packed_threshold: cutlass.Uint32,
    base: int,
    bf16: bool,
    *,
    loc=None,
    ip=None,
):
    """OR into ``mask`` one bit per element of a 16-byte vector of eight 16-bit floats that is
    <= the threshold.

    Element e of the vector lives in word e // 2, low half first, and sets bit ``base + e``.
    NaN compares false, so NaN never sets a bit (it "survives" a threshold, matching the
    NaN-on-top key order).  Uses the two-result packed compare ``setp.le.{f16x2,bf16x2}``:
    4 compares and 8 predicated ORs for 8 elements, half the instruction count of scalar
    fp32 compares.  ``bf16x2`` needs SM90 or newer; callers choose the scalar path below that.
    """
    fmt = "bf16x2" if bf16 else "f16x2"
    body = "{\n.reg .pred p0, p1, p2, p3, p4, p5, p6, p7;\nmov.b32 $0, $1;\n"
    for j in range(4):
        body += f"setp.le.{fmt} p{2 * j}|p{2 * j + 1}, ${2 + j}, $6;\n"
    for e in range(8):
        body += f"@p{e} or.b32 $0, $0, 0x{(1 << (base + e)) & 0xFFFFFFFF:08x};\n"
    body += "}"
    result = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Int32(mask).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w0).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w1).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w2).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w3).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(packed_threshold).ir_value(loc=loc, ip=ip),
        ],
        body,
        "=r,r,r,r,r,r,r",
        has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)
