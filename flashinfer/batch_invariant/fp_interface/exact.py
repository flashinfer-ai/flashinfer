"""Exact IEEE-754 binary32 / bfloat16 / binary16 arithmetic on the CPU.

Values are Python floats restricted to the target format (every binary32 and every
bf16 / f16 value is exactly representable as a double). Every op computes the exact
rational result with ``fractions.Fraction`` and rounds once, round-to-nearest-even,
including subnormals, overflow to infinity, and the IEEE sign-of-zero rules. This is
the reference backend used to evaluate a canonical spec bit-exactly.
"""

from __future__ import annotations

import math
import struct
from decimal import Decimal, getcontext
from fractions import Fraction

INF = float("inf")
NAN = float("nan")

# (precision bits incl. hidden bit, min normal exponent, max exponent)
F32 = (24, -126, 127)
BF16 = (8, -126, 127)
F16 = (11, -14, 15)
# fp8 (saturating-finite conversion): (precision, emin, emax, max finite)
E4M3FN = (4, -6, 8, 448.0)
E5M2 = (3, -14, 15, 57344.0)


def _is_neg(x: float) -> bool:
    return math.copysign(1.0, x) < 0


def round_to_format(fr: Fraction, fmt=F32, *, zero_sign_neg: bool = False) -> float:
    """Round an exact rational to the given binary format (RNE)."""
    p, emin, emax = fmt
    if fr == 0:
        return -0.0 if zero_sign_neg else 0.0
    sign = -1.0 if fr < 0 else 1.0
    a = -fr if fr < 0 else fr
    # exponent e with 2^e <= a < 2^(e+1)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    elif Fraction(2) ** (e + 1) <= a:
        e += 1
    e = max(e, emin)
    q = Fraction(2) ** (e - p + 1)
    n = a / q
    r = n.numerator // n.denominator
    rem = n - r
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (r & 1)):
        r += 1
    val = r * q
    if val >= Fraction(2) ** (emax + 1):
        return sign * INF
    return sign * float(val)  # exact: val is representable in binary64


def _finite(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


def add(a: float, b: float, fmt=F32) -> float:
    if not _finite(a, b):
        return a + b
    fr = Fraction(a) + Fraction(b)
    if fr == 0:
        # exact zero: -0 only if both operands are -0 (RNE rules)
        neg = a == 0 and b == 0 and _is_neg(a) and _is_neg(b)
        return round_to_format(fr, fmt, zero_sign_neg=neg)
    return round_to_format(fr, fmt)


def mul(a: float, b: float, fmt=F32) -> float:
    if not _finite(a, b):
        return a * b
    fr = Fraction(a) * Fraction(b)
    return round_to_format(fr, fmt, zero_sign_neg=(_is_neg(a) != _is_neg(b)))


def fma(a: float, b: float, c: float, fmt=F32) -> float:
    if not _finite(a, b, c):
        return a * b + c
    prod = Fraction(a) * Fraction(b)
    fr = prod + Fraction(c)
    if fr == 0:
        prod_neg = _is_neg(a) != _is_neg(b)
        if prod == 0 and c == 0:
            neg = prod_neg and _is_neg(c)
        else:
            neg = False  # exact cancellation -> +0 under RNE
        return round_to_format(fr, fmt, zero_sign_neg=neg)
    return round_to_format(fr, fmt)


def div(a: float, b: float, fmt=F32) -> float:
    if not _finite(a, b) or b == 0:
        if (
            b == 0
        ):  # IEEE: x / 0 is the signed infinity of the operands' signs, NaN for 0 / 0 and NaN / 0
            if a == 0 or math.isnan(a):
                return NAN
            return math.copysign(INF, a) * math.copysign(1.0, b)
        return a / b
    fr = Fraction(a) / Fraction(b)
    return round_to_format(fr, fmt, zero_sign_neg=(_is_neg(a) != _is_neg(b)))


def sqrt(a: float, fmt=F32) -> float:
    if not math.isfinite(a) or a == 0:
        return math.sqrt(a) if a >= 0 or a == 0 else NAN
    if a < 0:
        return NAN
    fr = Fraction(a)
    # a = m * 2^e with m odd integer; make e even then sqrt(m) * 2^(e/2)
    m, d = fr.numerator, fr.denominator
    e = -(d.bit_length() - 1)  # d is a power of two
    if e % 2:
        m *= 2
        e -= 1
    getcontext().prec = 80
    sm = Fraction(Decimal(m).sqrt())  # 80 significant digits >> any rounding boundary
    return round_to_format(sm * Fraction(2) ** (e // 2), fmt)


def fmaxc(a: float, b: float) -> float:
    """The attention class's PTX ``max.f32`` (NVVM ``fmax``, IEEE maximumNumber): a NaN operand yields the other operand;
    zeros of opposite sign yield ``+0`` whatever the operand order (so ``fmax`` is bitwise
    commutative on non-NaN inputs and a join of the total order ``-inf < ... < -0 < +0 < ... < inf``);
    probed on sm_107, 2026-10-27."""
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    if a == b:
        if a == 0.0:
            return -0.0 if (_is_neg(a) and _is_neg(b)) else 0.0
        return a
    return a if a > b else b


def to_format(x: float, fmt) -> float:
    """Round a binary32 value to bf16 / f16 (RNE). Identity for F32."""
    if fmt == F32 or not math.isfinite(x):
        return x
    if x == 0:
        return x
    return round_to_format(Fraction(x), fmt)


def to_fp8(x: float, kind: str) -> float:
    """``cvt.rn.satfinite.{e4m3,e5m2}x2.f32`` semantics: RNE, then saturate to the
    largest finite value (no infinities). NaN stays NaN."""
    p, emin, emax, fmax = {"e4m3": E4M3FN, "e5m2": E5M2}[kind]
    if math.isnan(x):
        return x
    if x == 0 or math.isinf(x):
        return math.copysign(0.0, x) if x == 0 else math.copysign(fmax, x)
    v = round_to_format(Fraction(x), (p, emin, emax))
    if math.isinf(v) or abs(v) > fmax:
        v = math.copysign(fmax, v)
    return v


def neg(a: float) -> float:
    """``neg.f32``: a sign flip, exact (``-(+0) = -0``)."""
    return -a


def fmin(a: float, b: float) -> float:
    """``min.f32`` on non-NaN operands. Equal values (incl. ``+0`` / ``-0``) return ``a``; the specs
    only clamp against non-zero constants, so that case never reaches the bits."""
    return a if a <= b else b


def fmax(a: float, b: float) -> float:
    return a if a >= b else b


def exp2i(k: float) -> float:
    """The binary32 value ``2^k`` for an integer-valued ``k`` in ``[-126, 127]`` (the device builds
    the exponent field from the integer; both are exact)."""
    n = int(k)
    assert n == k and -126 <= n <= 127, k
    return math.ldexp(1.0, n)


def rowmax(*vals: float) -> float:
    """``max`` of non-NaN values. Its value does not depend on the association; only the sign of a
    zero maximum could, and the MoE spec shows that sign never reaches the bits."""
    m = vals[0]
    for v in vals[1:]:
        if v > m:
            m = v
    return m


# ----------------------------------------------------------------------------- NVFP4 quantization primitives
F32_MAX = 3.4028234663852886e38  # FLT_MAX, 0x7F7FFFFF

E2M1_MAGNITUDES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
)  # e2m1 code k (sign bit clear) = this magnitude
E2M1_MAX = 6.0
_E2M1_MIDPOINTS = (
    0.25,
    0.75,
    1.25,
    1.75,
    2.5,
    3.5,
    5.0,
)  # between codes k and k+1; exact binary32 values
# `cvt.rn.satfinite.e2m1x2.f32` maps a NaN input to the positive maximum (code 0x7 = +6.0): measured on
# sm_107 (2026-10-27) and pinned by the device sweep in tests/test_nvfp4_quant_gpu.py. NaN inputs are
# out of scope for the specification; the value is recorded so the CPU model is total.
E2M1_NAN_CODE = 0x7


def fabs(a: float) -> float:
    """``abs.f32``: clears the sign bit, exact (``fabs(-0) = +0``; NaN stays NaN)."""
    return abs(a)


def nzor(a: float, b: float) -> float:
    """``a`` if ``a != 0`` else ``b`` (``setp.ne`` + ``selp``): an exact select. Both signed zeros count
    as zero; a NaN ``a`` is ``!= 0`` and is returned."""
    return a if a != 0 else b


def to_e2m1(x: float) -> float:
    """``cvt.rn.satfinite.e2m1x2.f32`` on one value, as a value: round to the nearest e2m1 magnitude
    {0, 0.5, 1, 1.5, 2, 3, 4, 6}, ties to the even code (0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2,
    2.5 -> 2, 3.5 -> 4, 5 -> 4), magnitudes beyond 6 (infinity included) saturate to 6, the sign is
    kept (``-0`` and small negatives give ``-0``). NaN gives ``+6.0`` (code ``E2M1_NAN_CODE``), as the
    device does. Verified bit for bit against the device conversion over all 2^32 inputs (sm_107,
    2026-10-27)."""
    if math.isnan(x):
        return E2M1_MAX
    s = math.copysign(1.0, x)
    a = abs(x)
    if a >= E2M1_MAX:
        return s * E2M1_MAX
    code = 0
    for k, m in enumerate(_E2M1_MIDPOINTS):
        if a > m:
            code = k + 1
        else:
            if a == m:  # tie: the even code
                code = k + 1 if (k + 1) % 2 == 0 else k
            break
    return s * E2M1_MAGNITUDES[code]


def e2m1_code(v: float) -> int:
    """The 4-bit code of an e2m1 value (sign in bit 3)."""
    if math.isnan(v):
        return E2M1_NAN_CODE
    k = E2M1_MAGNITUDES.index(abs(v))
    return k | (0x8 if math.copysign(1.0, v) < 0 else 0)


def e2m1_value(code: int) -> float:
    """The value of a 4-bit e2m1 code."""
    v = E2M1_MAGNITUDES[code & 0x7]
    return -v if code & 0x8 else v


def _e4m3_table():
    out = []
    for b in range(256):
        sign, exp, mant = b >> 7, (b >> 3) & 0xF, b & 0x7
        if exp == 15 and mant == 7:
            v = NAN
        elif exp == 0:
            v = mant * 2.0**-9
        else:
            v = (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
        out.append(-v if sign else v)
    return tuple(out)


E4M3_VALUES = (
    _e4m3_table()
)  # byte -> value (e4m3fn: 0x7F / 0xFF are NaN, no infinities)


def e4m3_bits(v: float) -> int:
    """The byte of an e4m3fn value (``v`` must be exactly representable, e.g. the result of
    ``to_fp8(x, "e4m3")``)."""
    if math.isnan(v):
        return 0x7F
    neg = math.copysign(1.0, v) < 0
    a = abs(v)
    for b in range(127):
        if E4M3_VALUES[b] == a:
            return b | (0x80 if neg else 0)
    raise ValueError(f"{v!r} is not an e4m3 value")


# ----------------------------------------------------------------------------- MX (UE8M0) primitives
UE8M0_BIAS = 127
UE8M0_MAX_BYTE = 254  # the largest finite byte (2^127); 255 is the UE8M0 NaN
UE8M0_MIN_VALUE = 2.0**-127  # byte 0: a binary32 subnormal (0x00400000)
UE8M0_MAX_VALUE = 2.0**127  # byte 254
UE8M0_NAN_BYTE = 0xFF
UE8M0_ROUNDINGS = ("rp", "rz", "rn")


def to_ue8m0(x: float, rnd: str = "rp") -> float:
    """``cvt.<rnd>.satfinite.ue8m0x2.f32`` on one value, as the VALUE ``2^(byte - 127)`` of the resulting UE8M0 byte
    (every such value is a binary32: ``2^-127`` is the subnormal ``0x00400000``). ``rp``: the least power of two
    ``>= x`` (the hardware mode TE and FlashInfer use, "RCEIL"); ``rz``: the greatest power of two ``<= x``; ``rn``:
    the nearest power of two in linear space, ties up (no hardware mode: an exact software rule). Every mode converts the
    MAGNITUDE of ``x`` (the hardware ignores the sign: ``-2.0`` gives byte 128; never reached on the classes' domain, whose
    amax is ``fabs(x) >= +0``; found by the exhaustive device sweep, 2026-10-30), maps both zeros and ``0 < |x| <= 2^-127`` to
    byte 0 (the format has nothing smaller), ``|x| >= 2^127`` and infinities to byte 254 (satfinite), NaN to NaN (byte 255;
    out of scope). The device conversions are verified against this function over every binary32 bit pattern
    (tests/test_mx_quant_gpu.py)."""
    if rnd not in UE8M0_ROUNDINGS:
        raise ValueError(rnd)
    if math.isnan(x):
        return NAN
    x = abs(x)
    if x == 0:
        return UE8M0_MIN_VALUE
    if math.isinf(x):
        return UE8M0_MAX_VALUE
    fr = Fraction(x)
    e = fr.numerator.bit_length() - fr.denominator.bit_length()
    if Fraction(2) ** e > fr:
        e -= 1
    elif Fraction(2) ** (e + 1) <= fr:
        e += 1
    lo = Fraction(2) ** e  # 2^e <= x < 2^(e+1)
    if rnd == "rz":
        k = e
    elif rnd == "rp":
        k = e if fr == lo else e + 1
    else:
        k = e if fr < lo * 3 / 2 else e + 1  # nearest in linear space, ties up
    k = max(-UE8M0_BIAS, min(127, k))
    return math.ldexp(1.0, k)


def ue8m0_byte(v: float) -> int:
    """The byte of a UE8M0 value (a power of two in ``[2^-127, 2^127]``, as :func:`to_ue8m0` returns); NaN -> 0xFF."""
    if math.isnan(v):
        return UE8M0_NAN_BYTE
    m, e = math.frexp(v)  # v = m * 2^e with m in [0.5, 1)
    if m != 0.5 or not (-UE8M0_BIAS <= e - 1 <= 127):
        raise ValueError(f"{v!r} is not a UE8M0 value")
    return e - 1 + UE8M0_BIAS


def ue8m0_value(byte: int) -> float:
    """The value of a UE8M0 byte: ``2^(byte - 127)``; 0xFF is NaN."""
    if byte == UE8M0_NAN_BYTE:
        return NAN
    return math.ldexp(1.0, byte - UE8M0_BIAS)


CANONICAL_NAN_F32 = 0x7FFFFFFF  # what sm_100-class FP32 ops produce for any NaN result
CANONICAL_NAN_16 = 0x7FFF  # bf16 / f16 canonical NaN after cvt.rn from an f32 NaN


def to_f32(x: float) -> float:
    """The Python float ``x`` (a double) rounded to binary32 with RNE; inf/nan pass through.
    Every constant that enters a kernel or a spec goes through this, so the CPU reference,
    the symbolic spec and the device constant are the same binary32 value (F03)."""
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return x
    return round_to_format(
        Fraction(x), F32, zero_sign_neg=(x == 0.0 and math.copysign(1.0, x) < 0)
    )


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bf16_bits(x: float) -> int:
    return f32_bits(x) >> 16


def f16_bits(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]
