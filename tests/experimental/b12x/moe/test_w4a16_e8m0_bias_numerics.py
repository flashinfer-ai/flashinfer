"""CPU reference for the BF16 W4A16 E8M0 dequant exponent bias (no GPU, NumPy only).

Emulates, bit for bit, what the W4A16 kernel does with one FP4 value ``v`` and
one E8M0 scale byte ``e``:

1. ``packed_dequant_e2m1x4_to_bfloat2x2``: the nibble's sign and 3 magnitude
   bits are moved into a BF16 so that the value is ``v * 2^-126`` (0.5 lands
   on a BF16 subnormal).
2. ``packed_dequant_e8m0x4_to_bfloat2x2``: the scale is the BF16 with
   exponent field ``e + bias`` (16-bit pattern ``((e + bias) << 7) & 0xFFFF``).
3. ``mul.bf16x2``: the MMA weight is ``bf16_rn(w_fp4 * scale)``.
4. The MMA multiplies by BF16 activations and accumulates in FP32; the
   epilogue multiplies by ``2^(126 - bias)``.

Previous constants: bias 7, compensation 2^119, prepare clamp 247.
Current constants:  bias 71, compensation 2^55, prepare clamp 183.

Run with pytest, or ``python3 test_w4a16_e8m0_bias_numerics.py`` for a
per-byte report.  The source-constant test checks that the kernel, prepare and
reference constants in this checkout match the ones modelled here.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

OLD = dict(bias=7, clamp=247)
NEW = dict(bias=71, clamp=183)

# All 16 E2M1 codes (sign bit 3, exponent bits 2..1, mantissa bit 0).
FP4_CODES = np.arange(16)
FP4_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_VALUES = np.where(FP4_CODES >= 8, -1.0, 1.0) * FP4_MAG[FP4_CODES & 7]

BF16_MAX = float(np.ldexp(255.0 / 128.0, 127))  # 0x7F7F
FP32_MAX = float(np.finfo(np.float32).max)


# ---------------------------------------------------------------- bit models
def bf16_from_bits(bits: int) -> float:
    """Decode a 16-bit BF16 pattern exactly (float64)."""
    bits &= 0xFFFF
    sign = -1.0 if bits & 0x8000 else 1.0
    exp = (bits >> 7) & 0xFF
    man = bits & 0x7F
    if exp == 0xFF:
        return sign * math.inf if man == 0 else math.nan
    if exp == 0:
        return sign * math.ldexp(man, -126 - 7)
    return sign * math.ldexp(128 + man, exp - 127 - 7)


def fp4_unpack_bits(code: int) -> int:
    """The PTX in packed_dequant_e2m1x4_to_bfloat2x2 for one nibble at bits 12..15."""
    word = (code & 0xF) << 12
    return (word & 0x8000) | ((word & 0x7000) >> 6)


def scale_bits(e: int, bias: int) -> int:
    """The PTX in packed_dequant_e8m0x4_to_bfloat2x2 for one byte (16-bit lane)."""
    return ((e + bias) << 7) & 0xFFFF


def round_to(
    x: float, mant_bits: int, emin: int, max_finite: float, ftz: bool
) -> float:
    """Round-to-nearest-even into a binary format (float64 in, float64 out)."""
    if x == 0.0 or not math.isfinite(x):
        return x
    ax = abs(x)
    if ax < math.ldexp(1.0, emin):
        if ftz:
            return math.copysign(0.0, x)
        q = math.ldexp(1.0, emin - mant_bits)
    else:
        q = math.ldexp(1.0, math.frexp(ax)[1] - 1 - mant_bits)
    r = math.copysign(float(np.rint(ax / q)) * q, x)
    if abs(r) > max_finite:
        return math.copysign(math.inf, x)
    return r


def bf16_rn(x: float, ftz: bool = False) -> float:
    return round_to(x, 7, -126, BF16_MAX, ftz)


def fp32_rn(x: float, ftz: bool = False) -> float:
    return round_to(x, 23, -126, FP32_MAX, ftz)


def kernel_weight(
    code: int, e: int, cfg: dict, *, apply_clamp: bool = True
) -> tuple[float, float]:
    """(MMA-input BF16 weight, compensation) for one FP4 code and scale byte."""
    eb = min(e, cfg["clamp"]) if apply_clamp else e
    w4 = bf16_from_bits(fp4_unpack_bits(code))
    s = bf16_from_bits(scale_bits(eb, cfg["bias"]))
    return bf16_rn(w4 * s), math.ldexp(1.0, 126 - cfg["bias"])


def true_weight(code: int, e: int) -> float:
    return FP4_VALUES[code] * math.ldexp(1.0, e - 127)


def is_bf16_normal_or_zero(x: float) -> bool:
    return x == 0.0 or (math.isfinite(x) and abs(x) >= math.ldexp(1.0, -126))


# ------------------------------------------------------------------- sweeps
def sweep(cfg: dict, *, apply_clamp: bool = True) -> dict[int, dict]:
    """Per scale byte: exact? flushed nonzero codes? subnormal? non-finite?"""
    out = {}
    for e in range(0, 255):
        exact = flushed = subnormal = nonfinite = 0
        for code in FP4_CODES:
            w, comp = kernel_weight(int(code), e, cfg, apply_clamp=apply_clamp)
            t = true_weight(int(code), e)
            if not math.isfinite(w):
                nonfinite += 1
                continue
            if t != 0.0 and w == 0.0:
                flushed += 1
            if not is_bf16_normal_or_zero(w):
                subnormal += 1
            if w * comp == t:
                exact += 1
        out[e] = dict(
            exact=exact, flushed=flushed, subnormal=subnormal, nonfinite=nonfinite
        )
    return out


def dot_rel_error(
    cfg: dict, e: int, *, ftz: bool, k: int = 2048, seed: int = 0
) -> float:
    """Error of one K-long dot product (FP32 products and FP32 sum), normalised
    by sum |x_i * w_i| so random cancellation does not inflate it."""
    rng = np.random.default_rng(seed + e)
    codes = rng.integers(0, 16, size=k)
    # BF16 activations, log-uniform over 2^-14 .. 2^4 (post-norm hidden states
    # and SwiGLU outputs both reach far below 1).
    x = [
        bf16_rn(float(sgn) * math.ldexp(1.0, int(p)) * float(m))
        for sgn, p, m in zip(
            rng.choice([-1, 1], size=k),
            rng.integers(-14, 5, size=k),
            rng.uniform(1.0, 2.0, size=k),
        )
    ]
    acc = 0.0
    ref = 0.0
    l1 = 0.0
    comp = 1.0
    for c, xi in zip(codes, x):
        w, comp = kernel_weight(int(c), e, cfg)
        acc = fp32_rn(acc + fp32_rn(xi * w, ftz), ftz)
        ref += xi * true_weight(int(c), e)
        l1 += abs(xi * true_weight(int(c), e))
    got = fp32_rn(acc * comp)
    return abs(got - ref) / max(l1, 1e-300)


# -------------------------------------------------------------------- tests
def test_fp4_unpack_is_v_times_2_pow_minus_126():
    for code in FP4_CODES:
        assert (
            bf16_from_bits(fp4_unpack_bits(int(code))) == FP4_VALUES[code] * 2.0**-126
        )


def test_old_path_underflows_below_121():
    old = sweep(OLD)
    # Window where every nonzero weight is a normal BF16 and exact.
    assert all(
        old[e]["subnormal"] == 0 and old[e]["exact"] == 16 for e in range(121, 248)
    )
    # Bytes 111-120 (seen in real MXFP4 checkpoints): BF16-subnormal weights.
    assert all(old[e]["subnormal"] > 0 for e in range(111, 121))
    # 0.5 flushes to zero below 114; below 111 most of the grid is gone.
    assert all(old[e]["flushed"] == 0 for e in range(114, 255))
    assert all(old[e]["flushed"] >= 2 for e in range(0, 114))
    assert old[113]["flushed"] == 2 and old[106]["flushed"] >= 8


def test_new_path_exact_and_normal_on_real_ranges():
    new = sweep(NEW)
    for e in range(57, 184):
        assert new[e] == dict(exact=16, flushed=0, subnormal=0, nonfinite=0), (
            e,
            new[e],
        )
    # Subnormal but still exact for 50..56; first flush at 49 (vs 113 before).
    assert all(new[e]["exact"] == 16 for e in range(50, 57))
    assert new[49]["flushed"] == 2


def test_no_nonfinite_anywhere_after_prepare_clamp():
    for cfg in (OLD, NEW):
        s = sweep(cfg)
        assert all(s[e]["nonfinite"] == 0 for e in range(255)), cfg
    # Without the clamp the new bias would overflow from 184 (old: 248).
    raw = sweep(NEW, apply_clamp=False)
    assert raw[183]["nonfinite"] == 0 and raw[184]["nonfinite"] > 0


def test_clamped_bytes_saturate_identically_to_before():
    # Above the clamp both paths return the weight at the clamp byte: finite,
    # wrong by 2^(e - clamp). Upstream accepted this at 247; the patch moves it
    # to 183 (a 2^56 scale), which no trained checkpoint approaches.
    for cfg in (OLD, NEW):
        c = cfg["clamp"]
        for code in FP4_CODES:
            w, comp = kernel_weight(int(code), 254, cfg)
            assert w * comp == true_weight(int(code), c)


def test_dot_products_old_vs_new():
    # Two FP32 models of the MMA: gradual underflow, and flush-to-zero of
    # subnormal products/partial sums. The GPU scale-shift experiment
    # (scales -5 breaks a clean layer, +6 heals a broken one) behaves like the
    # FTZ column.
    for ftz in (False, True):
        for e in (100, 106, 111, 114, 117, 120, 122, 130):
            new_err = dot_rel_error(NEW, e, ftz=ftz)
            assert new_err < 1e-6, (ftz, e, new_err)
    # Old path: the weight flush alone wrecks bytes <= 111 even with gradual
    # underflow; under FTZ every byte in 111-120 loses > 1e-4 of |signal|.
    assert dot_rel_error(OLD, 111, ftz=False) > 1e-2
    for e in (111, 114, 117, 120):
        assert dot_rel_error(OLD, e, ftz=True) > 1e-4, e
    # Well inside the old window (byte 130) the old path is fine.
    assert dot_rel_error(OLD, 130, ftz=True) < 1e-6


def test_accumulator_headroom_at_clamp():
    # Worst case: every weight 6 * 2^(183-127), BF16 max-ish activations,
    # K = 8192: the FP32 accumulator holds true * 2^-55.
    w, comp = kernel_weight(7, 183, NEW)
    acc = 8192 * 65504.0 * w
    assert math.isfinite(fp32_rn(acc)) and acc < 2.0**60
    # Epilogue lift itself and the BF16 top-k scale (topk * 2^55) are finite.
    assert math.isfinite(bf16_rn(1.0 * comp))


REPO = Path(__file__).resolve().parents[4]
B12X = REPO / "flashinfer/experimental/b12x"


def check_source_constants(root: Path = B12X, cfg: dict = NEW) -> None:
    intr = (root / "_lib/intrinsics.py").read_text()
    i = intr.index("def packed_dequant_e8m0x4_to_bfloat2x2")
    j = intr.index("def packed_dequant_e8m0x4_to_half2x2")
    adds = re.findall(r"add\.u32 h\d, b\d, (\d+);", intr[i:j])
    assert adds == [str(cfg["bias"])] * 4, adds
    kern = (root / "moe/_shared/kernels/w4a16/kernel.py").read_text()
    m = re.search(r"^_E8M0_K32_BF16_SCALE_BIAS = (\d+)$", kern, re.M)
    assert m and int(m.group(1)) == cfg["bias"]
    assert (
        "_E8M0_K32_BF16_GLOBAL_COMPENSATION = "
        "float(2.0 ** (126 - _E8M0_K32_BF16_SCALE_BIAS))"
    ) in kern
    for rel in (
        "moe/_shared/kernels/w4a16/prepare.py",
        "moe/_shared/kernels/reference.py",
    ):
        src = (root / rel).read_text()
        m = re.search(r"^_E8M0_K32_BF16_MAX_SCALE_BYTE = (\d+)$", src, re.M)
        assert m and int(m.group(1)) == cfg["clamp"] == 254 - cfg["bias"], rel


def test_source_constants_match_model():
    if not (B12X / "_lib/intrinsics.py").is_file():
        pytest.skip("b12x sources not found next to this test")
    check_source_constants()


def _report() -> None:
    old, new = sweep(OLD), sweep(NEW)
    print("byte | old: exact/16 flushed subnormal | new: exact/16 flushed subnormal")
    for e in (
        0,
        40,
        49,
        50,
        57,
        100,
        106,
        110,
        111,
        113,
        114,
        117,
        120,
        121,
        122,
        130,
        183,
        184,
        247,
        248,
        254,
    ):
        o, n = old[e], new[e]
        print(
            f"{e:4d} | {o['exact']:2d} {o['flushed']:2d} {o['subnormal']:2d}"
            f"            | {n['exact']:2d} {n['flushed']:2d} {n['subnormal']:2d}"
        )
    print("dot-product error / sum|x*w|, K=2048 (gradual / FTZ fp32):")
    for e in (100, 106, 111, 114, 117, 120, 121, 122, 125, 130):
        print(
            f"  e={e}: old {dot_rel_error(OLD, e, ftz=False):.2e} / {dot_rel_error(OLD, e, ftz=True):.2e}"
            f"   new {dot_rel_error(NEW, e, ftz=False):.2e} / {dot_rel_error(NEW, e, ftz=True):.2e}"
        )


if __name__ == "__main__":
    _report()
