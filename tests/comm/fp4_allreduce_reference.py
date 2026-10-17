"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Exact float32 reference of the NVFP4-transport all-reduce codec.
#
# Every float32 operation of the kernels (``__fmul_rn``, ``__fdiv_rn``,
# ``__fadd_rn``, ``__fmaf_rn``, the saturating E4M3 / E2M1 conversions) is
# reproduced with one IEEE round-to-nearest-even rounding, so the kernels'
# bfloat16 outputs must match these references bitwise.  The codec follows the
# NVFP4 all-reduce reference by Fan Shi (https://github.com/fans-nv/fp4-all-reduce,
# commit 2878bf5), re-derived from its specification.
#
# Row record: 384 blocks of 16 E2M1 values, one E4M3 scale per block and one
# float32 row scale.  ``inv = 2688 / amax`` (``448 * 6``: E4M3 max times E2M1
# max) maps the row into the representable range; the row scale ``rs =
# amax / 2688`` restores it on decode.  Decoding accumulates
# ``fma(e2m1(q), e4m3(sf) * rs, acc)`` in rank order from +0.  The one-shot
# output is ``bf16(acc)``; the two-shot owner re-encodes the float32 sum with
# fresh row scales and the output is the decode of that record.
#
# The fused residual + RMSNorm family keeps the float32 one-shot sum:
# ``z = sum + residual`` (float32), ``residual_out = bf16(z)`` (bitwise),
# ``y = (z * rsqrt(mean(z^2) + eps)) * (1 + weight)`` in float32, then
# ``bf16(y)`` or the ``fp4_quantize`` / ``mxfp8_quantize`` quantisation of that
# bfloat16 ``y`` (bitwise against those kernels on the fused kernel's own ``y``).
# The kernels sum ``z^2`` in their own order, so ``y`` is compared with a
# tolerance (``atol = rtol = 1e-2``) and the bfloat16 ulp distance is reported.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

HIDDEN = 6144
BLOCK = 16
BLOCKS_PER_ROW = HIDDEN // BLOCK  # 384
VALUE_BYTES = HIDDEN // 2  # 3072 packed E2M1 bytes
SCALE_BYTES = BLOCKS_PER_ROW  # 384 E4M3 bytes
RECORD_BYTES = VALUE_BYTES + SCALE_BYTES  # 3456
ROW_AMAX_TARGET = 2688.0  # row amax maps onto the largest encodable product: E4M3 scale 448 x E2M1 value 6
E4M3_MAX = 448.0
SOURCE_MAGNITUDES = (1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.125)
OUTLIER_COUNT = 32
OUTLIER_MAGNITUDE = 40.0
# Fused residual + RMSNorm family (world size 2).
NORM_EPS = 1e-6
NORM_WEIGHT_BIAS = 1.0  # Gemma: y = z * rinv * (weight_bias + weight)
NEGATIVE_CONTROL_WEIGHT_BIAS = (
    1.0 + 2.0**-7
)  # a reference with this bias must fail the y tolerance
RESIDUAL_SOURCE_RANK = (
    5  # residual stream: magnitude 0.75 plus outliers (same tensor on every rank)
)
RESIDUAL_SEED_XOR = 0x5E5
WEIGHT_SOURCE_RANK = (
    7  # norm weight: magnitude 0.125, no outliers, so 1 + weight stays in [0.4, 1.6]
)
WEIGHT_SEED = 4242
SCALE_ROW_TILE = (
    128  # swizzled output scale tensors cover rows padded to a multiple of 128
)
NVFP4_OUTPUT_SCALES_PER_ROW = 384  # one E4M3 scale per 16 values
MXFP8_OUTPUT_SCALES_PER_ROW = 192  # one UE8M0 scale per 32 values
RSQRT_FORM_DEVICE = "rsqrt.approx.f32"  # torch.rsqrt on CUDA float32 (libdevice rsqrtf)
RSQRT_FORM_IEEE = (
    "ieee_rn_f64"  # 1 / sqrt in float64 rounded once to float32 (CPU tensors)
)
_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_U32_MASK = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Exact float32 primitives (one IEEE rounding each)
# ---------------------------------------------------------------------------


def f32_mul_rn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``__fmul_rn``: the float64 product of two float32 values is exact; one rounding to float32."""
    return (a.to(torch.float64) * b.to(torch.float64)).to(torch.float32)


def f32_div_rn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``__fdiv_rn``: a float64 quotient rounded to float32 is the correctly rounded float32 quotient."""
    return (a.to(torch.float64) / b.to(torch.float64)).to(torch.float32)


def f32_add_rn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``__fadd_rn``: the float64 sum of two float32 values is exact; one rounding to float32."""
    return (a.to(torch.float64) + b.to(torch.float64)).to(torch.float32)


def f32_fma_rn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``__fmaf_rn`` through float64 round-to-odd.

    ``a * b`` is exact in float64.  ``s = fl64(p + c)`` with its exact TwoSum error
    ``e``; when ``e != 0`` and ``s`` has an even last bit, ``s`` is nudged one
    float64 ulp toward the exact value (round-to-odd).  Rounding that float64 to
    float32 is then one correct rounding of ``a * b + c``.
    """
    p = a.to(torch.float64) * b.to(torch.float64)
    c64 = c.to(torch.float64)
    s = p + c64
    bb = s - p
    err = (p - (s - bb)) + (c64 - bb)
    even = (s.view(torch.int64) & 1) == 0
    adjust = (err != 0) & even
    toward = torch.where(
        err > 0, torch.full_like(s, math.inf), torch.full_like(s, -math.inf)
    )
    s_odd = torch.where(adjust, torch.nextafter(s, toward), s)
    return s_odd.to(torch.float32)


def e4m3_rn_satfinite(x: torch.Tensor) -> torch.Tensor:
    """Saturating round-to-nearest-even float32 -> E4M3 codes (uint8); NaN stays NaN (0x7F)."""
    clamped = torch.where(torch.isnan(x), x, x.clamp(min=-E4M3_MAX, max=E4M3_MAX))
    return clamped.to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_to_f32(codes: torch.Tensor) -> torch.Tensor:
    """E4M3 code -> float32 (exact)."""
    return codes.view(torch.float8_e4m3fn).to(torch.float32)


def e2m1_rn_satfinite(x: torch.Tensor) -> torch.Tensor:
    """Saturating round-to-nearest-even float32 -> E2M1 codes 0..15 (bit 3 = sign).

    Grid ``{0, .5, 1, 1.5, 2, 3, 4, 6}``; ``|x| >= 6`` saturates; NaN -> +6 (code 7).
    Negative values that round to zero yield -0 (code 8); the kernels canonicalise
    it to +0 on the wire (:func:`canon_neg0_codes`).
    """
    a = x.abs()
    code = torch.zeros(x.shape, dtype=torch.uint8, device=x.device)
    for threshold, strict, value in (
        (0.25, True, 1),
        (0.75, False, 2),
        (1.25, True, 3),
        (1.75, False, 4),
        (2.5, True, 5),
        (3.5, False, 6),
        (5.0, True, 7),
    ):
        hit = a > threshold if strict else a >= threshold
        code = torch.where(hit, torch.full_like(code, value), code)
    sign = (torch.signbit(x) & ~torch.isnan(x)).to(torch.uint8) * 8
    code = code | sign
    return torch.where(torch.isnan(x), torch.full_like(code, 7), code)


def e2m1_to_f32(codes: torch.Tensor) -> torch.Tensor:
    """E2M1 code -> float32 (exact table)."""
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=codes.device)
    return table[codes.to(torch.int64)]


def canon_neg0_codes(codes: torch.Tensor) -> torch.Tensor:
    """E2M1 -0 (code 8) -> +0 (code 0); every other code unchanged."""
    return torch.where(codes == 8, torch.zeros_like(codes), codes)


def pack_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    """``[..., 2k]`` codes -> ``[..., k]`` bytes with element ``2i`` in the low nibble."""
    even = codes[..., 0::2].to(torch.uint8)
    odd = codes[..., 1::2].to(torch.uint8)
    return even | (odd << 4)


def unpack_e2m1_bytes(packed: torch.Tensor) -> torch.Tensor:
    """``[..., k]`` bytes -> ``[..., 2k]`` codes (inverse of :func:`pack_e2m1_codes`)."""
    low = packed & 0x0F
    high = packed >> 4
    return torch.stack((low, high), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )


# ---------------------------------------------------------------------------
# Row codec
# ---------------------------------------------------------------------------


def row_amax(values_f32: torch.Tensor) -> torch.Tensor:
    """Row maximum of ``|values|``: ``[R, H] -> [R]``."""
    return values_f32.abs().amax(dim=-1)


def row_scales(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``inv = amax > 0 ? 2688 / amax : 0`` and ``rs = amax / 2688`` (correctly rounded)."""
    target = torch.full_like(amax, ROW_AMAX_TARGET)
    inv = torch.where(amax > 0, f32_div_rn(target, amax), torch.zeros_like(amax))
    rs = f32_div_rn(amax, target)
    return inv, rs


def encode_rows(
    values_f32: torch.Tensor, inv: torch.Tensor, *, canon: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode float32 rows with the row scale ``inv`` -> (values ``[R, 3072]`` u8, scales ``[R, 384]`` u8).

    Per 16-block: ``bm = max|f|``; ``sv = (bm * inv) / 6``; ``sf = e4m3(sv)``;
    ``m = float(sf) > 0 ? inv / float(sf) : 0``; ``g = f * m``; ``q = e2m1(g)``;
    element ``2i`` in the low nibble.  ``canon`` applies the -0 -> +0 wire
    canonicalisation (the decoded value is identical either way).
    """
    rows = values_f32.shape[0]
    blocks = values_f32.reshape(rows, BLOCKS_PER_ROW, BLOCK)
    bm = blocks.abs().amax(dim=-1)
    inv_b = inv.reshape(rows, 1).expand(rows, BLOCKS_PER_ROW)
    sv = f32_div_rn(f32_mul_rn(bm, inv_b), torch.full_like(bm, 6.0))
    sf = e4m3_rn_satfinite(sv)
    sff = e4m3_to_f32(sf)
    m = torch.where(sff > 0, f32_div_rn(inv_b, sff), torch.zeros_like(sff))
    g = f32_mul_rn(blocks, m.reshape(rows, BLOCKS_PER_ROW, 1).expand_as(blocks))
    codes = e2m1_rn_satfinite(g)
    if canon:
        codes = canon_neg0_codes(codes)
    return pack_e2m1_codes(codes.reshape(rows, HIDDEN)), sf


def decode_accumulate(
    acc_f32: torch.Tensor,
    values_u8: torch.Tensor,
    scales_u8: torch.Tensor,
    row_scale,
) -> torch.Tensor:
    """``acc = fma(e2m1(q), e4m3(sf) * rs, acc)`` for one source.

    ``row_scale`` is a ``[R]`` float32 tensor (per-row ``rs``) or a Python float
    (the static scale of pre-quantised NVFP4 records).
    """
    rows = values_u8.shape[0]
    if isinstance(row_scale, float | int):
        rs = torch.full(
            (rows,), float(row_scale), dtype=torch.float32, device=values_u8.device
        )
    else:
        rs = row_scale.to(torch.float32)
    scale = f32_mul_rn(
        e4m3_to_f32(scales_u8), rs.reshape(rows, 1).expand(rows, BLOCKS_PER_ROW)
    )
    codes = unpack_e2m1_bytes(values_u8)
    v = e2m1_to_f32(codes).reshape(rows, BLOCKS_PER_ROW, BLOCK)
    s = scale.reshape(rows, BLOCKS_PER_ROW, 1).expand_as(v)
    return f32_fma_rn(v, s, acc_f32.reshape(rows, BLOCKS_PER_ROW, BLOCK)).reshape(
        rows, HIDDEN
    )


def _codec_sum(x_bf16_by_rank: Sequence[torch.Tensor]) -> torch.Tensor:
    acc = None
    for x in x_bf16_by_rank:
        f = x.to(torch.float32)
        inv, rs = row_scales(row_amax(f))
        q, sf = encode_rows(f, inv)
        if acc is None:
            acc = torch.zeros_like(f)
        acc = decode_accumulate(acc, q, sf, rs)
    return acc


def _twoshot_from_sum(acc: torch.Tensor) -> torch.Tensor:
    """Owner re-encode of the float32 sum with its own row amax; the output is the decode of that record."""
    inv2, rs2 = row_scales(row_amax(acc))
    q2, sf2 = encode_rows(acc, inv2)
    return decode_accumulate(torch.zeros_like(acc), q2, sf2, rs2).to(torch.bfloat16)


def reference_oneshot(x_bf16_by_rank: Sequence[torch.Tensor]) -> torch.Tensor:
    """World size 2: ``bf16(sum in rank order of decode(encode(x_r)))``."""
    return _codec_sum(x_bf16_by_rank).to(torch.bfloat16)


def reference_twoshot(x_bf16_by_rank: Sequence[torch.Tensor]) -> torch.Tensor:
    """World size 4: rank-order float32 codec sum -> owner re-encode -> decode -> bfloat16."""
    return _twoshot_from_sum(_codec_sum(x_bf16_by_rank))


def reference_allreduce(x_bf16_by_rank: Sequence[torch.Tensor]) -> torch.Tensor:
    """The bfloat16 result of ``fp4_allreduce`` for the partials of every rank (rank order)."""
    if len(x_bf16_by_rank) == 2:
        return reference_oneshot(x_bf16_by_rank)
    if len(x_bf16_by_rank) == 4:
        return reference_twoshot(x_bf16_by_rank)
    raise ValueError(f"fp4_allreduce supports 2 or 4 ranks, got {len(x_bf16_by_rank)}")


# ---------------------------------------------------------------------------
# Pre-quantised NVFP4 input records
# ---------------------------------------------------------------------------


def tensor_global_scale(x_bf16: torch.Tensor) -> float:
    """Static per-tensor scale of NVFP4 input records: ``fl32(amax(x) / 2688)``."""
    amax = x_bf16.to(torch.float32).abs().amax().reshape(1)
    return float(f32_div_rn(amax, torch.full_like(amax, ROW_AMAX_TARGET)).item())


def encode_nvfp4_records(x_bf16: torch.Tensor, global_scale: float) -> torch.Tensor:
    """Records ``[R, 3456]`` u8 (3072 value bytes, then 384 scale bytes) with ``inv = 1 / global_scale`` for every row."""
    f = x_bf16.to(torch.float32)
    rows = f.shape[0]
    one = torch.ones((rows,), dtype=torch.float32, device=f.device)
    inv = f32_div_rn(one, torch.full_like(one, float(global_scale)))
    q, sf = encode_rows(f, inv)
    return torch.cat((q, sf), dim=1).contiguous()


def split_nvfp4_records(records_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[R, 3456] -> (values [R, 3072], scales [R, 384])``."""
    return records_u8[:, :VALUE_BYTES], records_u8[:, VALUE_BYTES:]


def _records_sum(
    records_by_rank: Sequence[torch.Tensor], global_scales: Sequence[float]
) -> torch.Tensor:
    acc = None
    for records, scale in zip(records_by_rank, global_scales, strict=True):
        q, sf = split_nvfp4_records(records)
        if acc is None:
            acc = torch.zeros(
                (records.shape[0], HIDDEN), dtype=torch.float32, device=records.device
            )
        acc = decode_accumulate(acc, q, sf, float(scale))
    return acc


def reference_nvfp4_oneshot(
    records_by_rank: Sequence[torch.Tensor], global_scales: Sequence[float]
) -> torch.Tensor:
    """World size 2 with NVFP4 input records: decode-sum with ``rs = global_scales[r]``."""
    return _records_sum(records_by_rank, global_scales).to(torch.bfloat16)


def reference_nvfp4_twoshot(
    records_by_rank: Sequence[torch.Tensor], global_scales: Sequence[float]
) -> torch.Tensor:
    """World size 4 with NVFP4 input records: decode-sum, owner re-encode, decode."""
    return _twoshot_from_sum(_records_sum(records_by_rank, global_scales))


def reference_allreduce_nvfp4(
    records_by_rank: Sequence[torch.Tensor], global_scales: Sequence[float]
) -> torch.Tensor:
    """The bfloat16 result of ``fp4_allreduce_nvfp4`` for the records of every rank (rank order)."""
    if len(records_by_rank) == 2:
        return reference_nvfp4_oneshot(records_by_rank, global_scales)
    if len(records_by_rank) == 4:
        return reference_nvfp4_twoshot(records_by_rank, global_scales)
    raise ValueError(f"fp4_allreduce supports 2 or 4 ranks, got {len(records_by_rank)}")


# ---------------------------------------------------------------------------
# Deterministic inputs: per-source magnitude-scaled Gaussian plus 32 outliers
# ---------------------------------------------------------------------------


def _mul_u32(a: torch.Tensor, c: int) -> torch.Tensor:
    lo = a * (c & 0xFFFF)
    hi = (a * (c >> 16)) & 0xFFFF
    return (lo + (hi << 16)) & _U32_MASK


def _hash_u32(a: torch.Tensor) -> torch.Tensor:
    a = a ^ (a >> 16)
    a = _mul_u32(a, 0x7FEB352D)
    a = a ^ (a >> 15)
    a = _mul_u32(a, 0x846CA68B)
    return a ^ (a >> 16)


def _hash_u32_int(a: int) -> int:
    a &= _U32_MASK
    a ^= a >> 16
    a = (a * 0x7FEB352D) & _U32_MASK
    a ^= a >> 15
    a = (a * 0x846CA68B) & _U32_MASK
    return a ^ (a >> 16)


def generate_rank_input(
    rows: int, source_rank: int, seed: int, device, *, outliers: bool = True
) -> torch.Tensor:
    """bfloat16 ``[rows, 6144]`` partial of ``source_rank``.

    Box-Muller Gaussian scaled by ``SOURCE_MAGNITUDES[source_rank % 8]`` plus 32
    outliers of ``+-40 * magnitude`` at hashed positions (``outliers=False``
    omits them); deterministic in ``(rows, source_rank, seed)`` on any device.
    """
    n = rows * HIDDEN
    mag = SOURCE_MAGNITUDES[source_rank & 7]
    i = torch.arange(n, dtype=torch.int64, device=device)
    t = (i * 2 + 1 + (i >> 31)) & _U32_MASK
    h1 = _hash_u32((int(seed) & _U32_MASK) ^ _hash_u32(t))
    h2 = _hash_u32(h1 ^ 0x9E3779B9 ^ (int(source_rank) & _U32_MASK))
    u1 = (h1 >> 8).to(torch.float32) * (1.0 / 16777216.0) + 1e-7
    u2 = (h2 >> 8).to(torch.float32) * (1.0 / 16777216.0)
    x = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2) * mag
    x = x.to(torch.bfloat16)
    if not outliers:
        return x.reshape(rows, HIDDEN)
    positions: list[int] = []
    values: list[float] = []
    for k in range(OUTLIER_COUNT):
        h = _hash_u32_int(int(seed) * 7919 + 104729 * (k + 1))
        positions.append((h * 2654435761) % n)
        values.append(
            (OUTLIER_MAGNITUDE if _hash_u32_int(h) & 1 else -OUTLIER_MAGNITUDE) * mag
        )
    x[torch.tensor(positions, dtype=torch.int64, device=device)] = torch.tensor(
        values, dtype=torch.float32, device=device
    ).to(torch.bfloat16)
    return x.reshape(rows, HIDDEN)


def generate_residual(rows: int, seed: int, device) -> torch.Tensor:
    """Residual stream of one call: bfloat16 ``[rows, 6144]``, the same on every rank (seeded by the call, not the rank)."""
    return generate_rank_input(
        rows, RESIDUAL_SOURCE_RANK, (int(seed) ^ RESIDUAL_SEED_XOR) & _U32_MASK, device
    )


def generate_weight(seed: int, device) -> torch.Tensor:
    """Norm weight: bfloat16 ``[6144]`` of magnitude 0.125 without outliers (``1 + weight`` stays in ``[0.4, 1.6]``)."""
    return generate_rank_input(
        1,
        WEIGHT_SOURCE_RANK,
        (WEIGHT_SEED ^ int(seed)) & _U32_MASK,
        device,
        outliers=False,
    )[0]


# ---------------------------------------------------------------------------
# Fused residual + RMSNorm epilogue (family "rmsnorm")
# ---------------------------------------------------------------------------


def rsqrt_f32(x_f32: torch.Tensor) -> tuple[torch.Tensor, str]:
    """``rsqrt`` of a float32 tensor and the form used.

    On CUDA tensors ``torch.rsqrt`` evaluates ``rsqrtf`` = the ``rsqrt.approx.f32``
    instruction, which agrees with the kernels' ``rsqrt.approx.ftz.f32`` on the
    normal arguments the norm produces (``mean(z^2) + eps >= eps``).  On CPU
    tensors the IEEE float64 ``1 / sqrt`` rounded once to float32 is used.
    """
    if x_f32.is_cuda:
        return torch.rsqrt(x_f32), RSQRT_FORM_DEVICE
    return (1.0 / torch.sqrt(x_f32.to(torch.float64))).to(
        torch.float32
    ), RSQRT_FORM_IEEE


@dataclass
class FusedNormReference:
    """Reference of one fused call (tensors ``[rows, 6144]`` unless noted)."""

    y_f32: torch.Tensor  # (z * rinv) * (weight_bias + weight) in float32
    y_bf16: torch.Tensor  # bf16(y_f32): the comparison target of the bfloat16 output
    residual_out: torch.Tensor  # bf16(z), compared bitwise
    ss: torch.Tensor  # [rows] float32 sum of z^2 (exact float64 sum rounded once)
    rinv: torch.Tensor  # [rows] float32 rsqrt(ss / 6144 + eps)
    weight_bias: float
    eps: float
    rsqrt_form: str


def fused_norm_from_sum(
    acc_f32: torch.Tensor,
    residual_bf16: torch.Tensor,
    weight_bf16: torch.Tensor,
    *,
    weight_bias: float = NORM_WEIGHT_BIAS,
    eps: float = NORM_EPS,
) -> FusedNormReference:
    """``z = acc + float(residual)``; ``residual_out = bf16(z)``; ``ss = sum z^2``; ``rinv = rsqrt(ss / 6144 + eps)``;
    ``y = (z * rinv) * (weight_bias + float(weight))`` -- one float32 rounding per operation; ``ss`` is the exact
    float64 sum rounded once (the best float32 value; a kernel's summation order deviates by a few float32 ulps)."""
    rows = acc_f32.shape[0]
    z = f32_add_rn(acc_f32, residual_bf16.to(torch.float32))
    residual_out = z.to(torch.bfloat16)
    z64 = z.to(torch.float64)
    ss = (z64 * z64).sum(dim=1).to(torch.float32)
    arg = f32_add_rn(
        f32_div_rn(ss, torch.full_like(ss, float(HIDDEN))),
        torch.full_like(ss, float(eps)),
    )
    rinv, form = rsqrt_f32(arg)
    scale = f32_add_rn(
        torch.full((HIDDEN,), float(weight_bias), dtype=torch.float32, device=z.device),
        weight_bf16.to(torch.float32),
    )
    y = f32_mul_rn(
        f32_mul_rn(z, rinv.reshape(rows, 1).expand(rows, HIDDEN)),
        scale.reshape(1, HIDDEN).expand(rows, HIDDEN),
    )
    return FusedNormReference(
        y_f32=y,
        y_bf16=y.to(torch.bfloat16),
        residual_out=residual_out,
        ss=ss,
        rinv=rinv,
        weight_bias=float(weight_bias),
        eps=float(eps),
        rsqrt_form=form,
    )


def reference_fused_rmsnorm(
    x_bf16_by_rank: Optional[Sequence[torch.Tensor]] = None,
    residual_bf16: Optional[torch.Tensor] = None,
    weight_bf16: Optional[torch.Tensor] = None,
    *,
    records_by_rank: Optional[Sequence[torch.Tensor]] = None,
    global_scales: Optional[Sequence[float]] = None,
    weight_bias: float = NORM_WEIGHT_BIAS,
    eps: float = NORM_EPS,
) -> FusedNormReference:
    """Reference of ``fp4_allreduce_rmsnorm`` (bfloat16 partials) or ``fp4_allreduce_rmsnorm_nvfp4`` (records).

    The float32 one-shot codec sum of the two ranks (rank order) feeds
    :func:`fused_norm_from_sum`; ``weight_bias`` other than 1 builds a wrong
    reference for the negative control.
    """
    if residual_bf16 is None or weight_bf16 is None:
        raise ValueError("residual_bf16 and weight_bf16 are required")
    if tuple(weight_bf16.shape) != (HIDDEN,):
        raise ValueError(f"weight must be a [{HIDDEN}] bfloat16 row")
    if records_by_rank is not None:
        if global_scales is None:
            raise ValueError("records_by_rank needs global_scales")
        if len(records_by_rank) != 2:
            raise ValueError("the fused-norm family is the two-rank one-shot exchange")
        acc = _records_sum(records_by_rank, global_scales)
    else:
        if x_bf16_by_rank is None:
            raise ValueError("x_bf16_by_rank or records_by_rank is required")
        if len(x_bf16_by_rank) != 2:
            raise ValueError("the fused-norm family is the two-rank one-shot exchange")
        acc = _codec_sum(x_bf16_by_rank)
    return fused_norm_from_sum(
        acc, residual_bf16, weight_bf16, weight_bias=weight_bias, eps=eps
    )


def bf16_ulp_distance(a_bf16: torch.Tensor, b_bf16: torch.Tensor) -> torch.Tensor:
    """Elementwise distance of two bfloat16 tensors in bfloat16 steps (signed-magnitude key; ``+-0`` coincide)."""

    def key(t: torch.Tensor) -> torch.Tensor:
        bits = t.view(torch.int16).to(torch.int32) & 0xFFFF
        mag = bits & 0x7FFF
        return torch.where((bits & 0x8000) != 0, -mag, mag)

    return (key(a_bf16) - key(b_bf16)).abs()


def check_normalised_output(
    y_bf16: torch.Tensor,
    ref: FusedNormReference,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict:
    """Compare a bfloat16 ``y`` with the reference: tolerance violations (the acceptance), non-finite count and the
    bfloat16 ulp profile (a correct kernel differs from ``bf16(y_f32)`` by at most one step on a small fraction of the
    elements, from its own ``z^2`` summation order)."""
    y = y_bf16.to(torch.float32)
    bound = atol + rtol * ref.y_f32.abs()
    violations = int(((y - ref.y_f32).abs() > bound).sum().item())
    non_finite = int((~torch.isfinite(y)).sum().item())
    ulp = bf16_ulp_distance(y_bf16, ref.y_bf16)
    n = int(ulp.numel())
    return {
        "elements": n,
        "violations": violations,
        "non_finite": non_finite,
        "max_ulp": int(ulp.max().item()) if n else 0,
        "ulp1_fraction": (float((ulp == 1).sum().item()) / n) if n else 0.0,
        "passed": violations == 0 and non_finite == 0,
    }


def output_global_scale(y_bf16: torch.Tensor) -> torch.Tensor:
    """Float32 ``[1]`` device tensor ``fl32(2688 / amax(y))``: the input global scale of ``fp4_quantize`` for ``y``."""
    amax = y_bf16.to(torch.float32).abs().amax().reshape(1)
    return f32_div_rn(torch.full_like(amax, ROW_AMAX_TARGET), amax).contiguous()


def padded_scale_rows(rows: int) -> int:
    """Rows covered by a swizzled scale tensor (``rows`` rounded up to a multiple of 128)."""
    return (int(rows) + SCALE_ROW_TILE - 1) // SCALE_ROW_TILE * SCALE_ROW_TILE


def unswizzle_scales_128x4(
    flat_u8: torch.Tensor, rows: int, scales_per_row: int
) -> torch.Tensor:
    """``[padded rows * scales_per_row]`` swizzled bytes -> ``[rows, scales_per_row]`` block-order scales.

    The 128x4 layout stores scale ``(m, k)`` at ``(m // 128) * kt * 512 + (k // 4) * 512
    + (m % 32) * 16 + ((m % 128) // 32) * 4 + (k % 4)`` with ``kt = scales_per_row // 4``.
    """
    mt = padded_scale_rows(rows) // SCALE_ROW_TILE
    kt = int(scales_per_row) // 4
    need = mt * kt * 512
    if flat_u8.numel() < need:
        raise ValueError(
            f"swizzled scale tensor has {flat_u8.numel()} bytes, need {need}"
        )
    tiles = flat_u8.reshape(-1)[:need].reshape(mt, kt, 32, 4, 4)
    return (
        tiles.permute(0, 3, 2, 1, 4)
        .reshape(mt * SCALE_ROW_TILE, kt * 4)[: int(rows), : int(scales_per_row)]
        .contiguous()
    )


def nvfp4_output_reference(
    y_bf16: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``fp4_quantize`` of ``y`` (16-value E4M3 block scales, 128x4 swizzle, candidate search off):
    ``(values u8 [rows, 3072], scales u8 [padded rows * 384])``."""
    from flashinfer import fp4_quantize

    rows = int(y_bf16.shape[0])
    values, scales = fp4_quantize(
        y_bf16.contiguous(),
        global_scale=global_scale.reshape(1).to(torch.float32),
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
        is_sf_8x4_layout=False,
        is_global_scale_inversed=False,
        backend="cuda",
        nvfp4_4over6=None,
    )
    values = values.view(torch.uint8).reshape(rows, VALUE_BYTES).contiguous()
    need = padded_scale_rows(rows) * NVFP4_OUTPUT_SCALES_PER_ROW
    scales = scales.view(torch.uint8).reshape(-1)
    if scales.numel() < need:
        raise RuntimeError(
            f"fp4_quantize returned {scales.numel()} scale bytes, expected >= {need}"
        )
    return values, scales[:need].contiguous()


def mxfp8_output_reference(y_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``mxfp8_quantize`` of ``y`` (32-value UE8M0 block scales, 128x4 swizzle):
    ``(values u8 [rows, 6144], scales u8 [padded rows * 192])``."""
    from flashinfer import mxfp8_quantize

    rows = int(y_bf16.shape[0])
    values, scales = mxfp8_quantize(
        y_bf16.contiguous(), is_sf_swizzled_layout=True, alignment=32, backend="cuda"
    )
    values = values.view(torch.uint8).reshape(rows, HIDDEN).contiguous()
    need = padded_scale_rows(rows) * MXFP8_OUTPUT_SCALES_PER_ROW
    scales = scales.view(torch.uint8).reshape(-1)
    if scales.numel() < need:
        raise RuntimeError(
            f"mxfp8_quantize returned {scales.numel()} scale bytes, expected >= {need}"
        )
    return values, scales[:need].contiguous()
