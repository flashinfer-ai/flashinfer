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

Exact float32 reference of the Cake NVFP4-transport all-reduce family.

Codec v2 of the NVFP4 all-reduce baseline (https://github.com/fans-nv/fp4-all-reduce,
commit 2878bf5, ``include/fp4_codec.cuh``; its float32 reference
``harness/ref.cuh``), reproduced operation by operation: every float32
multiply, divide, add and fused multiply-add is one IEEE round-to-nearest-even
rounding (products and quotients of two float32 values are exact in float64;
the FMA goes through float64 round-to-odd), the E4M3 / E2M1 conversions are the
saturating ``cvt.rn.satfinite`` forms, and E2M1 ``-0`` is canonicalised to
``+0`` on the wire.

* **Rows.** A row of 6144 values is encoded with its row scale ``rs = amax / 2688``
  (``inv = 2688 / amax``, 0 for an all-zero row); per block of 16 the E4M3 scale is
  ``sf = e4m3(div_rn(mul_rn(blockmax, inv), 6))`` and the values are
  ``e2m1(mul_rn(x, div_rn(inv, sf)))`` (0 when ``sf`` is 0).  Decoding
  accumulates ``fma(e2m1(q), mul_rn(e4m3(sf), rs), acc)`` in rank order from
  ``+0``.
* **One-shot (world size 2).** ``out = bf16(acc)`` over both ranks' records.
* **Two-shot (world size 4).** The owner re-encodes the float32 sum under the
  sum's own row scale; every rank's output is ``bf16`` of the decode of that one
  record.
* **NVFP4 input.** Rank ``r`` holds the records ``[rows, 3456]`` (3072 value bytes
  then 384 scale bytes) a producing GEMM emitted under the static per-rank scale
  ``gs_r = amax(x_r) / 2688`` (``inv = 1 / gs_r``); the decode uses ``gs_r`` as
  the row scale.
* **Fused residual + Gemma RMSNorm (world size 2).** ``z = acc + f32(residual)``,
  ``residual_out = bf16(z)``, ``ss = sum z^2`` in the kernel's float32 order (the
  program's ``ss_order`` -- the registry record says which chunk order a target's
  build sums in; :func:`sum_of_squares`), ``rinv = rsqrt(ss / 6144 + eps)``,
  ``y = f32(bf16(z)) * rinv * (weight_bias + f32(weight))`` (the baseline
  multiplies the BF16-rounded ``z``), written as BF16
  or quantised from ``bf16(y)`` with the arithmetic and layouts of FlashInfer's
  ``fp4_quantize`` (16-value E4M3 block scales) / ``mxfp8_quantize`` (32-value
  UE8M0 block scales), the scales in the swizzled 128 x 4 layout over
  ``round_up(rows, 128)`` rows.  The kernels take ``rsqrt.approx.ftz`` and
  ``rcp.approx.ftz`` where this reference takes the IEEE values, so ``y`` is
  compared within ``atol = rtol = 1e-2``, ``residual_out`` bit for bit, and the
  quantised outputs exactly outside the E2M1 rounding ties (cleared on both sides
  by :func:`canonical`) but for at most :data:`QUANTISED_MISMATCH_ALLOWANCE` of
  the elements.

Pure torch; every function works on CPU and CUDA tensors.  No FlashInfer kernel
is used as a reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

HIDDEN = 6144
BLOCK = 16
BLOCKS_PER_ROW = HIDDEN // BLOCK  # 384
VALUE_BYTES = HIDDEN // 2  # 3072 packed E2M1 bytes per row
SCALE_BYTES = BLOCKS_PER_ROW  # 384 E4M3 bytes per row
RECORD_BYTES = VALUE_BYTES + SCALE_BYTES  # 3456
ROW_AMAX_TARGET = 2688.0  # 448 x 6: the largest encodable product
E4M3_MAX = 448.0
E2M1_VALUES = (
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
WORLD_SIZES = (2, 4)
TWOSHOT_WORLD_SIZE = 4
TP4_ROW_MULTIPLE = 4

# Fused residual + Gemma RMSNorm family.
EPS = 1e-6
WEIGHT_BIAS = 1.0  # Gemma: y = bf16(z) * rinv * (weight_bias + weight)
NEGATIVE_CONTROL_WEIGHT_BIAS = (
    1.0 + 2.0**-7
)  # a reference with this bias must fail the y tolerance
Y_ATOL = 1e-2
Y_RTOL = 1e-2
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
NVFP4_SCALES_PER_ROW = HIDDEN // NVFP4_BLOCK  # 384 E4M3 scales per row
MXFP8_SCALES_PER_ROW = HIDDEN // MXFP8_BLOCK  # 192 UE8M0 scales per row
NVFP4_VALUE_BYTES = VALUE_BYTES  # 3072 packed E2M1 bytes per row
MXFP8_VALUE_BYTES = HIDDEN  # one E4M3 byte per value
SCALE_ROW_TILE = 128  # swizzled scale tensors cover rows padded to a multiple of 128
E2M1_TIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)  # the midpoints of the E2M1 grid
TIE_ULPS = (
    8  # float32 ulps of the scaled product the approximate reciprocals can move it
)
QUANTISED_MISMATCH_ALLOWANCE = (
    1e-4  # fraction of quantised elements an rsqrt.approx ulp may move
)


# ---------------------------------------------------------------------------
# Exact float32 primitives (one IEEE rounding each)
# ---------------------------------------------------------------------------


def f32_mul_rn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``mul.rn.f32``: the float64 product of two float32 values is exact; one rounding to float32."""
    return (a.to(torch.float64) * b.to(torch.float64)).to(torch.float32)


def f32_div_rn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``div.rn.f32``: the float64 quotient rounded to float32 is the correctly rounded float32 quotient."""
    return (a.to(torch.float64) / b.to(torch.float64)).to(torch.float32)


def f32_add_rn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``add.rn.f32``: the float64 sum of two float32 values is exact; one rounding to float32."""
    return (a.to(torch.float64) + b.to(torch.float64)).to(torch.float32)


def f32_fma_rn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``fma.rn.f32`` through float64 round-to-odd.

    ``a * b`` is exact in float64.  ``s = fl64(p + c)`` with its exact TwoSum
    error; when the error is non-zero and ``s`` has an even last bit, ``s`` moves
    one float64 ulp toward the exact value (round-to-odd), and rounding that
    float64 to float32 is then the single correct rounding of ``a * b + c``.
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
    return torch.where(adjust, torch.nextafter(s, toward), s).to(torch.float32)


def e4m3_rn_satfinite(x: torch.Tensor) -> torch.Tensor:
    """``cvt.rn.satfinite.e4m3x2.f32``: finite ``|x| > 448`` saturates to ``+-448``; NaN stays NaN (0x7F).

    The rounding is torch's round-to-nearest-even ``float8_e4m3fn`` conversion.
    """
    clamped = torch.where(torch.isnan(x), x, x.clamp(min=-E4M3_MAX, max=E4M3_MAX))
    return clamped.to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_to_f32(codes: torch.Tensor) -> torch.Tensor:
    """E4M3 code -> float32 (exact)."""
    return codes.view(torch.float8_e4m3fn).to(torch.float32)


def e2m1_rn_satfinite(x: torch.Tensor) -> torch.Tensor:
    """``cvt.rn.satfinite.e2m1x2.f32``: round-to-nearest-even on the E2M1 grid, codes 0..15 (bit 3 = sign).

    Grid ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}``; ``|x| >= 6`` saturates; NaN is ``+6``
    (code 7); a negative value that rounds to zero is ``-0`` (code 8).
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
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=codes.device)
    return table[codes.to(torch.int64)]


def canon_neg0(codes: torch.Tensor) -> torch.Tensor:
    """The wire canonicalisation: E2M1 ``-0`` (code 8) becomes ``+0`` (code 0)."""
    return torch.where(codes == 8, torch.zeros_like(codes), codes)


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """``[..., 2k]`` codes -> ``[..., k]`` bytes, element ``2i`` in the low nibble."""
    even = codes[..., 0::2].to(torch.uint8)
    odd = codes[..., 1::2].to(torch.uint8)
    return even | (odd << 4)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """``[..., k]`` bytes -> ``[..., 2k]`` codes (inverse of :func:`pack_e2m1`)."""
    return torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )


# ---------------------------------------------------------------------------
# Codec v2 rows
# ---------------------------------------------------------------------------


def row_scales(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``inv = amax > 0 ? div_rn(2688, amax) : 0`` and ``rs = div_rn(amax, 2688)`` of row maxima ``[R]``."""
    target = torch.full_like(amax, ROW_AMAX_TARGET)
    inv = torch.where(amax > 0, f32_div_rn(target, amax), torch.zeros_like(amax))
    return inv, f32_div_rn(amax, target)


def encode_rows(
    values: torch.Tensor, inv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float32 rows ``[R, 6144]`` with the row scales ``inv`` ``[R]`` -> (values ``[R, 3072]`` u8, scales ``[R, 384]`` u8).

    Per block of 16: ``sf = e4m3(div_rn(mul_rn(blockmax, inv), 6))``,
    ``m = sf > 0 ? div_rn(inv, sf) : 0``, ``q = e2m1(mul_rn(x, m))`` with ``-0``
    canonicalised to ``+0``.
    """
    rows = values.shape[0]
    blocks = values.reshape(rows, BLOCKS_PER_ROW, BLOCK)
    bm = blocks.abs().amax(dim=-1)
    inv_b = inv.reshape(rows, 1).expand(rows, BLOCKS_PER_ROW)
    sv = f32_div_rn(f32_mul_rn(bm, inv_b), torch.full_like(bm, 6.0))
    sf = e4m3_rn_satfinite(sv)
    sff = e4m3_to_f32(sf)
    m = torch.where(sff > 0, f32_div_rn(inv_b, sff), torch.zeros_like(sff))
    g = f32_mul_rn(blocks, m.reshape(rows, BLOCKS_PER_ROW, 1).expand_as(blocks))
    codes = canon_neg0(e2m1_rn_satfinite(g))
    return pack_e2m1(codes.reshape(rows, HIDDEN)), sf


def decode_accumulate(
    acc: torch.Tensor,
    values_u8: torch.Tensor,
    scales_u8: torch.Tensor,
    rs: torch.Tensor,
) -> torch.Tensor:
    """``acc = fma(e2m1(q), mul_rn(e4m3(sf), rs), acc)`` for one source; ``rs`` is ``[R]`` float32."""
    rows = values_u8.shape[0]
    scale = f32_mul_rn(
        e4m3_to_f32(scales_u8),
        rs.to(torch.float32).reshape(rows, 1).expand(rows, BLOCKS_PER_ROW),
    )
    v = e2m1_to_f32(unpack_e2m1(values_u8)).reshape(rows, BLOCKS_PER_ROW, BLOCK)
    s = scale.reshape(rows, BLOCKS_PER_ROW, 1).expand_as(v)
    return f32_fma_rn(v, s, acc.reshape(rows, BLOCKS_PER_ROW, BLOCK)).reshape(
        rows, HIDDEN
    )


def encode_bf16(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A bfloat16 partial ``[R, 6144]`` to its wire record: (values, scales, row scales ``[R]``)."""
    f = x.to(torch.float32)
    inv, rs = row_scales(f.abs().amax(dim=-1))
    q, sf = encode_rows(f, inv)
    return q, sf, rs


def global_scale(x: torch.Tensor) -> torch.Tensor:
    """The static per-tensor NVFP4 input scale ``gs = div_rn(amax(x), 2688)``: a float32 ``[1]`` tensor."""
    amax = x.to(torch.float32).abs().amax().reshape(1)
    return f32_div_rn(amax, torch.full_like(amax, ROW_AMAX_TARGET))


def encode_records(x: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    """The pre-quantised input records ``[R, 3456]`` u8 of ``x`` under the static scale ``gs``: ``inv = div_rn(1, gs)`` for every row."""
    f = x.to(torch.float32)
    rows = f.shape[0]
    one = torch.ones((rows,), dtype=torch.float32, device=f.device)
    inv = f32_div_rn(one, gs.to(torch.float32).reshape(-1)[:1].expand(rows))
    q, sf = encode_rows(f, inv)
    return torch.cat((q, sf), dim=1).contiguous()


def split_records(records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[R, 3456]`` records -> (values ``[R, 3072]``, scales ``[R, 384]``)."""
    if records.dim() != 2 or records.shape[1] != RECORD_BYTES:
        raise ValueError(
            f"records must be [rows, {RECORD_BYTES}] bytes, got {tuple(records.shape)}"
        )
    return records[:, :VALUE_BYTES], records[:, VALUE_BYTES:]


def sources_bf16(
    partials: Sequence[torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """The (values, scales, row scales) records of every rank's bfloat16 partial, in rank order."""
    return [encode_bf16(x) for x in partials]


def sources_records(
    records: Sequence[torch.Tensor], global_scales: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """The (values, scales, row scales) sources of every rank's NVFP4 records; ``global_scales[r]`` is rank ``r``'s row scale."""
    gs = global_scales.to(torch.float32).reshape(-1)
    if gs.numel() < len(records):
        raise ValueError(
            f"global_scales holds {gs.numel()} scales for {len(records)} ranks"
        )
    sources = []
    for rank, rec in enumerate(records):
        q, sf = split_records(rec)
        sources.append((q, sf, gs[rank].reshape(1).expand(rec.shape[0])))
    return sources


def oneshot_sum(
    sources: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """The float32 rank-ordered decode-sum of (values, scales, row scales) records, from ``+0``."""
    acc = None
    for q, sf, rs in sources:
        if acc is None:
            acc = torch.zeros(
                (q.shape[0], HIDDEN), dtype=torch.float32, device=q.device
            )
        acc = decode_accumulate(acc, q, sf, rs)
    if acc is None:
        raise ValueError("at least one source is required")
    return acc


def twoshot(acc: torch.Tensor) -> torch.Tensor:
    """The owner's re-encode of the float32 sum under its own row scale and the decode of that one record, from ``+0``."""
    inv2, rs2 = row_scales(acc.abs().amax(dim=-1))
    q2, sf2 = encode_rows(acc, inv2)
    return decode_accumulate(torch.zeros_like(acc), q2, sf2, rs2)


def _check_world(count: int) -> None:
    if count not in WORLD_SIZES:
        raise ValueError(
            f"the NVFP4-transport all-reduce runs on {WORLD_SIZES} ranks, got {count}"
        )


def _finish(acc: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == TWOSHOT_WORLD_SIZE:
        if acc.shape[0] % TP4_ROW_MULTIPLE:
            raise ValueError(
                f"the two-shot all-reduce needs a multiple of {TP4_ROW_MULTIPLE} rows, got {acc.shape[0]}"
            )
        acc = twoshot(acc)
    return acc.to(torch.bfloat16)


def reference_allreduce(partials: Sequence[torch.Tensor]) -> torch.Tensor:
    """The bfloat16 all-reduce of every rank's bfloat16 partial (rank order): one-shot for 2 ranks, two-shot for 4."""
    _check_world(len(partials))
    return _finish(oneshot_sum(sources_bf16(partials)), len(partials))


def reference_allreduce_nvfp4(
    records: Sequence[torch.Tensor], global_scales: torch.Tensor
) -> torch.Tensor:
    """The bfloat16 all-reduce of every rank's NVFP4 records under the static scales ``global_scales`` ``[world_size]``."""
    _check_world(len(records))
    return _finish(oneshot_sum(sources_records(records, global_scales)), len(records))


# ---------------------------------------------------------------------------
# Deterministic inputs
# ---------------------------------------------------------------------------


def partials(rows: int, seed: int, world_size: int, device) -> list[torch.Tensor]:
    """Every rank's bfloat16 partial ``[rows, 6144]`` from one seed (rank ``r`` from ``seed + 104729 r``).

    ``randn x 2`` with, per row, one element in a hundred scaled by 32 (block
    scales that saturate the E4M3 range), every 97th row all zero (the
    ``inv = 0`` path) and every 89th row scaled by ``2^-20`` (small row scales).
    """
    xs = []
    for rank in range(world_size):
        g = torch.Generator().manual_seed(int(seed) + 104729 * rank)
        x = torch.randn((rows, HIDDEN), generator=g) * 2.0
        picks = torch.randint(0, HIDDEN, (rows, HIDDEN // 100 + 1), generator=g)
        x.scatter_(1, picks, x.gather(1, picks) * 32.0)
        x[96::97] = 0.0
        x[88::89] *= 2.0**-20
        xs.append(x.to(torch.bfloat16).to(device))
    return xs


def launch_seed(seed: int, launch_index: int) -> int:
    """The seed of launch ``launch_index`` of a case: ``seed + 7 * launch_index``."""
    return int(seed) + 7 * int(launch_index)


def residual(rows: int, seed: int, device) -> torch.Tensor:
    """The bfloat16 residual ``[rows, 6144]`` of one launch (``randn``), the same on every rank."""
    g = torch.Generator().manual_seed(int(seed) + 31)
    return torch.randn((rows, HIDDEN), generator=g).to(torch.bfloat16).to(device)


def weight(seed: int, device) -> torch.Tensor:
    """The bfloat16 norm weight ``[6144]`` of a case (``0.5 randn``)."""
    g = torch.Generator().manual_seed(int(seed) + 17)
    return (torch.randn((HIDDEN,), generator=g) * 0.5).to(torch.bfloat16).to(device)


def records_of(
    xs: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Every rank's NVFP4 records and the ``[world_size]`` float32 tensor of their static scales."""
    scales = [global_scale(x) for x in xs]
    return [encode_records(x, g) for x, g in zip(xs, scales, strict=True)], torch.cat(
        scales
    ).contiguous()


# ---------------------------------------------------------------------------
# Fused residual + Gemma RMSNorm
# ---------------------------------------------------------------------------


# The fused-norm programs' ``ss_order`` (a registry key): the chunk order of each lane's pass-1 chains.
SS_ORDERS = ("natural", "perm")


def sum_of_squares(z: torch.Tensor, *, order: str = "natural") -> torch.Tensor:
    """``ss`` of each row as the kernel accumulates it.

    Lane ``l`` of the warp owns the pairs ``l + 32 j`` (elements ``1024 j + 32 l +
    8 k + 2 i + p``), accumulates the even and the odd elements in two
    ``fma.rn.f32`` chains in pair, chunk, word order, adds the two, and the 32
    lane sums meet in the ``add.rn.f32`` butterfly over lane masks 16, 8, 4, 2,
    1 (every lane ends with the same bits).  ``order`` is the program's
    ``ss_order``: ``"natural"`` takes a pair's four chunks in the row's order
    ``k = 0..3``; ``"perm"`` takes them in the lane's loaded order ``f ^ k`` with
    ``f = (l >> 1) & 3`` (the same squares in another float32 order, so ``ss``
    may differ in its last ulp).
    """
    if order not in SS_ORDERS:
        raise ValueError(f"order must be one of {SS_ORDERS}, got {order!r}")
    rows = z.shape[0]
    # [rows, lane, pair, chunk, word, parity]
    chains = z.reshape(rows, 6, 32, 4, 4, 2).permute(0, 2, 1, 3, 4, 5)
    if order == "perm":
        lanes = torch.arange(32, device=z.device)
        # slot k of lane l holds chunk f ^ k
        chunk = ((lanes >> 1) & 3).reshape(32, 1) ^ torch.arange(
            4, device=z.device
        ).reshape(1, 4)
        chains = torch.gather(
            chains, 3, chunk.reshape(1, 32, 1, 4, 1, 1).expand(rows, 32, 6, 4, 4, 2)
        )
    chains = chains.reshape(rows, 32, 96, 2)
    acc = torch.zeros((rows, 32, 2), dtype=torch.float32, device=z.device)
    for step in range(96):
        v = chains[:, :, step, :]
        acc = f32_fma_rn(v, v, acc)
    ss = f32_add_rn(acc[:, :, 0], acc[:, :, 1])
    lanes = torch.arange(32, device=z.device)
    for mask in (16, 8, 4, 2, 1):
        ss = f32_add_rn(ss, ss[:, lanes ^ mask])
    return ss[:, 0]


@dataclass
class FusedNormReference:
    """Reference outputs of one fused call: ``y`` (bfloat16 ``[rows, 6144]``), ``residual_out`` (bfloat16, bitwise),
    ``ss`` and ``rinv`` (float32 ``[rows]``), the scalars and the sum-of-squares order the reference was built with."""

    y: torch.Tensor
    residual_out: torch.Tensor
    ss: torch.Tensor
    rinv: torch.Tensor
    eps: float
    weight_bias: float
    ss_order: str


def norm(
    acc: torch.Tensor,
    residual_bf16: torch.Tensor,
    weight_bf16: torch.Tensor,
    eps: float = EPS,
    weight_bias: float = WEIGHT_BIAS,
    *,
    ss_order: str = "natural",
) -> FusedNormReference:
    """The fused epilogue of the float32 sum ``acc``: the kernel's arithmetic with the IEEE ``rsqrt`` (float64, rounded once)
    in place of ``rsqrt.approx.ftz``; ``ss_order`` is the program's chunk order of the sum of squares
    (:func:`sum_of_squares`)."""
    rows = acc.shape[0]
    z = f32_add_rn(acc, residual_bf16.to(torch.float32))
    residual_out = z.to(torch.bfloat16)
    zb = residual_out.to(torch.float32)
    ss = sum_of_squares(z, order=ss_order)
    arg = f32_add_rn(
        f32_div_rn(ss, torch.full_like(ss, float(HIDDEN))),
        torch.full_like(ss, float(eps)),
    )
    rinv = (1.0 / torch.sqrt(arg.to(torch.float64))).to(torch.float32)
    wb = f32_add_rn(
        torch.full(
            (HIDDEN,), float(weight_bias), dtype=torch.float32, device=acc.device
        ),
        weight_bf16.to(torch.float32),
    )
    y = f32_mul_rn(
        f32_mul_rn(zb, rinv.reshape(rows, 1).expand(rows, HIDDEN)),
        wb.reshape(1, HIDDEN).expand(rows, HIDDEN),
    )
    return FusedNormReference(
        y=y.to(torch.bfloat16),
        residual_out=residual_out,
        ss=ss,
        rinv=rinv,
        eps=float(eps),
        weight_bias=float(weight_bias),
        ss_order=ss_order,
    )


def reference_fused_rmsnorm(
    partials_bf16: Optional[Sequence[torch.Tensor]] = None,
    residual_bf16: Optional[torch.Tensor] = None,
    weight_bf16: Optional[torch.Tensor] = None,
    *,
    records: Optional[Sequence[torch.Tensor]] = None,
    global_scales: Optional[torch.Tensor] = None,
    eps: float = EPS,
    weight_bias: float = WEIGHT_BIAS,
    ss_order: str = "natural",
) -> FusedNormReference:
    """Reference of the fused call on two ranks' bfloat16 partials, or (``records`` + ``global_scales``) NVFP4 records.

    A ``weight_bias`` other than 1 or another ``eps`` builds the reference of a
    call with those runtime scalars (or a deliberately wrong one for a negative
    control).  ``ss_order`` is the ``ss_order`` of the program the call runs
    (its registry record): the chunk order of the float32 sum of squares.
    """
    if residual_bf16 is None or weight_bf16 is None:
        raise ValueError("residual_bf16 and weight_bf16 are required")
    if tuple(weight_bf16.shape) != (HIDDEN,):
        raise ValueError(
            f"weight must be a [{HIDDEN}] row, got {tuple(weight_bf16.shape)}"
        )
    if records is not None:
        if global_scales is None:
            raise ValueError("records need global_scales")
        if len(records) != 2:
            raise ValueError("the fused-norm family is the two-rank one-shot exchange")
        acc = oneshot_sum(sources_records(records, global_scales))
    else:
        if partials_bf16 is None:
            raise ValueError("partials_bf16 or records is required")
        if len(partials_bf16) != 2:
            raise ValueError("the fused-norm family is the two-rank one-shot exchange")
        acc = oneshot_sum(sources_bf16(partials_bf16))
    if residual_bf16.shape != acc.shape:
        raise ValueError(
            f"residual must be {tuple(acc.shape)}, got {tuple(residual_bf16.shape)}"
        )
    return norm(
        acc,
        residual_bf16,
        weight_bf16,
        eps=eps,
        weight_bias=weight_bias,
        ss_order=ss_order,
    )


# ---------------------------------------------------------------------------
# FlashInfer-layout quantisers of bf16(y)
# ---------------------------------------------------------------------------


def sf_rows(rows: int) -> int:
    """Rows of a swizzled scale tensor: FlashInfer pads to a multiple of 128."""
    return (int(rows) + SCALE_ROW_TILE - 1) // SCALE_ROW_TILE * SCALE_ROW_TILE


def sf_offsets(rows: int, blocks: int, device) -> torch.Tensor:
    """The byte offset of each (row, block) scale in FlashInfer's swizzled 128 x 4 layout: ``[rows, blocks]`` int64.

    ``(row / 128) * ktiles * 512 + (block / 4) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 + block % 4``
    with ``ktiles = blocks / 4``.
    """
    ktiles = blocks // 4
    r = torch.arange(rows, dtype=torch.int64, device=device).reshape(rows, 1)
    b = torch.arange(blocks, dtype=torch.int64, device=device).reshape(1, blocks)
    return (
        (r // 128) * (ktiles * 512)
        + (b // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (b % 4)
    )


def scale_bytes(rows: int, scales_per_row: int) -> int:
    """Bytes of a swizzled scale tensor for ``rows`` rows of ``scales_per_row`` scales."""
    return sf_rows(rows) * int(scales_per_row)


def unswizzle_scales(
    flat: torch.Tensor, rows: int, scales_per_row: int
) -> torch.Tensor:
    """A swizzled scale tensor -> ``[rows, scales_per_row]`` block-order scales (the consumer's view)."""
    need = scale_bytes(rows, scales_per_row)
    if flat.numel() < need:
        raise ValueError(f"swizzled scale tensor has {flat.numel()} bytes, need {need}")
    offsets = sf_offsets(rows, scales_per_row, flat.device)
    return flat.reshape(-1)[:need][offsets.reshape(-1)].reshape(rows, scales_per_row)


def global_scale_out(y_bf16: torch.Tensor) -> torch.Tensor:
    """FlashInfer's ``input_global_scale`` of ``y``: ``div_rn(2688, amax(y))`` as a float32 ``[1]`` tensor (1 for an all-zero tensor)."""
    amax = y_bf16.to(torch.float32).abs().amax().reshape(1)
    gs = f32_div_rn(torch.full_like(amax, ROW_AMAX_TARGET), amax)
    return torch.where(amax > 0, gs, torch.ones_like(gs)).contiguous()


def nvfp4_products(
    y_bf16: torch.Tensor, gs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer ``fp4_quantize`` (16-value E4M3 block scales) of ``y`` under ``gs`` up to the E2M1 rounding, with IEEE
    reciprocals in place of ``rcp.approx.ftz``: the scaled products ``[rows, 384, 16]`` and the scale codes ``[rows, 384]``.

    Per block: ``sf = e4m3(gs * (vmax * rcp(6)))``; the output scale ``rcp(e4m3(sf)
    * rcp(gs))`` (0 for an all-zero block); the products ``y * scale``.
    """
    rows = y_bf16.shape[0]
    device = y_bf16.device
    y = y_bf16.to(torch.float32).reshape(rows, NVFP4_SCALES_PER_ROW, NVFP4_BLOCK)
    vm = y.abs().amax(dim=-1)
    gs = gs.to(torch.float32).reshape(-1)[:1]
    one = torch.ones((1,), dtype=torch.float32, device=device)
    rcp6 = f32_div_rn(one, torch.full((1,), 6.0, dtype=torch.float32, device=device))
    sfv = f32_mul_rn(gs.expand_as(vm), f32_mul_rn(vm, rcp6.expand_as(vm)))
    sf = e4m3_rn_satfinite(sfv)
    sff = e4m3_to_f32(sf)
    rcp_gs = f32_div_rn(one, gs)
    scale = f32_div_rn(torch.ones_like(sff), f32_mul_rn(sff, rcp_gs.expand_as(sff)))
    scale = torch.where(vm != 0, scale, torch.zeros_like(scale))
    return f32_mul_rn(y, scale.reshape(rows, -1, 1).expand_as(y)), sf


def quant_nvfp4(
    y_bf16: torch.Tensor, gs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer ``fp4_quantize`` (16-value E4M3 block scales, swizzled 128 x 4 layout) of ``y`` under ``gs``:
    ``(values u8 [rows, 3072], scales u8 [sf_rows(rows) * 384])``.  E2M1 ``-0`` is kept, as the quantiser keeps it."""
    rows = y_bf16.shape[0]
    device = y_bf16.device
    g, sf = nvfp4_products(y_bf16, gs)
    values = pack_e2m1(e2m1_rn_satfinite(g).reshape(rows, HIDDEN))
    scales = torch.zeros(
        (scale_bytes(rows, NVFP4_SCALES_PER_ROW),), dtype=torch.uint8, device=device
    )
    scales[sf_offsets(rows, NVFP4_SCALES_PER_ROW, device).reshape(-1)] = sf.reshape(-1)
    return values.contiguous(), scales


def nvfp4_tie_mask(y_bf16: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    """The nibbles of the NVFP4 value rows whose scaled product lies within :data:`TIE_ULPS` float32 ulps of an E2M1
    rounding midpoint: a u8 mask ``[rows, 3072]`` (``0x0F`` the even element, ``0xF0`` the odd one).

    There the one-ulp differences between the kernel's ``rcp.approx.ftz``
    reciprocals (of the global scale, then of the block scale times that) and
    the IEEE reciprocals of this reference decide the code, so the comparison
    leaves those nibbles out (:func:`canonical`).  A tensor whose output scales
    come out round has many such products; elsewhere they are about one in a
    million.
    """
    rows = y_bf16.shape[0]
    g = nvfp4_products(y_bf16, gs)[0].abs().reshape(rows, HIDDEN)
    ulp = torch.ldexp(
        torch.ones_like(g), torch.frexp(g)[1] - 24
    )  # one float32 ulp of g
    tie = torch.zeros_like(g, dtype=torch.bool)
    for value in E2M1_TIES:
        tie |= (g - value).abs() <= TIE_ULPS * ulp
    tie = tie.reshape(rows, HIDDEN // 2, 2).to(torch.uint8)
    return (tie[..., 0] * 0x0F) | (tie[..., 1] * 0xF0)


def canonical(values_u8: torch.Tensor, tie_mask: torch.Tensor) -> torch.Tensor:
    """NVFP4 value rows with the E2M1 rounding-tie nibbles of ``tie_mask`` cleared (apply to both sides of a comparison)."""
    return values_u8 & ~tie_mask


def ue8m0_rp(x: torch.Tensor) -> torch.Tensor:
    """``cvt.rp.satfinite.ue8m0x2.f32``: the exponent code of the power of two at or above ``x`` (0 for 0), clamped to 254; int32."""
    mant, exp = torch.frexp(x)  # x = mant * 2^exp, mant in [0.5, 1)
    ceil_log2 = torch.where(mant == 0.5, exp - 1, exp)
    code = (ceil_log2 + 127).clamp(min=0, max=254)
    return torch.where(x > 0, code, torch.zeros_like(code)).to(torch.int32)


def quant_mxfp8(y_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer ``mxfp8_quantize`` (32-value UE8M0 block scales, swizzled 128 x 4 layout) of ``y`` with the IEEE reciprocal
    of 448 in place of ``rcp.approx.ftz``: ``(E4M3 bytes u8 [rows, 6144], UE8M0 scales u8 [sf_rows(rows) * 192])``.

    Per block: ``e = ue8m0_rp(vmax * rcp(448))``, the scale ``2^(e - 127)`` (code 0
    as 0), ``q = e4m3(y * rcp(scale))`` (a power of two: its reciprocal is exact).
    """
    rows = y_bf16.shape[0]
    device = y_bf16.device
    y = y_bf16.to(torch.float32).reshape(rows, MXFP8_SCALES_PER_ROW, MXFP8_BLOCK)
    vm = y.abs().amax(dim=-1)
    one = torch.ones((1,), dtype=torch.float32, device=device)
    rcp448 = f32_div_rn(
        one, torch.full((1,), E4M3_MAX, dtype=torch.float32, device=device)
    )
    sfv = f32_mul_rn(vm, rcp448.expand_as(vm))
    code = ue8m0_rp(sfv)
    scale = torch.where(
        code > 0, torch.ldexp(torch.ones_like(sfv), code - 127), torch.zeros_like(sfv)
    )
    inv = torch.where(scale != 0, 1.0 / scale, torch.zeros_like(scale))
    q = e4m3_rn_satfinite(f32_mul_rn(y, inv.reshape(rows, -1, 1).expand_as(y)))
    scales = torch.zeros(
        (scale_bytes(rows, MXFP8_SCALES_PER_ROW),), dtype=torch.uint8, device=device
    )
    scales[sf_offsets(rows, MXFP8_SCALES_PER_ROW, device).reshape(-1)] = code.reshape(
        -1
    ).to(torch.uint8)
    return q.reshape(rows, HIDDEN).contiguous(), scales


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def count_mismatches(actual: torch.Tensor, expected: torch.Tensor) -> int:
    """Elements of ``actual`` that differ from ``expected`` (same shape and dtype required; a NaN always differs)."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(
            f"shape / dtype mismatch: {tuple(actual.shape)} {actual.dtype} vs "
            f"{tuple(expected.shape)} {expected.dtype}"
        )
    return int((actual != expected).sum().item())


def mismatch_fraction(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """:func:`count_mismatches` over the element count (0 for empty tensors)."""
    n = int(actual.numel())
    return count_mismatches(actual, expected) / n if n else 0.0


def bf16_ulp_distance(a_bf16: torch.Tensor, b_bf16: torch.Tensor) -> torch.Tensor:
    """Elementwise distance of two bfloat16 tensors in bfloat16 steps (sign-magnitude key; ``+-0`` coincide)."""

    def key(t: torch.Tensor) -> torch.Tensor:
        bits = t.view(torch.int16).to(torch.int32) & 0xFFFF
        mag = bits & 0x7FFF
        return torch.where((bits & 0x8000) != 0, -mag, mag)

    return (key(a_bf16) - key(b_bf16)).abs()


def max_bf16_ulp(actual_bf16: torch.Tensor, expected_bf16: torch.Tensor) -> int:
    """The largest bfloat16 step between two tensors (0 for empty tensors)."""
    if actual_bf16.numel() == 0:
        return 0
    return int(bf16_ulp_distance(actual_bf16, expected_bf16).max().item())


def tolerance_violations(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = Y_ATOL,
    rtol: float = Y_RTOL,
) -> int:
    """Elements with ``|actual - expected| > atol + rtol * |expected|`` (in float32); a non-finite ``actual`` counts."""
    if actual.shape != expected.shape:
        raise ValueError(
            f"shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}"
        )
    a = actual.to(torch.float32)
    e = expected.to(torch.float32)
    bad = ((a - e).abs() > atol + rtol * e.abs()) | ~torch.isfinite(a)
    return int(bad.sum().item())
