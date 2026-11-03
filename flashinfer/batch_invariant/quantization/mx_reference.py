"""CPU references for the MX quantization classes (docs/mx_quant_scoping.md).

* the exact backend ``ExactF32`` (every op through :mod:`flashinfer.batch_invariant.fp_interface.exact`, Fractions) and the vectorised ``TorchF32`` backend
  of :mod:`flashinfer.batch_invariant.quantization.reference`, extended here with the vectorised UE8M0 conversion (``ue8m0_round``, validated against
  ``fp32.to_ue8m0`` by ``tests/test_mx_quant_cpu.py``);
* :func:`quantize_mx_reference`: the spec for a whole (M, K) tensor of fp32 values (bf16 / f16 inputs are widened
  exactly first), vectorised, in either scale-factor layout -- the oracle the GPU suites and the norm agent's fused
  classes compare against bit for bit; :func:`exact_rows` evaluates the symbolic spec with the exact backend for a few
  rows; :func:`dequantize_mx` is the fp64 yardstick.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from ..fp_interface import exact as fp32
from . import qterm as term
from . import mx_spec as S
from .layout import layout_name, sf_numel, swizzle_sf, unswizzle_sf
from .nvfp4_reference import (
    TorchF32,
    e2m1_codes,
    e4m3_bytes,
    pack_e2m1,
    unpack_e2m1,
    _E2M1_MAGS,
)

_MIN = fp32.UE8M0_MIN_VALUE
_MAX = fp32.UE8M0_MAX_VALUE


def ue8m0_round(a: torch.Tensor, rnd: str = "rp") -> torch.Tensor:
    """Vectorised :func:`flashinfer.batch_invariant.fp_interface.exact.to_ue8m0` (values, not bytes): the power of two ``rp`` (least ``>= |a|``), ``rz``
    (greatest ``<= |a|``) or ``rn`` (nearest in linear space, ties up) of the MAGNITUDE (the hardware ignores the sign);
    zeros and ``|a| <= 2^-127`` give ``2^-127``, ``|a| >= 2^127`` and infinities give ``2^127``, NaN stays NaN. A CPU model:
    torch's CUDA ``frexp`` / ``ldexp`` path differs on the subnormal UE8M0 values."""
    assert rnd in fp32.UE8M0_ROUNDINGS, rnd
    a = a.to(torch.float32)
    mag = a.abs()
    pos = torch.where(mag > 0, mag, torch.full_like(a, 1.0))
    m, e = torch.frexp(pos)  # pos = m * 2^e, m in [0.5, 1)
    lo = torch.ldexp(torch.ones_like(pos), e - 1)  # 2^floor(log2 pos)
    if rnd == "rz":
        v = lo
    elif rnd == "rp":
        v = torch.where(pos == lo, lo, 2 * lo)
    else:
        v = torch.where(pos < 1.5 * lo, lo, 2 * lo)
    v = v.clamp(min=_MIN, max=_MAX)
    v = torch.where(mag > 0, v, torch.full_like(v, _MIN))
    v = torch.where(torch.isinf(a), torch.full_like(v, _MAX), v)
    return torch.where(torch.isnan(a), a, v)


def ue8m0_bytes(v: torch.Tensor) -> torch.Tensor:
    """Bytes (uint8) of UE8M0 values (powers of two in ``[2^-127, 2^127]``, as :func:`ue8m0_round` returns); NaN -> 0xFF."""
    m, e = torch.frexp(torch.where(torch.isnan(v), torch.ones_like(v), v))
    assert bool(((m == 0.5) | torch.isnan(v)).all()), "not UE8M0 values"
    b = (e - 1 + fp32.UE8M0_BIAS).to(torch.int64)
    assert bool(((b >= 0) & (b <= fp32.UE8M0_MAX_BYTE) | torch.isnan(v)).all()), (
        "UE8M0 byte out of range"
    )
    return torch.where(torch.isnan(v), torch.full_like(b, fp32.UE8M0_NAN_BYTE), b).to(
        torch.uint8
    )


def ue8m0_values(b: torch.Tensor) -> torch.Tensor:
    """fp64 values of UE8M0 bytes (0xFF -> NaN)."""
    bi = b.to(torch.int64)
    v = torch.ldexp(
        torch.ones(bi.shape, dtype=torch.float64, device=b.device), bi - fp32.UE8M0_BIAS
    )
    return torch.where(bi == fp32.UE8M0_NAN_BYTE, torch.full_like(v, float("nan")), v)


def pack_elems(q: torch.Tensor, fmt: S.MXFormat) -> torch.Tensor:
    """Element values ``(M, K)`` -> the byte-packed codes: e4m3 bytes ``(M, K)`` or e2m1 nibbles ``(M, K/2)``."""
    if fmt.elem_fmt == "e4m3":
        return e4m3_bytes(q)
    return pack_e2m1(e2m1_codes(q))


def unpack_elems(packed: torch.Tensor, fmt: S.MXFormat) -> torch.Tensor:
    """Byte-packed codes -> fp64 element values."""
    if fmt.elem_fmt == "e4m3":
        return torch.tensor(fp32.E4M3_VALUES, dtype=torch.float64)[
            packed.cpu().to(torch.int64)
        ]
    codes = unpack_e2m1(packed.cpu()).to(torch.int64)
    return _E2M1_MAGS.double()[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)


def amax_ignoring_nan(v: torch.Tensor) -> torch.Tensor:
    """The canonical ``max.f32`` tree over the last dimension as the device computes it: a NaN operand is ignored (the
    other operand is returned), NaN only when every operand is NaN."""
    nan = torch.isnan(v)
    m = torch.where(nan, torch.full_like(v, float("-inf")), v).amax(dim=-1)
    return torch.where(nan.all(dim=-1), torch.full_like(m, float("nan")), m)


def quantize_mx_reference(
    x: torch.Tensor,
    fmt: S.MXFormat = S.MXFP8,
    rule: S.ScaleRule = S.DEFAULT_RULE,
    swizzled="128x4",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The spec for a whole (M, K) tensor (bf16 / f16 / fp32), vectorised: ``(packed codes, sf flat uint8 in the requested
    layout, sf (M, K/32) logical bytes)``. The amax is the canonical tree's value (the max of non-negative values is
    exact for every association)."""
    x = x.detach().cpu()
    M, K = x.shape
    nb = S.num_blocks(K, S.BLOCK)
    fp = TorchF32
    xf = x.to(torch.float32).view(M, nb, S.BLOCK)  # exact widening
    amax = amax_ignoring_nan(fp.fabs(xf))  # a NaN operand is ignored, as by max.f32
    sf_d, inv = S.scale_chain(fp, amax, fmt, rule)  # (M, nb) values
    q = S.quant_elem(fp, xf, inv.unsqueeze(-1), fmt, rule).view(M, K)
    packed = pack_elems(q, fmt)
    sf_rows = ue8m0_bytes(sf_d)
    sf = swizzle_sf(sf_rows, M, K, S.BLOCK, layout_name(swizzled))
    return packed, sf, sf_rows


def exact_rows(
    x: torch.Tensor, fmt: S.MXFormat, rows, rule: S.ScaleRule = S.DEFAULT_RULE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The symbolic spec evaluated with the exact (Fractions) backend for ``rows`` of ``x``: ``(packed codes (n, .), sf
    (n, K/32) bytes)``. Slow; a few rows."""
    x = x.detach().cpu()
    M, K = x.shape
    nb = S.num_blocks(K, S.BLOCK)
    spec = S.MXQuantSpec(K, fmt, rule)
    xf = x.to(torch.float32).double()
    rows = list(rows)
    q = torch.empty(len(rows), K, dtype=torch.float32)
    sfs = torch.empty(len(rows), nb, dtype=torch.uint8)
    for n, r in enumerate(rows):
        memo: Dict[int, float] = {}
        outs = spec.outputs(r)
        xr = xf[r].tolist()
        ev = lambda t: term.evaluate(t, lambda rr, ii: xr[ii], memo, {})
        for i, t in enumerate(outs["codes"]):
            q[n, i] = ev(t)
        for b, t in enumerate(outs["sf"]):
            sfs[n, b] = fp32.ue8m0_byte(ev(t))
    return pack_elems(q, fmt), sfs


def dequantize_mx(
    packed: torch.Tensor, sf_rows: torch.Tensor, fmt: S.MXFormat, K: int
) -> torch.Tensor:
    """fp64 values ``q * 2^(e8 - 127)``: exact (the scale is a power of two)."""
    M = sf_rows.shape[0]
    vals = unpack_elems(packed, fmt).view(M, K)
    return vals * ue8m0_values(sf_rows.cpu()).repeat_interleave(S.BLOCK, dim=1)


def logical_sf(sf_flat: torch.Tensor, M: int, K: int, swizzled="128x4") -> torch.Tensor:
    """The ``(M, K/32)`` scale bytes of any layout."""
    return unswizzle_sf(sf_flat, M, K, S.BLOCK, layout_name(swizzled))


def sf_bytes_for(M: int, K: int, swizzled="128x4") -> int:
    return sf_numel(M, K, S.BLOCK, layout_name(swizzled))
