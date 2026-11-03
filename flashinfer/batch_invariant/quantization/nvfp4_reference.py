"""CPU references for the NVFP4 quantization spec.

* ``ExactF32``: every op through :mod:`flashinfer.batch_invariant.fp_interface.exact` (Fractions): the ground truth, scalar, slow.
* ``TorchF32``: torch float32 CPU tensors (``+ - * /`` and ``min`` / ``max`` / ``abs`` are single IEEE
  roundings or exact; no flush-to-zero), vectorised; the e4m3 and e2m1 conversions are implemented here
  (``e4m3_round``, ``e2m1_round``) and validated against ``fp32.to_fp8`` / ``fp32.to_e2m1`` by
  ``tests/test_nvfp4_quant_cpu.py``.

Both implement the ``fp`` interface the spec is written against, so :func:`quantize_reference` evaluates
the *same* spec functions the kernel executes (rule I8). :func:`exact_rows` evaluates the symbolic spec
with the exact backend for a few rows and is compared bit for bit with the vectorised reference.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

from ..fp_interface import exact as fp32
from . import qterm as term
from . import nvfp4_spec as S
from .nvfp4_schedule import (
    BLOCK,
    num_blocks,
    padded_sf_cols,
    padded_sf_rows,
    sf_offset_128x4,
)


class ExactF32:
    """Scalar exact backend over Python floats restricted to binary32."""

    @staticmethod
    def const(v):
        return fp32.to_f32(v)

    add = staticmethod(fp32.add)
    mul = staticmethod(fp32.mul)
    div = staticmethod(fp32.div)
    fma = staticmethod(fp32.fma)
    fmin = staticmethod(fp32.fmin)
    fmax = staticmethod(fp32.fmax)
    fmaxc = staticmethod(
        fp32.fmaxc
    )  # the probed sm_107 max.f32 rule (shared with the attention class)
    fabs = staticmethod(fp32.fabs)
    nzor = staticmethod(fp32.nzor)

    @staticmethod
    def cvt_out(a, fmt):
        if fmt == "e4m3":
            return fp32.to_fp8(a, "e4m3")
        if fmt == "e2m1":
            return fp32.to_e2m1(a)
        if fmt.startswith(
            "ue8m0_"
        ):  # the MX classes' scale conversion (rlk/quant/mx_spec.py)
            return fp32.to_ue8m0(a, fmt[6:])
        return fp32.to_format(
            a, {"bf16": fp32.BF16, "f16": fp32.F16, "f32": fp32.F32}[fmt]
        )


# ----------------------------------------------------------------------------- vectorised conversions
_E2M1_MAGS = torch.tensor(fp32.E2M1_MAGNITUDES, dtype=torch.float32)
_E2M1_MIDS = torch.tensor(fp32._E2M1_MIDPOINTS, dtype=torch.float32)
_E4M3_POS = torch.tensor(
    fp32.E4M3_VALUES[:127], dtype=torch.float32
)  # bytes 0..126, ascending


def e2m1_round(a: torch.Tensor) -> torch.Tensor:
    """Vectorised :func:`flashinfer.batch_invariant.fp_interface.exact.to_e2m1` (values, not codes): RNE to the e2m1 magnitudes with ties to
    the even code, saturation beyond 6, sign kept; NaN gives +6.0 as the device does."""
    a = a.to(torch.float32)
    m = a.abs()
    mids = _E2M1_MIDS.to(a.device)
    gt = (m.unsqueeze(-1) > mids).sum(-1)  # midpoints strictly below |a|
    eq = m.unsqueeze(-1) == mids  # at most one tie
    even_up = torch.tensor([(k + 1) % 2 == 0 for k in range(7)], device=a.device)
    code = gt + (eq & even_up).sum(-1)
    code = torch.where(m >= fp32.E2M1_MAX, torch.full_like(code, 7), code)
    v = _E2M1_MAGS.to(a.device)[code.clamp(0, 7)]
    v = torch.where(torch.signbit(a), -v, v)
    return torch.where(torch.isnan(a), torch.full_like(v, fp32.E2M1_MAX), v)


def e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """4-bit codes (uint8) of e2m1 values (sign in bit 3)."""
    mags = _E2M1_MAGS.to(v.device)
    k = torch.searchsorted(mags, v.abs().contiguous()).clamp(max=7)
    assert bool((mags[k] == v.abs()).all()), "not e2m1 values"
    return k.to(torch.uint8) | torch.where(torch.signbit(v), 8, 0).to(torch.uint8)


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """(…, K) codes -> (…, K/2) bytes; element 2j in the low nibble."""
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    lo, hi = packed & 0xF, packed >> 4
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def e4m3_round(a: torch.Tensor) -> torch.Tensor:
    """Vectorised ``cvt.rn.satfinite.e4m3x2.f32`` as values: RNE to e4m3fn (3 mantissa bits, subnormals in
    steps of 2^-9), magnitudes beyond 448 saturate to 448, sign kept, NaN stays NaN."""
    a = a.to(torch.float32)
    m = a.abs().clamp(max=448.0)  # satfinite (also maps +inf to 448)
    small = m < 2.0**-6
    sub = torch.round(m * 2.0**9) * 2.0**-9  # torch.round is half-to-even
    mant, exp = torch.frexp(
        torch.where(small, torch.ones_like(m), m)
    )  # m = mant * 2^exp, mant in [0.5, 1)
    ulp = torch.ldexp(torch.ones_like(m), exp - 4)  # 2^(e-3) with e = exp - 1
    nrm = torch.round(m / ulp) * ulp
    v = torch.where(small, sub, nrm)
    v = torch.where(torch.signbit(a), -v, v)
    return torch.where(torch.isnan(a), a, v)


def e4m3_bytes(v: torch.Tensor) -> torch.Tensor:
    """Bytes (uint8) of e4m3fn values (exactly representable, as :func:`e4m3_round` returns)."""
    tab = _E4M3_POS.to(v.device)
    k = torch.searchsorted(tab, v.abs().contiguous()).clamp(max=126)
    ok = (tab[k] == v.abs()) | torch.isnan(v)
    assert bool(ok.all()), "not e4m3 values"
    b = k.to(torch.uint8) | torch.where(torch.signbit(v), 0x80, 0).to(torch.uint8)
    return torch.where(
        torch.isnan(v), torch.tensor(0x7F, dtype=torch.uint8, device=v.device), b
    )


class TorchF32:
    """Vectorised backend on float32 CPU (or CUDA) tensors with broadcasting. Scalars are 0-d tensors."""

    @staticmethod
    def const(v):
        return torch.tensor(fp32.to_f32(v), dtype=torch.float32)

    @staticmethod
    def add(a, b):
        return torch.add(a, b)

    @staticmethod
    def mul(a, b):
        return torch.mul(a, b)

    @staticmethod
    def div(a, b):
        return torch.div(a, b)

    @staticmethod
    def fma(a, b, c):
        """``fma.rn.f32`` for the MX classes' ``zero = "positive"`` rule, ``fma(x, inv, +0)``: the product of two binary32
        values is exact in binary64 (48 significant bits) and adding ``+0`` is exact (``-0 + +0 = +0``), so one rounding to
        binary32 remains -- the fused semantics. (A general ``c`` could double-round; the specs only use ``c = +0``.)"""
        c = torch.as_tensor(c, dtype=torch.float32, device=a.device)
        assert bool((c == 0).all()), "TorchF32.fma is exact only for c = +-0"
        return (
            a.double()
            * torch.as_tensor(b, dtype=torch.float32, device=a.device).double()
            + c.double()
        ).float()

    @staticmethod
    def fmin(a, b):
        """``fmin.f32`` as the exact model orders it: ``a`` when ``a <= b``, else ``b`` (equal values return ``a``)."""
        bt = torch.as_tensor(b, dtype=torch.float32, device=a.device)
        return torch.where(a <= bt, a, bt)

    @staticmethod
    def fmax(a, b):
        """``fmax.f32`` as the exact model orders it: ``a`` when ``a >= b``, else ``b`` (equal values return ``a``)."""
        bt = torch.as_tensor(b, dtype=torch.float32, device=a.device)
        return torch.where(a >= bt, a, bt)

    fmaxc = fmax

    @staticmethod
    def fabs(a):
        return torch.abs(a)

    @staticmethod
    def nzor(a, b):
        return torch.where(
            a != 0, a, torch.as_tensor(b, dtype=torch.float32, device=a.device)
        )

    @staticmethod
    def cvt_out(a, fmt):
        if fmt == "e4m3":
            return e4m3_round(a)
        if fmt == "e2m1":
            return e2m1_round(a)
        if fmt.startswith("ue8m0_"):
            from .mx_reference import ue8m0_round

            return ue8m0_round(a, fmt[6:])
        return a.to(
            {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[fmt]
        )


# ----------------------------------------------------------------------------- whole-tensor reference
def swizzle_sf_128x4(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """(M, K/16) scale bytes -> the (padded_M, padded_cols) 128x4 swizzled tensor, padding zero."""
    nb = num_blocks(K)
    pr, pc = padded_sf_rows(M), padded_sf_cols(K)
    out = torch.zeros(pr * pc, dtype=torch.uint8, device=sf.device)
    rows = torch.arange(M, device=sf.device).view(M, 1)
    cols = torch.arange(nb, device=sf.device).view(1, nb)
    idx = (
        (cols % 4)
        + (cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + (rows // 128) * (128 * pc)
    )
    if M > 0:  # an empty input has no cell to check
        assert int(idx[0, 0]) == sf_offset_128x4(0, 0, pc) and int(
            idx[M - 1, nb - 1]
        ) == sf_offset_128x4(M - 1, nb - 1, pc)
    out[idx.flatten()] = sf.reshape(-1)
    return out.view(pr, pc)


def unswizzle_sf_128x4(sf_sw: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Inverse of :func:`swizzle_sf_128x4` (the M x K/16 logical scales)."""
    nb = num_blocks(K)
    pc = padded_sf_cols(K)
    rows = torch.arange(M, device=sf_sw.device).view(M, 1)
    cols = torch.arange(nb, device=sf_sw.device).view(1, nb)
    idx = (
        (cols % 4)
        + (cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + (rows // 128) * (128 * pc)
    )
    return sf_sw.reshape(-1)[idx]


def _scale_value(gs) -> torch.Tensor:
    if isinstance(gs, torch.Tensor):
        return gs.detach().to("cpu", torch.float32).reshape(-1)[0]
    return TorchF32.const(float(gs))


def amax_ignoring_nan(v: torch.Tensor) -> torch.Tensor:
    """The canonical ``max.f32`` tree over the last dimension as the device computes it: a NaN operand is ignored (the
    other operand is returned), NaN only when every operand is NaN."""
    nan = torch.isnan(v)
    m = torch.where(nan, torch.full_like(v, float("-inf")), v).amax(dim=-1)
    return torch.where(nan.all(dim=-1), torch.full_like(m, float("nan")), m)


def quantize_reference(
    x: torch.Tensor, gs: Union[float, torch.Tensor], per_token: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """The spec for a whole (M, K) tensor, vectorised: ``(packed (M, K/2) uint8, sf 128x4 swizzled
    (padded_M, padded_cols) uint8, pts (M,) f32 | None)``. ``gs`` is the global scale (per-tensor) or
    the inverse base scale ``gs_inv`` (per-token)."""
    x = x.detach().cpu()
    M, K = x.shape
    nb = num_blocks(K)
    fp = TorchF32
    xf = x.to(torch.float32).view(M, nb, BLOCK)  # exact widening
    amax = amax_ignoring_nan(
        fp.fabs(xf)
    )  # the canonical tree's value (max of non-negatives; a NaN operand is ignored)
    g = _scale_value(gs)
    pts = None
    if per_token:
        ramax = amax_ignoring_nan(amax)
        ges, pts = S.row_scale_chain(fp, ramax, g)
        gsb = ges.view(M, 1)
    else:
        gsb = g
    sf_d, inv = S.block_chain(fp, amax, gsb)  # (M, nb)
    q = S.quant_elem(fp, xf, inv.unsqueeze(-1))  # (M, nb, 16) e2m1 values
    packed = pack_e2m1(e2m1_codes(q).view(M, K))
    sf = swizzle_sf_128x4(e4m3_bytes(sf_d), M, K)
    return packed, sf, pts


def exact_rows(
    x: torch.Tensor, gs: Union[float, torch.Tensor], rows, per_token: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """The symbolic spec evaluated with the exact backend for ``rows`` of ``x``: ``(packed (n, K/2),
    sf (n, K/16) logical bytes, pts (n,) | None)``. Slow (Fractions); a few rows."""
    x = x.detach().cpu()
    M, K = x.shape
    nb = num_blocks(K)
    spec = S.QuantSpec(K, S.Variant(per_token))
    xf = x.to(torch.float32).double()
    g = float(_scale_value(gs))
    syms = {"gs_inv" if per_token else "gs": fp32.to_f32(g)}
    rows = list(rows)
    codes = torch.empty(len(rows), K, dtype=torch.uint8)
    sfs = torch.empty(len(rows), nb, dtype=torch.uint8)
    pts = torch.empty(len(rows), dtype=torch.float32) if per_token else None
    for n, r in enumerate(rows):
        memo: Dict[int, float] = {}
        outs = spec.outputs(r)
        xr = xf[r].tolist()
        ev = lambda t: term.evaluate(t, lambda rr, ii: xr[ii], memo, syms)
        for i, t in enumerate(outs["codes"]):
            codes[n, i] = fp32.e2m1_code(ev(t))
        for b, t in enumerate(outs["sf"]):
            sfs[n, b] = fp32.e4m3_bits(ev(t))
        if pts is not None:
            pts[n] = ev(outs["pts"])
    return pack_e2m1(codes), sfs, pts


def dequantize(
    packed: torch.Tensor,
    sf_sw: torch.Tensor,
    gs: Union[float, torch.Tensor],
    M: int,
    K: int,
    pts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantised f64 values: ``q * sf_d / gs`` (per-tensor) or ``q * sf_d * pts`` (per-token). Used to
    compare implementations that differ in rounding details (FlashInfer's approximate path)."""
    codes = unpack_e2m1(packed.cpu()).to(torch.int64)
    qv = _E2M1_MAGS.double()[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    sf = unswizzle_sf_128x4(sf_sw.cpu(), M, K).to(torch.int64)
    sfv = torch.tensor(fp32.E4M3_VALUES, dtype=torch.float64)[sf].repeat_interleave(
        BLOCK, dim=1
    )
    if pts is not None:
        return qv * sfv * pts.cpu().double().view(M, 1)
    return qv * sfv / float(_scale_value(gs))
