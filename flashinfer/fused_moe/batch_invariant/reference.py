# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU references for the MoE spec: ``ExactF32`` (every op through
:mod:`flashinfer.batch_invariant.fp_interface.exact`, Fractions, the ground truth) and ``TorchF32`` (torch float32
CPU tensors: ``+ - * /`` are single IEEE roundings, conversions RNE, no flush-to-zero; vectorised, validated against
``ExactF32`` by the CPU test suite). Both implement the ``fp`` interface the spec is written against, so
``route_cpu`` / ``swiglu_cpu`` / ``relu2_cpu`` / ``combine_cpu`` evaluate the *same* spec functions the kernels
execute. The GEMM accumulators themselves come from the device (the opaque ``mma16`` K-step).
"""

from __future__ import annotations

import functools
from typing import List, Optional, Sequence, Tuple, Union

import torch

from ...batch_invariant.fp_interface import exact as fp32
from . import spec as S
from .mterm import frexpe_f32, frexpm_f32, rowmax_f32


class ExactF32:
    """Scalar exact backend over Python floats restricted to binary32."""

    @staticmethod
    def const(v: float) -> float:
        return fp32.to_f32(v)

    add = staticmethod(fp32.add)
    mul = staticmethod(fp32.mul)
    div = staticmethod(fp32.div)
    fma = staticmethod(fp32.fma)
    sqrt = staticmethod(fp32.sqrt)
    fabs = staticmethod(fp32.fabs)
    fmaxc = staticmethod(fp32.fmaxc)
    frexpm = staticmethod(frexpm_f32)
    frexpe = staticmethod(frexpe_f32)
    neg = staticmethod(fp32.neg)
    fmin = staticmethod(fp32.fmin)
    fmax = staticmethod(fp32.fmax)
    exp2i = staticmethod(fp32.exp2i)
    rowmax = staticmethod(
        rowmax_f32
    )  # the MoE rule: a NaN logit is skipped (the device's max.f32)

    @staticmethod
    def where_nan(x: float, a: float, b: float) -> float:
        return a if x != x else b

    @staticmethod
    def cvt_out(a: float, fmt: str) -> float:
        if fmt in ("e4m3", "e5m2"):
            return fp32.to_fp8(a, fmt)
        return fp32.to_format(
            a, {"bf16": fp32.BF16, "f16": fp32.F16, "f32": fp32.F32}[fmt]
        )

    @staticmethod
    def sym(
        name: str,
    ) -> float:  # pragma: no cover - specs bind scalars before evaluation
        raise KeyError(name)


_TORCH_FMT = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
E5M2_MAX = fp32.E5M2[3]


def e5m2_round(a: torch.Tensor) -> torch.Tensor:
    """Vectorised ``cvt.rn.satfinite.e5m2x2.f32`` as values (``fp32.to_fp8(., "e5m2")``): RNE to e5m2 (2 mantissa bits,
    subnormals in steps of 2^-16), magnitudes beyond the largest finite value saturate to it, sign kept, NaN stays NaN."""
    a = a.to(torch.float32)
    m = a.abs().clamp(
        max=E5M2_MAX
    )  # satfinite (also maps +inf to the largest finite value)
    small = m < 2.0**-14
    sub = torch.round(m * 2.0**16) * 2.0**-16  # torch.round is half-to-even
    _, exp = torch.frexp(
        torch.where(small, torch.ones_like(m), m)
    )  # m = mant * 2^exp, mant in [0.5, 1)
    ulp = torch.ldexp(torch.ones_like(m), exp - 3)  # 2^(e-2) with e = exp - 1
    nrm = torch.round(m / ulp) * ulp
    v = torch.where(small, sub, nrm)
    v = torch.where(torch.signbit(a), -v, v)
    return torch.where(torch.isnan(a), a, v)


class TorchF32:
    """Vectorised backend on float32 CPU tensors (broadcasting). Scalars are 0-d tensors."""

    @staticmethod
    def const(v: float) -> torch.Tensor:
        return torch.tensor(fp32.to_f32(v), dtype=torch.float32)

    @staticmethod
    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.add(a, b)

    @staticmethod
    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mul(a, b)

    @staticmethod
    def div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.div(a, b)

    @staticmethod
    def neg(a: torch.Tensor) -> torch.Tensor:
        return torch.neg(a)

    @staticmethod
    def fmin(a: torch.Tensor, b: Union[torch.Tensor, float]) -> torch.Tensor:
        # fp32.fmin returns ``a`` on ties; torch.fmin returns either (equal values): same bits except for +0 / -0, which
        # the specs never clamp against. torch.fmin (not torch.minimum) returns the other operand against a NaN, as the
        # device's min.f32 and fp32.fmin do: the declared exp of a NaN argument is the clamp's value on every backend.
        return torch.fmin(a, torch.as_tensor(b, dtype=torch.float32))

    @staticmethod
    def fmax(a: torch.Tensor, b: Union[torch.Tensor, float]) -> torch.Tensor:
        return torch.fmax(a, torch.as_tensor(b, dtype=torch.float32))

    @staticmethod
    def exp2i(k: torch.Tensor) -> torch.Tensor:
        n = k.to(torch.int32)
        return ((n + 127) << 23).view(torch.float32)

    @staticmethod
    def fabs(a: torch.Tensor) -> torch.Tensor:
        return torch.abs(a)

    @staticmethod
    def fmaxc(a: torch.Tensor, b: Union[torch.Tensor, float]) -> torch.Tensor:
        """The device's ``max.f32``: a zero tie is ``+0`` when either operand is ``+0`` (``-0`` only for two ``-0``)."""
        b = torch.as_tensor(b, dtype=torch.float32)
        r = torch.maximum(a, b)
        zero_tie = (a == 0) & (b == 0)
        pos = (torch.signbit(a) == False) | (torch.signbit(b) == False)  # noqa: E712
        return torch.where(zero_tie & pos, torch.zeros_like(r), r)

    @staticmethod
    def sqrt(a: torch.Tensor) -> torch.Tensor:
        # torch's vectorised float32 sqrt is a half-ulp-plus-epsilon library routine on some CPUs (x86: one ulp off on a
        # few percent of inputs); the binary64 sqrt is the IEEE instruction and its rounding to binary32 is innocuous for a
        # square root, so the composition is the correctly rounded binary32 sqrt the device computes
        return torch.sqrt(a.to(torch.float64)).to(torch.float32)

    @staticmethod
    def frexpm(a: torch.Tensor) -> torch.Tensor:
        bits = a.contiguous().view(torch.int32) & 0x7FFFFFFF
        return ((bits & 0x007FFFFF) | 0x3F000000).view(torch.float32)

    @staticmethod
    def frexpe(a: torch.Tensor) -> torch.Tensor:
        bits = a.contiguous().view(torch.int32) & 0x7FFFFFFF
        return (((bits >> 23) & 0xFF) - 126).to(torch.float32)

    @staticmethod
    def rowmax(*vals: torch.Tensor) -> torch.Tensor:
        # torch.fmax takes the other operand against a NaN, as the device's max.f32 and rowmax_f32 do
        return functools.reduce(torch.fmax, vals)

    @staticmethod
    def where_nan(x: torch.Tensor, a: float, b: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isnan(x), torch.tensor(a, dtype=torch.float32), b)

    @staticmethod
    def cvt_out(a: torch.Tensor, fmt: str) -> torch.Tensor:
        if fmt == "e4m3":
            # the vectorised cvt.rn.satfinite.e4m3 (values), validated against fp32.to_fp8 by the quantization suite
            from ...batch_invariant.quantization.nvfp4_reference import e4m3_round

            return e4m3_round(a)
        if fmt == "e5m2":
            return e5m2_round(a)
        return a.to(_TORCH_FMT[fmt])


# ----------------------------------------------------------------------------- vectorised references


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", torch.float32)


def exp_cpu(x: torch.Tensor) -> torch.Tensor:
    return S.exp_term(TorchF32, _f32(x))


def swiglu_cpu(g: torch.Tensor, u: torch.Tensor, out_fmt: str = "bf16") -> torch.Tensor:
    """``cvt(silu(g) * u)`` from fp32 accumulators, elementwise."""
    return S.swiglu_elem(TorchF32, _f32(g), _f32(u), out_fmt)


def glu_cpu(
    g: torch.Tensor, u: torch.Tensor, act: str = "silu", out_fmt: str = "bf16"
) -> torch.Tensor:
    """``cvt(act(g) * u)`` from fp32 accumulators (``swiglu_cpu`` for ``act = "silu"``)."""
    return S.glu_elem(TorchF32, _f32(g), _f32(u), act, out_fmt)


def relu2_cpu(acc: torch.Tensor, out_fmt: str = "bf16") -> torch.Tensor:
    """``cvt(relu(acc)^2)`` from one fp32 accumulator, elementwise (the ungated chain)."""
    return S.act_elem(TorchF32, _f32(acc), "relu2", out_fmt)


def glu_cpu_act(
    g: torch.Tensor,
    u: torch.Tensor,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
    out_fmt: str = "bf16",
) -> torch.Tensor:
    """The gated chain ``act`` with ``params`` on fp32 (gate, up) accumulators, vectorised: ``spec.glu_elem`` through
    :class:`TorchF32`."""
    return S.glu_elem(TorchF32, _f32(g), _f32(u), act, out_fmt, params)


def act_cpu(
    acc: torch.Tensor,
    act: str = "relu2",
    params: Sequence[Optional[float]] = (),
    out_fmt: str = "bf16",
) -> torch.Tensor:
    """The ungated chain ``act`` on fp32 accumulators, vectorised: ``spec.act_elem`` through :class:`TorchF32`."""
    return S.act_elem(TorchF32, _f32(acc), act, out_fmt, params)


def route_cpu(
    logits: torch.Tensor,
    k: int,
    method: S.RoutingMethod = S.SOFTMAX_RENORM,
    bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The routing spec for a whole batch: ``experts [T, k]`` int32, ``weights [T, k]`` f32 (rows whose selection
    scores are all non-NaN: the NaN fill rule is the exact model's, :func:`route_exact`)."""
    l = _f32(logits)
    T, E = l.shape
    cols = [l[:, e] for e in range(E)]
    b = [_f32(bias)[e] for e in range(E)] if method.bias else None
    q, w = S.per_expert_scores(TorchF32, cols, method, b)
    q = torch.stack(q, dim=1)
    w = torch.stack(w, dim=1)
    if method.grouped:
        # the group stage: per group the sum of its two largest scores (spec.group_score_term), the topk_group best groups
        # kept (both orders strict, ties to the lower index), the other groups' experts out of the selection
        n = method.n_group
        G = E // n
        top2 = torch.sort(q.view(T, n, G), dim=2, descending=True, stable=True).values[
            :, :, :2
        ]
        gs = S.group_score_term(TorchF32, top2[:, :, 0], top2[:, :, 1])
        kept = torch.sort(gs, dim=1, descending=True, stable=True).indices[
            :, : method.topk_group
        ]
        gmask = torch.zeros(T, n, dtype=torch.bool).scatter_(1, kept, True)
        q = torch.where(
            gmask.repeat_interleave(G, dim=1), q, torch.full_like(q, float("-inf"))
        )
    # (score desc, index asc): a stable descending sort keeps equal scores in index order
    order = torch.sort(q, dim=1, descending=True, stable=True).indices[:, :k]
    wsel = torch.gather(w, 1, order)
    ws = S.slot_weights(
        TorchF32,
        [wsel[:, r] for r in range(k)],
        list(range(k)),
        method,
        TorchF32.const(scale) if method.scaled else None,
    )
    return order.to(torch.int32), torch.stack(ws, dim=1)


def combine_cpu(
    y: torch.Tensor, dest: torch.Tensor, weights: torch.Tensor, out_fmt: str = "bf16"
) -> torch.Tensor:
    """``out[t] = cvt( fold_r add(acc, mul(f32(y[dest[t, r]]), w[t, r])) )``; a dropped slot (``dest = -1``) is the
    term ``+0``, whatever its weight (the kernel's rule)."""
    yc, d, w = y.detach().cpu(), dest.detach().cpu().long(), _f32(weights)
    T, k = d.shape
    ok = d >= 0
    leaves = [
        torch.where(
            ok[:, r : r + 1],
            TorchF32.mul(yc[d[:, r].clamp_min(0)].float(), w[:, r : r + 1]),
            torch.zeros(1, dtype=torch.float32),
        )
        for r in range(k)
    ]
    acc = S.combine_reduction(k, lambda fp, r: leaves[r]).canonical(TorchF32)
    return TorchF32.cvt_out(acc, out_fmt)


# ----------------------------------------------------------------------------- exact (scalar) references


def route_exact(
    logits_row: Sequence[float],
    k: int,
    method: S.RoutingMethod,
    bias: Optional[Sequence[float]] = None,
    scale: Optional[float] = None,
) -> Tuple[List[int], List[float]]:
    l = [fp32.to_f32(v) for v in logits_row]
    b = [fp32.to_f32(v) for v in bias] if bias is not None else None
    return S.routing_weights(
        ExactF32, l, k, method, b, fp32.to_f32(scale) if scale is not None else None
    )


def swiglu_exact(g: float, u: float, out_fmt: str = "bf16") -> float:
    return S.swiglu_elem(ExactF32, fp32.to_f32(g), fp32.to_f32(u), out_fmt)


def combine_exact(
    ys: Sequence[float], ws: Sequence[float], out_fmt: str = "bf16"
) -> float:
    return S.combine_elem(
        ExactF32,
        [fp32.to_f32(v) for v in ys],
        [fp32.to_f32(v) for v in ws],
        out_fmt,
    )


def ulp_error(got: torch.Tensor, ref64: torch.Tensor) -> float:
    """max |got - ref| in units of the binary32 ulp at ref (for accuracy yardsticks); a zero or subnormal reference
    counts in the subnormal spacing (its magnitude clamped to the smallest normal binary32 before the log)."""
    r = ref64.double()
    ulp = torch.ldexp(
        torch.ones_like(r), torch.floor(torch.log2(r.abs().clamp_min(2.0**-126))) - 23
    )
    return ((got.double() - r).abs() / ulp).max().item()
