# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The OUTPUT tolerance convention of the Qwen 3.8 Flash-Next gated attention block tests -- ONE definition, mirrored from the cuDNN
Frontend block suites' convention (the tree the kernels are vendored from; cuDNN Frontend #1995), shared by the decode and the extend
tests.

A bf16 output of the block is held to its fp32 oracle PER ELEMENT at

    |got - ref| <= atol_frac * max|ref| + rtol * |ref|        and        cos(got, ref) >= 0.999

with ``rtol = 2^-6`` and ``atol_frac = 2^-7`` for bf16 (``2^-9`` / ``2^-10`` for f16: three more mantissa bits at every rounding point).
Calibrated once on the cuDNN Frontend side (the block backward's first Rubin runs: accept cells at 0.40-0.66 of the bound) and never
widened; every assertion prints the worst element as a FRACTION of the bound (``bound_ratio``) beside ``max|d|``, so a calibration or a
regression is a number, never a silent widening.

Why a tensor-scaled per-element bound and not one absolute ``atol`` on the gated O: an absolute ``2e-2`` transplanted onto bf16 O values
of magnitude up to ~3 is 78 % of one bf16 ulp there (0.015625), and any single rounding flip at ``|O| >= 4`` (ulp 0.03125) would exceed
it for every implementation alike.  The block's documented bf16 ``(1 + w)`` QK-norm weight rounding moves a SHORT row's gated O (six
visible keys, two dominant keys with large opposite-sign V at one dim) by 0.0215 against the fp32 oracle -- 0.877 of this bound, cosine
0.999989 -- while every row with >= 1024 keys stays within ~2e-3; the oracle is unchanged by this convention.  Before this helper the
FlashInfer tests gated ``out`` by the cosine alone (``OUT_COS = 0.999``): the per-element bound is a TIGHTENING, never a widening."""

from __future__ import annotations

import math
from typing import Optional

import torch

OUTPUT_RTOL = {torch.bfloat16: 2.0**-6, torch.float16: 2.0**-9}
OUTPUT_ATOL_FRAC = {torch.bfloat16: 2.0**-7, torch.float16: 2.0**-10}
OUTPUT_COS_MIN = 0.999
_CHUNK = (
    1 << 26
)  # elements per fp64 reduction chunk of the cosine (an fp32 dot over 10^7+ elements drifts past 1.0)


def cos64(a: torch.Tensor, b: torch.Tensor) -> float:
    """The cosine in fp64 accumulation, chunked."""
    a, b = a.detach().reshape(-1), b.detach().reshape(-1)
    if a.numel() == 0:
        return 1.0
    dot = na = nb = 0.0
    for lo in range(0, a.numel(), _CHUNK):
        x, y = a[lo : lo + _CHUNK].double(), b[lo : lo + _CHUNK].double()
        dot += float(x @ y)
        na += float(x @ x)
        nb += float(y @ y)
    if na == 0.0 and nb == 0.0:
        return 1.0  # two all-zero tensors (every row dead) agree exactly
    denom = math.sqrt(na) * math.sqrt(nb)
    return dot / denom if denom > 0 else float("nan")


def _bound_terms(
    got: torch.Tensor, rtol: Optional[float], atol_frac: Optional[float]
) -> tuple:
    if rtol is None:
        if got.dtype not in OUTPUT_RTOL:
            raise ValueError(
                f"no output bound is defined for {got.dtype}; pass rtol / atol_frac explicitly"
            )
        rtol = OUTPUT_RTOL[got.dtype]
    if atol_frac is None:
        if got.dtype not in OUTPUT_ATOL_FRAC:
            raise ValueError(
                f"no output bound is defined for {got.dtype}; pass rtol / atol_frac explicitly"
            )
        atol_frac = OUTPUT_ATOL_FRAC[got.dtype]
    return float(rtol), float(atol_frac)


def output_bound_stats(
    got: torch.Tensor,
    ref: torch.Tensor,
    *,
    rtol: Optional[float] = None,
    atol_frac: Optional[float] = None,
) -> dict:
    """The statistics of ``got`` against ``ref`` under the output bound (the dtype of ``got`` picks ``rtol`` / ``atol_frac`` unless given):
    ``max_abs`` (max |d|, the recorded statistic), ``max_ref`` (max |ref|), ``bound_ratio`` (the worst element's |d| as a fraction of its
    bound; <= 1 means within), ``n_above_bound``, ``cos`` (fp64), ``finite``, the ``worst`` element's index / values, and the bound's terms.
    Both tensors are compared where they live (fp32 arithmetic on their device; the cosine in fp64)."""
    rtol, atol_frac = _bound_terms(got, rtol, atol_frac)
    got32, ref32 = got.detach().float(), ref.detach().float()
    if got32.shape != ref32.shape:
        raise ValueError(
            f"shape mismatch: got {tuple(got32.shape)} vs ref {tuple(ref32.shape)}"
        )
    finite = bool(torch.isfinite(got32).all())
    if got32.numel() == 0:
        return dict(
            max_abs=0.0,
            max_ref=0.0,
            bound_ratio=0.0,
            n_above_bound=0,
            cos=1.0,
            finite=True,
            rtol=rtol,
            atol_frac=atol_frac,
            worst=None,
            n_elements=0,
        )
    ref_max = float(ref32.abs().max())
    diff = (got32 - ref32).abs()
    bound = atol_frac * ref_max + rtol * ref32.abs()
    if ref_max > 0:
        ratio = diff / bound
        i = int(torch.argmax(ratio.reshape(-1)))
        bound_ratio = float(ratio.reshape(-1)[i])
    else:  # an all-zero reference: the bound is 0 everywhere, so any difference is outside it
        i = int(torch.argmax(diff.reshape(-1)))
        bound_ratio = 0.0 if float(diff.max()) == 0.0 else float("inf")
    n_above = int((diff > bound).sum())
    idx = [int(x) for x in torch.unravel_index(torch.tensor(i), diff.shape)]
    return dict(
        max_abs=float(diff.max()),
        max_ref=ref_max,
        bound_ratio=bound_ratio,
        n_above_bound=n_above,
        cos=cos64(got32, ref32),
        finite=finite,
        rtol=rtol,
        atol_frac=atol_frac,
        n_elements=int(diff.numel()),
        worst=dict(
            index=idx,
            got=float(got32.reshape(-1)[i]),
            ref=float(ref32.reshape(-1)[i]),
            abs_diff=float(diff.reshape(-1)[i]),
            bound=float(bound.reshape(-1)[i]),
        ),
    )


def format_output_stats(what: str, s: dict, *, cos_min: float = OUTPUT_COS_MIN) -> str:
    return (
        f"{what}: max|d|={s['max_abs']:.4g} max|ref|={s['max_ref']:.4g} worst cell {s['bound_ratio']:.3f} of the bound "
        f"(rtol={s['rtol']}, atol={s['atol_frac']}*max|ref|; {s['n_above_bound']} of {s['n_elements']} elements above it) cos={s['cos']:.6f} (floor {cos_min})"
        + (
            f" at {s['worst']['index']}: got {s['worst']['got']:.6g} ref {s['worst']['ref']:.6g}"
            if s.get("worst")
            else ""
        )
    )


def assert_output_close(
    got: torch.Tensor,
    ref: torch.Tensor,
    what: str,
    *,
    rtol: Optional[float] = None,
    atol_frac: Optional[float] = None,
    cos_min: float = OUTPUT_COS_MIN,
) -> dict:
    """The output bound as an assertion: finite, ``cos >= cos_min``, no element above ``atol_frac * max|ref| + rtol * |ref|``.  Prints the
    magnitudes (``max|d|`` kept as the recorded statistic beside the fraction of the bound) and returns ``output_bound_stats``."""
    s = output_bound_stats(got, ref, rtol=rtol, atol_frac=atol_frac)
    msg = format_output_stats(what, s, cos_min=cos_min)
    print(msg)
    assert s["finite"], f"{what}: non-finite cells"
    assert s["cos"] >= cos_min, msg
    assert s["n_above_bound"] == 0 and s["bound_ratio"] <= 1.0, msg
    return s
