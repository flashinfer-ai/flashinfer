# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Exact CPU model of ``support.row.v1`` (``support_spec.py``): per row the weights, ``S`` (the canonical tree), the
thresholds by radix selection over bit patterns (an exact integer procedure, here written as a sort over patterns: the
result -- the k-th largest pattern / the largest pattern whose upward-closed mass reaches the target -- is the same
mathematical object the kernel's histogram passes find), the kept mask, the kept mass (the masked canonical tree), the
renormalised probabilities and, for logits, the renormalised log-probabilities and the masked logits.

Outside every kernel cache closure (no kernel module imports it)."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from ..fp_interface import exact as fp32
from .mufu_device import ex2_device, lg2_device
from .reference import _scales, row_max, row_sum
from .spec import LN2_F32
from .support_spec import Filters, fixed_scale, fixed_target, fixed_weight


def _patterns(w: torch.Tensor) -> torch.Tensor:
    """The 31-bit patterns of non-negative f32 weights (``+0`` and ``-0`` both 0)."""
    return w.view(torch.int32).to(torch.int64) & 0x7FFFFFFF


def _as_f32(pattern: int) -> float:
    return float(torch.tensor([pattern], dtype=torch.int32).view(torch.float32)[0])


def top_k_threshold(w: torch.Tensor, k: int) -> float:
    """The k-th largest weight (ties counted), ``+inf`` for ``k <= 0`` (nothing kept), ``0`` for ``k >= N``."""
    N = w.numel()
    if k <= 0:
        return float("inf")
    if k >= N:
        return 0.0
    srt = torch.sort(_patterns(w), descending=True).values
    return _as_f32(int(srt[k - 1]))


def top_p_threshold(w: torch.Tensor, p: float, S: float) -> float:
    """The largest weight ``t`` such that the fixed-point mass of ``{w_j >= t}`` reaches ``trunc(p * 2^44)``; every
    positive weight when the positive mass does not reach it or ``p >= 1``; ``+inf`` when no weight is positive."""
    scale = fixed_scale(S)
    pats = _patterns(w)
    if not (scale < float("inf")):
        # the (kept) sum is below ~2^-84: the fixed-point scale overflows and every positive weight would saturate to
        # the same mass -- the row keeps nothing (SELECT_RULE: scale-overflow -> +inf) and is invalid
        return float("inf")
    if p >= 1.0:
        # every positive weight, directly (the kernel: the target is unreachable, the fallback the smallest positive)
        pos = pats[pats > 0]
        return float("inf") if pos.numel() == 0 else _as_f32(int(pos.min()))
    q = torch.tensor(
        [fixed_weight(float(x), scale) for x in w.tolist()], dtype=torch.int64
    )
    order = torch.argsort(pats, descending=True, stable=True)
    target = fixed_target(p)
    cum, thr = 0, None
    for idx in order.tolist():
        if pats[idx] == 0:
            break
        cum += int(q[idx])
        thr = int(pats[idx])
        if cum >= target:
            break
    if thr is None:
        return float("inf")
    # every weight equal to the threshold pattern is kept (ties); the threshold is that value
    return _as_f32(thr)


def _combine(thr: float, t: float) -> float:
    """The effective threshold of two filters: the larger of the two in the kernel's order -- the integer maximum of the
    31-bit patterns -- so a NaN threshold (``mul(w_max, min_p)`` with an infinite maximum and ``min_p = 0``) orders above
    every weight and nothing is kept; the row is invalid either way (its mass is not a positive finite number)."""
    if t != t:
        return t
    if thr != thr:
        return thr
    return t if t > thr else thr


def support_row(
    w: torch.Tensor, N: int, f: Filters, w_max: Optional[float] = None
) -> Dict[str, object]:
    """One row: ``{"mask" bool (N,), "threshold" f32 (the effective threshold: the largest of the applied filters', so
    that ``mask == (w >= threshold)``), "count" int, "mass" f32 (the masked canonical tree), "S" f32}`` for f32 weights
    ``w``."""
    S = row_sum(w, N)
    keep = torch.ones(N, dtype=torch.bool)
    thr = 0.0
    if f.order == "top_k_first":
        if f.top_k is not None:
            thr = max(thr, top_k_threshold(w, f.top_k))
            keep &= w >= thr
        if f.top_p is not None:
            wk = torch.where(keep, w, torch.zeros_like(w))
            Sk = row_sum(wk, N)
            thr = max(thr, top_p_threshold(wk, f.top_p, Sk) if Sk > 0 else float("inf"))
            keep &= w >= thr
    else:
        if f.top_k is not None:
            thr = max(thr, top_k_threshold(w, f.top_k))
        if f.top_p is not None:
            thr = max(thr, top_p_threshold(w, f.top_p, S) if S > 0 else float("inf"))
        keep &= w >= thr
    if f.min_p is not None:
        wm = row_max(w) if w_max is None else w_max
        thr = _combine(thr, fp32.mul(wm, f.min_p))
        keep &= w >= thr
    mass = row_sum(torch.where(keep, w, torch.zeros_like(w)), N)
    if bool(torch.isnan(w).any()):
        # a NaN weight: the kernel's bit-pattern mask keeps it (NaN orders above +inf), its mass is NaN, the row invalid
        mass = float("nan")
    return {
        "mask": keep,
        "threshold": thr,
        "count": int(keep.sum()),
        "mass": mass,
        "S": S,
    }


def kept_log_probs_reference(
    x: torch.Tensor,
    token_ids: Sequence[int],
    filters: Sequence[Filters],
    temperature=None,
) -> torch.Tensor:
    """The exact model of ``log_probs(logits, ids, top_k / top_p / min_p)``: per row the kept set and mass of
    ``support_row`` over ``lse.row``'s weights, then the id's log-probability under the kept set
    ``mul(add(z_t, neg(lg2(mass))), LN2)``; ``-inf`` for an id outside the kept set; NaN for an id outside ``[0, N)`` or a
    row whose kept mass is not a positive finite number. fp32 ``(M,)``."""
    x = x.detach().cpu()
    M, N = x.shape
    xf = x.to(torch.float32)
    cs = _scales(temperature, M)
    out = []
    for r in range(M):
        tid = int(token_ids[r])
        m = row_max(xf[r])
        z = torch.mul(
            torch.add(xf[r], torch.tensor(fp32.neg(m), dtype=torch.float32)),
            torch.tensor(cs[r], dtype=torch.float32),
        )
        w = ex2_device(z)
        res = support_row(w, N, filters[r], w_max=1.0)
        mass = float(res["mass"])  # type: ignore[arg-type]
        mask = res["mask"]
        if not (0 <= tid < N) or not (mass > 0 and mass < float("inf")):
            out.append(float("nan"))
        elif not bool(mask[tid]):  # type: ignore[index]
            out.append(float("-inf"))
        else:
            lg2m = float(lg2_device(torch.tensor([mass], dtype=torch.float32))[0])
            out.append(fp32.mul(fp32.add(float(z[tid]), fp32.neg(lg2m)), LN2_F32))
    return torch.tensor(out, dtype=torch.float64).to(torch.float32)


def support_reference(
    x: torch.Tensor,
    filters: Sequence[Filters],
    from_logits: bool = False,
    temperature=None,
    rows: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    """``rows`` of ``x`` (probabilities, or logits with ``from_logits``) under per-row ``filters`` (one per output row):
    the mask ``(n, N)``, ``threshold`` / ``mass`` / ``S`` f32 ``(n,)``, ``count`` int64, the renormalised probabilities
    ``probs`` ``(n, N)`` in ``x``'s dtype (``cvt(mul(w_j, div(1, mass)))``, ``+0`` outside the kept set; NaN where the
    kept mass is zero) and, for logits, ``logprobs`` ``(n, N)`` f32 of the kept tokens (NaN elsewhere):
    ``mul(add(z_j, neg(lg2(mass))), LN2)``, and ``masked_logits`` (the input with ``-inf`` outside the kept set)."""
    x = x.detach().cpu()
    M, N = x.shape
    rows = list(range(M)) if rows is None else list(rows)
    if not rows:  # an empty selection or a (0, N) input: shaped empty outputs instead of a stack of nothing
        empty: Dict[str, torch.Tensor] = {
            "mask": torch.zeros(0, N, dtype=torch.bool),
            "count": torch.zeros(0, dtype=torch.int64),
            "probs": torch.zeros(0, N, dtype=x.dtype),
            "threshold": torch.zeros(0, dtype=torch.float32),
            "mass": torch.zeros(0, dtype=torch.float32),
            "S": torch.zeros(0, dtype=torch.float32),
        }
        if from_logits:
            empty["logprobs"] = torch.zeros(0, N, dtype=torch.float32)
            empty["masked_logits"] = torch.zeros(0, N, dtype=x.dtype)
        return empty
    xf = x.to(torch.float32)
    cs = _scales(temperature, len(rows)) if from_logits else None
    out: Dict[str, list] = {
        k: []
        for k in (
            "mask",
            "threshold",
            "count",
            "mass",
            "S",
            "logprobs",
            "probs",
            "masked_logits",
        )
    }
    for i, r in enumerate(rows):
        if from_logits:
            assert cs is not None
            m = row_max(xf[r])
            z = torch.mul(
                torch.add(xf[r], torch.tensor(fp32.neg(m), dtype=torch.float32)),
                torch.tensor(cs[i], dtype=torch.float32),
            )
            w = ex2_device(z)
            res = support_row(
                w, N, filters[i], w_max=1.0
            )  # e_max = EX2(+0) = 1 by the shift
            mass = float(res["mass"])  # type: ignore[arg-type]
            lg2m = (
                float(lg2_device(torch.tensor([mass], dtype=torch.float32))[0])
                if mass > 0
                else float("nan")
            )
            lp = torch.mul(
                torch.add(z, torch.tensor(fp32.neg(lg2m), dtype=torch.float32)),
                torch.tensor(LN2_F32, dtype=torch.float32),
            )
            keep = res["mask"]
            out["logprobs"].append(
                torch.where(keep, lp, torch.full_like(lp, float("nan")))
            )
            out["masked_logits"].append(
                torch.where(keep, x[r], torch.full_like(x[r], float("-inf")))
            )
        else:
            w = xf[r]
            res = support_row(w, N, filters[i])
        # the renormalised probabilities: lse.row's prob chain, cvt(mul(w, div(1, mass))), +0 outside the kept set
        mass = float(res["mass"])  # type: ignore[arg-type]
        inv = torch.tensor(
            fp32.div(1.0, mass) if mass > 0 else float("inf"), dtype=torch.float32
        )
        pr = torch.where(res["mask"], torch.mul(w, inv), torch.zeros_like(w)).to(
            x.dtype
        )  # type: ignore[arg-type]
        out["probs"].append(pr)
        for k in ("mask", "threshold", "count", "mass", "S"):
            out[k].append(res[k])
    ret: Dict[str, torch.Tensor] = {
        "mask": torch.stack(out["mask"]),
        "count": torch.tensor(out["count"], dtype=torch.int64),
        "probs": torch.stack(out["probs"]),
    }
    for k in ("threshold", "mass", "S"):
        ret[k] = torch.tensor(out[k], dtype=torch.float64).to(torch.float32)
    if from_logits:
        ret["logprobs"] = torch.stack(out["logprobs"])
        ret["masked_logits"] = torch.stack(out["masked_logits"])
    return ret
