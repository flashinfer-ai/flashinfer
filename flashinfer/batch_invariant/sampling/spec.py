# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""``lse.row``: the canonical log-sum-exp dataflow over a logits row.

Everything numeric lives here once, parameterised by an ``fp`` backend (the symbolic :class:`sterm.TermAlgebra`, the
exact :mod:`flashinfer.batch_invariant.fp_interface.exact`, the CuTe :class:`cute_ext.CuteF32`): the two row reductions
and the chains around them.

* ``m = max_j x_j``: the ``fmaxc`` tree over the row (elem fold from ``-inf``, pairwise blocks, pairwise lanes).
* ``c = LOG2E`` (no temperature) or ``div(LOG2E, T)`` (temperature ``T``, scalar or per row): the one scale.
* ``z_j = mul(add(x_j, neg(m)), c)``, ``e_j = EX2(z_j)`` with ``EX2 = ex2.approx.ftz.f32`` (opaque).
* ``s = sum_j e_j``: the ``add`` tree with the same digits (elem fold from ``+0``).
* outputs: ``p_j = cvt(mul(e_j, div(1, s)))`` in the input format (softmax); ``lse = mul(add(mul(m, c), LG2(s)), LN2)``
  with ``LG2 = lg2.approx.ftz.f32`` (opaque), natural base; ``logp_j = mul(add(z_j, neg(LG2(s))), LN2)`` (the natural
  log-probability of element ``j``: the log-softmax row, and the log-probability of a given token id).

NaN propagates through the sum (``fmaxc`` drops it from the max). A ``-inf`` logit contributes ``EX2(-inf) = +0``.
Elements beyond the row are absent from the trees; a kernel fills them with ``-inf`` (max identity; ``EX2(-inf) = +0``
is the sum identity on the non-negative ``e_j``), which the digits' ``identity_note`` justify bit for bit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..canon import Digit, Reduction
from ..fp_interface import exact as fp32
from .schedule import LANES, V, padded_blocks
from .sterm import Term, TermAlgebra
from .sterm import fingerprint_dag as fingerprint

CLASS = "lse.row.v1"
LOG2E = 1.4426950408889634  # rounded to binary32 (0x3FB8AA3B) wherever it enters the dataflow
LN2 = 0.6931471805599453  # 0x3F317218
NEG_INF = float("-inf")


def flat_index(idx) -> int:
    return (idx["block"] * LANES + idx["lane"]) * V + idx["elem"]


def max_reduction(N: int, leaf_x: Callable) -> Reduction:
    """The row maximum as a canonical tree. ``leaf_x(fp, i)`` is the fp32 term / value of element ``i``."""
    note = "fmaxc(v, -inf) == v for every v (NaN included: fmaxc drops NaN); the kernel fills absent elements with -inf"
    elem = Digit(
        "elem",
        V,
        "fold",
        init=lambda fp: fp.const(NEG_INF),
        step=lambda fp, acc, x: fp.fmaxc(acc, x),
        identity_note=note,
    )
    block = Digit(
        "block",
        padded_blocks(N),
        "pairwise",
        combine=lambda fp, a, b: fp.fmaxc(a, b),
        identity_note=note,
    )
    lane = Digit(
        "lane",
        LANES,
        "pairwise",
        combine=lambda fp, a, b: fp.fmaxc(a, b),
        identity_note=note,
    )

    def leaf(fp, idx):
        i = flat_index(idx)
        return leaf_x(fp, i) if i < N else None

    return Reduction(f"{CLASS}.max[N={N}]", (elem, block, lane), leaf)


def sum_reduction(N: int, leaf_e: Callable) -> Reduction:
    """The row sum of the exponentials as a canonical tree. ``leaf_e(fp, i)`` is ``e_i`` (``>= +0`` or NaN)."""
    note = "add(v, +0) == v for v >= +0 or NaN (the e_j are ex2 values); an absent element's -inf leaf gives ex2(-inf) = +0"
    elem = Digit(
        "elem",
        V,
        "fold",
        init=lambda fp: fp.const(0.0),
        step=lambda fp, acc, e: fp.add(acc, e),
        identity_note=note,
    )
    block = Digit(
        "block",
        padded_blocks(N),
        "pairwise",
        combine=lambda fp, a, b: fp.add(a, b),
        identity_note=note,
    )
    lane = Digit(
        "lane",
        LANES,
        "pairwise",
        combine=lambda fp, a, b: fp.add(a, b),
        identity_note=note,
    )

    def leaf(fp, idx):
        i = flat_index(idx)
        return leaf_e(fp, i) if i < N else None

    return Reduction(f"{CLASS}.sum[N={N}]", (elem, block, lane), leaf)


# ---------------------------------------------------------------------------------------------------------- the chains
def scale(fp, T=None):
    """``c``: ``LOG2E`` without a temperature, ``div(LOG2E, T)`` with one (``T`` an fp32 value or term)."""
    return fp.const(LOG2E) if T is None else fp.div(fp.const(LOG2E), T)


def shifted(fp, x, nm, c):
    """``z = mul(add(x, nm), c)`` with ``nm = neg(m)``: the scaled, max-shifted logit (base 2)."""
    return fp.mul(fp.add(x, nm), c)


def weight(fp, z):
    """``e = EX2(z)``."""
    return fp.ex2(z)


def inv_sum(fp, s):
    return fp.div(fp.const(1.0), s)


def prob(fp, e, inv_s):
    """``p = mul(e, 1/s)`` before the output conversion."""
    return fp.mul(e, inv_s)


def lg2(fp, x):
    """``LG2(x) = lg2.approx.ftz.f32``: ``fp.lg2`` on the backends that name it, the opaque ``op`` on the term algebra."""
    if hasattr(fp, "lg2"):
        return fp.lg2(x)
    return fp.op("lg2", x)


def lse_nat(fp, m, c, lg2s):
    """``lse = mul(add(mul(m, c), LG2(s)), LN2)``: the natural log-sum-exp of ``x / T``."""
    return fp.mul(fp.add(fp.mul(m, c), lg2s), fp.const(LN2))


def logp(fp, z, nlg2s):
    """``logp = mul(add(z, neg(LG2(s))), LN2)``: the natural log-probability of the element with shifted logit ``z``."""
    return fp.mul(fp.add(z, nlg2s), fp.const(LN2))


# ---------------------------------------------------------------------------------------------------- the symbolic spec
@dataclass(frozen=True)
class Mode:
    """Which outputs a kernel writes: the probabilities (softmax, in the input dtype, or in fp32 with ``probs_f32``: the
    chain before its output conversion, which for a 16-bit input is the fp32 input's class since the leaves are the exact
    widenings), the full log-softmax row (f32), the per-row ``lse`` (f32), the log-probabilities of the given token ids per
    row (f32; one id or ``K`` ids per row)."""

    probs: bool = False
    logprobs: bool = False
    lse: bool = False
    logp: bool = False
    probs_f32: bool = False

    def __post_init__(self):
        if not (self.probs or self.logprobs or self.lse or self.logp):
            raise ValueError("a mode writes at least one output")
        if self.probs and self.logprobs:
            raise ValueError("one row output per launch (probs or logprobs)")
        if self.probs_f32 and not self.probs:
            raise ValueError("probs_f32 qualifies the probabilities output")

    @property
    def name(self) -> str:
        return "".join(
            t
            for t, on in (
                ("p", self.probs),
                ("f", self.probs_f32),
                ("l", self.logprobs),
                ("s", self.lse),
                ("t", self.logp),
            )
            if on
        )

    @property
    def row_output(self) -> bool:
        return self.probs or self.logprobs


SOFTMAX = Mode(probs=True)
SOFTMAX_F32 = Mode(probs=True, probs_f32=True)
LOG_SOFTMAX = Mode(logprobs=True)
LSE = Mode(lse=True)
LOGP = Mode(logp=True)
LOGP_LSE = Mode(logp=True, lse=True)


class LSESpec:
    """Symbolic (term) view of the class, row by row: ``m(row)``, ``s(row)``, ``outputs(row)``."""

    def __init__(
        self,
        N: int,
        in_fmt: str = "bf16",
        temperature: bool = False,
        alg: Optional[TermAlgebra] = None,
    ):
        self.N, self.in_fmt, self.temperature = N, in_fmt, temperature
        self.alg = alg or TermAlgebra()
        self.T = self.alg.sym("T") if temperature else None

    def c(self) -> Term:
        return scale(self.alg, self.T)

    def leaf(self, row: int, i: int) -> Term:
        return self.alg.x(row, i)

    def max_reduction(self, row: int) -> Reduction:
        return max_reduction(self.N, lambda fp, i: self.leaf(row, i))

    def m(self, row: int) -> Term:
        return self.max_reduction(row).canonical(self.alg)

    def z(self, row: int, i: int, m: Optional[Term] = None) -> Term:
        a = self.alg
        m = self.m(row) if m is None else m
        return shifted(a, self.leaf(row, i), a.neg(m), self.c())

    def sum_reduction(self, row: int) -> Reduction:
        a = self.alg
        nm, c = a.neg(self.m(row)), self.c()
        return sum_reduction(
            self.N, lambda fp, i: weight(fp, shifted(fp, self.leaf(row, i), nm, c))
        )

    def s(self, row: int) -> Term:
        return self.sum_reduction(row).canonical(self.alg)

    def lse(self, row: int) -> Term:
        a = self.alg
        return lse_nat(a, self.m(row), self.c(), lg2(a, self.s(row)))

    def logp(self, row: int, i: int) -> Term:
        a = self.alg
        return logp(a, self.z(row, i), a.neg(lg2(a, self.s(row))))

    def outputs(self, row: int) -> Dict[str, Any]:
        a = self.alg
        m, s, c = self.m(row), self.s(row), self.c()
        nm, inv_s, nlg = a.neg(m), inv_sum(a, s), a.neg(lg2(a, s))
        probs, logprobs = [], []
        for i in range(self.N):
            z = shifted(a, self.leaf(row, i), nm, c)
            probs.append(a.cvt_out(prob(a, weight(a, z), inv_s), self.in_fmt))
            logprobs.append(logp(a, z, nlg))
        return {
            "m": m,
            "s": s,
            "probs": probs,
            "logprobs": logprobs,
            "lse": lse_nat(a, m, c, lg2(a, s)),
        }


def spec_fingerprint(N: int, in_fmt: str = "bf16", temperature: bool = False) -> str:
    """Identifier of the numerical specification for vocabulary ``N``: a structural hash (``sterm.fingerprint_dag``,
    linear in the term DAG, so computable at every vocabulary) of the two reductions, the first and last probability (the
    last exercises the padding), the log-softmax element and the ``lse`` of row 0."""
    spec = LSESpec(N, in_fmt=in_fmt, temperature=temperature)
    outs = spec.outputs(0)
    parts = [
        fingerprint(outs["m"]),
        fingerprint(outs["s"]),
        fingerprint(outs["probs"][0]),
        fingerprint(outs["probs"][N - 1]),
        fingerprint(outs["logprobs"][0]),
        fingerprint(outs["lse"]),
    ]
    return "lse-row-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


LOG2E_F32 = fp32.to_f32(LOG2E)
LN2_F32 = fp32.to_f32(LN2)


def scale_value(T: Optional[float]) -> float:
    """The host-side ``c`` for a scalar temperature (``fp32.div``, the device's ``div.rn``) or none."""
    return LOG2E_F32 if T is None else fp32.div(LOG2E_F32, fp32.to_f32(T))
