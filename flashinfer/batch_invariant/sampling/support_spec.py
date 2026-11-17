# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""``support.row.v1``: the kept set of a row under top-k / top-p / min-p with ties kept -- the selection stage of the
truncated samplers ``top_k_sampling_from_probs`` / ``top_p_sampling_from_probs`` / ``min_p_sampling_from_probs`` /
``top_k_top_p_sampling_from_probs`` / ``top_k_top_p_sampling_from_logits`` under ``batch_invariant=True`` and of
``log_probs(..., top_k / top_p / min_p)``; no entry of its own.

Weights ``w_j >= 0`` (the given probabilities, or ``e_j = EX2(z_j)`` of ``lse.row`` for logits); ``S`` their canonical sum
tree. Every filter keeps ``{j : w_j >= threshold}`` for a per-row threshold, so the kept set is closed upwards (ties kept):

* top-k: ``threshold = the k-th largest weight`` (``k >= N`` keeps everything; ``k <= 0`` keeps nothing). Found by radix
  selection over the weights' 32-bit patterns (non-negative binary32 values order as unsigned integers): four 8-bit passes
  of 256-bin counts, the integer counts exact and order-free.
* top-p: ``threshold = the largest weight t with mass({w_j >= t}) >= p * S`` where the mass is compared in a declared fixed
  point: ``q_j = trunc(mul(w_j, div(2^FIX_BITS, S)))`` as a 64-bit integer (``mul`` then ``cvt.rzi``, the two declared float
  ops), ``mass(set) = sum of q_j`` (an integer sum, exact), and the target ``P = trunc(mul(p, 2^FIX_BITS))`` -- so the
  comparison is ``sum q_j >= P``. Radix selection over the patterns with per-bin integer masses. When the positive weights'
  mass does not reach ``P``, and whenever ``p >= 1`` (the filter is off: the target is declared unreachable, so the
  truncation's and the tree sum's rounding slack cannot drop a tail), every positive weight is kept; the zero weights are
  never kept (``threshold = +inf`` when no weight is positive).
* min-p: ``threshold = mul(w_max, min_p)`` (one declared multiply; ``w_max`` is the ``fmaxc`` tree over the weights, which
  for logits is ``e_max = EX2(0) = 1`` by construction of the shift -- the kernel uses that), kept iff ``w_j >= threshold``.
* order ``top_k_first``: top-k, then top-p on the kept set (its ``S`` is the kept mass, the masked tree); ``joint``: both on
  the original mass, intersected. min-p (when given) is applied last on the original maximum.

Outputs per row: the kept mask, ``(threshold, count, mass)`` with the threshold the effective one (the largest of the
applied filters': ``mask == (w >= threshold)``) and the mass the masked canonical tree (fp32) of the kept weights, the
renormalised probabilities ``cvt(mul(w_j, div(1, mass)))`` in the input dtype (``lse.row``'s ``prob`` chain with the kept
mass; ``+0`` outside the kept set) or the masked logits (the input with ``-inf`` outside the kept set), and, with
``lse.row``'s chains, the renormalised log-probabilities ``logp_j - log(mass / S)`` -- declared as
``mul(add(z_j, neg(lg2(mass))), LN2)`` for logits (the masked tree's ``lg2`` replaces the row's).

The specification identifier (:func:`spec_fingerprint`) hashes the filter set and order, the fixed-point resolution, the
selection rules as stated above (``SELECT_RULE``), the vocabulary and the weight form, and the structural hash of the sum
tree the mass is read from (the draw's total: the row class's ``s`` for logits, the sum over the given weights for
probabilities), so a change of any of them moves the class.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence

from ..fp_interface import exact as fp32
from ..fp_interface.term import Term

CLASS = "support.row.v1"
FIX_BITS = 44
FIX_SCALE = float(1 << FIX_BITS)
ORDERS = ("top_k_first", "joint")
INF_PATTERN = 0x7F800000  # the threshold that keeps nothing finite
#: The selection rules, named in the fingerprint (a change is a change of the class): the cut, the ties, the three
#: thresholds with their fallbacks, the two orders; since 2026-11-17 a row whose (kept) sum is below ~2^-84 -- where
#: ``div(2^44, S)`` overflows and every positive weight would saturate to the same fixed-point mass -- keeps nothing
#: under top-p and is invalid (unreachable for logits, whose maximum weight is 1).
SELECT_RULE = (
    "cut=w>=t;ties-kept;"
    "topk=kth-largest-pattern|k<=0:+inf|k>=N:0;"
    "topp=largest-positive-pattern:mass>=trunc(p*2^44)|unreached-or-p>=1:min-positive|no-positive-or-scale-overflow(sum<2^-84):+inf;"
    "minp=mul(wmax,minp);order=top_k_first:topp-on-kept-mass|joint:intersect;effective=max"
)


@dataclass(frozen=True)
class Filters:
    """One row's filter parameters (``None`` = the filter is off)."""

    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    order: str = "top_k_first"

    def __post_init__(self):
        if self.order not in ORDERS:
            raise ValueError(f"order={self.order!r}: one of {ORDERS}")
        if self.top_p is not None:
            object.__setattr__(self, "top_p", fp32.to_f32(float(self.top_p)))
        if self.min_p is not None:
            object.__setattr__(self, "min_p", fp32.to_f32(float(self.min_p)))

    @property
    def name(self) -> str:
        """The filter set's short name: the letters of the filters that are on (``k`` / ``p`` / ``m``) and ``_joint``."""
        on = (
            ("k", self.top_k is not None),
            ("p", self.top_p is not None),
            ("m", self.min_p is not None),
        )
        return "".join(t for t, flag in on if flag) + (
            "" if self.order == "top_k_first" else "_joint"
        )


def f32_pattern(x: float) -> int:
    """The unsigned 32-bit pattern of a non-negative binary32 value (orders as the value)."""
    b = fp32.f32_bits(x)
    assert b < 0x80000000 or x == 0.0, "weights are non-negative"
    return b & 0x7FFFFFFF


def fixed_scale(S: float) -> float:
    """``div(2^FIX_BITS, S)``, correctly rounded."""
    return fp32.div(FIX_SCALE, S)


def fixed_weight(w: float, scale: float) -> int:
    """``trunc(mul(w, scale))`` as an integer (``cvt.rzi.s64.f32`` of the ``.rn`` product: saturating, NaN reads 0): the
    fixed-point mass of a weight."""
    v = fp32.mul(w, scale)
    if v != v:
        return 0
    return int(v) if v != float("inf") else (1 << 63) - 1


UNREACHABLE = (
    1 << 63
)  # the target of ``p >= 1``: beyond any mass, so every positive weight is kept


def fixed_target(p: float) -> int:
    """``trunc(mul(p, 2^FIX_BITS))`` for ``p < 1``; ``p >= 1`` is the unreachable target (the filter keeps every positive
    weight)."""
    p = fp32.to_f32(p)
    return UNREACHABLE if p >= 1.0 else int(fp32.mul(p, FIX_SCALE))


def mass_term(N: int, keep: Sequence[bool], alg=None) -> Term:
    """The kept mass as a term: ``lse.row``'s canonical sum tree over the row's weights (symbols ``x_i``), the dropped
    elements ``+0`` (what the kernel's masked pass computes; ``Support.massTree`` in Lean)."""
    from .spec import sum_reduction
    from .sterm import TermAlgebra

    a = alg or TermAlgebra()
    return sum_reduction(
        N, lambda fp, i: fp.x(0, i) if keep[i] else fp.const(0.0)
    ).canonical(a)


def renorm_term(N: int, keep: Sequence[bool], j: int, alg=None) -> Term:
    """The renormalised probability of a kept element ``j`` as a term: ``lse.row``'s ``prob`` chain, ``mul(x_j, div(1,
    mass))`` with the kept mass tree (``Support.renormProb`` in Lean)."""
    from .spec import inv_sum, prob
    from .sterm import TermAlgebra

    a = alg or TermAlgebra()
    return prob(a, a.x(0, j), inv_sum(a, mass_term(N, keep, a)))


def kept_logp_term(N: int, keep: Sequence[bool], j: int, alg=None) -> Term:
    """The log-probability of a kept element ``j`` under the kept set as a term: ``lse.row``'s ``logp`` chain at ``j`` over
    the kept mass tree, ``mul(add(z_j, neg(lg2(mass))), LN2)`` with ``z_j`` the symbol of the token's shifted logit -- the
    drawn token's ``kept_logp`` and ``log_probs(..., top_k / top_p / min_p)`` at a given token (``SupportRow.keptLogp``)."""
    from .spec import lg2, logp
    from .sterm import TermAlgebra

    a = alg or TermAlgebra()
    return logp(a, a.sym(f"z{j}"), a.neg(lg2(a, mass_term(N, keep, a))))


def spec_fingerprint(
    filters: Filters,
    N: int,
    from_logits: bool = False,
    in_fmt: str = "f32",
    temperature: bool = False,
) -> str:
    """The class identifier: the filter set and order, the fixed-point resolution, the selection rules, the vocabulary,
    the weight form (the given probabilities, or ``lse.row``'s weights of the logits, with or without a temperature in
    the scale) and the structural hash of the sum tree the kept mass is read from (``draw_spec.DrawSpec.total``: the same
    tree for the draw, the row class and this class), with the input format (the renormalised probabilities convert to
    it)."""
    from .draw_spec import DrawSpec
    from .sterm import fingerprint_dag

    tree = fingerprint_dag(
        DrawSpec(
            int(N), bool(from_logits), in_fmt, temperature=bool(temperature)
        ).total(0)
    )
    parts = [
        CLASS,
        filters.name,
        f"fix{FIX_BITS}",
        SELECT_RULE,
        f"N{int(N)}",
        "logits" if from_logits else "probs",
        in_fmt,
        tree,
    ]
    return "support-row-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
