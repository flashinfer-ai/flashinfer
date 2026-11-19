# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The sampling class's symbolic terms: the engine's hash-consed terms
(:mod:`flashinfer.batch_invariant.fp_interface.term`) with the primitives the ``lse.row`` chains use (the rounded ``mul`` /
``div``, the commutative ``fmaxc`` join of the maximum tree, the exact ``neg``, the opaque ``ex2`` and the named opaque
``op("lg2")``, the input leaf ``x`` and the runtime scalar ``sym``), their canonical S-expression (what the Lean model
prints and the fingerprints hash) and their bit-exact evaluation through :mod:`flashinfer.batch_invariant.fp_interface.exact`
(the two opaque primitives are supplied by the caller: the device tabulates them).

The GEMM classes' algebra stays as it is: this module extends it for the sampling class only, so no other fingerprint
can move."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Optional

from ..fp_interface import exact as fp32
from ..fp_interface.term import Term
from ..fp_interface.term import TermAlgebra as _Base

__all__ = [
    "Term",
    "TermAlgebra",
    "evaluate",
    "fingerprint",
    "fingerprint_dag",
    "sexpr",
    "size",
]


class TermAlgebra(_Base):
    """The engine's algebra with the sampling class's primitives (``.rn`` where they round, exact otherwise)."""

    # -- leaves ----------------------------------------------------------------
    def x(self, row: int, i: int) -> Term:
        """Logit ``i`` of ``row`` (widened to binary32 exactly for the 16-bit inputs)."""
        return self._mk("x", row, i)

    def sym(self, name: str) -> Term:
        """A symbolic runtime scalar (the per-row temperature); evaluation needs a binding."""
        return self._mk("sym", name)

    # -- f32 ops (all .rn) -----------------------------------------------------
    def mul(self, a: Term, b: Term) -> Term:
        """``mul.rn.f32``, commutativity-normalised."""
        a, b = self._sorted2(a, b)
        return self._mk("mul", a, b)

    def div(self, a: Term, b: Term) -> Term:
        """``div.rn.f32``."""
        return self._mk("div", a, b)

    def fmaxc(self, a: Term, b: Term) -> Term:
        """``max.f32`` as a *commutative join*: bitwise commutative on every non-NaN pair (``max(+0, -0)`` and
        ``max(-0, +0)`` both give ``+0``; a NaN operand yields the other), so the operand order is normalised like
        ``add``; printed as ``(fmax a b)`` with the operands sorted."""
        a, b = self._sorted2(a, b)
        return self._mk("fmaxc", a, b)

    # -- exact and opaque primitives --------------------------------------------
    def neg(self, a: Term) -> Term:
        """``neg.f32``: a sign flip, exact."""
        return self._mk("neg", a)

    def ex2(self, a: Term) -> Term:
        """``ex2.approx.ftz.f32``: an opaque hardware primitive (no CPU model; the device tabulates its values for the
        exact evaluator). ``ex2(+0) = ex2(-0) = 1``, ``ex2(-inf) = +0``."""
        return self._mk("ex2", a)


def evaluate(
    t: Term,
    x_val: Callable[[int, int], float],
    memo: Optional[Dict[int, float]] = None,
    syms: Optional[Dict[str, Any]] = None,
) -> float:
    """Bit-exact evaluation of a term: ``x_val(row, i)`` gives the binary32 value of a logit, ``syms`` binds the symbolic
    scalars (:meth:`TermAlgebra.sym`) and the opaque primitives (``syms["ex2"]`` and ``syms["op:lg2"]``: functions of one
    binary32 value, tabulated on the device)."""
    if memo is None:
        memo = {}
    syms = syms or {}
    stack = [t]
    while stack:
        cur = stack[-1]
        if cur.uid in memo:
            stack.pop()
            continue
        deps = [a for a in cur.args if isinstance(a, Term) and a.uid not in memo]
        if deps:
            stack.extend(deps)
            continue
        stack.pop()
        op = cur.op
        A = cur.args
        if op == "x":
            v = float(x_val(A[0], A[1]))
        elif op == "const":
            v = A[0]
        elif op == "sym":
            v = float(syms[A[0]])
        elif op == "add":
            v = fp32.add(memo[A[0].uid], memo[A[1].uid])
        elif op == "mul":
            v = fp32.mul(memo[A[0].uid], memo[A[1].uid])
        elif op == "div":
            v = fp32.div(memo[A[0].uid], memo[A[1].uid])
        elif op == "fmaxc":
            v = fp32.fmaxc(memo[A[0].uid], memo[A[1].uid])
        elif op == "neg":
            v = fp32.neg(memo[A[0].uid])
        elif op == "ex2" or op.startswith("op:"):
            fn = syms.get(op)
            if fn is None:
                raise NotImplementedError(
                    f"no CPU model for the opaque primitive {op!r}; bind syms[{op!r}] to a device-produced function"
                )
            v = fn(memo[A[0].uid])
        elif op == "cvt_out":
            fmt = {"f32": fp32.F32, "bf16": fp32.BF16, "f16": fp32.F16}[A[1]]
            v = fp32.to_format(memo[A[0].uid], fmt)
        else:  # pragma: no cover
            raise ValueError(f"no CPU model for {op!r}")
        memo[cur.uid] = v
    return memo[t.uid]


def fingerprint(t: Term) -> str:
    """Structural sha256 of the canonical S-expression (op names, leaf coordinates, binary32 constants, association;
    commutative operands sorted): independent of interning ids and construction order."""
    return hashlib.sha256(sexpr(t).encode()).hexdigest()[:16]


def fingerprint_dag(t: Term) -> str:
    """A structural hash linear in the size of the term's DAG: every node's digest is the sha256 of its op, its constants
    and its children's digests, the children of the commutative nodes (``add``, ``mul``, ``fmaxc``) sorted by digest, so
    it is independent of interning ids, construction order and operand order like :func:`fingerprint`, but does not
    expand shared sub-terms (:func:`fingerprint` prints the S-expression, whose size is quadratic for a sum whose every
    leaf embeds the row's maximum). The class's fingerprint uses it; its values differ from :func:`fingerprint`'s."""
    memo: Dict[int, str] = {}
    stack = [t]
    while stack:
        cur = stack[-1]
        if cur.uid in memo:
            stack.pop()
            continue
        deps = [a for a in cur.args if isinstance(a, Term) and a.uid not in memo]
        if deps:
            stack.extend(deps)
            continue
        stack.pop()
        kids = [memo[a.uid] if isinstance(a, Term) else f"#{a!r}" for a in cur.args]
        if cur.op in ("add", "mul", "fmaxc"):
            kids = sorted(kids)
        memo[cur.uid] = hashlib.sha256(
            (cur.op + "(" + ",".join(kids) + ")").encode()
        ).hexdigest()
    return memo[t.uid][:16]


def size(t: Term) -> int:
    seen = set()
    stack = [t]
    while stack:
        c = stack.pop()
        if c.uid in seen:
            continue
        seen.add(c.uid)
        stack.extend(a for a in c.args if isinstance(a, Term))
    return len(seen)


def sexpr(t: Term) -> str:
    """Canonical S-expression of a term, the operands of the commutative nodes (``add``, ``mul``, ``fmaxc``) sorted by
    their own S-expression; identical on the Lean side (``Term.sexpr``), so the two implementations compare textually.
    The input leaf prints its column only: a row's term never depends on the row."""
    memo: Dict[int, str] = {}
    stack = [t]
    while stack:
        cur = stack[-1]
        if cur.uid in memo:
            stack.pop()
            continue
        deps = [a for a in cur.args if isinstance(a, Term) and a.uid not in memo]
        if deps:
            stack.extend(deps)
            continue
        stack.pop()
        op = cur.op
        A = [memo[a.uid] if isinstance(a, Term) else a for a in cur.args]
        if op == "x":
            out = f"(x {A[1]})"
        elif op == "const":
            out = (
                f"(const {A[0]!r})"
                if isinstance(A[0], str)
                else f"(const {float(A[0])!r})"
            )
        elif op == "sym":
            out = f"(sym {A[0]})"
        elif op in ("add", "mul"):
            a, b = sorted(A[:2])
            out = f"({op} {a} {b})"
        elif op == "fmaxc":
            a, b = sorted(A[:2])
            out = f"(fmax {a} {b})"
        elif op == "div":
            out = f"(div {A[0]} {A[1]})"
        elif op in ("neg", "ex2"):
            out = f"({op} {A[0]})"
        elif op == "cvt_out":
            out = f"(cvt {A[1]} {A[0]})"
        else:  # the named opaque primitives: ``(op:lg2 a)``
            out = "(" + op + " " + " ".join(map(str, A)) + ")"
        memo[cur.uid] = out
    return memo[t.uid]
