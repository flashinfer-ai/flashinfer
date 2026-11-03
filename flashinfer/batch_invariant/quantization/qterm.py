# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The quantization classes' symbolic terms: the engine's hash-consed terms
(:mod:`flashinfer.batch_invariant.fp_interface.term`) with the primitives the quantizers' chains use (the exact
``fabs`` / ``fmin`` / ``nzor`` / ``exp2i``, the rounded ``mul`` / ``div`` / ``fma``, the commutative ``fmaxc`` join of the
amax trees, the input leaf ``x`` and the runtime scalar ``sym``), their canonical S-expression (what the Lean model prints
and the fingerprints hash) and their bit-exact evaluation through :mod:`flashinfer.batch_invariant.fp_interface.exact`.

The GEMM classes' algebra stays as it is: this module extends it for the quantizers only, so the GEMM fingerprints
cannot move."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Dict, Optional

from ..fp_interface import exact as fp32
from ..fp_interface.term import Term, TermAlgebra as _Base, to_f32

__all__ = ["Term", "TermAlgebra", "evaluate", "fingerprint", "sexpr", "size"]


class TermAlgebra(_Base):
    """The engine's algebra with the quantizers' primitives (all ``.rn`` where they round, exact otherwise)."""

    # -- leaves ----------------------------------------------------------------
    def x(self, row: int, i: int) -> Term:
        """Input element ``i`` of ``row`` (widened to binary32 exactly for the 16-bit inputs)."""
        return self._mk("x", row, i)

    def sym(self, name: str) -> Term:
        """A symbolic runtime scalar (the per-tensor global scale); evaluation needs a binding."""
        return self._mk("sym", name)

    # -- f32 ops (all .rn) -----------------------------------------------------
    def mul(self, a: Term, b: Term) -> Term:
        """``mul.rn.f32``, commutativity-normalised."""
        a, b = self._sorted2(a, b)
        return self._mk("mul", a, b)

    def fma(self, a: Term, b: Term, c: Term) -> Term:
        a, b = self._sorted2(a, b)
        return self._mk("fma", a, b, c)

    def div(self, a: Term, b: Term) -> Term:
        """``div.rn.f32``."""
        return self._mk("div", a, b)

    def fmaxc(self, a: Term, b: Term) -> Term:
        """``max.f32`` as a *commutative join*: bitwise commutative on every non-NaN pair (``max(+0, -0)`` and
        ``max(-0, +0)`` both give ``+0``), so the operand order is normalised like ``add``; printed as ``(fmax a b)``
        with the operands sorted. The amax trees apply it to ``fabs(x) >= +0`` operands."""
        a, b = self._sorted2(a, b)
        return self._mk("fmaxc", a, b)

    # -- exact (rounding-free) primitives ---------------------------------------
    def fmin(self, a: Term, b: Term) -> Term:
        """``min.f32`` (exact; not commutativity-normalised: the operand order decides which signed zero is returned)."""
        return self._mk("fmin", a, b)

    def fmax(self, a: Term, b: Term) -> Term:
        return self._mk("fmax", a, b)

    def exp2i(self, k: Term) -> Term:
        """The binary32 ``2^k`` for an integer-valued ``k`` in ``[-126, 127]`` (exact: built from the exponent bits)."""
        return self._mk("exp2i", k)

    def fabs(self, a: Term) -> Term:
        """``abs.f32`` (exact: the sign bit is cleared)."""
        return self._mk("fabs", a)

    def nzor(self, a: Term, b: Term) -> Term:
        """``a`` if ``a != 0`` else ``b`` (an exact comparison-and-select, ``setp.ne`` + ``selp``)."""
        return self._mk("nzor", a, b)


def evaluate(
    t: Term,
    x_val: Callable[[int, int], float],
    memo: Optional[Dict[int, float]] = None,
    syms: Optional[Dict[str, Any]] = None,
) -> float:
    """Bit-exact evaluation of a term: ``x_val(row, i)`` gives the binary32 value of an input element, ``syms`` binds the
    symbolic scalars (:meth:`TermAlgebra.sym`)."""
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
        elif op == "fma":
            v = fp32.fma(memo[A[0].uid], memo[A[1].uid], memo[A[2].uid])
        elif op == "div":
            v = fp32.div(memo[A[0].uid], memo[A[1].uid])
        elif op == "fmaxc":
            v = fp32.fmaxc(memo[A[0].uid], memo[A[1].uid])
        elif op == "fmin":
            v = fp32.fmin(memo[A[0].uid], memo[A[1].uid])
        elif op == "fmax":
            v = fp32.fmax(memo[A[0].uid], memo[A[1].uid])
        elif op == "exp2i":
            v = fp32.exp2i(memo[A[0].uid])
        elif op == "fabs":
            v = fp32.fabs(memo[A[0].uid])
        elif op == "nzor":
            v = fp32.nzor(memo[A[0].uid], memo[A[1].uid])
        elif op == "cvt_out":
            kind = A[1]
            if kind in ("e4m3", "e5m2"):
                v = fp32.to_fp8(memo[A[0].uid], kind)
            elif kind == "e2m1":
                v = fp32.to_e2m1(memo[A[0].uid])
            elif kind.startswith(
                "ue8m0_"
            ):  # the MX scale conversion, "ue8m0_<rp|rz|rn>"
                v = fp32.to_ue8m0(memo[A[0].uid], kind[6:])
            else:
                fmt = {"f32": fp32.F32, "bf16": fp32.BF16, "f16": fp32.F16}[kind]
                v = fp32.to_format(memo[A[0].uid], fmt)
        else:  # pragma: no cover
            raise ValueError(f"no CPU model for {op!r}")
        memo[cur.uid] = v
    return memo[t.uid]


def fingerprint(t: Term) -> str:
    """Structural sha256 of the canonical S-expression (op names, leaf coordinates, binary32 constants, association;
    commutative operands sorted): independent of interning ids and construction order."""
    return hashlib.sha256(sexpr(t).encode()).hexdigest()[:16]


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
    """Canonical S-expression of a term, the operands of the commutative nodes (``add``, ``mul``, ``fma``'s first two,
    ``fmaxc``) sorted by their own S-expression; identical on the Lean side (``Term.sexpr``), so the two implementations
    compare textually. The input leaf prints its column only: a row's term never depends on the row."""
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
        elif op == "fma":
            a, b = sorted(A[:2])
            out = f"(fma {a} {b} {A[2]})"
        elif op == "fmaxc":
            a, b = sorted(A[:2])
            out = f"(fmax {a} {b})"
        elif op in ("exp2i", "fabs"):
            out = f"({op} {A[0]})"
        elif op in ("div", "fmin", "fmax", "nzor"):
            out = f"({op} {A[0]} {A[1]})"
        elif op == "cvt_out":
            out = f"(cvt {A[1]} {A[0]})"
        else:
            out = "(" + op + " " + " ".join(map(str, A)) + ")"
        memo[cur.uid] = out
    return memo[t.uid]


_ = (math, to_f32)  # re-exported names used by the specs' constants
