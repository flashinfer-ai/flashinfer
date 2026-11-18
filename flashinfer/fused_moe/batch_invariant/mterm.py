# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MoE classes' symbolic terms: the engine's hash-consed terms
(:mod:`flashinfer.batch_invariant.fp_interface.term`) with the primitives the MoE chains use (the exact ``neg`` /
``fmin`` / ``fmax`` / ``exp2i`` / ``rowmax``, the rounded ``div``, the named input leaves), their canonical
S-expression (what the Lean model prints), their DAG hash (what the fingerprints use) and their bit-exact evaluation
through :mod:`flashinfer.batch_invariant.fp_interface.exact`.

The GEMM classes' algebra stays as it is: this module extends it for the MoE chains only, so the GEMM fingerprints
cannot move. The quantization classes extend the same base separately (``quantization/qterm.py``); both extensions
share the engine's interning table and print the primitives they have in common alike."""

from __future__ import annotations

import struct
from typing import Any, Dict, Optional

from ...batch_invariant.fp_interface import exact as fp32
from ...batch_invariant.fp_interface.term import Term, TermAlgebra as _Base, to_f32
from ...batch_invariant.sampling.sterm import fingerprint_dag

__all__ = [
    "Term",
    "TermAlgebra",
    "evaluate",
    "fingerprint",
    "frexpe_f32",
    "frexpm_f32",
    "rowmax_f32",
    "sexpr",
    "size",
    "to_f32",
]


class TermAlgebra(_Base):
    """The engine's algebra with the MoE chains' primitives: ``div`` rounds (``.rn``), the others are exact."""

    # -- leaves ----------------------------------------------------------------
    def leaf(self, name: str, *coords: int) -> Term:
        """A named input leaf with integer coordinates (``leaf("l", e)``: a router logit; ``leaf("g", r, j)`` /
        ``leaf("u", r, j)``: the fp32 GEMM1 accumulators of slot ``r``, column ``j``; ``leaf("y", r, h)``: an expert
        output). Evaluation binds ``syms["leaf:" + name]`` to a function of the coordinates."""
        return self._mk("leaf:" + name, *coords)

    # -- f32 ops (.rn) ---------------------------------------------------------
    def div(self, a: Term, b: Term) -> Term:
        """``div.rn.f32``."""
        return self._mk("div", a, b)

    # -- exact (rounding-free) primitives ---------------------------------------
    def neg(self, a: Term) -> Term:
        """Sign flip (exact)."""
        return self._mk("neg", a)

    def fmin(self, a: Term, b: Term) -> Term:
        """``min.f32`` (exact, not commutativity-normalised: the operand order decides which signed zero is
        returned; the chains only clamp against non-zero constants)."""
        return self._mk("fmin", a, b)

    def fmax(self, a: Term, b: Term) -> Term:
        """``max.f32`` (exact, order-preserving like ``fmin``)."""
        return self._mk("fmax", a, b)

    def exp2i(self, k: Term) -> Term:
        """The binary32 ``2^k`` for an integer-valued ``k`` in ``[-126, 127]`` (exact; built from the exponent bits on
        the device, ``ldexp`` on the CPU)."""
        return self._mk("exp2i", k)

    def fabs(self, a: Term) -> Term:
        """``abs.f32`` (exact: clears the sign bit)."""
        return self._mk("fabs", a)

    def fmaxc(self, a: Term, b: Term) -> Term:
        """``max.f32`` with the device's zero rule (``max(-0, +0) = max(+0, -0) = +0``), commutativity-normalised: the
        clamp of a value against zero, whose sign the order-preserving ``fmax`` model would get wrong on a ``-0``
        operand (the quantizer classes' ``fmaxc``)."""
        a, b = self._sorted2(a, b)
        return self._mk("fmaxc", a, b)

    def sqrt(self, a: Term) -> Term:
        """``sqrt.rn.f32`` (correctly rounded)."""
        return self._mk("sqrt", a)

    def frexpm(self, a: Term) -> Term:
        """The mantissa of ``|a|`` as a binary32 in [1/2, 1): the significand bits under the exponent of 1/2 (exact bit
        operations; a subnormal or zero magnitude gives its raw significand field the same way)."""
        return self._mk("frexpm", a)

    def frexpe(self, a: Term) -> Term:
        """The exponent of ``|a|`` as a binary32: the biased exponent field minus 126 (so ``|a| = frexpm 2^frexpe`` for
        a normal ``a``; exact)."""
        return self._mk("frexpe", a)

    def rowmax(self, *vals: Term) -> Term:
        """The maximum of ``vals`` (n-ary, exact in value for every association; the sign of a zero maximum is
        unspecified and, where the chains use it, immaterial: docs/moe_scoping.md section 3.1)."""
        return self._mk("rowmax", *vals)

    def where_nan(self, x: Term, a: float, b: Term) -> Term:
        """The routing's NaN rule (``a`` where the logit ``x`` is NaN, else ``b``): a row rule outside the term algebra,
        so the symbolic term is ``b`` (the terms and the Lean statements are over non-NaN inputs)."""
        return b


def rowmax_f32(*vals: float) -> float:
    """``rowmax`` with the device's ``max.f32`` NaN rule: the maximum of the non-NaN values (NaN when there is none);
    its value does not depend on the association (the sign of a zero maximum never reaches the bits)."""
    m = float("nan")
    for v in vals:
        if v == v and not v <= m:
            m = v
    return m


def frexpm_f32(x: float) -> float:
    """``frexpm``: the significand field of ``|x|`` under the exponent of 1/2 (bit operations on the binary32 pattern)."""
    bits = fp32.f32_bits(x) & 0x7FFFFFFF
    return struct.unpack("<f", struct.pack("<I", (bits & 0x007FFFFF) | 0x3F000000))[0]


def frexpe_f32(x: float) -> float:
    """``frexpe``: the biased exponent field of ``|x|`` minus 126, as a float (exact)."""
    bits = fp32.f32_bits(x) & 0x7FFFFFFF
    return float(((bits >> 23) & 0xFF) - 126)


def evaluate(
    t: Term,
    syms: Optional[Dict[str, Any]] = None,
    memo: Optional[Dict[int, float]] = None,
) -> float:
    """Bit-exact evaluation of a term through :mod:`flashinfer.batch_invariant.fp_interface.exact`. ``syms`` binds
    the symbolic scalars (:meth:`TermAlgebra.sym`, by name) and the input leaves (:meth:`TermAlgebra.leaf`, under
    ``"leaf:" + name``, to a function of the coordinates). Every MoE term stands on named leaves, so the binding is the
    first argument; the GEMM classes' input callbacks have no counterpart here."""
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
        if op == "const":
            v = A[0]
        elif op == "sym":
            v = float(syms[A[0]])
        elif op == "add":
            v = fp32.add(memo[A[0].uid], memo[A[1].uid])
        elif op == "mul":
            v = fp32.mul(memo[A[0].uid], memo[A[1].uid])
        elif op == "div":
            v = fp32.div(memo[A[0].uid], memo[A[1].uid])
        elif op == "neg":
            v = fp32.neg(memo[A[0].uid])
        elif op == "fmin":
            v = fp32.fmin(memo[A[0].uid], memo[A[1].uid])
        elif op == "fmax":
            v = fp32.fmax(memo[A[0].uid], memo[A[1].uid])
        elif op == "fmaxc":
            v = fp32.fmaxc(memo[A[0].uid], memo[A[1].uid])
        elif op == "exp2i":
            v = fp32.exp2i(memo[A[0].uid])
        elif op == "fabs":
            v = fp32.fabs(memo[A[0].uid])
        elif op == "sqrt":
            v = fp32.sqrt(memo[A[0].uid])
        elif op == "frexpm":
            v = frexpm_f32(memo[A[0].uid])
        elif op == "frexpe":
            v = frexpe_f32(memo[A[0].uid])
        elif op == "rowmax":
            v = rowmax_f32(*[memo[a.uid] for a in A])
        elif op.startswith("leaf:"):
            fn = syms.get(op)
            if fn is None:
                raise NotImplementedError(f"no binding for leaf {op!r}")
            v = float(fn(*A))
        elif op == "cvt_out":
            kind = A[1]
            if kind in ("e4m3", "e5m2"):
                v = fp32.to_fp8(memo[A[0].uid], kind)
            else:
                fmt = {"f32": fp32.F32, "bf16": fp32.BF16, "f16": fp32.F16}[kind]
                v = fp32.to_format(memo[A[0].uid], fmt)
        else:  # pragma: no cover
            raise ValueError(f"no CPU model for {op!r}")
        memo[cur.uid] = v
    return memo[t.uid]


def fingerprint(t: Term) -> str:
    """Structural hash of a term, linear in its DAG (the sampling class's
    :func:`flashinfer.batch_invariant.sampling.sterm.fingerprint_dag`): every node's digest is the sha256 of its op, its
    constants and its children's digests, the commutative operands sorted by digest, so it is independent of interning
    ids, construction order and operand order and never expands a shared sub-term. The printed S-expression
    (:func:`sexpr`) is for the Lean cases: a routing weight's expansion embeds the softmax sum once per leaf, a size
    unbounded in the expert count."""
    return fingerprint_dag(t)


def size(t: Term) -> int:
    """Number of distinct nodes reachable from ``t``."""
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
    """Canonical S-expression of a term, the operands of the commutative nodes (``add``, ``mul``) sorted by their own
    S-expression, ``fmin`` / ``fmax`` / ``div`` in operand order, ``rowmax`` n-ary, a leaf as its name followed by its
    coordinates (``(l 3)``, ``(y 1 0)``); identical on the Lean side (``Term.sexpr``), so the two implementations
    compare textually."""
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
        if op == "const":
            out = (
                f"(const {A[0]!r})"
                if isinstance(A[0], str)
                else f"(const {float(A[0])!r})"
            )
        elif op == "sym":
            out = f"(sym {A[0]})"
        elif op in ("add", "mul", "fmaxc"):
            a, b = sorted(A[:2])
            out = f"({op} {a} {b})"
        elif op in ("neg", "exp2i", "fabs", "sqrt", "frexpm", "frexpe"):
            out = f"({op} {A[0]})"
        elif op in ("div", "fmin", "fmax"):
            out = f"({op} {A[0]} {A[1]})"
        elif op == "rowmax":
            out = "(rowmax " + " ".join(A) + ")"
        elif op.startswith("leaf:"):
            out = "(" + op[5:] + "".join(f" {a}" for a in A) + ")"
        elif op == "cvt_out":
            out = f"(cvt {A[1]} {A[0]})"
        else:
            out = "(" + op + " " + " ".join(map(str, A)) + ")"
        memo[cur.uid] = out
    return memo[t.uid]
