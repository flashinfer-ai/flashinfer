# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Hash-consed symbolic terms over uninterpreted floating-point primitives.

A term records *which* operations were applied in *which* association, not their numeric value. Two
configurations are canonical-equivalent iff they build the same term for every output element. The
commutative primitives (``add``, ``mul``) normalise their operand order (IEEE addition and multiplication are
bitwise commutative), so two placements that only swap the operands compare equal. The GEMM's tensor-core
step is an opaque primitive (:meth:`TermAlgebra.op`): the term fixes the order in which the steps are applied,
nothing inside a step. A runtime scalar the kernel reads as given (the fp8 classes' ``alpha``) is a ``sym`` leaf.
"""

from __future__ import annotations

import itertools
import math
import struct
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True, eq=False)
class Term:
    op: str
    args: Tuple[Any, ...]
    uid: int

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.op}#{self.uid}"


_UIDS = itertools.count()  # global: terms from different algebras may be mixed
_TABLE: Dict[
    Tuple[Any, ...], Term
] = {}  # global interning: structural equality == identity


def to_f32(value: float) -> float:
    """The binary32 nearest to ``value`` (round to nearest even), as a Python float."""
    return struct.unpack("f", struct.pack("f", float(value)))[0]


class TermAlgebra:
    """Builds interned terms."""

    def __init__(self):
        self._table = _TABLE

    # -- interning -------------------------------------------------------------
    def _mk(self, op: str, *args: Any) -> Term:
        key = (op,) + tuple(a.uid if isinstance(a, Term) else ("c", a) for a in args)
        t = self._table.get(key)
        if t is None:
            t = Term(op, args, next(_UIDS))
            self._table[key] = t
        return t

    @staticmethod
    def _sorted2(a: Term, b: Term):
        return (a, b) if a.uid <= b.uid else (b, a)

    # -- leaves ----------------------------------------------------------------
    def const(self, value: float) -> Term:
        # constants are binary32 values; keyed on the bit pattern so +0.0 and -0.0 differ
        value = to_f32(value)
        return self._mk("const", value, math.copysign(1.0, value))

    def sym(self, name: str) -> Term:
        """A runtime scalar the kernel reads as given (the fp8 classes' ``alpha``, the attention class's ``c = sm_scale * log2(e)``
        and ``nc = -c``): a leaf, not a computation."""
        return self._mk("sym", name)

    # -- primitives --------------------------------------------------------------
    def add(self, a: Term, b: Term) -> Term:
        """``add.rn.f32``, commutativity-normalised."""
        a, b = self._sorted2(a, b)
        return self._mk("add", a, b)

    def mul(self, a: Term, b: Term) -> Term:
        """``mul.rn.f32``, commutativity-normalised (IEEE multiplication is bitwise commutative)."""
        a, b = self._sorted2(a, b)
        return self._mk("mul", a, b)

    def fma(self, a: Term, b: Term, c: Term) -> Term:
        """``fma.rn.f32``; the product's operands commutativity-normalised."""
        a, b = self._sorted2(a, b)
        return self._mk("fma", a, b, c)

    def div(self, a: Term, b: Term) -> Term:
        """``div.rn.f32``."""
        return self._mk("div", a, b)

    def fmaxc(self, a: Term, b: Term) -> Term:
        """``max.f32`` (NVVM ``fmax``, maximumNumber) as a *commutative join*: it returns ``+0`` for both ``max(+0, -0)`` and
        ``max(-0, +0)`` on SM100 / SM103 / SM107, so it is bitwise commutative on every non-NaN pair and the operand order is
        normalised like ``add``. Printed as ``(fmax a b)`` with the operands sorted."""
        a, b = self._sorted2(a, b)
        return self._mk("fmaxc", a, b)

    def ex2(self, a: Term) -> Term:
        """``ex2.approx.ftz.f32``: an opaque hardware primitive, deterministic per architecture (``ex2(+0) = ex2(-0) = 1``,
        ``ex2(-inf) = 0``)."""
        return self._mk("ex2", a)

    def cvt_out(self, a: Term, fmt_name: str) -> Term:
        """``cvt.rn.<fmt>.f32``: the one conversion of the epilogue."""
        return self._mk("cvt_out", a, fmt_name)

    def op(self, name: str, *args) -> Term:
        """An opaque, non-commutative primitive whose internal rounding is hardware-defined (one ``tcgen05.mma``
        K-step, an operand slice); only the dataflow around it is specified."""
        return self._mk("op:" + name, *args)


def fingerprint(t: Term) -> str:
    """Structural sha256 of the canonical S-expression of a term (op names, leaf coordinates, binary32
    constants, association; commutative operands sorted), hence independent of interning ids and of the
    order in which terms were constructed in the process."""
    import hashlib

    return hashlib.sha256(sexpr(t).encode()).hexdigest()[:16]


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
    """Canonical S-expression of a term, with the operands of commutative nodes (``add``, ``mul``, ``fmax``, ``fma``'s
    product) sorted by their own S-expression. Identical on the Lean side (``GTerm.sexpr``), so the two implementations can be compared
    textually."""
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
        elif op in ("add", "mul"):
            a, b = sorted(A[:2])
            out = f"({op} {a} {b})"
        elif op == "fma":
            a, b = sorted(A[:2])
            out = f"(fma {a} {b} {A[2]})"
        elif op == "div":
            out = f"(div {A[0]} {A[1]})"
        elif op == "fmaxc":
            a, b = sorted(A[:2])
            out = f"(fmax {a} {b})"
        elif op == "ex2":
            out = f"(ex2 {A[0]})"
        elif op == "cvt_out":
            out = f"(cvt {A[1]} {A[0]})"
        elif op == "op:mma_bf16_k16":
            out = f"(mma16 {A[0]} {A[1]})"  # one tcgen05.mma K-step: (accumulator, operand slice)
        elif op == "op:mma_fp8_k32":
            out = f"(mma32 {A[0]} {A[1]})"  # one kind::f8f6f4 K-step of 32 fp8 elements
        elif op == "op:ab_kslice":
            out = f"(ab {A[0]} {A[1]})"  # the (A, B) K-slice of one step
        elif op == "op:a_rows":
            out = f"(a {A[0]} {A[1]})"  # A's row m at step t
        elif op == "op:b_cols":
            out = f"(b {A[0]} {A[1]})"  # B's column n at step t
        elif op == "op:bias_col":
            out = f"(bias {A[0]})"  # the bias of output column n, widened to binary32 exactly
        elif op == "op:mma_mxfp8_k32":
            out = f"(mma32mx {A[0]} {A[1]})"  # one kind::mxf8f6f4 block-scaled K-step of 32 fp8 elements
        elif op == "op:mma_mxfp8_k64":
            out = f"(mma64mx8 {A[0]} {A[1]})"  # the sm_107 block-scaled K-step of 64 fp8 elements
        elif op == "op:mma_nvfp4_k64":
            out = f"(mma64bs {A[0]} {A[1]})"  # one kind::mxf4nvf4 block-scaled K-step of 64 fp4 elements (e4m3 scales per 16)
        elif op == "op:mma_nvfp4_k128":
            out = f"(mma128bs {A[0]} {A[1]})"  # the sm_107 block-scaled K-step of 128 fp4 elements (e4m3 scales per 16)
        elif op == "op:mma_mxfp4_k64":
            out = f"(mma64mx {A[0]} {A[1]})"  # one kind::mxf4 block-scaled K-step of 64 fp4 elements (UE8M0 scales per 32)
        elif op == "op:mma_mxfp4_k128":
            out = f"(mma128mx4 {A[0]} {A[1]})"  # the sm_107 block-scaled K-step of 128 fp4 elements (UE8M0 scales per 32)
        elif op == "op:absf_kslice":
            out = f"(absf {A[0]} {A[1]} {A[2]} {A[3]})"  # the (A, SFA, B, SFB) K-slice of one block-scaled step
        elif op == "op:sfa_rows":
            out = f"(sfa {A[0]} {A[1]})"  # A's block scales of row m at step t
        elif op == "op:sfb_cols":
            out = f"(sfb {A[0]} {A[1]})"  # B's block scales of column n at step t
        elif op == "op:sfa_blk":
            out = f"(sfa_blk {A[0]} {A[1]})"  # the fp32 scale of A's row m for K block kb (the block-scale class)
        elif op == "op:sfb_blk":
            out = f"(sfb_blk {A[0]} {A[1]})"  # the fp32 scale of B's column block n // 128 for K block kb
        else:
            out = "(" + op + " " + " ".join(map(str, A)) + ")"
        memo[cur.uid] = out
    return memo[t.uid]
