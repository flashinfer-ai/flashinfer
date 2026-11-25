# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""``sample.icdf.v1``: the inverse-CDF draw as a reading of the row class's canonical sum tree (the class behind
``flashinfer.sampling.sampling_from_probs(..., batch_invariant=True)`` and ``sampling_from_logits(...,
batch_invariant=True)``).

Weights ``w_j``: the given probabilities (``_from_probs``, widened exactly) or ``e_j = EX2(z_j)`` of ``lse.row``
(``_from_logits``). ``S`` is the canonical ``add`` tree over the ``w_j`` (``spec.sum_reduction``), ``u`` the fp32 uniform of
the row's stream (``stream.uniform32`` of word 0 of block ``draw``), the threshold ``t = mul(u, S)``.

The tree order: element ``(lane, block, elem)`` has the address ``a = (lane * Kp + block) * 8 + elem`` -- the order in
which the canonical tree combines (elem fold innermost, block bits, lane bits outermost). The prefix of a slot is
``outer = add-chain over the set bits of its address, ascending, of the left-sibling node`` (the aligned group just below
the bit: blocks of the same lane for a block bit, whole lanes for a lane bit), from ``+0``; the inclusive prefix of an
element is ``add(outer, fold_e)`` with ``fold_e`` the slot fold's running value through the element. The drawn token is the
first present element in tree order with ``w_j > 0`` and ``add(outer, fold_e) >= t``; if none (rounding at ``u`` near 1), the
last present element with ``w_j > 0``; if ``S`` is not a positive finite number, the row is invalid (token ``-1``). The
positivity condition is part of the rule (``SELECT_RULE``): the inclusive prefixes are not monotone across slot boundaries
(the last prefix of a slot and the ``outer`` of the next are different ``add`` chains over the same nodes -- equal exactly,
a rounding apart in binary32), so without it a zero-weight slot head could be the first crossing and a masked token drawn
(found in review, 2026-11-11, at ~1e-5 per draw on half-masked rows).

Every node is a value every placement already holds (the thread's own block tree, the exchanged warp / CTA siblings, the
butterfly partners), so the draw adds no reduction and no numerics beyond these declared ``add`` chains, and is as
batch- and config-invariant as the tree.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple

from .schedule import LANES, V, padded_blocks
from .spec import LSESpec, shifted, sum_reduction, weight
from .sterm import Term, TermAlgebra
from .sterm import fingerprint_dag as fingerprint
from .stream import STREAM_NAME, stream_fingerprint

CLASS = "sample.icdf.v1"
#: The selection rule, named in the fingerprint (a change of the rule is a change of the class): the first present
#: element in tree order with a positive weight whose inclusive prefix reaches the threshold, else the last present
#: element with a positive weight; a row whose sum is not a positive finite number is invalid.
SELECT_RULE = (
    "first-present-positive-crossing;else-last-present-positive;invalid-unless-0<S<inf"
)


def tree_address(lane: int, block: int, elem: int, Kp: int) -> int:
    """The tree address of element ``elem`` of slot ``(lane, block)`` for ``Kp`` padded blocks."""
    return (lane * Kp + block) * V + elem


def address_of_index(i: int, Kp: int) -> int:
    """The tree address of flat element ``i`` (``i = (block * 32 + lane) * 8 + elem``)."""
    slot, elem = divmod(i, V)
    block, lane = divmod(slot, LANES)
    return tree_address(lane, block, elem, Kp)


def index_of_address(a: int, Kp: int) -> int:
    """The flat element index of tree address ``a`` (the inverse of :func:`address_of_index`)."""
    slot, elem = divmod(a, V)
    lane, block = divmod(slot, Kp)
    return (block * LANES + lane) * V + elem


def prefix_nodes(lane: int, block: int, Kp: int) -> List[Tuple[str, int, int]]:
    """The left-sibling nodes of slot ``(lane, block)`` in ascending bit order: ``("block", level, group)`` is the
    pairwise node over blocks ``[group << level, (group + 1) << level)`` of this lane; ``("lane", level, group)`` the node
    over lanes ``[group << level, (group + 1) << level)`` (all blocks)."""
    out = []
    kb = Kp.bit_length() - 1
    for b in range(kb):
        if (block >> b) & 1:
            out.append(("block", b, (block >> b) - 1))
    for k in range(5):
        if (lane >> k) & 1:
            out.append(("lane", k, (lane >> k) - 1))
    return out


class DrawSpec:
    """Symbolic view on a toy vocabulary: the node terms of the canonical sum tree and the inclusive prefix of an
    element."""

    def __init__(
        self,
        N: int,
        from_logits: bool = False,
        in_fmt: str = "f32",
        alg: Optional[TermAlgebra] = None,
        temperature: bool = False,
    ):
        """The symbolic view of vocabulary ``N``: the weights as given, or the ``lse.row`` weights of the
        logits with ``from_logits`` (``in_fmt`` and ``temperature`` as the row class takes them)."""
        self.N, self.from_logits, self.in_fmt = N, from_logits, in_fmt
        self.alg = alg or TermAlgebra()
        self.Kp = padded_blocks(N)
        self.lse = (
            LSESpec(N, in_fmt=in_fmt, temperature=temperature, alg=self.alg)
            if from_logits
            else None
        )
        self._nm: dict = {}  # the negated row maximum, one term per row

    def weight(self, row: int, i: int) -> Optional[Term]:
        """The weight term of element ``i`` of ``row`` (``None`` beyond the vocabulary): the leaf itself, or
        ``EX2(mul(add(x, neg(m)), c))`` for logits."""
        if i >= self.N:
            return None
        a = self.alg
        if self.from_logits:
            assert self.lse is not None
            nm = self._nm.get(row)
            if (
                nm is None
            ):  # built once per row: rebuilt per element, the term was quadratic in N (a day at 152064)
                nm = self._nm[row] = a.neg(self.lse.m(row))
            return weight(a, shifted(a, a.x(row, i), nm, self.lse.c()))
        return a.x(row, i)

    def slot_fold(self, row: int, lane: int, block: int) -> List[Optional[Term]]:
        """The running values of the slot fold (``add`` from ``+0``), one per element; absent elements ``None``."""
        a = self.alg
        acc: Optional[Term] = None
        out: List[Optional[Term]] = []
        for e in range(V):
            w = self.weight(row, (block * LANES + lane) * V + e)
            if w is None:
                out.append(None)
                continue
            acc = a.add(a.const(0.0) if acc is None else acc, w)
            out.append(acc)
        return out

    def slot_total(self, row: int, lane: int, block: int) -> Optional[Term]:
        """The slot fold through its last present element (``None`` for an absent slot)."""
        vals = [v for v in self.slot_fold(row, lane, block) if v is not None]
        return vals[-1] if vals else None

    def node(
        self, row: int, kind: str, level: int, group: int, lane: int = 0
    ) -> Optional[Term]:
        """The canonical sub-tree node (absent leaves dropped, as ``canon._pairwise`` does)."""
        a = self.alg

        def pair(vals):
            vals = list(vals)
            while len(vals) > 1:
                vals = [
                    (y if x is None else x if y is None else a.add(x, y))
                    for x, y in zip(vals[0::2], vals[1::2], strict=True)
                ]
            return vals[0]

        if kind == "block":
            return pair(
                self.slot_total(row, lane, b)
                for b in range(group << level, (group + 1) << level)
            )
        return pair(
            self.node(row, "block", self.Kp.bit_length() - 1, 0, ln)
            for ln in range(group << level, (group + 1) << level)
        )

    def outer(self, row: int, lane: int, block: int) -> Term:
        """The ``outer`` prefix of slot ``(lane, block)``: the ``add`` chain from ``+0`` over its left-sibling
        nodes in ascending bit order."""
        a = self.alg
        acc = a.const(0.0)
        for kind, level, group in prefix_nodes(lane, block, self.Kp):
            n = self.node(row, kind, level, group, lane)
            if n is not None:
                acc = a.add(acc, n)
        return acc

    def prefix_incl(self, row: int, i: int) -> Optional[Term]:
        """The inclusive prefix of flat element ``i``: ``add(outer, fold_e)`` (``None`` for an absent
        element)."""
        slot, e = divmod(i, V)
        block, lane = divmod(slot, LANES)
        f = self.slot_fold(row, lane, block)[e]
        return None if f is None else self.alg.add(self.outer(row, lane, block), f)

    def total(self, row: int) -> Term:
        """The total ``S``: the canonical sum tree over the row's weights (``spec.sum_reduction``)."""
        return sum_reduction(self.N, lambda fp, i: self.weight(row, i)).canonical(
            self.alg
        )

    def threshold(self, row: int) -> Term:
        """The threshold ``t = mul(u, S)`` with the uniform as the symbol ``u``."""
        return self.alg.mul(self.alg.sym("u"), self.total(row))


def spec_fingerprint(
    N: int, from_logits: bool = False, in_fmt: str = "f32", temperature: bool = False
) -> str:
    """The draw's specification identifier for vocabulary ``N``: the total, the threshold, and the inclusive prefixes of
    the first element, the last element and the first element of lane 1 (a lane-bit node) of row 0 (``sterm.fingerprint_dag``,
    linear in the term DAG), with the selection rule (``SELECT_RULE``) and the stream's identity (``stream.STREAM_NAME`` and
    its receipt): a change of the rule or of the uniform's derivation moves the class."""
    sp = DrawSpec(N, from_logits, in_fmt, temperature=temperature)
    i_lane1 = LANES * V if N > LANES * V else N - 1
    parts = [
        fingerprint(sp.total(0)),
        fingerprint(sp.threshold(0)),
        fingerprint(sp.prefix_incl(0, 0)),
        fingerprint(sp.prefix_incl(0, N - 1)),
        fingerprint(sp.prefix_incl(0, i_lane1)),
        SELECT_RULE,
        STREAM_NAME,
        stream_fingerprint(),
    ]
    return "sample-icdf-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
