# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel shards of ``lse.row`` (``README.md``, Sampling: the tensor-parallel protocol): the row trees are defined on the global
vocabulary, so a rank holding a slot-aligned shard ``[lo, hi)`` (``lo``, ``hi`` multiples of 8) can compute every
maximal complete sub-tree inside its shard, and the ranks' partials merged by the fixed upper part of the tree are the
global term node for node. :func:`shard_partials` lists a shard's partials (a term each, keyed by tree position);
:func:`merge_partials` rebuilds the row term from the union of the ranks' partials. The identity
``merge_partials(red, fp, [shard_partials(red, fp, lo_r, hi_r) for r]) is red.canonical(fp)`` is the TP-invariance
statement; the CPU suite checks it on the production shard sizes. The exchange per row is the number of partials
(``<= 32 * (2 log2 Kp + 1)`` fp32 per shard, a few hundred floats), not a vocabulary-sized buffer.

Tree positions: with digits (elem fold, block pairwise, lane pairwise) a node is ``("lane", level, index)`` (the lane
tree's node at ``level`` covering lanes ``[index << level, (index + 1) << level)`` over all blocks) or ``("block",
lane, level, index)`` (lane ``lane``'s block tree node covering blocks ``[index << level, (index + 1) << level)``).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..canon import Reduction, is_pow2
from .schedule import LANES, V, padded_blocks


def _slot_term(red: Reduction, fp, block: int, lane: int):
    """The elem fold of slot ``(block, lane)`` (``None`` when the slot has no present element)."""
    d = red.digit("elem")
    return d.reduce(
        fp, [red.leaf(fp, {"elem": e, "block": block, "lane": lane}) for e in range(V)]
    )


def _in_shard(block: int, lane: int, lo: int, hi: int) -> bool:
    s = (block * LANES + lane) * V
    return lo <= s < hi


def shard_partials(red: Reduction, fp, lo: int, hi: int, N: int) -> Dict[Tuple, object]:
    """The maximal complete sub-trees of the row tree whose leaves all lie in slots ``[lo, hi)`` (slot aligned), as
    ``{position: term}``. Slots entirely beyond ``N`` are absent and never produce a partial."""
    assert lo % V == 0 and (hi % V == 0 or hi == N) and 0 <= lo <= hi <= N, (
        lo,
        hi,
        N,
    )  # the last shard may end at a partial slot
    Kp = padded_blocks(N)
    assert is_pow2(Kp)
    db, dl = red.digit("block"), red.digit("lane")
    out: Dict[Tuple, object] = {}
    # per lane: the block tree (digit "block" is combined first, for a fixed lane)
    lane_vals = {}
    for lane in range(LANES):
        vals = [
            _slot_term(red, fp, b, lane) if _in_shard(b, lane, lo, hi) else None
            for b in range(Kp)
        ]
        inside = [
            _in_shard(b, lane, lo, hi) and (b * LANES + lane) * V < N for b in range(Kp)
        ]
        level = 0
        # climb the block tree while the sub-tree is complete inside the shard; emit maximal nodes
        nodes = [(b, vals[b], inside[b]) for b in range(Kp)]
        while len(nodes) > 1:
            nxt = []
            for i in range(0, len(nodes), 2):
                (ia, a, ina), (ib, b, inb) = nodes[i], nodes[i + 1]
                if ina and inb:
                    nxt.append(
                        (
                            i // 2,
                            db.combine(fp, a, b)
                            if a is not None and b is not None
                            else (a if b is None else b),
                            True,
                        )
                    )
                else:
                    if ina and a is not None:
                        out[("block", lane, level, ia)] = a
                    if inb and b is not None:
                        out[("block", lane, level, ib)] = b
                    nxt.append((i // 2, None, False))
            nodes = nxt
            level += 1
        ((i0, v0, in0),) = nodes
        lane_vals[lane] = (v0, in0)
        if in0 and v0 is not None:
            out[("block", lane, level, i0)] = (
                v0  # the whole block tree of this lane lies in the shard
            )
    # the lane tree over complete lanes
    nodes = [(l, lane_vals[l][0], lane_vals[l][1]) for l in range(LANES)]
    level = 0
    while len(nodes) > 1:
        nxt = []
        for i in range(0, len(nodes), 2):
            (ia, a, ina), (ib, b, inb) = nodes[i], nodes[i + 1]
            if ina and inb:
                nxt.append(
                    (
                        i // 2,
                        dl.combine(fp, a, b)
                        if a is not None and b is not None
                        else (a if b is None else b),
                        True,
                    )
                )
            else:
                if ina and a is not None:
                    out[("lane", level, ia)] = a
                    out.pop(
                        ("block", ia, Kp.bit_length() - 1, 0), None
                    )  # superseded by the lane-level node
                if inb and b is not None:
                    out[("lane", level, ib)] = b
                    out.pop(("block", ib, Kp.bit_length() - 1, 0), None)
                nxt.append((i // 2, None, False))
        nodes = nxt
        level += 1
    ((i0, v0, in0),) = nodes
    if in0 and v0 is not None:
        for l in range(LANES):
            out.pop(("block", l, Kp.bit_length() - 1, 0), None)
        out[("lane", level, 0)] = v0
    # lane-level nodes supersede the per-lane roots they contain
    for key in list(out):
        if key[0] == "lane":
            _, lv, idx = key
            for l in range(idx << lv, (idx + 1) << lv):
                out.pop(("block", l, Kp.bit_length() - 1, 0), None)
    return out


def merge_partials(red: Reduction, fp, parts: Sequence[Dict[Tuple, object]], N: int):
    """Rebuild the row term from the ranks' partials: every tree node is either given by a rank or combined from its
    children (absent children are dropped, as in the canonical tree)."""
    Kp = padded_blocks(N)
    db, dl = red.digit("block"), red.digit("lane")
    given: Dict[Tuple, object] = {}
    for p in parts:
        for k, v in p.items():
            assert k not in given, f"two ranks claim node {k}"
            given[k] = v
    top = Kp.bit_length() - 1

    def block_node(lane: int, level: int, idx: int):
        key = ("block", lane, level, idx)
        if key in given:
            return given[key]
        if level == 0:
            return None  # an absent slot (beyond N) or a slot no rank reported
        a, b = (
            block_node(lane, level - 1, 2 * idx),
            block_node(lane, level - 1, 2 * idx + 1),
        )
        return b if a is None else a if b is None else db.combine(fp, a, b)

    def lane_node(level: int, idx: int):
        key = ("lane", level, idx)
        if key in given:
            return given[key]
        if level == 0:
            return block_node(idx, top, 0)
        a, b = lane_node(level - 1, 2 * idx), lane_node(level - 1, 2 * idx + 1)
        return b if a is None else a if b is None else dl.combine(fp, a, b)

    return lane_node(5, 0)


def tp_shards(N: int, tp: int) -> List[Tuple[int, int]]:
    """The contiguous equal shards of a vocabulary over ``tp`` ranks (``N`` divisible by ``tp``, shards slot aligned)."""
    assert N % tp == 0 and (N // tp) % V == 0, (N, tp)
    n = N // tp
    return [(r * n, (r + 1) * n) for r in range(tp)]


# =====================================================================================================================
# The device protocol: the wire format of a shard's partials, the programs the
# partials and merge kernels execute, and their term-level oracles.
#
# Protocol decision: TWO ROUNDS. The class's sum tree is defined on ``e_j = ex2((x_j - m) c)`` with the
# GLOBAL maximum ``m``; a one-round online merge of per-shard ``(m_r, s_r)`` pairs (``s = sum_r s_r ex2((m_r - m) c)``) is a
# different term -- a different rounding sequence, hence a different class -- and is never bit-identical to the global
# tree (``test_online_merge_is_not_the_class`` in ``tests/utils/test_batch_invariant_sampling_tp.py``). So: round 1 = the max tree's partials -> all-gather ->
# merge -> ``m``; round 2 = the sum tree's partials over ``ex2((x - m) c)`` -> all-gather -> merge -> ``s``. Both rounds
# use the same layout and the same merge, with the class's ``fmaxc`` / ``add`` digits.
#
# Wire format (``ShardLayout``): for lane ``l`` the in-shard blocks of the row tree are the contiguous range
# ``[b_lo(l), b_hi(l))`` (slot ``32 b + l`` in ``[lo / 8, hi / 8)``); its partials are the maximal complete sub-trees of
# the lane's block tree inside that range = the dyadic decomposition of the range, listed in block order (the pieces
# climbing from ``b_lo`` by ascending level -- odd indices, "side 0" -- then the pieces descending to ``b_hi`` -- even
# indices, "side 1"); at most one piece per (lane, level, side). The lanes' lists are concatenated (``base[l]``), so a
# shard's partials are ``P = sum_l n_pieces(l)`` fp32 values (e.g. 104..264 for the 152064 / 8 shards), a function of
# ``(lo, hi, N)`` only -- never of the kernel configuration. The lane tree is never climbed by a shard (a contiguous shard
# holds a whole lane only when it holds the whole row), so the TP = 1 "shard" exchanges the 32 lane roots.
# =====================================================================================================================


def lane_block_range(lo: int, hi: int, lane: int) -> Tuple[int, int]:
    """``[b_lo, b_hi)``: the blocks whose slot of ``lane`` lies in the slot-aligned shard ``[lo, hi)``."""
    s_lo, s_hi = (
        lo // V,
        -(-hi // V),
    )  # a partial last slot (hi == N) belongs to the last shard
    b_lo = max(0, -(-(s_lo - lane) // LANES))
    b_hi = max(0, -(-(s_hi - lane) // LANES))
    return b_lo, max(b_lo, b_hi)


def dyadic_pieces(a: int, b: int) -> List[Tuple[int, int]]:
    """The maximal aligned sub-trees (``(level, index)``) tiling the block range ``[a, b)``, in block order: the segment-tree
    decomposition. Pieces climbing from ``a`` have odd indices, pieces descending to ``b`` even ones."""
    left, right = [], []
    L = 0
    while a < b:
        if a & 1:
            left.append((L, a))
            a += 1
        if b & 1:
            b -= 1
            right.append((L, b))
        a >>= 1
        b >>= 1
        L += 1
    return left + right[::-1]


def piece_side(idx: int) -> int:
    """The slot side of a piece from its index: 0 for the ascending (odd) pieces, 1 for the descending (even) ones."""
    return 1 - (idx & 1)


class ShardLayout:
    """The wire format of one shard's partials (see the module notes above). ``pieces[l]`` lists lane ``l``'s pieces in
    block order, ``base[l]`` is the index of its first piece, ``P = base[32]`` the shard's partial count."""

    def __init__(self, lo: int, hi: int, N: int):
        if not (lo % V == 0 and (hi % V == 0 or hi == N) and 0 <= lo <= hi <= N):
            raise ValueError(
                f"shard [{lo}, {hi}) of {N} is not slot aligned (multiples of {V}, 0 <= lo <= hi <= N)"
            )
        self.lo, self.hi, self.N = lo, hi, N
        self.Kp = padded_blocks(N)
        self.kb = self.Kp.bit_length() - 1
        self.ranges = tuple(lane_block_range(lo, hi, l) for l in range(LANES))
        self.pieces = tuple(tuple(dyadic_pieces(a, b)) for a, b in self.ranges)
        base = [0]
        for l in range(LANES):
            base.append(base[-1] + len(self.pieces[l]))
        self.base = tuple(base)
        self.P = base[LANES]
        self._index = {}
        for l in range(LANES):
            for k, (L, i) in enumerate(self.pieces[l]):
                self._index[(l, L, i)] = self.base[l] + k

    def __eq__(self, other):
        return isinstance(other, ShardLayout) and (self.lo, self.hi, self.N) == (
            other.lo,
            other.hi,
            other.N,
        )

    def __hash__(self):
        return hash((self.lo, self.hi, self.N))

    @property
    def block_span(self) -> Tuple[int, int]:
        """``[B_lo, B_hi)``: every block with an in-shard slot (empty when the shard is empty)."""
        if self.lo == self.hi:
            return 0, 0
        return (self.lo // V) // LANES, ((self.hi + V - 1) // V + LANES - 1) // LANES

    def index(self, lane: int, level: int, idx: int) -> int:
        return self._index[(lane, level, idx)]

    def positions(self) -> Dict[Tuple, int]:
        """``{("block", lane, level, idx): slot}``: the pieces as :func:`shard_partials` keys them."""
        return {("block", l, L, i): s for (l, L, i), s in self._index.items()}

    def slot_table(self) -> List[List[List[int]]]:
        """``table[lane][level][side]``: the slot of the lane's piece at that level and side, ``-1`` when there is none (the
        device gathers slots from this table; both kernels are specialised on the layout)."""
        tab = [[[-1, -1] for _ in range(self.kb + 1)] for _ in range(LANES)]
        for l in range(LANES):
            for L, i in self.pieces[l]:
                assert tab[l][L][piece_side(i)] == -1
                tab[l][L][piece_side(i)] = self.index(l, L, i)
        return tab

    def in_shard(self, lane: int, level: int, idx: int) -> bool:
        a, b = self.ranges[lane]
        return idx << level >= a and (idx + 1) << level <= b

    def is_piece(self, lane: int, level: int, idx: int) -> bool:
        return self.in_shard(lane, level, idx) and (
            level == self.kb or not self.in_shard(lane, level + 1, idx >> 1)
        )

    def describe(self) -> str:
        return f"lo{self.lo}_hi{self.hi}_N{self.N}"


def shard_block_partials(
    red: Reduction, fp, lo: int, hi: int, N: int
) -> Dict[Tuple, object]:
    """The protocol's partials of a shard as terms: :func:`shard_partials` without the lane-level nodes (the 32 lane
    roots stay separate when a shard holds the whole row)."""
    lay = ShardLayout(lo, hi, N)
    out: Dict[Tuple, object] = {}
    db = red.digit("block")
    for l in range(LANES):
        for L, i in lay.pieces[l]:
            vals = [_slot_term(red, fp, b, l) for b in range(i << L, (i + 1) << L)]
            out[("block", l, L, i)] = db.reduce(fp, vals)
    return out


def partials_width(lo: int, hi: int, N: int) -> int:
    return ShardLayout(lo, hi, N).P


def pairwise_nodes(digit, fp, vals: Sequence) -> List[List]:
    """Every node of the adjacent-pairs tree over ``vals`` (a power-of-two count, aligned to its size in the global tree):
    ``levels[L][i]`` is the sub-tree over ``vals[i << L : (i + 1) << L]`` with absent (``None``) leaves dropped as
    :func:`flashinfer.batch_invariant.canon._pairwise` drops them; ``levels[-1][0]`` is ``digit.reduce(fp, vals)``."""
    assert is_pow2(len(vals)), len(vals)
    levels = [list(vals)]
    while len(levels[-1]) > 1:
        prev = levels[-1]
        nxt = []
        for i in range(0, len(prev), 2):
            a, b = prev[i], prev[i + 1]
            nxt.append(b if a is None else a if b is None else digit.combine(fp, a, b))
        levels.append(nxt)
    return levels


def aligned_group_nodes(
    digit, fp, vals: Sequence, residue: int
) -> Dict[Tuple[int, int], object]:
    """The nodes over the globally aligned groups inside a list of ``len(vals)`` consecutive units whose first unit has
    global index ``residue`` modulo ``len(vals)``: ``{(level, start): value}`` for every group of ``2^level`` units starting
    at ``start`` with ``(residue + start) % 2^level == 0`` that lies inside the list (level 0 = the units themselves)."""
    n = len(vals)
    nodes = {(0, s): vals[s] for s in range(n)}
    L = 1
    while n >= (1 << L):
        half = 1 << (L - 1)
        for s in range(n):
            if (
                (residue + s) % (1 << L) == 0
                and s + (1 << L) <= n
                and (L - 1, s) in nodes
                and (L - 1, s + half) in nodes
            ):
                a, b = nodes[(L - 1, s)], nodes[(L - 1, s + half)]
                nodes[(L, s)] = (
                    b if a is None else a if b is None else digit.combine(fp, a, b)
                )
        L += 1
    return nodes


# ------------------------------------------------------------------------------------------------------ the partials kernel
class TPStructure:
    """The static structure of the partials kernel for a shard and a configuration: ``kpw`` blocks per thread (a power of
    two), ``W`` warps per row per CTA, ``C`` CTAs per row; the cover ``[A_lo, A_lo + kpw W C)`` of the shard's block span
    with ``A_lo`` aligned to the CTA unit ``kpw W`` (so every thread's chunk and every CTA's range is an aligned node of
    the global block tree; CTA groups are aligned by their residue). Thread (cta ``c``, warp ``w``) owns blocks
    ``A_lo + (c W + w) kpw + k``."""

    def __init__(self, lay: ShardLayout, kpw: int, W: int, C: int):
        assert is_pow2(kpw) and is_pow2(W) and is_pow2(C), (kpw, W, C)
        self.lay, self.kpw, self.W, self.C = lay, kpw, W, C
        self.g = kpw.bit_length() - 1
        unit = kpw * W
        B_lo, B_hi = lay.block_span
        self.A_lo = (B_lo // unit) * unit
        self.Kv = kpw * W * C
        if self.A_lo + self.Kv < B_hi:
            raise ValueError(
                f"cover of {self.Kv} blocks from {self.A_lo} does not reach the shard's block span [{B_lo}, {B_hi})"
            )
        self.cta_residue = (
            (self.A_lo // unit) % C if C > 1 else 0
        )  # global CTA-unit index of CTA 0 modulo C

    @staticmethod
    def resolve(lay: ShardLayout, W: int, C: int) -> "TPStructure":
        """The smallest ``kpw`` whose cover reaches the shard's span."""
        kpw = 1
        while True:
            try:
                return TPStructure(lay, kpw, W, C)
            except ValueError:
                kpw *= 2
                if kpw > max(lay.Kp, 1) * 2:
                    raise

    def chunk_index(self, c, w):
        """Global chunk (``kpw``-block unit) index of warp ``w`` of CTA ``c`` (ints or runtime values)."""
        return self.A_lo // self.kpw + c * self.W + w

    def describe(self) -> str:
        return f"kpw{self.kpw}_W{self.W}_C{self.C}_A{self.A_lo}"


def candidate_nodes_chunk(st: TPStructure, digit, fp, vals: Sequence, chunk):
    """Phase 1 of the partials program for one thread: the nodes of its chunk ``(level, global index, value)`` from the
    ``kpw`` slot folds ``vals`` (level 0 .. g), and the chunk root for the exchange. ``chunk`` is the thread's global
    chunk index (int or runtime)."""
    levels = pairwise_nodes(digit, fp, vals)
    out = []
    for L, nodes in enumerate(levels):
        n = st.kpw >> L
        base = chunk * n if n != 1 else chunk
        for i, v in enumerate(nodes):
            out.append((L, base + i if i else base, v))
    return out, levels[-1][0]


def candidate_nodes_cta(st: TPStructure, digit, fp, sibs: Sequence, c):
    """Phase 2: the nodes over aligned groups of the CTA's ``W`` chunk roots (levels g + 1 .. g + log2 W; CTA ``c``'s
    chunks are globally aligned to ``W``), and the CTA root."""
    levels = pairwise_nodes(digit, fp, sibs)
    out = []
    cta_unit = (
        st.A_lo // (st.kpw * st.W) + c
    )  # global index of this CTA's range in CTA units
    for L in range(1, len(levels)):
        n = st.W >> L  # groups of this level in the CTA
        base = cta_unit * n if n != 1 else cta_unit
        for i, v in enumerate(levels[L]):
            out.append((st.g + L, base + i if i else base, v))
    return out, levels[-1][0]


def candidate_nodes_cluster(st: TPStructure, digit, fp, sibs: Sequence):
    """Phase 3: the nodes over the globally aligned groups of the cluster's ``C`` CTA roots (static indices)."""
    nodes = aligned_group_nodes(digit, fp, sibs, st.cta_residue)
    lgW = st.W.bit_length() - 1
    out = []
    unit0 = st.A_lo // (st.kpw * st.W)
    for (L, s), v in nodes.items():
        if L == 0:
            continue
        out.append((st.g + lgW + L, (unit0 + s) >> L, v))
    return out


def trace_partials(
    red: Reduction, fp, lay: ShardLayout, st: TPStructure, leaf_of_slot
) -> Dict[int, object]:
    """Lock-step simulation of the partials kernel's program over every unit ``(cta, warp, lane)``: ``{slot: term}`` of
    every piece stored (each exactly once). ``leaf_of_slot(fp, block, lane)`` is the slot fold (``None`` outside the shard,
    as the kernel's predicated loads leave such slots at the identity, which never enters a piece)."""
    db = red.digit("block")
    units = [(c, w, l) for c in range(st.C) for w in range(st.W) for l in range(LANES)]
    stored: Dict[int, object] = {}

    def emit(l, cands):
        for L, gj, v in cands:
            if lay.kb >= L and lay.is_piece(l, L, gj):
                slot = lay.index(l, L, gj)
                assert slot not in stored, ("piece stored twice", l, L, gj)
                assert v is not None
                stored[slot] = v

    roots = {}
    for c, w, l in units:
        chunk = st.chunk_index(c, w)
        vals = [leaf_of_slot(fp, chunk * st.kpw + k, l) for k in range(st.kpw)]
        cands, root = candidate_nodes_chunk(st, db, fp, vals, chunk)
        emit(l, cands)
        roots[(c, w, l)] = root
    cta_roots = {}
    for c, w, l in units:
        sibs = [roots[(c, w2, l)] for w2 in range(st.W)]
        cands, cta_root = candidate_nodes_cta(st, db, fp, sibs, c)
        if w == 0:
            emit(l, cands)
        cta_roots[(c, l)] = cta_root
    if st.C > 1:
        for l in range(LANES):
            sibs = [cta_roots[(c2, l)] for c2 in range(st.C)]
            emit(l, candidate_nodes_cluster(st, db, fp, sibs))
    return stored


def verify_partials(red: Reduction, fp, lay: ShardLayout, st: TPStructure) -> None:
    """The partials program stores exactly the layout's pieces, each the canonical sub-tree term (identity)."""
    want = shard_block_partials(red, fp, lay.lo, lay.hi, lay.N)
    want_by_slot = {lay.index(l, L, i): t for (_, l, L, i), t in want.items()}

    def leaf_of_slot(fp_, b, l):
        return _slot_term(red, fp_, b, l) if _in_shard(b, l, lay.lo, lay.hi) else None

    got = trace_partials(red, fp, lay, st, leaf_of_slot)
    assert set(got) == set(want_by_slot), (
        sorted(set(got) ^ set(want_by_slot))[:8],
        lay.describe(),
        st.describe(),
    )
    bad = [s for s in got if got[s] is not want_by_slot[s]]
    assert not bad, (
        f"{len(bad)} pieces differ from the canonical sub-trees",
        lay.describe(),
        st.describe(),
    )


# -------------------------------------------------------------------------------------------------------- the merge kernel
def lane_classes(layouts: Sequence[ShardLayout]) -> Tuple[Tuple[int, ...], List[Tuple]]:
    """Lanes whose piece structure is identical in every rank form a class: ``(class_of_lane[32], representatives)``."""
    keys = [tuple(lay.pieces[l] for lay in layouts) for l in range(LANES)]
    reps: List[Tuple] = []
    cls = []
    for k in keys:
        if k not in reps:
            reps.append(k)
        cls.append(reps.index(k))
    return tuple(cls), reps


def merge_ops(layouts: Sequence[ShardLayout], lane: int) -> Tuple[List[Tuple], int]:
    """The straight-line program rebuilding lane ``lane``'s block root from the ranks' pieces: ops ``("load", rank, k)`` (the
    lane's ``k``-th piece of that rank, at slot ``base_r[lane] + k``: every lane of a class runs the same program with its own bases)
    and ``("comb", a, b)`` (indices into the op list), and the index of the result (``-1``: the lane has no present
    block, i.e. the root is absent). The recursion is :func:`merge_partials`'s ``block_node``: a node is a rank's piece
    or the combine of its children, absent children dropped."""
    Kp = layouts[0].Kp
    kb = Kp.bit_length() - 1
    given = {}
    for r, lay in enumerate(layouts):
        for L, i in lay.pieces[lane]:
            given[(L, i)] = (
                r,
                lay.index(lane, L, i) - lay.base[lane],
            )  # the position within the lane's piece list
    ops: List[Tuple] = []

    def node(L, i):
        if (L, i) in given:
            ops.append(("load",) + given[(L, i)])
            return len(ops) - 1
        if L == 0:
            return None
        a, b = node(L - 1, 2 * i), node(L - 1, 2 * i + 1)
        if a is None:
            return b
        if b is None:
            return a
        ops.append(("comb", a, b))
        return len(ops) - 1

    res = node(kb, 0)
    return ops, (-1 if res is None else res)


def run_merge_ops(ops: Sequence[Tuple], res: int, digit, fp, load):
    """Execute :func:`merge_ops` with ``load(rank, k)`` -> value and the digit's combine (``None`` for an absent root)."""
    vals = []
    for op in ops:
        if op[0] == "load":
            vals.append(load(op[1], op[2]))
        else:
            vals.append(digit.combine(fp, vals[op[1]], vals[op[2]]))
    return None if res < 0 else vals[res]


def check_bounds(bounds: Sequence[Tuple[int, int]], N: int) -> None:
    """The ranks' shards tile ``[0, N)`` contiguously, slot aligned, in rank order (a shard may be empty)."""
    if not bounds:
        raise ValueError("no shards")
    pos = 0
    for r, (lo, hi) in enumerate(bounds):
        if lo != pos or hi < lo or lo % V or (hi % V and hi != N):
            raise ValueError(
                f"shard {r} = [{lo}, {hi}) does not continue the tiling of [0, {N}) at {pos} on slot boundaries"
            )
        pos = hi
    if pos != N:
        raise ValueError(f"the shards end at {pos}, not at N = {N}")


def merge_programs(bounds: Sequence[Tuple[int, int]], N: int):
    """Everything the merge kernel is specialised on: the layouts, the lane classes and one program per class."""
    check_bounds(bounds, N)
    layouts = [ShardLayout(lo, hi, N) for lo, hi in bounds]
    cls, reps = lane_classes(layouts)
    progs = [merge_ops(layouts, cls.index(k)) for k in range(len(reps))]
    return layouts, cls, progs


def verify_merge(red: Reduction, fp, bounds: Sequence[Tuple[int, int]], N: int) -> None:
    """The merge program over the shards' pieces, followed by the lane tree, IS the canonical row term (identity)."""
    layouts, cls, progs = merge_programs(bounds, N)
    parts = [shard_block_partials(red, fp, lo, hi, N) for lo, hi in bounds]
    tabs = [
        {lay.index(l, L, i): t for (_, l, L, i), t in p.items()}
        for lay, p in zip(layouts, parts, strict=True)
    ]
    db, dl = red.digit("block"), red.digit("lane")
    roots = []
    for l in range(LANES):
        ops, res = progs[cls[l]]
        roots.append(
            run_merge_ops(
                ops, res, db, fp, lambda r, k, l=l: tabs[r][layouts[r].base[l] + k]
            )
        )
    got = dl.reduce(fp, roots)
    ref = red.canonical(fp)
    assert got is ref, ("merge program != canonical term", bounds, N)
    assert merge_partials(red, fp, parts, N) is ref


def tp_shards_padded(
    N: int, tp: int, divisible_by: int = 128
) -> Tuple[int, List[Tuple[int, int]]]:
    """Megatron's vocabulary padding: ``N`` rounded up to a multiple of ``divisible_by * tp`` and split equally. Returns
    the padded size and its shards (the class is then ``lse.row`` at the padded size)."""
    unit = divisible_by * tp
    Np = -(-N // unit) * unit
    return Np, tp_shards(Np, tp)
