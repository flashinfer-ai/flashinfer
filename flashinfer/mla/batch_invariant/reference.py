# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Exact CPU evaluation of the sparse MLA class.

``ExactFP`` is the backend the specification's functions are called with: correctly rounded binary32 arithmetic on
exact rationals (``fp_interface.exact``), the opaque ``MMA16`` step as one exactly summed, once rounded 17-term
sum (the hypothesis the device tests check against the kernel), and ``EX2`` from a caller-supplied function
(``ex2.approx.ftz.f32`` evaluated on the device, or :func:`ex2_stand_in` where no device is present).

``reference_rows`` evaluates rows with the specification's own ``score_term``, ``tile_stats``, ``pv_fold``, ``merge``
and ``output_elem``; it only caches what the per-column terms share (a tile's scores and statistics).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence

from ...attention.batch_invariant.spec import MMA_K, MMA_OP, State, pv_fold
from ...batch_invariant.fp_interface import exact
from . import spec
from .spec import D_LATENT, TILE_KEYS, SparseRow

__all__ = [
    "ExactFP",
    "RowResult",
    "decode_cache_row",
    "ex2_stand_in",
    "mma16_exact",
    "reference_rows",
]


def decode_cache_row(raw: Sequence, fmt: str) -> List[float]:
    """Cache row -> the bf16 values of the term: bf16 values pass through, e4m3 bytes are decoded exactly."""
    if fmt == "bf16":
        return [float(x) for x in raw]
    return [exact.E4M3_VALUES[int(b) & 0xFF] for b in raw]


def mma16_exact(acc: float, a: Sequence[float], b: Sequence[float]) -> float:
    """One ``MMA16`` K-step: ``RN(acc + sum_i a_i * b_i)`` with the sum formed exactly."""
    vals = [acc, *a, *b]
    if not all(math.isfinite(v) for v in vals):
        out = acc
        for x, y in zip(a, b, strict=True):
            out += x * y
        return exact.to_f32(out) if math.isfinite(out) else out
    num, sh = 0, 0
    for x, y in [(acc, 1.0), *zip(a, b, strict=True)]:
        nx, dx = x.as_integer_ratio()
        ny, dy = y.as_integer_ratio()
        n, s = nx * ny, (dx * dy).bit_length() - 1
        if s > sh:
            num <<= s - sh
            sh = s
        num += n << (sh - s)
    if num == 0:
        # an exact zero sum is +0 under round-to-nearest unless every addend is -0
        neg = all(
            math.copysign(1.0, x) < 0
            for x in [acc] + [p * q for p, q in zip(a, b, strict=True)]
        )
        return -0.0 if neg else 0.0
    return exact.round_to_format(Fraction(num, 1 << sh), exact.F32)


def ex2_stand_in(x: float) -> float:
    """A device-free stand-in for ``ex2.approx.ftz.f32`` (correctly rounded ``2**x``, flush to zero). It is not the
    device instruction: tests that compare with a kernel take the device table instead."""
    if math.isnan(x):
        return x
    if x == -math.inf:
        return 0.0
    if x >= 128:
        return math.inf
    if x < -126:
        return 0.0
    return exact.to_f32(2.0**x)


class ExactFP:
    """The exact backend. ``q_rows[r]`` is the query row ``r``; ``keys[j]`` the decoded cache row of list position
    ``j`` of the row being evaluated (set per row with :meth:`bind`)."""

    def __init__(self, ex2: Callable[[float], float], q_rows=None, keys=None):
        self._ex2, self.q_rows, self.keys = ex2, q_rows, keys

    def bind(self, q_rows, keys) -> "ExactFP":
        self.q_rows, self.keys = q_rows, keys
        return self

    @staticmethod
    def const(v: float) -> float:
        return exact.to_f32(v)

    add = staticmethod(exact.add)
    mul = staticmethod(exact.mul)
    fma = staticmethod(exact.fma)
    div = staticmethod(exact.div)
    fmaxc = staticmethod(exact.fmaxc)

    def ex2(self, x: float) -> float:
        return self._ex2(x)

    @staticmethod
    def cvt_out(x: float, fmt: str) -> float:
        assert fmt == "bf16", fmt
        return exact.to_format(x, exact.BF16)

    def op(self, name: str, *args):
        if name == "a_rows":
            row, t = args
            return tuple(self.q_rows[row][t * MMA_K : (t + 1) * MMA_K])
        if name == "b_cols":
            t, j = args
            return tuple(self.keys[j][t * MMA_K : (t + 1) * MMA_K])
        if name == "v_slice":
            ks, d = args
            return tuple(
                self.keys[j][d]
                if j < len(self.keys) and self.keys[j] is not None
                else 0.0
                for j in range(ks * MMA_K, (ks + 1) * MMA_K)
            )
        if name == spec.CVT_OP:
            # ``decode_cache_row`` already produced the values the bytes denote
            return args[0]
        if name in ("ab_kslice", "pv_kslice"):
            return (args[0], args[1])
        if name == "p_slice":
            return tuple(args)
        if name == MMA_OP:
            acc, (a, b) = args
            return mma16_exact(acc, a, b)
        raise KeyError(name)


@dataclass
class RowResult:
    out: List[float] = field(default_factory=list)  # 512 bf16 values
    tiles: List[tuple] = field(default_factory=list)  # (m, l, O[512]) per tile
    scores: List[List[float]] = field(
        default_factory=list
    )  # per tile, 128 scores (absent: None)


def _row(
    fp: ExactFP,
    row: SparseRow,
    D: int,
    c: float,
    nc: float,
    s2: float,
    fmt: str,
    cols: Sequence[int],
    stats_only: bool,
) -> RowResult:
    res = RowResult()
    states: Dict[int, Optional[State]] = {d: None for d in cols}
    for t in range(spec.n_tiles(row.n)):
        present = [k for k in row.present(t) if fp.keys[t * TILE_KEYS + k] is not None]
        if not present:
            continue
        scores = [
            spec.score_term(fp, row.row, t * TILE_KEYS + k, D, fmt)
            if k in present
            else None
            for k in range(TILE_KEYS)
        ]
        m, probs, l = spec.tile_stats(fp, scores, c, nc)
        res.scores.append(scores)
        o_t = {}
        if not stats_only:
            pb = [fp.cvt_out(p, "bf16") for p in probs]
            for d in cols:

                def v_at(j: int, d: int = d, t: int = t) -> tuple:
                    return fp.op("v_slice", t * (TILE_KEYS // MMA_K) + j, d)

                o_t[d] = pv_fold(fp, fp.const(0.0), pb, v_at)
        res.tiles.append((m, l, o_t))
        for d in cols:
            prev = spec.init_state(fp) if states[d] is None else states[d]
            states[d] = spec.merge(fp, prev, State(m, l, o_t.get(d, 0.0)), c, nc)
    res.out = [spec.output_elem(fp, states[d], s2) for d in cols]
    return res


def reference_rows(
    q_rows: Sequence[Sequence[float]],
    lists: Sequence[Sequence[Sequence]],
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    fmt: str = "bf16",
    ex2_device: Optional[Callable[[List[float]], List[float]]] = None,
    cols: Optional[Sequence[int]] = None,
) -> List[RowResult]:
    """Exact results for rows ``r``: ``q_rows[r]`` (``D_qk`` bf16 values) against ``lists[r]``, the raw cache rows of
    its live list positions in list order (bf16 values or e4m3 bytes; ``None`` for a dead entry inside the live range, which
    is an absent position). ``ex2_device(xs)`` evaluates the device
    instruction on a list of binary32 arguments (two passes: the arguments depend on scores and maxima only);
    without it :func:`ex2_stand_in` is used. ``cols`` restricts the output columns."""
    D = len(q_rows[0])
    assert D in (spec.d_qk(0), spec.d_qk(64)), D
    c = spec.scale_log2(bmm1_scale)
    nc, s2 = -c, spec.scale_out(bmm2_scale)
    cols = list(range(D_LATENT)) if cols is None else list(cols)
    keys = [
        [None if k is None else decode_cache_row(k, fmt) for k in lst] for lst in lists
    ]

    def run(ex2, stats_only):
        fp = ExactFP(ex2)
        return [
            _row(
                fp.bind(q_rows, keys[r]),
                SparseRow(len(keys[r]), r),
                D,
                c,
                nc,
                s2,
                fmt,
                cols,
                stats_only,
            )
            for r in range(len(q_rows))
        ]

    if ex2_device is None:
        return run(ex2_stand_in, False)
    args: Dict[int, float] = {}

    def collect(x):
        args[exact.f32_bits(x)] = x
        return 0.0

    run(collect, True)
    xs = list(args.values())
    table = {
        exact.f32_bits(x): float(y)
        for x, y in zip(xs, ex2_device(xs) if xs else [], strict=True)
    }
    return run(lambda x: table[exact.f32_bits(x)], False)
