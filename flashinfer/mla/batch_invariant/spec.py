# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Canonical dataflow of sparse latent (MLA) attention: the classes ``mla.sparse.<fmt>.d512[r64]``.

Stated for finite operands, like the standard-attention class. A row is one (request, query token, head): a query
row ``q`` of ``D_qk`` bf16 values, a compacted ascending list of ``n`` selected cache rows, ``c = fp32(bmm1_scale *
log2 e)``, ``nc = -c`` and ``s2 = fp32(bmm2_scale)``. Everything numeric is written here once over an ``fp`` backend
(``TermAlgebra`` for the symbolic term and the Lean cases, the exact backend of ``reference.py`` for the CPU model);
the kernel's two key reductions are emitted from the same engine plan (``codegen.py``).

* operand: a bf16 value is itself; in the e4m3 members the query and the cache are e4m3 bytes and an operand is
  ``cvt_e4m3_bf16`` of the byte, which is exact;
* the score of list position ``j`` is one ``kfold`` of ``D_qk / 16`` opaque ``MMA16`` steps ascending from ``+0``
  (the 512 latent values first, then the 64 RoPE values of the ``d512r64`` layout);
* tile ``t`` holds list positions ``[128 t, 128 t + 128)``; position ``k = 8 b + 2 c + e`` inside it. A position past
  the live count, or one whose entry is negative, is absent; a tile without a present position is dropped. Its state is
  ``(m, l, O[d])``: ``m`` the maximum and ``l`` the sum of ``p_k = EX2(fma(s_k, c, mul(m, nc)))`` over the present
  positions, both as the engine's pairwise trees over the digits ``e``, ``b``, ``c`` (innermost first), and ``O[d]``
  eight opaque ``MMA16`` key steps from ``+0`` over ``cvt_bf16(p_k)`` and the latent column ``d`` (an absent position
  carries ``p = +0`` and ``v = +0``);
* the row is the fold of its tiles in ascending order from ``(-inf, +0, +0)`` with the standard class's ``merge``;
* the epilogue is ``cvt_bf16(add(mul(mul(O[d], div(1, l)), s2), +0))``; a row with ``n = 0`` is ``+0``.

Tiles are cut by list position: the list is the row's own input, so the term of a token is the same from a prefill
call, a piece of a chunked prefill and a decode step (R4), and nothing in it depends on the batch, the other rows,
the context length or the configuration (R2, R3).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from ...attention.batch_invariant.spec import (
    FLT_MAX,
    MMA_K,
    MMA_OP,
    NEG_INF,
    State,
    init_state,
    merge,
    pv_fold,
    scale_log2,
)
from ...batch_invariant import canon
from ...batch_invariant.canon import Digit, Reduction
from ...batch_invariant.fp_interface.term import Term, TermAlgebra, to_f32

TILE_KEYS = 128  # T: list positions per tile; a tile is its own chunk
D_LATENT = 512  # the PV operand and the output width
ROPE_DIMS = (0, 64)
CACHE_FMTS = ("bf16", "e4m3")
MAX_HEADS = 128
# key position inside a tile: k = 8 b + 2 c + e
E_RADIX, B_RADIX, C_RADIX = 2, 16, 4
KEY_DIGITS = (("e", E_RADIX), ("b", B_RADIX), ("c", C_RADIX))
CVT_OP = "cvt_e4m3_bf16"

__all__ = [
    "FLT_MAX",
    "NEG_INF",
    "State",
    "init_state",
    "merge",
    "score_term",
    "verify_key_plans",
    "TILE_KEYS",
    "D_LATENT",
    "SparseRow",
    "SparseMlaSpec",
    "class_name",
    "d_qk",
    "key_of",
    "n_tiles",
    "output_elem",
    "row_reduction",
    "rowmax_reduction",
    "rowsum_reduction",
    "KEY_SPLITS",
    "key_placement",
    "scale_log2",
    "scale_out",
    "spec_fingerprint",
    "tile_state",
    "tile_stats",
]


def d_qk(rope: int) -> int:
    assert rope in ROPE_DIMS, rope
    return D_LATENT + rope


def class_name(fmt: str, rope: int) -> str:
    assert fmt in CACHE_FMTS and rope in ROPE_DIMS, (fmt, rope)
    return f"mla.sparse.{fmt}.d{D_LATENT}" + (f"r{rope}" if rope else "")


def scale_out(bmm2_scale: float) -> float:
    """``s2 = fp32(bmm2_scale)``, rounded once on the host."""
    return to_f32(float(bmm2_scale))


def key_of(b: int, c: int, e: int) -> int:
    return 8 * b + 2 * c + e


def n_tiles(n: int, T: int = TILE_KEYS) -> int:
    return -(-n // T)


@dataclass(frozen=True)
class SparseRow:
    """One logical row: ``n`` live list positions; ``row`` names the query row in the leaf terms."""

    n: int
    row: int = 0

    def present(self, t: int, T: int = TILE_KEYS) -> List[int]:
        """The present positions of tile ``t``, as offsets inside the tile."""
        return list(range(max(0, min(T, self.n - t * T))))


# ----------------------------------------------------------------------------- the two key reductions


def _key_reduction(name: str, combine, note: str, leaf: Callable[[Any, int], Any]):
    digits = tuple(
        Digit(d, r, "pairwise", combine=combine, identity_note=note)
        for d, r in KEY_DIGITS
    )
    return Reduction(
        name,
        digits,
        lambda fp, idx: leaf(fp, key_of(idx["b"], idx["c"], idx["e"])),
    )


def rowmax_reduction(leaf: Callable[[Any, int], Any]) -> Reduction:
    """``max`` over the positions of one tile; ``leaf(fp, k)`` is the score of position ``k`` or ``None``."""
    return _key_reduction(
        "mla.sparse.rowmax",
        lambda fp, a, b: fp.fmaxc(a, b),
        "fmax(v, -inf) == v for every non-NaN v",
        leaf,
    )


def rowsum_reduction(leaf: Callable[[Any, int], Any]) -> Reduction:
    """``add`` over the positions of one tile; ``leaf(fp, k)`` is ``p_k`` or ``None``."""
    return _key_reduction(
        "mla.sparse.rowsum",
        lambda fp, a, b: fp.add(a, b),
        "add(v, +0) == v for v != -0; every leaf is ex2(.) >= +0 and no partial sum is -0",
        leaf,
    )


KEY_SPLITS = (1, 4)  # warps one head group's tile is spread over


def key_placement(split: int = 1) -> canon.Placement:
    """Where the kernel holds the digits: a thread owns ``e`` and ``16 / split`` values of ``b`` of its two rows, the
    other values of ``b`` are in the ``split`` warps of the head group, ``c`` is the lane."""
    assert split in KEY_SPLITS, split
    return canon.Placement.make(
        {
            "e": [(E_RADIX, "reg")],
            "b": [(B_RADIX // split, "reg"), (split, "warp")],
            "c": [(C_RADIX, "lane")],
        }
    )


# ----------------------------------------------------------------------------- the arithmetic


def _operand(fp, x, fmt: str):
    return x if fmt == "bf16" else fp.op(CVT_OP, x)


def score_reduction(row: int, j: int, D: int, fmt: str = "bf16") -> Reduction:
    """The K-fold of the score of list position ``j`` for query row ``row``: the GEMM class's ``kfold[K = D]``."""
    kstep = Digit(
        "kstep",
        D // MMA_K,
        "fold",
        init=lambda fp: fp.const(0.0),
        step=lambda fp, acc, sl: fp.op(MMA_OP, acc, sl),
        identity_note="D is a whole number of 16-wide steps: no padded step",
    )

    def leaf(fp, idx):
        t = idx["kstep"]
        return fp.op(
            "ab_kslice",
            _operand(fp, fp.op("a_rows", row, t), fmt),
            _operand(fp, fp.op("b_cols", t, j), fmt),
        )

    return Reduction(f"mla.sparse.score[K={D}]", (kstep,), leaf)


def score_term(fp, row: int, j: int, D: int, fmt: str = "bf16"):
    return score_reduction(row, j, D, fmt).canonical(fp)


def tile_stats(fp, scores: Sequence[Any], c, nc):
    """``(m, probs, l)`` of one tile: ``scores[k]`` is the score of position ``k`` or ``None`` when absent (at least
    one present); ``probs[k]`` is ``+0`` for an absent position."""
    assert len(scores) == TILE_KEYS
    m = rowmax_reduction(lambda fp_, k: scores[k]).canonical(fp)
    nms = fp.mul(m, nc)
    probs = [None if s is None else fp.ex2(fp.fma(s, c, nms)) for s in scores]
    l = rowsum_reduction(lambda fp_, k: probs[k]).canonical(fp)
    return m, [fp.const(0.0) if p is None else p for p in probs], l


def tile_state(
    fp, row: SparseRow, t: int, c, nc, D: int, d: int, fmt: str = "bf16"
) -> Optional[State]:
    """The state of tile ``t`` for output column ``d``, ``None`` when the tile has no present position."""
    present = row.present(t)
    if not present:
        return None
    scores = [
        score_term(fp, row.row, t * TILE_KEYS + k, D, fmt) if k in present else None
        for k in range(TILE_KEYS)
    ]
    m, probs, l = tile_stats(fp, scores, c, nc)
    o = pv_fold(
        fp,
        fp.const(0.0),
        [fp.cvt_out(p, "bf16") for p in probs],
        lambda j: _operand(fp, fp.op("v_slice", t * (TILE_KEYS // MMA_K) + j, d), fmt),
    )
    return State(m, l, o)


def row_reduction(
    row: SparseRow,
    D: int,
    d: int,
    c,
    nc,
    fmt: str = "bf16",
    tiles: Optional[int] = None,
) -> Reduction:
    """The row as an engine ``Reduction``: one ``fold`` digit over the tiles with ``merge`` from the init state.
    ``tiles`` widens the digit past the row's own tiles (absent tiles are dropped)."""
    radix = max(1, n_tiles(row.n) if tiles is None else tiles)
    tile = Digit(
        "tile",
        radix,
        "fold",
        init=init_state,
        step=lambda fp, a, b: merge(fp, a, b, c, nc),
        identity_note="absent tiles are dropped (never merged)",
    )
    return Reduction(
        f"mla.sparse.{fmt}[D={D},T={TILE_KEYS}]",
        (tile,),
        lambda fp, idx: tile_state(fp, row, idx["tile"], c, nc, D, d, fmt),
    )


def output_elem(fp, st: Optional[State], s2):
    """``cvt_bf16(add(mul(mul(O, div(1, l)), s2), +0))``; ``+0`` for a row without a tile."""
    if st is None:
        return fp.cvt_out(fp.const(0.0), "bf16")
    inv = fp.div(fp.const(1.0), st.l)
    return fp.cvt_out(fp.add(fp.mul(fp.mul(st.o, inv), s2), fp.const(0.0)), "bf16")


def verify_key_plans(fp) -> None:
    """The kernel's placement of both key reductions builds the canonical tree (``canon.verify``)."""
    for red in (
        rowmax_reduction(lambda fp_, k: fp_.op("s", k)),
        rowsum_reduction(lambda fp_, k: fp_.op("p", k)),
    ):
        for split in KEY_SPLITS:
            canon.verify(canon.plan(red, key_placement(split)), fp)


# ----------------------------------------------------------------------------- symbolic spec + fingerprint


class SparseMlaSpec:
    """Symbolic view: ``output(row, d)`` is the term of one output element."""

    def __init__(
        self, rope: int = 64, fmt: str = "bf16", alg: Optional[TermAlgebra] = None
    ):
        self.rope, self.fmt, self.D = rope, fmt, d_qk(rope)
        self.name = class_name(fmt, rope)
        self.alg = alg or TermAlgebra()
        self.c, self.nc, self.s2 = (
            self.alg.sym("c"),
            self.alg.sym("nc"),
            self.alg.sym("s2"),
        )

    def reduction(
        self, row: SparseRow, d: int, tiles: Optional[int] = None
    ) -> Reduction:
        return row_reduction(row, self.D, d, self.c, self.nc, self.fmt, tiles)

    def state(self, row: SparseRow, d: int) -> Optional[State]:
        return self.reduction(row, d).canonical(self.alg)

    def output(self, row: SparseRow, d: int) -> Term:
        return output_elem(self.alg, self.state(row, d), self.s2)


_FINGERPRINTS: Dict[tuple, str] = {}
# sample row of the receipt: one full tile plus a tile of 67 present positions
FINGERPRINT_ROW = SparseRow(n=TILE_KEYS + 67)


def spec_fingerprint(rope: int = 64, fmt: str = "bf16") -> str:
    """Identifier of the class: the layout constants with the terms of output columns 0 and 1 of ``FINGERPRINT_ROW``
    (a full tile, a masked tile, the merge with the init state and the epilogue are inside)."""
    key = (rope, fmt)
    if key not in _FINGERPRINTS:
        from ...batch_invariant.fp_interface.term import fingerprint

        spec = SparseMlaSpec(rope, fmt)
        parts = [
            spec.name,
            f"Dqk={spec.D}",
            f"Dout={D_LATENT}",
            f"T={TILE_KEYS}",
            "digits=" + ",".join(f"{d}{r}" for d, r in KEY_DIGITS),
            fingerprint(spec.output(FINGERPRINT_ROW, 0)),
            fingerprint(spec.output(FINGERPRINT_ROW, 1)),
        ]
        tag = spec.name.replace(".", "-")
        _FINGERPRINTS[key] = (
            f"{tag}-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        )
    return _FINGERPRINTS[key]
