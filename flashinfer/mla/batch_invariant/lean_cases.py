# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Generate ``lean/BIKernels/SparseMLACases.lean`` from the Python side: the sparse MLA classes' conformance cases.

The file couples the Lean model (``lean/BIKernels/SparseMLA.lean``) to :mod:`.spec` and :mod:`.config` node for node:

* per case the S-expression of one output element as the spec builds it (``row_reduction(...).canonical`` through ``output_elem``) and the Lean row that must print identically (``SparseMLA.outputSexpr``). A case of at most :data:`THEOREM_MAX_BYTES` is a theorem decided by ``decide +kernel``; a larger one (a full tile prints at megabytes: every probability repeats the tile maximum) is compared through the ``sparsemlaconformance`` executable only (``tests/batch_invariant/test_sparse_mla_lean.py``), which also prints every case through the fixup kernel's model;
* per key placement (``spec.key_placement(split)`` for every ``spec.KEY_SPLITS``) the placement record, its validity for the three digits, and the tree one unit of its plan holds for the maximum and the sum of the first ``n`` positions' scores against the spec's canonical reduction (:data:`KEY_NS`);
* every record of the configuration space against ``ConfigOK`` (the mirror of ``well_formed``), the size of the space, and the dispatcher's picks over a grid of calls against the Lean dispatcher and the space.

The tiles keep the class's size and digits (128 positions, ``k = 8 b + 2 c + e``); the cases are kept small by their list lengths and, where the comment of a case says so, by a reduced number of K steps (``D_qk`` 16 or 32 instead of 512 / 576: the score is the same fold with fewer steps).

    python -m flashinfer.mla.batch_invariant.lean_cases     # rewrite lean/BIKernels/SparseMLACases.lean
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Tuple

from ...batch_invariant.fp_interface.term import TermAlgebra, sexpr
from . import config as C
from . import spec as S

LEAN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "batch_invariant",
    "lean",
)
CASES_PATH = os.path.join(LEAN_DIR, "BIKernels", "SparseMLACases.lean")

#: A case whose S-expression is at most this many bytes is a kernel-``decide`` theorem as well as a conformance line.
THEOREM_MAX_BYTES = 20_000

MMA_K = 16

#: ``(name, fmt, D_qk, n, row id, output column, what the case covers)``.
CASES: Tuple[Tuple[str, str, int, int, int, int, str], ...] = (
    ("empty", "bf16", 576, 0, 0, 0, "a row with n = 0"),
    ("one_d512", "bf16", 512, 1, 0, 0, "one position, the d512 layout's 32 K steps"),
    (
        "one_d512r64",
        "bf16",
        576,
        1,
        3,
        7,
        "one position, the d512r64 layout's 36 K steps",
    ),
    ("part3", "bf16", 16, 3, 2, 5, "digits e and c; reduced K (1 step)"),
    ("part9", "bf16", 16, 9, 0, 1, "digits e, c and b; reduced K (1 step)"),
    ("e4m3_one", "e4m3", 576, 1, 1, 0, "the e4m3 member, 36 K steps"),
    ("e4m3_part2", "e4m3", 32, 2, 5, 3, "the e4m3 member; reduced K (2 steps)"),
    ("mid40", "bf16", 16, 40, 0, 0, "40 positions of one tile; reduced K (1 step)"),
    (
        "full_part",
        "bf16",
        16,
        S.TILE_KEYS + 3,
        0,
        0,
        "a full tile and a partial tile; reduced K (1 step)",
    ),
    (
        "e4m3_full_part",
        "e4m3",
        16,
        S.TILE_KEYS + 2,
        4,
        2,
        "the e4m3 member, a full tile and a partial tile; reduced K (1 step)",
    ),
    (
        "three_tiles",
        "bf16",
        16,
        2 * S.TILE_KEYS + 1,
        1,
        0,
        "two full tiles and one position; reduced K (1 step)",
    ),
)

#: Present positions of the key-reduction cases (a ragged tile and a full one).
KEY_NS = (11, S.TILE_KEYS)
#: The unit asked in the key-reduction cases: lane 1 (and warp 2 of a split placement).
KEY_LANE = "[true, false]"
KEY_WARP = "[false, true]"
_KIND = {
    "reg": ".reg",
    "lane": ".lane",
    "warp": ".xchg",
    "cta": ".xchg",
    "grid": ".xchg",
}
_COMB = {"max": ".fmaxc", "sum": ".add"}

#: The dispatcher grid ``(rows, heads, width, sm_count)``.
DISPATCH_ROWS = (1, 2, 8, 64, 256, 1024, 8192)
DISPATCH_HEADS = (1, 16, 17, 32, 64, 128)
DISPATCH_WIDTHS = (64, 128, 2048, 4096)
DISPATCH_SMS = (80, 148, 160)


def lean_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def case_sexpr(fmt: str, D: int, n: int, row: int, d: int) -> str:
    """The spec's term of output column ``d`` of a row with ``n`` list positions, with ``D`` as the score width."""
    alg = TermAlgebra()
    c, nc, s2 = alg.sym("c"), alg.sym("nc"), alg.sym("s2")
    st = S.row_reduction(S.SparseRow(n, row), D, d, c, nc, fmt).canonical(alg)
    return sexpr(S.output_elem(alg, st, s2))


def mla_cases() -> List[Tuple[str, str, str, str, bool, str]]:
    """``(name, fmt, python_sexpr, lean_params, as_theorem, note)``."""
    out = []
    for name, fmt, D, n, row, d, note in CASES:
        assert D % MMA_K == 0, D
        sx = case_sexpr(fmt, D, n, row, d)
        params = f"⟨{D // MMA_K}, {n}, {row}, {d}⟩"
        out.append((name, fmt, sx, params, len(sx) <= THEOREM_MAX_BYTES, note))
    return out


def _bits(radix: int) -> int:
    assert radix & (radix - 1) == 0, radix
    return radix.bit_length() - 1


def lean_placement(split: int) -> Tuple[str, str]:
    """``spec.key_placement(split)`` as the Lean ``List (List Factor)`` and the coordinates of the unit asked."""
    pl = S.key_placement(split)
    digits, coords = [], []
    for name, _ in S.KEY_DIGITS:
        fs = pl.of(name)
        digits.append(
            "[" + ", ".join(f"⟨{_bits(f.radix)}, {_KIND[f.level]}⟩" for f in fs) + "]"
        )
        for f in fs:
            coords.append(
                KEY_LANE
                if f.level == "lane"
                else KEY_WARP
                if f.level == "warp"
                else "[]"
            )
    return "[" + ", ".join(digits) + "]", "[" + ", ".join(coords) + "]"


def _score_leaf(n: int) -> Callable[[Any, int], Any]:
    def leaf(fp: Any, k: int) -> Any:
        return S.score_term(fp, 0, k, MMA_K) if k < n else None

    return leaf


def key_cases() -> List[Tuple[str, int, str, int, str]]:
    """``(name, split, kind, n, python_sexpr)``: the canonical maximum / sum over the scores of positions ``< n``."""
    alg = TermAlgebra()
    out = []
    for split in S.KEY_SPLITS:
        for kind, mk in (("max", S.rowmax_reduction), ("sum", S.rowsum_reduction)):
            for n in KEY_NS:
                red = mk(_score_leaf(n))
                out.append(
                    (f"w{split}_{kind}_n{n}", split, kind, n, sexpr(red.canonical(alg)))
                )
    return out


def dispatch_cases() -> List[Tuple[int, int, int, int, C.SparseMlaConfig]]:
    return [
        (r, h, w, s, C.default_config(r, h, w, s))
        for r in DISPATCH_ROWS
        for h in DISPATCH_HEADS
        for w in DISPATCH_WIDTHS
        for s in DISPATCH_SMS
    ]


def render() -> str:
    space = C.all_valid_configs()
    L = [
        "import BIKernels.SparseMLA",
        "namespace BIK",
        "-- GENERATED by flashinfer/mla/batch_invariant/lean_cases.py from the Python spec and config; do not edit.",
        "set_option maxRecDepth 100000",
        "",
        "/-! The sparse MLA classes (`mla/batch_invariant/spec.py` vs `SparseMLA.lean`): one output element per case, "
        "`⟨K steps, n, row id, column⟩`; the cases without a `_spec` theorem are compared through "
        "`lake exe sparsemlaconformance`. -/",
    ]
    for name, fmt, sx, params, as_thm, note in mla_cases():
        L.append(f"-- {name}: {note}")
        L.append(f"def smlaP_{name} : SparseMLA.Params := {params}")
        if as_thm:
            L.append(
                f"theorem smla_{name}_spec : SparseMLA.outputSexpr .{fmt} smlaP_{name} = {lean_str(sx)} := "
                "by decide +kernel"
            )
        L.append(
            f"theorem smla_{name}_kernel : ∀ (b : SparseMLA.Config) (ws0 : Nat → Attn.State), "
            f"SparseMLA.kernel .{fmt} b (SparseMLA.unitsOf b.tilesPerUnit (SparseMLA.nTiles smlaP_{name}.n)) ws0 "
            f"smlaP_{name} = SparseMLA.output .{fmt} smlaP_{name} := "
            "fun b ws0 => SparseMLA.kernel_units_eq_output _ b ws0 _"
        )
    L.append(
        "def smlaCases : List (String × SparseMLA.Fmt × SparseMLA.Params) := ["
        + ", ".join(f'("{n}", .{fmt}, smlaP_{n})' for n, fmt, *_ in mla_cases())
        + "]"
    )
    L += [
        "",
        "/-! The key reductions (`spec.key_placement` per `KEY_SPLITS`): the placement, its validity, and the tree a "
        "unit of its plan holds against `Reduction.canonical`. -/",
    ]
    for split in S.KEY_SPLITS:
        pl, _ = lean_placement(split)
        L += [
            f"def smlaKeyPlacementW{split} : List (List Factor) := {pl}",
            f"theorem smla_key_placement_w{split} : smlaKeyPlacementW{split} = SparseMLA.keyPlacementOf {split} := rfl",
            f"theorem smla_key_placement_w{split}_ok : PlacementOK SparseMLA.keyDigits smlaKeyPlacementW{split} := "
            f"SparseMLA.keyPlacementOf_ok {split}",
        ]
    for name, split, kind, n, sx in key_cases():
        _, cs = lean_placement(split)
        if len(sx) <= THEOREM_MAX_BYTES:
            L.append(
                f"theorem smla_keys_{name} : SparseMLA.keysSexpr {_COMB[kind]} {n} smlaKeyPlacementW{split} {cs} = "
                f"some {lean_str(sx)} := by decide +kernel"
            )
    L.append(
        "def smlaKeyCases : List (String × Comb × Nat × List (List Factor) × List (List Bool)) := ["
        + ", ".join(
            f'("{name}", {_COMB[kind]}, {n}, smlaKeyPlacementW{split}, {lean_placement(split)[1]})'
            for name, split, kind, n, _ in key_cases()
        )
        + "]"
    )
    L += [
        "",
        "/-! The configuration space (`config.all_valid_configs`) and the dispatcher (`config.default_config`). -/",
        "def smlaSpace : List SparseMLA.Config := ["
        + ", ".join(c.lean() for c in space)
        + "]",
        f"theorem smla_space_size : smlaSpace.length = {len(space)} := by decide +kernel",
        "theorem smla_space_ok : smlaSpace.all SparseMLA.ConfigOK = true := by decide +kernel",
        "theorem smla_space_invariant (f : SparseMLA.Fmt) : ∀ b₁ ∈ smlaSpace, ∀ b₂ ∈ smlaSpace, "
        "∀ (o₁ o₂ : List Nat) (w₁ w₂ : Nat → Attn.State) (P : SparseMLA.Params), "
        "SparseMLA.Covers b₁.tilesPerUnit (SparseMLA.nTiles P.n) o₁ → "
        "SparseMLA.Covers b₂.tilesPerUnit (SparseMLA.nTiles P.n) o₂ → "
        "SparseMLA.kernel f b₁ o₁ w₁ P = SparseMLA.kernel f b₂ o₂ w₂ P := SparseMLA.space_invariant f smlaSpace",
        "theorem smla_dispatch_range_in_space : SparseMLA.dispatchRange.all smlaSpace.contains = true := "
        "by decide +kernel",
        "def smlaDispatch : List (Nat × Nat × Nat × Nat × SparseMLA.Config) := ["
        + ", ".join(
            f"({r}, {h}, {w}, {s}, {cfg.lean()})"
            for r, h, w, s, cfg in dispatch_cases()
        )
        + "]",
        "theorem smla_dispatch_ok : smlaDispatch.all (fun (r, h, w, s, b) => "
        "decide (SparseMLA.defaultConfig r h w s = b) && smlaSpace.contains b && SparseMLA.ConfigOK b) = true := "
        "by decide +kernel",
        "",
        "end BIK",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    text = render()
    with open(CASES_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    cases = mla_cases()
    print(
        f"wrote {CASES_PATH}: {len(cases)} cases "
        f"({sum(1 for c in cases if c[4])} kernel theorems), "
        f"key cases={len(key_cases())} space={len(C.all_valid_configs())} dispatch={len(dispatch_cases())}"
    )


if __name__ == "__main__":
    main()
