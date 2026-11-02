# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Generate ``lean/BIKernels/Cases.lean`` from the Python side.

For the bf16 GEMM class the file carries

* the spec terms of both classes at sampled (K, chunks): the *Python spec* (``gemm.spec.GemmSpec``) S-expression,
  with a Lean theorem that the Lean model (``BIK.GEMM.kfold`` / ``kchunk`` through ``outputElem``, an independent
  implementation) prints the same canonical S-expression by kernel ``decide``;
* the mainloop's blocked lowering (``canon.BlockedFold`` as ``MainloopBlock.lowering()`` emits it and
  ``gemm/blocks/mainloop.py`` executes it): its inner radix and the *Python engine's* trace at sampled block counts,
  reproduced by the Lean loop model ``kblockLoop`` node for node, plus ``kblockLoop_eq_kfold`` instantiated on the
  emitted radix;
* the epilogue chain per output format;
* the dispatcher's chunk rule ``chunks_for(N, K)`` as a table that Lean's ``chunksFor`` re-derives;
* the generator's configuration space at the reference shape and the split class's space at a chunked shape, every
  record checked against ``ConfigOK`` (the Lean mirror of ``Knobs.well_formed``), counted, and ``space_invariant``
  instantiated on them; and every record the dispatcher returns over a shape grid, each shown to be a member.

Consequently a change to the spec, the lowering, a knob constraint or the chunk rule that is not reflected in the
Lean model makes ``lake build`` fail; and a regenerated file that differs from the committed one fails the
device-free conformance test (``tests/batch_invariant/test_lean.py``), which needs no Lean toolchain.

    python -m flashinfer.batch_invariant.lean_cases          # rewrite lean/BIKernels/Cases.lean
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Tuple

from .fp_interface import term
from ..gemm.batch_invariant.cost_model import PARTS
from ..gemm.batch_invariant.generator import (
    REFERENCE_SHAPE,
    Knobs,
    MainloopBlock,
    enumerate_space,
)
from ..gemm.batch_invariant.schedule import chunks_for, default_config
from ..gemm.batch_invariant.spec import MMA_K, GemmSpec, reduction

LEAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lean")
CASES_PATH = os.path.join(LEAN_DIR, "BIKernels", "Cases.lean")

GEMM_CASES = [
    (64, 1),
    (128, 1),
    (256, 2),
    (256, 4),
    (512, 8),
]  # (K, chunks); 32 steps is what kernel `decide` carries
GEMM_LOOP_CASES = [
    1,
    2,
    3,
    5,
]  # block counts at which the mainloop lowering's trace is conformed node for node
GEMM_OUT_CASES = [
    (64, "bf16"),
    (128, "bf16"),
    (64, "f32"),
]  # (K, out_fmt): f32 is the split class's partial store
CHUNK_TABLE_SHAPES = [
    (0, 4096),  # an empty weight: no split (natural-number division in Lean)
    (64, 64),
    (64, 8192),
    (64, 262144),
    (256, 131072),
    (512, 65536),
    (1024, 8192),
    (4096, 4096),
    (8192, 8192),
    (8192, 262144),
    (2048, 2048),
    (128, 4096),
]
C4_SHAPE = (
    16,
    128,
    8192,
    4,
)  # (M, N, K, chunks): a split-class shape whose space has the in-CTA chunk-fold twins
DISPATCH_MS = (1, 8, 64, 512, 4096, 16384)
DISPATCH_NS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
DISPATCH_BIG = (16384, 65536, 262144)
#: The generated theorems closed by ``native_decide`` (the exhaustive checks over the configuration space); every
#: other case is kernel ``decide``. ``lean/Axioms.lean`` pins exactly these names.
DECIDED_THEOREMS = (
    "gemm_space_ok",
    "gemm_space_c4_ok",
    "gemm_space_c4_cta_twins",
    "gemm_dispatch_in_space",
)


def lean_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def spec_cases() -> List[Tuple[int, int, str]]:
    """``(K, chunks, sexpr)``: the Python spec's output element (0, 0) of each class."""
    return [(K, C, term.sexpr(GemmSpec(K, chunks=C).out(0, 0))) for K, C in GEMM_CASES]


def loop_cases() -> Tuple[int, List[Tuple[int, str]]]:
    """The lowering's inner radix and ``(nblk, sexpr)``: the engine's trace of the blocked fold at ``nblk`` blocks."""
    low = MainloopBlock().lowering()
    alg = term.TermAlgebra()
    return low.inner, [
        (
            nblk,
            term.sexpr(low.trace(reduction(nblk * low.inner * MMA_K, 0, 0), alg, nblk)),
        )
        for nblk in GEMM_LOOP_CASES
    ]


def out_cases() -> List[Tuple[int, str, str]]:
    return [
        (K, fmt, term.sexpr(GemmSpec(K, fmt).out(0, 0))) for K, fmt in GEMM_OUT_CASES
    ]


def chunk_table() -> List[Tuple[int, int, int]]:
    return [(N, K, chunks_for(N, K)) for N, K in CHUNK_TABLE_SHAPES]


def spaces() -> Tuple[List[Knobs], List[Knobs]]:
    """The generator's space at the reference shape and the split class's space at ``C4_SHAPE``."""
    M, N, K, C = C4_SHAPE
    return enumerate_space(*REFERENCE_SHAPE), enumerate_space(M, N, K, chunks=C)


def dispatch_cases() -> Dict[Knobs, Tuple[int, int, int]]:
    """Every record the dispatcher returns over the grid (both weight layouts, every part of `PARTS`), with the first shape that
    selects it."""
    disp: Dict[Knobs, Tuple[int, int, int]] = {}
    pairs = [(N, K) for N in DISPATCH_NS for K in DISPATCH_NS]
    pairs += [(N, K) for K in DISPATCH_BIG for N in DISPATCH_NS[::3]]
    pairs += [(N, K) for N in DISPATCH_BIG for K in DISPATCH_NS[::3]]
    for N, K in pairs:
        C = chunks_for(N, K)
        for M in DISPATCH_MS:
            for b_major in ("n", "k"):
                for family, sm in PARTS:
                    k = dataclasses.replace(
                        default_config(M, N, K, C, b_major, sm, family), grid_clusters=0
                    )
                    disp.setdefault(k, (M, N, K))
    return disp


def render() -> str:
    """The content of ``Cases.lean``."""
    L = [
        "import BIKernels.GEMM",
        "import BIKernels.GEMMConfig",
        "import BIKernels.Export",
        "namespace BIK",
        "-- GENERATED by flashinfer/batch_invariant/lean_cases.py from the Python spec / engine; do not edit.",
        "set_option maxRecDepth 4096",
        "open GEMM",
        "",
    ]
    L.append(
        "/-! The spec terms of the two classes (`gemm/spec.py` vs `GEMM.lean`): the Python spec's output element, "
        "reproduced by the Lean model node for node. -/"
    )
    for i, (K, C, sx) in enumerate(spec_cases()):
        steps = K // MMA_K
        t = f"kfold ⟨0⟩ ⟨0⟩ {steps}" if C == 1 else f"kchunk ⟨0⟩ ⟨0⟩ {C} {steps // C}"
        L.append(
            f'theorem gemm{i}_spec : (outputElem "bf16" ({t})).sexpr = {lean_str(sx)} := by decide  -- K={K} C={C}'
        )
    L.append(
        "def gemmCases : List (Nat × Nat) := ["
        + ", ".join(f"({K // MMA_K}, {C})" for K, C in GEMM_CASES)
        + "]"
    )
    L.append("")
    inner, loops = loop_cases()
    L.append(
        f"/-! The mainloop's blocked lowering (`canon.BlockedFold`, `MainloopBlock.lowering()`, executed by "
        f"`gemm/blocks/mainloop.py`): inner radix {inner}. `kblockLoop` on it equals `kfold` for every block count and "
        f"stage count, and the Python trace of the lowering is reproduced node for node at sampled block counts. -/"
    )
    L.append(f"def gemmMainloopInner : Nat := {inner}")
    L.append(
        "theorem gemm_mainloop_eq_kfold : ∀ (S nblk : Nat) (a : RowData) (b : ColData), "
        "kblockLoop a b gemmMainloopInner S .zero nblk = kfold a b (nblk * gemmMainloopInner) := "
        "fun S nblk a b => kblockLoop_eq_kfold a b gemmMainloopInner S nblk"
    )
    for i, (nblk, sx) in enumerate(loops):
        L.append(
            f"theorem gemmloop{i} : (kblockLoop ⟨0⟩ ⟨0⟩ gemmMainloopInner 3 .zero {nblk}).sexpr = {lean_str(sx)} := by decide"
        )
    L.append(
        "def gemmLoopCases : List Nat := ["
        + ", ".join(str(n) for n in GEMM_LOOP_CASES)
        + "]"
    )
    L.append("")
    L.append(
        "/-! The epilogue chain (`spec.output_elem`): `outputKfold` against the Python spec's output term per output format. -/"
    )
    for i, (K, fmt, sx) in enumerate(out_cases()):
        L.append(
            f'theorem gemmout{i} : (outputKfold "{fmt}" ⟨0⟩ ⟨0⟩ {K // MMA_K}).sexpr = {lean_str(sx)} := by decide'
        )
    L.append(
        "def gemmOutCases : List (Nat × String) := ["
        + ", ".join(f'({K // MMA_K}, "{f}")' for K, f in GEMM_OUT_CASES)
        + "]"
    )
    L.append("")
    L.append(
        "/-- `chunks_for(N, K)` as the Python dispatcher computes it; Lean re-derives it. -/"
    )
    L.append(
        "def chunkTable : List (Nat × Nat × Nat) := ["
        + ", ".join(f"({N}, {K}, {C})" for N, K, C in chunk_table())
        + "]"
    )
    L.append(
        "theorem chunk_rule_matches : chunkTable.all (fun ⟨N, K, C⟩ => chunksFor N K == C) = true := by decide"
    )
    L.append("")
    space, spacec = spaces()
    L.append(
        f"/-! The generator's configuration space at the reference shape {REFERENCE_SHAPE} "
        f"(`gemm.generator.enumerate_space`): every knob record, checked against the kernel's static constraints "
        f"(`ConfigOK`), counted, and the `config_invariance` theorem instantiated on it. -/"
    )
    L.append(
        "def gemmSpace : List Config := [" + ", ".join(k.lean() for k in space) + "]"
    )
    L.append(f"theorem gemm_space_size : gemmSpace.length = {len(space)} := by decide")
    L.append(
        "theorem gemm_space_ok : gemmSpace.all ConfigOK = true := by native_decide"
    )
    L.append(
        "theorem gemm_space_invariant : ∀ c₁ ∈ gemmSpace, ∀ c₂ ∈ gemmSpace, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat), "
        "tileResult c₁ S₁ p a b kb nblk = tileResult c₂ S₂ q a b kb nblk := space_invariant gemmSpace"
    )
    M4, N4, K4, C4 = C4_SHAPE
    n_cta = sum(k.cta_chunks for k in spacec)
    twins = [dataclasses.replace(k, cta_chunks=True) for k in space]
    assert [k for k in spacec if k.cta_chunks] == twins, (
        "the in-CTA chunk-fold records are not the twins of the dense space"
    )
    L.append(
        f"/-! The split class's space at ({M4}, {N4}, {K4}) with {C4} chunks: the dense records the shape admits plus the "
        f"in-CTA chunk-fold twin (`ctaChunks`) of every record of the reference space. -/"
    )
    L.append(
        "def gemmSpaceC4 : List Config := [" + ", ".join(k.lean() for k in spacec) + "]"
    )
    L.append(
        "set_option maxRecDepth 16384 in"
    )  # the 1776-record list exceeds the file's default recursion depth
    L.append(
        f"theorem gemm_space_c4_size : gemmSpaceC4.length = {len(spacec)} := by decide"
    )
    L.append(
        "theorem gemm_space_c4_ok : gemmSpaceC4.all ConfigOK = true := by native_decide"
    )
    L.append("set_option maxRecDepth 16384 in")
    L.append(
        f"theorem gemm_space_c4_cta : (gemmSpaceC4.filter (fun c => c.ctaChunks)).length = {n_cta} := by decide"
    )
    L.append(
        "theorem gemm_space_c4_cta_twins : gemmSpaceC4.filter (fun c => c.ctaChunks) = gemmSpace.map (fun c => { c with ctaChunks := true }) := by native_decide"
    )
    L.append(
        "theorem gemm_space_c4_invariant : ∀ c₁ ∈ gemmSpaceC4, ∀ c₂ ∈ gemmSpaceC4, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat), "
        "tileResult c₁ S₁ p a b kb nblk = tileResult c₂ S₂ q a b kb nblk := space_invariant gemmSpaceC4"
    )
    L.append("")
    disp = dispatch_cases()
    L.append(
        "/-! Every knob record the dispatcher returns over the shape grid (both weight layouts; Rubin NVL8 / NVL72, B200 / GB200, B300), "
        "with the first shape that selects it; each is a member of the enumerated space (R3 membership). -/"
    )
    L.append(
        "def gemmDispatch : List (Nat × Nat × Nat × Config) := ["
        + ", ".join(f"({M}, {N}, {K}, {k.lean()})" for k, (M, N, K) in disp.items())
        + "]"
    )
    L.append(
        "theorem gemm_dispatch_in_space : gemmDispatch.all (fun ⟨_, _, _, c⟩ => gemmSpace.any (fun s => s == c) || "
        "(c.ctaChunks && gemmSpaceC4.any (fun s => s == c))) = true := by native_decide"
    )
    L.append("end BIK")
    return "\n".join(L) + "\n"


def summary() -> str:
    space, spacec = spaces()
    return (
        f"lean_cases coverage: gemm specs={len(GEMM_CASES)} loop cases={len(GEMM_LOOP_CASES)} output cases={len(GEMM_OUT_CASES)} "
        f"chunk-table shapes={len(CHUNK_TABLE_SHAPES)} space={len(space)} split space={len(spacec)} dispatch records={len(dispatch_cases())}"
    )


def main(out_path: str = CASES_PATH) -> str:
    content = render()
    with open(out_path, "w") as f:
        f.write(content)
    s = summary()
    print(s)
    return s


if __name__ == "__main__":
    main()
