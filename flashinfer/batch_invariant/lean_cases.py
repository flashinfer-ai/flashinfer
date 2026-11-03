# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Generate ``lean/BIKernels/Cases.lean`` from the Python side.

For the GEMM classes (bf16 at the ``mma16`` step; e4m3 / e5m2 at the ``mma32`` step with the scaled chain) the file
carries

* the spec terms of both classes at sampled (K, chunks): the *Python spec* (``gemm.spec.GemmSpec``) S-expression,
  with a Lean theorem that the Lean model (``BIK.GEMM.kfold`` / ``kchunk`` / ``kfold32`` / ``kchunk32`` through
  ``outputElem``, an independent implementation) prints the same canonical S-expression by kernel ``decide``;
* the mainloop's blocked lowering (``canon.BlockedFold`` as ``MainloopBlock.lowering()`` emits it and
  ``gemm/blocks/mainloop.py`` executes it): its inner radix per format and the *Python engine's* trace at sampled
  block counts, reproduced by the Lean loop model ``kblockLoop`` / ``kblockLoop32`` node for node, plus
  ``kblockLoop_eq_kfold`` / ``kblockLoop32_eq_kfold32`` instantiated on the emitted radix;
* the epilogue chain per output format, unscaled (bf16) and scaled (fp8: ``mul.rn alpha`` then the conversion);
* the dispatcher's chunk rules ``chunks_for(N, K, fmt)`` as tables that Lean's ``chunksFor`` / ``chunksForFp8``
  re-derive;
* the generator's configuration spaces at the reference shape and the split classes' spaces at a chunked shape, every
  record checked against ``ConfigOK`` / ``ConfigOKFp8`` (the Lean mirrors of ``Knobs.well_formed``), counted, and
  ``space_invariant`` / ``space_invariant32`` instantiated on them; and every record the dispatchers return over a
  shape grid, each shown to be a member.

Consequently a change to a spec, a lowering, a knob constraint or a chunk rule that is not reflected in the
Lean model makes ``lake build`` fail; and a regenerated file that differs from the committed one fails the
device-free conformance test (``tests/batch_invariant/test_lean.py``), which needs no Lean toolchain.

    python -m flashinfer.batch_invariant.lean_cases          # rewrite lean/BIKernels/Cases.lean
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Tuple

from .fp_interface import term
from ..gemm.batch_invariant.heuristic import PARTS
from ..gemm.batch_invariant.generator import (
    REFERENCE_SHAPE,
    Knobs,
    MainloopBlock,
    enumerate_space,
)
from ..gemm.batch_invariant.schedule import chunks_for, default_config
from ..gemm.batch_invariant.spec import MMA_K_OF, GemmSpec, reduction

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
#: The fp8 classes (one Lean step, `mma32`, for both formats: e5m2 has the same spaces, rules and chain as e4m3).
FP8_FMT = "e4m3"
GEMM_FP8_CASES = [
    (128, 1),
    (256, 1),
    (512, 2),
    (1024, 4),
]  # (K, chunks) at the 32-wide step: 4 .. 32 steps
GEMM_FP8_LOOP_CASES = [1, 2, 3, 5]
GEMM_FP8_OUT_CASES = [(128, "bf16"), (256, "bf16"), (128, "f32")]
FP8_CHUNK_EXTRA = [
    (64, 4096 + 32 * 64),
    (64, 1040 * 16),
    (8, 1040),
    (64, 8208),
]  # the 32-alignment of the fp8 chunk rule matters here
#: The generated theorems closed by ``native_decide`` (the exhaustive checks over the configuration spaces); every
#: other case is kernel ``decide``. ``lean/Axioms.lean`` pins exactly these names.
DECIDED_THEOREMS = (
    "gemm_space_ok",
    "gemm_space_c4_ok",
    "gemm_space_c4_cta_twins",
    "gemm_dispatch_in_space",
    "gemm_fp8_space_ok",
    "gemm_fp8_space_c4_ok",
    "gemm_fp8_space_c4_cta_twins",
    "gemm_fp8_dispatch_in_space",
)


def lean_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def spec_cases(in_fmt: str = "bf16") -> List[Tuple[int, int, str]]:
    """``(K, chunks, sexpr)``: the Python spec's output element (0, 0) of each class of the format."""
    cases = GEMM_CASES if in_fmt == "bf16" else GEMM_FP8_CASES
    return [
        (K, C, term.sexpr(GemmSpec(K, chunks=C, in_fmt=in_fmt).out(0, 0)))
        for K, C in cases
    ]


def loop_cases(in_fmt: str = "bf16") -> Tuple[int, List[Tuple[int, str]]]:
    """The lowering's inner radix and ``(nblk, sexpr)``: the engine's trace of the blocked fold at ``nblk`` blocks."""
    low = MainloopBlock(in_fmt).lowering()
    alg = term.TermAlgebra()
    cases = GEMM_LOOP_CASES if in_fmt == "bf16" else GEMM_FP8_LOOP_CASES
    return low.inner, [
        (
            nblk,
            term.sexpr(
                low.trace(
                    reduction(nblk * low.inner * MMA_K_OF[in_fmt], 0, 0, in_fmt),
                    alg,
                    nblk,
                )
            ),
        )
        for nblk in cases
    ]


def out_cases(in_fmt: str = "bf16") -> List[Tuple[int, str, str]]:
    cases = GEMM_OUT_CASES if in_fmt == "bf16" else GEMM_FP8_OUT_CASES
    return [
        (K, fmt, term.sexpr(GemmSpec(K, fmt, in_fmt=in_fmt).out(0, 0)))
        for K, fmt in cases
    ]


def chunk_table(in_fmt: str = "bf16") -> List[Tuple[int, int, int]]:
    shapes = CHUNK_TABLE_SHAPES + (FP8_CHUNK_EXTRA if in_fmt != "bf16" else [])
    return [(N, K, chunks_for(N, K, in_fmt)) for N, K in shapes]


def spaces(in_fmt: str = "bf16") -> Tuple[List[Knobs], List[Knobs]]:
    """The generator's space at the reference shape and the split class's space at ``C4_SHAPE``: the bf16 spaces in the
    ``n`` layout (its space is layout-free), the fp8 spaces in the ``k`` layout (the whole knob product: the ``n`` layout
    drops the 2-CTA 16-wide records, `operands.can_implement`)."""
    M, N, K, C = C4_SHAPE
    lay = "n" if in_fmt == "bf16" else "k"
    return (
        enumerate_space(*REFERENCE_SHAPE, b_major=lay, in_fmt=in_fmt),
        enumerate_space(M, N, K, chunks=C, b_major=lay, in_fmt=in_fmt),
    )


def dispatch_cases(in_fmt: str = "bf16") -> Dict[Knobs, Tuple[int, int, int]]:
    """Every record the dispatcher returns over the grid (both weight layouts, every part of `PARTS`) at the parts' nominal
    numbers (``nominal=True``: a device's queried capacity or L2 size never enters the rendering), with the first shape
    that selects it."""
    disp: Dict[Knobs, Tuple[int, int, int]] = {}
    pairs = [(N, K) for N in DISPATCH_NS for K in DISPATCH_NS]
    pairs += [(N, K) for K in DISPATCH_BIG for N in DISPATCH_NS[::3]]
    pairs += [(N, K) for N in DISPATCH_BIG for K in DISPATCH_NS[::3]]
    for N, K in pairs:
        C = chunks_for(N, K, in_fmt)
        for M in DISPATCH_MS:
            for b_major in ("n", "k"):
                for family, sm in PARTS:
                    k = dataclasses.replace(
                        default_config(
                            M, N, K, C, b_major, sm, family, in_fmt, nominal=True
                        ),
                        grid_clusters=0,
                    )
                    disp.setdefault(k, (M, N, K))
    return disp


def _render_format(L: List[str], in_fmt: str) -> None:
    """The cases of one format: ``p`` is the Lean name infix (`` `` for bf16, ``32`` for the fp8 step)."""
    fp8 = in_fmt != "bf16"
    p = "32" if fp8 else ""
    kfold, kchunk, loop = f"kfold{p}", f"kchunk{p}", f"kblockLoop{p}"
    scaled = "true" if fp8 else "false"
    mk = MMA_K_OF[in_fmt]
    tag = "gemm32" if fp8 else "gemm"
    L.append(
        f"/-! The spec terms of the two {in_fmt} classes (`gemm/spec.py` vs `GEMM.lean`, the `mma{mk}` step"
        + (", the scaled chain" if fp8 else "")
        + "): the Python spec's output element, reproduced by the Lean model node for node. -/"
    )
    for i, (K, C, sx) in enumerate(spec_cases(in_fmt)):
        steps = K // mk
        t = (
            f"{kfold} ⟨0⟩ ⟨0⟩ {steps}"
            if C == 1
            else f"{kchunk} ⟨0⟩ ⟨0⟩ {C} {steps // C}"
        )
        L.append(
            f'theorem {tag}{i}_spec : (outputElem {scaled} "bf16" ({t})).sexpr = {lean_str(sx)} := by decide  -- K={K} C={C}'
        )
    cases = GEMM_FP8_CASES if fp8 else GEMM_CASES
    L.append(
        f"def {tag}Cases : List (Nat × Nat) := ["
        + ", ".join(f"({K // mk}, {C})" for K, C in cases)
        + "]"
    )
    L.append("")
    inner, loops = loop_cases(in_fmt)
    inner_name = "gemmMainloopInnerFp8" if fp8 else "gemmMainloopInner"
    L.append(
        f'/-! The {in_fmt} mainloop\'s blocked lowering (`canon.BlockedFold`, `MainloopBlock("{in_fmt}").lowering()`, executed by '
        f"`gemm/blocks/mainloop.py`): inner radix {inner}. `{loop}` on it equals `{kfold}` for every block count and "
        f"stage count, and the Python trace of the lowering is reproduced node for node at sampled block counts. -/"
    )
    L.append(f"def {inner_name} : Nat := {inner}")
    L.append(
        f"theorem {tag}_mainloop_eq_kfold : ∀ (S nblk : Nat) (a : RowData) (b : ColData), "
        f"{loop} a b {inner_name} S .zero nblk = {kfold} a b (nblk * {inner_name}) := "
        f"fun S nblk a b => {loop}_eq_{kfold} a b {inner_name} S nblk"
    )
    for i, (nblk, sx) in enumerate(loops):
        L.append(
            f"theorem {tag}loop{i} : ({loop} ⟨0⟩ ⟨0⟩ {inner_name} 3 .zero {nblk}).sexpr = {lean_str(sx)} := by decide"
        )
    loop_cases_ = GEMM_FP8_LOOP_CASES if fp8 else GEMM_LOOP_CASES
    L.append(
        f"def {tag}LoopCases : List Nat := ["
        + ", ".join(str(n) for n in loop_cases_)
        + "]"
    )
    L.append("")
    L.append(
        f"/-! The {in_fmt} epilogue chain (`spec.output_elem`"
        + (", `mul.rn alpha` then the conversion" if fp8 else "")
        + f"): `outputKfold{p}` against the Python spec's output term per output format. -/"
    )
    for i, (K, fmt, sx) in enumerate(out_cases(in_fmt)):
        head = f'outputKfold32 "{fmt}"' if fp8 else f'outputKfold false "{fmt}"'
        L.append(
            f"theorem {tag}out{i} : ({head} ⟨0⟩ ⟨0⟩ {K // mk}).sexpr = {lean_str(sx)} := by decide"
        )
    out_cases_ = GEMM_FP8_OUT_CASES if fp8 else GEMM_OUT_CASES
    L.append(
        f"def {tag}OutCases : List (Nat × String) := ["
        + ", ".join(f'({K // mk}, "{f}")' for K, f in out_cases_)
        + "]"
    )
    L.append("")
    rule = "chunksForFp8" if fp8 else "chunksFor"
    table = "chunkTableFp8" if fp8 else "chunkTable"
    L.append(
        f'/-- `chunks_for(N, K, "{in_fmt}")` as the Python dispatcher computes it; Lean re-derives it. -/'
    )
    L.append(
        f"def {table} : List (Nat × Nat × Nat) := ["
        + ", ".join(f"({N}, {K}, {C})" for N, K, C in chunk_table(in_fmt))
        + "]"
    )
    L.append(
        f"theorem chunk_rule_matches{'_fp8' if fp8 else ''} : {table}.all (fun ⟨N, K, C⟩ => {rule} N K == C) = true := by decide"
    )
    L.append("")
    space, spacec = spaces(in_fmt)
    sp = "gemmFp8Space" if fp8 else "gemmSpace"
    ok = "ConfigOKFp8" if fp8 else "ConfigOK"
    inv = "space_invariant32" if fp8 else "space_invariant"
    tile = "tileResult32" if fp8 else "tileResult"
    th = "gemm_fp8" if fp8 else "gemm"
    L.append(
        f"/-! The generator's {in_fmt} configuration space at the reference shape {REFERENCE_SHAPE} "
        f"(`gemm.generator.enumerate_space`): every knob record, checked against the kernel's static constraints "
        f"(`{ok}`), counted, and the `config_invariance` theorem instantiated on it. -/"
    )
    L.append(f"def {sp} : List Config := [" + ", ".join(k.lean() for k in space) + "]")
    L.append(f"theorem {th}_space_size : {sp}.length = {len(space)} := by decide")
    L.append(f"theorem {th}_space_ok : {sp}.all {ok} = true := by native_decide")
    L.append(
        f"theorem {th}_space_invariant : ∀ c₁ ∈ {sp}, ∀ c₂ ∈ {sp}, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat), "
        f"{tile} c₁ S₁ p a b kb nblk = {tile} c₂ S₂ q a b kb nblk := {inv} {sp}"
    )
    M4, N4, K4, C4 = C4_SHAPE
    n_cta = sum(k.cta_chunks for k in spacec)
    twins = [dataclasses.replace(k, cta_chunks=True) for k in space]
    assert [k for k in spacec if k.cta_chunks] == twins, (
        "the in-CTA chunk-fold records are not the twins of the dense space"
    )
    spc = f"{sp}C4"
    L.append(
        f"/-! The {in_fmt} split class's space at ({M4}, {N4}, {K4}) with {C4} chunks: the dense records the shape admits plus the "
        f"in-CTA chunk-fold twin (`ctaChunks`) of every record of the reference space. -/"
    )
    L.append(
        f"def {spc} : List Config := [" + ", ".join(k.lean() for k in spacec) + "]"
    )
    L.append(
        "set_option maxRecDepth 16384 in"
    )  # the 1776-record list exceeds the file's default recursion depth
    L.append(f"theorem {th}_space_c4_size : {spc}.length = {len(spacec)} := by decide")
    L.append(f"theorem {th}_space_c4_ok : {spc}.all {ok} = true := by native_decide")
    L.append("set_option maxRecDepth 16384 in")
    L.append(
        f"theorem {th}_space_c4_cta : ({spc}.filter (fun c => c.ctaChunks)).length = {n_cta} := by decide"
    )
    L.append(
        f"theorem {th}_space_c4_cta_twins : {spc}.filter (fun c => c.ctaChunks) = {sp}.map (fun c => {{ c with ctaChunks := true }}) := by native_decide"
    )
    L.append(
        f"theorem {th}_space_c4_invariant : ∀ c₁ ∈ {spc}, ∀ c₂ ∈ {spc}, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat), "
        f"{tile} c₁ S₁ p a b kb nblk = {tile} c₂ S₂ q a b kb nblk := {inv} {spc}"
    )
    L.append("")
    disp = dispatch_cases(in_fmt)
    dn = "gemmFp8Dispatch" if fp8 else "gemmDispatch"
    L.append(
        f"/-! Every knob record the {in_fmt} dispatcher returns over the shape grid (both weight layouts; the nominal parts "
        "Rubin NVL8 / NVL72, B200 / GB200, B300: the table's co-resident capacities and L2 sizes, never a device's queried "
        "numbers), with the first shape that selects it; each is a member of the enumerated space (R3 membership). -/"
    )
    L.append(
        f"def {dn} : List (Nat × Nat × Nat × Config) := ["
        + ", ".join(f"({M}, {N}, {K}, {k.lean()})" for k, (M, N, K) in disp.items())
        + "]"
    )
    L.append(
        f"theorem {th}_dispatch_in_space : {dn}.all (fun ⟨_, _, _, c⟩ => {sp}.any (fun s => s == c) || "
        f"(c.ctaChunks && {spc}.any (fun s => s == c))) = true := by native_decide"
    )
    L.append("")


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
    _render_format(L, "bf16")
    _render_format(L, FP8_FMT)
    L.append("end BIK")
    return "\n".join(L) + "\n"


def summary() -> str:
    space, spacec = spaces()
    space8, spacec8 = spaces(FP8_FMT)
    return (
        f"lean_cases coverage: gemm specs={len(GEMM_CASES)}+{len(GEMM_FP8_CASES)} loop cases={len(GEMM_LOOP_CASES)}+{len(GEMM_FP8_LOOP_CASES)} "
        f"output cases={len(GEMM_OUT_CASES)}+{len(GEMM_FP8_OUT_CASES)} chunk-table shapes={len(CHUNK_TABLE_SHAPES)}+{len(FP8_CHUNK_EXTRA)} "
        f"space={len(space)}/{len(space8)} split space={len(spacec)}/{len(spacec8)} dispatch records={len(dispatch_cases())}/{len(dispatch_cases(FP8_FMT))}"
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
