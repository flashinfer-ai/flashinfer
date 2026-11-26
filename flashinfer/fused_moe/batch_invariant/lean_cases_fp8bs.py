# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Generate ``lean/BIKernels/MoEFp8BsCases.lean`` from the Python side: the block-scale fp8 MoE classes' conformance cases.

The committed file is what the Python spec emits (``tests/batch_invariant/test_moe_fp8bs_lean.py`` checks it byte for byte without
a toolchain, and with ``lake`` builds it and compares the Lean model's terms through ``lake exe fp8bsconformance``):

* the ``fp8bs.quant`` class (``spec_fp8bs.py`` vs ``FP8BlockQuant.lean``): every placement of the block digit
  (``block_amax_placement(V)``) with the engine's block-amax term at unit (0, 0, 0) and the spec's, the placements' validity
  and their invariance theorems instantiated, the scale and code terms of two blocks (the block index enters the leaves only),
  the half-tree split the fused requantization relies on;
* the MoE class (``spec_fp8bs.Fp8BsMoESpec`` vs ``MoEFp8Bs.lean``): GEMM1's gated and ungated chains are the bf16 classes',
  GEMM2 the plain conversion;
* the grouped block-scale space (``schedule_fp8bs.grouped_knobs_fp8bs``): every record checked against the kernel's static
  constraints (``MoEFp8Bs.groupedBlockScaleFp8OK``), a member of the dense block-scale space, the ``space_invariantKB`` theorem
  instantiated on it; the dispatcher's picks over the bench geometries at every nominal part, both requantization spellings,
  members of the space;
* the 32-wide classes ``moe.e4m3bs32.<act>`` (``deepseek_fp8_block_size = 32``): the GEMM core ``gemm.e4m3.kblock32``'s output
  terms at (0, 0) and (3, 33) and the per-step promotion loop against the engine's one-step traces (``MoEFp8Bs.kblock32`` /
  ``promoteLoop32``), the radix-32 quantizer instance (``FP8BlockQuant.blockAmax32``: the placements, the chains), the dispatcher's
  picks at the class members of the same spaces.

    python -m flashinfer.fused_moe.batch_invariant.lean_cases_fp8bs     # rewrite lean/BIKernels/MoEFp8BsCases.lean
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from ...batch_invariant import canon
from ...batch_invariant.quantization import qterm
from ...batch_invariant.quantization.lean_cases import _with_ref, lean_factor, lean_str
from . import mterm
from . import spec_fp8bs as B
from .lean_cases import LEAN_DIR

CASES_PATH = os.path.join(LEAN_DIR, "BIKernels", "MoEFp8BsCases.lean")
FP8BSQ_K = 256  # two blocks: the chains of block 0 and block 1, codes 0 and 255
FP8BSQ_OUT_WHICH = ("s0", "s1", "code0", "codeL")
#: The geometries the dispatcher's membership is stated over: (H, I, E, k), the bench grid and the real-model targets.
GEOMS = (
    (2048, 768, 128, 8),
    (4096, 14336, 8, 2),
    (7168, 2048, 256, 8),
    (5120, 8192, 16, 1),
    (512, 256, 64, 6),
    (1024, 512, 64, 8),
    (2560, 640, 512, 10),
)
TS = (1, 7, 64, 300, 800, 4096, 16384)
ACTS = (("silu", True), ("gelu_tanh", True), ("relu2", False))
FP8BSQ32_K = 64  # two blocks of 32: the chains of block 0 and block 1, codes 0 and 63
#: The 32-wide core's output cases ``(K, out_fmt)`` and the block counts of its per-step loop cases.
KBLOCK32_OUT_CASES = ((32, "bf16"), (128, "bf16"), (256, "f32"))
KBLOCK32_LOOP_CASES = (1, 4, 5)


def fp8bsq_cases() -> List[Dict]:
    """Per placement of the block digit (a whole block per thread down to four elements per lane; the standalone kernel's
    ``V = 8``, the fused epilogue's half blocks ``V = 64``): the Lean placement, the engine's block-amax term at unit (0, 0, 0)
    and the spec's."""
    out = []
    sp = B.BSQuantSpec(B.BLOCK)
    red = sp.block_reduction(0, 0)
    spec_sx = qterm.sexpr(sp.block_amax(0, 0))
    for V in B.PLACEMENT_VS:
        pl = B.block_amax_placement(V)
        ps = (
            "[["
            + ", ".join(lean_factor(f.radix, f.level) for f in pl.of("elem"))
            + "]]"
        )
        got = canon.trace(canon.plan(red, pl), sp.alg)[(0, 0, 0)]
        out.append(dict(name=f"V{V}", ps=ps, kernel=qterm.sexpr(got), spec=spec_sx))
    return out


def fp8bsq_out_cases() -> Dict[str, str]:
    """The scale terms of blocks 0 and 1 and the codes 0 and ``K - 1`` at ``K = FP8BSQ_K``, plus the two block amaxes the terms
    embed."""
    sp = B.BSQuantSpec(FP8BSQ_K)
    o = sp.outputs(0)
    return dict(
        amax0=qterm.sexpr(sp.block_amax(0, 0)),
        amax1=qterm.sexpr(sp.block_amax(0, 1)),
        s0=qterm.sexpr(o["s"][0]),
        s1=qterm.sexpr(o["s"][1]),
        code0=qterm.sexpr(o["codes"][0]),
        codeL=qterm.sexpr(o["codes"][FP8BSQ_K - 1]),
    )


def fp8bsq32_cases() -> List[Dict]:
    """The 32-wide instance's placements (a whole block per thread down to four elements per lane; the standalone kernel's ``V =
    8``): the Lean placement, the engine's block-amax term at unit (0, 0, 0) and the spec's."""
    out = []
    sp = B.BSQuantSpec(32, block=32)
    red = sp.block_reduction(0, 0)
    spec_sx = qterm.sexpr(sp.block_amax(0, 0))
    for V in (32, 16, 8, 4):
        pl = B.block_amax_placement(V, 32)
        ps = (
            "[["
            + ", ".join(lean_factor(f.radix, f.level) for f in pl.of("elem"))
            + "]]"
        )
        got = canon.trace(canon.plan(red, pl), sp.alg)[(0, 0, 0)]
        out.append(dict(name=f"V{V}", ps=ps, kernel=qterm.sexpr(got), spec=spec_sx))
    return out


def fp8bsq32_out_cases() -> Dict[str, str]:
    """The 32-wide instance's scale terms of blocks 0 and 1 and the codes 0 and 63 at K = 64, plus the two block amaxes."""
    sp = B.BSQuantSpec(FP8BSQ32_K, block=32)
    o = sp.outputs(0)
    return dict(
        amax0=qterm.sexpr(sp.block_amax(0, 0)),
        amax1=qterm.sexpr(sp.block_amax(0, 1)),
        s0=qterm.sexpr(o["s"][0]),
        s1=qterm.sexpr(o["s"][1]),
        code0=qterm.sexpr(o["codes"][0]),
        codeL=qterm.sexpr(o["codes"][FP8BSQ32_K - 1]),
    )


def kblock32_cases() -> List[Tuple[int, str, str, str]]:
    """``(KB, out_fmt, sexpr at (0, 0), sexpr at (3, 33))`` of the 32-wide core's output terms (``spec_fp8bs.reduction_kblock`` at
    32 through the dense output chain); (3, 33) reads the second 32-column weight scale block."""
    from ...batch_invariant.fp_interface.term import TermAlgebra as GTermAlgebra
    from ...batch_invariant.fp_interface.term import sexpr as gsexpr
    from ...gemm.batch_invariant.spec import output_elem

    out = []
    for K, fmt in KBLOCK32_OUT_CASES:
        alg = GTermAlgebra()

        def term(m, n):
            return gsexpr(
                output_elem(alg, B.reduction_kblock(K, m, n, 32).canonical(alg), fmt)
            )

        out.append((K // 32, fmt, term(0, 0), term(3, 33)))
    return out


def kblock32_loop_cases() -> List[Tuple[int, str]]:
    """``(KB, sexpr)``: the engine's trace of ONE instruction step per block (the bodies' per-step commit) folded by the block digit in
    block order; ``MoEFp8Bs.promoteLoop32`` prints it."""
    from ...batch_invariant.fp_interface.term import TermAlgebra as GTermAlgebra
    from ...batch_invariant.fp_interface.term import sexpr as gsexpr

    low = canon.BlockedFold("kstep", 1)
    out = []
    for KB in KBLOCK32_LOOP_CASES:
        alg = GTermAlgebra()
        red = B.reduction_kblock(KB * 32, 0, 0, 32)
        partials = [low.trace(red, alg, 1, fixed={"block": kb}) for kb in range(KB)]
        out.append((KB, gsexpr(red.digits[1].reduce(alg, partials))))
    return out


def moe_cases() -> List[Tuple[str, str, str]]:
    """``(name, Python S-expression, Lean term)`` of the class's chains: the gated GEMM1 (silu), the ungated GEMM1 (relu2),
    GEMM2."""
    alg = mterm.TermAlgebra()
    s = B.Fp8BsMoESpec(256, 128, 2, 8, alg=alg)
    su = B.Fp8BsMoESpec(256, 128, 2, 8, act="relu2", gated=False, alg=alg)
    s32 = B.Fp8BsMoESpec(256, 128, 2, 8, alg=alg, block=32)
    su32 = B.Fp8BsMoESpec(256, 128, 2, 8, act="relu2", gated=False, alg=alg, block=32)
    return [
        (
            "e4m3bs_gemm1_silu",
            mterm.sexpr(s.gemm1()),
            'MoEFp8Bs.e4m3bsGemm1T .silu (.leaf "g" [0, 0]) (.leaf "u" [0, 0])',
        ),
        (
            "e4m3bs_gemm1_relu2",
            mterm.sexpr(su.gemm1()),
            'MoEFp8Bs.e4m3bsGemm1UngatedT .relu2 (.leaf "u" [0, 0])',
        ),
        (
            "e4m3bs_gemm2",
            mterm.sexpr(s.gemm2()),
            'MoEFp8Bs.e4m3bsGemm2T (.leaf "acc2" [0, 0])',
        ),
        # the 32-wide classes: the same chains
        (
            "e4m3bs32_gemm1_silu",
            mterm.sexpr(s32.gemm1()),
            'MoEFp8Bs.e4m3bs32Gemm1T .silu (.leaf "g" [0, 0]) (.leaf "u" [0, 0])',
        ),
        (
            "e4m3bs32_gemm1_relu2",
            mterm.sexpr(su32.gemm1()),
            'MoEFp8Bs.e4m3bs32Gemm1UngatedT .relu2 (.leaf "u" [0, 0])',
        ),
        (
            "e4m3bs32_gemm2",
            mterm.sexpr(s32.gemm2()),
            'MoEFp8Bs.e4m3bs32Gemm2T (.leaf "acc2" [0, 0])',
        ),
    ]


def space():
    from .schedule_fp8bs import grouped_knobs_fp8bs

    return grouped_knobs_fp8bs()


def swap_space():
    from .schedule_fp8bs import swapped_knobs_fp8bs

    return swapped_knobs_fp8bs()


def dispatch_cases(block: int = 128) -> Dict:
    """Every record the block-scale dispatcher returns at the class ``block`` over ``GEOMS`` x ``TS`` x every nominal part x the
    chains x both requantization spellings (the standalone one alone at 32), with the first (T, H, I, E) that selects it."""
    from ...gemm.batch_invariant.heuristic import PARTS
    from .blocks.epilogue_fp8bs import GEMM2_CHAIN, gemm1_chain
    from .schedule_fp8bs import can_implement_fp8bs, default_configs_fp8bs

    disp: Dict = {}
    for H, I, E, k in GEOMS:
        for T in TS:
            for family, sm in PARTS:
                for act, gated in ACTS:
                    for fused in (False, True) if block == 128 else (False,):
                        c1, c2 = default_configs_fp8bs(
                            T,
                            k,
                            H,
                            I,
                            E,
                            sm,
                            family,
                            act,
                            gated,
                            (),
                            fused,
                            nominal=True,
                            block=block,
                        )
                        assert can_implement_fp8bs(
                            c1,
                            2 * I if gated else I,
                            H,
                            E,
                            gemm1_chain(act, (), gated, fused),
                            None,
                            block,
                        )
                        assert can_implement_fp8bs(
                            c2, H, I, E, GEMM2_CHAIN, None, block
                        )
                        disp.setdefault(c1, (T, H, I, E))
                        disp.setdefault(c2, (T, H, I, E))
    return disp


def render() -> str:
    """The content of ``MoEFp8BsCases.lean``."""
    L = [
        "import BIKernels.Canon",
        "import BIKernels.QuantExport",
        "import BIKernels.FP8BlockQuant",
        "import BIKernels.MoE",
        "import BIKernels.MoEFp8Bs",
        "import BIKernels.Cases",
        "namespace BIK",
        "-- GENERATED by flashinfer/fused_moe/batch_invariant/lean_cases_fp8bs.py from the Python spec / engine; do not edit.",
        "set_option maxRecDepth 8192",
        "open GEMM",
        "",
        "def fp8bsqRow : Nat → Term := fun i => Term.x i",
        "",
        "/-! The fp8 block quantizer class `fp8bs.quant` (`fused_moe/batch_invariant/spec_fp8bs.py` vs `FP8BlockQuant.lean`): the block "
        "amax S-expression of blocks 0 and 1, every placement's kernel term (the engine's `execC .fmaxc` at unit (0,0,0)) and spec "
        "term with the invariance theorem instantiated, the scale and code terms, the half-tree split of the fused requantization. -/",
    ]
    bqs, bqo = fp8bsq_cases(), fp8bsq_out_cases()
    L.append(f"def fp8bsqPy_amax0 : String := {lean_str(bqo['amax0'])}")
    L.append(f"def fp8bsqPy_amax1 : String := {lean_str(bqo['amax1'])}")
    L.append(
        "theorem fp8bsq_spec_amax0 : osexpr (FP8BQ.blockAmax fp8bsqRow 0) = fp8bsqPy_amax0 := by decide +kernel"
    )
    L.append(
        "theorem fp8bsq_spec_amax1 : osexpr (FP8BQ.blockAmax fp8bsqRow 1) = fp8bsqPy_amax1 := by decide +kernel"
    )
    for i, c in enumerate(bqs):
        rhs = "fp8bsqPy_amax0" if c["kernel"] == bqo["amax0"] else lean_str(c["kernel"])
        L.append(f"def fp8bsqPl{i} : List (List Factor) := {c['ps']}  -- {c['name']}")
        L.append(
            f"theorem fp8bsq{i}_ok : PlacementOK [7] fp8bsqPl{i} := by decide +kernel"
        )
        L.append(
            f"theorem fp8bsq{i}_kernel : osexpr (execC .fmaxc (flatten fp8bsqPl{i}) (FP8BQ.blockLeaf fp8bsqRow 0) (zeroCoords (flatten fp8bsqPl{i}))) = {rhs} := by decide +kernel"
        )
        L.append(
            f"theorem fp8bsq{i}_inv : OEquiv (execC .fmaxc (flatten fp8bsqPl{i}) (FP8BQ.blockLeaf fp8bsqRow 0) (zeroCoords (flatten fp8bsqPl{i}))) (FP8BQ.blockAmax fp8bsqRow 0) :="
        )
        L.append(
            f"  FP8BQ.block_amax_invariance fp8bsqRow 0 fp8bsqPl{i} _ fp8bsq{i}_ok (coordsOK_zero _)"
        )
    L.append(
        "def fp8bsqCases : List (String × List (List Factor)) := ["
        + ", ".join(f'("{c["name"]}", fp8bsqPl{i})' for i, c in enumerate(bqs))
        + "]"
    )

    def ref(sx: str) -> str:
        return (
            _with_ref(sx, bqo["amax0"], "fp8bsqPy_amax0")
            if bqo["amax0"] in sx
            else _with_ref(sx, bqo["amax1"], "fp8bsqPy_amax1")
        )

    L.append(
        f"theorem fp8bsq_s0 : osexpr (FP8BQ.specS fp8bsqRow 0) = {ref(bqo['s0'])} := by decide +kernel"
    )
    L.append(
        f"theorem fp8bsq_s1 : osexpr (FP8BQ.specS fp8bsqRow 1) = {ref(bqo['s1'])} := by decide +kernel"
    )
    L.append(
        f"theorem fp8bsq_code0 : osexpr (FP8BQ.specCode fp8bsqRow 0) = {ref(bqo['code0'])} := by decide +kernel"
    )
    L.append(
        f"theorem fp8bsq_codeL : osexpr (FP8BQ.specCode fp8bsqRow {FP8BSQ_K - 1}) = {ref(bqo['codeL'])} := by decide +kernel"
    )
    L.append(
        "theorem fp8bsq_halves : ∀ (b : Nat), FP8BQ.blockAmax fp8bsqRow b = ocomb .fmaxc (FP8BQ.halfAmax fp8bsqRow b 0) "
        "(FP8BQ.halfAmax fp8bsqRow b 1) := fun b => FP8BQ.blockAmax_halves fp8bsqRow b"
    )
    L.append("")
    L.append(
        "/-! The MoE class (`spec_fp8bs.Fp8BsMoESpec` vs `MoEFp8Bs.lean`): GEMM1's gated and ungated chains are the bf16 classes', "
        "GEMM2 the plain conversion (`mterm.sexpr` vs `MTerm.sexpr`). -/"
    )
    mcs = moe_cases()
    for name, sx, lt in mcs:
        L.append(
            f"theorem moe_{name} : ({lt}).sexpr = {lean_str(sx)} := by decide +kernel"
        )
    L.append(
        "def moeFp8BsCases : List (String × MoE.MTerm) := ["
        + ", ".join(f'("{name}", {lt})' for name, _, lt in mcs)
        + "]"
    )
    L.append("")
    sp = space()
    L.append(
        "/-! The grouped block-scale space (`schedule_fp8bs.grouped_knobs_fp8bs`: the dense block-scale class's records at swizzle 1 "
        "with a tile of two promotion warpgroups): every record checked against the kernel's static constraints, a member of the "
        "dense block-scale space, the `space_invariantKB` theorem instantiated on it. -/"
    )
    L.append(
        "def moeE4m3BsSpace : List Config := [" + ", ".join(k.lean() for k in sp) + "]"
    )
    L.append(
        f"theorem moe_e4m3bs_space_size : moeE4m3BsSpace.length = {len(sp)} := by decide"
    )
    L.append(
        "theorem moe_e4m3bs_space_ok : moeE4m3BsSpace.all MoEFp8Bs.groupedBlockScaleFp8OK = true := by decide +kernel"
    )
    L.append(
        "theorem moe_e4m3bs_space_in_dense : moeE4m3BsSpace.all (fun c => gemmFp8BsSpace.any (fun s => s == c)) = true := by decide +kernel"
    )
    L.append(
        "theorem moe_e4m3bs_space_invariant : ∀ c₁ ∈ moeE4m3BsSpace, ∀ c₂ ∈ moeE4m3BsSpace, ∀ (S₁ S₂ : Nat) (p q : GTerm) "
        "(a : RowData) (b : ColData) (KB : Nat), tileResultKB c₁ S₁ p a b KB = tileResultKB c₂ S₂ q a b KB := "
        "space_invariantKB moeE4m3BsSpace"
    )
    sw = swap_space()
    L.append(
        "/-! The swapped decode space of the class (`blocks/swap.py` on the promoted operand block: the weights on the MMA's M in a 64- / 128-row "
        "tile, the tokens as the N tile): every record checked against the swapped body's static constraints, disjoint from the "
        "dense-derived records, the tile result invariant over the union of the two spaces (the promotion loop does not read the "
        "orientation). -/"
    )
    L.append(
        "def moeE4m3BsSwapSpace : List Config := ["
        + ", ".join(k.lean() for k in sw)
        + "]"
    )
    L.append(
        f"theorem moe_e4m3bs_swap_space_size : moeE4m3BsSwapSpace.length = {len(sw)} := by decide"
    )
    L.append(
        "theorem moe_e4m3bs_swap_space_ok : moeE4m3BsSwapSpace.all MoEFp8Bs.groupedBlockScaleFp8SwapOK = true := by decide +kernel"
    )
    L.append(
        "theorem moe_e4m3bs_swap_space_not_dense : moeE4m3BsSwapSpace.all (fun c => !MoEFp8Bs.groupedBlockScaleFp8OK c) = true := by decide +kernel"
    )
    L.append(
        "theorem moe_e4m3bs_both_spaces_invariant : ∀ c₁ ∈ moeE4m3BsSpace ++ moeE4m3BsSwapSpace, ∀ c₂ ∈ moeE4m3BsSpace ++ moeE4m3BsSwapSpace, "
        "∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (KB : Nat), tileResultKB c₁ S₁ p a b KB = tileResultKB c₂ S₂ q a b KB := "
        "space_invariantKB (moeE4m3BsSpace ++ moeE4m3BsSwapSpace)"
    )
    disp = dispatch_cases()
    L.append(
        "/-! Every knob record the block-scale MoE dispatcher returns over the bench geometries x token counts x every nominal "
        "part x the chains x both requantization spellings, with the first (T, H, I, E) that selects it: a member of one of the "
        "two spaces above (R3 membership). -/"
    )
    L.append(
        "def moeE4m3BsDispatch : List (Nat × Nat × Nat × Nat × Config) := ["
        + ", ".join(
            f"({T}, {H}, {I}, {E}, {k.lean()})" for k, (T, H, I, E) in disp.items()
        )
        + "]"
    )
    L.append(
        "theorem moe_e4m3bs_dispatch_in_space : moeE4m3BsDispatch.all (fun ⟨_, _, _, _, c⟩ => (moeE4m3BsSpace ++ moeE4m3BsSwapSpace).any (fun s => s == c)) = true := by decide +kernel"
    )
    # ---- the 32-wide classes
    L.append("")
    L.append(
        "/-! The 32-wide block-scale classes `moe.e4m3bs32.<act>` (`deepseek_fp8_block_size = 32`): the GEMM core `gemm.e4m3.kblock32` "
        "(`spec_fp8bs.reduction_kblock` at 32 vs `MoEFp8Bs.kblock32`) at (0, 0) and at (3, 33) -- the latter reads the second 32-column "
        "weight scale block -- node for node, the bodies' per-step promotion loop (`MoEFp8Bs.promoteLoop32`) against the engine's one-step "
        "traces assembled by the block digit, a different term from the 128-wide class at the same K, the tile result invariant over the "
        "union of both record spaces, and the dispatcher's picks at the class members. -/"
    )
    for i, (KB, fmt, sx0, sx1) in enumerate(kblock32_cases()):
        L.append(
            f'theorem kblock32_{i}_spec : (MoEFp8Bs.outputKblock32 "{fmt}" ⟨0⟩ ⟨0⟩ {KB}).sexpr = {lean_str(sx0)} := by decide +kernel  -- K={KB * 32}'
        )
        L.append(
            f'theorem kblock32_{i}_spec2 : (MoEFp8Bs.outputKblock32 "{fmt}" ⟨3⟩ ⟨33⟩ {KB}).sexpr = {lean_str(sx1)} := by decide +kernel'
        )
    L.append(
        "def kblock32OutCases : List (Nat × String) := ["
        + ", ".join(f'({K // 32}, "{f}")' for K, f in KBLOCK32_OUT_CASES)
        + "]"
    )
    L.append(
        "theorem kblock32_loop_eq_class : ∀ (S KB : Nat) (a : RowData) (b : ColData), MoEFp8Bs.promoteLoop32 a b S KB = MoEFp8Bs.kblock32 a b KB := "
        "fun S KB a b => MoEFp8Bs.promoteLoop32_eq_kblock32 a b S KB"
    )
    for i, (KB, sx) in enumerate(kblock32_loop_cases()):
        L.append(
            f"theorem kblock32loop{i} : (MoEFp8Bs.promoteLoop32 ⟨0⟩ ⟨0⟩ 3 {KB}).sexpr = {lean_str(sx)} := by decide +kernel"
        )
    L.append(
        "def kblock32LoopCases : List Nat := ["
        + ", ".join(str(n) for n in KBLOCK32_LOOP_CASES)
        + "]"
    )
    L.append(
        "theorem kblock32_ne_kblock128_sexpr : (MoEFp8Bs.kblock32 ⟨0⟩ ⟨0⟩ 4).sexpr ≠ (kblock128 ⟨0⟩ ⟨0⟩ 1).sexpr := by decide +kernel"
    )
    L.append(
        "theorem moe_e4m3bs32_both_spaces_invariant : ∀ c₁ ∈ moeE4m3BsSpace ++ moeE4m3BsSwapSpace, ∀ c₂ ∈ moeE4m3BsSpace ++ moeE4m3BsSwapSpace, "
        "∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (KB : Nat), MoEFp8Bs.tileResultKB32 c₁ S₁ p a b KB = MoEFp8Bs.tileResultKB32 c₂ S₂ q a b KB := "
        "MoEFp8Bs.space_invariantKB32 (moeE4m3BsSpace ++ moeE4m3BsSwapSpace)"
    )
    disp32 = dispatch_cases(32)
    L.append(
        "def moeE4m3Bs32Dispatch : List (Nat × Nat × Nat × Nat × Config) := ["
        + ", ".join(
            f"({T}, {H}, {I}, {E}, {k.lean()})" for k, (T, H, I, E) in disp32.items()
        )
        + "]"
    )
    L.append(
        "theorem moe_e4m3bs32_dispatch_in_space : moeE4m3Bs32Dispatch.all (fun ⟨_, _, _, _, c⟩ => (moeE4m3BsSpace ++ moeE4m3BsSwapSpace).any (fun s => s == c)) = true := by decide +kernel"
    )
    # the radix-32 quantizer instance
    L.append("")
    L.append(
        "/-! The 32-wide instance of the block quantizer (`spec_fp8bs` at `block = 32` vs `FP8BlockQuant.blockAmax32`): the 32-leaf block "
        "amax S-expression of blocks 0 and 1, every placement's kernel term and spec term with the invariance theorem instantiated, the "
        "scale and code chains at K = 64, and its distinctness from the 128-wide class's amax. -/"
    )
    bqs32, bqo32 = fp8bsq32_cases(), fp8bsq32_out_cases()
    L.append(f"def fp8bsq32Py_amax0 : String := {lean_str(bqo32['amax0'])}")
    L.append(f"def fp8bsq32Py_amax1 : String := {lean_str(bqo32['amax1'])}")
    L.append(
        "theorem fp8bsq32_spec_amax0 : osexpr (FP8BQ.blockAmax32 fp8bsqRow 0) = fp8bsq32Py_amax0 := by decide +kernel"
    )
    L.append(
        "theorem fp8bsq32_spec_amax1 : osexpr (FP8BQ.blockAmax32 fp8bsqRow 1) = fp8bsq32Py_amax1 := by decide +kernel"
    )
    L.append(
        "theorem fp8bsq32_ne_fp8bsq : osexpr (FP8BQ.blockAmax32 fp8bsqRow 0) ≠ osexpr (FP8BQ.blockAmax fp8bsqRow 0) := by decide +kernel"
    )
    for i, c in enumerate(bqs32):
        rhs = (
            "fp8bsq32Py_amax0"
            if c["kernel"] == bqo32["amax0"]
            else lean_str(c["kernel"])
        )
        L.append(f"def fp8bsq32Pl{i} : List (List Factor) := {c['ps']}  -- {c['name']}")
        L.append(
            f"theorem fp8bsq32_{i}_ok : PlacementOK [5] fp8bsq32Pl{i} := by decide +kernel"
        )
        L.append(
            f"theorem fp8bsq32_{i}_kernel : osexpr (execC .fmaxc (flatten fp8bsq32Pl{i}) (FP8BQ.blockLeaf32 fp8bsqRow 0) (zeroCoords (flatten fp8bsq32Pl{i}))) = {rhs} := by decide +kernel"
        )
        L.append(
            f"theorem fp8bsq32_{i}_inv : OEquiv (execC .fmaxc (flatten fp8bsq32Pl{i}) (FP8BQ.blockLeaf32 fp8bsqRow 0) (zeroCoords (flatten fp8bsq32Pl{i}))) (FP8BQ.blockAmax32 fp8bsqRow 0) :="
        )
        L.append(
            f"  FP8BQ.block_amax_invariance32 fp8bsqRow 0 fp8bsq32Pl{i} _ fp8bsq32_{i}_ok (coordsOK_zero _)"
        )
    L.append(
        "def fp8bsq32Cases : List (String × List (List Factor)) := ["
        + ", ".join(f'("{c["name"]}", fp8bsq32Pl{i})' for i, c in enumerate(bqs32))
        + "]"
    )

    def ref32(sx: str) -> str:
        return (
            _with_ref(sx, bqo32["amax0"], "fp8bsq32Py_amax0")
            if bqo32["amax0"] in sx
            else _with_ref(sx, bqo32["amax1"], "fp8bsq32Py_amax1")
        )

    L.append(
        f"theorem fp8bsq32_s0 : osexpr (FP8BQ.specS32 fp8bsqRow 0) = {ref32(bqo32['s0'])} := by decide +kernel"
    )
    L.append(
        f"theorem fp8bsq32_s1 : osexpr (FP8BQ.specS32 fp8bsqRow 1) = {ref32(bqo32['s1'])} := by decide +kernel"
    )
    L.append(
        f"theorem fp8bsq32_code0 : osexpr (FP8BQ.specCode32 fp8bsqRow 0) = {ref32(bqo32['code0'])} := by decide +kernel"
    )
    L.append(
        f"theorem fp8bsq32_codeL : osexpr (FP8BQ.specCode32 fp8bsqRow {FP8BSQ32_K - 1}) = {ref32(bqo32['codeL'])} := by decide +kernel"
    )
    L.append("end BIK")
    return "\n".join(L) + "\n"


def summary() -> str:
    return (
        f"fp8bs lean_cases coverage: quant placements={len(fp8bsq_cases())} quant output cases={len(FP8BSQ_OUT_WHICH)} "
        f"moe chains={len(moe_cases())} space={len(space())} dispatch records={len(dispatch_cases())} "
        f"kblock32 output cases={2 * len(KBLOCK32_OUT_CASES)} kblock32 loop cases={len(KBLOCK32_LOOP_CASES)} "
        f"quant32 placements={len(fp8bsq32_cases())} dispatch32 records={len(dispatch_cases(32))}"
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
