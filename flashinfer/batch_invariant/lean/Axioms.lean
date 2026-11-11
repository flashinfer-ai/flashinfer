-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import Lean
import BIKernels
/-! `lake exe axioms`: the enforced axiom policy of the library.

Every declaration of the `BIKernels.*` modules is checked:

* `sorryAx` anywhere fails the build;
* a theorem may depend on Lean's standard axioms (`propext`, `Classical.choice`, `Quot.sound`) and on nothing else,
  except the named theorems in `decidedTheorems` below, which are decided by `native_decide` and therefore depend on
  `Lean.ofReduceBool` as well: the axiom that `Lean.reduceBool c = true` implies `c = true`, i.e. it trusts the
  compiled evaluator (Lean's code generator and interpreter) for the evaluation of these closed Boolean terms. Every
  other generated conformance case (`BIKernels.Cases`) uses kernel `decide` and needs no such trust.

The claim the policy enforces is therefore "Lean's standard axioms, plus `Lean.ofReduceBool` for the named
exhaustive table checks", never "standard axioms only". -/
open Lean

def standardAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]
def decidedAxiom : Name := ``Lean.ofReduceBool
/-- The named `native_decide` theorems: the exhaustive checks over the generators' configuration spaces and the GEMM
dispatcher's picks (hundreds of records, beyond what kernel `decide` evaluates in reasonable time), the attention spec cases
(terms of tens of kilobytes) and the attention class-separation theorems. -/
def decidedTheorems : List Name :=
  [``BIK.gemm_space_ok, ``BIK.gemm_space_c4_ok, ``BIK.gemm_space_c4_cta_twins, ``BIK.gemm_dispatch_in_space,
   ``BIK.gemm_fp8_space_ok, ``BIK.gemm_fp8_space_c4_ok, ``BIK.gemm_fp8_space_c4_cta_twins, ``BIK.gemm_fp8_dispatch_in_space,
   ``BIK.gemm_fp8bs_space_ok, ``BIK.gemm_fp8bs_dispatch_in_space,
   ``BIK.gemm_mx8_space_ok, ``BIK.gemm_mx8_space_c4_ok, ``BIK.gemm_mx8_space_c4_cta_twins, ``BIK.gemm_mx8_dispatch_in_space,
   ``BIK.gemm_mx8_space_decode_ok, ``BIK.gemm_mx8_space_c2_ok,
   ``BIK.gemm_mx8k64_space_ok, ``BIK.gemm_mx8k64_space_c4_ok, ``BIK.gemm_mx8k64_space_c4_cta_twins, ``BIK.gemm_mx8k64_dispatch_in_space,
   ``BIK.gemm_mx8k64_space_decode_ok, ``BIK.gemm_mx8k64_space_c2_ok,
   ``BIK.gemm_nvfp4_space_ok, ``BIK.gemm_nvfp4_space_c4_ok, ``BIK.gemm_nvfp4_space_c4_cta_twins, ``BIK.gemm_nvfp4_dispatch_in_space,
   ``BIK.gemm_nvfp4_space_decode_ok, ``BIK.gemm_nvfp4_space_c2_ok,
   ``BIK.gemm_nvfp4k128_space_ok, ``BIK.gemm_nvfp4k128_space_c4_ok, ``BIK.gemm_nvfp4k128_space_c4_cta_twins, ``BIK.gemm_nvfp4k128_dispatch_in_space,
   ``BIK.gemm_nvfp4k128_space_decode_ok, ``BIK.gemm_nvfp4k128_space_c2_ok,
   ``BIK.gemm_mx4_space_ok, ``BIK.gemm_mx4_space_c4_ok, ``BIK.gemm_mx4_space_c4_cta_twins, ``BIK.gemm_mx4_dispatch_in_space,
   ``BIK.gemm_mx4_space_decode_ok, ``BIK.gemm_mx4_space_c2_ok,
   ``BIK.gemm_mx4k128_space_ok, ``BIK.gemm_mx4k128_space_c4_ok, ``BIK.gemm_mx4k128_space_c4_cta_twins, ``BIK.gemm_mx4k128_dispatch_in_space,
   ``BIK.gemm_mx4k128_space_decode_ok, ``BIK.gemm_mx4k128_space_c2_ok,
   ``BIK.attn_a0_spec, ``BIK.attn_a1_spec, ``BIK.attn_a2_spec, ``BIK.attn_a3_spec, ``BIK.attn_a4_spec, ``BIK.attn_a5_spec, ``BIK.attn_a6_spec, ``BIK.attn_a7_spec, ``BIK.attn_a8_spec, ``BIK.attn_a9_spec, ``BIK.attn_space_d64_size, ``BIK.attn_space_d64_ok, ``BIK.attn_space_d128_size, ``BIK.attn_space_d128_ok, ``BIK.attn_space_d256_size, ``BIK.attn_space_d256_ok, ``BIK.Attn.output_depends_on_C, ``BIK.Attn.output_depends_on_D, ``BIK.Attn.output_depends_on_D_64]
def moduleOf (env : Environment) (n : Name) : Option Name :=
  match env.getModuleIdxFor? n with
  | some idx => env.header.moduleNames[idx.toNat]?
  | none => none

def isOurs (m : Option Name) : Bool :=
  match m with
  | some m => m.toString.startsWith "BIKernels"
  | none => false

/-- Compiler-generated auxiliaries (`_cstage1` / `_cstage2` stages carrying `lcProof` / `lcUnreachable`, `_spec_n`
specialisations, `_elambda_n` closures, the `_auxLemma.n` helpers `native_decide` emits for a decided theorem, the
`_private` prefix of derived instances) are not proofs: every user theorem they belong to is checked by its own name. -/
def isGenerated (n : Name) : Bool :=
  n.components.any fun c => c.toString.startsWith "_"

def main : IO UInt32 := do
  initSearchPath (← findSysroot)
  let env ← importModules #[{ module := `BIKernels }] {} 0
  let mut failures : Array String := #[]
  let mut checked := 0
  let mut decided := 0
  let mut sorries := 0
  for (n, _) in env.constants.toList do
    let m := moduleOf env n
    unless isOurs m do continue
    if isGenerated n then continue
    let (_, st) := ((CollectAxioms.collect n).run env).run {}
    checked := checked + 1
    if st.axioms.contains ``sorryAx then
      sorries := sorries + 1
      failures := failures.push s!"{n}: depends on sorryAx"
    let allowDecided := decidedTheorems.contains n
    if st.axioms.contains decidedAxiom then decided := decided + 1
    for a in st.axioms do
      if a == ``sorryAx then continue
      if standardAxioms.contains a then continue
      if a == decidedAxiom && allowDecided then continue
      failures := failures.push s!"{n}: axiom {a} is outside the policy (module {m.getD .anonymous})"
  IO.println s!"axioms check: {checked} user declarations of BIKernels.*, {decided} named table theorems decided by native_decide (Lean.ofReduceBool), {sorries} with sorry, {failures.size} violations"
  for f in failures do
    IO.eprintln s!"  VIOLATION {f}"
  return (if failures.isEmpty then 0 else 1)
