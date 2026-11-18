-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.SparseMLA
import BIKernels.SparseMLACases
/-! `lake exe sparsemlaconformance`: print the Lean model's terms for the generated sparse MLA cases, as the class term (`smla`) and through the fixup kernel's model with two tiles per unit, the units in descending order and a non-empty workspace (`smlafixup`), and the key reductions as a unit of each placement holds them (`smlakeys`) and as the canonical tree (`smlakeysspec`). The Python test (`tests/batch_invariant/test_sparse_mla_lean.py`) compares both with the Python spec's S-expressions (`flashinfer/mla/batch_invariant/spec.py`). -/
open BIK BIK.SparseMLA

def main : IO Unit := do
  let junk : Attn.State := ⟨.sym "junk", .sym "junk", .sym "junk"⟩
  for (name, f, P) in smlaCases do
    IO.println s!"smla {name} {outputSexpr f P}"
    let order := (unitsOf 2 (nTiles P.n)).reverse
    IO.println s!"smlafixup {name} {sexprF f (kernel f ⟨16, 8, 0, 2, 128, 4⟩ order (fun _ => junk) P)}"
  for (name, cb, n, ps, cs) in smlaKeyCases do
    let got := (keysSexpr cb n ps cs).getD "none"
    let want := (keysSpecSexpr cb n).getD "none"
    IO.println s!"smlakeys {name} {got}"
    IO.println s!"smlakeysspec {name} {want}"
