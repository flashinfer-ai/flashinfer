-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.Gdn
import BIKernels.GdnCases
/-! `lake exe gdnconformance`: print the Lean model's arithmetic for the generated GDN cases. The Python test
(`tests/batch_invariant/test_gdn_lean.py`, the Lean tier) compares these lines with `gdn/spec.py`. -/
open BIK

def main : IO Unit := do
  for c in gdnCases do
    let p := c.1
    IO.println s!"gdn {p} boundary {Gdn.boundary p} phase {Gdn.phase p} commit {Gdn.commits (Gdn.phase p)}"
  for c in gdnExtendCases do
    let s := Gdn.extendSplit c.1 c.2.1
    IO.println s!"gdnextend {c.1} {c.2.1} whole {s.1} tail {s.2} step {Gdn.usesStepKernel c.2.1}"
