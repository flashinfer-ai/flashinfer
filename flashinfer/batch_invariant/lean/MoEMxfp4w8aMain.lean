import BIKernels.MoEMxfp4w8a
import BIKernels.MoEMxfp4w8aCases
/-! `lake exe moemxfp4w8aconformance`: print the Lean model's terms for the generated mixed MXFP4 x MXFP8 MoE cases and the
size of the grouped record space. The Python test (`tests/batch_invariant/test_moe_mxfp4w8a_lean.py`, the Lean tier)
compares the strings with the Python spec's S-expressions (`flashinfer/fused_moe/batch_invariant/spec_mxfp4w8a.py` through
`mterm.sexpr`). -/
open BIK BIK.MoE

def main : IO Unit := do
  for (name, t) in moeMxfp4w8aCases do
    IO.println s!"moemxfp4w8a {name} {t.sexpr}"
  IO.println s!"moemxfp4w8aspace {moeMxfp4w8aSpace.length}"
  IO.println s!"moemxfp4w8aswapspace {moeMxfp4w8aSwapSpace.length}"
