import BIKernels.MoEMxfp8
import BIKernels.MoEMxfp8Cases
/-! `lake exe moemxfp8conformance`: print the Lean model's terms for the generated MXFP8 MoE cases and the size of the
grouped record space. The Python test (`tests/batch_invariant/test_moe_mxfp8_lean.py`, the Lean tier) compares the strings
with the Python spec's S-expressions (`flashinfer/fused_moe/batch_invariant/spec_mxfp8.py` through `mterm.sexpr`). -/
open BIK BIK.MoE

def main : IO Unit := do
  for (name, t) in moeMxfp8Cases do
    IO.println s!"moemxfp8 {name} {t.sexpr}"
  IO.println s!"moemxfp8space {moeMxfp8Space.length}"
