import BIKernels.MoEMxfp4
import BIKernels.MoEMxfp4Cases
/-! `lake exe moemxfp4conformance`: print the Lean model's terms for the generated MXFP4 MoE cases. The Python test
(`tests/batch_invariant/test_moe_mxfp4_lean.py`, the Lean tier) compares these strings with the Python spec's
S-expressions (`flashinfer/fused_moe/batch_invariant/spec_mxfp4.py` through `mterm.sexpr`). -/
open BIK BIK.MoE

def main : IO Unit := do
  for (name, t) in moeMxfp4Cases do
    IO.println s!"moemxfp4 {name} {t.sexpr}"
