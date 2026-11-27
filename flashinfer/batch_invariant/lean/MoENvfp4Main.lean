import BIKernels.MoENvfp4
import BIKernels.MoENvfp4Cases
/-! `lake exe moenvfp4conformance`: print the Lean model's terms for the generated NVFP4 MoE cases. The Python test
(`tests/batch_invariant/test_moe_nvfp4_lean.py`, the Lean tier) compares these strings with the Python spec's
S-expressions (`flashinfer/fused_moe/batch_invariant/spec_nvfp4.py` through `mterm.sexpr`). -/
open BIK BIK.MoE

def main : IO Unit := do
  for (name, t) in moeNvfp4Cases do
    IO.println s!"moenvfp4 {name} {t.sexpr}"
