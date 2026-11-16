import BIKernels.MoE
import BIKernels.MoECases
/-! `lake exe moeconformance`: print the Lean model's terms for the generated MoE cases. The Python test
(`tests/batch_invariant/test_moe_lean.py`, the Lean tier) compares these strings with the Python spec's S-expressions
(`flashinfer/fused_moe/batch_invariant/spec.py` through `mterm.sexpr`). -/
open BIK BIK.MoE

def main : IO Unit := do
  for (name, t) in moeCases do
    IO.println s!"moe {name} {t.sexpr}"
