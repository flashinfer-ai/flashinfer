import BIKernels.Kda
import BIKernels.KdaCases
/-! `lake exe kdaconformance`: print the Lean model's replay arithmetic for the generated KDA cases. The Python test
(`tests/batch_invariant/test_kda_lean.py`, the Lean tier) compares these lines with `kda/spec.py`. -/
open BIK

def main : IO Unit := do
  for c in kdaCases do
    let p := c.1
    IO.println s!"kda {p} boundary {Kda.boundary p} hist {Kda.historyLen p} commit {Kda.commits (Kda.historyLen p)}"
  for c in kdaCountCases do
    IO.println s!"kdacount {c.1} next {Kda.nextCount c.1}"
