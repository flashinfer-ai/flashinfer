import BIKernels.MlaSparse
import BIKernels.MlaSparseCases
/-! `lake exe mlasparseconformance`: print the Lean model's tile arithmetic for the generated sparse MLA cases. The
Python test (`tests/batch_invariant/test_mla_sparse.py`, the Lean tier) compares these lines with `mla/spec.py`. -/
open BIK

def main : IO Unit := do
  for c in mlaSparseCases do
    let n := c.1
    IO.println s!"mlasparse {n} tiles {MlaSparse.keyTiles n} last {MlaSparse.lastLive n}"
