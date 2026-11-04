import BIKernels.Export
import BIKernels.Cases
/-! `lake exe conformance`: print the Lean model's terms for the generated cases. The Python test
(`tests/batch_invariant/test_lean.py`, the Lean tier) compares these strings with the Python spec's / engine's
S-expressions and the dispatcher's chunk rule. -/
open BIK BIK.GEMM

def main : IO Unit := do
  for (steps, C) in gemmCases do
    let t := if C == 1 then kfold ⟨0⟩ ⟨0⟩ steps else kchunk ⟨0⟩ ⟨0⟩ C (steps / C)
    IO.println s!"gemm {steps} {C} {(outputElem "bf16" t).sexpr}"
  for nblk in gemmLoopCases do
    IO.println s!"gemmloop {nblk} {(kblockLoop ⟨0⟩ ⟨0⟩ gemmMainloopInner 3 .zero nblk).sexpr}"
  for (steps, fmt) in gemmOutCases do
    IO.println s!"gemmout {steps} {fmt} {(outputKfold fmt ⟨0⟩ ⟨0⟩ steps).sexpr}"
  for (N, K, _) in chunkTable do
    IO.println s!"chunks {N} {K} {chunksFor N K}"
