import BIKernels.Export
import BIKernels.Cases
/-! `lake exe conformance`: print the Lean model's terms for the generated cases. The Python test
(`tests/batch_invariant/test_lean.py`, the Lean tier) compares these strings with the Python spec's / engine's
S-expressions and the dispatcher's chunk rule. -/
open BIK BIK.GEMM

def main : IO Unit := do
  for (steps, C) in gemmCases do
    let t := if C == 1 then kfold ⟨0⟩ ⟨0⟩ steps else kchunk ⟨0⟩ ⟨0⟩ C (steps / C)
    IO.println s!"gemm {steps} {C} {(outputElem false "bf16" t).sexpr}"
  for nblk in gemmLoopCases do
    IO.println s!"gemmloop {nblk} {(kblockLoop ⟨0⟩ ⟨0⟩ gemmMainloopInner 3 .zero nblk).sexpr}"
  for (steps, fmt) in gemmOutCases do
    IO.println s!"gemmout {steps} {fmt} {(outputKfold false fmt ⟨0⟩ ⟨0⟩ steps).sexpr}"
  for (N, K, _) in chunkTable do
    IO.println s!"chunks {N} {K} {chunksFor N K}"
  -- the fp8 classes: the mma32 step, the scaled chain, the 32-wide chunk rule
  for (steps, C) in gemm32Cases do
    let t := if C == 1 then kfold32 ⟨0⟩ ⟨0⟩ steps else kchunk32 ⟨0⟩ ⟨0⟩ C (steps / C)
    IO.println s!"gemm32 {steps} {C} {(outputElem true "bf16" t).sexpr}"
  for nblk in gemm32LoopCases do
    IO.println s!"gemm32loop {nblk} {(kblockLoop32 ⟨0⟩ ⟨0⟩ gemmMainloopInnerFp8 3 .zero nblk).sexpr}"
  for (steps, fmt) in gemm32OutCases do
    IO.println s!"gemm32out {steps} {fmt} {(outputKfold32 fmt ⟨0⟩ ⟨0⟩ steps).sexpr}"
  for (N, K, _) in chunkTableFp8 do
    IO.println s!"chunks32 {N} {K} {chunksForFp8 N K}"
