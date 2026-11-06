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
  for (steps, C) in gemmBiasCases do
    let t := if C == 1 then kfold ⟨0⟩ ⟨0⟩ steps else kchunk ⟨0⟩ ⟨0⟩ C (steps / C)
    IO.println s!"gemmbias {steps} {C} {(outputElemBias "bf16" t ⟨0⟩).sexpr}"
  -- the MXFP8 classes: the mma32mx step (sm_100 / sm_103) and the mma64mx8 step (sm_107), the unscaled chain, the
  -- 128-granule chunk rule shared by both
  for (steps, C) in gemmmx8Cases do
    let t := if C == 1 then kfoldMX8 ⟨0⟩ ⟨0⟩ steps else kchunkMX8 ⟨0⟩ ⟨0⟩ C (steps / C)
    IO.println s!"gemmmx8 {steps} {C} {(outputElem false "bf16" t).sexpr}"
  for nblk in gemmmx8LoopCases do
    IO.println s!"gemmmx8loop {nblk} {(kblockLoopMX8 ⟨0⟩ ⟨0⟩ gemmMainloopInnerMx8 3 .zero nblk).sexpr}"
  for (steps, fmt) in gemmmx8OutCases do
    IO.println s!"gemmmx8out {steps} {fmt} {(outputKfoldMX8 fmt ⟨0⟩ ⟨0⟩ steps).sexpr}"
  for (N, K, _) in chunkTableMx do
    IO.println s!"chunksmx {N} {K} {chunksForMx N K}"
  for (steps, C) in gemmmx8k64Cases do
    let t := if C == 1 then kfoldMX8K64 ⟨0⟩ ⟨0⟩ steps else kchunkMX8K64 ⟨0⟩ ⟨0⟩ C (steps / C)
    IO.println s!"gemmmx8k64 {steps} {C} {(outputElem false "bf16" t).sexpr}"
  for nblk in gemmmx8k64LoopCases do
    IO.println s!"gemmmx8k64loop {nblk} {(kblockLoopMX8K64 ⟨0⟩ ⟨0⟩ gemmMainloopInnerMx8K64 3 .zero nblk).sexpr}"
  for (steps, fmt) in gemmmx8k64OutCases do
    IO.println s!"gemmmx8k64out {steps} {fmt} {(outputKfoldMX8K64 fmt ⟨0⟩ ⟨0⟩ steps).sexpr}"
