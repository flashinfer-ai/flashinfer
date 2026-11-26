import BIKernels.QuantExport
import BIKernels.MoEFp8BsCases
/-! `lake exe fp8bsconformance`: print the Lean model's terms for the generated block-scale fp8 MoE cases (the `fp8bs.quant`
class's placements and chains, the MoE class's chains). The Python test (`tests/batch_invariant/test_moe_fp8bs_lean.py`)
compares these strings with the Python spec's / engine's S-expressions. -/
open BIK

def main : IO Unit := do
  for (name, ps) in fp8bsqCases do
    IO.println s!"fp8bsq {name} spec {osexpr (FP8BQ.blockAmax fp8bsqRow 0)}"
    IO.println s!"fp8bsq {name} kernel {osexpr (execC .fmaxc (flatten ps) (FP8BQ.blockLeaf fp8bsqRow 0) (zeroCoords (flatten ps)))}"
  IO.println s!"fp8bsqout s0 {osexpr (FP8BQ.specS fp8bsqRow 0)}"
  IO.println s!"fp8bsqout s1 {osexpr (FP8BQ.specS fp8bsqRow 1)}"
  IO.println s!"fp8bsqout code0 {osexpr (FP8BQ.specCode fp8bsqRow 0)}"
  IO.println s!"fp8bsqout codeL {osexpr (FP8BQ.specCode fp8bsqRow 255)}"
  for (name, t) in moeFp8BsCases do
    IO.println s!"moe {name} {t.sexpr}"
  -- the 32-wide classes: the quantizer instance at radix 32, the GEMM core at (0, 0) and (3, 33), the per-step promotion loop
  for (name, ps) in fp8bsq32Cases do
    IO.println s!"fp8bsq32 {name} spec {osexpr (FP8BQ.blockAmax32 fp8bsqRow 0)}"
    IO.println s!"fp8bsq32 {name} kernel {osexpr (execC .fmaxc (flatten ps) (FP8BQ.blockLeaf32 fp8bsqRow 0) (zeroCoords (flatten ps)))}"
  IO.println s!"fp8bsq32out s0 {osexpr (FP8BQ.specS32 fp8bsqRow 0)}"
  IO.println s!"fp8bsq32out s1 {osexpr (FP8BQ.specS32 fp8bsqRow 1)}"
  IO.println s!"fp8bsq32out code0 {osexpr (FP8BQ.specCode32 fp8bsqRow 0)}"
  IO.println s!"fp8bsq32out codeL {osexpr (FP8BQ.specCode32 fp8bsqRow 63)}"
  for (KB, fmt) in kblock32OutCases do
    IO.println s!"kblock32 {KB} {fmt} {(MoEFp8Bs.outputKblock32 fmt ⟨0⟩ ⟨0⟩ KB).sexpr}"
    IO.println s!"kblock32b {KB} {fmt} {(MoEFp8Bs.outputKblock32 fmt ⟨3⟩ ⟨33⟩ KB).sexpr}"
  for KB in kblock32LoopCases do
    IO.println s!"kblock32loop {KB} {(MoEFp8Bs.promoteLoop32 ⟨0⟩ ⟨0⟩ 3 KB).sexpr}"
