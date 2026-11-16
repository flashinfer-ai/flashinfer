-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
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
  -- the block-scale fp8 class: the output at (0, 0) and at (3, 129) (the second weight scale block) per (blocks, format);
  -- the kernel's promotion loop at sampled block counts
  for (KB, fmt) in gemmFp8BsOutCases do
    IO.println s!"gemmfp8bs {KB} {fmt} {(outputKblock fmt ⟨0⟩ ⟨0⟩ KB).sexpr}"
    IO.println s!"gemmfp8bs2 {KB} {fmt} {(outputKblock fmt ⟨3⟩ ⟨129⟩ KB).sexpr}"
  for KB in gemmFp8BsLoopCases do
    IO.println s!"gemmfp8bsloop {KB} {(promoteLoop ⟨0⟩ ⟨0⟩ 3 KB).sexpr}"
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
  -- the 4-bit classes: the mma64bs / mma128bs (NVFP4) and mma64mx / mma128mx4 (MXFP4) steps, the alpha chain, the
  -- 64-granule and 128-granule chunk rules, the sm_107 NVFP4 class rule
  for (steps, C) in gemmbsCases do
    let t := if C == 1 then kfoldBS ⟨0⟩ ⟨0⟩ steps else kchunkBS ⟨0⟩ ⟨0⟩ C (steps / C)
    IO.println s!"gemmbs {steps} {C} {(outputElemAlpha "bf16" t).sexpr}"
  for nblk in gemmbsLoopCases do
    IO.println s!"gemmbsloop {nblk} {(kblockLoopBS ⟨0⟩ ⟨0⟩ gemmMainloopInnerNvfp4 3 .zero nblk).sexpr}"
  for (steps, fmt) in gemmbsOutCases do
    IO.println s!"gemmbsout {steps} {fmt} {(outputKfoldBS fmt ⟨0⟩ ⟨0⟩ steps).sexpr}"
  for (N, K, _) in chunkTableFp4 do
    IO.println s!"chunksfp4 {N} {K} {chunksForFp4 N K}"
  for (steps, C) in gemmbs128Cases do
    let t := if C == 1 then kfoldBS128 ⟨0⟩ ⟨0⟩ steps else kchunkBS128 ⟨0⟩ ⟨0⟩ C (steps / C)
    IO.println s!"gemmbs128 {steps} {C} {(outputElemAlpha "bf16" t).sexpr}"
  for nblk in gemmbs128LoopCases do
    IO.println s!"gemmbs128loop {nblk} {(kblockLoopBS128 ⟨0⟩ ⟨0⟩ gemmMainloopInnerNvfp4K128 3 .zero nblk).sexpr}"
  for (steps, fmt) in gemmbs128OutCases do
    IO.println s!"gemmbs128out {steps} {fmt} {(outputKfoldBS128 fmt ⟨0⟩ ⟨0⟩ steps).sexpr}"
  for (N, K, _) in chunkTableFp4K128 do
    IO.println s!"chunksfp4k128 {N} {K} {chunksForFp4K128 N K}"
  for (N, K, _) in fp4ClassTable do
    IO.println s!"fp4class {N} {K} {if fp4ClassK128 N K then 1 else 0}"
  for (steps, C) in gemmmx4Cases do
    let t := if C == 1 then kfoldMX4 ⟨0⟩ ⟨0⟩ steps else kchunkMX4 ⟨0⟩ ⟨0⟩ C (steps / C)
    IO.println s!"gemmmx4 {steps} {C} {(outputElemAlpha "bf16" t).sexpr}"
  for nblk in gemmmx4LoopCases do
    IO.println s!"gemmmx4loop {nblk} {(kblockLoopMX4 ⟨0⟩ ⟨0⟩ gemmMainloopInnerMx4 3 .zero nblk).sexpr}"
  for (steps, fmt) in gemmmx4OutCases do
    IO.println s!"gemmmx4out {steps} {fmt} {(outputKfoldMX4 fmt ⟨0⟩ ⟨0⟩ steps).sexpr}"
  for (steps, C) in gemmmx4k128Cases do
    let t := if C == 1 then kfoldMX4K128 ⟨0⟩ ⟨0⟩ steps else kchunkMX4K128 ⟨0⟩ ⟨0⟩ C (steps / C)
    IO.println s!"gemmmx4k128 {steps} {C} {(outputElemAlpha "bf16" t).sexpr}"
  for nblk in gemmmx4k128LoopCases do
    IO.println s!"gemmmx4k128loop {nblk} {(kblockLoopMX4K128 ⟨0⟩ ⟨0⟩ gemmMainloopInnerMx4K128 3 .zero nblk).sexpr}"
  for (steps, fmt) in gemmmx4k128OutCases do
    IO.println s!"gemmmx4k128out {steps} {fmt} {(outputKfoldMX4K128 fmt ⟨0⟩ ⟨0⟩ steps).sexpr}"
  for (name, P) in attnCases do
    IO.println s!"attn {name} {(Attn.output P).sexpr}"
