import BIKernels.QuantExport
import BIKernels.QuantCases
/-! `lake exe quantconformance`: print the Lean model's terms for the generated quantization cases. The Python test
(`tests/batch_invariant/test_quant_lean.py`, the Lean tier) compares these strings with the Python specs' / engine's
S-expressions. -/
open BIK

/-- The NVFP4 quantizer's placements print their kernel term at unit (0, 0, 0): the engine's `execC` over the row leaf. -/
def quantKernelTerm (nb : Nat) (ps : List (List Factor)) : Option Term :=
  execC .fmaxc (flatten ps) (Quant.rowLeaf nb (fun i => Term.x i)) (zeroCoords (flatten ps))

def main : IO Unit := do
  let qrow : Nat → Term := fun i => Term.x i
  -- the NVFP4 classes (Quant.lean): the row amax per placement (spec and kernel), the chains per class, the block amaxes
  for (K, kb, name, ps) in quantCases do
    let nb := K / 16
    IO.println s!"quant {K} {kb} {name} spec {osexpr (Quant.rowAmax nb kb qrow)}"
    IO.println s!"quant {K} {kb} {name} kernel {osexpr (quantKernelTerm nb ps)}"
  for (K, kb, vname, v) in quantOutCases do
    let nb := K / 16
    IO.println s!"quantout {K} {vname} sf0 {osexpr (Quant.specSf v nb kb qrow 0)}"
    IO.println s!"quantout {K} {vname} sfL {osexpr (Quant.specSf v nb kb qrow (nb - 1))}"
    IO.println s!"quantout {K} {vname} code0 {osexpr (Quant.specCode v nb kb qrow 0)}"
    IO.println s!"quantout {K} {vname} codeL {osexpr (Quant.specCode v nb kb qrow (K - 1))}"
    IO.println s!"quantout {K} {vname} pts {if v.perToken then osexpr (Quant.specPts nb kb qrow) else "absent"}"
  for (b, name) in quantBlockCases do
    IO.println s!"quantblock {b} {name} {osexpr (Quant.blockAmax qrow b)}"
  -- the MX classes (MXQuant.lean): the block placements' kernel terms, the chains per format and rule, the block amaxes
  for (name, ps) in mxqCases do
    IO.println s!"mxq {name} spec {osexpr (MXQuant.blockAmax qrow 0)}"
    IO.println s!"mxq {name} kernel {osexpr (execC .fmaxc (flatten ps) (MXQuant.blockLeaf qrow 0) (zeroCoords (flatten ps)))}"
  for (fname, rname, f, r) in mxqOutCases do
    IO.println s!"mxqout {fname} {rname} sf0 {osexpr (MXQuant.specSf f r qrow 0)}"
    IO.println s!"mxqout {fname} {rname} sf1 {osexpr (MXQuant.specSf f r qrow 1)}"
    IO.println s!"mxqout {fname} {rname} code0 {osexpr (MXQuant.specCode f r qrow 0)}"
    IO.println s!"mxqout {fname} {rname} codeL {osexpr (MXQuant.specCode f r qrow 63)}"
  for (b, name) in mxqBlockCases do
    IO.println s!"mxqblock {b} {name} {osexpr (MXQuant.blockAmax qrow b)}"
