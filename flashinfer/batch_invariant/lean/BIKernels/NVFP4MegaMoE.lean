-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.MegaMoE

/-!
# NVFP4 MegaMoE's ordered numerical components

K64 and K128 block-scaled instructions are distinct opaque atoms. They are
not BF16 K16 folds. The first instruction initializes without reading D;
subsequent instructions preserve the persistent accumulator. Full physical
stages and padded instructions remain explicit. The input staging recipe
produces the packed-values/scales boundary; prequantized calls instead supply
that boundary. A quantization block never reads a different logical row.

Each architecture/provider fixes its own atom interpretation. These trace
proofs do not establish CUDA lowering, hardware equivalence, synchronization,
or remote-write completion. Publication assumptions are exactly those of the
shared placement model, and no theorem asserts cross-architecture bits.
-/
namespace BIK.NVFP4MegaMoE

open BIK.MegaMoE (Expr f32 zero cvt finalizer)

inductive Arch where | sm100 | sm103 | sm107
deriving Repr, BEq, DecidableEq
inductive Provider where | legacy | standard | genphase
deriving Repr, BEq, DecidableEq

structure Class where
  instructionK : Nat
  stageK : Nat
deriving Repr, BEq

def numericalClass (arch : Arch) (provider : Provider) : Class :=
  let instructionK := if arch == .sm107 then 128 else 64
  let stageK := if provider == .legacy || provider == .genphase || arch == .sm107 then 256 else 128
  ⟨instructionK, stageK⟩

def suffix (c : Class) : String :=
  s!"nvfp4.k{c.instructionK}.m256n128.cta2"

def operandSlice (c : Class) (K : Nat) (operand : String) (step : Nat) : Expr :=
  let valid : Nat := min c.instructionK (K - step * c.instructionK)
  .op s!"operand.nvfp4.k{c.instructionK}"
    [.str operand, .int step, .int valid,
     .str "e2m1.e4m3.block16"]

def mmaStep (c : Class) (K : Nat) (operand : String)
    (acc : Option Expr) (step : Nat) : Option Expr :=
  let operands := operandSlice c K operand step
  some (match acc with
    | none => .op ("mma.init." ++ suffix c) [operands]
    | some old => .op ("mma.acc." ++ suffix c) [old, operands])

def stages (c : Class) (K : Nat) : Nat := (K + c.stageK - 1) / c.stageK
def perStage (c : Class) : Nat := c.stageK / c.instructionK

def mmaFlat (c : Class) (K : Nat) (operand : String) : Expr :=
  ((List.range (stages c K * perStage c)).foldl (mmaStep c K operand) none).getD zero

def mmaBlocked (c : Class) (K : Nat) (operand : String) : Expr :=
  ((BIK.MegaMoE.tiledIndices (stages c K) (perStage c)).foldl
    (fun acc tile => tile.foldl (mmaStep c K operand) acc) none).getD zero

/-- Regrouping instructions preserves the exact fold, including its init atom. -/
theorem blocked_mma_eq_flat (c : Class) (K : Nat) (operand : String) :
    mmaBlocked c K operand = mmaFlat c K operand := by
  unfold mmaBlocked mmaFlat
  rw [← List.foldl_flatten, BIK.MegaMoE.tiled_indices_flat]

theorem legacy_k64 : (numericalClass .sm100 .legacy).instructionK = 64 := rfl
theorem rubin_k128 : (numericalClass .sm107 .standard).instructionK = 128 := rfl
theorem instruction_classes_distinct :
    suffix (numericalClass .sm100 .legacy) ≠ suffix (numericalClass .sm107 .standard) := by
  decide

def mul (a b : Expr) : Expr := .op "mul.rn.f32" [a, b]
def input (name : String) : Expr := .op "input.f32" [.str name]
def ref (name : String) : Expr := .op "ref.f32" [.str name]

def activation (early hasAlpha : Bool) (clamp : Option String) : Expr :=
  let gate := ref "fc1_gate"
  let up := ref "fc1_up"
  let (gate, up) := if hasAlpha then
      (mul gate (input "fc1_alpha"), mul up (input "fc1_alpha")) else (gate, up)
  let (gate, up) := match clamp with
    | none => (gate, up)
    | some bits => (.op "min.f32" [gate, f32 bits],
                    .op "min.xorsign.abs.f32" [up, f32 bits])
  let exponent := mul gate (f32 "bfb8aa3b")
  let sigmoid := Expr.op "rcp.approx.ftz.f32"
    [.op "add.rn.f32" [.op "ex2.approx.ftz.f32" [exponent], f32 "3f800000"]]
  let value := mul (mul up gate) sigmoid
  if early then mul value (input "route_weight") else value

def quantAmax (values : List Expr) : Expr :=
  .op "abs.bits.f32"
    [values.tail.foldl (fun acc x => .op "max.xorsign.abs.f32" [acc, x])
      (values.headD zero)]

def inputAmax (values : List Expr) : Expr :=
  values.foldl (fun acc x => .op "max.f32" [acc, .op "max.f32" [x, .op "neg.f32" [x]]]) zero

def quantizationScale (amax : Expr) (norm : Option Expr) : Expr :=
  let candidate := mul amax (f32 "3e2aaaab")
  let candidate := match norm with | none => candidate | some n => mul candidate n
  .op "cvt.rn.satfinite.e4m3.f32" [candidate]

def quantizationEncodingScale (scale : Expr) (norm : Option Expr) : Expr :=
  let scaleRT := Expr.op "widen.e4m3.f32" [scale]
  let reciprocal := Expr.op "rcp.approx.ftz.f32" [scaleRT]
  let reciprocal := match norm with | none => reciprocal | some n => mul n reciprocal
  let reciprocal := Expr.op "min.f32" [reciprocal, f32 "7f7fffff"]
  let mask := Expr.op "min.f32" [mul scaleRT (f32 "7149f2ca"), f32 "3f800000"]
  mul reciprocal mask

def quantizedData (values : List Expr) (enc : Expr) : Expr :=
  .op "block.e2m1" (values.map (fun value => .op "cvt.rn.satfinite.e2m1.f32" [mul value enc]))

def quantizeWithAmax (values : List Expr) (amax : Expr) (norm : Option Expr) : Expr × Expr :=
  let scale := quantizationScale amax norm
  let enc := quantizationEncodingScale scale norm
  (quantizedData values enc, scale)

def quantize (values : List Expr) (norm : Option Expr) : Expr × Expr :=
  quantizeWithAmax values (quantAmax values) norm

def inputQuantize (values : List Expr) (norm : Expr) : Expr × Expr :=
  quantizeWithAmax values (inputAmax values) (some norm)

def fc2Output (acc : Expr) (hasAlpha : Bool) : Expr :=
  cvt (if hasAlpha then mul acc (input "fc2_alpha") else acc)

structure Params where
  hidden : Nat
  intermediate : Nat
  slots : Nat
  experts : Nat
  arch : Arch
  provider : Provider
  early : Bool
  clamp : Option String
  inputNorm : String
  hasFc1Alpha : Bool
  hasFc2Alpha : Bool
  hasFc1Norm : Bool
  mask : Option (List Bool) := none

structure Components where
  inputAmax : Expr
  inputData : Expr
  inputScale : Expr
  inputEncodingScale : Expr
  fc1Gate : Expr
  fc1Up : Expr
  act : Expr
  handoffData : Expr
  handoffAmax : Expr
  handoffScale : Expr
  handoffEncodingScale : Expr
  fc2 : Expr
  finalize : Expr

def components (p : Params) (blocked : Bool) : Components :=
  let c := numericalClass p.arch p.provider
  let mma := if blocked then mmaBlocked c else mmaFlat c
  let inputValues := (List.range 16).map (fun i =>
    Expr.op "widen.bf16.f32" [.op "input.bf16" [.str "row.block16", .int (Int.ofNat i)]])
  let inputNorm := some (f32 p.inputNorm)
  let inputScale := quantizationScale (ref "input_amax") inputNorm
  let inputEncodingScale := quantizationEncodingScale (.op "ref.e4m3" [.str "input_scale"]) inputNorm
  let inputData := quantizedData inputValues (ref "input_encoding_scale")
  let values := (List.range 16).map (fun i => Expr.op "ref.f32" [.str "activation", .int (Int.ofNat i)])
  let norm := if p.hasFc1Norm then some (input "fc1_norm_const") else none
  let handoffScale := quantizationScale (ref "handoff_amax") norm
  let handoffEncodingScale := quantizationEncodingScale (.op "ref.e4m3" [.str "handoff_scale"]) norm
  let handoffData := quantizedData values (ref "handoff_encoding_scale")
  ⟨inputAmax inputValues, inputData, inputScale, inputEncodingScale,
   mma p.hidden "input_w13.gate", mma p.hidden "input_w13.up",
   activation p.early p.hasFc1Alpha p.clamp,
   handoffData, quantAmax values, handoffScale, handoffEncodingScale,
   fc2Output (mma p.intermediate "handoff_w2") p.hasFc2Alpha,
   finalizer p.slots p.experts p.early p.mask⟩

def componentPairs (c : Components) : List (String × Expr) :=
  [("input_amax", c.inputAmax), ("input_scale", c.inputScale),
   ("input_encoding_scale", c.inputEncodingScale), ("input_data", c.inputData),
   ("fc1_gate", c.fc1Gate), ("fc1_up", c.fc1Up), ("activation", c.act),
   ("handoff_amax", c.handoffAmax), ("handoff_scale", c.handoffScale),
   ("handoff_encoding_scale", c.handoffEncodingScale), ("handoff_data", c.handoffData),
   ("fc2", c.fc2), ("finalize", c.finalize)]

/-- All components agree without treating zero padding as an algebraic identity. -/
theorem components_blocked_eq_flat (p : Params) :
    components p true = components p false := by
  simp only [components, Bool.true_eq, ↓reduceIte, Bool.false_eq_true, blocked_mma_eq_flat]

/-- Schedulers/carriers may relocate cells only through the inverse publication
map. Exactly-once device writes and completion remain external assumptions. -/
theorem placement_preserves_nvfp4_output (p q : BIK.MegaMoE.Placement)
    (logical : BIK.MegaMoE.Route → Expr)
    (hp : BIK.MegaMoE.Inverse p) (hq : BIK.MegaMoE.Inverse q)
    (ep : BIK.MegaMoE.ExactlyOnce p) (eq : BIK.MegaMoE.ExactlyOnce q)
    (token slots : Nat) (step : Expr → Nat → Expr → Expr) :
    BIK.MegaMoE.placedFold p logical token slots step =
      BIK.MegaMoE.placedFold q logical token slots step := by
  exact BIK.MegaMoE.token_independent_of_batch p q logical hp hq ep eq token slots step

end BIK.NVFP4MegaMoE
