-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.NVFP4MegaMoE

/-!
# MXFP4 MegaMoE's ordered arithmetic

Mixed E2M1-weight/E4M3-activation K32/K64 instructions and Rubin pure-E2M1
K128 instructions are separate opaque atoms. The latter uses the emitter's
actual `mxf4nvf4.block32` spelling. Every instruction's initialization,
association and padded slice remain explicit. The input-stage and FC1
quantizers have distinct amax, normal-scale-floor and soft-mask recipes.

UE8M0 conversion, BF16 roundtrip, approximate reciprocal and MMA are opaque,
architecture-scoped operations. No theorem proves their hardware meaning,
CUDA lowering, race freedom or remote publication, nor compares architectures.
Schedulers and carriers rely on the shared inverse-publication assumptions.
-/
namespace BIK.MXFP4MegaMoE

open BIK.MegaMoE (Expr f32 zero cvt finalizer)
open BIK.NVFP4MegaMoE (Arch mul ref)

inductive Kind where | pure | mixed
deriving Repr, BEq, DecidableEq
inductive Provider where | standard | genphase
deriving Repr, BEq, DecidableEq

structure Class where
  kind : Kind
  instructionK : Nat
  stageK : Nat
deriving Repr, BEq

def numericalClass (arch : Arch) (kind : Kind) : Class :=
  if kind == .pure then ⟨kind, 128, 256⟩
  else ⟨kind, if arch == .sm107 then 64 else 32, 128⟩

def kindName : Kind → String
  | .pure => "mxfp4"
  | .mixed => "mxfp4_mxfp8"

def dataFormat : Kind → String
  | .pure => "e2m1"
  | .mixed => "e4m3"

def instructionKind : Kind → String
  | .pure => "mxf4nvf4.block32"
  | .mixed => "mxf8f6f4"

def suffix (c : Class) : String :=
  s!"{kindName c.kind}.{instructionKind c.kind}.k{c.instructionK}.m256n128.cta2"

def operandSlice (c : Class) (K : Nat) (operand : String) (step : Nat) : Expr :=
  let valid : Nat := min c.instructionK (K - step * c.instructionK)
  .op s!"operand.{kindName c.kind}.k{c.instructionK}"
    [.str operand, .int step, .int valid, .str "weight.e2m1.ue8m0.block32",
     .str s!"activation.{dataFormat c.kind}.ue8m0.block32"]

def mmaStep (c : Class) (K : Nat) (operand : String) (acc : Option Expr) (step : Nat) : Option Expr :=
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

theorem blocked_mma_eq_flat (c : Class) (K : Nat) (operand : String) :
    mmaBlocked c K operand = mmaFlat c K operand := by
  unfold mmaBlocked mmaFlat
  rw [← List.foldl_flatten, BIK.MegaMoE.tiled_indices_flat]

theorem blackwell_mixed_k32 : (numericalClass .sm100 .mixed).instructionK = 32 := rfl
theorem rubin_mixed_k64 : (numericalClass .sm107 .mixed).instructionK = 64 := rfl
theorem rubin_pure_k128 : (numericalClass .sm107 .pure).instructionK = 128 := rfl

def halfAmax (values : List Expr) : Expr := BIK.NVFP4MegaMoE.quantAmax values
def inputHalfAmax (values : List Expr) : Expr := BIK.NVFP4MegaMoE.inputAmax values
def combineHalfAmax (low high : Expr) : Expr := .op "max.f32" [low, high]

def blockAmax (values : List Expr) : Expr :=
  combineHalfAmax (halfAmax (values.take 16)) (halfAmax (values.drop 16))

def inputBlockAmax (values : List Expr) (kind : Kind) : Expr :=
  if kind == .pure then inputHalfAmax values
  else combineHalfAmax (inputHalfAmax (values.take 16)) (inputHalfAmax (values.drop 16))

def scaleValue (amax : Expr) (kind : Kind) (staging : Bool) : Expr :=
  let candidate := mul amax (f32 (if kind == .pure then "3e2aaaab" else "3b124925"))
  let candidate := if staging then Expr.op "max.f32" [candidate, f32 "00800000"] else candidate
  .op "widen.bf16.f32"
    [.op "cvt.rn.bf16.ue8m0" [.op "cvt.rp.satfinite.ue8m0.f32" [candidate]]]

def encodingScale (scale : Expr) (staging : Bool) : Expr :=
  let enc := Expr.op "min.f32" [.op "rcp.approx.ftz.f32" [scale], f32 "7f7fffff"]
  if staging then enc
  else mul enc (.op "min.f32" [mul scale (f32 "7149f2ca"), f32 "3f800000"])

structure QuantComponents where
  amax : Expr
  scaleValue : Expr
  scale : Expr
  encoding : Expr
  data : Expr

def quantComponents (name : String) (values : List Expr) (amax : Expr)
    (kind : Kind) (staging : Bool) : QuantComponents :=
  let scaleRef := ref (name ++ "_scale_value")
  let fmt := dataFormat kind
  ⟨amax, scaleValue (ref (name ++ "_amax")) kind staging,
   .op "cvt.rp.satfinite.ue8m0.f32" [scaleRef], encodingScale scaleRef staging,
   .op ("block." ++ fmt) (values.map (fun value =>
     .op ("cvt.rn.satfinite." ++ fmt ++ ".f32")
       [mul value (ref (name ++ "_encoding_scale"))]))⟩

structure Params where
  hidden : Nat
  intermediate : Nat
  slots : Nat
  experts : Nat
  arch : Arch
  kind : Kind
  provider : Provider
  early : Bool
  clamp : Option String
  mask : Option (List Bool) := none

structure Components where
  inputQ : QuantComponents
  fc1Gate : Expr
  fc1Up : Expr
  act : Expr
  handoffQ : QuantComponents
  fc2 : Expr
  finalize : Expr

def components (p : Params) (blocked : Bool) : Components :=
  let c := numericalClass p.arch p.kind
  let mma := if blocked then mmaBlocked c else mmaFlat c
  let inputs := (List.range 32).map (fun i =>
    Expr.op "widen.bf16.f32" [.op "input.bf16" [.str "row.block32", .int (Int.ofNat i)]])
  let values := (List.range 32).map (fun i => Expr.op "ref.f32" [.str "activation", .int (Int.ofNat i)])
  ⟨quantComponents "input" inputs (inputBlockAmax inputs p.kind) p.kind true,
   mma p.hidden "input_w13.gate", mma p.hidden "input_w13.up",
   BIK.NVFP4MegaMoE.activation p.early false p.clamp,
   quantComponents "handoff" values (blockAmax values) p.kind false,
   cvt (mma p.intermediate "handoff_w2"),
   finalizer p.slots p.experts p.early p.mask⟩

def quantPairs (name : String) (c : QuantComponents) : List (String × Expr) :=
  [(name ++ "_amax", c.amax), (name ++ "_scale_value", c.scaleValue),
   (name ++ "_scale", c.scale), (name ++ "_encoding_scale", c.encoding),
   (name ++ "_data", c.data)]

def componentPairs (c : Components) : List (String × Expr) :=
  quantPairs "input" c.inputQ ++
  [("fc1_gate", c.fc1Gate), ("fc1_up", c.fc1Up), ("activation", c.act)] ++
  quantPairs "handoff" c.handoffQ ++ [("fc2", c.fc2), ("finalize", c.finalize)]

theorem components_blocked_eq_flat (p : Params) : components p true = components p false := by
  simp only [components, Bool.true_eq, ↓reduceIte, Bool.false_eq_true, blocked_mma_eq_flat]

/-- Hardware ownership may change only through the shared inverse publication
contract. This statement establishes no CUDA synchronization property. -/
theorem placement_preserves_mxfp4_output (p q : BIK.MegaMoE.Placement)
    (logical : BIK.MegaMoE.Route → Expr)
    (hp : BIK.MegaMoE.Inverse p) (hq : BIK.MegaMoE.Inverse q)
    (ep : BIK.MegaMoE.ExactlyOnce p) (eq : BIK.MegaMoE.ExactlyOnce q)
    (token slots : Nat) (step : Expr → Nat → Expr → Expr) :
    BIK.MegaMoE.placedFold p logical token slots step =
      BIK.MegaMoE.placedFold q logical token slots step := by
  exact BIK.MegaMoE.token_independent_of_batch p q logical hp hq ep eq token slots step

end BIK.MXFP4MegaMoE
