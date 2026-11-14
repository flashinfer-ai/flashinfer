-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.NumericalVariants

/-!
# A16 MegaMoE's decoded-weight numerical trace

Weights are decoded and multiplied by their block scale in BF16 before the
dense K16 MMA. This is a distinct atom family from native block-scaled MMA.
The M256/N128/two-CTA instruction, operand roles, initialization and complete
physical K padding are fixed. Architecture selects the interpretation of the
opaque hardware atoms; identical syntax does not assert cross-architecture bits.

The independent stage lowering below retains every padded instruction, rather
than assuming that a zero operand makes an MMA an identity. This model proves
ordered arithmetic traces and conditional placement, not CUDA compiler lowering,
native BF16 instruction semantics, barriers, races or remote-write completion.
-/
namespace BIK.A16MegaMoE

open BIK.MegaMoE (Expr zero cvt finalizer)
open BIK.NumericalVariants (Variant)

inductive Kind where
  | w4a16 | w8a16e4m3 | w8a16e5m2
deriving Repr, BEq, DecidableEq

def stageK (kind : Kind) : Nat := if kind == .w4a16 then 256 else 128
def perStage (kind : Kind) : Nat := stageK kind / 16
def stages (kind : Kind) (K : Nat) : Nat := (K + stageK kind - 1) / stageK kind

def weightDecode (kind : Kind) : Expr :=
  let data := match kind with
    | .w4a16 => "e2m1" | .w8a16e4m3 => "e4m3" | .w8a16e5m2 => "e5m2"
  let scale := if kind == .w4a16 then "e4m3" else "ue8m0"
  .op "mul.rn.bf16"
    [.op ("decode." ++ data ++ ".bf16") [.op "input.weight_bits" []],
     .op ("decode." ++ scale ++ ".bf16") [.op "input.weight_scale_bits" []]]

def operandSlice (K : Nat) (operand : String) (step : Nat) : Expr :=
  -- Fix subtraction in Nat before coercing the valid-count operand to Int.
  -- Contextual Int elaboration would otherwise emit negative padded counts.
  let valid : Nat := min 16 (K - step * 16)
  .op "operand.k16" [.str operand, .int step, .int valid]

def mmaStep (K : Nat) (operand : String) (acc : Option Expr) (step : Nat) : Option Expr :=
  let operands := operandSlice K operand step
  some (match acc with
    | none => .op "mma.bf16.k16.m256n128.cta2.init" [operands]
    | some old => .op "mma.bf16.k16.m256n128.cta2.acc" [old, operands])

def mmaFlat (kind : Kind) (K : Nat) (operand : String) : Expr :=
  ((List.range (stages kind K * perStage kind)).foldl (mmaStep K operand) none).getD zero

def mmaBlocked (kind : Kind) (K : Nat) (operand : String) : Expr :=
  ((BIK.MegaMoE.tiledIndices (stages kind K) (perStage kind)).foldl
    (fun acc tile => tile.foldl (mmaStep K operand) acc) none).getD zero

/-- The first physical K16 instruction initializes independently of D. -/
theorem first_instruction_initializes (K : Nat) (operand : String) (step : Nat) :
    mmaStep K operand none step =
      some (.op "mma.bf16.k16.m256n128.cta2.init" [operandSlice K operand step]) := rfl

/-- Regrouping K16 instructions into physical stages preserves their exact order. -/
theorem blocked_mma_eq_flat (kind : Kind) (K : Nat) (operand : String) :
    mmaBlocked kind K operand = mmaFlat kind K operand := by
  unfold mmaBlocked mmaFlat
  rw [← List.foldl_flatten, BIK.MegaMoE.tiled_indices_flat]

def mul (a b : Expr) : Expr := .op "mul.rn.f32" [a, b]
def input (name : String) : Expr := .op "input.f32" [.str name]
def ref (name : String) : Expr := .op "ref.f32" [.str name]

structure Params where
  hidden : Nat
  intermediate : Nat
  slots : Nat
  experts : Nat
  kind : Kind
  early : Bool
  variant : Variant
  norm : Bool
  deferred : Bool

structure Components where
  weight : Expr
  fc1Gate : Expr
  fc1Up : Expr
  act : Expr
  fc2 : Expr
  finalize : Expr
  partials : Expr
  deferredWeight : Expr

def components (p : Params) (blocked : Bool) : Components :=
  let mma := if blocked then mmaBlocked p.kind else mmaFlat p.kind
  let gate := mma p.hidden "fc1.gate"
  let up := mma p.hidden "fc1.up"
  let gate := if p.kind == .w4a16 then mul gate (input "fc1_alpha") else gate
  let up := if p.kind == .w4a16 then mul up (input "fc1_alpha") else up
  let act := BIK.NumericalVariants.activation p.variant
    (ref "fc1_gate") (ref "fc1_up") none
    (if p.early then some (input "route_weight") else none)
    (if p.norm then some (input "handoff_norm") else none)
  let fc2 := mma p.intermediate "fc2.handoff_decoded_w2"
  let fc2 := if p.kind == .w4a16 then mul fc2 (input "fc2_alpha") else fc2
  let valid := Expr.op "route.valid" [.op "input.expert_id" [], .int p.experts]
  let partials := Expr.op "select.lazy"
    [valid, .op "ref.bf16" [.str "fc2"], .op "bf16_bits" [.str "0000"]]
  let deferredWeight := Expr.op "select.lazy"
    [valid, if p.early then BIK.MegaMoE.f32 "3f800000" else input "route_weight", zero]
  ⟨weightDecode p.kind, gate, up, cvt act, cvt fc2,
   finalizer p.slots p.experts p.early none, partials, deferredWeight⟩

def componentPairs (p : Params) (c : Components) : List (String × Expr) :=
  [("weight_decode", c.weight), ("fc1_gate", c.fc1Gate), ("fc1_up", c.fc1Up),
   ("activation", c.act), ("fc2", c.fc2)] ++
    (if p.deferred then [("partials", c.partials), ("deferred_weight", c.deferredWeight)]
     else [("finalize", c.finalize)])

/-- Every arithmetic component is independently reconstructed for both lowerings. -/
theorem components_blocked_eq_flat (p : Params) :
    components p true = components p false := by
  simp only [components, Bool.true_eq, ↓reduceIte, Bool.false_eq_true, blocked_mma_eq_flat]

/-- Native decoded-weight formats are distinct numerical classes. -/
theorem weight_classes_distinct :
    BIK.MegaMoE.sexpr (weightDecode .w4a16) ≠
      BIK.MegaMoE.sexpr (weightDecode .w8a16e4m3) := by
  decide

/-- Schedulers/carriers preserve route identity only under inverse, unique-cell
placement and correct publication assumptions. Device completion is external. -/
theorem placement_preserves_a16_output (p q : BIK.MegaMoE.Placement)
    (logical : BIK.MegaMoE.Route → Expr)
    (hp : BIK.MegaMoE.Inverse p) (hq : BIK.MegaMoE.Inverse q)
    (ep : BIK.MegaMoE.ExactlyOnce p) (eq : BIK.MegaMoE.ExactlyOnce q)
    (token slots : Nat) (step : Expr → Nat → Expr → Expr) :
    BIK.MegaMoE.placedFold p logical token slots step =
      BIK.MegaMoE.placedFold q logical token slots step := by
  exact BIK.MegaMoE.token_independent_of_batch p q logical hp hq ep eq token slots step

end BIK.A16MegaMoE
