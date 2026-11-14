-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.MXFP4MegaMoE

/-!
# W8A8 MXFP8 MegaMoE ordered traces

Legacy retains its separate operand orientation, NaN clamp, sigmoid-first
multiplication association, MAXIMUMF tree and two fixed-one multiplications.
Next reuses the paired-half MX recipe. All BI input stages use the normal
scale floor. Each architecture/provider interprets its own opaque K32/K64
atoms. These proofs do not establish CUDA lowering, race freedom, actual
publication, or cross-architecture equality of bits.
-/
namespace BIK.MXFP8MegaMoE

open BIK.MegaMoE (Expr f32 zero cvt finalizer)
open BIK.NVFP4MegaMoE (Arch mul ref)
open BIK.MXFP4MegaMoE (QuantComponents Components)

inductive Kind where | e4m3 | e5m2
deriving Repr, BEq, DecidableEq
inductive Provider where | legacy | standard | genphase
deriving Repr, BEq, DecidableEq

def kindName : Kind → String
  | .e4m3 => "mxfp8_e4m3"
  | .e5m2 => "mxfp8_e5m2"
def dataFormat : Kind → String
  | .e4m3 => "e4m3"
  | .e5m2 => "e5m2"

structure Class where
  kind : Kind
  instructionK : Nat
  tileN : Nat
  legacy : Bool

def numericalClass (arch : Arch) (kind : Kind) (provider : Provider) : Class :=
  ⟨kind, if arch == .sm107 then 64 else 32,
   if provider == .legacy then 256 else 128, provider == .legacy⟩

def suffix (c : Class) : String :=
  s!"{kindName c.kind}.mxf8f6f4.k{c.instructionK}.m256n{c.tileN}.cta2"

def operandSlice (c : Class) (K : Nat) (operand : String) (step : Nat) : Expr :=
  let valid : Nat := min c.instructionK (K - step * c.instructionK)
  let order := if c.legacy && operand.startsWith "input_w13" then
      "A.activation_B.weight" else "A.weight_B.activation"
  .op s!"operand.{kindName c.kind}.k{c.instructionK}"
    [.str operand, .int step, .int valid, .str order,
     .str s!"{dataFormat c.kind}.ue8m0.block32"]

def mmaStep (c : Class) (K : Nat) (operand : String) (acc : Option Expr) (step : Nat) : Option Expr :=
  let operands := operandSlice c K operand step
  some (match acc with
    | none => .op ("mma.init." ++ suffix c) [operands]
    | some old => .op ("mma.acc." ++ suffix c) [old, operands])

def stages (K : Nat) : Nat := (K + 127) / 128
def perStage (c : Class) : Nat := 128 / c.instructionK

def mmaFlat (c : Class) (K : Nat) (operand : String) : Expr :=
  ((List.range (stages K * perStage c)).foldl (mmaStep c K operand) none).getD zero

def mmaBlocked (c : Class) (K : Nat) (operand : String) : Expr :=
  ((BIK.MegaMoE.tiledIndices (stages K) (perStage c)).foldl
    (fun acc tile => tile.foldl (mmaStep c K operand) acc) none).getD zero

theorem blocked_mma_eq_flat (c : Class) (K : Nat) (operand : String) :
    mmaBlocked c K operand = mmaFlat c K operand := by
  unfold mmaBlocked mmaFlat
  rw [← List.foldl_flatten, BIK.MegaMoE.tiled_indices_flat]

def legacyActivation (early : Bool) (clamp : Option (String × String)) : Expr :=
  let gate := ref "fc1_gate"
  let up := ref "fc1_up"
  let (gate, up) := match clamp with
    | none => (gate, up)
    | some (pos, neg) => (.op "min.NaN.f32" [gate, f32 pos],
        .op "max.NaN.f32" [.op "min.NaN.f32" [up, f32 pos], f32 neg])
  let exponent := mul gate (f32 "bfb8aa3b")
  let sigmoid := Expr.op "rcp.approx.ftz.f32"
    [.op "add.rn.f32" [.op "ex2.approx.ftz.f32" [exponent], f32 "3f800000"]]
  let value := mul (mul sigmoid gate) up
  if early then mul value (.op "input.f32" [.str "route_weight"]) else value

def maxLevel (values : List Expr) : List Expr :=
  (List.range (values.length / 2)).map (fun i => .op "max.NaN.f32"
    [(values[i * 2]?).getD zero, (values[i * 2 + 1]?).getD zero])

def legacyAmax (values : List Expr) : Expr :=
  let magnitudes := values.map (fun value => Expr.op "abs.bits.f32" [value])
  let reduced := (List.range 5).foldl (fun level _ => maxLevel level) magnitudes
  .op "max.NaN.f32" [zero, reduced.headD zero]

def scaleValue (amax : Expr) (kind : Kind) (legacy staging : Bool) : Expr :=
  let candidate := mul amax (f32 (if kind == .e4m3 then "3b124925" else "37924925"))
  let candidate := if staging then Expr.op "max.f32" [candidate, f32 "00800000"]
                   else if legacy then mul candidate (f32 "3f800000") else candidate
  .op "widen.bf16.f32"
    [.op "cvt.rn.bf16.ue8m0" [.op "cvt.rp.satfinite.ue8m0.f32" [candidate]]]

def encodingScale (scale : Expr) (legacy staging : Bool) : Expr :=
  if legacy && !staging then
    .op "min.NaN.f32" [mul (f32 "3f800000") (.op "rcp.approx.ftz.f32" [scale]), f32 "7f7fffff"]
  else BIK.MXFP4MegaMoE.encodingScale scale staging

def quantComponents (name : String) (values : List Expr) (amax : Expr)
    (kind : Kind) (legacy staging : Bool) : QuantComponents :=
  let scaleRef := ref (name ++ "_scale_value")
  let fmt := dataFormat kind
  ⟨amax, scaleValue (ref (name ++ "_amax")) kind legacy staging,
   .op "cvt.rp.satfinite.ue8m0.f32" [scaleRef], encodingScale scaleRef legacy staging,
   .op ("block." ++ fmt) (values.map (fun value =>
     .op ("cvt.rn.satfinite." ++ fmt ++ ".f32") [mul value (ref (name ++ "_encoding_scale"))]))⟩

structure Params where
  hidden : Nat
  intermediate : Nat
  slots : Nat
  experts : Nat
  arch : Arch
  kind : Kind
  provider : Provider
  early : Bool
  clamp : Option (String × String)
  mask : Option (List Bool) := none

def components (p : Params) (blocked : Bool) : Components :=
  let c := numericalClass p.arch p.kind p.provider
  let mma := if blocked then mmaBlocked c else mmaFlat c
  let inputs := (List.range 32).map (fun i =>
    Expr.op "widen.bf16.f32" [.op "input.bf16" [.str "row.block32", .int (Int.ofNat i)]])
  let values := (List.range 32).map (fun i => Expr.op "ref.f32" [.str "activation", .int (Int.ofNat i)])
  let legacy := p.provider == .legacy
  let activation := if legacy then legacyActivation p.early p.clamp
    else BIK.NVFP4MegaMoE.activation p.early false (p.clamp.map Prod.fst)
  ⟨quantComponents "input" inputs (BIK.MXFP4MegaMoE.inputBlockAmax inputs .mixed) p.kind false true,
   mma p.hidden "input_w13.gate", mma p.hidden "input_w13.up", activation,
   quantComponents "handoff" values
     (if legacy then legacyAmax values else BIK.MXFP4MegaMoE.blockAmax values) p.kind legacy false,
   cvt (mma p.intermediate "handoff_w2"), finalizer p.slots p.experts p.early p.mask⟩

def componentPairs := BIK.MXFP4MegaMoE.componentPairs

theorem components_blocked_eq_flat (p : Params) : components p true = components p false := by
  simp only [components, Bool.true_eq, ↓reduceIte, Bool.false_eq_true, blocked_mma_eq_flat]

theorem placement_preserves_mxfp8_output (p q : BIK.MegaMoE.Placement)
    (logical : BIK.MegaMoE.Route → Expr)
    (hp : BIK.MegaMoE.Inverse p) (hq : BIK.MegaMoE.Inverse q)
    (ep : BIK.MegaMoE.ExactlyOnce p) (eq : BIK.MegaMoE.ExactlyOnce q)
    (token slots : Nat) (step : Expr → Nat → Expr → Expr) :
    BIK.MegaMoE.placedFold p logical token slots step =
      BIK.MegaMoE.placedFold q logical token slots step := by
  exact BIK.MegaMoE.token_independent_of_batch p q logical hp hq ep eq token slots step

end BIK.MXFP8MegaMoE
