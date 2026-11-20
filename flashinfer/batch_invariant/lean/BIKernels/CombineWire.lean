-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.MXFP8MegaMoE

/-! Ordered return-wire quantization and lazy slot reduction. The legacy MXFP8
producer starts at FP32; every other producer starts at rounded BF16. The two
producer layouts must restore logical hidden order before this model applies.
Device lowering, publication and code decoding remain separate conformance
obligations; opaque approximate/conversion primitives are not rewritten. -/
namespace BIK.CombineWire
open BIK.MegaMoE (Expr f32 zero cvt)
open BIK.NVFP4MegaMoE (mul ref)

inductive Kind where | nvfp4 | mxfp8 deriving Repr, BEq, DecidableEq

def source (kind : Kind) (legacy : Bool) : List Expr :=
  (List.range (if kind == .nvfp4 then 16 else 32)).map (fun (i : Nat) =>
    if legacy then .op "ref.f32" [.str "fc2", .int i]
    else .op "widen.bf16.f32" [.op "ref.bf16" [.str "fc2", .int i]])

def decodeScale (kind : Kind) : Expr :=
  if kind == .nvfp4 then mul (.op "widen.bf16.f32" [.op "ref.bf16" [.str "combine_scale"]]) (f32 "3e2aaaab")
  else .op "widen.bf16.f32" [.op "cvt.rn.bf16.ue8m0" [.op "ref.ue8m0" [.str "combine_scale"]]]

def componentPairs (kind : Kind) (legacy : Bool) : List (String × Expr) :=
  let values := source kind legacy
  let amax := if legacy then BIK.MXFP8MegaMoE.legacyAmax values else BIK.NVFP4MegaMoE.quantAmax values
  let maximum := ref "combine_amax"
  let scaleValue := if kind == .nvfp4 then decodeScale kind else
    BIK.MXFP8MegaMoE.scaleValue maximum .e4m3 legacy false
  let scale := if kind == .nvfp4 then cvt maximum else
    .op "cvt.rp.satfinite.ue8m0.f32" [ref "combine_scale_value"]
  let encoding := BIK.MXFP8MegaMoE.encodingScale (ref "combine_scale_value") legacy false
  let fmt := if kind == .nvfp4 then "e2m1" else "e4m3"
  let data := Expr.op ("block." ++ fmt) (values.map (fun value =>
    .op ("cvt.rn.satfinite." ++ fmt ++ ".f32") [mul value (ref "combine_encoding_scale")]))
  let decoded := mul (.op ("widen." ++ fmt ++ ".f32") [.op ("ref." ++ fmt) [.str "combine_data"]]) (decodeScale kind)
  [("combine_amax", amax), ("combine_scale", scale),
   ("combine_scale_value", scaleValue), ("combine_encoding_scale", encoding),
   ("combine_data", data), ("combine_decode", decoded)]

def slot (experts index : Nat) (mask : Option (List Bool)) : Expr × Expr :=
  let value := Expr.op "ref.f32" [.str "combine_decode", .int index]
  let score := Expr.op "input.f32" [.str "route_weight", .int index]
  match mask with
  | some bits => if (bits[index]?).getD false then (value, score) else (zero, zero)
  | none =>
    let valid := Expr.op "route.valid" [.op "input.expert_id" [.int index], .int experts]
    (.op "select.lazy" [valid, value, zero], .op "select.lazy" [valid, score, zero])

def finalize (slots experts : Nat) (early : Bool) (mask : Option (List Bool)) : Expr :=
  cvt ((List.range slots).foldl (fun acc index =>
    let (value, score) := slot experts index mask
    if early then .op "add.rn.f32" [acc,value] else .op "fma.rn.f32" [value,score,acc]) zero)

def partialSlot (experts index : Nat) (mask : Option (List Bool)) : Expr :=
  let value := Expr.op "ref.bf16" [.str "fc2", .int index]
  let z := Expr.op "bf16_bits" [.str "0000"]
  match mask with
  | some bits => if (bits[index]?).getD false then value else z
  | none => .op "select.lazy" [.op "route.valid" [.op "input.expert_id" [.int index], .int experts], value, z]

def partials (slots experts : Nat) (mask : Option (List Bool)) : Expr :=
  .op "block.bf16" ((List.range slots).map (partialSlot experts · mask))

def deferredWeight (experts index : Nat) (early : Bool) (mask : Option (List Bool)) : Expr :=
  let value := if early then f32 "3f800000" else .op "input.f32" [.str "route_weight", .int index]
  match mask with
  | some bits => if (bits[index]?).getD false then value else zero
  | none => .op "select.lazy" [.op "route.valid" [.op "input.expert_id" [.int index], .int experts], value, zero]

def deferredWeights (slots experts : Nat) (early : Bool) (mask : Option (List Bool)) : Expr :=
  .op "block.f32" ((List.range slots).map (fun index => deferredWeight experts index early mask))

/-- The wrapper clears physical staging separately; this states the observed
masked value and never uses a stale slot as an arithmetic operand. -/
theorem invalid_partial_zero (slot experts : Nat) :
    partialSlot experts slot (some []) = .op "bf16_bits" [.str "0000"] := by
  simp [partialSlot]

theorem placement_preserves_quantized_output (p q : BIK.MegaMoE.Placement)
    (logical : BIK.MegaMoE.Route → Expr)
    (hp : BIK.MegaMoE.Inverse p) (hq : BIK.MegaMoE.Inverse q)
    (ep : BIK.MegaMoE.ExactlyOnce p) (eq : BIK.MegaMoE.ExactlyOnce q)
    (token slots : Nat) (step : Expr → Nat → Expr → Expr) :
    BIK.MegaMoE.placedFold p logical token slots step =
    BIK.MegaMoE.placedFold q logical token slots step := by
  exact BIK.MegaMoE.token_independent_of_batch p q logical hp hq ep eq token slots step
end BIK.CombineWire
