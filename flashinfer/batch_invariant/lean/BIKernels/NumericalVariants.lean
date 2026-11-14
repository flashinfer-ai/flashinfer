-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.MegaMoE

/-! Ordered activation interpretation. Coefficients are the actual emitted
FP32 bit strings, including host-folded constants. This module proves trace
identity for a fixed interpretation of approximate atoms; it does not prove
MUFU accuracy, CUDA lowering, memory safety or remote publication. -/
namespace BIK.NumericalVariants
open BIK.MegaMoE (Expr f32 zero)

structure Variant where
  family : String
  activation : String
  custom : Bool
  linear : Bool
  clamp : Bool
  coefficients : List (String × String)

def mul (a b : Expr) : Expr := .op "mul.rn.f32" [a,b]
def add (a b : Expr) : Expr := .op "add.rn.f32" [a,b]
def coefficient (v : Variant) (name : String) : Expr :=
  f32 (((v.coefficients.find? (fun pair => pair.1 == name)).map Prod.snd).getD "00000000")
def dequantScale (alpha token : Option Expr) : Option Expr :=
  match token, alpha with
  | none, a => a
  | some t, none => some t
  | some t, some a => some (mul t a)
def sigmoid (value negativeLog2e one : Expr) : Expr :=
  .op "rcp.approx.ftz.f32" [add (.op "ex2.approx.ftz.f32" [mul value negativeLog2e]) one]
def tanh (v : Variant) (value : Expr) : Expr :=
  add (mul (coefficient v "two") (.op "rcp.approx.ftz.f32"
    [add (.op "ex2.approx.ftz.f32" [mul value (coefficient v "negative_two_log2e")])
      (coefficient v "one")])) (coefficient v "negative_one")

def activation (v : Variant) (gate up : Expr)
    (alpha weight norm : Option Expr := none) : Expr := Id.run do
  let c := coefficient v
  let mut gate := gate
  let mut up := up
  if let some a := alpha then
    gate := mul gate a
    up := mul up a
  if v.clamp then
    if ["bf16", "mxfp8_legacy", "w4a16"].contains v.family then
      gate := .op "min.NaN.f32" [gate, c "clamp"]
      up := .op "max.NaN.f32" [.op "min.NaN.f32" [up, c "clamp"], c "negative_clamp"]
    else
      gate := .op "min.f32" [gate, c "clamp"]
      up := .op "min.xorsign.abs.f32" [up, c "clamp"]
  let one := c "one"
  let mut result := zero
  if v.activation == "swiglu" then
    let mut sigmoidInput := gate
    let mut exponentCoefficient := c "negative_log2e"
    if v.custom then
      if ["nvfp4_legacy", "mxfp8_legacy"].contains v.family then
        sigmoidInput := mul gate (c "swiglu_alpha")
      else
        exponentCoefficient := c "negative_alpha_log2e"
      up := add up (c "swiglu_beta")
    let s := sigmoid sigmoidInput exponentCoefficient one
    result := if v.family == "mxfp8_legacy" then mul (mul s gate) up
      else if v.family == "w4a16" && !v.custom then mul up (mul gate s)
      else mul (mul up gate) s
  else
    let s := sigmoid gate (c "negative_log2e") one
    if v.family == "w4a16" then
      let bounded := mul (c "situ_beta") (tanh v (mul gate (c "inverse_beta")))
      if v.linear then
        up := mul (c "linear_beta") (tanh v (mul up (c "inverse_linear_beta")))
      result := mul up (mul bounded s)
    else
      let bounded := add (mul (sigmoid (mul gate (c "two_over_beta")) (c "negative_log2e") one)
        (c "two_beta")) (c "negative_beta")
      if v.linear then
        up := add (mul (sigmoid (mul up (c "two_over_linear_beta")) (c "negative_log2e") one)
          (c "two_linear_beta")) (c "negative_linear_beta")
      result := mul (mul bounded s) up
  if let some w := weight then result := mul result w
  if let some n := norm then result := mul result n
  return result

/-- Static lowering is independent of token placement once logical operands,
coefficient bits and runtime scalar operands have been restored. -/
theorem activation_placement (v : Variant) (g u g' u' : Expr)
    (a w n : Option Expr) (hg : g = g') (hu : u = u') :
    activation v g u a w n = activation v g' u' a w n := by
  subst g'; subst u'; rfl
end BIK.NumericalVariants
