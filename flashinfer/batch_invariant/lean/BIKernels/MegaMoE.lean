-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import Std

/-!
# SM100/103/107 BF16 MegaMoE's ordered numerical trace

This model keeps every floating-point operation opaque and ordered. In
particular it assumes neither associativity nor a zero identity for floating
point addition. K padding and invalid top-k slots remain explicit in the tree.
Each architecture fixes its own interpretation of the opaque MMA/MUFU atoms;
the Python receipt names that architecture. Reusing a trace theorem on another
architecture does not assert cross-architecture equality of output bits.

The placement theorem assumes an inverse addressing map and collision-free
logical cells. Correct, exactly-once physical publication is an external
assumption of `published`; it is not established by the injectivity predicate
`ExactlyOnce`. The model does not prove a CUDA program race free, establish the hardware
meaning of MMA/MUFU atoms, or verify compiler lowering. Python conformance
couples the definitions below to the executable spec, not to generated SASS.
-/
namespace BIK.MegaMoE

inductive Expr where
  | op : String → List Expr → Expr
  | str : String → Expr
  | int : Int → Expr
deriving Repr, BEq

mutual
  private def sexprImpl : Expr → String
    | .str s => s
    | .int i => toString i
    | .op name [] => "(op:" ++ name ++ " )"
    | .op name args => "(op:" ++ name ++ sexprsImpl args ++ ")"
  termination_by e => sizeOf e

  private def sexprsImpl : List Expr → String
    | [] => ""
    | arg :: rest => " " ++ sexprImpl arg ++ sexprsImpl rest
  termination_by args => sizeOf args
end

/-- The recursor keeps kernel reduction small; the compiled implementation has
the same traversal and is checked against the kernel-decided Python pins. -/
@[implemented_by sexprImpl]
def sexpr : Expr → String :=
  Expr.rec (motive_1 := fun _ => String) (motive_2 := fun _ => String)
    (fun name original args => "(op:" ++ name ++ (if original.isEmpty then " " else args) ++ ")")
    (fun value => value) (fun value => toString value) ""
    (fun _ _ arg rest => " " ++ arg ++ rest)

def zero : Expr := .op "f32_bits" [.str "00000000"]

def slice (K : Nat) (operand : String) (step : Nat) : Expr :=
  let valid : Nat := min 16 (K - step * 16)
  .op "operand.k16" [.str operand, .int step, .int valid]

/-- Initializing MMA is a distinct atom; it does not read a zero accumulator. -/
def mmaStep (K : Nat) (operand : String) (acc : Option Expr) (step : Nat) : Option Expr :=
  some (match acc with
    | none => .op "mma.init.bf16.k16" [slice K operand step]
    | some acc => .op "mma.acc.bf16.k16" [acc, slice K operand step])

def kTiles (K : Nat) : Nat := (K + 63) / 64
def paddedSteps (K : Nat) : Nat := kTiles K * 4

def blockedIndices (tiles : Nat) : List (List Nat) :=
  (List.range tiles).map (fun tile => (List.range 4).map (fun inner => tile * 4 + inner))

theorem blocked_indices_flat (tiles : Nat) :
    (blockedIndices tiles).flatten = List.range (tiles * 4) := by
  induction tiles with
  | zero => rfl
  | succ n ih =>
      simp only [blockedIndices, List.range_succ, List.map_append, List.map_cons,
        List.map_nil, List.flatten_append, List.flatten_cons, List.flatten_nil,
        List.append_nil] at *
      rw [ih, Nat.succ_mul, List.range_add]
      rfl

def mmaFlat (K : Nat) (operand : String) : Expr :=
  ((List.range (paddedSteps K)).foldl (mmaStep K operand) none).getD zero

def mmaBlocked (K : Nat) (operand : String) : Expr :=
  ((blockedIndices (kTiles K)).foldl
    (fun acc tile => tile.foldl (mmaStep K operand) acc) none).getD zero

/-- The two nested device loops preserve every K16 step, including padded steps. -/
theorem blocked_k64_four_k16_eq_flat (K : Nat) (operand : String) :
    mmaBlocked K operand = mmaFlat K operand := by
  unfold mmaBlocked mmaFlat paddedSteps
  rw [← List.foldl_flatten, blocked_indices_flat]

/-- Padding does not disappear by invoking any floating-point identity. -/
theorem padded_step_count (K : Nat) :
    (blockedIndices (kTiles K)).flatten.length = ((K + 63) / 64) * 4 := by
  rw [blocked_indices_flat, List.length_range]
  rfl

/-- Physical stages regroup K16 updates; they do not choose canonical padding. -/
def tiledIndices (tiles stepsPerStage : Nat) : List (List Nat) :=
  (List.range tiles).map (fun tile =>
    (List.range stepsPerStage).map (fun inner => tile * stepsPerStage + inner))

theorem tiled_indices_flat (tiles stepsPerStage : Nat) :
    (tiledIndices tiles stepsPerStage).flatten = List.range (tiles * stepsPerStage) := by
  induction tiles with
  | zero => simp [tiledIndices]
  | succ n ih =>
      simp only [tiledIndices, List.range_succ, List.map_append, List.map_cons,
        List.map_nil, List.flatten_append, List.flatten_cons, List.flatten_nil,
        List.append_nil] at *
      rw [ih, Nat.succ_mul, List.range_add]

def PaddingCompatible (K stageK : Nat) : Prop := (kTiles K * 64) % stageK = 0

instance (K stageK : Nat) : Decidable (PaddingCompatible K stageK) := by
  unfold PaddingCompatible
  infer_instance

/-- Larger stages are admitted only when their total instruction count agrees
with the original K64 padding, including partially filled K16 slices. -/
theorem compatible_stage_count (K stageK : Nat)
    (hs : stageK = 64 ∨ stageK = 128 ∨ stageK = 256)
    (hp : PaddingCompatible K stageK) :
    ((K + stageK - 1) / stageK) * (stageK / 16) = paddedSteps K := by
  rcases hs with hs | hs | hs
  all_goals subst stageK
  all_goals simp only [PaddingCompatible, kTiles, paddedSteps] at *
  all_goals omega

def mmaTiled (K stageK : Nat) (operand : String) : Expr :=
  ((tiledIndices ((K + stageK - 1) / stageK) (stageK / 16)).foldl
    (fun acc tile => tile.foldl (mmaStep K operand) acc) none).getD zero

/-- The proof uses exact step sequences, not a zero-MMA identity. -/
theorem tiled_mma_eq_canonical (K stageK : Nat) (operand : String)
    (hs : stageK = 64 ∨ stageK = 128 ∨ stageK = 256)
    (hp : PaddingCompatible K stageK) : mmaTiled K stageK operand = mmaFlat K operand := by
  unfold mmaTiled mmaFlat
  rw [← List.foldl_flatten, tiled_indices_flat, compatible_stage_count K stageK hs hp]

inductive Scheduler where | static | atomicCounter
deriving Repr, BEq, DecidableEq
inductive Carrier where | epi | standalone | reuseDispatch
deriving Repr, BEq, DecidableEq
structure Plan where
  scheduler : Scheduler
  carrier : Carrier
deriving Repr, BEq, DecidableEq

def plans : List Plan :=
  [⟨.static, .epi⟩, ⟨.static, .standalone⟩, ⟨.static, .reuseDispatch⟩,
   ⟨.atomicCounter, .epi⟩, ⟨.atomicCounter, .standalone⟩,
   ⟨.atomicCounter, .reuseDispatch⟩]

theorem six_plans : plans.length = 6 := rfl

theorem every_plan_is_legal (p : Plan) : p ∈ plans := by
  obtain ⟨scheduler, carrier⟩ := p
  cases scheduler <;> cases carrier <;> simp [plans]

structure PlanConfig where
  tileM : Nat
  tileN : Nat
  tileK : Nat
  clusterM : Nat
  useTwoCta : Bool
deriving Repr, BEq, DecidableEq

def GeometryAdmitted (g : PlanConfig) : Prop :=
  ((g.tileM = 128 ∧ g.clusterM = 1 ∧ g.useTwoCta = false) ∨
   (g.tileM = 256 ∧ g.clusterM = 2 ∧ g.useTwoCta = true)) ∧
  (g.tileN = 128 ∨ g.tileN = 256) ∧
  (g.tileK = 64 ∨ g.tileK = 128 ∨ g.tileK = 256)

instance (g : PlanConfig) : Decidable (GeometryAdmitted g) := by
  unfold GeometryAdmitted
  infer_instance

def PlanCompatible (g : PlanConfig) (hidden intermediate : Nat) : Prop :=
  GeometryAdmitted g ∧ PaddingCompatible hidden g.tileK ∧ PaddingCompatible intermediate g.tileK

instance (g : PlanConfig) (hidden intermediate : Nat) :
    Decidable (PlanCompatible g hidden intermediate) := by
  unfold PlanCompatible
  infer_instance

/-- M/N ownership is explicit, independently of the K-fold expression. -/
structure TileCoord where
  rowTile : Nat
  colTile : Nat
  rowWithin : Nat
  colWithin : Nat
deriving Repr, BEq, DecidableEq

def outputTileWidth (g : PlanConfig) (handoff : Bool) : Nat :=
  if handoff then g.tileN / 2 else g.tileN

def outputTileCoord (g : PlanConfig) (handoff : Bool) (row col : Nat) : TileCoord :=
  let width := outputTileWidth g handoff
  ⟨row / g.tileM, col / width, row % g.tileM, col % width⟩

def outputLogicalCoord (g : PlanConfig) (handoff : Bool) (p : TileCoord) : Nat × Nat :=
  (p.rowTile * g.tileM + p.rowWithin,
   p.colTile * outputTileWidth g handoff + p.colWithin)

/-- Both FC1's N/2 handoff and FC2's N-column layout recover each original
logical element. This does not establish the CUDA publication protocol. -/
theorem tile_output_inverse (g : PlanConfig) (handoff : Bool) (row col : Nat) :
    outputLogicalCoord g handoff (outputTileCoord g handoff row col) = (row, col) := by
  apply Prod.ext
  · simp only [outputLogicalCoord, outputTileCoord]
    simpa only [Nat.mul_comm] using Nat.div_add_mod row g.tileM
  · simp only [outputLogicalCoord, outputTileCoord]
    simpa only [Nat.mul_comm] using Nat.div_add_mod col (outputTileWidth g handoff)

theorem tile_coordinates_in_bounds (g : PlanConfig) (hg : GeometryAdmitted g)
    (handoff : Bool) (row col : Nat) :
    (outputTileCoord g handoff row col).rowWithin < g.tileM ∧
    (outputTileCoord g handoff row col).colWithin < outputTileWidth g handoff := by
  have hm : 0 < g.tileM := by
    rcases hg.1 with h | h <;> omega
  have hn : 0 < outputTileWidth g handoff := by
    rcases hg.2.1 with h | h <;> cases handoff <;> simp [outputTileWidth, h]
  exact ⟨Nat.mod_lt row hm, Nat.mod_lt col hn⟩

theorem tile_cta_rows (g : PlanConfig) (hg : GeometryAdmitted g) :
    g.tileM / g.clusterM = 128 := by
  rcases hg.1 with ⟨hm, hc, _⟩ | ⟨hm, hc, _⟩ <;> simp [hm, hc]

theorem tile_cells_unique (g : PlanConfig) (handoff : Bool) (r c r' c' : Nat)
    (h : outputTileCoord g handoff r c = outputTileCoord g handoff r' c') :
    (r, c) = (r', c') := by
  have h' := congrArg (outputLogicalCoord g handoff) h
  simpa only [tile_output_inverse] using h'

def cvt (x : Expr) : Expr := .op "cvt.rn.bf16.f32" [x]
def f32 (bits : String) : Expr := .op "f32_bits" [.str bits]

/-- The device's shared scalar chain; both clamp literals retain their exact bits. -/
def activation (early : Bool) (clamp : Option (String × String)) : Expr :=
  let gate := Expr.op "ref.f32" [.str "fc1_gate"]
  let up := Expr.op "ref.f32" [.str "fc1_up"]
  let (gate, up) := match clamp with
    | none => (gate, up)
    | some (pos, neg) =>
        (Expr.op "min.NaN.f32" [gate, f32 pos],
         Expr.op "max.NaN.f32" [.op "min.NaN.f32" [up, f32 pos], f32 neg])
  let product := Expr.op "mul.rn.f32" [up, gate]
  let exponent := Expr.op "mul.rn.f32" [gate, f32 "bfb8aa3b"]
  let sigmoid := Expr.op "rcp.approx.ftz.f32"
    [.op "add.rn.f32" [.op "ex2.approx.ftz.f32" [exponent], f32 "3f800000"]]
  let value := Expr.op "mul.rn.f32" [product, sigmoid]
  cvt (if early then .op "mul.rn.f32" [value, .op "input.f32" [.str "route_weight"]]
       else value)

def finalizerSlot (experts slot : Nat) (mask : Option (List Bool)) : Expr × Expr :=
  let wire := Expr.op "ref.bf16" [.str "fc2", .int slot]
  let score := Expr.op "input.f32" [.str "route_weight", .int slot]
  let wireZero := Expr.op "bf16_bits" [.str "0000"]
  match mask with
  | some bits => if (bits[slot]?).getD false then (wire, score) else (wireZero, zero)
  | none =>
      let valid := Expr.op "route.valid" [.op "input.expert_id" [.int slot], .int experts]
      (.op "select.lazy" [valid, wire, wireZero], .op "select.lazy" [valid, score, zero])

def finalizerStep (experts : Nat) (early : Bool) (mask : Option (List Bool))
    (acc : Expr) (slot : Nat) : Expr :=
  let (wire, score) := finalizerSlot experts slot mask
  let value := Expr.op "widen.bf16.f32" [wire]
  if early then .op "add.rn.f32" [acc, value]
  else .op "fma.rn.f32" [value, score, acc]

def finalizer (slots experts : Nat) (early : Bool) (mask : Option (List Bool) := none) : Expr :=
  cvt ((List.range slots).foldl (finalizerStep experts early mask) zero)

/-- No algebraic identity removes an invalid slot: its ordered add/FMA remains. -/
theorem invalid_slot_keeps_add (experts slot : Nat) (mask : List Bool) (acc : Expr)
    (h : mask[slot]? = some false) :
    finalizerStep experts true (some mask) acc slot =
      .op "add.rn.f32" [acc, .op "widen.bf16.f32" [.op "bf16_bits" [.str "0000"]]] := by
  simp only [finalizerStep, finalizerSlot, h, Option.getD_some, Bool.false_eq_true,
    ↓reduceIte]

theorem invalid_slot_keeps_fma (experts slot : Nat) (mask : List Bool) (acc : Expr)
    (h : mask[slot]? = some false) :
    finalizerStep experts false (some mask) acc slot =
      .op "fma.rn.f32" [.op "widen.bf16.f32" [.op "bf16_bits" [.str "0000"]], zero, acc] := by
  simp only [finalizerStep, finalizerSlot, h, Option.getD_some, Bool.false_eq_true,
    ↓reduceIte]

structure Components where
  fc1Gate : Expr
  fc1Up : Expr
  act : Expr
  fc2 : Expr
  finalize : Expr

def components (hidden intermediate slots experts : Nat) (early : Bool)
    (clamp : Option (String × String)) (blocked : Bool)
    (mask : Option (List Bool) := none) : Components :=
  let mma := if blocked then mmaBlocked else mmaFlat
  ⟨mma hidden "fc1.gate", mma hidden "fc1.up", activation early clamp,
   cvt (mma intermediate "fc2.handoff_w2"), finalizer slots experts early mask⟩

def componentPairs (c : Components) : List (String × Expr) :=
  [("fc1_gate", c.fc1Gate), ("fc1_up", c.fc1Up), ("activation", c.act),
   ("fc2", c.fc2), ("finalize", c.finalize)]

/-- Every numerical component of the blocked implementation equals the flat spec. -/
theorem components_blocked_eq_flat (hidden intermediate slots experts : Nat) (early : Bool)
    (clamp : Option (String × String)) (mask : Option (List Bool)) :
    components hidden intermediate slots experts early clamp true mask =
      components hidden intermediate slots experts early clamp false mask := by
  simp only [components, Bool.true_eq, ↓reduceIte, Bool.false_eq_true,
    blocked_k64_four_k16_eq_flat]

def tiledComponents (g : PlanConfig) (hidden intermediate slots experts : Nat) (early : Bool)
    (clamp : Option (String × String)) (mask : Option (List Bool) := none) : Components :=
  ⟨mmaTiled hidden g.tileK "fc1.gate", mmaTiled hidden g.tileK "fc1.up",
   activation early clamp, cvt (mmaTiled intermediate g.tileK "fc2.handoff_w2"),
   finalizer slots experts early mask⟩

/-- Every compatible geometry has the existing canonical five-component trace.
The hardware meaning of different MMA M/N shapes remains an opaque-atom
interpretation assumption; equality here does not verify CUDA lowering. -/
theorem tile_components_eq_canonical (g : PlanConfig) (hidden intermediate slots experts : Nat)
    (early : Bool) (clamp : Option (String × String)) (mask : Option (List Bool))
    (hp : PlanCompatible g hidden intermediate) :
    tiledComponents g hidden intermediate slots experts early clamp mask =
      components hidden intermediate slots experts early clamp false mask := by
  obtain ⟨hg, hh, hi⟩ := hp
  obtain ⟨_, _, hk⟩ := hg
  simp only [tiledComponents, components, Bool.false_eq_true, ↓reduceIte,
    tiled_mma_eq_canonical hidden g.tileK "fc1.gate" hk hh,
    tiled_mma_eq_canonical hidden g.tileK "fc1.up" hk hh,
    tiled_mma_eq_canonical intermediate g.tileK "fc2.handoff_w2" hk hi]

theorem all_tile_lowerings_same_trace (g h : PlanConfig) (hidden intermediate slots experts : Nat)
    (early : Bool) (clamp : Option (String × String)) (mask : Option (List Bool))
    (hg : PlanCompatible g hidden intermediate) (hh : PlanCompatible h hidden intermediate) :
    tiledComponents g hidden intermediate slots experts early clamp mask =
      tiledComponents h hidden intermediate slots experts early clamp mask := by
  rw [tile_components_eq_canonical g hidden intermediate slots experts early clamp mask hg,
      tile_components_eq_canonical h hidden intermediate slots experts early clamp mask hh]

/-- The ordered slot fold retains a masked slot as a zero-valued term. -/
def slotTerm (valid : Bool) (term : Expr) : Expr := if valid then term else zero
def slotFold (n : Nat) (values : Nat → Expr) : Expr :=
  (List.range n).foldl (fun acc slot => .op "add.rn.f32" [acc, values slot]) zero

theorem masked_slot_not_dropped (a : Expr) :
    (Expr.op "add.rn.f32" [a, slotTerm false zero]) = .op "add.rn.f32" [a, zero] := rfl

abbrev Route := Nat × Nat

/-- A physical placement selected by one of the six communication plans. -/
structure Placement where
  plan : Plan
  place : Route → Nat
  inverse : Nat → Route

def Inverse (p : Placement) : Prop := ∀ route, p.inverse (p.place route) = route
/-- Collision-free logical cells (injective addressing). This predicate
does not count physical publications; their correctness and exactly-once
execution remain an external assumption of `published`. -/
def ExactlyOnce (p : Placement) : Prop :=
  ∀ a b, p.place a = p.place b → a = b

theorem inverse_implies_exactly_once (p : Placement) (hp : Inverse p) : ExactlyOnce p := by
  intro a b hab
  have h := congrArg p.inverse hab
  simpa only [hp a, hp b] using h

/-- Assumed publication contents, not a proof of a CUDA write protocol. -/
def published (p : Placement) (logical : Route → Expr) : Nat → Expr :=
  fun address => logical (p.inverse address)
def placedRead (p : Placement) (logical : Route → Expr) (route : Route) : Expr :=
  published p logical (p.place route)

theorem placement_preserves_token_slot (p : Placement) (logical : Route → Expr)
    (hp : Inverse p) (token slot : Nat) :
    placedRead p logical (token, slot) = logical (token, slot) := by
  unfold placedRead published
  rw [hp]

def placedOutput (p : Placement) (logical : Route → Expr) (token slots : Nat) : Expr :=
  slotFold slots (fun slot => placedRead p logical (token, slot))
def logicalOutput (logical : Route → Expr) (token slots : Nat) : Expr :=
  slotFold slots (fun slot => logical (token, slot))

/-- Every physical plan preserves the original token and slot before reduction. -/
theorem placement_output (p : Placement) (logical : Route → Expr)
    (hp : Inverse p) (_once : ExactlyOnce p) (token slots : Nat) :
    placedOutput p logical token slots = logicalOutput logical token slots := by
  unfold placedOutput logicalOutput
  have h : (fun slot => placedRead p logical (token, slot)) =
      (fun slot => logical (token, slot)) := by
    funext slot
    exact placement_preserves_token_slot p logical hp token slot
  rw [h]

theorem all_legal_plans_same_trace (p q : Placement) (logical : Route → Expr)
    (_pLegal : p.plan ∈ plans) (_qLegal : q.plan ∈ plans)
    (hp : Inverse p) (hq : Inverse q) (ep : ExactlyOnce p) (eq : ExactlyOnce q)
    (token slots : Nat) :
    placedOutput p logical token slots = placedOutput q logical token slots := by
  rw [placement_output p logical hp ep, placement_output q logical hq eq]

/-- Rows may be batched in any order; the addressing proof still precedes the fold. -/
def batch (p : Placement) (logical : Route → Expr) (rows : List Nat) (slots : Nat) : List Expr :=
  rows.map (fun token => placedOutput p logical token slots)

theorem batch_invariance (p : Placement) (logical : Route → Expr)
    (hp : Inverse p) (ep : ExactlyOnce p) (rows : List Nat) (slots : Nat) :
    batch p logical rows slots = rows.map (fun token => logicalOutput logical token slots) := by
  unfold batch
  apply List.map_congr_left
  intro token _
  exact placement_output p logical hp ep token slots

/-- A batch permutation carries each token's existing numerical trace with it. -/
theorem permutation_invariance (p : Placement) (logical : Route → Expr)
    (hp : Inverse p) (ep : ExactlyOnce p) (a b : List Nat) (slots : Nat)
    (hperm : a.Perm b) : (batch p logical a slots).Perm (batch p logical b slots) := by
  rw [batch_invariance p logical hp ep, batch_invariance p logical hp ep]
  exact hperm.map _

/-- A subbatch keeps the exact traces of the retained tokens, including order. -/
theorem subbatch_invariance (p : Placement) (logical : Route → Expr)
    (hp : Inverse p) (ep : ExactlyOnce p) (rows : List Nat) (slots : Nat)
    (keep : Nat → Bool) :
    batch p logical (rows.filter keep) slots =
      (rows.filter keep).map (fun token => logicalOutput logical token slots) := by
  exact batch_invariance p logical hp ep (rows.filter keep) slots

/-- A generic ordered combine covers both the add and FMA finalizer branches. -/
def placedFold (p : Placement) (logical : Route → Expr) (token slots : Nat)
    (step : Expr → Nat → Expr → Expr) : Expr :=
  (List.range slots).foldl (fun acc slot => step acc slot
    (placedRead p logical (token, slot))) zero

theorem placed_fold_eq (p : Placement) (logical : Route → Expr)
    (hp : Inverse p) (_once : ExactlyOnce p) (token slots : Nat)
    (step : Expr → Nat → Expr → Expr) :
    placedFold p logical token slots step =
      (List.range slots).foldl (fun acc slot => step acc slot (logical (token, slot))) zero := by
  unfold placedFold
  have h : (fun acc slot => step acc slot (placedRead p logical (token, slot))) =
      (fun acc slot => step acc slot (logical (token, slot))) := by
    funext acc slot
    rw [placement_preserves_token_slot p logical hp token slot]
  rw [h]

/-- The guarantee is also pointwise: selecting a token never changes its trace. -/
theorem token_independent_of_batch (p q : Placement) (logical : Route → Expr)
    (hp : Inverse p) (hq : Inverse q) (ep : ExactlyOnce p) (eq : ExactlyOnce q)
    (token slots : Nat) (step : Expr → Nat → Expr → Expr) :
    placedFold p logical token slots step = placedFold q logical token slots step := by
  rw [placed_fold_eq p logical hp ep, placed_fold_eq q logical hq eq]

/-- Trace equality entails equal bits under any fixed interpretation of opaque atoms. -/
theorem interpreted_outputs_equal {Bits : Type} (interpret : Expr → Bits)
    (p q : Placement) (logical : Route → Expr)
    (hp : Inverse p) (hq : Inverse q) (ep : ExactlyOnce p) (eq : ExactlyOnce q)
    (token slots : Nat) (step : Expr → Nat → Expr → Expr) :
    interpret (placedFold p logical token slots step) =
      interpret (placedFold q logical token slots step) := by
  exact congrArg interpret (token_independent_of_batch p q logical hp hq ep eq token slots step)

theorem subbatch_preserves_retained_traces (p : Placement) (logical : Route → Expr)
    (small large : List Nat) (slots : Nat) (h : small.Sublist large) :
    (batch p logical small slots).Sublist (batch p logical large slots) := by
  exact h.map _

end BIK.MegaMoE
