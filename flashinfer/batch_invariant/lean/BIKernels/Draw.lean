-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.Canon
import BIKernels.Semantics
import BIKernels.Support
import BIKernels.Sampling
/-!
# `sample.icdf.v1`: the inverse-CDF draw as a reading of the canonical sum tree

The draw (`sampling/draw_spec.py`) adds no reduction to the row class: its threshold is `mul(u, S)` with `S` the
canonical `add` tree over the row's weights (the `[kb, 5]` digits of `Sampling.specSum`, leaves the slot folds), and the
prefix of an element is built from *nodes of that tree*: the slot fold's running value, plus, for every set bit of the
element's tree address (block bits low to high, then lane bits), the left-sibling sub-tree -- `blockNode` (the `treeC
.add` over an aligned group of blocks of one lane) or `laneNode` (the `treeC .add` over an aligned group of lanes) --
chained by `add` in ascending bit order from `+0`. Because every operand is a node of the tree, the prefix is as
placement-invariant as the tree (`config_invarianceC`); this file pins the declared dataflow, `total_logits_eq_specSum`
states that the logits form's `S` is the row class's `s` term (the R4 coupling of `sampling_from_logits` with
`log_probs`), and the generated cases (`SamplingCases.lean`, `draw*`) couple the terms node for node to the Python
spec's S-expressions. `select` states the selection rule over the evaluated row, and `select_pos` that a drawn element has a positive
weight (a masked token is never drawn).
-/
namespace BIK
namespace Draw

/-- The slot fold's running value through element `e` of slot `s`: `add` from `+0` over the present elements `j <= e`
(absent when none is present). -/
def slotRun (N : Nat) (w : Nat → Term) (s e : Nat) : Option Term :=
  let js := (List.range (e + 1)).filter (fun j => s * 8 + j < N)
  match js with
  | [] => none
  | _ => some (js.foldl (fun acc j => Term.add acc (w (s * 8 + j))) (Term.const "0.0"))

/-- The slot total: the fold through the last element. -/
def slotTotal (N : Nat) (w : Nat → Term) (s : Nat) : Option Term := slotRun N w s 7

/-- The block node of lane `lane` at level `lv`, group `g`: the pairwise tree over the slot totals of blocks
`[g <<< lv, (g + 1) <<< lv)` of that lane (low block first, as `canon._pairwise`). -/
def blockNode (N : Nat) (w : Nat → Term) (lane lv g : Nat) : Option Term :=
  treeC .add lv (fun bs => slotTotal N w (((g <<< lv) + bitsToNat bs) * 32 + lane))

/-- The lane node at level `lv`, group `g`: the pairwise tree over the block-tree roots of lanes `[g <<< lv, (g + 1) <<< lv)`. -/
def laneNode (N kb : Nat) (w : Nat → Term) (lv g : Nat) : Option Term :=
  treeC .add lv (fun bs => blockNode N w ((g <<< lv) + bitsToNat bs) kb 0)

def addO (acc : Term) : Option Term → Term
  | none => acc
  | some t => Term.add acc t

/-- `outer` of slot `(lane, block)`: the `add` chain from `+0`, in ascending bit order, of the left-sibling nodes of the
slot's tree address (the group just below each set bit). -/
def outer (N kb : Nat) (w : Nat → Term) (lane block : Nat) : Term :=
  let b := (List.range kb).foldl
    (fun acc i => if (block >>> i) % 2 = 1 then addO acc (blockNode N w lane i ((block >>> i) - 1)) else acc) (Term.const "0.0")
  (List.range 5).foldl
    (fun acc k => if (lane >>> k) % 2 = 1 then addO acc (laneNode N kb w k ((lane >>> k) - 1)) else acc) b

/-- The inclusive prefix of element `i` (flat index `(block * 32 + lane) * 8 + elem`): `add(outer, fold_e)`. -/
def prefixIncl (N kb : Nat) (w : Nat → Term) (i : Nat) : Option Term :=
  (slotRun N w (i / 8) (i % 8)).map (fun f => Term.add (outer N kb w ((i / 8) % 32) ((i / 8) / 32)) f)

/-- The total `S`: the canonical sum tree over the weights (`Sampling.specSum`'s digits with the weight leaves). -/
def total (N kb : Nat) (w : Nat → Term) : Option Term :=
  canonicalDigitsC .add [kb, 5] (fun bs => slotTotal N w (bitsToNat (takeN kb bs) * 32 + bitsToNat (dropN kb bs)))

/-- The threshold `t = mul(u, S)`, `u` the row's uniform (a symbol). -/
def threshold (N kb : Nat) (w : Nat → Term) : Option Term := (total N kb w).map (fun S => Term.mul (Term.sym "u") S)

/-- The weights of the logits form: `lse.row`'s `e_j = EX2(mul(add(x_j, nm), c))`. -/
def logitWeights (row : Nat → Term) (nm c : Term) : Nat → Term :=
  fun j => Sampling.weight (Sampling.shifted (row j) nm c)

/-- **The logits form's `S` is the row class's `s`**: the draw's total over the `lse.row` weights is `Sampling.specSum`
of the same row, negated maximum and scale, node for node (so `sampling_from_logits(batch_invariant=True)` draws against
the statistics `log_probs` reports). -/
theorem total_logits_eq_specSum (N kb : Nat) (row : Nat → Term) (nm c : Term) :
    total N kb (logitWeights row nm c) = Sampling.specSum N kb row nm c := rfl

/-! ### The selection rule

Stated over the evaluated row (binary32 values, not terms): in tree order, element `i` is *present* when it exists,
*positive* when its weight is `> 0`, and *crosses* when its inclusive prefix is `>= t`. The drawn element is the first
present positive element that crosses, else the last present positive element; `none` when no present element is
positive (then `S` is not a positive finite number and the row is invalid). The positivity condition is part of the
rule (`draw_spec.SELECT_RULE`, named in the fingerprint): the inclusive prefixes are not monotone across slot
boundaries -- the last prefix of a slot and the `outer` of the next are different `add` chains over the same nodes,
equal exactly and a rounding apart in binary32 -- so without it a zero-weight slot head could be the first crossing
and a masked token drawn (found in review, 2026-11-11). -/

/-- The first index in `[0, n)` that is present, positive and crosses; else the last present positive index. -/
def select (n : Nat) (present pos cross : Nat → Bool) : Option Nat :=
  match (List.range n).find? (fun i => present i && pos i && cross i) with
  | some i => some i
  | none => (List.range n).reverse.find? (fun i => present i && pos i)

/-- **A drawn element has a positive weight**: a masked token (weight `+0`) is never drawn. -/
theorem select_pos (n : Nat) (present pos cross : Nat → Bool) (i : Nat)
    (h : select n present pos cross = some i) : pos i = true := by
  unfold select at h
  split at h
  · rename_i j hj
    cases h
    have hp := List.find?_some hj
    simp only [Bool.and_eq_true] at hp
    exact hp.1.2
  · have hp := List.find?_some h
    simp only [Bool.and_eq_true] at hp
    exact hp.2

/-- A drawn element is present (it exists in the vocabulary). -/
theorem select_present (n : Nat) (present pos cross : Nat → Bool) (i : Nat)
    (h : select n present pos cross = some i) : present i = true := by
  unfold select at h
  split at h
  · rename_i j hj
    cases h
    have hp := List.find?_some hj
    simp only [Bool.and_eq_true] at hp
    exact hp.1.1
  · have hp := List.find?_some h
    simp only [Bool.and_eq_true] at hp
    exact hp.1

/-- A drawn element lies in `[0, n)`. -/
theorem select_lt (n : Nat) (present pos cross : Nat → Bool) (i : Nat)
    (h : select n present pos cross = some i) : i < n := by
  unfold select at h
  split at h
  · rename_i j hj
    cases h
    exact List.mem_range.mp (List.mem_of_find?_eq_some hj)
  · exact List.mem_range.mp (List.mem_reverse.mp (List.mem_of_find?_eq_some h))

/-- The rule never selects an absent or zero-weight element even when such an element crosses first: `select` with a
crossing predicate restricted to positive elements is `select` itself (the restriction is already in the rule). -/
theorem select_restrict (n : Nat) (present pos cross : Nat → Bool) :
    select n present pos (fun i => pos i && cross i) = select n present pos cross := by
  unfold select
  have : (fun i => present i && pos i && (pos i && cross i)) = (fun i => present i && pos i && cross i) := by
    funext i
    cases present i <;> cases pos i <;> cases cross i <;> rfl
  rw [this]

end Draw
end BIK
