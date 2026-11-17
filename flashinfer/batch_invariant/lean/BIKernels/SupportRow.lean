-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.Canon
import BIKernels.Semantics
import BIKernels.Support
import BIKernels.Sampling
import BIKernels.Draw
/-!
# `support.row.v1`: the kept set under top-k / top-p / min-p

The class (`sampling/support_spec.py`) keeps, per row, the set `{j : w_j >= threshold}` for a per-row threshold, so
every kept set is closed upwards and ties at the cut are kept. The thresholds are *integer* procedures over the weights'
binary32 bit patterns (non-negative values order as their patterns): top-k's is the largest pattern whose cut keeps at
least `k` elements, top-p's the largest positive pattern whose cut's fixed-point mass reaches the target -- or, when the
positive mass does not reach it, the smallest positive pattern (every positive weight kept). This file states those
procedures over lists of patterns and proves what the class declares of them: top-k keeps at least `k` and no cut at a
larger weight does (the cut one step up keeps fewer than `k`); top-p reaches the target and no cut at a larger positive
weight does, keeps only positive weights, and in the fallback keeps every positive weight; the intersection of cuts is
the cut at the largest threshold (so the kernel's single comparison against the effective threshold is the mask). The
kept mass is `lse.row`'s canonical sum tree over the kept weights with the dropped ones `+0`; as a tree over a leaf
function it inherits the engine's placement invariance (`mass_invariance`), and keeping everything gives the draw's
total (`massTree_all`). The renormalised probability of a kept element is `lse.row`'s `prob` chain with the kept mass
(`renormProb`). The generated cases (`SamplingCases.lean`, `sup*`) evaluate the procedures on concrete pattern lists
against the Python reference (kernel `decide` at the small vocabulary; the conformance executable prints every
threshold and the mass and renormalisation terms for the Python comparison).
-/
namespace BIK
namespace SupportRow

/-- The pattern of `+inf`: the threshold that keeps nothing finite. -/
def INF : Nat := 0x7F800000

/-- How many patterns the cut `v` keeps. -/
def countGE (pats : List Nat) (v : Nat) : Nat := pats.countP (fun p => decide (v ≤ p))

/-- The fixed-point mass the cut `v` keeps (`q` the mass of a pattern; an integer sum, so its order is immaterial). -/
def massGE (pats : List Nat) (q : Nat → Nat) (v : Nat) : Nat :=
  (pats.filter (fun p => decide (v ≤ p))).foldl (fun acc p => acc + q p) 0

def maxOf (l : List Nat) : Nat := l.foldl max 0

def minOf : List Nat → Nat
  | [] => INF
  | x :: xs => xs.foldl min x

def positives (pats : List Nat) : List Nat := pats.filter (fun p => decide (0 < p))

/-- top-k: `k = 0` keeps nothing, `k >= N` everything; otherwise the largest pattern whose cut keeps at least `k`. -/
def topK (pats : List Nat) (k : Nat) : Nat :=
  if k = 0 then INF
  else if pats.length ≤ k then 0
  else maxOf (pats.filter (fun v => decide (k ≤ countGE pats v)))

/-- top-p: no positive pattern keeps nothing; when the positive mass reaches the target, the largest positive pattern
whose cut reaches it; otherwise the smallest positive pattern. -/
def topP (pats : List Nat) (q : Nat → Nat) (target : Nat) : Nat :=
  if positives pats = [] then INF
  else if target ≤ massGE pats q 1 then maxOf ((positives pats).filter (fun v => decide (target ≤ massGE pats q v)))
  else minOf (positives pats)

/-! ## Extrema of lists -/

theorem le_foldl_max_init : ∀ (l : List Nat) (a : Nat), a ≤ l.foldl max a
  | [], a => Nat.le_refl a
  | x :: xs, a => Nat.le_trans (Nat.le_max_left a x) (le_foldl_max_init xs (max a x))

theorem le_foldl_max : ∀ (l : List Nat) (a v : Nat), v ∈ l → v ≤ l.foldl max a
  | [], _, _, h => absurd h (List.not_mem_nil _)
  | x :: xs, a, v, h => by
    rcases List.mem_cons.mp h with rfl | h'
    · exact Nat.le_trans (Nat.le_max_right a v) (le_foldl_max_init xs (max a v))
    · exact le_foldl_max xs (max a x) v h'

theorem foldl_max_mem : ∀ (l : List Nat) (a : Nat), l.foldl max a ∈ l ∨ l.foldl max a = a
  | [], _ => Or.inr rfl
  | x :: xs, a => by
    simp only [List.foldl_cons]
    rcases foldl_max_mem xs (max a x) with h | h
    · exact Or.inl (List.mem_cons_of_mem _ h)
    · rw [h, Nat.max_def]
      split
      · exact Or.inl (List.mem_cons_self _ _)
      · exact Or.inr rfl

theorem mem_maxOf {l : List Nat} (h : l ≠ []) : maxOf l ∈ l := by
  unfold maxOf
  rcases foldl_max_mem l 0 with hm | hz
  · exact hm
  · cases l with
    | nil => exact absurd rfl h
    | cons x xs =>
      have hx : x ≤ (x :: xs).foldl max 0 := le_foldl_max _ 0 x (List.mem_cons_self _ _)
      have hx0 : x = 0 := by omega
      rw [hz, ← hx0]
      exact List.mem_cons_self _ _

theorem le_maxOf {l : List Nat} {v : Nat} (h : v ∈ l) : v ≤ maxOf l := le_foldl_max l 0 v h

theorem foldl_min_le_init : ∀ (l : List Nat) (a : Nat), l.foldl min a ≤ a
  | [], a => Nat.le_refl a
  | x :: xs, a => Nat.le_trans (foldl_min_le_init xs (min a x)) (Nat.min_le_left a x)

theorem foldl_min_le : ∀ (l : List Nat) (a v : Nat), v ∈ l → l.foldl min a ≤ v
  | [], _, _, h => absurd h (List.not_mem_nil _)
  | x :: xs, a, v, h => by
    rcases List.mem_cons.mp h with rfl | h'
    · exact Nat.le_trans (foldl_min_le_init xs (min a v)) (Nat.min_le_right a v)
    · exact foldl_min_le xs (min a x) v h'

theorem foldl_min_mem : ∀ (l : List Nat) (a : Nat), l.foldl min a ∈ l ∨ l.foldl min a = a
  | [], _ => Or.inr rfl
  | x :: xs, a => by
    simp only [List.foldl_cons]
    rcases foldl_min_mem xs (min a x) with h | h
    · exact Or.inl (List.mem_cons_of_mem _ h)
    · rw [h, Nat.min_def]
      split
      · exact Or.inr rfl
      · exact Or.inl (List.mem_cons_self _ _)

theorem mem_minOf {l : List Nat} (h : l ≠ []) : minOf l ∈ l := by
  cases l with
  | nil => exact absurd rfl h
  | cons x xs =>
    show List.foldl min x xs ∈ x :: xs
    rcases foldl_min_mem xs x with hm | he
    · exact List.mem_cons_of_mem _ hm
    · rw [he]
      exact List.mem_cons_self _ _

theorem minOf_le {l : List Nat} {v : Nat} (h : v ∈ l) : minOf l ≤ v := by
  cases l with
  | nil => exact absurd h (List.not_mem_nil _)
  | cons x xs =>
    show List.foldl min x xs ≤ v
    rcases List.mem_cons.mp h with rfl | h'
    · exact foldl_min_le_init xs v
    · exact foldl_min_le xs x v h'

theorem filter_ext {p q : Nat → Bool} : ∀ (l : List Nat), (∀ x ∈ l, p x = q x) → l.filter p = l.filter q
  | [], _ => rfl
  | x :: xs, h => by
    rw [List.filter_cons, List.filter_cons, h x (List.mem_cons_self _ _),
      filter_ext xs (fun y hy => h y (List.mem_cons_of_mem _ hy))]

/-! ## top-k -/

theorem countGE_min (pats : List Nat) : countGE pats (minOf pats) = pats.length := by
  unfold countGE
  exact List.countP_eq_length.mpr (fun p hp => decide_eq_true (minOf_le hp))

/-- The cut keeps at least `k` elements (`0 < k < N`; outside that range the threshold is `+inf` / `0` by definition). -/
theorem topK_keeps (pats : List Nat) (k : Nat) (h1 : 0 < k) (h2 : k < pats.length) :
    k ≤ countGE pats (topK pats k) := by
  have hne : pats ≠ [] := by
    intro h
    rw [h] at h2
    exact absurd h2 (Nat.not_lt_zero _)
  have hk0 : k ≠ 0 := by omega
  have hlen : ¬ pats.length ≤ k := by omega
  unfold topK
  rw [if_neg hk0, if_neg hlen]
  have hmem : minOf pats ∈ pats.filter (fun v => decide (k ≤ countGE pats v)) :=
    List.mem_filter.mpr ⟨mem_minOf hne, decide_eq_true (by rw [countGE_min pats]; exact Nat.le_of_lt h2)⟩
  exact of_decide_eq_true (List.mem_filter.mp (mem_maxOf (List.ne_nil_of_mem hmem))).2

/-- No cut at a larger weight of the row keeps `k`. -/
theorem topK_maximal (pats : List Nat) (k : Nat) (h1 : 0 < k) (h2 : k < pats.length) (v : Nat) (hv : v ∈ pats)
    (hgt : topK pats k < v) : countGE pats v < k := by
  have hk0 : k ≠ 0 := by omega
  have hlen : ¬ pats.length ≤ k := by omega
  unfold topK at hgt
  rw [if_neg hk0, if_neg hlen] at hgt
  refine Nat.not_le.mp ?_
  intro hc
  have hle : v ≤ maxOf (pats.filter (fun v => decide (k ≤ countGE pats v))) :=
    le_maxOf (List.mem_filter.mpr ⟨hv, decide_eq_true hc⟩)
  omega

/-- Dropping the threshold's own value leaves fewer than `k`: the kept set is the smallest upward-closed set with `k`. -/
theorem topK_drop (pats : List Nat) (k : Nat) (h1 : 0 < k) (h2 : k < pats.length) :
    countGE pats (topK pats k + 1) < k := by
  refine Nat.not_le.mp ?_
  intro hc
  have hA : pats.filter (fun p => decide (topK pats k + 1 ≤ p)) ≠ [] := by
    intro h
    have h0 : countGE pats (topK pats k + 1) = 0 := by
      unfold countGE
      rw [List.countP_eq_length_filter, h]
      rfl
    omega
  have hm := List.mem_filter.mp (mem_minOf hA)
  have hcnt : countGE pats (topK pats k + 1) ≤
      countGE pats (minOf (pats.filter (fun p => decide (topK pats k + 1 ≤ p)))) := by
    unfold countGE
    apply List.countP_mono_left
    intro p hp hpp
    exact decide_eq_true (minOf_le (List.mem_filter.mpr ⟨hp, hpp⟩))
  have hgt : topK pats k < minOf (pats.filter (fun p => decide (topK pats k + 1 ≤ p))) := by
    have := of_decide_eq_true hm.2
    omega
  have hlt := topK_maximal pats k h1 h2 _ hm.1 hgt
  omega

/-! ## top-p -/

theorem massGE_minPos (pats : List Nat) (q : Nat → Nat) (h : positives pats ≠ []) :
    massGE pats q (minOf (positives pats)) = massGE pats q 1 := by
  unfold massGE
  congr 1
  apply filter_ext
  intro x hx
  have hpos : 0 < minOf (positives pats) := of_decide_eq_true (List.mem_filter.mp (mem_minOf h)).2
  by_cases hx0 : 0 < x
  · have hle : minOf (positives pats) ≤ x := minOf_le (List.mem_filter.mpr ⟨hx, decide_eq_true hx0⟩)
    rw [decide_eq_true hle, decide_eq_true (show 1 ≤ x from hx0)]
  · have hx' : x = 0 := by omega
    subst hx'
    rw [decide_eq_false (Nat.not_le.mpr hpos), decide_eq_false (by omega : ¬ (1 ≤ 0))]

/-- When the positive mass reaches the target, so does the cut's. -/
theorem topP_reaches (pats : List Nat) (q : Nat → Nat) (target : Nat) (h : positives pats ≠ [])
    (ht : target ≤ massGE pats q 1) : target ≤ massGE pats q (topP pats q target) := by
  unfold topP
  rw [if_neg h, if_pos ht]
  have hmem : minOf (positives pats) ∈ (positives pats).filter (fun v => decide (target ≤ massGE pats q v)) :=
    List.mem_filter.mpr ⟨mem_minOf h, decide_eq_true (by rw [massGE_minPos pats q h]; exact ht)⟩
  exact of_decide_eq_true (List.mem_filter.mp (mem_maxOf (List.ne_nil_of_mem hmem))).2

/-- No cut at a larger positive weight of the row reaches the target. -/
theorem topP_maximal (pats : List Nat) (q : Nat → Nat) (target : Nat) (h : positives pats ≠ [])
    (ht : target ≤ massGE pats q 1) (v : Nat) (hv : v ∈ positives pats) (hgt : topP pats q target < v) :
    massGE pats q v < target := by
  unfold topP at hgt
  rw [if_neg h, if_pos ht] at hgt
  refine Nat.not_le.mp ?_
  intro hc
  have hle : v ≤ maxOf ((positives pats).filter (fun v => decide (target ≤ massGE pats q v))) :=
    le_maxOf (List.mem_filter.mpr ⟨hv, decide_eq_true hc⟩)
  omega

/-- When the positive mass does not reach the target, every positive weight is kept. -/
theorem topP_fallback (pats : List Nat) (q : Nat → Nat) (target : Nat) (h : positives pats ≠ [])
    (ht : massGE pats q 1 < target) (p : Nat) (hp : p ∈ pats) (hpos : 0 < p) : topP pats q target ≤ p := by
  unfold topP
  rw [if_neg h, if_neg (Nat.not_le.mpr ht)]
  exact minOf_le (List.mem_filter.mpr ⟨hp, decide_eq_true hpos⟩)

/-- top-p never keeps a zero weight. -/
theorem topP_positive (pats : List Nat) (q : Nat → Nat) (target : Nat) (h : positives pats ≠ []) :
    0 < topP pats q target := by
  unfold topP
  rw [if_neg h]
  split
  · next ht =>
    have hmem : minOf (positives pats) ∈ (positives pats).filter (fun v => decide (target ≤ massGE pats q v)) :=
      List.mem_filter.mpr ⟨mem_minOf h, decide_eq_true (by rw [massGE_minPos pats q h]; exact ht)⟩
    have hin := List.mem_filter.mp (List.mem_filter.mp (mem_maxOf (List.ne_nil_of_mem hmem))).1
    exact of_decide_eq_true hin.2
  · exact of_decide_eq_true (List.mem_filter.mp (mem_minOf h)).2

theorem topP_none (pats : List Nat) (q : Nat → Nat) (target : Nat) (h : positives pats = []) :
    topP pats q target = INF := by
  unfold topP
  rw [if_pos h]

/-! ## The effective threshold -/

/-- Two cuts intersect in the cut at the larger threshold: the kernel's one comparison against `max` of the applied
thresholds is the intersection of the filters' kept sets. -/
theorem cut_inter (a b p : Nat) : (a ≤ p ∧ b ≤ p) ↔ max a b ≤ p := Nat.max_le.symm

/-! ## The kept mass: `lse.row`'s tree over the kept weights -/

/-- The slot fold over the kept weights, the dropped ones `+0` (absent elements dropped, as `Sampling.slotSum`). -/
def maskedSlotSum (N : Nat) (w : Nat → Term) (keep : Nat → Bool) (s : Nat) : Option Term :=
  let js := (List.range 8).filter (fun j => s * 8 + j < N)
  match js with
  | [] => none
  | _ => some (js.foldl (fun acc j => Term.add acc (if keep (s * 8 + j) then w (s * 8 + j) else Term.const "0.0"))
      (Term.const "0.0"))

def maskedLeaf (N kb : Nat) (w : Nat → Term) (keep : Nat → Bool) (bs : List Bool) : Option Term :=
  maskedSlotSum N w keep (bitsToNat (takeN kb bs) * 32 + bitsToNat (dropN kb bs))

/-- The spec's kept mass: the canonical `add` tree over the masked leaves. -/
def massTree (N kb : Nat) (w : Nat → Term) (keep : Nat → Bool) : Option Term :=
  canonicalDigitsC .add [kb, 5] (maskedLeaf N kb w keep)

/-- What the kernel's masked pass computes under placement `ps` at coordinates `cs`. -/
def kernelMassTree (N kb : Nat) (w : Nat → Term) (keep : Nat → Bool) (ps : List (List Factor))
    (cs : List (List Bool)) : Option Term :=
  execC .add (flatten ps) (maskedLeaf N kb w keep) cs

theorem mass_invariance (N kb : Nat) (w : Nat → Term) (keep : Nat → Bool) (ps : List (List Factor))
    (cs : List (List Bool)) (hp : PlacementOK [kb, 5] ps) (hc : CoordsOK (flatten ps) cs) :
    OEquiv (kernelMassTree N kb w keep ps cs) (massTree N kb w keep) :=
  config_invarianceC .add [kb, 5] ps (maskedLeaf N kb w keep) cs hp hc

/-- Keeping everything gives the row's own tree (the draw's total). -/
theorem massTree_all (N kb : Nat) (w : Nat → Term) : massTree N kb w (fun _ => true) = Draw.total N kb w := rfl

/-- The renormalised probability of a kept element: `lse.row`'s `prob` chain (`mul(w, div(1, s))`) with the kept mass
as `s`. -/
def renormProb (N kb : Nat) (w : Nat → Term) (keep : Nat → Bool) (j : Nat) : Option Term :=
  (massTree N kb w keep).map (fun m => Sampling.prob (w j) (Sampling.invSum m))

/-! ## The kept-set log-probability of a token: `lse.row`'s `logp` chain over the kept mass -/

/-- The log-probability of token `j` under the kept set -- the drawn token's `kept_logp` and the trainer-side
`log_probs(logits, ids, top_k / top_p / min_p)` at a given token (`support_spec.kept_logp_term`): `lse.row`'s `logp` chain
at `j` with the kept mass in place of the row sum, `mul(add(z_j, neg(lg2(mass))), LN2)`, `z` the shifted logits. -/
def keptLogp (N kb : Nat) (w z : Nat → Term) (keep : Nat → Bool) (j : Nat) : Option Term :=
  (massTree N kb w keep).map (fun m => Sampling.logp (z j) (Term.neg (Sampling.lg2 m)))

/-- What the kernel's score form computes under placement `ps` at coordinates `cs`: the same chain over its masked pass. -/
def kernelKeptLogp (N kb : Nat) (w z : Nat → Term) (keep : Nat → Bool) (ps : List (List Factor))
    (cs : List (List Bool)) (j : Nat) : Option Term :=
  (kernelMassTree N kb w keep ps cs).map (fun m => Sampling.logp (z j) (Term.neg (Sampling.lg2 m)))

/-- Every valid placement's kept-set log-probability is `Equiv` to the spec's: the chain is a congruence of the mass tree,
whose invariance is `mass_invariance`. -/
theorem keptLogp_invariance (N kb : Nat) (w z : Nat → Term) (keep : Nat → Bool) (ps : List (List Factor))
    (cs : List (List Bool)) (hp : PlacementOK [kb, 5] ps) (hc : CoordsOK (flatten ps) cs) (j : Nat) :
    OEquiv (kernelKeptLogp N kb w z keep ps cs j) (keptLogp N kb w z keep j) := by
  have h := mass_invariance N kb w keep ps cs hp hc
  unfold kernelKeptLogp keptLogp
  generalize kernelMassTree N kb w keep ps cs = o at h ⊢
  generalize massTree N kb w keep = p at h ⊢
  cases h with
  | absent => exact OEquiv.absent
  | present he =>
    exact OEquiv.present (Equiv.mul_congr (Equiv.add_congr (Equiv.refl _) (Equiv.neg_congr (Equiv.opq_congr "lg2" he))) (Equiv.refl _))

/-- With everything kept the mass is the row's own tree, so the kept-set log-probability is `lse.row`'s `logp` over the
draw's total: `top_p = 1` / `top_k >= N` give `log_probs` (the GPU suite pins the bits). -/
theorem keptLogp_all (N kb : Nat) (w z : Nat → Term) (j : Nat) :
    keptLogp N kb w z (fun _ => true) j
      = (Draw.total N kb w).map (fun m => Sampling.logp (z j) (Term.neg (Sampling.lg2 m))) := rfl

end SupportRow
end BIK
