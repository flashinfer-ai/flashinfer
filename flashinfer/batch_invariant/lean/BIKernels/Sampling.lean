import BIKernels.Canon
import BIKernels.Semantics
import BIKernels.Support
/-!
# `lse.row` on the engine: the end-to-end statement

Instantiates the engine theorems for the sampling class behind `softmax(batch_invariant=True)` and `log_probs`
(`sampling/spec.py`, `sampling/schedule.py`):

* the two row reductions -- the maximum (`fmaxc`) and the sum of exponentials (`add`) -- have digits `[kb, 5]`
  (block bits, then the 32 lanes) with the 8-element slot fold folded into the leaf; the sum's leaves are the chains
  `ex2(mul(add(x, neg(m)), c))` of the maximum;
* `LSEConfig.placement` (`T >= 32`: block = reg x warp-exchange [x cluster-exchange], lane = lanes) is a placement of
  those digits;
* `max_invariance` / `sum_invariance`: for every valid placement and unit the kernel's trees are `Equiv` to the spec's;
  `lse_end_to_end`: the three outputs (the probabilities, the log-softmax elements, the natural `lse`) built from the
  kernel's statistics are `Equiv` to the spec's, hence (`eval_equiv`) bitwise equal under any commutative
  interpretation of the primitives (`lse_bits_equal`);
* `max_padded_bits` / `sum_padded_bits`: the *bucket kernel* (every slot present, elements beyond the runtime `N`
  reading `-inf`) computes the spec's statistics bit for bit under the restricted identity hypotheses the spec's
  `identity_note`s claim (`fmaxc(v, -inf) = v` on non-NaN `v`; the weight of a `-inf` logit is an additive identity
  on the non-negative partial sums).

`sampling/lean_cases.py` couples the printed terms to the Python spec's S-expressions (`SamplingCases.lean`,
`lake exe samplingconformance`).
-/
namespace BIK
namespace Sampling

/-! ## The maximum -/

/-- `Digit elem` (fold) of the maximum: the `fmaxc` chain from `-inf` over the present elements of slot
`s`; absent when the slot is entirely beyond `N`. -/
def slotMax (N : Nat) (row : Nat → Term) (s : Nat) : Option Term :=
  let js := (List.range 8).filter (fun j => s * 8 + j < N)
  match js with
  | [] => none
  | _ => some (js.foldl (fun acc j => Term.fmaxc acc (row (s*8+j))) (Term.const "-inf"))

/-- Leaf of the pairwise part: bits = block bits (`kb`, combined first) ++ lane bits (5); slot `s = block * 32 + lane`. -/
def maxLeaf (N kb : Nat) (row : Nat → Term) (bs : List Bool) : Option Term :=
  slotMax N row (bitsToNat (takeN kb bs) * 32 + bitsToNat (dropN kb bs))

/-- The spec: `lse.row.v1.max` as `Reduction.canonical`. -/
def specMax (N kb : Nat) (row : Nat → Term) : Option Term :=
  canonicalDigitsC .fmaxc [kb, 5] (maxLeaf N kb row)

/-- What the kernel's first pass computes (`canon.init_state` + `canon.run` with the `fmaxc` digits). -/
def kernelMax (N kb : Nat) (row : Nat → Term) (ps : List (List Factor)) (cs : List (List Bool)) : Option Term :=
  execC .fmaxc (flatten ps) (maxLeaf N kb row) cs

theorem max_invariance (N kb : Nat) (row : Nat → Term) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [kb, 5] ps) (hc : CoordsOK (flatten ps) cs) :
    OEquiv (kernelMax N kb row ps cs) (specMax N kb row) :=
  config_invarianceC .fmaxc [kb, 5] ps (maxLeaf N kb row) cs hp hc

/-- `LSEConfig.placement`: W = 2^wb warps own aligned block ranges (T >= 32 always). -/
def placement (kb wb : Nat) : List (List Factor) :=
  [[⟨kb - wb, .reg⟩, ⟨wb, .xchg⟩], [⟨5, .lane⟩]]

theorem placement_ok (kb wb : Nat) (h : wb ≤ kb) : PlacementOK [kb, 5] (placement kb wb) := by
  refine ⟨?_, ?_, trivial⟩
  · show kb - wb + (wb + 0) = kb
    omega
  · rfl

/-- `LSEConfig.placement` with a cluster: C = 2^cb CTAs own aligned block ranges of the row, W = 2^wb warps aligned
sub-ranges within each (block = reg x warp-exchange x cluster-exchange); the per-lane partials cross the cluster through
distributed shared memory. The same digits, so the same tree. -/
def placementCluster (kb wb cb : Nat) : List (List Factor) :=
  [[⟨kb - wb - cb, .reg⟩, ⟨wb, .xchg⟩, ⟨cb, .xchg⟩], [⟨5, .lane⟩]]

theorem placementCluster_ok (kb wb cb : Nat) (h : wb + cb ≤ kb) : PlacementOK [kb, 5] (placementCluster kb wb cb) := by
  refine ⟨?_, ?_, trivial⟩
  · show kb - wb - cb + (wb + (cb + 0)) = kb
    omega
  · rfl

/-! ## The chains (`spec.scale / shifted / weight / inv_sum / prob / lg2 / lse_nat / logp`)

Python prints the binary32 constants with `repr`: `LOG2E = 1.4426950216293335`, `LN2 = 0.6931471824645996`. -/

def log2e : Term := .const "1.4426950216293335"
def ln2 : Term := .const "0.6931471824645996"

/-- `c`: `LOG2E` without a temperature, `div(LOG2E, T)` with one. -/
def scale : Option Term → Term
  | none => log2e
  | some T => .div log2e T

/-- `z = mul(add(x, nm), c)` with `nm = neg(m)`. -/
def shifted (x nm c : Term) : Term := .mul (.add x nm) c

/-- `e = EX2(z)`. -/
def weight (z : Term) : Term := .ex2 z

def invSum (s : Term) : Term := .div (.const "1.0") s
def prob (e invs : Term) : Term := .mul e invs
def lg2 (s : Term) : Term := .opq "lg2" s
def lseNat (m c lg2s : Term) : Term := .mul (.add (.mul m c) lg2s) ln2
def logp (z nlg : Term) : Term := .mul (.add z nlg) ln2

/-! ## The sum of exponentials -/

/-- `Digit elem` (fold) of the sum: the `add` chain from `+0` over the weights of the present elements of slot `s`. -/
def slotSum (N : Nat) (row : Nat → Term) (nm c : Term) (s : Nat) : Option Term :=
  let js := (List.range 8).filter (fun j => s * 8 + j < N)
  match js with
  | [] => none
  | _ => some (js.foldl (fun acc j => Term.add acc (weight (shifted (row (s*8+j)) nm c))) (Term.const "0.0"))

def sumLeaf (N kb : Nat) (row : Nat → Term) (nm c : Term) (bs : List Bool) : Option Term :=
  slotSum N row nm c (bitsToNat (takeN kb bs) * 32 + bitsToNat (dropN kb bs))

/-- The spec: `lse.row.v1.sum` as `Reduction.canonical`, for the given (negated) maximum and scale. -/
def specSum (N kb : Nat) (row : Nat → Term) (nm c : Term) : Option Term :=
  canonicalDigitsC .add [kb, 5] (sumLeaf N kb row nm c)

/-- What the kernel's second pass computes. -/
def kernelSum (N kb : Nat) (row : Nat → Term) (nm c : Term) (ps : List (List Factor)) (cs : List (List Bool)) : Option Term :=
  execC .add (flatten ps) (sumLeaf N kb row nm c) cs

theorem sum_invariance (N kb : Nat) (row : Nat → Term) (nm c : Term) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [kb, 5] ps) (hc : CoordsOK (flatten ps) cs) :
    OEquiv (kernelSum N kb row nm c ps cs) (specSum N kb row nm c) :=
  config_invarianceC .add [kb, 5] ps (sumLeaf N kb row nm c) cs hp hc

/-- The sum's leaves are congruent in the negated maximum: `Equiv nm nm'` lifts through every slot fold. -/
theorem slotSum_congr (N : Nat) (row : Nat → Term) {nm nm' : Term} (c : Term) (h : Equiv nm nm') (s : Nat) :
    OEquiv (slotSum N row nm c s) (slotSum N row nm' c s) := by
  unfold slotSum
  have key : ∀ (js : List Nat) (acc acc' : Term), Equiv acc acc' →
      Equiv (js.foldl (fun acc j => Term.add acc (weight (shifted (row (s*8+j)) nm c))) acc)
            (js.foldl (fun acc j => Term.add acc (weight (shifted (row (s*8+j)) nm' c))) acc') := by
    intro js
    induction js with
    | nil => intro acc acc' ha; exact ha
    | cons j js ih =>
        intro acc acc' ha
        simp only [List.foldl]
        apply ih
        exact .add_congr ha (.ex2_congr (.mul_congr (.add_congr (.refl _) h) (.refl _)))
  cases (List.range 8).filter (fun j => s * 8 + j < N) with
  | nil => exact .absent
  | cons j js => exact .present (key (j :: js) _ _ (.refl _))

theorem specSum_congr (N kb : Nat) (row : Nat → Term) {nm nm' : Term} (c : Term) (h : Equiv nm nm') :
    OEquiv (specSum N kb row nm c) (specSum N kb row nm' c) := by
  unfold specSum
  rw [canonicalDigitsC_eq_treeC, canonicalDigitsC_eq_treeC]
  exact treeC_congr .add _ (fun bs => slotSum_congr N row c h _)

/-! ## The outputs -/

/-- The spec's outputs of a row from its statistics `(m, s)` and scale `c`. -/
structure Outputs where
  probs : Nat → Term      -- `cvt(fmt, mul(ex2(z_i), div(1, s)))`
  logprobs : Nat → Term   -- `mul(add(z_i, neg(lg2(s))), LN2)`
  lse : Term              -- `mul(add(mul(m, c), lg2(s)), LN2)`

def outputsOf (fmt : String) (row : Nat → Term) (m s c : Term) : Outputs :=
  let nm := Term.neg m
  let invs := invSum s
  let nlg := Term.neg (lg2 s)
  { probs := fun i => .cvt fmt (prob (weight (shifted (row i) nm c)) invs),
    logprobs := fun i => logp (shifted (row i) nm c) nlg,
    lse := lseNat m c (lg2 s) }

/-- The spec's outputs (`LSESpec.outputs(row)`): absent only for an empty row. -/
def specOutputs (fmt : String) (N kb : Nat) (row : Nat → Term) (T : Option Term) : Option Outputs :=
  match specMax N kb row with
  | none => none
  | some m =>
    match specSum N kb row (.neg m) (scale T) with
    | none => none
    | some s => some (outputsOf fmt row m s (scale T))

/-- What the kernel writes: the same chains over its own two trees. -/
def kernelOutputs (fmt : String) (N kb : Nat) (row : Nat → Term) (T : Option Term) (ps : List (List Factor)) (cs : List (List Bool)) : Option Outputs :=
  match kernelMax N kb row ps cs with
  | none => none
  | some m =>
    match kernelSum N kb row (.neg m) (scale T) ps cs with
    | none => none
    | some s => some (outputsOf fmt row m s (scale T))

/-- The second pass as launched: the kernel's sum over its own maximum (`nm = neg(m')`). -/
def kernelSumOfMax (N kb : Nat) (row : Nat → Term) (T : Option Term) (ps : List (List Factor)) (cs : List (List Bool)) : Option Term :=
  match kernelMax N kb row ps cs with
  | none => none
  | some m => kernelSum N kb row (.neg m) (scale T) ps cs

/-- The spec's sum over the spec's maximum (`LSESpec.s`). -/
def specSumOfMax (N kb : Nat) (row : Nat → Term) (T : Option Term) : Option Term :=
  match specMax N kb row with
  | none => none
  | some m => specSum N kb row (.neg m) (scale T)

/-- The spec's output terms, one at a time (printed by the conformance executable). -/
def specProb (fmt : String) (N kb : Nat) (row : Nat → Term) (T : Option Term) (i : Nat) : Option Term :=
  (specOutputs fmt N kb row T).map (fun o => o.probs i)
def specLogp (fmt : String) (N kb : Nat) (row : Nat → Term) (T : Option Term) (i : Nat) : Option Term :=
  (specOutputs fmt N kb row T).map (fun o => o.logprobs i)
def specLse (fmt : String) (N kb : Nat) (row : Nat → Term) (T : Option Term) : Option Term :=
  (specOutputs fmt N kb row T).map (fun o => o.lse)

theorem outputs_congr (fmt : String) (row : Nat → Term) {m m' s s' : Term} (c : Term) (hm : Equiv m m') (hs : Equiv s s') :
    (∀ i, Equiv ((outputsOf fmt row m s c).probs i) ((outputsOf fmt row m' s' c).probs i))
    ∧ (∀ i, Equiv ((outputsOf fmt row m s c).logprobs i) ((outputsOf fmt row m' s' c).logprobs i))
    ∧ Equiv (outputsOf fmt row m s c).lse (outputsOf fmt row m' s' c).lse := by
  refine ⟨fun i => ?_, fun i => ?_, ?_⟩
  · exact .cvt_congr fmt (.mul_congr (.ex2_congr (.mul_congr (.add_congr (.refl _) (.neg_congr hm)) (.refl _)))
      (.div_congr (.refl _) hs))
  · exact .mul_congr (.add_congr (.mul_congr (.add_congr (.refl _) (.neg_congr hm)) (.refl _)) (.neg_congr (.opq_congr "lg2" hs))) (.refl _)
  · exact .mul_congr (.add_congr (.mul_congr hm (.refl _)) (.opq_congr "lg2" hs)) (.refl _)

/-- Equivalence of two optional output records: absent together, or present with every output `Equiv`. -/
def OutputsEquiv : Option Outputs → Option Outputs → Prop
  | none, none => True
  | some a, some b => (∀ i, Equiv (a.probs i) (b.probs i)) ∧ (∀ i, Equiv (a.logprobs i) (b.logprobs i)) ∧ Equiv a.lse b.lse
  | _, _ => False

/-- **End-to-end `lse.row` (absent-dropping model).** For every valid placement and unit, every output the
kernel writes is `Equiv` to the spec's: the kernel's maximum is `Equiv` to the spec's (`max_invariance`), its sum
over its own negated maximum is `Equiv` to the spec's sum over the same (`sum_invariance`), which is `Equiv` to
the spec's sum over the spec's maximum (`specSum_congr`), and the chains are congruences. -/
theorem lse_end_to_end (fmt : String) (N kb : Nat) (row : Nat → Term) (T : Option Term) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [kb, 5] ps) (hc : CoordsOK (flatten ps) cs) :
    OutputsEquiv (kernelOutputs fmt N kb row T ps cs) (specOutputs fmt N kb row T) := by
  have hm := max_invariance N kb row ps cs hp hc
  unfold kernelOutputs specOutputs
  cases hkm : kernelMax N kb row ps cs with
  | none =>
      cases hsm : specMax N kb row with
      | none => trivial
      | some m => rw [hkm, hsm] at hm; cases hm
  | some m' =>
      cases hsm : specMax N kb row with
      | none => rw [hkm, hsm] at hm; cases hm
      | some m =>
          rw [hkm, hsm] at hm
          have hmm : Equiv m' m := by cases hm with | present h => exact h
          have hs1 := sum_invariance N kb row (.neg m') (scale T) ps cs hp hc
          have hs2 := specSum_congr N kb row (scale T) (Equiv.neg_congr hmm)
          have hs := hs1.trans hs2
          dsimp only
          cases hks : kernelSum N kb row (.neg m') (scale T) ps cs with
          | none =>
              cases hss : specSum N kb row (.neg m) (scale T) with
              | none => trivial
              | some s => rw [hks, hss] at hs; cases hs
          | some s' =>
              cases hss : specSum N kb row (.neg m) (scale T) with
              | none => rw [hks, hss] at hs; cases hs
              | some s =>
                  rw [hks, hss] at hs
                  have hss' : Equiv s' s := by cases hs with | present h => exact h
                  exact outputs_congr fmt row (scale T) hmm hss'

/-- Bitwise form: under any commutative interpretation of the primitives, the kernel's and the spec's outputs are
the same values (`eval_equiv`). -/
theorem lse_bits_equal {α : Type} (P : Prims α) (fmt : String) (N kb : Nat) (row : Nat → Term) (T : Option Term)
    (ps : List (List Factor)) (cs : List (List Bool)) (hp : PlacementOK [kb, 5] ps) (hc : CoordsOK (flatten ps) cs)
    {k o : Outputs} (hk : kernelOutputs fmt N kb row T ps cs = some k) (ho : specOutputs fmt N kb row T = some o) :
    (∀ i, eval P (k.probs i) = eval P (o.probs i)) ∧ (∀ i, eval P (k.logprobs i) = eval P (o.logprobs i)) ∧ eval P k.lse = eval P o.lse := by
  have h := lse_end_to_end fmt N kb row T ps cs hp hc
  rw [hk, ho] at h
  unfold OutputsEquiv at h
  obtain ⟨h1, h2, h3⟩ := h
  exact ⟨fun i => eval_equiv P (h1 i), fun i => eval_equiv P (h2 i), eval_equiv P h3⟩

/-! ## The bucket kernel: `-inf` padding under the restricted identity

The kernel fills every absent element with `-inf` (`_RowLoader.block`), so each slot's fold runs over all 8
positions and every slot of every block up to `Kp` is present. For the maximum, `-inf` is an identity of `fmaxc`
on every non-NaN value (`Good`); for the sum, the padded element's weight `ex2(mul(add(-inf, nm), c))` is an
additive identity on the non-negative partial sums (`Good`: not `-0`). Both are the spec's `identity_note`s,
stated as hypotheses on the interpretation. -/

/-- The row extended by `-inf` beyond `N`. -/
def padRowInf (N : Nat) (row : Nat → Term) : Nat → Term := fun i => if i < N then row i else .const "-inf"

def slotMaxPadded (N : Nat) (row : Nat → Term) (s : Nat) : Term :=
  (List.range 8).foldl (fun acc j => Term.fmaxc acc (padRowInf N row (s*8+j))) (Term.const "-inf")

def maxLeafPadded (N kb : Nat) (row : Nat → Term) (bs : List Bool) : Option Term :=
  some (slotMaxPadded N row (bitsToNat (takeN kb bs) * 32 + bitsToNat (dropN kb bs)))

def kernelMaxPadded (N kb : Nat) (row : Nat → Term) (ps : List (List Factor)) (cs : List (List Bool)) : Option Term :=
  execC .fmaxc (flatten ps) (maxLeafPadded N kb row) cs

def slotSumPadded (N : Nat) (row : Nat → Term) (nm c : Term) (s : Nat) : Term :=
  (List.range 8).foldl (fun acc j => Term.add acc (weight (shifted (padRowInf N row (s*8+j)) nm c))) (Term.const "0.0")

def sumLeafPadded (N kb : Nat) (row : Nat → Term) (nm c : Term) (bs : List Bool) : Option Term :=
  some (slotSumPadded N row nm c (bitsToNat (takeN kb bs) * 32 + bitsToNat (dropN kb bs)))

def kernelSumPadded (N kb : Nat) (row : Nat → Term) (nm c : Term) (ps : List (List Factor)) (cs : List (List Bool)) : Option Term :=
  execC .add (flatten ps) (sumLeafPadded N kb row nm c) cs

section Padded
variable {α : Type} (P : Prims α) (Good : α → Prop)

/-- **Padded fold steps are exact, for any step.** A left fold `acc := f acc (pv j)` over all positions `js`,
where the absent positions (`p j = false`) are absorbed (`f a (pv j) = a` on Good `a`) and the present ones hold
the spec's value, equals the spec's fold over the present positions; Goodness is preserved from the start value.
(`foldl_fma_pad` of `Semantics.lean` is the `fma x x ·` instance.) -/
theorem foldl_step_pad (f : α → α → α) (pv rv : Nat → α) (p : Nat → Bool)
    (hstep : ∀ a j, Good a → Good (f a (rv j)))
    (hpad : ∀ j, p j = false → ∀ a, Good a → f a (pv j) = a)
    (hpres : ∀ j, p j = true → pv j = rv j) :
    ∀ (js : List Nat) (a : α), Good a →
      js.foldl (fun a j => f a (pv j)) a = (js.filter p).foldl (fun a j => f a (rv j)) a
      ∧ Good ((js.filter p).foldl (fun a j => f a (rv j)) a)
  | [], _, ha => ⟨rfl, ha⟩
  | j :: js, a, ha => by
      cases hp : p j with
      | true =>
          have ih := foldl_step_pad f pv rv p hstep hpad hpres js (f a (rv j)) (hstep a j ha)
          simp only [List.filter, hp, List.foldl, hpres j hp]
          exact ih
      | false =>
          have ih := foldl_step_pad f pv rv p hstep hpad hpres js a ha
          simp only [List.filter, hp, List.foldl, hpad j hp a ha]
          exact ih

theorem eval_foldl_fmaxc (r : Nat → Term) :
    ∀ (js : List Nat) (acc : Term),
      eval P (js.foldl (fun acc j => Term.fmaxc acc (r j)) acc) = js.foldl (fun a j => P.fmaxc a (eval P (r j))) (eval P acc)
  | [], _ => rfl
  | j :: js, acc => by
      simp only [List.foldl]
      exact eval_foldl_fmaxc r js (Term.fmaxc acc (r j))

theorem eval_foldl_addw (r : Nat → Term) (nm c : Term) :
    ∀ (js : List Nat) (acc : Term),
      eval P (js.foldl (fun acc j => Term.add acc (weight (shifted (r j) nm c))) acc)
        = js.foldl (fun a j => P.add a (eval P (weight (shifted (r j) nm c)))) (eval P acc)
  | [], _ => rfl
  | j :: js, acc => by
      simp only [List.foldl]
      exact eval_foldl_addw r nm c js (Term.add acc (weight (shifted (r j) nm c)))

/-- **The padded max slot equals the spec's slot.** Hypotheses: `-inf` is Good; `fmaxc` keeps Good values Good;
`fmaxc(a, -inf) = a` on Good `a`. -/
theorem slotMax_pad_good (N : Nat) (row : Nat → Term) (s : Nat)
    (hGz : Good (P.cv "-inf"))
    (hcl : ∀ a b, Good a → Good (P.fmaxc a b))
    (hid : ∀ a, Good a → P.fmaxc a (P.cv "-inf") = a) :
    eval P (slotMaxPadded N row s) = evalO P (P.cv "-inf") (slotMax N row s)
    ∧ Good (evalO P (P.cv "-inf") (slotMax N row s)) := by
  have key := foldl_step_pad Good P.fmaxc (fun j => eval P (padRowInf N row (s*8+j))) (fun j => eval P (row (s*8+j)))
      (fun j => decide (s*8+j < N)) (fun a j ha => hcl a _ ha)
      (fun j hj a ha => by
        have hn : ¬ (s*8+j < N) := of_decide_eq_false hj
        show P.fmaxc a (eval P (padRowInf N row (s*8+j))) = a
        simp only [padRowInf, if_neg hn]
        exact hid a ha)
      (fun j hj => by
        have hy : s*8+j < N := of_decide_eq_true hj
        show eval P (padRowInf N row (s*8+j)) = eval P (row (s*8+j))
        simp only [padRowInf, if_pos hy])
      (List.range 8) (P.cv "-inf") hGz
  have hl : eval P (slotMaxPadded N row s)
      = (List.range 8).foldl (fun a j => P.fmaxc a (eval P (padRowInf N row (s*8+j)))) (P.cv "-inf") :=
    eval_foldl_fmaxc P (fun j => padRowInf N row (s*8+j)) (List.range 8) (Term.const "-inf")
  have hr : evalO P (P.cv "-inf") (slotMax N row s)
      = ((List.range 8).filter (fun j => decide (s*8+j < N))).foldl (fun a j => P.fmaxc a (eval P (row (s*8+j)))) (P.cv "-inf") := by
    simp only [slotMax]
    cases h : (List.range 8).filter (fun j => decide (s*8+j < N)) with
    | nil => rfl
    | cons j js => exact eval_foldl_fmaxc P (fun j => row (s*8+j)) (j :: js) (Term.const "-inf")
  rw [hl, hr]
  exact key

/-- **The bucket kernel's maximum equals the spec's, bit for bit, under the restricted identity.** -/
theorem max_padded_bits (N kb : Nat) (row : Nat → Term) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [kb, 5] ps) (hc : CoordsOK (flatten ps) cs)
    (hGz : Good (P.cv "-inf"))
    (hcl : ∀ a b, Good a → Good b → Good (P.fmaxc a b))
    (hcl1 : ∀ a b, Good a → Good (P.fmaxc a b))
    (hid : ∀ a, Good a → P.fmaxc (P.cv "-inf") a = a ∧ P.fmaxc a (P.cv "-inf") = a) :
    evalO P (P.cv "-inf") (kernelMaxPadded N kb row ps cs) = evalO P (P.cv "-inf") (specMax N kb row)
    ∧ Good (evalO P (P.cv "-inf") (specMax N kb row)) :=
  execC_pad_good P Good (P.cv "-inf") .fmaxc hGz hcl hid [kb, 5] ps (maxLeafPadded N kb row) (maxLeaf N kb row) cs hp hc
    (fun _ => (slotMax_pad_good P Good N row _ hGz hcl1 (fun a ha => (hid a ha).2)).1)
    (fun _ => (slotMax_pad_good P Good N row _ hGz hcl1 (fun a ha => (hid a ha).2)).2)

/-- **The padded sum slot equals the spec's slot.** Hypotheses: `+0` is Good; adding a weight keeps Good values
Good; the weight of a `-inf` logit is an additive identity on Good values (`ex2(-inf) = +0`, `add(v, +0) = v`). -/
theorem slotSum_pad_good (N : Nat) (row : Nat → Term) (nm c : Term) (s : Nat)
    (hG0 : Good (P.cv "0.0"))
    (hcl : ∀ a x, Good a → Good (P.add a (P.ex2 x)))
    (hw : ∀ a, Good a → P.add a (eval P (weight (shifted (.const "-inf") nm c))) = a) :
    eval P (slotSumPadded N row nm c s) = evalO P (P.cv "0.0") (slotSum N row nm c s)
    ∧ Good (evalO P (P.cv "0.0") (slotSum N row nm c s)) := by
  have key := foldl_step_pad Good P.add (fun j => eval P (weight (shifted (padRowInf N row (s*8+j)) nm c)))
      (fun j => eval P (weight (shifted (row (s*8+j)) nm c)))
      (fun j => decide (s*8+j < N)) (fun a j ha => hcl a _ ha)
      (fun j hj a ha => by
        have hn : ¬ (s*8+j < N) := of_decide_eq_false hj
        show P.add a (eval P (weight (shifted (padRowInf N row (s*8+j)) nm c))) = a
        simp only [padRowInf, if_neg hn]
        exact hw a ha)
      (fun j hj => by
        have hy : s*8+j < N := of_decide_eq_true hj
        show eval P (weight (shifted (padRowInf N row (s*8+j)) nm c)) = eval P (weight (shifted (row (s*8+j)) nm c))
        simp only [padRowInf, if_pos hy])
      (List.range 8) (P.cv "0.0") hG0
  have hl : eval P (slotSumPadded N row nm c s)
      = (List.range 8).foldl (fun a j => P.add a (eval P (weight (shifted (padRowInf N row (s*8+j)) nm c)))) (P.cv "0.0") :=
    eval_foldl_addw P (fun j => padRowInf N row (s*8+j)) nm c (List.range 8) (Term.const "0.0")
  have hr : evalO P (P.cv "0.0") (slotSum N row nm c s)
      = ((List.range 8).filter (fun j => decide (s*8+j < N))).foldl (fun a j => P.add a (eval P (weight (shifted (row (s*8+j)) nm c)))) (P.cv "0.0") := by
    simp only [slotSum]
    cases h : (List.range 8).filter (fun j => decide (s*8+j < N)) with
    | nil => rfl
    | cons j js => exact eval_foldl_addw P (fun j => row (s*8+j)) nm c (j :: js) (Term.const "0.0")
  rw [hl, hr]
  exact key

/-- **The bucket kernel's sum equals the spec's, bit for bit, under the restricted identity.** -/
theorem sum_padded_bits (N kb : Nat) (row : Nat → Term) (nm c : Term) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [kb, 5] ps) (hc : CoordsOK (flatten ps) cs)
    (hG0 : Good (P.cv "0.0"))
    (hadd : ∀ a b, Good a → Good b → Good (P.add a b))
    (hid : ∀ a, Good a → P.add (P.cv "0.0") a = a ∧ P.add a (P.cv "0.0") = a)
    (hcl : ∀ a x, Good a → Good (P.add a (P.ex2 x)))
    (hw : ∀ a, Good a → P.add a (eval P (weight (shifted (.const "-inf") nm c))) = a) :
    evalO P (P.cv "0.0") (kernelSumPadded N kb row nm c ps cs) = evalO P (P.cv "0.0") (specSum N kb row nm c)
    ∧ Good (evalO P (P.cv "0.0") (specSum N kb row nm c)) :=
  execC_pad_good P Good (P.cv "0.0") .add hG0 hadd hid [kb, 5] ps (sumLeafPadded N kb row nm c) (sumLeaf N kb row nm c) cs hp hc
    (fun _ => (slotSum_pad_good P Good N row nm c _ hG0 hcl hw).1)
    (fun _ => (slotSum_pad_good P Good N row nm c _ hG0 hcl hw).2)

end Padded

end Sampling
end BIK
