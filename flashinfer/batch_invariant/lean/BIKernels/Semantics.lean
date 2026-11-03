import BIKernels.Canon
/-!
# From term equivalence to bit equality

`Prims α` is any interpretation of the primitives on a carrier `α` (e.g. the IEEE
binary32 bit patterns of one GPU architecture) in which `add` and `mul` are
commutative. `eval_equiv` says that `Equiv`-related terms evaluate to the *same
element of α*: tree equality up to commutativity is bit equality.

`tree_pad` is the identity-padding rule used by the kernels (I6/I9): if absent leaves
are replaced by a term that evaluates to an additive identity, the padded tree
evaluates like the tree that drops them. This is what makes a kernel compiled for a
`Kp` bucket exact for every hidden size in the bucket.
-/
namespace BIK

structure Prims (α : Type) where
  xv : Nat → α
  wv : Nat → α
  rv : Nat → α
  cv : String → α
  sv : String → α
  add : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  div : α → α → α
  sqrt : α → α
  cvt : String → α → α
  fabs : α → α
  fmin : α → α → α
  fmaxc : α → α → α
  nzor : α → α → α
  add_comm : ∀ a b, add a b = add b a
  mul_comm : ∀ a b, mul a b = mul b a
  fmaxc_comm : ∀ a b, fmaxc a b = fmaxc b a

variable {α : Type}

/-- The interpretation of a `Comb`. -/
def Prims.comb (P : Prims α) : Comb → α → α → α
  | .add => P.add
  | .fmaxc => P.fmaxc

def eval (P : Prims α) : Term → α
  | .x i => P.xv i
  | .w i => P.wv i
  | .r i => P.rv i
  | .const c => P.cv c
  | .sym s => P.sv s
  | .add a b => P.add (eval P a) (eval P b)
  | .mul a b => P.mul (eval P a) (eval P b)
  | .fma a b c => P.fma (eval P a) (eval P b) (eval P c)
  | .div a b => P.div (eval P a) (eval P b)
  | .sqrt a => P.sqrt (eval P a)
  | .cvt f a => P.cvt f (eval P a)
  | .fabs a => P.fabs (eval P a)
  | .fmin a b => P.fmin (eval P a) (eval P b)
  | .fmaxc a b => P.fmaxc (eval P a) (eval P b)
  | .nzor a b => P.nzor (eval P a) (eval P b)

theorem eval_comb (P : Prims α) (c : Comb) (a b : Term) : eval P (c.app a b) = P.comb c (eval P a) (eval P b) := by
  cases c <;> rfl

/-- Equivalent terms are bitwise equal under any commutative interpretation. -/
theorem eval_equiv (P : Prims α) {a b : Term} (h : Equiv a b) : eval P a = eval P b := by
  induction h with
  | refl a => rfl
  | symm _ ih => exact ih.symm
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂
  | add_comm a b => exact P.add_comm _ _
  | mul_comm a b => exact P.mul_comm _ _
  | add_congr _ _ ih₁ ih₂ => simp only [eval, ih₁, ih₂]
  | mul_congr _ _ ih₁ ih₂ => simp only [eval, ih₁, ih₂]
  | fma_congr _ _ _ ih₁ ih₂ ih₃ => simp only [eval, ih₁, ih₂, ih₃]
  | div_congr _ _ ih₁ ih₂ => simp only [eval, ih₁, ih₂]
  | sqrt_congr _ ih => simp only [eval, ih]
  | cvt_congr fmt _ ih => simp only [eval, ih]
  | fmaxc_comm a b => exact P.fmaxc_comm _ _
  | fabs_congr _ ih => simp only [eval, ih]
  | fmin_congr _ _ ih₁ ih₂ => simp only [eval, ih₁, ih₂]
  | fmaxc_congr _ _ ih₁ ih₂ => simp only [eval, ih₁, ih₂]
  | nzor_congr _ _ ih₁ ih₂ => simp only [eval, ih₁, ih₂]

/-- Evaluate an optional term, reading an absent value as `z`. -/
def evalO (P : Prims α) (z : α) : Option Term → α
  | none => z
  | some t => eval P t

theorem OEquiv.eval {P : Prims α} {z : α} {o p : Option Term} (h : OEquiv o p) : evalO P z o = evalO P z p := by
  cases h with
  | absent => rfl
  | present h => exact eval_equiv P h

/-- The kernel's view of absence: substitute a concrete zero term for every absent leaf. -/
def pad (zt : Term) (f : List Bool → Option Term) : List Bool → Option Term :=
  fun bs => some ((f bs).getD zt)

theorem evalO_oadd (P : Prims α) (z : α) (hl : ∀ a, P.add z a = a) (hr : ∀ a, P.add a z = a) :
    ∀ a b, evalO P z (oadd a b) = P.add (evalO P z a) (evalO P z b)
  | none, none => (hl z).symm
  | none, some _ => (hl _).symm
  | some _, none => (hr _).symm
  | some _, some _ => rfl

/-- **Identity padding is exact (global-identity form).** If `zt` evaluates to a *global*
additive identity `z`, the zero-padded tree evaluates to the same bits as the spec's tree
with absent leaves dropped.

WARNING: the hypotheses `hl`/`hr` are NOT satisfied by IEEE binary32 bit
patterns, because `(+0) + (-0) = +0`: `+0` is not an identity for `-0`. This theorem is
kept for reference only. The realistic statement, with the identity restricted to the
values a sum of squares can take (`Good`, i.e. "not `-0`"), is `tree_pad_good` below. -/
theorem tree_pad (P : Prims α) (zt : Term) (z : α) (hz : eval P zt = z)
    (hl : ∀ a, P.add z a = a) (hr : ∀ a, P.add a z = a) :
    ∀ (n : Nat) (f : List Bool → Option Term), evalO P z (tree n (pad zt f)) = evalO P z (tree n f)
  | 0, f => by
      show evalO P z (some ((f []).getD zt)) = evalO P z (f [])
      cases f [] with
      | none => exact hz
      | some t => rfl
  | n+1, f => by
      show evalO P z (oadd (tree n (pad zt fun bs => f (bs ++ [false]))) (tree n (pad zt fun bs => f (bs ++ [true]))))
         = evalO P z (oadd (tree n fun bs => f (bs ++ [false])) (tree n fun bs => f (bs ++ [true])))
      rw [evalO_oadd P z hl hr, evalO_oadd P z hl hr,
          tree_pad P zt z hz hl hr n, tree_pad P zt z hz hl hr n]

/-! ## Restricted identity padding (the realistic hypothesis)

`tree_pad` asks for a global identity `∀ a, add z a = a`. IEEE binary32 does not satisfy
it: `(+0) + (-0) = +0`. What does hold is the restricted law `add(v, +0) = v` for every
`v` that is not `-0` (every `v ≥ +0`, `+∞` included; NaN payloads are out of scope), and
a sum of squares only ever produces such values: the accumulator starts at `+0`
and every step `fma(x, x, acc)` adds a square. `Good : α → Prop` below is that predicate,
kept abstract. The theorems need only

* `Good z` (the padding value is itself Good),
* closure of `add` (and, for the leaf fold, of `fma x x ·`) on Good values,
* the identity law on Good values,
* every *present* leaf evaluating to a Good value.

This is exactly what `Digit.identity_note` of a sum-based spec claims
("add(v, +0) == v for v >= +0"; "fma(0, 0, acc) == acc; acc starts at +0 and stays >= +0").
The kernel pads at two levels, and both are covered:

* (a) whole absent slots / blocks beyond `H` become present leaves evaluating to `+0`:
  `tree_pad_good`, `tree_eval_eq_of_leaves`, `exec_pad_good`;
* (b) absent elements *inside* a partial slot are real `fma(+0, +0, acc)` steps of the
  leaf fold: `foldl_fma_pad`.
-/
section Restricted
variable (P : Prims α) (Good : α → Prop) (z : α)

/-- `oadd` on Good operands evaluates to `add`, and stays Good. -/
theorem evalO_oadd_good (hGz : Good z) (hadd : ∀ a b, Good a → Good b → Good (P.add a b))
    (hid : ∀ a, Good a → P.add z a = a ∧ P.add a z = a) :
    ∀ a b : Option Term, Good (evalO P z a) → Good (evalO P z b) →
      evalO P z (oadd a b) = P.add (evalO P z a) (evalO P z b) ∧ Good (evalO P z (oadd a b))
  | none, none, _, _ => ⟨(hid z hGz).1.symm, hGz⟩
  | none, some _, _, hb => ⟨(hid _ hb).1.symm, hb⟩
  | some _, none, ha, _ => ⟨(hid _ ha).2.symm, ha⟩
  | some _, some _, ha, hb => ⟨rfl, hadd _ _ ha hb⟩

/-- **Restricted padding, general form.** A kernel leaf function `g` that agrees in value,
leaf by leaf, with the spec's absent-dropping leaf function `f` (whose present leaves are
Good) gives a tree that agrees in value with the spec's tree, and the spec's value is Good.
No assumption is made on `g` beyond pointwise value agreement, so `g` may pad with any term
that evaluates to `z`, including a chain of `fma(+0, +0, ·)` steps (see `foldl_fma_pad`). -/
theorem tree_eval_eq_of_leaves (hGz : Good z) (hadd : ∀ a b, Good a → Good b → Good (P.add a b))
    (hid : ∀ a, Good a → P.add z a = a ∧ P.add a z = a) :
    ∀ (n : Nat) (g f : List Bool → Option Term),
      (∀ bs, evalO P z (g bs) = evalO P z (f bs)) → (∀ bs, Good (evalO P z (f bs))) →
      evalO P z (tree n g) = evalO P z (tree n f) ∧ Good (evalO P z (tree n f))
  | 0, _, _, he, hg => ⟨he [], hg []⟩
  | n+1, g, f, he, hg => by
      have L := tree_eval_eq_of_leaves hGz hadd hid n (fun bs => g (bs ++ [false]))
        (fun bs => f (bs ++ [false])) (fun bs => he _) (fun bs => hg _)
      have R := tree_eval_eq_of_leaves hGz hadd hid n (fun bs => g (bs ++ [true]))
        (fun bs => f (bs ++ [true])) (fun bs => he _) (fun bs => hg _)
      show evalO P z (oadd (tree n fun bs => g (bs ++ [false])) (tree n fun bs => g (bs ++ [true])))
         = evalO P z (oadd (tree n fun bs => f (bs ++ [false])) (tree n fun bs => f (bs ++ [true])))
         ∧ Good (evalO P z (oadd (tree n fun bs => f (bs ++ [false])) (tree n fun bs => f (bs ++ [true]))))
      have hgl : Good (evalO P z (tree n fun bs => g (bs ++ [false]))) := by rw [L.1]; exact L.2
      have hgr : Good (evalO P z (tree n fun bs => g (bs ++ [true]))) := by rw [R.1]; exact R.2
      obtain ⟨e1, _⟩ := evalO_oadd_good P Good z hGz hadd hid _ _ hgl hgr
      obtain ⟨e2, good2⟩ := evalO_oadd_good P Good z hGz hadd hid _ _ L.2 R.2
      exact ⟨by rw [e1, e2, L.1, R.1], good2⟩

/-- **(a) Restricted identity padding is exact.** `tree_pad` under the realistic hypotheses:
the padding term `zt` evaluates to a Good value `z` that is an identity on Good values, `add`
is closed on Good values, and every *present* leaf of the spec's tree is Good. Then the
padded tree (every absent leaf replaced by `zt`) evaluates to the spec's absent-dropping
tree, and both values are Good. -/
theorem tree_pad_good (zt : Term) (hz : eval P zt = z) (hGz : Good z)
    (hadd : ∀ a b, Good a → Good b → Good (P.add a b))
    (hid : ∀ a, Good a → P.add z a = a ∧ P.add a z = a)
    (n : Nat) (f : List Bool → Option Term) (hleaf : ∀ bs t, f bs = some t → Good (eval P t)) :
    evalO P z (tree n (pad zt f)) = evalO P z (tree n f) ∧ Good (evalO P z (tree n f)) :=
  tree_eval_eq_of_leaves P Good z hGz hadd hid n (pad zt f) f
    (fun bs => by
      show eval P ((f bs).getD zt) = evalO P z (f bs)
      cases f bs with
      | none => exact hz
      | some t => rfl)
    (fun bs => by
      cases h : f bs with
      | none => exact hGz
      | some t => exact hleaf bs t h)

/-- **(a) through the engine.** For every valid placement and unit, the executed plan over the
kernel's (padded) leaf `g` evaluates to the spec's canonical reduction over the
absent-dropping leaf `f`, under pointwise value agreement of the leaves and Goodness of the
spec's leaves. -/
theorem exec_pad_good (hGz : Good z) (hadd : ∀ a b, Good a → Good b → Good (P.add a b))
    (hid : ∀ a, Good a → P.add z a = a ∧ P.add a z = a)
    (ds : List Nat) (ps : List (List Factor)) (g f : List Bool → Option Term) (cs : List (List Bool))
    (hp : PlacementOK ds ps) (hc : CoordsOK (flatten ps) cs)
    (he : ∀ bs, evalO P z (g bs) = evalO P z (f bs)) (hg : ∀ bs, Good (evalO P z (f bs))) :
    evalO P z (exec (flatten ps) g cs) = evalO P z (canonicalDigits ds f)
    ∧ Good (evalO P z (canonicalDigits ds f)) := by
  rw [canonicalDigits_eq_tree, ← sumBits_flatten ds ps hp]
  have h1 : evalO P z (exec (flatten ps) g cs) = evalO P z (tree (sumBits (flatten ps)) g) :=
    OEquiv.eval (exec_eq_tree (flatten ps) g cs hc)
  have h2 := tree_eval_eq_of_leaves P Good z hGz hadd hid (sumBits (flatten ps)) g f he hg
  exact ⟨h1.trans h2.1, h2.2⟩

/-- **(b) Padded fold steps are exact.** A fold `acc := fma(v_j, v_j, acc)` over *all*
positions `js` of a slot, where the absent positions (`p j = false`) hold a padding value
`pv j` with `fma(pv j, pv j, acc) = acc` on Good accumulators (`+0` in the kernels) and the
present positions hold the spec's value, evaluates like the spec's fold over the present
positions only; the result is Good whenever the start value is. No ordering of present and
absent positions is assumed. -/
theorem foldl_fma_pad (pv rv : Nat → α) (p : Nat → Bool)
    (hsq : ∀ a x, Good a → Good (P.fma x x a))
    (hpad : ∀ j, p j = false → ∀ a, Good a → P.fma (pv j) (pv j) a = a)
    (hpres : ∀ j, p j = true → pv j = rv j) :
    ∀ (js : List Nat) (a : α), Good a →
      js.foldl (fun a j => P.fma (pv j) (pv j) a) a
        = (js.filter p).foldl (fun a j => P.fma (rv j) (rv j) a) a
      ∧ Good ((js.filter p).foldl (fun a j => P.fma (rv j) (rv j) a) a)
  | [], _, ha => ⟨rfl, ha⟩
  | j :: js, a, ha => by
      cases hp : p j with
      | true =>
          have ih := foldl_fma_pad pv rv p hsq hpad hpres js (P.fma (rv j) (rv j) a) (hsq a (rv j) ha)
          simp only [List.filter, hp, List.foldl, hpres j hp]
          exact ih
      | false =>
          have ih := foldl_fma_pad pv rv p hsq hpad hpres js a ha
          simp only [List.filter, hp, List.foldl, hpad j hp a ha]
          exact ih

end Restricted

/-! ## Restricted identity padding for any commutative combine (`treeC` / `execC`)

The same statements for `Comb` (used by the MX quantizer's block amax with `.fmaxc`, `z = +0`, `Good v := v >= +0
and v is not -0`: every leaf is `fabs(x)`, `fmaxc(v, +0) = v` for Good `v`, and the padded (absent) blocks
and rows are present leaves evaluating to `fabs(+0) = +0`). -/
section RestrictedC
variable (P : Prims α) (Good : α → Prop) (z : α) (c : Comb)

theorem evalO_ocomb_good (hGz : Good z) (hcl : ∀ a b, Good a → Good b → Good (P.comb c a b))
    (hid : ∀ a, Good a → P.comb c z a = a ∧ P.comb c a z = a) :
    ∀ a b : Option Term, Good (evalO P z a) → Good (evalO P z b) →
      evalO P z (ocomb c a b) = P.comb c (evalO P z a) (evalO P z b) ∧ Good (evalO P z (ocomb c a b))
  | none, none, _, _ => ⟨(hid z hGz).1.symm, hGz⟩
  | none, some _, _, hb => ⟨(hid _ hb).1.symm, hb⟩
  | some _, none, ha, _ => ⟨(hid _ ha).2.symm, ha⟩
  | some _, some _, ha, hb => ⟨eval_comb P c _ _, by rw [show evalO P z (ocomb c (some _) (some _)) = P.comb c (evalO P z (some _)) (evalO P z (some _)) from eval_comb P c _ _]; exact hcl _ _ ha hb⟩

theorem treeC_eval_eq_of_leaves (hGz : Good z) (hcl : ∀ a b, Good a → Good b → Good (P.comb c a b))
    (hid : ∀ a, Good a → P.comb c z a = a ∧ P.comb c a z = a) :
    ∀ (n : Nat) (g f : List Bool → Option Term),
      (∀ bs, evalO P z (g bs) = evalO P z (f bs)) → (∀ bs, Good (evalO P z (f bs))) →
      evalO P z (treeC c n g) = evalO P z (treeC c n f) ∧ Good (evalO P z (treeC c n f))
  | 0, _, _, he, hg => ⟨he [], hg []⟩
  | n+1, g, f, he, hg => by
      have L := treeC_eval_eq_of_leaves hGz hcl hid n (fun bs => g (bs ++ [false]))
        (fun bs => f (bs ++ [false])) (fun bs => he _) (fun bs => hg _)
      have R := treeC_eval_eq_of_leaves hGz hcl hid n (fun bs => g (bs ++ [true]))
        (fun bs => f (bs ++ [true])) (fun bs => he _) (fun bs => hg _)
      show evalO P z (ocomb c (treeC c n fun bs => g (bs ++ [false])) (treeC c n fun bs => g (bs ++ [true])))
         = evalO P z (ocomb c (treeC c n fun bs => f (bs ++ [false])) (treeC c n fun bs => f (bs ++ [true])))
         ∧ Good (evalO P z (ocomb c (treeC c n fun bs => f (bs ++ [false])) (treeC c n fun bs => f (bs ++ [true]))))
      have hgl : Good (evalO P z (treeC c n fun bs => g (bs ++ [false]))) := by rw [L.1]; exact L.2
      have hgr : Good (evalO P z (treeC c n fun bs => g (bs ++ [true]))) := by rw [R.1]; exact R.2
      obtain ⟨e1, _⟩ := evalO_ocomb_good P Good z c hGz hcl hid _ _ hgl hgr
      obtain ⟨e2, good2⟩ := evalO_ocomb_good P Good z c hGz hcl hid _ _ L.2 R.2
      exact ⟨by rw [e1, e2, L.1, R.1], good2⟩

/-- **Restricted identity padding through the engine, for any commutative combine.** -/
theorem execC_pad_good (hGz : Good z) (hcl : ∀ a b, Good a → Good b → Good (P.comb c a b))
    (hid : ∀ a, Good a → P.comb c z a = a ∧ P.comb c a z = a)
    (ds : List Nat) (ps : List (List Factor)) (g f : List Bool → Option Term) (cs : List (List Bool))
    (hp : PlacementOK ds ps) (hc : CoordsOK (flatten ps) cs)
    (he : ∀ bs, evalO P z (g bs) = evalO P z (f bs)) (hg : ∀ bs, Good (evalO P z (f bs))) :
    evalO P z (execC c (flatten ps) g cs) = evalO P z (canonicalDigitsC c ds f)
    ∧ Good (evalO P z (canonicalDigitsC c ds f)) := by
  rw [canonicalDigitsC_eq_treeC, ← sumBits_flatten ds ps hp]
  have h1 : evalO P z (execC c (flatten ps) g cs) = evalO P z (treeC c (sumBits (flatten ps)) g) :=
    OEquiv.eval (execC_eq_treeC c (flatten ps) g cs hc)
  have h2 := treeC_eval_eq_of_leaves P Good z c hGz hcl hid (sumBits (flatten ps)) g f he hg
  exact ⟨h1.trans h2.1, h2.2⟩

end RestrictedC

theorem ocomb_some (c : Comb) : ∀ {a b : Option Term}, (∃ x, a = some x) → (∃ y, b = some y) → ∃ t, ocomb c a b = some t
  | some _, some _, _, _ => ⟨_, rfl⟩
  | none, _, ⟨_, h⟩, _ => Option.noConfusion h
  | some _, none, _, ⟨_, h⟩ => Option.noConfusion h

theorem treeC_some (c : Comb) : ∀ (n : Nat) (f : List Bool → Option Term), (∀ bs, ∃ t, f bs = some t) → ∃ t, treeC c n f = some t
  | 0, _, h => h []
  | n+1, _, h => ocomb_some c (treeC_some c n _ fun _ => h _) (treeC_some c n _ fun _ => h _)

theorem bflyC_some (c : Comb) : ∀ (i : Nat) (val : List Bool → Option Term) (v : List Bool),
    (∀ u, ∃ t, val u = some t) → ∃ t, bflyC c i val v = some t
  | 0, _, v, h => h v
  | i+1, val, v, h => ocomb_some c (bflyC_some c i val v h) (bflyC_some c i val _ h)

theorem execC_some (c : Comb) : ∀ (fs : List Factor) (leaf : List Bool → Option Term) (cs : List (List Bool)),
    (∀ bs, ∃ t, leaf bs = some t) → ∃ t, execC c fs leaf cs = some t
  | [], _, _, h => h []
  | f :: fs, leaf, cs, h => by
      cases hk : f.kind with
      | lane =>
          simp only [execC, hk]
          exact execC_some c fs _ cs.tail (fun _ => bflyC_some c f.bits _ _ (fun _ => h _))
      | reg =>
          simp only [execC, hk]
          exact execC_some c fs _ cs.tail (fun _ => treeC_some c f.bits _ (fun _ => h _))
      | xchg =>
          simp only [execC, hk]
          exact execC_some c fs _ cs.tail (fun _ => treeC_some c f.bits _ (fun _ => h _))

/-! ## Presence: a plan over an all-present leaf function yields a present result -/

theorem oadd_some : ∀ {a b : Option Term}, (∃ x, a = some x) → (∃ y, b = some y) → ∃ t, oadd a b = some t
  | some _, some _, _, _ => ⟨_, rfl⟩
  | none, _, ⟨_, h⟩, _ => Option.noConfusion h
  | some _, none, _, ⟨_, h⟩ => Option.noConfusion h

theorem tree_some : ∀ (n : Nat) (f : List Bool → Option Term), (∀ bs, ∃ t, f bs = some t) → ∃ t, tree n f = some t
  | 0, _, h => h []
  | n+1, _, h => oadd_some (tree_some n _ fun _ => h _) (tree_some n _ fun _ => h _)

theorem bfly_some : ∀ (i : Nat) (val : List Bool → Option Term) (v : List Bool),
    (∀ u, ∃ t, val u = some t) → ∃ t, bfly i val v = some t
  | 0, _, v, h => h v
  | i+1, val, v, h => oadd_some (bfly_some i val v h) (bfly_some i val _ h)

theorem exec_some : ∀ (fs : List Factor) (leaf : List Bool → Option Term) (cs : List (List Bool)),
    (∀ bs, ∃ t, leaf bs = some t) → ∃ t, exec fs leaf cs = some t
  | [], _, _, h => h []
  | f :: fs, leaf, cs, h => by
      cases hk : f.kind with
      | lane =>
          simp only [exec, hk]
          exact exec_some fs _ cs.tail (fun _ => bfly_some f.bits _ _ (fun _ => h _))
      | reg =>
          simp only [exec, hk]
          exact exec_some fs _ cs.tail (fun _ => tree_some f.bits _ (fun _ => h _))
      | xchg =>
          simp only [exec, hk]
          exact exec_some fs _ cs.tail (fun _ => tree_some f.bits _ (fun _ => h _))

end BIK
