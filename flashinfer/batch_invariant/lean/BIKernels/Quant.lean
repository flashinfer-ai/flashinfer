import BIKernels.Canon
import BIKernels.Semantics
import BIKernels.Support
/-!
# NVFP4 quantization on the engine: classes `nvfp4.quant.te` and `nvfp4.quant.te.pertoken`

Mirrors `quantization/nvfp4_spec.py` and `quantization/nvfp4_schedule.py`.

* The amax reductions are `pairwise` digits combined with `fmaxc` (the bitwise-commutative max of
  non-negative values): the block amax has the digit `[4]` (16 elements), the row amax of the per-token
  class the digits `[4, kb]` (`kb` = log2 of the padded block count, blocks beyond `K / 16` absent). The
  leaf is `fabs(x)`. `config_invarianceC .fmaxc` (Canon.lean) is the engine theorem these instantiate.
* The chains are TransformerEngine's recipe as FlashInfer's exact path computes it: `sfValue`,
  `sfQuant` (the `cvt e4m3`), `invScale`, `quantElem` (the `cvt e2m1`) per block and element,
  `rowScale` per row for the per-token class. `quantization/lean_cases.py` couples every definition below to
  the Python S-expressions node for node (`quant*` cases in `QuantCases.lean`).
* `block_amax_invariance` / `row_amax_invariance`: every valid placement and unit computes the canonical
  trees; `sf_equiv` / `code_equiv` / `pts_equiv` carry the invariance through the chains (absent-dropping
  model); `rowamax_padded_bits` is the bitwise statement for the zero-filling bucket kernel (blocks and
  rows beyond `K` / `M` read as `+0`, whose `fabs` is `+0`, the identity of `fmaxc` on non-negative values)
  under the realistic restricted-identity hypotheses (cf. `Semantics.lean`, section `RestrictedC`).
* The launch model: a row's outputs are a function of the row and the scale input alone;
  determinism, batch, permutation and sub-batch invariance hold by construction.
* `e2m1Code` is the exact statement of what `cvt.rn.satfinite.e2m1x2.f32` is assumed to compute on the
  exact value `m * 2^e` of a binary32 input (assumption A4 of the class): RNE to the e2m1 magnitudes with
  ties to the even code, saturation at 6, the sign kept. The Python model `exact.to_e2m1` and the device
  are each checked against it (conformance cases; the exhaustive device sweep).
-/
namespace BIK.Quant
open BIK

/-! ## The reductions -/

/-- The leaf: `fabs` of the widened input element. -/
def leaf (row : Nat → Term) (i : Nat) : Term := .fabs (row i)

/-- Leaf of the row reduction: bits = element bits (4, combined first) ++ block bits (`kb`);
element index `16 b + e`; blocks `>= nb` (beyond `K / 16`) are absent. -/
def rowLeaf (nb : Nat) (row : Nat → Term) (bs : List Bool) : Option Term :=
  let e := bitsToNat (takeN 4 bs)
  let b := bitsToNat (dropN 4 bs)
  if b < nb then some (leaf row (16 * b + e)) else none

/-- The spec's block amax of block `b`: `spec.block_reduction(...).canonical`. -/
def blockAmax (row : Nat → Term) (b : Nat) : Option Term :=
  treeC .fmaxc 4 (fun bs => some (leaf row (16 * b + bitsToNat bs)))

/-- The spec's row amax: `spec.row_reduction(K, ...).canonical`, digits `[4, kb]`. -/
def rowAmax (nb kb : Nat) (row : Nat → Term) : Option Term :=
  canonicalDigitsC .fmaxc [4, kb] (rowLeaf nb row)

theorem blockAmax_some (row : Nat → Term) (b : Nat) : ∃ a, blockAmax row b = some a :=
  treeC_some .fmaxc 4 _ (fun _ => ⟨_, rfl⟩)

/-- **The block amax is config-invariant**: any placement of the `elem` digit (register factors, a
lane factor finishing the block by shuffles) computes the canonical block tree in every unit. This is
what the kernel reads from the engine's state before the `blk` digit (`canon.run_steps(..., "blk")`). -/
theorem block_amax_invariance (row : Nat → Term) (b : Nat) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [4] ps) (hc : CoordsOK (flatten ps) cs) :
    OEquiv (execC .fmaxc (flatten ps) (fun bs => some (leaf row (16 * b + bitsToNat bs))) cs) (blockAmax row b) := by
  have h := config_invarianceC .fmaxc [4] ps (fun bs => some (leaf row (16 * b + bitsToNat bs))) cs hp hc
  rw [canonicalDigitsC_eq_treeC] at h
  exact h

/-- **The row amax is config-invariant** (per-token class): every valid placement of the digits
`[4, kb]` -- `elem = [(V, reg), (16/V, lane)]`, `blk` = block-lane, register and warp factors in either
order (`QuantConfig.placement`) -- and every unit computes the canonical row tree. -/
theorem row_amax_invariance (nb kb : Nat) (row : Nat → Term) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [4, kb] ps) (hc : CoordsOK (flatten ps) cs) :
    OEquiv (execC .fmaxc (flatten ps) (rowLeaf nb row) cs) (rowAmax nb kb row) :=
  config_invarianceC .fmaxc [4, kb] ps (rowLeaf nb row) cs hp hc

/-- `QuantConfig.placement`: `elem = [(vb bits, reg), (lb bits, lane)]` (`vb + lb = 4`), `blk` =
`[(blb, lane), (bptb, reg), (wb, xchg)]` (lane-major) or `[(bptb, reg), (blb, lane), (wb, xchg)]`
(register-major), `blb + bptb + wb = kb`. -/
def placementQuant (vb lb blb bptb wb : Nat) (laneMajor : Bool) : List (List Factor) :=
  [[⟨vb, .reg⟩, ⟨lb, .lane⟩],
   if laneMajor then [⟨blb, .lane⟩, ⟨bptb, .reg⟩, ⟨wb, .xchg⟩] else [⟨bptb, .reg⟩, ⟨blb, .lane⟩, ⟨wb, .xchg⟩]]

theorem placementQuant_ok (vb lb blb bptb wb kb : Nat) (lm : Bool) (h1 : vb + lb = 4) (h2 : blb + bptb + wb = kb) :
    PlacementOK [4, kb] (placementQuant vb lb blb bptb wb lm) := by
  unfold placementQuant
  cases lm <;> refine ⟨?_, ?_, trivial⟩
  · show vb + (lb + 0) = 4
    omega
  · show bptb + (blb + (wb + 0)) = kb
    omega
  · show vb + (lb + 0) = 4
    omega
  · show blb + (bptb + (wb + 0)) = kb
    omega

/-! ## The chains (`spec.sf_value`, `sf_quant`, `inv_scale`, `quant_pre` / `quant_elem`, `row_scale_chain`)

Python prints binary32 constants as `repr(float(c))`: the rounded `1/6` is `0.1666666716337204`,
`FLT_MAX` is `3.4028234663852886e+38`. -/

def cSixth : Term := .const "0.1666666716337204"
def fltMax : Term := .const "3.4028234663852886e+38"
def one : Term := .const "1.0"

/-- `sf_f = mul(amax, mul(gs, 1/6))`. -/
def sfValue (amax gs : Term) : Term := .mul amax (.mul gs cSixth)

/-- `sf_d = cvt_e4m3(sf_f)` (the stored byte, as its exactly widened value). -/
def sfQuant (sf : Term) : Term := .cvt "e4m3" sf

/-- `inv = fmin(div(1, mul(sf_d, div(1, gs))), FLT_MAX)`. -/
def invScale (sfd gs : Term) : Term := .fmin (.div one (.mul sfd (.div one gs))) fltMax

/-- `q = cvt_e2m1(mul(x, inv))`. -/
def quantPre (x inv : Term) : Term := .mul x inv
def quantElem (x inv : Term) : Term := .cvt "e2m1" (quantPre x inv)

/-- `(sf_d, inv)` of a block from its amax and the global scale. -/
def blockChain (amax gs : Term) : Term × Term :=
  let sfd := sfQuant (sfValue amax gs)
  (sfd, invScale sfd gs)

/-- Per-token: `gs0 = div(1, gs_inv)`, `res = fmin(div(gs0, ramax), FLT_MAX)`, `ges = nzor(res, 1)`,
`pts = div(1, ges)`; returns `(ges, pts)`. -/
def rowScale (ramax gsInv : Term) : Term × Term :=
  let gs0 := Term.div one gsInv
  let res := Term.fmin (.div gs0 ramax) fltMax
  let ges := Term.nzor res one
  (ges, .div one ges)

/-- `spec.Variant`: the per-tensor class (the scale is the input `gs`) or the per-token class (the scale
is derived from the row amax and the input `gs_inv`). -/
structure Variant where
  perToken : Bool := false
  deriving Repr, DecidableEq

namespace Variant
def perTensor : Variant := {}
def perTok : Variant := { perToken := true }

/-- The scale the row's blocks are quantised with (`QuantSpec.row_scale`). -/
def gs (v : Variant) (nb kb : Nat) (row : Nat → Term) : Option Term :=
  if v.perToken then (rowAmax nb kb row).map (fun r => (rowScale r (.sym "gs_inv")).1) else some (.sym "gs")
end Variant

/-- `QuantSpec.block(row, b)["sf"]`: the scale byte's value of block `b`. -/
def specSf (v : Variant) (nb kb : Nat) (row : Nat → Term) (b : Nat) : Option Term :=
  (v.gs nb kb row).bind fun gs => (blockAmax row b).map fun a => (blockChain a gs).1

/-- `QuantSpec.block(row, i / 16)["codes"][i % 16]`: the e2m1 value of element `i`. -/
def specCode (v : Variant) (nb kb : Nat) (row : Nat → Term) (i : Nat) : Option Term :=
  (v.gs nb kb row).bind fun gs => (blockAmax row (i / 16)).map fun a => quantElem (row i) (blockChain a gs).2

/-- `QuantSpec.row_scale(row)[1]` of the per-token class: the per-token scale output. -/
def specPts (nb kb : Nat) (row : Nat → Term) : Option Term :=
  (rowAmax nb kb row).map fun r => (rowScale r (.sym "gs_inv")).2

/-! ## Invariance through the chains (absent-dropping model) -/

theorem sfValue_equiv {a a' gs gs' : Term} (ha : Equiv a' a) (hg : Equiv gs' gs) : Equiv (sfValue a' gs') (sfValue a gs) :=
  .mul_congr ha (.mul_congr hg (.refl _))

theorem blockChain_equiv {a a' gs gs' : Term} (ha : Equiv a' a) (hg : Equiv gs' gs) :
    Equiv (blockChain a' gs').1 (blockChain a gs).1 ∧ Equiv (blockChain a' gs').2 (blockChain a gs).2 := by
  have hs : Equiv (sfQuant (sfValue a' gs')) (sfQuant (sfValue a gs)) := .cvt_congr _ (sfValue_equiv ha hg)
  exact ⟨hs, .fmin_congr (.div_congr (.refl _) (.mul_congr hs (.div_congr (.refl _) hg))) (.refl _)⟩

theorem code_equiv {a a' gs gs' : Term} (x : Term) (ha : Equiv a' a) (hg : Equiv gs' gs) :
    Equiv (quantElem x (blockChain a' gs').2) (quantElem x (blockChain a gs).2) :=
  .cvt_congr _ (.mul_congr (.refl x) (blockChain_equiv ha hg).2)

theorem rowScale_equiv {r r' : Term} (h : Equiv r' r) (gsInv : Term) :
    Equiv (rowScale r' gsInv).1 (rowScale r gsInv).1 ∧ Equiv (rowScale r' gsInv).2 (rowScale r gsInv).2 := by
  have hges : Equiv (rowScale r' gsInv).1 (rowScale r gsInv).1 :=
    .nzor_congr (.fmin_congr (.div_congr (.refl _) h) (.refl _)) (.refl _)
  exact ⟨hges, .div_congr (.refl _) hges⟩

/-- **End-to-end, per-tensor class.** For every valid placement of the `elem` digit and every unit, the
kernel's block amax exists and the scale and every code of the block are `Equiv` to the spec's, hence
(`eval_equiv`) bitwise equal under any commutative interpretation. -/
theorem block_outputs_invariant (row : Nat → Term) (b : Nat) (gs : Term) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [4] ps) (hc : CoordsOK (flatten ps) cs) :
    ∃ a a', blockAmax row b = some a ∧ execC .fmaxc (flatten ps) (fun bs => some (leaf row (16 * b + bitsToNat bs))) cs = some a' ∧
      Equiv (blockChain a' gs).1 (blockChain a gs).1 ∧
      ∀ e, Equiv (quantElem (row (16 * b + e)) (blockChain a' gs).2) (quantElem (row (16 * b + e)) (blockChain a gs).2) := by
  obtain ⟨a, ha⟩ := blockAmax_some row b
  have h := block_amax_invariance row b ps cs hp hc
  rw [ha] at h
  obtain ⟨a', hk, he⟩ := OEquiv.some_right h
  exact ⟨a, a', ha, hk, (blockChain_equiv he (.refl gs)).1, fun e => code_equiv _ he (.refl gs)⟩

/-- **End-to-end, per-token class.** Whenever the spec's row amax exists, the kernel's exists for every
valid placement and unit, and the per-token scale and the row's global scale are `Equiv` to the spec's. -/
theorem row_outputs_invariant (nb kb : Nat) (row : Nat → Term) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [4, kb] ps) (hc : CoordsOK (flatten ps) cs) (r : Term) (hr : rowAmax nb kb row = some r) :
    ∃ r', execC .fmaxc (flatten ps) (rowLeaf nb row) cs = some r' ∧
      Equiv (rowScale r' (.sym "gs_inv")).1 (rowScale r (.sym "gs_inv")).1 ∧
      Equiv (rowScale r' (.sym "gs_inv")).2 (rowScale r (.sym "gs_inv")).2 := by
  have h := row_amax_invariance nb kb row ps cs hp hc
  rw [hr] at h
  obtain ⟨r', hk, he⟩ := OEquiv.some_right h
  exact ⟨r', hk, (rowScale_equiv he _).1, (rowScale_equiv he _).2⟩

/-- Bitwise form: the per-token scale of the kernel model equals the spec's under any commutative
interpretation of the primitives. -/
theorem pts_bits_equal {α : Type} (P : Prims α) (nb kb : Nat) (row : Nat → Term) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [4, kb] ps) (hc : CoordsOK (flatten ps) cs) (r : Term) (hr : rowAmax nb kb row = some r) :
    ∃ r', execC .fmaxc (flatten ps) (rowLeaf nb row) cs = some r' ∧
      eval P (rowScale r' (.sym "gs_inv")).2 = eval P (rowScale r (.sym "gs_inv")).2 := by
  obtain ⟨r', hk, _, h2⟩ := row_outputs_invariant nb kb row ps cs hp hc r hr
  exact ⟨r', hk, eval_equiv P h2⟩

/-! ## The zero-filling bucket kernel (runtime K and M)

The compiled kernel knows only the bucket `kb`: blocks of a row beyond `K / 16` and whole rows beyond
`M` are zero-filled by the predicated loads, so every leaf is present and an absent block reads as
`fabs(+0)`. Under the restricted identity -- `+0` is Good, `fmaxc` keeps Good values Good, `fmaxc(+0, v)
= fmaxc(v, +0) = v` for Good `v` (true of IEEE max for every `v >= +0`), `fabs(x)` is Good for every `x`
(it is `>= +0` and never `-0`) and `fabs(+0) = +0` -- the padded kernel's row amax equals the spec's
absent-dropping tree bit for bit. -/

/-- The zero-extended row (`padRow`): elements beyond `K` read as `+0`. -/
def rowLeafPadded (nb : Nat) (row : Nat → Term) (bs : List Bool) : Option Term :=
  some (leaf (padRow (16 * nb) (.const "0.0") row) (16 * bitsToNat (dropN 4 bs) + bitsToNat (takeN 4 bs)))

theorem bitsToNat_lt : ∀ l : List Bool, bitsToNat l < 2 ^ l.length
  | [] => Nat.lt_succ_self 0
  | b :: bs => by
      have ih := bitsToNat_lt bs
      show (if b then 1 else 0) + 2 * bitsToNat bs < 2 ^ (bs.length + 1)
      rw [Nat.pow_succ]
      cases b <;> simp <;> omega

theorem takeN_length_le : ∀ (n : Nat) (l : List Bool), (takeN n l).length ≤ n
  | 0, _ => Nat.le_refl 0
  | _+1, [] => Nat.zero_le _
  | n+1, _ :: bs => Nat.succ_le_succ (takeN_length_le n bs)

theorem elem_lt_16 (bs : List Bool) : bitsToNat (takeN 4 bs) < 16 := by
  have h1 := bitsToNat_lt (takeN 4 bs)
  have h2 := takeN_length_le 4 bs
  have h3 : 2 ^ (takeN 4 bs).length ≤ 2 ^ 4 := Nat.pow_le_pow_right (by decide) h2
  omega

/-- **The bucket kernel's row amax equals the spec's, bit for bit, under the restricted identity.** -/
theorem rowamax_padded_bits {α : Type} (P : Prims α) (Good : α → Prop) (nb kb : Nat) (row : Nat → Term)
    (ps : List (List Factor)) (cs : List (List Bool)) (hp : PlacementOK [4, kb] ps) (hc : CoordsOK (flatten ps) cs)
    (z : α) (hz : P.fabs (P.cv "0.0") = z) (hGz : Good z)
    (hcl : ∀ a b, Good a → Good b → Good (P.fmaxc a b))
    (hid : ∀ a, Good a → P.fmaxc z a = a ∧ P.fmaxc a z = a)
    (habs : ∀ x, Good (P.fabs x)) :
    evalO P z (execC .fmaxc (flatten ps) (rowLeafPadded nb row) cs) = evalO P z (rowAmax nb kb row)
    ∧ Good (evalO P z (rowAmax nb kb row)) := by
  refine execC_pad_good P Good z .fmaxc hGz hcl hid [4, kb] ps (rowLeafPadded nb row) (rowLeaf nb row) cs hp hc ?_ ?_
  · intro bs
    simp only [rowLeafPadded, rowLeaf, leaf, padRow]
    by_cases hb : bitsToNat (dropN 4 bs) < nb
    · have hi : 16 * bitsToNat (dropN 4 bs) + bitsToNat (takeN 4 bs) < 16 * nb := by
        have := elem_lt_16 bs
        omega
      simp only [hb, hi, if_true, evalO, eval]
    · have hi : ¬ (16 * bitsToNat (dropN 4 bs) + bitsToNat (takeN 4 bs) < 16 * nb) := by omega
      simp only [hb, hi, if_false, evalO, eval]
      exact hz
  · intro bs
    simp only [rowLeaf]
    by_cases hb : bitsToNat (dropN 4 bs) < nb
    · simp only [hb, if_true, evalO, leaf, eval]
      exact habs _
    · simp only [hb, if_false, evalO]
      exact hGz

/-! ## The launch model: R1 / R2 by construction -/

/-- A row's outputs: its codes, its scale bytes and (per-token) its scale, as functions of the row
alone (and of the scale input, a symbol). No batch, no position, no other rows. -/
def rowOut (v : Variant) (nb kb : Nat) (row : Nat → Term) : (Nat → Option Term) × (Nat → Option Term) × Option Term :=
  (specCode v nb kb row, specSf v nb kb row, if v.perToken then specPts nb kb row else none)

def launch (M : Nat) (rows : Fin M → Nat → Term) (v : Variant) (nb kb : Nat) (r : Fin M) :=
  rowOut v nb kb (rows r)

/-- **R1 determinism**: the model is a function. -/
theorem determinism (v : Variant) (nb kb : Nat) (row : Nat → Term) : rowOut v nb kb row = rowOut v nb kb row := rfl

/-- **R2 batch invariance**: equal rows give equal outputs whatever the batch size, position or neighbours. -/
theorem batch_invariance (v : Variant) (nb kb : Nat) {M₁ M₂ : Nat} (rows₁ : Fin M₁ → Nat → Term) (rows₂ : Fin M₂ → Nat → Term)
    (r₁ : Fin M₁) (r₂ : Fin M₂) (h : rows₁ r₁ = rows₂ r₂) : launch M₁ rows₁ v nb kb r₁ = launch M₂ rows₂ v nb kb r₂ := by
  simp only [launch, h]

/-- **Permutation invariance**: permuting the rows permutes the outputs. -/
theorem permutation_invariance (v : Variant) (nb kb : Nat) {M : Nat} (rows : Fin M → Nat → Term) (σ : Fin M → Fin M) (r : Fin M) :
    launch M (rows ∘ σ) v nb kb r = launch M rows v nb kb (σ r) := rfl

/-- **Sub-batch invariance**: any selection of rows reproduces exactly those rows' outputs. -/
theorem subbatch_invariance (v : Variant) (nb kb : Nat) {M k : Nat} (rows : Fin M → Nat → Term) (sel : Fin k → Fin M) (i : Fin k) :
    launch k (rows ∘ sel) v nb kb i = launch M rows v nb kb (sel i) := rfl

/-- **The per-tensor global scale is an input.** A dynamic per-tensor scale would be a function of the
whole batch (`amax` over every row); `rowOut` takes only the row, so no such dependence can be
expressed in the class (docs/nvfp4_quant_scoping.md section 4). The per-token scale depends on the row
alone (`specPts`), which is why that dynamic variant is invariant. -/
theorem scale_is_row_local (nb kb : Nat) (row : Nat → Term) :
    specPts nb kb row = (rowAmax nb kb row).map fun r => (rowScale r (.sym "gs_inv")).2 := rfl

/-! ## The e2m1 conversion (assumption A4)

`e2m1Code neg a e` is the code `cvt.rn.satfinite.e2m1x2.f32` is assumed to produce for an input of sign
`neg` whose magnitude is exactly `a * 2^e` (every binary32 value has such a representation): the
magnitude is compared with the midpoints `1/4, 3/4, 5/4, 7/4, 10/4, 14/4, 20/4` between the e2m1
magnitudes `0, 1/2, 1, 3/2, 2, 3, 4, 6` (exact integer comparisons of `4 a 2^e` against `n`); the code
is the number of midpoints strictly below the magnitude, plus one on a tie when the upper code is even
(round to nearest, ties to even); magnitudes `>= 6` give code 7 (`satfinite`); the sign is bit 3 (so
`-0` is code 8). `exact.to_e2m1` and the device conversion are checked against this function
(`QuantCases.lean`: Python-emitted `(neg, a, e, code)` tuples; the GPU test: every binary32 bit pattern). -/

/-- `4 * a * 2^e` as a fraction `num / den` with a power-of-two denominator. -/
def scaled4 (a : Nat) (e : Int) : Nat × Nat :=
  if e ≥ 0 then (4 * a * 2 ^ e.toNat, 1) else (4 * a, 2 ^ (-e).toNat)

def e2m1Mids : List Nat := [1, 3, 5, 7, 10, 14, 20]

def e2m1Magnitude (a : Nat) (e : Int) : Nat :=
  let (num, den) := scaled4 a e
  if num ≥ 24 * den then 7 else
    let below := (e2m1Mids.filter (fun n => num > n * den)).length
    let tieUp := e2m1Mids.enum.any (fun ⟨k, n⟩ => num == n * den && (k + 1) % 2 == 0)
    below + (if tieUp then 1 else 0)

def e2m1Code (neg : Bool) (a : Nat) (e : Int) : Nat :=
  e2m1Magnitude a e + (if neg then 8 else 0)

theorem e2m1_ties_to_even : e2m1Code false 1 (-2) = 0 ∧ e2m1Code false 3 (-2) = 2 ∧ e2m1Code false 5 (-2) = 2
    ∧ e2m1Code false 7 (-2) = 4 ∧ e2m1Code false 5 (-1) = 4 ∧ e2m1Code false 7 (-1) = 6 ∧ e2m1Code false 5 0 = 6 := by decide

theorem e2m1_saturates : e2m1Code false 7 0 = 7 ∧ e2m1Code false 1 30 = 7 ∧ e2m1Code true 13 0 = 15 := by decide

theorem e2m1_magnitudes : e2m1Code false 0 0 = 0 ∧ e2m1Code true 0 0 = 8 ∧ e2m1Code false 1 (-1) = 1 ∧ e2m1Code false 1 0 = 2
    ∧ e2m1Code false 3 (-1) = 3 ∧ e2m1Code false 2 0 = 4 ∧ e2m1Code false 3 0 = 5 ∧ e2m1Code false 4 0 = 6
    ∧ e2m1Code false 6 0 = 7 ∧ e2m1Code true 1 0 = 10 := by decide

end BIK.Quant
