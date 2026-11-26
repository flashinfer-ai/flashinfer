-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.Canon
import BIKernels.Semantics
import BIKernels.Quant
/-!
# The fp8 block quantizer on the engine: class `fp8bs.quant`

Mirrors `fused_moe/batch_invariant/spec_fp8bs.py` (the quantizer of the block-scale fp8 MoE classes): one fp32 scale per
block of 128 consecutive elements (the DeepSeek-style recipe `QuantFormat.DeepSeekFp8` names: 1x128 activation blocks, 128x128
weight blocks), e4m3 codes.

* The block amax is one `pairwise` digit of radix 128 (`[7]`) combined with `fmaxc` over the leaf `fabs(x)`; every placement
  `[(V, reg), (128 / V, lane)]` of the standalone kernel (`blocks/fp8bs_small.py`: `V = 8`) computes it (`config_invarianceC
  .fmaxc` instantiated). There is no row digit: a block's bits depend on its 128 inputs alone (`block_locality`), which is why
  the class is batch invariant without any global-scale argument.
* The fused requantization of the block-scale MoE class (`blocks/epilogue_fp8bs.py` on the unified grouped body) holds a block's two 64-output halves in two
  warpgroups: `blockAmax_halves` says the canonical tree IS the combine of the two half trees (the first half's leaves are
  elements 0 .. 63), so the exchange of the half amaxes computes the class's tree (definitionally).
* The chains are fixed: `scaleT` -- `s = fmaxc(div(amax, 448), 1e-12)` (the dequantization scale the block-scale GEMM class
  reads; the floor gives a zero block a finite scale), `invT` -- `inv = div(1, s)` (the correctly rounded reciprocal, formed
  once per block), `quantElemT` -- `q = cvt.rn.satfinite.e4m3(mul(x, inv))`. The generated `MoEFp8BsCases.lean` couples them to
  the Python S-expressions node for node (`fp8bsq*` cases).
* `block_outputs_invariant`: every valid placement and unit computes the canonical tree and, through the chains, `Equiv` scale
  and codes (bit equality under any commutative interpretation: `s_bits_equal`). The launch model: a row's outputs are a
  function of the row alone; determinism, batch, permutation and sub-batch invariance by construction.
-/
namespace BIK.FP8BQ
open BIK

/-! ## The block amax -/

/-- The leaf: `fabs` of the widened input element. -/
def leaf (row : Nat → Term) (i : Nat) : Term := .fabs (row i)

/-- Leaf of block `b` (128 elements): the 7 bits are the element index inside the block (`takeN 7` bounds the index for every
list; on the tree's 7-bit lists it is the identity). -/
def blockLeaf (row : Nat → Term) (b : Nat) (bs : List Bool) : Option Term :=
  some (leaf row (128 * b + bitsToNat (takeN 7 bs)))

/-- The spec's block amax: `spec_fp8bs.block_reduction(...).canonical`. -/
def blockAmax (row : Nat → Term) (b : Nat) : Option Term := treeC .fmaxc 7 (blockLeaf row b)

theorem blockAmax_some (row : Nat → Term) (b : Nat) : ∃ a, blockAmax row b = some a :=
  treeC_some .fmaxc 7 _ (fun _ => ⟨_, rfl⟩)

theorem elem_lt_128 (bs : List Bool) : bitsToNat (takeN 7 bs) < 128 := by
  have h1 := Quant.bitsToNat_lt (takeN 7 bs)
  have h2 := Quant.takeN_length_le 7 bs
  have h3 : 2 ^ (takeN 7 bs).length ≤ 2 ^ 7 := Nat.pow_le_pow_right (by decide) h2
  omega

/-- **The block amax is config-invariant**: any placement of the digit `[7]` (register factors, lane factors finishing the
block by shuffles) computes the canonical block tree in every unit. -/
theorem block_amax_invariance (row : Nat → Term) (b : Nat) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [7] ps) (hc : CoordsOK (flatten ps) cs) :
    OEquiv (execC .fmaxc (flatten ps) (blockLeaf row b) cs) (blockAmax row b) := by
  have h := config_invarianceC .fmaxc [7] ps (blockLeaf row b) cs hp hc
  rw [canonicalDigitsC_eq_treeC] at h
  exact h

/-- `spec_fp8bs.block_amax_placement(V)`: `elem = [(vb bits, reg), (lb bits, lane)]`, `vb + lb = 7` (the standalone kernel:
`V = 8`, `lb = 4`; the fused epilogue's half blocks: `V = 64`, one lane factor of one bit realised by the warpgroup exchange). -/
def placementElem (vb lb : Nat) : List (List Factor) := [[⟨vb, .reg⟩, ⟨lb, .lane⟩]]

theorem placementElem_ok (vb lb : Nat) (h : vb + lb = 7) : PlacementOK [7] (placementElem vb lb) := by
  unfold placementElem
  refine ⟨?_, trivial⟩
  show vb + (lb + 0) = 7
  omega

/-- The leaves of a half block: elements `64 h + j` of block `b` (`j` the 6-bit index). -/
def halfLeaf (row : Nat → Term) (b h : Nat) (bs : List Bool) : Option Term :=
  some (leaf row (128 * b + 64 * h + bitsToNat (takeN 6 bs)))

/-- The half-block tree (what one promotion warpgroup of the fused requantization holds: its 64 consecutive outputs of the block). -/
def halfAmax (row : Nat → Term) (b h : Nat) : Option Term := treeC .fmaxc 6 (halfLeaf row b h)

theorem treeC_congr_eq (c : Comb) : ∀ (n : Nat) {f g : List Bool → Option Term},
    (∀ bs, bs.length = n → f bs = g bs) → treeC c n f = treeC c n g
  | 0, f, g, h => by simp only [treeC]; exact h [] rfl
  | n+1, f, g, h => by
      simp only [treeC]
      rw [treeC_congr_eq c n (fun bs hbs => h (bs ++ [false]) (by simp [hbs])),
          treeC_congr_eq c n (fun bs hbs => h (bs ++ [true]) (by simp [hbs]))]

theorem bitsToNat_takeN_append_false : ∀ (n : Nat) (bs : List Bool), bs.length = n →
    bitsToNat (takeN (n + 1) (bs ++ [false])) = bitsToNat (takeN n bs)
  | 0, [], _ => rfl
  | 0, _ :: _, h => by simp at h
  | n+1, [], h => by simp at h
  | n+1, b :: bs, h => by
      have := bitsToNat_takeN_append_false n bs (by simpa using h)
      show bitsToNat (b :: takeN (n + 1) (bs ++ [false])) = bitsToNat (b :: takeN n bs)
      simp only [bitsToNat, this]

theorem bitsToNat_takeN_append_true : ∀ (n : Nat) (bs : List Bool), bs.length = n →
    bitsToNat (takeN (n + 1) (bs ++ [true])) = bitsToNat (takeN n bs) + 2 ^ n
  | 0, [], _ => rfl
  | 0, _ :: _, h => by simp at h
  | n+1, [], h => by simp at h
  | n+1, b :: bs, h => by
      have := bitsToNat_takeN_append_true n bs (by simpa using h)
      show bitsToNat (b :: takeN (n + 1) (bs ++ [true])) = bitsToNat (b :: takeN n bs) + 2 ^ (n + 1)
      simp only [bitsToNat, this, Nat.pow_succ]
      omega

/-- **The fused requantization's exchange computes the class's tree**: the canonical 128-leaf tree is the combine of the two
half trees (the engine's `treeC` splits on the most significant bit: the first half is elements 0 .. 63, the second 64 .. 127).
The two warpgroups each hold one half's register tree and combine them after the exchange, in this order. -/
theorem blockAmax_halves (row : Nat → Term) (b : Nat) :
    blockAmax row b = ocomb .fmaxc (halfAmax row b 0) (halfAmax row b 1) := by
  have h0 : treeC .fmaxc 6 (fun bs => blockLeaf row b (bs ++ [false])) = treeC .fmaxc 6 (halfLeaf row b 0) :=
    treeC_congr_eq _ 6 (fun bs h => by
      have hf : bitsToNat (takeN 7 (bs ++ [false])) = bitsToNat (takeN 6 bs) := bitsToNat_takeN_append_false 6 bs h
      show some (leaf row (128 * b + bitsToNat (takeN 7 (bs ++ [false])))) = some (leaf row (128 * b + 64 * 0 + bitsToNat (takeN 6 bs)))
      have e : 128 * b + bitsToNat (takeN 6 bs) = 128 * b + 64 * 0 + bitsToNat (takeN 6 bs) := by omega
      rw [hf, e])
  have h1 : treeC .fmaxc 6 (fun bs => blockLeaf row b (bs ++ [true])) = treeC .fmaxc 6 (halfLeaf row b 1) :=
    treeC_congr_eq _ 6 (fun bs h => by
      have ht : bitsToNat (takeN 7 (bs ++ [true])) = bitsToNat (takeN 6 bs) + 64 := bitsToNat_takeN_append_true 6 bs h
      show some (leaf row (128 * b + bitsToNat (takeN 7 (bs ++ [true])))) = some (leaf row (128 * b + 64 * 1 + bitsToNat (takeN 6 bs)))
      have e : 128 * b + (bitsToNat (takeN 6 bs) + 64) = 128 * b + 64 * 1 + bitsToNat (takeN 6 bs) := by omega
      rw [ht, e])
  show ocomb .fmaxc (treeC .fmaxc 6 (fun bs => blockLeaf row b (bs ++ [false]))) (treeC .fmaxc 6 (fun bs => blockLeaf row b (bs ++ [true])))
     = ocomb .fmaxc (treeC .fmaxc 6 (halfLeaf row b 0)) (treeC .fmaxc 6 (halfLeaf row b 1))
  rw [h0, h1]

/-! ## The chains (`spec_fp8bs.scale_term`, `inv_term`, `quant_elem`)

Python prints binary32 constants as `repr(float(c))`: `fp32(1e-12)` is `9.999999960041972e-13`. -/

def e4m3Max : Term := .const "448.0"
def scaleEps : Term := .const "9.999999960041972e-13"
def one : Term := .const "1.0"

/-- `s = fmaxc(div(amax, 448), 1e-12)`: the block's dequantization scale (`fmaxc` is the `max.f32` the kernel issues, as the
Python spec spells it). -/
def scaleT (amax : Term) : Term := .fmaxc (.div amax e4m3Max) scaleEps

/-- `inv = div(1, s)`: the correctly rounded reciprocal, once per block. -/
def invT (s : Term) : Term := .div one s

def quantPreT (x inv : Term) : Term := .mul x inv

/-- `q = cvt.rn.satfinite.e4m3(mul(x, inv))`, as the element value. -/
def quantElemT (x inv : Term) : Term := .cvt "e4m3" (quantPreT x inv)

/-- `BSQuantSpec.block(row, b)["s"]`. -/
def specS (row : Nat → Term) (b : Nat) : Option Term := (blockAmax row b).map scaleT

/-- `BSQuantSpec.block(row, i / 128)["codes"][i % 128]`. -/
def specCode (row : Nat → Term) (i : Nat) : Option Term :=
  (blockAmax row (i / 128)).map fun a => quantElemT (row i) (invT (scaleT a))

/-! ## Invariance through the chains -/

theorem scaleT_equiv {a a' : Term} (ha : Equiv a' a) : Equiv (scaleT a') (scaleT a) :=
  .fmaxc_congr (.div_congr ha (.refl _)) (.refl _)

theorem invT_equiv {s s' : Term} (hs : Equiv s' s) : Equiv (invT s') (invT s) := .div_congr (.refl _) hs

theorem code_equiv (x : Term) {a a' : Term} (ha : Equiv a' a) :
    Equiv (quantElemT x (invT (scaleT a'))) (quantElemT x (invT (scaleT a))) :=
  .cvt_congr _ (.mul_congr (.refl x) (invT_equiv (scaleT_equiv ha)))

/-- **End-to-end.** For every valid placement of the block digit and every unit, the kernel's block amax exists and the scale
and every code of the block are `Equiv` to the spec's, hence (`eval_equiv`) bitwise equal under any commutative interpretation. -/
theorem block_outputs_invariant (row : Nat → Term) (b : Nat) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [7] ps) (hc : CoordsOK (flatten ps) cs) :
    ∃ a a', blockAmax row b = some a ∧ execC .fmaxc (flatten ps) (blockLeaf row b) cs = some a' ∧
      Equiv (scaleT a') (scaleT a) ∧
      ∀ e, Equiv (quantElemT (row (128 * b + e)) (invT (scaleT a'))) (quantElemT (row (128 * b + e)) (invT (scaleT a))) := by
  obtain ⟨a, ha⟩ := blockAmax_some row b
  have h := block_amax_invariance row b ps cs hp hc
  rw [ha] at h
  obtain ⟨a', hk, he⟩ := OEquiv.some_right h
  exact ⟨a, a', ha, hk, scaleT_equiv he, fun e => code_equiv _ he⟩

/-- Bitwise form of the scale. -/
theorem s_bits_equal {α : Type} (P : Prims α) (row : Nat → Term) (b : Nat) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [7] ps) (hc : CoordsOK (flatten ps) cs) :
    ∃ a a', blockAmax row b = some a ∧ execC .fmaxc (flatten ps) (blockLeaf row b) cs = some a' ∧
      eval P (scaleT a') = eval P (scaleT a) := by
  obtain ⟨a, a', ha, hk, hs, _⟩ := block_outputs_invariant row b ps cs hp hc
  exact ⟨a, a', ha, hk, eval_equiv P hs⟩

/-! ## The 32-wide instance (`deepseek_fp8_block_size = 32`: the `moe.e4m3bs32` classes' quantizer, `spec_fp8bs` at `block = 32`)

The same chains over a block of 32 leaves: the amax digit is `[5]`, the standalone kernel's placement `[(3, reg), (2, lane)]`
(eight elements per lane, four lanes per block). A different class on the Python side (its fingerprint names the block); the
generated `fp8bsq32*` cases couple its terms. -/

def blockLeaf32 (row : Nat → Term) (b : Nat) (bs : List Bool) : Option Term :=
  some (leaf row (32 * b + bitsToNat (takeN 5 bs)))

def blockAmax32 (row : Nat → Term) (b : Nat) : Option Term := treeC .fmaxc 5 (blockLeaf32 row b)

theorem blockAmax32_some (row : Nat → Term) (b : Nat) : ∃ a, blockAmax32 row b = some a :=
  treeC_some .fmaxc 5 _ (fun _ => ⟨_, rfl⟩)

theorem elem_lt_32 (bs : List Bool) : bitsToNat (takeN 5 bs) < 32 := by
  have h1 := Quant.bitsToNat_lt (takeN 5 bs)
  have h2 := Quant.takeN_length_le 5 bs
  have h3 : 2 ^ (takeN 5 bs).length ≤ 2 ^ 5 := Nat.pow_le_pow_right (by decide) h2
  omega

theorem block_amax_invariance32 (row : Nat → Term) (b : Nat) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [5] ps) (hc : CoordsOK (flatten ps) cs) :
    OEquiv (execC .fmaxc (flatten ps) (blockLeaf32 row b) cs) (blockAmax32 row b) := by
  have h := config_invarianceC .fmaxc [5] ps (blockLeaf32 row b) cs hp hc
  rw [canonicalDigitsC_eq_treeC] at h
  exact h

theorem placementElem_ok32 (vb lb : Nat) (h : vb + lb = 5) : PlacementOK [5] (placementElem vb lb) := by
  unfold placementElem
  refine ⟨?_, trivial⟩
  show vb + (lb + 0) = 5
  omega

def specS32 (row : Nat → Term) (b : Nat) : Option Term := (blockAmax32 row b).map scaleT

def specCode32 (row : Nat → Term) (i : Nat) : Option Term :=
  (blockAmax32 row (i / 32)).map fun a => quantElemT (row i) (invT (scaleT a))

theorem block_outputs_invariant32 (row : Nat → Term) (b : Nat) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [5] ps) (hc : CoordsOK (flatten ps) cs) :
    ∃ a a', blockAmax32 row b = some a ∧ execC .fmaxc (flatten ps) (blockLeaf32 row b) cs = some a' ∧
      Equiv (scaleT a') (scaleT a) ∧
      ∀ e, Equiv (quantElemT (row (32 * b + e)) (invT (scaleT a'))) (quantElemT (row (32 * b + e)) (invT (scaleT a))) := by
  obtain ⟨a, ha⟩ := blockAmax32_some row b
  have h := block_amax_invariance32 row b ps cs hp hc
  rw [ha] at h
  obtain ⟨a', hk, he⟩ := OEquiv.some_right h
  exact ⟨a, a', ha, hk, scaleT_equiv he, fun e => code_equiv _ he⟩

theorem block_locality32 (row row' : Nat → Term) (b : Nat) (h : ∀ j, j < 32 → row (32 * b + j) = row' (32 * b + j)) :
    specS32 row b = specS32 row' b ∧ ∀ j, j < 32 → specCode32 row (32 * b + j) = specCode32 row' (32 * b + j) := by
  have hl : blockLeaf32 row b = blockLeaf32 row' b := by
    funext bs
    simp only [blockLeaf32, leaf]
    rw [h _ (elem_lt_32 bs)]
  have ha : blockAmax32 row b = blockAmax32 row' b := by unfold blockAmax32; rw [hl]
  refine ⟨by unfold specS32; rw [ha], fun j hj => ?_⟩
  unfold specCode32
  have hb : (32 * b + j) / 32 = b := by omega
  rw [hb, ha, h j hj]

/-! ## The launch model: a block's outputs depend on its 128 inputs alone -/

/-- Block `b` of a row reads elements `128 b .. 128 b + 127` only: two rows that agree there give the same scale and codes. -/
theorem block_locality (row row' : Nat → Term) (b : Nat) (h : ∀ j, j < 128 → row (128 * b + j) = row' (128 * b + j)) :
    specS row b = specS row' b ∧ ∀ j, j < 128 → specCode row (128 * b + j) = specCode row' (128 * b + j) := by
  have hl : blockLeaf row b = blockLeaf row' b := by
    funext bs
    simp only [blockLeaf, leaf]
    rw [h _ (elem_lt_128 bs)]
  have ha : blockAmax row b = blockAmax row' b := by unfold blockAmax; rw [hl]
  refine ⟨by unfold specS; rw [ha], fun j hj => ?_⟩
  unfold specCode
  have hb : (128 * b + j) / 128 = b := by omega
  rw [hb, ha, h j hj]

/-- A row's outputs under the class: the scales and codes of its blocks (K = 128 nb). -/
def rowOut (nb : Nat) (row : Nat → Term) : (Nat → Option Term) × (Nat → Option Term) :=
  (fun b => if b < nb then specS row b else none, fun i => if i < 128 * nb then specCode row i else none)

def launch (M : Nat) (rows : Fin M → Nat → Term) (nb : Nat) (r : Fin M) := rowOut nb (rows r)

theorem determinism (M : Nat) (rows : Fin M → Nat → Term) (nb : Nat) (r : Fin M) : launch M rows nb r = launch M rows nb r := rfl

theorem batch_invariance (nb : Nat) {M₁ M₂ : Nat} (r₁ : Fin M₁ → Nat → Term) (r₂ : Fin M₂ → Nat → Term) (i : Fin M₁) (j : Fin M₂)
    (h : r₁ i = r₂ j) : launch M₁ r₁ nb i = launch M₂ r₂ nb j := by unfold launch; rw [h]

theorem permutation_invariance (nb : Nat) {M : Nat} (rows : Fin M → Nat → Term) (σ : Fin M → Fin M) (i : Fin M) :
    launch M (rows ∘ σ) nb i = launch M rows nb (σ i) := rfl

theorem subbatch_invariance (nb : Nat) {M n : Nat} (rows : Fin M → Nat → Term) (sel : Fin n → Fin M) (i : Fin n) :
    launch n (rows ∘ sel) nb i = launch M rows nb (sel i) := rfl

end BIK.FP8BQ
