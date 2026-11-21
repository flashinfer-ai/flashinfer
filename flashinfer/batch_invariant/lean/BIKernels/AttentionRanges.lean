-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.Attention

/-!
# Paged-KV attention with bidirectional ranges: the ranged class of `attention.bf16`

Mirrors the ranged rows of `flashinfer/attention/batch_invariant/spec.py` (`RowSpec.brange`). A row of the ranged class has
one more input than a row of the causal class (`Attention.lean`): a key range `[ra, rb)` of its request. It sees its causal
keys `[lo, p]` (`lo` from the sliding window) united with the range, and everything else is the one dataflow of
`Attention.lean` stated over a row's visible set (`Attn.VParams`): the ranged row is the instance `Params.toV` of it, with the
same tile step, the same chunks on the absolute grid, the same merge in chunk order from the init state, the same epilogue. A
tile without a visible key is absent and a chunk without a present tile is absent, so a range behind the window or ahead of
the position leaves absent tiles and chunks between the two parts.

What the module states:

* a row without a range (`rb ≤ ra`), and a row whose range lies inside its causal keys, have the causal class's term
  (`no_range_is_causal`, `range_inside_causal_is_causal`): tokens outside every range are unaffected by the class;
* the split kernel computes the fused term for every chunk-to-CTA assignment, and every configuration computes the term;
* the launch model: a row's term is a function of its own position, window, range and identity (R1, R2);
* the explicit-key-set row whose bits are the ranged row's visible set has the ranged row's term
  (`key_set_row_of_range_row`): the ranged class is the explicit-key-set class on the sets `causal ∪ [ra, rb)`;
* the range is an input that reaches the term (decided on toy rows): the ranged class is not the causal class.
-/
namespace BIK.AttnR
open BIK.Attn

/-- The fixed data of one logical row of the ranged class: `Attn.Params` and the row's key range `[ra, rb)`
(`rb ≤ ra`: no range). -/
structure Params where
  steps : Nat
  tbits : Nat
  C : Nat
  p : Nat
  win : Int
  ra : Nat
  rb : Nat
  row : Nat
  d : Nat

/-- The causal class's row at the same position, window and identity. -/
def Params.causal (P : Params) : Attn.Params := ⟨P.steps, P.tbits, P.C, P.p, P.win, P.row, P.d, true, P.p + 1⟩

/-! The geometry of a ranged row as functions of `(tbits, C, p, win, ra, rb)` alone. -/

def visibleG (p : Nat) (win : Int) (ra rb k : Nat) : Bool := Attn.visibleG p win k || decide (ra ≤ k ∧ k < rb)

/-- One past the last visible key: `p + 1`, or the range's end when the range reaches further. -/
def endG (p ra rb : Nat) : Nat := if ra < rb then max (p + 1) rb else p + 1

/-- The ranged row as a visible-set row (`Attn.VParams`): its visible keys and one past the last of them. -/
def Params.toV (P : Params) : VParams :=
  ⟨P.steps, P.tbits, P.C, endG P.p P.ra P.rb, visibleG P.p P.win P.ra P.rb, P.row, P.d⟩

def Params.visible (P : Params) (k : Nat) : Bool := visibleG P.p P.win P.ra P.rb k
def Params.tilesOfChunk (P : Params) (j : Nat) : List Nat := P.toV.tilesOfChunk j
def Params.presentChunks (P : Params) : List Nat := P.toV.presentChunks

/-- The per-chunk fold: the present tiles of chunk `j`, ascending, from the init state (`Attn.vChunkState`). -/
def chunkState (P : Params) (j : Nat) : State := vChunkState P.toV j

/-- The fused path: present chunks merged in index order from the init state (`Attn.vFused`). -/
def fused (P : Params) : State := vFused P.toV

/-- The ranged row's term: the visible-set formulation at its keys. A ranged row always sees its own position, so the
`+0` rule of a row without a key never applies to it (`output_is_vOutput`). -/
def output (P : Params) : ATerm := vTerm P.toV

theorem output_is_vOutput (P : Params) (hne : P.presentChunks ≠ []) : vOutput P.toV = output P :=
  vOutput_eq_term P.toV hne

/-! ## The split kernel -/

def workspace (P : Params) (_σ : Nat → Nat) (j : Nat) : State := chunkState P j

/-- The fixup folds the present slots in index order (`Attn.fixup`): the present chunks of a row are those whose slot holds
a key of the row, the list `presentChunks` -- the one rule of `Attn.vSplitKernel`, shared with the explicit-key-set class. -/
def splitKernel (P : Params) (σ : Nat → Nat) : ATerm := epilogue (fixup (workspace P σ) P.presentChunks)

theorem splitKernel_eq_fused (P : Params) (σ : Nat → Nat) : splitKernel P σ = output P := rfl

theorem splitKernel_assignment_irrelevant (P : Params) (σ τ : Nat → Nat) : splitKernel P σ = splitKernel P τ := rfl

/-! ## Rows whose range adds no key are rows of the causal class -/

theorem tilesOfChunkV_causal (tbits C p : Nat) (win : Int) (j : Nat) :
    tilesOfChunkV tbits C (Attn.visibleG p win) j = Attn.tilesOfChunkR tbits C (Attn.loOf p win) (p + 1) j := rfl

theorem presentChunksV_causal (tbits C p : Nat) (win : Int) :
    presentChunksV tbits C (p + 1) (Attn.visibleG p win) = Attn.presentChunksR tbits C (Attn.loOf p win) (p + 1) := rfl

/-- A ranged row whose visible keys and last key are the causal row's is the causal row as a visible-set row, and has its
term. -/
theorem output_eq_causal (P : Params) (hv : P.visible = Attn.visibleG P.p P.win) (hL : endG P.p P.ra P.rb = P.p + 1) :
    output P = Attn.output P.causal := by
  have hvis : visibleG P.p P.win P.ra P.rb = Attn.visibleG P.p P.win := hv
  have h : P.toV = P.causal.toV := by
    rw [Attn.causal_toV P.causal rfl]
    simp [Params.toV, Params.causal, hvis, hL]
  show vTerm P.toV = vTerm P.causal.toV
  rw [h]

/-- **No range, the causal class**: a row with `rb ≤ ra` has the term of the causal class's row at the same position,
window and identity. Tokens outside every range keep the causal class's bits. -/
theorem no_range_is_causal (P : Params) (h : P.rb ≤ P.ra) : output P = Attn.output P.causal := by
  apply output_eq_causal
  · funext k
    have hk : ¬ (P.ra ≤ k ∧ k < P.rb) := by omega
    simp [Params.visible, visibleG, hk]
  · have hlt : ¬ (P.ra < P.rb) := by omega
    simp [endG, hlt]

/-- **A range inside the causal keys adds nothing**: the same term again. -/
theorem range_inside_causal_is_causal (P : Params) (hlo : Attn.loOf P.p P.win ≤ P.ra) (hhi : P.rb ≤ P.p + 1) :
    output P = Attn.output P.causal := by
  apply output_eq_causal
  · funext k
    simp only [Params.visible, visibleG, Attn.visibleG, Attn.visibleR]
    by_cases hk : P.ra ≤ k ∧ k < P.rb
    · have hc : Attn.loOf P.p P.win ≤ k ∧ k < P.p + 1 := by omega
      simp [hk, hc]
    · simp [hk]
  · unfold endG
    split <;> omega

/-! ## The launch model -/

/-- A row of a launch: the token's absolute position, its range, its head and its identity. -/
structure Row where
  p : Nat
  ra : Nat
  rb : Nat
  h : Nat
  row : Nat
  deriving DecidableEq, Repr

def rowOutput (steps tbits C : Nat) (win : Int) (d : Nat) (r : Row) : ATerm :=
  output ⟨steps, tbits, C, r.p, win, r.ra, r.rb, r.row, d⟩

def launch (M : Nat) (rows : Fin M → Row) (steps tbits C : Nat) (win : Int) (d : Nat) (r : Fin M) : ATerm :=
  launchOf (rowOutput steps tbits C win d) M rows r

/-- **Batch invariance (R2)**: equal rows -- position, range, head, identity -- give equal outputs whatever the batch
size, the row's place in it or the other rows' ranges. -/
theorem batch_invariance (steps tbits C : Nat) (win : Int) (d : Nat) {M₁ M₂ : Nat}
    (rows₁ : Fin M₁ → Row) (rows₂ : Fin M₂ → Row) (r₁ : Fin M₁) (r₂ : Fin M₂) (h : rows₁ r₁ = rows₂ r₂) :
    launch M₁ rows₁ steps tbits C win d r₁ = launch M₂ rows₂ steps tbits C win d r₂ :=
  launchOf_batch_invariance _ rows₁ rows₂ r₁ r₂ h

/-- **Row locality** (`Attn.launchOf_row_locality`): two launches that agree on a row agree on its output, whatever the
ranges of the other rows. -/
theorem row_locality (steps tbits C : Nat) (win : Int) (d : Nat) {M : Nat} (rows rows' : Fin M → Row) (r : Fin M)
    (h : rows r = rows' r) : launch M rows steps tbits C win d r = launch M rows' steps tbits C win d r :=
  launchOf_row_locality _ rows rows' r h

theorem permutation_invariance (steps tbits C : Nat) (win : Int) (d : Nat) {M : Nat} (rows : Fin M → Row) (σ : Fin M → Fin M) (r : Fin M) :
    launch M (rows ∘ σ) steps tbits C win d r = launch M rows steps tbits C win d (σ r) := rfl

theorem subbatch_invariance (steps tbits C : Nat) (win : Int) (d : Nat) {M k : Nat} (rows : Fin M → Row) (sel : Fin k → Fin M) (i : Fin k) :
    launch k (rows ∘ sel) steps tbits C win d i = launch M rows steps tbits C win d (sel i) := rfl

/-- A row without a range in a ranged launch is the causal launch's row, whatever ranges the other rows carry. -/
theorem launch_row_without_range (M : Nat) (rows : Fin M → Row) (steps tbits C : Nat) (win : Int) (d : Nat) (r : Fin M)
    (h : (rows r).rb ≤ (rows r).ra) :
    launch M rows steps tbits C win d r
      = Attn.rowOutput true steps tbits C win d ⟨(rows r).p, (rows r).p + 1, (rows r).h, (rows r).row⟩ :=
  no_range_is_causal ⟨steps, tbits, C, (rows r).p, win, (rows r).ra, (rows r).rb, (rows r).row, d⟩ h

/-! ## The ranged row is a row of the explicit-key-set class -/

/-- No key at or past `endG` is visible. -/
theorem not_visible_past_end (p : Nat) (win : Int) (ra rb k : Nat) (hk : endG p ra rb ≤ k) :
    visibleG p win ra rb k = false := by
  have hc : ¬ (Attn.loOf p win ≤ k ∧ k < p + 1) := by
    unfold endG at hk
    split at hk <;> omega
  have hr : ¬ (ra ≤ k ∧ k < rb) := by
    unfold endG at hk
    split at hk <;> omega
  simp [visibleG, Attn.visibleG, Attn.visibleR, hc, hr]

/-- **A key-range row equals the explicit-key-set row with the same visible set**: the mask row whose bits are
`causal ∪ [ra, rb)` over `[0, endG)` has the ranged row's term (`Attn.key_set_row_of_visible_set`), so the two classes give
that row the same bits. -/
theorem key_set_row_of_range_row (P : Params) (hne : P.presentChunks ≠ []) :
    Attn.maskOutput (Attn.MParams.ofV P.toV) = output P :=
  Attn.key_set_row_of_visible_set P.toV (fun k hk => not_visible_past_end P.p P.win P.ra P.rb k hk) hne

/-! ## The range reaches the term -/

/-- **The ranged class is not the causal class**: a range ahead of the position is a different term (a toy row, decided). -/
theorem output_depends_on_range :
    output ⟨1, 2, 8, 9, -1, 12, 14, 0, 0⟩ ≠ Attn.output ⟨1, 2, 8, 9, -1, 0, 0, true, 10⟩ := by native_decide

/-- A row inside its range with no window sees the prefix up to the range's last key: the term of the causal row at that
key, for the same query row (a toy row, decided). -/
theorem range_around_position_is_causal_at_end :
    output ⟨1, 2, 8, 5, -1, 3, 12, 0, 0⟩ = Attn.output ⟨1, 2, 8, 11, -1, 0, 0, true, 12⟩ := by native_decide

/-! ## The configuration space -/

/-- A ranged row's output under a configuration (`Attn.Config`): the fused fold or the split kernel, as for the causal
class; the row-visibility block, the row body and the fold's rule for present chunks are placements of this term. -/
def kernel (b : Config) (fusedAuto : Bool) (P : Params) (σ : Nat → Nat) : ATerm :=
  if b.policy == 2 || (b.policy == 0 && fusedAuto) then output P else splitKernel P σ

theorem kernel_eq_output (b : Config) (f : Bool) (P : Params) (σ : Nat → Nat) : kernel b f P σ = output P := by
  unfold kernel
  split <;> rfl

/-- **Configuration invariance (R3)** of the ranged class. -/
theorem config_invariance (b₁ b₂ : Config) (f₁ f₂ : Bool) (P : Params) (σ τ : Nat → Nat) :
    kernel b₁ f₁ P σ = kernel b₂ f₂ P τ := by
  rw [kernel_eq_output, kernel_eq_output]

end BIK.AttnR
