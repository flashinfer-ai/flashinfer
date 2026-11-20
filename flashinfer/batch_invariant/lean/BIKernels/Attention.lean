-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.GEMM
import BIKernels.Export

/-!
# Paged-KV attention: the class `attention.bf16` (per head dim) and the batch-invariance guarantees

Mirrors `flashinfer/attention/batch_invariant/spec.py`. The model is per logical row: the row's identity (its query token and
head, hence its own K / V) is the leaf index of every score and value term, so rows of different KV heads or requests are
different terms even when their values coincide, and the theorems quantify over every row. Per logical row `(p, h)` and output
column `d` the dataflow is

* the score of key `k`: the GEMM's `kfold` over `D` (opaque `mma16` steps, `GEMM.kfold`);
* a **tile step** on the state `(m, l, O)`: row maximum (pairwise `fmax` tree over the tile's visible
  keys), `alpha = ex2(add(mul(m, c), mul(m', nc)))`, `p_k = ex2(fma(s_k, c, mul(m', nc)))` (masked
  keys absent), row sum (pairwise `add` tree), `l' = fma(l, alpha, r)`, `O' = mma`-fold of the
  bf16 `P` slices with `V` from `mul(O, alpha)` (one opaque `mma16` per 16 keys);
* a **chunk**: the tile steps of its present tiles, ascending, from the init `(-inf, +0, +0)`;
* the **row**: the present chunks merged in index order from the init state (`merge`);
* the epilogue `cvt_bf16(add(mul(O, div(1, l)), +0))`.

Tile and chunk boundaries are absolute multiples of `T` and `C`, functions of `D` only: the term of
a token depends on its position `p`, the window and the head, never on the batch, the page table
(paged or ragged KV) or the configuration. What is modelled: the per-row dataflow, the split kernel's
workspace and fixup, the launch model (prefill batch / decode batch -> per-row terms). What is not, and
is covered by the hardware tests: the plumbing (TMA, TMEM, barriers), the instruction semantics (`mma16`,
`ex2`), and the null steps (a kernel's identity steps on rows whose tile has no visible key, exact
up to the sign of an exact-zero accumulator, erased by the epilogue's `add +0`).

The S-expression printer reproduces `term.sexpr` textually; `Cases.lean` (generated) checks
the Python spec's terms node for node by `native_decide`.
-/
namespace BIK.Attn
open BIK.GEMM

/-- Dataflow terms of the attention class. `score row k steps` is the GEMM `kfold` of `steps`
`mma16` K-steps (printed through `GTerm.sexpr`); `mma acc pslice vt d` is one PV K-step, its bf16 P
slice carried as the printed list of its 16 probability terms (the slice is opaque to the fold).
Commutative nodes print their operands sorted, as the Python printer does. -/
inductive ATerm where
  | const (c : String)
  | sym (s : String)
  | score (row k steps : Nat)
  | add (a b : ATerm)
  | mul (a b : ATerm)
  | fma (a b c : ATerm)
  | div (a b : ATerm)
  | fmax (a b : ATerm)
  | ex2 (a : ATerm)
  | cvt (fmt : String) (a : ATerm)
  | mma (acc : ATerm) (pslice : String) (vt d : Nat)
  deriving DecidableEq, Repr, Inhabited

def ATerm.sexpr : ATerm → String
  | .const c => s!"(const {c})"
  | .sym s => s!"(sym {s})"
  | .score row k steps => (kfold ⟨row⟩ ⟨k⟩ steps).sexpr
  | .add a b => let (p, q) := sortPair a.sexpr b.sexpr; s!"(add {p} {q})"
  | .mul a b => let (p, q) := sortPair a.sexpr b.sexpr; s!"(mul {p} {q})"
  | .fma a b c => let (p, q) := sortPair a.sexpr b.sexpr; s!"(fma {p} {q} {c.sexpr})"
  | .div a b => s!"(div {a.sexpr} {b.sexpr})"
  | .fmax a b => let (p, q) := sortPair a.sexpr b.sexpr; s!"(fmax {p} {q})"
  | .ex2 a => s!"(ex2 {a.sexpr})"
  | .cvt f a => s!"(cvt {f} {a.sexpr})"
  | .mma acc ps vt d => s!"(mma16 {acc.sexpr} (op:pv_kslice (op:p_slice{ps}) (op:v_slice {vt} {d})))"

/-! ## Pairwise trees over the key digit (`canon._pairwise`: adjacent pairs, low index first,
absent leaves dropped) -/

def oComb (f : ATerm → ATerm → ATerm) : Option ATerm → Option ATerm → Option ATerm
  | none, b => b
  | some a, none => some a
  | some a, some b => some (f a b)

def pairUp (f : ATerm → ATerm → ATerm) : List (Option ATerm) → List (Option ATerm)
  | a :: b :: rest => oComb f a b :: pairUp f rest
  | l => l

/-- The tree over `2^n` leaves: `n` rounds of adjacent pairing. -/
def ptree (f : ATerm → ATerm → ATerm) : Nat → List (Option ATerm) → Option ATerm
  | 0, xs => xs.headD none
  | n+1, xs => ptree f n (pairUp f xs)

/-! ## The row, its tiles and chunks -/

/-- The fixed data of one logical row's dataflow: head dim as `steps = D/16`, tile `T = 2^tbits`,
chunk `C`, the row's position `p` (so `L = p + 1`), FlashInfer's `window_left` (negative = none),
the row id of the Q leaves, the output column `d`. -/
structure Params where
  steps : Nat
  tbits : Nat
  C : Nat
  p : Nat
  win : Int
  row : Nat
  d : Nat

/-! The geometry of a row (visible keys, present tiles, present chunks) as functions of
`(tbits, C, p, win)` alone: the page table, the batch and the configuration do not enter. -/

def loOf (p : Nat) (win : Int) : Nat := if win < 0 then 0 else ((p : Int) - win).toNat

def visibleG (p : Nat) (win : Int) (k : Nat) : Bool := decide (loOf p win ≤ k ∧ k < p + 1)

def tilePresentG (tbits p : Nat) (win : Int) (t : Nat) : Bool :=
  (List.range (2 ^ tbits)).any (fun i => visibleG p win (t * 2 ^ tbits + i))

def tilesOfChunkG (tbits C p : Nat) (win : Int) (j : Nat) : List Nat :=
  ((List.range (C / 2 ^ tbits)).map (fun i => j * (C / 2 ^ tbits) + i)).filter (tilePresentG tbits p win)

def presentChunksG (tbits C p : Nat) (win : Int) : List Nat :=
  (List.range ((p + 1 + C - 1) / C)).filter (fun j => !(tilesOfChunkG tbits C p win j).isEmpty)

def Params.T (P : Params) : Nat := 2 ^ P.tbits
def Params.visible (P : Params) (k : Nat) : Bool := visibleG P.p P.win k
def Params.tilesOfChunk (P : Params) (j : Nat) : List Nat := tilesOfChunkG P.tbits P.C P.p P.win j
def Params.presentChunks (P : Params) : List Nat := presentChunksG P.tbits P.C P.p P.win

structure State where
  m : ATerm
  l : ATerm
  o : ATerm
  deriving DecidableEq, Repr, Inhabited

def c : ATerm := .sym "c"
def nc : ATerm := .sym "nc"
def initState : State := ⟨.const "-inf", .const "0.0", .const "0.0"⟩

/-- `spec.alpha_term`: `ex2(add(mul(m_old, c), mul(m_new, nc)))`. -/
def alphaTerm (mOld mNew : ATerm) : ATerm := .ex2 (.add (.mul mOld c) (.mul mNew nc))

/-- The PV fold of one tile from `o0`: one opaque step per 16 keys, the slice's bf16 probabilities
printed into the step (the Python term carries them as the `p_slice` operands). -/
def pvFold (o0 : ATerm) (pbf : List ATerm) (vt0 d : Nat) : ATerm :=
  let n := pbf.length / 16
  (List.range n).foldl (fun acc j =>
    let slice := (pbf.drop (16 * j)).take 16
    .mma acc (String.join (slice.map (fun q => " " ++ q.sexpr))) (vt0 + j) d) o0

/-- `spec.tile_step` on a present tile `t`. -/
def tileStep (P : Params) (st : State) (t : Nat) : State :=
  let scores : List (Option ATerm) :=
    (List.range P.T).map (fun i => if P.visible (t * P.T + i) then some (.score P.row (t * P.T + i) P.steps) else none)
  let mT := (ptree ATerm.fmax P.tbits scores).getD (.const "-inf")
  let mNew := ATerm.fmax st.m mT
  let alpha := alphaTerm st.m mNew
  let nms := ATerm.mul mNew nc
  let probs : List (Option ATerm) := scores.map (fun s => s.map (fun s => .ex2 (.fma s c nms)))
  let rT := (ptree ATerm.add P.tbits probs).getD (.const "0.0")
  let lNew := ATerm.fma st.l alpha rT
  let pbf := probs.map (fun q => ATerm.cvt "bf16" (q.getD (.const "0.0")))
  let oNew := pvFold (.mul st.o alpha) pbf (t * (P.T / 16)) P.d
  ⟨mNew, lNew, oNew⟩

/-- `spec.merge`: the running state `a` with the finished chunk `b`. -/
def merge (a b : State) : State :=
  let m := ATerm.fmax a.m b.m
  let nms := ATerm.mul m nc
  let a1 := ATerm.ex2 (.add (.mul a.m c) nms)
  let a2 := ATerm.ex2 (.add (.mul b.m c) nms)
  ⟨m, .fma b.l a2 (.mul a.l a1), .fma b.o a2 (.mul a.o a1)⟩

/-- The per-chunk fold: the present tiles of chunk `j`, ascending, from the init state. -/
def chunkState (P : Params) (j : Nat) : State := (P.tilesOfChunk j).foldl (tileStep P) initState

/-- The fused (one-CTA) path: present chunks merged in index order from the init state. -/
def fused (P : Params) : State := P.presentChunks.foldl (fun acc j => merge acc (chunkState P j)) initState

/-- `spec.output_elem`. -/
def epilogue (st : State) : ATerm := .cvt "bf16" (.add (.mul st.o (.div (.const "1.0") st.l)) (.const "0.0"))

def output (P : Params) : ATerm := epilogue (fused P)

/-! ## The split kernel: one chunk per CTA, an index-addressed workspace, the fixup fold -/

/-- Slot `j` of the workspace holds chunk `j`'s state, whatever CTA `σ` computed it on. -/
def workspace (P : Params) (_σ : Nat → Nat) (j : Nat) : State := chunkState P j

/-- The fixup merges the present slots in index order from the init state (`fixup.py`; the in-kernel
fold by the last-arriving CTA is the same fold). -/
def fixup (ws : Nat → State) (js : List Nat) : State := js.foldl (fun acc j => merge acc (ws j)) initState

def splitKernel (P : Params) (σ : Nat → Nat) : ATerm := epilogue (fixup (workspace P σ) P.presentChunks)

/-- **The split kernel computes the fused class**, for every assignment of chunks to CTAs (and hence
for every unit kind, ring depth and grid the planner may choose). -/
theorem splitKernel_eq_fused (P : Params) (σ : Nat → Nat) : splitKernel P σ = output P := rfl

theorem splitKernel_assignment_irrelevant (P : Params) (σ τ : Nat → Nat) : splitKernel P σ = splitKernel P τ := rfl

/-- The merge is not associative as a dataflow: a CTA may hold exactly one chunk or all of them. A
unit covering a proper prefix merged with a later state would compute a different term. -/
theorem merge_not_assoc :
    let a : State := ⟨.sym "ma", .sym "la", .sym "oa"⟩
    let b : State := ⟨.sym "mb", .sym "lb", .sym "ob"⟩
    let d : State := ⟨.sym "md", .sym "ld", .sym "od"⟩
    merge (merge a b) d ≠ merge a (merge b d) := by decide

/-! ## The launch model: R1, R2 and R4 -/

/-- A row of a launch: the token's absolute position, its head, and the fixed class data. -/
structure Row where
  p : Nat
  h : Nat
  row : Nat
  deriving DecidableEq, Repr

def rowOutput (steps tbits C : Nat) (win : Int) (d : Nat) (r : Row) : ATerm :=
  output ⟨steps, tbits, C, r.p, win, r.row, d⟩

/-- A prefill request: `q_len` tokens whose last one sits at position `kv_len - 1`; token `i` is at
`kv_len - q_len + i` (bottom-right causal). -/
def prefillRow (kv_len q_len i h row : Nat) : Row := ⟨kv_len - q_len + i, h, row⟩

/-- A decode step for a request with `kv_len` keys: the single token at position `kv_len - 1`. -/
def decodeRow (kv_len h row : Nat) : Row := ⟨kv_len - 1, h, row⟩

/-- **Prefill == decode (R4).** The token at position `p` has the same term whether it is row `i` of
a prefill request (`kv_len = p + 1 + (q_len - 1 - i)`) or the decode step of a request with
`p + 1` keys: both are `rowOutput` of `⟨p, h, row⟩`. The paged and the ragged KV layouts hand the
kernel the same keys at the same positions, so they are the same rows of this model. -/
theorem prefill_decode_agree (steps tbits C : Nat) (win : Int) (d : Nat) (kv_len q_len i h row : Nat)
    (hi : i < q_len) (hq : q_len ≤ kv_len) :
    rowOutput steps tbits C win d (prefillRow kv_len q_len i h row)
      = rowOutput steps tbits C win d (decodeRow (kv_len - q_len + i + 1) h row) := by
  unfold prefillRow decodeRow
  simp only [Nat.add_sub_cancel]

/-- The launch applies the per-row term to every row; there is no batch parameter inside. -/
def launch (M : Nat) (rows : Fin M → Row) (steps tbits C : Nat) (win : Int) (d : Nat) (r : Fin M) : ATerm :=
  rowOutput steps tbits C win d (rows r)

/-- **Determinism (R1)**: the same inputs give the same term. -/
theorem determinism (M : Nat) (rows : Fin M → Row) (steps tbits C : Nat) (win : Int) (d : Nat) (r : Fin M) :
    launch M rows steps tbits C win d r = launch M rows steps tbits C win d r := rfl

/-- **Batch invariance (R2)**: equal rows give equal outputs whatever the batch size, position or
neighbours. -/
theorem batch_invariance (steps tbits C : Nat) (win : Int) (d : Nat) {M₁ M₂ : Nat}
    (rows₁ : Fin M₁ → Row) (rows₂ : Fin M₂ → Row) (r₁ : Fin M₁) (r₂ : Fin M₂) (h : rows₁ r₁ = rows₂ r₂) :
    launch M₁ rows₁ steps tbits C win d r₁ = launch M₂ rows₂ steps tbits C win d r₂ := by
  simp only [launch, h]

/-- **Permutation invariance**: permuting the rows permutes the outputs row-wise. -/
theorem permutation_invariance (steps tbits C : Nat) (win : Int) (d : Nat) {M : Nat} (rows : Fin M → Row) (σ : Fin M → Fin M) (r : Fin M) :
    launch M (rows ∘ σ) steps tbits C win d r = launch M rows steps tbits C win d (σ r) := rfl

/-- **Sub-batch invariance**: any selection of rows reproduces exactly those rows' outputs. -/
theorem subbatch_invariance (steps tbits C : Nat) (win : Int) (d : Nat) {M k : Nat} (rows : Fin M → Row) (sel : Fin k → Fin M) (i : Fin k) :
    launch k (rows ∘ sel) steps tbits C win d i = launch M rows steps tbits C win d (sel i) := rfl

/-- **The chunk length must not depend on the batch or the request's total length**: the same
position under `C` and `2C` is a different term (stated on a concrete toy row, decided). -/
theorem output_depends_on_C :
    output ⟨1, 2, 8, 9, -1, 0, 0⟩ ≠ output ⟨1, 2, 16, 9, -1, 0, 0⟩ := by native_decide

/-- **One class per head dim**: the same position under `D = 128` (8 MMA K-steps per score) and `D = 256` (16 steps) is a
different term (the score fold differs; everything else -- `T`, `C`, the folds, the merge, the epilogue -- is shared). Stated
on a concrete toy row, decided. The classes at D = 64 / 128 / 256 are therefore three fingerprints. -/
theorem output_depends_on_D :
    output ⟨8, 2, 8, 9, -1, 0, 0⟩ ≠ output ⟨16, 2, 8, 9, -1, 0, 0⟩ := by native_decide

theorem output_depends_on_D_64 :
    output ⟨4, 2, 8, 9, -1, 0, 0⟩ ≠ output ⟨8, 2, 8, 9, -1, 0, 0⟩ := by native_decide

/-- The visible keys, the tiles and the chunks of a row depend on `(p, win, T, C)` only: the page
table, the batch and the configuration do not appear. -/
theorem boundaries_absolute (P Q : Params) (hp : P.p = Q.p) (hw : P.win = Q.win) (hT : P.tbits = Q.tbits) (hC : P.C = Q.C) :
    P.presentChunks = Q.presentChunks := by
  unfold Params.presentChunks
  rw [hp, hw, hT, hC]

theorem tiles_absolute (P Q : Params) (j : Nat) (hp : P.p = Q.p) (hw : P.win = Q.win) (hT : P.tbits = Q.tbits) (hC : P.C = Q.C) :
    P.tilesOfChunk j = Q.tilesOfChunk j := by
  unfold Params.tilesOfChunk
  rw [hp, hw, hT, hC]

/-! ## The kernel's configuration space

The body's knobs choose placements: the smem ring depth, the number of softmax warpgroups, the planner's unit kind
(`policy`: 0 auto, 1 split, 2 fused), where the split groups' chunk fold runs (`merge`) and which kernel body a batch of
small units takes (`small`). In the model a configuration selects at most WHICH kernel formula a row goes through -- the
fused fold or the split kernel (under some chunk-to-CTA assignment σ) -- and both are the class term; nothing else a knob
decides appears in the term. The generated `Cases.lean` emits the enumerated space per head dim (`attnSpaceD<D>`), checks
every record against the static constraints below (`ConfigOKD`), counts them, and instantiates `space_invariant` on it.
Mirrors `AttnConfig.well_formed(D)` of `attention/batch_invariant/engine.py`. -/

/-- The knob record (`AttnConfig.lean()`): ring depth, unit-kind policy, softmax warpgroups, where a split group's chunk fold
runs (`merge`: 0 auto, 1 the fixup kernel, 2 in-kernel by the unit that completes the group) and the small-unit body (`small`:
0 the row body's replicated rows, 1 the Swaps-AB kernel). The fold is the same `fixup` in chunk order wherever it runs:
`splitKernel_eq_fused` holds for any assignment, so the merge placement never reaches the term. -/
structure Config where
  kvSlots : Nat
  policy : Nat
  softmaxWgs : Nat
  merge : Nat
  small : Nat
  deriving DecidableEq, Repr

/-- The knob constraints at head dim `D`: a KV ring slot is `128 x D` bf16 and the ring holds 128 KB of K/V at every head dim,
so `kvSlots * D ≤ 512` (2..8 slots at D = 64, 2..4 at D = 128, exactly 2 at D = 256); the other knobs do not depend on `D`. -/
def ConfigOKD (D : Nat) (c : Config) : Bool :=
  (2 ≤ c.kvSlots && c.kvSlots * D ≤ 512) && c.policy ≤ 2 && (c.softmaxWgs == 1 || c.softmaxWgs == 2) && c.merge ≤ 2 && c.small ≤ 1

/-- A row's output under a configuration: the fused fold when the planner keeps all chunks in one unit (policy 2, or policy 0
when its SM-count rule `fusedAuto` says so), the split kernel under the assignment `σ` otherwise. Ring depth, warpgroups
and the small-unit body are absent: they never reach the term. -/
def kernel (b : Config) (fusedAuto : Bool) (P : Params) (σ : Nat → Nat) : ATerm :=
  if b.policy == 2 || (b.policy == 0 && fusedAuto) then output P else splitKernel P σ

theorem kernel_eq_output (b : Config) (f : Bool) (P : Params) (σ : Nat → Nat) : kernel b f P σ = output P := by
  unfold kernel
  split <;> rfl

/-- **Configuration invariance (R3)**: every configuration, every planner decision and every chunk-to-CTA assignment
computes the class term. -/
theorem config_invariance (b₁ b₂ : Config) (f₁ f₂ : Bool) (P : Params) (σ τ : Nat → Nat) :
    kernel b₁ f₁ P σ = kernel b₂ f₂ P τ := by
  rw [kernel_eq_output, kernel_eq_output]

theorem space_invariant (space : List Config) :
    ∀ b₁ ∈ space, ∀ b₂ ∈ space, ∀ (f₁ f₂ : Bool) (P : Params) (σ τ : Nat → Nat), kernel b₁ f₁ P σ = kernel b₂ f₂ P τ :=
  fun b₁ _ b₂ _ f₁ f₂ P σ τ => config_invariance b₁ b₂ f₁ f₂ P σ τ

end BIK.Attn
