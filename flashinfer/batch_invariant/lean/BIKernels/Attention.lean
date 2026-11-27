-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.GEMM
import BIKernels.Export

/-!
# Paged-KV attention: the classes `attention.bf16` (per head dim and mask mode) and the batch-invariance guarantees

Mirrors `flashinfer/attention/batch_invariant/spec.py`. The model is per logical row: the row's identity (its query token and
head, hence its own K / V) is the leaf index of every score and value term, so rows of different KV heads or requests are
different terms even when their values coincide, and the theorems quantify over every row. The mask mode is a class dimension:
a causal row at position `p` sees the keys `[lo, p + 1)` (`lo` from the optional window), a non-causal row sees every key of
its request, `[0, kvLen)`; the dataflow below is the same for both and only the visible set differs. Per logical row `(p, h)`
and output column `d` the dataflow is

* the score of key `k`: the GEMM's `kfold` over `D` (opaque `mma16` steps, `GEMM.kfold`);
* a **tile step** on the state `(m, l, O)`: row maximum (pairwise `fmax` tree over the tile's visible
  keys), `alpha = ex2(add(mul(m, c), mul(m', nc)))`, `p_k = ex2(fma(s_k, c, mul(m', nc)))` (masked
  keys absent), row sum (pairwise `add` tree), `l' = fma(l, alpha, r)`, `O' = mma`-fold of the
  bf16 `P` slices with `V` from `mul(O, alpha)` (one opaque `mma16` per 16 keys);
* a **chunk**: the tile steps of its present tiles, ascending, from the init `(-inf, +0, +0)`;
* the **row**: the present chunks merged in index order from the init state (`merge`);
* the epilogue `cvt_bf16(add(mul(O, div(1, l)), +0))`;
* the LSE member, a second output of the row's final `(m, l)`: `add(mul(m, c), lg2(l))` in base 2, its `mul` by
  `fp32(ln 2)` in base e (`lg2` opaque, like `ex2`);
* attention sinks, the member of a call with sinks: one more merge of the row's final `(m, l)` with a virtual key of the
  head's sink logit and zero value (`sinkMerge`), after the last chunk merge and before the epilogue, whose normaliser
  becomes `mul(a, div(1, l'))`; the LSE member is then `add(m', lg2(l'))`.

Both members are functions of the row's final state, so they exist in both mask modes: under the non-causal class a row's LSE
and its sink step read the fold over `[0, kvLen)` and depend on the request's key count, the head and the row's own query alone.

Tile and chunk boundaries are absolute multiples of `T` and `C`, functions of `D` only: the term of
a token depends on its position `p` and the window (causal) or its request's key count (non-causal), and on the head, never
on the batch, the page table (paged or ragged KV) or the configuration. What is modelled: the per-row dataflow, the split kernel's
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
chunk `C`, the row's position `p`, FlashInfer's `window_left` (negative = none), the row id of the Q
leaves, the output column `d`, and the mask mode: `causal = true` gives the visible keys `[lo, p + 1)`,
`causal = false` gives `[0, kvLen)` (every key of the request; `p` and `win` do not enter). -/
structure Params where
  steps : Nat
  tbits : Nat
  C : Nat
  p : Nat
  win : Int
  row : Nat
  d : Nat
  causal : Bool
  kvLen : Nat

/-! The geometry of a row (visible keys, present tiles, present chunks) as functions of its visible range `[lo, L)`, which is
a function of `(causal, p, win, kvLen)` alone: the page table, the batch and the configuration do not enter. -/

def loOf (p : Nat) (win : Int) : Nat := if win < 0 then 0 else ((p : Int) - win).toNat

/-- The first visible key: the window's start under the causal class, `0` under the non-causal one. -/
def loMode (causal : Bool) (p : Nat) (win : Int) : Nat := if causal then loOf p win else 0

/-- One past the last visible key: `p + 1` under the causal class, the request's `kvLen` under the non-causal one. -/
def LMode (causal : Bool) (p kvLen : Nat) : Nat := if causal then p + 1 else kvLen

theorem LMode_true (p k : Nat) : LMode true p k = p + 1 := rfl
theorem LMode_false (p k : Nat) : LMode false p k = k := rfl
theorem loMode_true (p : Nat) (win : Int) : loMode true p win = loOf p win := rfl
theorem loMode_false (p : Nat) (win : Int) : loMode false p win = 0 := rfl

def visibleR (lo L k : Nat) : Bool := decide (lo ≤ k ∧ k < L)

def tilePresentR (tbits lo L t : Nat) : Bool :=
  (List.range (2 ^ tbits)).any (fun i => visibleR lo L (t * 2 ^ tbits + i))

def tilesOfChunkR (tbits C lo L j : Nat) : List Nat :=
  ((List.range (C / 2 ^ tbits)).map (fun i => j * (C / 2 ^ tbits) + i)).filter (tilePresentR tbits lo L)

def presentChunksR (tbits C lo L : Nat) : List Nat :=
  (List.range ((L + C - 1) / C)).filter (fun j => !(tilesOfChunkR tbits C lo L j).isEmpty)

/-- What the dataflow reads of a row: its fixed dims, its visible range and its identity. The mask mode and the request
metadata reach the term only through `lo` and `L` (`Params.core`). -/
structure Core where
  steps : Nat
  tbits : Nat
  C : Nat
  lo : Nat
  L : Nat
  row : Nat
  d : Nat

def Params.lo (P : Params) : Nat := loMode P.causal P.p P.win
def Params.L (P : Params) : Nat := LMode P.causal P.p P.kvLen
def Params.core (P : Params) : Core := ⟨P.steps, P.tbits, P.C, P.lo, P.L, P.row, P.d⟩

def Core.T (G : Core) : Nat := 2 ^ G.tbits
def Core.visible (G : Core) (k : Nat) : Bool := visibleR G.lo G.L k
def Core.tilesOfChunk (G : Core) (j : Nat) : List Nat := tilesOfChunkR G.tbits G.C G.lo G.L j
def Core.presentChunks (G : Core) : List Nat := presentChunksR G.tbits G.C G.lo G.L

def Params.T (P : Params) : Nat := P.core.T
def Params.visible (P : Params) (k : Nat) : Bool := P.core.visible k
def Params.tilesOfChunk (P : Params) (j : Nat) : List Nat := P.core.tilesOfChunk j
def Params.presentChunks (P : Params) : List Nat := P.core.presentChunks

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
def tileStep (P : Core) (st : State) (t : Nat) : State :=
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
def chunkState (P : Core) (j : Nat) : State := (P.tilesOfChunk j).foldl (tileStep P) initState

/-- The fused (one-CTA) path: present chunks merged in index order from the init state. -/
def fused (P : Core) : State := P.presentChunks.foldl (fun acc j => merge acc (chunkState P j)) initState

/-- `spec.output_elem`. -/
def epilogue (st : State) : ATerm := .cvt "bf16" (.add (.mul st.o (.div (.const "1.0") st.l)) (.const "0.0"))

def outputCore (P : Core) : ATerm := epilogue (fused P)

/-- The term of one output element of a row: the dataflow over the row's visible range. -/
def output (P : Params) : ATerm := outputCore P.core

/-- Under the causal class the request's key count does not reach the term (only the position and the window do). -/
theorem causal_output_ignores_kvLen (P : Params) (h : P.causal = true) (k : Nat) :
    output { P with kvLen := k } = output P := by
  simp [output, Params.core, Params.L, Params.lo, LMode, loMode, h]

/-- Under the non-causal class the position and the window do not reach the term (only the request's key count does). -/
theorem noncausal_output_ignores_position (P : Params) (h : P.causal = false) (p : Nat) (win : Int) :
    output { P with p := p, win := win } = output P := by
  simp [output, Params.core, Params.L, Params.lo, LMode, loMode, h]

/-! ## The LSE member (`spec.lse_term`) -/

/-- `lse2 = add(mul(m, c), lg2(l))` as `term.sexpr` prints it (`lg2` is the opaque primitive `(op:lg2 l)`; the commutative
`add` sorts its operands). -/
def lse2Sexpr (st : State) : String :=
  let (p, q) := sortPair (ATerm.mul st.m c).sexpr s!"(op:lg2 {st.l.sexpr})"
  s!"(add {p} {q})"

/-- `fp32(ln 2)` as the Python printer spells the constant. -/
def ln2Const : String := "(const 0.6931471824645996)"

/-- `lse_e = mul(lse2, LN2)`. -/
def lseESexpr (st : State) : String :=
  let (p, q) := sortPair (lse2Sexpr st) ln2Const
  s!"(mul {p} {q})"

def lse2 (P : Params) : String := lse2Sexpr (fused P.core)
def lseE (P : Params) : String := lseESexpr (fused P.core)

/-! ## One `(m, l)` for every output column and the LSE

The statistics of the fold never read the output column: `tileStep` forms `m'` and `l'` from the scores (functions of
`steps`, `T`, the visible range, `row`) and the previous `(m, l)`, and `merge` forms them from the two states' `(m, l)`. So the
output columns of a row and its LSE are normalised by one pair whatever the column, the LSE is the same string for every
column's parameters, and `output` -- which has no LSE parameter -- is the same term whether or not the LSE is requested. -/

/-- A row's dataflow data, and its parameters, with another output column. -/
def Core.withD (G : Core) (d : Nat) : Core := { G with d := d }
def Params.withD (P : Params) (d : Nat) : Params := { P with d := d }

theorem Params.withD_core (P : Params) (d : Nat) : (P.withD d).core = P.core.withD d := rfl

theorem tileStep_ml (G : Core) (d : Nat) (st st' : State) (hm : st.m = st'.m) (hl : st.l = st'.l) (t : Nat) :
    (tileStep (G.withD d) st t).m = (tileStep G st' t).m ∧ (tileStep (G.withD d) st t).l = (tileStep G st' t).l := by
  simp only [tileStep, Core.withD, Core.T, Core.visible, hm, hl, and_self]

theorem tiles_ml (G : Core) (d : Nat) (ts : List Nat) (st st' : State) (hm : st.m = st'.m) (hl : st.l = st'.l) :
    (ts.foldl (tileStep (G.withD d)) st).m = (ts.foldl (tileStep G) st').m ∧
    (ts.foldl (tileStep (G.withD d)) st).l = (ts.foldl (tileStep G) st').l := by
  induction ts generalizing st st' with
  | nil => exact ⟨hm, hl⟩
  | cons t ts ih =>
    simp only [List.foldl]
    have h := tileStep_ml G d st st' hm hl t
    exact ih _ _ h.1 h.2

theorem chunkState_ml (G : Core) (d j : Nat) :
    (chunkState (G.withD d) j).m = (chunkState G j).m ∧ (chunkState (G.withD d) j).l = (chunkState G j).l := by
  unfold chunkState
  have ht : (G.withD d).tilesOfChunk j = G.tilesOfChunk j := rfl
  rw [ht]
  exact tiles_ml G d _ initState initState rfl rfl

theorem merge_ml (a a' b b' : State) (ham : a.m = a'.m) (hal : a.l = a'.l) (hbm : b.m = b'.m) (hbl : b.l = b'.l) :
    (merge a b).m = (merge a' b').m ∧ (merge a b).l = (merge a' b').l := by
  simp only [merge, ham, hal, hbm, hbl, and_self]

theorem chunks_ml (G : Core) (d : Nat) (js : List Nat) (acc acc' : State) (hm : acc.m = acc'.m) (hl : acc.l = acc'.l) :
    (js.foldl (fun a j => merge a (chunkState (G.withD d) j)) acc).m = (js.foldl (fun a j => merge a (chunkState G j)) acc').m ∧
    (js.foldl (fun a j => merge a (chunkState (G.withD d) j)) acc).l = (js.foldl (fun a j => merge a (chunkState G j)) acc').l := by
  induction js generalizing acc acc' with
  | nil => exact ⟨hm, hl⟩
  | cons j js ih =>
    simp only [List.foldl]
    have hc := chunkState_ml G d j
    have h := merge_ml acc acc' (chunkState (G.withD d) j) (chunkState G j) hm hl hc.1 hc.2
    exact ih _ _ h.1 h.2

theorem coreStats_independent_of_column (G : Core) (d : Nat) :
    (fused (G.withD d)).m = (fused G).m ∧ (fused (G.withD d)).l = (fused G).l := by
  unfold fused
  have hc : (G.withD d).presentChunks = G.presentChunks := rfl
  rw [hc]
  exact chunks_ml G d _ initState initState rfl rfl

/-- **One pair `(m, l)` for the whole row**, in both mask modes: the final statistics do not depend on the output column. -/
theorem stats_independent_of_column (P : Params) (d : Nat) :
    (fused (P.withD d).core).m = (fused P.core).m ∧ (fused (P.withD d).core).l = (fused P.core).l := by
  rw [Params.withD_core]
  exact coreStats_independent_of_column P.core d

/-- The LSE member of every column's parameters is the same string: it reads the row's one `(m, l)`. -/
theorem lse_independent_of_column (P : Params) (d : Nat) : lse2 (P.withD d) = lse2 P := by
  unfold lse2 lse2Sexpr
  rw [(stats_independent_of_column P d).1, (stats_independent_of_column P d).2]

theorem lseE_independent_of_column (P : Params) (d : Nat) : lseE (P.withD d) = lseE P := by
  unfold lseE lseESexpr lse2Sexpr
  rw [(stats_independent_of_column P d).1, (stats_independent_of_column P d).2]

/-- Under the causal class the request's key count does not reach the LSE (only the position and the window do). -/
theorem causal_lse_ignores_kvLen (P : Params) (h : P.causal = true) (k : Nat) :
    lse2 { P with kvLen := k } = lse2 P ∧ lseE { P with kvLen := k } = lseE P := by
  simp [lse2, lseE, Params.core, Params.L, Params.lo, LMode, loMode, h]

/-- Under the non-causal class the position and the window do not reach the LSE: every query row of a request reads the
same visible keys, so its LSE depends on the request's key count, its head and its own query (the row identity) alone. -/
theorem noncausal_lse_ignores_position (P : Params) (h : P.causal = false) (p : Nat) (win : Int) :
    lse2 { P with p := p, win := win } = lse2 P ∧ lseE { P with p := p, win := win } = lseE P := by
  simp [lse2, lseE, Params.core, Params.L, Params.lo, LMode, loMode, h]

/-- The LSE of the init state: `mul(-inf, c)` and `lg2(+0)`, both `-inf` on the device for the class's `c > 0`. No row of
either class is empty (the planner requires `kv_len >= q_len` under the causal class and a key for every request with
query tokens under the non-causal one), so this is the term's value at the fold's start only. -/
theorem lse_init : lse2Sexpr initState = "(add (mul (const -inf) (sym c)) (op:lg2 (const 0.0)))" := by decide

/-! ## Attention sinks (`spec.sink_merge`)

A call with sinks appends, per head, one virtual key to the softmax denominator: logit `sink` (natural-log units of the
scaled score, so `mul(sink, fp32(log2 e))` in the fold's base-2 units; the scale `c` is not applied to it), value zero. It
is one step on the row's FINAL state, after the chunk fold: `ms = mul(m, c)`, `s2 = mul(sink, LOG2E)`,
`m' = fmax(ms, s2)`, `nm = mul(m', -1)`, `a = ex2(add(ms, nm))`, `b = ex2(add(s2, nm))`, `l' = fma(l, a, b)`. The tile
steps, the chunk states and the chunk merges are those of the class without sinks: `sinkMerge` is applied to the row's
final state (`fused P.core`) and to nothing else, whatever the mask mode: under the non-causal class the sink joins the
request's keys `[0, kvLen)`. -/

def sink : ATerm := .sym "sink"
/-- `fp32(log2 e)` as the Python printer spells the constant. -/
def log2e : ATerm := .const "1.4426950216293335"

/-- The row's statistics after the sink step: the maximum over the keys and the sink in scaled base-2 units, the sum
relative to it, and the rescale `a` of the keys' accumulator. -/
structure SinkState where
  m : ATerm
  l : ATerm
  a : ATerm
  deriving DecidableEq, Repr, Inhabited

/-- `spec.sink_merge`. -/
def sinkMerge (st : State) : SinkState :=
  let ms := ATerm.mul st.m c
  let s2 := ATerm.mul sink log2e
  let m := ATerm.fmax ms s2
  let nm := ATerm.mul m (.const "-1.0")
  let a := ATerm.ex2 (.add ms nm)
  let b := ATerm.ex2 (.add s2 nm)
  ⟨m, .fma st.l a b, a⟩

/-- `spec.output_elem_sink`: the epilogue with the normaliser `mul(a, div(1, l'))` in the place of `div(1, l)`. -/
def epilogueSink (st : State) : ATerm :=
  let sk := sinkMerge st
  .cvt "bf16" (.add (.mul st.o (.mul sk.a (.div (.const "1.0") sk.l))) (.const "0.0"))

def outputSink (P : Params) : ATerm := epilogueSink (fused P.core)

/-- `spec.lse_term_sink`: `add(m', lg2(l'))`; `m'` is in scaled units already. -/
def lse2SinkSexpr (st : State) : String :=
  let sk := sinkMerge st
  let (p, q) := sortPair sk.m.sexpr s!"(op:lg2 {sk.l.sexpr})"
  s!"(add {p} {q})"

def lseESinkSexpr (st : State) : String :=
  let (p, q) := sortPair (lse2SinkSexpr st) ln2Const
  s!"(mul {p} {q})"

def lse2Sink (P : Params) : String := lse2SinkSexpr (fused P.core)
def lseESink (P : Params) : String := lseESinkSexpr (fused P.core)

/-- The sink step reads the row's `(m, l)` alone. -/
theorem sinkMerge_ml (st st' : State) (hm : st.m = st'.m) (hl : st.l = st'.l) : sinkMerge st = sinkMerge st' := by
  simp only [sinkMerge, hm, hl]

/-- **One sink step per row**: every output column of a row is normalised by the same merged statistics. -/
theorem sinkStats_independent_of_column (P : Params) (d : Nat) :
    sinkMerge (fused (P.withD d).core) = sinkMerge (fused P.core) :=
  sinkMerge_ml _ _ (stats_independent_of_column P d).1 (stats_independent_of_column P d).2

theorem lseSink_independent_of_column (P : Params) (d : Nat) :
    lse2Sink (P.withD d) = lse2Sink P ∧ lseESink (P.withD d) = lseESink P := by
  unfold lse2Sink lseESink lseESinkSexpr lse2SinkSexpr
  rw [sinkStats_independent_of_column P d]
  exact ⟨rfl, rfl⟩

/-- The member without sinks is the term it was: its epilogue reads the fold's state and no sink. -/
theorem output_without_sinks (P : Params) : output P = epilogue (fused P.core) := rfl

/-- The sink step is taken on the fold's final state, the one the class without sinks normalises. -/
theorem outputSink_of_final_state (P : Params) : outputSink P = epilogueSink (fused P.core) := rfl

/-- **The sink member is another class**: the same row with and without the step is a different term (a toy row, decided). -/
theorem output_depends_on_sink :
    outputSink ⟨1, 2, 8, 9, -1, 0, 0, true, 10⟩ ≠ output ⟨1, 2, 8, 9, -1, 0, 0, true, 10⟩ := by native_decide

/-- The same under the non-causal class (the row of `output_depends_on_mode`). -/
theorem output_depends_on_sink_noncausal :
    outputSink ⟨1, 2, 8, 9, -1, 0, 0, false, 12⟩ ≠ output ⟨1, 2, 8, 9, -1, 0, 0, false, 12⟩ := by native_decide

/-- Under the causal class the request's key count does not reach the sink member. -/
theorem causal_sink_ignores_kvLen (P : Params) (h : P.causal = true) (k : Nat) :
    outputSink { P with kvLen := k } = outputSink P ∧ lse2Sink { P with kvLen := k } = lse2Sink P ∧
    lseESink { P with kvLen := k } = lseESink P := by
  simp [outputSink, lse2Sink, lseESink, Params.core, Params.L, Params.lo, LMode, loMode, h]

/-- Under the non-causal class the position and the window do not reach the sink member: the sink joins the request's key
set, the same for every query row of the request. -/
theorem noncausal_sink_ignores_position (P : Params) (h : P.causal = false) (p : Nat) (win : Int) :
    outputSink { P with p := p, win := win } = outputSink P ∧ lse2Sink { P with p := p, win := win } = lse2Sink P ∧
    lseESink { P with p := p, win := win } = lseESink P := by
  simp [outputSink, lse2Sink, lseESink, Params.core, Params.L, Params.lo, LMode, loMode, h]

/-- **The members of the two classes are different terms**: the LSE and the sink member of the token at position 9 of a
request with 12 keys under the causal and the non-causal class (toy rows, decided), hence their own receipts per mode. -/
theorem lse_depends_on_mode :
    lse2 ⟨1, 2, 8, 9, -1, 0, 0, true, 12⟩ ≠ lse2 ⟨1, 2, 8, 9, -1, 0, 0, false, 12⟩ := by native_decide

theorem sink_depends_on_mode :
    outputSink ⟨1, 2, 8, 9, -1, 0, 0, true, 12⟩ ≠ outputSink ⟨1, 2, 8, 9, -1, 0, 0, false, 12⟩ ∧
    lse2Sink ⟨1, 2, 8, 9, -1, 0, 0, true, 12⟩ ≠ lse2Sink ⟨1, 2, 8, 9, -1, 0, 0, false, 12⟩ := by native_decide

/-- The sink step at the fold's init state: a row without keys would hold the sink alone (`mul(-inf, c) = -inf` for the
class's `c > 0`, so `m' = s2`, `a = ex2(-inf) = +0`, `b = ex2(+0) = 1`, `l' = 1` on the device). -/
theorem sinkMerge_init :
    (sinkMerge initState).m = ATerm.fmax (.mul (.const "-inf") c) (.mul sink log2e) ∧
    (sinkMerge initState).l = ATerm.fma (.const "0.0") (sinkMerge initState).a
      (.ex2 (.add (.mul sink log2e) (.mul (sinkMerge initState).m (.const "-1.0")))) := ⟨rfl, rfl⟩

/-! ## The split kernel: one chunk per CTA, an index-addressed workspace, the fixup fold -/

/-- Slot `j` of the workspace holds chunk `j`'s state, whatever CTA `σ` computed it on. -/
def workspace (P : Params) (_σ : Nat → Nat) (j : Nat) : State := chunkState P.core j

/-- The fixup merges the present slots in index order from the init state (`fixup.py`; the in-kernel
fold by the last-arriving CTA is the same fold). -/
def fixup (ws : Nat → State) (js : List Nat) : State := js.foldl (fun acc j => merge acc (ws j)) initState

def splitKernel (P : Params) (σ : Nat → Nat) : ATerm := epilogue (fixup (workspace P σ) P.presentChunks)

/-- **The split kernel computes the fused class**, for every assignment of chunks to CTAs (and hence
for every unit kind, ring depth and grid the planner may choose). -/
theorem splitKernel_eq_fused (P : Params) (σ : Nat → Nat) : splitKernel P σ = output P := rfl

theorem splitKernel_assignment_irrelevant (P : Params) (σ τ : Nat → Nat) : splitKernel P σ = splitKernel P τ := rfl

/-- The split kernel's LSE: the member of the state the fixup folds (the fixup kernel and the in-kernel fold alike). -/
def splitLse2 (P : Params) (σ : Nat → Nat) : String := lse2Sexpr (fixup (workspace P σ) P.presentChunks)
def splitLseE (P : Params) (σ : Nat → Nat) : String := lseESexpr (fixup (workspace P σ) P.presentChunks)

/-- **The split kernel's LSE is the fused one**, for every assignment of chunks to CTAs. -/
theorem splitLse_eq_fused (P : Params) (σ : Nat → Nat) : splitLse2 P σ = lse2 P ∧ splitLseE P σ = lseE P := ⟨rfl, rfl⟩

/-- The split kernel under sinks: the partial rows carry no sink (`workspace` holds the chunk states of the class without
sinks); the fold that completes the row takes the step once. -/
def splitKernelSink (P : Params) (σ : Nat → Nat) : ATerm := epilogueSink (fixup (workspace P σ) P.presentChunks)
def splitLse2Sink (P : Params) (σ : Nat → Nat) : String := lse2SinkSexpr (fixup (workspace P σ) P.presentChunks)
def splitLseESink (P : Params) (σ : Nat → Nat) : String := lseESinkSexpr (fixup (workspace P σ) P.presentChunks)

/-- **The sink step does not depend on where the row is completed**: the split kernel (the fixup kernel or the in-kernel
fold) computes the fused unit's sink member, output and LSE, for every assignment of chunks to CTAs. -/
theorem splitKernelSink_eq_fused (P : Params) (σ : Nat → Nat) : splitKernelSink P σ = outputSink P := rfl

theorem splitLseSink_eq_fused (P : Params) (σ : Nat → Nat) :
    splitLse2Sink P σ = lse2Sink P ∧ splitLseESink P σ = lseESink P := ⟨rfl, rfl⟩

/-- A sink merged into a chunk's partial instead would be another term: the step is not a chunk merge that could be
taken early (stated on symbolic states, decided). -/
theorem sink_not_in_partials :
    let a : State := ⟨.sym "ma", .sym "la", .sym "oa"⟩
    let b : State := ⟨.sym "mb", .sym "lb", .sym "ob"⟩
    (sinkMerge (merge a b)).l ≠ (merge ⟨(sinkMerge a).m, (sinkMerge a).l, a.o⟩ b).l := by decide

/-- The merge is not associative as a dataflow: a CTA may hold exactly one chunk or all of them. A
unit covering a proper prefix merged with a later state would compute a different term. -/
theorem merge_not_assoc :
    let a : State := ⟨.sym "ma", .sym "la", .sym "oa"⟩
    let b : State := ⟨.sym "mb", .sym "lb", .sym "ob"⟩
    let d : State := ⟨.sym "md", .sym "ld", .sym "od"⟩
    merge (merge a b) d ≠ merge a (merge b d) := by decide

/-! ## The launch model: R1, R2 and R4 -/

/-- A row of a launch: the token's absolute position, its request's key count, its head, and the fixed class data. -/
structure Row where
  p : Nat
  kvLen : Nat
  h : Nat
  row : Nat
  deriving DecidableEq, Repr

/-- The term of a row under the class of mask mode `causal`. -/
def rowOutput (causal : Bool) (steps tbits C : Nat) (win : Int) (d : Nat) (r : Row) : ATerm :=
  output ⟨steps, tbits, C, r.p, win, r.row, d, causal, r.kvLen⟩

/-- A prefill request: `q_len` tokens whose last one sits at position `kv_len - 1`; token `i` is at
`kv_len - q_len + i` (bottom-right causal). -/
def prefillRow (kv_len q_len i h row : Nat) : Row := ⟨kv_len - q_len + i, kv_len, h, row⟩

/-- A decode step for a request with `kv_len` keys: the single token at position `kv_len - 1`. -/
def decodeRow (kv_len h row : Nat) : Row := ⟨kv_len - 1, kv_len, h, row⟩

/-- **Prefill == decode (R4), causal class.** The token at position `p` has the same term whether it is row `i` of
a prefill request (`kv_len = p + 1 + (q_len - 1 - i)`) or the decode step of a request with
`p + 1` keys: the request's key count does not reach the causal term. The paged and the ragged KV layouts hand the
kernel the same keys at the same positions, so they are the same rows of this model. -/
theorem prefill_decode_agree (steps tbits C : Nat) (win : Int) (d : Nat) (kv_len q_len i h row : Nat)
    (hi : i < q_len) (hq : q_len ≤ kv_len) :
    rowOutput true steps tbits C win d (prefillRow kv_len q_len i h row)
      = rowOutput true steps tbits C win d (decodeRow (kv_len - q_len + i + 1) h row) := by
  unfold rowOutput prefillRow decodeRow
  simp [output, Params.core, Params.L, Params.lo, LMode, loMode]

/-- **Prefill == multi-token decode (R4), non-causal class.** Every token of a request with `kv_len` keys has the same term
whichever token of whichever query length it is (its position does not reach the non-causal term), so a prefill row and the
rows of a multi-token decode step of the same request agree. -/
theorem prefill_decode_agree_noncausal (steps tbits C : Nat) (win : Int) (d : Nat) (kv_len q_len q_len' i j h row : Nat) :
    rowOutput false steps tbits C win d (prefillRow kv_len q_len i h row)
      = rowOutput false steps tbits C win d (prefillRow kv_len q_len' j h row) := by
  unfold rowOutput prefillRow
  simp [output, Params.core, Params.L, Params.lo, LMode, loMode]

/-- The LSE member of a launch row under the class of mask mode `causal` (base 2; the base-e member is its `mul` by the
constant). -/
def rowLse (causal : Bool) (steps tbits C : Nat) (win : Int) (r : Row) : String :=
  lse2 ⟨steps, tbits, C, r.p, win, r.row, 0, causal, r.kvLen⟩

/-- **Prefill == decode for the LSE (R4), causal class**, as for the output. -/
theorem lse_prefill_decode_agree (steps tbits C : Nat) (win : Int) (kv_len q_len i h row : Nat)
    (_hi : i < q_len) (_hq : q_len ≤ kv_len) :
    rowLse true steps tbits C win (prefillRow kv_len q_len i h row)
      = rowLse true steps tbits C win (decodeRow (kv_len - q_len + i + 1) h row) := by
  unfold rowLse prefillRow decodeRow
  simp [lse2, Params.core, Params.L, Params.lo, LMode, loMode]

/-- **Prefill == multi-token decode for the LSE (R4), non-causal class**: every token of a request with `kv_len` keys has
the LSE of its own query over the same key set, whichever token of whichever query length it is. -/
theorem lse_prefill_decode_agree_noncausal (steps tbits C : Nat) (win : Int) (kv_len q_len q_len' i j h row : Nat) :
    rowLse false steps tbits C win (prefillRow kv_len q_len i h row)
      = rowLse false steps tbits C win (prefillRow kv_len q_len' j h row) := by
  unfold rowLse prefillRow
  simp [lse2, Params.core, Params.L, Params.lo, LMode, loMode]

/-- **Batch invariance of the LSE (R2)**: equal rows give equal LSE terms whatever the batch (both classes). -/
theorem lse_batch_invariance (causal : Bool) (steps tbits C : Nat) (win : Int) {M₁ M₂ : Nat}
    (rows₁ : Fin M₁ → Row) (rows₂ : Fin M₂ → Row) (r₁ : Fin M₁) (r₂ : Fin M₂) (h : rows₁ r₁ = rows₂ r₂) :
    rowLse causal steps tbits C win (rows₁ r₁) = rowLse causal steps tbits C win (rows₂ r₂) := by
  rw [h]

/-- The sink member of a launch row under the class of mask mode `causal`: its output and its base-2 LSE. The row's head
selects the sink value; in the term it is the leaf `sink`, like the row's own Q / K / V leaves. -/
def rowOutputSink (causal : Bool) (steps tbits C : Nat) (win : Int) (d : Nat) (r : Row) : ATerm :=
  outputSink ⟨steps, tbits, C, r.p, win, r.row, d, causal, r.kvLen⟩
def rowLseSink (causal : Bool) (steps tbits C : Nat) (win : Int) (r : Row) : String :=
  lse2Sink ⟨steps, tbits, C, r.p, win, r.row, 0, causal, r.kvLen⟩

/-- **Prefill == decode with sinks (R4), causal class**, output and LSE. -/
theorem sink_prefill_decode_agree (steps tbits C : Nat) (win : Int) (d : Nat) (kv_len q_len i h row : Nat)
    (_hi : i < q_len) (_hq : q_len ≤ kv_len) :
    rowOutputSink true steps tbits C win d (prefillRow kv_len q_len i h row)
      = rowOutputSink true steps tbits C win d (decodeRow (kv_len - q_len + i + 1) h row) ∧
    rowLseSink true steps tbits C win (prefillRow kv_len q_len i h row)
      = rowLseSink true steps tbits C win (decodeRow (kv_len - q_len + i + 1) h row) := by
  unfold rowOutputSink rowLseSink prefillRow decodeRow
  simp [outputSink, lse2Sink, Params.core, Params.L, Params.lo, LMode, loMode]

/-- **Prefill == multi-token decode with sinks (R4), non-causal class**, output and LSE. -/
theorem sink_prefill_decode_agree_noncausal (steps tbits C : Nat) (win : Int) (d : Nat)
    (kv_len q_len q_len' i j h row : Nat) :
    rowOutputSink false steps tbits C win d (prefillRow kv_len q_len i h row)
      = rowOutputSink false steps tbits C win d (prefillRow kv_len q_len' j h row) ∧
    rowLseSink false steps tbits C win (prefillRow kv_len q_len i h row)
      = rowLseSink false steps tbits C win (prefillRow kv_len q_len' j h row) := by
  unfold rowOutputSink rowLseSink prefillRow
  simp [outputSink, lse2Sink, Params.core, Params.L, Params.lo, LMode, loMode]

/-- **Batch invariance with sinks (R2)**: equal rows give equal outputs and LSEs whatever the batch (both classes). -/
theorem sink_batch_invariance (causal : Bool) (steps tbits C : Nat) (win : Int) (d : Nat) {M₁ M₂ : Nat}
    (rows₁ : Fin M₁ → Row) (rows₂ : Fin M₂ → Row) (r₁ : Fin M₁) (r₂ : Fin M₂) (h : rows₁ r₁ = rows₂ r₂) :
    rowOutputSink causal steps tbits C win d (rows₁ r₁) = rowOutputSink causal steps tbits C win d (rows₂ r₂) ∧
    rowLseSink causal steps tbits C win (rows₁ r₁) = rowLseSink causal steps tbits C win (rows₂ r₂) := by
  rw [h]
  exact ⟨rfl, rfl⟩

/-- The launch applies the per-row term to every row; there is no batch parameter inside. -/
def launch (causal : Bool) (M : Nat) (rows : Fin M → Row) (steps tbits C : Nat) (win : Int) (d : Nat) (r : Fin M) : ATerm :=
  rowOutput causal steps tbits C win d (rows r)

/-- **Determinism (R1)**: the same inputs give the same term (both classes). -/
theorem determinism (causal : Bool) (M : Nat) (rows : Fin M → Row) (steps tbits C : Nat) (win : Int) (d : Nat) (r : Fin M) :
    launch causal M rows steps tbits C win d r = launch causal M rows steps tbits C win d r := rfl

/-- **Batch invariance (R2)**: equal rows give equal outputs whatever the batch size, position or
neighbours (both classes). -/
theorem batch_invariance (causal : Bool) (steps tbits C : Nat) (win : Int) (d : Nat) {M₁ M₂ : Nat}
    (rows₁ : Fin M₁ → Row) (rows₂ : Fin M₂ → Row) (r₁ : Fin M₁) (r₂ : Fin M₂) (h : rows₁ r₁ = rows₂ r₂) :
    launch causal M₁ rows₁ steps tbits C win d r₁ = launch causal M₂ rows₂ steps tbits C win d r₂ := by
  simp only [launch, h]

/-- **Permutation invariance**: permuting the rows permutes the outputs row-wise (both classes). -/
theorem permutation_invariance (causal : Bool) (steps tbits C : Nat) (win : Int) (d : Nat) {M : Nat} (rows : Fin M → Row) (σ : Fin M → Fin M) (r : Fin M) :
    launch causal M (rows ∘ σ) steps tbits C win d r = launch causal M rows steps tbits C win d (σ r) := rfl

/-- **Sub-batch invariance**: any selection of rows reproduces exactly those rows' outputs (both classes). -/
theorem subbatch_invariance (causal : Bool) (steps tbits C : Nat) (win : Int) (d : Nat) {M k : Nat} (rows : Fin M → Row) (sel : Fin k → Fin M) (i : Fin k) :
    launch causal k (rows ∘ sel) steps tbits C win d i = launch causal M rows steps tbits C win d (sel i) := rfl

/-- **The chunk length must not depend on the batch or the request's total length**: the same
position under `C` and `2C` is a different term (stated on a concrete toy row, decided). -/
theorem output_depends_on_C :
    output ⟨1, 2, 8, 9, -1, 0, 0, true, 10⟩ ≠ output ⟨1, 2, 16, 9, -1, 0, 0, true, 10⟩ := by native_decide

/-- **One class per head dim**: the same position under `D = 128` (8 MMA K-steps per score) and `D = 256` (16 steps) is a
different term (the score fold differs; everything else -- `T`, `C`, the folds, the merge, the epilogue -- is shared). Stated
on a concrete toy row, decided. The classes at D = 64 / 128 / 256 are therefore three fingerprints. -/
theorem output_depends_on_D :
    output ⟨8, 2, 8, 9, -1, 0, 0, true, 10⟩ ≠ output ⟨16, 2, 8, 9, -1, 0, 0, true, 10⟩ := by native_decide

theorem output_depends_on_D_64 :
    output ⟨4, 2, 8, 9, -1, 0, 0, true, 10⟩ ≠ output ⟨8, 2, 8, 9, -1, 0, 0, true, 10⟩ := by native_decide

/-- **The mask mode is a class dimension**: the token at position 9 of a request with 12 keys sees the keys `0..9` under the
causal class and `0..11` under the non-causal one -- a different term (stated on a concrete toy row, decided), hence a
different fingerprint per head dim. -/
theorem output_depends_on_mode :
    output ⟨1, 2, 8, 9, -1, 0, 0, true, 12⟩ ≠ output ⟨1, 2, 8, 9, -1, 0, 0, false, 12⟩ := by native_decide

/-- The visible keys, the tiles and the chunks of a row depend on `(causal, p, win, kvLen, T, C)` only: the page
table, the batch and the configuration do not appear. -/
theorem boundaries_absolute (P Q : Params) (hp : P.p = Q.p) (hw : P.win = Q.win) (hT : P.tbits = Q.tbits) (hC : P.C = Q.C)
    (hm : P.causal = Q.causal) (hk : P.kvLen = Q.kvLen) :
    P.presentChunks = Q.presentChunks := by
  simp only [Params.presentChunks, Params.core, Core.presentChunks, Params.lo, Params.L, hp, hw, hT, hC, hm, hk]

theorem tiles_absolute (P Q : Params) (j : Nat) (hp : P.p = Q.p) (hw : P.win = Q.win) (hT : P.tbits = Q.tbits) (hC : P.C = Q.C)
    (hm : P.causal = Q.causal) (hk : P.kvLen = Q.kvLen) :
    P.tilesOfChunk j = Q.tilesOfChunk j := by
  simp only [Params.tilesOfChunk, Params.core, Core.tilesOfChunk, Params.lo, Params.L, hp, hw, hT, hC, hm, hk]

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

/-- A row's output under a configuration in a call with sinks: the fused fold or the split kernel, then the one sink step. -/
def kernelSink (b : Config) (fusedAuto : Bool) (P : Params) (σ : Nat → Nat) : ATerm :=
  if b.policy == 2 || (b.policy == 0 && fusedAuto) then outputSink P else splitKernelSink P σ

/-- **Configuration invariance with sinks (R3)**. -/
theorem sink_config_invariance (b₁ b₂ : Config) (f₁ f₂ : Bool) (P : Params) (σ τ : Nat → Nat) :
    kernelSink b₁ f₁ P σ = kernelSink b₂ f₂ P τ := by
  have h : ∀ (b : Config) (f : Bool) (ρ : Nat → Nat), kernelSink b f P ρ = outputSink P := by
    intro b f ρ
    unfold kernelSink
    split <;> rfl
  rw [h, h]

theorem space_invariant (space : List Config) :
    ∀ b₁ ∈ space, ∀ b₂ ∈ space, ∀ (f₁ f₂ : Bool) (P : Params) (σ τ : Nat → Nat), kernel b₁ f₁ P σ = kernel b₂ f₂ P τ :=
  fun b₁ _ b₂ _ f₁ f₂ P σ τ => config_invariance b₁ b₂ f₁ f₂ P σ τ

end BIK.Attn
