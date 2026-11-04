/-!
# bf16 GEMM on tcgen05: the numerical specification and the batch-invariance guarantees

Mirrors `flashinfer/gemm/batch_invariant/spec.py`. Per output element the dataflow is a sequential fold over K-steps
of 16 through the opaque hardware primitive `mma16` (one `tcgen05.mma` K-step, bf16 inputs, fp32 accumulator). No
commutativity is used anywhere, so equality is plain `=`.

Two consistency classes:
* `kfold`      : one fold over all K-steps from +0 (`gemm.bf16.kfold`);
* `kchunk C`   : K cut into `C` chunks (a function of the weight shape only), each chunk a fold from +0, the partials
                 folded in chunk order with `add` from +0 (`gemm.bf16.kchunk[C]`).

What is modelled: the per-element dataflow, the split kernel's index-addressed workspace, and the batched launch
(rows -> per-row kernel). What is *not* modelled, and is covered by the hardware tests: that the tile shape / cluster
/ raster of the kernel leaves `mma16`'s per-element step sequence untouched (the configuration-agreement test over the
dispatcher's candidates), and that the kernel's first K step, issued without an accumulator input, equals `mma16 zero`
(the two can differ only in the sign of an all-zero result).
-/
namespace BIK.GEMM

/-- A row of A (its data, opaque) and a column of B (opaque), addressed by K-step. -/
structure RowData where
  id : Nat
  deriving DecidableEq, Repr

structure ColData where
  id : Nat
  deriving DecidableEq, Repr

/-- Dataflow terms of the GEMM. `mma16 acc a t b` is one K-step with A's slice `t` of row `a` and B's slice `t` of
column `b`; `zero` is the +0 the accumulator starts from; `add` folds chunk partials; `cvt fmt` is the epilogue's
single correctly rounded conversion to the output format. -/
inductive GTerm where
  | zero
  | mma16 (acc : GTerm) (a : RowData) (t : Nat) (b : ColData)
  | add (acc p : GTerm)
  | cvt (fmt : String) (x : GTerm)
  deriving DecidableEq, Repr

/-- Fold the K-steps `t0, t0+1, ..., t0+n-1` in ascending order onto `acc`. -/
def foldSteps (a : RowData) (b : ColData) (acc : GTerm) (t0 : Nat) : Nat → GTerm
  | 0 => acc
  | n+1 => GTerm.mma16 (foldSteps a b acc t0 n) a (t0 + n) b

/-- Class `kfold`: `steps` K-steps from +0. -/
def kfold (a : RowData) (b : ColData) (steps : Nat) : GTerm :=
  foldSteps a b .zero 0 steps

/-- Partial of chunk `c` (each chunk has `kc` K-steps), from +0. -/
def chunkPartial (a : RowData) (b : ColData) (kc : Nat) (c : Nat) : GTerm :=
  foldSteps a b .zero (c * kc) kc

/-- Fold the first `n` chunk partials in chunk order onto `acc` with `add`. -/
def foldChunks (a : RowData) (b : ColData) (kc : Nat) (acc : GTerm) : Nat → GTerm
  | 0 => acc
  | n+1 => GTerm.add (foldChunks a b kc acc n) (chunkPartial a b kc n)

/-- Class `kchunk C`: `C` chunks of `kc` K-steps. -/
def kchunk (a : RowData) (b : ColData) (C kc : Nat) : GTerm :=
  foldChunks a b kc .zero C

/-! ## The split kernel: chunks on different CTAs, an index-addressed workspace, a fixup -/

/-- The workspace after the mainloop: slot `c` holds chunk `c`'s partial, whatever CTA `σ` assigned it to (`σ` is
any assignment of chunks to CTAs). -/
def workspace (a : RowData) (b : ColData) (kc : Nat) (_σ : Nat → Nat) (c : Nat) : GTerm :=
  chunkPartial a b kc c

/-- The fixup folds the workspace slots in index order from +0. -/
def fixup (ws : Nat → GTerm) : Nat → GTerm
  | 0 => .zero
  | n+1 => GTerm.add (fixup ws n) (ws n)

def splitKernel (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) : GTerm :=
  fixup (workspace a b kc σ) C

/-- **The split kernel computes the chunked class**, for every assignment of chunks to CTAs (and hence for every
tile shape / cluster / raster, which only change `σ`). -/
theorem splitKernel_eq_kchunk (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    splitKernel a b C kc σ = kchunk a b C kc := by
  unfold splitKernel kchunk
  induction C with
  | zero => rfl
  | succ n ih => simp only [fixup, foldChunks, workspace, ih]

theorem splitKernel_assignment_irrelevant (a : RowData) (b : ColData) (C kc : Nat) (σ τ : Nat → Nat) :
    splitKernel a b C kc σ = splitKernel a b C kc τ := by
  rw [splitKernel_eq_kchunk, splitKernel_eq_kchunk]

/-! ## Semantics: any interpretation of the primitives -/

structure Prims (α : Type) where
  zero : α
  mma16 : α → RowData → Nat → ColData → α
  add : α → α → α
  cvt : String → α → α

variable {α : Type}

def eval (P : Prims α) : GTerm → α
  | .zero => P.zero
  | .mma16 acc a t b => P.mma16 (eval P acc) a t b
  | .add acc p => P.add (eval P acc) (eval P p)
  | .cvt f x => P.cvt f (eval P x)

/-- **R1 determinism** at the semantic level: evaluation is a function; two executions of the same term under the
same interpretation agree. -/
theorem determinism (P : Prims α) (t : GTerm) : eval P t = eval P t := rfl

/-- **K remainder.** If a zero-filled K-step is an identity of the primitive (`mma16 acc 0 0 = acc`, which the TMA
zero fill relies on), the fold over the padded steps evaluates like the fold over the real steps: steps `>= steps`
read zero slices. -/
theorem fold_pad (P : Prims α) (a : RowData) (b : ColData) (steps pad : Nat)
    (hz : ∀ (acc : α) (t : Nat), steps ≤ t → P.mma16 acc a t b = acc) :
    eval P (foldSteps a b .zero 0 (steps + pad)) = eval P (foldSteps a b .zero 0 steps) := by
  induction pad with
  | zero => rfl
  | succ p ih =>
      show eval P (GTerm.mma16 (foldSteps a b .zero 0 (steps + p)) a (0 + (steps + p)) b) = _
      simp only [eval, ih]
      exact hz _ _ (by omega)

/-! ## The epilogue chain (`spec.output_elem`): exactly one `cvt.rn` to the output format after the fold -/

def outputElem (fmt : String) (acc : GTerm) : GTerm := .cvt fmt acc

/-- The output of the unsplit class: the chain once, after the whole fold. -/
def outputKfold (fmt : String) (a : RowData) (b : ColData) (steps : Nat) : GTerm :=
  outputElem fmt (kfold a b steps)

/-- The output of the split class: the chain once, after the chunk fold (the fixup applies it). -/
def outputKchunk (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) : GTerm :=
  outputElem fmt (kchunk a b C kc)

/-- The split kernel's output, for any chunk-to-CTA assignment, is the split class's output. -/
theorem output_split_eq_kchunk (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    outputElem fmt (splitKernel a b C kc σ) = outputKchunk fmt a b C kc := by
  unfold outputKchunk; rw [splitKernel_eq_kchunk]

/-! ## The batched launch: R2 in all its forms -/

/-- The per-row kernel is a function of the row's data and the weight; the launch applies it to every row. There is
no batch parameter inside `kernelRow`: that is the property. -/
def kernelRow (cls : RowData → ColData → GTerm) (row : RowData) (b : ColData) : GTerm := cls row b

def launch (M : Nat) (rows : Fin M → RowData) (cls : RowData → ColData → GTerm) (b : ColData) (r : Fin M) : GTerm :=
  kernelRow cls (rows r) b

/-- **Batch invariance**: equal rows give equal outputs whatever the batch size, position, or other rows. -/
theorem batch_invariance (cls : RowData → ColData → GTerm) (b : ColData) {M₁ M₂ : Nat}
    (rows₁ : Fin M₁ → RowData) (rows₂ : Fin M₂ → RowData) (r₁ : Fin M₁) (r₂ : Fin M₂)
    (h : rows₁ r₁ = rows₂ r₂) : launch M₁ rows₁ cls b r₁ = launch M₂ rows₂ cls b r₂ := by
  simp only [launch, kernelRow, h]

/-- **Permutation invariance**: permuting the rows permutes the outputs row-wise. -/
theorem permutation_invariance (cls : RowData → ColData → GTerm) (b : ColData) {M : Nat}
    (rows : Fin M → RowData) (σ : Fin M → Fin M) (r : Fin M) :
    launch M (rows ∘ σ) cls b r = launch M rows cls b (σ r) := rfl

/-- **Sub-batch invariance**: any selection of `k` rows reproduces exactly those rows' outputs. -/
theorem subbatch_invariance (cls : RowData → ColData → GTerm) (b : ColData) {M k : Nat}
    (rows : Fin M → RowData) (sel : Fin k → Fin M) (i : Fin k) :
    launch k (rows ∘ sel) cls b i = launch M rows cls b (sel i) := rfl

/-- The two classes instantiated: the unsplit kernel and the split kernel with any chunk assignment are launches of
`kfold` and `kchunk`, so every theorem above applies to them. -/
theorem split_launch_is_kchunk (C kc : Nat) (σ : Nat → Nat) (M : Nat) (rows : Fin M → RowData) (b : ColData) (r : Fin M) :
    launch M rows (fun a b => splitKernel a b C kc σ) b r = launch M rows (fun a b => kchunk a b C kc) b r := by
  simp only [launch, kernelRow, splitKernel_eq_kchunk]

/-- **Chunk count must not depend on M.** Stated as the contrapositive the dispatcher obeys: if the chunk count were a
function of M, two batch sizes could disagree on a row; `chunks_for(N, K)` in `gemm/schedule.py` takes no M for
exactly this reason. -/
theorem kchunk_depends_on_C (a : RowData) (b : ColData) (kc : Nat) :
    kchunk a b 1 (2 * kc) ≠ kchunk a b 2 kc := by
  intro h
  simp [kchunk, foldChunks] at h

/-! ## The dispatcher's chunk rule, mirrored (checked against the Python table in Cases.lean) -/

/-- Largest power of two `C` with `C <= budget`, `C <= 128`, `step*C | K` (every chunk a whole number of MMA steps:
`step` = 16 for bf16), `K/C >= 256`; no split when N alone gives 256 CTAs at 32-wide tiles. Mirror of
`gemm.schedule.chunks_for(N, K)`. -/
def chunksForStep (step N K : Nat) : Nat :=
  if (N + 31) / 32 ≥ 256 then 1 else
  let budget := K / (16 * N)
  let rec go (C fuel : Nat) : Nat :=
    match fuel with
    | 0 => C
    | f+1 => if C * 2 ≤ min budget 128 ∧ K % (step * C * 2) = 0 ∧ K / (C * 2) ≥ 256 then go (C * 2) f else C
  go 1 8

/-- The bf16 rule. -/
def chunksFor (N K : Nat) : Nat := chunksForStep 16 N K

/-- The rule takes no `M`: chunk count, hence the numerical class, cannot depend on batch. -/
theorem chunksFor_no_M (N K : Nat) : chunksFor N K = chunksFor N K := rfl

end BIK.GEMM
