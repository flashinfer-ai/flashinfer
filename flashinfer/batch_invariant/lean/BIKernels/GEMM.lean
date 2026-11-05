/-!
# GEMM on tcgen05: the numerical specifications and the batch-invariance guarantees

Mirrors `flashinfer/gemm/batch_invariant/spec.py`. Per output element the dataflow is a sequential fold over K-steps
through an opaque hardware primitive: `mma16` for the bf16 class (one `tcgen05.mma.kind::f16` K-step of 16 bf16
elements, fp32 accumulator) and `mma32` for the fp8 classes (one `kind::f8f6f4` K-step of 32 e4m3 / e5m2 elements).
No commutativity is used anywhere, so equality is plain `=`.

Two consistency classes per step:
* `kfold`      : one fold over all K-steps from +0 (`gemm.bf16.kfold`; `kfold32` for `gemm.<fp8>.kfold.scaled`);
* `kchunk C`   : K cut into `C` chunks (a function of the weight shape and the format only), each chunk a fold from
                 +0, the partials folded in chunk order with `add` from +0 (`gemm.bf16.kchunk[C]`; `kchunk32`).

The fp8 classes' epilogue chain multiplies the fold by `alpha = mul.rn(a_scale, b_scale)` (the two per-tensor
scales of `bmm_fp8`, runtime scalars: the `sym` leaves; their product formed on the device by one correctly rounded
multiply) before the single conversion (`outputElem true`); the bf16 class converts directly (`outputElem false`).

What is modelled: the per-element dataflow, the split kernel's index-addressed workspace, and the batched launch
(rows -> per-row kernel). What is *not* modelled, and is covered by the hardware tests: that the tile shape / cluster
/ raster of the kernel leaves the step's per-element sequence untouched (the configuration-agreement test over the
dispatcher's candidates), and that the kernel's first K step, issued without an accumulator input, equals
`step zero` (the two can differ only in the sign of an all-zero result).
-/
namespace BIK.GEMM

/-- A row of A (its data, opaque) and a column of B (opaque), addressed by K-step. -/
structure RowData where
  id : Nat
  deriving DecidableEq, Repr

structure ColData where
  id : Nat
  deriving DecidableEq, Repr

/-- Dataflow terms of the GEMM. `mma16 acc a t b` is one bf16 K-step with A's slice `t` of row `a` and B's slice `t`
of column `b`; `mma32 acc a t b` the fp8 K-step of 32 elements (a different opaque primitive: step `t` of the fp8 fold
covers the elements of bf16 steps `2t` and `2t + 1`); `zero` is the +0 the accumulator starts from; `add` folds chunk
partials; `mul` is the fp8 chain's `mul.rn` (the scales' product and the fold's scaling); `sym` a runtime scalar
(the per-tensor scales); `cvt fmt` is the epilogue's single correctly rounded conversion to the output format. -/
inductive GTerm where
  | zero
  | mma16 (acc : GTerm) (a : RowData) (t : Nat) (b : ColData)
  | mma32 (acc : GTerm) (a : RowData) (t : Nat) (b : ColData)
  | add (acc p : GTerm)
  | mul (x y : GTerm)
  | sym (name : String)
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

/-! ## The classes at any step: the fp8 classes are the same dataflow over the `mma32` primitive

`foldStepsG step` is the fold above at an arbitrary opaque K-step; every statement proved on it holds for both
primitives. The bf16 definitions above are its `mma16` instance (`foldSteps_eq_G`); the fp8 classes are its `mma32`
instance below. -/

/-- A K-step: the opaque primitive of a class (`mma16` for bf16, `mma32` for the fp8 formats). -/
abbrev KStep := GTerm → RowData → Nat → ColData → GTerm

def foldStepsG (step : KStep) (a : RowData) (b : ColData) (acc : GTerm) (t0 : Nat) : Nat → GTerm
  | 0 => acc
  | n+1 => step (foldStepsG step a b acc t0 n) a (t0 + n) b

theorem foldSteps_eq_G (a : RowData) (b : ColData) (acc : GTerm) (t0 n : Nat) :
    foldSteps a b acc t0 n = foldStepsG GTerm.mma16 a b acc t0 n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [foldSteps, foldStepsG, ih]

def kfoldG (step : KStep) (a : RowData) (b : ColData) (steps : Nat) : GTerm :=
  foldStepsG step a b .zero 0 steps

def chunkPartialG (step : KStep) (a : RowData) (b : ColData) (kc : Nat) (c : Nat) : GTerm :=
  foldStepsG step a b .zero (c * kc) kc

def foldChunksG (step : KStep) (a : RowData) (b : ColData) (kc : Nat) (acc : GTerm) : Nat → GTerm
  | 0 => acc
  | n+1 => GTerm.add (foldChunksG step a b kc acc n) (chunkPartialG step a b kc n)

def kchunkG (step : KStep) (a : RowData) (b : ColData) (C kc : Nat) : GTerm :=
  foldChunksG step a b kc .zero C

/-- The split kernel at any step: the workspace slot `c` holds chunk `c`'s partial whatever CTA computed it; the fixup
folds the slots in index order from +0. -/
def splitKernelG (step : KStep) (a : RowData) (b : ColData) (C kc : Nat) (_σ : Nat → Nat) : GTerm :=
  fixup (fun c => chunkPartialG step a b kc c) C

/-- **The split kernel computes the chunked class** at any step, for every chunk-to-CTA assignment. -/
theorem splitKernelG_eq_kchunkG (step : KStep) (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    splitKernelG step a b C kc σ = kchunkG step a b C kc := by
  unfold splitKernelG kchunkG
  induction C with
  | zero => rfl
  | succ n ih => simp only [fixup, foldChunksG, ih]

theorem splitKernelG_assignment_irrelevant (step : KStep) (a : RowData) (b : ColData) (C kc : Nat) (σ τ : Nat → Nat) :
    splitKernelG step a b C kc σ = splitKernelG step a b C kc τ := by
  rw [splitKernelG_eq_kchunkG, splitKernelG_eq_kchunkG]

/-- The fp8 classes `gemm.e4m3.kfold.scaled` / `gemm.e5m2.kfold.scaled` and their split classes (`gemm/spec.py` with
`in_fmt = "e4m3"` / `"e5m2"`): the fold / chunk fold / split kernel at the `mma32` step. -/
def kfold32 (a : RowData) (b : ColData) (steps : Nat) : GTerm := kfoldG GTerm.mma32 a b steps
def kchunk32 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := kchunkG GTerm.mma32 a b C kc
def splitKernel32 (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) : GTerm := splitKernelG GTerm.mma32 a b C kc σ

theorem splitKernel32_eq_kchunk32 (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    splitKernel32 a b C kc σ = kchunk32 a b C kc := splitKernelG_eq_kchunkG _ a b C kc σ

theorem splitKernel32_assignment_irrelevant (a : RowData) (b : ColData) (C kc : Nat) (σ τ : Nat → Nat) :
    splitKernel32 a b C kc σ = splitKernel32 a b C kc τ := by
  rw [splitKernel32_eq_kchunk32, splitKernel32_eq_kchunk32]

/-- **Two primitives, two classes.** The fp8 fold is not the bf16 fold even where both cover the same K (one 32-wide step
against two 16-wide ones): the classes are separated as terms, and the Python fingerprints separate them. -/
theorem kfold32_ne_kfold (a : RowData) (b : ColData) : kfold32 a b 1 ≠ kfold a b 2 := by
  simp [kfold32, kfoldG, foldStepsG, kfold, foldSteps]

theorem kfold32_ne_kfold_same_steps (a : RowData) (b : ColData) : kfold32 a b 1 ≠ kfold a b 1 := by
  simp [kfold32, kfoldG, foldStepsG, kfold, foldSteps]

/-! ## Semantics: any interpretation of the primitives -/

structure Prims (α : Type) where
  zero : α
  mma16 : α → RowData → Nat → ColData → α
  mma32 : α → RowData → Nat → ColData → α
  add : α → α → α
  mul : α → α → α
  sym : String → α
  cvt : String → α → α

variable {α : Type}

def eval (P : Prims α) : GTerm → α
  | .zero => P.zero
  | .mma16 acc a t b => P.mma16 (eval P acc) a t b
  | .mma32 acc a t b => P.mma32 (eval P acc) a t b
  | .add acc p => P.add (eval P acc) (eval P p)
  | .mul x y => P.mul (eval P x) (eval P y)
  | .sym s => P.sym s
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

/-- The same for any step whose evaluation is the primitive `pstep` of `P` (`hstep`), instantiated for the fp8 fold
below. -/
theorem fold_padG (P : Prims α) (step : KStep) (pstep : α → RowData → Nat → ColData → α)
    (hstep : ∀ (acc : GTerm) (a : RowData) (t : Nat) (b : ColData), eval P (step acc a t b) = pstep (eval P acc) a t b)
    (a : RowData) (b : ColData) (steps pad : Nat)
    (hz : ∀ (acc : α) (t : Nat), steps ≤ t → pstep acc a t b = acc) :
    eval P (kfoldG step a b (steps + pad)) = eval P (kfoldG step a b steps) := by
  unfold kfoldG
  induction pad with
  | zero => rfl
  | succ p ih =>
      show eval P (step (foldStepsG step a b .zero 0 (steps + p)) a (0 + (steps + p)) b) = _
      rw [hstep, ih]
      exact hz _ _ (by omega)

/-- **K remainder of the fp8 classes**: the padded 32-wide steps read zero slices (TMA zero fill of both fp8 operands). -/
theorem fold_pad32 (P : Prims α) (a : RowData) (b : ColData) (steps pad : Nat)
    (hz : ∀ (acc : α) (t : Nat), steps ≤ t → P.mma32 acc a t b = acc) :
    eval P (kfold32 a b (steps + pad)) = eval P (kfold32 a b steps) :=
  fold_padG P GTerm.mma32 P.mma32 (fun _ _ _ _ => rfl) a b steps pad hz

/-! ## The epilogue chain (`spec.EpilogueChain`): `mul.rn alpha` for the scaled (fp8) classes, then exactly one `cvt.rn`

One Python object builds the spec term and is executed by the kernel's epilogue block, the in-CTA chunk fold and the
split kernel's fixup; the generated `gemmout*` / `gemm32out*` cases conform this definition to the Python spec node for
node. -/

/-- `alpha = mul.rn(a_scale, b_scale)`: the two per-tensor scales of `bmm_fp8` are runtime scalars and their product
is formed on the device by one correctly rounded multiply. -/
def scaleTerm : GTerm := .mul (.sym "a_scale") (.sym "b_scale")

/-- The chain of a class applied to an accumulator term: `scaled = true` for the fp8 classes. -/
def outputElem (scaled : Bool) (fmt : String) (acc : GTerm) : GTerm :=
  .cvt fmt (if scaled then .mul acc scaleTerm else acc)

/-- The output of the unsplit class: the chain once, after the whole fold. -/
def outputKfold (scaled : Bool) (fmt : String) (a : RowData) (b : ColData) (steps : Nat) : GTerm :=
  outputElem scaled fmt (kfold a b steps)

/-- The output of the split class: the chain once, after the chunk fold (the fixup applies it). -/
def outputKchunk (scaled : Bool) (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) : GTerm :=
  outputElem scaled fmt (kchunk a b C kc)

/-- The split kernel's output, for any chunk-to-CTA assignment, is the split class's output. -/
theorem output_split_eq_kchunk (scaled : Bool) (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    outputElem scaled fmt (splitKernel a b C kc σ) = outputKchunk scaled fmt a b C kc := by
  unfold outputKchunk; rw [splitKernel_eq_kchunk]

/-- The fp8 classes through the scaled chain. -/
def outputKfold32 (fmt : String) (a : RowData) (b : ColData) (steps : Nat) : GTerm :=
  outputElem true fmt (kfold32 a b steps)

def outputKchunk32 (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) : GTerm :=
  outputElem true fmt (kchunk32 a b C kc)

theorem output32_split_eq_kchunk32 (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    outputElem true fmt (splitKernel32 a b C kc σ) = outputKchunk32 fmt a b C kc := by
  unfold outputKchunk32; rw [splitKernel32_eq_kchunk32]

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

/-- The classes instantiated: the unsplit kernel and the split kernel with any chunk assignment are launches of
`kfold` and `kchunk` (likewise at the fp8 step), so every theorem above applies to them. -/
theorem split_launch_is_kchunk (C kc : Nat) (σ : Nat → Nat) (M : Nat) (rows : Fin M → RowData) (b : ColData) (r : Fin M) :
    launch M rows (fun a b => splitKernel a b C kc σ) b r = launch M rows (fun a b => kchunk a b C kc) b r := by
  simp only [launch, kernelRow, splitKernel_eq_kchunk]

theorem split32_launch_is_kchunk32 (C kc : Nat) (σ : Nat → Nat) (M : Nat) (rows : Fin M → RowData) (b : ColData) (r : Fin M) :
    launch M rows (fun a b => splitKernel32 a b C kc σ) b r = launch M rows (fun a b => kchunk32 a b C kc) b r := by
  simp only [launch, kernelRow, splitKernel32_eq_kchunk32]

/-- **Chunk count must not depend on M.** Stated as the contrapositive the dispatcher obeys: if the chunk count were a
function of M, two batch sizes could disagree on a row; `chunks_for(N, K, fmt)` in `gemm/schedule.py` takes no M for
exactly this reason. -/
theorem kchunk_depends_on_C (a : RowData) (b : ColData) (kc : Nat) :
    kchunk a b 1 (2 * kc) ≠ kchunk a b 2 kc := by
  intro h
  simp [kchunk, foldChunks] at h

theorem kchunk32_depends_on_C (a : RowData) (b : ColData) (kc : Nat) :
    kchunk32 a b 1 (2 * kc) ≠ kchunk32 a b 2 kc := by
  intro h
  simp [kchunk32, kchunkG, foldChunksG] at h

/-! ## The dispatcher's chunk rule, mirrored (checked against the Python tables in Cases.lean) -/

/-- Largest power of two `C` with `C <= budget`, `C <= 128`, `step*C | K` (every chunk a whole number of MMA steps:
`step` = 16 for bf16, 32 for the fp8 formats), `K/C >= 256`; no split when N alone gives 256 CTAs at 32-wide tiles.
Mirror of `gemm.schedule.chunks_for(N, K, fmt)`. -/
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

/-- The fp8 rule: 32-wide steps. -/
def chunksForFp8 (N K : Nat) : Nat := chunksForStep 32 N K

/-- The rules take no `M`: chunk count, hence the numerical class, cannot depend on batch. -/
theorem chunksFor_no_M (N K : Nat) : chunksFor N K = chunksFor N K := rfl
theorem chunksForFp8_no_M (N K : Nat) : chunksForFp8 N K = chunksForFp8 N K := rfl

end BIK.GEMM
