/-!
# GEMM on tcgen05: the numerical specifications and the batch-invariance guarantees

Mirrors `flashinfer/gemm/batch_invariant/spec.py`. Per output element the dataflow is a sequential fold over K-steps
through an opaque hardware primitive: `mma16` for the bf16 class (one `tcgen05.mma.kind::f16` K-step of 16 bf16
elements, fp32 accumulator), `mma32` for the fp8 classes (one `kind::f8f6f4` K-step of 32 e4m3 / e5m2 elements), and
`mma32mx` / `mma64mx8` for the MXFP8 classes (one `kind::mxf8f6f4.block_scale` K-step: 32 e4m3 elements with one UE8M0
block scale per operand row on the sm_100 / sm_103 instruction, 64 elements with two scales per row on the sm_107 one;
the step's leaf carries the scale-factor slices and the instruction applies them, no software step touches a block
scale), and `mma64bs` / `mma128bs` / `mma64mx` / `mma128mx4` for the 4-bit classes (NVFP4: `kind::mxf4nvf4.block_scale`, 64 e2m1
elements per step with e4m3 block scales per 16 on the sm_100 / sm_103 instruction, 128 on the sm_107 one; MXFP4: `kind::mxf4`
with UE8M0 scales per 32, 64 per step, and the sm_107 instruction at that scale, 128 per step). No commutativity is used
anywhere, so equality is plain `=`.

Two consistency classes per step:
* `kfold`      : one fold over all K-steps from +0 (`gemm.bf16.kfold`; `kfold32` for `gemm.<fp8>.kfold.scaled`;
                 `kfoldMX8` / `kfoldMX8K64` for `gemm.mxfp8.kfold` / `gemm.mxfp8.kfold64`);
* `kchunk C`   : K cut into `C` chunks (a function of the weight shape and the format only), each chunk a fold from
                 +0, the partials folded in chunk order with `add` from +0 (`gemm.bf16.kchunk[C]`; `kchunk32`;
                 `kchunkMX8` / `kchunkMX8K64`).

A third class on the `mma32` step, `kblock128 KB` (`gemm.e4m3.kblock128`, `in_fmt = "e4m3_bs"`, the class behind
`gemm_fp8_nt_groupwise`): K a whole number of 128-element blocks, each block an `mma32` fold from +0, the block partials
folded in block order from +0 with `fma (mul sfa sfb) P acc` -- the fp32 block scales of the row and of the column's
128-column block are leaves like the operand slices; it has no split variant and its chain is the conversion alone.

The fp8 classes' epilogue chain multiplies the fold by `alpha = mul.rn(a_scale, b_scale)` (the two per-tensor
scales of `bmm_fp8`, runtime scalars: the `sym` leaves; their product formed on the device by one correctly rounded
multiply) before the single conversion (`outputElem true`); the bf16 class converts directly (`outputElem false`), and
so do the MXFP8 classes (`mm_mxfp8` takes no per-tensor scale: every scale is a block scale inside the step); the 4-bit
classes multiply by the caller's scalar `alpha` (`mm_fp4`'s 1-element tensor, one `sym`) before the conversion
(`outputElemAlpha`).
Each has a biased chain (`outputElemBias`): the bias of the output column, a leaf, added once to the whole fold before the
single conversion; every identity below lifts to it by congruence.

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
(the per-tensor scales); `cvt fmt` is the epilogue's single correctly rounded conversion to the output format; `bias n` is the bias of
output column `n` (the bf16 bias widened to binary32, which is exact), a leaf like the operand slices, which the biased
chain's `add` adds once to the whole fold. `mma32mx acc a t b` is one *block-scaled* K-step of the sm_100 / sm_103 MXFP8
instruction: its leaf is the A slice, the A scale-factor slice, the B slice and the B scale-factor slice of step `t`
(`(absf (a a t) (sfa a t) (b t b) (sfb t b))` in the printer); the scale factors are addressed by the same (row, step) /
(step, column) as the values, so they are determined by `a`, `t`, `b` and the constructor carries no further argument.
`mma64mx8` is the same leaf folded by the sm_107 instruction (64 elements per step: step `t` covers the elements of
`mma32mx` steps `2t` and `2t + 1`): a different opaque step, hence a different class (`kfoldMX8K64_ne_kfoldMX8`). The 4-bit
steps carry the same leaf: `mma64bs` / `mma128bs` (NVFP4, e4m3 scales per 16 elements, 64 / 128 elements per step) and
`mma64mx` / `mma128mx4` (MXFP4, UE8M0 scales per 32, 64 / 128 per step); four more opaque steps, four more classes.
`fma x y acc` is `fma.rn(x, y, acc)` (the block-scale class's per-block promotion), and `sfaBlk a kb` / `sfbBlk b kb`
are the fp32 block scales of row `a` and of column `b`'s 128-column block at the 128-element K block `kb`: leaves, like
the operand slices (`(sfa_blk m kb)` / `(sfb_blk (n / 128) kb)` in the printer). -/
inductive GTerm where
  | zero
  | mma16 (acc : GTerm) (a : RowData) (t : Nat) (b : ColData)
  | mma32 (acc : GTerm) (a : RowData) (t : Nat) (b : ColData)
  | mma32mx (acc : GTerm) (a : RowData) (t : Nat) (b : ColData)
  | mma64mx8 (acc : GTerm) (a : RowData) (t : Nat) (b : ColData)
  | mma64bs (acc : GTerm) (a : RowData) (t : Nat) (b : ColData)
  | mma128bs (acc : GTerm) (a : RowData) (t : Nat) (b : ColData)
  | mma64mx (acc : GTerm) (a : RowData) (t : Nat) (b : ColData)
  | mma128mx4 (acc : GTerm) (a : RowData) (t : Nat) (b : ColData)
  | add (acc p : GTerm)
  | mul (x y : GTerm)
  | fma (x y acc : GTerm)
  | sym (name : String)
  | cvt (fmt : String) (x : GTerm)
  | bias (n : ColData)
  | sfaBlk (a : RowData) (kb : Nat)
  | sfbBlk (b : ColData) (kb : Nat)
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

/-! ## The block-scale fp8 class `gemm.e4m3.kblock128` (`in_fmt = "e4m3_bs"`, `spec.reduction_blockscale`)

The same `mma32` step, folded in blocks: each 128-element K block is `stepsPerBlock` opaque steps from +0
(`blockPartial32`), and the partials are folded in block order from +0 with `fma (blockScale kb) (blockPartial32 kb) acc`
-- one correctly rounded multiply for the product of the two fp32 block scales (leaves the fold reads: the row's scale and
the column block's), one fused multiply-add per block. The chain is the conversion alone (`outputElem false`). The kernel
computes exactly this (`GEMMConfig.promoteLoop`); the class has no split variant. -/

/-- `BLOCK_K / 32`: the opaque steps of one 128-element block (one fp8 smem stage of the body). -/
def stepsPerBlock : Nat := 4

/-- The partial of block `kb`: its `stepsPerBlock` steps, ascending, from +0. -/
def blockPartial32 (a : RowData) (b : ColData) (kb : Nat) : GTerm :=
  foldStepsG GTerm.mma32 a b .zero (kb * stepsPerBlock) stepsPerBlock

/-- `mul.rn(s_a[m, kb], s_b[n / 128, kb])`: the block's scale, one product. -/
def blockScale (a : RowData) (b : ColData) (kb : Nat) : GTerm := .mul (.sfaBlk a kb) (.sfbBlk b kb)

/-- Fold the first `n` block partials in block order onto `acc`: `acc' = fma(scale_kb, P_kb, acc)`. -/
def foldBlocks (a : RowData) (b : ColData) (acc : GTerm) : Nat → GTerm
  | 0 => acc
  | n+1 => GTerm.fma (blockScale a b n) (blockPartial32 a b n) (foldBlocks a b acc n)

/-- Class `gemm.e4m3.kblock128`: `KB = K / 128` blocks from +0. -/
def kblock128 (a : RowData) (b : ColData) (KB : Nat) : GTerm := foldBlocks a b .zero KB

/-- **A different class from the per-tensor fp8 fold** at every K: one block against its four steps folded without the
scales (the scales are leaves the fold reads; the Python fingerprints separate the two), and not the bf16 fold either. -/
theorem kblock128_ne_kfold32 (a : RowData) (b : ColData) : kblock128 a b 1 ≠ kfold32 a b stepsPerBlock := by
  simp [kblock128, foldBlocks, kfold32, kfoldG, foldStepsG, stepsPerBlock]

theorem kblock128_ne_kfold (a : RowData) (b : ColData) : kblock128 a b 1 ≠ kfold a b stepsPerBlock := by
  simp [kblock128, foldBlocks, kfold, foldSteps, stepsPerBlock]

/-- **The block count is the class**: one block of 2 x 128 elements is not two blocks (the scales enter per block), so a
kernel may not merge or split blocks -- the block size is a constant of the class, a function of nothing. -/
theorem kblock128_depends_on_blocks (a : RowData) (b : ColData) : kblock128 a b 1 ≠ kblock128 a b 2 := by
  simp [kblock128, foldBlocks]

/-! ## The MXFP8 classes: the same dataflow over the block-scaled steps

`gemm.mxfp8.kfold` / `gemm.mxfp8.kchunk[C]` (`gemm/spec.py` with `in_fmt = "mxfp8"`: the sm_100 / sm_103 instruction,
`mma32mx`) and `gemm.mxfp8.kfold64` / `gemm.mxfp8.kchunk64[C]` (`in_fmt = "mxfp8_k64"`: the sm_107 instruction,
`mma64mx8`). Which of the two a device runs is a function of its architecture family only (`schedule.mx_class_for`),
never of the shape: the fingerprint is scoped to the family. -/

def kfoldMX8 (a : RowData) (b : ColData) (steps : Nat) : GTerm := kfoldG GTerm.mma32mx a b steps
def kchunkMX8 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := kchunkG GTerm.mma32mx a b C kc
def splitKernelMX8 (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) : GTerm := splitKernelG GTerm.mma32mx a b C kc σ

theorem splitKernelMX8_eq_kchunkMX8 (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    splitKernelMX8 a b C kc σ = kchunkMX8 a b C kc := splitKernelG_eq_kchunkG _ a b C kc σ

theorem splitKernelMX8_assignment_irrelevant (a : RowData) (b : ColData) (C kc : Nat) (σ τ : Nat → Nat) :
    splitKernelMX8 a b C kc σ = splitKernelMX8 a b C kc τ := by
  rw [splitKernelMX8_eq_kchunkMX8, splitKernelMX8_eq_kchunkMX8]

def kfoldMX8K64 (a : RowData) (b : ColData) (steps : Nat) : GTerm := kfoldG GTerm.mma64mx8 a b steps
def kchunkMX8K64 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := kchunkG GTerm.mma64mx8 a b C kc
def splitKernelMX8K64 (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) : GTerm := splitKernelG GTerm.mma64mx8 a b C kc σ

theorem splitKernelMX8K64_eq_kchunkMX8K64 (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    splitKernelMX8K64 a b C kc σ = kchunkMX8K64 a b C kc := splitKernelG_eq_kchunkG _ a b C kc σ

theorem splitKernelMX8K64_assignment_irrelevant (a : RowData) (b : ColData) (C kc : Nat) (σ τ : Nat → Nat) :
    splitKernelMX8K64 a b C kc σ = splitKernelMX8K64 a b C kc τ := by
  rw [splitKernelMX8K64_eq_kchunkMX8K64, splitKernelMX8K64_eq_kchunkMX8K64]

/-- **Four primitives, four classes.** The block-scaled fold is not the fp8 fold at the same step count (the step reads scale
factors the fp8 step has not got), and the sm_107 fold is not the sm_100 fold even where both cover the same K (one 64-wide
step against two 32-wide ones): the classes are separated as terms, and the Python fingerprints separate them (the
architecture family is part of the MXFP8 fingerprint). -/
theorem kfoldMX8_ne_kfold32 (a : RowData) (b : ColData) : kfoldMX8 a b 1 ≠ kfold32 a b 1 := by
  simp [kfoldMX8, kfold32, kfoldG, foldStepsG]

theorem kfoldMX8_ne_kfold (a : RowData) (b : ColData) : kfoldMX8 a b 1 ≠ kfold a b 2 := by
  simp [kfoldMX8, kfoldG, foldStepsG, kfold, foldSteps]

theorem kfoldMX8K64_ne_kfoldMX8 (a : RowData) (b : ColData) : kfoldMX8K64 a b 1 ≠ kfoldMX8 a b 2 := by
  simp [kfoldMX8K64, kfoldMX8, kfoldG, foldStepsG]

theorem kfoldMX8K64_ne_kfoldMX8_same_steps (a : RowData) (b : ColData) : kfoldMX8K64 a b 1 ≠ kfoldMX8 a b 1 := by
  simp [kfoldMX8K64, kfoldMX8, kfoldG, foldStepsG]

theorem kfoldMX8K64_ne_kfold32 (a : RowData) (b : ColData) : kfoldMX8K64 a b 1 ≠ kfold32 a b 2 := by
  simp [kfoldMX8K64, kfold32, kfoldG, foldStepsG]

/-! ## The 4-bit classes: the same dataflow over the NVFP4 and MXFP4 steps

The caller's scalar `alpha` is their chain's multiplier (`outputElemAlpha` below). Which NVFP4 class a device runs is a
function of the weight shape and the architecture family (`schedule.fp4_class_for`: the sm_107 instruction's class for the
shapes whose chunk rule at that step is unsplit, on the sm_107 parts), which MXFP4 class of the family only
(`schedule.mx_class_for`); either way never of M. -/

/-- `gemm.nvfp4.kfold` / `gemm.nvfp4.kchunk[C]` (`in_fmt = "nvfp4"`: the sm_100 / sm_103 instruction): the fold / chunk fold / split kernel at the `mma64bs` step. -/
def kfoldBS (a : RowData) (b : ColData) (steps : Nat) : GTerm := kfoldG GTerm.mma64bs a b steps
def kchunkBS (a : RowData) (b : ColData) (C kc : Nat) : GTerm := kchunkG GTerm.mma64bs a b C kc
def splitKernelBS (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) : GTerm := splitKernelG GTerm.mma64bs a b C kc σ

theorem splitKernelBS_eq_kchunkBS (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    splitKernelBS a b C kc σ = kchunkBS a b C kc := splitKernelG_eq_kchunkG _ a b C kc σ

theorem splitKernelBS_assignment_irrelevant (a : RowData) (b : ColData) (C kc : Nat) (σ τ : Nat → Nat) :
    splitKernelBS a b C kc σ = splitKernelBS a b C kc τ := by
  rw [splitKernelBS_eq_kchunkBS, splitKernelBS_eq_kchunkBS]

/-- `gemm.nvfp4.kfold128` / `gemm.nvfp4.kchunk128[C]` (`in_fmt = "nvfp4_k128"`: the sm_107 instruction): the fold / chunk fold / split kernel at the `mma128bs` step. -/
def kfoldBS128 (a : RowData) (b : ColData) (steps : Nat) : GTerm := kfoldG GTerm.mma128bs a b steps
def kchunkBS128 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := kchunkG GTerm.mma128bs a b C kc
def splitKernelBS128 (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) : GTerm := splitKernelG GTerm.mma128bs a b C kc σ

theorem splitKernelBS128_eq_kchunkBS128 (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    splitKernelBS128 a b C kc σ = kchunkBS128 a b C kc := splitKernelG_eq_kchunkG _ a b C kc σ

theorem splitKernelBS128_assignment_irrelevant (a : RowData) (b : ColData) (C kc : Nat) (σ τ : Nat → Nat) :
    splitKernelBS128 a b C kc σ = splitKernelBS128 a b C kc τ := by
  rw [splitKernelBS128_eq_kchunkBS128, splitKernelBS128_eq_kchunkBS128]

/-- `gemm.mxfp4.kfold` / `gemm.mxfp4.kchunk[C]` (`in_fmt = "mxfp4"`: the sm_100 / sm_103 instruction): the fold / chunk fold / split kernel at the `mma64mx` step. -/
def kfoldMX4 (a : RowData) (b : ColData) (steps : Nat) : GTerm := kfoldG GTerm.mma64mx a b steps
def kchunkMX4 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := kchunkG GTerm.mma64mx a b C kc
def splitKernelMX4 (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) : GTerm := splitKernelG GTerm.mma64mx a b C kc σ

theorem splitKernelMX4_eq_kchunkMX4 (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    splitKernelMX4 a b C kc σ = kchunkMX4 a b C kc := splitKernelG_eq_kchunkG _ a b C kc σ

theorem splitKernelMX4_assignment_irrelevant (a : RowData) (b : ColData) (C kc : Nat) (σ τ : Nat → Nat) :
    splitKernelMX4 a b C kc σ = splitKernelMX4 a b C kc τ := by
  rw [splitKernelMX4_eq_kchunkMX4, splitKernelMX4_eq_kchunkMX4]

/-- `gemm.mxfp4.kfold128` / `gemm.mxfp4.kchunk128[C]` (`in_fmt = "mxfp4_k128"`: the sm_107 instruction): the fold / chunk fold / split kernel at the `mma128mx4` step. -/
def kfoldMX4K128 (a : RowData) (b : ColData) (steps : Nat) : GTerm := kfoldG GTerm.mma128mx4 a b steps
def kchunkMX4K128 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := kchunkG GTerm.mma128mx4 a b C kc
def splitKernelMX4K128 (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) : GTerm := splitKernelG GTerm.mma128mx4 a b C kc σ

theorem splitKernelMX4K128_eq_kchunkMX4K128 (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    splitKernelMX4K128 a b C kc σ = kchunkMX4K128 a b C kc := splitKernelG_eq_kchunkG _ a b C kc σ

theorem splitKernelMX4K128_assignment_irrelevant (a : RowData) (b : ColData) (C kc : Nat) (σ τ : Nat → Nat) :
    splitKernelMX4K128 a b C kc σ = splitKernelMX4K128 a b C kc τ := by
  rw [splitKernelMX4K128_eq_kchunkMX4K128, splitKernelMX4K128_eq_kchunkMX4K128]

/-- **Four more primitives, four more classes.** The wider sm_107 fold is not the sm_100 fold of its format over the same K (one
128-wide step against two 64-wide ones); the two formats' folds differ at equal step counts (e4m3 scales per 16 against UE8M0
per 32: a different leaf is not needed, the steps are different constructors); and a 4-bit fold is not the MXFP8 fold over the
same K. The Python fingerprints separate all of them (the format and, for NVFP4, the weight shape's class are in the tag). -/
theorem kfoldBS128_ne_kfoldBS (a : RowData) (b : ColData) : kfoldBS128 a b 1 ≠ kfoldBS a b 2 := by
  simp [kfoldBS128, kfoldBS, kfoldG, foldStepsG]

theorem kfoldMX4K128_ne_kfoldMX4 (a : RowData) (b : ColData) : kfoldMX4K128 a b 1 ≠ kfoldMX4 a b 2 := by
  simp [kfoldMX4K128, kfoldMX4, kfoldG, foldStepsG]

theorem kfoldMX4_ne_kfoldBS (a : RowData) (b : ColData) : kfoldMX4 a b 1 ≠ kfoldBS a b 1 := by
  simp [kfoldMX4, kfoldBS, kfoldG, foldStepsG]

theorem kfoldMX4K128_ne_kfoldBS128 (a : RowData) (b : ColData) : kfoldMX4K128 a b 1 ≠ kfoldBS128 a b 1 := by
  simp [kfoldMX4K128, kfoldBS128, kfoldG, foldStepsG]

theorem kfoldBS_ne_kfoldMX8 (a : RowData) (b : ColData) : kfoldBS a b 1 ≠ kfoldMX8 a b 2 := by
  simp [kfoldBS, kfoldMX8, kfoldG, foldStepsG]

theorem kfoldMX4_ne_kfoldMX8K64 (a : RowData) (b : ColData) : kfoldMX4 a b 1 ≠ kfoldMX8K64 a b 1 := by
  simp [kfoldMX4, kfoldMX8K64, kfoldG, foldStepsG]

/-! ## Semantics: any interpretation of the primitives -/

structure Prims (α : Type) where
  zero : α
  mma16 : α → RowData → Nat → ColData → α
  mma32 : α → RowData → Nat → ColData → α
  mma32mx : α → RowData → Nat → ColData → α
  mma64mx8 : α → RowData → Nat → ColData → α
  mma64bs : α → RowData → Nat → ColData → α
  mma128bs : α → RowData → Nat → ColData → α
  mma64mx : α → RowData → Nat → ColData → α
  mma128mx4 : α → RowData → Nat → ColData → α
  add : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  sym : String → α
  cvt : String → α → α
  bias : ColData → α
  sfaBlk : RowData → Nat → α
  sfbBlk : ColData → Nat → α

variable {α : Type}

def eval (P : Prims α) : GTerm → α
  | .zero => P.zero
  | .mma16 acc a t b => P.mma16 (eval P acc) a t b
  | .mma32 acc a t b => P.mma32 (eval P acc) a t b
  | .mma32mx acc a t b => P.mma32mx (eval P acc) a t b
  | .mma64mx8 acc a t b => P.mma64mx8 (eval P acc) a t b
  | .mma64bs acc a t b => P.mma64bs (eval P acc) a t b
  | .mma128bs acc a t b => P.mma128bs (eval P acc) a t b
  | .mma64mx acc a t b => P.mma64mx (eval P acc) a t b
  | .mma128mx4 acc a t b => P.mma128mx4 (eval P acc) a t b
  | .add acc p => P.add (eval P acc) (eval P p)
  | .mul x y => P.mul (eval P x) (eval P y)
  | .fma x y acc => P.fma (eval P x) (eval P y) (eval P acc)
  | .sym s => P.sym s
  | .cvt f x => P.cvt f (eval P x)
  | .bias n => P.bias n
  | .sfaBlk a kb => P.sfaBlk a kb
  | .sfbBlk b kb => P.sfbBlk b kb

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

/-- **K remainder of the MXFP8 classes**: the padded steps read zero value slices and zero-filled scale bytes (TMA zero fill
of the operands and of their scale factors; the UE8M0 byte 0x00 is the finite 2^-127, so every padded product is +0:
`spec.identity_note`), which the hypothesis `hz` states for the step. -/
theorem fold_padMX8 (P : Prims α) (a : RowData) (b : ColData) (steps pad : Nat)
    (hz : ∀ (acc : α) (t : Nat), steps ≤ t → P.mma32mx acc a t b = acc) :
    eval P (kfoldMX8 a b (steps + pad)) = eval P (kfoldMX8 a b steps) :=
  fold_padG P GTerm.mma32mx P.mma32mx (fun _ _ _ _ => rfl) a b steps pad hz

theorem fold_padMX8K64 (P : Prims α) (a : RowData) (b : ColData) (steps pad : Nat)
    (hz : ∀ (acc : α) (t : Nat), steps ≤ t → P.mma64mx8 acc a t b = acc) :
    eval P (kfoldMX8K64 a b (steps + pad)) = eval P (kfoldMX8K64 a b steps) :=
  fold_padG P GTerm.mma64mx8 P.mma64mx8 (fun _ _ _ _ => rfl) a b steps pad hz

/-- **K remainder of the 4-bit classes**: the padded steps read zero values and zero-filled scale bytes (+0 for the e4m3
scales of NVFP4, the finite 2^-127 for the UE8M0 scales of MXFP4: every padded product is +0, `spec.identity_note`). -/
theorem fold_padBS (P : Prims α) (a : RowData) (b : ColData) (steps pad : Nat)
    (hz : ∀ (acc : α) (t : Nat), steps ≤ t → P.mma64bs acc a t b = acc) :
    eval P (kfoldBS a b (steps + pad)) = eval P (kfoldBS a b steps) :=
  fold_padG P GTerm.mma64bs P.mma64bs (fun _ _ _ _ => rfl) a b steps pad hz

theorem fold_padBS128 (P : Prims α) (a : RowData) (b : ColData) (steps pad : Nat)
    (hz : ∀ (acc : α) (t : Nat), steps ≤ t → P.mma128bs acc a t b = acc) :
    eval P (kfoldBS128 a b (steps + pad)) = eval P (kfoldBS128 a b steps) :=
  fold_padG P GTerm.mma128bs P.mma128bs (fun _ _ _ _ => rfl) a b steps pad hz

theorem fold_padMX4 (P : Prims α) (a : RowData) (b : ColData) (steps pad : Nat)
    (hz : ∀ (acc : α) (t : Nat), steps ≤ t → P.mma64mx acc a t b = acc) :
    eval P (kfoldMX4 a b (steps + pad)) = eval P (kfoldMX4 a b steps) :=
  fold_padG P GTerm.mma64mx P.mma64mx (fun _ _ _ _ => rfl) a b steps pad hz

theorem fold_padMX4K128 (P : Prims α) (a : RowData) (b : ColData) (steps pad : Nat)
    (hz : ∀ (acc : α) (t : Nat), steps ≤ t → P.mma128mx4 acc a t b = acc) :
    eval P (kfoldMX4K128 a b (steps + pad)) = eval P (kfoldMX4K128 a b steps) :=
  fold_padG P GTerm.mma128mx4 P.mma128mx4 (fun _ _ _ _ => rfl) a b steps pad hz

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

/-- The block-scale class through the unscaled chain (`gemm_fp8_nt_groupwise`: every scale is a block scale inside the fold,
the conversion alone follows it). -/
def outputKblock (fmt : String) (a : RowData) (b : ColData) (KB : Nat) : GTerm :=
  outputElem false fmt (kblock128 a b KB)

/-- The MXFP8 classes through the unscaled chain (`mm_mxfp8`: the conversion alone after the fold). -/
def outputKfoldMX8 (fmt : String) (a : RowData) (b : ColData) (steps : Nat) : GTerm :=
  outputElem false fmt (kfoldMX8 a b steps)

def outputKchunkMX8 (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) : GTerm :=
  outputElem false fmt (kchunkMX8 a b C kc)

theorem outputMX8_split_eq_kchunkMX8 (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    outputElem false fmt (splitKernelMX8 a b C kc σ) = outputKchunkMX8 fmt a b C kc := by
  unfold outputKchunkMX8; rw [splitKernelMX8_eq_kchunkMX8]

def outputKfoldMX8K64 (fmt : String) (a : RowData) (b : ColData) (steps : Nat) : GTerm :=
  outputElem false fmt (kfoldMX8K64 a b steps)

def outputKchunkMX8K64 (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) : GTerm :=
  outputElem false fmt (kchunkMX8K64 a b C kc)

theorem outputMX8K64_split_eq_kchunkMX8K64 (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    outputElem false fmt (splitKernelMX8K64 a b C kc σ) = outputKchunkMX8K64 fmt a b C kc := by
  unfold outputKchunkMX8K64; rw [splitKernelMX8K64_eq_kchunkMX8K64]

/-! ## The alpha chain of the 4-bit classes (`spec.scale_term` for the formats whose alpha is given)

`mm_fp4`'s `alpha` is one runtime scalar (the caller forms it; how is part of the input, like the block scales): the chain is
`cvt fmt (mul acc (sym "alpha"))`, one `mul.rn` then the conversion, applied once after the whole fold (the epilogue of the
direct stores, the fixup of the workspace split). -/

def outputElemAlpha (fmt : String) (acc : GTerm) : GTerm := .cvt fmt (.mul acc (.sym "alpha"))

def outputKfoldBS (fmt : String) (a : RowData) (b : ColData) (steps : Nat) : GTerm :=
  outputElemAlpha fmt (kfoldBS a b steps)

def outputKchunkBS (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) : GTerm :=
  outputElemAlpha fmt (kchunkBS a b C kc)

theorem outputBS_split_eq_kchunkBS (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    outputElemAlpha fmt (splitKernelBS a b C kc σ) = outputKchunkBS fmt a b C kc := by
  unfold outputKchunkBS; rw [splitKernelBS_eq_kchunkBS]

def outputKfoldBS128 (fmt : String) (a : RowData) (b : ColData) (steps : Nat) : GTerm :=
  outputElemAlpha fmt (kfoldBS128 a b steps)

def outputKchunkBS128 (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) : GTerm :=
  outputElemAlpha fmt (kchunkBS128 a b C kc)

theorem outputBS128_split_eq_kchunkBS128 (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    outputElemAlpha fmt (splitKernelBS128 a b C kc σ) = outputKchunkBS128 fmt a b C kc := by
  unfold outputKchunkBS128; rw [splitKernelBS128_eq_kchunkBS128]

def outputKfoldMX4 (fmt : String) (a : RowData) (b : ColData) (steps : Nat) : GTerm :=
  outputElemAlpha fmt (kfoldMX4 a b steps)

def outputKchunkMX4 (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) : GTerm :=
  outputElemAlpha fmt (kchunkMX4 a b C kc)

theorem outputMX4_split_eq_kchunkMX4 (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    outputElemAlpha fmt (splitKernelMX4 a b C kc σ) = outputKchunkMX4 fmt a b C kc := by
  unfold outputKchunkMX4; rw [splitKernelMX4_eq_kchunkMX4]

def outputKfoldMX4K128 (fmt : String) (a : RowData) (b : ColData) (steps : Nat) : GTerm :=
  outputElemAlpha fmt (kfoldMX4K128 a b steps)

def outputKchunkMX4K128 (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) : GTerm :=
  outputElemAlpha fmt (kchunkMX4K128 a b C kc)

theorem outputMX4K128_split_eq_kchunkMX4K128 (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    outputElemAlpha fmt (splitKernelMX4K128 a b C kc σ) = outputKchunkMX4K128 fmt a b C kc := by
  unfold outputKchunkMX4K128; rw [splitKernelMX4K128_eq_kchunkMX4K128]

/-! ## The biased chain (`spec.output_elem` with a bias column, `spec.BIAS_NOTE`)

The bias of the output column is added once, in fp32, to the fully reduced accumulator, then the single `cvt.rn`:
`cvt fmt (add acc (bias n))`. The bias is a leaf of the term (the bf16 bias widened to binary32, exactly), so a biased
and an unbiased output are different terms (`outputElemBias_ne_outputElem`), and every fold identity above lifts to
the biased output by congruence: the bias step is a per-element postfix of the fold, applied after the whole fold
whatever implementation computed it (the unsplit kernel's epilogue, the in-CTA chunk fold's epilogue, the workspace
split's fixup -- the partials stay bias-free). -/

def outputElemBias (fmt : String) (acc : GTerm) (n : ColData) : GTerm := .cvt fmt (.add acc (.bias n))

/-- The biased output of the unsplit class. -/
def outputKfoldBias (fmt : String) (a : RowData) (b : ColData) (steps : Nat) : GTerm :=
  outputElemBias fmt (kfold a b steps) b

/-- The biased output of the split class. -/
def outputKchunkBias (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) : GTerm :=
  outputElemBias fmt (kchunk a b C kc) b

/-- The workspace split with the bias added in the fixup after the chunk fold is the biased split class, for any
chunk-to-CTA assignment. -/
theorem outputBias_split_eq_kchunk (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) (σ : Nat → Nat) :
    outputElemBias fmt (splitKernel a b C kc σ) b = outputKchunkBias fmt a b C kc := by
  unfold outputKchunkBias; rw [splitKernel_eq_kchunk]

/-- The bias step is a congruence: equal folds give equal biased outputs, so every identity of the folds is an
identity of the biased outputs. -/
theorem outputElemBias_congr (fmt : String) {x y : GTerm} (n : ColData) (h : x = y) :
    outputElemBias fmt x n = outputElemBias fmt y n := by rw [h]

/-- A biased output is never an unbiased one, whatever the fold: the two are different specifications (the Python
fingerprint carries the bias flag for the same reason). -/
theorem outputElemBias_ne_outputElem (fmt : String) (acc : GTerm) (n : ColData) :
    outputElemBias fmt acc n ≠ outputElem false fmt acc := by
  intro h
  have h' : GTerm.add acc (.bias n) = acc := by
    simpa [outputElemBias, outputElem] using h
  have hs := congrArg sizeOf h'
  simp at hs
  omega

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

theorem splitMX8_launch_is_kchunkMX8 (C kc : Nat) (σ : Nat → Nat) (M : Nat) (rows : Fin M → RowData) (b : ColData) (r : Fin M) :
    launch M rows (fun a b => splitKernelMX8 a b C kc σ) b r = launch M rows (fun a b => kchunkMX8 a b C kc) b r := by
  simp only [launch, kernelRow, splitKernelMX8_eq_kchunkMX8]

theorem splitMX8K64_launch_is_kchunkMX8K64 (C kc : Nat) (σ : Nat → Nat) (M : Nat) (rows : Fin M → RowData) (b : ColData) (r : Fin M) :
    launch M rows (fun a b => splitKernelMX8K64 a b C kc σ) b r = launch M rows (fun a b => kchunkMX8K64 a b C kc) b r := by
  simp only [launch, kernelRow, splitKernelMX8K64_eq_kchunkMX8K64]

theorem splitBS_launch_is_kchunkBS (C kc : Nat) (σ : Nat → Nat) (M : Nat) (rows : Fin M → RowData) (b : ColData) (r : Fin M) :
    launch M rows (fun a b => splitKernelBS a b C kc σ) b r = launch M rows (fun a b => kchunkBS a b C kc) b r := by
  simp only [launch, kernelRow, splitKernelBS_eq_kchunkBS]

theorem splitBS128_launch_is_kchunkBS128 (C kc : Nat) (σ : Nat → Nat) (M : Nat) (rows : Fin M → RowData) (b : ColData) (r : Fin M) :
    launch M rows (fun a b => splitKernelBS128 a b C kc σ) b r = launch M rows (fun a b => kchunkBS128 a b C kc) b r := by
  simp only [launch, kernelRow, splitKernelBS128_eq_kchunkBS128]

theorem splitMX4_launch_is_kchunkMX4 (C kc : Nat) (σ : Nat → Nat) (M : Nat) (rows : Fin M → RowData) (b : ColData) (r : Fin M) :
    launch M rows (fun a b => splitKernelMX4 a b C kc σ) b r = launch M rows (fun a b => kchunkMX4 a b C kc) b r := by
  simp only [launch, kernelRow, splitKernelMX4_eq_kchunkMX4]

theorem splitMX4K128_launch_is_kchunkMX4K128 (C kc : Nat) (σ : Nat → Nat) (M : Nat) (rows : Fin M → RowData) (b : ColData) (r : Fin M) :
    launch M rows (fun a b => splitKernelMX4K128 a b C kc σ) b r = launch M rows (fun a b => kchunkMX4K128 a b C kc) b r := by
  simp only [launch, kernelRow, splitKernelMX4K128_eq_kchunkMX4K128]

/-- The biased classes are launches of per-row functions too (no batch parameter inside), so `batch_invariance`,
`permutation_invariance` and `subbatch_invariance` apply to them unchanged; the biased split kernel is a launch of
the biased split class. -/
theorem biased_split_launch_is_kchunk (fmt : String) (C kc : Nat) (σ : Nat → Nat) (M : Nat) (rows : Fin M → RowData)
    (b : ColData) (r : Fin M) :
    launch M rows (fun a b => outputElemBias fmt (splitKernel a b C kc σ) b) b r
      = launch M rows (fun a b => outputKchunkBias fmt a b C kc) b r := by
  simp only [launch, kernelRow, outputBias_split_eq_kchunk]

/-- The block-scale class is a launch of a per-row function too (its scales are the row's and the column block's:
`kblock128` reads `a` and `b` alone), so `batch_invariance`, `permutation_invariance` and `subbatch_invariance` apply to
it unchanged. -/
theorem kblock_launch_is_per_row (KB : Nat) (M : Nat) (rows : Fin M → RowData) (b : ColData) (r : Fin M) :
    launch M rows (fun a b => kblock128 a b KB) b r = kblock128 (rows r) b KB := rfl

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

theorem kchunkMX8_depends_on_C (a : RowData) (b : ColData) (kc : Nat) :
    kchunkMX8 a b 1 (2 * kc) ≠ kchunkMX8 a b 2 kc := by
  intro h
  simp [kchunkMX8, kchunkG, foldChunksG] at h

theorem kchunkMX8K64_depends_on_C (a : RowData) (b : ColData) (kc : Nat) :
    kchunkMX8K64 a b 1 (2 * kc) ≠ kchunkMX8K64 a b 2 kc := by
  intro h
  simp [kchunkMX8K64, kchunkG, foldChunksG] at h

theorem kchunkBS_depends_on_C (a : RowData) (b : ColData) (kc : Nat) :
    kchunkBS a b 1 (2 * kc) ≠ kchunkBS a b 2 kc := by
  intro h
  simp [kchunkBS, kchunkG, foldChunksG] at h

theorem kchunkBS128_depends_on_C (a : RowData) (b : ColData) (kc : Nat) :
    kchunkBS128 a b 1 (2 * kc) ≠ kchunkBS128 a b 2 kc := by
  intro h
  simp [kchunkBS128, kchunkG, foldChunksG] at h

theorem kchunkMX4_depends_on_C (a : RowData) (b : ColData) (kc : Nat) :
    kchunkMX4 a b 1 (2 * kc) ≠ kchunkMX4 a b 2 kc := by
  intro h
  simp [kchunkMX4, kchunkG, foldChunksG] at h

theorem kchunkMX4K128_depends_on_C (a : RowData) (b : ColData) (kc : Nat) :
    kchunkMX4K128 a b 1 (2 * kc) ≠ kchunkMX4K128 a b 2 kc := by
  intro h
  simp [kchunkMX4K128, kchunkG, foldChunksG] at h

/-! ## The dispatcher's chunk rule, mirrored (checked against the Python tables in Cases.lean) -/

/-- Largest power of two `C` with `C <= budget`, `C <= 128`, `step*C | K` (every chunk a whole number of K granules:
`step` = 16 for bf16, 32 for the fp8 formats, 128 for the MXFP8 classes -- a whole number of MMA steps and of
scale-factor atom columns, `spec.chunk_unit`), `K/C >= 256`; no split when N alone gives 256 CTAs at 32-wide tiles.
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

/-- The MXFP8 and MXFP4 rule, the same for both instructions of either format: 128-element granules (four 32-element block
scales, the K extent of one scale-factor atom column). -/
def chunksForMx (N K : Nat) : Nat := chunksForStep 128 N K

/-- The NVFP4 rules: 64-element granules at the sm_100 / sm_103 step (four 16-element block scales: one atom column, one
step), 128 at the sm_107 step (two atom columns, one step). -/
def chunksForFp4 (N K : Nat) : Nat := chunksForStep 64 N K
def chunksForFp4K128 (N K : Nat) : Nat := chunksForStep 128 N K

/-- The NVFP4 class rule of the sm_107 parts (`schedule.fp4_class_for`): the sm_107 instruction's class for every weight shape
whose chunk rule at that step is unsplit, the sm_100 instruction's class for the split shapes; a function of (N, K) only, so a
weight takes one class at every M. `Cases.lean` checks it against the Python rule over the chunk-table shapes. -/
def fp4ClassK128 (N K : Nat) : Bool := chunksForFp4K128 N K == 1

/-- The rules take no `M`: chunk count, hence the numerical class, cannot depend on batch. -/
theorem chunksFor_no_M (N K : Nat) : chunksFor N K = chunksFor N K := rfl
theorem chunksForFp8_no_M (N K : Nat) : chunksForFp8 N K = chunksForFp8 N K := rfl
theorem chunksForMx_no_M (N K : Nat) : chunksForMx N K = chunksForMx N K := rfl
theorem chunksForFp4_no_M (N K : Nat) : chunksForFp4 N K = chunksForFp4 N K := rfl
theorem fp4ClassK128_no_M (N K : Nat) : fp4ClassK128 N K = fp4ClassK128 N K := rfl

end BIK.GEMM
