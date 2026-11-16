import BIKernels.GEMM

/-!
# The configuration space of the tcgen05 GEMM, made explicit

`gemm/generator.py` enumerates configurations `(tm, tn, cluster, 2-CTA, raster, swizzle, in-CTA chunk fold)` and, for
the block-scaled classes, the in-cluster split factor and the swapped decode tile; the kernel body (`gemm/blocks`)
derives from them the number of smem pipeline stages `S`, the tile assignment and the tile visiting order. This file
states exactly what the proofs claim about that space and where the single hardware hypothesis sits:

* the mainloop consumes K in blocks of `kb` steps through a ring of `S` stages; for every `S` and every `kb` the
  per-element result is the class `kfold` (`kblockLoop_eq_kfold`);
* the accumulator is reset at every tile, so a persistent CTA's result for a tile does not depend on what it
  computed before (`tileResult_reset`);
* tile shape, cluster, 2-CTA mode, raster order and swizzle choose which CTA owns which tile and in which order; they
  never enter the per-element term (`config_invariance`).

Everything is stated for the opaque primitive of the class: `mma16` (bf16) and, through the same definitions at any
step (`tileResultG`), `mma32` (the fp8 classes, whose knob constraints `ConfigOKFp8` lack the 8-wide tile) and
`mma32mx` / `mma64mx8` (the MXFP8 classes, whose knob constraints `ConfigOKMx8` / `ConfigOKMx8K64` are the block-scaled
instruction's) and `mma64bs` / `mma128bs` / `mma64mx` / `mma128mx4` (the NVFP4 and MXFP4 classes, `ConfigOKNvfp4` /
`ConfigOKNvfp4K128` / `ConfigOKMx4` / `ConfigOKMx4K128`); the block-scale fp8 class has its own kernel model, the
promotion loop `promoteLoop` (`ConfigOKFp8BS`). The hardware hypothesis is that every MMA instruction variant the space selects (`tm ∈ {64,128}` 1-CTA,
`{128,256}` 2-CTA, `tn ∈ 32..256` and the decode tiles; for the block-scaled classes also the swapped operand order of
the decode records and the B-reuse pair of the sm_107 class) is the same per-element function of its primitive. No Lean
statement can discharge it; the GPU test `test_config_agreement` checks the dispatcher's candidates bit for bit against
each other.
-/
namespace BIK.GEMM

/-- A kernel configuration: the generator's knob record (`gemm.generator.Knobs.lean()`): tile M/N of the MMA tiler,
cluster shape, 2-CTA MMA, raster order, swizzle, the in-CTA chunk fold `ctaChunks` (the split class only: one CTA
folds every chunk of its tile through a TMEM running sum in chunk order; the constant chunk-to-CTA assignment of
`splitKernel`, `ctaSplit_eq_kchunk` below), the in-cluster split factor `clusterK` (1 for every dense-class record; the
block-scaled body's split classes admit 2 / 4: the chunks of a tile are folded by the CTAs of one cluster, the identity
chunk-to-CTA assignment, `clusterSplitMX8_eq_kchunkMX8` below) and the swapped decode tile `swapAb` (block-scaled body:
the operands exchanged, one token tile for all tokens). -/
structure Config where
  tm : Nat
  tn : Nat
  clusterM : Nat
  clusterN : Nat
  twoCta : Bool
  rasterM : Bool
  swizzle : Nat
  ctaChunks : Bool
  clusterK : Nat
  swapAb : Bool
  deriving DecidableEq, Repr

/-- The in-CTA chunk fold's knob constraint (mirror of the `cta_chunks` clause of `Knobs.well_formed`): a dense record
(no in-cluster split, no swapped tile, no B-reuse tile) whose TMEM running sum fits next to the accumulator stages, tile N
at most `maxTn` (`generator.CTA_CHUNKS_MAX_TN`: 256, or 192 for the sm_100 block-scaled class under the 512-column
budget with the scale-factor columns). -/
def CtaChunksOK (maxTn : Nat) (c : Config) : Bool :=
  !c.ctaChunks || (c.clusterK == 1 && !c.swapAb && c.tm != 512 && c.tn ≤ maxTn)

/-- The kernel body's static knob constraints (mirror of `Knobs.well_formed` and of the decode tiles `knob_product`
prunes, `generator.PRUNED_DENSE`): tile M 64/128 for 1-CTA, 128/256 for 2-CTA MMA; tile N 32..256 step 32, or the
decode tiles 16 (both CTA groups but the 256-row 2-CTA pair) and 8 (the 64-row 1-CTA tile); power-of-two cluster
dims, at most 16 CTAs, even cluster M with 2-CTA MMA; swizzle 1/2/4/8. The generated `gemm_space_ok` checks every
emitted record against it. -/
def ConfigOK (c : Config) : Bool :=
  (if c.twoCta then c.tm == 128 || c.tm == 256 else c.tm == 64 || c.tm == 128) &&
  ((32 ≤ c.tn && c.tn ≤ 256 && c.tn % 32 == 0) || (c.tn == 16 && !(c.twoCta && c.tm == 256)) || (c.tn == 8 && !c.twoCta && c.tm == 64)) &&
  (c.clusterM == 1 || c.clusterM == 2 || c.clusterM == 4 || c.clusterM == 8 || c.clusterM == 16) &&
  (c.clusterN == 1 || c.clusterN == 2 || c.clusterN == 4 || c.clusterN == 8 || c.clusterN == 16) &&
  (c.clusterM * c.clusterN ≤ 16) && (!c.twoCta || c.clusterM % 2 == 0) &&
  (c.swizzle == 1 || c.swizzle == 2 || c.swizzle == 4 || c.swizzle == 8) && (c.clusterK == 1) && (!c.swapAb) &&
  CtaChunksOK 256 c

/-- The fp8 classes' knob constraints (mirror of `Knobs.well_formed("e4m3")`): `kind::f8f6f4` has no 8-wide tile. The
generated `gemm_fp8_space_ok` checks every emitted fp8 record against it. -/
def ConfigOKFp8 (c : Config) : Bool := ConfigOK c && c.tn != 8

/-- The block-scaled classes' knob constraints (mirror of `Knobs.well_formed` for the MXFP8 formats): power-of-two cluster
dims of at most 4 (a scale-factor tile multicasts to at most 4 CTAs), at most 16 CTAs, even cluster M with 2-CTA MMA,
swizzle 1/2/4/8; MMA M 128 (1-CTA) / 256 (2-CTA) -- the scale-factor atom spans 128 rows -- or, where `breuse`, the
512-row B-reuse tiler on the 2-CTA instruction (two instructions per K step on the tile's halves sharing one B fragment,
dense records only); tile N 64 / 128 / 192 / 256, or for the swapped decode record (`swapAb`: no N cluster, no in-cluster
split) the narrow token tiles 8 / 16 / 32 on the 1-CTA instruction or, where `wideSwap` (the UE8M0 formats), the 64-wide one
on either; the in-cluster split `clusterK` 2 / 4 on a 1-CTA (1, 1) cluster with tile N at most 128 / 64 (the mailbox); the
in-CTA chunk fold under `maxTn`. The generated `gemm_*_space_ok` theorems check every emitted record against it. -/
def ConfigOKBlockScaled (breuse wideSwap : Bool) (maxTn : Nat) (c : Config) : Bool :=
  (c.clusterM == 1 || c.clusterM == 2 || c.clusterM == 4) && (c.clusterN == 1 || c.clusterN == 2 || c.clusterN == 4) &&
  (c.clusterM * c.clusterN ≤ 16) && (!c.twoCta || c.clusterM % 2 == 0) &&
  (c.swizzle == 1 || c.swizzle == 2 || c.swizzle == 4 || c.swizzle == 8) &&
  ((if c.twoCta then c.tm == 256 else c.tm == 128) || (breuse && c.tm == 512 && c.twoCta && !c.swapAb && c.clusterK == 1)) &&
  (c.clusterK == 1 || c.clusterK == 2 || c.clusterK == 4) &&
  (if c.swapAb then
      c.clusterN == 1 && c.clusterK == 1 && (((c.tn == 8 || c.tn == 16 || c.tn == 32) && !c.twoCta) || (wideSwap && c.tn == 64))
    else (c.tn == 64 || c.tn == 128 || c.tn == 192 || c.tn == 256)) &&
  (c.clusterK == 1 || (c.clusterM == 1 && c.clusterN == 1 && !c.twoCta && c.tn ≤ (if c.clusterK == 4 then 64 else 128))) &&
  CtaChunksOK maxTn c

/-- `gemm.mxfp8.*` (the sm_100 / sm_103 instruction): no B-reuse, the in-CTA chunk fold up to tile N 192. -/
def ConfigOKMx8 (c : Config) : Bool := ConfigOKBlockScaled false true 192 c

/-- `gemm.mxfp8.*64` (the sm_107 instruction): the B-reuse records, the in-CTA chunk fold at every tile N. -/
def ConfigOKMx8K64 (c : Config) : Bool := ConfigOKBlockScaled true true 256 c

/-- The NVFP4 classes (e4m3 scales per 16: the narrow token tiles only) and the MXFP4 classes (UE8M0 scales per 32: the 64-wide
token tile too), each at its instruction's TMEM budget and, on the sm_107 instruction, with the B-reuse records. -/
def ConfigOKNvfp4 (c : Config) : Bool := ConfigOKBlockScaled false false 192 c
def ConfigOKNvfp4K128 (c : Config) : Bool := ConfigOKBlockScaled true false 256 c
def ConfigOKMx4 (c : Config) : Bool := ConfigOKBlockScaled false true 192 c
def ConfigOKMx4K128 (c : Config) : Bool := ConfigOKBlockScaled true true 256 c

/-- The block-scale fp8 class's knob constraints (mirror of `Knobs.well_formed("e4m3_bs")`): a 128-row CTA tile (MMA M 128
1-CTA / 256 2-CTA: the promotion gives each epilogue thread one output row, hence one A scale), tile N 16 / 32 / 64 / 128
(it divides the 128-column weight scale block: one B scale per tile and block; the register running sum bounds it), the
16-wide tile on the 1-CTA instruction only, the dense classes' cluster / raster / swizzle constraints, and no in-cluster
split, swapped tile or in-CTA chunk fold (the class has no split variant). The generated `gemm_fp8bs_space_ok` checks every
emitted record against it. -/
def ConfigOKFp8BS (c : Config) : Bool :=
  (if c.twoCta then c.tm == 256 else c.tm == 128) &&
  (c.tn == 16 || c.tn == 32 || c.tn == 64 || c.tn == 128 || c.tn == 256) && (32 ≤ c.tn || !c.twoCta) &&
  (c.clusterM == 1 || c.clusterM == 2 || c.clusterM == 4 || c.clusterM == 8 || c.clusterM == 16) &&
  (c.clusterN == 1 || c.clusterN == 2 || c.clusterN == 4 || c.clusterN == 8 || c.clusterN == 16) &&
  (c.clusterM * c.clusterN ≤ 16) && (!c.twoCta || c.clusterM % 2 == 0) &&
  (c.swizzle == 1 || c.swizzle == 2 || c.swizzle == 4 || c.swizzle == 8) && (c.clusterK == 1) && (!c.swapAb) && (!c.ctaChunks)

/-- The in-CTA chunk fold (`ctaChunks`, `gemm/blocks/epilogue.py`) is the split kernel with the CONSTANT assignment
of every chunk to one CTA: that CTA computes each chunk's partial from +0 in turn (the mainloop's fold per chunk) and
folds the partials in chunk order from +0 with `add` (the TMEM running sum) -- the `fixup` of `splitKernel`, so the
record computes the split class `kchunk C` exactly as the workspace split does (R3 between the two implementations
of one `C`). -/
def ctaSplit (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernel a b C kc (fun _ => 0)

theorem ctaSplit_eq_kchunk (a : RowData) (b : ColData) (C kc : Nat) : ctaSplit a b C kc = kchunk a b C kc :=
  splitKernel_eq_kchunk a b C kc _

/-- The same for the fp8 classes (the in-CTA fold applies the chain after its last chunk exactly as the fixup does). -/
def ctaSplit32 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernel32 a b C kc (fun _ => 0)

theorem ctaSplit32_eq_kchunk32 (a : RowData) (b : ColData) (C kc : Nat) : ctaSplit32 a b C kc = kchunk32 a b C kc :=
  splitKernel32_eq_kchunk32 a b C kc _

/-- The MXFP8 classes have three implementations of one `C`: the workspace split (any chunk-to-CTA assignment), the in-CTA
chunk fold (the constant assignment) and the in-cluster split (`clusterK = C`, `gemm/blocks/blockscaled.py`: the
identity assignment -- CTA `c` of the cluster folds chunk `c` from +0, the owner of an epilogue subtile folds the `C`
partials in chunk order from +0 with `add` through distributed shared memory, then the chain). All three are the term
`kchunk` (`splitKernelMX8_assignment_irrelevant`), so they are members of one class. -/
def ctaSplitMX8 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernelMX8 a b C kc (fun _ => 0)
def clusterSplitMX8 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernelMX8 a b C kc (fun c => c)

theorem ctaSplitMX8_eq_kchunkMX8 (a : RowData) (b : ColData) (C kc : Nat) : ctaSplitMX8 a b C kc = kchunkMX8 a b C kc :=
  splitKernelMX8_eq_kchunkMX8 a b C kc _

theorem clusterSplitMX8_eq_kchunkMX8 (a : RowData) (b : ColData) (C kc : Nat) : clusterSplitMX8 a b C kc = kchunkMX8 a b C kc :=
  splitKernelMX8_eq_kchunkMX8 a b C kc _

def ctaSplitMX8K64 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernelMX8K64 a b C kc (fun _ => 0)
def clusterSplitMX8K64 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernelMX8K64 a b C kc (fun c => c)

theorem ctaSplitMX8K64_eq_kchunkMX8K64 (a : RowData) (b : ColData) (C kc : Nat) : ctaSplitMX8K64 a b C kc = kchunkMX8K64 a b C kc :=
  splitKernelMX8K64_eq_kchunkMX8K64 a b C kc _

theorem clusterSplitMX8K64_eq_kchunkMX8K64 (a : RowData) (b : ColData) (C kc : Nat) :
    clusterSplitMX8K64 a b C kc = kchunkMX8K64 a b C kc :=
  splitKernelMX8K64_eq_kchunkMX8K64 a b C kc _

/-- The 4-bit classes' three implementations of one `C` are one term too (the same bodies as the MXFP8 classes'). -/
def ctaSplitBS (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernelBS a b C kc (fun _ => 0)
def clusterSplitBS (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernelBS a b C kc (fun c => c)

theorem ctaSplitBS_eq_kchunkBS (a : RowData) (b : ColData) (C kc : Nat) : ctaSplitBS a b C kc = kchunkBS a b C kc :=
  splitKernelBS_eq_kchunkBS a b C kc _

theorem clusterSplitBS_eq_kchunkBS (a : RowData) (b : ColData) (C kc : Nat) : clusterSplitBS a b C kc = kchunkBS a b C kc :=
  splitKernelBS_eq_kchunkBS a b C kc _

def ctaSplitBS128 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernelBS128 a b C kc (fun _ => 0)
def clusterSplitBS128 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernelBS128 a b C kc (fun c => c)

theorem ctaSplitBS128_eq_kchunkBS128 (a : RowData) (b : ColData) (C kc : Nat) : ctaSplitBS128 a b C kc = kchunkBS128 a b C kc :=
  splitKernelBS128_eq_kchunkBS128 a b C kc _

theorem clusterSplitBS128_eq_kchunkBS128 (a : RowData) (b : ColData) (C kc : Nat) : clusterSplitBS128 a b C kc = kchunkBS128 a b C kc :=
  splitKernelBS128_eq_kchunkBS128 a b C kc _

def ctaSplitMX4 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernelMX4 a b C kc (fun _ => 0)
def clusterSplitMX4 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernelMX4 a b C kc (fun c => c)

theorem ctaSplitMX4_eq_kchunkMX4 (a : RowData) (b : ColData) (C kc : Nat) : ctaSplitMX4 a b C kc = kchunkMX4 a b C kc :=
  splitKernelMX4_eq_kchunkMX4 a b C kc _

theorem clusterSplitMX4_eq_kchunkMX4 (a : RowData) (b : ColData) (C kc : Nat) : clusterSplitMX4 a b C kc = kchunkMX4 a b C kc :=
  splitKernelMX4_eq_kchunkMX4 a b C kc _

def ctaSplitMX4K128 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernelMX4K128 a b C kc (fun _ => 0)
def clusterSplitMX4K128 (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernelMX4K128 a b C kc (fun c => c)

theorem ctaSplitMX4K128_eq_kchunkMX4K128 (a : RowData) (b : ColData) (C kc : Nat) : ctaSplitMX4K128 a b C kc = kchunkMX4K128 a b C kc :=
  splitKernelMX4K128_eq_kchunkMX4K128 a b C kc _

theorem clusterSplitMX4K128_eq_kchunkMX4K128 (a : RowData) (b : ColData) (C kc : Nat) : clusterSplitMX4K128 a b C kc = kchunkMX4K128 a b C kc :=
  splitKernelMX4K128_eq_kchunkMX4K128 a b C kc _

/-- `foldSteps` splits anywhere: `m + n` steps from `t0` are `m` steps, then `n` from `t0 + m`. -/
theorem foldSteps_add (a : RowData) (b : ColData) (acc : GTerm) (t0 m n : Nat) :
    foldSteps a b acc t0 (m + n) = foldSteps a b (foldSteps a b acc t0 m) (t0 + m) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    show foldSteps a b acc t0 (m + n + 1) = _
    simp [foldSteps, ih, Nat.add_assoc]

/-- The mainloop: `nblk` K-blocks of `kb` steps; block `blk` goes through smem stage `blk % S`. The stage is a buffer
index only; what is loaded into it is block `blk`'s slice.

This is the model of the kernel's blocked lowering (`canon.BlockedFold`, instantiated by `MainloopBlock.lowering()`
with `kb = 4` and executed by `gemm/blocks/mainloop.py`): blocks ascending, steps ascending inside a block, the
accumulator reset carried by the first step. The generated `gemmloop*` cases (`Cases.lean`) reproduce the Python
trace of that lowering node for node at sampled block counts, and `gemm_mainloop_eq_kfold` instantiates the theorem
below on the emitted inner radix. -/
def kblockLoop (a : RowData) (b : ColData) (kb S : Nat) (acc : GTerm) : Nat → GTerm
  | 0 => acc
  | blk+1 =>
    let _stage := blk % S
    foldSteps a b (kblockLoop a b kb S acc blk) (blk * kb) kb

theorem kblockLoop_eq_foldSteps (a : RowData) (b : ColData) (kb S : Nat) (acc : GTerm) (nblk : Nat) :
    kblockLoop a b kb S acc nblk = foldSteps a b acc 0 (nblk * kb) := by
  induction nblk with
  | zero => rw [Nat.zero_mul]; rfl
  | succ n ih =>
    simp only [kblockLoop, ih]
    rw [Nat.add_mul, Nat.one_mul, foldSteps_add, Nat.zero_add]

/-- **k-block pipeline = the class**, for every stage count and block size. -/
theorem kblockLoop_eq_kfold (a : RowData) (b : ColData) (kb S nblk : Nat) :
    kblockLoop a b kb S .zero nblk = kfold a b (nblk * kb) := by
  unfold kfold; exact kblockLoop_eq_foldSteps a b kb S .zero nblk

/-- `foldStepsG` splits anywhere, at any step. -/
theorem foldStepsG_add (step : KStep) (a : RowData) (b : ColData) (acc : GTerm) (t0 m n : Nat) :
    foldStepsG step a b acc t0 (m + n) = foldStepsG step a b (foldStepsG step a b acc t0 m) (t0 + m) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    show foldStepsG step a b acc t0 (m + n + 1) = _
    simp [foldStepsG, ih, Nat.add_assoc]

/-- The same loop model at any step: the model of the mainloop executing `MainloopBlock(in_fmt).lowering()` (the same
`canon.BlockedFold`) with the format's step; the fp8 mainloop issues `kind::f8f6f4` steps of 32 elements, four per
128-byte stage like the bf16 one. -/
def kblockLoopG (step : KStep) (a : RowData) (b : ColData) (kb S : Nat) (acc : GTerm) : Nat → GTerm
  | 0 => acc
  | blk+1 =>
    let _stage := blk % S
    foldStepsG step a b (kblockLoopG step a b kb S acc blk) (blk * kb) kb

theorem kblockLoopG_eq_foldStepsG (step : KStep) (a : RowData) (b : ColData) (kb S : Nat) (acc : GTerm) (nblk : Nat) :
    kblockLoopG step a b kb S acc nblk = foldStepsG step a b acc 0 (nblk * kb) := by
  induction nblk with
  | zero => rw [Nat.zero_mul]; rfl
  | succ n ih =>
    simp only [kblockLoopG, ih]
    rw [Nat.add_mul, Nat.one_mul, foldStepsG_add, Nat.zero_add]

theorem kblockLoopG_eq_kfoldG (step : KStep) (a : RowData) (b : ColData) (kb S nblk : Nat) :
    kblockLoopG step a b kb S .zero nblk = kfoldG step a b (nblk * kb) := by
  unfold kfoldG; exact kblockLoopG_eq_foldStepsG step a b kb S .zero nblk

/-- The fp8 mainloop computes the fp8 class, for every stage count and block size; the generated `gemm32loop*` cases
conform it to the Python trace node for node. -/
def kblockLoop32 (a : RowData) (b : ColData) (kb S : Nat) (acc : GTerm) (nblk : Nat) : GTerm :=
  kblockLoopG GTerm.mma32 a b kb S acc nblk

theorem kblockLoop32_eq_kfold32 (a : RowData) (b : ColData) (kb S nblk : Nat) :
    kblockLoop32 a b kb S .zero nblk = kfold32 a b (nblk * kb) :=
  kblockLoopG_eq_kfoldG GTerm.mma32 a b kb S nblk

/-- The block-scaled mainloops (`gemm/blocks/blockscaled.py`: a stage holds a k-tile of both operands and of their scale
factors, which are copied to TMEM before the stage's instructions) compute the MXFP8 classes, for every stage count and
block size; the generated `gemmmx8loop*` / `gemmmx8k64loop*` cases conform them to the Python trace node for node (the
sm_107 instruction's radix is 2 per 128-byte stage, the sm_100 one's 4). -/
def kblockLoopMX8 (a : RowData) (b : ColData) (kb S : Nat) (acc : GTerm) (nblk : Nat) : GTerm :=
  kblockLoopG GTerm.mma32mx a b kb S acc nblk

theorem kblockLoopMX8_eq_kfoldMX8 (a : RowData) (b : ColData) (kb S nblk : Nat) :
    kblockLoopMX8 a b kb S .zero nblk = kfoldMX8 a b (nblk * kb) :=
  kblockLoopG_eq_kfoldG GTerm.mma32mx a b kb S nblk

def kblockLoopMX8K64 (a : RowData) (b : ColData) (kb S : Nat) (acc : GTerm) (nblk : Nat) : GTerm :=
  kblockLoopG GTerm.mma64mx8 a b kb S acc nblk

theorem kblockLoopMX8K64_eq_kfoldMX8K64 (a : RowData) (b : ColData) (kb S nblk : Nat) :
    kblockLoopMX8K64 a b kb S .zero nblk = kfoldMX8K64 a b (nblk * kb) :=
  kblockLoopG_eq_kfoldG GTerm.mma64mx8 a b kb S nblk

/-- The block-scaled mainloops at the 4-bit steps (radix 4 per 128-byte stage at the sm_100 / sm_103 instructions, 2 at the
sm_107 ones). -/
def kblockLoopBS (a : RowData) (b : ColData) (kb S : Nat) (acc : GTerm) (nblk : Nat) : GTerm :=
  kblockLoopG GTerm.mma64bs a b kb S acc nblk

theorem kblockLoopBS_eq_kfoldBS (a : RowData) (b : ColData) (kb S nblk : Nat) :
    kblockLoopBS a b kb S .zero nblk = kfoldBS a b (nblk * kb) :=
  kblockLoopG_eq_kfoldG GTerm.mma64bs a b kb S nblk

def kblockLoopBS128 (a : RowData) (b : ColData) (kb S : Nat) (acc : GTerm) (nblk : Nat) : GTerm :=
  kblockLoopG GTerm.mma128bs a b kb S acc nblk

theorem kblockLoopBS128_eq_kfoldBS128 (a : RowData) (b : ColData) (kb S nblk : Nat) :
    kblockLoopBS128 a b kb S .zero nblk = kfoldBS128 a b (nblk * kb) :=
  kblockLoopG_eq_kfoldG GTerm.mma128bs a b kb S nblk

def kblockLoopMX4 (a : RowData) (b : ColData) (kb S : Nat) (acc : GTerm) (nblk : Nat) : GTerm :=
  kblockLoopG GTerm.mma64mx a b kb S acc nblk

theorem kblockLoopMX4_eq_kfoldMX4 (a : RowData) (b : ColData) (kb S nblk : Nat) :
    kblockLoopMX4 a b kb S .zero nblk = kfoldMX4 a b (nblk * kb) :=
  kblockLoopG_eq_kfoldG GTerm.mma64mx a b kb S nblk

def kblockLoopMX4K128 (a : RowData) (b : ColData) (kb S : Nat) (acc : GTerm) (nblk : Nat) : GTerm :=
  kblockLoopG GTerm.mma128mx4 a b kb S acc nblk

theorem kblockLoopMX4K128_eq_kfoldMX4K128 (a : RowData) (b : ColData) (kb S nblk : Nat) :
    kblockLoopMX4K128 a b kb S .zero nblk = kfoldMX4K128 a b (nblk * kb) :=
  kblockLoopG_eq_kfoldG GTerm.mma128mx4 a b kb S nblk

/-- One output tile under `cfg`: the owning CTA (a function of the config) runs the k-block loop from a reset
accumulator. `prev` is whatever that CTA's accumulator held from the previous tile of its persistent schedule. -/
def tileResult (cfg : Config) (S : Nat) (_prev : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) : GTerm :=
  let _owner := (cfg.tm, cfg.tn, cfg.clusterM, cfg.clusterN, cfg.twoCta, cfg.rasterM, cfg.swizzle, cfg.clusterK, cfg.swapAb)
  kblockLoop a b kb S .zero nblk

/-- **Accumulator reset**: the persistent schedule leaves no trace in a tile's result. -/
theorem tileResult_reset (cfg : Config) (S : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResult cfg S p a b kb nblk = tileResult cfg S q a b kb nblk := rfl

/-- Every tile result is the class term. -/
theorem tileResult_eq_kfold (cfg : Config) (S : Nat) (p : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResult cfg S p a b kb nblk = kfold a b (nblk * kb) := by
  unfold tileResult; exact kblockLoop_eq_kfold a b kb S nblk

/-- **Config invariance (R3) of the tiled GEMM.** Any two configurations, any two stage counts, any carried
accumulators: the same per-element term, hence (by `eval`) the same bits under any interpretation of `mma16` -- which
is where the hardware hypothesis enters: the instruction variants of both configurations must have the same
interpretation. -/
theorem config_invariance (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResult c₁ S₁ p a b kb nblk = tileResult c₂ S₂ q a b kb nblk := by
  rw [tileResult_eq_kfold, tileResult_eq_kfold]

/-- The invariance over any enumerated space: the generated `gemm_space_invariant` instantiates this on the records
the generator emits. -/
theorem space_invariant (space : List Config) :
    ∀ c₁ ∈ space, ∀ c₂ ∈ space, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat),
      tileResult c₁ S₁ p a b kb nblk = tileResult c₂ S₂ q a b kb nblk :=
  fun c₁ _ c₂ _ S₁ S₂ p q a b kb nblk => config_invariance c₁ c₂ S₁ S₂ p q a b kb nblk

/-- Through the chain: the *output* element of a tile is the unsplit class's output for either kind of chain, so no
configuration, stage count or carried accumulator reaches the output. -/
theorem tileOutput_eq_outputKfold (scaled : Bool) (fmt : String) (cfg : Config) (S : Nat) (p : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElem scaled fmt (tileResult cfg S p a b kb nblk) = outputKfold scaled fmt a b (nblk * kb) := by
  unfold outputKfold; rw [tileResult_eq_kfold]

theorem output_config_invariance (scaled : Bool) (fmt : String) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElem scaled fmt (tileResult c₁ S₁ p a b kb nblk) = outputElem scaled fmt (tileResult c₂ S₂ q a b kb nblk) := by
  rw [config_invariance c₁ c₂ S₁ S₂ p q a b kb nblk]

/-! ## The fp8 tile: the same statements at the `mma32` step

`tileResultG step` is the tile model at any opaque step (the `_owner` erases the configuration exactly as `tileResult`
does); `tileResult32` instantiates it for the fp8 classes, whose A2 surface -- the `kind::f8f6f4` instruction variants
of the fp8 space -- is checked bit for bit by the fp8 configuration-agreement tests. -/

def tileResultG (step : KStep) (cfg : Config) (S : Nat) (_prev : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) : GTerm :=
  let _owner := (cfg.tm, cfg.tn, cfg.clusterM, cfg.clusterN, cfg.twoCta, cfg.rasterM, cfg.swizzle, cfg.clusterK, cfg.swapAb)
  kblockLoopG step a b kb S .zero nblk

theorem tileResultG_eq_kfoldG (step : KStep) (cfg : Config) (S : Nat) (p : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResultG step cfg S p a b kb nblk = kfoldG step a b (nblk * kb) := by
  unfold tileResultG; exact kblockLoopG_eq_kfoldG step a b kb S nblk

theorem config_invarianceG (step : KStep) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResultG step c₁ S₁ p a b kb nblk = tileResultG step c₂ S₂ q a b kb nblk := by
  rw [tileResultG_eq_kfoldG, tileResultG_eq_kfoldG]

def tileResult32 (cfg : Config) (S : Nat) (prev : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) : GTerm :=
  tileResultG GTerm.mma32 cfg S prev a b kb nblk

theorem tileResult32_eq_kfold32 (cfg : Config) (S : Nat) (p : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResult32 cfg S p a b kb nblk = kfold32 a b (nblk * kb) := tileResultG_eq_kfoldG GTerm.mma32 cfg S p a b kb nblk

/-- **Config invariance (R3) of the fp8 classes**, and its instantiation over any enumerated fp8 space (the generated
`gemm_fp8_space_invariant`). -/
theorem config_invariance32 (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResult32 c₁ S₁ p a b kb nblk = tileResult32 c₂ S₂ q a b kb nblk := config_invarianceG GTerm.mma32 c₁ c₂ S₁ S₂ p q a b kb nblk

theorem space_invariant32 (space : List Config) :
    ∀ c₁ ∈ space, ∀ c₂ ∈ space, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat),
      tileResult32 c₁ S₁ p a b kb nblk = tileResult32 c₂ S₂ q a b kb nblk :=
  fun c₁ _ c₂ _ S₁ S₂ p q a b kb nblk => config_invariance32 c₁ c₂ S₁ S₂ p q a b kb nblk

/-- Through the scaled chain: the output element of an fp8 tile is the fp8 class's output. -/
theorem tileOutput32_eq_outputKfold32 (fmt : String) (cfg : Config) (S : Nat) (p : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElem true fmt (tileResult32 cfg S p a b kb nblk) = outputKfold32 fmt a b (nblk * kb) := by
  unfold outputKfold32; rw [tileResult32_eq_kfold32]

theorem output32_config_invariance (fmt : String) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElem true fmt (tileResult32 c₁ S₁ p a b kb nblk) = outputElem true fmt (tileResult32 c₂ S₂ q a b kb nblk) := by
  rw [config_invariance32 c₁ c₂ S₁ S₂ p q a b kb nblk]

theorem config_invariance32_bits {α : Type} (P : Prims α) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    eval P (tileResult32 c₁ S₁ p a b kb nblk) = eval P (tileResult32 c₂ S₂ q a b kb nblk) := by
  rw [config_invariance32 c₁ c₂ S₁ S₂ p q a b kb nblk]

/-- The in-CTA chunk fold with the bias added once after the last chunk's fold (its epilogue's chain) is the biased
split class, as the workspace split with the bias in its fixup is (`outputBias_split_eq_kchunk`). -/
theorem outputBias_ctaSplit_eq_kchunk (fmt : String) (a : RowData) (b : ColData) (C kc : Nat) :
    outputElemBias fmt (ctaSplit a b C kc) b = outputKchunkBias fmt a b C kc := by
  unfold outputKchunkBias; rw [ctaSplit_eq_kchunk]

/-- Through the biased chain: the biased output of a tile is the biased unsplit class's output, so no configuration,
stage count or carried accumulator reaches it. -/
theorem tileOutputBias_eq_outputKfoldBias (fmt : String) (cfg : Config) (S : Nat) (p : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElemBias fmt (tileResult cfg S p a b kb nblk) b = outputKfoldBias fmt a b (nblk * kb) := by
  unfold outputKfoldBias; rw [tileResult_eq_kfold]

theorem outputBias_config_invariance (fmt : String) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElemBias fmt (tileResult c₁ S₁ p a b kb nblk) b = outputElemBias fmt (tileResult c₂ S₂ q a b kb nblk) b := by
  rw [config_invariance c₁ c₂ S₁ S₂ p q a b kb nblk]

/-- Bits, not terms: under any primitives `P`, both configurations evaluate to the same value. -/
theorem config_invariance_bits {α : Type} (P : Prims α) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    eval P (tileResult c₁ S₁ p a b kb nblk) = eval P (tileResult c₂ S₂ q a b kb nblk) := by
  rw [config_invariance c₁ c₂ S₁ S₂ p q a b kb nblk]

/-! ## The MXFP8 tiles: the same statements at the `mma32mx` and `mma64mx8` steps

The `_owner` of `tileResultG` now also erases `clusterK` and `swapAb`: the swapped decode record computes the transposed
product with the operands exchanged (the block-scaled step is symmetric in its two operands, a hardware hypothesis the
configuration-agreement test checks bit for bit together with the instruction variants), and the in-cluster split's
per-element term is `clusterSplitMX8`, the class. -/

def tileResultMX8 (cfg : Config) (S : Nat) (prev : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) : GTerm :=
  tileResultG GTerm.mma32mx cfg S prev a b kb nblk

theorem tileResultMX8_eq_kfoldMX8 (cfg : Config) (S : Nat) (p : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResultMX8 cfg S p a b kb nblk = kfoldMX8 a b (nblk * kb) := tileResultG_eq_kfoldG GTerm.mma32mx cfg S p a b kb nblk

theorem config_invarianceMX8 (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResultMX8 c₁ S₁ p a b kb nblk = tileResultMX8 c₂ S₂ q a b kb nblk :=
  config_invarianceG GTerm.mma32mx c₁ c₂ S₁ S₂ p q a b kb nblk

theorem space_invariantMX8 (space : List Config) :
    ∀ c₁ ∈ space, ∀ c₂ ∈ space, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat),
      tileResultMX8 c₁ S₁ p a b kb nblk = tileResultMX8 c₂ S₂ q a b kb nblk :=
  fun c₁ _ c₂ _ S₁ S₂ p q a b kb nblk => config_invarianceMX8 c₁ c₂ S₁ S₂ p q a b kb nblk

theorem tileOutputMX8_eq_outputKfoldMX8 (fmt : String) (cfg : Config) (S : Nat) (p : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElem false fmt (tileResultMX8 cfg S p a b kb nblk) = outputKfoldMX8 fmt a b (nblk * kb) := by
  unfold outputKfoldMX8; rw [tileResultMX8_eq_kfoldMX8]

theorem outputMX8_config_invariance (fmt : String) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElem false fmt (tileResultMX8 c₁ S₁ p a b kb nblk) = outputElem false fmt (tileResultMX8 c₂ S₂ q a b kb nblk) := by
  rw [config_invarianceMX8 c₁ c₂ S₁ S₂ p q a b kb nblk]

theorem config_invarianceMX8_bits {α : Type} (P : Prims α) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    eval P (tileResultMX8 c₁ S₁ p a b kb nblk) = eval P (tileResultMX8 c₂ S₂ q a b kb nblk) := by
  rw [config_invarianceMX8 c₁ c₂ S₁ S₂ p q a b kb nblk]

def tileResultMX8K64 (cfg : Config) (S : Nat) (prev : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) : GTerm :=
  tileResultG GTerm.mma64mx8 cfg S prev a b kb nblk

theorem tileResultMX8K64_eq_kfoldMX8K64 (cfg : Config) (S : Nat) (p : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResultMX8K64 cfg S p a b kb nblk = kfoldMX8K64 a b (nblk * kb) :=
  tileResultG_eq_kfoldG GTerm.mma64mx8 cfg S p a b kb nblk

theorem config_invarianceMX8K64 (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResultMX8K64 c₁ S₁ p a b kb nblk = tileResultMX8K64 c₂ S₂ q a b kb nblk :=
  config_invarianceG GTerm.mma64mx8 c₁ c₂ S₁ S₂ p q a b kb nblk

theorem space_invariantMX8K64 (space : List Config) :
    ∀ c₁ ∈ space, ∀ c₂ ∈ space, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat),
      tileResultMX8K64 c₁ S₁ p a b kb nblk = tileResultMX8K64 c₂ S₂ q a b kb nblk :=
  fun c₁ _ c₂ _ S₁ S₂ p q a b kb nblk => config_invarianceMX8K64 c₁ c₂ S₁ S₂ p q a b kb nblk

theorem tileOutputMX8K64_eq_outputKfoldMX8K64 (fmt : String) (cfg : Config) (S : Nat) (p : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElem false fmt (tileResultMX8K64 cfg S p a b kb nblk) = outputKfoldMX8K64 fmt a b (nblk * kb) := by
  unfold outputKfoldMX8K64; rw [tileResultMX8K64_eq_kfoldMX8K64]

theorem outputMX8K64_config_invariance (fmt : String) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElem false fmt (tileResultMX8K64 c₁ S₁ p a b kb nblk) = outputElem false fmt (tileResultMX8K64 c₂ S₂ q a b kb nblk) := by
  rw [config_invarianceMX8K64 c₁ c₂ S₁ S₂ p q a b kb nblk]

theorem config_invarianceMX8K64_bits {α : Type} (P : Prims α) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    eval P (tileResultMX8K64 c₁ S₁ p a b kb nblk) = eval P (tileResultMX8K64 c₂ S₂ q a b kb nblk) := by
  rw [config_invarianceMX8K64 c₁ c₂ S₁ S₂ p q a b kb nblk]

/-- **Two parts, two classes.** A tile of the sm_107 class is never a tile of the sm_100 class, whatever the configurations
and stage counts (one 64-wide step against two 32-wide ones over the same K): the family is a dimension of the MXFP8
specification, which is why `mx_class_for` keys the class on it and the fingerprint carries it. -/
theorem tileResultMX8K64_ne_tileResultMX8 (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) :
    tileResultMX8K64 c₁ S₁ p a b 1 1 ≠ tileResultMX8 c₂ S₂ q a b 2 1 := by
  rw [tileResultMX8K64_eq_kfoldMX8K64, tileResultMX8_eq_kfoldMX8]
  exact kfoldMX8K64_ne_kfoldMX8 a b

/-! ## The 4-bit tiles: the same statements at the `mma64bs`, `mma128bs`, `mma64mx` and `mma128mx4` steps, through the alpha chain -/

def tileResultBS (cfg : Config) (S : Nat) (prev : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) : GTerm :=
  tileResultG GTerm.mma64bs cfg S prev a b kb nblk

theorem tileResultBS_eq_kfoldBS (cfg : Config) (S : Nat) (p : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResultBS cfg S p a b kb nblk = kfoldBS a b (nblk * kb) := tileResultG_eq_kfoldG GTerm.mma64bs cfg S p a b kb nblk

theorem config_invarianceBS (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResultBS c₁ S₁ p a b kb nblk = tileResultBS c₂ S₂ q a b kb nblk :=
  config_invarianceG GTerm.mma64bs c₁ c₂ S₁ S₂ p q a b kb nblk

theorem space_invariantBS (space : List Config) :
    ∀ c₁ ∈ space, ∀ c₂ ∈ space, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat),
      tileResultBS c₁ S₁ p a b kb nblk = tileResultBS c₂ S₂ q a b kb nblk :=
  fun c₁ _ c₂ _ S₁ S₂ p q a b kb nblk => config_invarianceBS c₁ c₂ S₁ S₂ p q a b kb nblk

theorem tileOutputBS_eq_outputKfoldBS (fmt : String) (cfg : Config) (S : Nat) (p : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElemAlpha fmt (tileResultBS cfg S p a b kb nblk) = outputKfoldBS fmt a b (nblk * kb) := by
  unfold outputKfoldBS; rw [tileResultBS_eq_kfoldBS]

theorem outputBS_config_invariance (fmt : String) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElemAlpha fmt (tileResultBS c₁ S₁ p a b kb nblk) = outputElemAlpha fmt (tileResultBS c₂ S₂ q a b kb nblk) := by
  rw [config_invarianceBS c₁ c₂ S₁ S₂ p q a b kb nblk]

theorem config_invarianceBS_bits {α : Type} (P : Prims α) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    eval P (tileResultBS c₁ S₁ p a b kb nblk) = eval P (tileResultBS c₂ S₂ q a b kb nblk) := by
  rw [config_invarianceBS c₁ c₂ S₁ S₂ p q a b kb nblk]

def tileResultBS128 (cfg : Config) (S : Nat) (prev : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) : GTerm :=
  tileResultG GTerm.mma128bs cfg S prev a b kb nblk

theorem tileResultBS128_eq_kfoldBS128 (cfg : Config) (S : Nat) (p : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResultBS128 cfg S p a b kb nblk = kfoldBS128 a b (nblk * kb) := tileResultG_eq_kfoldG GTerm.mma128bs cfg S p a b kb nblk

theorem config_invarianceBS128 (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResultBS128 c₁ S₁ p a b kb nblk = tileResultBS128 c₂ S₂ q a b kb nblk :=
  config_invarianceG GTerm.mma128bs c₁ c₂ S₁ S₂ p q a b kb nblk

theorem space_invariantBS128 (space : List Config) :
    ∀ c₁ ∈ space, ∀ c₂ ∈ space, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat),
      tileResultBS128 c₁ S₁ p a b kb nblk = tileResultBS128 c₂ S₂ q a b kb nblk :=
  fun c₁ _ c₂ _ S₁ S₂ p q a b kb nblk => config_invarianceBS128 c₁ c₂ S₁ S₂ p q a b kb nblk

theorem tileOutputBS128_eq_outputKfoldBS128 (fmt : String) (cfg : Config) (S : Nat) (p : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElemAlpha fmt (tileResultBS128 cfg S p a b kb nblk) = outputKfoldBS128 fmt a b (nblk * kb) := by
  unfold outputKfoldBS128; rw [tileResultBS128_eq_kfoldBS128]

theorem outputBS128_config_invariance (fmt : String) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElemAlpha fmt (tileResultBS128 c₁ S₁ p a b kb nblk) = outputElemAlpha fmt (tileResultBS128 c₂ S₂ q a b kb nblk) := by
  rw [config_invarianceBS128 c₁ c₂ S₁ S₂ p q a b kb nblk]

theorem config_invarianceBS128_bits {α : Type} (P : Prims α) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    eval P (tileResultBS128 c₁ S₁ p a b kb nblk) = eval P (tileResultBS128 c₂ S₂ q a b kb nblk) := by
  rw [config_invarianceBS128 c₁ c₂ S₁ S₂ p q a b kb nblk]

def tileResultMX4 (cfg : Config) (S : Nat) (prev : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) : GTerm :=
  tileResultG GTerm.mma64mx cfg S prev a b kb nblk

theorem tileResultMX4_eq_kfoldMX4 (cfg : Config) (S : Nat) (p : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResultMX4 cfg S p a b kb nblk = kfoldMX4 a b (nblk * kb) := tileResultG_eq_kfoldG GTerm.mma64mx cfg S p a b kb nblk

theorem config_invarianceMX4 (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResultMX4 c₁ S₁ p a b kb nblk = tileResultMX4 c₂ S₂ q a b kb nblk :=
  config_invarianceG GTerm.mma64mx c₁ c₂ S₁ S₂ p q a b kb nblk

theorem space_invariantMX4 (space : List Config) :
    ∀ c₁ ∈ space, ∀ c₂ ∈ space, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat),
      tileResultMX4 c₁ S₁ p a b kb nblk = tileResultMX4 c₂ S₂ q a b kb nblk :=
  fun c₁ _ c₂ _ S₁ S₂ p q a b kb nblk => config_invarianceMX4 c₁ c₂ S₁ S₂ p q a b kb nblk

theorem tileOutputMX4_eq_outputKfoldMX4 (fmt : String) (cfg : Config) (S : Nat) (p : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElemAlpha fmt (tileResultMX4 cfg S p a b kb nblk) = outputKfoldMX4 fmt a b (nblk * kb) := by
  unfold outputKfoldMX4; rw [tileResultMX4_eq_kfoldMX4]

theorem outputMX4_config_invariance (fmt : String) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElemAlpha fmt (tileResultMX4 c₁ S₁ p a b kb nblk) = outputElemAlpha fmt (tileResultMX4 c₂ S₂ q a b kb nblk) := by
  rw [config_invarianceMX4 c₁ c₂ S₁ S₂ p q a b kb nblk]

theorem config_invarianceMX4_bits {α : Type} (P : Prims α) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    eval P (tileResultMX4 c₁ S₁ p a b kb nblk) = eval P (tileResultMX4 c₂ S₂ q a b kb nblk) := by
  rw [config_invarianceMX4 c₁ c₂ S₁ S₂ p q a b kb nblk]

def tileResultMX4K128 (cfg : Config) (S : Nat) (prev : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) : GTerm :=
  tileResultG GTerm.mma128mx4 cfg S prev a b kb nblk

theorem tileResultMX4K128_eq_kfoldMX4K128 (cfg : Config) (S : Nat) (p : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResultMX4K128 cfg S p a b kb nblk = kfoldMX4K128 a b (nblk * kb) := tileResultG_eq_kfoldG GTerm.mma128mx4 cfg S p a b kb nblk

theorem config_invarianceMX4K128 (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) :
    tileResultMX4K128 c₁ S₁ p a b kb nblk = tileResultMX4K128 c₂ S₂ q a b kb nblk :=
  config_invarianceG GTerm.mma128mx4 c₁ c₂ S₁ S₂ p q a b kb nblk

theorem space_invariantMX4K128 (space : List Config) :
    ∀ c₁ ∈ space, ∀ c₂ ∈ space, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat),
      tileResultMX4K128 c₁ S₁ p a b kb nblk = tileResultMX4K128 c₂ S₂ q a b kb nblk :=
  fun c₁ _ c₂ _ S₁ S₂ p q a b kb nblk => config_invarianceMX4K128 c₁ c₂ S₁ S₂ p q a b kb nblk

theorem tileOutputMX4K128_eq_outputKfoldMX4K128 (fmt : String) (cfg : Config) (S : Nat) (p : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElemAlpha fmt (tileResultMX4K128 cfg S p a b kb nblk) = outputKfoldMX4K128 fmt a b (nblk * kb) := by
  unfold outputKfoldMX4K128; rw [tileResultMX4K128_eq_kfoldMX4K128]

theorem outputMX4K128_config_invariance (fmt : String) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElemAlpha fmt (tileResultMX4K128 c₁ S₁ p a b kb nblk) = outputElemAlpha fmt (tileResultMX4K128 c₂ S₂ q a b kb nblk) := by
  rw [config_invarianceMX4K128 c₁ c₂ S₁ S₂ p q a b kb nblk]

theorem config_invarianceMX4K128_bits {α : Type} (P : Prims α) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    eval P (tileResultMX4K128 c₁ S₁ p a b kb nblk) = eval P (tileResultMX4K128 c₂ S₂ q a b kb nblk) := by
  rw [config_invarianceMX4K128 c₁ c₂ S₁ S₂ p q a b kb nblk]

/-- **Two instructions, two classes**, for either 4-bit format (one 128-wide step against two 64-wide ones over the same K). -/
theorem tileResultBS128_ne_tileResultBS (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) :
    tileResultBS128 c₁ S₁ p a b 1 1 ≠ tileResultBS c₂ S₂ q a b 2 1 := by
  rw [tileResultBS128_eq_kfoldBS128, tileResultBS_eq_kfoldBS]
  exact kfoldBS128_ne_kfoldBS a b

theorem tileResultMX4K128_ne_tileResultMX4 (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) :
    tileResultMX4K128 c₁ S₁ p a b 1 1 ≠ tileResultMX4 c₂ S₂ q a b 2 1 := by
  rw [tileResultMX4K128_eq_kfoldMX4K128, tileResultMX4_eq_kfoldMX4]
  exact kfoldMX4K128_ne_kfoldMX4 a b

/-! ## The block-scale fp8 tile (`gemm.e4m3.kblock128`, `gemm/blocks/blockscale_fp8.py`)

The kernel's structure IS the class's fold, block by block: the MMA warp runs the dense lowering's one-stage fold
(`stepsPerBlock` steps from a reset accumulator; accumulator stage `kb % S`) for block `kb`, and the epilogue warps fold
the completed partial into their running sum with `fma(blockScale kb, P_kb, sum)` from +0, blocks ascending
(`promoteLoop`). `promoteLoop_eq_kblock128` shows the loop is the class for every stage count and block count, and the
tile model erases the configuration as for the other classes (which tile a CTA owns never enters the term). The generated
`gemmfp8bsloop*` cases conform the loop to the Python engine's per-block traces assembled by the block digit; the A2
surface of this class (the `kind::f8f6f4` variants of the space and the promotion pass on every tile shape) is checked bit
for bit against the exact CPU model by the GPU test. -/

/-- The kernel's block loop: block `kb` through accumulator stage `kb % S`, its partial folded into the running sum. -/
def promoteLoop (a : RowData) (b : ColData) (S : Nat) : Nat → GTerm
  | 0 => .zero
  | kb+1 =>
    let _stage := kb % S
    GTerm.fma (blockScale a b kb) (foldStepsG GTerm.mma32 a b .zero (kb * stepsPerBlock) stepsPerBlock) (promoteLoop a b S kb)

/-- **The promotion loop computes the class**, for every stage count and block count. -/
theorem promoteLoop_eq_kblock128 (a : RowData) (b : ColData) (S KB : Nat) : promoteLoop a b S KB = kblock128 a b KB := by
  unfold kblock128
  induction KB with
  | zero => rfl
  | succ n ih => simp only [promoteLoop, foldBlocks, blockPartial32, ih]

def tileResultKB (cfg : Config) (S : Nat) (_prev : GTerm) (a : RowData) (b : ColData) (KB : Nat) : GTerm :=
  let _owner := (cfg.tm, cfg.tn, cfg.clusterM, cfg.clusterN, cfg.twoCta, cfg.rasterM, cfg.swizzle)
  promoteLoop a b S KB

theorem tileResultKB_reset (cfg : Config) (S : Nat) (p q : GTerm) (a : RowData) (b : ColData) (KB : Nat) :
    tileResultKB cfg S p a b KB = tileResultKB cfg S q a b KB := rfl

theorem tileResultKB_eq_kblock128 (cfg : Config) (S : Nat) (p : GTerm) (a : RowData) (b : ColData) (KB : Nat) :
    tileResultKB cfg S p a b KB = kblock128 a b KB := by
  unfold tileResultKB; exact promoteLoop_eq_kblock128 a b S KB

/-- **Config invariance (R3) of the block-scale class**: any two configurations, stage counts and carried accumulators give
the same per-element term; the hardware hypothesis is that the `kind::f8f6f4` variants of the space are one per-element
function (A2). -/
theorem config_invarianceKB (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (KB : Nat) :
    tileResultKB c₁ S₁ p a b KB = tileResultKB c₂ S₂ q a b KB := by
  rw [tileResultKB_eq_kblock128, tileResultKB_eq_kblock128]

theorem space_invariantKB (space : List Config) :
    ∀ c₁ ∈ space, ∀ c₂ ∈ space, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (KB : Nat),
      tileResultKB c₁ S₁ p a b KB = tileResultKB c₂ S₂ q a b KB :=
  fun c₁ _ c₂ _ S₁ S₂ p q a b KB => config_invarianceKB c₁ c₂ S₁ S₂ p q a b KB

/-- Through the chain (the conversion alone): the output element of a tile is the class's output. -/
theorem tileOutputKB_eq_outputKblock (fmt : String) (cfg : Config) (S : Nat) (p : GTerm) (a : RowData) (b : ColData) (KB : Nat) :
    outputElem false fmt (tileResultKB cfg S p a b KB) = outputKblock fmt a b KB := by
  unfold outputKblock; rw [tileResultKB_eq_kblock128]

theorem outputKB_config_invariance (fmt : String) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (KB : Nat) :
    outputElem false fmt (tileResultKB c₁ S₁ p a b KB) = outputElem false fmt (tileResultKB c₂ S₂ q a b KB) := by
  rw [config_invarianceKB c₁ c₂ S₁ S₂ p q a b KB]

theorem config_invarianceKB_bits {α : Type} (P : Prims α) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (KB : Nat) :
    eval P (tileResultKB c₁ S₁ p a b KB) = eval P (tileResultKB c₂ S₂ q a b KB) := by
  rw [config_invarianceKB c₁ c₂ S₁ S₂ p q a b KB]

/-- A tile of the block-scale class is never a tile of the per-tensor fp8 class, whatever the configurations and stage
counts. -/
theorem tileResultKB_ne_kfold32 (cfg : Config) (S : Nat) (p : GTerm) (a : RowData) (b : ColData) :
    tileResultKB cfg S p a b 1 ≠ kfold32 a b stepsPerBlock := by
  rw [tileResultKB_eq_kblock128]; exact kblock128_ne_kfold32 a b

end BIK.GEMM
