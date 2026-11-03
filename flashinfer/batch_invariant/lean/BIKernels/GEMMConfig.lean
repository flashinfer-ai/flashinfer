import BIKernels.GEMM

/-!
# The configuration space of the tcgen05 GEMM, made explicit

`gemm/generator.py` enumerates configurations `(tm, tn, cluster, 2-CTA, raster, swizzle, in-CTA chunk fold)`; the
kernel body (`gemm/blocks`) derives from them the number of smem pipeline stages `S`, the tile assignment and the
tile visiting order. This file states exactly what the proofs claim about that space and where the single hardware
hypothesis sits:

* the mainloop consumes K in blocks of `kb` steps through a ring of `S` stages; for every `S` and every `kb` the
  per-element result is the class `kfold` (`kblockLoop_eq_kfold`);
* the accumulator is reset at every tile, so a persistent CTA's result for a tile does not depend on what it
  computed before (`tileResult_reset`);
* tile shape, cluster, 2-CTA mode, raster order and swizzle choose which CTA owns which tile and in which order; they
  never enter the per-element term (`config_invariance`).

Everything is stated for the opaque primitive `mma16`. The hardware hypothesis is that every MMA instruction variant
the space selects (`tm ∈ {64,128}` 1-CTA, `{128,256}` 2-CTA, `tn ∈ 32..256` and the decode tiles 8 / 16) is this
same per-element function. No Lean statement can discharge it; the GPU test `test_config_agreement` checks the
dispatcher's candidates bit for bit against each other.
-/
namespace BIK.GEMM

/-- A kernel configuration: the generator's knob record (`gemm.generator.Knobs.lean()`): tile M/N of the MMA tiler,
cluster shape, 2-CTA MMA, raster order, swizzle, and the in-CTA chunk fold `ctaChunks` (the split class only: one CTA
folds every chunk of its tile through a TMEM running sum in chunk order; the constant chunk-to-CTA assignment of
`splitKernel`, `ctaSplit_eq_kchunk` below). -/
structure Config where
  tm : Nat
  tn : Nat
  clusterM : Nat
  clusterN : Nat
  twoCta : Bool
  rasterM : Bool
  swizzle : Nat
  ctaChunks : Bool
  deriving DecidableEq, Repr

/-- The in-CTA chunk fold's knob constraint (mirror of the `cta_chunks` clause of `Knobs.well_formed`): the TMEM running
sum fits next to the accumulator stages, tile N at most `maxTn` (`generator.CTA_CHUNKS_MAX_TN` = 256). -/
def CtaChunksOK (maxTn : Nat) (c : Config) : Bool :=
  !c.ctaChunks || c.tn ≤ maxTn

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
  (c.swizzle == 1 || c.swizzle == 2 || c.swizzle == 4 || c.swizzle == 8) && CtaChunksOK 256 c

/-- The in-CTA chunk fold (`ctaChunks`, `gemm/blocks/epilogue.py`) is the split kernel with the CONSTANT assignment
of every chunk to one CTA: that CTA computes each chunk's partial from +0 in turn (the mainloop's fold per chunk) and
folds the partials in chunk order from +0 with `add` (the TMEM running sum) -- the `fixup` of `splitKernel`, so the
record computes the split class `kchunk C` exactly as the workspace split does (R3 between the two implementations
of one `C`). -/
def ctaSplit (a : RowData) (b : ColData) (C kc : Nat) : GTerm := splitKernel a b C kc (fun _ => 0)

theorem ctaSplit_eq_kchunk (a : RowData) (b : ColData) (C kc : Nat) : ctaSplit a b C kc = kchunk a b C kc :=
  splitKernel_eq_kchunk a b C kc _

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

/-- One output tile under `cfg`: the owning CTA (a function of the config) runs the k-block loop from a reset
accumulator. `prev` is whatever that CTA's accumulator held from the previous tile of its persistent schedule. -/
def tileResult (cfg : Config) (S : Nat) (_prev : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat) : GTerm :=
  let _owner := (cfg.tm, cfg.tn, cfg.clusterM, cfg.clusterN, cfg.twoCta, cfg.rasterM, cfg.swizzle)
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

/-- Through the chain: the *output* element of a tile is the unsplit class's output, so no configuration, stage
count or carried accumulator reaches the output. -/
theorem tileOutput_eq_outputKfold (fmt : String) (cfg : Config) (S : Nat) (p : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElem fmt (tileResult cfg S p a b kb nblk) = outputKfold fmt a b (nblk * kb) := by
  unfold outputKfold; rw [tileResult_eq_kfold]

theorem output_config_invariance (fmt : String) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    outputElem fmt (tileResult c₁ S₁ p a b kb nblk) = outputElem fmt (tileResult c₂ S₂ q a b kb nblk) := by
  rw [config_invariance c₁ c₂ S₁ S₂ p q a b kb nblk]

/-- Bits, not terms: under any primitives `P`, both configurations evaluate to the same value. -/
theorem config_invariance_bits {α : Type} (P : Prims α) (c₁ c₂ : Config) (S₁ S₂ : Nat) (p q : GTerm)
    (a : RowData) (b : ColData) (kb nblk : Nat) :
    eval P (tileResult c₁ S₁ p a b kb nblk) = eval P (tileResult c₂ S₂ q a b kb nblk) := by
  rw [config_invariance c₁ c₂ S₁ S₂ p q a b kb nblk]

end BIK.GEMM
