-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.MoE
import BIKernels.FP8BlockQuant
/-!
# The block-scale fp8 MoE classes `moe.e4m3bs.<act>` / `moe.e4m3bs32.<act>` (`fused_moe/batch_invariant/spec_fp8bs.py`); namespace `BIK.MoEFp8Bs` over the bf16 MoE model `BIK.MoE`

DeepSeek-style fp8: e4m3 experts with one fp32 scale per 128x128 block, e4m3 activations with one fp32 scale per 1x128 block,
both by the `fp8bs.quant` class (`FP8BlockQuant.lean`). The grouped kernel is a placement of the dense block-scale class
`gemm.e4m3.kblock128` per expert (`GEMM.kblock128`: the per-block `mma32` fold promoted in block order with
`fma(mul(s_a, s_b), P_kb, acc)`), so the class has no software scale after the fold: GEMM1 is the bf16 class's chain on the
promoted accumulators, GEMM2 the plain conversion, and the intermediate's requantization is the `fp8bs.quant` class on the
bf16-rounded chain value -- fused into GEMM1's epilogue or as the standalone pass, two placements of one term.
-/
namespace BIK.MoEFp8Bs

open BIK.MoE

/-- The grouped block-scale kernel's result for an expanded row: the dense class's block fold over `KB` blocks; the placement (the
expert segment, the load, the position in the segment, the tile and its scale ring) is erased. -/
def groupedRowBS (_grouping : Nat → Nat × Nat) (_load : Nat → Nat) (_pos : Nat → Nat)
    (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) : GEMM.GTerm :=
  GEMM.kblock128 a b KB

/-- **R4: grouped == standalone.** -/
theorem grouped_eq_dense_bs (g : Nat → Nat × Nat) (l p : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) :
    groupedRowBS g l p a b KB = GEMM.kblock128 a b KB := rfl

/-- **Expert-load, grouping and position independence (R2 b, c).** -/
theorem grouped_placement_irrelevant_bs (g₁ g₂ : Nat → Nat × Nat) (l₁ l₂ p₁ p₂ : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) :
    groupedRowBS g₁ l₁ p₁ a b KB = groupedRowBS g₂ l₂ p₂ a b KB := rfl

/-- The grouped kernel restarts the running sum per tile (the dense `tileResultKB_reset`); the promotion loop is the class
(`GEMMConfig.promoteLoop_eq_kblock128`) whatever the accumulator ring's depth. -/
theorem grouped_tile_reset_bs (cfg : GEMM.Config) (S : Nat) (prev₁ prev₂ : GEMM.GTerm) (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) :
    GEMM.tileResultKB cfg S prev₁ a b KB = GEMM.tileResultKB cfg S prev₂ a b KB := GEMM.tileResultKB_reset cfg S prev₁ prev₂ a b KB

/-- A different class from the fp8 per-tensor fold (`moe.e4m3.<act>`: `kfold32`) at the same operands. -/
theorem grouped_bs_ne_per_tensor (g : Nat → Nat × Nat) (l p : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) :
    groupedRowBS g l p a b 1 ≠ GEMM.kfold32 a b GEMM.stepsPerBlock := GEMM.kblock128_ne_kfold32 a b

/-- GEMM1 of the block-scale class: the bf16 class's gated chain on the promoted accumulators (`Fp8BsMoESpec.gemm1`). -/
def e4m3bsGemm1T (a : GAct) (g u : MTerm) : MTerm := gluT a g u

/-- The ungated chains of the class (`moe.e4m3bs.relu2` and the other ungated activations): the bf16 class's `actElemT`. -/
def e4m3bsGemm1UngatedT (a : UAct) (x : MTerm) : MTerm := actElemT a x

/-- GEMM2 of the block-scale class: the plain conversion (`Fp8BsMoESpec.gemm2`). -/
def e4m3bsGemm2T (acc : MTerm) : MTerm := .cvt "bf16" acc

theorem e4m3bs_gemm1_is_bf16_chain (a : GAct) (g u : MTerm) : e4m3bsGemm1T a g u = gluT a g u := rfl
theorem e4m3bs_gemm1_silu_is_swiglu (g u : MTerm) : e4m3bsGemm1T .silu g u = swigluT g u := rfl
theorem e4m3bs_gemm1_ungated_is_relu2 (x : MTerm) : e4m3bsGemm1UngatedT .relu2 x = relu2T x := rfl

/-- **The intermediate the requantization reads is the bf16-rounded chain value**: the `fp8bs.quant` class applied to GEMM1's
rounded output (the declared second quantization). The fused epilogue quantizes the same rounded value, so fused-vs-standalone
is a placement question (the same term on the same values), not a spec one. -/
theorem e4m3bs_requant_input_is_rounded_chain (a : GAct) (g u : MTerm) : e4m3bsGemm1T a g u = .cvt "bf16" (gluPreT a g u) := rfl

/-- The grouped block-scale space: the dense block-scale class's records at swizzle 1 (the group scheduler has no raster swizzle)
with a tile of two promotion warpgroups (N 64 / 128 / 256: the narrow decode tiles have one subtile). -/
def groupedBlockScaleFp8OK (cfg : GEMM.Config) : Bool :=
  GEMM.ConfigOKFp8BS cfg && cfg.swizzle == 1 && (cfg.tn == 64 || cfg.tn == 128 || cfg.tn == 256)

theorem groupedBlockScaleFp8OK_imp_dense (cfg : GEMM.Config) (h : groupedBlockScaleFp8OK cfg = true) : GEMM.ConfigOKFp8BS cfg = true := by
  unfold groupedBlockScaleFp8OK at h
  rw [Bool.and_eq_true, Bool.and_eq_true] at h
  exact h.1.1

/-- The swapped decode space of the block-scale MoE class (`blocks/swap.py` on the promoted operand block; the MoE class's own records, which the
dense block-scale GEMM's rule does not admit): the weights as the MMA's A operand on a 64- or 128-row tile, the group's tokens as its
B operand in an 8 / 16 / 32 / 64 tile, 1-CTA, no cluster, swizzle 1, no split. The fp8 step is symmetric in its operands (the dense
block-scale class's swapped records, verified bit for bit against the exact model on both parts), so each record's result is
`promoteLoop`'s -- the per-block `mma32` partial promoted in block order; `tileResultKB` does not read the orientation, and
`space_invariantKB` over the union of both spaces is the statement the suites check. -/
def groupedBlockScaleFp8SwapOK (cfg : GEMM.Config) : Bool :=
  cfg.swapAb && (cfg.tm == 64 || cfg.tm == 128) && (cfg.tn == 8 || cfg.tn == 16 || cfg.tn == 32 || cfg.tn == 64) &&
  cfg.clusterM == 1 && cfg.clusterN == 1 && !cfg.twoCta && cfg.swizzle == 1 && cfg.clusterK == 1 && !cfg.ctaChunks

/-- The two spaces are disjoint (a record is dense-derived or swapped). -/
theorem groupedBlockScaleFp8_spaces_disjoint (cfg : GEMM.Config) :
    ¬ (groupedBlockScaleFp8OK cfg = true ∧ groupedBlockScaleFp8SwapOK cfg = true) := by
  intro ⟨h1, h2⟩
  have hd := groupedBlockScaleFp8OK_imp_dense cfg h1
  unfold GEMM.ConfigOKFp8BS at hd
  unfold groupedBlockScaleFp8SwapOK at h2
  simp only [Bool.and_eq_true] at hd h2
  have hs := h2.1.1.1.1.1.1.1.1
  simp [hs] at hd

/-! ## The 32-wide block-scale classes `moe.e4m3bs32.<act>` (`deepseek_fp8_block_size = 32`, `spec_fp8bs.reduction_kblock` at 32)

The same recipe on 32x32 weight blocks and 1x32 activation blocks (the re-blocked TP shards prims-TS serves): the GEMM core
`gemm.e4m3.kblock32` -- `sfbBlk32 b kb` the fp32 scale of column `b`'s 32-row weight block at the 32-element K block `kb`, a block
ONE `mma32` step from +0 (`stepsPerBlock32 = 1`), the partials folded in block order with the same `fma`. The MoE bodies compute it
at the class with one accumulator stage per instruction step (`promoteLoop32`); the loop is the class for every stage count, the
tile result erases the configuration over the same record spaces, and a tile of this class is never a tile of the 128-wide one at
the same operands (`kblock32_ne_kblock128`: the block size is a constant of each class, so the two take separate fingerprints). The
chains are the 128-wide classes' and the quantizer the radix-32 instance (`FP8BlockQuant.blockAmax32`); the requantization is the
standalone pass. The dense block-scale GEMM module keeps the 128-wide class: the core is declared here, the MoE package's own. -/

def stepsPerBlock32 : Nat := 1

def blockPartial32s (a : GEMM.RowData) (b : GEMM.ColData) (kb : Nat) : GEMM.GTerm :=
  GEMM.foldStepsG GEMM.GTerm.mma32 a b .zero (kb * stepsPerBlock32) stepsPerBlock32

/-- `mul.rn(s_a[m, kb], s_b[n / 32, kb])`: the 32-wide block's scale. -/
def blockScale32 (a : GEMM.RowData) (b : GEMM.ColData) (kb : Nat) : GEMM.GTerm := .mul (.sfaBlk a kb) (.sfbBlk32 b kb)

def foldBlocks32 (a : GEMM.RowData) (b : GEMM.ColData) (acc : GEMM.GTerm) : Nat → GEMM.GTerm
  | 0 => acc
  | n+1 => GEMM.GTerm.fma (blockScale32 a b n) (blockPartial32s a b n) (foldBlocks32 a b acc n)

/-- The core `gemm.e4m3.kblock32`: `KB = K / 32` one-step blocks from +0. -/
def kblock32 (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) : GEMM.GTerm := foldBlocks32 a b .zero KB

/-- **A different class from `kblock128`** at the same operands (128 elements of K: four one-step blocks against one four-step block). -/
theorem kblock32_ne_kblock128 (a : GEMM.RowData) (b : GEMM.ColData) : kblock32 a b 4 ≠ GEMM.kblock128 a b 1 := by
  simp [kblock32, GEMM.kblock128, foldBlocks32, GEMM.foldBlocks, blockScale32, GEMM.blockScale, blockPartial32s, GEMM.blockPartial32,
        GEMM.foldStepsG, stepsPerBlock32, GEMM.stepsPerBlock]

/-- Not the per-tensor fp8 fold either: one block against its one step without the scale. -/
theorem kblock32_ne_kfold32 (a : GEMM.RowData) (b : GEMM.ColData) : kblock32 a b 1 ≠ GEMM.kfold32 a b stepsPerBlock32 := by
  simp [kblock32, foldBlocks32, blockScale32, blockPartial32s, GEMM.kfold32, GEMM.kfoldG, GEMM.foldStepsG, stepsPerBlock32]

theorem kblock32_depends_on_blocks (a : GEMM.RowData) (b : GEMM.ColData) : kblock32 a b 1 ≠ kblock32 a b 2 := by
  simp [kblock32, foldBlocks32]

/-- The bodies' per-step promotion loop: block `kb` through accumulator stage `kb % S`, its one-step partial folded into the running sum. -/
def promoteLoop32 (a : GEMM.RowData) (b : GEMM.ColData) (S : Nat) : Nat → GEMM.GTerm
  | 0 => .zero
  | kb+1 =>
    let _stage := kb % S
    GEMM.GTerm.fma (blockScale32 a b kb) (GEMM.foldStepsG GEMM.GTerm.mma32 a b .zero (kb * stepsPerBlock32) stepsPerBlock32) (promoteLoop32 a b S kb)

/-- **The per-step promotion loop computes the class**, for every stage count and block count. -/
theorem promoteLoop32_eq_kblock32 (a : GEMM.RowData) (b : GEMM.ColData) (S KB : Nat) : promoteLoop32 a b S KB = kblock32 a b KB := by
  unfold kblock32
  induction KB with
  | zero => rfl
  | succ n ih => simp only [promoteLoop32, foldBlocks32, blockPartial32s, ih]

def tileResultKB32 (cfg : GEMM.Config) (S : Nat) (_prev : GEMM.GTerm) (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) : GEMM.GTerm :=
  let _owner := (cfg.tm, cfg.tn, cfg.clusterM, cfg.clusterN, cfg.twoCta, cfg.rasterM, cfg.swizzle)
  promoteLoop32 a b S KB

theorem tileResultKB32_reset (cfg : GEMM.Config) (S : Nat) (p q : GEMM.GTerm) (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) :
    tileResultKB32 cfg S p a b KB = tileResultKB32 cfg S q a b KB := rfl

theorem tileResultKB32_eq_kblock32 (cfg : GEMM.Config) (S : Nat) (p : GEMM.GTerm) (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) :
    tileResultKB32 cfg S p a b KB = kblock32 a b KB := by
  unfold tileResultKB32; exact promoteLoop32_eq_kblock32 a b S KB

/-- **Config invariance (R3) of the 32-wide class**: any two records of either space, stage counts and carried accumulators give the
same per-element term (the padded and the swapped body, every tile, cluster and ring depth). -/
theorem config_invarianceKB32 (c₁ c₂ : GEMM.Config) (S₁ S₂ : Nat) (p q : GEMM.GTerm) (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) :
    tileResultKB32 c₁ S₁ p a b KB = tileResultKB32 c₂ S₂ q a b KB := by
  rw [tileResultKB32_eq_kblock32, tileResultKB32_eq_kblock32]

theorem space_invariantKB32 (space : List GEMM.Config) :
    ∀ c₁ ∈ space, ∀ c₂ ∈ space, ∀ (S₁ S₂ : Nat) (p q : GEMM.GTerm) (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat),
      tileResultKB32 c₁ S₁ p a b KB = tileResultKB32 c₂ S₂ q a b KB :=
  fun c₁ _ c₂ _ S₁ S₂ p q a b KB => config_invarianceKB32 c₁ c₂ S₁ S₂ p q a b KB

/-- Through the chain (the conversion alone): the output element of a tile is the class's output. -/
def outputKblock32 (fmt : String) (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) : GEMM.GTerm := GEMM.outputElem false fmt (kblock32 a b KB)

theorem config_invarianceKB32_bits {α : Type} (P : GEMM.Prims α) (c₁ c₂ : GEMM.Config) (S₁ S₂ : Nat) (p q : GEMM.GTerm) (a : GEMM.RowData)
    (b : GEMM.ColData) (KB : Nat) :
    GEMM.eval P (tileResultKB32 c₁ S₁ p a b KB) = GEMM.eval P (tileResultKB32 c₂ S₂ q a b KB) := by
  rw [config_invarianceKB32 c₁ c₂ S₁ S₂ p q a b KB]

/-- A tile of the 32-wide class is never a tile of the 128-wide class at the same K (128 elements), whatever the configurations. -/
theorem tileResultKB32_ne_tileResultKB (c₁ c₂ : GEMM.Config) (S₁ S₂ : Nat) (p q : GEMM.GTerm) (a : GEMM.RowData) (b : GEMM.ColData) :
    tileResultKB32 c₁ S₁ p a b 4 ≠ GEMM.tileResultKB c₂ S₂ q a b 1 := by
  rw [tileResultKB32_eq_kblock32, GEMM.tileResultKB_eq_kblock128]; exact kblock32_ne_kblock128 a b

/-- The grouped kernel's result for an expanded row at the 32-wide class; the placement is erased. -/
def groupedRowBS32 (_grouping : Nat → Nat × Nat) (_load : Nat → Nat) (_pos : Nat → Nat)
    (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) : GEMM.GTerm :=
  kblock32 a b KB

theorem grouped_eq_core_bs32 (g : Nat → Nat × Nat) (l p : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) :
    groupedRowBS32 g l p a b KB = kblock32 a b KB := rfl

theorem grouped_placement_irrelevant_bs32 (g₁ g₂ : Nat → Nat × Nat) (l₁ l₂ p₁ p₂ : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) (KB : Nat) :
    groupedRowBS32 g₁ l₁ p₁ a b KB = groupedRowBS32 g₂ l₂ p₂ a b KB := rfl

/-- A different class from the 128-wide one at the same operands (128 elements of K). -/
theorem grouped_bs32_ne_bs (g : Nat → Nat × Nat) (l p : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) :
    groupedRowBS32 g l p a b 4 ≠ groupedRowBS g l p a b 1 := kblock32_ne_kblock128 a b

/-- GEMM1 and GEMM2 of the 32-wide class are the 128-wide class's chains (`Fp8BsMoESpec.gemm1` / `gemm2` at `block = 32`). -/
def e4m3bs32Gemm1T (a : GAct) (g u : MTerm) : MTerm := e4m3bsGemm1T a g u
def e4m3bs32Gemm1UngatedT (a : UAct) (x : MTerm) : MTerm := e4m3bsGemm1UngatedT a x
def e4m3bs32Gemm2T (acc : MTerm) : MTerm := e4m3bsGemm2T acc

theorem e4m3bs32_chains_are_e4m3bs (a : GAct) (u : UAct) (g x acc : MTerm) :
    (e4m3bs32Gemm1T a g x, e4m3bs32Gemm1UngatedT u x, e4m3bs32Gemm2T acc) = (e4m3bsGemm1T a g x, e4m3bsGemm1UngatedT u x, e4m3bsGemm2T acc) := rfl

end BIK.MoEFp8Bs
