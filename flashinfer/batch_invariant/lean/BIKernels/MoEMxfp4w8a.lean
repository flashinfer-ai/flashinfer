-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.GEMM
import BIKernels.GEMMConfig
import BIKernels.MoE

/-! # The mixed MXFP4-weight x MXFP8-activation MoE classes `moe.mxfp4w8a.<act>`
(`flashinfer/fused_moe/batch_invariant/spec_mxfp4w8a.py`)

The experts are MXFP4 (e2m1 values, one UE8M0 scale per 32-element block, by the `mxfp4.quant` class of `MXQuant.lean`),
the activations MXFP8 (by the `mxfp8.quant` class); both operands' block scales are operands of the mixed-format
block-scaled step, so the class has no software scale: GEMM1 is the bf16 classes' chain on the accumulators (`MoE.gluT` /
`MoE.actElemT`) and GEMM2 the plain conversion. The grouped kernel is a placement of the dense MIXED class of the part --
the SM107 K = 64 mixed step (`GEMM.kfoldMXW48K64`) or the SM100 / SM103 K = 32 one (`GEMM.kfoldMXW48`) -- as
`MoE.groupedRow` is of the bf16 class; the mixed step is its own fold (`GEMM.kfoldMXW48_ne_kfoldMX8`: the e4m3 x e4m3 class
on the re-coded weight computes different bits), so the class is distinct from the MXFP8 MoE class on the same values.
The generated `MoEMxfp4w8aCases.lean` checks the chain terms against the Python specification and the grouped record
space against the dense mixed spaces. -/

namespace BIK.MoEMxfp4w8a
open BIK.MoE

/-! ## The grouped GEMMs: placements of the dense mixed classes -/

/-- The grouped mixed kernel's result for an expanded row on the SM107 class: the dense K = 64 mixed fold; the placement
(the expert segment, the load, the position inside the padded 128-row atoms) is erased. -/
def groupedRowMXW48K64 (_grouping : Nat → Nat × Nat) (_load : Nat → Nat) (_pos : Nat → Nat)
    (a : GEMM.RowData) (b : GEMM.ColData) (steps : Nat) : GEMM.GTerm :=
  GEMM.kfoldMXW48K64 a b steps

/-- The same on the SM100 / SM103 class: the dense K = 32 mixed fold. -/
def groupedRowMXW48 (_grouping : Nat → Nat × Nat) (_load : Nat → Nat) (_pos : Nat → Nat)
    (a : GEMM.RowData) (b : GEMM.ColData) (steps : Nat) : GEMM.GTerm :=
  GEMM.kfoldMXW48 a b steps

/-- **R4: grouped == standalone** on both instruction classes. -/
theorem grouped_eq_dense_mxw48k64 (g : Nat → Nat × Nat) (l p : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) (steps : Nat) :
    groupedRowMXW48K64 g l p a b steps = GEMM.kfoldMXW48K64 a b steps := rfl
theorem grouped_eq_dense_mxw48 (g : Nat → Nat × Nat) (l p : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) (steps : Nat) :
    groupedRowMXW48 g l p a b steps = GEMM.kfoldMXW48 a b steps := rfl

/-- **Expert-load, grouping, position and padding independence (R2 b, c)** on both classes. -/
theorem grouped_placement_irrelevant_mxw48k64 (g₁ g₂ : Nat → Nat × Nat) (l₁ l₂ p₁ p₂ : Nat → Nat)
    (a : GEMM.RowData) (b : GEMM.ColData) (steps : Nat) :
    groupedRowMXW48K64 g₁ l₁ p₁ a b steps = groupedRowMXW48K64 g₂ l₂ p₂ a b steps := rfl
theorem grouped_placement_irrelevant_mxw48 (g₁ g₂ : Nat → Nat × Nat) (l₁ l₂ p₁ p₂ : Nat → Nat)
    (a : GEMM.RowData) (b : GEMM.ColData) (steps : Nat) :
    groupedRowMXW48 g₁ l₁ p₁ a b steps = groupedRowMXW48 g₂ l₂ p₂ a b steps := rfl

/-- The two instruction classes are different folds (one K = 64 step vs two K = 32 steps over the same elements): the
part's class is a declared dimension of the specification (`spec_fingerprint_mxfp4w8a` names it), not a placement. -/
theorem grouped_mxw48_classes_distinct (g : Nat → Nat × Nat) (l p : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) :
    groupedRowMXW48K64 g l p a b 1 ≠ groupedRowMXW48 g l p a b 2 := GEMM.kfoldMXW48K64_ne_kfoldMXW48 a b

/-- The mixed class and the MXFP8 class are different folds on both parts (the mixed step is its own fold of the same
products): the weight format is a dimension of the class, never a re-coding. -/
theorem grouped_mxw48_ne_mx8 (g : Nat → Nat × Nat) (l p : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) :
    groupedRowMXW48 g l p a b 1 ≠ GEMM.kfoldMX8 a b 1 := GEMM.kfoldMXW48_ne_kfoldMX8 a b
theorem grouped_mxw48k64_ne_mx8k64 (g : Nat → Nat × Nat) (l p : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) :
    groupedRowMXW48K64 g l p a b 1 ≠ GEMM.kfoldMX8K64 a b 1 := GEMM.kfoldMXW48K64_ne_kfoldMX8K64 a b

/-- The grouped kernel restarts the accumulator per tile on both mixed classes (the dense `tileResult*_eq_kfold*`). -/
theorem grouped_tile_reset_mxw48k64 (cfg : GEMM.Config) (S : Nat) (prev₁ prev₂ : GEMM.GTerm) (a : GEMM.RowData) (b : GEMM.ColData)
    (kb nblk : Nat) :
    GEMM.tileResultMXW48K64 cfg S prev₁ a b kb nblk = GEMM.tileResultMXW48K64 cfg S prev₂ a b kb nblk := by
  rw [GEMM.tileResultMXW48K64_eq_kfoldMXW48K64, GEMM.tileResultMXW48K64_eq_kfoldMXW48K64]
theorem grouped_tile_reset_mxw48 (cfg : GEMM.Config) (S : Nat) (prev₁ prev₂ : GEMM.GTerm) (a : GEMM.RowData) (b : GEMM.ColData)
    (kb nblk : Nat) :
    GEMM.tileResultMXW48 cfg S prev₁ a b kb nblk = GEMM.tileResultMXW48 cfg S prev₂ a b kb nblk := by
  rw [GEMM.tileResultMXW48_eq_kfoldMXW48, GEMM.tileResultMXW48_eq_kfoldMXW48]

/-! ## The chains: the bf16 classes' terms, no scale -/

/-- GEMM1 of the mixed class for a gated activation: the bf16 classes' chain on the dequantised accumulators
(`MoEMxfp4w8aSpec.gemm1`, inherited from `MoESpec`). -/
def gemm1T (act : GAct) (g u : MTerm) : MTerm := gluT act g u

/-- GEMM1 for an ungated activation. -/
def gemm1UngatedT (act : UAct) (x : MTerm) : MTerm := actElemT act x

/-- GEMM2 of the mixed class: the plain conversion of the accumulator (`MoEMxfp4w8aSpec.gemm2`). -/
def gemm2T (acc : MTerm) : MTerm := .cvt "bf16" acc

theorem gemm1_is_bf16_chain (act : GAct) (g u : MTerm) : gemm1T act g u = gluT act g u := rfl
theorem gemm1_silu_is_swiglu (g u : MTerm) : gemm1T .silu g u = swigluT g u := rfl
theorem gemm1_ungated_is_bf16_chain (act : UAct) (x : MTerm) : gemm1UngatedT act x = actElemT act x := rfl
theorem gemm1_relu2_is_relu2 (x : MTerm) : gemm1UngatedT .relu2 x = relu2T x := rfl

/-- **The intermediate the second quantization pass reads is the bf16-rounded chain value**: the `mxfp8.quant` class applied
to GEMM1's chain output (fused in the epilogue or standalone on the stored row: the same rounded value, so a placement
question, not a specification one). -/
theorem requant_input_is_rounded_chain (act : GAct) (g u : MTerm) : gemm1T act g u = .cvt "bf16" (gluPreT act g u) := rfl

/-! ## The grouped record space -/

/-- The grouped mixed space: the dense records of BOTH mixed MX spaces at swizzle 1 -- no in-cluster split, no operand swap,
no in-CTA chunk fold, no B-reuse tile (tile M 512: the group scheduler has none of them) -- so one record list serves the
SM107 and the SM100 / SM103 class (mirror of `blocks/compile_mxfp4w8a.py::can_implement_grouped_mxfp4w8a`'s record clause
and `schedule_mxfp4w8a.grouped_knobs_mx`). -/
def groupedOK (cfg : GEMM.Config) : Bool :=
  GEMM.ConfigOKMxW48 cfg && GEMM.ConfigOKMxW48K64 cfg && cfg.swizzle == 1 && cfg.clusterK == 1 && !cfg.swapAb &&
  !cfg.ctaChunks && cfg.tm != 512

/-- The swapped decode space of the mixed classes (`blocks/swap_mxfp8.py` on the mixed plans): the dense mixed classes'
swapped 1-CTA records at a narrow token tile (128 weight rows x 8 / 16 / 32 tokens), no cluster, swizzle 1, no in-cluster
split, no in-CTA chunk fold, on both instruction classes. The mixed step is symmetric in its operands as every block-scaled
step (the dense classes' swapped decode records are in their verified spaces), so each is a placement of the class; the stage
width is a lowering choice, not a field of the record. Disjoint from `groupedOK` (`swapAb`). -/
def groupedSwapOK (cfg : GEMM.Config) : Bool :=
  GEMM.ConfigOKMxW48 cfg && GEMM.ConfigOKMxW48K64 cfg && cfg.swizzle == 1 && cfg.clusterK == 1 && cfg.swapAb &&
  !cfg.ctaChunks && cfg.tm == 128 && cfg.tn ≤ 32 && cfg.clusterM == 1 && cfg.clusterN == 1 && !cfg.twoCta

/-- The two record spaces of the class are disjoint. -/
theorem grouped_spaces_disjoint (cfg : GEMM.Config) (h : groupedOK cfg = true) : groupedSwapOK cfg = false := by
  unfold groupedOK at h
  unfold groupedSwapOK
  simp only [Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨_, _⟩, _⟩, _⟩, hswap⟩, _⟩, _⟩ := h
  simp [Bool.not_eq_true'] at hswap
  simp [hswap]

end BIK.MoEMxfp4w8a
