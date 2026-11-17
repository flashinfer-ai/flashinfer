-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.GEMM
import BIKernels.GEMMConfig
import BIKernels.MoE

/-! # The MXFP8 x MXFP8 MoE classes `moe.mxfp8.<act>` (`flashinfer/fused_moe/batch_invariant/spec_mxfp8.py`)

The experts and the activations are MXFP8 (e4m3 values, one UE8M0 scale per 32-element block, both by the `mxfp8.quant`
class of `MXQuant.lean`); the block scales are operands of the block-scaled step, so the class has no software scale:
GEMM1 is the bf16 classes' chain on the accumulators (`MoE.gluT` / `MoE.actElemT`) and GEMM2 the plain conversion. The
grouped kernel is a placement of the dense MXFP8 class of the part -- the SM107 K = 64 step (`GEMM.kfoldMX8K64`) or the
SM100 / SM103 K = 32 step (`GEMM.kfoldMX8`) -- as `MoE.groupedRow` is of the bf16 class. The generated `MoEMxfp8Cases.lean`
checks the chain terms against the Python specification and the grouped record space against the dense MX spaces. -/

namespace BIK.MoEMxfp8
open BIK.MoE

/-! ## The grouped GEMMs: placements of the dense MX classes -/

/-- The grouped MX kernel's result for an expanded row on the SM107 class: the dense K = 64 fold; the placement (the
expert segment, the load, the position inside the padded 128-row atoms) is erased. -/
def groupedRowMX8K64 (_grouping : Nat → Nat × Nat) (_load : Nat → Nat) (_pos : Nat → Nat)
    (a : GEMM.RowData) (b : GEMM.ColData) (steps : Nat) : GEMM.GTerm :=
  GEMM.kfoldMX8K64 a b steps

/-- The same on the SM100 / SM103 class: the dense K = 32 fold. -/
def groupedRowMX8 (_grouping : Nat → Nat × Nat) (_load : Nat → Nat) (_pos : Nat → Nat)
    (a : GEMM.RowData) (b : GEMM.ColData) (steps : Nat) : GEMM.GTerm :=
  GEMM.kfoldMX8 a b steps

/-- **R4: grouped == standalone** on both instruction classes. -/
theorem grouped_eq_dense_mx8k64 (g : Nat → Nat × Nat) (l p : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) (steps : Nat) :
    groupedRowMX8K64 g l p a b steps = GEMM.kfoldMX8K64 a b steps := rfl
theorem grouped_eq_dense_mx8 (g : Nat → Nat × Nat) (l p : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) (steps : Nat) :
    groupedRowMX8 g l p a b steps = GEMM.kfoldMX8 a b steps := rfl

/-- **Expert-load, grouping, position and padding independence (R2 b, c)** on both classes. -/
theorem grouped_placement_irrelevant_mx8k64 (g₁ g₂ : Nat → Nat × Nat) (l₁ l₂ p₁ p₂ : Nat → Nat)
    (a : GEMM.RowData) (b : GEMM.ColData) (steps : Nat) :
    groupedRowMX8K64 g₁ l₁ p₁ a b steps = groupedRowMX8K64 g₂ l₂ p₂ a b steps := rfl
theorem grouped_placement_irrelevant_mx8 (g₁ g₂ : Nat → Nat × Nat) (l₁ l₂ p₁ p₂ : Nat → Nat)
    (a : GEMM.RowData) (b : GEMM.ColData) (steps : Nat) :
    groupedRowMX8 g₁ l₁ p₁ a b steps = groupedRowMX8 g₂ l₂ p₂ a b steps := rfl

/-- The two instruction classes are different folds (one K = 64 step vs two K = 32 steps over the same elements): the
part's class is a declared dimension of the specification (`spec_fingerprint_mxfp8` names it), not a placement. -/
theorem grouped_mx8_classes_distinct (g : Nat → Nat × Nat) (l p : Nat → Nat) (a : GEMM.RowData) (b : GEMM.ColData) :
    groupedRowMX8K64 g l p a b 1 ≠ groupedRowMX8 g l p a b 2 := GEMM.kfoldMX8K64_ne_kfoldMX8 a b

/-- The grouped kernel restarts the accumulator per tile on both MX classes (the dense `tileResult*_eq_kfold*`). -/
theorem grouped_tile_reset_mx8k64 (cfg : GEMM.Config) (S : Nat) (prev₁ prev₂ : GEMM.GTerm) (a : GEMM.RowData) (b : GEMM.ColData)
    (kb nblk : Nat) :
    GEMM.tileResultMX8K64 cfg S prev₁ a b kb nblk = GEMM.tileResultMX8K64 cfg S prev₂ a b kb nblk := by
  rw [GEMM.tileResultMX8K64_eq_kfoldMX8K64, GEMM.tileResultMX8K64_eq_kfoldMX8K64]
theorem grouped_tile_reset_mx8 (cfg : GEMM.Config) (S : Nat) (prev₁ prev₂ : GEMM.GTerm) (a : GEMM.RowData) (b : GEMM.ColData)
    (kb nblk : Nat) :
    GEMM.tileResultMX8 cfg S prev₁ a b kb nblk = GEMM.tileResultMX8 cfg S prev₂ a b kb nblk := by
  rw [GEMM.tileResultMX8_eq_kfoldMX8, GEMM.tileResultMX8_eq_kfoldMX8]

/-! ## The chains: the bf16 classes' terms, no scale -/

/-- GEMM1 of the MXFP8 class for a gated activation: the bf16 classes' chain on the dequantised accumulators
(`MoEMxfp8Spec.gemm1`, inherited from `MoESpec`). -/
def gemm1T (act : GAct) (g u : MTerm) : MTerm := gluT act g u

/-- GEMM1 for an ungated activation. -/
def gemm1UngatedT (act : UAct) (x : MTerm) : MTerm := actElemT act x

/-- GEMM2 of the MXFP8 class: the plain conversion of the accumulator (`MoEMxfp8Spec.gemm2`). -/
def gemm2T (acc : MTerm) : MTerm := .cvt "bf16" acc

theorem gemm1_is_bf16_chain (act : GAct) (g u : MTerm) : gemm1T act g u = gluT act g u := rfl
theorem gemm1_silu_is_swiglu (g u : MTerm) : gemm1T .silu g u = swigluT g u := rfl
theorem gemm1_ungated_is_bf16_chain (act : UAct) (x : MTerm) : gemm1UngatedT act x = actElemT act x := rfl
theorem gemm1_relu2_is_relu2 (x : MTerm) : gemm1UngatedT .relu2 x = relu2T x := rfl

/-- **The intermediate the second quantization pass reads is the bf16-rounded chain value**: the standalone `mxfp8.quant`
class applied to GEMM1's stored row (the class's declared second pass). A fused spelling would have to produce the same
bytes from the same rounded value, so fused-vs-standalone is a placement question, not a specification one. -/
theorem requant_input_is_rounded_chain (act : GAct) (g u : MTerm) : gemm1T act g u = .cvt "bf16" (gluPreT act g u) := rfl

/-! ## The grouped record space -/

/-- The grouped MX space: the dense records of BOTH MX spaces at swizzle 1 -- no in-cluster split, no operand swap, no
in-CTA chunk fold, no B-reuse tile (tile M 512: the group scheduler has none of them) -- so one record list serves the
SM107 and the SM100 / SM103 class (mirror of `schedule_mxfp8.can_implement_grouped_mxfp8`'s record clause and
`schedule_mxfp8.grouped_knobs_mx`). -/
def groupedOK (cfg : GEMM.Config) : Bool :=
  GEMM.ConfigOKMx8 cfg && GEMM.ConfigOKMx8K64 cfg && cfg.swizzle == 1 && cfg.clusterK == 1 && !cfg.swapAb &&
  !cfg.ctaChunks && cfg.tm != 512

/-! ## The swapped records: a second placement space of the grouped MX GEMM

The swapped body (`fused_moe/batch_invariant/blocks/swap.py`, the unified bodies' swapped form) computes `Cᵀ` per expert: the expert's weight rows on
the MMA's M side (128 rows per CTA, one scale-factor atom) and the group's tokens on its N side as a narrow tile of 8 / 16 / 32
rows read from the dense packing (their scale factors gathered from the padded positions). The block-scaled step is symmetric in
its operands and folds the same K sequence per output element whichever carries the token, so every output element is the dense
MX class's fold of the row's data and the expert's column (`grouped_eq_dense_*`), and the dispatcher ranks the swapped records
with the grouped ones. The fused requantization in the transposed orientation is the quantizer's term on the same bf16-rounded
value (`requant_input_is_rounded_chain`), its block amax gathered across lanes: the max of non-negative values is exact in every
association (`MXQuant.lean`), so the bytes are the standalone quantizer's. The predicate mirrors
`blocks/swap.is_swapped_record` at the MX format; the space is the three token tiles, a member of both dense MX classes' predicates. -/
def groupedSwapOK (cfg : GEMM.Config) : Bool :=
  GEMM.ConfigOKMx8 cfg && GEMM.ConfigOKMx8K64 cfg && cfg.swizzle == 1 && cfg.clusterK == 1 && cfg.swapAb && cfg.tm == 128 &&
  (cfg.tn == 8 || cfg.tn == 16 || cfg.tn == 32) && cfg.clusterM == 1 && cfg.clusterN == 1 && !cfg.twoCta && !cfg.ctaChunks

/-- A swapped record is a record of both dense MX classes (the swapped grouped kernel is a placement of either). -/
theorem grouped_swap_in_mx8 (cfg : GEMM.Config) (h : groupedSwapOK cfg = true) : GEMM.ConfigOKMx8 cfg = true := by
  unfold groupedSwapOK at h
  simp only [Bool.and_eq_true] at h
  exact h.1.1.1.1.1.1.1.1.1.1

theorem grouped_swap_in_mx8k64 (cfg : GEMM.Config) (h : groupedSwapOK cfg = true) : GEMM.ConfigOKMx8K64 cfg = true := by
  unfold groupedSwapOK at h
  simp only [Bool.and_eq_true] at h
  exact h.1.1.1.1.1.1.1.1.1.2

/-- The two spaces are disjoint: no record is both (the swapped flag). -/
theorem grouped_spaces_disjoint (cfg : GEMM.Config) (h : groupedOK cfg = true) : groupedSwapOK cfg = false := by
  unfold groupedOK at h
  unfold groupedSwapOK
  simp only [Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨_, _⟩, _⟩, _⟩, hswap⟩, _⟩, _⟩ := h
  simp [Bool.not_eq_true'] at hswap
  simp [hswap]

end BIK.MoEMxfp8
