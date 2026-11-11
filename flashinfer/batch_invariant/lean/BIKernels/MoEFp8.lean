-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.MoE
/-!
# The fp8 per-tensor MoE classes (W8A8): the declared chains around the fp8 GEMM folds

Mirrors `flashinfer/fused_moe/batch_invariant/spec_fp8.py` (the classes `moe.e4m3.<act>`). The routing, the activations
(`gluPreT` / `ungatedT`: every typed activation of the layer) and the slot-order combine are the bf16 classes' (`MoE.lean`); the
GEMM folds are the dense fp8 class's (`kfold` at the fp8 step). This module states what the quantized class adds, every op
`.rn` and the two conversions the formats' saturating RNE conversions:

* the scales: FlashInfer's per-tensor scales are multipliers (`q = cvt(mul(x, m))`); the class's dequantization scales are
  their correctly rounded reciprocals `invScaleT m = div(1, m)` (formed once at weight preparation), and the per-tile alpha
  `w8a8Alpha sa sw = mul(sa, sw)` is formed on the device as the dense fp8 class forms its alpha;
* the input quantization `quantE4m3T x m_x = cvt_e4m3(mul(x, m_x))`;
* GEMM1's fused quantization `gluQuantE4m3T` (gated: the activation on the alpha-scaled (gate, up) accumulators, the `mul.rn`
  by the intermediate multiplier, the e4m3 conversion) and `actQuantE4m3T` (ungated, on the single accumulator);
* GEMM2's chain `scaledOutT acc alpha2 = cvt_bf16(mul(acc, alpha2))`.

The theorems state that the fused epilogues are the standalone quantization term applied to the activation value (the fusion
moves no arithmetic), and that the chains are the bf16 classes' activation terms on alpha-scaled accumulators.
-/
namespace BIK.MoEFp8

open BIK.MoE

/-- `spec_fp8.w8a8_alpha`: `alpha_e = mul(s_a, s_w[e])`, the fp8 dense class's alpha, formed on the device per tile. -/
def w8a8Alpha (sa sw : MTerm) : MTerm := .mul sa sw

/-- `spec_fp8.inv_scale`: the correctly rounded reciprocal of a per-tensor multiplier, the class's dequantization scale. -/
def invScaleT (m : MTerm) : MTerm := .div (c "1.0") m

/-- The per-tile alpha from the multipliers of the A operand and of the expert weight: `mul(div(1, m_a), div(1, m_w))`. -/
def alphaOfT (ma mw : MTerm) : MTerm := w8a8Alpha (invScaleT ma) (invScaleT mw)

/-- The static e4m3 quantization of a value by a multiplier: `cvt_e4m3(mul(v, m))` (`spec_fp8.x_quant_e4m3_elem`). -/
def quantE4m3T (v m : MTerm) : MTerm := .cvt "e4m3" (.mul v m)

/-- GEMM1 of the gated W8A8 classes (`spec_fp8.glu_quant_e4m3_elem`), fused:
`cvt_e4m3(mul(glu_pre(mul(g, alpha), mul(u, alpha)), inv))`. -/
def gluQuantE4m3T (a : GAct) (g u alpha inv : MTerm) : MTerm :=
  .cvt "e4m3" (.mul (gluPreT a (.mul g alpha) (.mul u alpha)) inv)

/-- GEMM1 of the ungated W8A8 classes (`spec_fp8.act_quant_e4m3_elem`), fused: `cvt_e4m3(mul(act(mul(acc, alpha)), inv))`. -/
def actQuantE4m3T (a : UAct) (acc alpha inv : MTerm) : MTerm :=
  .cvt "e4m3" (.mul (ungatedT a (.mul acc alpha)) inv)

/-- GEMM2 of the W8A8 classes (`spec_fp8.scaled_out_elem`): `cvt_bf16(mul(acc, alpha2))`. -/
def scaledOutT (acc s : MTerm) : MTerm := .cvt "bf16" (.mul acc s)

/-- **The fused gated epilogue is the standalone quantization term applied to the activation value**: the fusion moves no
arithmetic, it only skips the materialisation of the fp32 activation. -/
theorem gluQuantE4m3_eq_quant_of_gluPre (a : GAct) (g u alpha inv : MTerm) :
    gluQuantE4m3T a g u alpha inv = quantE4m3T (gluPreT a (.mul g alpha) (.mul u alpha)) inv := rfl

/-- The ungated counterpart. -/
theorem actQuantE4m3_eq_quant_of_ungated (a : UAct) (acc alpha inv : MTerm) :
    actQuantE4m3T a acc alpha inv = quantE4m3T (ungatedT a (.mul acc alpha)) inv := rfl

/-- The W8A8 chains are the bf16 classes' activation terms on the alpha-scaled accumulators: the bf16 class converts the same
fp32 activation value to bf16, the W8A8 class multiplies it by the intermediate multiplier and converts it to e4m3. -/
theorem gluQuant_activation_is_bf16_classs (a : GAct) (g u alpha inv : MTerm) :
    gluQuantE4m3T a g u alpha inv = .cvt "e4m3" (.mul (gluPreT a (.mul g alpha) (.mul u alpha)) inv)
      ∧ gluT a (.mul g alpha) (.mul u alpha) = .cvt "bf16" (gluPreT a (.mul g alpha) (.mul u alpha)) := ⟨rfl, rfl⟩

theorem actQuant_activation_is_bf16_classs (a : UAct) (acc alpha inv : MTerm) :
    actQuantE4m3T a acc alpha inv = .cvt "e4m3" (.mul (ungatedT a (.mul acc alpha)) inv)
      ∧ actElemT a (.mul acc alpha) = .cvt "bf16" (ungatedT a (.mul acc alpha)) := ⟨rfl, rfl⟩

/-- The default SwiGLU class's fused chain is the SwiGLU term on the alpha-scaled pair, quantized. -/
theorem gluQuantE4m3_silu (g u alpha inv : MTerm) :
    gluQuantE4m3T .silu g u alpha inv = quantE4m3T (.mul (siluT (.mul g alpha)) (.mul u alpha)) inv := rfl

/-- **The alpha of a tile is one multiply of two reciprocals**: the dequantization scales are the multipliers' correctly
rounded reciprocals, and nothing else of the scales enters the chain. -/
theorem alphaOf_is_mul_of_reciprocals (ma mw : MTerm) :
    alphaOfT ma mw = .mul (.div (c "1.0") ma) (.div (c "1.0") mw) := rfl

/-- The input quantization and the fused intermediate quantization are one declared term (`quantE4m3T`) on different values. -/
theorem fused_quant_is_input_quant_on_activation (a : GAct) (g u alpha inv : MTerm) :
    gluQuantE4m3T a g u alpha inv = quantE4m3T (gluPreT a (.mul g alpha) (.mul u alpha)) inv ∧
    quantE4m3T (.leaf "x" [0]) inv = .cvt "e4m3" (.mul (.leaf "x" [0]) inv) := ⟨rfl, rfl⟩

end BIK.MoEFp8

/-! ## The swapped records of the fp8 per-tensor classes

The fp8 swapped body (`fused_moe/batch_invariant/blocks/fp8_swap_kernel.py`) is the bf16 swapped body on e4m3 operands with the W8A8
chains: `Cᵀ` per expert, the expert's weight rows on the MMA's M side (128 or 64 rows per CTA) and the group's tokens on its N
side as a narrow tile of 16 or 32 rows (`kind::f8f6f4` takes N >= 16). The tcgen05 step folds the same K sequence per output element
whichever operand carries the token, so every output element is the dense fp8 class's `kfold` of the row's data and the expert's
column, and the chain per element is the grouped fp8 body's (`gluQuantE4m3T` / `actQuantE4m3T` / `scaledOutT` on the same
accumulator): the swapped records are placements, and the dispatcher ranks them with the grouped records. The 64-row weight tile's
TMEM layout gives a thread the rows `r` and `r + 8` (the interleaved weight's up / gate pair) instead of lane `+8`: which register
holds the gate, never the term. The predicate mirrors `blocks/fp8_swap_kernel.is_swapped_record`; the space is the four records. -/

namespace BIK.MoEFp8

open BIK.MoE

/-- A swapped record of the fp8 classes: the swapped 1-CTA record at 128 or 64 weight rows and a token tile of 16 or 32, no cluster,
swizzle 1, no in-cluster split, no in-CTA chunk fold. -/
def swappedOKFp8 (cfg : GEMM.Config) : Bool :=
  cfg.swapAb && (cfg.tm == 128 || cfg.tm == 64) && (cfg.tn == 16 || cfg.tn == 32) && cfg.clusterM == 1 && cfg.clusterN == 1 &&
  !cfg.twoCta && cfg.swizzle == 1 && cfg.clusterK == 1 && !cfg.ctaChunks

/-- The fp8 swapped record space (`schedule_fp8.swapped_knobs_fp8`): the weight tiles 128 then 64, each at the token tiles 16 and 32,
raster along the token tiles. -/
def swappedSpaceFp8 : List GEMM.Config :=
  [⟨128, 16, 1, 1, false, true, 1, false, 1, true⟩, ⟨128, 32, 1, 1, false, true, 1, false, 1, true⟩,
   ⟨64, 16, 1, 1, false, true, 1, false, 1, true⟩, ⟨64, 32, 1, 1, false, true, 1, false, 1, true⟩]

theorem swapped_fp8_space_size : swappedSpaceFp8.length = 4 := by decide
theorem swapped_fp8_space_ok : swappedSpaceFp8.all swappedOKFp8 = true := by decide

/-- The fp8 swapped records are disjoint from the dense records (the swapped flag): no record is in both spaces. -/
theorem swapped_fp8_not_dense (cfg : GEMM.Config) (h : swappedOKFp8 cfg = true) : GEMM.ConfigOK cfg = false := by
  cases hs : cfg.swapAb with
  | false => simp [swappedOKFp8, hs] at h
  | true => simp [GEMM.ConfigOK, hs]

/-- **Every record of either fp8 space computes the same tile**: a swapped record's tile result is any dense record's (the dense
class's configuration erasure, `GEMM.config_invariance`), so the two spaces together are one guarantee space; the W8A8 chain is
applied to that tile's accumulators after the fold, record-independently. -/
theorem swapped_fp8_eq_dense (c₁ c₂ : GEMM.Config) (S₁ S₂ : Nat) (p q : GEMM.GTerm) (a : GEMM.RowData) (b : GEMM.ColData) (kb nblk : Nat) :
    GEMM.tileResult c₁ S₁ p a b kb nblk = GEMM.tileResult c₂ S₂ q a b kb nblk :=
  GEMM.config_invariance c₁ c₂ S₁ S₂ p q a b kb nblk

theorem swapped_fp8_space_invariant : ∀ c₁ ∈ swappedSpaceFp8, ∀ c₂ ∈ swappedSpaceFp8, ∀ (S₁ S₂ : Nat) (p q : GEMM.GTerm) (a : GEMM.RowData) (b : GEMM.ColData) (kb nblk : Nat),
    GEMM.tileResult c₁ S₁ p a b kb nblk = GEMM.tileResult c₂ S₂ q a b kb nblk :=
  GEMM.space_invariant swappedSpaceFp8

end BIK.MoEFp8
