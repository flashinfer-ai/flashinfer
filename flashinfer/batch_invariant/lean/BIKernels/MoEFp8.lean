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
namespace BIK.MoE

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

end BIK.MoE
