import BIKernels.MoE
import BIKernels.MoENvfp4
/-!
# The MXFP4 MoE classes (`moe.mxfp4.<act>`): the bf16 chains on the raw MX accumulators

Mirrors `flashinfer/fused_moe/batch_invariant/spec_mxfp4.py`. The class is the bf16 MoE class (`BIKernels.MoE`: the
routing, the slot-order combine, the launch model and the grouped placement erasure) with MXFP4 experts and MXFP4
activations: e2m1 elements with one UE8M0 scale per 32 elements on both operands and no global or per-token scale. The
block scales are applied inside the instruction, so the epilogue arithmetic on the raw accumulators of the dense MXFP4 GEMM
classes (opaque: `gemm.mxfp4.kfold` / `kfold128`) is the bf16 class's own:

* `mxGluT`: GEMM1 of the gated classes, `cvt_bf16(glu_pre(g, u))` -- the bf16 class's gated chain (`gluT`, every
  activation `GAct` with its scalars) on the raw accumulator pair;
* `mxActT`: GEMM1 of the ungated classes, `cvt_bf16(act(acc))` -- the bf16 class's ungated chain (`UAct`);
* `mxOutT`: GEMM2, `cvt_bf16(acc)` (the dense MX class's conversion at a unit alpha).

The two block quantizations (of the input row and of the materialised bf16 intermediate) are the `mxfp4.quant` class's
own terms (`BIKernels.MXQuant`); the class contains them by reference (the quantizer's fingerprint is part of its
identifier). The instruction class of both expert GEMMs is the architecture family's (`mx_class_for` on the Python side):
`expertClassesMx` states it as a function of the family alone. The grouped record spaces are the NVFP4 module's shapes
over the dense MXFP4 space. The generated `MoEMxfp4Cases.lean` couples every definition to the Python spec node for node.
-/
namespace BIK.MoEMxfp4
open BIK.MoE

/-- GEMM1 of the gated MXFP4 classes: `cvt_bf16(glu_pre(g, u))` on the raw accumulators (`spec.glu_elem`). -/
def mxGluT (a : GAct) (g u : MTerm) : MTerm := gluT a g u

/-- GEMM1 of the ungated MXFP4 classes: `cvt_bf16(act(acc))` on the raw accumulator (`spec.act_elem`). -/
def mxActT (a : UAct) (acc : MTerm) : MTerm := actElemT a acc

/-- GEMM2 of the MXFP4 classes: `cvt_bf16(acc)` (`spec_mxfp4.mx_out_elem`). -/
def mxOutT (acc : MTerm) : MTerm := .cvt "bf16" acc

/-! ### The hardware-arithmetic classes `moe.mxfp4.<act>.hw`: the chains of `MoE.gluHwT` / `MoE.actElemHwT` on the raw
accumulators (the part's `ex2` / `rcp` in place of the declared exponential and divisions). -/

/-- GEMM1 of the gated hardware classes. -/
def mxGluHwT (a : GAct) (g u : MTerm) : MTerm := gluHwT a g u

/-- GEMM1 of the ungated hardware classes. -/
def mxActHwT (a : UAct) (acc : MTerm) : MTerm := actElemHwT a acc

theorem mxGluHwT_is_gluHwT (a : GAct) (g u : MTerm) : mxGluHwT a g u = gluHwT a g u := rfl
theorem mxActHwT_relu2 (acc : MTerm) : mxActHwT .relu2 acc = mxActT .relu2 acc := rfl

/-- **The MXFP4 GEMM1 chain is the bf16 class's chain**: no scale enters the epilogue (the block scales are the
instruction's), nothing of the bf16 chain is re-ordered. -/
theorem mxGluT_is_gluT (a : GAct) (g u : MTerm) : mxGluT a g u = gluT a g u := rfl

theorem mxActT_is_actElemT (a : UAct) (acc : MTerm) : mxActT a acc = actElemT a acc := rfl

/-- The default gated chain is the bf16 class's SwiGLU chain. -/
theorem mxGluT_silu (g u : MTerm) : mxGluT .silu g u = swigluT g u := rfl

/-- GEMM1 is the conversion of the bf16 class's pre-activation value. -/
theorem mxGluT_is_cvt_of_pre (a : GAct) (g u : MTerm) : mxGluT a g u = .cvt "bf16" (gluPreT a g u) := rfl

theorem mxActT_is_cvt_of_pre (a : UAct) (acc : MTerm) : mxActT a acc = .cvt "bf16" (ungatedT a acc) := rfl

/-- GEMM2 is the NVFP4 class's GEMM2 chain with the row scale removed, not a unit scale: the MX chain has no `mul`. -/
theorem mxOutT_eq (acc : MTerm) : mxOutT acc = .cvt "bf16" acc := rfl

/-- The MXFP4 instruction class of both expert GEMMs: the architecture family's (`true` for the sm_107 K = 128 step), the
same for GEMM1 and GEMM2 and independent of the model's dimensions, the batch, the expert load or the slot. -/
def expertClassesMx (k128 : Bool) : Bool × Bool := (k128, k128)

theorem expert_classes_no_shape (k128 : Bool) (_N1 _H _I : Nat) : expertClassesMx k128 = expertClassesMx k128 := rfl

/-- The grouped block-scaled space at the MXFP4 instruction classes: the dense records of the mxfp4 space at swizzle 1 (no
in-cluster split, no operand swap, no B-reuse tile, no in-CTA chunk fold), the NVFP4 module's shape over `ConfigOKMx4`. -/
def groupedMxOK (cfg : GEMM.Config) : Bool :=
  GEMM.ConfigOKMx4 cfg && (cfg.swizzle == 1 && (cfg.clusterK == 1 && (!cfg.swapAb && (cfg.tm != 512 && !cfg.ctaChunks))))

/-- A grouped record is a record of the dense MXFP4 class (the grouped kernel is a placement of it). -/
theorem grouped_mx_in_dense (cfg : GEMM.Config) (h : groupedMxOK cfg = true) : GEMM.ConfigOKMx4 cfg = true :=
  (Bool.and_eq_true _ _).mp h |>.1

/-- The swapped decode space at the MXFP4 instruction classes (`blocks/swap.py`): the dense class's swapped 1-CTA records
at a narrow token tile (128 weight rows x 8 / 16 / 32 tokens), no cluster, swizzle 1, no in-cluster split. -/
def groupedMxSwapOK (cfg : GEMM.Config) : Bool :=
  GEMM.ConfigOKMx4 cfg && (cfg.swizzle == 1 && (cfg.clusterK == 1 && (cfg.swapAb && (cfg.tm == 128 && (cfg.tn ≤ 32 &&
    (cfg.clusterM == 1 && (cfg.clusterN == 1 && (!cfg.twoCta && !cfg.ctaChunks))))))))

/-- A swapped decode record is a record of the dense MXFP4 class. -/
theorem grouped_mx_swap_in_dense (cfg : GEMM.Config) (h : groupedMxSwapOK cfg = true) : GEMM.ConfigOKMx4 cfg = true :=
  (Bool.and_eq_true _ _).mp h |>.1

/-- The two spaces are disjoint: no record is both. -/
theorem grouped_mx_spaces_disjoint (cfg : GEMM.Config) (h : groupedMxOK cfg = true) : groupedMxSwapOK cfg = false := by
  unfold groupedMxOK at h
  unfold groupedMxSwapOK
  simp only [Bool.and_eq_true] at h
  obtain ⟨_, _, _, hswap, _⟩ := h
  simp [Bool.not_eq_true'] at hswap
  simp [hswap]

end BIK.MoEMxfp4
