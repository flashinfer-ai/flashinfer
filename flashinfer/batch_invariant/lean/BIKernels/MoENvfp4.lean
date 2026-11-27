-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.MoE
/-!
# The NVFP4 MoE classes (`moe.nvfp4.<act>`): the W4A4 chains on the bf16 MoE model

Mirrors `flashinfer/fused_moe/batch_invariant/spec_nvfp4.py`. The class is the bf16 MoE class (`BIKernels.MoE`: the
routing, the slot-order combine, the launch model and the grouped placement erasure) with NVFP4 experts and per-token
NVFP4 activations. What is new is the epilogue arithmetic on the raw accumulators of the dense NVFP4 GEMM classes
(opaque: `gemm.nvfp4.kfold` / `kfold128` on the expert weight shape, the block scales applied inside the instruction):

* `w4a4RowScale`: `s = mul(alpha_e, pts_row)`, the expert's alpha (the caller's `1 / global_scale[e]`, as given) times the
  row's per-token scale (the per-token quantizer's output), one `mul.rn` per (row, expert);
* `gluScaledT`: GEMM1 of the gated classes, `cvt_bf16(glu_pre(mul(g, s), mul(u, s)))` -- the bf16 class's gated chain
  (every activation `GAct` with its scalars) on the scaled accumulator pair, its exponential and divisions on the part's
  SFU (`MoE.gluHwT`: `ex2Approx` / `rcpApprox`, opaque per-part functions; the chains without them -- PowLU, the ReLU
  family -- are the declared sequences);
* `actScaledT`: GEMM1 of the ungated classes, `cvt_bf16(act(mul(acc, s)))` -- the bf16 class's ungated chain (`UAct`) on
  the scaled accumulator (`MoE.actElemHwT`);
* `scaledOutT`: GEMM2, `cvt_bf16(mul(acc, s))`.

The two per-token quantizations (of the input row and of the materialised bf16 intermediate) are the quantizer class's
own terms (`BIKernels.Quant`: the per-token variant); the W4A4 class contains them by reference (their fingerprints are
part of its identifier). The instruction class of each expert weight is the dense rule's (`GEMM.fp4ClassK128` on the
weight shape, gated by the architecture family on the Python side): `expertClassesK128` states it as a function of the
model's dimensions only. The generated `MoENvfp4Cases.lean` couples every definition to the Python spec node for node.
-/
namespace BIK.MoENvfp4
open BIK.MoE

/-- W4A4: `s = mul(alpha_e, pts_row)` (the expert's alpha as given times the token's per-token scale). -/
def w4a4RowScale (alpha pts : MTerm) : MTerm := .mul alpha pts

/-- GEMM1 of the gated W4A4 classes: `cvt_bf16(glu_pre(mul(g, s), mul(u, s)))` on the raw accumulators, the exponential and
the divisions on the part's SFU (`spec_nvfp4.glu_scaled_elem`). -/
def gluScaledT (a : GAct) (g u s : MTerm) : MTerm := gluHwT a (.mul g s) (.mul u s)

/-- GEMM1 of the ungated W4A4 classes: `cvt_bf16(act(mul(acc, s)))` on the raw accumulator (`spec_nvfp4.act_scaled_elem`). -/
def actScaledT (a : UAct) (acc s : MTerm) : MTerm := actElemHwT a (.mul acc s)

/-- GEMM2 of the W4A4 classes: `cvt_bf16(mul(acc, s))` (`spec_nvfp4.scaled_out_elem`). -/
def scaledOutT (acc s : MTerm) : MTerm := .cvt "bf16" (.mul acc s)

/-- **The W4A4 GEMM1 chain is the bf16 class's chain on the scaled accumulators**: the scale enters once per accumulator,
before any activation arithmetic; nothing of the chain is re-ordered. -/
theorem gluScaledT_is_gluHwT_of_scaled (a : GAct) (g u s : MTerm) : gluScaledT a g u s = gluHwT a (.mul g s) (.mul u s) := rfl

theorem actScaledT_is_actElemHwT_of_scaled (a : UAct) (acc s : MTerm) : actScaledT a acc s = actElemHwT a (.mul acc s) := rfl

/-- The chains without an exponential or a division are the declared sequences: the ReLU family and PowLU. -/
theorem actScaledT_relu2_exact (acc s : MTerm) : actScaledT .relu2 acc s = actElemT .relu2 (.mul acc s) := rfl

theorem gluScaledT_powlu_exact (m : String) (l : Option String) (g u s : MTerm) :
    gluScaledT (.powlu m l) g u s = gluT (.powlu m l) (.mul g s) (.mul u s) := rfl

/-- The default SwiGLU class's GEMM1: the SFU SiLU of the scaled gate times the scaled up accumulator. -/
theorem gluScaledT_silu (g u s : MTerm) : gluScaledT .silu g u s = .cvt "bf16" (.mul (siluHwT (.mul g s)) (.mul u s)) := rfl

/-- **The row scale is one product**: with a unit per-token scale the chain's scale is the expert's alpha alone (the
per-tensor pack of the TRTLLM backends enters the same chain). -/
theorem w4a4RowScale_unit (alpha : MTerm) : w4a4RowScale alpha (c "1.0") = .mul alpha (c "1.0") := rfl

/-- GEMM2's chain is the dense 4-bit class's scaled conversion (`mm_fp4`'s `cvt(mul(acc, alpha))`) with the row scale as
the alpha. -/
theorem scaledOutT_eq (acc s : MTerm) : scaledOutT acc s = .cvt "bf16" (.mul acc s) := rfl

/-! ## The static-intermediate class `moe.nvfp4.<act>.static`: GEMM1's epilogue quantizes the fp32 gated value per 16-column block
with the expert's static global scale `gs2[e]` by the per-tensor NVFP4 quantizer's chain (`Quant.lean`'s `sfValue` / `sfQuant` /
`invScale` / `quantElem`, restated on `MTerm`; the Python spec pins the two restatements to one S-expression), GEMM2 carries the
per-expert constant `inv2[e] = div(1, gs2[e])`. -/

def nvfp4CSixth : MTerm := c "0.1666666716337204"
def nvfp4FltMax : MTerm := c "3.4028234663852886e+38"

/-- `sf_f = mul(amax, mul(gs, 1/6))`. -/
def nvfp4SfValue (amax gs : MTerm) : MTerm := .mul amax (.mul gs nvfp4CSixth)

/-- `sf_d = cvt_e4m3(sf_f)`: the stored scale byte as its widened value. -/
def nvfp4SfQuant (sf : MTerm) : MTerm := .cvt "e4m3" sf

/-- `inv = fmin(div(1, mul(sf_d, div(1, gs))), FLT_MAX)`: the quantizer's encode scale, the same `div` primitive as `staticInv2T`. -/
def nvfp4InvScale (sfd gs : MTerm) : MTerm := .fmin (.div (c "1.0") (.mul sfd (.div (c "1.0") gs))) nvfp4FltMax

/-- `q = cvt_e2m1(mul(x, inv))`. -/
def nvfp4QuantElem (x inv : MTerm) : MTerm := .cvt "e2m1" (.mul x inv)

def pairUp : List MTerm → List MTerm
  | a :: b :: rest => MTerm.fmaxc a b :: pairUp rest
  | rest => rest

/-- One pairing round per unit of fuel (the list length bounds the rounds: structural recursion, no measure). -/
def pairwiseMaxFuel : Nat → List MTerm → MTerm
  | _, [x] => x
  | 0, xs => xs.headD (c "0.0")
  | n + 1, xs => pairwiseMaxFuel n (pairUp xs)

/-- The adjacent pairwise `fmaxc` tree over a power-of-two list: the quantizer's canonical block reduction (exact in every
association: the max of non-negative values). -/
def pairwiseMax (xs : List MTerm) : MTerm := pairwiseMaxFuel xs.length xs

/-- The block amax of 16 values: the pairwise `fmaxc` tree over `fabs(x)`. -/
def nvfp4BlockAmax (xs : List MTerm) : MTerm := pairwiseMax (xs.map MTerm.fabs)

/-- The fp32 gated value of the W4A4 chain before any rounding: `glu_pre(mul(g, s), mul(u, s))` (`spec_nvfp4.glu_scaled_pre`). -/
def gluScaledPreT (a : GAct) (g u s : MTerm) : MTerm := gluPreHwT a (.mul g s) (.mul u s)

/-- The fp32 ungated value before any rounding: `act(mul(acc, s))`. -/
def actScaledPreT (a : UAct) (acc s : MTerm) : MTerm := ungatedHwT a (.mul acc s)

/-- The block's scale byte: the per-tensor chain on the block's 16 fp32 values with the expert's static scale. -/
def staticBlockSfT (vals : List MTerm) (gs : MTerm) : MTerm := nvfp4SfQuant (nvfp4SfValue (nvfp4BlockAmax vals) gs)

/-- The block's `i`-th code: `cvt_e2m1(mul(v_i, inv))` with `inv` from the block's scale byte. -/
def staticBlockCodeT (vals : List MTerm) (gs : MTerm) (i : Nat) : MTerm :=
  nvfp4QuantElem (vals.getD i (c "0.0")) (nvfp4InvScale (staticBlockSfT vals gs) gs)

/-- GEMM2's static factor `inv2[e] = div(1, gs2[e])`: the quantizer's `div`, formed once per expert on the host and read as given. -/
def staticInv2T (gs : MTerm) : MTerm := .div (c "1.0") gs

/-- The static class's GEMM2: `cvt_bf16(mul(acc, mul(alpha2[e], inv2[e])))`. -/
def staticGemm2T (acc alpha gs : MTerm) : MTerm := scaledOutT acc (w4a4RowScale alpha (staticInv2T gs))

/-- The gated W4A4 chain of the per-token class is the bf16 rounding of the static class's fp32 value: the two classes share
the value and differ only in what quantizes it (the per-token row pass vs the in-place block chain). -/
theorem gluScaledT_is_cvt_of_pre (a : GAct) (g u s : MTerm) : gluScaledT a g u s = .cvt "bf16" (gluScaledPreT a g u s) := rfl

theorem actScaledT_is_cvt_of_pre (a : UAct) (acc s : MTerm) : actScaledT a acc s = .cvt "bf16" (actScaledPreT a acc s) := rfl

/-- **The fused epilogue is the per-tensor quantizer's chain on the fp32 gated values**: the block's scale and codes are
`sfQuant (sfValue amax gs)` and `quantElem v (invScale sf gs)` with `amax` the canonical block tree -- the fusion moves no
arithmetic. -/
theorem static_block_is_quant_chain (vals : List MTerm) (gs : MTerm) (i : Nat) :
    staticBlockSfT vals gs = .cvt "e4m3" (.mul (pairwiseMax (vals.map MTerm.fabs)) (.mul gs nvfp4CSixth))
      ∧ staticBlockCodeT vals gs i = .cvt "e2m1" (.mul (vals.getD i (c "0.0")) (nvfp4InvScale (staticBlockSfT vals gs) gs)) := ⟨rfl, rfl⟩

/-- The static class's GEMM2 is the W4A4 GEMM2 chain with the per-expert constant in place of the per-token scale. -/
theorem staticGemm2T_is_scaledOut (acc alpha gs : MTerm) : staticGemm2T acc alpha gs = scaledOutT acc (.mul alpha (.div (c "1.0") gs)) := rfl

/-- The NVFP4 instruction class of each expert weight shape by the dense rule (Rule A per weight: GEMM1 on `(N1, H)`,
GEMM2 on `(H, I)`), a function of the model's fixed dimensions only: a token's class cannot depend on the batch, the
expert load or the slot. -/
def expertClassesK128 (N1 H I : Nat) : Bool × Bool := (GEMM.fp4ClassK128 N1 H, GEMM.fp4ClassK128 H I)

theorem expert_classes_no_batch (N1 H I : Nat) : expertClassesK128 N1 H I = expertClassesK128 N1 H I := rfl

/-- The grouped block-scaled space: the dense records of the nvfp4 space at swizzle 1 (no in-cluster split, no operand swap,
no B-reuse tile, no in-CTA chunk fold: the group scheduler has none of them), the same records for both instruction classes
(the step is the class). -/
def groupedBlockScaledOK (cfg : GEMM.Config) : Bool :=
  GEMM.ConfigOKNvfp4 cfg && (cfg.swizzle == 1 && (cfg.clusterK == 1 && (!cfg.swapAb && (cfg.tm != 512 && !cfg.ctaChunks))))

/-- A grouped block-scaled record is a record of the dense NVFP4 class (the grouped kernel is a placement of it). -/
theorem grouped_blockscaled_in_dense (cfg : GEMM.Config) (h : groupedBlockScaledOK cfg = true) : GEMM.ConfigOKNvfp4 cfg = true :=
  (Bool.and_eq_true _ _).mp h |>.1

/-- The swapped decode space of the grouped block-scaled classes (`blocks/swap.py`): the dense class's swapped 1-CTA
records at a narrow token tile (128 weight rows x 8 / 16 / 32 tokens), no cluster, swizzle 1, no in-cluster split. The
block-scaled step is symmetric in its operands, so each is a placement of the class (the dense class's swapped decode
records). Disjoint from `groupedBlockScaledOK` (`swapAb`). -/
def groupedBlockScaledSwapOK (cfg : GEMM.Config) : Bool :=
  GEMM.ConfigOKNvfp4 cfg && (cfg.swizzle == 1 && (cfg.clusterK == 1 && (cfg.swapAb && (cfg.tm == 128 && (cfg.tn ≤ 32 &&
    (cfg.clusterM == 1 && (cfg.clusterN == 1 && (!cfg.twoCta && !cfg.ctaChunks))))))))

/-- A swapped decode record is a record of the dense NVFP4 class (the swapped grouped kernel is a placement of it). -/
theorem grouped_blockscaled_swap_in_dense (cfg : GEMM.Config) (h : groupedBlockScaledSwapOK cfg = true) : GEMM.ConfigOKNvfp4 cfg = true :=
  (Bool.and_eq_true _ _).mp h |>.1

/-- The two spaces are disjoint: no record is both. -/
theorem grouped_spaces_disjoint (cfg : GEMM.Config) (h : groupedBlockScaledOK cfg = true) : groupedBlockScaledSwapOK cfg = false := by
  unfold groupedBlockScaledOK at h
  unfold groupedBlockScaledSwapOK
  simp only [Bool.and_eq_true] at h
  obtain ⟨_, _, _, hswap, _⟩ := h
  simp [Bool.not_eq_true'] at hswap
  simp [hswap]

end BIK.MoENvfp4
