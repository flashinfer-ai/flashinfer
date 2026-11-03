import BIKernels.Canon
import BIKernels.Semantics
import BIKernels.Support
/-!
# MX quantization on the engine: the class `mxfp8.quant`

Mirrors `quantization/mx_spec.py`, `quantization/mx_schedule.py` and `quantization/layout.py` of `flashinfer/batch_invariant/`.

* The block amax is one `pairwise` digit of radix 32 (`[5]`) combined with `fmaxc` over the leaf `fabs(x)`; every
  placement `[(V, reg), (32 / V, lane)]` of the kernels (`quantization.blocks.block_amax`) computes it: `config_invarianceC .fmaxc` (Canon.lean) instantiated. There is no row
  digit: a block's bits depend on its 32 inputs alone (`block_locality`), which is why the MX classes are batch
  invariant without any global-scale argument.
* The recipe is a template with **explicit constants** (`Rule`): the normalisation of the amax (`Norm`: TE's rounded
  reciprocal, the correctly rounded quotient, OCP's power of two), the UE8M0 rounding (`Rnd`: the hardware ceil / floor,
  the software nearest), the saturation (`Sat`) and the sign of a zero product (`Zero`). `Rule.te` (= the Python
  `DEFAULT_RULE`) is TransformerEngine's MXFP8 chain, the pinned recipe of both classes;
  the other rules are documented, unpinned variants. `lean_cases.py` couples every chain to the Python
  S-expressions node for node (`mxq*` cases in `Cases.lean`), for every rule.
* `block_amax_invariance` / `block_outputs_invariant`: every valid placement and unit computes the canonical tree and,
  through the chains, `Equiv` scale and codes (bit equality under any commutative interpretation: `sf_bits_equal`);
  `blockamax_padded_bits`: the zero-filling bucket kernel (blocks beyond `K / 32`, rows beyond `M` read as `+0`) equals
  the absent-dropping spec bit for bit under the restricted identity; the launch model: determinism, batch /
  permutation / sub-batch invariance by construction.
* The two scale-factor layouts are index maps of the store: `sfOffset128x4` (FlashInfer's 128x4 swizzle) and
  `sfOffsetLinear`; both are bijections of the padded grid onto the buffer (decided on concrete grids), so the two
  placements carry the same bytes (R4 within the class).
* `ue8m0Byte` is the exact statement of what `cvt.<rp|rz>.satfinite.ue8m0x2.f32` and the software `rn` rule compute
  on the exact magnitude `a * 2^e` of a binary32 input (assumption A5): the Python model `fp32.to_ue8m0` and the device
  are each checked against it (conformance table; the exhaustive device sweep).
-/
namespace BIK.MXQuant
open BIK

/-! ## The block amax -/

/-- The leaf: `fabs` of the widened input element. -/
def leaf (row : Nat → Term) (i : Nat) : Term := .fabs (row i)

/-- Leaf of block `b` (32 elements): the 5 bits are the element index inside the block (`takeN 5` bounds the index for
every list, as `Quant.rowLeaf` does; on the tree's 5-bit lists it is the identity). -/
def blockLeaf (row : Nat → Term) (b : Nat) (bs : List Bool) : Option Term :=
  some (leaf row (32 * b + bitsToNat (takeN 5 bs)))

/-- The spec's block amax: `mx_spec.block_reduction(...).canonical`. -/
def blockAmax (row : Nat → Term) (b : Nat) : Option Term := treeC .fmaxc 5 (blockLeaf row b)

theorem blockAmax_some (row : Nat → Term) (b : Nat) : ∃ a, blockAmax row b = some a :=
  treeC_some .fmaxc 5 _ (fun _ => ⟨_, rfl⟩)

theorem elem_lt_32 (bs : List Bool) : bitsToNat (takeN 5 bs) < 32 := by
  have h1 := bitsToNat_lt (takeN 5 bs)
  have h2 := takeN_length_le 5 bs
  have h3 : 2 ^ (takeN 5 bs).length ≤ 2 ^ 5 := Nat.pow_le_pow_right (by decide) h2
  omega

/-- **The block amax is config-invariant**: any placement of the digit `[5]` (register factors, lane factors finishing
the block by shuffles) computes the canonical block tree in every unit. -/
theorem block_amax_invariance (row : Nat → Term) (b : Nat) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [5] ps) (hc : CoordsOK (flatten ps) cs) :
    OEquiv (execC .fmaxc (flatten ps) (blockLeaf row b) cs) (blockAmax row b) := by
  have h := config_invarianceC .fmaxc [5] ps (blockLeaf row b) cs hp hc
  rw [canonicalDigitsC_eq_treeC] at h
  exact h

/-- `mx_spec.block_amax_placement(V)` / `MXQuantConfig.placement`: `elem = [(vb bits, reg), (lb bits, lane)]`, `vb + lb = 5`
(`V = 32 / 16 / 8` elements per thread: `lb = 0 / 1 / 2`). -/
def placementElem (vb lb : Nat) : List (List Factor) := [[⟨vb, .reg⟩, ⟨lb, .lane⟩]]

theorem placementElem_ok (vb lb : Nat) (h : vb + lb = 5) : PlacementOK [5] (placementElem vb lb) := by
  unfold placementElem
  refine ⟨?_, trivial⟩
  show vb + (lb + 0) = 5
  omega

/-! ## The formats and the recipe constants

Python prints binary32 constants as `repr(float(c))`: `fp32(1/448) = 0x3B124925` is `0.0022321429569274187`,
`2^-8` is `0.00390625`. -/

inductive Fmt where
  | mxfp8
  | mxfp4
  deriving DecidableEq, Repr

def Fmt.elemFmt : Fmt → String
  | .mxfp8 => "e4m3"
  | .mxfp4 => "e2m1"

def Fmt.elemMax : Fmt → String
  | .mxfp8 => "448.0"
  | .mxfp4 => "6.0"

def Fmt.negElemMax : Fmt → String
  | .mxfp8 => "-448.0"
  | .mxfp4 => "-6.0"

/-- TE's `max_norm_rcp`: the binary32 nearest to `1 / elem_max`. -/
def Fmt.cRcp : Fmt → String
  | .mxfp8 => "0.0022321429569274187"
  | .mxfp4 => "0.1666666716337204"

/-- `2^-emax`: the exact normalisation of OCP's `floor(log2(amax)) - emax` rule. -/
def Fmt.cPow2 : Fmt → String
  | .mxfp8 => "0.00390625"
  | .mxfp4 => "0.25"

inductive Norm where
  | rcp   -- `mul(amax, c_rcp)`: TE, FlashInfer's CuTe mxfp8 (the default)
  | div   -- `div(amax, elem_max)`: the correctly rounded quotient
  | pow2  -- `mul(amax, 2^-emax)`: exact; with `rz` the OCP floor rule, with `rp` torchao's CEIL
  deriving DecidableEq, Repr

inductive Rnd where
  | rp   -- `cvt.rp.satfinite.ue8m0x2.f32`: ceil to a power of two (RCEIL; TE / FlashInfer)
  | rz   -- `cvt.rz.satfinite.ue8m0x2.f32`: floor
  | rn   -- the nearest power of two in linear space, ties up (software, exact)
  deriving DecidableEq, Repr

def Rnd.fmt : Rnd → String
  | .rp => "ue8m0_rp"
  | .rz => "ue8m0_rz"
  | .rn => "ue8m0_rn"

inductive Sat where
  | satfinite   -- the element conversion's own saturation (TE)
  | clamp       -- FlashInfer's `fmax(fmin(y, elem_max), -elem_max)` before the conversion
  deriving DecidableEq, Repr

inductive Zero where
  | keep       -- `mul(x, inv)`: a zero product keeps its sign
  | positive   -- `fma(x, inv, +0)`: TE's CuTe-DSL / pre-13.1 forms, `-0 + +0 = +0`
  deriving DecidableEq, Repr

structure Rule where
  norm : Norm := .rcp
  rnd : Rnd := .rp
  sat : Sat := .satfinite
  zero : Zero := .keep
  deriving DecidableEq, Repr

namespace Rule
/-- TE's chain (the Python `DEFAULT_RULE`): the pinned recipe of `mxfp8.quant`. -/
def te : Rule := {}
/-- OCP's shared exponent `floor(log2(amax)) - emax` (lower reconstruction error, clips the block maximum; declined for parity). -/
def ocpFloor : Rule := { norm := .pow2, rnd := .rz }
/-- FlashInfer's mxfp8 element path (the same bits on finite inputs). -/
def fiClamp : Rule := { sat := .clamp }
/-- TE's `fma(x, inv, +0)` element path. -/
def teFma : Rule := { zero := .positive }
def div : Rule := { norm := .div }
def rn : Rule := { rnd := .rn }
end Rule

def one : Term := .const "1.0"
def zeroT : Term := .const "0.0"

/-! ## The chains (`mx_spec.scale_norm`, `scale_quant`, `inv_scale`, `quant_pre`, `saturate`, `quant_elem`) -/

def scaleNorm (f : Fmt) (r : Rule) (amax : Term) : Term :=
  match r.norm with
  | .rcp => .mul amax (.const f.cRcp)
  | .div => .div amax (.const f.elemMax)
  | .pow2 => .mul amax (.const f.cPow2)

/-- `sf_d = cvt_ue8m0_<rnd>(v)`: the stored byte, as its exactly widened value `2^(e8 - 127)`. -/
def scaleQuant (r : Rule) (v : Term) : Term := .cvt r.rnd.fmt v

/-- `inv = div(1, sf_d)`: exact for every UE8M0 value. -/
def invScale (sf : Term) : Term := .div one sf

def scaleChain (f : Fmt) (r : Rule) (amax : Term) : Term × Term :=
  let sf := scaleQuant r (scaleNorm f r amax)
  (sf, invScale sf)

def quantPre (r : Rule) (x inv : Term) : Term :=
  match r.zero with
  | .keep => .mul x inv
  | .positive => .fma x inv zeroT

def saturate (f : Fmt) (r : Rule) (y : Term) : Term :=
  match r.sat with
  | .satfinite => y
  | .clamp => .fmaxc (.fmin y (.const f.elemMax)) (.const f.negElemMax)

/-- `q = cvt.rn.satfinite.<elem>(SAT(quant_pre(x, inv)))`, as the element value. -/
def quantElem (f : Fmt) (r : Rule) (x inv : Term) : Term := .cvt f.elemFmt (saturate f r (quantPre r x inv))

/-- `MXQuantSpec.block(row, b)["sf"]`. -/
def specSf (f : Fmt) (r : Rule) (row : Nat → Term) (b : Nat) : Option Term :=
  (blockAmax row b).map fun a => (scaleChain f r a).1

/-- `MXQuantSpec.block(row, i / 32)["codes"][i % 32]`. -/
def specCode (f : Fmt) (r : Rule) (row : Nat → Term) (i : Nat) : Option Term :=
  (blockAmax row (i / 32)).map fun a => quantElem f r (row i) (scaleChain f r a).2

/-! ## Invariance through the chains (absent-dropping model) -/

theorem scaleNorm_equiv (f : Fmt) (r : Rule) {a a' : Term} (ha : Equiv a' a) : Equiv (scaleNorm f r a') (scaleNorm f r a) := by
  unfold scaleNorm
  cases r.norm
  · exact .mul_congr ha (.refl _)
  · exact .div_congr ha (.refl _)
  · exact .mul_congr ha (.refl _)

theorem scaleChain_equiv (f : Fmt) (r : Rule) {a a' : Term} (ha : Equiv a' a) :
    Equiv (scaleChain f r a').1 (scaleChain f r a).1 ∧ Equiv (scaleChain f r a').2 (scaleChain f r a).2 := by
  have hs : Equiv (scaleQuant r (scaleNorm f r a')) (scaleQuant r (scaleNorm f r a)) := .cvt_congr _ (scaleNorm_equiv f r ha)
  exact ⟨hs, .div_congr (.refl _) hs⟩

theorem quantPre_equiv (r : Rule) (x : Term) {i i' : Term} (hi : Equiv i' i) : Equiv (quantPre r x i') (quantPre r x i) := by
  unfold quantPre
  cases r.zero
  · exact .mul_congr (.refl x) hi
  · exact .fma_congr (.refl x) hi (.refl _)

theorem saturate_equiv (f : Fmt) (r : Rule) {y y' : Term} (hy : Equiv y' y) : Equiv (saturate f r y') (saturate f r y) := by
  unfold saturate
  cases r.sat
  · exact hy
  · exact .fmaxc_congr (.fmin_congr hy (.refl _)) (.refl _)

theorem code_equiv (f : Fmt) (r : Rule) (x : Term) {a a' : Term} (ha : Equiv a' a) :
    Equiv (quantElem f r x (scaleChain f r a').2) (quantElem f r x (scaleChain f r a).2) :=
  .cvt_congr _ (saturate_equiv f r (quantPre_equiv r x (scaleChain_equiv f r ha).2))

/-- **End-to-end.** For every valid placement of the block digit and every unit, under every rule, the kernel's block
amax exists and the scale and every code of the block are `Equiv` to the spec's, hence (`eval_equiv`) bitwise equal
under any commutative interpretation. -/
theorem block_outputs_invariant (f : Fmt) (r : Rule) (row : Nat → Term) (b : Nat) (ps : List (List Factor)) (cs : List (List Bool))
    (hp : PlacementOK [5] ps) (hc : CoordsOK (flatten ps) cs) :
    ∃ a a', blockAmax row b = some a ∧ execC .fmaxc (flatten ps) (blockLeaf row b) cs = some a' ∧
      Equiv (scaleChain f r a').1 (scaleChain f r a).1 ∧
      ∀ e, Equiv (quantElem f r (row (32 * b + e)) (scaleChain f r a').2) (quantElem f r (row (32 * b + e)) (scaleChain f r a).2) := by
  obtain ⟨a, ha⟩ := blockAmax_some row b
  have h := block_amax_invariance row b ps cs hp hc
  rw [ha] at h
  obtain ⟨a', hk, he⟩ := OEquiv.some_right h
  exact ⟨a, a', ha, hk, (scaleChain_equiv f r he).1, fun e => code_equiv f r _ he⟩

/-- Bitwise form of the scale: the kernel's scale byte value equals the spec's under any commutative interpretation. -/
theorem sf_bits_equal {α : Type} (P : Prims α) (f : Fmt) (r : Rule) (row : Nat → Term) (b : Nat) (ps : List (List Factor))
    (cs : List (List Bool)) (hp : PlacementOK [5] ps) (hc : CoordsOK (flatten ps) cs) :
    ∃ a a', blockAmax row b = some a ∧ execC .fmaxc (flatten ps) (blockLeaf row b) cs = some a' ∧
      eval P (scaleChain f r a').1 = eval P (scaleChain f r a).1 := by
  obtain ⟨a, a', ha, hk, hs, _⟩ := block_outputs_invariant f r row b ps cs hp hc
  exact ⟨a, a', ha, hk, eval_equiv P hs⟩

/-! ## The zero-filling bucket kernel (runtime K and M)

The compiled kernel knows only the bucket; blocks beyond `K / 32` (`b >= nb`) and whole rows beyond `M` are zero-filled
by the predicated loads. A block is either fully present or fully absent (`K % 32 == 0`), so the spec's absent block has
no term and the kernel's reads `fabs(+0)` at every leaf. Under the restricted identity (`+0` Good, `fmaxc` keeps Good
values Good and has `+0` as its identity on Good values, `fabs` is Good) the kernel's block amax equals the spec's bit
for bit for present blocks and is `+0` for absent ones -- the zero scale byte the padded 128x4 layout requires. -/

/-- The spec's leaf of a row with `nb` blocks: blocks `>= nb` absent. -/
def blockLeafSpec (nb : Nat) (row : Nat → Term) (b : Nat) (bs : List Bool) : Option Term :=
  if b < nb then blockLeaf row b bs else none

/-- The kernel's leaf: the zero-extended row (`Support.padRow`), every leaf present. -/
def blockLeafPadded (nb : Nat) (row : Nat → Term) (b : Nat) (bs : List Bool) : Option Term :=
  some (leaf (padRow (32 * nb) (.const "0.0") row) (32 * b + bitsToNat (takeN 5 bs)))

/-- **The bucket kernel's block amax equals the spec's, bit for bit, under the restricted identity.** -/
theorem blockamax_padded_bits {α : Type} (P : Prims α) (Good : α → Prop) (nb : Nat) (row : Nat → Term) (b : Nat)
    (ps : List (List Factor)) (cs : List (List Bool)) (hp : PlacementOK [5] ps) (hc : CoordsOK (flatten ps) cs)
    (z : α) (hz : P.fabs (P.cv "0.0") = z) (hGz : Good z)
    (hcl : ∀ a b, Good a → Good b → Good (P.fmaxc a b))
    (hid : ∀ a, Good a → P.fmaxc z a = a ∧ P.fmaxc a z = a)
    (habs : ∀ x, Good (P.fabs x)) :
    evalO P z (execC .fmaxc (flatten ps) (blockLeafPadded nb row b) cs) = evalO P z (canonicalDigitsC .fmaxc [5] (blockLeafSpec nb row b))
    ∧ Good (evalO P z (canonicalDigitsC .fmaxc [5] (blockLeafSpec nb row b))) := by
  refine execC_pad_good P Good z .fmaxc hGz hcl hid [5] ps (blockLeafPadded nb row b) (blockLeafSpec nb row b) cs hp hc ?_ ?_
  · intro bs
    simp only [blockLeafPadded, blockLeafSpec, blockLeaf, leaf, padRow]
    by_cases hb : b < nb
    · have hi : 32 * b + bitsToNat (takeN 5 bs) < 32 * nb := by
        have := elem_lt_32 bs
        omega
      simp only [hb, hi, if_true, evalO, eval]
    · have hi : ¬ (32 * b + bitsToNat (takeN 5 bs) < 32 * nb) := by omega
      simp only [hb, hi, if_false, evalO, eval]
      exact hz
  · intro bs
    simp only [blockLeafSpec]
    by_cases hb : b < nb
    · simp only [hb, if_true, evalO, blockLeaf, leaf, eval]
      exact habs _
    · simp only [hb, if_false, evalO]
      exact hGz

/-! ## The launch model: R1 / R2 by construction -/

/-- A row's outputs under `(f, r)`: its codes and scale bytes as functions of the row alone. No batch, no position, no
neighbours, no global scale. -/
def rowOut (f : Fmt) (r : Rule) (row : Nat → Term) : (Nat → Option Term) × (Nat → Option Term) :=
  (specCode f r row, specSf f r row)

def launch (M : Nat) (rows : Fin M → Nat → Term) (f : Fmt) (r : Rule) (i : Fin M) := rowOut f r (rows i)

theorem determinism (f : Fmt) (r : Rule) (row : Nat → Term) : rowOut f r row = rowOut f r row := rfl

theorem batch_invariance (f : Fmt) (r : Rule) {M₁ M₂ : Nat} (rows₁ : Fin M₁ → Nat → Term) (rows₂ : Fin M₂ → Nat → Term)
    (r₁ : Fin M₁) (r₂ : Fin M₂) (h : rows₁ r₁ = rows₂ r₂) : launch M₁ rows₁ f r r₁ = launch M₂ rows₂ f r r₂ := by
  simp only [launch, h]

theorem permutation_invariance (f : Fmt) (r : Rule) {M : Nat} (rows : Fin M → Nat → Term) (σ : Fin M → Fin M) (i : Fin M) :
    launch M (rows ∘ σ) f r i = launch M rows f r (σ i) := rfl

theorem subbatch_invariance (f : Fmt) (r : Rule) {M k : Nat} (rows : Fin M → Nat → Term) (sel : Fin k → Fin M) (i : Fin k) :
    launch k (rows ∘ sel) f r i = launch M rows f r (sel i) := rfl

/-- **A block's outputs depend on its 32 inputs alone**: two rows that agree on block `b` give the same scale and codes
for block `b`, whatever the rest of the rows (the per-block form of R2: no per-tensor or per-row scale exists). -/
theorem block_locality (f : Fmt) (r : Rule) (row row' : Nat → Term) (b : Nat)
    (h : ∀ e, e < 32 → row (32 * b + e) = row' (32 * b + e)) :
    specSf f r row b = specSf f r row' b ∧ ∀ e, e < 32 → specCode f r row (32 * b + e) = specCode f r row' (32 * b + e) := by
  have hamax : blockAmax row b = blockAmax row' b := by
    apply treeC_ext
    intro bs
    simp only [blockLeaf, leaf]
    rw [h _ (elem_lt_32 bs)]
  refine ⟨by simp only [specSf, hamax], fun e he => ?_⟩
  have hb : (32 * b + e) / 32 = b := by omega
  simp only [specCode, hb, hamax, h e he]

/-! ## The scale-factor layouts (two placements of the store of one class) -/

/-- `layout.sf_offset_128x4`: FlashInfer's 128x4 swizzle (`compute_sf_index_swizzled_128x4_gpu`). -/
def sfOffset128x4 (row col pcols : Nat) : Nat :=
  (col % 4) + (col / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + (row / 128) * (128 * pcols)

/-- `layout.sf_offset_linear`. -/
def sfOffsetLinear (row col cols : Nat) : Nat := row * cols + col

/-- `covers n l`: `l` has `n` entries and the set of its entries is `{0, ..., n-1}` (the bitmask fold equals `2^n - 1`:
no entry is `>= n`, every offset below `n` occurs; with `n` entries each occurs once). With `l` the offsets of the `n`
cells of a padded grid, the store is a bijection of the grid onto the buffer. The fold runs on the kernel's
GMP-accelerated `Nat` operations, so the instances below decide in the kernel. -/
def covers (n : Nat) (l : List Nat) : Bool :=
  l.length == n && l.foldl (fun acc o => acc ||| (1 <<< o)) 0 == (1 <<< n) - 1

def offsets128x4 (rt ct : Nat) : List Nat :=
  (List.range (128 * rt)).flatMap fun r => (List.range (4 * ct)).map fun c => sfOffset128x4 r c (4 * ct)

def offsetsLinear (rows cols : Nat) : List Nat :=
  (List.range rows).flatMap fun r => (List.range cols).map fun c => sfOffsetLinear r c cols

/-- The 128x4 swizzle is a bijection of the padded grid onto the buffer (one row tile by one column tile; two by two;
three by four): the swizzled store and the linear store write the same bytes to different places. -/
theorem layout128x4_bijective_1x1 : covers (128 * 4) (offsets128x4 1 1) = true := by decide +kernel

theorem layout128x4_bijective_2x2 : covers (256 * 8) (offsets128x4 2 2) = true := by decide +kernel

theorem layout128x4_bijective_3x4 : covers (384 * 16) (offsets128x4 3 4) = true := by decide +kernel

theorem layoutLinear_bijective : covers (37 * 7) (offsetsLinear 37 7) = true := by decide +kernel

/-! ## The UE8M0 conversion (assumption A5)

`ue8m0Byte rnd a e` is the byte the conversion produces for the magnitude `a * 2^e` (`a` a natural, `e` an integer; every
positive binary32 has such a form): with `f = floor(log2(a 2^e)) = log2 a + e`, `rz` takes `f`, `rp` takes `f` when
`a` is a power of two and `f + 1` otherwise, `rn` takes `f + 1` when `a 2^e >= 1.5 x 2^f` (ties up) and `f` otherwise;
the byte is `k + 127` clamped to `[0, 254]` (satfinite: values `<= 2^-127` give 0, `>= 2^127` give 254); the magnitude
0 gives 0. `fp32.to_ue8m0` (Python) and the device conversions are checked against it (the conformance table and the
exhaustive device sweep). -/
/-- `floor(log2 n)` by structural recursion on a fuel (binary32 mantissas are below `2^25`, so 64 steps suffice):
the kernel evaluates it, where `Nat.log2` (well-founded recursion) would not reduce under `decide`. -/
def log2Fuel : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, n => if n < 2 then 0 else log2Fuel fuel (n / 2) + 1

def ue8m0Byte (rnd : Rnd) (a : Nat) (e : Int) : Nat :=
  if a = 0 then 0 else
    let l := log2Fuel 160 a
    let f : Int := (l : Int) + e
    let k : Int := match rnd with
      | .rz => f
      | .rp => if a = 2 ^ l then f else f + 1
      | .rn => if 2 * a ≥ 3 * 2 ^ l then f + 1 else f
    (max (-127 : Int) (min 127 k) + 127).toNat

theorem ue8m0_powers_fixed : ue8m0Byte .rp 1 0 = 127 ∧ ue8m0Byte .rz 1 0 = 127 ∧ ue8m0Byte .rn 1 0 = 127
    ∧ ue8m0Byte .rp 1 (-126) = 1 ∧ ue8m0Byte .rp 1 100 = 227 := by decide

theorem ue8m0_roundings : ue8m0Byte .rp 3 (-1) = 128 ∧ ue8m0Byte .rz 3 (-1) = 127 ∧ ue8m0Byte .rn 3 (-1) = 128
    ∧ ue8m0Byte .rn 5 (-2) = 127 ∧ ue8m0Byte .rp 5 (-2) = 128 ∧ ue8m0Byte .rp 7 0 = 130 ∧ ue8m0Byte .rz 7 0 = 129
    ∧ ue8m0Byte .rp 7 6 = 136 ∧ ue8m0Byte .rz 7 6 = 135 := by decide     -- 448 = 7 x 2^6: ceil 2^9, floor 2^8

theorem ue8m0_saturates : ue8m0Byte .rp 1 (-127) = 0 ∧ ue8m0Byte .rp 1 (-140) = 0 ∧ ue8m0Byte .rp 3 (-129) = 0
    ∧ ue8m0Byte .rn 3 (-128) = 1 ∧ ue8m0Byte .rp 1 127 = 254 ∧ ue8m0Byte .rp 3 126 = 254 ∧ ue8m0Byte .rz 3 127 = 254
    ∧ ue8m0Byte .rp 0 0 = 0 := by decide

end BIK.MXQuant
