import BIKernels.GEMM
/-! Canonical S-expressions, textually identical to `term.sexpr` on the Python side (`flashinfer/batch_invariant/fp_interface/term.py`)
for the GEMM terms. The commutative `add` and `mul` print their operands sorted, as Python does. -/
namespace BIK

def sortPair (a b : String) : String × String := if compare a b == .gt then (b, a) else (a, b)

namespace GEMM
def GTerm.sexpr : GTerm → String
  | .zero => "(const 0.0)"
  | .mma16 acc a t b => s!"(mma16 {acc.sexpr} (ab (a {a.id} {t}) (b {t} {b.id})))"
  | .mma32 acc a t b => s!"(mma32 {acc.sexpr} (ab (a {a.id} {t}) (b {t} {b.id})))"
  | .mma32mx acc a t b => s!"(mma32mx {acc.sexpr} (absf (a {a.id} {t}) (sfa {a.id} {t}) (b {t} {b.id}) (sfb {t} {b.id})))"
  | .mma64mx8 acc a t b => s!"(mma64mx8 {acc.sexpr} (absf (a {a.id} {t}) (sfa {a.id} {t}) (b {t} {b.id}) (sfb {t} {b.id})))"
  | .mma64bs acc a t b => s!"(mma64bs {acc.sexpr} (absf (a {a.id} {t}) (sfa {a.id} {t}) (b {t} {b.id}) (sfb {t} {b.id})))"
  | .mma128bs acc a t b => s!"(mma128bs {acc.sexpr} (absf (a {a.id} {t}) (sfa {a.id} {t}) (b {t} {b.id}) (sfb {t} {b.id})))"
  | .mma64mx acc a t b => s!"(mma64mx {acc.sexpr} (absf (a {a.id} {t}) (sfa {a.id} {t}) (b {t} {b.id}) (sfb {t} {b.id})))"
  | .mma128mx4 acc a t b => s!"(mma128mx4 {acc.sexpr} (absf (a {a.id} {t}) (sfa {a.id} {t}) (b {t} {b.id}) (sfb {t} {b.id})))"
  | .add acc p => let (x, y) := sortPair acc.sexpr p.sexpr; s!"(add {x} {y})"
  | .mul x y => let (p, q) := sortPair x.sexpr y.sexpr; s!"(mul {p} {q})"
  | .fma x y acc => let (p, q) := sortPair x.sexpr y.sexpr; s!"(fma {p} {q} {acc.sexpr})"    -- the product's operands sorted, as Python does
  | .sym s => s!"(sym {s})"
  | .cvt f x => s!"(cvt {f} {x.sexpr})"
  | .bias n => s!"(bias {n.id})"
  | .sfaBlk a kb => s!"(sfa_blk {a.id} {kb})"
  | .sfbBlk b kb => s!"(sfb_blk {b.id / 128} {kb})"    -- the weight scale block of the column: `spec.block_scale_term` writes n // SCALE_BLOCK_N
end GEMM

end BIK
