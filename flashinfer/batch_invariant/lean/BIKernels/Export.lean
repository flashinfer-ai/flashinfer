import BIKernels.GEMM
/-! Canonical S-expressions, textually identical to `term.sexpr` on the Python side (`flashinfer/batch_invariant/fp_interface/term.py`)
for the GEMM terms. The commutative `add` prints its operands sorted, as Python does. -/
namespace BIK

def sortPair (a b : String) : String × String := if compare a b == .gt then (b, a) else (a, b)

namespace GEMM
def GTerm.sexpr : GTerm → String
  | .zero => "(const 0.0)"
  | .mma16 acc a t b => s!"(mma16 {acc.sexpr} (ab (a {a.id} {t}) (b {t} {b.id})))"
  | .add acc p => let (x, y) := sortPair acc.sexpr p.sexpr; s!"(add {x} {y})"
  | .cvt f x => s!"(cvt {f} {x.sexpr})"
end GEMM

end BIK
