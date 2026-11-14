import BIKernels.Canon
import BIKernels.Export
/-! Canonical S-expressions, textually identical to `term.sexpr` on the Python side (`quantization/qterm.py`, `sampling/sterm.py`; `sortPair` is the GEMM export's).
Commutative nodes (add, mul, fma's first two operands, the commutative max) print their operands sorted. -/
namespace BIK

def Term.sexpr : Term → String
  | .x i => s!"(x {i})"
  | .w i => s!"(w {i})"
  | .r i => s!"(r {i})"
  | .const c => s!"(const {c})"
  | .sym s => s!"(sym {s})"
  | .add a b => let (p, q) := sortPair a.sexpr b.sexpr; s!"(add {p} {q})"
  | .mul a b => let (p, q) := sortPair a.sexpr b.sexpr; s!"(mul {p} {q})"
  | .fma a b c => let (p, q) := sortPair a.sexpr b.sexpr; s!"(fma {p} {q} {c.sexpr})"
  | .div a b => s!"(div {a.sexpr} {b.sexpr})"
  | .sqrt a => s!"(sqrt {a.sexpr})"
  | .cvt f a => s!"(cvt {f} {a.sexpr})"
  | .fabs a => s!"(fabs {a.sexpr})"
  | .fmin a b => s!"(fmin {a.sexpr} {b.sexpr})"
  | .fmaxc a b => let (p, q) := sortPair a.sexpr b.sexpr; s!"(fmax {p} {q})"   -- `term.sexpr` prints the commutative max as (fmax p q), sorted
  | .nzor a b => s!"(nzor {a.sexpr} {b.sexpr})"
  | .neg a => s!"(neg {a.sexpr})"
  | .ex2 a => s!"(ex2 {a.sexpr})"
  | .opq name a => s!"(op:{name} {a.sexpr})"

def osexpr : Option Term → String
  | none => "absent"
  | some t => t.sexpr

/-- Zero lane coordinates of the right lengths for a flattened placement. -/
def zeroCoords : List Factor → List (List Bool)
  | [] => []
  | f :: fs => List.replicate f.bits false :: zeroCoords fs

theorem coordsOK_zero : ∀ fs : List Factor, CoordsOK fs (zeroCoords fs)
  | [] => trivial
  | _ :: fs => ⟨fun _ => List.length_replicate _ _, coordsOK_zero fs⟩

/-- `PlacementOK` is decidable, so generated instances can be closed by `decide`. -/
def PlacementOK.dec : (ds : List Nat) → (ps : List (List Factor)) → Decidable (PlacementOK ds ps)
  | [], [] => isTrue trivial
  | [], _ :: _ => isFalse (fun h => h)
  | _ :: _, [] => isFalse (fun h => h)
  | d :: ds, p :: ps =>
    match PlacementOK.dec ds ps with
    | isTrue h =>
      if hs : sumBits p = d then isTrue ⟨hs, h⟩ else isFalse (fun h' => hs h'.1)
    | isFalse h => isFalse (fun h' => h h'.2)

instance (ds : List Nat) (ps : List (List Factor)) : Decidable (PlacementOK ds ps) := PlacementOK.dec ds ps

end BIK
