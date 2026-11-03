/-!
# The canon engine, formalised

This file mirrors `canon.py` and proves the unbounded version of what
`canon.verify` checks for individual sizes.

* `Term`   : uninterpreted dataflow terms, one constructor per rounded primitive.
* `Equiv`  : equality of terms up to commutativity of `add` / `mul` (IEEE add/mul are
             bitwise commutative); a congruence.
* `tree`   : the adjacent-pairs-first binary reduction over a bit-indexed leaf
             function (`canon._pairwise`), with `none` for an absent leaf.
* `bfly`   : the lane butterfly (xor-partner exchange over ascending bits), i.e.
             `canon.apply_step` at level `lane`.
* `exec`   : what one unit computes when the plan's factors are processed in order:
             `reg` and `xchg` factors reduce the factor with `tree` (in registers /
             through an exchange), `lane` factors with the butterfly.
* `exec_eq_tree` : every unit ends with the canonical tree, for every placement.
* `config_invariance`, `units_agree` : the statements used by the RL guarantee.

Model-to-code correspondence (not provable here): the kernel's reduction is the same
`apply_step` sequence executed with hardware peers; only loads, shuffles, barriers and
predicates are outside this model and are covered by the GPU tests.
-/
namespace BIK

inductive Term where
  | x (i : Nat)
  | w (i : Nat)
  | r (i : Nat)
  | const (c : String)
  | sym (s : String)
  | add (a b : Term)
  | mul (a b : Term)
  | fma (a b c : Term)
  | div (a b : Term)
  | sqrt (a : Term)
  | cvt (fmt : String) (a : Term)
  -- exact primitives (`fp/term.py`): `fabs`, the order-sensitive `fmin`, the commutativity-normalised max of
  -- non-negative values `fmaxc`; the model keeps `sqrt`, `sym` and the select `nzor` as constructors
  | fabs (a : Term)
  | fmin (a b : Term)
  | fmaxc (a b : Term)
  | nzor (a b : Term)
  deriving Repr, DecidableEq, Inhabited

/-- Term equality up to commutativity of the commutative primitives. -/
inductive Equiv : Term → Term → Prop where
  | refl (a : Term) : Equiv a a
  | symm {a b : Term} : Equiv a b → Equiv b a
  | trans {a b c : Term} : Equiv a b → Equiv b c → Equiv a c
  | add_comm (a b : Term) : Equiv (.add a b) (.add b a)
  | mul_comm (a b : Term) : Equiv (.mul a b) (.mul b a)
  | add_congr {a a' b b' : Term} : Equiv a a' → Equiv b b' → Equiv (.add a b) (.add a' b')
  | mul_congr {a a' b b' : Term} : Equiv a a' → Equiv b b' → Equiv (.mul a b) (.mul a' b')
  | fma_congr {a a' b b' c c' : Term} :
      Equiv a a' → Equiv b b' → Equiv c c' → Equiv (.fma a b c) (.fma a' b' c')
  | div_congr {a a' b b' : Term} : Equiv a a' → Equiv b b' → Equiv (.div a b) (.div a' b')
  | sqrt_congr {a a' : Term} : Equiv a a' → Equiv (.sqrt a) (.sqrt a')
  | cvt_congr {a a' : Term} (fmt : String) : Equiv a a' → Equiv (.cvt fmt a) (.cvt fmt a')
  | fmaxc_comm (a b : Term) : Equiv (.fmaxc a b) (.fmaxc b a)
  | fabs_congr {a a' : Term} : Equiv a a' → Equiv (.fabs a) (.fabs a')
  | fmin_congr {a a' b b' : Term} : Equiv a a' → Equiv b b' → Equiv (.fmin a b) (.fmin a' b')
  | fmaxc_congr {a a' b b' : Term} : Equiv a a' → Equiv b b' → Equiv (.fmaxc a b) (.fmaxc a' b')
  | nzor_congr {a a' b b' : Term} : Equiv a a' → Equiv b b' → Equiv (.nzor a b) (.nzor a' b')

/-- `none` is an absent leaf (dropped from the tree), as in `canon._pairwise`. -/
def oadd : Option Term → Option Term → Option Term
  | none, b => b
  | some a, none => some a
  | some a, some b => some (.add a b)

inductive OEquiv : Option Term → Option Term → Prop where
  | absent : OEquiv none none
  | present {a b : Term} : Equiv a b → OEquiv (some a) (some b)

theorem OEquiv.refl : ∀ o : Option Term, OEquiv o o
  | Option.none => .absent
  | Option.some a => .present (.refl a)

theorem OEquiv.symm : ∀ {o p : Option Term}, OEquiv o p → OEquiv p o
  | Option.none, Option.none, .absent => .absent
  | Option.some _, Option.some _, .present h => .present (.symm h)

theorem OEquiv.trans : ∀ {o p q : Option Term}, OEquiv o p → OEquiv p q → OEquiv o q
  | Option.none, Option.none, Option.none, .absent, .absent => .absent
  | Option.some _, Option.some _, Option.some _, .present h₁, .present h₂ => .present (.trans h₁ h₂)

theorem OEquiv.some_right : ∀ {o : Option Term} {b : Term}, OEquiv o (Option.some b) → ∃ a, o = Option.some a ∧ Equiv a b
  | Option.some _, _, .present h => ⟨_, rfl, h⟩

theorem oadd_congr : ∀ {a a' b b' : Option Term}, OEquiv a a' → OEquiv b b' → OEquiv (oadd a b) (oadd a' b')
  | none, none, none, none, .absent, .absent => .absent
  | none, none, some _, some _, .absent, .present h => .present h
  | some _, some _, none, none, .present h, .absent => .present h
  | some _, some _, some _, some _, .present h, .present h' => .present (.add_congr h h')

theorem oadd_comm : ∀ a b : Option Term, OEquiv (oadd a b) (oadd b a)
  | none, none => .absent
  | none, some b => .present (.refl b)
  | some a, none => .present (.refl a)
  | some a, some b => .present (.add_comm a b)

/-! ## Trees over bit-indexed leaves

Leaves are indexed by little-endian bit lists (bit 0 is combined first). `tree n f`
splits on the highest bit at the top, which is exactly "adjacent pairs first" at the
bottom: `tree 2 f = oadd (oadd (f [0,0]) (f [1,0])) (oadd (f [0,1]) (f [1,1]))`. -/
def tree : Nat → (List Bool → Option Term) → Option Term
  | 0, f => f []
  | n+1, f => oadd (tree n (fun bs => f (bs ++ [false]))) (tree n (fun bs => f (bs ++ [true])))

theorem tree_congr : ∀ (n : Nat) {f g : List Bool → Option Term},
    (∀ bs, OEquiv (f bs) (g bs)) → OEquiv (tree n f) (tree n g)
  | 0, _, _, h => h []
  | n+1, _, _, h => oadd_congr (tree_congr n fun _ => h _) (tree_congr n fun _ => h _)

theorem tree_ext (n : Nat) {f g : List Bool → Option Term} (h : ∀ bs, f bs = g bs) : tree n f = tree n g := by
  rw [show f = g from funext h]

/-- The key lemma: a tree over `n + m` bits is the tree over the high `m` bits of the
trees over the low `n` bits. Every digit factorisation of the engine is an instance. -/
theorem tree_split (n : Nat) : ∀ (m : Nat) (f : List Bool → Option Term),
    tree (n + m) f = tree m (fun hi => tree n (fun lo => f (lo ++ hi)))
  | 0, f => by
      show tree n f = tree n (fun lo => f (lo ++ []))
      exact tree_ext n (fun lo => by rw [List.append_nil])
  | m+1, f => by
      show tree (n + m + 1) f =
        oadd (tree m (fun hi => tree n (fun lo => f (lo ++ (hi ++ [false])))))
             (tree m (fun hi => tree n (fun lo => f (lo ++ (hi ++ [true])))))
      rw [tree, tree_split n m (fun bs => f (bs ++ [false])), tree_split n m (fun bs => f (bs ++ [true]))]
      simp only [List.append_assoc]

/-! ## The lane butterfly -/
def getB : Nat → List Bool → Bool
  | _, [] => false
  | 0, b :: _ => b
  | i+1, _ :: bs => getB i bs

def dropN : Nat → List Bool → List Bool
  | 0, l => l
  | _+1, [] => []
  | i+1, _ :: bs => dropN i bs

/-- Toggle bit `i` (the xor partner of a lane in the butterfly). -/
def flipBit : Nat → List Bool → List Bool
  | _, [] => []
  | 0, b :: bs => (!b) :: bs
  | i+1, b :: bs => b :: flipBit i bs

theorem length_flipBit : ∀ (i : Nat) (v : List Bool), (flipBit i v).length = v.length
  | 0, [] => rfl
  | _+1, [] => rfl
  | 0, _ :: _ => rfl
  | i+1, _ :: bs => congrArg (· + 1) (length_flipBit i bs)

theorem dropN_length : ∀ (v : List Bool), dropN v.length v = []
  | [] => rfl
  | _ :: bs => dropN_length bs

theorem dropN_of_length_eq {v : List Bool} {b : Nat} (h : v.length = b) : dropN b v = [] :=
  h ▸ dropN_length v

theorem dropN_eq_cons : ∀ (i : Nat) (v : List Bool), i < v.length → dropN i v = getB i v :: dropN (i+1) v
  | 0, [], h => absurd h (Nat.lt_irrefl 0)
  | _+1, [], h => absurd h (Nat.not_lt_zero _)
  | 0, _ :: _, _ => rfl
  | i+1, _ :: bs, h => dropN_eq_cons i bs (Nat.lt_of_succ_lt_succ h)

theorem dropN_flipBit_eq : ∀ (i : Nat) (v : List Bool), i < v.length →
    dropN i (flipBit i v) = (!getB i v) :: dropN (i+1) v
  | 0, [], h => absurd h (Nat.lt_irrefl 0)
  | _+1, [], h => absurd h (Nat.not_lt_zero _)
  | 0, _ :: _, _ => rfl
  | i+1, _ :: bs, h => dropN_flipBit_eq i bs (Nat.lt_of_succ_lt_succ h)

/-- `bfly i val v`: the value lane `v` holds after butterfly micro-steps 0..i-1, when
every lane `u` started with `val u` (`canon.apply_step`, level `lane`). -/
def bfly : Nat → (List Bool → Option Term) → List Bool → Option Term
  | 0, val, v => val v
  | i+1, val, v => oadd (bfly i val v) (bfly i val (flipBit i v))

/-- After `i` micro-steps every lane holds the tree over its aligned group of `2^i`
lanes, up to commutativity. The butterfly computes the canonical tree in every lane. -/
theorem bfly_tree : ∀ (i : Nat) (val : List Bool → Option Term) (v : List Bool), i ≤ v.length →
    OEquiv (bfly i val v) (tree i (fun lo => val (lo ++ dropN i v)))
  | 0, val, v, _ => by
      show OEquiv (val v) (val ([] ++ v))
      exact OEquiv.refl _
  | i+1, val, v, h => by
      have hi : i < v.length := h
      have A := bfly_tree i val v (Nat.le_of_lt hi)
      have B := bfly_tree i val (flipBit i v) (by rw [length_flipBit]; exact Nat.le_of_lt hi)
      rw [dropN_eq_cons i v hi] at A
      rw [dropN_flipBit_eq i v hi] at B
      show OEquiv (oadd (bfly i val v) (bfly i val (flipBit i v)))
                  (oadd (tree i (fun bs => val ((bs ++ [false]) ++ dropN (i+1) v)))
                        (tree i (fun bs => val ((bs ++ [true]) ++ dropN (i+1) v))))
      have e1 : ∀ bs : List Bool, val ((bs ++ [false]) ++ dropN (i+1) v) = val (bs ++ (false :: dropN (i+1) v)) :=
        fun bs => by rw [List.append_assoc]; rfl
      have e2 : ∀ bs : List Bool, val ((bs ++ [true]) ++ dropN (i+1) v) = val (bs ++ (true :: dropN (i+1) v)) :=
        fun bs => by rw [List.append_assoc]; rfl
      rw [tree_ext i e1, tree_ext i e2]
      cases hb : getB i v
      · rw [hb] at A B
        simp only [Bool.not_false] at B
        exact oadd_congr A B
      · rw [hb] at A B
        simp only [Bool.not_true] at B
        exact (oadd_congr A B).trans (oadd_comm _ _)

/-! ## Plans: digits factored into placed factors, executed in combine order -/

/-- Where a factor lives (`canon.LEVELS`; `warp`, `cta`, `grid` all exchange). -/
inductive Kind where
  | reg
  | lane
  | xchg
  deriving DecidableEq, Repr

structure Factor where
  bits : Nat
  kind : Kind
  deriving Repr

def sumBits : List Factor → Nat
  | [] => 0
  | f :: fs => f.bits + sumBits fs

/-- Reducing one factor of `b` bits in registers or through an exchange: the leaf
function over the remaining bits becomes the tree over the factor. -/
def shrink (b : Nat) (leaf : List Bool → Option Term) : List Bool → Option Term :=
  fun rest => tree b (fun v => leaf (v ++ rest))

/-- Reducing a lane factor: this unit (lane bits `c`) holds the butterfly result. -/
def shrinkLane (b : Nat) (leaf : List Bool → Option Term) (c : List Bool) : List Bool → Option Term :=
  fun rest => bfly b (fun v => leaf (v ++ rest)) c

/-- What a unit with per-factor coordinates `cs` computes from `leaf` by processing
the factors in order (`canon.init_state` + `canon.run`). Coordinates of `reg` and
`xchg` factors are ignored: registers hold every value, an exchange gathers them. -/
def exec : List Factor → (List Bool → Option Term) → List (List Bool) → Option Term
  | [], leaf, _ => leaf []
  | f :: fs, leaf, cs =>
    match f.kind with
    | .lane => exec fs (shrinkLane f.bits leaf (cs.headD [])) cs.tail
    | _ => exec fs (shrink f.bits leaf) cs.tail

/-- A unit's lane coordinate for a lane factor has exactly that factor's bits. -/
def CoordsOK : List Factor → List (List Bool) → Prop
  | [], _ => True
  | f :: fs, cs => (f.kind = .lane → (cs.headD []).length = f.bits) ∧ CoordsOK fs cs.tail

/-- **Every unit computes the canonical tree**, whatever the factorisation and the
assignment of factors to registers, lanes or exchanges. -/
theorem exec_eq_tree : ∀ (fs : List Factor) (leaf : List Bool → Option Term) (cs : List (List Bool)),
    CoordsOK fs cs → OEquiv (exec fs leaf cs) (tree (sumBits fs) leaf)
  | [], _, _, _ => OEquiv.refl _
  | f :: fs, leaf, cs, h => by
      have h1 := h.1
      have h2 := h.2
      have split : tree (sumBits (f :: fs)) leaf = tree (sumBits fs) (shrink f.bits leaf) := by
        show tree (f.bits + sumBits fs) leaf = tree (sumBits fs) (fun rest => tree f.bits (fun v => leaf (v ++ rest)))
        exact tree_split f.bits (sumBits fs) leaf
      rw [split]
      cases hk : f.kind with
      | lane =>
          have hc : (cs.headD []).length = f.bits := h1 hk
          simp only [exec, hk]
          refine OEquiv.trans (exec_eq_tree fs _ cs.tail h2) ?_
          apply tree_congr
          intro rest
          have hb := bfly_tree f.bits (fun v => leaf (v ++ rest)) (cs.headD []) (by rw [hc]; exact Nat.le_refl _)
          rw [dropN_of_length_eq hc] at hb
          have e : ∀ lo : List Bool, (fun v => leaf (v ++ rest)) (lo ++ []) = leaf (lo ++ rest) :=
            fun lo => by rw [List.append_nil]
          rw [tree_ext f.bits e] at hb
          exact hb
      | reg =>
          simp only [exec, hk]
          exact exec_eq_tree fs _ cs.tail h2
      | xchg =>
          simp only [exec, hk]
          exact exec_eq_tree fs _ cs.tail h2

/-- **All units agree** (the result does not depend on which lane / warp / CTA you ask). -/
theorem units_agree (fs : List Factor) (leaf : List Bool → Option Term) (cs₁ cs₂ : List (List Bool))
    (h₁ : CoordsOK fs cs₁) (h₂ : CoordsOK fs cs₂) : OEquiv (exec fs leaf cs₁) (exec fs leaf cs₂) :=
  (exec_eq_tree fs leaf cs₁ h₁).trans (exec_eq_tree fs leaf cs₂ h₂).symm

/-! ## The spec side: digits -/

/-- `Reduction.canonical`: reduce digit 0 innermost, the last digit outermost. -/
def canonicalDigits : List Nat → (List Bool → Option Term) → Option Term
  | [], leaf => leaf []
  | d :: ds, leaf => canonicalDigits ds (shrink d leaf)

def sumNat : List Nat → Nat
  | [] => 0
  | d :: ds => d + sumNat ds

theorem canonicalDigits_eq_tree : ∀ (ds : List Nat) (leaf : List Bool → Option Term),
    canonicalDigits ds leaf = tree (sumNat ds) leaf
  | [], _ => rfl
  | d :: ds, leaf => by
      show canonicalDigits ds (shrink d leaf) = tree (d + sumNat ds) leaf
      rw [canonicalDigits_eq_tree ds, tree_split]
      rfl

/-- `canon.Placement`: per digit, its factors innermost first; `flatten` is the plan. -/
def flatten : List (List Factor) → List Factor
  | [] => []
  | p :: ps => p ++ flatten ps

def PlacementOK : List Nat → List (List Factor) → Prop
  | [], [] => True
  | d :: ds, p :: ps => sumBits p = d ∧ PlacementOK ds ps
  | _, _ => False

theorem sumBits_append : ∀ (a b : List Factor), sumBits (a ++ b) = sumBits a + sumBits b
  | [], b => (Nat.zero_add _).symm
  | f :: a, b => by
      show f.bits + sumBits (a ++ b) = f.bits + sumBits a + sumBits b
      rw [sumBits_append a b, Nat.add_assoc]

theorem sumBits_flatten : ∀ (ds : List Nat) (ps : List (List Factor)), PlacementOK ds ps → sumBits (flatten ps) = sumNat ds
  | [], [], _ => rfl
  | d :: ds, p :: ps, h => by
      show sumBits (p ++ flatten ps) = d + sumNat ds
      rw [sumBits_append, h.1, sumBits_flatten ds ps h.2]
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim

/-- **Config invariance (R3), unbounded.** For every digit list, every valid
placement and every unit, the executed plan equals the canonical reduction. -/
theorem config_invariance (ds : List Nat) (ps : List (List Factor)) (leaf : List Bool → Option Term)
    (cs : List (List Bool)) (hp : PlacementOK ds ps) (hc : CoordsOK (flatten ps) cs) :
    OEquiv (exec (flatten ps) leaf cs) (canonicalDigits ds leaf) := by
  rw [canonicalDigits_eq_tree, ← sumBits_flatten ds ps hp]
  exact exec_eq_tree _ _ _ hc

/-- **Dispatch irrelevance.** Two valid placements (any configs) agree. -/
theorem dispatch_irrelevant (ds : List Nat) (ps₁ ps₂ : List (List Factor)) (leaf : List Bool → Option Term)
    (cs₁ cs₂ : List (List Bool)) (hp₁ : PlacementOK ds ps₁) (hp₂ : PlacementOK ds ps₂)
    (hc₁ : CoordsOK (flatten ps₁) cs₁) (hc₂ : CoordsOK (flatten ps₂) cs₂) :
    OEquiv (exec (flatten ps₁) leaf cs₁) (exec (flatten ps₂) leaf cs₂) :=
  (config_invariance ds ps₁ leaf cs₁ hp₁ hc₁).trans (config_invariance ds ps₂ leaf cs₂ hp₂ hc₂).symm

/-! ## The engine for any bitwise-commutative combine

`canon.Digit` accepts any `pairwise` combine; a sum uses `add`, the MX quantizer's amax
digits use `fmaxc` (the max of non-negative values, bitwise commutative). `Comb` names the combines
the Lean model knows; `treeC`, `bflyC`, `execC` are `tree`, `bfly`, `exec` with the combine as a
parameter, and the theorems below are the same proofs. `treeC .add` is `tree` (`treeC_add`), so the
sum-based statements are instances. -/

inductive Comb where
  | add
  | fmaxc
  deriving DecidableEq, Repr

def Comb.app : Comb → Term → Term → Term
  | .add, a, b => .add a b
  | .fmaxc, a, b => .fmaxc a b

theorem Comb.app_comm (c : Comb) (a b : Term) : Equiv (c.app a b) (c.app b a) := by
  cases c
  · exact .add_comm a b
  · exact .fmaxc_comm a b

theorem Comb.app_congr (c : Comb) {a a' b b' : Term} (h₁ : Equiv a a') (h₂ : Equiv b b') : Equiv (c.app a b) (c.app a' b') := by
  cases c
  · exact .add_congr h₁ h₂
  · exact .fmaxc_congr h₁ h₂

def ocomb (c : Comb) : Option Term → Option Term → Option Term
  | none, b => b
  | some a, none => some a
  | some a, some b => some (c.app a b)

theorem ocomb_add (a b : Option Term) : ocomb .add a b = oadd a b := by
  cases a <;> cases b <;> rfl

theorem ocomb_congr (c : Comb) : ∀ {a a' b b' : Option Term}, OEquiv a a' → OEquiv b b' → OEquiv (ocomb c a b) (ocomb c a' b')
  | none, none, none, none, .absent, .absent => .absent
  | none, none, some _, some _, .absent, .present h => .present h
  | some _, some _, none, none, .present h, .absent => .present h
  | some _, some _, some _, some _, .present h, .present h' => .present (c.app_congr h h')

theorem ocomb_comm (c : Comb) : ∀ a b : Option Term, OEquiv (ocomb c a b) (ocomb c b a)
  | none, none => .absent
  | none, some b => .present (.refl b)
  | some a, none => .present (.refl a)
  | some a, some b => .present (c.app_comm a b)

def treeC (c : Comb) : Nat → (List Bool → Option Term) → Option Term
  | 0, f => f []
  | n+1, f => ocomb c (treeC c n (fun bs => f (bs ++ [false]))) (treeC c n (fun bs => f (bs ++ [true])))

theorem treeC_add : ∀ (n : Nat) (f : List Bool → Option Term), treeC .add n f = tree n f
  | 0, _ => rfl
  | n+1, f => by
      show ocomb .add (treeC .add n _) (treeC .add n _) = oadd (tree n _) (tree n _)
      rw [ocomb_add, treeC_add n, treeC_add n]

theorem treeC_congr (c : Comb) : ∀ (n : Nat) {f g : List Bool → Option Term},
    (∀ bs, OEquiv (f bs) (g bs)) → OEquiv (treeC c n f) (treeC c n g)
  | 0, _, _, h => h []
  | n+1, _, _, h => ocomb_congr c (treeC_congr c n fun _ => h _) (treeC_congr c n fun _ => h _)

theorem treeC_ext (c : Comb) (n : Nat) {f g : List Bool → Option Term} (h : ∀ bs, f bs = g bs) : treeC c n f = treeC c n g := by
  rw [show f = g from funext h]

theorem treeC_split (c : Comb) (n : Nat) : ∀ (m : Nat) (f : List Bool → Option Term),
    treeC c (n + m) f = treeC c m (fun hi => treeC c n (fun lo => f (lo ++ hi)))
  | 0, f => by
      show treeC c n f = treeC c n (fun lo => f (lo ++ []))
      exact treeC_ext c n (fun lo => by rw [List.append_nil])
  | m+1, f => by
      show treeC c (n + m + 1) f =
        ocomb c (treeC c m (fun hi => treeC c n (fun lo => f (lo ++ (hi ++ [false])))))
                (treeC c m (fun hi => treeC c n (fun lo => f (lo ++ (hi ++ [true])))))
      rw [treeC, treeC_split c n m (fun bs => f (bs ++ [false])), treeC_split c n m (fun bs => f (bs ++ [true]))]
      simp only [List.append_assoc]

def bflyC (c : Comb) : Nat → (List Bool → Option Term) → List Bool → Option Term
  | 0, val, v => val v
  | i+1, val, v => ocomb c (bflyC c i val v) (bflyC c i val (flipBit i v))

theorem bflyC_tree (c : Comb) : ∀ (i : Nat) (val : List Bool → Option Term) (v : List Bool), i ≤ v.length →
    OEquiv (bflyC c i val v) (treeC c i (fun lo => val (lo ++ dropN i v)))
  | 0, val, v, _ => by
      show OEquiv (val v) (val ([] ++ v))
      exact OEquiv.refl _
  | i+1, val, v, h => by
      have hi : i < v.length := h
      have A := bflyC_tree c i val v (Nat.le_of_lt hi)
      have B := bflyC_tree c i val (flipBit i v) (by rw [length_flipBit]; exact Nat.le_of_lt hi)
      rw [dropN_eq_cons i v hi] at A
      rw [dropN_flipBit_eq i v hi] at B
      show OEquiv (ocomb c (bflyC c i val v) (bflyC c i val (flipBit i v)))
                  (ocomb c (treeC c i (fun bs => val ((bs ++ [false]) ++ dropN (i+1) v)))
                           (treeC c i (fun bs => val ((bs ++ [true]) ++ dropN (i+1) v))))
      have e1 : ∀ bs : List Bool, val ((bs ++ [false]) ++ dropN (i+1) v) = val (bs ++ (false :: dropN (i+1) v)) :=
        fun bs => by rw [List.append_assoc]; rfl
      have e2 : ∀ bs : List Bool, val ((bs ++ [true]) ++ dropN (i+1) v) = val (bs ++ (true :: dropN (i+1) v)) :=
        fun bs => by rw [List.append_assoc]; rfl
      rw [treeC_ext c i e1, treeC_ext c i e2]
      cases hb : getB i v
      · rw [hb] at A B
        simp only [Bool.not_false] at B
        exact ocomb_congr c A B
      · rw [hb] at A B
        simp only [Bool.not_true] at B
        exact (ocomb_congr c A B).trans (ocomb_comm c _ _)

def shrinkC (c : Comb) (b : Nat) (leaf : List Bool → Option Term) : List Bool → Option Term :=
  fun rest => treeC c b (fun v => leaf (v ++ rest))

def shrinkLaneC (c : Comb) (b : Nat) (leaf : List Bool → Option Term) (cs : List Bool) : List Bool → Option Term :=
  fun rest => bflyC c b (fun v => leaf (v ++ rest)) cs

/-- `exec` for the combine `c`: what one unit computes when the plan's factors are processed in order. -/
def execC (c : Comb) : List Factor → (List Bool → Option Term) → List (List Bool) → Option Term
  | [], leaf, _ => leaf []
  | f :: fs, leaf, cs =>
    match f.kind with
    | .lane => execC c fs (shrinkLaneC c f.bits leaf (cs.headD [])) cs.tail
    | _ => execC c fs (shrinkC c f.bits leaf) cs.tail

theorem execC_eq_treeC (c : Comb) : ∀ (fs : List Factor) (leaf : List Bool → Option Term) (cs : List (List Bool)),
    CoordsOK fs cs → OEquiv (execC c fs leaf cs) (treeC c (sumBits fs) leaf)
  | [], _, _, _ => OEquiv.refl _
  | f :: fs, leaf, cs, h => by
      have h1 := h.1
      have h2 := h.2
      have split : treeC c (sumBits (f :: fs)) leaf = treeC c (sumBits fs) (shrinkC c f.bits leaf) := by
        show treeC c (f.bits + sumBits fs) leaf = treeC c (sumBits fs) (fun rest => treeC c f.bits (fun v => leaf (v ++ rest)))
        exact treeC_split c f.bits (sumBits fs) leaf
      rw [split]
      cases hk : f.kind with
      | lane =>
          have hc : (cs.headD []).length = f.bits := h1 hk
          simp only [execC, hk]
          refine OEquiv.trans (execC_eq_treeC c fs _ cs.tail h2) ?_
          apply treeC_congr
          intro rest
          have hb := bflyC_tree c f.bits (fun v => leaf (v ++ rest)) (cs.headD []) (by rw [hc]; exact Nat.le_refl _)
          rw [dropN_of_length_eq hc] at hb
          have e : ∀ lo : List Bool, (fun v => leaf (v ++ rest)) (lo ++ []) = leaf (lo ++ rest) :=
            fun lo => by rw [List.append_nil]
          rw [treeC_ext c f.bits e] at hb
          exact hb
      | reg =>
          simp only [execC, hk]
          exact execC_eq_treeC c fs _ cs.tail h2
      | xchg =>
          simp only [execC, hk]
          exact execC_eq_treeC c fs _ cs.tail h2

theorem unitsC_agree (c : Comb) (fs : List Factor) (leaf : List Bool → Option Term) (cs₁ cs₂ : List (List Bool))
    (h₁ : CoordsOK fs cs₁) (h₂ : CoordsOK fs cs₂) : OEquiv (execC c fs leaf cs₁) (execC c fs leaf cs₂) :=
  (execC_eq_treeC c fs leaf cs₁ h₁).trans (execC_eq_treeC c fs leaf cs₂ h₂).symm

def canonicalDigitsC (c : Comb) : List Nat → (List Bool → Option Term) → Option Term
  | [], leaf => leaf []
  | d :: ds, leaf => canonicalDigitsC c ds (shrinkC c d leaf)

theorem canonicalDigitsC_eq_treeC (c : Comb) : ∀ (ds : List Nat) (leaf : List Bool → Option Term),
    canonicalDigitsC c ds leaf = treeC c (sumNat ds) leaf
  | [], _ => rfl
  | d :: ds, leaf => by
      show canonicalDigitsC c ds (shrinkC c d leaf) = treeC c (d + sumNat ds) leaf
      rw [canonicalDigitsC_eq_treeC c ds, treeC_split]
      rfl

/-- **Config invariance (R3) for any commutative combine**: every valid placement and unit computes the
canonical reduction. -/
theorem config_invarianceC (c : Comb) (ds : List Nat) (ps : List (List Factor)) (leaf : List Bool → Option Term)
    (cs : List (List Bool)) (hp : PlacementOK ds ps) (hc : CoordsOK (flatten ps) cs) :
    OEquiv (execC c (flatten ps) leaf cs) (canonicalDigitsC c ds leaf) := by
  rw [canonicalDigitsC_eq_treeC, ← sumBits_flatten ds ps hp]
  exact execC_eq_treeC c _ _ _ hc

theorem dispatch_irrelevantC (c : Comb) (ds : List Nat) (ps₁ ps₂ : List (List Factor)) (leaf : List Bool → Option Term)
    (cs₁ cs₂ : List (List Bool)) (hp₁ : PlacementOK ds ps₁) (hp₂ : PlacementOK ds ps₂)
    (hc₁ : CoordsOK (flatten ps₁) cs₁) (hc₂ : CoordsOK (flatten ps₂) cs₂) :
    OEquiv (execC c (flatten ps₁) leaf cs₁) (execC c (flatten ps₂) leaf cs₂) :=
  (config_invarianceC c ds ps₁ leaf cs₁ hp₁ hc₁).trans (config_invarianceC c ds ps₂ leaf cs₂ hp₂ hc₂).symm

end BIK
