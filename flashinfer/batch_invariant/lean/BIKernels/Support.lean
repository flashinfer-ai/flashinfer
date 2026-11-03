import BIKernels.Canon
/-!
# Shared helpers of the op files

Little-endian bit lists as numbers, the prefix of a bit list and the zero-extended row (the kernels' view of a row
beyond its runtime length), with the two bounds `MXQuant.lean` needs. -/
namespace BIK

/-- little-endian bits to a number -/
def bitsToNat : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + 2 * bitsToNat bs

def takeN : Nat → List Bool → List Bool
  | 0, _ => []
  | _+1, [] => []
  | i+1, b :: bs => b :: takeN i bs

/-- The zero-extended row: elements beyond `H` read as the padding value `xz` (`+0` for the quantizers). -/
def padRow (H : Nat) (xz : Term) (row : Nat → Term) : Nat → Term := fun i => if i < H then row i else xz

theorem bitsToNat_lt : ∀ l : List Bool, bitsToNat l < 2 ^ l.length
  | [] => Nat.lt_succ_self 0
  | b :: bs => by
      have ih := bitsToNat_lt bs
      show (if b then 1 else 0) + 2 * bitsToNat bs < 2 ^ (bs.length + 1)
      rw [Nat.pow_succ]
      cases b <;> simp <;> omega

theorem takeN_length_le : ∀ (n : Nat) (l : List Bool), (takeN n l).length ≤ n
  | 0, _ => Nat.le_refl 0
  | _+1, [] => Nat.zero_le _
  | n+1, _ :: bs => Nat.succ_le_succ (takeN_length_le n bs)

end BIK
