-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0

/-!
# KDA: the class `kda.bf16.c16` and the replay decode

Mirrors `flashinfer/batch_invariant/kda/spec.py` (the arithmetic) and `pool.py` (the decode step). The kernel is
hand-written, so it is opaque here: a chunk step `step : S → List R → S × List R` takes the state at a chunk boundary
and a chunk's rows (at most 16) to the state after them and their output rows. The hypotheses the model takes about it
are the hardware hypotheses the GPU tests check bit for bit: the step is deterministic and a function of its own sequence
alone (so a packed call is the per-sequence terms, `batch_term`), and it is causal: the outputs of a chunk prefix are
the prefix of the chunk's outputs (`decode_eq_prefill` takes this as `hcausal`).

What is proved: the prefill of a sequence is the fold of `step` over its 16-row chunks from the initial state
(`stateAfter`, the states at the boundaries); the replay decode of position `p` (the step from the boundary state at
`16 * (p / 16)` over the rows since it and the new row) gives the prefill's row `p`; the pool's commit rule (a slot's
boundary state advances when its 16th row arrives) carries exactly the prefill's boundary states; the positional
arithmetic of `spec.py`, including the count and record rule of the multi-token verify. No configuration space is modelled (R3 is not claimed for this class). The generated
`KdaCases.lean` pins the arithmetic on the Python specification's cases by kernel `decide`.

The shipped decode step does not re-run the open chunk: it is a kernel that computes row `p` of that replay from the
boundary state and cached rows. That it returns the replay's row is a hardware hypothesis like the ones above, checked
bit for bit on the device; the statements here are about the replay it places.
-/
namespace BIK.Kda

universe u v
variable {S : Type u} {R : Type v}

/-! ## List helpers (self-contained: `takeN`, `dropN`, `nth`) -/

def takeN : Nat → List R → List R
  | 0, _ => []
  | _+1, [] => []
  | n+1, x :: xs => x :: takeN n xs

def dropN : Nat → List R → List R
  | 0, xs => xs
  | _+1, [] => []
  | n+1, _ :: xs => dropN n xs

def nth : List R → Nat → Option R
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: xs, n+1 => nth xs n

theorem takeN_takeN : ∀ (n m : Nat) (xs : List R), n ≤ m → takeN n (takeN m xs) = takeN n xs
  | 0, _, _, _ => rfl
  | n+1, 0, _, h => absurd h (by omega)
  | _+1, _+1, [], _ => rfl
  | n+1, m+1, x :: xs, h => by
      simp only [takeN]
      rw [takeN_takeN n m xs (Nat.le_of_succ_le_succ h)]

theorem nth_takeN : ∀ (n i : Nat) (xs : List R), i < n → nth (takeN n xs) i = nth xs i
  | 0, _, _, h => absurd h (by omega)
  | _+1, _, [], _ => by simp only [takeN]
  | _+1, 0, _ :: _, _ => rfl
  | n+1, i+1, _ :: xs, h => by
      simp only [takeN, nth]
      exact nth_takeN n i xs (Nat.lt_of_succ_lt_succ h)

theorem nth_map {α : Type u} {β : Type v} (f : α → β) : ∀ (l : List α) (i : Nat), nth (l.map f) i = (nth l i).map f
  | [], _ => rfl
  | _ :: _, 0 => rfl
  | _ :: l, i+1 => nth_map f l i

/-! ## The positional arithmetic (`spec.py`: `boundary`, `history_len`, `commits`, `next_count`) -/

def boundary (p : Nat) : Nat := 16 * (p / 16)
def historyLen (p : Nat) : Nat := p % 16
def commits (count : Nat) : Bool := count + 1 == 16
def nextCount (count : Nat) : Nat := (count + 1) % 16

theorem boundary_add_historyLen (p : Nat) : boundary p + historyLen p = p := by
  unfold boundary historyLen; omega

theorem historyLen_lt (p : Nat) : historyLen p < 16 := by
  unfold historyLen; omega

theorem commits_iff (count : Nat) : commits count = true ↔ count + 1 = 16 := by
  simp [commits]

theorem nextCount_of_commit (count : Nat) (h : commits count = true) : nextCount count = 0 := by
  have := (commits_iff count).mp h
  unfold nextCount; omega

/-! ## Prefill as the chunk fold, and the replay decode -/

/-- Chunk `k` of a sequence: rows `[16 k, 16 k + 16)` (shorter at the end). -/
def chunkOf (xs : List R) (k : Nat) : List R := takeN 16 (dropN (16 * k) xs)

/-- The state at boundary `k` (after `k` whole chunks): the fp32 state the pool slot holds. -/
def stateAfter (step : S → List R → S × List R) (s0 : S) (xs : List R) : Nat → S
  | 0 => s0
  | k+1 => (step (stateAfter step s0 xs k) (chunkOf xs k)).1

/-- The prefill's output row `p`: row `p % 16` of the chunk `p / 16`, stepped from its boundary state. -/
def prefillRow (step : S → List R → S × List R) (s0 : S) (xs : List R) (p : Nat) : Option R :=
  nth (step (stateAfter step s0 xs (p / 16)) (chunkOf xs (p / 16))).2 (p % 16)

/-- The replay decode's row `p`: the step from the boundary state over the `p % 16` rows since the boundary and the
new row (`takeN (p % 16 + 1)` of the rows from the boundary), its last output. -/
def decodeRow (step : S → List R → S × List R) (s0 : S) (xs : List R) (p : Nat) : Option R :=
  nth (step (stateAfter step s0 xs (p / 16)) (takeN (p % 16 + 1) (dropN (16 * (p / 16)) xs))).2 (p % 16)

/-- Decode row `p` is prefill row `p`, for every position, under the causality of the step (the hardware hypothesis:
the outputs of a prefix of a chunk are the prefix of the chunk's outputs). -/
theorem decode_eq_prefill (step : S → List R → S × List R)
    (hcausal : ∀ (s : S) (c : List R) (n : Nat), (step s (takeN n c)).2 = takeN n (step s c).2)
    (s0 : S) (xs : List R) (p : Nat) : decodeRow step s0 xs p = prefillRow step s0 xs p := by
  unfold decodeRow prefillRow chunkOf
  rw [← takeN_takeN (p % 16 + 1) 16 (dropN (16 * (p / 16)) xs) (by omega), hcausal,
    nth_takeN (p % 16 + 1) (p % 16) _ (by omega)]

/-! ## The pool's commit rule carries the prefill's boundary states -/

/-- The boundary state a pool slot holds after consuming `p` tokens one by one: it advances when the 16th row of a
chunk arrives (`pool.py`: the step that appends a slot's 16th row writes the chunk's final state to the pool). -/
def poolState (step : S → List R → S × List R) (s0 : S) (xs : List R) : Nat → S
  | 0 => s0
  | p+1 => if (p + 1) % 16 = 0 then (step (poolState step s0 xs p) (chunkOf xs (p / 16))).1 else poolState step s0 xs p

theorem pool_eq_stateAfter (step : S → List R → S × List R) (s0 : S) (xs : List R) :
    ∀ p, poolState step s0 xs p = stateAfter step s0 xs (p / 16)
  | 0 => by simp [poolState, stateAfter]
  | p+1 => by
      unfold poolState
      split
      · rename_i h
        have h2 : (p + 1) / 16 = p / 16 + 1 := by omega
        rw [h2, pool_eq_stateAfter step s0 xs p]
        simp only [stateAfter]
      · rename_i h
        have h2 : (p + 1) / 16 = p / 16 := by omega
        rw [h2, pool_eq_stateAfter step s0 xs p]

/-! ## The multi-token verify (`spec.py`: `verify_split`, `accepted_count`, `accepted_commits`)

A verify appends `n` rows to a slot with `count` open rows in one launch and is settled with `a ≤ n` of them kept. The
slot then has to be where `a` single decode steps from the verify's start leave it. What is stated here is the
arithmetic of that settlement only (a verify is at most 16 rows, so it crosses one boundary at most): the count after
the kept rows, and what the record must hold when the launch crosses a boundary, namely the boundary state and the
cached rows of the open chunk that the rows after the boundary replace. The kernel that applies it (and the decay
prefix it keeps per row) is not modelled; the device tests compare its result with single decode steps bit for bit. -/

/-- The open rows after `n` rows appended to a slot with `count` open rows. -/
def countAfter (count n : Nat) : Nat := (count + n) % 16

/-- The boundaries `n` appended rows cross. -/
def commitsIn (count n : Nat) : Nat := (count + n) / 16

/-- The record of a verify of `n ≤ 16` rows, as `saved rows + 1`: `0` while the rows stay inside the open chunk, else
one more than the number of the open chunk's cached rows that the rows after the boundary replace. -/
def verifySaved (count n : Nat) : Nat := if count + n < 16 then 0 else min count (count + n - 16) + 1

theorem countAfter_zero (count : Nat) (h : count < 16) : countAfter count 0 = count := by
  unfold countAfter; omega

/-- A verify's count is the count of its rows fed as single decode steps. -/
theorem countAfter_succ (count n : Nat) : countAfter count (n + 1) = nextCount (countAfter count n) := by
  unfold countAfter nextCount; omega

/-- A verify inside the open chunk commits nothing: the boundary state is the one before it. -/
theorem commitsIn_inside (count n : Nat) (h : count + n < 16) : commitsIn count n = 0 := by
  unfold commitsIn; omega

theorem verifySaved_inside (count n : Nat) (h : count + n < 16) : verifySaved count n = 0 := by
  simp [verifySaved, h]

/-- Every cached row `j` of the open chunk (`j < count`) that a crossing verify replaces (the rows after the first
boundary are rows `0 .. count + n - 16` of the next chunk) is among the saved ones. -/
theorem verifySaved_covers (count n j : Nat) (hcross : 16 ≤ count + n)
    (hj : j < count) (hover : j < count + n - 16) : j + 1 < verifySaved count n + 1 := by
  have hnot : ¬ (count + n < 16) := by omega
  simp only [verifySaved, hnot, if_false]
  have : j < min count (count + n - 16) := by
    rw [Nat.lt_min]; exact ⟨hj, hover⟩
  omega

/-- The pool rule at the kept position `p + a` (`pool_eq_stateAfter` there): the boundary state a settled slot has to
hold is the prefill's. That the settlement produces it is the device tests' statement, not this one. -/
theorem settled_pool_eq_stateAfter (step : S → List R → S × List R) (s0 : S) (xs : List R) (p a : Nat) :
    poolState step s0 xs (p + a) = stateAfter step s0 xs ((p + a) / 16) :=
  pool_eq_stateAfter step s0 xs (p + a)

/-! ## The launch model: a packed call is the per-sequence terms -/

/-- A packed call over sequences `(s0, xs)`: each sequence's term, nothing of the batch enters it. -/
def batch (step : S → List R → S × List R) (seqs : List (S × List R)) (p : Nat) : List (Option R) :=
  seqs.map (fun sq => prefillRow step sq.1 sq.2 p)

theorem batch_term (step : S → List R → S × List R) (seqs : List (S × List R)) (p i : Nat) :
    nth (batch step seqs p) i = (nth seqs i).map (fun sq => prefillRow step sq.1 sq.2 p) :=
  nth_map _ seqs i

end BIK.Kda
