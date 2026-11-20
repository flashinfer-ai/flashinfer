-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0

/-!
# GDN: the class `gdn.bf16.c64`, its decode step and its multi-token placements

Mirrors `flashinfer/batch_invariant/gdn/spec.py` (the arithmetic) and `pool.py` (the placements). The kernels are
hand-written, so they are opaque here: a chunk step `step : S → List R → S × List R` (the chunked prefill) takes the
state at a chunk boundary and a chunk's rows (at most 64) to the state after them and their output rows, and a row
step `dec : S → List R → Option R` (the decode step kernel) takes the boundary state and the open chunk's rows up to
the new one to the new row's output. The hypotheses the model takes about them are the hardware hypotheses the GPU
tests check bit for bit: the chunk step is deterministic and a function of its own sequence alone (which is what the
definition of a packed call, `batch`, assumes), it is causal (the outputs of a chunk prefix are the prefix of the chunk's
outputs, `hcausal`), and the row step returns the last output of the chunk step on the same rows (`hdec`).

Defined: the prefill of a sequence as the fold of `step` over its 64-row chunks from the initial state (`stateAfter`,
`prefillRow`), and a packed call as the per-sequence terms (`batch`; `batch_term` only unfolds that definition).
Proved: the decode step's row `p` is the prefill's row `p` (`decode_eq_prefill`); the prefill kernel run over the open
chunk plus new tokens returns the prefill's rows for them (`replay_eq_prefill`), so both multi-token placements are
the term; the commit rule carries exactly the prefill's boundary states (`pool_eq_stateAfter`); the arithmetic of a
call that appends `n` tokens to `count` open rows (`extend_boundary`). The pool's bookkeeping is not modelled. No configuration space is
modelled (R3 is not claimed for this class). The generated `GdnCases.lean` pins the arithmetic on the Python
specification's cases by kernel `decide`.
-/
namespace BIK.Gdn

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

/-! ## The arithmetic (`spec.py`: `boundary`, `phase`, `commits`, `next_count`, `extend_split`, `uses_step_kernel`) -/

def boundary (p : Nat) : Nat := 64 * (p / 64)
def phase (p : Nat) : Nat := p % 64
def commits (count : Nat) : Bool := count + 1 == 64
def nextCount (count : Nat) : Nat := (count + 1) % 64
/-- A call appending `n` tokens to `count` open rows: `(whole, tail)`. -/
def extendSplit (count n : Nat) : Nat × Nat := (boundary (count + n), phase (count + n))
def usesStepKernel (maxTokens : Nat) : Bool := decide (maxTokens < 64)

theorem boundary_add_phase (p : Nat) : boundary p + phase p = p := by
  unfold boundary phase; omega

theorem phase_lt (p : Nat) : phase p < 64 := by
  unfold phase; omega

theorem commits_iff (count : Nat) : commits count = true ↔ count + 1 = 64 := by
  simp [commits]

theorem nextCount_of_commit (count : Nat) (h : commits count = true) : nextCount count = 0 := by
  have := (commits_iff count).mp h
  unfold nextCount; omega

theorem extendSplit_sum (count n : Nat) : (extendSplit count n).1 + (extendSplit count n).2 = count + n :=
  boundary_add_phase (count + n)

/-- Appending `n` tokens at position `p` moves the slot's boundary by the call's whole chunks and leaves its tail. -/
theorem extend_boundary (p n : Nat) :
    boundary p + (extendSplit (phase p) n).1 = boundary (p + n) ∧ (extendSplit (phase p) n).2 = phase (p + n) := by
  unfold extendSplit boundary phase; omega

/-! ## Prefill as the chunk fold, the decode step and the replay -/

/-- Chunk `k` of a sequence: rows `[64 k, 64 k + 64)` (shorter at the end). -/
def chunkOf (xs : List R) (k : Nat) : List R := takeN 64 (dropN (64 * k) xs)

/-- The state at boundary `k` (after `k` whole chunks): the fp32 state the pool slot holds. -/
def stateAfter (step : S → List R → S × List R) (s0 : S) (xs : List R) : Nat → S
  | 0 => s0
  | k+1 => (step (stateAfter step s0 xs k) (chunkOf xs k)).1

/-- The prefill's output row `p`: row `p % 64` of the chunk `p / 64`, stepped from its boundary state. -/
def prefillRow (step : S → List R → S × List R) (s0 : S) (xs : List R) (p : Nat) : Option R :=
  nth (step (stateAfter step s0 xs (p / 64)) (chunkOf xs (p / 64))).2 (p % 64)

/-- The open chunk's rows up to and including row `p`: what the decode step's cached rows are functions of. -/
def openRows (xs : List R) (p : Nat) : List R := takeN (p % 64 + 1) (dropN (64 * (p / 64)) xs)

/-- The decode step's row `p`: the row step from the boundary state over the open rows. -/
def decodeRow (step : S → List R → S × List R) (dec : S → List R → Option R) (s0 : S) (xs : List R) (p : Nat) :
    Option R :=
  dec (stateAfter step s0 xs (p / 64)) (openRows xs p)

/-- The replay's row `p` of a call that ends at open row `e ≥ p % 64`: the chunk step over the open rows up to `e`. -/
def replayRow (step : S → List R → S × List R) (s0 : S) (xs : List R) (p e : Nat) : Option R :=
  nth (step (stateAfter step s0 xs (p / 64)) (takeN (e + 1) (dropN (64 * (p / 64)) xs))).2 (p % 64)

/-- The prefill kernel over the open chunk plus the new tokens gives the prefill's rows, under the causality of the
chunk step. -/
theorem replay_eq_prefill (step : S → List R → S × List R)
    (hcausal : ∀ (s : S) (c : List R) (n : Nat), (step s (takeN n c)).2 = takeN n (step s c).2)
    (s0 : S) (xs : List R) (p e : Nat) (hp : p % 64 ≤ e) (he : e < 64) :
    replayRow step s0 xs p e = prefillRow step s0 xs p := by
  unfold replayRow prefillRow chunkOf
  rw [← takeN_takeN (e + 1) 64 (dropN (64 * (p / 64)) xs) (by omega), hcausal,
    nth_takeN (e + 1) (p % 64) _ (by omega)]

/-- Decode row `p` is prefill row `p`, for every position: the row step returns the chunk step's last output on the
open rows (`hdec`) and the chunk step is causal (`hcausal`), the two hardware hypotheses. -/
theorem decode_eq_prefill (step : S → List R → S × List R) (dec : S → List R → Option R)
    (hcausal : ∀ (s : S) (c : List R) (n : Nat), (step s (takeN n c)).2 = takeN n (step s c).2)
    (hdec : ∀ (s : S) (xs : List R) (p : Nat),
      dec s (openRows xs p) = nth (step s (openRows xs p)).2 (p % 64))
    (s0 : S) (xs : List R) (p : Nat) : decodeRow step dec s0 xs p = prefillRow step s0 xs p := by
  unfold decodeRow
  rw [hdec]
  exact replay_eq_prefill step hcausal s0 xs p (p % 64) (Nat.le_refl _) (by omega)

/-! ## The pool's commit rule carries the prefill's boundary states -/

/-- The boundary state a pool slot holds after consuming `p` tokens one by one: it advances when the 64th row of a
chunk arrives (`pool.py`: the step kernel's completing row writes the next boundary state). -/
def poolState (step : S → List R → S × List R) (s0 : S) (xs : List R) : Nat → S
  | 0 => s0
  | p+1 => if (p + 1) % 64 = 0 then (step (poolState step s0 xs p) (chunkOf xs (p / 64))).1 else poolState step s0 xs p

theorem pool_eq_stateAfter (step : S → List R → S × List R) (s0 : S) (xs : List R) :
    ∀ p, poolState step s0 xs p = stateAfter step s0 xs (p / 64)
  | 0 => by simp [poolState, stateAfter]
  | p+1 => by
      unfold poolState
      split
      · rename_i h
        have h2 : (p + 1) / 64 = p / 64 + 1 := by omega
        rw [h2, pool_eq_stateAfter step s0 xs p]
        simp only [stateAfter]
      · rename_i h
        have h2 : (p + 1) / 64 = p / 64 := by omega
        rw [h2, pool_eq_stateAfter step s0 xs p]

/-! ## The launch model: a packed call is the per-sequence terms -/

/-- A packed call over sequences `(s0, xs)`: each sequence's term, nothing of the batch enters it. -/
def batch (step : S → List R → S × List R) (seqs : List (S × List R)) (p : Nat) : List (Option R) :=
  seqs.map (fun sq => prefillRow step sq.1 sq.2 p)

theorem batch_term (step : S → List R → S × List R) (seqs : List (S × List R)) (p i : Nat) :
    nth (batch step seqs p) i = (nth seqs i).map (fun sq => prefillRow step sq.1 sq.2 p) :=
  nth_map _ seqs i

end BIK.Gdn
