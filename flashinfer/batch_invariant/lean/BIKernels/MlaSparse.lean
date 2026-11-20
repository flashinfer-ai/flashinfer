-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0

/-!
# Sparse MLA: the classes `mla.sparse.<dtype>.d<576|512>` and their placements

Mirrors `flashinfer/batch_invariant/mla/spec.py` (the list arithmetic) and `lean_cases.py` (the placements). The
kernel is hand-written and pinned to one schedule, so it is opaque here: a row term `row : Q → List K → O` takes one
query row and its selected-key list (in list order) to that row's output. The hypotheses the model takes about it are
the hardware hypotheses the GPU tests check bit for bit: the kernel is deterministic, and a query row's output is the
row term of that row's own query and list (no other row, no extent of the call enters it).

What is proved: a call is the list of its rows' terms (`call`); a row's output does not depend on its position or on
the other rows (`call_nth`); a call cut into chunks is the uncut call (`chunked_eq_call`), so a prefill-sized call, a
chunked prefill and one-row decode calls give the same rows (`decode_eq_prefill_row`); rows grouped per request
(multi-token rows) are the same rows (`grouped_eq_call`); appending padding rows changes no earlier row
(`padded_prefix`). The row term is a fold over whole 128-key tiles in list order (`rowFold`): the tile count and the
live keys of the last tile are functions of the row's own live length (`keyTiles`, `lastLive`), which the generated
`MlaSparseCases.lean` pins on the Python specification's cases by kernel `decide`. No configuration space is modelled
(R3 is not claimed for these classes): the schedule is one.
-/
namespace BIK.MlaSparse

universe u v w
variable {Q : Type u} {K : Type v} {O : Type w}

/-! ## List helpers (self-contained) -/

def takeN {α : Type u} : Nat → List α → List α
  | 0, _ => []
  | _+1, [] => []
  | n+1, x :: xs => x :: takeN n xs

def dropN {α : Type u} : Nat → List α → List α
  | 0, xs => xs
  | _+1, [] => []
  | n+1, _ :: xs => dropN n xs

def nth {α : Type u} : List α → Nat → Option α
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: xs, n+1 => nth xs n

def flat {α : Type u} : List (List α) → List α
  | [] => []
  | x :: xs => x ++ flat xs

/-! ## A call is its rows' terms -/

/-- One call of the operator: a row term per query row, in row order. -/
def call (row : Q → List K → O) : List (Q × List K) → List O
  | [] => []
  | r :: rs => row r.1 r.2 :: call row rs

/-- A one-row decode call. -/
theorem call_single (row : Q → List K → O) (r : Q × List K) : call row [r] = [row r.1 r.2] := rfl

/-- Row `i` of any call is the row term of row `i`: batch size, position and neighbours do not enter it. -/
theorem call_nth (row : Q → List K → O) :
    ∀ (rows : List (Q × List K)) (i : Nat), nth (call row rows) i = (nth rows i).map (fun r => row r.1 r.2)
  | [], _ => rfl
  | _ :: _, 0 => rfl
  | _ :: rs, i+1 => call_nth row rs i

theorem call_append (row : Q → List K → O) (a b : List (Q × List K)) :
    call row (a ++ b) = call row a ++ call row b := by
  induction a with
  | nil => rfl
  | cons r rs ih => simp [call, ih]

/-- A prefill cut into chunks (each its own call) is the uncut call. -/
theorem chunked_eq_call (row : Q → List K → O) (chunks : List (List (Q × List K))) :
    flat (chunks.map (call row)) = call row (flat chunks) := by
  induction chunks with
  | nil => rfl
  | cons c cs ih => simp [flat, call_append, ih]

/-- Rows grouped per request (several query tokens per request) are the rows of the flat call. -/
theorem grouped_eq_call (row : Q → List K → O) (requests : List (List (Q × List K))) :
    flat (requests.map (call row)) = call row (flat requests) :=
  chunked_eq_call row requests

/-- The decode call of row `i` returns row `i` of the prefill-sized call. -/
theorem decode_eq_prefill_row (row : Q → List K → O) (rows : List (Q × List K)) (i : Nat) (r : Q × List K)
    (h : nth rows i = some r) : nth (call row rows) i = nth (call row [r]) 0 := by
  rw [call_nth, h]
  rfl

/-- Padding rows appended to a call (a larger bucket) leave the earlier rows' outputs in place. -/
theorem padded_prefix (row : Q → List K → O) (rows pad : List (Q × List K)) :
    call row (rows ++ pad) = call row rows ++ call row pad :=
  call_append row rows pad

/-! ## The row term: a fold over 128-key tiles in list order -/

/-- Whole 128-key tiles a list of `n` live keys spans (`spec.key_tiles`). -/
def keyTiles (n : Nat) : Nat := (n + 127) / 128

/-- Live keys in the last tile (`0` for an empty list); the rest of that tile is masked. -/
def lastLive (n : Nat) : Nat := n - 128 * (keyTiles n - 1)

theorem keyTiles_cover (n : Nat) : n ≤ 128 * keyTiles n := by
  unfold keyTiles; omega

theorem keyTiles_tight (n : Nat) (h : 0 < n) : 128 * (keyTiles n - 1) < n := by
  unfold keyTiles; omega

theorem lastLive_le (n : Nat) : lastLive n ≤ 128 := by
  unfold lastLive keyTiles; omega

/-- The first `t` tiles of a list, in list order. -/
def tiles {α : Type u} : Nat → List α → List (List α)
  | 0, _ => []
  | t+1, ks => takeN 128 ks :: tiles t (dropN 128 ks)

theorem tiles_length {α : Type u} : ∀ (t : Nat) (ks : List α), (tiles t ks).length = t
  | 0, _ => rfl
  | t+1, ks => by simp [tiles, tiles_length t]

/-- The shape of the kernel's row term: a state stepped once per tile, in list order, then finished. -/
def rowFold {S : Type w} (step : S → List K → S) (finish : S → O) (init : S) (keys : List K) : O :=
  finish ((tiles (keyTiles keys.length) keys).foldl step init)

/-- A call whose row term is the tile fold applies the same fold to every row, from the same initial state. -/
theorem call_rowFold {S : Type w} (step : Q → S → List K → S) (finish : Q → S → O) (init : S)
    (rows : List (Q × List K)) (i : Nat) :
    nth (call (fun q ks => rowFold (step q) (finish q) init ks) rows) i
      = (nth rows i).map (fun r => rowFold (step r.1) (finish r.1) init r.2) :=
  call_nth _ rows i

end BIK.MlaSparse
