-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
/-!
# The request-keyed stream of the batch-invariant samplers (`sampling/stream.py`)

The samplers under `batch_invariant=True` draw their uniforms from Philox4x32-10 on the counter `(n, 0, 0, 0)` under
the key `K = mix64(seed ^ mix64(offset))` (SplitMix64's finalizer): word `i` of a request is `block(i / 4)[i % 4]`,
draw `d` reads block `d` (word 0 -> the fp32 uniform, words 1, 2 -> the fp64 uniform). The row's slot in the batch
enters nothing. This module is the Lean model of the integer stream; the published Random123 known-answer vectors and
the SplitMix64 vectors are decided here (kernel `decide`), as they are in the Python test for the specification whose
words the device kernel reproduces bit for bit; the generated cases (`SamplingCases.lean`, `stream*`) decide the
Python specification's blocks on a few keys against this model.
-/
namespace BIK.Stream

def M0 : UInt32 := 0xD2511F53
def M1 : UInt32 := 0xCD9E8D57
def W0 : UInt32 := 0x9E3779B9
def W1 : UInt32 := 0xBB67AE85

def mulhi (a b : UInt32) : UInt32 := ((a.toUInt64 * b.toUInt64) >>> 32).toUInt32

structure Ctr where
  c0 : UInt32
  c1 : UInt32
  c2 : UInt32
  c3 : UInt32
  deriving Repr, DecidableEq

structure Key where
  k0 : UInt32
  k1 : UInt32
  deriving Repr, DecidableEq

/-- One Philox4x32 round (Random123): `(hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0)` with `(hi0, lo0) = M0 * c0`,
`(hi1, lo1) = M1 * c2`. -/
def round (c : Ctr) (k : Key) : Ctr :=
  ⟨mulhi M1 c.c2 ^^^ c.c1 ^^^ k.k0, M1 * c.c2, mulhi M0 c.c0 ^^^ c.c3 ^^^ k.k1, M0 * c.c0⟩

/-- The key schedule: `k += (W0, W1)` between rounds. -/
def bump (k : Key) : Key := ⟨k.k0 + W0, k.k1 + W1⟩

def philox : Nat → Ctr → Key → Ctr
  | 0, c, _ => c
  | n+1, c, k => philox n (round c k) (bump k)

/-- Philox4x32-10. -/
def philox10 (c : Ctr) (k : Key) : Ctr := philox 10 c k

/-- SplitMix64's finalizer (`stream.mix64`). -/
def mix64 (v : UInt64) : UInt64 :=
  let v := v + 0x9E3779B97F4A7C15
  let v := (v ^^^ (v >>> 30)) * 0xBF58476D1CE4E5B9
  let v := (v ^^^ (v >>> 27)) * 0x94D049BB133111EB
  v ^^^ (v >>> 31)

/-- `stream.request_key`: the 64-bit key of a request. -/
def requestKey (seed offset : UInt64) : UInt64 := mix64 (seed ^^^ mix64 offset)

def keyWords (k : UInt64) : Key := ⟨k.toUInt32, (k >>> 32).toUInt32⟩

/-- `stream.block`: block `n` of the request's stream. -/
def block (seed offset n : UInt64) : Ctr :=
  philox10 ⟨n.toUInt32, (n >>> 32).toUInt32, 0, 0⟩ (keyWords (requestKey seed offset))

/-- `stream.word`: word `i` of the request's stream. -/
def word (seed offset : UInt64) (i : Nat) : UInt32 :=
  let b := block seed offset (UInt64.ofNat (i / 4))
  match i % 4 with
  | 0 => b.c0
  | 1 => b.c1
  | 2 => b.c2
  | _ => b.c3

/-! ## The published vectors (Random123 `kat_vectors`, philox4x32 10 rounds; SplitMix64 from the reference sequence) -/

theorem kat_zero : philox10 ⟨0, 0, 0, 0⟩ ⟨0, 0⟩ = ⟨0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8⟩ := by decide
theorem kat_ones : philox10 ⟨0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF⟩ ⟨0xFFFFFFFF, 0xFFFFFFFF⟩
    = ⟨0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD⟩ := by decide
theorem kat_pi : philox10 ⟨0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344⟩ ⟨0xA4093822, 0x299F31D0⟩
    = ⟨0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1⟩ := by decide
theorem mix64_zero : mix64 0 = 0xE220A8397B1DCDAF := by decide
theorem mix64_golden : mix64 0x9E3779B97F4A7C15 = 0x6E789E6AA1B965F4 := by decide
theorem mix64_two_golden : mix64 (2 * 0x9E3779B97F4A7C15) = 0x06C45D188009454F := by decide

/-- Two requests with different offsets under one seed have different keys (the decorrelation the stream relies
on, on a sample): the construction never maps `(seed, offset)` and `(seed, offset + 1)` to one key here. -/
theorem keys_differ_sample : requestKey 42 7 ≠ requestKey 42 8 := by decide

/-- The key has no slot term: it is a function of `(seed, offset)` alone, by its definition. -/
theorem key_slot_free (seed offset : UInt64) (slot : Nat) :
    requestKey seed offset = (fun (_ : Nat) => requestKey seed offset) slot := rfl

end BIK.Stream
