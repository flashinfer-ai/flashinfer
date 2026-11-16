-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.MXFP8MegaMoE

/-! Stage regrouping preserves an ordered native-instruction fold. Each step
includes the original init/acc atom, M/N/CTA suffix and physical padded operand
bytes. The theorem does not cancel zero MMAs or equate different native K. -/
namespace BIK.QuantizedStagePartition
open BIK.MegaMoE (Expr)

def blocked (step : Option Expr → Nat → Option Expr) (stages perStage : Nat) : Option Expr :=
  (BIK.MegaMoE.tiledIndices stages perStage).foldl (fun acc tile => tile.foldl step acc) none

theorem same_instruction_count (step : Option Expr → Nat → Option Expr)
    (s p s' p' : Nat) (h : s * p = s' * p') :
    blocked step s p = blocked step s' p' := by
  unfold blocked
  rw [← List.foldl_flatten, BIK.MegaMoE.tiled_indices_flat,
      ← List.foldl_flatten, BIK.MegaMoE.tiled_indices_flat, h]

/-- Equal padded extents imply equal instruction counts when the native K is
identical. Divisibility of admitted stages is checked by each numerical class. -/
theorem equal_padding_count (K stage stage' native : Nat)
    (h : ((K + stage - 1) / stage) * stage = ((K + stage' - 1) / stage') * stage') :
    (((K + stage - 1) / stage) * stage) / native =
    (((K + stage' - 1) / stage') * stage') / native := by rw [h]

theorem equal_padding_preserves_fold (step : Option Expr → Nat → Option Expr)
    (K stage stage' native : Nat) (hd : native ∣ stage) (hd' : native ∣ stage')
    (h : ((K + stage - 1) / stage) * stage = ((K + stage' - 1) / stage') * stage') :
    blocked step ((K + stage - 1) / stage) (stage / native) =
    blocked step ((K + stage' - 1) / stage') (stage' / native) := by
  have count := congrArg (fun n => n / native) h
  change (((K + stage - 1) / stage) * stage) / native =
    (((K + stage' - 1) / stage') * stage') / native at count
  rw [Nat.mul_div_assoc _ hd, Nat.mul_div_assoc _ hd'] at count
  exact same_instruction_count step _ _ _ _ count
end BIK.QuantizedStagePartition
