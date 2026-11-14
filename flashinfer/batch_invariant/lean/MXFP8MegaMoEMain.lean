-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.MXFP8MegaMoE
import BIKernels.MXFP8MegaMoECases

open BIK.MXFP8MegaMoE

def main : IO Unit := do
  for (name, params) in conformanceCases do
    for (blocked, label) in [(false, "flat"), (true, "blocked")] do
      for (component, expression) in componentPairs (components params blocked) do
        IO.println s!"mx8mega {name} {label} {component} {BIK.MegaMoE.sexpr expression}"
