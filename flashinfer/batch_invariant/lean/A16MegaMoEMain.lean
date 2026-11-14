-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.A16MegaMoE
import BIKernels.A16MegaMoECases

open BIK.A16MegaMoE

def main : IO Unit := do
  for (name, params) in conformanceCases do
    for (schedule, carrier) in [("static", "epi_warps"), ("static", "reuse_dispatch_warps"),
        ("atomic_counter", "epi_warps"), ("atomic_counter", "reuse_dispatch_warps")] do
      for (blocked, label) in [(false, "flat"), (true, "blocked")] do
        for (component, expression) in componentPairs params (components params blocked) do
          IO.println s!"a16mega {name} {schedule} {carrier} {label} {component} {BIK.MegaMoE.sexpr expression}"
