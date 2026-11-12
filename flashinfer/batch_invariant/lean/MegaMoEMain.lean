-- Copyright (c) 2026 by FlashInfer team.
-- SPDX-License-Identifier: Apache-2.0
import BIKernels.MegaMoE
import BIKernels.MegaMoECases

open BIK.MegaMoE

def scheduleName : Scheduler → String
  | .static => "static"
  | .atomicCounter => "atomic_counter"

def carrierName : Carrier → String
  | .epi => "epi_warps"
  | .standalone => "standalone_warps"
  | .reuseDispatch => "reuse_dispatch_warps"

def tileName (g : PlanConfig) : String :=
  s!"m{g.tileM}n{g.tileN}k{g.tileK}c{g.clusterM}"

def main : IO Unit := do
  for c in conformanceCases do
    for p in plans do
      for (blocked, path) in [(false, "flat"), (true, "blocked")] do
        let values := components c.hidden c.intermediate c.slots c.experts
          c.early c.clamp blocked c.mask
        for (name, value) in componentPairs values do
          IO.println s!"mega {c.name} {scheduleName p.scheduler} {carrierName p.carrier} {path} {name} {sexpr value}"
      for g in c.configs do
        let values := tiledComponents g c.hidden c.intermediate c.slots c.experts
          c.early c.clamp c.mask
        for (name, value) in componentPairs values do
          IO.println s!"megatiled {c.name} {tileName g} {scheduleName p.scheduler} {carrierName p.carrier} {name} {sexpr value}"
  for k in mmaCases do
    IO.println s!"mma {k} flat {sexpr (mmaFlat k "boundary")}"
    IO.println s!"mma {k} blocked {sexpr (mmaBlocked k "boundary")}"
