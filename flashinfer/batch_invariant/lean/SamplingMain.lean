import BIKernels.QuantExport
import BIKernels.Sampling
import BIKernels.SamplingCases
/-! `lake exe samplingconformance`: print the Lean model's terms for the generated sampling cases. The Python test
(`tests/batch_invariant/test_sampling_lean.py`, the Lean tier) compares these strings with the Python spec's / engine's
S-expressions. -/
open BIK

def main : IO Unit := do
  let lseRow : Nat → Term := fun i => Term.x i
  -- the row class `lse.row` (Sampling.lean): per placement the engine's maximum and, where the case carries it, the
  -- engine's sum over its own maximum, at unit (0, 0, 0), and the spec's
  for (N, kb, name, withSum, ps) in lseCases do
    IO.println s!"lse {N} {kb} {name} maxkernel {osexpr (Sampling.kernelMax N kb lseRow ps (zeroCoords (flatten ps)))}"
    IO.println s!"lse {N} {kb} {name} maxspec {osexpr (Sampling.specMax N kb lseRow)}"
    if withSum then
      IO.println s!"lse {N} {kb} {name} sumkernel {osexpr (Sampling.kernelSumOfMax N kb lseRow none ps (zeroCoords (flatten ps)))}"
      IO.println s!"lse {N} {kb} {name} sumspec {osexpr (Sampling.specSumOfMax N kb lseRow none)}"
      -- the entropy member's tree W over the engine's own maximum, and the spec's
      IO.println s!"lse {N} {kb} {name} wkernel {osexpr (Sampling.kernelWOfMax N kb lseRow none ps (zeroCoords (flatten ps)))}"
      IO.println s!"lse {N} {kb} {name} wspec {osexpr (Sampling.specWOfMax N kb lseRow none)}"
  -- the chains: the first / last probability, the first log-softmax element, the lse and the entropy, with and without a temperature
  for (N, kb, temp) in lseOutCases do
    let T : Option Term := if temp then some (Term.sym "T") else none
    IO.println s!"lseout {N} {temp} probs0 {osexpr (Sampling.specProb "bf16" N kb lseRow T 0)}"
    IO.println s!"lseout {N} {temp} probsL {osexpr (Sampling.specProb "bf16" N kb lseRow T (N - 1))}"
    IO.println s!"lseout {N} {temp} logp0 {osexpr (Sampling.specLogp "bf16" N kb lseRow T 0)}"
    IO.println s!"lseout {N} {temp} lse {osexpr (Sampling.specLse "bf16" N kb lseRow T)}"
    IO.println s!"lseout {N} {temp} entropy {osexpr (Sampling.specEntropy N kb lseRow T)}"
  -- the entropy chain over symbolic statistics (the kernel-decided pin `lse_entropy_chain` states the same string)
  IO.println s!"lsechain {osexpr (some (Sampling.entropyNat (Sampling.lg2 (Term.sym "s")) (Term.sym "W") (Term.sym "s")))}"
