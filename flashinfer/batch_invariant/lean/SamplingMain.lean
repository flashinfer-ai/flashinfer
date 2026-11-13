import BIKernels.QuantExport
import BIKernels.Sampling
import BIKernels.Stream
import BIKernels.Draw
import BIKernels.SamplingCases
/-! `lake exe samplingconformance`: print the Lean model's terms for the generated sampling cases (the row class
`lse.row`, the draw `sample.icdf.v1`, the stream's blocks). The Python test (`tests/batch_invariant/test_sampling_lean.py`,
the Lean tier) compares these strings with the Python spec's / engine's S-expressions and words. -/
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
  -- the chains: the first / last probability, the first log-softmax element and the lse, with and without a temperature
  for (N, kb, temp) in lseOutCases do
    let T : Option Term := if temp then some (Term.sym "T") else none
    IO.println s!"lseout {N} {temp} probs0 {osexpr (Sampling.specProb "bf16" N kb lseRow T 0)}"
    IO.println s!"lseout {N} {temp} probsL {osexpr (Sampling.specProb "bf16" N kb lseRow T (N - 1))}"
    IO.println s!"lseout {N} {temp} logp0 {osexpr (Sampling.specLogp "bf16" N kb lseRow T 0)}"
    IO.println s!"lseout {N} {temp} lse {osexpr (Sampling.specLse "bf16" N kb lseRow T)}"
  -- the draw `sample.icdf.v1` (Draw.lean): the threshold and the inclusive prefixes of the probe elements of row 0
  for (N, kb, probe) in drawCases do
    IO.println s!"drawt {N} {osexpr (Draw.threshold N kb lseRow)}"
    for i in probe do
      IO.println s!"draw {N} {i} {osexpr (Draw.prefixIncl N kb lseRow i)}"
  -- the stream (Stream.lean): the four words of the generated keys' blocks
  for (s, o, n) in streamCases do
    let b := Stream.block s o n
    IO.println s!"stream {s} {o} {n} {b.c0} {b.c1} {b.c2} {b.c3}"
