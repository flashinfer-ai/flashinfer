# Prototype evidence and validation boundary

These measurements motivated the experimental backend. They are **not timings
of the adapted upstream revision**: that revision adds device global scales,
output-buffer handling and both scale layouts, and still needs GPU validation.
The companion CSV contains all 60 paired case medians, including regressions.

Hardware: RTX 5090; Torch 2.13.0+cu130; Triton 3.7.1; FlashInfer 0.6.18.post1;
CuTe DSL 4.6.2; CUDA 13.0; driver 580.142. One GPU was used at a time. Inputs
were resident, global scale was fixed to one, and scales used 128x4 layout.
The CuTe baseline used its default PDL behavior. Public output allocation was
captured in the graph pool. Each case used five randomized rounds, with
50 calls per graph and 20 replays. No launch-time constant was subtracted.

For each run, shape and seed, take the median across rounds; calculate the
baseline/prototype ratio within that case; then take an equal-weight geometric
mean across cases. This gives **1.1423x preprocessing** and **1.0192x full-chain**.
The latter uses identical static quantized weights and the same numerically
validated cuBLASLt algorithm per case; it is a synthetic single-layer result,
not a model-serving result. The two independent preprocessing ratios were
1.1468x and 1.1378x; chain ratios were 1.0200x and 1.0185x. GPU clocks were not
locked. A 2% chain result should not be extrapolated to service throughput.

## Representative cases

Medians across the per-run, per-seed medians, in microseconds:

| M / N / K | CuTe preprocessing | Prototype preprocessing | CuTe chain | Prototype chain |
|---|---:|---:|---:|---:|
| 4096 / 2048 / 7168 | 23.165 | 17.437 | 137.814 | 132.248 |
| 4 / 2048 / 7168 | 1.971 | 1.984 | 20.549 | 20.550 |
| 16384 / 4096 / 4096 | 93.145 | 91.659 | 564.575 | 566.014 |

The second row shows no gain; the last shows that faster preprocessing can
still accompany a slightly slower complete chain. These observations motivate
explicit opt-in and separate operator/chain measurements.

## Correctness and provenance

The prototype passed 4,198 native E2M1 encoding cases, ten independent-reference
fused cases and memcheck with zero errors. Two formal runs produced 3,600 timing
rows and 720 numerical checks across all compared methods (120 of those checks
were for the Triton prototype). FP4 quantization error is distinct from kernel
implementation error; no model quality was evaluated.

Prototype source SHA256:
`d7be409db2d87d95a82fa2cba3be70876b409ea7438b18d595273ddc33e4f7ea`.
CuTe RMSNorm FP4 module SHA256:
`ec32fae9254adb9b888c0affd99822c89806a5881de04db0b0a755d81f6f90a3`.
The latter matches that module at base commit
`5d0c89eacae6ca08f2a1ce92eba557bbad7a1bfc`; it does not establish equivalence of
all surrounding package code. The CSV supports recomputing these aggregates;
re-running the new benchmark validates the adapted revision, not this historical
prototype. No raw cluster identifiers or local account paths are included.

## Adapted revision

Repository-wide pre-commit and changed-file mypy checks passed locally.
SM120 tests, memcheck, paired timing, and upstream CI are pending. The draft PR
must not be marked ready on the strength of prototype validation alone.
