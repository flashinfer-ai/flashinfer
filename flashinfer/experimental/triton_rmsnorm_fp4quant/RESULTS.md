# Prototype evidence and validation boundary

These measurements motivated the experimental backend. They are **not timings
of the adapted upstream revision**: that revision adds device global scales,
output-buffer handling and both scale layouts, and is validated separately below.
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

## Adapted revision: SM120 validation completed

On 2026-09-14, the implementation at commit
`1cbddaadacd874bacba5eae0afce8d972bd7ae50` was validated on a second RTX 5090.
The subsequent evidence update changes documentation and CSV data only.
Python 3.12.3, Torch 2.13.0+cu130, Triton 3.7.1, CuTe DSL 4.6.2,
NVCC 13.0.88 and driver 595.71.05 were used. The API, kernels, tests and
benchmark were loaded from the pinned repository snapshot; the installed
0.6.18.post1 wheel supplied dependencies and third-party headers, not the
measured Python implementations.

Validation results:

- The new test file plus `tests/experimental/test_experimental_api.py`:
  **91 passed, zero skipped**.
- The new test file under the full NVIDIA CUDA 13.0 Compute Sanitizer
  (2025.3.1): **77 passed, ERROR SUMMARY: 0 errors**. The pip-distributed
  sanitizer failed to launch the target; the complete NVIDIA distribution
  was used for the successful run, with default process tracking.
- Two independent, sequential benchmark processes, forward and reverse:
  **120 cases, 360 numerical checks, 1,800 timing records** in total.
  Shape/seed/scale/method/round coverage, finite samples, source hashes and
  matching environments were audited. Smoke, public-allocation and eager
  row-major diagnostics also passed and are excluded from the main table.
- Repository-wide pre-commit and changed-file mypy passed locally.
  Upstream GPU CI authorization and maintainer review are separate pending
  steps; no upstream GPU CI pass is claimed.

The fixed benchmark has 15 shapes, two seeds and global scales 1 and 32.
For each case it measures CuTe default PDL, CuTe PDL disabled, and Triton,
using preallocated outputs, swizzled scales, five randomized method rounds,
50 calls per graph and 20 replay samples. Ratios are paired within each
run/shape/seed/scale after taking the median across rounds, then geometrically
averaged within each scale and baseline. [adapted_measurements.csv](adapted_measurements.csv)
contains all 240 baseline/Triton pairs, including any regressions.

| Global scale | CuTe default / Triton | CuTe PDL-off / Triton |
|---|---:|---:|
| 1 | 1.1422x | 1.2210x |
| 32 | 1.1375x | 1.2157x |

Against the default baseline, the forward/reverse geometric means were
1.1388x/1.1456x for scale 1 and 1.1356x/1.1394x for scale 32. The default
baseline is the primary comparison; PDL-off is an explanatory control.
Clocks were not locked. The new driver, Python version, allocation protocol
and transitive dependencies differ from the historical prototype environment;
absolute latencies across these experiments should not be divided.

These are **adapted preprocessing measurements only**. The adapted GEMM chain
and model serving were not benchmarked. The prototype's 1.0192x chain result
above remains historical and must not be attributed to this adapted revision.

Source SHA256 values for the validated adaptation:

- API: `1de99b7dcbec2ef0426f517bfa41d3ae6b7f758f20b03f8921af528684f6b074`
- Backend: `2383b932ef5a21dfef08998ee095a2f8c10c688a0a9f86aef3321a2790b6e93f`
- Kernel: `9b438e7877bc11d6d6b1134b909d81fd466ecde62adf6ef974e53979c0a39c45`
- Tests: `7343eb31f4d6bdd5fa9cec7124e20813991d7854d1ea30b65478fb6cba241a6c`
- Benchmark: `dc324f5ac1fa6d269e9c01fd192313355c194f7d6fc4ebc65a4eab606b4e176c`
