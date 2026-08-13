# CAKE KDA K1 parallelism validation

This report records the upstream-readiness checks for the B200 owner/helper
kernel. It is intentionally separate from an upstream pull request: no PR has
been opened from this branch.

## Validated revision and environment

- Source revision: `406978e99ad1065386055b1201df46ca4145f7b3`
- Date: 2026-08-13
- Device-reported name: NVIDIA L20C
- Compute capability: 10.0
- SM count: 148
- L2 cache: 126 MiB
- Driver: 580.95.05
- Python: 3.12.3
- PyTorch: 2.13.0+cu130
- CUDA runtime reported by PyTorch: 13.0

The remote platform exposes a B200-class 148-SM CC 10.0 device under the L20C
product string. Dispatch and kernel compatibility are based on compute
capability, not that product string.

## Correctness and repository gates

| Gate | Result |
| --- | --- |
| Targeted JIT and recurrent-KDA tests | 78 passed, 3 warnings |
| C8 and C4 output versus FP32 reference | Passed |
| C8 and C4 recurrent state versus FP32 reference | Passed |
| Benchmark output/state versus CAKE-M64 | Bitwise identical for all measured shapes |
| CUDA graph capture/replay | Passed |
| Pre-commit, all files | Passed |

The full repository test suite and AOT build matrix were not run. They remain
required CI gates before an upstream merge.

## Compute Sanitizer

Focused C8 and C4 runs of the retained implementation passed both tools:

| Route | memcheck | synccheck |
| --- | --- | --- |
| C8, one owner plus seven helpers | `ERROR SUMMARY: 0 errors` | `ERROR SUMMARY: 0 errors` |
| C4, one owner plus three helpers | `ERROR SUMMARY: 0 errors` | `ERROR SUMMARY: 0 errors` |

The commands used `--report-api-errors no` because CuTe DSL capability probing
calls `cuGetProcAddress_v2`; those API-probe diagnostics are unrelated to the
generated CAKE kernel. A focused kernel filter was also used so the report
covers the M128 owner/helper kernel.

An experimental device-wide helper pool was not retained. Although it matched
the baseline bitwise and passed memcheck, synccheck reported divergent barrier
use. Public routing and the FFI binding therefore accept only co-scheduled C4
and C8 cluster launches.

## Cold-L2 performance

Run from the public `recurrent_kda` API with:

```bash
python benchmarks/bench_kda_k1_parallelism.py \
  --warmup-ms 10 --bench-ms 30 --state-rotations 2048 \
  --json k1_parallel.json
```

The benchmark uses deterministic tensors, preinitialized rotating state
buffers, no CUDA graph, and bitwise output/state checks before timing. The
baseline is selected independently for every shape as
`min(CAKE-M64, CAKE-M128)`. The result JSON SHA256 was
`68ba0caa9c41dc6cdc78c13e0e1a92e9b5bcb2ef80421c4126f53d537ce098cf`.

| T | H=8 | H=16 | H=24 | H=32 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.998x | 1.000x | 1.000x | 1.010x |
| 2048 | 1.070x | 1.084x | 1.065x | 1.049x |
| 4096 | 1.154x | 1.142x | 1.109x | 1.110x |
| 8192 | 1.196x | 1.189x | 1.147x | 1.125x |

The router therefore keeps T=1024 on the exact M64/M128 fallback and enables
owner/helper execution only from T=2048. Across the enabled table above, the
measured range is 1.049x to 1.196x over the per-shape oracle.

## Upstream PR preparation

FlashInfer requires the default pull-request template, pre-commit, updated
tests, and reproducible before/after performance numbers for optimization
PRs. Public CI does not start automatically: a `ci-users` member must comment
`@flashinfer-bot run` or reapply the `run-ci` label after the final commit.
Internal CI, when available, is started with `/bot run` and includes B200,
GB200, B300, and GB300 coverage.

Before opening the upstream PR:

1. rebase onto the current upstream main and rerun this entire report;
2. run M64 and M128 fallback memcheck/synccheck alongside C4 and C8;
3. run the AOT build matrix and the broader recurrent-KDA test set;
4. include the exact benchmark JSON and environment metadata in the PR body;
5. state explicitly if the full repository suite remains unrun.
