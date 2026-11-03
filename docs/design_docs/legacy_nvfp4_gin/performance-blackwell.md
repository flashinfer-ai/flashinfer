# Blackwell performance verification

Historical pre-fix measurements. See [the correction report](performance-fix-blackwell.md)
for the final source, 24 passing off gates, internode host corrections and
remaining baseline-variance limitation. The original measurements below are retained.

Completed 2026-11-02 on NVIDIA GB300 (SM103). Output checks passed throughout
all accepted cases. **The off-mode performance gate did not pass overall.**
Of 24 off comparisons, 10 passed the predeclared 5% limit,
2 exceeded it, and 12 were inconclusive because baseline dispersion exceeded 5%.
On-mode ratios are descriptive; this experiment had no predeclared on speedup gate.

The explicit user request for Blackwell measurements superseded the earlier
local-performance waiver and the donor Rubin-only performance rule for this
experiment. No production kernel changes or tuning were made.

## Sources, hardware and workload

- F_BASE: `e74a1677ab95a1b97aa1c6b6a80b3cf5c0ebac34`.
- Fixed MR !100 donor: `f863d7540614bd546a42867c21adbf51d573061a`.
- Receiver/benchmark: `7138d8d514e0b07baf63141327b424a8c6fc7ced`;
  production runtime unchanged since `fbd41421b`.
- Program SHA-256: `6379f2eff6e868478c8283902e339680883f1f1b5630cb3ae03ff0c894cedff8`.
- Exclusive GB300 allocations, 152 SM/device, aarch64; paired sources use the
  same allocation. EP1/EP2 used theia0096; EP4 used theia0273. On used
  theia0028/theia0095, two ranks per host in independent NVLink domains.
- Container `nvcr.io#nvidia/pytorch:26.07-py3`; Torch
  `2.13.0a0+9186a08b2c.nv26.07`, CUDA 13.3, DSL 4.6.2, nccl4py 0.3.1,
  NCCL 2.30.7, driver 580.173.10. Per-rank UUIDs, clocks, versions, loaded
  modules and kernel settings remain in manifests. These are clock snapshots,
  not a claim that clocks were locked throughout measurement.

| Mode | Topology | Tokens/rank | H / public intermediate / total experts / K |
| --- | --- | --- | --- |
| Off | EP1, EP2, EP4; single host | 64, 1024 | 2048 / 1024 / 8 / 4 |
| On | N2R2; two hosts | 32 | 7168 / 1536 / 48 / 4 |

Both sources use identical kernel-ready quantized payloads and configuration
hashes. On uses a shared packed-input/weight fixture whose expected output
was independently checked using ordered BF16 FC2 terms. All formal cases
check output bits before and after each timing block. All eighteen per-rank workload
payload comparisons match across ten source/round manifests each.

## Measurement contract

Eager execution, five rounds with alternating source order, five warmups and
20 measured calls per source/case/cache/scope/round. Each reported primary
group has 100 samples. Warm and cold results remain separate. Cold L2 writes
512 MiB/rank, exceeding twice the observed 135,528,448-byte L2, and completes
eviction and host Gloo alignment outside timing.

- **Pipeline** is a CUDA-event span around the prebuilt compiled mega +
  separate TopkReduce callable, including their launch gap. It is not the
  isolated fused-kernel time.
- **Full** is wall time through staging, production validation/control, launch,
  owned output and completion. Off uses the public BF16-input FI API on both
  sides. On uses the public prequantized FI API versus a donor adapter that
  stages the same packed inputs into its native arena and returns an owned
  output. Donor has no equivalent FI public API. Their full-call ratio
  therefore includes a host-wrapper scope difference.
- For every iteration, select the slowest rank by that scope's primary metric;
  retain GPU, wall and enqueue values from that same rank. Enqueue is the
  interval through function return and may include blocking operations.
- Median is the primary statistic. The linked JSON also records p10/p90,
  per-sample CV, five round medians and their CV. Off passes only at ratio
  ≤1.05; baseline round-median CV >5% makes the comparison inconclusive.
  All rounds are retained. No favorable round selection is applied.
- Separate five-call Kineto captures in round zero provide kernel attribution.
  Primary timing samples are unprofiled. Mega and TopkReduce each have exactly
  five events per rank/cache/case. Profile durations are never added to an
  independently measured pipeline/full total.

## Off: F_BASE versus receiver

Times are medians in microseconds; positive change means slower. `No conclusion`
means baseline round-median CV exceeded 5%. Decisions use unrounded values.

| EP | T | Cache | Scope | F_BASE µs | Receiver µs | Change | Baseline round CV | Gate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 64 | cold | Full | 201.31 | 206.07 | +2.36% | 13.83% | No conclusion |
| 1 | 64 | cold | Pipeline | 90.67 | 98.50 | +8.63% | 6.84% | No conclusion |
| 1 | 64 | warm | Full | 212.48 | 217.62 | +2.42% | 1.41% | Pass |
| 1 | 64 | warm | Pipeline | 88.19 | 94.83 | +7.53% | 0.58% | Regression |
| 1 | 1024 | cold | Full | 239.46 | 241.11 | +0.69% | 2.07% | Pass |
| 1 | 1024 | cold | Pipeline | 136.74 | 141.76 | +3.67% | 1.68% | Pass |
| 1 | 1024 | warm | Full | 238.95 | 241.19 | +0.94% | 2.76% | Pass |
| 1 | 1024 | warm | Pipeline | 132.67 | 136.88 | +3.17% | 1.38% | Pass |
| 2 | 64 | cold | Full | 257.83 | 266.29 | +3.28% | 4.12% | Pass |
| 2 | 64 | cold | Pipeline | 127.17 | 132.29 | +4.03% | 1.50% | Pass |
| 2 | 64 | warm | Full | 261.75 | 266.37 | +1.77% | 1.61% | Pass |
| 2 | 64 | warm | Pipeline | 124.54 | 130.78 | +5.01% | 1.52% | Regression |
| 2 | 1024 | cold | Full | 300.98 | 311.78 | +3.59% | 5.53% | No conclusion |
| 2 | 1024 | cold | Pipeline | 174.24 | 182.53 | +4.76% | 4.08% | Pass |
| 2 | 1024 | warm | Full | 300.79 | 313.51 | +4.23% | 4.98% | Pass |
| 2 | 1024 | warm | Pipeline | 170.94 | 179.89 | +5.23% | 5.06% | No conclusion |
| 4 | 64 | cold | Full | 277.27 | 237.89 | -14.20% | 91.31% | No conclusion |
| 4 | 64 | cold | Pipeline | 142.77 | 126.29 | -11.54% | 107.41% | No conclusion |
| 4 | 64 | warm | Full | 276.26 | 244.88 | -11.36% | 143.67% | No conclusion |
| 4 | 64 | warm | Pipeline | 143.58 | 132.66 | -7.61% | 11.23% | No conclusion |
| 4 | 1024 | cold | Full | 329.78 | 301.28 | -8.64% | 67.00% | No conclusion |
| 4 | 1024 | cold | Pipeline | 202.16 | 188.69 | -6.66% | 75.84% | No conclusion |
| 4 | 1024 | warm | Full | 329.30 | 312.39 | -5.14% | 39.22% | No conclusion |
| 4 | 1024 | warm | Pipeline | 188.93 | 185.78 | -1.67% | 9.42% | No conclusion |

All eight EP4 off comparisons are inconclusive: baseline round-median CV
ranges from 9.42% to 143.67%. The lower receiver medians cannot establish
a speedup under this level of baseline variation.

## On: fixed donor versus receiver

All rows use EP4/N2R2, T32. The full-call comparison measures different host
wrappers as defined above; its approximately 18× ratio is **not** a kernel
slowdown. Pipeline regression and profiler kernel durations are separate
observations. This measurement does not establish their causes.

| Cache | Scope | Donor µs (p10–p90) | Receiver µs (p10–p90) | Ratio | Round CV donor / receiver |
| --- | --- | ---: | ---: | ---: | ---: |
| cold | Full | 400.37 (366.13–456.13) | 7216.52 (6274.70–7874.65) | 18.025× | 7.22% / 1.35% |
| cold | Pipeline | 331.44 (310.01–376.72) | 372.66 (322.66–401.00) | 1.124× | 5.33% / 7.94% |
| warm | Full | 380.63 (347.23–440.47) | 6876.83 (5515.18–7490.24) | 18.067× | 9.24% / 7.57% |
| warm | Pipeline | 315.76 (300.92–345.09) | 375.26 (335.99–396.50) | 1.188× | 2.21% / 3.82% |

## Separate kernel profiles

Medians of five per-call critical-rank durations in microseconds. The fused
mega duration includes its internal communication and waits. These short
profile samples support attribution, not the 100-sample primary gate.
EP4 off baseline profiles also contain millisecond-scale durations; they
do not establish a speedup while that baseline is unstable.

| Mode | EP | T | Cache | Fused control → receiver µs | Reduce control → receiver µs |
| --- | --- | --- | --- | ---: | ---: |
| off | 1 | 64 | cold | 33.888 → 33.312 | 1.888 → 1.888 |
| off | 1 | 64 | warm | 28.000 → 27.744 | 1.888 → 1.888 |
| off | 1 | 1024 | cold | 77.282 → 76.575 | 3.936 → 3.936 |
| off | 1 | 1024 | warm | 71.649 → 70.241 | 3.904 → 3.904 |
| off | 2 | 64 | cold | 46.626 → 42.593 | 1.888 → 1.856 |
| off | 2 | 64 | warm | 86.881 → 43.649 | 1.888 → 1.888 |
| off | 2 | 1024 | cold | 95.394 → 94.880 | 4.032 → 3.936 |
| off | 2 | 1024 | warm | 89.792 → 92.224 | 3.904 → 3.872 |
| off | 4 | 64 | cold | 2086.303 → 50.562 | 1.984 → 2.016 |
| off | 4 | 64 | warm | 2958.963 → 89.697 | 2.016 → 2.016 |
| off | 4 | 1024 | cold | 3345.742 → 107.808 | 4.064 → 4.096 |
| off | 4 | 1024 | warm | 133.826 → 107.394 | 3.999 → 4.001 |
| on | 4 | 32 | cold | 288.579 → 372.994 | 3.616 → 3.584 |
| on | 4 | 32 | warm | 244.610 → 310.243 | 3.456 → 3.744 |

## Coverage, failures and reproducibility

Accepted coverage: 40 source/round points, 110 per-rank manifests, 56 primary
groups, 5,600 critical-rank samples, 28 comparisons, 28 attributed profile
groups and 72 per-rank traces. Output bit checks and payload identity pass.

- `3577460`: failed initial on smoke due to a benchmark host-wrapper mistake
  treating the compiled callable's integer status as an output tensor. Its
  independently validated fixtures were retained. No timing from this job is
  accepted. Versioned program and failure record remain in the run root.
- `3577625`: replacement on job, completed exit 0; all five paired rounds accepted.
- `3577461`: original off job reached its 15-minute Slurm limit. All EP1/EP2
  paired rounds completed and are accepted; partial EP4 is retained and excluded.
- `3577696`: fresh complete EP4 paired run, completed exit 0. No samples are
  pooled across its allocation and the partial EP4 allocation.

Raw run root:
`/lustre/fsw/coreai_devtech_all/qiangf/ncclGIN/port/migration-evidence/blackwell-perf-20261102/`.
It retains PLAN.md, expected-matrix.json, immutable launchers, failed/partial
runs, fixture payloads, module/source hashes, rank logs, JSONL and Kineto traces.
Accepted on root is `on-retry-v2/`; accepted EP4 root is `off-ep4-repeat/`.
[Artifact hashes](evidence/blackwell-performance-artifacts.sha256) bind accepted manifests, samples, traces, fixtures
and launch provenance. `donor-source-hashes.json` independently verifies 49
donor Python files against fixed-commit Git blob IDs.

- [Complete statistics, configuration and provenance](evidence/blackwell-performance-summary.json).
- [Critical-rank samples (CSV, gzip), including corresponding GPU/wall/enqueue times](evidence/blackwell-performance-samples.csv.gz).
- [Benchmark program](../../../benchmarks/moe_ep/bench_legacy_nvfp4_gin.py).
- [Aggregation program](../../../benchmarks/moe_ep/report_legacy_nvfp4_gin.py).

Regenerate with Python 3 (the aggregator needs no GPU or third-party packages):

```bash
python3 benchmarks/moe_ep/report_legacy_nvfp4_gin.py /path/to/blackwell-perf-20261102 \
  --on-root /path/to/blackwell-perf-20261102/on-retry-v2 \
  --ep4-root /path/to/blackwell-perf-20261102/off-ep4-repeat --require-complete
```

These results qualify the listed eager workloads on GB300. They do not cover
other Blackwell models, Rubin, broader shapes, or CUDA Graph performance.
Correctness completion remains recorded separately in [validation.md](validation.md).
Observed regressions remain unresolved; no overall performance pass is claimed.
