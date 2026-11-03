# Blackwell performance verification

Status: submitted, no timing conclusion yet. The user's explicit Blackwell
performance request supersedes the earlier local-performance waiver and the
donor's Rubin-only performance rule for this experiment. The fixed DeepSeek
CI guard and its timing/cache policy are unchanged.

## Sources and matrix

Compare F_BASE `e74a1677ab95a1b97aa1c6b6a80b3cf5c0ebac34` with the migrated
legacy backend in off mode, and MR !100 donor
`f863d7540614bd546a42867c21adbf51d573061a` with the migrated backend in on mode.
No kernel source changes or tuning are part of this measurement.

| Mode | EP / topology | Tokens/rank | H / public intermediate / total experts / K |
| --- | --- | --- | --- |
| Off | EP1, EP2, EP4, one GB300 host | 64, 1024 | 2048 / 1024 / 8 / 4 |
| On | N2R2, two independent GB300 NVLink domains | 32 | 7168 / 1536 / 48 / 4 |

Both sides of each comparison run on the same exclusive allocation. Input
seeds are fixed and quantized payload hashes must agree. On mode uses one
shared fixture of kernel-ready tensors and an independently ordered-BF16
validated expected output. Check output bits before and after timing blocks.

## Timing contract fixed before measurement

Eager execution only. Each case/source/cache/scope has five alternating rounds,
five warmups and twenty measured calls/round. Cold L2 writes 512 MiB/rank,
synchronizes and aligns ranks before timing; verify that this exceeds twice
the actual device L2 size. Warm and cold results stay separate.

- **Raw GPU pipeline:** CUDA events around the prebuilt compiled callable.
  This includes the mega kernel, separate TopkReduce and their launch gap.
  It is not an isolated fused-kernel duration.
- **Full operator:** wall time through staging, validation/control, launch,
  owned output and completion, with per-rank GPU-span/enqueue samples retained.
  Off uses the public BF16-input FI API on both sides. On uses the public
  prequantized FI API; the native donor full adapter stages the same packed
  inputs into its arena, calls the unchanged runner's compiled target and
  copies into an owned output. The donor has no corresponding FI public API;
  this host-wrapper difference is part of the reported comparison.
- **Fused kernel:** separate five-call Kineto traces in round zero. Keep mega
  and TopkReduce events separately. These profiler-derived values cannot be
  added to an independently measured total.

For each iteration choose the slowest rank for that scope's primary metric,
then report median, p10, p90, sample count and dispersion. Keep all per-rank
samples. The predeclared off gate is at most 5% median regression for each
raw/full and warm/cold case; baseline round-median CV above 5% makes the
comparison inconclusive. On ratios are descriptive, with no speedup gate.

## Execution artifacts

Driver jobs: `3577460` (on), `3577461` (off). Actual hardware, versions, clocks,
source/script hashes and selected kernel settings are recorded by each worker.
The jobs begin with smoke checks; a failed case is retained and not timed as
passing. No damaged communicator is reused.

Run root:
`/lustre/fsw/coreai_devtech_all/qiangf/ncclGIN/port/migration-evidence/blackwell-perf-20261102/`.
It contains the premeasurement PLAN, immutable launcher versions, per-rank
manifests/JSONL samples, fixtures, traces and the generated summary/CSV.

The measurement entry point is
[bench_legacy_nvfp4_gin.py](../../../benchmarks/moe_ep/bench_legacy_nvfp4_gin.py).
The selected checkout is supplied through PYTHONPATH in a fresh worker
process; donor measurements never import FI's raw source aliases.
