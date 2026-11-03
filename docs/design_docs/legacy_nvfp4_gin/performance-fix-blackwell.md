# Blackwell performance correction

Updated 2026-11-03. The corrected receiver passes all **24 off-mode performance
gates** against the original FlashInfer legacy implementation. For internode
mode, repeated host validation and argument construction were reduced; a paired
experiment measured public-call latency falling from 6155.76 to 2208.60 us.
The final implementation uses two collective agreements per steady-state call,
instead of five. Its final standard warm public-call median is 1630.15 us.

The final internode raw pipeline medians are close to the native legacy donor,
but baseline round dispersion exceeds the predeclared 5% threshold. Those
comparisons remain **inconclusive**, not a performance pass. The native donor's
prepared adapter and FlashInfer's public API have different host-validation
scopes; their full-call ratio is not a kernel regression measurement.

## Fixed implementations and changes

- Off baseline: original legacy FlashInfer at
  `e74a1677ab95a1b97aa1c6b6a80b3cf5c0ebac34` (F_BASE).
- On baseline: actual native runner
  `pre_unification/moe_nvfp4_swapab/mega_runner.py` from MR !100 commit
  `f863d7540614bd546a42867c21adbf51d573061a`. Its 49 Python sources were
  checked against fixed git blobs. No next provider is used.
- Final receiver: `74c0362878cb6dd5d0c90c346c7b420848563f1d`.
  Corrections are `efb7b3d29`, `e930de841`, `38f650e63`, and `74c036287`.
  Runtime/backend identity assertions supplement source-path manifests.

The receiver reuses session-bound GIN launch arguments, combines duplicate
mandatory validation, and uses a fixed CPU header agreement on Gloo-backed
default groups. Errors or signature mismatches retain the detailed collective
diagnostic path. NCCL-only groups retain the original agreement transport.
Ordinary off cases compile the fixed donor's existing `launch_without_gin`
entry; persistent finalization and FC1 per-token scaling retain the full entry.
Eager GIN launch arguments bind the current stream at execution; off/graph
launches keep stream-specific caching. Current-device mismatch is rejected
collectively before staging.

No vendor device arithmetic, communication protocol, tolerance or tactic was
changed by these corrections. All 17 device-hunk ledger hashes still match.
In paired job3577960, before/after PTX and cubins match byte-for-byte on all
four ranks; that result applies to this experiment, not every compiler build.

## Measurement contract

Exclusive GB300/SM103, 152 SM/device, aarch64; image
`nvcr.io#nvidia/pytorch:26.07-py3`, Torch
`2.13.0a0+9186a08b2c.nv26.07`, CUDA13.3, DSL4.6.2, nccl4py0.3.1,
NCCL2.30.7, driver580.173.10. Clock snapshots are retained; clocks were not
locked. On-mode jobs use two distinct NVLink domains, two ranks per host,
with runtime LSA/rail checks. The user explicitly authorized Blackwell tests,
superseding the donor's Rubin-only measurement instruction for this experiment.
The DeepSeek CI guard is unchanged.

Off: EP1/2/4, T64/1024, H2048, public intermediate1024, E8, K4.
On: N2R2, T32, H7168, public intermediate1536, E48, K4.
Five alternating source rounds, five warmups, 20 measured calls per group per
round: 100 samples/group. Cold runs evict 512 MiB/rank outside timing;
warm/cold remain separate. The primary sample is the maximum rank per
iteration; wall/enqueue/device values come from that same critical rank.

Raw pipeline uses CUDA events around the compiled mega kernel, separate
TopkReduce, and intervening launch gap. It is not an isolated fused-kernel
duration. Full operator uses completed wall time, including host validation,
staging and allocation. Five-call profiler traces follow primary measurements
and validate mega/reducer counts; profiler timings are separate evidence.
Every accepted group passes exact output checks and per-rank input-payload
identity. The gate is median ratio <=1.05 only when baseline round-median
CV<=5%; otherwise the result is inconclusive.

The on performance fixture uses seeded random scores and top-k routing,
**without exact balancing**. This corrects the initial plan's description of
balanced routing: 512 route slots span all 48 experts, with 4–23 slots/expert.
All providers use identical tensors. The separately qualified balanced,
sparse-poison and empty-rank correctness fixtures are unchanged. The full
histogram is retained in `evidence/blackwell-fix-routing.json`.

## Off results

Times are median us, baseline → receiver. Every table entry passes its gate;
the two scopes and two cache modes give 24 comparisons in total.

| EP | T | Warm raw | Cold raw | Warm full | Cold full |
| --- | --- | --- | --- | --- | --- |
| 1 | 64 | 91.44 → 87.50 | 95.68 → 90.91 | 211.91 → 207.97 | 202.06 → 199.99 |
| 1 | 1024 | 132.40 → 127.07 | 136.98 → 131.42 | 237.33 → 234.87 | 238.56 → 231.43 |
| 2 | 64 | 106.98 → 101.52 | 106.46 → 102.11 | 226.71 → 217.63 | 217.39 → 212.37 |
| 2 | 1024 | 152.82 → 146.02 | 156.82 → 152.78 | 264.48 → 256.95 | 259.99 → 254.90 |
| 4 | 64 | 129.73 → 124.43 | 130.90 → 124.03 | 249.62 → 241.95 | 246.21 → 240.35 |
| 4 | 1024 | 184.62 → 181.89 | 186.27 → 180.62 | 299.97 → 293.81 | 296.12 → 287.70 |

Accepted roots: EP1 job3578010/theia0248; EP2 job3578087/theia0154;
EP4 job3578088/theia0095. There are 48 groups, 4800 critical-rank samples,
70 rank manifests and 24 rank/cache profile groups. EP1 measured `e930de841`;
EP2/4 measured `38f650e63`. The final patch changes internode stream/device
handling; the measured off path is unchanged, with final compatibility tests
run again on `74c036287`.

The measurement environment matters. A controlled EP4 experiment alternated
the unchanged F_BASE with shared CPUs0–31 versus disjoint GPU-local sets
0–7,8–15,72–79,80–87, with both layouts covering32 distinct worker CPUs.
Raw baseline round CV fell from6.31/16.85/88.38/40.03% to
2.42/3.09/0.97/2.01% for T64warm/cold,T1024warm/cold. This supports the joint
placement recipe as a cause of launcher instability in that experiment.
It does not isolate NUMA locality from CPU partitioning or identify a driver
defect. CPU binding is a benchmark recipe, not a correctness requirement.

## Internode results and remaining uncertainty

Final job3578176, theia0048/theia0115, compares two resident provider processes
on the same GPUs, serializing provider turns. Workers bind to GPU-local CPUs
before Torch/CUDA initialization. All primary samples precede profiling.
Standard five-warmup and separate 200-call conditioning protocols are not pooled.

| Protocol / cache | Native raw us | Receiver raw us | Change | Baseline round CV | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Standard warm | 350.37 | 349.31 | -0.30% | 6.38% | Inconclusive |
| Standard cold | 363.98 | 356.43 | -2.07% | 6.39% | Inconclusive |
| Conditioned warm | 359.44 | 319.79 | -11.03% | 8.51% | Inconclusive |
| Conditioned cold | 373.90 | 321.76 | -13.95% | 9.16% | Inconclusive |

Standard public full-call medians are1630.15 us warm and1656.86 us cold;
native prepared-adapter medians are402.74 and404.10 us. The receiver retains
collective API validation absent from that adapter. This remaining host cost
is explicit; the implementation does not reach native-adapter full latency.
The same-allocation earlier host correction measured6155.76→2208.60 us in
3577960, with identical device binaries. Comparing that old number directly
with the final1630.15 us across allocations is not a paired speedup estimate.

The initial +18.8% raw ratio did not reproduce consistently: unchanged code
reversed direction on another allocation, and paired host corrections did not
change device binaries. GPU launch/arrival timing contributes to some spans;
one old trace has6.3 ms start skew but0.1 us end skew between peers. The exact
cause of all N2R2 variation remains unresolved. Local CPU placement alone did
not eliminate it in3578114 or3578176. No unconditional intrinsic device
slowdown, or unconditional absence of one, is established by this record.

## Correctness and retained failures

Final job3578176 COMPLETED with exit0. GPU qualification passes17 strong/public tests per rank, including
bitwise protocol/byte checks, independent math, rank-local error recovery,
two sessions, capture rejection, actual stream-handle observation and
wrong-current-device recovery. The old receiver38f650e63 is first reproduced
rejecting a one-rank stream change; the corrected receiver passes it.
Off finalizer and workspace/cache/graph checks run on the final source too.

Failures remain preserved:3577867 failed the artifact extractor before timing;
3578114 completed1600 validated measurements but failed afterward when its
wrapper called a nonexistent main function, so its correctness phase did not
run. Neither whole job is relabeled successful. Initial negative and noisy
results remain in [the historical report](performance-blackwell.md) and
[checkpoints03–05](checkpoint-03.md). Profiling deferral alone did not remove
noise. See [investigation](investigation.md) for hypotheses PH1–PH5 and all
counterexamples.

## Evidence and reproduction

Raw root:
`/lustre/fsw/coreai_devtech_all/qiangf/ncclGIN/port/migration-evidence/blackwell-fix-20261102/`.
Compact evidence includes [off statistics](evidence/blackwell-fix-off-summary.json),
[on statistics](evidence/blackwell-fix-on-summary.json),
[paired host correction](evidence/blackwell-fix-paired-host-summary.json),
[qualification status](evidence/blackwell-fix-qualification.json), and the
[1356-artifact SHA-256 index](evidence/blackwell-fix-artifacts.sha256).
Compressed critical-rank samples are alongside these summaries.
Raw fixtures, source snapshots, launchers, all rank samples, traces and compiler
artifacts remain in the raw root rather than being copied into the repository.

```bash
python3 benchmarks/moe_ep/report_legacy_nvfp4_gin.py \
  ../migration-evidence/blackwell-fix-20261102/off-final-v1/ep12 \
  --ep2-root ../migration-evidence/blackwell-fix-20261102/affinity-final/ep2/pinned \
  --ep4-root ../migration-evidence/blackwell-fix-20261102/affinity-final/ep4/pinned \
  --off-only --require-complete
python3 benchmarks/moe_ep/report_legacy_nvfp4_gin_resident.py \
  ../migration-evidence/blackwell-fix-20261102/final-confirmation/job-3578176
```

The useful outcome is a qualified off correction and a measured reduction in
public GIN host overhead. Internode performance promotion remains limited by
baseline variance and the documented public/native API scope difference.
No additional geometry, graph support or kernel tuning is implied.
