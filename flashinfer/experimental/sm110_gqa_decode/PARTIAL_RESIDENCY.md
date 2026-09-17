# Prepared decode: controlled KV reuse on Thor

## Scope and implementation

The prepared entry points add opt-in sustained decode to the existing SM110
backend. They preserve the original four shapes (FP16, Hq=32, Hkv=8, D=128):

| Batch / KV capacity | Valid lengths | Default route |
| --- | --- | --- |
| 1 / 64 | [1] | Original short |
| 4 / 256 | [64, 127, 191, 256] | N32 direct output, no global workspace |
| 1 / 1024 | [1024] | N64 KV-last, ten splits, fused merge |
| 1 / 4096 | [3968] | N32 three-slot pipeline, ten splits, fused merge |

QK and PV use `tcgen05.mma`, with TMA and TMEM on exact SM110a. The direct
B4 epilogue eliminates single-contributor partial stores, the completion ticket
and merge. Both long routes retain one kernel launch with an in-kernel final
merge. The kernel bodies here are the source-qualified schedules from the
study below, with mechanical symbol/include renames for this package.

The convenience `sm110_gqa_decode` API remains unchanged. Preparation is outside
the timed region and Graph capture; the new launch API reuses caller output,
views and per-instance workspace. These measurements are **GPU target latency**,
not public Python API latency or end-to-end model speedup.

## Frozen protocol

The source-qualified study completed on 2026-09-17, on aarch64 NVIDIA Thor
(compute capability 11.0, 20 SMs, 32 MiB L2), CUDA 13.4 and PyTorch
2.14.0a0+b2c75dd062.nvinternal.main. It is separate from the earlier CUDA 13.5
results in [RESULTS.md](RESULTS.md); absolute times are not pooled across images.

The original is pinned to PR #5052 head
`223e0706c9398ec628d87a8a93ed1548efcd7900`. XQA is a fixed specialization,
loaded directly (not the public XQA dispatcher), with cubin SHA-256
`259b035bf26b9dc92cd9682802933107e71f5d4171685cbf37c8b3278c441786`.
The original source hashes are preserved by the existing package manifest;
the added generated source hashes are in `csrc/prepared/manifest.json`.

Each cell uses identical inputs across original/candidate/XQA, separate outputs,
four nonidentical producer slots, q_scale=1, and fresh Q plus the final valid
K/V token for every sample. History and valid lengths remain fixed within a
cell. A sequential FP32 GEMM heater, instruction prime and 2x-L2 eviction sweep
run outside the target bracket. There is no concurrent heater.

- **cold:** update the producer, then evict.
- **producer_hot:** evict, then update Q and the valid tail token.
- **recent64 / recent256:** producer_hot plus reads of the last min(length, N)
  valid K/V tokens with default load policy.
- **full_kv:** producer_hot plus reads of all valid K/V.
- **same_arm_hot:** producer_hot plus repeated launches of the measured arm.

These are controlled footprints, not measured cache hit rates or guaranteed
residency. Cross-scenario ratios show sensitivity rather than a hardware-counter
attribution to a specific cache level.

Strict CUPTI activities time the whole target span, including any inter-kernel
gaps, with no event fallback. All targets here have one expected kernel and zero
observed gaps. Sample zero is retained in raw data but excluded unconditionally.
Six counterbalanced arm orders each retain 32 samples: 432 Graph cells. Observed
pre/post clocks match across arms at GPC 1.386 GHz / EMC 3.2 GHz; no privileged
clock lock is used. Equal endpoints do not establish in-kernel clock constancy.
A clock-rejected earlier run is excluded as a whole, without replacing cells.

Within each group, R=original/candidate and X=XQA/candidate use that group's
median latencies. A condition result is the median of its six paired ratios.
The primary score is the equal-weight geometric mean of the nine condition
results for B4/cap1024/cap4096 crossed with producer_hot/recent64/recent256.
It is not a ratio of marginal medians. The frozen targets were >=15% primary
reduction, >=10% on each long-shape partial aggregate, long hot XQA parity,
and <=5% regression for cold and B4/short guards.

## Results

The primary paired ratio is **1.615800996**, or **38.111191% lower latency**.
The cap1024/cap4096 partial aggregates improve **52.217038% / 16.734255%**.
Their hot candidate/XQA latencies are **11.144 / 13.376 us** and
**21.816 / 37.792 us**, with paired reductions **16.885627% / 42.240636%**
relative to XQA. The separate ordinary-launch crosscheck (162 cells, same nine
primary combinations, six groups x32) records **32.696199%** aggregate reduction
and **51.637262% / 16.915182%** on the long shapes.

Latency columns below are marginal medians of six group medians in microseconds.
R and X are medians of paired ratios, so need not equal ratios of those columns.
Negative delta means lower candidate latency. Asterisks mark primary cells.

| Shape | Condition | Original / candidate / XQA μs | R | Candidate Δ vs original | X | Candidate Δ vs XQA |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| cap64 | cold | 5.424 / 5.536 / 5.888 | 0.983551 | +1.672% | 1.060519 | -5.707% |
| cap64 | producer_hot | 5.408 / 5.512 / 5.856 | 0.981134 | +1.923% | 1.062412 | -5.875% |
| cap64 | recent64 | 5.416 / 5.328 / 6.176 | 0.993769 | +0.627% | 1.161468 | -13.902% |
| cap64 | recent256 | 5.248 / 5.472 / 6.184 | 0.980568 | +1.982% | 1.130164 | -11.517% |
| cap64 | full_kv | 5.248 / 5.328 / 6.216 | 0.982977 | +1.732% | 1.164596 | -14.133% |
| cap64 | same_arm_hot | 3.360 / 3.456 / 5.824 | 0.972222 | +2.857% | 1.685185 | -40.659% |
| b4_cap256 | cold | 31.848 / 13.304 / 36.848 | 2.393391 | -58.218% | 2.766232 | -63.850% |
| b4_cap256 | producer_hot * | 31.344 / 13.392 / 36.352 | 2.337548 | -57.220% | 2.714880 | -63.166% |
| b4_cap256 | recent64 * | 21.272 / 11.656 / 22.560 | 1.826810 | -45.260% | 1.935485 | -48.333% |
| b4_cap256 | recent256 * | 11.728 / 10.576 / 11.728 | 1.107274 | -9.688% | 1.108931 | -9.823% |
| b4_cap256 | full_kv | 11.728 / 10.536 / 11.776 | 1.113138 | -10.164% | 1.117535 | -10.517% |
| b4_cap256 | same_arm_hot | 11.264 / 9.696 / 11.360 | 1.161716 | -13.920% | 1.171617 | -14.648% |
| cap1024 | cold | 44.968 / 22.784 / 49.344 | 1.977141 | -49.422% | 2.168363 | -53.882% |
| cap1024 | producer_hot * | 44.808 / 21.608 / 49.152 | 2.077759 | -51.871% | 2.275161 | -56.047% |
| cap1024 | recent64 * | 42.256 / 20.416 / 45.400 | 2.070538 | -51.703% | 2.214005 | -54.833% |
| cap1024 | recent256 * | 35.384 / 16.600 / 37.320 | 2.130603 | -53.065% | 2.245305 | -55.463% |
| cap1024 | full_kv | 18.152 / 13.992 / 13.696 | 1.296741 | -22.884% | 0.978845 | +2.161% |
| cap1024 | same_arm_hot | 17.888 / 11.144 / 13.376 | 1.605170 | -37.701% | 1.203161 | -16.886% |
| cap4096 | cold | 112.200 / 94.208 / 112.192 | 1.191550 | -16.076% | 1.189179 | -15.908% |
| cap4096 | producer_hot * | 112.264 / 93.728 / 111.952 | 1.198699 | -16.576% | 1.194780 | -16.303% |
| cap4096 | recent64 * | 109.104 / 91.000 / 108.336 | 1.199381 | -16.624% | 1.189982 | -15.965% |
| cap4096 | recent256 * | 102.432 / 85.136 / 100.536 | 1.204852 | -17.002% | 1.180822 | -15.313% |
| cap4096 | full_kv | 65.520 / 22.632 / 38.120 | 2.896430 | -65.475% | 1.684490 | -40.635% |
| cap4096 | same_arm_hot | 59.680 / 21.816 / 37.792 | 2.735241 | -63.440% | 1.731321 | -42.241% |


## Variation and guards

All 14 Graph guard point estimates pass. Three short-shape group ranges crossed
the 5% regression boundary, triggering a separately declared full cap64
confirmation (six conditions, six groups x32, 108 cells). All six condition
medians pass in both receipts, and the three original range crossings clear in
the confirmation. One confirmation recent64 group retains a **6.542% regression**;
its median regression is **0.627% in both receipts**. The receipts remain
separate; observed ranges are not confidence intervals or an every-group gate.
The six-group Graph ratios, including the cap4096 producer_hot reversal, are:

| Shape / condition | R values, groups0–5 | R min / median / max |
| --- | --- | --- |
| cap64 / cold | 0.988473, 0.975460, 0.981595, 0.968481, 0.985507, 0.985591 | 0.968481 / 0.983551 / 0.988473 |
| cap64 / producer_hot | 0.939306, 0.984615, 0.985465, 0.982558, 0.976879, 0.979710 | 0.939306 / 0.981134 / 0.985465 |
| cap64 / recent64 | 0.993750, 0.993789, 0.985465, 1.043210, 0.991304, 0.994152 | 0.985465 / 0.993769 / 1.043210 |
| cap64 / recent256 | 0.975385, 0.981366, 0.979769, 0.918605, 0.982659, 1.000000 | 0.918605 / 0.980568 / 1.000000 |
| cap64 / full_kv | 0.924638, 0.981481, 1.055901, 0.991228, 0.984472, 0.973988 | 0.924638 / 0.982977 / 1.055901 |
| cap64 / same_arm_hot | 0.972222, 0.972222, 0.972222, 0.972222, 0.972222, 0.972222 | 0.972222 / 0.972222 / 0.972222 |
| b4_cap256 / cold | 2.508413, 2.389892, 2.401205, 2.392298, 2.375149, 2.394484 | 2.375149 / 2.393391 / 2.508413 |
| b4_cap256 / producer_hot | 2.330559, 1.197183, 2.366587, 2.344538, 2.322657, 2.371841 | 1.197183 / 2.337548 / 2.371841 |
| b4_cap256 / recent64 | 1.838134, 1.818681, 1.815068, 1.834938, 1.783083, 1.873626 | 1.783083 / 1.826810 / 1.873626 |
| b4_cap256 / recent256 | 1.075758, 1.105787, 1.294770, 1.109091, 1.108761, 1.105422 | 1.075758 / 1.107274 / 1.294770 |
| b4_cap256 / full_kv | 1.137821, 1.171975, 1.112121, 1.114155, 1.108761, 1.105740 | 1.105740 / 1.113138 / 1.171975 |
| b4_cap256 / same_arm_hot | 1.161716, 1.165017, 1.161716, 1.161716, 1.161716, 1.165017 | 1.161716 / 1.161716 / 1.165017 |
| cap1024 / cold | 2.040645, 2.018571, 1.923024, 1.973296, 1.967018, 1.980986 | 1.923024 / 1.977141 / 2.040645 |
| cap1024 / producer_hot | 2.085757, 2.083086, 2.059297, 2.072432, 2.059735, 2.132186 | 2.059297 / 2.077759 / 2.132186 |
| cap1024 / recent64 | 2.098580, 2.043478, 2.027735, 2.065934, 2.075142, 2.083333 | 2.027735 / 2.070538 / 2.098580 |
| cap1024 / recent256 | 2.130058, 2.160976, 1.931818, 2.131148, 2.118774, 2.154070 | 1.931818 / 2.130603 / 2.160976 |
| cap1024 / full_kv | 1.297483, 1.293850, 1.293044, 1.296000, 1.303100, 1.333333 | 1.293044 / 1.296741 / 1.333333 |
| cap1024 / same_arm_hot | 1.604017, 1.622642, 1.590327, 1.592011, 1.615051, 1.606322 | 1.590327 / 1.605170 / 1.622642 |
| cap4096 / cold | 1.191869, 1.545970, 1.191230, 1.191067, 1.192234, 1.188072 | 1.188072 / 1.191550 / 1.545970 |
| cap4096 / producer_hot | 1.202330, 1.199077, 1.204127, 1.195235, 1.198321, 0.761727 | 0.761727 / 1.198699 / 1.204127 |
| cap4096 / recent64 | 1.199431, 1.136548, 1.209454, 1.199331, 1.200204, 1.198453 | 1.136548 / 1.199381 / 1.209454 |
| cap4096 / recent256 | 1.204853, 1.196026, 1.200451, 1.671856, 1.204851, 1.223704 | 1.196026 / 1.204852 / 1.671856 |
| cap4096 / full_kv | 2.911994, 2.834157, 2.898161, 2.902128, 2.894700, 2.885067 | 2.834157 / 2.896430 / 2.911994 |
| cap4096 / same_arm_hot | 2.752768, 2.737344, 2.725146, 2.733138, 2.742670, 2.726038 | 2.725146 / 2.735241 / 2.752768 |


All cold outliers are retained. These are retained-sample min/median/max values
in microseconds, with excluded sample zero kept separately in raw observations.

| Shape / arm | Overall min / max μs | Group0–5 min/median/max μs |
| --- | --- | --- |
| cap64 / original | 4.256 / 11.840 | 4.320/5.488/10.528; 4.256/5.088/5.600; 4.320/5.120/10.016; 4.544/5.408/5.824; 4.544/5.440/11.840; 4.512/5.472/10.880 |
| cap64 / candidate | 4.384 / 11.328 | 4.640/5.552/7.040; 4.544/5.216/5.600; 4.384/5.216/6.336; 4.608/5.584/5.856; 4.608/5.520/8.768; 4.640/5.552/11.328 |
| cap64 / xqa | 5.792 / 10.784 | 5.792/5.888/6.400; 5.824/5.888/10.784; 5.792/5.920/6.272; 5.824/5.888/7.008; 5.792/5.840/10.464; 5.824/5.888/8.672 |
| b4_cap256 / original | 30.752 / 42.912 | 30.976/33.392/40.832; 31.424/31.776/35.840; 30.752/31.888/32.416; 31.008/31.808/34.880; 30.912/31.808/38.144; 31.136/31.952/42.912 |
| b4_cap256 / candidate | 12.832 / 19.712 | 12.896/13.312/14.336; 12.864/13.296/16.864; 12.928/13.280/14.112; 12.832/13.296/14.208; 13.088/13.392/19.712; 12.832/13.344/14.784 |
| b4_cap256 / xqa | 19.264 / 56.096 | 35.712/36.880/37.888; 35.712/36.992/37.696; 35.328/36.768/56.096; 19.264/19.728/20.864; 35.808/36.816/43.712; 35.232/36.880/43.392 |
| cap1024 / original | 23.392 / 67.136 | 41.440/46.592/51.904; 23.392/45.216/67.136; 44.128/44.768/50.304; 43.360/44.928/59.168; 43.232/44.848/66.720; 42.912/45.008/53.152 |
| cap1024 / candidate | 20.544 / 40.864 | 21.632/22.832/24.576; 21.280/22.400/33.696; 20.544/23.280/40.864; 21.568/22.768/28.544; 21.792/22.800/29.952; 21.856/22.720/31.904 |
| cap1024 / xqa | 20.352 / 57.728 | 20.352/22.064/27.008; 48.128/49.344/56.480; 47.521/49.344/50.112; 45.760/49.984/54.400; 48.256/49.472/57.728; 48.224/49.232/51.936 |
| cap4096 / original | 110.016 / 168.128 | 110.592/112.112/138.240; 110.912/146.064/168.128; 110.816/112.224/127.936; 110.624/112.208/128.288; 110.016/111.936/129.408; 110.528/112.192/132.576 |
| cap4096 / candidate | 90.976 / 156.032 | 92.704/94.064/105.184; 92.608/94.480/156.032; 92.256/94.209/98.336; 90.976/94.208/101.696; 92.384/93.888/103.712; 93.088/94.432/102.528 |
| cap4096 / xqa | 97.664 / 172.768 | 97.664/99.744/168.544; 108.960/112.368/143.136; 110.496/112.017/123.072; 110.784/111.664/121.952; 109.952/158.272/172.768; 106.816/112.528/131.200 |


The cap4096 candidate cold maximum is 156.032 us versus its 94.208 us marginal
median; original/XQA maxima are 168.128/172.768 us. B4 XQA cold group3 and cap1024
XQA cold group0 have lower latency regimes than their other groups. Matching
observed clocks does not remove this variation or establish equal cache state.

## Correctness, reproduction and timing boundary

The source-qualified implementation passed 84 independent FP32-oracle cases /
216 producer states at FP16 atol=rtol=1e-2, 13 dynamic Graph states, four stream
and caller-output ownership routes, and explicit B4 num_splits=1 fallback.
Coverage includes q_scale in {0.5, 1, 1.5}, multiple seeds and valid-length
boundaries from 1 through 4096 with nonzero random padding. Separate synccheck
and racecheck checks passed with zero errors/hazards, in 8.465/9.864 seconds,
each under a hard 20-second process-group timeout.

The fixed-XQA driver, cubin and full raw study receipts are not distributed in
this PR. The tables and protocol above describe the source-qualified study,
not a new measurement of this package revision. The actual public package integration was separately validated on SM110 with
CUDA 13.4: **42/42 focused tests passed** (zero skipped), all **84 oracle cases /
216 producer states** and **13 dynamic Graph states** passed through the new
public entry points, and the Graph example ran successfully. Source equivalence
checks passed for all six added CUDA/binding files. Separate bounded synccheck
and racecheck passed in **7.554 / 9.155 s**, with zero errors/hazards. The full
integration workflow took **83.582 s**; it did not rerun or replace the timing
study above.

Run the public prepared example and focused tests with:

```bash
python examples/experimental/sm110_gqa_decode_prepared.py --capacity 1024 --graph
pytest -q tests/experimental/test_sm110_gqa_decode.py tests/experimental/test_sm110_gqa_decode_prepared.py
```

Do not use the convenience API/SDPA benchmark to reproduce this timing boundary:
it includes allocations and host-visible length validation. A matching paired
benchmark must prepare all arms first and apply the conditioning, strict CUPTI
identity checks, observed-clock matching and aggregation specified above.

Physical source-study turnaround, from 2026-09-17 03:39 UTC through GPU
confirmation, was **12 h 54 min 13.045 s**. The accepted Graph workflow took
1009.641 s (1024.200 s managed), ordinary 369.587 s (382.221 s managed), and
confirmation 245.633 s (256.632 s managed). This includes experiment iteration
and scheduling; these workflow durations are separate from GPU kernel latency.
