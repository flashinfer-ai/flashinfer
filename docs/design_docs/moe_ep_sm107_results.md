# Rubin MegaMoE results

## Setup

Measured September 18, 2026 on the refreshed generic inference kernel:

- FlashInfer: `5bd5aeef60c44a99341e6b6a183968d73bf582e7`, clean before/after both perf jobs.
- Vendor: `1667b47a3c911ecade464ab524baf192a0bf5962`; see [VENDOR.md](../../flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/VENDOR.md).
- EP4 perf: four SM107 GPUs, reported as NVIDIA Graphics Device, 212 SMs and
  278.5 GiB each; NV36 connectivity between every pair. One ARM64 node,
  `hecate0252`, initially idle for both series.
- Prepared ARM PyTorch image SHA-256:
  `074e75e470a71833bd68e7280d5192047513e23599169270026efc87bc802c6e`.
- Python 3.12.3; Torch `2.14.0a0+4fdf77b940.nvinternal.rubin.0.8dev`;
  CUDA 13.5 (nvcc 13.5.7), driver 620.43;
  CuTe DSL `4.8.0a0+20260821210818.ac70faa`;
  cuda-python `13.4.2.dev6+g1c864bfe`; NVSHMEM4py 0.3.1,
  NVSHMEM 3.8.0, NCCL 2.31.2; `NVSHMEM_REMOTE_TRANSPORT=none`.

The [qualification guide](moe_ep_sm107_qualification.md) describes the contracts
and rerun commands. The [tuning guide](../../flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/TUNING.md)
defines the timing boundaries and exact benchmark options.

## Tests and benchmark runs

| Evidence | Result |
| --- | --- |
| Native single-GPU, Hecate job 601865 | 50 passed, zero failures/errors/skips |
| Native EP4, Hecate job 605150 | 16 cases on each of four ranks passed, zero failures/errors/skips; Slurm `0:0` |
| Compute/eager, L2-flushed, job 605266 | 84/84 records passed; 34m42s allocated; Slurm `0:0` |
| Kernel/forward × eager/graph, unflushed, job 605267 | 336/336 records passed; 2h10m15s allocated; Slurm `0:0` |

Correctness covers NVFP4 and both MXFP8 formats. Performance covers NVFP4
activations/weights with BF16 combine, using separate reduction (`bf16`) and
in-kernel atomic reduction (`ikr`). IKR is an explicit nondeterministic option.
All 420 benchmark records passed whole-output finiteness and sampled checks
against a collective Torch oracle over the actual quantized bytes. Maximum
relative L2 error was 0.0040795 (reference) and 0.0040787 (characterization),
below the existing 0.06 threshold. This does not measure quantization loss
against an unquantized BF16 model.

## Measurement protocol

H=7168, EP4, with I=2048/E=256/top-k=8 and I=3072/E=384/top-k=6. Both use
8/64/512/1024/2048/4096/8192 tokens per rank, capacity
`max(64, next_power_of_two(tokens))`, Gaussian-score top-k, the historical
Blackwell input profile, seed 0, and per-size heuristic knobs.

Each of 140 configurations ran in three fresh processes with **20 warmups and
50 timed iterations per rank**. Tables show microseconds, taking the median
of each statistic across the three repetitions. The [measurement CSV](moe_ep_sm107_results.csv)
retains each statistic's minimum and maximum across repetitions as well.
CUDA-event timing, output ownership, cache policy, and rank aggregation are
part of the protocol; do not subtract flushed compute from unflushed
kernel/forward to estimate overhead.

## Compute reference — eager, L2-flushed

Metric: rank-zero p50 (`p50_rank0_us`), for historical benchmark-table comparisons.

| Tokens/rank | 256E BF16 | 256E +IKR | 384E BF16 | 384E +IKR |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 164.8 | 150.3 | 208.9 | 197.1 |
| 64 | 215.6 | 195.3 | 376.6 | 358.8 |
| 512 | 306.1 | 211.9 | 448.7 | 376.8 |
| 1024 | 437.7 | 258.9 | 524.4 | 393.3 |
| 2048 | 704.1 | 361.2 | 705.2 | 444.9 |
| 4096 | 1,273.5 | 609.5 | 1,083.8 | 582.2 |
| 8192 | 2,449.0 | 1,132.9 | 1,848.9 | 855.8 |

## Kernel/forward characterization — unflushed

Metric: maximum of per-rank p50s (`max_rank_p50_us`). Each cell is **BF16 / +IKR**.

### 256 experts

| Tokens/rank | Kernel eager | Kernel graph | Forward eager | Forward graph |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 157.8 / 143.6 | 153.7 / 142.5 | 523.3 / 500.5 | 260.8 / 249.5 |
| 64 | 204.0 / 188.0 | 200.0 / 187.0 | 528.8 / 507.1 | 380.6 / 364.0 |
| 512 | 270.8 / 204.8 | 265.4 / 203.1 | 1,263.4 / 1,166.6 | 1,206.7 / 1,114.4 |
| 1024 | 432.0 / 252.0 | 426.5 / 251.6 | 2,213.9 / 2,037.5 | 2,147.5 / 1,973.2 |
| 2048 | 697.1 / 353.9 | 693.0 / 352.7 | 4,125.8 / 3,782.8 | 4,060.1 / 3,721.3 |
| 4096 | 1,261.5 / 598.2 | 1,256.3 / 597.7 | 8,026.6 / 7,361.3 | 7,960.8 / 7,297.9 |
| 8192 | 2,425.6 / 1,112.0 | 2,421.3 / 1,108.8 | 15,903.8 / 14,593.0 | 15,843.5 / 14,532.8 |

### 384 experts

| Tokens/rank | Kernel eager | Kernel graph | Forward eager | Forward graph |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 202.5 / 189.9 | 198.8 / 189.3 | 532.3 / 512.8 | 306.4 / 296.9 |
| 64 | 366.2 / 352.0 | 362.4 / 351.2 | 585.6 / 571.0 | 541.4 / 528.4 |
| 512 | 419.7 / 369.4 | 414.9 / 368.2 | 1,410.7 / 1,338.2 | 1,347.7 / 1,278.8 |
| 1024 | 507.4 / 385.7 | 502.4 / 385.1 | 2,293.7 / 2,163.2 | 2,234.9 / 2,106.6 |
| 2048 | 698.1 / 437.8 | 693.8 / 436.4 | 4,126.1 / 3,865.8 | 4,062.9 / 3,804.8 |
| 4096 | 1,071.4 / 572.8 | 1,066.3 / 571.3 | 7,847.6 / 7,346.0 | 7,766.4 / 7,268.4 |
| 8192 | 1,828.5 / 834.8 | 1,822.5 / 833.5 | 15,304.6 / 14,310.7 | 15,241.0 / 14,252.4 |

Source: [measurement CSV](moe_ep_sm107_results.csv); includes min/max across repetitions and both latency statistics. All 420 records passed benchmark numerical and reporting checks.

## Observations and limitations

IKR reduces latency across all measured reference points. The full public
forward path includes Torch input staging: at 8192 tokens/rank, graph forward
is 14.3–15.8 ms versus 0.8–2.4 ms for the graph kernel. This is measured API
cost, not a serving benchmark; fused staging remains a follow-up.

For historical context, the [SM100 tuning tables](../../flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/TUNING.md)
at the measured FlashInfer revision report GB200 EP4 results (July 22 main
256E table and July 21 v4-pro 384E table). Comparing the same variant using
rank-zero p50, geometry, token count, and compute/cache protocol, IKR is
1.56–2.20 times faster on these Rubin runs. Separate-reduction 256E is faster
at 8–512 tokens but slower from 1024 onward, reaching 29% higher latency at
8192; 384E is faster at every published matching point, approaching parity
at the largest sizes. The 384E historical table has no 1024/4096 rows.
Different kernel revisions, compilers, heuristics, and hardware prevent
attributing these ratios to hardware alone. No same-Rubin before/after
optimization result or new Blackwell run is claimed.

The current refreshed-export native evidence covers single-GPU and EP4 on the
recorded runtime. EP2/EP8, native sanitizers, a clean installed wheel, and the
minimum public compiler build are not qualified by these runs. GenPhase,
SiTU, `combine_nvfp4`, and `combine_mxfp8` remain future integration work.
DeepGEMM was excluded from the agreed comparison.

## Raw evidence

The canonical campaign is on Equator, explicitly collected and hash-verified
from Hecate; both use the following path on separate filesystems:

`/home/akaashp/scratch/flashinfer-artifacts/runs/rubin-perf/20260918T012411Z-ep4-5bd5aeef`

Its `PLAN.md`, submission scripts, `final-evidence/` per-rank samples/logs,
`manifest.json`, environment/topology captures, UTC accounting, and historical
comparison CSV retain the reproducibility record. All 912 collected files
passed hash verification. The total perf allocation was 10.997 GPU-hours,
including setup. No failed points were dropped or retried; successful jobs'
Torch NCCL teardown warnings remain in the raw logs. This source report and
its CSV are the portable summary; the raw artifacts are kept outside source Git.
