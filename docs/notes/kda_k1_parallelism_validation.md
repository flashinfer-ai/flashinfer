# CAKE KDA K1 parallelism validation

This report records the upstream-readiness checks for the SM100-family owner/helper
kernel. It is intentionally separate from an upstream pull request: no PR has
been opened from this branch.

## Validated revision and environment

- Base source revision before this retuning diff:
  `d651fad743d2dca713f4bddeeb0c422cb7f1d06e`
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
| Targeted JIT and recurrent-KDA tests, B200 | 109 passed, 3 warnings |
| Targeted JIT and recurrent-KDA tests, B300 | 86 passed |
| C8 and C4 output versus FP32 reference | Passed |
| C8 and C4 recurrent state versus FP32 reference | Passed |
| Owner/helper output/state versus CAKE-M128 | Bitwise identical for all measured shapes |
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
| C4, packed varlen 2304/1792 | `ERROR SUMMARY: 0 errors` | `ERROR SUMMARY: 0 errors` |

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

### Four-head support

Four-head inputs reuse the generated eight-head beta TMA box by padding only
the beta staging tensor; task indexing, the owner/helper grid, output, and
recurrent state retain the real head count of four. Fixed and packed-varlen
H=4 owner/helper routes are bitwise identical to CAKE-M128 for output and
final recurrent state. Both M64 and M128 also pass the FP32-reference checks.
The targeted FlashKDA JIT and recurrent-KDA suite passed 109 tests on B200.
Focused H=4 C4/D15 runs passed Compute Sanitizer 2025.3.1 memcheck and
synccheck with `ERROR SUMMARY: 0 errors`.

A two-round cold-L2 B200 sweep used the same benchmark method with C4/D15 and
C4/D30 forced routes:

| T | Automatic route | Speedup vs `min(M64, M128)` |
| ---: | --- | ---: |
| 1024 | M64 fallback | 1.001x |
| 2048 | C4/D15 | 1.066x |
| 4096 | C4/D15 | 1.145x |
| 8192 | C4/D15 | 1.185x |

At T=1024, forced C4/D15 reached only 0.973x of the oracle, confirming that
the existing average-length threshold of 2048 remains appropriate for H=4.
The result JSON SHA256 was
`445d619caf5d0c3711060ae6503f8e9f1b0e0453e6d62c8b82efb4027465fff7`.

## B200 packed-varlen validation

Packed-varlen dispatch was enabled only after an independent B200 sweep. The
router uses `total_tokens // num_sequences` as a host-known average and does
not inspect device-resident `cu_seqlens`. Sequences are still consumed using
their real offsets and may have arbitrary, non-multiple-of-32 lengths.

```bash
python benchmarks/bench_kda_k1_parallelism.py \
  --varlen-profiles 8192 4096,4096 \
    4096,3072,2048,1024 8192,512,256,256 \
    1024,1024,1024,1024 \
  --num-heads 8 \
  --warmup-ms 10 --bench-ms 30 --state-rotations 2048 \
  --measurement-rounds 4 \
  --forced-configs 4:15 4:30 4:45 8:35 \
  --json k1_b200_varlen_sweep.json
```

Every automatic and forced owner/helper route was bitwise identical to
CAKE-M128 for output and final recurrent state before timing. Results use
cold-L2 CUDA-event timing
and report against `min(CAKE-M64, CAKE-M128)`:

| Sequence lengths | Tasks | Average T | Auto route | Speedup |
| --- | ---: | ---: | --- | ---: |
| 8192 | 8 | 8192 | C4/D15 | 1.184x |
| 4096, 4096 | 16 | 4096 | C4/D30 | 1.148x |
| 4096, 3072, 2048, 1024 | 32 | 2560 | C4/D30 | 1.131x |
| 8192, 512, 256, 256 | 32 | 2304 | C4/D30 | 1.192x |
| 1024, 1024, 1024, 1024 | 32 | 1024 | M128 fallback | unchanged from public packed baseline |

The uneven and high-skew profiles show that helpers overlap K1 production at
chunk granularity; K2 does not wait for a batch-wide K1 phase. C8 was not
selected: it was 0.605x and 0.731x of the oracle for the two- and four-sequence
balanced/skewed profiles, while C4 remained profitable. The result JSON SHA256
was `79208f48a7164f317b3e98e3fb348bf0de955a27e41e5f0cfcbd911e36e7cb51`.

Packed B300 input remains on the pre-existing M128 route. That is a validation
boundary, not a claim that the technique is B200-specific.

The packed-varlen bitwise test for sequence lengths 2304/1792 also passed
memcheck and synccheck with Compute Sanitizer 2025.3.1 from the CUDA 13.0.85
sanitizer package. Both runs reported `ERROR SUMMARY: 0 errors`.

After removing the unreachable global-pool/C2 code paths, the same B200 test
set passed again. A two-round cold-L2 confirmation measured 1.203x for profile
8192 and 1.131x for profile 4096/3072/2048/1024 versus the per-shape M64/M128
oracle. The cleaned kernel also passed the packed-varlen memcheck and synccheck
runs above.

## Cluster-size tuning sweep

C4 and C8 were compared by an explicit forced-route sweep rather than by the
producer-capacity model alone. Each shape was measured in four rounds with a
deterministically shuffled route order to reduce clock and route-order bias:

```bash
python benchmarks/bench_kda_k1_parallelism.py \
  --sequence-lengths 1024 2048 4096 8192 \
  --num-heads 8 16 24 32 \
  --warmup-ms 10 --bench-ms 30 --state-rotations 4096 \
  --measurement-rounds 4 \
  --forced-configs 4:15 4:30 4:45 8:35 8:70 \
  --json k1_cluster_depth_sweep.json
```

The full 16-shape result JSON SHA256 was
`97d6a4836a82de3ef712da94e2da6cce2aa554ba429c79dfdd05c3b343d938bf`.
All forced owner/helper routes were bitwise identical to CAKE-M128 for output
and recurrent state before timing.

For the only region previously assigned to C8, C4/D15 was faster at every
enabled sequence length:

| B | H | T | C4/D15 | C8/D35 | C4 advantage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 2048 | 0.119808 ms | 0.120000 ms | 1.002x |
| 1 | 8 | 4096 | 0.210816 ms | 0.213024 ms | 1.010x |
| 1 | 8 | 8192 | 0.394400 ms | 0.396208 ms | 1.005x |

This focused post-retuning confirmation used the same four-round shuffled
method; its JSON SHA256 was
`883f90133c85facfbe601db72e88c62e151d1e4450ce722ae2c08c0035c99571`.
Two additional batch/head decompositions confirmed that the decision follows
the batch-head task count rather than only the B=1 shapes:

| B | H | Tasks | Best tested C4 range vs C8/D35 |
| ---: | ---: | ---: | ---: |
| 2 | 8 | 16 | 1.859x-1.938x faster |
| 4 | 8 | 32 | 2.700x-2.718x faster |

Those JSON SHA256 values were
`61fe07ee5108a20f718956aefa88fe9266d727fad99c3431905a9d6dd4f7fcac`
and
`2d83dadff687f964a6e51e700648ffd11c707cf1a33d3a63a658b10548012803`,
respectively.

At H=16, 24, and 32, C8 expands the grid to 128, 192, and 256 CTAs,
respectively, and was substantially slower than C4 because the launch spans
more waves on 148 SMs. The retained policy is therefore C4/D15 for up to eight
batch-head tasks and C4/D30 for 9-32 tasks. C8 remains in the kernel ABI and
sanitizer coverage so future devices or a future scheduling design can retune
it, but it is not claimed as the B200 end-to-end optimum.

## B300 cluster-size tuning and validation

The CC 10.3 path was independently swept on a 148-SM B300-class device exposed
by the remote platform under the NVIDIA L20D product string:

- PyTorch: 2.9.1+cu130
- CUDA runtime reported by PyTorch: 13.0
- Compute capability: 10.3
- SM count: 148

The same cold-L2, four-round shuffled benchmark compared C4/D15, C4/D30,
C4/D45, and C8/D35 against both frozen CAKE baselines. Every forced
owner/helper route was bitwise identical to CAKE-M128 for output and recurrent
state before timing.
The selected B300 policy is C4/D30 through eight batch-head tasks and C4/D45
for 9-32 tasks:

| B | H | T=2048 | T=4096 | T=8192 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 1.075x | 1.149x | 1.182x |
| 1 | 16 | 1.084x | 1.136x | 1.181x |
| 1 | 24 | 1.067x | 1.136x | 1.179x |
| 1 | 32 | 1.066x | 1.105x | 1.126x |
| 2 | 8 | 1.085x | 1.136x | 1.191x |
| 4 | 8 | 1.069x | 1.115x | 1.130x |

Each value is the final automatic route's speedup over
`min(CAKE-M64, CAKE-M128)` for that shape. The enabled range is therefore
1.066x-1.191x. T=1024 remains on the baseline
fallback because no helper configuration beat the oracle across the table.
C8 again lost once its grid crossed additional waves, so it remains a forced
benchmark and sanitizer configuration rather than an automatic route.

The result JSON SHA256 values were:

- B=1, H=8/16/24/32:
  `e3c87cfdad23b5660e9ce4eb024b0dd5861a077fa9f74c7b7825119344be170a`
- B=2, H=8:
  `0166fba8f084448199155191c2c90059c6d031aa3b381f9591ea18324561ff10`
- B=4, H=8:
  `3ebde3b8579f68aee04cfe6491d13a89c13d1824f4965b98ba41aa7db563ee12`

The post-dispatch automatic-route confirmation JSON SHA256 values were:

- B=1, H=8/16/24/32:
  `c4c84c01cb818d5d4c4822c564cf6fc4d032202b007d019e6632709aa34c8f81`
- B=2, H=8:
  `73491f7eadd6ebc98b6e510ae6c2a63473386c638282c875d6276987f3a17be4`
- B=4, H=8:
  `dfeeef44863e998cad414066c7fe712789efc6d429018566680d75da6caf13e8`

B300 memcheck and synccheck were not rerun in this performance-validation
pass. They remain required on CC 10.3 before opening the upstream pull
request; the sanitizer results above cover CC 10.0 only.

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
