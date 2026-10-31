# SM120 mega attention: upstream workloads and merge optimization

Baseline implementation commit: `da8e50cb`.
Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120, 96 GiB.
The original unfused comparison module is built from
`18fb0bfcc6d7e8a0c25a7784af559cc1e58124cb`.

## Workloads traced from the original PRs

- [FlashInfer #3395](https://github.com/flashinfer-ai/flashinfer/pull/3395):
  `benchmarks/bench_sparse_mla_sm120.py`. Published V4 cases include single
  caches with K=128/512/1024/2048; T=1/256/512/1024; H=16/64/128; and dual
  caches with K=128 plus 512/768/1536/1664 entries and extra page sizes 2/64.
  The built-in primary pool has 1,048,576 slots (612 MB in FP8); dual-cache
  extra pools target 256 MiB. Its separate 576-dimensional V3.2 cases are
  outside this 512-dimensional mega API.
- [FlashInfer #4955](https://github.com/flashinfer-ai/flashinfer/pull/4955):
  `benchmarks/bench_sparse_mla_nvfp4_prefill.py` and
  `benchmarks/bench_sparse_mla_nvfp4_decode.py`. Published cases include
  T=8192, H=64/128, K=128 or 128+512, and decode T=8/H=64/K=128+512.
  The built-in default pool is only 128 pages of 64 slots (8192 slots),
  and the decode benchmark sweeps chunks per block. This differs materially
  from #3395's large pool. The prefill benchmark explicitly invokes native
  NVFP4 streaming prefill.
- [SGLang #38646](https://github.com/sgl-project/sglang/pull/38646): cites
  those FlashInfer operator benchmarks, with H=16/32/64 and
  T=1024/2048/4096/8192, K=128+512. It reports sensitivity to heads per rank
  and a 64-query-token decode/planner regression. These motivate the H=32
  and T=64 cases, without assuming NVFP4 is faster than FP8 for every shape.

`benchmarks/sparse_mla_sm120_mega_cases.json` contains 61 deduplicated cases,
with a source URL on every row. It includes the published shapes, the
SGLang head-count sweep, the T=64 boundary, and explicitly named large-pool
extensions of the NVFP4 workloads. Primary and extra slot counts stay fixed
between FP8 and NVFP4. Inputs follow the built-ins' normal/10, clamp(-1,1)
distribution with no sink; the separate decode regression matrix also uses
unit-scale inputs and sinks. Each format's fusion gate uses its own unfused
attention arithmetic and output contract.

The built-in scripts were run before changing the merge. Example observations:

| Built-in workload | FP8 core (us) | NVFP4 core (us) |
|---|---:|---:|
| T=8192, H=64, K=128+512, default pool | 6380.400 | 4605.952 |
| T=8, H=64, K=128+512, best CPB per format | 34.432 | 34.912 |

Selected #3395 functions also ran directly: single-cache T=256/H=16/K=128
measured 55.296 us; T=256/H=64/K=2048 measured 574.368 us;
T=1024/H=128/K=2048 measured 3917.920 us. The dual-cache
T=512/H=128/K=128+1664/extra-page-size=2 case measured 1883.200 us.
These are observations on current source/hardware, not expected constants
from the PR tables. The mega timing harness checks the complete fused output
contract and times prologue, attention, epilogue, and their sequential pipeline.

## Retained kernel change

For a mega general merge with at most 32 split slots, each warp loads one
LSE per lane. Both warps reproduce the original first warp's XOR max/sum
reduction. The result from lane zero is broadcast: other lanes' butterfly
sums can differ in floating-point rounding order. Split LSE values then reach
the output accumulation loop through shuffles rather than shared memory.

This removes two CTA barriers and all explicit merge shared memory. It keeps
the split accumulation order, sink handling, BF16 rounding, inverse RoPE,
FP8 output, and UE8M0 scale bytes. More than 32 slots use the previous merge;
unfused calls also retain the previous merge. Native NVFP4's direct and
special two-split paths retain their existing implementation.

Main files:

- `include/flashinfer/attention/sparse_mla_sm120/execution/merge.cuh`
- `include/flashinfer/attention/sparse_mla_sm120/execution/decode_launch.cuh`
- `include/flashinfer/attention/sparse_mla_sm120/execution/dsv4_nvfp4_launch.cuh`

A second experiment reused each split's exponentiated weight. It passed exact
correctness but did not improve timings consistently and was reverted.
Its raw results remain in `results/optimization/warp_weight_decode.jsonl`
in the enclosing workspace.

## Measurements and validation

Before optimization, all 61 added workloads passed both original-unfused
performance gates. The retained optimization's initial paired decode sweep
covered 20 cases across V4 FP8, native NVFP4, V4.1 FP8 and V4.1 mixed caches.
Affected cases improved roughly 1–5% against the committed mega implementation;
cases using NVFP4 merge2 were effectively unchanged. Comparisons load the
previous mega and untouched unfused modules in the same process and use
identical tensors. FP8 numeric output, scale bytes and LSE are checked exactly
before timing both mega implementations.

A focused NCU basic report at T=64/H=64/V4.1 mixed cache showed:

| Merge metric | Committed mega | Warp-local reduction |
|---|---:|---:|
| Registers/thread | 54 | 48 |
| Explicit dynamic shared bytes/block | 40 | 0 |
| Explicit static shared bytes/block | 16 | 0 |
| Theoretical occupancy | 75.00% | 83.33% |
| Profiled duration (us) | 10.75 | 9.44 |

These are 9-pass diagnostic profiles, not the latency acceptance measurement.
NCU 2026.3 used `--clock-control none --pipeline-boost-state dynamic
--cache-control none --set basic`, one selected merge launch after warmup.
No manual clock changes were made. Reports were re-imported to verify the
kernel and launch: grid 4096, block 64. Driver-reserved shared memory is
separate from the explicit allocation above. Reports are
`results/optimization/merge_before.ncu-repz` and `merge_after.ncu-repz` in
the enclosing workspace; they are not source-line-correlated builds.

The regression suite passed **268 tests**, including 24 new split-boundary
cases at 10/31/32/33 splits for FP8 and native NVFP4, with normal, empty,
and underflow inputs. Output FP8 values, UE8M0 scale bytes and LSE match the
unfused path exactly. Full memcheck on these 24 cases reported zero errors;
racecheck filtered to the changed general merge reported zero hazards.
The unfiltered racecheck attempt was interrupted because it spent most of
its time instrumenting the unchanged attention core; that partial run is
not counted as a completed check.

The initial final sweep passed all 61 warm PR workload gates and all 20
cold-cache decode gates against both the original stage sum and sequential
pipeline. Cold-cache timings used a 256 MiB eviction before each single call.
Against the previous mega build, the cold-cache ratios ranged from 0.9793
to 1.0556; these measurements do not establish an improvement for every case.

The multi-stage harness initially showed up to 2% differences on unchanged
prefill kernels. FP8 prefill cubins were byte-identical, and all 32 native
NVFP4 streaming kernels had identical SASS. A separate balanced 30-sample
comparison, priming each arm before measurement, put the three worst cases
within 0.04% of the previous build. The benchmark now rotates the starting
arm, reverses each complete rotation, and primes each warm arm separately;
cold-cache arms still receive an eviction instead. Use a sample count that
is a multiple of twice the number of arms for complete order balance
(12 with both comparison modules and `--fused`).

The corrected warm sweep passed **61/61 PR workload gates** and **20/20
additional decode gates**, with exact output/scale/LSE comparisons against
the original unfused module before timing. Every case is faster than both
the separately measured stage sum and the sequential unfused pipeline.
The PR matrix stage-sum speedups range from 1.020x to 1.914x. In the additional
warm decode matrix, previous-mega/new-mega ratios range from 1.000x to 1.050x.
The PR matrix's unchanged prefill paths match within 0.1%.

Largest query shape: T=8192, H=128, D=512, K=128+512. Each of the primary and
extra pools has the stated number of slots. Timings below are milliseconds.

| Cache format | Slots per pool | Unfused stage sum (ms) | Mega (ms) | Speedup |
|---|---:|---:|---:|---:|
| fp8 | 8,192 | 14.660 | 12.217 | 1.200x |
| fp8 | 1,048,576 | 14.688 | 12.342 | 1.190x |
| nvfp4 | 8,192 | 11.271 | 8.724 | 1.292x |
| nvfp4 | 1,048,576 | 13.802 | 11.236 | 1.228x |

Raw artifacts in the enclosing workspace:

- `results/optimization/balanced_pr_matrix.jsonl`: final 61-case warm sweep.
- `results/optimization/balanced_decode_matrix.jsonl`: final 20-case warm sweep.
- `results/optimization/final_decode_cold.jsonl`: 20-case cold sweep.
- `results/optimization/final_tests.log`: 268 passing tests.
- `results/optimization/final_memcheck.log`: full boundary-case memcheck.
- `results/optimization/final_merge_racecheck.log`: targeted merge racecheck.
- `results/optimization/paired_prefill.jsonl`: timing-order investigation.
- `results/optimization/prefill_sass_comparison.json`: NVFP4 streaming SASS hashes.
- `results/optimization/validated_sha256.json`: compiled modules and execution-header hashes.

These results establish SM120 performance against the matching unfused
implementation. They do not constitute an SM100/FlashMLA hardware comparison.


## Reproduction

From the FlashInfer checkout, using an environment with the SM120 dependencies:

```bash
export FLASHINFER_CUDA_ARCH_LIST=12.0a MAX_JOBS=8 FLASHINFER_NVCC_THREADS=2
python -m pytest tests/attention/test_sparse_mla_sm120_mega.py -q
PYTHONPATH=. python benchmarks/bench_sparse_mla_sm120_mega.py \
  --fused --cases benchmarks/sparse_mla_sm120_mega_cases.json \
  --calls 5 --samples 12 --warmup-seconds 2 --output mega-pr-workloads.jsonl
```

For in-process comparisons, add `--baseline-module ORIGINAL.so` for an
untouched unfused build, and `--mega-baseline-module PREVIOUS_MEGA.so` for the
committed mega build. The modules must have matching execution-plan ABIs.
Without these options, the unfused side uses the current module. Add `--cold`
to evict 256 MiB before every measured single invocation. Raw samples, plans,
workload metadata and source revision are emitted as JSONL.

The built-in NVFP4 commands from #4955 remain directly runnable:

```bash
python benchmarks/bench_sparse_mla_nvfp4_prefill.py \
  --num-tokens 8192 --num-heads 64 --topk 128 \
  --extra-topk 512 --extra-page-size 64
python benchmarks/bench_sparse_mla_nvfp4_decode.py \
  --num-tokens 8 --num-heads 64 --topk 128 \
  --extra-topk 512 --extra-page-size 64
```
