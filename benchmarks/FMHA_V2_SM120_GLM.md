# GLM FMHA-v2 on SM120: full tiles and direct output

Measured on an NVIDIA RTX PRO 5000 72GB, CUDA 12.9. Baseline:
`481a4e4a` (GLM FP8 support, before the tile/epilogue changes).
The optimization applies only to separate E4M3 Q/K/V, dimensions 256/256,
64-token tiles, and BF16/FP16 output on SM120. It does not change quantization
scales, softmax precision, the MMA reduction order, or the public interface.

## Changes

- Load the full 256-column Q/K head and use the complete 64-token PV reduction
  tile. This avoids the previous granular Q reloads.
- Write the accumulator fragments directly as four adjacent 16-bit output
  elements, reusing the existing scale/packing conversion. This removes the
  64x256 FP32 shared-memory output staging allocation.
- Synchronize after the last K and V shared reads before reusing their single
  buffers. Multiple resident CTAs exposed a race in an intermediate full-tile
  experiment; the final code includes the required barriers.

The direct output mapping follows the existing V-loader layout: accumulator
registers `(0,4,1,5)` and `(2,6,3,7)` form the two output rows. The final row is
predicated against the actual sequence length; batch/head offsets use 64-bit
address arithmetic. Other shapes retain the original output staging path.

## Kernel measurements

Real captured GLM layer-23 operands, B=1, H=20, Dqk=Dv=256, causal E4M3 to
BF16. Quantization/packing is outside the timed region. CUDA graph/event
medians from four alternating rounds; GPU boost remains enabled. There is no
artificial hardware-L2 flush in the microbenchmark. This is distinct from
the serving benchmark, which flushes the prefix KV cache before every repeat.

| Tokens | Baseline ms | Optimized ms | Time reduction | Kernel speedup | Prior fused reference ms |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.065893 | 0.049800 | 24.42% | 32.32% | 0.053351 |
| 2048 | 0.212594 | 0.157437 | 25.94% | 35.03% | 0.150080 |
| 5093 | 1.217822 | 0.887180 | 27.15% | 37.27% | 0.799090 |

Time reduction is `1 - optimized / baseline`; speedup is
`baseline / optimized - 1`. Neither is an end-to-end QPS improvement.
The previous fused CuTe FP8 implementation uses a different quantization
scheme. The optimized FMHA-v2 still takes approximately 11% longer at 5093 tokens,
but has approximately 6.7% lower latency at 1024 tokens.

## Profiler findings

The profiler comparison below uses the preceding full-tile experiment to
isolate the shared-memory occupancy limitation. Nsight Compute clocks/replays
differ from the event timing above; its durations are not mixed into that table.

| Metric | Before direct output | Optimized | Fused CuTe reference |
|---|---:|---:|---:|
| Registers/thread | 248 | 248 | 255 |
| Dynamic shared memory/block | 64 KiB | 48 KiB | 49 KiB |
| Shared-memory block limit/SM | 1 | 2 | 2 |
| Achieved occupancy | 8.33% | 16.07% | 16.44% |
| Compute throughput (% of peak) | 52.00% | 62.46% | 72.87% |
| Local load/store instructions | 0 / 0 | 0 / 0 | 0 / 0 |

Disassembly shows `LDGSTS.E.BYPASS.128` (cp.async) in FMHA-v2 and `UTMALDG.3D`
(TMA) in the fused reference. Both execute `QMMA.16832.F32.E4M3.E4M3`.
High register counts did not produce spills in these profiled kernels.
The original limitation was shared-memory residency, not DRAM bandwidth.
After fixing residency, the remaining gap involves pipeline scheduling,
synchronization and shared-memory access. This experiment does not isolate
TMA alone, so it cannot assign the remaining difference entirely to TMA.

## Reproduction and integration

`bench_fmha_v2_sm120_glm.py` provides a standalone synthetic-input benchmark
using the public source JIT and preallocated operands/output. Its
`--baseline-library` option loads a saved baseline FMHA-v2 shared library for
alternating comparisons and exact output checks. `--jit-only` excludes prebuilt
AOT artifacts; stale `data/aot/trtllm_fmha_v2` binaries can otherwise mask source
edits. Default sizes match the table; captured model operands are not distributed.
Machine-readable measurements are in `results/fmha_v2_sm120_glm_20261014.json`.

The tested SGLang environment keeps FlashInfer 0.6.15.post1 for its MoE stack
and compiles this checkout's CUDA source through the existing `fmha_v2_glm`
adapter. Native tests use an isolated FlashInfer 0.7.2 environment. Replacing
the serving environment's entire FlashInfer package is not part of this change.
FMHA-v2 remains an explicit prefill hook in that environment; decode is unchanged.

## Validation

- 100 native tests passed, including other SM120 layouts, output types, device
  scales, graph replay, and repeated execution. Prebuilt package AOT kernels
  were excluded and the test process staged only its freshly compiled kernels.
- 28 paired real-input output/LSE checks and seven updated-input/device-scale
  graph cases are bitwise identical to the initial FMHA-v2 baseline.
- Compute Sanitizer racecheck reports zero hazards after the buffer barriers;
  memcheck reports zero errors for the final source build.
- The standalone synthetic benchmark also passes exact baseline comparisons:
  0.066009 -> 0.050106 ms (1024), 0.212052 -> 0.157485 ms (2048), and
  1.204308 -> 0.885844 ms (5093). Select this checkout on Python's import path;
  the benchmark prints its imported package path/version for verification.

## SGLang service comparison

Each server was independently restarted with identical model/settings and the
existing MoE/dense/LM-head optimizations. Only the prefill candidate changed.
Concurrency 11, 200 requests x 2 repeats, non-streaming, max_tokens=128,
temperature=0.6, identical request seeds. Nine warmup requests precede the
measured repeats; the prefix KV cache is flushed before **each** repeat.
QPS below is 400 / total measured wall time; latency percentiles pool all 400
successful requests. This is a fixed-concurrency comparison, not a capacity search.

| Prefill | QPS | Mean E2E ms | P95 E2E ms | Mean output tokens | Output tokens/s | Success |
|---|---:|---:|---:|---:|---:|---:|
| Initial FMHA-v2 | 4.7071 | 2309.70 | 3563.72 | 93.16 | 438.50 | 400/400 |
| Optimized FMHA-v2 | 4.8412 | 2241.70 | 3277.89 | 96.17 | 465.58 | 400/400 |
| Prior fused CuTe FP8 | 5.3037 | 2042.20 | 3031.27 | 95.25 | 505.17 | 400/400 |

Optimized versus initial FMHA-v2: QPS +2.85%, mean E2E -2.94%, P95 E2E -8.02%,
and output token throughput +6.18%. Mean completion length also increased
3.23%; serving outputs were not identical despite fixed request seeds. These
are observations for this sample, not a throughput guarantee.

Attention is only part of prefill; projection, MoE, quantization, batching and
approximately 93-96 decode tokens per request remain. The FMHA-v2 adapter also
retains its padding/packing overhead. Kernel speedup therefore does not carry
through proportionally to service QPS. The prior fused implementation remains
faster overall in this comparison, and the service was restored to it after testing.
