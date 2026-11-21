# SM90 MegaMoE: EP8 and EP16

Standalone layer: E256, hidden 4096, intermediate 2048, top-6; FP8 activations and BF16 output. FC1 weights are [256,4096,4096], FC2 [256,4096,2048]: 6.442 billion expert parameters, 150.995 million active per unmasked token. H20-3e has 78 SMs/GPU. EP8 uses eight GPUs on one node; EP16 uses sixteen on two nodes. This measures a fused expert layer, not a full-model speedup. Physical records are canonical MXFP4 8704 B, compact MXFP4 8448 B or raw FP8 16384 B.

Root review accepted all 23 measured paired configurations: 12 FP8/FP8 controls, 5 compact/compact and 6 FP8/compact points. All passed source/input/weight binding, the original 2% within/cross-arm gates and long-replay BF16 checks. Main-table format choices follow the predeclared policy; all variants remain in the CSV.

Operating protocol: requested 1740 MHz, 10000 fused calls per event sample, three warmup graphs and five samples per arm; order EP16/EP8/EP8/EP16. The actual clock receipts and observations are separate columns; missing observations remain empty. No time is extrapolated to maximum clocks. All67, full-output long-replay BF16 and within/cross-arm 2% gates are preserved.

## Same tokens per rank

| Requested | M EP8/16 | Global tokens EP8/16 | Storage EP8/16 | Cap EP8/16 | EP8 µs | EP16 µs | Global throughput EP16/8 | Per-GPU throughput EP16/8 | Common hot SOL µs EP8/16 | Common hot % EP8/16 | Within spread % EP8/16 | Cross spread % EP8/16 | Status |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1/1 | 8/16 | fp8/mxfp4_compact | 2/2 | 65.858 | 88.599 | 1.487 | 0.743 | 3.104/3.104 | 4.713/3.503 | 0.056/0.593 | 0.074/0.387 | PASS |
| 4 | 4/4 | 32/64 | mxfp4_compact/mxfp4_compact | 4/4 | 155.532 | 146.411 | 2.125 | 1.062 | 51.620/30.057 | 33.189/20.529 | 0.432/0.085 | 0.068/0.169 | PASS |
| 8 | 8/8 | 64/128 | mxfp4_compact/mxfp4_compact | 8/8 | 207.721 | 163.282 | 2.544 | 1.272 | 73.182/30.057 | 35.231/18.408 | 0.385/0.203 | 0.164/0.010 | PASS |
| 16 | 16/16 | 128/256 | fp8/fp8 | 64/64 | 224.582 | 168.196 | 2.670 | 1.335 | 73.182/30.057 | 32.586/17.870 | 0.272/0.121 | 0.064/0.035 | PASS |
| 32 | 32/32 | 256/512 | fp8/fp8 | 64/64 | 230.549 | 178.266 | 2.587 | 1.293 | 73.182/30.553 | 31.743/17.139 | 0.083/0.104 | 0.037/0.022 | PASS |
| 64 | 64/64 | 512/1024 | fp8/fp8 | 64/64 | 249.792 | 197.258 | 2.533 | 1.266 | 73.182/61.106 | 29.297/30.978 | 0.464/0.183 | 0.049/0.013 | PASS |

## Same global tokens

| Requested | M EP8/16 | Global tokens EP8/16 | Storage EP8/16 | Cap EP8/16 | EP8 µs | EP16 µs | Global throughput EP16/8 | Per-GPU throughput EP16/8 | Common hot SOL µs EP8/16 | Common hot % EP8/16 | Within spread % EP8/16 | Cross spread % EP8/16 | Status |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 2/1 | 16/16 | mxfp4_compact/mxfp4_compact | 2/2 | 95.392 | 88.619 | 1.076 | 0.538 | 19.276/3.104 | 20.207/3.502 | 0.047/0.193 | 0.005/0.049 | PASS |
| 64 | 8/4 | 64/64 | mxfp4_compact/mxfp4_compact | 8/4 | 207.769 | 146.201 | 1.421 | 0.711 | 73.182/30.057 | 35.223/20.559 | 0.527/0.160 | 0.251/0.003 | PASS |
| 128 | 16/8 | 128/128 | fp8/mxfp4_compact | 64/8 | 223.482 | 163.569 | 1.366 | 0.683 | 73.182/30.057 | 32.746/18.376 | 0.368/0.161 | 0.070/0.054 | PASS |
| 256 | 32/16 | 256/256 | fp8/fp8 | 64/64 | 232.172 | 168.193 | 1.380 | 0.690 | 73.182/30.057 | 31.521/17.871 | 0.110/0.131 | 0.075/0.011 | PASS |
| 512 | 64/32 | 512/512 | fp8/fp8 | 64/64 | 250.030 | 178.662 | 1.399 | 0.700 | 73.182/30.553 | 29.269/17.101 | 0.221/0.133 | 0.037/0.017 | PASS |
| 1024 | 128/64 | 1024/1024 | fp8/fp8 | 128/64 | 277.131 | 197.555 | 1.403 | 0.701 | 122.212/61.106 | 44.099/30.931 | 0.883/0.061 | 0.103/0.026 | PASS |

Same-per-rank comparisons double global work at EP16; same-global comparisons preserve total work. Per-point capacity and formats are explicit. Preparing or retaining multiple weight layouts is outside kernel timing.

## Complete FP8/FP8 scaling control

| Scope / requested | EP8 µs | EP16 µs | Global throughput EP16/8 | Per-GPU throughput EP16/8 | Status |
|---|---:|---:|---:|---:|---|
| per-rank / 1 | 65.489 | 94.390 | 1.388 | 0.694 | PASS |
| per-rank / 4 | 168.004 | 155.077 | 2.167 | 1.083 | PASS |
| per-rank / 8 | 221.898 | 161.665 | 2.745 | 1.373 | PASS |
| per-rank / 16 | 224.582 | 168.196 | 2.670 | 1.335 | PASS |
| per-rank / 32 | 230.549 | 178.266 | 2.587 | 1.293 | PASS |
| per-rank / 64 | 249.792 | 197.258 | 2.533 | 1.266 | PASS |
| total / 16 | 96.738 | 94.685 | 1.022 | 0.511 | PASS |
| total / 64 | 222.137 | 154.669 | 1.436 | 0.718 | PASS |
| total / 128 | 224.213 | 161.667 | 1.387 | 0.693 | PASS |
| total / 256 | 232.172 | 168.193 | 1.380 | 0.690 | PASS |
| total / 512 | 250.030 | 178.662 | 1.399 | 0.700 | PASS |
| total / 1024 | 277.131 | 197.555 | 1.403 | 0.701 | PASS |

## SOL and evidence

Modeled SOL is an ideal-overlap resource lower bound at nominal 1980 MHz / 316.29312 TFLOP/s, HBM 4.814304 TB/s and the declared communication envelope. The request was 1740 MHz; selected observations span 1410–1740 MHz and include software power-limit events. The nominal hardware-ceiling target remains 1980 MHz. Unknown L2/NVLink/startup/fixed lower bounds are conservatively omitted. Cold assumes no prior weight residency or uncharged compression; hot max(U−60MiB,0) is a necessary-byte bound, not a residency guarantee.

The shared compact8448 denominator is the common optimization target: rawFP8's extra IO cannot raise it. Own-format and common cold/hot SOL and attainment are retained in `fp8-control-SOL.csv` (24 EP-side rows) and `all-cohort-details.csv`. Unsupported high-M compact plans remain hypothetical targets. Attainment is 100×modeled SOL/full-kernel latency; it is neither a production adoption decision nor a proof of an attainable physical optimum.

See `main-results.csv`, `fp8-control.csv`, `all-cohort-details.csv` and the companion CSVs for source core, declared commit, seed, graph calls, clocks, all cohorts and exclusions. Original records are preserved separately. Raw rank/arm records were independently reaggregated; every published row passes its source, accuracy, replay and stability gates.


Model context: [DeepSeek-V4-Flash-0731 pinned config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json): 43 hidden layers, 1 shared expert, E256/H4096/I2048/top-6. Timings use generated routed-core inputs. Benchmark source for the EP8/EP16 tables is `f5fd55a4e`, core `01293028…`; the subsequent H512/H768 column-batch correction keeps H4096 executable source equivalent. A later M1 compact return-handoff optimization has separate measurements below; its timings are not substituted into these paired tables.


[Complete SOL derivation and all 46 measured EP-side comparisons](SOL.md).

## M1 compact return handoff

EP16 M1 on the same 2×8 H20 setup now uses a CTA0 math-warp barrier after complete return/TMA waits. Other supported configurations retain their existing synchronization. Against `e8ed127a`, independent ABBA and BAAB series measured +0.5594% and +0.5572% throughput, respectively: combined latency 88.4005 → 87.9097 µs. Both series and their cross-series rank medians passed the original 2% gate; 67 routing cases on all 16 ranks retained identical BF16 outputs and maximum relative L2 0.008995.

These measurements use compact8448/cap2, requested 1740 MHz, 10000 calls per sample, three warmup graphs and five samples per arm, with a fresh process per arm. Complete BF16 outputs are checked between timing windows. They compare the two EP16 implementations with the same method and do not update the EP8 scaling ratios above. See [per-arm data](m1-return-handoff.csv); parent core `fce73dc0…`, candidate core `03129830…`. Actual release SASS confirms the M1 branch skips the return grid while preserving full TMA completion and the fallback; both binaries use REG160, STACK0 and LOCAL0.
