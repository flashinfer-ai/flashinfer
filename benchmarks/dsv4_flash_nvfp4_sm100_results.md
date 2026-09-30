# DeepSeek-V4-Flash-0731 routed NVFP4 decode: public API results

On NVIDIA B200, the public API passes strict BF16 correctness and beats `flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe` at every integer T=1..32. Public/FlashInfer geometric-mean speedup is **1.023459402081×**, minimum **1.004378404700× at T15**. The source launcher also wins every shape: **1.023425096839×** geometric mean, minimum **1.003905540862× at T15**. Neither comparison has ties or baseline regressions.

Geometry is H=4096, I=2048, E=256 routed experts, top-k=6 and clamped SwiGLU limit=10. The shared expert is outside this routed kernel. The matched execution path uses NVFP4 shuffled MajorK/R128c4 weights and activations, identical physical inputs/routes/scales and equivalent activation parameters.

The packed FC1 epilogue now packs only the two FP4 values stored in its output byte. This replaces an eight-value pack whose remaining six values were zero and unused. The same RN/satfinite conversion, operand and nibble order, scaling, clamp placement and BF16 output rounding are preserved. Synchronization, output ownership, persistent phases and launch ABI are unchanged. Canonical and pilot CUDA/SASS match after symbol normalization; the optimized epilogue removes 24 SASS instructions while preserving 48 registers, 352 threads and 64,128 bytes of shared memory.

The existing schedule retains early dependent launch, FC2 terminal-buffer recycling, vectorized stores, cluster-scoped descriptor synchronization, launch-local shared-memory carveout and the selected T9 FC2 variant. The first packed-FC1 epilogue barrier remains removed, with its final barrier and four release arrivals retained. Thirteen generated device translation units implement the dispatch; only the packed FC1 device unit changes. The occupancy API reports one block per SM; this static query does not establish dynamic residency.

Fresh validation passes 112 CPU feature tests, 13-module source/JIT closure, all 32 strict BF16 cases at atol=rtol=0.01, all 576 strict timing postchecks, seven API/graph cases and eight routing fixtures. Public output equals the source launcher bitwise in eager and graph execution. Independent reduction verifies retained medians, source/library identities and complete-call native activity envelopes.

Eight disjoint four-shape shards ran with up to four concurrent GPU workers. Every shape's three arms use the same GPU and physical inputs. Timing uses CUPTI and cold L2, six balanced captures per arm, with the geometric mean of capture medians. The boundary includes the complete per-call planner, projections, finalization and required workspace synchronization. Worker resource/clock records are retained. No causal improvement over a previous cohort is inferred from different runs or devices.

Source/public geometric mean is **1.000033520032×**; ratios above 1 favor the public API. The public API is slower than source at T=1, 7, 8, 9, 16, 17, 20, 21, 23, 24, 26, 31; every regression is included below.

| T | Public (µs) | FlashInfer (µs) | FlashInfer/public | Source (µs) | FlashInfer/source | Source/public |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.071518 | 29.736578 | 1.186070133× | 24.991364 | 1.189874138× | 0.996803019× |
| 2 | 41.290610 | 44.885033 | 1.087051829× | 41.296007 | 1.086909772× | 1.000130699× |
| 3 | 55.988913 | 60.394405 | 1.078685084× | 56.063943 | 1.077241489× | 1.001340085× |
| 4 | 67.770456 | 72.010556 | 1.062565609× | 68.010289 | 1.058818562× | 1.003538894× |
| 5 | 79.887947 | 83.567780 | 1.046062428× | 79.930629 | 1.045503838× | 1.000534278× |
| 6 | 91.679933 | 93.876738 | 1.023961673× | 91.711970 | 1.023603978× | 1.000349447× |
| 7 | 100.703310 | 103.044326 | 1.023246663× | 100.628407 | 1.024008316× | 0.999256204× |
| 8 | 110.037070 | 111.657964 | 1.014730439× | 109.946473 | 1.015566584× | 0.999176671× |
| 9 | 121.866222 | 123.263468 | 1.011465404× | 121.738432 | 1.012527150× | 0.998951391× |
| 10 | 128.388470 | 129.114133 | 1.005652085× | 128.484821 | 1.004897949× | 1.000750460× |
| 11 | 135.786287 | 137.119802 | 1.009820689× | 135.808060 | 1.009658798× | 1.000160342× |
| 12 | 145.951977 | 146.917652 | 1.006616393× | 146.229487 | 1.004706060× | 1.001901385× |
| 13 | 155.097992 | 156.436589 | 1.008630653× | 155.444789 | 1.006380399× | 1.002235987× |
| 14 | 161.962485 | 162.767964 | 1.004973238× | 162.036983 | 1.004511199× | 1.000459965× |
| 15 | 169.871981 | 170.615749 | 1.004378405× | 169.951995 | 1.003905541× | 1.000471024× |
| 16 | 179.370308 | 180.495247 | 1.006271600× | 179.349299 | 1.006389472× | 0.999882877× |
| 17 | 185.573155 | 188.426524 | 1.015375977× | 185.210478 | 1.017364274× | 0.998045639× |
| 18 | 189.477465 | 192.277354 | 1.014776896× | 189.488143 | 1.014719715× | 1.000056352× |
| 19 | 195.034647 | 198.175790 | 1.016105565× | 195.135647 | 1.015579642× | 1.000517855× |
| 20 | 202.479094 | 205.673506 | 1.015776504× | 202.442090 | 1.015962173× | 0.999817248× |
| 21 | 210.757811 | 212.832740 | 1.009845085× | 210.597318 | 1.010614671× | 0.999238496× |
| 22 | 221.887309 | 225.295705 | 1.015360933× | 221.951806 | 1.015065879× | 1.000290675× |
| 23 | 225.215165 | 228.665447 | 1.015319935× | 225.124487 | 1.015728898× | 0.999597370× |
| 24 | 237.951820 | 240.298228 | 1.009860853× | 237.690150 | 1.010972593× | 0.998900326× |
| 25 | 245.605328 | 247.781084 | 1.008858748× | 245.626475 | 1.008771893× | 1.000086100× |
| 26 | 250.921795 | 253.428091 | 1.009988357× | 250.404298 | 1.012075645× | 0.997937616× |
| 27 | 257.199821 | 259.829145 | 1.010222885× | 257.221326 | 1.010138426× | 1.000083612× |
| 28 | 264.821988 | 266.922769 | 1.007932806× | 264.939138 | 1.007487122× | 1.000442372× |
| 29 | 272.121570 | 275.150709 | 1.011131565× | 272.174932 | 1.010933325× | 1.000196096× |
| 30 | 282.383975 | 284.757124 | 1.008403981× | 282.388985 | 1.008386089× | 1.000017743× |
| 31 | 286.315480 | 288.389561 | 1.007244040× | 286.149811 | 1.007827193× | 0.999421376× |
| 32 | 290.287654 | 293.935598 | 1.012566653× | 290.434282 | 1.012055449× | 1.000505114× |


Timings use matched synthetic NVFP4 fixtures, not full actual-checkpoint pipeline timings. The pinned checkpoint packing and conversion were separately verified; equivalence to its different FP8-activation arithmetic path is not claimed. Earlier separate synccheck and racecheck 20-second timeouts remain skipped, not passes. This conversion-only change adds no synchronization, ownership or layout change; no new sanitizer invocation or transferred timeout credit is claimed. The inherited static collector refusal remains unresolved. Hardware-limit evidence and remaining promotion requirements are incomplete.

GPU batch physical turnaround is 2304.645341 seconds; aggregate worker runtime is 5631.373211 seconds. Independent raw reduction used 69.901346 worker seconds. These durations are separate from the GPU timings in the table.

Related: [FlashInfer issue #5184](https://github.com/flashinfer-ai/flashinfer/issues/5184).
