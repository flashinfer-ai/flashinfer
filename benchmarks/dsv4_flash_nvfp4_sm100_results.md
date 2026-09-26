# DeepSeek-V4-Flash-0731 routed NVFP4 decode: public API results

On NVIDIA B200 (148 SMs, CUDA 13.3), the public API passes strict BF16 correctness
for every T=1..32 and is faster than
`flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe` at every shape.
Geometric-mean speedup: **1.013609403×**. Minimum:
**1.000606155× at T9** (122.948442 µs public API versus
123.022968 µs baseline). These measurements alone do not establish a hardware limit.

Geometry: H=4096, I=2048, E=256 routed experts, top-k=6, clamped SwiGLU limit=10.
The shared expert is excluded. Candidate and baseline use identical physical
NVFP4 shuffled MajorK / R128c4 inputs, routes and scales, with equivalent clamped
activation parameters. BF16 checks use atol=rtol=0.01. Public outputs also match
the source implementation bitwise across all 32 shapes. This validates matched
NVFP4 execution, not equivalence to native FP8 checkpoint execution. The separate
six-expert checkpoint check does not establish a full 256-expert checkpoint run.

This integration preserves upstream optional activation parameters, owned scratch,
fused FC1 routes and per-source JIT flags while adding the clamped decode path.
It passed 112 CPU feature tests, a fresh source/JIT build, seven API/graph cases,
eight dynamic-route cases and all 32 numerical cases. The eleven generated device
translation units remain unchanged from the selected export.

The selected schedule includes packed-FC2 vectorized global stores and launch-local
shared-memory carveout preferences. CUDA graph checks verify matching requested
attributes through the public API and source launcher for all 32 shapes.

Timing uses CUPTI with cold L2, six balanced captures per arm, and the geometric
mean of all six capture medians. Every measurement includes the complete per-call
planner, projections and finalization, plus public workspace synchronization.
No row, tie or regression is omitted. Ratios above 1 favor the public API.
“Source/public” compares the same selected schedule through its source launcher
and the public API; the public API is slower on 20 of 32 shapes in that comparison.

| T | Public API (µs) | FlashInfer (µs) | FlashInfer/public | Source launcher (µs) | Source/public |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.430730 | 29.121236 | 1.101794614× | 26.261017 | 0.993578952× |
| 2 | 41.754165 | 44.462586 | 1.064865894× | 41.743569 | 0.999746238× |
| 3 | 56.794130 | 60.233688 | 1.060561867× | 56.900754 | 1.001877383× |
| 4 | 69.316908 | 72.212391 | 1.041771672× | 69.306472 | 0.999849448× |
| 5 | 79.876454 | 82.980306 | 1.038858153× | 79.962296 | 1.001074682× |
| 6 | 91.812939 | 93.497950 | 1.018352646× | 91.748638 | 0.999299649× |
| 7 | 101.183150 | 103.369909 | 1.021611892× | 101.140664 | 0.999580108× |
| 8 | 110.191287 | 111.444089 | 1.011369338× | 110.234094 | 1.000388474× |
| 9 | 122.948442 | 123.022968 | 1.000606155× | 122.985964 | 1.000305186× |
| 10 | 128.430946 | 129.226098 | 1.006191278× | 128.479087 | 1.000374841× |
| 11 | 136.617375 | 136.921511 | 1.002226194× | 136.420416 | 0.998558318× |
| 12 | 146.334810 | 146.670784 | 1.002295926× | 146.276128 | 0.999598987× |
| 13 | 156.542967 | 156.676280 | 1.000851601× | 156.436306 | 0.999318648× |
| 14 | 162.019941 | 162.622914 | 1.003721595× | 161.844314 | 0.998916012× |
| 15 | 169.689318 | 170.403953 | 1.004211431× | 169.695047 | 1.000033764× |
| 16 | 179.214826 | 180.206549 | 1.005533710× | 179.323681 | 1.000607401× |
| 17 | 185.017443 | 186.195983 | 1.006369882× | 185.161097 | 1.000776431× |
| 18 | 189.443985 | 190.243918 | 1.004222530× | 189.316310 | 0.999326055× |
| 19 | 195.033276 | 196.019791 | 1.005058188× | 195.268160 | 1.001204327× |
| 20 | 202.718651 | 203.737377 | 1.005025319× | 202.931983 | 1.001052354× |
| 21 | 210.569472 | 211.454420 | 1.004202643× | 210.851981 | 1.001341643× |
| 22 | 221.945135 | 223.321236 | 1.006200187× | 221.955803 | 1.000048069× |
| 23 | 225.881117 | 227.107956 | 1.005431346× | 225.640976 | 0.998936868× |
| 24 | 237.625330 | 238.633098 | 1.004240993× | 237.316147 | 0.998698862× |
| 25 | 245.315487 | 246.297154 | 1.004001652× | 245.019894 | 0.998795049× |
| 26 | 251.747485 | 251.982340 | 1.000932899× | 250.931830 | 0.996760026× |
| 27 | 257.294149 | 257.883449 | 1.002290374× | 257.086633 | 0.999193468× |
| 28 | 265.219471 | 265.539739 | 1.001207558× | 264.872651 | 0.998692326× |
| 29 | 272.371481 | 273.379170 | 1.003699688× | 271.934321 | 0.998394989× |
| 30 | 282.131324 | 282.691816 | 1.001986634× | 282.120643 | 0.999962141× |
| 31 | 286.232818 | 286.605996 | 1.001303757× | 286.014141 | 0.999236018× |
| 32 | 291.512982 | 292.222097 | 1.002432535× | 291.416976 | 0.999670663× |

The public API is slower than the source launcher at T=1,2,4,6,7,11,12,13,14,18,23,24,25,26,27,28,29,30,31,32.
These differences are separate from the 32 wins against the named FlashInfer baseline.

Earlier synccheck and racecheck invocations each reached their hard 20-second
timeout and were recorded as skipped. Neither establishes a sanitizer pass for
this exported library. They were not retried in this integration run.

The validation worker took 6631.225986 seconds; its controller took
6635.316411 seconds. End-to-end managed turnaround was
6743.948974 seconds. These orchestration durations are separate from the GPU
timings in the table.
