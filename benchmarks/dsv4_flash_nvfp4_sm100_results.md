# DeepSeek-V4-Flash-0731 routed NVFP4 decode: public API results

On NVIDIA B200 (148 SMs, CUDA 13.3), the selected public API passes strict BF16
correctness for every T=1..32 and beats
`flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe` at every shape.
Geometric-mean speedup is **1.021482434×**; minimum is
**1.004732110× at T14**, with 32 wins, zero ties and zero baseline regressions.
Hardware-limit evidence and remaining promotion work are still pending.

Geometry: H=4096, I=2048, E=256 routed experts, top-k=6, clamped SwiGLU limit=10.
The shared expert is outside this routed path. Candidate and baseline consume
identical physical NVFP4 shuffled MajorK / R128c4 inputs, routes and scales,
with equivalent clamped activation parameters and BF16 rounding semantics.

Benchmarks use matched NVFP4 physical fixtures at E=256. Separately, all 256 layer-0
routed experts from the pinned FP4 checkpoint passed exact scale conversion and
FlashInfer MajorK/R128c4 layout validation on B200. The native model's
FP8-activation pipeline is a different arithmetic path; numerical equivalence
to it is not claimed. A full 256-expert checkpoint pipeline run is not part of
the reported benchmark results.

The selected schedule includes early dependent launch, FC2 terminal-buffer
recycling, packed-FC2 vectorized stores, cluster-scoped descriptor synchronization
and launch-local shared-memory carveout preferences. Twelve generated device
translation units implement the selected dispatch. Existing optional activation,
owned-scratch, fused-FC1 and per-source JIT behavior is preserved.

Qualification passed 112 CPU feature tests, a fresh source/JIT build, 32 strict
BF16 numerical cases at atol=rtol=0.01, 576 strict timing postchecks, seven public
API/graph cases and eight dynamic-routing fixtures. Public outputs match the
source launcher bitwise across all 32 shapes. The subsequent fixed-main merge
preserves the selected source closure and passes the 112 CPU feature tests and
12-module JIT inventory check; the table retains the original GPU measurements.

Timing uses CUPTI with cold L2, six balanced captures per arm, and the geometric
mean of all six capture medians. Every measurement includes the complete per-call
planner, projections, finalization and public workspace synchronization. Ratios
above 1 favor the public API. Source/public has geometric mean 1.000155299×;
the export measured slower than its source-launcher control on 15 shapes.

| T | Public API (µs) | FlashInfer (µs) | FlashInfer/public | Source launcher (µs) | Source/public |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.387665 | 29.407550 | 1.114443037× | 26.484781 | 1.003680355× |
| 2 | 41.583749 | 44.756510 | 1.076298114× | 41.610606 | 1.000645873× |
| 3 | 56.335622 | 59.976458 | 1.064627597× | 56.319751 | 0.999718277× |
| 4 | 68.719672 | 71.460615 | 1.039885852× | 68.698188 | 0.999687362× |
| 5 | 79.242466 | 81.903009 | 1.033574715× | 79.417959 | 1.002214624× |
| 6 | 90.970399 | 92.421005 | 1.015945910× | 90.676993 | 0.996774710× |
| 7 | 100.111570 | 102.271917 | 1.021579391× | 100.031435 | 0.999199544× |
| 8 | 108.884969 | 110.191307 | 1.011997414× | 109.098616 | 1.001962131× |
| 9 | 121.039818 | 121.801850 | 1.006295710× | 121.044567 | 1.000039235× |
| 10 | 127.055646 | 127.818044 | 1.006000504× | 127.055800 | 1.000001217× |
| 11 | 134.602436 | 135.252960 | 1.004832932× | 134.394619 | 0.998456072× |
| 12 | 144.255821 | 145.081703 | 1.005725116× | 144.351252 | 1.000661538× |
| 13 | 153.775485 | 154.943467 | 1.007595376× | 153.588958 | 0.998787021× |
| 14 | 159.930292 | 160.687099 | 1.004732110× | 159.893160 | 0.999767824× |
| 15 | 167.605098 | 168.473853 | 1.005183347× | 167.679609 | 1.000444564× |
| 16 | 177.204607 | 178.063016 | 1.004844168× | 177.029323 | 0.999010840× |
| 17 | 182.618485 | 186.756839 | 1.022661203× | 183.183631 | 1.003094685× |
| 18 | 187.130155 | 190.772834 | 1.019466015× | 187.055491 | 0.999601003× |
| 19 | 192.548644 | 196.388681 | 1.019943206× | 192.847419 | 1.001551686× |
| 20 | 200.266476 | 204.143575 | 1.019359703× | 200.554117 | 1.001436294× |
| 21 | 208.058486 | 211.711656 | 1.017558381× | 208.276615 | 1.001048405× |
| 22 | 218.890141 | 223.204720 | 1.019711163× | 219.001817 | 1.000510190× |
| 23 | 222.186324 | 227.204805 | 1.022586814× | 222.233951 | 1.000214357× |
| 24 | 234.409988 | 238.415085 | 1.017085865× | 234.324972 | 0.999637320× |
| 25 | 242.127158 | 246.201867 | 1.016828795× | 242.047487 | 0.999670951× |
| 26 | 248.366962 | 251.828828 | 1.013938512× | 247.866121 | 0.997983462× |
| 27 | 253.812662 | 257.402025 | 1.014141782× | 253.962309 | 1.000589599× |
| 28 | 261.673974 | 264.964460 | 1.012574755× | 261.407290 | 0.998980855× |
| 29 | 268.911818 | 272.527208 | 1.013444519× | 268.751785 | 0.999404889× |
| 30 | 278.458135 | 281.999030 | 1.012716075× | 278.511635 | 1.000192130× |
| 31 | 282.244813 | 285.871512 | 1.012849479× | 282.217774 | 0.999904201× |
| 32 | 286.580986 | 291.407602 | 1.016842068× | 286.617819 | 1.000128527× |

The public API is slower than its source launcher at T=3,4,6,7,11,13,14,16,18,24,25,26,28,29,31. All 32 comparisons against the named FlashInfer baseline win.

Separate synccheck and racecheck invocations each reached their hard 20-second
timeout and were recorded as skipped. Neither is a sanitizer pass, and unchanged
timed-out checks were not retried.

Validation worker runtime was 5243.462359 seconds; the controller took 5246.206024
seconds. Managed turnaround was 5371.635377 seconds (5371.736368 seconds from
submission to completion). These durations are separate from GPU timings above.

Related public request: [FlashInfer issue #5184](https://github.com/flashinfer-ai/flashinfer/issues/5184).
