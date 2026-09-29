# DeepSeek-V4-Flash-0731 routed NVFP4 decode: public API results

On NVIDIA B200 (148 SMs, CUDA 13.3), the selected public API passes strict BF16
correctness for every T=1..32 and beats
`flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe` at every shape.
Geometric-mean speedup is **1.021459540×**; minimum is
**1.004273777× at T11**, with 32 wins, zero ties and zero baseline regressions.
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
and launch-local shared-memory carveout preferences. The T=1..9 direct FC2
uses a uniform register allocation with a 100% shared-memory carveout preference;
measured residency remains one block per SM. Thirteen generated device
translation units implement the selected dispatch. T=9 alone uses the
loader-bypass FC2 variant; the other 31 shape dispatches are unchanged.
The prior T=9 holdout split three winning and three losing pairs with a
1.000126478 source-control ratio, so a robust speedup is not claimed.
The fresh table compares against the named baseline and the matched source
launcher, not against the previous public release. Existing optional activation,
owned-scratch, fused-FC1 and per-source JIT behavior is preserved.

Qualification passed 112 CPU feature tests, a fresh source/JIT build, 32 strict
BF16 numerical cases at atol=rtol=0.01, 576 strict timing postchecks, seven public
API/graph cases and eight dynamic-routing fixtures. Public outputs match the
source launcher bitwise across all 32 shapes. The table reports fresh measurements
of the T=9 loader-bypass selection. All 576 raw capture medians were independently
recomputed; no previous GPU measurements were reused.

Timing uses CUPTI with cold L2, six balanced captures per arm, and the geometric
mean of all six capture medians. Every measurement includes the complete per-call
planner, projections, finalization and public workspace synchronization. Ratios
above 1 favor the public API. Source/public has geometric mean 1.000087017×;
the export measured slower than its source-launcher control on 16 shapes.

| T | Public API (µs) | FlashInfer (µs) | FlashInfer/public | Source launcher (µs) | Source/public |
|---:|---:|---:|---:|---:|---:|
| 1 | 25.173127 | 29.290267 | 1.163552987× | 25.210700 | 1.001492588× |
| 2 | 41.482569 | 44.942979 | 1.083418402× | 41.466595 | 0.999614913× |
| 3 | 56.042492 | 60.600281 | 1.081327391× | 55.904145 | 0.997531400× |
| 4 | 67.962326 | 72.042202 | 1.060031431× | 67.930470 | 0.999531264× |
| 5 | 79.551985 | 82.538102 | 1.037536680× | 79.578621 | 1.000334822× |
| 6 | 91.482481 | 93.194303 | 1.018712020× | 91.290664 | 0.997903242× |
| 7 | 100.415731 | 103.189436 | 1.027622215× | 100.495604 | 1.000795421× |
| 8 | 109.610817 | 111.268987 | 1.015127800× | 109.877267 | 1.002430880× |
| 9 | 121.824315 | 123.066404 | 1.010195739× | 122.016046 | 1.001573830× |
| 10 | 128.458479 | 129.109113 | 1.005064933× | 128.421464 | 0.999711848× |
| 11 | 136.058427 | 136.639910 | 1.004273777× | 135.871960 | 0.998629507× |
| 12 | 145.892985 | 146.618580 | 1.004973471× | 145.978595 | 1.000586797× |
| 13 | 155.530651 | 156.687844 | 1.007440289× | 155.359971 | 0.998902600× |
| 14 | 161.797200 | 162.490324 | 1.004283908× | 161.733329 | 0.999605238× |
| 15 | 169.551771 | 170.325087 | 1.004560944× | 169.642953 | 1.000537781× |
| 16 | 179.226770 | 180.063649 | 1.004669388× | 179.077485 | 0.999167063× |
| 17 | 184.714818 | 187.503729 | 1.015098469× | 185.306625 | 1.003203896× |
| 18 | 189.279992 | 191.546311 | 1.011973369× | 189.226818 | 0.999719074× |
| 19 | 194.783983 | 197.252972 | 1.012675524× | 195.125270 | 1.001752129× |
| 20 | 202.586812 | 205.002358 | 1.011923509× | 202.906785 | 1.001579437× |
| 21 | 210.501315 | 212.733043 | 1.010601966× | 210.762536 | 1.001240946× |
| 22 | 221.514646 | 224.426649 | 1.013145870× | 221.631981 | 1.000529692× |
| 23 | 224.901325 | 228.352244 | 1.015344149× | 224.901294 | 0.999999862× |
| 24 | 237.280317 | 239.823974 | 1.010720051× | 237.109479 | 0.999280018× |
| 25 | 245.055985 | 247.647873 | 1.010576720× | 244.992076 | 0.999739209× |
| 26 | 251.383972 | 253.450579 | 1.008220918× | 250.890948 | 0.998038762× |
| 27 | 256.933994 | 259.021343 | 1.008124070× | 257.077472 | 1.000558424× |
| 28 | 264.917306 | 266.699377 | 1.006726896× | 264.666635 | 0.999053779× |
| 29 | 272.191986 | 274.416535 | 1.008172721× | 272.010802 | 0.999334354× |
| 30 | 281.845625 | 283.866746 | 1.007171024× | 281.893642 | 1.000170369× |
| 31 | 285.669488 | 287.872028 | 1.007710098× | 285.733447 | 1.000223891× |
| 32 | 290.042993 | 293.274817 | 1.011142570× | 290.053654 | 1.000036758× |

The public API is slower than its source launcher at T=2, 3, 4, 6, 10, 11, 13, 14, 16, 18, 23, 24, 25, 26, 28, 29. All 32 comparisons against the named FlashInfer baseline win.

Separate synccheck and racecheck invocations each reached their hard 20-second
timeout and were recorded as skipped. Neither is a sanitizer pass, and unchanged
timed-out checks were not retried.

Validation worker runtime was 5656.004092 seconds; the controller took
5658.897162 seconds. Physical turnaround from submission to completion
was 5774.637452 seconds. These durations are separate from GPU timings above.

Related public request: [FlashInfer issue #5184](https://github.com/flashinfer-ai/flashinfer/issues/5184).
