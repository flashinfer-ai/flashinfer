# DeepSeek-V4-Flash-0731 routed NVFP4 decode: public API results

On NVIDIA B200 (148 SMs, CUDA 13.3), the selected public API passes strict BF16
correctness for every T=1..32 and beats
`flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe` at every shape.
Geometric-mean speedup is **1.022362975×**; minimum is
**1.004966348× at T14**, with 32 wins, zero ties and zero baseline regressions.
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
measured residency remains one block per SM. Twelve generated device
translation units implement the selected dispatch. Existing optional activation,
owned-scratch, fused-FC1 and per-source JIT behavior is preserved.

Qualification passed 112 CPU feature tests, a fresh source/JIT build, 32 strict
BF16 numerical cases at atol=rtol=0.01, 576 strict timing postchecks, seven public
API/graph cases and eight dynamic-routing fixtures. Public outputs match the
source launcher bitwise across all 32 shapes. The table reports fresh measurements
of the uniform direct-FC2 export. All 576 raw capture medians were independently
recomputed; no previous GPU measurements were reused.

Timing uses CUPTI with cold L2, six balanced captures per arm, and the geometric
mean of all six capture medians. Every measurement includes the complete per-call
planner, projections, finalization and public workspace synchronization. Ratios
above 1 favor the public API. Source/public has geometric mean 0.999687122×;
the export measured slower than its source-launcher control on 19 shapes.

| T | Public API (µs) | FlashInfer (µs) | FlashInfer/public | Source launcher (µs) | Source/public |
|---:|---:|---:|---:|---:|---:|
| 1 | 25.044879 | 29.257821 | 1.168215706× | 24.991565 | 0.997871261× |
| 2 | 41.279047 | 44.740972 | 1.083866393× | 41.282344 | 1.000079877× |
| 3 | 56.580648 | 60.543063 | 1.070031286× | 56.484784 | 0.998305713× |
| 4 | 67.866206 | 72.538281 | 1.068842430× | 67.823695 | 0.999373612× |
| 5 | 79.503819 | 83.167395 | 1.046080512× | 79.487474 | 0.999794420× |
| 6 | 91.722309 | 93.706057 | 1.021627752× | 91.124664 | 0.993484183× |
| 7 | 100.650142 | 103.359305 | 1.026916638× | 100.591638 | 0.999418742× |
| 8 | 109.791614 | 111.567925 | 1.016178937× | 110.105992 | 1.002863405× |
| 9 | 121.844650 | 123.274140 | 1.011732065× | 122.084656 | 1.001969769× |
| 10 | 128.634292 | 129.444557 | 1.006298983× | 128.586144 | 0.999625703× |
| 11 | 136.100819 | 137.076587 | 1.007169449× | 135.897631 | 0.998507076× |
| 12 | 146.202482 | 146.932423 | 1.004992668× | 146.079494 | 0.999158783× |
| 13 | 155.679316 | 156.991292 | 1.008427428× | 155.225984 | 0.997088042× |
| 14 | 162.116569 | 162.921696 | 1.004966348× | 161.823478 | 0.998192095× |
| 15 | 169.732985 | 170.745890 | 1.005967638× | 169.854988 | 1.000718796× |
| 16 | 179.450239 | 180.425887 | 1.005436871× | 179.444389 | 0.999967401× |
| 17 | 185.108808 | 187.785782 | 1.014461624× | 185.470984 | 1.001956561× |
| 18 | 189.785985 | 191.780728 | 1.010510488× | 189.551135 | 0.998762555× |
| 19 | 195.247485 | 197.641654 | 1.012262226× | 195.305661 | 1.000297957× |
| 20 | 202.932487 | 205.337706 | 1.011852311× | 203.113989 | 1.000894394× |
| 21 | 210.735319 | 213.097768 | 1.011210504× | 211.151070 | 1.001972858× |
| 22 | 221.737807 | 224.745567 | 1.013564488× | 221.849802 | 1.000505075× |
| 23 | 224.991316 | 228.526920 | 1.015714401× | 224.953656 | 0.999832619× |
| 24 | 237.495308 | 240.105730 | 1.010991470× | 237.385986 | 0.999539692× |
| 25 | 245.358974 | 248.036535 | 1.010912834× | 245.305987 | 0.999784042× |
| 26 | 251.529825 | 253.788615 | 1.008980206× | 251.113828 | 0.998346131× |
| 27 | 257.092312 | 259.391066 | 1.008941355× | 257.359078 | 1.001037629× |
| 28 | 264.948655 | 267.047019 | 1.007919890× | 264.895164 | 0.999798109× |
| 29 | 272.522156 | 274.767224 | 1.008238112× | 272.276655 | 0.999099152× |
| 30 | 282.159158 | 284.292866 | 1.007562070× | 282.255573 | 1.000341704× |
| 31 | 285.844489 | 288.281844 | 1.008526857× | 286.052661 | 1.000728271× |
| 32 | 289.855322 | 293.609739 | 1.012952728× | 290.063468 | 1.000718104× |

The public API is slower than its source launcher at T=1, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 16, 18, 23, 24, 25, 26, 28, 29. All 32 comparisons against the named FlashInfer baseline win.

Separate synccheck and racecheck invocations each reached their hard 20-second
timeout and were recorded as skipped. Neither is a sanitizer pass, and unchanged
timed-out checks were not retried.

Validation worker runtime was 5502.463432 seconds; the controller took
5505.299887 seconds. Physical turnaround from submission to completion
was 5632.043565 seconds. These durations are separate from GPU timings above.

Related public request: [FlashInfer issue #5184](https://github.com/flashinfer-ai/flashinfer/issues/5184).
