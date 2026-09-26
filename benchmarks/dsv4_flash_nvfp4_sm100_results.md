# DeepSeek-V4-Flash-0731 routed NVFP4 decode: public API results

On NVIDIA B200 (148 SMs, CUDA 13.3), the public API passes strict BF16 correctness
for every T=1..32 and is faster than
`flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe` at every shape.
Geometric-mean speedup: **1.015944701×**. Minimum:
**1.001463411× at T32** (291.540136 µs public API versus
291.966779 µs baseline). These measurements alone do not establish a hardware limit.

Geometry: H=4096, I=2048, E=256 routed experts, top-k=6, clamped SwiGLU limit=10.
The shared expert is excluded. Candidate and baseline use identical physical
NVFP4 shuffled MajorK / R128c4 inputs, routes and scales, with equivalent clamped
activation parameters. BF16 checks use atol=rtol=0.01. Public outputs also match
the source implementation bitwise across all 32 shapes. Source/JIT integration
passed 49 CPU tests; API lifecycle and dynamic-route checks are preserved.

The selected schedule includes packed-FC2 vectorized global stores and launch-local
shared-memory carveout preferences. CUDA graph checks verify matching requested
attributes through the public API and source launcher for all 32 shapes.

Timing uses CUPTI with cold L2, six balanced captures per arm, and the geometric
mean of all six capture medians. Every measurement includes the complete per-call
planner, projections and finalization, plus public workspace synchronization.
No row, tie or regression is omitted. Ratios above 1 favor the public API.
“Source/public” compares the same selected schedule through its source launcher
and the public API; the public API is slower on 10 of 32 shapes in that comparison.

| T | Public API (µs) | FlashInfer (µs) | FlashInfer/public | Source launcher (µs) | Source/public |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.058110 | 29.273335 | 1.123386714× | 26.159885 | 1.003905695× |
| 2 | 41.754518 | 44.420616 | 1.063851732× | 41.775757 | 1.000508655× |
| 3 | 56.351451 | 60.042119 | 1.065493739× | 56.378090 | 1.000472726× |
| 4 | 68.276751 | 71.226102 | 1.043197000× | 68.250290 | 0.999612435× |
| 5 | 79.535625 | 83.017744 | 1.043780624× | 79.573144 | 1.000471729× |
| 6 | 91.519474 | 93.935243 | 1.026396233× | 91.530291 | 1.000118195× |
| 7 | 101.418295 | 103.833925 | 1.023818484× | 101.109150 | 0.996951783× |
| 8 | 109.902971 | 111.465524 | 1.014217570× | 110.009973 | 1.000973607× |
| 9 | 122.503377 | 123.162315 | 1.005378937× | 122.484810 | 0.999848441× |
| 10 | 128.057971 | 129.641935 | 1.012369109× | 128.388619 | 1.002582017× |
| 11 | 136.305808 | 137.802230 | 1.010978414× | 136.729966 | 1.003111809× |
| 12 | 146.495102 | 147.636455 | 1.007791066× | 146.761812 | 1.001820602× |
| 13 | 157.183146 | 157.503279 | 1.002036688× | 156.943270 | 0.998473906× |
| 14 | 162.964487 | 163.407241 | 1.002716873× | 162.788972 | 0.998922989× |
| 15 | 170.356480 | 171.140090 | 1.004599820× | 170.447492 | 1.000534243× |
| 16 | 179.572654 | 180.649637 | 1.005997474× | 179.545816 | 0.999850541× |
| 17 | 185.460325 | 186.583025 | 1.006053586× | 185.257940 | 0.998908743× |
| 18 | 189.727324 | 190.319119 | 1.003119190× | 189.769794 | 1.000223851× |
| 19 | 195.263160 | 196.158909 | 1.004587390× | 195.561743 | 1.001529128× |
| 20 | 203.052292 | 203.823269 | 1.003796939× | 203.156567 | 1.000513539× |
| 21 | 210.846941 | 211.588242 | 1.003515825× | 210.900463 | 1.000253844× |
| 22 | 222.169605 | 223.305533 | 1.005112888× | 222.153607 | 0.999927994× |
| 23 | 226.001561 | 227.158903 | 1.005120946× | 225.790967 | 0.999068176× |
| 24 | 237.572395 | 238.873327 | 1.005475937× | 237.908627 | 1.001415283× |
| 25 | 245.582839 | 246.425752 | 1.003432297× | 245.806936 | 1.000912512× |
| 26 | 250.782970 | 252.105818 | 1.005274871× | 251.401970 | 1.002468271× |
| 27 | 256.714148 | 258.073707 | 1.005296005× | 257.118815 | 1.001576334× |
| 28 | 264.564135 | 265.604412 | 1.003932040× | 264.943121 | 1.001432489× |
| 29 | 272.495143 | 273.396348 | 1.003307233× | 272.438983 | 0.999793906× |
| 30 | 282.148142 | 282.750764 | 1.002135837× | 282.388320 | 1.000851250× |
| 31 | 285.934791 | 286.660021 | 1.002536348× | 286.323980 | 1.001361111× |
| 32 | 291.540136 | 291.966779 | 1.001463411× | 291.822808 | 1.000969580× |

The public API is slower than the source launcher at T=4,7,9,13,14,16,17,22,23,29.
These differences are separate from the 32 wins against the named FlashInfer baseline.

Earlier synccheck and racecheck invocations each reached their hard 20-second
timeout and were recorded as skipped. Neither establishes a sanitizer pass for
this exported library.

The validation worker took 6822.060809 seconds; its controller took
6827.345842 seconds. End-to-end managed turnaround was
6945.966330 seconds. These orchestration durations are separate from the GPU
timings in the table.
