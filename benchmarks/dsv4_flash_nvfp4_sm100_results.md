# DeepSeek-V4-Flash-0731 routed NVFP4 decode: public API results

On NVIDIA B200 (148 SMs, CUDA 13.3), the public API passes strict BF16 correctness
for every T=1..32 and is faster than
`flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe` at every shape.
Geometric-mean speedup: **1.012004951×**. Minimum: **1.000042980× at T9**
(123.045108 µs public API versus 123.050396 µs baseline). The minimum margin is
approximately **0.00430%**; these measurements do not establish a hardware limit.

Geometry: H=4096, I=2048, E=256 routed experts, top-k=6, clamped SwiGLU limit=10.
The shared expert is excluded. Candidate and baseline use identical physical
NVFP4 shuffled MajorK / R128c4 inputs, routes and scales, with equivalent clamped
activation parameters. BF16 checks use atol=rtol=0.01. Public outputs also match
the source implementation bitwise across all 32 shapes. Source/JIT integration
passed 49 CPU tests; the API lifecycle and dynamic-route checks are preserved.

Timing uses CUPTI with cold L2, six balanced captures per arm, and the geometric
mean of all six capture medians. Every measurement includes the complete per-call
planner, projections and finalization, plus public workspace synchronization.
No row, tie or regression is omitted. Ratios above1 favor the public API.
“Source/public” compares the same selected schedule through its source launcher
and the public API; the public API is slower on 18 of 32 shapes in that comparison.

| T | Public API (µs) | FlashInfer (µs) | FlashInfer/public | Source launcher (µs) | Source/public |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.378429 | 28.703039 | 1.088125401× | 26.260864 | 0.995543127× |
| 2 | 41.818307 | 44.202015 | 1.057001532× | 41.781102 | 0.999110319× |
| 3 | 56.778432 | 60.079341 | 1.058136667× | 56.901133 | 1.002161041× |
| 4 | 69.456042 | 71.951565 | 1.035929534× | 69.333123 | 0.998230273× |
| 5 | 79.914084 | 82.538124 | 1.032835763× | 80.421104 | 1.006344567× |
| 6 | 91.759604 | 93.290607 | 1.016684933× | 91.631785 | 0.998607027× |
| 7 | 101.023934 | 103.231936 | 1.021856221× | 101.072085 | 1.000476624× |
| 8 | 110.031776 | 111.274427 | 1.011293561× | 110.143957 | 1.001019528× |
| 9 | 123.045108 | 123.050396 | 1.000042980× | 122.986649 | 0.999524898× |
| 10 | 128.442434 | 129.023696 | 1.004525463× | 128.650614 | 1.001620802× |
| 11 | 136.671776 | 136.794132 | 1.000895255× | 136.629115 | 0.999687858× |
| 12 | 146.164956 | 146.596926 | 1.002955362× | 146.692977 | 1.003612504× |
| 13 | 156.554608 | 156.644757 | 1.000575832× | 156.394470 | 0.998977109× |
| 14 | 162.101304 | 162.410248 | 1.001905872× | 161.999944 | 0.999374713× |
| 15 | 169.733027 | 170.351646 | 1.003644663× | 170.074467 | 1.002011630× |
| 16 | 179.471795 | 180.159585 | 1.003832305× | 179.727819 | 1.001426546× |
| 17 | 185.167736 | 186.431849 | 1.006826851× | 185.509147 | 1.001843792× |
| 18 | 189.738662 | 190.431960 | 1.003653963× | 189.359797 | 0.998003227× |
| 19 | 195.221144 | 196.037318 | 1.004180766× | 195.322297 | 1.000518145× |
| 20 | 202.980448 | 203.674614 | 1.003419863× | 202.917303 | 0.999688908× |
| 21 | 210.714657 | 211.413001 | 1.003314169× | 211.311488 | 1.002832411× |
| 22 | 222.239816 | 223.135933 | 1.004032210× | 222.042154 | 0.999110592× |
| 23 | 226.047814 | 226.938059 | 1.003938306× | 225.727803 | 0.998584324× |
| 24 | 237.845158 | 238.565440 | 1.003028366× | 237.370630 | 0.998004883× |
| 25 | 245.290136 | 246.293279 | 1.004089619× | 245.343975 | 1.000219492× |
| 26 | 251.434646 | 251.887802 | 1.001802281× | 251.205326 | 0.999087953× |
| 27 | 257.231980 | 257.903959 | 1.002612346× | 257.343971 | 1.000435371× |
| 28 | 265.285159 | 265.679810 | 1.001487648× | 264.975816 | 0.998833924× |
| 29 | 272.602311 | 273.466572 | 1.003170410× | 272.042325 | 0.997945780× |
| 30 | 282.239645 | 282.629308 | 1.001380609× | 282.143830 | 0.999660517× |
| 31 | 286.319987 | 286.591986 | 1.000949981× | 286.319745 | 0.999999156× |
| 32 | 291.482470 | 292.159519 | 1.002322777× | 291.525159 | 1.000146454× |

The public API is slower than the source launcher at T=1,2,4,6,9,11,13,14,18,20,
22,23,24,26,28,29,30,31. These regressions are separate from the all 32 wins
against the named FlashInfer baseline.

Synccheck and racecheck were invoked separately on the measured public library.
Both were **skipped after their hard 20-second timeout**; both logs were empty
and neither produced a final error summary. They are not passes and were not
retried. Library SHA256:
`3ad967f0413e174bfc16a8680c0bbf48b329cc3971435dc6b682485e0c548b2f`.

This result describes the currently exported schedule. A separate packed-FC2
vectorized-global-store experiment has not been exported or included in these
numbers. Final schedule selection remains under review.

The resumed validation attempt took 1638.254seconds of worker time and
1759.894 seconds of managed turnaround. An earlier cancelled attempt took
3768.903 seconds of managed turnaround; its 3648.179-second worker checkpoint was
not a final runtime measurement. Only sealed rows with matching source/library,
input and timing identities were retained across the resume. Resume-inclusive orchestration turnaround from the first submission to final
completion was **5827.299 seconds**, including the interruption and restart. Total
worker time across both attempts is unknown. These orchestration durations are
separate from the GPU timings in the table.
