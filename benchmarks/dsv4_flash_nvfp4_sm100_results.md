# DeepSeek-V4-Flash-0731 routed NVFP4 decode: public API results

On NVIDIA B200 (148 SMs, CUDA 13.3), the selected public API passes strict BF16 correctness for every T=1..32 and beats `flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe` at every shape. Public/FlashInfer geometric-mean speedup is **1.022287477905×**, with minimum **1.003460733490× at T15**. All 32 shapes win, with no ties or baseline regressions. Hardware-limit evidence and remaining promotion work are still pending.

Geometry: H=4096, I=2048, E=256 routed experts, top-k=6, clamped SwiGLU limit=10. The shared expert is outside this routed path. Candidate and baseline consume identical physical NVFP4 shuffled MajorK / R128c4 inputs, routes and scales, with equivalent clamped activation parameters and BF16 rounding semantics.

Benchmarks use matched NVFP4 physical fixtures at E=256. Separately, all 256 layer-0 routed experts from the pinned FP4 checkpoint passed exact scale conversion and FlashInfer MajorK/R128c4 layout validation on B200. The native model's FP8-activation pipeline is a different arithmetic path; numerical equivalence to it is not claimed. These are not full actual-checkpoint pipeline timings.

The packed FC1 epilogue removes one 64-thread barrier after the waited TMEM load and before warp-local register arithmetic. The two epilogue warps own disjoint output and scale locations. The final 64-thread barrier, all four accumulator-release arrivals, persistent phases, clamp/quantization arithmetic and teardown remain unchanged. Dispatch, launch arguments, resources and the other 12 generated device units are preserved.

The existing schedule retains early dependent launch, FC2 terminal-buffer recycling, packed-FC2 vectorized stores, cluster-scoped descriptor synchronization and launch-local shared-memory carveout preferences. T=1..9 direct FC2 uses uniform registers and a 100% shared-memory carveout preference. The occupancy API reports one block per SM; this static query does not establish dynamic residency. T=9 retains its loader-bypass FC2 variant. Thirteen generated device translation units implement the selected dispatch. Existing optional activation, owned-scratch, fused-FC1 and per-source JIT behavior is preserved.

Qualification passed 112 CPU feature tests, a fresh normal source/JIT check, 32 strict BF16 numerical cases at atol=rtol=0.01, 576 strict timing postchecks, seven public API/graph cases and eight routing fixtures. Public outputs match the source launcher bitwise for all 32 shapes. Independent reduction verified all576 retained capture medians and native complete-call activity boundaries.

Eight disjoint four-shape workers ran on four GPUs. Every shape's three arms ran on the same GPU with the same physical inputs. Timing uses CUPTI/cold-L2, six balanced captures per arm and the geometric mean of all six capture medians. Each measurement includes the complete per-call planner, projections, finalization and required public workspace synchronization. Before/after clock queries succeeded for every worker. These are fresh matched comparisons; no causal improvement over the previous public cohort is inferred from cross-run results.

Source/FlashInfer geometric-mean speedup is **1.022237384293×**, with all 32 shapes winning; minimum source speedup is **1.003684405007× at T12**. Source/public geometric mean is **1.000049003894×**. Ratios above 1 in the source/public column favor the public API.

| T | Public (µs) | FlashInfer (µs) | FlashInfer/public | Source (µs) | FlashInfer/source | Source/public |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.172214 | 29.234316 | 1.161372425× | 25.054903 | 1.166810195× | 0.995339628× |
| 2 | 41.231727 | 44.628984 | 1.082394251× | 41.290289 | 1.080859076× | 1.001420328× |
| 3 | 55.940880 | 59.999714 | 1.072555785× | 56.085098 | 1.069797792× | 1.002578050× |
| 4 | 67.951446 | 72.500703 | 1.066948649× | 68.314436 | 1.061279386× | 1.005341914× |
| 5 | 79.902947 | 82.991199 | 1.038650040× | 79.935122 | 1.038231962× | 1.000402683× |
| 6 | 91.562269 | 93.439651 | 1.020503881× | 91.663484 | 1.019377040× | 1.001105421× |
| 7 | 100.862626 | 103.588135 | 1.027021988× | 100.809586 | 1.027562349× | 0.999474134× |
| 8 | 109.764132 | 111.395551 | 1.014862953× | 109.860227 | 1.013975250× | 1.000875468× |
| 9 | 121.646641 | 122.996392 | 1.011095675× | 121.662900 | 1.010960551× | 1.000133660× |
| 10 | 128.654794 | 129.459975 | 1.006258460× | 128.660277 | 1.006215577× | 1.000042618× |
| 11 | 135.753780 | 136.703091 | 1.006992886× | 135.732786 | 1.007148637× | 0.999845355× |
| 12 | 145.694736 | 146.611808 | 1.006294475× | 146.073614 | 1.003684405× | 1.002600489× |
| 13 | 155.299980 | 156.857562 | 1.010029506× | 155.513486 | 1.008642823× | 1.001374801× |
| 14 | 161.657729 | 162.404493 | 1.004619419× | 161.583098 | 1.005083421× | 0.999538345× |
| 15 | 169.715257 | 170.302597 | 1.003460733× | 169.667444 | 1.003743514× | 0.999718275× |
| 16 | 179.166452 | 180.227688 | 1.005923182× | 179.102249 | 1.006283781× | 0.999641653× |
| 17 | 185.357977 | 187.512741 | 1.011624878× | 185.016945 | 1.013489554× | 0.998160143× |
| 18 | 189.150623 | 191.907772 | 1.014576470× | 188.974315 | 1.015523047× | 0.999067892× |
| 19 | 194.804303 | 197.689427 | 1.014810372× | 194.569814 | 1.016033386× | 0.998796286× |
| 20 | 202.942471 | 205.214393 | 1.011194910× | 202.638484 | 1.012711846× | 0.998502106× |
| 21 | 210.531465 | 213.101792 | 1.012208756× | 210.094256 | 1.014315174× | 0.997923310× |
| 22 | 221.497299 | 224.606595 | 1.014037626× | 221.647142 | 1.013352092× | 1.000676502× |
| 23 | 225.214292 | 229.331592 | 1.018281700× | 225.304788 | 1.017872697× | 1.000401821× |
| 24 | 237.811629 | 241.038041 | 1.013567091× | 237.389943 | 1.015367535× | 0.998226806× |
| 25 | 245.373940 | 248.461707 | 1.012583923× | 245.261961 | 1.013046240× | 0.999543637× |
| 26 | 250.931989 | 254.083300 | 1.012558424× | 250.979987 | 1.012364782× | 1.000191277× |
| 27 | 256.953644 | 259.177639 | 1.008655240× | 256.767311 | 1.009387207× | 0.999274840× |
| 28 | 264.584410 | 267.187417 | 1.009838096× | 264.637939 | 1.009633836× | 1.000202311× |
| 29 | 272.382639 | 274.894030 | 1.009220083× | 272.755483 | 1.007840528× | 1.001368823× |
| 30 | 281.876331 | 284.105268 | 1.007907502× | 281.766948 | 1.008298774× | 0.999611948× |
| 31 | 285.773785 | 288.003002 | 1.007800634× | 285.682820 | 1.008121531× | 0.999681688× |
| 32 | 290.253644 | 293.139251 | 1.009941673× | 290.413491 | 1.009385788× | 1.000550716× |

The public API is slower than its source launcher at T=1, 7, 11, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 30, 31. All 32 public and source comparisons against the named FlashInfer baseline win.

Separate synccheck and racecheck checks reached their hard 20-second process-level limits and remain skipped. Neither is a sanitizer pass, and unchanged timed-out checks were not retried. The inherited static cross-rank collector refusal remains unresolved; static analysis is not claimed as complete synchronization proof.

The GPU batch's physical submission-to-completion turnaround was 2213.087809s; aggregate worker time was 5609.447757s across eight workers. Independent reduction used 68.990126s of worker time and 81.470710s physical turnaround. These are separate from the GPU timings in the table.

Related public request: [FlashInfer issue #5184](https://github.com/flashinfer-ai/flashinfer/issues/5184).
