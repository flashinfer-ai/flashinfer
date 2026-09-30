# DeepSeek-V4-Flash-0731 routed NVFP4 decode: B200 results

The public API passes strict BF16 correctness and beats `flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe` at every integer T=1..32. Public/FlashInfer geometric-mean speedup is **1.023236704395×**, with a minimum of **1.003166005574× at T14**. The source launcher also wins every shape: **1.023174595913×** geometric mean, minimum **1.001675704624× at T12**. Neither comparison has ties or baseline regressions.

Geometry is H=4096, I=2048, E=256 routed experts, top-k=6 and clamped SwiGLU limit=10, with BF16 output. The shared expert is outside this routed kernel. Baseline and candidate consume identical physical NVFP4 shuffled MajorK/R128c4 weights and activation representations, routes, scales and equivalent clamped-activation parameters.

Packed FC1 now transposes logical tile order within each complete eight-group stripe. The partial final stripe keeps its original order. Constant shifts and masks implement the permutation; each tile retains its exact arithmetic, quantization, clamp, scale association and output address. Raw work-stealing identities, phases, synchronization, launch ABI and all other device units remain unchanged. The change applies to packed FC1 at T10..32, preserving the existing T1..9 dispatch. Resources remain 352 threads, 48 registers and 64,128 bytes of dynamic shared memory.

Validation covers 112 CPU feature tests, 13-module source/JIT and loaded-binding closure, 32 strict numerical/source-bitwise cases at atol=rtol=0.01, 576 strict timing postchecks, seven API/graph cases and eight routing fixtures. Eager and graph public outputs match the source launcher bitwise. A separate registered source benchmark also passed all 32 shapes with unchanged thresholds. Its initial current-run attempt timed out after 30 sealed shapes; those exact rows were retained, and a continuation measured T31/T32 afresh. This is a resumed current run, not an uninterrupted run or reuse of historical qualification.

Eight disjoint four-shape timing shards use six balanced captures per arm, CUPTI and cold L2. Each shape's source, public and FlashInfer arms use the same GPU and physical inputs. The timing boundary includes all required per-call planning, FC1, FC2, routing-weighted finalization and synchronization. Reported times are geometric means of the six capture medians.

Source/public geometric mean is **1.000060701744×**; a ratio above 1 favors the public API. The public API is slower than source at T=1, 7, 10, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 31. All 15 such regressions are included below. These measurements compare the current source and exported implementation against FlashInfer; they are not an all 32 causal A/B comparison against the preceding pair-pack kernel. The earlier matched T15 pilot showed a small source regression, which remains part of the optimization record; no threshold was selected to hide it.

| T | Public (µs) | FlashInfer (µs) | FlashInfer/public | Source (µs) | FlashInfer/source | Source/public |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.140630 | 29.309442 | 1.165819713× | 25.012054 | 1.171812667× | 0.994885741× |
| 2 | 41.231592 | 44.613083 | 1.082012131× | 41.237215 | 1.081864610× | 1.000136358× |
| 3 | 55.924570 | 59.999018 | 1.072856144× | 56.074261 | 1.069992133× | 1.002676665× |
| 4 | 67.845123 | 72.543520 | 1.069251807× | 68.175605 | 1.064068589× | 1.004871132× |
| 5 | 79.935599 | 83.119246 | 1.039827648× | 79.988789 | 1.039136188× | 1.000665419× |
| 6 | 91.583283 | 93.962133 | 1.025974718× | 91.620982 | 1.025552560× | 1.000411640× |
| 7 | 100.820792 | 103.631304 | 1.027876319× | 100.719751 | 1.028907473× | 0.998997817× |
| 8 | 109.732649 | 111.369506 | 1.014916770× | 109.855410 | 1.013782623× | 1.001118727× |
| 9 | 121.668812 | 123.012721 | 1.011045635× | 121.679314 | 1.010958368× | 1.000086321× |
| 10 | 128.516425 | 129.343341 | 1.006434322× | 128.495290 | 1.006599864× | 0.999835543× |
| 11 | 135.972942 | 137.146476 | 1.008630643× | 136.079152 | 1.007843406× | 1.000781111× |
| 12 | 145.919108 | 146.655458 | 1.005046290× | 146.410118 | 1.001675705× | 1.003364947× |
| 13 | 155.215640 | 156.727345 | 1.009739386× | 155.535819 | 1.007660777× | 1.002062806× |
| 14 | 161.897580 | 162.410148 | 1.003166006× | 161.780948 | 1.003889212× | 0.999279595× |
| 15 | 169.753923 | 170.409592 | 1.003862465× | 169.716446 | 1.004084141× | 0.999779226× |
| 16 | 179.385980 | 180.452385 | 1.005944753× | 179.338151 | 1.006213035× | 0.999733375× |
| 17 | 185.599653 | 189.145753 | 1.019106180× | 185.092475 | 1.021898668× | 0.997267353× |
| 18 | 189.375291 | 191.940283 | 1.013544493× | 189.092490 | 1.015060320× | 0.998506662× |
| 19 | 194.996793 | 197.700431 | 1.013865042× | 194.783160 | 1.014977023× | 0.998904428× |
| 20 | 202.868631 | 205.081861 | 1.010909673× | 202.735323 | 1.011574391× | 0.999342888× |
| 21 | 210.841967 | 213.220362 | 1.011280463× | 210.303428 | 1.013870120× | 0.997445771× |
| 22 | 221.796818 | 225.252734 | 1.015581448× | 221.924637 | 1.014996519× | 1.000576287× |
| 23 | 225.428287 | 229.236077 | 1.016891356× | 225.449791 | 1.016794363× | 1.000095391× |
| 24 | 237.694963 | 241.070886 | 1.014202756× | 237.241795 | 1.016140037× | 0.998093490× |
| 25 | 245.470947 | 248.490010 | 1.012299064× | 245.417975 | 1.012517562× | 0.999784203× |
| 26 | 250.943159 | 253.961973 | 1.012029870× | 250.820485 | 1.012524843× | 0.999511149× |
| 27 | 256.863321 | 259.802145 | 1.011441195× | 257.103324 | 1.010497029× | 1.000934358× |
| 28 | 264.532330 | 267.337781 | 1.010605323× | 264.590951 | 1.010381421× | 1.000221602× |
| 29 | 272.127320 | 274.660244 | 1.009307866× | 272.564485 | 1.007689042× | 1.001606472× |
| 30 | 281.689660 | 284.260596 | 1.009126837× | 281.860288 | 1.008515949× | 1.000605730× |
| 31 | 285.684274 | 288.169482 | 1.008699141× | 285.476324 | 1.009433912× | 0.999272096× |
| 32 | 289.769800 | 294.804113 | 1.017373489× | 290.100254 | 1.016214598× | 1.001140400× |


Timings use matched synthetic NVFP4 fixtures, not full actual-checkpoint pipeline timings. Checkpoint packing/conversion and clamp semantics were verified separately. The separate synchronization and race sanitizer invocations each timed out after 20 seconds and remain skipped, not passes; they were not retried. Hardware-limit evidence remains unfinished.

GPU timing-batch physical turnaround was 3978.445475 seconds from first submission to last completion, including scheduling and orchestration gaps. Aggregate worker runtime was 5595.750496 seconds; independent raw reduction used 69.460126 worker seconds. These durations are separate from the GPU timings in the table.

Related: [FlashInfer issue #5184](https://github.com/flashinfer-ai/flashinfer/issues/5184).
