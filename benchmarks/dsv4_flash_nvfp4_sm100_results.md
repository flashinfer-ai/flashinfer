# Clamped NVFP4 activation rounding — measured results

**Fresh public qualification passed all 32 shapes. Hardware-limit evidence remains incomplete.**

The repair preserves the matched clamped-SwiGLU instruction sequence before FP4 quantization. Up is clamped symmetrically and gate is capped above in the raw accumulator domain. Explicit float32 multiplication boundaries preserve alpha/log2/gate-scale association, FMA, approximate base-2 exponential and reciprocal, and gate-times-reciprocal ordering. ScaleC scales the pairwise maxima before their reduction and is applied after inverse block scaling before FP4 conversion. These instructions reproduce the named baseline globally; they do not introduce input-dependent approximation or corrections for specific values.

At one observed T4 boundary,13.999999046325684 versus14.0 changed the normalized activation across the FP4 midpoint1.75, producing1.5 versus2.0. The selected kernel and FlashInfer agreed there; the prior independent reference differed. A separate T17 case showed agreement between the old kernel and reference but disagreement with FlashInfer. Recomputing both complete saved cases with the globally matched activation/ScaleC sequence produced BF16 output byte-identical to FlashInfer. This diagnostic motivated the repair and does not replace fresh exported-kernel qualification.

This arithmetic update changes three generated FC1 device files. The public ABI and13-module dispatch remain unchanged, and the other 10 device units are identical. The public benchmark already uses the named FlashInfer implementation as its clamped-SwiGLU reference.

## Validation snapshot

| Check | Status |
|---|---|
| Public CPU tests |112 passed;GPU tests excluded from this count|
| Independent source primitive and two rounding-block GPU tests |Passed|
| Focused source/reference/FlashInfer eager and graph checks |Passed|
| Source-only32-shape measurements |Separate source-only qualification passed; no public credit|
| Default source registered benchmark |Fresh uninterrupted run passed; distinct from exported-public results|
| Fresh public/source bitwise and API/graph checks |Passed: 32 strict/source-bitwise rows, 7 API/graph cases and 8 routing cases|
| Fresh public 32-shape comparison |Independent eight-shard union passed; 576 strict timing postchecks|
| Synccheck |Skipped after hard 20 s timeout; no reported errors before termination|
| Racecheck |Skipped after hard 20 s timeout; no reported errors before termination|

A separate source-side physical-test module still cannot collect because three inherited regression manifests are absent. Isolated unchanged functions passed only as diagnostics. Source-generator classification coverage retains98 inherited gaps; targeted passing checks do not close that gate. Prior evidence retains its original measured-revision and binary-association limits. Hardware-limit evidence is still incomplete.

## Measurement scope

Target: NVIDIA B200, T1..32, H4096, I2048,256routed experts, top-k6, logical clamp10, BF16 output. The shared expert is outside this routed-expert kernel. All arms consume identical matched NVFP4 physical inputs, routes, scales and equivalent clamp parameters. GPU time uses CUPTI with cold L2 and the same complete-call boundary, retaining all six captures per arm and every shape, including ties or regressions. Per-shape qualification requires speedup strictly greater than1.0 against `flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe`.

The source-only measurements and exported public measurements are separate cohorts. The public table below contains the repaired source, exported public kernel and baseline measured together in this fresh cohort. The source-only results and historical measurements are separate. The fixtures are synthetic matched NVFP4 inputs; full actual-checkpoint pipeline performance is not claimed.

Public geometric-mean speedup **1.022945505290×**, source control **1.023093566763×**; all 32 shapes beat FlashInfer, with no shape-level ties or regressions. Minimum public win: T14, 161.733106057487 vs 162.415976097617 µs, **1.004222203214×**. Maximum: T1, 24.949130388465 vs 28.994138921309 µs, **1.162130241410×**. Public was slower than its source control at 18 shapes: **1, 5, 6, 7, 10, 14, 15, 18, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32**; every comparison is retained.

Capture-level counts across the retained 192 captures (wins/ties/losses): public speedup 192/0/0; source speedup 192/0/0; source over public 77/5/110. Fixed six-capture descriptive variation is not a confidence interval; marginal order positions are balanced but are not independent factorial trials.

Successful r1 public GPU cohort physical turnaround: **3264.050982952 s** (earliest controller submission to latest completion); aggregate wrapper worker duration: **5573.945842369 s**. This span includes scheduler queues and the profiling interleave, but excludes failed r0 identity-guard attempts, preparation/JIT and later CPU reductions. It is separate from measured complete-call GPU latency.

| T | Source µs | Public µs | FlashInfer µs | Public speedup | Source speedup | Source/public |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.916853497 | 24.949130388 | 28.994138921 | 1.162130241410× | 1.163635646255× | 0.998706291914× |
| 2 | 41.088108903 | 40.997406600 | 44.479939659 | 1.084945203805× | 1.082550179274× | 1.002212391237× |
| 3 | 56.085264332 | 55.888036205 | 59.989135346 | 1.073380627051× | 1.069606001876× | 1.003528986532× |
| 4 | 67.439808452 | 67.290632619 | 71.535364670 | 1.063080578754× | 1.060729060647× | 1.002216888548× |
| 5 | 79.279968237 | 79.338631133 | 82.469447278 | 1.039461433859× | 1.040230579194× | 0.999260601110× |
| 6 | 91.642475134 | 91.818624510 | 93.535797501 | 1.018701793892× | 1.020659878123× | 0.998081550698× |
| 7 | 99.690428753 | 99.893127867 | 102.170367233 | 1.022796757039× | 1.024876394959× | 0.997970840259× |
| 8 | 108.778934345 | 108.624144982 | 110.324987087 | 1.015658048266× | 1.014212795436× | 1.001424999602× |
| 9 | 120.501650255 | 120.373469133 | 121.935770862 | 1.012978787935× | 1.011901252840× | 1.001064861905× |
| 10 | 126.901316302 | 127.036992477 | 127.839617428 | 1.006318041186× | 1.007393943208× | 0.998931994747× |
| 11 | 136.031955112 | 135.951962663 | 136.746649409 | 1.005845349562× | 1.005253870648× | 1.000588387602× |
| 12 | 144.693258245 | 144.325738380 | 145.210815248 | 1.006132494991× | 1.003576925483× | 1.002546461006× |
| 13 | 153.695626740 | 153.343651122 | 154.874079472 | 1.009980382877× | 1.007667444789× | 1.002295338705× |
| 14 | 161.637104021 | 161.733106057 | 162.415976098 | 1.004222203214× | 1.004818646568× | 0.999406416913× |
| 15 | 167.680443028 | 167.695933099 | 168.586891897 | 1.005312942194× | 1.005405811513× | 0.999907630016× |
| 16 | 177.077115062 | 177.039141950 | 178.244728266 | 1.006809716219× | 1.006593811988× | 1.000214489924× |
| 17 | 182.746465127 | 182.725145782 | 185.482437967 | 1.015089834373× | 1.014971413198× | 1.000116674395× |
| 18 | 186.944326057 | 187.077465023 | 189.712229168 | 1.014083813592× | 1.014806028988× | 0.999288321733× |
| 19 | 194.853325961 | 194.730307394 | 197.557450103 | 1.014518247039× | 1.013877741777× | 1.000631738162× |
| 20 | 200.207439319 | 200.106293516 | 203.226346551 | 1.015591978544× | 1.015078896378× | 1.000505460382× |
| 21 | 207.477943525 | 208.037619620 | 211.077882481 | 1.014614005229× | 1.017350947744× | 0.997309736113× |
| 22 | 221.839982498 | 222.005133183 | 224.549412359 | 1.011460452014× | 1.012213442458× | 0.999256095195× |
| 23 | 222.511655302 | 222.458333021 | 226.665975254 | 1.018914293638× | 1.018670122904× | 1.000239695589× |
| 24 | 234.155119011 | 234.528475284 | 237.403362303 | 1.012258157635× | 1.013872185697× | 0.998408055685× |
| 25 | 241.856623078 | 242.027612606 | 244.790074026 | 1.011413827499× | 1.012128884092× | 0.999293512315× |
| 26 | 247.300799866 | 247.626439363 | 251.106107685 | 1.014052087213× | 1.015387365596× | 0.998684956670× |
| 27 | 255.770801786 | 256.554796993 | 259.143951703 | 1.010092014418× | 1.013188174308× | 0.996944141308× |
| 28 | 260.454281395 | 260.811656416 | 264.507419915 | 1.014170238975× | 1.015561804164× | 0.998629758246× |
| 29 | 268.410287202 | 268.394457534 | 271.898001586 | 1.013053712377× | 1.012993966886× | 1.000058979118× |
| 30 | 281.397122493 | 281.573325200 | 284.021081691 | 1.008693140548× | 1.009324754904× | 0.999374220881× |
| 31 | 281.472322457 | 281.802797313 | 284.672669119 | 1.010183972031× | 1.011370022581× | 0.998827283266× |
| 32 | 286.010163499 | 286.158981661 | 289.807710400 | 1.012750704933× | 1.013277664173× | 0.999479945864× |
