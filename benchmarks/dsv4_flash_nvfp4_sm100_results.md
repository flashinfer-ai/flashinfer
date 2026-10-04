# B200 routed NVFP4 decode: twelve-warp FC2 full32 results

The current public candidate beats the named FlashInfer baseline at every integer T=1..32 and all 192 capture comparisons, with **1.024376430585× geometric-mean speedup**. The minimum is **1.006363903484× at T14**. The equivalent source implementation measures **1.024490735084×**, minimum **1.006266404635× at T14**. Hardware-limit evidence and broader promotion remain incomplete.

The target is NVIDIA B200 (sm_100a), H=4096, I=2048, 256 routed experts, top-k=6 and clamped SwiGLU limit 10.0. The kernel includes FC1 gate/up, exact clamped activation, FC2 and routing-weighted finalization; the shared expert is outside its scope. Checkpoint FP4 conversion, NVFP4 weight/activation representations, block scales and BF16 output retain the established ABI and rounding contract.

This change compacts an unused physical FC2 warp, reducing the block from 416 to 384 threads while preserving useful roles, logical PDL dependencies, barriers, work-ring arrivals, MMA, scaling and quantization. It applies only to T12,14,15,17,18,19,22,24,26,27,28,31,32; the other 19 routes and 12 generated kernels are unchanged.

## Current synthetic-bank cohort

The public/source/baseline arms consume identical cached physical NVFP4 weights, activations, routes and scales with equivalent clamped-SwiGLU parameters. Timing uses `loom.bench.bench_gpu_time`, CUPTI and cold L2 across complete-call CUDA graph replay, including planner, projections, activation/quantization and finalization. Each reported time is the equal-weight geometric mean of six balanced capture medians; speedup is FlashInfer/candidate. There is no best-capture selection. The six fixed captures describe variation, not independent randomized trials or a confidence interval. All times are microseconds.

| T | Source µs | Public µs | FlashInfer µs | Source speedup | Public speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 24.901284 | 24.970491 | 29.245450 | 1.174455491× | 1.171200407× |
| 2 | 41.103885 | 41.034588 | 44.469101 | 1.081870990× | 1.083697988× |
| 3 | 56.399634 | 56.260895 | 60.378235 | 1.070543039× | 1.073182986× |
| 4 | 68.137968 | 68.009617 | 72.393664 | 1.062457047× | 1.064462162× |
| 5 | 80.026313 | 79.999626 | 83.151366 | 1.039050313× | 1.039396934× |
| 6 | 91.796810 | 91.775102 | 93.951008 | 1.023467023× | 1.023709112× |
| 7 | 99.754599 | 99.936131 | 102.234704 | 1.024862066× | 1.023000423× |
| 8 | 108.778759 | 108.640304 | 110.308810 | 1.014065720× | 1.015358079× |
| 9 | 120.496315 | 120.384135 | 121.927762 | 1.011879584× | 1.012822505× |
| 10 | 126.944152 | 127.034818 | 127.872444 | 1.007312602× | 1.006593670× |
| 11 | 136.031137 | 135.924301 | 137.087208 | 1.007763457× | 1.008555551× |
| 12 | 145.972984 | 145.844641 | 147.023640 | 1.007197607× | 1.008083938× |
| 13 | 155.161467 | 155.065479 | 156.852085 | 1.010895868× | 1.011521628× |
| 14 | 161.732984 | 161.717315 | 162.746468 | 1.006266405× | 1.006363903× |
| 15 | 167.263606 | 167.259104 | 168.576064 | 1.007846648× | 1.007873774× |
| 16 | 176.890612 | 176.965643 | 178.234876 | 1.007599409× | 1.007172202× |
| 17 | 182.571122 | 182.496311 | 185.589630 | 1.016533326× | 1.016950036× |
| 18 | 186.602824 | 186.634619 | 189.930611 | 1.017833529× | 1.017660136× |
| 19 | 195.076813 | 194.697991 | 197.978245 | 1.014873280× | 1.016847910× |
| 20 | 202.712815 | 202.504648 | 205.667454 | 1.014575494× | 1.015618434× |
| 21 | 210.297825 | 210.388488 | 213.415371 | 1.014824435× | 1.014387114× |
| 22 | 221.603801 | 221.785487 | 224.937070 | 1.015041566× | 1.014210050× |
| 23 | 222.187161 | 221.984998 | 226.757731 | 1.020570811× | 1.021500247× |
| 24 | 233.899291 | 234.229649 | 237.901771 | 1.017111977× | 1.015677440× |
| 25 | 241.584801 | 241.728612 | 245.216508 | 1.015032847× | 1.014428975× |
| 26 | 246.885976 | 247.547083 | 251.232444 | 1.017605163× | 1.014887517× |
| 27 | 255.971827 | 256.681291 | 259.374713 | 1.013293985× | 1.010493254× |
| 28 | 263.630980 | 264.308641 | 267.519079 | 1.014748265× | 1.012146551× |
| 29 | 271.912483 | 271.928623 | 275.347218 | 1.012631767× | 1.012571661× |
| 30 | 281.597076 | 281.716820 | 284.469124 | 1.010199138× | 1.009769754× |
| 31 | 281.200140 | 281.546588 | 284.741192 | 1.012592640× | 1.011346627× |
| 32 | 286.016492 | 286.072732 | 289.989887 | 1.013892186× | 1.013692863× |


All 32 strict BF16 correctness shapes pass at atol=rtol=0.01 with stricter checks preserved, along with 576 timing postchecks,7 API/graph cases and8 dynamic-route cases. Both arms win 32/32 shape and 192/192 capture comparisons against FlashInfer. Source/public differences remain mixed: public is faster in 87 captures,8 tie and source is faster in 97. The source column is the same candidate's source implementation, not the previously selected schedule. Shape-level differences are visible above. Current and older full32 cohorts do not establish a controlled causal improvement attributable to this change.

Successful cohort physical turnaround was 3093.474566s. Successful wrapper-worker durations sum to 5561.045760s and validation-worker durations to 5514.159343s; parallel worker sums are not elapsed time. Prior failed orchestration attempts remain separate.

## Baseline, inputs and validation limits

The baseline is `flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe`, using the stock API's observed default-tactic fallback. A T1 pre-capture warning reflects a non-tuning cache miss: tactic −1 maps to backend default-selection sentinels [-1, -1]. This does not identify concrete FC1/FC2 tactics or establish an explicitly tuned optimum. The same prepared baseline closure is captured; warning logging lies outside measured timing.

The current cohort uses a fixed synthetic physical bank with supplied activations/routes. It is not a new actual-checkpoint qualification or native full-model/router/shared-expert execution. The earlier checkpoint-weight cohort used all 256 stored layer-0 experts and supplied activations/routes, reaching public 1.023640533547× with 32 shape / 192 capture wins. Its T1–4 exact per-capture baseline-binary association remains incomplete. Natural timing inputs activated no clamps; its separate stress case exercised793 up and379 gate clamps. Earlier primary and singleton cohorts below remain independent evidence, including their source-control regressions.

Affected source CPU tests 111 and public CPU contracts 112 passed. The separate current-source literal GPU e2e command passed one test with zero failures/errors/skips, all 32 strict rows and 8 route cases, with all 13 loaded binary/ABI identities matched. Its physical turnaround was 521.298456s and worker duration 405.829490s. The separate literal registered benchmark is now independently qualified below. Its evidence is distinct from the synthetic export cohort.

Separate synccheck and racecheck checks were each **SKIPPED: sanitizer timeout after 20 seconds**, with no reported errors. Timeout is not pass. Hardware-limit evidence and broader promotion remain incomplete; serialized Nsight Compute measurements do not prove a normal-PDL hardware ceiling.

## Separate current-source registered cohort

The literal registered benchmark independently passes all 32 shapes and all 192 capture comparisons against the same named FlashInfer baseline: **1.025457383583× geometric-mean speedup**. Minimum T16: **1.006571425137×**, with source 179.380997µs and FlashInfer 180.559785µs. This is a fresh source-only synthetic seed5184 cohort, not a new public-export or checkpoint-weight measurement. Do not pool its timings with the public/source/baseline cohort above or infer a causal gain across cohorts.

The original strict BF16 atol=rtol=0.01, all 32 eager/graph checks, eight route cases and 384 timing postchecks pass. Independent reduction verifies all 384 raw capture medians (3,271,495 samples), CUPTI backend without fallback, cold L2, equal six-capture geometric-mean aggregation, 13 loaded binary/ABI identities and the actual baseline library. Timing arms share identical physical tensor objects; recorded small-input hashes and weight metadata accompany the deterministic source/seed recipe. This audit does not claim whole-weight hashing. Clamp stress records 46,622 up and 23,223 gate clamps.

Physical turnaround: 3478.628871s; wrapper worker: 3336.854336s; evaluator: 3306.177978s. All times below are microseconds.

| T | Registered source µs | FlashInfer µs | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 24.922609 | 29.311853 | 1.176114930× |
| 2 | 41.290565 | 44.815369 | 1.085365855× |
| 3 | 56.703482 | 60.682246 | 1.070167892× |
| 4 | 67.834476 | 72.527902 | 1.069189387× |
| 5 | 79.434639 | 83.338183 | 1.049141574× |
| 6 | 91.130469 | 94.138635 | 1.033009438× |
| 7 | 100.506293 | 103.647837 | 1.031257182× |
| 8 | 110.010490 | 111.834345 | 1.016578923× |
| 9 | 121.866321 | 123.492968 | 1.013347796× |
| 10 | 128.693116 | 129.658440 | 1.007500982× |
| 11 | 135.754636 | 137.402409 | 1.012137879× |
| 12 | 146.074134 | 147.124685 | 1.007191902× |
| 13 | 155.205153 | 157.125284 | 1.012371567× |
| 14 | 161.636818 | 163.098483 | 1.009042894× |
| 15 | 169.375659 | 170.852511 | 1.008719390× |
| 16 | 179.380997 | 180.559785 | 1.006571425× |
| 17 | 184.938304 | 188.314020 | 1.018253198× |
| 18 | 188.933155 | 192.372904 | 1.018206171× |
| 19 | 194.938484 | 198.159486 | 1.016523174× |
| 20 | 203.199812 | 205.863729 | 1.013109839× |
| 21 | 210.442302 | 213.503803 | 1.014547934× |
| 22 | 222.346151 | 225.145804 | 1.012591418× |
| 23 | 224.748571 | 229.018145 | 1.018997114× |
| 24 | 237.786156 | 240.511339 | 1.011460646× |
| 25 | 245.316807 | 248.362053 | 1.012413525× |
| 26 | 250.639307 | 254.084760 | 1.013746658× |
| 27 | 256.687662 | 259.924910 | 1.012611622× |
| 28 | 264.026482 | 267.508935 | 1.013189785× |
| 29 | 272.260995 | 275.237065 | 1.010930947× |
| 30 | 281.647657 | 284.655046 | 1.010677841× |
| 31 | 285.231652 | 288.633999 | 1.011928366× |
| 32 | 289.972990 | 293.983333 | 1.013830060× |

## Earlier cohorts — separate historical evidence

The following prior report is retained verbatim. Its headings, present-tense statements, source identities, timings and limitations apply to those earlier cohorts, not the current twelve-warp candidate.

# B200 routed NVFP4 decode: singleton FC2 full32 results

Every T=1..32 passed strict BF16 checks (atol=rtol=0.01). Public and source control singleton implementations beat the named FlashInfer baseline at all 32 shapes and all 192 captures. Public geometric-mean speedup: **1.022889482173x**; source control: **1.023026241466x**. The minimum public speedup is **1.005775565624x at T14**.

NVIDIA B200/sm_100a; H=4096, I=2048, 256 routed experts, top-k=6, clamped SwiGLU limit=10.0. Source control and public arms are the same singleton candidate in their respective source/artifact paths. All arms consume the same cached synthetic physical NVFP4 bank, routes, scales and equivalent activation parameters. Baseline: flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe. This is routed-kernel qualification, not actual-weight or full-model/router/shared-expert execution.

Each latency is the equal-weight geometric mean of six balanced capture medians. Every raw GPU capture uses complete-call CUPTI timing with cold L2, including planner, FC1, FC2 and finalization. Independently verified: 32 strict/eager/graph checks, 7 API/graph cases, 8 routing cases, 192 capture triplets and 576 strict timing postchecks. Six fixed captures describe variation; they are not independent randomized trials or a confidence interval.

FI/source control and FI/public are speedups (larger is faster). Public/source control is a latency ratio; >1 marks a public regression against source control. The source control column is the new candidate's source implementation, not the previously selected kernel.

| T | Source control us | Public us | FlashInfer us | FI/source control | FI/public | Public/source control | Public regression |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 25.044561 | 25.188729 | 28.804539 | 1.150131534 | 1.143548735 | 1.005756465 | yes |
| 2 | 41.401838 | 41.290277 | 44.842436 | 1.083102532 | 1.086028931 | 0.997305413 | no |
| 3 | 56.218293 | 56.069013 | 60.095528 | 1.068967501 | 1.071813553 | 0.997344638 | no |
| 4 | 67.957299 | 67.807950 | 72.501304 | 1.066865587 | 1.069215393 | 0.997802308 | no |
| 5 | 79.993980 | 79.951276 | 83.166778 | 1.039662957 | 1.040218262 | 0.999466165 | no |
| 6 | 91.610808 | 91.797310 | 93.546545 | 1.021129999 | 1.019055404 | 1.002035802 | yes |
| 7 | 100.820633 | 100.943293 | 103.471130 | 1.026289228 | 1.025042151 | 1.001216611 | yes |
| 8 | 109.972977 | 109.962252 | 111.530024 | 1.014158448 | 1.014257367 | 0.999902471 | no |
| 9 | 121.988789 | 121.775413 | 123.145884 | 1.009485253 | 1.011254084 | 0.998250854 | no |
| 10 | 128.516770 | 128.580440 | 129.396852 | 1.006847998 | 1.006349429 | 1.000495423 | yes |
| 11 | 135.935966 | 135.850626 | 136.794560 | 1.006316165 | 1.006948326 | 0.999372201 | no |
| 12 | 145.940651 | 145.822973 | 147.012139 | 1.007341948 | 1.008154864 | 0.999193660 | no |
| 13 | 155.258301 | 154.986481 | 156.746786 | 1.009587151 | 1.011357796 | 0.998249240 | no |
| 14 | 161.652757 | 161.604939 | 162.538299 | 1.005478055 | 1.005775566 | 0.999704198 | no |
| 15 | 169.257827 | 169.251984 | 170.542541 | 1.007590281 | 1.007625060 | 0.999965484 | no |
| 16 | 179.096940 | 179.107630 | 180.200887 | 1.006163966 | 1.006103911 | 1.000059691 | yes |
| 17 | 184.904618 | 184.739811 | 187.470448 | 1.013876507 | 1.014780991 | 0.999108691 | no |
| 18 | 189.150976 | 189.182981 | 192.116111 | 1.015676024 | 1.015504194 | 1.000169206 | yes |
| 19 | 194.847493 | 194.553800 | 197.642118 | 1.014342626 | 1.015873852 | 0.998492701 | no |
| 20 | 202.735648 | 202.575892 | 205.639708 | 1.014324368 | 1.015124288 | 0.999211998 | no |
| 21 | 210.254990 | 210.364321 | 213.417288 | 1.015040298 | 1.014512761 | 1.000519991 | yes |
| 22 | 221.312138 | 221.552305 | 224.970574 | 1.016530659 | 1.015428720 | 1.001085196 | yes |
| 23 | 224.964393 | 224.751215 | 228.505460 | 1.015740569 | 1.016704007 | 0.999052391 | no |
| 24 | 237.188308 | 237.471322 | 240.676411 | 1.014706050 | 1.013496742 | 1.001193204 | yes |
| 25 | 245.156811 | 245.023331 | 248.154089 | 1.012225960 | 1.012777388 | 0.999455529 | no |
| 26 | 250.313659 | 250.878976 | 253.281485 | 1.011856431 | 1.009576370 | 1.002258434 | yes |
| 27 | 255.643139 | 256.336126 | 259.595082 | 1.015458828 | 1.012713603 | 1.002710761 | yes |
| 28 | 263.646652 | 264.259979 | 267.513252 | 1.014665842 | 1.012310880 | 1.002326323 | yes |
| 29 | 272.021318 | 271.989299 | 275.311628 | 1.012095778 | 1.012214925 | 0.999882291 | no |
| 30 | 281.311117 | 281.481991 | 284.196900 | 1.010258332 | 1.009645053 | 1.000607421 | yes |
| 31 | 285.006727 | 285.582479 | 288.283900 | 1.011498579 | 1.009459338 | 1.002020132 | yes |
| 32 | 289.639993 | 289.672154 | 293.170821 | 1.012190405 | 1.012078025 | 1.000111038 | yes |

Public-versus-source control captures: 93 wins, 4 ties and 95 losses. Aggregate regressions: 1, 6, 7, 10, 16, 18, 21, 22, 24, 26, 27, 28, 30, 31, 32. All are retained.

Successful measured-shard physical turnaround: **2763.677077s**. Wrapper worker sum: **5557.437188s**; validation worker sum: **5508.673213s**. These include successful-shard scheduling gaps and are distinct from GPU latency. Preparation/JIT/reduction and the preserved earlier argument-only failure are outside that measured-cohort span.

The canonical registered benchmark is a separate cohort and is not substituted here. The earlier controlled 13-shape comparison against the previously selected source control kernel was 1.000541549x, with regressions at T19/T26/T32; cross-cohort geometric means do not establish another causal gain. Hardware-limit evidence and broader promotion remain incomplete. No singleton publication is implied by this report.

The selected OOB FC2 device uses independent CTA-local MMA and work scheduling
for T=12,14,15,17,18,19,22,24,26,27,28,31,32. Only its generated device and
associated cluster metadata change. The remaining12 device units, selector,
route metadata and public ABI are identical to the prior qualification.

112 CPU tests and normal JIT compilation passed. The physical-reference caller
requires nonzero gate and up clamp counts before each successful shard; exact
counts were not serialized. Both clamps were therefore exercised with this
artifact. Separate synccheck and racecheck reached their mandatory20-second
deadlines and remain SKIPPED. Canonical source GPU e2e and the separate full32
registered benchmark passed, the latter at1.023009825330x versus FlashInfer.
Its physical turnaround was3556.888271s and benchmark runtime3426.5s. The saved
review preserves outer receipt failures caused by an unused analysis-library
observer; actual GPU commands and raw timing checks passed.

## Preserved earlier paired-CTA evidence

The following complete report belongs to the prior paired-CTA FC2 artifact,
including its synthetic1.024004388551x and supplemental actual-weight
1.023640533547x cohorts. These historical results are retained verbatim and do
not qualify the singleton artifact on actual weights. The actual-weight scope
and T1–4 baseline-binary association limitation below remain unchanged.

# Packed NVFP4 FC1 scheduling — measured results

Fresh public qualification passed all 32 shapes. Hardware-limit evidence remains incomplete.

The packed FC1 schedule uses four-column stripes and removes redundant startup writes to activation scale-factor shared memory after complete producer coverage was verified. Clamp placement, projection arithmetic, routing, scale handling, quantization, rounding, synchronization and the public ABI retain their existing semantics. One generated packed-FC1 device unit changes; the other 12 device units and 13-module dispatch remain unchanged. Direct dispatch for T1–9 is unchanged, so timing variation there is not credited to this optimization.

Target: NVIDIA B200 (sm_100a), every integer T=1..32, H=4096, I=2048, E=256 routed experts, top-k=6 and swiglu_limit=10.0. Output is BF16; the shared expert is outside this routed-expert kernel. The named baseline is `flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe`. All three arms use identical matched physical NVFP4 inputs, routes, scales and equivalent clamped-SwiGLU parameters. These are synthetic matched fixtures, not full actual-checkpoint pipeline performance measurements.

## Validation and timing

- 112 CPU tests passed. These test results are reused for the identical executable sources; only this result document is added after qualification.
- All 32 strict BF16 checks passed with atol=rtol=0.01, including source-bitwise eager and graph output comparisons.
- Seven public API/graph cases and eight routing cases passed.
- Six balanced complete-call captures per arm and shape used CUPTI with cold L2; all 576 timing postchecks passed. The timing boundary includes every required per-call pipeline step.
- Synccheck and racecheck for the corresponding source schedule were separately skipped after the mandatory 20-second process-group deadline, with no reported errors. These skips are not passes; no unchanged sanitizer check was repeated.

Public geometric-mean speedup is **1.024004388551×**; matched source control is **1.024027935471×**. Every public shape strictly beats FlashInfer, with no aggregate ties or regressions. Minimum: T15, 169.796310796 versus 170.639060655 µs, **1.004963298995×**. Maximum: T1, **1.173878999853×**.

Public is slower than its matched source control at 18 shapes: 1, 5, 6, 7, 10, 14, 15, 18, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32. Every comparison remains in the table. Capture-level wins/ties/losses are public/FI (192, 0, 0), source/FI (192, 0, 0) and source/public (86, 4, 102). Six balanced captures provide descriptive variation, not a confidence interval or independent randomized trials.

Successful measured-shard cohort physical turnaround: **4301.582590 seconds**, earliest selected shard submission to latest completion including scheduling gaps. Aggregate wrapper worker time: **5508.653243 seconds**. These spans exclude preparation/JIT, prior argument-only failed attempts and later CPU reductions, and are distinct from measured GPU latency.

Source-only paired and default registered measurements are separate cohorts; their results are not substituted for this exported artifact's fresh qualification. Actual source and public runtime-library identities remain distinct. Broader generator validation and hardware-limit evidence remain incomplete; the data does not establish a hardware ceiling.

| T | Source µs | Public µs | FlashInfer µs | Public speedup | Source speedup | Source/public |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.082184060 | 25.082313439 | 29.443601014 | 1.173878999853× | 1.173885054986× | 0.999994841801× |
| 2 | 41.359722292 | 41.253094663 | 44.943595440 | 1.089459974017× | 1.086651286537× | 1.002584718313× |
| 3 | 56.021097712 | 55.876921727 | 60.047497462 | 1.074638609398× | 1.071872917772× | 1.002580242098× |
| 4 | 68.063287879 | 67.897953899 | 72.682319831 | 1.070464066393× | 1.067863779373× | 1.002435036256× |
| 5 | 79.967636709 | 79.989133047 | 83.146179674 | 1.039468444106× | 1.039747866705× | 0.999731259271× |
| 6 | 91.650221812 | 91.829289334 | 93.535519155 | 1.018580453290× | 1.020570570431× | 0.998049995563× |
| 7 | 99.935489806 | 100.122145413 | 102.868614154 | 1.027431181473× | 1.029350177339× | 0.998135721052× |
| 8 | 108.729417954 | 108.585464442 | 110.270406144 | 1.015517193854× | 1.014172688663× | 1.001325716227× |
| 9 | 120.446985838 | 120.324137613 | 121.886551045 | 1.012985037436× | 1.011951857464× | 1.001020977396× |
| 10 | 127.140628542 | 127.199271787 | 128.074296527 | 1.006879164691× | 1.007343584780× | 0.999538965556× |
| 11 | 135.898126919 | 135.807138988 | 136.740600863 | 1.006873437449× | 1.006199305043× | 1.000669979002× |
| 12 | 145.887936818 | 145.711412551 | 146.724972716 | 1.006955942210× | 1.005737526462× | 1.001211464936× |
| 13 | 155.284311717 | 155.028981502 | 156.756606159 | 1.011143881877× | 1.009481282597× | 1.001646983762× |
| 14 | 161.567269831 | 161.615260856 | 162.505815739 | 1.005510339048× | 1.005809010133× | 0.999703053879× |
| 15 | 169.657819769 | 169.796310796 | 170.639060655 | 1.004963298995× | 1.005783646676× | 0.999184369636× |
| 16 | 179.193709413 | 179.183104897 | 180.313460178 | 1.006308380924× | 1.006248828536× | 1.000059182566× |
| 17 | 184.815291061 | 184.804446749 | 187.609564448 | 1.015178843087× | 1.015119275960× | 1.000058679929× |
| 18 | 189.460139775 | 189.497641976 | 192.127022141 | 1.013875529730× | 1.014076218715× | 0.999802096745× |
| 19 | 194.735492233 | 194.505964366 | 197.567285071 | 1.015738955438× | 1.014541739699× | 1.001180055677× |
| 20 | 202.815326521 | 202.553473587 | 205.311086620 | 1.013614247063× | 1.012305579373× | 1.001292759535× |
| 21 | 209.913599656 | 210.223977468 | 213.514099515 | 1.015650555595× | 1.017152294396× | 0.998523585101× |
| 22 | 221.471313177 | 221.620456458 | 224.713554532 | 1.013956735417× | 1.014639554479× | 0.999327032876× |
| 23 | 222.196643048 | 221.945468884 | 226.356199025 | 1.019873035314× | 1.018720156705× | 1.001131693136× |
| 24 | 233.957376082 | 234.269802755 | 237.336208575 | 1.013089206481× | 1.014442085775× | 0.998666380947× |
| 25 | 241.565469779 | 241.682620281 | 244.695894420 | 1.012467897509× | 1.012958907759× | 0.999515271305× |
| 26 | 247.395988000 | 247.790990529 | 250.894834953 | 1.012526058421× | 1.014142698843× | 0.998405904392× |
| 27 | 255.716483691 | 256.292460329 | 259.300389634 | 1.011736316009× | 1.014015154172× | 0.997752658672× |
| 28 | 263.535448177 | 264.111823853 | 267.663078381 | 1.013446026291× | 1.015662523704× | 0.997817683177× |
| 29 | 272.036155012 | 272.035965994 | 274.964220506 | 1.010764218258× | 1.010763515953× | 1.000000694826× |
| 30 | 281.241690801 | 281.407324254 | 284.073751110 | 1.009475328559× | 1.010069845268× | 0.999411410298× |
| 31 | 285.321145833 | 285.737484629 | 288.451993332 | 1.009500009097× | 1.010973065069× | 0.998542932524× |
| 32 | 289.620324691 | 289.622567532 | 293.279259183 | 1.012625713809× | 1.012633555658× | 0.999992255985× |

## Supplemental actual-weight routed-kernel cohort

This supplemental result is separate from the primary synthetic-fixture qualification above (**1.024004388551×**). No executable source changed for this report. It follows issue [#5184](https://github.com/flashinfer-ai/flashinfer/issues/5184) with the same B200 geometry, clamped SwiGLU and named FlashInfer baseline.

The cohort uses all 256 stored layer-0 routed-expert weights with supplied deterministic activations and routes, matched NVFP4 weight/activation representations, scales and BF16 output. T32 touches 135 experts. It does not execute the native model, router, tokenizer or shared expert. All arms consume identical physical inputs for each comparison. Actual checkpoint packing/scales are converted to the matched MajorK/R128c4 execution ABI; checkpoint FP4 is not assumed interchangeable solely from its dtype label.

All 32 shapes passed strict BF16 comparisons at atol=rtol=0.01, source-control/public eager and graph equivalence, and launch checks. The cohort retains 576 complete-call timing postchecks and 4,926,912 CUPTI samples. Timing uses six balanced complete-call captures per arm and shape with cold L2, including required per-call pipeline work. Each table latency is the geometric mean of the six capture medians. These captures describe measured variation; they are not independent randomized trials or confidence intervals.

Natural timing inputs triggered zero up clamps and zero gate clamps. A separate T32 correctness stress with factor 2 triggered **793 up clamps and 379 gate clamps**, with strict and bitwise checks and launch attributes passing. Stress results are not substituted into natural timing.

Public geometric-mean speedup versus FlashInfer is **1.023640533547×**; source control is **1.024597830145×**. Both win all 32 aggregate shapes and 192 captures. The public minimum is T15: 169.930311429 versus 170.644928547 µs (**1.004205354017×**); maximum is T1 (**1.154595712788×**).

Public versus source control has 48 capture wins, 2 ties and 142 losses. Public/source-control aggregate speedup is 0.999065685512×; public is slower at 26 shapes: 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 30, 31. A ratio below 1 in the Source/Public column indicates a public regression. Every result is retained below.

**Provenance limit for T1–4:** their captures survived an interrupted attempt. The later completion's loaded-baseline-library receipt does not retrospectively establish the exact baseline binary separately for each earlier capture. Numerical/timing rows remain recorded, but uniform per-capture binary association is not claimed. Other reviewed cohort differences were limited to a read-only observer hook and stress-cache path, with other recorded identity pins and timing protocol matching.

Matrix physical turnaround was **13964.198328 s**, from first submission to last completion including interruptions and scheduling. Successful matrix-worker runtime summed to **3571.191417 s**. Setup submission through matrix completion was 15066.549052 s; summed matrix-attempt physical duration was 9792.439456 s. These wall-clock spans are distinct from GPU latency. Partial worker checkpoints are not counted as final worker runtimes.

This supplemental cohort does not replace the primary qualification and does not establish a hardware limit. Existing sanitizer timeouts remain skipped; this report adds no sanitizer pass or public-CI numerical pass.

### All 32 shapes

| T | Source control µs | Public µs | FlashInfer µs | FI/Source | FI/Public | Source/Public | Public vs source W/T/L |
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|25.123783559|25.257681401|29.162410661|1.160749159945|1.154595712788|0.994698727883|3/0/3|
|2|41.498574435|41.605416763|45.018169046|1.084812422089|1.082026633754|0.997432009186|0/0/6|
|3|56.762546130|56.788956344|60.841797914|1.071865200951|1.071366720419|0.999534941025|3/0/3|
|4|68.090446281|67.957104818|72.783658411|1.068926147299|1.071023531762|1.001962141603|5/0/1|
|5|79.695648061|79.727654354|83.535591288|1.048182596165|1.047761808176|0.999598554688|2/0/4|
|6|91.402131804|91.508968953|93.759752787|1.025793938675|1.024596319458|0.998832495326|0/0/6|
|7|100.479627900|100.596818793|103.242465828|1.027496498404|1.026299509940|0.998835043754|0/0/6|
|8|109.748955289|109.994481879|111.596639897|1.016835555318|1.014565803576|0.997767828112|0/0/6|
|9|121.685301537|121.711280612|123.156842904|1.012093008344|1.011876978738|0.999786551627|2/1/3|
|10|128.591624820|128.805140972|129.636499079|1.008125523420|1.006454386062|0.998342332062|0/0/6|
|11|135.834474199|136.042481585|137.290447102|1.010718728895|1.009173351609|0.998471011527|0/0/6|
|12|145.930148235|146.207488217|147.151579883|1.008369974698|1.006457204607|0.998103106857|0/0/6|
|13|155.258318663|155.306320522|157.065751183|1.011641453647|1.011328776926|0.999690921404|2/0/4|
|14|161.764828624|161.967659620|162.884648360|1.006922516754|1.005661554547|0.998747706814|0/0/6|
|15|169.583298368|169.930311429|170.644928547|1.006260228388|1.004205354017|0.997957909581|0/0/6|
|16|179.796825018|179.647652353|180.554600837|1.004214622917|1.005048485030|1.000830362449|6/0/0|
|17|185.002146595|185.263325243|188.121759603|1.016862577355|1.015429035161|0.998590230159|0/0/6|
|18|189.343820744|189.503634354|192.089787587|1.014502542687|1.013646984882|0.999156672587|0/1/5|
|19|194.788650487|194.970463887|197.737947480|1.015141010453|1.014194373535|0.999067482342|1/0/5|
|20|202.196614810|202.580716756|205.348864408|1.015590021629|1.013664418294|0.998103956031|0/0/6|
|21|207.605123370|207.962639248|211.007998309|1.016391093259|1.014643779631|0.998280864876|0/0/6|
|22|219.093140539|219.130646870|222.693361156|1.016432374871|1.016258402631|0.999828840320|3/0/3|
|23|221.818624019|222.149307680|226.436908846|1.020820095010|1.019300538052|0.998511435105|0/0/6|
|24|234.495812956|234.666743700|237.834603672|1.014238167725|1.013499398859|0.999271602184|0/0/6|
|25|242.405491900|242.122442587|245.359607744|1.012186670449|1.013369950852|1.001169033774|6/0/0|
|26|247.642475592|247.833986513|251.013258947|1.013611491110|1.012828234251|0.999227261267|1/0/5|
|27|253.679640032|253.599808539|256.863747624|1.012551687599|1.012870431976|1.000314793192|4/0/2|
|28|261.167637939|261.562491941|264.244522126|1.011781261307|1.010253879160|0.998490402812|0/0/6|
|29|268.879829059|268.874489452|272.138210814|1.012118356986|1.012138456749|1.000019859103|3/0/3|
|30|278.474331092|278.490330058|281.295803115|1.010131892630|1.010073861654|0.999942551090|2/0/4|
|31|281.951805627|282.159822078|285.354260658|1.012067505733|1.011321380047|0.999262770831|0/0/6|
|32|286.501327724|286.415829339|290.495289823|1.013940466280|1.014243139050|1.000298511382|5/0/1|

### Descriptive capture ranges

Min–max over the six capture medians or matched ratios at each shape. Range width is descriptive only.

| T | Source control µs range | Public µs range | FlashInfer µs range | FI/Source range | FI/Public range | Source/Public range |
|---:|---:|---:|---:|---:|---:|---:|
|1|24.832000000–25.440000000|25.056000000–25.504000000|28.928000000–29.344000000|1.137106918–1.175257732|1.142857143–1.168199371|0.974905897–1.011494253|
|2|41.344000000–41.632000000|41.473000000–41.728000000|44.832000000–45.120000000|1.076863951–1.090533088|1.076036866–1.087141032|0.993865031–0.999231951|
|3|56.672000000–56.831000000|56.672000000–56.863000000|60.671000000–61.183000000|1.068151408–1.077165493|1.066967976–1.079598391|0.996641050–1.002258611|
|4|68.000000000–68.224000000|67.840000000–68.096000000|72.576000000–73.056000000|1.066306216–1.074352941|1.068235294–1.073113208|0.998590226–1.005660377|
|5|79.616000000–79.775000000|79.648000000–79.776000000|83.328000000–83.936000000|1.044537762–1.053836882|1.045728039–1.052146009|0.998395507–1.001594516|
|6|91.263000000–91.519000000|91.424000000–91.584000000|93.535000000–93.888000000|1.022028213–1.028412391|1.021302848–1.026251313|0.996495021–0.999989062|
|7|100.416000000–100.671000000|100.544000000–100.703000000|103.136000000–103.392000000|1.025439302–1.029318944|1.025113452–1.028325907|0.998101524–0.999682234|
|8|109.631000000–109.952000000|109.888000000–110.080000000|111.360000000–111.775000000|1.013378565–1.018811078|1.011627907–1.016438992|0.996802326–0.999127653|
|9|121.568000000–121.792000000|121.567000000–121.855000000|122.816000000–123.648000000|1.009713282–1.016039968|1.009464551–1.014714210|0.998695170–1.000788851|
|10|128.480000000–128.735000000|128.672000000–128.928000000|129.311000000–130.015000000|1.006467933–1.011947385|1.003468773–1.009684083|0.997020114–0.999743725|
|11|135.711000000–135.968000000|135.936000000–136.128000000|137.152000000–137.440000000|1.009658422–1.012012432|1.007522332–1.010821269|0.997647595–0.999529301|
|12|145.823000000–146.048000000|146.111000000–146.304000000|146.848000000–147.328000000|1.006580390–1.010320731|1.003718285–1.007880911|0.997156605–0.999568821|
|13|155.136000000–155.359000000|155.200000000–155.392000000|156.767000000–157.247000000|1.010513356–1.012987097|1.008848589–1.012570876|0.998352554–1.001024485|
|14|161.727000000–161.824000000|161.920000000–162.047000000|162.751000000–162.976000000|1.006325435–1.007518843|1.004939766–1.006317935|0.998420211–0.999209642|
|15|169.471000000–169.759000000|169.823000000–170.080000000|170.368000000–170.879000000|1.005292941–1.006610956|1.002822201–1.006022749|0.996425212–0.999428928|
|16|179.743000000–179.872000000|179.519000000–179.712000000|180.416000000–180.768000000|1.003024373–1.005702586|1.003917379–1.006055209|1.000183628–1.001966366|
|17|184.896000000–185.152000000|185.183000000–185.343000000|187.935000000–188.351000000|1.015549386–1.018515414|1.014154507–1.016931674|0.998099845–0.999135990|
|18|189.248000000–189.472000000|189.312000000–189.663000000|191.711000000–192.480000000|1.012501056–1.016396214|1.011641701–1.015537916|0.998813291–1.000000000|
|19|194.719000000–194.912000000|194.816000000–195.135000000|197.567000000–197.984000000|1.013621532–1.016105259|1.012627156–1.016261498|0.997868143–1.000323382|
|20|202.015000000–202.367000000|202.431500000–202.751000000|204.960000000–205.664000000|1.014078382–1.018063015|1.010895137–1.015965855|0.996848351–0.999207139|
|21|207.424000000–207.744000000|207.808000000–208.160000000|210.528000000–211.424000000|1.013557233–1.019284172|1.012621210–1.017400678|0.997537704–0.999076497|
|22|218.880000000–219.231000000|218.976000000–219.264000000|222.400000000–223.200000000|1.014455073–1.018694319|1.015191353–1.019289785|0.998685940–1.000725789|
|23|221.568000000–221.984000000|221.952000000–222.304000000|226.240000000–226.847000000|1.019316708–1.023825643|1.018145161–1.020435980|0.996689218–0.999279123|
|24|234.368000000–234.624000000|234.560000000–234.720000000|237.568000000–238.560000000|1.012547736–1.017886401|1.012409655–1.016359918|0.998500341–0.999863630|
|25|242.336000000–242.497000000|241.920000000–242.336000000|245.152000000–245.632000000|1.011609971–1.013199578|1.012144296–1.014141823|1.000400272–1.001855004|
|26|247.456000000–247.776000000|247.711000000–247.936000000|250.816000000–251.296000000|1.012661499–1.014871331|1.011615901–1.014341475|0.998454642–1.000004036|
|27|253.504000000–253.887000000|253.472000000–253.759000000|256.448000000–257.152000000|1.011613229–1.013878375|1.010847629–1.014262274|0.999243189–1.001009973|
|28|261.023000000–261.376000000|261.440000000–261.632000000|263.935000000–264.608000000|1.010532762–1.013734422|1.009049272–1.011745993|0.997924558–0.999388005|
|29|268.800000000–268.960000000|268.736000000–268.960000000|271.776000000–272.480000000|1.010714219–1.013449179|1.010593953–1.013931889|0.999761961–1.000476304|
|30|278.431000000–278.527000000|278.432000000–278.528000000|281.056000000–281.440000000|1.009427830–1.010803356|1.009192233–1.010803356|0.999655331–1.000341196|
|31|281.824000000–282.175000000|282.080000000–282.303000000|285.024000000–285.631000000|1.011342041–1.013048320|1.010092992–1.012473769|0.998752552–0.999886557|
|32|286.432000000–286.592000000|286.368000000–286.496000000|290.143000000–290.975000000|1.012955955–1.015747179|1.013069134–1.016087691|0.999888306–1.000670466|

### All 192 capture comparisons

Capture IDs 0–5 identify the six balanced capture triplets per shape. Latencies are capture medians in µs. FI/Source and FI/Public greater than 1 indicate a win over FlashInfer; Source/Public greater than 1 indicates a public win over source control.

| T | Capture | Source control µs | Public µs | FlashInfer µs | FI/Source | FI/Public | Source/Public |
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|0|25.376000000|25.119000000|29.344000000|1.156368221942|1.168199370994|1.010231299017|
|1|1|24.832000000|25.088000000|29.184000000|1.175257731959|1.163265306122|0.989795918367|
|1|2|25.344000000|25.056000000|29.152000000|1.150252525253|1.163473818646|1.011494252874|
|1|3|25.440000000|25.312000000|28.928000000|1.137106918239|1.142857142857|1.005056890013|
|1|4|24.864000000|25.504000000|29.184000000|1.173745173745|1.144291091593|0.974905897114|
|1|5|24.895000000|25.471000000|29.184000000|1.172283591083|1.145773624907|0.977386046877|
|2|0|41.472000000|41.536000000|44.864000000|1.081790123457|1.080123266564|0.998459167951|
|2|1|41.344000000|41.473000000|45.087000000|1.090533088235|1.087141031514|0.996889542594|
|2|2|41.536000000|41.600000000|45.087000000|1.085492103236|1.083822115385|0.998461538462|
|2|3|41.632000000|41.664000000|44.832000000|1.076863950807|1.076036866359|0.999231950845|
|2|4|41.536000000|41.632000000|45.120000000|1.086286594761|1.083781706380|0.997694081476|
|2|5|41.472000000|41.728000000|45.120000000|1.087962962963|1.081288343558|0.993865030675|
|3|0|56.800000000|56.863000000|60.671000000|1.068151408451|1.066967975661|0.998892073932|
|3|1|56.672000000|56.863000000|60.672000000|1.070581592321|1.066985561789|0.996641049540|
|3|2|56.800000000|56.672000000|61.183000000|1.077165492958|1.079598390740|1.002258610954|
|3|3|56.800500000|56.768000000|60.768000000|1.069849737238|1.070462232244|1.000572505637|
|3|4|56.672000000|56.832000000|60.671000000|1.070563946923|1.067549971847|0.997184684685|
|3|5|56.831000000|56.736000000|61.088000000|1.074906301138|1.076706147772|1.001674421884|
|4|0|68.032000000|68.000000000|72.640000000|1.067732831609|1.068235294118|1.000470588235|
|4|1|68.063000000|67.840000000|72.576000000|1.066306216300|1.069811320755|1.003287146226|
|4|2|68.192000000|67.999000000|72.895000000|1.068967034256|1.072001058839|1.002838277033|
|4|3|68.224000000|67.840000000|72.800000000|1.067073170732|1.073113207547|1.005660377358|
|4|4|68.032000000|67.968000000|72.736000000|1.069143932267|1.070150659134|1.000941619586|
|4|5|68.000000000|68.096000000|73.056000000|1.074352941176|1.072838345865|0.998590225564|
|5|0|79.712000000|79.743000000|83.520000000|1.047771979125|1.047364658967|0.999611251144|
|5|1|79.616000000|79.712000000|83.392000000|1.047427652733|1.046166198314|0.998795664392|
|5|2|79.744000000|79.711000000|83.615000000|1.048542837079|1.048976929157|1.000413995559|
|5|3|79.775000000|79.648000000|83.328000000|1.044537762457|1.046203294496|1.001594515870|
|5|4|79.679000000|79.776000000|83.424000000|1.047001091881|1.045728038508|0.998784095467|
|5|5|79.648000000|79.776000000|83.936000000|1.053836882282|1.052146008825|0.998395507421|
|6|0|91.423000000|91.424000000|93.792000000|1.025912516544|1.025901295065|0.999989061953|
|6|1|91.360000000|91.424000000|93.824000000|1.026970227671|1.026251312566|0.999299964998|
|6|2|91.392000000|91.455000000|93.664000000|1.024859943978|1.024153955497|0.999311136625|
|6|3|91.519000000|91.584000000|93.535000000|1.022028212721|1.021302847659|0.999290269043|
|6|4|91.456000000|91.583000000|93.888000000|1.026592022393|1.025168426455|0.998613279757|
|6|5|91.263000000|91.584000000|93.856000000|1.028412390564|1.024807826695|0.996495020964|
|7|0|100.671000000|100.703000000|103.232000000|1.025439302282|1.025113452429|0.999682233896|
|7|1|100.416000000|100.607000000|103.231000000|1.028033381134|1.026081684177|0.998101523751|
|7|2|100.416000000|100.544000000|103.168000000|1.027405991077|1.026098026735|0.998726925525|
|7|3|100.447000000|100.544000000|103.392000000|1.029318944319|1.028325907066|0.999035248250|
|7|4|100.448000000|100.575000000|103.136000000|1.026760114686|1.025463584390|0.998737260751|
|7|5|100.480000000|100.608000000|103.296000000|1.028025477707|1.026717557252|0.998727735369|
|8|0|109.631000000|109.888000000|111.647000000|1.018388959327|1.016007207338|0.997661255096|
|8|1|109.952000000|110.048000000|111.423000000|1.013378565192|1.012494547834|0.999127653388|
|8|2|109.728000000|110.016000000|111.616000000|1.017206182561|1.014543339151|0.997382198953|
|8|3|109.759000000|109.983000000|111.775000000|1.018367514281|1.016293427166|0.997963321604|
|8|4|109.728000000|110.080000000|111.360000000|1.014873140857|1.011627906977|0.996802325581|
|8|5|109.696000000|109.952000000|111.759500000|1.018811077888|1.016438991560|0.997671711292|
|9|0|121.760000000|121.823000000|122.976000000|1.009986859396|1.009464551029|0.999482856275|
|9|1|121.728000000|121.759000000|123.392000000|1.013669821241|1.013411739584|0.999745398697|
|9|2|121.568000000|121.567000000|123.136000000|1.012898131087|1.012906463103|1.000008225917|
|9|3|121.792000000|121.696000000|122.975000000|1.009713281661|1.010509794899|1.000788850907|
|9|4|121.696000000|121.855000000|123.648000000|1.016039968446|1.014714209511|0.998695170490|
|9|5|121.568000000|121.568000000|122.816000000|1.010265859437|1.010265859437|1.000000000000|
|10|0|128.512000000|128.672000000|129.631000000|1.008707358068|1.007453058941|0.998756528227|
|10|1|128.672000000|128.831000000|129.536000000|1.006714747575|1.005472285397|0.998765824996|
|10|2|128.735000000|128.768000000|129.664000000|1.007216374723|1.006958250497|0.999743725149|
|10|3|128.671000000|128.928000000|129.663000000|1.007709584910|1.005700856292|0.998006639365|
|10|4|128.480000000|128.864000000|129.311000000|1.006467932752|1.003468773280|0.997020114229|
|10|5|128.480000000|128.768000000|130.015000000|1.011947384807|1.009684083002|0.997763419483|
|11|0|135.711000000|136.031000000|137.152000000|1.010618151808|1.008240768648|0.997647595033|
|11|1|135.968000000|136.096000000|137.440000000|1.010826076724|1.009875382083|0.999059487421|
|11|2|135.904000000|135.968000000|137.376000000|1.010831174947|1.010355377736|0.999529301012|
|11|3|135.840000000|136.128000000|137.152000000|1.009658421673|1.007522331923|0.997884344147|
|11|4|135.808000000|136.096000000|137.216000000|1.010367577757|1.008229485069|0.997883846696|
|11|5|135.776000000|135.936000000|137.407000000|1.012012432241|1.010821268832|0.998822975518|
|12|0|145.887000000|146.207000000|147.168000000|1.008780768677|1.006572872708|0.997811322303|
|12|1|145.952000000|146.207000000|147.071000000|1.007666904188|1.005909429781|0.998255897460|
|12|2|145.983000000|146.240000000|147.296000000|1.008994197955|1.007221006565|0.998242614880|
|12|3|146.048000000|146.111000000|147.199000000|1.007880970640|1.007446393495|0.999568820965|
|12|4|145.888000000|146.304000000|146.848000000|1.006580390436|1.003718285214|0.997156605424|
|12|5|145.823000000|146.176000000|147.328000000|1.010320731298|1.007880910683|0.997585102890|
|13|0|155.264000000|155.263000000|157.183000000|1.012359593982|1.012366114271|1.000006440685|
|13|1|155.264000000|155.328000000|157.087000000|1.011741292251|1.011324423156|0.999587968686|
|13|2|155.231000000|155.296000000|157.247000000|1.012987096650|1.012563105296|0.999581444467|
|13|3|155.136000000|155.392000000|156.767000000|1.010513356023|1.008848589374|0.998352553542|
|13|4|155.296000000|155.359000000|156.960000000|1.010715021636|1.010305164168|0.999594487606|
|13|5|155.359000000|155.200000000|157.151000000|1.011534574759|1.012570876289|1.001024484536|
|14|0|161.792000000|162.016000000|162.976000000|1.007318037975|1.005925340707|0.998617420502|
|14|1|161.824000000|161.952000000|162.848000000|1.006327862369|1.005532503458|0.999209642363|
|14|2|161.727000000|161.920000000|162.847000000|1.006925250577|1.005725049407|0.998808053360|
|14|3|161.791000000|162.047000000|162.943000000|1.007120297174|1.005529260029|0.998420211420|
|14|4|161.728000000|161.951000000|162.751000000|1.006325435299|1.004939765732|0.998623040302|
|14|5|161.727000000|161.920000000|162.943000000|1.007518843483|1.006317934783|0.998808053360|
|15|0|169.567000000|169.984000000|170.688000000|1.006610956141|1.004141566265|0.997546827937|
|15|1|169.696000000|169.887000000|170.720000000|1.006034320196|1.004903259225|0.998875723275|
|15|2|169.759000000|169.856000000|170.879000000|1.006597588346|1.006022748681|0.999428928033|
|15|3|169.471000000|169.823000000|170.368000000|1.005292940975|1.003209223721|0.997927253670|
|15|4|169.472000000|170.080000000|170.560000000|1.006419939577|1.002822201317|0.996425211665|
|15|5|169.535000000|169.952000000|170.655000000|1.006606305483|1.004136462060|0.997546366033|
|16|0|179.775000000|179.712000000|180.416000000|1.003565568071|1.003917378917|1.000350560897|
|16|1|179.744000000|179.711000000|180.512000000|1.004272743457|1.004457156212|1.000183628159|
|16|2|179.743000000|179.680000000|180.768000000|1.005702586471|1.006055209261|1.000350623330|
|16|3|179.872000000|179.584000000|180.416000000|1.003024372887|1.004632929437|1.001603706344|
|16|4|179.872000000|179.519000000|180.448000000|1.003202277175|1.005174939700|1.001966365677|
|16|5|179.775000000|179.680000000|180.768000000|1.005523571131|1.006055209261|1.000528717720|
|17|0|185.023000000|185.183000000|188.063000000|1.016430389735|1.015552183516|0.999135989805|
|17|1|185.152000000|185.343000000|188.031000000|1.015549386450|1.014502840679|0.998969478211|
|17|2|184.927000000|185.215000000|188.351000000|1.018515414190|1.016931674000|0.998445050347|
|17|3|184.896000000|185.248000000|188.000000000|1.016787815853|1.014855760926|0.998099844533|
|17|4|185.056000000|185.312000000|187.935000000|1.015557452879|1.014154506994|0.998618546020|
|17|5|184.959000000|185.279000000|188.351000000|1.018339199498|1.016580400369|0.998272874962|
|18|0|189.248000000|189.471000000|192.159000000|1.015381932702|1.014186867647|0.998823038882|
|18|1|189.312000000|189.312000000|191.903000000|1.013686401285|1.013686401285|1.000000000000|
|18|2|189.375000000|189.600000000|192.480000000|1.016396039604|1.015189873418|0.998813291139|
|18|3|189.472000000|189.663000000|191.871000000|1.012661501436|1.011641701333|0.998992950655|
|18|4|189.344000000|189.504000000|191.711000000|1.012501056279|1.011646192165|0.999155690645|
|18|5|189.312000000|189.472000000|192.416000000|1.016396213658|1.015537915893|0.999155548049|
|19|0|194.912000000|195.040000000|197.567000000|1.013621531768|1.012956316653|0.999343724364|
|19|1|194.720000000|194.848000000|197.727000000|1.015442686935|1.014775619970|0.999343077681|
|19|2|194.879000000|194.816000000|197.984000000|1.015932963531|1.016261498029|1.000323382063|
|19|3|194.719000000|195.135000000|197.599000000|1.014790544323|1.012627155559|0.997868142568|
|19|4|194.783000000|194.912000000|197.696000000|1.014955103885|1.014283368905|0.999338162863|
|19|5|194.719000000|195.072000000|197.855000000|1.016105259374|1.014266527231|0.998190411745|
|20|0|202.367000000|202.655000000|205.216000000|1.014078382345|1.012637240631|0.998578865560|
|20|1|202.048000000|202.592000000|205.248000000|1.015837820716|1.013110093192|0.997314800190|
|20|2|202.271000000|202.431500000|205.439000000|1.015662156216|1.014856877512|0.999207139205|
|20|3|202.367000000|202.623000000|205.567000000|1.015812854863|1.014529446312|0.998736569886|
|20|4|202.112000000|202.751000000|204.960000000|1.014091196960|1.010895137385|0.996848350933|
|20|5|202.015000000|202.432000000|205.664000000|1.018063015123|1.015965855201|0.997940049004|
|21|0|207.615000000|207.968000000|211.072000000|1.016651012692|1.014925373134|0.998302623481|
|21|1|207.744000000|208.160000000|210.912000000|1.015249537893|1.013220599539|0.998001537279|
|21|2|207.424000000|207.808000000|211.424000000|1.019284171552|1.017400677549|0.998152140437|
|21|3|207.712000000|207.904000000|210.528000000|1.013557233092|1.012621209789|0.999076496845|
|21|4|207.712000000|208.000000000|211.105000000|1.016335117855|1.014927884615|0.998615384615|
|21|5|207.424000000|207.936000000|211.008000000|1.017278617711|1.014773776547|0.997537703909|
|22|0|218.880000000|219.168000000|222.593000000|1.016963633041|1.015627281355|0.998685939553|
|22|1|219.231000000|219.072000000|222.400000000|1.014455072503|1.015191352615|1.000725788782|
|22|2|219.136000000|219.104000000|222.720000000|1.016355140187|1.016503578209|1.000146049365|
|22|3|219.072000000|219.264000000|222.688000000|1.016505988899|1.015615878576|0.999124343257|
|22|4|219.136000000|219.200000000|222.560000000|1.015625000000|1.015328467153|0.999708029197|
|22|5|219.104000000|218.976000000|223.200000000|1.018694318680|1.019289785182|1.000584538945|
|23|0|221.984000000|222.145000000|226.272000000|1.019316707510|1.018577955840|0.999275248149|
|23|1|221.856000000|222.112000000|226.496000000|1.020914467042|1.019737789944|0.998847428324|
|23|2|221.952000000|222.208000000|226.240000000|1.019319492503|1.018145161290|0.998847926267|
|23|3|221.760000000|222.175000000|226.336000000|1.020634920635|1.018728479802|0.998132103072|
|23|4|221.792000000|221.952000000|226.431000000|1.020915993363|1.020180038927|0.999279123414|
|23|5|221.568000000|222.304000000|226.847000000|1.023825642692|1.020435979560|0.996689218368|
|24|0|234.495000000|234.560000000|237.696000000|1.013650610887|1.013369713506|0.999722885402|
|24|1|234.432000000|234.656500000|237.697000000|1.013927279552|1.012957237494|0.999043282415|
|24|2|234.624000000|234.720000000|237.696000000|1.013093289689|1.012678936605|0.999591002045|
|24|3|234.624000000|234.656000000|237.568000000|1.012547735952|1.012409654984|0.999863630165|
|24|4|234.432000000|234.688000000|237.792000000|1.014332514333|1.013226070357|0.998909190074|
|24|5|234.368000000|234.720000000|238.560000000|1.017886400874|1.016359918200|0.998500340832|
|25|0|242.464000000|242.336000000|245.279000000|1.011609970965|1.012144295524|1.000528192262|
|25|1|242.432000000|242.335000000|245.632000000|1.013199577614|1.013605133390|1.000400272350|
|25|2|242.336000000|241.952000000|245.152000000|1.011620229764|1.013225763788|1.001587091655|
|25|3|242.497000000|242.048000000|245.471000000|1.012264069246|1.014141823109|1.001855003966|
|25|4|242.368000000|242.144000000|245.440000000|1.012674940586|1.013611735166|1.000925069380|
|25|5|242.336000000|241.920000000|245.184000000|1.011752277829|1.013492063492|1.001719576720|
|26|0|247.456000000|247.839000000|251.136000000|1.014871330661|1.013302991055|0.998454641925|
|26|1|247.776000000|247.871000000|251.168000000|1.013689784321|1.013301273646|0.999616736125|
|26|2|247.680000000|247.711000000|250.816000000|1.012661498708|1.012534768339|0.999874854165|
|26|3|247.647000000|247.936000000|250.816000000|1.012796440094|1.011615900878|0.998834376613|
|26|4|247.552000000|247.904000000|250.848000000|1.013314374354|1.011875564735|0.998580095521|
|26|5|247.744000000|247.743000000|251.296000000|1.014337380522|1.014341474835|1.000004036441|
|27|0|253.728000000|253.472000000|256.864000000|1.012359692269|1.013382148719|1.001009973488|
|27|1|253.887000000|253.664000000|256.928000000|1.011977769638|1.012867415163|1.000879115681|
|27|2|253.696000000|253.472000000|256.864000000|1.012487386478|1.013382148719|1.000883726802|
|27|3|253.504000000|253.696000000|256.448000000|1.011613228983|1.010847628658|0.999243188698|
|27|4|253.631000000|253.759000000|256.927000000|1.012995256889|1.012484286272|0.999495584393|
|27|5|253.632000000|253.536000000|257.152000000|1.013878374968|1.014262274391|1.000378644453|
|28|0|261.088000000|261.631000000|264.543000000|1.013233086163|1.011130179528|0.997924557870|
|28|1|261.376000000|261.632000000|264.383000000|1.011504499265|1.010514768836|0.999021526419|
|28|2|261.152000000|261.568000000|263.967000000|1.010779163093|1.009171611206|0.998409591387|
|28|3|261.280000000|261.440000000|264.032000000|1.010532761788|1.009914320685|0.999388004896|
|28|4|261.087000000|261.568000000|263.935000000|1.010908241314|1.009049272082|0.998161090042|
|28|5|261.023000000|261.536000000|264.608000000|1.013734421871|1.011745992903|0.998038510951|
|29|0|268.864000000|268.736000000|272.480000000|1.013449178767|1.013931888545|1.000476303882|
|29|1|268.896000000|268.960000000|272.447000000|1.013205849102|1.012964753123|0.999762046401|
|29|2|268.800000000|268.864000000|272.096000000|1.012261904762|1.012020947393|0.999761961438|
|29|3|268.895000000|268.927000000|271.776000000|1.010714219305|1.010593953006|0.999881008601|
|29|4|268.960000000|268.928000000|271.904000000|1.010945865556|1.011066158972|1.000118990957|
|29|5|268.864000000|268.832000000|272.127000000|1.012136247322|1.012256725390|1.000119033448|
|30|0|278.432000000|278.528000000|281.440000000|1.010803355936|1.010454963235|0.999655330882|
|30|1|278.464000000|278.432000000|281.440000000|1.010687198345|1.010803355936|1.000114929318|
|30|2|278.431000000|278.496000000|281.056000000|1.009427829516|1.009192232563|0.999766603470|
|30|3|278.496000000|278.527000000|281.312000000|1.010111455820|1.009999030615|0.999888700198|
|30|4|278.496000000|278.527000000|281.247000000|1.009878059290|1.009765660062|0.999888700198|
|30|5|278.527000000|278.432000000|281.280000000|1.009884140496|1.010228709344|1.000341196414|
|31|0|282.175000000|282.303000000|285.536000000|1.011911048108|1.011452233947|0.999546586469|
|31|1|281.952000000|282.112000000|285.631000000|1.013048320281|1.012473769283|0.999432849365|
|31|2|282.048000000|282.080000000|285.247000000|1.011342041071|1.011227311401|0.999886557005|
|31|3|281.824000000|282.176000000|285.024000000|1.011354604292|1.010092991608|0.998752551599|
|31|4|281.856000000|282.208000000|285.248000000|1.012034514078|1.010772196394|0.998752693049|
|31|5|281.856000000|282.080000000|285.440000000|1.012715712988|1.011911514464|0.999205899036|
|32|0|286.464000000|286.368000000|290.975000000|1.015747179401|1.016087691362|1.000335232987|
|32|1|286.464000000|286.496000000|290.592000000|1.014410187668|1.014296883726|0.999888305596|
|32|2|286.432000000|286.400000000|290.143000000|1.012955954642|1.013069134078|1.000111731844|
|32|3|286.560000000|286.368000000|290.304000000|1.013065326633|1.013744552464|1.000670465974|
|32|4|286.496000000|286.463000000|290.335000000|1.013399838043|1.013516579803|1.000115198123|
|32|5|286.592000000|286.400000000|290.623500000|1.014067036065|1.014746857542|1.000670391061|


## Packed-FC1 epilogue and FC2 operand-staging cohort (2026-10-04)

This separate current public cohort qualifies the updated packed-FC1 epilogue
and FC2 operand staging. All historical sections above remain independent
evidence; their uses of “current” refer to those earlier cohorts. The new
public implementation beats `flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe`
at **32/32 shapes and 192/192 capture comparisons**, with no ties or regressions
against that named baseline. Public geometric-mean speedup is
**1.024821927359×**; the minimum is **T14,
161.502948222µs versus 162.606988718µs,
1.006836039267×**. The equivalent source implementation measures
1.024913981537× against the same baseline.

The geometry remains B200 sm_100a, H=4096, I=2048, E=256, top-k=6, every integer
T=1..32 and clamped SwiGLU limit 10.0. This is the routed expert kernel; the
shared expert remains outside it. All three arms consume the same physical
synthetic NVFP4 weights, activations, routes and scales, with equivalent clamp
parameters and BF16 output. The normal public JIT library was built and
qualified independently; a source implementation's binary is not substituted
for the public library.

Timing uses `loom.bench.bench_gpu_time`, CUPTI and cold L2 around the complete
call, including planning, projections, activation/quantization and weighted
finalization. Each row is the equal-weight geometric mean of six balanced
capture medians per arm:192 captures per arm,576 arm-capture observations.
No best-capture selection or historical timing reuse is used. The six fixed
captures describe variation, not independent randomized trials or a confidence
interval. All times below are microseconds; speedup is FlashInfer/implementation.
Source/public below1 means that the public implementation is slower than its
equivalent source control; that comparison is reported, not the baseline gate.

| T | Captures per arm | Source µs | Public µs | FlashInfer µs | Source speedup | Public speedup | Source/public |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 24.975504270 | 25.097973262 | 29.390014399 | 1.176753593482× | 1.171011463457× | 0.995120363297× |
| 2 | 6 | 41.306282206 | 41.194393657 | 44.687346548 | 1.081853513842× | 1.084791948136× | 1.002716111060× |
| 3 | 6 | 55.999754338 | 55.860900600 | 60.079534165 | 1.072853530795× | 1.075520328515× | 1.002485705312× |
| 4 | 6 | 68.058127881 | 67.914132468 | 72.697992025 | 1.068175018740× | 1.070439824285× | 1.002120256985× |
| 5 | 6 | 79.940631763 | 79.967282618 | 83.188698665 | 1.040630988646× | 1.040284175498× | 0.999666728022× |
| 6 | 6 | 91.706132468 | 91.871605624 | 93.780500232 | 1.022619727909× | 1.020777851819× | 0.998198865091× |
| 7 | 6 | 100.815145060 | 100.996295439 | 103.715839503 | 1.028772407573× | 1.026927166503× | 0.998206366095× |
| 8 | 6 | 110.014811760 | 109.977577560 | 111.561274343 | 1.014056857964× | 1.014400178819× | 1.000338561741× |
| 9 | 6 | 122.025595810 | 121.817580934 | 123.284571718 | 1.010317310070× | 1.012042521064× | 1.001707593225× |
| 10 | 6 | 128.265780539 | 128.265763020 | 129.465865723 | 1.009356238109× | 1.009356375972× | 1.000000136585× |
| 11 | 6 | 135.721788573 | 135.588281223 | 136.964234250 | 1.009154356795× | 1.010148023232× | 1.000984652576× |
| 12 | 6 | 145.694769097 | 145.444251906 | 146.745470214 | 1.007211659857× | 1.008946508999× | 1.001722427580× |
| 13 | 6 | 155.044303155 | 154.734980423 | 156.772120549 | 1.011144023733× | 1.013165349688× | 1.001999048511× |
| 14 | 6 | 161.513607854 | 161.502948222 | 162.606988718 | 1.006769589749× | 1.006836039267× | 1.000066002707× |
| 15 | 6 | 169.102780142 | 169.113447213 | 170.499759003 | 1.008261122966× | 1.008197525468× | 0.999936923584× |
| 16 | 6 | 179.070136295 | 178.958307645 | 180.552736570 | 1.008279439023× | 1.008909499343× | 1.000624886609× |
| 17 | 6 | 184.782309514 | 184.664982242 | 188.600895862 | 1.020665324279× | 1.021313806070× | 1.000635352035× |
| 18 | 6 | 188.627990366 | 188.787961534 | 191.779769952 | 1.016708970821× | 1.015847453377× | 0.999152641053× |
| 19 | 6 | 194.686988602 | 194.409460044 | 197.641621202 | 1.015176322883× | 1.016625534361× | 1.001427546571× |
| 20 | 6 | 202.558578291 | 202.345306738 | 205.630812529 | 1.015167139623× | 1.016237123777× | 1.001053998019× |
| 21 | 6 | 209.721428779 | 210.009478510 | 212.361012477 | 1.012586142069× | 1.011197275396× | 0.998628396523× |
| 22 | 6 | 221.252140882 | 221.454812662 | 225.108050848 | 1.017427673020× | 1.016496540049× | 0.999084816547× |
| 23 | 6 | 224.873474533 | 224.547800544 | 228.504758061 | 1.016148118561× | 1.017621893902× | 1.001450354839× |
| 24 | 6 | 237.027151895 | 237.283320805 | 240.680600934 | 1.015413630925× | 1.014317399624× | 0.998920409115× |
| 25 | 6 | 244.851466891 | 244.862247245 | 248.157915448 | 1.013503895234× | 1.013459274512× | 0.999955973803× |
| 26 | 6 | 250.214241872 | 250.851318079 | 253.352818811 | 1.012543558335× | 1.009972045395× | 0.997460343391× |
| 27 | 6 | 255.544978473 | 256.211956916 | 259.699756041 | 1.016258498183× | 1.013612944406× | 0.997396770820× |
| 28 | 6 | 263.128766316 | 263.779491283 | 266.414179197 | 1.012485950994× | 1.009988221227× | 0.997533072173× |
| 29 | 6 | 271.816821541 | 271.752795733 | 275.326076692 | 1.012910367839× | 1.013149012689× | 1.000235603127× |
| 30 | 6 | 281.112951863 | 281.235661963 | 284.216907302 | 1.011041666414× | 1.010600523840× | 0.999563675179× |
| 31 | 6 | 284.504975861 | 284.957945421 | 287.896688079 | 1.011921451313× | 1.010312899517× | 0.998410398560× |
| 32 | 6 | 289.453439999 | 289.346977832 | 294.344052286 | 1.016896024064× | 1.017270180225× | 1.000367939447× |

Public/source differences remain mixed: public is slower at 15
aggregate shapes (T1, T5, T6, T7, T15, T18, T21, T22, T24, T25, T26, T27, T28, T30, T31). At capture level, public is faster in
89 comparisons, 5 tie, and source is faster in
98. Both source and public beat FlashInfer in all 192 captures.
This cohort does not establish a controlled causal improvement over historical
cohorts or attribute a gain to either changed device unit alone.

All 32 strict BF16 correctness shapes pass at atol=rtol=0.01 with stricter checks
preserved, together with 576 timing postchecks, 7 API/graph cases and 8 dynamic
route cases. Eight non-overlapping shape shards were reduced into the full
domain; individual shard coverage flags are not whole-domain performance gates.
The full-domain result requires FlashInfer/public>1 at every T. Public JIT
library SHA-256: `c59667990a9254947a272cc28f93798cdd927948c2f15de675b25ada5bf5c750`. Observed loaded FlashInfer baseline library
SHA-256: `74d38e3ec9ddae49151300a3bb106f09ac58c874884eb11d3a6b6f5c6dc9d7c5`. These identify this cohort, not earlier captures.

Successful physical turnaround was 2372.639219s.
Successful wrapper-worker durations sum to 5553.121231s;
validation-worker durations sum to 5504.204544s.
Parallel worker sums are not elapsed time. Prior failed orchestration attempts
are separate from these successful cohort durations.

The stock baseline remains its observed default-tactic fallback, not a claim
of an explicitly tuned optimum. This synthetic cohort is not actual checkpoint,
full-model, router or shared-expert execution. Earlier checkpoint qualification
and its T1–4 per-capture baseline-binary association gap remain unchanged.
Separate synccheck and racecheck timeouts remain **SKIPPED: sanitizer timeout
after 20 seconds**, not passes; no unchanged timed-out check was retried here.
Hardware-limit evidence and broader promotion remain incomplete. Keep the
review draft; serialized profiler evidence does not establish a normal-PDL
whole-call hardware ceiling.
