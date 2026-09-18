## Explicit K64 FC1 layout through FE and FlashInfer — 2026-09-18

Paired SM100 SwiGLU FC1 now accepts `weight_layout="k_blocked_64_v1"`.
The physical BF16 parent is contiguous `[E,K/64,2N,64]`, with gate then up
on its feature axis. Feature slices keep their parent strides and live pointer.
The TMA descriptor reads this layout directly; execution performs no packing,
allocation, compilation or synchronization. Layout participates in plan/cache
identity. Ordinary SM100/SM120 engines decline this declaration.

FlashInfer exposes the append-only `CudnnMoeConfig.fc1_weight_layout` field
and matching preparation keyword. Both must select the same layout. The
preparation helper creates the K64 parent directly from ordinary `[up,gate]`
weights; FC2 is unchanged. Prepared FC1 requires SM100, standard SwiGLU,
BF16, H/I divisible by64 and the paired engine's1–513 routed-row contract.
An unsupported graph or device declines instead of reinterpreting the weights.
The default canonical layout and existing tactics retain their behavior.

Same-input full MoE comparison,148-SM B200 at1000W, BF16 E128/top8/H2048/I768,
T8, identical FE/FI source and paired12-stage FC2 in both arms:

| Routing | Canonical FC1 | K64 FC1 | Latency reduction |
|---|---:|---:|---:|
| Unpacked | 102.855000 us | 100.863000 us | 1.937% |
| Packed | 102.914750 us | 100.942750 us | 1.916% |

Job4385856 / audit2267 passes296 raw checks,120 independent changed-reference
controls and768 CUPTI timing spans: normal/memcheck/racecheck, live inputs,
routing and both weights, retained and legacy captures, bitwise cross-arm
agreement, and fresh ABBA processes. PDL overlaps FC1/FC2, so component durations
must not be added. FC2 code is unchanged. These are synthetic full-operator
measurements, not model E2E or a fresh strongest-TRT comparison.

Native graph job4385625 / audit2272 passes144 zero-skip test executions across
normal/memcheck/racecheck,840 raw checks,252 changed-reference controls and168
actual paired-engine routes. Coverage includes R1..513 boundaries, singleton
K blocks, E257, live parents, retained graphs and multiwave schedules. Initial
job4385479's two prelaunch singleton-stride failures are preserved; flatten/view
canonicalizes those strides during preparation. No tolerance was relaxed.
CPU contracts: FE20 and FI21 pass; six new positive FE cases fail the prior
source as expected. Changed-file hooks pass; full repository CI is unclaimed.

A subsequent preparation-only improvement eliminates one768MiB canonical FC1
intermediate by concatenating directly into K64 order. On the local CPU host,
eight torch threads and three ABBA cycles measure199.338774→68.9948285ms
(65.388% lower); bytes, strides and aliases match. This excludes input loading
and H2D and is not GPU preparation latency. Audit2267 used the older two-step
helper; the final direct-helper GPU validation2287 also passes (below).
Preparation remains outside hot-path timing and must be included for frequent
weight-update workloads. Retained prepared storage is unchanged.

The final direct-helper validation has now passed on B2001000W: job4386434,
audit2287,296 raw checks,120 independent changed-reference controls and768
CUPTI spans. It additionally checks CPU-prepared versus GPU-prepared weights
for exact bytes, strides and parent aliases before capture in every process.
Normal/memcheck/racecheck, retained/live/legacy replay and fresh ABBA timings
all pass. This exact-source repeat gives:

| Routing | Canonical FC1 | K64 FC1 | Latency reduction |
|---|---:|---:|---:|
| Unpacked | 102.366875 us | 101.046750 us | 1.290% |
| Packed | 102.326750 us | 100.982750 us | 1.313% |

This repeat validates the revised helper used by the published FI integration.
Its preparation performance remains separately scoped to the CPU benchmark;
no GPU preparation speedup or model E2E claim is made.

Latest companion evidence available separately: public FC2 depth audit2199;
strongest-native T8 comparator audit2167 (Frost102.232750us vs TRT96.686313us,
Frost5.737% slower, all704 FC1/FC2/PDL combinations); SM120 audit2120 (Frost
82.794563us vs CUTLASS87.114375us,4.959% lower after128 joint configurations).
Those use their own matched snapshots and must not be combined with this
K64 percentage. TRT preparation includes gate/up interleave, row shuffle and
BlockMajorK128B conversion outside timing. The pre-existing FC2 cancellation
limitation documented by audit2214 remains unresolved; no broad accuracy or
performance-roof claim is made.

Credit: NVIDIA Frost/CuTeDSL and the existing paired implementation; Kernel
Factory624/2f5c/f19299 and later FC2-depth work; Yanqin Zhai's PR1090 and
Yanqin/Yihua's parallel distillation; CUTLASS example113. TRT-LLM preparation
motivated this exploration; its kernel body and shuffle implementation were
not copied. The separate KF M64 scheduler ablation remains private experimental
work pending combination and integration evidence.

## Paired FC2 admission extended to 513 routed rows — 2026-09-17

The SM100 paired BF16 projection engine20402 now accepts1–513 total routed
rows, so FlashInfer can select it for T8/top8 and T64/top8 as well as T1.
The canonical weight layout, stride/alignment checks, public knobs, caller
workspace and stream contracts are unchanged. The executable kernel is
unchanged; this extends access to its existing persistent multiwave path.
Ordinary engine20400 remains an eligible alternative; enumeration is not a
performance ranking and no universal dispatch preference is introduced.

With paired FC1 held fixed, E128/top8/H2048/I768, synthetic BF16 inputs and
fixed saved CPU bytes, one148SM/1000W B200 measured complete FI MoE:

| Tokens | Routing | Ordinary FC2 baseline | Paired FC2 | Latency reduction |
|---|---|---:|---:|---:|
| 8 | Unpacked | 115.871625 us | 103.140125 us | 10.988% |
| 8 | Packed | 115.879500 us | 103.183750 us | 10.956% |
| 64 | Unpacked | 209.404250 us | 208.520125 us | 0.422% |
| 64 | Packed | 209.419875 us | 208.451500 us | 0.462% |

Audit2096 independently validates both shapes:592raw output comparisons,
240observable negative controls and1536cold-L2CUPTI spans. Both routing modes
pass normal/memcheck/racecheck, retained and legacy captures, live activations,
IDs/scales and both expert weights, bitwise cross-arm agreement, then four
fresh ABBA processes. The native finalizer is held identical. These numbers
are direct comparisons, not sums of separate optimization percentages.
T8's FC2 completion tail falls40.924->28.272 us while FC1 stays about69 us;
T64's tail changes63.576->62.604 us. The latter benefit is modest.

Graph audit2092 passes105zero-skip test executions:13GPU cases plus22contract
cases in each normal/memcheck/racecheck mode,390raw output comparisons,
117input-mutation controls and78actual paired-engine routes. New GPU cases
include R9/17/64/512/513, empty/skewed groups, E257, pitched weights and
multiple persistent waves. CPU regression validation first observed five
new accepted-row cases rejected by the original admission, then all22contract
cases passed. R0 and R514 declarations remain explicitly rejected.

The published runtime and test files match the independently tested sources;
all tracked Python runtime source bytes match the full-FI capsule. This is
an interface/admission benefit, not a new kernel-code speedup. A fresh T8
comparison against the strongest exported TRT configuration remains pending;
no best-backend superiority, all-shape benefit, broad CI or model E2E claim is
made. The separate FC2 small-row scheduler candidate remains unpublished
until its full-FI composition measurement passes.

Credit: NVIDIA Frost and its persistent scheduler; KF624/2f5c paired and
compact-resource contributions; Yanqin Zhai PR1090, CUTLASS113/rank5 guidance,
and Yanqin/Yihua. FlashInfer supplies the integration and NVIDIA TRT-LLM the
native finalizer. The new contribution here is validating and exposing the
existing kernel across a wider graph contract.

This supersedes earlier pending T64/admission statements below. Earlier
measurements and frozen-source scopes remain separately identified.

## Latest validated configuration and interface results — 2026-09-17

On RTX PRO 6000 Blackwell Server (188 SMs, 600W), synthetic BF16
T1/E128/top8/H2048/I768 with PackedPrecomputed routing, Frost measures
**82.794563 us** versus **87.114375 us** for the freshly selected CUTLASS
backend winner: **4.959% lower complete-MoE latency** (audit2120). This is a
direct same-card comparison, not a sum of gains from different experiments.
Both weight preparations are outside timing; the fixed CPU fixture and
actual loaded sources, generated kernels and binary hashes are recorded.

Frost uses the existing ordinary engine20400 for both stages: FC1 tile
`CONFIG_sm120_16x64x128_16x16x32_cluster1x1_warps1x4`, FC2 tile
`CONFIG_sm120_32x128x128_16x16x32_cluster1x1_warps1x8`, with public
`SCHED_POLICY=1` for both. Relative to the preceding SM120 comparison, only
FC2's scheduler configuration changes from dynamic to static. No kernel code
or global dispatch default changes in this update. Separate fixed-geometry
ablation2099 measures84.440625->82.884375 us packed (1.843% lower) and
84.368500->82.876500 us unpacked (1.768% lower), with bitwise cross-arm outputs.
These are configuration benefits, not new kernel implementation speedups.

The competitor search covers all128 exported joint (PDL off/on, FC1, FC2)
configurations:96 execute;32 remain explicit prelaunch zero-occupancy rejects.
Winner PDL-on/(1,11) passes normal, memcheck and racecheck before four fresh
ABBA processes using cold-L2 CUPTI spans. Independent audit2120 reconstructs
829 passing raw comparisons,616 negative controls and5520 spans. The fixed
Frost geometry is source-matched to ablation2099. The exact frozen FI runtime
predates the later SM100-only native finalizer change; its separate SM120 API
regression is reported below, not presented as a full-MoE retest of a new build.

Timeline2133 helps direct further work:

| Interval | Frost | CUTLASS |
|---|---:|---:|
| Before FC1 | 2.254 us | 5.120 us |
| FC1 kernel interval | 52.532 us | 51.416 us |
| Between FC1 and FC2 | 0.380 us | 0.116 us |
| FC2 kernel interval | 25.552 us | 26.020 us |
| After FC2 through final output | 2.018 us | 4.384 us |

Frost has4kernel launches versus5. The largest advantages are routing and
finalization; FC1 remains slower. These averages of per-field medians are
not strictly additive. PDL kernel intervals can include dependency waits,
so they are not isolated compute costs. There is no performance-roof claim.

The published native-finalizer helper now also passes its separate SM120
regression (audit2110):105zero-skip tests across normal/memcheck/racecheck,
including60PDL-off live-capture executions,18expected unsupported-PDL
rejections and27invalid-metadata rejections. SM120 default PDL stays disabled.
This closes the pending SM120 regression statement in the previous update.

On B200 (148 SMs, 1000W), the unpublished expansion of existing paired FC2
admission beyond8routed rows has passed the complete T8 subexperiment:
115.871625->103.140125 us unpacked (10.988% lower), and
115.879500->103.183750 us packed (10.956% lower), audit2127. Each route passes
normal/memcheck/racecheck, retained and legacy captures, live inputs and both
weights, bitwise cross-arm agreement, and four fresh ABBA processes;296raw
comparisons,120negative controls,768spans total. The FC2 completion tail falls
40.924->28.272 us, while FC1 stays about69 us. This is an interface/admission
benefit from an existing kernel, not a newly optimized kernel. T64 and a fresh
strongest-TRT T8 comparison are still pending, so no broader adoption or
competitor speedup is claimed. The separate FC2 small-row scheduler code
candidate also remains unpublished pending complete-MoE validation.

Credit: NVIDIA Frost supplies the configurable GEMMs/scheduler; NVIDIA
TRT-LLM/CUTLASS and FlashInfer supply the competing kernels, routing and
finalization. Paired FC2 retains KF624/2f5c contributions and
YanqinPR1090/CUTLASS113 guidance; small-row work builds on KFf19299. Yanqin
and Yihua's contributions and the other specific credits below remain.
All results here are synthetic model-shaped operator measurements, not
checkpoint inference, model E2E, broad CI or all-shape superiority.

Earlier sections below are timestamped historical measurements and states.
This update supersedes their SM120 comparison and pending-regression status.

## Packed decode finalizer composition — 2026-09-17

For SM100-family BF16 packed T1/H2048/top8, the FI cuDNN runner now enables
native consumer PDL and a fixed-top8 scalar finalizer specialization. The
dependency wait stays before every routing-metadata and FC2-data read. Ordered
FP32 accumulation and the live FP32-to-BF16 scale-rounding boundary are retained.
Unpacked dispatch keeps the original finalizer. The helper API permits explicit
PDL only for SM100-family scalar T1..64/H2048/top8; default runner adoption is
narrower, limited to the measured packed T1 case. Its cache identity advances.

On a 1000W B200, E128/I768 with the already-published small-row FC1 and paired
FC2, the directly measured complete synthetic FI MoE comparison is:

| Routing | Published finalizer | New composition | Latency change |
|---|---:|---:|---:|
| Packed | 28.652000 us | 28.043375 us | 2.124% lower |
| Unpacked | 28.631750 us | 28.655750 us | 0.084% higher |

Audit2080 reconstructs296 passing raw outputs,120 independent mutation controls
and768 cold-L2 spans. Both routing modes pass normal/memcheck/racecheck,
retained captures, live X/IDs/scales/FC1+FC2 weights and legacy separate-weight
checks. Outputs agree bitwise across the two arms. Four fresh ABBA processes
measure the selected composition directly; separate scheduler/PDL/unroll gains
are not added. Unpacked kernel routes are identical and show no measured win.
The longer PDL finalizer interval includes earlier launch and waiting; it is
not an isolated compute cost. This is operator evidence, not model E2E or a
refreshed comparison against the strongest TRT configuration.

The updated public native-finalizer test retains the old scalar/vector, masking,
rounding and live-capture cases. It adds bounded PDL atT1/T8/T64 and invalid
token/hidden/topk bounds. On B200, the old implementation fails exactly6new
PDL cases; the candidate passes35tests in each normal/memcheck/racecheck mode,
zero skips (audit2103). That regression proof covers the exact runtime used by
audit2080. Python ASTs and CUDA token streams of the formatted public files
match those GPU-tested sources. SM120 runner dispatch remains PDL-disabled;
the separate SM120 helper regression subsequently passed (audit2110, above).
No broad architecture CI or all-shape speedup is claimed.

Why this composition: the prior top8-only ablation on an already-PDL baseline
improved packed latency1.086% but regressed unpacked0.381% (audit2065). The
selected default therefore uses it only for packed T1. T8/T64 PDL-only gains
were small and do not broaden default adoption. The broad FC2 row-range
experiment remains separate and unpublished pending complete FI timing.

Credit: NVIDIA TRT-LLM authored the native finalizer and its PDL machinery.
KF candidateb0409f identified consumer PDL as a useful component direction;
the fixed-top8 unroll adapts that native scalar kernel. Existing Frost baseline
credits remain: NVIDIAFrost, KF624/2f5c, KFf19299 small-row scheduler,
YanqinPR1090/CUTLASS113 and Yanqin/Yihua. These are distinct contributions.

## B200 comparison including both PDL settings — 2026-09-17

The completed exhaustive native comparison changes the size of the remaining
B200 gap. On one 1000W B200 (148 SMs), synthetic BF16 E128/top8/H2048/I768,
UnpackedPrecomputed routing and fixed CPU input bytes:

| Tokens | Frost | TRT-LLM | Frost higher latency | Native winner (PDL, FC1, FC2) |
|---|---:|---:|---:|---|
| 1 | 29.498000 us | 25.822125 us | 14.235% | on, 16, 20 |
| 8 | 115.135375 us | 96.579125 us | 19.214% | on, 8, 17 |
| 64 | 206.553750 us | 195.895250 us | 5.441% | on, 8, 63 |

Each shape searches all 704 exported joint configurations across PDL off/on.
All execute successfully. Each selected route passes normal execution,
memcheck and racecheck before four fresh ABBA processes with cold-L2 CUPTI.
Audit2069 independently checks15,255 raw passing comparisons,10,968 negative
controls and104,112 spans, including the search. Captured graph attributes,
kernel names and native binary hashes prove the selected PDL path. Both weight
preparations are outside timing; the CPU TRT preparation is checked byte for
byte against its GPU helper. No candidate is dropped after a timing failure.

Frost uses paired FC1 at all three sizes, paired FC2 at T1, and the previously
selected ordinary static128x128 FC2 at T8/T64. This is not a fresh Frost sweep.
The frozen T1 FC1 predates the small-row scheduler published below; T8/T64
execute the same current generic path. Neither the small-row gain nor pending
finalizer gains are added to these separately measured comparison numbers.

Timeline2086 points to FC2 as the main remaining T8 opportunity: the interval
from FC1 completion to FC2 completion is41.049 us for Frost versus24.178 us for
TRT, while FC1 intervals are68.017 versus66.415 us. At T1 those FC2 completion
tails are8.282 versus5.896 us. PDL kernel intervals include dependency waits;
they are not isolated compute costs and must not be summed across overlap.
This attribution motivates FC2 work; it does not establish a performance roof.

This supersedes the pending B200 status and older PDL-off-only comparison
below. It is complete operator evidence, not model inference or a new kernel
speedup from configuration search. The SM120 result retains its separate GPU,
packed-routing and fixture scope. Native consumer-PDL/top8 composition remains
an unpublished experiment until its direct public-baseline check completes.

Credit: NVIDIA TensorRT-LLM supplies the native kernels, weight preparation and
PDL implementation. Frost retains NVIDIA, KF624/2f5c, Yanqin PR1090/CUTLASS113,
Yanqin/Yihua and KF small-row scheduling credits documented below.

## SM120 comparison including both PDL settings — 2026-09-17

On RTX PRO 6000 Blackwell Server (188 SMs, 600W), BF16 T1/E128/top8/H2048/I768,
PackedPrecomputed routing, the current Frost path measures **84.9495625 us**
versus **88.0897500 us** for the freshly selected CUTLASS winner: **3.565% lower
full-MoE latency**. Dimensions match Qwen3-30B-A3B; weights, activations and
routing are synthetic, with a saved CPU fixture. This is operator evidence,
not checkpoint inference, a model E2E result or a new kernel speedup from tuning.

The comparison attempts all 128 exposed (PDL off/on, FC1, FC2) combinations.
96 execute successfully. The remaining 32 fail the backend's exact prelaunch
zero-occupancy check; they remain explicit unsupported records, never timings
or silently removed candidates. Winner PDL-on/(1,11) passes normal execution,
memcheck and racecheck, then four fresh ABBA processes with cold-L2 CUPTI.
Independent audit2058 reconstructs829 raw passing comparisons,616 observable
negative controls and5520 spans. Both backends prepare weights outside timing.
The exposed candidate domains match the pre-fix capture exactly.

This PR's companion FI change checks opt-in shared-memory capacity before
CUTLASS's occupancy probe. An oversized candidate previously issued an invalid
cudaFuncSetAttribute call and cleared its error internally, causing sanitizer
failure during configuration discovery. The query now returns zero occupancy
before that invalid call. The execution kernel and selected configuration are
unchanged. This fixes discovery behavior; it does not explain the timing gain.
The exact tested header differs from the formatted public header only in
whitespace; C++ token streams match.

This supersedes the SM120 PDL-off-only comparison and pending status below.
The B200 PDL-off/on comparator is still being validated. Finalizer-PDL and
native top8-unroll experiments remain unpublished pending their full gates.
No broad CI, all-shape superiority or performance-roof claim is made.

Credit: NVIDIA Frost/FlashInfer/TRT-LLM and CUTLASS provide the kernels and
occupancy-query machinery. Existing Frost integration credits, including
Yanqin/Yihua, Yanqin PR1090/CUTLASS113 and KF contributions, remain below.

## Small-row scheduler specialization — 2026-09-17

For paired FC1 declarations with at most 8 routed rows, each nonempty expert
has one token tile. The plan selects a smaller scheduler path that eliminates
row tile-count division, L2 row rasterization and unused shared-ring fields.
MMA, epilogue, resources and the persistent generic path for R9–513 are unchanged.
Frozen template parameters separate the specialized compilation/cache identity.

On a 1000W B200, BF16 T1/E128/top8/H2048/I768, complete synthetic FI MoE:

| Routing | Published generic scheduler | Specialized scheduler | Latency reduction |
|---|---:|---:|---:|
| Unpacked | 29.787375 us |  29.439375 us | 1.168% |
| Packed | 29.703875 us |  29.407875 us | 0.997% |

These are fixed-geometry ablations with the same paired FC2, routing and native
finalizer, using four fresh ABBA processes and cold-L2 CUPTI spans. Independent
audit 2018 checks 296 raw outputs, 120 observable input/weight mutation controls
and 768 spans, with normal/memcheck/racecheck all passing. Graph audit 1972 checks
51 zero-skip test executions, 420 outputs, 126 controls and 84 actual routes,
including R8/E257 pitched weights and generic R9–513 persistent scheduling.

A separate 44-case large finite cancellation diagnostic compares generic and
specialized outputs bitwise: all 44 match. Both versions fail 15 FP32-reference
checks. This establishes no new numerical change on these cases; it does not
waive the existing cancellation behavior or claim all-input numerical accuracy.
No new competing-backend, SM120, model-E2E or full-repository-CI claim is made.
The native PDL-off/on comparator refresh and finalizer PDL experiment remain
pending, as qualified below.

Credit: KF candidate f19299a8e3a54d5767c2e3b96fb2b21a3a46922d89663c88921ad18fc0a081a1
identified the small-row scheduler simplification. This builds on NVIDIA Frost,
KF 624/2f5c, Yanqin Zhai PR #1090 / CUTLASS example 113 and Yanqin/Yihua's parallel work.

## Comparator PDL scope clarification — 2026-09-17

The latest B200 TRT-LLM comparison (audit1913, Frost29.484375us versus
TRT28.620375us) uses `ExecutionConfig(enable_pdl=False)` for TRT. Its earlier
352-pair search varied native tactics with that flag fixed. Frost's SM100
kernels internally enable PDL despite the adapter configuration flag, so
these numbers do not establish the gap to the fastest PDL-enabled TRT path.

Likewise, the RTX PRO6000 Blackwell Server comparison (audit1965,
Frost84.904125us versus CUTLASS87.9780625us) searched all64 native tactic pairs
with CUTLASS PDL disabled. Its3.494% advantage is scoped to that configuration,
not an exhaustive optimum across launch settings. Both native runners forward
the flag to their implementation; neither setting can be assumed neutral.

These measurements and correctness checks remain evidence for their recorded
settings. The fixed-source Frost ablations, including the9.66–9.82% compact
resource gain and larger-row admission results, keep their stated scope.
Claims of a strongest competing backend require searching PDL off and on,
then revalidating and retiming the selected route. That broader comparison
is pending; no new performance numbers or runtime changes are published here.
This clarification supersedes unqualified "strongest" wording below.

## Paired FC1 row-range extension and current SM120 evidence — 2026-09-17

Engine20401 now accepts1..513 routed rows, retaining the same generic compact
persistent kernel, public knob vocabulary and FP32 SwiGLU/BF16-output semantics.
This is an admission extension with wider validation, not a new MMA kernel or
a sweep-based optimization. It enables explicit paired FC1 for T8 and T64 at
top8. Engine20402 remains limited to1..8 routed rows. Ordinary20400 remains
available; enumerate/tune the full eligible configuration set before choosing.

On one1000W B200, BF16 E128/top8/H2048/I768, full synthetic FI MoE:

| Tokens / routing | Ordinary FC1 | Paired FC1 | Latency reduction |
|---|---:|---:|---:|
| T8 / unpacked |122.884500 us|116.789750 us|4.960%|
| T8 / packed |123.106000 us|116.486000 us|5.377%|
| T64 / unpacked |212.322750 us|206.230500 us|2.869%|
| T64 / packed |212.286375 us|206.279125 us|2.830%|

Both arms use ordinary static128x128 FC2. Ordinary FC1 uses the previously
audited dynamic64x64 configuration; this is not a fresh best-configuration
sweep or a current TRT/CUTLASS comparison. Paired FC1 also removes the separate
FC1 counter reset. PDL intervals overlap: stage medians must not be summed.
Weight preparation is outside replay. Four fresh ABBA processes per case,
normal/memcheck/racecheck, retained parents and live input/routing/scale/weight
changes pass independent audit1962 (592 raw outputs,240 negative controls,
1536 cold-L2 spans). Public graph audit1958 passes45 zero-skip test executions,
390 raw outputs,117 changed-reference controls and78 paired routes, covering
R9/17/64/512/513, pitched weights, skewed groups and persistent multiwave work.
The three runtime/test files match validated archive
`74acba5290f6f58056f7163b99f77ddbb81110c267bdd00d514cf4dd143ad87a`.

Current SM120 BF16 T1/E128/top8/H2048/I768 packed comparison, on600W RTX PRO6000
Blackwell Server, measures Frost84.904125us versus CUTLASS87.9780625us (-3.494%).
All64 CUTLASS FC1/FC2 tactic pairs were searched; winner(4,11) passes sanitizers
and four fresh ABBA processes (audit1965). This is current workload-specific
evidence, not a new SM120 code optimization or a transfer of RTX5090 timings.
Frost's routing/finalization explains the advantage; its GEMMs are slightly
slower. Same-fixture NCU audit1979 observes FC1 DRAM throughput83.77% of peak
versus CUTLASS84.76%, FC2 70.58% versus75.59%, with near-identical DRAM bytes.
These counters guide bandwidth/integration work; they do not establish a roof.

The small-R scheduler specialization and FC2+finalize fusion remain separate
experiments. Packed scheduler metadata reduced fullFI latency by only0.17-0.25%
despite a component gain; it is not promoted. These results do not change the
published T1 B200 comparison of29.484375us Frost versus28.620375us TRT-LLM.
No model-E2E or full-repository-CI success is claimed. Historical sections below
retain their original source scope; this section supersedes their pending status.

Credit: NVIDIA Frost/FlashInfer/TRT-LLM, Yanqin Zhai PR1090 and Yanqin/Yihua's
parallel work, KF624/2f5c, CUTLASS113 and canonical rank5/early-PDL contributions.

## Current paired Frost versus tuned TRT-LLM — 2026-09-17

The published compact-resource FC1/FC2 implementation now measures **29.484375 us**
against TRT-LLM **28.620375 us**, or **3.019% higher latency**, on the same 1000 W
B200. This is complete synthetic BF16 MoE, T1/E128/top8/H2048/I768, unpacked
routing. It is not yet a competitor win and does not update the T64 or SM120 result.

Independent audit 1913 reconstructs 68 raw output checks, 30 changed-reference
controls and 384 cold-L2 timing spans. Both paths pass normal, memcheck and
racecheck before four fresh ABBA processes. Actual Frost engines are 20401/20402;
TRT-LLM tactic (8,129) and its BMM symbols/cubin hashes match the prior 352-joint-tactic
search. This is a fresh replay of that winner, not a new exhaustive search.

Weight preparation is outside inference timing for both paths. TRT BF16 uses
FC1 gate/up interleave, MMA row shuffles and 128-byte K-block BlockMajorK copies;
dtype stays BF16. CPU-prepared bytes match the GPU helper exactly. Preparation
is measured separately in audit 1713. PDL stage intervals overlap; their medians
must not be summed to reconstruct full latency.

Evidence: `.fi-cudnn-work/20260914/artifacts/analysis1913/sm100.json`,
capsule `native1911`, raw output `native1912`, allocation 4366434,
GPU UUID `197a4e8d-b2bd-906c-faa7-c44f5d99c67b`. Tested runtime corresponds to
FE `7a38cf20d1df401a793a2f899ba2c37d09349172` and FI `fa75f6970e95cef61f9c75fe914cc587daeb9275`.
Credit: NVIDIA Frost/FlashInfer/TensorRT-LLM, KF 624 and candidate 2f5c,
Yanqin Zhai PR #1090, CUTLASS 113 and canonical rank5/early-PDL guidance.

## Compact resources in paired SM100 kernels

Both small-row graph engines now allocate 32 TMEM columns instead of 512, use
one accumulator stage instead of two, and omit dynamic register redistribution.
The persistent scheduler, 256-thread CTA, MMA geometry and public contracts
are unchanged. This implementation change originated in Kernel Factory
candidate 2f5c073d; its FC1 transfer was separately measured and validated.
The later bounded single-pass scheduler is not included: its FC1 transfer
regressed, and the small FC2 component gain still needs full-FI confirmation.

Fixed-tactic, full synthetic FI BF16 MoE on B200 at 1000 W, T1/E128/top8/H2048/I768:

| Routing | Paired baseline | FC1 resources only | FC2 resources only | Both | Both latency reduction |
|---|---:|---:|---:|---:|---:|
| Unpacked | 32.956125 us | 31.176125 us | 31.408125 us | 29.772000 us | 9.662% |
| Packed | 32.924000 us | 31.180375 us | 31.328000 us | 29.692000 us | 9.817% |

The baseline already includes paired graph engines 20401 and 20402. These
are same-card four-arm comparisons, not gains from selecting a different
existing tactic. Both routing modes pass normal/memcheck/racecheck; eight
fresh mirrored timing processes per mode yield 1536 cold-L2 raw spans.
Independent audit 1899 reconstructs 592 output comparisons and 240 changed
references, including retained parents, live tokens/IDs/scales/FC1 and FC2
weights, and legacy independent-weight execution. Overlapping PDL stage
durations are not summed. Weight preparation is outside both arms' timing.

Public graph tests now include 256-tile and 1024-tile persistent workloads for
both stages. Audit 1919 passes 35 cases in each of normal/memcheck/racecheck:
105 zero-skip executions, 480 raw output checks, 144 changed-reference controls
and 96 paired-kernel routes. Execute remains allocation/JIT/sync-free and
uses the caller's stream. Source archive:
`fe88c8b14abbaa254ac6de82b0257f5991b31d2c30259cbcdb56939ccbfe1cfd`.

NCU confirms essentially unchanged DRAM/L2 traffic and 210944 B dynamic shared
memory, with increased achieved bandwidth and a small active-warp change.
It does not establish a large occupancy gain or a performance roof. A fresh
strongest-TRT comparison for this version is still required; the resource
table alone is not a competitor or model-E2E win. SM120 kernels are unchanged.

# Frost / FlashInfer MoE pathfinding handoff

## September 17 followup: formal paired FC2 projection engine

Small-row BF16 grouped down projections can now explicitly select engine20402
(`frost_moe_fc2_pair`). It computes both output-channel halves in the paired
MMA kernel and stores both directly. The supported graph is one untransformed
grouped projection with BF16 input/output, FP32 accumulation,1..8 routed rows,
output width divisible by128 and K divisible by64 on SM100. Canonical expert
weights may have nonoverlapping16-byte-aligned pitches. Unsupported transforms,
epilogues, dynamic graph dimensions and knob combinations decline at planning.
Ordinary engine20400 and paired SwiGLU engine20401 retain their meanings.

FI selects the new engine through the existing `fc2_tactic` interface. Execute
uses current input/weight/offset pointers and caller workspace/stream; it does
not repack, allocate, synchronize or compile. This followup needs no FI runtime
override or API change. The engine is an explicit candidate, not a heuristic
ranking claim.

The public FE suite passes31 collected tests under each of normal execution,
memcheck and racecheck on B200, zero skips:93 executions,360 independently
audited raw outputs,108 changed-reference pairs and36 actual FC2 routes. This
includes the retained FC1/metadata regressions, pitched storage, two captures,
live weights/tokens/offsets and allocation/compile/synchronization guards.

Actual full FI on B2001000W, BF16 T1/E128/top8/H2048/I768, now measures:

| Precomputed routing | Paired FC1 + ordinary FC2 | Paired FC1 + engine20402 | Latency reduction |
|---|---:|---:|---:|
| Unpacked | 32.888000 us | 32.192000 us | 2.116% |
| Packed | 32.884125 us | 32.208000 us | 2.056% |

These are same-source, same-card cold-L2 CUDA-Graph/CUPTI full-MoE spans from
fresh ABBA processes. Both routes pass normal/memcheck/racecheck, with296 raw
output checks,120 independently changed-reference controls and768 spans.
Two retained parent captures, live X/IDs/scales/FC1 and FC2 weights, and
independent legacy weights are covered. PDL intervals overlap; stage durations
must not be added. This is a synthetic model-shaped operator result, not model
E2E or a new competitor victory. It supersedes the earlier private FC2 override
as evidence for the real public graph path.

Evidence: local `artifacts/analysis1883/public.json` and
`artifacts/analysis1887/fc2.json` under `.fi-cudnn-work/20260914`.
The tested formal-engine archive SHA256 is
`814dee6cdeaea97e01f468e5fbf1979e892335bc66cbb5fba6efd806ac5e2047`.

The preceding published-source TRT comparison is now independently audited:
Frost/TRT T1=33.276/28.424125us and T64=214.043625/197.01975us,
or17.07%/8.64% Frost latency overhead. TRT kernels match the prior352-joint-
tactic winners by BMM symbols/cubin hashes. Both preparations are outside
timing; TRT BF16 includes gate/up row interleave, MMA row shuffle and BlockMajorK.
These measurements precede engine20402 and must not be combined with the table
above to infer a new TRT gap. Current parent-binding integration also passes
on RTX PRO6000 Blackwell Server600W:72 raw checks,36 poisoned-output controls,
36 changed-reference pairs and36 routes under all three modes. That is SM120
compatibility evidence; the earlier RTX5090 performance result remains separate.

The FC2 implementation retains credit to NVIDIA Frost, Kernel Factory624,
Yanqin Zhai's PR1090, NVIDIA CUTLASS example113, and canonical rank5 pairing/
early-PDL work. TRT layout motivated exploration; no TRT kernel body is copied.
New KF compact-resource and single-pass scheduler experiments remain outside
this publication until complete-FI validation. Full repository CI and a
performance roof are not claimed. Yanqin/Yihua can continue distilling these
same consolidated drafts.

---

## September 17: real paired MoE graph engine and FI parent binding

This followup carries the small-token path into the real graph API. Engine
20401 (`frost_moe_swiglu_pair`) recognizes two BF16 grouped projections and
FP32 SwiGLU over explicit slices of one canonical gate/up weight parent.
Engine20400 remains available. The specialization is limited to SM100,
1..8 routed rows, and positive N/K divisible by64; unsupported graphs and
knobs decline. Compilation occurs while building the plan. Execute binds the
current parent and caller workspace/stream without repacking, allocating,
synchronizing or compiling.

FI now declares the parent relationship only after checking the actual view
pointers, dtype, device, dimensions and strides. Its cache key distinguishes
true shared storage from independent tensors with identical strides. A
planning-only parent-graph decline retries the existing separate-input fused
FC1 graph, preserving weight views and explicit tactics. This retry does not
catch compiler or runtime errors. Actual old/new FE compatibility passed on
B200 under normal execution, memcheck and racecheck: 144 raw output checks,
72 poisoned-output negative controls, 72 independently changed-reference pairs
and 72 captured routes. The new FE binds the shared parent; the old FE declines
that graph and preserves the existing fused separate-input path.

The FE changes include Yanqin Zhai's merged
[PR #1090](https://github.com/NVIDIA/cudnn-frontend/pull/1090) and its regression
coverage, with the prior SM120 scheduler/shared-A path retained. The new paired
kernel derives from Kernel Factory optimizer624 plus the validated early-PDL
change, using NVIDIA CUTLASS example113 layout concepts and canonical rank5
pairing guidance. TRT-LLM's gated-row interleave motivated layout exploration;
no TRT-LLM kernel body is copied. The larger-token KF candidate and subsequent
ballot-prefix experiment are not included in this engine.

On a full B2001000W, BF16 T1/E128/top8/H2048/I768 with precomputed routing,
cold-L2 CUDA-Graph/CUPTI full MoE timing measured:

| Routing | Tuned ordinary Frost | Early-PDL prototype | Real FI graph engine20401 |
|---|---:|---:|---:|
| Unpacked |33.915625us|33.127875us|33.123625us|
| Packed |33.991375us|33.147250us|33.159500us|

The real integration is2.34–2.45% lower latency than ordinary Frost and within
0.04% of the prototype. These numbers are not an additional gain over the
previously reported KF/PDL improvement. No fresh TRT/CUTLASS or model comparison
is claimed. All arms use the same FI/FE source; preprocessing is outside replay.
The full-MoE audit passes414 raw BF16 comparisons,150 changed-input negative
controls and1152 timing spans, after normal/memcheck/racecheck for both routing
modes. Two retained captures, independent legacy weights, and live X/IDs/scales/
weight updates are covered. Overlapping PDL stage durations must not be summed.

The standalone paired graph engine separately passed180 raw outputs and54
negative controls under normal/memcheck/racecheck, including16-byte-aligned
pitched storage and six geometries. The actual collected public pytest port
now also passes all9 cases (3 metadata plus6 GPU) in each of normal execution,
memcheck and racecheck on B200:27 test executions, independently audited180 raw
outputs and54 changed-reference controls, with no skips or tolerance changes.

The earlier public-port failures are retained. Its first run called the DSL
version helper without a required argument; that test-only call was repaired.
A subsequent run hit `cudaErrorStreamCaptureInvalidated` (901) after correct
eager output. GC callbacks located collection inside MagicMock construction
during capture; original/force-GC diagnostic processes were1/0/0/0 exits.
The test now constructs simple allocation/JIT guards before capture and resets
retained graphs after use. Runtime kernels, default GC behavior, input mutations,
route assertions and numerical thresholds are unchanged. All three public-port
validation modes then passed. This fixes the test harness, not a kernel speedup.
The explicit-parent CPU regression is confirmed RED against the preceding FE
source (`no lowering for node type SLICE`).

Timing above used frozen pair1820/fi_pair1825. This publication additionally
carries the planning-only compatibility retry and SM120 source reconciliation,
validated separately; the final assembled head has not been retimed.

The earlier pending SM120 parent-binding gate and published-source TRT refresh
are completed in the followup above. An initial SM120 integration run failed before kernel execution
because its source copy lacked the already validated static scheduler; the
combined source restores it without relaxing the gate. Full repository CI is
not claimed. These remain consolidated pathfinding drafts for Yanqin/Yihua to
distill; new validation and improvements will be stacked on these branches.

---

Updated September 17, 2026. Companion drafts:
[cuDNN Frontend #1080](https://github.com/NVIDIA/cudnn-frontend/pull/1080) and
[FlashInfer #5250](https://github.com/flashinfer-ai/flashinfer/pull/5250).
Yanqin and Yihua can distill these drafts asynchronously. Validated followups
continue on the same branches; the owners can choose the eventual split.

This uses open-source Frost/CuTeDSL engine **20400**, with FlashInfer routing,
permutation and weighted finalization. It does not benchmark closed-source
cuDNN GEMM kernels. Frost needs explicit per-workload tuning; no heuristic
quality claim is made.

## September 16 routing followup and current comparison

### Shared BF16 FC1 storage

The BF16 preparation now stores FC1 once. The returned up and gate tensors are
views into gate_up: each expert matrix remains contiguous, and the expert stride
includes both projections. Updating a prepared up/gate view also updates the
corresponding region used by the unfused route. Canonical FC1 input weights are
copied during preparation; the contiguous down tensor can alias the caller's
FC2 weights. Existing independently contiguous weight packs remain accepted.

Resource and autotuning keys include weight shapes and strides, separating old
contiguous packs from shared views. The BF16 cache version advances to 8; FP8's
existing serialized cache-key structure is preserved. No execution-time packing
or conversion is introduced.

For BF16 E128/H2048/I768, retained prepared weight storage is **1.125 GiB instead
of 1.875 GiB**, saving **0.75 GiB (40%)**. This removes duplicated FC1 storage
from our adapter; it does not establish a memory advantage over other backends.
Canonical input weights, allocator reservations and peak preparation memory are
excluded; preparation costs are separately measured below.

Both B200 and RTX5090 pass nine shared-weight tests in each of normal
execution, unfiltered memcheck and racecheck (27 test executions and 36 captured
routes per architecture). Coverage includes both FC1 routes, both routing
formats, interleaved legacy/shared packs and live input/routing/weight changes.
The complete-MoE regression passes both-arm normal/memcheck/racecheck and fresh
ABBA on each architecture: 272 output checks, 120 negative controls and 1536
raw spans each, with identical generated GEMM source across arms.

| GPU / tokens / routing | Legacy us | Shared us | Latency change |
|---|---:|---:|---:|
| B200 / 1 / unpacked |33.524|33.728|+0.61%|
| B200 / 1 / packed |33.572|33.752|+0.54%|
| B200 / 64 / unpacked |215.177|214.068|-0.51%|
| B200 / 64 / packed |215.721|214.340|-0.64%|
| RTX5090 / 1 / unpacked |59.496|58.976|-0.87%|
| RTX5090 / 1 / packed |59.556|59.024|-0.89%|
| RTX5090 / 64 / unpacked |790.957|789.269|-0.21%|
| RTX5090 / 64 / packed |790.692|789.220|-0.19%|

The main benefit is the 40% reduction in our retained prepared-weight storage.
B200 T1 is slightly slower; this is not a universal speedup or a claim of
unchanged latency. Fixture and timing boundaries are the same as below.
Local independent evidence: audit1704/1677 (unit), audit1710/1685 (full MoE).

Reproduce the new contracts with:

```bash
pytest -q tests/moe/test_moe_cudnn_bf16_weight_views.py
```

### Weight preparation, separate from inference

The TRT-LLM BF16 comparator uses its required gate/up interleaving, MMA row
shuffle and 128-byte K-block BlockMajorK layout. The uint8 view is a byte-level
layout operation; weights remain BF16. Both backends prepare before timing.
The historical competitor measurement used Frost's legacy dense preparation;
the shared and private blocked candidates have not yet been freshly paired
against TRT-LLM.

On a 1000 W B200, resident canonical BF16 E128/H2048/I768 weights occupy
1.125 GiB. Actual helpers, 12 mirrored warm calls each, give:

| Helper | Completion wall ms | Prepared GiB | Additional GiB | Incremental peak GiB |
|---|---:|---:|---:|---:|
| Frost legacy |2.620|1.875|1.500|1.500|
| Frost shared |1.749|1.125|0.750|0.750|
| Frost blocked (private) |3.288|1.125|1.125|1.875|
| TRT-LLM BlockMajorK |4.637|1.125|1.125|2.250|
| CUTLASS dense alias |0.040|1.125|0|0|

Audit1713 verifies 65 samples, including five first calls reported separately,
and bit-exact layout checks. Peak is PyTorch allocated memory, not reserved.
Prepared storage includes aliases; additional storage excludes canonical
weights. This measures current helpers, excluding loading, H2D and inference.
Credit NVIDIA TensorRT-LLM's preparation algorithm and FlashInfer's helpers.
The private blocked candidate extends Frost's prior blocked-weight path and
is not part of this shared-storage delivery.

### Routing fusion

The new private helper fuses stable expert ranking, input permutation and all
expert offsets into one CUDA kernel. The Frost adapter selects it only for
BF16 E128/top-k8/H2048 with T1..64. It checks tensor metadata and storage aliases
without reading device IDs on the host; IDs must already name valid local
experts. Other shapes retain the existing native sort/permutation path. The
helper writes only the three operands Frost consumes and does not replace the
generic `moe_sort` metadata interface.

With the native finalizer already enabled, the actual integrated module gives
these incremental full-MoE results. Both arms use the same source and fixed
ordinary GEMMs; the control disables only the new routing helper.

| GPU / tokens / routing | Sort + permutation, us | Fused routing, us | Reduction |
|---|---:|---:|---:|
| B200 / 1 / unpacked |36.776|33.720|8.31%|
| B200 / 1 / packed |36.760|33.680|8.38%|
| B200 / 64 / unpacked |218.448|215.460|1.37%|
| B200 / 64 / packed |218.452|215.444|1.38%|
| RTX5090 / 1 / unpacked |133.951|131.679|1.70%|
| RTX5090 / 1 / packed |133.899|131.616|1.71%|
| RTX5090 / 64 / unpacked |796.565|793.613|0.37%|
| RTX5090 / 64 / packed |796.749|793.613|0.39%|

Fixture: synthetic BF16 E128/top-k8/H2048/I768/SwiGLU, uniform precomputed
routing, TP1/EP1, full B200 (148 SMs, 1000 W) or RTX5090 (170 SMs, 600 W).
Each architecture passes both-arm normal/memcheck/racecheck and four fresh ABBA
processes per case: 272 output checks, 120 skipped-replay negative controls and
1536 raw cold-L2 CUDA Graph/CUPTI spans. Inputs, IDs and scales change; outputs,
intermediates and exposed workspaces are poisoned. Preparation/JIT, logits/top-k,
expert communication and model execution are excluded. The public FI API uses
`enable_pdl=False`; Frost generated launchers retain their own PDL behavior.

The performance candidate placed the same CUDA body in `csrc`; publication
moves it into `include/flashinfer/moe_route_permute.cuh` with a namespace-qualified
launcher. Python AST and normalized CUDA-body equivalence were checked. This
final layout additionally passes all 26 new routing and 36 existing finalizer
cases in normal/memcheck/racecheck on each GPU, including 12 captured route
checks per architecture. The eight-case timing table predates that relocation;
the fixed-geometry experiment below additionally measures the published layout.

Reproduce the targeted module checks from the repository root:

```bash
pytest -q tests/moe/test_moe_route_permute_small.py tests/moe/test_moe_native_finalize.py tests/moe/test_moe_unpermute_round_scales.py
compute-sanitizer --tool memcheck --target-processes all --error-exitcode 86 python -m pytest -q tests/moe/test_moe_route_permute_small.py tests/moe/test_moe_native_finalize.py tests/moe/test_moe_unpermute_round_scales.py
compute-sanitizer --tool racecheck --target-processes all --error-exitcode 86 python -m pytest -q tests/moe/test_moe_route_permute_small.py tests/moe/test_moe_native_finalize.py tests/moe/test_moe_unpermute_round_scales.py
```

The earlier native-finalizer change now also has complete RTX5090 confirmation:
T1 unpacked/packed 134.144/134.167 -> 133.224/133.167 us (0.69/0.75% reduction),
T64 797.267/797.084 -> 795.247/795.325 us (0.25/0.22%). These are independent
incremental comparisons; do not add their percentages to the routing table.

The stronger B200 baseline still wins. After selecting TRT-LLM from 352 joint
tactics, a separate same-card comparison measures Frost/TRT-LLM at 33.816/29.120
us for T1 and 215.600/197.120 us for T64: Frost is 16.13%/9.38% slower. Actual
native BMM symbols and cubin hashes match the search winners. Both arms pass
normal/memcheck/racecheck and fresh ABBA timing. Weight-layout preparation is
outside racecheck; normal reference creators prove bit-exact CPU/GPU prepared
weights, and every child verifies those hashes. This comparator uses the prior
validated standalone fused router; it is not a timing of the publication layout.

Separately, selecting an existing SM120 M16x64 static FC1 configuration gives
59.71/59.74 us at T1 unpacked/packed after independent gates. This is tuning,
not a new kernel optimization, and it is not included in the routing-gain table.
Best-to-best swap-AB tests have not beaten ordinary Frost configurations;
the fixed-geometry swap observation retained below is historical. Current
counter work and grid-size experiments do not yet establish a hardware roof.

### Fixed fast geometry: separating integration gains from tuning

On RTX5090, T1 packed with the same fixture above, hold both GEMMs fixed:
FC1 M16/N64/K128, warps 1x4, static scheduling; FC2 M32/N128/K128,
warps 1x8, dynamic scheduling. The published `de982b1` runtime gives:

| Routing / finalization | Full MoE, us | Reduction from baseline |
|---|---:|---:|
| Existing router / existing finalizer |62.700|0%|
| Fused router / existing finalizer |60.752|3.11%|
| Existing router / native finalizer |61.620|1.72%|
| Fused router / native finalizer |59.644|4.87%|

All four arms have identical generated FC1/FC2 source hashes and pass normal,
unfiltered memcheck and racecheck, followed by eight fresh mirrored timing
processes: 136 output checks, 60 skipped-replay negative controls and 768 raw
full-MoE spans. Thus the combined 4.87% reduction excludes configuration-search
credit. It includes adoption of NVIDIA TensorRT-LLM's finalizer and our routing
fusion, not solely new GEMM code.

A separate selected-backend comparison measures **Frost 59.762 us versus
CUTLASS 61.466 us**, or **2.77% lower latency**, for this same T1 packed shape
on a 600 W RTX5090. The search attempted all 8x8 offered stage combinations;
two FC1 tactics exceed shared-memory capacity, and all 6x8 supported joint
pairs were measured. The selected pair then passed independent normal,
unfiltered memcheck/racecheck and four fresh ABBA processes (148 output checks,
21 negative controls, 768 spans). Actual native symbols and library hashes
match the selected search result. Unused tactic enumeration is outside this
selected replay gate; its earlier sanitizer CUDA API errors remain failures.
This comparison uses the integration before file relocation. CUTLASS's two
GEMMs remain slightly faster; routing and finalization give Frost the full-MoE
advantage. This is a small fixture-specific benefit, not model-level evidence.

### Weight preparation and timing boundary

The B200 TRT-LLM BF16 comparator uses the public `prepare_weights` path:
FC1 gate/up row interleaving plus an MMA row shuffle (`epilogue_tile_m=128`),
FC2 MMA row shuffle, then `BlockMajorK` conversion of both matrices using
128-byte K blocks. These are byte-preserving permutations; weights remain BF16.
Both backends prepare weights before timing. The historical Frost BF16 comparator
splits/copies gate/up and concatenates gate/up, duplicating FC1 storage; the
shared-view followup above removes that duplication. Neither version uses the
TRT-LLM MMA shuffle or BlockMajorK layout. Dynamic weight
replacement has not been benchmarked, so this evidence supports steady-state
inference only. SM120's comparator is CUTLASS, not this TRT-LLM path.

NVIDIA TensorRT-LLM retains credit for the native finalizer and fallback router;
FlashInfer provides the contracts, JIT and permutation infrastructure. The new
stable-rank routing kernel and its bounded integration are our additions.
Yanqin/Yihua, the MegaMoE design reference and KF retain their separate
acknowledgements below. No KF gain is promoted into this runtime.

## Earlier source changes

The September 16 followup carries the runtime/test source from frozen Frontend
`ea4b86dd9823ed5849d3d6f2994f26906b2f0d8f` and FlashInfer
`0fb8ffad58680b1dac03317d75f417f1aa5fd298`. Publication commits add this updated
handoff; these frozen IDs identify the tested source, not the published heads.

| Repository | Change | Purpose |
|---|---|---|
| Frontend | Async shared-memory lifetime fence before releasing an A/B stage | Fixes the ordinary-execution FC1 corruption found with the large SM120 tile; prevents the next TMA load from reusing a stage too early. |
| Frontend | Explicit SM120 static scheduling alongside dynamic scheduling | Allows tuning away atomic work distribution when a regular workload benefits; does not assume static always wins. |
| Frontend | Shared-A paired grouped GEMM on SM120, with epilogue and shared-memory capacity accounting | Enables gate/up projection and activation fusion, saving the FP32 intermediate roundtrip and separate activation launch. |
| Frontend | Multi-output, pitched-weight, multiwave and scheduler regression coverage | Checks the new fusion and scheduler contracts under live-input capture replay. |
| FlashInfer | Explicit fused/unfused BF16 FC1 route choice | Makes the fusion route selectable and replayable rather than silently conflating different graphs. |
| FlashInfer | Independent FC1/FC2 configuration domains, joint tactic replay, cache version 7 | Allows each stage to use an appropriate tactic and prevents reuse of stale tactic records. |
| FlashInfer | Public fused-path, capture and live-weight/interleave tests | Exercises the actual interface and changing inputs without local architecture overrides. |

Earlier work remains included: SM100 scheduler/reset improvements, absolute-A
addressing, shared-A fusion, aligned independent expert strides, explicit
`blocked_128x128_v1` weight layout, native FI helper integration, and capture-safe
prepared plans. The packed layout is an operation attribute, not a freely
selectable tuning knob. Packed E4M3 support is scoped to eligible SM100 tiles;
SM120 uses ordinary BF16 weights. SM121 and SM120 FP8 support are not added.

## Validation of the followup

Correctness precedes timing; no tolerance was relaxed. Source import paths and
hashes were recorded. Tests check live inputs/routing/scales, poisoned buffers,
captured replay and actual Frost routing.

| Target | Completed checks |
|---|---|
| RTX PRO 6000 Blackwell Server Edition, 188 SMs, 600 W | 8 shared-A FE cases, 42 FI public cases, 30 single-GEMM scheduler cases in normal execution; all 8 shared-A cases under unfiltered memcheck and racecheck; 2 mandatory-fusion public cases under each sanitizer. |
| RTX 5090, 600 W | 44 public cases in normal execution (42 existing plus 2 joint-tactic cases); 2 joint-tactic cases under unfiltered memcheck. |

Historical audit receipts are retained by the pathfinding owner for handoff.

The formerly failing SM120 BF16 `256x256x64/warps2x4` FC1 case is repaired by
the included lifetime fence. On full RTX PRO 6000, T4096/E32/top2/H2048/I1024
with the retained skewed inputs, old source fails 1,643 FC1 elements while the
fence candidate passes 15 full-output and 75 stage checks at unchanged 0.02
tolerances (audit960). The repair additionally passed 12 ring and 40 public
cases in normal execution and both unfiltered sanitizers (audit993). Those
earlier fence-only gates are distinct from the expanded followup coverage above.

The original full public racecheck gate remains open: instrumented runs ran
out of host memory, including a 512-GiB allocation. A diagnostic that explicitly
synchronizes after each of 40 replays passes both modes, but changes execution
concurrency and therefore does not close the original gate. Earlier failures
and logs are retained. Full repository CI is not claimed green.

An earlier B200 captured native-sort / FP32 reference failure also remains in
the record. A standalone cooperative CUDA Graph kernel, importing neither
FlashInfer nor Frost, reproduces the subsequent `CUBLAS_STATUS_INTERNAL_ERROR`
under racecheck while its own output is correct. This narrows the interaction;
it does not turn the original full-MoE failure into a pass.

Run the focused FE cases from `test/python` and the FI cases from its root:

```bash
pytest -m L1 gemm/frost/test_moe_shared_a_sm120.py gemm/frost/test_moe_scheduler_sm120.py
pytest -q tests/moe/test_moe_cudnn_sm120.py tests/moe/test_moe_cudnn_bf16_joint.py
```

Install both matching branches, CuTe DSL >= 4.7, and verify `cudnn.__file__` /
`flashinfer.__file__`. Enable `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`; SM120 uses
`enable_pdl=False`. An editable install can silently select another checkout.

## Performance evidence and next experiments

Historical full-B200 FP8 results, E128/top8/H4096/I2048, cold-L2 CUDA Graph/CUPTI:
at T256, transferred tactics 526.522 us -> per-shape tactics 500.219 us
(5.00% lower latency), versus the measured TRT-LLM control at 519.523 us
(3.72% lower). T16 is near parity: Frost 225.129 us versus TRT-LLM 224.433 us.
These are earlier validated snapshots, not measurements of this assembled head;
Frost helper PDL was off and TRT-LLM PDL on. Preparation, logits/top-k, expert
communication and full model execution are excluded.

Separate latest RTX5090 research, BF16 T64/E128/top8/H2048/I768, precomputed
uniform routing, TP1/EP1, PDL off: four fresh same-card ABBA processes and 768
complete-MoE CUPTI spans measured 797.650 -> 775.778 us (-2.742%) for M64 compact
FC1; a later matched comparison measured 777.960 -> 773.547 us (-0.567%) for M32
padless FC1. The comparisons used different physical cards. Both include
additional experimental resource hooks that are **not in these product
commits**; do not attribute their timings to this PR head or compound the raw
latencies. FC1 accounts for roughly 67% and FC2 32% of the first candidate span.
Later counters are recorded in the current investigation; a hardware roof is not proven.

Historical combination experiment: combine Yanqin's
[SM100 swap-AB #1090](https://github.com/NVIDIA/cudnn-frontend/pull/1090) with the
FI integration and independently choose FC1/FC2 orientation. An isolated merged
prototype resolves both textual conflicts and the compiled scheduler-reset
handoff. Its CPU render/replay checks and the first target-GPU screen pass.
The first screen covers all four orientations at one fixed geometry, not a
complete tile sweep. The experimental merge is not included in these commits.

#1090 currently declines SM120 MoE swap. SM120 benefit remains an independent
question; an existing FI MXFP8 swap-AB implementation offers a design reference,
but does not establish a speedup for Frost BF16. Continue promising ideas and
stack validated changes onto these drafts so reviewers can consume them
asynchronously. Negative results should retain shapes, routes and timings.

KF found a remotely faster component candidate, but a finite-BF16 stress case
exposed NaNs. A minimal parenthesization repair passes the targeted diagnostic
and the original normal/memcheck checks. The full racecheck reached its 3,300-s
bound (exit 124) without a completed artifact, so its gate remains incomplete;
fresh fixed-candidate timing is also required. No KF speedup is attributed to
the FI path.

## Acknowledgements and provenance

The grouped-GEMM extensions build on the existing NVIDIA Frost/CuTeDSL kernels
and FlashInfer MoE infrastructure; their original copyright/license headers
and implementation provenance are retained.

Yanqin Zhai (@yanqinz2) authored [Frontend #1090](https://github.com/NVIDIA/cudnn-frontend/pull/1090),
head `068ffd87f053543a63ddf929aeb87fcc8b231380`. Credit for its MoE swap-AB
lowering, separate plain/block-scaled SM100 templates, public knob replay and
coverage belongs to that work, with thanks to Yanqin and Yihua for the parallel
distillation effort. Our isolated combination adds FI wiring, conflict/reset
integration and matched measurements. #1090's code is in that experimental
combination, not this published runtime followup. The bounded B200 result below credits
that implementation; a later runtime publication must retain this attribution.

For SM120, the design reference is the NVIDIA CuTeDSL MegaMoE kernel team's
implementation in the `bangyus/cutedsl_megamoe` fork, vendored by FI at
`d19d30a748f9e402b8a2a33083fdf530231cf647`; see
`flashinfer/moe_ep/kernel_src/sm120/swapab_cutedsl_megakernel/VENDOR.md` in the
companion FlashInfer tree for the dirty-snapshot exceptions and original source.
A private SM120 BF16 prototype now references its accumulator-coordinate
reasoning, together with #1090's ragged-N design. No MegaMoE source body was
copied. Our adaptation adds BF16 operand binding and a direct FI-layout
epilogue. This prototype is outside the published runtime; component numerical
and sanitizer checks pass, but no SM120 speedup is established yet.

Kernel Factory assisted separate candidate exploration. Its exported candidate
and local numerical repair remain experimental and are not promoted by this
followup. Future gains will distinguish upstream contribution, adaptation,
integration and configuration search, with workload-specific evidence.


## Combined prototype: first B200 result (September 16)

BF16 T64/E128/top8/H2048/I768, synthetic Qwen-shaped inputs, precomputed uniform
routing, TP1/EP1, FI API `enable_pdl=False` (see qualification below),
full 148-SM/1000-W B200. At one fixed 128x64 geometry,
complete-MoE cold-L2 CUPTI spans are:

| FC1 swap | FC2 swap | Complete MoE, us |
|---|---|---:|
| No | No | 276.481 |
| No | Yes | **226.992** |
| Yes | No | 286.817 |
| Yes | Yes | 238.288 |

FC2-only swap lowers complete latency **17.899%** at this geometry. FC2 changes
117.680 -> 68.160 us; FC1 remains about142.6 us. Credit for the swap lowering
and templates belongs to Yanqin's #1090; this pathfinding work adds FI stage
selection, compiled-reset integration and measurement. This is evidence that
the two efforts combine usefully on this fixture, not a comparison between
independently tuned best configurations.

All four arms pass20 strict full-output checks and2 skipped-replay negative
controls before384 raw spans. Inputs/routing change and buffers are poisoned.
Independent audit1453 verifies exact source hashes, all spans and actual stage
routes. Synchronization validation, fresh-process repetition, tile tuning and a
tuned competing-backend comparison remain open. This experimental
merge is not included in this PR's runtime changes; no model-E2E gain is claimed.

## Qualification of experimental observations

The combined prototype's initial timing observations above do not establish a
production-ready performance gain. Successful synchronization validation and
confirmation against separately tuned configurations remain required before
promotion. No SM120 orientation-specific or complete-MoE benefit is claimed.

The PDL label refers to the public FI API flag; Frost's generated launchers
control their own PDL setting. It should not be read as all kernels disabling
PDL. These qualifications do not change the previously reported arithmetic
or the acknowledgements above.

### Native finalizer reuse and attribution

The BF16 finalizer implementation comes from NVIDIA TensorRT-LLM, as preserved
in FlashInfer's existing
[`trtllm_fused_moe_dev_kernel.cu`](https://github.com/flashinfer-ai/flashinfer/blob/d75167cfc927c11aac42954fb6cb0ce7e304f7ee/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu),
including its original
copyright and license. This follow-up reuses that implementation through the
existing `moe_utils` JIT module. FlashInfer's staged finalize binding provides
the integration reference.

Our additions are the checked helper interface, Frost adapter wiring and fused
FP32-to-BF16 routing-weight rounding required by `PackedPrecomputed`. The rounding
is performed inside the existing scalar/vector kernels and retains live weights
on every CUDA Graph replay. Kernel authorship remains with NVIDIA TensorRT-LLM;
our contribution is the integration and this numeric-contract adaptation.

Yanqin Zhai's PR1090 implementation and Yanqin/Yihua's collaboration retain the
separate attribution above. Each performance claim should name the tested GPU,
fixture, baseline and scope; component timing and complete MoE timing are separate.

The new helper option is append-only and defaults off. The Frost adapter selects
it for BF16 H2048/top-k8 with 1..64 tokens; it preserves the existing wide-finalizer
selection elsewhere. The reused native kernel remains responsible for scalar or
vector dispatch. Unsupported metadata is rejected before its launch.

The proposed combined `moe_utils` module passed the targeted native-finalizer and
existing scale-rounding tests on B200 and RTX5090, including CUDA Graph replay,
memcheck and racecheck. SM90 target compilation also passed; this is not Hopper
execution coverage. Reproduce the targeted checks from the FI repository root:

```bash
pytest -q tests/moe/test_moe_native_finalize.py tests/moe/test_moe_unpermute_round_scales.py
compute-sanitizer --tool memcheck --target-processes all --error-exitcode 86 python -m pytest -q tests/moe/test_moe_native_finalize.py tests/moe/test_moe_unpermute_round_scales.py
compute-sanitizer --tool racecheck --target-processes all --error-exitcode 86 python -m pytest -q tests/moe/test_moe_native_finalize.py tests/moe/test_moe_unpermute_round_scales.py
```

Complete-MoE confirmation now passes on B200 for the actual combined FI module.
The following synthetic BF16 E128/top-k8/H2048/I768/SwiGLU cases use precomputed
uniform routing on one full 148-SM/1000-W B200. Both arms use the same source and
fixed ordinary GEMMs; the control explicitly disables the new finalizer. Each
case passes both-arm normal/memcheck/racecheck and four fresh ABBA processes,
using cold-L2 CUDA Graph/CUPTI full spans and cached independent references.

| Tokens / routing | Existing finalizer, us | Native finalizer, us | Full-MoE reduction |
|---|---:|---:|---:|
| 1 / unpacked |38.423|36.552|4.87%|
| 1 / packed |38.455|36.536|4.99%|
| 64 / unpacked |219.981|217.077|1.32%|
| 64 / packed |220.001|217.054|1.34%|

FC1 uses the 64x64x128 ordinary tile (static at T1, dynamic at T64). FC2 uses
64x128x128 at T1 and 128x128x128 at T64, both static. The frozen FE integration
contains #1090 plus the integration/reset/scheduler adaptations; these runs
select ordinary kernels and establish no swap-AB benefit. They validate the
published FI module, not every combination of FE revisions or the entire draft.
The public FI API uses `enable_pdl=False`; generated Frost launchers retain their
own PDL behavior.

These are gains over existing Frost finalization. They are not new comparisons
against another MoE backend or a model-E2E claim. SM120 complete-MoE confirmation is now complete; see the current update above.
Its API/capture and component sanitizer checks remain separate evidence.
This remains a draft for asynchronous review and further measured follow-ups.
