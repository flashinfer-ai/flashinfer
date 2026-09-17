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
