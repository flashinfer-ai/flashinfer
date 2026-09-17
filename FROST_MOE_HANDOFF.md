# Frost / FlashInfer MoE pathfinding handoff

Updated September 16, 2026. Companion drafts:
[cuDNN Frontend #1080](https://github.com/NVIDIA/cudnn-frontend/pull/1080) and
[FlashInfer #5250](https://github.com/flashinfer-ai/flashinfer/pull/5250).
Yanqin and Yihua can distill these drafts asynchronously. Validated followups
continue on the same branches; the owners can choose the eventual split.

This uses open-source Frost/CuTeDSL engine **20400**, with FlashInfer routing,
permutation and weighted finalization. It does not benchmark closed-source
cuDNN GEMM kernels. Frost needs explicit per-workload tuning; no heuristic
quality claim is made.

## September 16 routing followup and current comparison

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
Both backends prepare weights before timing. Frost BF16 currently splits and
copies gate/up and concatenates gate/up for its fused FC1 route; it does not use
the TRT-LLM MMA shuffle or BlockMajorK layout. Its prepared pack currently
duplicates FC1 storage. Preparation latency, peak memory and dynamic weight
replacement have not been benchmarked, so this evidence supports steady-state
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
