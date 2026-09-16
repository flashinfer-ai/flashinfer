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

## Latest source changes

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

Audit receipts: `analysis1055/result.json` and `analysis1188/result.json` under
the local experiment root
`/home/scratch.yanxu_libs/cudnn_frontend/.fi-cudnn-work/20260914`.

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
Actual memory/tensor counters are being collected; a hardware roof is not proven.

Primary next direction: combine Yanqin's
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
