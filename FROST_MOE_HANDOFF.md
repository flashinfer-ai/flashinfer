# Frost / FlashInfer MoE integration handoff

This draft collects the complete integration for continued development and review.
Yanqin can adjust the design, choose which changes to keep, and decide whether to
split it later. The companion repository carries the other half of the same work;
the individual optimization experiments are not separate PRs.

## Code and validation baseline

The measured implementation uses OSS Frost/CuTeDSL engine **20400** for both GEMMs.
FlashInfer's native kernels provide routing, permutation, and weighted finalization.
This is not a closed-source cuDNN kernel performance result.

The integration has been ported to these pinned upstream revisions:

- cuDNN Frontend: `6ca9fa2aa37fb483faddafb1b0f3a8bd01b5f261`.
- FlashInfer: `c11c1090172f578bad37b8bca2b40e4161d72144`.
- Validation environment: full B200, 148 SMs, 1000 W;
  CuTe DSL 4.7, CUDA 13.2, cuDNN 9.27, PyTorch 2.13.

Frontend retains upstream's architecture-family compiler/codegen facades, with
SM100 changes in the SM100 tree. SM120 declines the packed layout and static
scheduler. The near-zero tanh correction applies to both families. FlashInfer
uses the upstream independent weight/activation/output quantization axes, with
cross-product capability tests and updated curated fuzzer inputs. Both imports
and the rebuilt native bindings were verified against these checkouts.

Historical timings below predate this port. Their initial bases were Frontend
`b2712de2da3832e9ad607867d22b2a1429c89cdc` and FlashInfer
`df8b5c1745c51f44b1e8a492b11c3f6cb157cfd2`; no ported-head timing is claimed.

## What is included

| Area | Change and purpose |
|---|---|
| Frost grouped GEMM | Correct scheduler reset/termination, selectable static scheduling, absolute-A addressing, and shared-A wide MMA for fused gate/up GEMMs. These remove reset/addressing work or improve reuse under suitable shapes. |
| Graph layout | Versioned `weight_layout="blocked_128x128_v1"` attribute and native rank-5 TMA addressing. Packing is preparation work; execution does not repack. |
| Expert strides | Independent aligned expert pitches, including shared up/gate storage. This removes unnecessary preparation copies; it does not reduce total retained weight bytes. |
| Configuration domain | Public engine/knob records, eligible packed cluster M/N shapes, explicit per-stage candidates and joint FC1/FC2 selection. Frost still requires workload-specific tuning. |
| FlashInfer integration | BF16 and calibrated per-tensor FP8 MoE runners, typed gated activations, live scales, native offsets, exact plan replay, and capture-safe prepared resources. |
| Native finalization | BF16 tiled weighted finalizer with alignment/type guards and a proposed larger-token dispatch range. Component benefit is measured; the newest full-MoE range comparison remains pending. |
| Correctness | Physical scale-capacity preservation, near-zero tanh lowering, graph/input-lifetime checks, changed-input capture, negative controls, and sanitizer regressions. |

Packed support is scoped to SM100 E4M3, N/K multiples of 128, one-CTA MMA,
eligible N64/128 K128 tiles, and power-of-two cluster M/N whose product is at most
16. These results establish neither SM120 MoE support nor an architecture-wide
advantage. Layout is an operation attribute, not a freely selectable tactic knob.
Classic backend serialization cannot encode the Python-only layout; replay must
reconstruct the graph contract before selecting the engine and knobs.

## Measured performance and limits

Full B200, FP8 SwiGLU, E128/top8/H4096/I2048. Complete MoE starts with prepared
top-k and quantized activations and ends at weighted output. Preparation,
logits/top-k, expert parallel communication and model execution are excluded.
Cold-L2 CUDA Graph/CUPTI; mean of two fresh-process medians. Frost helper PDL is
off; the tuned TRT-LLM control uses PDL on. These are previously validated source
snapshots, not new measurements of the assembled PR branches.

| Tokens | Frost with transferred T64 tactics (us) | Frost with per-shape tactics (us) | TRT-LLM (us) |
|---:|---:|---:|---:|
| 16 | 231.506 | 225.129 | 224.433 |
| 256 | 526.522 | 500.219 | 519.523 |

T256 is 5.00% faster than transferred tactics and 3.72% faster than the measured
TRT-LLM control. T16 is near parity with TRT-LLM. A separate T64 comparison was
403.489 us versus 415.362 us (2.86% lower latency). This is a small, shape-specific
benefit, not a broad backend or model-E2E win.

The explicit configurations for the two rows are:

- T16: FC1 `CONFIG_sm100_64x64x128_64x64x32_cluster1x4_1ctamma`, static;
  FC2 `CONFIG_sm100_64x128x128_64x128x32_cluster1x4_1ctamma`, static.
- T256: FC1 `CONFIG_sm100_64x128x128_64x128x32_cluster1x4_1ctamma`, dynamic;
  FC2 `CONFIG_sm100_128x128x128_128x128x32_cluster1x2_1ctamma`, static.

Both stages use packed weights. Each shape was exhaustively searched over the
then-supported 80-by-80 stage domain before fresh-process confirmation. No
heuristic-quality claim follows from these winners. The CUTLASS FP8 comparator
failed the unchanged strict relative-L2 gate in one investigated path; it is not
used as a performance headline. Strict failures are not timed or hidden.

## Validation and continuation

The ported sources passed 361 Frontend tests, 4 grouped block-scale cases and
211 FlashInfer tests on full B200, with no skips. A subsequent curated fuzzer
run passed 5 cases and memcheck passed 43 cases with zero errors. Changed-file
hooks pass; runtime, test and binding hashes were frozen during these runs.

A subsequent fix catches cuDNN's dedicated `cudnnGraphNotSupportedError` when
shared-input FC1 fusion declines, allowing the existing FP32 FC1 plus activation
fallback to run. On full B200, the exact new regression fails against the old
method with that exception; the corrected source passes both decline variants,
four typed-activation fallback cases and two capability-contract tests (8 tests,
no skips). Capture replay changes inputs and poisons intermediate/output buffers.
This is an interface/correctness fix; no speedup is claimed for it.

**Racecheck is unresolved; this is not an all-tests-passing or merge-ready claim.**
The full selected racecheck suite failed, first in the 8193-token paired expert
stride case. Focused reruns fail in the independent reference's `cublasSgemm`,
including runs with zero reported race hazards. The separately reported PyTorch
`MaxNanFunctor` hazards reproduce without Frost, but that does not explain the
cuBLAS failure. Staged module/plan/forward diagnostics pass; the public test's
capture-and-live-weight sequence still fails under racecheck. Keep the failure
visible and continue isolating it. No numerical tolerance was relaxed.
Further staged diagnostics pass after separately capturing FC1, FC2, or both
GEMMs and then making eight reference calls. Capturing the complete native-routing
MoE still fails on the first subsequent reference call. Native routing,
permutation and finalization capture interactions remain under investigation;
this localization is not a root-cause finding or a blanket kernel clearance.

The old-snapshot finalizer range comparison also remains incomplete. A native
routing-only reproducer passes unfiltered racecheck but fails when the original
finalizer-only filter is added, before any Frost/finalizer execution. This makes
filtering a causal condition, not a complete diagnosis. The full comparison has
not been declared passed or timed through a failing gate.

Full repository CI and performance confirmation of the port remain open.

Install both matching branches and confirm `cudnn.__file__` and
`flashinfer.__file__` before testing. An existing editable install may still point
at a different checkout. Start with these targeted checks on a supported GPU:

```bash
# Frontend: run from test/python.
pytest -m 'L0 or L1 or L2 or L3 or L4' gemm/frost/test_moe_counter_reset.py gemm/frost/test_moe_absolute_a.py gemm/frost/test_moe_wide_mma.py gemm/frost/test_moe_packed_cluster_m.py test_variant_pack_normalization.py

# FlashInfer: run from the repository root.
pytest tests/moe/test_unified_moe_cudnn.py tests/moe/test_moe_cudnn_fp8.py tests/moe/test_moe_cudnn_fp8_joint.py tests/moe/test_moe_cudnn_fp8_blocked.py tests/moe/test_moe_cudnn_fp8_expert_stride.py tests/moe/test_moe_native_tiled_token_range.py
```

Recommended continuation order:

1. Resolve or precisely isolate the current-port racecheck/reference failure;
   retain the original failing public regression and its artifacts.
2. Confirm complete-MoE performance on the port with identical tactics and
   strict gates; finish the T1025/T2048/T3072 finalizer-range comparison.
3. Finish SM120 interface validation before changing public support. A separate
   full RTX PRO 6000 Blackwell Server Edition experiment (188 SMs, 600 W) passed
   BF16 numerics, live-input capture and actual Frost routes for two tile
   configurations and both routing implementations. It required process-local
   fixes for the native utility JIT architecture whitelist and the dedicated
   `cudnnGraphNotSupportedError` fallback. The fallback fix is now included in
   this draft; the native utility JIT whitelist and public SM120 support remain
   unchanged. Sanitizer and performance comparisons are pending.
4. Continue kernel and tuning work only when candidates survive correctness,
   changed-input capture, actual routes and complete-MoE confirmation.

Recent raw FC2 Kernel Factory candidates passed strict local correctness and
sanitizers but did not establish a useful large-shape performance gain. A
wide/compact dispatcher regressed at R512/R2048; forcing its compact branch
measured 134.047/134.160 us at R512 and 173.648/173.640 us at R2048
(seed/candidate, same full B200, fresh-process cold-L2 CUPTI ABBA). These are
component experiments, not full MoE. They are not selected by the public runner.
Private partition, inactive-rank and resource/grid experiments also remain
research candidates. No performance roof has been established.
