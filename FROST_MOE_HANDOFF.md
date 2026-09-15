# Frost / FlashInfer MoE integration handoff

This draft collects the complete integration for continued development and review.
Yanqin can adjust the design, choose which changes to keep, and decide whether to
split it later. The companion repository carries the other half of the same work;
the individual optimization experiments are not separate PRs.

## Code and validation baseline

The measured implementation uses OSS Frost/CuTeDSL engine **20400** for both GEMMs.
FlashInfer's native kernels provide routing, permutation, and weighted finalization.
This is not a closed-source cuDNN kernel performance result.

The initial draft deliberately preserves the tested source architecture:

- cuDNN Frontend base: `b2712de2da3832e9ad607867d22b2a1429c89cdc`.
- FlashInfer base: `df8b5c1745c51f44b1e8a492b11c3f6cb157cfd2`.
- Primary measurement environment: full B200, 148 SMs, 1000 W;
  CuTe DSL 4.7, CUDA 13.2, cuDNN 9.27, PyTorch 2.13.

Upstream has since changed both integration boundaries. Frontend
[PR #1068](https://github.com/NVIDIA/cudnn-frontend/pull/1068) moves the GEMM
compiler/codegen into architecture families. FlashInfer
[PR #4952](https://github.com/flashinfer-ai/flashinfer/pull/4952) and
[PR #5061](https://github.com/flashinfer-ai/flashinfer/pull/5061) replace
`QuantVariant` with separate quantization axes and runner capability checks.
Porting to these interfaces and validating the resulting branch is explicit
follow-up work. The old-source measurements do not validate that future port.
The drafts must not be merged by choosing the old files over upstream refactors.

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

The prior source snapshots have numerical, capture, actual-route and sanitizer
evidence. The current B200 rerun passed the public packed-cluster regression's
80 cases and its old-source negative control. Formatting of the consolidated
Python changes preserves their ASTs; changed-file hooks pass. Full repository CI
and target-GPU execution of the assembled branches remain pending.

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

1. Decide the upstream architecture/API port and revalidate both actual imports.
2. Finish the pending T1025/T2048/T3072 before/after finalizer comparison with
   identical GEMM tactics and exact-shape sanitizer gates.
3. Evaluate cheaper configuration search against exhaustive selection. A first
   T16 coordinate-search run is promising, but no public tuning policy is changed.
4. Continue kernel work only when a candidate survives strict correctness,
   changed-input capture, actual routes and complete-MoE confirmation.

Private partition, inactive-rank, resource/grid and Kernel Factory experiments
remain research candidates. Recent smaller-buffer/smaller-tile ablations passed
correctness but did not establish a stable gain. They are not silently selected
by the public runner. No performance roof has been established.
