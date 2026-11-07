# FlashInfer Unified GEMM API

*Design Document  ·  v0.1  ·  November 2026  ·  describes the MVP as built (PR-0)*

The GEMM counterpart of the [unified MoE API](flashinfer_moe_api.md). One frozen
configuration object, per-backend capability records, two packs with two lifetimes
(weights once, activations per call), and a stateful layer that selects a backend and
caches the choice per row-count bucket. The vocabulary (`QuantFormat`, `QuantConfig`,
`BackendOptions`, the typed activation values) is the MoE API's, imported unchanged, so a
model author configures a FlashInfer linear layer and a FlashInfer MoE layer the same way.
Everything lives under `flashinfer.gemm` beside the existing functions; nothing old
changes.

## 1. Motivation

FlashInfer exposes dense GEMMs through roughly eighty flat functions (`mm_fp4`,
`bmm_fp8`, `mm_fp8`, `mm_bf16`, `gemm_fp8_nt_groupwise`, `mm_mxfp8`, ...). Each has its
own positional order, its own `backend=` strings, its own weight layout (most take the
weight as a `(K, N)` column-major view, the opposite of `torch.nn.Linear`'s `[N, K]`),
and its own meaning of `backend="auto"`. Nothing in the library answers "which of these
runs on this GPU for this shape?", so every caller builds that table by hand.

The clearest statement of the cost is this repository's own PyTorch example,
`examples/pytorch/flashinfer_modules.py::FlashInferLinear`, a re-implementation of
`nn.Linear` on top of FlashInfer:

| Hand-written in `FlashInferLinear` | Count | What it stands in for |
|---|---|---|
| `GEMMBackend` enum with a `"-<kernel>"` suffix convention | 12 members | the candidate list (`BackendOptions`) |
| `_check_gemm_backend_support(backend, device)` plus a parallel string table the comment says to "keep in sync" | 11 branches | each backend's own capability record |
| shape gates in `__init__` (`K % 128`, `N >= 128`, `K % 32`) that fall back to torch with a warning | 4 blocks | each backend's `check_support()` |
| `_init_*_weights` / `_prepare_*_weights` (quantize, transpose, swizzle, cache) | 9 + 10 methods | `prepare_weights()` filling a weight pack |
| `_forward_*` (quantize activations, call one function, add the bias in torch afterwards) | 12 methods | `prepare_activations()` plus the layer's `__call__` |

The serving frameworks carry the same tables: vLLM's `FlashInfer*LinearKernel` classes
and SGLang's `fp8_utils` / `modelopt_quant` each keep their own backend oracle (which
FlashInfer function to call, on which SM, with which weight layout) because FlashInfer
cannot be asked. Three further defects of the old surface recur in the issue tracker: an
explicit `backend=` sometimes silently ran a different backend; a missing optional
dependency (no CuTe-DSL, old cuDNN) crashed instead of being skipped; and the selection
state lived on a shared function object rather than on the layer that owns the weights.

## 2. Design principles

- **Same shape as the unified MoE API.** `GemmConfig` is to GEMM what `MoEConfig` is to
  MoE; `GemmWeightPack` / `GemmActivationPack` follow the `prepare_for` / `get_view`
  protocol of `MoEWeightPack` / `MoEActivationPack`; `GemmLayer` has `MoELayer`'s
  methods (`winner_backend`, `winner_report()`, `reset_winner()`); the runner adapters
  delegate to the kernels' existing runner objects exactly as `TrtllmFp4RoutedRunner`
  delegates to `core.MoERunner`.
- **Shared vocabulary, imported not copied.** `QuantFormat` (the numeric format the
  tensor-core instruction consumes), `QuantConfig` (weight / activation / output formats),
  `BackendOptions` (an ordered tuple of candidates with `valid_for(arch)`) and the typed
  activation values (`Identity()`, `SwiGLU(...)`, `SiLU()`, ...) come from
  `flashinfer.fused_moe.api`.
- **`flashinfer.gemm.Linear` is a `torch.nn.Linear`.** Same constructor, same `weight
  [N, K]` and `bias [N]` parameters, same state-dict keys, same `y = x @ W.T + b`
  semantics, plus one keyword-only `config`. Checkpoints, FSDP and quantization tooling
  see an ordinary linear layer. A functional twin, `linear(input, weight, bias)`, follows
  `torch.nn.functional.linear`.
- **Selection is separate from viability.** A backend first says whether it *can* run a
  configuration (`check_support`, `check_problem`, the per-call `validate`), and only then
  does the layer choose among the survivors. The one way for a backend to drop out is to
  raise `BackendUnsupported`, a `ValueError` subclass carrying a reason; any other
  exception is a bug and propagates. An explicit single candidate therefore never falls
  back to something else.
- **`auto` is deterministic outside `autotune(True)` and measured inside it.** Without a
  tuning context the layer takes the first eligible candidate in `BackendOptions` order
  and benchmarks nothing; inside `autotune(True)` it times every admitted candidate at
  its best tactic and caches the winner per row-count (`M`) bucket.
- **`supported_backends()` is public from day one.** The MoE design's `find_backends` was
  never built; GEMM ships the discovery call in the MVP because it is the piece the
  frameworks' own oracles exist to replace.

## 3. Config hierarchy

`GemmConfig` is a frozen dataclass: hashable, `repr` round-trips to constructor syntax,
`dataclasses.replace` makes variants, and `keys()` / `__getitem__` let a call site write
`**config`.

| Field | Type | Meaning |
|---|---|---|
| `quant` | `QuantConfig` | `(weight, activation, output)` formats. The `(weight, activation)` pair selects the kernel family; `output` is BF16 or FP16. Default `QuantConfig()` is BF16 x BF16, which no MVP runner serves. |
| `epilogue` | `EpilogueConfig(bias, activation, per_row_alpha)` | Extra work on the output: add a `[N]` bias; a fused activation (`Identity()` only in the MVP); one output scale per row (post-MVP). |
| `backend` | `BackendOptions` | Ordered candidates; also the `auto` order. Default `DEFAULT_GEMM_BACKENDS`. |
| `execution` | `ExecutionConfig(enable_pdl, tune_max_m, batch_invariant)` | Launch knobs that do not change the math: programmatic dependent launch (the next kernel's prologue overlaps this kernel's tail; `None` keeps the backend default), the ceiling of the `M` buckets the layer caches winners for, and a fixed reduction order (post-MVP). |

**Backend configs are capability records.** Each is a frozen dataclass whose
`backend_key` equals the legacy `backend=` string, so
`BackendOptions((CutlassGemmConfig(),))` is the config spelling of `backend="cutlass"`.
Class-level sets answer the cheap questions (`supported_quant_pairs`, `supported_forms`,
`supported(arch)`); the pair-by-architecture gate (`supports_pair_on`) is read from the
legacy functions' own `@backend_requirement` tables through `is_backend_supported`, so
the two surfaces cannot drift; everything finer is a predicate in the runner.

| Config | `backend_key` | Quant pairs (MVP) | Architectures | Weight recipe |
|---|---|---|---|---|
| `CutlassGemmConfig` | `cutlass` | NVFP4 x NVFP4, FP8PerTensor x FP8PerTensor | SM100, 103, 107, 110, 120, 121 | `nvfp4_128x4` / `fp8_per_tensor` |
| `CuteDslGemmConfig` | `cute-dsl` | NVFP4 x NVFP4 (`N % 8 == 0`) | SM100, 103, 107 | `nvfp4_128x4` |
| `TrtllmGemmConfig(use_8x4_sf_layout=False)` | `trtllm` | NVFP4 x NVFP4 (BF16 output only) | SM100, 103, 107 | `nvfp4_128x4_shuffled` |
| `CudnnGemmConfig` | `cudnn` | both pairs (needs `nvidia-cudnn-frontend`) | SM89 through SM121 | `nvfp4_128x4` / `fp8_per_tensor` |
| `CublasGemmConfig` | `cublas` | FP8PerTensor x FP8PerTensor | SM89 through SM121 | `fp8_per_tensor` |

**Weight recipes are shared between backends.** A recipe names the prepared form of the
weight a backend reads: `nvfp4_128x4` is the weight quantized to packed 4-bit values with
block scales in the 128x4 swizzled layout; `nvfp4_128x4_shuffled` is the same data in the
row-shuffled order the trtllm-gen kernels want; `fp8_per_tensor` is a row-major e4m3
weight with one scale. Backends with the same recipe share one prepared view, so the
default candidate set costs one quantized copy of the weight; adding trtllm costs a
second.

```python
DEFAULT_GEMM_BACKENDS = BackendOptions(
    candidates=(CutlassGemmConfig(), CublasGemmConfig(), CudnnGemmConfig())
)
```

Candidates that do not serve the configured pair are skipped silently (cuBLAS for
NVFP4); the order is the deterministic `auto` order.

## 4. Packs

**`GemmWeightPack`** is long-lived: built once after the weights are loaded and read on
every call. It holds the logical shape `(N, K)`, the optional `bias`, and one
backend-native view per candidate (`prepare_for(backend_key, view)` / `get_view(key)`).
A view is a dictionary carrying its `recipe` name, the quantized weight and its scales,
and the derived tensors the legacy kernels take, so the per-call path is dictionary reads
only: for NVFP4 `b`, `b_descale`, `b_global_scale`, the `(K/2, N)` column-major views
`b_t` / `b_descale_t` and an optional pre-folded `alpha`; for FP8 `b`, `b_scale` and the
batch-of-one `(1, K, N)` view `b3`. `view_for_recipe(recipe)` is how two backends end up
sharing one view; `shapes(backend_key)` reports the tensors' shapes for fake
implementations.

**`GemmActivationPack`** is per call and carries the activation in the canonical layout
of its quant pair. Every backend serving a pair consumes the same pack, so a mixed
candidate set runs on one quantization of the input.

| Pair (weight x activation) | `a` | `a_scale` | Also carried |
|---|---|---|---|
| NVFP4 x NVFP4 | `[M, K/2]` packed e2m1 (uint8) | `[M, K/16]` block scales (e4m3, or uint8 bytes) in `sf_layout` (`layout_128x4` default; `layout_8x4` only for `TrtllmGemmConfig(use_8x4_sf_layout=True)`) | `a_global_scale`: the per-tensor quantization scale `(448 * 6) / amax`, needed to form `alpha = 1 / (a_global_scale * b_global_scale)` when the weights were prepared without a static activation scale |
| FP8PerTensor x FP8PerTensor | `[M, K]` e4m3 or e5m2 | one fp32 dequantization scale | |

`prepare_activations(a, quant=..., global_scale=None)` builds the pack from a bf16 / fp16
`[M, K]` tensor; `global_scale=None` derives a dynamic per-tensor scale from `amax(a)`
(one extra reduction per call), a static scale skips it. `form` is derived from the
tensors (`dense` for a 2-D `a`); batched and grouped forms are post-MVP.

## 5. Public API

```python
import torch
from flashinfer.autotuner import autotune
from flashinfer.gemm import (
    BackendUnsupported, CudnnGemmConfig, CutlassGemmConfig, EpilogueConfig,
    GemmConfig, GemmLayer, GemmProblem, Linear, gemm, linear,
    prepare_activations, prepare_weight_pack, supported_backends,
)
from flashinfer.gemm.api import BackendOptions, QuantConfig, QuantFormat

# 1. Config, once. The (weight, activation) pair picks the kernel family.
nvfp4 = GemmConfig(
    quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
    epilogue=EpilogueConfig(bias=True),
    backend=BackendOptions((CutlassGemmConfig(), CudnnGemmConfig())),
)

# 2. Ask before loading: one report per candidate, in auto order, with reasons.
for r in supported_backends(nvfp4, GemmProblem(n=4096, k=7168)):
    print(r.backend_key, r.ok, r.auto_rank, r.reason)

# 3. The module door: a torch.nn.Linear with a config.
proj = Linear(7168, 4096, bias=True, device="cuda", dtype=torch.bfloat16, config=nvfp4)
proj.load_state_dict(torch.nn.Linear(7168, 4096).to("cuda", torch.bfloat16).state_dict())
y = proj(x)                      # before prepare_weights: plain F.linear on the parameters
proj.prepare_weights()           # vLLM's process_weights_after_loading: one view per candidate
y = proj(x)                      # deterministic auto: first eligible candidate, nothing timed
with autotune(True):
    y = proj(x)                  # every admitted candidate timed; winner cached per M bucket
print(proj.winner_backend, proj.winner_report().fused_bias)   # e.g. "cutlass", False

# 4. The functional door, torch.nn.functional.linear's argument order.
y = linear(x, w_bf16, b, config=nvfp4)          # weight prepared on first use, cached by identity
y = linear(x, w_bf16, b)                         # no config: torch.nn.functional.linear

# 5. The pack-level door for callers that quantize activations themselves.
pack = prepare_weight_pack(w_bf16, nvfp4, bias=b)
act = prepare_activations(x, quant=nvfp4.quant)
y = gemm(act, pack, config=nvfp4)

# 6. The stateful layer the two doors compose (not an nn.Module, as MoELayer is not).
layer = GemmLayer(nvfp4, device=torch.device("cuda"))
layer.prepare(pack)
y = layer(act)
try:
    GemmLayer(GemmConfig(quant=nvfp4.quant, backend=BackendOptions((CudnnGemmConfig(),))))
except (RuntimeError, BackendUnsupported) as e:
    print(e)                     # names every rejected backend and why
```

| Entry | Role |
|---|---|
| `Linear(in, out, bias, device, dtype, *, config)` | `torch.nn.Linear` subclass. `prepare_weights(for_backends=None, activation_global_scale=None, sf_layout=...)`, `forward(Tensor | GemmActivationPack)`, `winner_backend`, `winner_report()`, `reset_winner()`, `weight_pack`, `layer`, `prepared`. The `bias` argument overrides `config.epilogue.bias`. |
| `linear(input, weight, bias=None, *, config=None)` | Functional twin. `weight` is a `[N, K]` tensor (prepared once, cached by tensor identity) or a `GemmWeightPack`. The layer adds a pack bias only when `config.epilogue.bias` is set, so pass `EpilogueConfig(bias=True)` with a bias. |
| `gemm(act, weights, *, config, out=None)` | Stateless pack-level entry; the `GemmLayer` is cached per `(config, device)`. |
| `GemmLayer(config, *, device)` | `prepare(weights)`, `__call__(act, weights=None, *, out=None)`, `supported_backends(problem)`, `winner_backend`, `winner_report()`, `reset_winner()`, `runners`, `rejected`. |
| `supported_backends(config, problem=None, device=None)` | One `BackendReport(backend_key, ok, reason, auto_rank, needs_prepare, fused_bias)` per candidate. Loads no kernel module. |
| `prepare_weight_pack(weight, config, *, bias, for_backends, activation_global_scale)` / `prepare_activations(a, *, quant, global_scale, sf_layout)` | The two preparation helpers; backend configs' `prepare_weights` / `prepare_activations` delegate to them. |

## 6. How selection works

1. **Construction** (`GemmLayer.__init__`, also run by `Linear.prepare_weights`). For
   each candidate in `BackendOptions` order: skip it if it does not serve the quant pair
   or has no adapter for it; otherwise build its runner and call `check_support()`,
   which checks the pair-by-architecture gate, the output format, the epilogue and
   execution flags, and the backend's optional dependencies. Rejections are kept as
   strings (`layer.rejected`); if nothing survives the error lists them all. Surviving
   runners `build()` once (kernel module loaded, inner runner constructed).
2. **Per call.** The eligible set is the runners whose `accepts(act, view)` is true: the
   pack has the right shape for the pair, the right scale layout, and a prepared view
   with the right recipe. The cache key is `(M bucket, eligible set, activation dtype,
   scale layout)`; the bucket is `map_to_hybrid_bucket(M, tune_max_m)`, the same
   bucketing the autotuner uses, so one tuning pass covers every smaller `M`. A hit is a
   dictionary read followed by `pack_inputs` and one `forward`; the cache holds 128
   entries, oldest out first.
3. **On a miss** (`_select_winner`). Each eligible runner runs `validate(act, view, out)`:
   the legacy requirement functions on the real tensors, with their `ValueError`
   translated to `BackendUnsupported`. The admitted runners are then chosen from:
   - outside `autotune(True)`, or with one admitted runner: the first in order; its tactic
     comes from `AutoTuner.choose_one`, which is a cache lookup here (tactic `-1` on a
     miss), as for the old names;
   - inside `autotune(True)`: `choose_one` picks each runner's best tactic in its own
     input schema, `bench_gpu_time` (CUDA-graph timed) compares the per-runner winners,
     and the fastest `(runner, tactic)` is cached. The two stages exist because the
     runners' input lists differ, so one `choose_one` over all of them is not possible.
4. **Bias.** After the GEMM the layer adds `weights.bias` when `epilogue.bias` is set and
   the winner does not fuse it; `winner_report().fused_bias` says which happened.

Consequences: `BackendOptions((X(),))` runs X or raises X's reason; a `BackendUnsupported`
from `validate` is a skip, anything else surfaces; tuning never happens inside CUDA graph
capture if the caller warms the layer up first (selection and JIT happen on the warm-up
calls, the captured call is lookup plus launch).

## 7. Compatibility

Nothing old changes. Every existing name, positional order, backend string, `(K, N)`
column-major weight view, scale layout, `ValueError` from validation and
`is_backend_supported` attribute is untouched; the new names are additions under
`flashinfer.gemm`. `fused_moe.api` still owns the shared vocabulary.

The autotuner sees the inner runners, not the adapters: each adapter wraps the runner
object the legacy function already builds, uses the legacy op name (`"fp4_gemm"` for
NVFP4, `"fp8_gemm"` for FP8) and the legacy tuning config, and hands the autotuner the
same input list the legacy function would. Tuning caches, `skip_ops` entries and cache
keys are therefore shared with `mm_fp4` / `bmm_fp8` by construction, not by a mapping
that could rot. The GPU test asserts output parity with the legacy call per backend.

## 8. MVP As-Built Reference

What PR-0 ships, exactly.

**Files.** `flashinfer/gemm/api.py` (configs, backend configs, packs, preparation
helpers), `runners.py` (adapters), `layer.py` (`GemmLayer`, `supported_backends`),
`nn.py` (`Linear`, `linear`, `gemm`), exports in `flashinfer/gemm/__init__.py`;
`tests/cpu/gemm/test_unified_gemm_api.py` (no GPU: config validation, `repr`
round-trip, `**config`, capability sets, the pack protocol, the `nn.Linear` contract) and
`tests/gemm/test_unified_gemm_api.py` (per backend: accuracy against a reference built
from the same quantized operands, parity with the legacy call, bias and its report,
deterministic versus measured selection, an explicit candidate never falling back,
`supported_backends` reasons, CUDA-graph replay, the two functional doors, the module
against `torch.nn.Linear`); two adapters (`unified_nvfp4`, `unified_fp8`) in
`tests/gemm/test_unified_gemm_fuzz.py`, declared in the `mm_fp4` / `bmm_fp8`
conformance groups and fed the same quantized operands, so "unified equals legacy" is
enforced by the fuzzer's auditor; `benchmarks/routines/unified_gemm.py` (one row per
candidate plus the all-candidates layer with its `winner_backend` and host time per
call); `--api unified` in `benchmarks/bench_host_overhead.py` (the layer after `prepare`
next to the legacy call it delegates to, on the same operands); this document; the
`docs/api/gemm.rst` section; and the `unified_fp4` / `unified_fp8` backends of
`examples/pytorch/flashinfer_modules.py`.

**Scope.** Dense `[M, K] x [N, K]^T` only. Two quant pairs. BF16 or FP16 output (trtllm:
BF16 only). `Identity()` epilogue; a bias is applied by the layer after the GEMM and
reported as `fused_bias=False` by every adapter. `enable_pdl` and `tune_max_m` honoured.
Accepted by the dataclasses but rejected by every runner's `check_support()` with a named
reason: any other quant pair, `per_row_alpha`, a non-identity activation,
`batch_invariant`, `QuantConfig.swizzled_scale_factors` / `per_token_scale`, batched or
grouped forms.

**The seven adapters.**

| Backend | Pair | Delegates to | Op name | Notes |
|---|---|---|---|---|
| `cutlass` | NVFP4 | `get_cutlass_fp4_gemm_module(major, minor).cutlass_fp4_gemm_runner()` | `fp4_gemm` | `mm_fp4`'s input list unchanged |
| `cudnn` | NVFP4 | `_cudnn_gemm_fp4_runner(tuning_config)` | `fp4_gemm` | needs `nvidia-cudnn-frontend` and the cuDNN FP4 check |
| `trtllm` | NVFP4 | `get_trtllm_gemm_module().trtllm_fp4_gemm_runner(use_8x4)` | `fp4_gemm` | shuffled weight view; BF16 output; `use_8x4_sf_layout` selects the small-M scale layout and the matching tuning config |
| `cute-dsl` | NVFP4 | `_cute_dsl_gemm_fp4_runner(major, minor, enable_pdl, out_dtype, True)` | `fp4_gemm` | SM100/103/107; `N % 8 == 0`; CuTe-DSL availability and arch checks |
| `cublas` | FP8 | `get_gemm_module().cublas_fp8_gemm_runner()` | `fp8_gemm` | `bmm_fp8` on a batch of one |
| `cudnn` | FP8 | `_cudnn_gemm_fp8_runner()` | `fp8_gemm` | on SM120/121 needs cuDNN `override_shape`, as `bmm_fp8`'s heuristic requires |
| `cutlass` | FP8 | `get_gemm_sm100_module_cutlass_fp8()` / `..._sm120_...` `.cutlass_fp8_gemm_runner()` | `fp8_gemm` | no e5m2 operands |

Per-call admission reuses the legacy checks verbatim: `_check_mm_fp4_problem_size` plus
the per-backend `_*_gemm_fp4_requirement` for NVFP4; `_validate_fp8_output_dtype` plus
the per-backend `_*_bmm_fp8_requirement` for FP8.

**Preparation.** `prepare_weight_pack` quantizes a bf16 / fp16 `[N, K]` weight with
`nvfp4_quantize` (global scale `(448 * 6) / amax`, 128x4 layout, shuffled for trtllm) or
to per-tensor e4m3 (scale `amax / 448`), once per recipe. An optional
`activation_global_scale` folds a calibrated static activation scale into `alpha` at
load time; otherwise `alpha` is formed per call from the pack's `a_global_scale`.
`prepare_activations` quantizes with the same helpers; a static `global_scale` skips the
per-call `amax`.

**Not in PR-0** (listed in the roadmap for it, deferred to keep the PR additive): a
separate `prepare.py` (the helpers live in `api.py`), `TraceTemplate`s beyond the
`@flashinfer_api` decorator on the entry points, and the move of the shared vocabulary to
a neutral module.

## 9. Decision Log

Decisions taken while building the MVP, so reviewers see the why.

- **Vocabulary imported from `fused_moe.api`, not moved.** The design proposed moving
  `QuantFormat` / `QuantConfig` to `flashinfer/quantization/config.py` and
  `BackendOptions` to its own module. The MVP imports them unchanged so the PR touches no
  MoE file and no MoE test. The neutral-module move is left to the review thread (open
  question 1).
- **Adapters delegate to the existing runner factories** (table in §8). The MoE MVP's one
  blocker was an adapter that had re-implemented its inner call and rotted after a merge;
  these adapters only translate `(activation pack, weight view)` into the legacy input
  list, and the tuning config is read per instance, not off the class.
- **The layer never benchmarks outside tuning mode.** The MoE layer's open carry-over
  ("gate `_select_winner` behind the tuner's tuning mode") is done here from day one:
  deterministic first-eligible outside `autotune(True)`, two-stage measurement inside.
- **Default candidate order is `cutlass`, `cublas`, `cudnn`.** It mirrors what the legacy
  functions pick untuned: CUTLASS first for FP8 on Blackwell (`bmm_fp8`), cuBLAS where
  CUTLASS is absent (SM89 / SM90), cuDNN last. `trtllm` is not a default because it needs
  a second, shuffled copy of the weights; `cute-dsl` is not one because its autotune
  compiles every tactic. Both are one explicit candidate away.
- **Bias is unfused in the MVP.** No adapter declares a fused bias; `GemmLayer` adds
  `weights.bias` after the GEMM and reports `fused_bias=False`, so a caller can see the
  extra kernel rather than discover it in a profile. Kernel-side bias support lands per
  backend later and flips the flag.
- **Architecture gates are read from the legacy tables.** `supports_pair_on(pair, arch)`
  calls `mm_fp4.is_backend_supported` / `bmm_fp8.is_backend_supported`, so the capability
  record and the old `@backend_requirement` decorators are one source of truth.
- **`Linear` runs `F.linear` before `prepare_weights()`** so a freshly built or partially
  loaded model works, as `torch.nn.Linear` does; the module computes the quantized path
  only after the weights are final.
- **`Linear.forward` accepts a `GemmActivationPack`** (open question 3 answered yes) for
  callers that quantize activations themselves or share one quantized input across
  layers.
- **The form is taken from the operands**, not from a `GemmConfig.form` field, matching
  `torch` (open question 5 answered for the MVP).
- **`linear()` caches the prepared pack by weight-tensor identity** (a `Parameter` is a
  stable object) and the `GemmLayer` per `(config, device)`, so a functional caller pays
  preparation once per weight without owning a layer object.
- **FP8 runs through the batched kernels on a batch of one**, exactly as `mm_fp8`'s
  per-tensor path does, so the adapters and cache keys are `bmm_fp8`'s.

## 10. Roadmap

The PR sequence after PR-0, from `mvp-roadmap.md`; each row is independently
shippable, with the MoE PR it mirrors.

| # | PR | Adds | Gate | MoE analogue |
|---|---|---|---|---|
| 1 | BF16 and W4A16 | `(BF16, BF16)` over the `mm_bf16` backends incl. `BatchInvariantBf16Config` (`ExecutionConfig.batch_invariant`); `(NVFP4, BF16)` and `(MXFP4, BF16)` over `mm_bf16_fp4`; `Linear` without a config becomes the bf16 fast path | host overhead vs `mm_bf16(cute-dsl)` and `plan_mm_bf16`; `tinygemm` / `tgv` small-M crossover reproduced by `auto_rank` | #4328, #8361 |
| 2 | MXFP8 and MXFP4 | `(MXFP8, MXFP8)`, `(MXFP4, MXFP4)`; `QuantConfig(swizzled_scale_factors)` for the flat MXFP8 layout | fuzzer MXFP8 conformance group; the shared `mxfp8_gemm` tuner key split with a release note | #4159, #5230 |
| 3 | DeepSeek block-FP8 | `(DeepSeekFp8, DeepSeekFp8)`: the `gemm_fp8_nt_groupwise` backends and `fp8_blockscale_gemm_sm90`; public blockwise-128 quantizer | DeepGEMM parity on the frameworks' shapes | #4026 |
| 4 | Docs and official tag | `@flashinfer_api` on every entry, `TraceTemplate`s, a generated capability matrix with a CPU test that it matches the registry, top-level `flashinfer.Linear` | matrix test green; `/api/gemm` shows the unified section first | #5200, #4903 |
| 5 | `torch.library` registration | `linear` and the per-backend forwards as custom ops with fakes built from `GemmWeightPack.shapes()` | `torch.compile` of a `Linear` stack with no graph breaks | b12x precedent |
| 6 | Port the first old names | `mm_fp4`, `bmm_fp8`, `mm_fp8` become adapters over the controller (`backend="x"` -> single-candidate `BackendOptions`); op names kept | fuzzer unchanged; surface diff empty; host overhead not worse | #4376, #5663 |
| 7 | Batched form | `GemmProblem.form="batched"`: `bmm_bf16`, `bmm_fp8`, `bmm_mxfp8` as adapters; new `bmm_fp4` and batched block-FP8 | MLA absorption shapes run on the unified path | #4104 |
| 8 | Grouped form | `GroupLayout` (offsets / row ids / masked); `grouped_gemm`; the grouped duplicates as adapters; grouped autotuning on bucketed total `M` | DeepGEMM semantics parity | #3892, #5453 |
| 9 | Remaining backends | `cutile`, `cake`, `prims_ts`, `b12x`, `deepgemm`, `tinygemm`, `tgv` configs; low-rank (SVDQuant / LoRA) as an epilogue later | capability matrix complete for dense | #4646, #5289, #6626 |
| 10 | Example and docs switch | `FlashInferLinear` becomes a thin subclass of `flashinfer.gemm.Linear` or is deleted; README tables generated from the matrix | the WAN pipeline runs on `Linear` for every backend it supports today | |
| 11 | Framework PRs | vLLM `FlashInfer*LinearKernel` classes collapse to one per pair using `prepare_weights` + `linear`; SGLang ops call `linear` with `EpilogueConfig(bias=True)`; both feature-detect `flashinfer.gemm.linear` | merged and released on both sides before any deprecation | `moe_ep` integration |
| 12 | Deprecations | markers on the names with zero framework hits; load-bearing old names only after PR-11 ships; backend-string alias table kept; positional adapters for `mm_fp4` / `bmm_fp8` kept indefinitely | downstream grep CI green at removal time | deprecation survey |

PR-5 and PR-6 can swap; PR-7 and PR-8 need PR-6; PR-11 depends on PR-4 and PR-6.

## 11. Open questions

For the PR-0 review thread; the MVP's current position is in parentheses.

1. Where `QuantFormat`, `QuantConfig` and `BackendOptions` live once two APIs share them
   (imported from `fused_moe.api`; proposal `flashinfer/quantization/config.py` and
   `flashinfer/backend_options.py`, both re-exported from `fused_moe.api`).
2. Low-latency cute-dsl: a tactic of `CuteDslGemmConfig` or its own config (its own, since
   the weight view differs; not in the MVP either way).
3. `torch.library` registration of the tensor-level `linear`: op or fake only (op, in
   PR-5, after the tracing maintainers weigh in; the pack layer stays Python).
4. Whether `Linear.forward` should keep accepting a `GemmActivationPack` (yes in the MVP).
5. The default candidate order per pair for `auto` (`cutlass`, `cublas`, `cudnn`, written
   as `DEFAULT_GEMM_BACKENDS` and printed by `supported_backends`).
6. The host-overhead budget for the layer's hot path (the matching old name's figure as
   the PR-0 ceiling, `plan_mm_bf16`'s as the PR-1 target; measured with
   `benchmarks/bench_host_overhead.py`, not yet wired for the unified entry).
7. Whether `GemmConfig` gets a `form` field or the form is always taken from the operands
   (from the operands in the MVP).
8. One weight view per candidate multiplies weight memory; whether a "prepare once,
   retarget later" conversion between views is wanted (`for_backends` and
   `supported_backends` before load are the MVP's answer).
9. Layer adoption: the MoE layer has no framework callers after four months, so the
   functional doors ship in PR-0; whether to promote the module tier depends on the
   frameworks' `*LinearKernel` classes moving to it.
10. The generated capability matrix shows only the coarse class-level sets; whether
    reviewers want the finer predicates exposed (an `explain()` on each config) rather
    than left to the fuzzer.
