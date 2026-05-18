# FlashInfer Unified MoE API

*Design Document  ·  v0.1  ·  March 2026*

## 1. Motivation

FlashInfer currently exposes MoE functionality through a family of flat, positional-argument functions:

```
trtllm_fp4_block_scale_moe(routing_logits, routing_bias, hidden_states,
    hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm1_bias,
    gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights, ...)
# 30 positional arguments. Backend selected by function name.
```

Problems this creates:

- Users must know which backend (cutlass vs trtllm_fp4 vs trtllm_fp8) to call for their hardware and quant config
- 30-argument flat signatures are error-prone — type mismatches at param 18 are silent until C++ segfaults
- Three near-identical functions for fp4 / fp8 block / fp8 per-tensor diverge over time
- No autotuning — optimal backend varies by batch size, hardware, and routing config
- No component benchmarking — impossible to attribute latency to routing vs gemm vs finalize
- No repro tracing — user-filed issues require manual reconstruction of call site

## 2. Design Principles

- Config is pure data — frozen dataclasses, no behavior, serializable via repr()
- Frontend owns the schema — backends are consumers
- PyTorch-familiar surface — follows nn.Module / F.functional pattern
- TVM FFI end-to-end — structured objects cross the C++ boundary, no positional arg counting
- Immutable by default — use dataclasses.replace() for variants, never mutate

### Example Overview

```
# --- Define config once ---
config = MoEConfig(
    routing=RoutingConfig(
        num_experts=256,
        top_k=8,
        method=RoutingMethodType.DeepSeekV3,
    ),
    quant=QuantConfig(QuantDtype.FP4, QuantGranularity.BlockScale),
    experts=ExpertConfig(intermediate_size=2048, local_num_experts=32),
    backends=[TrtllmFp4Config(extra_backend_params...), CutlassConfig(extra_backend_params...)],
)
# --- Find possible backends ---
backends = MoELayer.find_backends(**config)
# this contains {"trtllm_fp4":TrtllmFp4Config(), "cutlass_fp4":CutlassConfig()}
# or {"trtllm_fp4":"unsupported reason...", "cutlass_fp4":CutlassConfig()}
# more modification to the backends' parameters could be done here
backends=["trtllm_fp4":TrtllmFp4Config(extra_backend_params...),"cutlass_fp4":CutlassConfig(extra_backend_params...)]
# --- Prepare Inputs Data ---
weight_pack = MoEWeightPack()
# the data is possibly obtained through helper functions then added here
weight_pack.prepare_for("trtllm_fp4", trtllm_weights) weight_pack.prepare_for("cutlass_fp4", cutlass_weights)
act_pack = MoEActivationPack(
    hidden_states_q=cute_dsl_data["x"],
    hidden_states_scale=x_sf,
    selected_experts=cute_dsl_data["token_selected_experts"],
    final_scales=cute_dsl_data["token_final_scales"],
)
tensors = (act_pack, weight_pack)
# --- Eager (heuristic backend) ---
output = moe_layer(tensors, **config, backends=backends)  # optional backends selection
# --- Autotuned eager ---
with autotune(True):
    for tensors in calibration_data:
        output = moe_layer(tensors, **config)
# --- Production layer (amortized, cached) ---
layer = MoELayer(**config)
layer = layer.get_tuned_layer(tensors)  # ensures selection has been done
output = layer(tensors)
# --- Benchmark ---
layer.benchmark(Gemm1Tensors)   # isolate gemm1
layer.benchmark_all()           # full breakdown
# --- Variant via immutable replace ---
fp8_config = dataclasses.replace(config, quant=QuantConfig(QuantDtype.FP8))
fp8_layer  = MoELayer(**fp8_config)
# --- Repro from issue log ---
repro = MoERepro.from_file("user_issue.log")
repro.run()
repro.benchmark_all()
```

## 3. Config Hierarchy

All configs are frozen dataclasses registered with TVM's object system. The hierarchy:

| Config | Owns |
| --- | --- |
| RoutingConfig | num_experts, top_k, routing method, grouping params, scaling factor |
| QuantConfig | dtype (fp4/fp8/bf16), granularity (per-tensor/per-token/block) |
| ExpertConfig | intermediate_size, local sharding params |
| ActivationConfig | activation type (swiglu/geglu/relu2/identity) |
| BackendOptions | ordered candidate set via \| operator |
| ExecutionConfig | do_finalize, enable_pdl, tune_max_num_tokens, output tensor |
| MoEConfig | assembles all above; supports \*\*unpacking protocol |

### 3.1 RoutingConfig

```
@tvm.register_object('flashinfer.RoutingConfig')
@dataclass(frozen=True)
class RoutingConfig:
    num_experts:           int
    top_k:                 int
    method:                RoutingMethodType      = RoutingMethodType.Default
    n_group:               Optional[int]          = None
    topk_group:            Optional[int]          = None
    routed_scaling_factor: Optional[float]        = None
```

### 3.2 BackendOptions

Individual backend configs provided in an ordered list. The autotuner or heuristic selects among valid candidates at runtime.

```
# Single backend
backends = [TrtllmFp4Config()]
# Multiple candidates — autotuner or heuristic picks best
backends = [TrtllmFp4Config(), TrtllmFp8BlockConfig(), CutlassConfig()]
# | is associative, returns BackendOptions
# CutlassConfig is always the universal fallback
```

Each backend config declares its own preconditions:

```
class TrtllmFp4Config:
    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch >= 90  # Hopper+
class CutlassConfig:
    @classmethod
    def supported(cls, arch: int) -> bool:
        return True  # universal fallback
```

### 3.3 MoEConfig — \*\*unpack protocol

MoEConfig implements keys() and __getitem__ so it can be unpacked directly with \*\*. This allows the same config object to be passed to both the eager function and MoELayer.

```
config = MoEConfig(
    routing=RoutingConfig(num_experts=256, top_k=8, method=RoutingMethodType.DeepSeekV3),
    quant=QuantConfig(QuantDtype.FP4, QuantGranularity.BlockScale),
    experts=ExpertConfig(intermediate_size=2048, local_num_experts=32),
    backends=[TrtllmFp4Config(), CutlassConfig()],
)
# Unpack into any call accepting these kwargs
output = moe_layer(tensors, **config)
layer  = MoELayer(**config)
# Immutable variant
fp8_config = dataclasses.replace(config, quant=QuantConfig(QuantDtype.FP8))
```

## 4. Public API

### 4.1 Eager function

Stateless. No autotuning. Backend selected by heuristic priority table. Equivalent to the current flat functions but with structured config.

```
@flashinfer_api
def moe_layer(tensors: MoETensors, *, routing, quant, experts,
              activation=..., backends=..., execution=...) -> Tensor:
    ...
# Usage
output = moe_layer(tensors, **config)
```

### 4.2 MoELayer — stateful, autotuned

Holds workspace and backend selection cache. On first call, selects and caches the best valid backend for the observed tensor shapes and arch. Subsequent calls are zero-overhead dispatch.

```
layer  = MoELayer(**config)
output = layer(tensors)        # first call: selects backend
output = layer(tensors)        # subsequent: cached dispatch
# Component benchmark
layer.benchmark(Gemm1Tensors)  # isolate gemm1 latency
layer.benchmark_all()          # full breakdown by component
```

### 4.3 autotune context manager

Within the with block, every moe_layer call profiles all valid BackendOptions candidates. On exit, the best backend is cached keyed by (device, shape, config hash).

```
with autotune(True):
    for tensors in calibration_data:
        output = moe_layer(tensors, **config)
# After the block, MoELayer uses the measured best
layer = MoELayer(**config)
```

Without the context manager, moe_layer uses the heuristic DEFAULT_PRIORITY table — no profiling overhead, reasonable defaults.

## 5. TVM FFI — Structured Boundary Crossing

The structured config crosses the Python/C++ boundary via TVM's object system. Reflection fires once at the crossing; all downstream C++ access is direct struct member access.

```
// C++ side — mirrors Python dataclass
class RoutingConfigNode : public Object {
 public:
  int num_experts;
  int top_k;
  int method;
  Optional<int> n_group;
  // ...
  static RoutingConfigNode FromObject(ObjectRef obj);  // reflection once
  TVM_DECLARE_FINAL_OBJECT_INFO(RoutingConfigNode, Object);
};
TVM_REGISTER_GLOBAL('flashinfer.moe_layer_fp4')
.set_body_typed([](MoEConfig config, ...) {
    auto routing = RoutingConfigNode::FromObject(config->routing);
    dispatch_fp4(routing.num_experts, routing.top_k, ...);  // native access
});
```

With CUDA graph capture: FFI crossing happens at capture time. Graph replay is pure device-side — zero host overhead on the hot path.

## 6. Repro Tracing

### 6.1 @flashinfer_api decorator

The existing decorator gains one additional responsibility: emit a single-line repro log on every MoE call.

```
logger.debug(
    f'REPRO fn={fn.__name__} '
    f'config={repr(config)} '
    f'shapes={tensor_shapes} '
    f'device={torch.cuda.get_device_name()} '
    f'version={flashinfer.__version__}'
)
```

Because `repr(config)` emits valid Python constructor syntax (all enums use qualified repr), the log line is directly eval-able. No JSON parsing, no schema versioning needed.

### 6.2 MoERepro

Takes a repro log line, reconstructs the config, synthesizes tensors of the correct shape, and reproduces the call.

```
repro = MoERepro.from_file('issue_42.log')
# Reproduce
output = repro.run()
# Component breakdown
repro.benchmark_all()
# RoutingConfig     0.12ms
# Gemm1Config       1.43ms
# ActivationConfig  0.08ms
# Gemm2Config       1.39ms
# FinalizeConfig    0.11ms
# Total             3.13ms
# Isolate backend to narrow regression
repro.isolate_backend(CutlassConfig())
repro.isolate_backend(TrtllmFp4Config())
```

### 6.3 MoEConfig.from_repr — safe eval

eval() with a restricted namespace. Only config constructors in scope — no arbitrary execution risk.

```
@classmethod
def from_repr(cls, s: str) -> MoEConfig:
    ns = {
        'MoEConfig': cls, 'RoutingConfig': RoutingConfig,
        'RoutingMethodType': RoutingMethodType, ...,
        '__builtins__': {},  # no arbitrary execution
    }
    return eval(s, ns)
```

## 7. File Layout

```
flashinfer/
  moe_layer/
    __init__.py       # public exports
    config.py         # all dataclasses and enums
    layer.py          # MoELayer
    functional.py     # moe_layer eager function
    autotune.py       # autotune context manager
    backends/
      __init__.py     # BACKEND_REGISTRY, DEFAULT_PRIORITY
      trtllm_fp4.py   # TrtllmFp4Config + adapter
      trtllm_fp8.py   # Fp8Block + Fp8PerTensor + adapters
      cutlass.py      # CutlassConfig + adapter
    repro.py          # MoERepro
    tensors.py        # MoETensors, Gemm1Tensors, Gemm2Tensors
```

## 8. Migration Path

The flat functions are not removed. They become internal adapters called only from BACKEND_REGISTRY. Public surface is the new API.

| Phase | Action | User Impact |
| --- | --- | --- |
| 1. Config layer | Add dataclasses + MoEConfig | None — flat API unchanged |
| 2. Adapters | Wrap flat functions in adapters | None — flat API unchanged |
| 3. moe_layer eager | Publish new unified function | Opt-in to new API |
| 4. MoELayer + autotune | Add stateful layer + context manager | Opt-in |
| 5. Repro tracing | Extend @flashinfer_api | Automatic for all users |
| 6. Deprecate flat APIs | Add deprecation warnings | Warning on old call sites |

Each phase is independently valuable and independently shippable.

## 9. Review Comments

These comments were transcribed from the DOCX review metadata. Anchors refer to the text range the comment was attached to in Word.

### C0 — Minseok Lee US

**Anchor:** `FlashInfer Unified MoE API`

> Would it be helpful to describe first (1) what FlashInfer should be responsible for and (2) what frameworks should be responsible for, e.g., weight preprocessing (and what FlashInfer provides to support/supplement it)

### C1 — Daniel Stokes NZ

**Anchor:** `Design Principles`

> I think one thing we are missing is backend introspection. Frameworks should be able to ask questions like "what are the supported backends for WideEP", "what data-types can I use for my model" etc. to help with their own heuristics. Maybe this doesn't fall under this scope, but it is something we likely need to expose longer-term

### C2 — Alex Yang US

**Anchor:** `Design Principles`

> agree. i'll add it to the doc before resolving. thanks

### C3 — Alex Yang US

**Anchor:** `Design Principles`

> please see "Find possible backends" in "Example Overview" if that workflow sound reasonable

### C4 — Albert Cheng (Engrg-Hardware 1) US

**Anchor:** `Config is pure data — frozen dataclasses, no behavior, serializable via repr()`

> frozen dataclasses is good. i just have one concern on repr() serialization, its output is not a versioned format, if a constructor signature changes between releases, old repro logs cannot be parseable. Have you considered to_dict() and from_dict() with an explicit schema version field?

### C5 — Alex Yang US

**Anchor:** `Config is pure data — frozen dataclasses, no behavior, serializable via repr()`

> what could go wrong if we don't worry about parsing old repro logs? e.g. limiting the usage to same version repro. as a "repro" the version should align anyway

### C6 — Siyuan Fu US

**Anchor:** `Frontend owns the schema — backends are consumers`

> A tricky part is scale factor swizzling pattern and the interleaving required by trtllm-gen MoE. Many users aren't clear why these are needed and Flashinfer needs to emphasize these and provide useful tools. For example, MoEConfig can have a method called weight_preprocessing()2 total reactionsMinseok Lee US reacted with ➕ at 2026-05-12 20:00 PMCyrus Chang CN reacted with ➕ at 2026-05-13 01:13 AM

### C7 — Alex Yang US

**Anchor:** `Frontend owns the schema — backends are consumers`

> i missed this detail in the doc. but the plan is to have prepare functions like this https://github.com/flashinfer-ai/flashinfer/pull/3093/changes#diff-0995b259d16c6c37f02b6cc5d8825f9147ddf095a1c9b29846bbdee96ce2ab96R2278-R2280 weight_pack = MoEWeightPack()weight_pack.prepare_for("cute_dsl_nvfp4", cute_dsl_view)weight_pack.prepare_for("trtllm_fp4_routed", trtllm_view)and it's organized into something like a static helper function in TrtllmFp4Config(). (wrapping current standalone functions)

### C8 — Julien Debache CH

**Anchor:** `Frontend owns the schema — backends are consumers`

> Could you elaborate on this? I'm not sure I get it.

### C9 — Julien Debache CH

**Anchor:** `Immutable by default — use dataclasses.replace() for variants, never mutate`

> Nit: even though I'm all for this, I think it mixes up design and implementation details

### C10 — Julien Debache CH

**Anchor:** `# --- Define config once ---`

> Is it fair to summarize the idea of this design as:- User knows the dimensions of their MoE layer- User knows how it is quantized- User knows how routing should happen- User knows batch sizes that they are interested inThis information is mapped to a config type that FlashInfer defines, FlashInfer takes that in and returns something the user can call on their inputs.The `backends` are is just here for optionally overriding the selection? I would almost leave it out of the config type, and pass it as an additional argument.Could we have another function which returns the compatible backends for a config?3 total reactionsPavani Majety US reacted with ➕ at 2026-05-12 16:02 PMSiyuan Fu US reacted with ➕ at 2026-05-12 16:33 PMMinseok Lee US reacted with ➕ at 2026-05-12 20:01 PM

### C11 — Alex Yang US

**Anchor:** `# --- Define config once ---`

> that makes a lot of sense to me! thx for the suggestion

### C12 — Alex Yang US

**Anchor:** `# --- Define config once ---`

> >Could we have another function which returns the compatible backends for a config?that sounds super reasonable too

### C13 — Alex Yang US

**Anchor:** `# --- Define config once ---`

> pls see updated "Example Overview"

### C14 — Siyuan Fu US

**Anchor:** `quant=QuantConfig(QuantDtype.FP4, QuantGranularity.BlockScale)`

> BTW we also have "MXFP4", so the block size (16 or 32) needs to be exposed

### C15 — Pavani Majety US

**Anchor:** `experts=ExpertConfig(intermediate_size=2048, local_num_experts=32),`

> hidden_size as well?

### C16 — Pavani Majety US

**Anchor:** `backends=[TrtllmFp4Config(extra_backend_params...), CutlassConfig(extra_backend_params...)]`

> Very often community developers are not aware of which backends are supported for a given routing + quant config. Agree with Julien here that there should be an interface where users can query for a list of backends given quant config, hardware architecture. This can also extend to a benchmark option where we give the users an option to also see which backend performs best for a set of shapes of interest.

### C17 — Alex Yang US

**Anchor:** `backends=[TrtllmFp4Config(extra_backend_params...), CutlassConfig(extra_backend_params...)]`

> added a "Find possible backends" step in the example

### C18 — Daniel Stokes NZ

**Anchor:** `"unsupported reason..."`

> I wonder if this format would be confusing. I would suggest a `find_backends()` that returns only the valid ones for the config. And then a separate `request_backend(backend_name, config, extra_backend_params)` that returns `Config | Error`

### C19 — Daniel Stokes NZ

**Anchor:** `"unsupported reason..."`

> This way if frameworks dont care about customizing backends they can use `find_backends` and just directly use the result. But if the framework is aware of the different backends and wants to customize them it could instead explicitly call request_backend() with the customised parameters

### C20 — Minseok Lee US

**Anchor:** `output = layer(tensors)`

> When is the backend decided in this workflow? During the first call of layer(tensors)?

### C21 — Alex Yang US

**Anchor:** `output = layer(tensors)`

> added a line above as a possible solution to make it more explicit and clear

### C22 — Minseok Lee US

**Anchor:** `layer.benchmark(Gemm1Tensors) # isolate gemm1`

> Should it be layer's responsibility to do benchmark? I thought it should be something like benchmark(layer, tensors).

### C23 — Alex Yang US

**Anchor:** `layer.benchmark(Gemm1Tensors) # isolate gemm1`

> yeah your suggestion sounds better

### C24 — Dimitrios Bariamis CH

**Anchor:** `routing method`

> Since the interface is being changed, it would be a good chance to rework how the routing method is passed to Flashinfer. I would find it a lot better to not expose the internal enumeration of possible routing methods, but instead receive the routing configuration (scoring function, topK, renorm, topK groups, etc.) and select the correct routing internally. A `supported()` classmethod can be added to allow queries.The drawback of the current enumeration is that it encodes multiple pieces of information. This leads to a large number of methods and to the inference frameworks implementing some version of the above `supported()` function to find out if a model can use Flashinfer MoE, which is error prone and not easily extendable.

### C25 — Minseok Lee US

**Anchor:** `routing method`

> @siyuanf@nvidia.com @jiahanc@nvidia.com What do you think?

### C26 — Siyuan Fu US

**Anchor:** `routing method`

> Having a decoupled routing configuration looks reasonable. However, in the CPP side, we probably need to preserve the enum list to align with Trtllm. Additionally, the user should be able to directly provide the topk_ids, in order to support custom score functions or DeepEP.

### C27 — Cyrus Chang CN

**Anchor:** `routing method`

> @siyuanf@nvidia.com 's point is valid. I agree have a dataclass instead of just enumeration, but we should keep the enumeration in cpp to align with trtllm

### C28 — Pavani Majety US

**Anchor:** `dtype (fp4/fp8/bf16),`

> nit: it should reflect both activation and weight dtypes

### C29 — Trevor Morris US

**Anchor:** `backends = [TrtllmFp4Config(), TrtllmFp8BlockConfig(), CutlassConfig()]`

> I'm not sure how realistic this is since usually each backend requires different weight processing at model load time or different checkpoints entirely.

### C30 — Julien Debache CH

**Anchor:** `backends = [TrtllmFp4Config(), TrtllmFp8BlockConfig(), CutlassConfig()]`

> Agree with Trevor: these backends are different operations with different inputs and potentially different outputs. I'm not sure there's a way of hiding that from the caller.

### C31 — Pavani Majety US

**Anchor:** `backends = [TrtllmFp4Config(), TrtllmFp8BlockConfig(), CutlassConfig()]`

> Agree, is the framework still defining and calling weight processing methods like swizzling, shuffling etc?

### C32 — Pavani Majety US

**Anchor:** `backends = [TrtllmFp4Config(), TrtllmFp8BlockConfig(), CutlassConfig()]`

> >I'm not sure there's a way of hiding that from the caller.It would be nice to have something like `moe_wrapper.process_weights_after_loading(layer.weight, ... , Backend=<>`1 total reactionTrevor Morris US reacted with ➕ at 2026-05-12 19:27 PM

### C33 — Alex Yang US

**Anchor:** `backends = [TrtllmFp4Config(), TrtllmFp8BlockConfig(), CutlassConfig()]`

> sorry it was not meant to be hidden. the doc got a bit old but i sync'd back the idea from the WIP PR. please see "Example Overview"

### C34 — Julien Debache CH

**Anchor:** `layer = MoELayer(**config)`

> Do we need to expose layers without implementations? Could we directly pass tensors to the constructor, or have a utility function that returns the "tuned" layer directly? This way no need to track "has this layer been tuned already".

### C35 — Alex Yang US

**Anchor:** `layer = MoELayer(**config)`

> i see your point

### C36 — Daniel Stokes NZ

**Anchor:** `layer = MoELayer(**config)`

> Yes I like the idea of an explicit "compile" api. Maybe we could do something similar to CuTe-DSL, though I don't have enough CuTe-DSL experience to know if that is a good design or not (or if it matches how FI/frameworks work)

## 10. Codex Reviews

Review lens: the current WIP is the MVP for PR #3093, not the full long-range design. The MVP target is NVFP4 only, CuteDSL MoE plus TRTLLM-Gen MoE only, pre-routed inputs only, cross-backend autotuning, CUDA graph tests, and benchmarks.

### MVP Gaps In Current WIP

**CR1 — Config/test API drift.** The current implementation exposes `QuantVariant` and explicit `BackendOptions(candidates=...)`, while `tests/moe/test_moe_api.py` still imports `QuantDtype`, `QuantGranularity`, and `Fp8Variant`, and uses `TrtllmFp4Config() | CutlassConfig()`. This is not a request to widen the MVP; it is a request to make the MVP API and its CPU config tests agree.

**CR2 — Backend preparation is proven but not yet first-class.** `MoEActivationPack` and `MoEWeightPack.prepare_for(...)` give the MVP a clean canonical input shape, but the actual TRTLLM NVFP4 preparation logic still lives in benchmark/test helpers. The MVP goal says per-backend prepare funcs should handle differences, so the shared CuteDSL/TRTLLM NVFP4 prepare entrypoints should move into the implementation surface.

**CR3 — TRTLLM expert offset is not wired through pack input conversion.** `ExpertConfig.local_expert_offset` is stored in config and passed to the TRTLLM kernel, but `TrtllmFp4RoutedRunner.pack_inputs(...)` defaults `local_expert_offset=0`, and `MoELayer` calls it without passing the config offset. For expert-parallel pre-routed inputs, the packed top-k IDs can be wrong when the local shard does not start at expert 0.

**CR4 — Winner caching is single-shape unless the caller rebuilds the layer.** `MoELayer` caches a single `_winner` and `_winner_tactic` after the first call. If the same layer instance is reused for a different token count within `tune_max_num_tokens`, it will not re-run cross-backend selection. The MVP can either make this explicitly one-layer-per-shape/bucket or make the cache shape/bucket-aware.

**CR5 — Token ceiling is only partially threaded into tuning.** `ExecutionConfig.tune_max_num_tokens` gates `MoELayer.__call__`, but the TRTLLM runner's dynamic tuning buckets are still built with an 8192-token ceiling. The MVP benchmark script includes a 16384-token sweep, so the runner tuning config and benchmark expectations need to agree.

**CR6 — MVP scope should fail fast.** The config objects describe more than the MVP can execute, while `MoELayer` silently skips non-MVP backend configs and the current TRTLLM runner hard-codes NVFP4/SwiGLU assumptions internally. For the MVP, unsupported quant variants, activation variants, backend choices, and non-pre-routed call shapes should produce explicit errors.

### Implementation Innovations To Fold Back Into The Doc

**CR7 — Activation and weight packs are sharper than the original `MoETensors` sketch.** The WIP splits per-call activation/routing data (`MoEActivationPack`) from long-lived backend-native weight materializations (`MoEWeightPack`). That directly addresses reviewer concerns about backend-specific preprocessing without pretending all backends share one weight layout.

**CR8 — Cross-backend autotune is a two-stage decision.** Each runner first uses the existing `AutoTuner` to select its best tactic, then `MoELayer` measures each runner at its winning tactic and caches the fastest backend. The original doc says "autotune across backends" but does not spell out this tactic-then-backend selection structure.

**CR9 — Minimal runtime introspection already exists.** `winner_backend` and `reset_winner()` are practical MVP observability hooks. They are smaller than a full backend discovery API but useful for benchmarks, debugging, and validating winner selection.

**CR10 — The WIP tests are more concrete than the design doc.** `tests/moe/test_unified_moe_api.py` checks the selected `MoELayer` output against a shared BF16 reference, checks each candidate backend against the same reference, verifies autotune visits all candidates, and checks CUDA graph replay against eager output.

**CR11 — The benchmark path reports candidates, not only the selected layer.** `unified_nvfp4_moe` builds both backend-native weight views on one `MoEWeightPack`, triggers winner selection, then emits one result row per candidate backend with winner metadata. That is useful MVP evidence and should be reflected in the doc.

## 11. Task Tracking

This tracker is scoped to the PR #3093 MVP, not the full long-range API design. Its purpose is continuity: what already landed in the current branch, what still needs tightening before review, and what should stay out of scope.

### Landed In Current Branch

| Status | Task | Continuity notes |
| --- | --- | --- |
| [x] | Add the MVP config and pack surface. | `MoEConfig`, component configs, `MoEActivationPack`, and `MoEWeightPack` exist in `flashinfer/fused_moe/api.py`. |
| [x] | Add two MVP backend runners. | `CuteDslNvfp4Runner` and `TrtllmFp4RoutedRunner` exist as `TunableRunner` adapters in `flashinfer/fused_moe/runners.py`. |
| [x] | Add cross-backend `MoELayer` dispatch. | `MoELayer` builds compatible runners, selects a winner, caches it, and exposes `winner_backend` plus `reset_winner()`. |
| [x] | Preserve legacy flat MoE APIs. | `flashinfer/fused_moe/__init__.py` exports the new MVP API while keeping the existing flat APIs available. |
| [x] | Add unified NVFP4 benchmark path. | `unified_nvfp4_moe` is wired into the benchmark registry and has a runnable sweep script. |
| [x] | Add MVP accuracy, autotune, and CUDA graph tests. | `tests/moe/test_unified_moe_api.py` covers shared-reference accuracy, candidate visitation, and graph replay. |

### Remaining MVP Follow-Ups

| Status | Task | Review refs |
| --- | --- | --- |
| [ ] | Align the MVP config API and config tests: either update tests/docs to `QuantVariant` plus explicit `BackendOptions(candidates=...)`, or add small compatibility helpers if the `QuantDtype` plus pipe-operator spelling is intentionally kept. | CR1 |
| [ ] | Promote per-backend NVFP4 preparation into first-class MVP helpers for CuteDSL and TRTLLM, then remove duplicated TRTLLM preparation logic from tests and benchmarks. | CR2, CR7 |
| [ ] | Wire `ExpertConfig.local_expert_offset` into TRTLLM `pack_inputs(...)` and add an EP-offset test for the pre-routed cross-backend path. | CR3 |
| [ ] | Decide the layer reuse contract: document/enforce one `MoELayer` per shape or make winner/tactic caching key off the relevant shape or tuning bucket. | CR4 |
| [ ] | Thread `ExecutionConfig.tune_max_num_tokens` into runner tuning configs and validate the 16384-token benchmark sweep. | CR5 |
| [ ] | Add explicit MVP validation for NVFP4, pre-routed activation packs, supported activation assumptions, and the CuteDSL/TRTLLM backend set. | CR6 |
| [ ] | Update the MVP section/examples in this design doc to describe `MoEActivationPack`, `MoEWeightPack`, backend-native views, two-stage autotune, and winner introspection. | CR7-CR9 |
| [ ] | Make benchmark output/validation capture `winner_backend`, per-candidate latency, and the expected-winner checks described by the benchmark script. | CR10-CR11 |

### Explicit Non-Goals For This MVP

| Status | Task | Notes |
| --- | --- | --- |
| [ ] | Keep FP8, MXFP4, and additional backend families out of this PR. | Track as follow-up runners after the NVFP4 path is solid. |
| [ ] | Keep general routing unification out of this PR. | This MVP assumes pre-routed inputs through `MoEActivationPack`. |
| [ ] | Keep broader public API ergonomics separate from MVP correctness. | Backend discovery, pipe-operator sugar, repro replay, and a full eager functional API can evolve after the MVP contract is stable. |
