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

> **Not shipped in the MVP (post-MVP).** This eval-based deserializer belongs to
> the long-range repro design only. The implementation intentionally omits
> `from_repr` — eval-based deserialization is a security smell (review C4-C5/C39)
> and is deferred with the rest of the repro tooling (see "Post-MVP Carryover").
> `repr(config)` still round-trips for logging; only the parser is deferred.

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

### C0 — Reviewer 1

**Anchor:** `FlashInfer Unified MoE API`

> Would it be helpful to describe first (1) what FlashInfer should be responsible for and (2) what frameworks should be responsible for, e.g., weight preprocessing (and what FlashInfer provides to support/supplement it)

### C1 — Reviewer 2

**Anchor:** `Design Principles`

> I think one thing we are missing is backend introspection. Frameworks should be able to ask questions like "what are the supported backends for WideEP", "what data-types can I use for my model" etc. to help with their own heuristics. Maybe this doesn't fall under this scope, but it is something we likely need to expose longer-term

### C2 — Reviewer 3

**Anchor:** `Design Principles`

> agree. i'll add it to the doc before resolving. thanks

### C3 — Reviewer 3

**Anchor:** `Design Principles`

> please see "Find possible backends" in "Example Overview" if that workflow sound reasonable

### C4 — Reviewer 4

**Anchor:** `Config is pure data — frozen dataclasses, no behavior, serializable via repr()`

> frozen dataclasses is good. i just have one concern on repr() serialization, its output is not a versioned format, if a constructor signature changes between releases, old repro logs cannot be parseable. Have you considered to_dict() and from_dict() with an explicit schema version field?

### C5 — Reviewer 3

**Anchor:** `Config is pure data — frozen dataclasses, no behavior, serializable via repr()`

> what could go wrong if we don't worry about parsing old repro logs? e.g. limiting the usage to same version repro. as a "repro" the version should align anyway

### C6 — Reviewer 5

**Anchor:** `Frontend owns the schema — backends are consumers`

> A tricky part is scale factor swizzling pattern and the interleaving required by trtllm-gen MoE. Many users aren't clear why these are needed and Flashinfer needs to emphasize these and provide useful tools. For example, MoEConfig can have a method called weight_preprocessing()2 total reactionsReviewer 1 reacted with ➕ at 2026-05-12 20:00 PMReviewer 6 reacted with ➕ at 2026-05-13 01:13 AM

### C7 — Reviewer 3

**Anchor:** `Frontend owns the schema — backends are consumers`

> i missed this detail in the doc. but the plan is to have prepare functions like this https://github.com/flashinfer-ai/flashinfer/pull/3093/changes#diff-0995b259d16c6c37f02b6cc5d8825f9147ddf095a1c9b29846bbdee96ce2ab96R2278-R2280 weight_pack = MoEWeightPack()weight_pack.prepare_for("cute_dsl_nvfp4", cute_dsl_view)weight_pack.prepare_for("trtllm_fp4_routed", trtllm_view)and it's organized into something like a static helper function in TrtllmFp4Config(). (wrapping current standalone functions)

### C8 — Reviewer 7

**Anchor:** `Frontend owns the schema — backends are consumers`

> Could you elaborate on this? I'm not sure I get it.

### C9 — Reviewer 7

**Anchor:** `Immutable by default — use dataclasses.replace() for variants, never mutate`

> Nit: even though I'm all for this, I think it mixes up design and implementation details

### C10 — Reviewer 7

**Anchor:** `# --- Define config once ---`

> Is it fair to summarize the idea of this design as:- User knows the dimensions of their MoE layer- User knows how it is quantized- User knows how routing should happen- User knows batch sizes that they are interested inThis information is mapped to a config type that FlashInfer defines, FlashInfer takes that in and returns something the user can call on their inputs.The `backends` are is just here for optionally overriding the selection? I would almost leave it out of the config type, and pass it as an additional argument.Could we have another function which returns the compatible backends for a config?3 total reactionsReviewer 8 reacted with ➕ at 2026-05-12 16:02 PMReviewer 5 reacted with ➕ at 2026-05-12 16:33 PMReviewer 1 reacted with ➕ at 2026-05-12 20:01 PM

### C11 — Reviewer 3

**Anchor:** `# --- Define config once ---`

> that makes a lot of sense to me! thx for the suggestion

### C12 — Reviewer 3

**Anchor:** `# --- Define config once ---`

> >Could we have another function which returns the compatible backends for a config?that sounds super reasonable too

### C13 — Reviewer 3

**Anchor:** `# --- Define config once ---`

> pls see updated "Example Overview"

### C14 — Reviewer 5

**Anchor:** `quant=QuantConfig(QuantDtype.FP4, QuantGranularity.BlockScale)`

> BTW we also have "MXFP4", so the block size (16 or 32) needs to be exposed

### C15 — Reviewer 8

**Anchor:** `experts=ExpertConfig(intermediate_size=2048, local_num_experts=32),`

> hidden_size as well?

### C16 — Reviewer 8

**Anchor:** `backends=[TrtllmFp4Config(extra_backend_params...), CutlassConfig(extra_backend_params...)]`

> Very often community developers are not aware of which backends are supported for a given routing + quant config. Agree with Reviewer 7 here that there should be an interface where users can query for a list of backends given quant config, hardware architecture. This can also extend to a benchmark option where we give the users an option to also see which backend performs best for a set of shapes of interest.

### C17 — Reviewer 3

**Anchor:** `backends=[TrtllmFp4Config(extra_backend_params...), CutlassConfig(extra_backend_params...)]`

> added a "Find possible backends" step in the example

### C18 — Reviewer 2

**Anchor:** `"unsupported reason..."`

> I wonder if this format would be confusing. I would suggest a `find_backends()` that returns only the valid ones for the config. And then a separate `request_backend(backend_name, config, extra_backend_params)` that returns `Config | Error`

### C19 — Reviewer 2

**Anchor:** `"unsupported reason..."`

> This way if frameworks dont care about customizing backends they can use `find_backends` and just directly use the result. But if the framework is aware of the different backends and wants to customize them it could instead explicitly call request_backend() with the customised parameters

### C20 — Reviewer 1

**Anchor:** `output = layer(tensors)`

> When is the backend decided in this workflow? During the first call of layer(tensors)?

### C21 — Reviewer 3

**Anchor:** `output = layer(tensors)`

> added a line above as a possible solution to make it more explicit and clear

### C22 — Reviewer 1

**Anchor:** `layer.benchmark(Gemm1Tensors) # isolate gemm1`

> Should it be layer's responsibility to do benchmark? I thought it should be something like benchmark(layer, tensors).

### C23 — Reviewer 3

**Anchor:** `layer.benchmark(Gemm1Tensors) # isolate gemm1`

> yeah your suggestion sounds better

### C24 — Reviewer 9

**Anchor:** `routing method`

> Since the interface is being changed, it would be a good chance to rework how the routing method is passed to Flashinfer. I would find it a lot better to not expose the internal enumeration of possible routing methods, but instead receive the routing configuration (scoring function, topK, renorm, topK groups, etc.) and select the correct routing internally. A `supported()` classmethod can be added to allow queries.The drawback of the current enumeration is that it encodes multiple pieces of information. This leads to a large number of methods and to the inference frameworks implementing some version of the above `supported()` function to find out if a model can use Flashinfer MoE, which is error prone and not easily extendable.

### C25 — Reviewer 1

**Anchor:** `routing method`

> @siyuanf@nvidia.com @jiahanc@nvidia.com What do you think?

### C26 — Reviewer 5

**Anchor:** `routing method`

> Having a decoupled routing configuration looks reasonable. However, in the CPP side, we probably need to preserve the enum list to align with Trtllm. Additionally, the user should be able to directly provide the topk_ids, in order to support custom score functions or DeepEP.

### C27 — Reviewer 6

**Anchor:** `routing method`

> @siyuanf@nvidia.com 's point is valid. I agree have a dataclass instead of just enumeration, but we should keep the enumeration in cpp to align with trtllm

### C28 — Reviewer 8

**Anchor:** `dtype (fp4/fp8/bf16),`

> nit: it should reflect both activation and weight dtypes

### C29 — Reviewer 10

**Anchor:** `backends = [TrtllmFp4Config(), TrtllmFp8BlockConfig(), CutlassConfig()]`

> I'm not sure how realistic this is since usually each backend requires different weight processing at model load time or different checkpoints entirely.

### C30 — Reviewer 7

**Anchor:** `backends = [TrtllmFp4Config(), TrtllmFp8BlockConfig(), CutlassConfig()]`

> Agree with Reviewer 10: these backends are different operations with different inputs and potentially different outputs. I'm not sure there's a way of hiding that from the caller.

### C31 — Reviewer 8

**Anchor:** `backends = [TrtllmFp4Config(), TrtllmFp8BlockConfig(), CutlassConfig()]`

> Agree, is the framework still defining and calling weight processing methods like swizzling, shuffling etc?

### C32 — Reviewer 8

**Anchor:** `backends = [TrtllmFp4Config(), TrtllmFp8BlockConfig(), CutlassConfig()]`

> >I'm not sure there's a way of hiding that from the caller.It would be nice to have something like `moe_wrapper.process_weights_after_loading(layer.weight, ... , Backend=<>`1 total reactionReviewer 10 reacted with ➕ at 2026-05-12 19:27 PM

### C33 — Reviewer 3

**Anchor:** `backends = [TrtllmFp4Config(), TrtllmFp8BlockConfig(), CutlassConfig()]`

> sorry it was not meant to be hidden. the doc got a bit old but i sync'd back the idea from the WIP PR. please see "Example Overview"

### C34 — Reviewer 7

**Anchor:** `layer = MoELayer(**config)`

> Do we need to expose layers without implementations? Could we directly pass tensors to the constructor, or have a utility function that returns the "tuned" layer directly? This way no need to track "has this layer been tuned already".

### C35 — Reviewer 3

**Anchor:** `layer = MoELayer(**config)`

> i see your point

### C36 — Reviewer 2

**Anchor:** `layer = MoELayer(**config)`

> Yes I like the idea of an explicit "compile" api. Maybe we could do something similar to CuTe-DSL, though I don't have enough CuTe-DSL experience to know if that is a good design or not (or if it matches how FI/frameworks work)

### C37 — Reviewer 11

**Anchor:** `Config`

> when I instantiate a config, do I have guarantee that it is supported? or is that still "trial and error"?

### C38 — Reviewer 11

**Anchor:** `Config`

> nice I see `find_backend` in the later section

### C39 — Reviewer 4

**Anchor:** `Config is pure data — frozen dataclasses, no behavior, serializable via repr()`

> My concern was mainly like bug reports where the fix lands on a newer version, but agreed that same version repro covers the majority case.

### C40 — Reviewer 2

**Anchor:** `routing method`

> As Reviewer 5 mentioned, an important thing I would consider is ease of supporting custom routing methods. e.g. when DeepSeek released it was a lot of work to bring up the new routing method because it was originally fused. If frameworks are dependent on flashinfer to supply their routing methods this limits their flexibility in enabling new/experimental/research models.I am not sure what the best approach is, since we still want to support all the various fusions, but maybe a "RoutingMethodType.Custom" that accepts a python function would allow users to inject custom scoring functions at the appropriate place, without having to reimplement the routing themselves.Im partial to the API from TRT-LLM where we have a BaseMoeRoutingMethod interface the user can override, but I understand that probably doesnt play well with trtllm backend with its fusions

### C41 — Reviewer 11

**Anchor:** `RoutingConfig`

> does the interface consider routed MoE as well?

### C42 — Reviewer 5

**Anchor:** `method: RoutingMethodType = RoutingMethodType.Default`

> Kindly remind that we now have this file to store enum: flashinfer/tllm_enums.py. It's desirable to put new enums there

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

> **Status (2026-05-31): MVP scope complete.** Every "Today's MVP Cut" item and every "Remaining MVP Follow-Up" (CR1–CR11) is done and validated on a B200 (SM100): the unified-MoE GPU tests **9/9** (layer + per-backend accuracy vs bf16, autotune visits both candidates, CUDA-graph replay), the CPU config tests **97/97** (CPU config + fail-fast validation), and the `unified_nvfp4_moe` benchmark sweep (128→16384 tokens) with `--refcheck` passing for both backends. Only **Post-MVP Carryover** and **Explicit Non-Goals** remain open by design.

> **Status (2026-06-01): PR #3093 review-comment pass.** Addressed the open GitHub review threads without widening MVP scope. Bot threads: the redundant `pack_inputs(local_expert_offset=...)` was already fixed (reads from config — CR3); removed the eval-based `MoEConfig.from_repr` (eval is a security smell and repro/serialization is documented Post-MVP — see C4-C5/C39); broke the import-time `layer.py → testing.utils` coupling by lazy-importing `bench_gpu_time` on the autotune path only. Human threads (benchmark merge damage): `benchmarks/routines/moe.py` was reconciled against `main` so its diff is now **purely additive** (the `unified_nvfp4_moe` routine) — the accidentally-dropped `bgmv_moe` routine, `warn_if_pdl_unsupported`, and all `enable_pdl=args.enable_pdl` call-sites are restored; the `unified_nvfp4_moe` arch table was trimmed to the SM100 entries it actually supports (`10.0`/`10.3`); the sweep script moved to `benchmarks/bench_unified_moe.sh`; and the two MoE test files were merged + renamed to `tests/moe/test_unified_moe.py` (96 CPU config tests after dropping the `from_repr` round-trip + 9 SM100 GPU tests). Re-validated on B200: benchmark/library imports clean, **96 CPU config tests pass**, full file collects 105 tests. (Nightly/dashboard tracking of the sweep script is left as a separate CI-infra follow-up.)

### Release Gates (do before this ships in a tagged release)

The branch can merge to `main` early for team review and may land in a nightly/early release. To avoid implying any stability/observability commitment on a still-evolving API surface, the MVP **intentionally ships the new unified MoE APIs without the `@flashinfer_api` decorator** (no logging / repro-trace / stability contract). This is deliberate — not an oversight — and reserves the right to change `MoEConfig` / `MoELayer` / `MoEActivationPack` / `MoEWeightPack` / the runners / `prepare_weights` freely pre-release.

| Status | Gate | Notes |
| --- | --- | --- |
| [ ] | Add `@flashinfer_api` (+ a `TraceTemplate` per the `CLAUDE.md` "Trace Template Checklist") to the public unified MoE APIs **at release time**, not before. | The decorator carries logging/repro + an implied stability contract; §4.1/§6 describe the intended end-state. The decorated legacy MoE functions (`trtllm_*_moe`, `cutlass_fused_moe`) already ship in v0.6.12 and are untouched here. |

### Landed In Current Branch

| Status | Task | Continuity notes |
| --- | --- | --- |
| [x] | Add the MVP config and pack surface. | `MoEConfig`, component configs, `MoEActivationPack`, and `MoEWeightPack` exist in `flashinfer/fused_moe/api.py`. |
| [x] | Add two MVP backend runners. | `CuteDslNvfp4Runner` and `TrtllmFp4RoutedRunner` exist as `TunableRunner` adapters in `flashinfer/fused_moe/runners.py`. |
| [x] | Add cross-backend `MoELayer` dispatch. | `MoELayer` builds compatible runners, selects a winner, caches it, and exposes `winner_backend` plus `reset_winner()`. |
| [x] | Preserve legacy flat MoE APIs. | `flashinfer/fused_moe/__init__.py` exports the new MVP API while keeping the existing flat APIs available. |
| [x] | Add unified NVFP4 benchmark path. | `unified_nvfp4_moe` is wired into the benchmark registry; the sweep script is `benchmarks/bench_unified_moe.sh`. |
| [x] | Add MVP accuracy, autotune, and CUDA graph tests. | `tests/moe/test_unified_moe.py` covers CPU config tests plus shared-reference accuracy, candidate visitation, and graph replay. |

### MVP As-Built Reference

The aspirational API in §2–§4 (eager `moe_layer(...)`, `MoETensors`, `find_backends`, pipe-operator backends) describes the long-range design. What actually shipped for the PR #3093 MVP is narrower and pack-based; this section is the authoritative end-to-end description of the as-built surface (CR7–CR9).

```python
import torch
from flashinfer.fused_moe import (
    MoEConfig, RoutingConfig, QuantConfig, QuantVariant, ExpertConfig,
    ActivationConfig, ExecutionConfig, MoELayer,
    MoEActivationPack, MoEWeightPack, CuteDslConfig, TrtllmFp4Config,
)
from flashinfer.fused_moe.api import BackendOptions
from flashinfer.autotuner import autotune

# 1. Config — single-knob QuantVariant; explicit candidate set.
config = MoEConfig(
    routing=RoutingConfig(num_experts=32, top_k=2),
    quant=QuantConfig(variant=QuantVariant.NVFP4),       # MVP: NVFP4 only
    experts=ExpertConfig(intermediate_size=512, local_num_experts=32),
    activation=ActivationConfig(),                       # MVP: Swiglu only
    backend=BackendOptions(candidates=(CuteDslConfig(), TrtllmFp4Config())),
    execution=ExecutionConfig(tune_max_num_tokens=8192),
)

# 2. Long-lived weights: one MoEWeightPack holds a backend-native view per
#    backend, built from canonical bf16 weights by first-class prepare helpers.
weights = MoEWeightPack()
weights.prepare_for("cute_dsl_nvfp4",
    CuteDslConfig.prepare_weights(w1_bf16, w2_bf16, num_local_experts=32,
                                  hidden_size=1024, intermediate_size=512))
weights.prepare_for("trtllm_fp4_routed",
    TrtllmFp4Config.prepare_weights(w1_bf16, w2_bf16, num_local_experts=32,
                                    hidden_size=1024, intermediate_size=512))

# 3. Per-call activations: pre-routed (selected_experts/final_scales supplied).
act = MoEActivationPack(
    hidden_states_q=x_q,             # [M, H//2] uint8 packed NVFP4
    hidden_states_scale=x_sf,        # [M, H//16] float8_e4m3fn (or uint8 bytes)
    selected_experts=topk_ids,       # [M, top_k] int32
    final_scales=topk_weights,       # [M, top_k] float32
)

# 4. Dispatch. First call per token-bucket runs cross-backend selection.
layer = MoELayer(config)
with autotune(True):
    out = layer(act, weights)        # tunes + selects winner for this bucket
print(layer.winner_backend)          # e.g. "cute_dsl_nvfp4"
out = layer(act, weights)            # subsequent calls: cached winner dispatch
```

Key mechanisms (and where they live):

- **Two packs, two lifetimes.** `MoEWeightPack` holds long-lived, backend-native weight materializations keyed by `backend_key` (`prepare_for` / `get_view`); `MoEActivationPack` carries per-call pre-routed activations. This is the concrete answer to reviewers' "backends need different weight preprocessing" concern (C29–C32): each backend stores its own view, none is hidden from the caller.
- **First-class prep.** `TrtllmFp4Config.prepare_weights(...)` / `CuteDslConfig.prepare_weights(...)` (backed by `flashinfer/fused_moe/prepare.py`) turn canonical bf16 weights into the native views (C6/C7).
- **Two-stage cross-backend autotune** (`MoELayer._select_winner`, runners' delegation): for each candidate, the `AutoTuner.choose_one` picks the best *within-backend tactic* (each backend tuned in its own native input schema), then `bench_gpu_time` compares the candidates at their winning tactics and the fastest backend is dispatched. A single `choose_one` over both runners is not possible because their input schemas differ — hence the explicit two stages.
- **Winner caching is per token-bucket** (`map_to_hybrid_bucket`): reusing one `MoELayer` across token counts re-selects per bucket; `winner_backend` reports the most-recent choice and `reset_winner()` clears the cache.
- **Fail-fast scope** (`MoELayer._validate_mvp_scope`): non-NVFP4 quant or non-Swiglu activation raises `NotImplementedError` at construction.
- **Runners delegate** to canonical inner runners (`CuteDslFusedMoENvfp4Runner` / `core.MoERunner`); the unified adapters only translate Packs ⇄ the inner runner's native tensor list.

### Today's MVP Cut

This is the May 27, 2026 working slice (executed May 31, 2026). It should improve the current PR without expanding it beyond NVFP4, CuteDSL plus TRTLLM-Gen, pre-routed inputs, cross-backend autotune, CUDA graph tests, and benchmark evidence.

| Priority | Status | Task | Why it fits today |
| --- | --- | --- | --- |
| P0 | [x] | Keep local `moe_api`, `origin/moe_api`, and PR #3093 on the same head commit before editing. | Avoids iterating on a stale branch or accidentally reviewing a different PR state. |
| P1 | [x] | Align the CPU config tests with the actual MVP API surface. | Fast, no GPU required, and removes obvious API/test drift before deeper validation. |
| P1 | [x] | Wire `local_expert_offset` into TRTLLM routed top-k packing and add a focused pre-routed EP-offset test. | Small correctness fix inside MVP scope; prevents wrong packed expert IDs for nonzero local shard offsets. |
| P1 | [x] | Add fail-fast MVP validation for quant variant, activation assumptions, backend set, and pre-routed-only inputs. | Answers the config-support review concern without building a full backend discovery API today. |
| P2 | [x] | Decide and document the `MoELayer` reuse contract for this PR: one layer per tuned shape/bucket, or shape-aware winner cache. | Avoids a hidden behavioral trap while keeping implementation scope explicit. |
| P2 | [x] | Thread `tune_max_num_tokens` into runner tuning configs enough to make the current benchmark sweep honest. | Needed if the 16K-token row remains in the MVP evidence path. |
| P2 | [x] | Run the unified NVFP4 benchmark sweep on the intended GPU and record winner/per-candidate latency evidence. | Converts the branch from "implemented" to "PR argument is supported by measurements." |

> **Mid-cut discovery (blocker, resolved).** While validating on B200, both MVP runner adapters turned out to have *never* run against the post-`main`-merge `core.py`: `TrtllmFp4RoutedRunner` targeted a raw-`moe_op` API the module factory does not expose, and `CuteDslNvfp4Runner` read its `tuning_config` off the class and under-populated its input list. The runners are listed as "Landed" above, but the layer / autotune / CUDA-graph tests could not actually execute. Fixing this was a prerequisite for the P2 evidence items and is recorded in the Decision Log below; it stayed within MVP scope (no new backends, dtypes, or routing modes).

#### Decision Log — May 31, 2026 working slice

Decisions made while executing the cut above, recorded so reviewers see the *why*, not just the diff.

- **P0 — branch alignment verified.** Local `moe_api`, `origin/moe_api`, and PR #3093 head all resolve to the same commit (`1f74494b` at the time of writing), so the cut edits the live PR state. The most recent prior change on the branch (`fix(fused_moe): align TrtllmFp4RoutedRunner with hybrid token buckets`) is already reflected in `runners.py`.
- **P1 / CR1 — single-knob `QuantVariant`, explicit `BackendOptions(candidates=...)`.** The CPU config tests (`tests/moe/test_moe_api.py`) were rewritten to match the implementation rather than the other way around. Rationale: the implementation deliberately collapsed the older `QuantDtype` + `QuantGranularity` + `Fp8Variant` triple into one `QuantVariant` enum (`NVFP4`, `MxFp8`, `DeepSeekFp8`, `FP8PerTensor`, `MxInt4`, `MXFP4`, `BF16`). One knob is simpler for the MVP and still distinguishes the cases reviewers flagged (C14 MXFP4 block size, C28 activation+weight dtype) because each becomes a distinct enum member. The `|` pipe-operator sugar and a richer multi-field `QuantConfig` are listed as Explicit Non-Goals for this MVP, so the canonical spelling is the explicit `BackendOptions(candidates=(...))` already used by the GPU test (`tests/moe/test_unified_moe_api.py`) and the benchmark. Verified: 84 CPU tests pass in the B200 container.
- **Runner rework (blocker fix) — delegate to the canonical inner runners.** `TrtllmFp4RoutedRunner` now wraps `core.MoERunner` (newly exported from `get_trtllm_moe_sm100_module()`), mirroring how `CuteDslNvfp4Runner` wraps `CuteDslFusedMoENvfp4Runner`. `pack_inputs` builds the `MoEInputs` list (with an allocated output buffer and the kernel-required `topk_weights` placeholder for `PackedPrecomputed`) plus a static weight/config kwargs dict; `forward`/`get_valid_tactics` delegate to the inner runner, which owns the one fragile raw-op launch. This keeps the unified adapters thin and resistant to future `core.py` signature drift. The CuteDSL adapter additionally appends the optional `moe_output` buffer (index 11) its tuning_config declares as dynamic. The nvfp4 activation scale is viewed to `float8_e4m3fn` (the canonical Pack may carry raw `uint8` bytes; trtllm-gen accepts the *linear* scale layout, so no per-call swizzle is needed). Validated on B200: all 9 `tests/moe/test_unified_moe_api.py` pass.
- **P1 / CR3 — offset read from config, not a dead parameter.** `pack_inputs` no longer takes a `local_expert_offset` argument (no caller ever passed it, so it silently defaulted to 0); it reads `ExpertConfig.local_expert_offset` off the runner's own config. A focused SM100 test (`TestTrtllmRoutedPackingContract`) decodes the packed ids for offsets 0/32/96 and asserts GLOBAL ids are packed, with `local_expert_offset` passed to the kernel separately.
- **P1 / CR6 — fail fast at construction.** `MoELayer._validate_mvp_scope` raises `NotImplementedError` for any non-`NVFP4` quant variant or non-`Swiglu` activation, and the "no usable backend" error now names the MVP-supported backend set. Pre-routed-only is structural (the layer consumes `MoEActivationPack`, which carries `selected_experts`/`final_scales`). Covered by CPU tests in `TestMoELayerMVPValidation`.
- **P2 / CR4 — bucket-keyed winner cache.** The cross-backend winner can legitimately differ across token-count buckets (the per-tactic autotuner is already bucket-aware), so `MoELayer` now caches `(runner, tactic)` keyed by `map_to_hybrid_bucket(num_tokens, tune_max_num_tokens)` instead of a single `_winner`. Reusing one layer across token counts re-selects correctly per bucket; `winner_backend` reports the most-recent call's choice and `reset_winner()` clears all buckets. This removes the silent stale-winner trap without forcing one-layer-per-shape on callers.
- **P2 / CR5 — token ceiling threaded for free.** Because the reworked TRTLLM runner builds its tuning config via `MoERunner._make_tuning_config(tune_max_num_tokens=ExecutionConfig.tune_max_num_tokens)`, the num_tokens buckets now honor the configured ceiling, so a 16384-token sweep tunes against 16384-token buckets rather than a hard-coded 8192.
- **CR2/CR7 — first-class NVFP4 weight prep.** Added `flashinfer/fused_moe/prepare.py` (`prepare_trtllm_fp4_weights`, `prepare_cute_dsl_nvfp4_weights`) and exposed them as `TrtllmFp4Config.prepare_weights` / `CuteDslConfig.prepare_weights` so callers do `weight_pack.prepare_for("trtllm_fp4_routed", TrtllmFp4Config.prepare_weights(w1, w2, ...))` (the workflow reviewer C7 described). Deleted the byte-identical `_build_trtllm_view` / `_build_trtllm_nvfp4_view` copies from the test and benchmark; the benchmark now builds *both* backend views from the same bf16 weights via the helpers. The `prepare_weights` staticmethods lazily import the heavy prep so `api.py` stays pure-data. Validated: test suite 9/9 (TRTLLM view), benchmark runs both candidates (CuteDSL view). Activation-scale prep remains a smaller follow-up.
- **P2 — benchmark evidence (B200, SM100).** `benchmarks/flashinfer_benchmark.py --routine unified_nvfp4_moe` referenced an undefined `_create_cute_dsl_moe_test_data`; pointed it at the canonical `create_moe_tensors` (the same helper the GPU test uses). Sweep at `hidden=1024, intermediate=512, num_experts=32, top_k=2`, NVFP4 + Swiglu, CUDA-graph timing (CUPTI unavailable → CUDA events). One row per candidate; `*` marks the cross-backend winner. The 16384-token row exercises the CR5 ceiling.

  | num_tokens | winner | cute_dsl_nvfp4 (ms / TFLOP·s⁻¹) | trtllm_fp4_routed (ms / TFLOP·s⁻¹) |
  | --- | --- | --- | --- |
  | 128   | cute_dsl_nvfp4 | 0.016 / 49.5   | 0.017 / 48.3  |
  | 512   | cute_dsl_nvfp4 | 0.018 / 182.1  | 0.020 / 163.2 |
  | 2048  | cute_dsl_nvfp4 | 0.023 / 554.8  | 0.033 / 392.2 |
  | 8192  | cute_dsl_nvfp4 | 0.040 / 1274.9 | 0.067 / 766.3 |
  | 16384 | cute_dsl_nvfp4 | 0.067 / 1546.8 | 0.108 / 954.3 |

  CuteDSL wins across the swept range for this geometry; the cross-backend selection, per-candidate latency, and winner introspection (`winner_backend`) all flow through to CSV/stdout as the benchmark intends (CR10/CR11).
- **CR10/CR11 — `--refcheck` for the unified routine.** Because both backend views now derive from the same bf16 weights, the benchmark can verify each candidate against one `compute_reference_moe_fp4` bf16 reference. With `--refcheck`, each row prints `[REFCHECK] unified/<backend>: PASS/FAIL`; a failure errors unless `--allow_output_mismatch`. Validated on B200: both `cute_dsl_nvfp4` and `trtllm_fp4_routed` report 100% within tolerance (atol≈0.13).

#### Cross-backend autotune value + a selection bug (DeepSeek-V3, B200)

The whole point of `MoELayer` is to pick the faster backend *per shape*. A DeepSeek-V3 sweep (hidden=7168, intermediate=2048, num_experts=256, top_k=8, NVFP4+Swiglu; EP=1) makes the case — and surfaced a real selection bug.

All numbers below were **regenerated end-to-end (2026-06-01, B200) via `benchmarks/flashinfer_benchmark.py --routine unified_nvfp4_moe`** — the perf-tracking driver, not a side harness. One invocation per shape emits both per-candidate `[PERF]` rows *and* the `MoELayer` winner, so a single sweep yields all three comparisons (`benchmarks/bench_unified_moe.sh` drives it). The per-candidate latency *is* that backend's within-backend-autotuned time.

| num_tokens (EP=1) | cute_dsl_nvfp4 (ms) | trtllm_fp4_routed (ms) | MoELayer winner | regime |
| --- | --- | --- | --- | --- |
| 1     | 0.046 | **0.042** | cute_dsl_nvfp4 † | noise tie |
| 16    | 0.428 | **0.365** | trtllm_fp4_routed | low-latency |
| 64    | 0.938 | **0.822** | trtllm_fp4_routed | low-latency |
| 128   | 1.072 | **0.901** | trtllm_fp4_routed | low-latency |
| 256   | 1.115 | **0.936** | trtllm_fp4_routed | low-latency |
| 512   | 1.134 | **0.942** | trtllm_fp4_routed | low-latency |
| 1024  | **1.199** | 1.306 | cute_dsl_nvfp4 | throughput |
| 2048  | **1.226** | 1.311 | cute_dsl_nvfp4 | throughput |
| 4096  | **1.624** | 1.711 | cute_dsl_nvfp4 | throughput |
| 16384 | **3.742** | 4.613 | cute_dsl_nvfp4 | throughput |

The winner **flips with a sharp crossover between 512 and 1024 tokens**: TRTLLM-gen wins the entire low-latency regime (16–512 tokens, ~15–17% faster) — consistent with its known small-batch specialization (cf. PR #2529) — while CuteDSL wins large-batch throughput (≥1024, up to ~19% faster at 16384). Neither single-backend strategy dominates, so cross-backend autotune is ≥ either backend alone and strictly faster wherever the other clearly loses. Both backends pass `--refcheck` at this geometry (t=1024: 100% within tol vs the bf16 reference), so the comparison is between two numerically-correct implementations — not one that is fast because it is wrong.

† **t=1 is a noise tie.** At ~40-µs kernels the two backends are within ~4 µs (~9%); the selector's internal timing picked CuteDSL on this run while the benchmark's independent per-candidate re-timing shows TRTLLM marginally faster, and the pick flips run-to-run (an earlier cross-check picked TRTLLM at t=1). The ~4 µs cost of the "wrong" choice at the smallest shape is negligible. This is the measurement noise floor, distinct from the systematic bug below.

**Selection bug found & fixed.** An earlier sweep mis-picked the *slower* backend even at well-separated shapes (e.g. EP1 t=1024 picked TRTLLM though CuteDSL was clearly faster). Cause: `MoELayer._select_winner` timed candidates with a no-CUDA-graph 10-iter `bench_gpu_time`, so at low token counts launch/Python overhead dominated the median. Fix: time the selection with CUDA graph + 30 iters (matching deployment and the benchmark's own per-candidate timing). After the fix the winner tracks the faster backend at every well-separated shape; only genuine near-ties (t=1, and t≈4096 where the gap is a few %) remain coin-flips, as expected. (Requires a warmed-up layer — the autotune pass — not a cold graph capture.)

**The optimal backend depends on geometry *and* batch — which is the whole motivation.** The small-geometry sweep above (hidden=1024, intermediate=512, 32 experts) has CuteDSL winning at *every* token count, whereas the DeepSeek-V3 geometry (hidden=7168, 256 experts) hands the entire ≤512-token regime to TRTLLM-gen. So there is no fixed "use backend X" rule even per-batch-size — the right choice moves with the full problem shape. A per-shape cross-backend selector is therefore the only way to stay on the frontier without hand-tuning a routing table, which is exactly what `MoELayer` automates.

**Wide-EP: local-only MVP proxy (realistic EP deferred).** The first EP=16 sweep was unfaithful — it fed *global* routing ids (range 256) against only 16 local experts at `offset=0`, so the kernel skipped most tokens (implausibly low time) and the metrics over-counted weight bytes (impossible >50 TB/s). Fixed for the MVP by routing the activation *within* the local experts (`selected ∈ [0, local_num_experts)`), modeling a single rank as a complete MoE over its local experts. Every token is now computed locally, so latencies are real and the derived metrics are correct. Validated (DeepSeek-V3, local=16):

| EP16 tokens | CuteDSL (ms) | TRTLLM-gen (ms) | winner | bandwidth |
| --- | --- | --- | --- | --- |
| 1    | 0.045 | 0.045 | ~tie | ~4.4 TB/s |
| 16   | 0.077 | **0.073** | trtllm_fp4_routed | ~5.5 TB/s |
| 4096 | **0.696** | 0.975 | cute_dsl_nvfp4 | ~0.68 TB/s |

Bandwidth is now physically sane (≤6 TB/s) and both backends pass `--refcheck` at every shape. **Realistic wide-EP** — global top-k-of-N routing with cross-rank dispatch and the resulting per-rank load imbalance — is **out of scope for this PR** and tracked for the separate follow-on `moe_ep` API PR (see Post-MVP Carryover). The building blocks already exist: `compute_reference_moe_fp4` accepts `num_local_experts`/`local_expert_offset` and skips non-local tokens, and `bench_moe_deepseek.py` scales work by `local_fraction = num_local_experts/num_experts` (uniform-distribution assumption); a faithful version would feed each rank only its dispatched tokens rather than assume uniformity.

#### Legacy-vs-unified kernel equivalence (cross-check, 2026-06-01)

Diligence requested in review: confirm the unified benchmark measures the *same*
underlying trtllm-gen kernel as the legacy flat routine — not a different or
no-op path. Both go through `get_trtllm_moe_sm100_module()`; the legacy
`trtllm_fp4_block_scale_moe` is fed routing logits (routing runs *inside* the
kernel), whereas the unified `trtllm_fp4_routed` uses
`RoutingInputMode.PackedPrecomputed` (pre-routed → GEMM/activation/finalize
only). Same DeepSeek-V3 geometry (hidden=7168, intermediate=2048, 256 experts,
top_k=8, n_group=8, topk_group=4, routed_scaling_factor=2.5), B200, CUDA-graph
timing.

| num_tokens | legacy `trtllm_fp4_block_scale_moe` | unified `trtllm_fp4_routed` | Δ (unified − legacy) | unified `cute_dsl_nvfp4` | MoELayer winner |
| --- | --- | --- | --- | --- | --- |
| 1    | 0.047 ms | 0.041 ms | −13% | 0.045 ms | trtllm_fp4_routed |
| 1024 | 1.412 ms | 1.300 ms | −8%  | 1.199 ms | cute_dsl_nvfp4 |

The unified routed path tracks the legacy kernel to within ~8–13% and is
consistently *slightly faster* — by ~the in-kernel routing cost it legitimately
skips (≈6 µs at t=1, ≈0.11 ms at t=1024). That is the expected signature of "same
GEMM kernel minus routing," not a discrepancy; a no-op/half-pipeline would show a
5–10× gap. The per-shape winner flip (trtllm at t=1, CuteDSL at t=1024)
independently reproduces the DeepSeek crossover above. **Conclusion:** the
`unified_nvfp4_moe` benchmark measures the real kernel — functionality confirmed.

Repro: `benchmarks/flashinfer_benchmark.py --routine {trtllm_fp4_block_scale_moe, unified_nvfp4_moe}` at the geometry above (the legacy routine adds `--use_routing_bias --routing_method deepseek_v3 --use_shuffled_weight`).

### Remaining MVP Follow-Ups

| Status | Task | Review refs |
| --- | --- | --- |
| [x] | Align the MVP config API and config tests: `tests/moe/test_moe_api.py` was rewritten to `QuantVariant` plus explicit `BackendOptions(candidates=...)`; the `QuantDtype`/`Fp8Variant`/pipe-operator spelling was dropped (see Decision Log). | CR1 |
| [x] | NVFP4 **weight** prep is now first-class: `flashinfer/fused_moe/prepare.py` provides `prepare_trtllm_fp4_weights` / `prepare_cute_dsl_nvfp4_weights`, exposed as `TrtllmFp4Config.prepare_weights` / `CuteDslConfig.prepare_weights` (per C7). The duplicated `_build_trtllm_view` (test) and `_build_trtllm_nvfp4_view` (benchmark) are removed; both call sites use the helper. **Activation** prep is the remaining slice: the Pack carries one `hidden_states_scale`, and trtllm-gen happily consumes the linear-layout scale (the runner only re-`view`s `uint8`→`float8_e4m3fn`), so a first-class activation-prep helper (and a future swizzled-activation backend) is the leftover follow-up. | CR2, CR7 |
| [x] | Wire `ExpertConfig.local_expert_offset` into TRTLLM `pack_inputs(...)` (read from config) and add an EP-offset test (`TestTrtllmRoutedPackingContract`) for the pre-routed path. | CR3 |
| [x] | Layer reuse contract decided: bucket-keyed winner cache (`map_to_hybrid_bucket`), so reuse across token counts re-selects per bucket. See Decision Log. | CR4 |
| [x] | `ExecutionConfig.tune_max_num_tokens` is threaded into the TRTLLM runner tuning config via `MoERunner._make_tuning_config`; benchmark-sweep validation is the remaining P2 evidence item. | CR5 |
| [x] | Added `MoELayer._validate_mvp_scope` (NVFP4 + Swiglu fail-fast) and a clearer no-usable-backend error; pre-routed-only is structural via `MoEActivationPack`. Covered by `TestMoELayerMVPValidation`. | CR6, C37-C38 |
| [x] | Documented the as-built MVP in the new "MVP As-Built Reference" subsection: end-to-end example plus `MoEActivationPack` / `MoEWeightPack`, first-class `prepare_weights`, two-stage cross-backend autotune, per-bucket winner caching, and `winner_backend` / `reset_winner` introspection. | CR7-CR9 |
| [x] | `unified_nvfp4_moe` runs end-to-end and emits `winner_backend` + per-candidate latency (one row per candidate; see the Decision Log evidence table) and now supports `--refcheck`: each candidate is verified against a shared bf16 reference (both views derive from the same bf16 weights). Validated on B200 — both backends 100% within tolerance. | CR10-CR11 |

### PR #3093 Review Threads — Reviewer Pass 2 (2026-06-02/03)

> **Status (2026-06-09): all resolved.** The second human-reviewer pass (G1–G7;
> `G1–G6` Reviewer 12 on `flashinfer/fused_moe/api.py`, `G7` Reviewer 13) is fully
> addressed and pushed — the bot threads (CodeRabbit ×6, Gemini ×3) and earlier
> self-notes were resolved in the 2026-06-01 pass. The per-thread decisions and
> rationale live in the commits (`fix(moe): … review comments` updates 1–3) and,
> for the structural threads, in the code + sections above: G1 enum
> consolidation (the enum block in `api.py` / `tllm_enums.py`), G4/G5 the
> `MoETensors` cluster drop ("Two packs, two lifetimes" + the pack rationale in
> `api.py`). The Reviewer 14 `_select_winner` thread remains tracked under Post-MVP
> Carryover (a deferred design decision, not part of this pass). The *open* MoE
> work is now the two fuzzer-filed bugs — gh #3547 / #3548 — described under
> "Test Harness" above.


### Test Harness — Forward-Compatible Fuzzer (PR #6, merged 2026-06-09)

`tests/moe/test_unified_moe_fuzz.py` (merged from `aleozlx/flashinfer#6`,
branch `yanxu/unified-moe-api-fuzzer`) drives the **real user-facing surface** —
one `MoEConfig` → `XxxConfig.prepare_weights(w1_bf16, w2_bf16, …)` →
`MoELayer`'s per-backend runners — so the production dispatch + the `prepare.py`
scale/layout plumbing are what's under test, where low-precision-MoE bugs
cluster.

**Forward-compatible by construction:**
- Backends are **auto-discovered from the live runner registry** (`layer.runners`);
  an unwired backend is skipped and gets covered the moment its runner lands —
  zero new test code.
- Weight prep is the uniform `cfg.prepare_weights(...)` (canonical bf16 in,
  quantize+layout internal).
- Per-dtype specifics live in one `_DTYPE` table (golden-input snap / activation
  pack / canonical reference / poison / tolerances). New dtype = one
  `DTypeHandler`; new backend = free.

**Verification model** (uniform per config): (1) no crash / no NaN-Inf where the
reference is finite; (2) numeric agreement vs a **single authoritative
quant-aware reference** — inputs snapped to the exact nvfp4 grid + sparsified so
a structural bug (dropped expert / wrong index / wrong scale role) is a gross
error, tolerance at the fp4 requant floor (~0.08); (3) per-backend determinism
(bitwise reproduce unless declared non-deterministic, e.g. CuteDSL atomic
finalize); (4) output-buffer poison (garbage+NaN/Inf in the kernel's `new_empty`
output → the torch→JAX buffer-hygiene guard); (5) autotune-tactic sweep (every
valid tactic matches, not just the default); (6) autotune-ON real path
(`autotune(True)` profiles+selects+caches a winner, output still matches); (7)
device-state probe (turns a context-corrupting IMA into a clean failure). A
sibling `test_autotune_cache_coherence` scenario covers the cross-call winner
cache (token-count sequence across bucket boundaries 4095/4096/4097).
Cross-backend agreement is **intentionally not** a check — an authoritative
tight reference already catches (and names) a deviating backend.

**Config space:** random non-pow2 (aligned) hidden/intermediate, odd/tile-boundary
token counts, routing-load skew, and ~30% **expert-parallel shards** (global >
local + `local_expert_offset` — the real deployment shape, in scope for the
single-GPU harness; the EP *collective* is not), all under a weight-memory
budget so one config never hogs the GPU.

**Known-failure ledger** (`_KNOWN_FAILURES`): a filed-and-tracked bug is `xfail`ed
by `(backend_key, predicate)` — the case is **still run**, so the suite stays
green yet flags loudly (`xpass` → "remove this entry") the day the bug is fixed.
A crash is never tolerated, only a wrong answer.

**CI-safety gate (waived, opt-in).** The ledger tolerates a *wrong answer* but
cannot absorb a *process abort*, and a single-process run of this suite on SM100
hit `CUDA error: device-side assert triggered` → `Fatal Python error: Aborted`
(triage 2026-06-09) — which would block B200 CI. Per-config isolation passes
68/86 incl. EP `offset>0`, so the abort is **not** cleanly attributable to one
config (the #3547 EP case returns tolerated zeros under `synchronize`, no
assert); it surfaces only in the accumulated single-process run CI uses, and
`--forked` can't isolate it (CUDA inits at collection). So the suite is gated
behind `FLASHINFER_UMOE_FUZZ` (`pytestmark` skip): **unset (CI default) →
collected-and-skipped, launches no kernel, cannot abort the job**; set → runs
(developer / nightly). The follow-up PR fixes #3547, root-causes the abort, and
removes the gate.

**Bugs this fuzzer found + filed** (the EP/scale regimes the prior suite never
exercised end-to-end):
- **gh #3547** — `trtllm_fp4_routed` returns all-zeros for EP shards
  (`local_expert_offset > 0`): the offset is applied twice (pre-subtracted in
  `pack_inputs` *and* forwarded to the kernel). `cute_dsl_nvfp4` is correct
  (passes global ids + offset, kernel localizes once). Encoded as the current
  `_KNOWN_FAILURES` entry. Fix = stop pre-subtracting (pass global ids), then
  delete the ledger entry so the case flips to passing.
- **gh #3548** — activation **global-scale** gap: `prepare_*_weights` hardcodes
  `gs=1.0`/`fc2_input_scale=1.0`/`alpha=ones` and `MoEActivationPack` has no
  global-scale field, so calibrated-checkpoint scales are silently dropped
  (~2400× output inflation). This is roadmap item #5 below (a standardized
  intermediate-scale **QuantSpec** policy), not a quick fix; quick mitigation is
  to make `prepare_*` fail **loud** on a non-default scale.

**Roadmap (ranked, from the 2026-06-09 audit of 51 past MoE issues):** (1) a
Blackwell/SM120 **PR-CI runner** — highest leverage, since PR-gating CI tops out
at SM90 and the dominant fp4/MoE bug class is collected-then-skipped at PR time;
(2) N-run stress + per-test timeout under `--forked`; (3) curated production
shapes (DeepSeek-V3 / Llama-4 / Qwen3 / Mixtral + tile-window enumeration); (4) a
build-manifest oracle (assert each advertised backend×quant×arch instantiates a
kernel); (5) tighten the quantized-numeric net via the QuantSpec scale policy
(also unblocks #3548 and an independent fp32 reference).

### Post-MVP Carryover

| Status | Task | Review refs |
| --- | --- | --- |
| [ ] | Design a backend discovery/support-query API that can tell users whether a config is supported without trial-and-error execution. | C16, C18-C19, C37-C38 |
| [ ] | Decide the long-term custom routing extension point, including how routed MoE, caller-provided top-k IDs, and custom scoring functions should compose with fused backends. | C24-C27, C40-C41 |
| [ ] | Keep new routing enums aligned with the shared enum home instead of creating a parallel enum surface in the MoE API. | C42 |
| [ ] | Decide whether repro logs remain same-version-only or need a versioned schema for cross-version bug reports. | C4-C5, C39 |
| [ ] | **Realistic wide-EP** lands in the separate `moe_ep` API (PR #3453): global top-k-of-N routing + cross-rank dispatch/combine + per-rank load imbalance. This MVP only ships a *local-only* per-rank proxy (route within local experts). Coordination seam: `moe_ep`'s `MoEEpLayer.forward` does `dispatch → inner_compute → combine`, and `inner_compute` (identity today) is where this unified `fused_moe` path becomes the per-rank expert compute — so our `ExpertConfig.local_expert_offset` wiring and local-only benchmark proxy already model that compute side. Building blocks for faithful EP: `compute_reference_moe_fp4`'s `num_local_experts`/`local_expert_offset` local-skip, `bench_moe_deepseek.py`'s `local_fraction` metric scaling. (Detailed coordination review in `var/log/`.) | #3453 |
| [ ] | **Make the low-level trtllm-gen TVM-FFI ops take structured config objects instead of long positional argument lists** (§5). The mid-cut blocker (the unified runner rotting after a `main` merge silently inserted `routing_input_mode` / `topk_weights` / `per_token_scale` and moved the tactic arg) was a *positional-argument-drift* failure with no compile-time signal. Delegating to `core.MoERunner` reduced the fragile call to one site; a structured `…Node::FromObject(config)` boundary (C++ reads named struct members) would remove the failure mode entirely and let adapters pass dataclass configs through unchanged. Out of MVP scope — sequence it after the NVFP4 MVP lands. | §5 |
| [ ] | **Gate `MoELayer._select_winner` behind the tuner's tuning mode.** Today a bucket-miss unconditionally runs the cross-backend `bench_gpu_time` shootout — even outside `autotune(True)` and even for ops `AutoTuner` would treat as `skip_ops` (where `choose_one` is a pure cache lookup / immediate fallback). The MVP is always exercised under `autotune(True)`, so this is correct for the shipped tests/benchmark, but a production layer built without an autotune context still silently benchmarks on first use per bucket. Follow-up: gate the shootout behind `self.tuner.is_tuning_mode`; when false and the bucket is uncached, select via a deterministic priority order (the §4 `DEFAULT_PRIORITY` heuristic) instead of benchmarking, and honor `skip_ops`. Needs a decision on the priority order, so deferred rather than rushed. | review (Reviewer 14) |

### Explicit Non-Goals For This MVP

| Status | Task | Notes |
| --- | --- | --- |
| [ ] | Keep FP8, MXFP4, and additional backend families out of this PR. | Track as follow-up runners after the NVFP4 path is solid. |
| [ ] | Keep general routing unification out of this PR. | This MVP assumes pre-routed inputs through `MoEActivationPack`. |
| [ ] | Keep broader public API ergonomics separate from MVP correctness. | Backend discovery, pipe-operator sugar, repro replay, and a full eager functional API can evolve after the MVP contract is stable. |
