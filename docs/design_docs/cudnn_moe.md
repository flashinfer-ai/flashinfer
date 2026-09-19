# cuDNN BF16 MoE adapter

`CudnnMoeConfig()` is an explicit candidate for the unified `MoELayer`. It
consumes canonical BF16 `[up, gate]` FC1 weights and `[E, H, I]` FC2 weights.
Prepare the `"cudnn"` weight view once, as for the other unified backends:

```python
from flashinfer.fused_moe import (
    BackendOptions, CudnnMoeConfig, ExecutionConfig, ExpertConfig,
    MoEActivationPack, MoEConfig, MoELayer, MoEWeightPack,
    QuantConfig, QuantFormat, RoutingConfig, RoutingInputMode,
)

config = MoEConfig(
    routing=RoutingConfig(num_experts=E, top_k=top_k),
    quant=QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
    experts=ExpertConfig(intermediate_size=I),
    backend=BackendOptions((CudnnMoeConfig(),)),
    execution=ExecutionConfig(enable_pdl=False),
)
weights = MoEWeightPack()
weights.prepare_for("cudnn", CudnnMoeConfig.prepare_weights(
    w1, w2, num_local_experts=E, hidden_size=H, intermediate_size=I,
))
layer = MoELayer(config)
act = MoEActivationPack(
    x, None, expert_ids, routing_weights,
    routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
)
out = layer(act, weights)
```

The adapter supports typed gated activations (`SwiGLU`, `SiTU`, `GeGLU`,
`GeGLUTanh`, and `SwiGLUStep`) and precomputed top-k routing. Set
`MoEConfig.activation` and pass the same activation to `prepare_weights`.
SwiGLU preserves its alpha, beta, and pre-activation clamp; SiTU preserves
gate/linear scales, `linear_scale=None`, and the optional pre-activation clamp.
SwiGLUStep clamps the gate after SiLU, as its public contract specifies.
Non-gated activations use a different FC1 weight layout and are not supported.
Unpacked routing preserves FP32 weights; packed routing follows the unified
BF16 routing-weight boundary. Quantized inputs, logits routing, expert
parallelism, shared experts, PDL and unfinalized output are not implemented.
The candidate is excluded from the default backend list.

The graph path uses `moe_grouped_matmul`, with two FC1 nodes sharing tokens and
expert offsets, followed by the typed activation, then an FC2 graph. Setting
`CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` offers the opt-in FROST engine. The backend
can accept the same FC1 expression; graph acceptance alone does not promise a
single kernel. If the shared-input graph declines, the adapter uses FP32 FC1
outputs and a separate cuDNN activation graph before the BF16 intermediate.
Activation scalars use FP32 device tensors allocated during plan preparation,
held by each stage, and rebound on execution. The current Frost grouped engine
does not accept graph-owned pass-by-value scalars. No scalar tensor is allocated
or copied by the activation execution path.

`CudnnMoeConfig(use_native_routing=True)` reuses FlashInfer's native sorter and
permutation with tile size 1. This produces unpadded grouped rows, including
empty experts. The sorter writes the expert boundaries directly into the
cuDNN offset buffer; no subsequent search is needed. Its optional offset output
preserves the existing six-output Python result and original native entry point.
Expert-count scratch for large token counts is prepared once.
The default uses torch sorting and gathering; both paths reuse FlashInfer's
native weighted unpermute. These stages are part of its latency. Empty experts
and non-aligned group lengths require no token padding. All shape-dependent
plans and storage must be warmed outside CUDA graph capture; changed routing
is processed again inside every replay. Keep the runner alive for the lifetime
of its captures. As with `MoELayer`'s other adapters, use one layer per concurrent
thread/stream. Outputs use reusable runner-owned storage.

The accompanying FROST change includes a one-thread scheduler-counter reset
in each compiled MoE plan, on the plan's execution stream. This preserves
initialization on every call and graph replay while avoiding the scheduling
gaps observed around separate four-byte asynchronous memsets on B200. Both
dense and block-scaled grouped MoE templates use the compiled reset.

Each stage uses a stable `(engine_id, sorted_knob_items)` identity. The BF16
runner tunes pairs of FC1/FC2 identities using the complete MoE operation.
`fc1_tactic` and `fc2_tactic` pin exact plans; `fc1_tactics` and `fc2_tactics`
provide immutable candidate domains. A pin and domain for the same stage are
mutually exclusive. Empty domains use the frontend's offered plans, which
currently do not constitute an exhaustive Frost catalog search.

Each distinct stage plan is prepared once, and its workspace covers every
candidate. The runner retains all captured resources. An unsupported explicit
plan fails during preparation; a stale pair is rejected before routing or
GEMM execution. FI's persisted tactic format retains both stage identities.
Existing direct FC1 tactic replay remains accepted, using FC2's default plan.

`fc1_fusion=None` preserves the preference for supported shared-input fusion
with an unfused fallback. `True` requires fusion and reports an unsupported
plan instead of changing routes; `False` prepares concatenated FC1 followed by
FP32 activation. Fusion changes resource requirements, so select tile domains
for each route independently. Only supported candidates should be supplied.

To compare routes and all pairs in their declared domains:

```python
from dataclasses import replace
from flashinfer.autotuner import autotune

candidates = tuple(
    CudnnMoeConfig(
        use_native_routing=True,
        fc1_tactics=tuple(first_stage_records),
        fc2_tactics=tuple(fc2_records),
        fc1_fusion=fused,
    )
    for fused, first_stage_records in (
        (True, fused_fc1_records),
        (False, unfused_fc1_records),
    )
)
layer = MoELayer(replace(config, backend=BackendOptions(candidates)))
with autotune():
    out = layer(act, weights)
```

Route, domain order, and exact stage identities participate in the cache key.
Normal execution reuses the measured winner. This searches the declared domains;
full catalog coverage requires explicitly supplying all supported records.
Record actual engines, coverage, and full-MoE scope when reporting performance.

On a frontend with the new Frost MoE scheduler option, the existing public
`cudnn.knob_type.SCHED_POLICY` field can select dynamic (omitted/0) or static (1)
in each stage record. This is independent of the tile geometry. A static plan
omits the scheduler reset; ordinary pointwise graph plans may also compile
absolute-A addressing and compatible shared-A wide MMA. These optimizations are
in the open-source Frost engine; the `cudnn` adapter name alone does not identify
the engine that executed. Record actual plans and kernel traces.

This is exhaustive only over the declared Cartesian product. Preparation costs
grow with distinct stage plans and routes, so retain a measured winner
for normal execution. FI's cross-runner graph selection and a separately reported
cold-L2 measurement can use different cache conditions; record both scopes.

`grouped_mm_bf16` and `grouped_mm_fp8` participate in ordinary `autotune()` using
exact shape keys and uniform offsets. All grouped cuDNN APIs accept stable
`(engine_id, sorted_knob_items)` tactics; missing identities raise instead of
silently choosing another plan. Their graph caches distinguish the frontend,
backend and FROST opt-in setting. Block-scale autotuning requires additional
work: per-expert scale-factor padding and offsets must be synthesized together.

The initial MoE architecture declaration covers SM100/103/107/110, with runtime
graph support checks. This is not an SM120 MoE implementation. FROST's separate
SM120 dense/block-scale GEMM templates do not provide grouped MoE coverage.

For block-scale grouped GEMM, F8_128x4 SFA is swizzled and padded to 128 rows
**per routed group**, then concatenated. Runtime Frost currently requires the
static capacity envelope `128 * (A + (M - A) // 128)` rows, where
`A = min(M, number_of_groups)`, and scale columns are padded to a multiple of
four. A tail allocation beyond the exact current partition preserves that
capacity without changing its scale values. SF values must be repacked whenever
routing changes. Tokens themselves stay unpadded.

The cuDNN descriptor describes global logical padding; it does not describe the
larger allocation. The accompanying frontend variant-pack fix preserves that
physical capacity for the runtime guard. Ragged MXFP8/FP4 use therefore requires
that frontend fix as well as this FlashInfer adapter. A globally quantized SFA
blob is only layout-compatible when the group boundaries satisfy its packing;
changing offsets alone is not a valid quantized autotune input generator.

### Additional cached-decode integration validation

On a 148-SM, 1000 W B200, a further integration refinement reduced complete
BF16 MoE latency from 29.756 us to 29.003 us (2.53%) for a single-token
Qwen3-30B-A3B layer-0 cached-decode fixture (E=128, top-k=8, H=2048, I=768)
with unpacked routing inputs. The unchanged packed-input control measured
28.732 us and 28.728 us. Values average four trial medians across two fresh
processes per implementation in forward and reverse order; both updated
unpacked processes were faster than both previous processes.

This is an integration benefit using the existing NVIDIA TensorRT-LLM/FlashInfer
native finalizer, whose authors retain implementation credit. It is measured
against the previous Frost integration with fixed GEMM configurations. It
excludes weight preparation, router-logit/top-k computation, attention and
model execution, and must not be added to results with different baselines.
The measured source passed direct-reference, live-input/weight, retained-graph,
actual-route, memcheck and racecheck gates, with an independent audit of
296 raw checks, 120 changed-reference controls and 768 timing spans. Publication
preserves the validated executable path; CPU dispatch checks also cover excluded
shapes and routing precision. This is not a new full-GPU run of the publication
checkout or a model-level speedup claim.
