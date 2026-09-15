# cuDNN Frost FP8 per-tensor MoE

`CudnnFp8PerTensorConfig` is an explicit candidate for precomputed routing on
SM100/SM103. It uses open-source Frost engine20400 for both grouped GEMMs and
FlashInfer native routing, permutation and weighted finalize. It supports the
typed gated activations accepted by `CudnnMoeConfig`, including parameterized
SiTU. It requires `enable_pdl=False` and weighted finalize. Logits routing,
expert parallelism, shared experts and SM120 are not implemented by this adapter.

This local extension is undergoing validation; it is not released.

```python
import os
os.environ["CUDNN_FRONTEND_ENABLE_FROST_ENGINES"] = "1"

from flashinfer.fused_moe import (
    BackendOptions, CudnnFp8PerTensorConfig, ExecutionConfig, ExpertConfig,
    MoEActivationPack, MoEConfig, MoELayer, MoEWeightPack, QuantConfig,
    QuantVariant, RoutingConfig, SiTU,
)

activation = SiTU(linear_scale=None, clamp_limit=0.25)
backend = CudnnFp8PerTensorConfig()
config = MoEConfig(
    activation=activation,
    routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
    quant=QuantConfig(variant=QuantVariant.FP8PerTensor),
    experts=ExpertConfig(intermediate_size=intermediate_size),
    backend=BackendOptions((backend,)),
    execution=ExecutionConfig(enable_pdl=False),
)
weights = MoEWeightPack()
weights.prepare_for("cudnn_fp8_per_tensor", backend.prepare_weights(
    w1_bf16, w2_bf16,  # [E,2*I,H] in [up,gate] order, then [E,H,I]
    num_local_experts=num_experts, hidden_size=hidden_size,
    intermediate_size=intermediate_size, activation=activation,
    hidden_states_scale_global=input_multiplier,
    intermediate_scale_global=intermediate_multiplier,
))
x_fp8, scale = backend.prepare_activations(
    x_bf16, hidden_states_scale_global=input_multiplier,
)
act = MoEActivationPack(x_fp8, scale, topk_ids, topk_weights)
layer = MoELayer(config)
y = layer(act, weights)  # eager preparation before CUDA Graph capture
```

Input and intermediate multipliers are positive calibrated FP32 values supplied
by the caller. Activation preparation computes `E4M3(clamp(x * input_multiplier,
-448,448))`. Each expert gets one FC1 weight multiplier shared by up/gate and
one FC2 weight multiplier. Weight storage is unshuffled; a TRTLLM physically
shuffled weight view cannot be passed to this runner. Preparation returns
independently contiguous up/gate tensors by default. Pass
`copy_fc1_weights=False` to `CudnnFp8PerTensorConfig.prepare_weights` to retain
views of one canonical `[E, 2*I, H]` E4M3 allocation and avoid two preparation
copies. Each `[E, I, H]` view has strides `(2*I*H, H, 1)`: each expert matrix is
contiguous, with the other half between consecutive experts. Both layouts are
accepted. Choose the weight layout before tuning: reduced preparation cost does
not guarantee reduced GEMM latency for every workload. Down weights and scale tensors remain
contiguous; other noncontiguous layouts are rejected. The graph and TMA
descriptor use the declared expert pitch directly, without forward copies.

For expert `e`, the mathematical sequence is:

1. `up = matmul(x_fp8, up_fp8[e]) * fc1_scale[e]`, likewise for gate.
2. Apply the typed gated activation in FP32, multiply by the intermediate
   multiplier, clamp to `[-448,448]`, and store E4M3.
3. FP8 FC2 accumulates in FP32, multiplies by `fc2_scale[e]`, and stores BF16.
4. Apply routing weights and reduce top-k contributions. PackedPrecomputed
   rounds the FP32 routing weights to BF16; UnpackedPrecomputed retains FP32.

`fc1_scale = 1/(input_multiplier * fc1_weight_multiplier)` and
`fc2_scale = 1/(intermediate_multiplier * fc2_weight_multiplier)`. The runner
validates metadata without reading device values during execution. Custom weight
views must preserve the same finite scale contract. Activation calibration must
match the weight view; `hidden_states_scale` and `per_token_scale` remain `None`.

Weight and scale tensors are rebound on each call. Preparing a second weight
pack with the same shapes does not change what an older packed call or captured
graph references. Prepared resources and tuning cache entries distinguish the
weight strides, so switching between split and contiguous layouts prepares
separate plans outside capture and preserves older captures. The cache schema
does not reuse records from before split-layout support. One layer per
thread/stream is still required; output storage is reusable and is overwritten
by subsequent calls for the same prepared shape and weight layout.

`fc1_tactic` and `fc2_tactic` accept stable `(20400, sorted_knob_pairs)` records.
The default is Frost's proposal, which is not an exhaustive performance choice.
Default plans expose the concrete knobs chosen by FE, so their records can be
replayed explicitly. Optional `fc1_tactics` and `fc2_tactics` accept immutable
tuples of explicit records. Each distinct stage plan is prepared once, with
workspace sized for the largest plan. Ordinary FI autotuning measures every
FC1/FC2 combination in one runner:

```python
from dataclasses import replace
from flashinfer.autotuner import autotune

# These contain stable (20400, sorted_integer_knob_pairs) records.
# Ten FC1 choices and ten FC2 choices declare a100-pair search domain.
backend = CudnnFp8PerTensorConfig(
    fc1_tactics=tuple(fc1_records), fc2_tactics=tuple(fc2_records),
)
layer = MoELayer(replace(config, backend=BackendOptions((backend,))))
with autotune():
    y = layer(act, weights)
# Reuse this prepared layer for eager calls and CUDA Graph capture.
```

A stage's singular tactic and candidate domain are mutually exclusive.
Candidate order defines the untuned fallback; tuning is exhaustive only over
the supplied domains. An unsupported explicit plan fails during preparation.
Joint tactic cache records contain `(fc1_engine_knobs, fc2_engine_knobs)` and
are distinct from the older FC1-only cache schema. Captures retain their own
operands when another candidate or weight pack is used later.

Use explicit configuration sweeps for performance comparisons. Full-MoE timing
includes native routing and finalize; quantization during model/input preparation
must be accounted for separately if the application performs it on the hot path.

## Experimental blocked weights

This draft adds `weight_layout="blocked_128x128_v1"`
to both `CudnnFp8PerTensorConfig` and its `prepare_weights` method. Preparation
returns contiguous physical `[E,N/128,K/128,128,128]` E4M3 weights; both calls must
select the same layout. Packing occurs once before inference. The runner checks
the physical shape and binds it directly through a versioned Frost graph
attribute. The layout participates in FI cache identity; it is not a tactic knob.

Initial support is SM100 with H/I multiples of 128 and explicit supported
one-CTA, N64/128 K128 tactics, and power-of-two cluster M/N with at most 16 CTAs. Other layouts and targets decline.
Existing ordinary weight behavior remains the default. See
[the draft handoff](../../FROST_MOE_HANDOFF.md) for graph replay, validation scope and remaining integration work.

Packed weights may also have an independent expert stride. For physical shape
`[E,N/128,K/128,128,128]`, the inner strides must be
`[K*128,16384,128,1]`; the expert stride must be at least `N*K` and a multiple
of 16 bytes. Up, gate and down may each use a different valid expert stride.
Frost describes that stride directly in TMA, without execution-time copies.
Overlapping expert blocks, misaligned pitches and noncontiguous inner blocks
are rejected. Ordinary unblocked weights retain their existing contract.

With `weight_layout="blocked_128x128_v1"`, `copy_fc1_weights=False` packs the
whole FC1 once and returns up/gate views of that allocation. The expert stride
is `2*I*H` and the gate address is `I*H` bytes after the up address. Quantization
and packing still happen during preparation. This option does not reduce the
total retained weight bytes and is not a universal forward speedup; choose it
before tuning and compare it for the intended workload. The default still
packs up and gate separately.


## Experimental native finalizer token range

The proposed range extension keeps the native tiled finalizer's existing
BF16/FP32-scale, PDL-off, 16-byte-alignment contract. Its explicit helper accepts
up to8192tokens for hidden4096/top-k8. The Frost MoE runner selects tiling only
through3072tokens for that shape, retaining the existing1..1024range for
hidden4096/8192 and top-k2/4/8/16 elsewhere. The kernel body is unchanged.

Independent component sweeps on full B200 find a material component
regression at the old1024/1025 boundary and diminishing gains at larger sizes.
These experiments support the proposed range; fresh target-GPU, sanitizer
and complete-MoE validation of the extension are pending. No full-MoE speedup is claimed here.
