# SM90 NVFP4 × FP8 expert-parallel MegaMoE

This backend runs packed NVFP4 expert weights with FP8 E4M3 activations and
BF16 output on Hopper. It uses the public `flashinfer.moe_ep.MoEEpLayer` API,
pull-style expert parallel communication, fused gate/up and FC2 computation,
and a separate top-k reduction. The weight payload stays packed until the
consumer converts fragments for FP8 WGMMA. This uses **single-promotion**:
each NVFP4 fragment is decoded into one E4M3 fragment, rather than a
dual-fragment representation.

The NV frontend, configuration, backend, and kernel specialize their MXFP4
counterparts. They share dispatch, scheduling, and the kernel entry, with
format-specific scale layouts, promotion, stage budgets, and FC1 overlap.
NV tuning and workspace/cache identities remain separate from MX.

This is the CuTe DSL backend of the unified `flashinfer.moe_ep.MoEEpLayer`
interface. The CUDA implementation introduced in
[#6343](https://github.com/flashinfer-ai/flashinfer/pull/6343) instead exposes
the independent `flashinfer.megamoe` API. Use this backend for MoE EP layer,
workspace and tuning integration; use `flashinfer.megamoe` when calling its
standalone CUDA kernels and symmetric-buffer API directly.

## Weight and numerical contract

Canonical FC1 weights have shape `[local_experts, 2 * intermediate, hidden]`
(gate rows followed by up rows); FC2 has shape
`[local_experts, hidden, intermediate]`. Both widths must be multiples of 128.
The prequantized representation has packed E2M1 bytes along K, an E4M3 scale
per 16 values, and an optional FP32 per-expert global scale. `MoEWeightPack`
also accepts BF16/FP16/FP32 canonical weights for one-time NVFP4 preparation.
For a signed E2M1 payload q, K16 scale s and global scale G, the kernel forms
`RN_E4M3(q*s/8)` and restores `8G` after MMA. This adds E4M3 rounding (and
possible underflow) to exact NVFP4 dequantization `G*q*s`; it is not an exact
conversion for every scale. The instruction reference and FP32 accumulation
reference in the end-to-end test use the promoted operands.

`nvfp4_kperm=64` is the default. Preparation reorders weights inside each K64 block;
input staging applies the corresponding permutation to natural-order inputs.
The prepared two-leg tuple carries a read-only `nvfp4_kperm` layout tag.
Backend validation rejects a tag that differs from the config, and rejects
untagged tuples. Keep the returned container intact with
`preprocess_weights=False`; rebuilding it as a plain tuple loses the tag.
This reduces repeated scale/LUT loads, but changes FP8 accumulation order:
outputs need not be bitwise equal to `nvfp4_kperm=0` or `32`. K64 is not inherently
less accurate than K32. Quantization error depends on the model and inputs;
assess it against FP32 computation using the effective quantized operands.

`nvfp4_kperm=32` remains available as a byte-identical alternative to K0.
`nvfp4_kperm=0` uses natural K order. Activations use per-token E4M3 scaling, the
gate/up clamp defaults to 10, accumulation uses `1xacc`, and combine/output
uses BF16. Preparation, activation staging, and kernel settings form one ABI;
do not mix transformed layouts from different `nvfp4_kperm` values.

## Public API

Run all EP ranks collectively, after initializing a torch distributed process
group and selecting the rank's CUDA device. Given canonical `weights` and a
`MoEEpTensors` input bundle:

```python
from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpLayer,
    Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig,
    preprocess_sm90_pull_nvfp4_mega_weights,
)

prepared = preprocess_sm90_pull_nvfp4_mega_weights(
    weights, hidden_size=7168, intermediate_size=3072, nvfp4_kperm=64
)
layer = MoEEpLayer(
    bootstrap=BootstrapConfig(world_size=world_size, rank=rank, process_group=pg),
    fleet_params=FleetParams(
        num_experts=384, max_tokens_per_rank=tokens, token_hidden_size=7168
    ),
    weights=weights,
    backend=MegaConfig(
        megakernel=Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=3072, top_k=6, nvfp4_kperm=64
        ),
        preprocess_weights=False,
        transformed_weights=prepared,
    ),
)
layer.warmup(inputs)
layer.stage_inputs(inputs)
layer.compute_staged(output=preallocated_bf16_output)
# ...
layer.destroy()  # collective, after outstanding work finishes
```

Alternatively, leave `MegaConfig.preprocess_weights=True` and let the layer
prepare weights once. `layer.forward(inputs)` includes staging; use
`stage_inputs` plus `compute_staged` when measuring computation separately.
Neither preparation nor staging is free, so include them when evaluating an
end-to-end inference integration.

## Defaults and autotuning

The reference defaults were measured on **H200, EP4, H7168/I3072, 384
experts, top-6, K64, `block_permutation_v1` routing** at these capacities:

```text
8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
```

For the reference model the heuristic rounds capacity up to the next bucket,
and uses the largest bucket above 32768. Clamp and SM90 product name do not
gate it. Other model shapes, routing or K layouts use a shape-compatible
fallback. These are legal starting configurations, not measured performance
guarantees for other capacities or hardware.

`knobs=None` first reads the separate NV tuning cache, then uses the default
heuristic on a miss. `knobs="auto"` performs a collective bounded search on
first compute; every rank must participate. A complete `knobs` dictionary
selects a manual tactic. NV has its own candidate/cache identity and includes
LUT replication, mapping and compact pull-buffer choices. Cache provenance
binds the layout and scheduling policy, while capacity/live-token bucketing
is handled by the cache itself.

The bounded NV pool combines the shape-compatible MX candidates with NV-specific
choices. New MX candidates therefore also enter the NV pool and are checked by
the NV compatibility filters and autotuner.

An explicit `token_back_mode` with `knobs=None` is a manual execution request:
it uses local defaults instead of inheriting a tuned profile. With a complete
`knobs` dictionary, the explicit token-back mode overrides only that field and
must fit the selected tactic's resource budget.

`nvfp4_fc1_k256_overlap` accepts `None`, `True`, or `False`. `None` preserves
the profile's schedule; an explicit value overrides it. The
defaults disable overlap at T8/T64/T256 and enable it in the other ten buckets.
An explicit overlap override is independent of a complete tactic dictionary.
K32/K0 and unmatched shapes use the corresponding legal fallback policy.

## Benchmark

After installing FlashInfer with its SM90 CuTeDSL/MoE dependencies, use four
H200 GPUs and at least two physical CPU cores per rank. Configure supported
GPU clocks externally if comparing with locked-clock measurements. The script
records device UUIDs and CPU affinity; it does not change clock settings.

```bash
# Short public-API sanity check; not a performance measurement.
torchrun --standalone --nproc-per-node=4 \
  benchmarks/bench_moe_ep_sm90_nvfp4.py \
  --tokens 8 --rounds 1 --warmup 2 --iters 5 --output sanity.json

# All 13 capacities. Each round pairs NV and MX in one resident process.
torchrun --standalone --nproc-per-node=4 \
  benchmarks/bench_moe_ep_sm90_nvfp4.py \
  --tokens 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 \
  --rounds 3 --warmup 10 --iters 50 --padding-us 300 \
  --cooldown-seconds 5 --mx-compact --output results.json
```

The benchmark quantizes the same BF16 weights for both formats and shares
inputs and routing assignments. Weight preparation is outside timing. NV and
MX use their respective configuration defaults; `--mx-compact` is an explicit
MX configuration choice, not a new MX autotuning axis.

Use `--nv-knobs-json` or `--mx-knobs-json` for a complete tactic. For a paired
before/after NV comparison, also pass `--nv-reference-knobs-json`: the three
cases share inputs and prepared weights. A per-capacity JSON can contain
`{"tactic": {...}, "config": {"nvfp4_kperm": 64,
"nvfp4_fc1_k256_overlap": true}}`; multiple capacities use
`{"per_token": {"512": {"tactic": {...}, "config": {...}}}}`.
The config object for MX can contain `compact_pull_buffer`.

CUDA events include MegaMoE and top-k reduction only. The calibrated GPU
padding, rank barrier, allocation, input staging, warmup, and cooldown are
outside the event interval. Each block's score is the maximum of four rank
sample medians. The JSON preserves attempts, raw samples, submission timing,
and same-round ratios; ratio summaries are medians of paired ratios. Up to
five attempts retain the first block passing the rank-skew rule. Three rounds
in one process are not three independent fresh-process runs. Use separate
`torchrun` processes to measure process-to-process variation, and account for
actual GPU clock and power states when interpreting results.

## Two-slot scale ring

`nvfp4_sf_stages=0` follows the AB pipeline's scale-buffer stage count.
Eligible non-pingpong M256N64K256 tactics can instead use
`nvfp4_sf_stages=2`, `compact_pull_buffer=True`, lane LUT mapping, and one
or two LUT replicas. Two scale slots reduce shared-memory use; the supplied
profiles use four AB stages. The AB and scale-ring barriers use their own
slot counts. This changes buffering, not the numerical operation: same-core
sf0/sf2 output remains byte-identical.

The candidate table includes five measured SF2 profiles: R1/R2 variants
of the T512 and T1024 seeds, and one R1 variant of the T4096/T16384 seed. Their
admission uses geometry, protocol and shared-memory legality, without GPU
product-name or live-token restrictions. T512/T1024 reference capacity
buckets use R1 SF2 defaults; the other eleven reference defaults use SF0.
Autotuning can select any legal profile by measurement, including with
FC1 overlap explicitly disabled. Explicit token-back selection without
`knobs` uses manual configuration rather than inheriting an SF2 default.

A complete manual tactic can set `"nvfp4_sf_stages": 2`. The AB stage count
must be a multiple of twice the scale-ring depth, and rounding must not
reduce it below the corresponding SF0 stage count. Illegal resource
combinations are rejected before launch. SF1 remains available for legal
manual configurations that meet this rule.
