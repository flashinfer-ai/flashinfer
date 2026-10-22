# SM90 NVFP4 × FP8 expert-parallel MegaMoE

This backend runs packed NVFP4 expert weights with FP8 E4M3 activations and
BF16 output on Hopper. It uses the public `flashinfer.moe_ep.MoEEpLayer` API,
pull-style expert parallel communication, fused gate/up and FC2 computation,
and a separate top-k reduction. The weight payload stays packed until the
consumer converts fragments for FP8 WGMMA.

## Weight and numerical contract

Canonical FC1 weights have shape `[local_experts, 2 * intermediate, hidden]`
(gate rows followed by up rows); FC2 has shape
`[local_experts, hidden, intermediate]`. Both widths must be multiples of 128.
The prequantized representation has packed E2M1 bytes along K, an E4M3 scale
per 16 values, and an optional FP32 per-expert global scale. `MoEWeightPack`
also accepts BF16/FP16/FP32 canonical weights for one-time NVFP4 preparation.

`kperm=64` is the default. Preparation reorders weights inside each K64 block;
input staging applies the corresponding permutation to natural-order inputs.
Callers must use the same `kperm` for preprocessing and the backend config.
This reduces repeated scale/LUT loads, but changes FP8 accumulation order:
outputs need not be bitwise equal to `kperm=0` or `32`. K64 is not inherently
less accurate than K32. Historical tests observed about 1.3% relative L2 error
against the FP32 reference for this quantized workload, and required K64's
error to differ from K0 by no more than 0.05 percentage points. This is an
observed validation result, not an error guarantee for arbitrary models.

`kperm=32` remains available as the tested byte-identical alternative to K0.
`kperm=0` uses natural K order. Activations use per-token E4M3 scaling, the
gate/up clamp defaults to 10, accumulation uses `1xacc`, and combine/output
uses BF16. Preparation, activation staging, and kernel settings form one ABI;
do not mix transformed layouts from different `kperm` values.

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
    weights, hidden_size=7168, intermediate_size=3072, kperm=64
)
layer = MoEEpLayer(
    bootstrap=BootstrapConfig(world_size=world_size, rank=rank, process_group=pg),
    fleet_params=FleetParams(
        num_experts=384, max_tokens_per_rank=tokens, token_hidden_size=7168
    ),
    weights=weights,
    backend=MegaConfig(
        megakernel=Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=3072, top_k=6, kperm=64
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

The built-in measured defaults target **NVIDIA H200 (132 SMs), EP4,
H7168/I3072, 384 experts, top-6, clamp 10, K64, `block_permutation_v1` routing**.
There is a separate exact-capacity profile for every tokens/rank value:

```text
8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
```

The larger buckets have their own profiles; they do not reuse T2048. Other
hardware, model shapes, routing profiles, or capacities use the legal
heuristic fallback rather than claiming these measured defaults apply.
`knobs=None` selects the matching default or fallback. `knobs="auto"` performs
a collective bounded search at warmup; a complete `knobs` dictionary selects
a manual tactic. NV has its own candidate/cache identity and includes LUT
replication, mapping, and compact pull-buffer choices. This initial backend
does not include the separate two-stage scale-ring (`sf2`) optimization.

`nvfp4_fc1_k256_overlap` accepts `None`, `True`, or `False`. `None` preserves
the profile's measured schedule; an explicit value overrides it. The measured
defaults disable overlap at T8/T64/T256 and enable it in the other ten buckets.
An explicit overlap override is independent of a complete tactic dictionary.
K32/K0 and unmatched shapes use their applicable legal policy, not the K64
performance claim.

## Portable benchmark

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

The portable fixture follows the historical deterministic recipe: both formats
quantize the same BF16 Gaussian weights divided by `sqrt(K)`, with per-expert
seed `20260914 + global_expert * 17 + leg * 100003`. NV uses variable K16 E4M3
scales and per-expert global scales; MX uses K32 E8M0 scales. Each token bucket
resets the input/score generator to `42 + rank`, uses softmax-normalized Gaussian
routing scores, and uses block-permutation route seed 1234. Weight preparation
is outside timing. Matching the recipe does not establish saved tensor byte
identity across PyTorch/CUDA versions or promise identical historical latency.
NV and MX use their respective configuration defaults; `--mx-compact` is an
explicit MX configuration choice, not a new MX autotuning axis.

Use `--nv-knobs-json` or `--mx-knobs-json` for a complete tactic. For a paired
before/after NV comparison, also pass `--nv-reference-knobs-json`: the three
cases share inputs and prepared weights. A single-capacity JSON can contain
`{"tactic": {...}, "config": {"kperm": 64,
"nvfp4_fc1_k256_overlap": true}}`; multiple capacities use
`{"per_token": {"512": {"tactic": {...}, "config": {...}}}}`.
The config object for MX can contain `compact_pull_buffer`.

CUDA events include MegaMoE and top-k reduction only. The calibrated GPU
padding, rank barrier, allocation, input staging, warmup, and cooldown are
outside the event interval. Each block's score is the maximum of four rank
sample medians. The JSON preserves attempts, raw samples, submission timing,
and same-round ratios; ratio summaries are medians of paired ratios. Up to
five attempts retain the first block passing the rank-skew rule. Three rounds
in one process are not three independent fresh-process runs. Repeat torchrun
for independent validation. Clock requests/ACKs do not prove continuous target
frequency, and a low-latency run can have too few telemetry samples to do so.

NVFP4's payload plus local scales uses `0.5 + 1/16 = 0.5625` bytes/weight;
MXFP4 uses `0.5 + 1/32 = 0.53125`. The 5.9% difference describes this part of
weight traffic. It is not a hard lower bound on the latency gap: reuse,
conversion, pipeline depth, communication, and scheduling also matter.
