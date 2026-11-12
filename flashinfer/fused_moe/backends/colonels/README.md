# Colonels fused MoE

This backend ports `poolside/colonels/fused_moe_up_down` into FlashInfer. All
kernel headers, launch templates, and configuration enumeration live in this
repository. Compilation uses FlashInfer's NVCC JIT/cache and AOT infrastructure.
The runtime does not import Colonels or vLLM.

| GPU | Weights × activations | Kernel families | Activations |
| --- | --- | --- | --- |
| SM90 | BF16 × BF16 | standard, persistent, GEMV | SwiGLU, ReLU2, ClampedReLU2 |
| SM90 | E4M3 × E4M3 | standard, persistent | SwiGLU |
| SM103 | BF16 × BF16 | SMEM, TMEM, GMEM; each standard/persistent; GEMV | SwiGLU, ReLU2, ClampedReLU2 |

`bf16_sm100*` tactic names retain the source kernel names; this contribution
advertises their tested target, SM103. FP8 uses K128 activation scales and
128×128 weight scales (`QuantFormat.DeepSeekFp8`). All kernels produce finalized
BF16 output. Their atomic accumulation permits small differences across runs.

## Unified API

```python
import torch
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    BackendOptions, ColonelsBf16Config, ExpertConfig, MoEActivationPack,
    MoEConfig, MoELayer, MoEWeightPack, QuantConfig, RoutingConfig, SwiGLU,
)

# x: BF16 [M, H]; ids: int32 [M, K]; scores: FP32 [M, K], all CUDA.
# Canonical w1: [E_local + S, 2*I, H], ordered [up, gate].
# Canonical w2: [E_local + S, H, I].
backend = ColonelsBf16Config()
activation = SwiGLU()
config = MoEConfig(
    routing=RoutingConfig(num_experts=E, top_k=K),
    quant=QuantConfig(),
    experts=ExpertConfig(intermediate_size=I, local_num_experts=E_local,
                         local_expert_offset=offset, num_fused_shared_experts=S),
    activation=activation,
    backend=BackendOptions(candidates=(backend,)),
)
weights = MoEWeightPack()
weights.prepare_for("colonels_bf16", backend.prepare_weights(
    w1, w2, num_local_experts=E_local + S, hidden_size=H,
    intermediate_size=I, activation=activation,
))
layer = MoELayer(config)
act = MoEActivationPack(x, None, ids, scores)
with autotune(True):
    output = layer(act, weights)
```

Preparation converts canonical `[up, gate]` weights into the kernels' interleaved
`[gate, up]` layout. Ungated activations use `w1: [E_local + S, I, H]`.
`ClampedReLU2(limit)` computes `(limit*tanh(relu(x)/limit))**2`.
SwiGLU supports the default alpha, beta, and limit. These are the shared
activation configs from `flashinfer.fused_moe.api`; unsupported activations
or parameters are rejected explicitly.

For FP8, use `ColonelsFp8BlockConfig`, the `colonels_fp8_block` weight-view key,
and `QuantConfig(QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8)`.
`prepare_weights` accepts BF16 sources or prequantized E4M3 weights with canonical
FP32 `w1_scale` and `w2_scale`. `prepare_activations(x)` returns an E4M3 payload
and canonical FP32 scales shaped `[H/128, M]`.

## Routing and expert parallelism

Both `PackedPrecomputed` and `UnpackedPrecomputed` packs supply logical int32
expert IDs and final weights (FP32, or BF16 for UnpackedPrecomputed). FlashInfer's CUDA `moe_sort` groups and pads routes
by expert; GEMV consumes the IDs directly. IDs must be distinct within each
token's selected experts. Negative, out-of-range, and nonlocal IDs contribute
zero. The output is this rank's local contribution; communication is caller-owned.

For noncontiguous expert placement, pass an int32 `expert_map` to
`prepare_weights`: one entry per global routed expert, containing its local
weight row or `-1`. Set `local_expert_offset=0`. Preparation validates and retains
the map in the weight view; different packs can use different maps on one layer.

Shared experts occupy the last `S` weight rows. A pack may include the shared
slots (global IDs `E` through `E+S-1`), as in the existing precomputed API, or
supply just the routed slots; the latter appends unit-weight shared routes.
Precomputed weights are final, including any routed scaling; the runner consumes
them exactly as supplied. The direct launch API also exposes `scaling_factor`. Use Colonels-only candidates for precomputed shared experts;
other backends retain their existing DeepSeekV3 shared-expert contract.
The sorter supports at most 32 combined routes per token and 2048 local plus
shared experts. FromLogits routing, PDL, and unfinalized output are rejected.

## Tuning, graphs, and direct launches

`ColonelsBf16Config(tactics=...)` and `ColonelsFp8BlockConfig(tactics=...)` accept
immutable `ColonelsTactic` values. An empty tuple selects a bounded search across
families. `get_colonels_tactics(arch, fp8=False, exhaustive=True)` exposes the
complete source catalogue, including tile sizes, stages, clusters, reduction
stores, preferred cluster fallback, weight eviction, producer/epilogue warps,
and GEMV z-splits. Geometry and launch-resource checks filter candidates before
timing; GEMV is searched automatically for at most 16 tokens.

Warm each token count and tactic before graph capture. Workspace buffers are
retained by the runner, keyed by shape and stream, with stream-specific GMEM
scratch shared across token counts. New capture streams allocate through the
PyTorch graph memory pool; replay can change the
activation and routing tensor contents. Callers must keep tensors alive and
order use of shared tensor storage across streams. Output is cleared on every
unified-API call because the kernels accumulate atomically.

For caller-controlled scheduling, import `Workspace` and `launch` from
`flashinfer.fused_moe.backends.colonels.runtime`. `Workspace.sort` returns the
aligned `(sorted_token_ids, expert_ids, padded_count)` consumed by `launch`.
For GEMV the first entry is the original `[M, K]` IDs. The direct interface takes
prepared weights, optional FP8 scales (input scales here are `[M, H/128]`),
caller-owned output, GMEM scratch, and `experts_done`.
`zero_output=False` leaves initialization/accumulation to the caller. A false
return means the tactic cannot launch for that geometry or available resources.

GMEM scratch requires `SM_count * min(block_m, 128) * I` BF16 elements.
Hopper persistent kernels support int32 completion storage of at least
`M*K*max(1, w1_rows/(block_n*warp_n)) + 1` elements. The caller initializes these
flags and coordinates the overlapping consumer. Other kernels reject them.

Run `pytest tests/moe/test_unified_moe_colonels.py` on SM90 and SM103. The suite
covers independent numerical references, activation/layout preparation, kernel
families and options, invalid/local/shared routes, graph replay, tuning, and
buffer contracts. Use `benchmarks/flashinfer_benchmark.py --routine unified_moe
--backends colonels cutlass cudnn --quant-variant bf16` to compare backends on
the same inputs; see `benchmarks/README.md` for a complete command. Colonels
also supports `--quant-variant deepseek_fp8` on SM90. The unified benchmark
times runner execution, including sorting and output initialization, after
`pack_inputs`. `benchmarks/bench_colonels_moe.py` measures the entire layer.
