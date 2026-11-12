# MOSCA MegaMoE (experimental, SM100)

This is an experimental **backend** behind the stable `MoEEpMegaLayer` API.
- Select it by name, `kernel_name="sm100_fp8_fp4_bf16_mosca"`, through
  `Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig`.
- Constructing a layer with it emits one `ExperimentalWarning`.
- It does not take part in automatic backend selection, autotuning or
  trace-apply.
- It is not registered for AOT (JIT only).

`../README.md` covers how it is selected, what is compiled, where MOSCA changes
the code, and the validated coverage.

## What it is

The kernel is DeepGEMM's SM100 FP8 x FP4 MegaMoE kernel (`fp8_fp4_mega_moe`).
It runs dispatch, FC1, SwiGLU, FC2 and combine in one persistent launch over
NVLink symmetric memory. Here it is a FlashInfer-owned copy with MOSCA's changes
(`../csrc/megamoe/sm100_fp8_fp4_mega_moe.cuh`), derived from a pinned upstream
commit (`../VENDOR.md`).

It differs from the stable `sm100_fp8_fp4_bf16_deepgemm` backend in where the
kernel comes from:
- The stable backend runs the same kernel family through whatever `deep_gemm`
  package is installed.
- This backend carries the source, so its kernel is fixed by the FlashInfer
  checkout, and it does not import `deep_gemm`.

Each MOSCA change is a separate switch, and with all of them off the kernel is
the upstream kernel:

| Config field | What changes | Outputs |
|---|---|---|
| `small_t=True` (default) | When every token fits SM 0's first dispatch round, SM 0 publishes the final per-expert counts. The cross-SM count round (global atomics) and one grid sync are skipped. The round holds `(dispatch threads / 32) * (32 // top_k)` tokens per rank, which is 20 for DeepSeek-V4-Pro (top-6), so the switch helps decode-sized batches. | bitwise identical |
| `sparse_count=True` | Only the SMs that own tokens add to the global send counters (requires `small_t`). | bitwise identical |
| `token_layout="replicated"` | In-kernel all-gather (see the next section). | bitwise identical to slice + all-gather |
| `tile_rule="..."` | Per-call tile override by the per-rank token count `t`. Example: `DSV4_PRO_DEP8_TILE_RULE` = `t>=41&t<=56:block_m=32;t>=57&t<=80:block_m=64;t>=193&t<=451:block_m=96,store_block_m=16`. Keys are `block_m`, `store_block_m` and `num_stages`. The rule is validated against the kernel's static constraints, and an invalid rule raises. The example rule was tuned in SGLang with the shared expert in kernel, and its speed is not validated on this backend (`../README.md`). | bitwise identical |
| `num_sms=N` | Persistent grid of N SMs (even). `None` uses all SMs. | bitwise identical |

### Replicated token layout (in-kernel all-gather)

With attention tensor parallelism, every rank of the EP group holds the same
batch. The usual MegaMoE pattern is:
1. split the batch (`torch.tensor_split`);
2. run the MoE on each rank's slice;
3. all-gather the slices back with NCCL.

With `token_layout="replicated"`, the post-MoE all-gather disappears:
- Every rank passes the same full batch `[T, hidden]`, and the backend stages
  only its `tensor_split` slice.
- During combine, the kernel writes every combined row into a symmetric
  gathered region on every peer.
- `forward` returns the full `[T, hidden]` output on every rank. It copies the
  gathered region into the caller's output, and a later zero-copy variant can
  drop this copy.

In this layout, `FleetParams.max_tokens_per_rank` bounds the replicated batch
`T`, and each rank's kernel capacity is `ceil(T_max / world_size)`.

## Contract

| Item | Value |
|---|---|
| Compute capability | Builds for 10.x (`sm_100f`: B200/GB200, B300/GB300), which needs nvcc with `sm_100f` support (CUDA 12.9 or newer). Validated on B200 only: see `../README.md`, "Validated coverage". |
| Activations | BF16 `hidden_states`, either quantized while staging to E4M3 with per-32 UE8M0 scales (`quantize_input=True`; same bytes as `sm100_fp8_fp4_bf16_deepgemm`, see `staging.py` for the one difference from DeepGEMM's own cast), or pre-quantized E4M3 plus packed scales (`quantize_input=False`) |
| SwiGLU clamp | `activation_clamp` is required and finite. "No clamp" (+inf upstream) is not supported yet: the launcher instantiates the kernel in host code, and nvcc's host stub cannot spell a +inf template argument. |
| Weights | Per-rank `w13 [E_local, 2 * intermediate, hidden]` (gate rows, then up rows) and `w2 [E_local, hidden, intermediate]`. Either BF16 (quantized at load to E2M1 with per-32 UE8M0 scales) or packed E2M1 bytes with FP32/UE8M0 per-32 scales. Same layout as `sm100_fp8_fp4_bf16_deepgemm`. Kernel-ready weights (`preprocess_weights=False`) must pack the expert groups of each scale tensor back to back. |
| Routing | int64 `topk_ids` (global expert ids), float32 `topk_weights` |
| Geometry | `hidden % 256 == 0`, `intermediate % 128 == 0`, `num_experts % world_size == 0`, top-k <= 32 |
| Shared experts | not supported (the caller computes them), as in `sm100_fp8_fp4_bf16_deepgemm` |
| Output | owned BF16 output; `return_workspace_view` is not supported |
| Communication | torch symmetric memory over the EP process group (one NVLink domain); runtime requirement `torch_dist` |
| CUDA graphs | `warmup()` on every rank, then capture. With `precompile=True` (default), warmup builds every launch configuration, so any token count up to the capacity can be captured. `precompile=False` (one EP rank only): a configuration is built on first use, and capturing one that was never built raises. |

Every EP rank must launch every forward, including ranks with zero tokens,
because the kernel synchronizes the ranks through symmetric memory.

## Opt-in

```python
from flashinfer.experimental.mosca.megamoe import Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig
from flashinfer.moe_ep import (
    BootstrapConfig, FleetParams, MegaConfig, MoEEpLayer, MoEEpTensors,
)

layer = MoEEpLayer(
    bootstrap=BootstrapConfig(world_size=8, rank=rank),
    fleet_params=FleetParams(num_experts=384, max_tokens_per_rank=512,
                             token_hidden_size=7168),
    weights=weights,  # MoEWeightPack for this rank's 48 experts
    backend=MegaConfig(megakernel=Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig(
        intermediate_size=3072, top_k=6, activation_clamp=10.0,
        token_layout="replicated")),
)
layer.warmup()
y = layer.forward(MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=w))
```

`examples/experimental/mosca_megamoe_ep.py` is a runnable version.

## Build

JIT only. `jit.py` builds two kinds of tvm-ffi modules with FlashInfer's JIT:
- **A host planner**, `../csrc/megamoe/sm100_fp8_fp4_planner.cu`. It computes
  the symmetric-buffer layout and the per-call launch configuration, using
  DeepGEMM's tile heuristic, or a tile rule validated against the kernel's
  static constraints.
- **One kernel module per launch configuration**,
  `../csrc/megamoe/sm100_fp8_fp4_launcher.cu`. Its generated header fixes the
  template arguments and the MOSCA feature macros.

## Tests and benchmark

`tests/experimental/mosca/` holds:
- CPU tests: config, catalog, vendored-tree manifest, packaging, row split,
  weight transform and planner.
- A multi-rank test (`torchrun --nproc-per-node 1..8`) on two shapes: a small
  one, and the DeepSeek-V4-Pro MoE layer. It checks:
  - every switch and `num_sms`, bitwise against the switches-off kernel, at
    equal and at uneven per-rank token counts;
  - the same against the stable `sm100_fp8_fp4_bf16_deepgemm` backend, when
    `deep_gemm` is installed;
  - every switch against an independent torch oracle, in both layouts;
  - `fast_math=False`, against the stable backend;
  - pre-quantized input;
  - the replicated layout, against slice + all-gather;
  - CUDA-graph replays against eager, including a token count never run before
    capture;
  - `precompile=False`: refused with more than one rank, and the capture guard
    with one.

`benchmarks/moe_ep/bench_mosca_megamoe.py` times both layouts against the
stable backend.
