# Cake StepFun fused MoE: host contract

This directory holds the host side of the `fused_moe_cake_stepfun_*` JIT modules:
the trtllm-gen fused-MoE pipeline (routing, FC1, activation requantization, FC2,
finalize) with one or more stages served by exported Cake kernels. The generated
units under `generated/` are produced by the Cake exporter and consumed through
`generated/cake_stepfun_inventory.json` and the launch manifest
`generated/cake_stepfun_generated_manifest.cuh`; nothing in this directory is
edited by hand except the files listed under *Hand-written sources*.

## Module variants

| macro | module URI | stages on Cake kernels |
|---|---|---|
| `-DCAKE_STEPFUN_FC1` | `fused_moe_cake_stepfun_sm100` / `_sm103` | FC1 (GEMM1 + fused SwiGLU-Step) |
| `-DCAKE_STEPFUN_FC1 -DCAKE_STEPFUN_FULL` | `fused_moe_cake_stepfun_full_sm100` / `_sm103` | routing, FC1, NVFP4 per-token requantization, FC2, finalize |

`flashinfer/jit/cake_stepfun_moe.py` selects the variant: the full path is built
exactly when the inventory lists kernels of every stage for the target
(`FLASHINFER_CAKE_STEPFUN_FULL_PATH=auto`, the default), is required by
`FLASHINFER_CAKE_STEPFUN_FULL_PATH=1` (a missing stage is an error naming the
stage) and disabled by `FLASHINFER_CAKE_STEPFUN_FULL_PATH=0`. The FC1-only module
runs the trtllm-gen routing, GEMM2 and finalize kernels of the public artifact.

The stages a (family, tile) pair does not cover have **no fallback**: the tile has
no tactic, and an explicit or default selection of it fails with the family, the
tile and the exported tiles.

## Hand-written sources

| file | role |
|---|---|
| `cake_stepfun_fc1_runner.{cuh,cu}` | `cake_stepfun::Fc1Runner`, the GEMM1 slot of `MoE::Runner` (same surface as `PermuteGemm1::Runner`) |
| `cake_stepfun_routing_tail.cuh` | benign routing tail for GEMM units that do not bound cluster-launch-control acquired tiles |
| `cake_stepfun_abi.cuh` | kernel ABI of the routing, FC2, requantization and finalize stages (`*Args`, `*KernelSpec`) |
| `cake_stepfun_stages.{cuh,cu}` | full-path stage runners: `RoutingRunner`, `Fc2Runner`, `requant::run`, `finalize::run` |
| `cake_stepfun_moe_binding.cu` | standalone TVM-FFI operations (`cake_stepfun_fc1*`, `cake_stepfun_fc2*`, `cake_stepfun_requant`, `cake_stepfun_finalize`, `cake_stepfun_stages`, `cake_stepfun_full_path`) |

Hooks outside this directory: `include/flashinfer/trtllm/fused_moe/runner.h`
(`Gemm1Runner` / `Gemm2Runner` aliases, header includes),
`csrc/trtllm_fused_moe_runner.cu` (requantization and finalize calls),
`csrc/trtllm_fused_moe_kernel_launcher.cu` (`RoutingRunner` alias at the five
routing call sites). Buffer allocation, workspace layout and the FFI surface are
those of the public module.

## Inventory (`flashinfer.cake_stepfun.inventory.v3`)

Top-level keys: `schema`, `manifest` (path of the launch manifest), `files`
(path -> SHA-256 of every generated file), `families` (FC1 families), optional
`fc2_families` (FC2 families, same shape), `kernels`, `program_hash` (SHA-256 of
`json.dumps({kernels, families, fc2_families, files}, sort_keys=True,
separators=(",", ":")) + "\n"`, keys present only when the inventory has them).

Every `kernels[]` record carries `stage` (`routing`, `fc1`, `requant`, `fc2`,
`finalize`), `arch` (`sm_100a` / `sm_103a`), `device` (translation unit path),
`compile_flags`, `kernel_symbol`, `name`, `parameters` (ordered `[type, name]`
pairs of the kernel signature), `block`, `cluster`, `dynamic_smem_bytes` and
`min_blocks`. Stage-specific fields:

| stage | record keys | uniqueness |
|---|---|---|
| `fc1`, `fc2` | `family`, `tile_n`, `output_rows_per_cta`, `block_k`, `bounds_acquired_tiles`; `fc2` adds `sf_layout_a` (`none`, `linear`, `r8c4`, `r128c4`) | one record per (arch, stage, family, tile), every tile of the family mapping present |
| `routing` | `variant`, `logits_dtype` (`float32` / `bfloat16`), `min_tokens`, `max_tokens`, `grid_rule` (`fixed` / `token_blocks`), `grid`, `tokens_per_cta`, `writes_benign_tail` | one record per (arch, stage, variant) |
| `requant` | `variant`, `sf_layout`, `rows_per_cta` | one record per (arch, stage, variant) |
| `finalize` | `variant` (`scalar` / `vector`), `max_top_k` | one record per (arch, stage, variant) |

A target with any kernel must have `fc1` kernels. The manifest must define the
kernel table of every stage the inventory lists (`kFc1Kernels`, `kRoutingKernels`,
`kRequantKernels`, `kFc2Kernels`, `kFinalizeKernels`); the loader checks this
before compiling anything.

## Launch manifest

The manifest declares every exported kernel, renders one `EncodeTensorMap_<i>`
thunk per TMA operand, one `Configure_<i>` thunk (dynamic shared memory
attribute) and one `Submit_<i>` thunk per kernel that unpacks the stage's `*Args`
structure into the kernel's parameter list, and defines a per-architecture table
of `*KernelSpec` entries under `#if FLASHINFER_CAKE_STEPFUN_TARGET_MINOR == 0 /
#else`. The host runners consume only the tables; the parameter order and the
pointer-vs-descriptor form of each operand are the unit's business.

The FC1 types (`TensorLayout`, `Fc1Family`, `Fc1Args`, `Fc1KernelSpec`) are
defined in the manifest; the types of the other stages are defined in
`cake_stepfun_abi.cuh`, which includes the manifest. Every table is
`inline constexpr <Spec> k<Stage>Kernels[]` with
`inline constexpr size_t k<Stage>KernelCount`.

## Stage contracts

Shapes: `T` tokens, `E` local experts, `top_k`, `H` hidden size, `I` intermediate
size, `tile` tokens per CTA. `max_padded = Routing::getMaxPermutedPaddedCount(T,
top_k, E, tile)`, `max_ctas = Routing::getMaxNumCtasInBatchDim(T, top_k, E, tile)`.
All launches honour the caller's programmatic-dependent-launch flag; kernels
wait with `griddepcontrol.wait` before their first global read and trigger
`griddepcontrol.launch_dependents` after their last global write.

### Routing (`RoutingArgs`, `RoutingKernelSpec`)

Renormalize top-k (top-k over the logits, then softmax over the selected
scores) with routed experts only. Inputs: `routing_logits` `[T, num_experts]`
(`logits_dtype` selects the kernel), `num_experts`, `top_k`,
`local_expert_offset`, `local_num_experts`, `tile_tokens_dim`, `max_num_ctas`.
Outputs (the trtllm-gen routing tables, byte for byte): `topk_packed`
`[T, top_k]` (packed bf16 score / int16 expert as the native router stores it),
`topk_weights` bf16 `[T, top_k]`, `expert_count_histogram` (scratch,
`max(2 * num_experts, 512)` int32), `total_num_padded_tokens[1]`,
`expanded_idx_to_permuted_idx[T * top_k]` (-1 for non-local experts),
`permuted_idx_to_token_idx[max_padded + 1]` (-1 in padded slots),
`cta_idx_xy_to_batch_idx[max_ctas]`, `cta_idx_xy_to_mn_limit[max_ctas]`,
`num_non_exiting_ctas[1]`, and `num_tokens_per_expert[num_experts]` when the
pointer is non-null. The host selects the first table entry whose
`logits_dtype` matches and whose `[min_tokens, max_tokens]` contains `T`; the
grid is either fixed (`grid_rule = kFixed`, cluster kernels) or
`ceil(T / tokens_per_cta)` blocks. A kernel with `writes_benign_tail` fills
`[num_non_exiting_ctas, max_num_ctas)` with expert 0 / `mn_limit = tile_idx *
tile` / slot -1 so the GEMM stages never launch the routing-tail kernel.

Rejected (not supported, no fallback): every other `RoutingMethodType`,
pre-computed expert ids, fused shared experts, routing bias, routing scales on
the input, DeepSeek FP8, `routing_replay_out`, `permuted_idx_to_expanded_idx`
(the GEMM1 Mn-bias row map).

### FC1 (`Fc1Args`, `Fc1KernelSpec`)

GEMM1 over the gathered activations with the fused SwiGLU-Step epilogue
(per-expert `clamp_limit`). Families: `nvfp4` (E2m1 in, E2m1 out with 8x4
block scales), `nvfp4_bf16tok` (E2m1 in with fp32 per-token scales, bf16 out),
`bf16` (BlockMajorK weights), `fp8` (per-tensor E4m3, raw-unit clamp), `mxfp8`
(UE8M0 block scales). Operand views, by family:

| family | weights `A` | weight scales `SFA` | activations `B` | activation scales `SFB` | output `C` |
|---|---|---|---|---|---|
| `nvfp4`, `nvfp4_bf16tok` | `[E, 2I, H/2]` | `[E * grid_m, H/64, 2, 256]` | `[T, H/2]` | `[T, H/16]` | `[max_padded, I/2]` (+ `SFC`) / `[max_padded, I]` bf16 |
| `bf16` | `[E, H/64, 2I, 64]` | - | `[T, H]` | - | `[max_padded, I]` |
| `fp8` | `[E, 2I, H]` | - | `[T, H]` | - | `[max_padded, I]` |
| `mxfp8` | `[E, 2I, H]` | `[E * grid_m, H/128, 2, 256]` | `[T, H]` | `[T, H/32]` | `[max_padded, I]` (+ `SFC`) |

Grid `(grid_m = 2I / output_rows_per_cta, grid_n = max_ctas)`; the routing arrays
are `route_map = permuted_idx_to_token_idx`, `tile_expert =
cta_idx_xy_to_batch_idx`, `tile_mn_limit = cta_idx_xy_to_mn_limit`,
`num_non_exiting_ctas`. Kernels with `bounds_acquired_tiles == false` get the
benign routing tail written by `cake_stepfun_routing_tail_kernel` before the
launch (unless the routing kernels of the module write it).

### Requantization (`RequantArgs`, `RequantKernelSpec`)

`nvfp4_bf16tok` only: the bf16 FC1 output `[max_padded, I]` becomes packed E2m1
`[max_padded, I/2]` plus E4m3 block scales in `sf_layout` (what the selected FC2
kernel reads for its activation operand, `Fc2Runner::sfLayoutA`) and fp32
per-token scales `[max_padded]`, using the NVFP4 recipe of the forward
(`global_scale_inv`, `e4m3_max`). Rows are visited through
`expanded_idx_to_permuted_idx[T * top_k]`; grid `ceil(T * top_k / rows_per_cta)`.
Same numerics as `invokeNvfp4QuantAndPerTokenScale<__nv_bfloat16>`.

### FC2 (`Fc2Args`, `Fc2KernelSpec`)

GEMM2 over the permuted FC1 output, bf16 output in permuted order
`[max_padded, H]`. Families mirror FC1 with the roles of `H` and `I` exchanged:

| family | weights `A` | weight scales `SFA` | activations `B` | activation scales `SFB` | extra |
|---|---|---|---|---|---|
| `nvfp4` | `[E, H, I/2]` | `[E * grid_m, I/64, 2, 256]` | `[max_padded, I/2]` | `[max_padded, I/16]` (`sf_layout_a`) | `scale_c` per expert |
| `nvfp4_bf16tok` | as `nvfp4` | as `nvfp4` | requantized rows | requantized scales (`sf_layout_a`) | `scale_c`, `per_token_scale[max_padded]` |
| `bf16` | `[E, I/64, H, 64]` | - | `[max_padded, I]` | - | - |
| `fp8` | `[E, H, I]` | - | `[max_padded, I]` | - | `scale_c` per expert |
| `mxfp8` | `[E, H, I]` | `[E * grid_m, I/128, 2, 256]` | `[max_padded, I]` | `[max_padded, I/32]` | - |

Grid `(grid_m = H / output_rows_per_cta, grid_n = max_ctas)`; routing arrays
`tile_expert`, `tile_mn_limit`, `total_tiles = num_non_exiting_ctas`,
`total_num_padded_tokens`; the same routing-tail rule as FC1. GEMM2 bias,
per-channel scales, output scales and valid (unpadded) dimensions smaller than
`H` / `I` are rejected.

### Finalize (`FinalizeArgs`, `FinalizeKernelSpec`)

Unpermute the bf16 FC2 output `[max_padded, hidden_dim_padded]` and reduce the
`top_k` experts of each token with the bf16 `expert_weights[T, top_k]` into
`output[T, hidden_dim]`, skipping `expanded_idx_to_permuted_idx == -1`. Two
variants with the native dispatcher's rule: the `scalar` kernel (grid
`(ceil(hidden_dim / 256), min(8192, T))`) when that grid has fewer than 1184
CTAs, the `vector` kernel (grid `(T)`, 128-bit loads, `top_k <= max_top_k`)
otherwise. DeepSeek FP8 (dequantization scales) is rejected.

## Standalone operations

`cake_stepfun_fc1_families()`, `cake_stepfun_fc1_tiles(family)`,
`cake_stepfun_fc1(...)` run on every module build; `cake_stepfun_stages()` lists
the stages of the build in pipeline order and `cake_stepfun_full_path()` tells the
variant. On the full path the module adds `cake_stepfun_fc2_tiles(family)`,
`cake_stepfun_fc2_activation_sf_layout(family, tile)`, `cake_stepfun_fc2(...)`,
`cake_stepfun_requant(...)` and `cake_stepfun_finalize(...)`; routing alone is
reachable through the module's `trtllm_moe_run_routing*` operations, which the
full path serves with the Cake router. The operands of every standalone
operation are the tensors `MoE::Runner` passes to the stage; see the binding's
doc comments.
