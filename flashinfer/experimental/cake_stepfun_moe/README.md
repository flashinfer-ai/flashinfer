# Experimental StepFun fused MoE (Cake backend, cake-build/1)

Both this API and its Cake backend are experimental and may change or be
removed without compatibility guarantees. Naming the backend
(`CakeStepFunConfig(backend="cake_stepfun")` in `flashinfer.fused_moe`) or
calling one of the package helpers is the explicit opt-in and emits
FlashInfer's experimental API warning once. There is no automatic backend
selection. Tracking: history in flashinfer-ai/flashinfer#6046 (the C++
stage-runner form this package supersedes); the tracking issue of the
experimental package is pending (none is opened yet; the PR checklist slot
stays unchecked until review decides).

The backend runs the StepFun fused MoE -- `SwiGLUStep(limit)` with a
per-expert `gemm1_clamp_limit`, `clamp(up, -L, L) * min(silu(gate), L)` -- on
exact SM100 (B200 / GB200) and SM103 (B300 / GB300). Every stage of the
forward is an exported Cake kernel: routing (`RoutingMethodType.Renormalize`
top-k, one or two launches), FC1 with the StepFun epilogue, the NVFP4
per-token requantization (per-token family only), FC2 and finalize. A forward
launches no trtllm-gen kernel.

| family          | MMA pair (weights x activations) | FC1 output / per-token scale |
| --------------- | -------------------------------- | ---------------------------- |
| `nvfp4`         | NVFP4 x NVFP4                    | E2m1 with block scales       |
| `nvfp4_bf16tok` | NVFP4 x NVFP4                    | bf16, fp32 per-token scale   |
| `bf16`          | BF16 x BF16 (BlockMajorK)        | bf16                         |
| `fp8`           | FP8 per-tensor x FP8 per-tensor  | E4m3                         |
| `mxfp8`         | MXFP8 x MXFP8                    | MxE4m3 with UE8M0 scales     |

The physical weight and activation layouts are exactly those of the matching
trtllm-gen config (`TrtllmFp4Config`, `TrtllmBf16Config`,
`TrtllmFp8PerTensorConfig`, `TrtllmFp8BlockConfig`); the `cake_stepfun` weight
view of `CakeStepFunConfig.prepare_weights` adds the per-expert fp32
`gemm1_clamp_limit` in the units the kernels clamp.

## Package layout

| Path | Content |
| --- | --- |
| `build.json` | cake-build/1 manifest: one entry `cake_stepfun_moe_<20hex>_<arch>` per program and architecture, its two sources, digests and closure |
| `cake_build.py` | the shared loader: prebuilt `aot/` library when present, else FlashInfer's JIT build of the manifest entry |
| `cake_jit.py` | the registry rendered by the export: `PROGRAMS` (one record per program with its argument plan, launch shape and stage metadata), `STAGE_TABLES` (the five dispatcher tables), `ROUTES`, `HOST_PLAN_CONSTANTS` |
| `cake_launch.py` | one positional launcher per program, rendered from its argument plan |
| `cake_backend.py` | host plan, validation, buffer set, binding and the prepared launches (hand-written) |
| `csrc/cake_stepfun_moe/` | the generated kernels (`<program>_kernel.cu`) and their TVM-FFI bindings (`<program>_binding.cu`), shared device / host headers |
| `csrc/cake_device_helpers/` | content-addressed device helper headers the kernels include |
| `csrc/cake_ffi_checks.h` | the binding runtime header |

The export writes `build.json`, `cake_build.py`, `cake_jit.py`,
`cake_launch.py` and everything under `csrc/`; `__init__.py`,
`cake_backend.py` and this README are hand-written. The package imports only
`torch`, `tvm_ffi` and its own modules (never `flashinfer.fused_moe`) and
reads no environment variable.

## Usage

Through the unified MoE API (the stable entry; `MoELayer` builds the runner
from the config):

```python
from flashinfer.fused_moe import (
    CakeStepFunConfig, MoEConfig, MoEWeightPack, MoEActivationPack, SwiGLUStep,
)

config = MoEConfig(routing=..., quant=..., experts=..., activation=SwiGLUStep(limit=7.0),
                   backend=CakeStepFunConfig(backend="cake_stepfun"))
view = CakeStepFunConfig.prepare_weights(w1, w2, quant=config.quant, num_local_experts=E,
                                         hidden_size=H, intermediate_size=I,
                                         activation=config.activation, step_limits=limits)
weights = MoEWeightPack(...); weights.prepare_for("cake_stepfun", view)
```

Directly on the package (prepare once, launch many; CUDA-graph capturable):

```python
from flashinfer.experimental.cake_stepfun_moe import prepare_stepfun_moe

launch = prepare_stepfun_moe(
    family="nvfp4", hidden_states=x_q, hidden_states_scale=x_sf,
    gemm1_weights=view["gemm1_weights"], gemm1_weights_scale=view["gemm1_weights_scale"],
    output1_scale_scalar=view["output1_scale_scalar"],
    output1_scale_gate_scalar=view["output1_scale_gate_scalar"],
    gemm1_clamp_limit=view["gemm1_clamp_limit"],
    gemm2_weights=view["gemm2_weights"], gemm2_weights_scale=view["gemm2_weights_scale"],
    output2_scale_scalar=view["output2_scale_scalar"],
    routing_logits=logits, num_experts=64, top_k=8, local_expert_offset=0, local_num_experts=64,
)
launch.launch()                  # eager once (binding smem opt-in), then capture / replay
output, = launch.outputs()       # [gemm2_output, expert_weights, expanded_idx_to_permuted_idx] with do_finalize=False
```

`prepare_<stage>` / `routing`, `fc1`, `requant`, `fc2`, `finalize` run one
stage on caller-owned buffers (the operand lists of the former standalone FFI
operations, as keywords).

## Host contract

* Tactic = the FC1 / FC2 tile. `plans_for(family, num_tokens, hidden_size,
  intermediate_size)` enumerates the tiles the package exports for the family
  in both stages and that pass the shape rules (`hidden_size % block_k == 0`,
  `intermediate_size % 64 == 0`, cluster divisibility, the FC2 K-tile
  geometry); `default_plan` is the planner's centre tile (the first exported
  tile `>= num_tokens * top_k / num_experts`, else the largest). A
  `(family, tile)` without an exported unit is not servable; nothing falls
  back to a native kernel.
* Buffers follow the Cake stage launcher: the routing tables in the router's
  sizes, `gemm1_output` (+ block scales), the requantization triple,
  `gemm2_output` `[max_padded, hidden_size]` bf16 and `output`
  `[num_tokens, hidden_size]` bf16, allocated once in `prepare_stepfun_moe`
  (`max_padded = max_num_ctas * tile`, the trtllm-gen row capacity).
* Launch attributes (cluster, cooperative, PDL) are baked into every binding;
  `enable_pdl` is a per-launch argument of the bindings. The cooperative
  routing grid is `SM count - reserved_sms` (`reserved_sms` keyword, default 8;
  the facade resolves `FLASHINFER_TRTLLM_MOE_OVERLAP_RESERVED_SMS` per call,
  where the native C++ reads it once per process).
* `do_finalize=False` returns `[gemm2_output, expert_weights,
  expanded_idx_to_permuted_idx]` in the native layout; the finalize launch is
  omitted at prepare time.
* Not supported: fused shared experts, LoRA deltas (`gemm1_lora_delta`),
  routing biases, packed pre-computed ids, routing methods other than
  Renormalize, NVFP4 4over6 recipes (only the standard `e4m3_max = 448`
  requantization unit exists).

## Migration from `csrc/fused_moe/cake_stepfun`

Supersedes #6046 (merged e39a1d22b, 2026-10-17T05:51Z): migrates the StepFun
backend to cake-build/1. The JIT modules `fused_moe_cake_stepfun_sm100` /
`_sm103` and `fused_moe_cake_stepfun_full_*`, `flashinfer/jit/cake_stepfun_moe.py`,
`csrc/fused_moe/cake_stepfun/**` and `tests/moe/test_cake_stepfun_fused_moe.py`
are replaced by the package `flashinfer.experimental.cake_stepfun_moe`
(`build.json` + `cake_build.py` + per-unit TVM-FFI bindings + Python stage
orchestration in `cake_backend.py`) and
`tests/experimental/test_cake_stepfun_moe{,_api}.py`. `CakeStepFunConfig` /
`CakeStepFunRunner` keep their names and import paths in `flashinfer.fused_moe`.
`FLASHINFER_CAKE_STEPFUN_FULL_PATH` is read only by the `flashinfer.fused_moe`
entry; the FC1-only hybrid of #6046 is removed, not migrated, so `0` raises
`NotImplementedError`. The FI-global `csrc/cake_device_helpers/` root is
unchanged except for the four files only StepFun used.
`CakeStepFunRunner.native_fc2_twin` and the `CAKE_STEPFUN_FC1` /
`CAKE_STEPFUN_FULL` hooks of the trtllm-gen host sources are removed.

### Limitations

* No FC1-only hybrid. The `FLASHINFER_CAKE_STEPFUN_FULL_PATH=0` path of #6046
  (the Cake FC1 inside the native trtllm-gen pipeline) is removed: `0` and
  `CakeStepFunConfig(full_path=False)` raise `NotImplementedError`. The
  per-family error table therefore has no Cake-vs-native `fc1` rows (`n/a`);
  the Cake FC1 stage is compared directly against the fp32 reference by the
  backend suite and covered indirectly through the FC2 inputs and the
  whole-forward comparison of the API suite.
* `CakeStepFunRunner.native_fc2_twin` is removed; the Cake FC2 stage is
  compared against the native FC2 kernel on the same permuted input by the
  API suite instead.
* LoRA deltas, fused shared experts, routing biases, packed pre-computed ids,
  routing methods other than Renormalize and NVFP4 4over6 recipes are
  rejected (see "Host contract").

## Tests

`tests/experimental/test_cake_stepfun_moe.py` is the backend suite (imports the
package only through `tests/test_helpers/cake_package.py`);
`tests/experimental/test_cake_stepfun_moe_api.py` tests the
`flashinfer.fused_moe` route against the native trtllm-gen kernels and writes
the per-family error table; its `fc1` Cake-vs-native rows are `n/a` (no
FC1-only native output exists without the hybrid), so the FC1 stage is gated
by the backend suite against the fp32 reference. Both are delivered by the
export together with the generated programs and skip without an SM100 / SM103
device.
