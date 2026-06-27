# moe_ep Design

Expert-Parallel MoE with two execution modes:

| Mode | Flow | When to use |
|------|------|-------------|
| **Split** | dispatch → local kernel → combine | Pluggable comm + compute; NCCL-EP / NIXL-EP transport |
| **Mega** | fused comm + MoE kernel | Single symmetric-memory kernel; no separate Fleet/Handle |

Entry point: `flashinfer.moe_ep.MoEEpLayer` (factory → `MoEEpSplitLayer` or `MoEEpMegaLayer`).

## Package layout

```
moe_ep/
  config.py, tensors.py, weights.py, layer.py   # shared types + factory
  algo_knobs.py                                  # split-path tuning knobs
  core/comm, core/kernel, core/runtime, core/validation
  backends/split/comm/{nccl_ep,nixl_ep}
  backends/split/kernel/{identity,fused_moe}
  backends/mega/kernel/{deep_gemm_mega,mxfp8_cutedsl,nvfp4_cutedsl}
  backends/mega/kernel/cutedsl_backend_kernels/{common,frontend,moe_mxfp8_glu,moe_nvfp4_swapab,src}
  modes/{split_layer,mega_layer,config}.py
```

Plugins register at import time: kernels via `@register_*` when `backends` is imported;
comm fleets when their `fleet.py` is imported from `__init__.py`.

## Core types

| Type | Role |
|------|------|
| `BootstrapConfig` | `world_size`, `rank`, `stream`, `nccl_comm`, `tcp_store`, `auto_bootstrap=True` |
| `FleetParams` | EP sizing + required `weights`; split transport fields (`algorithm`, `layout`, `dtype_bytes`) default and are ignored by mega |
| `MoEEpTensors` | `hidden_states`, `topk_ids`, `topk_weights`; optional `scales` (pre-staged mega) |
| `MoEWeightPack` | Canonical `w13` / `w2` (+ optional scales); `dummy_moe_weights()` for comm-only split |
| `SplitConfig` | `comm` + `kernel` slots (default `NcclEpConfig` + `IdentityConfig`) |
| `MegaConfig` | `megakernel`, `stage_inputs`, `preprocess_weights`, optional `transformed_weights` |

**Split:** pass `FleetParams` + `SplitConfig(comm=..., kernel=...)` or a comm string (kernel defaults to `IdentityConfig`). Fleet is lazy-created on first `forward()`; a new Handle per forward.

**Mega:** pass `FleetParams` + `MegaConfig(megakernel=...)`. Weights are required on `fleet_params`. Workspace allocated on first forward. Output is bf16 `[num_tokens, token_hidden_size]` where `num_tokens = MoEEpTensors.num_tokens` (may be `< max_tokens_per_rank`). `fleet_knobs` are ignored. NIXL-EP split layers require `BootstrapConfig.tcp_store` at init.

## Architecture

```mermaid
classDiagram
    direction TB

    MoEEpLayer --> MoEEpSplitLayer : SplitConfig
    MoEEpLayer --> MoEEpMegaLayer : MegaConfig

    MoEEpSplitLayer --> Fleet
    MoEEpSplitLayer --> SplitKernelBackend
    MoEEpSplitLayer --> Handle : per forward

    MoEEpMegaLayer --> MegaKernelBackend

    Fleet <|-- NcclEpFleet
    Fleet <|-- NixlEpFleet
    Handle <|-- NcclEpHandle
    Handle <|-- NixlEpHandle

    SplitKernelBackend <|-- FusedMoeSplitKernelBackend
    SplitKernelBackend <|-- IdentitySplitKernelBackend
    MegaKernelBackend <|-- DeepGemmMegaKernelBackend
    MegaKernelBackend <|-- Nvfp4CutedslMegaKernelBackend
    MegaKernelBackend <|-- Mxfp8CutedslMegaKernelBackend
```

## Forward flow

**Split:** `dispatch` → `kernel.compute(ctx)` → `combine` → destroy handle.

**Mega:** `prepare_workspace` → `stage_inputs` → `compute` → (workspace kept until `destroy()`).

Both paths call `ensure_moe_ep_cuda_device()` at init. With `auto_bootstrap=True` (default), layers acquire a ref-counted process runtime and release it in `destroy()`.

| Runtime need | Used by |
|--------------|---------|
| `torch_dist` | split comm, all mega kernels |
| `nvshmem` | `nvfp4_cutedsl`, `mxfp8_cutedsl` only (skip with `MEGA_NO_DIST=1`) |

## Built-in plugins

| Kind | Name | Config |
|------|------|--------|
| Comm | `nccl_ep` | `NcclEpConfig` (`NCCLEPConfig` alias) |
| Comm | `nixl_ep` | `NvepConfig` (needs `tcp_store`) |
| Split kernel | `identity` | `IdentityConfig` — comm-only; weights required on params but kernel ignores them (`dummy_moe_weights` OK) |
| Split kernel | `fused_moe` | `FusedMoeKernelConfig(moe_config=...)` — bridges to `flashinfer.fused_moe.MoELayer`; needs weights |
| Mega kernel | `deep_gemm_mega` | `DeepGemmMegaMoeConfig` — FP8/FP4, sm_100+ |
| Mega kernel | `nvfp4_cutedsl` | `Nvfp4CutedslMegaMoeConfig` — NVFP4, sm_100+ |
| Mega kernel | `mxfp8_cutedsl` | `Mxfp8CutedslMegaMoeConfig` — MXFP8, sm_100+ |

**Mega weights:** with `preprocess_weights=True` (default), canonical bf16 or logical pre-quantized `MoEWeightPack` is transformed at init. With `preprocess_weights=False`, supply `MegaConfig.transformed_weights` (kernel-ready layout from `preprocess_*_mega_weights`).

**Mega activations:** with `stage_inputs=True` (default), bf16 `[T, hidden]` is quantized into the symm workspace at forward. With `stage_inputs=False`, copy pre-quantized activations (+ `MoEEpTensors.scales`) into the workspace.

## Usage

```python
from flashinfer.fused_moe.api import MoEConfig
from flashinfer.moe_ep import (
    MoEEpLayer, BootstrapConfig, FleetParams, FleetParams,
    MoEEpTensors, MoEWeightPack, dummy_moe_weights,
    SplitConfig, NcclEpConfig, FusedMoeKernelConfig,
    MegaConfig, DeepGemmMegaMoeConfig,
)

# Split: NCCL-EP dispatch/combine + fused MoE compute
layer = MoEEpLayer(
    bootstrap=BootstrapConfig(world_size=4, rank=rank),
    fleet_params=FleetParams(
        num_experts=32, max_tokens_per_rank=256, token_hidden_size=2048,
        weights=MoEWeightPack(w13=w13_local, w2=w2_local),
    ),
    backend=SplitConfig(
        comm=NcclEpConfig(),
        kernel=FusedMoeKernelConfig(moe_config=moe_config),
    ),
)
out = layer.forward(MoEEpTensors(hidden_states=..., topk_ids=..., topk_weights=...))

# Split: comm-only identity (dummy weights)
layer = MoEEpLayer(
    bootstrap=BootstrapConfig(world_size=4, rank=rank),
    fleet_params=FleetParams(
        num_experts=32, max_tokens_per_rank=256, token_hidden_size=2048,
        weights=dummy_moe_weights(num_local_experts=8, hidden=2048),
    ),
    backend="nccl_ep",
)

# Mega: fused kernel (wrap megakernel config in MegaConfig)
layer = MoEEpLayer(
    bootstrap=BootstrapConfig(world_size=4, rank=rank),
    fleet_params=FleetParams(
        num_experts=32, max_tokens_per_rank=256, token_hidden_size=2048,
        weights=MoEWeightPack(w13=..., w2=...),
    ),
    backend=MegaConfig(megakernel=DeepGemmMegaMoeConfig(intermediate_size=1024, top_k=4)),
)
out = layer.forward(MoEEpTensors(...))
layer.destroy()
```

Under torchrun, dist/NVSHMEM init is automatic. Set `auto_bootstrap=False` when tests manage runtime themselves.

Useful exports: `bootstrap_moe_ep_runtime`, `finalize_moe_ep_runtime`, `ensure_moe_ep_cuda_device`, `preprocess_mega_weights`, `preprocess_nvfp4_cutedsl_mega_weights`, `preprocess_mxfp8_cutedsl_mega_weights`.

## Lifetimes

| Object | Created | Destroyed |
|--------|---------|-----------|
| Kernel backend | layer init | layer destroy |
| Process runtime | layer init (if `auto_bootstrap`) | layer destroy (ref-counted) |
| Fleet | first split forward | layer destroy |
| Handle | each split forward | end of forward |
| Mega workspace | first mega forward | layer destroy |

## Extending

1. **Split kernel** — `backends/split/kernel/<name>/`: subclass `SplitKernelBackend`, `@register_split_kernel`, import in `backends/split/kernel/__init__.py`.
2. **Mega kernel** — `backends/mega/kernel/<name>/`: subclass `MegaKernelBackend`, implement `compute` / `prepare_workspace` / `stage_inputs`, override `runtime_requirements()` if needed, `@register_mega_kernel`, import in `backends/mega/kernel/__init__.py`.
3. **Comm backend** (split only) — `backends/split/comm/<name>/` with `config.py`, `fleet.py`, `handle.py`; import fleet from `moe_ep.__init__.py`.

## Tests

`tests/moe_ep/run_tests.sh [unit|multirank|correctness|mega|smoke|all]`:

- **unit** — host-only pytest (no multirank)
- **multirank** — 4-GPU split path (NCCL-EP)
- **correctness** — 4-GPU LL/HT + NVFP4 numerics (Blackwell)
- **mega** — 4-GPU DeepGEMM + NVFP4 mega parity (Blackwell, sm_100+)
- **smoke** — quick NCCL-EP smoke scripts

Runtime bootstrap is exercised through layer construction, not direct front-end init calls.

## Sequence diagrams

### Split path

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Factory as MoEEpLayer
    participant Layer as MoEEpSplitLayer
    participant Runtime as bootstrap_moe_ep_runtime
    participant Kernel as SplitKernelBackend
    participant Fleet
    participant Handle

    Caller->>Factory: MoEEpLayer(..., SplitConfig)
    Factory->>Layer: __init__

    Layer->>Layer: ensure_moe_ep_cuda_device()
    Layer->>Kernel: create_split_kernel(config)
    opt auto_bootstrap
        Layer->>Runtime: bootstrap (torch_dist)
        Runtime-->>Layer: runtime handle
    end
    Layer->>Layer: validate fleet / arch
    opt kernel requires weights
        Layer->>Kernel: preprocess_weights(MoEWeightPack)
    end

    Caller->>Layer: forward(MoEEpTensors)
    Layer->>Layer: validate_split_forward_inputs()
    opt first forward
        Layer->>Fleet: create_fleet(bootstrap, fleet_params)
    end
    Layer->>Fleet: create_handle(topk_ids, knobs)
    Fleet-->>Handle: per-iteration handle

    Handle->>Handle: dispatch(hidden_states)
    Handle-->>Layer: DispatchOutput
    Layer->>Kernel: compute(SplitKernelContext)
    Kernel-->>Layer: expert_out
    Handle->>Handle: combine(expert_out)
    Handle->>Handle: complete()
    Handle->>Handle: destroy()
    Layer-->>Caller: combined hidden_states

    Caller->>Layer: destroy()
    Layer->>Fleet: destroy()
    opt auto_bootstrap
        Layer->>Runtime: finalize_moe_ep_runtime()
    end
```

### Mega path

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Factory as MoEEpLayer
    participant Layer as MoEEpMegaLayer
    participant Runtime as bootstrap_moe_ep_runtime
    participant Kernel as MegaKernelBackend

    Caller->>Factory: MoEEpLayer(..., MegaConfig)
    Factory->>Layer: __init__

    Layer->>Layer: ensure_moe_ep_cuda_device()
    Layer->>Kernel: create_mega_kernel(config)
    opt auto_bootstrap
        Layer->>Runtime: bootstrap (torch_dist [+ nvshmem])
        Runtime-->>Layer: runtime handle
    end
    Layer->>Kernel: validate_init()
    opt preprocess_weights
        Layer->>Kernel: preprocess_weights(MoEWeightPack)
        Kernel-->>Layer: transformed weights
    end

    Caller->>Layer: forward(MoEEpTensors)
    Layer->>Kernel: validate_forward()
    opt first forward
        Layer->>Kernel: prepare_workspace()
        Kernel-->>Layer: workspace
    end
    Layer->>Kernel: stage_inputs(tensors, workspace)
    Layer->>Kernel: compute(workspace, weights, output)
    Kernel-->>Layer: bf16 output
    Layer-->>Caller: [num_tokens, hidden]

    Caller->>Layer: destroy()
    Layer->>Kernel: destroy(workspace)
    opt auto_bootstrap
        Layer->>Runtime: finalize_moe_ep_runtime()
    end
```
