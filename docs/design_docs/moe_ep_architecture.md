# moe_ep Design

> For build/test/how-to-extend instructions, see the
> [moe_ep runbook](./moe_ep_runbook.md).
> For the CuTeDSL mega backends' tuning surface, measured performance, and
> benchmark methodology, see
> [kernel_src/cutedsl_megamoe/TUNING.md](../../flashinfer/moe_ep/kernel_src/cutedsl_megamoe/TUNING.md).

Expert-Parallel MoE with two execution modes:

| Mode | Flow | When to use |
|------|------|-------------|
| **Split** | dispatch → inner kernel → combine | Pluggable comm + compute; NCCL-EP / NIXL-EP transport |
| **Mega** | fused comm + MoE kernel | Single symmetric-memory kernel; no separate Fleet/Handle |

Entry point: `MoEEpLayer(bootstrap, fleet_params, weights, fleet_knobs=(), backend=...)` → `MoEEpSplitLayer` or `MoEEpMegaLayer`.

## Layout

```
moe_ep/
  config.py, tensors.py, weights.py, layer.py, algo_knobs.py, errors.py
  core/comm, core/kernel, core/runtime, core/validation, core/bootstrap_utils.py
  backends/split/comm/{nccl_ep,nixl_ep}
  backends/split/kernel/{identity,fused_moe}
  backends/mega/kernel/sm100/{nvfp4_nvfp4_bf16_cutedsl,mxfp8_mxfp8_bf16_cutedsl,fp8_fp4_bf16_deepgemm}
  backends/mega/kernel/sm90/{fp8_fp8_bf16_pull_cutedsl,fp8_fp8_bf16_push_cuda}
  kernel_src/cutedsl_megamoe/  ← Blackwell CuTeDSL kernel src (kernel team) + FI shim
    src/                       ← VERBATIM kernel team drop (common, moe_nvfp4_swapab, moe_mxfp8_glu, src)
    __init__.py                ← public API consumed by the sm100 cutedsl backends
    shim/                      ← thin adapters over src/ (_paths, comm, nvfp4, mxfp8, kernel_helpers, correctness, autotune, tuner)
    SKILL.md                   ← how to resync src/ when kernel team drops a new version
    TUNING.md                  ← tuning surface, measured perf, benchmark methodology
    ACKNOWLEDGEMENT.md         ← kernel authors
  kernel_src/sm90/pull_style_cutedsl_megakernel/  ← Hopper pull-style FP8 kernel src + FI shim
    src/                       ← VERBATIM drop, fork of the sm100 kernel repo (common, src, moe_nvfp4_swapab, moe_hopper_fp8)
    shim/, __init__.py, SKILL.md  ← same layering; process-exclusive with the sm100 tree (module names collide)
  kernel_src/sm90/push_style_megamoe/  ← Hopper push-style FP8 (raw CUDA, JIT-compiled)
    src/{a2a,fp8_gemm}/        ← VERBATIM drop from flashinfer PR #4069 (.cu/.cuh)
    shim/, __init__.py, VENDOR.md  ← shim is part of the upstream PR here (vendored with it)
  modes/{split_layer,mega_layer,config}.py
```

Layout rule — taxonomy vs provenance:

- `backends/mega/kernel/` is organized by **taxonomy** (the user view):
  `sm<arch>/<act_dtype>_<weight_dtype>_<out_dtype>_<kernel_style>/`. Backends are
  thin adapters; several may wrap kernels from the same vendored repo.
- `kernel_src/` is organized by **provenance** (the kernel-dev view): one
  directory per upstream kernel repo snapshot, mirroring the vendor repo, with
  `src/` verbatim, all adaptation in `shim/`, and a `VENDOR.md` recording the
  upstream repo, pinned commit, sync date, and any pending local diffs. Never
  hand-merge two upstream states into one directory. (`kernel_src/sm90/…` is a
  separate snapshot of a fork and keeps its current path for now; fold it into
  `cutedsl_megamoe/` if upstream merges the SM90 kernel.)

Kernels register via `@register_split_kernel` / `@register_mega_kernel` when `backends` is imported; comm fleets register when their `fleet.py` is imported from `__init__.py`.

## Core types

| Type | Role |
|------|------|
| `BootstrapConfig` | `world_size`, `rank`, `stream`, `nccl_comm`, `tcp_store`, optional `process_group` (EP comm; defaults to WORLD), `auto_bootstrap=True` |
| `FleetParams` | EP sizing only (no weights); split transport fields (`algorithm`, `layout`, `dtype_bytes`) default and are ignored by mega |
| `MoEEpTensors` | `hidden_states`, `topk_ids`, `topk_weights`; optional `scales`, `fc1_alpha`, `fc2_alpha`, `fc1_norm_const`, `recv_count`, `num_tokens_per_expert` |
| `MoEWeightPack` | Canonical `w13` / `w2` (+ optional `w13_scale` / `w2_scale`); required `weights` arg at layer construction; `dummy_moe_weights()` for comm-only split |
| `SplitConfig` | `comm` + `kernel` slots (default `NcclEpConfig` + `IdentityConfig`) |
| `MegaConfig` | `megakernel`, `quantize_input`, `preprocess_weights`, optional `transformed_weights` |
| `FleetAlgoKnobFaultTolerance` | Opt-in rank masking (`enabled`, `timeout_ms`, reconcile budgets) — see **Fault tolerance** |

**Split:** pass `SplitConfig(comm=..., kernel=...)` or a comm string/config (kernel defaults to `IdentityConfig`). `fleet_knobs` tune transport. Fleet is lazy-created on first `forward()`; a new Handle per forward. `MoEEpSplitLayer.enable_timing` optionally records per-stage GPU ms in `last_timings_ms`.

**Split compute:** the `fused_moe` kernel bridges the 3D EP dispatch buffer to `flashinfer.fused_moe` (a token-major `MoEActivationPack`) via `backends/split/kernel/fused_moe/bridge.py`:

- LL **EXPERT_MAJOR** — `[num_local_experts, cap, hidden]` (`cap = max_tokens_per_rank * world`), each row pre-assigned to one expert; the bridge synthesizes `top_k=1` / `final_scales=1` and **combine owns the real top-k reweight**.
- LL **RANK_MAJOR** / **HT FLAT** — `[world, max_tokens_per_rank, hidden]` carrying received `topk_idx` / `topk_weights`; the runner uses the real `top_k` with non-local picks masked to weight 0, and combine just sums across ranks.

Both bf16 and NVFP4 are supported in the compute path (`MoEConfig.quant.variant`); NVFP4 activations are quantized in the bridge (linear SF layout). Correctness is currently validated for bf16 only (see **Tests**).

**Mega:** pass `MegaConfig(megakernel=...)`. Weights required as the layer's `weights` argument. Workspace allocated on first forward. Output is bf16 `[num_tokens, token_hidden_size]` where `num_tokens = MoEEpTensors.num_tokens` (may be `< max_tokens_per_rank`). `fleet_knobs` are ignored. NIXL-EP split layers require `BootstrapConfig.tcp_store` at init.

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

## Built-in plugins

| Kind | Name | Config |
|------|------|--------|
| Comm | `nccl_ep` | `NcclEpConfig` (`NCCLEPConfig` alias) |
| Comm | `nixl_ep` | `NvepConfig` (needs `tcp_store`) |
| Split kernel | `identity` | `IdentityConfig` — comm-only; `dummy_moe_weights` OK |
| Split kernel | `fused_moe` | `FusedMoeKernelConfig(moe_config=...)` — bridges to `flashinfer.fused_moe`; bf16 + NVFP4; LL EXPERT_MAJOR / RANK_MAJOR / HT FLAT |
| Mega kernel | `sm100_fp8_fp4_bf16_deepgemm` | `Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig` — FP8/FP4, sm_100+ |
| Mega kernel | `sm100_nvfp4_nvfp4_bf16_cutedsl` | `Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig` — NVFP4, sm_100+ |
| Mega kernel | `sm100_mxfp8_mxfp8_bf16_cutedsl` | `Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig` — MXFP8 (`kind` e4m3/e5m2), sm_100+ |

**Mega weights:** with `preprocess_weights=True` (default), canonical bf16 or pre-quantized `MoEWeightPack` is transformed at init. With `preprocess_weights=False`, supply `MegaConfig.transformed_weights` (from `preprocess_*_mega_weights`).

**Mega activations:** with `quantize_input=True` (default), bf16 `[T, hidden]` is quantized into symm workspace at forward. Non-bf16 with `quantize_input=True` raises `MoEEpConfigError`; use `quantize_input=False` and pre-quantized activations plus `MoEEpTensors.scales`.

## Runtime

Both paths call `ensure_moe_ep_cuda_device()` at init. With `auto_bootstrap=True` (default), layers acquire a ref-counted process runtime and release it in `destroy()`.

| Requirement | Used by |
|-------------|---------|
| `torch_dist` | split comm, all mega kernels |
| `nvshmem` | `sm100_nvfp4_nvfp4_bf16_cutedsl`, `sm100_mxfp8_mxfp8_bf16_cutedsl` (skip with `MEGA_NO_DIST=1`) |

**Host framework bootstrap (e.g. vLLM):** when the host already initialized `torch.distributed` and EP uses a subgroup, pass `BootstrapConfig(process_group=ep_group, world_size=ep_size, rank=ep_rank, auto_bootstrap=False)` and call `bootstrap_moe_ep_runtime(bootstrap, reqs)` once per worker after dist init. Mega kernels resolve comm via `bootstrap_comm_group` / `bootstrap_ep_rank_world` (`MegaKernelBackend.bind_ep_bootstrap`).

When `auto_bootstrap=False`: dist must be up at layer construction if `process_group` is set; rank/world cross-checks run at init or first `forward()`; call `bootstrap_moe_ep_runtime` yourself.

## Build / availability

Split comm backends ship native libs under `backends/split/comm/*/_libs/`. Probe with `have_nccl_ep()`, `have_nixl_ep()`, `available_backends()`. Missing libs raise `MoEEpNotBuiltError`.

**Recommended build:** `docker/install/build_flashinfer_ep_pytorch.sh` builds the full NCCL-EP + Mega environment inside the NVIDIA PyTorch base image (`nvcr.io/nvidia/pytorch`): it pins the NCCL-EP runtime wheels (`nvidia-nccl-cu13`, `nccl4py`, `cuda-core`, `cuda-bindings`), installs the mega deps (DeepGEMM, NVSHMEM, CUTLASS DSL), then runs `BUILD_NIXL_EP=0 pip install --no-build-isolation -e .`. The EP backends are ON by default: NCCL-EP needs no build step (`nccl4py>=0.3.1` is a base dependency), and the NIXL-EP meson build runs best-effort unless opted out with `BUILD_NIXL_EP=0` (set `BUILD_NIXL_EP=1` to make missing build deps a hard error; `BUILD_NVEP=0` turns both backends off).

## Lifetimes

| Object | Created | Destroyed |
|--------|---------|-----------|
| Kernel backend | layer init | layer destroy |
| Process runtime | layer init (if `auto_bootstrap`) | layer destroy (ref-counted) |
| Fleet | first split forward | layer destroy |
| Handle | each split forward | end of forward |
| Mega workspace | first mega forward | layer destroy |

## Usage

```python
from flashinfer.moe_ep import (
    MoEEpLayer, BootstrapConfig, FleetParams, MoEEpTensors,
    MoEWeightPack, SplitConfig, NcclEpConfig, FusedMoeKernelConfig,
    MegaConfig, Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig,
)

# Split: NCCL-EP + fused MoE
layer = MoEEpLayer(
    bootstrap=BootstrapConfig(world_size=4, rank=rank),
    fleet_params=FleetParams(num_experts=32, max_tokens_per_rank=256,
        token_hidden_size=2048),
    weights=MoEWeightPack(w13=..., w2=...),
    backend=SplitConfig(comm=NcclEpConfig(),
        kernel=FusedMoeKernelConfig(moe_config=moe_config)),
)
out = layer.forward(MoEEpTensors(hidden_states=..., topk_ids=..., topk_weights=...))

# Mega: wrap megakernel config in MegaConfig
layer = MoEEpLayer(..., backend=MegaConfig(
    megakernel=Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig(intermediate_size=1024, top_k=4)))
out = layer.forward(MoEEpTensors(...))
layer.destroy()
```

Raw megakernel or split-kernel configs cannot be passed as `backend=`; wrap in `MegaConfig` / `SplitConfig`.

## Extending

See the [runbook's mega-kernel walkthrough](./moe_ep_runbook.md#adding-a-new-mega-kernel-backend) for a step-by-step example (frontend contract, config, registration).

1. **Split kernel** — `backends/split/kernel/<name>/`: subclass `SplitKernelBackend`, `@register_split_kernel`, import in `backends/split/kernel/__init__.py`.
2. **Mega kernel** — `backends/mega/kernel/sm<arch>/<act>_<weight>_<out>_<style>/`: subclass `MegaKernelBackend`, implement `compute` / `_allocate_workspace` / `stage_inputs`, override `runtime_requirements()` if needed, `@register_mega_kernel`, import in `backends/mega/kernel/__init__.py`.
3. **Comm backend** (split only) — `backends/split/comm/<name>/` with `config.py`, `fleet.py`, `handle.py`; import fleet from `moe_ep.__init__.py`.

## Tests

See the [runbook's build & test section](./moe_ep_runbook.md#build--test-environment) for the container setup and per-target requirements.

`tests/moe_ep/run_tests.sh [unit|oracle|oracle_sm90|multirank|split_path_correctness_{bf16,nvfp4,ht}|mega|mega_sm90|smoke|ft|all]`:

- **unit** — host-only pytest (mocks + single-GPU; no multirank)
- **oracle** — single-GPU torch-oracle correctness for every SM100 compute path (see **Torch oracles** below)
- **oracle_sm90** — single-GPU (Hopper) torch oracle for the sm90_fp8_fp8_bf16_pull_cutedsl mega kernel
- **multirank** — 4-GPU split path: `test_moe_ep_layer_multirank.py` + `test_split_kernels.py` over NCCL-EP (and NIXL-EP when built)
- **split_path_correctness_{bf16,nvfp4,ht}** — 4-GPU split-path numerics (LL EXPERT_MAJOR + RANK_MAJOR / NVFP4 / HT FLAT) vs a single-process `MoELayer` reference (Blackwell)
- **mega** — 4-GPU DeepGEMM + NVFP4 + MXFP8 mega parity **and multi-rank torch oracles**, plus single-rank preprocess/kernel-vs-reference checks (`MEGA_NO_DIST=1`) (Blackwell, sm_100+)
- **mega_sm90** — 4-GPU (Hopper) sm90_fp8_fp8_bf16_pull_cutedsl mega parity + multi-rank torch oracle; own torchrun process (the SM90/SM100 kernel trees share top-level module names and are mutually exclusive per process)
- **smoke** — NCCL-EP smoke script (and NIXL-EP when built)
- **ft** — 4-GPU fault-tolerance (stalled-rank pytest half + dead-rank smoke half)

`all` runs the eight Blackwell-relevant sections (everything above except the two `*_sm90` targets, which need Hopper).

Multirank/smoke/correctness need the NCCL-EP build (see **Build / availability** — `docker/install/build_flashinfer_ep_pytorch.sh`); mega additionally needs Blackwell, deep_gemm, triton.

## Torch oracles

Every compute path is anchored to plain-torch ground truth, at two levels.

**Why they exist.** The other correctness tests are *parity* tests: layer path
vs direct-shim path, or EP vs non-EP. For the split paths that is a genuine
cross-check (dispatch/combine and the compute kernel are separate stages, and
the non-EP side exercises different communication code). For the **mega**
paths it is not: communication and compute are fused into one kernel, so both
sides of a parity test run the same CUDA kernel and a kernel that is *wrong
but self-consistent* passes silently. Worse, bit-exact layer-vs-shim tests
share the preprocessing on both sides, so wrong-preprocess bugs also pass
silently (this class produced three real bugs on the day the single-GPU
oracles landed: nvfp4 weight strides, nvfp4 norm_const, split-bf16 gated
reorder). The oracles close both gaps.

**What an oracle does.**

- *Single-GPU oracles* (`run_tests.sh oracle` / `oracle_sm90`): stage inputs
  exactly as the layer does, launch the kernel once on one rank, and recompute
  the output with plain torch — dequant → fp32 `@` GEMMs → SwiGLU
  (`torch.sigmoid`) → the path's exact fc1-out quant round-trip → fp32 fc2 →
  topk combine — asserting a tight band (typically `rel_l2 < 0.02`; residual
  is quant RTNE flips + accumulation-order noise, since both sides consume the
  same quantized operands).
- *Multi-rank oracles* (inside `mega` / `mega_sm90`, 4 GPUs, names
  `test_moe_ep_*_mega_multirank_torch_oracle`): each rank stages its own
  shard and launches the fused kernel with **real cross-rank NVSHMEM
  traffic**, then `all_gather`s the *actual* operands the kernel consumed —
  the plain (pre-swizzle) quantized weight legs of every rank, and for
  mxfp8/sm90 also the staged activation payloads + routing — and recomputes
  its own output slice with torch math over the **global** expert set
  (`y[t] = Σ_k w_k · expert_{id_k}(x_t)` is per-token math, so local tokens
  vs global weights is full ground truth). The all-gather is evidence
  collection, not part of the math: it removes any reliance on cross-rank RNG
  determinism. Token 0 is force-routed to one expert per rank so cross-rank
  traffic exists by construction. This is the only test where the multi-rank
  numerics (peer-pull addressing, expert→rank ownership, peer-token dequant,
  cross-rank combine) are judged by something other than the kernel itself.
- *Variant coverage*: the nvfp4 multirank oracle is parametrized over
  `in_kernel_fc2_reduce` and the quantized combine wires (`16e2m1xbf16`,
  `32e4m3xe8m0`; the reference models the wire exactly via
  `combine_roundtrip_to_fp32` per fc2 term); mxfp8 over
  `in_kernel_fc2_reduce`; sm90 over `per_tensor`/`blockwise` × `swap_ab`.
  ikr variants keep the explicit-reduce reference and widen the band by the
  bf16 K-term accumulation bound (`_assert_ikr_close`).

**Independence contract.** No oracle executes the kernel under test or any
reference *device* kernel — GEMMs are torch fp32, elementwise is torch/triton
host code. What is *deliberately shared* is the host-side quantization recipe
(`nvfp4_quantize_per_block_16`, `per_token_cast_to_fp8/fp4`,
`combine_roundtrip_to_fp32`), so the oracle consumes the kernel's exact
operands and the band stays tight; bugs inside those shared helpers are
covered separately by the preprocess-vs-plain-quant tests (two independent
quant implementations compared bit-exactly). Provenance caveat: the
nvfp4/deep_gemm references live in the test files; the mxfp8/sm90 references
(`compute_megamoe_reference_{mxfp8,fp8}`) ship with the kernel drop —
independent in execution, same-team in authorship.

**Last passed** (update when re-validated; single-GPU + multi-rank oracle
unless noted):

| Kernel path | Last passed | Where |
|---|---|---|
| split trtllm bf16 (LL + HT share the compute kernel) | 2026-07-31 | `run_tests.sh all`, 4x GB200 (oracle + EP-vs-non-EP) |
| split trtllm nvfp4 | 2026-07-31 | same run |
| mega deep_gemm (fp8_fp4) | 2026-07-31 | same run + variant job (multirank oracle) |
| mega sm100_nvfp4_nvfp4_bf16_cutedsl (default, ikr, nvfp4/mxfp8 combine wires) | 2026-07-31 | 4x GB200 variant job |
| mega sm100_mxfp8_mxfp8_bf16_cutedsl (default, ikr) | 2026-07-31 | 4x GB200 variant job |
| mega sm90_fp8_fp8_bf16_pull_cutedsl (per_tensor/blockwise × swap_ab) | 2026-07-30 | Hopper, when landed (commit 7169aca9); not runnable on the SM100 cluster |

## Forward flow

### Split

```mermaid
sequenceDiagram
    participant Caller
    participant Layer as MoEEpSplitLayer
    participant Kernel as SplitKernelBackend
    participant Runtime
    participant Fleet
    participant Handle

    Note over Caller,Handle: Init
    Caller->>Layer: __init__
    Layer->>Layer: validate bootstrap / fleet / arch
    Layer->>Kernel: create_split_kernel
    opt auto_bootstrap
        Layer->>Runtime: bootstrap (torch_dist)
    end
    Layer->>Kernel: validate_init
    opt requires weights
        Layer->>Kernel: preprocess_weights
    end

    Note over Caller,Handle: Forward
    Caller->>Layer: forward(tensors)
    Layer->>Layer: validate inputs
    opt first forward
        Layer->>Fleet: create_fleet
    end
    Layer->>Fleet: create_handle
    Handle->>Handle: dispatch
    Layer->>Kernel: inner_compute
    Handle->>Handle: combine, complete
    Handle->>Handle: destroy
    Layer-->>Caller: combine.x

    Note over Caller,Handle: Destroy
    Caller->>Layer: destroy()
    Layer->>Fleet: destroy
    opt auto_bootstrap
        Layer->>Runtime: finalize
    end
```

### Mega

```mermaid
sequenceDiagram
    participant Caller
    participant Layer as MoEEpMegaLayer
    participant Kernel as MegaKernelBackend
    participant Runtime

    Note over Caller,Runtime: Init
    Caller->>Layer: __init__
    Layer->>Layer: validate bootstrap / fleet / arch
    Layer->>Kernel: create_mega_kernel
    Layer->>Kernel: bind_ep_bootstrap
    opt auto_bootstrap
        Layer->>Runtime: bootstrap torch_dist + nvshmem
    end
    Layer->>Kernel: validate_init
    alt preprocess_weights
        Layer->>Kernel: preprocess_weights
    else transformed_weights at init
        Layer->>Kernel: validate_transformed_weights
    end

    Note over Caller,Runtime: Forward
    Caller->>Layer: forward(tensors)
    Layer->>Layer: validate inputs, quantize_input rules
    opt first forward
        Layer->>Kernel: prepare_workspace
    end
    Layer->>Kernel: stage_inputs
    Layer->>Kernel: compute, output y
    Layer-->>Caller: y

    Note over Caller,Runtime: Destroy
    Caller->>Layer: destroy()
    Layer->>Kernel: destroy(workspace)
    opt auto_bootstrap
        Layer->>Runtime: finalize
    end
```

## Fault tolerance

Opt-in per Fleet with `FleetAlgoKnobFaultTolerance()`. Without it, a peer that
stops responding during dispatch/combine trips a GPU `trap()` and takes the job
down; with it, both transports mask the offending rank, skip it, and let the
collective complete.

**LOW_LATENCY only, on both transports** — `nccl_ep` leaves its mask buffer
NULL under HIGH_THROUGHPUT (the mask APIs then abort the process) and `nixl_ep`
has no HT mask at all. `validate_fleet_params` rejects the combination.

### Mask convention

Canonical across the package: **`int32[world_size]`, `1 = active`, `0 = masked`**,
a CUDA tensor. This matches `ncclEpMaskQuery` and vLLM's `query_active_mask()`
naming, so `nccl_ep` is a pass-through.

`nixl_ep` differs on both axes and normalizes inside its own Fleet, so nothing
else in the tree ever sees its convention:

| | nccl_ep | nixl_ep |
|---|---|---|
| polarity | `1 = active` | **nonzero = masked** (buffer is `0xFF`-memset, so an untouched entry reads back as `-1`; kernels test `!= 0`) |
| length | `world_size` | topology **capacity** (`query_mask_buffer` asserts on it) |
| normalization | identity | `active = (raw[:world_size] == 0)` — **not** `1 - raw`, which yields `2` for never-connected tail ranks |

### Fleet API

| Method | Collective? | Blocks host? | Stream-ordered? |
|---|---|---|---|
| `supports_fault_tolerance` | — | no | no |
| `query_fault()` | local | nccl no / nixl yes (small D2H) | nccl no / nixl yes |
| `query_active_mask(out=None)` | local | no | **yes** |
| `set_active_mask(mask)` | local (but must be applied identically everywhere) | no | **yes** |
| `reconcile_active_mask()` | **store-collective**, tolerates dead ranks | yes (≤ timeout) | yes |
| `clear_faults(readmit=False)` | local | no | no |
| `clear_faults(readmit=True)` | **collective over survivors** | **yes** | yes |
| `active_mask_epoch` | — | no | no |

Reconciliation goes through the bootstrap rendezvous store
(`resolve_rendezvous_store(..., subsystem="ft")`), never a `torch.distributed`
allreduce: an allreduce over the EP group would hang on exactly the rank being
masked out. See `core/comm/fault_tolerance.py` and the runbook for the protocol
and the recovery state machine.
