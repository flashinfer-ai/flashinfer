# moe_ep Runbook

Practical how-to for building, testing, and extending the Expert-Parallel MoE
stack. For the design/architecture, see
[`moe_ep_architecture.md`](./moe_ep_architecture.md).

---

## Build & test environment

### Create the container

Tested on 1×4 GB200 GPUs.

Build the flashinfer-EP PyTorch image (pins NCCL-EP runtime wheels + mega deps:
DeepGEMM, NVSHMEM, CUTLASS DSL) into a `.sqsh`, then launch it:

```shell
export RW=/path/to/flashinfer/repo

# 1. Build the container image
srun --jobid="$SLURM_JOB_ID" -N1 \
  --container-image=nvcr.io/nvidia/pytorch:26.05-py3 \
  --container-save=$RW/flashinfer-ep-pt2605-mega_moe_ep.sqsh \
  --container-mounts=$RW:/host \
  bash -lc 'bash /host/flashinfer/docker/install/build_flashinfer_ep_pytorch.sh'

# 2. Launch an interactive shell in the saved image
export IMG=$RW/flashinfer-ep-pt2605-mega_moe_ep.sqsh
export ROOT=/workspace

srun --jobid="$SLURM_JOB_ID" \
  --overlap \
  --container-image="$IMG" \
  --container-mounts="$ROOT:$ROOT" \
  --container-workdir="$ROOT/flashinfer" \
  --pty bash -l

# 3. (Re)build FlashInfer in editable mode (EP backends are on by default;
#    NCCL-EP needs no build step — nccl4py is a base dependency)
BUILD_NIXL_EP=0 \
    pip install --no-cache-dir --no-build-isolation -e .
```

To exercise the NIXL-EP tests, drop `BUILD_NIXL_EP=0` (best-effort build) or set
`BUILD_NIXL_EP=1` (strict — missing build deps abort the install). See
[Running the NIXL-EP tests](#running-the-nixl-ep-tests).

Build flags (tri-state; unset = on, best-effort): `BUILD_NIXL_EP=0` skips the
NIXL-EP meson build, `BUILD_NIXL_EP=1` makes its missing build deps a hard
error, `BUILD_NVEP=0` turns both backends off. Probe availability at runtime
with `have_nccl_ep()`, `have_nixl_ep()`, `available_backends()`.

### Run tests

`tests/moe_ep/run_tests.sh <target>` — targets and requirements:

| Command | GPUs | Requires |
|---------|------|----------|
| `bash tests/moe_ep/run_tests.sh unit` | 1 (host-only) | none — mocks + single GPU, no multirank |
| `bash tests/moe_ep/run_tests.sh multirank` | 4 | NCCL-EP (NIXL-EP too if built) |
| `bash tests/moe_ep/run_tests.sh split_path_correctness_bf16` | 4 | Blackwell |
| `bash tests/moe_ep/run_tests.sh mega` | 4 | Blackwell sm_100+; DeepGEMM + NVFP4 + MXFP8 |

- **unit** — host-only pytest (mocks + single-GPU).
- **multirank** — 4-GPU split path over NCCL-EP (and NIXL-EP when built).
- **split_path_correctness_bf16** — 4-GPU bf16 split-path numerics vs a
  single-process `MoELayer` reference.
- **mega** — 4-GPU DeepGEMM + NVFP4 + MXFP8 mega parity, plus a single-rank
  MXFP8 preprocess-vs-reference check.

`all` and `smoke` targets also exist. Split-path numerics are **bf16-only** for
now.

### Running the NIXL-EP tests

NIXL-EP is the second split-path transport (`backend="nixl_ep"`), currently
**low-latency + `EXPERT_MAJOR` only**. Constraints enforced by
`validate_fleet_params`: `max_tokens_per_rank ≤ 1024`, `token_hidden_size ∈
{2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192}`, torch built for CUDA ≥ 13,
sm_90+.

**1. Build with NIXL-EP enabled.** Build deps: `meson`, `ninja`, `pkg-config`,
`nvcc`, UCX (`pkg-config --exists ucx`), `libibverbs-dev`. The build hook
pre-installs the `nixl-cu13` wheel it links against (`_ensure_nixl_wheel`), so
no manual NIXL install is needed.

**UCX caveat:** the UCX found via pkg-config must ship the *device API*
(`ucp/api/device/ucp_device_impl.h`) — NGC images' HPC-X UCX and Ubuntu's apt
UCX both predate it, and the `flashinfer-ep-pt2605*` images skip this
provisioning. Follow `docker/Dockerfile.flashinfer-nvep`: install DOCA 3.2
host packages + GDRCopy, then build UCX `v1.21.x` from source with
`--enable-experimental-api --with-cuda --with-verbs --with-dm` and put its
`lib/pkgconfig` first on `PKG_CONFIG_PATH`. Without it, `BUILD_NIXL_EP=1`
fails compiling `nixl_device.cuh`:

```shell
BUILD_NIXL_EP=1 pip install --no-cache-dir --no-build-isolation -e .

# Verify the backend actually built (best-effort builds skip it silently):
python -c "from flashinfer.moe_ep import available_backends; print(available_backends())"
# expect: [..., 'nixl_ep']
```

**2. Smoke test** (4 GPUs, single node — fixed shape: 64 tokens, 8 experts,
hidden 4096, topk 4, bf16, LL):

```shell
torchrun --nproc_per_node=4 tests/moe_ep/smoke_nixl_ep.py
# each rank prints: SMOKE_RESULT: nixl_ep OK
```

**3. Multirank roundtrip + split kernels** (4 GPUs). The `multirank` target
runs NCCL-EP first, then repeats over NIXL-EP when `have_nixl_ep()` is true:

```shell
bash tests/moe_ep/run_tests.sh multirank
```

or NIXL-only, directly:

```shell
torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_layer_multirank.py -v \
    -m "nvep and gpu_4" --backend=nixl_ep
torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_split_kernels.py -v \
    -m "nvep and gpu_4" --backend=nixl_ep
```

**4. Host-only mock tests** (no GPU fabric / no NIXL build needed) are part of
the `unit` target — `tests/moe_ep/nixl_ep/test_fleet_mock.py` stubs the NIXL
`Buffer` and checks fleet sizing, `update_topology` rank diffs, and combine
knob validation.

Notes:

- **Rendezvous**: unlike NCCL-EP (which mirrors the torch process group), the
  NIXL `Buffer` rendezvous needs a `torch.distributed.TCPStore` passed via
  `BootstrapConfig(tcp_store=...)`. The tests open one themselves on
  `MASTER_PORT + 1` (sharing torch's port would clash) — see
  `tests/moe_ep/smoke_nixl_ep.py` for the pattern.
- `NPROC_SMOKE` / `NPROC_MULTIRANK` (default 4) override the rank count for
  the `run_tests.sh` targets.
- UCX/ibverbs are **build-time** deps; the tests set no `NIXL_*`/`UCX_*` env.
- NIXL-EP coverage today is smoke + multirank + mocked unit tests only; the
  correctness/mega targets are NCCL-EP-only.

---

## Adding a new mega-kernel backend

A mega kernel owns fused comm + local MoE. To wire a new one, add a subpackage
under `flashinfer/moe_ep/backends/mega/kernel/<name>/`. The kernel sources
themselves live under
`flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/` and are exposed
through the `kernel_src/cutedsl_megamoe/` public API (e.g. `mxfp8_mega_moe`,
`get_symm_buffer_for_mxfp8_mega_moe`). Use the existing `mxfp8_cutedsl` backend
as the reference template.

### 1. Kernel + frontend (the "backend config" it links to)

Every mega kernel exposes exactly **two entry points** through a thin frontend,
and the `MegaKernelBackend` subclass links to nothing else. Keep this contract
stable so new kernels — including future **SM90 (Hopper)** and **SM120
(Blackwell-consumer)** variants — drop in behind the same backend shape without
touching `modes/` or the registry:

**(a) Workspace allocator** — problem sizes first, tuning knobs keyword-only;
returns a symm-buffer object with the staging views the backend fills and a
`.destroy()`. Model it on
`get_symm_buffer_for_mxfp8_mega_moe` / `get_symm_buffer_for_mega_moe`:

```python
def get_symm_buffer_for_<name>_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,        # == fleet_params.max_tokens_per_rank
    num_topk: int,
    hidden: int,                # fleet_params.token_hidden_size
    intermediate: int,          # post-SwiGLU width
    rank: int,                  # self.ep_rank
    world_size: int,            # self.ep_world_size
    *,
    kind=...,                   # dtype selector, if applicable
    # ... kernel knobs: clamps, in_kernel_fc2_reduce, token_back_by_dispatch, ...
) -> <Name>SymmBuffer: ...
```

The returned buffer must expose the staging tensors the backend's `stage_inputs`
writes — at minimum `x`, `x_sf` (quantized paths), `topk_idx`, `topk_weights` —
plus `destroy()`. Expert weights are **not** owned by the workspace; they are
passed to the compute call each launch.

**(b) Compute entry** — output tensor first, then the two kernel-ready
`(weight, scale)` weight tuples, the workspace, and keyword-only knobs. Model it
on `mxfp8_mega_moe` / `nvfp4_mega_moe`:

```python
def <name>_mega_moe(
    y: torch.Tensor,                  # bf16 [num_tokens, hidden] output
    transformed_l1,                   # (w13, w13_scale) kernel-ready fc1
    transformed_l2,                   # (w2, w2_scale)  kernel-ready fc2
    symm_buffer: <Name>SymmBuffer,
    *,
    num_tokens: int | None = None,
    # ... clamps, fast_math (accept for API parity even if a no-op), ...
) -> None: ...
```

`compute` fuses dispatch + fc1 + fc2 + combine and writes `y[:num_tokens]`. The
caller (the backend's `stage_inputs`) must have filled `symm_buffer.x` and the
routing slices first.

Add both functions under
`kernel_src/cutedsl_megamoe/shim/` (alongside `nvfp4.py` / `mxfp8.py`) and
re-export them from the package `__init__.py` (or point at your own kernel
module). Raw kernel sources live under `kernel_src/cutedsl_megamoe/src/` — see
`kernel_src/cutedsl_megamoe/SKILL.md` for how to update that directory when the
kernel team ships a new drop. The kernel-specific tuning knobs
(intermediate size, top_k, clamps, dtype `kind`, fast-math, reduce/dispatch
flags) live on the **config** dataclass in step 2 and are threaded through to
these two calls by the backend in step 4 — so an SM90/SM120 kernel that needs
different knobs only changes its own config + these two signatures, not the
`MegaKernelBackend` plumbing.

### 2. `config.py` — the user-facing config dataclass

```python
@dataclass
class MyMegaMoeConfig:
    intermediate_size: int          # post-SwiGLU width
    top_k: int
    kernel_name: str = "my_mega"    # MUST match the @register_mega_kernel name
    # ... kernel-specific knobs (dtype kind, clamps, fast_math, ...)
```

`kernel_name` is how the registry resolves a config to a backend
(`_kernel_name()` reads this attribute) — it must be a non-empty string equal to
the registration name.

### 3. `weights.py` — weight transform + validation

Provide `preprocess_mega_weights(weights: MoEWeightPack, ...) -> Transformed...`
that turns canonical bf16 (or pre-quantized) `w13`/`w2` into the kernel-ready
layout, and a `validate_transformed_mega_weights(...)` for the
`preprocess_weights=False` path (user supplies `MegaConfig.transformed_weights`).

### 4. `backend.py` — subclass `MegaKernelBackend` + register

```python
from .....core.kernel.base import MegaKernelBackend
from .....core.kernel.registry import register_mega_kernel

@register_mega_kernel("my_mega")           # == config.kernel_name
class MyMegaKernelBackend(MegaKernelBackend):
    @classmethod
    def kernel_name(cls) -> str:
        return "my_mega"

    # Required abstracts:
    def _allocate_workspace(self, fleet_params): ...   # call frontend allocator
    def compute(self, workspace, transformed_weights, *, output): ...  # call frontend kernel

    # Common overrides:
    def runtime_requirements(self, bootstrap): ...     # add "nvshmem" if needed
    def validate_init(self, bootstrap, fleet_params): ...
    def preprocess_weights(self, weights, fleet_params): ...
    def validate_transformed_weights(self, tw, bootstrap, fleet_params): ...
    def validate_forward(self, t, fleet_params, *, quantize_input): ...
    def stage_inputs(self, t, workspace, *, quantize_input): ...  # copy/quantize acts
    def destroy(self, workspace): ...
```

EP rank/world/comm are available via `self.ep_rank`, `self.ep_world_size`,
`self.ep_comm_group` (bound by `bind_ep_bootstrap`, resolved lazily once dist is
up). If the kernel needs NVSHMEM, return it from `runtime_requirements()` (see
`mxfp8_cutedsl_runtime_requirements`).

### 5. Register imports + export

- `<name>/__init__.py`: export the backend, config, transformed-weights type, and
  `preprocess_mega_weights`.
- `backends/mega/kernel/__init__.py`: add `<name>` to the `from . import ...` so
  `@register_mega_kernel` runs on import (registration is import-triggered).
- `flashinfer/moe_ep/__init__.py`: re-export the config (e.g.
  `MyMegaMoeConfig`) and any `preprocess_*` helper for user imports.

### 6. Use it

```python
from flashinfer.moe_ep import MoEEpLayer, MegaConfig, MyMegaMoeConfig

layer = MoEEpLayer(
    bootstrap=..., fleet_params=...,
    weights=...,                       # canonical MoEWeightPack, required
    backend=MegaConfig(megakernel=MyMegaMoeConfig(intermediate_size=1024, top_k=4)),
)
out = layer.forward(tensors)
layer.destroy()
```

The raw megakernel config must be wrapped in `MegaConfig` — `MoEEpLayer` routes
`MegaConfig` → `MoEEpMegaLayer` → `create_mega_kernel(cfg)`, which looks up
`cfg.kernel_name` in `_MEGA_KERNEL_REGISTRY`.

## Fault tolerance

Enable with `FleetAlgoKnobFaultTolerance()` in `fleet_knobs`. Check support
first — it needs more than the backend being built:

```python
from flashinfer.moe_ep import supports_fault_tolerance
supports_fault_tolerance("nccl_ep")   # needs nccl4py with GroupConfig.enable_mask
                                      # AND a libnccl_ep exporting ncclEpMask*
supports_fault_tolerance("nixl_ep")   # true whenever the backend is staged
```

If `nccl_ep` returns False, upgrade the nccl4py wheel that ships
`libnccl_ep.so` and confirm it is the one actually loaded
(`python -m nccl show_versions`). The probe never raises, and the Fleet
constructor fails with the same diagnosis rather than waiting for a real fault.

### Recovery state machine

```
HEALTHY ──query_fault()──> FAULT_DETECTED      (local; quiesce in-flight combine/complete)
   ──reconcile_active_mask()──> RECONCILING     (store-collective, tolerates dead ranks)
   ──clear_faults(readmit=False)──> DEGRADED    (serving continues; dead ranks' tokens dropped)

DEGRADED ──peer came back──> clear_faults(readmit=True)    [COLLECTIVE, blocking] ──> HEALTHY
DEGRADED ──peer gone for good──> update_topology(...)      [COLLECTIVE, blocking] ──> HEALTHY
```

### Ordering rules (all load-bearing)

1. **Mutate the mask only between iterations.** Both transports read it *live*
   from the dispatch/combine kernels, so changing it while a collective is in
   flight is a race that can produce a half-masked dispatch. Retire the
   iteration's `combine` (and `Handle.complete()` under
   `HandleAlgoKnobSplitOperation`) first.
2. **All survivors must reconcile in the same iteration slot** — they must pass
   the same `active_mask_epoch`. Make the host's fault poll a synchronous
   decision point.
3. `clear_faults(readmit=True)` **before** `update_topology`, never after:
   `ncclEpMaskClean` needs a live handle on the *current* group, which
   `update_topology` destroys and recreates.
4. **No FT call may be issued during CUDA-graph capture.** A captured query
   replays stale offsets; a captured set freezes one mask vector into every
   replay. All stream-ordered entry points raise on capture. Call them from the
   host between replays.

### Backend asymmetries worth knowing

* **`nixl_ep` cannot evict a rank from the middle.** `disconnect_ranks`
  requires the removed ranks to form a *suffix*, so masked-and-degraded is the
  terminal state for a mid-list failure; only a highest-numbered failure can be
  shrunk away with `update_topology`.
* **`nccl_ep` re-admission under `EXPERT_MAJOR` warns.** `ncclEpMaskClean`
  computes its buffer-reset offsets assuming `RANK_MAJOR`, so re-admission may
  leave stale bytes in the LL staging buffer. Prefer degraded serving or a full
  `update_topology` rebuild there. (Degraded serving itself is sound under
  `EXPERT_MAJOR`.)
* **Growing past the topology capacity is rejected.** Every per-rank array is
  sized to the capacity at construction; pass
  `FleetAlgoKnobTopologyCapacity(n=<max ranks>)` up front.

### Dropped tokens — no renormalization

A masked rank's experts contribute nothing and combine does **not** renormalize
`topk_weights`. An affected token therefore comes out scaled by the surviving
weight fraction:

```
y_degraded[t] = y_healthy[t] * sum(topk_weights[t][k] for k where the owning rank is alive)
```

This is deliberate. Renormalizing implicitly would add an unconditional kernel
to every forward for a case that is almost never active, hide a serving-quality
event the operator must see, and divide by zero for a token whose entire top-k
landed on dead ranks. To preserve magnitude yourself:

```python
alive = fleet.query_active_mask()                       # [world], 1 = active
owner = torch.arange(num_experts, device=d) // (num_experts // world_size)
ok = (topk_ids >= 0) & alive[owner[topk_ids.clamp_min(0)]].bool()
surviving_w = (topk_weights * ok).sum(-1)               # [num_tokens]
out.div_(surviving_w.clamp_min(1e-6).unsqueeze(-1))     # or drop requests below a threshold
```

### Being evicted

`reconcile_active_mask()` raises `MoEEpRankEvictedError` when the survivors
agreed *this* rank is dead — it stalled long enough for their kernels to give
up on it. It cannot apply that decision (a rank may not mask itself) and must
not keep serving (its peers stopped sending it tokens). Tear the worker down,
or rejoin through a fresh Fleet after the survivors call
`clear_faults(readmit=True)`.

### Running the FT tests

```bash
bash tests/moe_ep/run_tests.sh ft          # both backends, gated on support
```

Or directly (needs ≥4 GPUs):

```bash
# stalled-rank pytest half — every process survives
torchrun --nproc_per_node=4 -m pytest \
  tests/moe_ep/test_moe_ep_fault_tolerance_multirank.py -v \
  -m "nvep and gpu_4" --backend=nccl_ep      # then --backend=nixl_ep

# hard-kill half — a rank really dies, so judge by SMOKE_RESULT count, not exit code
torchrun --nproc_per_node=4 --max-restarts=0 \
  tests/moe_ep/smoke_ft_ep.py --backend nixl_ep
```

The kill half is a script rather than a pytest test on purpose: torchrun
reports the victim's non-zero exit, which would fail the survivors' pytest
session even when they behaved correctly. `run_tests.sh ft` counts
`SMOKE_RESULT:` lines (expects `nproc - 1`) and is **not** part of `run_all`.

The host-only protocol tests need no GPU at all:

```bash
pytest tests/moe_ep/test_fault_tolerance_reconcile.py \
       tests/moe_ep/test_fault_tolerance_api.py \
       tests/moe_ep/nccl_ep/test_mask_ffi.py -q
```
