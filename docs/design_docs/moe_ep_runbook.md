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
