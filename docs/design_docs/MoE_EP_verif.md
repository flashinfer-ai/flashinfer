# MoE Expert-Parallel — Build & Verification Methodology

Covers building FlashInfer with the NCCL-EP / NIXL-EP transport backends and the
compute API wired into `MoEEpLayer`, then verifying correctness and performance on
GB200 (SM100) / GB300 (SM103).

This complements the compute-side design in
[`flashinfer_moe_api.md`](flashinfer_moe_api.md): that doc describes the unified MoE
compute API (`MoEConfig` / `MoELayer` / runners); this doc describes how that compute
API is wired *behind* the expert-parallel `dispatch → compute → combine` flow and how to
build and validate the combined stack.

## 0. Architecture recap

`flashinfer.moe_ep.MoEEpLayer.forward()` runs, per iteration:

```
dispatch  →  inner-compute (per-expert grouped GEMM)  →  combine  →  complete
```

- **dispatch / combine** are the NCCL-EP or NIXL-EP all-to-all transport (PR #3315 build
  infra, PR #3453 comm code).
- **inner-compute** is the unified MoE compute API (PR #3093): a `MoELayer` driven by a
  `MoEConfig` (`RoutingConfig` / `QuantConfig` / `ExpertConfig`).

The dispatch output is **expert-major, padded** —
`[num_local_experts, cap, hidden]` with `cap = max_tokens_per_rank * world_size` — plus a
per-expert `recv_count`. The compute runners want a **token-major** `[M, hidden]` batch
with `selected_experts` / `final_scales`. The layout bridge
(`flashinfer/moe_ep/_compute_bridge.py`) reconciles them: each dispatched row is already
assigned to exactly one local expert (`expert = row // cap`), so the bridge flattens to
`[num_local_experts*cap, hidden]`, synthesizes `top_k=1` routing with `final_scales = 1`
(the real top-k reweight is owned by `combine`), runs the runner with
`do_finalize=True` (an identity-order scatter at `top_k=1`), then reshapes back to the 3D
combine layout.

## 1. Build

### Prerequisites
- CUDA 13+, recent driver; `nvcc`, `cmake` / `meson`, `ninja`, `pkg-config`.
- **NCCL-EP** is provided by the released **`nccl4py>=0.3.1`** wheel (the `[nvep]`
  extra installs it) — no in-tree build, no submodule needed. The wheel ships the
  `nccl.ep` API and bundles `libnccl_ep.so` (loaded via cuda-pathfinder, so no
  `LD_LIBRARY_PATH` setup is required).
- **NIXL-EP** is still built from `3rdparty/nixl` (meson) and needs UCX +
  `libibverbs` (DOCA / RDMA): `git submodule update --init --recursive 3rdparty/nixl`.

### Build commands
```bash
# Editable dev install with both EP backends
BUILD_NCCL_EP=1 BUILD_NIXL_EP=1 \
  pip install --no-build-isolation -e . -v
# NCCL-EP only:  BUILD_NCCL_EP=1 ...
# NIXL-EP only:  BUILD_NIXL_EP=1 ...
```
- `build_backend.py` probes deps, builds each backend, stages `.so`s into
  `flashinfer/moe_ep/{nccl_ep,nixl_ep}/_libs/`, and installs runtime wheels
  (`nvidia-nccl-cu{N}`, `nixl-cu{N}`).
- Container reference: `docker/Dockerfile.flashinfer-nvep`.

### Build sanity
```python
from flashinfer.moe_ep import have_nccl_ep, have_nixl_ep, available_backends
assert have_nccl_ep() or have_nixl_ep()
print(available_backends())
# nccl-ep-v0.1.0 ships the EP API inside nccl4py as nccl.ep (no flat nccl_ep module).
import nccl.ep
print(nccl.ep.get_lib_version())   # nccl-ep-v0.1.0
```

## 2. Correctness verification

### Levels
1. **Unit (CPU / single-GPU, no comm):** layout-bridge tests — flattening
   `[E_local, cap, hidden] → [E_local*cap, hidden]`, synthesized `selected_experts`
   (`row // cap + local_expert_offset`), `final_scales == 1`, `top_k == 1`, and reshape
   back to the 3D combine layout.
   `pytest tests/moe_ep/test_compute_bridge.py`.
2. **Single-GPU numerics:** `world_size=1` EP run (`dispatch → compute → combine`) vs a
   dense reference (`cutlass_fused_moe` / `tests/moe/test_unified_moe.py` helpers) on
   identical weights + routing. Tolerances: bf16 `rtol=2e-2`; NVFP4 compared against the
   **bf16 reference** (shared-reference method from PR #3093), `rtol≈5e-2`.
3. **Multi-GPU round-trip:** `torchrun --nproc_per_node=8` over
   `tests/moe_ep/smoke_{nccl,nixl}_ep.py` (extended with `compute_config`); assert the
   gathered output matches the single-GPU reference for the same global tokens / weights.

### Pass criteria
- All unit + single-GPU numerics pass within tolerance for NVFP4 **and** bf16.
- Multi-GPU output matches single-GPU reference for both NCCL-EP and NIXL-EP.
- `MoELayer.winner_backend` resolves to a valid runner on the target arch.
- No leaks / hangs across repeated forwards (run N=100 iters under
  `compute-sanitizer --tool memcheck` on the single-GPU path).

## 3. Performance verification

Sweep: tokens ∈ {8K, 16K} × GPUs ∈ {8, 16} × backend ∈ {NCCL-EP, NIXL-EP}
× quant ∈ {NVFP4, bf16}, on GB200 (SM100) and GB300 (SM103).

- Tool: `flashinfer.testing.bench_gpu_time` (CUPTI, CUDA-graph capture, ≥30 iters, warmup
  includes the autotune pass).
- Break out **dispatch / compute / combine** via per-stage NVTX so the new compute cost
  is isolated.
- Baseline = the comm-only `_inner_compute_identity` path at the same config → shows the
  compute overhead added on top of pure transport.
- Driver:
  `benchmarks/bench_moe_ep.py --tokens {8192,16384} --world-size {8,16} --backend {nccl_ep,nixl_ep} --quant {nvfp4,bf16}`.
- Fix model geometry (DeepSeek-V3-class: `num_experts=256`, `top_k=8`, `hidden=7168`,
  `intermediate=2048`) and record it with each result table.

Result table (one per HW × quant), all latencies median µs:

| Tokens | GPUs | Backend | dispatch | compute | combine | e2e | tok/s | vs identity |
|--------|------|---------|----------|---------|---------|-----|-------|-------------|
| 8K  | 8  | NCCL-EP | | | | | | |
| 8K  | 8  | NIXL-EP | | | | | | |
| 8K  | 16 | NCCL-EP | | | | | | |
| 8K  | 16 | NIXL-EP | | | | | | |
| 16K | 8  | NCCL-EP | | | | | | |
| 16K | 8  | NIXL-EP | | | | | | |
| 16K | 16 | NCCL-EP | | | | | | |
| 16K | 16 | NIXL-EP | | | | | | |

## 4. Running on lyris (GB200 / GB300)
- Allocate GB200 / GB300 nodes via SLURM; launch with `torchrun` (single node up to 8
  GPUs; 16 GPUs = 2 nodes, set `--nnodes 2 --rdzv_backend c10d --rdzv_endpoint <host:port>`).
- NIXL-EP requires an RDMA fabric reachable between ranks; verify `ucx_info -d` and the IB
  devices (`ibv_devinfo`) before multi-node runs.
- Pin one rank per GPU; export `CUDA_VISIBLE_DEVICES` per local rank as the launcher sets
  it.

## 5. CI hook (follow-up)
- Gate unit + single-GPU numerics in CI (1 GPU).
- Multi-GPU + perf run as a nightly / on-demand job on the GB200 / GB300 pool.
