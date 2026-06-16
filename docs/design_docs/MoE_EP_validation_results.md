# MoE-EP Compute Integration — Lyris Validation Checkpoint

Checkpoint of the on-cluster validation of the unified MoE **compute API wired into
expert-parallel** (`MoEEpLayer`: dispatch → grouped-GEMM → combine) over both
NCCL-EP and NIXL-EP at `nccl-ep-v0.1.0`, on Lyris **GB200 (SM100)** and **GB300
(SM103)**. Branch: `feat/moe_ep/enable_compute` (fork `Anerudhan/flashinfer`).

Status: **bf16 compute path validated end-to-end on both architectures and both
backends.** NVFP4 + a couple of harness items are tracked follow-ups (below).

## What was validated

| Check | Result |
|---|---|
| NVEP build at `nccl-ep-v0.1.0` (`build_backend.py` + `nccl.ep` backend rewrite) | ✅ `import flashinfer`; `backends=['nccl_ep','nixl_ep']` |
| 4-GPU smoke — nccl_ep / nixl_ep transport (identity path) | ✅ both |
| Bridge unit tests (`tests/moe_ep/test_compute_bridge.py`) | ✅ 4/4 |
| bf16 compute, dispatch→GEMM→combine, **8 & 16 GPU × 8K/16K × {nccl,nixl}**, B200 + B300 | ✅ (matrix below) |
| NVFP4 compute | ❌ CUDA illegal memory access in the fused kernel — follow-up |

## bf16 results (per-stage median µs; tok/s on the global batch)

Model geometry: `num_experts=16, hidden=4096, intermediate=2048, top_k(model)=8`,
GB200 = 4 GPU/node. nixl_ep has a 1024 per-rank token cap (so 16384/8-GPU is N/A).

**GB200 (SM100):**

| tokens | GPUs | backend | dispatch | compute | combine | e2e | tok/s |
|---|---|---|---|---|---|---|---|
| 8192 | 8 | nccl_ep | 559 | 1416 | 132 | 2306 | 3.55M |
| 8192 | 8 | nixl_ep | 133 | 1440 | 3.2 | 1836 | 4.46M |
| 16384 | 8 | nccl_ep | 985 | 2176 | 245 | 3592 | 4.56M |
| 16384 | 8 | nixl_ep | — | — | — | — | n/a (per-rank>1024) |
| 8192 | 16 | nccl_ep | 361 | 1038 | 80 | 1680 | 4.88M |
| 8192 | 16 | nixl_ep | 151 | 1073 | 2.2 | 1433 | 5.72M |
| 16384 | 16 | nccl_ep | 605 | 1402 | 137 | 2349 | 6.97M |
| 16384 | 16 | nixl_ep | 135 | 1423 | 2.7 | 1811 | 9.05M |

**GB300 (SM103):**

| tokens | GPUs | backend | dispatch | compute | combine | e2e | tok/s |
|---|---|---|---|---|---|---|---|
| 8192 | 8 | nccl_ep | 551 | 1392 | 129 | 2263 | 3.62M |
| 8192 | 8 | nixl_ep | 120 | 1360 | 2.2 | 1739 | 4.71M |
| 16384 | 8 | nccl_ep | 957 | 2126 | 241 | 3525 | 4.65M |
| 16384 | 8 | nixl_ep | — | — | — | — | n/a (per-rank>1024) |
| 8192 | 16 | nccl_ep | 363 | 1064 | 78 | 1744 | 4.70M |
| 8192 | 16 | nixl_ep | 135 | 1023 | 3.1 | 1370 | 5.98M |
| 16384 | 16 | nccl_ep | 571 | 1410 | 139 | 2345 | 6.99M |
| 16384 | 16 | nixl_ep | 134 | 1402 | 3.1 | 1817 | 9.02M |

Notes: tok/s scales with GPU count; compute dominates e2e; nixl_ep dispatch is ~8×
cheaper than nccl_ep and combine is near-free, but it has the 1024 per-rank cap;
B300 ≈ B200 (B300 slightly faster compute).

## Bugs found by hardware & fixed (on the branch)

1. EP-local compute must run at `top_k=1` (each dispatched row → one local expert);
   inner `MoELayer` built with `routing.top_k=1`. (`c40bc00a`)
2. NVFP4 bridge passed `global_scale=None`; default to unit global scale. (`c40bc00a`)
3. NVFP4 activation scale must be **linear** layout (`is_sf_swizzled_layout=False`). (`a1c5f94d`)
4. Bench timing was bogus (host-sync defeats `bench_gpu_time`); added opt-in
   per-stage CUDA-event timing to `MoEEpLayer` + wall-clock e2e. (`94014f8b`)
5. Bench didn't set `bootstrap.tcp_store` for nixl_ep. (`117466e6`)
6. `nccl-ep-v0.1.0` migration: dropped legacy `contrib/nccl_ep/python` install in
   `build_in_container.sh` + Dockerfile; `nccl.ep` ships in nccl4py.

## Open follow-ups (revisit)

- **NVFP4 compute illegal memory access** — fails in the fused grouped-GEMM (CuteDSL /
  trtllm-fp4) on EP-shaped inputs; surfaces async at `nccl_ep.cc:1556` (combine).
  Needs `compute-sanitizer` to localize (output-buffer/alignment or padded-row region).
- **Multi-node traceback capture** — torch 2.12 torchrun buffers worker stderr; add a
  `@record`/`TORCHELASTIC_ERROR_FILE` path to smoke/bench so errors surface without the
  single-process workaround.
- **Perf numbers are functional medians**, not tuned; combine NVTX + CUDA-graph capture
  for production-grade measurement.
- **nccl_ep build of `libnccl_ep.so` is vestigial** now (probe uses `find_spec('nccl.ep')`);
  the lib loads via `LD_LIBRARY_PATH` to `build_nvep/nccl/lib` — consider staging it onto
  nccl4py's loader path so `LD_LIBRARY_PATH` isn't required.

## How to reproduce on Lyris

- Container image (built once, reused): `/home/agopal/flashinfer-work/flashinfer-ep.sqsh`
  (cuda:13.0 base + UCX/DOCA/GDRCopy + torch 2.12 + flashinfer + nccl4py + cutlass-dsl).
- Checkout: `/home/agopal/flashinfer-work/flashinfer-compute` (this branch, editable install).
- Env preamble inside the container (see `run_*.sh`): prepend
  `build_nvep/nccl/lib` + the nvidia-nccl wheel lib dir to `LD_LIBRARY_PATH`.
- Sweep: `sbatch /home/agopal/flashinfer-work/sweep_{gb200,gb300}_N{2,4}.sbatch`
  (account `coreai_libraries_cudnn`, partitions `gb200-backfill` / `gb300-backfill`).
- Driver: `benchmarks/bench_moe_ep.py --tokens .. --backend .. --quant {bf16,nvfp4}`.
- See `docs/design_docs/MoE_EP_verif.md` for the build + verification methodology.
