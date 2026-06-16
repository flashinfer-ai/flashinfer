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

## LL vs HT (algorithm axis)

`MoEEpLayer`/`NcclEpHandle` is now **algorithm-aware** (`FleetParams.algorithm`):
LL uses the `EXPERT_MAJOR` receive layout; HT uses `FLAT` (dispatch exchanges
`topk_weights`/`topk_idx`, recv `[num_recv, H]` reshaped to the bridge's 3D view).
`bench_moe_ep.py --algorithm {ll,ht}` selects it; `--reference` is the ep_bench
geometry (hidden=7168, experts=256, top_k=8, 128 tok/rank).

**LL — B200, reference config (full bf16 compute):**

| GPUs | Nodes | dispatch µs | compute µs | combine µs | e2e µs | tok/s |
|---|---|---|---|---|---|---|
| 8 | 2 | 172.5 | 3428.9 | 45.1 | 3837.8 | 0.27M |
| 16 | 4 | 195.0 | 3381.3 | 45.2 | 3813.9 | 0.54M |
| 32 | 8 | 184.8 | 3408.3 | 47.2 | 3851.4 | 1.06M |
| 64 | 16 | 204.8 | 3292.2 | 46.9 | 3736.3 | 2.19M |

**HT — single-rank works; multi-rank blocked by the nccl4py library on Blackwell.**
HT (FLAT) runs end-to-end at `world_size=1`, but **every multi-rank HT run (≥2 GPUs,
both intra-node NVLink and inter-node RDMA) aborts inside the library**:
`CUDA error nccl_ep.cc:3269 'invalid argument'`. The flashinfer wrapper drives the
HT API per `contrib/nccl_ep/ep_test.py` (single-rank validates it); the failure is
below our layer. Consistent with HT being Hopper-optimized + experimental
(README: "Hopper architecture features … warp-specialized pipelines, TMA";
RELEASE.md: "HT mode … being tuned," "limited QA coverage"). The HT table is
therefore unfillable on GB200 with nccl4py 0.3.1 — escalate to the NCCL-EP team.
(The README reference is itself LL-only, so there is no HT reference to match.)

## Platforms & NCCL-EP runtime dependency (DOCA-GPUNetIO + GDRCopy)

NCCL-EP's group creation (`ncclEpCreateGroup`) sets up the **GIN (GPU-Initiated
Networking)** transport, which requires **DOCA-GPUNetIO + GDRCopy** at runtime —
**even single-node**. The `nccl4py` wheel + matching `nvidia-nccl-cuXX` alone are
NOT sufficient: a minimal container without DOCA/GDRCopy fails at
`NCCL error 5 (ncclInvalidUsage) at nccl_ep.cc:1438`. (This is why Lyris worked —
its full `build_in_container.sh` installs DOCA/GDRCopy — while early minimal
Pre-Nyx/Ptyche builds did not.) So "no NCCL build" ≠ "no heavy deps": DOCA +
GDRCopy are required; UCX/NIXL are only needed for NIXL-EP.

| Platform | GPU | Build | NCCL-EP LL |
|---|---|---|---|
| Lyris | GB200 (4/node) | full build_in_container.sh | ✅ (table above) |
| Ptyche | GB200-NVL36 (4/node, aarch64) | minimal + DOCA-GPUNetIO + GDRCopy | ✅ full sweep below |
| Pre-Nyx | B200 (8/node, x86) | minimal + DOCA/GDRCopy + **NCCL≥2.30.7** | ✅ full sweep below (multi-node needs `NCCL_MNNVL_ENABLE=1`) |

Minimal NCCL-EP-only container = cuda:13 base + IB userspace (`rdma-core`,
`libibverbs`, `ibverbs-providers`) + **DOCA-GPUNetIO + GDRCopy** + torch +
flashinfer runtime deps + `nccl4py` (+ matching `nvidia-nccl-cuXX`) + `pip install
--no-deps -e .`. ~15–20 min (vs ~40 for the full NIXL stack).

**Ptyche (GB200-NVL36) — LL, reference config, bf16, full compute** (per-stage median µs):

| GPUs | nodes | dispatch µs | compute µs | combine µs | e2e µs | tok/s |
|------|-------|-------------|------------|------------|--------|-------|
| 8    | 2     | 170.2 | 3508.8 | 46.8 | 3930.7 | 0.26 M |
| 16   | 4     | 165.0 | 3427.3 | 49.6 | 3827.9 | 0.54 M |
| 32   | 8     | 187.1 | 3323.5 | 47.3 | 3741.9 | 1.09 M |
| 64   | 16    | 193.0 | 3296.1 | 49.6 | 3736.2 | 2.19 M |

Matches Lyris GB200: dispatch ~165–193 µs and combine ~47–50 µs flat across scale;
compute-bound; tok/s ~linear with GPU count.

**NCCL version note:** `libnccl_ep.so` binds whatever `libnccl.so.2` is first on
`LD_LIBRARY_PATH`. The cuda:13 base image ships system NCCL **2.27.7**, and the
wheel-path autodetect (`nvidia.nccl.__file__`, a namespace pkg → empty) silently
left the wheel NCCL off the path — so the GB200 runs above actually used
**2.27.7** (works fine on GB200). Always put the wheel lib dir
(`.../site-packages/nvidia/nccl/lib`) explicitly first.

**Pre-Nyx B200 — RESOLVED: needs NCCL ≥ 2.30.7.** The earlier `ncclInvalidUsage`
at `nccl_ep.cc:1438` was an **NCCL-version** problem, not a hardware limitation:
NCCL **2.27.7** (base-image system) and **2.29.7** both fail EP group-create on
B200, but **2.30.7** carries the B200 EP support and succeeds. Two more
requirements on B200:
- bind the **2.30.7** wheel `libnccl` first on `LD_LIBRARY_PATH` (resolve by
  `ncclGetVersion>=23007`; the `nvidia.nccl.__file__` autodetect returns empty);
- set **`NCCL_MNNVL_ENABLE=1` for multi-node** runs (single-node intra-tray NVLink
  works without it).
DOCA-GPUNetIO + GDRCopy are still required (GIN), as on GB200.

**Pre-Nyx (B200, 8 GPU/node) — LL, reference config, bf16, full compute, NCCL 2.30.7:**

| GPUs | nodes | MNNVL | dispatch µs | compute µs | combine µs | e2e µs | tok/s |
|------|-------|-------|-------------|------------|------------|--------|-------|
| 8    | 1     | 0 | 125.6 | 3650.0 | 49.4 | 3959.8 | 0.26 M |
| 16   | 2     | 1 | 254.7 | 3468.4 | 76.4 | 3946.5 | 0.52 M |
| 32   | 4     | 1 | 360.3 | 3338.7 | 66.0 | 3906.3 | 1.05 M |
| 64   | 8     | 1 | 394.6 | 3209.2 | 60.1 | 3807.5 | 2.15 M |

8-GPU single-node (intra-tray NVLink) has the cheapest dispatch (~126 µs); multi-node
(IB, MNNVL) dispatch rises with node count. tok/s ~linear; compute-bound throughout —
consistent with GB200. **`report.md` (the standalone-B200 bug report) is now obsolete**
(B200 works with NCCL ≥ 2.30.7); keep only as a note that the fix is the NCCL version.

## Open follow-ups (revisit)

- **NVFP4 compute illegal memory access** — fails in the fused grouped-GEMM (CuteDSL /
  trtllm-fp4) on EP-shaped inputs; surfaces async at `nccl_ep.cc:1556` (combine).
  Needs `compute-sanitizer` to localize (output-buffer/alignment or padded-row region).
- **Multi-node traceback capture** — torch 2.12 torchrun buffers worker stderr; add a
  `@record`/`TORCHELASTIC_ERROR_FILE` path to smoke/bench so errors surface without the
  single-process workaround.
- **Perf numbers are functional medians**, not tuned; combine NVTX + CUDA-graph capture
  for production-grade measurement.
- ~~nccl_ep build of `libnccl_ep.so` is vestigial~~ **RESOLVED:** NCCL-EP now comes from
  the released `nccl4py>=0.3.1` wheel (validated `0.3.1` on GB200) — no submodule build,
  no `LD_LIBRARY_PATH` to `build_nvep/nccl/lib`. The `[nvep]` extra installs it; only
  NIXL-EP is compiled in-tree. See `MoE_EP_verif.md` §1 (Install & run flow).

## How to reproduce on Lyris

- Container image (built once, reused): `/home/agopal/flashinfer-work/flashinfer-ep.sqsh`
  (cuda:13.0 base + UCX/DOCA/GDRCopy + torch 2.12 + flashinfer + **nccl4py 0.3.1 wheel** +
  cutlass-dsl).
- Checkout: `/home/agopal/flashinfer-work/flashinfer-compute` (this branch, editable install).
- Env preamble inside the container (see `run_*.sh`): prepend the **nvidia-nccl wheel lib
  dir** + UCX/DOCA to `LD_LIBRARY_PATH`. The old `build_nvep/nccl/lib` entry is no longer
  needed (NCCL-EP is now the nccl4py wheel).
- Sweep: `sbatch /home/agopal/flashinfer-work/sweep_{gb200,gb300}_N{2,4}.sbatch`
  (account `coreai_libraries_cudnn`, partitions `gb200-backfill` / `gb300-backfill`).
- Driver: `benchmarks/bench_moe_ep.py --tokens .. --backend .. --quant {bf16,nvfp4}`.
- See `docs/design_docs/MoE_EP_verif.md` for the build + verification methodology.
