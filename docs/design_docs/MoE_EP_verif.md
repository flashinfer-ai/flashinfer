# MoE-EP (NCCL-EP) Low-Latency benchmark — Pre-Nyx B200 repro

How to reproduce the NCCL-EP **Low-Latency (LL)** dispatch→compute→combine
benchmark end-to-end on **Pre-Nyx** (8× B200 per node, x86, NDR IB), and the
results obtained. Branch: `feat/moe_ep/enable_compute` (fork `Anerudhan/flashinfer`).

## Configuration under test

| Parameter | Value |
|---|---|
| Backend | **NCCL-EP** (`nccl4py` ≥ 0.3.1, `nccl.ep` API; EP lib `nccl-ep-v0.1.0`) |
| Algorithm | **Low-Latency (LL)** |
| Receive layout | **`EXPERT_MAJOR`** (`nccl.ep.Layout.EXPERT_MAJOR`) — recv buffer `[num_local_experts, max_tokens_per_rank × world, hidden]`; combine reweights per-token on receive. The results table below is EXPERT_MAJOR. LL also supports **`RANK_MAJOR`** (recv `[world, max_tokens_per_rank, hidden]`) — selectable with `--layout rank_major` (see §3a); HT uses `FLAT`. |
| Data type | **bf16** (`QuantVariant.BF16`) for tokens, dispatch, combine, and the expert GEMMs |
| Compute | per-expert grouped GEMM via `flashinfer.fused_moe` **`TrtllmBf16Config`** (trtllm-gen bf16 routed MoE). EP-local compute runs at **top_k = 1** (each dispatched row → one local expert); the model top-k lives in dispatch/combine |
| `hidden_size` | **7168** |
| `intermediate_size` | **2048** |
| `top_k` (model) | **8** |
| `num_experts` | **256** |
| Tokens/rank | **128** (global tokens = 128 × world size) |
| GPUs | 8 / 16 / 32 / 64 (= 1 / 2 / 4 / 8 nodes @ 8 GPU/node) |

Geometry mirrors the upstream NCCL-EP `ep_bench` reference. Selected in the
benchmark via `--reference --algorithm ll --backend nccl_ep --quant bf16`.

## B200 requirements (all three are mandatory)

1. **DOCA-GPUNetIO + GDRCopy** — NCCL-EP's GIN transport needs them even single-node.
2. **NCCL ≥ 2.30.7** — 2.27.x / 2.29.x fail `ncclEpCreateGroup` with
   `NCCL error 5 (ncclInvalidUsage) at nccl_ep.cc:1438` on B200; 2.30.7 has the
   B200 EP support. Make sure the ≥2.30.7 `libnccl` is **first** on `LD_LIBRARY_PATH`.
3. **`NCCL_MNNVL_ENABLE=1` for multi-node** (single-node intra-tray NVLink works without it).

## 1. Clone (Pre-Nyx login node)

```bash
ssh prenyx && kinit                       # Lustre needs a Kerberos ticket
WD=/lustre/fsw/coreai_libraries_cudnn/agopal-moe-ep && mkdir -p $WD/logs && cd $WD
git clone -b feat/moe_ep/enable_compute https://github.com/Anerudhan/flashinfer.git
cd flashinfer && git submodule update --init --recursive 3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog
# (no nccl/nixl submodule needed — NCCL-EP is the nccl4py wheel)
```

## 2. Build the container (one SLURM job, ~15 min) — `$WD/build.sh`

```bash
#!/bin/bash
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
# IB userspace + build tools
apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    git curl ca-certificates build-essential \
    rdma-core libibverbs1 libibverbs-dev ibverbs-providers libibumad3 librdmacm1 infiniband-diags
# DOCA-GPUNetIO (GIN)  [amd64 on Pre-Nyx]
curl -fSL --retry 3 "https://www.mellanox.com/downloads/DOCA/DOCA_v3.2.0/host/doca-host_3.2.0-125000-25.10-ubuntu2404_$(dpkg --print-architecture).deb" -o /tmp/doca.deb
dpkg -i /tmp/doca.deb || true; apt-get update -qq
apt-get install -y -qq --no-install-recommends doca-sdk-gpunetio libdoca-sdk-gpunetio-dev libdoca-sdk-verbs-dev
# GDRCopy
git clone -q --depth=1 --branch v2.5.1 https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy
(cd /tmp/gdrcopy && make -j lib lib_install); ldconfig
# venv + deps (note nvidia-nccl-cu13>=2.30.7)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH; V=/opt/flashinfer-venv
uv venv --python 3.12 --clear $V; export PATH=$V/bin:$PATH VIRTUAL_ENV=$V
uv pip install --python $V/bin/python \
    torch setuptools packaging "apache-tvm-ffi>=0.1.6,<0.2,!=0.1.8,!=0.1.8.post0" cython pybind11 \
    numpy einops ninja nvidia-ml-py click requests tabulate tqdm \
    "nvidia-cutlass-dsl>=4.5.0" "nvidia-cudnn-frontend>=1.13.0" "cuda-tile>=1.4.0" \
    "cuda-python>=13.0" "nccl4py>=0.3.1" "nvidia-nccl-cu13>=2.30.7"
cd /host/flashinfer
uv pip install --python $V/bin/python --no-build-isolation --no-deps -e .   # NCCL-EP only (no NIXL build)
python -c "import nccl.ep;from flashinfer.moe_ep import available_backends;print(nccl.ep.get_lib_version(),available_backends())"
```

Run it, saving the image:
```bash
sbatch -A coreai_libraries_cudnn -p batch -N1 --time=00:40:00 -o $WD/logs/build_%j.log --wrap \
 "srun --container-image='nvcr.io#nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04' --container-writable \
       --container-save=$WD/flashinfer-nccl-b200.sqsh --container-mounts=$WD:/host bash /host/build.sh"
```

## 3. Run the LL sweep — `$WD/run_ll.sh`

```bash
#!/bin/bash
export PATH=/opt/flashinfer-venv/bin:/root/.local/bin:$PATH
# bind the >=2.30.7 wheel libnccl FIRST (resolve by ncclGetVersion; nvidia.nccl.__file__ is empty)
NCCLLIB=$(python - <<'PY'
import glob,ctypes,os
for p in sorted(set(glob.glob("/opt/flashinfer-venv/**/libnccl.so.2*",recursive=True))):
    try:
        l=ctypes.CDLL(p);v=ctypes.c_int();l.ncclGetVersion(ctypes.byref(v))
        if v.value>=23007: print(os.path.dirname(p)); break
    except Exception: pass
PY
)
export LD_LIBRARY_PATH=$NCCLLIB:/opt/mellanox/doca/lib/x86_64-linux-gnu:/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
[ "${SLURM_NNODES:-1}" -gt 1 ] && export NCCL_MNNVL_ENABLE=1     # multi-node only
cd /host/flashinfer
torchrun --nnodes="$SLURM_NNODES" --nproc_per_node=8 --node_rank="$SLURM_NODEID" \
  --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:29500" \
  benchmarks/bench_moe_ep.py --reference --algorithm ll --backend nccl_ep --quant bf16 \
  --warmup 5 --repeat 20
```

Submit one job per node count (8/16/32/64 GPU = 1/2/4/8 nodes):
```bash
for N in 1 2 4 8; do
  sbatch -A coreai_libraries_cudnn -p batch -N $N --ntasks-per-node=1 --time=00:45:00 \
    -o $WD/logs/ll_N${N}_%j.log --wrap \
    "export MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -1); \
     srun --export=ALL --container-image=$WD/flashinfer-nccl-b200.sqsh --container-mounts=$WD:/host bash /host/run_ll.sh"
done
```
Rank 0 prints one CSV line per job:
`BENCH_CSV,algo,layout,tokens,gpus,backend,quant,dispatch_us,compute_us,combine_us,e2e_us,tok_s`
(Use ≥45-min walltime — the first run JIT-compiles the trtllm bf16 kernel and autotunes, ~12–15 min.)

## 3a. RANK_MAJOR variant — `$WD/run_ll.sh` + `--layout rank_major`

The same LL sweep with the **`RANK_MAJOR`** receive layout: append `--layout
rank_major` to the `torchrun … bench_moe_ep.py` line in `run_ll.sh` (everything
else — build, env, MNNVL, walltime — is identical). RANK_MAJOR groups received
tokens by source rank (`[world, max_tokens_per_rank, hidden]`) instead of padding
per-expert.

```bash
torchrun … benchmarks/bench_moe_ep.py --reference --algorithm ll \
  --backend nccl_ep --quant bf16 --layout rank_major --warmup 5 --repeat 20
```

The RANK_MAJOR compute is driven by the library's received per-token routing at
the model's real top_k (`do_finalize` pre-reduce across local experts), with
non-local picks masked to weight 0. ABI notes: the received `topk_idx` /
`topk_weights` are rank-grouped 3D (`[world, max_tokens_per_rank, top_k]`) with
`topk_idx` in **int32**; the `topk_idx` values are **LOCAL** expert indices
(0-based within this rank), with `-1` marking a non-local pick. The benchmark uses
random data (latency, not accuracy); numerical correctness is validated by
`tests/moe_ep/test_moe_ep_compute_correctness.py` (see "Correctness" below).

### Results — Pre-Nyx B200, LL, RANK_MAJOR, bf16 (per-stage median µs)

| GPUs | nodes | MNNVL | dispatch µs | compute µs | combine µs | e2e µs | tok/s |
|------|-------|-------|-------------|------------|------------|--------|-------|
| 8    | 1     | 0 | 151.9 | 1465.0 | 79.2 | 1840.0 | 0.56 M |
| 16   | 2     | 1 | 286.5 | 2059.7 | 194.9 | 2687.3 | 0.76 M |
| 32   | 4     | 1 | 369.8 | 3368.0 | 346.5 | 4223.0 | 0.97 M |
| 64   | 8     | 1 | 416.5 | 6317.2 | 311.8 | 7196.9 | 1.14 M |

**EXPERT_MAJOR vs RANK_MAJOR — a compute crossover at 32 GPUs.** The two layouts
do equal expert-GEMM work when `world × top_k = num_experts` (here `world × 8 =
256` → **world = 32**), which the measured compute term tracks exactly:

- EXPERT_MAJOR compute = `num_local_experts × cap = num_experts × tokens_per_rank`
  = **constant** in world size (flat ~3.3–3.5 ms across 8→64 GPU).
- RANK_MAJOR compute = `world × tokens_per_rank × top_k` = **grows linearly** with
  world (8 GPU: 1.5 ms, 32 GPU: ≈3.4 ms = EM, 64 GPU: 6.3 ms).

So RANK_MAJOR is ~2.4× faster at 8 GPU (compute 1.47 vs 3.53 ms; 0.56 M vs 0.25 M
tok/s) but ~1.6× slower at 64 GPU (1.14 M vs 1.83 M tok/s). The cause: the unified
runner evaluates **all** `top_k` slots per received token, but only ~1 is this
rank's local expert (the other `top_k − 1` are masked to weight 0 yet still
computed). An exact local-only gather (compute just the local picks,
≈ `world × tokens_per_rank × 1`) would stay below EXPERT_MAJOR at every scale —
the optimization headroom for a follow-up.

## Results — Pre-Nyx B200, LL, EXPERT_MAJOR, bf16 (per-stage median µs)

| GPUs | nodes | MNNVL | dispatch µs | compute µs | combine µs | e2e µs | tok/s |
|------|-------|-------|-------------|------------|------------|--------|-------|
| 8    | 1     | 0 | 125.5 | 3529.7 | 362.8 | 4127.6 | 0.25 M |
| 16   | 2     | 1 | 255.2 | 3438.2 | 204.1 | 4080.3 | 0.50 M |
| 32   | 4     | 1 | 401.4 | 3308.5 | 714.2 | 4569.2 | 0.90 M |
| 64   | 8     | 1 | 439.5 | 3264.9 | 633.7 | 4475.2 | 1.83 M |

- **Compute-bound**: the grouped bf16 GEMM (~3.3–3.5 ms) dominates e2e; tok/s scales
  ~linearly with GPU count.
- **Dispatch**: 8-GPU single-node (intra-tray NVLink) is cheapest (~126 µs); multi-node
  (IB + MNNVL) dispatch grows with node count (~255 → 440 µs).
- Numbers are consistent with GB200 (Lyris / Ptyche) runs at the same config.

Notes: `dispatch_us` includes a host-sync (per-expert recv-count readback), so it is a
latency, not a pure-transport-bandwidth, figure. `combine_us` is noisier here (per-stage
CUDA-event timing) and larger than in earlier runs after the dispatch now issues an
unconditional `complete()` (correctness fix; see below).

## Correctness

These timings were measured **after** the multi-rank correctness fixes (the earlier
numbers were taken while ranks with `local_expert_offset > 0` silently skipped their
experts — fast but wrong — so they understated real per-rank compute). Two distinct
bugs, both surfaced by the functional test and invisible to this latency benchmark
(random data, no accuracy check):

1. The trtllm routed runners packed **local** expert ids while the kernel expects
   **global** ids + filters by `local_expert_offset` → offset>0 ranks dropped their
   experts. Fixed: pack global ids.
2. RANK_MAJOR's received `topk_idx` are **local** indices (`-1` = non-local), not
   global; the compute bridge misread them as global. Fixed: local→global conversion.

Both LL layouts now pass the multi-GPU functional test
(`tests/moe_ep/test_moe_ep_compute_correctness.py`, 8× B200, bf16): EP
dispatch→compute→combine matches the same `MoELayer` kernel run non-EP to
`rel-err ≈ 0.0045` (EXPERT_MAJOR **and** RANK_MAJOR). Single-GPU layout-bridge /
numerics / smoke remain covered by the rest of `tests/moe_ep/`.
