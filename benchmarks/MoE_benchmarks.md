# MoE-EP benchmarks (NCCL-EP) — container, how to run, and results

Benchmarking the FlashInfer **MoE expert-parallel** path (`nccl.ep` dispatch / combine,
optionally + the expert grouped-GEMM). Two drivers:

| Driver | Measures | Comparable to |
|---|---|---|
| `benchmarks/bench_ep_matrix.py` | **comm only** — dispatch + combine, no FFN | the upstream `contrib/nccl_ep/ep_bench` C++ reference |
| `benchmarks/bench_moe_ep.py` | **full** MoE — dispatch → grouped GEMM → combine | (no external ref; LL has the NCCL-EP README config) |

The headline study is `bench_ep_matrix.py` vs `ep_bench`: it confirms FlashInfer and
ep_bench launch the **same** kernels with the **same** GPU time, and quantifies (and then
removes) the Python host-call overhead. See also the implementation/usage guide,
`docs/design_docs/MoE_EP_impl.md`.

---

## 1. Container

**Use `docker/Dockerfile.flashinfer-ep-pytorch`** (base `nvcr.io/nvidia/pytorch:26.05-py3`,
CUDA 13.2). This image is required for **cross-node High-Throughput** (HT/GIN/GDAKI): the
CUDA-13.0 image (`docker/Dockerfile.flashinfer-nvep`) aborts multi-node HT dispatch with
`CUDA error nccl_ep.cc:2884 'illegal memory access'` — verified to be the base image's
IB-RDMA/GPUDirect runtime stack, not FlashInfer (single-node HT and all LL cases pass on
both). The PyTorch base already ships the GDAKI/GPUDirect userspace stack, so the heavy
DOCA / UCX-from-source / GDRCopy layers of the NIXL image are unnecessary for NCCL-EP.

Build (`docker/install/build_flashinfer_ep_pytorch.sh` does the install): it pins the
verified set over the base image's constraints — `nvidia-nccl-cu13==2.30.7` (via
`PIP_CONSTRAINT=` to beat torch's 2.30.4 pin), `nccl4py[cu13]==0.3.1`, `cuda-core==1.0.1`,
`cuda-bindings==13.2.0` — then `BUILD_NCCL_EP=1 BUILD_NIXL_EP=0 pip install -e .`
(the moe_ep deps are base dependencies now; no extra needed).

```bash
# local docker
docker build -f docker/Dockerfile.flashinfer-ep-pytorch -t flashinfer-ep:pt2605 .
# SLURM/pyxis: build a .sqsh by running the install in the base image + --container-save
RW=/lustre/.../agopal-moe-ep            # holds the flashinfer checkout
srun -N1 --container-image=nvcr.io/nvidia/pytorch:26.05-py3 \
  --container-save=$RW/flashinfer-ep-pt2605.sqsh --container-mounts=$RW:/host \
  bash -lc 'bash /host/flashinfer/docker/install/build_flashinfer_ep_pytorch.sh'
```

**Runtime requirements** (B200/Pre-Nyx): NCCL-EP's GIN transport needs the GDAKI/GPUDirect
stack even single-node; **NCCL ≥ 2.30.7** (2.27/2.29 fail group-create at `nccl_ep.cc:1438`
on B200) bound **first** on `LD_LIBRARY_PATH`; and **`NCCL_MNNVL_ENABLE=1` for multi-node**
(single-node intra-tray NVLink works without it). The PyTorch image + the pinned wheels
above satisfy these; `nccl.ep` is the `nccl4py` wheel (no in-tree NCCL build).

Smoke: `python -c "import nccl.ep; from flashinfer.moe_ep import available_backends; print(available_backends())"` → `['nccl_ep', ...]`.

---

## 2. How to run

### 2a. Comm matrix vs ep_bench (`bench_ep_matrix.py`)
Standalone — needs only FlashInfer (EP is in the default install), torch, and a multi-rank launcher; it does
**not** call `ep_bench` (that's a separate C++ reference). It emits ep_bench-compatible text
so `scripts/parse_results.py` parses both. The 28-case driver issues one `srun` per config:

```bash
# inside an salloc (-N 8); JOBID set; RW holds the checkout + the .sqsh
ssh prenyx "cd $RW && JOBID=<jid> REMOTE_WORK=$RW \
  IMAGE=$RW/flashinfer-ep-pt2605.sqsh ONE_SCRIPT=run_ep_matrix_one_pt.sh \
  bash $RW/flashinfer/benchmarks/run_ep_matrix.sh"
```
- `run_ep_matrix.sh` drives the matrix: LL `{em,rm}` @128 tok/rank and HT `fl` @4096/8192,
  × {8,16,32,64} GPU, × {IB, MNNVL}; `COMMON="--hidden 7168 --top-k 8 --experts 256
  --warmup 20 --iters 100"`. `NCCL_GIN_TYPE=3` per case; MNNVL rows add `NCCL_MNNVL_ENABLE=1`.
- `ONE_SCRIPT=run_ep_matrix_one_pt.sh` selects the PyTorch-image per-rank wrapper (system
  python; HT JIT uses the base `nvcc` with `nccl_device.h` via `CPATH`). It uses a `file://`
  rendezvous (`EP_SYNC`), not MPI.
- Single case, directly: `srun … bash run_ep_matrix_one_pt.sh --algorithm ll --layout em
  --tokens 128 $COMMON`.

**Env knobs** (read by `bench_ep_matrix.py` / the `nccl_ep` wrapper):
- `EP_TIMING=pipeline` — drop the per-iter `torch.cuda.synchronize()` so each stage's host
  prep overlaps the prior kernel (`EP_NO_BARRIER=1` for LL). Default `baseline`.
- `EP_REUSE_PARAMS=1` — reuse the outer `Dispatch/CombineInputParams` objects.
- `NV_FI_EP_FAST_PATH=1` — wrapper fast path (cached FFI wrappers + cached LL recv buffer) —
  see §3.1. **Off by default** (opt-in; read once at `flashinfer.moe_ep` import).
- `EP_PROFILE_HOST=1` (`EP_PROFILE_SKIP=N`) — print a per-step host-wall burn-down on rank 0.

### 2b. Full MoE compute (`bench_moe_ep.py`)
```bash
torchrun --nnodes=$N --nproc_per_node=8 --node_rank=$SLURM_NODEID \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
  benchmarks/bench_moe_ep.py --reference --algorithm ll --backend nccl_ep --quant bf16 \
  --warmup 5 --repeat 20            # --layout rank_major  | --algorithm ht --ep-test-geometry
```
`--reference` selects the ep_bench geometry (hidden 7168, experts 256, top_k 8, 128 tok/rank).
Rank 0 prints one CSV line: `BENCH_CSV,algo,layout,tokens,gpus,backend,quant,dispatch_us,
compute_us,combine_us,e2e_us,tok_s`. Allow ≥45 min walltime — the first run JIT-compiles +
autotunes the trtllm bf16 kernel (~12–15 min).

---

## 3. Results — comm matrix vs ep_bench

All µs/call, BF16, hidden 7168 / top-k 8 / 256 experts, Pre-Nyx B200 (8 GPU/node, NDR IB).
`kernel` = pure GPU device time (ep: CUPTI; FI: Nsight Systems `cuda_gpu_kern_sum`).
`event measured` = the per-call `cudaEvent`-bracketed host-call (ep_bench "total"; FI bench
"kernel-only"). FI columns use `NV_FI_EP_FAST_PATH=1` (§3.1). Cross-node confirmed to 64 GPU.

### 3.1 Same kernels, same GPU time — and the host-call comparison

**Low-Latency (em), 128 tok/rank** (LL 64-GPU fails in FI at `nccl_ep.cc:1491`, a separate
LL-GIN dev-comm issue — omitted):

| GPUs | stage | ep kernel | ep event meas | FI kernel | FI measured (fast) |
|---:|---|---:|---:|---:|---:|
| 8 | dispatch | 43.9 | 50.0 | 45.3 | 65.0 |
| 8 | combine | 43.8 | 49.8 | 39.3 | 71.1 |
| 16 | dispatch | 170.5 | 177.7 | 163.4 | 217.4 |
| 16 | combine | 200.4 | 207.9 | 187.8 | 236.5 |
| 32 | dispatch | 254.7 | 262.6 | 240.8 | 284.7 |
| 32 | combine | 284.7 | 291.0 | 255.0 | 318.6 |

**High-Throughput (flat), 4096 tok/rank:**

| GPUs | stage | ep kernel | ep event meas | FI kernel | FI measured (fast) |
|---:|---|---:|---:|---:|---:|
| 8 | dispatch | 545.9 | 691.2 | 529.2 | 714.6 |
| 8 | combine | 482.2 | 629.8 | 466.5 | 645.8 |
| 16 | dispatch | 1426.6 | 1603.5 | 1412.4 | 1627.2 |
| 16 | combine | 1548.8 | 1697.7 | 1521.8 | 1863.9 |
| 32 | dispatch | 3619.1 | 3801.9 | 3639.8 | 3890.6 |
| 32 | combine | 3518.2 | 3666.9 | 3542.3 | 4118.0 |
| 64 | dispatch | 6016.0 | 6220.7 | 6032.9 | 6284.3 |
| 64 | combine | 6024.0 | 6172.8 | 5933.7 | 7116.3 |

**Findings:**
- **Kernel columns match at every scale** — FlashInfer and ep_bench launch byte-identical
  `internode_ll::dispatch/combine` (LL) and `nccl_ep_jit_ht_dispatch/combine` (HT) kernels
  with the same GPU time, single-node and to 64 GPU.
- **HT: FI ≈ ep_bench on both kernel and event-measured** (dispatch within ~1–3%). The
  host gap over the kernel (~145 µs ep, ~185 µs FI) is inherent to the blocking GIN op —
  present in the C++ caller too, not a Python artifact. (HT combine at 16g+ reads higher in
  FI because its event bracket includes `complete()`'s scan/drain, which ep_bench records
  outside its combine window — a measurement-boundary difference.)
- **LL: FI(fast) is ~1.08–1.3× ep's event-measured** (was ~2.2× before the fast path). ep's
  C++ caller has near-zero host overhead (event ≈ kernel, ~6 µs apart); FI retains ~20 µs/call
  of Python/FFI (input-tensor wrap + launch) — the eager-Python floor.

The full 28-case bandwidth matrix (LL em/rm + HT 4096/8192 × scale × IB/MNNVL) passes the
ep_bench-parity check **24/28** (the 4 LL 64-GPU cases fail at `nccl_ep.cc:1491`).

**Validation.** The fast path round-trips correctly (`--validate`: dispatch+combine OK)
single-node and **multi-node** — LL EXPERT_MAJOR at 8/16/32 GPU and HT FLAT at 8/16/32/64 GPU
(1/2/4/8 nodes, `NCCL_MNNVL_ENABLE=1`). The §3.1 "FI measured (fast)" cross-scale numbers were
all taken with `NV_FI_EP_FAST_PATH=1`. Caveat: the fast path's wrapper-object + recv-buffer
caching currently covers LL EXPERT_MAJOR + HT (LL RANK_MAJOR rebuilds per call), and it makes
LL dispatch return a reused recv buffer — so it stays opt-in until extended/audited and run
through the numerical correctness suite with the flag on. (The per-dispatch recv-count
host-sync readback is now removed unconditionally — `DispatchOutput` no longer carries
`num_tokens` — so it is no longer a fast-path-only win.)

---

## 4. Results — full MoE compute (`bench_moe_ep.py`)

Per-stage median µs (compute = grouped bf16 GEMM via `flashinfer.fused_moe` `TrtllmBf16Config`,
EP-local `top_k=1`). Compute-bound throughout; tok/s ~linear in GPU count.

**LL EXPERT_MAJOR — Pre-Nyx B200, reference config, bf16:**

| GPUs | nodes | MNNVL | dispatch | compute | combine | e2e | tok/s |
|---:|---:|:--:|---:|---:|---:|---:|---:|
| 8 | 1 | 0 | 125.5 | 3529.7 | 362.8 | 4127.6 | 0.25 M |
| 16 | 2 | 1 | 255.2 | 3438.2 | 204.1 | 4080.3 | 0.50 M |
| 32 | 4 | 1 | 401.4 | 3308.5 | 714.2 | 4569.2 | 0.90 M |
| 64 | 8 | 1 | 439.5 | 3264.9 | 633.7 | 4475.2 | 1.83 M |

**LL RANK_MAJOR** trades a layout: EXPERT_MAJOR compute is flat in world size
(`num_experts × tok/rank`); RANK_MAJOR grows (`world × tok/rank × top_k`). They cross at
`world × top_k = num_experts` (world = 32 here) — RANK_MAJOR is ~2.4× faster at 8 GPU
(1.47 vs 3.53 ms compute) but slower past 32. HT FLAT runs single-node (8 GPU: 4096 tok/rank
→ 1.15 M tok/s; 8192 → 1.21 M); cross-node HT is blocked by the library bug above.

(Validated bf16 end-to-end on GB200/SM100, GB300/SM103, GB200-NVL36/Ptyche, and Pre-Nyx
B200, both `nccl_ep` and `nixl_ep`. Correctness is covered in `docs/design_docs/MoE_EP_impl.md`.)

---

## 5. Artifacts / reproduce
- Drivers: `benchmarks/bench_ep_matrix.py`, `benchmarks/bench_moe_ep.py`; SLURM harness
  `benchmarks/run_ep_matrix.sh` + per-rank `run_ep_matrix_one_pt.sh` (PyTorch image) /
  `run_ep_matrix_one.sh` (legacy CUDA-13.0 image).
- nsys kernel times: `nsys profile -t cuda … bench_ep_matrix.py` then
  `nsys stats --report cuda_gpu_kern_sum`. Host burn-down: `EP_PROFILE_HOST=1`.
- ep_bench reference (separate C++ build) for the parity numbers above.
