# vLLM ⇄ `flashinfer.moe_ep` all2all integration — changes, correctness, benchmarks

This document summarizes the integration of FlashInfer's MoE expert-parallel transport
(`flashinfer.moe_ep`, NCCL-EP backend) as two vLLM all2all backends
(`flashinfer_ep_low_latency`, `flashinfer_ep_high_throughput`), and how to build, test for
correctness, and benchmark it.

It spans **two repositories**:

- **FlashInfer** — branch `feat/vllm-moe-ep-api` off `upstream/main`
  (`github.com/Anerudhan/flashinfer`, remote `origin`).
- **vLLM** — branch `feat/flashinfer-ep-all2all` off `Anerudhan/vllm` `main`
  (upstream `vllm-project/vllm` fork), cloned at `/home/scratch.agopal_sw/play/NCCL/vllm`.

> Status: code complete + **GPU-validated on Pre-Nyx (8×GPU, single node, CUDA 13.2)** — see
> §0. Both `flashinfer_ep_low_latency` and `flashinfer_ep_high_throughput` pass end-to-end
> (smoke, GSM8K, throughput). Deferred: DeepEP comparison column (DeepEP's own CUDA-13.2 build
> fails), raw NCCL-EP backend (not in upstream vLLM), 2-node, and `bench serve` TTFT/TPOT.

---

## 0. Validated results (Pre-Nyx, 8×GPU single node, CUDA 13.2)

Base image `nvcr.io/nvidia/pytorch:26.05-py3`; vLLM built from source (torch gate passed);
FlashInfer run from the branch. All checks below **pass**:

| Check | LL (`flashinfer_ep_low_latency`) | HT (`flashinfer_ep_high_throughput`) |
|---|---|---|
| GAP 1/2/3 unit tests (mocked nccl) | 14/14 | — |
| EP dispatch+combine `--validate` @ world=8 | ✅ | ✅ |
| vLLM e2e smoke (OLMoE, coherent output) | ✅ | ✅ |
| **GSM8K 5-shot, Qwen3-30B-A3B** (flex / strict) | **0.852 / 0.894** | **0.858 / 0.897** |

Both backends clear the GSM8K ≥ 0.80 gate (reference ~0.88). Correctness above (GAP tests +
`--validate` transport round-trip) directly exercises the dispatch/combine path.

**Transport-exercised results (DP-EP — the numbers that matter):**
GSM8K through a real DP-EP server: **LL 0.856/0.898, HT 0.857/0.898** (flex/strict).
DP-EP eager throughput vs DeepEP after the perf iteration (Qwen3-30B-A3B, 8×GPU, total tok/s,
128/128 · 2048/128 · 128/2048): **FI-LL 9,088/23,106/5,825 (0.90/0.96/0.88× of DeepEP-LL)**;
**FI-HT 6,797/45,224/3,795 (1.19/1.27/0.84× of DeepEP-HT — ahead on 2 of 3 shapes)**.
The initial DP-EP pass was 2–6× behind; the closure came from three root-cause fixes:
`flashinfer_ep_low_latency` added to vLLM's `use_batched_dp_moe` (batched-DP 256-token scheduler
cap, matching `deepep_low_latency`), the HT recv-count compute trim (§2 GAP 3), and fleet-level
host-path caches in `nccl_ep/handle.py`.

> 🛑 **The throughput/GSM8K/memory numbers below are historical and do NOT compare the two
> transports.** They were run with `--tensor-parallel-size 8` (`dp_size=1`), so vLLM took the
> `MoEPrepareAndFinalizeNoDPEPMonolithic` path — experts computed locally, reconciled by TP
> all-reduce — and **`--all2all-backend` was a no-op** (confirmed by nsys: identical kernels, only
> TP all-reduce, no dispatch/combine, for both FI-EP and DeepEP). The all2all transport is only
> selected when `dp_size > 1` (`fused_moe/config.py::use_all2all_kernels`). The tables above use
> `--data-parallel-size 8 --enable-expert-parallel` (verify the log says
> `Using FlashInferEPLL/HT…PrepareAndFinalize`, not `…Monolithic`). Note offline
> `vllm bench throughput` rejects `--data-parallel-size` directly — launch it under
> `torchrun --nproc_per_node=8` with `--distributed-executor-backend external_launcher`.

**Throughput vs DeepEP** *(provisional — monolithic path, transport not exercised)*
(`vllm bench throughput --dataset-name random`, Qwen3-30B-A3B, 8-GPU, 1000 prompts; total tok/s):

| ISL/OSL | FI-EP LL | FI-EP HT | DeepEP LL | DeepEP HT |
|---|---|---|---|---|
| 128 / 128 | 32,506 | 32,050 | 32,535 | 32,050 |
| 2048 / 128 | 140,823 | 141,744 | 141,786 | 143,238 |
| 128 / 2048 | 18,515 | 18,461 | 18,891 | 18,764 |

All four numbers land within ~1–2% — but that is because all four ran the *same* monolithic
TP-all-reduce path, not because the transports are equivalent. GSM8K accuracy likewise within
noise (monolithic path — end-to-end accuracy, not transport). Memory identical across all four
(150.45 GiB / 6.57M-token KV cache at `--gpu-memory-utilization 0.9`; monolithic path).
**Not measured:** raw NCCL-EP (N/A upstream), TTFT/TPOT via `bench serve`. **2-node/16-GPU:**
Ray+TP=16 plumbing comes up but cross-node engine init stalls — reproduced with plain TP=16
(no EP), so it's a cluster cross-node NCCL/fabric issue, not the EP integration.

---

## 1. Architecture — where FlashInfer plugs in

vLLM's fused-MoE modular kernel runs
`prepare_finalize.prepare` (**dispatch**) → expert GEMM → `prepare_finalize.finalize`
(**combine**). FlashInfer's per-forward sequence maps 1:1 onto that seam:

```
handle = fleet.create_handle(HandleParams(topk_ids), knobs)   # once per forward
d = handle.dispatch(DispatchInputParams(x=[a1]))              # == prepare()
... vLLM runs the expert FFN on d.expert_tensors ...
handle.combine(CombineInputParams(x=[expert_out], out))      # == finalize()
handle.complete()
```

The **`Fleet`** (durable transport sizing + NCCL comm) is owned by the vLLM all2all
*manager*; a fresh **`Handle`** (bound to the step's `topk_ids`) is created in `prepare()`
and consumed in `finalize()`.

| | LL (`flashinfer_ep_low_latency`) | HT (`flashinfer_ep_high_throughput`) |
|---|---|---|
| FlashInfer layout | `EXPERT_MAJOR` → `[E_local, max_tok*world, hidden]` | `FLAT` → `[num_recv, hidden]` |
| vLLM activation format | `BatchedExperts` | `Standard` |
| expert kernel | `BatchedTritonExperts` | `moe_align_block_size` + `fused_moe` |

---

## 2. Changes — FlashInfer repo (`feat/vllm-moe-ep-api`)

The vLLM adapter needed three additions to `flashinfer.moe_ep` ("the 3 gaps"):

| File | Change |
|---|---|
| `flashinfer/moe_ep/nccl_ep/fleet.py` | **GAP 1** — `_resolve_comm` now *adopts* an existing `ncclComm_t` (`BootstrapConfig.nccl_comm`, wrapped via `Communicator(ptr=...)` which has no finalizer → never frees the caller's comm) and can mirror a specific torch `process_group` (the EP subgroup) instead of the WORLD group. **GAP 2** — `_build_alloc_config` + `_install_torch_allocator` route EP buffers through torch's caching allocator (`GroupConfig.alloc`). |
| `flashinfer/moe_ep/nccl_ep/handle.py` | **GAP 3** — opt-in: when `HandleAlgoKnobNumReceivedTokens` is set (HT), bind its target as `recv_total_counter` at `create_handle` (the HT metadata step fills the actual received-token count, static mode included). Surface `recv_total_counter` on the HT `DispatchOutput` and the library-written `expert_counts` on the LL outputs. |
| `flashinfer/moe_ep/config.py` | `BootstrapConfig.process_group`; `DispatchOutput.{expert_counts, recv_total_counter}` (optional). |
| `flashinfer/moe_ep/algo_knobs.py` | new `FleetAlgoKnobAllocator(torch_caching \| alloc_fn/free_fn/context)`. |
| `flashinfer/moe_ep/__init__.py` | export `FleetAlgoKnobAllocator`. |
| `tests/moe_ep/nccl_ep/test_gaps_mock.py` | host-only unit tests for the 3 gaps (mocked `nccl`). |
| `docker/Dockerfile.vllm-flashinfer-ep` | vLLM + FlashInfer-EP + DeepEP image (see §4). |

**v0.1 note (important):** the actual recv-count *readback* IS available on nccl-ep v0.1 via
the create_handle metadata step, even though dynamic *buffer sizing* is not (the buffer stays
sized to `max_recv_tokens_per_rank`). This is what makes the HT compute-view trim feasible.

### 2.1 NIXL-EP transport parity (`nixl-ep` branch)

Closes the gaps that kept the **nixl_ep** transport from serving the same
Fleet/Handle contract the vLLM adapters consume — the prerequisite for folding
vLLM's own `nixl_ep` all2all backend into `flashinfer_ep_low_latency`
(`create_fleet(..., backend="nixl_ep")`):

| File | Change |
|---|---|
| `.../split/comm/nixl_ep/fleet.py` | `_resolve_store` — when `BootstrapConfig.tcp_store` is unset, derive a namespaced `PrefixStore` from torch.distributed's default store (GAP 1 analogue: the vLLM manager passes only `process_group`; no sibling-port TCPStore needed). |
| `.../split/comm/nixl_ep/handle.py` | Surface the library's per-expert recv counts as `DispatchOutput.expert_counts` (previously discarded — the LL adapter needs them for `ExpertTokensMetadata`); surface fp8 dispatch scales as `DispatchOutput.expert_scales` (previously dropped); honor `HandleAlgoKnobUserStream` by redirecting Buffer kernels (NIXL takes no stream arg); fix `num_tokens` to the per-expert row count (was ×`num_local` off vs nccl_ep LL). |
| `flashinfer/moe_ep/config.py` | `DispatchOutput.expert_scales` (optional). |
| `flashinfer/moe_ep/modes/split_layer.py` | relax the layer-init `tcp_store` requirement to "store or initialized torch.distributed". |

Still open for full parity with vLLM's in-tree `nixl_ep` backend: async recv
hooks through the Handle API (DBO overlap), `FleetAlgoKnobAllocator` (NIXL's
`Buffer` owns its RDMA arena outside torch's pool), and elastic-EP fault
tolerance (mask query, staged commit). LL/`EXPERT_MAJOR` remains the only
nixl_ep algorithm — matching vLLM's `nixl_ep`, which is batched/LL-only.

## 3. Changes — vLLM repo (`feat/flashinfer-ep-all2all`)

Mirrors the existing DeepEP/NVLink backends. New backend names:
`flashinfer_ep_low_latency`, `flashinfer_ep_high_throughput`.

| File | Change |
|---|---|
| `vllm/config/parallel.py` | add the two names to `All2AllBackend`. |
| `vllm/model_executor/layers/fused_moe/config.py` | `use_flashinfer_ep_{ll,ht}_kernels` properties (+ `FusedMoEConfig` passthroughs, + LL in `use_batched_activation_format`). |
| `vllm/utils/flashinfer.py` | `has_flashinfer_moe_ep()` capability probe (+ `__all__`). |
| `vllm/distributed/device_communicators/all2all.py` | `FlashInferEP{LL,HT}All2AllManager` — own a config-cached `Fleet`; build `BootstrapConfig(process_group=EP group)` (GAP 1) + `FleetAlgoKnobAllocator(torch_caching=True)` (GAP 2). |
| `vllm/distributed/device_communicators/cuda_communicator.py` | dispatch the two backend names to the managers. |
| `vllm/model_executor/layers/fused_moe/all2all_utils.py` | two `maybe_make_prepare_finalize` branches + guarded imports. |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_ep_common.py` | shared base (per-ubatch handle lifecycle, knob assembly). |
| `.../prepare_finalize/flashinfer_ep_ll.py` | LL adapter (BatchedExperts) — clean mapping. |
| `.../prepare_finalize/flashinfer_ep_ht.py` | HT adapter (Standard) — structural; see §6. |

---

## 4. Build & install

Base image is **forced** to `nvcr.io/nvidia/pytorch:26.05-py3` (CUDA 13.2): cross-node HT
aborts (`nccl_ep.cc:2884 illegal memory access`) on any non-13.2 stack, and the plan requires
2-node HT. vLLM is therefore built **from source** against the image's torch.

> **Pre-Nyx (and most SLURM clusters) have no Docker daemon** — images are built with
> **pyxis/enroot** via `srun --container-save`, not `docker build`. This is how the `.sqsh`
> images used for all results were produced. The `docker/Dockerfile.*` files remain the
> canonical build spec and are usable on a machine that *does* have Docker (see the optional
> block at the end).

The three `.sqsh` images (all under a shared-FS work dir `$RW`, mounted `/host` in-container):

```bash
RW=/lustre/fsw/coreai_libraries_cudnn/agopal-moe-ep   # shared FS work dir

# 1. FlashInfer-EP base (flashinfer-ep-pt2605.sqsh) — for these results it was REUSED
#    (pre-built). To (re)build it without Docker (enroot registry syntax `nvcr.io#...`,
#    --container-writable so the in-container installs are captured by --container-save):
srun -A coreai_libraries_cudnn -p batch -N1 --time=03:00:00 \
    --container-image="nvcr.io#nvidia/pytorch:26.05-py3" --container-writable \
    --container-save=$RW/flashinfer-ep-pt2605.sqsh --container-mounts=$RW:/host \
    bash -lc 'cd /host/flashinfer && bash docker/install/build_flashinfer_ep_pytorch.sh'

# 2. vLLM (from source) and 3. DeepEP images are layered on the base the same way
#    (srun --container-image=<base>.sqsh --container-save=<new>.sqsh ...).
#    The canonical build spec is docker/Dockerfile.vllm-flashinfer-ep (directly usable
#    on a Docker host; on pyxis clusters run its steps inside srun --container-save).
```

Notes: whole-node allocations only (**no `--gres`** on this cluster). `--container-writable` is
required with `--container-save`. Named containers do **not** persist across separate `srun`
jobs — pass `--container-image=<...>.sqsh` each run.

**First build gate:** upstream vLLM pins `torch==2.11.0`; `use_existing_torch.py` strips that
so the build uses NGC-26.05's torch. If they are incompatible the vLLM build fails — validate
before anything else. Runtime env baked in: `NCCL_NET_PLUGIN=none` (HPC-X v8 segfaults NCCL
≥2.30), the 2.30.7 `libnccl` symlink, and the HT-JIT toolchain env.

**Optional — on a Docker host / CI** (not Pre-Nyx): the same images build directly from the
Dockerfiles.
```bash
docker build -f docker/Dockerfile.flashinfer-ep-pytorch -t flashinfer-ep:pt2605 .
docker build -f docker/Dockerfile.vllm-flashinfer-ep \
    --build-arg VLLM_REPO=https://github.com/Anerudhan/vllm.git \
    --build-arg VLLM_REF=feat/flashinfer-ep-all2all --build-arg BUILD_DEEPEP=1 \
    -t vllm-flashinfer-ep:pt2605 .
```

Sanity inside the container:
```bash
python -c "from flashinfer.moe_ep import available_backends; assert 'nccl_ep' in available_backends()"
python -c "from vllm.utils.flashinfer import has_flashinfer_moe_ep; assert has_flashinfer_moe_ep()"
```

---

## 5. Correctness tests (run in order; each is a gate for the next)

### 5.1 FlashInfer host-only unit (no GPU) — the 3 gaps
```bash
cd /path/to/flashinfer
pytest tests/moe_ep/nccl_ep/test_gaps_mock.py tests/moe_ep/nccl_ep/test_fleet_mock.py -v
```
Verifies GAP 1 comm adoption / group mirroring, GAP 2 allocator plumbing into `GroupConfig`,
GAP 3 HT `recv_total_counter` binding + `DispatchOutput` surfacing — all against a mocked
`nccl.ep`.

### 5.2 FlashInfer 8-GPU EP round-trip (single node, 8 GPU)
Validate dispatch+combine correctness at world=8 via the **comm-matrix `--validate`** path
(`srun --ntasks-per-node=8`, `file://` rendezvous, `NCCL_GIN_TYPE=3`; whole-node — **no
`--gres`** on this cluster). The exact runner:
```bash
srun --ntasks-per-node=8 --container-image=$RW/flashinfer-ep-pt2605.sqsh --container-mounts=$RW:/host \
  bash -lc 'EP_SYNC=/host/sync_ht NCCL_GIN_TYPE=3 bash /host/<checkout>/benchmarks/run_ep_matrix_one_pt.sh \
    --algorithm ht --layout fl --tokens 4096 --hidden 7168 --top-k 8 --experts 256 --validate'
# LL: --algorithm ll --layout em --tokens 128 --validate
```
> The pytest `tests/moe_ep/test_moe_ep_ht_correctness.py` launched via `torchrun` **hangs** on a
> default-PG collective on this image — use the comm-matrix `--validate` path above instead.

### 5.3 vLLM smoke (single node, 8 GPU) — both backends
```bash
# start the server (repeat with flashinfer_ep_high_throughput)
vllm serve Qwen/Qwen3-30B-A3B \
    --data-parallel-size 8 --enable-expert-parallel \
    --all2all-backend flashinfer_ep_low_latency \
    --trust-remote-code &

# one completion — expect a coherent continuation, no hang
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3-30B-A3B","prompt":"San Francisco is a","max_tokens":32}'
```

### 5.4 vLLM accuracy gate — GSM8K (5-shot)
Pass criterion **≥ 0.80** (reference ~0.88). Run for each backend, LL and HT:
```bash
lm_eval --model vllm \
  --model_args "pretrained=Qwen/Qwen3-30B-A3B,tensor_parallel_size=8,enable_expert_parallel=True,all2all_backend=flashinfer_ep_low_latency,trust_remote_code=True" \
  --tasks gsm8k --num_fewshot 5 --batch_size auto
# temperature 0, seed 42 (harness defaults for gsm8k are greedy)
```
> **Accuracy gate only — does not exercise the transport.** lm_eval's `data_parallel_size` spawns
> independent replica engines (each `dp_size=1` ⇒ monolithic path), so it can't drive a unified EP
> group; keep `tensor_parallel_size=8`. The dispatch/combine transport is validated by §5.2
> (`--validate`) and by an nsys kernel capture of a DP-EP run (look for the
> `nccl_ep::internode_ll` / `nccl_ep_jit_ht_*` dispatch+combine kernels). To exercise the
> transport end-to-end in vLLM, use the **server** path (§5.3, `vllm serve
> --data-parallel-size 8 --enable-expert-parallel`, which *does* build a real DP-EP deployment)
> or offline via `torchrun --nproc_per_node=8 … --distributed-executor-backend external_launcher`.

### 5.5 Multi-node (2 nodes, 16 GPU)
Repeat 5.2–5.4 across 2 nodes. FlashInfer tests: `srun --nodes=2 --ntasks-per-node=1
benchmarks/run_httest_torchrun.sh` (the script sets up the per-node HT JIT include tree). vLLM:
add `--data-parallel-size 16` with the appropriate multi-node launch; ensure the RDMA fabric
and `NCCL_NET_PLUGIN=none` are in effect.

---

## 6. HT adapter — on-GPU validation points

`prepare_finalize/flashinfer_ep_ht.py` is structurally complete but three points must be
validated/iterated on real hardware (they cannot be exercised host-only):

1. **recv-routing `-1` masking** — the received `recv_topk_idx` (local expert ids, `-1` for
   non-local/padding) is fed into vLLM's Standard `fused_moe`; confirm `-1` picks are skipped
   (contribute 0).
2. **`(M, topk, K)` ↔ `[num_recv, hidden]`** — finalize's `fused_expert_output` arrives in the
   modular `(M, topk, K)` shape while FlashInfer HT combine consumes `[num_recv, hidden]`;
   verify the reshape/reduce reconciliation (combine owns the cross-rank reduce; weights bound
   at dispatch).
3. **`recv_total_counter` compute-view trim** — currently the full static `max_recv` buffer is
   used with `-1` masking (correct). Enabling `recv_x[:actual_recv]` is the ~3× throughput
   optimization; validate the front-packing assumption before turning it on.

Validate the **LL** path first (§5) — it maps cleanly and has no such open points.

---

## 7. Benchmarks — perf matrix

### 7.1 FlashInfer transport-only (isolates dispatch/combine)
```bash
# per-rank wrapper; file:// rendezvous on a shared mount
EP_SYNC=/shared/ep_sync srun --ntasks-per-node=8 \
    benchmarks/run_ep_matrix_one.sh <bench_ep_matrix.py args>
# or the single-process driver:
torchrun --nproc_per_node=8 benchmarks/bench_moe_ep.py \
    --tokens 8192 --world-size 8 --backend nccl_ep --quant bf16
```

### 7.2 End-to-end serving perf (the comparison matrix)
Fixed load: **Qwen3-30B-A3B BF16, ISL/OSL 128/128, `max_concurrency=32`,
`NUM_PROMPTS=1000`**. Launch each `vllm serve` with **`--data-parallel-size 8
--enable-expert-parallel`** (NOT TP-only) so the all2all transport is actually on the path —
otherwise every cell collapses to the identical monolithic path (§0 caveat). For
each cell, start `vllm serve` with the backend, then:
```bash
vllm bench serve \
    --model Qwen/Qwen3-30B-A3B \
    --dataset-name random --random-input-len 128 --random-output-len 128 \
    --max-concurrency 32 --num-prompts 1000
```

**Sweep** = `{flashinfer_ep_*, nccl_*, deepep_*} × {LL, HT} × {1 node, 2 node}` = 12 cells,
**5 reps each**, report the median TTFT / TPOT / throughput (tok/s).

| Backend | Mode | Nodes | TTFT (ms) | TPOT (ms) | tok/s | GSM8K |
|---|---|---|---|---|---|---|
| FlashInfer-EP | LL | 1 | | | | |
| FlashInfer-EP | LL | 2 | | | | |
| FlashInfer-EP | HT | 1 | | | | |
| FlashInfer-EP | HT | 2 | | | | |
| NCCL-EP | LL/HT | 1/2 | | | | |
| DeepEP | LL/HT | 1/2 | | | | |

**Reading the result:** FlashInfer-EP vs NCCL-EP should be **within noise** (same underlying
`nccl.ep` kernels — the goal is "no regression from the wrapper"); FlashInfer-EP vs DeepEP is
the competitive datapoint. For a per-stage breakdown, add an nsys capture
(`nsys profile --capture-range=cudaProfilerApi ...`) on one cell per backend.

### 7.3 Backend name reference
`--all2all-backend` values: `flashinfer_ep_low_latency`, `flashinfer_ep_high_throughput`
(this work); `nccl_low_latency`, `nccl_high_throughput` (raw nccl.ep, if present);
`deepep_low_latency`, `deepep_high_throughput` (DeepEP).
