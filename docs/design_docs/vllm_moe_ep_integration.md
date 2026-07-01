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

> Status: the code in both repos is complete and `py_compile`-clean. Multi-GPU correctness
> and the perf matrix have **not** been run yet — they require GPUs (lyris). The LL
> (BatchedExperts) path maps cleanly onto the FlashInfer API; the HT (Standard) path is
> structurally complete with three on-GPU validation points called out in §6.

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

## 4. Build & install (Docker)

Base image is **forced** to `nvcr.io/nvidia/pytorch:26.05-py3` (CUDA 13.2): cross-node HT
aborts (`nccl_ep.cc:2884 illegal memory access`) on any non-13.2 stack, and the plan requires
2-node HT. vLLM is therefore built **from source** against the image's torch.

```bash
# 1. FlashInfer-EP base (from the FlashInfer repo root)
docker build -f docker/Dockerfile.flashinfer-ep-pytorch -t flashinfer-ep:pt2605 .

# 2. Layer vLLM (from source) + DeepEP on top
docker build -f docker/Dockerfile.vllm-flashinfer-ep \
    --build-arg VLLM_REPO=https://github.com/Anerudhan/vllm.git \
    --build-arg VLLM_REF=feat/flashinfer-ep-all2all \
    --build-arg BUILD_DEEPEP=1 \
    -t vllm-flashinfer-ep:pt2605 .
```

**First build gate:** upstream vLLM pins `torch==2.11.0`; `use_existing_torch.py` strips that
so the build uses NGC-26.05's torch. If they are incompatible the vLLM build fails — validate
before anything else. Runtime env baked in: `NCCL_NET_PLUGIN=none` (HPC-X v8 segfaults NCCL
≥2.30), the 2.30.7 `libnccl` symlink, and the HT-JIT toolchain env.

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
Uses the repo's runner (wires NCCL 2.30.7 + DOCA + the HT JIT include tree). Under SLURM,
**one task per node** (the script owns `torchrun --nproc_per_node=8`):
```bash
srun --ntasks-per-node=1 --gres=gpu:8 benchmarks/run_httest_torchrun.sh
# → tests/moe_ep/test_moe_ep_ht_correctness.py  (HT FLAT dispatch/combine + recv-count)
```
Also run the LL/compute correctness + multirank suites on 8 GPU:
```bash
torchrun --nproc_per_node=8 tests/moe_ep/test_moe_ep_compute_correctness.py
torchrun --nproc_per_node=8 tests/moe_ep/test_moe_ep_layer_multirank.py
```

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
  --model_args "pretrained=Qwen/Qwen3-30B-A3B,data_parallel_size=8,enable_expert_parallel=True,all2all_backend=flashinfer_ep_low_latency,trust_remote_code=True" \
  --tasks gsm8k --num_fewshot 5 --batch_size auto
# temperature 0, seed 42 (harness defaults for gsm8k are greedy)
```

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
`NUM_PROMPTS=1000`**. For each cell, start `vllm serve` with the backend, then:
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
