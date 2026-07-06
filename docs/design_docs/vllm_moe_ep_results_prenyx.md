# vLLM ⇄ `flashinfer.moe_ep` — validated results & reproduction (Pre-Nyx)

Measured results for the two vLLM all2all backends backed by `flashinfer.moe_ep`
(`flashinfer_ep_low_latency`, `flashinfer_ep_high_throughput`) and the exact steps to
reproduce them on the **Pre-Nyx** cluster (SLURM + pyxis/enroot, B200-class GPUs, CUDA 13.2).

For the design/architecture and the full test/bench catalog see
[`vllm_moe_ep_integration.md`](vllm_moe_ep_integration.md).

Code under test:
- FlashInfer: branch `feat/vllm-moe-ep-api` (`github.com/Anerudhan/flashinfer`, `cfc93a9c`).
- vLLM: branch `feat/flashinfer-ep-all2all` (`github.com/Anerudhan/vllm`, `f4e2618`).

---

## 1. Results

> 🛑 **IMPORTANT CAVEAT — the throughput/GSM8K/memory numbers below did NOT exercise the all-to-all
> transport.** They were collected with `--tensor-parallel-size 8 --enable-expert-parallel`
> (`dp_size = 1`). vLLM only routes MoE through the modular EP dispatch/combine path — the only
> path that uses `--all2all-backend` — when **`dp_size > 1`** (`config.py::use_all2all_kernels =
> dp_size > 1 and use_ep`). With `dp_size = 1` it falls back to
> `MoEPrepareAndFinalizeNoDPEPMonolithic`, where experts run locally and are reconciled by the
> ordinary **TP all-reduce**. nsys confirmed this: for *both* `flashinfer_ep_low_latency` and
> `deepep_low_latency` the kernel summaries were dominated by `multimem_all_reduce_kernel` /
> `vllm::cross_device_reduce_*` with **no** dispatch/combine kernels, and both logged
> `Using MoEPrepareAndFinalizeNoDPEPMonolithic`. **Consequence:** the ~1–2% FI-EP↔DeepEP closeness
> below reflects the *shared monolithic path*, NOT a comparison of the two transports.
> **Correctness (§1.1: GAP tests + `--validate`) is unaffected** — those exercise the transport
> directly. The transport-level comparison is being re-collected with `--data-parallel-size 8
> --enable-expert-parallel` (see runbook §3.0); tables below are marked *provisional (monolithic)*
> until then.

All runs: **single node, 8×GPU, 8-way expert parallel, CUDA 13.2**, base image
`nvcr.io/nvidia/pytorch:26.05-py3`, vLLM built from source (upstream `torch==2.11.0` pin
stripped via `use_existing_torch.py`; builds cleanly against NGC 26.05 torch).

### 1.1 Correctness

| Check | LL (`flashinfer_ep_low_latency`) | HT (`flashinfer_ep_high_throughput`) |
|---|---|---|
| `moe_ep` GAP 1/2/3 unit tests (mocked nccl, host-only) | 14 / 14 passed | — |
| EP dispatch+combine `--validate` @ world=8 | ✅ `ll/em per_rank=128 OK` | ✅ `ht/fl per_rank=4096 OK` |
| vLLM e2e smoke (OLMoE-1B-7B, greedy) | ✅ coherent | ✅ coherent (identical text) |
| **GSM8K 5-shot, Qwen3-30B-A3B** — flexible-extract | **0.8522** ± 0.0098 | **0.8575** ± 0.0096 |
| **GSM8K 5-shot, Qwen3-30B-A3B** — strict-match | **0.8939** ± 0.0085 | **0.8969** ± 0.0084 |

Both clear the **≥ 0.80** gate and match the ~0.88 reference for Qwen3-30B-A3B.

### 1.1b DP-EP transport-verified nsys capture (the all2all transport IS exercised)

To exercise the all2all transport in vLLM (not just `--validate`), the MoE layer must take the
**modular EP** path, which requires `dp_size > 1` (§1 caveat). Offline `vllm bench throughput`
can't take `--data-parallel-size` directly — it must be launched under `torchrun --nproc_per_node=8`
with `--distributed-executor-backend external_launcher` (see runbook §3.0). Config: Qwen3-30B-A3B,
8×GPU, `--data-parallel-size 8 --enable-expert-parallel`, `--enforce-eager`, 8/64 prompts, nsys
`-t cuda,nvtx,nccl --trace-fork-before-exec=true`.

With DP-EP the oracle switches the expert backend to a *batched* one and selects the EP
prepare/finalize — the log now prints (per backend):

| Backend | expert backend | prepare/finalize (log) | all2all transport engaged? |
|---|---|---|---|
| `flashinfer_ep_low_latency` | `BATCHED_TRITON` | `Using FlashInferEPLLPrepareAndFinalize` | ✅ |
| `deepep_low_latency` | `BATCHED_TRITON` | `Using DeepEPLLPrepareAndFinalize` | ✅ |
| `deepep_high_throughput` | `TRITON` | `Using DeepEPHTPrepareAndFinalize` | ✅ |
| `flashinfer_ep_high_throughput` | `TRITON` | `Using FlashInferEPHTPrepareAndFinalize` | ✅ (after the 3 HT fixes below) |

**Actual dispatch/combine kernels now on the GPU** (`nsys stats cuda_gpu_kern_sum`), which were
**absent** in the earlier `--tensor-parallel-size 8` (monolithic) capture:

| Backend | dispatch kernel | combine kernel | extra |
|---|---|---|---|
| `flashinfer_ep_low_latency` (nccl.ep) | `nccl_ep::internode_ll::dispatch<…,(ncclEpLayout_t)1,…>` | `nccl_ep::internode_ll::combine<…,(ncclEpLayout_t)1>` | — |
| `deepep_low_latency` (DeepEP+NVSHMEM) | `deep_ep::legacy::internode_ll::dispatch<…>` | `deep_ep::legacy::internode_ll::combine<…>` | `nvshmemi_init_array_kernel<…>` |
| `deepep_high_throughput` (DeepEP intranode) | `deep_ep::legacy::intranode::notify_dispatch<8>` + `dispatch<8,768,8192>` | `intranode::cached_notify_combine<8>` + `combine<bf16,8,768,4096>` | `intranode::layout::get_dispatch_layout` |

So the FlashInfer-EP LL vs DeepEP LL **transport difference is real and now visible**:
FlashInfer-EP LL runs the **`nccl_ep` GIN internode-LL** dispatch/combine; DeepEP LL runs the
**`deep_ep` NVSHMEM internode-LL** dispatch/combine (+ an NVSHMEM init kernel). Both LL backends
picked the *internode* LL path even on a single node.

> ⚠ **Do NOT read these as a perf comparison.** The runs are tiny (8/64 prompts, decode-heavy, no
> CUDA graph), and the LL dispatch/combine kernels **busy-wait on the network**, so their
> `Total/Max` times are dominated by wait/sync (e.g. multi-second `Max` on an 8-prompt run) and are
> not throughput. These captures **prove the transport is exercised and identify the exact kernels**;
> a real perf comparison needs the DP-EP throughput sweep (runbook §3c) with CUDA graphs + larger
> batches. Raw dumps: `$RW/logs/kern_dpep_<backend>.txt`, `kern_a2a_<backend>.txt`,
> `nsys_dpep_<backend>.nsys-rep`.

### 1.1c FlashInfer-EP HT under DP-EP — 3 bugs found & fixed (was SIGABRT, now works)

Enabling the real DP-EP path surfaced three sequential HT-only bugs (all masked by the monolithic
path, which never builds the HT prepare/finalize). Each was root-caused from per-rank
`CUDA_LAUNCH_BLOCKING=1` stderr on Pre-Nyx and fixed; **HT now runs end-to-end and is GSM8K-validated**
(§1.1d):

1. **SIGABRT at `ncclEpCreateGroup` (nccl_ep.cc:1253).** HT asserts
   `max_dispatch_tokens_per_rank ≤ MAX_SUPPORTED_TOKENS_PER_RANK` (build-time `8192` in the nccl4py
   wheel, `nccl/ep/include/nccl_ep/common.hpp`). vLLM sizes the HT fleet from
   `moe.max_num_tokens = scheduler max_num_batched_tokens` (16384 here) → abort on all ranks. LL is
   uncapped, so only HT hit it. **Fix (flashinfer `feat/nvep-default`, `60ff0fc1`):** clamp the HT
   fleet's `max_tokens_per_rank` to the cap in `NcclEpFleet` (+ a clear `MoEEpConfigError` guard in
   `_dispatch_ht`). Run HT with `--max-num-batched-tokens ≤ 8192`.
2. **Triton illegal-memory-access at `moe_align_block_size.py:101` (`expert_map[expert_ids]`).**
   FlashInfer's FLAT recv gives **local** expert ids with `-1` for non-local/padding picks, but
   vLLM's Standard experts feed `topk_ids` through `moe_align` + `expert_map` expecting **global**
   ids (the skip is applied via `expert_map`, never as `-1` in `topk_ids`). **Fix (vLLM
   `flashinfer_ep_ht.py`):** rebuild global ids from `expert_map`, remap `-1`→a non-owned global id
   (which `expert_map` re-tags skipped) — matching the DeepEP HT contract.
3. **`finalize` over-strict assert.** Standard Triton experts declare `TopKWeightAndReduceNoOP`
   (they already applied the dispatched routing weights and reduced their local picks), but finalize
   only accepted `TopKWeightAndReduceDelegate`. FlashInfer HT combine applies **no** weights
   (captured at dispatch) and only reduces per-rank partials across ranks → no double-weighting.
   **Fix (vLLM `flashinfer_ep_ht.py`):** accept `NoOP` too.

`deepep_high_throughput` with the same `TRITON` experts was unaffected (it returns global ids), which
is why only FlashInfer-EP HT hit bugs 2–3.

### 1.1d GSM8K over a REAL DP-EP deployment (transport-exercised accuracy)

Run via `vllm serve --data-parallel-size 8 --enable-expert-parallel --all2all-backend <B>`
(the online server path builds a genuine DP-EP engine, unlike lm_eval's own `data_parallel_size`
which spins up independent monolithic replicas) + `lm_eval --model local-completions` (needs
`pip install lm-eval[api]`). Both backends log the modular EP prepare/finalize and clear ≥0.80:

| Backend | prepare/finalize | flexible-extract | strict-match |
|---|---|---|---|
| `flashinfer_ep_low_latency` | `FlashInferEPLLPrepareAndFinalize` | **0.8582** | **0.8976** |
| `flashinfer_ep_high_throughput` | `FlashInferEPHTPrepareAndFinalize` | **0.8461** | **0.8946** |

On par with each other, with the monolithic-path GSM8K (§1.1), and with the ~0.88 Qwen3-30B-A3B
reference — i.e. the FlashInfer-EP dispatch/combine transport (both LL and HT) is numerically
correct end-to-end, not just at `--validate`. (This *replaces* the earlier "GSM8K is TP/accuracy-only"
caveat: it is now measured through the actual all2all transport.)

### 1.1e DP-EP transport-exercised throughput (all 4 backends)

Offline `vllm bench throughput` launched under `torchrun --nproc_per_node=8
--distributed-executor-backend external_launcher --data-parallel-size 8 --enable-expert-parallel`
(runbook §3.0), Qwen3-30B-A3B, `--enforce-eager`, NP=256, `--dataset-name random`. HT backends
add `--max-num-batched-tokens 8192` (the nccl_ep HT cap); **LL backends must leave the flag
unset** (§1.1f fix #1 — the batched-DP 256 auto-cap must engage; DeepEP-LL also rejects 8192).
Deployment total = sum of the 8 DP ranks' `Throughput:` lines. Every cell verified
`Using <Backend>PrepareAndFinalize` in the log.

**Final (after the §1.1f optimizations; single same-day pass, total tok/s):**

| Backend | 128/128 | 2048/128 | 128/2048 |
|---|---|---|---|
| `flashinfer_ep_low_latency`  | **9,088** | **23,106** | **5,825** |
| `deepep_low_latency`         | 10,116 | 24,013 | 6,595 |
| *FI-LL / DeepEP-LL* | *0.90×* | *0.96×* | *0.88×* |
| `flashinfer_ep_high_throughput` | **6,797** | **45,224** | 3,795 |
| `deepep_high_throughput`     | 5,736 | 35,623 | 4,539 |
| *FI-HT / DeepEP-HT* | ***1.19×*** | ***1.27×*** | *0.84×* |

**FlashInfer-EP HT is now ahead of DeepEP-HT by 19–27%** on the balanced and prefill-heavy
shapes; LL is within 4–12% of DeepEP-LL; the decode-heavy shape is within 12–16% for both modes.

For the record, the **first** transport-exercised pass (before the §1.1f fixes) was
FI-LL 1,926/12,975/1,038 and FI-HT 2,761/21,011/1,485 — i.e. **2–6× behind DeepEP** — so the
gap closure came from the three root-cause fixes below, each GSM8K-gated.

> ⚠ **Interpretation.** Eager-mode (no CUDA graphs) DP-EP runs over the Triton/batched-Triton
> expert backends with a small NP — chosen so the all2all transport is genuinely on the critical
> path and comparable across backends, **not** a production throughput number (the CUDA-graph
> monolithic numbers in §1.2 are much higher). CUDA-graph capture of dispatch/combine is the
> remaining big lever for both backends.

### 1.1f Closing the 2–6× gap: root causes & fixes (perf iteration log)

All found by diffing `nsys cuda_gpu_kern_sum` per-kernel medians between the FI-EP and DeepEP
runs, then `EP_PROFILE_HOST=1` host-phase timing. Each fix validated by GSM8K over the real
DP-EP server (§1.1d method).

1. **[vLLM, 1 line] `flashinfer_ep_low_latency` was missing from `use_batched_dp_moe`**
   (`vllm/config/parallel.py`). That property auto-caps the scheduler to the 256-token
   batched-DP budget for BatchedExperts-format backends (`deepep_low_latency`, `nixl_ep`) —
   without it FI-LL ran with the 8192-token offline default, so the padded
   `[local_experts, max_tokens×world, N]` workspaces were **32× larger** than DeepEP-LL's:
   every `fill_(0)` (394µs vs 13µs), full-workspace `act_and_mul` (2.07ms vs 36µs), the padded
   batched-GEMM grid (1.29ms vs 256µs) and the LL transport slot buffers all paid it.
   *Effect: FI-LL 1,926/12,975/1,038 → 8,854/22,939/5,416.* After the fix the compute kernels
   are **byte-identical** to DeepEP-LL's (280µs/35.8µs/13.2µs medians on both).
2. **[vLLM adapter] HT recv-count trim.** FI-HT ran the whole Standard MoE stack (`moe_align`
   241µs, `count_and_sort` 344µs, 2×`fused_moe` ~1ms, `act_and_mul` 1.06ms, `moe_sum` 344µs)
   over the **static 65,536-row** recv buffer every forward. The GAP-3 `recv_total` counter
   (written by the HT metadata step at create_handle) now trims the compute view to
   `round_up(actual,128)` rows (`.item()` host sync — eager-only); finalize copies the trimmed
   expert output into a persistent full-size buffer (nccl.ep combine needs the address-stable
   static staging; padding rows carry no routing state and are never sent). Also cached the
   static local→global expert-id remap (was 2 `nonzero()` device syncs/layer/step).
   *Effect: FI-HT 2,761/21,011/1,485 → 6,636/44,238/3,755 — ahead of DeepEP-HT on 2 of 3 shapes.*
3. **[flashinfer] Fleet-level host-path caches** (`nccl_ep/handle.py`). vLLM creates a fresh
   `NcclEpHandle` every MoE layer×step (routing binds at `create_handle`), so the per-handle
   `NV_FI_EP_FAST_PATH` caches never hit and each forward paid **~149µs host** (measured:
   FFI descriptor builds 32.6+31.3µs, handle setup 35.9µs, create/destroy/dispatch/combine C
   calls ~45µs). At decode the GPU is host-paced and nccl.ep's **fused send+recv** dispatch
   kernel absorbs the inter-rank lag as in-kernel spin (median 256µs/launch, 33% of GPU time).
   Fix: anchor recv buffers, counter tensors, static FFI tuples and a
   `(data_ptr,dtype,shape)`-keyed wrap memo on the long-lived Fleet — **restricted to tensors
   ≤2 MiB**, because the nccl.ep Tensor wrapper pins the torch tensor (memoizing large prefill
   activations pinned GBs → OOM at `--gpu-memory-utilization 0.9`; found the hard way).
   *Effect: host 149→119µs/layer; decode shapes +4–7%; final matrix above.*

**Kernel-level notes for the remaining decode-heavy delta (~0.85×):**
- FI-LL fused `internode_ll::dispatch` (send+recv in one kernel) median 256µs vs DeepEP's
  split send (14µs) + deferred recv hook (13µs): the fused kernel spins for the slowest peer,
  absorbing per-layer host-path lag; the residual ≈ the remaining ~119µs host path (of which
  ~45µs is create/destroy/dispatch/combine C calls — an nccl.ep handle-reuse/update API would
  remove most of it).
- Remaining levers, in expected-impact order: CUDA-graph capture of dispatch/combine (removes
  host pacing entirely), an nccl.ep API to reuse/update a handle instead of per-forward
  create/destroy, splitting send/recv (staged mode) to overlap like DeepEP's hook, and
  trimming the ~28µs combine-side FFI container builds.

### 1.2 Throughput sweep — FlashInfer-EP vs DeepEP  *(provisional — monolithic path, see §1 caveat)*

`vllm bench throughput`, Qwen3-30B-A3B, 8-GPU EP, 1000 prompts, `--dataset-name random`,
`--enforce-eager`, `--max-model-len 4096`. DeepEP built into `vllm-fi-ep-deepep.sqsh` (§3.3).
Three ISL/OSL shapes: balanced (128/128), prefill-heavy (2048/128), decode-heavy (128/2048).

| ISL / OSL | Backend | total tok/s | output tok/s | req/s |
|---|---|---|---|---|
| **128 / 128** | `flashinfer_ep_low_latency` | 32,506 | 16,253 | 126.98 |
| | `flashinfer_ep_high_throughput` | 32,050 | 16,025 | 125.20 |
| | `deepep_low_latency` | 32,535 | 16,267 | 127.09 |
| | `deepep_high_throughput` | 32,050 | 16,025 | 125.19 |
| **2048 / 128** (prefill-heavy) | `flashinfer_ep_low_latency` | 140,823 | 8,284 | 64.72 |
| | `flashinfer_ep_high_throughput` | 141,744 | 8,338 | 65.14 |
| | `deepep_low_latency` | 141,786 | 8,340 | 65.16 |
| | `deepep_high_throughput` | 143,238 | 8,426 | 65.83 |
| **128 / 2048** (decode-heavy) | `flashinfer_ep_low_latency` | 18,515 | 17,426 | 8.51 |
| | `flashinfer_ep_high_throughput` | 18,461 | 17,376 | 8.48 |
| | `deepep_low_latency` | 18,891 | 17,780 | 8.68 |
| | `deepep_high_throughput` | 18,764 | 17,660 | 8.62 |

**Relative throughput (best-of-each-backend total tok/s, FlashInfer-EP ÷ DeepEP):**

| ISL / OSL | FlashInfer-EP best | DeepEP best | FI-EP / DeepEP |
|---|---|---|---|
| 128 / 128 | 32,506 (LL) | 32,535 (LL) | **0.999** (−0.1%) |
| 2048 / 128 | 141,744 (HT) | 143,238 (HT) | **0.990** (−1.0%) |
| 128 / 2048 | 18,515 (LL) | 18,891 (LL) | **0.980** (−2.0%) |

**Takeaway (provisional):** across all three shapes the four configs land within ~1–2% of each
other — but note (§1 caveat) that with `dp_size=1` all four ran the **same monolithic TP-all-reduce
path**, so this closeness is largely an artifact of the transport not being on the critical path,
not evidence that the FlashInfer-EP and DeepEP dispatch/combine transports perform equivalently.
The genuine transport comparison requires the `--data-parallel-size 8` re-run. (LL vs HT ordering
within a backend is likewise not meaningful here.)

> ⚠ **Correction:** earlier throughput numbers reported for this work (~99–103k tok/s
> "128/128") were actually the vLLM **`sonnet` default dataset (~1024 in / 128 out)** —
> `--input-len`/`--output-len` are ignored unless `--dataset-name random` is passed. The table
> above is the corrected sweep (token counts verified: 128/128→128k+128k, 2048/128→2048k+128k,
> 128/2048→128k+2048k). GSM8K numbers were unaffected (lm_eval uses its own data).

### 1.3 GSM8K — FlashInfer-EP vs DeepEP (accuracy on par)  *(monolithic path — accuracy only)*

| Backend | flexible-extract | strict-match |
|---|---|---|
| `flashinfer_ep_low_latency` | 0.8522 | 0.8939 |
| `flashinfer_ep_high_throughput` | 0.8575 | 0.8969 |
| `deepep_low_latency` | 0.8514 | 0.8931 |
| `deepep_high_throughput` | 0.8544 | 0.8946 |

All four within run-to-run noise. Note this run used the monolithic path (§1 caveat), so it
validates end-to-end model accuracy but does **not** compare the dispatch/combine transports; the
transport is validated separately by the `--validate` correctness checks in §1.1.

### 1.3b Memory footprint  *(monolithic path, see §1 caveat)*

From vLLM's engine-init memory profiling (Qwen3-30B-A3B, 8-GPU EP, `--gpu-memory-utilization
0.9`, `--max-model-len 4096`), **all four backends are identical**:

| Backend | Available KV cache | KV cache size | Max concurrency @ 4096 tok |
|---|---|---|---|
| `flashinfer_ep_low_latency` | 150.45 GiB | 6,573,312 tokens | 1604.8× |
| `flashinfer_ep_high_throughput` | 150.45 GiB | 6,573,312 tokens | 1604.8× |
| `deepep_low_latency` | 150.45 GiB | 6,573,312 tokens | 1604.8× |
| `deepep_high_throughput` | 150.45 GiB | 6,573,312 tokens | 1604.8× |

**EP backend choice is memory-neutral** here — the dispatch/combine transport buffers do not
measurably reduce the usable KV-cache budget on a B200 (≈180 GiB HBM); the model weights +
activation + transport all fit in the non-KV reservation identically for every backend, leaving
the same 150.45 GiB for KV cache. (vLLM in this build does not emit a grep-able
weights/non-torch/activation split; the identical KV-cache size is the operative footprint
metric.)

### 1.4 Multi-node (2-node / 16-GPU) — plumbing works, cross-node init blocked (environmental)

The 2-node path (`benchmarks/_perf_2node.sh`, §4.7) **stands up correctly**: a Ray cluster forms
across both containers (`cluster GPUs so far=16`) and `vllm bench throughput
--data-parallel-size 16 --enable-expert-parallel --distributed-executor-backend ray` with the EP
backend launches. But the vLLM **engine-core init then stalls** (>40 min, no throughput; no error
surfaced in the driver or Ray worker logs). (Original runs used `--tensor-parallel-size 16`, which
would additionally take the monolithic path per §1 caveat; the DP-EP form is the correct config
and stalls at the same cross-node init step below regardless.)

**Isolation test — this is NOT the EP integration.** A plain 2-node run with
`--tensor-parallel-size 16` and **no** `--enable-expert-parallel` / all2all backend **stalls
identically** at engine init. So the blocker is the cluster's **cross-node vLLM/NCCL bring-up**
(fabric / NCCL cross-node env / ray-executor init), independent of FlashInfer-EP or DeepEP.
Resolving it is cluster-config work (e.g. `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` tuning, verifying
IB reachability between the allocated nodes) — out of scope for this integration, which is fully
validated at 8-GPU single node.

### 1.5 Not measured

- **Raw NCCL-EP backend** — exists only in the GitLab `vllm-nccl-moe-integration` fork, not in
  upstream vLLM; N/A here.
- **`vllm bench serve` TTFT/TPOT** — the server path hit a >20-min per-worker FlashInfer
  cubin-download + JIT startup (plus an shm-broadcast wait) on this image; a serving-startup
  issue, not the EP backend (GSM8K drives the same path). Offline `vllm bench throughput` was
  used for the headline numbers.

---

## 2. Environment

- Login: `ssh prenyx` (→ `login-prenyx`, MFA/GSSAPI, `ProxyJump`; a persistent ControlMaster
  socket lets subsequent non-interactive `ssh prenyx '<cmd>'` reuse the session).
- SLURM: `--account=coreai_libraries_cudnn --partition=batch`; **whole-node allocations — do
  NOT pass `--gres`** (`Invalid generic resource` otherwise).
- Containers: pyxis/enroot. `--container-name` does **not** persist across separate `srun`
  jobs — pass `--container-image=<...>.sqsh` every time (squashfs mounts fast).
- Work dir (shared lustre, mounted `/host` in-container):
  `RW=/lustre/fsw/coreai_libraries_cudnn/agopal-moe-ep`.

```bash
RW=/lustre/fsw/coreai_libraries_cudnn/agopal-moe-ep
```

---

## 3. Build the container images (one-time, **no Docker** — pyxis/enroot)

Pre-Nyx has no Docker daemon; images are `.sqsh` files built with **pyxis/enroot** via
`srun --container-save`. Conventions used throughout:
- Enroot registry syntax **`nvcr.io#nvidia/<img>`** (not the docker-style `nvcr.io/nvidia/<img>`).
- **`--container-writable`** — required so in-container `apt`/`pip` installs are captured by
  `--container-save`.
- **Whole-node** allocation (no `--gres`); `--container-mounts=$RW:/host` (shared lustre work dir).
- `docker/Dockerfile.*` are the canonical spec (usable only on a machine with Docker).

### 3.1 FlashInfer-EP base (`flashinfer-ep-pt2605.sqsh`)

> For the results in this doc the base `.sqsh` was **pre-built and reused** (not rebuilt this
> run). The recipe below is how to (re)create it. Mirrors the real `build.sbatch` artifact.

Clone FlashInfer at `feat/vllm-moe-ep-api` into `$RW/flashinfer`, then (the base build is long
— run it as an sbatch or a plain `srun`):

```bash
srun --account=coreai_libraries_cudnn --partition=batch -N1 --ntasks-per-node=1 --time=03:00:00 \
  --container-image="nvcr.io#nvidia/pytorch:26.05-py3" --container-writable \
  --container-save=$RW/flashinfer-ep-pt2605.sqsh --container-mounts=$RW:/host \
  bash -lc 'cd /host/flashinfer && bash docker/install/build_flashinfer_ep_pytorch.sh'
```
As an sbatch wrapper (mirrors `$RW/build.sbatch`):
```bash
#!/bin/bash
#SBATCH -A coreai_libraries_cudnn -p batch -N1 --time=03:00:00 -J fi_ep_build
RW=/lustre/fsw/coreai_libraries_cudnn/agopal-moe-ep
srun --container-image="nvcr.io#nvidia/pytorch:26.05-py3" --container-writable \
     --container-save=$RW/flashinfer-ep-pt2605.sqsh --container-mounts=$RW:/host \
     bash -lc 'cd /host/flashinfer && bash docker/install/build_flashinfer_ep_pytorch.sh'
```
The install script pins `nvidia-nccl-cu13==2.30.7`, `nccl4py[cu13]==0.3.1`, `cuda-core==1.0.1`,
`cuda-bindings==13.2.0` and runs `BUILD_NCCL_EP=1 pip install -e ".[nvep]"` (editable, from
`/host/flashinfer`). See `docker/install/build_flashinfer_ep_pytorch.sh` /
`docker/Dockerfile.flashinfer-ep-pytorch`.

### 3.2 vLLM-from-source image (`vllm-flashinfer-ep.sqsh`)

Clone vLLM at `feat/flashinfer-ep-all2all` into `$RW/vllm`. Build script `build_vllm.sh`
(strip torch pins, install build-deps under `--no-build-isolation`, add a Rust toolchain since
vLLM bundles `rust/Cargo.toml`, pin torch to the base version):

```bash
#!/bin/bash
set -eo pipefail
cd /host/vllm
python use_existing_torch.py
TORCH_VER=$(python -c 'import torch;print(torch.__version__.split("+")[0])')
echo "torch==$TORCH_VER" > /tmp/tc.txt
command -v cargo >/dev/null 2>&1 || \
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"
PIP_CONSTRAINT=/tmp/tc.txt pip install --no-cache-dir -r requirements/build/cuda.txt
MAX_JOBS=32 VLLM_USE_PRECOMPILED=0 \
  PIP_CONSTRAINT=/tmp/tc.txt pip install --no-cache-dir --no-build-isolation -e . -v
# vLLM's deps pull `flashinfer-python` from PyPI and shadow the branch editable (its moe_ep
# lacks EpLayout/FleetAlgoKnobAllocator) — restore the branch editable.
pip install --no-cache-dir --no-build-isolation --no-deps -e /host/flashinfer
python -c 'import vllm, flashinfer; from flashinfer.moe_ep import EpLayout, FleetAlgoKnobAllocator; \
  print("vllm", vllm.__version__, "| flashinfer", flashinfer.__file__)'
```
```bash
srun --account=coreai_libraries_cudnn --partition=batch -N1 --ntasks-per-node=1 --time=03:00:00 \
  --container-image=$RW/flashinfer-ep-pt2605.sqsh --container-writable \
  --container-save=$RW/vllm-flashinfer-ep.sqsh --container-mounts=$RW:/host \
  bash /host/build_vllm.sh
```
(`--container-writable` is required so the vLLM install is captured by `--container-save`; the
base image here is a local `.sqsh` file, so no `nvcr.io#` registry prefix.) Equivalent one-shot
on a Docker host: `docker/Dockerfile.vllm-flashinfer-ep`.

### 3.3 DeepEP image (`vllm-fi-ep-deepep.sqsh`) — for the comparison

Layer DeepEP + NVSHMEM on the vLLM image via vLLM's own installer. **Two fixes are required
on CUDA 13.2:** (a) `UV_BREAK_SYSTEM_PACKAGES=1` (the installer's `uv pip install --system`
trips PEP-668), and (b) **`TORCH_CUDA_ARCH_LIST=10.0a`** — otherwise DeepEP compiles for
`sm_75` and `ptxas` fails (`Feature 'elect'/'mbarrier'/'cp.async.bulk' requires .target
sm_90 or higher`) because its kernels are Hopper/Blackwell-only.

```bash
#!/bin/bash   # build_deepep.sh
export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 UV_SYSTEM_PYTHON=1
set -eo pipefail
export TORCH_CUDA_ARCH_LIST="10.0a"                       # B200; sm_90+ features
command -v uv >/dev/null 2>&1 || pip install -q uv        # installer uses `uv pip install --system`
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"       # ensure uv is on PATH
TORCH_VER=$(python -c 'import torch;print(torch.__version__.split("+")[0])')
echo "torch==$TORCH_VER" > /tmp/tc.txt
PIP_CONSTRAINT=/tmp/tc.txt UV_CONSTRAINT=/tmp/tc.txt \
  bash /host/vllm/tools/ep_kernels/install_python_libraries.sh --workspace /host/ep_kernels_workspace
# DeepEP asserts torch nccl == the nvidia-nccl wheel (2.30.7); the NGC image also ships a system
# libnccl 2.30.4 that torch loads by default. Force the wheel + bake into /etc/profile.d so
# `bash -lc` runtime runs load 2.30.7 too.
NCCL_LIB=$(python -c 'import nvidia.nccl,os;print(os.path.join(list(nvidia.nccl.__path__)[0],"lib"))')
echo "export LD_LIBRARY_PATH=\"$NCCL_LIB:\${LD_LIBRARY_PATH:-}\"" > /etc/profile.d/zz_nccl_wheel.sh
export LD_LIBRARY_PATH="$NCCL_LIB:${LD_LIBRARY_PATH:-}"
# `deep_ep OK` = DeepEP's assert (loaded libnccl == wheel) passed. Don't judge by
# torch.cuda.nccl.version() — that's torch's BUILD-time NCCL (cosmetic), not the loaded .so.
python -c 'import ctypes, torch; torch.cuda.init(); import deep_ep; \
  m=sorted({l.split()[-1] for l in open("/proc/self/maps") if "libnccl.so" in l}); \
  lib=ctypes.CDLL(m[0]); v=ctypes.c_int(); lib.ncclGetVersion(ctypes.byref(v)); \
  print("deep_ep OK; loaded", m, "ncclGetVersion", v.value)'
```
```bash
srun --account=coreai_libraries_cudnn --partition=batch -N1 --ntasks-per-node=1 --time=02:00:00 \
  --container-image=$RW/vllm-flashinfer-ep.sqsh --container-writable \
  --container-save=$RW/vllm-fi-ep-deepep.sqsh --container-mounts=$RW:/host \
  bash /host/build_deepep.sh
```
Installs NVSHMEM 3.3.24 + `DeepEP@d4f41e4e93` (with the installer's CUDA-13 cccl patch).
The DeepEP runs in §4.4/§4.5 use this image with `BACKEND=deepep_low_latency` /
`deepep_high_throughput`.

> **Getting the FlashInfer branch code into the vLLM image at runtime.** `moe_ep` is pure
> Python, editable-installed from `/host/flashinfer`. To run branch code without rebuilding,
> mount a checkout of `feat/vllm-moe-ep-api` and prepend it to `PYTHONPATH`. In this
> validation a **git worktree** `$RW/fi-vllmep` was used (to avoid touching a dirty primary
> checkout); it needs `3rdparty/{cutlass,spdlog,cccl}` present (init or symlink to a populated
> checkout), `flashinfer/_build_meta.py`, and the `flashinfer/data/*` symlinks. Simplest for a
> fresh repro: check the branch out directly in `$RW/flashinfer` before building 3.1 so the
> editable install already points at branch code, and skip `PYTHONPATH`.

---

## 4. Reproduce each result

Common env inside every run: `export PYTHONPATH=/host/fi-vllmep` (only if using the worktree),
`HF_HOME=/host/hf_cache`, `FLASHINFER_WORKSPACE_BASE=/host/fi_cache`,
`FLASHINFER_CUBIN_DIR=/host/fi_cubins` (persist JIT + cubins across runs), `NCCL_GIN_TYPE=3`.

### 4.1 GAP unit tests (host-only, ~1 min)
```bash
srun ... --container-image=$RW/flashinfer-ep-pt2605.sqsh --container-mounts=$RW:/host bash -lc '
  export PYTHONPATH=/host/fi-vllmep; python -m pip install -q pytest
  cd /host/fi-vllmep && python -m pytest \
    tests/moe_ep/nccl_ep/test_gaps_mock.py tests/moe_ep/nccl_ep/test_fleet_mock.py -q'
```

### 4.2 EP dispatch+combine correctness @ world=8 (`--validate`)
Per-rank runner `benchmarks/run_ep_matrix_one_pt.sh` (points `cd /host/flashinfer`; for the
worktree use a copy that `cd`s to `/host/fi-vllmep` + sets `PYTHONPATH`). Launch **8 tasks/node**
with a `file://` rendezvous:
```bash
srun ... --ntasks-per-node=8 --container-image=$RW/flashinfer-ep-pt2605.sqsh \
  --container-mounts=$RW:/host bash -lc \
  'EP_SYNC=/host/sync_ht NCCL_GIN_TYPE=3 bash /host/<wt>/benchmarks/run_ep_matrix_one_pt.sh \
     --algorithm ht --layout fl --tokens 4096 --hidden 7168 --top-k 8 --experts 256 \
     --warmup 5 --iters 10 --validate'
# LL: --algorithm ll --layout em --tokens 128 (same COMMON args)
```
Expect: `[validate] ht/fl world=8 per_rank=4096 dispatch+combine OK` (and `ll/em ... OK`).

> The pytest `tests/moe_ep/test_moe_ep_ht_correctness.py` (launched via `torchrun`) hangs on a
> default-PG collective on this image — use the comm-matrix `file://` `--validate` path above.

### 4.3 vLLM e2e smoke (both backends)
Run a **file-based** program (vLLM uses `spawn`; a heredoc/`stdin` program fails the workers):
```python
# _vllm_smoke_prog.py
import os
from vllm import LLM, SamplingParams
def main():
    llm = LLM(model=os.environ["MODEL"], tensor_parallel_size=8,
              enable_expert_parallel=True, all2all_backend=os.environ["BACKEND"],
              trust_remote_code=True, enforce_eager=True, max_model_len=2048)
    out = llm.generate(["San Francisco is a"], SamplingParams(max_tokens=32, temperature=0.0))
    print(repr(out[0].outputs[0].text))
if __name__ == "__main__":
    main()
```
```bash
srun ... --container-image=$RW/vllm-flashinfer-ep.sqsh --container-mounts=$RW:/host \
  --export=ALL,MODEL=allenai/OLMoE-1B-7B-0924,BACKEND=flashinfer_ep_low_latency bash -lc '
  export PYTHONPATH=/host/fi-vllmep HF_HOME=/host/hf_cache NCCL_GIN_TYPE=3 \
    FLASHINFER_WORKSPACE_BASE=/host/fi_cache
  python -u /host/fi-vllmep/benchmarks/_vllm_smoke_prog.py'
# repeat with BACKEND=flashinfer_ep_high_throughput
```

### 4.4 GSM8K 5-shot (accuracy gate)
```bash
srun ... --container-image=$RW/vllm-flashinfer-ep.sqsh --container-mounts=$RW:/host \
  --export=ALL,BACKEND=flashinfer_ep_low_latency bash -lc '
  export PYTHONPATH=/host/fi-vllmep HF_HOME=/host/hf_cache NCCL_GIN_TYPE=3 \
    FLASHINFER_WORKSPACE_BASE=/host/fi_cache
  python -m pip install -q lm_eval
  lm_eval --model vllm --tasks gsm8k --num_fewshot 5 --batch_size auto \
    --model_args pretrained=Qwen/Qwen3-30B-A3B,tensor_parallel_size=8,enable_expert_parallel=True,all2all_backend=$BACKEND,trust_remote_code=True,max_model_len=4096,enforce_eager=True'
# repeat with BACKEND=flashinfer_ep_high_throughput
```
> ⚠ **GSM8K is an accuracy gate only — it does not exercise the all2all transport.** lm_eval's own
> `data_parallel_size` launches independent replica engines (each `dp_size=1` ⇒ monolithic path),
> so there is no unified EP group to dispatch through; keep `tensor_parallel_size=8`. The transport
> is validated separately by the §4.2 `--validate` round-trip and the nsys dispatch/combine capture
> (runbook §3e).

### 4.5 Throughput sweep
**Use `--dataset-name random` with `--random-input-len`/`--random-output-len`** — otherwise
`vllm bench throughput` falls back to the `sonnet` dataset (~1024/128) and silently ignores
`--input-len`/`--output-len`.
```bash
# ISL/OSL ∈ {128/128, 2048/128, 128/2048}; backend ∈ {flashinfer_ep_low_latency, ...high_throughput}
srun ... --container-image=$RW/vllm-flashinfer-ep.sqsh --container-mounts=$RW:/host \
  --export=ALL,BACKEND=flashinfer_ep_low_latency,ISL=2048,OSL=128 bash -lc '
  export PYTHONPATH=/host/fi-vllmep HF_HOME=/host/hf_cache NCCL_GIN_TYPE=3 \
    FLASHINFER_WORKSPACE_BASE=/host/fi_cache FLASHINFER_CUBIN_DIR=/host/fi_cubins
  cd /tmp; torchrun --nproc_per_node=8 /host/dprun/driver.py --model Qwen/Qwen3-30B-A3B \
    --dataset-name random --random-input-len $ISL --random-output-len $OSL --num-prompts 1000 \
    --data-parallel-size 8 --distributed-executor-backend external_launcher \
    --enable-expert-parallel --all2all-backend $BACKEND \
    --trust-remote-code --max-model-len 4096 --enforce-eager'
```
> ⚠ **DP-EP via `torchrun` + `external_launcher`** (needs `$RW/dprun/driver.py`, runbook §3.0) —
> plain `vllm bench throughput --data-parallel-size 8` errors offline. With `--tensor-parallel-size
> 8` you get the monolithic path (§1 caveat). Each of the 8 ranks prints its own `Throughput:`;
> the deployment total ≈ their sum. Confirm the log says `Using FlashInferEP…/DeepEP…`, not
> `…Monolithic`.

### 4.6 DeepEP comparison (§1.2/§1.3)
Identical to §4.4/§4.5 but use the DeepEP image and DeepEP backend names:
`--container-image=$RW/vllm-fi-ep-deepep.sqsh` and
`BACKEND=deepep_low_latency` / `deepep_high_throughput`.

### 4.7 Multi-node (2-node / 16-GPU)
`benchmarks/_perf_2node.sh` stands up a Ray cluster across the two nodes' containers (rank-0
`ray start --head`, rank-1 `ray start --address=<head>` via a shared `/host/ray_head_ip.$JOBID`
file), waits for 16 GPUs, then runs `vllm bench throughput --data-parallel-size 16
--enable-expert-parallel --distributed-executor-backend ray` with `NCCL_MNNVL_ENABLE=1` for the
cross-node EP fabric. (Data-parallel EP, not TP-16 — see §1 caveat / runbook §3.0.)
```bash
srun --account=coreai_libraries_cudnn --partition=batch -N2 --ntasks-per-node=1 \
  --container-image=$RW/vllm-flashinfer-ep.sqsh --container-mounts=$RW:/host \
  --export=ALL,BACKEND=flashinfer_ep_low_latency,ISL=128,OSL=128 \
  bash /host/fi-vllmep/benchmarks/_perf_2node.sh
```
Warm `/host/fi_cubins` first (from the single-node runs) — a cold cubin cache makes the 16-way
init stall for tens of minutes while every worker downloads FlashInfer cubins.

### 4.8 Memory footprint (§1.3b)
Same as §4.5 but `--num-prompts 8` (init does the memory profiling regardless) and grep the
KV-cache line from the **full** stream (don't `tail`-truncate):
```bash
cd /tmp; torchrun --nproc_per_node=8 /host/dprun/driver.py --model Qwen/Qwen3-30B-A3B \
  --dataset-name random --random-input-len 128 --random-output-len 128 --num-prompts 8 \
  --data-parallel-size 8 --distributed-executor-backend external_launcher \
  --enable-expert-parallel --all2all-backend $BACKEND \
  --gpu-memory-utilization 0.9 --trust-remote-code --max-model-len 4096 --enforce-eager 2>&1 \
  | grep -iE "Available KV cache|GPU KV cache size|Maximum concurrency|Using .*PrepareAndFinalize"
```

---

## 5. Gotchas (learned during this validation)

- **⚠ TP-only EP silently disables the all2all backend (biggest gotcha).** vLLM only takes the
  modular EP dispatch/combine path when `dp_size > 1`
  (`fused_moe/config.py::use_all2all_kernels = dp_size > 1 and use_ep`). Running
  `--tensor-parallel-size 8 --enable-expert-parallel` (dp_size=1) picks
  `MoEPrepareAndFinalizeNoDPEPMonolithic` — experts run locally, reconciled by TP all-reduce, and
  `--all2all-backend` is a **no-op**. Use `--data-parallel-size 8 --enable-expert-parallel`
  instead. **Always confirm** the log prints `Using FlashInferEPLL/HT…` or `DeepEPLL/HT…`
  `PrepareAndFinalize`, never `…Monolithic`. (This is why the first §1.2/§1.3 sweep showed
  FI-EP ≈ DeepEP and identical nsys kernels — the transport was never on the GPU.)
- **No `--gres`** on this cluster (whole-node); `--container-name` doesn't persist across jobs.
- **`NCCL_GIN_TYPE=3`** for the EP GIN transport; multi-node also needs `NCCL_MNNVL_ENABLE=1`.
- **vLLM `spawn`** re-imports the main module → run a real `.py` file, never a heredoc/stdin.
- **`pytest` / `lm_eval`** are not in the image → `pip install` them in the job.
- Don't recursive-glob `/usr/**` inside the container (pathologically slow → looks like a hang).
- Persist JIT to `/host` (`FLASHINFER_WORKSPACE_BASE`) and cubins
  (`FLASHINFER_CUBIN_DIR`) so the ~267-unit CUTLASS MoE compile happens once.
- First vLLM forward JIT-compiles the FlashInfer CUTLASS MoE expert kernel — needs the `cccl`
  submodule present in the checkout used at runtime.
- **DeepEP build on CUDA 13.2:** the vLLM installer calls `uv pip install --system`, so `uv`
  must be present — `pip install uv` first (the NGC/vLLM image has `pip`, not `uv`). Also set
  `UV_BREAK_SYSTEM_PACKAGES=1` (PEP-668) **and** `TORCH_CUDA_ARCH_LIST=10.0a` — without the arch
  it builds for `sm_75` and `ptxas` rejects the Hopper/Blackwell-only features (`elect`,
  `mbarrier`, `cp.async.bulk`).
- **vLLM shadows the branch flashinfer:** `pip install -e .` (vLLM) pulls `flashinfer-python`
  from PyPI and uninstalls the branch editable → `ImportError: cannot import name 'EpLayout'`
  (and the `flashinfer_ep_*` backend breaks: no `FleetAlgoKnobAllocator`). Fix: re-run
  `pip install --no-build-isolation --no-deps -e /host/flashinfer` after the vLLM install (or
  set `PYTHONPATH=/host/flashinfer` on every run). Verify `flashinfer.__file__` →
  `/host/flashinfer/...`.
- **DeepEP NCCL-version assert:** DeepEP requires torch's loaded NCCL to equal the nvidia-nccl
  wheel (2.30.7), but the NGC image also has a system `libnccl.so.2.30.4` that torch loads by
  default → `AssertionError: Invalid NCCL versions: ...2.30.4 (loaded) v.s. ...wheel...`. Fix:
  prepend the wheel's `nvidia/nccl/lib` to `LD_LIBRARY_PATH` (2.30.x is ABI-compatible) and bake
  it into `/etc/profile.d` so `bash -lc` runs inherit it.
