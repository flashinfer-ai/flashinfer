# SM89 FP8 MLA: Kernel Design and Validation

This document explains how the SM89 FP8 MLA path fits the shared-memory budget,
assigns QK/PV work, and preserves the existing API. It also provides correctness
and performance reproduction commands for RTX 4090.

## 1. Scope and Supported Inputs

The extension preserves the public `plan` and `run` signatures of
`BatchMLAPagedAttentionWrapper(backend="fa2")`, including the existing cache scaling
semantics.

| Property | Supported configuration |
|---|---|
| GPU | SM89; validated on RTX 4090 |
| Query length | One token per request; batching is supported |
| Query / output | BF16 |
| KV cache | Split paged FP8 E4M3 CKV/KPE, dimensions 512/64 |
| Page-table metadata | Existing CSR and dense forms |
| Execution | Causal or non-causal, optional LSE, CUDA Graph replay/replan |

Each request has one query token and may have multiple query heads. The SM89 FP8 path
rejects multi-token prefill, FP16 queries, FP8 E5M2 caches, and dimensions other than
CKV512/KPE64 before launching a kernel.

At upstream commit `595bc5746cd57efc0006ff888b4bf8144f8235b9`, this FP8 FA2
configuration is rejected with an SM90 requirement. Section 4 includes a command to
reproduce that rejection on an unmodified checkout.

## 2. Kernel Design

### Dataflow and Compute Precision

CKV and KPE are stored as FP8 E4M3 in global memory to reduce cache footprint and data
transfers. Inside attention, each cache tile is converted, scaled, and stored as BF16 in
shared memory. Tensor Core products use BF16 operands and FP32 accumulators, and the API
returns BF16 output. FP8 is the cache format; this path does not use FP8 MMA.

Q32/Q64 denote packed query/head rows per block, not decode-token counts. KV16/KV32
denote compute-tile widths, independent of the cache page size. Here, each request has
one query token, so the packed rows index query heads.

The per-tile dataflow is:

```text
Paged FP8 CKV/KPE in global memory
    │ page-table lookup + asynchronous load
    ▼
Raw FP8 shared tile
    │ convert, scale and repack
    ▼
BF16 shared tile ──► QK → softmax → PV ──► FP32 O accumulators (registers)
    │                                      │ after the assigned KV range
    │ current tile is consumed             ▼
    └─ next FP8 tile loads into raw storage  output staging / split merge → BF16 O
```

The raw FP8 and BF16 tiles occupy separate shared-memory regions. Barriers and copy
waits control their reuse, as described below.

### Shared Memory Budget

SM89 permits 101376 bytes (99 KiB) of application shared memory per block. The upstream
API rejects this FP8 configuration before launch. If that guard alone were removed, the
generic launcher would choose Q64/KV32 with one stage, requiring 129536 bytes (126.5
KiB). This is a layout calculation, not a measured failed launch. See
`SharedStorageQKVO` and `BatchMLAPagedAttention` in
[`mla.cuh`](../../include/flashinfer/attention/mla.cuh).

The generic FP8 layout keeps a raw FP8 tile alongside a BF16 tile repacked for MMA. With
CKV512/KPE64, each row contains 576 elements. Q64 needs `64 * 576 * 2 = 73728` bytes;
KV32 needs `32 * 576 = 18432` raw bytes plus `32 * 576 * 2 = 36864` repacked bytes.
Including 512 bytes of group-statistics storage gives 129536. FP8 reduces cache storage
in global memory, but this conversion layout holds two representations in shared memory.

The following counts are bytes per block for one stage and BF16 Q/O:

| Shared allocation | Existing BF16 Q64/KV16 | Generic FP8 Q64/KV32 | This PR FP8 Q32/KV32 | This PR FP8 Q64/KV16 |
|---|---:|---:|---:|---:|
| Q content and position | 73728 | 73728 | 36864 | 73728 |
| Raw FP8 CKV/KPE | 0 | 18432 | 18432 | 9216 |
| BF16 CKV/KPE | 18432 | 36864 | 36864 | 18432 |
| Statistics and alignment | 544 | 512 | 256 | 0 |
| **Total** | **92704** | **129536** | **92416** | **101376** |

The BF16 path reads already converted cache tiles, so it needs no raw FP8 staging area
or full repacking buffer. Its 544-byte remainder includes 512 bytes of statistics and 32
bytes from aligned placeholder fields. The separate converter uses registers and
global-memory output, not a full-cache shared allocation. This alternative fits by
placing the complete BF16 cache copy in global memory, at the cost of conversion and
extra global-memory traffic.

Both SM89 configurations retain conversion inside attention and keep the raw FP8 and
repacked BF16 representations. Q32 halves Q storage; Q64 instead halves the KV tile and
removes the unused 512-byte cross-group statistics buffer (`QK_SHARD=false`). Without
that removal, Q64/KV16 would need 99.5 KiB. The output staging array aliases the input
storage through an existing union, so its size is not added to these totals. Recorded
NCU dynamic shared sizes are 92416 and 101376 bytes for the two new paths, with no
static shared allocation. CUDA reserves another 1024 bytes per block; both layouts allow
only one resident block per SM. Both shared memory and registers limit residency to one block in the two
profiled configurations; fitting the shared budget alone does not remove this limit.

### Warp Assignment and Buffer Reuse

Both paths use 256 threads per block and follow this sequence:

```text
load raw FP8 tile → repack/scale into BF16 shared → finish all raw readers
    → prefetch next FP8 tile into raw storage while QK/PV use current BF16 tile
    → finish current readers and wait for next load before repacking again
```

Q32 has four groups of two warps, each owning 128 output columns. This gives 64 FP32
output accumulators per thread, versus 128 for Q64's two groups of four warps. These
counts describe output state only, not total compiler registers.

Q32 splits QK across groups 0 and 1. Each group computes one 16-column half of
KV32 with two warps, using the fixed `s_frag[0]` register fragment. Four warps
therefore produce the full Q32-by-KV32 score tile without duplicate QK work.
After all KPE readers finish, a block barrier permits reuse of the retired
4096-byte BF16 KPE buffer for FP32 scores. The two producers write disjoint
halves; a second barrier makes both halves visible to all four output groups.
Each group reads the complete score tile, maintains its softmax state and computes
its own 128 PV columns. Fixed fragment indexing avoids the local-memory traffic
observed with a runtime-indexed register array.

Q64 uses two groups of four warps. Group 0 computes QK once; group 1 receives its
scores instead of recomputing them. The retired BF16 KPE buffer is only 2048
bytes, while the Q64-by-KV16 FP32 scores need 4096 bytes. The exchange therefore
uses two passes of 32 rows. Each pass has a barrier before writing (to protect
KPE or previous score readers) and another before reading. This trades additional
shared-memory communication and synchronization for less repeated computation,
without increasing the shared allocation.

| Layout | QK producers | Score exchange | PV ownership |
|---|---|---|---|
| Q32/KV32 | Groups 0 and 1, two warps each; 16 KV columns per group | One 4 KiB tile, read by all four groups | 128 output columns per group |
| Q64/KV16 | Group 0, four warps | Two 2 KiB passes to group 1 | 256 output columns per group |

Output accumulators remain in registers until output staging/merge; partial LSE
has one writer. PV groups own disjoint output columns, so there is no cross-group
sum of their O fragments. The existing merge across KV splits remains necessary.
See `compute_qk_`, `compute_mla_qk` and the main KV loop in
[`mla.cuh`](../../include/flashinfer/attention/mla.cuh). Q64 reuses KV over more
query/head rows, but needs more output registers. Neither layout is claimed to
be fastest for every shape.

### Kernel Configuration Selection

Let `H` be the query-head count. The planner selects Q32 when either condition holds,
and Q64 otherwise:

```text
H <= 32
sum(kv_lengths) * ceil(H / 64) <= 16384
```

The policy was tuned and evaluated using the 36 benchmark configurations described in
Section 4. All were available during development. It is an RTX 4090 heuristic. The
cooperative grid is sized from the device SM count; split lengths are aligned to 32.

### CUDA Graph Capture and Replanning

The first successful plan in CUDA Graph mode fixes the compiled tile configuration and
split policy. Replanning can update sequence lengths and page indices within reserved
capacity, but cannot replace the kernel configuration captured by the graph.

Changes to captured structural or scalar arguments are rejected before metadata is
updated. A failed replan preserves the previous usable backend. Use a new wrapper to
change the captured configuration. Tests cover both directions of the Q32/Q64 threshold
and failure caused by insufficient workspace.

FA backend plan state owns the frozen schedule and captured-argument signature.
The wrapper passes opaque previous-plan state and invokes the previous backend's
replan validator before backend selection or metadata writes. The generic request
and wrapper contain no SM89-specific fields; a failed validation leaves the old
plan intact, including when a caller attempts to change dtype or backend eligibility.

## 3. Correctness and Memory Safety

The tests below cover numerical results, API behavior and memory safety.
`memcheck` checks invalid accesses, `racecheck` shared-memory races, and
`synccheck` invalid synchronization.

### Numerical and API Tests

Run these tests from a development checkout with dependencies and submodules installed:

```bash
export FLASHINFER_CUDA_ARCH_LIST=8.9
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/attention/test_mla_sm89_fp8.py tests/attention/test_mla_wrapper.py
python -m pytest -q tests/attention/test_mla_page.py::test_append_mla_paged_kv_cache \
  'tests/attention/test_deepseek_mla.py::test_batch_mla_without_kpe[fa2]' \
  'tests/attention/test_deepseek_mla.py::test_batch_mla_page_attention_cuda_graph_replan[fa2]'
```

Tests compare O and LSE against an independent reference; see
[`test_mla_sm89_fp8.py`](../../tests/attention/test_mla_sm89_fp8.py) for the reference
implementation.

| Area | Tested cases |
|---|---|
| Cache addressing | Permuted pages, partially occupied pools, NaN-filled unused pages/tails, non-contiguous storage with explicit strides |
| Inputs and outputs | Empty requests, head counts such as 17, scalar/group scaling, caller-provided output buffers |
| Temporary workspace | Q32/Q64 layouts; capacity exactly sufficient or one byte short; aligned/unaligned starting addresses; unchanged guard bytes outside the workspace |
| Planning and replay | Explicit FA2 and automatic backend selection; Graph replay after replanning; previous plan remains usable after a failed replan |

Small-input tests use `atol=rtol=1e-3` against the BF16-rounded reference output. Larger
random benchmark inputs use the existing FP8 tolerances: output `rtol=0.03, atol=0.004`;
LSE `rtol=0.001, atol=0.003`. These input distributions have different amplitudes; no
original test tolerances were relaxed.

The dominant-token tests complement random inputs: alternating query heads attend
to different keys at the last valid token, tile edges and actual split-KV edges.
Distinct values across output columns expose ownership errors. Removing a selected
key in the reference changes its output by more than 0.5, well beyond the 1e-3
comparison tolerance. Both Q32 and Q64 layouts and both mask modes are tested.
Two causal H192 cases exercise a head group spanning multiple CTA clusters,
where the key bound must round up. Replacing that bound with floor division in an
isolated source copy makes both cases fail numerically; the production source is
unchanged by this negative control.

### Compute Sanitizer Checks

Run the following commands to check the native FP8 implementation:

```bash
for tool in memcheck racecheck synccheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 99 \
    python benchmarks/mla_sm89_fp8/check_native.py --output "$tool.json"
done
```

The driver checks four representative workloads against a CPU FP32 reference. Each
workload also runs three Graph replays with fixed inputs; pytest covers replanning
separately. The native driver avoids cuBLAS initialization and does not suppress checks.
An earlier baseline-only check at upstream `595bc5746cd57efc0006ff888b4bf8144f8235b9`
reported 80 racecheck warnings (zero errors) for B1/H16/KV257/page16 BF16 MLA,
between shared-memory accesses in `smem_t::store_128b` and `mla::write_o`.
This is a recorded upstream observation, not a claim that the warnings are harmless.
To check that path with the benchmark in this patch, run from the candidate root:

```bash
PYTHONPATH=/path/to/upstream compute-sanitizer --tool racecheck --error-exitcode 99 \
  python benchmarks/bench_mla_sm89_fp8.py --baseline-only --no-timing \
  --cases tail257 --output upstream_bf16_racecheck.json
```

A separate GPU-reference check reported `CUDA_ERROR_INVALID_VALUE` in
`cuCtxGetLimit` during cuBLAS initialization. The native sanitizer driver uses a
CPU reference and avoids that initialization; it does not suppress diagnostics.
The local review bundle retains `results/review_revision/upstream_bf16_racecheck.log`,
its source-identifying JSON, and `final_memcheck_gpu_oracle.log`. These diagnostics
are outside the native-path result; no library-wide sanitizer-clean claim is made.

### Cross-Architecture Validation

Shared planner/header changes affect existing FA2 paths. SM89 execution and
foreign-architecture compilation are separate checks. Compile-only checks cover
SM80 FP16/BF16 CKV512 with KPE64 and BF16 KPE0, plus SM90 FP8 KPE64;
they do not establish runtime correctness on those devices.

Runtime results in this report cover RTX 4090 only. The SM89-specific GPU tests
skip other architectures. The compile-only results do not extend the runtime
validation scope, and no recurring CI hardware arrangement is claimed here.

## 4. Performance Evaluation

### Conversion Baseline

The baseline converts and scales both allocated FP8 CKV/KPE pools into BF16
global-memory buffers in one CUDA launch, then calls existing FA2 BF16 MLA.
Buffers are preallocated; conversion repeats on every call. The script reports
the full call as well as conversion-only and BF16-attention-only times.

Only pages assigned to the batch are in the timed pool. The converter is checked
bitwise against Torch conversion; attention uses the independent reference.
Applications retaining a BF16 cache should compare with BF16-attention-only time.
This baseline models repeated conversion of an FP8-only cache.

### Timing Boundaries and Benchmark Code

All benchmark-specific code below is included in this change. The timing helper is
existing FlashInfer code.

| Source | What to review |
|---|---|
| [`bench_mla_sm89_fp8.py`](../../benchmarks/bench_mla_sm89_fp8.py) | `baseline()` calls `convert()` then `attention()`; `native()` calls the FP8 wrapper; `measure()` sets the timing protocol |
| [`fused_dequant.cu`](../../benchmarks/mla_sm89_fp8/fused_dequant.cu) | One CUDA launch converts both allocated pools into BF16 global-memory buffers |
| [`testing/utils.py`](../../flashinfer/testing/utils.py) | `bench_gpu_time_with_cuda_event` and `bench_gpu_time_with_cuda_graph` implement event placement and sample normalization |

The measured call boundaries are:

```text
baseline(): [ FP8→BF16 pool conversion → existing BF16 wrapper.run() ]
native():   [ FP8 wrapper.run(), including tile conversion and output merge ]
speedup = baseline call latency / native call latency
```

The benchmark defines this baseline explicitly; upstream does not select it as an
automatic fallback. The reported latency measures the complete call directly, rather
than adding separately measured conversion and attention latencies.

### Workloads and Measurement Protocol

The benchmark script, fused converter, and fixed workload configurations are included in
the source tree:

```bash
python benchmarks/bench_mla_sm89_fp8.py --group all --output results_run1.json
python benchmarks/bench_mla_sm89_fp8.py --group all --output results_run2.json
# Short smoke run:
python benchmarks/bench_mla_sm89_fp8.py --cases tail257,h64_8k \
  --iters 5 --rounds 1 --output smoke.json
```

The benchmark runs 36 fixed configurations defined in
[`cases.json`](../../benchmarks/mla_sm89_fp8/cases.json), covering:

- Batch sizes from 1 to 64 and KV lengths from 0 to 65536, including mixed-length batches.
- Query-head counts of 8, 16, 17, 32, 64, 128 and 192.
- Page sizes of 1, 4, 16, 32, 64 and 128 tokens, including partial pages and empty requests.

These are selected combinations, all available during tuning. Performance
configurations and correctness test cases are counted separately.

Inputs use deterministic seeds. Each run alternates the native/baseline order
over three rounds of 15 samples. `measure()` sets five warmup calls or Graph
replays, in addition to the helper's setup warmup. The commands above collect two
runs: take each path's median of 45 samples per run, then the median of those two
medians. Divide the baseline aggregate by the native aggregate for each
configuration's speedup. JSON records all samples and the environment.

Both execution modes use CUDA Events to measure elapsed GPU time. They differ in
submission and cache preparation:

| Submission mode | Cache preparation | Event interval and normalization |
|---|---|---|
| Graph replay | Reuses addresses; no explicit L2 flush | Events bracket one replay of five calls; divide elapsed time by five |
| Ordinary launches | Flushes L2 before each sample's start event | Events bracket one call; launch gaps may fall inside the interval |

"Warm" describes address reuse without flushing, not a guarantee that all data hits L2.
JIT, planning, allocation, reference checks, Graph construction and L2 flush preparation
are excluded. These measurements cover the operator, not end-to-end serving latency.

`fallback_bf16_temporary_bytes` is the avoided BF16 pool allocation:

```text
2 * allocated_tokens * (512 + 64) bytes
```

It includes page tails. FP8 storage and native split-reduction workspace remain
allocated, so this value is not the application's total peak-memory reduction.

### Reproducing the Upstream Rejection

Select an unmodified upstream checkout with `PYTHONPATH` and use absolute paths:

```bash
PYTHONPATH=/path/to/upstream python /path/to/candidate/benchmarks/bench_mla_sm89_fp8.py \
  --baseline-only --check-unsupported --cases tail257,h64_8k \
  --label upstream --output upstream.json
```

The JSON output records the loaded source path, commit/diff hashes, benchmark/case
hashes, GPU, driver and Torch/CUDA versions. Check these fields before comparing an
installed package with a source checkout.

### Running the Upstream Benchmark

```bash
python benchmarks/flashinfer_benchmark.py --routine BatchMLAPagedAttentionWrapper \
  --backends fa2 --batch_size 1 --s_qo 1 --s_kv 8192 --num_qo_heads 64 \
  --num_kv_heads 1 --head_dim_ckv 512 --head_dim_kpe 64 --page_size 32 \
  --q_dtype bfloat16 --kv_dtype fp8_e4m3 --refcheck --use_cuda_events
```

This command checks the upstream benchmark entry point with page size 32. The fixed
`h64_8k` configuration uses page size 16, so these are different measurements.

## 5. Nsight Compute Profiling

Use the named NVTX ranges to select native or baseline execution:

```bash
ncu --set full --nvtx --nvtx-include 'native/' \
  python benchmarks/bench_mla_sm89_fp8.py --profile --cases h64_8k --output profile.json
```

`--profile` skips reference checks and timing samples, and marks correctness as
unchecked. Run correctness validation separately. Inspect register count, local
load/store traffic, occupancy, shared-memory use, Tensor Core activity and DRAM/L2
traffic. Low Tensor activity alone does not quantify a possible speedup. Profiled kernel
duration and uninstrumented public-call latency must be interpreted separately.

## 6. Environment and Validation Boundaries

The recorded environment is Python 3.12.4, Torch 2.10.0+cu128, CUDA 12.8.93 and
driver 535.216.01. For these measurements, native SM89 cubins were built with
`FLASHINFER_EXTRA_CUDAFLAGS=--compress-mode=none`; this is a local build setting.

Runtime validation covers RTX 4090; SM80/SM90 checks were compile-only. The
layout threshold is empirical.
The PR description and accompanying results report contain measured latencies,
test counts and NCU observations. They do not establish model throughput or a
hardware performance ceiling.
