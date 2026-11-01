## Experimental HPC-Ops parallel scheduling port (SM100)

Owner: slhslh. Tracking: https://github.com/flashinfer-ai/flashinfer/issues/6468.
Explicit opt-in, JIT-only; not enabled in automatic dispatch.
FlashInfer base: `388e7a311e9f79941134dcc298cbc942b5c607a1`.
HPC-Ops source: `1cd332980ed46bd0172091c1c35d55338fcae47a`.
Incubation review target: 2026-11-11. Graduation requires wider shapes, graph-safe
planning, Blackwell Tensor Core backend comparisons and maintainer approval.

### What is actually ported

`include/flashinfer/attention/hpc_original_assign.cuh` vendors the original
`assign_attention_decode_task_kernel` from HPC-Ops, not a serial imitation.
The original per-CTA length loading/reduction, fast-forward to its own bin,
head-major assignment, 48-byte records, chunk counts, sentinel slots, and
cross-CTA completion protocol are retained. Only namespace/vector load-store
helpers and deterministic zero padding are adapted. Original MIT notices are
in the adjacent `HPC_OPS_LICENSE.txt`.

SM100 single-Q settings match HPC-Ops: **128-token scheduling tiles,
default one persistent CTA per SM, 128 threads per planner CTA**.
Storage pages remain 64 tokens: scheduling tile size and KV page size differ.
An explicit `cta_per_sm` override is a tuning experiment, not the SM100 default.
Pools exceeding the planner's concurrent residency are rejected because the
original kernel contains inter-CTA waits.

The execution chain is:

1. GPU prepares KV lengths from page metadata and clears synchronization flags
   at the current map offsets. This matters when lengths change and the original
   variable-layout workspace moves its flag region.
2. Original parallel kernel generates the original HPC-Ops map.
3. A parallel block scan converts original chunk/task counts into compact
   FI state/CTA offsets. Per-CTA parallel adapters translate actual records to
   `[batch, head, begin, end, state]`; they do not regenerate a second schedule.
4. Each persistent compute CTA consumes its own adapted records using the
   existing FlashInfer CUDA-core decode device routine.
5. BF16 partial outputs and FP32 LSE use FlashInfer variable-length merge.
   Partial-state order is batch / KV head / chunk / GQA head.

Thus the original parallel **scheduler** is ported; HPC-Ops's complete attention
math kernel, Tensor Core pipeline and adaptive combine are **not** substituted.
The FI compute block remains `(16,4,2)` (128 threads). This isolates scheduling
effects within the same CUDA-core math implementation, not whole-library speed.

### Limits and safety

- Tested on B200, compiled for `sm_100a`; wrapper accepts SM100 only.
- BF16 Q/K/V/O, contiguous NHD, page64, head128, GQA4, one Q token per request.
- 1–2048 requests, positive KV lengths. Device page IDs and tail lengths are
  trusted, as in FI execution: valid page IDs and `1 <= tail <= 64` required.
- No CUDA Graph, PDL, RoPE, custom mask, sliding window, multi-token prediction,
  automatic selection or Blackwell Tensor Core backend.
- A wrapper owns mutable workspaces; do not use it concurrently on streams.
  The original global-wait planner assumes available concurrent GPU residency.
  Validate deployment under other concurrent GPU workloads separately.
- Plan allocates/grows workspace as needed. Benchmark warmup excludes growth/JIT;
  per-step CPU metadata validation, H2D indptr copy and all GPU adapters are
  included in full plan/step wall time.
- No runtime work stealing: bins are regenerated for each plan.
- `schedule()` reads adapted GPU tasks; `original_schedule()` reads original
  map storage. These diagnostic methods synchronize and are outside timing.
  Only active records/counts/flags are meaningful; unused padding is not data.

### Run

```bash
# From the repository root, with an editable FlashInfer installation:
python -m pytest tests/experimental/test_hpc_flat_bin_decode.py tests/experimental/test_hpc_flat_bin_planner.py -q
python examples/experimental/compare_dynamic_decode.py --ported-hpc --cases small,uniform,skewed,short_uniform,medium_uniform,mixed --iters 30 --warmup 5 --timing cupti --json-out scheduler_parallel_sm100.json
```

The planner test compiles a vendored, unmodified HPC-Ops CPU reference with
`g++` and compares all 9 meaningful fields of original GPU records, as well
as the adapted execution map. Other tests cover output/reference comparisons,
page boundaries, changing batch/head/length/workspace layouts and guard failures.

The benchmark compares explicit FI `backend="fa2", use_tensor_cores=False`
against the parallel port, sharing Q/K/V and shuffled physical page IDs.
It does **not** compare against the fastest FI SM100 backend or claim general
HPC-Ops/FlashInfer superiority. Native planner is CPU-based on this path.

GPU measurements use CUPTI (CUDA event fallback) with cold L2; multi-kernel
timings span first start to last end, including launch gaps, not summed durations.
Plan and step wall timings synchronize and do not flush L2. Therefore GPU
subtimes need not sum to independently measured plan or end-to-end wall times.
JSON includes original assign-only, input prepare, adaptation, full GPU assign,
full plan wall, attention+merge GPU and complete plan+run wall measurements.

The shipped default is a reference setting, **not** an automatic performance
recommendation: skewed long/short batches can benefit; uniform batches and a
single long request can regress with one persistent CUDA-core CTA per SM.
