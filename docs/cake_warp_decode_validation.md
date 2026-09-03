# Cake warp-decode validation

`benchmarks/cake_warp_decode.py` is the standalone SM103 validation and
benchmark entry point. It prepares one public TRTLLM NVFP4 physical fixture per
geometry and shares those exact activation, routing, weight, and scale tensors
between Cake and `trtllm_fp4_block_scale_routed_moe`.

## Correctness gate

Run from an editable FlashInfer checkout on an SM103 GPU. The editable install
stages the repository CUDA/include trees under ``flashinfer/data``; setting
``PYTHONPATH`` alone is not sufficient for JIT builds:

```bash
git submodule update --init --depth 1 \
  3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog
FLASHINFER_BUILD_NO_PIP=1 BUILD_NIXL_EP=0 BUILD_NCCL_EP=0 \
  python -m pip install --no-build-isolation --no-deps -e .
```

Then run the dense gate:

```bash
python benchmarks/cake_warp_decode.py \
  --mode correctness \
  --geometry all \
  --json-out artifacts/cake_warp_decode_correctness.json
```

The mandatory dense matrix is:

| Hidden | Intermediate | Experts | Top-k | Tokens |
|---:|---:|---:|---:|---:|
| 2048 | 512 | 512 | 10 | every value from 1 through 32 |
| 2048 | 1536 | 60 | 4 | every value from 1 through 32 |

Every row prepares the exported workspace on non-default stream A, launches on
distinct non-default stream B, compares finalized BF16 output against the
official routed-MoE baseline with `atol=1e-2` and `rtol=1e-2`, and launches twice
into the same caller-owned output and workspace. Pointer and byte-capacity
stability are part of the gate. Workspace preparation returns a generation
receipt only after its asynchronous initialization is complete; launch validates
that receipt on stream B. The receipt is explicitly released after the last
launch, with a weak-reference finalizer retained for exceptional exits.

The public runner validates during input packing that every expert ID satisfies
`0 <= id < num_experts`. It records the tensor identity, storage address,
in-place version, and expert count, so repeated calls with the same unchanged
routing tensor do not add a device-to-host synchronization. PyTorch inference
tensors do not expose a version counter; their receipt is identity/storage based,
so post-validation mutation carries the same caller-side range obligation as a
graph replay. CUDA Graph capture accepts only a tensor covered by a pre-capture
validation receipt. Graph replay may mutate routing tensors in place without
re-entering Python, so every replay must still preserve the documented
value-range contract.

The runner retains at most 64 live routing-validation receipts. It fails
explicitly on the next distinct tensor instead of silently invalidating an older
graph-warmed tensor; construct a new runner to start another bounded lifetime.

The runner caches a distinct prepared workspace per execution stream and keeps
a strong reference to the stream together with the workspace. The cache is
bounded; callers that create more than 64 distinct stream/geometry entries must
construct another runner. FlashInfer's ordered warmup-to-capture transition may
reuse the already prepared packed workspace when PyTorch enters an internal
capture stream. Completion-event dependencies serialize ordinary low-level FFI
submissions that share a receipt, while the public runner isolates eager
multi-stream calls automatically. Direct callers must not replay multiple graph
executables concurrently on one receipt.

The binding records a CUDA completion event after every submitted launch. It
keeps one stable event handle per device/workspace address for process lifetime,
bounded at 4096 addresses, because captured external-event nodes may outlive a
released receipt. If accepted GPU work cannot be covered by a completion event,
the binding quarantines that workspace generation and refuses to release or
re-prepare it rather than risk an unsafe overwrite. The Python receipt lease is
anchored to the runner, not the tensor; if strict retirement fails, it retains a
strong reference to the workspace in a process-lifetime quarantine so PyTorch's
allocator cannot recycle the storage.
Explicit re-preparation and receipt release wait on those events before
overwriting or freeing workspace state, without retaining a borrowed raw stream
handle. Receipts are positive, generation-specific, and single-use: releasing
an unknown, stale, or already released receipt is an error rather than an
idempotent success, so a bookkeeping mistake cannot masquerade as proven GPU
retirement. One retirement case per geometry launches on stream B, immediately
re-prepares the same workspace on stream C without synchronizing B, launches the
replacement generation, and releases it without synchronizing C. The
replacement output must match the official baseline.

A captured graph retains workspace pointers and completion-event nodes. Keep the
runner, workspace, and receipt alive until every possible replay is finished;
do not release or re-prepare that workspace and then replay the old graph.

One deterministic receipt-generation case per geometry creates a distinct
tensor view at the exact same workspace address, prepares it again to obtain
receipt r2, verifies that stale receipt r1 is rejected before launch, then checks
that r2 launches correctly on stream B. After releasing r2 through the fourth
FFI entry point, the case verifies that r2 is rejected as well. This models the
same-address generation change caused by caching-allocator address reuse without
depending on a nondeterministic allocator choice.

CUDA Graph capture and post-capture mutation are additionally exercised at all
selector transitions:

- E512: T=1, 2, 22, 23, 32.
- E60: T=1, 7, 8, 10, 11, 12, 16, 17, 32.

Workspace preparation occurs on stream A before capture on distinct stream B.
The harness first replays the graph without mutation and checks the original
baseline. Before the next replay it changes the graph-stable expert IDs, BF16
routing weights, and the packed GEMM1/GEMM2 weights and scales of a routed
expert, then replays from distinct stream C. The mutated replay must match a
fresh baseline evaluation, the baseline output must differ from its pre-mutation
value, and caller output/workspace pointers and capacity must remain stable
across both replays. Receipt release occurs immediately after replay submission,
without a caller-side stream synchronization, and must safely wait for the
completion event embedded in the captured graph. While that graph executable is
still alive, the harness immediately re-prepares and releases the exact same
workspace address before synchronizing the replay stream; this makes premature
retirement observable as a destructive workspace race. A 50,000,000-cycle GPU
delay before replay makes the work deterministically in flight when release is
called, and the replacement generation performs a full launch and correctness
comparison rather than prepare-only reuse.

One public `MoELayer` case per geometry additionally clears the standalone
tuner cache, enters actual autotune mode with the exact token bucket, and
requires a positive successful-profile count from the real single-backend
profiling and winner-selection path (including its internal CUDA Graph timing),
captures a
subsequent layer call, and replays it from another stream. This is the
framework-level regression gate for runner workspace selection and reusable
routing-validation receipts.

The selector boundary labels in the JSON receipt distinguish E60 `_e64_scan1`
at T=11, `_e64_scan2` at T=12..16, and the general route packer at T=17..32.
They distinguish E512 direct routing through T=22 from the general route packer
at T=23..32.

## CUPTI benchmark gate

The default benchmark matrix is the selector-boundary set above:

```bash
python benchmarks/cake_warp_decode.py \
  --mode benchmark \
  --geometry all \
  --warmup 5 \
  --repetitions 30 \
  --paired-rounds 2 \
  --json-out artifacts/cake_warp_decode_benchmark.json
```

The script calls the repository `bench_gpu_time` entry point with
`enable_cupti=True`, `cold_l2_cache=True`, and `use_cuda_graph=False`. It refuses
to benchmark without `cupti-python>=13`, so the timing path cannot silently fall
back to another timer. Quantization, workspace allocation/preparation, first-use
compilation, and warm-up occur outside the measured samples.

For every benchmark shape, exported workspace preparation runs on non-default
stream A while parity checks and all CUPTI sessions run on distinct non-default
stream B. The exported output and workspace must retain their caller-owned
address/capacity through the complete ABBA/BAAB sequence, after which the
generation receipt is explicitly released.

Each result row alternates exported Cake and the official FlashInfer routed-MoE
baseline in ABBA and BAAB rounds. Every position is a separate repository
`bench_gpu_time` CUPTI/cold-L2 session. The receipt records the exact order and
measurement for every position, each arm's aggregate median,
`exported_over_flashinfer_baseline`, and the worst per-round ratio.
`--paired-rounds` must be even and at least two so both orderings have equal
representation.

Use `--benchmark-tokens 1 11 17 24 32` to request a smaller explicit
performance slice; correctness mode always retains the full two-geometry,
T=1..32 matrix.

## Compute Sanitizer entry point

The launch-only mode keeps the kernel invocation separate from the official
baseline and accepts a one-case slice for focused sanitizer runs:

```bash
compute-sanitizer --tool synccheck --target-processes all \
  python benchmarks/cake_warp_decode.py \
  --mode sanitizer \
  --geometry e60_i1536_k4 \
  --sanitizer-tokens 11

compute-sanitizer --tool memcheck --target-processes all \
  python benchmarks/cake_warp_decode.py \
  --mode sanitizer \
  --geometry e512_i512_k10 \
  --sanitizer-tokens 24
```

Omitting `--sanitizer-tokens` launches every selector boundary for the selected
geometry. Workspace preparation remains outside the launch-only call on
non-default stream A, the call runs on distinct non-default stream B, and the
receipt is released only after stream B has completed.
