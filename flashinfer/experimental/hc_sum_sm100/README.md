# SM100/SM103 HC with carried square sums

Experimental, inference-only low-rank HyperConnection. The public opt-in is
`flashinfer.hyperconnection.prepare_hc_sum_weights` or
`bind_hc_sum_weights`. No SGLang installation or model object is required.
This implementation does not change DeepSeek mHC or the SM12x backend.

Wheels built from this revision retain the HC CUDA JIT sources even with
`FLASHINFER_WHEEL_EXCLUDE_EXPERIMENTAL_SOURCES=1`. HC remains JIT-only and
requires CUDA build tools; a source checkout is also supported.
Requirements include PyTorch's CUDA
`bmm(out_dtype=torch.float32)` API (available in PyTorch 2.8), Triton with Gluon,
and CuTe-DSL >=4.8. The validated environment is PyTorch2.14.1, Triton3.8.0,
CuTe-DSL4.8.0 and CUDA13.3 on GB300. MIG/green-context partitions and SM100
hardware have not been validated; the small path needs non-portable16-CTA
cluster launch support.

## Contract

- Four branches, CUDA BF16/FP16, SM100 or SM103.
- `H` is a positive multiple of 512, at most 16384. `L` is a positive multiple
  of 64 with `L + 4 <= round_up(L, 128)` (for example 64, 192, 320).
- Residual: contiguous `[T, 4*H]`; core input/output: `[T,H]`; final injection
  alpha and square sums: FP32 `[T,4]`.
- Value buffers and native weights have 32-byte-aligned base addresses. The
  small-T mix also needs 32-byte-aligned current sums; large-T mix accepts 16.
  The tensor-only `hc_apply_sum` accepts 16-byte-aligned alpha/sums for row
  partitions, and does not require retaining the whole mix workspace. An odd-row
  slice is not automatically a supported small-T mix input.
- Bootstrap sums observe stored residuals. Apply sums observe the FP32 update
  **before** the residual BF16/FP16 store. They are not the exact sum of squares
  of the stored residual. Consumers reconstruct inverse RMS using their epsilon.
- Alpha is already `2 * sigmoid(...)`. Apply must not sigmoid it again.
- Both paths share a single folded Down+Inject storage `[4,Npad,H]`, where
  `Npad=round_up(L+4,8)`. RMS weights are interleaved `[H,4]`. Up keeps its
  ordinary weight plus a derived interleaved/padded layout for the small path.
- This is a new state/numerical contract, not a drop-in replacement for an API
  returning a full normalized residual. Small/large GEMMs can differ in rounding.
  Folding RMS gamma into BF16/FP16 weights also changes rounding relative to
  the unfused checkpoint program. The staged reference models this explicitly;
  an arbitrary weight distribution has no universal unfused-reference tolerance.
  Model-level accuracy must be validated by the integrating framework.

## Use

```python
from flashinfer.hyperconnection import prepare_hc_sum_weights

# Checkpoint-layout weights; gamma = 1 + norm_weight.
plan = prepare_hc_sum_weights(
    down_cpu, up_cpu, inject_cpu, norm_cpu, rms_eps=1e-6, device=residual.device
)
workspace = plan.allocate_workspace(residual.shape[0], deterministic=False)
sums = plan.bootstrap(residual)
mixed, alpha = plan.mix(residual, sums, workspace)
block_output = core(mixed)  # attention, MLP, etc., owned by the caller
next_residual, next_sums = plan.apply(block_output, residual, workspace)
```

Use `bind_hc_sum_weights` after a native checkpoint load to avoid another fold
or another permanent Down+Inject copy. Its padding must already be zero; it
does not inspect tensor contents or synchronize the GPU to verify this.
Raw checkpoint Down/Inject/RMS tensors passed to preparation are not retained.
An Up tensor already on the target GPU shares storage with the caller and must
stay valid and unchanged while the plan is used; CPU Up is uploaded once.
The caller owns releasing obsolete raw
weights and its own checkpoint/parameter naming.
CPU preparation folds and packs before uploading, so it never creates raw
Down/Inject tensors on the GPU. It also accepts Up already on the target GPU.
For a streaming native loader, bind the final canonical storage. The CUDA-input
convenience route is for callers that already own those inputs; the single-storage
production path should use CPU preparation or native binding.

## Workspace and Graph lifetime

Prepare weights and allocate one workspace per live invocation outside capture.
Warm each shape/dtype/mode once before capturing. To capture bootstrap, pass a
preallocated `out_sum`. Mix and Apply use supplied storage without tensor
allocation. The small-path Down scratch padding is initialized once and retained.
In this FlashInfer revision, a source-built NVCC module checks freshness through
Ninja on its first load in a process. Setting `FLASHINFER_DISABLE_JIT=1` before
that load rejects it even if its artifact exists; prepare/warm before freezing
JIT. Full-graph `torch.compile` and autograd are not supported by this version;
the tested execution modes are eager and explicit CUDA Graph replay.

Mix consumes CURRENT sums and clears NEXT sums in its fused Up/Mix kernel.
Every atomic Apply needs its corresponding preceding Mix, including every
Graph replay. Never alias CURRENT with NEXT. Deterministic Apply overwrites ten
partials per branch and reduces them in a fixed order; its workspace is explicit.
Large-T deterministic Down uses a fixed-order BMM. The existing core batch-invariant
GEMM currently accepts 2-D BF16 operands and BF16 output; HC needs a strided
3-D BF16/FP16 BMM with FP32 output. The preserved local BMM supplies that
contract without extra input copies or changes to stable core APIs. The supported determinism
claim is repeatability at fixed T, not bitwise invariance across batch sizes.

Two concurrent calls need disjoint workspaces. Live output tensors cannot be
overwritten by another invocation. Alternate workspaces for a recurrent chain,
or copy the required state at a deliberate framework boundary. Device/dtype or
weight changes, including in-place content updates, require rebinding and graph
recapture before reusing the operator. The small-T Up layout is derived at
preparation; changing the original tensor does not refresh that layout. There is no hidden
global output-buffer cache or framework weight-update hook.

## Preserved execution policy

For `T <= 24`, Down+Inject uses cluster split-K16 and emits lowrank SiLU plus
final FP32 alpha. Up fuses gate, normalized residual Mix and next-sum reset.
For `T > 24`, a branch BMM produces FP32 partials without copying the residual,
then a Gluon kernel merges/normalizes them. Up uses ordinary GEMM, followed by
fused Mix/reset. Both use the same two-CTA-per-token/branch, 160-thread Apply.
Deterministic mode adds a fixed-order statistics reduction.

Correctness, Graph and matched NCU results must accompany any performance claim;
this description of retained optimizations is not a claim that all shapes are
faster. The standalone example is `examples/hc_sum_sm100.py`; tests are in
`tests/experimental/test_hc_sum_sm100.py`.

## Scope and upstream admission

Owner, tracking issue and graduation target remain to be assigned.
Before graduation: intended-hardware correctness/Graph coverage, matched NCU
comparison, API/trace and packaging review, and a maintainer-approved stable
contract. Backend code is JIT-only and is not registered in AOT.
Source attributions and licenses are recorded in [SOURCES.md](SOURCES.md).

### Allocator-backed invocation runtime

`HCSumRuntime(plan, deterministic=False)` is an optional framework-neutral
execution adapter. Construct it before CUDA Graph capture. It retains only the
initialized small-T Down scratch; each `mix(residual, prepared=None)` allocates
invocation outputs from the current PyTorch allocator (including a graph memory
pool during capture) and returns `(mixed, context)`. Compute the core output,
then call `context.apply(core)` to obtain an `HCSumState(residual, sum_sq)`.
Keep the runtime and its bound weights alive as long as any captured graph
references them: graphs do not retain the runtime's preallocated scratch.
Pass that state to the next compatible runtime's `mix` to preserve the FP32
pre-storage-cast statistics. These calls use the same kernels and dispatch as
an explicitly allocated `HCSumPlan` workspace.
Missing or non-matching prepared state makes `mix` bootstrap from the stored
residual. This intentionally differs from Apply, which rejects a mismatched
residual after alpha has already been computed for particular rows.

A runtime is one ordered execution lane: do not use it concurrently on multiple
streams or overlapping graph replays. Independent lanes need separate runtimes
or explicit workspaces. Invocation outputs remain distinct even when multiple
outputs escape the layer stack; there is no per-token-count output cache.

`context.slice_rows(size, rank)` partitions all Apply inputs and destinations
with `torch.tensor_split` semantics (the first `T % size` ranks get one extra
row) and keeps their workspace alive. Independently partitioned core outputs
should pass the corresponding residual to `apply` for identity validation.
Odd-row partitions of sum tensors are realigned
only if the next small-T mix needs it. `with_residual` rejects a mismatched tensor.
Pointer/shape checks cannot detect in-place communication: the framework must
invalidate prepared state whenever communication, row reordering, or another
operation changes residual contents. This adapter deliberately makes no tensor
parallel, expert parallel, or model-specific communication decisions.

For native model parameters, `hc_sum_weight_shapes(H, L)` reports the storage
shapes and `pack_hc_sum_weights_into` folds CPU checkpoint weights directly into
caller-owned Down+Inject/Norm storage. There is no temporary native or raw GPU
Down/Inject allocation. Up retains its checkpoint layout. Bind only after all
weights have loaded; native checkpoint/IPC weights must not be folded again.
`hc_sum_support_reason` is an allocation-free construction-time eligibility
query. Unsupported devices/configurations can be dispatched by the caller before
creating native parameters; selected-runtime contract violations raise.
