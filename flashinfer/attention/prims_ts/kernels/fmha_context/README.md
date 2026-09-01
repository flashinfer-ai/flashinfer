# Task-Scheduled FMHA Context

This directory contains the CuTe DSL task-scheduled (TS) FMHA context/prefill
kernel used by FlashInfer's experimental Blackwell APIs. One implementation
serves fixed contiguous, packed-ragged contiguous, and packed-query paged-KV
attention with MHA or GQA.

The public API exposes attention semantics, not scheduling controls. Contiguous
and paged plans select a nonpersistent, static-persistent, or CLC-persistent
launch from logical work, task topology, live-metadata requirements, causal
domain structure, and GPU capacity. Paired, live-ragged, and zero-offset
triangular contiguous domains use CLC. Immutable single-instance
bottom-right-offset domains launch directly within one resident wave and use
static persistence above one wave. Single-instance uniform causal paged plans
use static persistence: zero-offset triangular domains run a heavy-first
raster, while bottom-right-offset or windowed domains keep sequence-local
order. A positive causal left window selects an internal head-paired GQA
mapping; other cases use the query-paired mapping.

## Public APIs

Import these entry points from `flashinfer.attention.prims_ts`:

| API | Use |
| --- | --- |
| `BatchPrefillTSWrapper` | Reusable fixed or packed-ragged contiguous Q/K/V plan. |
| `batch_prefill` | One-shot fixed or packed-ragged contiguous attention. |
| `BatchPrefillPagedTSWrapper` | Reusable packed-Q, paged-K/V plan. |
| `batch_prefill_with_paged_kv_cache` | One-shot packed-Q, paged-K/V attention. |

Both wrappers are experimental and may change incompatibly while the PrimTS
API family is stabilized.

These experimental context entry points are not currently registered with
`fi_trace`; tracing support is limited to the PrimTS decode APIs.

`BatchPrefillTSWrapper` keeps the existing tensor-driven lifecycle: planning
validates Q/K/V and reads cumulative metadata when needed. Packed contiguous
plans retain `qo_indptr` and `kv_indptr` as live inputs; general ragged kernels
reload their values on every run, while a uniform packed plan may compile its
fixed offsets into the specialization.

`BatchPrefillPagedTSWrapper` uses a static-spec lifecycle. `plan()` receives
only device, capacity, head, dtype, page, mask, and scale information and may
compile; it does not bind Q, cache, or request metadata and allocates no
workspace. Every `run()` supplies Q, K/V cache, Q and K/V cumulative offsets,
the dense K/V page table, and live K/V lengths. Runtime structural validation
is enabled by default. `validate=False` skips those checks for a previously
validated steady-state or CUDA Graph launch; the caller then owns every value,
bounds, aliasing, and lifetime precondition. Neither mode copies metadata
values to the host.

## Supported contract

| Feature | Support |
| --- | --- |
| GPU | SM100a/B200 (qualified); SM103a/B300 (architecture-gated, not yet signoff-qualified) |
| Head dimension | 128 or 256 |
| Head mapping | MHA/GQA; `Hq` must be divisible by `Hkv` |
| Q/K/V dtype | Matching `torch.float16`, `torch.bfloat16`, or `torch.float8_e4m3fn` |
| Output dtype | `torch.float16`, `torch.bfloat16`, or `torch.float8_e4m3fn` |
| Contiguous storage | Fixed BSHD or packed-ragged THD |
| Paged storage | Packed Q plus separate compact HND K/V page pools |
| Page size | 16, 32, 64, or 128 tokens |
| Mask | Dense or bottom-right causal |
| Sliding window | Positive causal left window; `window_left=-1` disables it |
| Scheduling | Automatic nonpersistent, static-persistent, or CLC-persistent selection; no public tuning knob |
| Accumulation | FP32 QK/PV and softmax state |

Current accuracy and performance signoff is on SM100a/B200. SM103a/B300 is
admitted by the runtime architecture guard but remains to be qualified.

A positive left window requires GQA with an even `Hq/Hkv` ratio greater than
one. Causal attention requires `Sq <= Sk` for every request, both when the plan
is created and after any live cumulative-offset update. All tensor extents and
packed request lengths must be positive. Total logical Q and K extents—
`B*Sq`/`B*Sk` for fixed storage and `total_q`/`total_k` for packed storage—must
be at most `2**31 - 256`; this coordinate-representation limit reserves 255
values for the padded tail of the largest supported 256-row query work tile.

Q, K, V, and `out` must be compact, 16-byte-aligned CUDA tensors on one
device. Metadata must be compact CUDA `torch.int32` on that device and at
least 4-byte aligned. A caller-provided `out` must not overlap Q, K, V, or any
metadata supplied to or retained by the wrapper. The launch conservatively
rejects overlapping storage spans. The API returns O only; rowwise LSE and
other softmax state remain internal to the kernel.

## Tensor and metadata layouts

Contiguous inputs:

- Fixed Q/O: `[B, Sq, Hq, D]`; K/V: `[B, Sk, Hkv, D]`.
- Packed Q/O: `[total_q, Hq, D]`; K/V:
  `[total_kv, Hkv, D]`.
- Packed metadata: compact CUDA `int32[B + 1]` `qo_indptr` and `kv_indptr`.
  Both start at zero, increase strictly, and end at the corresponding packed
  tensor extent.

Paged inputs:

- Q/O: `[total_q, Hq, D]`.
- Separate K and V pools: `[num_pages, Hkv, page_size, D]`.
- Reusable-wrapper metadata: `qo_indptr[B + 1]`, token-based
  `logical_kv_indptr[B + 1]`, `seq_lens_kv[B]`, and a compact dense page table
  `dense_page_idx_kv[B, 2, max_num_pages_per_seq_kv]`, all CUDA `int32`.
- The two dense page-table planes address the separate K and V pools. When
  those pools use the same physical page numbering, the planes contain the
  same IDs. The column capacity is static, covers `max_seq_len_k`, and is a
  multiple of `128 / page_size`; padded entries must still name valid pages.
- The one-shot API continues to accept FlashInfer CSR metadata:
  `qo_indptr[B + 1]`, `paged_kv_indptr[B + 1]`,
  `paged_kv_indices[num_used_pages]`, and `paged_kv_last_page_len[B]`.
- Physical page IDs may be arbitrary, repeated, and nonidentity ordered.

Every cumulative-offset vector starts at zero and increases strictly.
`qo_indptr[-1]` equals `total_q`; wrapper `logical_kv_indptr` deltas equal
`seq_lens_kv`; every live length is positive and no greater than
`max_seq_len_k`; and every page ID touched by a live or padded tile indexes the
physical cache. For the one-shot CSR API, `paged_kv_indptr[-1]` equals the
number of page-index entries and each last-page length is in `[1, page_size]`.

For request `b`, bottom-right causal row `i` can see through
`Sk[b] - Sq[b] + i`. With `window_left=W>0`, the row retains that key and at
most `W` preceding keys. `sm_scale` defaults to `1 / sqrt(D)` and
`output_scale` defaults to 1; supplied scales must be finite, positive, and
representable as positive `float32` values.

For packed contiguous attention, the host reads cumulative metadata once
during planning to establish the static geometry and maximum Q/K capacities.
The plan keeps `qo_indptr` and `kv_indptr` as live device inputs; their storage
must remain valid and stable. Their values may change between runs while
preserving the planned batch, zero starting offsets, final packed extents,
strictly positive deltas, and these per-request capacity bounds. Each capacity
is the corresponding global plan maximum,
`max_b(Sq_plan[b])` or `max_b(Sk_plan[b])`, and applies independently to every
runtime request:

```text
0 < Sq[b] <= planned max_seq_len_q
0 < Sk[b] <= planned max_seq_len_k
```

Every causal replay must additionally satisfy `Sq[b] <= Sk[b]`. The
request-local bottom-right offset `Sk[b] - Sq[b]` may change; it is derived
from the live offsets. Fixed totals plus the per-request capacity bounds force
plan-time uniform Q or K lengths to remain unchanged. In particular, when a
dense plan compiles away request-local K-tail masking because every K length
equals the same 128-row-aligned maximum, the replay conditions preserve that
specialization.

Paged wrapper planning fixes only static capacities and compile-time semantics.
Each run may provide different valid Q offsets, K/V offsets and lengths, and
physical page IDs without another plan. The batch remains exact; Q and K/V
deltas stay positive and within `max_seq_len_q` and `max_seq_len_k`; the final
Q offset matches the packed Q/O extent; and the dense page-table shape remains
the planned shape. For causal attention, every live `Sq[b]` is no greater than
`Sk[b]`. The kernel derives the request-local causal offset from the live
metadata.

With the default `validate=True`, `run()` checks tensor structure, shapes,
dtypes, devices, scales, output, and aliasing, but deliberately does not read
metadata values back to the host. `validate=False` skips those structural
checks. In either mode, invalid offsets, lengths, or page IDs can produce
incorrect results or out-of-bounds access. CUDA Graph replay also requires
stable tensor shapes and addresses even though values may change between
completed replays.

## Dataflow and source map

```text
Q + contiguous or paged K/V
    -> staged Q and streamed K/V
    -> QK MMA -> masked online softmax -> P + row statistics
    -> PV MMA -> online-softmax correction
    -> staged O -> output
```

The TS graph assigns load, MMA, softmax, correction, epilogue, page-offset,
and scheduling work to cooperating tasks. Resources own the corresponding
SMEM/TMEM buffers and pipeline state.

Paged D256 uses topology-derived page-ID staging. For a dense static domain
that is divisible by the complete staged window and whose exact SMEM footprint
fits the K/V cadence, each of the 32 producer lanes loads one page ID for each
of the two head-dimension stages, so one handoff covers 64 page IDs. Other
dtype footprints, short or partial domains, and causal domains retain the
natural 32-lane window or the ordinary per-tile path. This is an internal
consequence of the task topology, static geometry, and resource capacity; it
is not a user-selectable tuning parameter.

| Source | Responsibility |
| --- | --- |
| [`../../context.py`](../../context.py) | Public validation, metadata translation, automatic scheduling, JIT caching, and launch adaptation |
| [`fmha_kernel.py`](fmha_kernel.py) | Unified TS kernel and task graph construction |
| [`fmha_tasks.py`](fmha_tasks.py) | Load, MMA, softmax, correction, epilogue, page-offset, and scheduler work |
| [`fmha_resources.py`](fmha_resources.py) | GMEM/SMEM/TMEM resources and pipelines |
| [`helpers.py`](helpers.py) | Contiguous coordinates, masking, and schedule helpers |
| [`helpers_paged.py`](helpers_paged.py) | Paged-KV addressing and page-ID staging |

## Examples

Fixed contiguous causal attention:

```python
import torch
from flashinfer.attention.prims_ts import BatchPrefillTSWrapper

device = "cuda"
B, Sq, Sk, Hq, Hkv, D = 2, 256, 512, 8, 2, 128
q = torch.randn(B, Sq, Hq, D, device=device, dtype=torch.bfloat16)
k = torch.randn(B, Sk, Hkv, D, device=device, dtype=torch.bfloat16)
v = torch.randn_like(k)

wrapper = BatchPrefillTSWrapper()
wrapper.plan(q, k, v, mask_type="causal")
out = wrapper.run(q, k, v)
assert out.shape == q.shape
```

Packed Q with a paged K/V cache:

```python
import torch
from flashinfer.attention.prims_ts import BatchPrefillPagedTSWrapper

device = "cuda"
B, Hq, Hkv, D, page_size = 2, 8, 2, 128, 32
q_lens, kv_lens = (32, 48), (64, 80)
num_pages = 5

q = torch.randn(sum(q_lens), Hq, D, device=device, dtype=torch.float16)
k_cache = torch.randn(
    num_pages, Hkv, page_size, D, device=device, dtype=torch.float16
)
v_cache = torch.randn_like(k_cache)
qo_indptr = torch.tensor((0, 32, 80), device=device, dtype=torch.int32)
logical_kv_indptr = torch.tensor((0, 64, 144), device=device, dtype=torch.int32)
seq_lens_kv = torch.tensor(kv_lens, device=device, dtype=torch.int32)

# A 128-token K tile spans four 32-token pages. Pad each row to four
# columns by repeating its final valid page ID, and provide one plane for
# each of the separate K and V pools.
page_rows = torch.tensor(
    ((0, 1, 1, 1), (2, 3, 4, 4)), device=device, dtype=torch.int32
)
dense_page_idx_kv = torch.stack((page_rows, page_rows), dim=1)

wrapper = BatchPrefillPagedTSWrapper(kv_layout="HND")
wrapper.plan(
    device=q.device,
    batch_size=B,
    max_seq_len_q=max(q_lens),
    max_seq_len_k=max(kv_lens),
    max_num_pages_per_seq_kv=dense_page_idx_kv.shape[-1],
    num_qo_heads=Hq,
    num_kv_heads=Hkv,
    head_dim=D,
    q_dtype=q.dtype,
    kv_dtype=k_cache.dtype,
    out_dtype=q.dtype,
    page_size=page_size,
    mask_type="causal",
)
out = wrapper.run(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    logical_kv_indptr,
    dense_page_idx_kv,
    seq_lens_kv,
)
assert out.shape == q.shape
```

For CUDA graph capture, call `plan()` and perform one default-validating
`run()` first. Capture subsequent calls with `validate=False`, keep every
run-time tensor at a stable address, and pass a preallocated, non-overlapping
`out`.

## Limitations

- Paged context accepts separate compact HND K/V pools with page size 16, 32,
  64, or 128.
- `window_left=0` is unsupported; use `-1` to disable the window or a positive
  value to enable it.
- Positive windows are restricted to even-ratio GQA because the kernel pairs
  query heads that share a K/V head.
- Attention sinks, custom masks, and mixed Q/K/V dtypes are not exposed.
- Re-plan the paged wrapper after changing a static capacity, head or dtype
  geometry, page size, mask, window, or default scale. Request metadata may
  change between completed runs while remaining within the static plan.

## Validation

The public suite covers fixed, ragged, and paged layouts; MHA/GQA; both head
dimensions; `torch.float16`, `torch.bfloat16`, and `torch.float8_e4m3fn`
inputs; dense, causal, and left-window masks; nonidentity pages; scheduler
safety; CUDA graphs; and reference accuracy. Explicit input-to-output dtype
conversion coverage spans all nine pairings of FP16, BF16, and FP8 input and
output state.

```bash
pytest -q tests/attention/test_attention_ts_context.py
pytest -q tests/attention/test_attention_ts_mask.py
```
