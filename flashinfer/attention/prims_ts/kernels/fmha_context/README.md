# Task-Scheduled FMHA Context

This directory contains the CuTe DSL task-scheduled (TS) FMHA context/prefill
kernel used by FlashInfer's experimental Blackwell APIs. One implementation
serves fixed contiguous, packed contiguous with uniform or ragged request
lengths, and packed-query paged-KV attention with MHA or GQA.

The public API exposes attention semantics, not scheduling controls. Contiguous
and paged plans select a nonpersistent, static-persistent, or CLC-persistent
launch from logical work, task topology, live-metadata requirements, causal
domain structure, and GPU capacity. Paired, live-ragged, and zero-offset
triangular contiguous domains use CLC. Immutable single-instance
bottom-right-offset domains launch directly within one resident wave and use
static persistence above one wave. Paged causal plans use CLC under the default
dynamic-length contract; an explicit exact-uniform contract permits a static
schedule where the remaining topology allows it. Dense paged plans select a
direct or persistent launch from their logical work and topology. A positive
causal left window selects an internal head-paired GQA mapping; other cases use
the query-paired mapping.

## Public APIs

Import these entry points from `flashinfer.attention.prims_ts`:

| API | Use |
| --- | --- |
| `BatchPrefillTSWrapper` | Reusable fixed or packed contiguous Q/K/V plan. |
| `batch_prefill` | One-shot fixed or packed contiguous attention. |
| `BatchPrefillPagedTSWrapper` | Reusable packed-Q, paged-K/V plan. |
| `batch_prefill_with_paged_kv_cache` | One-shot packed-Q, paged-K/V attention. |

These experimental context entry points are not currently registered with
`fi_trace`. This exclusion is specific to context and does not limit tracing
support for other PrimTS APIs.

Both reusable wrappers use a static-spec lifecycle. `plan()` receives only
device, capacity, head, dtype, mask, window, and default-scale information. The
contiguous plan also freezes its `packed` storage-mode choice (`False` for fixed
BSHD, `True` for packed THD); the paged plan additionally receives page size
and optional `uniform_packed_lengths` / `has_q_offset` metadata contracts.
Neither plan retains Q/K/V tensors or request metadata. Every `run()` supplies
the current tensors and metadata: packed
contiguous offsets, per-token variable-window bounds for fixed-shape inputs, or
paged Q offsets, fixed page-table rows, and K/V lengths. Both wrappers own
one-element device tensors for their default softmax and output scales.
Contiguous variable-window plans additionally own mutable scratch that reduces
only the start bounds to per-CTA minima; end bounds remain per-token. Paged
plans own no workspace beyond the default scale tensors. Runtime validation is
enabled by default and may read metadata back to the host. `validate=False`
skips those checks for a previously validated steady state or CUDA Graph
launch; the caller then owns every dtype, device, shape, stride, alignment,
value, aliasing, and lifetime obligation in the runtime contract.

## Supported contract

| Feature | Support |
| --- | --- |
| GPU | SM100a/B200 (qualified); SM103a/B300 (architecture-gated, not yet signoff-qualified) |
| Head dimension | 128 or 256 |
| Head mapping | MHA/GQA; `Hq` must be divisible by `Hkv` |
| Q/K/V dtype | Matching `torch.float16`, `torch.bfloat16`, or `torch.float8_e4m3fn` |
| Output dtype | `torch.float16`, `torch.bfloat16`, or `torch.float8_e4m3fn` |
| Contiguous storage | Fixed BSHD or packed THD with uniform or ragged request lengths |
| Paged storage | Packed Q plus separate compact HND K/V page pools |
| Page size | 16, 32, 64, or 128 tokens |
| Mask | Dense or bottom-right causal; fixed contiguous also supports variable-window bounds |
| Sliding window | Positive causal left window; `window_left=-1` disables it |
| Scheduling | Automatic nonpersistent, static-persistent, or CLC-persistent selection; no public tuning knob |
| Accumulation | FP32 QK/PV and softmax state |

Current accuracy and performance signoff is on SM100a/B200. SM103a/B300 is
admitted by the runtime architecture guard but remains to be qualified.

A positive left window requires GQA with an even `Hq/Hkv` ratio greater than
one. Causal attention requires `Sq <= Sk` for every request at run time. All
tensor extents and packed request lengths must be positive. For each contiguous
run, the aggregate logical Q and K extents—`B*Sq` and `B*Sk` for fixed storage,
or `total_q` and `total_k` for packed storage—must each be at most
`2**31 - 256`. Paged runs apply that limit only to `total_q`. At plan time,
contiguous `B*max_seq_len_q` and `B*max_kv_len` must each satisfy the same cap;
paged plans cap only `B*max_seq_len_q`. This coordinate-representation limit
reserves 255 values for the padded tail of the largest supported 256-row query
work tile.

Q, K, V, and `out` must be compact, 16-byte-aligned CUDA tensors on one
device. Cumulative offsets, sequence lengths, and variable-window bounds must
be compact CUDA `torch.int32` tensors on that device and at least 4-byte
aligned. `block_tables` instead permits the row-strided layout documented
below. A caller-provided `out` must not overlap Q, K, V, any runtime metadata,
or plan-owned scale/scratch storage. The launch conservatively rejects
overlapping storage spans. The API returns O only; rowwise LSE and other
softmax state remain internal to the kernel.

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
- Wrapper and one-shot metadata: `qo_indptr[B + 1]`, `block_tables[B, C]`, and
  `seq_lens_kv[B]`, all CUDA `int32`.
- `block_tables` has unit column stride and a row stride at least `C`; compact
  `[B, C]` storage and padded views such as `[B, 2, C][:, 0, :]` are both
  accepted. `C` must be at least `ceil(max_kv_len / page_size)`.
  `seq_lens_kv` defines each row's active prefix and partial tail. Entries
  after `ceil(seq_lens_kv[b] / page_size)` are padding and are never
  dereferenced, so they need not contain valid page IDs. K and V pools use the
  same physical page IDs.
- Physical page IDs may be arbitrary, repeated, and nonidentity ordered.

Every cumulative-offset vector starts at zero and increases strictly. For
packed contiguous runs, `qo_indptr[-1]` equals `total_q`, `kv_indptr[-1]`
equals `total_k`, each Q delta is at most `max_seq_len_q`, and each K/V delta
is at most `max_kv_len`. For paged runs, `qo_indptr[-1]` equals `total_q`, each
Q delta is at most `max_seq_len_q`, and each `seq_lens_kv` value is at most
`max_kv_len`. All deltas and lengths are positive, and every page ID selected
for an active page indexes the physical cache.

For request `b`, bottom-right causal row `i` can see through
`Sk[b] - Sq[b] + i`. With `window_left=W>0`, the row retains that key and at
most `W` preceding keys. `sm_scale` defaults to `1 / sqrt(D)` and
`output_scale` defaults to 1; supplied scales must be finite, positive, and
representable as positive `float32` values.

For packed contiguous attention, planning fixes only static capacities and
compile-time semantics. Every run supplies `qo_indptr` and `kv_indptr`; their
values and packed tensor totals may change between runs while preserving the
exact planned batch, zero starting offsets, matching terminal tensor extents,
strictly positive deltas, and these per-request capacity bounds:

```text
0 < Sq[b] <= planned max_seq_len_q
0 < Sk[b] <= planned max_kv_len
```

Every causal replay must additionally satisfy `Sq[b] <= Sk[b]`. The
request-local bottom-right offset `Sk[b] - Sq[b]` may change and is derived
from the live offsets. Fixed variable-window plans likewise receive current
`[B, max_seq_len_q]` inclusive start/end bounds on every run. Only the start
bounds are reduced to per-CTA minima; end bounds remain per-token inputs. The
contiguous wrapper owns the mutable scratch used for that start reduction.

Paged wrapper planning fixes static capacities and one compile-time metadata
contract. The conservative defaults, `uniform_packed_lengths=False` and
`has_q_offset=True`, allow each run to provide different valid Q offsets,
block-table rows, K/V lengths, and physical page IDs without another plan. The
batch remains exact; Q deltas stay positive and within `max_seq_len_q`, K/V
lengths stay within `max_kv_len`, and the final Q offset matches the packed Q/O
extent. For causal attention, every per-run `Sq[b]` is no greater than `Sk[b]`.

`uniform_packed_lengths=True` is a caller promise that every Q delta equals
`max_seq_len_q` and every K/V length equals `max_kv_len`.
`has_q_offset=False` is a separate causal promise that `Sq[b] == Sk[b]` for
every request; dense attention ignores and canonicalizes this flag. These
promises compile exactly one narrower specialization rather than a runtime
choice between kernels. Re-plan before changing a promise. The one-shot paged
API already reads the metadata and derives the tightest valid flags for its
temporary plan.

With the default `validate=True`, `run()` checks tensor structure, shapes,
dtypes, devices, scales, output, aliasing, page-table strides, sequence
lengths, and active page IDs. Those metadata checks read device values back to
the host and may synchronize. `validate=False` skips validation and host
readback; callers using that path must enforce every dtype, device, shape,
stride, alignment, value, aliasing, lifetime, and selected plan-promise
obligation because invalid offsets, lengths, page IDs, or false compile-time
promises can produce incorrect results or out-of-bounds access.
CUDA Graph capture requires `validate=False` plus stable tensor shapes,
strides, and addresses, although values may change between completed replays.

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
| [`../../context.py`](../../context.py) | Public validation, automatic scheduling, JIT caching, and plan/run adaptation |
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
wrapper.plan(
    device=q.device,
    batch_size=B,
    max_seq_len_q=Sq,
    max_kv_len=Sk,
    num_qo_heads=Hq,
    num_kv_heads=Hkv,
    head_dim=D,
    q_dtype=q.dtype,
    kv_dtype=k.dtype,
    mask_type="causal",
)
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
block_tables = torch.tensor(
    ((0, 1, -1), (2, 3, 4)), device=device, dtype=torch.int32
)
seq_lens_kv = torch.tensor(kv_lens, device=device, dtype=torch.int32)

wrapper = BatchPrefillPagedTSWrapper(kv_layout="HND")
wrapper.plan(
    device=q.device,
    batch_size=B,
    max_seq_len_q=max(q_lens),
    max_kv_len=max(kv_lens),
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
    block_tables,
    seq_lens_kv,
)
assert out.shape == q.shape
```

For CUDA graph capture, call `plan()` and perform one default-validating
`run()` first. Capture subsequent calls with `validate=False`, keep every
run-time tensor shape, stride, and address stable, preserve any explicit
`uniform_packed_lengths` / `has_q_offset` promises, and pass a preallocated,
non-overlapping `out`. Callers must keep storage unmodified until queued work
completes. Before running on a CUDA stream that is not already ordered after the
planning stream, the caller must establish that dependency. Keep the wrapper and
all captured runtime tensors alive until every graph using that plan is destroyed.

## Limitations

- Paged context accepts separate compact HND K/V pools with page size 16, 32,
  64, or 128.
- `window_left=0` is unsupported; use `-1` to disable the window or a positive
  value to enable it.
- Positive windows are restricted to even-ratio GQA because the kernel pairs
  query heads that share a K/V head.
- Attention sinks, custom masks, and mixed Q/K/V dtypes are not exposed.
- Re-plan either wrapper after changing a static capacity, head or dtype
  geometry, mask, window, or default scale; page size and explicit metadata
  promises are also static for paged plans. Request tensors and metadata may
  change between completed runs while remaining within the static plan and its
  promises.
- A variable-window wrapper owns mutable CTA-minimum scratch, so its launches
  must not overlap across streams or captured graphs. Replanning either wrapper
  replaces plan-owned tensors and invalidates graphs captured from the prior
  plan; finish all prior launches and replays before replanning.

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
