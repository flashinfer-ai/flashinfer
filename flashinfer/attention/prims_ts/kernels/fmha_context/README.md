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
and optional `uniform_packed_lengths`, `has_q_offset`, and
`paged_v_tail_is_zero` contracts.
Neither plan retains Q/K/V tensors or request metadata. Every `run()` supplies
the current tensors and metadata: packed
contiguous offsets, per-token variable-window bounds and optional precomputed
CTA start minima for fixed-shape inputs, or paged Q offsets, fixed page-table
rows, and K/V lengths. Both wrappers own one-element device tensors for their
default softmax and output scales. Contiguous variable-window plans
additionally own mutable fallback scratch that reduces start bounds to per-CTA
minima only when the caller omits those minima; end bounds remain per-token.
Paged plans own no workspace beyond the default scale tensors. Runtime
validation is enabled by default and may read metadata back to the host.
`validate=False` skips those checks for a previously validated steady state or
CUDA Graph launch; the caller then owns every dtype, device, shape, stride,
alignment, value, aliasing, and lifetime obligation in the runtime contract.

## Supported contract

| Feature | Support |
| --- | --- |
| GPU | Blackwell SM100 and SM103; see validation scope below |
| Head dimensions | Equal QK/V: 128 or 256; separate contiguous MLA: QK=192, V=128 |
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

The established equal-dimension paths have accuracy and performance signoff
on SM100a/B200. The 192/128 extension has been validated on SM103/GB300;
its B200 performance remains to be measured.

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
device. Cumulative offsets, sequence lengths, and variable-window metadata must
be compact CUDA `torch.int32` tensors on that device and at least 4-byte
aligned. `block_tables` instead permits the row-strided layout documented
below. A caller-provided `out` must not overlap Q, K, V, any runtime metadata,
or active plan-owned scale/scratch storage. The launch conservatively rejects
overlapping storage spans. The API returns O only; rowwise LSE and other
softmax state remain internal to the kernel.

## Tensor and metadata layouts

Contiguous inputs:

- Fixed Q: `[B, Sq, Hq, Dqk]`; K: `[B, Sk, Hkv, Dqk]`;
  V: `[B, Sk, Hkv, Dv]`; O: `[B, Sq, Hq, Dv]`.
- Packed Q: `[total_q, Hq, Dqk]`; K: `[total_kv, Hkv, Dqk]`;
  V: `[total_kv, Hkv, Dv]`; O: `[total_q, Hq, Dv]`.
- Supported contiguous `(Dqk, Dv)` pairs: `(128, 128)`, `(192, 128)`,
  `(256, 256)`. The 192/128 path uses separate Q, K, and V tensors
  with compact rows; callers do not pad the head dimension.
- Packed metadata: compact CUDA `int32[B + 1]` `qo_indptr` and `kv_indptr`.
  Both start at zero, increase strictly, and end at the corresponding packed
  tensor extent.
- Fixed variable-window metadata: inclusive per-token starts and ends shaped
  `[B, max_seq_len_q]`. `variable_window_cta_starts` may additionally provide
  the exact minimum token start for every Q work tile, shaped
  `[B, ceil(max_seq_len_q / Tq)]`, where `Tq=256` for head dimension 128 and
  `Tq=128` for head dimension 256. The final entry in each row covers only the
  remaining real Q rows, without conceptual padding.

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
most `W` preceding keys. `sm_scale` defaults to `1 / sqrt(Dqk)` and
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
bounds need per-CTA minima; end bounds remain per-token inputs. A caller may
provide those minima once for all layers that share the same geometry and
metadata. Otherwise, the contiguous wrapper derives them on every run using
its mutable fallback scratch.

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
every request; dense attention ignores and canonicalizes this flag.
`paged_v_tail_is_zero=True` promises that unused rows following each request's
logical K/V length in its active final V page contain zero. This removes the
consumer-side post-TMA V-tail clear. The default `False` preserves correctness
for arbitrary contents, including NaNs, in those unused rows. These promises
compile exactly one specialization rather than a runtime choice between
kernels. Re-plan before changing a promise. The one-shot paged API derives the
tightest valid length flags for its temporary plan and conservatively keeps the
V-tail clear.

With the default `validate=True`, `run()` checks tensor structure, shapes,
dtypes, devices, scales, output, aliasing, page-table strides, sequence
lengths, and active page IDs. Those metadata checks read device values back to
the host and may synchronize. Caller-provided variable-window CTA starts are
also checked against the exact minimum of the corresponding per-token starts.
Validation does not inspect V-cache contents, so the caller owns
`paged_v_tail_is_zero=True` even with `validate=True`.
`validate=False` skips validation and host readback; callers using that path
must enforce every dtype, device, shape, stride, alignment, value, aliasing,
lifetime, and selected plan-promise obligation because invalid offsets,
lengths, page IDs, or false compile-time promises can produce incorrect results
or out-of-bounds access.
CUDA Graph capture requires `validate=False` plus stable tensor shapes,
strides, and addresses, although values may change between completed replays.
When caller-provided CTA starts are used, their values must be updated
consistently with the per-token starts before each replay.

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

Non-absorbed MLA (QK=192, V=128) reuses the paired 128-row Q schedule.
K is streamed in two 128-wide stages, with MMA restricted to the 128+64
logical columns. Two query tiles share each K stage, and two softmax groups
interleave with QK/PV work. Separate task-local bindings retain each K
descriptor for both query tiles. The MMA stage loops include partial slices
without overwriting another slice's binding. Q is rounded only to a 128-byte TMA
fragment in shared memory; BF16 Q therefore stores exactly 192 elements.
For BF16 input and output, O is staged in 64-wide pieces to fit both Q tiles
and the K/V ring. Ring depth follows the complete shared-memory footprint.
On SM103, unmasked score tiles use LDTM.STAT to combine their TMEM load
with the FP32 maximum reduction. DSL 4.7 uses a PTX 8.8 compatibility helper;
DSL versions that expose the native primitive use it directly. SM100 keeps
the software reduction, and masked tiles still apply masks before reducing.
Single-query schedules use one O handoff stage so the next PV cannot write
the accumulator until correction finishes, regardless of statistics storage.

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

Non-absorbed MLA with separate compact Q/K/V:

```python
q = torch.randn(1, 8192, 96, 192, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 8192, 1, 192, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 8192, 1, 128, device="cuda", dtype=torch.bfloat16)
wrapper = BatchPrefillTSWrapper()
wrapper.plan(q, k, v, mask_type="causal", out_dtype=torch.bfloat16)
out = wrapper.run(q, k, v)
assert out.shape == (1, 8192, 96, 128)
```

FP8 E4M3 uses the same shapes. For quantized operands, pass
`sm_scale=q_descale * k_descale / sqrt(192)` and
`output_scale=v_descale`; select the output dtype with `out_dtype`.

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
length or zero-tail promises, keep variable-window token and CTA starts
consistent when supplying both, and pass a preallocated, non-overlapping `out`.
Callers must keep storage unmodified until queued work completes. Before
running on a CUDA stream that is not already ordered after the planning stream,
the caller must establish that dependency. Keep the wrapper and all captured
runtime tensors alive until every graph using that plan is destroyed.

## Limitations

- The 192/128 MLA extension requires contiguous separate Q/K/V and does not
  support a positive left window. Paged K/V retains equal head dimensions.

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
- Variable-window launches that omit `variable_window_cta_starts` mutate the
  wrapper's fallback CTA-minimum scratch and must not overlap across streams or
  captured graphs. Supplying caller-owned CTA starts avoids that mutable plan
  state and permits overlap when the remaining tensor-lifetime requirements
  are satisfied. Replanning either wrapper replaces plan-owned tensors and
  invalidates graphs captured from the prior plan; finish all prior launches
  and replays before replanning.

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
