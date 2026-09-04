# Task-Scheduled FMHA Decode

This directory contains the CuTe DSL task-scheduled (TS) FMHA kernel used by
FlashInfer's paged decode APIs on NVIDIA Blackwell GPUs. It
supports token-at-a-time decode, small fixed speculative-query batches, and
packed variable-length queries over a paged K/V cache.

The public API describes attention semantics and cache metadata. Tile shapes
and launch policy are selected internally for the problem and GPU. Fixed-Q
plans may use direct, persistent, or split-KV execution. Packed-Q and
sliding-window plans remain nonsplit, but may use direct or CLC-persistent
execution. There is no public scheduler or tuning knob and no fallback to
another attention backend.

For eligible nonsplit grids with more than one resident wave, cluster launch
control (CLC) assigns work to resident CTAs. Underfilled fixed-Q grids may
instead split the K/V sequence and reduce partial outputs; other grids use the
direct static launch.

## Public APIs

Import these entry points from `flashinfer.attention.prims_ts`:

| API | Use |
| --- | --- |
| `BatchDecodePagedTSWrapper` | Reusable static `plan()` plus per-run request-metadata `run()` interface. |
| `batch_decode_with_paged_kv_cache` | One-shot convenience interface. |
| `get_prims_ts_batch_decode_workspace_size` | Size caller-owned scratch for the standalone launch. |
| `prims_ts_batch_decode_with_kv_cache` | Standalone launch with caller-owned scratch and explicit `seq_lens`. |

Trace a planned stateful wrapper with `flashinfer.fi_trace(wrapper.run, ...)`.
The unbound `wrapper.run.fi_trace(...)` form is rejected because it cannot
carry the wrapper's plan-owned query mode and output dtype.

Prefer the reusable wrapper when a static cache geometry is used repeatedly.
`plan()` receives the device, exact batch and head geometry, page size, static
Q and K/V bounds, dtypes, mask, and window. It compiles the specialization and
either binds an optional caller-owned workspace or allocates private scratch;
it does not retain request metadata. Every `run()` supplies the current query,
cache, K/V lengths, and a fixed row-strided page table, plus query offsets for
packed Q. Validation is enabled by default. `validate=False` skips explicit
wrapper checks and host metadata reads for a previously validated steady state
or CUDA Graph launch; the caller then owns every value, bounds, aliasing, and
lifetime precondition.

An optional host sequence-length list or CPU tensor passed to `plan()` is
specialization evidence, not run-time metadata. It may prove that every row is
exactly `max_kv_len` or that every configured split is full. Every subsequent
run must preserve whichever predicate was selected. Default run validation
rechecks that predicate; with `validate=False`, preserving it is the caller's
responsibility. Omit plan-time `seq_lens` when those properties are not stable.
Sliding-window plans retain run-time K/V lengths because leading-tile skips
change the effective domain; persistent Q-dependent causal plans do the same
while recycling the task graph. These are automatic implementation choices.

## Supported contract

| Feature | Support |
| --- | --- |
| GPU | SM100a/B200 (qualified); SM103a/B300 (architecture-gated, not yet signoff-qualified) |
| Head dimension | 64, 128, or 256 |
| Fixed Q length | Any positive integer representable by the metadata and tensor extents |
| Packed Q | Positive per-request lengths no greater than a positive static maximum |
| Head mapping | MHA/GQA; `Hq` must be divisible by `Hkv` and `1 <= Hq/Hkv <= 32` |
| Q/K/V dtype | Q and K/V must match: `torch.float16`, `torch.bfloat16`, or `torch.float8_e4m3fn` |
| Output dtype | `torch.float16` for `torch.float16` input; `torch.bfloat16` for `torch.bfloat16` input; `torch.float16` or `torch.float8_e4m3fn` for `torch.float8_e4m3fn` input |
| K/V layout | HND paged cache, combined or separate K/V tensors |
| Page size | 16, 32, 64, or 128 tokens |
| Maximum K/V length | `2,147,483,392` (`INT32_MAX - 255`), reserving the padded endpoint of a 256-token K/V tile |
| Mask | Dense or bottom-right causal |
| Sliding window | Causal left window; `window_left=-1` disables it and non-negative values include the current token |
| Scheduling | Automatic direct or CLC-persistent launch; eligible underfilled fixed-Q grids may use split-KV. Packed-Q and sliding-window grids remain nonsplit. No public tuning knob. |
| Accumulation | FP32 QK/PV and softmax state |

Current accuracy and performance signoff is on SM100a/B200. SM103a/B300 is
admitted by the runtime architecture guard but remains to be qualified.

The public paths require compact, 16-byte-aligned Q and output storage. K/V
pages must have compact HND inner strides; a padded outer page stride is
allowed when pages do not overlap and both the tensor base and outer stride
are 16-byte aligned. All query, cache, metadata, output, and workspace tensors
must be on one CUDA device. Metadata uses 4-byte-aligned CUDA `torch.int32`;
the page table is contiguous within each row but may have padding between
rows. A caller-provided `out` must not overlap Q, K/V page
storage, run-time metadata, or caller-owned workspace. The launch
conservatively rejects overlapping storage spans. The API returns O only; LSE
and split-KV statistics are internal scratch.

The fixed table controls logical-to-physical lookup only. Native TMA tensor
maps still span the complete physical page pool and use each cache tensor's
runtime outer page stride, so page IDs may be arbitrary and physical pages may
have padded storage.

## Tensor and metadata layouts

- SQ=1 fixed Q/O: `[B, Hq, D]`.
- Fixed SQ>1 Q/O: `[B, SQ, Hq, D]`.
- Packed Q/O: `[total_q, Hq, D]`, with contiguous `int32[B + 1]`
  `qo_indptr`. Offsets start at zero, increase strictly, and end at
  `total_q`.
- The planned fixed-capacity Q/head extent, `B * max_seq_len_q * Hq`, must fit
  in signed `int32`. This also bounds every packed `total_q * Hq` extent.
- Combined K/V cache: `[num_pages, 2, Hkv, page_size, D]`.
- Separate K/V cache: a `(K, V)` tuple whose members are
  `[num_pages, Hkv, page_size, D]`.
- Wrapper and standalone metadata is supplied on every run as contiguous
  `seq_lens[B]` plus `block_tables[B, C]`. The table has unit inner stride and
  a non-overlapping row stride of at least `C`; padding between rows is
  supported. Packed runs additionally supply contiguous `qo_indptr[B + 1]`.
- The one-shot convenience API continues to use FlashInfer CSR metadata with
  `paged_kv_last_page_len[B]`. It validates and converts that metadata before
  invoking the fixed-table wrapper, so it is not CUDA-graph-capturable.
  Equal-width CSR rows can use a zero-copy view; ragged rows require a
  temporary dense table. The standalone launch uses explicit `seq_lens[B]`
  and a static `max_seq_len` upper bound.

Valid CSR metadata starts `paged_kv_indptr` at zero, increases it strictly,
and ends it at the number of used page-index entries. Every request owns at
least one page, every page ID indexes the physical cache, and one-shot
last-page lengths are in `[1, page_size]`. For every wrapper or standalone
request `b`, the fixed table must satisfy
`ceil(seq_lens[b] / page_size) <= C`. Only that active row prefix must contain
valid physical page IDs; inactive tail entries are never read.
Query offsets start at zero, increase strictly, end at the packed Q extent,
and have every delta no larger than the planned `max_seq_len_q`. Causal
attention additionally requires each fixed or packed per-request Q length to
be no greater than the corresponding K/V length.

For request `b`, bottom-right causal row `i` can see through
`seq_len_k[b] - seq_len_q[b] + i`. A causal left window further retains the
current key and at most `window_left` preceding keys. `bmm1_scale` defaults to
`1 / sqrt(D)` and `bmm2_scale` defaults to 1; supplied scales must be finite,
positive Python scalars representable as positive `float32` values.

## Dataflow and source map

```text
Q + paged K/V
    -> staged Q/K/V
    -> QK MMA -> masked online softmax -> P
    -> PV MMA -> corrected O + internal log-normalizer state
    -> direct O, or split-KV partials -> reduction -> O
```

Eligible nonsplit work that exceeds one resident SM wave uses CLC-persistent
scheduling. A scheduler warp discovers each schedule token once and broadcasts it to
the worker tasks. Underfilled fixed-Q grids may instead split the K/V sequence
and reduce partial outputs. Packed-Q and sliding-window work remains nonsplit:
it uses CLC above one resident wave and the direct static path otherwise.

K/V lengths, fixed-table page IDs, and packed-Q offsets are per-run inputs
loaded on every run and graph replay. Their storage and values may change
between completed launches without recompiling while the static plan contract
remains satisfied. CUDA Graph replay additionally requires stable captured
addresses and shapes.

| Source | Responsibility |
| --- | --- |
| [`../../decode.py`](../../decode.py) | Public validation, planning, workspace binding, JIT caching, and launch adaptation |
| [`fmha_decode_config.py`](fmha_decode_config.py) | Kernel configuration and automatic launch selection |
| [`fmha_decode_kernel.py`](fmha_decode_kernel.py) | TS kernel construction and launch |
| [`fmha_decode_tasks.py`](fmha_decode_tasks.py) | Ordered load, MMA, softmax, correction, store, and scheduler work |
| [`fmha_decode_resources/`](fmha_decode_resources/) | GMEM/SMEM/TMEM resources and pipeline state |
| [`reduction.py`](reduction.py) | Separate split-KV reduction |

## Example

```python
import torch
from flashinfer.attention.prims_ts import (
    BatchDecodePagedTSWrapper,
    get_prims_ts_batch_decode_workspace_size,
    prims_ts_batch_decode_with_kv_cache,
)

device = "cuda"
B, Hq, Hkv, D = 2, 32, 4, 128
page_size, pages_per_request = 32, 4
num_pages = B * pages_per_request

q = torch.randn(B, Hq, D, device=device, dtype=torch.float16)
kv = torch.randn(
    num_pages, 2, Hkv, page_size, D,
    device=device,
    dtype=torch.float16,
)
block_tables = torch.arange(num_pages, device=device, dtype=torch.int32).view(
    B, pages_per_request
)
max_seq_len = pages_per_request * page_size
seq_lens = torch.full((B,), max_seq_len, device=device, dtype=torch.int32)

wrapper = BatchDecodePagedTSWrapper(kv_layout="HND")
wrapper.plan(
    q.device,
    B,
    Hq,
    Hkv,
    D,
    page_size,
    max_seq_len,
    max_seq_len_q=1,
    packed_query=False,
    q_data_type=q.dtype,
    kv_data_type=kv.dtype,
    o_data_type=q.dtype,
    mask_type="causal",
    # Optional stable evidence enables a fixed-length specialization.
    seq_lens=[max_seq_len] * B,
)
out = wrapper.run(q, kv, seq_lens, block_tables)
assert out.shape == q.shape

# The standalone API uses caller-owned scratch and explicit K/V lengths.
workspace_bytes = get_prims_ts_batch_decode_workspace_size(
    B,
    Hq,
    Hkv,
    D,
    page_size,
    max_seq_len,
    q_dtype=q.dtype,
    mask_type="causal",
    device=q.device,
)
workspace = torch.zeros(workspace_bytes, device=device, dtype=torch.int8)
standalone_out = prims_ts_batch_decode_with_kv_cache(
    q,
    kv,
    workspace,
    block_tables,
    seq_lens,
    max_seq_len,
    mask_type="causal",
)
assert standalone_out.shape == q.shape
```

The wrapper owns its compiled specialization and plan-bound workspace, but not
request metadata. If no `workspace_buffer` is passed to `plan()`, the wrapper
allocates private scratch. A workspace is mutable and supports only one
in-flight run or captured-graph replay; use separate wrappers and workspaces
for concurrent execution. Caller-owned scratch must remain alive and must not
overlap Q, K/V cache, metadata, or output storage.

With default `validate=True`, each wrapper run checks the per-run fixed table,
K/V lengths, packed offsets when present, tensors, output, and any selected
sequence-length specialization. Once the caller has established those
conditions, `validate=False` avoids the explicit checks and host metadata
reads. Invalid per-run lengths, page IDs, offsets, aliases, or specialization
predicates in that mode may cause incorrect results or out-of-bounds access.
Do not mutate metadata concurrently with a launch or replay that reads it.

For the standalone workflow, call
`get_prims_ts_batch_decode_workspace_size()` with the same shape, dtype, mask,
window, and Q-layout arguments as the launch. Allocate at least that many
bytes as a contiguous, 32-byte-aligned CUDA `torch.int8` or `torch.uint8`
tensor. Zero it before first use and re-zero it whenever any workspace-layout
input, including batch size, changes because the internal workspace section
offsets can move even when the compiled callable is reused. Do not share it
between concurrent launches or captured graphs. It must not overlap Q, K/V
cache, metadata, or output storage. The standalone hot path trusts
`block_tables`, `seq_lens`, and
packed-Q values: keep lengths positive and within their static bounds, keep
enough table columns for every request, and keep all active page IDs valid.
Sequence lengths, page IDs, and packed-Q offsets may change between
completed launches or graph replays while preserving those contracts and
stable captured storage. Do not mutate them concurrently with an execution
that reads them. These per-run values are not host-synchronized or fully
value-checked at launch; invalid lengths or IDs may cause incorrect results or
out-of-bounds access.

For CUDA graph capture, call `plan()` and perform one default-validating
`run()` first. Capture subsequent calls with `validate=False`, retain all
run-time metadata and workspace storage at stable addresses, and pass a
preallocated compact, 16-byte-aligned `out` tensor.

## Limitations

- Only HND paged K/V is supported; contiguous K/V and NHD caches are outside
  this API.
- Attention sinks and custom masks are not exposed.
- Q, K, and V cannot use mixed dtypes.
- Runtime K lengths must be positive and no greater than the static plan bound.
- Packed offsets are run-time wrapper inputs. Default wrapper validation checks
  them; `validate=False` and the standalone hot path trust them to preserve a
  synchronization-free launch. Per-run causal metadata must preserve
  `q_len[b] <= kv_len[b]`.

## Validation

The public accuracy, layout, mask, variable-Q, page-size, dtype, CUDA-graph,
split-KV, and resource-safety coverage lives in:

```bash
pytest -q tests/attention/test_attention_ts_decode.py
pytest -q tests/trace/test_fi_trace_template_consistency.py
```
