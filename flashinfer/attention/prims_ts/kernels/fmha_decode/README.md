# Task-Scheduled FMHA Decode

Plans select cached kernel policies and compile/load kernels using explicit
batch size and Q/KV bounds. Live request metadata belongs to `run()`.
The [shared planning contract](../../README.md#reusable-plans) describes
workspace sizing, standalone queries, cache reuse and CUDA graph lifetime.

This directory contains the CuTe DSL task-scheduled (TS) FMHA kernel used by
FlashInfer's experimental paged decode APIs on NVIDIA Blackwell GPUs. It
supports token-at-a-time decode, small fixed speculative-query batches, and
packed variable-length queries over a paged K/V cache.

The public API describes attention semantics and cache metadata. Tile shapes
and launch policy are selected internally for the problem and GPU. Fixed-Q
and packed-Q plans may use direct, persistent, or split-KV execution according
to the caller's split permission. Sliding-window plans remain nonsplit, but
may use direct or CLC-persistent execution. There is no public scheduler or
tuning knob and no fallback to another attention backend.

Fixed multi-query paged plans compare legal Swaps Q8/Q16/Q32 and Keeps Q64/Q128 candidates across page sizes 4, 8, 16, 32, 64, and 128. Each candidate passes the shared profile validator; eligibility does not imply every dtype, MMA orientation, or reducer supports every combination. Sliding-window scoring uses the visible union, bounded by `window_size + Q - 1`, instead of the model context capacity. Partial final query groups remain candidates: the cost model uses the ceil-divided Q grid and complete per-CTA cost. Q1, packed variable-Q, and explicitly pinned choices retain their existing selection paths. Windowed split-KV remains disabled.

D192 Keeps retains 128-wide SMEM stages while issuing PV N128+N64 into 192 logical O columns. Swaps keeps its padded PV computation and transposed output staging; output stores mask channels beyond D192. `smem_q_head_dim` expresses the 128-byte TMA fragment alignment: 192 elements for FP16/BF16, 256 for FP8. It does not widen the public Q tensor.

Fixed-Q D192 Keeps supports contiguous and paged K/V, dense/causal masks, direct/CLC scheduling, and the shared GMEM split reducer. FP16/BF16 uses 128- or 192-wide head stages; FP8 E4M3 uses 128-wide stages and FP16/BF16/E4M3 output, subject to the reducer's dtype contract. Automatic windows retain 128-wide staging. Sparse D192 Keeps has its own profile validator and staging policy.

The remaining dense D192 Keeps exclusions have different causes:

- Cluster-SMEM reduction has no Keeps implementation; use the GMEM reducer.
- Packed variable-Q and attention sinks have not been qualified for the one-instance staged D192 recipe, including its partial-row and sink-statistics handling. Their existing legal Swaps routes remain available; this is a qualification boundary rather than a Tensor Core dimension limitation.
- Windowed split-KV has not been qualified in the shared scheduler/reduction policy; visible windows use direct or CLC execution.
- Full-192 FP8 staging requires a different TMA fragment layout and validation. FP8 currently uses the validated 128-wide physical stages.

For eligible nonsplit grids with more than one resident wave, cluster launch
control (CLC) assigns work to resident CTAs. Underfilled fixed- or packed-Q grids may
instead split the K/V sequence and reduce partial outputs; other grids use the
direct static launch.

QToken-KvBlock-Sparse-Attention metadata uses one CUDA C++ CTA per route. Q1 maps its selected logical
blocks and causal tail directly through the dense page table. Q2--Q8 union at
most ``group_size * (block_topk + 1)`` selected/tail candidates with a shared
byte/bit map and compact the sorted logical union. A cached occupancy check
selects bounded radix sort when the map would exceed shared memory or add CTA
waves. The map is sized by model context, never global cache capacity. Plain
Int32 locators and packed membership words remain separate outputs; membership
bits are never fused into a locator.

Compatible FP8 G2–G8/page-4 Keeps plans instead use eight query words per
32-fragment tile, emitted directly into the same workspace with unused slots
and tail bits zero. The public
standalone metadata format remains page-byte masks. Sparse FP8 D256 Keeps
loads a full head stage as two SW128 planes and uses compact single-head
TensorMaps when cache strides permit; other cache layouts retain their descriptors.

Attention caches a grouped membership row in SMEM only when the complete
resource layout, including barriers, fits the compilation budget. Larger rows
stay in the existing immutable metadata buffer; Softmax reads packed words for
the current KV tile directly from GMEM. This does not change query grouping or
the caller's split-KV permission.

Packed QToken-KvBlock-Sparse-Attention uses an ordinary rank-3 Q TensorMap
over `[D, Hq, total_q_tokens]`. A grouped box loads the configured query group,
not spare token slots in `tileSizeQ`; the Q barrier credits rows outside that
box separately. `qo_indptr` still supplies sequence starts and valid lengths.
Loads across an internal sequence boundary occupy only inactive query rows:
existing row masks and guarded stores discard those rows, and QK/softmax/PV
do not mix different queries. TMA zero-fills beyond the packed allocation.
Other packed attention paths retain their ragged Q descriptors.

The combined QToken-KvBlock-Sparse-Attention metadata+attention API uses programmatic dependent launch
(PDL) for its final metadata-to-attention handoff. QToken-KvBlock-Sparse-Attention metadata producers
release at entry to overlap the attention prologue; the dependency wait still
covers completion of every metadata CTA. Attention allocates and initializes its task
barriers, SMEM, and TMEM first, then waits immediately before TaskManager can
read either output. Split-KV QToken-KvBlock-Sparse-Attention sends every configured split CTA through that
initialization and acquire, then contracts the useful runtime prefix. Pruned
split CTAs use the same TMEM teardown and dependent-release helpers before a
CTA-uniform PTX exit. Padded packed-Q CTAs have no task resources; they still
acquire through their explicit zero-work path and signal a following reducer
when one exists. A final nonsplit attention grid has no dependent to release.
Standalone attention over an already-built QToken-KvBlock-Sparse-Attention metadata triple remains stream
ordered and does not enter this PDL chain.

When split-KV uses a separate reduction kernel, each active attention CTA
signals at its true tail after task completion and TMEM teardown. Deferred QToken-KvBlock-Sparse-Attention
split padding retires as described above, while other runtime-inactive CTAs use
their terminal zero-work branch. The reducer initializes its register state
and any required shared-memory storage, then waits before reading any
producer-written partial output or statistics.
Independent query-offset metadata may be read before that wait. QToken-KvBlock-Sparse-Attention sequence
lengths remain behind it because they originate in the metadata producer two
PDL stages upstream. This preserves producer-to-reducer overlap while gating
every producer-dependent global-memory read.

The separate reducer uses one 512-thread CTA per 8 KiB output slice for
S2--S4. S5--S16 use one 512-thread CTA per 2 KiB slice: four adjacent lanes
fold interleaved split slots and merge through warp shuffles. Larger split
counts retain clustered 128-thread CTAs and distributed-SMEM merging.

## Public APIs

Import these entry points from `flashinfer.attention.prims_ts`:

| API | Use |
| --- | --- |
| `BatchDecodePagedTSWrapper` | Reusable eager `plan()` with explicit problem bounds and optional owned K/V lengths. |
| `batch_decode_with_paged_kv_cache` | One-shot interface with optional caller scratch, explicit bounds, and trusted capture-safe execution. |
| `get_prims_ts_batch_decode_workspace_size` | Size caller-owned scratch for the standalone launch. |
| `prepare_prims_ts_batch_decode_with_kv_cache` | Validate and compile a standalone launch once for a lightweight graph-safe `run()`. |

Trace a warmed stateful wrapper with `flashinfer.fi_trace(wrapper.run, ...)`.
The bound state supplies the selected query mode and output dtype. Trace
extensions for the new capabilities are deferred.

`plan()` takes model configuration, exact batch size, Q/KV bounds and optional
scratch. It selects cached policy, obtains compiled kernels and binds resources.
Every `run()` supplies query/cache, page tables and packed-Q offsets when needed.
Live K/V lengths come from each run unless `plan(seq_lens_kv=...)` owns a CUDA copy.
Fixed Q length is exact; packed Q lengths stay within planned `max_seq_len_q`.
Validation is enabled by default and may synchronize metadata to the host.

All profiles accept nonpositive **per-run** K/V lengths as inactive requests. Their page-table rows and Q/K/V data are not read. Caller-owned output stays untouched; newly allocated output for inactive rows is unspecified. Direct CTAs skip data work, persistent workers advance their queues, and reducers skip stale partials while retaining synchronization and PDL progress. Lengths can change from empty to live between graph replays. A length and table row are still required for every planned request; inactive table entries may be invalid. Plan-owned CPU lengths may also mark inactive requests and are canonicalized to zero, but remain immutable for the plan lifetime.

## Dense page tables only

Use `BatchDecodePagedTSWrapper` or the standalone PrimTS APIs with a dense
`[B, max_pages]` block table. CSR inputs and the shared
`BatchDecodeWithPagedKVCacheWrapper(backend="prims-ts")` adapter are not
supported. Rows may have padding between them; each row must be contiguous.

## Supported contract

| Feature | Support |
| --- | --- |
| GPU | SM100a/B200 and SM103/GB300; see the shared validation scope |
| Head dimension | 64, 128, 192, or 256 |
| Fixed Q length | Any positive integer representable by the metadata and tensor extents |
| Packed Q | Positive per-request lengths no greater than a positive static maximum |
| Head mapping | MHA/GQA; `Hq` must be divisible by `Hkv`, with `1 <= Hq/Hkv <= 128`. Qualified fixed-Q FP8 D64/D128/D256 paged profiles use grouped Swaps Q8/Q16/Q32 through ratio 32, Keeps Q64 through ratio 64, and Keeps Q128 through ratio 128. |
| Q/K dtype | Matching `torch.float16`, `torch.bfloat16`, or `torch.float8_e4m3fn` |
| V dtype | Equal to Q/K, or `torch.float8_e4m3fn` with `torch.bfloat16` Q/K (`v_dtype`; defaults to `k_dtype`) |
| Output dtype | `torch.float16` for `torch.float16` input; `torch.bfloat16` for `torch.bfloat16` input; `torch.float16`, `torch.bfloat16`, or `torch.float8_e4m3fn` for `torch.float8_e4m3fn` input |
| K/V layout | HND or NHD paged cache, combined or separate K/V tensors; a V dtype that differs from K requires separate tensors |
| Page size | 4, 8, 16, 32, 64, or 128 tokens; a logical fragment must divide the physical cache page |
| Maximum K/V length | `2,147,483,392` (`INT32_MAX - 255`), reserving the padded endpoint of a 256-token K/V tile |
| Mask | Dense or bottom-right causal |
| Sliding window | Causal left window; `window_left=-1` disables it and non-negative values include the current token |
| Scheduling | Automatic direct or CLC-persistent launch; `split_kv=True` permits eligible underfilled fixed/packed-Q grids to split. False disables splitting. Automatic sliding-window splits remain unqualified. |
| Accumulation | FP32 QK/PV and softmax state |

FP8-E4M3 Q with FP8-E4M3 or NVFP4 K/V can write BF16 directly. Set
`out_dtype=torch.bfloat16` when planning a wrapper, or
`out_dtype=torch.bfloat16` for one-shot execution and workspace sizing.
Prepared launches also accept BF16 output buffers. Both 16-bit output types
share split/reduction eligibility, including cluster SMEM reduction and the
standalone reducer; the existing shape and layout restrictions still apply.

See the shared [validation scope](../../README.md#validation-scope) for the
measured hardware and DSL versions.

The public paths require compact, 16-byte-aligned Q and output storage. Native
K/V views have logical HND shape with unit head-dimension stride and positive,
non-overlapping page/head/token strides aligned to 16 bytes. This includes
NHD physical storage and separate slices of interleaved K/V content. All query,
cache, metadata, output, and workspace tensors
must be on one CUDA device. Metadata uses 4-byte-aligned CUDA `torch.int32`;
the page table is contiguous within each row but may have padding between
rows. A caller-provided `out` must not overlap Q, K/V page
storage, run-time metadata, or caller-owned workspace. This is an unchecked
caller precondition in both validation modes. The default API returns O;
plans with LSE enabled can also return final FP32 LSE. Intermediate split-KV
statistics remain internal scratch.

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
- The execution Q/head extent, `B * max_seq_len_q * Hq`, must fit
  in signed `int32`. This also bounds every packed `total_q * Hq` extent.
- Combined K/V cache: `[num_pages, 2, Hkv, page_size, D]`.
- Separate K/V cache: a `(K, V)` tuple whose members are
  `[num_pages, Hkv, page_size, D]`. This is the only form that can carry a
  `torch.float8_e4m3fn` V next to `torch.bfloat16` K.
- Wrapper, standalone, and one-shot metadata use contiguous logical K/V
  lengths plus `block_tables[B, C]`. Every launch supplies live lengths. The table has unit inner stride and a
  non-overlapping row stride of at least `C`; padding between rows is
  supported. Packed runs additionally supply contiguous `qo_indptr[B + 1]`.
  The one-shot argument is named `seq_lens_kv`; wrapper and standalone entry
  points use `seq_lens_kv`. The standalone launch also takes a static
  `max_kv_len` upper bound. A convenience call may read metadata and allocate
  scratch. The unified one-shot API also supports trusted capture-safe calls
  with caller workspace, `out`, explicit `max_kv_len`, and `validate=False`;
  packed Q additionally requires `max_seq_len_q`. Warm the selected execution
  before capture and supply `lse` when LSE is enabled. See the shared planning
  contract for workspace initialization and lifetime rules.

For every request `b`, the fixed table must satisfy
`ceil(seq_lens_kv[b] / page_size) <= C`. Only that active row prefix must contain
valid physical page IDs; inactive tail entries are never read.
Query offsets start at zero, increase strictly, end at the packed Q extent,
and have every delta no larger than the planned `max_seq_len_q`. Causal
attention additionally requires each fixed or packed per-request Q length to
be no greater than the corresponding K/V length.

For request `b`, bottom-right causal row `i` can see through
`seq_len_k[b] - seq_len_q[b] + i`. A causal left window further retains the
current key and at most `window_left` preceding keys. Pages wholly before the
earliest visible key in a query group are not loaded, so recycled storage outside
the window cannot introduce NaNs into PV. The first partially visible page is
retained. `softmax_scale` defaults to
`1 / sqrt(D)` and `output_scale` defaults to 1; supplied scales must be finite,
positive Python scalars representable as positive `float32` values.

D64 NVFP4 K/V with page size 16 remains unsupported because its native staging
produces a misaligned access. Other transformed profiles use the normal
page-size and staging checks.

## Dataflow and source map

```text
Q + paged K/V
    -> staged Q/K/V
    -> QK MMA -> masked online softmax -> P
    -> PV MMA -> corrected O + internal log-normalizer state
    -> direct O, or split-KV partials -> reduction -> O
```

Profiles that stream score fragments (KV256 tiles, block-sparse routes, and
dense Q128/KV128 tiles with 16-bit K and V) choose their softmax arithmetic per
architecture; dense 8-bit Q128/KV128 tiles stage P in SMEM instead. SM100
evaluates three eighths of each fragment's exponentials (a quarter on KV256
tiles) as FMA polynomials to offload MUFU. SM103, with twice the MUFU ex2 rate,
keeps every exponential on MUFU, and its unmasked max pass takes fragment maxima
(or Sage scale-group maxima) from LDTM.STAT instead of a register reduction.

Eligible nonsplit work that exceeds one resident SM wave uses CLC-persistent
scheduling. A scheduler warp discovers each schedule token once and broadcasts it to
the worker tasks. Underfilled fixed- or packed-Q grids may instead split the
K/V sequence when the caller permits it with `split_kv=True`. False disables
splitting independently of Q layout. Automatic sliding-window splitting
remains unqualified; nonsplit work uses the same CLC/direct selection.

Fixed-table page IDs, packed-Q offsets and K/V lengths are per-run bindings.
Their values may change between completed launches; graph replay retains
captured addresses and shapes.

| Source | Responsibility |
| --- | --- |
| [`../../decode.py`](../../decode.py) | Public validation, planning, workspace binding, JIT caching, and launch adaptation |
| [`fmha_decode_config.py`](fmha_decode_config.py) | Kernel configuration and automatic launch selection |
| [`fmha_decode_kernel.py`](fmha_decode_kernel.py) | TS kernel construction and launch |
| [`fmha_decode_tasks.py`](fmha_decode_tasks.py) | Ordered load, MMA, softmax, correction, store, and scheduler work |
| [`fmha_decode_resources/`](fmha_decode_resources/) | GMEM/SMEM/TMEM resources and pipeline state |
| [`reduction.py`](reduction.py) | Separate split-KV reduction |
| [`block_sparse_prepare.py`](block_sparse_prepare.py) | Prepare of a block-sparse plan's compact route records from BSR or bitmask inputs |
| [`block_sparse_inspect.py`](block_sparse_inspect.py) | One-shot validation of canonical BSR and live paged metadata |
| [`../../sage/`](../../sage/) | Sage attention shared with the context kernel: recipe and scale validation, the names of the K transport and score scaling, the prepared-scale layout and its prepare kernel, index math, INT32 scores |

## Example

```python
import torch
from flashinfer.attention.prims_ts import (
    BatchDecodePagedTSWrapper,
    get_prims_ts_batch_decode_workspace_size,
    batch_decode_with_paged_kv_cache,
)

device = "cuda"
B, Hq, Hkv, D = 2, 32, 4, 128
page_size, pages_per_request = 32, 4
num_pages = B * pages_per_request

q = torch.randn(B, Hq, D, device=device, dtype=torch.float16)
kv = torch.randn(
    num_pages,
    2,
    Hkv,
    page_size,
    D,
    device=device,
    dtype=torch.float16,
)
block_tables = torch.arange(num_pages, device=device, dtype=torch.int32).view(
    B, pages_per_request
)
max_kv_len = pages_per_request * page_size
seq_lens_kv = torch.full((B,), max_kv_len, device=device, dtype=torch.int32)

wrapper = BatchDecodePagedTSWrapper(kv_layout="HND")
wrapper.plan(q.device, batch_size=B, max_seq_len_q=1, max_kv_len=max_kv_len, num_qo_heads=Hq, num_kv_heads=Hkv, head_dim=D, page_size=page_size, packed_query=False, q_dtype=q.dtype, k_dtype=kv.dtype, v_dtype=kv.dtype, out_dtype=q.dtype, mask_type='causal')
out = wrapper.run(q, kv, seq_lens_kv, block_tables)
assert out.shape == q.shape

# The standalone API uses caller-owned scratch and explicit K/V lengths.
workspace_bytes = get_prims_ts_batch_decode_workspace_size(
    batch_size=B, num_qo_heads=Hq, num_kv_heads=Hkv, head_dim=D,
    page_size=page_size, max_kv_len=max_kv_len, q_dtype=q.dtype, device=q.device,
)
workspace = torch.zeros(workspace_bytes, device=device, dtype=torch.int8)
standalone_out = batch_decode_with_paged_kv_cache(
    q,
    kv,
    block_tables,
    seq_lens_kv,
    workspace_buffer=workspace,
    max_kv_len=max_kv_len,
    mask_type="causal",
)
assert standalone_out.shape == q.shape
```

The wrapper owns its compiled specialization, plan-bound workspace, and any
sequence lengths supplied to `plan()`. If no `workspace_buffer` is passed to
`plan()`, the wrapper allocates private scratch. A workspace is mutable and
supports only one in-flight run or captured-graph replay; use separate wrappers
and workspaces for concurrent execution. Caller-owned scratch must remain alive
and must not overlap Q, K/V cache, metadata, or output storage.

With default `validate=True`, each wrapper run checks the fixed table, effective
K/V lengths, packed offsets when present, tensors, and output. When the plan
owns sequence lengths, pass `None` for the run's length argument. Once the caller has established
the remaining conditions, `validate=False` avoids explicit checks and host
metadata reads. Invalid run-owned lengths, page IDs, or offsets in that mode
may cause incorrect results or out-of-bounds access. Storage overlap is
unsupported and unchecked in both modes. Do not mutate
run-owned metadata concurrently with a launch or replay that reads it.

For the standalone workflow, query scratch with batch size, Q/KV bounds, head geometry, page
size, device and split capability. Allocate contiguous, 32-byte-aligned CUDA
`torch.int8` or `torch.uint8` scratch and zero it before its first trusted
standalone launch and whenever workspace-layout inputs change. Wrappers
initialize control storage by default; `initialize_workspace=False` transfers
that responsibility to the caller. Tensor-bound prepared plans also require
reinitialization before sharing scratch with a different prepared layout. Do not share it
between concurrent launches or captured graphs. It must not overlap Q, K/V
cache, metadata, or output storage. The standalone hot path trusts
`block_tables`, `seq_lens_kv`, and
packed-Q values: keep positive lengths within their static bounds; nonpositive per-run lengths mark inactive requests, keep
enough table columns for every request, and keep all active page IDs valid.
Sequence lengths, page IDs, and packed-Q offsets may change between
completed launches or graph replays while preserving those contracts and
stable captured storage. Do not mutate them concurrently with an execution
that reads them. These live values are not host-synchronized or fully
value-checked at launch; invalid lengths or IDs may cause incorrect results or
out-of-bounds access.

For CUDA graph capture, call `plan()` and perform one default-validating
`run()` first. Capture subsequent calls with `validate=False`, retain all
run-owned metadata and workspace storage at stable addresses, and pass a
preallocated compact, 16-byte-aligned `out` tensor. Replanning invalidates captured graphs and requires recapture.

## Limitations

- HND and NHD paged K/V are supported; contiguous K/V is outside this API.
  NHD separate caches use `[num_pages, page_size, Hkv, D]`; combined caches
  use `[num_pages, 2, page_size, Hkv, D]`. Select `kv_layout="NHD"` in the
  wrapper, convenience API, or prepared-plan API. Caches are
  addressed without repacking and reuse the HND compiled callable. NVFP4
  `kv_scale_factors` retain their documented canonical HND layout independently
  of the packed K/V data layout.
- Custom masks are not exposed.
- Sage attention (8-bit Q/K of one dtype with `torch.float8_e4m3fn` V and
  dequantization scales) runs only through `BlockSparseTSWrapper` and
  `block_sparse_attention`; see
  [Sage attention](../../README.md#sage-attention).
- Positive K/V lengths must be no greater than the static plan
  bound.
- Packed offsets are run-time wrapper inputs. Default wrapper validation checks
  them; `validate=False` and the standalone hot path trust them to preserve a
  synchronization-free launch. Live causal metadata must preserve
  `q_len[b] <= kv_len[b]`.

## Validation

The public accuracy, layout, mask, variable-Q, page-size, dtype, CUDA-graph,
split-KV, and resource-safety coverage lives in:

```bash
pytest -q tests/attention/test_attention_ts_decode.py
pytest -q tests/trace/test_fi_trace_template_consistency.py
```


### Final softmax statistics

See the shared [LSE, FP8 scaling and sink contract](../../README.md#final-statistics-and-sink-semantics).


### Runtime split-KV policy

See the shared [runtime split-KV policy](../../README.md#runtime-split-kv-policy).

Dense FMHA prepares two page-ID schedules together. If the host KV bound and
runtime count guarantee that the complete instruction-aligned CTA span fits in
32 page IDs, one CTA-relative window is retained through K and V. Longer spans
use the staged pipeline. The selection stays inside the compiled launcher, so
switching counts or crossing this span bound requires no additional JIT. The
window origin follows the CTA, including splits that start inside an aligned
32-ID region. Scratch-buffer sizing and ownership do not depend on this choice.
