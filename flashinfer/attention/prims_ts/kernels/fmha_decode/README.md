# Task-Scheduled FMHA Decode

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

For eligible nonsplit grids with more than one resident wave, cluster launch
control (CLC) assigns work to resident CTAs. Underfilled fixed- or packed-Q grids may
instead split the K/V sequence and reduce partial outputs; other grids use the
direct static launch.

QToken-KvBlock-Sparse-Attention metadata uses one CUDA C++ CTA per route. Q1 maps its selected logical
blocks and causal tail directly through the dense page table. Q2--Q8 sort at
most ``group_size * (block_topk + 1)`` tagged selected/tail candidates in
shared memory, unique equal logical IDs while OR-reducing query-membership
bits, and map only the compact union. Work and temporary storage therefore do
not scale with the configured model length or global cache capacity. Plain
Int32 locators and packed membership words remain separate outputs; membership
bits are never fused into a locator.

Attention caches a grouped membership row in SMEM only when the complete
resource layout, including barriers, fits the compilation budget. Larger rows
stay in the existing immutable metadata buffer; Softmax reads packed words for
the current KV tile directly from GMEM. This does not change query grouping or
the caller's split-KV permission.

The combined QToken-KvBlock-Sparse-Attention metadata+attention API uses programmatic dependent launch
(PDL) for its final metadata-to-attention handoff. QToken-KvBlock-Sparse-Attention metadata producers
release only after their page indices, membership words, and sequence lengths
are published. Every active attention CTA allocates and initializes its task
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

## Public APIs

Import these entry points from `flashinfer.attention.prims_ts`:

| API | Use |
| --- | --- |
| `BatchDecodePagedTSWrapper` | Reusable static `plan()` with plan- or run-owned K/V lengths. |
| `batch_decode_with_paged_kv_cache` | One-shot interface with optional caller scratch, explicit bounds, and trusted capture-safe execution. |
| `get_prims_ts_batch_decode_workspace_size` | Size caller-owned scratch for the standalone launch. |
| `prepare_prims_ts_batch_decode_with_kv_cache` | Validate and compile a standalone launch once for a lightweight graph-safe `run()`. |

Trace a planned stateful wrapper with `flashinfer.fi_trace(wrapper.run, ...)`.
The unbound `wrapper.run.fi_trace(...)` form is rejected because it cannot
carry the wrapper's plan-owned query mode, output dtype, and length ownership.
Pass the same length argument as `run()`: a CUDA tensor for a run-owned plan or
`None` for a plan-owned vector. Trace names and tags distinguish the two
lifecycles even when both compiled K/V modes are dynamic.

Prefer the reusable wrapper when static attention geometry and capacity are
used repeatedly.
`plan()` receives the device, exact batch and head geometry, page size, static
Q and K/V bounds, dtypes, mask, and window. It compiles the specialization and
either binds an optional caller-owned workspace or allocates private scratch;
it also copies optional host `seq_lens` into plan-owned CUDA storage. Every
`run()` supplies the current query, cache, and a fixed row-strided page table,
plus query offsets for packed Q. Exactly one lifecycle owns K/V lengths:

- If `plan()` receives a host list or CPU tensor, every `run()` must pass
  `seq_lens=None` and uses the plan-owned CUDA copy.
- If `plan()` receives `seq_lens=None`, every `run()` must pass a CUDA
  `seq_lens` tensor. The compiled K/V-prefix and K/V-length modes are dynamic.

Supplying lengths to both calls or neither call is rejected regardless of
`validate`. Plan-owned lengths may prove that every row is exactly
`max_kv_len` or that the full configured split-CTA fanout is active for every
batch/Q group. They are frozen until the next successful `plan()` call; omit
plan-time `seq_lens` when lengths must change between runs or graph replays.
Validation is enabled by default. `validate=False` skips explicit wrapper
checks and host metadata reads for a previously validated steady state or CUDA
Graph launch; the caller then owns every remaining value, bounds, aliasing, and
lifetime precondition.
Sliding-window plans retain dynamic K/V-length handling because leading-tile
skips change the effective domain; persistent Q-dependent causal plans do the
same while recycling the task graph. This kernel mode is independent of
whether the plan or run owns the length vector.

## Dense page tables only

Use `BatchDecodePagedTSWrapper` or the standalone PrimTS APIs with a dense
`[B, max_pages]` block table. CSR inputs and the shared
`BatchDecodeWithPagedKVCacheWrapper(backend="prims-ts")` adapter are not
supported. Rows may have padding between them; each row must be contiguous.

## Supported contract

| Feature | Support |
| --- | --- |
| GPU | SM100a/B200 (qualified); SM103a/B300 (architecture-gated, not yet signoff-qualified) |
| Head dimension | 64, 128, or 256 |
| Fixed Q length | Any positive integer representable by the metadata and tensor extents |
| Packed Q | Positive per-request lengths no greater than a positive static maximum |
| Head mapping | MHA/GQA; `Hq` must be divisible by `Hkv`, with `1 <= Hq/Hkv <= 128`. Qualified fixed-Q FP8 D64/D128/D256 page-32 profiles use grouped Swaps Q8/Q16/Q32 through ratio 32, Keeps Q64 through ratio 64, and Keeps Q128 through ratio 128. |
| Q/K dtype | Matching `torch.float16`, `torch.bfloat16`, or `torch.float8_e4m3fn` |
| V dtype | Equal to Q/K, or `torch.float8_e4m3fn` with `torch.bfloat16` Q/K (`v_data_type`; defaults to `k_data_type`) |
| Output dtype | `torch.float16` for `torch.float16` input; `torch.bfloat16` for `torch.bfloat16` input; `torch.float16` or `torch.float8_e4m3fn` for `torch.float8_e4m3fn` input |
| K/V layout | HND paged cache, combined or separate K/V tensors; a V dtype that differs from K requires separate tensors |
| Page size | 4, 8, 16, 32, 64, or 128 tokens; a logical fragment must divide the physical cache page |
| Maximum K/V length | `2,147,483,392` (`INT32_MAX - 255`), reserving the padded endpoint of a 256-token K/V tile |
| Mask | Dense or bottom-right causal |
| Sliding window | Causal left window; `window_left=-1` disables it and non-negative values include the current token |
| Scheduling | Automatic direct or CLC-persistent launch; `split_kv=True` permits eligible underfilled fixed/packed-Q grids to split. False disables splitting. Automatic sliding-window splits remain unqualified. |
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
storage, run-time metadata, or caller-owned workspace. This is an unchecked
caller precondition in both validation modes. The API returns O only; LSE
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
  `[num_pages, Hkv, page_size, D]`. This is the only form that can carry a
  `torch.float8_e4m3fn` V next to `torch.bfloat16` K.
- Wrapper, standalone, and one-shot metadata use contiguous logical K/V
  lengths plus `block_tables[B, C]`. Wrapper lengths are either copied into
  plan-owned CUDA storage or supplied to each run; the other APIs always take
  them at launch. The table has unit inner stride and a
  non-overlapping row stride of at least `C`; padding between rows is
  supported. Packed runs additionally supply contiguous `qo_indptr[B + 1]`.
  The one-shot argument is named `seq_lens_kv`; wrapper and standalone entry
  points use `seq_lens`. The standalone launch also takes a static
  `max_seq_len` upper bound. The one-shot helper reads request metadata to
  derive its plan bounds and therefore is not CUDA-graph-capturable; reusable
  graph paths plan the wrapper before capture.

For every request `b`, the fixed table must satisfy
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
the worker tasks. Underfilled fixed- or packed-Q grids may instead split the
K/V sequence when the caller permits it with `split_kv=True`. False disables
splitting independently of Q layout. Automatic sliding-window splitting
remains unqualified; nonsplit work uses the same CLC/direct selection.

Fixed-table page IDs and packed-Q offsets are per-run bindings and are loaded
on every run and graph replay. K/V lengths come from exactly one source. A
plan-owned vector has stable storage and values until replan and may enable
full-prefix or uniform-maximum specialization. A run-owned vector may change
between completed launches without recompiling; CUDA Graph replay additionally
requires its captured address and shape to remain stable.

| Source | Responsibility |
| --- | --- |
| [`../../decode.py`](../../decode.py) | Public validation, planning, workspace binding, JIT caching, and launch adaptation |
| [`fmha_decode_config.py`](fmha_decode_config.py) | Kernel configuration and automatic launch selection |
| [`fmha_decode_kernel.py`](fmha_decode_kernel.py) | TS kernel construction and launch |
| [`fmha_decode_tasks.py`](fmha_decode_tasks.py) | Ordered load, MMA, softmax, correction, store, and scheduler work |
| [`fmha_decode_resources/`](fmha_decode_resources/) | GMEM/SMEM/TMEM resources and pipeline state |
| [`fmha_decode_resources/sage_scales.py`](fmha_decode_resources/sage_scales.py) | Sage scale addressing for the softmax and epilogue (flat layout, contiguous and block-sparse providers) |
| [`../../sage.py`](../../sage.py) | `SageAttentionConfig`, `SageAttentionParams`, flat-layout helpers and host-side scale validation |
| [`reduction.py`](reduction.py) | Separate split-KV reduction |

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
    k_data_type=kv.dtype,
    v_data_type=kv.dtype,
    o_data_type=q.dtype,
    mask_type="causal",
    # Optional fixed lengths enable a plan-owned specialization.
    seq_lens=[max_seq_len] * B,
)
out = wrapper.run(q, kv, None, block_tables)
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
standalone_out = batch_decode_with_paged_kv_cache(
    q,
    kv,
    block_tables,
    seq_lens,
    workspace_buffer=workspace,
    max_kv_len=max_seq_len,
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
K/V lengths, packed offsets when present, tensors, and output. Plan-owned
lengths and their specialization predicates were validated by `plan()` and are
not revalidated against a second length vector. Once the caller has established
the remaining conditions, `validate=False` avoids explicit checks and host
metadata reads. Invalid run-owned lengths, page IDs, or offsets in that mode
may cause incorrect results or out-of-bounds access. Storage overlap is
unsupported and unchecked in both modes. Do not mutate
run-owned metadata concurrently with a launch or replay that reads it.

For the standalone workflow, call
`get_prims_ts_batch_decode_workspace_size()` with the same shape, Q/K/V and
output dtype, mask,
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
that reads them. These live values are not host-synchronized or fully
value-checked at launch; invalid lengths or IDs may cause incorrect results or
out-of-bounds access.

For CUDA graph capture, call `plan()` and perform one default-validating
`run()` first. Capture subsequent calls with `validate=False`, retain all
run-owned metadata and workspace storage at stable addresses, and pass a
preallocated compact, 16-byte-aligned `out` tensor. A successful replan changes
plan-owned length storage and requires graph recapture.

## Sage attention

The same kernel runs 8-bit Q/K/V with per-block dequantization scales
("Sage attention") when planned through the contiguous
`BlockSparseTSWrapper` with `sage_config=SageAttentionConfig(...)` and run with
the
scale tensors in `sage=SageAttentionParams(...)`. The user-facing
contract (recipes, scale layouts, block-sparse summaries, examples) is in the
[PrimTS guide](../../README.md#sage-attention-8-bit-qkv-with-dequantization-scales);
this section records what the kernel does with the scales.

### Gating

`FmhaDecodeConfig.sage_k_block_size > 0` enables the feature
(`use_sage_attention`); `sage_q_block_size` and `sage_v_mean` are only legal
with it. The Sage rules live next to the rules they extend. `validate_dtypes`
keeps Q == K for every recipe (`kv_dtype` is the K dtype) and, with Sage on,
requires Q and K in `Float8E4M3FN` or `Int8`, `Float8E4M3FN` V and a 16-bit
output; `Int8` K is legal only with Sage scales. `v_dtype` always holds a
concrete type: `make_decode_config` sets it from `kv_dtype` when a caller
leaves it out, and code that assigns `kv_dtype` assigns it as well.
`validate_sage_profile` checks the rest of the recipe: `sage_k_block_size` in
`SAGE_K_BLOCK_SIZES`, `sage_q_block_size` a power of two no larger than
`tile_size_q`, contiguous K/V with `headdim == 128`, a two-instance Keeps
profile (Q64/KV256 or Q128/KV128), and a nonsplit grid (static direct or the
persistent scheduler): split-KV, the separate and cluster reductions, packed
variable-length Q, sliding windows and attention sinks are all rejected;
`validate_block_sparse_profile` keeps its matching-IO rule for block-sparse
plans without Sage. The
qualified Sage recipes are the `_SAGE_GROUPED_KEEPS_PROFILES` table beside
the contiguous grouped-Keeps recipes, selected by `use_sage_attention`
because the same E4M3 dtypes without scales name the static-only FP8
profile. The block-sparse wrapper adds only what its compile key needs (Q and
K share one dtype, 8-bit Q/K require `sage_config`, a 16-bit V matches K) and
validates the per-run scale tensors; the recipe rules above live in the kernel
configuration alone.

Every Sage branch is a compile-time predicate on the configuration, so a
16-bit kernel carries no Sage code. The softmax and correction mechanisms
that do not depend on the element width are written once and shared by every
dtype: the exponent addend formed once per row, the rolled masked max pass
with a compile-time tail location, the seeded probability sum chains, the CLC
response slot in the unified SMEM block and one correction store wait per O
stage. The byte-wide-only policies (`prefetches_next_p_fragment`,
`splits_kv_tile_256_tail_columns`, the four-stage KV256 ring) are measured
performance choices gated on `use_8bit_qkv`, not dtype requirements.

### Where the scales enter softmax

Each softmax lane owns one Q row and streams its 128 score columns as four
K32 fragments. Per tile the lane loads `sfQ[row]` once. Per fragment it loads
the K scales covering its 32 columns: with `k_block_size >= 32` one scale,
with 16 two, with 4 eight and with 1 thirty-two, giving `sage_k_groups_per_fragment`
compile-time groups (the fragment is unrolled, so group boundaries cost no
per-element work). With `c = sm_scale * log2(e)`:

```text
max pass:  gmax_g  = max(s in group g)              INT8: the biased FP32 scores, minus the bias per group
           fragmax = max_g(gmax_g * sfK_g)          one packed FMUL per group pair
           rowmax_true = rowmax * sfQ               once, after the last fragment
exp pass:  p = EX2(FFMA(s, (c * sfQ) * sfK_g, -c * rowmax_true + log2(448)))
           INT8: the bias leaves through the group's addend
```

The row sum, the correction factor `exp2(c * (m_old - m_new))`, the
correction skip and LSE all operate on the dequantized maximum and are
unchanged. Both passes use one algebra: the tile's raw `sfK` words per scale
group, and the row's `sfQ` applied once per tile (to the tile maximum, to the
`c * sfQ` factor of the exponent multipliers, and to the proxy tail shift).
Where the `sfK` words live is a compile-time strategy in `sage_scales.py`,
selected by `sage_k_scales_in_smem` and shared by the S and P resources and
the route metadata consumer of one softmax instance: `RegisterSageKScales`
(K blocks of 16 tokens and larger) keeps the lane's words in a rotating
register array that both passes read fragment by fragment without runtime
indexing; `SmemSageKScales` (blocks of 4 and 1 token, too many groups for
registers) keeps the tile's words in a two-tile SMEM ring of the instance
(filled from `k_scale` by the lanes of a dense tile or copied from the route
stage, published with one named barrier) that both passes read with 16-byte
broadcast loads per fragment. The passes see only the strategy's `open`,
`fragment` and `advance`. The sources are the same for both: the contiguous
provider derives each fragment's first token from the tile offset and loads
`sfK` from the softmax threads before the score wait (`load_lane_k_scales`);
the block-sparse provider moves the loads off the softmax warps, the load
warp resolving and staging the route's `sfK` words (`sage_k_scale_words` per
route, ordered by consuming half and lane array) next to the route metadata
from the route's K64 atom origins; a proxy route switches the K source to
`k_summary_scale` indexed by summary position (`block_sparse_k_scale_source`).
Tokens beyond the sequence end and Q rows beyond the valid count clamp to the
last valid slot, so masked columns keep a finite scale and exponentiate to
zero.

The epilogue (`_store_final_o_columns`) multiplies each output column by
`norm_scale * sfV[c]` and adds `v_mean[c]` when configured. Both vectors are
per KV head: the correction warps copy them into SMEM once per tile while no
output is pending (`stage_v_channel_scales`), and the epilogue reads its
column range back with vector loads (`load_staged_v_channel_scales`, which
returns a zero mean array without `sage_v_mean`, so the epilogue is one fused
multiply-add). This is exact because `sfV` is constant along the PV
reduction.

### K/V ring depth

The M64N256 profile stages one complete 256-row K or V tile per shared-ring
slot, so the ring depth follows the element width
(`KV_TILE_256_SHARED_FIFO_STAGES = 3`,
`KV_TILE_256_BYTE_WIDE_SHARED_FIFO_STAGES = 4`). A 16-bit tile is 64 KiB:
three stages occupy 192 KiB, and with the 16-KiB Q stage, the dedicated tail
exchange and the metadata and barrier allocations the CTA sits at about
227 KiB, the SM100 carveout. A byte-wide tile (E4M3 or Int8 K with E4M3 V) is
32 KiB, so three stages leave about 111 KiB unused, and three is also too
shallow: the ring alternates K and V tiles, a V slot stays held until its
tile's PV has run, and with three slots the load of K(t+2) waits for that
release, so the QK of tile t+2 sees the whole TMA latency. With four slots
every K load takes the slot the previous QK freed and every V load the slot
the previous PV freed. On B200 the block-sparse Sage kernels run about 7.5%
faster with four stages than with three and gain nothing past four. The
four-stage build demotes one pair of persistent scheduler words per warp role
to an 8-byte stack (18 STL and 28 LDL per kernel, reloaded once per tile away
from the softmax critical path); against a five-stage build, which keeps every
role spill-free, four stages measure equal or faster on every byte-wide case
(CUDA Graph replay minimum, three paired legs): dense S=10800 H=40 0.982, INT8
proxy at the SOL shape 0.967, FP8 proxy 0.996, FP8 and INT8 exact within 0.3%,
Q128 exact 0.995. The 32 KiB the shallower ring leaves free stay available
for the small-block `sfK` buffers. An explicit `kv_stages` overrides the
default in both directions.

### INT32 scores

`Int8` Q/K use the `INT8` tcgen05 kind with an `Int32` accumulator
(`uses_int32_scores`). Over `D = 128` every dot product satisfies
`|s| <= 128 * 127^2 < 2^21`, so the score is exact; input quantization is
the only QK-side error. E4M3 Q/K use `F8F6F4` with FP32 accumulation. BMM2
is `F8F6F4` on E4M3 P and V in both recipes.

The softmax never converts INT32 scores. BMM1 accumulates onto the bit
pattern of `INT32_SCORE_BIAS = 1.5 * 2^23`: every integer in `[2^23, 2^24)`
is an FP32 number with unit spacing, so with `|s| <= 128 * 128 * 128 = 2^21`
(INT8 magnitudes over `D = 128`) the final
accumulator `bias + s` is exactly the FP32 number `12582912.0 + s`. The seed
is written by one `kind::f16` MMA step over constant BF16 operands
(one BF16 tile as both operands, every `K = 16` row `[1024, 1024, 1024, 0]`
repeated, so the twelve products `2^20` sum to exactly `1.5 * 2^23`) issued
right before the INT8 K steps
(`_seed_score_bias` in `tmem_s.py`); the tensor core writes the slot at
accumulator bandwidth and the INT8 steps accumulate onto it in tcgen05 issue
order. Seeding the 64 KiB slot with `tcgen05.cp` or `tcgen05.st` costs several
times more than the conversions the bias removes.

The max pass reduces biased scores with FP32 maxima and subtracts the bias
once per scale group (exact on the unit spacing; `-FLT_MAX` is unchanged by
it), so masked tiles share the FP32 path: `-FLT_MAX` on masked lanes, the
proxy tail shift applied in FP32. The P pass folds `-bias * c * sfQ * sfK_g`
into each group's exponent addend with one FMA per group, so the per-element
work is the same FFMA/EX2 stream as for FP32 scores for every K block size
(with the one-token K block every score is its own group, so the fold costs
one FMA per score). Two roundings differ
from converting every element, both below the INT8 quantization noise: the
per-group addend is rounded once, to at most half an ulp of
`bias * c * sfQ * sfK_g` (at most 0.75 quantized score units), and the proxy
tail shift lands on the biased score's unit spacing (at most 0.5 quantized
score units on that one lane).

### Precision notes

- P is E4M3 with the static scale `FP8_P_QUANT_SCALE = 448`, which represents
  probabilities down to about 4.4e-6 with 3-bit relative precision. Row sums
  are accumulated from the quantized P so the tail truncation cancels in
  normalization.
- `defers_softmax_anchor_updates` is off for every 8-bit profile: the 448
  scale requires `p <= 1`, which a deferred anchor (lag up to `2^8`) would
  violate, so FP8 P always anchors on the exact row maximum and pays the
  TMEM rescale on every maximum increase. A `448 / 2^k` static scale would
  buy anchor headroom at the cost of P range; the kernel does not make that
  trade.
- One quarter of the score pairs in a streamed fragment
  (`KV_TILE_256_EX2_EMULATED_PAIRS = 4` of 16) evaluate `exp2` as an FMA
  polynomial instead of MUFU. Its relative error moves about one to two
  percent of the E4M3 probabilities to the neighbouring quantization step.
  Dense rows of about a thousand tokens average this out (the dense fidelity
  tests hold an absolute 2e-3), while a sparse row that attends to a few
  hundred tokens
  exposes up to about 1e-2, so the block-sparse fidelity bound is 2e-2. A
  misaddressed scale moves the output by far more than either bound.
- Proxy routes add the represented token mass to the logit in the max pass
  (`log2(kv_block_size) / c` per proxy route; the ragged final summary shifts
  its score by `(log2(tail_mass) - log2(kv_block_size)) / c`, divided by the
  lane's group scale when Sage is on) and one route-uniform
  `log2(kv_block_size)` to the exp-pass addend, so the denominator weight of a
  proxy column is one and `p <= 1` holds for the 448 scaling. The 16-bit
  proxy profiles follow the same mean-summary contract.

## Limitations

- Only HND paged K/V is supported by the APIs above; contiguous BSHD K/V
  (dense and block-sparse, including Sage attention) goes through
  `BlockSparseTSWrapper`, and NHD caches are outside this API.
- Attention sinks and custom masks are not exposed.
- Q and K must share one dtype. Two mixed K/V combinations exist:
  `torch.bfloat16` Q/K with `torch.float8_e4m3fn` V on the dense KV128
  profiles, and the INT8 Sage recipe (INT8 Q/K with E4M3 V), which is only
  reachable through `BlockSparseTSWrapper` and `block_sparse_attention`.
- Effective K/V lengths must be positive and no greater than the static plan
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

Sage attention fidelity (random 8-bit inputs and scales against an FP32
reference that models the 448 P quantization) and recipe tests (the torch
reference quantizer in `tests/attention/sage_quant_reference.py` against BF16
attention) live in:

```bash
pytest -q tests/attention/test_attention_ts_sage.py
```
