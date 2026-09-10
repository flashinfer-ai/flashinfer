# Task-Scheduled FMHA Decode

This directory contains the CuTe DSL task-scheduled (TS) FMHA kernel used by
FlashInfer's experimental paged decode APIs on NVIDIA Blackwell GPUs. It
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
| `BatchDecodePagedTSWrapper` | Reusable static `plan()` with plan- or run-owned K/V lengths. |
| `batch_decode_with_paged_kv_cache` | One-shot convenience interface. |
| `get_prims_ts_batch_decode_workspace_size` | Size caller-owned scratch for the standalone launch. |
| `prims_ts_batch_decode_with_kv_cache` | Standalone launch with caller-owned scratch and explicit `seq_lens`. |

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

## Shared decode wrapper

`flashinfer.BatchDecodeWithPagedKVCacheWrapper(..., backend="prims-ts")`
adapts the shared CSR planning API to the native fixed-table interface.
Optional `seq_lens` accepts `uint32`, `int32`, or `int64` CPU/CUDA tensors;
planning copies validated lengths into owned int32 CUDA storage. Call `plan()`
again to change these lengths. An explicit `block_tables` must be an int32 or
uint32 CUDA tensor on the wrapper device, with unit inner stride and
non-overlapping rows. The adapter retains int32 tables directly and uses an
int32 view of uint32 tables, preserving storage and subsequent caller updates.
Active page IDs must fit in signed int32 and index the physical cache;
inactive entries are ignored. When omitted, the table is derived from the CSR
inputs during planning.

This backend requires `kv_layout="HND"` and does not support the shared
wrapper's `use_cuda_graph=True` replanning flow. Manual capture of `run()` is
supported after planning, but binds to that completed plan. Keep the wrapper
and captured tensors alive, and recapture after re-planning. Page IDs may
change between completed replays while the captured storage and layout stay
fixed; plan-owned sequence lengths may not.

## Supported contract

| Feature | Support |
| --- | --- |
| GPU | SM100a/B200 (qualified); SM103a/B300 (architecture-gated, not yet signoff-qualified) |
| Head dimension | 64, 128, or 256 |
| Fixed Q length | Any positive integer representable by the metadata and tensor extents |
| Packed Q | Positive per-request lengths no greater than a positive static maximum |
| Head mapping | MHA/GQA; `Hq` must be divisible by `Hkv`, with `1 <= Hq/Hkv <= 128`. Qualified fixed-Q FP8 D64/D128/D256 page-32 profiles use grouped Swaps Q8/Q16/Q32 through ratio 32, Keeps Q64 through ratio 64, and Keeps Q128 through ratio 128. |
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
the worker tasks. Underfilled fixed-Q grids may instead split the K/V sequence
and reduce partial outputs. Packed-Q and sliding-window work remains nonsplit:
it uses CLC above one resident wave and the direct static path otherwise.

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
| [`../../sage.py`](../../sage.py) | `SageAttentionParams`, flat-layout helpers and host-side scale validation |
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
metadata reads. Invalid run-owned lengths, page IDs, offsets, or aliases in that
mode may cause incorrect results or out-of-bounds access. Do not mutate
run-owned metadata concurrently with a launch or replay that reads it.

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
`BatchDecodeTSWrapper` with `sage=SageAttentionParams(...)`. The user-facing
contract (recipes, scale layouts, block-sparse summaries, examples) is in the
[PrimTS guide](../../README.md#sage-attention-8-bit-qkv-with-dequantization-scales);
this section records what the kernel does with the scales.

### Gating

`FmhaDecodeConfig.sage_k_block_size > 0` enables the feature
(`use_sage_attention`); `sage_q_block_size` and `sage_v_mean` are only legal
with it. `validate_sage_profile` admits one profile family: contiguous K/V,
`headdim == 128`, Q and K in `Float8E4M3FN` or `Int8` with `Float8E4M3FN` V
and 16-bit output, `sage_k_block_size` in `SAGE_K_BLOCK_SIZES`,
`sage_q_block_size` a power of two no larger than `tile_size_q`, a
two-instance Keeps profile (Q64/KV256 or Q128/KV128), and a
nonsplit grid (static direct or the persistent scheduler):
split-KV, the separate and cluster reductions, packed variable-length Q,
sliding windows and attention sinks are all rejected. `validate_dtypes` keeps Q == K for every recipe
(`kv_dtype` is the K dtype); `Int8` K additionally requires Sage scales and `Float8E4M3FN` V
(`v_dtype`, read through `value_dtype`; `None` follows `kv_dtype`).

Every Sage branch is a compile-time predicate on the configuration. With Sage
off, the 16-bit profiles produce byte-identical SASS to the kernel without
the feature; shared helpers may take Sage parameters only while that holds.

### Where the scales enter softmax

Each softmax lane owns one Q row and streams its 128 score columns as four
K32 fragments. Per tile the lane loads `sfQ[row]` once. Per fragment it loads
the K scales covering its 32 columns: with `k_block_size >= 32` one scale,
with `k_block_size == 16` two, giving `sage_k_groups_per_fragment`
compile-time groups (the fragment is unrolled, so group boundaries cost no
per-element work). With `c = sm_scale * log2(e)`:

```text
max pass:  gmax_g  = max(s in group g)              INT8: integer maximum, then one I2F per group
           fragmax = max_g(gmax_g * sfK_g)          one FMUL per group
           rowmax_true = rowmax * sfQ               after the last fragment
exp pass:  p = EX2(FFMA(s, c * sfQ * sfK_g, -c * rowmax_true + log2(448)))
           INT8: one I2F per element before the FFMA
```

The row sum, the correction factor `exp2(c * (m_old - m_new))`, the
correction skip and LSE all operate on the dequantized maximum and are
unchanged. `load_dequant_scales` in `sage_scales.py` returns the
`sfQ * sfK_g` multipliers for every fragment group of the tile; the softmax
bodies never see where the scales came from. The contiguous provider derives
each fragment's first token from the tile offset, the block-sparse provider
from the route's K64 atom origins; a proxy route switches the K source to
`k_summary_scale` indexed by summary position (`block_sparse_k_scale_source`).
Tokens beyond the sequence end and Q rows beyond the valid count clamp to the
last valid slot, so masked columns keep a finite scale and exponentiate to
zero. The contiguous provider issues the `sfK` loads from the softmax threads
before the score wait. The block-sparse provider moves them off the softmax
warps: the load warp resolves and stages the route's `sfK` words
(`sage_route_scale_words` per route, ordered by consuming half and lane
array) next to the route metadata, and each softmax thread reads its words
with contiguous vector loads (`dequant_scales_from_staged_k_scales`).

The epilogue (`_store_final_o_columns`) multiplies each output column by
`norm_scale * sfV[c]` and adds `v_mean[c]` when configured. Both vectors are
per KV head: the correction warps copy them into SMEM once per tile while no
output is pending (`stage_v_channel_scales`), and the epilogue reads its
column range back with vector loads (`load_staged_v_channel_scales`, which
returns a zero mean array without `sage_v_mean`, so the epilogue is one fused
multiply-add). This is exact because `sfV` is constant along the PV
reduction.

### INT32 scores

`Int8` Q/K use the `INT8` tcgen05 kind with an `Int32` accumulator
(`uses_int32_scores`). Over `D = 128` every dot product satisfies
`|s| <= 128 * 127^2 < 2^21`, so the score is exact; input quantization is
the only QK-side error. E4M3 Q/K use `F8F6F4` with FP32 accumulation. BMM2
is `F8F6F4` on E4M3 P and V in both recipes.

The softmax never converts INT32 scores. BMM1 accumulates onto the bit
pattern of `INT32_SCORE_BIAS = 1.5 * 2^23`: every integer in `[2^23, 2^24)`
is an FP32 number with unit spacing, so with `|s| < 2^21` the final
accumulator `bias + s` is exactly the FP32 number `12582912.0 + s`. The seed
is written by one `kind::f16` MMA step over constant BF16 operands
(`1.0 * 786432.0` summed over `K = 16`) issued right before the INT8 K steps
(`_seed_score_bias` in `tmem_s.py`); the tensor core writes the slot at
accumulator bandwidth and the INT8 steps accumulate onto it in tcgen05 issue
order. Seeding the 64 KB slot with `tcgen05.cp` or `tcgen05.st` instead
measured several times more expensive than the conversions it removes.

The max pass reduces biased scores with FP32 maxima and subtracts the bias
once per scale group (exact on the unit spacing; `-FLT_MAX` is unchanged by
it), so masked tiles share the FP32 path: `-FLT_MAX` on masked lanes, the
proxy tail shift applied in FP32. The P pass folds `-bias * c * sfQ * sfK_g`
into each group's exponent addend with one FMA per group, so the per-element
work is the same FFMA/EX2 stream as for FP32 scores. Two roundings differ
from converting every element, both below the INT8 quantization noise: the
per-group addend is rounded once, to at most half an ulp of
`bias * c * sfQ * sfK_g` (at most 0.75 quantized score units), and the proxy
tail shift lands on the biased score's unit spacing (at most 0.5 quantized
score units on that one lane). Block-sparse SOL shape, replay minimum,
against the per-element conversion: exact 484.7 -> 467.0 us, proxy
691.1 -> 594.9 us.

### Precision notes

- P is E4M3 with the static scale `FP8_P_QUANT_SCALE = 448`, which represents
  probabilities down to about 4.4e-6 with 3-bit relative precision. Row sums
  are accumulated from the quantized P so the tail truncation cancels in
  normalization.
- `defers_softmax_anchor_updates` is off for every 8-bit profile: the 448
  scale requires `p <= 1`, which a deferred anchor (lag up to `2^8`) would
  violate, so FP8 P always anchors on the exact row maximum and pays the
  TMEM rescale on every maximum increase. Trading a `448 / 2^k` anchor
  headroom against extra rescales is an open performance item.
- One quarter of the score pairs in a streamed fragment
  (`KV_TILE_256_EX2_EMULATED_PAIRS = 4` of 16) evaluate `exp2` as an FMA
  polynomial instead of MUFU. Its relative error moves about one to two
  percent of the E4M3 probabilities to the neighbouring quantization step.
  Dense rows of about a thousand tokens average this out (the dense fidelity
  tests hold 2e-3), while a sparse row that attends to a few hundred tokens
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
  `BatchDecodeTSWrapper`, and NHD caches are outside this API.
- Attention sinks and custom masks are not exposed.
- Q, K, and V cannot use mixed dtypes; the one exception is the INT8 Sage
  recipe (INT8 Q/K with E4M3 V), which is only reachable through
  `BatchDecodeTSWrapper`.
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
