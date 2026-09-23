# Experimental Task-Scheduled Attention

`flashinfer.attention.prims_ts` exposes experimental CuTe DSL attention
kernels for NVIDIA Blackwell GPUs. Scheduling, tile selection, and split-KV
reduction are implementation details; the public interfaces expose attention
and cache semantics without tuning knobs.

Public entry points marked with `@flashinfer_experimental_api` warn once on
first use and provide no compatibility guarantee. Calling an API is the opt-in;
no environment variable is required. Existing API logging and `fi_trace`
bindings remain available.

Current accuracy and performance signoff is on SM100a/B200. SM103a/B300 is
admitted by the runtime architecture guard but is not yet signoff-qualified.
The QToken-KvBlock-Sparse-Attention extension is separately validated on SM103/GB300.

## Guides and public APIs

Import all entries below from `flashinfer.attention.prims_ts`.

| Kernel | Guide | Public APIs |
| --- | --- | --- |
| FMHA context/prefill | [Task-Scheduled FMHA Context](kernels/fmha_context/README.md) | `BatchPrefillTSWrapper`, `batch_prefill`, `BatchPrefillPagedTSWrapper`, `batch_prefill_with_paged_kv_cache` |
| FMHA decode | [Task-Scheduled FMHA Decode](kernels/fmha_decode/README.md) | `BatchDecodePagedTSWrapper`, `batch_decode_with_paged_kv_cache`, `get_prims_ts_batch_decode_workspace_size`, `prepare_prims_ts_batch_decode_with_kv_cache` |
| QToken-KvBlock-Sparse-Attention | [Packed-prefill and fixed-decode example](https://github.com/PerkzZheng/prims-ts-examples/blob/main/q_token_kv_block_sparse_attention.py) | `QTokenKvBlockSparsePagedTSWrapper`, `q_token_kv_block_sparse_attention_with_paged_kv_cache`, `get_q_token_kv_block_sparse_workspace_size`, `suggest_q_token_kv_block_sparse_group_size`, `validate_q_token_kv_block_sparse_group_size`, `make_q_token_kv_block_sparse_qo_indptr` |
| Block-sparse FMHA | [Sage attention](#sage-attention-8-bit-qkv-with-dequantization-scales) below; kernel notes in [Task-Scheduled FMHA Decode](kernels/fmha_decode/README.md#sage-attention) | `BlockSparseTSWrapper`, `block_sparse_attention`; `plan(use_block_sparse=False)` for dense contiguous K/V; `SageAttentionConfig` and `SageAttentionParams` for 8-bit Q/K/V with scales; fixed-Q paged KV: `BlockSparsePagedTSWrapper`, `block_sparse_attention_with_paged_kv_cache` |
| MLA decode | [Task-Scheduled MLA Decode](kernels/mla_decode/README.md) | `BatchMLADecodePagedTSWrapper`, `batch_mla_decode_with_paged_kv_cache`, `get_prims_ts_batch_mla_decode_workspace_size` |

The component guides define supported shapes, layouts, metadata lifetime,
output/workspace ownership, examples, limitations, and validation commands.

### Unified decode entry points

`batch_decode_with_paged_kv_cache` and `batch_mla_decode_with_paged_kv_cache`
support both convenience and explicit caller-workspace execution. Existing
calls retain their validated convenience behavior, including FMHA's
length-specialized policies. For allocation-free, synchronization-free
steady-state launches, provide `workspace_buffer`, `max_kv_len`, `out`,
and `validate=False`. Packed Q additionally requires `max_seq_len_q`
(FMHA also accepts its non-default `seq_len_q` alias).
Both APIs size and allocate scratch only when it is omitted, then pass it to
the same wrapper plan/run path. Trusted calls select the existing cached
kernel from static Python/tensor attributes without allocating or reading
metadata values back to the host. FMHA preserves initialized caller scratch
by planning with `initialize_workspace=False`.

Use the existing `get_prims_ts_batch_*_workspace_size` helpers outside
capture. FMHA scratch must be zero-initialized and re-zeroed when any workspace
layout input (including batch size) changes. Scratch and output must not
overlap any live input or each other; each concurrent launch/graph needs its
own scratch. Warm up the exact topology before capture, retain stable storage,
and update metadata only between completed replays. Validation is enabled by
default and reads metadata values on the host; `validate=False` transfers
all shape, dtype, stride, capacity, active page-ID, packed-offset, and lifetime
obligations to the caller. The required decode control resets are unchanged.

For example, given validated FMHA Q/cache/metadata and a correctly sized,
zero-initialized `workspace`:

```python
import torch
from flashinfer.attention.prims_ts import batch_decode_with_paged_kv_cache

def decode():
    return batch_decode_with_paged_kv_cache(
        q, paged_kv_cache, block_tables, seq_lens,
        workspace_buffer=workspace, max_kv_len=max_kv_len,
        out=out, validate=False,
    )

decode()  # Warm up outside capture.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    decode()
graph.replay()
```

The contiguous and paged context, FMHA decode, and MLA decode wrappers separate
reusable static state from per-run request state. `plan()` compiles a static
capacity, shape, dtype, and storage-mode specialization without retaining
request tensors or metadata. FMHA decode can optionally own a fixed sequence-length
vector for the plan's lifetime. Paged context plans may additionally freeze
explicit exact-uniform-length, zero-causal-offset, or zeroed-V-tail promises.
Context `run()` receives current packed offsets or fixed-table paged metadata;
per-token variable-window bounds for fixed-shape inputs are also per-run, with
optional caller-precomputed per-CTA start minima. Both context wrappers own
their default scale tensors. Contiguous variable-window plans additionally own
mutable fallback scratch that derives CTA minima only when the caller omits
them, while paged context owns no other workspace. With `validate=False`,
supplying CTA minima avoids that repeated preprocessing and leaves all
variable-window metadata caller-owned. Runtime validation is enabled by
default; callers that have already validated their inputs may use
`validate=False` for steady-state timing or CUDA Graph capture and then own
every dtype, device, shape, stride, alignment, value, aliasing, and lifetime
obligation.

All PrimTS APIs leave cross-tensor storage overlap unchecked, regardless of
`validate`. Output must not overlap inputs, live metadata, or plan-owned
buffers. Caller-provided workspace must be disjoint from the public input and
output tensors; its internal views retain their documented layout. Callers
must preserve these preconditions when rebinding tensors and replaying graphs.
Writable-buffer aliasing is unsupported, not an in-place execution mode.
TensorMap stride/alignment and workspace-capacity checks remain in place.

The standalone sparse-attention example suggests G for both packed prefill
and fixed decode using a caller-cached SM count, then fixes G for each plan.

## QToken-KvBlock-Sparse-Attention interface

QToken-KvBlock-Sparse-Attention consumes per-query indexer output directly.
`indexer_block_ids` contains logical K/V-block IDs: `[total_q, block_topk]`
when `share_pattern_across_kv_heads=True` (default), otherwise
`[total_q, Hkv, block_topk]`. It is deliberately not named `block_indices`,
which belongs to BSR. The last dimension is contiguous.
`block_table[num_requests, max_storage_pages]` maps each request's logical
storage pages to separate HND K/V caches shaped
`[num_pages, Hkv, page_size, D]`. `kv_block_size` is the semantic indexer
atom, supporting 4/8/16/32/64/128 tokens; `page_size` is the independent
physical cache-page extent and must be a positive multiple of four. Semantic
blocks may cross physical pages. The loader resolves each fragment of size
`gcd(kv_block_size, page_size)` through the dense table, without assuming
physical adjacency. The KV-cache writer must zero-fill unused page padding.

The public lifecycle matches paged block-sparse attention:

```python
wrapper = QTokenKvBlockSparsePagedTSWrapper()
wrapper.plan(...)  # geometry, capacity, dtypes, workspace; outside capture
wrapper.run(...)  # live indexer IDs and request metadata; graph hot path
```

`plan` binds the single caller-owned byte workspace and fixes
`batch_size`, `seq_len_q`, head geometry, `block_topk`,
`max_seq_len_kv`, dtypes, and packed-versus-fixed Q layout. The first eager
`run` binds the live tensor ABI, compiles the selected kernels, and
initializes split-KV state. Capture only later `run` calls. One wrapper
revision owns mutable route and split-KV scratch, so unordered concurrent runs
need distinct wrappers.

Packed prefill uses `q[total_q, Hq, D]`, `qo_indptr`, and
`use_packed_q=True`; planned `seq_len_q` is the maximum request-safe route
length. Fixed MTP decode uses `q[B, Nq, G, Hq, D]` without `qo_indptr`,
where planned `batch_size = B * Nq` and `seq_len_q = G`. Q lengths need not
be divisible by a suggested group: packed mode permits a shorter final route,
while fixed mode may append consecutive semantic dummy rows and discard their
outputs.

`split_kv=False` selects nonsplit execution and omits reduction scratch;
`split_kv=True` permits useful automatic splitting. The caller chooses the
phase policy: typically False for prefill and True for decode. This choice is
independent of packed/fixed Q and must match workspace sizing and planning.

`get_q_token_kv_block_sparse_workspace_size` sizes persistent compact route
metadata plus disjoint attention scratch. The model's per-request
`max_seq_len_kv`, rather than the global physical page-pool capacity, bounds
the plan. Each group and pattern head forms at most
`G * (block_topk + 1)` selected/tail candidates, sorts and unique-reduces
them in one CTA, and ORs per-query membership bits. It processes only candidate
IDs, with no full-context bitmap or scan. Shared patterns prepare one row per
group; independent patterns prepare one per group and KV head. G1 resolves
selected blocks and the causal tail inside attention, without a metadata launch.

Nonsplit sparse grids larger than one service wave use the common CLC
persistent scheduler. Each work item resolves its own request, query group and
KV-head metadata. Page producers stage locators with `cp.async`; grouped
membership storage is retained until all softmax consumers finish the item.
The same paths support fixed and packed Q, including partial and empty packed
groups. `split_kv=False` disables splitting, not persistent scheduling.

The production specialization supports D64/D128/D256, matching FP16/BF16
Q/K/V/output, or FP8 E4M3 Q/K/V with FP16/BF16 output. It is causal and
non-windowed, uses KV128 for Q1--Q8, and requires
`seq_len_q * (Hq / Hkv) <= TileQ128`. Groups exceeding 64 token/head rows
use TileQ128; per-fragment query membership still occupies one byte.
The pure-host
`suggest_q_token_kv_block_sparse_group_size` helper takes a caller-cached SM
count and prefers the largest legal group that can fill one service wave using
independent routes plus useful split-KV work. It falls back toward Q1 when a
large union would leave SMs idle. The helper never queries device properties
or reads tensors.

`q_token_kv_block_sparse_attention_with_paged_kv_cache` is the eager
plan-plus-run convenience API and is not graph-capturable. Shared argument
spellings intentionally match paged block-sparse attention: `q`,
`paged_kv_cache`, `kv_block_size`, `mask_type`,
`sm_scale`, `seq_len_q`, and `max_seq_len_kv`. The optional runtime
`v_scale` applies the value-cache dequantization scale and defaults to one.
The convenience API infers physical page size from `paged_kv_cache`; only
the prepared wrapper's `plan` takes `page_size` explicitly.

On SM90 and newer, the combined route-builder and attention launch use PDL.
Attention initializes its independent resources before acquiring immediately
ahead of the first metadata-dependent read. Split-KV attention releases its
reducer only after producer completion and TMEM teardown; the reducer
initializes local resources before all threads acquire. Older architectures
retain stream ordering.

For `BlockSparsePagedTSWrapper`, `plan` freezes only the compact fixed-Q
geometry, dtypes, sparse-route capacity, and `max_seq_len_kv`; it retains no
request metadata. Every `run` reads live page tables, per-request K/V lengths,
sparse routes, and optional token bits from device tensors.
`block_tables` is Int32 `[B, C]`, contiguous within each row and free to use a
padded outer row stride; `C * page_size` must cover `max_seq_len_kv`, and only
the first `ceil(seq_lens_kv[b] / page_size)` entries of each row are read. The
caller owns every live value contract: dense K/V lengths must be in
`[1, max_seq_len_kv]`, causal lengths must be in `[Sq, max_seq_len_kv]`, and
every live physical page ID must lie in `[0, P)`. Every BSR row must have
bounded offsets, strictly increasing unique block IDs, and at most the planned
`max_blocks_per_row` entries. Contiguous IDs must lie below
`ceil(seq_len_kv / kv_block_size)`; paged IDs must start below the owning
request's live K/V length.

Reusable wrappers validate tensor structure by default and read values
directly without host synchronization; `validate=False` skips the
structural checks as well. Invalid values therefore have undefined behavior and
may access out of bounds. Set `CUTE_DSL_ENABLE_ASSERTIONS=1` before the process
first compiles these kernels to diagnose violations encountered while preparing
selected routes; such assertions report asynchronously and leave the CUDA
context unusable. The one-shot APIs instead synchronize once to validate all
live values, including the complete physical-page-ID prefix, before creating
their temporary plans and cannot run during CUDA Graph capture.

The one-shot `block_sparse_attention_with_paged_kv_cache` API takes
`max_seq_len_kv` as the static capacity and requires `seq_lens_kv` with the
live per-request logical lengths. This paged block-sparse API does not support packed or
mixed/variable Q lengths.
Eager launches retain all launch tensors on the run stream; CUDA Graph users
must keep the wrapper and Q/cache/output/runtime-metadata tensors alive and
unmodified until replay completes. Values may change between completed replays
while tensor addresses, shapes, dtypes, and strides remain stable.

For the separate block-sparse FMHA API, qualified Q64/coarse-KV profiles retain
KV256 routes for page sizes 64 and 128. Optional `kv_valid_bits` is a
`torch.uint32` per-request bitset with shape
`[B, ceil(max_seq_len_kv / 32)]` over logical KV tokens; it is shared by all KV
heads and independent of the physical page mapping.

For contiguous block-sparse attention, both `BlockSparseTSWrapper.plan` and
the `block_sparse_attention` one-shot API can opt into
`sparse_format="bitmask"` and/or `use_proxy_routes=True`. BSR and packed
exact-block bitmaps are alternative frontends; both are prepared into the same
route stream before attention. The bitmask one-shot uses the full structural
KV-block count as its temporary plan capacity, while reusable plans accept a
tighter caller-provided bound. Proxy routes are supported across the existing
contiguous block-sparse profiles and preserve the profile's Q tile, KV route,
and KeepsAB/SWAPAB geometry. Proxy routes currently require
`mask_type="dense"`; paged K/V proxy execution remains unsupported. Route rows
are owned by `(batch, pattern head, Q block)`. Both block-sparse APIs default
to `share_pattern_across_kv_heads=False`; True uses a singleton pattern-head
axis shared by all KV heads. K/V and proxy summaries retain their physical
head axis in either mode. A proxy run supplies one K arithmetic mean and one V
arithmetic mean per semantic KV block; each proxy block then counts as its
number of structural tokens in the softmax. V summaries must be per-block
means; the kernel cannot tell a per-block sum apart and would scale the output
by the block mass. The final partial block averages only its structural tokens.
Optional `kv_valid_bits` filters exact K/V tokens only and does not change
proxy summaries or their represented mass. With Sage attention
the K summaries use the K dtype and are dequantized with
`SageAttentionParams.k_summary_scale`; the V summaries are E4M3 and share
`v_scale` (see below).

## Sage attention (8-bit Q/K/V with dequantization scales)

`BlockSparseTSWrapper.plan(..., sage_config=SageAttentionConfig(...))` compiles
the
contiguous decode kernel for 8-bit Q, K and V in both modes (dense and
block-sparse, exact and proxy routes), and every
`run(..., sage=SageAttentionParams(...))` supplies the scale tensors of that
call. The math is the one implemented by trtllm-gen and consumed by
TensorRT-LLM's `sageQuant`-based pipeline:

```text
S[r][c] = sfQ[blkQ(r)] * sfK[blkK(c)] * (Q8 . K8^T)[r][c]      INT8 -> INT32 scores, E4M3 -> FP32 scores
P       = softmax_row(S)                                        FP32; quantized to E4M3 as 448 * p
O[r][d] = sfV[d] * (sum_c P8[r][c] * V8[c][d]) / (448 * l[r])  (+ v_mean[d] when supplied)
```

The Q/K scales apply to the scores before the row maximum and the
exponential, never to P; P has one static scale (448, the E4M3 maximum); V
has one scale per channel. There is no K or V smoothing inside the kernel;
`v_mean` only adds a caller-provided per-channel mean back after
normalization, for callers that quantized `V - v_mean`. FlashInfer does not
quantize: the 8-bit tensors and every scale come from TensorRT-LLM's
`sageQuant` (or from the test-only torch reference in
`tests/attention/sage_quant_reference.py`). Every scale must be finite and
positive; the kernel does not check the values.

### Recipes

| Input | Accepted values |
| --- | --- |
| Q/K dtype (`q_data_type`, `kv_data_type`) | `torch.float8_e4m3fn` or `torch.int8`; Q and K must match |
| V dtype (`v_data_type`) | `torch.float8_e4m3fn`; defaults to `kv_data_type`, so INT8 K must name it explicitly |
| Output dtype (`o_data_type`) | `torch.bfloat16` (default with `sage`) or `torch.float16` |
| `SageAttentionConfig.q_block_size` | Power of two no larger than the Q tile (64 or 128, see below); default 1 |
| `SageAttentionConfig.k_block_size` | One of `SAGE_K_BLOCK_SIZES == (1, 4, 16, 32, 64, 128, 256)`; default 16 |
| `SageAttentionConfig.k_summary_block_size` | K block size of a proxy plan's summary scales, one of `SAGE_K_BLOCK_SIZES`; default `None` = `k_block_size`. A plan without proxy routes uses `k_block_size` |
| `SageAttentionConfig.v_mean` | `True` when every run supplies `v_mean`; default `False` |
| Head dimension | 128 |
| K/V storage | Contiguous `[B, S, Hkv, D]` only; the paged wrappers accept neither `sage` nor INT8 |
| Profiles | Streamed Keeps Q64/KV256 and Q128/KV128 (dense and block-sparse) |
| Mask | `dense` or `causal` (proxy routes stay `dense`) |
| Scheduling | Static direct grid or the persistent scheduler, selected as for 16-bit plans in both modes; no split-KV, sliding window or attention sinks |

`kv_data_type` is the K dtype; the INT8 recipe (`int8` Q/K, `float8_e4m3fn`
V) names `v_data_type` explicitly, and `(int8, int8)` and
`(float8_e4m3fn, int8)` are rejected. The defaults
(`q_block_size=1`, `k_block_size=16`) are TensorRT-LLM's production recipe
`(1, 16, 1)`; its `(1, 4, 1)` and `(1, 1, 1)` recipes run with eight and 32
scale groups per K32 score fragment, whose `sfK` words the softmax reads from
SMEM per fragment instead of holding one multiplier per group in registers.

The profile follows from the same `q_block_size`/`kv_block_size` plan
arguments as 16-bit decode: `kv_block_size` must be a positive multiple of
64, and the Q tile is the largest power of two no larger than
`min(q_block_size * Hq / Hkv, 128)` that fits the block geometry. A Q64 tile
selects Q64/KV256, a Q128 tile selects Q128/KV128; smaller tiles are Swaps
profiles and reject `sage`, as does a `kv_block_size` that is not a multiple
of 64. Every 8-bit Q128/KV128 plan streams its P fragments, as every KV256
and every block-sparse plan does; the streamed fragment passes are where the
scales enter.

### Scale tensors

All scale tensors are contiguous, 16-byte aligned `torch.float32` on the run
device. Q and K scales use the trtllm-gen flat layout: within one head,
sequence `b` of a fixed-shape `[B, S, H, D]` tensor starts at slot
`(b * S) // blk + b` and token `t` uses slot `t // blk` inside it, so one head
owns `flat_scale_numel(B, S, blk) == ceil(B * S / blk) + B - 1` slots. This is
`sageQuant`'s `cumSeqLens[b] / blk + b + t / blk` with `cumSeqLens[b] = b * S`;
the extra `b` keeps the last block of one sequence and the first block of the
next in distinct slots when `blk` does not divide `S`. `flat_scale_slot`
(with `log2_block_size`) and `flat_scale_numel` in
`flashinfer.attention.prims_ts.sage` compute both; the kernel indexes the
layout with the same `flat_scale_slot`.

| Field | Shape | Meaning |
| --- | --- | --- |
| `q_scale` | `[Hq, flat_scale_numel(B, Sq, q_block_size)]` | One scale per Q head and Q token block |
| `k_scale` | `[Hkv, flat_scale_numel(B, Skv, k_block_size)]` | One scale per KV head and K token block |
| `v_scale` | `[Hkv, D]` | One scale per KV head and channel, shared across the batch |
| `v_mean` (optional) | `[Hkv, D]` | Per-channel mean added back after normalization |
| `k_summary_scale` | `[Hkv, flat_scale_numel(B, ceil(Skv / kv_block_size), summary_k_block_size)]` | Required by, and only by, block-sparse proxy plans; `summary_k_block_size` is `k_summary_block_size` or, unset, `k_block_size` |

The recipe (`SageAttentionConfig`) is compile-time and joins the kernel cache
key; the scale tensors are run inputs, validated for shape, dtype, device and
contiguity against the plan on every `run()`, and `v_mean` must be present
exactly when the plan was configured with `v_mean=True`. Sequence tails and
masked columns beyond the valid length read the last valid slot, so a scale
array never needs padding beyond `flat_scale_numel`.

### Block-sparse contract with Sage

Proxy routes treat the K summaries as one more K sequence of
`ceil(Skv / kv_block_size)` tokens: `k_summary` holds the per-block K means
quantized in the K dtype (INT8 or E4M3) with `sageQuant` applied to the
summary tensor as if it were K with the recipe's summary K block size
(`k_summary_block_size`, defaulting to `k_block_size`), and
`k_summary_scale` is the resulting flat-layout array. Summaries are block
means whose magnitudes spread more than the tokens', so an INT8 recipe can
take one scale per summary (`k_summary_block_size=1`) while its exact routes
keep the 16-token block; the kernel then reads the two route kinds' scales
with their own group geometry and strategy (register array for blocks of 16
and larger, an SMEM ring for 4 and 1), selected per tile on the route kind.
A route kind with one scale per score (the one-token K block) has its
scores written back dequantized by the max pass, so its P pass avoids
reloading and applying all 32 `sfK` words of each fragment.
A mixed geometry costs a few percent on proxy plans: the proxy kernel carries
both geometries' max passes and gathers the summary scales in its softmax
warps; exact-only plans are unaffected. `v_summary` holds the per-block V means (the final partial
block averages only its structural tokens) quantized to E4M3 with the shared
`v_scale`; with `v_mean`, build them from `V - v_mean`. A proxy block stands
for as many identical tokens as it covers, so its mass enters the proxy
logit and its probability carries the block's weight. Token masks
(`kv_valid_bits`) and ragged final blocks are supported exactly as without
Sage.

### Performance

The Sage kernels keep the warp roles, pipelines and scheduler of the 16-bit
decode kernels. The 8-bit tensors halve the K/V traffic, and the byte-wide
tiles free shared memory that the kernels reinvest in a deeper K/V ring and a
lighter output tail. The softmax warps stay the bottleneck, so the scale
machinery is kept off their critical path: exact routes' K scales travel with
the route metadata, dense tiles read theirs ahead of the score wait, one
multiplier serves each compile-time scale group, INT8 scores accumulate on an
FP32 bias so they need no conversion, and the one-token K block writes its
scores back dequantized so its P pass reads no scales.

Rough figures on B200, kernel time relative to the BF16 kernel of the same
plan under the same launch heuristic: the FP8 and INT8 recipes run the dense
and block-sparse decode shapes (SOL and VSA geometries, long and short) about
15-20% faster; a small dense shape that a persistent BF16 grid already
saturates gains nothing; INT8 is within a few percent of FP8; the 4- and
1-token K blocks cost 10-20% over the 16-token block; the results do not
depend on the logit distribution.

### Examples

Dense contiguous decode, E4M3 Q/K/V with the production `(1, 16, 1)` recipe:

```python
import torch
from flashinfer.attention.prims_ts import (
    BlockSparseTSWrapper,
    SageAttentionConfig,
    SageAttentionParams,
)
from flashinfer.attention.prims_ts.sage import flat_scale_numel

device = torch.device("cuda")
B, Sq, Skv, Hq, Hkv, D = 2, 128, 1000, 4, 4, 128
fp8 = torch.float8_e4m3fn

wrapper = BlockSparseTSWrapper()
wrapper.plan(
    B,
    Sq,
    Skv,
    Hq,
    Hkv,
    D,
    64,  # q_block_size: 64 rows per KV head select the Q64/KV256 profile
    64,  # kv_block_size: a multiple of 64
    device=device,
    use_block_sparse=False,
    q_data_type=fp8,
    kv_data_type=fp8,
    o_data_type=torch.bfloat16,
    sage_config=SageAttentionConfig(q_block_size=1, k_block_size=16),
)

# 8-bit tensors and scales come from the quantizer (TensorRT-LLM sageQuant).
q = torch.randn(B, Sq, Hq, D, device=device).to(fp8)
k = torch.randn(B, Skv, Hkv, D, device=device).to(fp8)
v = torch.randn(B, Skv, Hkv, D, device=device).to(fp8)
scales = SageAttentionParams(
    q_scale=torch.rand(Hq, flat_scale_numel(B, Sq, 1), device=device),
    k_scale=torch.rand(Hkv, flat_scale_numel(B, Skv, 16), device=device),
    v_scale=torch.rand(Hkv, D, device=device),
)
out = wrapper.run(q, k, v, sage=scales)  # [B, Sq, Hq, D] bfloat16
```

Block-sparse decode with proxy routes, INT8 Q/K and E4M3 V:

```python
import math

kv_block_size, max_blocks_per_row = 64, 8
num_kv_blocks = math.ceil(Skv / kv_block_size)

wrapper = BlockSparseTSWrapper()
wrapper.plan(
    B,
    Sq,
    Skv,
    Hq,
    Hkv,
    D,
    64,
    kv_block_size,
    device=device,
    max_blocks_per_row=max_blocks_per_row,
    use_kv_valid_bits=False,
    use_proxy_routes=True,
    q_data_type=torch.int8,
    kv_data_type=torch.int8,
    v_data_type=fp8,  # INT8 K requires E4M3 V, named explicitly
    o_data_type=torch.bfloat16,
    sage_config=SageAttentionConfig(),  # the (1, 16, 1) recipe
)

q = (
    (torch.randn(B, Sq, Hq, D, device=device) * 40)
    .round()
    .clamp(-127, 127)
    .to(torch.int8)
)
k = (
    (torch.randn(B, Skv, Hkv, D, device=device) * 40)
    .round()
    .clamp(-127, 127)
    .to(torch.int8)
)
v = torch.randn(B, Skv, Hkv, D, device=device).to(fp8)
scales = SageAttentionParams(
    q_scale=torch.rand(Hq, flat_scale_numel(B, Sq, 1), device=device),
    k_scale=torch.rand(Hkv, flat_scale_numel(B, Skv, 16), device=device),
    v_scale=torch.rand(Hkv, D, device=device),
    # The K summaries are quantized as their own sequence of num_kv_blocks tokens.
    k_summary_scale=torch.rand(
        Hkv, flat_scale_numel(B, num_kv_blocks, 16), device=device
    ),
)
# block_indptr [B, Hkv, ceil(Sq / q_block_size) + 1] and block_indices select
# the exact blocks; k_summary (int8) and v_summary (fp8) are the per-block
# means of the remaining blocks, [B, num_kv_blocks, Hkv, D].
out = wrapper.run(
    q,
    k,
    v,
    block_indptr,
    block_indices,
    k_summary=k_summary,
    v_summary=v_summary,
    sage=scales,
)
```

`block_sparse_attention(..., sage=SageAttentionParams(...))` runs the same
launch in one shot, in the dense mode and in the block-sparse modes; the recipe
is `sage_config=SageAttentionConfig(...)`, or the default recipe with a V mean
exactly when `sage.v_mean` is present. The INT8 recipe is inferred from the
dtypes of `q`, `k` and `v`. The paged wrappers do not take `sage`.

`BlockSparseTSWrapper.plan(..., use_block_sparse=False)` plans dense attention
over the whole contiguous K/V sequence with the same Q-tile and KV-route
selection and the same profile matrix as a block-sparse plan, and
`block_sparse_attention(..., use_block_sparse=False)` runs it in one shot. A
dense plan owns no route workspace, launches no route preparation, and its
`run` takes Q/K/V only; its scheduler follows the decode kernel's launch
heuristic, which picks the persistent scheduler once the static grid exceeds
one resident wave because the work-tile loop overlaps a tile's epilogue with
the next tile's QK head.

## Validation

Run the numerical, graph, scheduler/resource, and public-surface
contracts:

```bash
pytest -q \
  tests/attention/test_attention_ts_context.py \
  tests/attention/test_attention_ts_decode.py \
  tests/attention/test_attention_ts_q_token_kv_block_sparse_metadata.py \
  tests/attention/test_attention_ts_block_sparse.py \
  tests/attention/test_attention_ts_mask.py \
  tests/attention/test_attention_ts_mla_decode.py \
  tests/attention/test_attention_ts_sage.py
```
