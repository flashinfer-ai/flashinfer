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
| Block-sparse FMHA | [Sage attention](#sage-attention) below | `BlockSparseTSWrapper`, `block_sparse_attention`, `SageAttentionConfig`, `SageAttentionParams`; fixed-Q paged KV: `BlockSparsePagedTSWrapper`, `block_sparse_attention_with_paged_kv_cache` |
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
wrapper.run(...)   # live indexer IDs and request metadata; graph hot path
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
arithmetic mean per semantic KV block; the final partial block averages only
its structural tokens, and each proxy block counts as that many tokens in the
softmax.
Optional `kv_valid_bits` filters exact K/V tokens only and does not change
proxy summaries or their represented mass.

`BlockSparseTSWrapper.plan(..., use_block_sparse=False)` and
`block_sparse_attention(..., use_block_sparse=False)` run dense attention over
the whole contiguous K/V sequence with the same Q-tile selection. A dense plan
owns no route workspace, and its `run` takes Q/K/V only.

## Sage attention

Sage attention runs the contiguous decode kernel on 8-bit Q, K and V with
dequantization scales, in the dense and block-sparse modes of
`BlockSparseTSWrapper` and `block_sparse_attention`. The plan fixes the recipe
with `sage_config=SageAttentionConfig(...)`; every run supplies the scales as
`sage=SageAttentionParams(...)`:

```text
S[r][c] = sfQ[blkQ(r)] * sfK[blkK(c)] * (Q8 . K8^T)[r][c]
P       = softmax_row(S), quantized to E4M3 as 448 * p
O[r][d] = sfV[d] * (sum_c P8[r][c] * V8[c][d]) / (448 * l[r])  (+ v_mean[d])
```

FlashInfer does not quantize: the 8-bit tensors and scales come from the
caller, for example TensorRT-LLM's `sageQuant`. `v_mean` adds a per-channel
mean back after normalization, for callers that quantized `V - v_mean`. Every
scale must be finite and positive; the kernel does not check the values.

| Input | Supported values |
| --- | --- |
| Q/K dtype | Matching `torch.float8_e4m3fn`, or `torch.int8` on SM100a/B200 only |
| V dtype (`v_data_type`) | `torch.float8_e4m3fn`; it defaults to the K dtype, so INT8 plans set it |
| Output dtype | `torch.bfloat16` (default) or `torch.float16` |
| `q_block_size` | Power of two no larger than the Q tile; default 1 |
| `k_block_size` | 1, 4, 16, 32, 64, 128 or 256; default 16 |
| `k_summary_block_size` | K block size of proxy summary scales; default `k_block_size` |
| `v_mean` | Whether every run supplies `v_mean`; default `False` |
| Geometry | `head_dim=128`; a Q64/KV256 or Q128/KV128 Keeps tile, which needs a `kv_block_size` multiple of 64 and at least 64 grouped Q rows (`q_block_size * Hq / Hkv`) |

The block-size fields belong to `SageAttentionConfig`; the defaults are
TensorRT-LLM's production recipe. Masks and scheduling follow the 16-bit
plans. The paged block-sparse APIs do not support Sage attention.

### Scale tensors

All scales are contiguous, 16-byte-aligned `torch.float32` tensors on the run
device; a validating `run()` checks them against the plan. Q and K scales use
the trtllm-gen flat layout: within one head, sequence `b` of `[B, S, H, D]`
starts at slot `b * S // blk + b` and token `t` uses slot `t // blk` inside
it, so one head owns `flat_scale_numel(B, S, blk) == ceil(B * S / blk) + B - 1`
slots.

| Field | Shape |
| --- | --- |
| `q_scale` | `[Hq, flat_scale_numel(B, Sq, q_block_size)]` |
| `k_scale` | `[Hkv, flat_scale_numel(B, Skv, k_block_size)]` |
| `v_scale` | `[Hkv, D]` |
| `v_mean` | `[Hkv, D]`; exactly when the recipe sets `v_mean=True` |
| `k_summary_scale` | `[Hkv, flat_scale_numel(B, ceil(Skv / kv_block_size), k_summary_block_size)]`; exactly for proxy plans |

With proxy routes, `k_summary` holds the per-block K means in the K dtype,
quantized as one more K sequence of `ceil(Skv / kv_block_size)` tokens with
`k_summary_block_size`; `k_summary_scale` holds its scales. `v_summary` holds
the per-block V means in E4M3 and shares `v_scale` (built from `V - v_mean`
when a mean is used).

On B200 both recipes run roughly 15-20% faster than BF16 on the same plan.
K blocks of 4 and 1 tokens cost roughly 10-20% more than 16-token blocks.

### Example

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

# q_block_size=64 and kv_block_size=64 select the Q64/KV256 tile.
wrapper = BlockSparseTSWrapper()
wrapper.plan(
    B, Sq, Skv, Hq, Hkv, D, 64, 64,
    device=device,
    use_block_sparse=False,
    q_data_type=fp8,
    kv_data_type=fp8,
    sage_config=SageAttentionConfig(q_block_size=1, k_block_size=16),
)

# The quantizer (for example sageQuant) produces the tensors and scales.
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

A block-sparse INT8 plan with proxy routes names its E4M3 V, and its runs add
the routes, the summaries and `k_summary_scale`:

```python
from dataclasses import replace

wrapper.plan(
    B, Sq, Skv, Hq, Hkv, D, 64, 64,
    device=device,
    max_blocks_per_row=8,
    use_proxy_routes=True,
    q_data_type=torch.int8,
    kv_data_type=torch.int8,
    v_data_type=fp8,
    sage_config=SageAttentionConfig(),
)
# q, k and k_summary are int8; v and v_summary are fp8. The summaries are
# [B, num_kv_blocks, Hkv, D].
num_kv_blocks = -(-Skv // 64)
summary_scale = torch.rand(
    Hkv, flat_scale_numel(B, num_kv_blocks, 16), device=device
)
out = wrapper.run(
    q, k, v, block_indptr, block_indices,
    k_summary=k_summary,
    v_summary=v_summary,
    sage=replace(scales, k_summary_scale=summary_scale),
)
```

`block_sparse_attention(..., sage=...)` runs the same launches in one shot;
its recipe is `sage_config` or, when omitted, the default recipe with a V mean
exactly when `sage.v_mean` is set.

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
