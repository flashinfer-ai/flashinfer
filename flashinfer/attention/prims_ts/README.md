# Experimental Task-Scheduled Attention

`flashinfer.attention.prims_ts` exposes experimental CuTe DSL attention
kernels for NVIDIA Blackwell GPUs. Scheduling, tile selection, and split-KV
reduction are implementation details; the public interfaces expose attention
and cache semantics without tuning knobs.

Current accuracy and performance signoff is on SM100a/B200. SM103a/B300 is
admitted by the runtime architecture guard but is not yet signoff-qualified.
The QToken-KvBlock-Sparse-Attention extension is separately validated on SM103/GB300.

## Guides and public APIs

Import all entries below from `flashinfer.attention.prims_ts`.

| Kernel | Guide | Public APIs |
| --- | --- | --- |
| FMHA context/prefill | [Task-Scheduled FMHA Context](kernels/fmha_context/README.md) | `BatchPrefillTSWrapper`, `batch_prefill`, `BatchPrefillPagedTSWrapper`, `batch_prefill_with_paged_kv_cache` |
| FMHA decode | [Task-Scheduled FMHA Decode](kernels/fmha_decode/README.md) | `BatchDecodePagedTSWrapper`, `batch_decode_with_paged_kv_cache`, `get_prims_ts_batch_decode_workspace_size`, `prepare_prims_ts_batch_decode_with_kv_cache`, `prims_ts_batch_decode_with_kv_cache` |
| QToken-KvBlock-Sparse-Attention | [Packed-prefill and fixed-decode example](https://github.com/PerkzZheng/prims-ts-examples/blob/main/q_token_kv_block_sparse_attention.py) | `QTokenKvBlockSparsePagedTSWrapper`, `q_token_kv_block_sparse_attention_with_paged_kv_cache`, `get_q_token_kv_block_sparse_workspace_size`, `suggest_q_token_kv_block_sparse_group_size`, `validate_q_token_kv_block_sparse_group_size`, `make_q_token_kv_block_sparse_qo_indptr` |
| Block-sparse FMHA | — | `BlockSparseTSWrapper`, `block_sparse_attention`; fixed-Q paged KV: `BlockSparsePagedTSWrapper`, `block_sparse_attention_with_paged_kv_cache` |
| MLA decode | [Task-Scheduled MLA Decode](kernels/mla_decode/README.md) | `BatchMLADecodePagedTSWrapper`, `batch_mla_decode_with_paged_kv_cache`, `get_prims_ts_batch_mla_decode_workspace_size`, `prims_ts_batch_mla_decode_with_kv_cache` |

The component guides define supported shapes, layouts, metadata lifetime,
output/workspace ownership, examples, limitations, and validation commands.

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
`indexer_block_ids[total_q, block_topk]` contains logical K/V-block IDs;
it is deliberately not named `block_indices`, which belongs to BSR.
`block_table[num_requests, max_storage_pages]` maps each request's logical
storage pages to separate HND K/V caches shaped
`[num_pages, Hkv, page_size, D]`. `kv_block_size` is the semantic indexer
atom and currently supports only four tokens; `page_size` is the independent
physical cache-page extent.

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

`get_q_token_kv_block_sparse_workspace_size` sizes persistent compact route
metadata plus disjoint attention scratch. The model's per-request
`max_seq_len_kv`, rather than the global physical page-pool capacity, bounds
the plan. The private route builder forms at most
`G * (block_topk + 1)` selected/tail candidates, sorts and unique-reduces
them in one CTA, and ORs per-query membership bits. Its work is independent of
model context length. Q1 maps selected and tail blocks directly without a
membership table.

The production specialization is causal and non-windowed, uses KV128 for
Q1/Q2/Q4/Q5, and requires
`seq_len_q * (Hq / Hkv) <= TileQ64`. The pure-host
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
request metadata. Every `run` reads live paged-KV row offsets, physical page
IDs, per-request K/V lengths, per-KV-head sparse routes, and optional token
bits from device tensors. The physical-page ID tensor is capacity: its live
prefix ends at `paged_kv_indptr[-1]`, which may be smaller than its `numel()`.
The caller owns every live value contract: dense K/V lengths must be in
`[1, max_seq_len_kv]`, and causal lengths must be in `[Sq, max_seq_len_kv]`.
`paged_kv_indptr` must start at zero and contain bounded, monotone rows with at
least `ceil(seq_lens_kv[b] / page_size)` entries; every physical page ID in
the live prefix ending at `paged_kv_indptr[-1]` must lie in `[0, P)`. Every BSR
row must have bounded offsets, strictly increasing unique block IDs, and at
most the planned `max_blocks_per_row` entries. Contiguous IDs must lie below
`ceil(seq_len_kv / kv_block_size)`; paged IDs must start below the owning
request's live K/V length.

Reusable wrappers validate tensor structure but read values directly without
host synchronization. Invalid values therefore have undefined behavior and
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
are owned by `(batch, KV head, Q block)`, so all Q heads in one GQA/MQA group
share sparsity. A proxy run supplies one K arithmetic mean and one V sum per
semantic KV block. The final partial block uses only its structural tokens.
Optional `kv_valid_bits` filters exact K/V tokens only and does not change
proxy summaries or their represented mass.

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
  tests/attention/test_attention_ts_mla_decode.py
```
