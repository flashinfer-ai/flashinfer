# Experimental Task-Scheduled Attention

`flashinfer.attention.prims_ts` exposes experimental CuTe DSL attention
kernels for NVIDIA Blackwell GPUs. Scheduling, tile selection, and split-KV
reduction are implementation details; the public interfaces expose attention
and cache semantics without tuning knobs.

The task-scheduled kernels support CuTe DSL 4.7 and 4.8. Internal task calls
adapt to the installed API's resource-context signature.

Public entry points marked with `@flashinfer_experimental_api` warn once on
first use and provide no compatibility guarantee. Calling an API is the opt-in;
no environment variable is required. Existing API logging and `fi_trace`
bindings remain available.

Hardware eligibility and the exercised software/hardware matrix are listed in
[Validation scope](#validation-scope).

## Compilation cache

Attention adapters use FlashInfer's persistent CuTe DSL object cache across
processes, including context, paged decode, MLA, sparse/Sage preparation, and
sparse metadata inspection. Specializations include compile options and static
kernel configuration; device indices are excluded so ranks on equivalent GPUs
can share artifacts. Physical GPU capability, compile target, installed DSL
stack, and attention source changes invalidate the cache. Device-specific
occupancy probes run once per device and cluster size in each process.

By default, object files and `meta.json` are stored under
`~/.cache/flashinfer/<flashinfer-version>/<cuda-archs>/cached_ops/prims_ts_<family>_<compile-arch>_cute_dsl/`.
For example, FlashInfer 0.7.2 on a GB300 uses
`~/.cache/flashinfer/0.7.2/103a/cached_ops/prims_ts_decode_sm103a_cute_dsl/`
for decode. Inspect the resolved `cached_ops` directory for the current process:

```bash
python -c 'from flashinfer.jit import env; print(env.FLASHINFER_JIT_DIR)'
```

Setting `FLASHINFER_WORKSPACE_BASE=/path` places this cache under
`/path/.cache/flashinfer/`. Set `FLASHINFER_CUTE_DSL_DISABLE_CACHE=1` to compile
without disk persistence.
Warm up plans before CUDA graph capture as usual.

## Guides and public APIs

Import all entries below from `flashinfer.attention.prims_ts`.

| Kernel | Guide | Public APIs |
| --- | --- | --- |
| FMHA context/prefill | [Task-Scheduled FMHA Context](kernels/fmha_context/README.md), [VC-Attention](#vc-attention) below; Sage plans: [Sage attention](#sage-attention) | `BatchPrefillTSWrapper`, `batch_prefill`, `VCAttentionConfig`, `VCAttentionParams`, `BatchPrefillPagedTSWrapper`, `batch_prefill_with_paged_kv_cache` |
| FMHA decode | [Task-Scheduled FMHA Decode](kernels/fmha_decode/README.md) | `BatchDecodePagedTSWrapper`, `batch_decode_with_paged_kv_cache`, `get_prims_ts_batch_decode_workspace_size`, `prepare_prims_ts_batch_decode_with_kv_cache` |
| QToken-KvBlock-Sparse-Attention | [Packed-prefill and fixed-decode example](https://github.com/PerkzZheng/prims-ts-examples/blob/main/q_token_kv_block_sparse_attention.py) | `QTokenKvBlockSparsePagedTSWrapper`, `q_token_kv_block_sparse_attention_with_paged_kv_cache`, `get_q_token_kv_block_sparse_workspace_size`, `suggest_q_token_kv_block_sparse_group_size`, `validate_q_token_kv_block_sparse_group_size`, `make_q_token_kv_block_sparse_qo_indptr` |
| Block-sparse FMHA | [Sage attention](#sage-attention) below | `BlockSparseTSWrapper`, `block_sparse_attention`, `SageAttentionConfig`, `SageAttentionParams`; fixed-Q paged KV: `BlockSparsePagedTSWrapper`, `block_sparse_attention_with_paged_kv_cache` |
| MLA decode | [Task-Scheduled MLA Decode](kernels/mla_decode/README.md) | `BatchMLADecodePagedTSWrapper`, `batch_mla_decode_with_paged_kv_cache`, `get_prims_ts_batch_mla_decode_workspace_size` |

The component guides define supported shapes, layouts, metadata lifetime,
output/workspace ownership, examples, limitations, and validation commands.

### Unified decode entry points

`batch_decode_with_paged_kv_cache` and `batch_mla_decode_with_paged_kv_cache`
support both convenience and explicit caller-workspace execution. Convenience calls validate inputs and allocate scratch when omitted. For allocation-free, synchronization-free
steady-state launches, provide `workspace_buffer`, `max_kv_len`, `out`,
and `validate=False`. Packed Q additionally requires `max_seq_len_q`
(FMHA also accepts its non-default `seq_len_q` alias).
Both APIs size and allocate scratch only when it is omitted, then pass it to
the same wrapper plan/run path. Trusted calls select the existing cached
kernel from static Python/tensor attributes without allocating or reading
metadata values back to the host. FMHA preserves initialized caller scratch
by planning with `initialize_workspace=False`.

Use the existing `get_prims_ts_batch_*_workspace_size` helpers outside
capture. Zero caller-owned FMHA scratch before its first trusted standalone
launch and re-zero it when workspace-layout inputs change; wrappers
initialize control storage by default. Scratch and output must not
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
        q, paged_kv_cache, block_tables, seq_lens_kv,
        workspace_buffer=workspace, max_kv_len=max_kv_len,
        out=out, validate=False,
    )

decode()  # Warm up outside capture.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    decode()
graph.replay()
```

### Reusable plans

`plan()` takes model configuration, exact batch size, and Q/KV bounds. It
selects a cached policy, obtains cached compiled kernels, and binds workspace
before execution. `run()` uses that prepared state with live tensors and
metadata; it does not select policies or compile kernels. Replan when the
planned shape/capacity contract changes. Growing live KV lengths within the
planned capacity does not itself require replanning.

Decode and MLA retain their geometry-based standalone workspace-size queries.
Pass the same problem geometry and dtypes to the query and `plan()`. Caller
scratch can be passed directly to `plan()` without a sizing query; otherwise
planning allocates exactly the selected layout. To reuse one buffer across
configurations, allocate the maximum queried size and initialize decode control
storage for each new layout. Context has no split-reduction workspace or public
workspace-size query; variable-window metadata retains its existing behavior.

```python
import torch
from flashinfer.attention.prims_ts import (
    BatchDecodePagedTSWrapper,
    get_prims_ts_batch_decode_workspace_size,
)

device = torch.device("cuda:0")
dtype = torch.bfloat16
num_qo_heads, num_kv_heads, head_dim, page_size = 32, 4, 128, 32
batch_size, max_seq_len_q, max_kv_len = 2, 2, 128

# Query scratch for the same geometry and dtypes used by plan().
qo_indptr = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
workspace_bytes = get_prims_ts_batch_decode_workspace_size(
    batch_size, num_qo_heads, num_kv_heads, head_dim, page_size, max_kv_len,
    max_seq_len_q=max_seq_len_q, q_dtype=dtype, kv_layout="NHD",
    mask_type="causal", device=device,
    qo_indptr=qo_indptr,
)
workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
wrapper = BatchDecodePagedTSWrapper("NHD")
wrapper.plan(
    device, batch_size, num_qo_heads, num_kv_heads, head_dim, page_size, max_kv_len,
    max_seq_len_q=max_seq_len_q, q_dtype=dtype, packed_query=True,
    mask_type="causal", workspace_buffer=workspace,
)

# Two requests: Q lengths [2, 1], KV lengths [128, 96].
q = torch.randn(3, num_qo_heads, head_dim, dtype=dtype, device=device)
seq_lens_kv = torch.tensor([128, 96], dtype=torch.int32, device=device)
block_tables = torch.tensor(
    [[0, 1, 2, 3], [4, 5, 6, 0]], dtype=torch.int32, device=device
)
# Separate NHD K/V: [pages, page_size, kv_heads, head_dim].
k_cache = torch.randn(7, page_size, num_kv_heads, head_dim, dtype=dtype, device=device)
v_cache = torch.randn_like(k_cache)
out = torch.empty_like(q)

wrapper.run(q, (k_cache, v_cache), seq_lens_kv, block_tables, qo_indptr=qo_indptr, out=out)
result = wrapper.run(
    q, (k_cache, v_cache), seq_lens_kv, block_tables,
    qo_indptr=qo_indptr, out=out, validate=False,
)
print(f"Workspace: {workspace.numel():,} bytes; output: {tuple(result.shape)}")
```

An existing arena can instead be supplied directly as
`BatchDecodePagedTSWrapper("NHD", workspace_buffer=existing_workspace)`.
Standalone workspace-size queries return the same hardware-based bound. A
selected auxiliary/forced format that exceeds that bound is checked in `plan()`.

CPU policy/spec resolvers and compiled-callable caches use `functools.cache`,
so equivalent plans across layers reuse selection and compiled code. The
persistent compilation and occupancy caches from
[#6529](https://github.com/flashinfer-ai/flashinfer/pull/6529) remain in use.
Each wrapper retains one prepared execution state and owns its resource
bindings; there is no additional shared-policy or multi-key GPU-binding cache.

Complete planning and representative validation before CUDA graph capture.
Capture subsequent `run()` calls with `validate=False` and caller output/LSE
buffers. Graph replay executes captured GPU work without Python planning or
dispatch. Keep all captured addresses and owners alive; replanning invalidates
the previous graph contract and requires prior launches/replays to finish.
Decode controls occupy a fixed arena tail disjoint from partial-result views.
Concurrent execution requires independent wrappers/workspace.

Validation reads metadata values on the host. With `validate=False`, callers own
shape, dtype, stride, capacity, page-ID, offset, aliasing and lifetime conditions.
Output/LSE/workspace must be disjoint from inputs and live buffers. Update live
metadata only between completed launches/replays and remain within plan bounds.

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

`count_current_block_in_topk=True` preserves the completed-block selection
contract and appends the current block only while it is incomplete. Set False
to select up to `block_topk` historical blocks and always append the current
block, including when its last token completes it. For query position `p`,
the live ID prefix then has `min(block_topk, p // kv_block_size)` entries,
all strictly less than `p // kv_block_size`. No live-count tensor is required.
Pass the same option to workspace sizing and planning. Both modes accept
independent per-KV-head patterns and fixed or packed queries.

`plan` binds the single caller-owned byte workspace and fixes
`batch_size`, `group_size`, head geometry, `block_topk`,
`max_seq_len_kv`, dtypes, and packed-versus-fixed Q layout. The first eager
`run` binds the live tensor ABI, compiles the selected kernels, and
initializes split-KV state. Capture only later `run` calls. One wrapper
revision owns mutable route and split-KV scratch, so unordered concurrent runs
need distinct wrappers.

Packed prefill uses `q[total_q, Hq, D]`, `qo_indptr`, and
`packed_query=True`; planned `group_size` is the maximum request-safe route
length. Fixed MTP decode uses `q[B, Nq, G, Hq, D]` without `qo_indptr`,
where planned `batch_size = B * Nq` and `group_size = G`. The default
`group_size=1` disables query grouping; pass a larger G explicitly for either
layout, using the same value for workspace sizing and planning. Q lengths need not
be divisible by a suggested group: packed mode permits a shorter final route,
while fixed mode may append consecutive semantic dummy rows and discard their
outputs.

For request-aligned fixed Q, `run(..., seq_lens=kv_lengths)` may omit
`token_to_request` and `query_positions`. `kv_lengths` is a contiguous CUDA
Int32/Int64 `[B]` tensor including the request's `Nq * G` query tokens.
The existing attention/metadata kernels derive request IDs and positions
without allocating per-token metadata or launching another preparation
kernel. Nonpositive KV lengths mark inactive requests; positive lengths must
cover all query tokens. Packed Q and arbitrary mappings continue to accept
explicit per-token metadata.

Packed Q uses a bounded `[D, Hq, total_q_tokens]` TensorMap. Per-route masks
handle tiles crossing a request boundary, and the physical token extent
bounds the final tile. FP8 queries need no extra tail allocation or staging copy.

`split_kv=False` selects nonsplit execution and omits reduction scratch;
`split_kv=True` permits useful automatic splitting. The caller chooses the
phase policy: typically False for prefill and True for decode. This choice is
independent of packed/fixed Q and must match workspace sizing and planning.

`get_q_token_kv_block_sparse_workspace_size` sizes persistent compact route
metadata plus disjoint attention scratch. The model's per-request
`max_seq_len_kv`, rather than the global physical page-pool capacity, bounds
the plan. Each group and pattern head forms at most
`G * (block_topk + 1)` selected/tail candidates and ORs per-query membership
bits in one CTA. A shared byte/bit map forms the sorted union when its footprint
fits without adding CTA waves; otherwise bounded radix sort is used. This
temporary map is sized by model context, not physical cache capacity, and adds
no caller-owned workspace. Shared patterns prepare one row per
group; independent patterns prepare one per group and KV head. G1 resolves
selected blocks and the causal tail inside attention, without a metadata launch.

Eligible prepared FP8 G2–G8/page-4 Keeps routes store query-oriented membership
words internally, with eight query slots and complete tail tiles reserved in
the workspace. Unused slots and tail bits are zero. Standalone metadata outputs
retain four membership bytes per Int32 word; the public plan/run API is unchanged.

Nonsplit sparse grids larger than one service wave use the common CLC
persistent scheduler. Each work item resolves its own request, query group and
KV-head metadata. Page producers stage locators with `cp.async`; grouped
membership storage is retained until all softmax consumers finish the item.
The same paths support fixed and packed Q, including partial and empty packed
groups. `split_kv=False` disables splitting, not persistent scheduling.

The production specialization supports D64/D128/D192/D256, matching FP16/BF16
Q/K/V/output, or FP8 E4M3 Q/K/V with FP16/BF16 output. It is causal and
non-windowed, uses KV128 for Q1--Q8, and requires
`group_size * (Hq / Hkv) <= TileQ128`. Groups exceeding 64 token/head rows
use TileQ128; per-fragment query membership still occupies one byte.
Equal Q/K/V D192 uses the same sparse dtype contract, with native
192-channel global tensors and internal
head-dimension staging. FP8 uses 128-wide K/V stages; Q staging rounds to
256 elements internally, with the last 64 supplied by TMA zero fill. The D192
BF16/FP8 paths and bounded packed-Q loads are additionally tested on GB300/SM103
with CUDA graphs and allocation-boundary checks. Decode's
native logical HND cache views can use strided NHD/interleaved physical storage.
The pure-host
`suggest_q_token_kv_block_sparse_group_size` helper takes a caller-cached SM
count and prefers the largest legal group that can fill one service wave using
independent routes plus useful split-KV work. It falls back toward Q1 when a
large union would leave SMs idle. The helper never queries device properties
or reads tensors. Its `seq_len_q` input is the actual per-request query
length, distinct from the suggested `group_size`.

`q_token_kv_block_sparse_attention_with_paged_kv_cache` is the eager
plan-plus-run convenience API and is not graph-capturable. Shared argument
spellings intentionally match paged block-sparse attention: `q`,
`paged_kv_cache`, `kv_block_size`, `mask_type`,
`sm_scale`, and `max_seq_len_kv`. Query grouping is specified separately with
`group_size`, not a sequence-length argument. The optional runtime
`v_scale` applies the value-cache dequantization scale and defaults to one.
The convenience API infers physical page size from `paged_kv_cache`; only
the prepared wrapper's `plan` takes `page_size` explicitly.

On SM90 and newer, the combined route-builder and attention launch use PDL.
Metadata releases at entry so the attention prologue can overlap it; the
attention dependency wait still covers all metadata stores.
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
owns no route workspace, and its `run` takes Q/K/V only, plus `sage=` for a
Sage plan. A dense plan may omit `q_block_size` and `kv_block_size`; each then
defaults to 128, which selects the Q128/KV128 tile. Block-sparse plans require
both.

Every plan takes `softmax_rescale_threshold_log2` (0 to 8, default 8). On the
Q64/KV256 and Q128/KV128 tiles of block-sparse plans, and on dense Q64/KV256
plans, a row keeps its softmax exponent anchor while a tile raises the row
maximum by at most that many binades, so the accumulated output is not
rescaled; 0 rescales on every new maximum.

## Sage attention

Sage attention runs attention on 8-bit Q, K and V with dequantization scales
on two kernels: the contiguous decode kernel, in the dense and block-sparse
modes of `BlockSparseTSWrapper` and `block_sparse_attention`, and the
contiguous context kernel, on dense fixed-length plans of
`BatchPrefillTSWrapper`. The plan fixes the recipe with a
`SageAttentionConfig` (`sage_config=` for the block-sparse APIs, `sage=` for
`BatchPrefillTSWrapper.plan`); every run supplies the scales as
`sage=SageAttentionParams(...)`:

```text
S[r][c] = sfQ[blkQ(r)] * sfK[blkK(c)] * (Q8 . K8^T)[r][c]
P       = softmax_row(S), quantized to E4M3 as C * p
O[r][d] = sfV[d] * (sum_c P8[r][c] * V8[c][d]) / (C * l[r])  (+ v_mean[d])
```

`C` is the kernel's P scale ([Kernels and features](#kernels-and-features)).
FlashInfer does not quantize: the 8-bit tensors and scales come from the
caller, for example TensorRT-LLM's `sageQuant`. `v_mean` adds a per-channel
mean back after normalization, for callers that quantized `V - v_mean`.

### Recipe and scale tensors

Both kernels take the same recipe and scale tensors, and both wrappers check
the recipe at plan time (`validate_sage_recipe`). The defaults are
TensorRT-LLM's production recipe.

| Input | Supported values |
| --- | --- |
| Q/K dtype | One dtype: `torch.float8_e4m3fn`, or `torch.int8` on SM100a/B200 only |
| V dtype (context/decode `v_dtype`; block-sparse `v_data_type`) | `torch.float8_e4m3fn`; it defaults to the K dtype, so INT8 plans set it |
| Output dtype | `torch.bfloat16` (the default) or `torch.float16` |
| `q_block_size` | Power of two, which may exceed the Q tile; default 1 |
| `k_block_size` | 1, 4, 16, 32, 64, 128 or 256; default 16 |
| `k_summary_block_size` | One of the `k_block_size` values, for proxy summary scales; default `k_block_size` |
| `v_mean` | Whether every run supplies `v_mean`; default `False` |

All scales are contiguous, 16-byte-aligned `torch.float32` tensors on the run
device; a validating `run()` checks them against the plan. Every scale must
be finite and positive; the kernels do not check the values. Q and K scales
use the trtllm-gen flat layout: within one head, sequence `b` of
`[B, S, H, D]` starts at slot `b * S // blk + b` and token `t` uses slot
`t // blk` inside it, so one head owns
`flat_scale_numel(B, S, blk) == ceil(B * S / blk) + B - 1` slots.

| Field | Shape |
| --- | --- |
| `q_scale` | `[Hq, flat_scale_numel(B, Sq, q_block_size)]` |
| `k_scale` | `[Hkv, flat_scale_numel(B, Skv, k_block_size)]` |
| `v_scale` | `[B, Hkv, D]` |
| `v_mean` | `[B, Hkv, D]`; exactly when the recipe sets `v_mean=True` |
| `k_summary_scale` | `[Hkv, flat_scale_numel(B, ceil(Skv / kv_block_size), k_summary_block_size)]`; exactly for proxy plans |

With proxy routes, `k_summary` holds the per-block K means in the K dtype,
quantized as one more K sequence of `ceil(Skv / kv_block_size)` tokens with
`k_summary_block_size`; `k_summary_scale` holds its scales. `v_summary` holds
the per-block V means in E4M3 and shares `v_scale` (built from `V - v_mean`
when a mean is used).

Every run first launches a small kernel that gathers the plan's scales into
one buffer in the order the attention kernel reads them: the K scales per
64-token atom, the Q scales per Q-tile row and the V scales (and means) per
KV head; attention reads that buffer alone. A block-sparse run's route
prepare follows it as a programmatic dependent launch, starting while the
scale kernel drains and acquiring it before exiting, so the attention launch
that follows in stream order finds both the scales and the routes complete.

### Kernels and features

Each kernel obtains a K tile's scales by a transport (`SageKTransport`) and
applies them by a score scaling (`SageScoreScaling`), both chosen by the K
block size. `BLOCK` scaling keeps the scores raw and folds `sfQ * sfK` into
each block's maximum and exp2 multiplier; `DEQUANTIZED` scaling multiplies
the scores in place in the max pass, which costs more per score.

| | Decode kernel | Context kernel |
| --- | --- | --- |
| APIs | `BlockSparseTSWrapper`, `block_sparse_attention` | `BatchPrefillTSWrapper` |
| K/V storage | Contiguous; the paged block-sparse APIs do not support Sage | Contiguous fixed-length `[B, S, H, D]` (`packed_query=False`) |
| Geometry | `head_dim=128`; a Q64/KV256 or Q128/KV128 Keeps tile, which needs a `kv_block_size` multiple of 64 and at least 64 grouped Q rows (`q_block_size * Hq / Hkv`) | `head_dim=128`; 128-token Q tiles of one Q head |
| Masks and routes | Dense and causal masks; exact and proxy block-sparse routes (BSR or bitmask) | `mask_type="dense"` and `window_left=-1`; no routes, so `k_summary_block_size` is unused |
| Split-KV | Rejected: only the direct output store applies `sfV` and `v_mean` | No split-KV launch |
| K transport | `RING` for every K block size: the load warp stages each K tile's scales in shared memory | `GLOBAL` for 4-token and coarser blocks: the softmax loads them before the score wait; `RING` for 1-token blocks |
| Score scaling | `BLOCK` for 4-token and coarser blocks; `DEQUANTIZED` for 1-token blocks | `BLOCK` per 16-column block for 16-token and coarser blocks; `DEQUANTIZED` for 4- and 1-token blocks |
| P scale `C` | `448 * 2**-h`, with `h` the plan's `softmax_rescale_threshold_log2` (default 8) on Q64/KV256 tiles and block-sparse plans, and 0 elsewhere | 1: the lazy correction lets `p` reach `2**8` |

Masks and scheduling otherwise follow each kernel's 16-bit plans. In the
decode kernel the default threshold quantizes FP8 P at
`C = 448 * 2**-8 = 1.75`: the E4M3 grid of P moves down by the threshold, so
probabilities far below the anchor round to subnormals or zero. Rows whose
largest scores arrive first, such as an attention sink in the leading blocks,
lose the most. A smaller threshold trades rescales back for range; 0 keeps
exact anchors and `C = 448`.

### Performance

One dense layer (B=2, S=75600, 40 heads, D=128) with per-token Q scales,
smoothed K and per-channel V scales with a V mean, attention kernel only, in
Mcycles per call at the GPC clock (GB300: nsys GPU metrics; B200: ncu
`sm__cycles_elapsed.max`). The Sage rows list K blocks of 16, 4 and 1 tokens.

| Plan | GB300 | B200 |
| --- | ---: | ---: |
| Context, per-tensor FP8 (no Sage) | 162 | 213 |
| Context, Sage FP8 | 169 / 202 / 204 | 223 / 251 / 244 |
| Context, Sage INT8 | not supported | 229 / 264 / 277 |
| Decode (Q64/KV256), Sage FP8 | 202 / 220 / 260 | 274 / 288 / 317 |

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
    q_dtype=fp8,
    kv_dtype=fp8,
    sage_config=SageAttentionConfig(q_block_size=1, k_block_size=16),
)

# The quantizer (for example sageQuant) produces the tensors and scales.
q = torch.randn(B, Sq, Hq, D, device=device).to(fp8)
k = torch.randn(B, Skv, Hkv, D, device=device).to(fp8)
v = torch.randn(B, Skv, Hkv, D, device=device).to(fp8)
scales = SageAttentionParams(
    q_scale=torch.rand(Hq, flat_scale_numel(B, Sq, 1), device=device),
    k_scale=torch.rand(Hkv, flat_scale_numel(B, Skv, 16), device=device),
    v_scale=torch.rand(B, Hkv, D, device=device),
)
out = wrapper.run(q, k, v, sage=scales)  # [B, Sq, Hq, D] bfloat16
```

Continuing the example, a block-sparse INT8 plan (SM100a/B200 only) with proxy
routes names its E4M3 V; each Q block attends exactly to the selected KV
blocks, and the other blocks enter through their per-block summaries:

```python
wrapper.plan(
    B, Sq, Skv, Hq, Hkv, D, 64, 64,
    device=device,
    max_blocks_per_row=8,
    use_proxy_routes=True,
    q_dtype=torch.int8,
    k_dtype=torch.int8,
    v_dtype=fp8,
    sage_config=SageAttentionConfig(),
)

num_q_blocks, num_kv_blocks = -(-Sq // 64), -(-Skv // 64)
i8 = dict(low=-127, high=128, dtype=torch.int8, device=device)
q = torch.randint(size=(B, Sq, Hq, D), **i8)
k = torch.randint(size=(B, Skv, Hkv, D), **i8)
v = torch.randn(B, Skv, Hkv, D, device=device).to(fp8)
k_summary = torch.randint(size=(B, num_kv_blocks, Hkv, D), **i8)
v_summary = torch.randn(B, num_kv_blocks, Hkv, D, device=device).to(fp8)

# BSR routes: every Q block of every (batch, KV head) row selects the same
# KV blocks; block_indptr holds offsets into the flat block_indices.
selected = torch.tensor([0, 1, num_kv_blocks - 1], dtype=torch.int32, device=device)
rows = B * Hkv * num_q_blocks
block_indices = selected.repeat(rows)
block_indptr = (
    torch.arange(B * Hkv, dtype=torch.int32, device=device).view(B, Hkv, 1)
    * num_q_blocks
    + torch.arange(num_q_blocks + 1, dtype=torch.int32, device=device)
) * selected.numel()

scales = SageAttentionParams(
    q_scale=torch.rand(Hq, flat_scale_numel(B, Sq, 1), device=device),
    k_scale=torch.rand(Hkv, flat_scale_numel(B, Skv, 16), device=device),
    v_scale=torch.rand(B, Hkv, D, device=device),
    k_summary_scale=torch.rand(
        Hkv, flat_scale_numel(B, num_kv_blocks, 16), device=device
    ),
)
out = wrapper.run(
    q, k, v, block_indptr, block_indices,
    k_summary=k_summary,
    v_summary=v_summary,
    sage=scales,
)
```

`block_sparse_attention(..., sage=...)` runs the same launches in one shot;
its recipe is `sage_config` or, when omitted, the default recipe with a V mean
exactly when `sage.v_mean` is set.

## VC-Attention

VC-Attention, an adaptation of
[Li et al., 2026](https://arxiv.org/html/2609.15810v1), runs the dense
fixed-length context kernel (`BatchPrefillTSWrapper` and `batch_prefill`) with
bf16 Q and K, E4M3 V with value smoothing, and a direct probability cast.
The plan selects the recipe at compile time with
`vc_config=VCAttentionConfig(...)`. Every run passes the permuted K, the E4M3 V
residuals and `vc=VCAttentionParams(...)`:

```text
S[r][c] = (Q . K[perm]^T)[r][c]
P8      = softmax_row(S), quantized to E4M3
O[r][d] = sfV[b, h, d] * (sum_c P8[r][c] * V8[perm(c)][d] + sum_t rowsum_t(P8[r]) * mu_t[d]) / l[r]
```

`V8` holds the E4M3 residuals of the k-means-permuted values around their
128-token tile means `mu_t` (stored in bf16, divided by `sfV`); the kernel adds
the mean terms back inside the online softmax recurrence, two K=16 UMMA steps per
group of 16 tiles, so the row-max correction covers them. `VCAttentionPreprocessor` turns bf16 K/V into the
operands with the paper's V-Smooth schedule: grouping and demeaning run on the
first `smooth_step_fraction` of the denoising steps, the permutation is
refreshed every `perm_refresh_every` steps inside that window and kept
afterwards. `vc_quantize` runs the same preparation without the schedule, and
`vc_reference` is its torch reference. The `vc_attention` package holds the
operand contract, the quantizer in `quantization` and its CuTe DSL kernels in
`prepare`.

| Input | Supported values |
| --- | --- |
| Q/K dtype | `torch.bfloat16` |
| V dtype | `torch.float8_e4m3fn` |
| Output dtype | `torch.bfloat16` or `torch.float16` |
| Geometry | `head_dim=128`, `num_qo_heads == num_kv_heads`, `packed_query=False`, `mask_type="dense"`; runs the two-CTA UMMA form |
| LSE / attention sinks | Unsupported for VC's quantized softmax sums |

### Operand tensors

`v_scale` is the `[B, Hkv, D]` fp32 per-channel residual scale and
`tile_means` the packed bf16 `[B, Hkv, ceil(Skv / 2048), 16, 256]` mean operands
the preparation writes; both are contiguous and 16-byte aligned on the run
device, and a validating `run()` checks them against the plan. The plan's
`VCAttentionConfig` sets `demean`. It is off after the V-Smooth window, when
the means are zero, so a caller holds one plan of each kind and picks by
`ops.demean`. `vc_quantize` runs the two CuTe DSL preparation kernels and
returns them with the permuted K and the E4M3 V as `VCAttentionOperands`, whose
`.params` is the run-time object. `VCAttentionPreprocessor.prepare` drives
`vc_quantize` with the V-Smooth schedule, and `vc_reference` is the torch
reference over those operands. `vc_quantize_repair` builds the V repair operands
instead and their plan has demean off and a positive `repair_tiles`.

### Example

```python
from flashinfer.attention.prims_ts import (
    BatchPrefillTSWrapper, VCAttentionConfig, VCAttentionPreprocessor,
)

wrappers = {}
for demean in (True, False):
    wrappers[demean] = BatchPrefillTSWrapper()
    wrappers[demean].plan(
        device="cuda", batch_size=1, max_seq_len_q=S, max_kv_len=S,
        num_qo_heads=H, num_kv_heads=H, head_dim=128,
        q_dtype=torch.bfloat16, k_dtype=torch.bfloat16, v_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16, vc_config=VCAttentionConfig(demean=demean),
    )
prep = VCAttentionPreprocessor()
ops = prep.prepare(k_bf16, v_bf16, denoise_step=(step, num_steps))
out = wrappers[ops.demean].run(q_bf16, ops.k, ops.v, vc=ops.params)
```

The one-shot `batch_prefill(..., vc=ops.params)` plans the recipe from the
operands; `vc_config` without `vc` is rejected.

## Validation

### Validation scope

The established kernels retain their earlier SM100a/B200 qualification.
Reusable eager plans, NHD caches, final LSE/sinks, generic MLA heads, and
runtime split-KV have representative correctness and latency validation on
SM103/GB300 with PyTorch 26.08, CUDA 13.4, and CuTe DSL 4.7.0. The contiguous
QK192/V128 extension is also covered on that stack. This is the measured
matrix for these extensions. Additional SM103/GB300 correctness smoke covers
CuTe DSL 4.8.1, including NHD prefill, decode/MLA runtime split counts, tracing,
clustered LSE and graph replay. Performance qualification remains specific to
the measured DSL 4.7.0 stack.

Run the numerical, graph, scheduler/resource, and public-surface contracts:

```bash
pytest -q \
  tests/attention/test_attention_ts_context.py \
  tests/attention/test_attention_ts_decode.py \
  tests/attention/test_attention_ts_q_token_kv_block_sparse_metadata.py \
  tests/attention/test_attention_ts_block_sparse.py \
  tests/attention/test_attention_ts_features.py \
  tests/attention/test_attention_ts_mla_decode.py \
  tests/attention/test_attention_ts_sage.py \
  tests/attention/test_attention_ts_context_sage.py \
  tests/attention/test_attention_ts_vc.py
```


## Final statistics and sink semantics

PrimTS context and dense FMHA/MLA decode plans accept `lse_mode="none"`
(default) or `"base2"`. Base-2 is the single output convention, matching
FlashInfer FMHA/cascade statistics. Enable LSE in planning, then request
`return_lse=True` to receive `(output, lse)`. An optional compact CUDA FP32
`lse` output has the query shape minus its final dimension; it is written
whenever supplied to an enabled plan, even if the return flag is false.
LSE excludes output scaling and removes internal FP8 probability amplification.
Keep output and LSE buffers disjoint from live inputs and workspace.
Sparse MLA retains its existing `return_lse` contract: natural-log QK LSE
before the attention-sink contribution. Its scale arguments also accept scalar
CUDA FP32 tensors, while dense wrappers use host scalar scale operands.

Context and dense FMHA decode plans additionally support
`use_attention_sinks=True`, binding `attention_sinks` on every run. This FP32
CUDA `[Hq]` vector contains natural-log sink logits after QK scaling and adds
denominator mass with zero value contribution. Callers must supply finite
values or `-inf`; validation checks tensor metadata without reading these
values. The stable final merge happens once, after any split reduction and
before output rounding; returned LSE includes the sink. Values may change
between completed graph replays. Supply caller outputs and disable validation
for allocation-free replay.

FP8 scalar dequantization remains caller-composed: effective QK scale is
`attention_scale * q_scale * k_scale`; effective output scale includes `v_scale`.
Dense context, FMHA decode and MLA decode use `softmax_scale` / `output_scale`.
Where applicable, their public interfaces use `q_dtype` / `k_dtype` / `v_dtype` /
`kv_dtype` / `out_dtype`, `packed_query`, `num_qo_heads`, `seq_lens_kv`, and
`max_kv_len`. Sparse MLA uses `attention_sinks` and `use_attention_sinks`.
The QK scale is in natural-exponential units; context converts it to base two
internally. Context planning sets defaults that `run()` may override. Context
and FMHA decode default to `1 / sqrt(D)`; MLA retains its default of 1.
Context scales are host FP32 scalar operands; device scale tensors are not
supported. Captured graphs retain the scalar values used during capture;
changing them requires recapture. Scalar values are validated on the host and do not specialize JIT. Packed NVFP4 block
scales remain a separate input contract.

### Runtime split-KV policy

Split counts are automatic and private, with the physical SM count as their
ceiling. The policy bounds useful work using execution geometry and KV capacity,
then prepares scratch/reduction capacity in ranges 1, 2–4, 5–8, 9–16, 17–32,
33–64, and sixteen-split ranges above 64 clipped to the hardware ceiling.
No count, ceiling, or count tensor is exposed to users. Reduction selection checks
cluster residency with the rounded capacity and falls back to the standalone
reducer when that physical cluster grid no longer fits one resident wave.

Producers and reducers receive typed runtime counts. Scratch strides and
cluster/barrier participation retain prepared capacity; inactive slots/ranks
contribute neutral state and preserve teardown. Count changes within a capacity
reuse the compiled callables, including reduction. Static internal kernel
recipes remain available for comparison.

MLA 2CTA plans size scratch from the prepared bound, replacing the implicit
128-split reservation. Parallel reduction is selected from the capacity/topology
envelope. The launcher prepares exact integer-division metadata once per call
and shares it with scheduled tile consumers. Full Q128 runtime-split graphs
reserve eight additional registers per Softmax thread from the correction warpgroup.
