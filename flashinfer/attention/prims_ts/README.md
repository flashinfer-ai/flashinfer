# Experimental Task-Scheduled Attention

`flashinfer.attention.prims_ts` exposes experimental CuTe DSL attention
kernels for NVIDIA Blackwell GPUs. Scheduling, tile selection, and split-KV
reduction are implementation details; the public interfaces expose attention
and cache semantics without tuning knobs.

Current accuracy and performance signoff is on SM100a/B200. SM103a/B300 is
admitted by the runtime architecture guard but is not yet signoff-qualified.

## Guides and public APIs

Import all entries below from `flashinfer.attention.prims_ts`.

| Kernel | Guide | Public APIs |
| --- | --- | --- |
| FMHA context/prefill | [Task-Scheduled FMHA Context](kernels/fmha_context/README.md) | `BatchPrefillTSWrapper`, `batch_prefill`, `BatchPrefillPagedTSWrapper`, `batch_prefill_with_paged_kv_cache` |
| FMHA decode | [Task-Scheduled FMHA Decode](kernels/fmha_decode/README.md) | `BatchDecodePagedTSWrapper`, `batch_decode_with_paged_kv_cache`, `get_prims_ts_batch_decode_workspace_size`, `prims_ts_batch_decode_with_kv_cache` |
| Contiguous FMHA decode | [Sage attention](#sage-attention-8-bit-qkv-with-dequantization-scales) below; kernel notes in [Task-Scheduled FMHA Decode](kernels/fmha_decode/README.md#sage-attention) | `BatchDecodeTSWrapper` over compact BSHD Q/K/V; `plan(use_block_sparse=...)` selects the dense or block-sparse mode; `SageAttentionParams` for 8-bit Q/K/V with scales |
| Block-sparse FMHA | — | `BlockSparseTSWrapper`, `block_sparse_attention`; fixed-Q paged KV: `BlockSparsePagedTSWrapper`, `block_sparse_attention_with_paged_kv_cache` |
| MLA decode | [Task-Scheduled MLA Decode](kernels/mla_decode/README.md) | `BatchMLADecodePagedTSWrapper`, `batch_mla_decode_with_paged_kv_cache`, `get_prims_ts_batch_mla_decode_workspace_size`, `prims_ts_batch_mla_decode_with_kv_cache` |

The component guides define supported shapes, layouts, metadata lifetime,
output/workspace ownership, examples, limitations, and validation commands.

The contiguous and paged context, FMHA decode, and MLA decode wrappers separate
reusable static state from per-run request state. `plan()` compiles a static
capacity, shape, dtype, and storage-mode specialization without retaining
request tensors or metadata. Paged context plans may additionally freeze
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
live per-request logical lengths. Paged PrimTS does not support packed or
mixed/variable Q lengths.
Eager launches retain all launch tensors on the run stream; CUDA Graph users
must keep the wrapper and Q/cache/output/runtime-metadata tensors alive and
unmodified until replay completes. Values may change between completed replays
while tensor addresses, shapes, dtypes, and strides remain stable.

Qualified Q64/coarse-KV profiles retain KV256 routes for page sizes 64 and
128. Optional `kv_valid_bits` is a `torch.uint32` per-request bitset with shape
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
share sparsity. A proxy run supplies one K arithmetic mean and one V arithmetic
mean per semantic KV block; each proxy block then counts as its number of
structural tokens in the softmax. V summaries must be per-block means: a
per-block sum is not detectable and yields outputs scaled by the block mass.
The final partial block averages only its structural tokens. Optional `kv_valid_bits` filters exact K/V tokens only and
does not change proxy summaries or their represented mass. With Sage attention
the K summaries use the K dtype and are dequantized with
`SageAttentionParams.k_summary_scale`; the V summaries are E4M3 and share
`v_scale` (see below).

## Sage attention (8-bit Q/K/V with dequantization scales)

`BatchDecodeTSWrapper.plan(..., sage=SageAttentionParams(...))` runs the
contiguous decode kernel on 8-bit Q, K and V in both modes (dense and
block-sparse, exact and proxy routes). The math is the one implemented by
trtllm-gen and consumed by TensorRT-LLM's `sageQuant`-based pipeline:

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
| `SageAttentionParams.q_block_size` | Power of two no larger than the Q tile (64 or 128, see below); default 1 |
| `SageAttentionParams.k_block_size` | One of `SAGE_K_BLOCK_SIZES == (16, 32, 64, 128, 256)`; default 16 |
| Head dimension | 128 |
| K/V storage | Contiguous `[B, S, Hkv, D]` only; the paged wrappers accept neither `sage` nor INT8 |
| Profiles | Streamed Keeps Q64/KV256 and Q128/KV128 (dense and block-sparse) |
| Mask | `dense` or `causal` (proxy routes stay `dense`) |
| Scheduling | Static direct grid or the persistent scheduler: block-sparse plans follow the 16-bit policy, dense KV256 plans go persistent past one resident wave; no split-KV, sliding window or attention sinks |

`kv_data_type` is the K dtype. V follows K unless `v_data_type` is given,
which is how the INT8 recipe (`int8` Q/K, `float8_e4m3fn` V) is expressed;
`(int8, int8)` and `(float8_e4m3fn, int8)` are rejected. The defaults
(`q_block_size=1`, `k_block_size=16`) are TensorRT-LLM's production recipe
`(1, 16, 1)`. Its `(1, 1, 1)` and `(1, 4, 1)` recipes are not supported: the
softmax applies one multiplier per compile-time group of a K32 score
fragment, and K blocks below 16 tokens would need distinct multipliers inside
a group.

The profile follows from the same `q_block_size`/`kv_block_size` plan
arguments as 16-bit decode: `kv_block_size` must be a positive multiple of
64, and the Q tile is the largest power of two no larger than
`min(q_block_size * Hq / Hkv, 128)` that fits the block geometry. A Q64 tile
selects Q64/KV256, a Q128 tile selects Q128/KV128; smaller tiles are Swaps
profiles and reject `sage`, as does a `kv_block_size` that is not a multiple
of 64. Dense Q128/KV128 with `sage` always streams its P fragments, because
the scales enter through the streamed fragment passes. Block-sparse plans pick
the static grid or the persistent scheduler exactly as 16-bit plans do. Dense
contiguous KV256 plans switch to the persistent scheduler once the static
grid exceeds one resident wave, because the work-tile loop overlaps a tile's
epilogue with the next tile's QK head; 16-bit dense plans and the Q128/KV128
profile keep the static grid.

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
| `k_summary_scale` | `[Hkv, flat_scale_numel(B, ceil(Skv / kv_block_size), k_block_size)]` | Required by, and only by, block-sparse proxy plans |

The block sizes are compile-time and join the kernel cache key; the tensors
are bound at `plan()` and validated again for shape, dtype, device and
contiguity on every `run()`. Sequence tails and masked columns beyond the
valid length read the last valid slot, so a scale array never needs padding
beyond `flat_scale_numel`.

### Block-sparse contract with Sage

Proxy routes treat the K summaries as one more K sequence of
`ceil(Skv / kv_block_size)` tokens: `k_summary` holds the per-block K means
quantized in the K dtype (INT8 or E4M3) with `sageQuant` applied to the
summary tensor as if it were K, and `k_summary_scale` is the resulting
flat-layout array. `v_summary` holds the per-block V means (the final partial
block averages only its structural tokens) quantized to E4M3 with the shared
`v_scale`; with `v_mean`, build them from `V - v_mean`. A proxy block stands
for as many identical tokens as it covers, so its mass enters the proxy
logit and its probability carries the block's weight. Token masks
(`kv_valid_bits`) and ragged final blocks are supported exactly as without
Sage.

### Performance

The kernel is issue-bound in its softmax warps, not tensor-bound, so the
8-bit tensors alone do not speed it up. The gain comes from keeping the scale
machinery off the critical path (scales staged with the route metadata and at
tile start, one multiplier per compile-time score group) and from
reinvesting the resources the byte-wide tiles free: a five-stage K/V ring,
the persistent scheduler for dense plans, the output tile's columns split
between the two lane groups of the tail, an INT8 accumulator seeded with an
FP32 bias so scores need no conversion, a smaller instruction footprint in
the masked softmax paths, and a probability pass that stops at a route's last
nonempty K32 fragment. On B200 with the `(fp8, fp8, (1, 16, 1))` recipe,
kernel time relative to the BF16 kernel of the same plan (BF16 dense plans
run the static grid, 8-bit dense plans the persistent scheduler) is 0.81 on
dense S=10800 H=40, 0.94 on dense S=4096 H=8, 0.85 on block-sparse S=4096
H=8 (density 0.25), 0.76 on a VSA-shaped block-sparse case (S=15360, H=40,
density 0.125), 0.76 on the SOL exact case (S=10800, H=40, density 0.175)
and 0.79 on its proxy variant; the numbers do not depend on the logit
distribution. The INT8 recipe is within 0-4% of the FP8 recipe.
`k_block_size=32` is neutral for FP8 and 3-4% faster than 16 for INT8.

### Examples

Dense contiguous decode, E4M3 Q/K/V with the production `(1, 16, 1)` recipe:

```python
import torch
from flashinfer.attention.prims_ts import BatchDecodeTSWrapper, SageAttentionParams
from flashinfer.attention.prims_ts.sage import flat_scale_numel

device = torch.device("cuda")
B, Sq, Skv, Hq, Hkv, D = 2, 128, 1000, 4, 4, 128
fp8 = torch.float8_e4m3fn

# 8-bit tensors and scales come from the quantizer (TensorRT-LLM sageQuant).
q = torch.randn(B, Sq, Hq, D, device=device).to(fp8)
k = torch.randn(B, Skv, Hkv, D, device=device).to(fp8)
v = torch.randn(B, Skv, Hkv, D, device=device).to(fp8)
sage = SageAttentionParams(
    q_scale=torch.rand(Hq, flat_scale_numel(B, Sq, 1), device=device),
    k_scale=torch.rand(Hkv, flat_scale_numel(B, Skv, 16), device=device),
    v_scale=torch.rand(Hkv, D, device=device),
    q_block_size=1,
    k_block_size=16,
)

wrapper = BatchDecodeTSWrapper()
wrapper.plan(
    B, Sq, Skv, Hq, Hkv, D,
    64,   # q_block_size: 64 rows per KV head select the Q64/KV256 profile
    64,   # kv_block_size: a multiple of 64
    device=device,
    use_block_sparse=False,
    q_data_type=fp8,
    kv_data_type=fp8,
    o_data_type=torch.bfloat16,
    sage=sage,
)
out = wrapper.run(q, k, v)  # [B, Sq, Hq, D] bfloat16
```

Block-sparse decode with proxy routes, INT8 Q/K and E4M3 V:

```python
import math

kv_block_size, max_blocks_per_row = 64, 8
num_kv_blocks = math.ceil(Skv / kv_block_size)
q = (torch.randn(B, Sq, Hq, D, device=device) * 40).round().clamp(-127, 127).to(torch.int8)
k = (torch.randn(B, Skv, Hkv, D, device=device) * 40).round().clamp(-127, 127).to(torch.int8)
v = torch.randn(B, Skv, Hkv, D, device=device).to(fp8)
sage = SageAttentionParams(
    q_scale=torch.rand(Hq, flat_scale_numel(B, Sq, 1), device=device),
    k_scale=torch.rand(Hkv, flat_scale_numel(B, Skv, 16), device=device),
    v_scale=torch.rand(Hkv, D, device=device),
    # The K summaries are quantized as their own sequence of num_kv_blocks tokens.
    k_summary_scale=torch.rand(Hkv, flat_scale_numel(B, num_kv_blocks, 16), device=device),
)

wrapper = BatchDecodeTSWrapper()
wrapper.plan(
    B, Sq, Skv, Hq, Hkv, D,
    64, kv_block_size,
    device=device,
    use_block_sparse=True,
    max_blocks_per_row=max_blocks_per_row,
    use_kv_valid_bits=False,
    use_proxy_routes=True,
    q_data_type=torch.int8,
    kv_data_type=torch.int8,
    v_data_type=fp8,           # INT8 K requires E4M3 V, named explicitly
    o_data_type=torch.bfloat16,
    sage=sage,
)
# block_indptr [B, Hkv, ceil(Sq / q_block_size) + 1] and block_indices select
# the exact blocks; k_summary (int8) and v_summary (fp8) are the per-block
# means of the remaining blocks, [B, num_kv_blocks, Hkv, D].
out = wrapper.run(
    q, k, v, block_indptr, block_indices, k_summary=k_summary, v_summary=v_summary
)
```

`BlockSparseTSWrapper`, `block_sparse_attention` and the paged wrappers do
not take `sage`; block-sparse Sage plans go through `BatchDecodeTSWrapper`
with `use_block_sparse=True`.

## Validation

Run the numerical, graph, scheduler/resource, alias-safety, and public-surface
contracts:

```bash
pytest -q \
  tests/attention/test_attention_ts_context.py \
  tests/attention/test_attention_ts_decode.py \
  tests/attention/test_attention_ts_block_sparse.py \
  tests/attention/test_attention_ts_mask.py \
  tests/attention/test_attention_ts_mla_decode.py \
  tests/attention/test_attention_ts_sage.py
```
