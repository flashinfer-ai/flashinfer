# Cake DSA sparse-attention training backend (SM100 / SM103 / SM107)

Experimental backend for the native 64-query-head DeepSeek Sparse Attention
training kernels (top-k sparse MLA with absorbed queries; GLM-5.2 geometry:
64 heads, 512 latent + 64 rope query/key dimensions, 512 value dimensions,
top-k 2048).  Tracking issue: flashinfer-ai/flashinfer#5657.

Both the API and the backend are experimental: no compatibility guarantee,
SM100 (B200 / GB200) is the acceptance architecture of the native kernels and
SM103 (B300 / GB300) is compiled from the same sources; SM107 (Rubin R200) is the
acceptance architecture of the packed / strided-layout extension (flashinfer-ai/flashinfer#5675)
and is compiled from the same sources with an nvcc that emits `compute_107a`.  The feature is JIT-only and does not
participate in automatic backend selection, autotuning or trace-apply.

## Public entry points (`flashinfer/dsa_sparse_attention.py`)

```python
from flashinfer.dsa_sparse_attention import dsa_sparse_attention, dsa_sparse_attention_varlen

out = dsa_sparse_attention(q_latent, q_rope, kv_latent, k_rope, indices,
                           topk_length=None, softmax_scale=None, return_lse=False)
out = dsa_sparse_attention_varlen(q_latent, q_rope, kv_latent, k_rope, gather_kv_indices,
                                  cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                                  topk_length=None, softmax_scale=None, return_lse=False)
```

* `q_latent [T, 64, 512]`, `q_rope [T, 64, 64]` BF16; `kv_latent [S, 512]`
  (K = V) and `k_rope [S, 64]` BF16.  Strided views are consumed in place --
  see "Packed and strided inputs" below.
* `indices [T, topk]` int32 hold **global** key rows; `-1` or `>= S` marks an
  invalid slot anywhere in the row; `topk_length [T]` int32 optionally
  invalidates slots `>= topk_length[t]`; any positive `topk`.
* The varlen form takes per-document indices (`gather_kv_indices`, relative
  to `cu_seqlens_k[d]`) and offsets them on device before the flat kernels
  run.  `cu_seqlens_q` and `cu_seqlens_k` are independent: a query segment
  may be shorter than its key segment, in which case it is the tail of that
  key prefix -- query `local_q` of document `d` sits at key position
  `(seqlen_k[d] - seqlen_q[d]) + local_q`.  With `causal=True` the
  offsetting also drops selected keys after that position (the rule of the
  Cake facade: `offset_gather_kv_indices(..., causal=True)` keeps a slot iff
  `0 <= idx < seqlen_k[d]` and `idx <= (seqlen_k[d] - seqlen_q[d]) +
  local_q`); with `causal=False` (the default, unchanged from the first
  release) a query's key set is exactly its index row.
  Either way the kernels apply no positional mask of their own: the offset
  index rows define the key sets.  A zero-length query segment contributes
  no rows; a row whose every slot ends up `-1` gives `out = 0`, `lse = -inf`
  and zero gradients.
* Forward: `out [T, 64, 512]` BF16 and (with `return_lse=True`) the natural-log
  `lse [T, 64]` FP32 over the valid keys; fully masked rows give `out = 0` and
  `lse = -inf`.  The autograd wrapper also keeps the BF16 output residual
  `o_lo = bf16(fp32(O) - bf16(O))` for the backward's exact `delta`.
* Backward: `dq_latent`, `dq_rope` (computed once per row, bitwise
  deterministic), `dkv_latent [S, 512]`, `dk_rope [S, 64]` (FP32 `red.global`
  accumulation, then a cast to natural-layout BF16).  BF16 MMA operands, FP32
  accumulation throughout.  `cake_backend.backward(..., dkv_fp32=True)` (and a
  runner prepared with `dkv_fp32=True`) returns the dK/dV gradients as
  natural-layout FP32 tensors instead; the kernels' internal accumulator layout
  (registry field `dkv_acc_layout`, `natural` or `permuted`) never crosses the
  API boundary -- a permuted program serves the FP32 mode through its cast
  stage.  With a caller-provided `dkv_acc` (see "Packed fp32 dKV accumulation
  and destination mapping" below) the dK/dV gradients are accumulated into that
  packed FP32 buffer instead, no `dkv_latent` / `dk_rope` tensors are produced
  and the autograd wrapper returns no gradient for `kv_latent` / `k_rope`.

### Packed and strided inputs

The kernels read the query operands through TMA descriptors encoded from the
tensor's own strides and gather the key operands by row stride, so the
trainer's packed layouts (flashinfer-ai/flashinfer#5675) are consumed in place
-- no `.contiguous()`, `cat` or copy on the host:

* `q_rope` may be the `192:256` channel slice of the pre-absorption
  `[T, 64, 256]` query (head stride 256, token stride 16384, storage offset
  192 elements = 384 B), `q_latent` a view of a packed `[T, 64, 576]` query,
  or both contiguous.  Rule (`cake_backend._check_head_tensor`): unit channel
  stride; head and token strides positive multiples of 8 elements (16 bytes,
  the granule of `cuTensorMapEncodeTiled`'s global strides); head stride at
  least the slice width; base address 16-byte aligned.
* `kv_latent` / `k_rope` may be the `0:512` / `512:576` column slices of a
  packed `[S, 576]` or `[S, 704]` row (a frozen 128-channel indexer key stored
  alongside).  Rule (`cake_backend._check_key_tensor`): unit column stride;
  row stride at least the slice width and a multiple of 8 elements; base
  16-byte aligned (the TMA gather descriptors and the 16-byte rope loads).
* `indices` stay contiguous int32 `[T, topk]`; `dout` is made contiguous by
  the eager backward and the autograd wrapper (`bwd_delta` addresses it as
  flat `[T * 64, 512]` rows).
* A strided view yields bitwise the same `out`, `lse` and `dq` as the
  contiguous copy (only descriptor fields and pointer offsets change); dK/dV
  agree within the FP32 `red.global` reduction spread.

Explicit forward / backward entry points without autograd, a prepared
allocation-free runner (`prepare_dsa_train`, CUDA-graph capturable) and the
workspace sizing helper live in `cake_backend.py`.  The eager entry points
(and the autograd wrapper behind the public API) validate and bind once per
input binding -- `(data_ptr, shape, stride, dtype)` of every input plus the
scale -- and launch later calls from the remembered argument plans with
freshly allocated outputs and per-call scratch (`cake_backend.BINDING_CACHE`).
A remembered binding pins no caller tensor and holds no problem-sized
scratch: `delta`, the FP32 dK/dV accumulators and the key-range-pass regions
come from the caching allocator on every call like the outputs, and the
binding owns only the descriptor workspace and a materialized `topk_length`
(kilobytes).  The cache keeps up to 256 bindings
(`FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY` sets the capacity) and
evicts the least recently used one, so a model whose layers cycle through up
to that many forward / backward bindings per step binds each of them once;
`FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE=0` disables it.  A call without
query rows (`T == 0`) returns empty outputs and zero gradients without binding
or launching; `S == 0` is rejected.

Host cost through the autograd wrapper: the `Function.backward` runs on
PyTorch's autograd device thread, where the two thread handoffs (about 30 us
each with an idle GPU, about 170 us each while kernels are queued on the
stream) and the Python body (3-4x slower there than on the main thread) add
roughly 400 us per backward at 4k tokens on B200 that are not in this package
-- a trivial `autograd.Function` with the same saved tensors and gradient
shapes shows the same cost, and no synchronization is involved.  In a
GPU-bound training step this is hidden behind the backward kernels (6-8 ms at
4k tokens).  Host-bound loops should call `cake_backend.forward` /
`cake_backend.backward` directly (about 25 / 55-90 us per call with a
remembered binding on B200 -- the backward figure grows with the per-call
scratch of the key-range-pass rows) or capture the prepared runner into a CUDA
graph.

### Packed fp32 dKV accumulation and destination mapping

`cake_backend.backward(..., dkv_acc=, dkv_dst_map=)`, `prepare_dsa_train(...,
dkv_acc=, dkv_dst_map=)` and the autograd `Function`
(`DSASparseAttentionFunction.apply(q_latent, q_rope, kv_latent, k_rope, indices,
topk_length, softmax_scale, key_passes, dkv_acc, dkv_dst_map)`) accumulate the
dK/dV gradients directly into a caller-provided FP32 buffer instead of
returning `dkv_latent` / `dk_rope` (flashinfer-ai/flashinfer#5675: the packed
dKV of a trainer, with the repeated / remapped rows of a context-parallel window):

* `dkv_acc`: FP32 `[S_dst, >= 576]` -- the latent gradient is added into columns
  `0:512`, the rope gradient into columns `512:576`; further columns are never
  touched (a `[S_dst, 704]` buffer holding a frozen 128-channel indexer key next
  to the 576 channels is fine).  Row stride `>= 576` elements and a multiple of
  4, 16-byte-aligned base; the 576-column view of a wider buffer is accepted.
  Accumulation is `+=`: the caller zeroes the buffer when it wants fresh
  gradients, and two backward calls into the same buffer sum.
* `dkv_dst_map`: optional contiguous int32 `[S]` giving the destination row of
  every source key row (values in `[0, S_dst)`, not checked on device;
  duplicates allowed and summed).  Without a map the identity is used, which
  needs `S_dst >= S`.
* With `dkv_acc` the backward returns `(dq_latent, dq_rope, None, None)`, the
  autograd wrapper returns `None` gradients for `kv_latent` / `k_rope`, and
  `dkv_fp32=True` is rejected.  The kernels' permuted FP32 accumulators remain
  per-call scratch; `bwd_cast` un-permutes, remaps and adds in one pass
  (`accumulate = 1`).  With a map the adds are `red.global.add.v4.f32`, whose
  order is not fixed for repeated destination rows (an injective map is still
  exactly one add per element); without a map every 16-byte vector is a
  load-add-store, bitwise `previous + value`.
* Requires a registered program whose `bwd_cast` argument plan declares
  `dst_packed, dst_row_stride, dst_map, has_dst_map, accumulate`
  (`cake_backend.record_cast_accumulates`); an older program raises
  `NotImplementedError`.  The binding cache keys the backward on `(data_ptr,
  shape, stride, dtype)` of `dkv_acc` and `dkv_dst_map` as well (the row stride
  and the presence of a map are baked into the cast's launch).

## Kernel structure of one training step

* `fwd`: one CTA per query token gathers its top-k keys once (TMA gather) and
  produces `out`, the natural-log `lse` and the BF16 output residual `o_lo`.
* `bwd_delta`: `delta = rowsum(dO * (O + O_lo))`, one (token, head) row per warp.
* `bwd_main`: one CTA per query token (20 warps: gather, compute, reduce, MMA,
  load and metadata roles) recomputes S and P from the BF16 Q and the gathered
  K, forms dP and dS, accumulates dQ / dQ_rope in tensor memory (written once
  per row: bitwise deterministic) and scatters the per-token dK/dV and dK_rope
  contributions with vectorized FP32 `red.global.add` into the accumulators.
* `bwd_compact` / `bwd_main_pass`: the key-range-pass form of the main stage
  for the DRAM regime (see below).  Per pass, `bwd_compact` (one warp per
  token) writes the token's keys inside the pass range, in slot order and
  under the main kernel's validity rules (`-1`, `>= S`, `topk_length`), to
  `key_scratch` with their count in `pass_counts`; `bwd_main_pass` is the same
  main kernel consuming that list, carrying the token's FP32 dQ / dQ_rope
  partial in `dq_partial` between passes (`dq_mode` 1: store, 2: load-add-
  store, 3: load-add and BF16 output).
* `bwd_cast`: converts the FP32 accumulators to the natural `[S, 512]` /
  `[S, 64]` BF16 outputs (or FP32 in the `dkv_fp32` mode), or adds them into
  the caller's packed FP32 rows through the optional destination-row map
  (`dkv_acc` / `dkv_dst_map`).

Grid rules live in the registry record (`num_queries` CTAs for `fwd`,
`bwd_main` and `bwd_main_pass`, `num_queries*8` for `bwd_delta`,
`num_queries/4` for `bwd_compact`, `num_kv*18/256` for `bwd_cast`) and are
evaluated by the host from the problem scalars.

### Key-range passes for the DRAM regime

With many keys the FP32 dK/dV accumulators (2304 B per key) outgrow the L2,
and the `red.global.add` scatter of `bwd_main` runs at DRAM speed.  The host
then runs the backward in `P = ceil(S * 2304 B / 100 MiB)` passes over
disjoint key ranges (`R = ceil(S / P)` keys each), so that the accumulator
slice one pass touches stays L2-resident: `bwd_delta`, then per pass
`bwd_compact` + `bwd_main_pass` over the whole row, then `bwd_cast`.  The
policy the record carries (`key_pass_policy`: L2 budget 100 MiB, 2304 B per
key, workspace budget 640 MiB, token chunk multiple 128) takes the pass path
only when `P > 1` and the whole row fits the pass workspace budget
(`T <= 4224` tokens at top-k 2048; there is no token chunking); otherwise the
single-pass `bwd_main` runs unchanged.  At top-k 2048 that is `T <= 4224` and
`S >= 45,512`: two passes at `S = 65,536`, three at `131,072`; 4k x 4k rows
and 32k-token rows stay single-pass.  The passes add
`T * (147,456 + 4 * topk + 4)` B to the workspace (`dq_partial`,
`key_scratch`, `pass_counts`; 608 MiB at `T = 4096`, top-k 2048;
`dsa_train_workspace_size` includes them).  dQ is
still written once per row from the carried FP32 partial (bitwise
deterministic run to run; its partial sums are re-associated, so it differs
from the single pass in the last FP32 places), and the dK/dV reductions are
the same reds in another order.  `key_passes=` on the entry points overrides
the policy (`1` = single pass, `n` = that many passes); a program without the
pass stages serves the single pass only.

## Layout of this package

* `cake_jit.py` -- `MODULES` registry (one record per architecture, filled by
  the generated-program export), stage names and the JIT specs.
* `cake_backend.py` -- validation, workspace layout, varlen index offsetting,
  argument-plan binding, the prepared runner, the autograd `Function` and the
  eager entry points.
* `csrc/cake_dsa_h64_train/<arch>/` -- generated kernel and binding
  translation units (`.clang-format` disables formatting: the sources are
  identity-checked by the registry's closure digests).

## Status

The registry holds one record per architecture (`sm_100a`, `sm_103a`, `sm_107a`) with the
forward, backward preprocess, backward main (single-pass and key-range-pass
form with its compaction) and cast stages plus the key-range-pass policy,
exported from the kernel snapshot named in the pull request.  The host
binding supports two argument profiles, selected by the record's `abi` field:
`dsa_h64_v1` (the native kernels) and `flashmla_v41_prefill_seed` (a
forward-only FlashMLA-derived prefill program used to exercise the export
pipeline; it produces no output residual, so backward is unavailable with it).

Tests: `tests/experimental/test_cake_dsa_train.py` (skips without a registered
program or a compute capability 10.0 / 10.3 / 10.7 device).  Benchmark:
`benchmarks/bench_cake_dsa_train.py`.
