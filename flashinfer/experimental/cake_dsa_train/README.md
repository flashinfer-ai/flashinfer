# Cake DSA sparse-attention training backend (SM100 / SM103 / SM107)

Experimental backend for the native 64-query-head DeepSeek Sparse Attention
training kernels (top-k sparse MLA with absorbed queries; GLM-5.2 geometry:
64 heads, 512 latent + 64 rope query/key dimensions, 512 value dimensions,
top-k 2048).  Tracking issue: flashinfer-ai/flashinfer#5657.

Both the API and the backend are experimental: no compatibility guarantee,
SM100 (B200 / GB200) is the acceptance architecture of the native kernels and
SM103 (B300 / GB300) is compiled from the same sources; SM107 (Rubin R200) is
the acceptance architecture of the packed / strided-layout extension
(flashinfer-ai/flashinfer#5675) and is compiled from the same sources with an
nvcc that emits `compute_107a`.  The feature does not participate in
automatic backend selection, autotuning or trace-apply.  Its libraries are
loaded prebuilt from `aot/` when the package ships them and JIT-built by
FlashInfer from `build.json` otherwise (see "Layout of this package").

## Public entry points (`flashinfer/dsa_sparse_attention.py`)

```python
from flashinfer.dsa_sparse_attention import dsa_sparse_attention, dsa_sparse_attention_varlen

out = dsa_sparse_attention(q_latent, q_rope, kv_latent, k_rope, indices,
                           topk_length=None, softmax_scale=None, return_lse=False,
                           dkv_acc=None, dkv_dst_map=None)
out = dsa_sparse_attention_varlen(q_latent, q_rope, kv_latent, k_rope, gather_kv_indices,
                                  cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                                  causal=False, topk_length=None, softmax_scale=None,
                                  return_lse=False, dkv_acc=None, dkv_dst_map=None)
```

* `q_latent [T, 64, 512]`, `q_rope [T, 64, 64]` BF16; `kv_latent [S, 512]`
  (K = V) and `k_rope [S, 64]` BF16.  Strided views are consumed in place --
  see "Packed and strided inputs" below.
* `indices [T, topk]` int32 (any row stride) hold **global** key rows; `-1` or `>= S` marks an
  invalid slot anywhere in the row; `topk_length [T]` int32 optionally
  invalidates slots `>= topk_length[t]`; any positive `topk`.  Without
  `topk_length` the route follows the table size (`NOPE_CONVERT_SLOT_LIMIT`,
  `nope_flat_index_mode`): below the limit the forward kernel derives each
  row's length (last valid slot + 1) itself while it runs and the training
  entry hands that vector to the backward with the rows as given -- one
  launch, a `-1` anywhere stays a masked slot, both kernels skip only the
  trailing invalid key blocks; at or above it the index-preparation stage
  compacts the rows first (valid keys first in ascending slot order, then
  `-1`, the valid count as the length) into a fresh buffer the forward and the
  backward read, so rows with `-1` holes cost their valid slots.  The results
  are those of the full row either way; hole-free rows give the same bytes on
  both routes.
* The varlen form takes per-document indices (`gather_kv_indices`, relative
  to `cu_seqlens_k[d]`) and turns them into global key rows on device.  The
  NoPE entry follows the same size rule (`nope_varlen_index_mode`): below the
  limit, without caller lengths, the forward kernel converts the rows itself in
  its single launch, position-preserving -- the global key or `-1` at the slot
  it came from, the length the last valid slot + 1, bitwise the torch twin
  `offset_gather_kv_indices(...)`; at or above the limit, and always with a
  caller `topk_length`, one launch of the NoPE record's index-preparation
  stage right before the forward kernel (`index_prep`, `compact = 1`, the
  forward launched behind it with the programmatic-dependent-launch attribute,
  see below) compacts every row stably: its valid keys come first in ascending
  slot order, then `-1`, and the row's length is its valid count (bitwise the
  torch twin `offset_gather_kv_indices(..., compact=True)`), so rows with `-1`
  holes cost their valid slots in the forward and the backward.  The rope varlen
  entry always launches the compacting stage (the torch twin where the stage
  is not available -- host tensors, an unserved device, a non-contiguous prefix
  sum -- with bitwise the same result).  Rows whose valid slots already come
  first are unchanged bytes with the same length on every route.  A caller
  `topk_length` is a position cut applied before compaction (slots at positions
  `>= topk_length[t]` are invalid regardless of content).  `cu_seqlens_q`
  and `cu_seqlens_k` are independent: a query segment may be shorter than its
  key segment, in which case it is the tail of that key prefix -- query
  `local_q` of document `d` sits at key position
  `(seqlen_k[d] - seqlen_q[d]) + local_q`.  With `causal=True` the offsetting
  also drops selected keys after that position
  (`offset_gather_kv_indices(..., causal=True)` keeps a slot iff
  `0 <= idx < seqlen_k[d]` and `idx <= (seqlen_k[d] - seqlen_q[d]) + local_q`);
  with `causal=False` (the default) a query's key set is exactly its index row.
  Either way the kernels apply no positional mask of their own: the offset
  index rows define the key sets.  A zero-length query segment contributes no
  rows; a row whose every slot ends up `-1` gives `out = 0`, `lse = -inf` and
  zero gradients.  The segment count `len(cu_seqlens_k) - 1` is passed on as
  `num_segments`: the backward's whole-row key-range passes apply to
  one-segment rows only (see "Key-range passes").
* Forward: `out [T, 64, 512]` BF16 and (with `return_lse=True`) the natural-log
  `lse [T, 64]` FP32 over the valid keys; fully masked rows give `out = 0` and
  `lse = -inf`.  The autograd wrapper also keeps the BF16 output residual
  `o_lo = bf16(fp32(O) - bf16(O))` for the backward's exact `delta`.
* Backward: `dq_latent`, `dq_rope` (computed once per row, bitwise
  deterministic), `dkv_latent [S, 512]`, `dk_rope [S, 64]` (FP32 `red.global`
  accumulation, then a cast to natural-layout BF16).  BF16 MMA operands, FP32
  accumulation throughout.  `cake_backend.backward(..., dkv_fp32=True)` (and a
  runner prepared with `dkv_fp32=True`) returns the dK/dV gradients as
  natural-layout FP32 tensors instead; the kernels' internal permuted
  accumulator layout never crosses the API boundary -- the cast stage serves
  both modes.  With a caller-provided `dkv_acc` (see "Packed fp32 dKV
  accumulation and destination mapping" below) the dK/dV gradients are
  accumulated into that packed FP32 buffer instead, no `dkv_latent` /
  `dk_rope` tensors are produced and the autograd wrapper returns no gradient
  for `kv_latent` / `k_rope`.

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
* `indices` rows stay contiguous int32 (any row stride; a column slice of a
  wider index buffer is accepted); `dout` follows the layout rules of
  `q_latent` and is read in place -- the head-major `[512, T * 512, 1]` view an
  output projection hands back included -- by the main stage (stride-carrying
  tensor map) and by `bwd_delta` (the strided delta kernel shared with the
  NoPE record: `dout_offset + token * dout_token_stride + head *
  dout_head_stride + channel`).  The eager `backward` copies a `dout` only when
  the descriptor cannot address it (base not 16-byte aligned, strides not
  multiples of 8 elements, channel stride != 1); `prepare_dsa_train` rejects
  such a view.
* A strided view yields bitwise the same `out`, `lse` and `dq` as the
  contiguous copy (only descriptor fields and pointer offsets change); dK/dV
  agree within the FP32 `red.global` reduction spread.

### Packed fp32 dKV accumulation and destination mapping

`cake_backend.backward(..., dkv_acc=, dkv_dst_map=)`, `prepare_dsa_train(...,
dkv_acc=, dkv_dst_map=)`, the public entries and the autograd `Function`
(`DSASparseAttentionFunction.apply(q_latent, q_rope, kv_latent, k_rope, indices,
topk_length, softmax_scale, key_passes, dkv_acc, dkv_dst_map, num_segments)`)
accumulate the dK/dV gradients directly into a caller-provided FP32 buffer
instead of returning `dkv_latent` / `dk_rope` (flashinfer-ai/flashinfer#5675:
the packed dKV of a trainer, with the repeated / remapped rows of a
context-parallel window):

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
* The kernel does not range-check `dkv_dst_map`: every value must lie in
  `[0, S_dst)` (the rows of `dkv_acc`), and a value outside that range would
  add into memory outside the buffer.  The keyword `check_dst_map` of the
  entry points (default `False`) validates the values on every call when
  `True` (one device synchronization) and raises `ValueError` on a violation;
  leave it off in production.  The backend reads no environment variable:
  flashinfer's public wrapper `flashinfer.dsa_sparse_attention` maps
  `FLASHINFER_CAKE_DSA_CHECK_DST_MAP` onto the keyword when the caller leaves
  it at `None`.
* Through the autograd entries `dkv_acc` may also be a zero-argument callable
  that returns the accumulator, or an `(accumulator, dkv_dst_map)` pair; the
  backward calls it, so the buffer need not exist at forward time (a callable's
  result is validated in the backward, where `check_dst_map` then
  synchronizes).  The autograd `Function`s save every tensor their forward
  produces for the backward -- `out`, `lse`, `o_lo`, the index rows, the
  lengths -- through `ctx.save_for_backward` (a fixed list with a zero-size
  placeholder for an absent `topk_length`, so non-reentrant
  `torch.utils.checkpoint` can match the recomputed tensors by position) and
  keep nothing else of size on `ctx`: after a checkpointed forward the entries
  retain ~0 MiB (flashinfer-ai/flashinfer#7039).  A tensor `dkv_acc` (and
  `dkv_dst_map`) is caller-owned and stays on the autograd node as before,
  outside the saved list: zeroing or otherwise touching it in place between
  the forward and the backward is fine.
* With `dkv_acc` the backward returns `(dq_latent, dq_rope, None, None)`, the
  autograd wrapper returns `None` gradients for `kv_latent` / `k_rope`, and
  `dkv_fp32=True` is rejected.  The kernels' permuted FP32 accumulators remain
  per-call scratch; `bwd_cast` un-permutes, remaps and adds in one pass
  (`accumulate = 1`).  With a map the adds are `red.global.add.v4.f32`, whose
  order is not fixed for repeated destination rows (an injective map is still
  exactly one add per element); without a map every 16-byte vector is a
  load-add-store, bitwise `previous + value`.
* The generated `bwd_cast` launcher takes the packed-accumulate operands
  (`dst_packed, dst_row_stride, dst_map, has_dst_map, accumulate`); a plan
  that does not accumulate passes inert placeholders with `accumulate = 0`.
  The eager plan cache keys the backward on the geometry (shape, strides,
  dtype, device, alignment) of `dkv_acc` and `dkv_dst_map` as well -- the row
  stride and the presence of a map are launch constants of the accumulating
  launch -- and the caller's buffer and map are re-supplied per call.

#### Direct accumulation from the main stage

A program may register natural-layout variants of the main stage
(`bwd_main_natural`, and `bwd_main_pass_natural` next to the key-range-pass
form): the same kernel traced with a drain that transposes each key's FP32
contributions across lanes and `red.global.add`s them at their natural
position straight into the caller's packed rows (`dkv_acc` through
`dkv_dst_map`).  Such a binding has no FP32 accumulators, no zero fill and no
`bwd_cast` launch (`dsa_train_workspace_size(..., dkv_acc=True)` is smaller by
`S * 2304` B, the runner's `dkv_direct` is set).  The lane transpose costs the
reduce-bound rows per tile, and only rows with many keys per query token pay
it back (B200 backward: `S / T >= 16` 0.5-4 % faster, `S = T` 3.7 % slower), so
the host takes the direct path by the size rule the record carries
(`dkv_direct`: `min_keys_per_query`, 4 -- direct iff `S >= 4 T`) whenever
`dkv_acc` is given; the keyword `dkv_direct` of the entry points (default
`None` = the rule) forces (`True`) or disables (`False`) it, and the resolved
value is part of the binding key (flashinfer's public wrapper maps
`FLASHINFER_CAKE_DSA_DKV_DIRECT=auto|1|0` onto the keyword when the caller
leaves it at `None`).  Both
paths add the same FP32 contributions into the same rows in another summation
order; dq is bitwise the same.  Without the natural stages every `dkv_acc`
call takes the cast's accumulate path.

Explicit forward / backward entry points without autograd, a prepared
allocation-free runner (`prepare_dsa_train`, CUDA-graph capturable) and the
workspace sizing helper live in `cake_backend.py`.  A binding is validated and
resolved once per input geometry (shapes, strides, dtypes, device, alignment
and the call options): the prepared runner carves its scratch from the
workspace it is given; the eager entry points keep the resolved plans in a
bounded, lock-protected cache (`BINDING_CACHE`, 64 entries by default,
`BINDING_CACHE.configure(capacity=...)` resizes it -- flashinfer's public
wrapper applies `FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY` through it
when that variable is set; least recently used first out; a plan owns only the
full-length `topk_length` vector it materializes when the caller passes none)
and, per call, allocate the outputs
and the backward scratch from the caching allocator and launch the stages
through the generated positional launchers of `cake_launch.py`.  During
CUDA-graph capture the eager entry points plan privately and leave the cache
untouched.  The bindings encode the tensor maps by value, so a step is its
kernels -- one launch for the forward; three for the single-pass backward
(`bwd_delta`, `bwd_main`, `bwd_cast`); `2 + 2 x chunks x passes` for the
key-range-pass backward (`bwd_delta`, then `bwd_compact` + `bwd_main_pass` per
token chunk and pass, `bwd_cast`)
-- plus the two fills of the FP32 dK/dV accumulators; a direct binding (see
"Direct accumulation from the main stage") launches the natural-layout variant
of the main stage instead and has neither the fills nor the cast.  A call without query rows
(`T == 0`) returns empty outputs and zero gradients without launching; `S == 0`
is rejected.

Host cost through the autograd wrapper: the `Function.backward` runs on
PyTorch's autograd device thread, where the two thread handoffs and the Python
body add a few hundred microseconds per backward that are not in this package
-- a trivial `autograd.Function` with the same saved tensors and gradient
shapes shows the same cost, and no synchronization is involved.  In a
GPU-bound training step this is hidden behind the backward kernels (6-8 ms at
4k tokens).  Host-bound loops should call `cake_backend.forward` /
`cake_backend.backward` directly or capture the prepared runner into a CUDA
graph (`benchmarks/bench_cake_dsa_train.py --host-us` reports both; `--host-calls`
/ `--host-rounds` set the sample size).

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
* `bwd_main_natural` / `bwd_main_pass_natural`: the main stage (single-pass /
  pass form) whose reduce warps add straight into the caller's packed FP32
  rows (see "Direct accumulation from the main stage"); selected instead of
  `bwd_main` / `bwd_main_pass` for a `dkv_acc` binding that meets the record's
  size rule, with no `bwd_cast` afterwards.  The reduce traffic of every main
  stage carries an L2 `evict_last` policy for the accumulator lines.

Launch grids are functions of the problem scalars in `cake_launch.py`
(`num_queries` CTAs for `fwd`, `bwd_main`, `bwd_main_pass` and their
natural-layout variants,
`num_queries * 8` for `bwd_delta`, `ceil(num_queries / 4)` for `bwd_compact`,
`ceil(num_kv * 18 / 256)` for `bwd_cast`).

### Key-range passes for the DRAM regime

With many keys the FP32 dK/dV accumulators (2304 B per key) outgrow the L2,
and the `red.global.add` scatter of `bwd_main` runs at DRAM speed.  The host
then runs the backward in `P = ceil(S * 2304 B / 100 MiB)` passes over
disjoint key ranges (`R = ceil(S / P)` keys each), so that the accumulator
slice one pass touches stays L2-resident: `bwd_delta`, then per pass
`bwd_compact` + `bwd_main_pass` over the whole row, then `bwd_cast`.  The
policy the record carries (`key_pass_policy`: L2 budget 100 MiB, 2304 B per
key, workspace budget 640 MiB, token chunk multiple 128, at most
`max_passes = 4` passes, and one rule per architecture under `rules`) takes
the pass path only for `P > 1` and under the rule of the device's
architecture, measured in paired sweeps per target:

| target | tail rows (`2 T <= S`) | passes | floor (one-chunk row / tail row) | tail cap |
|---|---|---|---|---|
| `sm_100a` | no | `P` | - / - | - |
| `sm_103a` | yes | `P` | - / 131,072 keys | 3 |
| `sm_107a` | yes | 2 | 131,072 / 131,072 keys | - |

A row qualifies when it is one token chunk (`T <= 4224` tokens at top-k
2048: the chunk is the largest multiple of 128 whose pass scratch fits the
workspace budget) or, where the rule admits tail rows, when `2 T <= S`; a
qualifying row below its floor stays single-pass, and the count is the rule's
fixed value or `P`, capped for tail rows of more than one chunk.  So at top-k
2048 on `sm_100a`: `T <= 4224` and `S >= 45,512`, two passes at `S = 65,536`,
three at `131,072`, 4k x 4k rows and 32k-token rows single-pass; on `sm_103a`
also three passes on the 32k-token rows of 131,072 or more keys; on `sm_107a`
two passes on every qualifying row of 131,072 or more keys and one pass
below.  A count above `max_passes` runs one pass: beyond four passes the
per-pass FP32 dQ-partial round trip and pipeline fill outweigh the L2 benefit
(the `S = 225,280` / `267,520` one-chunk rows run one pass on `sm_100a` and
`sm_103a`; `sm_107a`'s two passes are under the cap).  A record without
`rules` (the first release) is the `sm_100a` rule everywhere.  The tokens of a multi-pass backward run in chunks of 4224 (every
chunk through every pass, `token_base` = the chunk's first token), so the pass
scratch holds one chunk.  The passes split the whole key row
into equal ranges, which matches an index row whose keys spread over the whole
row (one document); in a packed multi-segment row every token's keys lie
inside its own segment, so the policy applies to one-segment rows only
(`num_segments > 1`, the varlen entry's `len(cu_seqlens_k) - 1`, plans one
pass unless `key_passes` overrides).  The passes add
`min(T, 4224) * (147,456 + 4 * topk + 4)` B to the workspace (`dq_partial`,
`key_scratch`, `pass_counts`; 608 MiB at `T = 4096`, top-k 2048;
`dsa_train_workspace_size` includes them).  dQ is
still written once per row from the carried FP32 partial (bitwise
deterministic run to run; its partial sums are re-associated, so it differs
from the single pass in the last FP32 places), and the dK/dV reductions are
the same reds in another order.  `key_passes=` on the entry points overrides
the policy (`1` = single pass, `n` = that many passes); a program without the
pass stages serves the single pass only.

## NoPE (zero-rope) program

A second registry record, `cake_dsa_h64_nope_train`, carries the native kernels
of the zero-rope form of the same attention (GLM-5.3 geometry: 64 heads, 512
shared latent dimensions and no rope part, softmax scale `1 / 16`, top-k slot
counts such as 2176 or 1152 -- any `K_slots >= 1`).  The score is the latent
product alone, `softmax(scale * q . kv^T) . kv`, and the backward produces one
combined gradient per key row, `dkv = dK + dV`, natural layout, FP32 `[S, 512]`.
The rope entry points above are unchanged; the NoPE entries are additive.

Slot-count rule: the host accepts every int32 `indices [T, K_slots]` with
`K_slots >= 1` (any width, odd widths included; `topk_length` may be shorter
per row) and rejects `K_slots <= 0`, a non-int32 dtype and a row stride below
`K_slots` explicitly (`validate_dsa_nope_inputs`); the kernels bound the per-row
work by `topk_length` and read no slot beyond `K_slots`.

```python
from flashinfer.dsa_sparse_attention import dsa_sparse_attention_nope, dsa_sparse_attention_nope_varlen

out = dsa_sparse_attention_nope(q, kv, indices, topk_length=None, softmax_scale=None,
                                return_lse=False, dkv_acc=None, dkv_dst_map=None)
out = dsa_sparse_attention_nope_varlen(q, kv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k,
                                       max_seqlen_q=None, max_seqlen_k=None, causal=False,
                                       topk_length=None, softmax_scale=None, return_lse=False,
                                       dkv_acc=None, dkv_dst_map=None, check_cu_seqlens=True)
```

* `q [T, 64, 512]` BF16 (contiguous or a strided token / head view, consumed in
  place); `kv [S, 512]` BF16 or its one-head form `[S, 1, 512]` (squeezed on the
  host; the gradient reaches that leaf through the view); `indices [T, K_slots]`
  int32 global key rows, `topk_length` and the validity rules as above.
  `softmax_scale` defaults to `1 / 16`.  Without `topk_length` the flat entry
  follows the table size (`nope_flat_index_mode`): below
  `NOPE_CONVERT_SLOT_LIMIT` slots the forward alone reads the rows as given and
  derives each row's length (last valid slot + 1) into a `[T]` vector the
  backward reads with the caller's rows (one launch; the caller's tensor is
  then a saved tensor, so autograd rejects an in-place edit of it before the
  backward); at or above the limit it compacts the rows on device in one launch
  of the index-preparation stage's flat mode right before the forward (valid
  keys `0 <= idx < S` first in ascending slot order, then `-1`; the valid count
  as the row's length) into a fresh `[T, K_slots]` buffer and a `[T]` vector
  the backward reads, so rows with `-1` holes cost their valid slots and the
  caller's tensor is untouched.  With `topk_length` the rows and lengths reach
  the kernels as given.
* Rows with holes and the kernel-level entries: `forward_nope`,
  `backward_nope` and the prepared runner read rows as given (work up to the
  last valid slot, or `topk_length`).  `nope_compact_indices(indices, S,
  topk_length=None)` returns `(rows, counts)` -- one flat stage launch, any
  row stride, optional `out=` / `counts=` buffers for a capturing caller -- to
  pass as `indices` / `topk_length` to those entries and to the flat autograd
  entry with caller lengths.  (A backward over raw hole rows could instead run
  the key-range-pass form at one pass -- `nope_bwd_compact` writes each row's
  valid keys compacted, `nope_bwd_main_pass*` consumes them with `dq_mode 0` --
  which is not wired up in this round.)
* Backend entries (`cake_backend`): `forward_nope`, `backward_nope`,
  `forward_nope_varlen` (the packed forward without autograd: the public
  entry's index route; returns `(out, lse, o_lo, indices, topk_length)`, the
  global rows and lengths of that route -- position-preserving rows with last
  valid slot + 1 below the slot limit, compacted rows with their valid counts
  on the stage route -- that `backward_nope` takes with
  `num_segments = cu_seqlens_k.numel() - 1` and a caller-zeroed `dkv_acc`),
  `nope_compact_indices`, `nope_globalize_indices(..., compact=False)` (one
  stage launch over per-document rows: the position-preserving rows and last
  valid slot + 1 by default, bitwise `offset_gather_kv_indices`; `compact=True`
  the compacted rows and counts, with an optional `topk_length` cut),
  `prepare_dsa_nope_train` / `DSANopeTrainRunner`, `dsa_nope_train_workspace_size`,
  `validate_dsa_nope_inputs`, `DSASparseAttentionNopeFunction` (12 forward inputs,
  gradients for `q` and `kv` only).  The binding cache is shared; the NoPE keys are
  tagged (`"nope_fwd"` / `"nope_bwd"`) and never collide with the rope keys.
* `dkv` semantics.  Fresh (no `dkv_acc`): `backward_nope` returns a new natural
  FP32 `[S, 512]` buffer -- filled by the natural-layout main stage adding into
  the zero-filled buffer when the record's direct rule selects it (`dkv_direct`:
  `min_keys_per_query` 1, so whenever the row has at least one key per query --
  every packed causal problem; measured faster than the permuted drain on every
  row), else -- more query rows than keys, or `dkv_direct=False` -- by the cast
  stage writing the permuted accumulator out densely (`out_f32`).  Accumulate
  (`dkv_acc` FP32 `[S_dst, >= 512]`, optional `dkv_dst_map`): `+=` into columns
  `0:512` of the caller's rows by the same two
  paths and `None` for `dkv`; chunked backward passes of one long row sum into
  one buffer without intermediate rounding.  Through autograd without
  `dkv_acc`, the FP32 `dkv` becomes the `kv` gradient and the autograd engine
  casts it to the leaf's dtype (BF16 for a BF16 leaf); `dkv_acc` may be the
  tensor or a zero-argument callable producing it (or an `(accumulator,
  dkv_dst_map)` pair) at backward time, as for the rope entries.  Both NoPE
  autograd `Function`s save every produced backward operand -- the saved
  outputs, `o_lo`, the global index rows, the lengths -- through
  `ctx.save_for_backward` (seven tensors in a fixed order) and keep only
  options, geometry keys and the caller-owned accumulator on `ctx`, so a
  checkpointed forward retains nothing (flashinfer-ai/flashinfer#7039).
* `dout` may be a head-major view (strides `[512, T * 512, 1]`) or any strided
  `[T, 64, 512]` view under the layout rules of `q` (unit channel stride, token
  and head strides multiples of 8 elements, 16-byte base): it is read in place
  by every entry, the eager `backward_nope` and a prepared runner alike -- the
  main stage through a stride-carrying tensor map, the delta stage through the
  token / head strides (elements) and the element offset the launcher passes
  (`dout_token_stride`, `dout_head_stride`, `dout_offset = 0`).  No host copy.
* Varlen: per-document indices become global key rows on device; the route
  follows the table size (`nope_varlen_index_mode`).  Below
  `NOPE_CONVERT_SLOT_LIMIT` slots (`1 << 20`; both small bench tables, 1024 x
  128 and 1024 x 512, sit below it) and without caller lengths the forward
  converts the rows itself in its single launch (`convert = 2`): its scan lanes
  write the position-preserving global rows -- the global key or `-1` at the
  slot it came from -- into the fresh `indices_out` the backward reads and the
  last valid slot + 1 as the row's length (bitwise the torch twin
  `offset_gather_kv_indices(...)`), so no host gap separates a preparation
  launch from the forward; a small table with holes scans them like the
  previous version did (routing reads the table size alone: the host cannot
  see holes without a device sync).  At or above the limit, and always with a
  caller `topk_length`, one launch of the record's index-preparation stage runs
  right before the forward: `index_prep` (one warp per index row, four rows per
  128-thread CTA, any index row stride read in place) with `compact = 1`, every
  row stably compacted -- valid keys first in ascending slot order, then `-1`
  -- and its valid count as the length, after the position cut of a caller
  `topk_length` passed as `length_in`; the forward and the saved backward read
  the stage's rows and counts (bitwise the torch twin
  `offset_gather_kv_indices(..., compact=True)`).  The limit is a measured
  crossover (the largest table at which the one-launch forward stays within
  0.95x of the stage route on the eager forward and 1.02x on the GPU-busy
  step, plus one).  Both forward
  stages (`fwd`, `nope_fwd`) launch with the programmatic-dependent-launch
  attribute (`use_pdl` in their launch records, a fixed attribute of their
  bindings): `index_prep` signals `launch_dependents` at entry and the forward
  kernels execute `griddepcontrol.wait` after their barrier / TMEM prologue,
  before any global read, so behind the stage the forward's prologue overlaps
  the stage; behind any other kernel the attribute changes nothing.  The host prepares
  the forward's plan, outputs and launch arguments and converts its operands
  before the first launch, then issues the forward from those pre-bound values
  inside the same stream context, so only the forward's binding call sits
  between the two launches (it hides behind the stage's kernel).  The forward
  program's conversion operands (`cu_seqlens_q`, `cu_seqlens_k`,
  `indices_out`, `num_docs`, `search_iters`, `causal`, `convert` at the end of
  its launch arguments) take three values: `convert = 0` for global rows (the
  operands never read), `convert = 2` (`NOPE_CONVERT_MASK`) for the
  position-preserving conversion of the small-table route, and `convert = 1`
  (`NOPE_CONVERT_COMPACT`) for the compacting single launch -- the same stable
  compaction and valid counts as the stage, bitwise -- which no size selects
  (`_packed_forward_nope(..., "kernel")` by name only).  A registry without the
  stage takes the position-preserving single launch at every size and no
  caller `topk_length`.  `causal=True` drops keys after the query's own
  position, query segments are tails of their key prefix.  The NoPE entries
  have no torch fallback.
* Backward host order: the autograd wrappers settle their backward at forward
  time (the binding key without `dout`, the operands autograd does not save,
  the caller's `dkv_acc` validated once), so the backward's host work before
  its first launch is the key completion with `dout`'s geometry, the binding
  lookup, the `dq` / `delta` allocations and the value binding; the zero fills
  of the FP32 buffers the call owns and the delta / main stages then follow
  back to back, with no allocation, validation or synchronization between the
  launches.  The kernel-level `backward_nope` keeps the same launch order.
  Every call checks the structure of the prefix sums (`validate_cu_seqlens`:
  contiguous int32 `[num_docs + 1]` of one length) and never synchronizes with
  the device; `check_cu_seqlens=True` also checks their values -- start at 0,
  non-decreasing, `cu_seqlens_q[-1] == T`, `cu_seqlens_k[-1] == S` (a shorter
  or longer key window is rejected, not truncated), `seqlen_q <= seqlen_k` per
  document -- at the cost of one blocking device-to-host copy before any launch
  (skipped during CUDA-graph capture).  The rope varlen entry launches the same
  stage (compacting, caller lengths as the cut) and keeps the torch twin as its
  fallback; its structural checks are unchanged.

Stages of the record, in launch order and with the kernels they share with the
rope program: `nope_fwd` (own kernel, 320 threads), `nope_bwd_delta` (the
strided delta kernel, shared with the rope program's `bwd_delta`, 256 threads:
the `[T * 64, 512]` rows of `dout` addressed as `dout_offset + token *
dout_token_stride + head * dout_head_stride + channel`, strides and offset
multiples of 8 elements; with contiguous strides it is bitwise the dense form
it replaced),
`nope_bwd_main` /
`nope_bwd_main_natural` (own kernels, 640 threads: no rope operands, one
accumulator `dkv_f32` with `dkv_stride >= 512`, a multiple of 4, and the
destination map; a persistent token loop, so the host launches them through
`nope_bwd_launch_grid(num_queries, num_kv, sm_count)`: one CTA per SM --
`min(num_queries, SMs)`, CTA `b` taking tokens `b, b + grid, ...` of the
launch's `num_queries` tokens (the whole row or the chunk) -- while the launch's
FP32 `dkv` window `num_kv x 2048 B` is within 64 MiB, else one CTA per token;
the SM count is read once per device), `nope_bwd_compact` (the rope compaction kernel),
`nope_bwd_main_pass` / `nope_bwd_main_pass_natural` (the key-range-pass forms)
and `nope_bwd_cast` (the rope cast kernel with `rope_groups = 0`, grid
`ceil(num_kv * 16 / 256)`).  `index_prep` (own kernel, 128 threads, grid
`ceil(T / 4)`; arguments `compact`, `length_in`, `has_length_in` after
`write_out`) is the index preparation of both varlen entries (document mode,
`compact = 1`), of the flat NoPE entry without caller lengths and of
`nope_compact_indices` (flat mode: `has_docs = 0`, `flat_bound = S - 1`, the
prefix-sum operands never read); it is launched right before the forward and
is not part of a prepared runner's step.  The key-range-pass policy of the record plans one
pass on every row (`max_passes` 1, no per-architecture rules: one pass measured
fastest on every row, build and GPU of this form), so the single-pass main stage
is the default and `dq_partial` is not allocated; `key_passes=n` on the entry
points forces the pass form, which plans with 2048 B of accumulator per key and
a dQ partial of 131,072 B per token: the token chunk of a multi-pass backward is
4736 at `K_slots` 2048 / 2176, 4864 at 1152 and 4992 at 128 (the 640 MiB budget,
a multiple of 128).

`generated_program_available(..., program=cake_jit.NOPE_PROGRAM)` reports
whether the record registers the stages a call needs for the device; the NoPE
GPU tests additionally skip while `build.json` lacks a registered NoPE library
(the export delivers the libraries; `scripts/check_cake_build_manifests.py`
reports that gap).

## Layout of this package

* `cake_jit.py` -- the `MODULES` registry (one record per program, the rope
  `cake_dsa_h64_train` and the NoPE `cake_dsa_h64_nope_train`, each shared by
  every architecture it compiles for; per stage the `library` stem of its
  `build.json` entries, the FFI entry and the launch geometry -- block,
  cluster, cooperative, dynamic shared memory, PDL -- that the export rewrites),
  the stage names of both programs (`STAGES` / `NOPE_STAGES`, `PROGRAMS`) and
  the loader glue: `library_name(stage, arch, program)` is the `build.json`
  entry `<library>_<arch>` that `load_cake_dsa_train_module(stage, arch,
  program)` loads through `cake_build`; `specs_for_arch(arch)` lists the JIT
  specs of an architecture for `flashinfer.jit.build_jit_specs`.  Stages of
  the two programs that share a kernel name the same library stem.
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package (byte-identical copies) and this package's build manifest (schema
  `cake-build/1`, written by the export): one entry per `<library>_<arch>`
  (fourteen libraries x three architectures) with its translation units, digests,
  nvcc flags (`--use_fast_math` per stage), closure digest and `aot/<name>.so`
  path.  `cake_build.load_library(name)` loads the prebuilt library when
  `aot/` holds it (next to the builder's `aot/build_info.json` sidecar, which
  must agree with the library's exported `build_cuda_version()`) and otherwise
  JIT-builds the entry through FlashInfer (`gen_jit_spec`, sources
  digest-verified first; `sm_107a` needs an nvcc that lists `compute_107`).
  `aot/` is never tracked (`.gitignore`).
* `cake_launch.py` -- generated positional launchers and grid functions, one
  per stage of either program, over the kernels' own argument names
  (`STAGE_ARGS` / `LAUNCH` / `GRID` keyed by stage name).
* `cake_backend.py` -- validation, workspace layout, varlen index offsetting,
  the prepared runner, the autograd `Function` and the eager entry points.
* `csrc/cake_ffi_checks.h` -- the TVM-FFI helper header the bindings include
  (tensor-argument checks, `TVM_FFI_CHECK_CUDA_ERROR`, `build_cuda_version`),
  shipped by the export so the package compiles without FlashInfer's `csrc/`.
* `csrc/cake_dsa_h64_train/` -- generated kernel and binding translation units:
  one pair per library (the main stage and its key-range-pass form each with a
  natural-layout variant; the delta, compaction and cast pairs serve both
  programs), shared by `sm_100a`, `sm_103a` and `sm_107a` and
  compiled once per architecture (`.clang-format` disables formatting: the sources are
  identity-checked by the digests in `build.json`).

## Status

The registry holds the rope program for `sm_100a`, `sm_103a` and `sm_107a` with the forward,
backward preprocess, backward main (single-pass and key-range-pass form with
its compaction, each with its natural-layout direct-accumulation variant) and
cast stages plus the key-range-pass and direct-accumulation policies, exported
from the kernel snapshot named in the pull request, and the NoPE program's
record with its nine stages (the delta, compaction and cast kernels shared with
the rope program) and its own key-range-pass and direct-accumulation policies
(see "NoPE (zero-rope) program").

Tests: `tests/experimental/test_cake_dsa_train.py` (the backend suite: imports
only this package; skips without a registered program or a compute capability
10.0 / 10.3 / 10.7 device) and `tests/experimental/test_cake_dsa_train_api.py`
(the flashinfer public entry points and the environment variables the wrapper
resolves into this package's keyword switches).  Benchmark:
`benchmarks/bench_cake_dsa_train.py`.
