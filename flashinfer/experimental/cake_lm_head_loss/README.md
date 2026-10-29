# Cake chunked LM-head + loss training backend (SM100 / SM103)

Experimental backend for the chunked large-vocabulary LM-head projection with
a fused log-sum-exp / cross-entropy / policy-gradient loss and a
memory-bounded backward (GLM-5.2 geometry: hidden size 6144, vocabulary
154880, irregular token counts such as 16231 or 16172 per step, default token
chunk 4096).  Tracking issue: flashinfer-ai/flashinfer#5680.

Both the API and the backend are experimental: no compatibility guarantee,
SM100 (B200 / GB200) is the acceptance architecture and SM103 (B300 / GB300)
is compiled from the same sources.  The feature does not participate in
automatic backend selection, autotuning or trace-apply.  Its libraries are
loaded prebuilt from `aot/` when the package ships them and JIT-built by
FlashInfer from `build.json` otherwise (see "Layout of this package").

## Public entry points (`flashinfer/chunked_lm_head.py`)

```python
from flashinfer.chunked_lm_head import chunked_lm_head_loss, chunked_lm_head_logprob

loss = chunked_lm_head_loss(X, W, labels, objective="ce", loss_div=loss_div, chunk_size=4096)
loss, logp = chunked_lm_head_loss(X, W, labels, objective="policy", infer_logp=infer_logp,
                                  loss_weights=loss_weights, return_logp=True)
logp = chunked_lm_head_logprob(X, W, labels, chunk_size=4096)   # differentiable [T] FP32
```

* `X [T, H]` BF16 hidden states, row-major with any leading stride (`H` a
  multiple of 256; a row pitch that is not a multiple of 16 bytes is copied
  to a contiguous tensor first); `W [V, H]` BF16 contiguous output weight
  (`V` a multiple of 256); `labels [T]` int64 with `-100` marking an ignored
  row (zero loss, zero gradient, `logp = 0`).  The generated kernels take the
  call geometry as launch scalars, so one registered program per architecture
  (`cake_lm_head_loss_<arch>`, `cake_jit.select_module`) serves every `(H, V)`
  its record's geometry admits: `H` a multiple of the dX / weight-gradient work
  items' 512 columns, `V` a multiple of the logits work item's 256 columns.
* Per row `logp_t = z[t, y_t] - logsumexp_v z[t, v]` with `z = X @ W^T`
  computed as a BF16 GEMM (BF16 output of an FP32 accumulation) and promoted
  to FP32 for the max / log-sum-exp / loss arithmetic.
* `objective="ce"`: `loss = -sum(logp[valid]) / loss_div` with the
  caller-supplied positive divisor `loss_div` (never replaced by a local
  token mean).  `objective="policy"`:
  `loss = -sum(loss_weights * min(exp(logp - infer_logp), 2))` over valid rows
  with FP32 `infer_logp [T]` and signed FP32 `loss_weights [T]` (already
  masked and normalized; 0 on ignored rows); the logit-gradient scale is
  `-loss_weights_t * ratio_t` when `ratio_t <= 2` and `0` above the clipping
  boundary.
* Backward: `dz = d_t * (1[v = y_t] - softmax(z_t))` in BF16 (the `dlogits`
  boundary), `dX = dz @ W` and `dW = dz^T @ X` accumulated in FP32 across the
  chunks in a fixed order (no atomics) and cast once at the output: `dX` BF16,
  `dW` BF16 by default or FP32 with `grad_weight_dtype=torch.float32` (the
  FP32 form is served by the explicit `cake_backend.forward_loss` /
  `backward_loss` pair: the autograd entry casts every gradient to its leaf's
  dtype and therefore requires `grad_weight_dtype == W.dtype`; a compacted
  forward hands its `row_index` / `num_rows` to `backward_loss`, which casts
  the compact `dX` rows and scatters them into the `[T, H]` output).
  Gradients are produced only for inputs that require them (a frozen `X` or
  `W` skips its GEMM); the upstream scalar gradient is applied once, in the
  cast.  The saved FP32 accumulators are re-scaled, never mutated, so a
  retained graph may run the backward repeatedly.
* `chunked_lm_head_logprob` returns the differentiable `logp [T]` for
  arbitrary downstream losses; its forward saves the FP32 row statistic `lse`
  and the valid-row index, and the backward recomputes each chunk's logits
  from the saved inputs (four GEMMs per chunk) without counting the rows again
  (no host synchronization).
* Valid-row compaction (`compact_rows`, a keyword of every eager and autograd
  entry point, default `True`; `prepare_lm_head_loss` takes it as a bool or the
  valid-row tensor, default `False`): when some labels are `-100` the chunk loop runs over the valid rows
  only.  The int64 row index
  `(labels >= 0).nonzero()` is formed once per call (one device
  synchronization: `nonzero` returns the count and the index together; a call
  of at most `HOST_COUNT_MAX_ROWS` (1024) rows reads the bool mask back instead
  and forms the same index on the host, without `nonzero`'s count and select kernels
  (with some rows ignored, its index copy back makes this step a few microseconds
  slower than `nonzero` from about 128 rows up);
  the log-probability backward reuses the forward's index), every chunk gathers its
  rows of `X` into one reusable BF16 `[chunk_size, H]` buffer (`x_c`, a
  workspace region; a strided `X` then needs no contiguous copy), the row
  operands (`labels`, `infer_logp`, `loss_weights`, `dlogp`) are compacted to
  `[T_v]`, the FP32 `dX` accumulator is `[T_v, H]`, and `logp` / `dX` are
  scattered back to `[T]` / `[T, H]` with exact zeros on the ignored rows.
  Ignored rows contribute exactly zero to the loss and both gradients, so this
  is the same computation over fewer rows: per row, `logp` / `lse` are bitwise
  those of the uncompacted path, `dX` rows too whenever the containing chunk's
  K-slice count agrees; `loss` and `dW` are fixed-order reductions over
  different chunk boundaries and differ by FP32 rounding (deterministic run to
  run).  All rows valid: the uncompacted path (no gather, no scatter).  Every
  row ignored: zeros without a launch.  On this host-count path (the calls the
  hidden valid-row count below does not serve: one and two chunks) a
  remembered forward launches chunk 0's row gather and logits GEMM right
  behind the count and binds the rest of the call -- the compacted labels, the
  policy operands, the row vectors, the accumulators -- behind them
  (`cake_backend.forward_early_keys`; the log-probability forward's
  label-column GEMM binds the labels first): the same kernels on the same
  operands in the same stream order per kernel, so every output is bitwise
  the in-order sequence's.  The prepared runner
  (`prepare_lm_head_loss(..., compact_rows=True)`) fixes the valid-row set at
  preparation and returns the compact forms (`runner.scatter` restores `[T]`).
* Fused weight-gradient cast (`fuse_dw_cast`, keyword, default `True`): the
  last chunk's
  weight-gradient GEMM runs in the backward, where the upstream scalar
  gradient `g` is known, with the scale and the output cast fused into its
  epilogue (`gemm_dw_cast_bf16` / `gemm_dw_cast_f32`): the forward
  accumulates the chunks before it into the FP32 `dW_acc` (a one-chunk call
  has no `[V, H]` accumulator at all), the backward writes
  `dW = cast(g * (dW_acc + dz_c^T @ X_c))` straight from the GEMM, and the
  separate `scale_cast` pass over `dW` disappears.  The same FP32 operations
  in the same order, so `dW` is bitwise the unfused result.  The last chunk's
  BF16 `dlogits` rows -- a view of the `[min(T, C), V]` chunk buffer, which
  therefore stays alive until the backward; `memory_report(...)["saved_dz_bytes"]`
  counts that whole buffer, and the eager path allocates it on its own from the
  first call on, so the view keeps nothing else alive -- and a view of its rows of `X` (or `X` with the
  chunk's row index when compacted) are saved for the backward; an in-place write to `X` between the forward and the
  backward raises PyTorch's saved-tensor version error.  `ForwardResult`
  carries them (`dz_last`, `x_last` / `x_src` + `x_idx`) for the explicit pair
  and `ForwardResult.backward` / `backward_loss(..., dz_last=...)` finish
  `dW`; the prepared runner keeps that chunk's `dz` in the workspace between
  `forward()` and `backward()`.
* Fused dX finalize (`dx_finalize`, keyword, default `True`): a K-sliced `dX`
  GEMM's FP32
  slabs are added into `dX_acc` by one fixed-order kernel (`slab_sum`: one RN
  add per element per slab in ascending slab order -- the evaluation order of
  the previous chain of in-place `add_` launches, with `dX_acc` read and
  written once), and a compacted call writes its `[T, H]` BF16 `dX` in one pass
  (`scale_cast_scatter_bf16`: `bf16(g * dX_acc[compact row])` on the valid
  rows, exact zeros elsewhere, addressed through the inclusive int32 scan of
  the valid-row mask -- one `cumsum` issued in the backward -- and validated
  against the compact index) instead of the flat cast, the zero fill and the
  `index_copy_` of the compact rows.  The forward keeps the bool valid-row mask
  next to the row index (`ForwardResult.row_valid`); `backward_loss(...,
  row_valid=...)` / `finalize_dx` select the one-pass form, the log-probability
  backward finalizes its accumulator the same way, and every row valid stays
  the single flat cast.  The same FP32 operations in the same order: `dX` is
  bitwise the previous path's, and `dx_finalize=False` restores that path.  The prepared runner
  keeps its compact `dx_out` contract (`runner.scatter` restores `[T, H]`) and
  gains the kernel slab reduction only.
* dW side stream (`dw_stream`, keyword: `None` / `"auto"` (default) = calls of
  three or more chunks, `True` = every multi-chunk call, `False` = never;
  `cake_backend.dw_stream_mode` normalises the value): each chunk's
  weight-gradient accumulate GEMM is launched on a per-device side
  stream forked after the chunk's `row_grad` (`dz` ready) and joined before the
  next chunk reuses the chunk buffer and before the call returns, so its CTAs
  fill the tail wave of the chunk's `dX` GEMM instead of queueing behind it.
  Kernels, buffers, launch order per kernel and numerics are unchanged: `loss`,
  `logp`, `dX` and `dW` are bitwise the single-stream path's.  Two-chunk calls
  pay the cross-stream join without an overlap gain and a one-chunk call
  defers its only accumulate to the backward, hence the `auto` rule
  (`cake_backend.dw_side_stream`).  The prepared runner takes the same keyword
  (`prepare_lm_head_loss(..., dw_stream=...)`; `runner.dw_stream` holds the
  mode of its launches, and a CUDA graph captured from it records the fork /
  join).
* Hidden valid-row count (`hidden_count`, keyword, default `True`; `False`
  restores the previous path): a compacted call used to count its valid rows on the host (`sum().item()`)
  and form their index (`nonzero`) before its first launch.  It now forms the
  index on the device (`labels >= 0` -> an inclusive int32 `cumsum` ->
  `searchsorted`), gathers chunk 0's rows of `X` with the `gather_rows_bf16`
  kernel and runs chunk 0's logits GEMM in its device-count form
  (`gemm_logits_mcnt` / `gemm_logits_nostats_mcnt`: the stores bounded by the
  count read from device memory, the buffer extent `min(chunk_size, T)` as
  `M`), and reads the count back through a pinned cell and a CUDA event only
  after both are queued -- the host waits for the three compaction kernels and
  a 4-byte copy instead of the count and the index.  Every row valid: the
  uncompacted plan with chunk 0's GEMM already done (its `X` rows for the
  weight gradient stay the input view); every row ignored: zeros without a
  further launch; otherwise the compacted plan over the counted rows with
  chunk 0's gather and GEMM skipped.  The same kernels per row, so `loss` /
  `logp` / `dX` / `dW` are bitwise the previous path's.  Calls the
  device-count GEMM cannot serve -- an `X` that needs a contiguous copy or
  whose base is not 16-byte aligned, a program without the kernels --, calls
  of fewer than three chunks (`ceil(T / chunk_size) < 3`,
  `cake_backend.HIDDEN_COUNT_MIN_CHUNKS`: the path's fixed host cost per call
  is hidden behind chunk 0's logits GEMM only from three chunks on; one- and
  two-chunk calls measured 5-16 % slower with it, so, like the dW side
  stream's `auto` rule, it serves calls of three or more chunks) and every
  `sm_103a` call on CUDA 13.0 (the toolchain note below) take the previous
  path; the prepared runner (`prepare_lm_head_loss`) is
  unchanged.  The chunk buffers and the gather buffer exist at the buffer
  extent before the count is known (`memory_report(..., hidden_count=True)`;
  `cake_backend.hidden_count_eligible` / `_hidden_count_begin`).
* Log-probability logits GEMM arms (`logprob_dz` / `logprob_lbl`, keywords of
  the log-probability entry points, default `True`; `False` restores the
  previous path): the backward's recompute of each chunk is one
  `gemm_logits_dz` launch whose epilogue writes
  `dz_c = bf16(d_t * (1[v = y_t] - exp(z - lse_t)))` -- the per-element
  sequence of `row_grad` on the same BF16-rounded logits, zeros on ignored
  rows -- in place of the logits, so the logits store and the `row_grad`
  launch disappear; the forward's logits GEMM `gemm_logits_lbl` writes the row
  statistics and the logit at the (clamped) label column alone, the one
  element `row_finalize` reads.  The same arithmetic per element, so `lse`,
  `logp`, `dX` and `dW` are bitwise the previous path's.  A record that does
  not register an arm keeps the previous stages; the hidden valid-row count's
  chunk 0 keeps its device-count GEMM (`gemm_logits_mcnt` /
  `gemm_logits_nostats_mcnt` + `row_grad`).
* Memory rule: no logits, probability or `dlogits` buffer ever spans more than
  `chunk_size` tokens; a batch smaller than `chunk_size` is one chunk, a tail
  `T % chunk_size` is neither dropped nor padded, and `T == 0` returns loss 0,
  an empty `logp` and zero gradients without binding or launching.
  `cake_backend.memory_report` reports the peak temporary bytes separately
  from the weights, the outputs and the FP32 accumulators.
* `deterministic=True` only: fixed sequential chunk order, bitwise
  reproducible `loss`, `logp`, `dX` and `dW` run to run.

Explicit forward / backward entry points without autograd (`forward_loss`,
`backward_loss`, `forward_logprob`, `backward_logprob`), a prepared
allocation-free runner (`prepare_lm_head_loss`, CUDA-graph capturable) and the
workspace sizing helper (`lm_head_loss_workspace_size`) live in
`cake_backend.py`.  The eager entry points (and the autograd wrappers behind
the public API) validate and bind once per input binding -- `(data_ptr, shape,
stride, dtype)` of every input plus the options -- and launch later calls from
the remembered argument plans with freshly allocated outputs and per-call
scratch (`cake_backend.BINDING_CACHE`).  A remembered binding pins no caller
tensor and holds no problem-sized scratch: the chunk workspace, the FP32
accumulators and the outputs come from the caching allocator on every call.
The cache keeps up to 64 bindings (`cake_backend.BINDING_CACHE_DEFAULT_CAPACITY`)
and evicts the least recently used one;
`cake_backend.BINDING_CACHE.configure(enabled=..., capacity=...)` changes both
settings at runtime (a shrink evicts, `enabled=False` bypasses the cache and
drops the remembered bindings).  `backend="reference"` on the `cake_backend`
entry points runs the same chunk loop with PyTorch operators at the same
rounding boundaries on any device (the host-layer tests use it); the public API
accepts `backend="cake"` only.

## Switches and environment variables

Every switch above is a keyword argument of each entry point it affects
(`chunked_lm_head_loss`, `chunked_lm_head_logprob`, `forward_loss`,
`backward_loss`, `forward_logprob`, `backward_logprob`,
`prepare_lm_head_loss`; `prepare_lm_head_loss` takes `compact_rows` as a bool or
the valid-row tensor, default `False`, and no `hidden_count`; `forward_logprob`
and `backward_loss` take no `dw_stream`; `logprob_lbl` belongs to the
log-probability forward and `logprob_dz` to its backward), `None` selecting the documented default
(`cake_backend.compact_rows_default()` and friends return the constants); the
autograd entries resolve every switch once in the forward.  **The package reads
no environment variable.**  The `FLASHINFER_CAKE_LM_HEAD_LOSS_*` variables are
FlashInfer's own A/B knobs and are read by `flashinfer.chunked_lm_head` only: a
switch keyword left at `None` there falls back to its variable and the resolved
value is passed down; the two binding-cache variables are read when that
module is imported and applied through `BINDING_CACHE.configure` when it first
reaches the backend.  A build that vendors the package without the wrapper
sees the backend defaults.  The wrapper leaves `logprob_dz` / `logprob_lbl`
at their backend defaults (no variable).

| Keyword (backend default) | Variable read by `flashinfer.chunked_lm_head` | Parsing |
| --- | --- | --- |
| `compact_rows` (`True`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_COMPACT_ROWS` | `0` -> `False`, else `True` |
| `fuse_dw_cast` (`True`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_FUSE_DW_CAST` | `0` -> `False`, else `True` |
| `dx_finalize` (`True`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_DX_FINALIZE` | `0` -> `False`, else `True` |
| `dw_stream` (`"auto"`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM` | `0` / `false` / `off` / `no` -> `False`; `1` / `true` / `on` / `yes` -> `True`; else `"auto"` |
| `hidden_count` (`True`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_HIDDEN_COUNT` | `0` -> `False`, else `True` |
| `gemm_tuning` (`{}`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING` | JSON object -> dict; unset / blank -> no knob; invalid -> `ValueError` |
| `BINDING_CACHE.configure(enabled=...)` (`True`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE` | `0` -> disabled (read at import, applied on first use) |
| `BINDING_CACHE.configure(capacity=...)` (`64`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE_CAPACITY` | positive integer (read at import, applied on first use) |

## Kernel structure of one token chunk

Per chunk of `rows_c <= chunk_size` rows the host launches, in this order
(a compacted plan first gathers the chunk's valid rows of `X` into the `x_c`
workspace buffer with a torch `index_select`, `gather_rows`; the row operands
are gathered once per step; under the hidden valid-row count chunk 0's
gather and logits GEMM ran before the count was known, see
`gather_rows_bf16` below):

* `gemm_logits`: `z_c = bf16(X_c @ W^T)` (2-CTA tensor-core GEMM, 128-row
  tiles, 256 vocabulary columns per accumulator) together with the
  per-(row, 256-column tile) online `(max, sum-exp)` partials of the
  BF16-rounded logits, written from the epilogue.  The log-probability
  backward's recompute uses `gemm_logits_nostats`, the same GEMM without the
  statistics.  The log-probability entry's arms (`logprob_lbl` /
  `logprob_dz`): `gemm_logits_lbl` writes the partials and `z_c` at the label
  column only (forward); `gemm_logits_dz` writes `dz_c` (the `row_grad`
  sequence below) in place of `z_c` (backward recompute, no `row_grad`).
* `row_finalize`: one warp per row merges the partials into `lse`, gathers
  the selected logit, writes `logp` (0 on ignored rows) and, per objective,
  the per-row logit-gradient scale `d_t` and the loss term; the
  log-probability backward takes `d_t` from the incoming `dlogp` instead.
* `loss_reduce`: one CTA sums the chunk's loss terms in a fixed order into
  the loss accumulator; the last chunk writes the finished loss.
* `row_grad`: `dz_c = d_t * (1[v = y_t] - exp(z - lse_t))` in BF16, in place
  over `z_c`; ignored rows become zero.
* `gemm_dx`: `dX_acc[rows] = fp32(dz_c @ W)`.  A record also registers
  `gemm_dx_s`, the same GEMM as `k_slices` (2 .. 4, a launch scalar) K-slice
  work items per output tile (the persistent grid fills its last wave): slice
  0 writes `dX_acc`, slices `>= 1` write FP32 workspace slabs (`dx_ws`,
  temporary bucket) that the host adds into `dX_acc` in fixed slab order (one
  RN add per element per slab, no atomics).  The slice count of a chunk
  follows from its row count and the SM count
  (`cake_backend.recommended_k_slices`), so it is deterministic in the shapes.
  Both forms exist with the 512-column pair tile (the default) and the
  256-column tile (`_tn256`), see the structural forms below.
* `gemm_dw_acc`: `dW_acc = fp32(dz_c^T @ X_c)` on the first chunk,
  `dW_acc += ...` afterwards (the chunk order is the reduction order).
* `gemm_dw_cast_bf16` / `gemm_dw_cast_f32`: the last chunk's `dz_c^T @ X_c`
  with the upstream scale and the output cast fused into the epilogue,
  `dW = cast(g * (dW_acc + tile))` -- `cast(g * tile)` for a one-chunk plan
  -- run in the backward (`fuse_dw_cast`); `dW_acc` is read, never written.
* `scale_cast_bf16` / `scale_cast_f32`: `out = g * acc` over the flat FP32
  accumulators -- the single output cast of `dX` (and of `dW` without
  `fuse_dw_cast`) in the backward, with the upstream gradient `g` read from a
  device scalar.
* `slab_sum`: after a K-sliced `gemm_dx_s`, `dX_acc[rows] += slab_0 + ...
  + slab_{k-2}` in one launch, one RN add per element per slab in ascending
  slab order (the fused dX finalize; `dx_reduce`, the host's `add_` chain,
  when it is off).
* `scale_cast_scatter_bf16`: the dX output boundary of a compacted call with
  the fused dX finalize -- `out[r] = bf16(g * acc[scan[r] - 1])` when
  `row_index[scan[r] - 1] == r`, exact zeros otherwise, one CTA per output row
  per iteration, every element written once (launched by `finalize_dx`, not
  by the chunk loop).
* `gather_rows_bf16`: chunk 0's row gather of the hidden valid-row count
  (the eager entry points) -- `out[r] = X[idx[r]]` for `r < min(count,
  num_rows)` with the count read from device memory, exact zeros after; one
  CTA per row per iteration over 16-byte words, `X` addressed by its base
  pointer and row pitch (no contiguous copy), the int64 index read as int32
  pairs.  Launched before chunk 0's logits GEMM, which runs in its
  device-count form (`gemm_logits_mcnt` / `gemm_logits_nostats_mcnt`: `M`,
  `m_tiles` and the raster are the buffer extent `min(chunk_size, T)`, every
  store and statistics write is bounded by `min(count, M)`; bitwise the
  host-bound form per row).

### Structural forms and launch scalars

The GEMM stages are registered as structural *forms* -- one stage name per
kernel body the program carries, named by the outputs of the per-chunk rules
the production launchers of the kernel source apply that select a different
body (`cake_backend.stage_variant` / `parse_stage`; suffixes in this order).
The call geometry and the rules' numeric outputs are *launch scalars* of every
form (`cake_backend.gemm_scalars`: `ldc` and `k_iters` from `H` / `V`, the
K-slice count `k_slices` and slice length `k_slice_iters`, the raster height
`group_m` in row tiles, the raster block width `group_n` in column items), so
a record serves every admissible geometry and the host never selects by shape:

* `_mcnt`: the device-count form of the logits GEMMs (`gemm_logits_mcnt`,
  `gemm_logits_nostats_mcnt`) -- chunk 0 of the hidden valid-row count, whose
  valid-row bound is read from device memory while `M`, `m_tiles` and the
  raster height are evaluated at the buffer extent `min(chunk_size, T)`
  (`cake_backend.hidden_count_eligible`, `Plan.hidden_logits_stage`).
  Bitwise: the same per-row arithmetic, rows at and beyond the count are not
  written;
* `_s`: the K-sliced dX GEMM (`k_slices` 2 .. 4 from `recommended_k_slices`;
  one slice runs the plain form);
* `_tn256`: the narrow dX tile, taken when `H % 512 != 0`, when the launch's
  512-wide work items would not fill the device's SM pairs, or -- on SM100
  below `H` 7168 -- when a one-slice launch fills the SM pairs' waves below
  0.83 (`DX_WIDE_MIN_EFF`: the badly wave-quantized chunk tails such as the
  3591 / 3663-row tails of `T` 65031 / 32463 at chunk 4096)
  (`cake_backend.dx_tile_rule`, evaluated after the slice count; `dX` is
  bitwise the same on both tiles);
* `_st3`: the 3-deep operand ring of the 512-wide `dX` tile below `H` 7168 for
  chunks of more than 4096 rows on both architectures
  (`cake_backend.dx_stages_variant`, `DX_LONG_CHUNK_STAGES`); the 256-wide
  fallback keeps its depth.  Bitwise: the ring depth only changes when a stage
  is refilled;
* `_split`: the 512-wide `dX` tile whose launch's last wave would leave SM
  pairs idle: when the `W` wide work items (row pairs x `H / 512` x
  `k_slices`) leave `r = W mod P` items in the last wave of the `P` SM pairs
  with `0 < r <= P / 2`, the last `r` wide items run as `2 r` 256-wide half
  items in the same launch (the grid carries `W + r` clusters; the split count
  is the launch scalar `dx_split`).  Taken on both architectures for chunks
  that are not a whole number of 512-token tiles; for chunks of whole
  512-token tiles below `H` 7168 on sm_103a in every call
  (`DX_SPLIT_WHOLE_ARCHES`) and on sm_100a in calls of fewer than three
  chunks, whose dW accumulate shares the dX stream
  (`DX_SPLIT_WHOLE_FEW_CHUNKS_ARCHES`, `DW_STREAM_MIN_CHUNKS`); on the
  long-chunk ring (`_st3_split`) on sm_103a only
  (`cake_backend.dx_split_rule`, `cake_backend.dx_split_eligible`,
  `DX_SPLIT_ARCHES`, `DX_SPLIT_LONG_RING_ARCHES`).  Bitwise: the same K
  slices and per-element K order on both widths;
* `_swab`: the swapped 512-wide `dX` tile (`dX^T = W^T dz^T`: a CTA's 128
  hidden columns of `W` on the MMA's M, the chunk's 512-token tile on its N,
  the output stored per thread), at the table ring depth, on sm_103a only
  (`DX_SWAB_ARCHES`): a chunk of whole 512-token tiles of a cross-entropy
  loss call (the loss entry point with objective `"ce"`) of at least three
  chunks (`DW_STREAM_MIN_CHUNKS`) takes it ahead of the split-tail rule
  (`cake_backend.dx_swab_rule`); the policy objective, the log-probability
  entry point and calls of fewer chunks keep the other forms.  Bitwise: the
  same bf16 products summed over the same K slices in the same K order.

On sm_100a the record's 512-wide `dX` stages (`gemm_dx`, `gemm_dx_s`, their
`_st3` and `_split` forms) run programs built with a 3-deep operand ring and a
4-deep epilogue staging of the output tile (one program serves a base form and
its `_st3` form); sm_103a keeps the table depths.  Bitwise: only the pipeline
depths differ.
* `_gn`: the 2-D blocked raster of the weight-gradient accumulate and the fp32
  fused cast below `H` 7168 for chunks of 2049 .. 4096 rows on both
  architectures, with `group_n` = 12 column tiles per block
  (`cake_backend.dw_block_variant`, `DW_BLOCK_GROUPS`); the bf16 cast keeps the
  1-D raster.  Bitwise: a permutation of the same tiles.
* `_tma`: the TMA-epilogue form of the bf16 fused weight-gradient cast
  (`gemm_dw_cast_bf16_tma`): the deferred cast of the last chunk of a call of
  at least two chunks whose last chunk has at most 1792 rows
  (`DW_CAST_TMA_MAX_ROWS`, both architectures; `cake_backend.dw_cast_tma_rule`)
  reads the old FP32 sum and stores the bf16 tile as TMA sub-tiles.  A
  one-chunk call (no old sum) and a longer last chunk keep the shipped cast;
  an explicit `"epi_store"` knob of `gemm_tuning`'s `"dw"` wins over the rule.
  Bitwise: the same per-element `(old + acc) * g` and round-to-nearest.

The raster height is a scalar of the base forms: `group_m` 16 for the logits
GEMMs and 32 for the weight-gradient GEMMs (accumulate and fused cast) at
`H >= 7168` for chunks of more than 2048 rows (`cake_backend.raster_variant`,
`RASTER_WIDE_GROUPS` per architecture); below that `H`, 16 for the logits
GEMMs of chunks of at least 3841 rows (31 row tiles of 128:
`LOGITS_LONG_RASTER_GROUPS` / `LOGITS_LONG_RASTER_MIN_ROWS`) and 16 for the
weight-gradient GEMMs of chunks of more than 4096 rows, both on both
architectures (`DW_LONG_CHUNK_GROUP_M` / `DW_LONG_CHUNK_MIN_ROWS`); every
other chunk -- the default geometry's chunks of up to 30 row tiles, short tail
chunks -- takes the GEMM's default height from the record geometry
(`logits_group_m` / `dx_group_m` / `dw_group_m`).  Bitwise: only the tile
order changes.

`make_plan` resolves the forms and scalars per chunk (`Plan.variants`,
`Plan.logits_stage` / `dx_stage_of` / `dw_acc_stage` / `dw_cast_stage`;
`stage_values` forms the scalars); a record carries exactly the forms its
architecture's rules can reach (`cake_backend.reachable_variants`, checked by
`generated_program_available`), and a form the record lacks fails closed.
`gemm_tuning` (a keyword of `prepare_lm_head_loss` and of every eager /
autograd entry point; default `None` = no explicit knob) pins knobs explicitly
per GEMM --
`{"logits": {"group_m": 16}, "dx": {"tile_n": 256, "k_slices": 2, "stages": 4},
"dw": {"group_m": 32, "group_n": 0}}`; the `dX` GEMM also takes `"split"` (the
number of trailing wide items run as half items) and `"swap_ab"` (the `_swab`
tile) -- a knob given as `null` pins the default -- and wins over the rule for
the knobs it names.  An explicit `"split"`, or a `"stages"` other than the
table ring depth 4, keeps that form and turns the `_swab` rule off.
`"swap_ab": true` applies to every chunk; it raises with such a `"stages"`,
with a nonzero `"split"`, on an architecture without the swapped program
(only SM103 has it), and on a chunk whose dX tile is the 256-column one
(pin `"tile_n": 512` as well).

Grid rules live in the registry record (e.g. `rows_c/8` CTAs for
`row_finalize`, `V/8/1024 x rows_c` for `row_grad`) and are evaluated by the
host from the chunk scalars.  A GEMM stage runs in thread-block clusters whose
CTA count the record's geometry declares per GEMM (`logits_cluster_ctas`,
`dx_cluster_ctas`, `dw_cluster_ctas`; the host rounds `m_tiles` up to it and
checks it against the module's launch cluster): every registered GEMM
instance launches a grid over its whole work-item domain (no grid rule reads
a co-resident cluster count), and the `dX` K-slice rule uses the device's
co-resident two-CTA clusters (`cluster_resident`: the SM pair
count).  A GEMM compiled for the
preferred-cluster launch (the generated host binding passes the preferred
cluster dimension next to the required one, so the hardware forms the wider
clusters where the GPC topology allows and the required ones elsewhere)
declares its required cluster in the geometry; its grid rule counts the
wider work items, and the kernel reads the runtime cluster size.  The GEMM operands are
described by TMA descriptors the host prepares in a per-runner descriptor
workspace; the logits GEMM's output descriptor covers the chunk's rows only.

## Layout of this package

* `cake_jit.py` -- `MODULES` registry (one record per architecture, expanded
  from the compact tables the generated-program export writes between the
  `generated registry` markers; a stage's physical record names its program
  and the architecture-free `library` stem `cake_lm_head_loss_<unit>`), stage
  names (`STAGES`, `BASE_STAGES`, `GEMM_STAGES`), record selection
  (`select_module`) and the loader glue: `library_name(name, stage)` is the
  `build.json` entry `<library>_<arch>` that
  `load_cake_lm_head_loss_module(name, stage)` loads through `cake_build`.
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package (byte-identical copies) and this package's build manifest (schema
  `cake-build/1`, written by the export): one entry per `<library>_<arch>`
  with its translation units, digests, nvcc flags (including the
  toolchain-conditional `nvcc_flags_if`, see the toolchain note), closure
  digest and `aot/<name>.so` path.  `cake_build.load_library(name)` loads the
  prebuilt library when `aot/` holds it and otherwise JIT-builds the entry
  through FlashInfer (`gen_jit_spec`, sources digest-verified first);
  `cake_jit.specs_for_arch(arch)` lists the JIT specs of an architecture for
  `flashinfer.jit.build_jit_specs`.
* `cake_backend.py` -- validation, chunk plans, workspace layout, memory
  accounting, argument-plan binding, the prepared runner, the torch reference
  path, the autograd `Function`s and the eager entry points.
* `csrc/cake_lm_head_loss/` -- generated translation units, two per program
  `cake_lm_head_loss_<unit>`: the kernel unit
  (`cake_lm_head_loss_<unit>_kernel.cu`, architecture-neutral, the
  per-architecture lowering under `__CUDA_ARCH__` guards) and the tvm-ffi
  binding (`cake_lm_head_loss_<unit>_binding.cu`: argument checks, the
  descriptor encoders, the dynamic shared memory and the launch).  Sibling
  forms of one kernel share a kernel unit: the shared body defines its
  kernel through `CAKE_PROGRAM_KERNEL` and the knob macros select each form's
  lines (`#if` arms), set per library by the compile definitions of its
  `build.json` entry -- the eight logits GEMM forms of both architectures,
  the 512-wide and the 256-wide `dX` forms, the weight-gradient accumulate
  forms, the fused casts and the two scale casts.  Equal bindings are one
  file whose kernel is `CAKE_KERNEL_SYMBOL`, likewise set per library;
  `csrc/cake_ffi_checks.h` -- the tvm-ffi tensor checks the bindings include
  (shipped with the package, no FlashInfer header).
  `.clang-format` disables formatting: the sources are identity-checked by
  the manifest's digests.

## Status

The registry holds one record per architecture (`sm_100a`, `sm_103a`),
`cake_lm_head_loss_<arch>`, geometry-free (`H` / `V` and the rules' outputs are
launch scalars: every `H` a multiple of 512 and `V` a multiple of 256 runs on
it), with the sixteen base stages above plus the sixteen structural forms the
rules can reach (`abi = lm_head_loss_v1`), exported from the kernel snapshot
named in the pull request.

Tests: `tests/experimental/test_cake_lm_head_loss.py` and the public-API suite
`tests/experimental/test_cake_lm_head_loss_api.py` (the host-layer tests
run on any device through the reference path; the device tests skip without a
registered program or a compute capability 10.0 / 10.3 device), and the
CPU-only `tests/experimental/test_cake_lm_head_loss_standalone.py` (a copy of
the package imports with `flashinfer` blocked and raises its host-side errors;
`scripts/check_cake_build_manifests.py --vendored-gate` builds and loads the
copy's libraries on a GPU node).  Benchmark:
`benchmarks/bench_cake_lm_head_loss.py`.

Toolchain note: on CUDA 13.0 (nvcc 13.0.x) the `sm_103a` programs are built
with `-Xptxas -O0` (build data: every `sm_103a` entry of `build.json` carries
`{"cuda": "==13.0", "flags": ["-Xptxas", "-O0"]}` under `nvcc_flags_if`; the
JIT path applies it when the local toolkit matches, a prebuilt `aot/` built by
that toolkit already carries it).  That toolchain's
ptxas mis-schedules their 2-CTA TMA producer loops: at -O3 (and -O2, which
emits the same code), with a short K loop (`chunk_size <= 1024`) the dW
accumulate GEMM's second B-operand TMA load is rejected by the TMA unit with
`cudaErrorIllegalInstruction` although every operand is legal; at -O1 the
structural-form programs' dW cast GEMM (the last chunk's `dz_c^T @ X_c`) faults
the same way on every call whose last chunk is partial (`T % chunk_size != 0`).
The -O0 code and the code of CUDA 12.9, 13.3 and 13.4 are correct.  With the
earlier -O1 code the hidden
valid-row count's schedule (the compaction index, chunk 0's row gather and
device-count logits GEMM queued before the count is read back) still reaches a
`cudaErrorIllegalInstruction` from the dW accumulate GEMM at its shortest K
schedule (the one-row tail chunk of a two-chunk call) in every run of the test
file, while the host-count path passes (as does the -O0 code with the hidden
count); so on CUDA 13.0 the `sm_103a` calls also take the host-count path
(`cake_jit.toolchain_runs_hidden_count`, which reads the building toolkit's
version from `cake_build.record_cuda_version` -- the prebuilt libraries'
`aot/build_info.json`, else the local toolkit; the same kernels per row,
bitwise the same outputs).  Every other architecture / toolchain pair builds at the default
optimization level and runs the hidden count.  Both are mitigations of the
observed ptxas 13.0 fault, not a root-cause fix.
