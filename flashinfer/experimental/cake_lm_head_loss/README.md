# Cake chunked LM-head + loss training backend (SM100 / SM103)

Experimental backend for the chunked large-vocabulary LM-head projection with
a fused log-sum-exp / cross-entropy / policy-gradient loss and a
memory-bounded backward (GLM-5.2 geometry: hidden size 6144, vocabulary
154880, irregular token counts such as 16231 or 16172 per step, default token
chunk 4096).  Tracking issue: flashinfer-ai/flashinfer#5680.

Both the API and the backend are experimental: no compatibility guarantee,
SM100 (B200 / GB200) is the acceptance architecture and SM103 (B300 / GB300)
is compiled from the same sources.  The feature is JIT-only and does not
participate in automatic backend selection, autotuning or trace-apply.

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
  row (zero loss, zero gradient, `logp = 0`).  A registered program is built
  for one `(H, V)` geometry, which its registry record pins; the host selects
  the record of the inputs' geometry (`cake_jit.select_module`: the default
  6144 x 154880 record `cake_lm_head_loss_<arch>`, or `..._h<H>_v<V>` for the
  other registered geometries) and rejects geometries no record serves.
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
  arbitrary downstream losses; its forward saves only the FP32 row statistics
  (`lse` and the selected logit) and the backward recomputes each chunk's
  logits from the saved inputs (four GEMMs per chunk).
* Valid-row compaction (`compact_rows`, default on unless
  `FLASHINFER_CAKE_LM_HEAD_LOSS_COMPACT_ROWS=0`): when some labels are `-100`
  the chunk loop runs over the valid rows only.  The int64 row index
  `(labels >= 0).nonzero()` is formed once per call (one device
  synchronization for the count, one for the index), every chunk gathers its
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
  row ignored: zeros without a launch.  The prepared runner
  (`prepare_lm_head_loss(..., compact_rows=True)`) fixes the valid-row set at
  preparation and returns the compact forms (`runner.scatter` restores `[T]`).
* Fused weight-gradient cast (`fuse_dw_cast`, default on unless
  `FLASHINFER_CAKE_LM_HEAD_LOSS_FUSE_DW_CAST=0`): the last chunk's
  weight-gradient GEMM runs in the backward, where the upstream scalar
  gradient `g` is known, with the scale and the output cast fused into its
  epilogue (`gemm_dw_cast_bf16` / `gemm_dw_cast_f32`): the forward
  accumulates the chunks before it into the FP32 `dW_acc` (a one-chunk call
  has no `[V, H]` accumulator at all), the backward writes
  `dW = cast(g * (dW_acc + dz_c^T @ X_c))` straight from the GEMM, and the
  separate `scale_cast` pass over `dW` disappears.  The same FP32 operations
  in the same order, so `dW` is bitwise the unfused result.  The last chunk's
  BF16 `dlogits` rows (`memory_report(...)["saved_dz_bytes"]`) and a view of
  its rows of `X` (or `X` with the chunk's row index when compacted) are saved
  for the backward; an in-place write to `X` between the forward and the
  backward raises PyTorch's saved-tensor version error.  `ForwardResult`
  carries them (`dz_last`, `x_last` / `x_src` + `x_idx`) for the explicit pair
  and `ForwardResult.backward` / `backward_loss(..., dz_last=...)` finish
  `dW`; the prepared runner keeps that chunk's `dz` in the workspace between
  `forward()` and `backward()`.
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
The cache keeps up to 64 bindings
(`FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE_CAPACITY` sets the capacity) and
evicts the least recently used one; `FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE=0`
disables it.  `backend="reference"` on the `cake_backend` entry points runs
the same chunk loop with PyTorch operators at the same rounding boundaries on
any device (the host-layer tests use it); the public API accepts
`backend="cake"` only.

## Kernel structure of one token chunk

Per chunk of `rows_c <= chunk_size` rows the host launches, in this order
(a compacted plan first gathers the chunk's valid rows of `X` into the `x_c`
workspace buffer with a torch `index_select`, `gather_rows`; the row operands
are gathered once per step):

* `gemm_logits`: `z_c = bf16(X_c @ W^T)` (2-CTA tensor-core GEMM, 128-row
  tiles, 256 vocabulary columns per accumulator) together with the
  per-(row, 256-column tile) online `(max, sum-exp)` partials of the
  BF16-rounded logits, written from the epilogue.  The log-probability
  backward's recompute uses `gemm_logits_nostats`, the same GEMM without the
  statistics.
* `row_finalize`: one warp per row merges the partials into `lse`, gathers
  the selected logit, writes `logp` (0 on ignored rows) and, per objective,
  the per-row logit-gradient scale `d_t` and the loss term; the
  log-probability backward takes `d_t` from the incoming `dlogp` instead.
* `loss_reduce`: one CTA sums the chunk's loss terms in a fixed order into
  the loss accumulator; the last chunk writes the finished loss.
* `row_grad`: `dz_c = d_t * (1[v = y_t] - exp(z - lse_t))` in BF16, in place
  over `z_c`; ignored rows become zero.
* `gemm_dx`: `dX_acc[rows] = fp32(dz_c @ W)`.  A record may also register
  `gemm_dx_s2` / `gemm_dx_s3` / `gemm_dx_s4`, the same GEMM as 2 / 3 / 4
  K-slice work items per output tile (the persistent grid fills its last
  wave; a record registers a contiguous prefix of them): slice 0 writes
  `dX_acc`, slices `>= 1` write FP32 workspace slabs (`dx_ws`, temporary
  bucket) that the host adds into `dX_acc` in fixed slab order (one RN add per
  element per slab, no atomics).  The slice count of a chunk follows from its
  row count and the SM count (`cake_backend.recommended_k_slices`), so it is
  deterministic in the shapes.  Each slice form exists with the 512-column
  pair tile (the default) and the 256-column tile (`_tn256`), see the
  instance variants below.
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

### Instance variants

The GEMM stages are registered as instance *variants*, one stage name per
output of the per-chunk rules the production launchers of the kernel source
apply -- the host selects by rule output, never by shape
(`cake_backend.stage_variant` / `parse_stage`; suffixes in this order):

* `_s<k>`: the K-slice count of the dX GEMM (`recommended_k_slices`);
* `_tn256`: the narrow dX tile, taken when `H % 512 != 0`, when the launch's
  512-wide work items would not fill the device's SM pairs, or -- on SM100
  below `H` 7168 -- when a one-slice launch fills the SM pairs' waves below
  0.90 (`DX_WIDE_MIN_EFF`: the badly wave-quantized chunk tails such as the
  3591 / 3663-row tails of `T` 65031 / 32463 at chunk 4096)
  (`cake_backend.dx_tile_rule`, evaluated after the slice count; `dX` is
  bitwise the same on both tiles);
* `_g<group_m>`: the grouped-raster height of the logits GEMMs (`_g16`) and
  of the weight-gradient GEMMs (`_g32`, accumulate and fused cast) at
  `H >= 7168` for chunks of more than 2048 rows (`cake_backend.raster_variant`,
  `RASTER_WIDE_GROUPS` per architecture), and of the weight-gradient GEMMs
  alone (`_g16`) below that `H` for chunks of more than 4096 rows on both
  architectures (`DW_LONG_CHUNK_GROUP_M` / `DW_LONG_CHUNK_MIN_ROWS`); every
  other chunk -- the default geometry's chunks of up to 4096 rows, short tail
  chunks -- keeps the default raster.  Bitwise: only the tile order changes;
* `_st3`: the 3-deep operand ring of the 512-wide `dX` tile below `H` 7168 for
  chunks of more than 4096 rows on both architectures
  (`cake_backend.dx_stages_variant`, `DX_LONG_CHUNK_STAGES`); the 256-wide
  fallback keeps its depth.  Bitwise: the ring depth only changes when a stage
  is refilled;
* `_gn12`: the 2-D blocked raster (12 column tiles per block) of the
  weight-gradient accumulate and the fp32 fused cast below `H` 7168 for chunks
  of 2049 .. 4096 rows on both architectures (`cake_backend.dw_block_variant`,
  `DW_BLOCK_GROUPS`); the bf16 cast keeps the 1-D raster.  Bitwise: a
  permutation of the same tiles.

`make_plan` resolves the variants per chunk (`Plan.variants`,
`Plan.logits_stage` / `dx_stage_of` / `dw_acc_stage` / `dw_cast_stage`); a
record carries exactly the variants its geometry and architecture can reach
(`cake_backend.reachable_variants`, checked by
`generated_program_available`), and a variant the record lacks fails closed.
`gemm_tuning` (`prepare_lm_head_loss`; `FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING`
as a JSON object for the eager entry points) pins knobs explicitly per GEMM --
`{"logits": {"group_m": 16}, "dx": {"tile_n": 256, "k_slices": 2, "stages": 4},
"dw": {"group_m": 32, "group_n": 0}}`, a knob given as `null` pins the default
form -- and wins over the rule for the knobs it names.

Grid rules live in the registry record (e.g. `rows_c/8` CTAs for
`row_finalize`, `V/8/1024 x rows_c` for `row_grad`) and are evaluated by the
host from the chunk scalars.  A GEMM stage runs in thread-block clusters whose
CTA count the record's geometry declares per GEMM (`logits_cluster_ctas`,
`dx_cluster_ctas`, `dw_cluster_ctas`; the host rounds `m_tiles` up to it and
checks it against the module's launch cluster): a statically scheduled
instance launches a persistent grid capped by `resident`, the device's
co-resident clusters of that width (`cluster_resident`: the SM pairs for a
two-CTA cluster, the driver's occupancy answer for wider ones), while a
dynamically scheduled instance launches its whole work-item domain.  The
`dX` K-slice rule uses the same co-resident count.  A GEMM compiled for the
preferred-cluster launch (the generated host binding passes the preferred
cluster dimension next to the required one, so the hardware forms the wider
clusters where the GPC topology allows and the required ones elsewhere)
declares its required cluster in the geometry; its grid rule counts the
wider work items, and the kernel reads the runtime cluster size.  The GEMM operands are
described by TMA descriptors the host prepares in a per-runner descriptor
workspace; the logits GEMM's output descriptor covers the chunk's rows only.

## Layout of this package

* `cake_jit.py` -- `MODULES` registry (one record per architecture and
  geometry, filled by the generated-program export), stage names
  (`STAGES`, `BASE_STAGES`, `GEMM_STAGES`), record selection
  (`select_module`) and the JIT specs.
* `cake_backend.py` -- validation, chunk plans, workspace layout, memory
  accounting, argument-plan binding, the prepared runner, the torch reference
  path, the autograd `Function`s and the eager entry points.
* `csrc/cake_lm_head_loss/<arch>/` -- generated kernel and binding
  translation units (`.clang-format` disables formatting: the sources are
  identity-checked by the registry's closure digests).

## Status

The registry holds one record per architecture (`sm_100a`, `sm_103a`) and
geometry -- `cake_lm_head_loss_<arch>` pinned to hidden size 6144 and
vocabulary 154880, `..._h7168_v129280` and `..._h8192_v128256` -- with the
eleven base stages above plus the instance variants the record's rules can
reach (`abi = lm_head_loss_v1`), exported from the kernel snapshot named in
the pull request.

Tests: `tests/experimental/test_cake_lm_head_loss.py` (the host-layer tests
run on any device through the reference path; the device tests skip without a
registered program or a compute capability 10.0 / 10.3 device).  Benchmark:
`benchmarks/bench_cake_lm_head_loss.py`.
