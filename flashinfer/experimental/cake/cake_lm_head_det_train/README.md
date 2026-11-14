# Cake chunked LM-head + loss training backend (SM100 / SM103 / SM107)

Experimental backend for the chunked large-vocabulary LM-head projection with
a fused log-sum-exp / cross-entropy / policy-gradient loss and a
memory-bounded backward (GLM-5.2 geometry: hidden size 6144, vocabulary
154880, irregular token counts such as 16231 or 16172 per step, default token
chunk 4096).  Tracking issue: flashinfer-ai/flashinfer#5680.

The public module `flashinfer.cake_lm_head_det_train` is a stable API
(`@flashinfer_api`, no `ExperimentalWarning`); this backend package is
experimental, with no compatibility guarantee for its internal modules.
SM100 (B200 / GB200) is the acceptance architecture; SM103 (B300 / GB300) and
SM107 (R200) are compiled from the same sources (SM107 runs SM103's programs,
built for `sm_107a`, under SM103's dispatch rules except the logits rasters,
the long-chunk `dX` ring and the whole-chunk `dX` split).  The feature does
not participate in automatic backend selection, autotuning or trace-apply.
Its libraries are loaded prebuilt from `aot/` when the package ships them and
JIT-built by FlashInfer from `build.json` otherwise (see "Layout of this
package").

## Public entry points (`flashinfer/cake_lm_head_det_train.py`)

```python
from flashinfer.cake_lm_head_det_train import cake_lm_head_det_train_loss, cake_lm_head_det_train_logprob

loss = cake_lm_head_det_train_loss(X, W, labels, objective="ce", loss_div=loss_div, chunk_size=4096)
loss, logp = cake_lm_head_det_train_loss(X, W, labels, objective="policy", infer_logp=infer_logp,
                                      loss_weights=loss_weights, return_logp=True)
logp = cake_lm_head_det_train_logprob(X, W, labels, chunk_size=4096)   # differentiable [T] FP32
logp = cake_lm_head_det_train_logprob(X, W, labels, temperature=0.7)   # scalar temperature
logp = cake_lm_head_det_train_logprob(X, W, labels, temperature=tau)   # per-token: FP32 [T] on X's device
```

`det`: for fixed inputs, hardware, chunk plan and initial accumulation
buffers, repeated forward and backward calls give bitwise-identical outputs and
gradients (not promised across architectures, shapes or chunk plans).  The
entries are `cake_lm_head_det_train_{loss,logprob,preload,support_logprob,support_kl}`
(the module's `__all__`); the former names fail at import (see
`docs/training_backend_names.md`).  These entries take no `deterministic`
keyword.

* `X [T, H]` BF16 hidden states, row-major with any leading stride (`H` a
  multiple of 256; a row pitch that is not a multiple of 16 bytes is copied
  to a contiguous tensor first); `W [V, H]` BF16 contiguous output weight
  (`V` a multiple of 256); `labels [T]` int64 with `-100` marking an ignored
  row (zero loss, zero gradient, `logp = 0`).  The generated kernels take the
  call geometry as launch scalars, so one registered program per architecture
  (`cake_lm_head_det_loss_<arch>`, `cake_jit.select_module`) serves every `(H, V)`
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
  retained graph may run the backward repeatedly (see "Saved-state release"
  below for a backward of a budget form -- `memory_mode="budget"`, or a
  default call whose automatic form is a budget form -- that frees the
  graph).
* Gradient placement (`cake_backend.set_gradient_placement(mode)`, returns the
  previous mode; `cake_backend.gradient_placement()` reads it; default
  `"auto"`; read once per call, it covers `cake_lm_head_det_train_loss` in every
  form, the opt-in forms included).  `"auto"`: a call no backward can follow
  (grad mode off -- `torch.no_grad`, `torch.inference_mode`, the first pass of
  a reentrant `torch.utils.checkpoint` -- or neither `X` nor `W` trainable)
  computes `loss` (and `logp`) only, without a dX / dW GEMM; inside a
  non-reentrant `torch.utils.checkpoint` (its own saved-tensor hooks
  innermost) the checkpointed forward computes the loss only and saves its
  inputs, and the recomputation in the backward runs the fused forward and
  hands the gradient accumulators to the checkpointed call's backward, so a
  checkpointed step forms them once and keeps no FP32 accumulator from the
  forward to the backward (paired per checkpoint frame and graph by call
  order and checked against the inputs' metadata and the call's options;
  without a handed result the backward runs the fused forward itself).  Other
  saved-tensor hooks innermost (CPU offload inside the checkpointed function)
  or a torch whose checkpoint lacks those hooks keep the fused forward.  Under
  selective activation checkpointing (`context_fn` of
  `create_selective_checkpoint_contexts`, whose recomputation replays the
  checkpointed forward's operators by position) every det entry -- the loss,
  log-probability (default, budget, kept logits, with a temperature) and
  candidate-support entries -- runs its forward with all active torch
  dispatch modes popped, the checkpoint's and any user mode on the stack (a
  `FlopCounterMode` does not see the entry's operators; a tracing mode such
  as `FakeTensorMode` or a `torch.compile` trace gets real launches on fake
  inputs and fails loudly), and behaves as in a plain non-reentrant
  checkpoint.
  `"forward"`: every call with a trainable input forms the gradients in its
  forward (the previous behavior).  The same `loss`, `logp`, `dX` and `dW`
  bits either way.  `cake_lm_head_det_train_logprob` (with or without a
  temperature; a default call also when its backward runs a budget form, an
  explicit `"budget"` call and a call with `num_valid_rows`; not its
  kept-logits form) forms no gradient in its
  forward; under `"auto"` inside a non-reentrant `torch.utils.checkpoint` its
  checkpointed forward keeps the call's row results (`logp`, the FP32 `lse`,
  the valid-row index and the compact temperature: O(T) bytes until the
  backward), and the recomputation in the backward returns a copy of that
  `logp` and saves those results instead of running the logits GEMMs again,
  so a checkpointed step runs them once, as an uncheckpointed one (paired per
  checkpoint frame and graph by call order; taken only for the same input
  tensors, unchanged since -- version counters -- and the same options, else
  the recomputation runs the forward).  The backward reads the same bits.
* `cake_lm_head_det_train_logprob` returns the differentiable `logp [T]` for
  arbitrary downstream losses; its forward saves the FP32 row statistic `lse`
  and the valid-row index, and the backward recomputes each chunk's logits
  from the saved inputs (four GEMMs per chunk) without counting the rows again
  (no host synchronization; exception: a `memory_mode="budget"` call with
  `compact_rows=False` and no caller's count counts its valid rows on the host
  in the backward -- a default call never does: uncompacted without a count it
  keeps the fast form; a caller's count checked with `check_num_valid_rows=True`
  is read back in the forward, never in the backward).
* Temperature (`temperature`, a keyword of `cake_lm_head_det_train_logprob` only,
  default `None`, the same as `1.0`; `check_temperature`, default `False`):
  `None` or a positive Python
  number or a float32 `[T]` tensor on `X`'s device (one value per row, any
  stride; not differentiable, so a tensor with `requires_grad=True` is
  rejected).  `logp_t = z[t, y_t] / tau_t - logsumexp_v(z[t, v] / tau_t)` and
  `dlogp_t / dz[t, v] = (1[v = y_t] - softmax(z_t / tau_t)_v) / tau_t`.  The
  BF16 projection is unchanged: each BF16 logit is widened to FP32 and
  divided once by an IEEE FP32 division (`q = RN(z / tau_t)`, the same `q` for
  the statistics, the label logit and the gradient), and the upstream
  `dlogp_t` is divided once per row (`RN(dlogp_t / tau_t)`); `dX` and `dW`
  see the temperature only through `dz`.  A scalar `s` gives the same bits as
  `torch.full((T,), s)`; `1.0` (and any value whose FP32 rounding is 1.0)
  runs the path without a temperature, bitwise the same as omitting it.
  Every value must be positive and finite, also on ignored rows (`-100`),
  whose temperature reaches no output (`logp`, `dX`, `dW`): on CUDA a bad
  tensor value stops the stream with a device-side assertion queued before the
  first kernel that reads the temperature (no host synchronization; it ends the
  CUDA context, like a wrong `num_valid_rows`), and `check_temperature=True`
  reads the check back once and raises `ValueError` instead.  Outputs are defined while
  `|z / tau_t| < 3e38` on every row.  The temperature runs the
  log-probability arms' temperature forms (`gemm_logits_lbl_tau`,
  `gemm_logits_dz_tau`, `row_finalize_tau` and, for the hidden valid-row count,
  `gemm_logits_mcnt_tau`), so `logprob_lbl=False` / `logprob_dz=False` are
  rejected with a temperature, and a record without these forms raises
  `NotImplementedError` for a call with a temperature (never a fallback to the
  kernels without it).  The forward saves the compact FP32 temperature rows
  for the backward (`memory_report_temperature(..., temperature=True)` counts
  them); a recount backward with a temperature (`backward_logprob_temperature`
  without the forward's `row_index`) takes the host count.  The temperature
  path has entry points of its own in both backend modules
  (`cake_lm_head_det_train_logprob_temperature`, `forward_logprob_temperature`,
  `backward_logprob_temperature`, `prepare_lm_head_det_loss_temperature`,
  `memory_report_temperature`); the public entry calls them only for a
  `temperature` other than `None`, so a call without one runs the backend
  entries without a temperature parameter.  `cake_lm_head_det_train_loss` takes
  no temperature, and `memory_mode="keep_logits"` takes none either
  (`ValueError`; its gradient form `row_grad` has no temperature form).
* Valid-row compaction (`compact_rows`, a keyword of every eager and autograd
  entry point, default `True`; `prepare_lm_head_det_loss` takes it as a bool or the
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
  (`prepare_lm_head_det_loss(..., compact_rows=True)`) fixes the valid-row set at
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
  (`prepare_lm_head_det_loss(..., dw_stream=...)`; `runner.dw_stream` holds the
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
  path; the prepared runner (`prepare_lm_head_det_loss`) is
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
* Kept logits (opt-in, `cake` backend: `cake_lm_head_det_train_logprob_keep`,
  the explicit pair `forward_logprob_keep` / `backward_logprob_keep`, the
  public entry's `memory_mode="keep_logits"`): the log-probability forward
  runs `gemm_logits` (the whole `z_c`; the label-column arm is off) and keeps
  every chunk's BF16 logits -- chunk 0 in the chunk buffer, the later chunks
  in one `[T_v - rows_0, V]` buffer (`KeptForwardResult.logits`, one
  `[rows_c, V]` view per chunk; `kept_memory_report` adds them to the
  `memory_report` of the kept plan as `saved_logits_bytes`, `2 * T_v * V`) --
  and the backward runs `row_grad` over them in place, then the dX and
  weight-gradient GEMMs: no logits GEMM in the backward (three GEMMs per chunk
  instead of four).  All logits forms of an architecture share one mainloop
  and round the same FP32 accumulator to BF16 with RN, so the kept `z_c` is
  the recompute's and `lse`, `logp`, `dX` and `dW` are bitwise the default
  path's.  The autograd entry keeps them only where a backward can follow
  (grad mode on and `X` or `W` trainable); elsewhere -- `torch.no_grad`,
  `torch.inference_mode`, the first pass of a reentrant
  `torch.utils.checkpoint`, frozen inputs, the reference backend -- it is
  `cake_lm_head_det_train_logprob` (the label-column forward).  It saves them
  with `save_for_backward`: a non-reentrant checkpoint drops them after the
  checkpointed forward and the recomputation in the backward keeps them
  again, so only the memory held between the forward and the backward drops;
  the checkpointed forward and the recomputation still allocate the kept
  `2 * T_v * V` bytes for a while, and the checkpointed step's peak includes
  them (measured at `T` = 65031 on B200 and GB300: 25501 MiB for the plain
  and the checkpointed kept step against 8461 MiB for the default ones, 0 MiB
  held after the checkpointed forward).  `row_grad` overwrites them, so a
  second backward through a retained graph recomputes (bitwise the same), and
  `backward_logprob_keep` marks the views it consumed (a second call on them
  is a `ValueError`).  A forward with a trainable input whose backward never
  runs pays the full-logits forward instead of the label-column arm
  (measured per call, both call orders combined: at `T` = 1 +0.4 to +1.6 us on
  B200 and +9.6 to +13.1 us on GB300; at `T` = 32 and 128 -3.3 to +1.2 us on
  B200 and 5.4 to 13.0 us faster on GB300) and holds the kept logits until its
  graph is freed.  The kept logits are outside the chunk bound of the memory
  rule below.  The kept form has its own entry points, plan (`_KeptPlan`),
  binding and cache keys; a default log-probability call reaches it through
  the one-chunk rule below.
  `prepare_lm_head_det_loss(..., logprob_keep=True)` prepares the kept plan for
  those entry points only (its runner's `forward` / `backward` raise
  `NotImplementedError`).  Like `"budget"`, `memory_mode="keep_logits"` is a
  `ValueError` on the reference backend (the backend-level
  `cake_lm_head_det_train_logprob_keep` runs the default entry there).
* One-chunk kept logits (the default log-probability call;
  `cake_backend_opt_in.set_logprob_one_chunk_keep`, on by default; the one
  rule is `cake_backend_opt_in._one_chunk_keep_rule`): a default call of
  `cake_lm_head_det_train_logprob` (`memory_mode=None`, no
  `num_valid_rows`, no temperature) runs the kept form above when all of
  these hold: the `cake` backend with `X` on CUDA; a backward can follow
  (grad mode on, `X` or `W` trainable -- through the public entry also with
  a frozen `W`); no `torch.utils.checkpoint` (no saved-tensor hooks on top,
  so neither pass of a non-reentrant checkpoint and no CPU offload; outside
  a backward, so not a reentrant checkpoint's recomputation, whose first
  pass runs with grad mode off); fewer than three chunks of rows
  (`cake_backend.HIDDEN_COUNT_MIN_CHUNKS`; a longer call counts its valid
  rows on the device and keeps the default form); the valid rows fit one
  chunk (`T_v <= chunk_size`; every row of an uncompacted call); and the
  fast backward
  would run no zero-gradient row skip on the call (fewer valid rows than
  the device's `cake_backend.zero_dlogp_min_rows`, or the skip off; see
  "Zero-gradient rows of the log-probability backward").  The kept backward
  has no skip: with valid rows of a zero `dlogp` its `dW` would add the rows
  in another order than the skip, so a call the skip can take keeps the
  default form, and every output stays the bits of `memory_mode="fast"`, in
  a checkpointed step too (which keeps the default form on every pass).
  The call's forward counts its valid rows once, on the host, as the
  default forward of a call of fewer than three chunks of rows does, and
  applies the rule right after the count
  (`cake_backend_opt_in.CakeLMHeadDetLogprobOneChunkFunction`): either form
  runs on that count, with no other work between the count and its first
  launch (a longer call keeps its route).  A kept call runs three GEMMs
  instead of four: its
  forward writes the whole `z_c` (the label-column arm is off) and its
  backward forms `dz` with `row_grad` over the kept logits.  Memory: the
  forward keeps `2 * T_v * V` bytes until the backward, at most one chunk
  (`chunk_size` rows) of BF16 logits, which the backward overwrites with
  `dz` in place; no logits or `dlogits` allocation spans more than one
  chunk, so the memory rule below (#5680) holds.  Each pending forward
  holds its own chunk: a caller that runs several forwards before their backwards (heads
  of several positions, for example) holds one chunk per forward until each
  backward runs, and a forward whose backward never runs holds it until its
  graph is freed.  `memory_mode="fast"` (per call) or
  `set_logprob_one_chunk_keep(False)` (per process) runs the call without it.
* Memory rule (except the opt-in kept logits above; the default call's
  one-chunk kept logits hold at most one chunk): no logits, probability or
  `dlogits` buffer ever spans more than `chunk_size` tokens; a batch smaller than `chunk_size` is one chunk, a tail
  `T % chunk_size` is neither dropped nor padded, and `T == 0` returns loss 0,
  an empty `logp` and zero gradients without binding or launching (`dW` is
  the FP32 product `0 * g` of the upstream scalar `g` cast once, as for a
  launched call: `-0` for a negative `g`, NaN for a non-finite one).
  `cake_backend.memory_report` reports the peak temporary bytes separately
  from the weights, the outputs and the FP32 accumulators.
* Opt-in forms (`cake_backend_opt_in.py`): a call that passes `memory_mode`
  or `num_valid_rows` (below) runs the planner, binding, forward, backward and
  autograd functions of `cake_backend_opt_in.py`, which reuse the kernels,
  launch helpers and per-device state of `cake_backend.py` and keep their
  own binding cache (`cake_backend_opt_in.BINDING_CACHE`, same settings).  So
  does a default call (`memory_mode=None`) whose step forms a weight
  gradient -- `W` requiring a gradient, with a BF16 `W` and `X` on CUDA on
  the `cake` backend.  With grad mode on its automatic memory form (below)
  can pick a budget form: the loss entry then runs through the opt-in
  autograd function (its fast form bitwise `cake_backend.py`'s); the
  log-probability entries without a count run `cake_backend.py`'s forward
  with a backward that picks its form
  (`cake_backend_opt_in.CakeLMHeadDetLogprobAutoFunction` and its temperature
  twin: `cake_backend.py`'s backward where the fast form fits).  With grad
  mode off (a reentrant checkpoint's first pass, an evaluation call of a
  trainable head) it runs the fast form's forward -- the loss entry's in
  `cake_backend_opt_in.py`, the log-probability entries' in
  `cake_backend.py` -- with its hidden valid-row count checked as below
  (bitwise the same outputs).  So does a default log-probability call
  without a temperature whose `X` requires a gradient under grad mode and
  whose rows fit two chunks, also with a frozen `W` (the one-chunk kept
  logits below).  Every other call without either keyword -- a
  frozen `W`, an FP32 `W`, `X` off CUDA, another backend -- runs
  `cake_backend.py`'s entry unchanged, with its buffer lifetimes; the public
  entries of `flashinfer.cake_lm_head_det_train` run their argument checks
  first (`check_num_valid_rows`, which needs a count: `ValueError` without
  one; then `memory_mode` and `num_valid_rows` against `None`, then
  `W.requires_grad`).  An explicit `memory_mode="fast"` without a count first
  goes through the dispatch of `cake_backend_opt_in.py` (a few more Python
  calls per call) and then runs the same `cake_backend.py` entry.
  Every output is bitwise the same in both, except where a default call's
  automatic form is a budget form (below).  The opt-in forms also change
  buffer lifetimes, below.
* Saved-state release (`memory_mode="budget"`, and a default call whose
  automatic form is a budget form): a backward of
  `cake_lm_head_det_train_loss` that
  frees the graph (no `retain_graph` / `create_graph`) drops the forward's
  FP32 `dX` accumulator as soon as `dX` is written -- before the weight
  gradient's buffers exist -- and the rest of the saved state when it ends,
  instead of when PyTorch frees the graph after the backward returns.  The
  backward reads `retain_graph` from PyTorch's current graph task through a
  private API (`torch._C._autograd._get_current_graph_task_keep_graph`); when
  that API is missing or cannot answer, nothing is released early (the
  previous lifetime).  A second backward through a freed graph raises
  PyTorch's "backward through the graph a second time" error;
  `retain_graph=True` keeps everything and the backward stays repeatable.
  A checkpointed default call learns its form from the recomputation in its
  backward and releases the same way.  In the fast form the saved state
  lives until the graph is freed.  In every form, `cake_backend.py`'s and
  `cake_backend_opt_in.py`'s, a compacted `backward_logprob` drops its runner
  (the chunk workspace) before it allocates the `[T, H]` `dX` output (the
  finalize or scatter); an uncompacted call's `dX` is the runner's own cast
  output, allocated with its workspace.  Lifetimes only: every output is
  bitwise the same.
* Memory budget (`memory_mode`, keyword of `cake_lm_head_det_train_loss` /
  `cake_lm_head_det_train_logprob`, default `None`, the automatic form below;
  `"fast"` always runs the fast form; `"budget"` is opt-in, `cake` backend and
  BF16 `dW` only; `"keep_logits"`, the log-probability entry's kept logits
  above, also takes `num_valid_rows` (its forward then runs on the index
  formed from the caller's count, no readback, bitwise the same outputs) and
  is a `ValueError` on the loss entry).
  The fast form keeps the FP32 `[V, H]`
  weight-gradient accumulator next to the chunk buffers, so its step peak
  (forward through backward, outputs included) is above that of a head that
  materializes the BF16 logits of the valid rows, `B = 2 * T_v * V + 2 * V *
  H + 2 * T * H` bytes.  `"budget"` picks, per call, the fastest exact
  weight-gradient form whose predicted peak stays within `B` less a 2 % margin
  (`cake_backend_opt_in.memory_plan`, which returns the form and both byte counts;
  `ForwardResult.memory_plan` carries it): `"fused"` (the fast form),
  `"fused_ip"` (two or more chunks: the fast form's kernels with
  the FP32 accumulator split at the vocabulary row `a =
  cake_backend_opt_in.dw_split_rows(V)`, half the vocabulary rounded up to
  the weight-gradient block, so each chunk runs the accumulate GEMM on rows
  `[0, a)` and `[a, V)`; a backward that frees the graph casts `dW` in place:
  the deferred last chunk's fused scale + cast GEMM writes the BF16 rows of
  `[0, a)` into the first part's own storage, in pieces: the first one, `a /
  16` rows rounded up to the block, through a temporary, and after it each
  piece `[m0, m1)` ends at most twice as far in as it starts (`m1 <= 2 *
  m0`), so its BF16 rows overwrite only FP32 rows that earlier pieces have
  read; then the rows `[a, V)`, so no `[V, H]` BF16 output is held beside
  the accumulator; a backward that keeps the graph, and
  `ForwardResult.backward`, cast out of place; the explicit pair carries the
  split as `ForwardResult.dw_split` = `(a, V)`, `ForwardResult.dw_acc` = the
  FP32 `[a, H]` rows `[0, a)` and `ForwardResult.dw_acc_hi` = the FP32 `[V -
  a, H]` rows `[a, V)`, which `backward_loss(..., dw_acc_hi=...)` with the
  deferred operands casts out of place, bitwise `ForwardResult.backward`; the
  log-probability backward runs the same split accumulator in its own chunk
  loop and always casts in place),
  `"unfused"` (every chunk accumulated, one flat cast after `dX` is written),
  `"partial_ip"` (two or more chunks: the split accumulator
  covers the vocabulary rows `[0, dw_rows)` only, `dw_rows >= a`, as the
  parts `[0, a)` and `[a, dw_rows)`; the backward keeps the deferred last
  chunk's BF16 `dz` (and the loss entry's per-row gradient scale; the
  log-probability entry's is `dlogp`), casts the rows `[0, dw_rows)`
  as `"fused_ip"` does and recomputes the other rows as below, reading the
  last chunk's block from the kept `dz`, so per block it runs one logits
  GEMM fewer than `"partial"`),
  `"partial"` (the chunk loop accumulates the vocabulary rows `[0, dw_rows)`
  only) or `"recompute"` (none).  The last three recompute the other rows of
  `dW` in the backward, one vocabulary block at a time: per block and chunk the
  logits GEMM's `dz` arm (`gemm_logits_dz`) on `W[v0:v1]` recomputes the
  block's `dz` from the saved row statistic `lse` and per-row gradient scale
  `d` (labels shifted into the block), the weight-gradient accumulate GEMM adds
  `dz_c^T @ X_c` and the last chunk runs the fused scale + cast GEMM into
  `dW[v0:v1]`.  Every element of `dW` sees the same FP32 operations in the same
  order as in the fast form, so `loss`, `logp`, `dX` and `dW` are bitwise the
  fast form's; the recompute forms run up to one more logits GEMM per chunk,
  so they are slower.  The loss entry saves `X`, `W` and the labels for the
  recompute (an in-place write between the forward and the backward raises
  PyTorch's saved-tensor version error); the log-probability entry plans its backward
  only.  `T_v` counts the valid rows also when `compact_rows=False` (the
  chunk loop then runs over all `T` rows, the budget still covers the logits
  of the valid rows only).  Without a caller's count, a `"budget"` call counts
  its valid rows on the host and ignores `hidden_count`: the hidden valid-row
  count would allocate a whole chunk's buffers before the count is known,
  which can exceed `B` when fewer rows than a chunk are valid.  The bound
  assumes a backward that frees the graph and PyTorch's caching allocator in
  its default configuration (`PYTORCH_CUDA_ALLOC_CONF` unset).  A `"budget"`
  call for which no form fits runs the form with the smallest predicted
  peak, which can exceed `B`.  The margin is measured on GB200, GB300 and
  B200 (other allocator settings are not measured): every measured
  `"budget"` step peak of a call whose plan fits, single calls and repeated
  calls that miss or hit the binding cache, at the issue's shapes and at
  small heads, stayed within `B`; the highest reached 0.9794 `B`.  The
  forms whose `dW` the cake kernels finish (`"fused_ip"`, `"partial_ip"`,
  `"partial"`, `"recompute"`) take no `backend="reference"` override in
  `ForwardResult.backward` (`ValueError`; their forward runs on the `cake`
  backend only).  No environment variable selects it.
* Automatic memory form (the default, `memory_mode=None`;
  `cake_backend_opt_in.MEMORY_AUTO`, accepted by the resolver but not one of
  `MEMORY_MODES`): a call that forms a BF16 `dW` on the `cake` backend on
  CUDA runs the fast form while its predicted step peak, with the fast form's
  own buffer lifetimes (the FP32 `dX` accumulator kept through the weight
  gradient; `cake_backend_opt_in.fast_form_fits`), stays within `B` less the
  2 % margin, else the form `"budget"` picks for it when that form fits
  (`cake_backend_opt_in.auto_memory_plan`), else the fast form: no budget
  form fits a tiny `T_v`, where the outputs and the allocator's rounding
  alone exceed the budget, and recomputing in single vocabulary granules
  there would cost time for no bound.  The decision is host integer
  arithmetic on the call's shapes and counts (no allocator state, free
  memory or environment variable), so a checkpoint's recomputation picks the
  same form.  It plans on the valid rows of the forward's count (compacted,
  the hidden valid-row count included) or the caller's `num_valid_rows`; an
  uncompacted call without a count keeps the fast form (`B` is not enforced
  there: it never counts on the host).  The log-probability entries decide
  in the backward (the bound covers it), from the forward's row index.
  `cake_backend_opt_in._AUTO_MIN_CHUNKS` sets the smallest chunk count of
  the rows the fast form chunks over that is planned at all; a shorter call
  keeps the fast form (its step peak is then not bounded by `B`).  The
  shipped value 2 plans every call of at least two chunks.  A multi-chunk call fits
  the fast form roughly
  while `T_v * (1.96 V - 4 H) >= 4.04 V H + 2 C V` (the fast form's last
  phase: the `dX` output, the FP32 `dX` and `[V, H]` accumulators, the
  deferred chunk's logits and the `dW` output against `0.98 B`): about 18 400
  valid rows at `H = 6144`, `V = 154880`, `C = 4096`, about 13 500 at
  `H = 4096`.  A one-chunk call always exceeds `B` in the fast form (its
  deferred chunk holds the logits of every valid row beside both gradient
  outputs and the FP32 `dX` accumulator), but with the fused weight-gradient
  cast (the default) it holds no FP32 `[V, H]` accumulator, so its excess is
  small; with the shipped value 2 it keeps
  the fast form and its step peak exceeds `B` (measured 1.02-1.12 `B` at
  `H = 6144` and `H = 4096`, `T` 2048-4097, 5 % of the labels ignored),
  while a budget form there recomputes most of the vocabulary and costs
  15-35 % of the step.  A two-chunk call holds the FP32 `[V, H]`
  accumulator beside a small `B` (measured 1.67-1.71 `B` at `T = 8117`)
  and runs a budget form.  Below
  the threshold the cost depends on the form the call gets (B200, step
  medians of the default and `"fast"` calls in one process, `C = 4096`):
  at the issue's shapes (`T = 16384`, a third of the labels ignored) the
  step peak went from 6.971 to 5.015 GiB at `H = 6144` (`"partial_ip"`,
  42.6 against 41.5 ms, about 3 % slower) and from 5.041 to 3.838 GiB at
  `H = 4096` (`"fused_ip"`, 27.3 against 27.2 ms, within the noise); with
  5 % of the labels ignored, at `H = 6144`, from 7.081 to 5.430 GiB
  (`"fused_ip"`, 57.0 against 56.9 ms, within the noise; a checkpointed step
  about 2 % slower); at `T = 8117` (two chunks, `H = 6144`, `"partial_ip"`)
  from 6.811 to 3.996 GiB, 31.8 against 29.7 ms (about 7 % slower).  The
  log-probability entry's backward gets the same forms: `"partial_ip"` at
  `T = 16384` with a third of the labels ignored (7.081 to 5.015 GiB, 54.5
  against 53.0 ms, about 3 % slower) and at `T = 8117` (7.005 to 3.996 GiB,
  40.6 against 38.2 ms, about 6 % slower), `"fused_ip"` at `H = 4096` (4.989
  to 3.838 GiB, 36.5 against 35.7 ms, about 2 % slower) and with 5 % of the
  labels ignored (7.185 to 5.430 GiB, 76.6 against 75.8 ms, within the
  noise).  A short
  call that recomputes most of the vocabulary costs more.  The margin above was measured for the `"budget"`
  form; the fast form's own-lifetime model is checked against measured peaks
  of the issue's shapes.  The hidden valid-row count allocates a whole
  chunk's buffers before the count is known.  A call that misses the binding
  cache (a new valid-row count: usually every training batch) prepares its
  binding on views of those buffers (the template's logits, statistics and
  gather regions; the rest of the chunk workspace and the aliased
  accumulators in one allocation), so no second chunk buffer exists beside
  them and that phase never exceeds the forward's.  A default call whose step
  forms a BF16 `dW` -- every pass of it, a checkpointed step's gradient-free
  first pass included -- keeps the count only where those chunk-0 buffers
  and the call's `O(T)` vectors fit the budget of a call without a valid
  row, `2 * V * H + 2 * T * H` less the margin, which bounds that phase for
  every valid-row count (at `V = 154880`, `C = 4096`: from `T = 10264` at
  `H = 4096`, from `T = 33418` at `H = 3584`); otherwise it runs without
  the count, on the host count's path.  Its outputs are the same bits: the
  loss entry's zero-weight row split is still decided as for the call with
  the count.  A call of fewer valid rows than a chunk that keeps the count
  holds the count's `chunk_size`-row buffers (a one-chunk call, whose step
  peak is not bounded by `B`, above).  A
  call that keeps the count plans its budget form with the count's chunk-0
  buffers (`chunk_size` rows, the gather buffer included) and that
  cache-miss phase, and keeps the count when that plan fits; else, when the
  plan with the valid rows' buffers (an explicit `"budget"` call's) fits, it
  drops the count's queued chunk-0 work and runs that plan; else it runs the
  fast form with the count.  A loss call with zero-weight policy rows checks
  the fast call the memory rule picks (the split or the call without it,
  with the logp-only pass where it runs) and, when it does not fit and a
  budget plan does, runs as a `"budget"` call (without the hidden count).
  The package reads no environment variable; through the public facade the
  `FLASHINFER_CAKE_LM_HEAD_LOSS_*` variables (`COMPACT_ROWS`,
  `FUSE_DW_CAST`, `DX_FINALIZE`, `HIDDEN_COUNT`, `GEMM_TUNING`) set switches
  that are inputs of this rule -- an uncompacted default call keeps the fast
  form, `fuse_dw_cast=False` removes the fused candidates, the hidden count
  changes the chunk-0 buffers and an explicit `k_slices` the dX slabs -- so
  they can change the chosen form.  Bits: the budget forms are bitwise the
  fast form's (above), with two exceptions, both within the accuracy budgets
  and the same on every run: a log-probability backward in a budget form runs
  without the zero-gradient row skip (its `dX` / `dW` bits can differ from
  the fast form's where valid rows have a zero `dlogp`), and a loss call with
  zero-weight policy rows decides its split as `"budget"` does, which can
  differ from the fast form's decision (see "Zero-weight policy rows of the
  loss entry").  Behaviour of a call that switches: the loss entry's backward
  that frees the graph releases its saved state early (a second backward
  raises), and a recompute form saves `X` and `W` (an in-place write between
  the forward and the backward raises).  The candidate-support entries
  (`cake_support.py`) have no automatic form: they keep their FP32 `[V, H]`
  accumulator.
* Caller's valid-row count (`num_valid_rows`, keyword of
  `cake_lm_head_det_train_loss` / `cake_lm_head_det_train_logprob` and of the
  `cake_backend_opt_in` eager entry points, default `None`): the number of labels
  `>= 0` as a host integer (a Python or NumPy integer or a CPU scalar tensor;
  a CUDA tensor is rejected because reading it would wait for the device).
  A compacted forward then plans its chunks on that count without reading a
  count back: it forms the valid-row index on the device
  (`cake_backend_opt_in.given_valid_rows`: `labels >= 0` -> int32 `cumsum` ->
  `searchsorted`, whose first `num_valid_rows` slots are the ascending valid
  rows) and queues one `torch._assert_async(scan[T - 1] == num_valid_rows)`
  ahead of every kernel that reads the index.  A wrong count therefore stops
  the stream with a device-side assertion, which ends the CUDA context.
  `check_num_valid_rows=True` reads the count back instead and raises
  `ValueError` on a mismatch (one synchronization, for debugging); so do
  CPU calls and every `sm_103a` call whose library CUDA 13.0 built (the calls
  that keep the host-count schedule, see the toolchain note).  The given-count
  call runs the host-count path's kernels on the same plan, so its outputs are
  bitwise those of `num_valid_rows=None`.  An uncompacted call
  (`compact_rows=False`) forms no index from the count (an uncompacted call
  of a budget form, explicit or automatic, sizes its budget on it); there
  only
  `check_num_valid_rows=True` checks it, by reading the labels' count back.
  So with `check_num_valid_rows=True` every call that takes the count checks
  it once (the autograd entries in their forward) and raises the same
  `ValueError` on a mismatch, whatever `compact_rows`, `memory_mode` or the
  inputs that need a gradient.  The count is not derived from `loss_div`.
* First call of a process (`cake_lm_head_det_train_preload(device=None, *,
  temperature=False)`, `cake_backend_opt_in.cake_lm_head_det_train_preload`): loads every library of
  the device's architecture except the temperature forms
  (a JIT build when the package ships no `aot/` library; `temperature=True`
  also loads the temperature forms and launches the temperature path's
  PyTorch operators once, so a caller that never passes a temperature builds
  no temperature library), creates the
  per-device side stream (the process's first `torch.cuda.Stream`), the count
  cells and constants, launches the PyTorch operators of the valid-row count
  paths once and synchronizes.  It does not launch the package's kernels:
  under the driver's lazy module loading (`CUDA_MODULE_LOADING=LAZY`, the
  default) each kernel form still loads at its first launch.
* The det name is the mode: fixed sequential chunk order, bitwise
  reproducible `loss`, `logp`, `dX` and `dW` run to run.  No entry, function
  or autograd `Function` of this package takes a `deterministic` keyword
  (passing one is a `TypeError`), and torch's deterministic mode
  (`torch.use_deterministic_algorithms`) changes nothing.

Explicit forward / backward entry points without autograd (`forward_loss`,
`backward_loss`, `forward_logprob`, `backward_logprob`), a prepared
allocation-free runner (`prepare_lm_head_det_loss`, CUDA-graph capturable) and the
workspace sizing helper (`lm_head_det_loss_workspace_size`) live in
`cake_backend.py`.  The eager entry points (and the autograd wrappers behind
the public API) validate and bind once per input binding -- `(data_ptr, shape,
stride, dtype)` of every input plus the options -- and launch later calls from
the remembered argument plans with freshly allocated outputs and per-call
scratch (`cake_backend.BINDING_CACHE`; the calls through `cake_backend_opt_in.py`
-- the default loss calls that form a weight gradient among them, and a
log-probability backward in a budget form -- in
`cake_backend_opt_in.BINDING_CACHE`).  A remembered binding pins no caller
tensor and holds no problem-sized scratch: the chunk workspace, the FP32
accumulators and the outputs come from the caching allocator on every call.
In the opt-in forms (default loss calls that form a weight gradient
included) a call that misses the cache -- with compaction, every
call whose valid-row count or `loss_div` is new, so usually every training
batch -- binds on
buffers that share one allocation of the largest of them (the binding is
built from their addresses and never launches on them), so its peak stays
close to that of a call that hits the cache instead of adding up every
buffer the binding needs.
The cache keeps up to 64 bindings (`cake_backend.BINDING_CACHE_DEFAULT_CAPACITY`)
and evicts the least recently used one;
`cake_backend.BINDING_CACHE.configure(enabled=..., capacity=...)` changes both
settings at runtime (a shrink evicts, `enabled=False` bypasses the cache and
drops the remembered bindings).  It configures that cache only:
`cake_backend_opt_in.BINDING_CACHE` (the default loss calls that form a
weight gradient, a log-probability backward in a budget form) takes its own
`configure` call; the facade's two binding-cache variables apply to both.  `backend="reference"` on the `cake_backend`
entry points runs the same chunk loop with PyTorch operators at the same
rounding boundaries on any device (the host-layer tests use it); the public API
accepts `backend="cake"` only.

## Candidate-support entry points (`cake_support.py`)

```python
from flashinfer.cake_lm_head_det_train import (
    cake_lm_head_det_train_support_kl,
    cake_lm_head_det_train_support_logprob,
)

logp = cake_lm_head_det_train_support_logprob(X, W, ids, offsets, k_cap=k_cap)          # packed [M] FP32
sel = cake_lm_head_det_train_support_logprob(X, W, ids, offsets, labels, k_cap=k_cap)   # selected [T] FP32
kl, logp = cake_lm_head_det_train_support_kl(X, W, ids, offsets, teacher_logprobs, k_cap=k_cap,
                                          temperature=tau, return_logprobs=True)
```

* A ragged candidate support per token (sampling-mask replay, restricted-support
  self-distillation): row `t` normalizes `z = X @ W^T` over its candidates
  `ids[offsets[t]:offsets[t + 1]]` only.  `ids [M]` and `offsets [T + 1]` are
  int32 or int64; ids are unique inside a row, in any order, and repeat freely
  across rows; `K_t <= k_cap` (a recorded support is never trimmed).  Limits:
  `H` and `V` multiples of 256, `V <= 1572864` (the validation stage's
  shared-memory bitmap), `M <= 2^31 - 2^16`.
* Numerics: the BF16 logits at the support (FP32 accumulation, one rounding),
  then FP32 `q = RN(z / tau_t)`, an exact row maximum, the row sums in a fixed
  32-lane order, precise `expf` / `logf`, no fast math.  Rows of one candidate
  or none give `logp = 0` and `KL = 0` (selects).  `KL_t = sum_j Q_j (lq_j -
  logp_j)` over the entries with `Q_j = exp(lq_j) > 0` of a detached FP32
  teacher aligned with `ids`.  Gradients flow to `X` and `W` only (BF16 `dz` at
  the support, FP32 accumulation, one cast each); the temperature (a positive
  number or an FP32 `[T]` tensor, the same rules as `cake_lm_head_det_train_logprob`,
  checked on every row) is a constant.
* Validation: one kernel forms an int64 error word over the offsets, the cap,
  the id range, repeated ids and the labels (`-100` ignored; any other label
  must be one of the row's candidates; a label on an empty row is an error).
  `check_inputs=False` queues an asynchronous device assertion (no
  synchronization; a violation ends the CUDA context, like an out-of-range
  PyTorch index); `check_inputs=True` reads the word back once and raises
  `ValueError` naming the rule and the row.  Every later kernel clamps the
  indices it reads, and the sparse `dW` pass drops a pair whose id is outside
  `[0, V)`, so it stays in bounds for any input.
* Passes: the logits at the support, the forward row stage, the backward row
  stage (`dz [M]`), `dX` and `dW`.  Each of the logits, `dX` and `dW` passes has
  a sparse form (work proportional to `M`: a gather GEMM over tiles of 128
  pairs; one CTA per token adding its pairs' `W` rows; the sorted unique keys
  `(id << 32) | pair` and one owner per id adding its pairs' `X` rows, a long
  segment cut into parts summed in part order) and a dense form (the shipped
  logits / dX / weight-gradient GEMMs per token chunk and vocabulary block on
  `W[v0:v1]` views, plus a gather / scatter at the support; `H` a multiple of
  512 for dX / dW).  A host rule on `M / T` picks the form per pass
  (`cake_support.SUPPORT_DENSE_MIN_MEAN_K`, constants calibrated per
  architecture); no
  synchronization.  The sparse `dW` is stored directly in BF16 up to `2^23`
  pairs; above that pair ranges accumulate in an FP32 `[V, H]` sum.
* Memory: no `[T, V]` or `[chunk, V]` tensor in the sparse form; the packed
  state is `O(M)` (BF16 `z` and `dz`, int32 copies of int64 indices) and
  `O(T)`; `cake_support.support_memory_report(T, H, V, M, ...)` gives the device
  bytes of a step by bucket, with the measured `torch.sort` bytes per key
  (`SORT_BYTES_PER_KEY`) and the split of a long `dW` segment chosen to keep its
  FP32 partial rows within 128 MiB (table below).
* Determinism: one owner and one order per sum, no atomics on values; reruns
  are bitwise equal.  The bits of `dX` / `dW` depend on the pass form,
  `chunk_size`, the dense vocabulary block and the sparse split (fixed
  functions of the sizes).  `backend="reference"` on the `cake_support`
  entry points runs every stage with PyTorch operators at the same rounding
  boundaries (a development aid that materializes `[chunk, V]` blocks).

Device bytes of one step from `support_memory_report` at `T = 16384`,
`H = 6144`, `V = 154880` (packed log-probabilities and KL, int64 indices;
MB = 10^6 bytes).  Each workspace column counts every temporary of its phase
at once, so it is an upper bound.

| pass forms, pairs `M` | packed state + outputs | forward workspace | backward workspace | FP32 `dW` sum | BF16 `dX` + `dW` |
|---|---|---|---|---|---|
| sparse, `2^20` | 11 | 0 | 166 | 0 | 2104 |
| sparse, `2^23` (the largest BF16-stored `dW`) | 84 | 0 | 623 | 0 | 2104 |
| sparse, `2^24` (`dW` in ranges of `2^22` pairs) | 168 | 0 | 387 | 3806 | 2104 |
| dense, `2^23`, `chunk_size` 4096 | 84 | 128 | 346 | 3806 | 2104 |
| dense, `2^23`, `chunk_size` 16384 | 84 | 512 | 1334 | 0 | 2104 |

The sparse backward workspace is `dz` (2 B per pair), 60.25 B per pair of
the `dW` pair range (pair and row indices, keys, sort) and at most 128 MiB
of FP32 partial rows; it does not depend on `V`.  The dense workspace is `dz`
plus blocks of `chunk_size` rows and at most 16384 vocabulary columns.

## Zero-weight policy rows of the loss entry

A policy row with a valid label whose `loss_weights` entry is exactly zero
(`+0` or `-0`) contributes exactly zero loss and zero gradient.  Its weight is
known before the forward, like a `-100` label, so the loss entry leaves such a
row out of the gradient path:

* The chunk loop runs over the gradient rows `G = (labels >= 0) &
  (loss_weights != 0)` only: the row compaction uses this mask (the host
  count path and the hidden valid-row count), and both row counts come back
  in the call's one count readback (one 8-byte copy; no further host
  synchronization).  A NaN weight keeps its row in `G`.
* The zero-weight rows `Z = (labels >= 0) & (loss_weights == 0)` run no
  `row_grad`, dX GEMM or weight-gradient GEMM.  Their `dX` rows are exact
  `+0`.  With `return_logp=True` their `logp` comes from a logp-only pass of
  the log-probability forward kernels (`gemm_logits_lbl` + `row_finalize`,
  chunk-bounded buffers) into the same `[T]` output; without `return_logp`
  that pass does not run.
* A call whose valid rows all have zero weight launches no gradient GEMM: both
  gradients are zeros and the loss is the plain path's (`+0`).  It holds no
  FP32 `[V, H]` weight-gradient accumulator from the forward to the
  backward; the backward allocates the zero `dW` in its output dtype only.
* The logp-only pass runs ahead of the chunk loop, except when the hidden
  valid-row count already holds chunk 0's buffers and either the chunk loop
  keeps no buffer until the backward (`W` without a gradient, or
  `fuse_dw_cast=False`) or no chunk loop follows: it then runs after the loop
  (or in its place) once those buffers are released.  With the hidden count,
  a weight gradient and `fuse_dw_cast`, it runs where the byte model below
  predicts the lower peak: ahead of the loop beside chunk 0's buffers, or
  after it beside the accumulators and the deferred chunk's `dz` rows (ahead
  on a tie).  The memory report adds
  `zero_weight_rows`, `zero_weight_loss_mode`,
  `zero_weight_logp_temporary_bytes`, `zero_weight_logp_phase_bytes` and, in
  the exact mode, `zero_weight_exact_loss_temporary_bytes`.
* A call of fewer than `cake_backend.zero_weight_min_rows()` rows `T` (default
  4096, `set_zero_weight_min_rows`) keeps its zero-weight rows in the chunk
  loop.  The default is the floor timed for the non-deterministic package's
  split of the same entry; this package re-times it before relying on it.

### The loss: `"exact"` (default) and `"regroup"`

`cake_backend.set_zero_weight_loss_mode(mode)` (process setting, returns the
previous mode; `zero_weight_loss_mode()` reads it):

* `"exact"`: the loss is bitwise the loss of the same call without the split.
  The plain path forms one term per valid row in `row_finalize`
  (`term = w * clipped`, `clipped = ratio < 2 ? ratio : 2`, so a NaN or
  infinite ratio clips to 2) and reduces the terms chunk by chunk over the
  valid rows in `loss_reduce`: one 256-thread CTA, thread `i` adds
  `term[i], term[i + 256], ...` into an FP64 register that starts at `+0`,
  then a warp butterfly and a cross-warp butterfly; the first chunk stores
  the chunk sum in the FP64 accumulator, every later chunk adds its sum, and
  the last chunk stores `float(0 - acc)`.  The split keeps every chunk's
  terms without another buffer: chunk 0's stay in the chunk-local `term`
  buffer, chunk `c >= 1` writes its terms into the compact `loss_weights`
  rows of chunk `c - 1` (gathered before the loop and read by no launch
  after that chunk's `row_finalize`).  After the loop the terms go to their
  rows' positions among the valid rows (`k + |{z < g_k}|`), each zero-weight
  row's slot takes its weight (`+0` / `-0`: the term the plain path forms for
  that row for any ratio), and the shipped `loss_reduce` runs over the plain
  path's chunks of that array (`ceil(T_v / C)` launches, the same `rows_c`
  and first / last flags) on the call's stream.  Every thread adds the same
  values in the same order, so the FP64 sums and the FP32 rounding are the
  plain path's.
* `"regroup"`: the chunk loop's own `loss_reduce` over `G` (the same FP64 sum
  of the same nonzero terms over the split's chunk boundaries); the loss may
  differ from the plain path's in its last FP32 bits.  No term placement and
  no extra launch.

### Bits against the call without the split

| output | bits |
| --- | --- |
| `logp`, `lse` of every row (`Z` rows from the logp-only pass) | identical (the same per-row logits and row-statistics arithmetic) |
| `loss`, mode `"exact"` | identical, non-finite values in `Z` rows included (their term is `+-0` either way) |
| `loss`, mode `"regroup"` | the same FP64 sum over other chunk boundaries: may differ in the last FP32 bits |
| `dX` row of a `Z` row | exact `+0` (the plain path: `+-0`, or NaN through `0 * inf` when the row's `X`, logits or `infer_logp` is not finite) |
| `dX` row of a `G` row | identical when its chunk runs the same dX form and K-slice count in both calls; otherwise within the accuracy budget |
| `dW` | the same products, summed over the split's chunk rows (the K reduction groups differ): within the accuracy budget; a `Z` row with a non-finite `X` row or logit no longer makes it NaN |

Run to run every output is bitwise the same, as everywhere in this package.

### Memory

The split never holds more of the plain path's per-row buffers: on the host
count path the index of `G` and the index of `Z` are formed at their exact
sizes (`8 |G| + 8 |Z| = 8 T_v` bytes, the plain index's); on the hidden
valid-row count path the `[T]` gradient-row index takes the place of the
`[T]` valid-row index and the `Z` index adds `8 |Z|`; the gradient-row mask
kept for the fused dX finalize is `T` bytes (the valid-row mask's); the chunk
loop's compact row vectors and `dX_acc` cover `G` only.  Per phase, against the same call
without the split (`Cb = min(C, |G|) <= Cm = min(C, T_v)`):

| phase | bytes added | bytes saved |
| --- | --- | --- |
| count (before the loop) | the `Z` mask and the second int32 scan, `T + 4 T` (released before the loop) | -- |
| chunk loop | `Z` index `8 \|Z\|` (hidden valid-row count path only); the `Z` rows' `logp` `4 \|Z\|` when the pass ran first | compact rows `24 \|Z\|` (labels, `lse`, `logp`, `infer_logp`, `loss_weights`); `dX_acc` `4 H \|Z\|`; chunk buffers of `Cm - Cb` rows |
| exact loss (after the loop, mode `"exact"`) | term array `4 T_v`, positions `8 \|G\|` (then `12 \|Z\|` for the `Z` slots); the two kept term buffers `4 \|G\| + 4 Cb` | the loop's statistics (`8 Cm ceil(V / 256)`), `lse` (`4 T_v`), `d` / `term` (`8 Cm`), gather buffer (`2 Cm H`) and, when compacted, `infer_logp` / labels (`12 T_v`), released before this phase; `dX_acc` `4 H \|Z\|` |
| logp-only pass | its chunk buffers of `min(C, \|Z\|)` rows and `O(\|Z\|)` vectors | runs before the loop allocates (or after it released its buffers) |

The chunk loop and the exact-loss phase stay below the plain path's chunk
loop for every `|Z| >= 1` within the documented geometry (the exact-loss
phase's additions are covered by the released statistics and `lse` alone
while `12 |G| <= Cm * V / 32`, at the default chunk and `V = 154880` up to
1.6 million gradient rows).  Two further terms depend on the shapes, and the
split bounds both:

* **dX K slices.** The K-slice workspace (`4 H Cb` bytes per slab) follows the
  largest slice count over the chunks, and the slice rule gives a small tail
  chunk more slices.  The split's chunk loop therefore runs no chunk at more
  slices than the largest count of the same call without the split
  (`dx_slices_cap`; a count above it becomes the cap fitted to the K steps).
  A chunk the cap lowers runs another dX slice count than the rule picks, so
  its `dX` rows can differ in their last bits (within the accuracy budget,
  as any `G` row of another dX form); an explicit `k_slices` knob is the
  same in both calls and is not capped.
* **The guard.** After the count readback the host predicts both peaks from
  the two counts (`cake_backend.zero_weight_memory`, no further
  synchronization; `zero_weight_plan` adds the slice counts of this device):
  `plain = R(T_v) + A(T_v)` and `split = max(loop, pass, exact)` with
  `loop = R(|G|) + A(|G|) + 8 |Z| (+ 4 |Z| when the pass ran first)`,
  `pass = P + I (+ chunk 0's logits / statistics / gather buffer when it runs
  ahead of a hidden count) or P + I + A(|G|) + dz + 4 (after the loop)`,
  `exact = I + A(|G|) + dz + 4 T_v + max(8 |G|, 12 |Z|) + 4 |G| + 4 Cb`, where
  `R` / `A` are `memory_report`'s temporaries / accumulators of a chunk loop
  over that many rows (with its K-slice slabs), `P` the logp-only pass's
  workspace plus `8 |Z|`, `I = 8 |G| + 8 |Z| (+ T for the fused dX
  finalize's mask)` the held indices and `dz` the deferred chunk's rows.  A
  call with `split > plain` runs without the split (its valid-row index
  formed on the device from the two counts; on the hidden count path chunk
  0's queued gather and logits GEMM are dropped).  A budget form
  (`memory_mode="budget"`, or a default call whose automatic form is one)
  compares `memory_plan`'s predicted peaks of the two calls (the same
  budget) instead.

The model leaves out what both calls hold alike (the outputs, the inputs) and
some small terms: the record's TMA-descriptor and scratch bytes of the
logp-only pass, the allocator's block rounding, the count phase's transient
masks and scans before the loop (`~5 T` bytes), and reserved (cached) bytes.

Not applied (the plain path runs, every valid row in the chunk loop): a call
below the row floor above, `compact_rows=False`, `objective="ce"`, the
log-probability entry (see the next section), the prepared runner
(`prepare_lm_head_det_loss`, a fixed row set) and a caller-given
`num_valid_rows` (its plan needs no readback, so the split count is not
available).  A budget form (explicit or automatic) splits the rows too: its
chunk loop and step-peak model run over the gradient rows, its bound still
counts every valid row, and the model adds the logp-only pass and the exact
loss's term phase (`memory_plan(..., zero_rows=..., zero_total=...,
exact_loss=...)`).  The
loss entry has no temperature, so the split has no temperature form.

## Zero-gradient rows of the log-probability backward

A valid row whose upstream gradient `dlogp` is exactly zero (`+0` or `-0`;
for example a token the caller's loss masks) contributes exact zeros to `dX`
and `dW`.  The log-probability backward knows `dlogp` only when it runs, so
it handles these rows on the device, without a host synchronization:

* The backward orders the forward's valid rows so that the live rows (a
  nonzero `dlogp`, NaN included) come first, each group in ascending row
  order, and forms the per-chunk count of live rows in device memory, in one
  kernel launch per call (`zero_dlogp_partition`; the same outputs as its
  PyTorch reference `zero_dlogp_partition_reference`, bit for bit).  The
  order is a stable partition formed from an inclusive scan: a function of
  the labels and of which `dlogp` entries are zero.
* The chunk's dz arm of the logits GEMM (`gemm_logits_dz_dcnt`; with a
  temperature `gemm_logits_dz_dcnt_tau`), its weight-gradient accumulate
  (`gemm_dw_acc_dcnt`, `gemm_dw_acc_gn_dcnt`) and its bf16 fused cast
  (`gemm_dw_cast_bf16_dcnt`, `gemm_dw_cast_bf16_tma_dcnt`) read the chunk's
  count and skip the work items (dz) or the K steps (dW) past it.  The launch
  shape stays the chunk's full row count, so the call is the same set of
  launches for every `dlogp` and a CUDA graph capture of the backward replays
  for other `dlogp` values.  These forms are this package's GEMM arithmetic
  with a device row bound: each output element has one owner per launch and
  the chunks run in order.
* The dX GEMM has no counted form in this package: it computes every row of
  each chunk with the forms and K-slice counts the plan takes without the
  skip (the chunks keep their sizes).  The rows past a chunk's count may read
  `dz` rows the counted dz arm did not write, so their `dX_acc` rows hold
  values that nothing reads: the fused dX finalize (`scale_cast_scatter_bf16`)
  addresses the accumulator through the partition's live-row scan only and
  writes exact `+0` on every other row.  The skip therefore requires the
  fused dX finalize whenever the call forms `dX`.
* Memory: the call holds the partition's int64 row order (8 bytes per valid
  row), int32 live-row scan (4 bytes per row of `T`) and per-chunk counts
  until the dX output boundary, and its FP32 reordered `lse` (4 bytes per
  valid row) through the chunk loop.  In exchange it forms neither the bool
  valid-row mask nor the finalize's int32 scan, and its chunk loop binds no
  row-statistics buffer (`min(T_v, chunk_size) * ceil(V / 256) * 8` bytes,
  which no launch of this backward writes or reads).

Bits.  Every output is the same on every run.  `logp` and `lse` come from
the forward, which the skip does not change.  A skipped row's `dX` row is
exact `+0`, except when every valid row's `dlogp` is zero: the first valid
row (the partition's `order[0]`) then still runs one GEMM tile as a live
row, and its `dX` row is `+0` for finite inputs but NaN when that row's `X`,
logits or `lse` (or `W`) are non-finite.  A live row's `dX` row is the same
per-row arithmetic; its chunk
can change (the live rows move to the front), and with it that chunk's dX
K-slice count, so its bits can differ from a call without the skip within
the accuracy budgets.  `dW` sums the same nonzero products over the live
rows in the new row order (FP32), so its bits can differ from a call without
the skip within the accuracy budgets.  A call with no zero-gradient row keeps
the row order, the chunks and the K steps of the call without the skip:
`dX` and `dW` are then bit-identical to it.

Non-finite values: a NaN `dlogp` keeps its row live (NaN in its `dX` row and
in `dW`, as without the skip).  A non-finite `X` row or logit of a
zero-gradient row makes `dW` NaN without the skip (`0 * inf`); with the
skip it still does when the row falls within the 63 rows after a chunk's
live rows (the weight-gradient GEMM's last 64-row K step), and not
elsewhere.  With the skip its `dX` row is exact `+0` (except the first
valid row when every `dlogp` is zero, above); without the skip a
zero-gradient row's `dX` row can be NaN (`0 * NaN`).

The skip applies to a compacted backward that reuses the forward's row index
(the autograd entries), with the dz arm (`logprob_dz`) and, when it forms
`dX`, the fused dX finalize (`dx_finalize`), on a record that registers the
`_dcnt` forms and the partition kernel, of at least
`cake_backend.zero_dlogp_min_rows(device)` valid rows: the override of
`cake_backend.set_zero_dlogp_min_rows` when one is set (an integer, or
`None` = never, on every architecture), else the architecture's default
floor below.  With no override set the setter returns
`ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT`; passing it back (or a copy or an
unpickled copy of it) restores the per-architecture default, and
`zero_dlogp_min_rows()` without a device returns 4097.

| architecture | floor (valid rows of one call) | source |
| --- | --- | --- |
| sm_100a (B200) | 1946 (`T` = 2048 with 5 % ignored rows) | `ZERO_DLOGP_MIN_ROWS_BY_ARCH` |
| sm_103a (GB300) | 3891 (`T` = 4096 with 5 % ignored rows) | `ZERO_DLOGP_MIN_ROWS_BY_ARCH` |
| any other (sm_107a included) | 4097 (`ZERO_DLOGP_MIN_ROWS_DEFAULT`, one row above a default chunk) | not timed below one default chunk |

The call's plan decides: under the skip the last chunk's bf16 fused cast runs
its counted form -- the counted TMA-epilogue form where the plan without the
skip takes the TMA-epilogue cast (`dw_cast_tma_rule`), else the counted
per-thread form; the same `dW` bits either way -- and a plan with a form
that has no counted one (an explicitly pinned weight-gradient TMA epilogue,
an fp32 fused cast) runs the backward without the skip: no partition
launch, rows in the forward's order.  A call with a temperature runs the
counted temperature form of the dz arm
(`cake_backend.set_zero_dlogp_temperature_dz_counted(False)`, or a record
without it: the uncounted `gemm_logits_dz_tau` over every row, the same
outputs) beside the counted dW forms; a per-token temperature is gathered in
the partition's order.

Not applied (the backward runs as without the skip): `compact_rows=False`
or a call whose labels are all valid (no row index), a backward that
recounts its valid rows (no forward row index), `logprob_dz=False`,
`dx_finalize=False` with `dX`, the kept logits (`memory_mode="keep_logits"`),
the opt-in forms (`memory_mode="budget"`, `num_valid_rows`) and a default
call's backward in a budget form (its automatic form: its `dX` / `dW` bits can
then differ from the fast form's for rows with `dlogp == 0`, within the
accuracy budgets, the same on every run), the prepared runner, the reference
backend and a temperature backward given `temperature_rows` without the
temperature itself.

## Switches and environment variables

Every switch above is a keyword argument of each entry point it affects
(`cake_lm_head_det_train_loss`, `cake_lm_head_det_train_logprob`, `forward_loss`,
`backward_loss`, `forward_logprob`, `backward_logprob`,
`prepare_lm_head_det_loss`; `prepare_lm_head_det_loss` takes `compact_rows` as a bool or
the valid-row tensor, default `False`, and no `hidden_count`; `forward_logprob`
and `backward_loss` take no `dw_stream`; `logprob_lbl` belongs to the
log-probability forward and `logprob_dz` to its backward), `None` selecting the documented default
(`cake_backend.compact_rows_default()` and friends return the constants); the
autograd entries resolve every switch once in the forward.  **The package reads
no environment variable.**  The `FLASHINFER_CAKE_LM_HEAD_LOSS_*` variables are
FlashInfer's own A/B knobs, read only by the LM-head facades (through
`flashinfer/_cake_lm_head_train_knobs.py`): a
switch keyword left at `None` there falls back to its variable and the resolved
value is passed down; the two binding-cache variables are read when that
module is imported and applied through `BINDING_CACHE.configure` when it first
reaches the backend.  A build that vendors the package without the wrapper
sees the backend defaults.  The wrapper leaves `logprob_dz` / `logprob_lbl`
at their backend defaults (no variable), reaches the kept logits through
`memory_mode="keep_logits"` only and passes `memory_mode` (no
variable either) and the non-switch keywords `num_valid_rows` /
`check_num_valid_rows` through unchanged; it routes a call whose
`memory_mode` is `None` (the automatic form) to `cake_backend_opt_in.py` when
`W` requires a gradient, whatever the grad mode: with grad mode on its
automatic form can pick a budget backward, with grad mode off (a reentrant
checkpoint's first pass) its hidden valid-row count is still checked against
the step's bound.  The switch variables are inputs
of the automatic form's rule, so they can change its chosen form.  The gradient placement of the loss
entry is a process setting of the backend
(`cake_backend.set_gradient_placement`, see "Public entry points"), neither a
keyword nor a variable; so are the two settings of the zero-gradient row skip
(`cake_backend.set_zero_dlogp_min_rows`,
`cake_backend.set_zero_dlogp_temperature_dz_counted`, see "Zero-gradient rows
of the log-probability backward"), each read once per call and part of the
backward binding key, and the two settings of the loss entry's zero-weight
split (`cake_backend.set_zero_weight_min_rows`,
`cake_backend.set_zero_weight_loss_mode`, see "Zero-weight policy rows of the
loss entry"), each read once per call (no binding depends on them).

| Keyword (backend default) | Variable read by `flashinfer.cake_lm_head_det_train` | Parsing |
| --- | --- | --- |
| `compact_rows` (`True`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_COMPACT_ROWS` | `0` -> `False`, else `True` |
| `fuse_dw_cast` (`True`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_FUSE_DW_CAST` | `0` -> `False`, else `True` |
| `dx_finalize` (`True`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_DX_FINALIZE` | `0` -> `False`, else `True` |
| `dw_stream` (`"auto"`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM` | `0` / `false` / `off` / `no` -> `False`; `1` / `true` / `on` / `yes` -> `True`; else `"auto"` |
| `hidden_count` (`True`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_HIDDEN_COUNT` | `0` -> `False`, else `True` |
| `gemm_tuning` (`{}`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING` | JSON object -> dict; unset / blank -> no knob; invalid -> `ValueError` |
| `BINDING_CACHE.configure(enabled=...)` (`True`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE` | `0` -> disabled (read at import, applied on first use) |
| `BINDING_CACHE.configure(capacity=...)` (`64`) | `FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE_CAPACITY` | positive integer (read at import, applied on first use) |
| `cake_backend.set_zero_dlogp_min_rows(rows)` (per architecture: `1946` on sm_100a, `3891` on sm_103a, `4097` elsewhere) | none (process setting) | a log-probability backward of fewer valid rows runs without the zero-gradient row skip; an integer or `None` (the skip off) applies to every architecture; `ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT` restores the per-architecture default |
| `cake_backend.set_zero_dlogp_temperature_dz_counted(counted)` (`True`) | none (process setting) | `True`: the skip's dz arm with a temperature runs `gemm_logits_dz_dcnt_tau`; `False`: `gemm_logits_dz_tau` over every row (the same outputs) |
| `cake_backend.set_zero_weight_min_rows(rows)` (`4096`) | none (process setting) | a non-negative integer; a policy loss call of fewer rows `T` keeps its zero-weight rows in the chunk loop (`0`: every compacted policy call splits) |
| `cake_backend.set_zero_weight_loss_mode(mode)` (`"exact"`) | none (process setting) | `"exact"`: the loss bits of the call without the split; `"regroup"`: the split's own chunk sums |

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
  Their temperature forms (`gemm_logits_lbl_tau`, `gemm_logits_dz_tau`, and
  `gemm_logits_mcnt_tau` for chunk 0 of the hidden valid-row count) read the
  compact per-row FP32 temperature and divide each BF16-rounded logit once
  before the statistics and the `exp`; the `dz` form also divides its row's
  `dlogp_t` once.  `row_finalize_tau` divides the label logit the same way.
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
  (`cake_backend.recommended_k_slices`) and is then fitted to the `V / 64`
  K steps (`cake_backend.dx_slices_fit`: the largest count whose
  `ceil(k_iters / k_slices)`-step slices all hold a K step, e.g. 2 in place of
  3 at `V` = 256), so it is deterministic in the shapes.
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
body (`cake_backend.stage_variant`, `stage_form` with the temperature forms'
`_tau` last / `parse_stage`; suffixes in this order).
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
  chunks of more than 4096 rows on sm_100a and sm_103a (sm_107a keeps the
  table depth)
  (`cake_backend.dx_stages_variant`, `DX_LONG_CHUNK_STAGES`); the 256-wide
  fallback keeps its depth.  Bitwise: the ring depth only changes when a stage
  is refilled;
* `_split`: the 512-wide `dX` tile whose launch's last wave would leave SM
  pairs idle: when the `W` wide work items (row pairs x `H / 512` x
  `k_slices`) leave `r = W mod P` items in the last wave of the `P` SM pairs
  with `0 < r <= P / 2`, the last `r` wide items run as `2 r` 256-wide half
  items in the same launch (the grid carries `W + r` clusters; the split count
  is the launch scalar `dx_split`).  Taken on every architecture for chunks
  that are not a whole number of 512-token tiles; for chunks of whole
  512-token tiles below `H` 7168 on sm_103a in every call
  (`DX_SPLIT_WHOLE_ARCHES`) and on sm_100a and sm_107a in calls of fewer
  than three chunks, whose dW accumulate shares the dX stream
  (`DX_SPLIT_WHOLE_FEW_CHUNKS_ARCHES`, `DW_STREAM_MIN_CHUNKS`); on the
  long-chunk ring (`_st3_split`) on sm_103a only
  (`cake_backend.dx_split_rule`, `cake_backend.dx_split_eligible`,
  `DX_SPLIT_ARCHES`, `DX_SPLIT_LONG_RING_ARCHES`).  Bitwise: the same K
  slices and per-element K order on both widths;
* `_swab`: the swapped 512-wide `dX` tile (`dX^T = W^T dz^T`: a CTA's 128
  hidden columns of `W` on the MMA's M, the chunk's 512-token tile on its N,
  the output stored per thread), at the table ring depth, on sm_103a and
  sm_107a
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
  fused cast below `H` 7168 for chunks of 2049 .. 4096 rows on every
  architecture, with `group_n` = 12 column tiles per block
  (`cake_backend.dw_block_variant`, `DW_BLOCK_GROUPS`); the bf16 cast keeps the
  1-D raster.  Bitwise: a permutation of the same tiles.
* `_tma`: the TMA-epilogue form of the bf16 fused weight-gradient cast
  (`gemm_dw_cast_bf16_tma`): the deferred cast of the last chunk of a call of
  at least two chunks whose last chunk has at most 1792 rows
  (`DW_CAST_TMA_MAX_ROWS`, every architecture; `cake_backend.dw_cast_tma_rule`)
  reads the old FP32 sum and stores the bf16 tile as TMA sub-tiles.  A
  one-chunk call (no old sum) and a longer last chunk keep the shipped cast;
  an explicit `"epi_store"` knob of `gemm_tuning`'s `"dw"` wins over the rule.
  Bitwise: the same per-element `(old + acc) * g` and round-to-nearest.
* `_dcnt`: the forms of the zero-gradient row skip (see "Zero-gradient rows
  of the log-probability backward"): the dz arm (`gemm_logits_dz_dcnt`, its
  temperature form `gemm_logits_dz_dcnt_tau`), the weight-gradient accumulate
  (`gemm_dw_acc_dcnt`, `gemm_dw_acc_gn_dcnt`) and the bf16 fused cast
  (`gemm_dw_cast_bf16_dcnt`, `gemm_dw_cast_bf16_tma_dcnt`) bounded by the
  chunk's device count of live rows (`CNT`); the accumulate's TMA epilogue
  and the fp32 cast have none (`cake_backend.DCNT_BASES`).  Bitwise: the same
  arithmetic on the rows before the count.

The raster height is a scalar of the base forms: `group_m` 16 for the logits
GEMMs (32 on sm_107a, its record's height) and 32 for the weight-gradient GEMMs
(accumulate and fused cast) at `H >= 7168` for chunks of more than 2048 rows
(`cake_backend.raster_variant`, `RASTER_WIDE_GROUPS` per architecture); below
that `H`, 16 for the logits GEMMs of chunks of at least 3841 rows (31 row tiles
of 128: `LOGITS_LONG_RASTER_GROUPS` / `LOGITS_LONG_RASTER_MIN_ROWS`, sm_100a
and sm_103a) and 16 for the weight-gradient GEMMs of chunks of more than 4096
rows on every architecture (`DW_LONG_CHUNK_GROUP_M` / `DW_LONG_CHUNK_MIN_ROWS`); every
other chunk -- the default geometry's chunks of up to 30 row tiles, short tail
chunks -- takes the GEMM's default height from the record geometry
(`logits_group_m` / `dx_group_m` / `dw_group_m`).  Bitwise: only the tile
order changes.

`make_plan` resolves the forms and scalars per chunk (`Plan.variants`,
`Plan.logits_stage` / `dx_stage_of` / `dw_acc_stage` / `dw_cast_stage`;
`stage_values` forms the scalars); a record carries exactly the forms its
architecture's rules can reach (`cake_backend.reachable_variants`, checked by
`generated_program_available`), and a form the record lacks fails closed.
`gemm_tuning` (a keyword of `prepare_lm_head_det_loss` and of every eager /
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
(SM103 and SM107 have it), and on a chunk whose dX tile is the 256-column one
(pin `"tile_n": 512` as well).  A `"k_slices"` pin that leaves a slice without
a K step raises (e.g. 3 slices over `V` = 256's four K steps).

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
  and the architecture-free `library` stem `cake_lm_head_det_loss_<unit>`), stage
  names (`STAGES`, `BASE_STAGES`, `GEMM_STAGES`; the temperature forms in
  `TEMPERATURE_GEMM_STAGES` / `TEMPERATURE_ROW_STAGES` and every name in
  launch order in `STAGES_WITH_TEMPERATURE`), record selection
  (`select_module`) and the loader glue: `library_name(name, stage)` is the
  `build.json` entry `<library>_<arch>` that
  `load_cake_lm_head_det_loss_module(name, stage)` loads through `cake_build`.
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
* `cake_support.py` -- the candidate-support entry points: their host checks,
  the shared temperature resolver's use, the plan of the pass forms, the
  launches of the support kernels and of the shipped GEMMs (dense form), the
  memory report, the reference engine and the autograd `Function`.
* `cake_backend.py` -- validation, chunk plans, workspace layout, memory
  accounting, argument-plan binding, the prepared runner, the torch reference
  path, the autograd `Function`s and the eager entry points.
* `csrc/cake_lm_head_det_loss/` -- generated translation units, two per program
  `cake_lm_head_det_loss_<unit>`: the kernel unit
  (`cake_lm_head_det_loss_<unit>_kernel.cu`, architecture-neutral, the
  per-architecture lowering under `__CUDA_ARCH__` guards) and the tvm-ffi
  binding (`cake_lm_head_det_loss_<unit>_binding.cu`: argument checks, the
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

The registry holds one record per architecture (`sm_100a`, `sm_103a`,
`sm_107a`), `cake_lm_head_det_loss_<arch>`, geometry-free (`H` / `V` and the rules'
outputs are launch scalars: every `H` a multiple of 512 and `V` a multiple of
256 runs on it), with the sixteen base stages above plus the sixteen structural
forms the rules can reach on `sm_100a` and `sm_103a` (`abi = lm_head_loss_v1`),
the four temperature forms of the log-probability entry and the fourteen
candidate-support programs (`cake_support.SUPPORT_STAGES`: the validation stage
per index width, the two row stages, the gather and scatter of the dense form,
the support logits gather GEMM and the segment gather-accumulate dX / dW /
combine kernels in their BF16 and FP32-accumulate modes), exported from the
kernel snapshot named in the pull request.  The `sm_107a` record registers
every stage of the `sm_103a` record; its rule tables take the `sm_103a`
entries except the logits rasters (`LOGITS_LONG_RASTER_GROUPS`; the logits slot
of `RASTER_WIDE_GROUPS` keeps the record's 32), the long-chunk `dX` ring
(`DX_LONG_CHUNK_STAGES`) and the whole-chunk `dX` split, which follows
`sm_100a` (`DX_SPLIT_WHOLE_FEW_CHUNKS_ARCHES`), so its rules reach twelve of
the sixteen forms (all but the four `_st3` forms).
`tests/experimental/test_cake_lm_head_det_train_support.py` tests the candidate-support
entries (host rules and the reference backend on any device, the cake backend
on a registered device).

Tests: `tests/experimental/test_cake_lm_head_det_train.py` and the public-API suite
`tests/experimental/test_cake_lm_head_det_train_api.py` (the host-layer tests
run on any device through the reference path; the device tests skip without a
registered program or a compute capability 10.0 / 10.3 / 10.7 device), and the
CPU-only `tests/experimental/test_cake_lm_head_det_train_standalone.py` (a copy of
the package imports with `flashinfer` blocked and raises its host-side errors;
`scripts/check_cake_build_manifests.py --vendored-gate` builds and loads the
copy's libraries on a GPU node).  Benchmark:
`benchmarks/bench_cake_lm_head_det_train.py`.

Toolchain note: the package needs CUDA 12.9 or newer.  Every `build.json`
entry has `min_cuda` 12.9 (`common.requires.cuda_toolkit.min` is 12.9), and
the loader refuses an older toolkit: `cake_build.jit_spec` raises and the JIT
spec lists leave the entry out.  CUDA 12.8's ptxas cannot build the 256-bit
global loads and stores (`ld.global.v8.b32` / `st.global.v8.b32`) of eight
`sm_100a` programs, the logits GEMM among them ("Illegal vector size: 8");
`sm_103a` needs 12.9 and `sm_107a` 13.4 by their architecture floors (every
`sm_107a` entry has `min_cuda` 13.4).

The `sm_100a` 2-CTA GEMM producer loops (logits, dX and dW) have two orders,
and the toolkit that compiles them selects one.  Built by nvcc 12.9 or 13.0,
they issue each stage's `mbarrier.arrive.expect_tx` on the stage's full
barrier before the stage's TMA loads, the order CUTLASS's producer acquire
uses.  Built by any other toolkit, they keep the earlier order (the arrive
after the loads), so other toolkits keep the previously shipped machine code.
Each such kernel source defines `CAKE_ARRIVE_FIRST` once from `__CUDA_ARCH__`
and `__CUDACC_VER_MAJOR__` / `__CUDACC_VER_MINOR__`.  The arrive counts and
transaction bytes of every phase are the same in both orders, so the outputs
are bitwise the same.

With the earlier order, the `sm_100a` programs built by CUDA 12.9.86 and
13.0.88 faulted intermittently on GB200 with `cudaErrorIllegalInstruction`
at a 2-CTA TMA load of the dW and dX GEMMs (in the dW faults examined, the
second B-operand load, with legal operands): 35 and 64 runs of a 132-run
test set, and 20 and 155 runs of a 624-run soak.  The builds of CUDA
13.1.115, 13.2.86, 13.3.73 and 13.4.92 had no fault in that 132-run set,
and no B200 run faulted; the cause is not established.  With the arrive
first, the `sm_100a` builds of CUDA 12.9.86 and 13.0.88 each ran 850 test
runs and a 1872-run soak on GB200 without a fault.  B200 and GB200 run the
same `sm_100a` code, so B200 builds of CUDA 12.9 and 13.0 also compile the
arrive-first order.  On B200 with CUDA 13.0.88, over 37 shapes in two
timing states, the geometric-mean time ratio of the two orders is 0.99987
(CI95 0.99955 to 1.00022) and 1.00022 (0.99970 to 1.00076); no
shape was slower at the family-wise level, and two shape-states were
nominally slower by 0.45 % and 0.56 % in kernel time.  CUDA 12.9 was
not timed.  The `sm_103a` programs keep the earlier order at every toolkit,
so their machine code is unchanged; on GB300 their builds of CUDA 12.9,
13.0, 13.3 and 13.4 ran without a fault.

On CUDA 13.0 (nvcc 13.0.x) the `sm_103a` programs are built
with `-Xptxas -O0` (build data: every `sm_103a` entry of `build.json` carries
`{"cuda": "==13.0", "flags": ["-Xptxas", "-O0"]}` under `nvcc_flags_if`; the
JIT path applies it when the local toolkit matches, a prebuilt `aot/` built by
that toolkit already carries it).  In earlier testing that toolkit's code of
their 2-CTA TMA producer loops faulted: at -O3 (and -O2, which emits the same
code), with a short K loop (`chunk_size <= 1024`) the dW
accumulate GEMM's second B-operand TMA load is rejected by the TMA unit with
`cudaErrorIllegalInstruction` although every operand is legal; at -O1 the
structural-form programs' dW cast GEMM (the last chunk's `dz_c^T @ X_c`) faults
the same way on every call whose last chunk is partial (`T % chunk_size != 0`).
The -O0 code and the code of CUDA 12.9, 13.3 and 13.4 ran without the fault.
With the earlier -O1 code the hidden
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
observed CUDA 13.0 fault, not a root-cause fix.
