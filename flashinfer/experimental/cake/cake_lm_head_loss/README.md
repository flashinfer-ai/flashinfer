# Cake chunked LM-head + loss training backend, non-deterministic form (SM100 / SM103)

Experimental backend for the chunked large-vocabulary LM-head projection with a
fused log-sum-exp / cross-entropy / policy-gradient loss and a memory-bounded
backward, in the form whose dX reduction is unordered.  It is the
`deterministic=False` implementation of the entries of
`flashinfer/chunked_lm_head.py`; the deterministic implementation is
`flashinfer.experimental.cake_lm_head_det_loss` (its README describes the
shared contract, the kernels and the toolchain rules in full).  Tracking issue
of the family: https://github.com/flashinfer-ai/flashinfer/issues/5680.

Both the API and the backend are experimental: no compatibility guarantee,
SM100 (B200 / GB200) is the acceptance architecture and SM103 (B300 / GB300) is
compiled from the same sources.  The feature does not participate in automatic
backend selection, autotuning or trace-apply.  Its libraries are loaded
prebuilt from `aot/` when the package ships them and JIT-built by FlashInfer
from `build.json` otherwise (see "Layout of this package").

## Deterministic and non-deterministic routes

| | deterministic | non-deterministic (this package) |
| --- | --- | --- |
| package | `flashinfer.experimental.cake_lm_head_det_loss` | `flashinfer.experimental.cake.cake_lm_head_loss` |
| public call | `chunked_lm_head_det_*(...)`, or `chunked_lm_head_*(...)` with the default `deterministic=True` | `chunked_lm_head_*(..., deterministic=False)` |
| `loss`, `logp`, `lse` | fixed order, bitwise run to run | fixed order, bitwise run to run (the same programs) |
| zero-weight policy rows (`loss_weights == 0`) | in the chunk loop | left out of the gradient path (see "Zero-weight policy rows") |
| log-probability rows with `dlogp == 0` | in the chunk loop | ordered last; the backward's GEMMs skip their work (see "Zero-gradient rows of the log-probability backward") |
| `dW` | chunk order, bitwise run to run | chunk order (the same programs); no run-to-run promise |
| `dX` of a K-sliced chunk | slice 0 stored, slices `>= 1` in FP32 slabs added in slab order | one launch per chunk adds every slice into the `-0.0`-filled FP32 accumulator rows in an unordered FP32 reduction; a one-slice chunk stores its rows as the deterministic package does |
| precision boundaries | BF16 logits and `dlogits`, FP32 loss arithmetic, FP32 `dX` / `dW` accumulation, one final cast each | the same |
| under `torch.use_deterministic_algorithms(True)` | runs | the public entries run the deterministic package (one warning per entry point); this package's gradient entries raise PyTorch's alert |

## Public entry points (`flashinfer/chunked_lm_head.py`)

```python
from flashinfer.chunked_lm_head import chunked_lm_head_loss, chunked_lm_head_logprob

loss = chunked_lm_head_loss(X, W, labels, objective="ce", loss_div=loss_div, deterministic=False)
loss, logp = chunked_lm_head_loss(X, W, labels, objective="policy", infer_logp=infer_logp,
                                  loss_weights=loss_weights, return_logp=True, deterministic=False)
logp = chunked_lm_head_logprob(X, W, labels, chunk_size=4096, deterministic=False)
```

The det-free entries `chunked_lm_head_loss`, `chunked_lm_head_logprob`,
`chunked_lm_head_preload`, `chunked_lm_head_support_logprob` and
`chunked_lm_head_support_kl` take a keyword-only `deterministic: bool = True`:

* `True` (the default): the deterministic entry (`chunked_lm_head_det_*`) with
  the same arguments -- the same bits, no added allocation or synchronization;
  the entry adds one identity test of host work and its own frame.
* `False`: this package, after the same argument checks and with the same
  `FLASHINFER_CAKE_LM_HEAD_LOSS_*` fallbacks of the switches.  While torch's
  deterministic algorithms are enabled (also with `warn_only=True`), the
  deterministic entry runs instead and a `UserWarning` names the entry once
  per process.
* Anything else: `TypeError`.

The arguments, outputs, memory rule, switches, ignored-row semantics,
compaction, the policy objective and its clip, the opt-in forms
(`memory_mode`, `num_valid_rows`, `check_num_valid_rows`) and the returned
dtypes are those of the deterministic entries (`chunked_lm_head_det_loss` /
`chunked_lm_head_det_logprob`).  Statements there that two forms of a call
give "bitwise the same" output compare the same operations in one run; in
this package the K-sliced dX reduction may round differently from one run to
the next.

The package's own entries (`cake_backend.forward_loss`, `backward_loss`,
`forward_logprob`, `backward_logprob`, `forward_logprob_keep`,
`backward_logprob_keep`, `prepare_lm_head_loss` and the runner,
`chunked_lm_head_loss` / `chunked_lm_head_logprob` /
`chunked_lm_head_logprob_keep`, the same in `cake_backend_opt_in`, and the
candidate-support entries) take
`deterministic` with the default `True` and raise `ValueError` for any value
but `False`: a caller asks for this package's mode explicitly.  With
`torch.use_deterministic_algorithms(True)` every entry that launches the
gradient GEMMs raises PyTorch's alert in PyTorch's wording (`RuntimeError`, or
a `UserWarning` under `warn_only=True`) before it binds or launches anything:
`forward_loss` with `need_dx` or `need_dw` (the forward runs the dX and dW
GEMMs), the deferred last-chunk weight-gradient GEMM of `backward_loss` (and
the budget form's recompute), `backward_logprob`, `backward_logprob_keep`, a
runner's `forward()` /
`backward()` that launch them, and the candidate-support backward.  The
log-probability forward, a forward without trainable inputs and (gradient
placement `"auto"`) a loss-entry call under `torch.no_grad` launch no gradient
GEMM and do not alert.  The autograd contexts record the mode
(`ctx.deterministic`).

## Kept log-probability logits (`memory_mode="keep_logits"`)

The log-probability entry's opt-in `memory_mode="keep_logits"` is the
deterministic package's kept logits (`chunked_lm_head_det_logprob_keep`) under
this package's names: `cake_backend.chunked_lm_head_logprob_keep`, the
explicit pair `forward_logprob_keep` / `backward_logprob_keep`
(`KeptForwardResult.logits`, one BF16 `[rows_c, V]` view per chunk),
`kept_memory_report` (`saved_logits_bytes`, `2 * T_v * V`) and
`prepare_lm_head_loss(..., logprob_keep=True)` (its runner's `forward` /
`backward` raise `NotImplementedError`); the kept form has its own plan
(`_KeptPlan`), binding and cache keys, and the default entries are unchanged.

* Where a backward can follow (grad mode on, `X` or `W` trainable, `cake`
  backend) the forward runs `gemm_logits` (the whole `z_c`; the label-column
  arm is off) and keeps every chunk's BF16 logits -- chunk 0 in the chunk
  buffer, the later chunks in one `[T_v - rows_0, V]` buffer -- as saved
  tensors.  The first backward runs `row_grad` over them in place, then the dX
  and weight-gradient GEMMs: three GEMMs per chunk instead of four.  A second
  backward through a retained graph recomputes; `backward_logprob_keep` marks
  the views it consumed (a second call on them is a `ValueError`).
* Elsewhere -- `torch.no_grad`, `torch.inference_mode`, the first pass of a
  reentrant `torch.utils.checkpoint`, frozen inputs, the reference backend --
  the call is `chunked_lm_head_logprob`.  A non-reentrant checkpoint drops the
  kept logits after the checkpointed forward and its recomputation keeps them
  again.
* The kept `2 * T_v * V` bytes live from the forward to the backward, outside
  the memory rule's chunk bound.  `memory_mode="keep_logits"` takes no
  `num_valid_rows` and is a `ValueError` on the loss entry and, at the opt-in
  level, on the reference backend.
* `logp`, `lse` and `dW` are the default path's bits.  `dX` is formed from the
  same `dz` and the same products; a chunk of more than one dX slice adds them
  in the unordered reduction below, as the default path does.
* The dispatcher `chunked_lm_head_logprob` passes `memory_mode="keep_logits"`
  on unchanged: to the deterministic entry (`deterministic=True`, or the
  deterministic-algorithms fallback) or to this package (`deterministic=False`).

## Candidate-support entry points (`cake_support.py`)

`chunked_lm_head_support_logprob` / `chunked_lm_head_support_kl` have the
deterministic entries' contract (`chunked_lm_head_det_support_logprob` /
`_kl`) plus the `deterministic` keyword (the deterministic entries have none).
As part of this package they make no run-to-run promise for the gradients, and
their backward raises the alert above.  Their programs are the deterministic
package's (every sum has one owner and one order) with two changes on the
support path:

* Dense dX (the rows whose mean support size reaches the dense threshold):
  each vocabulary block's dX GEMM adds its FP32 product into the token chunk's
  FP32 sum from its epilogue (`gemm_dx_red`, the loss path's reduction form)
  instead of writing an FP32 product block that a separate pass adds; the
  `[chunk, H]` FP32 product block (100 MB at 4096 x 6144) and the add pass are
  gone.  At one K slice per block the chunk's first block stores and the later
  blocks add in block order: dX has the deterministic entries' bits.
  `cake_support.support_dx_form` runs more slices only inside a cell of
  `SUPPORT_DX_SLICE_CELLS` (B200 shape classes, in the table below).
  There the chunk sum is filled with `-0.0` first and the slices add in the
  order they finish (round to nearest, subnormals kept): an unordered FP32 sum
  within the accuracy budgets.  Every other shape, SM count and architecture
  runs one slice.  The split tail (the last wave's work items run as half
  items) runs on the architectures with split forms (`sm_100a`, `sm_103a`)
  and keeps the bits.
* Sparse dW zero fill (global mode): the BF16 `dW [V, H]` is zeroed as FP32
  words (a 4-byte fill): the same bits.

Cells of the dense dX rule (B200, 148 SMs, `sm_100a`; `S`: K slices per
vocabulary block, `F`: split-tail items):

| H | token chunk rows | vocabulary block K steps | S (F) |
|---|---|---|---|
| 2048 | 2817 to 3072 | 224 to 244 | 3 (0) |
| 3072 | 1793 to 2048 | 224 to 244 | 3 (0) |
| 4096 | 257 to 512 | 224 to 244 | 2 (32) |
| 5120 | 3841 to 4096 | 216 to 240 | 2 (24) |
| 6144 | 769 to 1024 | 224 to 244 | 3 (0) |

By architecture: `sm_100a` (B200) is tested on the device.  On
`sm_103a` (GB300) the rule has no cell, so the dense dX runs one slice
everywhere (the deterministic bits); the support tests, forced two to four
slices included, pass on GB300.  `sm_107a` is pending: this package has no
`sm_107a` build.

Accuracy on B200, worst of 20 runs against the FP64 oracle: the production
row at H 5120 (T 16231, V 152064, chunks of 4096 and 3943 rows, two slices on
every block) has dX relative L2 2.517e-3 in
all 20 runs, as the deterministic entry (2.517e-3) and the masked PyTorch
baseline (2.517e-3); 1.6e-4 of its elements differ from the deterministic dX
(largest difference 1.9e-6), 94 % of those by one BF16 ulp and 2.7 % by two,
and the 20 runs gave one distinct result.  The three-slice cell at H 2048
(3072 rows, V 29952 in blocks of 244 and 224 K steps, the rule's own choice)
has worst dX relative L2 2.477e-3 over 20 runs, as the deterministic entry
and the baseline; at most 1.8e-5 of its elements differ from the
deterministic dX (largest difference 3.8e-6), with 11 to 17 distinct results
in four test runs.  Forced two to four slices at H 1536 (three column items,
vocabulary blocks of 8, 8 and 4 K steps, T 1300 and 4097) match the
deterministic entry's error to four digits (2.552e-3 and 2.366e-3) with 2 to
20 distinct results in 20 runs.  Every element is within one BF16 ulp of the
deterministic dX plus an FP32 floor of 2^-14 of its row's largest element:
the elements more than one ulp apart (in five runs each, at most 24 ulp at
H 1536, 16 ulp at H 2048 and 128 ulp at the H 5120 cell of the rule, none of
them changing sign; 456 elements of the H 5120 production row in every run,
three of them changing sign) are all below 3.2e-5 of their row's largest
element, where the floor sets the allowance.  Rows with no nonzero product
(an ignored label, a single candidate) are exactly 0, as in the deterministic
dX.

The other support stages keep the deterministic arithmetic: the sparse dW
keeps its ordered part sum (the long segments' parts are summed in part
order); the sparse dX has one owner per element; the dense dW accumulates in
chunk order (as the loss path's `gemm_dw_acc`); the row stages are per-row
warps.

## Switches and environment variables

The switches and the `FLASHINFER_CAKE_LM_HEAD_LOSS_*` variables are those of
the deterministic package (its README's table): every switch is a keyword of
the entries it affects, `None` selecting the documented default, and the
package reads no environment variable.  `flashinfer.chunked_lm_head` resolves
the variables for both modes and configures this package's two binding caches
(`cake_backend.BINDING_CACHE`, `cake_backend_opt_in.BINDING_CACHE`) with the
same binding-cache settings on the first `deterministic=False` call.  This
package adds no variable.

Process-wide settings of this package, set in code only (each setter returns
the previous value; an entry call reads them once):

| setter | default | effect |
| --- | --- | --- |
| `cake_backend.set_dx_slice_policy` | `"det"` | the dX slice rule (see "Kernel structure of one token chunk"); part of every binding key |
| `cake_backend.set_dx_red_min_rows` | `0` | a chunk of fewer rows keeps one K slice (the plain store); part of every binding key |
| `cake_backend.set_zero_weight_min_rows` | `4096` | a call of fewer rows `T` keeps its zero-weight rows in the chunk loop (see "Zero-weight policy rows") |
| `cake_backend.set_zero_dlogp_min_rows` | `0` | a log-probability backward of fewer valid rows runs without the zero-gradient row skip; `None` turns the skip off (see "Zero-gradient rows of the log-probability backward"); part of the backward binding key |
| `cake_backend.set_gradient_placement` | `"auto"` | where the loss entries (`chunked_lm_head_loss` of both modules, every form) run the dX / dW GEMMs. `"auto"`: a call no backward can follow (grad mode off, or neither `X` nor `W` trainable) computes the loss (and `logp`) only; inside a non-reentrant `torch.utils.checkpoint` (its own saved-tensor hooks innermost) the checkpointed forward computes the loss only and saves its inputs, and the recomputation in the backward runs the fused forward and hands the gradient accumulators to the checkpointed call's backward, so a checkpointed step forms them once and keeps no FP32 accumulator from the forward to the backward (paired per checkpoint frame and graph by call order and checked against the inputs' metadata and the call's options; without a handed result the backward runs the fused forward itself, with a `RuntimeWarning`); other hooks innermost (CPU offload), torch's deterministic algorithms, or a torch whose checkpoint lacks those hooks keep the fused forward. Under selective activation checkpointing (`context_fn` of `create_selective_checkpoint_contexts`, whose recomputation replays the checkpointed forward's operators by position) every entry of this package -- the loss, log-probability (default, budget, kept logits) and candidate-support entries -- runs its forward with all active torch dispatch modes popped, the checkpoint's and any user mode on the stack (a `FlopCounterMode` does not see the entry's operators; a tracing mode such as `FakeTensorMode` or a `torch.compile` trace gets real launches on fake inputs and fails loudly), and behaves as in a plain non-reentrant checkpoint; the per-device constant cells and the calling thread's count cells and `arange` constant are formed outside active dispatch modes. `"forward"`: every call with a trainable input forms the gradients in its forward (the previous behavior). The same kernels on the same operands either way: the same loss, `logp` and dW bits, and dX up to the order of its unordered K-slice reduction (the same bits with one dX slice) |
| `cake_backend.set_dx_fill_side_stream` | `False` | `True`: a remembered binding issues the `-0.0` fill of `dX_acc` on the calling thread's fill stream (its own allocator pool, joined before the first dX GEMM), so it runs beside the queued chunk-0 logits GEMM; not under CUDA graph capture, not in the prepared runner; `True` keeps up to `T * H * 4` bytes more reserved in the fill stream's pool |

The side streams, events, pinned count cells and the `arange` constant of the
hidden valid-row count are per thread (`_THREAD_STATE`): a thread's calls
never share them in flight.  A caller that runs each call on a new thread
(such as `torch.nn.DataParallel`) forms that state again on every call (a
stream, events, a pinned cell and one small allocation and kernel);
`chunked_lm_head_preload(deterministic=False)` forms it for the calling thread
only.

## Zero-weight policy rows

A policy row with a valid label whose `loss_weights` entry is exactly zero
(`+0` or `-0`) contributes exactly zero loss and zero gradient.  Its weight is
known before the forward, like a `-100` label, so this package leaves such a
row out of the gradient path:

* The chunk loop runs over the gradient rows `G = (labels >= 0) &
  (loss_weights != 0)` only: the row compaction machinery uses this mask (the
  host count path and the hidden valid-row count), and both row counts come
  back in the call's one count readback (no further host synchronization).  A
  NaN weight keeps its row in `G`.
* The zero-weight rows `Z = (labels >= 0) & (loss_weights == 0)` run no
  `row_grad`, dX GEMM or weight-gradient GEMM, and add no loss term.  Their
  `dX` rows are exact zeros.  With `return_logp=True` their `logp` comes from
  a logp-only pass of the log-probability forward kernels (`gemm_logits_lbl`
  + `row_finalize`, chunk-bounded buffers) and goes into the same `[T]`
  output.  Without `return_logp` that pass does not run.
* A call whose valid rows all have zero weight launches no gradient GEMM: the
  loss is 0 and both gradients are zeros.
* The work left out depends on the data: it grows with the share of
  zero-weight rows and is nil without them (the split itself adds a few small
  kernels to the count).  The logits GEMM of a zero-weight row runs once
  either way: with `return_logp=True` in the logp-only pass, so the call
  leaves out those rows' `row_grad`, dX GEMM, weight-gradient GEMMs and loss
  terms; without it the call also skips their logits GEMM and `row_finalize`.
* The logp-only pass runs ahead of the chunk loop, except when the hidden
  valid-row count already holds chunk 0's buffers and either the chunk loop
  keeps no buffer until the backward (`W` without a gradient, or
  `fuse_dw_cast=False`) or no chunk loop follows (every valid row has zero
  weight; the order is decided once the count is known): it then runs after
  the loop, or in its place, once those buffers are released, so the two
  passes' chunk buffers do not coexist.  The memory report's
  `zero_weight_logp_phase_bytes` is the pass's phase (with chunk 0's buffers
  when they coexist) and `peak_temporary_bytes` the larger of the loop's and
  the pass's phase.  Neither counts the FP32 `dX_acc` / `dW_acc` accumulators
  (`accumulator_bytes`): in the ahead order the pass ends before they exist,
  so adding the two overstates the call's peak.
* A call of fewer than `cake_backend.zero_weight_min_rows()` rows `T`
  (default 4096, `set_zero_weight_min_rows`) keeps its zero-weight rows in the
  chunk loop, as the deterministic package does (the split adds row kernels
  and a second count to every call that takes it).

| | deterministic package | this package |
| --- | --- | --- |
| zero-weight row, finite inputs | full chunk path; its terms are exact zeros | skipped for the gradients; the same mathematics |
| `loss` | FP64 sum over the valid rows | FP64 sum over `G` (the `Z` terms are 0; the grouping differs) |
| `dW` | sum over every valid row of a chunk | sum over the `G` rows (other chunk boundaries; bits may differ within the accuracy budgets) |
| `logp` of a zero-weight row | loss-path logits GEMM + `row_finalize` | logp-only pass (`return_logp=True`); 0 in the internal output otherwise |
| non-finite `X` row, logits or `infer_logp` in a zero-weight row | NaN in `loss` / `dX` / `dW` (`0 * inf`) | does not reach `loss` / `dX` / `dW` (the row is treated like an ignored row for the gradients) |
| `dX` row of a zero-weight row under a non-finite upstream loss gradient | NaN (`bf16(g * 0)`) | exact 0 (as an ignored row's) |

Not applied (the plain path runs, every valid row in the chunk loop): a call
below the row floor above, `compact_rows=False`, `objective="ce"` (no weights), the log-probability entry
(a zero `dlogp` row is known only in the backward; see the next section), the prepared runner
(`prepare_lm_head_loss`, a fixed row set) and a caller-given
`num_valid_rows` (its plan needs no readback, so the split count is not
available).  `memory_mode="budget"` splits the rows too: its chunk loop and
step-peak model run over the gradient rows, its bound still counts every valid
row, and the model adds the logp-only pass as one more phase
(`memory_plan(..., zero_rows=...)`).

## Zero-gradient rows of the log-probability backward

A valid row whose upstream gradient `dlogp` is exactly zero (`+0` or `-0`;
for example a token the caller's loss masks) contributes exact zeros to `dX`
and `dW`.  The log-probability backward knows `dlogp` only when it runs, so
it handles these rows on the device, without a host synchronization:

* The backward orders the forward's valid rows so that the live rows (a
  nonzero `dlogp`, NaN included) come first, each group in ascending order,
  and forms the per-chunk count of live rows in device memory (a few small
  kernels per call; `zero_dlogp_partition`).
* The chunk's dz arm of the logits GEMM, its K-sliced dX reduction, its
  weight-gradient accumulate and its bf16 fused cast run their `_dcnt` forms:
  each reads the chunk's count and skips the work items (or the K steps) past
  it.  The launch shape stays the chunk's full row count, so the call is the
  same set of launches for every `dlogp` and a CUDA graph capture of the
  backward replays for other `dlogp` values.  A chunk with no live row runs
  its launches without their GEMM work.
* The fused dX finalize writes exact `+0` rows for the zero-gradient rows (as
  for an ignored row); `dX` of a live row is the same mathematics as without
  the skip.  `dW` sums the same nonzero products over the live rows in the new
  row order (FP32 accumulation; bits may differ within the accuracy budgets).
* The skip applies to a compacted backward that reuses the forward's row index
  (the autograd entries), with the dz arm (`logprob_dz`) and, when it forms
  `dX`, the fused dX finalize, on a record that registers the `_dcnt` forms,
  and of at least `cake_backend.zero_dlogp_min_rows()` valid rows (default 0,
  `set_zero_dlogp_min_rows`; `None` = never).  A chunk plan with a form that has
  no counted form (the swapped dX tile, the weight-gradient TMA epilogue)
  runs the shipped forms over the reordered rows (the same results).

Non-finite values: a NaN `dlogp` keeps its row live (NaN in its `dX` row and
in `dW`, as without the skip).  A non-finite `X` row of a zero-gradient row
makes `dW` NaN without the skip (`0 * inf`); with the skip it still does when
the row falls within the 63 rows after a chunk's live rows (the weight-gradient
GEMM's last 64-row K step), and not elsewhere.  Its `dX` row is an exact zero
either way.

Not applied: `compact_rows=False`, a backward that recounts its valid rows,
the kept-logits entry, `memory_mode="budget"`, the prepared runner and the
reference backend.  The one-slice dX chunks (the plain store) compute every
row of the chunk.

## Kernel structure of one token chunk

The stages and their order are the deterministic package's: `gemm_logits`
(and its forms), `row_finalize`, `loss_reduce`, `row_grad`, the dX GEMM,
`gemm_dw_acc`, `gemm_dw_cast_bf16` / `gemm_dw_cast_f32`, the scale casts,
`scale_cast_scatter_bf16` and `gather_rows_bf16`, rendered from the same kernel
modules with the same configurations under this package's prefix.  The one
stage that differs is the dX GEMM (`dX_c = dz_c @ W`):

* A chunk the slice rule leaves at one K slice stores its `dX_acc` rows
  directly (the same arithmetic as the deterministic package).
* A chunk split into `S > 1` K slices (along the vocabulary, the GEMM's K) is
  one launch of a `gemm_dx_red*` stage: the call first fills the `dX_acc`
  rows of its sliced chunks with `-0.0` (the exact additive identity) once,
  on its stream ahead of its launches (inside a captured CUDA graph too), and
  every slice's FP32 partial tile, slice 0 included, is added into them with
  an unordered FP32 reduction -- the TMA bulk tensor reduce
  (`cp.reduce.async.bulk.tensor ... .add`) from the TMA-store epilogue of
  every registered `gemm_dx_red*` form (no per-thread-store reduction form is
  registered).  It keeps IEEE round-to-nearest without flushing
  subnormals (the deterministic package's FP32 add); every partial is added
  exactly once (no lost update, no race), only the order of the adds varies.
  No `dx_ws` slab workspace, no `slab_sum` launch and no host `add_` chain
  remain, so the temporary bucket shrinks by `(S - 1) * rows * H * 4` bytes
  (`S` the call's largest slice count, `rows` the chunk buffer's rows).  The
  loss entry's step peak does not change (the same requested bytes, measured
  at `T = 16231`, chunk 4096 on B200): it is reached after the chunk loop,
  where the slabs no longer existed.
* The slice count is a host rule per architecture and chunk shape
  (`cake_backend.dx_slices`), free of the deterministic package's bit-identity
  constraint and selectable in code (`cake_backend.set_dx_slice_policy`; no
  environment variable): `"det"` (the default) takes the deterministic
  package's slice counts, `"measured"` takes the slice count among the
  measured ones (1, 2, 3, 4, 6, 8) with the best wave efficiency of the
  chunk's actual dX form, less a small per-slice penalty (fitted to B200
  measurements).  Either policy fits the count to the K steps
  (`cake_backend.dx_slices_fit`, the largest count whose slices all hold a K
  step): at `V = 256` (four K steps) a count of three runs as two.  A chunk
  of fewer rows than `cake_backend.set_dx_red_min_rows` keeps one slice.

The host plans and binds every dX GEMM launch through one helper section of
`cake_backend.py` (`_dx_fill_keys`, `_dx_keys`, `_dx_values`, `_dx_slots`),
shared by `cake_backend_opt_in.py`.

## Name map

| Name | What it is now |
| --- | --- |
| `flashinfer.chunked_lm_head.chunked_lm_head_{loss,logprob,preload,support_logprob,support_kl}` | dispatchers: `deterministic=True` -> `chunked_lm_head_det_*`, `False` -> this package |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_*` | the deterministic entries, unchanged (`deterministic=False` raises) |
| `flashinfer.experimental.cake_lm_head_loss` | the deterministic package under its former path (each module is the deterministic module) |
| `flashinfer.experimental.cake.cake_lm_head_loss` | this package |
| `cake_backend.{prepare_lm_head_loss, validate_lm_head_inputs, lm_head_loss_workspace_size, LmHeadLossRunner, ChunkedLmHeadLossFunction, ChunkedLmHeadLogprobFunction, chunked_lm_head_loss, chunked_lm_head_logprob}` | this package's objects (the deterministic modules keep these names as aliases of their `det` objects) |
| `cake_backend.{chunked_lm_head_logprob_keep, ChunkedLmHeadLogprobKeepFunction, forward_logprob_keep, backward_logprob_keep, KeptForwardResult, kept_memory_report}` | this package's kept-logits objects (the deterministic package's `chunked_lm_head_det_logprob_keep`, `ChunkedLmHeadDetLogprobKeepFunction` and the same eager names) |
| `cake_support.{SupportLogprobFunction, chunked_lm_head_support_logprob, chunked_lm_head_support_kl}` | this package's objects |
| `cake_jit.load_cake_lm_head_loss_module` | this package's loader |
| libraries `cake_lm_head_loss_<20 hex>_<arch>`, symbols `kernel_cake_lm_head_loss_<20 hex>` | this package's programs (named from their current kernel modules; no name of the libraries the deterministic package shipped under this prefix before its `det` naming is reused) |

## Layout of this package

* `cake_jit.py` -- the `MODULES` registry (one record per architecture,
  expanded from the compact tables the generated-program export writes between
  the `generated registry` markers; a stage's physical record names its program
  and the library stem `cake_lm_head_loss_<unit>`), stage names, record
  selection (`select_module`) and the loader glue (`library_name`,
  `load_cake_lm_head_loss_module`, `specs_for_arch`,
  `toolchain_runs_hidden_count`).
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package (byte-identical copies) and this package's build manifest (schema
  `cake-build/1`, `package` = `flashinfer.experimental.cake.cake_lm_head_loss`).
* `cake_backend.py` -- validation and the mode contract, chunk plans,
  workspace layout, memory accounting, argument-plan binding, the prepared
  runner, the torch reference path, the autograd `Function`s and the eager
  entry points.
* `cake_backend_opt_in.py` -- `memory_mode` / `num_valid_rows` forms and
  `preload`.
* `cake_support.py` -- the candidate-support entry points.
* `csrc/cake_lm_head_loss/` -- generated translation units (kernel and tvm-ffi
  binding per program) and `csrc/cake_ffi_checks.h`; `.clang-format` disables
  formatting (the sources are identity-checked by the manifest's digests).
* `.gitignore` -- `aot/` (prebuilt libraries and their `build_info.json`
  sidecar, built locally, never tracked).

The package imports only itself, `torch` and `tvm_ffi`; its only `flashinfer`
import is the JIT fallback of `cake_build.py`.

## Status

* `sm_100a`, `sm_103a`: the deterministic package's stages rendered under this
  package's prefix, except the K-sliced dX GEMM: its `gemm_dx_red*` forms
  replace the deterministic sliced forms and `slab_sum`.  The `sm_103a`
  reduction forms pass this package's support tests on GB300.
* `sm_107a`: not built yet (`SUPPORTED_COMPUTE_CAPABILITIES` lists `sm_100a`
  and `sm_103a`); it follows the deterministic family's `sm_107a` programs.
* Temperature in the loss / log-probability entries
  (https://github.com/flashinfer-ai/flashinfer/issues/7637): follows the
  deterministic package when it lands there.

Tests: `tests/experimental/test_cake_lm_head_loss.py` (the backend: the mode
contract and the alert, the reference backend, and on a registered device the
deterministic suite's accuracy budgets over repeated runs, the comparison with
the deterministic package, the zero-weight policy rows, changing shapes on reused scratch, checkpoint
replay, gradient accumulation, a retained graph, concurrent streams, a
CUDA graph of the prepared runner and the kept logits),
`test_cake_lm_head_loss_api.py` (the
dispatchers), `test_cake_lm_head_loss_standalone.py` (a copy imports with
`flashinfer` blocked; the prebuilt path) and `test_cake_lm_head_support.py`
(the candidate-support entries).  Benchmark:
`benchmarks/bench_cake_lm_head_loss.py` (deterministic vs non-deterministic).

Toolchain: the deterministic package's rules apply unchanged (CUDA 12.9 or
newer; the `sm_100a` 2-CTA producer loops issue the stage's
`mbarrier.arrive.expect_tx` first when nvcc 12.9 / 13.0 builds them; the
`sm_103a` programs carry `-Xptxas -O0` for CUDA 13.0 in `nvcc_flags_if`, and
`sm_103a` calls built by CUDA 13.0 take the host-count path,
`cake_jit.toolchain_runs_hidden_count`).
