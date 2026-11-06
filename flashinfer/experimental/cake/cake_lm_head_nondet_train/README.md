# `cake_lm_head_nondet_train`: Cake chunked LM-head + loss training backend, non-deterministic (SM100 / SM103 / SM107)

Experimental backend for the chunked large-vocabulary LM-head projection with a
fused log-sum-exp / cross-entropy / policy-gradient loss and a memory-bounded
backward, in the form whose dX reduction is unordered.  It is the backend of
the public module `flashinfer.cake_lm_head_nondet_train`
(`flashinfer/cake_lm_head_nondet_train.py`); the deterministic counterpart is
`flashinfer.cake_lm_head_det_train`, with the backend
`flashinfer.experimental.cake.cake_lm_head_det_train` (its README describes
the shared contract, the kernels and the toolchain rules in full).  Tracking issue
of the family: https://github.com/flashinfer-ai/flashinfer/issues/5680.

The public module is a stable API; this backend package is experimental (its
own names carry no compatibility guarantee).  SM100 (B200 / GB200) is the
acceptance architecture; SM103 (B300 / GB300) and
SM107 (R200) are compiled from the same sources (SM107 runs SM103's programs,
built for `sm_107a`, with SM103's GEMM tile rule tables except the long chunks'
3-deep dX operand ring; its zero-gradient row skip has a 1946-row floor and
SM103's two-slice counted dX).  The feature does not participate in automatic
backend selection, autotuning or trace-apply.  Its libraries are loaded
prebuilt from `aot/` when the package ships them and JIT-built by FlashInfer
from `build.json` otherwise (see "Layout of this package").

## Mode rules

The mode is part of the module name (issue #8928): no entry of either package
takes a `deterministic` keyword, and passing one is Python's own `TypeError`.

| | deterministic | non-deterministic (this package) |
| --- | --- | --- |
| public module | `flashinfer.cake_lm_head_det_train` | `flashinfer.cake_lm_head_nondet_train` |
| backend package | `flashinfer.experimental.cake.cake_lm_head_det_train` | `flashinfer.experimental.cake.cake_lm_head_nondet_train` |
| public call | `cake_lm_head_det_train_*(...)` | `cake_lm_head_nondet_train_*(...)` |
| `loss`, `logp`, `lse` | fixed order, bitwise run to run | fixed order, bitwise run to run (the same programs) |
| zero-weight policy rows (`loss_weights == 0`) | in the chunk loop | left out of the gradient path (see "Zero-weight policy rows") |
| log-probability rows with `dlogp == 0` | in the chunk loop | ordered last; the backward's GEMMs skip their work (see "Zero-gradient rows of the log-probability backward") |
| `dW` | chunk order, bitwise run to run | chunk order, bitwise run to run (the same programs) |
| `dX` of a K-sliced chunk | slice 0 stored, slices `>= 1` in FP32 slabs added in slab order | one launch per chunk adds every slice into the `-0.0`-filled FP32 accumulator rows in an unordered FP32 reduction; a one-slice chunk stores its rows as the deterministic package does |
| precision boundaries | BF16 logits and `dlogits`, FP32 loss arithmetic, FP32 `dX` / `dW` accumulation, one final cast each | the same |
| under `torch.use_deterministic_algorithms(True)` | runs, never alerts | raises PyTorch's alert before a K-sliced dX launch (below); with `warn_only=True` it warns and runs this package; nothing falls back to the deterministic package |

## Public entry points (`flashinfer/cake_lm_head_nondet_train.py`)

```python
from flashinfer.cake_lm_head_nondet_train import (
    cake_lm_head_nondet_train_logprob,
    cake_lm_head_nondet_train_loss,
)

loss = cake_lm_head_nondet_train_loss(X, W, labels, objective="ce", loss_div=loss_div)
loss, logp = cake_lm_head_nondet_train_loss(X, W, labels, objective="policy", infer_logp=infer_logp,
                                            loss_weights=loss_weights, return_logp=True)
logp = cake_lm_head_nondet_train_logprob(X, W, labels, chunk_size=4096)
logp = cake_lm_head_nondet_train_logprob(X, W, labels, temperature=0.7)  # scalar temperature
logp = cake_lm_head_nondet_train_logprob(X, W, labels, temperature=tau)  # FP32 [T] on X's device
```

The module's entries are `cake_lm_head_nondet_train_loss`,
`cake_lm_head_nondet_train_logprob`, `cake_lm_head_nondet_train_preload`,
`cake_lm_head_nondet_train_support_logprob` and
`cake_lm_head_nondet_train_support_kl`.  They keep the argument checks and the
`FLASHINFER_CAKE_LM_HEAD_LOSS_*` fallbacks of the switches
(`flashinfer._cake_lm_head_train_knobs`, shared with the deterministic
module).  The module imports this package lazily, at its first call, and
imports no deterministic module.  The deterministic entries are
`flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_*`.

The arguments, outputs, memory rule, switches, ignored-row semantics,
compaction, the policy objective and its clip, the opt-in forms
(`memory_mode`, `num_valid_rows`, `check_num_valid_rows`), the log-probability
entry's `temperature` / `check_temperature` and the returned dtypes are those
of the deterministic entries (`cake_lm_head_det_train_loss` /
`cake_lm_head_det_train_logprob`).  Statements there that two forms of a call
give "bitwise the same" output compare the same operations in one run; in
this package the K-sliced dX reduction may round differently from one run to
the next.

The package's own entries (`cake_backend.forward_loss`,
`forward_logprob`, `backward_logprob`, `forward_logprob_keep`,
`backward_logprob_keep`, `prepare_lm_head_loss`,
`cake_lm_head_nondet_train_loss` / `cake_lm_head_nondet_train_logprob` /
`cake_lm_head_nondet_train_logprob_keep`, the temperature entries
`cake_lm_head_nondet_train_logprob_temperature`, `forward_logprob_temperature`,
`backward_logprob_temperature` and `prepare_lm_head_loss_temperature`, the same
in `cake_backend_opt_in`, and the candidate-support entries
`cake_support.cake_lm_head_nondet_train_support_logprob` /
`cake_lm_head_nondet_train_support_kl`) take no `deterministic` keyword, and the
autograd `Function`s take no `deterministic` position.  `backward_loss` and
`ForwardResult.backward` continue a `forward_loss` call of this package; pass
them only that call's accumulators and operands.

With `torch.use_deterministic_algorithms(True)` an entry raises PyTorch's
alert in PyTorch's wording (`RuntimeError`, or a `UserWarning` under
`warn_only=True`, after which the call runs on in this package) before it
launches the K-sliced dX reduction (`gemm_dx_red*`: a chunk of more than one
dX slice, the counted two-slice chunks of the zero-gradient row skip included;
or the candidate-support backward's dense dX at more than one slice):
`forward_loss` and a loss runner's `forward()` (the loss forward runs the dX
GEMM), `backward_logprob`, `backward_logprob_temperature` and a
log-probability runner's `backward()` (in both modules), and
`backward_logprob_keep`.  The alert names the deterministic counterpart of the
call's entry, `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_<entry>`;
nothing falls back to it, and the public module never reads torch's flag
itself.  An eager entry checks its plan where the plan is final -- after the
valid-row count, so a call on the hidden valid-row count path has launched
chunk 0's row gather and logits GEMM when a strict alert stops it -- and
before its first dX launch, and a strict alert leaves no partial binding.
Launches of one fixed order do not alert: a call whose dX chunks all have one
slice (the plain store), the weight-gradient GEMMs, the deferred last-chunk
weight gradient of `backward_loss` and the budget form's recompute, the casts,
the candidate-support backward at one dense dX slice and its sparse passes,
the log-probability forward and a call without trainable inputs.

## Kept log-probability logits (`memory_mode="keep_logits"`)

The log-probability entry's opt-in `memory_mode="keep_logits"` is the
deterministic package's kept logits (`cake_lm_head_det_train_logprob_keep`) under
this package's names: `cake_backend.cake_lm_head_nondet_train_logprob_keep`, the
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
  the call is `cake_lm_head_nondet_train_logprob`.  A non-reentrant checkpoint drops the
  kept logits after the checkpointed forward and its recomputation keeps them
  again.
* The kept `2 * T_v * V` bytes live from the forward to the backward, outside
  the memory rule's chunk bound.  `memory_mode="keep_logits"` takes no
  `num_valid_rows` and is a `ValueError` on the loss entry and, at the opt-in
  level, on the reference backend.
* `logp`, `lse` and `dW` are the default path's bits.  `dX` is formed from the
  same `dz` and the same products; a chunk of more than one dX slice adds them
  in the unordered reduction below, as the default path does.
* The public `cake_lm_head_nondet_train_logprob` passes
  `memory_mode="keep_logits"` unchanged to `cake_backend_opt_in`.

## Temperature of the log-probability entry (`temperature=`)

`cake_lm_head_nondet_train_logprob(..., temperature=..., check_temperature=...)` is the
deterministic entry's temperature (`cake_lm_head_det_train_logprob`) under this
package's names: the same contract (`None`, `1.0` or any scalar whose FP32
value is 1.0 runs the path without a temperature; a positive Python number or a
float32 `[T]` tensor on `X`'s device, not differentiable; `logp_t = q_y -
logsumexp_v(q_v)` with `q = RN(f32(z) / tau_t)`, the upstream `dlogp_t` divided
once per row, `dX` / `dW` reached only through `dz`; a bad value stops the
stream with a device-side assertion, `check_temperature=True` raises
`ValueError`), the same kernel forms under this package's prefix
(`gemm_logits_lbl_tau`, `gemm_logits_dz_tau`, `gemm_logits_mcnt_tau`,
`row_finalize_tau`) and the same separate entry points in both backend modules
(`cake_lm_head_nondet_train_logprob_temperature`, `forward_logprob_temperature`,
`backward_logprob_temperature`, `prepare_lm_head_loss_temperature`,
`memory_report_temperature`; the public entry calls them only for a
`temperature` other than `None`, so a call without one runs the entries
without a temperature parameter).

* `logp` and `lse` are the deterministic entry's bits for the same call, and
  so is `dW` when no valid row has a zero `dlogp`.  `dX` is formed from the
  same `dz` and the same products; a chunk of more than one dX slice adds them
  in the unordered reduction below.
* The zero-gradient row skip (below) runs with a temperature too: the
  temperature rows follow the skip's row order (a per-token temperature is
  gathered from the caller's `[T]` tensor, which the autograd entry keeps for
  the backward), the `dz` recompute runs the counted temperature form
  `gemm_logits_dz_dcnt_tau` (no work on the rows past the chunk's count; the
  same `dz` bits as `gemm_logits_dz_tau` on the rows before it) and the
  K-sliced dX reductions and the weight-gradient GEMMs run their `_dcnt`
  forms.  A record without the counted temperature form, or
  `set_zero_dlogp_temperature_dz_counted(False)`, runs `gemm_logits_dz_tau`
  over every row of the chunk instead (a zero `dlogp` gives `dz = 0`).  With
  zero-gradient rows `dW` sums the live rows in the new row order (bits may
  differ from the deterministic entry's within the accuracy budgets).
* `memory_mode="budget"` and `num_valid_rows` take a temperature;
  `memory_mode="keep_logits"` takes none (`ValueError`), and
  `cake_lm_head_nondet_train_loss` has no `temperature` keyword.
* `cake_lm_head_nondet_train_preload(temperature=True)` also loads
  the temperature forms; without it a process that never passes a temperature
  loads (or JIT-builds) no temperature library.

## Candidate-support entry points (`cake_support.py`)

`cake_lm_head_nondet_train_support_logprob` /
`cake_lm_head_nondet_train_support_kl` have the deterministic entries'
contract (`cake_lm_head_det_train_support_logprob` / `_kl`).  One autograd
`Function`, `CakeLMHeadNondetSupportFunction`, serves both.  As part of this
package they make no run-to-run promise for the gradients, and their backward
raises the alert above only when its dense dX runs a block of
more than one K slice (a cell below; never on `sm_103a` or `sm_107a`).  Their
programs are the deterministic
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
  items) runs on the architectures with split forms (`sm_100a`, `sm_103a`,
  `sm_107a`) and keeps the bits.
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
slices included, pass on GB300.  On `sm_107a` (R200) the rule has no cell
either: the dense dX runs one slice everywhere, with the split tail as on
`sm_103a`.

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
package reads no environment variable.  `flashinfer.cake_lm_head_nondet_train`
resolves the variables through `flashinfer._cake_lm_head_train_knobs` (the
parser the deterministic module uses too) and configures this package's two
binding caches (`cake_backend.BINDING_CACHE`,
`cake_backend_opt_in.BINDING_CACHE`) with the binding-cache settings it read
at its import, on its first call.  This package adds no variable.

Process-wide settings of this package, set in code only (each setter returns
the previous value; an entry call reads them once):

| setter | default | effect |
| --- | --- | --- |
| `cake_backend.set_dx_slice_policy` | `"det"` | the dX slice rule (see "Kernel structure of one token chunk"); part of every binding key |
| `cake_backend.set_dx_red_min_rows` | `0` | a chunk of fewer rows keeps one K slice (the plain store); part of every binding key |
| `cake_backend.set_zero_weight_min_rows` | `4096` | a call of fewer rows `T` keeps its zero-weight rows in the chunk loop (see "Zero-weight policy rows") |
| `cake_backend.set_zero_dlogp_min_rows` | per architecture: `1946` on sm_100a and sm_107a, `3891` on sm_103a, `4097` elsewhere | a log-probability backward of fewer valid rows runs without the zero-gradient row skip; an integer or `None` (the skip off) applies to every architecture; with no override set the setter returns `ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT`, which passed back restores the per-architecture default (see "Zero-gradient rows of the log-probability backward"); part of the backward binding key |
| `cake_backend.set_zero_dlogp_temperature_dz_counted` | `True` | under the zero-gradient row skip a call with a temperature runs the counted temperature form of the dz arm (`gemm_logits_dz_dcnt_tau`); `False` runs the uncounted one over every row (the same outputs); part of the backward binding key |
| `cake_backend.set_gradient_placement` | `"auto"` | where the loss entries (`cake_lm_head_nondet_train_loss` of both modules, every form) run the dX / dW GEMMs. `"auto"`: a call no backward can follow (grad mode off, or neither `X` nor `W` trainable) computes the loss (and `logp`) only; inside a non-reentrant `torch.utils.checkpoint` (its own saved-tensor hooks innermost) the checkpointed forward computes the loss only and saves its inputs, and the recomputation in the backward runs the fused forward and hands the gradient accumulators to the checkpointed call's backward, so a checkpointed step forms them once and keeps no FP32 accumulator from the forward to the backward (paired per checkpoint frame and graph by call order and checked against the inputs' metadata and the call's options; without a handed result the backward runs the fused forward itself, with a `RuntimeWarning`); other hooks innermost (CPU offload), torch's deterministic algorithms, or a torch whose checkpoint lacks those hooks keep the fused forward. Under selective activation checkpointing (`context_fn` of `create_selective_checkpoint_contexts`, whose recomputation replays the checkpointed forward's operators by position) every entry of this package -- the loss, log-probability (default, budget, kept logits, with a temperature) and candidate-support entries -- runs its forward with all active torch dispatch modes popped, the checkpoint's and any user mode on the stack (a `FlopCounterMode` does not see the entry's operators; a tracing mode such as `FakeTensorMode` or a `torch.compile` trace gets real launches on fake inputs and fails loudly), and behaves as in a plain non-reentrant checkpoint; the per-device constant cells and the calling thread's count cells and `arange` constant are formed outside active dispatch modes. `"forward"`: every call with a trainable input forms the gradients in its forward (the previous behavior). The same kernels on the same operands either way: the same loss, `logp` and dW bits, and dX up to the order of its unordered K-slice reduction (the same bits with one dX slice) |
| `cake_backend.set_dx_fill_side_stream` | `False` | `True`: a remembered binding issues the `-0.0` fill of `dX_acc` on the calling thread's fill stream (its own allocator pool, joined before the first dX GEMM), so it runs beside the queued chunk-0 logits GEMM; not under CUDA graph capture, not in the prepared runner; `True` keeps up to `T * H * 4` bytes more reserved in the fill stream's pool |

The side streams, events, pinned count cells and the `arange` constant of the
hidden valid-row count are per thread (`_THREAD_STATE`): a thread's calls
never share them in flight.  A caller that runs each call on a new thread
(such as `torch.nn.DataParallel`) forms that state again on every call (a
stream, events, a pinned cell and one small allocation and kernel);
`cake_lm_head_nondet_train_preload()` forms it for the calling thread
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
  and forms the per-chunk count of live rows in device memory, in one kernel
  launch per call (`zero_dlogp_partition`; the same outputs as its PyTorch
  reference `zero_dlogp_partition_reference`, bit for bit).
* The chunk's dz arm of the logits GEMM, its K-sliced dX reduction, its
  weight-gradient accumulate and its bf16 fused cast run their `_dcnt` forms:
  each reads the chunk's count and skips the work items (or the K steps) past
  it.  The launch shape stays the chunk's full row count, so the call is the
  same set of launches for every `dlogp` and a CUDA graph capture of the
  backward replays for other `dlogp` values.  A chunk with no live row runs
  its launches without their GEMM work.
* The one-slice dX plain store has no counted form, so on the architectures
  of `DX_COUNTED_MIN_SLICES_BY_ARCH` (sm_100a, sm_103a and sm_107a: two slices) the
  call's plan runs every dX chunk the slice rule gives one slice as a
  two-slice K-sliced reduction in its counted form
  (`cake_backend.counted_dx_slices`): the chunk computes no row past its
  count.  The chunks, the other chunks' slices and the other GEMMs' forms do
  not change.  Not raised: an
  explicit dX `k_slices` in `gemm_tuning`, a record that serves one slice, a
  vocabulary of one K step, and a plan whose two-slice forms would have no
  counted form (it keeps the slice rule's counts).
* The fused dX finalize writes exact `+0` rows for the zero-gradient rows (as
  for an ignored row); `dX` of a live row is the same mathematics as without
  the skip.  `dW` sums the same nonzero products over the live rows in the new
  row order (FP32 accumulation; bits may differ within the accuracy budgets).
* The skip applies to a compacted backward that reuses the forward's row index
  (the autograd entries), with the dz arm (`logprob_dz`) and, when it forms
  `dX`, the fused dX finalize, on a record that registers the `_dcnt` forms
  and the partition kernel, and of at least
  `cake_backend.zero_dlogp_min_rows(device)` valid rows, the floor of the
  call's CUDA device index: the override of `set_zero_dlogp_min_rows` when
  one is set (an integer, or `None` = never, on every architecture), else the
  architecture's default floor -- `ZERO_DLOGP_MIN_ROWS_BY_ARCH` (the table
  below), and `ZERO_DLOGP_MIN_ROWS_DEFAULT` = 4097 (one row above a default
  4096-row chunk) elsewhere.  With no override set the setter
  returns `ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT`; passing it back (or a copy or
  an unpickled copy of it) restores the per-architecture default, and
  `zero_dlogp_min_rows()` without a device returns 4097.  The
  call's plan decides: under the skip the last chunk's bf16 fused cast runs
  its counted form -- the counted TMA-epilogue form
  (`gemm_dw_cast_bf16_tma_dcnt`) where the plan without the skip takes the
  TMA-epilogue cast (`dw_cast_tma_rule`: two or more chunks and a last chunk
  of 1 .. 1792 rows), else the counted per-thread form; the same `dW` bits
  either way -- and a plan with a form that has no counted one (an explicitly
  pinned swapped dX tile or weight-gradient TMA epilogue) runs the backward
  without the skip: no partition launch, rows in the forward's order.
* The bf16 `dW` cast moves to the last chunk with a live row (`_lc` forms,
  sm_100a and sm_103a records; a backward of two chunks or more): the live
  rows of a call are a prefix of its chunks, so every chunk's weight-gradient
  accumulate but the last reads the next chunk's count too
  (`gemm_dw_acc_lc_dcnt`, `gemm_dw_acc_gn_lc_dcnt`).  With a live next chunk it
  adds its tile into the FP32 sum as before; else it writes
  `bf16(g * (sum + tile))` (`bf16(g * tile)` on the first chunk) into `dW` and
  leaves the sum, and the deferred cast of the last chunk
  (`gemm_dw_cast_bf16_lc_dcnt`, `gemm_dw_cast_bf16_tma_lc_dcnt`) writes nothing
  when that chunk has no live row.  One program writes `dW`, and a backward
  whose trailing chunks hold no live row runs no cast-only pass.  The plan
  takes the forms whole or not at all (a record without one of them: the
  counted forms above).  Only the eager skip path runs them: a runner of
  `prepare_lm_head_loss(dcnt=...)` keeps the counted forms above, so its
  `dw_acc` keeps holding the sum of the chunks before the last.

Non-finite values: a NaN `dlogp` keeps its row live (NaN in its `dX` row and
in `dW`, as without the skip).  A non-finite `X` row of a zero-gradient row
makes `dW` NaN without the skip (`0 * inf`); with the skip it still does when
the row falls within the 63 rows after a chunk's live rows (the weight-gradient
GEMM's last 64-row K step), and not elsewhere.  Its `dX` row is an exact zero
either way.

Default floors (`ZERO_DLOGP_MIN_ROWS_BY_ARCH`, valid rows of one call):

| architecture | floor | chosen from |
| --- | --- | --- |
| sm_100a (B200) | 1946 (T = 2048 with 5 % ignored rows) | timing on B200 at 1946 and 3891 valid rows |
| sm_103a (GB300) | 3891 (T = 4096 with 5 % ignored rows) | timing on GB300 at 1946 and 3891 valid rows |
| sm_107a (R200) | 1946 (T = 2048 with 5 % ignored rows) | timing on the 212-SM and 216-SM R200 boards at 1100, 1946 and 3891 valid rows, with the two-slice counted dX |
| any other | 4097 (`ZERO_DLOGP_MIN_ROWS_DEFAULT`) | not timed below one default chunk |

The per-architecture floors put one-chunk backwards from the floor (1946 valid
rows on sm_100a and sm_107a, 3891 on sm_103a) to 4096 valid rows on the skip, which the
generic 4097 floor keeps off it.  On those calls: the `dW` bits can differ
from the call without the skip (the new row order; within the accuracy
budgets, the same bits run to run); a
non-finite `X` row of a zero-gradient row makes `dW` NaN only within the
63-row window above, as for the skip at 4097 rows and more; and the call holds
the partition's int32 outputs (12 bytes per valid row plus 4 per row of T,
about 31 KB at T = 2048 with 1946 valid rows) for the whole backward, so its peak
memory is that much higher than without the skip.
`set_zero_dlogp_min_rows(4097)` keeps every architecture at the generic floor.

Not applied: `compact_rows=False`, a backward that recounts its valid rows,
the kept-logits entry, `memory_mode="budget"`, the prepared runner and the
reference backend.  A call with a temperature takes the skip with its own
plan (above): the counted temperature form of the dz arm and the counted dX /
dW GEMMs.  A one-slice dX chunk (the plain store: pinned with `gemm_tuning`,
or one of the cases not raised above) computes every row of the chunk.

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

Old names (before issue #8928) and the names that replace them.  The old
names are gone: importing them fails, and no alias is left.

| Old | New |
| --- | --- |
| `flashinfer.chunked_lm_head.chunked_lm_head_{loss,logprob,preload,support_logprob,support_kl}(..., deterministic=False)` | `flashinfer.cake_lm_head_nondet_train.cake_lm_head_nondet_train_{loss,logprob,preload,support_logprob,support_kl}(...)` |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_*`, and `chunked_lm_head_*(...)` without `deterministic=False` | `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_*` (the deterministic module) |
| package `flashinfer.experimental.cake.cake_lm_head_loss` | package `flashinfer.experimental.cake.cake_lm_head_nondet_train` |
| `cake_backend.chunked_lm_head_{loss,logprob,logprob_temperature,logprob_keep}`, the same three in `cake_backend_opt_in` | `cake_lm_head_nondet_train_{loss,logprob,logprob_temperature,logprob_keep}` |
| `cake_backend_opt_in.preload` | `cake_backend_opt_in.cake_lm_head_nondet_train_preload` |
| `cake_support.chunked_lm_head_support_{logprob,kl}` | `cake_support.cake_lm_head_nondet_train_support_{logprob,kl}` |
| `ChunkedLmHead{Loss,Logprob,LogprobTemperature,LogprobKeep}Function` (both backend modules) | `CakeLMHeadNondet{Loss,Logprob,LogprobTemperature,LogprobKeep}Function` |
| `cake_support.SupportLogprobFunction` | `cake_support.CakeLMHeadNondetSupportFunction` |
| the `deterministic` keyword of every entry and position of every autograd `Function` | removed |

Unchanged: the internal helpers (`prepare_lm_head_loss`,
`prepare_lm_head_loss_temperature`, `LmHeadLossRunner`,
`validate_lm_head_inputs`, `lm_head_loss_workspace_size`,
`cake_jit.load_cake_lm_head_loss_module` and the generic names), the build
identities (libraries `cake_lm_head_loss_<20 hex>_<arch>`, symbols
`kernel_cake_lm_head_loss_<20 hex>`, sources `csrc/cake_lm_head_loss/`, the
registry records `cake_lm_head_loss_sm_10{0,3,7}a`, the ABI `lm_head_loss_v2`)
and the `FLASHINFER_CAKE_LM_HEAD_LOSS_*` environment variables.  No name of
the libraries the deterministic package shipped under the `cake_lm_head_loss`
prefix before its `det` naming is reused.

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
  `cake-build/1`, `package` = `flashinfer.experimental.cake.cake_lm_head_nondet_train`).
* `cake_backend.py` -- validation, PyTorch's deterministic-algorithms alert, chunk plans,
  workspace layout, memory accounting, argument-plan binding, the prepared
  runner, the torch reference path, the autograd `Function`s and the eager
  entry points.
* `cake_backend_opt_in.py` -- `memory_mode` / `num_valid_rows` forms and
  `cake_lm_head_nondet_train_preload`.
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
* `sm_107a`: the `sm_107a` record registers every stage of the `sm_103a`
  record, each program built for `sm_107a` (the deterministic family's
  `sm_107a` programs under this package's prefix, the `sm_103a` reduction
  forms and the zero-gradient row skip's counted forms and partition kernel).
  The per-architecture GEMM tile rule tables give `sm_107a` the `sm_103a`
  entries (the logits and weight-gradient raster heights, the weight-gradient
  blocked raster, the split-tail and swapped dX tiles), except the 3-deep dX
  operand ring of the long chunks (`DX_LONG_CHUNK_STAGES`): on `sm_107a` those
  chunks keep the table ring depth.  The zero-gradient row skip has its own
  `sm_107a` entries, from R200 timing: the 1946-row floor of
  `ZERO_DLOGP_MIN_ROWS_BY_ARCH` (`sm_103a`: 3891) and the two-slice counted dX
  of `DX_COUNTED_MIN_SLICES_BY_ARCH`. `dw_cast_tma_rule`
  bounds `sm_107a` as `sm_103a` (1792 rows): a short last chunk takes the
  TMA-epilogue bf16 cast, counted under the zero-gradient row skip.  The
  `sm_107a` record does not register the `_lc` forms (the cast at the last
  live chunk): its skip runs the counted forms.
* Temperature of the log-probability entry
  (https://github.com/flashinfer-ai/flashinfer/issues/7637): the
  deterministic package's four temperature forms per architecture (the row
  finalize one program for both) under this package's prefix, on `sm_100a`,
  `sm_103a` and `sm_107a` (the `sm_103a` programs).  The loss entry takes no
  temperature, as in the deterministic package.

Tests: `tests/experimental/test_cake_lm_head_nondet_train.py` (the backend:
the alert, the reference backend, and on a registered device the
deterministic suite's accuracy budgets over repeated runs, the comparison with
the deterministic package, the zero-weight policy rows, changing shapes on reused scratch, checkpoint
replay, gradient accumulation, a retained graph, concurrent streams, a
CUDA graph of the prepared runner and the kept logits),
`test_cake_lm_head_nondet_train_api.py` (the public module),
`test_cake_lm_head_nondet_train_names.py` (the names of the module and its
backend objects),
`test_cake_lm_head_nondet_train_standalone.py` (a copy imports with
`flashinfer` blocked; the prebuilt path) and
`test_cake_lm_head_nondet_train_support.py` (the candidate-support entries).
Benchmark: `benchmarks/bench_cake_lm_head_nondet_train.py` (deterministic vs
non-deterministic).

Toolchain: the deterministic package's rules apply unchanged (CUDA 12.9 or
newer; the `sm_100a` 2-CTA producer loops issue the stage's
`mbarrier.arrive.expect_tx` first when nvcc 12.9 / 13.0 builds them; the
`sm_103a` programs carry `-Xptxas -O0` for CUDA 13.0 in `nvcc_flags_if`, and
`sm_103a` calls built by CUDA 13.0 take the host-count path,
`cake_jit.toolchain_runs_hidden_count`).
