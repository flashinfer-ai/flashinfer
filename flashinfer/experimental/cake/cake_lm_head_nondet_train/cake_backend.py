"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Cake backend: chunked large-vocabulary LM-head projection + loss with a
memory-bounded backward for training on SM100 / SM103 / SM107
(https://github.com/flashinfer-ai/flashinfer/issues/5680), non-deterministic
form.

This package is the backend of ``flashinfer.cake_lm_head_nondet_train``, the
non-deterministic implementation of the chunked LM-head + loss entries.  Its
contract is the deterministic package's (``flashinfer.cake_lm_head_det_train``,
backend ``flashinfer.experimental.cake.cake_lm_head_det_train``) with one
relaxation:
this package makes no run-to-run bitwise promise for its gradients (the
deterministic package makes it).  Every precision boundary below holds
(BF16 logits and ``dlogits``, FP32 probability / loss arithmetic, FP32
accumulation of ``dX`` and of the cross-chunk weight gradient, the one final
cast of each); the dX GEMM of a K-sliced chunk stores its K-slice partial
tiles into an FP32 slab and the slab sum launched after it stores their sum, in
slice order, into the FP32 ``dX_acc`` rows: one fixed order per element, so ``dX``
repeats bit for bit from run to run and may differ
from the deterministic package's by FP32 rounding of that sum (the ordered
split-K bullet below).  ``loss``, ``logp``
and ``lse`` keep their fixed order (a policy call with zero-weight rows sums
its loss over fewer rows, see the zero-weight bullet below).  A one-pass
cross-entropy call (:func:`set_one_pass_rows`; by default a one-row loss call)
sums ``lse`` and the loss in a fixed order of its own: its ``loss``, ``logp``,
``lse`` and ``dW`` are bitwise reproducible from run to run but are not the
deterministic package's bits (both within the accuracy budgets).  Statements below that two forms give "bitwise the
same" output compare the same operations within one run.

Contract
--------
* ``X [T, H]`` BF16 hidden states (row-major; any leading stride ``ld_X >= H``
  whose byte pitch is a multiple of 16 -- otherwise a contiguous copy is made
  and reported), ``W [V, H]`` BF16 output weight (contiguous), ``labels [T]``
  int64 with ``-100`` = ignored.  ``V`` and ``H`` must be multiples of 256
  (the GEMM tiles; GLM-class geometry ``H = 6144``, ``V = 154880``).  ``T`` is
  arbitrary (irregular, not a multiple of the chunk); the token chunk ``C``
  (default 4096) bounds every vocabulary-sized intermediate.
* Precision boundary: BF16 GEMM output ``z = bf16(X @ W^T)``, promoted to FP32
  for max / log-sum-exp / loss arithmetic; ``dlogits`` BF16; ``dW`` accumulated
  in FP32 across chunks and cast ONCE at the output boundary (BF16 by default,
  FP32 on request); ``dX`` accumulated in FP32 and cast once to BF16.
* ``logp_t = z[t, y_t] - logsumexp_v z[t, v]`` for valid rows, 0 for ignored
  rows.  Cross-entropy: ``loss = -sum(logp[valid]) / loss_div`` with a
  caller-supplied positive ``loss_div`` (never a local token mean);
  ``d_t = -1 / loss_div`` on valid rows.  Policy: ``loss = -sum(w_t * min(
  exp(logp_t - infer_logp_t), 2))`` over valid rows (``w`` FP32 signed, already
  masked and normalized); ``d_t = -w_t * ratio_t`` when ``ratio_t <= 2`` and
  ``0`` when ``ratio_t > 2``.  ``dz[t, v] = d_t * (1[v = y_t] - softmax(z_t)_v)``;
  ignored rows: zero loss, zero gradient, ``logp = 0``.  ``dX = dz @ W``,
  ``dW = dz^T @ X`` chunk by chunk.
* Entry (a) ``cake_lm_head_nondet_train_loss``: the forward produces ``dX_acc`` (FP32
  ``[T, H]``) and ``dW_acc`` (FP32 ``[V, H]``) directly per chunk (three GEMMs,
  no logits recomputation); the backward applies the incoming scalar
  gradient ``g`` and performs the single cast ``dX = bf16(g * dX_acc)``,
  ``dW = cast(g * dW_acc)`` -- never mutating the saved accumulators, so a
  retained graph may run the backward repeatedly.  With the gradient
  placement ``"auto"`` (the default, :func:`set_gradient_placement`) a call no
  backward can follow forms no accumulator, and inside a non-reentrant
  ``torch.utils.checkpoint`` the recomputation forms them instead of the
  checkpointed forward (the same bits).  Entry (b)
  ``cake_lm_head_nondet_train_logprob``: the forward saves the FP32 row statistic
  ``lse[T]`` and the valid-row index, so the backward plans its chunks without
  counting the rows again (no host synchronization); the backward recomputes
  each chunk's logits (four GEMMs) with the incoming ``dlogp`` as ``d_t``
  (masked to valid rows).
* Memory rule: no logits / dlogits / probability buffer spans more than ``C``
  tokens (a batch smaller than ``C`` is one chunk).  :func:`memory_report`
  states the peak temporary bytes separately from the model weights, the
  required outputs and the FP32 gradient accumulators.
* Valid-row compaction (``compact_rows``, default on): when some labels are
  ``-100`` the chunk loop runs over the valid rows only -- the row index
  ``idx = (labels >= 0).nonzero()`` is formed once per call (one device
  synchronization, ``nonzero``'s count, for the count and the index together:
  :func:`valid_rows`, which forms them on the host from the mask's readback
  for calls of at most :data:`HOST_COUNT_MAX_ROWS` rows; the log-probability
  backward reuses the forward's index), each chunk's rows of ``X`` are gathered into one reusable BF16 ``[C, H]`` buffer, the row operands
  are compacted to ``[T_v]``, the FP32 ``dX`` accumulator is ``[T_v, H]`` and
  ``logp`` / ``dX`` are scattered back to ``[T]`` with exact zeros on the
  ignored rows.  Ignored rows contribute exactly zero to the loss and both
  gradients, so the result is the same computation over fewer rows: per row
  ``logp`` / ``lse`` are bitwise those of the uncompacted path, ``dX`` rows too
  whenever the chunk's K-slice count agrees; the ``loss`` and ``dW``
  reductions run over different chunk boundaries, hence differ by FP32
  rounding.  All rows valid: the uncompacted path (no gather, no scatter).
  On this host-count path (calls the hidden valid-row count below does not
  serve) a remembered forward launches chunk 0's row gather and logits GEMM
  right behind the count and binds the rest of the call -- the compacted
  labels, the policy operands, the row vectors, the accumulators -- behind
  them (:func:`forward_early_keys`): the same kernels on the same operands in
  the same stream order per kernel, so every output is bitwise the in-order
  sequence's.
* Ordered split-K dX (:data:`DX_RED_STAGE`): the dX GEMM of a chunk the slice rule gives ``S > 1`` K slices
  (:func:`dx_slices`) is one launch of ``gemm_dx_red``, whose ``S`` K-slice work items per output tile each store
  their FP32 tile into their own rows of the call's slab (:func:`_dx_ordered_values`), followed right away by one
  launch of the slab sum :data:`DX_SLAB_SUM_STAGE` (:func:`_dx_keys`; its counted form under the zero-gradient row
  skip), which sums the ``S`` tiles in slice order and stores the sum into the chunk's rows of ``dX_acc``.  The slab
  sum is the only writer of those rows and never reads them, so no call fills or zeroes ``dX_acc``.  Each element is
  summed in one order, so ``dX`` repeats bit for bit from run to run (its bits need not equal the deterministic
  package's).  A chunk of one slice keeps the deterministic package's plain-store ``gemm_dx`` forms, so its ``dX``
  rows are that package's bits.  The slice rule is selectable programmatically (:func:`set_dx_slice_policy`:
  ``"det"``, the deterministic package's slice counts, the default; ``"measured"``, the measured form-aware rule;
  :func:`set_dx_red_min_rows`: one slice below a chunk size, default 0).
* Fused dX finalize (``dx_finalize``, default on): a compacted call's dX output boundary writes the
  ``[T, H]`` BF16 ``dX`` in one pass (``scale_cast_scatter_bf16``: ``bf16(g *
  dX_acc[compact row])`` on the valid rows, exact zeros elsewhere, addressed by
  the inclusive int32 scan of the valid-row mask) instead of the flat cast, the
  zero fill and the ``index_copy_``.  The same FP32 operations in the same
  order: ``dX`` is bitwise the previous path's (``dx_finalize=False`` restores
  that path).
* dW side stream (``dw_stream``: ``None`` / ``"auto"`` (default) = calls of
  three or more chunks, ``True`` = every multi-chunk call, ``False`` = never):
  each chunk's weight-gradient accumulate GEMM is launched on a
  per-device side stream of the calling thread, forked after the chunk's ``row_grad`` (``dz`` ready)
  and joined before the next chunk reuses the chunk buffer and before the call
  returns, so its CTAs fill the tail wave of the chunk's dX GEMM instead of
  queueing behind it.  Kernels, buffers, launch order per kernel and numerics
  are unchanged (the dX GEMM and its slab sum write the dX rows, the accumulate
  reads the same ``dz`` and read-modify-writes ``dW_acc``): ``loss`` / ``logp``
  / ``dX`` / ``dW`` are bitwise the single-stream path's (the slab sum adds the
  dX slices in slice order on either).  The ``auto`` rule's three chunks are a design
  choice (:func:`dw_side_stream`): the side stream adds one cross-stream join
  per chunk and overlaps only the chunks before a deferred last one, and a
  one-chunk call defers its only accumulate to the backward.
* Hidden valid-row count (``hidden_count``, default on; ``cake`` backend
  only): a compacted call forms its valid-row index on the device (``labels >= 0`` ->
  inclusive int32 ``cumsum`` -> ``searchsorted``), gathers chunk 0's rows of
  ``X`` with the ``gather_rows_bf16`` kernel and runs chunk 0's logits GEMM in
  its device-count form (``gemm_logits_mcnt`` / ``gemm_logits_nostats_mcnt``:
  the stores bounded by the count read from device memory, the buffer extent
  ``min(chunk_size, T)`` as ``M``) before the count reaches the host through a
  pinned cell and a CUDA event -- the host waits for the three compaction
  kernels and a 4-byte copy instead of a ``sum().item()`` and a ``nonzero``
  before the first launch (:func:`_hidden_count_begin`).  Every row valid: the
  uncompacted plan with chunk 0's GEMM already done (its ``X`` rows for the
  weight gradient stay the input view); every row ignored: zeros without a
  further launch; otherwise the compacted plan over the counted rows with
  chunk 0's gather and GEMM skipped.  The same kernels per row, so ``loss`` /
  ``logp`` / ``dX`` / ``dW`` are bitwise the shipped path's
  (``hidden_count=False`` restores it).  Calls the device-count GEMM cannot serve -- an ``X`` that needs a
  contiguous copy or whose base is not 16-byte aligned, a program without the
  kernels -- take the shipped path.
* Log-probability logits GEMM arms (``logprob_dz`` / ``logprob_lbl``, default
  on; ``cake`` backend, records that register them): the backward's
  recompute of each chunk is one ``gemm_logits_dz`` launch whose epilogue
  writes ``dz_c = bf16(d_t * (1[v = y_t] - exp(z - lse_t)))`` -- the
  per-element sequence of ``row_grad`` on the same BF16-rounded logits, zeros
  on ignored rows -- in place of the logits, so ``row_grad`` and the logits
  store disappear; the forward's logits GEMM ``gemm_logits_lbl`` writes the
  row statistics and the logit at the (clamped) label column alone, the one
  element ``row_finalize`` reads.  The same arithmetic per element, so
  ``lse`` / ``logp`` / ``dX`` / ``dW`` are bitwise the previous path's
  (``logprob_dz=False`` / ``logprob_lbl=False`` restore it).
* Zero-weight policy rows (:func:`gradient_rows`; ``objective="policy"``, compacted calls of
  :func:`forward_loss` and the loss autograd entry): a valid row whose ``loss_weights`` entry is exactly zero (``+0``
  or ``-0``; known before the forward, like a ``-100`` label) contributes exactly zero loss and zero gradient, so the
  chunk loop runs over the gradient rows ``G = (labels >= 0) & (loss_weights != 0)`` only (a NaN weight stays in
  ``G``).  Both row counts come back in the call's one count readback (host path: one 8-byte copy; hidden valid-row
  count: the same pinned copy, 8 bytes).  The zero-weight rows ``Z = (labels >= 0) & (loss_weights == 0)`` get their
  ``logp`` from a logp-only pass of the log-probability forward kernels (``gemm_logits_lbl`` + ``row_finalize``,
  chunk-bounded buffers) when ``logp`` is returned, scattered into the same ``[T]`` output (the same per-row logits
  arithmetic as the loss path's); their ``dX`` rows are exact zeros.  A zero-weight row is treated like an ignored
  row for the gradients: a non-finite value in its ``X`` row, its logits or its ``infer_logp`` does not reach
  ``loss`` / ``dX`` / ``dW`` (the deterministic package gives NaN there through ``0 * inf``).  Finite inputs: the
  same mathematics; ``dW`` (and the FP64 loss sum) group their additions over different chunk rows, so their bits
  may differ from the deterministic package's within the accuracy budgets.  Not applied (the plain path runs):
  ``compact_rows=False``, ``objective="ce"``, the log-probability entry, the prepared runner
  (:func:`prepare_lm_head_loss`) and a caller-given ``num_valid_rows`` (the opt-in module; its ``"budget"`` form splits
  the rows as here).
* Non-deterministic mode (the only mode of this package; the mode is in its
  name, and no entry takes a ``deterministic`` keyword): the chunks
  run in sequential order; the dX GEMM's K-slice partials combine through the
  ordered FP32 reduction (:func:`_dx_keys`, :data:`DX_RED_STAGE`), so ``dX``
  repeats bit for bit.  Even so, the package makes no run-to-run bitwise promise
  for its gradients (the deterministic package ``cake_lm_head_det_train``
  makes it) and keeps PyTorch's alert as part of that contract.  With
  ``torch.use_deterministic_algorithms(True)`` an entry raises PyTorch's alert
  where its plan is final and before it launches that K-sliced reduction (a
  chunk of more than one dX slice, :func:`_alert_unordered_dx`); a call
  without such a chunk (one-slice dX, the weight-gradient GEMMs, the
  deferred weight gradient, the casts) does not alert; nothing falls back to
  the deterministic package.  The alert names the deterministic counterpart of
  the entry, ``flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_<entry>``.
* Temperature (log-probability entry, ``temperature`` / ``check_temperature``;
  :func:`_resolve_temperature`): a positive scalar or a per-token FP32 ``[T]``
  ``tau``, not differentiable.  ``logp_t = z_y / tau_t - logsumexp_v(z_v /
  tau_t)``, ``dz = (dlogp_t / tau_t) * (1[v = y_t] - softmax(z_t / tau_t))``:
  the BF16 projection is unchanged, each BF16-rounded logit is divided once in
  FP32 (``q = RN(z / tau_t)``, IEEE division) and ``dlogp_t`` once per row; the
  kernels are the temperature forms of the arms and of ``row_finalize``
  (:data:`TEMPERATURE_STAGES`), indexed by the compact row like ``labels`` /
  ``lse``.  A scalar is a ``torch.full`` of its FP32 value (bitwise the same
  as that tensor); ``None`` / ``1.0`` run the path without a temperature
  (no new operation, no new kernel).  Every value must be positive and finite,
  ignored rows included (a device-side assertion on CUDA; ``check_temperature``
  reads it back).  ``|z / tau_t| < 3e38`` keeps every output defined.

Every allocation happens in :func:`prepare_lm_head_loss`; the returned runner
launches with no CUDA allocation and no host synchronization, so a runner (or
a CUDA graph capturing it) replays for new values written into the bound
tensors.  Kernels are reached through the argument plans of the registry
records in ``cake_jit.MODULES``; the ``abi`` field of a record names the
keyword set its kernels expect (see :data:`ABI_CONTRACT` and
:func:`stage_values`).  ``backend="reference"`` runs the same chunk loop with
PyTorch operators at the same rounding boundaries (a host-layer development
aid that needs no generated program and no CUDA device).

The eager entry points (:func:`forward_loss`, :func:`backward_loss`,
:func:`forward_logprob`, :func:`backward_logprob` and the autograd Functions
behind the public API) validate and bind once per *input binding* and remember
the result in :data:`BINDING_CACHE` (see :class:`BindingCache`): a later call
whose inputs have the same ``(data_ptr, shape, stride, dtype)`` and options
(the log-probability backward's ``lse`` / ``dlogp``: the same shape, stride
and dtype) slots the current tensors, freshly allocated outputs and per-call
scratch into the remembered argument plans and launches.  A call without rows
(``T == 0``) returns loss 0, an empty ``logp`` and zero gradients without binding or
launching anything.

Every switch of this module (``compact_rows``, ``fuse_dw_cast``,
``dx_finalize``, ``dw_stream``, ``hidden_count``, ``gemm_tuning``,
``logprob_dz``, ``logprob_lbl``) is a keyword argument of the entry points
that it affects, ``None`` selecting the documented default (the
``*_default()`` functions); the binding cache is configured through
:meth:`BindingCache.configure`, the dX slice rule through
:func:`set_dx_slice_policy` / :func:`set_dx_red_min_rows`, the zero-weight split's
smallest call through :func:`set_zero_weight_min_rows` and the loss entry's
gradient placement through :func:`set_gradient_placement`.  This module reads
no environment variable.
FlashInfer's public facade (``flashinfer.cake_lm_head_nondet_train``, through
``flashinfer._cake_lm_head_train_knobs``) owns the A/B environment variables
of the first six and passes the resolved values down; it leaves the
log-probability arms at their defaults.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import inspect
import math
import os
import re
import struct
import threading
import types
import warnings
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field, fields, replace
from typing import (
    Any,
    Callable,
    ClassVar,
    NamedTuple,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
)

import torch
import torch.utils._python_dispatch as _torch_dispatch
import torch.utils.checkpoint as _torch_checkpoint

from .cake_jit import (
    BASE_STAGES,
    GEMM_STAGES,
    MODULES,
    STAGES,
    STAGES_WITH_TEMPERATURE,
    TEMPERATURE_GEMM_STAGES,
    load_cake_lm_head_loss_module,
    registered_stages,
    registered_stages_with_temperature,
    select_module,
    toolchain_runs_hidden_count,
)

IGNORE_INDEX = -100
DEFAULT_CHUNK_SIZE = 4096
RATIO_CLIP = 2.0
OBJECTIVES = ("ce", "policy")
ENTRIES = ("loss", "logprob")
BACKENDS = ("cake", "reference")
GRAD_WEIGHT_DTYPES = (torch.bfloat16, torch.float32)
# ``mode`` scalar of ``row_finalize`` / ``loss_reduce``: which per-row quantities the stage forms.
MODE_CE, MODE_POLICY, MODE_EXTERNAL, MODE_NONE = 0, 1, 2, 3
WORKSPACE_ALIGN = 256
# One registry record and its build.json libraries per architecture (SM107 runs SM103's programs, built for sm_107a;
# the per-architecture GEMM tile rule tables give sm_107a the sm_103a entries, except DX_LONG_CHUNK_STAGES and the
# logits height of LOGITS_LONG_RASTER_GROUPS (the default height 32 on sm_107a); the zero-gradient row skip has R200-timed sm_107a entries: the 1946-row floor of ZERO_DLOGP_MIN_ROWS_BY_ARCH and the
# two-slice counted dX of DX_COUNTED_MIN_SLICES_BY_ARCH).
SUPPORTED_COMPUTE_CAPABILITIES = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}

# Host binding profile: the keyword set of the generated kernels (see
# :func:`stage_values`: the host provides, per stage, the kernel's own argument
# names; a kernel's argument plan selects from them).  A record with another
# ``abi`` (a plumbing placeholder program) is not served by this backend.
ABI_CONTRACT = "lm_head_loss_v2"
# The hidden valid-row count serves calls of this many chunks or more (``ceil(T / chunk)``): its fixed host cost per
# call (the plan on the count after the readback, the launches it orders) is hidden behind chunk 0's logits GEMM only
# from there; one- and two-chunk calls take the host-count path (the deterministic package's threshold; same kernels,
# bitwise the same outputs).
HIDDEN_COUNT_MIN_CHUNKS = 3
# Calls of at most this many rows form the valid-row count and index of the host-count path on the host
# (:func:`valid_rows`): the bool mask ``labels >= 0`` comes back in one copy of ``T`` bytes -- the call's one
# synchronization -- and the index goes back in one copy, instead of ``nonzero``'s count, scan and select kernels and
# their API calls.  Above it the device forms the index.  The same index and mask either way.  With some rows ignored,
# the index copy back adds host time to :func:`valid_rows` that grows with the rows (the deterministic package's
# threshold).
HOST_COUNT_MAX_ROWS = 1024
# Small calls (:func:`uncompacted_max_rows`, :func:`_small_call_rows`): a call of an autograd entry (loss, log-probability,
# temperature) that would run compacted on CUDA with ``backend="cake"``, of at most this many rows and of one chunk
# (``0 < T <= min(rows, chunk_size)``), runs uncompacted instead -- every row in the chunk loop, no valid-row count, so no
# host synchronization: the readback of :func:`valid_rows` and the host time after it, exposed on such a call, cost
# more than the ignored rows' share of its kernels.  The entry first stages ``X`` into a buffer whose ignored rows
# (``labels < 0``) are zero (one device ``where``; policy ``infer_logp`` / ``loss_weights`` likewise), so a non-finite
# value in an ignored row reaches no output, as on the compacted path that never reads such a row; the staged buffer
# is what the call saves for its backward.  All rows valid: the same launches and bits as the compacted call (which
# keeps no index then).  With ignored rows: exact zero contributions of the ignored rows (``loss`` / ``dW`` group their
# additions over other rows, so their bits may differ within the accuracy budgets).  Every row ignored: zeros through
# the kernels (no early return: that needs the count).  Not applied: a call the zero-weight split would serve
# (:func:`zero_weight_split`), an explicit ``compact_rows=False``, the reference backend, CPU tensors, the low-level
# ``forward_*`` / ``backward_*`` functions and the kept-logits entry (each keeps its own resolution).
UNCOMPACTED_MAX_ROWS_DEFAULT = 256
_UNCOMPACTED_MAX_ROWS = {"rows": UNCOMPACTED_MAX_ROWS_DEFAULT}
# Small-call kept logits (:func:`set_small_t_keep`, default on): the log-probability entry
# (:func:`cake_lm_head_nondet_train_logprob`) of at most :data:`SMALL_T_KEEP_MAX_ROWS` rows where a backward can follow
# runs the kept-logits Function (:func:`cake_lm_head_nondet_train_logprob_keep`; a small call on its staged rows,
# uncompacted): the backward forms ``dz`` from the kept BF16 logits instead of recomputing them (``2 * T * V`` bytes
# held from the forward to the backward, at most 79 MB at ``V = 154880``).  Bitwise the same ``logp``; ``dX`` the same
# products; bitwise the same ``dW`` for a call of one chunk (``T <= chunk_size``: with more chunks the kept form's one
# weight-gradient GEMM over every row groups ``dW`` differently from the default Function's chunk loop or
# vocabulary-major form, within the accuracy budgets).
SMALL_T_KEEP_MAX_ROWS = 256
_SMALL_T_KEEP = {"on": True}
# Small-call row padding (:func:`small_pad_rows`): a small call of ``T < rows`` rows, when ``rows <=
# min(uncompacted_max_rows(), chunk_size)``, stages ``rows`` rows -- its own, then ``rows - T`` ignored rows (zero ``X``
# rows, ``IGNORE_INDEX`` labels, zero weights / reference log-probabilities, temperature 1) -- and returns the first
# ``T`` rows of its per-row outputs: the logits GEMMs of a one-row call run well below the bandwidth of the same program
# at 16 rows and more (GB300 ``gemm_logits``: 0.47 ms at one row, 0.32 ms at 16 rows).  ``rows`` is
# :data:`SMALL_PAD_ROWS_DEFAULT`, except for the log-probability entries (plain and with a temperature) on the
# architectures of :data:`SMALL_PAD_ROWS_LOGPROB_BY_ARCH`: on R200 (sm_107a) a log-probability call of 256 rows runs
# below one of 16 to 64 rows (FlashInfer c8a4d8a2e + this padding, CUDA events, median of the timed rounds per arm,
# R200 dlcluster nodes, ms at 16 / 64 / 128 / 192 / 256 rows: T = 1 log-probability forward 0.352 / 0.359 / 0.366 / 0.252 /
# 0.223, forward + backward 0.856 / 0.863 / 0.836 / 0.687 / 0.676; T = 1 temperature forward + backward 1.234 /
# 1.262 / 1.225 / 0.997 / 1.009), while the loss entry keeps 16 rows there (its backward grows with the rows: 0.212 /
# 0.212 / 0.224 / 0.237 / 0.251) and B200 keeps 16 rows everywhere (256 rows: T = 1 log-probability forward + backward
# 1.095 -> 1.213).  :func:`set_small_pad_rows` replaces the defaults with one row count for every entry (0 = off).
# The loss entry's pad applies to the calls the one-pass cross-entropy does not take (:func:`one_pass_rows`; by default
# a T = 1 cross-entropy call at ``ONE_PASS_HIDDEN`` runs one-pass and is not staged; the loss figures above are of the
# staged call).
# The valid rows' products are those of the unpadded call (the padded rows add exact zeros).  Every staged tensor of a
# call is a view of ONE allocation (:func:`_small_call_stage`), so a repeated call meets the same addresses -- and the
# remembered bindings (:data:`BINDING_CACHE`) -- whenever the caching allocator returns the same block.
SMALL_PAD_ROWS_DEFAULT = 16
SMALL_PAD_ROWS_LOGPROB_BY_ARCH: dict[str, int] = {"sm_107a": 256}
# None: the per-entry / per-architecture defaults of small_pad_rows()
_SMALL_PAD_ROWS: dict[str, Optional[int]] = {"rows": None}
SUPPORTED_ABIS = (ABI_CONTRACT,)

# Geometry the kernels of a record were built for (record field ``geometry``;
# the defaults are the GLM-class contract).
GEOMETRY_DEFAULTS: dict[str, Any] = dict(
    stats_tile=256,  # vocabulary columns per (max, sum-exp) partial: stats[C, V / stats_tile, 2]
    row_tile=128,  # output rows per CTA of the GEMMs (m_tiles = ceil(rows / row_tile), rounded to the CTA pair)
    cta_group=2,  # CTAs per GEMM cluster (the pair takes adjacent row tiles)
    k_block=64,  # K step of the GEMMs (k_iters of the weight-gradient GEMM = ceil(rows_c / k_block))
    cast_vec=8,  # elements per vector of the scale-cast kernels (num_vecs = numel / cast_vec)
    vocab_multiple=256,  # V % vocab_multiple == 0
    hidden_multiple=256,  # H % hidden_multiple == 0
    ld_multiple=8,  # row stride of X in elements (16 B TMA pitch)
    labels_dtype="int64",  # element type the row kernels read (``int32`` = one host cast per call)
    hidden=None,  # accepted in a record's geometry table and not read: the kernels take H as a launch scalar
    vocab=None,  # ... and V
    # CTAs per thread-block cluster of each GEMM (the cluster takes adjacent row tiles, so ``m_tiles`` rounds up to it;
    # 2 = the CTA pair, 4 = two pairs sharing the B operand by multicast).  Checked against the module's launch cluster:
    # a module launched with a preferred cluster dimension (two pairs sharing the A operand where the GPC topology allows,
    # a pair elsewhere) declares its required / fallback cluster here, and its grid rule counts the wider work items.
    logits_cluster_ctas=2,
    dx_cluster_ctas=2,
    dw_cluster_ctas=2,
    # Output columns of one work item of each GEMM (the accumulator tile times the item's column tiles): ``N`` must be a
    # multiple of it, ``N // item columns`` is the kernel's column-item count and the default ``group_n`` (the 1-D raster).
    # A record declares its kernels' widths; the defaults (one 256-column tile) are the reference path's geometry.
    logits_item_cols=256,
    dx_item_cols=256,
    dw_item_cols=256,
    # Default raster height of each GEMM in row tiles (the ``group_m`` launch scalar when no rule or explicit knob sets one).
    logits_group_m=32,
    dx_group_m=16,
    dw_group_m=8,
)
LABEL_DTYPES = {"int64": torch.int64, "int32": torch.int32}
_GEOMETRY_OPTIONAL = ("hidden", "vocab")

# Values the host provides to every stage of one chunk (the kernels' own
# argument names; the argument plan of a stage selects from them).  Tensors:
# ``A`` / ``B`` / ``C`` / ``STATS_OUT`` of the GEMMs (per stage: X_c / W / z_c /
# stats; dz_c / W / dX_acc rows; dz_c / X_c / dW_acc), ``stats``, ``z`` (the
# ``[C, V]`` BF16 workspace: ``z_c`` then ``dz_c`` in place), full-length ``[T]``
# vectors ``labels`` / ``lse`` / ``logp`` / ``infer_logp`` / ``loss_weights`` /
# ``d_in`` (indexed by ``row0 + r``), chunk-local ``d`` / ``term`` (``[C]``), the
# ``loss_acc`` / ``loss_out`` cells, for the casts ``acc`` / ``g`` / ``out`` and,
# for the dX slab sum, ``dx`` / ``ws`` (:func:`dx_slab_sum_operands`).
# Scalars: the GEMM launch scalars ``GEMM_SCALARS`` (``M`` / ``m_tiles`` /
# ``k_iters`` / ``first_chunk`` / ``ws_slab`` and the geometry ``ldc``, the K
# slicing ``k_slices`` / ``k_slice_iters``, the raster ``group_m`` / ``group_n``:
# :func:`gemm_scalars`), ``rows_c`` / ``row0`` / ``V`` / ``num_tiles`` / ``mode``
# / ``loss_div`` / ``last_chunk`` / ``d_off`` of the row kernels, ``num_vecs`` of
# the casts and the slab sum, ``ws_slab`` / ``n_slabs`` / ``row_vecs`` of the slab
# sum, and ``T`` / ``H`` / ``C`` for grid rules.
STAGE_TENSORS = {
    **{stage: ("A", "B", "C", "STATS_OUT", "WS") for stage in GEMM_STAGES},
    # the log-probability arms of the logits GEMM also read the per-token rows (``d_in`` = the kernel's ``dlogp``)
    "gemm_logits_dz": ("A", "B", "C", "STATS_OUT", "WS", "labels", "lse", "d_in"),
    "gemm_logits_lbl": ("A", "B", "C", "STATS_OUT", "WS", "labels"),
    # the zero-gradient row skip's forms also read the chunk's count cell
    **{
        stage: ("A", "B", "C", "STATS_OUT", "WS", "CNT")
        for stage in GEMM_STAGES
        if stage.endswith("_dcnt")
    },
    # the accumulate's cast at the last live chunk also reads the next chunk's count cell
    **{
        stage: ("A", "B", "C", "STATS_OUT", "WS", "CNT", "CNT_NEXT")
        for stage in GEMM_STAGES
        if stage.startswith("gemm_dw_acc") and stage.endswith("_lc_dcnt")
    },
    "gemm_logits_dz_dcnt": (
        "A",
        "B",
        "C",
        "STATS_OUT",
        "WS",
        "labels",
        "lse",
        "d_in",
        "CNT",
    ),
    "row_finalize": (
        "stats",
        "z",
        "labels",
        "infer_logp",
        "loss_weights",
        "d_in",
        "lse",
        "logp",
        "d",
        "term",
    ),
    "loss_reduce": ("term", "loss_acc", "loss_out"),
    "row_grad": ("z", "labels", "lse", "d"),
    "scale_cast_bf16": ("acc", "g", "out"),
    "scale_cast_f32": ("acc", "g", "out"),
    "scale_cast_scatter_bf16": ("acc", "g", "scan", "idx_lo", "out"),
    # the slab sum after a K-sliced dX GEMM (:data:`DX_SLAB_SUM_STAGES`); its counted form also reads the chunk's count,
    # its accumulating form reads ``dx`` too (:data:`DX_SLAB_SUM_ACC_STAGE`)
    "dx_slab_sum": ("dx", "ws"),
    "dx_slab_sum_dcnt": ("dx", "ws", "CNT"),
    "dx_slab_sum_acc": ("dx", "ws"),
    "gather_rows_bf16": ("x", "idx_lo", "count", "out"),
    "zero_dlogp_partition": (
        "labels",
        "dlogp",
        "lse",
        "order",
        "lse_out",
        "scan",
        "counts",
    ),
}
# the temperature forms (:data:`TEMPERATURE_STAGES`): their twin's tensors plus the compact per-token
# ``temperature[row0 + r]``; kept out of :data:`STAGE_TENSORS`, which a call without a temperature reads
TEMPERATURE_STAGE_TENSORS = {
    "gemm_logits_mcnt_tau": ("A", "B", "C", "STATS_OUT", "WS", "temperature"),
    "gemm_logits_dz_tau": (
        "A",
        "B",
        "C",
        "STATS_OUT",
        "WS",
        "labels",
        "lse",
        "d_in",
        "temperature",
    ),
    # the counted form (the zero-gradient row skip): the twin ``gemm_logits_dz_dcnt``'s tensors plus the temperature
    "gemm_logits_dz_dcnt_tau": (
        "A",
        "B",
        "C",
        "STATS_OUT",
        "WS",
        "labels",
        "lse",
        "d_in",
        "temperature",
        "CNT",
    ),
    "gemm_logits_lbl_tau": ("A", "B", "C", "STATS_OUT", "WS", "labels", "temperature"),
    "row_finalize_tau": (
        "stats",
        "z",
        "labels",
        "infer_logp",
        "loss_weights",
        "d_in",
        "lse",
        "logp",
        "d",
        "term",
        "temperature",
    ),
}
COMMON_TENSORS = ("workspace", "tma_descriptor_workspace")
COMMON_SCALARS = (
    "rows_c",
    "row0",
    "T",
    "H",
    "V",
    "chunk",
    "num_tiles",
    "mode",
    "loss_div",
    "first_chunk",
    "last_chunk",
    "d_off",
    "ws_slab",
)
# Launch scalars of every GEMM stage (:func:`gemm_scalars`): ``M`` / ``m_tiles`` (output rows and their row tiles,
# rounded up to the cluster), ``k_iters`` (K steps of ``k_block``), ``first_chunk`` (store instead of accumulate),
# ``ws_slab`` (0: the deterministic package's K-slice slab stride, unused here), ``ldc`` (the output's leading dimension:
# V of the logits GEMM, H of the dX and weight-gradient GEMMs), ``k_slices`` / ``k_slice_iters`` (K-slice work items per
# tile and the K steps of one; the dX reduction form :data:`DX_RED_STAGE`), ``group_m`` (raster height in row tiles) and
# ``group_n`` (raster block width in column ITEMS: the whole row of items for the 1-D raster).  The generated kernels take the call geometry through them, so one record serves
# every admissible (H, V).
GEMM_SCALARS = (
    "M",
    "m_tiles",
    "k_iters",
    "first_chunk",
    "ws_slab",
    "ldc",
    "k_slices",
    "k_slice_iters",
    "group_m",
    "group_n",
)
# K-sliced reduction form of the dX GEMM: ``gemm_dx_red`` runs ``k_slices`` K-slice work items per output tile (a
# persistent grid with ``k_slices`` x more items fills the last wave; the count and the slice length ``k_slice_iters``
# are launch scalars, 2 .. :data:`DX_RED_SLICES_MAX`, every slice at least one K step).  A chunk of one slice runs the
# deterministic package's plain-store ``gemm_dx`` forms.  The slice count of a chunk comes from :func:`dx_slices`.
# ``WS`` of the plain-store GEMMs is an unused 16-float dummy (``ws_slab`` 0).
DX_RED_STAGE = "gemm_dx_red"
DX_BASES = (
    "gemm_dx",
    DX_RED_STAGE,
)  # the dX GEMM's base stages: the plain store (one K slice) and the K-sliced reduction
DX_RED_SLICES_MAX = (
    8  # K slices per tile the reduction form serves (a launch scalar of one program)
)
# The registered ``gemm_dx_red*`` programs are the ordered form (argument plans ``gemm_dx_ord`` /
# ``gemm_dx_ord_dcnt``): above one slice every K slice stores its FP32 tile into its own slab of the workspace
# ``SLAB`` (``[k_slices * m_tiles * 128, H]``; slice ``s`` writes rows ``s * m_tiles * 128 + r``) and the launch
# writes nothing into ``dX_acc``.  The slab sum :data:`DX_SLAB_SUM_STAGE`, launched right after it (:func:`_dx_keys`),
# stores ``(s_0 + s_1) + ...``, summed in slice order, into the chunk's rows: it never reads them and is their only
# writer, so the call neither fills nor zeroes ``dX_acc``, and dX repeats bit for bit.  A launch of one slice (the
# support path's later vocabulary blocks) adds its tile into ``C`` directly, touches no slab and has no slab sum.  The
# slab is one buffer per call workspace sized for the call's largest K-sliced chunk (:attr:`Plan.dx_slab_bytes`,
# allocated with the call's first K-sliced dX launch, :func:`_dx_ordered_values`); a chunk's slab sum reads it before
# the next chunk's dX launch writes it (both on the caller's stream).
DX_SLAB_SUM_STAGE = "dx_slab_sum"
# The slab sum after a counted (``_dcnt``) K-sliced dX launch: it reads the same count cell ``CNT``, stores the sum
# into the rows ``r < min(rows_c, 256 * ceil(CNT[0] / 256))`` (the row-tile pairs the counted GEMM ran) and ``-0.0``
# into the rows after (the skipped pairs' rows, whose upstream gradient is zero).
DX_SLAB_SUM_DCNT_STAGE = "dx_slab_sum_dcnt"
# The slab sums of the plan's K-sliced chunks (:meth:`Plan.dx_slab_sum_stage_of`)
DX_SLAB_SUM_STAGES = (DX_SLAB_SUM_STAGE, DX_SLAB_SUM_DCNT_STAGE)
# The accumulating slab sum: ``dx + (((s_0 + s_1) + ...) + s_{S - 1})``, one add per element, for rows that already hold
# earlier vocabulary blocks' products (the support entries' dense dX, whose first K-sliced block stores with
# :data:`DX_SLAB_SUM_STAGE`); no plan of this module launches it.
DX_SLAB_SUM_ACC_STAGE = "dx_slab_sum_acc"


def dx_slab_bytes(k_slices: int, m_tiles: int, hidden: int) -> int:
    """Bytes of the ordered dX slab of one launch: ``k_slices * m_tiles * 128`` FP32 rows of ``hidden`` (``0`` for one
    slice, which writes no slab)."""
    return (
        int(k_slices) * int(m_tiles) * 128 * int(hidden) * 4 if int(k_slices) > 1 else 0
    )


def dx_ordered_operands(
    C: torch.Tensor,
    slab: Optional[torch.Tensor],
    *,
    k_slices: int,
    m_tiles: int,
    ldc: int,
) -> dict[str, torch.Tensor]:
    """The ordered dX GEMM's workspace operand of one launch writing ``C``: ``SLAB``, the ``[k_slices * m_tiles * 128,
    ldc]`` FP32 head of ``slab`` (at least :func:`dx_slab_bytes`; never read before the launch writes it).  One slice
    touches no slab: ``C`` stands in for it (``slab`` may be ``None``)."""
    k, m, n = int(k_slices), int(m_tiles), int(ldc)
    if k > 1:
        rows = k * m * 128
        return dict(SLAB=slab.view(-1)[: rows * n].view(rows, n))
    return dict(SLAB=C)


def dx_slab_sum_stage(dx_stage: str) -> str:
    """The slab sum that follows a K-sliced dX launch of ``dx_stage``: :data:`DX_SLAB_SUM_DCNT_STAGE` after a counted
    (``_dcnt``) form, else :data:`DX_SLAB_SUM_STAGE`."""
    return (
        DX_SLAB_SUM_DCNT_STAGE
        if parse_stage(dx_stage)[1]["dcnt"]
        else DX_SLAB_SUM_STAGE
    )


def dx_slab_sum_operands(
    dx: torch.Tensor,
    slab: torch.Tensor,
    *,
    k_slices: int,
    m_tiles: int,
    ldc: int,
    cast_vec: int,
) -> dict[str, Any]:
    """The slab sum's operands after a launch of the ordered dX GEMM of ``k_slices > 1`` slices that wrote ``slab``
    (:func:`dx_ordered_operands`, the same ``m_tiles`` / ``ldc``) for the FP32 rows ``dx`` (``[rows, ldc]``): ``dx``,
    the flat slab ``ws``, the elements of one slice's slab ``ws_slab = m_tiles * 128 * ldc``, the ``num_vecs`` vectors
    of ``cast_vec`` elements of ``dx``, the slice count ``n_slabs`` and the vectors of one row ``row_vecs`` (read by the
    counted form only, which also takes the count cell ``CNT`` of the counted dX launch; the caller adds it)."""
    return dict(
        dx=dx,
        ws=slab,
        ws_slab=int(m_tiles) * 128 * int(ldc),
        num_vecs=int(dx.shape[0]) * int(ldc) // int(cast_vec),
        n_slabs=int(k_slices),
        row_vecs=int(ldc) // int(cast_vec),
    )


def _dx_slab(
    t: dict[str, Any], plan: "Plan", need: int, device: torch.device
) -> Optional[torch.Tensor]:
    """The call's ordered dX slab ``t["dx_slab"]`` (flat FP32) of at least ``need`` bytes: allocated here with the
    call's first K-sliced launch at :attr:`Plan.dx_slab_bytes` (``need`` when larger); ``None`` before that launch
    when ``need`` is 0."""
    slab = t.get("dx_slab")
    if need and (slab is None or slab.numel() * 4 < need):
        slab = torch.empty(
            (max(plan.dx_slab_bytes, need) // 4,), dtype=torch.float32, device=device
        )
        t["dx_slab"] = slab
    return slab


def _dx_ordered_values(
    t: dict[str, Any],
    plan: "Plan",
    C: torch.Tensor,
    *,
    k_slices: int,
    m_tiles: int,
    ldc: int,
) -> dict[str, torch.Tensor]:
    """:func:`dx_ordered_operands` of a launch of the call's plan writing ``C``: the slab is the call's
    ``t["dx_slab"]`` (:func:`_dx_slab`)."""
    slab = _dx_slab(t, plan, dx_slab_bytes(k_slices, m_tiles, ldc), C.device)
    return dx_ordered_operands(C, slab, k_slices=k_slices, m_tiles=m_tiles, ldc=ldc)


# Fused weight-gradient output: the LAST chunk's ``dz_c^T @ X_c`` GEMM run in the backward (where the upstream scalar
# gradient ``g`` is known) with the scale and the output cast fused into its epilogue, ``C = cast(g * (WS + tile))`` --
# or ``cast(g * tile)`` for a one-chunk plan, which then has no FP32 ``[V, H]`` accumulator at all.  ``WS`` is the FP32
# sum of the chunks before it (read, never written), ``STATS_OUT`` the ``[1]`` FP32 scale cell.  The same FP32
# operations in the same order as ``gemm_dw_acc`` followed by ``scale_cast_*``, so the weight gradient is bitwise
# identical; the flat ``scale_cast`` pass over ``dW`` disappears.  Selected by ``fuse_dw_cast`` (default on).
DW_CAST_STAGES = ("gemm_dw_cast_bf16", "gemm_dw_cast_f32")
# TMA-epilogue form of the deferred BF16 dW cast (``gemm_dw_cast_bf16_tma``): the old FP32 sum is read and the
# BF16 tile is stored as TMA sub-tiles.  It runs the cast of the LAST chunk of a call of >= 2 chunks whose last
# chunk has 1 .. DW_CAST_TMA_MAX_ROWS[arch] rows (:func:`dw_cast_tma_rule`); the per-element (old + acc) * g
# and its round-to-nearest order are the shipped form's, so dW is bitwise the same.  A one-chunk call has no old
# sum to read and keeps the shipped form.  sm_107a runs the sm_103a programs (the same 1792-row bound).
DW_CAST_TMA_MAX_ROWS: dict[str, int] = {
    "sm_100a": 1792,
    "sm_103a": 1792,
    "sm_107a": 1792,
}


# Its form under the zero-gradient row skip (:data:`DCNT_BASES`): the same TMA cast with the K steps bounded by the
# last chunk's device count of live rows, so a counted plan keeps the rule's cast (a record without it: the shipped
# cast's counted form).
DW_CAST_TMA_DCNT_STAGE = "gemm_dw_cast_bf16_tma_dcnt"


def dw_cast_tma_rule(num_chunks: int, rows_last: int, arch: Optional[str]) -> bool:
    """Does the deferred BF16 dW cast of a call of ``num_chunks`` chunks, whose last chunk has ``rows_last``
    rows, run the TMA-epilogue form on ``arch``?"""
    limit = DW_CAST_TMA_MAX_ROWS.get(arch) if arch is not None else None
    return (
        limit is not None and int(num_chunks) >= 2 and 0 < int(rows_last) <= int(limit)
    )


# Fused dX finalize (``dx_finalize``, default on): ``scale_cast_scatter_bf16`` writes a compacted call's ``[T, H]``
# BF16 ``dX`` in one pass: ``bf16(g * dX_acc[scan[r] - 1])`` on a valid output row ``r`` (``scan`` = the inclusive
# int32 count of valid rows, so ``idx[scan[r] - 1] == r`` exactly on the valid rows), exact zeros elsewhere -- the flat
# ``scale_cast``, the zero fill and the ``index_copy_`` of the compact rows in one kernel that writes every output
# element once.  Bitwise the same ``dX``.
DX_FINALIZE_STAGES = ("scale_cast_scatter_bf16",)
# Hidden valid-row count (:func:`hidden_count_default`, default on): chunk 0's rows of ``X`` are gathered by the
# ``gather_rows_bf16`` kernel -- ``out[r] = X[idx[r]]`` for ``r < min(count, num_rows)`` with the count read from device
# memory, exact zeros after; ``x`` is the int32 alias of ``X``'s storage span (``ld_x_words`` = the row pitch in 4-byte words),
# ``idx_lo`` the int64 compaction index viewed as int32 pairs, ``count`` the int32 cell -- and chunk 0's logits GEMM runs in
# its device-count form (the ``_mcnt`` variants: ``WS`` is that cell and every store is bounded by ``min(WS[0], M)`` while
# ``M`` / ``m_tiles`` / the raster rule are the buffer extent ``min(chunk, T)``), both before the count reaches the host.
HIDDEN_COUNT_STAGE = "gather_rows_bf16"
# The log-probability entry's logits GEMM arms (one launch per chunk each, no device-count form):
# ``gemm_logits_dz`` -- the backward's recompute writes ``dz_c = bf16(dlogp_t * (1[v = y_t] - exp(z - lse_t)))`` from its
# epilogue (``row_grad``'s per-element sequence on the same BF16-rounded logits, zeros on ignored rows), so the recompute
# is one launch and never stores ``z``: bitwise ``gemm_logits_nostats`` + ``row_grad``; ``gemm_logits_lbl`` -- the
# forward's logits GEMM writes the row statistics and ``z`` at the (clamped) label column alone, the one element
# ``row_finalize`` reads: bitwise the same ``lse`` / ``logp``.  Selected by the ``logprob_dz`` / ``logprob_lbl`` switches
# (default on: :func:`logprob_dz_default` / :func:`logprob_lbl_default`) when the record registers them.
LOGPROB_DZ_STAGE = "gemm_logits_dz"
LOGPROB_DZ_DCNT_STAGE = "gemm_logits_dz_dcnt"  # its form under the zero-gradient row skip (:data:`DCNT_BASES`)
# The zero-gradient row skip's row order (:func:`zero_dlogp_partition`): one launch per backward, before the chunk loop.
ZERO_DLOGP_PARTITION_STAGE = "zero_dlogp_partition"
LOGPROB_LBL_STAGE = "gemm_logits_lbl"
# Temperature forms (``_tau``; :func:`stage_form` ``temperature=True``) of the log-probability entry's kernels: the
# logits GEMM arms, the device-count logits GEMM with the statistics (chunk 0 of the hidden valid-row count path) and
# ``row_finalize``.  Each reads the compact per-token FP32 ``temperature[row0 + r]`` (the row of ``labels`` / ``lse``)
# and divides once per element, ``q = RN(f32(bf16(acc)) / tau_t)`` (IEEE division), before the statistics, the label
# logit and the ``dz`` arm's ``exp``; the ``dz`` arm also divides its row's ``dlogp_t`` once, ``dt = RN(dlogp_t /
# tau_t)``.  A call with a temperature runs exactly these forms on the ``cake`` backend
# (:func:`_check_temperature_operands` refuses a temperature for a stage without the operand); a call without one
# never runs them.
TEMPERATURE_FORM_BASES = (
    "gemm_logits",
    LOGPROB_DZ_STAGE,
    LOGPROB_LBL_STAGE,
    "row_finalize",
)
LOGPROB_DZ_TAU_STAGE = "gemm_logits_dz_tau"
# the ``dz`` arm's temperature form bounded by the device count of the chunk's live rows: the zero-gradient row skip's
# ``dz`` recompute of a call with a temperature (:data:`DCNT_BASES`; the one counted temperature form)
LOGPROB_DZ_DCNT_TAU_STAGE = "gemm_logits_dz_dcnt_tau"
LOGPROB_LBL_TAU_STAGE = "gemm_logits_lbl_tau"
HIDDEN_LOGITS_TAU_STAGE = "gemm_logits_mcnt_tau"
ROW_FINALIZE_TAU_STAGE = "row_finalize_tau"
TEMPERATURE_STAGES = (
    HIDDEN_LOGITS_TAU_STAGE,
    LOGPROB_DZ_TAU_STAGE,
    LOGPROB_DZ_DCNT_TAU_STAGE,
    LOGPROB_LBL_TAU_STAGE,
    ROW_FINALIZE_TAU_STAGE,
)
LOGITS_BASES = (
    "gemm_logits",
    "gemm_logits_nostats",
    LOGPROB_DZ_STAGE,
    LOGPROB_LBL_STAGE,
)
K_SLICE_PENALTY = 0.01  # the deterministic slice rule's score penalty per extra slab (its kernels' per-slab cost share)

# --------------------------------------------------------------------------- per-chunk instance rules
# The GEMM instances of a record come in structural FORMS -- one stage name per kernel body the generated program carries
# (:func:`stage_variant`: the device-count logits GEMM ``_mcnt``, the narrow-tile ``_tn256`` / 3-deep-ring ``_st3`` /
# split-tail ``_split`` / swapped ``_swab`` forms of both dX GEMMs -- the plain store and the K-sliced reduction
# :data:`DX_RED_STAGE` --, the 2-D blocked ``_gn`` weight-gradient GEMMs) -- and take the call geometry and the rules' numeric
# outputs as LAUNCH SCALARS (:func:`gemm_scalars`: ``ldc`` / ``k_iters`` from H and V, ``k_slices`` / ``k_slice_iters``,
# the raster height ``group_m`` and the block width ``group_n``).  The form and the scalars of a chunk are selected by the
# same rules the production launchers of the kernel source apply (same thresholds; an explicit knob wins) -- never by the
# shape itself, so a record carries exactly the forms its architecture's rules can reach, at every (H, V):
# * raster height (``group_m``): at H >= RASTER_RULE_MIN_HIDDEN a chunk of >= RASTER_RULE_MIN_ROWS rows runs the
#   logits GEMM with the shorter grouped raster and the weight-gradient GEMMs (accumulate and fused cast) with the
#   taller one (RASTER_WIDE_GROUPS per architecture); below that H, a chunk of >= LOGITS_LONG_RASTER_MIN_ROWS rows (31
#   row tiles of 128) runs the logits GEMM with the shorter raster (LOGITS_LONG_RASTER_GROUPS per architecture) and a
#   chunk of >= DW_LONG_CHUNK_MIN_ROWS rows runs the weight-gradient GEMMs with the long-chunk raster
#   (DW_LONG_CHUNK_GROUP_M: the same height on every supported architecture); every other chunk -- the default
#   geometry's chunks of up to 30 row tiles, short tail chunks -- keeps the GEMM's default height (the record geometry's
#   ``<gemm>_group_m``).  Bitwise: the raster changes the tile order only.
# * dX tile (``_tn256``): the 512-column pair tile unless H % 512 != 0 or the launch's wide work items would not fill
#   the device's SM pairs (:func:`dx_tile_rule`, evaluated after the K-slice count), or -- on an architecture listed in
#   DX_WIDE_MIN_EFF -- a one-slice launch below RASTER_RULE_MIN_HIDDEN whose wide items fill the SM pairs' waves below that
#   floor (the badly quantized chunk tails); the slice count (the reduction form :data:`DX_RED_STAGE` with ``k_slices``
#   2 .. DX_RED_SLICES_MAX) comes from :func:`dx_slices`.  The same per-element K order on both tiles (the K-sliced
#   reduction's slab sum adds the slices in slice order on either).
# * dX operand ring (``_st3``): below RASTER_RULE_MIN_HIDDEN a chunk of >= DW_LONG_CHUNK_MIN_ROWS rows runs the 512-wide dX
#   tile with the 3-deep operand ring (:func:`dx_stages_variant`, DX_LONG_CHUNK_STAGES per architecture); the 256-wide
#   fallback keeps its depth.  Bitwise: the ring depth only changes when a stage is refilled.
# * dW 2-D blocked raster (``_gn`` with ``group_n``): below RASTER_RULE_MIN_HIDDEN a chunk of DW_BLOCK_MIN_ROWS ..
#   DW_BLOCK_MAX_ROWS rows runs the weight-gradient accumulate and the fp32 fused cast with column blocks of
#   DW_BLOCK_GROUPS[arch] tiles (:func:`dw_block_variant`; the bf16 cast keeps the 1-D raster).  Bitwise: a permutation
#   of the same tiles.
# ``gemm_tuning`` (a keyword of :func:`prepare_lm_head_loss` and of the eager / autograd entry points; ``None`` = no
# explicit knob) pins knobs explicitly per GEMM -- ``{"logits": {"group_m": 16}, "dx": {"tile_n": 256, "k_slices": 2},
# "dw": {"group_m": 32, "group_n": 0}}``; a knob given as ``None`` pins the default -- and wins over the rule for the
# knobs it names.  A form the record does not register fails closed (``NotImplementedError``).
RASTER_RULE_MIN_HIDDEN = 7168  # first H whose [rows_c, H] BF16 operand panel no longer stays L2-resident at rows_c >= 2049
RASTER_RULE_MIN_ROWS = 2049
RASTER_WIDE_GROUPS = {
    "sm_100a": (16, 32),
    "sm_103a": (16, 32),
    "sm_107a": (16, 32),
}  # (logits group_m, weight-gradient group_m) of the wide geometries, per architecture
DW_LONG_CHUNK_GROUP_M = 16  # weight-gradient group_m of the long chunks (>= DW_LONG_CHUNK_MIN_ROWS rows) below RASTER_RULE_MIN_HIDDEN
DW_LONG_CHUNK_GROUPS = {
    "sm_100a": DW_LONG_CHUNK_GROUP_M,
    "sm_103a": DW_LONG_CHUNK_GROUP_M,
    "sm_107a": DW_LONG_CHUNK_GROUP_M,
}  # the same height on every supported architecture (the kernel source keeps a per-architecture table; the export mirrors it)
DW_LONG_CHUNK_MIN_ROWS = 4097
LOGITS_LONG_RASTER_GROUPS = {
    "sm_100a": 16,
    "sm_103a": 16,
    "sm_107a": 32,
}  # logits group_m of the GLM-class chunks of >= LOGITS_LONG_RASTER_MIN_ROWS rows below RASTER_RULE_MIN_HIDDEN, per architecture
# (sm_107a: the logits GEMM's default height, as the deterministic package's sm_107a programs run it; R200 timing in the
# README's sm_107a record notes)
LOGITS_LONG_RASTER_MIN_ROWS = (
    3841  # 30 row tiles of 128 + 1: the first chunk size with 31 row tiles
)
# sm_107a is not listed: its long chunks keep the table ring depth (DX_WIDE_TILE_STAGES).  Timed on FlashInfer main
# f2a5262c (#8612) with the sm_107a table entries patched in at run time (PyTorch 2.14.1+cu130, nvcc 13.4), CUPTI
# per-launch cycles at z = 0, one run on each of three 212-SM R200 nodes, T 16231, C 8192, cross-entropy and plain /
# temperature log-prob: the dX GEMM launches of a call took 18853-19014 kcycles in total on the 3-deep ring and
# 17851-18145 kcycles on the 4-deep ring (the split-tail dX form the shipped tables select).
DX_LONG_CHUNK_STAGES = {
    "sm_100a": 3,
    "sm_103a": 3,
}  # operand-ring depth of the 512-wide dX tile at the long chunks (>= DW_LONG_CHUNK_MIN_ROWS rows) below RASTER_RULE_MIN_HIDDEN
DX_WIDE_TILE_STAGES = 4  # the 512-wide dX tile's table ring depth (an explicit ``stages`` of 4 pins the default form)
DW_BLOCK_GROUPS = {
    "sm_100a": 12,
    "sm_103a": 12,
    "sm_107a": 12,
}  # column tiles per raster block of the weight-gradient accumulate / fp32 fused cast at the C 4096 chunk, per architecture
DW_BLOCK_MIN_ROWS = RASTER_RULE_MIN_ROWS  # 2049
DW_BLOCK_MAX_ROWS = (
    DW_LONG_CHUNK_MIN_ROWS - 1
)  # 4096: above, the long-chunk raster owns the instance
DW_BLOCK_BASES = (
    "gemm_dw_acc",
    "gemm_dw_cast_f32",
)  # the fp32 output streams of the weight-gradient GEMM
DW_ITEM_COLS = 2  # column tiles of the A-sharing weight-gradient work item (a block is a multiple of them)
DX_TILE_WIDE = 512
DX_TILE_NARROW = 256
DX_WIDE_MIN_EFF = {
    "sm_100a": 0.83,
}  # wave-efficiency floor of a one-slice 512-wide dX launch below RASTER_RULE_MIN_HIDDEN, per architecture; below it the
#    256-wide cluster form runs the chunk (absent = the wide tile stays)
DX_K_SLICES_MAX = 4  # the deterministic slice rule's largest count (:func:`recommended_k_slices`, the ``"det"`` policy)
# Mixed-width last wave of the wide dX tile (``_split``): on an architecture listed here, a chunk whose wide work items
# leave a last wave of ``r`` items on ``P`` SM pairs with ``0 < r <= P / 2`` runs its last ``F = r`` wide items as ``2 F``
# 256-wide half items in the same launch (the grid carries ``W + F`` clusters, :func:`dx_split_rule`).  A chunk of whole
# 512-token tiles takes the rule only below RASTER_RULE_MIN_HIDDEN (:func:`dx_split_eligible`): in every call on an
# architecture in DX_SPLIT_WHOLE_ARCHES, and in a call of fewer than DW_STREAM_MIN_CHUNKS chunks on one in
# DX_SPLIT_WHOLE_FEW_CHUNKS_ARCHES -- there the weight-gradient accumulate shares the dX GEMM's stream
# (:func:`dw_side_stream`), so nothing else fills the last wave; in a longer call the side stream's accumulate does.
# Elsewhere a whole chunk keeps its full items.  Bitwise: the same K slices and per-element K order on both widths.
DX_SPLIT_ARCHES = (
    "sm_100a",
    "sm_103a",
    "sm_107a",
)
DX_SPLIT_LONG_RING_ARCHES = (
    "sm_103a",
    "sm_107a",
)  # architectures whose long-chunk ring (``_st3``) also takes the split (the deterministic package's list, plus sm_107a,
#    where only an explicit ``stages`` pin selects the ring)
DX_SPLIT_WHOLE_ARCHES: tuple[str, ...] = (
    "sm_103a",
    "sm_107a",
)  # whole-tile chunks below RASTER_RULE_MIN_HIDDEN split in every call
DX_SPLIT_WHOLE_FEW_CHUNKS_ARCHES: tuple[str, ...] = (
    "sm_100a",
)  # ... only in a call of fewer than DW_STREAM_MIN_CHUNKS chunks
DX_TOKEN_TILE = 512  # rows of a whole dX token tile (:func:`dx_whole_tiles`)
# Swapped dX tile (``_swab``): dX^T = W^T dz^T, W's 128 hidden columns of a CTA on the MMA's M and the chunk's
# 512-token tile on its N (two 256-token blocks), C stored per thread.  On an architecture listed here it runs a chunk
# of whole 512-token tiles of a cross-entropy loss call of >= DW_STREAM_MIN_CHUNKS chunks (:func:`dx_swab_rule`), ahead
# of the split-tail rule and of the H >= RASTER_RULE_MIN_HIDDEN base form.  Bitwise: the same bf16 products summed
# over the same K slices in the same K order.
DX_SWAB_ARCHES: tuple[str, ...] = ("sm_103a", "sm_107a")
_TUNING_KNOBS = {
    "logits": ("group_m",),
    "dx": ("k_slices", "tile_n", "stages", "split", "swap_ab"),
    "dw": ("group_m", "epi_store", "group_n"),
}
_STAGE_RE = re.compile(
    r"(?P<base>" + "|".join(sorted(BASE_STAGES, key=len, reverse=True)) + r")"
    r"(?P<mcnt>_mcnt)?(?:_tn(?P<tile>\d+))?(?:_st(?P<stages>\d+))?(?P<split>_split)?(?P<swab>_swab)?"
    r"(?P<gn>_gn)?(?P<tma>_tma)?(?P<lc>_lc)?(?P<dcnt>_dcnt)?(?P<tau>_tau)?"
)
STAGE_KNOBS = (
    "count",
    "tile_n",
    "stages",
    "split",
    "swab",
    "blocked",
    "epi_store",
    "live_cast",
    "dcnt",
    "temperature",
)
# The zero-gradient row skip of the log-probability backward (``_dcnt``, :func:`zero_dlogp_partition`): the backward
# orders each chunk's rows so that the rows whose upstream gradient ``dlogp`` is nonzero come first and holds their
# number per chunk in an int32 device cell (``CNT``); the ``_dcnt`` form of a GEMM bounds its work by that cell at the
# launch shape of the chunk's full row count (no read-back, one launch sequence, CUDA-graph capturable).  The dz arm
# and the K-sliced dX reduction skip the work items whose pair of row tiles starts at or past the count (their rows keep
# what they held: never read) and the reduction's slab sum stores the sum into the rows of the row-tile pairs that ran
# and ``-0.0`` into the rest (:data:`DX_SLAB_SUM_DCNT_STAGE`), the weight-gradient accumulate and bf16 fused cast run
# ``ceil(count / 64)`` of its K steps (one at least on a first chunk; none leaves the accumulator as it was, and the
# cast writes ``bf16(g * WS)``).
# The bases that have the form (not the swapped dX tile, not the accumulate's TMA epilogue; the bf16 fused cast's TMA
# store has it too, :data:`DW_CAST_TMA_DCNT_STAGE`; the slab sum is a row kernel):
DCNT_BASES = (
    "gemm_logits_dz",
    "gemm_dx_red",
    DX_SLAB_SUM_STAGE,
    "gemm_dw_acc",
    "gemm_dw_cast_bf16",
)
# The cast at the last live chunk (``_lc``, before ``_dcnt``; :attr:`Plan.live_cast`): under the skip the live rows of a
# backward's chunks are a prefix (:func:`zero_dlogp_partition`), so the counted weight-gradient accumulate of every
# chunk but the last reads the next chunk's count cell too (``CNT_NEXT``): with a live next chunk it adds its FP32 tile
# into the sum as the counted accumulate does; else it writes ``bf16(g * (sum + tile))`` (``bf16(g * tile)`` on a first
# chunk) into the bf16 gradient and leaves the sum, and the counted deferred cast of the last chunk (per-thread or TMA
# store) writes nothing when that chunk has no K step.  So exactly one program writes the gradient and a backward whose
# trailing chunks hold no live row runs no cast-only pass.  The bases that have the form (the accumulate on its 1-D or
# 2-D raster, not its TMA epilogue; the bf16 fused cast in both stores):
DW_LIVE_CAST_BASES = ("gemm_dw_acc", "gemm_dw_cast_bf16")


def stage_variant(
    base: str,
    *,
    count: bool = False,
    tile_n: Optional[int] = None,
    stages: Optional[int] = None,
    split: bool = False,
    swab: bool = False,
    blocked: bool = False,
    epi_store: Optional[str] = None,
    live_cast: bool = False,
    dcnt: bool = False,
) -> str:
    """Stage name of one structural form of ``base``: the rule outputs that select a kernel body, in the fixed suffix
    order ``_mcnt``, ``_tn256``, ``_st<stages>``, ``_split``, ``_swab``, ``_gn``, ``_tma``, ``_lc``, ``_dcnt``; ``split``: the wide
    dX tile with its last wave's trailing items run as half items (:func:`dx_split_rule`); ``swab``: the wide dX tile
    with swapped operands at the table ring depth (:func:`dx_swab_rule`); a default output (the host-bound row
    count, the wide tile, the table ring depth, the 1-D raster, the default epilogue) adds nothing, so
    ``stage_variant(base)`` is ``base`` itself.  The dX knobs apply to both dX bases (:data:`DX_BASES`: the
    one-slice plain store ``gemm_dx`` and the K-sliced reduction :data:`DX_RED_STAGE`).  ``count``: the logits GEMM
    whose valid-row bound is read from device memory (chunk 0 of the hidden valid-row count path); ``blocked``: the
    weight-gradient accumulate / fp32 fused cast on the 2-D blocked raster
    (``group_n`` set); ``epi_store="tma"``: the weight-gradient accumulate's TMA reduce-add or the bf16 fused cast's
    TMA store (:func:`dw_cast_tma_rule`); ``dcnt``: the form bounded by the device count of the chunk's live rows
    (:data:`DCNT_BASES`, the zero-gradient row skip; the bf16 fused cast's TMA store has one too,
    :data:`DW_CAST_TMA_DCNT_STAGE`); ``live_cast``: the cast at the last live chunk of a counted weight-gradient form
    (:data:`DW_LIVE_CAST_BASES`, with ``dcnt`` only).  The slice count, the raster height and the block width
    themselves are launch scalars (:func:`gemm_scalars`), not part of the name."""
    if base not in BASE_STAGES:
        raise ValueError(f"unknown base stage {base!r}")
    name = base
    if count:
        if base not in ("gemm_logits", "gemm_logits_nostats"):
            raise ValueError(
                f"{base}: the device-count form exists for the logits GEMMs only"
            )
        name += "_mcnt"
    if tile_n is not None and int(tile_n) != DX_TILE_WIDE:
        if base not in DX_BASES or int(tile_n) != DX_TILE_NARROW:
            raise ValueError(
                f"{base}: the only non-default dX tile is {DX_TILE_NARROW}, got {tile_n}"
            )
        name += f"_tn{DX_TILE_NARROW}"
    if stages is not None and int(stages) != DX_WIDE_TILE_STAGES:
        if (
            base not in DX_BASES
            or int(stages) not in set(DX_LONG_CHUNK_STAGES.values())
            or (tile_n is not None and int(tile_n) != DX_TILE_WIDE)
        ):
            raise ValueError(
                f"{base}: the only ring-depth form is the 512-wide dX tile's {sorted(set(DX_LONG_CHUNK_STAGES.values()))}-deep ring, got {stages} (tile {tile_n})"
            )
        name += f"_st{int(stages)}"
    if split or swab:
        if base not in DX_BASES or (tile_n is not None and int(tile_n) != DX_TILE_WIDE):
            raise ValueError(
                f"{base}: the split-tail and swapped forms exist for the wide dX tile only"
            )
        if split and swab:
            raise ValueError(
                f"{base}: the split-tail and swapped dX forms exclude each other"
            )
        if swab and stages is not None and int(stages) != DX_WIDE_TILE_STAGES:
            raise ValueError(
                f"{base}: the swapped dX tile runs the table ring depth only"
            )
        name += "_split" if split else "_swab"
    if blocked:
        if base not in DW_BLOCK_BASES:
            raise ValueError(
                f"{base}: the 2-D blocked raster exists for the weight-gradient accumulate and the fp32 fused cast only"
            )
        name += "_gn"
    if epi_store is not None and epi_store != "redsm":
        if base not in ("gemm_dw_acc", "gemm_dw_cast_bf16") or epi_store != "tma":
            raise ValueError(
                f"{base}: the only epilogue variants are the weight-gradient accumulate's TMA reduce-add and the bf16 fused cast's TMA store, got {epi_store!r}"
            )
        name += "_tma"
    if live_cast:
        if (
            not dcnt
            or base not in DW_LIVE_CAST_BASES
            or (name.endswith("_tma") and base != "gemm_dw_cast_bf16")
        ):
            raise ValueError(
                f"{base}: the cast at the last live chunk exists for the counted forms of {DW_LIVE_CAST_BASES} only "
                "(not the accumulate's TMA epilogue)"
            )
        name += "_lc"
    if dcnt:
        if (
            base not in DCNT_BASES
            or swab
            or (
                name.removesuffix("_lc").endswith("_tma")
                and base != "gemm_dw_cast_bf16"
            )
        ):
            raise ValueError(
                f"{base}: the device-count row bound exists for {DCNT_BASES} only (not the swapped or the accumulate's TMA-epilogue forms)"
            )
        name += "_dcnt"
    return name


def stage_form(base: str, *, temperature: bool = False, **knobs: Any) -> str:
    """Stage name of one structural form of ``base`` over every knob of :data:`STAGE_KNOBS` (the inverse of
    :func:`parse_stage`): :func:`stage_variant` of ``knobs``, then ``temperature``: the temperature form
    (:data:`TEMPERATURE_STAGES`, suffix ``_tau`` last) of the log-probability entry's logits GEMM arms, of the
    device-count logits GEMM with the statistics and of ``row_finalize`` -- of these only the ``dz`` arm in a ``_dcnt``
    form (:data:`LOGPROB_DZ_DCNT_TAU_STAGE`: under the zero-gradient row skip a call with a temperature runs it beside
    the counted dX / dW forms, :func:`make_plan_temperature`).  The default path names its stages with
    :func:`stage_variant` alone."""
    name = stage_variant(base, **knobs)
    if temperature:
        if (
            base not in TEMPERATURE_FORM_BASES
            or (base == "gemm_logits" and name != "gemm_logits_mcnt")
            or (knobs.get("dcnt") and base != LOGPROB_DZ_STAGE)
        ):
            raise ValueError(
                f"{base}: the temperature form exists for gemm_logits_lbl, gemm_logits_dz (also counted, _dcnt), the "
                f"device-count gemm_logits (count=True) and row_finalize only, got {name!r}"
            )
        name += "_tau"
    return name


@functools.lru_cache(maxsize=1024)
def _parse_stage(stage: str) -> tuple[str, tuple[tuple[str, Any], ...]]:
    """:func:`parse_stage` with the knobs as an immutable tuple: the stage names of the records are a fixed set, so the
    regular expression and the :func:`stage_form` round trip run once per name (bounded; a failed parse is not
    remembered)."""
    m = _STAGE_RE.fullmatch(stage)
    if m is None:
        raise ValueError(f"not a stage name of this backend: {stage!r}")
    knobs: dict[str, Any] = dict(
        count=bool(m["mcnt"]),
        tile_n=int(m["tile"]) if m["tile"] else None,
        stages=int(m["stages"]) if m["stages"] else None,
        split=bool(m["split"]),
        swab=bool(m["swab"]),
        blocked=bool(m["gn"]),
        epi_store="tma" if m["tma"] else None,
        live_cast=bool(m["lc"]),
        dcnt=bool(m["dcnt"]),
        temperature=bool(m["tau"]),
    )
    if stage_form(m["base"], **knobs) != stage:
        raise ValueError(f"not a stage name of this backend: {stage!r}")
    return m["base"], tuple(knobs.items())


def parse_stage(stage: str) -> tuple[str, dict[str, Any]]:
    """``(base, {count, tile_n, stages, split, swab, blocked, epi_store, live_cast, dcnt, temperature})`` of a stage name (the inverse of
    :func:`stage_variant`; the knob keys are ``STAGE_KNOBS``)."""
    base, knobs = _parse_stage(stage)
    return base, dict(knobs)


def base_stage(stage: str) -> str:
    """The base stage a (variant) stage name instantiates."""
    return _parse_stage(stage)[0]


def gemm_op(stage: str) -> str:
    """``"logits"`` / ``"dx"`` / ``"dw"``: the GEMM a (variant) stage name instantiates (``ValueError`` for a row
    kernel)."""
    base = base_stage(stage)
    if base in LOGITS_BASES:
        return "logits"
    if base in DX_BASES:
        return "dx"
    if base == "gemm_dw_acc" or base in DW_CAST_STAGES:
        return "dw"
    raise ValueError(f"{stage!r} is not a GEMM stage")


def raster_variant(
    hidden: int, rows_c: int, arch: Optional[str]
) -> Optional[tuple[Optional[int], Optional[int]]]:
    """``(logits group_m, weight-gradient group_m)`` the raster rules select for a chunk -- ``None`` in a slot = that
    GEMM's default raster, ``None`` altogether = the default raster for both.  At ``H >= RASTER_RULE_MIN_HIDDEN`` a chunk
    of ``>= RASTER_RULE_MIN_ROWS`` rows takes ``RASTER_WIDE_GROUPS[arch]``; below that ``H`` a chunk of
    ``>= LOGITS_LONG_RASTER_MIN_ROWS`` rows takes the logits height ``LOGITS_LONG_RASTER_GROUPS[arch]`` and a chunk of
    ``>= DW_LONG_CHUNK_MIN_ROWS`` rows the weight-gradient height ``DW_LONG_CHUNK_GROUP_M``, on every supported
    architecture."""
    if arch is None:
        return None
    if int(hidden) >= RASTER_RULE_MIN_HIDDEN:
        groups = RASTER_WIDE_GROUPS.get(arch)
        if groups is None or int(rows_c) < RASTER_RULE_MIN_ROWS:
            return None
        return groups
    logits_group = LOGITS_LONG_RASTER_GROUPS.get(arch)
    if logits_group is None or int(rows_c) < LOGITS_LONG_RASTER_MIN_ROWS:
        logits_group = None
    dw_group = DW_LONG_CHUNK_GROUPS.get(arch)
    if dw_group is None or int(rows_c) < DW_LONG_CHUNK_MIN_ROWS:
        dw_group = None
    if logits_group is None and dw_group is None:
        return None
    return (logits_group, dw_group)


def dx_stages_variant(
    rows_c: int, hidden: int, tile_n: Optional[int], arch: Optional[str]
) -> Optional[int]:
    """The dX operand-ring depth the ring rule selects for a chunk (``None`` = the table depth): ``DX_LONG_CHUNK_STAGES``
    of the architecture for the 512-wide tile at ``>= DW_LONG_CHUNK_MIN_ROWS`` rows below ``RASTER_RULE_MIN_HIDDEN``."""
    st = DX_LONG_CHUNK_STAGES.get(arch) if arch is not None else None
    if (
        st is None
        or int(hidden) >= RASTER_RULE_MIN_HIDDEN
        or int(rows_c) < DW_LONG_CHUNK_MIN_ROWS
    ):
        return None
    if tile_n is not None and int(tile_n) != DX_TILE_WIDE:
        return None
    return int(st)


def _dw_block_group(arch: Optional[str], hidden: Optional[int]) -> Optional[int]:
    """The weight-gradient block width the block rule can select for a record (``None`` = no block form): the
    architecture's ``DW_BLOCK_GROUPS`` entry below ``RASTER_RULE_MIN_HIDDEN`` when it tiles the ``H / 256`` column tiles
    with whole work items and is narrower than the row of items."""
    gn = DW_BLOCK_GROUPS.get(arch) if arch is not None else None
    if gn is None or hidden is None or int(hidden) >= RASTER_RULE_MIN_HIDDEN:
        return None
    n_tiles = int(hidden) // DX_TILE_NARROW
    if gn % DW_ITEM_COLS or n_tiles % gn or gn >= n_tiles:
        return None
    return int(gn)


def dw_block_variant(rows_c: int, hidden: int, arch: Optional[str]) -> Optional[int]:
    """The weight-gradient block width the block rule selects for a chunk (``None`` = the 1-D raster):
    :func:`_dw_block_group` at ``DW_BLOCK_MIN_ROWS .. DW_BLOCK_MAX_ROWS`` rows."""
    gn = _dw_block_group(arch, hidden)
    if gn is None or not (DW_BLOCK_MIN_ROWS <= int(rows_c) <= DW_BLOCK_MAX_ROWS):
        return None
    return gn


def dx_tile_rule(
    rows_c: int,
    hidden: int,
    num_sms: int,
    k_slices: int,
    geometry: "Geometry",
    arch: Optional[str] = None,
) -> int:
    """Column width of the dX GEMM's pair tile for a chunk: ``DX_TILE_WIDE`` unless ``H % 512 != 0``, or the launch's
    wide work items -- ``(row pairs) x (H / 512) x k_slices`` -- are fewer than the device's SM pairs, or (on an
    architecture listed in ``DX_WIDE_MIN_EFF``) a one-slice launch below ``RASTER_RULE_MIN_HIDDEN`` fills the SM pairs'
    waves below that floor -- each takes the ``DX_TILE_NARROW`` form (the K-slice count is decided first, on the narrow
    form's inputs, so both tiles add the same products: at one slice ``dX`` is bitwise the same on both tiles, at more
    the same FP32 sum in slice order).  Deterministic in the shapes, the slice count, the SM
    count and the architecture."""
    if int(hidden) % DX_TILE_WIDE:
        return DX_TILE_NARROW
    ctas = int(geometry.cta_group)
    pairs = geometry.row_tiles(rows_c, ctas) // ctas
    items = pairs * (int(hidden) // DX_TILE_WIDE) * max(1, int(k_slices))
    device_pairs = max(1, int(num_sms) // ctas)
    if items < device_pairs:
        return DX_TILE_NARROW
    floor = DX_WIDE_MIN_EFF.get(arch) if arch is not None else None
    if (
        floor is not None
        and int(k_slices) == 1
        and int(hidden) < RASTER_RULE_MIN_HIDDEN
        and items / (-(-items // device_pairs) * device_pairs) < floor
    ):
        return DX_TILE_NARROW
    return DX_TILE_WIDE


def dx_split_rule(
    rows_c: int,
    hidden: int,
    num_sms: int,
    k_slices: int,
    geometry: "Geometry",
    arch: Optional[str],
    tile_n: Optional[int],
    stages: Optional[int] = None,
) -> int:
    """Wide dX items of a chunk run as half items (``F``, 0 = none): on an architecture in ``DX_SPLIT_ARCHES``, the
    wide tile's ``W = (row pairs) x (H / 512) x k_slices`` items leave ``r = W mod P`` items in the last wave of the
    ``P`` SM pairs; ``0 < r <= P / 2`` splits those ``r`` (the half items fill the wave's idle pairs).  A chunk on the
    long-chunk ring (``stages``) splits only on an architecture in ``DX_SPLIT_LONG_RING_ARCHES``.  Deterministic in the
    shapes, the slice count, the SM count and the architecture."""
    if arch not in DX_SPLIT_ARCHES or (
        tile_n is not None and int(tile_n) != DX_TILE_WIDE
    ):
        return 0
    if (
        stages is not None
        and int(stages) != DX_WIDE_TILE_STAGES
        and arch not in DX_SPLIT_LONG_RING_ARCHES
    ):
        return 0
    wide, pairs = dx_wide_items(rows_c, hidden, num_sms, k_slices, geometry)
    r = wide % pairs
    return r if 0 < r <= pairs // 2 else 0


def dx_wide_items(
    rows_c: int, hidden: int, num_sms: int, k_slices: int, geometry: "Geometry"
) -> tuple[int, int]:
    """The wide dX tile's work items ``W = (row pairs) x (H / 512) x k_slices`` of a chunk and the SM pairs ``P``."""
    ctas = int(geometry.cta_group)
    wide = (
        (geometry.row_tiles(rows_c, ctas) // ctas)
        * (int(hidden) // DX_TILE_WIDE)
        * max(1, int(k_slices))
    )
    return wide, max(1, int(num_sms) // ctas)


def dx_whole_tiles(rows_c: int, hidden: int, tile_n: Optional[int]) -> bool:
    """The chunk runs the wide tile and its rows are a whole number of ``DX_TOKEN_TILE`` token tiles."""
    if (tile_n is not None and int(tile_n) != DX_TILE_WIDE) or int(
        hidden
    ) % DX_TILE_WIDE:
        return False
    return int(rows_c) > 0 and int(rows_c) % DX_TOKEN_TILE == 0


def dx_split_eligible(
    rows_c: int,
    hidden: int,
    arch: Optional[str],
    tile_n: Optional[int],
    num_chunks: Optional[int] = None,
) -> bool:
    """The chunk may take the split-tail rule (:func:`dx_split_rule`): a chunk that is not a whole number of
    ``DX_TOKEN_TILE`` token tiles always; a chunk of whole token tiles only below ``RASTER_RULE_MIN_HIDDEN``, in every call
    on an architecture in ``DX_SPLIT_WHOLE_ARCHES`` and in a call of ``num_chunks < DW_STREAM_MIN_CHUNKS`` chunks on one
    in ``DX_SPLIT_WHOLE_FEW_CHUNKS_ARCHES`` (``None``: the call's chunk count is unknown -- no)."""
    if not dx_whole_tiles(rows_c, hidden, tile_n):
        return True
    if int(hidden) >= RASTER_RULE_MIN_HIDDEN:
        return False
    if arch in DX_SPLIT_WHOLE_ARCHES:
        return True
    return (
        arch in DX_SPLIT_WHOLE_FEW_CHUNKS_ARCHES
        and num_chunks is not None
        and int(num_chunks) < DW_STREAM_MIN_CHUNKS
    )


def dx_swab_rule(
    rows_c: int,
    hidden: int,
    arch: Optional[str],
    tile_n: Optional[int],
    ce_loss: bool = False,
    num_chunks: Optional[int] = None,
) -> bool:
    """The swapped dX tile runs a chunk of whole ``DX_TOKEN_TILE`` token tiles (wide tile) of a cross-entropy loss call
    (``ce_loss``: the loss entry point with objective ``"ce"``; the policy objective and the log-probability entry point
    keep the other forms) of ``num_chunks >= DW_STREAM_MIN_CHUNKS`` chunks on an architecture in ``DX_SWAB_ARCHES``."""
    return (
        arch in DX_SWAB_ARCHES
        and bool(ce_loss)
        and num_chunks is not None
        and int(num_chunks) >= DW_STREAM_MIN_CHUNKS
        and dx_whole_tiles(rows_c, hidden, tile_n)
    )


def gemm_tuning_default() -> dict[str, dict[str, Any]]:
    """Default of ``gemm_tuning`` (``{}``): no explicit GEMM knob, the per-chunk instance rules choose every form."""
    return {}


def _resolve_gemm_tuning(gemm_tuning) -> dict[str, dict[str, Any]]:
    """Validated copy of an explicit GEMM knob set (``None`` = :func:`gemm_tuning_default`)."""
    if gemm_tuning is None:
        return gemm_tuning_default()
    if not isinstance(gemm_tuning, dict):
        raise ValueError("gemm_tuning must be a dict {GEMM: {knob: value}}")
    out: dict[str, dict[str, Any]] = {}
    for op, knobs in gemm_tuning.items():
        if op not in _TUNING_KNOBS:
            raise ValueError(
                f"gemm_tuning: unknown GEMM {op!r}; one of {sorted(_TUNING_KNOBS)}"
            )
        if not isinstance(knobs, dict) or set(knobs) - set(_TUNING_KNOBS[op]):
            raise ValueError(
                f"gemm_tuning[{op!r}] must be a dict over the knobs {_TUNING_KNOBS[op]}, got {knobs!r}"
            )
        out[op] = {
            k: None if v is None else (str(v) if k == "epi_store" else int(v))
            for k, v in knobs.items()
        }
    return out


def _tuning_key(tuning: dict[str, dict[str, Any]]) -> tuple:
    if not tuning:  # the default (no explicit knob): every call's key
        return ()
    return tuple(
        (op, tuple(sorted(knobs.items()))) for op, knobs in sorted(tuning.items())
    )


@dataclass(frozen=True)
class ChunkVariants:
    """The instance knobs of one chunk's GEMMs as the rules (or the explicit knobs) resolved them; ``None`` = the
    default form of the base stage."""

    logits_group_m: Optional[int] = None
    dx_tile_n: Optional[int] = None  # None = DX_TILE_WIDE
    dx_stages: Optional[int] = None  # None = the table ring depth
    dx_split: Optional[int] = (
        None  # None = no split-tail; F = the wide items run as half items (dx_split_rule)
    )
    dx_swab: bool = False  # the swapped dX tile (dx_swab_rule)
    dw_group_m: Optional[int] = None
    dw_epi_store: Optional[str] = None
    dw_group_n: Optional[int] = None  # None = the 1-D raster
    dw_cast_tma: bool = (
        False  # the deferred bf16 cast's TMA epilogue (dw_cast_tma_rule; last chunk)
    )


def chunk_variants(
    rows_c: int,
    hidden: int,
    num_sms: int,
    k_slices: int,
    geometry: "Geometry",
    arch: Optional[str],
    tuning: Optional[dict[str, dict[str, Any]]] = None,
    num_chunks: Optional[int] = None,
    ce_loss: bool = False,
) -> ChunkVariants:
    """Per-chunk instance knobs: the explicit ``tuning`` knob when present (``None`` pins the default), else the rule
    of a registered program's architecture (``arch``; ``None`` -- the reference path -- plans the base stages).
    ``num_chunks`` is the call's chunk count (the whole-chunk split rule of :func:`dx_split_eligible`); ``ce_loss``: the
    call is a cross-entropy loss call (the swap rule :func:`dx_swab_rule`)."""
    tuning = tuning or {}
    logits_t, dx_t, dw_t = (tuning.get(op, {}) for op in ("logits", "dx", "dw"))
    raster = raster_variant(hidden, rows_c, arch)
    logits_g = (
        logits_t["group_m"]
        if "group_m" in logits_t
        else (raster[0] if raster else None)
    )
    dw_g = dw_t["group_m"] if "group_m" in dw_t else (raster[1] if raster else None)
    if "tile_n" in dx_t:
        tile = dx_t["tile_n"]
    else:
        tile = (
            dx_tile_rule(rows_c, hidden, num_sms, k_slices, geometry, arch)
            if arch is not None
            else None
        )
    if "stages" in dx_t:
        stages = dx_t["stages"]
    else:
        stages = (
            dx_stages_variant(rows_c, hidden, tile, arch) if arch is not None else None
        )
    # the swapped tile runs the table ring depth: an explicit ``stages`` other than it, like an explicit ``split``
    # (``None`` included), keeps the pinned form and turns the swap rule off; with ``swap_ab=True`` it is an error
    stages_off_table = (
        "stages" in dx_t
        and dx_t["stages"] is not None
        and int(dx_t["stages"]) != DX_WIDE_TILE_STAGES
    )
    if "swap_ab" in dx_t:
        # an explicit ``swap_ab=True`` applies to every chunk (as an explicit ``split`` does): a chunk whose resolved
        # form cannot run the swapped tile is an error here, not a missing stage at bind time
        swab = bool(dx_t["swap_ab"])
        if swab and arch is not None and arch not in DX_SWAB_ARCHES:
            raise ValueError(
                f"gemm_tuning dx swap_ab=True: {arch} has no swapped dX program (architectures "
                f"{', '.join(DX_SWAB_ARCHES)})"
            )
        if swab and stages_off_table:
            raise ValueError(
                f"gemm_tuning dx swap_ab=True: the swapped dX tile runs the {DX_WIDE_TILE_STAGES}-deep table ring "
                f"only, got stages={dx_t['stages']}"
            )
        if swab and tile is not None and int(tile) != DX_TILE_WIDE:
            how = (
                "pinned"
                if "tile_n" in dx_t
                else "selected by the tile rule (the wide items do not fill the SM pairs)"
            )
            hint = (
                f"H={int(hidden)} has no wide tile (H % {DX_TILE_WIDE} != 0)"
                if int(hidden) % DX_TILE_WIDE
                else f"pin tile_n={DX_TILE_WIDE} as well"
            )
            raise ValueError(
                f"gemm_tuning dx swap_ab=True: the swapped dX form exists for the {DX_TILE_WIDE}-column tile only, "
                f"but the chunk of {int(rows_c)} rows resolves tile_n={int(tile)} ({how}); {hint}"
            )
    else:
        swab = (
            arch is not None
            and "split" not in dx_t
            and not stages_off_table
            and dx_swab_rule(rows_c, hidden, arch, tile, ce_loss, num_chunks)
        )
    if "split" in dx_t:
        split = 0 if dx_t["split"] is None else int(dx_t["split"])
    elif swab:
        split = 0  # the swapped tile takes precedence over the split-tail rule
    elif arch is None or not dx_split_eligible(rows_c, hidden, arch, tile, num_chunks):
        split = 0  # a chunk of whole token tiles keeps its full items where its half items cost more at the power limit
    else:
        split = dx_split_rule(
            rows_c, hidden, num_sms, k_slices, geometry, arch, tile, stages
        )
    if swab:
        stages = None  # the swapped tile runs the table ring depth
    if split and swab:
        raise ValueError(
            "gemm_tuning: the dX split-tail and swapped forms exclude each other"
        )
    epi = dw_t.get(
        "epi_store"
    )  # no rule selects an epilogue form (the SM103 TMA reduce-add rule was retired): explicit pins only
    if "group_n" in dw_t:
        gn = dw_t["group_n"]
    else:
        gn = dw_block_variant(rows_c, hidden, arch) if arch is not None else None
    return ChunkVariants(
        logits_group_m=None if logits_g is None else int(logits_g),
        dx_tile_n=None if tile is None or int(tile) == DX_TILE_WIDE else int(tile),
        dx_stages=None
        if stages is None or int(stages) == DX_WIDE_TILE_STAGES
        else int(stages),
        dx_split=int(split) if split else None,
        dx_swab=bool(swab),
        dw_group_m=None if dw_g is None else int(dw_g),
        dw_epi_store=None if epi in (None, "redsm") else str(epi),
        dw_group_n=None if gn in (None, 0) else int(gn),
    )


def reachable_variants(bases, arch: str, dx_max: int) -> set[str]:
    """Every structural form the per-chunk rules can select on ``arch`` from the base GEMM stages ``bases`` for a record
    that serves ``dx_max`` dX slices (:func:`dx_max_slices`; above one: the K-sliced reduction's forms, the plain-store
    forms of a one-slice chunk always); a program is complete for an entry when it registers them.  The forms do not
    depend on the call geometry (``H`` / ``V`` and the rules' numeric outputs are launch scalars), so a record serves
    every admissible geometry with them."""
    st_long = DX_LONG_CHUNK_STAGES.get(arch)
    blocked = arch in DW_BLOCK_GROUPS
    out: set[str] = set()
    for base in bases:
        if base in ("gemm_logits", "gemm_logits_nostats"):
            # the device-count form of the hidden valid-row count path (chunk 0 at the buffer extent min(chunk, T))
            out.add(stage_variant(base, count=True))
        elif base == "gemm_dx":
            for dx_base in DX_BASES if int(dx_max) > 1 else ("gemm_dx",):
                for tile in (DX_TILE_WIDE, DX_TILE_NARROW):
                    out.add(stage_variant(dx_base, tile_n=tile))
                if st_long is not None:
                    out.add(stage_variant(dx_base, tile_n=DX_TILE_WIDE, stages=st_long))
                if arch in DX_SPLIT_ARCHES:
                    for st in (None, st_long):
                        out.add(stage_variant(dx_base, stages=st, split=True))
                if arch in DX_SWAB_ARCHES:
                    out.add(stage_variant(dx_base, swab=True))
        elif base in DW_BLOCK_BASES and blocked:
            out.add(stage_variant(base, blocked=True))
        elif base == "gemm_dw_cast_bf16" and arch in DW_CAST_TMA_MAX_ROWS:
            out.add(stage_variant(base, epi_store="tma"))
    return out


# Accepted spellings of the same host value (kernel side -> host side).
CONTRACT_ALIASES = {
    "num_rows": "T",
    "total_rows": "T",
    "hidden": "H",
    "vocab": "V",
    "chunk_size": "chunk",
    "num_vocab_tiles": "num_tiles",
    "vocab_tiles": "num_tiles",
    "objective": "mode",
    "grad_scale": "g",
    "scale": "g",
    "src": "acc",
    "dst": "out",
    "z_c": "z",
    "logits": "z",
    "dz": "z",
    "dz_c": "z",
    "dlogits": "z",
    "dlogp": "d_in",
    "y": "labels",
    "targets": "labels",
    "weights": "loss_weights",
    "w": "loss_weights",
}


@dataclass(frozen=True)
class Geometry:
    stats_tile: int
    row_tile: int
    cta_group: int
    k_block: int
    cast_vec: int
    vocab_multiple: int
    hidden_multiple: int
    ld_multiple: int
    labels_dtype: torch.dtype
    logits_cluster_ctas: int = 2
    dx_cluster_ctas: int = 2
    dw_cluster_ctas: int = 2
    logits_item_cols: int = 256
    dx_item_cols: int = 256
    dw_item_cols: int = 256
    logits_group_m: int = 32
    dx_group_m: int = 16
    dw_group_m: int = 8

    @classmethod
    def from_record(cls, record: Optional[dict[str, Any]]) -> "Geometry":
        raw = dict(GEOMETRY_DEFAULTS)
        if record is not None:
            declared = record.get("geometry", {})
            unknown = sorted(set(declared) - set(GEOMETRY_DEFAULTS))
            if unknown:
                raise ValueError(f"registry record: unknown geometry fields {unknown}")
            raw.update(declared)
        for name in GEOMETRY_DEFAULTS:
            if name == "labels_dtype" or (
                name in _GEOMETRY_OPTIONAL and raw[name] is None
            ):
                continue
            if isinstance(raw[name], bool) or int(raw[name]) < 1:
                raise ValueError(
                    f"registry record: geometry {name} must be a positive integer"
                )
        if raw["labels_dtype"] not in LABEL_DTYPES:
            raise ValueError(
                f"registry record: geometry labels_dtype must be one of {sorted(LABEL_DTYPES)}"
            )
        return cls(
            **{
                name: int(raw[name])
                for name in GEOMETRY_DEFAULTS
                if name != "labels_dtype" and name not in _GEOMETRY_OPTIONAL
            },
            labels_dtype=LABEL_DTYPES[raw["labels_dtype"]],
        )

    def row_tiles(self, rows: int, cluster_ctas: Optional[int] = None) -> int:
        """``ceil(rows / row_tile)`` rounded up to a multiple of the cluster's CTA count (default ``cta_group``: the
        pair takes adjacent row tiles; a wider cluster takes as many)."""
        ctas = self.cta_group if cluster_ctas is None else int(cluster_ctas)
        tiles = -(-int(rows) // self.row_tile)
        return -(-tiles // ctas) * ctas

    def cluster_ctas_of(self, stage: str) -> Optional[int]:
        """CTAs per cluster of a GEMM stage (``None`` for the row kernels and the casts)."""
        base = base_stage(stage) if stage in GEMM_STAGES else stage
        if base in LOGITS_BASES:
            return self.logits_cluster_ctas
        if base == "gemm_dx":
            return self.dx_cluster_ctas
        if base == "gemm_dw_acc" or base in DW_CAST_STAGES:
            return self.dw_cluster_ctas
        return None

    def k_iters(self, rows_c: int) -> int:
        return -(-int(rows_c) // self.k_block)


class _TemperatureGeometry(Geometry):
    """The geometry a call with a temperature binds with: a temperature GEMM form declares the clusters of its base
    stage (:meth:`Geometry.cluster_ctas_of` reads :data:`GEMM_STAGES`, which lists no temperature form)."""

    def cluster_ctas_of(self, stage: str) -> Optional[int]:
        if stage in TEMPERATURE_GEMM_STAGES:
            stage = base_stage(stage)
        return super().cluster_ctas_of(stage)

    @classmethod
    def of(cls, geometry: Geometry) -> "_TemperatureGeometry":
        return cls(**{f.name: getattr(geometry, f.name) for f in fields(geometry)})


DEFAULT_GEOMETRY = Geometry.from_record(None)


# ---------------------------------------------------------------------------
# Device / registry queries
# ---------------------------------------------------------------------------


def arch_for(device: Optional[torch.device] = None) -> Optional[str]:
    """Architecture tag of ``device`` (``None`` when unsupported or without CUDA)."""
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        return None
    return SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))


def record_for(device: Optional[torch.device] = None) -> tuple[str, dict[str, Any]]:
    """``(module_name, record)`` registered for ``device`` (the architecture's sole record, see
    :func:`cake_jit.select_module`); raises when absent."""
    arch = arch_for(device)
    if arch is None:
        raise ValueError(
            "the chunked LM-head + loss kernels require compute capability 10.0, 10.3 or 10.7"
        )
    name = select_module(arch)
    return name, MODULES[name]


def record_abi(record: dict[str, Any]) -> str:
    abi = str(record.get("abi", ABI_CONTRACT))
    if abi not in SUPPORTED_ABIS:
        raise NotImplementedError(f"unsupported host binding profile {abi!r}")
    return abi


def dx_stage(
    k_slices: int,
    tile_n: Optional[int] = None,
    stages: Optional[int] = None,
    split: bool = False,
    swab: bool = False,
    dcnt: bool = False,
) -> str:
    """Stage name of the dX GEMM run as ``k_slices`` K-slice work items per output tile: the plain-store ``gemm_dx``
    forms for one slice, the K-sliced reduction :data:`DX_RED_STAGE` forms above one; ``tile_n`` = the pair tile's
    column width, ``None`` / ``DX_TILE_WIDE`` = the wide default; ``stages`` = the operand-ring depth, ``None`` = the
    table depth; ``dcnt``: the reduction form bounded by the device count of live rows (the plain store has none)."""
    if not 1 <= int(k_slices) <= DX_RED_SLICES_MAX:
        raise ValueError(
            f"the dX GEMM runs 1 .. {DX_RED_SLICES_MAX} K slices per tile, got {k_slices}"
        )
    return stage_variant(
        DX_RED_STAGE if int(k_slices) > 1 else "gemm_dx",
        tile_n=tile_n,
        stages=stages,
        split=split,
        swab=swab,
        dcnt=bool(dcnt) and int(k_slices) > 1,
    )


def dx_max_slices(stages) -> int:
    """Largest slice count the registered stages serve: :data:`DX_RED_SLICES_MAX` when the K-sliced reduction form
    :data:`DX_RED_STAGE` is registered (the count is a launch scalar of that one form), else ``1``."""
    return DX_RED_SLICES_MAX if DX_RED_STAGE in stages else 1


def gemm_scalars(
    stage: str,
    geometry: "Geometry",
    *,
    hidden: int,
    vocab: int,
    rows_c: int,
    k_slices: int = 1,
    group_m: Optional[int] = None,
    group_n: Optional[int] = None,
    first_chunk: int = 1,
    ws_slab: int = 0,
) -> dict[str, int]:
    """The launch scalars ``GEMM_SCALARS`` of one GEMM stage from the record geometry, the call geometry and the rules'
    outputs: ``group_m`` (row tiles; ``None`` = the GEMM's default height ``<gemm>_group_m``), ``group_n`` (the
    weight-gradient block width in column TILES, :func:`dw_block_variant`; ``None`` = the 1-D raster), ``k_slices`` of
    the dX GEMM (``k_slice_iters`` = ``ceil(k_iters / k_slices)``, every slice at least one K step: see
    :func:`dx_slices_fit`).  The logits GEMM always stores (``first_chunk`` 0 = no accumulate), the dX GEMM always
    stores (plain form) or adds (reduction form) its chunk's rows (1); ``first_chunk`` / ``ws_slab`` are taken from the
    arguments for the weight-gradient / dX GEMM only.  The form of ``stage`` must agree with the scalars (one slice for
    the plain-store dX form, the ``_gn`` form with a ``group_n``).  A pure function of its arguments, remembered per argument tuple (bounded; a rejected tuple is not
    remembered); every call returns a fresh dict."""
    return dict(
        _gemm_scalars(
            stage,
            geometry,
            hidden,
            vocab,
            rows_c,
            k_slices,
            group_m,
            group_n,
            first_chunk,
            ws_slab,
        )
    )


@functools.lru_cache(maxsize=4096)
def _gemm_scalars(
    stage: str,
    geometry: "Geometry",
    hidden: int,
    vocab: int,
    rows_c: int,
    k_slices: int,
    group_m: Optional[int],
    group_n: Optional[int],
    first_chunk: int,
    ws_slab: int,
) -> dict[str, int]:
    base, knobs = parse_stage(stage)
    op = gemm_op(stage)
    hidden, vocab, rows_c = int(hidden), int(vocab), int(rows_c)
    item_rows = int(getattr(geometry, f"{op}_cluster_ctas"))
    item_cols = int(getattr(geometry, f"{op}_item_cols"))
    ldc = vocab if op == "logits" else hidden
    if ldc % item_cols:
        raise ValueError(
            f"{stage}: N = {ldc} is not a multiple of the {item_cols}-column work item"
        )
    n_items = ldc // item_cols
    gm = int(getattr(geometry, f"{op}_group_m")) if group_m is None else int(group_m)
    if gm < 1:
        raise ValueError(f"{stage}: group_m must be positive, got {group_m}")
    if bool(knobs["blocked"]) != (group_n is not None):
        raise ValueError(
            f"{stage}: the 2-D blocked form takes a group_n and the 1-D raster none, got group_n={group_n}"
        )
    if group_n is None:
        gn = n_items
    else:
        if op != "dw" or int(group_n) < DW_ITEM_COLS or int(group_n) % DW_ITEM_COLS:
            raise ValueError(
                f"{stage}: group_n = {group_n} is not a positive multiple of the {DW_ITEM_COLS}-tile weight-gradient work item"
            )
        gn = int(group_n) // DW_ITEM_COLS
        if gn > n_items or n_items % gn:
            raise ValueError(
                f"{stage}: group_n = {group_n} does not divide the {n_items} column items of N = {ldc}"
            )
    ks = int(k_slices)
    if op == "dw":
        if ks != 1:
            raise ValueError(f"{stage}: the weight-gradient GEMM is not K-sliced")
        k_iters = geometry.k_iters(rows_c)
        return dict(
            M=vocab,
            m_tiles=geometry.row_tiles(vocab, item_rows),
            k_iters=k_iters,
            first_chunk=int(bool(first_chunk)),
            ws_slab=0,
            ldc=ldc,
            k_slices=1,
            k_slice_iters=k_iters,
            group_m=gm,
            group_n=gn,
        )
    k_total = hidden if op == "logits" else vocab
    k_block = int(geometry.k_block)
    if k_total % k_block:
        raise ValueError(
            f"{stage}: K = {k_total} is not a multiple of the K step {k_block}"
        )
    k_iters = k_total // k_block
    if op == "logits":
        if ks != 1:
            raise ValueError(f"{stage}: the logits GEMM is not K-sliced")
    elif base == DX_RED_STAGE:
        if not 1 <= ks <= DX_RED_SLICES_MAX:
            raise ValueError(
                f"{stage}: the K-sliced reduction runs 1 .. {DX_RED_SLICES_MAX} slices, got k_slices={k_slices}"
            )
    elif ks != 1:
        raise ValueError(
            f"{stage}: the plain-store dX GEMM runs one K slice (the K-sliced form is {DX_RED_STAGE}), "
            f"got k_slices={k_slices}"
        )
    if ks > 1 and dx_slices_fit(ks, k_iters) != ks:
        raise ValueError(
            f"{stage}: {ks} K slices of {-(-k_iters // ks)} K steps over {k_iters} K steps leave the last slice "
            "without a K step"
        )
    return dict(
        M=rows_c,
        m_tiles=geometry.row_tiles(rows_c, item_rows),
        k_iters=k_iters,
        first_chunk=0 if op == "logits" else 1,
        ws_slab=int(ws_slab) if op == "dx" else 0,
        ldc=ldc,
        k_slices=ks,
        k_slice_iters=-(-k_iters // ks) if ks > 1 else k_iters,
        group_m=gm,
        group_n=gn,
    )


def wave_efficiency(
    rows_c: int,
    hidden: int,
    num_sms: int,
    k_slices: int,
    geometry: "Geometry",
    resident: Optional[int] = None,
) -> float:
    """Fraction of the last persistent wave of the dX GEMM that carries work: ``items / (ceil(items / clusters) * clusters)``
    with ``clusters`` = the co-resident ``dx_cluster_ctas``-wide clusters (``resident``; ``None`` = one per SM group of
    that width, the rule of the two-CTA cluster)."""
    ctas = geometry.dx_cluster_ctas
    clusters = (
        max(1, int(num_sms) // ctas) if resident is None else max(1, int(resident))
    )
    items = (
        (geometry.row_tiles(rows_c, ctas) // ctas)
        * (int(hidden) // geometry.hidden_multiple)
        * int(k_slices)
    )
    return items / (-(-items // clusters) * clusters)


def recommended_k_slices(
    rows_c: int,
    hidden: int,
    num_sms: int,
    max_slices: int,
    geometry: "Geometry",
    resident: Optional[int] = None,
) -> int:
    """Slice count of the dX GEMM for a chunk of ``rows_c`` rows under the deterministic package's rule (the ``"det"``
    policy of :func:`dx_slices`): the ``S`` in ``[1, max_slices]`` with the best wave efficiency net of
    ``K_SLICE_PENALTY`` per extra slab.  Deterministic in the shapes, the SM count and the device's co-resident cluster
    count (:func:`cluster_resident`)."""
    best, best_score = (
        1,
        wave_efficiency(rows_c, hidden, num_sms, 1, geometry, resident),
    )
    for s in range(2, max(1, int(max_slices)) + 1):
        score = wave_efficiency(
            rows_c, hidden, num_sms, s, geometry, resident
        ) - K_SLICE_PENALTY * (s - 1)
        if score > best_score + 1e-9:
            best, best_score = s, score
    return best


def cluster_resident(
    device: Optional[torch.device], cluster_ctas: int, num_sms: int
) -> int:
    """Co-resident clusters of ``cluster_ctas`` CTAs at one CTA per SM on ``device``: the SM pairs for the two-CTA
    cluster (the source launcher's rule).  Every registered launch is one or two CTAs wide; a wider cluster is
    rejected.  The dX slice rule takes it as ``resident``."""
    cluster_ctas = int(cluster_ctas)
    if cluster_ctas <= 2:
        return max(1, int(num_sms) // max(1, cluster_ctas))
    raise ValueError(
        f"the co-resident count of {cluster_ctas}-CTA clusters is not supported: every registered launch is one or two CTAs wide"
    )


# The dX slice rule of this package (:func:`dx_slices`), one function with two policies, selected programmatically
# (:func:`set_dx_slice_policy`; no environment variable):
# * ``"det"`` (the default): the deterministic package's slice count (:func:`recommended_k_slices` over 1 ..
#   DX_K_SLICES_MAX, its 256-column wave-efficiency model and its per-slab penalty), so a chunk takes the K-sliced
#   reduction exactly where that package slices it and keeps the plain store where it does not;
# * ``"measured"``: the form-aware rule measured on B200 (sm_100a, H 6144, V 154880, cross-entropy loss): ``S``
#   among the measured slice counts :data:`DX_MEASURED_SLICE_CANDIDATES` (at most the program's) maximizing
#   ``dx_form_efficiency(S) - DX_MEASURED_SLICE_PENALTY * (S - 1)`` -- the wave efficiency of the dX form the per-chunk
#   rules launch at that ``S`` (:func:`chunk_variants`: tile, split tail, swapped tile), with the reduction's measured
#   cost per extra slice.
#   On the measured contexts it differs from ``"det"`` on the 4096-row chunks of calls of three chunks or more
#   (``S = 3`` instead of 4).  The slice counts outside the measured set (5, 7) are not candidates: the model alone
#   would pick 5 on many rows, where no time was measured.
# Both are deterministic in the shapes, the SM count, the architecture and the call's chunk count; the resolved policy
# is part of every binding key, so switching it never reuses a binding planned under the other.
DX_SLICE_POLICIES = ("det", "measured")
# the reduction form's measured cost per extra K slice (fraction of the GEMM) and the slice counts the B200 measurement
# timed
DX_MEASURED_SLICE_PENALTY = 0.005
DX_MEASURED_SLICE_CANDIDATES = (1, 2, 3, 4, 6, 8)
_DX_SLICE_POLICY = {"policy": "det"}


def dx_slice_policy() -> str:
    """The dX slice rule's policy the planners use (:data:`DX_SLICE_POLICIES`; default ``"det"``)."""
    return _DX_SLICE_POLICY["policy"]


def set_dx_slice_policy(policy: str) -> str:
    """Select the dX slice rule's policy for the following calls of this process (``"det"`` or ``"measured"``) and
    return the previous one.  Calls planned under another policy keep their bindings (the policy is part of each
    binding key); a prepared runner keeps the slices it was planned with."""
    if policy not in DX_SLICE_POLICIES:
        raise ValueError(
            f"dx slice policy must be one of {DX_SLICE_POLICIES}, got {policy!r}"
        )
    previous = _DX_SLICE_POLICY["policy"]
    _DX_SLICE_POLICY["policy"] = str(policy)
    return previous


def _resolve_dx_slice_policy(policy: Optional[str]) -> str:
    if policy is None:
        return dx_slice_policy()
    if policy not in DX_SLICE_POLICIES:
        raise ValueError(
            f"dx slice policy must be one of {DX_SLICE_POLICIES}, got {policy!r}"
        )
    return str(policy)


# The smallest chunk that takes the K-sliced reduction (rows; :func:`set_dx_red_min_rows`, no environment variable;
# default 0 = every chunk follows the slice rule): a chunk of fewer rows keeps one slice -- the deterministic package's
# plain-store ``gemm_dx`` (its bits, no slab) -- for small calls whose slab and slab sum outweigh the waves the slices
# fill.  An explicit ``k_slices`` knob wins over it.  Default 0, set from A/B timing measured during
# development on B200 (sm_100a, CUDA 13.4 toolkit): log-probability and ce rows at T = 1 / 32 / 256 / 2048; forward,
# backward, step (forward + backward) and checkpointed step (two forwards + backward); each call after a GPU delay as
# long as one back-to-back call, after a 2 ms delay and back to back (96 cells); two independent runs, each interleaving
# the settings; pooled median time ratios over 0 with a 95 % CI.  1024 has its CI above 1 in 50 of 96 cells and 4096 in
# 68 (up to 1.51 on the T = 1 ce forward: one slice leaves the small call's dX GEMM without K parallelism), below 1 in 8
# and 5.  Five of 1024's cells below 1 are the T = 32 / 256 ce step (0.978 to 0.986), where the same setting's forward
# reads 1.31 to 1.46 and its checkpointed step 1.04 to 1.05; with 0, none of those steps has its CI above 1 over the
# deterministic package, so 0 stays the default at every call size.
_DX_RED_MIN_ROWS = {"rows": 0}


def dx_red_min_rows() -> int:
    """The smallest chunk (rows) that takes the K-sliced reduction (:func:`set_dx_red_min_rows`; default 0)."""
    return _DX_RED_MIN_ROWS["rows"]


def set_dx_red_min_rows(rows: int) -> int:
    """Set the smallest chunk (rows) that takes the K-sliced reduction for the following calls of this process (``0``:
    every chunk follows the slice rule) and return the previous value.  Like the policy it is part of every binding
    key; a prepared runner keeps the slices it was planned with."""
    if isinstance(rows, bool) or int(rows) != rows or int(rows) < 0:
        raise ValueError(
            f"dx red min rows must be a non-negative integer, got {rows!r}"
        )
    previous = _DX_RED_MIN_ROWS["rows"]
    _DX_RED_MIN_ROWS["rows"] = int(rows)
    return previous


def dx_slice_rule() -> tuple[str, int]:
    """The dX slice rule's settings ``(policy, red_min_rows)`` (:func:`dx_slice_policy`, :func:`dx_red_min_rows`).  An
    entry call reads them once and gives the same pair to its binding key and its plan, so a switch made by another
    thread in between never files a binding under settings it was not planned with."""
    return _DX_SLICE_POLICY["policy"], _DX_RED_MIN_ROWS["rows"]


def _resolve_dx_rule(dx_rule: Optional[tuple]) -> tuple[str, int]:
    if dx_rule is None:
        return dx_slice_rule()
    policy, red_min_rows = dx_rule
    return _resolve_dx_slice_policy(policy), max(0, int(red_min_rows))


def dx_slices_fit(k_slices: int, k_iters: int) -> int:
    """The slice count that runs ``k_slices`` requested slices over ``k_iters`` K steps with every slice holding at least
    one step (the reduction form's launch contract ``k_slice_iters * (k_slices - 1) < k_iters`` with ``k_slice_iters =
    ceil(k_iters / k_slices)``): ``ceil(k_iters / ceil(k_iters / k_slices))`` -- ``k_slices`` itself unless the
    ceil-sized slices cover ``K`` with fewer of them (e.g. 3 requested over 4 steps run as 2)."""
    ks, k = max(1, int(k_slices)), max(1, int(k_iters))
    ksi = -(-k // ks)
    return -(-k // ksi)


def dx_form_efficiency(
    rows_c: int,
    hidden: int,
    num_sms: int,
    k_slices: int,
    geometry: "Geometry",
    arch: Optional[str],
    *,
    num_chunks: Optional[int] = None,
    ce_loss: bool = False,
) -> float:
    """Wave efficiency of the dX form the per-chunk rules launch for a chunk at ``k_slices`` slices (the ``"measured"``
    policy of :func:`dx_slices`): ``items / (waves * slots)`` with the form of :func:`chunk_variants` -- the 512-wide
    pair tile (``(row pairs) x (H / 512) x S`` items on the SM pairs; a split tail runs its last ``F`` wide items as
    ``2 F`` half items, half a wave), the 256-wide A-sharing tile (the same items on the two-pair clusters) or the
    swapped tile (``(H / 256) x ceil(rows / 512) x S`` items on the SM pairs)."""
    v = chunk_variants(
        rows_c,
        hidden,
        num_sms,
        k_slices,
        geometry,
        arch,
        {},
        num_chunks=num_chunks,
        ce_loss=ce_loss,
    )
    ctas = int(geometry.dx_cluster_ctas)
    pairs = max(1, int(num_sms) // ctas)
    s = max(1, int(k_slices))
    if v.dx_swab:
        items = (int(hidden) // DX_TILE_NARROW) * -(-int(rows_c) // DX_TOKEN_TILE) * s
        return items / (-(-items // pairs) * pairs)
    items = (
        (geometry.row_tiles(rows_c, ctas) // ctas)
        * max(1, int(hidden) // DX_TILE_WIDE)
        * s
    )
    if v.dx_tile_n == DX_TILE_NARROW:
        quads = max(1, int(num_sms) // (2 * ctas))  # the A-sharing two-pair clusters
        return items / (-(-items // quads) * quads)
    if v.dx_split:
        return items / ((items // pairs + 0.5) * pairs)
    return items / (-(-items // pairs) * pairs)


def dx_slices(
    rows_c: int,
    hidden: int,
    vocab: int,
    num_sms: int,
    geometry: "Geometry",
    arch: Optional[str],
    *,
    max_slices: int,
    resident: Optional[int] = None,
    num_chunks: Optional[int] = None,
    ce_loss: bool = False,
    policy: Optional[str] = None,
) -> int:
    """The dX GEMM's K-slice count of a chunk of ``rows_c`` rows (1 = the plain store, above 1 = the K-sliced reduction
    :data:`DX_RED_STAGE`) under ``policy`` (``None`` = :func:`dx_slice_policy`; see :data:`DX_SLICE_POLICIES`), at most
    ``max_slices`` (:func:`dx_max_slices` of the record) and fitted to the K steps (:func:`dx_slices_fit`).
    ``num_chunks`` / ``ce_loss`` are the call's chunk count and loss kind (the form rules of :func:`chunk_variants`);
    the ``"measured"`` policy needs the architecture (``arch=None`` takes the ``"det"`` count).  The fit is applied to
    both policies: the largest count at most the rule's whose slices all hold a K step, e.g. at ``V = 256`` (four K
    steps) a rule count of 3, whose last slice would hold no K step, runs as 2 (the ``"det"`` policy: the deterministic
    rule's count, fitted)."""
    policy = _resolve_dx_slice_policy(policy)
    cap = max(1, int(max_slices))
    if policy == "measured" and arch is not None:
        best, best_score = 1, None
        for s in DX_MEASURED_SLICE_CANDIDATES:
            if s > cap:
                break
            score = dx_form_efficiency(
                rows_c,
                hidden,
                num_sms,
                s,
                geometry,
                arch,
                num_chunks=num_chunks,
                ce_loss=ce_loss,
            ) - DX_MEASURED_SLICE_PENALTY * (s - 1)
            if best_score is None or score > best_score + 1e-9:
                best, best_score = s, score
    else:
        best = recommended_k_slices(
            rows_c, hidden, num_sms, min(cap, DX_K_SLICES_MAX), geometry, resident
        )
    return dx_slices_fit(best, int(vocab) // int(geometry.k_block))


def plan_dx_slices(
    problem: "Problem",
    chunks: tuple[tuple[int, int], ...],
    *,
    need_dx: bool,
    tuning: dict[str, dict[str, Any]],
    dx_max_slices: int,
    num_sms: int,
    geometry: "Geometry",
    dx_resident: Optional[int],
    arch: Optional[str],
    policy: Optional[str] = None,
    red_min_rows: Optional[int] = None,
) -> tuple[int, ...]:
    """Per chunk, the dX GEMM's K-slice count (empty: one slice everywhere) -- an explicit ``k_slices`` knob of
    ``tuning`` (``None`` pins one slice; checked against ``dx_max_slices`` and the K steps), else :func:`dx_slices` when
    the record serves more than one slice, with one slice for a chunk of fewer than ``red_min_rows`` rows (``None`` =
    :func:`dx_red_min_rows`).  The planners of both backend modules take their slices from here."""
    dx_tuning = tuning.get("dx", {})
    if need_dx and "k_slices" in dx_tuning:
        # an explicit knob wins over the slice rule: a value pins that slice count, ``None`` pins the default form
        # (one slice, the plain-store ``gemm_dx`` stage) -- the same ``None`` semantics as every other pinned knob
        explicit_k = 1 if dx_tuning["k_slices"] is None else int(dx_tuning["k_slices"])
        if not 1 <= explicit_k <= int(dx_max_slices):
            raise ValueError(
                f"gemm_tuning dx k_slices={explicit_k}: the registered program serves 1 .. {int(dx_max_slices)} dX slices"
            )
        k_iters = int(problem.vocab) // int(geometry.k_block)
        if dx_slices_fit(explicit_k, k_iters) != explicit_k:
            raise ValueError(
                f"gemm_tuning dx k_slices={explicit_k}: V = {problem.vocab} has {k_iters} K steps, which "
                f"{explicit_k} slices of {-(-k_iters // explicit_k)} steps do not all reach"
            )
        return tuple(explicit_k for _ in chunks)
    if need_dx and int(dx_max_slices) > 1:
        min_rows = dx_red_min_rows() if red_min_rows is None else int(red_min_rows)
        return tuple(
            1
            if rows_c < min_rows
            else dx_slices(
                rows_c,
                problem.hidden,
                problem.vocab,
                num_sms,
                geometry,
                arch,
                max_slices=dx_max_slices,
                resident=dx_resident,
                num_chunks=len(chunks),
                ce_loss=problem.entry == "loss" and problem.objective == "ce",
                policy=policy,
            )
            for _, rows_c in chunks
        )
    return ()


# Under the zero-gradient row skip (:data:`DCNT_BASES`) the smallest dX slice count of a chunk, per architecture (an
# architecture not listed keeps the slice rule's counts).  The one-slice plain store has no counted form: it computes
# every row of its chunk, a chunk past the live rows included, while the K-sliced reduction's counted form skips the
# work items past the chunk's count.  So a counted plan runs its one-slice chunks as two-slice reductions
# (:func:`counted_dx_slices`).  sm_107a takes the rule with its 1946-row floor, the two timed together on R200
# (:data:`ZERO_DLOGP_MIN_ROWS_BY_ARCH`).
DX_COUNTED_MIN_SLICES_BY_ARCH: dict[str, int] = {
    "sm_100a": 2,
    "sm_103a": 2,
    "sm_107a": 2,
}


def counted_dx_slices(
    slices: tuple[int, ...],
    *,
    arch: Optional[str],
    dx_max_slices: int,
    k_iters: int,
    tuning: dict[str, dict[str, Any]],
) -> tuple[int, ...]:
    """The per-chunk dX slice counts of a plan under the zero-gradient row skip: ``slices`` (:func:`plan_dx_slices`)
    with every count below ``arch``'s :data:`DX_COUNTED_MIN_SLICES_BY_ARCH` entry raised to it.  ``slices`` itself
    when ``arch`` has no entry, when the record serves fewer slices (``dx_max_slices``), when that many slices do not
    all hold one of the ``k_iters`` K steps (:func:`dx_slices_fit`) or when ``tuning`` pins the dX slice count."""
    least = DX_COUNTED_MIN_SLICES_BY_ARCH.get(arch, 1) if arch is not None else 1
    if (
        least <= 1
        or not slices
        or "k_slices" in tuning.get("dx", {})
        or int(dx_max_slices) < least
        or dx_slices_fit(least, k_iters) != least
    ):
        return slices
    return tuple(max(int(s), least) for s in slices)


# --------------------------------------------------------------------------- valid-row compaction

# The switches below are keyword arguments of the entry points (``None`` = the documented default, the constant each
# ``*_default()`` returns); this module reads no environment variable.  FlashInfer's public facade
# (``flashinfer.cake_lm_head_nondet_train``) owns the A/B environment variables of these switches and passes the
# resolved values down.


def logprob_dz_default() -> bool:
    """Default of ``logprob_dz`` (``True``; the log-probability backward): each chunk's recompute is one
    ``gemm_logits_dz`` launch whose epilogue writes the row gradient ``dz`` in place of the logits where the record
    registers that arm; ``False`` (or a record without it) recomputes the logits with ``gemm_logits_nostats`` and forms
    ``dz`` with ``row_grad`` (bitwise the same ``dz``, hence the same ``dW`` and the same ``dX``; a chunk of more
    than one dX slice sums its slices in slice order on either path)."""
    return True


def logprob_lbl_default() -> bool:
    """Default of ``logprob_lbl`` (``True``; the log-probability forward): each chunk's logits GEMM is
    ``gemm_logits_lbl``, which writes the row statistics and the logit at the label column alone (the one element
    ``row_finalize`` reads) where the record registers that arm; ``False`` (or a record without it) writes the chunk's
    whole logits with ``gemm_logits`` (bitwise the same ``lse`` / ``logp``)."""
    return True


def _resolve_logprob_dz(logprob_dz) -> bool:
    return logprob_dz_default() if logprob_dz is None else bool(logprob_dz)


def _resolve_logprob_lbl(logprob_lbl) -> bool:
    return logprob_lbl_default() if logprob_lbl is None else bool(logprob_lbl)


def _logprob_arm(
    record: Optional[dict[str, Any]], stages: tuple, stage: str, switch: bool
) -> bool:
    """Whether a log-probability plan runs the logits GEMM arm ``stage`` (:data:`LOGPROB_DZ_STAGE` /
    :data:`LOGPROB_LBL_STAGE`): its resolved switch is on and a registered program registers the arm (otherwise the
    plan keeps the stages the arm replaces, bitwise the same outputs).  Resolved once per prepared binding."""
    return bool(switch) and record is not None and stage in stages


def _logprob_arm_temperature(
    record: Optional[dict[str, Any]],
    stages: tuple,
    stage: str,
    switch: bool,
    *,
    temperature: bool = False,
) -> bool:
    """Whether a log-probability plan runs the logits GEMM arm ``stage`` (:data:`LOGPROB_DZ_STAGE` /
    :data:`LOGPROB_LBL_STAGE`): its resolved switch is on and a registered program registers the arm (otherwise the
    plan keeps the stages the arm replaces, bitwise the same outputs).  Resolved once per prepared binding.

    ``temperature`` (the call has one; a registered program): the arm's temperature form is the only kernel that
    serves the call, so the switch must be on (``ValueError`` otherwise) and a record without that form raises
    ``NotImplementedError`` naming it -- never a silent fallback to the stages without the temperature."""
    if not temperature or record is None:
        return bool(switch) and record is not None and stage in stages
    if not switch:
        raise ValueError(
            f"a temperature runs the logits GEMM arm {stage!r}: its switch must stay on"
        )
    tau_stage = stage_form(stage, temperature=True)
    if tau_stage not in stages:
        raise NotImplementedError(
            f"the registered program {record.get('arch')!r} lacks the temperature stage {tau_stage!r} "
            f"(registered: {stages})"
        )
    return True


def compact_rows_default() -> bool:
    """Default of ``compact_rows`` (``True``): chunk over the valid rows only when some labels are ignored."""
    return True


def fuse_dw_cast_default() -> bool:
    """Default of ``fuse_dw_cast`` (``True``): the last chunk's weight-gradient GEMM runs in the backward with the upstream
    scale and the output cast fused into its epilogue (:data:`DW_CAST_STAGES`); ``False`` accumulates every chunk into
    the FP32 ``dW_acc`` and casts it with ``scale_cast`` (bitwise the same ``dW``)."""
    return True


def _resolve_fuse(fuse_dw_cast) -> bool:
    return fuse_dw_cast_default() if fuse_dw_cast is None else bool(fuse_dw_cast)


def dx_finalize_default() -> bool:
    """Default of ``dx_finalize`` (``True``): a compacted call's ``dX`` output is written by the
    ``scale_cast_scatter_bf16`` kernel (:data:`DX_FINALIZE_STAGES`); ``False`` casts, zero-fills and
    ``index_copy_``-scatters the compact rows (bitwise the same ``dX`` from the same accumulator).  Resolved once per
    forward; the backward follows the forward's choice through the saved valid-row mask."""
    return True


def _resolve_dx_finalize(dx_finalize) -> bool:
    return dx_finalize_default() if dx_finalize is None else bool(dx_finalize)


DW_STREAM_MODES = (
    "0",
    "1",
    "auto",
)  # the normalised ``dw_stream`` modes: never / every multi-chunk call / the chunk rule (:func:`dw_stream_mode`)
DW_STREAM_MIN_CHUNKS = 3  # the side stream pays one cross-stream join per chunk and overlaps only the chunks before a deferred last one
# ``torch.utils._python_dispatch._disable_current_modes`` (private; None on a torch without it): every active torch
# dispatch mode popped for a block -- the per-device / per-thread state below is formed under it, and an entry runs
# under it inside selective activation checkpointing (:func:`_selective_checkpoint_modes`)
_DISABLE_MODES = getattr(_torch_dispatch, "_disable_current_modes", None)


def _modes_off():
    """A context that pops every active torch dispatch mode for its block (a no-op without the private torch API)."""
    return contextlib.nullcontext() if _DISABLE_MODES is None else _DISABLE_MODES()


# the per-thread, per-device host state of the calls (:func:`_thread_state`)
_THREAD_STATE = threading.local()


def _thread_state(name: str) -> dict:
    """The calling thread's per-device dict ``name`` of the calls' host state (created empty on first use) --
    ``"dw_streams"`` (device index -> (dW side stream, fork event), :func:`_dw_stream`), ``"cells"`` /
    ``"pair_cells"`` (device index -> (pinned int32 [1] / [2] cell, event) of the hidden valid-row count,
    :func:`_hidden_count_cell` / :func:`_hidden_count_pair_cell`), ``"arange"`` (device index -> the growing int32
    constant 1, 2, .., :func:`_hidden_count_arange`) and ``"arange_streams"`` (device index -> the raw streams the
    constant was recorded on, :func:`_current_raw_stream`).  None is a tensor of a workspace or a binding.  A call
    records its fork event and waits on it, and writes a count cell and reads it back behind its event, before it
    returns, so the calls of one thread never share them in flight; calls of different threads on one device (each on
    its own stream) never share a stream, an event, a cell or the constant: one thread's call cannot re-record another's
    fork event, overwrite its count before the read, or reallocate the constant its queued kernels read; one thread
    runs one call at a time."""
    state = getattr(_THREAD_STATE, name, None)
    if state is None:
        state = {}
        setattr(_THREAD_STATE, name, state)
    return state


def dw_stream_mode(dw_stream: Optional[Union[bool, str]] = None) -> str:
    """The ``dw_stream`` keyword as its mode: ``"0"`` (never), ``"1"`` (every multi-chunk call) or ``"auto"`` (the
    default: calls of at least :data:`DW_STREAM_MIN_CHUNKS` chunks, :func:`dw_side_stream`).  ``None`` and ``"auto"``
    select the default, ``True`` / ``"1"`` every multi-chunk call, ``False`` / ``"0"`` never; any other value is a
    ``ValueError``.  A pure normaliser of the keyword (idempotent on a mode); nothing is read from the environment."""
    if dw_stream is None:
        return "auto"
    if isinstance(dw_stream, bool):
        return "1" if dw_stream else "0"
    if isinstance(dw_stream, str):
        value = dw_stream.strip().lower()
        if value in DW_STREAM_MODES:
            return value
    raise ValueError(
        f"dw_stream must be None, True, False or one of {DW_STREAM_MODES}, got {dw_stream!r}"
    )


def dw_side_stream(
    num_chunks: int, dw_stream: Optional[Union[bool, str]] = None
) -> bool:
    """Whether a call of ``num_chunks`` chunks under the ``dw_stream`` keyword (or its mode, :func:`dw_stream_mode`;
    ``None`` = the ``auto`` rule) launches each chunk's weight-gradient accumulate GEMM on the per-device
    side stream: forked right after the chunk's ``row_grad`` (``dz`` is ready; the accumulate depends on it alone, not
    on the dX GEMM launched next) and joined before the next chunk touches the chunk buffer and before the call returns,
    so the accumulate's CTAs fill the tail wave of the chunk's dX GEMM instead of queueing behind it (GLM-class 4096-row
    chunks: 768 dX work items on 76 CTA pairs = 11 waves at 0.919 fill).  The ``auto`` rule of
    :data:`DW_STREAM_MIN_CHUNKS` chunks is a design choice: the side stream adds one cross-stream join per chunk and
    overlaps only the chunks before a deferred last one, and a one-chunk call defers its only accumulate to the
    backward.  Kernels, buffers, launch order per kernel and numerics are unchanged (the dX GEMM and its slab sum write
    the dX rows, the accumulate reads the same ``dz`` and read-modify-writes ``dW_acc``; each kernel keeps its own
    order): only the launch stream differs, so ``loss`` / ``logp`` / ``dX`` / ``dW`` are bitwise the single-stream
    path's (the slab sum adds the dX slices in slice order on either).  The stream and its fork event are per thread
    and per device
    (:func:`_dw_stream`, :data:`_THREAD_STATE`): a call records the fork event and the caller's stream joins the side
    stream before the call returns, so the calls of one thread never share them in flight, and two threads never wait on
    each other's fork.  A thread creates them at its first multi-chunk call (:func:`.cake_backend_opt_in.cake_lm_head_nondet_train_preload`
    creates the calling thread's ahead of it); a caller that runs each call on a new thread pays that creation -- a
    stream and an event per device -- on every call."""
    n = int(num_chunks)
    if n <= 1:
        return False
    mode = dw_stream_mode(dw_stream)
    if mode == "0":
        return False
    if mode == "1":
        return True
    return n >= DW_STREAM_MIN_CHUNKS


def _dw_stream(index: int) -> tuple:
    """This thread's ``(side stream, fork event)`` of device ``index`` (created on first use; never part of a binding or
    a workspace)."""
    streams = _thread_state("dw_streams")
    pair = streams.get(index)
    if pair is None:
        pair = streams[index] = (
            torch.cuda.Stream(device=index),
            torch.cuda.Event(),
        )
    return pair


def hidden_count_default() -> bool:
    """Default of ``hidden_count`` (``True``; ``cake`` backend, compacted calls): the valid-row index is formed on the
    device and chunk 0's row gather (:data:`HIDDEN_COUNT_STAGE`) and logits GEMM (its ``_mcnt`` variant, the stores
    bounded by the count read from device memory) are launched before the count reaches the host through a pinned cell
    and a CUDA event (:func:`_hidden_count_begin`); ``False`` counts and indexes on the host before the first launch
    (:func:`valid_rows`; bitwise the same outputs).  Resolved once per call."""
    return True


def compaction_index(
    labels: torch.Tensor,
    *,
    valid: Optional[torch.Tensor] = None,
    pos: Optional[torch.Tensor] = None,
    idx_full: Optional[torch.Tensor] = None,
    ar: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The valid-row compaction on the device (no host synchronization): ``valid = labels >= 0``, ``pos`` = its
    inclusive int32 scan and ``idx_full[k] = searchsorted(pos, k + 1)`` -- the ascending original row of the ``k``-th
    valid row for ``k < count`` (``== valid.nonzero()``) and ``T`` for every slot after it, ``count = pos[T - 1]``.
    Pre-allocated outputs (``valid`` bool ``[T]``, ``pos`` int32 ``[T]``, ``idx_full`` int64 ``[T]``) and the int32
    constant ``ar = 1, 2, ..`` (``>= T`` entries; a cached prefix) keep the call at three kernels.  Returns
    ``(valid, pos, idx_full)``."""
    T = int(labels.numel())
    if ar is None:
        ar = torch.arange(1, T + 1, dtype=torch.int32, device=labels.device)
    valid = torch.ge(labels, 0, out=valid) if valid is not None else labels >= 0
    pos = (
        torch.cumsum(valid, 0, dtype=torch.int32, out=pos)
        if pos is not None
        else torch.cumsum(valid, 0, dtype=torch.int32)
    )
    idx_full = (
        torch.searchsorted(pos, ar[:T], out=idx_full)
        if idx_full is not None
        else torch.searchsorted(pos, ar[:T])
    )
    return valid, pos, idx_full


def _hidden_count_cell(index: int) -> tuple:
    """This thread's (pinned int32 ``[1]`` cell, event) of device ``index``, reused by its calls."""
    cells = _thread_state("cells")
    pair = cells.get(index)
    if pair is None:
        # thread state, not part of a call: formed outside active torch dispatch modes (see _device_constants)
        with _modes_off():
            pair = cells[index] = (
                torch.empty((1,), dtype=torch.int32, pin_memory=True),
                torch.cuda.Event(),
            )
    return pair


def _hidden_count_pair_cell(index: int) -> tuple:
    """This thread's (pinned int32 ``[2]`` cell, event) of device ``index``: the hidden count of a zero-weight split
    (two counts)."""
    cells = _thread_state("pair_cells")
    pair = cells.get(index)
    if pair is None:
        # thread state, not part of a call: formed outside active torch dispatch modes (see _device_constants)
        with _modes_off():
            pair = cells[index] = (
                torch.empty((2,), dtype=torch.int32, pin_memory=True),
                torch.cuda.Event(),
            )
    return pair


def _hidden_count_arange(index: int, T: int) -> torch.Tensor:
    """This thread's growing int32 constant ``1, 2, ...`` of device ``index`` with at least ``T`` entries (no kernel
    per call; grown by doubling), recorded on the current stream so that a replaced constant is not reused before the
    kernels queued on it have read it.  The allocator keeps every recorded stream until the constant is freed, so
    each stream is recorded once per constant (``"arange_streams"``: device index -> the raw streams the constant was
    recorded on, :func:`_current_raw_stream`), not once per call; a grown constant starts a new set.  Thread state,
    not part of a call: formed and recorded outside active torch dispatch modes (see :func:`_device_constants`; the
    backward's recomputation of a checkpoint runs on the autograd engine's thread)."""
    aranges = _thread_state("arange")
    recorded = _thread_state("arange_streams")
    ar = aranges.get(index)
    if ar is None or ar.numel() < T:
        cap = max(int(T), 2 * (0 if ar is None else int(ar.numel())))
        with _modes_off():
            ar = aranges[index] = torch.arange(
                1, cap + 1, dtype=torch.int32, device=torch.device("cuda", index)
            )
        recorded[index] = set()
    seen = recorded[index]
    raw = _current_raw_stream(index)
    if raw not in seen:
        with _modes_off():
            ar.record_stream(torch.cuda.current_stream(index))
        seen.add(raw)
    return ar


class _TemperatureCountState:
    """The temperature state of a :class:`_HiddenCount` (a call with a temperature,
    :func:`_hidden_count_begin_temperature`): ``tau_c`` the FP32 [T] compact temperature of the valid rows in order
    (``tau[idx_full[k]]``, clamped past the count), filled on the device before chunk 0's GEMM read it; the later
    chunks bind it too.  A plain base class: a typed class default (``None``), not a dataclass field, so the count of
    a call without a temperature is built as before; the temperature prologue sets it on its count."""

    tau_c: Optional[torch.Tensor] = None


@dataclass
class _HiddenCount(_TemperatureCountState):
    """One call's hidden valid-row count prologue: chunk 0's buffers (written before the count is known), the
    compaction outputs and the count once it was read back (:meth:`finish`)."""

    rows0: int  # the buffer extent min(chunk, T): rows of the chunk-0 buffers, M of the device-count GEMM
    stats: bool  # the logits GEMM form launched (True: with the row statistics)
    stage: str  # the device-count logits stage launched
    group_m: Optional[
        int
    ]  # the raster rule's height at the buffer extent (a launch scalar; None = the default)
    valid: torch.Tensor  # bool [T]
    idx_full: torch.Tensor  # int64 [T]: the valid rows in order, then T
    count_host: torch.Tensor  # the pinned int32 [1] cell the count is copied into
    event: Any  # the CUDA event recorded behind that copy
    logits: torch.Tensor = field(
        repr=False
    )  # bf16 [rows0, V]: chunk 0's z, rows < min(count, rows0) written
    stats_buf: torch.Tensor = field(repr=False)  # fp32 [rows0, num_tiles, 2]
    x_c: torch.Tensor = field(
        repr=False
    )  # bf16 [rows0, H]: chunk 0's gathered X rows, exact zeros after the count
    count: Optional[int] = None
    # the zero-weight split (:func:`gradient_index`; ``None`` without it): ``pos[T:]``, the inclusive scan of the
    # zero-weight mask offset by the gradient-row count, the arange constant and the zero-weight row count
    zero_pos: Optional[torch.Tensor] = field(default=None, repr=False)
    arange: Optional[torch.Tensor] = field(default=None, repr=False)
    zero_count: int = 0
    # a kept-logits forward's contiguous BF16 ``[T, V]`` kept buffer (``kept_rows`` of :func:`_hidden_count`):
    # ``logits`` is its first ``rows0`` rows, the later chunks' kept rows follow in row order (``None``: none)
    kept: Optional[torch.Tensor] = field(default=None, repr=False)

    def finish(self) -> int:
        """Wait for the count -- the compaction kernels and the 4-byte copy (8 bytes: both counts of a zero-weight
        split); chunk 0's GEMM is already queued -- and read it (once)."""
        if self.count is None:
            self.event.synchronize()
            self.count = int(self.count_host[0])
            if self.zero_pos is not None:
                self.zero_count = int(self.count_host[1]) - self.count
        return self.count

    def zero_index(self) -> Optional[torch.Tensor]:
        """The ascending int64 index of the zero-weight rows of a zero-weight split (``None``: no such row, or no
        split); formed on the device from the counts already read back (no synchronization)."""
        count = self.finish()
        if self.zero_pos is None or not self.zero_count:
            return None
        return torch.searchsorted(
            self.zero_pos, self.arange[count : count + self.zero_count]
        )

    @property
    def num_rows(self) -> int:
        return int(self.valid.numel())

    def row_index(self) -> Optional[torch.Tensor]:
        """The compact int64 row index: ``None`` when every row is valid, empty when every row is ignored."""
        count = self.finish()
        return None if count == self.num_rows else self.idx_full[:count]


def hidden_count_eligible(
    problem: "Problem",
    X: torch.Tensor,
    record: Optional[dict[str, Any]],
    module_name: Optional[str],
    *,
    stats: bool,
) -> bool:
    """Whether a compacted call runs the hidden valid-row count: a registered program with the gather kernel and the
    device-count form of chunk 0's logits GEMM (its raster height at the buffer extent ``min(chunk, T)`` is a launch
    scalar), a call of
    :data:`HIDDEN_COUNT_MIN_CHUNKS` chunks or more (``ceil(T / chunk) >= 3``: one- and two-chunk calls cannot hide the
    path's fixed host cost behind chunk 0's logits GEMM), an ``X`` the GEMM reads in
    place -- a 16-byte row pitch (no contiguous copy) and a 16-byte-aligned base -- and a toolchain whose code for the
    record's architecture runs the hidden count's schedule (:func:`cake_jit.toolchain_runs_hidden_count`: not nvcc
    13.0.x for sm_103a).  Every other call takes the shipped :func:`valid_rows` path."""
    rows, chunk = int(problem.num_rows), int(problem.chunk)
    if (
        record is None
        or module_name is None
        or rows <= 0
        or (rows + chunk - 1) // chunk < HIDDEN_COUNT_MIN_CHUNKS
        or problem.x_copy
        or X.data_ptr() % 16
    ):
        return False
    if not toolchain_runs_hidden_count(record["arch"]):
        return False
    stages = registered_stages(module_name)
    stage = stage_variant("gemm_logits" if stats else "gemm_logits_nostats", count=True)
    return HIDDEN_COUNT_STAGE in stages and stage in stages


def hidden_count_eligible_temperature(
    problem: "Problem",
    X: torch.Tensor,
    record: Optional[dict[str, Any]],
    module_name: Optional[str],
    *,
    stats: bool,
) -> bool:
    """:func:`hidden_count_eligible` of a call with a temperature (:attr:`Problem.temperature`): the same rules, with
    the temperature form of the device-count GEMM with the statistics (:data:`HIDDEN_LOGITS_TAU_STAGE`; the forward
    only -- there is no stats-free device-count form with a temperature, so a recount takes the host count)."""
    if not stats:
        return False
    rows, chunk = int(problem.num_rows), int(problem.chunk)
    if (
        record is None
        or module_name is None
        or rows <= 0
        or (rows + chunk - 1) // chunk < HIDDEN_COUNT_MIN_CHUNKS
        or problem.x_copy
        or X.data_ptr() % 16
    ):
        return False
    if not toolchain_runs_hidden_count(record["arch"]):
        return False
    stages = registered_stages_with_temperature(module_name)
    return HIDDEN_COUNT_STAGE in stages and HIDDEN_LOGITS_TAU_STAGE in stages


def _hidden_count_begin(
    problem: "Problem",
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    record: dict[str, Any],
    module_name: str,
    stats: bool,
    loss_weights: Optional[torch.Tensor] = None,
    kept_rows: bool = False,
) -> _HiddenCount:
    """Queue the device compaction, chunk 0's row gather and chunk 0's device-count logits GEMM of a compacted call
    without the valid-row count on the host; the count is read back asynchronously (:meth:`_HiddenCount.finish`).

    Order: (1) chunk 0's buffers at the buffer extent ``rows0 = min(chunk, T)`` -- the BF16 ``[rows0, V]`` logits
    buffer, its row statistics and the BF16 ``[rows0, H]`` gather buffer -- and the compaction outputs; (2) the two
    launches are bound (argument plans, grids: host work only, nothing enqueued yet); (3) ``labels >= 0`` -> ``cumsum``
    -> ``searchsorted`` (:func:`compaction_index`), the count's ``copy_(non_blocking=True)`` into the pinned cell and
    the event behind it; (4) the gather (rows ``< count`` of ``X`` in valid-row order, exact zeros after) and the GEMM
    (``M`` / ``m_tiles`` / the raster rule at ``rows0``, the stores bounded by ``min(count, rows0)``) are launched.  The
    caller resolves the plan on the count afterwards and skips chunk 0's gather and GEMM (:attr:`Plan.hidden_count`).

    ``loss_weights`` (the zero-weight split of a policy call, :func:`gradient_index`): the counted rows are the gradient
    rows ``(labels >= 0) & (loss_weights != 0)``; the zero-weight row count travels in the same pinned copy (8 bytes)
    and :meth:`_HiddenCount.zero_index` forms their index after the readback.

    ``kept_rows``: the logits buffer is the first ``rows0`` rows of a contiguous BF16 ``[T, V]`` kept buffer
    (:attr:`_HiddenCount.kept`) instead of its own allocation."""
    device = X.device
    index = int(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    geometry = Geometry.from_record(record)
    T, H, V, C = (
        int(problem.num_rows),
        int(problem.hidden),
        int(problem.vocab),
        int(problem.chunk),
    )
    rows0 = min(C, T)
    num_tiles = -(-V // geometry.stats_tile)
    kept = (
        torch.empty((T, V), dtype=torch.bfloat16, device=device) if kept_rows else None
    )
    logits = (
        torch.empty((rows0, V), dtype=torch.bfloat16, device=device)
        if kept is None
        else kept[:rows0]
    )
    stats_buf = torch.empty((rows0, num_tiles, 2), dtype=torch.float32, device=device)
    x_c = torch.empty((rows0, H), dtype=torch.bfloat16, device=device)
    Xd, Wd = X.detach(), W.detach()
    split = loss_weights is not None
    count_host, event = (_hidden_count_pair_cell if split else _hidden_count_cell)(
        index
    )
    ar = _hidden_count_arange(index, T)
    masks = torch.empty(((2 if split else 1) * T,), dtype=torch.bool, device=device)
    valid = masks[:T]
    pos = torch.empty(((2 if split else 1) * T,), dtype=torch.int32, device=device)
    idx_full = torch.empty((T,), dtype=torch.int64, device=device)
    count_dev = pos[T - 1 : T]
    ld = int(Xd.stride(0))
    raster = raster_variant(H, rows0, record["arch"])
    group_m = None if raster is None else raster[0]
    stage = stage_variant("gemm_logits" if stats else "gemm_logits_nostats", count=True)
    common: dict[str, Any] = {name: None for name in COMMON_TENSORS}
    common.update({name: 0 for name in COMMON_SCALARS})
    common.update(
        T=T, H=H, V=V, chunk=C, num_tiles=num_tiles, loss_div=1.0, rows_c=rows0
    )
    # the kernel addresses X by base pointer + ld_x_words: its storage span (padding columns included) as ONE contiguous
    # int32 alias, so a padded-pitch X passes the launch shim's contiguity check without a copy
    gather = dict(
        common,
        x=Xd.as_strided(((T - 1) * ld + H,), (1,)).view(torch.int32),
        idx_lo=idx_full.view(torch.int32),
        count=count_dev,
        out=x_c.view(torch.int32),
        ld_x_words=ld // 2,
        row_vecs=H // geometry.cast_vec,
        num_rows=rows0,
    )
    gemm = dict(
        common,
        A=x_c,
        B=Wd,
        C=logits,
        STATS_OUT=stats_buf,
        WS=count_dev,
        **gemm_scalars(
            stage, geometry, hidden=H, vocab=V, rows_c=rows0, group_m=group_m
        ),
    )
    keys = ((HIDDEN_COUNT_STAGE, "hidden"), (stage, "hidden"))
    launches = _bind_all(
        record,
        module_name,
        keys,
        {keys[0]: gather, keys[1]: gemm},
        device,
        geometry,
    )
    if split:
        gradient_index(
            labels, loss_weights, masks=masks, pos=pos, idx_full=idx_full, ar=ar
        )
        count_host.copy_(
            pos[T - 1 :: T], non_blocking=True
        )  # both counts: gradient rows, gradient + zero-weight rows
    else:
        compaction_index(labels, valid=valid, pos=pos, idx_full=idx_full, ar=ar)
        count_host.copy_(count_dev, non_blocking=True)
    event.record()
    with _ffi_stream_context(index):
        for key in keys:
            launch = launches[key]
            if launch.prepare is not None:
                launch.prepare(*launch.arguments)
            launch()
    return _HiddenCount(
        rows0=rows0,
        stats=stats,
        stage=stage,
        group_m=group_m,
        valid=valid,
        idx_full=idx_full,
        count_host=count_host,
        event=event,
        logits=logits,
        stats_buf=stats_buf,
        x_c=x_c,
        zero_pos=pos[T:] if split else None,
        arange=ar if split else None,
        kept=kept,
    )


def _hidden_count_begin_temperature(
    problem: "Problem",
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    record: dict[str, Any],
    module_name: str,
    stats: bool,
    temperature: TemperatureForm = None,
) -> _HiddenCount:
    """:func:`_hidden_count_begin` of a call with a temperature.  Queue the device compaction, chunk 0's row gather
    and chunk 0's device-count logits GEMM of a compacted call without the valid-row count on the host; the count is read back asynchronously (:meth:`_HiddenCount.finish`).

    ``temperature`` (the call's resolved form, :func:`_resolve_temperature`; requires ``stats``): the FP32 ``[T]``
    compact temperature ``tau_c`` is allocated with the chunk-0 buffers (the GEMM's binding captures it) and filled
    behind the count's copy, before the launches: ``tau_c[k] = tau[min(idx_full[k], T - 1)]`` (the valid rows in
    order; the slots past the count, whose ``idx_full`` is ``T``, reach no output) or the scalar's value.  The GEMM is
    the temperature form (:data:`HIDDEN_LOGITS_TAU_STAGE`) with ``row0 = 0``.

    Order: (1) chunk 0's buffers at the buffer extent ``rows0 = min(chunk, T)`` -- the BF16 ``[rows0, V]`` logits
    buffer, its row statistics and the BF16 ``[rows0, H]`` gather buffer -- and the compaction outputs; (2) the two
    launches are bound (argument plans, grids: host work only, nothing enqueued yet); (3) ``labels >= 0`` -> ``cumsum``
    -> ``searchsorted`` (:func:`compaction_index`), the count's ``copy_(non_blocking=True)`` into the pinned cell and
    the event behind it; (4) the gather (rows ``< count`` of ``X`` in valid-row order, exact zeros after) and the GEMM
    (``M`` / ``m_tiles`` / the raster rule at ``rows0``, the stores bounded by ``min(count, rows0)``) are launched.  The
    caller resolves the plan on the count afterwards and skips chunk 0's gather and GEMM (:attr:`Plan.hidden_count`)."""
    device = X.device
    index = int(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    geometry = Geometry.from_record(record)
    T, H, V, C = (
        int(problem.num_rows),
        int(problem.hidden),
        int(problem.vocab),
        int(problem.chunk),
    )
    rows0 = min(C, T)
    num_tiles = -(-V // geometry.stats_tile)
    logits = torch.empty((rows0, V), dtype=torch.bfloat16, device=device)
    stats_buf = torch.empty((rows0, num_tiles, 2), dtype=torch.float32, device=device)
    x_c = torch.empty((rows0, H), dtype=torch.bfloat16, device=device)
    Xd, Wd = X.detach(), W.detach()
    count_host, event = _hidden_count_cell(index)
    ar = _hidden_count_arange(index, T)
    valid = torch.empty((T,), dtype=torch.bool, device=device)
    pos = torch.empty((T,), dtype=torch.int32, device=device)
    idx_full = torch.empty((T,), dtype=torch.int64, device=device)
    count_dev = pos[T - 1 : T]
    ld = int(Xd.stride(0))
    raster = raster_variant(H, rows0, record["arch"])
    group_m = None if raster is None else raster[0]
    stage = stage_variant("gemm_logits" if stats else "gemm_logits_nostats", count=True)
    if temperature is not None:
        if not stats:
            raise ValueError(
                "the hidden valid-row count with a temperature runs the device-count GEMM with the statistics"
            )
        stage = HIDDEN_LOGITS_TAU_STAGE
    # allocated before the binding below captures it; filled on the device behind the compaction
    tau_c = (
        None
        if temperature is None
        else torch.empty((T,), dtype=torch.float32, device=device)
    )
    common: dict[str, Any] = {name: None for name in COMMON_TENSORS}
    common.update({name: 0 for name in COMMON_SCALARS})
    common.update(
        T=T, H=H, V=V, chunk=C, num_tiles=num_tiles, loss_div=1.0, rows_c=rows0
    )
    # the kernel addresses X by base pointer + ld_x_words: its storage span (padding columns included) as ONE contiguous
    # int32 alias, so a padded-pitch X passes the launch shim's contiguity check without a copy
    gather = dict(
        common,
        x=Xd.as_strided(((T - 1) * ld + H,), (1,)).view(torch.int32),
        idx_lo=idx_full.view(torch.int32),
        count=count_dev,
        out=x_c.view(torch.int32),
        ld_x_words=ld // 2,
        row_vecs=H // geometry.cast_vec,
        num_rows=rows0,
    )
    gemm = dict(
        common,
        A=x_c,
        B=Wd,
        C=logits,
        STATS_OUT=stats_buf,
        WS=count_dev,
        **gemm_scalars(
            stage, geometry, hidden=H, vocab=V, rows_c=rows0, group_m=group_m
        ),
    )
    if tau_c is not None:
        # chunk 0: the compact rows 0 .. rows0 - 1
        gemm.update(temperature=tau_c, row0=0)
    keys = ((HIDDEN_COUNT_STAGE, "hidden"), (stage, "hidden"))
    _check_temperature_operands(module_name, keys, {keys[0]: gather, keys[1]: gemm})
    launches = _bind_all(
        record,
        module_name,
        keys,
        {keys[0]: gather, keys[1]: gemm},
        device,
        _TemperatureGeometry.of(geometry),
    )
    compaction_index(labels, valid=valid, pos=pos, idx_full=idx_full, ar=ar)
    count_host.copy_(count_dev, non_blocking=True)
    event.record()
    if tau_c is not None:
        kind, value = temperature
        if kind == "scalar":
            tau_c.fill_(value)
        else:  # idx_full holds T past the count: clamped, those slots reach no output
            torch.index_select(value, 0, idx_full.clamp_max(T - 1), out=tau_c)
    with _ffi_stream_context(index):
        for key in keys:
            launch = launches[key]
            if launch.prepare is not None:
                launch.prepare(*launch.arguments)
            launch()
    hc = _HiddenCount(
        rows0=rows0,
        stats=stats,
        stage=stage,
        group_m=group_m,
        valid=valid,
        idx_full=idx_full,
        count_host=count_host,
        event=event,
        logits=logits,
        stats_buf=stats_buf,
        x_c=x_c,
    )
    hc.tau_c = tau_c
    return hc


def _hidden_count_memory(
    memory: dict[str, Any],
    *,
    num_rows: int,
    hidden: int,
    vocab: int,
    chunk: int,
    stats_tile: int,
) -> dict[str, Any]:
    """The memory report of a call on the hidden valid-row count path: the chunk buffers and the gather buffer exist
    at the buffer extent ``min(chunk, T)`` BEFORE the count is known -- also when the outcome is the uncompacted plan,
    which counts no gather buffer."""
    rows0 = min(int(chunk), max(int(num_rows), 1))
    tiles = -(-int(vocab) // int(stats_tile))
    temporary = dict(memory["temporary"])
    temporary["logits"] = rows0 * int(vocab) * 2
    temporary["stats"] = rows0 * tiles * 2 * 4
    temporary["x_c"] = rows0 * int(hidden) * 2
    out = dict(
        memory,
        temporary=temporary,
        temporary_bytes=sum(temporary.values()),
        peak_temporary_bytes=sum(temporary.values()),
        gather_bytes=temporary["x_c"],
        hidden_count=True,
    )
    if memory.get("saved_dz_bytes"):
        out["saved_dz_bytes"] = rows0 * int(vocab) * 2
    return out


def _hidden_count_memory_temperature(
    memory: dict[str, Any],
    *,
    num_rows: int,
    hidden: int,
    vocab: int,
    chunk: int,
    stats_tile: int,
    temperature: bool = False,
) -> dict[str, Any]:
    """The memory report of a call on the hidden valid-row count path: the chunk buffers and the gather buffer exist
    at the buffer extent ``min(chunk, T)`` BEFORE the count is known -- also when the outcome is the uncompacted plan,
    which counts no gather buffer.  ``temperature``: the compact temperature is the ``[T]`` FP32 buffer allocated
    before the count (the saved ``[rows]`` view keeps all of it alive) and its gather reads a transient ``[T]`` int64
    clamped index."""
    rows0 = min(int(chunk), max(int(num_rows), 1))
    tiles = -(-int(vocab) // int(stats_tile))
    temporary = dict(memory["temporary"])
    temporary["logits"] = rows0 * int(vocab) * 2
    temporary["stats"] = rows0 * tiles * 2 * 4
    temporary["x_c"] = rows0 * int(hidden) * 2
    accumulators = memory["accumulators"]
    if temperature:
        temporary["temperature_index"] = int(num_rows) * 8
        accumulators = dict(accumulators, temperature_rows=int(num_rows) * 4)
    out = dict(
        memory,
        temporary=temporary,
        temporary_bytes=sum(temporary.values()),
        peak_temporary_bytes=sum(temporary.values()),
        accumulators=accumulators,
        accumulator_bytes=sum(accumulators.values()),
        gather_bytes=temporary["x_c"],
        hidden_count=True,
    )
    if memory.get("saved_dz_bytes"):
        out["saved_dz_bytes"] = rows0 * int(vocab) * 2
    return out


def valid_rows(
    labels: torch.Tensor, *, mask: bool = False
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``(row_index, row_valid)``.  ``row_index`` is ``None`` when every row is valid (or ``T == 0``: the uncompacted
    path, no gather cost), else the ascending int64 index of the valid rows (``labels >= 0``; ``-100`` is the ignore
    index).  Rows with a negative label contribute exactly zero to the loss, ``dX`` and ``dW``, so skipping them is
    exact.  One device synchronization: ``nonzero`` reads its count back once and returns the index with it, so the
    count is the index's length (every row valid: the index is dropped).  ``mask=True`` (the fused dX finalize of a
    call that produces ``dX``) also returns the already materialized bool valid-row mask ``labels >= 0`` (``[T]``;
    :func:`finalize_dx` turns it into the inclusive int32 scan behind the backward's queued GEMMs, not here in the
    synchronization shadow of ``nonzero``); ``None`` whenever ``row_index`` is.  A CUDA call of at most
    :data:`HOST_COUNT_MAX_ROWS` rows reads the mask back instead (the one synchronization) and forms the same index on
    the host, copied back to the labels' device: no count or select kernels."""
    T = int(labels.numel())
    if T == 0:
        return None, None
    valid = labels >= 0
    if T <= HOST_COUNT_MAX_ROWS and labels.is_cuda:
        index = valid.cpu().nonzero().squeeze(1)
        if int(index.numel()) == T:
            return None, None
        # pageable host memory: the copy is staged before it returns, so the host index may go right away
        return index.to(labels.device, non_blocking=True), (valid if mask else None)
    index = valid.nonzero().squeeze(1)
    if int(index.numel()) == T:
        return None, None
    return index, (valid if mask else None)


def valid_row_index(labels: torch.Tensor) -> Optional[torch.Tensor]:
    """The valid-row index of :func:`valid_rows` alone (``None`` when every row is valid)."""
    return valid_rows(labels)[0]


def gradient_index(
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    *,
    masks: torch.Tensor,
    pos: torch.Tensor,
    idx_full: torch.Tensor,
    ar: torch.Tensor,
) -> None:
    """The zero-weight split of :func:`compaction_index` on the device (no host synchronization), into pre-allocated
    outputs: ``masks[:T]`` = the gradient-row mask ``G = (labels >= 0) & (loss_weights != 0)`` (a NaN weight is in
    ``G``), ``masks[T:]`` = the zero-weight mask ``Z = (labels >= 0) & (loss_weights == 0)``, ``pos`` = the inclusive
    int32 scan of ``masks`` (``pos[T - 1] = |G|``, ``pos[2T - 1] = |G| + |Z|``) and ``idx_full[k] = searchsorted(pos[:T],
    k + 1)`` -- the ``k``-th gradient row for ``k < |G|``, ``T`` after it.  The ``k``-th zero-weight row is
    ``searchsorted(pos[T:], |G| + k + 1)`` (:meth:`_HiddenCount.zero_index`, :func:`gradient_rows`)."""
    T = int(labels.numel())
    valid = labels >= 0
    torch.logical_and(valid, loss_weights != 0, out=masks[:T])
    torch.logical_xor(valid, masks[:T], out=masks[T:])
    torch.cumsum(masks, 0, dtype=torch.int32, out=pos)
    torch.searchsorted(pos[:T], ar[:T], out=idx_full)


# The smallest call (rows ``T``) that splits off its zero-weight rows (:func:`set_zero_weight_min_rows`, no environment
# variable): below it the call keeps every valid row on the gradient path, as the deterministic package does (the split
# adds row kernels and a second count to every call that takes it).  Default 4096 (one default chunk), set from A/B
# timing measured during development on B200 (sm_100a, CUDA 13.4 toolkit): policy rows, T = 512 / 1024 / 2048 / 3000 at
# no and at 16.6 % zero-weight rows, floors 0 / 1024 / 2048 / 4096; forward, backward, step (forward + backward) and
# checkpointed step (two forwards + backward); each call after a GPU delay as long as one back-to-back call ("cooled"),
# after a 2 ms delay ("d1") or back to back ("hot"); two independent runs, each interleaving the floors; each number is
# a pooled median time ratio of the two runs:
# * the split over the deterministic package, step at T = 512 and 1024: 1.001 to 1.18;
# * the split (floor 2048) over no split (floor 4096) back to back (hot), step / checkpointed step: T = 2048 1.025 /
#   1.012 at no and 1.019 / 1.024 at 16.6 % zero-weight rows; T = 3000 1.020 / 1.009 at no and 0.9594 / 0.9524 at
#   16.6 % zero-weight rows (both runs below 1);
# * the same after an idle gap (cooled / d1): step 0.977 to 0.993 at T = 2048 and 0.924 to 0.992 at T = 3000;
#   checkpointed step 0.937 to 0.991 in five cells, 0.996 at T = 3000 without zero-weight rows after d1 (its runs
#   read 1.004 and 0.995), 1.001 (d1) and 1.002 (cooled) at T = 2048 with 16.6 % zero-weight rows.
# The 4096 floor runs no split at T <= 3000: it keeps the back-to-back ratios above 1 at T = 2048 / 3000 out and
# gives up the ratios below 1 after an idle gap and back to back at T = 3000 with 16.6 % zero-weight rows.  Over the
# deterministic package, its step and checkpointed step at T <= 3000 have no pooled 95 % CI above 1; the cells whose
# pooled CI lies above 1 are backward at T = 512 / 1024 (1.0006 to 1.0077; the runs of the 1.0077 cell, T = 1024 hot,
# read 0.930 and 1.011) and one d1 forward at T = 3000 (1.0014).
# The same sweep on GB300 (sm_103a, CUDA 13.4 toolkit, the same code), the split (floor 2048) over no split: back to
# back 1.040 to 1.051 (step) in the three cells whose two runs are both above 1 (T = 2048 and 3000 without zero-weight
# rows, T = 2048 with 16.6 %); at T = 3000 with 16.6 % zero-weight rows back to back the runs disagree (1.152 and 0.992;
# direction not settled), where the B200 runs both read below 1 (0.9594 pooled).  After an idle gap the step reads
# 0.9567 to 0.9867 (at T = 3000 with 16.6 % zero-weight rows after d1 its runs read 1.029 and 0.932).  The same floor
# applies there.
ZERO_WEIGHT_MIN_ROWS_DEFAULT = 4096
_ZERO_WEIGHT_MIN_ROWS = {"rows": ZERO_WEIGHT_MIN_ROWS_DEFAULT}


def zero_weight_min_rows() -> int:
    """The smallest call (rows) that splits off its zero-weight rows (:func:`set_zero_weight_min_rows`; default
    :data:`ZERO_WEIGHT_MIN_ROWS_DEFAULT`)."""
    return _ZERO_WEIGHT_MIN_ROWS["rows"]


def set_zero_weight_min_rows(rows: int) -> int:
    """Set the smallest call (rows ``T``) that splits off its zero-weight rows for the following calls of this process
    (``0``: every compacted policy call) and return the previous value.  Read once per call; no binding depends on it
    (the split changes the call's valid-row count, which the binding key holds)."""
    if isinstance(rows, bool) or int(rows) != rows or int(rows) < 0:
        raise ValueError(
            f"zero-weight min rows must be a non-negative integer, got {rows!r}"
        )
    previous = _ZERO_WEIGHT_MIN_ROWS["rows"]
    _ZERO_WEIGHT_MIN_ROWS["rows"] = int(rows)
    return previous


# The smallest log-probability backward (rows of its chunk loop) that runs the zero-gradient row skip
# (:func:`set_zero_dlogp_min_rows`, no environment variable; ``None`` = never).  The skip orders the rows on the device
# (one partition launch per call, no host synchronization) at the start of the backward, where the GPU waits for the
# host.  The generic floor is one row above one default chunk: a smaller call keeps the shipped backward on an
# architecture without a floor of its own (:data:`ZERO_DLOGP_MIN_ROWS_BY_ARCH`).
ZERO_DLOGP_MIN_ROWS_DEFAULT: Optional[int] = DEFAULT_CHUNK_SIZE + 1
# Per-architecture default floors (valid rows); an architecture not listed uses :data:`ZERO_DLOGP_MIN_ROWS_DEFAULT`.
# Timed on FlashInfer PR #8687 at its commit 6ab51864 (PyTorch 2.14.1+cu130, nvcc 13.4), plain and temperature
# log-probability, cooled and after an idle gap, against the same head with the skip off: ratio = time without the
# skip / time with it, with its CI95.
# B200 (sm_100a), floor 1946 valid rows (T = 2048 with 5 % ignored rows): with no zero-gradient row, of the backward,
# step and checkpointed step of 1946 and 3891 valid rows none is slower under the timing rule (CI95 below
# 1 - max(1 %, A/A)); the lowest point ratio is 0.984 [0.974, 0.996] (temperature backward, 1946 valid rows, cooled).
# GB300 (sm_103a), floor 3891 valid rows (T = 4096): at 3891 valid rows none is slower under the timing rule; the
# lowest point ratio is 0.989 [0.985, 0.994] (temperature backward, cooled).  At 1946 valid rows the GB300
# temperature backward with no zero-gradient row has point ratio 0.981 [0.978, 0.990] (the CI95 crosses the band
# edge) and 0.984 on each of two clusters alone (each CI95 below the band, fewer steps than the rule needs), so the
# GB300 floor is 3891.  With 16.6 % zero-gradient rows the step at the shipped floors is 4.6 % to 10.6 % faster on
# B200 (1946 and 3891 valid rows) and 8.8 % to 9.5 % faster on GB300 (3891 valid rows), each faster under the timing
# rule.
# R200 (sm_107a), floor 1946 valid rows with the two-slice counted dX rule (:data:`DX_COUNTED_MIN_SLICES_BY_ARCH`).
# The two were chosen together by timing on FlashInfer main ed5b3885 with both set at run time (backward, step and
# checkpointed step of 1100, 1946 and 3891 valid rows: none slower on either R200 board), then timed in this code at
# commit 237877ca (PyTorch 2.14.1+cu130, nvcc 13.4), cooled, against the same code without its two sm_107a entries
# (FlashInfer main d3e87763: the generic floor and the slice rule's dX counts), the 212-SM and the 216-SM R200 board
# each alone, two machines each: ratio = time without the entries / time with them, with its CI95.  Plain, per-token
# and scalar temperature log-probability, T = 4600 to 65031 with 0, 16.6 % and 50 % zero-gradient rows (195 cells of
# the forward, backward, step and checkpointed step per board): none is slower under the timing rule; 21 cells on the
# 212-SM board and 30 on the 216-SM board are faster, all with zero-gradient rows, up to 1.2225 [1.2158, 1.2342] and
# 1.2562 [1.2493, 1.2659] (per-token temperature backward, 15419 valid rows, chunk 2048, 50 % zero-gradient rows).
# The lowest point ratio is 0.9913 [0.9760, 1.0077] on the 212-SM board (checkpointed step, 8550 valid rows, 50 %
# zero-gradient rows; band 2.35 %) and 0.9815 [0.9751, 1.0000] on the 216-SM board (per-token temperature forward,
# 61779 valid rows, 50 %; band 1.88 %).  At T = 4600 with 4000 valid rows (on the skip with this floor, off it with
# the generic one), plain and per-token temperature, none of the 18 cells of the forward, backward and step is slower:
# with no zero-gradient row the lowest point ratio is 0.9814 [0.9665, 0.9929] on the 212-SM board (plain backward;
# band 2.23 %) and 0.9973 [0.9951, 0.9993] on the 216-SM board (temperature backward); with 16.6 % and 50 %
# zero-gradient rows the backward reads 1.0678 to 1.8747 on the 212-SM board and 1.1087 to 1.8036 on the 216-SM
# board, the step 1.0536 to 1.4244 and 1.0764 to 1.4940, each faster.
ZERO_DLOGP_MIN_ROWS_BY_ARCH: dict[str, int] = {
    "sm_100a": 1946,
    "sm_103a": 3891,
    "sm_107a": 1946,
}


class _ArchDefaultFloor:
    """The type of :data:`ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT`, one object per process: a copy, a deep copy and a pickle
    round trip give that object back, and :func:`set_zero_dlogp_min_rows` stores it for any value of this type."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT"

    def __reduce__(self) -> str:
        return "ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT"

    def __copy__(self) -> "_ArchDefaultFloor":
        return ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT

    def __deepcopy__(self, memo: dict) -> "_ArchDefaultFloor":
        return ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT


# The floor setting "no override" (:func:`set_zero_dlogp_min_rows`): each device uses its architecture's default floor.
# ``set_zero_dlogp_min_rows`` returns it when no override was set, and passing it back restores that state.
ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT = _ArchDefaultFloor()
_ZERO_DLOGP_MIN_ROWS: dict[str, Any] = {"rows": ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT}
# Per CUDA device index: its architecture tag (:func:`arch_for`; ``""`` when unsupported), resolved once per device.
_ZERO_DLOGP_ARCH: dict[int, str] = {}


def zero_dlogp_min_rows(device: Optional[int] = None) -> Optional[int]:
    """The smallest log-probability backward (rows of its chunk loop) that runs the zero-gradient row skip, ``None`` =
    never: the override of :func:`set_zero_dlogp_min_rows` when one is set, else the default floor -- of CUDA device
    index ``device`` (its architecture's, :data:`ZERO_DLOGP_MIN_ROWS_BY_ARCH`), or the generic
    :data:`ZERO_DLOGP_MIN_ROWS_DEFAULT` without a device."""
    rows = _ZERO_DLOGP_MIN_ROWS["rows"]
    if rows is not ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT:
        return rows
    if device is None:
        return ZERO_DLOGP_MIN_ROWS_DEFAULT
    arch = _ZERO_DLOGP_ARCH.get(device)
    if arch is None:
        arch = _ZERO_DLOGP_ARCH.setdefault(
            device, arch_for(torch.device("cuda", device)) or ""
        )
    return ZERO_DLOGP_MIN_ROWS_BY_ARCH.get(arch, ZERO_DLOGP_MIN_ROWS_DEFAULT)


def set_zero_dlogp_min_rows(
    rows: Union[Optional[int], _ArchDefaultFloor],
) -> Union[Optional[int], _ArchDefaultFloor]:
    """Set the smallest log-probability backward (rows of its chunk loop) that runs the zero-gradient row skip on every
    architecture for the following calls of this process (``0``: every eligible call; ``None``: never;
    :data:`ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT`: no override, each architecture's default floor; any value of its type, a
    copy or an unpickled one included, stores that object) and return the previous setting, which passed back restores
    it exactly.  Read once per call; the binding key holds whether a call runs it."""
    if (
        rows is not None
        and not isinstance(rows, _ArchDefaultFloor)
        and (isinstance(rows, bool) or int(rows) != rows or int(rows) < 0)
    ):
        raise ValueError(
            "zero-dlogp min rows must be None, a non-negative integer or ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT, "
            f"got {rows!r}"
        )
    previous = _ZERO_DLOGP_MIN_ROWS["rows"]
    if rows is None:
        _ZERO_DLOGP_MIN_ROWS["rows"] = None
    elif isinstance(rows, _ArchDefaultFloor):
        _ZERO_DLOGP_MIN_ROWS["rows"] = ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT
    else:
        _ZERO_DLOGP_MIN_ROWS["rows"] = int(rows)
    return previous


# Whether a log-probability backward with a temperature under the zero-gradient row skip runs the counted temperature
# form of the ``dz`` arm (:data:`LOGPROB_DZ_DCNT_TAU_STAGE`, where the record registers it) or the uncounted one over
# every row (the mixed plan of :func:`make_plan_temperature`; :func:`set_zero_dlogp_temperature_dz_counted`).
_ZERO_DLOGP_TAU_DZ: dict[str, bool] = {"counted": True}


def zero_dlogp_temperature_dz_counted() -> bool:
    """Whether the zero-gradient row skip of a call with a temperature runs the counted temperature form of the ``dz``
    arm (default) or the uncounted one (:func:`set_zero_dlogp_temperature_dz_counted`)."""
    return _ZERO_DLOGP_TAU_DZ["counted"]


def set_zero_dlogp_temperature_dz_counted(counted: bool) -> bool:
    """Set whether the zero-gradient row skip of the following calls with a temperature runs the counted temperature
    form of the ``dz`` arm (``True``, the default) or the uncounted one over every row (``False``: the mixed plan, the
    same outputs) and return the previous value.  Read once per call; the binding key holds it."""
    if not isinstance(counted, bool):
        raise ValueError(f"counted must be a bool, got {counted!r}")
    previous = _ZERO_DLOGP_TAU_DZ["counted"]
    _ZERO_DLOGP_TAU_DZ["counted"] = counted
    return previous


# Per device index (:func:`_skip_record`): the record :func:`record_for` selects and whether it registers the partition
# kernel and the whole zero-gradient row skip (the partition kernel and the counted dz arm).  The registry is fixed at
# import and a device keeps its architecture, so each device resolves this once; the skip path's per-call host work
# (:func:`_zero_dlogp_skip`, :func:`zero_dlogp_partition`, both ahead of the backward's first launch) reads it instead of
# selecting the record and scanning its stages on every call.
_SKIP_RECORDS: dict[int, tuple[str, dict[str, Any], bool, bool]] = {}


def _skip_record(index: int) -> tuple[str, dict[str, Any], bool, bool]:
    """``(module_name, record, partition, skip)`` of CUDA device ``index``: its record (:func:`record_for`; raises when
    absent), whether the record registers :data:`ZERO_DLOGP_PARTITION_STAGE` and whether it also registers
    :data:`LOGPROB_DZ_DCNT_STAGE` (the zero-gradient row skip).  Resolved once per device; threads that resolve it
    together all return the first stored entry (the same values)."""
    entry = _SKIP_RECORDS.get(index)
    if entry is None:
        module_name, record = record_for(torch.device("cuda", index))
        stages = registered_stages(module_name)
        partition = ZERO_DLOGP_PARTITION_STAGE in stages
        entry = _SKIP_RECORDS.setdefault(
            index,
            (
                module_name,
                record,
                partition,
                partition and LOGPROB_DZ_DCNT_STAGE in stages,
            ),
        )
    return entry


def _flat(t: torch.Tensor) -> torch.Tensor:
    """``t.reshape(-1).contiguous()``, without the view when ``t`` already is a contiguous 1-D tensor (``t`` itself)."""
    return t if t.dim() == 1 and t.is_contiguous() else t.reshape(-1).contiguous()


def _partition_outputs(
    rows: int, T: int, num_chunks: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The partition kernel's four outputs split from ONE int32 allocation (int64 ``order [rows]``, FP32
    ``lse_out [rows]``, int32 ``scan [T]`` and ``counts [num_chunks]``), each region at a :data:`WORKSPACE_ALIGN`-byte
    boundary as the workspace regions (:func:`_align`); one caching-allocator call instead of four ahead of the backward's
    first launch.  The regions are disjoint views of one block that lives as long as any of them."""
    o_lse = _align(8 * rows) // 4
    o_scan = o_lse + _align(4 * rows) // 4
    o_counts = o_scan + _align(4 * T) // 4
    block = torch.empty((o_counts + num_chunks,), dtype=torch.int32, device=device)
    order, _, lse_out, _, scan, _, counts = block.split_with_sizes(
        (
            2 * rows,
            o_lse - 2 * rows,
            rows,
            o_scan - o_lse - rows,
            T,
            o_counts - o_scan - T,
            num_chunks,
        )
    )
    return order.view(torch.int64), lse_out.view(torch.float32), scan, counts


def zero_dlogp_partition(
    labels: torch.Tensor,
    dlogp: torch.Tensor,
    row_index: torch.Tensor,
    lse: torch.Tensor,
    chunk: int,
    *,
    backend: str = "cake",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The zero-gradient row skip's row order of a compacted log-probability backward, on the device (no host
    synchronization; capturable): ``(row_index, lse, scan, counts)``, the outputs of
    :func:`zero_dlogp_partition_reference` bit for bit.

    On a CUDA device whose record registers the partition kernel (:data:`ZERO_DLOGP_PARTITION_STAGE`) and
    ``backend="cake"`` this is ONE launch: the reference's sixteen small kernels each pay a host launch at the start of
    the backward, ahead of its first GEMM.  The kernel reads ``labels`` for the valid rows instead of ``row_index``,
    which must be the ascending index of the rows with a non-negative label (the forward's valid-row index, as the
    backward passes it).  Any other call (another device or backend, a record without the kernel, an int64 / FP32
    contract the inputs do not meet) runs the reference.

    The call sits on the host path ahead of the backward's first launch, where the GPU waits: the device's record is
    resolved once (:func:`_skip_record`), the four outputs are one allocation (:func:`_partition_outputs`) and, while
    :data:`BINDING_CACHE` is enabled, the templated launch of each ``(device, T, chunk)`` form is remembered with the
    eager output-boundary launches (:func:`_eager_launch`'s cache, under a key of those values) -- a later call only
    slots its tensors into it: the same entry, grid and arguments as a fresh binding."""
    rows = int(row_index.numel())
    device = dlogp.device
    if (
        backend != "cake"
        or device.type != "cuda"
        or rows == 0
        or labels.dtype != torch.int64
        or row_index.dtype != torch.int64
        or lse.dtype != torch.float32
        or dlogp.dtype != torch.float32
    ):
        return zero_dlogp_partition_reference(labels, dlogp, row_index, lse, chunk)
    index = device.index if device.index is not None else torch.cuda.current_device()
    module_name, record, partition, _skip = _skip_record(index)
    if not partition:
        return zero_dlogp_partition_reference(labels, dlogp, row_index, lse, chunk)
    T = int(labels.numel())
    chunk = int(chunk)
    num_chunks = -(-rows // chunk)
    order, lse_ordered, scan, counts = _partition_outputs(rows, T, num_chunks, device)
    tensors = dict(
        labels=_flat(labels),
        dlogp=_flat(dlogp),
        lse=_flat(lse),
        order=order,
        lse_out=lse_ordered,
        scan=scan,
        counts=counts,
    )
    cache = BINDING_CACHE
    form = (ZERO_DLOGP_PARTITION_STAGE, "form", index, T, chunk, num_chunks)
    launch = cache.eager_lookup(form, record) if cache.enabled else None
    if launch is None:
        record_abi(record)
        values: dict[str, Any] = {name: None for name in COMMON_TENSORS}
        values.update({name: 0 for name in COMMON_SCALARS})
        values.update(tensors, T=T, chunk=chunk, num_chunks=num_chunks)
        launch, arguments = _eager_launch(
            record,
            module_name,
            ZERO_DLOGP_PARTITION_STAGE,
            values,
            device,
            Geometry.from_record(record),
        )
        if cache.enabled:
            cache.eager_remember(form, record, launch.templated())
    else:
        arguments = launch.arguments_for(tensors)
    with _ffi_stream_context(index):
        if launch.prepare is not None:
            launch.prepare(*arguments)
        launch.entry(*arguments)
    return order, lse_ordered, scan, counts


def zero_dlogp_partition_reference(
    labels: torch.Tensor,
    dlogp: torch.Tensor,
    row_index: torch.Tensor,
    lse: torch.Tensor,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The PyTorch form of :func:`zero_dlogp_partition` (any device; no host synchronization): ``(row_index, lse, scan,
    counts)``.

    A valid row whose upstream gradient ``dlogp`` (FP32 ``[T]``) is ``+0`` or ``-0`` contributes exact zeros to ``dX``
    and ``dW``; the others (a NaN included) are the live rows.  The returned ``row_index`` holds the forward's valid
    rows (``row_index``, ascending) with the live rows first, each group in ascending order, and ``lse`` the forward's
    compact statistic in that order; ``scan`` is the inclusive int32 count of live rows over the ``[T]`` rows (the
    fused dX finalize's scan: ``scan[r] - 1`` is the position of a live row ``r``, and the rows past the live count are
    never read), ``counts`` the int32 per-chunk count of live rows of the ``chunk``-row chunks (the ``_dcnt`` forms'
    ``CNT``), at least one in the first chunk (whose weight-gradient GEMM stores its first K step)."""
    rows = int(row_index.numel())
    device = dlogp.device
    live = torch.logical_and(labels.reshape(-1) >= 0, dlogp != 0)
    scan = torch.cumsum(live, 0, dtype=torch.int32)
    count = scan[-1:]
    # live rows up to and including each valid row
    before = scan.index_select(0, row_index)
    position = torch.arange(rows, dtype=torch.int32, device=device)
    # a live row goes to its rank among the live rows, a zero-gradient row after every live row, by its rank among them
    dest = torch.where(
        live.index_select(0, row_index), before - 1, count + position - before
    )
    dest = dest.long()
    order = torch.empty_like(row_index).index_copy_(0, dest, row_index)
    lse_ordered = torch.empty_like(lse).index_copy_(0, dest, lse)
    num_chunks = -(-rows // int(chunk))
    starts = torch.arange(
        0, num_chunks * int(chunk), int(chunk), dtype=torch.int32, device=device
    )
    counts = torch.clamp(count - starts, 0, int(chunk))
    counts[:1].clamp_(min=1)
    return order, lse_ordered, scan, counts


def zero_weight_split(
    problem: "Problem", compact: bool, min_rows: Optional[int] = None
) -> bool:
    """Whether a loss-entry call splits off its zero-weight rows: a compacted policy call of at least
    ``min_rows`` rows (``None`` = :func:`zero_weight_min_rows`)."""
    floor = zero_weight_min_rows() if min_rows is None else int(min_rows)
    return bool(
        compact
        and problem.entry == "loss"
        and problem.objective == "policy"
        and int(problem.num_rows) >= floor
    )


def uncompacted_max_rows() -> int:
    """The largest call (rows ``T``, one chunk) that runs uncompacted without a valid-row count
    (:func:`set_uncompacted_max_rows`; default :data:`UNCOMPACTED_MAX_ROWS_DEFAULT`; ``0`` = never)."""
    return _UNCOMPACTED_MAX_ROWS["rows"]


def set_uncompacted_max_rows(rows: int) -> int:
    """Set the largest call (rows ``T``) that runs uncompacted without a valid-row count for the following calls of
    this process (``0``: none; every compacted call counts its rows) and return the previous value.  Read once per
    call; the binding key holds the resolved path (its valid-row count)."""
    if isinstance(rows, bool) or int(rows) != rows or int(rows) < 0:
        raise ValueError(
            f"uncompacted max rows must be a non-negative integer, got {rows!r}"
        )
    previous = _UNCOMPACTED_MAX_ROWS["rows"]
    _UNCOMPACTED_MAX_ROWS["rows"] = int(rows)
    return previous


def small_t_keep() -> bool:
    """Does a small log-probability call (:data:`SMALL_T_KEEP_MAX_ROWS`) where a backward can follow run the kept-logits
    entry (:func:`set_small_t_keep`; default on)?"""
    return _SMALL_T_KEEP["on"]


def set_small_t_keep(on: bool) -> bool:
    """Set whether a log-probability call of at most :data:`SMALL_T_KEEP_MAX_ROWS` rows where a backward can follow runs
    the kept-logits entry, for the following calls of this process, and return the previous value."""
    if not isinstance(on, bool):
        raise ValueError(f"small_t_keep takes a bool, got {on!r}")
    previous = _SMALL_T_KEEP["on"]
    _SMALL_T_KEEP["on"] = on
    return previous


def small_pad_rows(entry: str = "loss", device: Optional[torch.device] = None) -> int:
    """Rows a small call of ``entry`` (``"loss"``, or ``"logprob"`` for the log-probability entries, plain and with a
    temperature) on ``device`` (default: the current CUDA device) is padded to: the value of :func:`set_small_pad_rows`
    when one is set, else :data:`SMALL_PAD_ROWS_LOGPROB_BY_ARCH` for a log-probability call on a listed architecture,
    else :data:`SMALL_PAD_ROWS_DEFAULT` (0 = no padding)."""
    if entry not in ("loss", "logprob"):
        raise ValueError(
            f"small_pad_rows entry must be 'loss' or 'logprob', got {entry!r}"
        )
    forced = _SMALL_PAD_ROWS["rows"]
    if forced is not None:
        return forced
    arch = arch_for(device) if entry == "logprob" else None
    if arch is not None:
        return SMALL_PAD_ROWS_LOGPROB_BY_ARCH.get(arch, SMALL_PAD_ROWS_DEFAULT)
    return SMALL_PAD_ROWS_DEFAULT


def set_small_pad_rows(rows: Optional[int]) -> Optional[int]:
    """Pad every small call to ``rows`` rows (0 = off; ``None`` = the per-entry / per-architecture defaults of
    :func:`small_pad_rows`) for the following calls of this process, and return the previous setting."""
    if rows is not None and (
        isinstance(rows, bool) or not isinstance(rows, int) or rows < 0
    ):
        raise ValueError(
            f"small_pad_rows takes a non-negative int or None, got {rows!r}"
        )
    previous = _SMALL_PAD_ROWS["rows"]
    _SMALL_PAD_ROWS["rows"] = rows
    return previous


def _small_call_rows(
    X,
    labels,
    backend: str,
    chunk_size,
    compact_rows,
    *,
    objective: Optional[str] = None,
    per_row: tuple = (),
    logprob: bool = False,
) -> int:
    """Rows a call of an autograd entry stages as a small call (:data:`UNCOMPACTED_MAX_ROWS_DEFAULT`), from host values
    alone: ``0`` for any other call, else ``T`` or the padded row count (:func:`small_pad_rows` of the loss entry, or of
    the log-probability entries when ``logprob``, on ``X``'s device).  A small call has a BF16 ``[T, H]`` ``X`` and
    int64 ``[T]`` ``labels`` on one CUDA device, ``backend="cake"``, ``compact_rows`` resolving to compacted, ``0 < T <=
    min(uncompacted_max_rows(), chunk_size)``, every tensor of ``per_row`` (policy ``infer_logp`` / ``loss_weights``, a
    temperature) an FP32 ``[T]`` tensor on that device that requires no gradient, and is not a policy call the
    zero-weight split would serve (``T >= zero_weight_min_rows()``); malformed inputs are left to the entry's
    validation.  The padded count applies when ``T < small_pad_rows(...) <= min(uncompacted_max_rows(), chunk_size)`` (a
    policy call: also below the zero-weight floor)."""
    if (
        backend != "cake"
        or X.ndim != 2
        or X.dtype != torch.bfloat16
        or not X.is_cuda
        or labels.ndim != 1
        or labels.dtype != torch.int64
        or labels.device != X.device
        or int(labels.shape[0]) != int(X.shape[0])
    ):
        return 0
    try:
        chunk = int(chunk_size)
    except (TypeError, ValueError):
        return 0
    T = int(X.shape[0])
    limit = min(_UNCOMPACTED_MAX_ROWS["rows"], chunk)
    if not 0 < T <= limit or not _resolve_compact(compact_rows):
        return 0
    for t in per_row:
        if isinstance(t, torch.Tensor) and (
            t.ndim != 1
            or int(t.shape[0]) != T
            or t.dtype != torch.float32
            or t.device != X.device
            or t.requires_grad
        ):
            return 0
    floor = zero_weight_min_rows() if objective == "policy" else None
    if floor is not None and floor <= T:
        return 0
    pad = small_pad_rows("logprob" if logprob else "loss", X.device)
    if T < pad <= limit and (floor is None or pad < floor):
        return pad
    return T


class _SmallCallStage(torch.autograd.Function):
    """The staged inputs of a small call (:func:`_small_call_rows`) as ``rows`` rows: ``(X, labels, infer_logp,
    loss_weights, temperature)``.  ``X``'s ignored rows (``labels < 0``) become zero rows and ``rows - T`` zero rows
    follow; ``labels`` get ``rows - T`` ``IGNORE_INDEX`` rows; ``infer_logp`` / ``loss_weights`` get zeros on the
    ignored and the padded rows; a temperature tensor gets ones on the padded rows.  Every staged tensor is a view of
    ONE allocation (a repeated call meets the same addresses when the caching allocator returns the same block); an
    input left as is (``labels`` / a temperature of an unpadded call, ``None``, a number) comes back unchanged.  Device
    work only: one comparison, one zeroed word, one ``where`` per masked tensor and the padded rows' fills -- no host
    synchronization.  The backward hands ``dX``'s first ``T`` rows back (an ignored row's ``dX`` row is an exact zero
    either way); the other outputs are not differentiable."""

    @staticmethod
    def forward(ctx, X, labels, rows, infer_logp, loss_weights, temperature):
        T, H = int(X.shape[0]), int(X.shape[1])
        pad = rows > T
        tau = temperature if pad and isinstance(temperature, torch.Tensor) else None
        regions = [("x", 2 * rows * H), ("valid", T), ("zero", 4)]
        if pad:
            regions.append(("labels", 8 * rows))
        for name, t in (("infer", infer_logp), ("weights", loss_weights), ("tau", tau)):
            if isinstance(t, torch.Tensor):
                regions.append((name, 4 * rows))
        offsets, total = {}, 0
        for name, nbytes in regions:
            offsets[name] = (total, nbytes)
            total += _align(nbytes)
        block = torch.empty((total,), dtype=torch.uint8, device=X.device)

        def view(name, dtype):
            start, nbytes = offsets[name]
            return block[start : start + nbytes].view(dtype)

        valid = torch.ge(labels, 0, out=view("valid", torch.bool))
        # ``where(..., out=)`` takes no number as ``other``: a zeroed word of the block is that zero.
        zero = view("zero", torch.uint8).zero_()
        xs = view("x", torch.bfloat16).view(rows, H)
        torch.where(valid.unsqueeze(1), X, zero[:2].view(torch.bfloat16), out=xs[:T])
        lab = labels
        if pad:
            xs[T:].zero_()
            lab = view("labels", torch.int64)
            lab[:T].copy_(labels)
            lab[T:].fill_(IGNORE_INDEX)

        def staged(name, t, fill, masked):
            if not isinstance(t, torch.Tensor):
                return t
            out = view(name, torch.float32)
            if masked:
                torch.where(valid, t, zero.view(torch.float32), out=out[:T])
            else:
                out[:T].copy_(t)
            if pad:
                out[T:].fill_(fill)
            return out

        infer = staged("infer", infer_logp, 0.0, True)
        weights = staged("weights", loss_weights, 0.0, True)
        tau_s = staged("tau", tau, 1.0, False) if tau is not None else temperature
        ctx.rows = T
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(
            *[t for t in (lab, infer, weights, tau_s) if isinstance(t, torch.Tensor)]
        )
        return xs, lab, infer, weights, tau_s

    @staticmethod
    def backward(ctx, dxs, *_):
        if dxs is None:
            return (None,) * 6
        dx = dxs if int(dxs.shape[0]) == ctx.rows else dxs[: ctx.rows]
        return (dx,) + (None,) * 5


def gradient_rows(
    labels: torch.Tensor, loss_weights: torch.Tensor, *, mask: bool = False
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """:func:`valid_rows` of a policy call that splits off its zero-weight rows: ``(row_index, row_valid,
    zero_index)``.  ``row_index`` / ``row_valid`` are :func:`valid_rows`' outputs for the gradient rows ``G = (labels
    >= 0) & (loss_weights != 0)`` (a NaN weight is in ``G``; ``None`` when every row is in ``G``), ``zero_index`` the
    ascending int64 index of the zero-weight rows ``Z = (labels >= 0) & (loss_weights == 0)`` (``None`` when there is
    none).  One device synchronization, as :func:`valid_rows`: a CUDA call of at most :data:`HOST_COUNT_MAX_ROWS` rows
    reads both masks back and forms both indices on the host; a longer one reads both counts back in one 8-byte copy
    and forms both indices on the device (:func:`gradient_index`)."""
    T = int(labels.numel())
    if T == 0:
        return None, None, None
    device = labels.device
    if T <= HOST_COUNT_MAX_ROWS and labels.is_cuda:
        valid = labels >= 0
        grad = valid & (loss_weights != 0)
        both = torch.stack((grad, valid & ~grad)).cpu()
        g_host, z_host = both[0].nonzero().squeeze(1), both[1].nonzero().squeeze(1)
        if int(g_host.numel()) == T:
            return None, None, None
        # pageable host memory: the copies are staged before they return, so the host indices may go right away
        return (
            g_host.to(device, non_blocking=True),
            grad if mask else None,
            z_host.to(device, non_blocking=True) if z_host.numel() else None,
        )
    masks = torch.empty((2 * T,), dtype=torch.bool, device=device)
    pos = torch.empty((2 * T,), dtype=torch.int32, device=device)
    idx_full = torch.empty((T,), dtype=torch.int64, device=device)
    ar = (
        _hidden_count_arange(
            int(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            ),
            T,
        )
        if device.type == "cuda"
        else torch.arange(1, T + 1, dtype=torch.int32, device=device)
    )
    gradient_index(labels, loss_weights, masks=masks, pos=pos, idx_full=idx_full, ar=ar)
    count, total = (int(v) for v in pos[T - 1 :: T].tolist())  # the one synchronization
    zeros = total - count
    zero_index = (
        torch.searchsorted(pos[T:], ar[count : count + zeros]) if zeros else None
    )
    if count == T:
        return None, None, None
    return idx_full[:count], (masks[:T] if mask else None), zero_index


def scatter_rows(
    rows: torch.Tensor, idx: Optional[torch.Tensor], num_rows: int
) -> torch.Tensor:
    """Rows of a compacted ``[T_v, ...]`` tensor back into a fresh ``[num_rows, ...]`` tensor with exact zeros
    elsewhere (``idx=None``: the identity, the tensor itself)."""
    if idx is None:
        return rows
    out = torch.zeros(
        (int(num_rows),) + tuple(rows.shape[1:]), dtype=rows.dtype, device=rows.device
    )
    if idx.numel():
        out.index_copy_(0, idx, rows)
    return out


def _gather_rows(values: dict[str, Any]) -> None:
    """Gather ``src[idx]`` into the preallocated ``out`` (host side; a chunk's ``X`` rows into the reusable ``x_c``
    buffer, or a caller's ``[T]`` operand into its compact ``[T_v]`` form)."""
    torch.index_select(values["src"], 0, values["idx"], out=values["out"])


# Host-side steps of the chunk loop (torch operators over the bound tensors, no kernel of the program, no allocation).
HOST_STAGES: dict[str, Callable[[dict[str, Any]], None]] = {
    "gather_rows": _gather_rows,
}


def stages_for_entry(
    entry: str,
    *,
    need_dx: bool = True,
    need_dw: bool = True,
    grad_weight_dtype=torch.bfloat16,
    fuse_dw_cast: Optional[bool] = None,
    num_chunks: Optional[int] = None,
    dx_cast: bool = True,
) -> tuple[str, ...]:
    """Stages an entry point launches (``need_dx`` / ``need_dw`` drop the GEMMs of frozen inputs).

    ``fuse_dw_cast`` (default :func:`fuse_dw_cast_default`): the weight gradient's last chunk comes out of the fused
    GEMM ``gemm_dw_cast_bf16`` / ``gemm_dw_cast_f32`` instead of the flat ``scale_cast``; ``gemm_dw_acc`` then
    accumulates the chunks before it and is absent when ``num_chunks`` is given as 1 (unknown: kept).
    ``dx_cast=False`` leaves the flat ``dX`` cast out: the caller finalizes ``dx_acc`` itself (the fused dX finalize of
    a compacted log-probability backward scatters it in one pass, :func:`finalize_dx`).  The K-sliced reduction form
    :data:`DX_RED_STAGE` of the dX GEMM and its slab sum :data:`DX_SLAB_SUM_STAGE` are per-chunk instances
    (:attr:`Plan.stages`), not listed here.
    """
    if entry not in ENTRIES:
        raise ValueError(f"entry must be one of {ENTRIES}")
    fp32 = entry == "loss" and grad_weight_dtype == torch.float32
    stages = ["gemm_logits", "row_finalize"]
    if entry == "loss":
        stages.append("loss_reduce")
    else:
        stages.append("gemm_logits_nostats")
    if need_dx or need_dw:
        stages.append("row_grad")
    if need_dx:
        stages += ["gemm_dx"] + (["scale_cast_bf16"] if dx_cast else [])
    if need_dw:
        if _resolve_fuse(fuse_dw_cast):
            if num_chunks is None or int(num_chunks) > 1:
                stages.append("gemm_dw_acc")
            stages.append("gemm_dw_cast_f32" if fp32 else "gemm_dw_cast_bf16")
        else:
            stages += ["gemm_dw_acc", "scale_cast_f32" if fp32 else "scale_cast_bf16"]
    return tuple(s for s in STAGES if s in stages)


def generated_program_available(
    device: Optional[torch.device] = None,
    *,
    entry: str = "loss",
) -> bool:
    """True when this checkout registers a contract program for ``device`` with every stage the entry point needs --
    the base stages plus every variant the per-chunk instance rules can select for the record's geometry and
    architecture (:func:`reachable_variants`) and, when the record serves K-sliced dX chunks, their slab sum
    (:data:`DX_SLAB_SUM_STAGE`)."""
    arch = arch_for(device)
    if arch is None:
        return False
    try:
        name = select_module(arch)
    except NotImplementedError:
        return False
    record = MODULES[name]
    if str(record.get("abi", ABI_CONTRACT)) != ABI_CONTRACT:
        return False
    stages = registered_stages(name)
    needed = set(stages_for_entry(entry)) | set(
        stages_for_entry(entry, grad_weight_dtype=torch.float32)
    )
    needed |= reachable_variants(needed, arch, dx_max_slices(stages))
    if dx_max_slices(stages) > 1:
        # every K-sliced dX launch is followed by its slab sum
        needed.add(DX_SLAB_SUM_STAGE)
    if dx_finalize_default():
        needed |= set(DX_FINALIZE_STAGES)
    if hidden_count_default() and toolchain_runs_hidden_count(arch):
        needed.add(HIDDEN_COUNT_STAGE)
    return all(stage in stages for stage in needed)


# ---------------------------------------------------------------------------
# Validation and planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """Validated geometry and options of one call."""

    num_rows: int  # T
    hidden: int  # H
    vocab: int  # V
    chunk: int  # C
    ld_x: int  # leading stride of X as launched (after a copy: H)
    x_copy: bool  # X is materialized contiguously (leading stride not a TMA pitch)
    objective: str
    loss_div: Optional[float]
    grad_weight_dtype: torch.dtype
    entry: str = "loss"
    # the call has a temperature (log-probability entry; :class:`_TemperatureProblem`, :func:`_with_temperature`): the
    # row vectors include the compact FP32 ``temperature`` every logits / row stage of the chunk loop reads
    # (:func:`stage_values`); the value never enters a plan or a key.  A class attribute, not a field: a call without
    # a temperature constructs exactly the problem it constructed before
    temperature: ClassVar[bool] = False

    @property
    def mode(self) -> int:
        """``mode`` of ``row_finalize`` in the forward."""
        if self.entry == "logprob":
            return MODE_NONE
        return MODE_CE if self.objective == "ce" else MODE_POLICY


class _TemperatureProblem(Problem):
    """The :class:`Problem` of a log-probability call with a temperature (:func:`_with_temperature`)."""

    temperature: ClassVar[bool] = True


def _with_temperature(problem: Problem) -> Problem:
    """``problem`` of a call that has a temperature (:attr:`Problem.temperature`; log-probability entry only)."""
    if problem.entry != "logprob":
        raise ValueError("temperature belongs to the log-probability entry")
    return _TemperatureProblem(*(getattr(problem, f.name) for f in fields(Problem)))


def _assert_on_device(condition: torch.Tensor, message: str) -> None:
    try:
        torch._assert_async(condition, message)
    except TypeError:  # a PyTorch without the message argument
        torch._assert_async(condition)


def _positive_scalar(value, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        if value.is_cuda:
            raise ValueError(
                f"{name} must be a Python number or a CPU scalar (a device value would synchronize)"
            )
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return value


class _Temperature(NamedTuple):
    """A validated temperature (:func:`_resolve_temperature`): ``kind`` ``"scalar"`` with ``value`` the FP32 value as
    a Python float, or ``"tensor"`` with ``value`` the caller's FP32 ``[T]`` tensor (contiguous).  Passed on as is by
    a later :func:`_resolve_temperature` of the same call (no second check)."""

    kind: str
    value: Any


# ``_resolve_temperature`` results: ``None`` (the default path, no temperature) or a :class:`_Temperature`.
TemperatureForm = Optional[_Temperature]


def _fp32(value: float) -> float:
    """``value`` rounded once to FP32 (round to nearest even), as a Python float; ``OverflowError`` past FP32's
    range."""
    return struct.unpack("f", struct.pack("f", value))[0]


def _resolve_temperature(
    temperature, T: int, device: torch.device, *, check: bool = False
) -> TemperatureForm:
    """Validate the ``temperature`` keyword of the log-probability entry for a call of ``T`` rows on ``device`` and
    return its form: ``None`` for the default path (no temperature, or a scalar whose FP32 value is exactly 1.0: the
    call runs the path without a temperature, bitwise today's), ``_Temperature("scalar", s)`` (``s`` the FP32 value as
    a Python float) or ``_Temperature("tensor", tau)`` (the per-token FP32 ``[T]`` tensor, made contiguous: a copy
    keeps the bits).  An already resolved :class:`_Temperature` is returned as is (its tensor's shape and device are
    checked again; its values are not).

    * A Python ``int`` / ``float`` or a 0-dim CPU tensor: a positive finite number whose FP32 rounding is still
      positive and finite (no underflow to 0, no overflow to inf); ``bool`` is rejected.  A 0-dim CUDA tensor is
      rejected (reading it would synchronize).
    * A 1-D tensor: ``float32``, shape ``(T,)``, on ``device``; any stride.  Every entry -- ignored rows included --
      must be positive and finite.  On CUDA the values are checked by a device-side assertion queued before every
      kernel that reads them (no host synchronization; like a wrong ``num_valid_rows``, a failing assertion ends the
      CUDA context); ``check=True`` reads the check back once and raises ``ValueError`` instead.  CPU tensors are
      checked on the host.
    * A tensor with ``requires_grad=True`` is rejected: the temperature is not differentiable (pass
      ``temperature.detach()``).
    * Anything else is a ``ValueError``.
    """
    if temperature is None:
        return None
    if type(temperature) in (float, int) and temperature == 1:
        return None  # the default value (``bool`` is not ``int`` here)
    if isinstance(temperature, _Temperature):  # resolved earlier in the same call
        if temperature.kind == "tensor" and (
            tuple(temperature.value.shape) != (int(T),)
            or temperature.value.device != device
        ):
            raise ValueError(
                "the resolved temperature does not match the call's rows or device"
            )
        return temperature
    if isinstance(temperature, torch.Tensor):
        if temperature.requires_grad:
            raise ValueError(
                "temperature is not differentiable: pass temperature.detach() (a tensor with requires_grad=True was given)"
            )
        if temperature.ndim == 0:
            if temperature.device.type != "cpu":
                raise ValueError(
                    "a scalar temperature must be a Python number or a CPU scalar: pass a Python number; reading a "
                    f"{temperature.device.type} value would synchronize with the device"
                )
            value = _positive_scalar(temperature, "temperature")
        elif temperature.ndim == 1:
            if temperature.dtype != torch.float32:
                raise ValueError(
                    f"a per-token temperature must be a float32 tensor, got dtype {temperature.dtype}"
                )
            if int(temperature.shape[0]) != int(T):
                raise ValueError(
                    f"a per-token temperature must have shape ({int(T)},) (one value per row of X), got "
                    f"{tuple(temperature.shape)}"
                )
            if temperature.device != device:
                raise ValueError(
                    f"a per-token temperature must be on X's device {device}, got {temperature.device}"
                )
            tau = temperature.contiguous()
            if int(T) > 0:
                # NaN fails ``> 0``, +inf fails ``< inf``
                ok = (tau > 0) & (tau < math.inf)
                if check or tau.device.type != "cuda":
                    # the one synchronization of the checked path
                    if not bool(ok.all()):
                        raise ValueError(
                            "temperature values must be positive and finite on every row (ignored rows included)"
                        )
                else:
                    _assert_on_device(
                        ok.all(),
                        "temperature values must be positive and finite on every row (ignored rows included)",
                    )
            return _Temperature("tensor", tau)
        else:
            raise ValueError(
                f"temperature must be a number or a float32 [T] tensor, got a tensor of shape {tuple(temperature.shape)}"
            )
    elif isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError(
            "temperature must be a positive number or a float32 [T] tensor, got "
            f"{type(temperature).__name__}"
        )
    else:
        try:
            value = _positive_scalar(temperature, "temperature")
        except OverflowError:  # an int past the float range
            raise ValueError(
                "temperature is not a positive finite FP32 number (an int too large for a float)"
            ) from None
    try:
        s = _fp32(value)
    except OverflowError:
        s = math.inf
    if not 0.0 < s < math.inf:
        raise ValueError(
            f"temperature {value!r} is not a positive finite FP32 number (it rounds to {s!r})"
        )
    return None if s == 1.0 else _Temperature("scalar", s)


def _temperature_rows(
    form: TemperatureForm,
    row_index: Optional[torch.Tensor],
    rows: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """The compact FP32 ``[rows]`` temperature the kernels read at ``row0 + r`` (``None`` for the default path): the
    caller's ``[T]`` tensor itself when the chunk loop is not compacted, its ``index_select`` by ``row_index`` when it
    is; ``torch.full`` of the FP32 value for a scalar (the same bits in every row, so a scalar ``s`` and
    ``torch.full((T,), s)`` run the same kernels on the same operands)."""
    if form is None:
        return None
    kind, value = form
    if kind == "scalar":
        return torch.full((int(rows),), value, dtype=torch.float32, device=device)
    return value if row_index is None else value.index_select(0, row_index)


# This package's directory, the flashinfer package's directory (its facade and decorators) when the package is
# flashinfer's ``experimental/cake/cake_lm_head_nondet_train`` (a vendored copy skips its own directory only), and torch's
# (``Function.apply``, ``once_differentiable``, checkpoint and autocast wrappers): the frames a warning of this
# package skips (:func:`_caller_stacklevel`).
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_FLASHINFER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_PACKAGE_DIR)))
_TORCH_DIR = os.path.dirname(os.path.abspath(torch.__file__))
_SKIPPED_DIRS = tuple(
    d + os.sep
    for d in (_PACKAGE_DIR, _FLASHINFER_DIR, _TORCH_DIR)
    if d != _FLASHINFER_DIR or os.path.basename(d) == "flashinfer"
)


def _caller_stacklevel() -> int:
    """The ``stacklevel`` of the nearest frame outside :data:`_SKIPPED_DIRS` for a warning issued by the function that
    calls this one: the alert is issued at several depths (an eager entry, a runner, an autograd ``Function``, behind
    the facade and torch's wrappers), so a fixed ``stacklevel`` could name a frame of the package or of torch.  A
    backward alert raised on the autograd engine's thread has no caller frame outside them and falls back to 2 (a
    frame of the package)."""
    frame = inspect.currentframe()
    level = 0
    while frame is not None:
        parent = frame.f_back
        level += 1
        if parent is not None and not os.path.abspath(
            parent.f_code.co_filename
        ).startswith(_SKIPPED_DIRS):
            return level
        frame = parent
    return 2


def _alert_nondeterministic(entry: str, kind: str) -> None:
    """PyTorch's alert for an operation without a deterministic implementation, when torch's deterministic
    algorithms are enabled: a ``RuntimeError``, or a ``UserWarning`` under
    ``torch.use_deterministic_algorithms(True, warn_only=True)`` (the call then runs on in this package);
    nothing otherwise.  ``kind`` (``"loss"``, ``"logprob"``, ``"support_logprob"`` or ``"support_kl"``): the entry of
    ``flashinfer.cake_lm_head_nondet_train`` the call serves; the alert names
    its deterministic counterpart ``flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_<kind>``, and nothing
    falls back to it.  Raised by :func:`_alert_unordered_dx` (the dX GEMM's K-sliced reduction) and by the
    candidate-support backward (its dense dX at more than one K slice), before the call's first such launch.  The
    alert is part of this package's contract: the package makes no run-to-run bitwise promise for its gradients
    (the deterministic package makes it), although both reductions have one fixed order and repeat bit for bit.  The
    alert is raised here, not through a private PyTorch hook (PyTorch exposes no public Python entry
    for it)."""
    if not torch.are_deterministic_algorithms_enabled():
        return
    counterpart = f"flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_{kind}"
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(
            f"{entry} does not have a deterministic implementation, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'; "
            f"{counterpart} is its deterministic counterpart",
            stacklevel=_caller_stacklevel(),
        )
        return
    raise RuntimeError(
        f"{entry} does not have a deterministic implementation, but you set "
        f"'torch.use_deterministic_algorithms(True)'; {counterpart} is its deterministic counterpart, or use "
        "the 'warn_only=True' option if that is acceptable for your application"
    )


def _alert_unordered_dx(plan: "Plan", entry: str, kind: str) -> None:
    """:func:`_alert_nondeterministic` for a call whose ``plan`` launches the dX GEMM's K-sliced reduction
    (:data:`DX_RED_STAGE`, a chunk of more than one slice: :attr:`Plan.dx_sliced`), whose slab sum adds the slices in
    slice order and stores the sum (one fixed order: ``dX`` repeats bit for bit; the alert is kept by this package's
    contract, :func:`_alert_nondeterministic`).  A plan without one (the one-slice dX store, the weight-gradient GEMMs,
    the casts) does not alert.  Each entry calls it where its plan is final (after a
    valid-row count, before the call's first dX launch).  Torch's flag is read first: outside its deterministic
    algorithms the call does no other host work here.  ``kind``: as in :func:`_alert_nondeterministic`."""
    if torch.are_deterministic_algorithms_enabled() and plan.dx_sliced:
        _alert_nondeterministic(entry, kind)


def _check_temperature_arms(
    has_temperature: bool, logprob_lbl=None, logprob_dz=None
) -> None:
    """A temperature runs the log-probability entry's logits GEMM arms (the label-column forward and the ``dz``
    recompute, :data:`TEMPERATURE_STAGES`): with one, an explicit ``logprob_lbl=False`` / ``logprob_dz=False`` is
    rejected on every backend."""
    if not has_temperature:
        return
    for name, value in (("logprob_lbl", logprob_lbl), ("logprob_dz", logprob_dz)):
        if value is not None and not bool(value):
            raise ValueError(
                f"{name}=False is not available with a temperature: the temperature forms are the logits GEMM arms "
                "(leave the switch at its default)"
            )


def validate_lm_head_inputs(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div=None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    entry: str = "loss",
    geometry: Geometry = DEFAULT_GEOMETRY,
) -> Problem:
    """Shape / dtype / stride validation shared by the entry points.

    Returns the :class:`Problem`.  Device placement is checked separately so
    this runs on host tensors.  ``T == 0`` is accepted (the eager entry points
    return zeros for it without launching).
    """
    if entry not in ENTRIES:
        raise ValueError(f"entry must be one of {ENTRIES}")
    if X.ndim != 2 or X.dtype != torch.bfloat16:
        raise ValueError("X must be a BF16 [T, H] tensor")
    if W.ndim != 2 or W.dtype != torch.bfloat16:
        raise ValueError("W must be a BF16 [V, H] tensor")
    if not W.is_contiguous():
        raise ValueError("W must be contiguous")
    num_rows, hidden = int(X.shape[0]), int(X.shape[1])
    vocab, w_hidden = int(W.shape[0]), int(W.shape[1])
    if w_hidden != hidden:
        raise ValueError(
            f"W has {w_hidden} columns, X has {hidden}: the hidden sizes differ"
        )
    if hidden < 1 or hidden % geometry.hidden_multiple:
        raise ValueError(
            f"H must be a positive multiple of {geometry.hidden_multiple}, got {hidden}"
        )
    if vocab < 1 or vocab % geometry.vocab_multiple:
        raise ValueError(
            f"V must be a positive multiple of {geometry.vocab_multiple}, got {vocab}"
        )
    if hidden % geometry.dx_item_cols or hidden % geometry.dw_item_cols:
        raise ValueError(
            f"H must be a multiple of the {geometry.dx_item_cols}-column dX and {geometry.dw_item_cols}-column "
            f"weight-gradient work items, got {hidden}"
        )
    if vocab % geometry.logits_item_cols:
        raise ValueError(
            f"V must be a multiple of the {geometry.logits_item_cols}-column logits work item, got {vocab}"
        )
    if X.stride(1) != 1:
        raise ValueError("X rows must be contiguous (stride(1) == 1)")
    ld_x = int(X.stride(0))
    if num_rows > 1 and ld_x < hidden:
        raise ValueError("X must be row-major with a leading stride of at least H")
    # The TMA operands need a 16 B row pitch and base; otherwise X is materialized contiguously (reported).
    x_copy = bool(
        num_rows and (ld_x < hidden or ld_x % geometry.ld_multiple or X.data_ptr() % 16)
    )
    if x_copy or num_rows == 0:
        ld_x = hidden
    if labels.ndim != 1 or int(labels.shape[0]) != num_rows:
        raise ValueError("labels must be a [T] tensor")
    if labels.dtype != torch.int64:
        raise ValueError("labels must be int64 (-100 marks ignored rows)")
    if (
        isinstance(chunk_size, bool)
        or int(chunk_size) != chunk_size
        or int(chunk_size) < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    if int(chunk_size) > 65535:
        raise ValueError(
            "chunk_size must not exceed 65535 (the row kernels index the chunk row by blockIdx.y)"
        )
    if grad_weight_dtype not in GRAD_WEIGHT_DTYPES:
        raise ValueError("grad_weight_dtype must be torch.bfloat16 or torch.float32")
    if entry == "logprob":
        if (
            objective != "ce"
            or loss_div is not None
            or infer_logp is not None
            or loss_weights is not None
        ):
            raise ValueError(
                "the log-probability entry point takes no objective arguments"
            )
        return Problem(
            num_rows,
            hidden,
            vocab,
            int(chunk_size),
            ld_x,
            x_copy,
            "ce",
            None,
            grad_weight_dtype,
            entry,
        )
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if objective == "ce":
        if loss_div is None:
            raise ValueError(
                "objective='ce' needs loss_div (a caller-supplied positive scalar)"
            )
        if infer_logp is not None or loss_weights is not None:
            raise ValueError("infer_logp and loss_weights belong to objective='policy'")
        loss_div = _positive_scalar(loss_div, "loss_div")
    else:
        if loss_div is not None:
            raise ValueError("loss_div belongs to objective='ce'")
        if infer_logp is None or loss_weights is None:
            raise ValueError("objective='policy' needs infer_logp and loss_weights")
        for name, t in (("infer_logp", infer_logp), ("loss_weights", loss_weights)):
            if t.ndim != 1 or int(t.shape[0]) != num_rows or t.dtype != torch.float32:
                raise ValueError(f"{name} must be an FP32 [T] tensor")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
    return Problem(
        num_rows,
        hidden,
        vocab,
        int(chunk_size),
        ld_x,
        x_copy,
        objective,
        loss_div,
        grad_weight_dtype,
        entry,
    )


def plan_chunks(
    num_rows: int, chunk: int, *, fold: bool = False
) -> tuple[tuple[int, int], ...]:
    """``(row0, rows_c)`` of every chunk in launch order: full chunks first, the
    tail (``T % C``) last, never dropped, never padded; empty for ``T == 0``.
    ``fold`` (the short-tail fold, :func:`tail_fold_eligible`): a two-chunk plan
    whose last chunk has fewer than ``chunk / 2`` rows runs as one chunk of
    fewer than ``1.5 * chunk`` rows (at most :data:`CHUNK_MAX_ROWS`)."""
    if int(chunk) < 1:
        raise ValueError("chunk must be positive")
    chunks = tuple(
        (row0, min(int(chunk), int(num_rows) - row0))
        for row0 in range(0, int(num_rows), int(chunk))
    )
    if (
        fold
        and len(chunks) == 2
        and 2 * chunks[1][1] < int(chunk)
        and int(num_rows) <= CHUNK_MAX_ROWS
    ):
        return ((0, int(num_rows)),)
    return chunks


# The most rows of one chunk: the row kernels index the chunk row by ``blockIdx.y`` (:func:`validate_lm_head_inputs`
# caps ``chunk_size`` the same way).
CHUNK_MAX_ROWS = 65535


def tail_fold_eligible(num_rows: int, chunk: int) -> bool:
    """Whether the default plan of a call of ``num_rows`` rows (``T``, before compaction) folds its short tail
    (:func:`plan_chunks` ``fold``): a call of fewer than :data:`HIDDEN_COUNT_MIN_CHUNKS` chunks of ``T``.  The tail's
    own logits GEMM, ``row_grad``, dX GEMM and weight-gradient GEMM -- each a short ``M`` (or ``K``) that reads all of
    ``W`` or of the ``[V, H]`` weight gradient -- run inside the chunk before it, and a call that becomes one chunk has
    no FP32 ``[V, H]`` weight-gradient accumulator.  The chunk buffers then hold the folded chunk (fewer than
    ``1.5 * chunk`` rows, :func:`chunk_rows_max`).  A call of three chunks or more keeps the shipped chunks: the hidden
    valid-row count serves those calls and its prologue buffers hold ``min(chunk, T)`` rows (:func:`_hidden_count`).
    The kept-logits plan folds by the same rule (:func:`loop_chunks`), and so does the zero-gradient row skip's counted
    plan: its live counts follow the plan's chunks (:func:`chunk_stride`)."""
    rows = int(num_rows)
    return rows > 0 and -(-rows // int(chunk)) < HIDDEN_COUNT_MIN_CHUNKS


def chunk_rows_max(num_rows: int, chunk: int, *, fold: bool = False) -> int:
    """Rows of the chunk buffers of a plan over ``num_rows`` loop rows: its largest chunk (``min(chunk, rows)``; a
    folded plan's one chunk, :func:`plan_chunks`), at least 1."""
    chunks = plan_chunks(max(int(num_rows), 1), chunk, fold=fold)
    return max(rows_c for _, rows_c in chunks)


def fold_rows_max(num_rows: int, chunk: int) -> int:
    """Rows of the chunk buffers that hold every plan of a call of ``num_rows`` rows (``T``) over any loop-row count up
    to ``T``: ``min(chunk, T)``, or the largest folded chunk (:func:`loop_chunks`: fewer than ``1.5 * chunk`` rows, at
    most ``T`` and :data:`CHUNK_MAX_ROWS`) where :func:`tail_fold_eligible` allows the fold."""
    rows, chunk = max(int(num_rows), 1), int(chunk)
    if rows > chunk and tail_fold_eligible(rows, chunk):
        return min(rows, (3 * chunk - 1) // 2, CHUNK_MAX_ROWS)
    return min(chunk, rows)


def chunk_stride(chunks: tuple[tuple[int, int], ...], chunk: int) -> int:
    """The row stride of a plan's ``chunks``: ``chunk``, or the rows of a folded plan's one chunk (:func:`plan_chunks`
    ``fold``).  The zero-gradient row skip counts its live rows per chunk of this stride (:func:`zero_dlogp_partition`)."""
    return max([int(chunk)] + [int(rows_c) for _, rows_c in chunks])


def loop_chunks(
    num_rows: int,
    rows: int,
    chunk: int,
    *,
    hidden_count: bool = False,
    fold: bool = True,
) -> tuple[tuple[int, int], ...]:
    """The chunks of :func:`make_plan`'s plan -- the default and the kept-logits forms alike -- of a call of
    ``num_rows`` rows (``T``) whose chunk loop runs ``rows`` rows (the valid rows when compacted): :func:`plan_chunks`
    with the short-tail fold where :func:`tail_fold_eligible` allows it.  ``hidden_count`` (a plan of the hidden
    valid-row count) and ``fold=False`` keep the chunks of ``chunk`` rows.  The one fold rule of the package: a form
    whose plan takes these chunks keeps the other forms' bits."""
    return plan_chunks(
        rows,
        chunk,
        fold=bool(fold and not hidden_count and tail_fold_eligible(num_rows, chunk)),
    )


# The most rows one chunk of a kept-logits plan may have (:data:`CHUNK_MAX_ROWS`): the kept plan folds by the one
# rule of :func:`loop_chunks`.
KEPT_CHUNK_MAX_ROWS = CHUNK_MAX_ROWS


@dataclass(frozen=True)
class Plan:
    """The chunk schedule of one binding."""

    problem: Problem
    chunks: tuple[tuple[int, int], ...]
    need_dx: bool
    need_dw: bool
    geometry: Geometry = DEFAULT_GEOMETRY
    dx_slices: tuple[
        int, ...
    ] = ()  # K-slice count of the dX GEMM per chunk (empty = one slice everywhere; > 1 = the reduction form)
    valid_rows: Optional[int] = (
        None  # rows of the compacted chunk loop (the valid rows); None = every row of the caller
    )
    fuse_dw_cast: bool = False  # the last chunk's weight-gradient GEMM runs in the backward with the fused scale + cast
    variants: tuple[
        ChunkVariants, ...
    ] = ()  # per chunk: the GEMM instance knobs the rules resolved (empty = defaults)
    dx_finalize: bool = False  # the call's resolved fused dX finalize (the compacted dX output boundary's one-pass scatter)
    dx_cast: bool = True  # the backward casts ``dx_acc`` into ``dx_out`` (False: the caller finalizes ``dx_acc``, :func:`finalize_dx`)
    # the hidden valid-row count (eager entry points): chunk 0's row gather and logits GEMM ran in device-count form before
    # the count was known (:func:`_hidden_count_begin`), so the forward / recompute key sequences start at chunk 0's row
    # kernels; ``hidden_logits_group_m`` is the raster rule's height at the buffer extent, ``hidden_stats`` the GEMM form
    hidden_count: bool = False
    hidden_logits_group_m: Optional[int] = None
    hidden_stats: bool = True
    # the log-probability entry's logits GEMM arms (:data:`LOGPROB_DZ_STAGE`, :data:`LOGPROB_LBL_STAGE`): the backward's
    # recompute writes ``dz`` itself (no ``row_grad``), the forward writes the statistics and the label column only
    logprob_dz: bool = False
    logprob_lbl: bool = False
    # the zero-gradient row skip of the log-probability backward (:data:`DCNT_BASES`): the dz arm, the K-sliced dX
    # reductions and their slab sums and the weight-gradient GEMMs run their ``_dcnt`` forms, bounded by the per-chunk
    # device counts ``t["dcnt"]`` of the rows the call ordered first (:func:`zero_dlogp_partition`)
    dcnt: bool = False
    # the cast at the last live chunk (:data:`DW_LIVE_CAST_BASES`, ``_lc``) of a counted plan of two chunks or more with
    # the fused bf16 weight-gradient cast: every accumulate but the last chunk's runs its ``_lc`` form and the deferred
    # cast its ``_lc`` form.  Set by the log-probability backward's eager skip path only (:func:`make_plan` resolves
    # it, ``_skip_forms``); a runner of the public entry points never runs it, so its ``dw_acc`` keeps its meaning
    live_cast: bool = False
    # the plan runs the temperature forms (:class:`_TemperaturePlan`, :func:`make_plan_temperature`).  A class
    # attribute, not a field: a plan without a temperature is built as before
    temperature: ClassVar[bool] = False
    # the kept-logits form of the log-probability entry (:class:`_KeptPlan`); not a field: the default plan is unchanged
    logprob_keep: ClassVar[bool] = False

    def variants_of(self, index: int) -> ChunkVariants:
        return self.variants[index] if self.variants else ChunkVariants()

    def hidden_logits_stage(self) -> str:
        """The device-count logits GEMM form chunk 0 ran on the hidden valid-row count path (its raster height
        ``hidden_logits_group_m`` is a launch scalar)."""
        return stage_variant(
            "gemm_logits" if self.hidden_stats else "gemm_logits_nostats", count=True
        )

    def logits_stage(self, index: int, *, stats: bool = True) -> str:
        """The logits GEMM form of chunk ``index`` (``stats=False``: the log-probability backward's recompute).  One
        form per entry: the chunk's raster height is a launch scalar (:attr:`ChunkVariants.logits_group_m`)."""
        self.variants_of(index)  # the chunk exists
        return stage_variant("gemm_logits" if stats else "gemm_logits_nostats")

    def forward_logits_stage(self, index: int) -> str:
        """The forward's logits GEMM of chunk ``index``: the label-column arm of the log-probability entry
        (:attr:`logprob_lbl`), else :meth:`logits_stage`."""
        stage = self.logits_stage(index)
        return LOGPROB_LBL_STAGE if self.logprob_lbl else stage

    def recompute_stage(self, index: int) -> str:
        """The log-probability backward's recompute GEMM of chunk ``index``: the ``dz`` arm (:attr:`logprob_dz`), else
        the stats-free logits GEMM that ``row_grad`` follows."""
        stage = self.logits_stage(index, stats=False)
        return (
            stage_variant(LOGPROB_DZ_STAGE, dcnt=self.dcnt)
            if self.logprob_dz
            else stage
        )

    @property
    def row_finalize_stage(self) -> str:
        """The forward's row kernel (its temperature form in a :class:`_TemperaturePlan`)."""
        return "row_finalize"

    def dx_stage_of(self, index: int) -> str:
        """The dX GEMM form of chunk ``index``: the plain store (one slice) or the K-sliced reduction
        :data:`DX_RED_STAGE`, its tile, ring depth, split tail or swap (the slice count is a launch scalar,
        :meth:`dx_slices_of`)."""
        v = self.variants_of(index)
        return dx_stage(
            self.dx_slices_of(index),
            v.dx_tile_n,
            v.dx_stages,
            split=v.dx_split is not None,
            swab=v.dx_swab,
            dcnt=self.dcnt,
        )

    def dx_slab_sum_stage_of(self, index: int) -> Optional[str]:
        """The slab sum launched right after the dX GEMM of chunk ``index`` (:func:`dx_slab_sum_stage`: its counted form
        after a counted reduction), or ``None`` for a chunk of one slice (the plain store has no slab)."""
        if self.dx_slices_of(index) <= 1:
            return None
        return dx_slab_sum_stage(self.dx_stage_of(index))

    def dw_acc_stage(self, index: int) -> str:
        """The weight-gradient accumulate form of chunk ``index`` (blocked when the block rule set a ``group_n``; the
        raster height and the block width are launch scalars)."""
        v = self.variants_of(index)
        return stage_variant(
            "gemm_dw_acc",
            blocked=v.dw_group_n is not None,
            epi_store=v.dw_epi_store,
            live_cast=self.live_cast,
            dcnt=self.dcnt,
        )

    @property
    def compact(self) -> bool:
        """The chunk loop runs over the valid rows only (gather / scatter around the kernels)."""
        return self.valid_rows is not None

    @property
    def rows(self) -> int:
        """Rows the chunk loop processes: the valid rows when compacted, else ``T``."""
        return int(self.valid_rows) if self.compact else int(self.problem.num_rows)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def num_tiles(self) -> int:
        return -(-self.problem.vocab // self.geometry.stats_tile)

    def dx_slices_of(self, index: int) -> int:
        return int(self.dx_slices[index]) if self.dx_slices else 1

    @property
    def dx_slab_bytes(self) -> int:
        """Bytes of the ordered dX slab the call's largest K-sliced chunk writes (:func:`dx_slab_bytes`; ``0`` without
        dX or a chunk of more than one slice): one buffer of the call serves every chunk (:func:`_dx_ordered_values`)."""
        if not self.need_dx:
            return 0
        g = self.geometry
        return max(
            (
                dx_slab_bytes(
                    self.dx_slices_of(i),
                    g.row_tiles(rows_c, g.dx_cluster_ctas),
                    self.problem.hidden,
                )
                for i, (_, rows_c) in enumerate(self.chunks)
            ),
            default=0,
        )

    @property
    def dx_sliced(self) -> bool:
        """The call launches the dX GEMM's K-sliced reduction (:data:`DX_RED_STAGE`): it needs dX and has a chunk of
        more than one slice."""
        return self.need_dx and any(
            self.dx_slices_of(i) > 1 for i in range(self.num_chunks)
        )

    @property
    def dw_deferred(self) -> bool:
        """The last chunk's weight-gradient GEMM runs in the backward with the upstream scale and the output cast fused
        into its epilogue (:data:`DW_CAST_STAGES`); the forward accumulates the chunks before it only."""
        return bool(self.need_dw and self.fuse_dw_cast and self.num_chunks > 0)

    @property
    def dw_acc_needed(self) -> bool:
        """The FP32 ``[V, H]`` accumulator exists: every unfused plan, and a fused plan of more than one chunk."""
        return bool(self.need_dw and not (self.dw_deferred and self.num_chunks == 1))

    @property
    def last_chunk(self) -> tuple[int, int]:
        """``(row0, rows_c)`` of the last chunk (``(0, 0)`` without chunks)."""
        return self.chunks[-1] if self.chunks else (0, 0)

    @property
    def stages(self) -> tuple[str, ...]:
        base = stages_for_entry(
            self.problem.entry,
            need_dx=self.need_dx,
            need_dw=self.need_dw,
            grad_weight_dtype=self.problem.grad_weight_dtype,
            fuse_dw_cast=self.fuse_dw_cast,
            num_chunks=self.num_chunks,
            dx_cast=self.dx_cast,
        )
        used = {s for s in base if s not in GEMM_STAGES}
        for index in range(self.num_chunks):
            if "gemm_logits" in base:
                used.add(self.forward_logits_stage(index))
            if "gemm_logits_nostats" in base:
                used.add(self.recompute_stage(index))
            if "gemm_dx" in base:
                used.add(self.dx_stage_of(index))
                slab_sum = self.dx_slab_sum_stage_of(index)
                if slab_sum is not None:
                    used.add(slab_sum)  # the K-sliced chunk's slab sum
            if "gemm_dw_acc" in base and not (
                self.dw_deferred and index == self.num_chunks - 1
            ):
                used.add(self.dw_acc_stage(index))
        if self.dw_deferred:
            used.add(self.dw_cast_stage)
        if self.hidden_count:
            used.add(HIDDEN_COUNT_STAGE)
            used.add(self.hidden_logits_stage())
        elif self.logprob_dz:
            used.discard("row_grad")  # every chunk's recompute writes ``dz`` itself
        return tuple(s for s in STAGES if s in used)

    @property
    def dw_cast_stage(self) -> str:
        """The stage that writes the final ``dW``: the fused GEMM of the deferred last chunk, else the flat cast."""
        fp32 = (
            self.problem.entry == "loss"
            and self.problem.grad_weight_dtype == torch.float32
        )
        if self.dw_deferred:
            v = self.variants_of(self.num_chunks - 1)
            return stage_variant(
                "gemm_dw_cast_f32" if fp32 else "gemm_dw_cast_bf16",
                blocked=fp32
                and v.dw_group_n is not None,  # the bf16 cast keeps the 1-D raster
                epi_store="tma"
                if v.dw_cast_tma
                else None,  # the TMA form of the bf16 cast (dw_cast_tma_rule)
                live_cast=self.live_cast,
                dcnt=self.dcnt,
            )
        return "scale_cast_f32" if fp32 else "scale_cast_bf16"


class _TemperaturePlan(Plan):
    """A :class:`Plan` with the temperature forms (:data:`TEMPERATURE_STAGES`, :func:`make_plan_temperature`: a
    log-probability plan of a registered program whose call has a temperature, which then runs both logits GEMM
    arms): the forward's logits GEMM, chunk 0's device-count GEMM, ``row_finalize`` and the backward's ``dz``
    recompute run their ``_tau`` forms.  The reference path keeps the plan without them and reads the temperature
    from the host values (:attr:`Problem.temperature`).  Under the zero-gradient row skip (:attr:`Plan.dcnt`,
    :func:`make_plan_temperature`) this plan is the mixed one: the ``dz`` recompute runs its uncounted temperature form
    over every row of the chunk (a zero-gradient row's ``dz`` is zero: ``RN(0 / tau) = 0``), the K-sliced dX
    reductions and the weight-gradient GEMMs their ``_dcnt`` forms; :class:`_CountedTemperaturePlan` counts the
    ``dz`` recompute too."""

    temperature: ClassVar[bool] = True
    # the plan class without the temperature forms (:meth:`without_temperature`)
    plain: ClassVar[type] = Plan
    # whether the ``dz`` recompute runs its counted temperature form under the skip (:class:`_CountedTemperaturePlan`)
    counted_dz: ClassVar[bool] = False

    def without_temperature(self) -> Plan:
        """The same plan without the temperature forms: the twin on which :func:`stage_values_temperature` checks a
        temperature form.  The twin of the mixed plan's uncounted ``dz`` temperature form is the uncounted
        ``gemm_logits_dz``, so the twin keeps :attr:`Plan.dcnt` off; the counted dX / dW forms of a plan under the
        zero-gradient row skip are checked on the plan itself (:class:`_CountedTemperaturePlan` keeps it on)."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values["dcnt"] = False
        values["live_cast"] = False
        return self.plain(**values)

    def hidden_logits_stage(self) -> str:
        """The temperature form of :meth:`Plan.hidden_logits_stage`."""
        return stage_form(
            "gemm_logits" if self.hidden_stats else "gemm_logits_nostats",
            count=True,
            temperature=True,
        )

    def forward_logits_stage(self, index: int) -> str:
        """The temperature form of the label-column arm (:attr:`logprob_lbl`)."""
        self.logits_stage(index)  # the chunk exists
        if not self.logprob_lbl:
            raise ValueError("a temperature plan runs the label-column arm")
        return LOGPROB_LBL_TAU_STAGE

    def recompute_stage(self, index: int) -> str:
        """The temperature form of the ``dz`` arm (:attr:`logprob_dz`): the counted one under the skip of a
        :class:`_CountedTemperaturePlan`."""
        self.logits_stage(index, stats=False)  # the chunk exists
        if not self.logprob_dz:
            raise ValueError("a temperature plan runs the dz arm")
        return (
            LOGPROB_DZ_DCNT_TAU_STAGE
            if self.counted_dz and self.dcnt
            else LOGPROB_DZ_TAU_STAGE
        )

    @property
    def row_finalize_stage(self) -> str:
        return ROW_FINALIZE_TAU_STAGE

    @property
    def stages(self) -> tuple[str, ...]:
        """:attr:`Plan.stages` with the temperature forms, in the launch order of
        :data:`cake_jit.STAGES_WITH_TEMPERATURE`."""
        base = stages_for_entry(
            self.problem.entry,
            need_dx=self.need_dx,
            need_dw=self.need_dw,
            grad_weight_dtype=self.problem.grad_weight_dtype,
            fuse_dw_cast=self.fuse_dw_cast,
            num_chunks=self.num_chunks,
            dx_cast=self.dx_cast,
        )
        used = {s for s in base if s not in GEMM_STAGES}
        if "row_finalize" in used:
            used.discard("row_finalize")
            used.add(ROW_FINALIZE_TAU_STAGE)
        for index in range(self.num_chunks):
            if "gemm_logits" in base:
                used.add(self.forward_logits_stage(index))
            if "gemm_logits_nostats" in base:
                used.add(self.recompute_stage(index))
            if "gemm_dx" in base:
                used.add(self.dx_stage_of(index))
                slab_sum = self.dx_slab_sum_stage_of(index)
                if slab_sum is not None:
                    used.add(slab_sum)  # the K-sliced chunk's slab sum
            if "gemm_dw_acc" in base and not (
                self.dw_deferred and index == self.num_chunks - 1
            ):
                used.add(self.dw_acc_stage(index))
        if self.dw_deferred:
            used.add(self.dw_cast_stage)
        if self.hidden_count:
            used.add(HIDDEN_COUNT_STAGE)
            used.add(self.hidden_logits_stage())
        if self.logprob_dz:
            # every chunk's recompute writes ``dz`` itself; a hidden-count plan keeps chunk 0's ``row_grad`` for its
            # recount, which a call with a temperature never takes (no temperature form of ``row_grad``)
            used.discard("row_grad")
        return tuple(s for s in STAGES_WITH_TEMPERATURE if s in used)


class _CountedTemperaturePlan(_TemperaturePlan):
    """A :class:`_TemperaturePlan` under the zero-gradient row skip (:attr:`Plan.dcnt`) whose ``dz`` recompute runs
    the counted temperature form (:data:`LOGPROB_DZ_DCNT_TAU_STAGE`): no work on the chunk's rows past its count,
    the same ``dz`` bits on the rows before it.  Its twin without the temperature forms keeps :attr:`Plan.dcnt` (the
    twin of the counted temperature form is ``gemm_logits_dz_dcnt``)."""

    counted_dz: ClassVar[bool] = True

    def without_temperature(self) -> Plan:
        """The same plan without the temperature forms, :attr:`Plan.dcnt` kept (the counted twin)."""
        return self.plain(**{f.name: getattr(self, f.name) for f in fields(self)})


class _KeptPlan(Plan):
    """The kept-logits plan of the log-probability entry (:func:`cake_lm_head_nondet_train_logprob_keep`): a :class:`Plan`
    whose forward runs the plain ``gemm_logits`` per chunk (neither logits GEMM arm) and whose backward recomputes
    nothing -- ``row_grad`` forms each chunk's ``dz`` over the chunk's kept logits (:func:`recompute_keys`).  The same
    fields; only :func:`make_plan` with ``logprob_keep=True`` builds it."""

    logprob_keep: ClassVar[bool] = True

    @property
    def stages(self) -> tuple[str, ...]:
        n = self.num_chunks
        recompute = {self.recompute_stage(i) for i in range(n)} - {
            self.forward_logits_stage(i) for i in range(n)
        }
        if self.hidden_count:
            recompute.discard(self.hidden_logits_stage())
        return tuple(s for s in super().stages if s not in recompute)


def make_plan(
    problem: Problem,
    *,
    need_dx: bool,
    need_dw: bool,
    geometry: Geometry = DEFAULT_GEOMETRY,
    dx_max_slices: int = 1,
    num_sms: int = 1,
    valid_rows: Optional[int] = None,
    dx_resident: Optional[int] = None,
    fuse_dw_cast: bool = False,
    arch: Optional[str] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    dx_finalize: bool = False,
    dx_cast: bool = True,
    hidden_count: bool = False,
    hidden_stats: bool = True,
    logprob_dz: bool = False,
    logprob_lbl: bool = False,
    dx_slice_policy: Optional[str] = None,
    dx_red_min_rows: Optional[int] = None,
    logprob_keep: bool = False,
    dcnt: bool = False,
    counted_tma_cast: bool = True,
    live_cast: bool = False,
    tail_fold: bool = True,
) -> Plan:
    """The chunk schedule; ``dx_max_slices`` > 1 (the K-sliced reduction form registered) picks each chunk's slice count
    with :func:`dx_slices` under ``dx_slice_policy`` (``None`` = :func:`dx_slice_policy`; ``dx_resident`` = the device's
    co-resident dX clusters, :func:`cluster_resident`), one slice below ``dx_red_min_rows`` rows (``None`` =
    :func:`dx_red_min_rows`); ``valid_rows`` (compaction)
    chunks that many rows instead of ``T``; ``fuse_dw_cast`` defers the last chunk's weight-gradient GEMM to the
    backward with the fused scale + cast epilogue.  ``arch`` (a registered program's architecture) applies the
    per-chunk instance rules (:func:`chunk_variants`; ``None`` = the reference path, base stages only) and
    ``gemm_tuning`` pins knobs explicitly (already validated by the caller or :func:`_resolve_gemm_tuning`);
    ``dx_finalize`` records the call's fused dX finalize (the compacted dX output boundary) and ``dx_cast=False`` leaves
    ``dx_acc`` uncast for the caller (:func:`finalize_dx`).  ``hidden_count`` (the eager entry points' hidden valid-row
    count, ``arch`` required): chunk 0's gather and logits GEMM already ran in device-count form (``hidden_stats``: with
    the statistics) at the buffer extent ``min(chunk, T)``, whose raster-rule height the plan records.  ``logprob_dz`` /
    ``logprob_lbl`` (a log-probability plan of a record that registers them) run the entry's logits GEMM arms
    (:attr:`Plan.logprob_dz`, :attr:`Plan.logprob_lbl`).  ``logprob_keep`` (a log-probability plan, a registered
    program): the kept-logits plan (:class:`_KeptPlan`), which runs neither arm.  ``dcnt`` (a log-probability plan
    with the dz arm, a registered program): the zero-gradient row skip's ``_dcnt`` forms (:attr:`Plan.dcnt`), each
    chunk's dX at the architecture's counted slice count at least (:func:`counted_dx_slices`; the slice rule's counts
    when the plan with those would have no counted forms), the
    last chunk's bf16 fused cast in the TMA epilogue's counted form where :func:`dw_cast_tma_rule` takes the TMA
    epilogue (``counted_tma_cast=False``, a record without that form: the shipped cast's counted form instead); off when
    a chunk takes a form without one (an explicitly pinned weight-gradient TMA epilogue or swapped dX tile).
    ``live_cast`` (the log-probability backward's eager skip path only): a counted plan of two chunks or more with the
    fused bf16 weight-gradient cast runs the cast at the last live chunk (:attr:`Plan.live_cast`); ignored otherwise.
    ``tail_fold`` (default on): the short-tail fold of :func:`loop_chunks` where :func:`tail_fold_eligible` allows it
    (the kept-logits plan included), except in a hidden-count plan."""
    if (logprob_dz or logprob_lbl or logprob_keep) and problem.entry != "logprob":
        raise ValueError("the logits GEMM arms belong to the log-probability entry")
    if logprob_keep:
        if arch is None:
            raise ValueError(
                "the kept logits (logprob_keep) belong to a registered program (arch required)"
            )
        logprob_dz = logprob_lbl = (
            False  # the whole z_c is stored and row_grad forms dz from it
        )
    rows = problem.num_rows if valid_rows is None else int(valid_rows)
    chunks = loop_chunks(
        problem.num_rows,
        rows,
        problem.chunk,
        hidden_count=hidden_count,
        fold=tail_fold,
    )
    tuning = gemm_tuning or {}
    dx_tuning = tuning.get("dx", {})
    slices = plan_dx_slices(
        problem,
        chunks,
        need_dx=need_dx,
        tuning=tuning,
        dx_max_slices=dx_max_slices,
        num_sms=num_sms,
        geometry=geometry,
        dx_resident=dx_resident,
        arch=arch,
        policy=dx_slice_policy,
        red_min_rows=dx_red_min_rows,
    )
    if need_dx and dx_tuning.get("split"):
        # the ``_split`` grid carries ``W + F`` clusters and runs those from ``W - F`` on as the half items of the last
        # ``F`` wide items: ``F < 0`` leaves wide items unrun, ``F > W`` maps half items to negative items
        explicit_f = int(dx_tuning["split"])
        for i, (_, rows_c) in enumerate(chunks):
            wide, _ = dx_wide_items(
                rows_c, problem.hidden, num_sms, slices[i] if slices else 1, geometry
            )
            if not 1 <= explicit_f <= wide:
                raise ValueError(
                    f"gemm_tuning dx split={explicit_f}: the split-tail form runs 1 .. W of a chunk's W wide dX items "
                    f"as half items; the chunk of {rows_c} rows has W = {wide}"
                )

    def variants_of(slices_: tuple[int, ...]) -> tuple[ChunkVariants, ...]:
        if arch is None and not tuning:
            return ()
        return tuple(
            chunk_variants(
                rows_c,
                problem.hidden,
                num_sms,
                slices_[i] if slices_ else 1,
                geometry,
                arch,
                tuning,
                num_chunks=len(chunks),
                ce_loss=problem.entry == "loss" and problem.objective == "ce",
            )
            for i, (_, rows_c) in enumerate(chunks)
        )

    def has_counted_forms(variants_: tuple[ChunkVariants, ...]) -> bool:
        # none for an explicitly pinned swapped dX tile or weight-gradient TMA epilogue (the swap rule itself serves
        # the cross-entropy loss only)
        return not any(
            v.dx_swab or v.dw_epi_store not in (None, "redsm") for v in variants_
        )

    # the zero-gradient row skip's counted forms
    dcnt = bool(
        dcnt
        and arch is not None
        and problem.entry == "logprob"
        and logprob_dz
        and not logprob_keep
        and not hidden_count
    )
    variants: Optional[tuple[ChunkVariants, ...]] = None
    if dcnt:
        # the counted plan's slices (:data:`DX_COUNTED_MIN_SLICES_BY_ARCH`) when that plan keeps the counted forms,
        # else the slice rule's
        raised = counted_dx_slices(
            slices,
            arch=arch,
            dx_max_slices=dx_max_slices,
            k_iters=int(problem.vocab) // int(geometry.k_block),
            tuning=tuning,
        )
        if raised != slices:
            raised_variants = variants_of(raised)
            if has_counted_forms(raised_variants):
                slices, variants = raised, raised_variants
    if variants is None:
        variants = variants_of(slices)
    dcnt = dcnt and has_counted_forms(variants)
    if (
        variants
        and (not dcnt or counted_tma_cast)
        and need_dw
        and fuse_dw_cast
        and not (problem.entry == "loss" and problem.grad_weight_dtype == torch.float32)
        and "epi_store" not in tuning.get("dw", {})
        and dw_cast_tma_rule(len(chunks), chunks[-1][1], arch)
    ):
        # the deferred last chunk's bf16 cast takes the TMA epilogue (a counted plan its counted form,
        # DW_CAST_TMA_DCNT_STAGE, unless ``counted_tma_cast`` is off); an explicit ``epi_store`` keeps the shipped
        # cast (the same dW bits)
        variants = variants[:-1] + (replace(variants[-1], dw_cast_tma=True),)
    hidden_group: Optional[int] = None
    if hidden_count:
        if arch is None:
            raise ValueError(
                "the hidden valid-row count belongs to a registered program (arch required)"
            )
        raster = raster_variant(
            problem.hidden, min(int(problem.chunk), int(problem.num_rows)), arch
        )
        hidden_group = None if raster is None else raster[0]
    return (_KeptPlan if logprob_keep else Plan)(
        problem,
        chunks,
        bool(need_dx),
        bool(need_dw),
        geometry,
        slices,
        None if valid_rows is None else int(valid_rows),
        bool(fuse_dw_cast),
        variants,
        bool(dx_finalize),
        bool(dx_cast),
        bool(hidden_count),
        hidden_group,
        bool(hidden_stats),
        logprob_dz=bool(logprob_dz),
        logprob_lbl=bool(logprob_lbl),
        dcnt=dcnt,
        # the cast at the last live chunk: a counted plan of two chunks or more whose bf16 weight gradient comes out of
        # the fused cast (the log-probability entry's dW is bf16)
        live_cast=bool(
            live_cast
            and dcnt
            and need_dw
            and fuse_dw_cast
            and len(chunks) >= 2
            and not (
                problem.entry == "loss" and problem.grad_weight_dtype == torch.float32
            )
        ),
    )


def make_plan_temperature(
    problem: Problem, *, counted_dz: bool = True, **kw: Any
) -> Plan:
    """:func:`make_plan` of a log-probability call with a temperature (:attr:`Problem.temperature`) on a registered
    program (``arch``) running both logits GEMM arms (``logprob_dz`` and ``logprob_lbl``): the plan runs the
    temperature forms (:class:`_TemperaturePlan`).  ``dcnt`` (the zero-gradient row skip, :func:`make_plan`'s rule
    and fallbacks): the K-sliced dX reductions and the weight-gradient GEMMs run their ``_dcnt`` forms over the rows
    the call ordered first, and the ``dz`` recompute its counted temperature form (``counted_dz``,
    :class:`_CountedTemperaturePlan`: no work past the chunk's count) or, with ``counted_dz=False``, the uncounted one
    over every row (the mixed plan: a zero-gradient row's ``dz`` is ``RN(0 / tau) * (...) = 0``).  No kept-logits
    plan."""
    if not (
        problem.temperature
        and kw.get("arch") is not None
        and kw.get("logprob_dz")
        and kw.get("logprob_lbl")
    ):
        raise ValueError(
            "the temperature forms belong to a log-probability plan of a registered program with a temperature, "
            "running both logits GEMM arms (logprob_dz and logprob_lbl)"
        )
    if kw.get("logprob_keep"):
        raise ValueError(
            "the temperature forms have no kept-logits plan (logprob_keep)"
        )
    plan = make_plan(problem, **kw)
    kind = _CountedTemperaturePlan if counted_dz and plan.dcnt else _TemperaturePlan
    return kind(**{f.name: getattr(plan, f.name) for f in fields(plan)})


# ---------------------------------------------------------------------------
# Workspace and memory accounting
# ---------------------------------------------------------------------------


def _align(nbytes: int) -> int:
    return (nbytes + WORKSPACE_ALIGN - 1) // WORKSPACE_ALIGN * WORKSPACE_ALIGN


def workspace_layout(
    num_rows: int,
    vocab: int,
    chunk: int,
    *,
    stats_tile: int = GEOMETRY_DEFAULTS["stats_tile"],
    entry: str = "loss",
    tma_workspace_bytes: int = 0,
    scratch_bytes: int = 0,
    hidden: int = 0,
    compact: bool = False,
) -> dict:
    """Byte ``(offset, size)`` of every workspace region plus ``"total"``.

    ``compact`` adds the BF16 ``x_c [min(T, C), H]`` buffer the valid rows of
    ``X`` are gathered into (``hidden`` = ``H``; ``num_rows`` is then the
    valid-row count).  The ordered K-sliced dX reduction's slab is not a
    region here: the call allocates it with its first K-sliced dX launch
    (:attr:`Plan.dx_slab_bytes`, reported by :func:`memory_report`).

    The workspace holds the temporaries only the launches read: the
    vocabulary buffer ``logits`` (``min(T, C) x V`` BF16, ``z_c`` then ``dz_c``
    in place), its row-statistics partials ``stats``, the chunk-local per-row
    gradient scale ``d`` and loss terms ``term``, the loss accumulator and the
    ``grad_scale`` cell, plus the descriptor storage of pointer-ABI programs
    and kernel-private scratch when a record declares them.  The ``O(T)`` row
    vectors a caller or the autograd graph keeps (``lse``, ``logp``, the
    ``loss`` cell) and the FP32 accumulators ``dX_acc`` / ``dW_acc`` are
    allocated separately so that no returned or saved tensor pins the
    vocabulary buffer.
    """
    rows = min(
        int(chunk), max(int(num_rows), 1)
    )  # the logits workspace never exceeds T rows either
    tiles = -(-int(vocab) // int(stats_tile))
    sizes = [
        ("logits", rows * int(vocab) * 2),
        ("stats", rows * tiles * 2 * 4),
        ("d", rows * 4),
        ("term", rows * 4),
        ("loss_acc", 8),  # FP64 accumulator of the fixed-order loss sum
        ("grad_scale", 4),
    ]
    if compact:
        if int(hidden) < 1:
            raise ValueError(
                "workspace_layout needs hidden > 0 for the gather buffer of the compacted rows"
            )
        sizes.append(("x_c", rows * int(hidden) * 2))
    if scratch_bytes:
        sizes.append(("workspace", int(scratch_bytes)))
    if tma_workspace_bytes:
        sizes.append(("tma_descriptor_workspace", int(tma_workspace_bytes)))
    layout: dict = {}
    offset = 0
    for name, nbytes in sizes:
        layout[name] = (offset, nbytes)
        offset += _align(nbytes)
    layout["total"] = offset
    return layout


def _record_tma_bytes(record: dict[str, Any], stages) -> int:
    return max(
        (int(record[s].get("tma_workspace_bytes", 0)) for s in stages), default=0
    )


def _record_scratch_bytes(record: dict[str, Any], stages) -> int:
    return max((int(record[s].get("workspace_bytes", 0)) for s in stages), default=0)


def memory_report(
    num_rows: int,
    hidden: int,
    vocab: int,
    chunk: int,
    *,
    stats_tile: int = GEOMETRY_DEFAULTS["stats_tile"],
    entry: str = "loss",
    need_dx: bool = True,
    need_dw: bool = True,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    return_logp: bool = False,
    x_copy: bool = False,
    tma_workspace_bytes: int = 0,
    scratch_bytes: int = 0,
    valid_rows: Optional[int] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    hidden_count: bool = False,
    tail_fold: bool = False,
    dx_slab_bytes: int = 0,
) -> dict[str, Any]:
    """The reporting buckets of the memory rule (bytes).

    ``temporary``: the workspace regions of :func:`workspace_layout`, the
    ``O(T)`` row vectors (``lse``; ``logp`` unless it is returned; the
    contiguous ``dlogp`` of the log-probability backward) and a contiguous copy
    of ``X`` when its leading stride forced one; ``outputs``: what the caller
    receives (``loss``, ``logp`` when requested, ``dX`` BF16, ``dW`` in
    ``grad_weight_dtype``); ``accumulators``: the FP32 ``dX_acc`` / ``dW_acc``
    the loss entry keeps alive between forward and backward (the
    log-probability entry allocates them in its backward only) and the row
    statistic the log-probability entry saves; ``weights``: the model tensors
    (``W``, ``X``).  ``vocab_rows_max`` is the largest token extent of any
    vocabulary-sized buffer (``min(T, C)``; the folded chunk of a short-tail
    fold, :func:`chunk_rows_max`).

    ``valid_rows`` (compaction): the chunk loop, the workspace, ``lse``, the
    ``dX`` accumulator and the saved statistic span the valid rows ``T_v``;
    the temporaries gain the BF16 ``x_c`` gather buffer (``gather_bytes``),
    the int64 row index, the compact ``[T_v]`` ``logp`` / ``dlogp`` before and
    after the scatter and the compact BF16 ``dX`` rows the backward casts
    before scattering them into the ``[T, H]`` output; a contiguous copy of
    ``X`` is never needed (the gather output is contiguous).

    ``fuse_dw_cast`` (default :func:`fuse_dw_cast_default`): the last chunk's
    weight-gradient GEMM runs in the backward with the scale + cast fused, so
    the BF16 chunk buffer holding its ``dz`` rows stays alive until the
    backward -- the saved rows are a view of the whole ``[min(T, C), V]``
    buffer, never a copy, so ``saved_dz_bytes`` counts that whole buffer (part
    of the ``temporary`` chunk buffer; the eager entry points allocate it on
    its own from the first call on, so it is also the only storage the saved
    view keeps alive, while a prepared runner keeps it inside its workspace) --
    and a one-chunk call has no FP32 ``dW_acc`` at all.

    ``dx_finalize`` (default :func:`dx_finalize_default`): a compacted call's
    ``dX`` is written in one pass from the accumulator, so the compact BF16
    ``dX`` rows disappear from the temporaries; the bool valid-row mask and its
    int32 scan (``row_valid`` / ``row_scan``, ``[T]`` each) take their place.

    ``hidden_count`` (the eager entry points' hidden valid-row count): the chunk
    buffers (``logits``, ``stats``) and the gather buffer ``x_c`` exist at the
    buffer extent ``min(C, T)`` before the count is known -- also when the
    outcome is the uncompacted plan -- and ``gather_bytes`` counts that buffer.

    ``tail_fold`` (the plan folds its short tail, :func:`make_plan`): where
    :func:`tail_fold_eligible` allows it, a two-chunk call whose last chunk has
    fewer than ``C / 2`` loop rows reports one chunk -- chunk buffers of all its
    loop rows (fewer than ``1.5 C``) and, with the fused cast, no ``dW_acc``.

    ``dx_slab_bytes`` (:attr:`Plan.dx_slab_bytes`): the slab the ordered K-sliced
    dX reduction writes each slice's FP32 tile into, one buffer for the call's
    chunks (``dx_slab`` among the temporaries; ``0``: no chunk of more than
    one slice).

    ``peak_temporary_bytes``: the call's peak of temporaries over its phases --
    ``temporary_bytes`` (the chunk loop) here; a loss call with zero-weight
    policy rows also has the phase of their logp-only pass, whose
    temporaries plus what of the loop's coexists with them may exceed the loop's
    (:func:`_with_zero_weight_rows`, which adds that phase).  It counts
    temporaries only: the accumulators, outputs and weights are reported apart
    and do not exist in every phase (the logp-only pass ahead of the loop runs
    before the FP32 accumulators are allocated), so ``peak_temporary_bytes``
    plus the accumulator bytes is an upper bound of the call's live bytes, not
    their peak.
    """
    compact = valid_rows is not None
    rows = (
        int(valid_rows) if compact else int(num_rows)
    )  # rows the chunk loop processes
    # the short-tail fold of the call's plan (:func:`loop_chunks`)
    chunks = loop_chunks(
        num_rows, rows, chunk, hidden_count=hidden_count, fold=tail_fold
    )
    fold = len(chunks) < len(plan_chunks(rows, chunk))
    extent = chunk_rows_max(rows, chunk, fold=fold)  # rows of the chunk buffers
    deferred = bool(need_dw and _resolve_fuse(fuse_dw_cast) and chunks)
    fused_dx = bool(need_dx and _resolve_dx_finalize(dx_finalize))
    layout = workspace_layout(
        rows,
        vocab,
        max(int(chunk), extent),
        stats_tile=stats_tile,
        entry=entry,
        tma_workspace_bytes=tma_workspace_bytes,
        scratch_bytes=scratch_bytes,
        hidden=hidden,
        compact=compact,
    )
    temporary = {
        name: size
        for name, (_, size) in ((k, v) for k, v in layout.items() if k != "total")
    }
    temporary["lse"] = rows * 4
    if dx_slab_bytes:
        temporary["dx_slab"] = int(
            dx_slab_bytes
        )  # the ordered dX slab of the largest K-sliced chunk (Plan.dx_slab_bytes)
    if entry == "loss" and (compact or not return_logp):
        temporary["logp"] = (
            rows * 4
        )  # compacted: the [T_v] rows before the scatter into the returned [T] logp
    if entry == "logprob":
        temporary["dlogp"] = int(num_rows) * 4
    if compact:
        temporary["row_index"] = rows * 8  # int64 valid-row index
        if entry == "logprob":
            temporary["dlogp_compact"] = rows * 4
        if need_dx and fused_dx:
            temporary["row_valid"] = int(
                num_rows
            )  # the bool valid-row mask the fused dX finalize scatters by
            temporary["row_scan"] = (
                int(num_rows) * 4
            )  # its inclusive int32 scan (formed in the backward)
        elif need_dx:
            temporary["dx_compact"] = (
                rows * int(hidden) * 2
            )  # the cast compact rows, transient before the scatter into dX
    if x_copy and not compact:
        temporary["x_copy"] = int(num_rows) * int(hidden) * 2
    outputs = {
        "loss": 4 if entry == "loss" else 0,
        "logp": int(num_rows) * 4 if (return_logp or entry == "logprob") else 0,
    }
    if need_dx:
        outputs["dX"] = int(num_rows) * int(hidden) * 2
    if need_dw:
        outputs["dW"] = (
            int(vocab)
            * int(hidden)
            * (4 if (entry == "loss" and grad_weight_dtype == torch.float32) else 2)
        )
    accumulators = {}
    if need_dx:
        accumulators["dX_acc"] = rows * int(hidden) * 4
    if need_dw and not (deferred and len(chunks) == 1):
        accumulators["dW_acc"] = int(vocab) * int(hidden) * 4
    if entry == "logprob":
        accumulators["saved_lse"] = rows * 4
    weights = {"W": int(vocab) * int(hidden) * 2, "X": int(num_rows) * int(hidden) * 2}
    if hidden_count:
        return _hidden_count_memory(
            dict(
                temporary_bytes=sum(temporary.values()),
                peak_temporary_bytes=sum(temporary.values()),
                temporary=temporary,
                outputs_bytes=sum(outputs.values()),
                outputs=outputs,
                accumulator_bytes=sum(accumulators.values()),
                accumulators=accumulators,
                weights_bytes=sum(weights.values()),
                weights=weights,
                vocab_rows_max=extent,
                chunk=int(chunk),
                num_chunks=len(chunks),
                compact_rows=compact,
                valid_rows=rows,
                gather_bytes=int(layout["x_c"][1]) if compact else 0,
                fuse_dw_cast=deferred,
                saved_dz_bytes=extent * int(vocab) * 2 if deferred else 0,
                dx_finalize=fused_dx,
                hidden_count=False,
            ),
            num_rows=int(num_rows),
            hidden=int(hidden),
            vocab=int(vocab),
            chunk=int(chunk),
            stats_tile=int(stats_tile),
        )
    return dict(
        temporary_bytes=sum(temporary.values()),
        peak_temporary_bytes=sum(temporary.values()),
        temporary=temporary,
        outputs_bytes=sum(outputs.values()),
        outputs=outputs,
        accumulator_bytes=sum(accumulators.values()),
        accumulators=accumulators,
        weights_bytes=sum(weights.values()),
        weights=weights,
        vocab_rows_max=extent,
        chunk=int(chunk),
        num_chunks=len(chunks),
        compact_rows=compact,
        valid_rows=rows,
        gather_bytes=int(layout["x_c"][1]) if compact else 0,
        fuse_dw_cast=deferred,
        saved_dz_bytes=extent * int(vocab) * 2 if deferred else 0,
        dx_finalize=fused_dx,
        hidden_count=False,
    )


def memory_report_temperature(
    num_rows: int,
    hidden: int,
    vocab: int,
    chunk: int,
    *,
    stats_tile: int = GEOMETRY_DEFAULTS["stats_tile"],
    entry: str = "loss",
    need_dx: bool = True,
    need_dw: bool = True,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    return_logp: bool = False,
    x_copy: bool = False,
    tma_workspace_bytes: int = 0,
    scratch_bytes: int = 0,
    valid_rows: Optional[int] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    hidden_count: bool = False,
    temperature: bool = False,
    tail_fold: bool = False,
    dx_slab_bytes: int = 0,
) -> dict[str, Any]:
    """The reporting buckets of the memory rule (bytes).

    ``temporary``: the workspace regions of :func:`workspace_layout`, the
    ``O(T)`` row vectors (``lse``; ``logp`` unless it is returned; the
    contiguous ``dlogp`` of the log-probability backward) and a contiguous copy
    of ``X`` when its leading stride forced one; ``outputs``: what the caller
    receives (``loss``, ``logp`` when requested, ``dX`` BF16, ``dW`` in
    ``grad_weight_dtype``); ``accumulators``: the FP32 ``dX_acc`` / ``dW_acc``
    the loss entry keeps alive between forward and backward (the
    log-probability entry allocates them in its backward only) and the row
    statistic the log-probability entry saves; ``weights``: the model tensors
    (``W``, ``X``).  ``vocab_rows_max`` is the largest token extent of any
    vocabulary-sized buffer (``min(T, C)``; the folded chunk of a short-tail
    fold, :func:`chunk_rows_max`).

    ``valid_rows`` (compaction): the chunk loop, the workspace, ``lse``, the
    ``dX`` accumulator and the saved statistic span the valid rows ``T_v``;
    the temporaries gain the BF16 ``x_c`` gather buffer (``gather_bytes``),
    the int64 row index, the compact ``[T_v]`` ``logp`` / ``dlogp`` before and
    after the scatter and the compact BF16 ``dX`` rows the backward casts
    before scattering them into the ``[T, H]`` output; a contiguous copy of
    ``X`` is never needed (the gather output is contiguous).

    ``fuse_dw_cast`` (default :func:`fuse_dw_cast_default`): the last chunk's
    weight-gradient GEMM runs in the backward with the scale + cast fused, so
    the BF16 chunk buffer holding its ``dz`` rows stays alive until the
    backward -- the saved rows are a view of the whole ``[min(T, C), V]``
    buffer, never a copy, so ``saved_dz_bytes`` counts that whole buffer (part
    of the ``temporary`` chunk buffer; the eager entry points allocate it on
    its own from the first call on, so it is also the only storage the saved
    view keeps alive, while a prepared runner keeps it inside its workspace) --
    and a one-chunk call has no FP32 ``dW_acc`` at all.

    ``dx_finalize`` (default :func:`dx_finalize_default`): a compacted call's
    ``dX`` is written in one pass from the accumulator, so the compact BF16
    ``dX`` rows disappear from the temporaries; the bool valid-row mask and its
    int32 scan (``row_valid`` / ``row_scan``, ``[T]`` each) take their place.

    ``hidden_count`` (the eager entry points' hidden valid-row count): the chunk
    buffers (``logits``, ``stats``) and the gather buffer ``x_c`` exist at the
    buffer extent ``min(C, T)`` before the count is known -- also when the
    outcome is the uncompacted plan -- and ``gather_bytes`` counts that buffer.

    ``tail_fold`` (the plan folds its short tail, :func:`make_plan`): where
    :func:`tail_fold_eligible` allows it, a two-chunk call whose last chunk has
    fewer than ``C / 2`` loop rows reports one chunk -- chunk buffers of all its
    loop rows (fewer than ``1.5 C``) and, with the fused cast, no ``dW_acc``.

    ``dx_slab_bytes`` (:attr:`Plan.dx_slab_bytes`): the slab the ordered K-sliced
    dX reduction writes each slice's FP32 tile into, one buffer for the call's
    chunks (``dx_slab`` among the temporaries; ``0``: no chunk of more than
    one slice).

    ``peak_temporary_bytes``: the call's peak of temporaries over its phases --
    ``temporary_bytes`` (the chunk loop) here; a loss call with zero-weight
    policy rows also has the phase of their logp-only pass, whose
    temporaries plus what of the loop's coexists with them may exceed the loop's
    (:func:`_with_zero_weight_rows`, which adds that phase).  It counts
    temporaries only: the accumulators, outputs and weights are reported apart
    and do not exist in every phase (the logp-only pass ahead of the loop runs
    before the FP32 accumulators are allocated), so ``peak_temporary_bytes``
    plus the accumulator bytes is an upper bound of the call's live bytes, not
    their peak.
    ``temperature`` (log-probability entry with a temperature): the compact
    FP32 temperature the forward saves for the backward, 4 bytes per loop row
    (``accumulators["temperature_rows"]``; on the hidden valid-row count path
    the ``[T]`` buffer allocated before the count, plus the transient ``[T]``
    int64 clamped index of its gather, ``temporary["temperature_index"]``).
    An uncompacted call with a contiguous caller tensor saves that tensor
    itself, so its new retained bytes are 0 although the bucket counts them.
    """
    compact = valid_rows is not None
    rows = (
        int(valid_rows) if compact else int(num_rows)
    )  # rows the chunk loop processes
    # the short-tail fold of the call's plan (:func:`loop_chunks`)
    chunks = loop_chunks(
        num_rows, rows, chunk, hidden_count=hidden_count, fold=tail_fold
    )
    fold = len(chunks) < len(plan_chunks(rows, chunk))
    extent = chunk_rows_max(rows, chunk, fold=fold)  # rows of the chunk buffers
    deferred = bool(need_dw and _resolve_fuse(fuse_dw_cast) and chunks)
    fused_dx = bool(need_dx and _resolve_dx_finalize(dx_finalize))
    layout = workspace_layout(
        rows,
        vocab,
        max(int(chunk), extent),
        stats_tile=stats_tile,
        entry=entry,
        tma_workspace_bytes=tma_workspace_bytes,
        scratch_bytes=scratch_bytes,
        hidden=hidden,
        compact=compact,
    )
    temporary = {
        name: size
        for name, (_, size) in ((k, v) for k, v in layout.items() if k != "total")
    }
    temporary["lse"] = rows * 4
    if dx_slab_bytes:
        temporary["dx_slab"] = int(
            dx_slab_bytes
        )  # the ordered dX slab of the largest K-sliced chunk (Plan.dx_slab_bytes)
    if entry == "loss" and (compact or not return_logp):
        temporary["logp"] = (
            rows * 4
        )  # compacted: the [T_v] rows before the scatter into the returned [T] logp
    if entry == "logprob":
        temporary["dlogp"] = int(num_rows) * 4
    if compact:
        temporary["row_index"] = rows * 8  # int64 valid-row index
        if entry == "logprob":
            temporary["dlogp_compact"] = rows * 4
        if need_dx and fused_dx:
            temporary["row_valid"] = int(
                num_rows
            )  # the bool valid-row mask the fused dX finalize scatters by
            temporary["row_scan"] = (
                int(num_rows) * 4
            )  # its inclusive int32 scan (formed in the backward)
        elif need_dx:
            temporary["dx_compact"] = (
                rows * int(hidden) * 2
            )  # the cast compact rows, transient before the scatter into dX
    if x_copy and not compact:
        temporary["x_copy"] = int(num_rows) * int(hidden) * 2
    outputs = {
        "loss": 4 if entry == "loss" else 0,
        "logp": int(num_rows) * 4 if (return_logp or entry == "logprob") else 0,
    }
    if need_dx:
        outputs["dX"] = int(num_rows) * int(hidden) * 2
    if need_dw:
        outputs["dW"] = (
            int(vocab)
            * int(hidden)
            * (4 if (entry == "loss" and grad_weight_dtype == torch.float32) else 2)
        )
    accumulators = {}
    if need_dx:
        accumulators["dX_acc"] = rows * int(hidden) * 4
    if need_dw and not (deferred and len(chunks) == 1):
        accumulators["dW_acc"] = int(vocab) * int(hidden) * 4
    if entry == "logprob":
        accumulators["saved_lse"] = rows * 4
    if temperature:
        if entry != "logprob":
            raise ValueError("temperature belongs to the log-probability entry")
        accumulators["temperature_rows"] = rows * 4
    weights = {"W": int(vocab) * int(hidden) * 2, "X": int(num_rows) * int(hidden) * 2}
    if hidden_count:
        return _hidden_count_memory_temperature(
            dict(
                temporary_bytes=sum(temporary.values()),
                peak_temporary_bytes=sum(temporary.values()),
                temporary=temporary,
                outputs_bytes=sum(outputs.values()),
                outputs=outputs,
                accumulator_bytes=sum(accumulators.values()),
                accumulators=accumulators,
                weights_bytes=sum(weights.values()),
                weights=weights,
                vocab_rows_max=extent,
                chunk=int(chunk),
                num_chunks=len(chunks),
                compact_rows=compact,
                valid_rows=rows,
                gather_bytes=int(layout["x_c"][1]) if compact else 0,
                fuse_dw_cast=deferred,
                saved_dz_bytes=extent * int(vocab) * 2 if deferred else 0,
                dx_finalize=fused_dx,
                hidden_count=False,
            ),
            num_rows=int(num_rows),
            hidden=int(hidden),
            vocab=int(vocab),
            chunk=int(chunk),
            stats_tile=int(stats_tile),
            temperature=bool(temperature),
        )
    return dict(
        temporary_bytes=sum(temporary.values()),
        peak_temporary_bytes=sum(temporary.values()),
        temporary=temporary,
        outputs_bytes=sum(outputs.values()),
        outputs=outputs,
        accumulator_bytes=sum(accumulators.values()),
        accumulators=accumulators,
        weights_bytes=sum(weights.values()),
        weights=weights,
        vocab_rows_max=extent,
        chunk=int(chunk),
        num_chunks=len(chunks),
        compact_rows=compact,
        valid_rows=rows,
        gather_bytes=int(layout["x_c"][1]) if compact else 0,
        fuse_dw_cast=deferred,
        saved_dz_bytes=extent * int(vocab) * 2 if deferred else 0,
        dx_finalize=fused_dx,
        hidden_count=False,
    )


def kept_memory_report(memory: dict[str, Any], plan: Plan) -> dict[str, Any]:
    """:func:`memory_report` of a kept-logits plan (:class:`_KeptPlan`): the forward keeps every chunk's BF16 logits for
    the backward -- chunk 0 in the chunk buffer (the ``logits`` temporary, alive until the backward), the chunks after
    it in one ``[T_v - rows_0, V]`` buffer (``accumulators["saved_logits"]``); ``saved_logits_bytes`` counts both,
    ``2 * T_v * V``.  The memory rule's chunk bound does not hold for them.  The default report has neither key.  A
    call whose kept buffer is the hidden valid-row count's ``[T, V]`` buffer reports :func:`_hidden_kept_memory_report`
    instead (:func:`forward_logprob_keep`)."""
    if not plan.chunks:
        return dict(memory, saved_logits_bytes=0)
    vocab = plan.problem.vocab
    accumulators = dict(memory["accumulators"])
    accumulators["saved_logits"] = (plan.rows - plan.chunks[0][1]) * vocab * 2
    return dict(
        memory,
        accumulators=accumulators,
        accumulator_bytes=sum(accumulators.values()),
        saved_logits_bytes=plan.rows * vocab * 2,
    )


def _hidden_kept_memory_report(
    memory: dict[str, Any], *, num_rows: int, vocab: int
) -> dict[str, Any]:
    """:func:`kept_memory_report` of a call whose kept buffer is the hidden valid-row count's contiguous ``[T, V]``
    buffer (:attr:`_HiddenCount.kept`, the one-pass kept form): allocated before the count, it holds a row for every
    row of the call -- its first rows are the chunk buffer (the ``logits`` temporary) -- and the backward keeps all of
    it alive, so ``saved_logits_bytes`` is ``2 * T * V`` (the ``T - T_v`` ignored rows included) and
    ``accumulators["saved_logits"]`` the rows after the chunk buffer."""
    total = 2 * int(num_rows) * int(vocab)
    accumulators = dict(memory["accumulators"])
    accumulators["saved_logits"] = total - int(memory["temporary"]["logits"])
    return dict(
        memory,
        accumulators=accumulators,
        accumulator_bytes=sum(accumulators.values()),
        saved_logits_bytes=total,
    )


def lm_head_loss_workspace_size(
    num_rows: int,
    vocab: int,
    chunk: int = DEFAULT_CHUNK_SIZE,
    device: Optional[torch.device] = None,
    *,
    entry: str = "loss",
    backend: str = "cake",
    hidden: Optional[int] = None,
    need_dx: bool = True,
    compact_rows: bool = False,
) -> int:
    """Workspace bytes :func:`prepare_lm_head_loss` needs for ``(T, V, C)`` on ``device`` (``hidden`` = ``H``, required
    for the compacted path).  ``compact_rows`` adds the gather buffer of the compacted path; sized for ``T`` rows it
    bounds every valid-row count.  The chunk buffers hold :func:`fold_rows_max` rows (a short-tail fold's chunk).  ``need_dx`` does not change the size: the K-sliced dX reduction stores each
    slice into the call's ordered FP32 slab (:attr:`Plan.dx_slab_bytes`), whose slab sum writes the ``dX_acc``
    accumulator; the slab and the accumulator are allocated outside the workspace (:func:`workspace_layout`;
    :func:`memory_report` lists ``dx_slab``)."""
    if compact_rows and not hidden and backend == "reference":
        raise ValueError(
            "lm_head_loss_workspace_size needs hidden= for the compacted path"
        )
    # the chunk buffers of every plan of the call, the short-tail fold's included
    extent = fold_rows_max(num_rows, chunk)
    if backend == "reference":
        return int(
            workspace_layout(
                num_rows,
                vocab,
                extent,
                entry=entry,
                hidden=hidden or 0,
                compact=compact_rows,
            )["total"]
        )
    name, record = record_for(device)
    stages = registered_stages(name)
    geometry = Geometry.from_record(record)
    return int(
        workspace_layout(
            num_rows,
            vocab,
            extent,
            stats_tile=geometry.stats_tile,
            entry=entry,
            tma_workspace_bytes=_record_tma_bytes(record, stages),
            scratch_bytes=_record_scratch_bytes(record, stages),
            hidden=int(hidden or 0),
            compact=bool(compact_rows),
        )["total"]
    )


def _carve(flat: torch.Tensor, layout: dict, name: str, dtype, shape) -> torch.Tensor:
    offset, nbytes = layout[name]
    needed = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if needed > nbytes:
        raise ValueError(
            f"workspace region {name!r} holds {nbytes} bytes, {needed} needed"
        )
    return flat[offset : offset + needed].view(dtype).view(shape)


# ---------------------------------------------------------------------------
# Launch binding
# ---------------------------------------------------------------------------

_GRID_FUNCTIONS = {"min": min, "max": max}


def grid_dims(rule, scalars: dict[str, Any], num_sms: int) -> tuple[int, int, int]:
    """Evaluate a registry grid rule.

    Each of the three entries is an integer or an integer expression over the
    scalar names of the stage's host values (``rows_c``, ``m_tiles``, ``V``,
    ``num_vecs``, ...) and ``sms`` (the SM count) with ``+``, ``-``, ``*``,
    ``/`` (rounds up), ``//`` (rounds down), parentheses and ``min`` / ``max``:
    ``"rows_c/8"`` (one CTA per eight rows), ``"max(1, m_tiles//2*605)*2"`` (the
    whole work-item domain of a dynamically scheduled kernel).
    """
    if len(rule) != 3:
        raise ValueError("grid rule must have three entries")
    # Only the integer host values take part; tensors and None never enter the evaluation environment, and the
    # evaluator is a plain module-level function -- a nested recursive closure would form a reference cycle that
    # keeps every captured host value (the accumulators / outputs of a scale_cast call) alive until the cyclic GC.
    names = {
        key: int(value)
        for key, value in scalars.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    names["sms"] = int(num_sms)
    x, y, z = (_grid_term(value, names) for value in rule)
    return max(1, x), max(1, y), max(1, z)


def _grid_term(value, names: dict[str, int]) -> int:
    if isinstance(value, bool):
        raise ValueError("grid rule entries must be integers or expressions")
    if isinstance(value, int):
        return int(value)
    try:
        tree = _grid_tree(str(value).strip())
    except SyntaxError as exc:
        raise ValueError(f"grid rule entry {value!r} is not an expression") from exc
    return _grid_eval(tree, names)


@functools.lru_cache(maxsize=512)
def _grid_tree(text: str) -> ast.Expression:
    """The parsed grid rule entry (the records carry a fixed set of rule texts; bounded, never mutated:
    :func:`_grid_eval` only reads it)."""
    return ast.parse(text, mode="eval")


def _grid_eval(node, names: dict[str, int]) -> int:
    if isinstance(node, ast.Expression):
        return _grid_eval(node.body, names)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError(
                f"grid rule constants must be integers, got {node.value!r}"
            )
        return int(node.value)
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise KeyError(f"grid rule names the unknown scalar {node.id!r}")
        return names[node.id]
    if isinstance(node, ast.BinOp):
        left, right = _grid_eval(node.left, names), _grid_eval(node.right, names)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, (ast.Div, ast.FloorDiv)):
            if right == 0:
                raise ValueError("grid rule divides by zero")
            return -(-left // right) if isinstance(node.op, ast.Div) else left // right
        raise ValueError(f"grid rule operator {type(node.op).__name__} is not allowed")
    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in _GRID_FUNCTIONS
            or node.keywords
        ):
            raise ValueError("grid rule calls must be min(...) or max(...)")
        if len(node.args) < 2:
            raise ValueError("grid rule min()/max() take at least two terms")
        return _GRID_FUNCTIONS[node.func.id](
            _grid_eval(arg, names) for arg in node.args
        )
    raise ValueError(f"grid rule syntax {type(node).__name__} is not allowed")


@dataclass(frozen=True)
class _Launch:
    stage: str
    module: str
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    grid: tuple[int, int, int]
    # Descriptor-preparation entry of a pointer-ABI module (same arguments).
    prepare: Optional[Callable[..., Any]] = field(default=None, repr=False)
    # ``(argument index, host value name)`` of every tensor argument (re-bound per call by a remembered binding).
    slots: tuple[tuple[int, str], ...] = ()

    def __call__(self) -> None:
        self.entry(*self.arguments)

    def templated(self) -> "_Launch":
        """Copy whose tensor arguments are ``None`` placeholders (holds no tensor)."""
        arguments = list(self.arguments)
        for index, _ in self.slots:
            arguments[index] = None
        return replace(self, arguments=tuple(arguments))

    def arguments_for(self, values: dict[str, Any]) -> list:
        """The argument list with every slot filled from ``values`` (fails closed on a missing one)."""
        arguments = list(self.arguments)
        for index, key in self.slots:
            value = values.get(key)
            if value is None:
                raise RuntimeError(
                    f"stage {self.stage!r} of {self.module!r}: the remembered argument plan needs "
                    f"{key!r} (argument {index}) and the call provides no value for it"
                )
            arguments[index] = value
        return arguments


def bind_stage(
    module_name: str, stage: str, values: dict[str, Any], grid: tuple[int, int, int]
) -> _Launch:
    """Order ``values`` by the generated argument plan of ``stage`` and load its entry.

    Fails closed: a keyword the kernel expects that the host does not provide
    raises ``KeyError`` naming both sides.  The returned launch records which
    argument positions hold tensors (re-bound per call by a remembered binding).
    """
    physical = MODULES[module_name][stage]
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    slots = []
    for kind, name in physical["arg_plan"]:
        key = name if name in values else CONTRACT_ALIASES.get(name, name)
        if kind == "grid":
            arguments.append(grid_values[name])
        elif (
            key in values and values[key] is not None
        ):  # buffer / tma_buffer / workspace / parameter
            value = values[key]
            if isinstance(value, torch.Tensor):
                slots.append((len(arguments), key))
            arguments.append(value)
        else:
            raise KeyError(
                f"generated module {module_name!r} stage {stage!r} expects argument {name!r} ({kind}); "
                f"the host binding provides {sorted(k for k, v in values.items() if v is not None)}"
            )
    module = load_cake_lm_head_loss_module(module_name, stage)
    prepare_entry = physical.get("tma_prepare_entry")
    prepare = getattr(module, prepare_entry) if prepare_entry else None
    return _Launch(
        stage,
        module_name,
        getattr(module, physical["ffi_entry"]),
        tuple(arguments),
        grid,
        prepare,
        tuple(slots),
    )


def _check_temperature_operands(
    module_name: str, keys, values: dict[Any, dict[str, Any]]
) -> None:
    """Fail closed before the launches of a call with a temperature are bound (:func:`_bind_all`): a ``temperature``
    host value for a launch whose stage's argument plan has no temperature operand raises ``ValueError`` (a kernel
    without the temperature must never serve a call with one).  Host steps (:data:`HOST_STAGES`) bind no kernel."""
    record = MODULES[module_name]
    for key in keys:
        stage = key[0]
        if stage in HOST_STAGES or values[key].get("temperature") is None:
            continue
        if not any(
            kind != "grid" and CONTRACT_ALIASES.get(name, name) == "temperature"
            for kind, name in record[stage]["arg_plan"]
        ):
            raise ValueError(
                f"generated module {module_name!r} stage {stage!r} has no temperature operand, but the call provides a "
                "temperature: a stage without the temperature cannot serve it"
            )


_FFI_DEVICES: dict[int, Any] = {}


def _constant_cells(device: torch.device) -> dict[str, torch.Tensor]:
    """The cells of :func:`_device_constants`, filled on the current stream."""
    return {
        "grad_scale": torch.ones((1,), dtype=torch.float32, device=device),
        "unit_scale": torch.ones((1,), dtype=torch.float32, device=device),
        "f32_dummy": torch.zeros((16,), dtype=torch.float32, device=device),
    }


# The constant cells of each device index (:func:`_device_constants`): the first cells stored for a device are the
# ones every thread uses.
_DEVICE_CELLS: dict[int, dict[str, torch.Tensor]] = {}


@functools.lru_cache(maxsize=None)
def _device_constants(index: int) -> dict[str, torch.Tensor]:
    """The constant cells a remembered binding binds on every call, allocated once per device:
    ``grad_scale`` / ``unit_scale`` (``[1] = 1.0``; read-only here -- the loss entry's backward
    casts through :func:`backward_loss`, the log-probability entry's cast scale is 1) and the
    unused fp32 ``WS`` of the unsliced GEMMs.  Per-call ``torch.ones`` / ``torch.zeros`` would add
    three fill launches to every remembered forward and backward.

    Every stream and thread shares the cells, so they are filled on a stream from torch's stream
    pool, and an event recorded behind the fills is waited for before they are cached: a call on
    another stream never reads a cell before its fill.  The wait is not a stream or device
    synchronization (PyTorch's synchronization debug mode does not flag it, and the mode is not
    changed), but it can also wait for other work: torch hands out its pool streams round-robin, so
    the caller's current stream can be that same stream (the fills then queue behind the caller's
    work), and with CUDA lazy loading a kernel's first launch can wait for the device.  The first
    calls of threads that miss the cache together each form cells, and all return the first cells
    stored in :data:`_DEVICE_CELLS`; the other cells were waited for and are never read, so freeing
    them is safe.  Under a CUDA graph capture, which cannot wait, the cells are made on the
    capturing stream (:func:`.cake_backend_opt_in.cake_lm_head_nondet_train_preload` creates them ahead of one).  The cells belong to no call, so
    they are made outside any active torch dispatch mode: selective activation checkpointing replays
    a recomputation's operators against its checkpointed forward's, and a first use inside a
    recomputation would add these."""
    device = torch.device("cuda", int(index))
    with _modes_off():
        if torch.cuda.is_current_stream_capturing():
            cells = _constant_cells(device)
        else:
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                cells = _constant_cells(device)
                filled = torch.cuda.Event()
                filled.record(stream)
            filled.synchronize()
    return _DEVICE_CELLS.setdefault(int(index), cells)


def _unit_scale(device: torch.device) -> torch.Tensor:
    """The ``[1] = 1.0`` fp32 scale of a cast whose incoming gradient is ``None`` (the log-probability
    entry's dX finalize): the per-device cached cell on CUDA -- a per-call ``torch.ones`` would be one
    fill launch inside the backward -- and a fresh cell elsewhere (the reference backend on the host)."""
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        return _device_constants(int(index))["unit_scale"]
    return torch.ones((1,), dtype=torch.float32, device=device)


def _current_raw_stream(index: int) -> int:
    """The raw handle of torch's current stream on device ``index``."""
    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    return (
        getter(index)
        if getter is not None
        else torch.cuda.current_stream(index).cuda_stream
    )


def _ffi_stream_context(index: int):
    """tvm-ffi environment-stream context for torch's current stream on device ``index``."""
    import tvm_ffi

    device = _FFI_DEVICES.get(index)
    if device is None:
        device = _FFI_DEVICES[index] = tvm_ffi.device(f"cuda:{index}")
    return tvm_ffi.use_raw_stream(device, _current_raw_stream(index))


# The launch keys after which a chunk's ``dz`` is complete, by role (every form of ``row_grad`` and of the ``dz``
# recompute, the temperature form included): the dW side stream's fork event is recorded behind them.
DZ_PRODUCER_STAGES = frozenset(
    stage
    for stage in STAGES_WITH_TEMPERATURE
    if base_stage(stage) in ("row_grad", LOGPROB_DZ_STAGE)
)


def side_stream_schedule(
    plan: Plan, keys: tuple, dw_stream: Optional[Union[bool, str]] = None
) -> Optional[tuple]:
    """Side-stream placement of :func:`dw_side_stream` over a launch-key sequence under the ``dw_stream`` keyword (or
    its mode; ``None`` = the ``auto`` rule): ``None`` when the plan has no ``dW``, the mode or the rule says no, or the
    sequence has no per-chunk accumulate (the loss entry's cast keys); else the
    positions ``(joins, forks, sides)`` -- the first key of every chunk after the first and the first key after the last
    chunk's keys (the caller's stream waits for the side stream: the chunk buffer is about to be reused / read), each
    chunk's ``row_grad`` key or ``dz`` recompute (:data:`LOGPROB_DZ_STAGE`; the fork event is recorded after it) and the per-chunk ``dW`` accumulate keys (launched
    on the side stream after it waited for the fork).  A deferred last chunk's cast GEMM and the flat casts stay on
    the caller's stream; the sequence always ends with a join."""
    if not (plan.need_dw and dw_side_stream(plan.num_chunks, dw_stream)):
        return None
    chunk_keys = [
        (pos, stage, index)
        for pos, (stage, index) in enumerate(keys)
        if isinstance(index, int) and not isinstance(index, bool)
    ]
    sides = frozenset(
        pos for pos, stage, index in chunk_keys if stage == plan.dw_acc_stage(index)
    )
    if not sides:
        return None
    joins, forks, seen = set(), set(), set()
    for pos, stage, index in chunk_keys:
        if index not in seen:
            seen.add(index)
            if index > 0:
                joins.add(pos)
        if stage in ("row_grad", LOGPROB_DZ_STAGE, LOGPROB_DZ_DCNT_STAGE):
            forks.add(pos)
    after = chunk_keys[-1][0] + 1
    if after < len(keys):
        joins.add(after)
    return frozenset(joins), frozenset(forks), sides


def _side_stream_schedule_temperature(
    plan: Plan, keys: tuple, dw_stream: Optional[Union[bool, str]] = None
) -> Optional[tuple]:
    """:func:`side_stream_schedule` of a plan with the temperature forms.  Side-stream placement of :func:`dw_side_stream` over a launch-key sequence under the ``dw_stream`` keyword (or
    its mode; ``None`` = the ``auto`` rule): ``None`` when the plan has no ``dW``, the mode or the rule says no, or the
    sequence has no per-chunk accumulate (the loss entry's cast keys); else the
    positions ``(joins, forks, sides)`` -- the first key of every chunk after the first and the first key after the last
    chunk's keys (the caller's stream waits for the side stream: the chunk buffer is about to be reused / read), each
    chunk's ``row_grad`` key or ``dz`` recompute (:data:`DZ_PRODUCER_STAGES`, by role: the temperature form too; the
    fork event is recorded after it) and the per-chunk ``dW`` accumulate keys (launched
    on the side stream after it waited for the fork).  A deferred last chunk's cast GEMM and the flat casts stay on
    the caller's stream; the sequence always ends with a join.  Fails closed (``RuntimeError``) when a chunk's
    accumulate has no ``dz`` producer before it in the same chunk: its side launch would wait on a stale event."""
    if not (plan.need_dw and dw_side_stream(plan.num_chunks, dw_stream)):
        return None
    chunk_keys = [
        (pos, stage, index)
        for pos, (stage, index) in enumerate(keys)
        if isinstance(index, int) and not isinstance(index, bool)
    ]
    sides = frozenset(
        pos for pos, stage, index in chunk_keys if stage == plan.dw_acc_stage(index)
    )
    if not sides:
        return None
    joins, forks, seen = set(), set(), set()
    forked: set = set()  # chunks whose fork key came before their side keys
    for pos, stage, index in chunk_keys:
        if index not in seen:
            seen.add(index)
            if index > 0:
                joins.add(pos)
        if stage in DZ_PRODUCER_STAGES:
            forks.add(pos)
            forked.add(index)
        elif pos in sides and index not in forked:
            # fail closed: the side launch would wait on an event recorded before this chunk's ``dz`` exists
            raise RuntimeError(
                f"side-stream schedule: chunk {index}'s dW accumulate {stage!r} has no dz producer "
                "(row_grad or a dz recompute) before it in the same chunk"
            )
    after = chunk_keys[-1][0] + 1
    if after < len(keys):
        joins.add(after)
    return frozenset(joins), frozenset(forks), sides


def _run_keys(
    plan: Plan,
    device_index: int,
    keys: tuple,
    run_key: Callable[[Any], None],
    dw_stream: Optional[Union[bool, str]] = None,
) -> None:
    """Run the launch keys in order on the caller's stream; with the placement of :func:`side_stream_schedule` under
    ``dw_stream`` the per-chunk ``dW`` accumulates go to the side stream.  The generated launches take tvm-ffi's
    environment stream, so a side launch enters its own stream context and the caller's context is re-entered afterwards
    (no nesting)."""
    schedule = side_stream_schedule(plan, keys, dw_stream)
    if schedule is None:
        with _ffi_stream_context(device_index):
            for key in keys:
                run_key(key)
        return
    joins, forks, sides = schedule
    side, fork = _dw_stream(device_index)
    main = torch.cuda.current_stream(device_index)
    with contextlib.ExitStack() as stack:
        stack.enter_context(_ffi_stream_context(device_index))
        for pos, key in enumerate(keys):
            if pos in joins:
                main.wait_stream(side)
            if pos in sides:
                stack.close()
                side.wait_event(fork)
                with torch.cuda.stream(side), _ffi_stream_context(device_index):
                    run_key(key)
                stack.enter_context(_ffi_stream_context(device_index))
                continue
            run_key(key)
            if pos in forks:
                fork.record(main)
    main.wait_stream(side)


def _run_keys_temperature(
    plan: Plan,
    device_index: int,
    keys: tuple,
    run_key: Callable[[Any], None],
    dw_stream: Optional[Union[bool, str]] = None,
) -> None:
    """:func:`_run_keys` of a plan with the temperature forms (:attr:`Plan.temperature`): the side-stream placement
    of :func:`_side_stream_schedule_temperature` (its forks follow the temperature form of the ``dz`` recompute)."""
    schedule = _side_stream_schedule_temperature(plan, keys, dw_stream)
    if schedule is None:
        with _ffi_stream_context(device_index):
            for key in keys:
                run_key(key)
        return
    joins, forks, sides = schedule
    side, fork = _dw_stream(device_index)
    main = torch.cuda.current_stream(device_index)
    with contextlib.ExitStack() as stack:
        stack.enter_context(_ffi_stream_context(device_index))
        for pos, key in enumerate(keys):
            if pos in joins:
                main.wait_stream(side)
            if pos in sides:
                stack.close()
                side.wait_event(fork)
                with torch.cuda.stream(side), _ffi_stream_context(device_index):
                    run_key(key)
                stack.enter_context(_ffi_stream_context(device_index))
                continue
            run_key(key)
            if pos in forks:
                fork.record(main)
    main.wait_stream(side)


# ---------------------------------------------------------------------------
# Host values of the stages
# ---------------------------------------------------------------------------


def stage_values(
    stage: str, t: dict[str, Any], plan: Plan, index: int, *, order_key: Any = None
) -> dict[str, Any]:
    """Host values of ``stage`` for chunk ``index`` (the kernel's own argument names).

    ``t`` holds the bound tensors: ``X`` (as launched), ``W``, ``labels``,
    ``lse``, ``logp``, ``infer_logp`` / ``loss_weights`` (policy), ``d_in``
    (the ``[T]`` incoming ``dlogp`` of the log-probability backward), the
    workspace regions ``logits`` / ``stats`` / ``d`` / ``term`` / ``loss_acc`` /
    ``grad_scale``, ``loss`` (the finished-loss cell), ``dx_acc`` / ``dw_acc``
    and the cast outputs ``dx_out`` / ``dw_out``.  For the casts ``index`` is
    ``"dx"`` or ``"dw"``.

    Compacted plan: the row-indexed tensors (``labels``, ``lse``, ``logp``,
    ``infer_logp``, ``loss_weights``, ``d_in``, ``dx_acc``, ``dx_out``) hold the
    valid rows only (``T`` is the valid-row count), ``t["row_index"]`` is the
    int64 valid-row index, ``t["x_c"]`` the BF16 ``[C, H]`` gather buffer the
    GEMMs read the chunk's ``X`` rows from, and the host stage ``gather_rows``
    fills them: chunk ``index`` gathers the chunk's rows of ``X``; a string
    ``index`` (``"infer_logp"``, ``"loss_weights"``, ``"d_in"``) gathers the
    caller's ``[T]`` operand ``t[index + "_full"]`` into ``t[index]``.
    """
    p, g = plan.problem, plan.geometry
    values: dict[str, Any] = {name: t.get(name) for name in COMMON_TENSORS}
    values.update(
        T=int(plan.rows),
        H=int(p.hidden),
        V=int(p.vocab),
        chunk=int(p.chunk),
        num_tiles=int(plan.num_tiles),
        mode=int(p.mode),
        loss_div=float(p.loss_div) if p.loss_div is not None else 1.0,
        ws_slab=0,
    )
    if (
        stage == "gather_rows"
    ):  # host-side row gather of the compacted path (no kernel of the program)
        if not plan.compact:
            raise ValueError("gather_rows belongs to a compacted plan")
        if isinstance(index, str):
            values.update(
                src=t[index + "_full"],
                idx=t["row_index"],
                out=t[index],
                rows_c=int(plan.rows),
                row0=0,
                first_chunk=0,
                last_chunk=0,
                d_off=0,
            )
            return values
        row0, rows_c = plan.chunks[index]
        values.update(
            src=t["X"],
            idx=t["row_index"][row0 : row0 + rows_c],
            out=t["x_c"][:rows_c],
            rows_c=int(rows_c),
            row0=int(row0),
            first_chunk=int(index == 0),
            last_chunk=int(index == plan.num_chunks - 1),
            d_off=0,
        )
        return values
    if stage == "scale_cast_scatter_bf16":
        raise ValueError(
            "scale_cast_scatter_bf16 is bound by finalize_dx at the dX output boundary, not by the chunk loop"
        )
    if stage in ("scale_cast_bf16", "scale_cast_f32"):
        acc = t["dx_acc"] if index == "dx" else t["dw_acc"]
        out = t["dx_out"] if index == "dx" else t["dw_out"]
        if acc.numel() != out.numel() or acc.numel() % g.cast_vec:
            raise ValueError(
                "scale-cast operands must match in size and hold whole vectors"
            )
        # loss entry: the backward writes the incoming scalar gradient into the ``grad_scale`` workspace cell;
        # log-probability entry: the scale is the constant 1, held outside the workspace (a caller may poison
        # the workspace between steps; ``dlogp`` carries the gradient)
        scale = t["grad_scale"] if p.entry == "loss" else t["unit_scale"]
        values.update(
            acc=acc.reshape(-1),
            g=scale,
            out=out.reshape(-1),
            num_vecs=int(acc.numel() // g.cast_vec),
            rows_c=0,
            row0=0,
            first_chunk=0,
            last_chunk=0,
            d_off=0,
        )
        return values
    row0, rows_c = plan.chunks[index]
    stop = row0 + rows_c
    first, last = int(index == 0), int(index == plan.num_chunks - 1)
    values.update(
        rows_c=int(rows_c), row0=int(row0), first_chunk=first, last_chunk=last, d_off=0
    )
    x_chunk = (
        t["x_c"][:rows_c] if plan.compact else t["X"][row0:stop]
    )  # compacted: the chunk's gathered valid rows
    if t.get("x_last") is not None and index == plan.num_chunks - 1:
        x_chunk = t[
            "x_last"
        ]  # the eager backward of a deferred plan: the saved view of the last chunk's X rows
    logits = t["logits"]
    dz_chunk = logits[:rows_c]
    base = base_stage(stage) if stage in GEMM_STAGES else stage
    if base in LOGITS_BASES:
        if base == LOGPROB_DZ_STAGE:
            want = plan.recompute_stage(index)
        elif base == LOGPROB_LBL_STAGE:
            want = plan.forward_logits_stage(index)
        else:
            want = plan.logits_stage(index, stats=base == "gemm_logits")
        if stage != want:
            raise ValueError(
                f"chunk {index} plans the logits GEMM {want!r}; stage {stage!r} was requested"
            )
        # ``C`` is the chunk's [rows_c, V] rows of the logits workspace: the TMA-store epilogue's tensor map takes its
        # row extent from it (rows >= rows_c are clipped, never written); a pointer store sees the same base address
        values.update(
            A=x_chunk,
            B=t["W"],
            C=logits[:rows_c],
            STATS_OUT=t["stats"],
            WS=t["f32_dummy"],
            **gemm_scalars(
                stage,
                g,
                hidden=p.hidden,
                vocab=p.vocab,
                rows_c=rows_c,
                group_m=plan.variants_of(index).logits_group_m,
            ),
        )
        if base in (LOGPROB_DZ_STAGE, LOGPROB_LBL_STAGE):
            # the epilogue reads the chunk's labels at ``labels[row0 + r]``; the ``dz`` arm also ``lse[row0 + r]`` and
            # the caller's ``[T]`` ``dlogp`` at ``d_in[d_off + r]`` (the operands ``row_grad`` reads)
            values.update(labels=t["labels"])
        if base == LOGPROB_DZ_STAGE:
            values.update(lse=t["lse"], d_in=t["d_in"], d_off=int(row0))
    elif base in DX_BASES:
        _dx_values(stage, t, plan, index, values, dz_chunk)
    elif stage in DX_SLAB_SUM_STAGES:
        _dx_slab_sum_values(stage, t, plan, index, values)
    elif base == "gemm_dw_acc":
        want = plan.dw_acc_stage(index)
        if stage != want:
            raise ValueError(
                f"chunk {index} plans the weight-gradient accumulate {want!r}; stage {stage!r} was requested"
            )
        v = plan.variants_of(index)
        lc = parse_stage(stage)[1]["live_cast"]
        if lc and last:
            raise ValueError(
                f"stage {stage!r} reads the next chunk's count; chunk {index} is the last of {plan.num_chunks}"
            )
        values.update(
            A=dz_chunk,
            B=x_chunk,
            **_dw_acc_operands(t, p, lc),
            **gemm_scalars(
                stage,
                g,
                hidden=p.hidden,
                vocab=p.vocab,
                rows_c=rows_c,
                group_m=v.dw_group_m,
                group_n=v.dw_group_n,
                first_chunk=first,
            ),
        )
        if lc:
            values.update(CNT_NEXT=t["dcnt"][index + 1 : index + 2])
    elif base in DW_CAST_STAGES:
        # the deferred last chunk (backward): the same GEMM as ``gemm_dw_acc`` with the upstream scale and the output cast
        # fused into its epilogue -- ``C`` is the final ``dW``, ``WS`` the read-only FP32 sum of the chunks before it (a
        # dummy for a one-chunk plan, whose ``first_chunk`` form stores ``cast(g * tile)``), ``STATS_OUT`` the scale cell
        if not plan.dw_deferred or not last:
            raise ValueError(
                f"stage {stage!r} belongs to the last chunk of a plan with fuse_dw_cast (chunk {index} of {plan.num_chunks})"
            )
        if stage != plan.dw_cast_stage:
            raise ValueError(
                f"plan casts its weight gradient through {plan.dw_cast_stage!r}; stage {stage!r} was requested"
            )
        v = plan.variants_of(index)
        values.update(
            A=dz_chunk,
            B=x_chunk,
            C=t["dw_out"],
            STATS_OUT=t["grad_scale"] if p.entry == "loss" else t["unit_scale"],
            WS=t["dw_acc"] if plan.num_chunks > 1 else t["f32_dummy"],
            **gemm_scalars(
                stage,
                g,
                hidden=p.hidden,
                vocab=p.vocab,
                rows_c=rows_c,
                group_m=v.dw_group_m,
                group_n=v.dw_group_n if base == "gemm_dw_cast_f32" else None,
                first_chunk=int(plan.num_chunks == 1),
            ),
        )
    elif stage == "row_finalize":
        d = t["d"]

        def present(
            name,
        ):  # policy / external operands: the chunk-local ``d`` stands in when the mode never reads them
            return d if t.get(name) is None else t[name]

        values.update(
            stats=t["stats"],
            z=logits,
            labels=t["labels"],
            infer_logp=present("infer_logp"),
            loss_weights=present("loss_weights"),
            d_in=present("d_in"),
            lse=t["lse"],
            logp=t["logp"],
            d=d,
            term=t["term"],
        )
    elif stage == "loss_reduce":
        values.update(term=t["term"], loss_acc=t["loss_acc"], loss_out=t["loss"])
    elif stage == "row_grad":
        external = (
            p.entry == "logprob"
        )  # the recompute reads the caller's [T] dlogp at d[row0 + r]
        values.update(
            z=logits,
            labels=t["labels"],
            lse=t["lse"],
            d=t["d_in"] if external else t["d"],
            d_off=int(row0) if external else 0,
        )
    else:
        raise ValueError(f"unknown stage {stage!r}")
    _dcnt_value(stage, t, plan, index, values)
    return values


def _dw_acc_operands(t: dict[str, Any], p: Problem, live_cast: bool) -> dict[str, Any]:
    """The tensor operands of a weight-gradient accumulate: the FP32 sum ``dw_acc`` as ``C`` (the statistics and a
    dummy in the unused slots); its cast at the last live chunk (``_lc``) writes the bf16 ``dw_out`` (``C``), reads and
    adds into the sum ``dw_acc`` (``WS``) and scales by the upstream cell (``STATS_OUT``), the deferred cast's operands
    (the next chunk's count ``CNT_NEXT`` is bound beside them).  The gradient slots are allocated before the chunk
    loop's first dX launch, ahead of any accumulate on the side stream."""
    if live_cast:
        return dict(
            C=t["dw_out"],
            STATS_OUT=t["grad_scale"] if p.entry == "loss" else t["unit_scale"],
            WS=t["dw_acc"],
        )
    return dict(C=t["dw_acc"], STATS_OUT=t["stats"], WS=t["f32_dummy"])


def _dcnt_value(
    stage: str, t: dict[str, Any], plan: Plan, index, values: dict[str, Any]
) -> None:
    """The count cell ``CNT`` of a ``_dcnt`` GEMM form of chunk ``index`` (the chunk's element of the call's int32
    ``t["dcnt"]``, :func:`zero_dlogp_partition`), added to ``values`` in place; nothing for any other stage."""
    if stage in GEMM_STAGES and parse_stage(stage)[1]["dcnt"]:
        if not plan.dcnt:
            raise ValueError(
                f"stage {stage!r} belongs to a plan of the zero-gradient row skip"
            )
        values.update(CNT=t["dcnt"][index : index + 1])


def stage_values_temperature(
    stage: str, t: dict[str, Any], plan: Plan, index: int, *, order_key: Any = None
) -> dict[str, Any]:
    """:func:`stage_values` of a call with a temperature (:attr:`Problem.temperature`): a temperature form is checked
    and valued as its twin without the suffix (``row_finalize_tau`` as ``row_finalize``) on the plan without the
    temperature forms, a stage a temperature form replaces is refused on a plan that runs it, and every logits GEMM /
    row kernel role also reads the compact ``temperature[row0 + r]`` (the ``cake`` backend binds it to the
    temperature forms only, :func:`_check_temperature_operands`; the reference engine reads it in the stages without
    the suffix)."""
    if stage in TEMPERATURE_STAGES:
        if not plan.temperature:
            raise ValueError(
                f"stage {stage!r} is a temperature form; the plan runs the stages without the temperature"
            )
        role = base_stage(stage)
        values = stage_values(
            stage[: -len("_tau")],
            t,
            cast(_TemperaturePlan, plan).without_temperature(),
            index,
            order_key=order_key,
        )
    else:
        role = base_stage(stage) if stage in GEMM_STAGES else stage
        if role == "row_finalize" and plan.temperature:
            raise ValueError(
                f"the plan's row kernel is {plan.row_finalize_stage!r}; stage {stage!r} was requested"
            )
        values = stage_values(stage, t, plan, index, order_key=order_key)
    if role in LOGITS_BASES or role in ("row_finalize", "row_grad"):
        values.update(temperature=t["temperature"])
    return values


def _slot_values(stage: str, t: dict[str, Any], plan: Plan, index) -> dict[str, Any]:
    """The tensor host values of :func:`stage_values` for chunk ``index`` -- the views a remembered launch slots in per
    call (:meth:`_Launch.arguments_for`) and the operands of the ``gather_rows`` host step -- without the scalars and
    the plan checks, which the binding fixed when it was prepared.  Any other host stage gets :func:`stage_values`.
    :meth:`_Binding.from_runner` checks every remembered launch against :func:`stage_values` once."""
    p = plan.problem
    values: dict[str, Any] = {name: t.get(name) for name in COMMON_TENSORS}
    if stage == "gather_rows":
        if isinstance(index, str):
            values.update(src=t[index + "_full"], idx=t["row_index"], out=t[index])
            return values
        row0, rows_c = plan.chunks[index]
        values.update(
            src=t["X"], idx=t["row_index"][row0 : row0 + rows_c], out=t["x_c"][:rows_c]
        )
        return values
    if stage in ("scale_cast_bf16", "scale_cast_f32"):
        acc = t["dx_acc"] if index == "dx" else t["dw_acc"]
        out = t["dx_out"] if index == "dx" else t["dw_out"]
        values.update(
            acc=acc.reshape(-1),
            g=t["grad_scale"] if p.entry == "loss" else t["unit_scale"],
            out=out.reshape(-1),
        )
        return values
    row0, rows_c = plan.chunks[index]
    stop = row0 + rows_c
    base = base_stage(stage) if stage in GEMM_STAGES else stage
    logits = t["logits"]
    if base in LOGITS_BASES or base == "gemm_dw_acc" or base in DW_CAST_STAGES:
        x_chunk = t["x_c"][:rows_c] if plan.compact else t["X"][row0:stop]
        if t.get("x_last") is not None and index == plan.num_chunks - 1:
            x_chunk = t["x_last"]
    if base in LOGITS_BASES:
        values.update(
            A=x_chunk,
            B=t["W"],
            C=logits[:rows_c],
            STATS_OUT=t["stats"],
            WS=t["f32_dummy"],
        )
        if base in (LOGPROB_DZ_STAGE, LOGPROB_LBL_STAGE):
            values.update(labels=t["labels"])
        if base == LOGPROB_DZ_STAGE:
            values.update(lse=t["lse"], d_in=t["d_in"])
    elif base in DX_BASES:
        _dx_slots(stage, t, plan, index, values)
    elif stage in DX_SLAB_SUM_STAGES:
        _dx_slab_sum_slots(stage, t, plan, index, values)
    elif base == "gemm_dw_acc":
        lc = parse_stage(stage)[1]["live_cast"]
        values.update(A=logits[:rows_c], B=x_chunk, **_dw_acc_operands(t, p, lc))
        if lc:
            values.update(CNT_NEXT=t["dcnt"][index + 1 : index + 2])
    elif base in DW_CAST_STAGES:
        values.update(
            A=logits[:rows_c],
            B=x_chunk,
            C=t["dw_out"],
            STATS_OUT=t["grad_scale"] if p.entry == "loss" else t["unit_scale"],
            WS=t["dw_acc"] if plan.num_chunks > 1 else t["f32_dummy"],
        )
    elif stage == "row_finalize":
        d = t["d"]
        values.update(
            stats=t["stats"],
            z=logits,
            labels=t["labels"],
            infer_logp=d if t.get("infer_logp") is None else t["infer_logp"],
            loss_weights=d if t.get("loss_weights") is None else t["loss_weights"],
            d_in=d if t.get("d_in") is None else t["d_in"],
            lse=t["lse"],
            logp=t["logp"],
            d=d,
            term=t["term"],
        )
    elif stage == "loss_reduce":
        values.update(term=t["term"], loss_acc=t["loss_acc"], loss_out=t["loss"])
    elif stage == "row_grad":
        values.update(
            z=logits,
            labels=t["labels"],
            lse=t["lse"],
            d=t["d_in"] if p.entry == "logprob" else t["d"],
        )
    else:
        return stage_values(stage, t, plan, index)
    _dcnt_value(stage, t, plan, index, values)
    return values


def _slot_values_temperature(
    stage: str, t: dict[str, Any], plan: Plan, index
) -> dict[str, Any]:
    """:func:`_slot_values` of a binding with the temperature forms (:class:`_TemperatureBinding`): the role's views
    (a temperature form's role is its base stage: ``row_finalize_tau`` is ``row_finalize``'s temperature
    form, ``gemm_logits_dz_tau`` the ``dz`` arm's; the counted ``gemm_logits_dz_dcnt_tau`` takes the role of
    ``gemm_logits_dz_dcnt``, whose views include the count cell) plus the compact ``temperature`` rows, which
    every temperature form reads (:meth:`_Launch.arguments_for` slots the names of its own argument plan only)."""
    if stage == LOGPROB_DZ_DCNT_TAU_STAGE:
        role = LOGPROB_DZ_DCNT_STAGE
    elif stage in TEMPERATURE_STAGES:
        role = base_stage(stage)
    else:
        role = stage
    values = _slot_values(role, t, plan, index)
    values["temperature"] = t["temperature"]
    return values


def _same_view(a, b) -> bool:
    """Whether two host values bind the same argument: the same object, or tensors with the same address, shape,
    strides, dtype and device."""
    if a is b:
        return True
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return False
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
        and a.device == b.device
    )


def forward_keys(plan: Plan) -> tuple[tuple[str, Any], ...]:
    """Launch keys ``(stage, chunk index)`` of the forward chunk loop (compacted plan: the host gathers first)."""
    keys: list[tuple[str, Any]] = []
    if (
        plan.compact
        and plan.problem.entry == "loss"
        and plan.problem.objective == "policy"
    ):
        keys += [("gather_rows", "infer_logp"), ("gather_rows", "loss_weights")]
    for index in range(plan.num_chunks):
        # the hidden valid-row count: chunk 0's rows were gathered and its logits GEMM queued before the count was known
        queued = plan.hidden_count and index == 0
        if plan.compact and not queued:
            keys.append(("gather_rows", index))
        if not queued:
            keys.append((plan.forward_logits_stage(index), index))
        keys.append(("row_finalize", index))
        if plan.problem.entry == "loss":
            keys.append(("loss_reduce", index))
            if plan.need_dx or plan.need_dw:
                keys.append(("row_grad", index))
            if plan.need_dx:
                keys += _dx_keys(plan, index)
            if plan.need_dw and not (plan.dw_deferred and index == plan.num_chunks - 1):
                keys.append(
                    (plan.dw_acc_stage(index), index)
                )  # a deferred last chunk runs in the backward (cast_keys)
    return tuple(keys)


def _forward_keys_temperature(plan: Plan) -> tuple[tuple[str, Any], ...]:
    """Launch keys ``(stage, chunk index)`` of the forward chunk loop (compacted plan: the host gathers first)."""
    keys: list[tuple[str, Any]] = []
    if (
        plan.compact
        and plan.problem.entry == "loss"
        and plan.problem.objective == "policy"
    ):
        keys += [("gather_rows", "infer_logp"), ("gather_rows", "loss_weights")]
    for index in range(plan.num_chunks):
        # the hidden valid-row count: chunk 0's rows were gathered and its logits GEMM queued before the count was known
        queued = plan.hidden_count and index == 0
        if plan.compact and not queued:
            keys.append(("gather_rows", index))
        if not queued:
            keys.append((plan.forward_logits_stage(index), index))
        keys.append((plan.row_finalize_stage, index))
        if plan.problem.entry == "loss":
            keys.append(("loss_reduce", index))
            if plan.need_dx or plan.need_dw:
                keys.append(("row_grad", index))
            if plan.need_dx:
                keys += _dx_keys(plan, index)
            if plan.need_dw and not (plan.dw_deferred and index == plan.num_chunks - 1):
                keys.append(
                    (plan.dw_acc_stage(index), index)
                )  # a deferred last chunk runs in the backward (cast_keys)
    return tuple(keys)


def forward_early_keys(plan: Plan) -> tuple[tuple[str, Any], ...]:
    """The keys of :func:`forward_keys` a remembered forward on the host-count path launches first, right behind the
    valid-row count (:meth:`_Binding.forward`): chunk 0's row gather (compacted plan) and its forward logits GEMM
    (:meth:`Plan.forward_logits_stage`).  They read ``X`` (the rows of the index), ``W`` and -- the log-probability
    entry's label-column arm only -- the call's labels, and write the chunk buffers only, so the launches ahead of
    them in :func:`forward_keys` (the policy operands' gathers) and the rest of the call's binding can follow them
    unchanged.  Empty on the hidden valid-row count path, whose chunk 0 is already queued."""
    if plan.hidden_count or not plan.chunks:
        return ()
    keys: list[tuple[str, Any]] = []
    if plan.compact:
        keys.append(("gather_rows", 0))
    keys.append((plan.forward_logits_stage(0), 0))
    return tuple(keys)


def _reads_labels(stage: str) -> bool:
    """Whether the host values of ``stage`` (:func:`stage_values`) include the call's labels."""
    base = base_stage(stage) if stage in GEMM_STAGES else stage
    return "labels" in STAGE_TENSORS.get(base, ())


# ---------------------------------------------------------------------------
# The dX GEMM ``dX_c = dz_c @ W`` of a chunk: every launch of it is planned by :func:`_dx_keys` and bound by
# :func:`_dx_values` / :func:`_dx_slab_sum_values` (first binding) and :func:`_dx_slots` / :func:`_dx_slab_sum_slots`
# (a remembered binding's per-call views); both backend modules (:mod:`.cake_backend`, :mod:`.cake_backend_opt_in`) go
# through these.  A chunk of ``S > 1`` K slices (:func:`dx_slices`) is one launch of the K-sliced reduction
# :data:`DX_RED_STAGE`, which stores the slices' FP32 tiles into the call's slab (:func:`_dx_ordered_values`), and
# right after it one launch of the slab sum (:meth:`Plan.dx_slab_sum_stage_of`), which stores the tiles, summed in
# slice order, into the chunk's rows of ``dX_acc`` (their only writer: nothing fills them first).  A chunk of one slice
# is one launch of the deterministic package's plain-store ``gemm_dx`` forms.
# ---------------------------------------------------------------------------


def _reads_labels_temperature(stage: str) -> bool:
    """Whether the host values of ``stage`` (:func:`stage_values`) include the call's labels."""
    base = (
        base_stage(stage)
        if stage in GEMM_STAGES or stage in TEMPERATURE_STAGES
        else stage
    )
    return "labels" in STAGE_TENSORS.get(base, ())


def _dx_keys(plan: Plan, index: int) -> list:
    """The dX GEMM of chunk ``index``: one launch, the K-sliced reduction or the plain store (:meth:`Plan.dx_stage_of`),
    and after the K-sliced reduction its slab sum (:meth:`Plan.dx_slab_sum_stage_of`)."""
    slab_sum = plan.dx_slab_sum_stage_of(index)
    keys = [(plan.dx_stage_of(index), index)]
    if slab_sum is not None:
        keys.append((slab_sum, index))
    return keys


def _dx_values(
    stage: str,
    t: dict[str, Any],
    plan: Plan,
    index: int,
    values: dict[str, Any],
    dz_chunk: torch.Tensor,
) -> None:
    """:func:`stage_values` of the dX GEMM ``stage`` of chunk ``index``, added to ``values`` in place; ``dz_chunk`` is the
    chunk's ``dz`` rows.  ``C`` is the chunk's rows of ``dX_acc``: the plain store (one slice; the deterministic
    package's argument plan, ``WS`` an unused dummy) writes them; the K-sliced reduction (``k_slices`` /
    ``k_slice_iters`` its launch scalars) stores its slices into the slab of :func:`_dx_ordered_values` and leaves them
    to its slab sum (:func:`_dx_slab_sum_values`)."""
    p, g = plan.problem, plan.geometry
    row0, rows_c = plan.chunks[index]
    stop = row0 + rows_c
    k = plan.dx_slices_of(index)
    want = plan.dx_stage_of(index)
    if stage != want:
        raise ValueError(
            f"chunk {index} plans {k} dX slice(s) as {want!r}; stage {stage!r} was requested"
        )
    values.update(
        A=dz_chunk,
        B=t["W"],
        C=t["dx_acc"][row0:stop],
        dx_split=int(
            plan.variants_of(index).dx_split or 0
        ),  # the split-tail form's grid term
        **gemm_scalars(
            stage,
            g,
            hidden=p.hidden,
            vocab=p.vocab,
            rows_c=rows_c,
            k_slices=k,
        ),
    )
    if base_stage(stage) == DX_RED_STAGE:
        values.update(
            _dx_ordered_values(
                t,
                plan,
                values["C"],
                k_slices=values["k_slices"],
                m_tiles=values["m_tiles"],
                ldc=values["ldc"],
            )
        )
    else:
        values.update(STATS_OUT=t["stats"], WS=t["f32_dummy"])


def _dx_slots(
    stage: str, t: dict[str, Any], plan: Plan, index: int, values: dict[str, Any]
) -> None:
    """:func:`_slot_values` of the dX GEMM ``stage`` of chunk ``index`` (the per-call tensor views of a remembered
    launch), added to ``values`` in place."""
    row0, rows_c = plan.chunks[index]
    values.update(
        A=t["logits"][:rows_c],
        B=t["W"],
        C=t["dx_acc"][row0 : row0 + rows_c],
    )
    if base_stage(stage) == DX_RED_STAGE:
        p, k = plan.problem, plan.dx_slices_of(index)
        scalars = _gemm_scalars(
            stage, plan.geometry, p.hidden, p.vocab, rows_c, k, None, None, 1, 0
        )
        values.update(
            _dx_ordered_values(
                t,
                plan,
                values["C"],
                k_slices=k,
                m_tiles=scalars["m_tiles"],
                ldc=scalars["ldc"],
            )
        )
    else:
        values.update(STATS_OUT=t["stats"], WS=t["f32_dummy"])


def _dx_slab_sum_operands(t: dict[str, Any], plan: Plan, index: int) -> dict[str, Any]:
    """:func:`dx_slab_sum_operands` of chunk ``index``: its rows of ``dX_acc`` and the call's slab (:func:`_dx_slab`) at
    the ``m_tiles`` / ``ldc`` of the chunk's dX launch, plus the chunk's count cell ``CNT`` after a counted reduction
    -- the cell that launch reads."""
    p, g = plan.problem, plan.geometry
    row0, rows_c = plan.chunks[index]
    k = plan.dx_slices_of(index)
    dx_stage = plan.dx_stage_of(index)
    scalars = _gemm_scalars(dx_stage, g, p.hidden, p.vocab, rows_c, k, None, None, 1, 0)
    m_tiles, ldc = scalars["m_tiles"], scalars["ldc"]
    dx = t["dx_acc"][row0 : row0 + rows_c]
    slab = _dx_slab(t, plan, dx_slab_bytes(k, m_tiles, ldc), dx.device)
    values = dx_slab_sum_operands(
        dx, slab, k_slices=k, m_tiles=m_tiles, ldc=ldc, cast_vec=g.cast_vec
    )
    if dx_slab_sum_stage(dx_stage) == DX_SLAB_SUM_DCNT_STAGE:
        values.update(CNT=t["dcnt"][index : index + 1])
    return values


def _dx_slab_sum_values(
    stage: str, t: dict[str, Any], plan: Plan, index: int, values: dict[str, Any]
) -> None:
    """:func:`stage_values` of the slab sum ``stage`` of chunk ``index`` (:meth:`Plan.dx_slab_sum_stage_of`), added to
    ``values`` in place: the chunk's rows of ``dX_acc`` (``dx``), the call's slab the chunk's K-sliced dX launch wrote
    (``ws``) and its scalars (:func:`_dx_slab_sum_operands`)."""
    want = plan.dx_slab_sum_stage_of(index)
    if stage != want:
        raise ValueError(
            f"chunk {index} plans {plan.dx_slices_of(index)} dX slice(s), whose slab sum is {want!r}; "
            f"stage {stage!r} was requested"
        )
    values.update(_dx_slab_sum_operands(t, plan, index))


def _dx_slab_sum_slots(
    stage: str, t: dict[str, Any], plan: Plan, index: int, values: dict[str, Any]
) -> None:
    """:func:`_slot_values` of the slab sum ``stage`` of chunk ``index`` (the per-call tensor views of a remembered
    launch: ``dx``, ``ws`` and, counted, ``CNT``), added to ``values`` in place."""
    operands = _dx_slab_sum_operands(t, plan, index)
    values.update({name: operands[name] for name in STAGE_TENSORS[stage]})


def recompute_keys(plan: Plan) -> tuple[tuple[str, Any], ...]:
    """Launch keys of the log-probability backward (recompute the logits and run the gradient GEMMs per chunk).  A
    kept-logits plan (:class:`_KeptPlan`) recomputes nothing: ``row_grad`` forms each chunk's ``dz`` over its kept
    logits (the chunk's ``X`` rows are gathered for the weight-gradient GEMM only)."""
    keys: list[tuple[str, Any]] = []
    if plan.compact:
        keys.append(("gather_rows", "d_in"))
    for index in range(plan.num_chunks):
        queued = (
            plan.hidden_count and index == 0
        )  # the hidden valid-row count: chunk 0's recompute is already queued
        if plan.logprob_keep:
            if plan.compact and plan.need_dw:
                keys.append(("gather_rows", index))
            keys.append(("row_grad", index))  # dz over the chunk's kept logits
        elif plan.compact and not queued:
            keys.append(("gather_rows", index))
        if plan.logprob_keep:
            pass  # nothing to recompute
        elif queued:
            # the queued device-count recompute stored ``z``
            keys.append(("row_grad", index))
        elif plan.logprob_dz:
            keys.append(
                (plan.recompute_stage(index), index)
            )  # the recompute stores ``dz`` itself (its ``_dcnt`` form under the zero-gradient row skip)
        else:
            keys.append((plan.logits_stage(index, stats=False), index))
            keys.append(("row_grad", index))
        if plan.need_dx:
            keys += _dx_keys(plan, index)
        if plan.need_dw:
            if plan.dw_deferred and index == plan.num_chunks - 1:
                keys.append(
                    (plan.dw_cast_stage, index)
                )  # the scale is 1 here: the GEMM writes the final BF16 dW
            else:
                keys.append((plan.dw_acc_stage(index), index))
    return tuple(keys)


def _recompute_keys_temperature(plan: Plan) -> tuple[tuple[str, Any], ...]:
    """Launch keys of the log-probability backward of a plan with the temperature forms (recompute the logits and run
    the gradient GEMMs per chunk).  No kept-logits plan.  Under the zero-gradient row skip (:attr:`Plan.dcnt`) the plan
    names the counted dX / dW forms and the counted ``dz`` temperature form (:class:`_CountedTemperaturePlan`) or the
    uncounted one (the mixed :class:`_TemperaturePlan`)."""
    keys: list[tuple[str, Any]] = []
    if plan.compact:
        keys.append(("gather_rows", "d_in"))
    for index in range(plan.num_chunks):
        queued = (
            plan.hidden_count and index == 0
        )  # the hidden valid-row count: chunk 0's recompute is already queued
        if plan.compact and not queued:
            keys.append(("gather_rows", index))
        if queued:
            # the queued device-count recompute stored ``z``
            keys.append(("row_grad", index))
        elif plan.logprob_dz:
            # the recompute stores ``dz`` itself
            keys.append((plan.recompute_stage(index), index))
        else:
            keys.append((plan.logits_stage(index, stats=False), index))
            keys.append(("row_grad", index))
        if plan.need_dx:
            keys += _dx_keys(plan, index)
        if plan.need_dw:
            if plan.dw_deferred and index == plan.num_chunks - 1:
                keys.append(
                    (plan.dw_cast_stage, index)
                )  # the scale is 1 here: the GEMM writes the final BF16 dW
            else:
                keys.append((plan.dw_acc_stage(index), index))
    return tuple(keys)


def cast_keys(plan: Plan) -> tuple[tuple[str, Any], ...]:
    """Launch keys of the output casts: the flat ``dX`` cast and, for ``dW``, the flat cast of the accumulator or -- when
    the plan defers its last chunk -- that chunk's fused GEMM (its ``X`` rows gathered again first when the rows are
    compacted; the log-probability backward runs it inside :func:`recompute_keys` instead)."""
    keys: list[tuple[str, Any]] = []
    if plan.need_dx and plan.dx_cast:
        keys.append(("scale_cast_bf16", "dx"))
    if plan.need_dw:
        if not plan.dw_deferred:
            keys.append((plan.dw_cast_stage, "dw"))
        elif plan.problem.entry == "loss":
            last = plan.num_chunks - 1
            if plan.compact:
                keys.append(("gather_rows", last))
            keys.append((plan.dw_cast_stage, last))
    return tuple(keys)


# ---------------------------------------------------------------------------
# Reference engine: the stages as PyTorch operators over the same host values
# ---------------------------------------------------------------------------


def _mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` of BF16 operands with FP32 accumulation and an FP32 result (no intermediate rounding)."""
    if a.is_cuda:
        try:
            return torch.mm(a, b, out_dtype=torch.float32)
        except (TypeError, RuntimeError):
            pass
    return torch.mm(a.float(), b.float())


class ReferenceEngine:
    """PyTorch implementation of every stage at the kernels' rounding boundaries.

    Each method consumes the host values a generated stage is bound to
    (:func:`stage_values`), so the chunk loop, the plans and the autograd
    Functions are exercised without a generated program.  Not
    allocation-free (a development aid).
    """

    def run(self, stage: str, values: dict[str, Any]) -> None:
        getattr(self, stage if hasattr(self, stage) else base_stage(stage))(values)

    @staticmethod
    def _rows(values: dict[str, Any]) -> tuple[int, int]:
        return int(values["row0"]), int(values["rows_c"])

    @staticmethod
    def _valid(values: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        row0, rows_c = ReferenceEngine._rows(values)
        labels = values["labels"][row0 : row0 + rows_c].long()
        return labels, labels >= 0

    @staticmethod
    def _temperature(values: dict[str, Any], rows: int) -> Optional[torch.Tensor]:
        """The chunk's ``[rows, 1]`` FP32 temperatures ``temperature[row0 + r]`` (``None`` without a temperature): every
        division below is tensor by tensor (one correctly rounded FP32 division per element, never a multiply by a
        reciprocal)."""
        tau = values.get("temperature")
        if tau is None:
            return None
        row0 = int(values["row0"])
        return tau[row0 : row0 + rows, None]

    @staticmethod
    def gemm_logits(values: dict[str, Any], stats: bool = True) -> None:
        rows = int(values["M"])
        z = values["C"][:rows]
        torch.mm(
            values["A"], values["B"].t(), out=z
        )  # BF16 result of an FP32 accumulation
        if stats:
            tiles = int(values["num_tiles"])
            zf = z.float()
            tau = ReferenceEngine._temperature(values, rows)
            if tau is not None:
                zf = zf / tau  # q = RN(f32(z) / tau_t)
            zf = zf.view(rows, tiles, -1)
            pmax = zf.amax(-1)
            psum = torch.exp(zf - pmax[..., None]).sum(-1)
            out = values["STATS_OUT"][:rows]
            out[..., 0] = pmax
            out[..., 1] = psum

    @staticmethod
    def gemm_logits_nostats(values: dict[str, Any]) -> None:
        ReferenceEngine.gemm_logits(values, stats=False)

    @staticmethod
    def row_finalize(values: dict[str, Any]) -> None:
        row0, rows_c = ReferenceEngine._rows(values)
        stop = row0 + rows_c
        stats = values["stats"][:rows_c]
        pmax, psum = stats[..., 0], stats[..., 1]
        m = pmax.amax(-1)
        lse = m + torch.log((psum * torch.exp(pmax - m[:, None])).sum(-1))
        labels, valid = ReferenceEngine._valid(values)
        index = torch.where(valid, labels, torch.zeros_like(labels)).clamp_max(
            int(values["V"]) - 1
        )
        zy = values["z"][:rows_c].gather(1, index[:, None]).squeeze(1).float()
        tau = ReferenceEngine._temperature(values, rows_c)
        if tau is not None:
            # zq = RN(z_y / tau_t), the statistics' q at the label column
            zy = zy / tau[:, 0]
        zero = torch.zeros_like(lse)
        logp = torch.where(valid, zy - lse, zero)
        values["lse"][row0:stop] = lse
        values["logp"][row0:stop] = logp
        mode = int(values["mode"])
        d, term = zero, zero
        if mode == MODE_CE:
            d = torch.where(
                valid, torch.full_like(lse, -1.0 / float(values["loss_div"])), zero
            )
            term = logp
        elif mode == MODE_POLICY:
            ratio = torch.exp(logp - values["infer_logp"][row0:stop])
            w = values["loss_weights"][row0:stop]
            d = torch.where(valid & (ratio <= RATIO_CLIP), -w * ratio, zero)
            term = torch.where(valid, w * torch.clamp_max(ratio, RATIO_CLIP), zero)
        elif mode == MODE_EXTERNAL:
            d = torch.where(valid, values["d_in"][row0:stop], zero)
        values["d"][:rows_c] = d
        values["term"][:rows_c] = term

    @staticmethod
    def loss_reduce(values: dict[str, Any]) -> None:
        rows_c = int(values["rows_c"])
        total = (
            values["term"][:rows_c].double().sum().reshape(1)
        )  # FP64 accumulation of the FP32 terms
        acc = values["loss_acc"]
        if int(values["first_chunk"]):
            acc.copy_(total)
        else:
            acc.add_(total)
        if int(values["last_chunk"]):
            neg = -acc
            values["loss_out"].copy_(
                neg / float(values["loss_div"])
                if int(values["mode"]) == MODE_CE
                else neg
            )  # one FP64 -> FP32 rounding

    @staticmethod
    def row_grad(values: dict[str, Any]) -> None:
        row0, rows_c = ReferenceEngine._rows(values)
        z = values["z"][:rows_c]
        labels, valid = ReferenceEngine._valid(values)
        lse = values["lse"][row0 : row0 + rows_c]
        d_off = int(values["d_off"])
        d = values["d"][d_off : d_off + rows_c]
        zf = z.float()
        tau = ReferenceEngine._temperature(values, rows_c)
        if tau is not None:
            d = d / tau[:, 0]  # dt = RN(d_t / tau_t)
            zf = zf / tau  # q = RN(f32(z) / tau_t)
        d = torch.where(valid, d, torch.zeros_like(lse))
        p = torch.exp(zf - lse[:, None])
        index = torch.where(valid, labels, torch.zeros_like(labels)).clamp_max(
            int(values["V"]) - 1
        )
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, index[:, None], 1.0)
        dz = d[:, None] * (onehot - p)
        dz[~valid] = 0.0
        z.copy_(dz)  # the BF16 dlogits boundary

    @staticmethod
    def gemm_dx(values: dict[str, Any]) -> None:
        values["C"].copy_(_mm_fp32(values["A"], values["B"]))

    @staticmethod
    def gemm_dx_red(values: dict[str, Any]) -> None:
        # the K-sliced reduction: each slice's FP32 product stored into its own slab of ``SLAB`` (slice ``s`` at rows
        # ``s * m_tiles * 128 + r``), stored into the chunk's rows by the slab sum that follows; one slice adds its
        # product into ``C`` directly
        A, B, C = values["A"], values["B"], values["C"]
        k_iters, slices = int(values["k_iters"]), int(values["k_slices"])
        if slices == 1:
            C.add_(_mm_fp32(A, B))
            return
        step = int(A.shape[1]) // k_iters
        span = int(values["k_slice_iters"]) * step
        rows, pitch = int(A.shape[0]), int(values["m_tiles"]) * 128
        slab = values["SLAB"]
        for s in range(slices):
            k0 = s * span
            k1 = int(A.shape[1]) if s == slices - 1 else k0 + span
            slab[s * pitch : s * pitch + rows].copy_(_mm_fp32(A[:, k0:k1], B[k0:k1]))

    @staticmethod
    def dx_slab_sum(values: dict[str, Any]) -> None:
        # the slab sum: the K-sliced dX GEMM's slabs (slab ``s`` at ``s * ws_slab`` of the flat ``ws``) summed in
        # slice order, the sum stored into the chunk's rows
        dx, ws = values["dx"], values["ws"]
        rows, cols = int(dx.shape[0]), int(dx.shape[1])
        pitch = int(values["ws_slab"])
        total = None
        for s in range(int(values["n_slabs"])):
            part = ws[s * pitch : s * pitch + rows * cols].view(rows, cols)
            total = part if total is None else total + part
        dx.copy_(total)

    # the counted form stores every row here: the reference reduction computes every row of the chunk (the rows past
    # the count, whose upstream gradient is zero, hold zeros where the kernel stores ``-0.0``)
    dx_slab_sum_dcnt = dx_slab_sum

    @staticmethod
    def scale_cast_scatter_bf16(values: dict[str, Any]) -> None:
        # ``out[idx] = bf16(g * acc)``, exact zeros elsewhere (``idx_lo`` is the int32 view of the int64 index)
        idx = values["idx_lo"].view(torch.int64)
        out = values["out"]
        out.zero_()
        if idx.numel():
            out.index_copy_(0, idx, (values["acc"] * values["g"]).to(out.dtype))

    @staticmethod
    def gemm_dw_acc(values: dict[str, Any]) -> None:
        product = _mm_fp32(values["A"].t(), values["B"])
        if int(values["first_chunk"]):
            values["C"].copy_(product)
        else:
            values["C"].add_(product)

    @staticmethod
    def gemm_dw_cast_bf16(values: dict[str, Any]) -> None:
        product = _mm_fp32(values["A"].t(), values["B"])
        acc = (
            product if int(values["first_chunk"]) else values["WS"] + product
        )  # the FP32 accumulate of gemm_dw_acc (one RN add per element), WS never written
        values["C"].copy_(
            acc * values["STATS_OUT"]
        )  # scale_cast's FP32 multiply, one rounding at the copy

    gemm_dw_cast_f32 = gemm_dw_cast_bf16

    @staticmethod
    def scale_cast_bf16(values: dict[str, Any]) -> None:
        values["out"].copy_(values["acc"] * values["g"])  # one rounding at the copy

    scale_cast_f32 = scale_cast_bf16


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class LmHeadLossRunner:
    """The prepared launches of one tensor binding.

    Loss entry: ``forward()`` runs the chunk loop (three GEMMs per chunk) and
    returns ``(loss [1], logp [T])`` while filling ``dx_acc`` (FP32 ``[T, H]``)
    and ``dw_acc`` (FP32 ``[V, H]``); ``backward(grad)`` casts ``dX = bf16(g *
    dX_acc)`` into ``dx_out`` and ``dW = cast(g * dW_acc)`` into ``dw_out``
    (``g`` from the ``[1]`` FP32 ``grad_scale`` cell; ``None`` = 1); with
    ``fuse_dw_cast`` the last chunk's weight-gradient GEMM runs there instead
    and writes ``dw_out = cast(g * (dw_acc + dz_c^T @ X_c))`` from its
    epilogue (that chunk's ``dz`` stays in the workspace between the calls).
    Log-probability entry: ``forward()`` returns ``(logp, lse)``;
    ``backward()`` reads ``dlogp`` (the bound FP32 ``[T]`` tensor
    ``tensors["d_in"]``, masked to valid rows by the row kernel), recomputes
    the logits per chunk and produces ``dx_out`` / ``dw_out`` (BF16) through
    the accumulators.  No launch of the ``cake`` backend allocates or
    synchronizes; capture into a CUDA graph belongs to the caller.  Prepare a
    new runner when a shape, dtype, stride, option or tensor binding changes;
    values may change freely.

    Compacted runner (``compact_rows``): the chunk loop covers the valid rows
    of the labels as of preparation (``row_index``, ``valid_rows``); ``logp``,
    ``lse``, ``dx_acc`` and ``dx_out`` are the compact ``[T_v]`` / ``[T_v, H]``
    forms (:meth:`scatter` restores ``[T]``), ``dlogp`` stays the caller's
    ``[T]`` tensor (gathered per step).  Label values decide the compaction,
    so a runner must be re-prepared when the set of ignored rows changes.

    ``dw_stream`` is the mode of the dW side stream the launches use (:func:`dw_stream_mode`: ``"auto"`` = the
    :data:`DW_STREAM_MIN_CHUNKS` rule, ``"1"`` every multi-chunk call, ``"0"`` never; the ``dw_stream`` keyword of
    :func:`prepare_lm_head_loss`); it may be reassigned between steps (a CUDA graph records the mode it was captured
    under).
    """

    backend: str
    module_name: Optional[str]
    plan: Plan
    tensors: dict[str, Any] = field(repr=False)
    values: dict[Any, dict[str, Any]] = field(repr=False)
    launches: dict[Any, _Launch] = field(repr=False)
    forward_order: tuple = ()
    backward_order: tuple = ()
    device_index: int = 0
    workspace: Optional[torch.Tensor] = field(default=None, repr=False)
    layout: dict = field(default_factory=dict, repr=False)
    memory: dict = field(default_factory=dict, repr=False)
    engine: Optional[ReferenceEngine] = field(default=None, repr=False)
    dw_stream: str = "auto"
    _tma_prepared: bool = False

    @property
    def problem(self) -> Problem:
        return self.plan.problem

    @property
    def loss(self) -> torch.Tensor:
        return self.tensors["loss"]

    @property
    def logp(self) -> torch.Tensor:
        return self.tensors["logp"]

    @property
    def lse(self) -> torch.Tensor:
        return self.tensors["lse"]

    @property
    def dx_acc(self) -> Optional[torch.Tensor]:
        return self.tensors.get("dx_acc")

    @property
    def dw_acc(self) -> Optional[torch.Tensor]:
        return self.tensors.get("dw_acc")

    @property
    def dx_out(self) -> Optional[torch.Tensor]:
        return self.tensors.get("dx_out")

    @property
    def dw_out(self) -> Optional[torch.Tensor]:
        return self.tensors.get("dw_out")

    @property
    def dlogp(self) -> Optional[torch.Tensor]:
        return self.tensors.get("d_in_full", self.tensors.get("d_in"))

    @property
    def row_index(self) -> Optional[torch.Tensor]:
        """Int64 ``[T_v]`` index of the valid rows the compacted chunk loop processes (``None`` when uncompacted)."""
        return self.tensors.get("row_index")

    @property
    def valid_rows(self) -> int:
        return self.plan.rows

    def scatter(self, rows: torch.Tensor) -> torch.Tensor:
        """A compact ``[T_v, ...]`` result (``logp``, ``lse``, ``dx_acc``, ``dx_out``) as the caller's ``[T, ...]`` tensor
        with exact zeros on the ignored rows (the tensor itself when the runner is not compacted)."""
        return scatter_rows(rows, self.row_index, self.problem.num_rows)

    @property
    def stages(self) -> tuple[str, ...]:
        return self.plan.stages

    def prepare_tma(self) -> None:
        """Encode the descriptors of pointer-ABI stages once (idempotent)."""
        if self._tma_prepared or self.backend != "cake":
            return
        with _ffi_stream_context(self.device_index):
            for launch in self.launches.values():
                if launch.prepare is not None:
                    launch.prepare(*launch.arguments)
        self._tma_prepared = True

    def _run(self, keys: tuple) -> None:
        if self.backend == "cake":
            self.prepare_tma()

            def run_key(key) -> None:
                host = HOST_STAGES.get(key[0])
                if host is not None:
                    host(self.values[key])
                else:
                    self.launches[key]()

            _run_keys(self.plan, self.device_index, keys, run_key, self.dw_stream)
        else:
            for key in keys:
                host = HOST_STAGES.get(key[0])
                if host is not None:
                    host(self.values[key])
                else:
                    self.engine.run(key[0], self.values[key])

    def forward(self):
        """Loss entry: ``(loss [1], logp [T])``; log-probability entry: ``(logp, lse)``.  The loss entry's forward
        runs the dX GEMMs: under torch's deterministic algorithms a plan with a K-sliced dX chunk raises PyTorch's
        alert first (:func:`_alert_unordered_dx`)."""
        if self.problem.entry == "loss":
            _alert_unordered_dx(self.plan, "LmHeadLossRunner.forward", "loss")
        return self._forward()

    def _forward(self):
        """:meth:`forward` without the alert (the eager entries raise their own first)."""
        t = self.tensors
        self._run(self.forward_order)
        return (
            (t["loss"], t["logp"])
            if self.problem.entry == "loss"
            else (t["logp"], t["lse"])
        )

    def backward(self, grad: Optional[torch.Tensor] = None):
        """``(dx_out, dw_out)`` (``None`` for a frozen input).

        Loss entry: ``grad`` is the incoming scalar gradient (``None`` = 1);
        the saved accumulators are read, never written.  Log-probability
        entry: the caller has written ``dlogp`` into the bound ``dlogp``
        tensor; ``grad`` must be ``None`` (the cast scale is 1).  The
        log-probability entry's backward runs the dX GEMMs: under torch's
        deterministic algorithms a plan with a K-sliced dX chunk raises
        PyTorch's alert first (:func:`_alert_unordered_dx`); the loss entry's
        backward (casts and the deferred weight-gradient GEMM) launches no dX
        GEMM and does not alert.
        """
        if self.problem.entry == "logprob":
            _alert_unordered_dx(self.plan, "LmHeadLossRunner.backward", "logprob")
        return self._backward(grad)

    def _backward(self, grad: Optional[torch.Tensor] = None):
        """:meth:`backward` without the alert (the eager entries raise their own first)."""
        t = self.tensors
        if self.problem.entry == "logprob":
            if grad is not None:
                raise ValueError(
                    "the log-probability runner takes its gradient from the bound dlogp tensor"
                )
            # the cast scale of this entry is the constant 1 (``unit_scale``, outside the workspace)
        elif grad is None:
            t["grad_scale"].fill_(1.0)
        else:
            t["grad_scale"].copy_(grad.reshape(1).to(torch.float32))
        self._run(self.backward_order)
        dx = (
            t.get("dx_out") if self.plan.dx_cast else t.get("dx_acc")
        )  # dx_cast=False: the caller finalizes the accumulator
        return dx, t.get("dw_out")

    def step(self, grad: Optional[torch.Tensor] = None):
        self.forward()
        return self.backward(grad)

    __call__ = step


class _TemperatureRunner(LmHeadLossRunner):
    """A prepared runner of a plan with the temperature forms (:attr:`Plan.temperature`): its chunk loop takes the
    side-stream placement of the temperature forms (:func:`_run_keys_temperature`)."""

    def _run(self, keys: tuple) -> None:
        if self.backend != "cake":
            super()._run(keys)
            return
        self.prepare_tma()

        def run_key(key) -> None:
            host = HOST_STAGES.get(key[0])
            if host is not None:
                host(self.values[key])
            else:
                self.launches[key]()

        _run_keys_temperature(
            self.plan, self.device_index, keys, run_key, self.dw_stream
        )


class _KeptLossRunner(LmHeadLossRunner):
    """The prepared runner of a kept-logits plan (:class:`_KeptPlan`): its kept views are bound per call, so only the
    kept-logits entry points launch it (through :class:`_KeptBinding`)."""

    def forward(self):
        raise NotImplementedError(_KEPT_RUNNER)

    def backward(self, grad: Optional[torch.Tensor] = None):
        raise NotImplementedError(_KEPT_RUNNER)

    _forward, _backward = forward, backward


_KEPT_RUNNER = (
    "a kept-logits plan (logprob_keep) binds its kept views per call: launch it through the eager entry points "
    "(forward_logprob_keep / backward_logprob_keep / cake_lm_head_nondet_train_logprob_keep)"
)


def _check_output(t: Optional[torch.Tensor], name: str, shape: tuple, dtype) -> None:
    if t is None:
        return
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous {dtype} tensor of shape {tuple(shape)}"
        )


def _prepare_x(
    X: torch.Tensor, problem: Problem, compact: bool = False
) -> torch.Tensor:
    return (
        X.contiguous() if (problem.x_copy and not compact) else X
    )  # the per-chunk gather yields contiguous rows


def _prepare_labels(
    labels: torch.Tensor, geometry: Geometry, row_index: Optional[torch.Tensor] = None
) -> torch.Tensor:
    labels = (
        labels.contiguous() if row_index is None else labels.index_select(0, row_index)
    )
    if geometry.labels_dtype != labels.dtype:
        labels = labels.to(
            geometry.labels_dtype
        )  # one O(T) cast per call when the kernels read int32
    return labels


def _bind_all(
    record, module_name, keys, values, device, geometry: Geometry
) -> dict[Any, _Launch]:
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    launches: dict[Any, _Launch] = {}
    for key in keys:
        stage = key[0]
        if stage in HOST_STAGES:  # host-side step (row gather), no kernel
            continue
        physical = record[stage]
        cluster = physical.get("launch", {}).get("cluster") or (1, 1, 1)
        cluster_ctas = int(math.prod(int(c) for c in cluster))
        declared = geometry.cluster_ctas_of(stage)
        if declared is not None and declared != cluster_ctas:
            raise ValueError(
                f"stage {stage!r}: the record's geometry declares {declared}-CTA clusters, its module launches {tuple(cluster)}"
            )
        grid = grid_dims(
            physical.get("grid", ["rows_c", 1, 1]),
            values[key],
            num_sms,
        )
        if cluster_ctas > 1 and any(g % c for g, c in zip(grid, cluster, strict=True)):
            raise ValueError(
                f"stage {stage!r}: grid {grid} is not a multiple of the cluster shape {tuple(cluster)} baked into the module"
            )
        launches[key] = bind_stage(module_name, stage, values[key], grid)
    return launches


def _eager_launch(
    record, module_name: str, stage: str, values: dict[str, Any], device, geometry
) -> tuple[_Launch, list]:
    """``(launch, arguments)`` of one eager stage of the output boundary (:func:`scale_cast`, :func:`scale_cast_scatter`,
    :func:`dw_cast`).  Bound by :func:`_bind_all`; while :data:`BINDING_CACHE` is enabled the templated launch is
    remembered there per form -- the record, the stage, the device and every non-tensor host value, which fix the grid
    and the scalar arguments -- and a later call of the same form slots its tensors into it: the same entry, grid and
    arguments as a fresh binding."""
    key = (stage, "eager")
    cache = BINDING_CACHE
    if not cache.enabled:
        launch = _bind_all(
            record, module_name, (key,), {key: values}, device, geometry
        )[key]
        return launch, list(launch.arguments)
    form = (
        module_name,
        stage,
        str(device),
        tuple(
            sorted(
                (name, isinstance(value, torch.Tensor))
                + ((None,) if isinstance(value, torch.Tensor) else (value,))
                for name, value in values.items()
            )
        ),
    )
    launch = cache.eager_lookup(form, record)
    if launch is None:
        launch = _bind_all(
            record, module_name, (key,), {key: values}, device, geometry
        )[key]
        cache.eager_remember(form, record, launch.templated())
        return launch, list(launch.arguments)
    return launch, launch.arguments_for(values)


def prepare_lm_head_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div=None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    need_dx: bool = True,
    need_dw: bool = True,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    entry: str = "loss",
    dlogp: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    dx_acc: Optional[torch.Tensor] = None,
    dw_acc: Optional[torch.Tensor] = None,
    dx_out: Optional[torch.Tensor] = None,
    dw_out: Optional[torch.Tensor] = None,
    logp: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    backend: str = "cake",
    compact_rows=False,
    fuse_dw_cast: Optional[bool] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    dx_finalize: Optional[bool] = None,
    dx_cast: bool = True,
    dw_stream: Optional[Union[bool, str]] = None,
    logprob_dz: Optional[bool] = None,
    logprob_lbl: Optional[bool] = None,
    dx_rule: Optional[tuple] = None,
    logprob_keep: bool = False,
    dcnt: Optional[torch.Tensor] = None,
    _skip_forms: bool = False,
    kept_fold: bool = False,
) -> LmHeadLossRunner:
    """Validate one binding and prepare its launches.

    Missing outputs, accumulators and the workspace are allocated here (the
    only allocations of the ``cake`` backend).  Pass ``workspace_buffer`` of
    :func:`lm_head_loss_workspace_size` bytes to reuse storage across steps.
    For the log-probability entry ``dlogp`` is the FP32 ``[T]`` tensor the
    backward reads (allocated here when absent) and ``lse`` the statistic the
    forward writes / the backward reads.  ``T == 0`` is rejected here (the
    eager entry points return zeros for it without a runner).
    ``backend="reference"`` builds the same runner over PyTorch operators.

    ``compact_rows=True`` compacts the chunk loop to the valid rows of
    ``labels`` (:func:`valid_row_index`; an int64 index tensor is accepted in
    place of ``True``): ``lse`` / ``logp`` / ``dx_acc`` / ``dx_out`` are then
    bound or allocated as compact ``[T_v]`` / ``[T_v, H]`` tensors, the row
    operands are gathered per step and every row ignored is rejected (the
    eager entry points return zeros for it).

    ``fuse_dw_cast`` (default :func:`fuse_dw_cast_default`): the last chunk's
    weight-gradient GEMM runs in ``backward()`` with the scale + cast fused
    (``dw_acc`` is then the FP32 sum of the chunks before it and absent for a
    one-chunk plan); the chunk buffer must not be overwritten between
    ``forward()`` and ``backward()``.

    The registered program of the ``(H, V)`` geometry is selected
    (:func:`record_for`) and each chunk's GEMM instance variants follow the
    per-chunk rules of its architecture (:func:`chunk_variants`);
    ``gemm_tuning`` (default :func:`gemm_tuning_default`) pins knobs
    explicitly, e.g. ``{"logits": {"group_m": 16}}``.

    ``dx_finalize`` (default :func:`dx_finalize_default`) is recorded in the
    plan (the eager entry points' compacted dX output boundary; the runner's
    own ``dX`` output is the flat cast).  A chunk of more than one dX slice
    stores the sum of its slices into ``dx_acc`` (:data:`DX_RED_STAGE`, then
    its slab sum :data:`DX_SLAB_SUM_STAGE`) without reading it, so a bound
    ``dx_acc`` may hold any values between steps;
    the slice rule's settings are ``dx_rule`` (``(policy, red_min_rows)``;
    ``None`` = :func:`dx_slice_rule` at preparation).  ``dx_cast=False``
    leaves the flat ``dX`` cast out of ``backward()``, which then returns
    ``dx_acc`` for the caller to finalize (:func:`finalize_dx`: the fused dX
    finalize of a compacted :func:`backward_logprob`); ``dx_out`` is then
    neither bound nor allocated.

    ``dw_stream`` (default ``None`` = ``"auto"``, :func:`dw_stream_mode`): the
    dW side stream of the runner's launches -- ``None`` / ``"auto"`` for plans
    of :data:`DW_STREAM_MIN_CHUNKS` chunks or more, ``True`` for every
    multi-chunk plan, ``False`` never (bitwise the same outputs; the runner's
    ``dw_stream`` attribute holds the mode).

    ``logprob_dz`` / ``logprob_lbl`` (log-probability entry only; defaults
    :func:`logprob_dz_default` / :func:`logprob_lbl_default`): the backward's
    recompute writes ``dz`` from the ``gemm_logits_dz`` epilogue and the
    forward's logits GEMM is ``gemm_logits_lbl``, where the record registers
    them (bitwise the same outputs); ``False`` keeps ``gemm_logits_nostats`` +
    ``row_grad`` / ``gemm_logits``.  A loss-entry runner rejects ``True``.

    ``logprob_keep`` (log-probability entry, ``cake`` backend; default
    ``False``): prepare the kept-logits plan (:class:`_KeptPlan`) for the
    kept-logits entry points, which bind the kept views per call
    (:func:`forward_logprob_keep`, :func:`backward_logprob_keep`); the runner
    itself does not launch it (its ``forward`` / ``backward`` raise
    ``NotImplementedError``).  ``kept_fold`` (a kept-logits plan only; the
    kept form's ``tail_fold``): the kept plan folds its short tail where
    :func:`loop_chunks` folds; ``False`` keeps the chunks of ``chunk_size``
    rows.  The other plans always take :func:`loop_chunks`.

    ``dcnt`` (log-probability entry, compacted rows ordered by
    :func:`zero_dlogp_partition`): the int32 per-chunk counts of live rows; the
    plan then runs the ``_dcnt`` forms (:attr:`Plan.dcnt`) where the record
    registers every one it needs, and the shipped forms otherwise (the same
    outputs: every row computed).
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if entry != "logprob" and (logprob_dz or logprob_lbl or logprob_keep):
        raise ValueError(
            "logprob_dz / logprob_lbl / logprob_keep select logits GEMM arms of the log-probability entry"
        )
    keep = bool(entry == "logprob" and logprob_keep)
    if keep and backend != "cake":
        raise ValueError("the kept logits (logprob_keep) run on the cake backend")
    device = X.device
    record = None
    module_name = None
    tuning = _resolve_gemm_tuning(gemm_tuning)
    stream_mode = dw_stream_mode(dw_stream)
    slice_policy, red_min_rows = _resolve_dx_rule(dx_rule)
    if backend == "cake":
        if not X.is_cuda:
            raise ValueError("the cake backend needs CUDA tensors")
        module_name, record = record_for(device)
        record_abi(record)
    geometry = Geometry.from_record(record)
    problem = validate_lm_head_inputs(
        X,
        W,
        labels,
        objective=objective,
        loss_div=loss_div,
        infer_logp=infer_logp,
        loss_weights=loss_weights,
        chunk_size=chunk_size,
        grad_weight_dtype=grad_weight_dtype,
        entry=entry,
        geometry=geometry,
    )
    if problem.num_rows == 0:
        raise ValueError(
            "a prepared runner needs at least one row (T > 0); the eager entry points return zeros for T == 0"
        )
    given = [
        t
        for t in (
            W,
            labels,
            infer_logp,
            loss_weights,
            dlogp,
            workspace_buffer,
            dx_acc,
            dw_acc,
            dx_out,
            dw_out,
            logp,
            lse,
        )
        if t is not None
    ]
    if not all(t.device == device for t in given):
        raise ValueError("Expected all tensors on one device")
    stages = registered_stages(module_name) if record is not None else tuple(STAGES)
    num_sms = (
        int(torch.cuda.get_device_properties(device).multi_processor_count)
        if device.type == "cuda"
        else 1
    )
    row_index = None
    if isinstance(compact_rows, torch.Tensor):
        row_index = compact_rows
        if (
            row_index.dtype != torch.int64
            or row_index.ndim != 1
            or row_index.device != device
        ):
            raise ValueError(
                "compact_rows given as a tensor must be the int64 [T_v] valid-row index on X's device"
            )
    elif compact_rows:
        row_index = valid_row_index(labels)
    if row_index is not None and row_index.numel() == 0:
        raise ValueError(
            "a prepared runner needs at least one valid row (every label is ignored); the eager entry points return zeros for it"
        )
    dx_resident = (
        cluster_resident(device, geometry.dx_cluster_ctas, num_sms)
        if record is not None
        else None
    )
    plan_of = functools.partial(
        make_plan,
        problem,
        need_dx=need_dx,
        need_dw=need_dw,
        geometry=geometry,
        dx_max_slices=dx_max_slices(stages) if record is not None else 1,
        num_sms=num_sms,
        valid_rows=None if row_index is None else int(row_index.numel()),
        dx_resident=dx_resident,
        fuse_dw_cast=_resolve_fuse(fuse_dw_cast) if need_dw else False,
        arch=record["arch"] if record is not None else None,
        gemm_tuning=tuning,
        dx_finalize=bool(need_dx and _resolve_dx_finalize(dx_finalize)),
        dx_cast=bool(dx_cast),
        logprob_dz=entry == "logprob"
        and _logprob_arm(
            record, stages, LOGPROB_DZ_STAGE, _resolve_logprob_dz(logprob_dz)
        ),
        logprob_lbl=entry == "logprob"
        and _logprob_arm(
            record, stages, LOGPROB_LBL_STAGE, _resolve_logprob_lbl(logprob_lbl)
        ),
        dx_slice_policy=slice_policy,
        dx_red_min_rows=red_min_rows,
        logprob_keep=keep,
        # a kept plan folds by the one rule (:func:`loop_chunks`) under the kept form's ``tail_fold`` only
        tail_fold=bool(kept_fold) if keep else True,
    )
    # the eager skip path (``_skip_forms``, :func:`backward_logprob`) tries the cast at the last live chunk first; a
    # plan takes it whole or not at all (every accumulate but the last chunk's and the deferred cast)
    for live_cast in (True, False) if _skip_forms else (False,):
        plan = plan_of(dcnt=dcnt is not None, live_cast=live_cast)
        if plan.dcnt and any(s not in stages for s in plan.stages):
            # a record without the counted TMA cast: the counted plan with the shipped cast's counted form
            plan = plan_of(dcnt=True, counted_tma_cast=False, live_cast=live_cast)
        if live_cast and not (plan.live_cast and all(s in stages for s in plan.stages)):
            continue
        break
    if plan.dcnt and any(s not in stages for s in plan.stages):
        # a record without the counted forms: the shipped plan, every row computed
        plan = plan_of(dcnt=False)
    missing = [s for s in plan.stages if s not in stages]
    if missing:
        raise NotImplementedError(
            f"the registered program {module_name!r} lacks the stages {missing} (registered: {stages})"
        )
    T, H, V, C = problem.num_rows, problem.hidden, problem.vocab, problem.chunk
    rows_loop = (
        plan.rows
    )  # rows the chunk loop processes (the valid rows when compacted)
    # the short-tail fold (:func:`plan_chunks`): the chunk buffers hold the folded chunk (more than C rows)
    folded = len(plan.chunks) < len(plan_chunks(rows_loop, C))
    tma_bytes = _record_tma_bytes(record, stages) if record is not None else 0
    scratch_bytes = _record_scratch_bytes(record, stages) if record is not None else 0
    layout = workspace_layout(
        rows_loop,
        V,
        max([int(C)] + [int(rows_c) for _, rows_c in plan.chunks]),
        stats_tile=geometry.stats_tile,
        entry=entry,
        tma_workspace_bytes=tma_bytes,
        scratch_bytes=scratch_bytes,
        hidden=H,
        compact=plan.compact,
    )
    if workspace_buffer is None:
        workspace_buffer = torch.empty(
            layout["total"], dtype=torch.uint8, device=device
        )
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < layout["total"]:
        raise ValueError(
            f"workspace_buffer needs {layout['total']} bytes, got {flat.numel()}"
        )

    def output(given, name, shape, dtype):
        _check_output(given, name, shape, dtype)
        return (
            given
            if given is not None
            else torch.empty(shape, dtype=dtype, device=device)
        )

    t: dict[str, Any] = dict(
        X=_prepare_x(X, problem, plan.compact),
        W=W,
        labels=_prepare_labels(labels, geometry, row_index),
    )
    rows = layout["logits"][1] // (V * 2)
    t["logits"] = _carve(flat, layout, "logits", torch.bfloat16, (rows, V))
    t["stats"] = _carve(flat, layout, "stats", torch.float32, (rows, plan.num_tiles, 2))
    t["d"] = _carve(flat, layout, "d", torch.float32, (rows,))
    t["term"] = _carve(flat, layout, "term", torch.float32, (rows,))
    t["loss_acc"] = _carve(flat, layout, "loss_acc", torch.float64, (1,))
    t["grad_scale"] = _carve(flat, layout, "grad_scale", torch.float32, (1,))
    t["unit_scale"] = torch.ones(
        (1,), dtype=torch.float32, device=device
    )  # the log-probability entry's cast scale
    t["f32_dummy"] = torch.zeros(
        (16,), dtype=torch.float32, device=device
    )  # unused ``WS`` of the plain-store GEMMs
    if plan.compact:
        t["row_index"] = row_index
        t["x_c"] = _carve(
            flat, layout, "x_c", torch.bfloat16, (rows, H)
        )  # the chunk's valid X rows, gathered per chunk
    # O(T) row vectors and the loss cell are separate allocations: a caller (or the autograd graph) keeping
    # ``loss`` / ``logp`` / ``lse`` alive must not pin the vocabulary workspace.
    t["lse"] = output(lse, "lse", (rows_loop,), torch.float32)
    t["logp"] = output(logp, "logp", (rows_loop,), torch.float32)
    t["loss"] = torch.empty((1,), dtype=torch.float32, device=device)
    if entry == "loss":
        if infer_logp is not None:
            if (
                plan.compact
            ):  # gathered into their compact [T_v] forms at the start of every step
                t["infer_logp_full"], t["loss_weights_full"] = infer_logp, loss_weights
                t["infer_logp"] = torch.empty(
                    (rows_loop,), dtype=torch.float32, device=device
                )
                t["loss_weights"] = torch.empty(
                    (rows_loop,), dtype=torch.float32, device=device
                )
            else:
                t["infer_logp"], t["loss_weights"] = infer_logp, loss_weights
    else:
        d_in = output(dlogp, "dlogp", (T,), torch.float32)
        if plan.compact:
            t["d_in_full"] = d_in
            t["d_in"] = torch.empty((rows_loop,), dtype=torch.float32, device=device)
        else:
            t["d_in"] = d_in
        if plan.dcnt:
            _check_output(dcnt, "dcnt", (plan.num_chunks,), torch.int32)
            t["dcnt"] = dcnt
    if plan.need_dx:
        t["dx_acc"] = output(dx_acc, "dx_acc", (rows_loop, H), torch.float32)
        if plan.dx_cast:
            t["dx_out"] = output(dx_out, "dx_out", (rows_loop, H), torch.bfloat16)
    if plan.need_dw:
        if plan.dw_acc_needed:
            t["dw_acc"] = output(dw_acc, "dw_acc", (V, H), torch.float32)
        t["dw_out"] = output(
            dw_out,
            "dw_out",
            (V, H),
            torch.float32
            if plan.dw_cast_stage in ("scale_cast_f32", "gemm_dw_cast_f32")
            else torch.bfloat16,
        )
    if layout.get("workspace"):
        t["workspace"] = _carve(
            flat, layout, "workspace", torch.uint8, (layout["workspace"][1],)
        )
    if layout.get("tma_descriptor_workspace"):
        t["tma_descriptor_workspace"] = _carve(
            flat,
            layout,
            "tma_descriptor_workspace",
            torch.uint8,
            (layout["tma_descriptor_workspace"][1],),
        )

    fwd = forward_keys(plan)
    bwd = (recompute_keys(plan) if entry == "logprob" else ()) + cast_keys(plan)
    # a ``dz``-arm plan also binds chunk 0's ``row_grad``: the hidden valid-row count's backward recomputes chunk 0 in
    # the device-count form, which stores ``z`` (:meth:`_Binding.from_runner`); the runner itself never launches it
    spare = (("row_grad", 0),) if plan.logprob_dz else ()
    values = {key: stage_values(key[0], t, plan, key[1]) for key in fwd + bwd + spare}
    launches: dict[Any, _Launch] = {}
    engine = None
    if backend == "cake":
        launches = _bind_all(
            record, module_name, fwd + bwd + spare, values, device, geometry
        )
    else:
        engine = ReferenceEngine()
    memory = memory_report(
        T,
        H,
        V,
        C,
        stats_tile=geometry.stats_tile,
        entry=entry,
        need_dx=plan.need_dx,
        need_dw=plan.need_dw,
        grad_weight_dtype=problem.grad_weight_dtype,
        return_logp=True,
        x_copy=problem.x_copy and not plan.compact,
        tma_workspace_bytes=tma_bytes,
        scratch_bytes=scratch_bytes,
        valid_rows=plan.valid_rows,
        fuse_dw_cast=plan.fuse_dw_cast,
        dx_finalize=plan.dx_finalize,
        tail_fold=folded,
        dx_slab_bytes=plan.dx_slab_bytes,
    )
    if plan.logprob_keep:
        memory = kept_memory_report(memory, plan)
    if device.type == "cuda":
        device_index = int(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    else:
        device_index = 0
    return (_KeptLossRunner if plan.logprob_keep else LmHeadLossRunner)(
        backend=backend,
        module_name=module_name,
        plan=plan,
        tensors=t,
        values=values,
        launches=launches,
        forward_order=fwd,
        backward_order=bwd,
        device_index=device_index,
        workspace=flat,
        layout=layout,
        memory=memory,
        engine=engine,
        dw_stream=stream_mode,
    )


def prepare_lm_head_loss_temperature(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div=None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    need_dx: bool = True,
    need_dw: bool = True,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    entry: str = "loss",
    dlogp: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    dx_acc: Optional[torch.Tensor] = None,
    dw_acc: Optional[torch.Tensor] = None,
    dx_out: Optional[torch.Tensor] = None,
    dw_out: Optional[torch.Tensor] = None,
    logp: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    backend: str = "cake",
    compact_rows=False,
    fuse_dw_cast: Optional[bool] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    dx_finalize: Optional[bool] = None,
    dx_cast: bool = True,
    dw_stream: Optional[Union[bool, str]] = None,
    logprob_dz: Optional[bool] = None,
    logprob_lbl: Optional[bool] = None,
    dx_rule: Optional[tuple] = None,
    temperature: Optional[torch.Tensor] = None,
    dcnt: Optional[torch.Tensor] = None,
    _skip_forms: bool = False,
) -> LmHeadLossRunner:
    """Validate one binding and prepare its launches.

    Missing outputs, accumulators and the workspace are allocated here (the
    only allocations of the ``cake`` backend).  Pass ``workspace_buffer`` of
    :func:`lm_head_loss_workspace_size` bytes to reuse storage across steps.
    For the log-probability entry ``dlogp`` is the FP32 ``[T]`` tensor the
    backward reads (allocated here when absent) and ``lse`` the statistic the
    forward writes / the backward reads.  ``T == 0`` is rejected here (the
    eager entry points return zeros for it without a runner).
    ``backend="reference"`` builds the same runner over PyTorch operators.

    ``compact_rows=True`` compacts the chunk loop to the valid rows of
    ``labels`` (:func:`valid_row_index`; an int64 index tensor is accepted in
    place of ``True``): ``lse`` / ``logp`` / ``dx_acc`` / ``dx_out`` are then
    bound or allocated as compact ``[T_v]`` / ``[T_v, H]`` tensors, the row
    operands are gathered per step and every row ignored is rejected (the
    eager entry points return zeros for it).

    ``fuse_dw_cast`` (default :func:`fuse_dw_cast_default`): the last chunk's
    weight-gradient GEMM runs in ``backward()`` with the scale + cast fused
    (``dw_acc`` is then the FP32 sum of the chunks before it and absent for a
    one-chunk plan); the chunk buffer must not be overwritten between
    ``forward()`` and ``backward()``.

    The registered program of the ``(H, V)`` geometry is selected
    (:func:`record_for`) and each chunk's GEMM instance variants follow the
    per-chunk rules of its architecture (:func:`chunk_variants`);
    ``gemm_tuning`` (default :func:`gemm_tuning_default`) pins knobs
    explicitly, e.g. ``{"logits": {"group_m": 16}}``.

    ``dx_finalize`` (default :func:`dx_finalize_default`) is recorded in the
    plan (the eager entry points' compacted dX output boundary; the runner's
    own ``dX`` output is the flat cast).  A chunk of more than one dX slice
    stores the sum of its slices into ``dx_acc`` (:data:`DX_RED_STAGE`, then
    its slab sum :data:`DX_SLAB_SUM_STAGE`) without reading it, so a bound
    ``dx_acc`` may hold any values between steps;
    the slice rule's settings are ``dx_rule`` (``(policy, red_min_rows)``;
    ``None`` = :func:`dx_slice_rule` at preparation).  ``dx_cast=False``
    leaves the flat ``dX`` cast out of ``backward()``, which then returns
    ``dx_acc`` for the caller to finalize (:func:`finalize_dx`: the fused dX
    finalize of a compacted :func:`backward_logprob`); ``dx_out`` is then
    neither bound nor allocated.

    ``dw_stream`` (default ``None`` = ``"auto"``, :func:`dw_stream_mode`): the
    dW side stream of the runner's launches -- ``None`` / ``"auto"`` for plans
    of :data:`DW_STREAM_MIN_CHUNKS` chunks or more, ``True`` for every
    multi-chunk plan, ``False`` never (bitwise the same outputs; the runner's
    ``dw_stream`` attribute holds the mode).

    ``logprob_dz`` / ``logprob_lbl`` (log-probability entry only; defaults
    :func:`logprob_dz_default` / :func:`logprob_lbl_default`): the backward's
    recompute writes ``dz`` from the ``gemm_logits_dz`` epilogue and the
    forward's logits GEMM is ``gemm_logits_lbl``, where the record registers
    them (bitwise the same outputs); ``False`` keeps ``gemm_logits_nostats`` +
    ``row_grad`` / ``gemm_logits``.  A loss-entry runner rejects ``True``.

    ``temperature`` (log-probability entry only; ``None`` = none): the contiguous FP32 per-row temperature of the
    chunk loop's rows -- like ``lse``, the compact ``[T_v]`` rows of the valid-row index when the runner is compacted,
    else ``[T]`` -- bound as ``tensors["temperature"]`` (new values may be written into it between steps).  Its values
    must be positive and finite; the runner does not check them (the eager entry points validate the caller's
    ``temperature``, :func:`_resolve_temperature`).  On the ``cake`` backend the plan runs the temperature forms
    (:data:`TEMPERATURE_STAGES`, both logits GEMM arms: ``logprob_dz`` / ``logprob_lbl`` must not be ``False``; a
    record without them raises ``NotImplementedError``); the reference backend reads it in the same stages.

    ``dcnt`` (log-probability entry, compacted rows ordered by :func:`zero_dlogp_partition`): the int32 per-chunk
    counts of live rows, as in :func:`prepare_lm_head_loss`; with a temperature the plan of
    :func:`make_plan_temperature` whose forms the record registers: the counted ``dz`` temperature form
    (:func:`zero_dlogp_temperature_dz_counted`), else the mixed plan (the uncounted ``dz`` temperature form), each
    with the counted dX / dW forms, and the shipped forms otherwise.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if entry != "logprob" and (logprob_dz or logprob_lbl):
        raise ValueError(
            "logprob_dz / logprob_lbl select logits GEMM arms of the log-probability entry"
        )
    if temperature is not None and entry != "logprob":
        raise ValueError("temperature belongs to the log-probability entry")
    _check_temperature_arms(temperature is not None, logprob_lbl, logprob_dz)
    device = X.device
    record = None
    module_name = None
    tuning = _resolve_gemm_tuning(gemm_tuning)
    stream_mode = dw_stream_mode(dw_stream)
    slice_policy, red_min_rows = _resolve_dx_rule(dx_rule)
    if backend == "cake":
        if not X.is_cuda:
            raise ValueError("the cake backend needs CUDA tensors")
        module_name, record = record_for(device)
        record_abi(record)
    geometry = Geometry.from_record(record)
    problem = validate_lm_head_inputs(
        X,
        W,
        labels,
        objective=objective,
        loss_div=loss_div,
        infer_logp=infer_logp,
        loss_weights=loss_weights,
        chunk_size=chunk_size,
        grad_weight_dtype=grad_weight_dtype,
        entry=entry,
        geometry=geometry,
    )
    if temperature is not None:
        problem = _with_temperature(problem)
    if problem.num_rows == 0:
        raise ValueError(
            "a prepared runner needs at least one row (T > 0); the eager entry points return zeros for T == 0"
        )
    given = [
        t
        for t in (
            W,
            labels,
            infer_logp,
            loss_weights,
            dlogp,
            workspace_buffer,
            dx_acc,
            dw_acc,
            dx_out,
            dw_out,
            logp,
            lse,
            temperature,
        )
        if t is not None
    ]
    if not all(t.device == device for t in given):
        raise ValueError("Expected all tensors on one device")
    stages = (
        (
            registered_stages_with_temperature(module_name)
            if problem.temperature
            else registered_stages(module_name)
        )
        if record is not None
        else tuple(STAGES)
    )
    num_sms = (
        int(torch.cuda.get_device_properties(device).multi_processor_count)
        if device.type == "cuda"
        else 1
    )
    row_index = None
    if isinstance(compact_rows, torch.Tensor):
        row_index = compact_rows
        if (
            row_index.dtype != torch.int64
            or row_index.ndim != 1
            or row_index.device != device
        ):
            raise ValueError(
                "compact_rows given as a tensor must be the int64 [T_v] valid-row index on X's device"
            )
    elif compact_rows:
        row_index = valid_row_index(labels)
    if row_index is not None and row_index.numel() == 0:
        raise ValueError(
            "a prepared runner needs at least one valid row (every label is ignored); the eager entry points return zeros for it"
        )
    dx_resident = (
        cluster_resident(device, geometry.dx_cluster_ctas, num_sms)
        if record is not None
        else None
    )
    plan_of = functools.partial(
        make_plan_temperature
        if problem.temperature and record is not None
        else make_plan,
        problem,
        need_dx=need_dx,
        need_dw=need_dw,
        geometry=geometry,
        dx_max_slices=dx_max_slices(stages) if record is not None else 1,
        num_sms=num_sms,
        valid_rows=None if row_index is None else int(row_index.numel()),
        dx_resident=dx_resident,
        fuse_dw_cast=_resolve_fuse(fuse_dw_cast) if need_dw else False,
        arch=record["arch"] if record is not None else None,
        gemm_tuning=tuning,
        dx_finalize=bool(need_dx and _resolve_dx_finalize(dx_finalize)),
        dx_cast=bool(dx_cast),
        logprob_dz=entry == "logprob"
        and _logprob_arm_temperature(
            record,
            stages,
            LOGPROB_DZ_STAGE,
            _resolve_logprob_dz(logprob_dz),
            temperature=problem.temperature,
        ),
        logprob_lbl=entry == "logprob"
        and _logprob_arm_temperature(
            record,
            stages,
            LOGPROB_LBL_STAGE,
            _resolve_logprob_lbl(logprob_lbl),
            temperature=problem.temperature,
        ),
        dx_slice_policy=slice_policy,
        dx_red_min_rows=red_min_rows,
    )
    # under the skip, the first plan whose stages the record registers: the counted temperature dz arm
    # (:func:`zero_dlogp_temperature_dz_counted`), then the uncounted one (the mixed plan), each with the counted TMA
    # cast, then with the shipped cast's counted form; else the shipped plan, every row computed
    dz_forms: tuple[dict[str, bool], ...] = ({},)
    if problem.temperature and record is not None:
        dz_forms = tuple(
            {"counted_dz": counted}
            for counted in (
                (True, False) if zero_dlogp_temperature_dz_counted() else (False,)
            )
        )
    plan = plan_of(dcnt=dcnt is not None)
    if plan.dcnt:
        # the eager skip path (``_skip_forms``, :func:`backward_logprob_temperature`) tries the cast at the last live
        # chunk first (whole or not at all), each with the dz forms, each with the two cast forms
        searched = False
        for live_cast in (True, False) if _skip_forms else (False,):
            for dz_form in dz_forms:
                for cast_form in ({}, {"counted_tma_cast": False}):
                    plan = plan_of(
                        dcnt=True, live_cast=live_cast, **dz_form, **cast_form
                    )
                    if live_cast and not plan.live_cast:
                        break
                    if not plan.dcnt or all(s in stages for s in plan.stages):
                        searched = True
                        break
                if searched or (live_cast and not plan.live_cast):
                    break
            if searched:
                break
    if plan.dcnt and any(s not in stages for s in plan.stages):
        # a record without the counted forms: the shipped plan, every row computed
        plan = plan_of(dcnt=False)
    missing = [s for s in plan.stages if s not in stages]
    if missing:
        raise NotImplementedError(
            f"the registered program {module_name!r} lacks the stages {missing} (registered: {stages})"
        )
    T, H, V, C = problem.num_rows, problem.hidden, problem.vocab, problem.chunk
    rows_loop = (
        plan.rows
    )  # rows the chunk loop processes (the valid rows when compacted)
    # the short-tail fold (:func:`plan_chunks`): the chunk buffers hold the folded chunk (more than C rows)
    folded = len(plan.chunks) < len(plan_chunks(rows_loop, C))
    tma_bytes = _record_tma_bytes(record, stages) if record is not None else 0
    scratch_bytes = _record_scratch_bytes(record, stages) if record is not None else 0
    layout = workspace_layout(
        rows_loop,
        V,
        max([int(C)] + [int(rows_c) for _, rows_c in plan.chunks]),
        stats_tile=geometry.stats_tile,
        entry=entry,
        tma_workspace_bytes=tma_bytes,
        scratch_bytes=scratch_bytes,
        hidden=H,
        compact=plan.compact,
    )
    if workspace_buffer is None:
        workspace_buffer = torch.empty(
            layout["total"], dtype=torch.uint8, device=device
        )
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < layout["total"]:
        raise ValueError(
            f"workspace_buffer needs {layout['total']} bytes, got {flat.numel()}"
        )

    def output(given, name, shape, dtype):
        _check_output(given, name, shape, dtype)
        return (
            given
            if given is not None
            else torch.empty(shape, dtype=dtype, device=device)
        )

    t: dict[str, Any] = dict(
        X=_prepare_x(X, problem, plan.compact),
        W=W,
        labels=_prepare_labels(labels, geometry, row_index),
    )
    rows = layout["logits"][1] // (V * 2)
    t["logits"] = _carve(flat, layout, "logits", torch.bfloat16, (rows, V))
    t["stats"] = _carve(flat, layout, "stats", torch.float32, (rows, plan.num_tiles, 2))
    t["d"] = _carve(flat, layout, "d", torch.float32, (rows,))
    t["term"] = _carve(flat, layout, "term", torch.float32, (rows,))
    t["loss_acc"] = _carve(flat, layout, "loss_acc", torch.float64, (1,))
    t["grad_scale"] = _carve(flat, layout, "grad_scale", torch.float32, (1,))
    t["unit_scale"] = torch.ones(
        (1,), dtype=torch.float32, device=device
    )  # the log-probability entry's cast scale
    t["f32_dummy"] = torch.zeros(
        (16,), dtype=torch.float32, device=device
    )  # unused ``WS`` of the plain-store GEMMs
    if plan.compact:
        t["row_index"] = row_index
        t["x_c"] = _carve(
            flat, layout, "x_c", torch.bfloat16, (rows, H)
        )  # the chunk's valid X rows, gathered per chunk
    # O(T) row vectors and the loss cell are separate allocations: a caller (or the autograd graph) keeping
    # ``loss`` / ``logp`` / ``lse`` alive must not pin the vocabulary workspace.
    t["lse"] = output(lse, "lse", (rows_loop,), torch.float32)
    t["logp"] = output(logp, "logp", (rows_loop,), torch.float32)
    t["loss"] = torch.empty((1,), dtype=torch.float32, device=device)
    if entry == "loss":
        if infer_logp is not None:
            if (
                plan.compact
            ):  # gathered into their compact [T_v] forms at the start of every step
                t["infer_logp_full"], t["loss_weights_full"] = infer_logp, loss_weights
                t["infer_logp"] = torch.empty(
                    (rows_loop,), dtype=torch.float32, device=device
                )
                t["loss_weights"] = torch.empty(
                    (rows_loop,), dtype=torch.float32, device=device
                )
            else:
                t["infer_logp"], t["loss_weights"] = infer_logp, loss_weights
    else:
        d_in = output(dlogp, "dlogp", (T,), torch.float32)
        if plan.compact:
            t["d_in_full"] = d_in
            t["d_in"] = torch.empty((rows_loop,), dtype=torch.float32, device=device)
        else:
            t["d_in"] = d_in
        if temperature is not None:
            # the chunk loop's rows (compact when compacted), read at ``row0 + r`` like ``lse``
            _check_output(temperature, "temperature", (rows_loop,), torch.float32)
            t["temperature"] = temperature
        if plan.dcnt:
            _check_output(dcnt, "dcnt", (plan.num_chunks,), torch.int32)
            t["dcnt"] = dcnt
    if plan.need_dx:
        t["dx_acc"] = output(dx_acc, "dx_acc", (rows_loop, H), torch.float32)
        if plan.dx_cast:
            t["dx_out"] = output(dx_out, "dx_out", (rows_loop, H), torch.bfloat16)
    if plan.need_dw:
        if plan.dw_acc_needed:
            t["dw_acc"] = output(dw_acc, "dw_acc", (V, H), torch.float32)
        t["dw_out"] = output(
            dw_out,
            "dw_out",
            (V, H),
            torch.float32
            if plan.dw_cast_stage in ("scale_cast_f32", "gemm_dw_cast_f32")
            else torch.bfloat16,
        )
    if layout.get("workspace"):
        t["workspace"] = _carve(
            flat, layout, "workspace", torch.uint8, (layout["workspace"][1],)
        )
    if layout.get("tma_descriptor_workspace"):
        t["tma_descriptor_workspace"] = _carve(
            flat,
            layout,
            "tma_descriptor_workspace",
            torch.uint8,
            (layout["tma_descriptor_workspace"][1],),
        )

    fwd = _forward_keys_temperature(plan)
    bwd = (_recompute_keys_temperature(plan) if entry == "logprob" else ()) + cast_keys(
        plan
    )
    # a ``dz``-arm plan also binds chunk 0's ``row_grad``: the hidden valid-row count's backward recomputes chunk 0 in
    # the device-count form, which stores ``z`` (:meth:`_Binding.from_runner`); the runner itself never launches it.  A
    # call with a temperature never recounts on the hidden path (no temperature form of ``row_grad``)
    spare = (("row_grad", 0),) if plan.logprob_dz and not problem.temperature else ()
    values_of = stage_values_temperature if problem.temperature else stage_values
    values = {key: values_of(key[0], t, plan, key[1]) for key in fwd + bwd + spare}
    launches: dict[Any, _Launch] = {}
    engine = None
    if backend == "cake":
        if problem.temperature:
            _check_temperature_operands(module_name, fwd + bwd + spare, values)
            launches = _bind_all(
                record,
                module_name,
                fwd + bwd + spare,
                values,
                device,
                _TemperatureGeometry.of(geometry),
            )
        else:
            launches = _bind_all(
                record, module_name, fwd + bwd + spare, values, device, geometry
            )
    else:
        engine = ReferenceEngine()
    memory = memory_report_temperature(
        T,
        H,
        V,
        C,
        stats_tile=geometry.stats_tile,
        entry=entry,
        need_dx=plan.need_dx,
        need_dw=plan.need_dw,
        grad_weight_dtype=problem.grad_weight_dtype,
        return_logp=True,
        x_copy=problem.x_copy and not plan.compact,
        tma_workspace_bytes=tma_bytes,
        scratch_bytes=scratch_bytes,
        valid_rows=plan.valid_rows,
        fuse_dw_cast=plan.fuse_dw_cast,
        dx_finalize=plan.dx_finalize,
        dx_slab_bytes=plan.dx_slab_bytes,
        temperature=problem.temperature,
        tail_fold=folded,
    )
    if device.type == "cuda":
        device_index = int(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    else:
        device_index = 0
    runner = _TemperatureRunner if plan.temperature else LmHeadLossRunner
    return runner(
        backend=backend,
        module_name=module_name,
        plan=plan,
        tensors=t,
        values=values,
        launches=launches,
        forward_order=fwd,
        backward_order=bwd,
        device_index=device_index,
        workspace=flat,
        layout=layout,
        memory=memory,
        engine=engine,
        dw_stream=stream_mode,
    )


# ---------------------------------------------------------------------------
# Binding cache of the eager entry points (cake backend)
# ---------------------------------------------------------------------------

BINDING_CACHE_DEFAULT_CAPACITY = 64  # bindings :data:`BINDING_CACHE` keeps by default (:meth:`BindingCache.configure`)


def _meta(t: Optional[torch.Tensor]):
    return (
        None
        if t is None
        else (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
    )


def _layout(t: Optional[torch.Tensor]):
    """``(shape, stride, dtype)`` of ``t``: the key part of a tensor a remembered binding slots per call and whose
    address no plan reads."""
    return None if t is None else (tuple(t.shape), tuple(t.stride()), t.dtype)


def _rule_key(dx_rule: Optional[tuple]) -> tuple:
    """The dX slice rule's part of a binding key: the pair the entry call read (:func:`dx_slice_rule`) as is, or the
    current one."""
    return dx_slice_rule() if dx_rule is None else tuple(dx_rule)


def forward_binding_key(
    X,
    W,
    labels,
    *,
    objective,
    loss_div,
    infer_logp,
    loss_weights,
    chunk_size,
    need_dx,
    need_dw,
    grad_weight_dtype,
    entry,
    valid_rows=None,
    fuse_dw_cast=False,
    gemm_tuning=None,
    dx_finalize=False,
    hidden_count=False,
    logprob_lbl=False,
    dx_rule=None,
) -> tuple:
    """Cache key of a forward binding: ``(data_ptr, shape, stride, dtype)`` of every
    input plus every option that shapes the argument plans; ``valid_rows`` is the
    compacted row count (a label-dependent fact of the plan; ``None`` = uncompacted);
    ``fuse_dw_cast`` the resolved weight-gradient form; ``gemm_tuning`` the explicit
    GEMM knobs (``None`` = none, :func:`gemm_tuning_default`);
    ``dx_finalize`` the resolved dX finalize form;
    ``hidden_count`` whether chunk 0's gather and logits GEMM ran before the plan
    (the remembered sequence starts at chunk 0's row kernels); ``logprob_lbl``
    the resolved log-probability forward's logits form (:func:`logprob_lbl_default`).
    The key ends with the dX slice rule's settings ``dx_rule`` (``(policy,
    red_min_rows)``; ``None`` = :func:`dx_slice_rule`), which shape the plan's dX
    forms: an entry call reads them once and plans its binding with the same pair."""
    return (
        "fwd",
        entry,
        _meta(X),
        _meta(W),
        _meta(labels),
        _meta(infer_logp),
        _meta(loss_weights),
        objective,
        None if loss_div is None else float(loss_div),
        int(chunk_size),
        bool(need_dx),
        bool(need_dw),
        grad_weight_dtype,
        None if valid_rows is None else int(valid_rows),
        bool(fuse_dw_cast),
        _tuning_key(_resolve_gemm_tuning(gemm_tuning)),
        bool(dx_finalize),
        bool(hidden_count),
        bool(logprob_lbl),
        _rule_key(dx_rule),
    )


def logprob_backward_binding_key(
    X,
    W,
    labels,
    lse,
    dlogp,
    *,
    chunk_size,
    need_dx,
    need_dw,
    valid_rows=None,
    fuse_dw_cast=False,
    gemm_tuning=None,
    dx_finalize=False,
    hidden_count=False,
    logprob_dz=False,
    dx_rule=None,
    zero_dlogp=False,
) -> tuple:
    """Cache key of a log-probability backward binding (as :func:`forward_binding_key`, the dX slice rule's settings
    ``dx_rule`` included); ``logprob_dz`` the resolved recompute form (:func:`logprob_dz_default`), ``zero_dlogp``
    whether the call is eligible for the zero-gradient row skip (:func:`zero_dlogp_partition`; the remembered plan
    tells whether it runs its counted forms).  ``lse`` (the
    forward's statistic) and ``dlogp`` enter by layout only (:func:`_layout`): the binding slots both per call and
    no plan reads their addresses, so a backward whose ``lse`` / ``dlogp`` the caching allocator placed elsewhere --
    each forward allocates a fresh ``lse`` -- runs from the remembered binding instead of binding again."""
    return (
        "bwd",
        "logprob",
        _meta(X),
        _meta(W),
        _meta(labels),
        _layout(lse),
        _layout(dlogp),
        int(chunk_size),
        bool(need_dx),
        bool(need_dw),
        None if valid_rows is None else int(valid_rows),
        bool(fuse_dw_cast),
        _tuning_key(_resolve_gemm_tuning(gemm_tuning)),
        bool(dx_finalize),
        bool(hidden_count),
        bool(logprob_dz),
        bool(zero_dlogp),
        _rule_key(dx_rule),
    )


def temperature_binding_key(key: tuple) -> tuple:
    """The binding key of a call with a temperature: its :func:`forward_binding_key` /
    :func:`logprob_backward_binding_key` plus the flag (the temperature forms) -- never the temperature tensor's
    address or values: every tensor of a remembered launch, the temperature included, is slotted per call."""
    return key + ("temperature",)


def _kept_views(plan: Plan, chunk_buffer: torch.Tensor, device) -> tuple:
    """The kept logits of a kept-logits forward (:class:`_KeptPlan`): one BF16 ``[rows_c, V]`` view per chunk --
    chunk 0's rows of the call's chunk buffer (also the hidden valid-row count's prologue buffer, which chunk 0's
    device-count GEMM already filled), the later chunks' rows of one fresh ``[rows - rows_0, V]`` allocation."""
    V = plan.problem.vocab
    _, rows_first = plan.chunks[0]
    views = [chunk_buffer[:rows_first]]
    rest_rows = plan.rows - rows_first
    if rest_rows > 0:
        rest = torch.empty((rest_rows, V), dtype=torch.bfloat16, device=device)
        for row0, rows_c in plan.chunks[1:]:
            start = row0 - rows_first
            views.append(rest[start : start + rows_c])
    return tuple(views)


def kept_rows_of(views) -> Optional[torch.Tensor]:
    """The one contiguous BF16 ``[rows, V]`` buffer whose consecutive row slices, in order from its first row, are the
    kept views ``views`` of a kept-logits forward (:meth:`_KeptBinding.forward_keep`): the buffer itself when the views
    cover it, else its first ``rows`` rows (the hidden valid-row count's ``[T, V]`` kept buffer holds ``T_v <= T``
    rows); ``None`` for any other views.  A backward may take it as one chunk (:func:`backward_logprob_keep` with
    ``chunk_size`` >= rows)."""
    views = tuple(views or ())
    if not views or not all(isinstance(v, torch.Tensor) for v in views):
        return None
    base = views[0]._base
    if (
        base is None
        or base.dim() != 2
        or base.dtype != torch.bfloat16
        or not base.is_contiguous()
    ):
        return None
    V, row = int(base.shape[1]), 0
    for view in views:
        if (
            view._base is not base
            or view.dim() != 2
            or int(view.shape[1]) != V
            or not view.is_contiguous()
            or view.storage_offset() != base.storage_offset() + row * V
        ):
            return None
        row += int(view.shape[0])
    if row == int(base.shape[0]):
        return base
    return base[:row] if 0 < row < int(base.shape[0]) else None


# the attribute :func:`backward_logprob_keep` sets on kept views it consumed (``row_grad`` overwrote them with ``dz``)
_KEPT_SPENT = "_cake_lm_head_kept_logits_spent"


def check_kept_logits(plan: Plan, logits) -> tuple:
    """The forward's kept logits as a backward of ``plan`` binds them: one BF16 contiguous ``[rows_c, V]`` CUDA view per
    chunk of the plan, on the plan's rows, not yet consumed by a backward (``ValueError`` otherwise)."""
    views = tuple(logits)
    if any(getattr(view, _KEPT_SPENT, False) for view in views):
        raise ValueError(
            "logits: these kept views were consumed by an earlier backward (row_grad overwrote them with dz); run "
            "forward_logprob_keep again for another backward"
        )
    V = plan.problem.vocab
    if len(views) != plan.num_chunks:
        raise ValueError(
            f"logits: {len(views)} kept chunk view(s) for a plan of {plan.num_chunks} chunk(s)"
        )
    for (_, rows_c), view in zip(plan.chunks, views, strict=True):
        if (
            not isinstance(view, torch.Tensor)
            or view.dtype != torch.bfloat16
            or tuple(view.shape) != (rows_c, V)
            or not view.is_contiguous()
            or not view.is_cuda
        ):
            raise ValueError(
                f"logits: every kept chunk view must be a contiguous BF16 [rows_c, V] CUDA tensor of the forward "
                f"(chunk of {rows_c} rows, V = {V}); got {getattr(view, 'dtype', type(view))} "
                f"{tuple(getattr(view, 'shape', ()))}"
            )
    return views


# Values a remembered binding owns: the descriptor workspace of pointer-ABI stages and kernel-private scratch (a few
# kilobytes).  The vocabulary workspace, the row vectors, the accumulators and the outputs are allocated per call from
# the caching allocator (their sizes are part of the plan; the allocator reuses the blocks).  No registered stage
# declares either (``workspace_bytes`` / ``tma_workspace_bytes`` 0, no ``tma_prepare_entry``), so a binding owns
# nothing: calls with the same inputs in flight at once on two streams share one binding -- the key holds no stream --
# and only its templated launches and plan; every buffer a call writes is its own allocation on its own stream, and
# the constant cells (:func:`_device_constants`) are read-only.  A stage that declares one needs the stream in the
# binding key first (the suite checks the registry for this).  Calls of two threads share a binding the same way; the
# cache that holds the bindings takes no lock and skips a step whose entry another thread removed (:class:`BindingCache`).
_OWNED_VALUES = ("workspace", "tma_descriptor_workspace")

# The per-call scratch slots of a remembered binding (:meth:`_Binding._scratch`): the row vectors ``d`` / ``term`` and
# the loss cells ``loss_acc`` / ``loss``.  A binding allocates per call only the slots its launch sequence binds
# (:func:`_scratch_slots`): the log-probability forward binds ``d`` and ``term`` (``row_finalize``), its backward none
# of them, the loss entry's forward all four.
_SCRATCH_SLOTS = ("d", "term", "loss_acc", "loss")


class _ReadRecorder(dict):
    """A copy of a call's host values that records the names read from it (``t[name]`` / ``t.get(name)``)."""

    def __init__(self, values: dict[str, Any]):
        super().__init__(values)
        self.read: set[str] = set()

    def __getitem__(self, name):
        self.read.add(name)
        return super().__getitem__(name)

    def get(self, name, default=None):
        self.read.add(name)
        return super().get(name, default)


def _scratch_slots(
    slot_values: Callable[..., dict[str, Any]],
    keys: tuple,
    t: dict[str, Any],
    plan: Plan,
    returned: tuple = (),
) -> tuple[str, ...]:
    """The scratch slots (:data:`_SCRATCH_SLOTS`, in that order) the launch sequence ``keys`` binds -- every name its
    per-call host values read (``slot_values``: :func:`_slot_values` or its temperature form, which the binding's
    launches and host steps take per call) -- plus ``returned`` (the slots the call returns).  Replayed once per binding
    on the runner's own tensors (:meth:`_Binding.from_runner`); a replay that fails keeps every slot."""
    recorder = _ReadRecorder(t)
    try:
        for key in keys:
            slot_values(key[0], recorder, plan, key[1])
    except Exception:  # fail safe: the shipped allocation of every slot
        return _SCRATCH_SLOTS
    read = recorder.read | set(returned)
    return tuple(name for name in _SCRATCH_SLOTS if name in read)


# The log-probability backward's gradient accumulators and outputs (:meth:`_Binding.backward_logprob`).
_GRADIENT_SLOTS = ("dx_acc", "dx_out", "dw_acc", "dw_out")


def _gradients_at_first_dx(
    slot_values: Callable[..., dict[str, Any]],
    keys: tuple,
    t: dict[str, Any],
    plan: Plan,
) -> bool:
    """Whether a log-probability backward can make its gradient accumulators and outputs (:data:`_GRADIENT_SLOTS`)
    right before the first dX GEMM key of ``keys`` (:func:`_before_first_dx`): the sequence has one and no key ahead of
    it reads them.  Replayed once per binding on the runner's own tensors (like :func:`_scratch_slots`); a replay that
    fails keeps them up front."""
    first = next(
        (i for i, key in enumerate(keys) if key[0].startswith("gemm_dx")), None
    )
    if first is None:
        return False
    recorder = _ReadRecorder(t)
    try:
        for key in keys[:first]:
            slot_values(key[0], recorder, plan, key[1])
    except Exception:  # fail safe: made up front
        return False
    return not (recorder.read & set(_GRADIENT_SLOTS))


def _before_first_dx(
    run_key: Callable[[Any], None], before_dx: Optional[Callable[[], None]]
) -> tuple[Callable[[Any], None], Callable[[], None]]:
    """``run_key`` that calls ``before_dx`` once right before the first dX GEMM key -- on the caller's stream:
    :func:`side_stream_schedule` never moves a dX key -- and the call that runs it after the sequence when the sequence
    had none (the host work it does follows the launches ahead of that key)."""
    pending = [] if before_dx is None else [before_dx]

    def run(key) -> None:
        if pending and key[0].startswith("gemm_dx"):
            pending.pop()()
        run_key(key)

    def finish() -> None:
        if pending:
            pending.pop()()

    return run, finish


@dataclass
class _Binding:
    """One remembered input binding: templated launches (no tensor pinned) plus the plan."""

    plan: Plan
    device: torch.device
    device_index: int
    launches: dict[Any, _Launch] = field(repr=False)
    forward_order: tuple = ()
    backward_order: tuple = ()
    # the host-count path's split of ``forward_order`` (:func:`forward_early_keys`): chunk 0's gather and logits GEMM,
    # launched right behind the count -- ``forward_early`` before the call's labels are bound, ``forward_early_labels``
    # (the early launches that read them: the label-column arm) right after -- then ``forward_late`` in order (all
    # empty on the hidden valid-row count path)
    forward_early: tuple = ()
    forward_early_labels: tuple = ()
    forward_late: tuple = ()
    owned: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    owned_bytes: int = 0
    layout: dict = field(default_factory=dict, repr=False)
    memory: dict = field(default_factory=dict, repr=False)
    # the scratch slots (:data:`_SCRATCH_SLOTS`) a forward / backward call allocates: the ones its sequence binds or
    # returns (:func:`_scratch_slots`)
    forward_scratch: tuple = _SCRATCH_SLOTS
    backward_scratch: tuple = _SCRATCH_SLOTS
    # the log-probability backward makes its gradient accumulators and outputs right before its first dX GEMM key
    # (behind the row gathers and the recompute launch) instead of up front (:func:`_gradients_at_first_dx`)
    backward_gradients_at_dx: bool = False

    @classmethod
    def from_runner(
        cls, runner: LmHeadLossRunner, hidden: Optional[_HiddenCount] = None
    ) -> "_Binding":
        """``hidden`` (the eager entry points' hidden valid-row count): the runner was prepared on the counted rows
        after chunk 0's gather and logits GEMM ran in device-count form; the remembered plan records that and its
        launch sequences start at chunk 0's row kernels (the runner's chunk-0 gather / GEMM launches are dropped)."""
        if runner.backend != "cake":
            raise ValueError("only cake-backend runners are remembered")
        t = runner.tensors
        owned = {name: torch.empty_like(t[name]) for name in _OWNED_VALUES if name in t}
        plan, memory = runner.plan, runner.memory
        forward_order, backward_order = runner.forward_order, runner.backward_order
        if hidden is not None:
            p = plan.problem
            plan = replace(
                plan,
                hidden_count=True,
                hidden_logits_group_m=hidden.group_m,
                hidden_stats=hidden.stats,
            )
            forward_order = forward_keys(plan)
            backward_order = (
                recompute_keys(plan) if p.entry == "logprob" else ()
            ) + cast_keys(plan)
            memory = _hidden_count_memory(
                memory,
                num_rows=p.num_rows,
                hidden=p.hidden,
                vocab=p.vocab,
                chunk=p.chunk,
                stats_tile=plan.geometry.stats_tile,
            )
        # the hidden-count path drops the runner's chunk-0 gather / GEMM launches (not in either sequence)
        keep = None if hidden is None else set(forward_order) | set(backward_order)
        # the per-call slots come from :func:`_slot_values`: check it against :func:`stage_values` on the runner's
        # own tensors for every launch and host gather of this binding (fail closed)
        for key, launch in runner.launches.items():
            if keep is None or key in keep:
                _check_slot_values(key, t, plan, [name for _, name in launch.slots])
        for key in forward_order + backward_order:
            if key[0] == "gather_rows":
                _check_slot_values(key, t, plan, ["src", "idx", "out"])
        early = forward_early_keys(plan)
        split = next(
            (i for i, key in enumerate(early) if _reads_labels(key[0])), len(early)
        )
        forward_late = tuple(key for key in forward_order if key not in early)
        return cls(
            plan=plan,
            device=t["X"].device,
            device_index=runner.device_index,
            launches={
                key: launch.templated()
                for key, launch in runner.launches.items()
                if keep is None or key in keep
            },
            forward_order=forward_order,
            backward_order=backward_order,
            forward_early=early[:split],
            forward_early_labels=early[split:],
            forward_late=forward_late,
            owned=owned,
            owned_bytes=sum(v.numel() * v.element_size() for v in owned.values()),
            layout=runner.layout,
            memory=memory,
            # the loss entry's forward returns the finished-loss cell
            forward_scratch=_scratch_slots(
                _slot_values,
                forward_order,
                t,
                plan,
                ("loss",) if plan.problem.entry == "loss" else (),
            ),
            backward_scratch=_scratch_slots(_slot_values, backward_order, t, plan),
            backward_gradients_at_dx=_gradients_at_first_dx(
                _slot_values, backward_order, t, plan
            ),
        )

    def holds_no_tensor(self) -> bool:
        return not any(
            isinstance(a, torch.Tensor)
            for launch in self.launches.values()
            for a in launch.arguments
        )

    def _scratch(
        self,
        t: dict[str, Any],
        hidden: Optional[_HiddenCount] = None,
        *,
        chunk_buffers: bool = True,
        backward: bool = False,
        logits: Optional[torch.Tensor] = None,
    ) -> None:
        """Per-call temporaries from the caching allocator into ``t`` (the runner's regions, minus the owned ones);
        with ``hidden`` the chunk buffers are the prologue's (chunk 0's logits / statistics / gathered rows already in
        them, ``min(chunk, T)`` rows) -- the entry points check that a hidden-count plan gets its prologue.
        ``chunk_buffers=False``: the chunk buffers are already bound (:meth:`_chunk_buffers`, the early launch of
        :meth:`forward`) and the rest is allocated.  Of the scratch slots (:data:`_SCRATCH_SLOTS`) only the ones the
        call's sequence binds or returns: ``backward`` selects the backward's (:attr:`backward_scratch`), else the
        forward's (:attr:`forward_scratch`).  ``logits``: the chunk buffer to bind instead of a new one (a kept-logits
        forward's first rows of its contiguous kept buffer, :meth:`_KeptBinding.forward_keep`)."""
        p = self.plan.problem
        rows = self.layout["logits"][1] // (p.vocab * 2)
        if chunk_buffers and hidden is not None:
            t["logits"], t["stats"] = hidden.logits, hidden.stats_buf
        elif chunk_buffers:
            t["logits"] = (
                torch.empty((rows, p.vocab), dtype=torch.bfloat16, device=self.device)
                if logits is None
                else logits
            )
            t["stats"] = torch.empty(
                (rows, self.plan.num_tiles, 2),
                dtype=torch.float32,
                device=self.device,
            )
        scratch = self.backward_scratch if backward else self.forward_scratch
        if "d" in scratch:
            t["d"] = torch.empty((rows,), dtype=torch.float32, device=self.device)
        if "term" in scratch:
            t["term"] = torch.empty((rows,), dtype=torch.float32, device=self.device)
        if "loss_acc" in scratch:
            t["loss_acc"] = torch.empty((1,), dtype=torch.float64, device=self.device)
        if "loss" in scratch:
            t["loss"] = torch.empty((1,), dtype=torch.float32, device=self.device)
        t.update(
            _device_constants(self.device_index)
        )  # constant cells: no per-call fill launches
        if self.plan.compact and chunk_buffers:
            t["x_c"] = (
                hidden.x_c
                if hidden is not None
                else torch.empty(
                    (rows, p.hidden), dtype=torch.bfloat16, device=self.device
                )
            )
        for name, tensor in self.owned.items():
            t[name] = tensor

    def _chunk_buffers(
        self, t: dict[str, Any], logits: Optional[torch.Tensor] = None
    ) -> None:
        """What chunk 0's row gather and logits GEMM bind (the early launch of :meth:`forward`): the chunk's logits,
        statistics and gather buffers of :meth:`_scratch` (the same allocations, ahead of the others), the constant
        cells and the owned regions.  ``logits``: the chunk buffer to bind instead of a new one (as in
        :meth:`_scratch`)."""
        p = self.plan.problem
        rows = self.layout["logits"][1] // (p.vocab * 2)
        t["logits"] = (
            torch.empty((rows, p.vocab), dtype=torch.bfloat16, device=self.device)
            if logits is None
            else logits
        )
        t["stats"] = torch.empty(
            (rows, self.plan.num_tiles, 2), dtype=torch.float32, device=self.device
        )
        if self.plan.compact:
            t["x_c"] = torch.empty(
                (rows, p.hidden), dtype=torch.bfloat16, device=self.device
            )
        t.update(_device_constants(self.device_index))
        for name, tensor in self.owned.items():
            t[name] = tensor

    def _check_hidden(self, hidden: Optional[_HiddenCount]) -> None:
        """A plan of the hidden valid-row count serves calls that ran the prologue, the shipped plan the others."""
        if (hidden is not None) != self.plan.hidden_count:
            raise ValueError(
                "the remembered binding was planned for the other valid-row count path"
            )

    def _compact(self, t: dict[str, Any], row_index: Optional[torch.Tensor]) -> None:
        """Check the call's valid-row index against the remembered plan and bind it."""
        plan = self.plan
        if plan.compact != (row_index is not None) or (
            plan.compact and int(row_index.numel()) != plan.rows
        ):
            raise ValueError(
                "the remembered binding was planned for another set of valid rows"
            )
        if plan.compact:
            t["row_index"] = row_index

    def _launch(
        self,
        keys: tuple,
        t: dict[str, Any],
        dw_stream: str = "auto",
        before_dx: Optional[Callable[[], None]] = None,
    ) -> None:
        """Run ``keys`` with the call's tensors ``t``; ``before_dx`` as in :func:`_before_first_dx`."""
        plan, launches = self.plan, self.launches

        def run_key(key) -> None:
            stage = key[0]
            host = HOST_STAGES.get(stage)
            if host is not None:
                host(
                    _slot_values(stage, t, plan, key[1])
                    if stage == "gather_rows"
                    else stage_values(stage, t, plan, key[1])
                )
                return
            launch = launches[key]
            arguments = launch.arguments_for(_slot_values(stage, t, plan, key[1]))
            if (
                launch.prepare is not None
            ):  # descriptors of a pointer-ABI stage see the fresh tensors
                launch.prepare(*arguments)
            launch.entry(*arguments)

        run, finish = _before_first_dx(run_key, before_dx)
        _run_keys(self.plan, self.device_index, keys, run, dw_stream)
        finish()

    def forward(
        self,
        X,
        W,
        labels,
        infer_logp,
        loss_weights,
        row_index=None,
        hidden=None,
        dw_stream: str = "auto",
    ):
        """One forward through the remembered launches; ``dw_stream`` is the call's dW side-stream mode
        (:func:`dw_stream_mode`), a launch placement outside the binding.

        Host-count path (``hidden`` None, :func:`forward_early_keys`): chunk 0's row gather and logits GEMM are bound
        and launched first, right behind the valid-row count, and the rest of the call -- the policy operands'
        gathers, the row vectors, the accumulators and the remaining launches -- is bound behind the queued GEMM
        instead of in front of it; the compacted labels are bound in front of the first early launch that reads them
        (:attr:`forward_early_labels`, the label-column arm) and behind the early launches otherwise.  The same
        kernels with the same operands on the same stream; the early launches neither read nor write what moved
        behind them, so every output is bitwise the in-order sequence's."""
        p, plan = self.plan.problem, self.plan
        T, H, V = (
            plan.rows,
            p.hidden,
            p.vocab,
        )  # rows of the chunk loop (the valid rows when compacted)
        self._check_hidden(hidden)
        early = hidden is None and bool(self.forward_early or self.forward_early_labels)
        t: dict[str, Any] = dict(X=_prepare_x(X, p, plan.compact), W=W)
        if early:
            self._compact(t, row_index)
            self._chunk_buffers(t)
            if self.forward_early:
                self._launch(self.forward_early, t, dw_stream)
            t["labels"] = _prepare_labels(labels, plan.geometry, row_index)
            if self.forward_early_labels:
                self._launch(self.forward_early_labels, t, dw_stream)
            self._scratch(t, chunk_buffers=False)
        else:
            t["labels"] = _prepare_labels(labels, plan.geometry, row_index)
            self._compact(t, row_index)
            self._scratch(t, hidden)
        t["lse"] = torch.empty((T,), dtype=torch.float32, device=self.device)
        t["logp"] = torch.empty((T,), dtype=torch.float32, device=self.device)
        if p.entry == "loss":
            if infer_logp is not None:
                if plan.compact:
                    t["infer_logp_full"], t["loss_weights_full"] = (
                        infer_logp,
                        loss_weights,
                    )
                    t["infer_logp"] = torch.empty(
                        (T,), dtype=torch.float32, device=self.device
                    )
                    t["loss_weights"] = torch.empty(
                        (T,), dtype=torch.float32, device=self.device
                    )
                else:
                    t["infer_logp"], t["loss_weights"] = infer_logp, loss_weights
            if plan.need_dx:
                t["dx_acc"] = torch.empty(
                    (T, H), dtype=torch.float32, device=self.device
                )
            if plan.dw_acc_needed:
                t["dw_acc"] = torch.empty(
                    (V, H), dtype=torch.float32, device=self.device
                )
        self._launch(self.forward_late if early else self.forward_order, t, dw_stream)
        if p.entry == "loss":
            return (
                t["loss"],
                t["logp"],
                t.get("dx_acc"),
                t.get("dw_acc"),
                _deferred_operands(plan, t),
            )
        return t["logp"], t["lse"]

    def backward_logprob(
        self,
        X,
        W,
        labels,
        lse,
        dlogp,
        row_index=None,
        hidden=None,
        dw_stream: str = "auto",
        dcnt: Optional[torch.Tensor] = None,
    ):
        p, plan = self.plan.problem, self.plan
        T, H, V = plan.rows, p.hidden, p.vocab
        self._check_hidden(hidden)
        t: dict[str, Any] = dict(
            X=_prepare_x(X, p, plan.compact),
            W=W,
            labels=_prepare_labels(labels, plan.geometry, row_index),
            lse=lse,
        )
        self._compact(t, row_index)
        if plan.compact:
            t["d_in_full"] = dlogp
            t["d_in"] = torch.empty((T,), dtype=torch.float32, device=self.device)
        else:
            t["d_in"] = dlogp
        if plan.dcnt:
            if dcnt is None:
                raise ValueError(
                    "the remembered binding runs the zero-gradient row skip: pass its counts"
                )
            t["dcnt"] = dcnt
        self._scratch(t, hidden, backward=True)

        def gradient_buffers() -> None:
            if plan.need_dx:
                t["dx_acc"] = torch.empty(
                    (T, H), dtype=torch.float32, device=self.device
                )
                if plan.dx_cast:
                    t["dx_out"] = torch.empty(
                        (T, H), dtype=torch.bfloat16, device=self.device
                    )
            if plan.need_dw:
                if plan.dw_acc_needed:
                    t["dw_acc"] = torch.empty(
                        (V, H), dtype=torch.float32, device=self.device
                    )
                t["dw_out"] = torch.empty(
                    (V, H), dtype=torch.bfloat16, device=self.device
                )

        # The gradient accumulators and outputs are made right before the first dX GEMM key, behind the row gathers and
        # the recompute launch, which read none of them: a call of a few rows is host-bound up to its first GEMM, and
        # those launches are no longer held back by this host work.  The same buffers and stream order ahead of every
        # launch that reads or writes them.  Without dX (the first gradient key
        # is a dW GEMM, possibly on the dW side stream) or when a key ahead of the first dX GEMM reads them
        # (:attr:`backward_gradients_at_dx`), they are made up front.
        if plan.need_dx and self.backward_gradients_at_dx:
            self._launch(self.backward_order, t, dw_stream, before_dx=gradient_buffers)
        else:
            gradient_buffers()
            self._launch(self.backward_order, t, dw_stream)
        dx = (
            t.get("dx_out") if plan.dx_cast else t.get("dx_acc")
        )  # dx_cast=False: the caller finalizes the accumulator
        return dx, t.get("dw_out")


class _TemperatureBinding(_Binding):
    """A remembered binding of a call with a temperature (:attr:`Problem.temperature`, the temperature forms).  Each
    call passes its compact FP32 temperature rows (``temperature=``, the rows of the chunk loop like ``lse``); they are
    bound before the first launch (:meth:`_compact`, which every sequence runs first) and every launch slots them
    (:func:`_slot_values_temperature`).  A call without them is refused; a binding without the temperature forms takes
    no ``temperature`` (:class:`_Binding`)."""

    @classmethod
    def from_runner(
        cls, runner: LmHeadLossRunner, hidden: Optional[_HiddenCount] = None
    ) -> "_Binding":
        """``hidden`` (the eager entry points' hidden valid-row count): the runner was prepared on the counted rows
        after chunk 0's gather and logits GEMM ran in device-count form; the remembered plan records that and its
        launch sequences start at chunk 0's row kernels (the runner's chunk-0 gather / GEMM launches are dropped)."""
        if runner.backend != "cake":
            raise ValueError("only cake-backend runners are remembered")
        t = runner.tensors
        owned = {name: torch.empty_like(t[name]) for name in _OWNED_VALUES if name in t}
        plan, memory = runner.plan, runner.memory
        forward_order, backward_order = runner.forward_order, runner.backward_order
        if hidden is not None:
            p = plan.problem
            plan = replace(
                plan,
                hidden_count=True,
                hidden_logits_group_m=hidden.group_m,
                hidden_stats=hidden.stats,
            )
            forward_order = _forward_keys_temperature(plan)
            backward_order = (
                _recompute_keys_temperature(plan) if p.entry == "logprob" else ()
            ) + cast_keys(plan)
            memory = _hidden_count_memory_temperature(
                memory,
                num_rows=p.num_rows,
                hidden=p.hidden,
                vocab=p.vocab,
                chunk=p.chunk,
                stats_tile=plan.geometry.stats_tile,
                temperature=p.temperature,
            )
        # the hidden-count path drops the runner's chunk-0 gather / GEMM launches (not in either sequence)
        keep = None if hidden is None else set(forward_order) | set(backward_order)
        # the per-call slots come from :func:`_slot_values`: check it against :func:`stage_values` on the runner's
        # own tensors for every launch and host gather of this binding (fail closed)
        tau = plan.problem.temperature
        slot_values = _slot_values_temperature if tau else _slot_values
        full_values = stage_values_temperature if tau else stage_values
        for key, launch in runner.launches.items():
            if keep is None or key in keep:
                _check_slot_values_temperature(
                    key,
                    t,
                    plan,
                    [name for _, name in launch.slots],
                    slot_values,
                    full_values,
                )
        for key in forward_order + backward_order:
            if key[0] == "gather_rows":
                _check_slot_values_temperature(
                    key, t, plan, ["src", "idx", "out"], slot_values, full_values
                )
        early = forward_early_keys(plan)
        split = next(
            (i for i, key in enumerate(early) if _reads_labels_temperature(key[0])),
            len(early),
        )
        forward_late = tuple(key for key in forward_order if key not in early)
        return (cls if tau else _Binding)(
            plan=plan,
            device=t["X"].device,
            device_index=runner.device_index,
            launches={
                key: launch.templated()
                for key, launch in runner.launches.items()
                if keep is None or key in keep
            },
            forward_order=forward_order,
            backward_order=backward_order,
            forward_early=early[:split],
            forward_early_labels=early[split:],
            forward_late=forward_late,
            owned=owned,
            owned_bytes=sum(v.numel() * v.element_size() for v in owned.values()),
            layout=runner.layout,
            memory=memory,
            forward_scratch=_scratch_slots(
                slot_values,
                forward_order,
                t,
                plan,
                ("loss",) if plan.problem.entry == "loss" else (),
            ),
            backward_scratch=_scratch_slots(slot_values, backward_order, t, plan),
            backward_gradients_at_dx=_gradients_at_first_dx(
                slot_values, backward_order, t, plan
            ),
        )

    # the rows of the call in progress, per thread (set and cleared around the inherited sequence)
    _rows = threading.local()

    def forward(
        self,
        X,
        W,
        labels,
        infer_logp,
        loss_weights,
        row_index=None,
        hidden=None,
        dw_stream: str = "auto",
        temperature: Optional[torch.Tensor] = None,
    ):
        """:meth:`_Binding.forward` with the call's compact temperature rows."""
        return self._with_rows(
            temperature,
            super().forward,
            X,
            W,
            labels,
            infer_logp,
            loss_weights,
            row_index,
            hidden,
            dw_stream,
        )

    def backward_logprob(
        self,
        X,
        W,
        labels,
        lse,
        dlogp,
        row_index=None,
        hidden=None,
        dw_stream: str = "auto",
        dcnt: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
    ):
        """:meth:`_Binding.backward_logprob` with the call's compact temperature rows -- in the order of the call's
        ``row_index``, which the zero-gradient row skip reorders -- and, for a plan under the skip
        (:attr:`Plan.dcnt`: the counted dX / dW forms), the per-chunk counts ``dcnt``."""
        return self._with_rows(
            temperature,
            super().backward_logprob,
            X,
            W,
            labels,
            lse,
            dlogp,
            row_index,
            hidden,
            dw_stream,
            dcnt,
        )

    def _with_rows(self, temperature, call, *args):
        if temperature is None:
            raise ValueError(
                "the remembered binding was planned for a call with a temperature"
            )
        _check_output(temperature, "temperature", (self.plan.rows,), torch.float32)
        self._rows.value = temperature
        try:
            return call(*args)
        finally:
            self._rows.value = None

    def _compact(self, t: dict[str, Any], row_index: Optional[torch.Tensor]) -> None:
        super()._compact(t, row_index)
        t["temperature"] = self._rows.value

    def _launch(
        self,
        keys: tuple,
        t: dict[str, Any],
        dw_stream: str = "auto",
        before_dx: Optional[Callable[[], None]] = None,
    ) -> None:
        plan, launches = self.plan, self.launches

        def run_key(key) -> None:
            stage = key[0]
            host = HOST_STAGES.get(stage)
            if host is not None:
                host(
                    _slot_values_temperature(stage, t, plan, key[1])
                    if stage == "gather_rows"
                    else stage_values_temperature(stage, t, plan, key[1])
                )
                return
            launch = launches[key]
            arguments = launch.arguments_for(
                _slot_values_temperature(stage, t, plan, key[1])
            )
            if (
                launch.prepare is not None
            ):  # descriptors of a pointer-ABI stage see the fresh tensors
                launch.prepare(*arguments)
            launch.entry(*arguments)

        run, finish = _before_first_dx(run_key, before_dx)
        _run_keys_temperature(self.plan, self.device_index, keys, run, dw_stream)
        finish()


class _KeptBinding(_Binding):
    """A remembered binding of a kept-logits plan (:class:`_KeptPlan`): the templated launches of :class:`_Binding`,
    with every launch of chunk ``i`` bound to the chunk's kept view (one BF16 ``[rows_c, V]`` view per chunk,
    :func:`_kept_views`) where the default plan binds the chunk buffer.  Launched only by the kept-logits entry points
    (:meth:`forward_keep`, :meth:`backward_logprob_keep`)."""

    def _launch(
        self,
        keys: tuple,
        t: dict[str, Any],
        dw_stream: str = "auto",
        before_dx: Optional[Callable[[], None]] = None,
    ) -> None:
        kept = t.get("logits_keep")
        if kept is None:
            # chunk 0's early launches, ahead of the kept views: chunk 0's view is the chunk buffer's first rows
            super()._launch(keys, t, dw_stream, before_dx)
            return
        run, finish = _before_first_dx(self._kept_key_runner(t), before_dx)
        _run_keys(self.plan, self.device_index, keys, run, dw_stream)
        finish()

    def _kept_key_runner(self, t: dict[str, Any]) -> Callable[[Any], None]:
        """One launch key of a kept-logits sequence over ``t``: a chunk's key binds the chunk's kept view
        (``t["logits_keep"]``) where the default plan binds the chunk buffer."""
        plan, launches, kept = self.plan, self.launches, t["logits_keep"]

        def run_key(key) -> None:
            stage = key[0]
            if isinstance(key[1], int):
                # the chunk's kept view stands in for the chunk buffer
                t["logits"] = kept[key[1]]
            host = HOST_STAGES.get(stage)
            if host is not None:
                host(
                    _slot_values(stage, t, plan, key[1])
                    if stage == "gather_rows"
                    else stage_values(stage, t, plan, key[1])
                )
                return
            launch = launches[key]
            arguments = launch.arguments_for(_slot_values(stage, t, plan, key[1]))
            if launch.prepare is not None:
                launch.prepare(*arguments)
            launch.entry(*arguments)

        return run_key

    def _launch_dx_side(
        self,
        keys: tuple,
        t: dict[str, Any],
        after_dz: Optional[Callable[[], None]] = None,
    ) -> None:
        """Run a kept backward's launch keys with every chunk's dX keys on the per-device side stream
        (:func:`_dw_stream`): the caller's stream runs the ``row_grad`` keys back to back (the fork event recorded
        behind each; the side stream waits on it before the chunk's dX keys), then ``after_dz`` (on the caller's
        stream, behind the last chunk's ``row_grad``: every ``dz`` is formed -- the one-pass weight gradient,
        :func:`backward_logprob_keep_ordered`), then joins the side stream before the keys after the chunks (the flat
        dX cast) and before it returns.  The same kernels on the same operands as :meth:`_launch`: only the launch
        stream of the dX keys differs."""
        run_key = self._kept_key_runner(t)
        index = self.device_index
        main = torch.cuda.current_stream(index)
        side, fork = _dw_stream(index)
        chunk_pos = [
            pos
            for pos, (_, chunk) in enumerate(keys)
            if isinstance(chunk, int) and not isinstance(chunk, bool)
        ]
        grads = [pos for pos in chunk_pos if keys[pos][0] == "row_grad"]
        last_grad = grads[-1] if grads else None
        after = chunk_pos[-1] + 1 if chunk_pos else len(keys)
        forked = False
        with contextlib.ExitStack() as stack:
            stack.enter_context(_ffi_stream_context(index))
            for pos, key in enumerate(keys):
                if pos == after:
                    main.wait_stream(side)
                if pos in chunk_pos and key[0] != "row_grad":
                    stack.close()
                    if forked:
                        side.wait_event(fork)
                        forked = False
                    with torch.cuda.stream(side), _ffi_stream_context(index):
                        run_key(key)
                    stack.enter_context(_ffi_stream_context(index))
                    continue
                run_key(key)
                if key[0] == "row_grad":
                    fork.record(main)
                    forked = True
                    if pos == last_grad and after_dz is not None:
                        stack.close()
                        after_dz()
                        stack.enter_context(_ffi_stream_context(index))
        main.wait_stream(side)

    def forward_keep(
        self, X, W, labels, row_index=None, hidden=None, dw_stream: str = "auto"
    ):
        """:meth:`_Binding.forward` of the log-probability entry that keeps the logits: ``(logp, lse, kept)``, ``kept``
        one BF16 ``[rows_c, V]`` view per chunk.  The views are consecutive row slices of ONE contiguous buffer
        (:func:`kept_rows_of`): without the hidden valid-row count a ``[rows, V]`` buffer whose first rows are bound as
        chunk 0's chunk buffer; with it the prologue's ``[T, V]`` kept buffer (:attr:`_HiddenCount.kept`, its first rows
        the prologue's chunk buffer), or -- a prologue without one -- :func:`_kept_views` (chunk 0's in the prologue's
        chunk buffer)."""
        p, plan = self.plan.problem, self.plan
        T = plan.rows
        self._check_hidden(hidden)
        early = hidden is None and bool(self.forward_early or self.forward_early_labels)
        # the contiguous kept buffer (``2 * rows * V`` bytes, the bytes of the chunk buffer plus the later chunks'
        # buffer it replaces): every chunk's logits in row order, so a backward can take them as one chunk.  With the
        # hidden valid-row count it is the prologue's ``[T, V]`` buffer (chunk 0's logits already in its first rows;
        # sized by the call's ``T`` rows, of which the plan's ``T_v`` hold logits), or none.
        if hidden is not None:
            kept_rows = hidden.kept
        else:
            kept_rows = (
                torch.empty((T, p.vocab), dtype=torch.bfloat16, device=self.device)
                if T > 0
                else None
            )
        first = None if kept_rows is None else kept_rows[: plan.chunks[0][1]]
        t: dict[str, Any] = dict(X=_prepare_x(X, p, plan.compact), W=W)
        if early:
            self._compact(t, row_index)
            self._chunk_buffers(t, logits=first)
            if self.forward_early:
                self._launch(self.forward_early, t, dw_stream)
            t["labels"] = _prepare_labels(labels, plan.geometry, row_index)
            if self.forward_early_labels:
                self._launch(self.forward_early_labels, t, dw_stream)
            self._scratch(t, chunk_buffers=False)
        else:
            t["labels"] = _prepare_labels(labels, plan.geometry, row_index)
            self._compact(t, row_index)
            self._scratch(t, hidden, logits=first)
        t["lse"] = torch.empty((T,), dtype=torch.float32, device=self.device)
        t["logp"] = torch.empty((T,), dtype=torch.float32, device=self.device)
        kept = t["logits_keep"] = (
            tuple(kept_rows[row0 : row0 + rows_c] for row0, rows_c in plan.chunks)
            if kept_rows is not None
            else _kept_views(plan, t["logits"], self.device)
        )
        self._launch(self.forward_late if early else self.forward_order, t, dw_stream)
        return t["logp"], t["lse"], kept

    def backward_logprob_keep(
        self,
        X,
        W,
        labels,
        lse,
        dlogp,
        logits: tuple,
        row_index=None,
        dw_stream: str = "auto",
        dz_ready: bool = False,
        dx_side: bool = False,
        after_dz: Optional[Callable[[], None]] = None,
    ):
        """:meth:`_Binding.backward_logprob` over the forward's kept views ``logits`` (:func:`check_kept_logits`):
        ``row_grad`` overwrites each with the chunk's ``dz``; no chunk logits buffer is allocated.  ``dz_ready``: the
        views already hold ``dz`` (an earlier backward's ``row_grad`` formed it): no ``row_grad``.  ``dx_side``: the dX
        keys run on the side stream and ``after_dz`` behind the last ``row_grad`` (:meth:`_launch_dx_side`)."""
        p, plan = self.plan.problem, self.plan
        T, H, V = plan.rows, p.hidden, p.vocab
        self._check_hidden(None)
        t: dict[str, Any] = dict(
            X=_prepare_x(X, p, plan.compact),
            W=W,
            labels=_prepare_labels(labels, plan.geometry, row_index),
            lse=lse,
        )
        self._compact(t, row_index)
        if plan.compact:
            t["d_in_full"] = dlogp
            t["d_in"] = torch.empty((T,), dtype=torch.float32, device=self.device)
        else:
            t["d_in"] = dlogp
        rows = self.layout["logits"][1] // (V * 2)
        t["logits"] = logits[0]  # the chunk buffer's place: chunk 0's kept view
        # no kept-backward stage reads or writes the row statistics (row_grad takes z, labels, lse and d; the dX and
        # weight-gradient GEMMs bind STATS_OUT unused): one row stands in for the [rows, tiles, 2] buffer
        t["stats"] = torch.empty(
            (1, plan.num_tiles, 2), dtype=torch.float32, device=self.device
        )
        if plan.compact:
            t["x_c"] = torch.empty((rows, H), dtype=torch.bfloat16, device=self.device)
        self._scratch(t, chunk_buffers=False, backward=True)
        t["logits_keep"] = logits
        if plan.need_dx:
            t["dx_acc"] = torch.empty((T, H), dtype=torch.float32, device=self.device)
            if plan.dx_cast:
                t["dx_out"] = torch.empty(
                    (T, H), dtype=torch.bfloat16, device=self.device
                )
        if plan.need_dw:
            if plan.dw_acc_needed:
                t["dw_acc"] = torch.empty(
                    (V, H), dtype=torch.float32, device=self.device
                )
            t["dw_out"] = torch.empty((V, H), dtype=torch.bfloat16, device=self.device)
        keys = self.backward_order
        if dz_ready:
            keys = tuple(
                key
                for key in keys
                if key[0] != "row_grad" and key != ("gather_rows", "d_in")
            )
        if dx_side and plan.need_dx and not plan.need_dw and not dz_ready:
            self._launch_dx_side(keys, t, after_dz)
        else:
            if after_dz is not None:
                raise ValueError(
                    "after_dz runs with dx_side only (a dX-only kept backward)"
                )
            self._launch(keys, t, dw_stream)
        dx = t.get("dx_out") if plan.dx_cast else t.get("dx_acc")
        return dx, t.get("dw_out")


def _check_slot_values(key, t: dict[str, Any], plan: Plan, names) -> None:
    """:func:`_slot_values` binds what :func:`stage_values` binds for the host values ``names`` of launch ``key``."""
    fast = _slot_values(key[0], t, plan, key[1])
    full = stage_values(key[0], t, plan, key[1])
    for name in names:
        if not _same_view(fast.get(name), full.get(name)):
            raise RuntimeError(
                f"launch {key!r}: the per-call host value {name!r} differs from the stage's host values"
            )


def _check_slot_values_temperature(
    key,
    t: dict[str, Any],
    plan: Plan,
    names,
    slot_values=_slot_values,
    full_values=stage_values,
) -> None:
    """:func:`_slot_values` (or the binding's ``slot_values``) binds what :func:`stage_values` (or the binding's
    ``full_values``) binds for the host values ``names`` of launch ``key``."""
    fast = slot_values(key[0], t, plan, key[1])
    full = full_values(key[0], t, plan, key[1])
    for name in names:
        if not _same_view(fast.get(name), full.get(name)):
            raise RuntimeError(
                f"launch {key!r}: the per-call host value {name!r} differs from the stage's host values"
            )


def _deferred_operands(plan: Plan, t: dict[str, Any]) -> tuple:
    """``(dz_last, x_last, x_src, x_idx)`` of a deferred plan after its forward: the last chunk's BF16 ``dz`` rows of the
    chunk buffer and its ``X`` rows -- a view of the launched ``X`` (uncompacted) or the ``X`` plus the chunk's int64
    row index, gathered again in the backward (compacted) -- so no ``[rows, H]`` copy outlives the forward.  The ``dz``
    rows are a view too: the whole ``[min(T, C), V]`` chunk buffer stays alive until the backward (what
    :func:`memory_report` reports as ``saved_dz_bytes``) rather than paying a tail copy per step; the eager entry points
    launch through the templated binding from the first call on, so that buffer is a per-call allocation of its own and
    the view pins nothing else (a prepared runner keeps the chunk's ``dz`` inside its workspace).  All ``None`` for an
    undeferred plan."""
    if not plan.dw_deferred:
        return None, None, None, None
    row0, rows_c = plan.last_chunk
    dz_last = t["logits"][:rows_c]
    if plan.compact:
        return dz_last, None, t["X"], t["row_index"][row0 : row0 + rows_c]
    return dz_last, t["X"][row0 : row0 + rows_c], None, None


class BindingCache:
    """Remembered input bindings of the eager entry points (least recently used, bounded).

    ``capacity`` (default :data:`BINDING_CACHE_DEFAULT_CAPACITY`, 64 bindings) bounds what is kept; ``enabled=False``
    bypasses the cache (every call validates and binds).  :meth:`configure` changes both at runtime.  The templated
    launches of the eager output-boundary casts (:func:`_eager_launch`) are kept alongside, at most
    :attr:`EAGER_CAPACITY` forms, and follow the same switch and :meth:`clear`.

    Threads share the cache without a lock: each ordered-dict operation is atomic (under CPython's global interpreter
    lock; a free-threaded build gives no such guarantee), and a step that finds its key or entry gone -- another
    thread's eviction between a lookup's ``get`` and its recency update, or between an eviction's size check and its
    ``popitem`` -- skips that step (a looked-up binding stays valid for its call; an eviction that finds the cache
    already within its capacity stops).  The hit / miss counters are statistics."""

    # templated launches of the eager output-boundary stages (:func:`_eager_launch`) kept alongside the bindings
    EAGER_CAPACITY = 64

    def __init__(self, capacity: Optional[int] = None, enabled: bool = True):
        self.capacity = BINDING_CACHE_DEFAULT_CAPACITY
        self.enabled = True
        self.hits = 0
        self.misses = 0
        self._bindings: OrderedDict[tuple, _Binding] = OrderedDict()
        self._eager: OrderedDict[tuple, tuple] = OrderedDict()
        self.configure(enabled=enabled, capacity=capacity)

    def configure(
        self, *, enabled: Optional[bool] = None, capacity: Optional[int] = None
    ) -> "BindingCache":
        """Apply ``capacity`` (a positive integer; the least recently used bindings above it are evicted) and / or
        ``enabled`` (``False`` bypasses the cache and drops the remembered bindings); ``None`` leaves a setting as it
        is.  Returns the cache."""
        if capacity is not None:
            if (
                isinstance(capacity, bool)
                or not isinstance(capacity, int)
                or capacity < 1
            ):
                raise ValueError(
                    "BindingCache needs a capacity of at least one binding (a positive integer; use enabled=False "
                    f"to bypass it), got {capacity!r}"
                )
            self.capacity = capacity
            _evict(self._bindings, self.capacity)
        if enabled is not None:
            self.enabled = bool(enabled)
            if not self.enabled:
                self.clear()
        return self

    def __len__(self) -> int:
        return len(self._bindings)

    @property
    def owned_bytes(self) -> int:
        return sum(b.owned_bytes for b in _values(self._bindings))

    def clear(self) -> None:
        self._bindings.clear()
        self._eager.clear()

    def eager_lookup(self, form: tuple, record) -> Optional[_Launch]:
        """The templated eager launch remembered for ``form`` and this very ``record`` (:func:`_eager_launch`)."""
        entry = self._eager.get(form)
        if entry is None or entry[0] is not record:
            return None
        # another thread's eviction since the get: the launch stays valid for this call (one line: a ``try`` line of
        # its own would add a NOP to every hit)
        # fmt: off
        try: self._eager.move_to_end(form)  # noqa: E701, SIM105
        except KeyError: pass  # noqa: E701
        # fmt: on
        return entry[1]

    def eager_remember(self, form: tuple, record, launch: _Launch) -> None:
        self._eager.pop(form, None)
        self._eager[form] = (record, launch)
        if len(self._eager) > self.EAGER_CAPACITY:
            _evict(self._eager, self.EAGER_CAPACITY)

    def lookup(self, key: tuple) -> Optional[_Binding]:
        binding = self._bindings.get(key)
        if binding is None:
            self.misses += 1
        else:
            self.hits += 1
            # another thread's eviction since the get: the binding stays valid for this call (one line, as in
            # :meth:`eager_lookup`)
            # fmt: off
            try: self._bindings.move_to_end(key)  # noqa: E701, SIM105
            except KeyError: pass  # noqa: E701
            # fmt: on
        return binding

    def peek(self, key: tuple) -> Optional[_Binding]:
        return self._bindings.get(key)

    def remember(self, key: tuple, binding: _Binding) -> _Binding:
        self._bindings.pop(key, None)
        self._bindings[key] = binding
        if len(self._bindings) > self.capacity:
            _evict(self._bindings, self.capacity)
        return binding


def _evict(entries: OrderedDict, capacity: int) -> None:
    """Drop the least recently used entries of ``entries`` above ``capacity``; stop when another thread's eviction
    emptied it first."""
    while len(entries) > capacity:
        try:
            entries.popitem(last=False)
        except KeyError:
            return


def _values(entries: OrderedDict) -> list:
    """The values of ``entries`` as a list, copied again when another thread changed it during the copy."""
    while True:
        try:
            return list(entries.values())
        except RuntimeError:  # OrderedDict mutated during iteration
            continue


# Enabled, 64 bindings.  ``flashinfer.cake_lm_head_nondet_train`` applies its environment variables through
# :meth:`BindingCache.configure` on its first backend use; this module reads none.
BINDING_CACHE = BindingCache()


# ---------------------------------------------------------------------------
# Eager entry points (allocate, launch, return)
# ---------------------------------------------------------------------------


class _TemperatureResultState:
    """The temperature state of a :class:`ForwardResult` (log-probability entry with a temperature):
    ``temperature_rows`` the compact FP32 ``[rows]`` temperature the forward's kernels read (the rows of
    ``row_index`` when compacted, like ``lse``); :func:`backward_logprob` takes it back (no second gather).  A plain
    base class: a typed class default (``None``), not a dataclass field, so building the result of a call without a
    temperature costs what it did before; a forward with a temperature sets it on its result."""

    temperature_rows: Optional[torch.Tensor] = None


class _ProblemResultState:
    """The validated :class:`Problem` of a log-probability forward (``problem``, without the temperature form): its
    autograd backward takes it back (:func:`backward_logprob`'s ``_problem``) instead of validating the same ``X``,
    ``W`` and ``labels`` again.  A plain base class like :class:`_TemperatureResultState`: a typed class default
    (``None``), not a dataclass field."""

    problem: Optional[Problem] = None


@dataclass
class ForwardResult(_TemperatureResultState, _ProblemResultState):
    loss: Optional[torch.Tensor]  # FP32 [] (loss entry)
    logp: (
        torch.Tensor
    )  # FP32 [T] (0 on ignored rows; scattered back when the rows were compacted)
    lse: Optional[torch.Tensor] = (
        None  # FP32 (log-probability entry: the saved statistic; compact [T_v] rows of ``row_index`` when compacted)
    )
    dx_acc: Optional[torch.Tensor] = (
        None  # FP32 [T, H] (loss entry, X trainable); compact [T_v, H] rows of ``row_index`` when compacted
    )
    dw_acc: Optional[torch.Tensor] = None  # FP32 [V, H] (loss entry, W trainable)
    memory: Optional[dict] = None
    backend: str = "cake"
    row_index: Optional[torch.Tensor] = (
        None  # int64 [T_v] valid-row index when the chunk loop was compacted (None otherwise)
    )
    # the bool [T] valid-row mask when the compacted forward ran with the fused dX finalize (None otherwise): the
    # backward follows the forward's choice through it and forms the inclusive scan the scatter kernel reads
    row_valid: Optional[torch.Tensor] = None
    num_rows: int = 0  # T, the caller's row count
    # Deferred weight gradient (``fuse_dw_cast``, loss entry): the last chunk's BF16 ``dz`` rows (a view of the chunk
    # buffer) and its ``X`` rows -- ``x_last`` (a view of ``X``) on the uncompacted path, ``x_src`` / ``x_idx`` (``X`` and
    # the chunk's int64 row index, gathered in the backward) when compacted.  ``backward_loss`` finishes ``dW`` from them;
    # ``dw_acc`` then holds the chunks before the last one and is ``None`` for a one-chunk call.
    dz_last: Optional[torch.Tensor] = None
    x_last: Optional[torch.Tensor] = None
    x_src: Optional[torch.Tensor] = None
    x_idx: Optional[torch.Tensor] = None
    # the explicit GEMM knobs the forward ran with (``{}`` = none); the deferred weight gradient's GEMM takes its knobs
    gemm_tuning: Optional[dict] = None
    # a one-pass call's kept state (:func:`_one_pass_forward`): ``(Z, ROWS, LAB, need_dx, need_dw)``, the operands of
    # :func:`_one_pass_backward` (``None`` for the chunked path)
    one_pass: Optional[tuple] = None

    def backward(self, grad=None, *, grad_weight_dtype=torch.bfloat16, backend=None):
        """``(dX, dW)`` of this forward through :func:`backward_loss` (``grad`` = the upstream scalar, ``None`` = 1)."""
        if self.loss is None:
            raise ValueError(
                "ForwardResult.backward belongs to the loss entry (use backward_logprob)"
            )
        return backward_loss(
            self.dx_acc,
            self.dw_acc,
            grad,
            grad_weight_dtype=grad_weight_dtype,
            backend=self.backend if backend is None else backend,
            row_index=self.row_index,
            num_rows=self.num_rows,
            dz_last=self.dz_last,
            x_last=self.x_last,
            x_src=self.x_src,
            x_idx=self.x_idx,
            row_valid=self.row_valid,
            gemm_tuning=self.gemm_tuning,
        )


@dataclass
class KeptForwardResult(ForwardResult):
    """:class:`ForwardResult` of :func:`forward_logprob_keep`: ``logits`` holds the kept BF16 logits, one
    ``[rows_c, V]`` view per chunk, for :func:`backward_logprob_keep` (which overwrites them with ``dz``); ``None``
    when nothing was launched (no row, or every row ignored)."""

    logits: Optional[tuple] = None


def _kept_result(result: ForwardResult, logits: Optional[tuple]) -> KeptForwardResult:
    return KeptForwardResult(
        **{f.name: getattr(result, f.name) for f in fields(result)}, logits=logits
    )


def _zero_accumulator(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """An FP32 zero accumulator of ``shape`` without its storage: a stride-0 view of one zero element.  A call that
    launches no gradient GEMM (:func:`_empty_forward`) returns it as ``dw_acc``, so no FP32 ``[V, H]`` buffer lives
    from the forward to the backward."""
    return torch.zeros((), dtype=torch.float32, device=device).expand(shape)


def _is_zero_accumulator(acc: torch.Tensor) -> bool:
    """``acc`` is a :func:`_zero_accumulator` (every stride 0, at least one element)."""
    return acc.numel() > 0 and all(s == 0 for s in acc.stride())


def _zero_gradient(
    acc: torch.Tensor, grad: Optional[torch.Tensor], out_dtype: torch.dtype
) -> torch.Tensor:
    """``cast(g * 0)`` of a :func:`_zero_accumulator` in a new ``out_dtype`` tensor of its shape, without reading or
    materializing the FP32 accumulator: the bits :func:`scale_cast` gives for a zero FP32 accumulator (``+0``, ``-0``
    for a negative ``g``, NaN for a non-finite one).  The product is formed in FP32 and cast once, as there, so a
    finite ``g`` beyond the BF16 range still gives a signed zero.

    The cast value is written through a 16-byte (complex128) view of the output, ``16 // itemsize`` copies of its
    bits per element, when the element count allows: the copy from a 0-dim tensor stores one element at a time, so an
    eighth as many BF16 stores (a quarter as many FP32 ones) write the same bits in less time, without reading ``g``
    on the host.  The same-dtype copy moves the bits without arithmetic."""
    if out_dtype not in GRAD_WEIGHT_DTYPES:
        raise ValueError("the cast produces bfloat16 or float32")
    zero = torch.zeros((), dtype=torch.float32, device=acc.device)
    if grad is not None:
        zero = zero * grad.detach().reshape(()).to(
            device=acc.device, dtype=torch.float32
        )
    out = torch.empty(acc.shape, dtype=out_dtype, device=acc.device)
    value = zero.to(out_dtype)
    copies = 16 // out.element_size()
    if out.numel() % copies:
        return out.copy_(value)
    bits = value.view(torch.int16 if out.element_size() == 2 else torch.int32)
    word = bits.reshape(1).expand(copies).contiguous().view(torch.complex128)
    out.view(-1).view(torch.complex128).copy_(word.reshape(()))
    return out


def _empty_forward(
    problem: Problem,
    X: torch.Tensor,
    *,
    need_dx: bool,
    need_dw: bool,
    row_index: Optional[torch.Tensor] = None,
    row_valid: Optional[torch.Tensor] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    backend: str = "cake",
) -> ForwardResult:
    """Zeros without binding or launching: ``T == 0``, or every row ignored (``row_index`` empty).

    ``fuse_dw_cast`` / ``dx_finalize`` are the call's switches (``None`` = the documented default), so the memory
    report records the form the caller asked for, as a launched call's would.  ``backend``: the call's, kept on the
    result for :meth:`ForwardResult.backward`.  The loss entry's ``dw_acc`` is a zero accumulator without ``[V, H]``
    storage (:func:`_zero_accumulator`): the backward forms the zero weight gradient in the output dtype, and the
    memory report counts the accumulator's one FP32 element."""
    device = X.device
    T = int(problem.num_rows)
    memory = memory_report(
        T,
        problem.hidden,
        problem.vocab,
        problem.chunk,
        entry=problem.entry,
        need_dx=need_dx,
        need_dw=need_dw,
        grad_weight_dtype=problem.grad_weight_dtype,
        return_logp=True,
        valid_rows=None if row_index is None else 0,
        fuse_dw_cast=fuse_dw_cast,
        dx_finalize=dx_finalize,
    )
    if problem.entry == "loss" and "dW_acc" in memory["accumulators"]:
        memory["accumulators"]["dW_acc"] = 4  # the zero accumulator's one element
        memory["accumulator_bytes"] = sum(memory["accumulators"].values())
    zeros = torch.zeros((T,), dtype=torch.float32, device=device)
    if problem.entry == "loss":
        return ForwardResult(
            loss=torch.zeros((), dtype=torch.float32, device=device),
            logp=zeros,
            dx_acc=torch.zeros((0, problem.hidden), dtype=torch.float32, device=device)
            if need_dx
            else None,
            dw_acc=_zero_accumulator((problem.vocab, problem.hidden), device)
            if need_dw
            else None,
            memory=memory,
            backend=backend,
            row_index=row_index,
            row_valid=row_valid,
            num_rows=T,
        )
    return ForwardResult(
        loss=None,
        logp=zeros,
        lse=zeros.clone(),
        memory=memory,
        row_index=row_index,
        num_rows=T,
    )


def _resolve_compact(compact_rows) -> bool:
    return compact_rows_default() if compact_rows is None else bool(compact_rows)


def _resolve_hidden_count(hidden_count) -> bool:
    return hidden_count_default() if hidden_count is None else bool(hidden_count)


def _hidden_count(
    problem: Problem,
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    backend: str,
    compact: bool,
    stats: bool,
    hidden_count: bool,
    loss_weights: Optional[torch.Tensor] = None,
    kept_rows: bool = False,
) -> Optional[_HiddenCount]:
    """The hidden valid-row count prologue of a compacted ``cake`` call when the switch (``hidden_count``, resolved by
    the caller) is on and the call is eligible (:func:`hidden_count_eligible`); ``None`` = the shipped
    :func:`valid_rows` path.  ``loss_weights``: the zero-weight split (:func:`gradient_index`).  ``kept_rows`` (a
    kept-logits forward): chunk 0's logits buffer is the first rows of one contiguous ``[T, V]`` kept buffer
    (:attr:`_HiddenCount.kept`), sized by ``T`` because the valid-row count is not known yet."""
    if not (compact and backend == "cake" and hidden_count):
        return None
    rows, chunk = int(problem.num_rows), int(problem.chunk)
    if rows <= 0 or (rows + chunk - 1) // chunk < HIDDEN_COUNT_MIN_CHUNKS:
        return None  # the chunk rule of :func:`hidden_count_eligible`, ahead of the record lookup
    module_name, record = record_for(X.device)
    if not hidden_count_eligible(problem, X, record, module_name, stats=stats):
        return None
    return _hidden_count_begin(
        problem,
        X,
        W,
        labels,
        record=record,
        module_name=module_name,
        stats=stats,
        loss_weights=loss_weights,
        kept_rows=kept_rows,
    )


def _hidden_count_temperature(
    problem: Problem,
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    backend: str,
    compact: bool,
    hidden_count: bool,
    temperature: TemperatureForm,
) -> Optional[_HiddenCount]:
    """:func:`_hidden_count` of a log-probability forward with a temperature (``temperature``: the call's resolved form,
    :func:`_resolve_temperature`; the problem records that it has one): the device-count GEMM with the statistics in
    its temperature form (:func:`hidden_count_eligible_temperature`)."""
    if temperature is None or not problem.temperature:
        raise ValueError("the temperature form and the problem disagree")
    if not (compact and backend == "cake" and hidden_count):
        return None
    rows, chunk = int(problem.num_rows), int(problem.chunk)
    if rows <= 0 or (rows + chunk - 1) // chunk < HIDDEN_COUNT_MIN_CHUNKS:
        return None  # the chunk rule of :func:`hidden_count_eligible`, ahead of the record lookup
    module_name, record = record_for(X.device)
    if not hidden_count_eligible_temperature(
        problem, X, record, module_name, stats=True
    ):
        return None
    return _hidden_count_begin_temperature(
        problem,
        X,
        W,
        labels,
        record=record,
        module_name=module_name,
        stats=True,
        temperature=temperature,
    )


def forward_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div=None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    need_dx: bool = True,
    need_dw: bool = True,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    dw_stream: Optional[Union[bool, str]] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    return_logp: bool = True,
) -> ForwardResult:
    """Forward of the loss entry: ``loss`` (FP32 scalar), ``logp`` (FP32 ``[T]``) and the
    FP32 gradient accumulators ``dx_acc`` / ``dw_acc`` of the trainable inputs.

    The first call for an input binding validates and binds through
    :func:`prepare_lm_head_loss` and launches through the templated binding it
    remembers (:data:`BINDING_CACHE`); later calls with the same binding take
    the remembered launches directly.  A call without rows returns
    loss 0, an empty ``logp`` and zero accumulators without binding or launching.
    ``compact_rows`` (default :func:`compact_rows_default`) chunks over the valid
    rows only: ``logp`` comes back scattered to ``[T]``, ``dx_acc`` is the compact
    ``[T_v, H]`` (``row_index`` / ``num_rows`` of the result restore ``[T, H]``
    through :func:`backward_loss`); every row ignored returns zeros without
    binding or launching.  ``fuse_dw_cast`` (default :func:`fuse_dw_cast_default`):
    the last chunk's weight-gradient GEMM is deferred to :func:`backward_loss`
    (``dw_acc`` holds the chunks before it, ``None`` for a one-chunk call; the
    result carries that chunk's operands ``dz_last`` / ``x_last`` / ``x_src`` /
    ``x_idx``).  ``dx_finalize`` (default :func:`dx_finalize_default`, resolved
    once here) keeps a compacted call's valid-row mask (``row_valid``) for
    :func:`backward_loss`'s one-pass scatter of ``dX``.  The dX GEMM of a chunk
    of more than one K slice stores its slices into the call's slab and its slab
    sum stores their sum, in slice order, into ``dx_acc`` (:data:`DX_RED_STAGE`,
    :data:`DX_SLAB_SUM_STAGE`).  ``dw_stream`` (default
    ``None`` = ``"auto"``, :func:`dw_stream_mode`) places each chunk's
    weight-gradient accumulate on the per-device side stream (``True`` every
    multi-chunk call, ``False`` never, ``"auto"`` calls of
    :data:`DW_STREAM_MIN_CHUNKS` chunks or more).  ``hidden_count`` (default
    :func:`hidden_count_default`) runs a compacted call's valid-row count on the
    device.  ``gemm_tuning`` (default :func:`gemm_tuning_default`) pins GEMM
    knobs explicitly; the result carries it for the deferred weight gradient.

    Zero-weight rows (a compacted policy call, :func:`zero_weight_split`): the chunk loop runs over the
    gradient rows ``(labels >= 0) & (loss_weights != 0)`` only (``row_index``, ``dx_acc`` and the deferred operands
    are theirs); the zero-weight rows contribute exact zeros to ``loss``, ``dX`` and ``dW``.  ``return_logp`` (default
    ``True``): their ``logp`` comes from a logp-only pass of the log-probability forward (:func:`_zero_weight_logp`);
    ``False`` skips that pass and leaves their ``logp`` at 0 (the loss autograd entry without ``return_logp``).  Both
    row counts come back in the call's one count readback.  A call whose valid rows all have zero weight launches no
    gradient GEMM and returns loss 0 and zero accumulators, as a call without valid rows.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = validate_lm_head_inputs(
        X,
        W,
        labels,
        objective=objective,
        loss_div=loss_div,
        infer_logp=infer_logp,
        loss_weights=loss_weights,
        chunk_size=chunk_size,
        grad_weight_dtype=grad_weight_dtype,
        entry="loss",
    )
    stream_mode = dw_stream_mode(dw_stream)
    tuning = _resolve_gemm_tuning(gemm_tuning)
    # read once: the binding key and the plan take the same settings
    rule = dx_slice_rule()
    T = problem.num_rows
    if T == 0:
        return _empty_forward(
            problem,
            X,
            need_dx=need_dx,
            need_dw=need_dw,
            fuse_dw_cast=fuse_dw_cast,
            dx_finalize=dx_finalize,
            backend=backend,
        )
    fused_dx = bool(
        need_dx and _resolve_dx_finalize(dx_finalize)
    )  # resolved once per call; the backward follows it through row_valid
    compact = _resolve_compact(compact_rows)
    # tensors on another device: the plain path raises the placement error
    split = (
        zero_weight_split(problem, compact)
        and loss_weights is not None
        and loss_weights.device == labels.device == X.device
    )
    hc = _hidden_count(
        problem,
        X,
        W,
        labels,
        backend=backend,
        compact=compact,
        stats=True,
        hidden_count=_resolve_hidden_count(hidden_count),
        loss_weights=loss_weights if split else None,
    )
    zero_index = None
    if hc is not None:
        # the hidden valid-row count: chunk 0's gather and logits GEMM are queued; the count arrives through the pinned cell
        row_index = hc.row_index()
        row_valid = hc.valid if (row_index is not None and fused_dx) else None
        zero_index = hc.zero_index()
    elif split:
        row_index, row_valid, zero_index = gradient_rows(
            labels, loss_weights, mask=fused_dx
        )
    else:
        row_index, row_valid = (
            valid_rows(labels, mask=fused_dx) if compact else (None, None)
        )
    zero_rows = 0 if zero_index is None else int(zero_index.numel())
    fuse = bool(need_dw and _resolve_fuse(fuse_dw_cast))
    # The zero-weight rows' logp-only pass runs ahead of the chunk loop -- its buffers return to the allocator
    # before the loop allocates -- unless the hidden valid-row count already holds chunk 0's buffers and the loop leaves
    # none alive until the backward (no deferred last chunk, ``fuse`` off) or no chunk loop follows (every valid row has
    # zero weight: the count is known here): then after the loop / in place of it, with ``hc`` and those buffers
    # released, so the two passes' chunk buffers never coexist.  With ``fuse`` and a chunk loop the pass stays ahead of
    # the loop (after it, the deferred chunk's ``dz`` rows would coexist with it) and coexists with chunk 0's buffers
    # (``chunk0_alive``, counted in the memory report's peak).
    zero_pass = bool(zero_rows and return_logp)
    zero_first = zero_pass and (
        hc is None or (fuse and (row_index is None or row_index.numel() > 0))
    )
    chunk0_alive = zero_first and hc is not None
    zero_logp = (
        _zero_weight_logp(
            problem, X, W, labels, zero_index, backend=backend, gemm_tuning=tuning
        )
        if zero_first
        else None
    )
    if (
        row_index is not None and row_index.numel() == 0
    ):  # every row ignored (or of zero weight)
        hc = None
        if zero_pass and not zero_first:
            zero_logp = _zero_weight_logp(
                problem, X, W, labels, zero_index, backend=backend, gemm_tuning=tuning
            )
        result = _empty_forward(
            problem,
            X,
            need_dx=need_dx,
            need_dw=need_dw,
            row_index=row_index,
            row_valid=row_valid,
            fuse_dw_cast=fuse_dw_cast,
            dx_finalize=fused_dx,
            backend=backend,
        )
        return _with_zero_weight_rows(
            result, zero_index, zero_logp, chunk0_alive=chunk0_alive
        )
    valid_count = None if row_index is None else int(row_index.numel())
    common = dict(
        objective=objective,
        loss_div=problem.loss_div,
        infer_logp=infer_logp,
        loss_weights=loss_weights,
        chunk_size=chunk_size,
        need_dx=need_dx,
        need_dw=need_dw,
        grad_weight_dtype=grad_weight_dtype,
    )
    cache = BINDING_CACHE
    key = None
    if backend == "cake" and cache.enabled:
        key = forward_binding_key(
            X,
            W,
            labels,
            entry="loss",
            valid_rows=valid_count,
            fuse_dw_cast=fuse,
            gemm_tuning=tuning,
            dx_finalize=fused_dx,
            hidden_count=hc is not None,
            dx_rule=rule,
            **common,
        )
        binding = cache.lookup(key)
        if binding is not None:
            _alert_unordered_dx(binding.plan, "forward_loss", "loss")
            loss, logp, dx_acc, dw_acc, deferred = binding.forward(
                X,
                W,
                labels,
                infer_logp,
                loss_weights,
                row_index,
                hc,
                stream_mode,
            )
            hc = None
            if zero_pass and not zero_first:
                zero_logp = _zero_weight_logp(
                    problem,
                    X,
                    W,
                    labels,
                    zero_index,
                    backend=backend,
                    gemm_tuning=tuning,
                )
            dz_last, x_last, x_src, x_idx = deferred
            return _with_zero_weight_rows(
                ForwardResult(
                    loss=loss.reshape(()),
                    logp=scatter_rows(logp, row_index, T),
                    dx_acc=dx_acc,
                    dw_acc=dw_acc,
                    memory=binding.memory,
                    backend=backend,
                    row_index=row_index,
                    row_valid=row_valid,
                    num_rows=T,
                    dz_last=dz_last,
                    x_last=x_last,
                    x_src=x_src,
                    x_idx=x_idx,
                    gemm_tuning=tuning,
                ),
                zero_index,
                zero_logp,
                chunk0_alive=chunk0_alive,
            )
    runner = prepare_lm_head_loss(
        X,
        W,
        labels,
        entry="loss",
        backend=backend,
        compact_rows=False if row_index is None else row_index,
        fuse_dw_cast=fuse,
        gemm_tuning=tuning,
        dx_finalize=fused_dx,
        dw_stream=stream_mode,
        dx_rule=rule,
        **common,
    )
    _alert_unordered_dx(runner.plan, "forward_loss", "loss")
    if backend == "cake":
        # The first call launches through the templated binding too: its per-call temporaries come from the caching
        # allocator, so the deferred ``dz`` rows keep the chunk buffer alive and nothing else -- a view into the
        # runner's single workspace allocation would pin the whole workspace until the backward.  The runner (and its
        # workspace) is dropped here; only the binding survives.
        binding = _Binding.from_runner(runner, hc)
        del runner
        if key is not None:
            cache.remember(key, binding)
        loss, logp, dx_acc, dw_acc, deferred = binding.forward(
            X,
            W,
            labels,
            infer_logp,
            loss_weights,
            row_index,
            hc,
            stream_mode,
        )
        memory = binding.memory
    else:
        loss, logp = runner._forward()
        dx_acc, dw_acc = runner.dx_acc, runner.dw_acc
        deferred = _deferred_operands(runner.plan, runner.tensors)
        memory = runner.memory
        del runner
    hc = None
    if zero_pass and not zero_first:
        zero_logp = _zero_weight_logp(
            problem, X, W, labels, zero_index, backend=backend, gemm_tuning=tuning
        )
    dz_last, x_last, x_src, x_idx = deferred
    return _with_zero_weight_rows(
        ForwardResult(
            loss=loss.reshape(()),
            logp=scatter_rows(logp, row_index, T),
            dx_acc=dx_acc,
            dw_acc=dw_acc,
            memory=memory,
            backend=backend,
            row_index=row_index,
            row_valid=row_valid,
            num_rows=T,
            dz_last=dz_last,
            x_last=x_last,
            x_src=x_src,
            x_idx=x_idx,
            gemm_tuning=tuning,
        ),
        zero_index,
        zero_logp,
        chunk0_alive=chunk0_alive,
    )


def _zero_weight_logp(
    problem: Problem,
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    zero_index: torch.Tensor,
    *,
    backend: str,
    gemm_tuning: dict[str, dict[str, Any]],
) -> tuple[torch.Tensor, dict]:
    """``(logp, memory)`` of the zero-weight rows ``zero_index`` of a loss call: the log-probability
    forward's chunk loop compacted to those rows (:func:`_forward_logprob_rows`: ``gemm_logits_lbl`` + ``row_finalize``
    where registered, chunk-bounded buffers, the call's ``chunk_size`` and logits GEMM knobs, no count, no host
    synchronization).  ``logp`` is compact (``[|Z|]``)."""
    zproblem = replace(
        problem,
        objective="ce",
        loss_div=None,
        grad_weight_dtype=torch.bfloat16,
        entry="logprob",
    )
    logp, _, memory = _forward_logprob_rows(
        zproblem,
        X,
        W,
        labels,
        zero_index,
        None,
        backend=backend,
        tuning=gemm_tuning,
        lbl=_resolve_logprob_lbl(None),
    )
    return logp, memory


class _ZeroWeightResult(Protocol):
    logp: torch.Tensor
    memory: Optional[dict]


_ResultT = TypeVar("_ResultT", bound=_ZeroWeightResult)


# Temporaries of the chunk loop that coexist with the zero-weight rows' logp-only pass in either order: the compaction
# outputs formed before both (:func:`_with_zero_weight_rows`); with ``chunk0_alive`` also the hidden valid-row count's
# chunk-0 buffers (the pass ahead of the loop of a call that defers its last chunk's weight gradient).
_ZERO_PASS_HELD = ("row_index", "row_valid")
_ZERO_PASS_CHUNK0 = ("logits", "stats", "x_c")


def _with_zero_weight_rows(
    result: _ResultT,
    zero_index: Optional[torch.Tensor],
    zero_logp: Optional[tuple[torch.Tensor, dict]],
    *,
    chunk0_alive: bool = False,
) -> _ResultT:
    """A loss forward's result with its zero-weight rows: their ``logp`` (when the logp-only pass ran)
    scattered into the ``[T]`` output, and the memory report's ``zero_weight_rows`` /
    ``zero_weight_logp_temporary_bytes`` (only when the call had zero-weight rows).  With the pass, the report's
    ``peak_temporary_bytes`` is the larger of the chunk loop's and the pass's phase -- its own temporaries plus the
    loop's that coexist with them (the compaction outputs; with ``chunk0_alive``, the pass ran ahead of the loop while
    the hidden valid-row count held chunk 0's logits / statistics / gather buffers, which count too)."""
    if zero_index is None:
        return result
    memory = dict(result.memory or {})
    memory["zero_weight_rows"] = int(zero_index.numel())
    zero_temp = 0 if zero_logp is None else int(zero_logp[1].get("temporary_bytes", 0))
    memory["zero_weight_logp_temporary_bytes"] = zero_temp
    if zero_logp is not None:
        temporary = memory.get("temporary") or {}
        held = sum(int(temporary.get(name, 0)) for name in _ZERO_PASS_HELD)
        if chunk0_alive:
            held += sum(int(temporary.get(name, 0)) for name in _ZERO_PASS_CHUNK0)
        loop = int(memory.get("peak_temporary_bytes", memory.get("temporary_bytes", 0)))
        memory["zero_weight_logp_phase_bytes"] = zero_temp + held
        memory["peak_temporary_bytes"] = max(loop, zero_temp + held)
    result.memory = memory
    if zero_logp is not None:
        result.logp.index_copy_(0, zero_index, zero_logp[0])
    return result


def scale_cast(
    acc: torch.Tensor,
    grad: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    *,
    backend: str = "cake",
) -> torch.Tensor:
    """``cast(g * acc)`` into a new tensor of ``out_dtype`` (one rounding); ``acc`` is never written.

    The single fp32 -> output cast of the backward, for ``dW`` and ``dX`` alike
    (``g`` = the incoming scalar gradient, ``None`` = 1).
    """
    if out_dtype not in GRAD_WEIGHT_DTYPES:
        raise ValueError("the cast produces bfloat16 or float32")
    if not acc.is_contiguous() or acc.dtype != torch.float32:
        raise ValueError("the accumulator must be a contiguous FP32 tensor")
    out = torch.empty(acc.shape, dtype=out_dtype, device=acc.device)
    if acc.numel() == 0:
        return out
    g = (
        _unit_scale(acc.device)
        if grad is None
        else grad.detach().reshape(1).to(device=acc.device, dtype=torch.float32)
    )
    if backend == "reference":
        out.reshape(-1).copy_(acc.reshape(-1) * g)
        return out
    module_name, record = record_for(acc.device)
    record_abi(record)
    geometry = Geometry.from_record(record)
    stage = "scale_cast_f32" if out_dtype == torch.float32 else "scale_cast_bf16"
    if acc.numel() % geometry.cast_vec:
        raise ValueError(f"the cast needs a multiple of {geometry.cast_vec} elements")
    values: dict[str, Any] = {name: None for name in COMMON_TENSORS}
    values.update({name: 0 for name in COMMON_SCALARS})
    values.update(
        acc=acc.reshape(-1),
        g=g,
        out=out.reshape(-1),
        num_vecs=int(acc.numel() // geometry.cast_vec),
        loss_div=1.0,
    )
    launch, arguments = _eager_launch(
        record, module_name, stage, values, acc.device, geometry
    )
    index = (
        acc.device.index
        if acc.device.index is not None
        else torch.cuda.current_device()
    )
    with _ffi_stream_context(int(index)):
        if launch.prepare is not None:
            launch.prepare(*arguments)
        launch.entry(*arguments)
    return out


def scale_cast_scatter(
    acc: torch.Tensor,
    grad: Optional[torch.Tensor],
    row_index: torch.Tensor,
    scan: torch.Tensor,
    num_rows: int,
    *,
    backend: str = "cake",
) -> torch.Tensor:
    """``scatter_rows(bf16(g * acc), row_index, num_rows)`` as ONE new BF16 ``[num_rows, H]`` tensor written in one
    pass: ``bf16(g * acc[compact row])`` on the valid rows, exact zeros elsewhere.  ``acc`` is the compact FP32
    ``[T_v, H]`` accumulator (never written), ``row_index`` its ascending int64 original rows, ``scan`` the inclusive
    int32 count of valid rows (``cumsum(labels >= 0)``: ``scan[r] - 1`` is the compact row of a valid output row
    ``r``, which the kernel validates through ``row_index[scan[r] - 1] == r``), ``g`` = ``grad`` (``None`` = 1).  The
    same FP32 multiply and BF16 rounding as :func:`scale_cast`, so the valid rows are bitwise the flat cast's; every
    output element is written exactly once.  ``T_v == 0`` (every row ignored) is one zero fill: the kernel reads a
    candidate compact row for every output row and would have none."""
    if acc.dtype != torch.float32 or acc.ndim != 2 or not acc.is_contiguous():
        raise ValueError("the accumulator must be a contiguous FP32 [T_v, H] tensor")
    T_v, H = (int(s) for s in acc.shape)
    num_rows = int(num_rows)
    if (
        row_index.dtype != torch.int64
        or tuple(row_index.shape) != (T_v,)
        or not row_index.is_contiguous()
        or row_index.device != acc.device
    ):
        raise ValueError(
            f"row_index must be a contiguous int64 [{T_v}] tensor on the accumulator's device"
        )
    if (
        scan.dtype != torch.int32
        or tuple(scan.shape) != (num_rows,)
        or not scan.is_contiguous()
        or scan.device != acc.device
    ):
        raise ValueError(
            f"scan must be a contiguous int32 [{num_rows}] tensor on the accumulator's device"
        )
    if T_v > num_rows:
        raise ValueError(
            f"the accumulator has more rows ({T_v}) than the output ({num_rows})"
        )
    out = torch.empty((num_rows, H), dtype=torch.bfloat16, device=acc.device)
    if out.numel() == 0:
        return out
    if T_v == 0:
        out.zero_()
        return out
    g = (
        _unit_scale(acc.device)
        if grad is None
        else grad.detach().reshape(1).to(device=acc.device, dtype=torch.float32)
    )
    if backend == "reference":
        out.zero_()
        out.index_copy_(0, row_index, (acc * g).to(torch.bfloat16))
        return out
    module_name, record = record_for(acc.device)
    record_abi(record)
    geometry = Geometry.from_record(record)
    stage = "scale_cast_scatter_bf16"
    if stage not in record:
        raise NotImplementedError(
            f"the registered program {module_name!r} lacks the stage {stage!r}"
        )
    if H % geometry.cast_vec:
        raise ValueError(
            f"the scatter needs a row width of a multiple of {geometry.cast_vec} elements"
        )
    values: dict[str, Any] = {name: None for name in COMMON_TENSORS}
    values.update({name: 0 for name in COMMON_SCALARS})
    values.update(
        acc=acc,
        g=g,
        scan=scan,
        idx_lo=row_index.view(
            torch.int32
        ),  # the low int32 word of each int64 index (stride 2)
        out=out,
        row_vecs=int(H // geometry.cast_vec),
        num_rows=num_rows,
        loss_div=1.0,
    )
    launch, arguments = _eager_launch(
        record, module_name, stage, values, acc.device, geometry
    )
    index = (
        acc.device.index
        if acc.device.index is not None
        else torch.cuda.current_device()
    )
    with _ffi_stream_context(int(index)):
        if launch.prepare is not None:
            launch.prepare(*arguments)
        launch.entry(*arguments)
    return out


def finalize_dx(
    acc: torch.Tensor,
    grad: Optional[torch.Tensor],
    row_index: Optional[torch.Tensor],
    row_valid: Optional[torch.Tensor],
    num_rows: Optional[int],
    *,
    backend: str = "cake",
    scan: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The dX output boundary: ``bf16(g * acc)`` restored to the caller's ``[num_rows, H]`` rows.  Every row valid
    (``row_index`` None): :func:`scale_cast`, already one pass over every row.  Compacted rows with the forward's
    valid-row mask (``row_valid``: the fused dX finalize): the inclusive int32 scan -- one ``cumsum``, issued here
    behind the backward's queued GEMMs -- and one :func:`scale_cast_scatter` launch.  Compacted rows without the mask
    (the forward ran with ``dx_finalize=False``): the flat cast, then :func:`scatter_rows`
    (zero fill + ``index_copy_``).  All three produce the same bits.  ``scan`` (the zero-gradient row skip,
    :func:`zero_dlogp_partition`): the inclusive int32 scan of the rows ``acc`` holds, already formed (``row_valid``
    is then not read): the accumulator's rows past the scan's count are never read and those output rows are zeros."""
    if row_index is not None and scan is not None:
        if num_rows is None:
            raise ValueError(
                "finalize_dx needs num_rows (the caller's T) with row_index"
            )
        return scale_cast_scatter(
            acc, grad, row_index, scan, int(num_rows), backend=backend
        )
    if row_index is None:
        return scale_cast(acc, grad, torch.bfloat16, backend=backend)
    if num_rows is None:
        raise ValueError("finalize_dx needs num_rows (the caller's T) with row_index")
    if row_valid is None:
        return scatter_rows(
            scale_cast(acc, grad, torch.bfloat16, backend=backend),
            row_index,
            int(num_rows),
        )
    if row_valid.dtype != torch.bool or tuple(row_valid.shape) != (int(num_rows),):
        raise ValueError(f"row_valid must be the bool [{int(num_rows)}] valid-row mask")
    scan = torch.cumsum(row_valid, 0, dtype=torch.int32)
    return scale_cast_scatter(
        acc, grad, row_index, scan, int(num_rows), backend=backend
    )


def dw_cast(
    dz_last: torch.Tensor,
    x_rows: torch.Tensor,
    dw_acc: Optional[torch.Tensor],
    grad: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    *,
    backend: str = "cake",
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
) -> torch.Tensor:
    """The deferred last chunk's weight gradient as a new tensor: ``cast(g * (dw_acc + dz_last^T @ x_rows))``, or
    ``cast(g * dz_last^T @ x_rows)`` when ``dw_acc`` is ``None`` (a one-chunk call), through the fused GEMM
    (:data:`DW_CAST_STAGES`); ``dw_acc`` is read, never written (repeatable).  ``dz_last`` BF16 ``[rows, V]``
    contiguous, ``x_rows`` BF16 ``[rows, H]`` with a 16-byte row pitch (else copied), ``g`` = ``grad`` (``None`` = 1).
    ``gemm_tuning`` (default :func:`gemm_tuning_default`): the call's explicit GEMM knobs, whose ``dw`` entry pins this
    GEMM's form (the forward's knobs, ``ForwardResult.gemm_tuning``)."""
    if out_dtype not in GRAD_WEIGHT_DTYPES:
        raise ValueError("the cast produces bfloat16 or float32")
    if (
        dz_last.dtype != torch.bfloat16
        or dz_last.ndim != 2
        or not dz_last.is_contiguous()
    ):
        raise ValueError("dz_last must be a contiguous BF16 [rows, V] tensor")
    rows, V = dz_last.shape
    if (
        x_rows.dtype != torch.bfloat16
        or x_rows.ndim != 2
        or x_rows.shape[0] != rows
        or x_rows.stride(1) != 1
    ):
        raise ValueError("x_rows must be a BF16 [rows, H] tensor with contiguous rows")
    if x_rows.stride(0) % 8 or x_rows.data_ptr() % 16:
        x_rows = x_rows.contiguous()
    H = int(x_rows.shape[1])
    if dw_acc is not None and (
        dw_acc.dtype != torch.float32
        or tuple(dw_acc.shape) != (V, H)
        or not dw_acc.is_contiguous()
    ):
        raise ValueError(
            "dw_acc must be a contiguous FP32 [V, H] tensor (or None for a one-chunk call)"
        )
    device = dz_last.device
    out = torch.empty((V, H), dtype=out_dtype, device=device)
    if rows == 0:
        return out.zero_()
    g = (
        _unit_scale(device)
        if grad is None
        else grad.detach().reshape(1).to(device=device, dtype=torch.float32)
    )
    if backend == "reference":
        product = _mm_fp32(dz_last.t(), x_rows)
        acc = product if dw_acc is None else dw_acc + product
        out.copy_(acc * g)
        return out
    module_name, record = record_for(device)
    record_abi(record)
    # the last chunk's form and scalars: the raster / block rules of the record's architecture on this chunk's rows, or
    # the explicit knobs (the block rule and knob apply to the fp32 cast only; the bf16 cast keeps the 1-D raster)
    fp32 = out_dtype == torch.float32
    dw_tuning = _resolve_gemm_tuning(gemm_tuning).get("dw", {})
    raster = raster_variant(H, int(rows), record["arch"])
    group_m = (
        dw_tuning["group_m"]
        if "group_m" in dw_tuning
        else (raster[1] if raster else None)
    )
    group_n = None
    if fp32:
        group_n = (
            dw_tuning["group_n"]
            if "group_n" in dw_tuning
            else dw_block_variant(int(rows), H, record["arch"])
        )
        group_n = None if group_n in (None, 0) else int(group_n)
    stage = stage_variant(
        "gemm_dw_cast_f32" if fp32 else "gemm_dw_cast_bf16",
        blocked=group_n is not None,
        epi_store="tma"
        if (
            not fp32
            and dw_acc is not None
            and "epi_store" not in dw_tuning
            and dw_cast_tma_rule(2, int(rows), record["arch"])
        )
        else None,
    )
    if stage not in registered_stages(module_name):
        raise NotImplementedError(
            f"the registered program {module_name!r} lacks the stage {stage!r} (fuse_dw_cast)"
        )
    geometry = Geometry.from_record(record)
    index = device.index if device.index is not None else torch.cuda.current_device()
    values: dict[str, Any] = {name: None for name in COMMON_TENSORS}
    values.update({name: 0 for name in COMMON_SCALARS})
    values.update(
        A=dz_last,
        B=x_rows,
        C=out,
        STATS_OUT=g,
        WS=dw_acc if dw_acc is not None else _device_constants(int(index))["f32_dummy"],
        rows_c=int(rows),
        T=int(rows),
        H=H,
        V=int(V),
        loss_div=1.0,
        **gemm_scalars(
            stage,
            geometry,
            hidden=H,
            vocab=int(V),
            rows_c=int(rows),
            group_m=group_m,
            group_n=group_n,
            first_chunk=int(dw_acc is None),
        ),
    )
    launch, arguments = _eager_launch(
        record, module_name, stage, values, device, geometry
    )
    with _ffi_stream_context(int(index)):
        if launch.prepare is not None:
            launch.prepare(*arguments)
        launch.entry(*arguments)
    return out


def backward_loss(
    dx_acc: Optional[torch.Tensor],
    dw_acc: Optional[torch.Tensor],
    grad: Optional[torch.Tensor] = None,
    *,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cake",
    row_index: Optional[torch.Tensor] = None,
    num_rows: Optional[int] = None,
    dz_last: Optional[torch.Tensor] = None,
    x_last: Optional[torch.Tensor] = None,
    x_src: Optional[torch.Tensor] = None,
    x_idx: Optional[torch.Tensor] = None,
    row_valid: Optional[torch.Tensor] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward of the loss entry from the saved accumulators: ``(dX, dW)`` = ``bf16(g * dX_acc)``,
    ``cast(g * dW_acc)``; the accumulators are read, never written (repeatable).  A compacted forward
    (``ForwardResult.row_index`` / ``num_rows``): the compact rows are cast once, then scattered into a
    zero BF16 ``[T, H]``.  A deferred weight gradient (``fuse_dw_cast``; ``ForwardResult.dz_last`` with
    ``x_last`` or ``x_src`` / ``x_idx``): ``dW = cast(g * (dW_acc + dz_last^T @ X_last))`` through the fused
    GEMM (:func:`dw_cast`), ``dW_acc`` being the chunks before the last one or ``None``;
    :meth:`ForwardResult.backward` passes everything.  ``row_valid`` (``ForwardResult.row_valid``, the
    valid-row mask of a compacted forward that ran with the fused dX finalize) selects the one-pass
    scale-cast-scatter of the compact rows (:func:`finalize_dx`); without it the compact rows are cast,
    then scattered into a zero BF16 ``[T, H]``.  Both give the same bits.  ``gemm_tuning`` (default
    :func:`gemm_tuning_default`; ``ForwardResult.gemm_tuning``): the forward's explicit GEMM knobs, which pin the
    deferred weight-gradient GEMM's form (:func:`dw_cast`).

    This call continues a :func:`forward_loss` call of this package; pass it only that call's accumulators and
    operands.  Its launches (the casts and the deferred weight-gradient GEMM) include no K-sliced dX GEMM: no alert
    under torch's deterministic algorithms."""
    dx = None
    if dx_acc is not None:
        if row_index is not None and num_rows is None:
            raise ValueError(
                "backward_loss needs num_rows (the caller's T) with row_index"
            )
        dx = finalize_dx(dx_acc, grad, row_index, row_valid, num_rows, backend=backend)
    if dz_last is not None:
        if x_last is None:
            if x_src is None or x_idx is None:
                raise ValueError(
                    "a deferred weight gradient needs x_last, or x_src with the chunk's row index x_idx"
                )
            x_last = torch.index_select(
                x_src, 0, x_idx
            )  # the compacted chunk's rows, gathered again
        dw = dw_cast(
            dz_last,
            x_last,
            dw_acc,
            grad,
            grad_weight_dtype,
            backend=backend,
            gemm_tuning=gemm_tuning,
        )
    elif dw_acc is not None and _is_zero_accumulator(dw_acc):
        # a call without a gradient GEMM (:func:`_empty_forward`): no FP32 accumulator to read
        dw = _zero_gradient(dw_acc, grad, grad_weight_dtype)
    else:
        dw = (
            None
            if dw_acc is None
            else scale_cast(dw_acc, grad, grad_weight_dtype, backend=backend)
        )
    return dx, dw


# :func:`backward_logprob`'s ``row_index`` default: count the valid rows again
_RECOUNT = object()


def forward_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    logprob_lbl: Optional[bool] = None,
) -> ForwardResult:
    """Forward of the log-probability entry: ``logp`` (FP32 ``[T]``, 0 on ignored rows) plus the saved statistic
    ``lse`` (compacted: the ``[T_v]`` rows of ``row_index``, exactly what :func:`backward_logprob` expects back,
    together with ``row_index``).  ``compact_rows`` (default :func:`compact_rows_default`), ``hidden_count`` (default
    :func:`hidden_count_default`) and ``gemm_tuning`` (default :func:`gemm_tuning_default`) as in
    :func:`forward_loss`; ``logprob_lbl`` (default :func:`logprob_lbl_default`) writes only the row statistics and
    the label column's logit from the logits GEMM (bitwise the same ``lse`` / ``logp``).  This forward has no
    weight gradient, so it takes no ``fuse_dw_cast`` / ``dx_finalize`` / ``dw_stream``."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = validate_lm_head_inputs(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        entry="logprob",
    )
    tuning = _resolve_gemm_tuning(gemm_tuning)
    lbl = _resolve_logprob_lbl(logprob_lbl)
    T = problem.num_rows
    if T == 0:
        return _empty_forward(problem, X, need_dx=False, need_dw=False)
    compact = _resolve_compact(compact_rows)
    hc = _hidden_count(
        problem,
        X,
        W,
        labels,
        backend=backend,
        compact=compact,
        stats=True,
        hidden_count=_resolve_hidden_count(hidden_count),
    )
    if hc is not None:
        row_index = hc.row_index()
    else:
        row_index = valid_row_index(labels) if compact else None
    if row_index is not None and row_index.numel() == 0:  # every row ignored
        return _empty_forward(
            problem, X, need_dx=False, need_dw=False, row_index=row_index
        )
    logp, lse, memory = _forward_logprob_rows(
        problem,
        X,
        W,
        labels,
        row_index,
        hc,
        backend=backend,
        tuning=tuning,
        lbl=lbl,
    )
    result = ForwardResult(
        loss=None,
        logp=scatter_rows(logp, row_index, T),
        lse=lse,
        memory=memory,
        backend=backend,
        row_index=row_index,
        num_rows=T,
    )
    result.problem = problem
    return result


def _forward_logprob_rows(
    problem: Problem,
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    row_index: Optional[torch.Tensor],
    hc: Optional[_HiddenCount],
    *,
    backend: str,
    tuning: dict[str, dict[str, Any]],
    lbl: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """``(logp, lse, memory)`` of the log-probability forward's chunk loop over the rows ``row_index`` (``None`` =
    every row; non-empty otherwise), through the remembered binding or a new one: :func:`forward_logprob`'s body after
    its count, and the zero-weight rows' logp-only pass of a loss call (:func:`_zero_weight_logp`).  ``logp`` / ``lse``
    are compact when ``row_index`` is given."""
    chunk_size = problem.chunk
    valid_rows = None if row_index is None else int(row_index.numel())
    cache = BINDING_CACHE
    key = None
    if backend == "cake" and cache.enabled:
        key = forward_binding_key(
            X,
            W,
            labels,
            objective="ce",
            loss_div=None,
            infer_logp=None,
            loss_weights=None,
            chunk_size=chunk_size,
            need_dx=False,
            need_dw=False,
            grad_weight_dtype=torch.bfloat16,
            entry="logprob",
            valid_rows=valid_rows,
            gemm_tuning=tuning,
            hidden_count=hc is not None,
            logprob_lbl=lbl,
        )
        binding = cache.lookup(key)
        if binding is not None:
            logp, lse = binding.forward(X, W, labels, None, None, row_index, hc)
            return logp, lse, binding.memory
    runner = prepare_lm_head_loss(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        need_dx=False,
        need_dw=False,
        entry="logprob",
        backend=backend,
        compact_rows=False if row_index is None else row_index,
        gemm_tuning=tuning,
        logprob_lbl=lbl,
    )
    if hc is not None:
        # the hidden valid-row count: chunk 0's logits already sit in the prologue's buffers, so the first call launches
        # through the templated binding too (the runner's own workspace is never written)
        binding = _Binding.from_runner(runner, hc)
        del runner
        if key is not None:
            cache.remember(key, binding)
        logp, lse = binding.forward(X, W, labels, None, None, row_index, hc)
        return logp, lse, binding.memory
    logp, lse = runner.forward()
    if key is not None:
        cache.remember(key, _Binding.from_runner(runner))
    return logp, lse, runner.memory


def _dlogp_rows(dlogp: torch.Tensor, T: int) -> torch.Tensor:
    """The incoming ``dlogp`` as the FP32 contiguous ``[T]`` tensor a log-probability backward binds
    (``dlogp.detach().reshape(T).to(torch.float32).contiguous()``): ``dlogp`` itself when it already is one and needs
    no detach (the autograd backward's gradient of the FP32 ``[T]`` ``logp``), without those four host calls ahead of
    the backward's first launch."""
    if (
        dlogp.dtype == torch.float32
        and dlogp.dim() == 1
        and dlogp.shape[0] == T
        and not dlogp.requires_grad
        and dlogp.is_contiguous()
    ):
        return dlogp
    return dlogp.detach().reshape(T).to(torch.float32).contiguous()


def _zero_dlogp_skip(
    device: torch.device,
    row_index: Optional[torch.Tensor],
    backend: str,
    dz_arm: bool,
    need_dx: bool,
    fused_dx: bool,
) -> bool:
    """Whether a log-probability backward that reuses the forward's row index is eligible for the zero-gradient row
    skip: a compacted call (``row_index`` given) of at least :func:`zero_dlogp_min_rows` rows on the ``cake`` backend,
    with the ``dz`` arm and, when it produces ``dX``, the fused dX finalize (which writes the skipped rows' exact zeros),
    on a record that registers the counted dz arm (:data:`LOGPROB_DZ_DCNT_STAGE`) and the partition kernel
    (:data:`ZERO_DLOGP_PARTITION_STAGE`).  An eligible call runs the skip when its plan has the counted forms
    (:attr:`Plan.dcnt`), and the backward without it otherwise.  The floor is the device's
    (:func:`zero_dlogp_min_rows` of its index: the override, else its architecture's default)."""
    if (
        row_index is None
        or backend != "cake"
        or not dz_arm
        or (need_dx and not fused_dx)
        or device.type != "cuda"
    ):
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    floor = zero_dlogp_min_rows(index)
    if floor is None or int(row_index.numel()) < floor:
        return False
    # the device's record and its stages, resolved once per device (:func:`_skip_record`)
    return _skip_record(index)[3]


def backward_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    dlogp: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    need_dx: bool = True,
    need_dw: bool = True,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    dw_stream: Optional[Union[bool, str]] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    logprob_dz: Optional[bool] = None,
    row_index: Any = _RECOUNT,
    _problem: Optional[Problem] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward of the log-probability entry: recompute each chunk's logits, ``dz = dlogp_t *
    (1[v = y_t] - softmax)`` (ignored rows zero), then ``dX`` (BF16) and ``dW`` (BF16) through
    the FP32 accumulators (the chunk loop; an eligible call that reuses the forward's row index runs the
    vocabulary-major form instead, :func:`_backward_logprob_vm`, with no FP32 ``dW`` accumulator).  ``(None, None)``
    when neither input is trainable.  ``compact_rows``
    must match the forward's: the valid-row index is the forward's ``row_index``
    (:attr:`ForwardResult.row_index`; ``None`` = uncompacted) when given -- no count, no host
    synchronization -- else recomputed from ``labels`` (identical to the forward's); ``lse`` is then
    the forward's compact ``[T_v]`` statistic and ``dX`` comes back scattered to ``[T, H]``; every
    row ignored returns zeros without launching.  ``fuse_dw_cast``
    (default :func:`fuse_dw_cast_default`): the last chunk's weight-gradient GEMM writes the final BF16
    ``dW`` from its epilogue (the scale is 1 here).  ``dx_finalize`` (default :func:`dx_finalize_default`,
    resolved once here): a compacted backward writes ``dX`` in one pass from the accumulator.  ``dw_stream``
    (default ``None`` = ``"auto"``, :func:`dw_stream_mode`), ``hidden_count`` (default
    :func:`hidden_count_default`; the recount path only) and ``gemm_tuning`` (default
    :func:`gemm_tuning_default`) as in :func:`forward_loss`.  ``logprob_dz`` (default
    :func:`logprob_dz_default`): each chunk's recompute writes ``dz`` from the logits GEMM's
    epilogue instead of storing the logits for ``row_grad`` (bitwise the same ``dz``, hence the same ``dW`` and the
    same ``dX`` operands; a chunk of more than one dX slice sums its slices in slice order).

    Zero-gradient row skip (:func:`zero_dlogp_partition`; a compacted backward that reuses the forward's row index,
    with the ``dz`` arm and, when it produces ``dX``, the fused dX finalize, of at least :func:`zero_dlogp_min_rows`
    rows, on a record with the ``_dcnt`` forms): the chunk loop takes the valid rows whose ``dlogp`` is nonzero first and
    its dz, dX and dW GEMMs bound their work by the per-chunk device count of those rows (no host synchronization;
    capturable).  The rows of a zero ``dlogp`` get exact-zero ``dX`` rows and add nothing to ``dW``; ``dX`` of every
    other row is the same; ``dW`` sums the same nonzero products over the live rows in the new row order (FP32).  The
    call's plan decides: a plan whose chunks take a form without a counted one (an explicitly pinned swapped dX tile or
    weight-gradient TMA epilogue) runs the backward without the skip, rows in the forward's order.

    ``_problem`` (internal, the autograd Function's): the forward's validated :class:`Problem` of these same ``X``,
    ``W`` and ``labels`` (:attr:`ForwardResult.problem`, passed only when no saved-tensor hooks could have unpacked
    another ``X``), taken instead of validating them again."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = (
        _problem
        if _problem is not None
        else validate_lm_head_inputs(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            entry="logprob",
        )
    )
    stream_mode = dw_stream_mode(dw_stream)
    tuning = _resolve_gemm_tuning(gemm_tuning)
    dz_arm = _resolve_logprob_dz(logprob_dz)
    # read once: the binding key and the plan take the same settings
    rule = dx_slice_rule()
    if not (need_dx or need_dw):
        return None, None
    T, H, V = problem.num_rows, problem.hidden, problem.vocab
    fused_dx = bool(
        need_dx and _resolve_dx_finalize(dx_finalize)
    )  # resolved once per call
    compact = bool(T and _resolve_compact(compact_rows))
    hc = None
    # the zero-gradient row skip: eligible for a call that reuses the forward's row index (resolved below), run when
    # the call's plan has its counted forms (``order_rows``)
    skip = False
    scan = counts = None
    # the fused dX finalize's valid-row mask (labels >= 0) of a call that reuses the forward's row index: one device
    # comparison, no synchronization, formed in finish_dx behind the queued GEMMs (the call's first launches are not
    # held back by it)
    labels_mask = False
    reuse_index = row_index is not _RECOUNT
    if reuse_index:
        # the forward's valid-row index: the plan is known without a count
        if row_index is not None and not compact:
            raise ValueError("a valid-row index belongs to a compacted backward")
        skip = _zero_dlogp_skip(X.device, row_index, backend, dz_arm, need_dx, fused_dx)
        row_valid = None
        labels_mask = row_index is not None and fused_dx and not skip
    else:
        hc = _hidden_count(
            problem,
            X,
            W,
            labels,
            backend=backend,
            compact=compact,
            stats=False,
            hidden_count=_resolve_hidden_count(hidden_count),
        )
        if hc is not None:
            # the hidden valid-row count: chunk 0's gather and its recompute GEMM (no statistics) are queued
            row_index = hc.row_index()
            row_valid = hc.valid if (row_index is not None and fused_dx) else None
        else:
            row_index, row_valid = (
                valid_rows(labels, mask=fused_dx) if compact else (None, None)
            )
    rows = T if row_index is None else int(row_index.numel())
    if rows == 0:  # no rows, or every row ignored
        return (
            torch.zeros((T, H), dtype=torch.bfloat16, device=X.device)
            if need_dx
            else None,
            torch.zeros((V, H), dtype=torch.bfloat16, device=X.device)
            if need_dw
            else None,
        )
    _check_output(lse, "lse", (rows,), torch.float32)
    dlogp = _dlogp_rows(dlogp, T)
    if reuse_index and _logprob_vm_eligible(
        problem,
        X,
        rows,
        backend=backend,
        need_dw=need_dw,
        dz_arm=dz_arm,
        tuning=tuning,
        temperature=False,
    ):
        return _backward_logprob_vm(
            problem,
            X,
            W,
            labels,
            lse,
            dlogp,
            row_index,
            need_dx=need_dx,
            fused_dx=fused_dx,
            skip=skip,
            stream_mode=stream_mode,
            backend=backend,
        )
    cache = BINDING_CACHE
    key = None
    valid_count = None if row_index is None else rows
    fuse = bool(need_dw and _resolve_fuse(fuse_dw_cast))
    # the fused dX finalize of a compacted backward: the runner leaves dx_acc uncast and finalize_dx writes the [T, H]
    # dX in one pass (uncompacted: the in-loop flat cast is already one pass over every row)
    dx_cast = not (fused_dx and row_index is not None)

    def finish_dx(dx):
        if dx is None:
            return None
        if dx_cast:
            return scatter_rows(dx, row_index, T)
        valid = (labels >= 0) if labels_mask else row_valid
        return finalize_dx(dx, None, row_index, valid, T, backend=backend, scan=scan)

    def order_rows(plan: Plan) -> None:
        # the plan decides the skip: its counted forms order the rows (one partition launch, the live counts per chunk
        # of the plan's stride: a folded plan's one chunk); a plan without them runs the shipped backward over the
        # forward's row order (no partition, the fused finalize's valid-row mask)
        nonlocal skip, row_index, lse, scan, counts, labels_mask
        skip = skip and plan.dcnt
        if skip:
            row_index, lse, scan, counts = zero_dlogp_partition(
                labels, dlogp, row_index, lse, chunk_stride(plan.chunks, problem.chunk)
            )
        elif row_valid is None and row_index is not None and fused_dx:
            labels_mask = True

    if backend == "cake" and cache.enabled:
        key = logprob_backward_binding_key(
            X,
            W,
            labels,
            lse,
            dlogp,
            chunk_size=chunk_size,
            need_dx=need_dx,
            need_dw=need_dw,
            valid_rows=valid_count,
            fuse_dw_cast=fuse,
            gemm_tuning=tuning,
            dx_finalize=fused_dx,
            hidden_count=hc is not None,
            logprob_dz=dz_arm,
            dx_rule=rule,
            zero_dlogp=skip,
        )
        binding = cache.lookup(key)
        if binding is not None:
            _alert_unordered_dx(binding.plan, "backward_logprob", "logprob")
            # the key of an eligible call remembers the plan it took (counted, or the shipped one)
            order_rows(binding.plan)
            dx, dw = binding.backward_logprob(
                X, W, labels, lse, dlogp, row_index, hc, stream_mode, counts
            )
            return finish_dx(dx), dw
    runner = prepare_lm_head_loss(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        need_dx=need_dx,
        need_dw=need_dw,
        entry="logprob",
        dlogp=dlogp,
        lse=lse,
        backend=backend,
        compact_rows=False if row_index is None else row_index,
        fuse_dw_cast=fuse,
        gemm_tuning=tuning,
        dx_finalize=fused_dx,
        dx_cast=dx_cast,
        dw_stream=stream_mode,
        logprob_dz=dz_arm,
        dx_rule=rule,
        # an eligible call plans with the counted forms; the counts follow the plan (the runner binds this slot, the
        # call launches from its binding with the counts of the rows it ordered)
        dcnt=torch.empty(
            (len(loop_chunks(T, rows, problem.chunk)),),
            dtype=torch.int32,
            device=X.device,
        )
        if skip
        else None,
        # the skip's private forms (the cast at the last live chunk): the counts come from zero_dlogp_partition (zeros
        # form a suffix) and the gradient buffers are this call's own
        _skip_forms=skip,
    )
    _alert_unordered_dx(runner.plan, "backward_logprob", "logprob")
    order_rows(runner.plan)
    if hc is not None or skip:
        binding = _Binding.from_runner(runner, hc)
        del runner
        if key is not None:
            cache.remember(key, binding)
        dx, dw = binding.backward_logprob(
            X, W, labels, lse, dlogp, row_index, hc, stream_mode, counts
        )
        return finish_dx(dx), dw
    dx, dw = runner._backward()
    if key is not None:
        cache.remember(key, _Binding.from_runner(runner))
    return finish_dx(dx), dw


# Vocabulary-major log-probability backward (:func:`backward_logprob` / :func:`backward_logprob_temperature` of a
# multi-chunk call): the bounded default form without the FP32 ``[V, H]`` weight-gradient accumulator.  Per vocabulary
# block ``[v0, v1)`` (:func:`_logprob_vm_block_rows` rows, about ``chunk x V / T_v``: the block's ``dz`` holds about one
# chunk's logits, the O(C V) workspace of the chunk loop) the ``dz`` arm recomputes ``dz[T_v, v1 - v0]`` over every
# row of the call (labels shifted into the block, the forward's ``lse``), ONE weight-gradient GEMM over ``K = T_v``
# writes the block's BF16 ``dW`` rows (``bf16(dz^T @ X)``: FP32 accumulation inside the GEMM, one cast; the bf16 fused
# cast's one-chunk form) and the dX GEMM writes ``dz @ W[v0:v1]`` into the FP32 ``[T_v, H]`` dX accumulator: the
# first block stores (the plain-store form; under the zero-gradient row skip every block adds into the ``-0.0`` fill,
# the exact additive identity, which reproduces the store bit for bit), each later block adds (the reduction form at
# ONE K slice: one writer per element per launch; its ordered form's slab operand is ``dX_acc`` itself,
# :func:`dx_ordered_operands`) in block order on one stream.  Every output therefore repeats bit for
# bit; ``dW`` / ``dX`` round differently from the chunk loop (other summation groupings, same FP32 contract).  The
# weight-gradient GEMM of a block runs on the dW side stream (:func:`dw_side_stream` of the block count) beside the
# block's dX GEMM and joins before the next block's recompute.  The dX GEMM takes the chunk rule's knobs for ``T_v``
# rows (:func:`chunk_variants`), except on an architecture whose long-chunk ring takes no split: there a launch that
# would run that ring with a short last wave runs the table ring with the split-tail form
# (:func:`_logprob_vm_dx_tuning`).  The kept logits (:func:`backward_logprob_keep`) keep
# their own chunk loop (no recompute to save there): their ``dX`` / ``dW`` round differently from this form's.
_LOGPROB_VM_MODES = (
    "0",
    "1",
    "auto",
)  # never / every eligible call (one block included) / the auto rule's eligible calls (the default)
_LOGPROB_VM = {"mode": "auto"}
# the auto rule's largest chunk count (``ceil(rows / chunk)``) of the vocabulary-major form; a call of more chunks keeps
# the chunk loop.  The form's block count follows the chunk count (``rows * block <= chunk * V``) and each block adds
# its dX into the FP32 ``[rows, H]`` accumulator (a read-modify-write of ``8 * rows * H`` bytes per block), so its cost
# over the chunk loop grows with the chunk count: measured on B200 and R200 (CUPTI, cold L2), calls of 8 and 16 chunks
# ran slower than the chunk loop, calls of 4 chunks at parity or faster.
VM_AUTO_MAX_CHUNKS = 4
# a call of 2 chunks whose second chunk holds at least half a chunk (``rows - chunk >= chunk / 2``) keeps the chunk
# loop under the auto rule: measured on B200 (CUPTI, cold L2), the 8117-token GLM call (C 4096, about 7700 valid
# rows) ran its backward 1.9 % and its checkpointed step 1.3 % slower than the chunk loop with the split-tail dX of
# :func:`_logprob_vm_dx_tuning` (its one dX launch per block over every row ran longer than the chunk loop's two
# launches); a call whose second chunk is short (4600 tokens) ran faster.


def _logprob_vm_auto_rows(rows: int, chunk: int) -> bool:
    """The auto rule's row counts of the vocabulary-major form: 2 .. :data:`VM_AUTO_MAX_CHUNKS` chunks
    (``chunk < rows <= VM_AUTO_MAX_CHUNKS * chunk``), except a call of 2 chunks whose second chunk holds at least
    half a chunk."""
    rows, chunk = int(rows), int(chunk)
    if not chunk < rows <= VM_AUTO_MAX_CHUNKS * chunk:
        return False
    return not (rows <= 2 * chunk and 2 * (rows - chunk) >= chunk)


# a block-shifted label outside the block (:func:`_backward_logprob_vm`): equal to no column, and not negative (a
# negative label is an ignored row)
VM_FOREIGN_LABEL = 1 << 30
# the largest row count of the vocabulary-major form (the partition kernel and the row kernels index rows of one
# chunk by blockIdx.y): a larger call keeps the chunk loop
VM_MAX_ROWS = 65535


def _logprob_vocab_major() -> str:
    """The vocabulary-major backward's mode (:data:`_LOGPROB_VM_MODES`; default ``"auto"``)."""
    return _LOGPROB_VM["mode"]


def _set_logprob_vocab_major(mode) -> str:
    """Internal (tests and measurements; no public switch): select the log-probability backward's form for the
    following calls of this process and return the previous mode: ``"auto"`` (``None``) the vocabulary-major form for
    an eligible call of the auto rule's row counts (:func:`_logprob_vm_auto_rows`), ``"1"`` (``True``) for every
    eligible call, ``"0"``
    (``False``) the chunk loop always."""
    if mode is None:
        value = "auto"
    elif isinstance(mode, bool):
        value = "1" if mode else "0"
    elif isinstance(mode, str) and mode.strip().lower() in _LOGPROB_VM_MODES:
        value = mode.strip().lower()
    else:
        raise ValueError(
            f"vocabulary-major mode must be one of {_LOGPROB_VM_MODES}, got {mode!r}"
        )
    previous = _LOGPROB_VM["mode"]
    _LOGPROB_VM["mode"] = value
    return previous


def _logprob_vm_block_rows(rows: int, vocab: int, chunk: int, step: int = 256) -> int:
    """Vocabulary rows of one block of the vocabulary-major backward: the largest multiple of ``step`` whose ``dz``
    block of ``rows`` rows holds at most ``chunk x vocab`` elements (one chunk's logits; at least ``step``), then
    balanced over the block count (``ceil(vocab / n)`` rounded up to ``step``)."""
    rows, vocab, chunk, step = int(rows), int(vocab), int(chunk), int(step)
    full = -(-vocab // step) * step
    cap = min(full, max(step, (chunk * vocab // max(rows, 1)) // step * step))
    n = -(-vocab // cap)
    per_block = -(-vocab // n)
    return min(cap, -(-per_block // step) * step)


def _logprob_vm_dx_tuning(
    rows: int,
    hidden: int,
    num_sms: int,
    geometry: "Geometry",
    arch: Optional[str],
    num_blocks: int,
    registered: set,
) -> Optional[dict[str, dict[str, Any]]]:
    """The dX knobs of the vocabulary-major backward (``None`` = the chunk rule's, :func:`chunk_variants`).  Its dX
    runs every row in one launch per block, so the launch's last wave is the whole call's.  On an architecture whose
    long-chunk ring takes no split (not in ``DX_SPLIT_LONG_RING_ARCHES``), a launch the ring rule puts on that ring
    runs the table ring instead (``{"dx": {"stages": None}}``) when the table ring's wide items split their last wave
    (:func:`dx_split_rule`) and the record registers the split forms of both dX stages of the backward.  Bitwise: the
    ring depth and the split change the tile schedule only (the same K order per element)."""
    if arch is None or arch in DX_SPLIT_LONG_RING_ARCHES:
        return None
    tile = dx_tile_rule(rows, hidden, num_sms, 1, geometry, arch)
    if dx_stages_variant(rows, hidden, tile, arch) is None or not dx_split_eligible(
        rows, hidden, arch, tile, num_blocks
    ):
        return None
    if not dx_split_rule(rows, hidden, num_sms, 1, geometry, arch, tile, None):
        return None
    tile_n = None if tile is None or int(tile) == DX_TILE_WIDE else int(tile)
    forms = {
        dx_stage(1, tile_n, None, split=True),
        stage_variant(DX_RED_STAGE, tile_n=tile_n, stages=None, split=True),
        stage_variant(DX_RED_STAGE, tile_n=tile_n, stages=None, split=True, dcnt=True),
    }
    return {"dx": {"stages": None}} if forms <= registered else None


def _logprob_vm_eligible(
    problem: Problem,
    X: torch.Tensor,
    rows: int,
    *,
    backend: str,
    need_dw: bool,
    dz_arm: bool,
    tuning: dict,
    temperature: bool,
    any_chunks: bool = False,
) -> bool:
    """Whether a log-probability backward that reuses the forward's row index runs the vocabulary-major form: mode
    ``"auto"`` with the auto rule's row counts (:func:`_logprob_vm_auto_rows`; ``"1"``, or ``any_chunks`` -- the
    opt-in budget form, which compares the forms' step peaks itself -- under ``"auto"``: any row count), the
    ``cake`` backend on CUDA, a weight gradient, the ``dz`` arm, no explicit GEMM knob, at most :data:`VM_MAX_ROWS`
    rows, and a record that registers the forms
    (the ``dz`` arm, its temperature form for a call with one, the plain-store dX GEMM, its one-slice reduction and
    the bf16 fused cast; the counted forms of the three for the zero-gradient row skip)."""
    mode = _logprob_vocab_major()
    if (
        mode == "0"
        or backend != "cake"
        or X.device.type != "cuda"
        or not need_dw
        or not dz_arm
        or tuning
        or not 1 <= int(rows) <= VM_MAX_ROWS
        or (
            mode == "auto"
            and not any_chunks
            and not _logprob_vm_auto_rows(rows, problem.chunk)
        )
    ):
        return False
    module_name, _record = record_for(X.device)
    stages = (
        registered_stages_with_temperature(module_name)
        if temperature
        else registered_stages(module_name)
    )
    need = {
        LOGPROB_DZ_TAU_STAGE if temperature else LOGPROB_DZ_STAGE,
        "gemm_dx",
        DX_RED_STAGE,
        DX_RED_STAGE + "_dcnt",
        "gemm_dw_cast_bf16",
        "gemm_dw_cast_bf16_dcnt",
    }
    return need <= set(stages)


def _backward_logprob_vm(
    problem: Problem,
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    dlogp: torch.Tensor,
    row_index: Optional[torch.Tensor],
    *,
    need_dx: bool,
    fused_dx: bool,
    skip: bool,
    stream_mode: str,
    backend: str,
    tau_rows_of: Optional[Callable[[Optional[torch.Tensor]], torch.Tensor]] = None,
    counted_tau_dz: bool = True,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """``(dX, dW)`` of an eligible log-probability backward (:func:`_logprob_vm_eligible`) in the vocabulary-major
    form (the section comment above).  ``row_index``: the forward's valid-row index (``None`` = uncompacted),
    ``dlogp`` the caller's FP32 ``[T]`` gradient, ``lse`` the forward's ``[rows]`` statistic.  ``skip``: the
    zero-gradient row skip (:func:`zero_dlogp_partition` over ONE chunk of all rows; the counted forms bound the
    recompute, the dX reductions and the weight-gradient K steps by the live-row count).  ``tau_rows_of`` (a call with
    a temperature): the compact temperature of a row order (:func:`backward_logprob_temperature`'s); the recompute
    runs the ``dz`` arm's temperature form (its counted form under the skip when ``counted_tau_dz``)."""
    T, H, V = int(problem.num_rows), int(problem.hidden), int(problem.vocab)
    device = X.device
    index = int(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    module_name, record = record_for(device)
    record_abi(record)
    geometry = Geometry.from_record(record)
    arch = record["arch"]
    temperature = tau_rows_of is not None
    rows = T if row_index is None else int(row_index.numel())
    scan = counts = None
    if skip:
        # the live rows first over ONE chunk of every row: counts[0] = the call's live-row count
        row_index, lse, scan, counts = zero_dlogp_partition(
            labels, dlogp, row_index, lse, rows
        )
    tau_c = tau_rows_of(row_index) if temperature else None
    labels_c = _prepare_labels(labels, geometry, row_index)
    d_c = dlogp if row_index is None else dlogp.index_select(0, row_index)
    if row_index is not None:
        x_rows = X.index_select(0, row_index)
    elif X.stride(1) == 1 and X.stride(0) % 8 == 0 and X.data_ptr() % 16 == 0:
        x_rows = X  # the GEMMs read the rows in place
    else:
        x_rows = X.contiguous()
    step = math.lcm(
        int(geometry.row_tile)
        * max(int(geometry.cta_group), int(geometry.dw_cluster_ctas)),
        int(geometry.logits_item_cols),
        int(geometry.stats_tile),
        int(geometry.vocab_multiple),
    )
    vb_max = _logprob_vm_block_rows(rows, V, int(problem.chunk), step)
    blocks = [(v0, min(V, v0 + vb_max)) for v0 in range(0, V, vb_max)]
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    registered = set(
        registered_stages_with_temperature(module_name)
        if temperature
        else registered_stages(module_name)
    )
    v = chunk_variants(
        rows,
        H,
        num_sms,
        1,
        geometry,
        arch,
        _logprob_vm_dx_tuning(
            rows, H, num_sms, geometry, arch, len(blocks), registered
        ),
        num_chunks=len(blocks),
        ce_loss=False,
    )
    dz_stage = stage_form(
        LOGPROB_DZ_STAGE,
        dcnt=bool(skip and (counted_tau_dz or not temperature)),
        temperature=temperature,
    )
    dz_scalar_stage = dz_stage[: -len("_tau")] if temperature else dz_stage
    split = v.dx_split is not None
    dx_store = dx_stage(1, v.dx_tile_n, v.dx_stages, split=split)
    dx_add = stage_variant(
        DX_RED_STAGE, tile_n=v.dx_tile_n, stages=v.dx_stages, split=split, dcnt=skip
    )
    dw_stage = stage_variant("gemm_dw_cast_bf16", dcnt=skip)
    for stage in (dx_store, dx_add, dw_stage, dz_stage):
        if stage not in registered:
            raise NotImplementedError(
                f"the registered program {module_name!r} lacks the stage {stage!r} (vocabulary-major backward)"
            )
    consts = _device_constants(index)
    dummy = consts["f32_dummy"]
    unit = _unit_scale(device)
    dz_buf = torch.empty((rows * vb_max,), dtype=torch.bfloat16, device=device)
    stats = torch.empty(
        (rows, -(-vb_max // int(geometry.stats_tile)), 2),
        dtype=torch.float32,
        device=device,
    )
    dx_acc = (
        torch.empty((rows, H), dtype=torch.float32, device=device) if need_dx else None
    )
    if dx_acc is not None and skip:
        dx_acc.fill_(-0.0)  # the exact additive identity: every block adds into it
    dw_out = torch.empty((V, H), dtype=torch.bfloat16, device=device)
    cnt = None if counts is None else counts[0:1]
    side_on = need_dx and dw_side_stream(len(blocks), stream_mode)
    main = torch.cuda.current_stream(index)
    side = fork = None
    if side_on:
        side, fork = _dw_stream(index)

    def base_values() -> dict[str, Any]:
        values: dict[str, Any] = {name: None for name in COMMON_TENSORS}
        values.update({name: 0 for name in COMMON_SCALARS})
        return values

    def launch(stage: str, values: dict[str, Any]) -> None:
        lau, arguments = _eager_launch(
            record, module_name, stage, values, device, geometry
        )
        with _ffi_stream_context(index):
            if lau.prepare is not None:
                lau.prepare(*arguments)
            lau.entry(*arguments)

    # every block's labels shifted into the block at once (one launch per op, not per block): the label minus the
    # block start inside the block, VM_FOREIGN_LABEL outside it, an ignored row's negative label unchanged; the last
    # block is at most vb_max wide and every valid label is below V, so ``shifted < vb_max`` is the block test
    starts = torch.arange(len(blocks), dtype=labels_c.dtype, device=device) * vb_max
    shifted = labels_c.unsqueeze(0) - starts.unsqueeze(1)
    lb_all = torch.where(
        labels_c.unsqueeze(0) < 0,
        labels_c.unsqueeze(0),
        torch.where(
            (shifted >= 0) & (shifted < vb_max),
            shifted,
            torch.full_like(shifted, VM_FOREIGN_LABEL),
        ),
    ).contiguous()
    del starts, shifted
    for b, (v0, v1) in enumerate(blocks):
        vb = v1 - v0
        if b and side_on:
            main.wait_stream(side)  # the block's dz buffer is about to be rewritten
        dz = dz_buf[: rows * vb].view(rows, vb)
        Wb = W[v0:v1]
        lb = lb_all[b]
        values = base_values()
        values.update(
            A=x_rows,
            B=Wb,
            C=dz,
            STATS_OUT=stats,
            WS=dummy,
            labels=lb,
            lse=lse,
            d_in=d_c,
            rows_c=rows,
            row0=0,
            d_off=0,
            T=rows,
            H=H,
            V=vb,
            loss_div=1.0,
            **gemm_scalars(
                dz_scalar_stage,
                geometry,
                hidden=H,
                vocab=vb,
                rows_c=rows,
                group_m=v.logits_group_m,
            ),
        )
        if temperature:
            values.update(temperature=tau_c)
        if parse_stage(dz_scalar_stage)[1]["dcnt"]:
            values.update(CNT=cnt)
        launch(dz_stage, values)
        del lb
        if side_on:
            fork.record(main)
        if need_dx:
            stage = dx_store if (b == 0 and not skip) else dx_add
            values = base_values()
            values.update(
                A=dz,
                B=Wb,
                C=dx_acc,
                rows_c=rows,
                T=rows,
                H=H,
                V=vb,
                loss_div=1.0,
                dx_split=int(v.dx_split or 0),
                **gemm_scalars(
                    stage, geometry, hidden=H, vocab=vb, rows_c=rows, k_slices=1
                ),
            )
            if base_stage(stage) != DX_RED_STAGE:
                values.update(STATS_OUT=stats, WS=dummy)
            else:
                # one K slice adds into dX_acc directly: the ordered form's slab operand is dX_acc itself
                values.update(
                    dx_ordered_operands(
                        dx_acc,
                        None,
                        k_slices=1,
                        m_tiles=values["m_tiles"],
                        ldc=values["ldc"],
                    )
                )
            if skip:
                values.update(CNT=cnt)
            launch(stage, values)
        values = base_values()
        values.update(
            A=dz,
            B=x_rows,
            C=dw_out[v0:v1],
            STATS_OUT=unit,
            WS=dummy,
            rows_c=rows,
            T=rows,
            H=H,
            V=vb,
            loss_div=1.0,
            **gemm_scalars(
                dw_stage,
                geometry,
                hidden=H,
                vocab=vb,
                rows_c=rows,
                group_m=v.dw_group_m,
                group_n=None,
                first_chunk=1,
            ),
        )
        if skip:
            values.update(CNT=cnt)
        if side_on:
            side.wait_event(fork)
            with torch.cuda.stream(side):
                launch(dw_stage, values)
        else:
            launch(dw_stage, values)
        del values, dz, Wb
    if side_on:
        main.wait_stream(side)
    # the block phase's buffers go before the dX output phase (the main stream has waited for the side stream's last
    # read of dz): that phase holds the dX accumulator, dW and the dX output only (memory_plan's second phase)
    del dz_buf, stats, lb_all, x_rows, d_c, labels_c, tau_c
    if dx_acc is None:
        return None, dw_out
    if row_index is None:
        return scale_cast(dx_acc, None, torch.bfloat16, backend=backend), dw_out
    if not fused_dx:
        return (
            scatter_rows(
                scale_cast(dx_acc, None, torch.bfloat16, backend=backend), row_index, T
            ),
            dw_out,
        )
    if scan is None:
        return (
            finalize_dx(dx_acc, None, row_index, labels >= 0, T, backend=backend),
            dw_out,
        )
    return (
        finalize_dx(dx_acc, None, row_index, None, T, backend=backend, scan=scan),
        dw_out,
    )


def forward_logprob_temperature(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    logprob_lbl: Optional[bool] = None,
    temperature=None,
    check_temperature: bool = False,
) -> ForwardResult:
    """:func:`forward_logprob` with a temperature (:func:`_forward_logprob` is the path without one).  Forward of
    the log-probability entry: ``logp`` (FP32 ``[T]``, 0 on ignored rows) plus the saved statistic
    ``lse`` (compacted: the ``[T_v]`` rows of ``row_index``, exactly what :func:`backward_logprob` expects back,
    together with ``row_index``).  ``compact_rows`` (default :func:`compact_rows_default`), ``hidden_count`` (default
    :func:`hidden_count_default`) and ``gemm_tuning`` (default :func:`gemm_tuning_default`) as in
    :func:`forward_loss`; ``logprob_lbl`` (default :func:`logprob_lbl_default`) writes only the row statistics and
    the label column's logit from the logits GEMM (bitwise the same ``lse`` / ``logp``).  This forward has no
    weight gradient, so it takes no ``fuse_dw_cast`` / ``dx_finalize`` / ``dw_stream``.

    ``temperature`` (``None`` / a positive number / a float32 ``[T]`` tensor; :func:`_resolve_temperature`, whose
    ``check`` is ``check_temperature``): ``logp_t = z_y / tau_t - logsumexp_v(z_v / tau_t)`` with ``z`` the BF16
    logits widened to FP32 and one FP32 division per element; ``lse`` is then the statistic of ``z / tau``.  A scalar
    whose FP32 value is 1.0 (and ``None``) runs the path without a temperature.  The result's ``temperature_rows`` is
    the compact FP32 temperature :func:`backward_logprob` takes back."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = validate_lm_head_inputs(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        entry="logprob",
    )
    # the validated problem without the temperature form (:attr:`ForwardResult.problem`)
    checked = problem
    tau = _resolve_temperature(
        temperature, problem.num_rows, X.device, check=check_temperature
    )
    _check_temperature_arms(tau is not None, logprob_lbl=logprob_lbl)
    if tau is not None:
        problem = _with_temperature(problem)
    tuning = _resolve_gemm_tuning(gemm_tuning)
    lbl = _resolve_logprob_lbl(logprob_lbl)
    T = problem.num_rows
    if T == 0:
        return _empty_forward(problem, X, need_dx=False, need_dw=False)
    compact = _resolve_compact(compact_rows)
    hc = (
        _hidden_count(
            problem,
            X,
            W,
            labels,
            backend=backend,
            compact=compact,
            stats=True,
            hidden_count=_resolve_hidden_count(hidden_count),
        )
        if tau is None
        else _hidden_count_temperature(
            problem,
            X,
            W,
            labels,
            backend=backend,
            compact=compact,
            hidden_count=_resolve_hidden_count(hidden_count),
            temperature=tau,
        )
    )
    if hc is not None:
        row_index = hc.row_index()
    else:
        row_index = valid_row_index(labels) if compact else None
    if row_index is not None and row_index.numel() == 0:  # every row ignored
        return _empty_forward(
            problem, X, need_dx=False, need_dw=False, row_index=row_index
        )
    valid_rows = None if row_index is None else int(row_index.numel())
    # the compact temperature the kernels read at ``row0 + r`` (the hidden count's [T] buffer, already filled on the
    # device: its first ``rows`` slots); the call without a temperature makes none
    tau_rows: Optional[torch.Tensor] = None
    if tau is not None:
        rows = T if valid_rows is None else valid_rows
        if hc is not None and hc.tau_c is not None:
            tau_rows = hc.tau_c[:rows]
        else:
            tau_rows = _temperature_rows(tau, row_index, rows, X.device)
    cache = BINDING_CACHE
    key = None
    if backend == "cake" and cache.enabled:
        key = forward_binding_key(
            X,
            W,
            labels,
            objective="ce",
            loss_div=None,
            infer_logp=None,
            loss_weights=None,
            chunk_size=chunk_size,
            need_dx=False,
            need_dw=False,
            grad_weight_dtype=torch.bfloat16,
            entry="logprob",
            valid_rows=valid_rows,
            gemm_tuning=tuning,
            hidden_count=hc is not None,
            logprob_lbl=lbl,
        )
        if tau is not None:
            key = temperature_binding_key(key)
        binding = cache.lookup(key)
        if binding is not None:
            logp, lse = (
                binding.forward(X, W, labels, None, None, row_index, hc)
                if tau is None
                else cast(_TemperatureBinding, binding).forward(
                    X, W, labels, None, None, row_index, hc, temperature=tau_rows
                )
            )
            result = ForwardResult(
                loss=None,
                logp=scatter_rows(logp, row_index, T),
                lse=lse,
                memory=binding.memory,
                backend=backend,
                row_index=row_index,
                num_rows=T,
            )
            result.temperature_rows = tau_rows
            result.problem = checked
            return result
    runner = prepare_lm_head_loss_temperature(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        need_dx=False,
        need_dw=False,
        entry="logprob",
        backend=backend,
        compact_rows=False if row_index is None else row_index,
        gemm_tuning=tuning,
        logprob_lbl=lbl,
        temperature=tau_rows,
    )
    if hc is not None:
        # the hidden valid-row count: chunk 0's logits already sit in the prologue's buffers, so the first call launches
        # through the templated binding too (the runner's own workspace is never written)
        binding = _TemperatureBinding.from_runner(runner, hc)
        del runner
        if key is not None:
            cache.remember(key, binding)
        logp, lse = (
            binding.forward(X, W, labels, None, None, row_index, hc)
            if tau is None
            else cast(_TemperatureBinding, binding).forward(
                X, W, labels, None, None, row_index, hc, temperature=tau_rows
            )
        )
        memory = binding.memory
    else:
        logp, lse = runner.forward()
        if key is not None:
            cache.remember(key, _TemperatureBinding.from_runner(runner))
        memory = runner.memory
    result = ForwardResult(
        loss=None,
        logp=scatter_rows(logp, row_index, T),
        lse=lse,
        memory=memory,
        backend=backend,
        row_index=row_index,
        num_rows=T,
    )
    result.temperature_rows = tau_rows
    result.problem = checked
    return result


def backward_logprob_temperature(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    dlogp: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    need_dx: bool = True,
    need_dw: bool = True,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    dw_stream: Optional[Union[bool, str]] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    logprob_dz: Optional[bool] = None,
    row_index: Any = _RECOUNT,
    temperature=None,
    check_temperature: bool = False,
    temperature_rows: Optional[torch.Tensor] = None,
    _problem: Optional[Problem] = None,
    _vm_any_chunks: bool = False,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """:func:`backward_logprob` with a temperature (:func:`_backward_logprob` is the path without one).  Backward
    of the log-probability entry: recompute each chunk's logits, ``dz = dlogp_t *
    (1[v = y_t] - softmax)`` (ignored rows zero), then ``dX`` (BF16) and ``dW`` (BF16) through
    the FP32 accumulators (the chunk loop; an eligible call that reuses the forward's row index runs the
    vocabulary-major form instead, :func:`_backward_logprob_vm`, with no FP32 ``dW`` accumulator).  ``(None, None)``
    when neither input is trainable.  ``compact_rows``
    must match the forward's: the valid-row index is the forward's ``row_index``
    (:attr:`ForwardResult.row_index`; ``None`` = uncompacted) when given -- no count, no host
    synchronization -- else recomputed from ``labels`` (identical to the forward's); ``lse`` is then
    the forward's compact ``[T_v]`` statistic and ``dX`` comes back scattered to ``[T, H]``; every
    row ignored returns zeros without launching.  ``fuse_dw_cast``
    (default :func:`fuse_dw_cast_default`): the last chunk's weight-gradient GEMM writes the final BF16
    ``dW`` from its epilogue (the scale is 1 here).  ``dx_finalize`` (default :func:`dx_finalize_default`,
    resolved once here): a compacted backward writes ``dX`` in one pass from the accumulator.  ``dw_stream``
    (default ``None`` = ``"auto"``, :func:`dw_stream_mode`), ``hidden_count`` (default
    :func:`hidden_count_default`; the recount path only) and ``gemm_tuning`` (default
    :func:`gemm_tuning_default`) as in :func:`forward_loss`.  ``logprob_dz`` (default
    :func:`logprob_dz_default`): each chunk's recompute writes ``dz`` from the logits GEMM's
    epilogue instead of storing the logits for ``row_grad`` (bitwise the same ``dX`` / ``dW``).

    Temperature (the forward's): ``temperature_rows`` -- the forward's compact FP32 ``[rows]`` temperature
    (:attr:`ForwardResult.temperature_rows`, the rows of ``row_index``; the autograd Function passes it) -- or
    ``temperature`` / ``check_temperature`` as given to :func:`forward_logprob` (validated again and gathered by the
    valid-row index; a recount with a temperature takes the host count).  ``dz = (dlogp_t / tau_t) * (1[v = y_t] -
    softmax(z_t / tau_t))`` with one FP32 division for ``dlogp_t / tau_t`` per row and one per logit.

    Zero-gradient row skip (:func:`backward_logprob`'s: a compacted backward that reuses the forward's row index, the
    same eligibility and floor, run when the call's plan has the counted forms): the chunk loop takes the live rows
    first (:func:`zero_dlogp_partition`) and the temperature rows follow that order -- a per-token temperature is
    gathered from the caller's ``[T]`` tensor by the reordered row index (one gather, as the forward's), a scalar's
    rows are the same in any order.  The ``dz`` recompute runs its counted temperature form (no work on the rows past
    the chunk's count; the uncounted form over every row where the record lacks it or
    :func:`set_zero_dlogp_temperature_dz_counted` turns it off: a zero-gradient row's ``dz`` is ``RN(0 / tau) * (...)
    = 0``), the K-sliced dX reductions and the weight-gradient GEMMs their counted forms
    (:func:`make_plan_temperature`).  ``logp`` and ``dX`` of a live row are
    the same as without the skip; ``dW`` sums the live rows in the new row order (FP32), so its bits may differ when a
    row is zero.  The skip needs the temperature itself: the call's ``temperature``, or with ``temperature_rows`` the
    forward's resolved temperature (the autograd Function passes both); ``temperature_rows`` alone runs the backward
    without the skip.  ``_problem`` (internal) as in :func:`backward_logprob`: the forward's validated problem without
    the temperature form.  ``_vm_any_chunks`` (internal; the opt-in budget form's pick of the vocabulary-major
    backward): the form at any chunk count (:func:`_logprob_vm_eligible`'s ``any_chunks``); a call that is not
    eligible raises (the budget form's peak needs it)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = (
        _problem
        if _problem is not None
        else validate_lm_head_inputs(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            entry="logprob",
        )
    )
    if (
        temperature_rows is not None
        and temperature is not None
        and not isinstance(temperature, _Temperature)
    ):
        raise ValueError(
            "pass the forward's temperature_rows or the call's temperature, not both"
        )
    # beside ``temperature_rows`` a temperature is the forward's resolved one (returned as is): the source of the
    # zero-gradient row skip's temperature rows
    tau = _resolve_temperature(
        temperature, problem.num_rows, X.device, check=check_temperature
    )
    has_tau = temperature_rows is not None or tau is not None
    _check_temperature_arms(has_tau, logprob_dz=logprob_dz)
    if not has_tau:
        # no temperature (a scalar whose FP32 value is 1.0): the backward without one, the zero-gradient row skip
        # included, so the call is bitwise the call without the keyword
        return backward_logprob(
            X,
            W,
            labels,
            lse,
            dlogp,
            chunk_size=chunk_size,
            need_dx=need_dx,
            need_dw=need_dw,
            backend=backend,
            compact_rows=compact_rows,
            fuse_dw_cast=fuse_dw_cast,
            dx_finalize=dx_finalize,
            dw_stream=dw_stream,
            hidden_count=hidden_count,
            gemm_tuning=gemm_tuning,
            logprob_dz=logprob_dz,
            row_index=row_index,
            _problem=problem,
        )
    problem = _with_temperature(problem)
    stream_mode = dw_stream_mode(dw_stream)
    tuning = _resolve_gemm_tuning(gemm_tuning)
    dz_arm = _resolve_logprob_dz(logprob_dz)
    # read once: the binding key and the plan take the same settings
    rule = dx_slice_rule()
    if not (need_dx or need_dw):
        return None, None
    T, H, V = problem.num_rows, problem.hidden, problem.vocab
    fused_dx = bool(
        need_dx and _resolve_dx_finalize(dx_finalize)
    )  # resolved once per call
    compact = bool(T and _resolve_compact(compact_rows))
    hc = None
    # the zero-gradient row skip: eligible for a call that reuses the forward's row index and has the temperature
    # itself (the source of the reordered temperature rows), run when the call's plan has its counted forms
    # (``order_rows``)
    skip = False
    scan = counts = None
    # the fused dX finalize's valid-row mask (labels >= 0) of a call that reuses the forward's row index: one device
    # comparison, no synchronization, formed in finish_dx behind the queued GEMMs (the call's first launches are not
    # held back by it)
    labels_mask = False
    reuse_index = row_index is not _RECOUNT
    if reuse_index:
        # the forward's valid-row index: the plan is known without a count
        if row_index is not None and not compact:
            raise ValueError("a valid-row index belongs to a compacted backward")
        skip = tau is not None and _zero_dlogp_skip(
            X.device, row_index, backend, dz_arm, need_dx, fused_dx
        )
        row_valid = None
        labels_mask = row_index is not None and fused_dx and not skip
    else:
        # a recount with a temperature takes the host count: the device-count recompute stores ``z`` for ``row_grad``,
        # which has no temperature form
        hc = (
            None
            if has_tau
            else _hidden_count(
                problem,
                X,
                W,
                labels,
                backend=backend,
                compact=compact,
                stats=False,
                hidden_count=_resolve_hidden_count(hidden_count),
            )
        )
        if hc is not None:
            # the hidden valid-row count: chunk 0's gather and its recompute GEMM (no statistics) are queued
            row_index = hc.row_index()
            row_valid = hc.valid if (row_index is not None and fused_dx) else None
        else:
            row_index, row_valid = (
                valid_rows(labels, mask=fused_dx) if compact else (None, None)
            )
    rows = T if row_index is None else int(row_index.numel())
    if rows == 0:  # no rows, or every row ignored
        return (
            torch.zeros((T, H), dtype=torch.bfloat16, device=X.device)
            if need_dx
            else None,
            torch.zeros((V, H), dtype=torch.bfloat16, device=X.device)
            if need_dw
            else None,
        )
    _check_output(lse, "lse", (rows,), torch.float32)
    if temperature_rows is not None:
        # the forward's compact temperature (checked like ``lse``: the same rows)
        _check_output(temperature_rows, "temperature_rows", (rows,), torch.float32)
        if temperature_rows.device != X.device:
            raise ValueError("temperature_rows must be on X's device")
    forward_index = row_index
    # the compact temperature of the chunk loop's rows, in the order of the call's row index (``order_rows``)
    tau_rows: Optional[torch.Tensor] = None

    def tau_rows_of(index: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        # the forward's rows serve the forward's order and, for a scalar, any order; else the gather by ``index``
        if temperature_rows is not None and (
            index is forward_index or (tau is not None and tau.kind == "scalar")
        ):
            return temperature_rows
        return _temperature_rows(tau, index, rows, X.device)

    dlogp = dlogp.detach().reshape(T).to(torch.float32).contiguous()
    cache = BINDING_CACHE
    key = None
    valid_count = None if row_index is None else rows
    fuse = bool(need_dw and _resolve_fuse(fuse_dw_cast))
    # the fused dX finalize of a compacted backward: the runner leaves dx_acc uncast and finalize_dx writes the [T, H]
    # dX in one pass (uncompacted: the in-loop flat cast is already one pass over every row)
    dx_cast = not (fused_dx and row_index is not None)

    def finish_dx(dx):
        if dx is None:
            return None
        if dx_cast:
            return scatter_rows(dx, row_index, T)
        valid = (labels >= 0) if labels_mask else row_valid
        return finalize_dx(dx, None, row_index, valid, T, backend=backend, scan=scan)

    vm = reuse_index and _logprob_vm_eligible(
        problem,
        X,
        rows,
        backend=backend,
        need_dw=need_dw,
        dz_arm=dz_arm,
        tuning=tuning,
        temperature=True,
        any_chunks=_vm_any_chunks,
    )
    if _vm_any_chunks and not vm:
        raise RuntimeError(
            "internal: the budget form picked the vocabulary-major backward for a call that is not eligible"
        )
    if vm:
        return _backward_logprob_vm(
            problem,
            X,
            W,
            labels,
            lse,
            dlogp,
            row_index,
            need_dx=need_dx,
            fused_dx=fused_dx,
            skip=skip,
            stream_mode=stream_mode,
            backend=backend,
            tau_rows_of=tau_rows_of,
            counted_tau_dz=zero_dlogp_temperature_dz_counted(),
        )

    def order_rows(plan: Plan) -> None:
        # :func:`backward_logprob`'s: the plan decides the skip (its counted forms order the rows, one partition
        # launch, the live counts per chunk of the plan's stride; a plan without them runs over the forward's row
        # order); the temperature rows follow the order
        nonlocal skip, row_index, lse, scan, counts, labels_mask, tau_rows
        skip = skip and plan.dcnt
        if skip:
            row_index, lse, scan, counts = zero_dlogp_partition(
                labels, dlogp, row_index, lse, chunk_stride(plan.chunks, problem.chunk)
            )
            tau_rows = tau_rows_of(row_index)
            return
        if row_valid is None and row_index is not None and fused_dx:
            labels_mask = True
        if tau_rows is None:
            tau_rows = tau_rows_of(row_index)

    if backend == "cake" and cache.enabled:
        key = logprob_backward_binding_key(
            X,
            W,
            labels,
            lse,
            dlogp,
            chunk_size=chunk_size,
            need_dx=need_dx,
            need_dw=need_dw,
            valid_rows=valid_count,
            fuse_dw_cast=fuse,
            gemm_tuning=tuning,
            dx_finalize=fused_dx,
            hidden_count=hc is not None,
            logprob_dz=dz_arm,
            dx_rule=rule,
            zero_dlogp=skip,
        )
        # an eligible call's key also holds which temperature form of the dz arm its plan may take
        key = temperature_binding_key(key) + (
            (zero_dlogp_temperature_dz_counted(),) if skip else ()
        )
        binding = cache.lookup(key)
        if binding is not None:
            _alert_unordered_dx(binding.plan, "backward_logprob_temperature", "logprob")
            # the key of an eligible call remembers the plan it took (counted, or the shipped one)
            order_rows(binding.plan)
            dx, dw = cast(_TemperatureBinding, binding).backward_logprob(
                X,
                W,
                labels,
                lse,
                dlogp,
                row_index,
                hc,
                stream_mode,
                temperature=tau_rows,
                dcnt=counts,
            )
            return finish_dx(dx), dw
    # the runner binds the forward's row order; an eligible call plans with the counted forms, the counts and the
    # reordered rows follow the plan (the call launches from its binding with the rows it ordered)
    tau_rows = tau_rows_of(row_index)
    runner = prepare_lm_head_loss_temperature(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        need_dx=need_dx,
        need_dw=need_dw,
        entry="logprob",
        dlogp=dlogp,
        lse=lse,
        backend=backend,
        compact_rows=False if row_index is None else row_index,
        fuse_dw_cast=fuse,
        gemm_tuning=tuning,
        dx_finalize=fused_dx,
        dx_cast=dx_cast,
        dw_stream=stream_mode,
        logprob_dz=dz_arm,
        dx_rule=rule,
        temperature=tau_rows,
        dcnt=torch.empty(
            (len(loop_chunks(T, rows, problem.chunk)),),
            dtype=torch.int32,
            device=X.device,
        )
        if skip
        else None,
        # the skip's private forms (the cast at the last live chunk): the counts come from zero_dlogp_partition (zeros
        # form a suffix) and the gradient buffers are this call's own
        _skip_forms=skip,
    )
    _alert_unordered_dx(runner.plan, "backward_logprob_temperature", "logprob")
    order_rows(runner.plan)
    if hc is not None or skip:
        binding = cast(_TemperatureBinding, _TemperatureBinding.from_runner(runner, hc))
        del runner
        if key is not None:
            cache.remember(key, binding)
        dx, dw = binding.backward_logprob(
            X,
            W,
            labels,
            lse,
            dlogp,
            row_index,
            hc,
            stream_mode,
            temperature=tau_rows,
            dcnt=counts,
        )
        return finish_dx(dx), dw
    dx, dw = runner._backward()
    if key is not None:
        cache.remember(key, _TemperatureBinding.from_runner(runner))
    return finish_dx(dx), dw


def forward_logprob_keep(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
) -> KeptForwardResult:
    """:func:`forward_logprob` that keeps every chunk's BF16 logits for the backward (``cake`` backend): the logits
    GEMM is the plain ``gemm_logits`` (the whole ``z_c``, not the label-column arm), chunk 0's logits stay in the
    chunk buffer and the later chunks' go to one ``[T_v - rows_0, V]`` buffer, ``2 * T_v * V`` bytes in all
    (:attr:`KeptForwardResult.logits`); under the hidden valid-row count with the one-pass form every chunk's logits
    are rows of the prologue's ``[T, V]`` buffer, ``2 * T * V`` bytes kept until the backward
    (:attr:`KeptForwardResult.memory`'s ``saved_logits_bytes``, :func:`_hidden_kept_memory_report`).  Pass them to
    :func:`backward_logprob_keep`, which forms ``dz`` from them instead of recomputing the logits.  Bitwise the same
    ``logp`` / ``lse`` as :func:`forward_logprob` (the gradients: :func:`backward_logprob_keep`).

    The kept-logits form (:func:`set_kept_logits_form`): ``hidden_count`` ``None`` takes the form's default (``True``:
    the hidden count where it is eligible, three chunks or more; ``False``: the valid rows are counted before the
    first launch); without the hidden count the views are consecutive row
    slices of one contiguous ``[T_v, V]`` buffer (:func:`kept_rows_of`) and, with ``tail_fold``, the plan folds a last
    chunk of fewer than ``chunk_size / 2`` rows into the one before it where :func:`loop_chunks` folds (a call of fewer
    than :data:`HIDDEN_COUNT_MIN_CHUNKS` chunks of ``T``)."""
    if backend != "cake":
        raise ValueError("the kept logits (logprob_keep) run on the cake backend")
    problem = validate_lm_head_inputs(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        entry="logprob",
    )
    tuning = _resolve_gemm_tuning(gemm_tuning)
    T = problem.num_rows
    if T == 0:
        return _kept_result(
            _empty_forward(problem, X, need_dx=False, need_dw=False), None
        )
    compact = _resolve_compact(compact_rows)
    hc = _hidden_count(
        problem,
        X,
        W,
        labels,
        backend=backend,
        compact=compact,
        stats=True,
        hidden_count=(
            bool(_KEPT_LOGITS_FORM["hidden_count"])
            if hidden_count is None
            else _resolve_hidden_count(hidden_count)
        ),
        # the contiguous kept buffer of the one-pass weight gradient: T rows (the count is not known yet)
        kept_rows=bool(_KEPT_LOGITS_FORM["one_pass"]),
    )
    if hc is not None:
        row_index = hc.row_index()
    else:
        row_index = valid_row_index(labels) if compact else None
    if row_index is not None and row_index.numel() == 0:  # every row ignored
        return _kept_result(
            _empty_forward(
                problem, X, need_dx=False, need_dw=False, row_index=row_index
            ),
            None,
        )
    # the short-tail fold (a plan of the hidden count keeps the shipped chunks: its prologue buffers hold min(C, T) rows)
    fold = bool(_KEPT_LOGITS_FORM["tail_fold"] and hc is None)
    cache = BINDING_CACHE
    key = None
    binding: Optional[_KeptBinding] = None
    if cache.enabled:
        key = (
            forward_binding_key(
                X,
                W,
                labels,
                objective="ce",
                loss_div=None,
                infer_logp=None,
                loss_weights=None,
                chunk_size=chunk_size,
                need_dx=False,
                need_dw=False,
                grad_weight_dtype=torch.bfloat16,
                entry="logprob",
                valid_rows=None if row_index is None else int(row_index.numel()),
                gemm_tuning=tuning,
                hidden_count=hc is not None,
                logprob_lbl=False,
            )
            + (("fold",) if fold else ())
            + ("keep",)
        )
        found = cache.lookup(key)  # a kept key binds a kept binding only
        if isinstance(found, _KeptBinding):
            binding = found
    if binding is None:
        runner = prepare_lm_head_loss(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            need_dx=False,
            need_dw=False,
            entry="logprob",
            backend=backend,
            compact_rows=False if row_index is None else row_index,
            gemm_tuning=tuning,
            logprob_lbl=False,
            logprob_keep=True,
            kept_fold=fold,
        )
        # the kept views are bound per call: every call launches through the templated binding
        binding = cast(_KeptBinding, _KeptBinding.from_runner(runner, hc))
        del runner
        if key is not None:
            cache.remember(key, binding)
    logp, lse, kept = binding.forward_keep(X, W, labels, row_index, hc)
    memory = binding.memory
    if hc is not None and hc.kept is not None:
        # the kept views are rows of the prologue's [T, V] buffer: the report counts all of it
        memory = _hidden_kept_memory_report(memory, num_rows=T, vocab=problem.vocab)
    return KeptForwardResult(
        loss=None,
        logp=scatter_rows(logp, row_index, T),
        lse=lse,
        memory=memory,
        backend=backend,
        row_index=row_index,
        num_rows=T,
        logits=kept,
    )


def backward_logprob_keep(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    dlogp: torch.Tensor,
    logits: tuple,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    need_dx: bool = True,
    need_dw: bool = True,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    dw_stream: Optional[Union[bool, str]] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    row_index: Any = _RECOUNT,
    dz_ready: bool = False,
    dx_side: bool = False,
    after_dz: Optional[Callable[[], None]] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """:func:`backward_logprob` over the kept logits ``logits`` of :func:`forward_logprob_keep`
    (:attr:`KeptForwardResult.logits`, ``cake`` backend): each chunk's ``dz`` is formed by ``row_grad`` over its kept
    view in place -- no logits GEMM; the chunk's ``X`` rows are gathered for the weight gradient only -- then ``dX`` /
    ``dW`` as in its chunk loop (a chunk of more than one dX slice sums its slices in slice order).  ``dW`` / ``dX``
    can round differently from :func:`backward_logprob`'s (another summation order: its vocabulary-major form of the
    auto rule's calls, its zero-gradient row skip), within the same accuracy budget.
    The views serve one backward: this call marks them consumed and a second
    backward on them is a ``ValueError``.  ``row_index`` as in :func:`backward_logprob`
    (the forward's valid-row index, or a recount); the hidden valid-row count and ``logprob_dz`` do not apply.
    The backward's chunks are the views': ``chunk_size``'s chunks (:func:`plan_chunks`) or, when the views have those
    rows, its folded chunks (:func:`loop_chunks`).  One view of all ``T_v`` rows with ``chunk_size >= T_v`` (the
    forward's contiguous kept buffer, :func:`kept_rows_of`) runs the backward as one chunk: one ``row_grad``, one dX
    GEMM over ``M = T_v`` and the fused weight-gradient cast over ``K = T_v`` (no FP32 ``dW`` accumulator); its
    ``dW`` sums the rows in one GEMM, so its bits differ from the chunked backward's (each repeats bit for bit).
    ``dz_ready`` / ``dx_side`` / ``after_dz``: the parts of the ordered one-pass backward
    (:func:`backward_logprob_keep_ordered`, :meth:`_KeptBinding.backward_logprob_keep`)."""
    if backend != "cake":
        raise ValueError("the kept logits (logprob_keep) run on the cake backend")
    problem = validate_lm_head_inputs(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        entry="logprob",
    )
    stream_mode = dw_stream_mode(dw_stream)
    tuning = _resolve_gemm_tuning(gemm_tuning)
    # read once: the binding key and the plan take the same settings
    rule = dx_slice_rule()
    if not (need_dx or need_dw):
        return None, None
    T, H, V = problem.num_rows, problem.hidden, problem.vocab
    fused_dx = bool(need_dx and _resolve_dx_finalize(dx_finalize))
    compact = bool(T and _resolve_compact(compact_rows))
    if row_index is not _RECOUNT:
        if row_index is not None and not compact:
            raise ValueError("a valid-row index belongs to a compacted backward")
        row_valid = (labels >= 0) if (row_index is not None and fused_dx) else None
    else:
        row_index, row_valid = (
            valid_rows(labels, mask=fused_dx) if compact else (None, None)
        )
    rows = T if row_index is None else int(row_index.numel())
    if rows == 0:  # no rows, or every row ignored
        return (
            torch.zeros((T, H), dtype=torch.bfloat16, device=X.device)
            if need_dx
            else None,
            torch.zeros((V, H), dtype=torch.bfloat16, device=X.device)
            if need_dw
            else None,
        )
    _check_output(lse, "lse", (rows,), torch.float32)
    dlogp = _dlogp_rows(dlogp, T)
    fuse = bool(need_dw and _resolve_fuse(fuse_dw_cast))
    dx_cast = not (fused_dx and row_index is not None)
    # the forward's chunking: its folded short tail (:func:`loop_chunks`) when the views have those rows
    view_rows = tuple(
        int(v.shape[0]) if isinstance(v, torch.Tensor) and v.dim() >= 1 else -1
        for v in tuple(logits or ())
    )
    fold = view_rows == tuple(
        rows_c for _, rows_c in loop_chunks(T, rows, problem.chunk)
    ) and view_rows != tuple(rows_c for _, rows_c in plan_chunks(rows, problem.chunk))
    cache = BINDING_CACHE
    key = None
    binding: Optional[_KeptBinding] = None
    if cache.enabled:
        key = (
            logprob_backward_binding_key(
                X,
                W,
                labels,
                lse,
                dlogp,
                chunk_size=chunk_size,
                need_dx=need_dx,
                need_dw=need_dw,
                valid_rows=None if row_index is None else rows,
                fuse_dw_cast=fuse,
                gemm_tuning=tuning,
                dx_finalize=fused_dx,
                hidden_count=False,
                logprob_dz=False,
                dx_rule=rule,
            )
            + (("fold",) if fold else ())
            + ("keep",)
        )
        found = cache.lookup(key)  # a kept key binds a kept binding only
        if isinstance(found, _KeptBinding):
            binding = found
    if binding is None:
        runner = prepare_lm_head_loss(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            need_dx=need_dx,
            need_dw=need_dw,
            entry="logprob",
            dlogp=dlogp,
            lse=lse,
            backend=backend,
            compact_rows=False if row_index is None else row_index,
            fuse_dw_cast=fuse,
            gemm_tuning=tuning,
            dx_finalize=fused_dx,
            dx_cast=dx_cast,
            dw_stream=stream_mode,
            logprob_dz=False,
            dx_rule=rule,
            logprob_keep=True,
            kept_fold=fold,
        )
        binding = cast(_KeptBinding, _KeptBinding.from_runner(runner))
        del runner
        if key is not None:
            cache.remember(key, binding)
    views = check_kept_logits(binding.plan, logits)
    _alert_unordered_dx(binding.plan, "backward_logprob_keep", "logprob")
    for view in views:  # consumed from here on: row_grad overwrites them with dz
        setattr(view, _KEPT_SPENT, True)
    dx, dw = binding.backward_logprob_keep(
        X,
        W,
        labels,
        lse,
        dlogp,
        views,
        row_index,
        stream_mode,
        dz_ready=dz_ready,
        dx_side=dx_side,
        after_dz=after_dz,
    )
    if dx is not None:
        dx = (
            scatter_rows(dx, row_index, T)
            if dx_cast
            else finalize_dx(dx, None, row_index, row_valid, T, backend=backend)
        )
    return dx, dw


# ---------------------------------------------------------------------------
# Autograd Functions and the backend-level entry points
# ---------------------------------------------------------------------------

# Gradient placement of the loss entry (:func:`set_gradient_placement`).  ``"auto"`` (the default): the loss entry forms
# its gradient accumulators only where a backward can use them -- no gradient GEMM without an autograd graph (grad mode
# off, ``torch.no_grad`` / ``torch.inference_mode``, the first pass of a reentrant ``torch.utils.checkpoint``, no input
# requiring a gradient), and inside a non-reentrant ``torch.utils.checkpoint`` (selective activation checkpointing
# included) the checkpointed forward computes the loss only while the recomputation in the backward forms the
# accumulators (one gradient pass per step instead of two).  CPU offload or other saved-tensor hooks innermost, torch's
# deterministic algorithms (the alert of a K-sliced dX stays in the forward) and a torch whose checkpoint lacks the hook
# structure read below keep the fused forward.  A reentrant checkpoint's first pass runs with grad mode off: like a
# ``torch.no_grad`` call it computes the loss only and raises no deterministic-algorithms alert; its recomputation, in
# the backward, alerts as any call does.  The log-probability entry forms no gradient in its forward; under ``"auto"``
# its checkpoint recomputation takes the checkpointed call's row results (:class:`_LogprobKept`).
# ``"forward"``: every call with an input that requires a gradient forms them in its forward, also when no backward
# follows.  The outputs are the same bits either way (the same kernels on the same operands, in the same order).
GRADIENT_PLACEMENTS = ("auto", "forward")
_GRADIENT_PLACEMENT = {"mode": "auto"}


def gradient_placement() -> str:
    """The gradient placement of the loss entry (default ``"auto"``, see :func:`set_gradient_placement`)."""
    return _GRADIENT_PLACEMENT["mode"]


def set_gradient_placement(mode: str) -> str:
    """Set the gradient placement of the following loss entry calls of this process and return the previous one:
    ``"auto"`` (the default) forms the loss entry's gradients only where a backward can use them and, inside a
    non-reentrant ``torch.utils.checkpoint``, once per step (in the recomputation); ``"forward"`` forms them in every
    forward with an input that requires a gradient.  Read once per call; a checkpointed region's recomputation follows
    the placement its forward ran under."""
    if mode not in GRADIENT_PLACEMENTS:
        raise ValueError(
            f"gradient placement must be one of {GRADIENT_PLACEMENTS}, got {mode!r}"
        )
    previous = _GRADIENT_PLACEMENT["mode"]
    _GRADIENT_PLACEMENT["mode"] = mode
    return previous


# torch.utils.checkpoint (non-reentrant) packs the tensors an autograd Function saves through its own saved-tensor hooks:
# the checkpointed forward through ``_checkpoint_hook``'s pack hook (a closure over the call's ``_CheckpointFrame``), the
# recomputation in the backward through ``_recomputation_hook``'s (a closure over a weak reference to that frame and the
# graph key).  The innermost (top) hooks tell which pass a call runs in; any other hooks on top (CPU offload, ...) keep
# the call's fused forward.  A torch whose checkpoint lacks this structure keeps the fused forward everywhere.
#
# Selective activation checkpointing (``context_fn`` of ``create_selective_checkpoint_contexts``) also records the
# checkpointed forward's operators through a torch dispatch mode and replays them by position in the recomputation,
# which must therefore run the same operators.  An entry's operators are its own -- its buffers, bindings (keyed by the
# inputs' addresses, which a recomputation can change) and per-thread state, and the loss-only or fused forward each
# pass takes -- so while a selective checkpoint's mode is active every autograd Function of this package runs its
# forward with ALL active torch dispatch modes popped (:func:`_modes_off`): the selective checkpoint's, and any other on
# the stack (a user mode such as ``FlopCounterMode`` then does not see the entry's operators; a tracing mode such as
# ``FakeTensorMode`` or a ``torch.compile`` trace then gets real launches on fake inputs and fails loudly).  The call
# then behaves as in a plain non-reentrant checkpoint.
_DISPATCH_MODES = getattr(torch._C, "_len_torch_dispatch_stack", None)
_SELECTIVE_MODES = tuple(
    mode
    for mode in (
        getattr(_torch_checkpoint, "_CachingTorchDispatchMode", None),
        getattr(_torch_checkpoint, "_CachedTorchDispatchMode", None),
    )
    if isinstance(mode, type)
)
_MODE_STACK = getattr(_torch_dispatch, "_get_current_dispatch_mode_stack", None)


def _selective_checkpoint_modes() -> bool:
    """Whether a selective activation checkpoint's dispatch mode is active (read by every autograd Function's forward of
    this package under saved-tensor hooks; the forward then runs again with every active dispatch mode popped).  False
    on a torch without the private APIs read here (the entries then behave as before)."""
    if (
        not _SELECTIVE_MODES
        or _MODE_STACK is None
        or _DISPATCH_MODES is None
        or _DISABLE_MODES is None
    ):
        return False
    return _DISPATCH_MODES() > 0 and any(
        isinstance(mode, _SELECTIVE_MODES) for mode in _MODE_STACK()
    )


_CHECKPOINT_FIRST = "_checkpoint_hook.__init__.<locals>.pack_hook"
_CHECKPOINT_RECOMPUTE = "_recomputation_hook.__init__.<locals>.pack_hook"
_CHECKPOINT_SUPPORT: dict[str, bool] = {}
_TOP_SAVED_HOOKS_C = getattr(
    torch._C._autograd, "_top_saved_tensors_default_hooks", None
)
# the loss entry's forward reads it once per call with an input requiring a gradient (None: no saved-tensor hooks)
_TOP_SAVED_HOOKS = (
    _TOP_SAVED_HOOKS_C
    if _TOP_SAVED_HOOKS_C is not None
    else (lambda ignore_is_tracing: None)
)


def _saved_problem(result: ForwardResult) -> Optional[Problem]:
    """The forward's validated problem for its autograd backward (:attr:`ForwardResult.problem`), or ``None`` when
    saved-tensor hooks are active (``torch.utils.checkpoint``, ``save_on_cpu``, offloading): those may unpack an ``X``
    of another stride or alignment, so the backward validates what it gets.  Without hooks the backward gets back the
    tensors the forward validated."""
    return result.problem if _TOP_SAVED_HOOKS(False) is None else None


def _nested_code(cls, name: str) -> Optional[types.CodeType]:
    code = getattr(getattr(cls, "__init__", None), "__code__", None)
    for const in getattr(code, "co_consts", ()):
        if isinstance(const, types.CodeType) and const.co_name == name:
            return const
    return None


def _checkpoint_supported() -> bool:
    """Whether this torch's ``torch.utils.checkpoint`` has the hook structure :func:`_checkpoint_pass` reads (checked
    once per process)."""
    ok = _CHECKPOINT_SUPPORT.get("ok")
    if ok is None:
        frame = getattr(_torch_checkpoint, "_CheckpointFrame", None)
        first = _nested_code(
            getattr(_torch_checkpoint, "_checkpoint_hook", None), "pack_hook"
        )
        again = _nested_code(
            getattr(_torch_checkpoint, "_recomputation_hook", None), "pack_hook"
        )
        ok = bool(
            _TOP_SAVED_HOOKS_C is not None
            and isinstance(frame, type)
            and "__weakref__" in vars(frame)
            and first is not None
            and "frame" in first.co_freevars
            and again is not None
            and {"target_frame_ref", "gid"} <= set(again.co_freevars)
        )
        _CHECKPOINT_SUPPORT["ok"] = ok
    return ok


class _CheckpointPass(NamedTuple):
    first: bool  # the checkpointed forward (False: its recomputation in a backward)
    frame: Any  # the call's torch.utils.checkpoint frame (None: not readable)
    graph: Any = None  # the recomputation's graph key


def _checkpoint_pass() -> Optional[_CheckpointPass]:
    """The non-reentrant ``torch.utils.checkpoint`` pass this call runs in, read from the top saved-tensor default
    hooks; ``None`` outside one (no hooks, other hooks on top, or an unsupported torch)."""
    hooks = _TOP_SAVED_HOOKS(False)
    if hooks is None or not _checkpoint_supported():
        return None
    # the recomputation's pack hook is wrapped by torch._dynamo.disable
    pack = inspect.unwrap(hooks[0])
    if getattr(pack, "__module__", None) != _torch_checkpoint.__name__:
        return None
    name = getattr(pack, "__qualname__", None)
    if name not in (_CHECKPOINT_FIRST, _CHECKPOINT_RECOMPUTE):
        return None
    cells = dict(zip(pack.__code__.co_freevars, pack.__closure__ or (), strict=False))
    frame_type = _torch_checkpoint._CheckpointFrame
    try:
        if name == _CHECKPOINT_FIRST:
            frame = cells["frame"].cell_contents
            return _CheckpointPass(
                True, frame if isinstance(frame, frame_type) else None
            )
        ref, graph = cells["target_frame_ref"].cell_contents, cells["gid"].cell_contents
    except (KeyError, ValueError):
        return _CheckpointPass(name == _CHECKPOINT_FIRST, None)
    frame = ref() if isinstance(ref, weakref.ReferenceType) else None
    return _CheckpointPass(
        False, frame if isinstance(frame, frame_type) else None, graph
    )


def _graph_key() -> Any:
    """The key ``torch.utils.checkpoint`` files a backward's recomputation under (its unpack hook's rule)."""
    group = getattr(
        getattr(_torch_checkpoint, "GraphExecGroup", None), "_get_current_group", None
    )
    key = group() if group is not None else None
    return torch._C._current_graph_task_id() if key is None else key


class _CheckpointCalls:
    """The loss entry calls of one checkpoint frame: per call of the checkpointed forward (in order) its key and whether
    it left its gradients to the recomputation; per graph the recomputation's call count; the state each recomputed
    call hands to its checkpointed call's backward (the fused forward's result, taken once)."""

    __slots__ = ("calls", "recomputed", "handed", "__weakref__")

    def __init__(self) -> None:
        self.calls: list = []
        self.recomputed: dict = {}
        self.handed: dict = {}


_CHECKPOINT_CALLS: "weakref.WeakKeyDictionary[Any, _CheckpointCalls]" = (
    weakref.WeakKeyDictionary()
)
_CHECKPOINT_LOCK = threading.Lock()


def _checkpoint_first_call(ckpt: _CheckpointPass, key: tuple, deferred: bool):
    """Record a call of the checkpointed forward; ``(calls, index)`` (``None``: the frame is not readable)."""
    if ckpt.frame is None:
        return None
    with _CHECKPOINT_LOCK:
        calls = _CHECKPOINT_CALLS.get(ckpt.frame)
        if calls is None:
            calls = _CHECKPOINT_CALLS[ckpt.frame] = _CheckpointCalls()
        calls.calls.append((key, deferred))
        return calls, len(calls.calls) - 1


def _checkpoint_recomputed_call(ckpt: _CheckpointPass):
    """The checkpointed call a call of the recomputation repeats: ``(calls, index, key, deferred)`` (``None``: unknown)."""
    if ckpt.frame is None:
        return None
    with _CHECKPOINT_LOCK:
        calls = _CHECKPOINT_CALLS.get(ckpt.frame)
        if calls is None:
            return None
        index = calls.recomputed.get(ckpt.graph, 0)
        calls.recomputed[ckpt.graph] = index + 1
        if index >= len(calls.calls):
            return None
        key, deferred = calls.calls[index]
        return calls, index, key, deferred


def _meta_key(t: Optional[torch.Tensor]):
    return None if t is None else (tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


def _detached(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return t.detach() if t is not None and t.requires_grad else t


class _SplitCall(NamedTuple):
    """A loss entry call whose gradients the checkpoint recomputation forms (gradient placement ``"auto"``)."""

    # the checkpointed call's frame record (None: not readable) and its position in the frame's calls
    calls: Optional[_CheckpointCalls]
    position: Optional[int]
    result: Optional[ForwardResult]  # the recomputed call: its fused forward's result


def _loss_call_key(
    X, W, labels, infer_logp, loss_weights, need_dx, need_dw, options
) -> tuple:
    """What a recomputed call must match to take over its checkpointed call's gradients: the inputs' metadata and the
    call's options (not their storage: hooks around the checkpoint, CPU offload for one, hand the recomputation copies)."""
    return (
        _meta_key(X),
        _meta_key(W),
        _meta_key(labels),
        _meta_key(infer_logp),
        _meta_key(loss_weights),
        need_dx,
        need_dw,
        repr(sorted(options.items())),
    )


def _checkpoint_defers() -> bool:
    """Whether a checkpointed call leaves its gradients to the recomputation: placement ``"auto"``, outside torch's
    deterministic algorithms (the alert of a K-sliced dX stays raised in the forward, before its dX launches).  Under
    selective activation checkpointing the call runs with every dispatch mode popped
    (:func:`_selective_checkpoint_modes`) and the same rule applies."""
    return (
        gradient_placement() == "auto"
        and not torch.are_deterministic_algorithms_enabled()
    )


def _loss_checkpoint_split(
    ctx,
    ckpt: _CheckpointPass,
    X,
    W,
    labels,
    infer_logp,
    loss_weights,
    need_dx,
    need_dw,
    options,
):
    """For a loss entry call in the checkpoint pass ``ckpt`` (:func:`_checkpoint_pass`, not ``None``): whether the call
    leaves its gradients to the recomputation (:func:`_checkpoint_defers`), and (a recomputed call) the checkpointed
    call it hands its result to.  A checkpointed call that does gets ``ctx.split`` (with the options its backward's own
    fused forward would take)."""
    key = _loss_call_key(
        X, W, labels, infer_logp, loss_weights, need_dx, need_dw, options
    )
    if ckpt.first:
        split = _checkpoint_defers()
        record = _checkpoint_first_call(ckpt, key, split)
        if split:
            calls, index = record if record is not None else (None, None)
            ctx.split = _SplitCall(calls, index, None)
            ctx.options, ctx.need = options, (need_dx, need_dw)
        return split, None
    # the recomputation follows its checkpointed call (unknown: the same rule decides, as it did there)
    record = _checkpoint_recomputed_call(ckpt)
    if record is None:
        return _checkpoint_defers(), None
    calls, index, first_key, split = record
    return split, ((calls, index) if split and first_key == key else None)


def _hand_over(ctx, ckpt: _CheckpointPass, handed, result):
    """A recomputed call's fused-forward result, as the gradient state of its checkpointed call's backward (the outputs
    and the autograd history of its inputs left out)."""
    state = replace(
        result,
        loss=None,
        logp=None,
        x_last=_detached(result.x_last),
        x_src=_detached(result.x_src),
    )
    # attributes set on the result past its fields (the opt-in budget forms' memory plan, row state, plan, inputs)
    names = {f.name for f in fields(result)}
    for name, value in vars(result).items():
        if name not in names:
            if isinstance(value, tuple):
                value = tuple(
                    _detached(v) if isinstance(v, torch.Tensor) else v for v in value
                )
            setattr(state, name, value)
    ctx.split = _SplitCall(None, None, state)
    if handed is not None:
        with _CHECKPOINT_LOCK:
            handed[0].handed[(ckpt.graph, handed[1])] = state


# The log-probability entries inside a non-reentrant ``torch.utils.checkpoint`` (gradient placement ``"auto"``): the
# checkpointed forward keeps its call's row results -- ``logp``, the FP32 row statistic ``lse``, the valid-row index
# and the compact temperature, O(T) bytes -- in the checkpoint frame, and the recomputation in the backward takes them
# instead of running the logits GEMMs again: it returns a copy of the same ``logp`` and saves the same row results, so
# the backward reads the bits the plain call's backward reads.  A recomputed call takes them only when it repeats its
# checkpointed call (paired per frame and graph by call order) on the same input tensors (the same objects, unchanged
# since: version counters) under the same options; otherwise, under the placement ``"forward"``, or after the kept
# ``logp`` was modified in place, it runs the forward.
class _LogprobKept(NamedTuple):
    """A checkpointed log-probability call's row results, kept for its recomputation."""

    key: tuple  # the call's options and its inputs' metadata and version counters
    refs: tuple  # weak references to the call's input tensors (None for an absent one)
    # an alias of the returned logp (its version counter tells an in-place change)
    logp: torch.Tensor
    logp_version: int
    saved: tuple  # the tensors the forward saves past X, W and labels
    extra: Any = None  # a non-tensor the backward reads (the scalar temperature)


class _LogprobCalls:
    """The log-probability entry calls of one checkpoint frame: per call of the checkpointed forward (in order) its kept
    row results (``None``: kept none) and per graph the recomputation's call count."""

    __slots__ = ("calls", "recomputed", "__weakref__")

    def __init__(self) -> None:
        self.calls: list = []
        self.recomputed: dict = {}


_LOGPROB_CHECKPOINT_CALLS: "weakref.WeakKeyDictionary[Any, _LogprobCalls]" = (
    weakref.WeakKeyDictionary()
)


def _logprob_call_key(inputs: tuple, options: tuple) -> tuple:
    """What a recomputed log-probability call must match to take its checkpointed call's row results."""
    meta = tuple(None if t is None else (_meta_key(t), t._version) for t in inputs)
    return meta + (repr(options),)


def _logprob_checkpoint_keep(
    ctx, ckpt: _CheckpointPass, key: tuple, inputs: tuple, logp, saved, extra=None
) -> None:
    """Record a log-probability call of the checkpointed forward, with its row results when a backward can follow
    (placement ``"auto"``, an input requiring a gradient)."""
    if ckpt.frame is None:
        return
    kept = None
    if gradient_placement() == "auto" and any(ctx.needs_input_grad[:2]):
        kept = _LogprobKept(
            key,
            tuple(None if t is None else weakref.ref(t) for t in inputs),
            logp.detach(),
            logp._version,
            tuple(_detached(t) for t in saved),
            extra,
        )
    with _CHECKPOINT_LOCK:
        calls = _LOGPROB_CHECKPOINT_CALLS.get(ckpt.frame)
        if calls is None:
            calls = _LOGPROB_CHECKPOINT_CALLS[ckpt.frame] = _LogprobCalls()
        calls.calls.append(kept)


def _logprob_checkpoint_take(
    ckpt: _CheckpointPass, key: tuple, inputs: tuple
) -> Optional[_LogprobKept]:
    """The kept row results of the checkpointed call a call of the recomputation repeats (``None``: run the
    forward)."""
    if ckpt.frame is None:
        return None
    with _CHECKPOINT_LOCK:
        calls = _LOGPROB_CHECKPOINT_CALLS.get(ckpt.frame)
        if calls is None:
            return None
        index = calls.recomputed.get(ckpt.graph, 0)
        calls.recomputed[ckpt.graph] = index + 1
        kept = calls.calls[index] if index < len(calls.calls) else None
    if (
        kept is None
        or kept.key != key
        or kept.logp._version != kept.logp_version
        or any(
            (ref is None) != (t is None) or (ref is not None and ref() is not t)
            for ref, t in zip(kept.refs, inputs, strict=True)
        )
    ):
        return None
    return kept


def _split_state(ctx, inputs, forward):
    """The gradient state of a call with ``ctx.split`` in its backward (after its inputs were unpacked, which ran the
    recomputation): the recomputed call's result, else ``forward`` (the fused forward) on the inputs."""
    split = ctx.split
    state = split.result
    if state is None and split.calls is not None:
        with _CHECKPOINT_LOCK:
            state = split.calls.handed.pop((_graph_key(), split.position), None)
    if state is None:
        warnings.warn(
            "a checkpointed loss entry call got no gradient state from its recomputation: its backward runs the fused "
            "forward again (two gradient passes in this step); set_gradient_placement('forward') runs one",
            RuntimeWarning,
            stacklevel=2,
        )
        X, W, labels, infer_logp, loss_weights = inputs
        need_dx, need_dw = ctx.need
        state = forward(
            X,
            W,
            labels,
            infer_logp=infer_logp,
            loss_weights=loss_weights,
            need_dx=need_dx,
            need_dw=need_dw,
            **ctx.options,
        )
    return state


def _loss_forward_hooked(
    ctx, X, W, labels, infer_logp, loss_weights, need_dx, need_dw, options
):
    """The loss entry's forward under saved-tensor hooks: inside a non-reentrant ``torch.utils.checkpoint`` with the
    gradient placement ``"auto"`` the checkpointed forward computes the loss (and logp) only and the recomputation
    hands its fused forward's gradient state to the checkpointed call's backward (``ctx.split``); both save the call's
    inputs.  ``None``: the fused forward (other hooks on top, an unsupported torch, the placement ``"forward"`` or
    torch's deterministic algorithms, :func:`_checkpoint_defers`)."""
    ckpt = _checkpoint_pass()
    if ckpt is None:
        return None
    split, handed = _loss_checkpoint_split(
        ctx, ckpt, X, W, labels, infer_logp, loss_weights, need_dx, need_dw, options
    )
    if not split:
        return None
    ctx.set_materialize_grads(False)
    ctx.num_rows = int(X.shape[0])
    ctx.grad_weight_dtype = options["grad_weight_dtype"]
    ctx.backend = options["backend"]
    # a checkpointed call saves its inputs in both passes: the same count and metadata for checkpoint's check
    ctx.save_for_backward(X, W, labels, infer_logp, loss_weights)
    first = ckpt.first
    result = _loss_forward(
        X,
        W,
        labels,
        infer_logp=infer_logp,
        loss_weights=loss_weights,
        need_dx=need_dx and not first,
        need_dw=need_dw and not first,
        **options,
    )
    if not first:
        _hand_over(ctx, ckpt, handed, result)
    ctx.mark_non_differentiable(result.logp)
    return result.loss, result.logp


def _loss_backward_split(ctx, grad_loss):
    """``(dX, dW)`` of a call with ``ctx.split``: its gradient state is the recomputation's (unpacking the inputs runs
    it) or, without a handed result, this backward's own fused forward."""
    state = _split_state(ctx, ctx.saved_tensors, _loss_forward)
    one_pass = getattr(state, "one_pass", None)
    if one_pass is not None:
        return _one_pass_backward(one_pass, state.x_last, state.dx_acc, grad_loss)
    if ctx.num_rows == 0 or (
        state.row_index is not None and state.row_index.numel() == 0
    ):
        dx = (
            None
            if state.dx_acc is None
            else torch.zeros(
                (ctx.num_rows, state.dx_acc.shape[1]),
                dtype=torch.bfloat16,
                device=state.dx_acc.device,
            )
        )
        dw = (
            None
            if state.dw_acc is None
            else _zero_gradient(state.dw_acc, grad_loss, ctx.grad_weight_dtype)
        )  # cast(g * 0), as ForwardResult.backward forms it
        return dx, dw
    return backward_loss(
        state.dx_acc,
        state.dw_acc,
        grad_loss,
        grad_weight_dtype=ctx.grad_weight_dtype,
        backend=ctx.backend,
        row_index=state.row_index,
        num_rows=ctx.num_rows,
        dz_last=state.dz_last,
        x_last=state.x_last,
        x_src=state.x_src,
        x_idx=state.x_idx,
        row_valid=state.row_valid,
        gemm_tuning=state.gemm_tuning,
    )


# One-pass cross-entropy of the smallest calls (:func:`set_one_pass_rows`; :func:`_one_pass_call` names the calls).  A
# loss entry call with ``objective="ce"`` of ``T`` rows, ``T`` one of :func:`one_pass_rows` (default ``{1}``) and one
# chunk, runs three programs instead of the chunk loop: ``one_pass_fwd_t<T>`` reads ``W`` once and forms the kept BF16
# logits ``Z [T, V]``, per CTA the online maximum and sum of ``exp(z - m)`` and (with a trainable ``X``) the
# softmax-weighted sum ``u`` of the head rows (``one_pass_fwd_nodx_t<T>`` without ``u``); ``one_pass_combine`` merges
# the CTAs' partials into ``lse``, ``logp``, the loss and the FP32 input-gradient accumulator ``d (W_y - u / s)``; the
# backward's ``one_pass_dw`` writes ``dW = bf16(g sum_t dz_t x_t)`` from the kept logits in one pass (no FP32 ``dW``
# accumulator) and ``dX = bf16(g dX_acc)``.  ``W`` is read once in the forward instead of twice (the logits GEMM and
# the dX GEMM), and the weight gradient is a plain write.  Bits: ``z`` is the chunked path's (the same FP32 products
# and BF16 rounding) and ``dz`` / ``dW`` take its per-element operation sequences, but ``lse`` and the loss sum in
# another fixed order, so ``loss``, ``logp``, ``lse``, ``dz`` and ``dW`` are not the deterministic package's bits;
# ``dX`` takes the FP32 softmax weights in place of the BF16 ``dz`` -- every output within the accuracy budgets, and
# run to run identical (no reduction whose order the launch leaves open).  An ignored row (a
# negative label) takes no part: its ``logp``, ``dX`` row and ``dW`` contribution are exact zeros and a non-finite ``X``
# row never reaches an output; every row ignored streams no ``W``.  Applies on CUDA with ``backend="cake"``, a
# contiguous BF16 ``W`` of ``ONE_PASS_HIDDEN`` columns, BF16 ``X`` rows with a 16-byte base and pitch, contiguous int64
# labels, ``compact_rows`` resolving to compacted, no explicit ``gemm_tuning`` and a record that registers the
# programs; checked before the small-call staging (:func:`_small_call_rows`), which such a call skips.  Under a
# non-reentrant ``torch.utils.checkpoint`` (gradient placement ``"auto"``) the checkpointed forward runs the forward
# without ``u`` and the recomputation the full forward, as the chunked path does.
ONE_PASS_HIDDEN = 6144
ONE_PASS_FWD_STAGES = {
    1: ("one_pass_fwd_t1", "one_pass_fwd_nodx_t1"),
    2: ("one_pass_fwd_t2", "one_pass_fwd_nodx_t2"),
}
ONE_PASS_COMBINE_STAGE = "one_pass_combine"
ONE_PASS_DW_STAGE = "one_pass_dw"
# vocabulary rows per weight-gradient CTA (the program's ``block_rows``)
ONE_PASS_DW_ROWS = 32
# input-gradient elements / 8 per CTA of ``one_pass_dw``
ONE_PASS_DX_THREADS = 256
# the most forward CTAs ``one_pass_combine`` merges
ONE_PASS_MAX_PARTS = 256
ONE_PASS_ROWS_DEFAULT = frozenset({1})
_ONE_PASS = {"rows": ONE_PASS_ROWS_DEFAULT}
# Per device index: ``(module_name, record, rows)`` -- the record and the row counts whose one-pass programs it
# registers (every forward form of the count, the merge and the backward).
_ONE_PASS_RECORDS: dict[int, tuple[str, dict[str, Any], frozenset]] = {}


def one_pass_rows() -> frozenset:
    """Row counts of the loss entry's one-pass cross-entropy (:func:`set_one_pass_rows`; default ``{1}``)."""
    return _ONE_PASS["rows"]


def set_one_pass_rows(rows) -> frozenset:
    """Set the row counts whose cross-entropy loss entry calls run the one-pass programs (an iterable of row counts, a
    subset of ``ONE_PASS_FWD_STAGES``; empty = never) for the following calls of this process, and return the previous
    set."""
    if isinstance(rows, (bool, int, str)) or not all(
        isinstance(r, int) and not isinstance(r, bool) for r in rows
    ):
        raise ValueError(f"one_pass_rows takes an iterable of row counts, got {rows!r}")
    new = frozenset(int(r) for r in rows)
    if new - set(ONE_PASS_FWD_STAGES):
        raise ValueError(
            f"one_pass_rows: no one-pass forward for {sorted(new - set(ONE_PASS_FWD_STAGES))} rows"
        )
    previous = _ONE_PASS["rows"]
    _ONE_PASS["rows"] = new
    return previous


def _one_pass_record(index: int) -> tuple[str, dict[str, Any], frozenset]:
    """``(module_name, record, rows)`` of CUDA device ``index`` (resolved once per device): its record and the row
    counts whose forward forms, merge and backward it registers (empty without those programs)."""
    entry = _ONE_PASS_RECORDS.get(index)
    if entry is None:
        module_name, record = record_for(torch.device("cuda", index))
        rows: frozenset = frozenset()
        if ONE_PASS_COMBINE_STAGE in record and ONE_PASS_DW_STAGE in record:
            rows = frozenset(
                t
                for t, stages in ONE_PASS_FWD_STAGES.items()
                if all(stage in record for stage in stages)
            )
        entry = _ONE_PASS_RECORDS.setdefault(index, (module_name, record, rows))
    return entry


def _one_pass_call(
    X,
    W,
    labels,
    *,
    objective,
    infer_logp,
    loss_weights,
    chunk_size,
    backend,
    compact_rows,
    gemm_tuning,
) -> bool:
    """Does this loss entry call run the one-pass cross-entropy (see :data:`ONE_PASS_HIDDEN`)?  Host values alone;
    malformed inputs answer ``False`` (the entry's validation reports them)."""
    rows = _ONE_PASS["rows"]
    if (
        not rows
        or backend != "cake"
        or objective != "ce"
        or infer_logp is not None
        or loss_weights is not None
        or gemm_tuning
        or not isinstance(X, torch.Tensor)
        or not isinstance(W, torch.Tensor)
        or not isinstance(labels, torch.Tensor)
        or X.ndim != 2
        or W.ndim != 2
        or labels.ndim != 1
        or X.dtype != torch.bfloat16
        or W.dtype != torch.bfloat16
        or labels.dtype != torch.int64
        or not X.is_cuda
        or W.device != X.device
        or labels.device != X.device
    ):
        return False
    T, H = int(X.shape[0]), int(X.shape[1])
    if (
        T not in rows
        or H != ONE_PASS_HIDDEN
        or int(W.shape[1]) != H
        or int(W.shape[0]) < 1
        or int(labels.shape[0]) != T
        or not W.is_contiguous()
        or W.data_ptr() % 16
        or X.stride(1) != 1
        or X.data_ptr() % 16
        or (T > 1 and (X.stride(0) < H or X.stride(0) % 8))
        or labels.stride(0) != 1
    ):
        return False
    try:
        if int(chunk_size) < T or not _resolve_compact(compact_rows):
            return False
    except (TypeError, ValueError):
        return False
    if arch_for(X.device) is None:
        return False
    index = (
        X.device.index if X.device.index is not None else torch.cuda.current_device()
    )
    return T in _one_pass_record(int(index))[2]


def _x_span(X: torch.Tensor, ldx: int) -> torch.Tensor:
    """The rows of ``X`` (row pitch ``ldx``) as the flat contiguous span of its storage they cover, the operand a
    one-pass program reads through its own pitch."""
    T, H = int(X.shape[0]), int(X.shape[1])
    return X.as_strided(((T - 1) * ldx + H,), (1,))


def _one_pass_launch(module_name, record, stage, values, device, index) -> None:
    """One eager launch of a one-pass program (the remembered templated launch of its form, :func:`_eager_launch`)."""
    full: dict[str, Any] = {name: None for name in COMMON_TENSORS}
    full.update({name: 0 for name in COMMON_SCALARS})
    full.update(values)
    launch, arguments = _eager_launch(
        record, module_name, stage, full, device, Geometry.from_record(record)
    )
    with _ffi_stream_context(index):
        if launch.prepare is not None:
            launch.prepare(*arguments)
        launch.entry(*arguments)


def _one_pass_forward(
    X,
    W,
    labels,
    *,
    objective: str = "ce",
    loss_div=None,
    infer_logp=None,
    loss_weights=None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    need_dx: bool = True,
    need_dw: bool = True,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    return_logp: bool = True,
    **_switches,
) -> ForwardResult:
    """The loss entry's forward of a one-pass call (:func:`_one_pass_call`): ``loss``, ``logp`` and, with ``need_dx``,
    the FP32 ``dx_acc [T, H]``; the backward's operands ride on the result as ``one_pass`` (the kept logits, the row
    results, the labels) and ``x_last`` (``X``, when ``need_dw``).  The chunked path's switches do not apply."""
    device = X.device
    index = int(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    module_name, record, _rows = _one_pass_record(index)
    record_abi(record)
    problem = validate_lm_head_inputs(
        X,
        W,
        labels,
        objective=objective,
        loss_div=loss_div,
        infer_logp=infer_logp,
        loss_weights=loss_weights,
        chunk_size=chunk_size,
        grad_weight_dtype=grad_weight_dtype,
        geometry=Geometry.from_record(record),
    )
    T, H, V = problem.num_rows, problem.hidden, problem.vocab
    sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    parts = min(sms, ONE_PASS_MAX_PARTS, V)
    rpc = -(-V // parts)
    parts = -(-V // rpc)
    ldx = int(X.stride(0)) if T > 1 else H
    # the kept state (logits, row results, labels, dX accumulator) in one allocation, the partials in another
    kept = [
        ("z", 2 * T * V),
        ("rows", 8 * T),
        ("logp", 4 * T),
        ("loss", 4),
        ("lab", 8 * T),
    ]
    if need_dx:
        kept.append(("dxa", 4 * T * H))
    offsets, total = {}, 0
    for name, nbytes in kept:
        offsets[name] = (total, nbytes)
        total += _align(nbytes)
    block = torch.empty((total,), dtype=torch.uint8, device=device)

    def view(name, dtype):
        start, nbytes = offsets[name]
        return block[start : start + nbytes].view(dtype)

    Z = view("z", torch.bfloat16).view(T, V)
    ROWS = view("rows", torch.float32).view(T, 2)
    LOGP = view("logp", torch.float32)
    LOSS = view("loss", torch.float32)
    LAB = view("lab", torch.int64)
    DXA = view("dxa", torch.float32).view(T, H) if need_dx else None
    o_pu = _align(8 * parts * T)
    scratch = torch.empty(
        (o_pu + (4 * parts * T * H if need_dx else 0)) // 4,
        dtype=torch.float32,
        device=device,
    )
    PMS = scratch[: 2 * parts * T]
    PU = scratch[o_pu // 4 :] if need_dx else None
    fwd, fwd_nodx = ONE_PASS_FWD_STAGES[T]
    flat = dict(
        W=W.reshape(-1), Z=Z.reshape(-1), DXA=None if DXA is None else DXA.reshape(-1)
    )
    values: dict[str, Any] = dict(
        W=flat["W"],
        X=_x_span(X, ldx),
        labels=labels,
        Z=flat["Z"],
        PMS=PMS,
        V=V,
        ldx=ldx,
        rpc=rpc,
        parts=parts,
    )
    if need_dx:
        values["PU"] = PU
    _one_pass_launch(
        module_name, record, fwd if need_dx else fwd_nodx, values, device, index
    )
    _one_pass_launch(
        module_name,
        record,
        ONE_PASS_COMBINE_STAGE,
        dict(
            PU=PU if need_dx else PMS,
            PMS=PMS,
            W=flat["W"],
            Z=flat["Z"],
            labels=labels,
            DXA=flat["DXA"] if need_dx else LOGP,
            ROWS=ROWS.reshape(-1),
            LOGP=LOGP,
            LOSS=LOSS,
            LAB=LAB,
            parts=parts,
            T=T,
            V=V,
            H=H,
            with_u=int(need_dx),
            loss_div=float(problem.loss_div),
        ),
        device,
        index,
    )
    return ForwardResult(
        loss=LOSS.view(()),
        logp=LOGP,
        dx_acc=DXA,
        backend="cake",
        num_rows=T,
        x_last=X if need_dw else None,
        one_pass=(Z, ROWS, LAB, bool(need_dx), bool(need_dw)),
    )


def _one_pass_backward(one_pass, x, dx_acc, grad):
    """``(dX, dW)`` of a one-pass forward (``one_pass`` its kept state, ``x`` the call's ``X`` when ``W`` is trainable,
    ``dx_acc`` its FP32 input-gradient accumulator when ``X`` is): one launch of ``one_pass_dw`` writes both, each
    ``None`` when the forward did not form it.  ``grad`` is the upstream scalar (``None`` = 1)."""
    Z, ROWS, LAB, need_dx, need_dw = one_pass
    T, V = int(Z.shape[0]), int(Z.shape[1])
    device = Z.device
    index = int(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    module_name, record, _rows = _one_pass_record(index)
    H = ONE_PASS_HIDDEN
    g = (
        _unit_scale(device)
        if grad is None
        else grad.detach().reshape(1).to(device=device, dtype=torch.float32)
    )
    DW = torch.empty((V, H), dtype=torch.bfloat16, device=device) if need_dw else None
    DX = torch.empty((T, H), dtype=torch.bfloat16, device=device) if need_dx else None
    if not need_dw and not need_dx:
        return None, None
    dw_blocks = -(-V // ONE_PASS_DW_ROWS) if need_dw else 0
    dx_blocks = -(-(T * H // 8) // ONE_PASS_DX_THREADS) if need_dx else 0
    ldx = int(x.stride(0)) if need_dw and T > 1 else H
    _one_pass_launch(
        module_name,
        record,
        ONE_PASS_DW_STAGE,
        dict(
            Z=Z.reshape(-1),
            ROWS=ROWS.reshape(-1),
            LAB=LAB,
            X=_x_span(x, ldx) if need_dw else Z.reshape(-1),
            DXA=dx_acc.reshape(-1) if need_dx else ROWS.reshape(-1),
            grad=g,
            DW=(DW if need_dw else DX).reshape(-1),
            DX=(DX if need_dx else DW).reshape(-1),
            V=V,
            T=T,
            ldx=ldx,
            dw_blocks=dw_blocks,
            dx_blocks=dx_blocks,
        ),
        device,
        index,
    )
    return DX, DW


def _loss_forward(X, W, labels, *, one_pass: bool = False, **kwargs) -> ForwardResult:
    """The loss entry's forward: the one-pass programs for a call :func:`_one_pass_call` names, else
    :func:`forward_loss`."""
    if one_pass:
        return _one_pass_forward(X, W, labels, **kwargs)
    return forward_loss(X, W, labels, **kwargs)


class CakeLMHeadNondetLossFunction(torch.autograd.Function):
    """Entry (a): the forward produces the FP32 gradient accumulators; the backward scales and casts them.

    Gradient placement ``"auto"`` (:func:`set_gradient_placement`) inside a non-reentrant ``torch.utils.checkpoint``:
    the checkpointed forward computes the loss only and saves the call's inputs; the recomputation in the backward runs
    the fused forward, saves the same inputs and hands its result to the checkpointed call's backward (which runs the
    fused forward itself when nothing was handed over).  The outputs are the fused forward's bits either way."""

    @staticmethod
    def forward(
        ctx,
        X,
        W,
        labels,
        objective,
        loss_div,
        infer_logp,
        loss_weights,
        chunk_size,
        return_logp,
        grad_weight_dtype,
        backend,
        compact_rows,
        fuse_dw_cast,
        dx_finalize,
        dw_stream,
        hidden_count,
        gemm_tuning,
    ):
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        # a cross-entropy call of the one-pass row counts (one_pass_rows) with a BF16 dW: the one-pass programs, in every
        # placement
        one_pass = grad_weight_dtype == torch.bfloat16 and _one_pass_call(
            X,
            W,
            labels,
            objective=objective,
            infer_logp=infer_logp,
            loss_weights=loss_weights,
            chunk_size=chunk_size,
            backend=backend,
            compact_rows=compact_rows,
            gemm_tuning=gemm_tuning,
        )
        # under saved-tensor hooks: a non-reentrant torch.utils.checkpoint's passes leave the gradients to the
        # recomputation (gradient placement "auto"); the plain step (no hooks) and other hooks run the fused forward.
        # Under a selective checkpoint's modes (any need_dx / need_dw): again, with the dispatch modes popped
        if _TOP_SAVED_HOOKS(False) is not None and (
            need_dx or need_dw or _selective_checkpoint_modes()
        ):
            if _selective_checkpoint_modes():
                with _modes_off():
                    return CakeLMHeadNondetLossFunction.forward(
                        ctx,
                        X,
                        W,
                        labels,
                        objective,
                        loss_div,
                        infer_logp,
                        loss_weights,
                        chunk_size,
                        return_logp,
                        grad_weight_dtype,
                        backend,
                        compact_rows,
                        fuse_dw_cast,
                        dx_finalize,
                        dw_stream,
                        hidden_count,
                        gemm_tuning,
                    )
            out = _loss_forward_hooked(
                ctx,
                X,
                W,
                labels,
                infer_logp,
                loss_weights,
                need_dx,
                need_dw,
                dict(
                    objective=objective,
                    loss_div=loss_div,
                    chunk_size=chunk_size,
                    grad_weight_dtype=grad_weight_dtype,
                    backend=backend,
                    compact_rows=compact_rows,
                    fuse_dw_cast=fuse_dw_cast,
                    dx_finalize=dx_finalize,
                    dw_stream=dw_stream,
                    hidden_count=hidden_count,
                    gemm_tuning=gemm_tuning,
                    return_logp=bool(return_logp),
                    one_pass=one_pass,
                ),
            )
            if out is not None:
                return out
        result = _loss_forward(
            X,
            W,
            labels,
            one_pass=one_pass,
            objective=objective,
            loss_div=loss_div,
            infer_logp=infer_logp,
            loss_weights=loss_weights,
            chunk_size=chunk_size,
            need_dx=need_dx,
            need_dw=need_dw,
            grad_weight_dtype=grad_weight_dtype,
            backend=backend,
            compact_rows=compact_rows,
            fuse_dw_cast=fuse_dw_cast,
            dx_finalize=dx_finalize,
            dw_stream=dw_stream,
            hidden_count=hidden_count,
            gemm_tuning=gemm_tuning,
            return_logp=bool(return_logp),
        )
        ctx.set_materialize_grads(False)
        ctx.dx_acc, ctx.dw_acc = (
            result.dx_acc,
            result.dw_acc,
        )  # saved state, never mutated
        # the deferred weight gradient's operands: the last chunk's dz rows and its X rows, the latter a view of the
        # input X (or X itself with the chunk's row index when compacted) -- saved through the autograd context so an
        # in-place write to X between the forward and the backward raises PyTorch's saved-tensor version error
        # instead of yielding a stale dW
        ctx.save_for_backward(result.dz_last, result.x_last, result.x_src, result.x_idx)
        ctx.row_index, ctx.num_rows = (
            result.row_index,
            int(X.shape[0]),
        )  # compacted: dx_acc holds the rows of row_index
        ctx.row_valid = (
            result.row_valid
        )  # the fused dX finalize's valid-row mask (None otherwise)
        ctx.grad_weight_dtype = grad_weight_dtype
        ctx.backend = backend
        # the forward's resolved GEMM knobs: the deferred weight-gradient GEMM takes them
        ctx.gemm_tuning = result.gemm_tuning
        # T == 0, every row ignored, or every valid row of zero weight: no gradient row
        ctx.empty = X.shape[0] == 0 or (
            result.row_index is not None and result.row_index.numel() == 0
        )
        ctx.split = None
        # the one-pass call's kept state (the backward's one launch), None for the chunked path
        ctx.one_pass = getattr(result, "one_pass", None)
        ctx.mark_non_differentiable(result.logp)
        return result.loss, result.logp

    @staticmethod
    def backward(ctx, grad_loss, grad_logp=None):
        none = (None,) * 17
        if grad_loss is None:
            return none
        if ctx.split is not None:
            return _loss_backward_split(ctx, grad_loss) + none[2:]
        if ctx.empty:  # T == 0 or every row ignored: zero gradients without binding or launching a program
            dx = (
                None
                if ctx.dx_acc is None
                else torch.zeros(
                    (ctx.num_rows, ctx.dx_acc.shape[1]),
                    dtype=torch.bfloat16,
                    device=ctx.dx_acc.device,
                )
            )
            dw = (
                None
                if ctx.dw_acc is None
                else _zero_gradient(ctx.dw_acc, grad_loss, ctx.grad_weight_dtype)
            )  # cast(g * 0), as ForwardResult.backward forms it
            return (dx, dw) + none[2:]
        dz_last, x_last, x_src, x_idx = ctx.saved_tensors
        if ctx.one_pass is not None:
            dx, dw = _one_pass_backward(ctx.one_pass, x_last, ctx.dx_acc, grad_loss)
            return (dx, dw) + none[2:]
        dx, dw = backward_loss(
            ctx.dx_acc,
            ctx.dw_acc,
            grad_loss,
            grad_weight_dtype=ctx.grad_weight_dtype,
            backend=ctx.backend,
            row_index=ctx.row_index,
            num_rows=ctx.num_rows,
            dz_last=dz_last,
            x_last=x_last,
            x_src=x_src,
            x_idx=x_idx,
            row_valid=ctx.row_valid,
            gemm_tuning=ctx.gemm_tuning,
        )
        return (dx, dw) + none[2:]


# The kept-logits form of the log-probability entry (``memory_mode="keep_logits"``).  Every kept forward keeps its
# chunks' logits as consecutive row slices of ONE contiguous BF16 buffer (:func:`kept_rows_of`): ``[T_v, V]`` when
# the valid rows are counted on the host, the hidden valid-row count's ``[T, V]`` (:attr:`_HiddenCount.kept`, sized
# before the count is known) when it runs.  Switches:
# * ``tail_fold`` -- a kept plan of a call without the hidden valid-row count folds a last chunk of fewer than
#   ``chunk / 2`` rows into the chunk before it by the default plan's rule (:func:`loop_chunks`: calls of fewer than
#   :data:`HIDDEN_COUNT_MIN_CHUNKS` chunks of ``T``);
# * ``one_pass`` -- the first backward forms the weight gradient as ONE fused-cast GEMM over ``K = T_v`` rows of the
#   contiguous buffer (no FP32 ``dW`` accumulator, the step's ``4 * V * H`` bytes) when it has at most
#   :func:`kept_one_pass_rows` rows (:func:`backward_logprob_keep_ordered`), after dX: in order (``dx_side`` off) the
#   dX GEMMs and their finalize end and free the FP32 dX accumulator before the weight gradient's output is allocated,
#   a one-chunk call too.  Above the limit, or off, every chunk accumulates as in the default backward.  Off, no
#   forward keeps a contiguous buffer under the hidden count;
# * ``one_pass_rows`` -- that row limit (``0``: :data:`KEPT_ONE_PASS_ROWS`);
# * ``dx_one_pass`` -- with ``one_pass``, the dX GEMM also runs once over ``M = T_v`` (one ``row_grad``, one dX GEMM;
#   its dX K-slice count is the one of ``T_v`` rows, so dX takes another FP32 order than the chunked dX).  Off (the
#   default): ``row_grad`` and dX run in the forward's chunks with the chunked backward's kernels, so dX is the chunked
#   backward's bits wherever those repeat;
# * ``dx_side`` -- with ``one_pass`` and the chunked dX: on, the dX GEMMs run on the side stream behind each chunk's
#   ``row_grad`` and the weight-gradient GEMM on the caller's stream behind the last ``row_grad``, so the two GEMMs
#   overlap; off (the default), the dX chunks and then the weight gradient run in order on the caller's stream (the
#   same kernels and bits; on B200 the two overlapped GEMMs contend and run slower than in order);
# * ``hidden_count`` -- the kept entry's default of ``hidden_count`` (``True``: the hidden valid-row count wherever it
#   is eligible -- calls of three chunks or more (:data:`HIDDEN_COUNT_MIN_CHUNKS`) -- with the shipped chunks;
#   ``False``: every call counts its valid rows on the host before the first launch).
# Process settings read per call (:func:`set_kept_logits_form`).
_KEPT_LOGITS_FORM: dict[str, int] = {
    "tail_fold": True,
    "one_pass": True,
    "one_pass_rows": 0,
    "dx_one_pass": False,
    "dx_side": False,
    "hidden_count": True,
}

# The default one-pass row limit (``one_pass_rows=0``): the weight gradient of every call a kept chunk plan can bind
# (:data:`KEPT_CHUNK_MAX_ROWS`).
KEPT_ONE_PASS_ROWS = KEPT_CHUNK_MAX_ROWS


def kept_logits_form() -> dict[str, int]:
    """The kept-logits form switches (:func:`set_kept_logits_form`): ``tail_fold``, ``one_pass``, ``one_pass_rows``,
    ``dx_one_pass``, ``dx_side``, ``hidden_count``."""
    return dict(_KEPT_LOGITS_FORM)


def kept_one_pass_rows() -> int:
    """The most kept rows whose weight gradient the one-pass backward forms in one GEMM (``one_pass`` of
    :func:`set_kept_logits_form`): ``one_pass_rows`` when set (``0``: :data:`KEPT_ONE_PASS_ROWS`), never above
    :data:`KEPT_CHUNK_MAX_ROWS`."""
    rows = int(_KEPT_LOGITS_FORM["one_pass_rows"]) or KEPT_ONE_PASS_ROWS
    return min(rows, KEPT_CHUNK_MAX_ROWS)


def set_kept_logits_form(
    *,
    tail_fold: Optional[bool] = None,
    one_pass: Optional[bool] = None,
    one_pass_rows: Optional[int] = None,
    dx_one_pass: Optional[bool] = None,
    dx_side: Optional[bool] = None,
    hidden_count: Optional[bool] = None,
) -> dict[str, int]:
    """Set the kept-logits form switches (``None`` leaves one unchanged) for the following calls of this process and
    return the previous settings.  ``tail_fold`` (default ``True``): fold a short last chunk into the one before it
    where the default plan folds (:func:`loop_chunks`; off: the chunks of ``chunk_size`` rows).
    ``one_pass`` (default ``True``): the first backward forms the weight gradient in one GEMM over every kept row, after
    dX, when there are at most :func:`kept_one_pass_rows` of them.  ``one_pass_rows`` (default ``0`` =
    :data:`KEPT_ONE_PASS_ROWS`): that row limit, an int >= 0.  ``dx_one_pass`` (default ``False``): the dX GEMM in one
    launch over every kept row too.  ``dx_side`` (default ``False``): the chunked dX GEMMs on the side stream beside
    the weight gradient (off: in order on the caller's stream).  ``hidden_count`` (default ``True``: the hidden count where it is eligible): the kept entry's
    default of its ``hidden_count`` keyword.  ``logp`` / ``lse`` are the same bits under every setting; ``dW`` sums its
    rows in another grouping under ``one_pass``, or with ``tail_fold`` off on a call the default plan folds (each
    setting repeats bit for bit); ``dX`` is the
    chunked dX of the forward's chunks unless ``dx_one_pass`` (the same products)."""
    given = {
        name: value
        for name, value in (
            ("tail_fold", tail_fold),
            ("one_pass", one_pass),
            ("one_pass_rows", one_pass_rows),
            ("dx_one_pass", dx_one_pass),
            ("dx_side", dx_side),
            ("hidden_count", hidden_count),
        )
        if value is not None
    }
    for name, value in given.items():
        if name == "one_pass_rows":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"the kept-logits form switch one_pass_rows must be an int >= 0, got {value!r}"
                )
        elif not isinstance(value, bool):
            raise ValueError(
                f"the kept-logits form switch {name} must be a bool, got {value!r}"
            )
    previous = dict(_KEPT_LOGITS_FORM)
    _KEPT_LOGITS_FORM.update(given)
    return previous


def backward_logprob_keep_ordered(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    dlogp: torch.Tensor,
    kept_rows: torch.Tensor,
    logits: tuple,
    *,
    need_dx: bool = True,
    need_dw: bool = True,
    dx_side: bool = False,
    **options: Any,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """The one-pass kept backward that keeps dX in the forward's chunks: ``logits`` are the forward's kept views
    (consecutive row slices of the contiguous ``kept_rows``, :func:`kept_rows_of`).  ``row_grad`` and the dX GEMM run
    per view as in :func:`backward_logprob_keep` without the weight gradient (the chunked backward's dX kernels, so the
    same dX bits wherever those repeat); then ONE fused-cast weight-gradient GEMM over ``K = T_v`` reads every ``dz``
    row of ``kept_rows`` (``dz_ready``: no second ``row_grad``) -- no FP32 ``dW`` accumulator.  ``dx_side``: the dX
    GEMMs run on the side stream so the weight-gradient GEMM overlaps them (:meth:`_KeptBinding._launch_dx_side`).
    ``options``: the other keywords of :func:`backward_logprob_keep` (``chunk_size`` the forward's)."""
    rows = int(kept_rows.shape[0])
    one = dict(options, chunk_size=max(rows, 1))
    if not need_dx:
        return backward_logprob_keep(
            X,
            W,
            labels,
            lse,
            dlogp,
            (kept_rows,),
            need_dx=False,
            need_dw=need_dw,
            **one,
        )
    if not need_dw:
        return backward_logprob_keep(
            X, W, labels, lse, dlogp, logits, need_dx=True, need_dw=False, **options
        )
    out: dict[str, Optional[torch.Tensor]] = {}

    def weight_gradient() -> None:
        out["dw"] = backward_logprob_keep(
            X,
            W,
            labels,
            lse,
            dlogp,
            (kept_rows,),
            need_dx=False,
            need_dw=True,
            dz_ready=True,
            **one,
        )[1]

    dx, _ = backward_logprob_keep(
        X,
        W,
        labels,
        lse,
        dlogp,
        logits,
        need_dx=True,
        need_dw=False,
        dx_side=dx_side,
        after_dz=weight_gradient if dx_side else None,
        **options,
    )
    if "dw" not in out:
        weight_gradient()
    return dx, out["dw"]


class CakeLMHeadNondetLogprobFunction(torch.autograd.Function):
    """Entry (b): the forward saves the row statistic; the backward recomputes the logits."""

    @staticmethod
    def forward(
        ctx,
        X,
        W,
        labels,
        chunk_size,
        backend,
        compact_rows,
        fuse_dw_cast,
        dx_finalize,
        dw_stream,
        hidden_count,
        gemm_tuning,
        logprob_dz,
        logprob_lbl,
    ):
        if _TOP_SAVED_HOOKS(False) is not None and _selective_checkpoint_modes():
            with _modes_off():  # under a selective checkpoint's modes: popped (see _selective_checkpoint_modes)
                return CakeLMHeadNondetLogprobFunction.forward(
                    ctx,
                    X,
                    W,
                    labels,
                    chunk_size,
                    backend,
                    compact_rows,
                    fuse_dw_cast,
                    dx_finalize,
                    dw_stream,
                    hidden_count,
                    gemm_tuning,
                    logprob_dz,
                    logprob_lbl,
                )
        # every switch is resolved (and validated) once here, before any launch: the backward plans its chunks on the
        # forward's valid-row index under the same switches, the forward and the backward bind with the same GEMM knobs
        compact = _resolve_compact(compact_rows)
        fuse = _resolve_fuse(fuse_dw_cast)
        fused_dx = _resolve_dx_finalize(dx_finalize)
        stream_mode = dw_stream_mode(dw_stream)
        hidden = _resolve_hidden_count(hidden_count)
        tuning = _resolve_gemm_tuning(gemm_tuning)
        dz_arm = _resolve_logprob_dz(logprob_dz)
        lbl_arm = _resolve_logprob_lbl(logprob_lbl)
        ckpt = _checkpoint_pass() if _TOP_SAVED_HOOKS(False) is not None else None
        if ckpt is not None:
            inputs = (X, W, labels)
            key = _logprob_call_key(
                inputs,
                (chunk_size, backend, compact, hidden, _tuning_key(tuning), lbl_arm),
            )
            kept = None if ckpt.first else _logprob_checkpoint_take(ckpt, key, inputs)
        ctx.set_materialize_grads(False)
        if ckpt is not None and kept is not None:
            # the recomputation of a checkpointed call: its row results, no logits GEMM (see _LogprobKept)
            logp, (lse, row_index) = kept.logp.clone(), kept.saved
            # no saved problem under the checkpoint's saved-tensor hooks (see _saved_problem)
            problem = None
        else:
            result = forward_logprob(
                X,
                W,
                labels,
                chunk_size=chunk_size,
                backend=backend,
                compact_rows=compact,
                hidden_count=hidden,
                gemm_tuning=tuning,
                logprob_lbl=lbl_arm,
            )
            logp, lse, row_index = result.logp, result.lse, result.row_index
            problem = _saved_problem(result)
            if ckpt is not None and ckpt.first:
                _logprob_checkpoint_keep(ctx, ckpt, key, inputs, logp, (lse, row_index))
        # the valid-row index rides along: the backward plans its chunks without counting the rows again
        ctx.save_for_backward(X, W, labels, lse, row_index)
        ctx.problem = problem
        ctx.chunk_size, ctx.backend, ctx.compact = (
            chunk_size,
            backend,
            compact,
        )
        ctx.fuse_dw_cast = fuse
        ctx.dx_finalize = fused_dx
        ctx.dw_stream = stream_mode
        ctx.hidden_count = hidden
        ctx.gemm_tuning = tuning
        ctx.logprob_dz = dz_arm
        return logp

    @staticmethod
    def backward(ctx, dlogp):
        if dlogp is None:
            return (None,) * 13
        X, W, labels, lse, row_index = ctx.saved_tensors
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        dx, dw = backward_logprob(
            X,
            W,
            labels,
            lse,
            dlogp,
            chunk_size=ctx.chunk_size,
            need_dx=need_dx,
            need_dw=need_dw,
            backend=ctx.backend,
            compact_rows=ctx.compact,
            fuse_dw_cast=ctx.fuse_dw_cast,
            dx_finalize=ctx.dx_finalize,
            dw_stream=ctx.dw_stream,
            hidden_count=ctx.hidden_count,
            gemm_tuning=ctx.gemm_tuning,
            logprob_dz=ctx.logprob_dz,
            row_index=row_index,
            _problem=ctx.problem,
        )
        return (dx, dw) + (None,) * 11


class CakeLMHeadNondetLogprobTemperatureFunction(torch.autograd.Function):
    """:class:`CakeLMHeadNondetLogprobFunction` of a call with a temperature (:func:`cake_lm_head_nondet_train_logprob`)."""

    @staticmethod
    def forward(
        ctx,
        X,
        W,
        labels,
        chunk_size,
        backend,
        compact_rows,
        fuse_dw_cast,
        dx_finalize,
        dw_stream,
        hidden_count,
        gemm_tuning,
        logprob_dz,
        logprob_lbl,
        temperature,
        check_temperature,
    ):
        if _TOP_SAVED_HOOKS(False) is not None and _selective_checkpoint_modes():
            with _modes_off():  # under a selective checkpoint's modes: popped (see _selective_checkpoint_modes)
                return CakeLMHeadNondetLogprobTemperatureFunction.forward(
                    ctx,
                    X,
                    W,
                    labels,
                    chunk_size,
                    backend,
                    compact_rows,
                    fuse_dw_cast,
                    dx_finalize,
                    dw_stream,
                    hidden_count,
                    gemm_tuning,
                    logprob_dz,
                    logprob_lbl,
                    temperature,
                    check_temperature,
                )
        # every switch is resolved (and validated) once here, before any launch: the backward plans its chunks on the
        # forward's valid-row index under the same switches, the forward and the backward bind with the same GEMM knobs
        compact = _resolve_compact(compact_rows)
        fuse = _resolve_fuse(fuse_dw_cast)
        fused_dx = _resolve_dx_finalize(dx_finalize)
        stream_mode = dw_stream_mode(dw_stream)
        hidden = _resolve_hidden_count(hidden_count)
        tuning = _resolve_gemm_tuning(gemm_tuning)
        dz_arm = _resolve_logprob_dz(logprob_dz)
        lbl_arm = _resolve_logprob_lbl(logprob_lbl)
        ckpt = _checkpoint_pass() if _TOP_SAVED_HOOKS(False) is not None else None
        if ckpt is not None:
            tau_in = temperature if isinstance(temperature, torch.Tensor) else None
            inputs = (X, W, labels, tau_in)
            key = _logprob_call_key(
                inputs,
                (
                    chunk_size,
                    backend,
                    compact,
                    hidden,
                    _tuning_key(tuning),
                    lbl_arm,
                    logprob_dz,
                    None if tau_in is not None else temperature,
                    bool(check_temperature),
                ),
            )
            kept = None if ckpt.first else _logprob_checkpoint_take(ckpt, key, inputs)
        ctx.set_materialize_grads(False)
        if ckpt is not None and kept is not None:
            # the recomputation of a checkpointed call: its row results, no logits GEMM (see _LogprobKept)
            logp, saved, temperature_scalar = kept.logp.clone(), kept.saved, kept.extra
            # no saved problem under the checkpoint's saved-tensor hooks (see _saved_problem)
            problem = None
        else:
            # the temperature is validated once, before any launch (both arms' explicit switches are known here)
            tau = _resolve_temperature(
                temperature,
                int(X.shape[0]) if X.ndim else 0,
                X.device,
                check=bool(check_temperature),
            )
            _check_temperature_arms(tau is not None, logprob_lbl, logprob_dz)
            result = forward_logprob_temperature(
                X,
                W,
                labels,
                chunk_size=chunk_size,
                backend=backend,
                compact_rows=compact,
                hidden_count=hidden,
                gemm_tuning=tuning,
                logprob_lbl=lbl_arm,
                temperature=tau,
            )
            logp = result.logp
            saved = (
                result.lse,
                result.row_index,
                result.temperature_rows,
                tau.value if tau is not None and tau.kind == "tensor" else None,
            )
            temperature_scalar = (
                tau.value if tau is not None and tau.kind == "scalar" else None
            )
            problem = _saved_problem(result)
            if ckpt is not None and ckpt.first:
                _logprob_checkpoint_keep(
                    ctx, ckpt, key, inputs, logp, saved, temperature_scalar
                )
        # the valid-row index rides along: the backward plans its chunks without counting the rows again; so does the
        # compact temperature the forward's kernels read (no second gather) and the resolved temperature itself -- the
        # caller's per-token [T] tensor (saved, not copied) or the scalar -- which the zero-gradient row skip gathers
        # in its own row order
        ctx.save_for_backward(X, W, labels, *saved)
        ctx.temperature_scalar = temperature_scalar
        ctx.problem = problem
        ctx.chunk_size, ctx.backend, ctx.compact = (
            chunk_size,
            backend,
            compact,
        )
        ctx.fuse_dw_cast = fuse
        ctx.dx_finalize = fused_dx
        ctx.dw_stream = stream_mode
        ctx.hidden_count = hidden
        ctx.gemm_tuning = tuning
        ctx.logprob_dz = dz_arm
        return logp

    @staticmethod
    def backward(ctx, dlogp):
        if dlogp is None:
            return (None,) * 15
        X, W, labels, lse, row_index, tau_rows, tau_tokens = ctx.saved_tensors
        # the forward's resolved temperature (None: the forward ran without one)
        tau = (
            _Temperature("tensor", tau_tokens)
            if tau_tokens is not None
            else _Temperature("scalar", ctx.temperature_scalar)
            if ctx.temperature_scalar is not None
            else None
        )
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        dx, dw = backward_logprob_temperature(
            X,
            W,
            labels,
            lse,
            dlogp,
            chunk_size=ctx.chunk_size,
            need_dx=need_dx,
            need_dw=need_dw,
            backend=ctx.backend,
            compact_rows=ctx.compact,
            fuse_dw_cast=ctx.fuse_dw_cast,
            dx_finalize=ctx.dx_finalize,
            dw_stream=ctx.dw_stream,
            hidden_count=ctx.hidden_count,
            gemm_tuning=ctx.gemm_tuning,
            logprob_dz=ctx.logprob_dz,
            row_index=row_index,
            temperature=tau if tau_rows is not None else None,
            temperature_rows=tau_rows,
            _problem=ctx.problem,
        )
        return (dx, dw) + (None,) * 13


class CakeLMHeadNondetLogprobKeepFunction(torch.autograd.Function):
    """Entry (b) with the kept logits (:func:`cake_lm_head_nondet_train_logprob_keep`): the forward saves the row statistic
    and the kept logits; the first backward forms ``dz`` from them, a later one (a retained graph) recomputes.  A
    forward that keeps one contiguous buffer (:func:`kept_rows_of`) saves it alone; its first backward runs dX in the
    forward's chunks and then the weight gradient in one GEMM over every kept row (``one_pass`` of
    :func:`set_kept_logits_form`, at most :func:`kept_one_pass_rows` rows, :func:`backward_logprob_keep_ordered`), the
    whole buffer as one chunk (``dx_one_pass``) or, above that row limit, the forward's chunks."""

    @staticmethod
    def forward(
        ctx,
        X,
        W,
        labels,
        chunk_size,
        compact_rows,
        fuse_dw_cast,
        dx_finalize,
        dw_stream,
        hidden_count,
        gemm_tuning,
        logprob_dz,
    ):
        if _TOP_SAVED_HOOKS(False) is not None and _selective_checkpoint_modes():
            with _modes_off():  # under a selective checkpoint's modes: popped (see _selective_checkpoint_modes)
                return CakeLMHeadNondetLogprobKeepFunction.forward(
                    ctx,
                    X,
                    W,
                    labels,
                    chunk_size,
                    compact_rows,
                    fuse_dw_cast,
                    dx_finalize,
                    dw_stream,
                    hidden_count,
                    gemm_tuning,
                    logprob_dz,
                )
        compact = _resolve_compact(compact_rows)
        hidden = (
            bool(_KEPT_LOGITS_FORM["hidden_count"])
            if hidden_count is None
            else _resolve_hidden_count(hidden_count)
        )
        tuning = _resolve_gemm_tuning(gemm_tuning)
        result = forward_logprob_keep(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            compact_rows=compact,
            hidden_count=hidden,
            gemm_tuning=tuning,
        )
        ctx.set_materialize_grads(False)
        # the kept logits are saved tensors too, so a non-reentrant checkpoint drops them with the rest; ``row_grad``
        # overwrites them with ``dz`` through a raw pointer (no version-counter bump), so the first backward spends
        # them (``ctx.kept`` True -> False) and a later one through a retained graph recomputes (the same ``dz``)
        kept = tuple(result.logits or ())
        rows_buffer = kept_rows_of(kept)
        # the contiguous kept buffer is saved alone (its chunk views are rebuilt from the row counts if needed)
        ctx.kept_chunks = (
            None if rows_buffer is None else tuple(int(v.shape[0]) for v in kept)
        )
        ctx.save_for_backward(
            X,
            W,
            labels,
            result.lse,
            result.row_index,
            *(kept if rows_buffer is None else (rows_buffer,)),
        )
        ctx.kept = result.logits is not None
        ctx.chunk_size, ctx.compact = (
            chunk_size,
            compact,
        )
        ctx.fuse_dw_cast = _resolve_fuse(fuse_dw_cast)
        ctx.dx_finalize = _resolve_dx_finalize(dx_finalize)
        ctx.dw_stream = dw_stream_mode(dw_stream)
        ctx.hidden_count = hidden
        ctx.gemm_tuning = tuning
        ctx.logprob_dz = _resolve_logprob_dz(logprob_dz)
        return result.logp

    @staticmethod
    def backward(ctx, dlogp):
        if dlogp is None:
            return (None,) * 11
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        X, W, labels, lse, row_index, *kept = ctx.saved_tensors
        common = dict(
            chunk_size=ctx.chunk_size,
            need_dx=need_dx,
            need_dw=need_dw,
            compact_rows=ctx.compact,
            fuse_dw_cast=ctx.fuse_dw_cast,
            dx_finalize=ctx.dx_finalize,
            dw_stream=ctx.dw_stream,
            gemm_tuning=ctx.gemm_tuning,
            row_index=row_index,
        )
        if ctx.kept and (need_dx or need_dw):
            ctx.kept = False  # spent: row_grad overwrites them with dz
            if ctx.kept_chunks is None:
                dx, dw = backward_logprob_keep(
                    X, W, labels, lse, dlogp, tuple(kept), **common
                )
            else:
                rows_buffer, row, chunk_views = kept[0], 0, []
                for rows_c in ctx.kept_chunks:
                    chunk_views.append(rows_buffer[row : row + rows_c])
                    row += rows_c
                one_pass = (
                    bool(_KEPT_LOGITS_FORM["one_pass"])
                    and int(rows_buffer.shape[0]) <= kept_one_pass_rows()
                )
                if one_pass and _KEPT_LOGITS_FORM["dx_one_pass"]:
                    # one chunk over every kept row: one row_grad, one dX GEMM, the fused dW cast over K = T_v
                    common["chunk_size"] = max(int(rows_buffer.shape[0]), 1)
                    dx, dw = backward_logprob_keep(
                        X, W, labels, lse, dlogp, (rows_buffer,), **common
                    )
                elif one_pass:
                    # dX in the forward's chunks, the fused dW cast over K = T_v (no FP32 dW accumulator)
                    dx, dw = backward_logprob_keep_ordered(
                        X,
                        W,
                        labels,
                        lse,
                        dlogp,
                        rows_buffer,
                        tuple(chunk_views),
                        dx_side=bool(_KEPT_LOGITS_FORM["dx_side"]),
                        **common,
                    )
                else:
                    dx, dw = backward_logprob_keep(
                        X, W, labels, lse, dlogp, tuple(chunk_views), **common
                    )
        else:
            dx, dw = backward_logprob(
                X,
                W,
                labels,
                lse,
                dlogp,
                hidden_count=ctx.hidden_count,
                logprob_dz=ctx.logprob_dz,
                **common,
            )
        return (dx, dw) + (None,) * 9


def cake_lm_head_nondet_train_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div=None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    return_logp: bool = False,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    dw_stream: Optional[Union[bool, str]] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
):
    """Differentiable chunked LM-head + loss (see the module docstring); returns the FP32 scalar
    ``loss`` and, with ``return_logp``, the detached FP32 ``logp [T]``.

    ``grad_weight_dtype`` must equal ``W.dtype`` here: PyTorch's autograd engine
    casts every gradient to its leaf's dtype, so an FP32 ``dW`` cannot leave
    this entry through ``W.grad`` for a BF16 ``W``.  Use ``forward_loss(...,
    need_dw=True)`` + ``backward_loss(dx_acc, dw_acc, g, grad_weight_dtype=
    torch.float32, row_index=result.row_index, num_rows=result.num_rows)`` for
    an FP32 weight gradient.

    The switches, each a keyword whose ``None`` (the default) selects the
    documented default -- this module reads no environment variable:

    * ``compact_rows`` (default ``True``, :func:`compact_rows_default`): chunk
      over the valid rows only; the ``dX`` rows of ignored tokens are exact
      zeros either way.
    * ``fuse_dw_cast`` (default ``True``, :func:`fuse_dw_cast_default`): the last
      chunk's weight-gradient GEMM runs in the backward with the upstream scale
      and the cast fused into its epilogue (bitwise the same ``dW``, one pass
      over ``dW`` fewer; a one-chunk call keeps no FP32 ``dW`` accumulator).
    * ``dx_finalize`` (default ``True``, :func:`dx_finalize_default`): a
      compacted call's ``[T, H]`` ``dX`` is written in one pass (bitwise the
      same ``dX`` from the same accumulator).
    * ``dw_stream`` (default ``None`` = ``"auto"``, :func:`dw_stream_mode`):
      each chunk's weight-gradient accumulate GEMM on the per-device side
      stream -- ``"auto"`` for calls of :data:`DW_STREAM_MIN_CHUNKS` chunks or
      more, ``True`` every multi-chunk call, ``False`` never (bitwise the same
      outputs).
    * ``hidden_count`` (default ``True``, :func:`hidden_count_default`): a
      compacted call counts its valid rows on the device and queues chunk 0's
      gather and logits GEMM before the count reaches the host (bitwise the
      same outputs).
    * ``gemm_tuning`` (default ``None`` = ``{}``, :func:`gemm_tuning_default`):
      explicit per-GEMM knobs, e.g. ``{"dx": {"tile_n": 256}}``, winning over
      the per-chunk rules for the knobs they name.

    Policy rows whose ``loss_weights`` entry is exactly zero (a compacted call)
    are left out of the gradient path, like ignored rows: exact zeros in
    ``loss``, ``dX`` and ``dW``, and with ``return_logp`` their ``logp`` from a
    logp-only pass (see :func:`forward_loss`; a non-finite value in such a row
    does not propagate into the gradients).

    A cross-entropy call of a row count in :func:`one_pass_rows` (default one
    row; ``H = ONE_PASS_HIDDEN``, one chunk, no explicit ``gemm_tuning``) runs
    the one-pass programs (see :data:`ONE_PASS_HIDDEN`): ``W`` is read once in
    the forward and ``dW`` written once in the backward; the other switches do
    not apply to it.

    Gradient placement (:func:`set_gradient_placement`, default ``"auto"``): a
    call that no backward can follow (grad mode off, also the first pass of a
    reentrant ``torch.utils.checkpoint``, or no input requiring a gradient)
    computes ``loss`` (and ``logp``) only, without a gradient GEMM; inside a
    non-reentrant ``torch.utils.checkpoint`` the checkpointed forward computes
    the loss only and the recomputation in the backward forms the gradient
    accumulators, so a checkpointed step runs the gradient GEMMs once (CPU
    offload or other saved-tensor hooks innermost and torch's deterministic
    algorithms keep the fused forward).  Under selective activation
    checkpointing the call runs with every active torch dispatch mode popped
    (``_selective_checkpoint_modes``) and behaves the same way.  The outputs
    are those of ``"forward"`` (gradients formed in every forward): the same
    ``loss``, ``logp``, ``dW`` and ``dX`` bits (the K-sliced dX reduction sums
    its slices in slice order either way).
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if grad_weight_dtype not in GRAD_WEIGHT_DTYPES:
        raise ValueError("grad_weight_dtype must be torch.bfloat16 or torch.float32")
    if grad_weight_dtype != W.dtype:
        raise ValueError(
            f"grad_weight_dtype={grad_weight_dtype} differs from W.dtype={W.dtype}: the autograd engine casts "
            "every gradient to its leaf's dtype, so this entry cannot return it; use forward_loss(..., need_dw=True) + "
            "backward_loss(dx_acc, dw_acc, g, grad_weight_dtype=torch.float32, row_index=..., num_rows=...) for an FP32 dW"
        )
    # a one-pass call (one_pass_rows) runs as it is: no small-call staging
    one_pass = _one_pass_call(
        X,
        W,
        labels,
        objective=objective,
        infer_logp=infer_logp,
        loss_weights=loss_weights,
        chunk_size=chunk_size,
        backend=backend,
        compact_rows=compact_rows,
        gemm_tuning=gemm_tuning,
    )
    rows = (
        0
        if one_pass
        else _small_call_rows(
            X,
            labels,
            backend,
            chunk_size,
            compact_rows,
            objective=objective,
            per_row=(infer_logp, loss_weights),
        )
    )
    if rows:
        # a small call (UNCOMPACTED_MAX_ROWS_DEFAULT): its staged rows, uncompacted; the per-row output cut back to T rows
        T = int(X.shape[0])
        Xs, lab, inf, wts, _ = _SmallCallStage.apply(
            X, labels, rows, infer_logp, loss_weights, None
        )
        out = cake_lm_head_nondet_train_loss(
            Xs,
            W,
            lab,
            objective=objective,
            loss_div=loss_div,
            infer_logp=inf,
            loss_weights=wts,
            chunk_size=chunk_size,
            return_logp=return_logp,
            grad_weight_dtype=grad_weight_dtype,
            backend=backend,
            compact_rows=False,
            fuse_dw_cast=fuse_dw_cast,
            dx_finalize=dx_finalize,
            dw_stream=dw_stream,
            hidden_count=hidden_count,
            gemm_tuning=gemm_tuning,
        )
        if not return_logp:
            return out
        return out[0], (out[1] if rows == T else out[1][:T])
    if (
        not (
            torch.is_grad_enabled()
            and (
                X.requires_grad
                or W.requires_grad
                or (infer_logp is not None and infer_logp.requires_grad)
                or (loss_weights is not None and loss_weights.requires_grad)
            )
        )
        and _GRADIENT_PLACEMENT["mode"] == "auto"
    ):
        # no autograd graph follows (grad mode off -- also a reentrant checkpoint's first pass -- or no input that
        # requires a gradient; an infer_logp / loss_weights that does keeps the autograd entry, whose forward forms no
        # gradient when X and W are frozen): the loss (and logp) only, no gradient GEMM
        result = _loss_forward(
            X,
            W,
            labels,
            one_pass=one_pass,
            objective=objective,
            loss_div=loss_div,
            infer_logp=infer_logp,
            loss_weights=loss_weights,
            chunk_size=int(chunk_size),
            need_dx=False,
            need_dw=False,
            grad_weight_dtype=grad_weight_dtype,
            backend=backend,
            compact_rows=compact_rows,
            fuse_dw_cast=fuse_dw_cast,
            dx_finalize=dx_finalize,
            dw_stream=dw_stream,
            hidden_count=hidden_count,
            gemm_tuning=gemm_tuning,
            return_logp=bool(return_logp),
        )
        return (result.loss, result.logp) if return_logp else result.loss
    loss, logp = CakeLMHeadNondetLossFunction.apply(
        X,
        W,
        labels,
        objective,
        loss_div,
        infer_logp,
        loss_weights,
        int(chunk_size),
        bool(return_logp),
        grad_weight_dtype,
        backend,
        compact_rows,
        fuse_dw_cast,
        dx_finalize,
        dw_stream,
        hidden_count,
        gemm_tuning,
    )
    return (loss, logp.detach()) if return_logp else loss


def cake_lm_head_nondet_train_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    dw_stream: Optional[Union[bool, str]] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    logprob_dz: Optional[bool] = None,
    logprob_lbl: Optional[bool] = None,
) -> torch.Tensor:
    """Differentiable FP32 ``logp [T]`` (0 on ignored rows) for arbitrary downstream losses.  The switches
    ``compact_rows`` / ``fuse_dw_cast`` / ``dx_finalize`` / ``dw_stream`` / ``hidden_count`` / ``gemm_tuning`` and
    their defaults (``None`` = the documented default) as in :func:`cake_lm_head_nondet_train_loss`, plus the logits GEMM arms
    of this entry: ``logprob_lbl`` (default ``True``, :func:`logprob_lbl_default`) -- the forward's logits GEMM
    writes the row statistics and the label column's logit only -- and ``logprob_dz`` (default ``True``,
    :func:`logprob_dz_default`) -- the backward's recompute writes ``dz`` from its epilogue, no ``row_grad`` launch
    (both bitwise the same outputs).  Every switch is resolved once in the forward; the backward reuses the
    forward's valid-row index (no count, no host synchronization) and recomputes under the same values."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    T = int(X.shape[0]) if X.ndim else 0
    rows = _small_call_rows(X, labels, backend, chunk_size, compact_rows, logprob=True)
    if rows:
        # a small call (UNCOMPACTED_MAX_ROWS_DEFAULT): its staged rows, uncompacted; cut back to T rows below
        X, labels, _, _, _ = _SmallCallStage.apply(X, labels, rows, None, None, None)
        compact_rows = False
    if (
        _SMALL_T_KEEP["on"]
        and backend == "cake"
        and X.ndim == 2
        and 0 < int(X.shape[0]) <= SMALL_T_KEEP_MAX_ROWS
        and torch.is_grad_enabled()
        and (X.requires_grad or W.requires_grad)
    ):
        # a small call where a backward can follow: the kept logits (set_small_t_keep)
        logp = CakeLMHeadNondetLogprobKeepFunction.apply(
            X,
            W,
            labels,
            int(chunk_size),
            compact_rows,
            fuse_dw_cast,
            dx_finalize,
            dw_stream,
            hidden_count,
            gemm_tuning,
            logprob_dz,
        )
    else:
        logp = CakeLMHeadNondetLogprobFunction.apply(
            X,
            W,
            labels,
            int(chunk_size),
            backend,
            compact_rows,
            fuse_dw_cast,
            dx_finalize,
            dw_stream,
            hidden_count,
            gemm_tuning,
            logprob_dz,
            logprob_lbl,
        )
    return logp[:T] if rows > T else logp


def cake_lm_head_nondet_train_logprob_temperature(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    dw_stream: Optional[Union[bool, str]] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    logprob_dz: Optional[bool] = None,
    logprob_lbl: Optional[bool] = None,
    temperature: Union[None, float, torch.Tensor] = None,
    check_temperature: bool = False,
) -> torch.Tensor:
    """Differentiable FP32 ``logp [T]`` (0 on ignored rows) for arbitrary downstream losses.  The switches
    ``compact_rows`` / ``fuse_dw_cast`` / ``dx_finalize`` / ``dw_stream`` / ``hidden_count`` / ``gemm_tuning`` and
    their defaults (``None`` = the documented default) as in :func:`cake_lm_head_nondet_train_loss`, plus the logits GEMM arms
    of this entry: ``logprob_lbl`` (default ``True``, :func:`logprob_lbl_default`) -- the forward's logits GEMM
    writes the row statistics and the label column's logit only -- and ``logprob_dz`` (default ``True``,
    :func:`logprob_dz_default`) -- the backward's recompute writes ``dz`` from its epilogue, no ``row_grad`` launch
    (both bitwise the same outputs).  Every switch is resolved once in the forward; the backward reuses the
    forward's valid-row index (no count, no host synchronization) and recomputes under the same values.

    ``temperature`` (default ``None``, the same as ``1.0``: no temperature): a positive number or a float32 ``[T]`` tensor on ``X``'s device (one value per
    row, any stride), not differentiable.  ``logp_t = z_y / tau_t - logsumexp_v(z_v / tau_t)`` and ``dlogp_t / dz_v =
    (1[v = y_t] - softmax(z_t / tau_t)_v) / tau_t``: the BF16 projection ``z`` is unchanged and widened to FP32, each
    logit is divided once (IEEE FP32 division, ``RN(z / tau_t)``), the upstream ``dlogp_t`` once per row; the
    statistics, the label logit and the gradient use the same divided values, and ``dX`` / ``dW`` see the temperature
    only through ``dz``.  A scalar is the same as ``torch.full((T,), s)`` bit for bit; ``1.0`` (and any scalar whose
    FP32 value is 1.0) runs the path without a temperature (bitwise the same as omitting it).  Every value must be
    positive and finite -- also on ignored rows (``-100``), whose temperature reaches no output (``logp``, ``dX``,
    ``dW``).  A bad tensor value stops the CUDA stream with a device-side assertion (no host synchronization; it ends
    the CUDA context, like a wrong ``num_valid_rows``); ``check_temperature=True`` reads the check back once and
    raises ``ValueError``.
    Outputs are defined while ``|z / tau_t| < 3e38`` on every row.  With a temperature, ``logprob_lbl`` /
    ``logprob_dz`` must stay on (the temperature forms are those arms)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    T = int(X.shape[0]) if X.ndim else 0
    rows = _small_call_rows(
        X,
        labels,
        backend,
        chunk_size,
        compact_rows,
        per_row=(temperature,),
        logprob=True,
    )
    if rows:
        # a small call (UNCOMPACTED_MAX_ROWS_DEFAULT): its staged rows (temperature 1 on the padded rows), uncompacted;
        # cut back to T rows
        X, labels, _, _, temperature = _SmallCallStage.apply(
            X, labels, rows, None, None, temperature
        )
        logp = _logprob_temperature_call(
            X,
            W,
            labels,
            chunk_size,
            backend,
            False,
            fuse_dw_cast,
            dx_finalize,
            dw_stream,
            hidden_count,
            gemm_tuning,
            logprob_dz,
            logprob_lbl,
            temperature,
            check_temperature,
        )
        return logp[:T] if rows > T else logp
    return _logprob_temperature_call(
        X,
        W,
        labels,
        chunk_size,
        backend,
        compact_rows,
        fuse_dw_cast,
        dx_finalize,
        dw_stream,
        hidden_count,
        gemm_tuning,
        logprob_dz,
        logprob_lbl,
        temperature,
        check_temperature,
    )


def _logprob_temperature_call(
    X,
    W,
    labels,
    chunk_size,
    backend,
    compact_rows,
    fuse_dw_cast,
    dx_finalize,
    dw_stream,
    hidden_count,
    gemm_tuning,
    logprob_dz,
    logprob_lbl,
    temperature,
    check_temperature,
) -> torch.Tensor:
    """The autograd Function call of :func:`cake_lm_head_nondet_train_logprob_temperature` (its switches resolved by
    the Function): the default log-probability Function without a temperature (``None``, or a number whose value is
    ``1.0``), else the temperature Function."""
    if temperature is None or (type(temperature) is float and temperature == 1.0):
        # the default path: no temperature operation (a scalar 1.0 resolves to none)
        return CakeLMHeadNondetLogprobFunction.apply(
            X,
            W,
            labels,
            int(chunk_size),
            backend,
            compact_rows,
            fuse_dw_cast,
            dx_finalize,
            dw_stream,
            hidden_count,
            gemm_tuning,
            logprob_dz,
            logprob_lbl,
        )
    return CakeLMHeadNondetLogprobTemperatureFunction.apply(
        X,
        W,
        labels,
        int(chunk_size),
        backend,
        compact_rows,
        fuse_dw_cast,
        dx_finalize,
        dw_stream,
        hidden_count,
        gemm_tuning,
        logprob_dz,
        logprob_lbl,
        temperature,
        check_temperature,
    )


def cake_lm_head_nondet_train_logprob_keep(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
    dw_stream: Optional[Union[bool, str]] = None,
    hidden_count: Optional[bool] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    logprob_dz: Optional[bool] = None,
    logprob_lbl: Optional[bool] = None,
) -> torch.Tensor:
    """:func:`cake_lm_head_nondet_train_logprob` with the kept logits (opt-in; the opt-in module's
    ``memory_mode="keep_logits"``).  Where a backward can follow -- grad mode on and ``X`` or ``W`` requiring a
    gradient, ``cake`` backend -- the forward keeps every chunk's BF16 logits as saved tensors (``2 * T_v * V``
    bytes until the backward; :func:`forward_logprob_keep`) and the first backward forms ``dz`` from them with
    ``row_grad`` instead of recomputing the logits (three GEMMs per chunk instead of four); a later backward through
    a retained graph recomputes.  Elsewhere -- grad mode off (``torch.no_grad``, ``torch.inference_mode``, the first
    pass of a reentrant ``torch.utils.checkpoint``), no input requiring a gradient, the reference backend -- the call
    is :func:`cake_lm_head_nondet_train_logprob` (the label-column forward).  A forward with a trainable input whose
    backward never runs pays the full-logits forward and holds the kept logits until the graph is freed.  Bitwise
    the same ``logp`` as the default entry; ``dW`` / ``dX`` can round differently from the default entry's (another
    summation order: its vocabulary-major backward of the auto rule's calls, its zero-gradient row skip), within the
    same accuracy budget (a chunk of more than one dX slice sums its slices in slice order, as there).  The kept form (:func:`set_kept_logits_form`; defaults ``tail_fold`` and
    ``one_pass`` on): the forward keeps its chunks' logits as row slices of one contiguous buffer
    (``[T_v, V]`` with the valid rows counted on the host, ``[T, V]`` under the hidden count); it folds a last chunk
    of fewer than ``chunk_size / 2`` rows into the one before it where the default entry folds (:func:`loop_chunks`);
    the first backward
    runs ``row_grad`` and dX in the forward's chunks, then the weight gradient as ONE fused-cast GEMM over ``K = T_v``
    rows (no FP32 ``dW`` accumulator; the FP32 dX accumulator is freed first; up to :func:`kept_one_pass_rows` rows,
    above that the forward's chunks).  On a call of two chunks or more that ``dW`` sums the rows in another grouping
    than the default entry's chunks (each form repeats bit for bit); ``one_pass=False`` (``tail_fold`` on: the default
    entry's chunks) gives the default entry's chunk-loop ``dW`` bits (the kept path never runs the
    vocabulary-major backward)."""
    if (
        backend == "cake"
        and torch.is_grad_enabled()
        and (X.requires_grad or W.requires_grad)
    ):
        return CakeLMHeadNondetLogprobKeepFunction.apply(
            X,
            W,
            labels,
            int(chunk_size),
            compact_rows,
            fuse_dw_cast,
            dx_finalize,
            dw_stream,
            hidden_count,
            gemm_tuning,
            logprob_dz,
        )
    return cake_lm_head_nondet_train_logprob(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        backend=backend,
        compact_rows=compact_rows,
        fuse_dw_cast=fuse_dw_cast,
        dx_finalize=dx_finalize,
        dw_stream=dw_stream,
        hidden_count=hidden_count,
        gemm_tuning=gemm_tuning,
        logprob_dz=logprob_dz,
        logprob_lbl=logprob_lbl,
    )
